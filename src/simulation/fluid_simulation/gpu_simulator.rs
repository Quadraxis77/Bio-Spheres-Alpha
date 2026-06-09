//! GPU-based fluid simulation using pair-based swapping
//!
//! 6 directional passes (X, Y, Z), each with 2 checkered phases.
//! Simple rule: swap neighbors unless it's air-above-water (anti-gravity).
//! Single buffer - no double buffering needed.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use super::{GRID_RESOLUTION, TOTAL_VOXELS};

/// GPU fluid simulation parameters (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuFluidParams {
    pub grid_resolution: u32,
    pub world_radius: f32,
    pub cell_size: f32,
    pub direction: u32, // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

    // vec3<f32> grid_origin - needs to match WGSL layout exactly
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub time: f32, // Time for wave animations

    // Gravity parameters
    pub gravity_magnitude: f32,
    pub gravity_dir_x: f32,
    pub gravity_dir_y: f32,
    pub gravity_dir_z: f32,

    // Per-fluid-type lateral flow probabilities (0.0 to 1.0)
    // Index: 0=Empty (unused), 1=Water, 2=Lava, 3=Steam
    pub lateral_flow_probability_empty: f32,
    pub lateral_flow_probability_water: f32,
    pub lateral_flow_probability_lava: f32,
    pub lateral_flow_probability_steam: f32,

    // Fluid type for spawning (0=Empty, 1=Water, 2=Lava, 3=Steam)
    pub spawn_fluid_type: u32,

    // Sub-step index for alternating checker phase (0..N)
    pub sub_step: u32,

    // Gravity mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial (toward origin)
    pub gravity_mode: u32,
    pub surface_pressure: f32, // Tangential smoothing strength for radial mode (0.0-1.0)
    // Overall sun brightness driving the thermal model's baseline air temperature
    // (0 = dark, 3 = comfortable max, >3 = extreme heat). Mirrors editor_state.sun_intensity.
    pub sun_brightness: f32,
    pub _pad_rg2: u32,
}

/// Extract params (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ExtractParams {
    pub grid_resolution: u32,
    pub gravity_mode: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub cell_size: f32,
    pub gravity_magnitude: f32,
    pub _pad3: f32,
    pub _pad4: f32,
    pub _pad5: f32,
}

/// Bitfield params (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BitfieldParams {
    pub grid_resolution: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Water bitfield grid params for cell physics (must match position_update.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WaterGridParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub buoyancy_multiplier: f32, // How strongly to reverse gravity in water (1.0 = full reversal)
    pub water_viscosity: f32, // Drag applied to cells moving through water (0.0 = off, 1.0 = heavy drag)
    pub _pad1: f32,
}

/// Nutrient population params (must match nutrient_populate.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NutrientPopulateParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub world_radius: f32,
    pub nutrient_density: f32,
    pub time: f32,
    pub delta_time: f32,
    pub epoch_duration: f32,
    pub epoch_spacing: f32,
    pub spawn_end: f32,
    pub despawn_start: f32,
    pub _pad: [f32; 3],
}

/// GPU Fluid Simulator
pub struct GpuFluidSimulator {
    // Single voxel state buffer
    state_buffer: wgpu::Buffer,

    // Solid mask buffer
    solid_mask_buffer: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,
    world_radius: f32,
    world_center: Vec3,
    time: std::cell::Cell<f32>, // Time for wave animations (mutable)

    // Continuous spawning control
    continuous_spawn_enabled: std::cell::Cell<bool>,

    // Fluid type control (0=Empty, 1=Water, 2=Lava, 3=Steam)
    fluid_type: std::cell::Cell<u32>,

    // Gravity mode: 0=X, 1=Y, 2=Z, 3=radial
    gravity_mode: std::cell::Cell<u32>,

    // Last gravity magnitude (updated each step, used by density extraction)
    gravity_magnitude: std::cell::Cell<f32>,

    // Surface pressure: tangential smoothing strength for radial mode (0.0-1.0)
    surface_pressure: std::cell::Cell<f32>,
    /// Overall sun brightness driving the thermal model's baseline air temperature.
    sun_brightness: std::cell::Cell<f32>,

    // Compute pipelines
    swap_pipeline: wgpu::ComputePipeline,
    update_temperature_pipeline: wgpu::ComputePipeline,
    init_sphere_pipeline: wgpu::ComputePipeline,
    spawn_continuous_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,

    // Density extraction
    extract_pipeline: wgpu::ComputePipeline,
    extract_params_buffer: wgpu::Buffer,
    extract_bind_group_layout: wgpu::BindGroupLayout,

    // Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    // Water velocity field for cell drag (128^3 packed u32 per voxel, ~8MB)
    // Each u32 encodes the last movement direction of water at that voxel
    water_velocity_buffer: wgpu::Buffer,

    // Water bitfield for fast cell-water detection (32x compressed)
    water_bitfield_buffer: wgpu::Buffer,
    water_grid_params_buffer: wgpu::Buffer,
    #[allow(dead_code)] // Kept alive - referenced by cached_bitfield_bind_group
    bitfield_params_buffer: wgpu::Buffer,
    bitfield_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)] // Kept alive - referenced by cached_bitfield_bind_group
    bitfield_bind_group_layout: wgpu::BindGroupLayout,

    // Sub-step staging buffer for multi-pass fluid simulation [0,1,2,3]
    sub_step_staging_buffer: wgpu::Buffer,

    // Nutrient system - stores which voxels have nutrients (1 = has nutrient, 0 = empty)
    nutrient_voxels_buffer: wgpu::Buffer,
    nutrient_populate_params_buffer: wgpu::Buffer,
    nutrient_populate_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)] // Kept alive - referenced by cached_nutrient_bind_group
    nutrient_populate_bind_group_layout: wgpu::BindGroupLayout,

    // Cached bind groups (created once, reused every frame)
    cached_sim_bind_group: wgpu::BindGroup,
    cached_bitfield_bind_group: wgpu::BindGroup,
    cached_nutrient_bind_group: wgpu::BindGroup,
    // Cached extract bind group (created lazily on first extract_to_surface_nets call)
    cached_extract_bind_group: std::cell::RefCell<Option<wgpu::BindGroup>>,

    /// Whether simulation is paused
    pub paused: bool,

    // -- Rolling-average temperature readback --
    temp_stats_buffer: wgpu::Buffer,
    temp_stats_staging_buffer: wgpu::Buffer,
    temp_stats_copy_pending: std::cell::Cell<bool>,
    temp_stats_map_receiver:
        std::cell::RefCell<Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    /// Exponential moving average of water temperature, in Celsius.
    avg_water_temp_c: std::cell::Cell<f32>,
    /// Exponential moving average of air (empty-voxel) temperature, in Celsius.
    avg_air_temp_c: std::cell::Cell<f32>,
}

impl GpuFluidSimulator {
    pub fn new(
        device: &wgpu::Device,
        world_radius: f32,
        world_center: Vec3,
        solid_mask_buffer: wgpu::Buffer,
    ) -> Self {
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);

        // Create single state buffer
        let buffer_size = (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64;
        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid State Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = GpuFluidParams {
            grid_resolution: GRID_RESOLUTION,
            world_radius,
            cell_size,
            direction: 0,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            time: 0.0,
            gravity_magnitude: 9.8,
            gravity_dir_x: 0.0,
            gravity_dir_y: -9.8, // Default gravity pointing down (-Y)
            gravity_dir_z: 0.0,
            lateral_flow_probability_empty: 1.0,
            lateral_flow_probability_water: 0.8,
            lateral_flow_probability_lava: 0.6,
            lateral_flow_probability_steam: 0.9, // [Empty, Water, Lava, Steam]
            spawn_fluid_type: 1u32,              // Default to water
            sub_step: 0,
            gravity_mode: 1, // default Y axis
            surface_pressure: 0.5,
            sun_brightness: 3.0,
            _pad_rg2: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Pre-create staging buffer with sub-step values [0,1,2,3] for multi-pass dispatch
        let sub_step_staging_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sub-step Staging Buffer"),
                contents: bytemuck::cast_slice(&[0u32, 1u32, 2u32, 3u32]),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Create water velocity buffer (128^3 * 4 bytes = ~8MB)
        // Each u32 encodes the last movement direction of water at that voxel
        let water_velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Velocity Buffer"),
            size: buffer_size, // Same as state buffer: TOTAL_VOXELS * sizeof(u32)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Temperature accumulator: 4 atomic u32 slots (16 bytes), cleared and
        // accumulated each tick by update_temperature, then copied to a small
        // staging buffer for async CPU readback to drive the rolling averages.
        const TEMP_STATS_SIZE: u64 = 4 * std::mem::size_of::<u32>() as u64;
        let temp_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Temperature Stats Buffer"),
            size: TEMP_STATS_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let temp_stats_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Temperature Stats Staging Buffer"),
            size: TEMP_STATS_SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create bind group layout (params + voxel buffer + solid mask + water velocity)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Sim Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Water velocity field (read-write atomic u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Temperature rolling-average accumulator
                // [water_temp_sum, water_count, air_temp_sum, air_count] (atomic u32 each;
                // temperatures stored as rounded-Celsius + 50 to keep values unsigned).
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Sim Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/fluid/fluid_sim.wgsl").into(),
            ),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Sim Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let swap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Swap Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fluid_swap"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_temperature_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fluid Update Temperature Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("update_temperature"),
                compilation_options: Default::default(),
                cache: None,
            });

        let init_sphere_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fluid Init Sphere Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("fluid_init_sphere"),
                compilation_options: Default::default(),
                cache: None,
            });

        let spawn_continuous_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fluid Spawn Continuous Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("fluid_spawn_continuous"),
                compilation_options: Default::default(),
                cache: None,
            });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fluid_clear"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create extraction bind group layout
        let extract_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fluid Extract Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fluid Extract Pipeline Layout"),
                bind_group_layouts: &[&extract_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create separate shader module for extraction
        let extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Extract Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/fluid/fluid_extract.wgsl").into(),
            ),
        });

        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Extract Density Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &extract_shader,
            entry_point: Some("extract_density"),
            compilation_options: Default::default(),
            cache: None,
        });

        let extract_params = ExtractParams {
            grid_resolution: GRID_RESOLUTION,
            gravity_mode: 1, // default Y axis, updated each frame
            _pad1: 0,
            _pad2: 0,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            cell_size,
            gravity_magnitude: 9.8,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };

        let extract_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Extract Params Buffer"),
            contents: bytemuck::cast_slice(&[extract_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Water Bitfield for fast cell-water detection ===
        // 128^3 / 32 = 65536 u32 values (256KB instead of 8MB)
        let bitfield_size = (TOTAL_VOXELS / 32) as u64 * std::mem::size_of::<u32>() as u64;
        let water_bitfield_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Bitfield Buffer"),
            size: bitfield_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bitfield generation params
        let bitfield_params = BitfieldParams {
            grid_resolution: GRID_RESOLUTION,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let bitfield_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bitfield Params Buffer"),
            contents: bytemuck::cast_slice(&[bitfield_params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Water grid params for cell physics (world-space to grid-space conversion)
        let water_grid_params = WaterGridParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            buoyancy_multiplier: 1.0, // Full gravity reversal in water
            water_viscosity: 0.0,     // Default off, set by UI
            _pad1: 0.0,
        };

        let water_grid_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Water Grid Params Buffer"),
                contents: bytemuck::cast_slice(&[water_grid_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Bitfield bind group layout
        let bitfield_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bitfield Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Bitfield shader and pipeline
        let bitfield_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Bitfield Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/fluid/water_bitfield.wgsl").into(),
            ),
        });

        let bitfield_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bitfield Pipeline Layout"),
                bind_group_layouts: &[&bitfield_bind_group_layout],
                push_constant_ranges: &[],
            });

        let bitfield_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generate Water Bitfield Pipeline"),
            layout: Some(&bitfield_pipeline_layout),
            module: &bitfield_shader,
            entry_point: Some("generate_water_bitfield"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Nutrient system - tracks which voxels have nutrients ===
        // One u32 per voxel (1 = has nutrient, 0 = empty)
        let nutrient_buffer_size = (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64;
        let nutrient_voxels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nutrient Voxels Buffer"),
            size: nutrient_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Nutrient populate params buffer
        let nutrient_populate_params = NutrientPopulateParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            world_radius,
            nutrient_density: 0.3,
            time: 0.0,
            delta_time: 0.016,
            epoch_duration: 10.0,
            epoch_spacing: 7.0,
            spawn_end: 0.4,
            despawn_start: 0.6,
            _pad: [0.0; 3],
        };

        let nutrient_populate_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Nutrient Populate Params Buffer"),
                contents: bytemuck::cast_slice(&[nutrient_populate_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Nutrient populate bind group layout
        let nutrient_populate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nutrient Populate Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Nutrient populate shader and pipeline
        let nutrient_populate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nutrient Populate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/nutrient_populate.wgsl").into(),
            ),
        });

        let nutrient_populate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nutrient Populate Pipeline Layout"),
                bind_group_layouts: &[&nutrient_populate_bind_group_layout],
                push_constant_ranges: &[],
            });

        let nutrient_populate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Nutrient Populate Pipeline"),
                layout: Some(&nutrient_populate_pipeline_layout),
                module: &nutrient_populate_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pre-create cached bind groups (buffers never change)
        let cached_sim_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cached Fluid Sim Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: water_velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: temp_stats_buffer.as_entire_binding(),
                },
            ],
        });

        let cached_bitfield_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cached Bitfield Bind Group"),
            layout: &bitfield_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bitfield_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: water_bitfield_buffer.as_entire_binding(),
                },
            ],
        });

        let cached_nutrient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cached Nutrient Populate Bind Group"),
            layout: &nutrient_populate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nutrient_populate_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nutrient_voxels_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            state_buffer,
            solid_mask_buffer,
            params_buffer,
            world_radius,
            world_center,
            time: std::cell::Cell::new(0.0),
            continuous_spawn_enabled: std::cell::Cell::new(false),
            swap_pipeline,
            update_temperature_pipeline,
            init_sphere_pipeline,
            spawn_continuous_pipeline,
            clear_pipeline,
            extract_pipeline,
            extract_params_buffer,
            extract_bind_group_layout,
            bind_group_layout,
            water_velocity_buffer,
            water_bitfield_buffer,
            water_grid_params_buffer,
            bitfield_params_buffer,
            bitfield_pipeline,
            bitfield_bind_group_layout,
            sub_step_staging_buffer,
            nutrient_voxels_buffer,
            nutrient_populate_params_buffer,
            nutrient_populate_pipeline,
            nutrient_populate_bind_group_layout,
            cached_sim_bind_group,
            cached_bitfield_bind_group,
            cached_nutrient_bind_group,
            cached_extract_bind_group: std::cell::RefCell::new(None),
            fluid_type: std::cell::Cell::new(1u32), // Default to water
            gravity_mode: std::cell::Cell::new(1),  // default Y axis
            gravity_magnitude: std::cell::Cell::new(9.8), // default gravity
            surface_pressure: std::cell::Cell::new(0.5),
            sun_brightness: std::cell::Cell::new(3.0),
            paused: false,
            temp_stats_buffer,
            temp_stats_staging_buffer,
            temp_stats_copy_pending: std::cell::Cell::new(false),
            temp_stats_map_receiver: std::cell::RefCell::new(None),
            avg_water_temp_c: std::cell::Cell::new(0.0),
            avg_air_temp_c: std::cell::Cell::new(0.0),
        }
    }

    /// Create bind group
    fn create_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Sim Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.water_velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.temp_stats_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Update params buffer
    fn update_params(
        &self,
        queue: &wgpu::Queue,
        direction: u32,
        time: f32,
        gravity_magnitude: f32,
        gravity_dir: [bool; 3],
        lateral_flow_probabilities: [f32; 4],
    ) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);

        // Calculate signed gravity direction components
        let mut grav_x = 0.0;
        let mut grav_y = 0.0;
        let mut grav_z = 0.0;

        if gravity_dir[0] {
            grav_x = gravity_magnitude;
        }
        if gravity_dir[1] {
            grav_y = gravity_magnitude;
        }
        if gravity_dir[2] {
            grav_z = gravity_magnitude;
        }

        let params = GpuFluidParams {
            grid_resolution: GRID_RESOLUTION,
            world_radius: self.world_radius,
            cell_size,
            direction,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            time,
            gravity_magnitude,
            gravity_dir_x: grav_x,
            gravity_dir_y: grav_y,
            gravity_dir_z: grav_z,
            lateral_flow_probability_empty: lateral_flow_probabilities[0],
            lateral_flow_probability_water: lateral_flow_probabilities[1],
            lateral_flow_probability_lava: lateral_flow_probabilities[2],
            lateral_flow_probability_steam: lateral_flow_probabilities[3],
            spawn_fluid_type: self.fluid_type.get(),
            sub_step: 0,
            gravity_mode: self.gravity_mode.get(),
            surface_pressure: self.surface_pressure.get(),
            sun_brightness: self.sun_brightness.get(),
            _pad_rg2: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Initialize with a water sphere
    pub fn init_water_sphere(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.update_params(
            queue,
            0,
            0.0,
            9.8,
            [false, true, false],
            [1.0, 0.8, 0.6, 0.9],
        );
        let bind_group = self.create_bind_group(device);
        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fluid Init Sphere"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.init_sphere_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
    }

    /// Continuously spawn water at the top of the world
    pub fn spawn_continuous(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        time: f32,
    ) {
        self.update_params(
            queue,
            0,
            time,
            9.8,
            [false, true, false],
            [1.0, 0.8, 0.6, 0.9],
        );
        let bind_group = self.create_bind_group(device);
        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fluid Spawn Continuous"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.spawn_continuous_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
    }

    /// Clear all fluid
    pub fn clear(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.update_params(
            queue,
            0,
            0.0,
            9.8,
            [false, true, false],
            [1.0, 0.8, 0.6, 0.9],
        );
        let bind_group = self.create_bind_group(device);
        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fluid Clear"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.clear_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
    }

    /// Set continuous spawning enabled/disabled
    pub fn set_continuous_spawn(&self, enabled: bool) {
        self.continuous_spawn_enabled.set(enabled);
        // Reset the epoch clock when water fill starts so the user always gets
        // a fresh nutrient epoch rather than landing mid-epoch (which could mean
        // nutrients despawn within seconds of the water being added).
        if enabled {
            self.time.set(0.0);
        }
    }

    /// Get continuous spawning enabled state
    pub fn is_continuous_spawn_enabled(&self) -> bool {
        self.continuous_spawn_enabled.get()
    }

    /// Set fluid type (0=Empty, 1=Water, 2=Lava, 3=Steam)
    pub fn set_fluid_type(&self, fluid_type: u32) {
        self.fluid_type.set(fluid_type);
    }

    /// Get current fluid type
    pub fn get_fluid_type(&self) -> u32 {
        self.fluid_type.get()
    }

    /// Set gravity mode (0=X, 1=Y, 2=Z, 3=radial)
    pub fn set_gravity_mode(&self, mode: u32) {
        self.gravity_mode.set(mode);
    }

    /// Set surface pressure (tangential smoothing strength for radial mode, 0.0-1.0)
    pub fn set_surface_pressure(&self, pressure: f32) {
        self.surface_pressure.set(pressure);
    }

    /// Set the overall sun brightness (drives the thermal model's baseline air
    /// temperature: 0 = dark, 3 = comfortable max, >3 = extreme heat).
    pub fn set_sun_brightness(&self, brightness: f32) {
        self.sun_brightness.set(brightness);
    }

    /// Get the current simulation time (seconds).
    pub fn time(&self) -> f32 {
        self.time.get()
    }

    /// Set the simulation time (used when restoring from a snapshot).
    pub fn set_time(&self, t: f32) {
        self.time.set(t);
    }

    /// Step the simulation - 100% GPU with zero CPU logic
    /// Uses the caller's command encoder to avoid a separate queue.submit (GPU sync point)
    pub fn step(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        gravity_magnitude: f32,
        gravity_dir: [bool; 3],
        lateral_flow_probabilities: [f32; 4],
    ) {
        if self.paused {
            return;
        }

        // Update time for wave animations.
        // Wrap at 65536s to prevent f32 precision loss in long runs - once the
        // raw time exceeds ~8M seconds, adding dt has no effect on a f32.
        let current_time = (self.time.get() + dt) % 65536.0;
        self.time.set(current_time);
        self.gravity_magnitude.set(gravity_magnitude);

        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        // Update parameters for GPU (required for shader logic) - sub_step starts at 0
        self.update_params(
            queue,
            3,
            current_time,
            gravity_magnitude,
            gravity_dir,
            lateral_flow_probabilities,
        );

        // Clear water velocity field before simulation (DMA zero-fill)
        encoder.clear_buffer(&self.water_velocity_buffer, 0, None);

        // Byte offset of sub_step field in GpuFluidParams (20 fields x 4 bytes each, last field)
        const SUB_STEP_OFFSET: u64 = 76;
        const NUM_FLUID_SUB_STEPS: u32 = 4;

        // Spawn continuous water once per frame (only on first sub-step)
        if self.continuous_spawn_enabled.get() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid Spawn Continuous Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spawn_continuous_pipeline);
            pass.set_bind_group(0, &self.cached_sim_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }

        // Per-tick thermal drift: water/ice ease toward ambient temperature with
        // inertia. Runs once per step (not per sub-step) since thermal change is
        // gradual relative to fluid movement, and as its own pass so it never
        // contends with the swap CAS loops below. The accumulator is cleared
        // immediately before so each tick's pass produces a fresh sum/count
        // snapshot, then copied out for async readback (the rolling average
        // itself is maintained as an EMA on the CPU side in poll_temperature_stats).
        encoder.clear_buffer(&self.temp_stats_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid Update Temperature Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_temperature_pipeline);
            pass.set_bind_group(0, &self.cached_sim_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }
        if !self.temp_stats_copy_pending.get() && self.temp_stats_map_receiver.borrow().is_none() {
            encoder.copy_buffer_to_buffer(
                &self.temp_stats_buffer,
                0,
                &self.temp_stats_staging_buffer,
                0,
                self.temp_stats_buffer.size(),
            );
            self.temp_stats_copy_pending.set(true);
        }

        // Multi-pass fluid simulation with alternating checker phase
        // Each sub-step flips which cells are active, allowing water to cascade
        // multiple cells per frame for dramatically faster lateral flow.
        for sub_step in 0..NUM_FLUID_SUB_STEPS {
            if sub_step > 0 {
                // Copy sub_step value from staging buffer to params uniform
                encoder.copy_buffer_to_buffer(
                    &self.sub_step_staging_buffer,
                    (sub_step as u64) * 4,
                    &self.params_buffer,
                    SUB_STEP_OFFSET,
                    4,
                );
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Fluid GPU Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.swap_pipeline);
                pass.set_bind_group(0, &self.cached_sim_bind_group, &[]);
                pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
            }
        }
    }

    /// Get current state buffer
    pub fn current_state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffer
    }

    /// Get solid mask buffer
    pub fn solid_mask_buffer(&self) -> &wgpu::Buffer {
        &self.solid_mask_buffer
    }

    /// Get grid parameters for surface nets
    pub fn grid_params(&self) -> (f32, Vec3) {
        (self.world_radius, self.world_center)
    }

    /// Extract density and fluid types to surface nets buffers
    pub fn extract_to_surface_nets(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        density_buffer: &wgpu::Buffer,
        fluid_type_buffer: &wgpu::Buffer,
    ) {
        // Update extract params with current gravity mode each frame
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        let extract_params = ExtractParams {
            grid_resolution: GRID_RESOLUTION,
            gravity_mode: self.gravity_mode.get(),
            _pad1: 0,
            _pad2: 0,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            cell_size,
            gravity_magnitude: self.gravity_magnitude.get(),
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        };
        queue.write_buffer(
            &self.extract_params_buffer,
            0,
            bytemuck::cast_slice(&[extract_params]),
        );

        // Lazily create and cache the extract bind group (buffers never change after init)
        let mut cached = self.cached_extract_bind_group.borrow_mut();
        if cached.is_none() {
            *cached = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Cached Fluid Extract Bind Group"),
                layout: &self.extract_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.extract_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.state_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: density_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: fluid_type_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.solid_mask_buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        let bind_group = cached.as_ref().unwrap();

        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fluid Extract Density"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.extract_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
    }

    /// Update the water bitfield from current voxel state
    /// Call this after fluid simulation step, before cell physics
    pub fn update_water_bitfield(
        &self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // 65536 bitfield entries / 256 threads per workgroup = 256 workgroups
        let bitfield_size = TOTAL_VOXELS / 32;
        let workgroup_count = (bitfield_size as u32 + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Generate Water Bitfield"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.bitfield_pipeline);
        pass.set_bind_group(0, &self.cached_bitfield_bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    /// Get the water velocity buffer for cell drag forces
    pub fn water_velocity_buffer(&self) -> &wgpu::Buffer {
        &self.water_velocity_buffer
    }

    /// Get the water bitfield buffer for cell physics
    pub fn water_bitfield_buffer(&self) -> &wgpu::Buffer {
        &self.water_bitfield_buffer
    }

    /// Get the water grid params buffer for cell physics
    pub fn water_grid_params_buffer(&self) -> &wgpu::Buffer {
        &self.water_grid_params_buffer
    }

    /// Update buoyancy multiplier
    pub fn set_buoyancy_multiplier(&self, queue: &wgpu::Queue, multiplier: f32) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);

        let water_grid_params = WaterGridParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            buoyancy_multiplier: multiplier,
            water_viscosity: 0.0, // Will be set separately
            _pad1: 0.0,
        };

        queue.write_buffer(
            &self.water_grid_params_buffer,
            0,
            bytemuck::cast_slice(&[water_grid_params]),
        );
    }

    /// Update water viscosity (drag applied to cells moving through water)
    pub fn set_water_drag_strength(&self, queue: &wgpu::Queue, strength: f32) {
        // water_viscosity is at byte offset 24 in WaterGridParams (field index 6, after 6 f32s)
        let offset = 6 * std::mem::size_of::<f32>() as u64;
        queue.write_buffer(
            &self.water_grid_params_buffer,
            offset,
            bytemuck::cast_slice(&[strength]),
        );
    }

    /// Get the nutrient voxels buffer
    pub fn nutrient_voxels_buffer(&self) -> &wgpu::Buffer {
        &self.nutrient_voxels_buffer
    }

    /// Populate nutrients in water voxels using drifting noise pattern
    /// Called every physics step to keep supply balanced with phagocyte consumption
    pub fn populate_nutrients(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        nutrient_density: f32,
        delta_time: f32,
        epoch_duration: f32,
        epoch_spacing: f32,
        spawn_end: f32,
        despawn_start: f32,
    ) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);

        // Use the fluid sim's own time directly. The fluid sim already wraps at 65536s
        // (well within f32 precision), and that wrap is a fixed constant so the epoch
        // counter resets cleanly at the same point every time. A separate wrap period
        // based on epoch_spacing caused misaligned discontinuities: when the fluid sim
        // time crossed a multiple of (epoch_spacing * 8192) the nutrient time would
        // jump, clearing all nutrients for one frame.
        let wrapped_time = self.time.get();

        let params = NutrientPopulateParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            world_radius: self.world_radius,
            nutrient_density,
            time: wrapped_time,
            delta_time,
            epoch_duration,
            epoch_spacing,
            spawn_end,
            despawn_start,
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            &self.nutrient_populate_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Nutrient Populate Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.nutrient_populate_pipeline);
        pass.set_bind_group(0, &self.cached_nutrient_bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
    }

    /// Polls the async readback of the rolling temperature accumulator and,
    /// once mapped, blends the snapshot into the exponential moving averages.
    pub fn poll_temperature_stats(&self, device: &wgpu::Device) {
        if self.temp_stats_copy_pending.get() && self.temp_stats_map_receiver.borrow().is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            self.temp_stats_staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).ok();
                });
            *self.temp_stats_map_receiver.borrow_mut() = Some(rx);
            self.temp_stats_copy_pending.set(false);
        }

        let mut finished = false;
        if let Some(rx) = self.temp_stats_map_receiver.borrow().as_ref() {
            let _ = device.poll(wgpu::PollType::Poll);
            match rx.try_recv() {
                Ok(Ok(())) => {
                    {
                        let view = self
                            .temp_stats_staging_buffer
                            .slice(..)
                            .get_mapped_range();
                        let stats: &[u32] = bytemuck::cast_slice(&view);
                        // Water has real thermal mass (THERMAL_INERTIA_RATE in the sim
                        // is intentionally tiny), so its displayed rolling average is
                        // smoothed heavily too - it should visibly lag and change far
                        // slower than air. Air has no per-voxel inertia at all (it's a
                        // direct ambient derivation), so it can react comparatively
                        // quickly and still read as a believable "rolling average"
                        // rather than flickering with every light change.
                        const WATER_EMA_RATE: f32 = 0.01;
                        const AIR_EMA_RATE: f32 = 0.05;

                        if stats[1] > 0 {
                            let avg_c = (stats[0] as f32 / stats[1] as f32) - 50.0;
                            let prev = self.avg_water_temp_c.get();
                            self.avg_water_temp_c.set(prev + (avg_c - prev) * WATER_EMA_RATE);
                        }
                        if stats[3] > 0 {
                            let avg_c = (stats[2] as f32 / stats[3] as f32) - 50.0;
                            let prev = self.avg_air_temp_c.get();
                            self.avg_air_temp_c.set(prev + (avg_c - prev) * AIR_EMA_RATE);
                        }
                    }
                    self.temp_stats_staging_buffer.unmap();
                    finished = true;
                }
                Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.temp_stats_staging_buffer.unmap();
                    finished = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if finished {
            *self.temp_stats_map_receiver.borrow_mut() = None;
        }
    }

    /// Rolling average water temperature, in Celsius.
    pub fn avg_water_temp_c(&self) -> f32 {
        self.avg_water_temp_c.get()
    }

    /// Rolling average air (empty-voxel) temperature, in Celsius.
    pub fn avg_air_temp_c(&self) -> f32 {
        self.avg_air_temp_c.get()
    }
}

// -- Snapshot support ----------------------------------------------------------

impl GpuFluidSimulator {
    /// Read back the fluid voxel state and nutrient voxels from the GPU.
    ///
    /// Returns `(fluid_voxels, nutrient_voxels)` - each a `Vec<u32>` of length
    /// `TOTAL_VOXELS` (128^3 = 2,097,152 elements, 8 MB each).
    ///
    /// This is a **blocking** operation: it submits a copy command, polls the
    /// device until complete, and maps the staging buffers synchronously.
    pub fn snapshot_voxels(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (Vec<u32>, Vec<u32>) {
        let byte_size = (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64;

        // Create staging buffers for readback.
        let fluid_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid State Snapshot Staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let nutrient_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nutrient Voxels Snapshot Staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy GPU -> staging.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fluid Snapshot Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.state_buffer, 0, &fluid_staging, 0, byte_size);
        encoder.copy_buffer_to_buffer(
            &self.nutrient_voxels_buffer,
            0,
            &nutrient_staging,
            0,
            byte_size,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read fluid state.
        let fluid_voxels = {
            let slice = fluid_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let voxels: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            fluid_staging.unmap();
            voxels
        };

        // Map and read nutrient state.
        let nutrient_voxels = {
            let slice = nutrient_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let voxels: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            nutrient_staging.unmap();
            voxels
        };

        (fluid_voxels, nutrient_voxels)
    }

    /// Restore fluid and nutrient voxel state from snapshot data.
    ///
    /// Writes the provided voxel arrays directly into the GPU buffers via
    /// `queue.write_buffer`.  The arrays must each have exactly `TOTAL_VOXELS`
    /// elements.
    pub fn restore_voxels(
        &self,
        queue: &wgpu::Queue,
        fluid_voxels: &[u32],
        nutrient_voxels: &[u32],
    ) {
        assert_eq!(
            fluid_voxels.len(),
            TOTAL_VOXELS,
            "fluid_voxels length {} != TOTAL_VOXELS {}",
            fluid_voxels.len(),
            TOTAL_VOXELS,
        );
        assert_eq!(
            nutrient_voxels.len(),
            TOTAL_VOXELS,
            "nutrient_voxels length {} != TOTAL_VOXELS {}",
            nutrient_voxels.len(),
            TOTAL_VOXELS,
        );
        queue.write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(fluid_voxels));
        queue.write_buffer(
            &self.nutrient_voxels_buffer,
            0,
            bytemuck::cast_slice(nutrient_voxels),
        );
    }
}
