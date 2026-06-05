//! Light Field System
//!
//! Computes per-voxel light intensity on the GPU using ray marching.
//! The light field considers cave solid voxels and cell occupancy as occluders.
//! Used by:
//!   - Photocyte cells for light-based nutrient gain
//!   - Volumetric fog renderer for god rays and shadow casting

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Grid resolution (must match fluid simulation)
const GRID_RESOLUTION: u32 = 128;
const TOTAL_VOXELS: usize = (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION) as usize;

/// Parameters for light field compute shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LightFieldParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub world_radius: f32,
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    pub max_steps: u32,
    pub step_size: f32,
    pub absorption_solid: f32,
    pub absorption_cell: f32,
    pub ambient_floor: f32,
    pub scattering_coefficient: f32,
    pub time: f32,
    pub sun_color_r: f32,
    pub sun_color_g: f32,
    pub sun_color_b: f32,
    pub _pad0: f32,
}

/// Parameters for cell occupancy grid builder
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OccupancyParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Parameters for photocyte light consumption shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PhotocyteParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub mass_per_second_full_light: f32,
    pub min_light_threshold: f32,
    pub _pad0: f32,
}

/// Parameters for surface shadow sampling (used by cave and cell shaders)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowFieldParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub shadow_strength: f32,
    pub shadow_enabled: u32,
    pub shadow_quality: f32,
    // Caustic parameters
    pub caustic_intensity: f32,
    pub caustic_scale: f32,
    pub caustic_speed: f32,
    pub time: f32,
    // Sun/light color
    pub sun_color_r: f32,
    pub sun_color_g: f32,
    pub sun_color_b: f32,
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    // Moss parameters
    pub moss_parallax_depth: f32,
    pub moss_scale: f32,
    // Moss appearance parameters
    pub moss_noise_type: u32,       // 0=value, 1=worley/cellular, 2=ridged
    pub moss_noise_frequency: f32,  // primary noise frequency (default 18.0)
    pub moss_noise_lacunarity: f32, // frequency multiplier between octaves (default 2.5)
    pub moss_height_sharpness_low: f32, // smoothstep lower bound (default 0.25)
    pub moss_height_sharpness_high: f32, // smoothstep upper bound (default 0.7)
    pub moss_bump_strength: f32,    // normal map intensity (default 5.0)
    pub moss_color_dark_r: f32,
    pub moss_color_dark_g: f32,
    pub moss_color_dark_b: f32,
    pub moss_color_bright_r: f32,
    pub moss_color_bright_g: f32,
    pub moss_color_bright_b: f32,
    pub _pad_moss: [f32; 2], // padding for alignment
}

/// GPU Light Field System
///
/// Manages the voxel-based light field computation and photocyte consumption.
/// Pipeline order per frame:
///   1. Clear cell occupancy grid
///   2. Build cell occupancy from cell positions
///   3. Compute light field (ray march with occlusion)
///   4. Photocyte light consumption (photocytes gain mass)
pub struct LightFieldSystem {
    // Buffers
    light_field_buffer: wgpu::Buffer,
    light_color_field_buffer: wgpu::Buffer,
    light_color_accum_buffer: wgpu::Buffer,
    cell_occupancy_buffer: wgpu::Buffer,
    light_field_params_buffer: wgpu::Buffer,
    occupancy_params_buffer: wgpu::Buffer,
    photocyte_params_buffer: wgpu::Buffer,
    shadow_field_params_buffer: wgpu::Buffer,
    /// Dummy water density buffer (all zeros) for when surface nets isn't available
    dummy_water_density: wgpu::Buffer,

    // Compute pipelines
    #[allow(dead_code)] // Pipeline kept alive but replaced by encoder.clear_buffer DMA
    clear_occupancy_pipeline: wgpu::ComputePipeline,
    build_occupancy_pipeline: wgpu::ComputePipeline,
    compute_light_pipeline: wgpu::ComputePipeline,
    photocyte_light_pipeline: wgpu::ComputePipeline,
    resolve_light_color_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    light_field_layout: wgpu::BindGroupLayout,
    occupancy_layout: wgpu::BindGroupLayout,
    photocyte_physics_layout: wgpu::BindGroupLayout,
    photocyte_system_layout: wgpu::BindGroupLayout,
    shadow_bind_group_layout: wgpu::BindGroupLayout,

    // Configurable parameters
    light_dir: [f32; 3],
    max_steps: u32,
    step_size: f32,
    absorption_solid: f32,
    absorption_cell: f32,
    ambient_floor: f32,
    scattering_coefficient: f32,
    mass_per_second_full_light: f32,
    min_light_threshold: f32,
    shadow_strength: f32,
    shadow_enabled: bool,
    shadow_quality: f32,
    // Caustic parameters
    caustic_intensity: f32,
    caustic_scale: f32,
    caustic_speed: f32,
    time: f32,
    // Sun/light color
    sun_color: [f32; 3],

    // Cave-specific shadow bind group layout (includes water bitfield)
    cave_shadow_bind_group_layout: wgpu::BindGroupLayout,

    // Moss parallax depth (passed to cave shader via ShadowFieldParams)
    moss_parallax_depth: f32,
    // Moss texture scale (higher = finer detail)
    moss_scale: f32,
    // Moss appearance parameters
    moss_noise_type: u32,
    moss_noise_frequency: f32,
    moss_noise_lacunarity: f32,
    moss_height_sharpness_low: f32,
    moss_height_sharpness_high: f32,
    moss_bump_strength: f32,
    moss_color_dark: [f32; 3],
    moss_color_bright: [f32; 3],

    // Grid params (cached)
    world_radius: f32,
    cell_size: f32,
    grid_origin: [f32; 3],
}

impl LightFieldSystem {
    pub fn new(device: &wgpu::Device, world_radius: f32) -> Self {
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = [
            -world_diameter / 2.0,
            -world_diameter / 2.0,
            -world_diameter / 2.0,
        ];

        // Default light direction: coming from upper-right-front (normalized)
        let light_dir = Self::normalize_dir([-0.5, 0.7, 0.5]);

        // Create light field buffer (f32 per voxel)
        let light_field_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Field Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_color_field_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Color Field Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_color_accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Luminocyte Light Color Accumulator"),
            size: (TOTAL_VOXELS * 4 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create cell occupancy buffer (u32 per voxel, atomic)
        let cell_occupancy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Occupancy Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffers
        let light_field_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Field Params"),
            size: std::mem::size_of::<LightFieldParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // OccupancyParams are derived purely from world geometry and never change
        // after construction, so pre-populate the buffer at creation time.
        let occupancy_params_init = OccupancyParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size,
            grid_origin_x: grid_origin[0],
            grid_origin_y: grid_origin[1],
            grid_origin_z: grid_origin[2],
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let occupancy_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Occupancy Params"),
            contents: bytemuck::cast_slice(&[occupancy_params_init]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let photocyte_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Photocyte Params"),
            size: std::mem::size_of::<PhotocyteParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Light field compute pipeline ===
        let light_field_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Field Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/light_field_compute.wgsl").into(),
            ),
        });

        // Light field bind group layout (group 0 for compute_light_field)
        let light_field_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Light Field Layout"),
                entries: &[
                    // Binding 0: LightFieldParams (uniform)
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
                    // Binding 1: solid_mask (read-only storage)
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
                    // Binding 2: cell_occupancy (read-only storage)
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
                    // Binding 3: light_field (read-write storage)
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
                    // Binding 4: light_color_field (read-write storage)
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
                    // Binding 5: water_density (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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

        let light_field_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Field Pipeline Layout"),
                bind_group_layouts: &[&light_field_layout],
                push_constant_ranges: &[],
            });

        let compute_light_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Light Field Pipeline"),
                layout: Some(&light_field_pipeline_layout),
                module: &light_field_shader,
                entry_point: Some("compute_light_field"),
                compilation_options: Default::default(),
                cache: None,
            });

        // === Cell occupancy pipelines ===
        // Occupancy bind group layout (group 0 for clear_occupancy and build_cell_occupancy)
        let occupancy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Occupancy Layout"),
            entries: &[
                // Binding 0: OccupancyParams (uniform)
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
                // Binding 1: cell_positions (read-only storage)
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
                // Binding 2: cell_count_buffer (read-only storage)
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
                // Binding 3: cell_occupancy_out (read-write storage, atomic)
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
            ],
        });

        let occupancy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Occupancy Pipeline Layout"),
                bind_group_layouts: &[&occupancy_layout],
                push_constant_ranges: &[],
            });

        let clear_occupancy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Occupancy Pipeline"),
                layout: Some(&occupancy_pipeline_layout),
                module: &light_field_shader,
                entry_point: Some("clear_occupancy"),
                compilation_options: Default::default(),
                cache: None,
            });

        let build_occupancy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Build Cell Occupancy Pipeline"),
                layout: Some(&occupancy_pipeline_layout),
                module: &light_field_shader,
                entry_point: Some("build_cell_occupancy"),
                compilation_options: Default::default(),
                cache: None,
            });

        // === Photocyte light consumption pipeline ===
        let photocyte_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Photocyte Light Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/photocyte_light.wgsl").into(),
            ),
        });

        // Physics bind group layout (group 0 for photocyte)
        let photocyte_physics_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Photocyte Physics Layout"),
                entries: &[
                    // Binding 0: PhysicsParams (uniform)
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
                    // Binding 1: positions (read-write storage)
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
                    // Binding 2: cell_count_buffer (read-only storage)
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
                ],
            });

        // Photocyte system bind group layout (group 1)
        let photocyte_system_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Photocyte System Layout"),
                entries: &[
                    // Binding 0: PhotocyteParams (uniform)
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
                    // Binding 1: light_field (read-write storage; luminocytes inject local light)
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
                    // Binding 2: cell_types (read-only storage)
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
                    // Binding 3: nutrients_buffer (read-write for atomic)
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
                    // Binding 4: split_nutrient_thresholds (read-only storage)
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
                    // Binding 5: death_flags (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 6: mode_indices (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 7: mode_properties_v7 (myocyte signal channel/threshold reused by luminocyte)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 8: mode_emissive (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 9: signal_flags (atomic packed signal values)
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 10: mode_colors (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 11: light_color_accum (atomic read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 12: light_color_field (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
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

        let photocyte_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Photocyte Light Pipeline Layout"),
                bind_group_layouts: &[&photocyte_physics_layout, &photocyte_system_layout],
                push_constant_ranges: &[],
            });

        let photocyte_light_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Photocyte Light Pipeline"),
                layout: Some(&photocyte_pipeline_layout),
                module: &photocyte_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let resolve_light_color_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Resolve Luminocyte Light Color Pipeline"),
                layout: Some(&photocyte_pipeline_layout),
                module: &photocyte_shader,
                entry_point: Some("resolve_luminocyte_light_color"),
                compilation_options: Default::default(),
                cache: None,
            });

        // === Shadow field bind group layout (for cave/cell surface shadows) ===
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Field Bind Group Layout"),
                entries: &[
                    // Binding 0: ShadowFieldParams (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: light_field (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: light_color_field (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shadow_field_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Field Params Buffer"),
            size: std::mem::size_of::<ShadowFieldParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dummy water density buffer (all zeros = no water anywhere)
        // Used when surface nets density buffer isn't available
        let density_size = TOTAL_VOXELS * std::mem::size_of::<f32>();
        let dummy_water_density = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Water Density Buffer"),
            size: density_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // === Cave-specific shadow bind group layout (includes water bitfield) ===
        let cave_shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cave Shadow Field Bind Group Layout"),
                entries: &[
                    // Binding 0: ShadowFieldParams (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: light_field (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: light_color_field (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: water_bitfield (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 4: moss_density (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        Self {
            light_field_buffer,
            light_color_field_buffer,
            light_color_accum_buffer,
            cell_occupancy_buffer,
            light_field_params_buffer,
            occupancy_params_buffer,
            photocyte_params_buffer,
            shadow_field_params_buffer,
            dummy_water_density,
            clear_occupancy_pipeline,
            build_occupancy_pipeline,
            compute_light_pipeline,
            photocyte_light_pipeline,
            resolve_light_color_pipeline,
            light_field_layout,
            occupancy_layout,
            photocyte_physics_layout,
            photocyte_system_layout,
            shadow_bind_group_layout,
            light_dir,
            max_steps: 128,
            step_size: 2.0,
            absorption_solid: 8.0,
            absorption_cell: 10.0,
            ambient_floor: 0.02,
            scattering_coefficient: 0.2,
            mass_per_second_full_light: 0.05,
            min_light_threshold: 0.05,
            shadow_strength: 0.7,
            shadow_enabled: true,
            shadow_quality: 0.8,
            caustic_intensity: 0.5,
            caustic_scale: 8.0,
            caustic_speed: 1.0,
            time: 0.0,
            sun_color: [1.0, 1.0, 1.0],
            cave_shadow_bind_group_layout,
            moss_parallax_depth: 0.08,
            moss_scale: 0.15,
            moss_noise_type: 0,
            moss_noise_frequency: 18.0,
            moss_noise_lacunarity: 2.5,
            moss_height_sharpness_low: 0.25,
            moss_height_sharpness_high: 0.7,
            moss_bump_strength: 5.0,
            moss_color_dark: [0.06, 0.12, 0.04],
            moss_color_bright: [0.20, 0.38, 0.10],
            world_radius,
            cell_size,
            grid_origin,
        }
    }

    fn normalize_dir(d: [f32; 3]) -> [f32; 3] {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < 1e-6 {
            return [0.0, 1.0, 0.0];
        }
        [d[0] / len, d[1] / len, d[2] / len]
    }

    /// Get the light field buffer (for volumetric fog renderer to read)
    pub fn light_field_buffer(&self) -> &wgpu::Buffer {
        &self.light_field_buffer
    }

    pub fn light_color_field_buffer(&self) -> &wgpu::Buffer {
        &self.light_color_field_buffer
    }

    /// Get the cell occupancy buffer
    pub fn cell_occupancy_buffer(&self) -> &wgpu::Buffer {
        &self.cell_occupancy_buffer
    }

    /// Get a reference to the cell occupancy buffer (for external DMA clear)
    pub fn cell_occupancy_buffer_ref(&self) -> &wgpu::Buffer {
        &self.cell_occupancy_buffer
    }

    /// Get a reference to the build occupancy pipeline
    pub fn build_occupancy_pipeline_ref(&self) -> &wgpu::ComputePipeline {
        &self.build_occupancy_pipeline
    }

    /// Get a reference to the compute light field pipeline
    pub fn compute_light_pipeline_ref(&self) -> &wgpu::ComputePipeline {
        &self.compute_light_pipeline
    }

    /// Get a reference to the photocyte light pipeline
    pub fn photocyte_light_pipeline_ref(&self) -> &wgpu::ComputePipeline {
        &self.photocyte_light_pipeline
    }

    pub fn resolve_light_color_pipeline_ref(&self) -> &wgpu::ComputePipeline {
        &self.resolve_light_color_pipeline
    }

    pub fn light_color_accum_buffer_ref(&self) -> &wgpu::Buffer {
        &self.light_color_accum_buffer
    }

    /// Set light direction (will be normalized)
    pub fn set_light_dir(&mut self, dir: [f32; 3]) {
        self.light_dir = Self::normalize_dir(dir);
    }

    /// Get current light direction
    pub fn light_dir(&self) -> [f32; 3] {
        self.light_dir
    }

    /// Set max ray march steps
    pub fn set_max_steps(&mut self, steps: u32) {
        self.max_steps = steps;
    }

    /// Set absorption for solid voxels
    pub fn set_absorption_solid(&mut self, absorption: f32) {
        self.absorption_solid = absorption;
    }

    /// Set absorption for cell-occupied voxels
    pub fn set_absorption_cell(&mut self, absorption: f32) {
        self.absorption_cell = absorption;
    }

    /// Set ambient light floor
    pub fn set_ambient_floor(&mut self, floor: f32) {
        self.ambient_floor = floor;
    }

    /// Set mass gain rate for photocytes at full light
    pub fn set_mass_per_second(&mut self, rate: f32) {
        self.mass_per_second_full_light = rate;
    }

    /// Set minimum light threshold for photocyte gain
    pub fn set_min_light_threshold(&mut self, threshold: f32) {
        self.min_light_threshold = threshold;
    }

    /// Set shadow strength (0.0 = no shadows, 1.0 = full shadows)
    pub fn set_shadow_strength(&mut self, strength: f32) {
        self.shadow_strength = strength;
    }

    /// Set shadow quality (0.0 = low, 1.0 = high - affects sample offset distance)
    pub fn set_shadow_quality(&mut self, quality: f32) {
        self.shadow_quality = quality;
    }

    /// Enable or disable surface shadows
    pub fn set_shadow_enabled(&mut self, enabled: bool) {
        self.shadow_enabled = enabled;
    }

    /// Set caustic intensity (0.0 = off, 1.0 = full)
    pub fn set_caustic_intensity(&mut self, intensity: f32) {
        self.caustic_intensity = intensity;
    }

    /// Set caustic scale (controls pattern size)
    pub fn set_caustic_scale(&mut self, scale: f32) {
        self.caustic_scale = scale;
    }

    /// Set caustic animation speed
    pub fn set_caustic_speed(&mut self, speed: f32) {
        self.caustic_speed = speed;
    }

    /// Set moss parallax depth for cave shader
    pub fn set_moss_parallax_depth(&mut self, depth: f32) {
        self.moss_parallax_depth = depth;
    }

    /// Set moss texture scale for cave shader
    pub fn set_moss_scale(&mut self, scale: f32) {
        self.moss_scale = scale;
    }

    /// Set moss noise type (0=value, 1=worley, 2=ridged)
    pub fn set_moss_noise_type(&mut self, noise_type: u32) {
        self.moss_noise_type = noise_type;
    }

    /// Set moss noise primary frequency
    pub fn set_moss_noise_frequency(&mut self, freq: f32) {
        self.moss_noise_frequency = freq;
    }

    /// Set moss noise lacunarity (frequency multiplier between octaves)
    pub fn set_moss_noise_lacunarity(&mut self, lac: f32) {
        self.moss_noise_lacunarity = lac;
    }

    /// Set moss height sharpness lower bound (smoothstep threshold)
    pub fn set_moss_height_sharpness_low(&mut self, val: f32) {
        self.moss_height_sharpness_low = val;
    }

    /// Set moss height sharpness upper bound (smoothstep threshold)
    pub fn set_moss_height_sharpness_high(&mut self, val: f32) {
        self.moss_height_sharpness_high = val;
    }

    /// Set moss bump/normal map strength
    pub fn set_moss_bump_strength(&mut self, val: f32) {
        self.moss_bump_strength = val;
    }

    /// Set moss dark (base/shadow) color
    pub fn set_moss_color_dark(&mut self, color: [f32; 3]) {
        self.moss_color_dark = color;
    }

    /// Set moss bright (tip/highlight) color
    pub fn set_moss_color_bright(&mut self, color: [f32; 3]) {
        self.moss_color_bright = color;
    }

    /// Set sun/light color
    pub fn set_sun_color(&mut self, color: [f32; 3]) {
        self.sun_color = color;
    }

    /// Set current time for caustic animation
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// Get the shadow bind group layout (for use by cell renderers)
    pub fn shadow_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.shadow_bind_group_layout
    }

    /// Get the cave shadow bind group layout (includes water bitfield, for use by cave renderer)
    pub fn cave_shadow_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.cave_shadow_bind_group_layout
    }

    /// Create the shadow bind group for surface shadow rendering (cells)
    pub fn create_shadow_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Field Bind Group"),
            layout: &self.shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.shadow_field_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.light_color_field_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the cave shadow bind group (includes water density for caustics and moss density)
    pub fn create_cave_shadow_bind_group(
        &self,
        device: &wgpu::Device,
        water_density_buffer: &wgpu::Buffer,
        moss_density_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cave Shadow Field Bind Group"),
            layout: &self.cave_shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.shadow_field_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.light_color_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: water_density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: moss_density_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Update shadow field params on the GPU
    pub fn update_shadow_params(&self, queue: &wgpu::Queue) {
        let params = ShadowFieldParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size: self.cell_size,
            grid_origin_x: self.grid_origin[0],
            grid_origin_y: self.grid_origin[1],
            grid_origin_z: self.grid_origin[2],
            shadow_strength: self.shadow_strength,
            shadow_enabled: if self.shadow_enabled { 1 } else { 0 },
            shadow_quality: self.shadow_quality,
            caustic_intensity: self.caustic_intensity,
            caustic_scale: self.caustic_scale,
            caustic_speed: self.caustic_speed,
            time: self.time,
            sun_color_r: self.sun_color[0],
            sun_color_g: self.sun_color[1],
            sun_color_b: self.sun_color[2],
            light_dir_x: self.light_dir[0],
            light_dir_y: self.light_dir[1],
            light_dir_z: self.light_dir[2],
            moss_parallax_depth: self.moss_parallax_depth,
            moss_scale: self.moss_scale,
            moss_noise_type: self.moss_noise_type,
            moss_noise_frequency: self.moss_noise_frequency,
            moss_noise_lacunarity: self.moss_noise_lacunarity,
            moss_height_sharpness_low: self.moss_height_sharpness_low,
            moss_height_sharpness_high: self.moss_height_sharpness_high,
            moss_bump_strength: self.moss_bump_strength,
            moss_color_dark_r: self.moss_color_dark[0],
            moss_color_dark_g: self.moss_color_dark[1],
            moss_color_dark_b: self.moss_color_dark[2],
            moss_color_bright_r: self.moss_color_bright[0],
            moss_color_bright_g: self.moss_color_bright[1],
            moss_color_bright_b: self.moss_color_bright[2],
            _pad_moss: [0.0; 2],
        };
        queue.write_buffer(
            &self.shadow_field_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }

    /// Get the dummy water density buffer (all zeros, for when surface nets isn't available)
    pub fn dummy_water_density(&self) -> &wgpu::Buffer {
        &self.dummy_water_density
    }

    /// Get grid origin
    pub fn grid_origin(&self) -> [f32; 3] {
        self.grid_origin
    }

    /// Get cell size
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Get world radius
    pub fn world_radius(&self) -> f32 {
        self.world_radius
    }

    /// Get scattering coefficient
    pub fn scattering_coefficient(&self) -> f32 {
        self.scattering_coefficient
    }

    /// Set scattering coefficient
    pub fn set_scattering_coefficient(&mut self, coeff: f32) {
        self.scattering_coefficient = coeff;
    }

    /// Update params and write to GPU
    pub fn update_light_field_params(&self, queue: &wgpu::Queue, time: f32) {
        let params = LightFieldParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size: self.cell_size,
            grid_origin_x: self.grid_origin[0],
            grid_origin_y: self.grid_origin[1],
            grid_origin_z: self.grid_origin[2],
            world_radius: self.world_radius,
            light_dir_x: self.light_dir[0],
            light_dir_y: self.light_dir[1],
            light_dir_z: self.light_dir[2],
            max_steps: self.max_steps,
            step_size: self.step_size,
            absorption_solid: self.absorption_solid,
            absorption_cell: self.absorption_cell,
            ambient_floor: self.ambient_floor,
            scattering_coefficient: self.scattering_coefficient,
            time,
            sun_color_r: self.sun_color[0],
            sun_color_g: self.sun_color[1],
            sun_color_b: self.sun_color[2],
            _pad0: 0.0,
        };
        queue.write_buffer(
            &self.light_field_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    pub fn update_occupancy_params(&self, queue: &wgpu::Queue) {
        let params = OccupancyParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size: self.cell_size,
            grid_origin_x: self.grid_origin[0],
            grid_origin_y: self.grid_origin[1],
            grid_origin_z: self.grid_origin[2],
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(
            &self.occupancy_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    pub fn update_photocyte_params(&self, queue: &wgpu::Queue) {
        let params = PhotocyteParams {
            grid_resolution: GRID_RESOLUTION,
            cell_size: self.cell_size,
            grid_origin_x: self.grid_origin[0],
            grid_origin_y: self.grid_origin[1],
            grid_origin_z: self.grid_origin[2],
            mass_per_second_full_light: self.mass_per_second_full_light,
            min_light_threshold: self.min_light_threshold,
            _pad0: 0.0,
        };
        queue.write_buffer(
            &self.photocyte_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    /// Create the light field compute bind group
    pub fn create_light_field_bind_group(
        &self,
        device: &wgpu::Device,
        solid_mask_buffer: &wgpu::Buffer,
        water_density_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Field Bind Group"),
            layout: &self.light_field_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.light_field_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cell_occupancy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.light_color_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: water_density_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the occupancy builder bind group
    pub fn create_occupancy_bind_group(
        &self,
        device: &wgpu::Device,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Occupancy Bind Group"),
            layout: &self.occupancy_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.occupancy_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cell_occupancy_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the photocyte physics bind group (group 0)
    pub fn create_photocyte_physics_bind_group(
        &self,
        device: &wgpu::Device,
        physics_params_buffer: &wgpu::Buffer,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Photocyte Physics Bind Group"),
            layout: &self.photocyte_physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: physics_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the photocyte system bind group (group 1)
    pub fn create_photocyte_system_bind_group(
        &self,
        device: &wgpu::Device,
        cell_types_buffer: &wgpu::Buffer,
        nutrients_buffer: &wgpu::Buffer,
        split_nutrient_thresholds_buffer: &wgpu::Buffer,
        death_flags_buffer: &wgpu::Buffer,
        mode_indices_buffer: &wgpu::Buffer,
        mode_properties_v7_buffer: &wgpu::Buffer,
        mode_emissive_buffer: &wgpu::Buffer,
        signal_flags_buffer: &wgpu::Buffer,
        mode_colors_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Photocyte System Bind Group"),
            layout: &self.photocyte_system_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.photocyte_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_types_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: nutrients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: split_nutrient_thresholds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: death_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mode_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: mode_properties_v7_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: mode_emissive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: signal_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: mode_colors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.light_color_accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.light_color_field_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Run the full light field pipeline:
    /// 1. Clear occupancy grid
    /// 2. Build occupancy from cell positions
    /// 3. Compute light field
    /// 4. Photocyte consumption
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        solid_mask_buffer: &wgpu::Buffer,
        water_density_buffer: &wgpu::Buffer,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
        physics_params_buffer: &wgpu::Buffer,
        cell_types_buffer: &wgpu::Buffer,
        nutrients_buffer: &wgpu::Buffer,
        split_nutrient_thresholds_buffer: &wgpu::Buffer,
        death_flags_buffer: &wgpu::Buffer,
        mode_indices_buffer: &wgpu::Buffer,
        mode_properties_v7_buffer: &wgpu::Buffer,
        mode_emissive_buffer: &wgpu::Buffer,
        signal_flags_buffer: &wgpu::Buffer,
        mode_colors_buffer: &wgpu::Buffer,
        cell_count: u32,
        time: f32,
    ) {
        // Update all params (occupancy params are static, pre-initialized at construction)
        self.update_light_field_params(queue, time);
        self.update_photocyte_params(queue);

        let total_voxels = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
        let light_workgroups = (total_voxels + 63) / 64;

        // Create bind groups
        let occupancy_bg =
            self.create_occupancy_bind_group(device, positions_buffer, cell_count_buffer);
        let light_field_bg =
            self.create_light_field_bind_group(device, solid_mask_buffer, water_density_buffer);
        let photocyte_physics_bg = self.create_photocyte_physics_bind_group(
            device,
            physics_params_buffer,
            positions_buffer,
            cell_count_buffer,
        );
        let photocyte_system_bg = self.create_photocyte_system_bind_group(
            device,
            cell_types_buffer,
            nutrients_buffer,
            split_nutrient_thresholds_buffer,
            death_flags_buffer,
            mode_indices_buffer,
            mode_properties_v7_buffer,
            mode_emissive_buffer,
            signal_flags_buffer,
            mode_colors_buffer,
        );

        // Step 1: Clear occupancy grid using DMA (faster than compute dispatch)
        encoder.clear_buffer(&self.cell_occupancy_buffer, 0, None);

        // Step 2: Build occupancy from cell positions
        if cell_count > 0 {
            let cell_workgroups = (cell_count + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Cell Occupancy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_occupancy_pipeline);
            pass.set_bind_group(0, &occupancy_bg, &[]);
            pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Step 3: Compute light field
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Light Field"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_light_pipeline);
            pass.set_bind_group(0, &light_field_bg, &[]);
            pass.dispatch_workgroups(light_workgroups, 1, 1);
        }

        // Step 4: Photocyte light consumption and luminocyte local light emission.
        encoder.clear_buffer(&self.light_color_accum_buffer, 0, None);
        encoder.clear_buffer(&self.light_color_field_buffer, 0, None);
        if cell_count > 0 {
            let cell_workgroups = (cell_count + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Photocyte Light Consumption"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.photocyte_light_pipeline);
            pass.set_bind_group(0, &photocyte_physics_bg, &[]);
            pass.set_bind_group(1, &photocyte_system_bg, &[]);
            pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        {
            let color_workgroups = (total_voxels + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Resolve Luminocyte Light Color"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_light_color_pipeline);
            pass.set_bind_group(0, &photocyte_physics_bg, &[]);
            pass.set_bind_group(1, &photocyte_system_bg, &[]);
            pass.dispatch_workgroups(color_workgroups, 1, 1);
        }
    }

    /// Run only the photocyte light consumption step.
    /// Call this inside each physics step so photocytes gain/lose nutrients at the same rate as other cells.
    /// Requires the light field to have been computed already this frame.
    pub fn run_photocyte_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
        physics_params_buffer: &wgpu::Buffer,
        cell_types_buffer: &wgpu::Buffer,
        nutrients_buffer: &wgpu::Buffer,
        split_nutrient_thresholds_buffer: &wgpu::Buffer,
        death_flags_buffer: &wgpu::Buffer,
        mode_indices_buffer: &wgpu::Buffer,
        mode_properties_v7_buffer: &wgpu::Buffer,
        mode_emissive_buffer: &wgpu::Buffer,
        signal_flags_buffer: &wgpu::Buffer,
        mode_colors_buffer: &wgpu::Buffer,
        cell_count: u32,
    ) {
        // Note: Don't early-out on cell_count == 0. The caller passes capacity
        // and the shader reads cell_count_buffer[0] for actual bounds checking.

        self.update_photocyte_params(queue);

        let photocyte_physics_bg = self.create_photocyte_physics_bind_group(
            device,
            physics_params_buffer,
            positions_buffer,
            cell_count_buffer,
        );
        let photocyte_system_bg = self.create_photocyte_system_bind_group(
            device,
            cell_types_buffer,
            nutrients_buffer,
            split_nutrient_thresholds_buffer,
            death_flags_buffer,
            mode_indices_buffer,
            mode_properties_v7_buffer,
            mode_emissive_buffer,
            signal_flags_buffer,
            mode_colors_buffer,
        );

        encoder.clear_buffer(&self.light_color_accum_buffer, 0, None);
        encoder.clear_buffer(&self.light_color_field_buffer, 0, None);

        {
            let cell_workgroups = (cell_count + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Photocyte Light Consumption (physics step)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.photocyte_light_pipeline);
            pass.set_bind_group(0, &photocyte_physics_bg, &[]);
            pass.set_bind_group(1, &photocyte_system_bg, &[]);
            pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        let total_voxels = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
        let color_workgroups = (total_voxels + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Resolve Luminocyte Light Color (physics step)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.resolve_light_color_pipeline);
        pass.set_bind_group(0, &photocyte_physics_bg, &[]);
        pass.set_bind_group(1, &photocyte_system_bg, &[]);
        pass.dispatch_workgroups(color_workgroups, 1, 1);
    }

    /// Run only the light field computation (without photocyte consumption)
    /// Useful when you only need the light field for rendering (volumetric fog)
    pub fn run_light_field_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        solid_mask_buffer: &wgpu::Buffer,
        water_density_buffer: &wgpu::Buffer,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
        cell_count: u32,
        time: f32,
    ) {
        self.update_light_field_params(queue, time);

        let total_voxels = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
        let light_workgroups = (total_voxels + 63) / 64;

        let occupancy_bg =
            self.create_occupancy_bind_group(device, positions_buffer, cell_count_buffer);
        let light_field_bg =
            self.create_light_field_bind_group(device, solid_mask_buffer, water_density_buffer);

        // Clear occupancy grid using DMA (faster than compute dispatch)
        encoder.clear_buffer(&self.cell_occupancy_buffer, 0, None);

        if cell_count > 0 {
            let cell_workgroups = (cell_count + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Cell Occupancy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_occupancy_pipeline);
            pass.set_bind_group(0, &occupancy_bg, &[]);
            pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Compute light field
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Light Field"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_light_pipeline);
            pass.set_bind_group(0, &light_field_bg, &[]);
            pass.dispatch_workgroups(light_workgroups, 1, 1);
        }
    }
}
