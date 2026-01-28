//! GPU-based fluid simulation using pair-based swapping
//!
//! 6 directional passes (±X, ±Y, ±Z), each with 2 checkered phases.
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
    pub direction: u32,  // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

    // vec3<f32> grid_origin - needs to match WGSL layout exactly
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub time: f32,  // Time for wave animations

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

    // Additional padding to match WGSL struct size
    pub _pad0: u32,
}

/// Extract params (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ExtractParams {
    pub grid_resolution: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
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
    pub buoyancy_multiplier: f32,  // How strongly to reverse gravity in water (1.0 = full reversal)
    pub _pad0: f32,
    pub _pad1: f32,
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
    time: std::cell::Cell<f32>,  // Time for wave animations (mutable)

    // Continuous spawning control
    continuous_spawn_enabled: std::cell::Cell<bool>,
    
    // Fluid type control (0=Empty, 1=Water, 2=Lava, 3=Steam)
    fluid_type: std::cell::Cell<u32>,

    // Compute pipelines
    swap_pipeline: wgpu::ComputePipeline,
    init_sphere_pipeline: wgpu::ComputePipeline,
    spawn_continuous_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,

    // Density extraction
    extract_pipeline: wgpu::ComputePipeline,
    extract_params_buffer: wgpu::Buffer,
    extract_bind_group_layout: wgpu::BindGroupLayout,

    // Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    // Water bitfield for fast cell-water detection (32x compressed)
    water_bitfield_buffer: wgpu::Buffer,
    water_grid_params_buffer: wgpu::Buffer,
    bitfield_params_buffer: wgpu::Buffer,
    bitfield_pipeline: wgpu::ComputePipeline,
    bitfield_bind_group_layout: wgpu::BindGroupLayout,

    /// Whether simulation is paused
    pub paused: bool,
}

impl GpuFluidSimulator {
    pub fn new(device: &wgpu::Device, world_radius: f32, world_center: Vec3, solid_mask_buffer: wgpu::Buffer) -> Self {
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);

        // Create single state buffer
        let buffer_size = (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64;
        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid State Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            gravity_dir_y: -9.8,  // Default gravity pointing down (-Y)
            gravity_dir_z: 0.0,
            lateral_flow_probability_empty: 1.0,
            lateral_flow_probability_water: 0.8,
            lateral_flow_probability_lava: 0.6,
            lateral_flow_probability_steam: 0.9,  // [Empty, Water, Lava, Steam]
            spawn_fluid_type: 1u32,  // Default to water
            _pad0: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout (params + voxel buffer + solid mask)
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
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Sim Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/fluid/fluid_sim.wgsl").into()),
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

        let init_sphere_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Init Sphere Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fluid_init_sphere"),
            compilation_options: Default::default(),
            cache: None,
        });

        let spawn_continuous_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
        let extract_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            ],
        });

        let extract_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Extract Pipeline Layout"),
            bind_group_layouts: &[&extract_bind_group_layout],
            push_constant_ranges: &[],
        });

        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Extract Density Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &shader,
            entry_point: Some("extract_density"),
            compilation_options: Default::default(),
            cache: None,
        });

        let extract_params = ExtractParams {
            grid_resolution: GRID_RESOLUTION,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let extract_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Extract Params Buffer"),
            contents: bytemuck::cast_slice(&[extract_params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // === Water Bitfield for fast cell-water detection ===
        // 128³ / 32 = 65536 u32 values (256KB instead of 8MB)
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
            buoyancy_multiplier: 1.0,  // Full gravity reversal in water
            _pad0: 0.0,
            _pad1: 0.0,
        };

        let water_grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Grid Params Buffer"),
            contents: bytemuck::cast_slice(&[water_grid_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bitfield bind group layout
        let bitfield_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/fluid/water_bitfield.wgsl").into()),
        });

        let bitfield_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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

        Self {
            state_buffer,
            solid_mask_buffer,
            params_buffer,
            world_radius,
            world_center,
            time: std::cell::Cell::new(0.0),
            continuous_spawn_enabled: std::cell::Cell::new(false),
            swap_pipeline,
            init_sphere_pipeline,
            spawn_continuous_pipeline,
            clear_pipeline,
            extract_pipeline,
            extract_params_buffer,
            extract_bind_group_layout,
            bind_group_layout,
            water_bitfield_buffer,
            water_grid_params_buffer,
            bitfield_params_buffer,
            bitfield_pipeline,
            bitfield_bind_group_layout,
            fluid_type: std::cell::Cell::new(1u32), // Default to water
            paused: false,
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
            ],
        })
    }

    /// Update params buffer
    fn update_params(&self, queue: &wgpu::Queue, direction: u32, time: f32, gravity_magnitude: f32, gravity_dir: [bool; 3], lateral_flow_probabilities: [f32; 4]) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);

        // Calculate signed gravity direction components
        let mut grav_x = 0.0;
        let mut grav_y = 0.0;
        let mut grav_z = 0.0;
        
        if gravity_dir[0] { grav_x = gravity_magnitude; }
        if gravity_dir[1] { grav_y = gravity_magnitude; }
        if gravity_dir[2] { grav_z = gravity_magnitude; }

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
            _pad0: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Initialize with a water sphere
    pub fn init_water_sphere(&self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.update_params(queue, 0, 0.0, 9.8, [false, true, false], [1.0, 0.8, 0.6, 0.9]);
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
    pub fn spawn_continuous(&self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, time: f32) {
        self.update_params(queue, 0, time, 9.8, [false, true, false], [1.0, 0.8, 0.6, 0.9]);
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
    pub fn clear(&self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.update_params(queue, 0, 0.0, 9.8, [false, true, false], [1.0, 0.8, 0.6, 0.9]);
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

    /// Step the simulation - 100% GPU with zero CPU logic
    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue, _encoder: &mut wgpu::CommandEncoder, dt: f32, gravity_magnitude: f32, gravity_dir: [bool; 3], lateral_flow_probabilities: [f32; 4]) {
        if self.paused {
            return;
        }

        // Update time for wave animations
        let current_time = self.time.get() + dt;
        self.time.set(current_time);
        
        let workgroup_count = (GRID_RESOLUTION + 3) / 4;
        let bind_group = self.create_bind_group(device);

        // Update parameters for GPU (required for shader logic)
        self.update_params(queue, 3, current_time, gravity_magnitude, gravity_dir, lateral_flow_probabilities); // Set direction to -Y for gravity

        // Pure GPU dispatch - no CPU direction processing whatsoever
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fluid GPU Encoder"),
        });

        // First: spawn continuous water (only if enabled)
        if self.continuous_spawn_enabled.get() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid Spawn Continuous Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spawn_continuous_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }

        // Then: run fluid simulation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid GPU Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.swap_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Get current state buffer
    pub fn current_state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffer
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
        density_buffer: &wgpu::Buffer,
        fluid_type_buffer: &wgpu::Buffer,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Extract Bind Group"),
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
            ],
        });

        let workgroup_count = (GRID_RESOLUTION + 3) / 4;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fluid Extract Density"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.extract_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
    }

    /// Update the water bitfield from current voxel state
    /// Call this after fluid simulation step, before cell physics
    pub fn update_water_bitfield(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bitfield Generation Bind Group"),
            layout: &self.bitfield_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bitfield_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.water_bitfield_buffer.as_entire_binding(),
                },
            ],
        });

        // 65536 bitfield entries / 256 threads per workgroup = 256 workgroups
        let bitfield_size = TOTAL_VOXELS / 32;
        let workgroup_count = (bitfield_size as u32 + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Generate Water Bitfield"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.bitfield_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
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
            _pad0: 0.0,
            _pad1: 0.0,
        };

        queue.write_buffer(&self.water_grid_params_buffer, 0, bytemuck::cast_slice(&[water_grid_params]));
    }
}
