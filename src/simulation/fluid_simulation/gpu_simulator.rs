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

    pub grid_origin: [f32; 3],
    pub phase: u32,  // 0 or 1 for checkered

    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
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

/// GPU Fluid Simulator
pub struct GpuFluidSimulator {
    // Single voxel state buffer
    state_buffer: wgpu::Buffer,

    // Parameters
    params_buffer: wgpu::Buffer,
    world_radius: f32,
    world_center: Vec3,

    // Compute pipelines
    swap_pipeline: wgpu::ComputePipeline,
    init_sphere_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,

    // Density extraction
    extract_pipeline: wgpu::ComputePipeline,
    extract_params_buffer: wgpu::Buffer,
    extract_bind_group_layout: wgpu::BindGroupLayout,

    // Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    /// Whether simulation is paused
    pub paused: bool,
}

impl GpuFluidSimulator {
    pub fn new(device: &wgpu::Device, world_radius: f32, world_center: Vec3) -> Self {
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
            grid_origin: grid_origin.to_array(),
            phase: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout (simpler: just params + single buffer)
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

        Self {
            state_buffer,
            params_buffer,
            world_radius,
            world_center,
            swap_pipeline,
            init_sphere_pipeline,
            clear_pipeline,
            extract_pipeline,
            extract_params_buffer,
            extract_bind_group_layout,
            bind_group_layout,
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
            ],
        })
    }

    /// Update params buffer
    fn update_params(&self, queue: &wgpu::Queue, direction: u32, phase: u32) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);

        let params = GpuFluidParams {
            grid_resolution: GRID_RESOLUTION,
            world_radius: self.world_radius,
            cell_size,
            direction,
            grid_origin: grid_origin.to_array(),
            phase,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Initialize with a water sphere
    pub fn init_water_sphere(&self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.update_params(queue, 0, 0);
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

    /// Clear all fluid
    pub fn clear(&self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.update_params(queue, 0, 0);
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

    /// Step the simulation - 6 directions × 2 phases = 12 swap passes
    /// Each pass needs separate submission to ensure params are visible
    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue, _encoder: &mut wgpu::CommandEncoder, _dt: f32) {
        if self.paused {
            return;
        }

        let workgroup_count = (GRID_RESOLUTION + 3) / 4;
        let bind_group = self.create_bind_group(device);

        // Direction order: -Y first (gravity), then X/Z for spreading, then +Y last
        // Directions: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
        let directions = [3u32, 0, 1, 4, 5, 2]; // -Y, +X, -X, +Z, -Z, +Y

        for &direction in &directions {
            for phase in 0..2u32 {
                self.update_params(queue, direction, phase);

                // Each pass needs its own encoder+submit to ensure params are visible
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fluid Swap Encoder"),
                });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Fluid Swap"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.swap_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
                }

                queue.submit(std::iter::once(encoder.finish()));
            }
        }
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
}
