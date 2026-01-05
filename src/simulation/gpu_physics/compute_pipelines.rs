//! GPU Compute Pipelines for Physics Simulation
//! 
//! Contains the 6 compute pipelines that make up the GPU physics pipeline.
//! 
//! ## Bind Group Layouts
//! 
//! ### Physics Bind Group (Group 0)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Uniform | physics_params |
//! | 1 | Storage (read) | positions_in |
//! | 2 | Storage (read) | velocities_in |
//! | 3 | Storage (read_write) | positions_out |
//! | 4 | Storage (read_write) | velocities_out |
//! 
//! ### Spatial Grid Bind Group (Group 1)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read_write) | spatial_grid_counts |
//! | 1 | Storage (read_write) | spatial_grid_offsets |
//! | 2 | Storage (read_write) | cell_grid_indices |
//! | 3 | Storage (read_write) | spatial_grid_cells |

use super::GpuTripleBufferSystem;

/// GPU physics compute pipelines
pub struct GpuPhysicsPipelines {
    pub spatial_grid_clear: wgpu::ComputePipeline,
    pub spatial_grid_assign: wgpu::ComputePipeline,
    pub spatial_grid_insert: wgpu::ComputePipeline,
    pub collision_detection: wgpu::ComputePipeline,
    pub position_update: wgpu::ComputePipeline,
    pub velocity_update: wgpu::ComputePipeline,
    
    // Bind group layouts
    pub physics_layout: wgpu::BindGroupLayout,
    pub spatial_grid_layout: wgpu::BindGroupLayout,
}

impl GpuPhysicsPipelines {
    /// Create all compute pipelines
    pub fn new(device: &wgpu::Device) -> Self {
        // Create bind group layouts
        let physics_layout = Self::create_physics_bind_group_layout(device);
        let spatial_grid_layout = Self::create_spatial_grid_bind_group_layout(device);
        
        // Create compute pipelines
        let spatial_grid_clear = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_clear.wgsl"),
            "main",
            &[&spatial_grid_layout],
            "Spatial Grid Clear",
        );
        
        let spatial_grid_assign = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_assign.wgsl"),
            "main",
            &[&physics_layout, &spatial_grid_layout],
            "Spatial Grid Assign",
        );
        
        let spatial_grid_insert = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_insert.wgsl"),
            "main",
            &[&physics_layout, &spatial_grid_layout],
            "Spatial Grid Insert",
        );
        
        let collision_detection = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/collision_detection.wgsl"),
            "main",
            &[&physics_layout, &spatial_grid_layout],
            "Collision Detection",
        );
        
        let position_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/position_update.wgsl"),
            "main",
            &[&physics_layout],
            "Position Update",
        );
        
        let velocity_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/velocity_update.wgsl"),
            "main",
            &[&physics_layout],
            "Velocity Update",
        );
        
        Self {
            spatial_grid_clear,
            spatial_grid_assign,
            spatial_grid_insert,
            collision_detection,
            position_update,
            velocity_update,
            physics_layout,
            spatial_grid_layout,
        }
    }
    
    /// Create bind groups for the current frame
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics Bind Group"),
            layout: &self.physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.velocity[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.position_and_mass[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.velocity[(buffer_index + 1) % 3].as_entire_binding(),
                },
            ],
        });
        
        let spatial_grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spatial Grid Bind Group"),
            layout: &self.spatial_grid_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_grid_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.spatial_grid_cells.as_entire_binding(),
                },
            ],
        });
        
        (physics_bind_group, spatial_grid_bind_group)
    }
    
    /// Create a compute pipeline from WGSL source
    fn create_compute_pipeline(
        device: &wgpu::Device,
        shader_source: &str,
        entry_point: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        label: &str,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts,
            push_constant_ranges: &[],
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", label)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }
    
    /// Create physics bind group layout
    fn create_physics_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physics Bind Group Layout"),
            entries: &[
                // Physics parameters (uniform)
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
                // Position input (read-only)
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
                // Velocity input (read-only)
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
                // Position output (read-write)
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
                // Velocity output (read-write)
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
        })
    }
    
    /// Create spatial grid bind group layout
    fn create_spatial_grid_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Grid Bind Group Layout"),
            entries: &[
                // Grid counts
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid offsets
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
                // Cell grid indices
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
                // Spatial grid cells (sorted cell indices by grid cell)
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
        })
    }
}