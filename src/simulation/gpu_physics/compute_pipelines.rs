//! GPU Compute Pipelines for Physics Simulation
//! 
//! Contains the compute pipelines that make up the GPU physics pipeline.
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
//! | 5 | Storage (read_write) | cell_count_buffer |
//! 
//! ### Spatial Grid Bind Group (Group 1)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read_write) | spatial_grid_counts |
//! | 1 | Storage (read_write) | spatial_grid_offsets |
//! | 2 | Storage (read_write) | cell_grid_indices |
//! | 3 | Storage (read_write) | spatial_grid_cells |
//!
//! ### Lifecycle Bind Group (Group 1 for lifecycle shaders)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read_write) | death_flags |
//! | 1 | Storage (read_write) | division_flags |
//! | 2 | Storage (read_write) | free_slot_indices |
//! | 3 | Storage (read_write) | division_slot_assignments |
//! | 4 | Storage (read_write) | lifecycle_counts |
//!
//! ### Cell State Bind Group (Group 2 for division shaders)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read/write) | birth_times |
//! | 1 | Storage (read/write) | split_intervals |
//! | 2 | Storage (read/write) | split_masses |
//! | 3 | Storage (read/write) | split_counts |
//! | 4 | Storage (read/write) | max_splits |
//! | 5 | Storage (read/write) | genome_ids |
//! | 6 | Storage (read/write) | mode_indices |
//! | 7 | Storage (read/write) | cell_ids |
//! | 8 | Storage (read_write) | next_cell_id |
//! | 9 | Storage (read_write) | nutrient_gain_rates |
//!
//! ### Mass Accumulation Bind Group (Group 1 for mass_accum shader)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read) | nutrient_gain_rates |

use super::GpuTripleBufferSystem;

/// Cached bind groups for GPU physics pipeline
/// Pre-created for all 3 buffer indices to avoid per-frame allocation
pub struct CachedBindGroups {
    /// Physics bind groups for each buffer index [0, 1, 2]
    pub physics: [wgpu::BindGroup; 3],
    /// Spatial grid bind group (same for all frames)
    pub spatial_grid: wgpu::BindGroup,
    /// Lifecycle bind group (same for all frames)
    pub lifecycle: wgpu::BindGroup,
    /// Cell state read bind group (same for all frames)
    pub cell_state_read: wgpu::BindGroup,
    /// Cell state write bind groups for each buffer index [0, 1, 2]
    pub cell_state_write: [wgpu::BindGroup; 3],
    /// Mass accumulation bind group (same for all frames)
    pub mass_accum: wgpu::BindGroup,
    /// Rotations bind groups for each buffer index [0, 1, 2]
    pub rotations: [wgpu::BindGroup; 3],
}

/// GPU physics compute pipelines
pub struct GpuPhysicsPipelines {
    pub spatial_grid_clear: wgpu::ComputePipeline,
    pub spatial_grid_assign: wgpu::ComputePipeline,
    pub spatial_grid_insert: wgpu::ComputePipeline,
    pub collision_detection: wgpu::ComputePipeline,
    pub position_update: wgpu::ComputePipeline,
    pub velocity_update: wgpu::ComputePipeline,
    pub mass_accum: wgpu::ComputePipeline,
    
    // Lifecycle pipelines for cell division
    pub lifecycle_death_scan: wgpu::ComputePipeline,
    pub lifecycle_prefix_sum: wgpu::ComputePipeline,
    pub lifecycle_division_scan: wgpu::ComputePipeline,
    pub lifecycle_division_execute: wgpu::ComputePipeline,
    
    // Bind group layouts
    pub physics_layout: wgpu::BindGroupLayout,
    pub spatial_grid_layout: wgpu::BindGroupLayout,
    pub lifecycle_layout: wgpu::BindGroupLayout,
    pub cell_state_read_layout: wgpu::BindGroupLayout,
    pub cell_state_write_layout: wgpu::BindGroupLayout,
    pub mass_accum_layout: wgpu::BindGroupLayout,
    pub rotations_layout: wgpu::BindGroupLayout,
}

impl GpuPhysicsPipelines {
    /// Create all compute pipelines
    pub fn new(device: &wgpu::Device) -> Self {
        // Create bind group layouts
        let physics_layout = Self::create_physics_bind_group_layout(device);
        let spatial_grid_layout = Self::create_spatial_grid_bind_group_layout(device);
        let lifecycle_layout = Self::create_lifecycle_bind_group_layout(device);
        let cell_state_read_layout = Self::create_cell_state_bind_group_layout(device, true);
        let cell_state_write_layout = Self::create_cell_state_bind_group_layout(device, false);
        let mass_accum_layout = Self::create_mass_accum_bind_group_layout(device);
        let rotations_layout = Self::create_rotations_bind_group_layout(device);
        
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
            &[&physics_layout, &rotations_layout],
            "Position Update",
        );
        
        let velocity_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/velocity_update.wgsl"),
            "main",
            &[&physics_layout],
            "Velocity Update",
        );
        
        let mass_accum = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/mass_accum.wgsl"),
            "main",
            &[&physics_layout, &mass_accum_layout],
            "Mass Accumulation",
        );
        
        // Lifecycle pipelines
        let lifecycle_death_scan = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_death_scan.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout],
            "Lifecycle Death Scan",
        );
        
        let lifecycle_prefix_sum = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_prefix_sum.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout],
            "Lifecycle Prefix Sum",
        );
        
        let lifecycle_division_scan = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_division_scan.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout, &cell_state_read_layout],
            "Lifecycle Division Scan",
        );
        
        let lifecycle_division_execute = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_division_execute.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout, &cell_state_write_layout],
            "Lifecycle Division Execute",
        );
        
        Self {
            spatial_grid_clear,
            spatial_grid_assign,
            spatial_grid_insert,
            collision_detection,
            position_update,
            velocity_update,
            mass_accum,
            lifecycle_death_scan,
            lifecycle_prefix_sum,
            lifecycle_division_scan,
            lifecycle_division_execute,
            physics_layout,
            spatial_grid_layout,
            lifecycle_layout,
            cell_state_read_layout,
            cell_state_write_layout,
            mass_accum_layout,
            rotations_layout,
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
            ],
        });
        
        (physics_bind_group, spatial_grid_bind_group)
    }
    
    /// Create lifecycle bind group for division pipeline
    pub fn create_lifecycle_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lifecycle Bind Group"),
            layout: &self.lifecycle_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.division_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.free_slot_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.division_slot_assignments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.lifecycle_counts.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create cell state bind group (read-only version for division scan)
    pub fn create_cell_state_read_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell State Read Bind Group"),
            layout: &self.cell_state_read_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.birth_times.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.split_intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.split_masses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.max_splits.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create cell state bind group (read-write version for division execute)
    pub fn create_cell_state_write_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let output_index = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell State Write Bind Group"),
            layout: &self.cell_state_write_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.birth_times.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.split_intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.split_masses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.next_cell_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.rotations[output_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.genome_mode_data.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create mass accumulation bind group (nutrient gain rates per cell)
    pub fn create_mass_accum_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mass Accumulation Bind Group"),
            layout: &self.mass_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create rotations bind group for position_update shader
    pub fn create_rotations_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let output_index = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rotations Bind Group"),
            layout: &self.rotations_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.rotations[output_index].as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create all cached bind groups for the physics pipeline
    /// Call once at initialization, not per-frame
    pub fn create_cached_bind_groups(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> CachedBindGroups {
        // Create physics bind groups for all 3 buffer indices
        let physics = [
            self.create_physics_bind_group_for_index(device, buffers, 0),
            self.create_physics_bind_group_for_index(device, buffers, 1),
            self.create_physics_bind_group_for_index(device, buffers, 2),
        ];
        
        // Spatial grid bind group (same for all frames)
        let spatial_grid = self.create_spatial_grid_bind_group_internal(device, buffers);
        
        // Lifecycle bind group (same for all frames)
        let lifecycle = self.create_lifecycle_bind_group(device, buffers);
        
        // Cell state read bind group (same for all frames)
        let cell_state_read = self.create_cell_state_read_bind_group(device, buffers);
        
        // Cell state write bind groups for all 3 buffer indices
        let cell_state_write = [
            self.create_cell_state_write_bind_group(device, buffers, 0),
            self.create_cell_state_write_bind_group(device, buffers, 1),
            self.create_cell_state_write_bind_group(device, buffers, 2),
        ];
        
        // Mass accumulation bind group (same for all frames)
        let mass_accum = self.create_mass_accum_bind_group(device, buffers);
        
        // Rotations bind groups for all 3 buffer indices
        let rotations = [
            self.create_rotations_bind_group(device, buffers, 0),
            self.create_rotations_bind_group(device, buffers, 1),
            self.create_rotations_bind_group(device, buffers, 2),
        ];
        
        CachedBindGroups {
            physics,
            spatial_grid,
            lifecycle,
            cell_state_read,
            cell_state_write,
            mass_accum,
            rotations,
        }
    }
    
    /// Create physics bind group for a specific buffer index
    fn create_physics_bind_group_for_index(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Physics Bind Group {}", buffer_index)),
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create spatial grid bind group (internal, for caching)
    fn create_spatial_grid_bind_group_internal(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
            ],
        })
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
                // Cell count buffer (read-write) - GPU-side cell count tracking
                // [0] = total cells, [1] = live cells
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
                // Stiffnesses (per-cell membrane stiffness from genome)
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
        })
    }
    
    /// Create lifecycle bind group layout for division pipeline
    fn create_lifecycle_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lifecycle Bind Group Layout"),
            entries: &[
                // Death flags
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
                // Division flags
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
                // Free slot indices
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
                // Division slot assignments
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
                // Lifecycle counts
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
    
    /// Create cell state bind group layout
    /// read_only: true for division scan (only reads), false for division execute (writes)
    fn create_cell_state_bind_group_layout(device: &wgpu::Device, read_only: bool) -> wgpu::BindGroupLayout {
        if read_only {
            // Read-only version for division scan (5 bindings)
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell State Read Bind Group Layout"),
                entries: &[
                    // Birth times
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Split intervals
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
                    // Split masses
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
                    // Split counts
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Max splits
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
            })
        } else {
            // Read-write version for division execute (10 bindings)
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell State Write Bind Group Layout"),
                entries: &[
                    // Birth times
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
                    // Split intervals
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
                    // Split masses
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
                    // Split counts
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
                    // Max splits
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
                    // Genome IDs
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Mode indices
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Cell IDs
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Next cell ID
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Nutrient gain rates
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
                    // Stiffnesses (membrane stiffness per cell)
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Rotations input (read-only, from current buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Rotations output (read-write, to next buffer)
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
                    // Genome mode data (child orientations)
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        }
    }
    
    /// Create mass accumulation bind group layout (nutrient gain rates per cell)
    fn create_mass_accum_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mass Accumulation Bind Group Layout"),
            entries: &[
                // Nutrient gain rates (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    
    /// Create rotations bind group layout for position_update shader
    fn create_rotations_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Rotations Bind Group Layout"),
            entries: &[
                // Rotations input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Rotations output (read-write)
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
        })
    }
}