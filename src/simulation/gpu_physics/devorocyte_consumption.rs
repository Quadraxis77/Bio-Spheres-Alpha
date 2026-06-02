//! Devorocyte Nutrient Consumption System
//!
//! Handles Devorocytes stealing nutrients from and killing foreign cells they touch.
//! Runs as a compute shader before the main physics step.

/// Devorocyte consumption system for GPU
pub struct DevorocyteConsumptionSystem {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout, // Physics group
    bind_group_layout_1: wgpu::BindGroupLayout, // Cell data group
    bind_group_layout_2: wgpu::BindGroupLayout, // Spatial grid group
}

impl DevorocyteConsumptionSystem {
    pub fn new(device: &wgpu::Device) -> Self {
        // Group 0: Standard physics bind group (subset - positions read-only, cell_count read)
        let bind_group_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Devorocyte Physics Layout"),
            entries: &[
                // 0: physics_params uniform
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
                // 1: positions_in (read)
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
                // 2: velocities_in (read - unused but keeps layout consistent with physics group)
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
                // 3: positions_out (read_only - Devorocyte only reads positions, never writes them)
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
                // 4: velocities_out (read_only - Devorocyte only reads positions, never writes them)
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
                // 5: cell_count_buffer (read)
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

        // Group 1: Cell data
        let bind_group_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Devorocyte Cell Data Layout"),
            entries: &[
                // 0: cell_types (read)
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
                // 1: nutrients_buffer (read_write atomic)
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
                // 2: split_nutrient_thresholds (read)
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
                // 3: death_flags (read_write - we write 1 to kill victims)
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
                // 4: organism_labels (read)
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
                // 5: genome_ids (read)
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
                // 6: mode_indices (read)
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
                // 7: mode_properties_v11 (read) - [consume_range, consume_rate, 0, 0] per mode
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
            ],
        });

        // Group 2: Spatial grid
        let bind_group_layout_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Devorocyte Spatial Grid Layout"),
            entries: &[
                // 0: spatial_grid_counts (read)
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
                // 1: spatial_grid_cells (read)
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
                // 2: cell_grid_indices (read)
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Devorocyte Consumption Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/devorocyte_consume.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Devorocyte Consumption Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1, &bind_group_layout_2],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Devorocyte Consumption Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_layout_2,
        }
    }

    /// Create physics bind group (group 0) - uses the standard 6-binding physics layout.
    pub fn create_physics_bind_group(
        &self,
        device: &wgpu::Device,
        physics_params: &wgpu::Buffer,
        positions_in: &wgpu::Buffer,
        velocities_in: &wgpu::Buffer,
        positions_out: &wgpu::Buffer,
        velocities_out: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Devorocyte Physics Bind Group"),
            layout: &self.bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: physics_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: positions_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: velocities_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_count_buffer.as_entire_binding() },
            ],
        })
    }

    /// Create cell data bind group (group 1).
    #[allow(clippy::too_many_arguments)]
    pub fn create_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        cell_types: &wgpu::Buffer,
        nutrients_buffer: &wgpu::Buffer,
        split_nutrient_thresholds: &wgpu::Buffer,
        death_flags: &wgpu::Buffer,
        organism_labels: &wgpu::Buffer,
        genome_ids: &wgpu::Buffer,
        mode_indices: &wgpu::Buffer,
        mode_properties_v11: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Devorocyte Cell Data Bind Group"),
            layout: &self.bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cell_types.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nutrients_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: split_nutrient_thresholds.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: death_flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: organism_labels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: genome_ids.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: mode_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: mode_properties_v11.as_entire_binding() },
            ],
        })
    }

    /// Create spatial grid bind group (group 2).
    pub fn create_spatial_bind_group(
        &self,
        device: &wgpu::Device,
        spatial_grid_counts: &wgpu::Buffer,
        spatial_grid_cells: &wgpu::Buffer,
        cell_grid_indices: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Devorocyte Spatial Bind Group"),
            layout: &self.bind_group_layout_2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: spatial_grid_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spatial_grid_cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_grid_indices.as_entire_binding() },
            ],
        })
    }

    /// Run the devorocyte consumption compute shader.
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        physics_bind_group: &wgpu::BindGroup,
        cell_data_bind_group: &wgpu::BindGroup,
        spatial_bind_group: &wgpu::BindGroup,
        cell_count: u32,
    ) {
        let workgroup_count = (cell_count + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Devorocyte Consumption Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, physics_bind_group, &[]);
        pass.set_bind_group(1, cell_data_bind_group, &[]);
        pass.set_bind_group(2, spatial_bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
