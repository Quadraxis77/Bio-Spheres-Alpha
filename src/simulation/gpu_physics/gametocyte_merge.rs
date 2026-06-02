//! Gametocyte Merge Detection System
//!
//! GPU compute pass that detects when two Gametocyte cells from different organisms
//! come into contact. On detection, both are marked for death and a merge event is
//! written to `merge_events_buffer` for CPU-side processing.
//!
//! The CPU reads the events asynchronously each frame (via a staging buffer) and:
//!   1. Performs genome crossover on the two parent genomes
//!   2. Adds the new hybrid genome to the simulation
//!   3. Spawns a new single-cell organism at the merge midpoint

/// Maximum gamete merge events recorded per frame.
pub const MAX_GAMETE_MERGE_EVENTS: u32 = 64;

/// u32s per merge event in the GPU buffer.
pub const EVENT_STRIDE: u32 = 8;

/// Total u32s in the events buffer (1 counter + events).
pub const EVENTS_BUFFER_U32S: u32 = 1 + MAX_GAMETE_MERGE_EVENTS * EVENT_STRIDE;

/// A single gamete merge event decoded from the GPU buffer.
#[derive(Debug, Clone, Copy)]
pub struct GameteMergeEvent {
    pub cell_a_idx: u32,
    pub cell_b_idx: u32,
    pub genome_a_id: u32,
    pub genome_b_id: u32,
    pub spawn_x: f32,
    pub spawn_y: f32,
    pub spawn_z: f32,
    /// Combined reserve from both gametes (x1000 fixed-point, capped at 65535000).
    /// Passed as `initial_reserve` when spawning the offspring Embryocyte.
    pub combined_reserve: u32,
}

/// GPU system for gamete contact detection and merge event collection.
pub struct GametocyteMergeSystem {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout, // Physics group (read-only)
    bind_group_layout_1: wgpu::BindGroupLayout, // Cell data + events buffer
    bind_group_layout_2: wgpu::BindGroupLayout, // Spatial grid

    /// GPU-side merge events buffer (1 counter u32 + events).
    /// Cleared to zero at the start of each frame before dispatch.
    pub merge_events_buffer: wgpu::Buffer,

    /// CPU-readable staging buffer for async readback of merge events.
    pub staging_buffer: wgpu::Buffer,
}

impl GametocyteMergeSystem {
    pub fn new(device: &wgpu::Device) -> Self {
        // --- Bind group layouts ---

        // Group 0: Standard physics (positions + cell_count, all read-only)
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gametocyte Physics Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // 0: physics_params
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
                        // 1: positions_in
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
                        // 2: velocities_in
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 3: positions_out
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 4: velocities_out
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 5: cell_count_buffer
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

        // Group 1: Cell data + merge events
        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gametocyte Cell Data Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // 0: cell_types
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 1: death_flags (read_write - marks cells dead)
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
                        // 2: organism_labels
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 3: genome_ids
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 4: mode_indices
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 5: mode_properties_v13
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 6: merge_events (atomic read_write)
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 7: embryocyte_reserves (read-only)
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
        let bind_group_layout_2 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gametocyte Spatial Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // 0: spatial_grid_counts
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // 1: spatial_grid_cells
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
                        // 2: cell_grid_indices
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

        // --- Compute pipeline ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gametocyte Merge Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/gametocyte_merge.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gametocyte Merge Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout_0,
                &bind_group_layout_1,
                &bind_group_layout_2,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gametocyte Merge Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Buffers ---
        let events_size = (EVENTS_BUFFER_U32S * 4) as u64;
        let merge_events_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gamete Merge Events"),
            size: events_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gamete Merge Events Staging"),
            size: events_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_layout_2,
            merge_events_buffer,
            staging_buffer,
        }
    }

    /// Clear the events counter before each frame's dispatch.
    pub fn clear_events(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.merge_events_buffer, 0, bytemuck::cast_slice(&[0u32]));
    }

    /// Create bind group for group 0 (physics params + positions + cell_count).
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
            label: Some("Gametocyte Physics BG"),
            layout: &self.bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: positions_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: velocities_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create bind group for group 1 (cell data + events buffer + reserves).
    pub fn create_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        cell_types: &wgpu::Buffer,
        death_flags: &wgpu::Buffer,
        organism_labels: &wgpu::Buffer,
        genome_ids: &wgpu::Buffer,
        mode_indices: &wgpu::Buffer,
        mode_properties_v13: &wgpu::Buffer,
        embryocyte_reserves: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gametocyte Cell Data BG"),
            layout: &self.bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: organism_labels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mode_properties_v13.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.merge_events_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: embryocyte_reserves.as_entire_binding(),
                },
            ],
        })
    }

    /// Create bind group for group 2 (spatial grid).
    pub fn create_spatial_bind_group(
        &self,
        device: &wgpu::Device,
        spatial_grid_counts: &wgpu::Buffer,
        spatial_grid_cells: &wgpu::Buffer,
        cell_grid_indices: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gametocyte Spatial BG"),
            layout: &self.bind_group_layout_2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_grid_indices.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the merge detection compute shader.
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        physics_bg: &wgpu::BindGroup,
        cell_data_bg: &wgpu::BindGroup,
        spatial_bg: &wgpu::BindGroup,
        cell_capacity: usize,
    ) {
        let workgroups = ((cell_capacity as u32) + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gametocyte Merge Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, physics_bg, &[]);
        pass.set_bind_group(1, cell_data_bg, &[]);
        pass.set_bind_group(2, spatial_bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Copy events buffer to staging buffer for CPU readback (called after dispatch, before submit).
    pub fn schedule_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.merge_events_buffer,
            0,
            &self.staging_buffer,
            0,
            (EVENTS_BUFFER_U32S * 4) as u64,
        );
    }

    /// Parse merge events from a raw byte slice (read from the staging buffer).
    /// Returns decoded events, clamped to what was actually recorded.
    pub fn parse_events(data: &[u8]) -> Vec<GameteMergeEvent> {
        if data.len() < 4 {
            return Vec::new();
        }
        let words: &[u32] = bytemuck::cast_slice(data);
        let event_count = (words[0] as usize).min(MAX_GAMETE_MERGE_EVENTS as usize);
        let mut events = Vec::with_capacity(event_count);
        for i in 0..event_count {
            let base = 1 + i * EVENT_STRIDE as usize;
            if base + 7 >= words.len() {
                break;
            }
            events.push(GameteMergeEvent {
                cell_a_idx: words[base],
                cell_b_idx: words[base + 1],
                genome_a_id: words[base + 2],
                genome_b_id: words[base + 3],
                spawn_x: f32::from_bits(words[base + 4]),
                spawn_y: f32::from_bits(words[base + 5]),
                spawn_z: f32::from_bits(words[base + 6]),
                combined_reserve: words[base + 7],
            });
        }
        events
    }
}
