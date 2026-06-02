//! Moss system for cave wall vegetation
//!
//! Manages a 128^3 voxel buffer of moss density values and two compute pipelines:
//! - Growth/erosion: moss grows on lit cave surfaces near water, erodes under flow
//! - Consumption: phagocytes eat moss and gain nutrients
//!
//! The moss_density buffer is also read by the cave fragment shader for
//! parallax occlusion mapping visualization.

use bytemuck::{Pod, Zeroable};

const GRID_RESOLUTION: u32 = 128;
const TOTAL_VOXELS: usize = (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION) as usize;

/// Uniform parameters for the moss growth compute shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MossGrowthParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub growth_rate: f32,
    pub erosion_rate: f32,
    pub decay_rate: f32,
    pub min_light: f32,
    pub delta_time: f32,
    pub world_radius: f32,
    pub water_radius: f32,
    pub wetness_evaporation: f32,
    pub slice_index: u32, // unused, kept for struct alignment
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Uniform parameters for the moss consumption compute shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MossConsumeParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub nutrient_per_moss: f32,
    pub graze_cooldown: f32,
    pub _pad0: f32,
}

pub struct MossSystem {
    /// 128^3 f32 buffer of moss density values [0.0, 1.0]
    moss_density_buffer: wgpu::Buffer,
    /// 128^3 f32 buffer of wetness values [0.0, 1.0] - moisture memory
    wetness_buffer: wgpu::Buffer,

    // Growth pipeline
    growth_params_buffer: wgpu::Buffer,
    growth_pipeline: wgpu::ComputePipeline,
    growth_bind_group_layout: wgpu::BindGroupLayout,

    // Consumption pipeline
    consume_params_buffer: wgpu::Buffer,
    consume_pipeline: wgpu::ComputePipeline,
    consume_physics_layout: wgpu::BindGroupLayout,
    consume_moss_layout: wgpu::BindGroupLayout,

    // Tunable parameters
    pub growth_rate: f32,
    pub erosion_rate: f32,
    pub decay_rate: f32,
    pub min_light: f32,
    pub nutrient_per_moss: f32,
    pub graze_cooldown: f32,
    pub water_radius: f32,
    pub wetness_evaporation: f32,

    // Grid params (cached)
    grid_resolution: u32,
    cell_size: f32,
    grid_origin: [f32; 3],

    // Frame counter for throttle (run every 4th frame)
    frame_counter: u32,
    /// How many rendered frames to skip between growth dispatches (default 3 = run every 4th frame)
    pub growth_frame_skip: u32,
}

impl MossSystem {
    pub fn new(
        device: &wgpu::Device,
        grid_resolution: u32,
        cell_size: f32,
        grid_origin: [f32; 3],
    ) -> Self {
        // Create moss density buffer (128^3 x 4 bytes = 8MB), initialized to zero
        let moss_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Moss Density Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create wetness buffer (128^3 x 4 bytes = 8MB), initialized to zero
        let wetness_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Moss Wetness Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Growth params buffer
        let growth_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Moss Growth Params"),
            size: std::mem::size_of::<MossGrowthParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Consume params buffer
        let consume_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Moss Consume Params"),
            size: std::mem::size_of::<MossConsumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Growth pipeline ===
        let growth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Moss Growth Bind Group Layout"),
                entries: &[
                    // binding 0: MossParams uniform
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
                    // binding 1: moss_density (read_write)
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
                    // binding 2: solid_mask (read)
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
                    // binding 3: light_field (read)
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
                    // binding 4: fluid_state (read)
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
                    // binding 5: water_velocity (read)
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
                    // binding 6: wetness (read_write)
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
                ],
            });

        let growth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Moss Growth Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/moss_growth.wgsl").into(),
            ),
        });

        let growth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Moss Growth Pipeline Layout"),
                bind_group_layouts: &[&growth_bind_group_layout],
                push_constant_ranges: &[],
            });

        let growth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Moss Growth Pipeline"),
            layout: Some(&growth_pipeline_layout),
            module: &growth_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Consumption pipeline ===
        let consume_physics_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Moss Consume Physics Layout"),
                entries: &[
                    // binding 0: PhysicsParams uniform
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
                    // binding 1: positions (read)
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
                    // binding 2: cell_count_buffer (read)
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

        let consume_moss_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Moss Consume Moss Layout"),
                entries: &[
                    // binding 0: MossConsumeParams uniform
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
                    // binding 1: moss_density (read_write, atomic)
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
                    // binding 2: cell_types (read)
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
                    // binding 3: nutrients_buffer (read_write, atomic)
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
                    // binding 4: split_nutrient_thresholds (read)
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
                    // binding 5: death_flags (read)
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

        let consume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Moss Consume Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/moss_consume.wgsl").into(),
            ),
        });

        let consume_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Moss Consume Pipeline Layout"),
                bind_group_layouts: &[&consume_physics_layout, &consume_moss_layout],
                push_constant_ranges: &[],
            });

        let consume_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Moss Consume Pipeline"),
            layout: Some(&consume_pipeline_layout),
            module: &consume_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            moss_density_buffer,
            wetness_buffer,
            growth_params_buffer,
            growth_pipeline,
            growth_bind_group_layout,
            consume_params_buffer,
            consume_pipeline,
            consume_physics_layout,
            consume_moss_layout,
            growth_rate: 0.15,
            erosion_rate: 0.3,
            decay_rate: 0.05,
            min_light: 0.05,
            nutrient_per_moss: 50.0,
            graze_cooldown: 5.0,
            water_radius: 20.0,
            wetness_evaporation: 0.02,
            grid_resolution,
            cell_size,
            grid_origin,
            frame_counter: 0,
            growth_frame_skip: 3,
        }
    }

    /// Get the moss density buffer for binding in the cave fragment shader
    pub fn moss_density_buffer(&self) -> &wgpu::Buffer {
        &self.moss_density_buffer
    }

    /// Update grid parameters (call when world size changes)
    pub fn update_grid_params(&mut self, cell_size: f32, grid_origin: [f32; 3]) {
        self.cell_size = cell_size;
        self.grid_origin = grid_origin;
    }

    /// Create the growth bind group (call once, buffers don't change)
    pub fn create_growth_bind_group(
        &self,
        device: &wgpu::Device,
        solid_mask_buffer: &wgpu::Buffer,
        light_field_buffer: &wgpu::Buffer,
        fluid_state_buffer: &wgpu::Buffer,
        water_velocity_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Moss Growth Bind Group"),
            layout: &self.growth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.growth_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.moss_density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fluid_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: water_velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.wetness_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the consumption physics bind group (one per triple buffer index)
    pub fn create_consume_physics_bind_group(
        &self,
        device: &wgpu::Device,
        physics_params_buffer: &wgpu::Buffer,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Moss Consume Physics Bind Group"),
            layout: &self.consume_physics_layout,
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

    /// Create the consumption moss bind group (shared across frames)
    pub fn create_consume_moss_bind_group(
        &self,
        device: &wgpu::Device,
        cell_types_buffer: &wgpu::Buffer,
        nutrients_buffer: &wgpu::Buffer,
        split_nutrient_thresholds_buffer: &wgpu::Buffer,
        death_flags_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Moss Consume Moss Bind Group"),
            layout: &self.consume_moss_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.consume_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.moss_density_buffer.as_entire_binding(),
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
            ],
        })
    }

    /// Run the moss growth/erosion compute pass.
    /// Throttled to every `growth_frame_skip + 1` frames (default: every 4th frame).
    pub fn run_growth(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        growth_bind_group: &wgpu::BindGroup,
        delta_time: f32,
        world_radius: f32,
    ) {
        self.frame_counter += 1;
        if self.frame_counter <= self.growth_frame_skip {
            return;
        }
        let accumulated_dt = delta_time * (self.growth_frame_skip + 1) as f32;
        self.frame_counter = 0;

        let params = MossGrowthParams {
            grid_resolution: self.grid_resolution,
            cell_size: self.cell_size,
            grid_origin_x: self.grid_origin[0],
            grid_origin_y: self.grid_origin[1],
            grid_origin_z: self.grid_origin[2],
            growth_rate: self.growth_rate,
            erosion_rate: self.erosion_rate,
            decay_rate: self.decay_rate,
            min_light: self.min_light,
            delta_time: accumulated_dt,
            world_radius,
            water_radius: self.water_radius,
            wetness_evaporation: self.wetness_evaporation,
            slice_index: 0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(
            &self.growth_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Moss Growth Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.growth_pipeline);
        pass.set_bind_group(0, growth_bind_group, &[]);
        let workgroup_size = 4u32;
        let workgroups = (self.grid_resolution + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(workgroups, workgroups, workgroups);
    }

    /// Run the moss consumption compute pass (phagocytes eating moss)
    pub fn run_consumption(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        physics_bind_group: &wgpu::BindGroup,
        moss_bind_group: &wgpu::BindGroup,
        cell_capacity: u32,
    ) {
        // Update consume params
        let params = MossConsumeParams {
            grid_resolution: self.grid_resolution,
            cell_size: self.cell_size,
            grid_origin_x: self.grid_origin[0],
            grid_origin_y: self.grid_origin[1],
            grid_origin_z: self.grid_origin[2],
            nutrient_per_moss: self.nutrient_per_moss,
            graze_cooldown: self.graze_cooldown,
            _pad0: 0.0,
        };
        queue.write_buffer(
            &self.consume_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Moss Consume Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.consume_pipeline);
        pass.set_bind_group(0, physics_bind_group, &[]);
        pass.set_bind_group(1, moss_bind_group, &[]);

        let workgroups = (cell_capacity + 255) / 256;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
