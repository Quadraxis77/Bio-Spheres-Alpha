//! Light Field System
//!
//! Computes per-voxel light intensity on the GPU using ray marching.
//! The light field considers cave solid voxels and cell occupancy as occluders.
//! Used by:
//!   - Photocyte cells for light-based nutrient gain
//!   - Volumetric fog renderer for god rays and shadow casting

use bytemuck::{Pod, Zeroable};

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
    cell_occupancy_buffer: wgpu::Buffer,
    light_field_params_buffer: wgpu::Buffer,
    occupancy_params_buffer: wgpu::Buffer,
    photocyte_params_buffer: wgpu::Buffer,

    // Compute pipelines
    clear_occupancy_pipeline: wgpu::ComputePipeline,
    build_occupancy_pipeline: wgpu::ComputePipeline,
    compute_light_pipeline: wgpu::ComputePipeline,
    photocyte_light_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    light_field_layout: wgpu::BindGroupLayout,
    occupancy_layout: wgpu::BindGroupLayout,
    photocyte_physics_layout: wgpu::BindGroupLayout,
    photocyte_system_layout: wgpu::BindGroupLayout,

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

        let occupancy_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Occupancy Params"),
            size: std::mem::size_of::<OccupancyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
        let light_field_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    // Binding 1: light_field (read-only storage)
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
                    // Binding 3: split_masses (read-only storage)
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

        Self {
            light_field_buffer,
            cell_occupancy_buffer,
            light_field_params_buffer,
            occupancy_params_buffer,
            photocyte_params_buffer,
            clear_occupancy_pipeline,
            build_occupancy_pipeline,
            compute_light_pipeline,
            photocyte_light_pipeline,
            light_field_layout,
            occupancy_layout,
            photocyte_physics_layout,
            photocyte_system_layout,
            light_dir,
            max_steps: 128,
            step_size: 2.0,
            absorption_solid: 8.0,
            absorption_cell: 10.0,
            ambient_floor: 0.02,
            scattering_coefficient: 0.3,
            mass_per_second_full_light: 0.3,
            min_light_threshold: 0.05,
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

    /// Get the cell occupancy buffer
    pub fn cell_occupancy_buffer(&self) -> &wgpu::Buffer {
        &self.cell_occupancy_buffer
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
    fn update_light_field_params(&self, queue: &wgpu::Queue, time: f32) {
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
        };
        queue.write_buffer(
            &self.light_field_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    fn update_occupancy_params(&self, queue: &wgpu::Queue) {
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

    fn update_photocyte_params(&self, queue: &wgpu::Queue) {
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
        split_masses_buffer: &wgpu::Buffer,
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
                    resource: split_masses_buffer.as_entire_binding(),
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
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
        physics_params_buffer: &wgpu::Buffer,
        cell_types_buffer: &wgpu::Buffer,
        split_masses_buffer: &wgpu::Buffer,
        cell_count: u32,
        time: f32,
    ) {
        // Update all params
        self.update_light_field_params(queue, time);
        self.update_occupancy_params(queue);
        self.update_photocyte_params(queue);

        let total_voxels = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
        let voxel_workgroups = (total_voxels + 255) / 256;
        let light_workgroups = (total_voxels + 63) / 64;

        // Create bind groups
        let occupancy_bg =
            self.create_occupancy_bind_group(device, positions_buffer, cell_count_buffer);
        let light_field_bg = self.create_light_field_bind_group(device, solid_mask_buffer);
        let photocyte_physics_bg = self.create_photocyte_physics_bind_group(
            device,
            physics_params_buffer,
            positions_buffer,
            cell_count_buffer,
        );
        let photocyte_system_bg = self.create_photocyte_system_bind_group(
            device,
            cell_types_buffer,
            split_masses_buffer,
        );

        // Step 1: Clear occupancy grid
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Cell Occupancy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_occupancy_pipeline);
            pass.set_bind_group(0, &occupancy_bg, &[]);
            pass.dispatch_workgroups(voxel_workgroups, 1, 1);
        }

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

        // Step 4: Photocyte light consumption
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
    }

    /// Run only the photocyte light consumption step.
    /// Call this inside each physics step so photocytes gain/lose mass at the same rate as other cells.
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
        split_masses_buffer: &wgpu::Buffer,
        cell_count: u32,
    ) {
        if cell_count == 0 {
            return;
        }

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
            split_masses_buffer,
        );

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

    /// Run only the light field computation (without photocyte consumption)
    /// Useful when you only need the light field for rendering (volumetric fog)
    pub fn run_light_field_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        solid_mask_buffer: &wgpu::Buffer,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
        cell_count: u32,
        time: f32,
    ) {
        self.update_light_field_params(queue, time);
        self.update_occupancy_params(queue);

        let total_voxels = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
        let voxel_workgroups = (total_voxels + 255) / 256;
        let light_workgroups = (total_voxels + 63) / 64;

        let occupancy_bg =
            self.create_occupancy_bind_group(device, positions_buffer, cell_count_buffer);
        let light_field_bg = self.create_light_field_bind_group(device, solid_mask_buffer);

        // Clear + build occupancy
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Cell Occupancy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_occupancy_pipeline);
            pass.set_bind_group(0, &occupancy_bg, &[]);
            pass.dispatch_workgroups(voxel_workgroups, 1, 1);
        }

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
