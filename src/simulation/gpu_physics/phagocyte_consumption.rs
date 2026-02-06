//! Phagocyte Nutrient Consumption System
//!
//! Handles phagocytes consuming nutrients from voxels and gaining mass.
//! Runs as a compute shader after the main physics step.

use bytemuck::{Pod, Zeroable};

/// Nutrient consumption parameters for GPU shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NutrientConsumptionParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub mass_per_nutrient: f32,  // How much mass a phagocyte gains per nutrient consumed
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Phagocyte consumption system for GPU
pub struct PhagocyteConsumptionSystem {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,  // Physics group
    bind_group_layout_1: wgpu::BindGroupLayout,  // Nutrient system group
    params_buffer: wgpu::Buffer,
    mass_per_nutrient: f32,
}

impl PhagocyteConsumptionSystem {
    pub fn new(device: &wgpu::Device, grid_resolution: u32, cell_size: f32, grid_origin: [f32; 3]) -> Self {
        let mass_per_nutrient = 0.2;  // Increased mass gain per nutrient (from 0.1 for better efficiency)
        
        // Create params buffer with initial values
        let _initial_params = NutrientConsumptionParams {
            grid_resolution,
            cell_size,
            grid_origin_x: grid_origin[0],
            grid_origin_y: grid_origin[1],
            grid_origin_z: grid_origin[2],
            mass_per_nutrient,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Phagocyte Consumption Params"),
            size: std::mem::size_of::<NutrientConsumptionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Note: initial_params will be written on first update_params call
        
        // Physics bind group layout (group 0) - simplified to only what shader needs
        let bind_group_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Phagocyte Consumption Physics Layout"),
            entries: &[
                // Binding 0: Physics params (uniform)
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
                // Binding 1: positions (read_write for reading position and writing mass)
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
                // Binding 2: cell_count_buffer (read)
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
        
        // Nutrient system bind group layout (group 1)
        let bind_group_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Phagocyte Consumption Nutrient Layout"),
            entries: &[
                // Binding 0: nutrient_params (uniform)
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
                // Binding 1: fluid_state (read)
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
                // Binding 2: nutrient_voxels (read_write for atomic)
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
                // Binding 3: cell_types (read)
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
                // Binding 4: max_cell_sizes (read)
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
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Phagocyte Consumption Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/phagocyte_consume.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Phagocyte Consumption Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Phagocyte Consumption Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Write initial params
        Self {
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            params_buffer,
            mass_per_nutrient,
        }
    }
    
    /// Update consumption parameters
    pub fn update_params(&self, queue: &wgpu::Queue, grid_resolution: u32, cell_size: f32, grid_origin: [f32; 3]) {
        let params = NutrientConsumptionParams {
            grid_resolution,
            cell_size,
            grid_origin_x: grid_origin[0],
            grid_origin_y: grid_origin[1],
            grid_origin_z: grid_origin[2],
            mass_per_nutrient: self.mass_per_nutrient,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }
    
    /// Set mass gained per nutrient consumed
    pub fn set_mass_per_nutrient(&mut self, mass: f32) {
        self.mass_per_nutrient = mass;
    }
    
    /// Create physics bind group (group 0) - simplified
    pub fn create_physics_bind_group(
        &self,
        device: &wgpu::Device,
        physics_params_buffer: &wgpu::Buffer,
        positions_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Phagocyte Consumption Physics Bind Group"),
            layout: &self.bind_group_layout_0,
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
    
    /// Create nutrient system bind group (group 1)
    pub fn create_nutrient_bind_group(
        &self,
        device: &wgpu::Device,
        fluid_state_buffer: &wgpu::Buffer,
        nutrient_voxels_buffer: &wgpu::Buffer,
        cell_types_buffer: &wgpu::Buffer,
        max_cell_sizes_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Phagocyte Consumption Nutrient Bind Group"),
            layout: &self.bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fluid_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nutrient_voxels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_types_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: max_cell_sizes_buffer.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Run the phagocyte consumption compute shader
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        physics_bind_group: &wgpu::BindGroup,
        nutrient_bind_group: &wgpu::BindGroup,
        cell_count: u32,
    ) {
        if cell_count == 0 {
            return;
        }
        
        let workgroup_count = (cell_count + 255) / 256;
        
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Phagocyte Consumption Pass"),
            timestamp_writes: None,
        });
        
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, physics_bind_group, &[]);
        pass.set_bind_group(1, nutrient_bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
