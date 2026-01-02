//! GPU-based instance buffer builder using compute shaders.
//!
//! Builds cell instance data on the GPU to eliminate CPU-side iteration
//! and reduce CPU→GPU data transfer. Includes dirty tracking and buffer reuse.

use crate::cell::types::CellTypeVisuals;
use crate::genome::Genome;
use crate::simulation::CanonicalState;
use bytemuck::{Pod, Zeroable};

/// GPU instance builder with compute shader pipeline.
pub struct InstanceBuilder {
    // Compute pipeline
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    
    // Uniform buffer
    params_buffer: wgpu::Buffer,
    
    // Input buffers (simulation data)
    positions_buffer: wgpu::Buffer,
    rotations_buffer: wgpu::Buffer,
    radii_buffer: wgpu::Buffer,
    mode_indices_buffer: wgpu::Buffer,
    cell_ids_buffer: wgpu::Buffer,
    genome_ids_buffer: wgpu::Buffer,
    
    // Lookup table buffers
    mode_visuals_buffer: wgpu::Buffer,
    cell_type_visuals_buffer: wgpu::Buffer,
    
    // Output buffer (instance data for rendering)
    pub instance_buffer: wgpu::Buffer,
    
    // Counter buffer for atomic operations
    counters_buffer: wgpu::Buffer,
    
    // Current bind group (recreated when buffers change size)
    bind_group: Option<wgpu::BindGroup>,
    
    // Capacity tracking
    cell_capacity: usize,
    mode_capacity: usize,
    cell_type_capacity: usize,
    
    // Dirty tracking
    positions_dirty: bool,
    rotations_dirty: bool,
    radii_dirty: bool,
    mode_indices_dirty: bool,
    cell_ids_dirty: bool,
    genome_ids_dirty: bool,
    mode_visuals_dirty: bool,
    cell_type_visuals_dirty: bool,
    
    // Hash for change detection
    last_cell_count: usize,
    last_mode_hash: u64,
    last_visuals_hash: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BuildParams {
    cell_count: u32,
    mode_count: u32,
    cell_type_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ModeVisuals {
    color: [f32; 3],
    opacity: f32,
    emissive: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuCellTypeVisuals {
    specular_strength: f32,
    specular_power: f32,
    fresnel_strength: f32,
    membrane_noise_scale: f32,
    membrane_noise_strength: f32,
    membrane_noise_speed: f32,
    _pad: [f32; 2],
}

// Instance struct must match shader and CellRenderer
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CellInstance {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    pub visual_params: [f32; 4],
    pub membrane_params: [f32; 4],
    pub rotation: [f32; 4],
}

impl InstanceBuilder {
    /// Create a new instance builder with the given capacity.
    pub fn new(device: &wgpu::Device, cell_capacity: usize) -> Self {
        let mode_capacity = 64; // Typical max modes
        let cell_type_capacity = 16; // Typical max cell types
        
        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Instance Builder Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/build_instances.wgsl").into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instance Builder Bind Group Layout"),
            entries: &[
                // params uniform
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
                // positions
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
                // rotations
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
                // radii
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
                // mode_indices
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
                // cell_ids
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
                // genome_ids
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
                // mode_visuals
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
                // cell_type_visuals
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
                // instances (output)
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
                // counters
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
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instance Builder Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Instance Builder Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create buffers
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Params"),
            size: std::mem::size_of::<BuildParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let positions_buffer = Self::create_storage_buffer(device, "Positions", cell_capacity * 16); // vec4
        let rotations_buffer = Self::create_storage_buffer(device, "Rotations", cell_capacity * 16); // vec4
        let radii_buffer = Self::create_storage_buffer(device, "Radii", cell_capacity * 4); // f32
        let mode_indices_buffer = Self::create_storage_buffer(device, "Mode Indices", cell_capacity * 4); // u32
        let cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", cell_capacity * 4); // u32
        let genome_ids_buffer = Self::create_storage_buffer(device, "Genome IDs", cell_capacity * 4); // u32
        
        let mode_visuals_buffer = Self::create_storage_buffer(device, "Mode Visuals", mode_capacity * std::mem::size_of::<ModeVisuals>());
        let cell_type_visuals_buffer = Self::create_storage_buffer(device, "Cell Type Visuals", cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>());
        
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (cell_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let counters_buffer = Self::create_storage_buffer(device, "Counters", 8); // 2 x u32
        
        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            positions_buffer,
            rotations_buffer,
            radii_buffer,
            mode_indices_buffer,
            cell_ids_buffer,
            genome_ids_buffer,
            mode_visuals_buffer,
            cell_type_visuals_buffer,
            instance_buffer,
            counters_buffer,
            bind_group: None,
            cell_capacity,
            mode_capacity,
            cell_type_capacity,
            positions_dirty: true,
            rotations_dirty: true,
            radii_dirty: true,
            mode_indices_dirty: true,
            cell_ids_dirty: true,
            genome_ids_dirty: true,
            mode_visuals_dirty: true,
            cell_type_visuals_dirty: true,
            last_cell_count: 0,
            last_mode_hash: 0,
            last_visuals_hash: 0,
        }
    }
    
    fn create_storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Instance Builder {}", label)),
            size: size.max(16) as u64, // Minimum 16 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    /// Mark all buffers as dirty (forces full update).
    pub fn mark_all_dirty(&mut self) {
        self.positions_dirty = true;
        self.rotations_dirty = true;
        self.radii_dirty = true;
        self.mode_indices_dirty = true;
        self.cell_ids_dirty = true;
        self.genome_ids_dirty = true;
        self.mode_visuals_dirty = true;
        self.cell_type_visuals_dirty = true;
    }
    
    /// Update input buffers from simulation state (only dirty data).
    pub fn update_from_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
    ) {
        let cell_count = state.cell_count;
        
        // Check if we need to resize buffers
        if cell_count > self.cell_capacity {
            self.resize_cell_buffers(device, cell_count * 2);
        }
        
        // Detect changes via cell count
        if cell_count != self.last_cell_count {
            self.mark_all_dirty();
            self.last_cell_count = cell_count;
        }
        
        if cell_count == 0 {
            return;
        }
        
        // Update positions (always dirty in simulation)
        // Convert Vec3 to vec4 for GPU alignment
        let positions_vec4: Vec<[f32; 4]> = state.positions[..cell_count]
            .iter()
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();
        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions_vec4));
        
        // Update rotations
        let rotations: Vec<[f32; 4]> = state.rotations[..cell_count]
            .iter()
            .map(|q| [q.x, q.y, q.z, q.w])
            .collect();
        queue.write_buffer(&self.rotations_buffer, 0, bytemuck::cast_slice(&rotations));
        
        // Update radii
        queue.write_buffer(&self.radii_buffer, 0, bytemuck::cast_slice(&state.radii[..cell_count]));
        
        // Update mode indices (convert usize to u32)
        let mode_indices: Vec<u32> = state.mode_indices[..cell_count]
            .iter()
            .map(|&i| i as u32)
            .collect();
        queue.write_buffer(&self.mode_indices_buffer, 0, bytemuck::cast_slice(&mode_indices));
        
        // Update cell IDs
        queue.write_buffer(&self.cell_ids_buffer, 0, bytemuck::cast_slice(&state.cell_ids[..cell_count]));
        
        // Update genome IDs (convert usize to u32)
        let genome_ids: Vec<u32> = state.genome_ids[..cell_count]
            .iter()
            .map(|&i| i as u32)
            .collect();
        queue.write_buffer(&self.genome_ids_buffer, 0, bytemuck::cast_slice(&genome_ids));
        
        // Update mode visuals from genome
        if let Some(genome) = genome {
            let mode_hash = Self::hash_genome_modes(genome);
            if mode_hash != self.last_mode_hash {
                self.update_mode_visuals(device, queue, genome);
                self.last_mode_hash = mode_hash;
            }
        }
        
        // Update cell type visuals
        if let Some(visuals) = cell_type_visuals {
            let visuals_hash = Self::hash_cell_type_visuals(visuals);
            if visuals_hash != self.last_visuals_hash {
                self.update_cell_type_visuals(device, queue, visuals);
                self.last_visuals_hash = visuals_hash;
            }
        }
        
        // Recreate bind group if needed
        if self.bind_group.is_none() {
            self.recreate_bind_group(device);
        }
    }
    
    fn hash_genome_modes(genome: &Genome) -> u64 {
        let mut hash = genome.modes.len() as u64;
        for mode in &genome.modes {
            hash = hash.wrapping_mul(31).wrapping_add((mode.color.x * 1000.0) as u64);
            hash = hash.wrapping_mul(31).wrapping_add((mode.opacity * 1000.0) as u64);
        }
        hash
    }
    
    fn hash_cell_type_visuals(visuals: &[CellTypeVisuals]) -> u64 {
        let mut hash = visuals.len() as u64;
        for v in visuals {
            hash = hash.wrapping_mul(31).wrapping_add((v.specular_strength * 1000.0) as u64);
            hash = hash.wrapping_mul(31).wrapping_add((v.membrane_noise_scale * 1000.0) as u64);
        }
        hash
    }
    
    fn update_mode_visuals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, genome: &Genome) {
        let mode_count = genome.modes.len();
        
        // Resize if needed
        if mode_count > self.mode_capacity {
            self.mode_capacity = mode_count * 2;
            self.mode_visuals_buffer = Self::create_storage_buffer(
                device,
                "Mode Visuals",
                self.mode_capacity * std::mem::size_of::<ModeVisuals>(),
            );
            self.bind_group = None;
        }
        
        let mode_visuals: Vec<ModeVisuals> = genome.modes
            .iter()
            .map(|mode| ModeVisuals {
                color: mode.color.to_array(),
                opacity: mode.opacity,
                emissive: mode.emissive,
                _pad: [0.0; 3],
            })
            .collect();
        
        if !mode_visuals.is_empty() {
            queue.write_buffer(&self.mode_visuals_buffer, 0, bytemuck::cast_slice(&mode_visuals));
        }
    }
    
    fn update_cell_type_visuals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, visuals: &[CellTypeVisuals]) {
        let count = visuals.len();
        
        // Resize if needed
        if count > self.cell_type_capacity {
            self.cell_type_capacity = count * 2;
            self.cell_type_visuals_buffer = Self::create_storage_buffer(
                device,
                "Cell Type Visuals",
                self.cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>(),
            );
            self.bind_group = None;
        }
        
        let gpu_visuals: Vec<GpuCellTypeVisuals> = visuals
            .iter()
            .map(|v| GpuCellTypeVisuals {
                specular_strength: v.specular_strength,
                specular_power: v.specular_power,
                fresnel_strength: v.fresnel_strength,
                membrane_noise_scale: v.membrane_noise_scale,
                membrane_noise_strength: v.membrane_noise_strength,
                membrane_noise_speed: v.membrane_noise_speed,
                _pad: [0.0; 2],
            })
            .collect();
        
        if !gpu_visuals.is_empty() {
            queue.write_buffer(&self.cell_type_visuals_buffer, 0, bytemuck::cast_slice(&gpu_visuals));
        }
    }
    
    fn resize_cell_buffers(&mut self, device: &wgpu::Device, new_capacity: usize) {
        self.cell_capacity = new_capacity;
        
        self.positions_buffer = Self::create_storage_buffer(device, "Positions", new_capacity * 16);
        self.rotations_buffer = Self::create_storage_buffer(device, "Rotations", new_capacity * 16);
        self.radii_buffer = Self::create_storage_buffer(device, "Radii", new_capacity * 4);
        self.mode_indices_buffer = Self::create_storage_buffer(device, "Mode Indices", new_capacity * 4);
        self.cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", new_capacity * 4);
        self.genome_ids_buffer = Self::create_storage_buffer(device, "Genome IDs", new_capacity * 4);
        
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (new_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.bind_group = None;
        self.mark_all_dirty();
    }
    
    fn recreate_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance Builder Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rotations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mode_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cell_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.genome_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.mode_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cell_type_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.counters_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    
    /// Run the compute shader to build instance data.
    pub fn build_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell_count: usize,
        mode_count: usize,
        cell_type_count: usize,
    ) {
        if cell_count == 0 {
            return;
        }
        
        // Update params
        let params = BuildParams {
            cell_count: cell_count as u32,
            mode_count: mode_count as u32,
            cell_type_count: cell_type_count as u32,
            _pad: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        
        // Ensure bind group exists
        if self.bind_group.is_none() {
            self.recreate_bind_group(device);
        }
        
        let bind_group = self.bind_group.as_ref().unwrap();
        
        // Create command encoder and run compute pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Instance Builder Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Instance Builder Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            
            // Dispatch with workgroup size of 64
            let workgroup_count = (cell_count as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        queue.submit(std::iter::once(encoder.finish()));
    }
    
    /// Get the instance buffer for rendering.
    pub fn get_instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }
    
    /// Get current cell capacity.
    pub fn capacity(&self) -> usize {
        self.cell_capacity
    }
}
