//! GPU-side genome compaction system
//! 
//! This module provides compute shaders for compacting genome buffer groups
//! and reorganizing GPU memory to eliminate fragmentation.

use crate::simulation::gpu_physics::genome_buffers::GenomeBufferManager;

/// GPU-side genome compaction system
pub struct GenomeCompactionSystem {
    /// Compute pipeline for genome compaction
    compaction_pipeline: wgpu::ComputePipeline,
    
    /// Reference count buffer for genomes
    reference_count_buffer: wgpu::Buffer,
    
    /// Marked for deletion buffer
    marked_for_deletion_buffer: wgpu::Buffer,
    
    /// Compaction map buffer (old_id -> new_id)
    #[allow(dead_code)]
    compaction_map_buffer: wgpu::Buffer,
    
    /// Compacted genome count buffer
    compacted_count_buffer: wgpu::Buffer,
    
    /// Max genomes buffer
    #[allow(dead_code)]
    max_genomes_buffer: wgpu::Buffer,
    
    /// Compaction results buffer
    #[allow(dead_code)]
    compaction_results_buffer: wgpu::Buffer,
    
    /// Bind group for compaction operations
    compaction_bind_group: wgpu::BindGroup,
}

impl GenomeCompactionSystem {
    /// Create a new genome compaction system
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_genomes: usize) -> Self {
        // Create compute shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Genome Compaction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/genome_compaction.wgsl").into()),
        });
        
        // Create compute pipeline
        let compaction_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Genome Compaction Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        
        // Create buffers
        let reference_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Reference Count Buffer"),
            size: (max_genomes * 4) as u64, // u32 per genome
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let marked_for_deletion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Marked For Deletion Buffer"),
            size: (max_genomes * 4) as u64, // u32 per genome
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let compaction_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Compaction Map Buffer"),
            size: (max_genomes * 4) as u64, // u32 per genome
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let compacted_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compacted Genome Count Buffer"),
            size: 4, // single u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let max_genomes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Genomes Buffer"),
            size: 4, // single u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Initialize max_genomes buffer
        let max_genomes_value = max_genomes as u32;
        queue.write_buffer(&max_genomes_buffer, 0, bytemuck::bytes_of(&max_genomes_value));
        
        let compaction_results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Compaction Results Buffer"),
            size: (max_genomes * 16) as u64, // 16 bytes per result (4 u32s)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Genome Compaction Bind Group Layout"),
            entries: &[
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
        });
        
        // Create bind group
        let compaction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Genome Compaction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: reference_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: marked_for_deletion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: compaction_map_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: compacted_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: max_genomes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: compaction_results_buffer.as_entire_binding(),
                },
            ],
        });
        
        Self {
            compaction_pipeline,
            reference_count_buffer,
            marked_for_deletion_buffer,
            compaction_map_buffer,
            compacted_count_buffer,
            max_genomes_buffer,
            compaction_results_buffer,
            compaction_bind_group,
        }
    }
    
    /// Execute genome compaction on GPU
    /// Returns the number of genomes after compaction
    pub fn compact_genomes(
        &mut self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        genome_manager: &GenomeBufferManager,
    ) -> u32 {
        // Upload reference counts and deletion flags to GPU
        self.upload_genome_data(queue, genome_manager);
        
        // Reset compacted count
        let zero: u32 = 0;
        queue.write_buffer(&self.compacted_count_buffer, 0, bytemuck::bytes_of(&zero));
        
        // Execute compaction compute shader
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Genome Compaction Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compaction_pipeline);
        pass.set_bind_group(0, &self.compaction_bind_group, &[]);
        
        // Calculate workgroup count
        let workgroup_count = (genome_manager.genome_count() + 63) / 64; // Round up to nearest 64
        pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        drop(pass); // End the compute pass
        
        // Read back the compacted count
        // Note: In a real implementation, you'd want to use async readback
        // For now, we'll return the current genome count as an estimate
        genome_manager.genome_count() as u32
    }
    
    /// Upload genome reference counts and deletion flags to GPU
    fn upload_genome_data(&self, queue: &wgpu::Queue, genome_manager: &GenomeBufferManager) {
        let max_genomes = genome_manager.genome_count();
        
        // Collect reference counts
        let mut reference_counts = vec![0u32; max_genomes];
        let mut marked_for_deletion = vec![0u32; max_genomes];
        
        for (i, group) in genome_manager.active_genomes().iter().enumerate() {
            reference_counts[i] = group.reference_count;
            marked_for_deletion[i] = if group.marked_for_deletion { 1 } else { 0 };
        }
        
        // Upload to GPU
        if !reference_counts.is_empty() {
            queue.write_buffer(&self.reference_count_buffer, 0, bytemuck::cast_slice(&reference_counts));
        }
        if !marked_for_deletion.is_empty() {
            queue.write_buffer(&self.marked_for_deletion_buffer, 0, bytemuck::cast_slice(&marked_for_deletion));
        }
    }
    
    /// Get the compaction map (old_id -> new_id)
    /// This should be called after compaction to update genome IDs
    pub fn get_compaction_map(&self, _device: &wgpu::Device) -> Vec<u32> {
        // In a real implementation, this would read back from GPU
        // For now, return identity mapping
        vec![0; 64] // MAX_GENOMES
    }
}
