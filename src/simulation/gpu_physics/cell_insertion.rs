//! GPU Cell Insertion System
//! 
//! Provides GPU-only cell insertion without CPU state management.
//! Uses atomic operations for safe slot allocation and initializes all cell properties.
//! Writes to ALL THREE triple buffer sets to ensure data is available regardless of
//! which buffer the physics pipeline reads from.

use super::{GpuTripleBufferSystem, CellInsertionParams};
use crate::genome::Genome;

/// GPU cell insertion system for direct GPU cell creation
pub struct GpuCellInsertion {
    /// Cell insertion compute pipeline
    pipeline: wgpu::ComputePipeline,
    
    /// Cell insertion parameters uniform buffer
    params_buffer: wgpu::Buffer,
    
    /// Bind groups for cell insertion (writes to all 3 buffer sets)
    physics_bind_group: wgpu::BindGroup,
    params_bind_group: wgpu::BindGroup,
    state_bind_group: wgpu::BindGroup,
}

impl GpuCellInsertion {
    /// Create a new GPU cell insertion system
    /// 
    /// Creates bind groups that write to ALL THREE triple buffer sets for positions,
    /// velocities, and rotations. This ensures inserted cell data is available
    /// regardless of which buffer the physics pipeline reads from.
    pub fn new(
        device: &wgpu::Device,
        pipeline: wgpu::ComputePipeline,
        physics_layout: &wgpu::BindGroupLayout,
        params_layout: &wgpu::BindGroupLayout,
        state_layout: &wgpu::BindGroupLayout,
        buffers: &GpuTripleBufferSystem,
    ) -> Self {
        // Create parameters uniform buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Insertion Params Buffer"),
            size: std::mem::size_of::<CellInsertionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create physics bind group (Group 0) - all 3 position and velocity buffer sets
        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Insertion Physics Bind Group"),
            layout: physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                // All 3 position buffers
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.position_and_mass[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.position_and_mass[2].as_entire_binding(),
                },
                // All 3 velocity buffers
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.velocity[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.velocity[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.velocity[2].as_entire_binding(),
                },
                // Cell count buffer
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create params bind group (Group 1) - all 3 rotation buffer sets
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Insertion Params Bind Group"),
            layout: params_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                // All 3 rotation buffers
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.rotations[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.rotations[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.rotations[2].as_entire_binding(),
                },
            ],
        });
        
        // Create state bind group (Group 2) - single buffers, not triple-buffered
        let state_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Insertion State Bind Group"),
            layout: state_layout,
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
                    resource: buffers.split_ready_frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.next_cell_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.max_cell_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.division_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: buffers.cell_types.as_entire_binding(),
                },
            ],
        });
        
        Self {
            pipeline,
            params_buffer,
            physics_bind_group,
            params_bind_group,
            state_bind_group,
        }
    }
    
    /// Insert a cell directly on GPU using compute shader
    /// 
    /// This method uploads cell parameters and dispatches the compute shader
    /// to atomically allocate a slot and initialize all cell properties.
    /// The shader writes to ALL THREE triple buffer sets.
    pub fn insert_cell(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        position: glam::Vec3,
        velocity: glam::Vec3,
        mass: f32,
        rotation: glam::Quat,
        genome_id: u32,
        mode_index: u32,
        birth_time: f32,
        genomes: &[Genome],
    ) {
        // Get parameters from genome mode
        let (split_interval, split_mass, stiffness, nutrient_gain_rate, max_cell_size, max_splits, cell_type) = 
            if (genome_id as usize) < genomes.len() {
                let genome = &genomes[genome_id as usize];
                if (mode_index as usize) < genome.modes.len() {
                    let mode = &genome.modes[mode_index as usize];
                    // Flagellocytes (cell_type == 1) don't generate their own nutrients
                    let nutrient_rate = if mode.cell_type == 1 { 0.0 } else { mode.nutrient_gain_rate };
                    (
                        mode.split_interval,
                        mode.split_mass,
                        mode.membrane_stiffness,
                        nutrient_rate,
                        mode.max_cell_size,
                        if mode.max_splits < 0 { 0 } else { mode.max_splits as u32 },
                        mode.cell_type as u32
                    )
                } else {
                    (10.0, 2.0, 50.0, 0.2, 2.0, 0, 0) // Default values (Test cell type)
                }
            } else {
                (10.0, 2.0, 50.0, 0.2, 2.0, 0, 0) // Default values (Test cell type)
            };
        
        // Calculate radius from mass
        let radius = (mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        // Convert local mode_index to ABSOLUTE mode_index (with genome offset)
        // This matches how mode_visuals buffer is organized in instance builder:
        // mode_visuals[0..genome0.modes.len()] = genome 0's modes
        // mode_visuals[genome0.modes.len()..] = genome 1's modes, etc.
        let absolute_mode_index = {
            let mut offset = 0u32;
            for (i, genome) in genomes.iter().enumerate() {
                if i == genome_id as usize {
                    break;
                }
                offset += genome.modes.len() as u32;
            }
            offset + mode_index
        };
        
        // Create cell insertion parameters
        let params = CellInsertionParams {
            position: [position.x, position.y, position.z],
            mass,
            velocity: [velocity.x, velocity.y, velocity.z],
            _pad0: 0.0,
            rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
            genome_id,
            mode_index: absolute_mode_index,
            birth_time,
            _pad1: 0.0,
            split_interval,
            split_mass,
            stiffness,
            radius,
            nutrient_gain_rate,
            max_cell_size,
            max_splits,
            cell_id: 0, // Let shader generate new cell ID
            cell_type,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        };
        
        // Upload parameters to GPU
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        
        // Dispatch compute shader (single workgroup as per requirements)
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cell Insertion Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.params_bind_group, &[]);
        compute_pass.set_bind_group(2, &self.state_bind_group, &[]);
        
        // Single workgroup dispatch (1,1,1) for atomic safety as per requirements
        compute_pass.dispatch_workgroups(1, 1, 1);
        
        drop(compute_pass);
    }
    
    /// Insert a cell with specific cell ID (for deterministic insertion)
    pub fn insert_cell_with_id(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        position: glam::Vec3,
        velocity: glam::Vec3,
        mass: f32,
        rotation: glam::Quat,
        genome_id: u32,
        mode_index: u32,
        birth_time: f32,
        cell_id: u32,
        genomes: &[Genome],
    ) {
        // Get parameters from genome mode
        let (split_interval, split_mass, stiffness, nutrient_gain_rate, max_cell_size, max_splits, cell_type) = 
            if (genome_id as usize) < genomes.len() {
                let genome = &genomes[genome_id as usize];
                if (mode_index as usize) < genome.modes.len() {
                    let mode = &genome.modes[mode_index as usize];
                    // Flagellocytes (cell_type == 1) don't generate their own nutrients
                    let nutrient_rate = if mode.cell_type == 1 { 0.0 } else { mode.nutrient_gain_rate };
                    (
                        mode.split_interval,
                        mode.split_mass,
                        mode.membrane_stiffness,
                        nutrient_rate,
                        mode.max_cell_size,
                        if mode.max_splits < 0 { 0 } else { mode.max_splits as u32 },
                        mode.cell_type as u32
                    )
                } else {
                    (10.0, 2.0, 50.0, 0.2, 2.0, 0, 0) // Default values (Test cell type)
                }
            } else {
                (10.0, 2.0, 50.0, 0.2, 2.0, 0, 0) // Default values (Test cell type)
            };
        
        // Calculate radius from mass
        let radius = (mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        // Convert local mode_index to ABSOLUTE mode_index (with genome offset)
        // This matches how mode_visuals buffer is organized in instance builder:
        // mode_visuals[0..genome0.modes.len()] = genome 0's modes
        // mode_visuals[genome0.modes.len()..] = genome 1's modes, etc.
        let absolute_mode_index = {
            let mut offset = 0u32;
            for (i, genome) in genomes.iter().enumerate() {
                if i == genome_id as usize {
                    break;
                }
                offset += genome.modes.len() as u32;
            }
            offset + mode_index
        };
        
        // Create cell insertion parameters with specific cell ID
        let params = CellInsertionParams {
            position: [position.x, position.y, position.z],
            mass,
            velocity: [velocity.x, velocity.y, velocity.z],
            _pad0: 0.0,
            rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
            genome_id,
            mode_index: absolute_mode_index,
            birth_time,
            _pad1: 0.0,
            split_interval,
            split_mass,
            stiffness,
            radius,
            nutrient_gain_rate,
            max_cell_size,
            max_splits,
            cell_id, // Use provided cell ID
            cell_type,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        };
        
        // DEBUG: Log cell insertion parameters
        let type_name = match cell_type {
            0 => "Test",
            1 => "Flagellocyte",
            _ => "Unknown",
        };
        println!("[DEBUG CELL_INSERTION] Inserting cell:");
        println!("  genome_id: {}, local_mode_index: {}, absolute_mode_index: {}", genome_id, mode_index, absolute_mode_index);
        println!("  cell_type: {} ({})", cell_type, type_name);
        println!("  position: ({:.2}, {:.2}, {:.2})", position.x, position.y, position.z);
        
        // Upload parameters to GPU
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        
        // Dispatch compute shader (single workgroup as per requirements)
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cell Insertion Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.params_bind_group, &[]);
        compute_pass.set_bind_group(2, &self.state_bind_group, &[]);
        
        // Single workgroup dispatch (1,1,1) for atomic safety as per requirements
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
}
