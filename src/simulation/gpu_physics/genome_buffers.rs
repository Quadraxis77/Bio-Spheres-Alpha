//! Per-Genome Buffer Management System
//! 
//! This module implements isolated buffer groups for each genome to eliminate
//! indexing conflicts and enable independent genome management.

use crate::genome::Genome;

/// Helper function to create aligned GPU buffers
fn create_aligned_buffer(
    device: &wgpu::Device,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    // Align size to 16-byte boundary for GPU compatibility
    let aligned_size = (size + 15) & !15; // Round up to nearest 16 bytes
    
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size,
        usage,
        mapped_at_creation: false,
    })
}

/// Maximum number of genomes supported simultaneously
/// Increased to support 20,000 genomes
pub const MAX_GENOMES: usize = 20_000;

/// Per-genome buffer group containing all genome-specific data
/// This eliminates global mode indexing conflicts between genomes
#[derive(Debug)]
pub struct GenomeBufferGroup {
    /// Genome ID (index in the global genomes array)
    pub genome_id: usize,
    
    /// Number of modes in this genome
    pub mode_count: usize,
    
    /// Mode properties buffer (per-mode data for division)
    /// Layout per mode: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, 
    ///                  split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, padding x3]
    pub mode_properties: wgpu::Buffer,
    
    /// Mode cell types lookup table
    pub mode_cell_types: wgpu::Buffer,
    
    /// Child mode indices (for division)
    pub child_mode_indices: wgpu::Buffer,
    
    /// Parent make adhesion flags
    pub parent_make_adhesion_flags: wgpu::Buffer,
    
    /// Child keep adhesion flags
    pub child_a_keep_adhesion_flags: wgpu::Buffer,
    pub child_b_keep_adhesion_flags: wgpu::Buffer,
    
    /// Genome mode data (child orientations, split directions)
    pub genome_mode_data: wgpu::Buffer,
    
    /// Reference count - how many cells are using this genome
    pub reference_count: u32,
    
    /// Whether this genome group is marked for deletion
    pub marked_for_deletion: bool,
    
    /// Whether buffers need synchronization with CPU data
    pub needs_sync: bool,
}

impl GenomeBufferGroup {
    /// Create a new genome buffer group for the given genome
    pub fn new(
        device: &wgpu::Device,
        genome_id: usize,
        genome: &Genome,
    ) -> Self {
        let mode_count = genome.modes.len();
        
        // Calculate buffer sizes
        let mode_properties_size = mode_count * 48; // 48 bytes per mode
        let mode_cell_types_size = mode_count * 4;  // 4 bytes per mode
        let child_mode_indices_size = mode_count * 8; // 8 bytes per mode (2 i32)
        let parent_flags_size = mode_count * 4;      // 4 bytes per mode
        let child_flags_size = mode_count * 4;       // 4 bytes per mode
        let genome_mode_data_size = mode_count * 48; // 48 bytes per mode (3 vec4)
        
        // Create buffers
        let mode_properties = create_aligned_buffer(
            device,
            mode_properties_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Mode Properties", genome_id),
        );
        
        let mode_cell_types = create_aligned_buffer(
            device,
            mode_cell_types_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Mode Cell Types", genome_id),
        );
        
        let child_mode_indices = create_aligned_buffer(
            device,
            child_mode_indices_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Child Mode Indices", genome_id),
        );
        
        let parent_make_adhesion_flags = create_aligned_buffer(
            device,
            parent_flags_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Parent Make Adhesion Flags", genome_id),
        );
        
        let child_a_keep_adhesion_flags = create_aligned_buffer(
            device,
            child_flags_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Child A Keep Adhesion Flags", genome_id),
        );
        
        let child_b_keep_adhesion_flags = create_aligned_buffer(
            device,
            child_flags_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Child B Keep Adhesion Flags", genome_id),
        );
        
        let genome_mode_data = create_aligned_buffer(
            device,
            genome_mode_data_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            &format!("Genome {} Mode Data", genome_id),
        );
        
        Self {
            genome_id,
            mode_count,
            mode_properties,
            mode_cell_types,
            child_mode_indices,
            parent_make_adhesion_flags,
            child_a_keep_adhesion_flags,
            child_b_keep_adhesion_flags,
            genome_mode_data,
            reference_count: 0,
            marked_for_deletion: false,
            needs_sync: true,
        }
    }
    
    /// Synchronize this genome's data from the CPU genome to GPU buffers
    pub fn sync_from_genome(&mut self, queue: &wgpu::Queue, genome: &Genome) {
        if self.mode_count != genome.modes.len() {
            // Mode count changed - need to recreate buffers
            // For now, mark as needing recreation
            self.needs_sync = true;
            return;
        }
        
        // Sync mode properties
        let mode_properties_data: Vec<[f32; 12]> = genome.modes.iter().map(|mode| {
            [
                mode.nutrient_gain_rate,
                mode.max_cell_size,
                mode.membrane_stiffness,
                mode.split_interval,
                mode.split_mass,
                mode.nutrient_priority,
                mode.swim_force,
                if mode.prioritize_when_low { 1.0 } else { 0.0 },
                mode.max_splits as f32,
                mode.split_ratio,
                mode.buoyancy_force,
                0.0, // padding
            ]
        }).collect();
        
        if !mode_properties_data.is_empty() {
            queue.write_buffer(&self.mode_properties, 0, bytemuck::cast_slice(&mode_properties_data));
        }
        
        // Sync mode cell types
        let mode_cell_types_data: Vec<u32> = genome.modes.iter()
            .map(|mode| mode.cell_type as u32)
            .collect();
        
        if !mode_cell_types_data.is_empty() {
            queue.write_buffer(&self.mode_cell_types, 0, bytemuck::cast_slice(&mode_cell_types_data));
        }
        
        // Sync child mode indices
        let child_mode_indices_data: Vec<[i32; 2]> = genome.modes.iter().map(|mode| {
            [mode.child_a.mode_number as i32, mode.child_b.mode_number as i32]
        }).collect();
        
        if !child_mode_indices_data.is_empty() {
            queue.write_buffer(&self.child_mode_indices, 0, bytemuck::cast_slice(&child_mode_indices_data));
        }
        
        // Sync parent make adhesion flags
        let parent_flags_data: Vec<u32> = genome.modes.iter()
            .map(|mode| if mode.parent_make_adhesion { 1 } else { 0 })
            .collect();
        
        if !parent_flags_data.is_empty() {
            queue.write_buffer(&self.parent_make_adhesion_flags, 0, bytemuck::cast_slice(&parent_flags_data));
        }
        
        // Sync child keep adhesion flags
        let child_a_flags_data: Vec<u32> = genome.modes.iter()
            .map(|mode| if mode.child_a.keep_adhesion { 1 } else { 0 })
            .collect();
        
        let child_b_flags_data: Vec<u32> = genome.modes.iter()
            .map(|mode| if mode.child_b.keep_adhesion { 1 } else { 0 })
            .collect();
        
        if !child_a_flags_data.is_empty() {
            queue.write_buffer(&self.child_a_keep_adhesion_flags, 0, bytemuck::cast_slice(&child_a_flags_data));
        }
        
        if !child_b_flags_data.is_empty() {
            queue.write_buffer(&self.child_b_keep_adhesion_flags, 0, bytemuck::cast_slice(&child_b_flags_data));
        }
        
        // Sync genome mode data (child orientations, split directions)
        let genome_mode_data: Vec<[f32; 12]> = genome.modes.iter().map(|mode| {
            let child_a_orientation = mode.child_a.orientation;
            let child_b_orientation = mode.child_b.orientation;
            let split_direction = mode.parent_split_direction;
            
            [
                child_a_orientation.x, child_a_orientation.y, child_a_orientation.z, child_a_orientation.w,
                child_b_orientation.x, child_b_orientation.y, child_b_orientation.z, child_b_orientation.w,
                split_direction.x, split_direction.y, 0.0, 0.0, // Vec2 -> Vec4 with padding
            ]
        }).collect();
        
        if !genome_mode_data.is_empty() {
            queue.write_buffer(&self.genome_mode_data, 0, bytemuck::cast_slice(&genome_mode_data));
        }
        
        self.needs_sync = false;
    }
    
    /// Increment reference count
    pub fn add_reference(&mut self) {
        self.reference_count += 1;
    }
    
    /// Decrement reference count, returns true if count reaches zero
    pub fn remove_reference(&mut self) -> bool {
        if self.reference_count > 0 {
            self.reference_count -= 1;
        }
        self.reference_count == 0
    }
    
    /// Mark this genome for deletion
    pub fn mark_for_deletion(&mut self) {
        self.marked_for_deletion = true;
    }
    
    /// Check if this genome can be safely deleted
    pub fn can_delete(&self) -> bool {
        self.marked_for_deletion && self.reference_count == 0
    }
}

/// Manages all genome buffer groups with reference counting and compaction
pub struct GenomeBufferManager {
    /// All genome buffer groups
    genome_groups: Vec<Option<GenomeBufferGroup>>,
    
    /// Free slots in the genome_groups array
    free_slots: Vec<usize>,
    
    /// Next genome ID to assign
    next_genome_id: usize,
    
    /// Maximum capacity
    max_genomes: usize,
}

impl GenomeBufferManager {
    /// Create a new genome buffer manager
    pub fn new(max_genomes: usize) -> Self {
        Self {
            genome_groups: (0..max_genomes).map(|_| None).collect(),
            free_slots: (0..max_genomes).collect(),
            next_genome_id: 0,
            max_genomes,
        }
    }
    
    /// Add a new genome buffer group
    /// Returns the genome ID or None if at capacity
    pub fn add_genome(
        &mut self,
        device: &wgpu::Device,
        genome: &Genome,
    ) -> Option<usize> {
        // Validate genome
        if genome.modes.is_empty() {
            log::warn!("Attempted to add genome with no modes");
            return None;
        }
        
        if genome.modes.len() > 1024 {
            log::warn!("Genome has too many modes: {} (max: 1024)", genome.modes.len());
            return None;
        }
        
        if self.free_slots.is_empty() {
            log::warn!("At maximum genome capacity: {}", self.max_genomes);
            return None; // At capacity
        }
        
        let slot = self.free_slots.remove(0);
        let genome_id = self.next_genome_id;
        self.next_genome_id += 1;
        
        // Validate genome ID doesn't exceed limits
        if genome_id >= self.max_genomes {
            log::error!("Genome ID {} exceeds maximum {}", genome_id, self.max_genomes);
            return None;
        }
        
        let buffer_group = GenomeBufferGroup::new(device, genome_id, genome);
        self.genome_groups[slot] = Some(buffer_group);
        
        log::info!("Added genome {} with {} modes (ID: {})", genome.name, genome.modes.len(), genome_id);
        Some(slot)
    }
    
    /// Get a mutable reference to a genome buffer group
    pub fn get_genome_group_mut(&mut self, genome_id: usize) -> Option<&mut GenomeBufferGroup> {
        self.genome_groups.iter_mut().find(|group| {
            group.as_ref().map_or(false, |g| g.genome_id == genome_id)
        }).and_then(|group| group.as_mut())
    }
    
    /// Get an immutable reference to a genome buffer group
    pub fn get_genome_group(&self, genome_id: usize) -> Option<&GenomeBufferGroup> {
        self.genome_groups.iter().find_map(|group| {
            group.as_ref().filter(|g| g.genome_id == genome_id)
        })
    }
    
    /// Remove a genome buffer group (marks for deletion, actual removal happens in compact)
    pub fn remove_genome(&mut self, genome_id: usize) -> bool {
        if let Some(group) = self.get_genome_group_mut(genome_id) {
            group.mark_for_deletion();
            true
        } else {
            false
        }
    }
    
    /// Increment reference count for a genome
    pub fn add_reference(&mut self, genome_id: usize) -> bool {
        if let Some(group) = self.get_genome_group_mut(genome_id) {
            group.add_reference();
            true
        } else {
            false
        }
    }
    
    /// Decrement reference count for a genome
    pub fn remove_reference(&mut self, genome_id: usize) -> bool {
        if let Some(group) = self.get_genome_group_mut(genome_id) {
            group.remove_reference()
        } else {
            false
        }
    }
    
    /// Compact genome buffer groups, removing those marked for deletion with zero references
    /// Returns the number of genomes removed
    pub fn compact(&mut self) -> usize {
        let mut removed_count = 0;
        
        for (slot, group) in self.genome_groups.iter_mut().enumerate() {
            if let Some(buffer_group) = group.as_ref() {
                if buffer_group.can_delete() {
                    // Remove this genome
                    *group = None;
                    self.free_slots.push(slot);
                    removed_count += 1;
                }
            }
        }
        
        // Sort free slots for deterministic allocation
        self.free_slots.sort_unstable();
        self.free_slots.dedup();
        
        removed_count
    }
    
    /// Get all active genome buffer groups
    pub fn active_genomes(&self) -> Vec<&GenomeBufferGroup> {
        self.genome_groups.iter()
            .filter_map(|group| group.as_ref())
            .filter(|group| !group.marked_for_deletion)
            .collect()
    }
    
    /// Sync all genomes that need synchronization
    pub fn sync_dirty_genomes(&mut self, queue: &wgpu::Queue, genomes: &[Genome]) {
        for group in self.genome_groups.iter_mut() {
            if let Some(buffer_group) = group.as_mut() {
                if buffer_group.needs_sync && !buffer_group.marked_for_deletion {
                    if let Some(genome) = genomes.get(buffer_group.genome_id) {
                        buffer_group.sync_from_genome(queue, genome);
                    }
                }
            }
        }
    }
    
    /// Get genome count
    pub fn genome_count(&self) -> usize {
        self.genome_groups.iter()
            .filter(|group| group.is_some())
            .count()
    }
    
    /// Check if at capacity
    pub fn at_capacity(&self) -> bool {
        self.free_slots.is_empty()
    }
}
