//! Adhesion Buffer System for GPU Physics
//!
//! Manages GPU buffers for the deterministic adhesion system.
//! Works alongside the triple buffer system for cell physics.
//!
//! ## Buffer Layout
//!
//! ### Adhesion Connections (96 bytes each)
//! - `adhesion_connections`: Array of GpuAdhesionConnection structs
//!
//! ### Per-Cell Adhesion Indices (80 bytes each = 20 * i32)
//! - `cell_adhesion_indices`: Array of 20 adhesion indices per cell (-1 = no connection)
//!
//! ### Adhesion Settings (48 bytes each)
//! - `adhesion_settings`: Per-mode adhesion settings from genome
//!
//! ### Adhesion Counts
//! - `adhesion_counts`: [total_count, live_count, free_top, padding]
//!
//! ### Free Slot Management
//! - `free_adhesion_slots`: Stack of free adhesion slot indices

use super::adhesion::{
    GpuAdhesionConnection, GpuAdhesionSettings, AdhesionCounts,
    CellAdhesionIndices, AdhesionSlotAllocator,
    MAX_ADHESIONS_PER_CELL, MAX_ADHESION_CONNECTIONS,
};

/// GPU buffers for the adhesion system
pub struct AdhesionBuffers {
    /// Adhesion connection data (96 bytes per connection)
    pub adhesion_connections: wgpu::Buffer,
    
    /// Per-cell adhesion indices (20 * i32 = 80 bytes per cell)
    pub cell_adhesion_indices: wgpu::Buffer,
    
    /// Per-mode adhesion settings (48 bytes per mode)
    pub adhesion_settings: wgpu::Buffer,
    
    /// Adhesion counts: [total, live, free_top, padding]
    pub adhesion_counts: wgpu::Buffer,
    
    /// Free adhesion slot stack
    pub free_adhesion_slots: wgpu::Buffer,
    
    /// Angular velocities for torque calculations (triple buffered)
    pub angular_velocities: [wgpu::Buffer; 3],
    
    /// Force accumulation buffer (for multi-adhesion cells)
    pub force_accum: wgpu::Buffer,
    
    /// Torque accumulation buffer (for multi-adhesion cells)
    pub torque_accum: wgpu::Buffer,
    
    /// Maximum adhesion connections
    pub max_connections: u32,
    
    /// Cell capacity (for per-cell buffers)
    pub cell_capacity: u32,
    
    /// Maximum modes (for settings buffer)
    pub max_modes: u32,
    
    /// CPU-side slot allocator for deterministic allocation
    slot_allocator: AdhesionSlotAllocator,
    
    /// CPU-side adhesion connection cache (for deterministic operations)
    connections_cache: Vec<GpuAdhesionConnection>,
    
    /// CPU-side per-cell adhesion indices cache
    cell_indices_cache: Vec<CellAdhesionIndices>,
    
    /// Whether buffers need sync to GPU
    needs_sync: bool,
}

impl AdhesionBuffers {
    /// Create new adhesion buffer system
    pub fn new(device: &wgpu::Device, cell_capacity: u32, max_modes: u32) -> Self {
        let max_connections = MAX_ADHESION_CONNECTIONS;
        
        // Adhesion connections: 96 bytes each
        let adhesion_connections = Self::create_storage_buffer(
            device,
            max_connections as u64 * 96,
            "Adhesion Connections",
        );
        
        // Per-cell adhesion indices: 20 * 4 = 80 bytes per cell
        let cell_adhesion_indices = Self::create_storage_buffer(
            device,
            cell_capacity as u64 * (MAX_ADHESIONS_PER_CELL as u64 * 4),
            "Cell Adhesion Indices",
        );
        
        // Per-mode adhesion settings: 48 bytes per mode
        let adhesion_settings = Self::create_storage_buffer(
            device,
            max_modes as u64 * 48,
            "Adhesion Settings",
        );
        
        // Adhesion counts: 4 * u32 = 16 bytes
        let adhesion_counts = Self::create_storage_buffer(
            device,
            16,
            "Adhesion Counts",
        );
        
        // Free adhesion slot stack
        let free_adhesion_slots = Self::create_storage_buffer(
            device,
            max_connections as u64 * 4,
            "Free Adhesion Slots",
        );
        
        // Angular velocities: Vec4 per cell, triple buffered
        let buffer_size = cell_capacity as u64 * 16;
        let angular_velocities = [
            Self::create_storage_buffer(device, buffer_size, "Angular Velocities 0"),
            Self::create_storage_buffer(device, buffer_size, "Angular Velocities 1"),
            Self::create_storage_buffer(device, buffer_size, "Angular Velocities 2"),
        ];
        
        // Force/torque accumulation buffers
        let force_accum = Self::create_storage_buffer(
            device,
            cell_capacity as u64 * 16,
            "Force Accumulation",
        );
        
        let torque_accum = Self::create_storage_buffer(
            device,
            cell_capacity as u64 * 16,
            "Torque Accumulation",
        );
        
        // Initialize CPU-side caches
        let connections_cache = vec![GpuAdhesionConnection::inactive(); max_connections as usize];
        let cell_indices_cache = vec![CellAdhesionIndices::default(); cell_capacity as usize];
        
        Self {
            adhesion_connections,
            cell_adhesion_indices,
            adhesion_settings,
            adhesion_counts,
            free_adhesion_slots,
            angular_velocities,
            force_accum,
            torque_accum,
            max_connections,
            cell_capacity,
            max_modes,
            slot_allocator: AdhesionSlotAllocator::new(max_connections),
            connections_cache,
            cell_indices_cache,
            needs_sync: true,
        }
    }
    
    /// Create a storage buffer with optimal settings
    fn create_storage_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    /// Initialize buffers with default values
    pub fn initialize(&mut self, queue: &wgpu::Queue) {
        // Clear adhesion counts
        let counts = AdhesionCounts {
            total_adhesion_count: 0,
            live_adhesion_count: 0,
            free_adhesion_top: 0,
            _padding: 0,
        };
        queue.write_buffer(&self.adhesion_counts, 0, bytemuck::bytes_of(&counts));
        
        // Initialize all cell adhesion indices to -1
        let empty_indices: Vec<i32> = vec![-1; self.cell_capacity as usize * MAX_ADHESIONS_PER_CELL];
        queue.write_buffer(&self.cell_adhesion_indices, 0, bytemuck::cast_slice(&empty_indices));
        
        // Reset CPU-side state
        self.slot_allocator.reset();
        for conn in &mut self.connections_cache {
            *conn = GpuAdhesionConnection::inactive();
        }
        for indices in &mut self.cell_indices_cache {
            *indices = CellAdhesionIndices::default();
        }
        
        self.needs_sync = false;
    }
    
    /// Sync adhesion settings from genomes to GPU
    pub fn sync_adhesion_settings(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut settings_data: Vec<GpuAdhesionSettings> = Vec::new();
        
        for genome in genomes {
            for mode in &genome.modes {
                let settings = GpuAdhesionSettings {
                    can_break: if mode.adhesion_settings.can_break { 1 } else { 0 },
                    break_force: mode.adhesion_settings.break_force,
                    rest_length: mode.adhesion_settings.rest_length,
                    linear_spring_stiffness: mode.adhesion_settings.linear_spring_stiffness,
                    linear_spring_damping: mode.adhesion_settings.linear_spring_damping,
                    orientation_spring_stiffness: mode.adhesion_settings.orientation_spring_stiffness,
                    orientation_spring_damping: mode.adhesion_settings.orientation_spring_damping,
                    max_angular_deviation: mode.adhesion_settings.max_angular_deviation,
                    twist_constraint_stiffness: mode.adhesion_settings.twist_constraint_stiffness,
                    twist_constraint_damping: mode.adhesion_settings.twist_constraint_damping,
                    enable_twist_constraint: if mode.adhesion_settings.enable_twist_constraint { 1 } else { 0 },
                    _padding: 0,
                };
                settings_data.push(settings);
            }
        }
        
        if !settings_data.is_empty() {
            queue.write_buffer(&self.adhesion_settings, 0, bytemuck::cast_slice(&settings_data));
        }
    }
    
    /// Create a sibling adhesion between two cells (deterministic)
    /// Returns the adhesion index if successful
    pub fn create_sibling_adhesion(
        &mut self,
        cell_a_index: u32,
        cell_b_index: u32,
        mode_index: u32,
        anchor_a: glam::Vec3,
        anchor_b: glam::Vec3,
        twist_ref_a: glam::Quat,
        twist_ref_b: glam::Quat,
    ) -> Option<u32> {
        // Allocate slot deterministically
        let slot = self.slot_allocator.allocate_slot()?;
        
        // Check if both cells have available adhesion slots
        let cell_a_idx = cell_a_index as usize;
        let cell_b_idx = cell_b_index as usize;
        
        if cell_a_idx >= self.cell_indices_cache.len() || cell_b_idx >= self.cell_indices_cache.len() {
            self.slot_allocator.free_slot(slot);
            return None;
        }
        
        // Find free slots in both cells
        let slot_a = self.cell_indices_cache[cell_a_idx].find_free_slot();
        let slot_b = self.cell_indices_cache[cell_b_idx].find_free_slot();
        
        if slot_a.is_none() || slot_b.is_none() {
            self.slot_allocator.free_slot(slot);
            return None;
        }
        
        // Create the connection
        let connection = GpuAdhesionConnection::new_sibling(
            cell_a_index,
            cell_b_index,
            mode_index,
            anchor_a,
            anchor_b,
            twist_ref_a,
            twist_ref_b,
        );
        
        // Update caches
        self.connections_cache[slot as usize] = connection;
        self.cell_indices_cache[cell_a_idx].add_adhesion(slot);
        self.cell_indices_cache[cell_b_idx].add_adhesion(slot);
        
        self.needs_sync = true;
        
        Some(slot)
    }
    
    /// Remove an adhesion connection
    pub fn remove_adhesion(&mut self, adhesion_index: u32) {
        let idx = adhesion_index as usize;
        if idx >= self.connections_cache.len() {
            return;
        }
        
        let connection = &self.connections_cache[idx];
        if connection.is_active == 0 {
            return;
        }
        
        // Remove from both cells
        let cell_a = connection.cell_a_index as usize;
        let cell_b = connection.cell_b_index as usize;
        
        if cell_a < self.cell_indices_cache.len() {
            self.cell_indices_cache[cell_a].remove_adhesion(adhesion_index);
        }
        if cell_b < self.cell_indices_cache.len() {
            self.cell_indices_cache[cell_b].remove_adhesion(adhesion_index);
        }
        
        // Mark connection as inactive
        self.connections_cache[idx].is_active = 0;
        
        // Return slot to free list
        self.slot_allocator.free_slot(adhesion_index);
        
        self.needs_sync = true;
    }
    
    /// Sync CPU caches to GPU buffers
    pub fn sync_to_gpu(&mut self, queue: &wgpu::Queue) {
        if !self.needs_sync {
            return;
        }
        
        // Sync adhesion connections
        let allocated = self.slot_allocator.allocated_count() as usize;
        if allocated > 0 {
            let data = &self.connections_cache[..allocated.min(self.connections_cache.len())];
            queue.write_buffer(&self.adhesion_connections, 0, bytemuck::cast_slice(data));
        }
        
        // Sync cell adhesion indices
        let indices_data: Vec<i32> = self.cell_indices_cache
            .iter()
            .flat_map(|c| c.indices.iter().copied())
            .collect();
        queue.write_buffer(&self.cell_adhesion_indices, 0, bytemuck::cast_slice(&indices_data));
        
        // Sync adhesion counts
        let counts = AdhesionCounts {
            total_adhesion_count: self.slot_allocator.allocated_count(),
            live_adhesion_count: self.count_active_adhesions(),
            free_adhesion_top: 0, // Not using GPU-side free stack
            _padding: 0,
        };
        queue.write_buffer(&self.adhesion_counts, 0, bytemuck::bytes_of(&counts));
        
        self.needs_sync = false;
    }
    
    /// Count active adhesions
    fn count_active_adhesions(&self) -> u32 {
        self.connections_cache
            .iter()
            .filter(|c| c.is_active == 1)
            .count() as u32
    }
    
    /// Get adhesion count
    pub fn adhesion_count(&self) -> u32 {
        self.count_active_adhesions()
    }
    
    /// Reset all adhesions
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.slot_allocator.reset();
        for conn in &mut self.connections_cache {
            *conn = GpuAdhesionConnection::inactive();
        }
        for indices in &mut self.cell_indices_cache {
            *indices = CellAdhesionIndices::default();
        }
        self.initialize(queue);
    }
    
    /// Get adhesions for a specific cell
    pub fn get_cell_adhesions(&self, cell_index: u32) -> Vec<u32> {
        let idx = cell_index as usize;
        if idx >= self.cell_indices_cache.len() {
            return Vec::new();
        }
        
        self.cell_indices_cache[idx]
            .indices
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| i as u32)
            .collect()
    }
    
    /// Get connection data for an adhesion
    pub fn get_connection(&self, adhesion_index: u32) -> Option<&GpuAdhesionConnection> {
        let idx = adhesion_index as usize;
        if idx < self.connections_cache.len() && self.connections_cache[idx].is_active == 1 {
            Some(&self.connections_cache[idx])
        } else {
            None
        }
    }
}
