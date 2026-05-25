//! Adhesion Buffer System for GPU Physics
//!
//! Manages GPU buffers for the deterministic adhesion system.
//! Works alongside the triple buffer system for cell physics.
//!
//! ## Buffer Layout
//!
//! ### Adhesion Connections (104 bytes each with WGSL padding)
//! - `adhesion_connections`: Array of GpuAdhesionConnection structs
//!
//! ### Per-Cell Adhesion Indices (40 bytes each = 10 * i32)
//! - `cell_adhesion_indices`: Array of 10 adhesion indices per cell (-1 = no connection)
//!
//! ### Adhesion Settings (split into 3 × 16-byte sub-buffers per mode)
//! - `adhesion_settings_v0`: [can_break, break_force, rest_length, linear_spring_stiffness]
//! - `adhesion_settings_v1`: [linear_spring_damping, orientation_spring_stiffness, orientation_spring_damping, max_angular_deviation]
//! - `adhesion_settings_v2`: [twist_constraint_stiffness, twist_constraint_damping, enable_twist_constraint, _padding]
//!
//! ### Adhesion Counts
//! - `adhesion_counts`: [total_count, live_count, free_top, padding]
//!
//! ### Free Slot Management
//! - `free_adhesion_slots`: Stack of free adhesion slot indices

use super::adhesion::{
    GpuAdhesionConnection, AdhesionCounts,
    CellAdhesionIndices, AdhesionSlotAllocator,
    MAX_ADHESIONS_PER_CELL,
};

/// GPU buffers for the adhesion system
pub struct AdhesionBuffers {
    /// Adhesion connection data (96 bytes per connection)
    pub adhesion_connections: wgpu::Buffer,
    
    /// Per-cell adhesion indices (20 * i32 = 80 bytes per cell)
    pub cell_adhesion_indices: wgpu::Buffer,
    
    /// Per-mode adhesion settings split into 3 × vec4 sub-buffers (16 bytes each).
    /// Split to stay under wgpu's 256 MB/buffer limit at 8M modes × 16 bytes = 128 MB each.
    /// v0: [can_break, break_force, rest_length, linear_spring_stiffness]
    /// v1: [linear_spring_damping, orientation_spring_stiffness, orientation_spring_damping, max_angular_deviation]
    /// v2: [twist_constraint_stiffness, twist_constraint_damping, enable_twist_constraint, _padding]
    pub adhesion_settings_v0: wgpu::Buffer,
    pub adhesion_settings_v1: wgpu::Buffer,
    pub adhesion_settings_v2: wgpu::Buffer,
    
    /// Adhesion counts: [total, live, free_top, padding]
    pub adhesion_counts: wgpu::Buffer,
    
    /// Free adhesion slot stack
    pub free_adhesion_slots: wgpu::Buffer,
    
    /// Next adhesion ID counter (atomic u32 for GPU-side allocation)
    pub next_adhesion_id: wgpu::Buffer,
    
    /// Angular velocities for torque calculations (triple buffered)
    pub angular_velocities: [wgpu::Buffer; 3],
    
    /// Force accumulation X component (atomic i32, fixed-point scaled)
    pub force_accum_x: wgpu::Buffer,
    
    /// Force accumulation Y component (atomic i32, fixed-point scaled)
    pub force_accum_y: wgpu::Buffer,
    
    /// Force accumulation Z component (atomic i32, fixed-point scaled)
    pub force_accum_z: wgpu::Buffer,
    
    /// Torque accumulation X component (atomic i32, fixed-point scaled)
    pub torque_accum_x: wgpu::Buffer,
    
    /// Torque accumulation Y component (atomic i32, fixed-point scaled)
    pub torque_accum_y: wgpu::Buffer,
    
    /// Torque accumulation Z component (atomic i32, fixed-point scaled)
    pub torque_accum_z: wgpu::Buffer,
    
    /// Maximum adhesion connections
    pub max_connections: u32,
    
    /// Cell capacity (for per-cell buffers)
    pub cell_capacity: u32,
    
    /// CPU-side slot allocator for deterministic allocation
    slot_allocator: AdhesionSlotAllocator,
    
    /// CPU-side adhesion connection cache (for deterministic operations)
    connections_cache: Vec<GpuAdhesionConnection>,
    
    /// CPU-side per-cell adhesion indices cache
    cell_indices_cache: Vec<CellAdhesionIndices>,
    
    /// Whether buffers need sync to GPU
    needs_sync: bool,

    /// Per-cell signal channels: 16 u32 per cell (channels 0-7 oculocyte, 8-15 regulation)
    /// Each u32 encodes: bits 16+ = direction flag, bits 11-15 = hops, bits 0-10 = signal value
    /// Size: cell_capacity * 16 * 4 bytes
    pub signal_flags: wgpu::Buffer,

    /// Second signal flags buffer for double-buffered propagation.
    /// During propagation: read from signal_flags, write to signal_flags_next, then swap.
    pub signal_flags_next: wgpu::Buffer,

    /// Current allocated size of the adhesion settings mode pool (in number of modes).
    /// Starts at 16K and doubles on demand up to MAX_TOTAL_MODES.
    pub adhesion_mode_pool_capacity: u64,
}

impl AdhesionBuffers {
    /// Create new adhesion buffer system
    pub fn new(device: &wgpu::Device, cell_capacity: u32) -> Self {
        // Each connection is shared by 2 cells, so theoretical max = cells * max_per_cell / 2
        let max_connections = cell_capacity * (MAX_ADHESIONS_PER_CELL as u32) / 2;
        
        // Adhesion connections: 104 bytes each (WGSL struct with implicit padding)
        let adhesion_connections = Self::create_storage_buffer(
            device,
            max_connections as u64 * 104,
            "Adhesion Connections",
        );
        
        // Per-cell adhesion indices: 20 * 4 = 80 bytes per cell
        let cell_adhesion_indices = Self::create_storage_buffer(
            device,
            cell_capacity as u64 * (MAX_ADHESIONS_PER_CELL as u64 * 4),
            "Cell Adhesion Indices",
        );
        
        // Per-mode adhesion settings split into 3 × vec4 sub-buffers (16 bytes each).
        // Start with the same initial pool size as triple_buffer.rs (16K modes).
        // grow_adhesion_mode_pool_if_needed() doubles the pool on demand up to MAX_TOTAL_MODES.
        const INITIAL_MODE_POOL_SIZE: u64 = 16_384;
        let adhesion_settings_v0 = Self::create_storage_buffer(
            device, INITIAL_MODE_POOL_SIZE * 16, "Adhesion Settings V0",
        );
        let adhesion_settings_v1 = Self::create_storage_buffer(
            device, INITIAL_MODE_POOL_SIZE * 16, "Adhesion Settings V1",
        );
        let adhesion_settings_v2 = Self::create_storage_buffer(
            device, INITIAL_MODE_POOL_SIZE * 16, "Adhesion Settings V2",
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
        
        // Next adhesion ID counter (atomic u32 for GPU-side allocation)
        // Use 16 bytes to meet wgpu minimum storage buffer size requirements
        let next_adhesion_id = Self::create_storage_buffer(
            device,
            16, // 4 u32s for alignment (only first one used)
            "Next Adhesion ID",
        );
        
        // Angular velocities: Vec4 per cell, triple buffered
        let buffer_size = cell_capacity as u64 * 16;
        let angular_velocities = [
            Self::create_storage_buffer(device, buffer_size, "Angular Velocities 0"),
            Self::create_storage_buffer(device, buffer_size, "Angular Velocities 1"),
            Self::create_storage_buffer(device, buffer_size, "Angular Velocities 2"),
        ];
        
        // Force/torque accumulation buffers (atomic i32 per component)
        // Each buffer is cell_capacity * 4 bytes (one i32 per cell)
        let atomic_buffer_size = cell_capacity as u64 * 4;
        let force_accum_x = Self::create_storage_buffer(device, atomic_buffer_size, "Force Accum X");
        let force_accum_y = Self::create_storage_buffer(device, atomic_buffer_size, "Force Accum Y");
        let force_accum_z = Self::create_storage_buffer(device, atomic_buffer_size, "Force Accum Z");
        let torque_accum_x = Self::create_storage_buffer(device, atomic_buffer_size, "Torque Accum X");
        let torque_accum_y = Self::create_storage_buffer(device, atomic_buffer_size, "Torque Accum Y");
        let torque_accum_z = Self::create_storage_buffer(device, atomic_buffer_size, "Torque Accum Z");
        
        // Per-cell signal flags: 16 channels per cell, 1 u32 per channel = 64 bytes per cell
        let signal_flags = Self::create_storage_buffer(
            device,
            cell_capacity as u64 * 16 * 4,
            "Signal Flags (16 channels)",
        );

        // Second signal flags buffer for double-buffered propagation
        let signal_flags_next = Self::create_storage_buffer(
            device,
            cell_capacity as u64 * 16 * 4,
            "Signal Flags Next (16 channels)",
        );

        // Initialize CPU-side caches
        let connections_cache = vec![GpuAdhesionConnection::inactive(); max_connections as usize];
        let cell_indices_cache = vec![CellAdhesionIndices::default(); cell_capacity as usize];
        
        Self {
            adhesion_connections,
            cell_adhesion_indices,
            adhesion_settings_v0,
            adhesion_settings_v1,
            adhesion_settings_v2,
            adhesion_counts,
            free_adhesion_slots,
            next_adhesion_id,
            angular_velocities,
            force_accum_x,
            force_accum_y,
            force_accum_z,
            torque_accum_x,
            torque_accum_y,
            torque_accum_z,
            max_connections,
            cell_capacity,
            slot_allocator: AdhesionSlotAllocator::new(max_connections),
            connections_cache,
            cell_indices_cache,
            needs_sync: true,
            signal_flags,
            signal_flags_next,
            adhesion_mode_pool_capacity: INITIAL_MODE_POOL_SIZE,
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
        
        // Initialize next_adhesion_id to 0
        let zero: [u32; 1] = [0];
        queue.write_buffer(&self.next_adhesion_id, 0, bytemuck::cast_slice(&zero));
        
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
    
    /// Grow the adhesion settings mode pool if `total_modes` exceeds the current capacity.
    ///
    /// Must be called before `sync_adhesion_settings` when new genomes are added.
    /// Returns `true` if the pool was grown (bind groups referencing these buffers must be rebuilt).
    pub fn grow_adhesion_mode_pool_if_needed(&mut self, device: &wgpu::Device, total_modes: u64) -> bool {
        use crate::simulation::gpu_physics::mutation::MAX_TOTAL_MODES;
        let hard_cap = MAX_TOTAL_MODES as u64;

        if total_modes <= self.adhesion_mode_pool_capacity {
            return false;
        }

        let mut new_capacity = self.adhesion_mode_pool_capacity;
        while new_capacity < total_modes {
            new_capacity = (new_capacity * 2).min(hard_cap);
        }

        log::info!(
            "Growing adhesion mode pool: {} → {} modes ({:.1} MB → {:.1} MB)",
            self.adhesion_mode_pool_capacity,
            new_capacity,
            self.adhesion_mode_pool_capacity as f64 * 3.0 * 16.0 / 1_048_576.0,
            new_capacity as f64 * 3.0 * 16.0 / 1_048_576.0,
        );

        let m16 = new_capacity * 16;
        self.adhesion_settings_v0 = Self::create_storage_buffer(device, m16, "Adhesion Settings V0");
        self.adhesion_settings_v1 = Self::create_storage_buffer(device, m16, "Adhesion Settings V1");
        self.adhesion_settings_v2 = Self::create_storage_buffer(device, m16, "Adhesion Settings V2");
        self.adhesion_mode_pool_capacity = new_capacity;
        true
    }

    /// Sync adhesion settings from genomes to GPU
    pub fn sync_adhesion_settings(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut v0: Vec<[f32; 4]> = Vec::new();
        let mut v1: Vec<[f32; 4]> = Vec::new();
        let mut v2: Vec<[f32; 4]> = Vec::new();
        
        for genome in genomes {
            for mode in &genome.modes {
                let s = &mode.adhesion_settings;
                v0.push([
                    if s.can_break { 1.0_f32 } else { 0.0 },
                    s.break_force,
                    s.rest_length,
                    s.linear_spring_stiffness,
                ]);
                v1.push([
                    s.linear_spring_damping,
                    s.orientation_spring_stiffness,
                    s.orientation_spring_damping,
                    s.max_angular_deviation,
                ]);
                v2.push([
                    s.twist_constraint_stiffness,
                    s.twist_constraint_damping,
                    if s.enable_twist_constraint { 1.0 } else { 0.0 },
                    0.0,
                ]);
            }
        }
        
        if !v0.is_empty() {
            queue.write_buffer(&self.adhesion_settings_v0, 0, bytemuck::cast_slice(&v0));
            queue.write_buffer(&self.adhesion_settings_v1, 0, bytemuck::cast_slice(&v1));
            queue.write_buffer(&self.adhesion_settings_v2, 0, bytemuck::cast_slice(&v2));
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
        birth_time: f32,
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
            birth_time,
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
    
    // ── Snapshot support ──────────────────────────────────────────────────────

    /// Return a clone of the CPU-side connections cache for snapshot serialisation.
    ///
    /// This is the authoritative CPU mirror of the adhesion GPU buffer and is
    /// always kept in sync, so no GPU readback is required.
    pub fn snapshot_connections(&self) -> Vec<GpuAdhesionConnection> {
        self.connections_cache.clone()
    }

    /// Return a clone of the CPU-side per-cell adhesion index cache for snapshot
    /// serialisation.
    pub fn snapshot_cell_indices(&self) -> Vec<CellAdhesionIndices> {
        self.cell_indices_cache.clone()
    }

    /// Return the number of adhesion slots that have been allocated (including
    /// inactive/freed ones up to the high-water mark).
    pub fn snapshot_allocated_count(&self) -> u32 {
        self.slot_allocator.allocated_count()
    }

    /// Restore adhesion state from snapshot data.
    ///
    /// Replaces the CPU caches with the provided data and marks the buffers as
    /// needing a GPU sync.  Call `sync_to_gpu` afterwards to push the data to
    /// the GPU.
    ///
    /// # Panics
    /// Panics if `connections` is longer than the buffer capacity or if
    /// `cell_indices` is longer than the cell capacity.
    pub fn restore_from_snapshot(
        &mut self,
        connections: Vec<GpuAdhesionConnection>,
        cell_indices: Vec<CellAdhesionIndices>,
        allocated_count: u32,
    ) {
        assert!(
            connections.len() <= self.connections_cache.len(),
            "snapshot adhesion connections ({}) exceed buffer capacity ({})",
            connections.len(),
            self.connections_cache.len(),
        );
        assert!(
            cell_indices.len() <= self.cell_indices_cache.len(),
            "snapshot cell adhesion indices ({}) exceed cell capacity ({})",
            cell_indices.len(),
            self.cell_indices_cache.len(),
        );

        // Reset allocator to match the snapshot's allocation state.
        self.slot_allocator.reset();
        // Advance the allocator's high-water mark to match the snapshot.
        // Slots that are inactive in the connections cache are treated as freed.
        for _ in 0..allocated_count {
            // Advance allocated_count by allocating each slot in order.
            let _ = self.slot_allocator.allocate_slot();
        }
        // Free any slots that are inactive in the snapshot (they were freed
        // before the snapshot was taken).
        for (slot, conn) in connections.iter().enumerate().take(allocated_count as usize) {
            if conn.is_active == 0 {
                self.slot_allocator.free_slot(slot as u32);
            }
        }

        // Copy connections into the cache (zero-fill the rest).
        let n = connections.len();
        self.connections_cache[..n].copy_from_slice(&connections);
        for entry in &mut self.connections_cache[n..] {
            *entry = GpuAdhesionConnection::inactive();
        }

        // Copy cell indices into the cache (default-fill the rest).
        let m = cell_indices.len();
        self.cell_indices_cache[..m].copy_from_slice(&cell_indices);
        for entry in &mut self.cell_indices_cache[m..] {
            *entry = CellAdhesionIndices::default();
        }

        self.needs_sync = true;
    }

    /// DEBUG: Synchronous readback of adhesion counts from GPU
    /// This is slow and should only be used for debugging!
    pub fn debug_sync_readback_adhesion_counts(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // Create a staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adhesion Counts Staging"),
            size: 16, // 4 u32s
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create encoder and copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Adhesion Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.adhesion_counts, 0, &staging_buffer, 0, 16);
        queue.submit(std::iter::once(encoder.finish()));
        
        // Map the staging buffer and read the data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&data);
            log::debug!("[GPU ADHESION] total={}, live={}, free_top={}", 
                counts[0], counts[1], counts[2]);
            drop(data);
        }
        
        staging_buffer.unmap();
    }
    
    /// DEBUG: Synchronous readback of first N adhesion connections from GPU
    pub fn debug_sync_readback_connections(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: u32,
    ) {
        let count = count.min(10); // Limit to 10 for readability
        let size = count as u64 * 104; // 104 bytes per connection
        
        // Create a staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adhesion Connections Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create encoder and copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Connections Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.adhesion_connections, 0, &staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        
        // Map the staging buffer and read the data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            
            let connections: &[GpuAdhesionConnection] = bytemuck::cast_slice(&data);
            for (i, conn) in connections.iter().enumerate() {
                log::debug!("[GPU CONN {}] cell_a={}, cell_b={}, mode={}, active={}", 
                    i, conn.cell_a_index, conn.cell_b_index, conn.mode_index, conn.is_active);
            }
            drop(data);
        }
        
        staging_buffer.unmap();
    }

}
