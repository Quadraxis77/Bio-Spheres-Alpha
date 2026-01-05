//! Triple Buffer System for GPU Physics
//! 
//! Manages three complete sets of simulation buffers to enable lock-free
//! GPU computation with zero CPU synchronization stalls.
//! 
//! ## Buffer Layout
//! 
//! ### Cell Data (Triple Buffered)
//! - `position_and_mass[3]`: Vec4(x, y, z, mass) per cell
//! - `velocity[3]`: Vec4(x, y, z, 0) per cell
//! 
//! ### Spatial Grid (64³ = 262,144 grid cells)
//! - `spatial_grid_counts`: u32 per grid cell (how many cells in each grid cell)
//! - `spatial_grid_offsets`: u32 per grid cell (prefix-sum for O(1) lookup)
//! - `spatial_grid_cells`: u32 per cell (sorted cell indices by grid cell)
//! - `cell_grid_indices`: u32 per cell (which grid cell each cell belongs to)
//! 
//! ### Uniforms
//! - `physics_params`: 256-byte aligned PhysicsParams struct
//! 
//! ## Synchronization
//! - `needs_sync`: Flag set when CPU data changes (cell insertion/removal)
//! - `sync_from_canonical_state()`: Uploads CPU data to all 3 buffer sets
//! - `output_buffer_index()`: Returns buffer with latest physics results

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::simulation::CanonicalState;

/// Triple-buffered GPU simulation state using Structure-of-Arrays layout
pub struct GpuTripleBufferSystem {
    /// Cell positions and masses: Vec4(x, y, z, mass) - triple buffered
    pub position_and_mass: [wgpu::Buffer; 3],
    
    /// Cell velocities: Vec4(x, y, z, padding) - triple buffered  
    pub velocity: [wgpu::Buffer; 3],
    
    /// Spatial grid cell counts (64³ = 262,144 grid cells)
    /// spatial_grid_counts[grid_idx] = number of cells in that grid cell
    pub spatial_grid_counts: wgpu::Buffer,
    
    /// Spatial grid offsets (prefix sum results)
    /// spatial_grid_offsets[grid_idx] = starting index in spatial_grid_cells
    pub spatial_grid_offsets: wgpu::Buffer,
    
    /// Sorted cell indices by grid cell
    /// spatial_grid_cells[offset..offset+count] = cell indices in grid cell
    pub spatial_grid_cells: wgpu::Buffer,
    
    /// Per-cell grid indices
    /// cell_grid_indices[cell_idx] = which grid cell this cell belongs to
    pub cell_grid_indices: wgpu::Buffer,
    
    /// Physics parameters uniform buffer
    pub physics_params: wgpu::Buffer,
    
    // === Lifecycle buffers for cell division ===
    
    /// Death flags: 1 = dead (free slot), 0 = alive
    pub death_flags: wgpu::Buffer,
    
    /// Division flags: 1 = wants to divide, 0 = not dividing
    pub division_flags: wgpu::Buffer,
    
    /// Compacted free slot indices (result of prefix-sum on death_flags)
    pub free_slot_indices: wgpu::Buffer,
    
    /// Division slot assignments: maps dividing cell to its assigned free slot index
    pub division_slot_assignments: wgpu::Buffer,
    
    /// Lifecycle counts: [0] = free slot count, [1] = division count
    pub lifecycle_counts: wgpu::Buffer,
    
    // === Cell state buffers for division ===
    
    /// Birth times for each cell
    pub birth_times: wgpu::Buffer,
    
    /// Split intervals (time between divisions)
    pub split_intervals: wgpu::Buffer,
    
    /// Split mass thresholds
    pub split_masses: wgpu::Buffer,
    
    /// Current split counts
    pub split_counts: wgpu::Buffer,
    
    /// Maximum splits allowed (0 = unlimited)
    pub max_splits: wgpu::Buffer,
    
    /// Genome IDs for each cell
    pub genome_ids: wgpu::Buffer,
    
    /// Mode indices for each cell
    pub mode_indices: wgpu::Buffer,
    
    /// Cell IDs (unique identifier)
    pub cell_ids: wgpu::Buffer,
    
    /// Next cell ID counter
    pub next_cell_id: wgpu::Buffer,
    
    /// Nutrient gain rates per cell (from genome mode)
    pub nutrient_gain_rates: wgpu::Buffer,
    
    /// GPU-side cell count buffer: [0] = total cells, [1] = live cells
    /// Updated by division shader, read by all other shaders
    pub cell_count_buffer: wgpu::Buffer,
    
    /// Current buffer index (atomic for lock-free rotation)
    current_index: AtomicUsize,
    
    /// Cell capacity
    pub capacity: u32,
    
    /// Whether buffers need full sync from canonical state (initial setup only)
    pub needs_sync: bool,
    
    /// Pending cell insertions (cell indices to sync)
    pending_cell_insertions: Vec<usize>,
}

impl GpuTripleBufferSystem {
    /// Create a new triple buffer system
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let buffer_size = capacity as u64 * 16; // Vec4<f32> = 16 bytes
        
        // Create triple-buffered simulation data
        let position_and_mass = [
            Self::create_storage_buffer(device, buffer_size, "Position Mass Buffer 0"),
            Self::create_storage_buffer(device, buffer_size, "Position Mass Buffer 1"), 
            Self::create_storage_buffer(device, buffer_size, "Position Mass Buffer 2"),
        ];
        
        let velocity = [
            Self::create_storage_buffer(device, buffer_size, "Velocity Buffer 0"),
            Self::create_storage_buffer(device, buffer_size, "Velocity Buffer 1"),
            Self::create_storage_buffer(device, buffer_size, "Velocity Buffer 2"),
        ];
        
        // Create spatial grid buffers (64³ = 262,144 grid cells)
        let grid_size = 64 * 64 * 64;
        let spatial_grid_counts = Self::create_storage_buffer(
            device, 
            grid_size * 4, // u32 = 4 bytes
            "Spatial Grid Counts"
        );
        
        let spatial_grid_offsets = Self::create_storage_buffer(
            device,
            grid_size * 4, // u32 = 4 bytes  
            "Spatial Grid Offsets"
        );
        
        let cell_grid_indices = Self::create_storage_buffer(
            device,
            capacity as u64 * 4, // u32 per cell
            "Cell Grid Indices"
        );
        
        // Sorted cell indices by grid cell (16 cells max per grid cell * 64³ grid cells)
        // This enables O(1) neighbor lookup in collision detection
        let spatial_grid_cells = Self::create_storage_buffer(
            device,
            16 * grid_size * 4, // 16 cells per grid cell * 262,144 grid cells * 4 bytes
            "Spatial Grid Cells"
        );
        
        // Physics params uniform buffer (256 bytes aligned)
        let physics_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Physics Params Uniform"),
            size: 256, // Aligned to 256 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // === Lifecycle buffers for cell division ===
        let u32_per_cell = capacity as u64 * 4; // u32 = 4 bytes
        let f32_per_cell = capacity as u64 * 4; // f32 = 4 bytes
        
        let death_flags = Self::create_storage_buffer(device, u32_per_cell, "Death Flags");
        let division_flags = Self::create_storage_buffer(device, u32_per_cell, "Division Flags");
        let free_slot_indices = Self::create_storage_buffer(device, u32_per_cell, "Free Slot Indices");
        let division_slot_assignments = Self::create_storage_buffer(device, u32_per_cell, "Division Slot Assignments");
        
        // Lifecycle counts: [0] = free slots, [1] = divisions, [2] = dead count
        let lifecycle_counts = Self::create_storage_buffer(device, 12, "Lifecycle Counts");
        
        // Cell state buffers
        let birth_times = Self::create_storage_buffer(device, f32_per_cell, "Birth Times");
        let split_intervals = Self::create_storage_buffer(device, f32_per_cell, "Split Intervals");
        let split_masses = Self::create_storage_buffer(device, f32_per_cell, "Split Masses");
        let split_counts = Self::create_storage_buffer(device, u32_per_cell, "Split Counts");
        let max_splits = Self::create_storage_buffer(device, u32_per_cell, "Max Splits");
        let genome_ids = Self::create_storage_buffer(device, u32_per_cell, "Genome IDs");
        let mode_indices = Self::create_storage_buffer(device, u32_per_cell, "Mode Indices");
        let cell_ids = Self::create_storage_buffer(device, u32_per_cell, "Cell IDs");
        let next_cell_id = Self::create_storage_buffer(device, 4, "Next Cell ID");
        let nutrient_gain_rates = Self::create_storage_buffer(device, f32_per_cell, "Nutrient Gain Rates");
        
        // GPU-side cell count: [0] = total cells, [1] = live cells
        let cell_count_buffer = Self::create_storage_buffer(device, 8, "Cell Count Buffer");
        
        Self {
            position_and_mass,
            velocity,
            spatial_grid_counts,
            spatial_grid_offsets,
            spatial_grid_cells,
            cell_grid_indices,
            physics_params,
            death_flags,
            division_flags,
            free_slot_indices,
            division_slot_assignments,
            lifecycle_counts,
            birth_times,
            split_intervals,
            split_masses,
            split_counts,
            max_splits,
            genome_ids,
            mode_indices,
            cell_ids,
            next_cell_id,
            nutrient_gain_rates,
            cell_count_buffer,
            current_index: AtomicUsize::new(0),
            capacity,
            needs_sync: true,
            pending_cell_insertions: Vec::new(),
        }
    }
    
    /// Create a storage buffer with optimal settings for compute shaders
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
    
    /// Upload canonical state data to all GPU buffer sets
    /// WARNING: This overwrites ALL GPU positions with CPU data.
    /// Only use for initial setup, not during simulation.
    pub fn sync_from_canonical_state(&mut self, queue: &wgpu::Queue, state: &CanonicalState, genomes: &[crate::genome::Genome]) {
        if state.cell_count == 0 {
            self.needs_sync = false;
            return;
        }
        
        // Build position_and_mass data: Vec4(x, y, z, mass)
        let mut position_mass_data: Vec<[f32; 4]> = Vec::with_capacity(state.cell_count);
        for i in 0..state.cell_count {
            position_mass_data.push([
                state.positions[i].x,
                state.positions[i].y,
                state.positions[i].z,
                state.masses[i],
            ]);
        }
        
        // Build velocity data: Vec4(x, y, z, 0)
        let mut velocity_data: Vec<[f32; 4]> = Vec::with_capacity(state.cell_count);
        for i in 0..state.cell_count {
            velocity_data.push([
                state.velocities[i].x,
                state.velocities[i].y,
                state.velocities[i].z,
                0.0,
            ]);
        }
        
        // Upload to all three buffer sets
        let position_bytes = bytemuck::cast_slice(&position_mass_data);
        let velocity_bytes = bytemuck::cast_slice(&velocity_data);
        
        for i in 0..3 {
            queue.write_buffer(&self.position_and_mass[i], 0, position_bytes);
            queue.write_buffer(&self.velocity[i], 0, velocity_bytes);
        }
        
        // Initialize GPU cell count buffer: [total_cells, live_cells]
        let cell_counts: [u32; 2] = [state.cell_count as u32, state.cell_count as u32];
        queue.write_buffer(&self.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        
        // Sync cell state buffers for division system
        self.sync_cell_state_buffers(queue, state, genomes);
        
        self.needs_sync = false;
    }
    
    /// Sync cell state buffers (birth times, split intervals, etc.) for division system
    pub fn sync_cell_state_buffers(&self, queue: &wgpu::Queue, state: &CanonicalState, genomes: &[crate::genome::Genome]) {
        if state.cell_count == 0 {
            return;
        }
        
        // Birth times
        let birth_times: Vec<f32> = state.birth_times[..state.cell_count].to_vec();
        queue.write_buffer(&self.birth_times, 0, bytemuck::cast_slice(&birth_times));
        
        // Split intervals
        let split_intervals: Vec<f32> = state.split_intervals[..state.cell_count].to_vec();
        queue.write_buffer(&self.split_intervals, 0, bytemuck::cast_slice(&split_intervals));
        
        // Split masses
        let split_masses: Vec<f32> = state.split_masses[..state.cell_count].to_vec();
        queue.write_buffer(&self.split_masses, 0, bytemuck::cast_slice(&split_masses));
        
        // Split counts
        let split_counts: Vec<u32> = state.split_counts[..state.cell_count].iter().map(|&x| x as u32).collect();
        queue.write_buffer(&self.split_counts, 0, bytemuck::cast_slice(&split_counts));
        
        // Max splits and nutrient gain rates (from genome modes)
        // Convert -1 (infinite) to 0 (unlimited in GPU)
        let mut max_splits_data: Vec<u32> = Vec::with_capacity(state.cell_count);
        let mut nutrient_gain_rates_data: Vec<f32> = Vec::with_capacity(state.cell_count);
        
        for i in 0..state.cell_count {
            let genome_id = state.genome_ids[i];
            let mode_idx = state.mode_indices[i];
            if genome_id < genomes.len() {
                let genome = &genomes[genome_id];
                if mode_idx < genome.modes.len() {
                    let mode = &genome.modes[mode_idx];
                    let ms = mode.max_splits;
                    max_splits_data.push(if ms < 0 { 0 } else { ms as u32 });
                    nutrient_gain_rates_data.push(mode.nutrient_gain_rate);
                } else {
                    max_splits_data.push(0); // Unlimited if mode not found
                    nutrient_gain_rates_data.push(0.2); // Default nutrient gain rate
                }
            } else {
                max_splits_data.push(0); // Unlimited if genome not found
                nutrient_gain_rates_data.push(0.2); // Default nutrient gain rate
            }
        }
        queue.write_buffer(&self.max_splits, 0, bytemuck::cast_slice(&max_splits_data));
        queue.write_buffer(&self.nutrient_gain_rates, 0, bytemuck::cast_slice(&nutrient_gain_rates_data));
        
        // Genome IDs
        let genome_ids: Vec<u32> = state.genome_ids[..state.cell_count].iter().map(|&x| x as u32).collect();
        queue.write_buffer(&self.genome_ids, 0, bytemuck::cast_slice(&genome_ids));
        
        // Mode indices
        let mode_indices: Vec<u32> = state.mode_indices[..state.cell_count].iter().map(|&x| x as u32).collect();
        queue.write_buffer(&self.mode_indices, 0, bytemuck::cast_slice(&mode_indices));
        
        // Cell IDs
        let cell_ids: Vec<u32> = state.cell_ids[..state.cell_count].iter().map(|&x| x as u32).collect();
        queue.write_buffer(&self.cell_ids, 0, bytemuck::cast_slice(&cell_ids));
        
        // Next cell ID
        let next_id: [u32; 1] = [state.next_cell_id as u32];
        queue.write_buffer(&self.next_cell_id, 0, bytemuck::cast_slice(&next_id));
        
        // Initialize death flags to 0 (all alive)
        let death_flags: Vec<u32> = vec![0u32; state.cell_count];
        queue.write_buffer(&self.death_flags, 0, bytemuck::cast_slice(&death_flags));
        
        // Initialize division flags to 0
        let division_flags: Vec<u32> = vec![0u32; state.cell_count];
        queue.write_buffer(&self.division_flags, 0, bytemuck::cast_slice(&division_flags));
    }
    
    /// Sync a single cell to all GPU buffer sets (for cell insertion during simulation)
    pub fn sync_single_cell(&self, queue: &wgpu::Queue, cell_idx: usize, position: glam::Vec3, velocity: glam::Vec3, mass: f32) {
        let position_mass: [f32; 4] = [position.x, position.y, position.z, mass];
        let velocity_data: [f32; 4] = [velocity.x, velocity.y, velocity.z, 0.0];
        
        let offset = (cell_idx * 16) as u64; // Vec4<f32> = 16 bytes
        
        // Upload to all three buffer sets
        for i in 0..3 {
            queue.write_buffer(&self.position_and_mass[i], offset, bytemuck::bytes_of(&position_mass));
            queue.write_buffer(&self.velocity[i], offset, bytemuck::bytes_of(&velocity_data));
        }
    }
    
    /// Sync a single cell's state data (for cell insertion during simulation)
    pub fn sync_single_cell_state(
        &self,
        queue: &wgpu::Queue,
        cell_idx: usize,
        birth_time: f32,
        split_interval: f32,
        split_mass: f32,
        genome_id: usize,
        mode_index: usize,
        cell_id: usize,
        max_splits: i32,
        nutrient_gain_rate: f32,
    ) {
        let offset = (cell_idx * 4) as u64; // f32/u32 = 4 bytes
        
        queue.write_buffer(&self.birth_times, offset, bytemuck::bytes_of(&birth_time));
        queue.write_buffer(&self.split_intervals, offset, bytemuck::bytes_of(&split_interval));
        queue.write_buffer(&self.split_masses, offset, bytemuck::bytes_of(&split_mass));
        queue.write_buffer(&self.split_counts, offset, bytemuck::bytes_of(&0u32));
        
        // Convert max_splits: -1 (infinite) -> 0 (unlimited in GPU), positive values stay as-is
        let gpu_max_splits: u32 = if max_splits < 0 { 0 } else { max_splits as u32 };
        queue.write_buffer(&self.max_splits, offset, bytemuck::bytes_of(&gpu_max_splits));
        
        queue.write_buffer(&self.nutrient_gain_rates, offset, bytemuck::bytes_of(&nutrient_gain_rate));
        queue.write_buffer(&self.genome_ids, offset, bytemuck::bytes_of(&(genome_id as u32)));
        queue.write_buffer(&self.mode_indices, offset, bytemuck::bytes_of(&(mode_index as u32)));
        queue.write_buffer(&self.cell_ids, offset, bytemuck::bytes_of(&(cell_id as u32)));
        queue.write_buffer(&self.death_flags, offset, bytemuck::bytes_of(&0u32)); // Alive
        queue.write_buffer(&self.division_flags, offset, bytemuck::bytes_of(&0u32)); // Not dividing
    }
    
    /// Mark that buffers need to be synced from canonical state
    pub fn mark_needs_sync(&mut self) {
        self.needs_sync = true;
    }
    
    /// Queue a cell insertion to be synced on next physics step
    pub fn queue_cell_insertion(&mut self, cell_idx: usize) {
        self.pending_cell_insertions.push(cell_idx);
    }
    
    /// Sync pending cell insertions to GPU buffers
    pub fn sync_pending_insertions(&mut self, queue: &wgpu::Queue, state: &crate::simulation::CanonicalState, genomes: &[crate::genome::Genome]) {
        for &cell_idx in &self.pending_cell_insertions {
            if cell_idx < state.cell_count {
                let position = state.positions[cell_idx];
                let velocity = state.velocities[cell_idx];
                let mass = state.masses[cell_idx];
                self.sync_single_cell(queue, cell_idx, position, velocity, mass);
                
                // Get max_splits and nutrient_gain_rate from genome mode
                let genome_id = state.genome_ids[cell_idx];
                let mode_idx = state.mode_indices[cell_idx];
                let (max_splits, nutrient_gain_rate) = if genome_id < genomes.len() {
                    let genome = &genomes[genome_id];
                    if mode_idx < genome.modes.len() {
                        let mode = &genome.modes[mode_idx];
                        (mode.max_splits, mode.nutrient_gain_rate)
                    } else {
                        (-1, 0.2) // Defaults if mode not found
                    }
                } else {
                    (-1, 0.2) // Defaults if genome not found
                };
                
                // Also sync cell state for division system
                self.sync_single_cell_state(
                    queue,
                    cell_idx,
                    state.birth_times[cell_idx],
                    state.split_intervals[cell_idx],
                    state.split_masses[cell_idx],
                    state.genome_ids[cell_idx],
                    state.mode_indices[cell_idx],
                    state.cell_ids[cell_idx] as usize,
                    max_splits,
                    nutrient_gain_rate,
                );
            }
        }
        
        // Update GPU cell count buffer after insertions
        if !self.pending_cell_insertions.is_empty() {
            let cell_counts: [u32; 2] = [state.cell_count as u32, state.cell_count as u32];
            queue.write_buffer(&self.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        }
        
        self.pending_cell_insertions.clear();
    }
    
    /// Rotate to the next buffer set (lock-free)
    pub fn rotate_buffers(&self) -> usize {
        let current = self.current_index.load(Ordering::Acquire);
        let next = (current + 1) % 3;
        self.current_index.store(next, Ordering::Release);
        next
    }
    
    /// Get the current buffer index
    pub fn current_index(&self) -> usize {
        self.current_index.load(Ordering::Acquire)
    }
    
    /// Get the output buffer index (where physics results are written)
    /// This is (current_index + 1) % 3 since physics reads from current and writes to next
    pub fn output_buffer_index(&self) -> usize {
        let current = self.current_index.load(Ordering::Acquire);
        (current + 1) % 3
    }
}