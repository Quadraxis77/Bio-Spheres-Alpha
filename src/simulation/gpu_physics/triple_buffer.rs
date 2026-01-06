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
    
    /// Cell rotations: Vec4(x, y, z, w) quaternion - triple buffered
    pub rotations: [wgpu::Buffer; 3],
    
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
    
    /// Max cell sizes per cell (from genome mode) - caps mass growth
    pub max_cell_sizes: wgpu::Buffer,
    
    /// Membrane stiffnesses per cell (from genome mode)
    /// Used for collision repulsion strength
    pub stiffnesses: wgpu::Buffer,
    
    /// GPU-side cell count buffer: [0] = total cells, [1] = live cells
    /// Updated by division shader, read by all other shaders
    pub cell_count_buffer: wgpu::Buffer,
    
    /// Staging buffer for async cell count readback
    cell_count_staging: wgpu::Buffer,
    
    /// Genome mode data buffer for division (child orientations, split direction)
    /// Layout per mode: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)]
    /// Total size: num_modes * 48 bytes
    pub genome_mode_data: wgpu::Buffer,
    
    /// Current buffer index (atomic for lock-free rotation)
    current_index: AtomicUsize,
    
    /// Cell capacity
    pub capacity: u32,
    
    /// Whether buffers need full sync from canonical state (initial setup only)
    pub needs_sync: bool,
    
    /// Pending cell insertions (cell indices to sync)
    pending_cell_insertions: Vec<usize>,
    
    /// Last known GPU cell count (updated asynchronously)
    gpu_cell_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
    
    /// Whether an async cell count read is in progress (copy issued, waiting for map)
    cell_count_read_pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    
    /// Whether map_async has been called for the current read
    cell_count_map_started: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
        
        let rotations = [
            Self::create_storage_buffer(device, buffer_size, "Rotations Buffer 0"),
            Self::create_storage_buffer(device, buffer_size, "Rotations Buffer 1"),
            Self::create_storage_buffer(device, buffer_size, "Rotations Buffer 2"),
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
        let max_cell_sizes = Self::create_storage_buffer(device, f32_per_cell, "Max Cell Sizes");
        let stiffnesses = Self::create_storage_buffer(device, f32_per_cell, "Stiffnesses");
        
        // GPU-side cell count: [0] = total cells, [1] = live cells
        let cell_count_buffer = Self::create_storage_buffer(device, 8, "Cell Count Buffer");
        
        // Staging buffer for async cell count readback (MAP_READ)
        let cell_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Count Staging Buffer"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Genome mode data: 40 modes per genome * 10 genomes max = 400 modes
        // Each mode: child_a_orientation (16 bytes) + child_b_orientation (16 bytes) + split_direction (16 bytes) = 48 bytes
        let max_modes = 40 * 10;
        let genome_mode_data = Self::create_storage_buffer(device, max_modes * 48, "Genome Mode Data");
        
        Self {
            position_and_mass,
            velocity,
            rotations,
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
            max_cell_sizes,
            stiffnesses,
            cell_count_buffer,
            cell_count_staging,
            genome_mode_data,
            current_index: AtomicUsize::new(0),
            capacity,
            needs_sync: true,
            pending_cell_insertions: Vec::new(),
            gpu_cell_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            cell_count_read_pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            cell_count_map_started: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
        // Always update GPU cell count buffer, even when cell_count is 0
        let cell_counts: [u32; 2] = [state.cell_count as u32, state.cell_count as u32];
        queue.write_buffer(&self.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        
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
        
        // Build rotation data: Vec4(x, y, z, w) quaternion
        let mut rotation_data: Vec<[f32; 4]> = Vec::with_capacity(state.cell_count);
        for i in 0..state.cell_count {
            let q = state.rotations[i];
            rotation_data.push([q.x, q.y, q.z, q.w]);
        }
        
        // Upload to all three buffer sets
        let position_bytes = bytemuck::cast_slice(&position_mass_data);
        let velocity_bytes = bytemuck::cast_slice(&velocity_data);
        let rotation_bytes = bytemuck::cast_slice(&rotation_data);
        
        for i in 0..3 {
            queue.write_buffer(&self.position_and_mass[i], 0, position_bytes);
            queue.write_buffer(&self.velocity[i], 0, velocity_bytes);
            queue.write_buffer(&self.rotations[i], 0, rotation_bytes);
        }
        
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
        
        // Max splits, nutrient gain rates, max cell sizes, and stiffnesses (from genome modes)
        // Convert -1 (infinite) to 0 (unlimited in GPU)
        let mut max_splits_data: Vec<u32> = Vec::with_capacity(state.cell_count);
        let mut nutrient_gain_rates_data: Vec<f32> = Vec::with_capacity(state.cell_count);
        let mut max_cell_sizes_data: Vec<f32> = Vec::with_capacity(state.cell_count);
        let mut stiffnesses_data: Vec<f32> = Vec::with_capacity(state.cell_count);
        
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
                    max_cell_sizes_data.push(mode.max_cell_size);
                    stiffnesses_data.push(mode.membrane_stiffness);
                } else {
                    max_splits_data.push(0); // Unlimited if mode not found
                    nutrient_gain_rates_data.push(0.2); // Default nutrient gain rate
                    max_cell_sizes_data.push(2.0); // Default max cell size
                    stiffnesses_data.push(50.0); // Default membrane stiffness
                }
            } else {
                max_splits_data.push(0); // Unlimited if genome not found
                nutrient_gain_rates_data.push(0.2); // Default nutrient gain rate
                max_cell_sizes_data.push(2.0); // Default max cell size
                stiffnesses_data.push(50.0); // Default membrane stiffness
            }
        }
        queue.write_buffer(&self.max_splits, 0, bytemuck::cast_slice(&max_splits_data));
        queue.write_buffer(&self.nutrient_gain_rates, 0, bytemuck::cast_slice(&nutrient_gain_rates_data));
        queue.write_buffer(&self.max_cell_sizes, 0, bytemuck::cast_slice(&max_cell_sizes_data));
        queue.write_buffer(&self.stiffnesses, 0, bytemuck::cast_slice(&stiffnesses_data));
        
        // Genome IDs
        let genome_ids: Vec<u32> = state.genome_ids[..state.cell_count].iter().map(|&x| x as u32).collect();
        queue.write_buffer(&self.genome_ids, 0, bytemuck::cast_slice(&genome_ids));
        
        // Mode indices - store ABSOLUTE indices (with genome offset applied)
        // This matches how the instance builder's mode_visuals buffer is organized:
        // mode_visuals[0..genome0.modes.len()] = genome 0's modes
        // mode_visuals[genome0.modes.len()..] = genome 1's modes, etc.
        let genome_mode_offsets: Vec<usize> = {
            let mut offsets = Vec::with_capacity(genomes.len());
            let mut offset = 0usize;
            for genome in genomes {
                offsets.push(offset);
                offset += genome.modes.len();
            }
            offsets
        };
        
        let mode_indices: Vec<u32> = (0..state.cell_count)
            .map(|i| {
                let genome_id = state.genome_ids[i];
                let local_mode_idx = state.mode_indices[i];
                let offset = genome_mode_offsets.get(genome_id).copied().unwrap_or(0);
                (offset + local_mode_idx) as u32
            })
            .collect();
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
        
        // Sync genome mode data for division (child orientations)
        self.sync_genome_mode_data(queue, genomes);
    }
    
    /// Sync genome mode data (child orientations) for division shader
    pub fn sync_genome_mode_data(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        // Layout per mode: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)] = 48 bytes
        let mut mode_data: Vec<[f32; 12]> = Vec::new();
        
        for genome in genomes {
            for mode in &genome.modes {
                let qa = mode.child_a.orientation;
                let qb = mode.child_b.orientation;
                
                // Calculate split direction from pitch/yaw (same as preview scene)
                let pitch = mode.parent_split_direction.x.to_radians();
                let yaw = mode.parent_split_direction.y.to_radians();
                let split_dir = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0) * glam::Vec3::Z;
                
                mode_data.push([
                    qa.x, qa.y, qa.z, qa.w,
                    qb.x, qb.y, qb.z, qb.w,
                    split_dir.x, split_dir.y, split_dir.z, 0.0,
                ]);
            }
        }
        
        if !mode_data.is_empty() {
            queue.write_buffer(&self.genome_mode_data, 0, bytemuck::cast_slice(&mode_data));
        }
    }
    
    /// Sync a single cell to all GPU buffer sets (for cell insertion during simulation)
    pub fn sync_single_cell(&self, queue: &wgpu::Queue, cell_idx: usize, position: glam::Vec3, velocity: glam::Vec3, mass: f32, rotation: glam::Quat) {
        let position_mass: [f32; 4] = [position.x, position.y, position.z, mass];
        let velocity_data: [f32; 4] = [velocity.x, velocity.y, velocity.z, 0.0];
        let rotation_data: [f32; 4] = [rotation.x, rotation.y, rotation.z, rotation.w];
        
        let offset = (cell_idx * 16) as u64; // Vec4<f32> = 16 bytes
        
        // Upload to all three buffer sets
        for i in 0..3 {
            queue.write_buffer(&self.position_and_mass[i], offset, bytemuck::bytes_of(&position_mass));
            queue.write_buffer(&self.velocity[i], offset, bytemuck::bytes_of(&velocity_data));
            queue.write_buffer(&self.rotations[i], offset, bytemuck::bytes_of(&rotation_data));
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
        max_cell_size: f32,
        stiffness: f32,
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
        queue.write_buffer(&self.max_cell_sizes, offset, bytemuck::bytes_of(&max_cell_size));
        queue.write_buffer(&self.stiffnesses, offset, bytemuck::bytes_of(&stiffness));
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
        // Pre-calculate genome mode offsets for absolute mode index calculation
        let genome_mode_offsets: Vec<usize> = {
            let mut offsets = Vec::with_capacity(genomes.len());
            let mut offset = 0usize;
            for genome in genomes {
                offsets.push(offset);
                offset += genome.modes.len();
            }
            offsets
        };
        
        for &cell_idx in &self.pending_cell_insertions {
            if cell_idx < state.cell_count {
                let position = state.positions[cell_idx];
                let velocity = state.velocities[cell_idx];
                let mass = state.masses[cell_idx];
                let rotation = state.rotations[cell_idx];
                self.sync_single_cell(queue, cell_idx, position, velocity, mass, rotation);
                
                // Get max_splits, nutrient_gain_rate, max_cell_size, and stiffness from genome mode
                let genome_id = state.genome_ids[cell_idx];
                let local_mode_idx = state.mode_indices[cell_idx];
                let (max_splits, nutrient_gain_rate, max_cell_size, stiffness) = if genome_id < genomes.len() {
                    let genome = &genomes[genome_id];
                    if local_mode_idx < genome.modes.len() {
                        let mode = &genome.modes[local_mode_idx];
                        (mode.max_splits, mode.nutrient_gain_rate, mode.max_cell_size, mode.membrane_stiffness)
                    } else {
                        (-1, 0.2, 2.0, 50.0) // Defaults if mode not found
                    }
                } else {
                    (-1, 0.2, 2.0, 50.0) // Defaults if genome not found
                };
                
                // Calculate absolute mode index (with genome offset)
                let absolute_mode_idx = genome_mode_offsets.get(genome_id).copied().unwrap_or(0) + local_mode_idx;
                
                // Also sync cell state for division system
                self.sync_single_cell_state(
                    queue,
                    cell_idx,
                    state.birth_times[cell_idx],
                    state.split_intervals[cell_idx],
                    state.split_masses[cell_idx],
                    state.genome_ids[cell_idx],
                    absolute_mode_idx, // Use absolute mode index
                    state.cell_ids[cell_idx] as usize,
                    max_splits,
                    nutrient_gain_rate,
                    max_cell_size,
                    stiffness,
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
    
    /// Start an async read of the GPU cell count.
    /// Call this once per frame, then use `gpu_cell_count()` to get the latest value.
    /// The read is non-blocking - it copies to a staging buffer and maps it asynchronously.
    pub fn start_async_cell_count_read(&self, encoder: &mut wgpu::CommandEncoder) {
        use std::sync::atomic::Ordering;
        
        // Don't start a new read if one is already pending OR if buffer is still mapped
        // map_started stays true until we unmap in try_read_mapped_cell_count
        if self.cell_count_read_pending.load(Ordering::Acquire) 
            || self.cell_count_map_started.load(Ordering::Acquire) {
            return;
        }
        
        // Copy from GPU buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.cell_count_buffer,
            0,
            &self.cell_count_staging,
            0,
            8, // 2 x u32
        );
        
        // Mark as pending (map_started will be set when we call map_async)
        self.cell_count_read_pending.store(true, Ordering::Release);
    }
    
    /// Poll for async cell count read completion.
    /// Call this after queue.submit() to check if the read is ready.
    pub fn poll_async_cell_count_read(&self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        
        // Only poll if a read is pending and map hasn't started yet
        if !self.cell_count_read_pending.load(Ordering::Acquire) {
            return;
        }
        
        // Only call map_async once per read cycle
        if self.cell_count_map_started.load(Ordering::Acquire) {
            // Already started map, just poll the device
            let _ = device.poll(wgpu::PollType::Poll);
            return;
        }
        
        // Mark that we're starting the map (this stays true until unmap)
        self.cell_count_map_started.store(true, Ordering::Release);
        
        let staging_slice = self.cell_count_staging.slice(..);
        let read_pending = self.cell_count_read_pending.clone();
        
        staging_slice.map_async(wgpu::MapMode::Read, move |_result| {
            // Mark read as no longer pending - the map is complete
            // Note: map_started stays true until we unmap in try_read_mapped_cell_count
            read_pending.store(false, Ordering::Release);
        });
        
        // Poll the device to process the map request (non-blocking)
        let _ = device.poll(wgpu::PollType::Poll);
    }
    
    /// Check if the staging buffer is mapped and read the cell count.
    /// Returns true if a new value was read.
    pub fn try_read_mapped_cell_count(&self) -> bool {
        use std::sync::atomic::Ordering;
        
        // If read is still pending, the map callback hasn't fired yet
        if self.cell_count_read_pending.load(Ordering::Acquire) {
            return false;
        }
        
        // If map hasn't been started, there's nothing to read
        if !self.cell_count_map_started.load(Ordering::Acquire) {
            return false;
        }
        
        // Try to get the mapped range - this will work if map_async completed successfully
        // We wrap in catch_unwind since get_mapped_range panics if not mapped
        let staging_slice = self.cell_count_staging.slice(..);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let view = staging_slice.get_mapped_range();
            let data: &[u32] = bytemuck::cast_slice(&view);
            let count = if !data.is_empty() { data[0] } else { 0 };
            drop(view);
            count
        }));
        
        if let Ok(count) = result {
            self.gpu_cell_count.store(count, Ordering::Release);
            self.cell_count_staging.unmap();
            // Reset map_started AFTER unmapping - this allows new reads to start
            self.cell_count_map_started.store(false, Ordering::Release);
            return true;
        }
        
        false
    }
    
    /// Get the last known GPU cell count (updated asynchronously).
    /// This value may be 1-2 frames behind the actual GPU state.
    pub fn gpu_cell_count(&self) -> u32 {
        self.gpu_cell_count.load(std::sync::atomic::Ordering::Acquire)
    }
}