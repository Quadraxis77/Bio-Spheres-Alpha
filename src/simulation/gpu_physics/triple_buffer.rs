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
//! ### Spatial Grid (128³ = 2,097,152 grid cells)
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

/// Deterministic slot allocation system for cell addition
/// 
/// Maintains a sorted list of free slots and allocates them in deterministic order.
/// Uses the same prefix-sum compaction pattern as the division system.
#[derive(Debug, Clone)]
pub struct DeterministicSlotAllocator {
    /// Free slot indices (sorted for deterministic allocation)
    free_slots: Vec<u32>,
    /// Total capacity
    capacity: u32,
    /// Next cell ID to assign
    next_cell_id: u32,
}

impl DeterministicSlotAllocator {
    /// Create new slot allocator with given capacity
    pub fn new(capacity: u32) -> Self {
        // Initialize with all slots free (0..capacity)
        let free_slots: Vec<u32> = (0..capacity).collect();
        
        Self {
            free_slots,
            capacity,
            next_cell_id: 1, // Start from 1 (0 reserved for invalid)
        }
    }
    
    /// Allocate next available slot deterministically
    /// Returns (slot_index, cell_id) or None if no slots available
    pub fn allocate_slot(&mut self) -> Option<(u32, u32)> {
        if self.free_slots.is_empty() {
            return None; // No free slots
        }
        
        // Always take the first (lowest) slot for deterministic allocation
        let slot = self.free_slots.remove(0);
        let cell_id = self.next_cell_id;
        
        self.next_cell_id += 1;
        
        Some((slot, cell_id))
    }
    
    /// Free a slot (mark for reuse)
    pub fn free_slot(&mut self, slot: u32) {
        if slot < self.capacity && !self.free_slots.contains(&slot) {
            // Add to free slots list (will be sorted during compaction)
            self.free_slots.push(slot);
        }
    }
    
    /// Compact free slots using deterministic sorting (like prefix-sum)
    /// This ensures consistent allocation order across runs
    pub fn compact_free_slots(&mut self) {
        // Sort all free slots for deterministic order
        self.free_slots.sort_unstable();
        
        // Remove duplicates while maintaining order
        self.free_slots.dedup();
    }
    
    /// Get current allocated count
    pub fn allocated_count(&self) -> u32 {
        self.capacity - self.free_slots.len() as u32
    }
    
    /// Get available slot count
    pub fn available_slots(&self) -> usize {
        self.free_slots.len()
    }
    
    /// Reset allocator to initial state
    pub fn reset(&mut self) {
        self.free_slots = (0..self.capacity).collect();
        self.next_cell_id = 1;
    }
    
    /// Set the next cell ID (for synchronization with canonical state)
    pub fn set_next_cell_id(&mut self, next_id: u32) {
        self.next_cell_id = next_id;
    }
}

/// Cell addition request for deterministic processing
#[derive(Debug, Clone)]
pub struct CellAdditionRequest {
    pub position: glam::Vec3,
    pub velocity: glam::Vec3,
    pub mass: f32,
    pub rotation: glam::Quat,
    pub genome_id: usize,
    pub mode_index: usize,
    pub birth_time: f32,
    pub split_interval: f32,
    pub split_mass: f32,
    pub stiffness: f32,
    pub radius: f32,
    pub genome_orientation: glam::Quat,
    pub angular_velocity: glam::Vec3,
}

impl CellAdditionRequest {
    /// Create deterministic hash for sorting
    pub fn deterministic_hash(&self) -> u64 {
        // Use position and genome info for deterministic ordering
        let x = (self.position.x * 1000.0) as i64;
        let y = (self.position.y * 1000.0) as i64;
        let z = (self.position.z * 1000.0) as i64;
        let g = self.genome_id as u64;
        let m = self.mode_index as u64;
        
        // Combine into deterministic hash
        ((x as u64) << 32) ^ ((y as u64) << 16) ^ (z as u64) ^ (g << 8) ^ m
    }
}

/// Deterministic cell addition pipeline
/// 
/// Processes cell additions in batches with deterministic ordering to ensure
/// the same input produces the same output across runs.
#[derive(Debug)]
pub struct DeterministicCellAddition {
    /// Pending addition requests
    pending_requests: Vec<CellAdditionRequest>,
}

impl DeterministicCellAddition {
    /// Create new cell addition pipeline
    pub fn new() -> Self {
        Self {
            pending_requests: Vec::new(),
        }
    }
    
    /// Queue a cell for addition
    pub fn queue_cell_addition(&mut self, request: CellAdditionRequest) {
        self.pending_requests.push(request);
    }
    
    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }
    
    /// Clear all pending requests
    pub fn clear_pending(&mut self) {
        self.pending_requests.clear();
    }
}

/// Triple-buffered GPU simulation state using Structure-of-Arrays layout
pub struct GpuTripleBufferSystem {
    /// Cell positions and masses: Vec4(x, y, z, mass) - triple buffered
    pub position_and_mass: [wgpu::Buffer; 3],
    
    /// Cell velocities: Vec4(x, y, z, padding) - triple buffered  
    pub velocity: [wgpu::Buffer; 3],
    
    /// Cell rotations: Vec4(x, y, z, w) quaternion - triple buffered
    pub rotations: [wgpu::Buffer; 3],
    
    /// Previous frame accelerations for Verlet integration: Vec4(x, y, z, padding)
    /// Used to compute velocity_change = 0.5 * (old_accel + new_accel) * dt
    pub prev_accelerations: wgpu::Buffer,
    
    /// Spatial grid cell counts (128³ = 2,097,152 grid cells)
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
    
    /// Mass deltas for nutrient transport (accumulates mass changes)
    pub mass_deltas_buffer: wgpu::Buffer,
    
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
    
    /// Split ready frame tracking (for nutrient transfer delay)
    /// -1 means cell is not ready to split
    /// >= 0 means cell is ready and this is the frame number when it became ready
    pub split_ready_frame: wgpu::Buffer,
    
    /// Maximum splits allowed (0 = unlimited)
    pub max_splits: wgpu::Buffer,
    
    /// Genome IDs for each cell
    pub genome_ids: wgpu::Buffer,
    
    /// Cell types for each cell (0 = Test, 1 = Flagellocyte, etc.)
    /// Derived from mode settings during cell insertion
    pub cell_types: wgpu::Buffer,
    
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
    

    
    /// Genome mode data buffer for division (child orientations, split direction)
    /// Layout per mode: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)]
    /// Total size: num_modes * 48 bytes
    pub genome_mode_data: wgpu::Buffer,
    
    /// Child mode indices buffer (two i32 per mode: child_a_mode, child_b_mode)
    /// Total size: num_modes * 8 bytes
    pub child_mode_indices: wgpu::Buffer,
    
    /// Parent make adhesion flags buffer (one u32 per mode)
    /// Stores whether each mode allows sibling adhesion creation during division
    pub parent_make_adhesion_flags: wgpu::Buffer,
    
    /// Child A keep adhesion flags buffer (one u32 per mode)
    /// Stores whether Child A should inherit adhesions during division
    pub child_a_keep_adhesion_flags: wgpu::Buffer,
    
    /// Child B keep adhesion flags buffer (one u32 per mode)
    /// Stores whether Child B should inherit adhesions during division
    pub child_b_keep_adhesion_flags: wgpu::Buffer,
    
    /// Mode properties buffer for division (per-mode properties)
    /// Layout per mode: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, padding x3] = 48 bytes
    pub mode_properties: wgpu::Buffer,
    
    /// Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
    /// Used by shaders to derive cell_type from mode_index (always up-to-date with genome settings)
    pub mode_cell_types: wgpu::Buffer,
    
    /// Current buffer index (atomic for lock-free rotation)
    current_index: AtomicUsize,
    
    /// Cell capacity
    pub capacity: u32,
    
    /// Whether buffers need full sync from canonical state (initial setup only)
    pub needs_sync: bool,
    
    /// Pending cell insertions (cell indices to sync)
    pending_cell_insertions: Vec<usize>,
    

    
    /// Deterministic slot allocation system
    slot_allocator: DeterministicSlotAllocator,
    
    /// Deterministic cell addition pipeline
    cell_addition_pipeline: DeterministicCellAddition,
    
    /// Cell count readback buffer for async GPU-to-CPU transfer
    cell_count_readback_buffer: wgpu::Buffer,
    
    /// Whether a cell count readback is pending
    cell_count_map_pending: bool,
    
    /// Channel receiver for cell count map completion
    cell_count_receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    
    /// Last known cell count from GPU readback
    last_cell_count: u32,
    
    /// Behavior flags per cell type for parameterized shader logic
    pub behavior_flags: wgpu::Buffer,
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
        
        // Previous accelerations for Verlet integration (single buffer, not triple buffered)
        let prev_accelerations = Self::create_storage_buffer(device, buffer_size, "Previous Accelerations");
        
        // Create spatial grid buffers (128³ = 2,097,152 grid cells)
        let grid_size = 128 * 128 * 128;
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
        
        // Sorted cell indices by grid cell (16 cells max per grid cell * 128³ grid cells)
        // This enables O(1) neighbor lookup in collision detection
        let spatial_grid_cells = Self::create_storage_buffer(
            device,
            16 * grid_size * 4, // 16 cells per grid cell * 2,097,152 grid cells * 4 bytes
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
        let mass_deltas_buffer = Self::create_storage_buffer(device, f32_per_cell, "Mass Deltas Buffer"); // i32 = 4 bytes, same as f32
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
        let split_ready_frame = Self::create_storage_buffer(device, u32_per_cell, "Split Ready Frame");
        let max_splits = Self::create_storage_buffer(device, u32_per_cell, "Max Splits");
        let genome_ids = Self::create_storage_buffer(device, u32_per_cell, "Genome IDs");
        let cell_types = Self::create_storage_buffer(device, u32_per_cell, "Cell Types");
        let mode_indices = Self::create_storage_buffer(device, u32_per_cell, "Mode Indices");
        let cell_ids = Self::create_storage_buffer(device, u32_per_cell, "Cell IDs");
        let next_cell_id = Self::create_storage_buffer(device, 4, "Next Cell ID");
        let nutrient_gain_rates = Self::create_storage_buffer(device, f32_per_cell, "Nutrient Gain Rates");
        let max_cell_sizes = Self::create_storage_buffer(device, f32_per_cell, "Max Cell Sizes");
        let stiffnesses = Self::create_storage_buffer(device, f32_per_cell, "Stiffnesses");
        
        // GPU-side cell count: [0] = total cells, [1] = live cells
        let cell_count_buffer = Self::create_storage_buffer(device, 8, "Cell Count Buffer");
        
        // Cell count readback buffer for async GPU-to-CPU transfer
        let cell_count_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Count Readback Buffer"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        
        // Genome mode data: 40 modes per genome * 20,000 genomes max = 800,000 modes
        // Each mode: child_a_orientation (16 bytes) + child_b_orientation (16 bytes) + split_direction (16 bytes) = 48 bytes
        // Increased to support 20,000 genomes
        let max_modes = 40 * 20_000; // Support for 20,000 genomes
        let genome_mode_data = Self::create_storage_buffer(device, max_modes * 48, "Genome Mode Data");
        
        // Child mode indices: two i32 per mode (child_a_mode, child_b_mode)
        let child_mode_indices = Self::create_storage_buffer(device, max_modes * 8, "Child Mode Indices");
        
        // Parent make adhesion flags: one u32 per mode
        let parent_make_adhesion_flags = Self::create_storage_buffer(device, max_modes * 4, "Parent Make Adhesion Flags");
        
        // Child A keep adhesion flags: one u32 per mode
        let child_a_keep_adhesion_flags = Self::create_storage_buffer(device, max_modes * 4, "Child A Keep Adhesion Flags");
        
        // Child B keep adhesion flags: one u32 per mode
        let child_b_keep_adhesion_flags = Self::create_storage_buffer(device, max_modes * 4, "Child B Keep Adhesion Flags");
        
        // Mode properties: 12 floats per mode (48 bytes) - nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, padding x3
        let mode_properties = Self::create_storage_buffer(device, max_modes * 48, "Mode Properties");
        
        // Mode cell types: one u32 per mode - lookup table for deriving cell_type from mode_index
        let mode_cell_types = Self::create_storage_buffer(device, max_modes * 4, "Mode Cell Types");
        
        // Behavior flags per cell type for parameterized shader logic
        // Each GpuCellTypeBehaviorFlags struct is 64 bytes (6 u32 fields + 10 u32 padding)
        let behavior_flags = Self::create_storage_buffer(device, 30 * 64, "Behavior Flags"); // CellType::MAX_TYPES = 30
        
        Self {
            position_and_mass,
            velocity,
            rotations,
            prev_accelerations,
            spatial_grid_counts,
            spatial_grid_offsets,
            spatial_grid_cells,
            cell_grid_indices,
            physics_params,
            death_flags,
            mass_deltas_buffer,
            division_flags,
            free_slot_indices,
            division_slot_assignments,
            lifecycle_counts,
            birth_times,
            split_intervals,
            split_masses,
            split_counts,
            split_ready_frame,
            max_splits,
            genome_ids,
            cell_types,
            mode_indices,
            cell_ids,
            next_cell_id,
            nutrient_gain_rates,
            max_cell_sizes,
            stiffnesses,
            cell_count_buffer,
            genome_mode_data,
            child_mode_indices,
            parent_make_adhesion_flags,
            child_a_keep_adhesion_flags,
            child_b_keep_adhesion_flags,
            mode_properties,
            mode_cell_types,
            behavior_flags,
            current_index: AtomicUsize::new(0),
            capacity,
            needs_sync: true,
            pending_cell_insertions: Vec::new(),
            slot_allocator: DeterministicSlotAllocator::new(capacity),
            cell_addition_pipeline: DeterministicCellAddition::new(),
            cell_count_readback_buffer,
            cell_count_map_pending: false,
            cell_count_receiver: None,
            last_cell_count: 0,
        }
    }
    
    /// Create a storage buffer with optimal settings for compute shaders
    fn create_storage_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        // Align size to 16-byte boundary for GPU compatibility
        let aligned_size = (size + 15) & !15; // Round up to nearest 16 bytes
        
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned_size,
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
        // Sync slot allocator with canonical state
        self.sync_slot_allocator_with_canonical(state);
        
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
        
        // Split ready frame - initialize to -1 (not ready to split)
        let split_ready_frame: Vec<i32> = vec![-1; state.cell_count];
        queue.write_buffer(&self.split_ready_frame, 0, bytemuck::cast_slice(&split_ready_frame));
        
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
                    // Flagellocytes (cell_type == 1) don't generate their own nutrients
                    let nutrient_rate = if mode.cell_type == 1 { 0.0 } else { mode.nutrient_gain_rate };
                    nutrient_gain_rates_data.push(nutrient_rate);
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
        
        // Cell types - derived from mode settings
        let cell_types_data: Vec<u32> = (0..state.cell_count)
            .map(|i| {
                let genome_id = state.genome_ids[i];
                let mode_idx = state.mode_indices[i];
                if genome_id < genomes.len() {
                    let genome = &genomes[genome_id];
                    if mode_idx < genome.modes.len() {
                        genome.modes[mode_idx].cell_type as u32
                    } else {
                        0 // Default to Test cell type
                    }
                } else {
                    0 // Default to Test cell type
                }
            })
            .collect();
        queue.write_buffer(&self.cell_types, 0, bytemuck::cast_slice(&cell_types_data));
        
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
        
        // Initialize mass deltas to 0 (no transfers)
        let mass_deltas: Vec<f32> = vec![0.0f32; state.cell_count];
        queue.write_buffer(&self.mass_deltas_buffer, 0, bytemuck::cast_slice(&mass_deltas));
        
        // Initialize division flags to 0
        let division_flags: Vec<u32> = vec![0u32; state.cell_count];
        queue.write_buffer(&self.division_flags, 0, bytemuck::cast_slice(&division_flags));
        
        // Sync genome mode data for division (child orientations)
        self.sync_genome_mode_data(queue, genomes);
        
        // Sync child mode indices for division
        self.sync_child_mode_indices(queue, genomes);
        
        // Sync mode properties for division (nutrient_gain_rate, max_cell_size, etc.)
        self.sync_mode_properties(queue, genomes);

        // Sync mode cell types lookup table (for deriving cell_type from mode_index)
        self.sync_mode_cell_types(queue, genomes);

        // Sync behavior flags for all cell types (applies_swim_force, etc.)
        self.sync_behavior_flags(queue);
    }
    
    /// Update cell_types buffer when genome mode cell_type settings change.
    /// 
    /// This should be called when a mode's cell_type is changed (e.g., from Test to Flagellocyte)
    /// to ensure existing cells are rendered with the correct pipeline.
    /// 
    /// # Arguments
    /// * `queue` - The wgpu queue for buffer writes
    /// * `genomes` - The current genomes with updated mode settings
    /// * `cell_count` - Current number of live cells
    /// * `genome_ids` - Genome ID for each cell (from GPU buffer readback or canonical state)
    /// * `mode_indices` - Mode index for each cell (from GPU buffer readback or canonical state)
    pub fn update_cell_types_from_genomes(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
        cell_count: usize,
        genome_ids: &[usize],
        mode_indices: &[usize],
    ) {
        if cell_count == 0 {
            return;
        }
        
        // Derive cell types from current genome mode settings
        let cell_types_data: Vec<u32> = (0..cell_count)
            .map(|i| {
                let genome_id = genome_ids[i];
                let mode_idx = mode_indices[i];
                if genome_id < genomes.len() {
                    let genome = &genomes[genome_id];
                    if mode_idx < genome.modes.len() {
                        genome.modes[mode_idx].cell_type as u32
                    } else {
                        0 // Default to Test cell type
                    }
                } else {
                    0 // Default to Test cell type
                }
            })
            .collect();
        
        queue.write_buffer(&self.cell_types, 0, bytemuck::cast_slice(&cell_types_data));
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
    
    /// Sync child mode indices for division shader
    pub fn sync_child_mode_indices(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        // Layout per mode: [child_a_mode (i32), child_b_mode (i32)] = 8 bytes
        let mut mode_indices: Vec<[i32; 2]> = Vec::new();
        
        let mut global_mode_offset = 0i32;
        for genome in genomes {
            for mode in &genome.modes {
                // Child mode numbers are relative to the genome, need to add global offset
                let child_a_mode = global_mode_offset + mode.child_a.mode_number.max(0);
                let child_b_mode = global_mode_offset + mode.child_b.mode_number.max(0);
                mode_indices.push([child_a_mode, child_b_mode]);
            }
            global_mode_offset += genome.modes.len() as i32;
        }
        
        if !mode_indices.is_empty() {
            queue.write_buffer(&self.child_mode_indices, 0, bytemuck::cast_slice(&mode_indices));
        }
    }
    
    /// Sync parent make adhesion flags for division shader
    pub fn sync_parent_make_adhesion_flags(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut flags_data: Vec<u32> = Vec::new();
        
        for genome in genomes {
            for mode in &genome.modes {
                flags_data.push(if mode.parent_make_adhesion { 1 } else { 0 });
            }
        }
        
        if !flags_data.is_empty() {
            queue.write_buffer(&self.parent_make_adhesion_flags, 0, bytemuck::cast_slice(&flags_data));
        }
    }
    
    /// Sync child keep adhesion flags for division shader (zone-based inheritance)
    pub fn sync_child_keep_adhesion_flags(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut child_a_flags: Vec<u32> = Vec::new();
        let mut child_b_flags: Vec<u32> = Vec::new();
        
        for genome in genomes {
            for mode in &genome.modes {
                child_a_flags.push(if mode.child_a.keep_adhesion { 1 } else { 0 });
                child_b_flags.push(if mode.child_b.keep_adhesion { 1 } else { 0 });
            }
        }
        
        if !child_a_flags.is_empty() {
            queue.write_buffer(&self.child_a_keep_adhesion_flags, 0, bytemuck::cast_slice(&child_a_flags));
            queue.write_buffer(&self.child_b_keep_adhesion_flags, 0, bytemuck::cast_slice(&child_b_flags));
        }
    }
    
    /// Sync mode properties for division shader (per-mode properties like nutrient_gain_rate, max_cell_size, etc.)
    pub fn sync_mode_properties(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        // Layout per mode: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval] (vec4)
        //                  [split_mass, nutrient_priority, swim_force, prioritize_when_low] (vec4)
        //                  [max_splits, split_ratio, padding, padding] (vec4)
        // Total: 12 floats = 48 bytes per mode
        let mut properties_data: Vec<[f32; 12]> = Vec::new();
        
        for genome in genomes {
            for mode in &genome.modes {
                // Convert max_splits: -1 (infinite) -> 0 (unlimited in GPU)
                let gpu_max_splits = if mode.max_splits < 0 { 0.0 } else { mode.max_splits as f32 };
                properties_data.push([
                    mode.nutrient_gain_rate,
                    mode.max_cell_size,
                    mode.membrane_stiffness,
                    mode.split_interval,
                    mode.split_mass,
                    mode.nutrient_priority,
                    mode.swim_force,
                    if mode.prioritize_when_low { 1.0 } else { 0.0 },
                    gpu_max_splits,
                    mode.split_ratio, // split_ratio instead of padding
                    0.0, // padding
                    0.0, // padding
                ]);
            }
        }
        
        if !properties_data.is_empty() {
            queue.write_buffer(&self.mode_properties, 0, bytemuck::cast_slice(&properties_data));
        }
    }
    
    /// Sync mode cell types lookup table (cell_type per mode)
    /// This allows shaders to derive cell_type from mode_index, which is always up-to-date
    pub fn sync_mode_cell_types(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mode_cell_types: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| mode.cell_type as u32)
            })
            .collect();

        if !mode_cell_types.is_empty() {
            queue.write_buffer(&self.mode_cell_types, 0, bytemuck::cast_slice(&mode_cell_types));
        }
    }

    /// Sync behavior flags for all cell types
    /// This populates the GPU buffer with behavior flags (applies_swim_force, etc.)
    /// for each cell type. Should be called once during initialization.
    pub fn sync_behavior_flags(&self, queue: &wgpu::Queue) {
        use crate::cell::types::{CellType, GpuCellTypeBehaviorFlags};

        let flags: Vec<GpuCellTypeBehaviorFlags> = CellType::iter()
            .map(|t| t.behavior_flags())
            .collect();

        queue.write_buffer(&self.behavior_flags, 0, bytemuck::cast_slice(&flags));
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
        cell_type: u32,
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
        queue.write_buffer(&self.cell_types, offset, bytemuck::bytes_of(&cell_type));
        queue.write_buffer(&self.mode_indices, offset, bytemuck::bytes_of(&(mode_index as u32)));
        queue.write_buffer(&self.cell_ids, offset, bytemuck::bytes_of(&(cell_id as u32)));
        queue.write_buffer(&self.death_flags, offset, bytemuck::bytes_of(&0u32)); // Alive
        queue.write_buffer(&self.division_flags, offset, bytemuck::bytes_of(&0u32)); // Not dividing
    }
    
    /// Add cell with deterministic slot assignment
    /// Returns (slot_index, cell_id) if successful, None if no slots available
    pub fn add_cell_deterministic(
        &mut self,
        queue: &wgpu::Queue,
        request: CellAdditionRequest,
        genomes: &[crate::genome::Genome],
    ) -> Option<(u32, u32)> {
        // Allocate slot deterministically
        let (slot, cell_id) = self.slot_allocator.allocate_slot()?;
        
        // Sync position/velocity/rotation to all buffer sets
        self.sync_single_cell(
            queue,
            slot as usize,
            request.position,
            request.velocity,
            request.mass,
            request.rotation,
        );
        
        // Calculate absolute mode index (with genome offset)
        let absolute_mode_idx = self.calculate_absolute_mode_index(
            request.genome_id,
            request.mode_index,
            genomes,
        );
        
        // Get cell_type from genome mode settings
        let cell_type = if request.genome_id < genomes.len() {
            let genome = &genomes[request.genome_id];
            if request.mode_index < genome.modes.len() {
                genome.modes[request.mode_index].cell_type as u32
            } else {
                0 // Default to Test cell type
            }
        } else {
            0 // Default to Test cell type
        };
        
        // Sync cell state for division system
        self.sync_single_cell_state(
            queue,
            slot as usize,
            request.birth_time,
            request.split_interval,
            request.split_mass,
            request.genome_id,
            absolute_mode_idx,
            cell_id as usize,
            -1, // Default max_splits (unlimited)
            0.2, // Default nutrient_gain_rate
            2.0, // Default max_cell_size
            request.stiffness,
            cell_type,
        );
        
        // Update GPU cell count
        let new_count = self.slot_allocator.allocated_count();
        self.update_gpu_cell_count(queue, new_count);
        
        Some((slot, cell_id))
    }
    
    /// Calculate absolute mode index with genome offset
    fn calculate_absolute_mode_index(
        &self,
        genome_id: usize,
        local_mode_idx: usize,
        genomes: &[crate::genome::Genome],
    ) -> usize {
        let mut offset = 0;
        for (i, genome) in genomes.iter().enumerate() {
            if i == genome_id {
                return offset + local_mode_idx;
            }
            offset += genome.modes.len();
        }
        offset + local_mode_idx // Fallback if genome not found
    }
    
    /// Update GPU cell count buffer
    pub fn update_gpu_cell_count(&self, queue: &wgpu::Queue, new_count: u32) {
        let cell_counts: [u32; 2] = [new_count, new_count];
        queue.write_buffer(&self.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
    }
    
    /// Start an async read of cell count from GPU.
    /// Call poll_cell_count() to check if the read is complete.
    /// This is non-blocking and won't cause frame spikes.
    pub fn start_cell_count_read(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // Don't start a new read if one is already pending
        if self.cell_count_map_pending {
            return;
        }
        
        // Copy cell count buffer to readback buffer
        encoder.copy_buffer_to_buffer(
            &self.cell_count_buffer,
            0,
            &self.cell_count_readback_buffer,
            0,
            8, // 2 x u32
        );
    }
    
    /// Initiate the async map operation after command buffer submission.
    /// Call this after queue.submit() to start the async readback.
    pub fn initiate_cell_count_map(&mut self) {
        // Don't start a new read if one is already pending
        if self.cell_count_map_pending {
            return;
        }
        
        let buffer_slice = self.cell_count_readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        self.cell_count_map_pending = true;
        self.cell_count_receiver = Some(rx);
    }
    
    /// Poll for async cell count read completion.
    /// Returns Some(count) if new count is available, None if still pending or no read started.
    /// This is non-blocking.
    pub fn poll_cell_count(&mut self, device: &wgpu::Device) -> Option<u32> {
        if !self.cell_count_map_pending {
            return None;
        }
        
        // Do a non-blocking poll to push GPU work forward
        let _ = device.poll(wgpu::PollType::Poll);
        
        // Check if the map operation completed
        if let Some(ref rx) = self.cell_count_receiver {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    // Map succeeded, read the data
                    let buffer_slice = self.cell_count_readback_buffer.slice(..);
                    let data = buffer_slice.get_mapped_range();
                    let counts: &[u32] = bytemuck::cast_slice(&data);
                    
                    // counts[0] = total cells, counts[1] = live cells
                    // Use live cells count for display
                    self.last_cell_count = counts[1];
                    
                    drop(data);
                    self.cell_count_readback_buffer.unmap();
                    
                    self.cell_count_map_pending = false;
                    self.cell_count_receiver = None;
                    return Some(self.last_cell_count);
                }
                Ok(Err(_)) => {
                    // Map failed, reset state
                    self.cell_count_map_pending = false;
                    self.cell_count_receiver = None;
                    return None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still pending
                    return None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Channel closed unexpectedly
                    self.cell_count_map_pending = false;
                    self.cell_count_receiver = None;
                    return None;
                }
            }
        }
        
        None
    }
    
    /// Get the last known cell count from GPU readback.
    pub fn last_cell_count(&self) -> u32 {
        self.last_cell_count
    }
    
    /// Check if a cell count readback is pending.
    pub fn is_cell_count_read_pending(&self) -> bool {
        self.cell_count_map_pending
    }
    
    /// Free a cell slot (for cell death)
    pub fn free_cell_slot(&mut self, slot: u32) {
        self.slot_allocator.free_slot(slot);
    }
    
    /// Compact free slots for deterministic allocation
    pub fn compact_free_slots(&mut self) {
        self.slot_allocator.compact_free_slots();
    }
    
    /// Get current allocated cell count
    pub fn allocated_cell_count(&self) -> u32 {
        self.slot_allocator.allocated_count()
    }
    
    /// Get available slot count
    pub fn available_slots(&self) -> usize {
        self.slot_allocator.available_slots()
    }
    
    /// Reset slot allocator (for scene reset)
    pub fn reset_slot_allocator(&mut self) {
        self.slot_allocator.reset();
        self.cell_addition_pipeline.clear_pending();
    }
    
    /// Synchronize slot allocator with canonical state
    pub fn sync_slot_allocator_with_canonical(&mut self, canonical_state: &CanonicalState) {
        // Reset allocator
        self.slot_allocator.reset();
        
        // Set next cell ID to match canonical state
        self.slot_allocator.set_next_cell_id(canonical_state.next_cell_id as u32);
        
        // Mark slots as allocated for existing cells
        for i in 0..canonical_state.cell_count {
            if let Some((slot, _)) = self.slot_allocator.allocate_slot() {
                // Slot allocated successfully
                assert_eq!(slot, i as u32, "Slot allocation mismatch during sync");
            }
        }
    }
    
    /// Queue a cell for deterministic addition
    /// 
    /// The cell will be added during the next call to `process_pending_cell_additions()`.
    /// This ensures all additions are processed in deterministic order.
    pub fn queue_cell_addition(&mut self, request: CellAdditionRequest) {
        self.cell_addition_pipeline.queue_cell_addition(request);
    }
    
    /// Process all pending cell additions in deterministic order
    /// 
    /// Returns list of (slot_index, cell_id) pairs for successfully added cells.
    /// This method should be called once per frame to process queued additions.
    pub fn process_pending_cell_additions(
        &mut self,
        queue: &wgpu::Queue,
        canonical_state: &mut CanonicalState,
        genomes: &[crate::genome::Genome],
    ) -> Vec<(u32, u32)> {
        // Extract pending requests to avoid borrowing issues
        let mut pending_requests = std::mem::take(&mut self.cell_addition_pipeline.pending_requests);
        
        if pending_requests.is_empty() {
            return Vec::new();
        }
        
        // Sort requests by deterministic hash for consistent ordering
        pending_requests.sort_by_key(|req| req.deterministic_hash());
        
        let mut added_cells = Vec::new();
        
        // Process each request in sorted order
        for request in pending_requests {
            if let Some((slot, cell_id)) = self.add_cell_deterministic(
                queue,
                request.clone(),
                genomes,
            ) {
                // Also add to canonical state at the same slot
                if let Some(_canonical_index) = canonical_state.add_cell_at_slot(
                    slot as usize,
                    request.position,
                    request.velocity,
                    request.rotation,
                    request.genome_orientation,
                    request.angular_velocity,
                    request.mass,
                    request.radius,
                    request.genome_id,
                    request.mode_index,
                    request.birth_time,
                    request.split_interval,
                    request.split_mass,
                    request.stiffness,
                    cell_id,
                ) {
                    added_cells.push((slot, cell_id));
                } else {
                    // Failed to add to canonical state, free the GPU slot
                    self.slot_allocator.free_slot(slot);
                }
            }
        }
        
        added_cells
    }
    
    /// Get number of pending cell additions
    pub fn pending_cell_addition_count(&self) -> usize {
        self.cell_addition_pipeline.pending_count()
    }
    
    /// Clear all pending cell additions
    pub fn clear_pending_cell_additions(&mut self) {
        self.cell_addition_pipeline.clear_pending();
    }
    
    /// Create a cell addition request from basic parameters
    pub fn create_cell_addition_request(
        position: glam::Vec3,
        velocity: glam::Vec3,
        mass: f32,
        rotation: glam::Quat,
        genome_id: usize,
        mode_index: usize,
        birth_time: f32,
        genomes: &[crate::genome::Genome],
    ) -> CellAdditionRequest {
        // Get parameters from genome mode
        let (split_interval, split_mass, stiffness) = if genome_id < genomes.len() {
            let genome = &genomes[genome_id];
            if mode_index < genome.modes.len() {
                let mode = &genome.modes[mode_index];
                (mode.split_interval, mode.split_mass, mode.membrane_stiffness)
            } else {
                (10.0, 2.0, 50.0) // Default values
            }
        } else {
            (10.0, 2.0, 50.0) // Default values
        };
        
        // Calculate radius from mass
        let radius = (mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        CellAdditionRequest {
            position,
            velocity,
            mass,
            rotation,
            genome_id,
            mode_index,
            birth_time,
            split_interval,
            split_mass,
            stiffness,
            radius,
            genome_orientation: rotation, // Use same as physics rotation initially
            angular_velocity: glam::Vec3::ZERO,
        }
    }
    
    /// Mark that buffers need to be synced from canonical state
    pub fn mark_needs_sync(&mut self) {
        self.needs_sync = true;
    }
    
    /// Queue a cell insertion to be synced on next physics step
    pub fn queue_cell_insertion(&mut self, cell_idx: usize) {
        self.pending_cell_insertions.push(cell_idx);
    }
    
    /// Check if there are pending cell insertions
    pub fn has_pending_insertions(&self) -> bool {
        !self.pending_cell_insertions.is_empty()
    }
    
    /// Get the number of pending cell insertions
    pub fn pending_insertion_count(&self) -> usize {
        self.pending_cell_insertions.len()
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
                let (max_splits, nutrient_gain_rate, max_cell_size, stiffness, cell_type) = if genome_id < genomes.len() {
                    let genome = &genomes[genome_id];
                    if local_mode_idx < genome.modes.len() {
                        let mode = &genome.modes[local_mode_idx];
                        // Flagellocytes (cell_type == 1) don't generate their own nutrients
                        let nutrient_rate = if mode.cell_type == 1 { 0.0 } else { mode.nutrient_gain_rate };
                        (mode.max_splits, nutrient_rate, mode.max_cell_size, mode.membrane_stiffness, mode.cell_type as u32)
                    } else {
                        (-1, 0.2, 2.0, 50.0, 0) // Defaults if mode not found
                    }
                } else {
                    (-1, 0.2, 2.0, 50.0, 0) // Defaults if genome not found
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
                    cell_type,
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
    

    
    /// DEBUG: Blocking readback of position/mass data from a specific buffer.
    /// This is expensive and should only be used for debugging!
    pub fn debug_read_positions_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue, buffer_index: usize, cell_count: usize) -> Vec<[f32; 4]> {
        if cell_count == 0 || buffer_index >= 3 {
            return Vec::new();
        }
        
        let read_size = (cell_count * 16) as u64; // Vec4<f32> = 16 bytes
        
        // Create staging buffer for readback
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Position Staging"),
            size: read_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy from GPU buffer to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.position_and_mass[buffer_index],
            0,
            &staging,
            0,
            read_size,
        );
        queue.submit(std::iter::once(encoder.finish()));
        
        // Map and read (blocking)
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let view = slice.get_mapped_range();
            let data: Vec<[f32; 4]> = bytemuck::cast_slice(&view).to_vec();
            drop(view);
            staging.unmap();
            data
        } else {
            Vec::new()
        }
    }
    
    /// DEBUG: Blocking readback of cell_count_buffer [total, live].
    pub fn debug_read_cell_count_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [u32; 2] {
        let read_size = 8u64; // 2 x u32
        
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Cell Count Staging"),
            size: read_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Cell Count Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.cell_count_buffer, 0, &staging, 0, read_size);
        queue.submit(std::iter::once(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let view = slice.get_mapped_range();
            let data: &[u32] = bytemuck::cast_slice(&view);
            let result = [data[0], data[1]];
            drop(view);
            staging.unmap();
            result
        } else {
            [0, 0]
        }
    }
    
    /// DEBUG: Blocking readback of mode_indices buffer.
    pub fn debug_read_mode_indices_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue, count: usize) -> Vec<u32> {
        if count == 0 {
            return Vec::new();
        }
        
        let read_size = (count * 4) as u64; // u32 = 4 bytes
        
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Mode Indices Staging"),
            size: read_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Mode Indices Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.mode_indices, 0, &staging, 0, read_size);
        queue.submit(std::iter::once(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let view = slice.get_mapped_range();
            let data: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
            drop(view);
            staging.unmap();
            data
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Quat};
    
    #[test]
    fn test_deterministic_slot_allocator() {
        let mut allocator = DeterministicSlotAllocator::new(10);
        
        // Test initial state
        assert_eq!(allocator.allocated_count(), 0);
        assert_eq!(allocator.available_slots(), 10);
        
        // Allocate some slots
        let (slot1, id1) = allocator.allocate_slot().unwrap();
        let (slot2, id2) = allocator.allocate_slot().unwrap();
        let (slot3, id3) = allocator.allocate_slot().unwrap();
        
        // Should allocate in order
        assert_eq!(slot1, 0);
        assert_eq!(slot2, 1);
        assert_eq!(slot3, 2);
        
        // IDs should increment
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        
        // Check counts
        assert_eq!(allocator.allocated_count(), 3);
        assert_eq!(allocator.available_slots(), 7);
        
        // Free a slot
        allocator.free_slot(1);
        assert_eq!(allocator.available_slots(), 8); // Now 8 (1 freed + 7 remaining)
        
        // Compact and check (should still be 8)
        allocator.compact_free_slots();
        assert_eq!(allocator.available_slots(), 8); // Still 8 after compaction
        
        // Next allocation should use slot 1 (freed slot, lowest available)
        let (slot4, id4) = allocator.allocate_slot().unwrap();
        assert_eq!(slot4, 1); // Reused freed slot
        assert_eq!(id4, 4);   // ID continues incrementing
    }
    
    #[test]
    fn test_cell_addition_request_deterministic_hash() {
        let request1 = CellAdditionRequest {
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::ZERO,
            mass: 1.0,
            rotation: Quat::IDENTITY,
            genome_id: 0,
            mode_index: 0,
            birth_time: 0.0,
            split_interval: 10.0,
            split_mass: 2.0,
            stiffness: 50.0,
            radius: 1.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };
        
        let request2 = CellAdditionRequest {
            position: Vec3::new(1.0, 2.0, 3.0), // Same position
            velocity: Vec3::new(1.0, 0.0, 0.0), // Different velocity (shouldn't affect hash)
            mass: 2.0,                           // Different mass (shouldn't affect hash)
            rotation: Quat::IDENTITY,
            genome_id: 0,                        // Same genome/mode
            mode_index: 0,
            birth_time: 5.0,                     // Different time (shouldn't affect hash)
            split_interval: 15.0,
            split_mass: 3.0,
            stiffness: 75.0,
            radius: 1.2,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };
        
        let request3 = CellAdditionRequest {
            position: Vec3::new(1.0, 2.0, 3.0), // Same position
            velocity: Vec3::ZERO,
            mass: 1.0,
            rotation: Quat::IDENTITY,
            genome_id: 1,                        // Different genome (should affect hash)
            mode_index: 0,
            birth_time: 0.0,
            split_interval: 10.0,
            split_mass: 2.0,
            stiffness: 50.0,
            radius: 1.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };
        
        // Same position and genome should have same hash
        assert_eq!(request1.deterministic_hash(), request2.deterministic_hash());
        
        // Different genome should have different hash
        assert_ne!(request1.deterministic_hash(), request3.deterministic_hash());
    }
    
    #[test]
    fn test_deterministic_cell_addition_sorting() {
        let mut pipeline = DeterministicCellAddition::new();
        
        // Add requests in non-deterministic order
        let request_high_hash = CellAdditionRequest {
            position: Vec3::new(10.0, 10.0, 10.0), // High coordinates = high hash
            velocity: Vec3::ZERO,
            mass: 1.0,
            rotation: Quat::IDENTITY,
            genome_id: 1,
            mode_index: 1,
            birth_time: 0.0,
            split_interval: 10.0,
            split_mass: 2.0,
            stiffness: 50.0,
            radius: 1.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };
        
        let request_low_hash = CellAdditionRequest {
            position: Vec3::new(1.0, 1.0, 1.0), // Low coordinates = low hash
            velocity: Vec3::ZERO,
            mass: 1.0,
            rotation: Quat::IDENTITY,
            genome_id: 0,
            mode_index: 0,
            birth_time: 0.0,
            split_interval: 10.0,
            split_mass: 2.0,
            stiffness: 50.0,
            radius: 1.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };
        
        // Add high hash first, then low hash
        pipeline.queue_cell_addition(request_high_hash.clone());
        pipeline.queue_cell_addition(request_low_hash.clone());
        
        assert_eq!(pipeline.pending_count(), 2);
        
        // Verify they get sorted by hash (low hash should come first)
        assert!(request_low_hash.deterministic_hash() < request_high_hash.deterministic_hash());
    }
}