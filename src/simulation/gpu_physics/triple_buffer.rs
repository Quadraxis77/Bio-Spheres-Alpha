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
//! ### Spatial Grid (128^3 = 2,097,152 grid cells)
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

use crate::simulation::CanonicalState;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEVELOPMENT_ROOT_LINEAGE_HASH: u64 = 0x9E37_79B9_7F4A_7C15;

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
    pub nutrients: f32,
    pub rotation: glam::Quat,
    pub genome_id: usize,
    pub mode_index: usize,
    pub birth_time: f32,
    pub split_interval: f32,
    pub split_nutrient_threshold: f32,
    pub stiffness: f32,
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

    /// Spatial grid cell counts (128^3 = 2,097,152 grid cells)
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

    /// Nutrients buffer: fixed-point i32 with scale 1000 (100.0 nutrients = 100000 i32)
    /// Stored as atomic for safe concurrent updates from multiple shaders
    pub nutrients_buffer: wgpu::Buffer,

    /// Mass deltas for nutrient transport (accumulates mass changes)
    pub mass_deltas_buffer: wgpu::Buffer,

    /// Division flags: 1 = wants to divide, 0 = not dividing
    pub division_flags: wgpu::Buffer,

    /// Compacted free slot indices (result of prefix-sum on death_flags)
    pub free_slot_indices: wgpu::Buffer,

    /// Division slot assignments: maps dividing cell to its assigned free slot index
    pub division_slot_assignments: wgpu::Buffer,

    /// Lifecycle counts: [0] = free slot count, [1] = division count (DEPRECATED - use ring buffer)
    pub lifecycle_counts: wgpu::Buffer,

    /// Ring buffer for free slot recycling (persistent across frames)
    /// Capacity: 262144 slots (256K) to support up to 200K cells
    pub free_slot_ring: wgpu::Buffer,

    /// Ring buffer state: [0]=head, [1]=tail, [2]=next_slot_id, [3]=reservation_count
    pub ring_state: wgpu::Buffer,

    // === Cell state buffers for division ===
    /// Birth times for each cell
    pub birth_times: wgpu::Buffer,

    /// Last mode-switch time for each cell (f32, seconds).
    /// Written by mode_switch.wgsl whenever a cell changes mode.
    /// Read by adhesion_physics.wgsl to extend the bond-break grace period
    /// so that maturation-triggered mode switches don't cause transient bond breaks.
    pub mode_switch_time: wgpu::Buffer,

    /// Split intervals (time between divisions)
    pub split_intervals: wgpu::Buffer,

    /// Split nutrient thresholds (nutrient level required for division)
    pub split_nutrient_thresholds: wgpu::Buffer,

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

    /// Genome mode data buffers for division (split into 5 vec4 sub-buffers to stay under wgpu 256 MB/buffer limit)
    /// Combined layout per mode: [child_a_orientation (v0), child_b_orientation (v1),
    ///                            child_a_split_orientation (v2), child_b_split_orientation (v3),
    ///                            split_rotation_quat XYZW (v4)]
    /// Each sub-buffer: num_modes * 16 bytes
    pub genome_mode_data_v0: wgpu::Buffer,
    pub genome_mode_data_v1: wgpu::Buffer,
    pub genome_mode_data_v2: wgpu::Buffer,
    pub genome_mode_data_v3: wgpu::Buffer,
    pub genome_mode_data_v4: wgpu::Buffer,

    /// Child mode indices buffer (two i32 per mode: child_a_mode, child_b_mode)
    /// Total size: num_modes * 8 bytes
    pub child_mode_indices: wgpu::Buffer,

    /// Per-mode flag: 1 if this mode is the genome's initial mode, 0 otherwise.
    /// Used by the division shader to assign fresh organism IDs to children that
    /// restart development at the initial mode (i.e., a new sub-organism is born).
    pub is_initial_mode: wgpu::Buffer,

    /// Parent make adhesion flags buffer (one u32 per mode)
    /// Stores whether each mode allows sibling adhesion creation during division
    pub parent_make_adhesion_flags: wgpu::Buffer,

    /// Child A keep adhesion flags buffer (one u32 per mode)
    /// Stores whether Child A should inherit adhesions during division
    pub child_a_keep_adhesion_flags: wgpu::Buffer,

    /// Child B keep adhesion flags buffer (one u32 per mode)
    /// Stores whether Child B should inherit adhesions during division
    pub child_b_keep_adhesion_flags: wgpu::Buffer,

    /// Child A after-split keep adhesion flags buffer (one u32 per mode)
    /// Stores whether Child A should inherit adhesions when max_splits is reached
    pub child_a_after_split_keep_adhesion_flags: wgpu::Buffer,

    /// Child B after-split keep adhesion flags buffer (one u32 per mode)
    /// Stores whether Child B should inherit adhesions when max_splits is reached
    pub child_b_after_split_keep_adhesion_flags: wgpu::Buffer,

    /// Mode properties buffers (split into 5 vec4 sub-buffers to stay under wgpu 256 MB/buffer limit)
    /// v0: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
    /// v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
    /// v2: [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
    /// v3: [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
    /// v4: [max_adhesions, mode_a_after_splits, mode_b_after_splits, buoyancy_force]
    /// Each sub-buffer: num_modes * 16 bytes
    pub mode_properties_v0: wgpu::Buffer,
    pub mode_properties_v1: wgpu::Buffer,
    pub mode_properties_v2: wgpu::Buffer,
    pub mode_properties_v3: wgpu::Buffer,
    pub mode_properties_v4: wgpu::Buffer,

    /// Mode properties v5-v6 for cilia parameters (16 bytes per mode each)
    /// v5: [cilia_speed, cilia_push_bonded as f32, cilia_use_signal as f32, cilia_signal_channel as f32]
    /// v6: [cilia_speed_below, cilia_speed_above, cilia_threshold, cilia_attract_force]
    pub mode_properties_v5: wgpu::Buffer,
    pub mode_properties_v6: wgpu::Buffer,

    /// Mode properties v7 for myocyte/luminocyte parameters (16 bytes per mode)
    /// Myocyte:    [myocyte_contraction, myocyte_use_signal as f32, myocyte_signal_channel as f32, myocyte_threshold]
    /// Luminocyte: [luminocyte_invert as f32, 0.0, luminocyte_signal_channel as f32, luminocyte_threshold]
    /// v8: [myocyte_contraction_above, myocyte_contraction_below, myocyte_pulse_rate, myocyte_pulse_phase as f32]
    pub mode_properties_v7: wgpu::Buffer,
    pub mode_properties_v8: wgpu::Buffer,

    /// Mode properties v9-v10 for Embryocyte parameters (16 bytes per mode each)
    /// v9:  [use_timer as f32, release_timer, use_threshold as f32, threshold_value as f32]
    /// v10: [use_signal as f32, signal_channel as f32, signal_value, 0.0]
    pub mode_properties_v9: wgpu::Buffer,
    pub mode_properties_v10: wgpu::Buffer,

    /// Mode properties v11 for Devorocyte parameters (16 bytes per mode each)
    /// v11: [consume_range, consume_rate, 0.0, 0.0]
    pub mode_properties_v11: wgpu::Buffer,

    /// Mode properties v12 for Vasculocyte parameters (16 bytes per mode each)
    /// v12: [nutrient_transport, nutrient_exchange, signal_transport, signal_exchange]
    pub mode_properties_v12: wgpu::Buffer,

    /// Mode properties v13 for Gametocyte parameters (16 bytes per mode each)
    /// v13: [merge_range, 0.0, 0.0, 0.0]
    pub mode_properties_v13: wgpu::Buffer,

    /// Mode properties v14 for Siphonocyte parameters (16 bytes per mode each)
    /// v14: [intake_rate, expel_rate, impulse, packed_signal_channel_and_mode]
    pub mode_properties_v14: wgpu::Buffer,

    /// Mode properties v15 for Plumocyte parameters (16 bytes per mode each)
    /// v15: [extension, drag_mult, flow_coupling, exposure_mult]
    pub mode_properties_v15: wgpu::Buffer,

    /// Per-cell Embryocyte reserve buffer (one u32 per cell).
    /// Embryocytes: sole energy source; burns at 10 units/sec when free.
    /// Non-Embryocytes: provides extended life; burns before normal nutrients.
    /// Halved on division: child_reserve = parent_reserve >> 1.
    pub embryocyte_reserve_buffer: wgpu::Buffer,

    /// Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
    /// Used by shaders to derive cell_type from mode_index (always up-to-date with genome settings)
    pub mode_cell_types: wgpu::Buffer,

    /// Current buffer index (atomic for lock-free rotation)
    current_index: AtomicUsize,

    /// Cell capacity
    pub capacity: u32,

    /// Current allocated size of all mode pool sub-buffers (in number of modes).
    /// Starts at INITIAL_MODE_POOL_SIZE and doubles on demand up to MAX_TOTAL_MODES.
    /// All mode sub-buffers (genome_mode_data_v*, mode_properties_v*, signal_settings_v*,
    /// child_mode_indices, flags, oculocyte_params, regulation_params, etc.) are always
    /// exactly this many modes in size.
    pub mode_pool_capacity: u64,

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

    /// Per-cell environment adhesion anchor: Vec4(anchor_x, anchor_y, anchor_z, is_active)
    /// w=1.0 means anchor is active, w=0.0 means no anchor.
    pub env_anchor_buffer: wgpu::Buffer,

    /// Per-cell muscle contraction value (one f32 per cell).
    /// Written by muscle_contraction shader each frame, read by adhesion_physics shader.
    /// Value 0.0 = relaxed (full rest length), 1.0 = fully contracted (zero length).
    /// Each cell only controls its own half of the adhesion bond.
    pub muscle_contraction_buffer: wgpu::Buffer,

    /// Per-cell grip (friction drag) value (one f32 per cell).
    /// Written by muscle_contraction shader alongside contraction.
    /// mix(grip_extended, grip_contracted, contraction) — read by position_update for medium friction.
    pub cell_grip_buffer: wgpu::Buffer,

    /// Per-cell physiology scalar buffers. Current values are read by the physiology pass,
    /// next values are written by that pass and copied back on the low-frequency cadence.
    pub cell_water: wgpu::Buffer,
    pub cell_heat_energy: wgpu::Buffer,
    pub cell_cached_temperature: wgpu::Buffer,
    pub cell_thermal_state: wgpu::Buffer,
    pub cell_water_next: wgpu::Buffer,
    pub cell_heat_energy_next: wgpu::Buffer,
    pub cell_cached_temperature_next: wgpu::Buffer,
    pub cell_thermal_state_next: wgpu::Buffer,
    pub cell_prev_muscle_contraction: wgpu::Buffer,

    /// Per-mode glueocyte environment adhesion flags (one u32 per mode)
    pub glueocyte_env_adhesion_flags: wgpu::Buffer,
    pub glueocyte_boulder_adhesion_flags: wgpu::Buffer,

    /// Per-mode glueocyte cell-adhesion flags packed as two u32 per mode:
    ///   [0] = cell_adhesion_enabled (0/1)
    ///   [1] = signal_channel as u32 (0xFFFFFFFF = disabled / always-active)
    ///   [2] = signal_threshold as f32 bits
    ///   [3] = padding
    /// Stored as a flat array of u32: mode_index * 4 + field_index
    pub glueocyte_cell_adhesion_flags: wgpu::Buffer,

    /// Per-mode oculocyte parameters: [sense_type(u32), ray_length(f32), signal_hops(u32), signal_channel(u32)] = 16 bytes per mode
    pub oculocyte_params: wgpu::Buffer,

    /// Per-mode oculocyte signal values: one f32 per mode (the value emitted when target detected)
    /// Stored separately from oculocyte_params to avoid breaking the mutation system's vec4<u32> stride.
    pub oculocyte_signal_values: wgpu::Buffer,
    /// Per-mode oculocyte light color filters: [target_r, target_g, target_b, tolerance].
    pub oculocyte_light_filters: wgpu::Buffer,

    /// Per-mode regulation emission parameters: [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding(u32)] = 16 bytes per mode
    /// emit_channel: 0xFFFFFFFF = disabled, 8-15 = regulation channel
    pub regulation_params: wgpu::Buffer,

    /// Per-mode signal-conditional settings (5 vec4 sub-buffers, 16 bytes each per mode)
    /// v0: [division_signal_channel(f32), division_signal_threshold(f32), division_signal_invert(f32 0/1), apoptosis_signal_channel(f32)]
    /// v1: [apoptosis_signal_threshold(f32), apoptosis_signal_invert(f32 0/1), signal_child_a_channel(f32), signal_child_a_threshold(f32)]
    /// v2: [signal_child_a_mode_above(f32), signal_child_a_mode_below(f32), signal_child_b_channel(f32), signal_child_b_threshold(f32)]
    /// v3: [signal_child_b_mode_above(f32), signal_child_b_mode_below(f32), mode_switch_signal_channel(f32), mode_switch_signal_threshold(f32)]
    /// v4: [mode_switch_target(f32), mode_switch_invert(f32 0/1), padding, padding]
    pub signal_settings_v0: wgpu::Buffer,
    pub signal_settings_v1: wgpu::Buffer,
    pub signal_settings_v2: wgpu::Buffer,
    pub signal_settings_v3: wgpu::Buffer,
    pub signal_settings_v4: wgpu::Buffer,

    /// Per-cell genome orientation: Vec4(x, y, z, w) quaternion
    /// Tracks the pure genome-derived orientation chain (parent * split_rotation * child_orientation)
    /// without any physics perturbation. Used by adhesion shaders for anchor transformation
    /// so that structures are defined purely by genome data.
    pub genome_orientations: wgpu::Buffer,

    /// Per-cell development address: [organism_id, lineage_hash_lo, lineage_hash_hi, depth_branch].
    /// depth_branch packs lineage_depth in the high 16 bits and branch_slot in the low 16 bits.
    pub development_addresses: wgpu::Buffer,

    /// Per-cell deterministic address within the current organism.
    pub organism_cell_ids: wgpu::Buffer,

    /// Per-cell parent lineage hash: [parent_lineage_hash_lo, parent_lineage_hash_hi].
    /// Set once at birth and never changed. Root cells store 0.
    /// Used by the scaffold resolver to verify preferred-branch chain membership.
    pub parent_lineage_hashes: wgpu::Buffer,

    /// Indirect dispatch buffer for GPU-driven workgroup counts
    /// Layout: [workgroup_count_x, workgroup_count_y, workgroup_count_z, padding]
    /// Written by compute shader based on cell_count, used for indirect dispatch
    pub indirect_dispatch_buffer: wgpu::Buffer,
}

impl GpuTripleBufferSystem {
    fn physiology_water_capacity_for_cell_type(cell_type: u32) -> f32 {
        if cell_type == 12 {
            10.0
        } else if cell_type == 17 {
            1.5
        } else {
            1.0
        }
    }

    fn physiology_heat_for_temperature(temperature: f32, water: f32) -> f32 {
        temperature * (1.0 + water)
    }

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
        let prev_accelerations =
            Self::create_storage_buffer(device, buffer_size, "Previous Accelerations");

        // Create spatial grid buffers (128^3 = 2,097,152 grid cells)
        let grid_size = 128 * 128 * 128;
        let spatial_grid_counts = Self::create_storage_buffer(
            device,
            grid_size * 4, // u32 = 4 bytes
            "Spatial Grid Counts",
        );

        let spatial_grid_offsets = Self::create_storage_buffer(
            device,
            grid_size * 4, // u32 = 4 bytes
            "Spatial Grid Offsets",
        );

        let cell_grid_indices = Self::create_storage_buffer(
            device,
            capacity as u64 * 4, // u32 per cell
            "Cell Grid Indices",
        );

        // Sorted cell indices by grid cell (16 cells max per grid cell * 128^3 grid cells)
        // This enables O(1) neighbor lookup in collision detection
        let spatial_grid_cells = Self::create_storage_buffer(
            device,
            16 * grid_size * 4, // 16 cells per grid cell * 2,097,152 grid cells * 4 bytes
            "Spatial Grid Cells",
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
        let i32_per_cell = capacity as u64 * 4; // i32 = 4 bytes
        let f32_per_cell = capacity as u64 * 4; // f32 = 4 bytes

        // death_flags must be zero-initialized to prevent cell_insertion shader from
        // finding false "dead" slots in uninitialized memory
        let death_flags =
            Self::create_zero_initialized_storage_buffer(device, u32_per_cell, "Death Flags");

        // Nutrients buffer: i32 with fixed-point scale 1000 (100.0 nutrients = 100000)
        // Pre-initialized to 100000 (full nutrients) for all slots
        let nutrients_buffer =
            Self::create_zero_initialized_storage_buffer(device, i32_per_cell, "Nutrients Buffer");

        let mass_deltas_buffer =
            Self::create_storage_buffer(device, f32_per_cell, "Mass Deltas Buffer"); // i32 = 4 bytes, same as f32
        let division_flags = Self::create_storage_buffer(device, u32_per_cell, "Division Flags");
        let free_slot_indices =
            Self::create_storage_buffer(device, u32_per_cell, "Free Slot Indices");
        let division_slot_assignments = Self::create_max_u32_initialized_storage_buffer(
            device,
            u32_per_cell,
            "Division Slot Assignments",
        );

        // Lifecycle counts: [0] = free slots, [1] = divisions, [2] = dead count (DEPRECATED)
        let lifecycle_counts = Self::create_storage_buffer(device, 12, "Lifecycle Counts");

        // Ring buffer for free slot recycling: 262144 slots (256K u32s = 1MB)
        // Supports up to 200K cells with headroom for churn
        const RING_BUFFER_CAPACITY: u64 = 262144;
        let free_slot_ring =
            Self::create_storage_buffer(device, RING_BUFFER_CAPACITY * 4, "Free Slot Ring");

        // Ring state: [head, tail, next_slot_id, reservation_count] = 16 bytes
        // Zero-initialized so ring starts empty and next_slot_id starts at 0
        let ring_state = Self::create_zero_initialized_storage_buffer(device, 16, "Ring State");

        // Cell state buffers
        let birth_times = Self::create_storage_buffer(device, f32_per_cell, "Birth Times");
        // Initialize mode_switch_time to -1000.0 so the grace period never fires on first frame.
        // Uses mapped_at_creation to avoid needing DeviceExt.
        let mode_switch_time = {
            let size = (capacity as u64 * 4 + 15) & !15;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mode Switch Time"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = buf.slice(..).get_mapped_range_mut();
                let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
                for v in floats.iter_mut() {
                    *v = -1000.0_f32;
                }
            }
            buf.unmap();
            buf
        };
        let split_intervals = Self::create_storage_buffer(device, f32_per_cell, "Split Intervals");
        let split_nutrient_thresholds =
            Self::create_storage_buffer(device, f32_per_cell, "Split Nutrient Thresholds");
        let split_counts = Self::create_storage_buffer(device, u32_per_cell, "Split Counts");
        let split_ready_frame =
            Self::create_storage_buffer(device, u32_per_cell, "Split Ready Frame");
        let max_splits = Self::create_storage_buffer(device, u32_per_cell, "Max Splits");
        let genome_ids = Self::create_storage_buffer(device, u32_per_cell, "Genome IDs");
        // cell_types must be zero-initialized so build_instances shader fallback works correctly
        // (fallback uses mode_cell_types when cell_types[idx] == 0)
        let cell_types =
            Self::create_zero_initialized_storage_buffer(device, u32_per_cell, "Cell Types");
        let mode_indices = Self::create_storage_buffer(device, u32_per_cell, "Mode Indices");
        let cell_ids = Self::create_storage_buffer(device, u32_per_cell, "Cell IDs");
        let next_cell_id = Self::create_storage_buffer(device, 4, "Next Cell ID");
        let nutrient_gain_rates =
            Self::create_storage_buffer(device, f32_per_cell, "Nutrient Gain Rates");
        let max_cell_sizes = Self::create_storage_buffer(device, f32_per_cell, "Max Cell Sizes");
        let stiffnesses = Self::create_storage_buffer(device, f32_per_cell, "Stiffnesses");

        // GPU-side cell count: [0] = total cells, [1] = live cells, [2] = dragged cell index (0xFFFFFFFF = none)
        // Must be zero-initialized so shaders start with 0 cells
        // Slot [2] is written each frame by GpuScene::render() with the current dragged_cell_index
        let cell_count_buffer =
            Self::create_zero_initialized_storage_buffer(device, 16, "Cell Count Buffer");

        // Cell count readback buffer for async GPU-to-CPU transfer
        let cell_count_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Count Readback Buffer"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Mode pool: start small and grow on demand via grow_mode_pool_if_needed().
        // Initial size covers ~200 genomes x 80 modes = 16K modes (~26 MB total across all
        // sub-buffers), vs the old fixed 8M allocation (~4.8 GB).
        // The pool doubles when sync_genome_mode_data detects it is too small, up to
        // MAX_TOTAL_MODES (8_000_000) which is the hard cap used by the mutation ring buffer.
        const INITIAL_MODE_POOL_SIZE: u64 = 16_384;
        let max_modes = INITIAL_MODE_POOL_SIZE;
        let genome_mode_data_v0 =
            Self::create_storage_buffer(device, max_modes * 16, "Genome Mode Data V0");
        let genome_mode_data_v1 =
            Self::create_storage_buffer(device, max_modes * 16, "Genome Mode Data V1");
        let genome_mode_data_v2 =
            Self::create_storage_buffer(device, max_modes * 16, "Genome Mode Data V2");
        let genome_mode_data_v3 =
            Self::create_storage_buffer(device, max_modes * 16, "Genome Mode Data V3");
        let genome_mode_data_v4 =
            Self::create_storage_buffer(device, max_modes * 16, "Genome Mode Data V4");

        // Child mode indices: two i32 per mode (child_a_mode, child_b_mode)
        let child_mode_indices =
            Self::create_storage_buffer(device, max_modes * 8, "Child Mode Indices");

        // Is-initial-mode flag: one u32 per mode (1 = this is the genome's initial mode)
        let is_initial_mode = Self::create_storage_buffer(device, max_modes * 4, "Is Initial Mode");

        // Parent make adhesion flags: one u32 per mode
        let parent_make_adhesion_flags =
            Self::create_storage_buffer(device, max_modes * 4, "Parent Make Adhesion Flags");

        // Child A keep adhesion flags: one u32 per mode
        let child_a_keep_adhesion_flags =
            Self::create_storage_buffer(device, max_modes * 4, "Child A Keep Adhesion Flags");

        // Child B keep adhesion flags: one u32 per mode
        let child_b_keep_adhesion_flags =
            Self::create_storage_buffer(device, max_modes * 4, "Child B Keep Adhesion Flags");

        // Child A after-split keep adhesion flags: one u32 per mode
        let child_a_after_split_keep_adhesion_flags = Self::create_storage_buffer(
            device,
            max_modes * 4,
            "Child A After Split Keep Adhesion Flags",
        );

        // Child B after-split keep adhesion flags: one u32 per mode
        let child_b_after_split_keep_adhesion_flags = Self::create_storage_buffer(
            device,
            max_modes * 4,
            "Child B After Split Keep Adhesion Flags",
        );

        // Mode properties split into 5 vec4 sub-buffers (16 bytes each)
        let mode_properties_v0 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V0");
        let mode_properties_v1 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V1");
        let mode_properties_v2 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V2");
        let mode_properties_v3 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V3");
        let mode_properties_v4 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V4");

        // Mode properties v5-v6 for cilia parameters (16 bytes per mode each)
        let mode_properties_v5 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V5");
        let mode_properties_v6 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V6");

        // Mode properties v7-v8 for myocyte parameters (16 bytes per mode each)
        let mode_properties_v7 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V7");
        let mode_properties_v8 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V8");

        // Mode properties v9-v10 for Embryocyte parameters (16 bytes per mode each)
        let mode_properties_v9 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V9");
        let mode_properties_v10 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V10");

        // Mode properties v11 for Devorocyte parameters (16 bytes per mode each)
        let mode_properties_v11 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V11");

        // Mode properties v12 for Vasculocyte parameters (16 bytes per mode each)
        let mode_properties_v12 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V12");

        // Mode properties v13 for Gametocyte parameters (16 bytes per mode each)
        let mode_properties_v13 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V13");
        let mode_properties_v14 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V14");
        let mode_properties_v15 =
            Self::create_storage_buffer(device, max_modes * 16, "Mode Properties V15");

        // Per-cell Embryocyte reserve buffer (one u32 per cell, zero-initialized)
        let embryocyte_reserve_buffer = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 4, // 4 bytes per u32
            "Embryocyte Reserve Buffer",
        );

        // Mode cell types: one u32 per mode - lookup table for deriving cell_type from mode_index
        let mode_cell_types = Self::create_storage_buffer(device, max_modes * 4, "Mode Cell Types");

        // Behavior flags per cell type for parameterized shader logic
        // Each GpuCellTypeBehaviorFlags struct is 64 bytes (6 u32 fields + 10 u32 padding)
        let behavior_flags = Self::create_storage_buffer(device, 30 * 64, "Behavior Flags"); // CellType::MAX_TYPES = 30

        // Per-cell env anchor buffer: Vec4(anchor_x, anchor_y, anchor_z, is_active)
        // Zero-initialized so all cells start with no anchor (w=0.0)
        let env_anchor_buffer = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 16, // vec4<f32> = 16 bytes
            "Env Anchor Buffer",
        );

        // Per-mode glueocyte env adhesion flags (one u32 per mode)
        let glueocyte_env_adhesion_flags =
            Self::create_storage_buffer(device, max_modes * 4, "Glueocyte Env Adhesion Flags");
        // Per-mode glueocyte boulder adhesion flags (one u32 per mode)
        let glueocyte_boulder_adhesion_flags =
            Self::create_storage_buffer(device, max_modes * 4, "Glueocyte Boulder Adhesion Flags");

        // Per-mode glueocyte cell-adhesion flags: 4 u32 per mode
        // [0]=enabled, [1]=signal_channel (0xFFFFFFFF=always active), [2]=signal_threshold bits, [3]=self_adhesion
        let glueocyte_cell_adhesion_flags =
            Self::create_storage_buffer(device, max_modes * 16, "Glueocyte Cell Adhesion Flags");

        // Per-cell muscle contraction value (one f32 per cell, zero-initialized = relaxed)
        let muscle_contraction_buffer = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 4, // f32 = 4 bytes per cell
            "Muscle Contraction Buffer",
        );

        // Per-cell grip (friction drag) value, written by muscle_contraction shader alongside contraction.
        // Zero = no friction contribution. Read by position_update shader.
        let cell_grip_buffer = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 4, // f32 = 4 bytes per cell
            "Cell Grip Buffer",
        );

        let cell_water =
            Self::create_storage_buffer(device, capacity as u64 * 4, "Cell Water Buffer");
        let cell_heat_energy =
            Self::create_storage_buffer(device, capacity as u64 * 4, "Cell Heat Energy Buffer");
        let cell_cached_temperature = Self::create_storage_buffer(
            device,
            capacity as u64 * 4,
            "Cell Cached Temperature Buffer",
        );
        let cell_thermal_state =
            Self::create_storage_buffer(device, capacity as u64 * 4, "Cell Thermal State Buffer");
        let cell_water_next =
            Self::create_storage_buffer(device, capacity as u64 * 4, "Cell Water Next Buffer");
        let cell_heat_energy_next = Self::create_storage_buffer(
            device,
            capacity as u64 * 4,
            "Cell Heat Energy Next Buffer",
        );
        let cell_cached_temperature_next = Self::create_storage_buffer(
            device,
            capacity as u64 * 4,
            "Cell Cached Temperature Next Buffer",
        );
        let cell_thermal_state_next = Self::create_storage_buffer(
            device,
            capacity as u64 * 4,
            "Cell Thermal State Next Buffer",
        );
        let cell_prev_muscle_contraction = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 4,
            "Cell Previous Muscle Contraction Buffer",
        );

        // Per-mode oculocyte parameters: vec4<u32> per mode (sense_type, sense_range_bits, signal_hops, signal_channel)
        let oculocyte_params =
            Self::create_storage_buffer(device, max_modes * 16, "Oculocyte Params");

        // Per-mode oculocyte signal values: one f32 per mode (value emitted when target detected)
        let oculocyte_signal_values =
            Self::create_storage_buffer(device, max_modes * 4, "Oculocyte Signal Values");

        let oculocyte_light_filters =
            Self::create_storage_buffer(device, max_modes * 16, "Oculocyte Light Filters");

        // Per-mode regulation emission parameters: vec4<u32> per mode (emit_channel, emit_value_bits, emit_hops, padding)
        let regulation_params =
            Self::create_storage_buffer(device, max_modes * 16, "Regulation Params");

        // Per-mode signal-conditional settings: 5 vec4<f32> sub-buffers per mode
        let signal_settings_v0 =
            Self::create_storage_buffer(device, max_modes * 16, "Signal Settings V0");
        let signal_settings_v1 =
            Self::create_storage_buffer(device, max_modes * 16, "Signal Settings V1");
        let signal_settings_v2 =
            Self::create_storage_buffer(device, max_modes * 16, "Signal Settings V2");
        let signal_settings_v3 =
            Self::create_storage_buffer(device, max_modes * 16, "Signal Settings V3");
        let signal_settings_v4 =
            Self::create_storage_buffer(device, max_modes * 16, "Signal Settings V4");

        // Per-cell genome orientation: vec4<f32> quaternion (identity = 0,0,0,1)
        // Tracks pure genome-derived orientation without physics perturbation
        let genome_orientations =
            Self::create_storage_buffer(device, buffer_size, "Genome Orientations");

        let development_addresses = Self::create_zero_initialized_storage_buffer(
            device,
            buffer_size,
            "Development Addresses",
        );
        let organism_cell_ids = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 4,
            "Organism Cell IDs",
        );

        // 8 bytes per cell: [parent_lineage_hash_lo: u32, parent_lineage_hash_hi: u32]
        let parent_lineage_hashes = Self::create_zero_initialized_storage_buffer(
            device,
            capacity as u64 * 8,
            "Parent Lineage Hashes",
        );

        // Indirect dispatch buffer: 3 x u32 for workgroup counts + padding
        // Used for GPU-driven dispatch that scales with actual cell count
        let indirect_dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Dispatch Buffer"),
            size: 16, // 4 x u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            nutrients_buffer,
            mass_deltas_buffer,
            division_flags,
            free_slot_indices,
            division_slot_assignments,
            lifecycle_counts,
            free_slot_ring,
            ring_state,
            birth_times,
            mode_switch_time,
            split_intervals,
            split_nutrient_thresholds,
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
            genome_mode_data_v0,
            genome_mode_data_v1,
            genome_mode_data_v2,
            genome_mode_data_v3,
            genome_mode_data_v4,
            child_mode_indices,
            is_initial_mode,
            parent_make_adhesion_flags,
            child_a_keep_adhesion_flags,
            child_b_keep_adhesion_flags,
            child_a_after_split_keep_adhesion_flags,
            child_b_after_split_keep_adhesion_flags,
            mode_properties_v0,
            mode_properties_v1,
            mode_properties_v2,
            mode_properties_v3,
            mode_properties_v4,
            mode_properties_v5,
            mode_properties_v6,
            mode_properties_v7,
            mode_properties_v8,
            mode_properties_v9,
            mode_properties_v10,
            mode_properties_v11,
            mode_properties_v12,
            mode_properties_v13,
            mode_properties_v14,
            mode_properties_v15,
            embryocyte_reserve_buffer,
            mode_cell_types,
            behavior_flags,
            env_anchor_buffer,
            muscle_contraction_buffer,
            cell_grip_buffer,
            cell_water,
            cell_heat_energy,
            cell_cached_temperature,
            cell_thermal_state,
            cell_water_next,
            cell_heat_energy_next,
            cell_cached_temperature_next,
            cell_thermal_state_next,
            cell_prev_muscle_contraction,
            glueocyte_env_adhesion_flags,
            glueocyte_boulder_adhesion_flags,
            glueocyte_cell_adhesion_flags,
            oculocyte_params,
            oculocyte_signal_values,
            oculocyte_light_filters,
            regulation_params,
            signal_settings_v0,
            signal_settings_v1,
            signal_settings_v2,
            signal_settings_v3,
            signal_settings_v4,
            genome_orientations,
            development_addresses,
            organism_cell_ids,
            parent_lineage_hashes,
            indirect_dispatch_buffer,
            current_index: AtomicUsize::new(0),
            capacity,
            mode_pool_capacity: max_modes,
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

    /// Create a storage buffer that is zero-initialized
    /// Used for buffers like death_flags that need deterministic initial state
    fn create_zero_initialized_storage_buffer(
        device: &wgpu::Device,
        size: u64,
        label: &str,
    ) -> wgpu::Buffer {
        // Align size to 16-byte boundary for GPU compatibility
        let aligned_size = (size + 15) & !15; // Round up to nearest 16 bytes

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        // Zero-initialize the buffer
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            view.fill(0);
        }
        buffer.unmap();

        buffer
    }

    /// Create a storage buffer that is initialized with 0xFFFFFFFFu (max u32)
    /// Used for division_slot_assignments to ensure sentinel values >= cell_capacity
    pub fn create_max_u32_initialized_storage_buffer(
        device: &wgpu::Device,
        size: u64,
        label: &str,
    ) -> wgpu::Buffer {
        // Align size to 16-byte boundary for GPU compatibility
        let aligned_size = (size + 15) & !15; // Round up to nearest 16 bytes

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: aligned_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        // Initialize with 0xFFFFFFFFu (max u32)
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            // Fill with 0xFF bytes (which creates 0xFFFFFFFFu values)
            view.fill(0xFF);
        }
        buffer.unmap();

        buffer
    }

    /// Upload canonical state data to all GPU buffer sets
    /// WARNING: This overwrites ALL GPU positions with CPU data.
    /// Only use for initial setup, not during simulation.
    pub fn sync_from_canonical_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        genomes: &[crate::genome::Genome],
    ) {
        // Grow mode pool if needed before any genome data sync
        let total_modes: u64 = genomes.iter().map(|g| g.modes.len() as u64).sum();
        if total_modes > 0 {
            self.grow_mode_pool_if_needed(device, total_modes);
        }

        // Sync slot allocator with canonical state
        self.sync_slot_allocator_with_canonical(state);

        // Always update GPU cell count buffer, even when cell_count is 0
        let cell_counts: [u32; 2] = [state.cell_count as u32, state.cell_count as u32];
        queue.write_buffer(
            &self.cell_count_buffer,
            0,
            bytemuck::cast_slice(&cell_counts),
        );

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

        // Build genome orientation data: Vec4(x, y, z, w) quaternion
        let mut genome_orientation_data: Vec<[f32; 4]> = Vec::with_capacity(state.cell_count);
        for i in 0..state.cell_count {
            let q = state.genome_orientations[i];
            genome_orientation_data.push([q.x, q.y, q.z, q.w]);
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

        // Upload genome orientations (single buffer, not triple-buffered)
        queue.write_buffer(
            &self.genome_orientations,
            0,
            bytemuck::cast_slice(&genome_orientation_data),
        );

        self.sync_development_addresses(queue, state);

        // Sync cell state buffers for division system
        self.sync_cell_state_buffers(queue, state, genomes);

        self.needs_sync = false;
    }

    /// Sync cell state buffers (birth times, split intervals, etc.) for division system
    pub fn sync_cell_state_buffers(
        &self,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        genomes: &[crate::genome::Genome],
    ) {
        if state.cell_count == 0 {
            return;
        }

        // Birth times
        let birth_times: Vec<f32> = state.birth_times[..state.cell_count].to_vec();
        queue.write_buffer(&self.birth_times, 0, bytemuck::cast_slice(&birth_times));

        // Split intervals
        let split_intervals: Vec<f32> = state.split_intervals[..state.cell_count].to_vec();
        queue.write_buffer(
            &self.split_intervals,
            0,
            bytemuck::cast_slice(&split_intervals),
        );

        // Split nutrient thresholds
        let split_nutrient_thresholds: Vec<f32> =
            state.split_nutrient_thresholds[..state.cell_count].to_vec();
        queue.write_buffer(
            &self.split_nutrient_thresholds,
            0,
            bytemuck::cast_slice(&split_nutrient_thresholds),
        );

        // Nutrients - convert f32 to i32 fixed-point (scale 1000: 100.0 nutrients = 100000)
        let nutrients: Vec<i32> = state.nutrients[..state.cell_count]
            .iter()
            .map(|&n| (n * 1000.0) as i32)
            .collect();
        queue.write_buffer(&self.nutrients_buffer, 0, bytemuck::cast_slice(&nutrients));

        let cell_water = &state.cell_water[..state.cell_count];
        let cell_heat_energy = &state.cell_heat_energy[..state.cell_count];
        let cell_cached_temperature = &state.cell_cached_temperature[..state.cell_count];
        let cell_thermal_state: Vec<u32> = state.cell_thermal_state[..state.cell_count]
            .iter()
            .map(|&s| s as u32)
            .collect();
        queue.write_buffer(&self.cell_water, 0, bytemuck::cast_slice(cell_water));
        queue.write_buffer(
            &self.cell_heat_energy,
            0,
            bytemuck::cast_slice(cell_heat_energy),
        );
        queue.write_buffer(
            &self.cell_cached_temperature,
            0,
            bytemuck::cast_slice(cell_cached_temperature),
        );
        queue.write_buffer(
            &self.cell_thermal_state,
            0,
            bytemuck::cast_slice(&cell_thermal_state),
        );
        queue.write_buffer(&self.cell_water_next, 0, bytemuck::cast_slice(cell_water));
        queue.write_buffer(
            &self.cell_heat_energy_next,
            0,
            bytemuck::cast_slice(cell_heat_energy),
        );
        queue.write_buffer(
            &self.cell_cached_temperature_next,
            0,
            bytemuck::cast_slice(cell_cached_temperature),
        );
        queue.write_buffer(
            &self.cell_thermal_state_next,
            0,
            bytemuck::cast_slice(&cell_thermal_state),
        );

        // Split counts
        let split_counts: Vec<u32> = state.split_counts[..state.cell_count]
            .iter()
            .map(|&x| x as u32)
            .collect();
        queue.write_buffer(&self.split_counts, 0, bytemuck::cast_slice(&split_counts));

        // Split ready frame - initialize to -1 (not ready to split)
        let split_ready_frame: Vec<i32> = vec![-1; state.cell_count];
        queue.write_buffer(
            &self.split_ready_frame,
            0,
            bytemuck::cast_slice(&split_ready_frame),
        );

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
                    max_splits_data.push(if ms < 0 { u32::MAX } else { ms as u32 });
                    // Only Test cells (cell_type == 0) auto-gain nutrients on GPU
                    // Phagocytes and Photocytes use specialized shaders (phagocyte_consume, photocyte_light)
                    let nutrient_rate = if mode.cell_type == 0 {
                        mode.nutrient_gain_rate
                    } else {
                        0.0
                    };
                    nutrient_gain_rates_data.push(nutrient_rate);
                    max_cell_sizes_data.push(mode.max_cell_size);
                    stiffnesses_data.push(mode.membrane_stiffness);
                } else {
                    max_splits_data.push(u32::MAX); // Unlimited if mode not found
                    nutrient_gain_rates_data.push(20.0); // Default nutrient gain rate (20 nutrients/sec)
                    max_cell_sizes_data.push(2.0); // Default max cell size
                    stiffnesses_data.push(50.0); // Default membrane stiffness
                }
            } else {
                max_splits_data.push(u32::MAX); // Unlimited if genome not found
                nutrient_gain_rates_data.push(20.0); // Default nutrient gain rate (20 nutrients/sec)
                max_cell_sizes_data.push(2.0); // Default max cell size
                stiffnesses_data.push(50.0); // Default membrane stiffness
            }
        }
        queue.write_buffer(&self.max_splits, 0, bytemuck::cast_slice(&max_splits_data));
        queue.write_buffer(
            &self.nutrient_gain_rates,
            0,
            bytemuck::cast_slice(&nutrient_gain_rates_data),
        );
        queue.write_buffer(
            &self.max_cell_sizes,
            0,
            bytemuck::cast_slice(&max_cell_sizes_data),
        );
        queue.write_buffer(
            &self.stiffnesses,
            0,
            bytemuck::cast_slice(&stiffnesses_data),
        );

        // Genome IDs
        let genome_ids: Vec<u32> = state.genome_ids[..state.cell_count]
            .iter()
            .map(|&x| x as u32)
            .collect();
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
                        2 // Default to Phagocyte cell type
                    }
                } else {
                    2 // Default to Phagocyte cell type
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
        let cell_ids: Vec<u32> = state.cell_ids[..state.cell_count]
            .iter()
            .map(|&x| x as u32)
            .collect();
        queue.write_buffer(&self.cell_ids, 0, bytemuck::cast_slice(&cell_ids));

        // Next cell ID
        let next_id: [u32; 1] = [state.next_cell_id as u32];
        queue.write_buffer(&self.next_cell_id, 0, bytemuck::cast_slice(&next_id));

        // Initialize death flags to 0 (all alive)
        let death_flags: Vec<u32> = vec![0u32; state.cell_count];
        queue.write_buffer(&self.death_flags, 0, bytemuck::cast_slice(&death_flags));

        // Initialize mass deltas to 0 (no transfers)
        let mass_deltas: Vec<f32> = vec![0.0f32; state.cell_count];
        queue.write_buffer(
            &self.mass_deltas_buffer,
            0,
            bytemuck::cast_slice(&mass_deltas),
        );

        // Initialize division flags to 0
        let division_flags: Vec<u32> = vec![0u32; state.cell_count];
        queue.write_buffer(
            &self.division_flags,
            0,
            bytemuck::cast_slice(&division_flags),
        );

        // Sync genome mode data for division (child orientations)
        self.sync_genome_mode_data(queue, genomes);

        // Sync child mode indices for division
        self.sync_child_mode_indices(queue, genomes);

        // Sync mode properties for division (nutrient_gain_rate, max_cell_size, etc.)
        self.sync_mode_properties(queue, genomes);

        // Sync cilia mode properties for cilia_force shader (v5, v6)
        self.sync_cilia_mode_properties(queue, genomes);
        self.sync_siphonocyte_mode_properties(queue, genomes);
        self.sync_plumocyte_mode_properties(queue, genomes);

        // Sync mode cell types lookup table (for deriving cell_type from mode_index)
        self.sync_mode_cell_types(queue, genomes);

        // Sync glueocyte env adhesion flags (one u32 per mode across all genomes)
        self.sync_glueocyte_env_adhesion_flags(queue, genomes);
        self.sync_glueocyte_boulder_adhesion_flags(queue, genomes);

        // Sync behavior flags for all cell types (applies_swim_force, etc.)
        self.sync_behavior_flags(queue);
    }

    pub fn sync_development_addresses(&self, queue: &wgpu::Queue, state: &CanonicalState) {
        if state.cell_count == 0 {
            return;
        }

        let data: Vec<[u32; 4]> = (0..state.cell_count)
            .map(|i| {
                let lineage_hash = state.lineage_hashes[i];
                [
                    state.organism_ids[i],
                    lineage_hash as u32,
                    (lineage_hash >> 32) as u32,
                    ((state.lineage_depths[i] as u32) << 16) | state.lineage_branch_slots[i] as u32,
                ]
            })
            .collect();
        queue.write_buffer(&self.development_addresses, 0, bytemuck::cast_slice(&data));

        let organism_cell_ids: Vec<u32> = state.organism_cell_ids[..state.cell_count].to_vec();
        queue.write_buffer(
            &self.organism_cell_ids,
            0,
            bytemuck::cast_slice(&organism_cell_ids),
        );

        let parent_data: Vec<[u32; 2]> = (0..state.cell_count)
            .map(|i| {
                let ph = state.parent_lineage_hashes[i];
                [ph as u32, (ph >> 32) as u32]
            })
            .collect();
        queue.write_buffer(
            &self.parent_lineage_hashes,
            0,
            bytemuck::cast_slice(&parent_data),
        );
    }

    pub fn sync_single_development_address(
        &self,
        queue: &wgpu::Queue,
        cell_idx: usize,
        organism_id: u32,
        lineage_hash: u64,
        lineage_depth: u16,
        branch_slot: u16,
        organism_cell_id: u32,
    ) {
        let data = [
            organism_id,
            lineage_hash as u32,
            (lineage_hash >> 32) as u32,
            ((lineage_depth as u32) << 16) | branch_slot as u32,
        ];
        queue.write_buffer(
            &self.development_addresses,
            (cell_idx * 16) as u64,
            bytemuck::bytes_of(&data),
        );
        queue.write_buffer(
            &self.organism_cell_ids,
            (cell_idx * 4) as u64,
            bytemuck::bytes_of(&organism_cell_id),
        );
    }

    fn mix_development_hash(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn development_root_hash(
        seed: u64,
        _organism_id: u32,
        genome_id: usize,
        mode_index: usize,
    ) -> u64 {
        Self::mix_development_hash(seed ^ ((genome_id as u64) << 16) ^ mode_index as u64)
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
                        2 // Default to Phagocyte cell type
                    }
                } else {
                    2 // Default to Phagocyte cell type
                }
            })
            .collect();

        queue.write_buffer(&self.cell_types, 0, bytemuck::cast_slice(&cell_types_data));
    }

    /// Grow all mode pool sub-buffers if the current pool is too small for `total_modes`.
    ///
    /// Call this before any sync that writes genome data to the GPU. The pool doubles
    /// until it fits `total_modes`, capped at `MAX_TOTAL_MODES` (8,000,000).
    /// All existing GPU data is lost on resize - callers must re-sync everything
    /// after calling this (which they do anyway, since this is called at sync time).
    ///
    /// Returns `true` if the pool was grown (bind groups that reference mode buffers
    /// must be rebuilt by the caller).
    pub fn grow_mode_pool_if_needed(&mut self, device: &wgpu::Device, total_modes: u64) -> bool {
        use crate::simulation::gpu_physics::mutation::MAX_TOTAL_MODES;
        let hard_cap = MAX_TOTAL_MODES as u64;

        if total_modes <= self.mode_pool_capacity {
            return false; // Already large enough
        }

        // Double until we fit, capped at the hard limit
        let mut new_capacity = self.mode_pool_capacity;
        while new_capacity < total_modes {
            new_capacity = (new_capacity * 2).min(hard_cap);
        }

        log::info!(
            "Growing mode pool: {} → {} modes ({:.1} MB → {:.1} MB across ~34 sub-buffers)",
            self.mode_pool_capacity,
            new_capacity,
            self.mode_pool_capacity as f64 * 34.0 * 16.0 / 1_048_576.0,
            new_capacity as f64 * 34.0 * 16.0 / 1_048_576.0,
        );

        // Reallocate every mode-indexed sub-buffer at the new capacity.
        // Buffers that are per-cell (position_and_mass, velocity, etc.) are NOT touched.
        let m16 = new_capacity * 16; // 16 bytes per mode (vec4)
        let m8 = new_capacity * 8; // 8 bytes per mode (2xi32)
        let m4 = new_capacity * 4; // 4 bytes per mode (1xu32)

        self.genome_mode_data_v0 = Self::create_storage_buffer(device, m16, "Genome Mode Data V0");
        self.genome_mode_data_v1 = Self::create_storage_buffer(device, m16, "Genome Mode Data V1");
        self.genome_mode_data_v2 = Self::create_storage_buffer(device, m16, "Genome Mode Data V2");
        self.genome_mode_data_v3 = Self::create_storage_buffer(device, m16, "Genome Mode Data V3");
        self.genome_mode_data_v4 = Self::create_storage_buffer(device, m16, "Genome Mode Data V4");

        self.child_mode_indices = Self::create_storage_buffer(device, m8, "Child Mode Indices");
        self.is_initial_mode = Self::create_storage_buffer(device, m4, "Is Initial Mode");
        self.parent_make_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Parent Make Adhesion Flags");
        self.child_a_keep_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Child A Keep Adhesion Flags");
        self.child_b_keep_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Child B Keep Adhesion Flags");
        self.child_a_after_split_keep_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Child A After Split Keep Adhesion Flags");
        self.child_b_after_split_keep_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Child B After Split Keep Adhesion Flags");

        self.mode_properties_v0 = Self::create_storage_buffer(device, m16, "Mode Properties V0");
        self.mode_properties_v1 = Self::create_storage_buffer(device, m16, "Mode Properties V1");
        self.mode_properties_v2 = Self::create_storage_buffer(device, m16, "Mode Properties V2");
        self.mode_properties_v3 = Self::create_storage_buffer(device, m16, "Mode Properties V3");
        self.mode_properties_v4 = Self::create_storage_buffer(device, m16, "Mode Properties V4");
        self.mode_properties_v5 = Self::create_storage_buffer(device, m16, "Mode Properties V5");
        self.mode_properties_v6 = Self::create_storage_buffer(device, m16, "Mode Properties V6");
        self.mode_properties_v7 = Self::create_storage_buffer(device, m16, "Mode Properties V7");
        self.mode_properties_v8 = Self::create_storage_buffer(device, m16, "Mode Properties V8");
        self.mode_properties_v9 = Self::create_storage_buffer(device, m16, "Mode Properties V9");
        self.mode_properties_v10 = Self::create_storage_buffer(device, m16, "Mode Properties V10");
        self.mode_properties_v11 = Self::create_storage_buffer(device, m16, "Mode Properties V11");
        self.mode_properties_v12 = Self::create_storage_buffer(device, m16, "Mode Properties V12");
        self.mode_properties_v13 = Self::create_storage_buffer(device, m16, "Mode Properties V13");
        self.mode_properties_v14 = Self::create_storage_buffer(device, m16, "Mode Properties V14");
        self.mode_properties_v15 = Self::create_storage_buffer(device, m16, "Mode Properties V15");

        self.mode_cell_types = Self::create_storage_buffer(device, m4, "Mode Cell Types");
        self.glueocyte_env_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Glueocyte Env Adhesion Flags");
        self.glueocyte_boulder_adhesion_flags =
            Self::create_storage_buffer(device, m4, "Glueocyte Boulder Adhesion Flags");
        self.glueocyte_cell_adhesion_flags =
            Self::create_storage_buffer(device, m16, "Glueocyte Cell Adhesion Flags");
        self.oculocyte_params = Self::create_storage_buffer(device, m16, "Oculocyte Params");
        self.oculocyte_signal_values =
            Self::create_storage_buffer(device, m4, "Oculocyte Signal Values");
        self.oculocyte_light_filters =
            Self::create_storage_buffer(device, m16, "Oculocyte Light Filters");
        self.regulation_params = Self::create_storage_buffer(device, m16, "Regulation Params");

        self.signal_settings_v0 = Self::create_storage_buffer(device, m16, "Signal Settings V0");
        self.signal_settings_v1 = Self::create_storage_buffer(device, m16, "Signal Settings V1");
        self.signal_settings_v2 = Self::create_storage_buffer(device, m16, "Signal Settings V2");
        self.signal_settings_v3 = Self::create_storage_buffer(device, m16, "Signal Settings V3");
        self.signal_settings_v4 = Self::create_storage_buffer(device, m16, "Signal Settings V4");

        self.mode_pool_capacity = new_capacity;
        true
    }

    /// Sync genome mode data (child orientations and split orientations) for division shader.
    /// Data is written into 5 separate vec4 sub-buffers (v0..v4), each 16 bytes/mode.
    pub fn sync_genome_mode_data(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut v0: Vec<[f32; 4]> = Vec::new(); // child_a_orientation
        let mut v1: Vec<[f32; 4]> = Vec::new(); // child_b_orientation
        let mut v2: Vec<[f32; 4]> = Vec::new(); // child_a_split_orientation
        let mut v3: Vec<[f32; 4]> = Vec::new(); // child_b_split_orientation
        let mut v4: Vec<[f32; 4]> = Vec::new(); // split_rotation_quat (XYZW)

        for genome in genomes {
            for mode in &genome.modes {
                let qa = mode.child_a.orientation;
                let qb = mode.child_b.orientation;
                let qa_split = mode.child_a_after_split_orientation;
                let qb_split = mode.child_b_after_split_orientation;
                let pitch = mode.parent_split_direction.x.to_radians();
                let yaw = mode.parent_split_direction.y.to_radians();
                let split_quat = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
                v0.push([qa.x, qa.y, qa.z, qa.w]);
                v1.push([qb.x, qb.y, qb.z, qb.w]);
                v2.push([qa_split.x, qa_split.y, qa_split.z, qa_split.w]);
                v3.push([qb_split.x, qb_split.y, qb_split.z, qb_split.w]);
                v4.push([split_quat.x, split_quat.y, split_quat.z, split_quat.w]);
            }
        }

        if !v0.is_empty() {
            queue.write_buffer(&self.genome_mode_data_v0, 0, bytemuck::cast_slice(&v0));
            queue.write_buffer(&self.genome_mode_data_v1, 0, bytemuck::cast_slice(&v1));
            queue.write_buffer(&self.genome_mode_data_v2, 0, bytemuck::cast_slice(&v2));
            queue.write_buffer(&self.genome_mode_data_v3, 0, bytemuck::cast_slice(&v3));
            queue.write_buffer(&self.genome_mode_data_v4, 0, bytemuck::cast_slice(&v4));
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
            queue.write_buffer(
                &self.child_mode_indices,
                0,
                bytemuck::cast_slice(&mode_indices),
            );
        }
    }

    /// Sync is-initial-mode flags for the division shader.
    /// Sets 1 for the mode at `genome.initial_mode` index, 0 for all others.
    pub fn sync_is_initial_mode(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut data: Vec<u32> = Vec::new();
        for genome in genomes {
            let initial_mode = genome.initial_mode.max(0) as usize;
            for mode_idx in 0..genome.modes.len() {
                data.push(if mode_idx == initial_mode { 1u32 } else { 0u32 });
            }
        }
        if !data.is_empty() {
            queue.write_buffer(&self.is_initial_mode, 0, bytemuck::cast_slice(&data));
        }
    }

    /// Sync parent make adhesion flags for division shader
    pub fn sync_parent_make_adhesion_flags(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut flags_data: Vec<u32> = Vec::new();

        for genome in genomes {
            for mode in &genome.modes {
                flags_data.push(if mode.parent_make_adhesion { 1 } else { 0 });
            }
        }

        if !flags_data.is_empty() {
            queue.write_buffer(
                &self.parent_make_adhesion_flags,
                0,
                bytemuck::cast_slice(&flags_data),
            );
        }
    }

    /// Sync child keep adhesion flags for division shader (zone-based inheritance)
    pub fn sync_child_keep_adhesion_flags(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut child_a_flags: Vec<u32> = Vec::new();
        let mut child_b_flags: Vec<u32> = Vec::new();
        let mut child_a_after_split_flags: Vec<u32> = Vec::new();
        let mut child_b_after_split_flags: Vec<u32> = Vec::new();

        for genome in genomes {
            for mode in &genome.modes {
                child_a_flags.push(if mode.child_a.keep_adhesion { 1 } else { 0 });
                child_b_flags.push(if mode.child_b.keep_adhesion { 1 } else { 0 });
                child_a_after_split_flags.push(if mode.child_a_after_split_keep_adhesion {
                    1
                } else {
                    0
                });
                child_b_after_split_flags.push(if mode.child_b_after_split_keep_adhesion {
                    1
                } else {
                    0
                });
            }
        }

        if !child_a_flags.is_empty() {
            queue.write_buffer(
                &self.child_a_keep_adhesion_flags,
                0,
                bytemuck::cast_slice(&child_a_flags),
            );
            queue.write_buffer(
                &self.child_b_keep_adhesion_flags,
                0,
                bytemuck::cast_slice(&child_b_flags),
            );
            queue.write_buffer(
                &self.child_a_after_split_keep_adhesion_flags,
                0,
                bytemuck::cast_slice(&child_a_after_split_flags),
            );
            queue.write_buffer(
                &self.child_b_after_split_keep_adhesion_flags,
                0,
                bytemuck::cast_slice(&child_b_after_split_flags),
            );
        }
    }

    /// Sync mode properties for division shader.
    /// Data is written into 5 separate vec4 sub-buffers (v0..v4), each 16 bytes/mode.
    pub fn sync_mode_properties(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut v0: Vec<[f32; 4]> = Vec::new(); // [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
        let mut v1: Vec<[f32; 4]> = Vec::new(); // [split_mass, nutrient_priority, swim_force, prioritize_when_low]
        let mut v2: Vec<[f32; 4]> = Vec::new(); // [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
        let mut v3: Vec<[f32; 4]> = Vec::new(); // [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
        let mut v4: Vec<[f32; 4]> = Vec::new(); // [max_adhesions, mode_a_after_splits, mode_b_after_splits, padding]

        let mut global_mode_idx = 0usize;
        let mut global_mode_offset = 0i32;
        for (genome_idx, genome) in genomes.iter().enumerate() {
            for (mode_idx, mode) in genome.modes.iter().enumerate() {
                log::debug!("[SYNC MODE PROPS] genome={} mode={} global={} nutrient_priority={} prioritize_when_low={} cell_type={:?}",
                    genome_idx, mode_idx, global_mode_idx, mode.nutrient_priority, mode.prioritize_when_low, mode.cell_type);
                global_mode_idx += 1;
                let gpu_max_splits = if mode.max_splits < 0 {
                    -1.0
                } else {
                    mode.max_splits as f32
                };
                let gpu_mode_a_after = if mode.mode_a_after_splits < 0 {
                    -1.0
                } else {
                    (global_mode_offset + mode.mode_a_after_splits.max(0)) as f32
                };
                let gpu_mode_b_after = if mode.mode_b_after_splits < 0 {
                    -1.0
                } else {
                    (global_mode_offset + mode.mode_b_after_splits.max(0)) as f32
                };
                v0.push([
                    mode.nutrient_gain_rate,
                    mode.max_cell_size,
                    mode.membrane_stiffness,
                    mode.split_interval,
                ]);
                v1.push([
                    mode.split_mass,
                    mode.nutrient_priority,
                    mode.swim_force,
                    if mode.prioritize_when_low { 1.0 } else { 0.0 },
                ]);
                v2.push([
                    gpu_max_splits,
                    mode.split_ratio,
                    mode.flagellocyte_signal_channel as f32,
                    mode.flagellocyte_speed_a,
                ]);
                v3.push([
                    mode.flagellocyte_speed_b,
                    mode.flagellocyte_threshold_c,
                    if mode.flagellocyte_use_signal {
                        1.0
                    } else {
                        0.0
                    },
                    mode.min_adhesions as f32,
                ]);
                v4.push([
                    mode.max_adhesions as f32,
                    gpu_mode_a_after,
                    gpu_mode_b_after,
                    mode.buoyancy_force,
                ]);
            }
            global_mode_offset += genome.modes.len() as i32;
        }

        if !v0.is_empty() {
            queue.write_buffer(&self.mode_properties_v0, 0, bytemuck::cast_slice(&v0));
            queue.write_buffer(&self.mode_properties_v1, 0, bytemuck::cast_slice(&v1));
            queue.write_buffer(&self.mode_properties_v2, 0, bytemuck::cast_slice(&v2));
            queue.write_buffer(&self.mode_properties_v3, 0, bytemuck::cast_slice(&v3));
            queue.write_buffer(&self.mode_properties_v4, 0, bytemuck::cast_slice(&v4));
        }
    }

    /// Sync cilia mode properties for the cilia_force shader.
    /// Data is written into 2 separate vec4 sub-buffers (v5, v6), each 16 bytes/mode.
    /// v5: [cilia_speed, cilia_push_bonded as f32, cilia_use_signal as f32, cilia_signal_channel as f32]
    /// v6: [cilia_speed_below, cilia_speed_above, cilia_threshold, cilia_attract_force]
    pub fn sync_cilia_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v5: Vec<[f32; 4]> = Vec::new();
        let mut v6: Vec<[f32; 4]> = Vec::new();

        for genome in genomes {
            for mode in &genome.modes {
                v5.push([
                    mode.cilia_speed,
                    if mode.cilia_push_bonded { 1.0 } else { 0.0 },
                    if mode.cilia_use_signal { 1.0 } else { 0.0 },
                    mode.cilia_signal_channel as f32,
                ]);
                v6.push([
                    mode.cilia_speed_below,
                    mode.cilia_speed_above,
                    mode.cilia_threshold,
                    mode.cilia_attract_force,
                ]);
            }
        }

        if !v5.is_empty() {
            queue.write_buffer(&self.mode_properties_v5, 0, bytemuck::cast_slice(&v5));
            queue.write_buffer(&self.mode_properties_v6, 0, bytemuck::cast_slice(&v6));
        }
    }

    /// Incremental sync of cilia mode properties for a single genome
    pub fn incremental_sync_cilia_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let mut v5: Vec<[f32; 4]> = Vec::new();
        let mut v6: Vec<[f32; 4]> = Vec::new();

        for mode in &genome.modes {
            v5.push([
                mode.cilia_speed,
                if mode.cilia_push_bonded { 1.0 } else { 0.0 },
                if mode.cilia_use_signal { 1.0 } else { 0.0 },
                mode.cilia_signal_channel as f32,
            ]);
            v6.push([
                mode.cilia_speed_below,
                mode.cilia_speed_above,
                mode.cilia_threshold,
                mode.cilia_attract_force,
            ]);
        }

        if !v5.is_empty() {
            let offset = (global_start_index * 16) as u64; // 16 bytes per mode per sub-buffer
            queue.write_buffer(&self.mode_properties_v5, offset, bytemuck::cast_slice(&v5));
            queue.write_buffer(&self.mode_properties_v6, offset, bytemuck::cast_slice(&v6));
        }
    }

    /// Sync myocyte mode properties for the muscle_contraction shader.
    /// Data is written into 2 separate vec4 sub-buffers (v7, v8), each 16 bytes/mode.
    /// v7: [myocyte_contraction, myocyte_use_signal as f32, myocyte_signal_channel as f32, myocyte_threshold]
    /// v8: [myocyte_contraction_above, myocyte_contraction_below, myocyte_pulse_rate, myocyte_pulse_phase as f32]
    pub fn sync_myocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v7: Vec<[f32; 4]> = Vec::new();
        let mut v8: Vec<[f32; 4]> = Vec::new();

        for genome in genomes {
            for mode in &genome.modes {
                if mode.cell_type == 16 {
                    v7.push([
                        if mode.luminocyte_invert { 1.0 } else { 0.0 },
                        0.0,
                        mode.luminocyte_signal_channel.clamp(0, 7) as f32,
                        mode.luminocyte_threshold,
                    ]);
                } else {
                    v7.push([
                        mode.myocyte_contraction,
                        if mode.myocyte_use_signal { 1.0 } else { 0.0 },
                        mode.myocyte_signal_channel as f32,
                        mode.myocyte_threshold,
                    ]);
                }
                v8.push([
                    mode.myocyte_contraction_above,
                    mode.myocyte_contraction_below,
                    mode.myocyte_pulse_rate,
                    mode.myocyte_pulse_phase as f32,
                ]);
            }
        }

        if !v7.is_empty() {
            queue.write_buffer(&self.mode_properties_v7, 0, bytemuck::cast_slice(&v7));
            queue.write_buffer(&self.mode_properties_v8, 0, bytemuck::cast_slice(&v8));
        }
    }

    /// Incremental sync of myocyte mode properties for a single genome
    pub fn incremental_sync_myocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let mut v7: Vec<[f32; 4]> = Vec::new();
        let mut v8: Vec<[f32; 4]> = Vec::new();

        for mode in &genome.modes {
            if mode.cell_type == 16 {
                v7.push([
                    if mode.luminocyte_invert { 1.0 } else { 0.0 },
                    0.0,
                    mode.luminocyte_signal_channel as f32,
                    mode.luminocyte_threshold,
                ]);
            } else {
                v7.push([
                    mode.myocyte_contraction,
                    if mode.myocyte_use_signal { 1.0 } else { 0.0 },
                    mode.myocyte_signal_channel as f32,
                    mode.myocyte_threshold,
                ]);
            }
            v8.push([
                mode.myocyte_contraction_above,
                mode.myocyte_contraction_below,
                mode.myocyte_pulse_rate,
                mode.myocyte_pulse_phase as f32,
            ]);
        }

        if !v7.is_empty() {
            let offset = (global_start_index * 16) as u64; // 16 bytes per mode per sub-buffer
            queue.write_buffer(&self.mode_properties_v7, offset, bytemuck::cast_slice(&v7));
            queue.write_buffer(&self.mode_properties_v8, offset, bytemuck::cast_slice(&v8));
        }
    }

    /// Sync Embryocyte mode properties for the lifecycle shaders.
    /// v9:  [use_timer as f32, release_timer, use_threshold as f32, threshold_value as f32]
    /// v10: [use_signal as f32, signal_channel as f32, signal_value, 0.0]
    pub fn sync_embryocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v9: Vec<[f32; 4]> = Vec::new();
        let mut v10: Vec<[f32; 4]> = Vec::new();

        for genome in genomes {
            for mode in &genome.modes {
                v9.push([
                    if mode.embryocyte_use_timer { 1.0 } else { 0.0 },
                    mode.embryocyte_release_timer,
                    if mode.embryocyte_use_threshold {
                        1.0
                    } else {
                        0.0
                    },
                    mode.embryocyte_threshold_value as f32,
                ]);
                v10.push([
                    if mode.embryocyte_use_signal { 1.0 } else { 0.0 },
                    mode.embryocyte_signal_channel as f32,
                    mode.embryocyte_signal_value,
                    0.0,
                ]);
            }
        }

        if !v9.is_empty() {
            queue.write_buffer(&self.mode_properties_v9, 0, bytemuck::cast_slice(&v9));
            queue.write_buffer(&self.mode_properties_v10, 0, bytemuck::cast_slice(&v10));
        }
    }

    /// Incremental sync of Embryocyte mode properties for a single genome.
    pub fn incremental_sync_embryocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let mut v9: Vec<[f32; 4]> = Vec::new();
        let mut v10: Vec<[f32; 4]> = Vec::new();

        for mode in &genome.modes {
            v9.push([
                if mode.embryocyte_use_timer { 1.0 } else { 0.0 },
                mode.embryocyte_release_timer,
                if mode.embryocyte_use_threshold {
                    1.0
                } else {
                    0.0
                },
                mode.embryocyte_threshold_value as f32,
            ]);
            v10.push([
                if mode.embryocyte_use_signal { 1.0 } else { 0.0 },
                mode.embryocyte_signal_channel as f32,
                mode.embryocyte_signal_value,
                0.0,
            ]);
        }

        if !v9.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(&self.mode_properties_v9, offset, bytemuck::cast_slice(&v9));
            queue.write_buffer(
                &self.mode_properties_v10,
                offset,
                bytemuck::cast_slice(&v10),
            );
        }
    }

    /// Sync Devorocyte mode properties and myocyte grip into v11.
    /// v11: [consume_range, consume_rate, myocyte_grip_contracted, myocyte_grip_extended]
    pub fn sync_devorocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v11: Vec<[f32; 4]> = Vec::new();
        for genome in genomes {
            for mode in &genome.modes {
                v11.push([
                    mode.devorocyte_consume_range,
                    mode.devorocyte_consume_rate,
                    mode.myocyte_grip_contracted,
                    mode.myocyte_grip_extended,
                ]);
            }
        }
        if !v11.is_empty() {
            queue.write_buffer(&self.mode_properties_v11, 0, bytemuck::cast_slice(&v11));
        }
    }

    /// Incremental sync of Devorocyte/myocyte-grip mode properties for a single genome.
    pub fn incremental_sync_devorocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let v11: Vec<[f32; 4]> = genome
            .modes
            .iter()
            .map(|mode| {
                [
                    mode.devorocyte_consume_range,
                    mode.devorocyte_consume_rate,
                    mode.myocyte_grip_contracted,
                    mode.myocyte_grip_extended,
                ]
            })
            .collect();
        if !v11.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(
                &self.mode_properties_v11,
                offset,
                bytemuck::cast_slice(&v11),
            );
        }
    }

    /// Sync Vasculocyte mode properties for the nutrient/signal transport shaders.
    /// v12: [nutrient_transport, nutrient_exchange, signal_transport, signal_exchange]
    pub fn sync_vasculocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v12: Vec<[f32; 4]> = Vec::new();
        for genome in genomes {
            for mode in &genome.modes {
                v12.push([
                    if mode.vascular_nutrient_transport {
                        1.0
                    } else {
                        0.0
                    },
                    if mode.vascular_outlet { 1.0 } else { 0.0 },
                    if mode.vascular_signal_transport {
                        1.0
                    } else {
                        0.0
                    },
                    if mode.vascular_signal_exchange {
                        1.0
                    } else {
                        0.0
                    },
                ]);
            }
        }
        if !v12.is_empty() {
            queue.write_buffer(&self.mode_properties_v12, 0, bytemuck::cast_slice(&v12));
        }
    }

    /// Incremental sync of Vasculocyte mode properties for a single genome.
    pub fn incremental_sync_vasculocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let v12: Vec<[f32; 4]> = genome
            .modes
            .iter()
            .map(|mode| {
                [
                    if mode.vascular_nutrient_transport {
                        1.0
                    } else {
                        0.0
                    },
                    if mode.vascular_outlet { 1.0 } else { 0.0 },
                    if mode.vascular_signal_transport {
                        1.0
                    } else {
                        0.0
                    },
                    if mode.vascular_signal_exchange {
                        1.0
                    } else {
                        0.0
                    },
                ]
            })
            .collect();
        if !v12.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(
                &self.mode_properties_v12,
                offset,
                bytemuck::cast_slice(&v12),
            );
        }
    }

    /// Sync Gametocyte mode properties.
    /// v13: [merge_range, 0.0, 0.0, 0.0]
    pub fn sync_gametocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v13: Vec<[f32; 4]> = Vec::new();
        for genome in genomes {
            for mode in &genome.modes {
                v13.push([mode.gametocyte_merge_range, 0.0, 0.0, 0.0]);
            }
        }
        if !v13.is_empty() {
            queue.write_buffer(&self.mode_properties_v13, 0, bytemuck::cast_slice(&v13));
        }
    }

    /// Incremental sync of Gametocyte mode properties for a single genome.
    pub fn incremental_sync_gametocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let v13: Vec<[f32; 4]> = genome
            .modes
            .iter()
            .map(|mode| [mode.gametocyte_merge_range, 0.0, 0.0, 0.0])
            .collect();
        if !v13.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(
                &self.mode_properties_v13,
                offset,
                bytemuck::cast_slice(&v13),
            );
        }
    }

    /// Sync Siphonocyte mode properties.
    /// v14: [intake_rate, expel_rate, impulse, packed signal settings]
    /// packed = threshold * 128 + mode * 32 + invert * 16 + channel
    pub fn sync_siphonocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v14: Vec<[f32; 4]> = Vec::new();
        for genome in genomes {
            for mode in &genome.modes {
                let packed = mode.siphon_signal_channel.clamp(0, 15)
                    + (if mode.siphon_signal_invert { 1 } else { 0 }) * 16
                    + mode.siphon_mode.clamp(0, 3) * 32
                    + mode.siphon_signal_threshold.clamp(0.0, 2047.0).round() as i32 * 128;
                v14.push([
                    mode.siphon_intake_rate,
                    mode.siphon_expel_rate,
                    mode.siphon_impulse,
                    packed as f32,
                ]);
            }
        }
        if !v14.is_empty() {
            queue.write_buffer(&self.mode_properties_v14, 0, bytemuck::cast_slice(&v14));
        }
    }

    /// Incremental sync of Siphonocyte mode properties for a single genome.
    pub fn incremental_sync_siphonocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let v14: Vec<[f32; 4]> = genome
            .modes
            .iter()
            .map(|mode| {
                let packed = mode.siphon_signal_channel.clamp(0, 15)
                    + (if mode.siphon_signal_invert { 1 } else { 0 }) * 16
                    + mode.siphon_mode.clamp(0, 3) * 32
                    + mode.siphon_signal_threshold.clamp(0.0, 2047.0).round() as i32 * 128;
                [
                    mode.siphon_intake_rate,
                    mode.siphon_expel_rate,
                    mode.siphon_impulse,
                    packed as f32,
                ]
            })
            .collect();
        if !v14.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(
                &self.mode_properties_v14,
                offset,
                bytemuck::cast_slice(&v14),
            );
        }
    }

    /// Sync Plumocyte mode properties.
    /// v15: [extension, drag_mult, flow_coupling, exposure_mult]
    pub fn sync_plumocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut v15: Vec<[f32; 4]> = Vec::new();
        for genome in genomes {
            for mode in &genome.modes {
                v15.push([
                    mode.plumocyte_extension,
                    mode.plumocyte_drag_mult,
                    mode.plumocyte_flow_coupling,
                    mode.plumocyte_exposure_mult,
                ]);
            }
        }
        if !v15.is_empty() {
            queue.write_buffer(&self.mode_properties_v15, 0, bytemuck::cast_slice(&v15));
        }
    }

    /// Incremental sync of Plumocyte mode properties for a single genome.
    pub fn incremental_sync_plumocyte_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let v15: Vec<[f32; 4]> = genome
            .modes
            .iter()
            .map(|mode| {
                [
                    mode.plumocyte_extension,
                    mode.plumocyte_drag_mult,
                    mode.plumocyte_flow_coupling,
                    mode.plumocyte_exposure_mult,
                ]
            })
            .collect();
        if !v15.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(
                &self.mode_properties_v15,
                offset,
                bytemuck::cast_slice(&v15),
            );
        }
    }

    /// Upload Embryocyte reserve values from canonical state to GPU.
    /// Called during full sync and after any cell addition that involves an Embryocyte.
    pub fn sync_embryocyte_reserves(
        &self,
        queue: &wgpu::Queue,
        reserves: &[u32],
        cell_count: usize,
    ) {
        if cell_count == 0 {
            return;
        }
        let data = &reserves[..cell_count.min(reserves.len())];
        queue.write_buffer(
            &self.embryocyte_reserve_buffer,
            0,
            bytemuck::cast_slice(data),
        );
    }

    /// Sync mode cell types lookup table (cell_type per mode)
    /// This allows shaders to derive cell_type from mode_index, which is always up-to-date
    pub fn sync_mode_cell_types(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mode_cell_types: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| genome.modes.iter().map(|mode| mode.cell_type as u32))
            .collect();

        if !mode_cell_types.is_empty() {
            queue.write_buffer(
                &self.mode_cell_types,
                0,
                bytemuck::cast_slice(&mode_cell_types),
            );
        }
    }

    /// Sync glueocyte env adhesion flags (one u32 per mode across all genomes)
    pub fn sync_glueocyte_env_adhesion_flags(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let flags: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| {
                    if mode.glueocyte_env_adhesion {
                        1u32
                    } else {
                        0u32
                    }
                })
            })
            .collect();
        if !flags.is_empty() {
            queue.write_buffer(
                &self.glueocyte_env_adhesion_flags,
                0,
                bytemuck::cast_slice(&flags),
            );
        }
    }

    pub fn sync_glueocyte_boulder_adhesion_flags(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let flags: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| {
                    if mode.glueocyte_boulder_adhesion {
                        1u32
                    } else {
                        0u32
                    }
                })
            })
            .collect();
        if !flags.is_empty() {
            queue.write_buffer(
                &self.glueocyte_boulder_adhesion_flags,
                0,
                bytemuck::cast_slice(&flags),
            );
        }
    }

    /// Sync glueocyte cell-adhesion flags (4 u32 per mode across all genomes).
    /// Layout per mode: [enabled(u32), signal_channel(u32), signal_threshold_bits(u32), self_adhesion(u32)]
    /// signal_channel = 0xFFFFFFFF means "always active" (no signal gate).
    /// self_adhesion = 1 means the glueocyte bonds to cells of its own organism.
    pub fn sync_glueocyte_cell_adhesion_flags(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let data: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().flat_map(|mode| {
                    let enabled = if mode.glueocyte_cell_adhesion {
                        1u32
                    } else {
                        0u32
                    };
                    let channel = if mode.glueocyte_cell_adhesion_signal_channel >= 0
                        && mode.glueocyte_cell_adhesion_signal_channel <= 7
                    {
                        mode.glueocyte_cell_adhesion_signal_channel as u32
                    } else {
                        0xFFFF_FFFFu32 // disabled / always-active
                    };
                    let threshold_bits = mode.glueocyte_cell_adhesion_signal_threshold.to_bits();
                    let self_adhesion = if mode.glueocyte_self_adhesion {
                        1u32
                    } else {
                        0u32
                    };
                    let invert = if mode.glueocyte_signal_gate_invert {
                        2u32
                    } else {
                        0u32
                    };
                    [enabled, channel, threshold_bits, self_adhesion | invert]
                })
            })
            .collect();
        if !data.is_empty() {
            queue.write_buffer(
                &self.glueocyte_cell_adhesion_flags,
                0,
                bytemuck::cast_slice(&data),
            );
        }
    }

    /// Sync oculocyte parameters for all modes across all genomes
    /// Layout per mode: [sense_type(u32), ray_length_bits(u32), signal_hops(u32), signal_channel(u32)]
    pub fn sync_oculocyte_params(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let params: Vec<[u32; 4]> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| {
                    [
                        mode.oculocyte_sense_type, // already u32 bitmask
                        mode.oculocyte_ray_length.to_bits(),
                        mode.oculocyte_signal_hops as u32,
                        mode.oculocyte_signal_channel.clamp(0, 7) as u32, // Oculocyte channels 0-7 only
                    ]
                })
            })
            .collect();
        if !params.is_empty() {
            queue.write_buffer(&self.oculocyte_params, 0, bytemuck::cast_slice(&params));
        }

        // Sync signal values separately (kept out of oculocyte_params to preserve mutation system's vec4<u32> stride)
        let signal_values: Vec<f32> = genomes
            .iter()
            .flat_map(|genome| {
                genome
                    .modes
                    .iter()
                    .map(|mode| mode.oculocyte_signal_value.clamp(0.0, 2047.0))
            })
            .collect();
        if !signal_values.is_empty() {
            queue.write_buffer(
                &self.oculocyte_signal_values,
                0,
                bytemuck::cast_slice(&signal_values),
            );
        }

        let light_filters: Vec<[f32; 4]> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| {
                    [
                        mode.oculocyte_light_target_color.x.clamp(0.0, 4.0),
                        mode.oculocyte_light_target_color.y.clamp(0.0, 4.0),
                        mode.oculocyte_light_target_color.z.clamp(0.0, 4.0),
                        mode.oculocyte_light_color_tolerance.clamp(0.0, 4.0),
                    ]
                })
            })
            .collect();
        if !light_filters.is_empty() {
            queue.write_buffer(
                &self.oculocyte_light_filters,
                0,
                bytemuck::cast_slice(&light_filters),
            );
        }
    }

    /// Sync regulation emission parameters for all modes across all genomes.
    /// Layout per mode: [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding(u32)]
    /// emit_channel: 0xFFFFFFFF (u32 max) = disabled, 8-15 = regulation channel
    pub fn sync_regulation_params(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let params: Vec<[u32; 4]> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| {
                    let channel = if mode.regulation_emit_channel < 0 {
                        0xFFFFFFFFu32 // Disabled sentinel
                    } else {
                        (mode.regulation_emit_channel as u32).clamp(8, 15)
                    };
                    [
                        channel,
                        mode.regulation_emit_value.clamp(0.0, 2047.0).to_bits(),
                        mode.regulation_emit_hops.clamp(1, 20) as u32,
                        0u32, // padding
                    ]
                })
            })
            .collect();
        if !params.is_empty() {
            queue.write_buffer(&self.regulation_params, 0, bytemuck::cast_slice(&params));
        }
    }

    /// Sync signal-conditional settings for all modes across all genomes.
    /// Packs division gating, apoptosis, signal-conditional child routing, and mode switching
    /// into 5 vec4<f32> sub-buffers per mode.
    pub fn sync_signal_settings(&self, queue: &wgpu::Queue, genomes: &[crate::genome::Genome]) {
        let mut v0: Vec<[f32; 4]> = Vec::new();
        let mut v1: Vec<[f32; 4]> = Vec::new();
        let mut v2: Vec<[f32; 4]> = Vec::new();
        let mut v3: Vec<[f32; 4]> = Vec::new();
        let mut v4: Vec<[f32; 4]> = Vec::new();

        let mut global_mode_offset = 0i32;
        for genome in genomes {
            for mode in &genome.modes {
                // Remap local mode indices to absolute GPU mode indices
                let remap = |local: i32| -> f32 {
                    if local < 0 {
                        -1.0
                    } else {
                        (global_mode_offset + local.max(0)) as f32
                    }
                };

                v0.push([
                    mode.division_signal_channel as f32,
                    mode.division_signal_threshold,
                    if mode.division_signal_invert {
                        1.0
                    } else {
                        0.0
                    },
                    mode.apoptosis_signal_channel as f32,
                ]);
                v1.push([
                    mode.apoptosis_signal_threshold,
                    if mode.apoptosis_signal_invert {
                        1.0
                    } else {
                        0.0
                    },
                    mode.signal_child_a_channel as f32,
                    mode.signal_child_a_threshold,
                ]);
                v2.push([
                    remap(mode.signal_child_a_mode_above),
                    remap(mode.signal_child_a_mode_below),
                    mode.signal_child_b_channel as f32,
                    mode.signal_child_b_threshold,
                ]);
                v3.push([
                    remap(mode.signal_child_b_mode_above),
                    remap(mode.signal_child_b_mode_below),
                    mode.mode_switch_signal_channel as f32,
                    mode.mode_switch_signal_threshold,
                ]);
                v4.push([
                    remap(mode.mode_switch_target),
                    if mode.mode_switch_invert { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ]);
            }
            global_mode_offset += genome.modes.len() as i32;
        }

        if !v0.is_empty() {
            queue.write_buffer(&self.signal_settings_v0, 0, bytemuck::cast_slice(&v0));
            queue.write_buffer(&self.signal_settings_v1, 0, bytemuck::cast_slice(&v1));
            queue.write_buffer(&self.signal_settings_v2, 0, bytemuck::cast_slice(&v2));
            queue.write_buffer(&self.signal_settings_v3, 0, bytemuck::cast_slice(&v3));
            queue.write_buffer(&self.signal_settings_v4, 0, bytemuck::cast_slice(&v4));
        }
    }

    /// Sync behavior flags for all cell types
    /// This populates the GPU buffer with behavior flags (applies_swim_force, etc.)
    /// for each cell type. Should be called once during initialization.
    pub fn sync_behavior_flags(&self, queue: &wgpu::Queue) {
        use crate::cell::types::{CellType, GpuCellTypeBehaviorFlags};

        let flags: Vec<GpuCellTypeBehaviorFlags> =
            CellType::iter().map(|t| t.behavior_flags()).collect();

        queue.write_buffer(&self.behavior_flags, 0, bytemuck::cast_slice(&flags));
    }

    /// Incremental sync of mode properties for a single genome
    pub fn incremental_sync_mode_properties(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let mut v0: Vec<[f32; 4]> = Vec::new();
        let mut v1: Vec<[f32; 4]> = Vec::new();
        let mut v2: Vec<[f32; 4]> = Vec::new();
        let mut v3: Vec<[f32; 4]> = Vec::new();
        let mut v4: Vec<[f32; 4]> = Vec::new();

        let global_mode_offset = global_start_index as i32;
        for mode in &genome.modes {
            let gpu_max_splits = if mode.max_splits < 0 {
                -1.0
            } else {
                mode.max_splits as f32
            };
            let gpu_mode_a_after = if mode.mode_a_after_splits < 0 {
                -1.0
            } else {
                (global_mode_offset + mode.mode_a_after_splits.max(0)) as f32
            };
            let gpu_mode_b_after = if mode.mode_b_after_splits < 0 {
                -1.0
            } else {
                (global_mode_offset + mode.mode_b_after_splits.max(0)) as f32
            };
            v0.push([
                mode.nutrient_gain_rate,
                mode.max_cell_size,
                mode.membrane_stiffness,
                mode.split_interval,
            ]);
            v1.push([
                mode.split_mass,
                mode.nutrient_priority,
                mode.swim_force,
                if mode.prioritize_when_low { 1.0 } else { 0.0 },
            ]);
            v2.push([
                gpu_max_splits,
                mode.split_ratio,
                mode.flagellocyte_signal_channel as f32,
                mode.flagellocyte_speed_a,
            ]);
            v3.push([
                mode.flagellocyte_speed_b,
                mode.flagellocyte_threshold_c,
                if mode.flagellocyte_use_signal {
                    1.0
                } else {
                    0.0
                },
                mode.min_adhesions as f32,
            ]);
            v4.push([
                mode.max_adhesions as f32,
                gpu_mode_a_after,
                gpu_mode_b_after,
                0.0,
            ]);
        }

        if !v0.is_empty() {
            let offset = (global_start_index * 16) as u64; // 16 bytes per mode per sub-buffer
            queue.write_buffer(&self.mode_properties_v0, offset, bytemuck::cast_slice(&v0));
            queue.write_buffer(&self.mode_properties_v1, offset, bytemuck::cast_slice(&v1));
            queue.write_buffer(&self.mode_properties_v2, offset, bytemuck::cast_slice(&v2));
            queue.write_buffer(&self.mode_properties_v3, offset, bytemuck::cast_slice(&v3));
            queue.write_buffer(&self.mode_properties_v4, offset, bytemuck::cast_slice(&v4));
        }
    }

    /// Incremental sync of child mode indices for a single genome
    pub fn incremental_sync_child_mode_indices(
        &self,
        _device: &wgpu::Device,
        genome_id: usize,
        global_start_index: usize,
        mode_count: usize,
    ) {
        log::info!(
            "Incremental sync of child mode indices for genome {} at index {} ({} modes)",
            genome_id,
            global_start_index,
            mode_count
        );
    }

    /// Incremental sync of genome mode data for a single genome
    pub fn incremental_sync_genome_mode_data(
        &self,
        _device: &wgpu::Device,
        genome_id: usize,
        global_start_index: usize,
        mode_count: usize,
    ) {
        log::info!(
            "Incremental sync of genome mode data for genome {} at index {} ({} modes)",
            genome_id,
            global_start_index,
            mode_count
        );
    }

    /// Incremental sync of mode cell types for a single genome
    pub fn incremental_sync_mode_cell_types(
        &self,
        queue: &wgpu::Queue,
        genome: &crate::genome::Genome,
        global_start_index: usize,
    ) {
        let cell_types: Vec<u32> = genome
            .modes
            .iter()
            .map(|mode| mode.cell_type as u32)
            .collect();
        if !cell_types.is_empty() {
            let offset = (global_start_index * 4) as u64; // u32 = 4 bytes per mode
            queue.write_buffer(
                &self.mode_cell_types,
                offset,
                bytemuck::cast_slice(&cell_types),
            );
        }
    }

    /// Sync a single cell to all GPU buffer sets (for cell insertion during simulation)
    pub fn sync_single_cell(
        &self,
        queue: &wgpu::Queue,
        cell_idx: usize,
        position: glam::Vec3,
        velocity: glam::Vec3,
        mass: f32,
        rotation: glam::Quat,
    ) {
        let position_mass: [f32; 4] = [position.x, position.y, position.z, mass];
        let velocity_data: [f32; 4] = [velocity.x, velocity.y, velocity.z, 0.0];
        let rotation_data: [f32; 4] = [rotation.x, rotation.y, rotation.z, rotation.w];

        let offset = (cell_idx * 16) as u64; // Vec4<f32> = 16 bytes

        // Upload to all three buffer sets
        for i in 0..3 {
            queue.write_buffer(
                &self.position_and_mass[i],
                offset,
                bytemuck::bytes_of(&position_mass),
            );
            queue.write_buffer(
                &self.velocity[i],
                offset,
                bytemuck::bytes_of(&velocity_data),
            );
            queue.write_buffer(
                &self.rotations[i],
                offset,
                bytemuck::bytes_of(&rotation_data),
            );
        }

        // Also write genome orientation (same as initial rotation for new cells)
        queue.write_buffer(
            &self.genome_orientations,
            offset,
            bytemuck::bytes_of(&rotation_data),
        );
    }

    /// Sync a single cell's state data (for cell insertion during simulation)
    pub fn sync_single_cell_state(
        &self,
        queue: &wgpu::Queue,
        cell_idx: usize,
        birth_time: f32,
        split_interval: f32,
        split_nutrient_threshold: f32,
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
        queue.write_buffer(
            &self.split_intervals,
            offset,
            bytemuck::bytes_of(&split_interval),
        );
        queue.write_buffer(
            &self.split_nutrient_thresholds,
            offset,
            bytemuck::bytes_of(&split_nutrient_threshold),
        );
        queue.write_buffer(&self.split_counts, offset, bytemuck::bytes_of(&0u32));

        // Convert max_splits: -1 (infinite) -> 0xFFFFFFFF (unlimited in GPU), 0+ stay as-is
        let gpu_max_splits: u32 = if max_splits < 0 {
            u32::MAX
        } else {
            max_splits as u32
        };
        queue.write_buffer(
            &self.max_splits,
            offset,
            bytemuck::bytes_of(&gpu_max_splits),
        );

        queue.write_buffer(
            &self.nutrient_gain_rates,
            offset,
            bytemuck::bytes_of(&nutrient_gain_rate),
        );
        queue.write_buffer(
            &self.max_cell_sizes,
            offset,
            bytemuck::bytes_of(&max_cell_size),
        );
        queue.write_buffer(&self.stiffnesses, offset, bytemuck::bytes_of(&stiffness));
        queue.write_buffer(
            &self.genome_ids,
            offset,
            bytemuck::bytes_of(&(genome_id as u32)),
        );
        queue.write_buffer(&self.cell_types, offset, bytemuck::bytes_of(&cell_type));
        queue.write_buffer(
            &self.mode_indices,
            offset,
            bytemuck::bytes_of(&(mode_index as u32)),
        );
        queue.write_buffer(
            &self.cell_ids,
            offset,
            bytemuck::bytes_of(&(cell_id as u32)),
        );
        queue.write_buffer(&self.death_flags, offset, bytemuck::bytes_of(&0u32)); // Alive
        queue.write_buffer(&self.division_flags, offset, bytemuck::bytes_of(&0u32)); // Not dividing

        // Initialize nutrients to 100.0 (full) = 100000 in fixed-point scale
        let initial_nutrients: i32 = 100_000;
        queue.write_buffer(
            &self.nutrients_buffer,
            offset,
            bytemuck::bytes_of(&initial_nutrients),
        );

        let initial_temperature = 105.0f32;
        let initial_water = Self::physiology_water_capacity_for_cell_type(cell_type);
        let initial_heat_energy =
            Self::physiology_heat_for_temperature(initial_temperature, initial_water);
        let initial_thermal_state = 4u32;
        queue.write_buffer(&self.cell_water, offset, bytemuck::bytes_of(&initial_water));
        queue.write_buffer(
            &self.cell_heat_energy,
            offset,
            bytemuck::bytes_of(&initial_heat_energy),
        );
        queue.write_buffer(
            &self.cell_cached_temperature,
            offset,
            bytemuck::bytes_of(&initial_temperature),
        );
        queue.write_buffer(
            &self.cell_thermal_state,
            offset,
            bytemuck::bytes_of(&initial_thermal_state),
        );
        queue.write_buffer(
            &self.cell_water_next,
            offset,
            bytemuck::bytes_of(&initial_water),
        );
        queue.write_buffer(
            &self.cell_heat_energy_next,
            offset,
            bytemuck::bytes_of(&initial_heat_energy),
        );
        queue.write_buffer(
            &self.cell_cached_temperature_next,
            offset,
            bytemuck::bytes_of(&initial_temperature),
        );
        queue.write_buffer(
            &self.cell_thermal_state_next,
            offset,
            bytemuck::bytes_of(&initial_thermal_state),
        );
        queue.write_buffer(
            &self.cell_prev_muscle_contraction,
            offset,
            bytemuck::bytes_of(&0.0f32),
        );
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
        // Derive mass from nutrients: mass = 1.0 + nutrients / 100.0
        let mass = 1.0 + request.nutrients / 100.0;
        self.sync_single_cell(
            queue,
            slot as usize,
            request.position,
            request.velocity,
            mass,
            request.rotation,
        );

        // Calculate absolute mode index (with genome offset)
        let absolute_mode_idx =
            self.calculate_absolute_mode_index(request.genome_id, request.mode_index, genomes);

        // Get cell properties from genome mode settings
        let (cell_type, nutrient_gain_rate, max_cell_size, max_splits) =
            if request.genome_id < genomes.len() {
                let genome = &genomes[request.genome_id];
                if request.mode_index < genome.modes.len() {
                    let mode = &genome.modes[request.mode_index];
                    // Only Test cells (cell_type == 0) auto-generate nutrients
                    let nutrient_rate = if mode.cell_type == 0 {
                        mode.nutrient_gain_rate
                    } else {
                        0.0
                    };
                    (
                        mode.cell_type as u32,
                        nutrient_rate,
                        mode.max_cell_size,
                        mode.max_splits,
                    )
                } else {
                    (0, 0.2, 2.0, -1) // Defaults if mode not found
                }
            } else {
                (0, 0.2, 2.0, -1) // Defaults if genome not found
            };

        // Sync cell state for division system
        self.sync_single_cell_state(
            queue,
            slot as usize,
            request.birth_time,
            request.split_interval,
            request.split_nutrient_threshold,
            request.genome_id,
            absolute_mode_idx,
            cell_id as usize,
            max_splits,
            nutrient_gain_rate,
            max_cell_size,
            request.stiffness,
            cell_type,
        );
        let organism_id = cell_id.saturating_add(1);
        let lineage_hash = Self::development_root_hash(
            DEVELOPMENT_ROOT_LINEAGE_HASH,
            organism_id,
            request.genome_id,
            request.mode_index,
        );
        self.sync_single_development_address(
            queue,
            slot as usize,
            organism_id,
            lineage_hash,
            0,
            0,
            1,
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
        queue.write_buffer(
            &self.cell_count_buffer,
            0,
            bytemuck::cast_slice(&cell_counts),
        );
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
    /// Returns Some((total, live)) if new count is available, None if still pending or no read started.
    /// This is non-blocking.
    pub fn poll_cell_count(&mut self, device: &wgpu::Device) -> Option<(u32, u32)> {
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

                    // counts[0] = total cells (high water mark), counts[1] = live cells
                    let total = counts[0];
                    let live = counts[1];

                    // Use live cells count for display
                    self.last_cell_count = live;

                    drop(data);
                    self.cell_count_readback_buffer.unmap();

                    self.cell_count_map_pending = false;
                    self.cell_count_receiver = None;
                    return Some((total, live));
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

    /// Clear cached cell-count readback state.
    ///
    /// Scene reset writes a fresh zero count to the GPU buffer. If an async
    /// readback from before the reset completes later, it must not overwrite the
    /// CPU-side display with stale counts.
    pub fn reset_cell_count_readback(&mut self) {
        if self.cell_count_map_pending {
            self.cell_count_readback_buffer.unmap();
        }
        self.cell_count_map_pending = false;
        self.cell_count_receiver = None;
        self.last_cell_count = 0;
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

    /// Reset ring buffers to empty state (for scene reset)
    /// Clears ring_state to [0, 0, 0, 0] (head=0, tail=0, next_slot_id=0, reservation_count=0)
    /// Also clears death_flags, division_flags, and lifecycle_counts so stale state
    /// from the previous simulation doesn't leak into the new one.
    /// Without clearing death_flags, the death_scan shader's `is_dead && !was_dead` check
    /// would treat recycled slots as "already dead" and never push them back into the ring.
    pub fn reset_ring_buffers(&self, queue: &wgpu::Queue) {
        let ring_state_data: [u32; 4] = [0, 0, 0, 0];
        queue.write_buffer(&self.ring_state, 0, bytemuck::cast_slice(&ring_state_data));

        // Clear death_flags so no slots appear "already dead" after reset
        let zeros = vec![0u8; self.capacity as usize * 4];
        queue.write_buffer(&self.death_flags, 0, &zeros);

        // Clear division_flags to prevent stale division attempts
        queue.write_buffer(&self.division_flags, 0, &zeros);

        // Clear lifecycle_counts (3 x u32 = 12 bytes)
        let lifecycle_zeros: [u32; 3] = [0, 0, 0];
        queue.write_buffer(
            &self.lifecycle_counts,
            0,
            bytemuck::cast_slice(&lifecycle_zeros),
        );
    }

    /// Synchronize slot allocator with canonical state
    pub fn sync_slot_allocator_with_canonical(&mut self, canonical_state: &CanonicalState) {
        // Reset allocator
        self.slot_allocator.reset();

        // Set next cell ID to match canonical state
        self.slot_allocator
            .set_next_cell_id(canonical_state.next_cell_id as u32);

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
        let mut pending_requests =
            std::mem::take(&mut self.cell_addition_pipeline.pending_requests);

        if pending_requests.is_empty() {
            return Vec::new();
        }

        // Sort requests by deterministic hash for consistent ordering
        pending_requests.sort_by_key(|req| req.deterministic_hash());

        let mut added_cells = Vec::new();

        // Process each request in sorted order
        for request in pending_requests {
            if let Some((slot, cell_id)) =
                self.add_cell_deterministic(queue, request.clone(), genomes)
            {
                // Also add to canonical state at the same slot
                if let Some(_canonical_index) = canonical_state.add_cell_at_slot(
                    slot as usize,
                    request.position,
                    request.velocity,
                    request.rotation,
                    request.genome_orientation,
                    request.angular_velocity,
                    request.nutrients,
                    request.genome_id,
                    request.mode_index,
                    request.birth_time,
                    request.split_interval,
                    request.split_nutrient_threshold,
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
        nutrients: f32,
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
                (
                    mode.split_interval,
                    mode.split_mass,
                    mode.membrane_stiffness,
                )
            } else {
                (10.0, 2.0, 50.0) // Default values
            }
        } else {
            (10.0, 2.0, 50.0) // Default values
        };

        // Convert split_mass to nutrient threshold
        let split_nutrient_threshold = (split_mass - 1.0) * 100.0;

        CellAdditionRequest {
            position,
            velocity,
            nutrients,
            rotation,
            genome_id,
            mode_index,
            birth_time,
            split_interval,
            split_nutrient_threshold,
            stiffness,
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
    pub fn sync_pending_insertions(
        &mut self,
        queue: &wgpu::Queue,
        state: &crate::simulation::CanonicalState,
        genomes: &[crate::genome::Genome],
    ) {
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
                let (max_splits, nutrient_gain_rate, max_cell_size, stiffness, cell_type) =
                    if genome_id < genomes.len() {
                        let genome = &genomes[genome_id];
                        if local_mode_idx < genome.modes.len() {
                            let mode = &genome.modes[local_mode_idx];
                            // Only Test cells (cell_type == 0) auto-generate nutrients
                            // All other cells rely on specialized functions (e.g., photosynthesis) or nutrient transport
                            let nutrient_rate = if mode.cell_type == 0 {
                                mode.nutrient_gain_rate
                            } else {
                                0.0
                            };
                            (
                                mode.max_splits,
                                nutrient_rate,
                                mode.max_cell_size,
                                mode.membrane_stiffness,
                                mode.cell_type as u32,
                            )
                        } else {
                            (-1, 0.2, 2.0, 50.0, 0) // Defaults if mode not found
                        }
                    } else {
                        (-1, 0.2, 2.0, 50.0, 0) // Defaults if genome not found
                    };

                // Calculate absolute mode index (with genome offset)
                let absolute_mode_idx =
                    genome_mode_offsets.get(genome_id).copied().unwrap_or(0) + local_mode_idx;

                // Also sync cell state for division system
                self.sync_single_cell_state(
                    queue,
                    cell_idx,
                    state.birth_times[cell_idx],
                    state.split_intervals[cell_idx],
                    state.split_nutrient_thresholds[cell_idx],
                    state.genome_ids[cell_idx],
                    absolute_mode_idx, // Use absolute mode index
                    state.cell_ids[cell_idx] as usize,
                    max_splits,
                    nutrient_gain_rate,
                    max_cell_size,
                    stiffness,
                    cell_type,
                );
                self.sync_single_development_address(
                    queue,
                    cell_idx,
                    state.organism_ids[cell_idx],
                    state.lineage_hashes[cell_idx],
                    state.lineage_depths[cell_idx],
                    state.lineage_branch_slots[cell_idx],
                    state.organism_cell_ids[cell_idx],
                );
            }
        }

        // Update GPU cell count buffer after insertions
        if !self.pending_cell_insertions.is_empty() {
            let cell_counts: [u32; 2] = [state.cell_count as u32, state.cell_count as u32];
            queue.write_buffer(
                &self.cell_count_buffer,
                0,
                bytemuck::cast_slice(&cell_counts),
            );
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
    pub fn debug_read_positions_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_index: usize,
        cell_count: usize,
    ) -> Vec<[f32; 4]> {
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
    pub fn debug_read_cell_count_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> [u32; 2] {
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
    pub fn debug_read_mode_indices_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        count: usize,
    ) -> Vec<u32> {
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
    use glam::{Quat, Vec3};

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
        assert_eq!(id4, 4); // ID continues incrementing
    }

    #[test]
    fn test_cell_addition_request_deterministic_hash() {
        let request1 = CellAdditionRequest {
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::ZERO,
            nutrients: 100.0,
            rotation: Quat::IDENTITY,
            genome_id: 0,
            mode_index: 0,
            birth_time: 0.0,
            split_interval: 10.0,
            split_nutrient_threshold: 50.0,
            stiffness: 50.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };

        let request2 = CellAdditionRequest {
            position: Vec3::new(1.0, 2.0, 3.0), // Same position
            velocity: Vec3::new(1.0, 0.0, 0.0), // Different velocity (shouldn't affect hash)
            nutrients: 50.0,                    // Different nutrients (shouldn't affect hash)
            rotation: Quat::IDENTITY,
            genome_id: 0, // Same genome/mode
            mode_index: 0,
            birth_time: 5.0, // Different time (shouldn't affect hash)
            split_interval: 15.0,
            split_nutrient_threshold: 75.0,
            stiffness: 75.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };

        let request3 = CellAdditionRequest {
            position: Vec3::new(1.0, 2.0, 3.0), // Same position
            velocity: Vec3::ZERO,
            nutrients: 100.0,
            rotation: Quat::IDENTITY,
            genome_id: 1, // Different genome (should affect hash)
            mode_index: 0,
            birth_time: 0.0,
            split_interval: 10.0,
            split_nutrient_threshold: 50.0,
            stiffness: 50.0,
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
            nutrients: 100.0,
            rotation: Quat::IDENTITY,
            genome_id: 1,
            mode_index: 1,
            birth_time: 0.0,
            split_interval: 10.0,
            split_nutrient_threshold: 50.0,
            stiffness: 50.0,
            genome_orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        };

        let request_low_hash = CellAdditionRequest {
            position: Vec3::new(1.0, 1.0, 1.0), // Low coordinates = low hash
            velocity: Vec3::ZERO,
            nutrients: 100.0,
            rotation: Quat::IDENTITY,
            genome_id: 0,
            mode_index: 0,
            birth_time: 0.0,
            split_interval: 10.0,
            split_nutrient_threshold: 50.0,
            stiffness: 50.0,
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
