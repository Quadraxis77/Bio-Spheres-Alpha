// Unified Lifecycle Shader - Stable Slot Allocation
// Combines death detection, slot reclamation, and division slot reservation
// Uses a persistent ring buffer for free slot management
//
// Ring Buffer Design:
// - Deaths push slot indices onto the tail
// - Divisions pop slot indices from the head
// - If ring buffer is empty, use next_slot_id (append mode)
// - Capacity check prevents overflow
//
// Atomic counters layout:
// [0] = ring_head (next slot to pop for division)
// [1] = ring_tail (next slot to push for death)
// [2] = next_slot_id (for append mode when ring is empty)
// [3] = division_reservation_count (for debugging)

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Cell type behavior flags for parameterized shader logic
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    _padding: array<u32, 10>,
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

// GPU-side cell count: [0] = total slots used, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<atomic<u32>>;

// Lifecycle bind group (group 1)
@group(1) @binding(0)
var<storage, read_write> death_flags: array<u32>;

@group(1) @binding(1)
var<storage, read_write> division_flags: array<u32>;

// Ring buffer for free slots (persistent across frames)
@group(1) @binding(2)
var<storage, read_write> free_slot_ring: array<u32>;

// Division slot assignments (which slot each dividing cell gets)
@group(1) @binding(3)
var<storage, read_write> division_slot_assignments: array<u32>;

// Ring buffer state: [head, tail, next_slot_id, reservation_count]
@group(1) @binding(4)
var<storage, read_write> ring_state: array<atomic<u32>>;

// Cell state bind group (group 2)
@group(2) @binding(0)
var<storage, read> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read> split_nutrient_thresholds: array<f32>;

@group(2) @binding(3)
var<storage, read> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read> split_ready_frame: array<i32>;

@group(2) @binding(5)
var<storage, read> max_splits: array<u32>;

@group(2) @binding(6)
var<storage, read> cell_types: array<u32>;

@group(2) @binding(7)
var<storage, read> mode_indices: array<u32>;

@group(2) @binding(8)
var<storage, read> mode_cell_types: array<u32>;

@group(2) @binding(9)
var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// Nutrients buffer for division checks (read_write for atomic operations)
@group(2) @binding(10)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// mode_properties split into 5 vec4 sub-buffers (bindings 11-15)
@group(2) @binding(11)
var<storage, read> mode_properties_v0: array<vec4<f32>>; // [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
@group(2) @binding(12)
var<storage, read> mode_properties_v1: array<vec4<f32>>; // [split_mass, nutrient_priority, swim_force, prioritize_when_low]
@group(2) @binding(13)
var<storage, read> mode_properties_v2: array<vec4<f32>>; // [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
@group(2) @binding(14)
var<storage, read> mode_properties_v3: array<vec4<f32>>; // [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
@group(2) @binding(15)
var<storage, read> mode_properties_v4: array<vec4<f32>>; // [max_adhesions, mode_a_after_splits, mode_b_after_splits, padding]

// Per-cell signal flags (packed u32: bits encode signal value, hops, direction)
// Non-zero means cell has received a signal from the oculocyte sensing system
@group(2) @binding(16)
var<storage, read> signal_flags_read: array<u32>;

// Per-mode signal-conditional settings (5 vec4<f32> sub-buffers)
// v0: [division_signal_channel, division_signal_threshold, division_signal_invert, apoptosis_signal_channel]
// v1: [apoptosis_signal_threshold, apoptosis_signal_invert, signal_child_a_channel, signal_child_a_threshold]
@group(2) @binding(17)
var<storage, read> signal_settings_v0_read: array<vec4<f32>>;

@group(2) @binding(18)
var<storage, read> signal_settings_v1_read: array<vec4<f32>>;

// Adhesion bind group (group 3) - read-only for neighbor deferral check in division_scan
@group(3) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read> cell_adhesion_indices: array<i32>;

// Adhesion connection structure (104 bytes matching Rust GpuAdhesionConnection)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _align_pad: vec2<u32>,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
};

const MAX_ADHESIONS_PER_CELL: u32 = 20u;

// Constants
const DEATH_NUTRIENT_THRESHOLD: f32 = 1.0;  // Death when nutrients < 1.0
const RING_BUFFER_CAPACITY: u32 = 262144u; // 256K slots, must match Rust side (supports 200K cells)

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

// Push a free slot onto the ring buffer tail
// Returns true if successful, false if ring buffer is full
fn push_free_slot(slot_idx: u32) -> bool {
    let tail = atomicAdd(&ring_state[1], 1u);
    let head = atomicLoad(&ring_state[0]);
    
    // Check if ring buffer is full (tail caught up to head)
    let used = tail - head;
    if (used >= RING_BUFFER_CAPACITY) {
        // Buffer full, undo the increment
        atomicSub(&ring_state[1], 1u);
        return false;
    }
    
    // Write slot to ring buffer (wrap around)
    let ring_idx = tail % RING_BUFFER_CAPACITY;
    free_slot_ring[ring_idx] = slot_idx;
    return true;
}

// Pop a free slot from the ring buffer head
// Returns 0xFFFFFFFF if ring buffer is empty
fn pop_free_slot() -> u32 {
    let head = atomicAdd(&ring_state[0], 1u);
    let tail = atomicLoad(&ring_state[1]);
    
    // Check if ring buffer is empty (head caught up to tail)
    if (head >= tail) {
        // Buffer empty, undo the increment
        atomicSub(&ring_state[0], 1u);
        return 0xFFFFFFFFu;
    }
    
    // Read slot from ring buffer (wrap around)
    let ring_idx = head % RING_BUFFER_CAPACITY;
    return free_slot_ring[ring_idx];
}

// Allocate a slot for a new cell (division or insertion)
// First tries ring buffer, then falls back to append mode
// Returns 0xFFFFFFFF if at capacity
fn allocate_slot() -> u32 {
    // Try to get a recycled slot from the ring buffer
    let recycled_slot = pop_free_slot();
    if (recycled_slot != 0xFFFFFFFFu) {
        return recycled_slot;
    }
    
    // Ring buffer empty, try to allocate new slot
    let new_slot = atomicAdd(&ring_state[2], 1u);
    if (new_slot >= params.cell_capacity) {
        // At capacity, undo and fail
        atomicSub(&ring_state[2], 1u);
        return 0xFFFFFFFFu;
    }
    
    // Update total cell count if needed
    atomicMax(&cell_count_buffer[0], new_slot + 1u);
    
    return new_slot;
}

// Pass 1: Death detection only - pushes free slots to ring buffer
// Must run BEFORE division_scan to ensure dead slots are available for reuse
@compute @workgroup_size(256)
fn death_scan(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = atomicLoad(&cell_count_buffer[0]);
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // === DEATH DETECTION ===
    // Read nutrients from nutrients_buffer (fixed-point i32)
    let nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    let was_dead = death_flags[cell_idx] == 1u;
    
    // Check for invalid mode index (corrupted cell from mutation)
    let mode_idx = mode_indices[cell_idx];
    let has_invalid_mode = mode_idx >= arrayLength(&mode_cell_types);
    
    // Cell is dead if: nutrients below threshold OR invalid mode index
    let is_dead = nutrients < DEATH_NUTRIENT_THRESHOLD || has_invalid_mode;
    
    // Signal-conditional apoptosis check:
    // If the mode has apoptosis_signal_channel >= 0, check if the cell's signal
    // meets the threshold condition. If so, trigger death.
    var apoptosis_triggered = false;
    if (!is_dead && !was_dead) {
        let mode_idx = mode_indices[cell_idx];
        if (mode_idx < arrayLength(&signal_settings_v0_read)) {
            let ss_v0 = signal_settings_v0_read[mode_idx];
            let apoptosis_channel = ss_v0.w; // apoptosis_signal_channel
            if (apoptosis_channel >= 8.0) {
                let ss_v1 = signal_settings_v1_read[mode_idx];
                let apoptosis_threshold = ss_v1.x; // apoptosis_signal_threshold
                let apoptosis_invert = ss_v1.y;    // apoptosis_signal_invert (0 or 1)
                
                // Decode signal value from signal_flags (lower 11 bits = value)
                // 16 channels per cell: signal_flags_read[cell_idx * 16 + channel]
                let apoptosis_ch = clamp(u32(apoptosis_channel), 8u, 15u);
                let raw_signal = signal_flags_read[cell_idx * 16u + apoptosis_ch];
                let signal_value = f32(raw_signal & 0x7FFu);
                
                // Check threshold: if invert=0, trigger when signal >= threshold
                //                   if invert=1, trigger when signal < threshold
                let above_threshold = signal_value >= apoptosis_threshold;
                let condition_met = select(above_threshold, !above_threshold, apoptosis_invert > 0.5);
                
                // Only trigger if there IS a signal (raw_signal > 0) or if inverted (no signal = trigger)
                let has_signal = raw_signal > 0u;
                apoptosis_triggered = select(condition_met && has_signal, condition_met, apoptosis_invert > 0.5);
            }
        }
    }
    
    let should_die = is_dead || apoptosis_triggered;
    
    if (should_die && !was_dead) {
        // Newly dead cell - push slot to ring buffer for recycling
        death_flags[cell_idx] = 1u;
        push_free_slot(cell_idx);
        
        // Decrement live cell count
        atomicSub(&cell_count_buffer[1], 1u);
        
        // Clear division flag
        division_flags[cell_idx] = 0u;
    } else if (should_die) {
        // Already dead, ensure division flag is clear
        division_flags[cell_idx] = 0u;
    }
}

// Pass 2: Division slot allocation - pops free slots from ring buffer
// Must run AFTER death_scan so recycled slots are available
@compute @workgroup_size(256)
fn division_scan(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = atomicLoad(&cell_count_buffer[0]);
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // === DIVISION DETECTION ===
    // Read nutrients from nutrients_buffer (fixed-point i32)
    let nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    let birth_time = birth_times[cell_idx];
    let split_interval = split_intervals[cell_idx];
    let split_nutrient_threshold = split_nutrient_thresholds[cell_idx];
    let current_splits = split_counts[cell_idx];
    let max_split = max_splits[cell_idx];
    
    // Check for "never split" condition (threshold > 100 means never split)
    if (split_nutrient_threshold > 100.0) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Get cell type for behavior flags
    let mode_idx = mode_indices[cell_idx];
    let cell_type = mode_cell_types[mode_idx];
    let behavior = type_behaviors[cell_type];
    
    // Check division criteria
    let age = params.current_time - birth_time;
    let nutrients_ready = nutrients >= split_nutrient_threshold;
    let time_ready = (behavior.ignores_split_interval != 0u) || (age >= split_interval);
    let splits_remaining = max_split >= 0xFFFFFFFFu || current_splits < max_split;
    
    let wants_to_divide = nutrients_ready && time_ready && splits_remaining;
    
    if (!wants_to_divide) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Signal-conditional division gating:
    // If the mode has division_signal_channel >= 0, check if the cell's signal
    // meets the threshold condition. If not, block division.
    if (mode_idx < arrayLength(&signal_settings_v0_read)) {
        let ss_v0 = signal_settings_v0_read[mode_idx];
        let div_channel = ss_v0.x; // division_signal_channel
        if (div_channel >= 8.0) {
            let div_threshold = ss_v0.y; // division_signal_threshold
            let div_invert = ss_v0.z;    // division_signal_invert (0 or 1)
            
            // Decode signal value from signal_flags (lower 11 bits = value)
            // 16 channels per cell: signal_flags_read[cell_idx * 16 + channel]
            let div_ch = clamp(u32(div_channel), 8u, 15u);
            let raw_signal = signal_flags_read[cell_idx * 16u + div_ch];
            let signal_value = f32(raw_signal & 0x7FFu);
            
            // Check threshold: if invert=0, allow division when signal >= threshold
            //                   if invert=1, allow division when signal < threshold
            let above_threshold = signal_value >= div_threshold;
            let gate_open = select(above_threshold, !above_threshold, div_invert > 0.5);
            
            // Gate requires signal presence (unless inverted — no signal = gate open)
            let has_signal = raw_signal > 0u;
            let division_allowed = select(gate_open && has_signal, gate_open, div_invert > 0.5);
            
            if (!division_allowed) {
                division_flags[cell_idx] = 0u;
                return;
            }
        }
    }
    
    // Count active adhesions for this cell (used for both max and min checks)
    // IMPORTANT: Must verify is_active on each referenced connection, not just
    // check for a non-negative index. Adhesion indices can become stale when
    // bonds break due to force in adhesion_physics — the connection is marked
    // inactive but the per-cell index is not cleared.
    let adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    var adhesion_count = 0u;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let adh_idx = cell_adhesion_indices[adhesion_base + i];
        if (adh_idx >= 0 && adhesion_connections[u32(adh_idx)].is_active != 0u) {
            adhesion_count++;
        }
    }

    // mode_properties: v3.w = min_adhesions, v4.x = max_adhesions
    let min_adh_raw = mode_properties_v3[mode_idx].w;
    let max_adh_raw = mode_properties_v4[mode_idx].x;
    let min_adh = u32(max(min_adh_raw, 0.0));
    // max_adhesions: 0 or negative means use hardware cap (MAX_ADHESIONS_PER_CELL)
    let max_adh = select(MAX_ADHESIONS_PER_CELL, u32(max_adh_raw), max_adh_raw > 0.0);

    // Max adhesions check: must have room for the new sibling bond
    if (adhesion_count >= max_adh) {
        division_flags[cell_idx] = 0u;
        return;
    }

    // Min adhesions check: cell must have at least min_adhesions active connections to divide
    if (min_adh > 0u && adhesion_count < min_adh) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // === DEFERRAL CHECK ===
    // Check if any neighbor (via adhesion) also wants to divide.
    // If so, the cell with the HIGHER index defers to avoid race conditions
    // during adhesion inheritance in division_execute.
    // This ensures deterministic behavior: when two connected cells both want
    // to divide, only the one with the lower index divides this frame.
    // The other will divide next frame after the adhesion inheritance is complete.
    //
    // CRITICAL: This check must happen BEFORE allocate_slot() to avoid
    // consuming and leaking ring buffer slots for cells that won't divide.
    
    var should_defer = false;
    
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let adh_idx_signed = cell_adhesion_indices[adhesion_base + i];
        
        // Skip empty slots
        if (adh_idx_signed < 0) {
            continue;
        }
        
        let adh_idx = u32(adh_idx_signed);
        let conn = adhesion_connections[adh_idx];
        
        // Skip inactive connections
        if (conn.is_active == 0u) {
            continue;
        }
        
        // Get the neighbor's index
        var neighbor_idx: u32;
        if (conn.cell_a_index == cell_idx) {
            neighbor_idx = conn.cell_b_index;
        } else if (conn.cell_b_index == cell_idx) {
            neighbor_idx = conn.cell_a_index;
        } else {
            // Connection doesn't involve this cell (shouldn't happen)
            continue;
        }
        
        // Skip if neighbor is out of bounds or dead
        if (neighbor_idx >= cell_count || death_flags[neighbor_idx] == 1u) {
            continue;
        }
        
        // Re-evaluate ALL division criteria for the neighbor
        // (can't trust flags - they haven't been written yet by other threads)
        let neighbor_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[neighbor_idx]));
        let neighbor_split_nutrient_threshold = split_nutrient_thresholds[neighbor_idx];
        
        // Quick check: neighbor set to "never split" (threshold > 100)
        if (neighbor_split_nutrient_threshold > 100.0) {
            continue;
        }
        
        let neighbor_birth_time = birth_times[neighbor_idx];
        let neighbor_split_interval = split_intervals[neighbor_idx];
        let neighbor_current_splits = split_counts[neighbor_idx];
        let neighbor_max_splits = max_splits[neighbor_idx];
        let neighbor_age = params.current_time - neighbor_birth_time;
        
        // Derive neighbor's cell_type from mode (always up-to-date)
        let neighbor_mode_idx = mode_indices[neighbor_idx];
        let neighbor_cell_type = mode_cell_types[neighbor_mode_idx];
        let neighbor_behavior = type_behaviors[neighbor_cell_type];
        
        let neighbor_nutrients_ready = neighbor_nutrients >= neighbor_split_nutrient_threshold;
        let neighbor_time_ready = (neighbor_behavior.ignores_split_interval != 0u) || (neighbor_age >= neighbor_split_interval);
        let neighbor_splits_remaining = neighbor_max_splits >= 0xFFFFFFFFu || neighbor_current_splits < neighbor_max_splits;
        
        let neighbor_wants_to_divide = neighbor_nutrients_ready && neighbor_time_ready && neighbor_splits_remaining;
        
        if (!neighbor_wants_to_divide) {
            continue;
        }
        
        // Also check neighbor's adhesion constraints (max/min adhesion gates).
        // Without this, we'd defer to a neighbor that passes nutrient/time/split
        // checks but will fail the adhesion gate in its own thread — neither cell
        // would divide.
        let neighbor_adhesion_base = neighbor_idx * MAX_ADHESIONS_PER_CELL;
        var neighbor_adhesion_count = 0u;
        for (var j = 0u; j < MAX_ADHESIONS_PER_CELL; j++) {
            let n_adh_idx = cell_adhesion_indices[neighbor_adhesion_base + j];
            if (n_adh_idx >= 0 && adhesion_connections[u32(n_adh_idx)].is_active != 0u) {
                neighbor_adhesion_count++;
            }
        }
        let neighbor_min_adh_raw = mode_properties_v3[neighbor_mode_idx].w;
        let neighbor_max_adh_raw = mode_properties_v4[neighbor_mode_idx].x;
        let neighbor_min_adh = u32(max(neighbor_min_adh_raw, 0.0));
        let neighbor_max_adh = select(MAX_ADHESIONS_PER_CELL, u32(neighbor_max_adh_raw), neighbor_max_adh_raw > 0.0);
        
        if (neighbor_adhesion_count >= neighbor_max_adh) {
            continue; // Neighbor can't divide (max adhesions full)
        }
        if (neighbor_min_adh > 0u && neighbor_adhesion_count < neighbor_min_adh) {
            continue; // Neighbor can't divide (min adhesions not met)
        }
        
        // Neighbor wants to split too - lower index wins priority
        if (neighbor_idx < cell_idx) {
            should_defer = true;
            break;
        }
    }
    
    // If we should defer, don't divide this frame
    if (should_defer) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Deferral check passed - now try to allocate a slot
    let child_slot = allocate_slot();
    
    if (child_slot != 0xFFFFFFFFu) {
        // Successfully allocated slot
        division_flags[cell_idx] = 1u;
        division_slot_assignments[cell_idx] = child_slot;
        atomicAdd(&ring_state[3], 1u); // Debug counter
    } else {
        // At capacity, defer division
        division_flags[cell_idx] = 0u;
    }
}

// Separate entry point for clearing ring state (run once at start)
@compute @workgroup_size(1)
fn init_ring_state(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        // Only clear reservation count each frame, preserve head/tail/next_slot_id
        atomicStore(&ring_state[3], 0u);
    }
}
