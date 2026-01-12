// Lifecycle Division Scan Shader
// Stage 3: Identify cells ready to divide using atomic operations
// Workgroup size: 64 threads for cell operations
//
// Input: 
// - Cell state (mass, birth_time, split_interval, split_mass)
// - lifecycle_counts[2] = D (dead cell count from prefix_sum)
//
// Output:
// - division_flags[cell_idx] = 1 for cells ready to divide
// - division_slot_assignments[cell_idx] = reservation index for this cell
// - lifecycle_counts[0] = N (total free slots = dead + unused capacity)
// - lifecycle_counts[1] = M (total reservations needed)
//
// Uses atomic operations for O(1) per thread instead of O(N) prefix sum
// NOTE: lifecycle_counts[0] and [1] are cleared in death_scan shader

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

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Lifecycle bind group (group 1)
@group(1) @binding(0)
var<storage, read_write> death_flags: array<u32>;

@group(1) @binding(1)
var<storage, read_write> division_flags: array<u32>;

@group(1) @binding(2)
var<storage, read_write> free_slot_indices: array<u32>;

@group(1) @binding(3)
var<storage, read_write> division_slot_assignments: array<u32>;

// Atomic counters: [0] = free slots, [1] = reservations, [2] = dead count
@group(1) @binding(4)
var<storage, read_write> lifecycle_counts: array<atomic<u32>>;

// Cell state bind group (group 2)
@group(2) @binding(0)
var<storage, read> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read> split_masses: array<f32>;

@group(2) @binding(3)
var<storage, read> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read> max_splits: array<u32>;

// Cell types: 0 = Test, 1 = Flagellocyte
// DEPRECATED - use mode_cell_types instead for up-to-date values
@group(2) @binding(5)
var<storage, read> cell_types: array<u32>;

// Mode indices (per-cell mode index)
@group(2) @binding(6)
var<storage, read> mode_indices: array<u32>;

// Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
// Always up-to-date with genome settings, unlike cell_types buffer
// Flagellocytes split based on mass only (ignore split_interval)
@group(2) @binding(7)
var<storage, read> mode_cell_types: array<u32>;

// Adhesion bind group (group 3) - for checking neighbor division status
@group(3) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read> cell_adhesion_indices: array<i32>;

// Adhesion connection structure (104 bytes matching Rust GpuAdhesionConnection)
// IMPORTANT: Use vec4 for anchor directions because vec3 has 16-byte alignment in WGSL
// which would cause layout mismatch with Rust's [f32; 3] + f32 padding
struct AdhesionConnection {
    cell_a_index: u32,          // offset 0
    cell_b_index: u32,          // offset 4
    mode_index: u32,            // offset 8
    is_active: u32,             // offset 12
    zone_a: u32,                // offset 16
    zone_b: u32,                // offset 20
    _align_pad: vec2<u32>,      // offset 24-31 (8 bytes)
    anchor_direction_a: vec4<f32>,  // offset 32-47 (xyz = direction, w = padding)
    anchor_direction_b: vec4<f32>,  // offset 48-63 (xyz = direction, w = padding)
    twist_reference_a: vec4<f32>,   // offset 64-79
    twist_reference_b: vec4<f32>,   // offset 80-95
    _padding: vec2<u32>,            // offset 96-103
};

const MAX_ADHESIONS_PER_CELL: u32 = 10u;

// Temporary storage for division candidates (before deferral check)
// We use division_flags as a two-pass system:
// Pass 1: Mark all cells that WANT to divide (wants_to_divide)
// Pass 2: Check neighbors and defer if needed

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    
    // First thread calculates total free slots
    // This must happen before any thread tries to reserve a slot
    if (cell_idx == 0u) {
        // Total free slots = dead cells + unused capacity
        let dead_count = atomicLoad(&lifecycle_counts[2]);
        let unused_capacity = params.cell_capacity - cell_count;
        atomicStore(&lifecycle_counts[0], dead_count + unused_capacity);
    }
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Skip dead cells - they can't divide
    if (death_flags[cell_idx] == 1u) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Read mass from OUTPUT buffer (physics results)
    let mass = positions_out[cell_idx].w;
    let birth_time = birth_times[cell_idx];
    let split_interval = split_intervals[cell_idx];
    let split_mass = split_masses[cell_idx];
    let current_splits = split_counts[cell_idx];
    let max_split = max_splits[cell_idx];
    
    // Derive cell_type from mode (always up-to-date with genome settings)
    let mode_idx = mode_indices[cell_idx];
    let cell_type = mode_cell_types[mode_idx];
    
    // Check for "never split" condition (split_mass > 3.0)
    if (split_mass > 3.0) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Calculate age
    let age = params.current_time - birth_time;
    
    // Check division criteria
    let mass_ready = mass >= split_mass;
    // Flagellocytes (cell_type == 1) split based on mass only, ignore split_interval
    // Test cells (cell_type == 0) require both mass and time conditions
    let time_ready = (cell_type == 1u) || (age >= split_interval);
    let splits_remaining = current_splits < max_split || max_split == 0u;
    
    let wants_to_divide = mass_ready && time_ready && splits_remaining;
    
    // If this cell doesn't want to divide, we're done
    if (!wants_to_divide) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Count current adhesions (matching reference: adhesionCount >= mode.maxAdhesions check)
    // If cell already has too many adhesions, it cannot split (splitting creates new sibling adhesion)
    let adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    var adhesion_count = 0u;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        if (cell_adhesion_indices[adhesion_base + i] >= 0) {
            adhesion_count++;
        }
    }
    
    // If cell has max adhesions, it cannot split (no room for sibling bond)
    // Using MAX_ADHESIONS_PER_CELL - 1 to leave room for the new sibling adhesion
    if (adhesion_count >= MAX_ADHESIONS_PER_CELL) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // === DEFERRAL CHECK ===
    // Check if any neighbor (via adhesion) also wants to divide.
    // If so, the cell with the HIGHER index defers to avoid race conditions
    // during adhesion inheritance in division_execute.
    //
    // This ensures deterministic behavior: when two connected cells both want
    // to divide, only the one with the lower index divides this frame.
    // The other will divide next frame after the adhesion inheritance is complete.
    
    var should_defer = false;
    // Note: adhesion_base already calculated above for adhesion count check
    
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
        
        // Check if neighbor is also ready to split (matching reference exactly)
        // Need to check ALL division criteria, not just time readiness
        let neighbor_birth_time = birth_times[neighbor_idx];
        let neighbor_split_interval = split_intervals[neighbor_idx];
        let neighbor_split_mass = split_masses[neighbor_idx];
        let neighbor_mass = positions_out[neighbor_idx].w;
        let neighbor_age = params.current_time - neighbor_birth_time;
        let neighbor_current_splits = split_counts[neighbor_idx];
        let neighbor_max_splits = max_splits[neighbor_idx];
        
        // Derive neighbor's cell_type from mode (always up-to-date)
        let neighbor_mode_idx = mode_indices[neighbor_idx];
        let neighbor_cell_type = mode_cell_types[neighbor_mode_idx];
        
        // Check if neighbor is set to "never split" (split_mass > 3.0)
        if (neighbor_split_mass > 3.0) {
            continue; // Neighbor will never split, no need to defer
        }
        
        // Check all division criteria for neighbor
        let neighbor_mass_ready = neighbor_mass >= neighbor_split_mass;
        let neighbor_time_ready = (neighbor_cell_type == 1u) || (neighbor_age >= neighbor_split_interval);
        let neighbor_splits_remaining = neighbor_current_splits < neighbor_max_splits || neighbor_max_splits == 0u;
        
        let neighbor_wants_to_divide = neighbor_mass_ready && neighbor_time_ready && neighbor_splits_remaining;
        
        // Skip if neighbor doesn't actually want to split
        if (!neighbor_wants_to_divide) {
            continue;
        }
        
        // Neighbor wants to split - compare priority
        // Lower index = higher priority (lower priority value)
        // If neighbor has lower index (higher priority), we defer
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
    
    // This cell can divide - write division flag and get reservation
    division_flags[cell_idx] = 1u;
    
    // Atomically get a reservation index
    let reservation_idx = atomicAdd(&lifecycle_counts[1], 1u);
    division_slot_assignments[cell_idx] = reservation_idx;
}
