// Lifecycle Division Execute Shader
// Stage 4: Execute cell division using assigned slots
// Workgroup size: 64 threads for cell operations
//
// Input:
// - division_flags: which cells are dividing
// - division_slot_assignments: reservation index for each dividing cell
// - free_slot_indices: compacted array of available slots
// - lifecycle_counts[0] = N (free slots), lifecycle_counts[1] = M (reservations)
//
// Assignment Formula (from diagram):
//   For each reservation i:
//     assignments[i] = freeSlots[reservations[i]]
//
// Availability Check: N >= M
// - If yes: proceed with division
// - If no: division fails (cells keep growing, death still happens)
//
// Division creates:
// - Child A: overwrites parent slot, keeps genome reference, 50% mass
// - Child B: placed in assigned free slot, copies genome, 50% mass
//
// Both children:
// - Offset position along deterministic split direction
// - Inherit parent velocity

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
    _padding2: vec3<f32>,
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

@group(1) @binding(4)
var<storage, read_write> lifecycle_counts: array<u32>;

// Cell state bind group (group 2) - read/write for updating after division
@group(2) @binding(0)
var<storage, read_write> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read_write> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read_write> split_masses: array<f32>;

@group(2) @binding(3)
var<storage, read_write> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read_write> max_splits: array<u32>;

@group(2) @binding(5)
var<storage, read_write> genome_ids: array<u32>;

@group(2) @binding(6)
var<storage, read_write> mode_indices: array<u32>;

@group(2) @binding(7)
var<storage, read_write> cell_ids: array<u32>;

@group(2) @binding(8)
var<storage, read_write> next_cell_id: array<u32>;

@group(2) @binding(9)
var<storage, read_write> nutrient_gain_rates: array<f32>;

const PI: f32 = 3.14159265359;

// Deterministic pseudo-random direction based on cell index and frame
// Same input → Same output (reproducible)
fn deterministic_split_direction(cell_idx: u32, frame: i32) -> vec3<f32> {
    // Use golden ratio based distribution for uniform sphere sampling
    let golden_ratio = 1.618033988749895;
    let seed = f32(cell_idx) * golden_ratio + f32(frame) * 0.1;
    
    // Spherical coordinates
    let theta = seed * 2.0 * PI;
    let phi = acos(1.0 - 2.0 * fract(seed * golden_ratio));
    
    return vec3<f32>(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    );
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    let volume = mass / 1.0;
    return pow(volume * 3.0 / (4.0 * PI), 1.0 / 3.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Check if this cell is dividing
    let is_dividing = division_flags[cell_idx] == 1u;
    if (!is_dividing) {
        // Not dividing - nothing to do
        return;
    }
    
    // Get counts: N = free slots, M = reservations, D = dead cells
    let free_slot_count = lifecycle_counts[0]; // N = total free slots
    let reservation_count = lifecycle_counts[1]; // M = cells wanting to divide
    let dead_count = lifecycle_counts[2]; // D = dead cell slots (from prefix sum)
    
    // Availability check: N >= M ?
    // If not enough free slots, division fails for ALL cells
    if (free_slot_count < reservation_count) {
        // Division fails - cells keep growing, death still happens
        return;
    }
    
    // Get this cell's reservation index
    let reservation_idx = division_slot_assignments[cell_idx];
    
    // Compute the actual slot index:
    // - If reservation_idx < dead_count: use free_slot_indices[reservation_idx] (dead cell slot)
    // - Otherwise: use cell_count + (reservation_idx - dead_count) (unused capacity slot)
    var child_b_slot: u32;
    var is_new_slot = false;
    if (reservation_idx < dead_count) {
        // Use a dead cell's slot
        child_b_slot = free_slot_indices[reservation_idx];
    } else {
        // Use an unused capacity slot - this increases cell count
        child_b_slot = cell_count + (reservation_idx - dead_count);
        is_new_slot = true;
    }
    
    // Parent state - read from OUTPUT buffer (physics results)
    let parent_pos = positions_out[cell_idx].xyz;
    let parent_mass = positions_out[cell_idx].w;
    let parent_vel = velocities_out[cell_idx].xyz;
    
    // Split mass 50/50
    let child_mass = parent_mass * 0.5;
    let child_radius = calculate_radius_from_mass(child_mass);
    
    // Deterministic split direction
    let split_dir = deterministic_split_direction(cell_idx, params.current_frame);
    
    // Offset children from parent position
    let offset = child_radius * 1.1; // Slight separation to avoid immediate collision
    let child_a_pos = parent_pos - split_dir * offset;
    let child_b_pos = parent_pos + split_dir * offset;
    
    // === Create Child A (overwrites parent slot) ===
    positions_out[cell_idx] = vec4<f32>(child_a_pos, child_mass);
    velocities_out[cell_idx] = vec4<f32>(parent_vel, 0.0);
    
    // Update Child A state
    birth_times[cell_idx] = params.current_time;
    split_counts[cell_idx] = split_counts[cell_idx] + 1u;
    // Child A keeps genome reference (genome_ids unchanged)
    // Child A keeps mode (mode_indices unchanged)
    
    // === Create Child B (placed in assigned free slot) ===
    positions_out[child_b_slot] = vec4<f32>(child_b_pos, child_mass);
    velocities_out[child_b_slot] = vec4<f32>(parent_vel, 0.0);
    
    // Copy state to Child B
    birth_times[child_b_slot] = params.current_time;
    split_intervals[child_b_slot] = split_intervals[cell_idx];
    split_masses[child_b_slot] = split_masses[cell_idx];
    split_counts[child_b_slot] = 0u; // Child B starts fresh
    max_splits[child_b_slot] = max_splits[cell_idx];
    genome_ids[child_b_slot] = genome_ids[cell_idx]; // Copy genome reference
    mode_indices[child_b_slot] = mode_indices[cell_idx];
    nutrient_gain_rates[child_b_slot] = nutrient_gain_rates[cell_idx]; // Copy nutrient gain rate
    
    // Assign new cell ID to Child B
    cell_ids[child_b_slot] = cell_count + cell_idx;
    
    // Clear death flag for the new slot (it's now occupied by Child B)
    death_flags[child_b_slot] = 0u;
    
    // Update cell count if we used a new slot (not a dead cell slot)
    // Only the first dividing cell (reservation_idx == 0) updates the count
    // to avoid race conditions. It adds the total number of new cells.
    if (reservation_idx == 0u && reservation_count > dead_count) {
        // Number of new slots used = reservation_count - dead_count
        let new_cells = reservation_count - dead_count;
        cell_count_buffer[0] = cell_count + new_cells;
    }
}
