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

@group(2) @binding(10)
var<storage, read_write> stiffnesses: array<f32>;

// Rotations input (from current buffer)
@group(2) @binding(11)
var<storage, read> rotations_in: array<vec4<f32>>;

// Rotations output (to next buffer)
@group(2) @binding(12)
var<storage, read_write> rotations_out: array<vec4<f32>>;

// Genome mode data: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)] per mode
// Total 48 bytes per mode, indexed by mode_index
@group(2) @binding(13)
var<storage, read> genome_mode_data: array<vec4<f32>>;

const PI: f32 = 3.14159265359;

// Rotate a vector by a quaternion
fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qvec = q.xyz;
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    let volume = mass / 1.0;
    return pow(volume * 3.0 / (4.0 * PI), 1.0 / 3.0);
}

// Quaternion multiplication: result = a * b
fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
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
    let parent_rotation = rotations_in[cell_idx];
    
    // Get parent's mode index for looking up child orientations and split direction
    let parent_mode_idx = mode_indices[cell_idx];
    
    // Read child orientations and split direction from genome mode data
    // Layout: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)] per mode
    // Each mode takes 3 vec4s (indices mode_idx * 3, mode_idx * 3 + 1, mode_idx * 3 + 2)
    let child_a_orientation = genome_mode_data[parent_mode_idx * 3u];
    let child_b_orientation = genome_mode_data[parent_mode_idx * 3u + 1u];
    let split_direction_local = genome_mode_data[parent_mode_idx * 3u + 2u].xyz;
    
    // Calculate child rotations: parent_rotation * child_orientation
    let child_a_rotation = quat_multiply(parent_rotation, child_a_orientation);
    let child_b_rotation = quat_multiply(parent_rotation, child_b_orientation);
    
    // Calculate parent radius for offset calculation (before mass split)
    let parent_radius = calculate_radius_from_mass(parent_mass);
    
    // Split mass 50/50
    let child_mass = parent_mass * 0.5;
    
    // Transform split direction from local to world space using parent rotation
    // If split direction is zero (default), use Z axis
    var split_dir_local = split_direction_local;
    if (length(split_dir_local) < 0.0001) {
        split_dir_local = vec3<f32>(0.0, 0.0, 1.0);
    }
    let split_dir = normalize(rotate_vector_by_quat(split_dir_local, parent_rotation));
    
    // 75% overlap means centers are 25% of combined diameter apart
    // Match preview scene: offset_distance = parent_radius * 0.25
    // Match reference convention: Child A at +offset, Child B at -offset
    let offset = parent_radius * 0.25;
    let child_a_pos = parent_pos + split_dir * offset;
    let child_b_pos = parent_pos - split_dir * offset;
    
    // === Create Child A (overwrites parent slot) ===
    positions_out[cell_idx] = vec4<f32>(child_a_pos, child_mass);
    velocities_out[cell_idx] = vec4<f32>(parent_vel, 0.0);
    rotations_out[cell_idx] = child_a_rotation;
    
    // Update Child A state
    birth_times[cell_idx] = params.current_time;
    split_counts[cell_idx] = split_counts[cell_idx] + 1u;
    // Child A keeps genome reference (genome_ids unchanged)
    // Child A keeps mode (mode_indices unchanged)
    
    // === Create Child B (placed in assigned free slot) ===
    positions_out[child_b_slot] = vec4<f32>(child_b_pos, child_mass);
    velocities_out[child_b_slot] = vec4<f32>(parent_vel, 0.0);
    rotations_out[child_b_slot] = child_b_rotation;
    
    // Copy state to Child B
    birth_times[child_b_slot] = params.current_time;
    split_intervals[child_b_slot] = split_intervals[cell_idx];
    split_masses[child_b_slot] = split_masses[cell_idx];
    split_counts[child_b_slot] = 0u; // Child B starts fresh
    max_splits[child_b_slot] = max_splits[cell_idx];
    genome_ids[child_b_slot] = genome_ids[cell_idx]; // Copy genome reference
    mode_indices[child_b_slot] = mode_indices[cell_idx];
    nutrient_gain_rates[child_b_slot] = nutrient_gain_rates[cell_idx]; // Copy nutrient gain rate
    stiffnesses[child_b_slot] = stiffnesses[cell_idx]; // Copy membrane stiffness
    
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
