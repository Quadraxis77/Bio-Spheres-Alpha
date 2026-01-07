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
var<storage, read_write> next_cell_id: array<atomic<u32>>;

@group(2) @binding(9)
var<storage, read_write> nutrient_gain_rates: array<f32>;

@group(2) @binding(10)
var<storage, read_write> max_cell_sizes: array<f32>;

@group(2) @binding(11)
var<storage, read_write> stiffnesses: array<f32>;

// Rotations input (from current buffer)
@group(2) @binding(12)
var<storage, read> rotations_in: array<vec4<f32>>;

// Rotations output (to next buffer)
@group(2) @binding(13)
var<storage, read_write> rotations_out: array<vec4<f32>>;

// Genome mode data: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)] per mode
// Total 48 bytes per mode, indexed by mode_index
@group(2) @binding(14)
var<storage, read> genome_mode_data: array<vec4<f32>>;

// Parent make adhesion flags: one bool per mode (stored as u32)
@group(2) @binding(15)
var<storage, read> parent_make_adhesion_flags: array<u32>;

// Adhesion creation buffers (group 3)
@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read_write> adhesion_counts: array<atomic<u32>>;

@group(3) @binding(2)
var<storage, read_write> cell_adhesion_indices: array<i32>;

@group(3) @binding(3)
var<storage, read_write> free_adhesion_slots: array<u32>;

// Adhesion connection structure (96 bytes, matching adhesion_physics.wgsl)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    anchor_direction_a: vec3<f32>,
    padding_a: f32,
    anchor_direction_b: vec3<f32>,
    padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    _padding: vec2<u32>,
};

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

@compute @workgroup_size(128)
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
    
    // Get this cell's reservation index
    let reservation_idx = division_slot_assignments[cell_idx];
    
    // Partial division: only cells with reservation_idx < free_slot_count can divide
    // This allows some cells to divide even when there aren't enough slots for all
    // Cells that don't get a slot will keep growing and try again next frame
    if (reservation_idx >= free_slot_count) {
        // This cell didn't get a slot - skip division, try again later
        return;
    }
    
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
    // Assign new cell ID to Child A using atomic increment for uniqueness
    let child_a_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[cell_idx] = child_a_id;
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
    max_cell_sizes[child_b_slot] = max_cell_sizes[cell_idx]; // Copy max cell size
    stiffnesses[child_b_slot] = stiffnesses[cell_idx]; // Copy membrane stiffness
    
    // Assign new cell ID to Child B using atomic increment for uniqueness
    let child_b_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[child_b_slot] = child_b_id;
    
    // Clear death flag for the new slot (it's now occupied by Child B)
    death_flags[child_b_slot] = 0u;
    
    // === Create Sibling Adhesion ===
    // Check if parent mode allows adhesion creation
    let global_mode_index = mode_indices[cell_idx]; // This is already the global mode index
    let make_adhesion = parent_make_adhesion_flags[global_mode_index] == 1u;
    
    if (make_adhesion) {
        // Try to allocate an adhesion slot atomically
        let total_adhesions = atomicAdd(&adhesion_counts[0], 1u);
        
        // Check if we have capacity (simplified check - in full implementation would use free slot stack)
        if (total_adhesions < 100000u) { // MAX_ADHESION_CONNECTIONS
            let adhesion_slot = total_adhesions;
            
            // Calculate anchor directions in local cell space
            // Child A anchor points toward Child B (negative split direction)
            // Child B anchor points toward Child A (positive split direction)
            let anchor_a_local = -split_dir_local; // Child A points toward Child B
            let anchor_b_local = split_dir_local;  // Child B points toward Child A
            
            // Normalize anchor directions
            let anchor_a_normalized = normalize(anchor_a_local);
            let anchor_b_normalized = normalize(anchor_b_local);
            
            // Store twist reference quaternions from child orientations at division time
            let twist_ref_a = child_a_rotation;
            let twist_ref_b = child_b_rotation;
            
            // Create the adhesion connection
            var connection: AdhesionConnection;
            connection.cell_a_index = cell_idx;        // Child A index
            connection.cell_b_index = child_b_slot;    // Child B index
            connection.mode_index = global_mode_index; // Mode index for settings lookup
            connection.is_active = 1u;                 // Active connection
            connection.zone_a = 1u;                    // Zone B for Child A (positive split direction)
            connection.zone_b = 0u;                    // Zone A for Child B (negative split direction)
            connection.anchor_direction_a = anchor_a_normalized;
            connection.padding_a = 0.0;
            connection.anchor_direction_b = anchor_b_normalized;
            connection.padding_b = 0.0;
            connection.twist_reference_a = twist_ref_a;
            connection.twist_reference_b = twist_ref_b;
            connection._padding = vec2<u32>(0u, 0u);
            
            // Store the connection
            adhesion_connections[adhesion_slot] = connection;
            
            // Update per-cell adhesion indices (simplified - assumes first slot is free)
            // In full implementation, would search for free slot in the 20-element array
            let cell_a_base = cell_idx * 20u;
            let cell_b_base = child_b_slot * 20u;
            
            // Find first free slot for Child A and add adhesion index
            for (var i = 0u; i < 20u; i++) {
                if (cell_adhesion_indices[cell_a_base + i] == -1) {
                    cell_adhesion_indices[cell_a_base + i] = i32(adhesion_slot);
                    break;
                }
            }
            
            // Find first free slot for Child B and add adhesion index
            for (var i = 0u; i < 20u; i++) {
                if (cell_adhesion_indices[cell_b_base + i] == -1) {
                    cell_adhesion_indices[cell_b_base + i] = i32(adhesion_slot);
                    break;
                }
            }
            
            // Update live adhesion count
            atomicAdd(&adhesion_counts[1], 1u);
        } else {
            // Failed to allocate - decrement the count we just incremented
            atomicSub(&adhesion_counts[0], 1u); // Subtract 1 atomically
        }
    }
    
    // Update cell count if we used a new slot (not a dead cell slot)
    // Only the first dividing cell (reservation_idx == 0) updates the count
    // to avoid race conditions. It adds the total number of new cells.
    // With partial division, we cap at free_slot_count divisions
    if (reservation_idx == 0u) {
        let actual_divisions = min(reservation_count, free_slot_count);
        if (actual_divisions > dead_count) {
            // Number of new slots used = actual_divisions - dead_count
            let new_cells = actual_divisions - dead_count;
            cell_count_buffer[0] = cell_count + new_cells;
        }
    }
}
