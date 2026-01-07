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

// Child mode indices: [child_a_mode, child_b_mode] per mode (stored as i32)
@group(2) @binding(16)
var<storage, read> child_mode_indices: array<vec2<i32>>;

// Mode properties: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, padding, padding, padding] per mode
// Total 32 bytes (8 floats) per mode, indexed by mode_index
@group(2) @binding(17)
var<storage, read> mode_properties: array<vec4<f32>>;

// Adhesion creation buffers (group 3)
@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read_write> adhesion_counts: array<atomic<u32>>;

@group(3) @binding(2)
var<storage, read_write> cell_adhesion_indices: array<i32>;

@group(3) @binding(3)
var<storage, read_write> free_adhesion_slots: array<u32>;

// Child A keep adhesion flags: one bool per mode (stored as u32)
@group(3) @binding(4)
var<storage, read> child_a_keep_adhesion_flags: array<u32>;

// Child B keep adhesion flags: one bool per mode (stored as u32)
@group(3) @binding(5)
var<storage, read> child_b_keep_adhesion_flags: array<u32>;

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

const PI: f32 = 3.14159265359;

// Zone classification constants
const ZONE_A: u32 = 0u; // Negative dot product - goes to Child B
const ZONE_B: u32 = 1u; // Positive dot product - goes to Child A
const ZONE_C: u32 = 2u; // Equatorial - duplicated for both children

// Inheritance angle threshold (degrees) - defines equatorial zone width
// Reference Biospheres-Master uses 2°
const EQUATORIAL_THRESHOLD_DEG: f32 = 2.0;

// Maximum adhesions per cell
const MAX_ADHESIONS_PER_CELL: u32 = 20u;

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

// Quaternion inverse (for unit quaternions, inverse = conjugate)
fn quat_inverse(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// Classify adhesion zone based on anchor direction and split direction
// Matches reference implementation EXACTLY:
// - Uses dot product threshold: sin(radians(2.0)) ≈ 0.0349
// - Zone C: abs(dot) <= threshold (almost perpendicular to split direction)
// - Zone B: dot > 0 (pointing same direction as split) -> goes to Child A
// - Zone A: dot < 0 (pointing opposite to split) -> goes to Child B
fn classify_adhesion_zone(anchor_dir_local: vec3<f32>, split_dir_local: vec3<f32>) -> u32 {
    let dot_product = dot(normalize(anchor_dir_local), normalize(split_dir_local));
    
    // Reference uses: sin(radians(2.0)) as threshold
    // sin(2°) ≈ 0.0349
    let equatorial_threshold = sin(EQUATORIAL_THRESHOLD_DEG * PI / 180.0);
    
    // Zone classification based on dot product (matches reference exactly)
    if (abs(dot_product) <= equatorial_threshold) {
        return ZONE_C; // Equatorial - almost perpendicular to split direction
    } else if (dot_product > 0.0) {
        return ZONE_B; // Positive dot product (same direction as split) -> Child A
    } else {
        return ZONE_A; // Negative dot product (opposite to split) -> Child B
    }
}

// Calculate child anchor direction using geometric approach (matches reference)
// child_pos_parent_frame: child position in parent's local frame
// neighbor_pos_parent_frame: neighbor position in parent's local frame (from anchor * distance)
// child_orientation_delta: child's orientation relative to parent (from genome mode)
fn calculate_child_anchor_direction(
    child_pos_parent_frame: vec3<f32>,
    neighbor_pos_parent_frame: vec3<f32>,
    child_orientation_delta: vec4<f32>
) -> vec3<f32> {
    // Direction from child to neighbor in parent frame
    let direction_to_neighbor = normalize(neighbor_pos_parent_frame - child_pos_parent_frame);
    // Transform to child's local space using inverse of orientation delta
    let inv_delta = quat_inverse(child_orientation_delta);
    return normalize(rotate_vector_by_quat(direction_to_neighbor, inv_delta));
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
    // Assign new cell ID to Child A using atomic increment for uniqueness
    let child_a_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[cell_idx] = child_a_id;
    // Child A keeps genome reference (genome_ids unchanged)
    // Child A gets its designated mode from genome
    let child_modes = child_mode_indices[parent_mode_idx];
    let child_a_mode_idx = u32(max(child_modes.x, 0));
    let child_b_mode_idx = u32(max(child_modes.y, 0));
    
    // Split count: reset to 0 if mode changes, otherwise increment
    let parent_split_count = split_counts[cell_idx];
    if (child_a_mode_idx != parent_mode_idx) {
        split_counts[cell_idx] = 0u;
    } else {
        split_counts[cell_idx] = parent_split_count + 1u;
    }
    mode_indices[cell_idx] = child_a_mode_idx;
    
    // Read Child A's properties from its mode
    // mode_properties layout: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval] (vec4)
    //                         [split_mass, padding, padding, padding] (vec4)
    let child_a_props_0 = mode_properties[child_a_mode_idx * 2u];
    let child_a_props_1 = mode_properties[child_a_mode_idx * 2u + 1u];
    nutrient_gain_rates[cell_idx] = child_a_props_0.x;
    max_cell_sizes[cell_idx] = child_a_props_0.y;
    stiffnesses[cell_idx] = child_a_props_0.z;
    split_intervals[cell_idx] = child_a_props_0.w;
    split_masses[cell_idx] = child_a_props_1.x;
    
    // === Create Child B (placed in assigned free slot) ===
    positions_out[child_b_slot] = vec4<f32>(child_b_pos, child_mass);
    velocities_out[child_b_slot] = vec4<f32>(parent_vel, 0.0);
    rotations_out[child_b_slot] = child_b_rotation;
    
    // Read Child B's properties from its mode
    let child_b_props_0 = mode_properties[child_b_mode_idx * 2u];
    let child_b_props_1 = mode_properties[child_b_mode_idx * 2u + 1u];
    
    // Set Child B state from its mode properties
    birth_times[child_b_slot] = params.current_time;
    split_intervals[child_b_slot] = child_b_props_0.w;
    split_masses[child_b_slot] = child_b_props_1.x;
    // Split count: reset to 0 if mode changes from parent, otherwise inherit parent's count + 1
    if (child_b_mode_idx != parent_mode_idx) {
        split_counts[child_b_slot] = 0u;
    } else {
        split_counts[child_b_slot] = parent_split_count + 1u;
    }
    max_splits[child_b_slot] = max_splits[cell_idx]; // TODO: Read from mode if needed
    genome_ids[child_b_slot] = genome_ids[cell_idx]; // Copy genome reference
    mode_indices[child_b_slot] = child_b_mode_idx; // Child B gets its designated mode
    nutrient_gain_rates[child_b_slot] = child_b_props_0.x;
    max_cell_sizes[child_b_slot] = child_b_props_0.y;
    stiffnesses[child_b_slot] = child_b_props_0.z;
    
    // Assign new cell ID to Child B using atomic increment for uniqueness
    let child_b_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[child_b_slot] = child_b_id;
    
    // Clear death flag for the new slot (it's now occupied by Child B)
    death_flags[child_b_slot] = 0u;
    
    // === Create Sibling Adhesion ===
    // Check if PARENT mode allows adhesion creation (use parent_mode_idx, not the updated mode_indices)
    let make_adhesion = parent_make_adhesion_flags[parent_mode_idx] == 1u;
    
    // Get keep_adhesion flags for zone-based inheritance (from parent mode)
    let child_a_keep = child_a_keep_adhesion_flags[parent_mode_idx] == 1u;
    let child_b_keep = child_b_keep_adhesion_flags[parent_mode_idx] == 1u;
    
    // Track the sibling adhesion slot (if created) to skip it during inheritance
    var sibling_adhesion_slot: u32 = 0xFFFFFFFFu; // Invalid slot by default
    
    if (make_adhesion) {
        // Try to allocate an adhesion slot atomically
        let total_adhesions = atomicAdd(&adhesion_counts[0], 1u);
        
        // Check if we have capacity (simplified check - in full implementation would use free slot stack)
        if (total_adhesions < 100000u) { // MAX_ADHESION_CONNECTIONS
            let adhesion_slot = total_adhesions;
            sibling_adhesion_slot = adhesion_slot; // Save for later skip check
            
            // Calculate anchor directions in each child's LOCAL space
            // CRITICAL: Must transform by child orientation inverse (matches CPU exactly)
            // 
            // CPU code (division.rs):
            //   let direction_a_to_b_parent_local = -data.split_direction_local;
            //   let direction_b_to_a_parent_local = data.split_direction_local;
            //   let anchor_direction_a = (mode.child_a.orientation.inverse() * direction_a_to_b_parent_local).normalize();
            //   let anchor_direction_b = (mode.child_b.orientation.inverse() * direction_b_to_a_parent_local).normalize();
            //
            // Child A is at +offset, Child B is at -offset
            // Child A points toward B (at -offset): -splitDirLocal
            // Child B points toward A (at +offset): +splitDirLocal
            let direction_a_to_b_parent_local = -split_dir_local;
            let direction_b_to_a_parent_local = split_dir_local;
            
            // Transform to each child's local space using inverse of child orientation
            // child_a_orientation and child_b_orientation are the orientation deltas from parent
            let inv_child_a_orientation = quat_inverse(child_a_orientation);
            let inv_child_b_orientation = quat_inverse(child_b_orientation);
            
            let anchor_a_local = normalize(rotate_vector_by_quat(direction_a_to_b_parent_local, inv_child_a_orientation));
            let anchor_b_local = normalize(rotate_vector_by_quat(direction_b_to_a_parent_local, inv_child_b_orientation));
            
            // Store twist reference quaternions from child orientations at division time
            let twist_ref_a = child_a_rotation;
            let twist_ref_b = child_b_rotation;
            
            // Create the adhesion connection
            var connection: AdhesionConnection;
            connection.cell_a_index = cell_idx;        // Child A index
            connection.cell_b_index = child_b_slot;    // Child B index
            connection.mode_index = parent_mode_idx;   // Parent's mode index for settings lookup
            connection.is_active = 1u;                 // Active connection
            connection.zone_a = 1u;                    // Zone B for Child A (positive split direction)
            connection.zone_b = 0u;                    // Zone A for Child B (negative split direction)
            connection._align_pad = vec2<u32>(0u, 0u);
            connection.anchor_direction_a = vec4<f32>(anchor_a_local, 0.0);
            connection.anchor_direction_b = vec4<f32>(anchor_b_local, 0.0);
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
    
    // === Zone-Based Adhesion Inheritance ===
    // Process parent's existing adhesions and distribute to children based on zone classification
    // Matches reference implementation (BioSpheres-Q/Biospheres-Master):
    // - Zone A (negative dot with split dir) -> Child B
    // - Zone B (positive dot with split dir) -> Child A  
    // - Zone C (equatorial) -> Both children (duplicate)
    // - Original connections are marked inactive AFTER processing
    // - Side assignment is preserved (if neighbor was cellA, stays cellA)
    
    let parent_adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let child_a_adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let child_b_adhesion_base = child_b_slot * MAX_ADHESIONS_PER_CELL;
    
    // IMPORTANT: Save parent's adhesion indices BEFORE clearing them
    // This avoids race conditions with other dividing cells
    var parent_adhesion_indices: array<i32, 20>;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        parent_adhesion_indices[i] = cell_adhesion_indices[parent_adhesion_base + i];
    }
    
    // Now clear Child A's adhesion indices (parent slot) - we'll rebuild them
    // Note: Child B's slot was already cleared when it was a dead/new slot
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        cell_adhesion_indices[child_a_adhesion_base + i] = -1;
        cell_adhesion_indices[child_b_adhesion_base + i] = -1;
    }
    
    // Track how many adhesions each child has
    var child_a_adhesion_count = 0u;
    var child_b_adhesion_count = 0u;
    
    // Re-add the sibling adhesion we just created (if it was created successfully)
    if (sibling_adhesion_slot != 0xFFFFFFFFu) {
        // Add to Child A
        if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
            cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count] = i32(sibling_adhesion_slot);
            child_a_adhesion_count++;
        }
        
        // Add to Child B
        if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
            cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count] = i32(sibling_adhesion_slot);
            child_b_adhesion_count++;
        }
    }
    
    // Calculate geometric parameters for anchor calculation (matches reference)
    // Use the same offset as the actual child positions
    let child_offset = parent_radius * 0.25;
    
    // Child positions in parent frame (matches reference convention)
    // Child A at +offset, Child B at -offset
    // split_dir_local is already normalized or set to default (0,0,1) earlier
    let split_dir_normalized = normalize(split_dir_local);
    let child_a_pos_parent_frame = split_dir_normalized * child_offset;
    let child_b_pos_parent_frame = -split_dir_normalized * child_offset;
    
    // Fixed radius for adhesion distance calculation (matches reference: independent of cell growth)
    let FIXED_RADIUS: f32 = 1.0;
    
    // Process inherited adhesions from parent using saved indices
    // This avoids race conditions with other dividing cells
    for (var parent_slot = 0u; parent_slot < MAX_ADHESIONS_PER_CELL; parent_slot++) {
        let adh_idx_signed = parent_adhesion_indices[parent_slot];
        
        // Skip empty slots
        if (adh_idx_signed < 0) {
            continue;
        }
        
        let adh_idx = u32(adh_idx_signed);
        
        // Skip the sibling adhesion we just created (it's already handled)
        // Use the saved sibling_adhesion_slot to avoid race conditions with other dividing cells
        if (adh_idx == sibling_adhesion_slot) {
            continue;
        }
        
        let conn = adhesion_connections[adh_idx];
        
        // Skip inactive connections
        if (conn.is_active == 0u) {
            continue;
        }
        
        // Determine which side of the connection the parent is on
        // Note: Due to race conditions, the cell indices might have been modified
        // by another dividing cell. We use the original parent cell_idx to check.
        let is_parent_cell_a = conn.cell_a_index == cell_idx;
        let is_parent_cell_b = conn.cell_b_index == cell_idx;
        
        // If neither matches, another cell already processed this adhesion
        // (race condition - the other cell changed the indices)
        if (!is_parent_cell_a && !is_parent_cell_b) {
            continue;
        }
        
        // Get neighbor index
        let neighbor_idx = select(conn.cell_a_index, conn.cell_b_index, is_parent_cell_a);
        
        // Get the LOCAL anchor direction for the parent's side of the connection
        // This is already in parent's local frame (matches reference)
        var parent_anchor_dir_local: vec3<f32>;
        if (is_parent_cell_a) {
            parent_anchor_dir_local = conn.anchor_direction_a.xyz;
        } else {
            parent_anchor_dir_local = conn.anchor_direction_b.xyz;
        }
        
        // Classify the zone based on LOCAL anchor direction relative to LOCAL split direction
        // This matches reference: zones are classified in parent's local frame
        let zone = classify_adhesion_zone(parent_anchor_dir_local, split_dir_local);
        
        // Determine which child(ren) should inherit this adhesion
        var give_to_child_a = false;
        var give_to_child_b = false;
        
        if (zone == ZONE_A && child_b_keep) {
            // Zone A: negative dot product -> goes to Child B
            give_to_child_b = true;
        } else if (zone == ZONE_B && child_a_keep) {
            // Zone B: positive dot product -> goes to Child A
            give_to_child_a = true;
        } else if (zone == ZONE_C) {
            // Zone C: equatorial -> duplicated for both children (if they keep adhesions)
            if (child_a_keep) {
                give_to_child_a = true;
            }
            if (child_b_keep) {
                give_to_child_b = true;
            }
        }
        
        // Skip if neither child inherits
        if (!give_to_child_a && !give_to_child_b) {
            // Mark original connection as inactive (adhesion is lost)
            adhesion_connections[adh_idx].is_active = 0u;
            continue;
        }
        
        // Calculate center-to-center distance using rest length (matches reference)
        // Use fixed radius to make adhesion independent of cell growth
        let rest_length = 1.0; // Default rest length - could be read from mode settings
        let center_to_center_dist = rest_length + FIXED_RADIUS + FIXED_RADIUS;
        
        // Calculate neighbor position in parent frame (from anchor direction * distance)
        let neighbor_pos_parent_frame = parent_anchor_dir_local * center_to_center_dist;
        
        // Process inheritance based on zone
        if (give_to_child_a && !give_to_child_b) {
            // Only Child A inherits
            
            // Get neighbor's genome orientation for anchor calculation
            let neighbor_rotation = rotations_in[neighbor_idx];
            
            // Calculate relative rotation: inv(neighbor_orientation) * parent_orientation
            let relative_rotation = quat_multiply(quat_inverse(neighbor_rotation), parent_rotation);
            
            // Calculate new anchor direction for child using geometric approach (matches reference)
            let child_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_a_orientation
            );
            
            // Calculate neighbor's anchor direction pointing to Child A
            // Direction from neighbor to Child A in parent frame
            let dir_to_child_parent_frame = normalize(child_a_pos_parent_frame - neighbor_pos_parent_frame);
            // Transform to neighbor's local frame
            let neighbor_anchor = normalize(rotate_vector_by_quat(dir_to_child_parent_frame, relative_rotation));
            
            // Update the connection - preserve original side assignment
            // Child A overwrites parent slot, so cell index stays the same
            if (is_parent_cell_a) {
                // Parent was cellA, so child stays as cellA
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_a_rotation;
                // Zone classification for child's side
                adhesion_connections[adh_idx].zone_a = classify_adhesion_zone(child_anchor, split_dir_local);
            } else {
                // Parent was cellB, so child stays as cellB
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_a_rotation;
                // Zone classification for child's side
                adhesion_connections[adh_idx].zone_b = classify_adhesion_zone(child_anchor, split_dir_local);
            }
            
            // Add to Child A's adhesion list
            if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count] = i32(adh_idx);
                child_a_adhesion_count++;
            }
            
            // Note: The connection is now owned by Child A, cell index unchanged since
            // Child A overwrites the parent slot
            
        } else if (give_to_child_b && !give_to_child_a) {
            // Only Child B inherits
            
            // Get neighbor's genome orientation for anchor calculation
            let neighbor_rotation = rotations_in[neighbor_idx];
            
            // Calculate relative rotation: inv(neighbor_orientation) * parent_orientation
            let relative_rotation = quat_multiply(quat_inverse(neighbor_rotation), parent_rotation);
            
            // Calculate new anchor direction for child using geometric approach (matches reference)
            let child_anchor = calculate_child_anchor_direction(
                child_b_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_b_orientation
            );
            
            // Calculate neighbor's anchor direction pointing to Child B
            // Direction from neighbor to Child B in parent frame
            let dir_to_child_parent_frame = normalize(child_b_pos_parent_frame - neighbor_pos_parent_frame);
            // Transform to neighbor's local frame
            let neighbor_anchor = normalize(rotate_vector_by_quat(dir_to_child_parent_frame, relative_rotation));
            
            // Update the connection - preserve original side assignment and update cell index
            if (is_parent_cell_a) {
                // Parent was cellA, so child B takes cellA position
                adhesion_connections[adh_idx].cell_a_index = child_b_slot;
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_b_rotation;
                adhesion_connections[adh_idx].zone_a = classify_adhesion_zone(child_anchor, split_dir_local);
            } else {
                // Parent was cellB, so child B takes cellB position
                adhesion_connections[adh_idx].cell_b_index = child_b_slot;
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_b_rotation;
                adhesion_connections[adh_idx].zone_b = classify_adhesion_zone(child_anchor, split_dir_local);
            }
            
            // Add to Child B's adhesion list
            if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count] = i32(adh_idx);
                child_b_adhesion_count++;
            }
            
        } else if (give_to_child_a && give_to_child_b) {
            // Zone C: Both children inherit - need to duplicate the adhesion
            // CRITICAL: Neighbor needs TWO different anchor directions - one for each child
            // This is the "intermediate anchor step" from the reference
            
            // Get neighbor's genome orientation for anchor calculation
            let neighbor_rotation = rotations_in[neighbor_idx];
            
            // Calculate relative rotation: inv(neighbor_orientation) * parent_orientation
            // This transforms directions from parent frame to neighbor's local frame
            let relative_rotation = quat_multiply(quat_inverse(neighbor_rotation), parent_rotation);
            
            // === Child A gets the original connection ===
            let child_a_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_a_orientation
            );
            
            // Calculate neighbor's anchor direction pointing to Child A
            // Direction from neighbor to Child A in parent frame
            let dir_to_child_a_parent_frame = normalize(child_a_pos_parent_frame - neighbor_pos_parent_frame);
            // Transform to neighbor's local frame
            let neighbor_anchor_to_a = normalize(rotate_vector_by_quat(dir_to_child_a_parent_frame, relative_rotation));
            
            // Update original connection for Child A with BOTH anchor directions updated
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_a_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor_to_a, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_a_rotation;
                adhesion_connections[adh_idx].zone_a = classify_adhesion_zone(child_a_anchor, split_dir_local);
            } else {
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_a_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor_to_a, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_a_rotation;
                adhesion_connections[adh_idx].zone_b = classify_adhesion_zone(child_a_anchor, split_dir_local);
            }
            
            // Add to Child A's adhesion list
            if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count] = i32(adh_idx);
                child_a_adhesion_count++;
            }
            
            // === Create duplicate connection for Child B ===
            let dup_slot = atomicAdd(&adhesion_counts[0], 1u);
            if (dup_slot < 100000u) { // MAX_ADHESION_CONNECTIONS
                let child_b_anchor = calculate_child_anchor_direction(
                    child_b_pos_parent_frame,
                    neighbor_pos_parent_frame,
                    child_b_orientation
                );
                
                // Calculate neighbor's anchor direction pointing to Child B
                // Direction from neighbor to Child B in parent frame
                let dir_to_child_b_parent_frame = normalize(child_b_pos_parent_frame - neighbor_pos_parent_frame);
                // Transform to neighbor's local frame
                let neighbor_anchor_to_b = normalize(rotate_vector_by_quat(dir_to_child_b_parent_frame, relative_rotation));
                
                // Create duplicate connection with BOTH anchor directions calculated
                var dup_conn: AdhesionConnection;
                dup_conn.mode_index = conn.mode_index;
                dup_conn.is_active = 1u;
                dup_conn._align_pad = vec2<u32>(0u, 0u);
                dup_conn._padding = vec2<u32>(0u, 0u);
                
                if (is_parent_cell_a) {
                    // Parent was cellA, so child B takes cellA position in duplicate
                    dup_conn.cell_a_index = child_b_slot;
                    dup_conn.cell_b_index = neighbor_idx;
                    dup_conn.anchor_direction_a = vec4<f32>(child_b_anchor, 0.0);
                    dup_conn.anchor_direction_b = vec4<f32>(neighbor_anchor_to_b, 0.0); // Calculated anchor to Child B
                    dup_conn.twist_reference_a = child_b_rotation;
                    dup_conn.twist_reference_b = neighbor_rotation;
                    dup_conn.zone_a = classify_adhesion_zone(child_b_anchor, split_dir_local);
                    dup_conn.zone_b = classify_adhesion_zone(neighbor_anchor_to_b, split_dir_local);
                } else {
                    // Parent was cellB, so child B takes cellB position in duplicate
                    dup_conn.cell_a_index = neighbor_idx;
                    dup_conn.cell_b_index = child_b_slot;
                    dup_conn.anchor_direction_a = vec4<f32>(neighbor_anchor_to_b, 0.0); // Calculated anchor to Child B
                    dup_conn.anchor_direction_b = vec4<f32>(child_b_anchor, 0.0);
                    dup_conn.twist_reference_a = neighbor_rotation;
                    dup_conn.twist_reference_b = child_b_rotation;
                    dup_conn.zone_a = classify_adhesion_zone(neighbor_anchor_to_b, split_dir_local);
                    dup_conn.zone_b = classify_adhesion_zone(child_b_anchor, split_dir_local);
                }
                
                // Store the duplicate
                adhesion_connections[dup_slot] = dup_conn;
                
                // Add to Child B's adhesion list
                if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                    cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count] = i32(dup_slot);
                    child_b_adhesion_count++;
                }
                
                // Add duplicate to neighbor's adhesion list
                let neighbor_adhesion_base = neighbor_idx * MAX_ADHESIONS_PER_CELL;
                for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
                    if (cell_adhesion_indices[neighbor_adhesion_base + i] == -1) {
                        cell_adhesion_indices[neighbor_adhesion_base + i] = i32(dup_slot);
                        break;
                    }
                }
                
                // Update live adhesion count
                atomicAdd(&adhesion_counts[1], 1u);
            } else {
                // Failed to allocate duplicate - decrement
                atomicSub(&adhesion_counts[0], 1u);
            }
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
