// Lifecycle Division Execute Shader - Ring Buffer Version
// Stage 2: Execute cell division using pre-allocated slots from ring buffer
// Workgroup size: 128 threads for cell operations
//
// Input:
// - division_flags[cell_idx] = 1 for cells that should divide
// - division_slot_assignments[cell_idx] = slot index for child B
//
// Output:
// - Child A overwrites parent slot with half mass
// - Child B created in assigned slot with half mass
// - Adhesion connections updated for both children

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

// Cell type behavior flags
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    _padding: array<u32, 10>,
}

// Adhesion connection structure (104 bytes)
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
    _padding: vec2<u32>,
};

const PI: f32 = 3.14159265359;

// Physics bind group (group 0)
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

@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<atomic<u32>>;

// Lifecycle bind group (group 1) - all read_write to match Rust layout
@group(1) @binding(0)
var<storage, read_write> death_flags: array<u32>;

@group(1) @binding(1)
var<storage, read_write> division_flags: array<u32>;

@group(1) @binding(2)
var<storage, read_write> free_slot_ring: array<u32>;

@group(1) @binding(3)
var<storage, read_write> division_slot_assignments: array<u32>;

@group(1) @binding(4)
var<storage, read_write> ring_state: array<atomic<u32>>;

// Cell state bind group (group 2)
@group(2) @binding(0)
var<storage, read_write> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read_write> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read_write> split_masses: array<f32>;

@group(2) @binding(3)
var<storage, read_write> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read_write> split_ready_frame: array<i32>;

@group(2) @binding(5)
var<storage, read_write> max_splits: array<u32>;

@group(2) @binding(6)
var<storage, read_write> genome_ids: array<u32>;

@group(2) @binding(7)
var<storage, read_write> mode_indices: array<u32>;

@group(2) @binding(8)
var<storage, read_write> cell_ids: array<u32>;

@group(2) @binding(9)
var<storage, read_write> next_cell_id: array<atomic<u32>>;

@group(2) @binding(10)
var<storage, read_write> nutrient_gain_rates: array<f32>;

@group(2) @binding(11)
var<storage, read_write> max_cell_sizes: array<f32>;

@group(2) @binding(12)
var<storage, read_write> stiffnesses: array<f32>;

@group(2) @binding(13)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(2) @binding(14)
var<storage, read_write> rotations_out: array<vec4<f32>>;

@group(2) @binding(15)
var<storage, read> genome_mode_data: array<vec4<f32>>;

@group(2) @binding(16)
var<storage, read> parent_make_adhesion_flags: array<u32>;

@group(2) @binding(17)
var<storage, read> child_mode_indices: array<vec2<i32>>;

@group(2) @binding(18)
var<storage, read> mode_properties: array<vec4<f32>>;

@group(2) @binding(19)
var<storage, read_write> cell_types: array<u32>;

@group(2) @binding(20)
var<storage, read> mode_cell_types: array<u32>;

// Child A keep adhesion flags: one bool per mode (stored as u32)
@group(2) @binding(21)
var<storage, read> child_a_keep_adhesion_flags: array<u32>;

// Child B keep adhesion flags: one bool per mode (stored as u32)
@group(2) @binding(22)
var<storage, read> child_b_keep_adhesion_flags: array<u32>;

// Adhesion bind group (group 3)
@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read_write> cell_adhesion_indices: array<i32>;

@group(3) @binding(2)
var<storage, read_write> next_adhesion_id: array<atomic<u32>>;

// Free adhesion slot stack (for reuse of freed adhesion slots)
@group(3) @binding(3)
var<storage, read_write> free_adhesion_slots: array<u32>;

// Adhesion counts: [0] = total, [1] = live, [2] = free_top, [3] = padding
@group(3) @binding(4)
var<storage, read_write> adhesion_counts: array<atomic<u32>>;

// Allocate an adhesion slot, preferring to reuse freed slots.
// Uses compare-and-swap on adhesion_counts[2] (free_top) to safely pop from the stack.
// Falls back to monotonic next_adhesion_id if the free stack is empty.
// Returns 0xFFFFFFFF if at capacity.
fn allocate_adhesion_slot() -> u32 {
    // Try to pop from free adhesion slot stack first
    loop {
        let free_top = atomicLoad(&adhesion_counts[2]);
        if (free_top == 0u) {
            break; // Stack empty, fall back to monotonic
        }
        let result = atomicCompareExchangeWeak(&adhesion_counts[2], free_top, free_top - 1u);
        if (result.exchanged) {
            let slot = free_adhesion_slots[free_top - 1u];
            // Increment live count
            atomicAdd(&adhesion_counts[1], 1u);
            return slot;
        }
        // CAS failed (another thread popped), retry
    }
    
    // Free stack empty - fall back to monotonic allocation
    let slot = atomicAdd(&next_adhesion_id[0], 1u);
    if (slot < arrayLength(&adhesion_connections)) {
        // Increment live count
        atomicAdd(&adhesion_counts[1], 1u);
        return slot;
    }
    // At capacity
    return 0xFFFFFFFFu;
}

// Helper functions
fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qvec = q.xyz;
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

// Quaternion conjugate (inverse for unit quaternions)
fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// Rotate vector by inverse quaternion (for local space transformation)
fn rotate_vector_by_quat_inverse(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return rotate_vector_by_quat(v, quat_conjugate(q));
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Zone classification constants (matching CPU: sin(2°) ≈ 0.0349)
const EQUATORIAL_THRESHOLD: f32 = 0.0349;
const ZONE_A: u32 = 0u;  // Negative dot (opposite to split)
const ZONE_B: u32 = 1u;  // Positive dot (same as split)
const ZONE_C: u32 = 2u;  // Equatorial (perpendicular to split)

// Classify adhesion bond direction relative to split direction
// Matches CPU classify_bond_direction() exactly
fn classify_zone(anchor_direction: vec3<f32>, split_direction: vec3<f32>) -> u32 {
    let dot_product = dot(normalize(anchor_direction), normalize(split_direction));
    
    if (abs(dot_product) <= EQUATORIAL_THRESHOLD) {
        return ZONE_C;  // Equatorial
    } else if (dot_product > 0.0) {
        return ZONE_B;  // Same direction as split
    } else {
        return ZONE_A;  // Opposite to split
    }
}

fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

const MAX_ADHESIONS_PER_CELL: u32 = 20u;

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
    let inv_delta = quat_conjugate(child_orientation_delta);
    return normalize(rotate_vector_by_quat(direction_to_neighbor, inv_delta));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = atomicLoad(&cell_count_buffer[0]);
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Check if this cell is dividing
    if (division_flags[cell_idx] != 1u) {
        return;
    }
    
    // Get the pre-assigned slot for child B
    let child_b_slot = division_slot_assignments[cell_idx];
    
    // Validate slot assignment
    if (child_b_slot >= params.cell_capacity) {
        return; // Invalid slot, skip division
    }
    
    // Parent state - read from OUTPUT buffer (physics results)
    let parent_pos = positions_out[cell_idx].xyz;
    let parent_mass = positions_out[cell_idx].w;
    let parent_vel = velocities_out[cell_idx].xyz;
    let parent_rotation = rotations_in[cell_idx];
    let parent_split_count = split_counts[cell_idx];
    
    // Get parent's mode index for looking up child orientations and split direction
    let parent_mode_idx = mode_indices[cell_idx];
    
    // Read child orientations and split direction from genome mode data
    let child_a_orientation = genome_mode_data[parent_mode_idx * 3u];
    let child_b_orientation = genome_mode_data[parent_mode_idx * 3u + 1u];
    let split_direction_local = genome_mode_data[parent_mode_idx * 3u + 2u].xyz;
    
    // Calculate child rotations
    let child_a_rotation = quat_multiply(parent_rotation, child_a_orientation);
    let child_b_rotation = quat_multiply(parent_rotation, child_b_orientation);
    
    // Calculate parent radius for offset calculation
    let parent_radius = calculate_radius_from_mass(parent_mass);
    
    // Split mass 50/50
    let child_mass = parent_mass * 0.5;
    
    // Transform split direction from local to world space
    var split_dir_local = split_direction_local;
    if (length(split_dir_local) < 0.0001) {
        split_dir_local = vec3<f32>(0.0, 0.0, 1.0);
    }
    let split_dir = normalize(rotate_vector_by_quat(split_dir_local, parent_rotation));
    
    // 75% overlap: offset_distance = parent_radius * 0.25
    let offset = parent_radius * 0.25;
    let child_a_pos = parent_pos + split_dir * offset;
    let child_b_pos = parent_pos - split_dir * offset;
    
    // Get child mode indices
    let child_modes = child_mode_indices[parent_mode_idx];
    let child_a_mode_idx = u32(max(child_modes.x, 0));
    let child_b_mode_idx = u32(max(child_modes.y, 0));
    
    // === Create Child A (overwrites parent slot) ===
    positions_out[cell_idx] = vec4<f32>(child_a_pos, child_mass);
    velocities_out[cell_idx] = vec4<f32>(parent_vel, 0.0);
    rotations_out[cell_idx] = child_a_rotation;
    
    birth_times[cell_idx] = params.current_time;
    let child_a_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[cell_idx] = child_a_id;
    mode_indices[cell_idx] = child_a_mode_idx;
    
    // Split count: reset if mode changes, otherwise increment
    if (child_a_mode_idx != parent_mode_idx) {
        split_counts[cell_idx] = 0u;
    } else {
        split_counts[cell_idx] = parent_split_count + 1u;
    }
    
    // Read Child A's properties from its mode
    let child_a_props_0 = mode_properties[child_a_mode_idx * 3u];
    let child_a_props_1 = mode_properties[child_a_mode_idx * 3u + 1u];
    let child_a_props_2 = mode_properties[child_a_mode_idx * 3u + 2u];
    
    // Only Test cells (cell_type == 0) auto-generate nutrients
    let child_a_cell_type = mode_cell_types[child_a_mode_idx];
    let child_a_nutrient_rate = select(0.0, child_a_props_0.x, child_a_cell_type == 0u);
    nutrient_gain_rates[cell_idx] = child_a_nutrient_rate;
    
    cell_types[cell_idx] = child_a_cell_type;
    max_cell_sizes[cell_idx] = child_a_props_0.y;
    stiffnesses[cell_idx] = child_a_props_0.z;
    split_intervals[cell_idx] = child_a_props_0.w;
    split_masses[cell_idx] = child_a_props_1.x;
    max_splits[cell_idx] = u32(child_a_props_2.x);
    
    // === Create Child B (in assigned slot) ===
    positions_out[child_b_slot] = vec4<f32>(child_b_pos, child_mass);
    velocities_out[child_b_slot] = vec4<f32>(parent_vel, 0.0);
    rotations_out[child_b_slot] = child_b_rotation;
    
    let child_b_props_0 = mode_properties[child_b_mode_idx * 3u];
    let child_b_props_1 = mode_properties[child_b_mode_idx * 3u + 1u];
    let child_b_props_2 = mode_properties[child_b_mode_idx * 3u + 2u];
    
    birth_times[child_b_slot] = params.current_time;
    split_intervals[child_b_slot] = child_b_props_0.w;
    split_masses[child_b_slot] = child_b_props_1.x;
    
    if (child_b_mode_idx != parent_mode_idx) {
        split_counts[child_b_slot] = 0u;
    } else {
        split_counts[child_b_slot] = parent_split_count + 1u;
    }
    
    max_splits[child_b_slot] = u32(child_b_props_2.x);
    genome_ids[child_b_slot] = genome_ids[cell_idx];
    mode_indices[child_b_slot] = child_b_mode_idx;
    
    let child_b_cell_type = mode_cell_types[child_b_mode_idx];
    let child_b_nutrient_rate = select(0.0, child_b_props_0.x, child_b_cell_type == 0u);
    nutrient_gain_rates[child_b_slot] = child_b_nutrient_rate;
    
    cell_types[child_b_slot] = child_b_cell_type;
    max_cell_sizes[child_b_slot] = child_b_props_0.y;
    stiffnesses[child_b_slot] = child_b_props_0.z;
    
    let child_b_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[child_b_slot] = child_b_id;
    
    // Clear death flag for the new slot
    death_flags[child_b_slot] = 0u;
    
    // Increment live cell count (one parent became two children)
    atomicAdd(&cell_count_buffer[1], 1u);
    
    // Update total cell count if we used a new slot beyond current count
    atomicMax(&cell_count_buffer[0], child_b_slot + 1u);
    
    // Track sibling adhesion slot for inheritance (0xFFFFFFFF = invalid/not created)
    var sibling_adhesion_slot: u32 = 0xFFFFFFFFu;
    
    // === Create sibling adhesion if parent_make_adhesion is enabled ===
    let make_adhesion = parent_make_adhesion_flags[parent_mode_idx];
    if (make_adhesion == 1u) {
        let adhesion_id = allocate_adhesion_slot();
        if (adhesion_id != 0xFFFFFFFFu) {
            // Calculate anchor directions in each child's LOCAL space
            // Normalize split direction first (matching CPU implementation)
            let split_dir_normalized = normalize(split_dir_local);
            
            // Direction from A to B in parent's local frame (A points toward B)
            let dir_a_to_b_parent = -split_dir_normalized;  // Child A anchor points toward Child B
            let dir_b_to_a_parent = split_dir_normalized;   // Child B anchor points toward Child A
            
            // Transform to each child's local space using inverse of child orientation
            // Check if child orientations are identity (or near-identity) to avoid numerical issues
            var anchor_a_local: vec3<f32>;
            var anchor_b_local: vec3<f32>;
            
            // Child orientation is identity if w ≈ 1 and xyz ≈ 0
            let child_a_is_identity = abs(child_a_orientation.w - 1.0) < 0.01 && length(child_a_orientation.xyz) < 0.01;
            let child_b_is_identity = abs(child_b_orientation.w - 1.0) < 0.01 && length(child_b_orientation.xyz) < 0.01;
            
            if (child_a_is_identity) {
                anchor_a_local = dir_a_to_b_parent;
            } else {
                anchor_a_local = normalize(rotate_vector_by_quat_inverse(dir_a_to_b_parent, child_a_orientation));
            }
            
            if (child_b_is_identity) {
                anchor_b_local = dir_b_to_a_parent;
            } else {
                anchor_b_local = normalize(rotate_vector_by_quat_inverse(dir_b_to_a_parent, child_b_orientation));
            }
            
            // Classify zones based on anchor direction vs split direction (matching CPU)
            // Each cell's zone is classified using its own anchor and split direction
            let zone_a = classify_zone(anchor_a_local, split_dir_normalized);
            let zone_b = classify_zone(anchor_b_local, split_dir_normalized);
            
            var connection: AdhesionConnection;
            connection.cell_a_index = cell_idx;
            connection.cell_b_index = child_b_slot;
            connection.mode_index = parent_mode_idx;
            connection.is_active = 1u;
            connection.zone_a = zone_a;
            connection.zone_b = zone_b;
            connection._align_pad = vec2<u32>(0u, 0u);
            connection.anchor_direction_a = vec4<f32>(anchor_a_local, 0.0);
            connection.anchor_direction_b = vec4<f32>(anchor_b_local, 0.0);
            // Set twist references to child rotations (parent_rotation * child_orientation)
            connection.twist_reference_a = child_a_rotation;
            connection.twist_reference_b = child_b_rotation;
            connection._padding = vec2<u32>(0u, 0u);
            adhesion_connections[adhesion_id] = connection;
            
            // Add to cell adhesion indices
            let base_a = cell_idx * MAX_ADHESIONS_PER_CELL;
            let base_b = child_b_slot * MAX_ADHESIONS_PER_CELL;
            
            for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
                if (cell_adhesion_indices[base_a + i] < 0) {
                    cell_adhesion_indices[base_a + i] = i32(adhesion_id);
                    break;
                }
            }
            for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
                if (cell_adhesion_indices[base_b + i] < 0) {
                    cell_adhesion_indices[base_b + i] = i32(adhesion_id);
                    break;
                }
            }
            
            // Track sibling adhesion slot to skip during inheritance
            sibling_adhesion_slot = adhesion_id;
        }
    }
    
    // === Zone-Based Adhesion Inheritance ===
    // Process parent's existing adhesions and distribute to children based on zone classification
    // - Zone A (negative dot with split dir) -> Child B
    // - Zone B (positive dot with split dir) -> Child A  
    // - Zone C (equatorial) -> Both children (duplicate)
    
    let child_a_keep = child_a_keep_adhesion_flags[parent_mode_idx] == 1u;
    let child_b_keep = child_b_keep_adhesion_flags[parent_mode_idx] == 1u;
    
    // If neither child keeps adhesions, clear parent's old inherited connections
    // but preserve the sibling adhesion that was just created
    if (!child_a_keep && !child_b_keep) {
        let clear_base = cell_idx * MAX_ADHESIONS_PER_CELL;
        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            let adh_idx_signed = cell_adhesion_indices[clear_base + i];
            if (adh_idx_signed >= 0 && u32(adh_idx_signed) != sibling_adhesion_slot) {
                let adh_idx = u32(adh_idx_signed);
                // Find the neighbor and remove their reference to this adhesion
                let conn = adhesion_connections[adh_idx];
                if (conn.is_active != 0u) {
                    var neighbor_idx: u32 = 0xFFFFFFFFu;
                    if (conn.cell_a_index == cell_idx) {
                        neighbor_idx = conn.cell_b_index;
                    } else if (conn.cell_b_index == cell_idx) {
                        neighbor_idx = conn.cell_a_index;
                    }
                    if (neighbor_idx != 0xFFFFFFFFu) {
                        let neighbor_base = neighbor_idx * MAX_ADHESIONS_PER_CELL;
                        for (var j = 0u; j < MAX_ADHESIONS_PER_CELL; j++) {
                            if (cell_adhesion_indices[neighbor_base + j] == adh_idx_signed) {
                                cell_adhesion_indices[neighbor_base + j] = -1;
                                break;
                            }
                        }
                    }
                }
                // Deactivate old inherited connection (not the sibling)
                adhesion_connections[adh_idx].is_active = 0u;
                cell_adhesion_indices[clear_base + i] = -1;
            }
        }
        return;
    }
    
    let parent_adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let child_a_adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let child_b_adhesion_base = child_b_slot * MAX_ADHESIONS_PER_CELL;
    
    // Save parent's adhesion indices BEFORE modifying them
    var parent_adhesion_indices: array<i32, 20>;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        parent_adhesion_indices[i] = cell_adhesion_indices[parent_adhesion_base + i];
    }
    
    // Clear Child A's adhesion indices (parent slot) - we'll rebuild them
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        cell_adhesion_indices[child_a_adhesion_base + i] = -1;
        cell_adhesion_indices[child_b_adhesion_base + i] = -1;
    }
    
    // Track adhesion counts for each child
    var child_a_adhesion_count = 0u;
    var child_b_adhesion_count = 0u;
    
    // Re-add the sibling adhesion (if created)
    if (sibling_adhesion_slot != 0xFFFFFFFFu) {
        if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
            cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count] = i32(sibling_adhesion_slot);
            child_a_adhesion_count++;
        }
        if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
            cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count] = i32(sibling_adhesion_slot);
            child_b_adhesion_count++;
        }
    }
    
    // Calculate geometric parameters for anchor calculation
    let child_offset = parent_radius * 0.25;
    let split_dir_normalized = normalize(split_dir_local);
    let child_a_pos_parent_frame = split_dir_normalized * child_offset;
    let child_b_pos_parent_frame = -split_dir_normalized * child_offset;
    
    // Fixed radius for adhesion distance calculation
    let FIXED_RADIUS: f32 = 1.0;
    
    // Process inherited adhesions
    for (var parent_slot = 0u; parent_slot < MAX_ADHESIONS_PER_CELL; parent_slot++) {
        let adh_idx_signed = parent_adhesion_indices[parent_slot];
        
        if (adh_idx_signed < 0) {
            continue;
        }
        
        let adh_idx = u32(adh_idx_signed);
        
        // Skip sibling adhesion (already handled)
        if (adh_idx == sibling_adhesion_slot) {
            continue;
        }
        
        let conn = adhesion_connections[adh_idx];
        
        if (conn.is_active == 0u) {
            continue;
        }
        
        // Determine which side of connection parent is on
        let is_parent_cell_a = conn.cell_a_index == cell_idx;
        let is_parent_cell_b = conn.cell_b_index == cell_idx;
        
        if (!is_parent_cell_a && !is_parent_cell_b) {
            continue;
        }
        
        let neighbor_idx = select(conn.cell_a_index, conn.cell_b_index, is_parent_cell_a);
        
        // Get parent's anchor direction (LOCAL space)
        var parent_anchor_dir_local: vec3<f32>;
        if (is_parent_cell_a) {
            parent_anchor_dir_local = conn.anchor_direction_a.xyz;
        } else {
            parent_anchor_dir_local = conn.anchor_direction_b.xyz;
        }
        
        // Classify zone based on LOCAL anchor direction vs LOCAL split direction
        let zone = classify_zone(parent_anchor_dir_local, split_dir_local);
        
        // Determine which child(ren) inherit
        var give_to_child_a = false;
        var give_to_child_b = false;
        
        if (zone == ZONE_A && child_b_keep) {
            give_to_child_b = true;
        } else if (zone == ZONE_B && child_a_keep) {
            give_to_child_a = true;
        } else if (zone == ZONE_C) {
            if (child_a_keep) { give_to_child_a = true; }
            if (child_b_keep) { give_to_child_b = true; }
        }
        
        if (!give_to_child_a && !give_to_child_b) {
            // Remove neighbor's reference to this adhesion before deactivating
            let deact_conn = adhesion_connections[adh_idx];
            var deact_neighbor: u32 = 0xFFFFFFFFu;
            if (deact_conn.cell_a_index == cell_idx) {
                deact_neighbor = deact_conn.cell_b_index;
            } else if (deact_conn.cell_b_index == cell_idx) {
                deact_neighbor = deact_conn.cell_a_index;
            }
            if (deact_neighbor != 0xFFFFFFFFu) {
                let nb = deact_neighbor * MAX_ADHESIONS_PER_CELL;
                for (var j = 0u; j < MAX_ADHESIONS_PER_CELL; j++) {
                    if (cell_adhesion_indices[nb + j] == i32(adh_idx)) {
                        cell_adhesion_indices[nb + j] = -1;
                        break;
                    }
                }
            }
            adhesion_connections[adh_idx].is_active = 0u;
            continue;
        }
        
        // Calculate center-to-center distance (PBD: adhesin_length * 50.0)
        let rest_offset = 0.0; // Default adhesin_length=0.0 → offset=0.0
        let center_to_center_dist = FIXED_RADIUS + FIXED_RADIUS + rest_offset;
        let neighbor_pos_parent_frame = parent_anchor_dir_local * center_to_center_dist;
        
        // Get neighbor rotation for anchor calculation
        let neighbor_rotation = rotations_in[neighbor_idx];
        let relative_rotation = quat_multiply(quat_conjugate(neighbor_rotation), parent_rotation);
        
        if (give_to_child_a && !give_to_child_b) {
            // Only Child A inherits
            let child_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_a_orientation
            );
            
            let dir_to_child_parent_frame = normalize(child_a_pos_parent_frame - neighbor_pos_parent_frame);
            let neighbor_anchor = normalize(rotate_vector_by_quat(dir_to_child_parent_frame, relative_rotation));
            
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_a_rotation;
                adhesion_connections[adh_idx].zone_a = classify_zone(child_anchor, split_dir_local);
            } else {
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_a_rotation;
                adhesion_connections[adh_idx].zone_b = classify_zone(child_anchor, split_dir_local);
            }
            
            if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count] = i32(adh_idx);
                child_a_adhesion_count++;
            }
            
        } else if (give_to_child_b && !give_to_child_a) {
            // Only Child B inherits
            let child_anchor = calculate_child_anchor_direction(
                child_b_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_b_orientation
            );
            
            let dir_to_child_parent_frame = normalize(child_b_pos_parent_frame - neighbor_pos_parent_frame);
            let neighbor_anchor = normalize(rotate_vector_by_quat(dir_to_child_parent_frame, relative_rotation));
            
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].cell_a_index = child_b_slot;
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_b_rotation;
                adhesion_connections[adh_idx].zone_a = classify_zone(child_anchor, split_dir_local);
            } else {
                adhesion_connections[adh_idx].cell_b_index = child_b_slot;
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_b_rotation;
                adhesion_connections[adh_idx].zone_b = classify_zone(child_anchor, split_dir_local);
            }
            
            if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count] = i32(adh_idx);
                child_b_adhesion_count++;
            }
            
        } else {
            // Zone C: Both children inherit - duplicate the adhesion
            let child_a_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_a_orientation
            );
            
            let dir_to_child_a_parent_frame = normalize(child_a_pos_parent_frame - neighbor_pos_parent_frame);
            let neighbor_anchor_to_a = normalize(rotate_vector_by_quat(dir_to_child_a_parent_frame, relative_rotation));
            
            // Update original connection for Child A
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_a_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor_to_a, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_a_rotation;
                adhesion_connections[adh_idx].zone_a = classify_zone(child_a_anchor, split_dir_local);
            } else {
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_a_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor_to_a, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_a_rotation;
                adhesion_connections[adh_idx].zone_b = classify_zone(child_a_anchor, split_dir_local);
            }
            
            if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count] = i32(adh_idx);
                child_a_adhesion_count++;
            }
            
            // Create duplicate for Child B
            let dup_slot = allocate_adhesion_slot();
            if (dup_slot != 0xFFFFFFFFu) {
                let child_b_anchor = calculate_child_anchor_direction(
                    child_b_pos_parent_frame,
                    neighbor_pos_parent_frame,
                    child_b_orientation
                );
                
                let dir_to_child_b_parent_frame = normalize(child_b_pos_parent_frame - neighbor_pos_parent_frame);
                let neighbor_anchor_to_b = normalize(rotate_vector_by_quat(dir_to_child_b_parent_frame, relative_rotation));
                
                var dup_conn: AdhesionConnection;
                dup_conn.mode_index = conn.mode_index;
                dup_conn.is_active = 1u;
                dup_conn._align_pad = vec2<u32>(0u, 0u);
                dup_conn._padding = vec2<u32>(0u, 0u);
                
                if (is_parent_cell_a) {
                    dup_conn.cell_a_index = child_b_slot;
                    dup_conn.cell_b_index = neighbor_idx;
                    dup_conn.anchor_direction_a = vec4<f32>(child_b_anchor, 0.0);
                    dup_conn.anchor_direction_b = vec4<f32>(neighbor_anchor_to_b, 0.0);
                    dup_conn.twist_reference_a = child_b_rotation;
                    dup_conn.twist_reference_b = neighbor_rotation;
                    dup_conn.zone_a = classify_zone(child_b_anchor, split_dir_local);
                    dup_conn.zone_b = classify_zone(neighbor_anchor_to_b, split_dir_local);
                } else {
                    dup_conn.cell_a_index = neighbor_idx;
                    dup_conn.cell_b_index = child_b_slot;
                    dup_conn.anchor_direction_a = vec4<f32>(neighbor_anchor_to_b, 0.0);
                    dup_conn.anchor_direction_b = vec4<f32>(child_b_anchor, 0.0);
                    dup_conn.twist_reference_a = neighbor_rotation;
                    dup_conn.twist_reference_b = child_b_rotation;
                    dup_conn.zone_a = classify_zone(neighbor_anchor_to_b, split_dir_local);
                    dup_conn.zone_b = classify_zone(child_b_anchor, split_dir_local);
                }
                
                adhesion_connections[dup_slot] = dup_conn;
                
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
            }
        }
    }
}
