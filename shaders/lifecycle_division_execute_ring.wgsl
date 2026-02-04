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

// Adhesion bind group (group 3)
@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read_write> cell_adhesion_indices: array<i32>;

@group(3) @binding(2)
var<storage, read_write> next_adhesion_id: array<atomic<u32>>;

// Helper functions
fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qvec = q.xyz;
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

const MAX_ADHESIONS_PER_CELL: u32 = 10u;

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
    
    // === Create sibling adhesion if parent_make_adhesion is enabled ===
    let make_adhesion = parent_make_adhesion_flags[parent_mode_idx];
    if (make_adhesion == 1u) {
        let adhesion_id = atomicAdd(&next_adhesion_id[0], 1u);
        if (adhesion_id < arrayLength(&adhesion_connections)) {
            var connection: AdhesionConnection;
            connection.cell_a_index = cell_idx;
            connection.cell_b_index = child_b_slot;
            connection.mode_index = parent_mode_idx;
            connection.is_active = 1u;
            connection.zone_a = 0u;
            connection.zone_b = 0u;
            connection._align_pad = vec2<u32>(0u, 0u);
            connection.anchor_direction_a = vec4<f32>(split_dir, 0.0);
            connection.anchor_direction_b = vec4<f32>(-split_dir, 0.0);
            connection.twist_reference_a = vec4<f32>(0.0, 1.0, 0.0, 0.0);
            connection.twist_reference_b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
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
        }
    }
}
