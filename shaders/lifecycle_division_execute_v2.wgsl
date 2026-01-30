//! Updated lifecycle division shader for per-genome buffer architecture
//! 
//! This shader uses the new per-genome buffer system to eliminate
//! indexing conflicts between genomes.

// Cell data buffers
@group(0) @binding(0)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> rotations_out: array<vec4<f32>>;

// Cell state buffers
@group(1) @binding(0)
var<storage, read_write> birth_times: array<f32>;

@group(1) @binding(1)
var<storage, read_write> split_intervals: array<f32>;

@group(1) @binding(2)
var<storage, read_write> split_masses: array<f32>;

@group(1) @binding(3)
var<storage, read_write> max_splits: array<u32>;

@group(1) @binding(4)
var<storage, read_write> genome_ids: array<u32>;

@group(1) @binding(5)
var<storage, read_write> mode_indices: array<u32>;

@group(1) @binding(6)
var<storage, read_write> cell_ids: array<u32>;

@group(1) @binding(7)
var<storage, read_write> nutrient_gain_rates: array<f32>;

@group(1) @binding(8)
var<storage, read_write> max_cell_sizes: array<f32>;

@group(1) @binding(9)
var<storage, read_write> stiffnesses: array<f32>;

// Per-genome buffer system
@group(2) @binding(0)
var<storage, read> genome_mode_properties: array<vec4<f32>>; // Flattened across all genomes

@group(2) @binding(1)
var<storage, read> genome_child_mode_indices: array<vec2<i32>>; // Flattened across all genomes

@group(2) @binding(2)
var<storage, read> genome_parent_make_adhesion_flags: array<u32>; // Flattened across all genomes

@group(2) @binding(3)
var<storage, read> genome_mode_data: array<vec4<f32>>; // Child orientations and split directions

@group(2) @binding(4)
var<storage, read> genome_mode_offsets: array<u32>; // Offset into per-genome arrays for each genome

@group(2) @binding(5)
var<storage, read> genome_mode_counts: array<u32>; // Number of modes per genome

// Division control buffers
@group(3) @binding(0)
var<storage, read> division_flags: array<u32>;

@group(3) @binding(1)
var<storage, read> free_slot_indices: array<u32>;

@group(3) @binding(2)
var<storage, read> division_slot_assignments: array<u32>;

@group(3) @binding(3)
var<storage, read> lifecycle_counts: array<u32>; // [0] = free slots, [1] = division count

@group(3) @binding(4)
var<storage, read_write> next_cell_id: array<u32>;

// Physics parameters
struct PhysicsParams {
    current_time: f32,
    dt: f32,
    padding1: f32,
    padding2: f32,
}

@group(3) @binding(5)
var<uniform> params: PhysicsParams;

// Helper functions
fn calculate_radius_from_mass(mass: f32) -> f32 {
    // Mass = 4/3 * pi * r^3 for unit density
    // r = (3 * mass / (4 * pi))^(1/3)
    let volume = mass / 1.0; // unit density
    let radius = pow(volume * 0.75 / 3.14159265359, 1.0 / 3.0);
    return max(radius, 0.1); // minimum radius
}

fn quat_multiply(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    // Quaternion rotation: v' = q * v * q^-1
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Get global mode index from genome_id and local mode index
fn get_global_mode_index(genome_id: u32, local_mode_idx: u32) -> u32 {
    let genome_offset = genome_mode_offsets[genome_id];
    return genome_offset + local_mode_idx;
}

// Get mode properties for a specific genome and mode
fn get_mode_properties(genome_id: u32, local_mode_idx: u32, property_index: u32) -> vec4<f32> {
    let global_idx = get_global_mode_index(genome_id, local_mode_idx);
    // Each mode has 3 property vectors (12 floats total)
    return genome_mode_properties[global_idx * 3u + property_index];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = lifecycle_counts[1]; // division count
    
    if cell_idx >= cell_count {
        return;
    }
    
    // Get the actual cell index from division assignment
    let actual_cell_idx = division_slot_assignments[cell_idx];
    
    // Get parent cell data
    let parent_pos = positions_in[actual_cell_idx].xyz;
    let parent_mass = positions_in[actual_cell_idx].w;
    let parent_vel = velocities_in[actual_cell_idx].xyz;
    let parent_rotation = rotations_in[actual_cell_idx];
    
    // Get parent genome and mode information
    let parent_genome_id = genome_ids[actual_cell_idx];
    let parent_local_mode_idx = mode_indices[actual_cell_idx];
    
    // Get child mode indices using per-genome data
    let child_modes = genome_child_mode_indices[get_global_mode_index(parent_genome_id, parent_local_mode_idx)];
    let child_a_local_mode_idx = u32(max(child_modes.x, 0));
    let child_b_local_mode_idx = u32(max(child_modes.y, 0));
    
    // Read child orientations and split direction from genome mode data
    // Layout: [child_a_orientation (vec4), child_b_orientation (vec4), split_direction (vec4)] per mode
    let global_parent_mode_idx = get_global_mode_index(parent_genome_id, parent_local_mode_idx);
    let child_a_orientation = genome_mode_data[global_parent_mode_idx * 3u];
    let child_b_orientation = genome_mode_data[global_parent_mode_idx * 3u + 1u];
    let split_direction_local = genome_mode_data[global_parent_mode_idx * 3u + 2u].xyz;
    
    // Calculate child rotations
    let child_a_rotation = quat_multiply(parent_rotation, child_a_orientation);
    let child_b_rotation = quat_multiply(parent_rotation, child_b_orientation);
    
    // Calculate parent radius
    let parent_radius = calculate_radius_from_mass(parent_mass);
    
    // Split mass 50/50
    let child_mass = parent_mass * 0.5;
    
    // Transform split direction from local to world space
    var split_dir_local = split_direction_local;
    if (length(split_dir_local) < 0.0001) {
        split_dir_local = vec3<f32>(0.0, 0.0, 1.0);
    }
    let split_dir = normalize(rotate_vector_by_quat(split_dir_local, parent_rotation));
    
    // Calculate child positions
    let offset = parent_radius * 0.25;
    let child_a_pos = parent_pos + split_dir * offset;
    let child_b_pos = parent_pos - split_dir * offset;
    
    // Get free slot for Child B
    let free_slot_idx = free_slot_indices[cell_idx];
    
    // === Create Child A (overwrites parent slot) ===
    positions_out[actual_cell_idx] = vec4<f32>(child_a_pos, child_mass);
    velocities_out[actual_cell_idx] = vec4<f32>(parent_vel, 0.0);
    rotations_out[actual_cell_idx] = child_a_rotation;
    
    // Update Child A state
    birth_times[actual_cell_idx] = params.current_time;
    let child_a_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[actual_cell_idx] = child_a_id;
    mode_indices[actual_cell_idx] = child_a_local_mode_idx;
    genome_ids[actual_cell_idx] = parent_genome_id; // Keep same genome
    
    // Get Child A mode properties
    let child_a_props_0 = get_mode_properties(parent_genome_id, child_a_local_mode_idx, 0u);
    let child_a_props_1 = get_mode_properties(parent_genome_id, child_a_local_mode_idx, 1u);
    let child_a_props_2 = get_mode_properties(parent_genome_id, child_a_local_mode_idx, 2u);
    
    nutrient_gain_rates[actual_cell_idx] = child_a_props_0.x;
    max_cell_sizes[actual_cell_idx] = child_a_props_0.y;
    stiffnesses[actual_cell_idx] = child_a_props_0.z;
    split_intervals[actual_cell_idx] = child_a_props_0.w;
    split_masses[actual_cell_idx] = child_a_props_1.x;
    max_splits[actual_cell_idx] = u32(child_a_props_2.x);
    
    // === Create Child B (in free slot) ===
    positions_out[free_slot_idx] = vec4<f32>(child_b_pos, child_mass);
    velocities_out[free_slot_idx] = vec4<f32>(parent_vel, 0.0);
    rotations_out[free_slot_idx] = child_b_rotation;
    
    // Update Child B state
    birth_times[free_slot_idx] = params.current_time;
    let child_b_id = atomicAdd(&next_cell_id[0], 1u);
    cell_ids[free_slot_idx] = child_b_id;
    mode_indices[free_slot_idx] = child_b_local_mode_idx;
    genome_ids[free_slot_idx] = parent_genome_id; // Keep same genome
    
    // Get Child B mode properties
    let child_b_props_0 = get_mode_properties(parent_genome_id, child_b_local_mode_idx, 0u);
    let child_b_props_1 = get_mode_properties(parent_genome_id, child_b_local_mode_idx, 1u);
    let child_b_props_2 = get_mode_properties(parent_genome_id, child_b_local_mode_idx, 2u);
    
    nutrient_gain_rates[free_slot_idx] = child_b_props_0.x;
    max_cell_sizes[free_slot_idx] = child_b_props_0.y;
    stiffnesses[free_slot_idx] = child_b_props_0.z;
    split_intervals[free_slot_idx] = child_b_props_0.w;
    split_masses[free_slot_idx] = child_b_props_1.x;
    max_splits[free_slot_idx] = u32(child_b_props_2.x);
}
