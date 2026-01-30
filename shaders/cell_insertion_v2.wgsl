//! Updated cell insertion shader for per-genome buffer architecture
//! 
//! This shader uses the new per-genome buffer system to eliminate
//! indexing conflicts between genomes.

// Cell insertion parameters
struct CellInsertionParams {
    world_position: vec3<f32>,
    initial_mass: f32,
    genome_id: u32,
    local_mode_idx: u32,
    birth_time: f32,
    cell_id: u32,
    max_splits: i32,
    padding: f32,
    padding2: f32,
}

@group(0) @binding(0)
var<uniform> insertion_params: CellInsertionParams;

// Cell data buffers (triple buffered)
@group(1) @binding(0)
var<storage, read_write> positions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read_write> velocities: array<vec4<f32>>;

@group(1) @binding(2)
var<storage, read_write> rotations: array<vec4<f32>>;

// Cell state buffers
@group(1) @binding(3)
var<storage, read_write> birth_times: array<f32>;

@group(1) @binding(4)
var<storage, read_write> split_intervals: array<f32>;

@group(1) @binding(5)
var<storage, read_write> split_masses: array<f32>;

@group(1) @binding(6)
var<storage, read_write> max_splits: array<u32>;

@group(1) @binding(7)
var<storage, read_write> genome_ids: array<u32>;

@group(1) @binding(8)
var<storage, read_write> mode_indices: array<u32>;

@group(1) @binding(9)
var<storage, read_write> cell_ids: array<u32>;

@group(1) @binding(10)
var<storage, read_write> nutrient_gain_rates: array<f32>;

@group(1) @binding(11)
var<storage, read_write> max_cell_sizes: array<f32>;

@group(1) @binding(12)
var<storage, read_write> stiffnesses: array<f32>;

// Per-genome buffer system
@group(2) @binding(0)
var<storage, read> genome_mode_properties: array<vec4<f32>>; // Flattened across all genomes

@group(2) @binding(1)
var<storage, read> genome_mode_offsets: array<u32>; // Offset into per-genome arrays for each genome

@group(2) @binding(2)
var<storage, read> genome_mode_counts: array<u32>; // Number of modes per genome

// Cell count buffer
@group(2) @binding(3)
var<storage, read_write> cell_count_buffer: array<u32>;

// Slot allocation buffer
@group(2) @binding(4)
var<storage, read_write> next_free_slot: array<u32>;

// Helper functions
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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Only first workgroup executes the insertion
    if global_id.x > 0u {
        return;
    }
    
    // Validate genome and mode bounds
    let genome_count = arrayLength(&genome_mode_offsets);
    if (insertion_params.genome_id >= genome_count) {
        return; // Invalid genome ID
    }
    
    let genome_mode_count = genome_mode_counts[insertion_params.genome_id];
    if (insertion_params.local_mode_idx >= genome_mode_count) {
        return; // Invalid mode index
    }
    
    // Allocate a slot for the new cell
    let slot = atomicAdd(&next_free_slot[0], 1u);
    let capacity = arrayLength(&positions);
    
    if (slot >= capacity) {
        return; // At capacity
    }
    
    // Set cell position and mass
    positions[slot] = vec4<f32>(insertion_params.world_position, insertion_params.initial_mass);
    
    // Set initial velocity (zero)
    velocities[slot] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    // Set initial rotation (identity)
    rotations[slot] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    // Set cell state
    birth_times[slot] = insertion_params.birth_time;
    genome_ids[slot] = insertion_params.genome_id;
    mode_indices[slot] = insertion_params.local_mode_idx;
    
    // Use provided cell ID or generate new one
    let final_cell_id = max(insertion_params.cell_id, 1u); // Ensure ID is at least 1
    cell_ids[slot] = final_cell_id;
    
    // Get mode properties from per-genome buffers
    let mode_props_0 = get_mode_properties(insertion_params.genome_id, insertion_params.local_mode_idx, 0u);
    let mode_props_1 = get_mode_properties(insertion_params.genome_id, insertion_params.local_mode_idx, 1u);
    let mode_props_2 = get_mode_properties(insertion_params.genome_id, insertion_params.local_mode_idx, 2u);
    
    // Set cell properties from genome mode
    nutrient_gain_rates[slot] = mode_props_0.x;
    max_cell_sizes[slot] = mode_props_0.y;
    stiffnesses[slot] = mode_props_0.z;
    split_intervals[slot] = mode_props_0.w;
    split_masses[slot] = mode_props_1.x;
    
    // Set max splits (use provided value or from genome)
    if (insertion_params.max_splits >= 0) {
        max_splits[slot] = u32(insertion_params.max_splits);
    } else {
        max_splits[slot] = u32(mode_props_2.x);
    }
    
    // Increment cell count
    let new_count = atomicAdd(&cell_count_buffer[0], 1u);
}
