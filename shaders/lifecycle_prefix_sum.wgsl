// Lifecycle Prefix Sum Shader
// Stage 2: Compact free slots from dead cells
// Workgroup size: 64 threads for cell operations
//
// Input: death_flags array (1 = dead, 0 = alive)
// Output: 
// - free_slot_indices: compacted array of dead cell indices
// - lifecycle_counts[0]: N = total free slots available
//
// This creates the dense slot array [slot₀, slot₁, ..., slotₙ₋₁]
// that will be used for division slot assignment.

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

// These bindings must be declared to match the physics bind group layout
// even though this shader doesn't use them
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    
    // Only process existing cells
    if (cell_idx >= cell_count) {
        return;
    }
    
    let is_dead = death_flags[cell_idx] == 1u;
    
    // Count dead cells before this one (exclusive prefix sum)
    var prefix_count = 0u;
    for (var i = 0u; i < cell_idx; i++) {
        if (death_flags[i] == 1u) {
            prefix_count += 1u;
        }
    }
    
    // If this cell is dead, write its index to the compacted free_slot_indices array
    if (is_dead) {
        free_slot_indices[prefix_count] = cell_idx;
    }
    
    // Last cell writes total count of free slots
    // Free slots = dead cells + unused capacity (cell_capacity - cell_count)
    if (cell_idx == cell_count - 1u) {
        var dead_count = prefix_count;
        if (is_dead) {
            dead_count += 1u;
        }
        
        // Store dead_count in lifecycle_counts[2] for the execute shader
        lifecycle_counts[2] = dead_count;
        
        // Total free slots = dead cells + unused capacity
        let unused_capacity = params.cell_capacity - cell_count;
        lifecycle_counts[0] = dead_count + unused_capacity;
    }
}
