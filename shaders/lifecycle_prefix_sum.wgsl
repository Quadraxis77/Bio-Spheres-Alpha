// Lifecycle Prefix Sum Shader
// Stage 2: Compact free slots from dead cells using atomic operations
// Workgroup size: 64 threads for cell operations
//
// Input: death_flags array (1 = dead, 0 = alive)
//        lifecycle_counts already cleared by death_scan shader
// Output: 
// - free_slot_indices: compacted array of dead cell indices
// - lifecycle_counts[2]: D = dead cell count
//
// Uses atomic operations for O(1) per thread instead of O(N) prefix sum
// NOTE: Counters are cleared in death_scan shader (runs before this)

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
// Already cleared by death_scan shader
@group(1) @binding(4)
var<storage, read_write> lifecycle_counts: array<atomic<u32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    
    // Only process existing cells
    if (cell_idx >= cell_count) {
        return;
    }
    
    let is_dead = death_flags[cell_idx] == 1u;
    
    // If this cell is dead, atomically get a slot and write to free_slot_indices
    if (is_dead) {
        let slot = atomicAdd(&lifecycle_counts[2], 1u);
        free_slot_indices[slot] = cell_idx;
    }
}
