// Lifecycle Division Scan Shader
// Stage 3: Identify cells ready to divide using atomic operations
// Workgroup size: 64 threads for cell operations
//
// Input: 
// - Cell state (mass, birth_time, split_interval, split_mass)
// - lifecycle_counts[2] = D (dead cell count from prefix_sum)
//
// Output:
// - division_flags[cell_idx] = 1 for cells ready to divide
// - division_slot_assignments[cell_idx] = reservation index for this cell
// - lifecycle_counts[0] = N (total free slots = dead + unused capacity)
// - lifecycle_counts[1] = M (total reservations needed)
//
// Uses atomic operations for O(1) per thread instead of O(N) prefix sum
// NOTE: lifecycle_counts[0] and [1] are cleared in death_scan shader

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
@group(1) @binding(4)
var<storage, read_write> lifecycle_counts: array<atomic<u32>>;

// Cell state bind group (group 2)
@group(2) @binding(0)
var<storage, read> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read> split_masses: array<f32>;

@group(2) @binding(3)
var<storage, read> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read> max_splits: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    
    // First thread calculates total free slots
    // This must happen before any thread tries to reserve a slot
    if (cell_idx == 0u) {
        // Total free slots = dead cells + unused capacity
        let dead_count = atomicLoad(&lifecycle_counts[2]);
        let unused_capacity = params.cell_capacity - cell_count;
        atomicStore(&lifecycle_counts[0], dead_count + unused_capacity);
    }
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Skip dead cells - they can't divide
    if (death_flags[cell_idx] == 1u) {
        division_flags[cell_idx] = 0u;
        return;
    }
    
    // Read mass from OUTPUT buffer (physics results)
    let mass = positions_out[cell_idx].w;
    let birth_time = birth_times[cell_idx];
    let split_interval = split_intervals[cell_idx];
    let split_mass = split_masses[cell_idx];
    let current_splits = split_counts[cell_idx];
    let max_split = max_splits[cell_idx];
    
    // Calculate age
    let age = params.current_time - birth_time;
    
    // Check division criteria
    let mass_ready = mass >= split_mass;
    let time_ready = age >= split_interval;
    let splits_remaining = current_splits < max_split || max_split == 0u;
    
    let wants_to_divide = mass_ready && time_ready && splits_remaining;
    
    // Write division flag
    division_flags[cell_idx] = select(0u, 1u, wants_to_divide);
    
    // If this cell wants to divide, atomically get a reservation index
    if (wants_to_divide) {
        let reservation_idx = atomicAdd(&lifecycle_counts[1], 1u);
        division_slot_assignments[cell_idx] = reservation_idx;
    }
}
