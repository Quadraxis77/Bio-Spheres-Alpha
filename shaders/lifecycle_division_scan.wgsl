// Lifecycle Division Scan Shader
// Stage 3: Identify cells ready to divide and compact reservations
// Workgroup size: 64 threads for cell operations
//
// Input: 
// - Cell state (mass, birth_time, split_interval, split_mass)
// - lifecycle_counts[0] = N (free slots available)
//
// Output:
// - division_flags[cell_idx] = 1 for cells ready to divide
// - division_slot_assignments[cell_idx] = reservation index for this cell
// - lifecycle_counts[1] = M (total reservations needed)
//
// Division criteria:
// - Cell must be alive (death_flags == 0)
// - Cell mass >= split_mass threshold
// - Time since birth >= split_interval
// - splits_remaining (current_splits < max_splits or max_splits == 0)
//
// The availability check (N >= M) happens in the execute shader.

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

// Helper function to check if a cell wants to divide
fn cell_wants_to_divide(cell_idx: u32) -> bool {
    // Skip dead cells - they can't divide
    if (death_flags[cell_idx] == 1u) {
        return false;
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
    
    return mass_ready && time_ready && splits_remaining;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let wants_to_divide = cell_wants_to_divide(cell_idx);
    
    // Write division flag
    division_flags[cell_idx] = select(0u, 1u, wants_to_divide);
    
    // Count dividing cells before this one (exclusive prefix sum)
    // This creates the compacted reservation array
    var prefix_count = 0u;
    for (var i = 0u; i < cell_idx; i++) {
        if (cell_wants_to_divide(i)) {
            prefix_count += 1u;
        }
    }
    
    // If this cell wants to divide, assign it a reservation index
    // This is the index into the free_slot_indices array
    if (wants_to_divide) {
        division_slot_assignments[cell_idx] = prefix_count;
    }
    
    // Last cell writes total reservation count (M)
    if (cell_idx == cell_count - 1u) {
        var total_reservations = prefix_count;
        if (wants_to_divide) {
            total_reservations += 1u;
        }
        lifecycle_counts[1] = total_reservations; // M = reservations needed
    }
}
