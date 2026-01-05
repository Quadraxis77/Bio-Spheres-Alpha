// Lifecycle Death Scan Shader
// Stage 1: Identify dead cells and mark them as free slots
// Workgroup size: 64 threads for cell operations
//
// This runs BEFORE division to make dead cell slots available.
// Dead cells and their adhesions are marked before division.
//
// Output:
// - death_flags[cell_idx] = 1 for dead cells, 0 for alive
// - These become the input for free slot compaction

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

// Death threshold - cells with mass below this are considered dead
const DEATH_MASS_THRESHOLD: f32 = 0.1;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Read mass from OUTPUT buffer (physics results)
    let mass = positions_out[cell_idx].w;
    
    // Check death condition: mass below threshold
    let is_dead = mass < DEATH_MASS_THRESHOLD;
    
    // Write death flag (1 = dead/free slot, 0 = alive)
    death_flags[cell_idx] = select(0u, 1u, is_dead);
}
