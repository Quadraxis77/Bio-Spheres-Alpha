// Stage 2: Assign cells to grid cells and count cells per grid cell
// Each cell computes its grid index and increments the count for that grid cell
// Workgroup size: 64 threads for cell operations
//
// This pass:
// 1. Calculates which grid cell each cell belongs to
// 2. Stores the grid index in cell_grid_indices[cell_idx]
// 3. Increments spatial_grid_counts[grid_idx] (using simple write, not atomic)
//
// NOTE: Since multiple cells may map to the same grid cell, we use a two-pass
// approach. This pass just stores the grid index. The insert pass will do
// the actual counting and insertion.

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

@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<u32>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

// This buffer stores the grid index for each cell (indexed by cell_idx)
@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

const GRID_RESOLUTION: i32 = 64;

fn world_pos_to_grid_index(pos: vec3<f32>, world_size: f32, grid_cell_size: f32) -> u32 {
    let grid_pos = (pos + world_size * 0.5) / grid_cell_size;
    let grid_x = clamp(i32(grid_pos.x), 0, GRID_RESOLUTION - 1);
    let grid_y = clamp(i32(grid_pos.y), 0, GRID_RESOLUTION - 1);
    let grid_z = clamp(i32(grid_pos.z), 0, GRID_RESOLUTION - 1);
    
    return u32(grid_x + grid_y * GRID_RESOLUTION + grid_z * GRID_RESOLUTION * GRID_RESOLUTION);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let pos = positions_in[cell_idx].xyz;
    
    // Calculate and store grid index for this cell
    let grid_idx = world_pos_to_grid_index(pos, params.world_size, params.grid_cell_size);
    cell_grid_indices[cell_idx] = grid_idx;
}
