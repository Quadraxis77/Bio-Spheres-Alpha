// Stage 3: Insert cells into spatial grid using atomic operations
// Each cell atomically claims a slot in its grid cell
// Workgroup size: 256 threads for optimal GPU occupancy
//
// Algorithm:
// 1. Read this cell's pre-computed grid index from cell_grid_indices
// 2. Atomically increment spatial_grid_counts[grid_idx] to get slot
// 3. Write cell_idx to spatial_grid_cells[grid_base + slot]
//
// This is O(1) per cell using atomics - no iteration required

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

// Atomic counts per grid cell
@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<atomic<u32>>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

// Pre-computed grid index for each cell (from spatial_grid_assign)
@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

// Cell indices stored per grid cell (16 cells max per grid cell)
@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

// Unused in this shader but required for bind group layout compatibility
@group(1) @binding(4)
var<storage, read> stiffnesses: array<f32>;

const MAX_CELLS_PER_GRID: u32 = 16u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Get this cell's grid index (pre-computed in spatial_grid_assign)
    let grid_idx = cell_grid_indices[cell_idx];
    
    // Atomically get a slot in this grid cell
    let slot = atomicAdd(&spatial_grid_counts[grid_idx], 1u);
    
    // Only insert if we have room (max 16 cells per grid cell)
    if (slot < MAX_CELLS_PER_GRID) {
        let grid_base_offset = grid_idx * MAX_CELLS_PER_GRID;
        spatial_grid_cells[grid_base_offset + slot] = cell_idx;
    }
}
