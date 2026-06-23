// Spatial Grid Build - Combined assign + insert in a single dispatch
// Replaces the previous two-pass (spatial_grid_assign + spatial_grid_insert) approach.
// Also skips dead cells (mass < 0.5) to avoid wasting grid slots and collision checks.
// Workgroup size: 256 threads for optimal GPU occupancy
//
// Each thread:
// 1. Computes which grid cell its cell belongs to
// 2. Stores the grid index in cell_grid_indices (needed by collision shader)
// 3. Skips dead cells (mass < 0.5) - they don't participate in collisions
// 4. Atomically claims a slot in the grid cell and inserts the cell index

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

// Per-cell grid index (indexed by cell_idx, read by collision shader)
@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

// Cell indices stored per grid cell (16 cells max per grid cell)
@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

// Unused in this shader but required for bind group layout compatibility
@group(1) @binding(4)
var<storage, read> stiffnesses: array<f32>;

// Unused in this shader but required for bind group layout compatibility
@group(1) @binding(5)
var<storage, read> organism_labels: array<u32>;

// Dense list of buckets occupied this frame. Appended when a bucket first receives a cell.
@group(1) @binding(6)
var<storage, read_write> occupied_grid_cells: array<u32>;

// [0] = occupied bucket count.
@group(1) @binding(7)
var<storage, read_write> occupied_grid_count: array<atomic<u32>>;

// Compact side list for cells that overflow the fixed per-bucket slots.
@group(1) @binding(8)
var<storage, read_write> spatial_grid_overflow_cells: array<u32>;

@group(1) @binding(9)
var<storage, read_write> spatial_grid_overflow_grid_indices: array<u32>;

@group(1) @binding(10)
var<storage, read_write> spatial_grid_overflow_count: array<atomic<u32>>;

const MAX_CELLS_PER_GRID: u32 = 16u;

fn world_pos_to_grid_index(pos: vec3<f32>, world_size: f32, grid_cell_size: f32, grid_resolution: i32) -> u32 {
    let grid_pos = (pos + world_size * 0.5) / grid_cell_size;
    let grid_x = clamp(i32(grid_pos.x), 0, grid_resolution - 1);
    let grid_y = clamp(i32(grid_pos.y), 0, grid_resolution - 1);
    let grid_z = clamp(i32(grid_pos.z), 0, grid_resolution - 1);
    
    return u32(grid_x + grid_y * grid_resolution + grid_z * grid_resolution * grid_resolution);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let pos = positions_in[cell_idx].xyz;
    let mass = positions_in[cell_idx].w;
    
    // Calculate and store grid index (collision shader reads this)
    let grid_idx = world_pos_to_grid_index(pos, params.world_size, params.grid_cell_size, params.grid_resolution);
    cell_grid_indices[cell_idx] = grid_idx;
    
    // Skip dead cells - don't insert into spatial grid.
    // This saves grid slots for live cells and avoids collision checks against dead cells.
    if (mass < 0.5) {
        return;
    }
    
    // Atomically claim a slot in this grid cell and insert
    let slot = atomicAdd(&spatial_grid_counts[grid_idx], 1u);
    if (slot == 0u) {
        let occupied_slot = atomicAdd(&occupied_grid_count[0], 1u);
        if (occupied_slot < params.cell_capacity) {
            occupied_grid_cells[occupied_slot] = grid_idx;
        }
    }
    if (slot < MAX_CELLS_PER_GRID) {
        spatial_grid_cells[grid_idx * MAX_CELLS_PER_GRID + slot] = cell_idx;
    } else {
        let overflow_slot = atomicAdd(&spatial_grid_overflow_count[0], 1u);
        if (overflow_slot < params.cell_capacity) {
            spatial_grid_overflow_cells[overflow_slot] = cell_idx;
            spatial_grid_overflow_grid_indices[overflow_slot] = grid_idx;
        }
    }
}
