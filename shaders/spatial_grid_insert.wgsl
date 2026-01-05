// Stage 3: Build spatial grid cell list using deterministic scatter
// Each cell computes its slot by counting lower-indexed cells in same grid cell
// Workgroup size: 64 threads for cell operations
//
// This uses workgroup shared memory to accelerate the counting within each workgroup

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

@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<u32>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

const MAX_CELLS_PER_GRID: u32 = 16u;
const WORKGROUP_SIZE: u32 = 64u;

// Shared memory for workgroup-local grid indices
var<workgroup> local_grid_indices: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let cell_idx = global_id.x;
    let local_idx = local_id.x;
    let workgroup_start = workgroup_id.x * WORKGROUP_SIZE;
    
    // Load this cell's grid index into shared memory (or invalid if out of bounds)
    if (cell_idx < params.cell_count) {
        local_grid_indices[local_idx] = cell_grid_indices[cell_idx];
    } else {
        local_grid_indices[local_idx] = 0xFFFFFFFFu; // Invalid marker
    }
    
    workgroupBarrier();
    
    if (cell_idx >= params.cell_count) {
        return;
    }
    
    let my_grid_idx = local_grid_indices[local_idx];
    
    // Count cells with lower global index in the same grid cell
    // First: count within this workgroup (fast, uses shared memory)
    var local_offset = 0u;
    for (var i = 0u; i < local_idx; i++) {
        if (local_grid_indices[i] == my_grid_idx) {
            local_offset += 1u;
        }
    }
    
    // Second: count cells in previous workgroups (slower, reads global memory)
    // But we only need to check cells that could be in the same grid cell
    for (var i = 0u; i < workgroup_start; i++) {
        if (cell_grid_indices[i] == my_grid_idx) {
            local_offset += 1u;
        }
    }
    
    // Write to spatial_grid_cells if we have room
    if (local_offset < MAX_CELLS_PER_GRID) {
        let grid_base_offset = my_grid_idx * MAX_CELLS_PER_GRID;
        spatial_grid_cells[grid_base_offset + local_offset] = cell_idx;
    }
    
    // Update the count for this grid cell (only the last cell in each grid cell does this)
    // We need to count total cells in this grid cell
    var is_last_in_grid = true;
    
    // Check if any cell after us in this workgroup has the same grid index
    for (var i = local_idx + 1u; i < WORKGROUP_SIZE; i++) {
        if (local_grid_indices[i] == my_grid_idx) {
            is_last_in_grid = false;
            break;
        }
    }
    
    // Check cells after this workgroup
    if (is_last_in_grid) {
        let next_workgroup_start = workgroup_start + WORKGROUP_SIZE;
        for (var i = next_workgroup_start; i < params.cell_count; i++) {
            if (cell_grid_indices[i] == my_grid_idx) {
                is_last_in_grid = false;
                break;
            }
        }
    }
    
    // If we're the last cell in this grid cell, write the total count
    if (is_last_in_grid) {
        spatial_grid_counts[my_grid_idx] = min(local_offset + 1u, MAX_CELLS_PER_GRID);
    }
}
