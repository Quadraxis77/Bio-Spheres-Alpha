// Stage 1: sparsely clear the buckets occupied by the previous physics step.
// The dense occupied list is at most one entry per live cell, avoiding a fixed
// 128^3 count-buffer clear on every catch-up step.

// Use atomic for consistency with spatial_grid_insert which uses atomicAdd
@group(0) @binding(0)
var<storage, read_write> spatial_grid_counts: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(0) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

// Unused in this shader but required for bind group layout compatibility
@group(0) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

// Unused in this shader but required for bind group layout compatibility
@group(0) @binding(4)
var<storage, read> stiffnesses: array<f32>;

@group(0) @binding(6)
var<storage, read_write> occupied_grid_cells: array<u32>;

@group(0) @binding(7)
var<storage, read_write> occupied_grid_count: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let occupied_idx = global_id.x;
    let occupied_count = atomicLoad(&occupied_grid_count[0]);
    if (occupied_idx >= occupied_count) {
        return;
    }

    let grid_idx = occupied_grid_cells[occupied_idx];
    atomicStore(&spatial_grid_counts[grid_idx], 0u);
}
