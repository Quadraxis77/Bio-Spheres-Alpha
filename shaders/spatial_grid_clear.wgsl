// Stage 1: Clear spatial grid counts
// Workgroup size: 256 threads for optimal grid operations
//
// Only clears spatial_grid_counts - the actual cell data doesn't need clearing
// because we use the count to know how many valid entries exist

@group(0) @binding(0)
var<storage, read_write> spatial_grid_counts: array<u32>;

@group(0) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(0) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

// Unused in this shader but required for bind group layout compatibility
@group(0) @binding(4)
var<storage, read> stiffnesses: array<f32>;

const GRID_RESOLUTION: u32 = 64u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.x;
    let total_grid_cells = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
    
    if (grid_idx >= total_grid_cells) {
        return;
    }
    
    // Only clear the count - we use this to know how many valid cells exist
    // No need to clear spatial_grid_cells since we only read up to count entries
    spatial_grid_counts[grid_idx] = 0u;
}
