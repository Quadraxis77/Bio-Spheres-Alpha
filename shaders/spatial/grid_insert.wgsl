//! # Spatial Grid Insertion Compute Shader
//!
//! This compute shader inserts cell indices into the spatial grid based on their
//! grid assignments. It builds a sorted list of cell indices for each grid cell
//! to enable efficient collision detection through spatial partitioning.
//!
//! ## Algorithm
//! 1. Each thread processes one cell
//! 2. Read the cell's grid assignment (calculated by grid_assign.wgsl)
//! 3. Atomically allocate a slot in the grid cell's index list
//! 4. Insert the cell index into the allocated slot
//!
//! ## Data Structure
//! - `grid_offsets`: Starting index in grid_indices for each grid cell
//! - `grid_indices`: Flat array of cell indices sorted by grid cell
//! - `grid_assignments`: Maps each cell to its grid cell index
//!
//! ## Memory Layout
//! Grid indices are stored in a flat array with offsets pointing to the start
//! of each grid cell's index list:
//! ```
//! grid_indices: [cell0, cell5, cell2, cell1, cell3, cell4, ...]
//! grid_offsets: [0,     2,     4,     6,     ...]
//!               ^      ^      ^      ^
//!            grid0  grid1  grid2  grid3
//! ```
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Atomic Operations**: One atomic increment per cell per frame
//! - **Memory Access**: Sequential read, random write with atomic contention
//! - **Cache Efficiency**: Good spatial locality within grid cells

// Physics parameters uniform buffer
struct PhysicsParams {
    // Time and frame info
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    
    // World and physics settings
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    
    // Grid settings
    grid_resolution: i32,        // 64 for 64³ grid
    grid_cell_size: f32,         // world_size / grid_resolution
    max_cells_per_grid: i32,     // Maximum cells per grid cell (32)
    enable_thrust_force: i32,
    
    // UI interaction
    dragged_cell_index: i32,
    _padding1: vec3<f32>,
    
    // Padding to 256 bytes
    _padding: array<f32, 48>,
}

// Bind group 0: Physics parameters and spatial grid buffers
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> grid_assignments: array<u32>;
@group(0) @binding(2) var<storage, read> grid_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> grid_insertion_counters: array<atomic<u32>>;

/// Main compute shader entry point.
///
/// Each thread processes one cell and inserts its index into the appropriate
/// position in the spatial grid index array. The insertion uses atomic operations
/// to handle multiple cells being inserted into the same grid cell simultaneously.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's grid insertion
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Atomic Operations
/// - Uses atomic increment on grid_insertion_counters for thread-safe insertion
/// - Ensures correct ordering when multiple cells map to the same grid cell
/// - Critical for maintaining data structure integrity
///
/// ## Memory Access Pattern
/// - Sequential read access to grid_assignments (cache-friendly)
/// - Random read access to grid_offsets (cached from prefix sum)
/// - Random write access to grid_indices (may cause cache misses)
/// - Atomic operations on grid_insertion_counters (may cause contention)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond the active cell count
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Read which grid cell this cell is assigned to
    let grid_index = grid_assignments[cell_index];
    
    // Get the starting offset for this grid cell in the indices array
    let grid_start_offset = grid_offsets[grid_index];
    
    // Atomically allocate a slot within this grid cell's index list
    // This ensures thread-safe insertion when multiple cells map to the same grid cell
    let insertion_offset = atomicAdd(&grid_insertion_counters[grid_index], 1u);
    
    // Calculate the final position in the grid_indices array
    let final_index = grid_start_offset + insertion_offset;
    
    // Bounds check: ensure we don't write beyond the allocated space for this grid cell
    // This prevents buffer overruns when too many cells are assigned to one grid cell
    let max_cells_per_grid = u32(physics_params.max_cells_per_grid);
    if (insertion_offset < max_cells_per_grid) {
        // Insert the cell index into the spatial grid structure
        grid_indices[final_index] = cell_index;
    }
    
    // Note: If insertion_offset >= max_cells_per_grid, the cell is silently dropped
    // This is a safety mechanism to prevent buffer overruns in dense regions
    // In practice, max_cells_per_grid should be set high enough to avoid this
}