//! # Spatial Grid Assignment Compute Shader
//!
//! This compute shader assigns each cell to its appropriate spatial grid cell based
//! on the cell's position. The spatial grid is used for efficient collision detection
//! by partitioning the simulation space into a 64³ grid.
//!
//! ## Algorithm
//! 1. Each thread processes one cell
//! 2. Convert cell world position to grid coordinates
//! 3. Handle boundary conditions and grid wrapping
//! 4. Calculate linear grid index from 3D coordinates
//! 5. Atomically increment grid cell count
//!
//! ## Grid Coordinate System
//! - World space: [-world_size/2, +world_size/2] in each dimension
//! - Grid space: [0, grid_resolution-1] in each dimension (typically [0, 63])
//! - Grid cell size: world_size / grid_resolution
//! - Linear index: x + y*64 + z*64*64 (row-major order)
//!
//! ## Boundary Handling
//! - Cells outside world bounds are clamped to edge grid cells
//! - This prevents cells from escaping the simulation boundary
//! - Ensures all cells are assigned to valid grid cells
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Random access to grid_counts (atomic operations)
//! - **Atomic Operations**: One atomic increment per cell per frame
//! - **Cache Efficiency**: Spatial locality in grid access patterns

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
    
    // Padding to 256 bytes (using vec4<f32> for proper 16-byte alignment)
    _padding: array<vec4<f32>, 12>,
}

// Bind group 0: Physics parameters and spatial grid buffers
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_assignments: array<u32>;

/// Convert world position to grid coordinates.
///
/// Transforms a world-space position to discrete grid coordinates within
/// the spatial partitioning grid. Handles boundary clamping to ensure
/// all positions map to valid grid cells.
///
/// # Arguments
/// * `world_pos` - Position in world space
/// * `world_size` - Size of the simulation world
/// * `grid_resolution` - Number of grid cells per dimension
///
/// # Returns
/// Grid coordinates as vec3<i32> clamped to [0, grid_resolution-1]
fn world_to_grid(world_pos: vec3<f32>, world_size: f32, grid_resolution: i32) -> vec3<i32> {
    // Convert from world space [-world_size/2, +world_size/2] to grid space [0, grid_resolution]
    let half_world = world_size * 0.5;
    let normalized_pos = (world_pos + vec3<f32>(half_world)) / world_size;
    let grid_pos = normalized_pos * f32(grid_resolution);
    
    // Clamp to valid grid range [0, grid_resolution-1]
    // This handles boundary conditions by assigning boundary cells to edge grid cells
    let grid_coords = vec3<i32>(
        clamp(i32(grid_pos.x), 0, grid_resolution - 1),
        clamp(i32(grid_pos.y), 0, grid_resolution - 1),
        clamp(i32(grid_pos.z), 0, grid_resolution - 1)
    );
    
    return grid_coords;
}

/// Convert 3D grid coordinates to linear grid index.
///
/// Converts 3D grid coordinates to a linear index for array access.
/// Uses row-major ordering: index = x + y*width + z*width*height
///
/// # Arguments
/// * `grid_coords` - 3D grid coordinates
/// * `grid_resolution` - Number of grid cells per dimension
///
/// # Returns
/// Linear grid index for array access
fn grid_coords_to_index(grid_coords: vec3<i32>, grid_resolution: i32) -> u32 {
    return u32(grid_coords.x + grid_coords.y * grid_resolution + grid_coords.z * grid_resolution * grid_resolution);
}

/// Main compute shader entry point.
///
/// Each thread processes one cell and assigns it to the appropriate spatial grid cell.
/// The assignment involves converting the cell's world position to grid coordinates,
/// calculating the linear grid index, and atomically incrementing the grid cell count.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's position
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Atomic Operations
/// - Uses atomic increment on grid_counts to handle multiple cells in same grid cell
/// - Ensures thread-safe counting when multiple cells map to the same grid cell
/// - Critical for correctness in dense cell regions
///
/// ## Memory Access Pattern
/// - Sequential read access to positions array (cache-friendly)
/// - Random write access to grid_counts (atomic, may cause contention)
/// - Sequential write access to grid_assignments (cache-friendly)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond the active cell count
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Read cell position (positions are stored as Vec4 with mass in w component)
    let position = positions[cell_index].xyz;
    
    // Convert world position to grid coordinates
    let grid_coords = world_to_grid(
        position,
        physics_params.world_size,
        physics_params.grid_resolution
    );
    
    // Convert 3D grid coordinates to linear grid index
    let grid_index = grid_coords_to_index(grid_coords, physics_params.grid_resolution);
    
    // Store the grid assignment for this cell
    // This will be used later by the grid insertion shader
    grid_assignments[cell_index] = grid_index;
    
    // Atomically increment the count for this grid cell
    // This is thread-safe and handles multiple cells mapping to the same grid cell
    atomicAdd(&grid_counts[grid_index], 1u);
}