//! # Spatial Grid Clear Compute Shader
//!
//! This compute shader resets all spatial grid cell counts to zero at the beginning
//! of each physics step. The spatial grid is used for efficient collision detection
//! by partitioning the simulation space into a 64³ grid.
//!
//! ## Algorithm
//! 1. Each thread processes one grid cell
//! 2. Grid cell count is reset to 0
//! 3. Grid cell offset is reset to 0 (will be calculated by prefix sum)
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 256 threads for optimal memory bandwidth utilization
//! - **Memory Access**: Sequential access pattern for cache efficiency
//! - **Grid Size**: 64³ = 262,144 grid cells total
//! - **Memory Bandwidth**: ~1MB of data cleared per frame (262,144 * 4 bytes)
//!
//! ## Memory Layout
//! - Grid cells are stored in row-major order: [x + y*64 + z*64*64]
//! - Each grid cell stores a count (u32) and offset (u32) for collision detection
//! - Total memory: 262,144 cells * 8 bytes = ~2MB for grid metadata

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

// Bind group 0: Physics parameters and grid buffers
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> grid_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> grid_offsets: array<u32>;

/// Main compute shader entry point.
///
/// Each thread processes one grid cell and resets its count and offset to zero.
/// The workgroup size of 256 is chosen for optimal memory bandwidth utilization
/// on modern GPUs, allowing efficient clearing of the entire 64³ grid.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to grid cell index
/// - Grid cells are indexed in row-major order: [x + y*64 + z*64*64]
/// - Total threads needed: 262,144 (64³)
/// - Workgroups needed: 1024 (262,144 / 256)
///
/// ## Memory Access Pattern
/// - Sequential access to grid_counts and grid_offsets arrays
/// - Coalesced memory access within each warp/wavefront
/// - Cache-friendly access pattern for maximum bandwidth
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_index = global_id.x;
    
    // Calculate total number of grid cells (64³ = 262,144)
    let grid_resolution = u32(physics_params.grid_resolution);
    let total_grid_cells = grid_resolution * grid_resolution * grid_resolution;
    
    // Bounds check: ensure we don't write beyond the grid arrays
    if (grid_index >= total_grid_cells) {
        return;
    }
    
    // Reset grid cell count to zero
    // This count will be incremented during the grid assignment phase
    grid_counts[grid_index] = 0u;
    
    // Reset grid cell offset to zero
    // This offset will be calculated during the prefix sum phase
    // and will point to the starting index in the grid_indices array
    grid_offsets[grid_index] = 0u;
}