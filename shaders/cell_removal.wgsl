//! GPU Cell Removal Compute Shader
//! 
//! Marks a specific cell for removal by setting its mass to 0.
//! The lifecycle death scan shader will detect this and handle the actual removal.
//! This shader is used by the GPU tool operations system to remove cells without CPU involvement.
//! 
//! Requirements:
//! - Mark a specific cell for removal in GPU buffers
//! - Set mass to 0 in ALL THREE triple buffer sets
//! - Use single workgroup (1,1,1) dispatch for single cell update
//! - Receive cell index via uniform buffer
//! - Validate cell index bounds before updating

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

/// Cell removal parameters passed from CPU
/// Layout must match Rust CellRemovalParams struct (16 bytes total)
struct CellRemovalParams {
    cell_index: u32,            // 4 bytes at offset 0
    _pad0: u32,                 // 4 bytes at offset 4
    _pad1: u32,                 // 4 bytes at offset 8
    _pad2: u32,                 // 4 bytes at offset 12
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

// Triple-buffered positions (all 3 sets for read/write)
@group(0) @binding(1)
var<storage, read_write> positions_0: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> positions_1: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_2: array<vec4<f32>>;

// Triple-buffered velocities (all 3 sets for read/write)
@group(0) @binding(4)
var<storage, read_write> velocities_0: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> velocities_1: array<vec4<f32>>;

@group(0) @binding(6)
var<storage, read_write> velocities_2: array<vec4<f32>>;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(7)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<uniform> removal_params: CellRemovalParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Single workgroup dispatch - only thread 0 does the work
    if (global_id.x != 0u) {
        return;
    }
    
    let cell_index = removal_params.cell_index;
    
    // Validate cell index bounds before updating
    let cell_count = cell_count_buffer[0];
    if (cell_index >= cell_count) {
        return;
    }
    
    // Read current position from buffer 0 (any buffer would work)
    let current_pos = positions_0[cell_index].xyz;
    
    // Set mass to 0 to mark cell for death
    // The lifecycle death scan shader will detect mass < DEATH_MASS_THRESHOLD (0.5)
    // and mark this cell as dead, triggering proper removal through the lifecycle pipeline
    let position_zero_mass = vec4<f32>(current_pos, 0.0);
    
    // Update position (with zero mass) in ALL THREE triple buffer sets
    positions_0[cell_index] = position_zero_mass;
    positions_1[cell_index] = position_zero_mass;
    positions_2[cell_index] = position_zero_mass;
    
    // Zero out velocity as well
    let zero_velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    velocities_0[cell_index] = zero_velocity;
    velocities_1[cell_index] = zero_velocity;
    velocities_2[cell_index] = zero_velocity;
}
