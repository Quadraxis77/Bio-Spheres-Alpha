//! GPU Cell Boost Compute Shader
//! 
//! Boosts a specific cell's mass to the maximum cap to trigger division.
//! Sets the cell's mass to a high value (10.0) that exceeds typical max_cell_size values.
//! This shader is used by the GPU tool operations system to boost cells without CPU involvement.
//! 
//! Requirements:
//! - Set a specific cell's mass to the maximum cap in GPU buffers
//! - Update mass in ALL THREE triple buffer sets
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

/// Cell boost parameters passed from CPU
/// Layout must match Rust CellBoostParams struct (16 bytes total)
struct CellBoostParams {
    cell_index: u32,            // 4 bytes at offset 0
    _pad0: u32,                 // 4 bytes at offset 4
    _pad1: u32,                 // 4 bytes at offset 8
    _pad2: u32,                 // 4 bytes at offset 12
}

// Maximum mass cap - high enough to exceed any reasonable max_cell_size
// Typical max_cell_size is 2x split_mass, so 10.0 should be plenty
const BOOST_MASS: f32 = 10.0;

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
var<uniform> boost_params: CellBoostParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Single workgroup dispatch - only thread 0 does the work
    if (global_id.x != 0u) {
        return;
    }
    
    let cell_index = boost_params.cell_index;
    
    // Validate cell index bounds before updating
    let cell_count = cell_count_buffer[0];
    if (cell_index >= cell_count) {
        return;
    }
    
    // Read current position from buffer 0 (any buffer would work)
    let current_pos = positions_0[cell_index].xyz;
    
    // Set mass to the maximum cap (high enough to trigger division)
    // The lifecycle division scan shader checks: mass >= split_mass
    let position_boosted_mass = vec4<f32>(current_pos, BOOST_MASS);
    
    // Update position (with boosted mass) in ALL THREE triple buffer sets
    positions_0[cell_index] = position_boosted_mass;
    positions_1[cell_index] = position_boosted_mass;
    positions_2[cell_index] = position_boosted_mass;
}
