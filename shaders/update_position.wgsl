//! GPU Position Update Compute Shader
//! 
//! Updates a specific cell's position directly in GPU buffers for cell dragging operations.
//! This shader is used by the GPU tool operations system to move cells without CPU involvement.
//! 
//! Requirements:
//! - Update a specific cell's position in GPU buffers (10.1)
//! - Preserve the cell's mass while updating position (10.2)
//! - Update position in ALL THREE triple buffer sets (10.3)
//! - Use single workgroup (1,1,1) dispatch for single cell update (10.4)
//! - Receive cell index and new position via uniform buffer (10.5)
//! - Validate cell index bounds before updating (10.6)

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

/// Position update parameters passed from CPU
/// Layout must match Rust PositionUpdateParams struct (32 bytes total)
struct PositionUpdateParams {
    cell_index: u32,            // 4 bytes at offset 0
    _pad0: u32,                 // 4 bytes at offset 4 (padding for vec3 alignment)
    _pad1: u32,                 // 4 bytes at offset 8
    _pad2: u32,                 // 4 bytes at offset 12
    new_position: vec3<f32>,    // 12 bytes at offset 16
    _padding: f32,              // 4 bytes at offset 28
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
var<uniform> update_params: PositionUpdateParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Single workgroup dispatch as required by 10.4
    // Only thread 0 does the work
    if (global_id.x != 0u) {
        return;
    }
    
    let cell_index = update_params.cell_index;
    let new_position = update_params.new_position;
    
    // Validate cell index bounds before updating (requirement 10.6)
    let cell_count = cell_count_buffer[0];
    if (cell_index >= cell_count) {
        return;
    }
    
    // Read current cell mass from all buffers
    // Division cells might only have data in one buffer, so we need to find the valid one
    // A valid mass is > 0 (cells always have positive mass)
    let mass_0 = positions_0[cell_index].w;
    let mass_1 = positions_1[cell_index].w;
    let mass_2 = positions_2[cell_index].w;
    
    // Find the first valid mass (> 0)
    var mass = 0.0;
    if (mass_0 > 0.0) {
        mass = mass_0;
    } else if (mass_1 > 0.0) {
        mass = mass_1;
    } else if (mass_2 > 0.0) {
        mass = mass_2;
    }
    
    // Skip if no valid mass found (cell doesn't exist in any buffer)
    if (mass <= 0.0) {
        return;
    }
    
    // Smooth boundary clamping with lerping to prevent teleporting
    let boundary_radius = params.world_size * 0.5;
    let dist = length(new_position);
    var final_position = new_position;
    
    if (dist > boundary_radius) {
        // Calculate penetration depth
        let penetration = dist - boundary_radius;
        
        // Smooth lerp factor based on penetration depth
        let max_penetration = 5.0; // Maximum penetration for full correction
        let lerp_factor = clamp(penetration / max_penetration, 0.0, 1.0);
        let smooth_lerp = lerp_factor * lerp_factor; // Quadratic for smoother transition
        
        // Calculate target position (just inside boundary)
        let target_position = normalize(new_position) * boundary_radius * 0.99;
        
        // Smoothly lerp new position toward target position
        final_position = mix(new_position, target_position, smooth_lerp);
    }
    
    // Create new position with preserved mass
    let position_mass = vec4<f32>(final_position, mass);
    
    // Update position in ALL THREE triple buffer sets (requirement 10.3)
    positions_0[cell_index] = position_mass;
    positions_1[cell_index] = position_mass;
    positions_2[cell_index] = position_mass;
    
    // Zero out velocity when dragging (cell is being moved by user, not physics)
    let zero_velocity = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    velocities_0[cell_index] = zero_velocity;
    velocities_1[cell_index] = zero_velocity;
    velocities_2[cell_index] = zero_velocity;
}
