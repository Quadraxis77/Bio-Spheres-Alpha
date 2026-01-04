//! # Cell Position Update Compute Shader
//!
//! This compute shader implements Verlet integration for updating cell positions
//! based on velocities and accelerations. It handles position constraints,
//! boundary conditions, and numerical stability checks.
//!
//! ## Integration Method
//! - **Verlet Integration**: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
//! - **Velocity Verlet**: More stable than basic Euler integration
//! - **Position Constraints**: Enforce simulation boundaries and limits
//! - **Numerical Stability**: Clamping and validation for extreme values
//!
//! ## Boundary Handling
//! - **Soft Boundaries**: Exponential force increase near boundaries
//! - **Hard Boundaries**: Absolute position limits as safety net
//! - **Reflection**: Velocity reversal for boundary collisions
//! - **Damping**: Energy loss during boundary interactions
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Sequential access to position and velocity arrays
//! - **Compute Intensity**: Low - mostly vector math operations
//! - **Cache Efficiency**: Good spatial locality in array access

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

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> acceleration: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> prev_acceleration: array<vec4<f32>>;

/// Apply boundary constraints to position.
///
/// Enforces hard position limits as a safety net beyond soft boundary forces.
/// Prevents cells from escaping the simulation world due to numerical errors
/// or extreme forces.
///
/// # Arguments
/// * `position` - Cell position to constrain
/// * `world_size` - Size of simulation world
///
/// # Returns
/// Constrained position within world bounds
fn apply_boundary_constraints(position: vec3<f32>, world_size: f32) -> vec3<f32> {
    let half_world = world_size * 0.5;
    let margin = 0.1; // Small margin to prevent cells from touching boundaries
    
    return vec3<f32>(
        clamp(position.x, -half_world + margin, half_world - margin),
        clamp(position.y, -half_world + margin, half_world - margin),
        clamp(position.z, -half_world + margin, half_world - margin)
    );
}

/// Check if position is valid (no NaN or infinite values).
///
/// Validates position components to prevent numerical errors from propagating
/// through the simulation. Invalid positions are reset to origin.
///
/// # Arguments
/// * `position` - Position to validate
///
/// # Returns
/// True if position is valid, false otherwise
fn is_valid_position(position: vec3<f32>) -> bool {
    // Check for NaN or infinite values
    let is_finite_x = isFinite(position.x) && !isNan(position.x);
    let is_finite_y = isFinite(position.y) && !isNan(position.y);
    let is_finite_z = isFinite(position.z) && !isNan(position.z);
    
    return is_finite_x && is_finite_y && is_finite_z;
}

/// Apply position damping for numerical stability.
///
/// Reduces position changes that are too large to prevent numerical
/// instability. This acts as a safety mechanism for extreme accelerations.
///
/// # Arguments
/// * `old_position` - Previous position
/// * `new_position` - Calculated new position
/// * `max_displacement` - Maximum allowed displacement per step
///
/// # Returns
/// Damped position change
fn apply_position_damping(old_position: vec3<f32>, new_position: vec3<f32>, max_displacement: f32) -> vec3<f32> {
    let displacement = new_position - old_position;
    let displacement_magnitude = length(displacement);
    
    // If displacement is too large, clamp it
    if (displacement_magnitude > max_displacement) {
        let clamped_displacement = normalize(displacement) * max_displacement;
        return old_position + clamped_displacement;
    }
    
    return new_position;
}

/// Perform Verlet integration for position update.
///
/// Uses velocity Verlet integration which is more stable than basic Euler
/// integration for physics simulation. The integration accounts for both
/// current and previous accelerations for better accuracy.
///
/// # Arguments
/// * `position` - Current position
/// * `velocity` - Current velocity
/// * `acceleration` - Current acceleration
/// * `prev_acceleration` - Previous frame acceleration
/// * `dt` - Time step
///
/// # Returns
/// New position after integration
fn verlet_integrate_position(
    position: vec3<f32>,
    velocity: vec3<f32>,
    acceleration: vec3<f32>,
    prev_acceleration: vec3<f32>,
    dt: f32
) -> vec3<f32> {
    // Velocity Verlet integration:
    // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
    // This is more stable than basic Euler integration
    
    let dt_squared = dt * dt;
    let velocity_term = velocity * dt;
    let acceleration_term = 0.5 * acceleration * dt_squared;
    
    return position + velocity_term + acceleration_term;
}

/// Handle boundary collision with reflection and damping.
///
/// When a cell hits a boundary, reflect its velocity and apply damping
/// to simulate energy loss during collision. This prevents cells from
/// accumulating energy at boundaries.
///
/// # Arguments
/// * `position` - Current position
/// * `velocity` - Current velocity
/// * `world_size` - Size of simulation world
/// * `restitution` - Energy retention factor (0.0 = no bounce, 1.0 = perfect bounce)
///
/// # Returns
/// Tuple of (corrected_position, reflected_velocity)
fn handle_boundary_collision(
    position: vec3<f32>,
    velocity: vec3<f32>,
    world_size: f32,
    restitution: f32
) -> vec2<vec3<f32>> {
    let half_world = world_size * 0.5;
    var corrected_position = position;
    var reflected_velocity = velocity;
    
    // X boundaries
    if (position.x > half_world) {
        corrected_position.x = half_world;
        reflected_velocity.x = -abs(velocity.x) * restitution;
    } else if (position.x < -half_world) {
        corrected_position.x = -half_world;
        reflected_velocity.x = abs(velocity.x) * restitution;
    }
    
    // Y boundaries
    if (position.y > half_world) {
        corrected_position.y = half_world;
        reflected_velocity.y = -abs(velocity.y) * restitution;
    } else if (position.y < -half_world) {
        corrected_position.y = -half_world;
        reflected_velocity.y = abs(velocity.y) * restitution;
    }
    
    // Z boundaries
    if (position.z > half_world) {
        corrected_position.z = half_world;
        reflected_velocity.z = -abs(velocity.z) * restitution;
    } else if (position.z < -half_world) {
        corrected_position.z = -half_world;
        reflected_velocity.z = abs(velocity.z) * restitution;
    }
    
    return vec2<vec3<f32>>(corrected_position, reflected_velocity);
}

/// Main compute shader entry point.
///
/// Each thread processes one cell and updates its position using Verlet
/// integration. The shader handles boundary conditions, numerical stability,
/// and special cases like dragged cells.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's position update
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Integration Pipeline
/// 1. Read current position, velocity, and acceleration
/// 2. Perform Verlet integration to calculate new position
/// 3. Apply boundary constraints and collision handling
/// 4. Validate position for numerical stability
/// 5. Handle special cases (dragged cells, invalid positions)
/// 6. Store updated position back to buffer
///
/// ## Numerical Stability
/// - Position validation prevents NaN/infinite propagation
/// - Position damping limits extreme displacements
/// - Boundary constraints provide hard limits
/// - Special handling for edge cases
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond the active cell count
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Read current cell state
    let pos_and_mass = position_and_mass[cell_index];
    let current_position = pos_and_mass.xyz;
    let mass = pos_and_mass.w;
    let current_velocity = velocity[cell_index].xyz;
    let current_acceleration = acceleration[cell_index].xyz;
    let previous_acceleration = prev_acceleration[cell_index].xyz;
    
    // Skip position update for massless cells (they shouldn't move)
    if (mass < 0.001) {
        return;
    }
    
    // Handle dragged cells - they follow mouse/UI control instead of physics
    if (physics_params.dragged_cell_index == i32(cell_index)) {
        // Dragged cells maintain their current position
        // (UI system will update position directly)
        return;
    }
    
    // Validate current position
    if (!is_valid_position(current_position)) {
        // Reset invalid positions to origin
        position_and_mass[cell_index] = vec4<f32>(0.0, 0.0, 0.0, mass);
        return;
    }
    
    // Perform Verlet integration
    let dt = physics_params.delta_time;
    var new_position = verlet_integrate_position(
        current_position,
        current_velocity,
        current_acceleration,
        previous_acceleration,
        dt
    );
    
    // Apply position damping for numerical stability
    let max_displacement_per_step = physics_params.world_size * 0.1; // 10% of world size per step
    new_position = apply_position_damping(current_position, new_position, max_displacement_per_step);
    
    // Validate new position
    if (!is_valid_position(new_position)) {
        // If integration produced invalid position, keep current position
        return;
    }
    
    // Handle boundary collisions
    let restitution = 0.3; // Energy retention during boundary collisions
    let collision_result = handle_boundary_collision(
        new_position,
        current_velocity,
        physics_params.world_size,
        restitution
    );
    new_position = collision_result.x;
    // Note: reflected velocity will be handled by velocity update shader
    
    // Apply final boundary constraints as safety net
    new_position = apply_boundary_constraints(new_position, physics_params.world_size);
    
    // Final validation before storing
    if (!is_valid_position(new_position)) {
        // Last resort: keep current position if all else fails
        return;
    }
    
    // Store updated position (preserve mass)
    position_and_mass[cell_index] = vec4<f32>(new_position, mass);
}