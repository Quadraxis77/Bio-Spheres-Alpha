//! # Cell Velocity Update Compute Shader
//!
//! This compute shader implements velocity integration from accelerations with
//! damping, velocity limits, and angular velocity/acceleration updates. It uses
//! the velocity Verlet method for stable integration.
//!
//! ## Integration Method
//! - **Velocity Verlet**: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
//! - **Angular Integration**: Similar method for angular velocity and acceleration
//! - **Damping**: Velocity and angular velocity damping for stability
//! - **Limits**: Maximum velocity and angular velocity constraints
//!
//! ## Velocity Handling
//! - **Linear Velocity**: Updated from linear acceleration
//! - **Angular Velocity**: Updated from angular acceleration (torque)
//! - **Damping**: Reduces velocities over time for realistic motion
//! - **Clamping**: Prevents extreme velocities that cause instability
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Sequential access to velocity and acceleration arrays
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
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> acceleration: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> prev_acceleration: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> angular_velocity: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> angular_acceleration: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> prev_angular_acceleration: array<vec4<f32>>;

/// Apply velocity damping for realistic motion.
///
/// Reduces velocity over time to simulate air resistance and internal
/// friction. This prevents cells from accelerating indefinitely and
/// provides more realistic motion dynamics.
///
/// # Arguments
/// * `velocity` - Current velocity vector
/// * `damping_factor` - Damping coefficient (0.0 = no damping, 1.0 = full damping)
/// * `dt` - Time step
///
/// # Returns
/// Damped velocity vector
fn apply_velocity_damping(velocity: vec3<f32>, damping_factor: f32, dt: f32) -> vec3<f32> {
    // Exponential damping: v_new = v_old * exp(-damping * dt)
    // Approximated as: v_new = v_old * (1 - damping * dt) for small dt
    let damping_multiplier = 1.0 - clamp(damping_factor * dt, 0.0, 0.99);
    return velocity * damping_multiplier;
}

/// Apply angular velocity damping for rotational stability.
///
/// Similar to linear velocity damping but for rotational motion.
/// Prevents cells from spinning indefinitely and provides realistic
/// rotational dynamics.
///
/// # Arguments
/// * `angular_velocity` - Current angular velocity vector
/// * `damping_factor` - Angular damping coefficient
/// * `dt` - Time step
///
/// # Returns
/// Damped angular velocity vector
fn apply_angular_damping(angular_velocity: vec3<f32>, damping_factor: f32, dt: f32) -> vec3<f32> {
    // Angular damping is typically stronger than linear damping
    let angular_damping_multiplier = 1.0 - clamp(damping_factor * dt * 2.0, 0.0, 0.99);
    return angular_velocity * angular_damping_multiplier;
}

/// Clamp velocity to maximum limits for numerical stability.
///
/// Prevents extreme velocities that could cause numerical instability
/// or unrealistic cell behavior. Uses both magnitude and component limits.
///
/// # Arguments
/// * `velocity` - Input velocity vector
/// * `max_speed` - Maximum allowed speed magnitude
///
/// # Returns
/// Clamped velocity vector
fn clamp_velocity(velocity: vec3<f32>, max_speed: f32) -> vec3<f32> {
    let speed = length(velocity);
    
    // Clamp magnitude if too large
    if (speed > max_speed) {
        return normalize(velocity) * max_speed;
    }
    
    // Also clamp individual components to prevent extreme values
    let max_component = max_speed * 0.8; // Allow some headroom
    return vec3<f32>(
        clamp(velocity.x, -max_component, max_component),
        clamp(velocity.y, -max_component, max_component),
        clamp(velocity.z, -max_component, max_component)
    );
}

/// Clamp angular velocity to maximum limits.
///
/// Prevents extreme rotational velocities that could cause visual
/// artifacts or numerical instability in orientation calculations.
///
/// # Arguments
/// * `angular_velocity` - Input angular velocity vector
/// * `max_angular_speed` - Maximum allowed angular speed (rad/s)
///
/// # Returns
/// Clamped angular velocity vector
fn clamp_angular_velocity(angular_velocity: vec3<f32>, max_angular_speed: f32) -> vec3<f32> {
    let angular_speed = length(angular_velocity);
    
    // Clamp magnitude if too large
    if (angular_speed > max_angular_speed) {
        return normalize(angular_velocity) * max_angular_speed;
    }
    
    return angular_velocity;
}

/// Check if velocity is valid (no NaN or infinite values).
///
/// Validates velocity components to prevent numerical errors from
/// propagating through the simulation.
///
/// # Arguments
/// * `velocity` - Velocity to validate
///
/// # Returns
/// True if velocity is valid, false otherwise
fn is_valid_velocity(velocity: vec3<f32>) -> bool {
    let is_finite_x = isFinite(velocity.x) && !isNan(velocity.x);
    let is_finite_y = isFinite(velocity.y) && !isNan(velocity.y);
    let is_finite_z = isFinite(velocity.z) && !isNan(velocity.z);
    
    return is_finite_x && is_finite_y && is_finite_z;
}

/// Perform velocity Verlet integration for velocity update.
///
/// Uses the velocity Verlet method which provides better stability
/// and accuracy than basic Euler integration for physics simulation.
///
/// # Arguments
/// * `velocity` - Current velocity
/// * `acceleration` - Current acceleration
/// * `prev_acceleration` - Previous frame acceleration
/// * `dt` - Time step
///
/// # Returns
/// New velocity after integration
fn verlet_integrate_velocity(
    velocity: vec3<f32>,
    acceleration: vec3<f32>,
    prev_acceleration: vec3<f32>,
    dt: f32
) -> vec3<f32> {
    // Velocity Verlet integration:
    // v(t+dt) = v(t) + 0.5 * (a(t) + a(t-dt)) * dt
    // This uses both current and previous acceleration for better accuracy
    
    let average_acceleration = 0.5 * (acceleration + prev_acceleration);
    return velocity + average_acceleration * dt;
}

/// Handle boundary velocity reflection.
///
/// When a cell is near a boundary, modify its velocity to prevent
/// it from moving further into the boundary. This works in conjunction
/// with boundary forces to keep cells contained.
///
/// # Arguments
/// * `position` - Current cell position
/// * `velocity` - Current velocity
/// * `world_size` - Size of simulation world
/// * `boundary_margin` - Distance from boundary to start reflection
///
/// # Returns
/// Modified velocity that respects boundaries
fn handle_boundary_velocity(
    position: vec3<f32>,
    velocity: vec3<f32>,
    world_size: f32,
    boundary_margin: f32
) -> vec3<f32> {
    let half_world = world_size * 0.5;
    var modified_velocity = velocity;
    
    // X boundaries
    if (position.x > half_world - boundary_margin && velocity.x > 0.0) {
        modified_velocity.x = min(velocity.x, 0.0); // Don't move further right
    } else if (position.x < -half_world + boundary_margin && velocity.x < 0.0) {
        modified_velocity.x = max(velocity.x, 0.0); // Don't move further left
    }
    
    // Y boundaries
    if (position.y > half_world - boundary_margin && velocity.y > 0.0) {
        modified_velocity.y = min(velocity.y, 0.0); // Don't move further up
    } else if (position.y < -half_world + boundary_margin && velocity.y < 0.0) {
        modified_velocity.y = max(velocity.y, 0.0); // Don't move further down
    }
    
    // Z boundaries
    if (position.z > half_world - boundary_margin && velocity.z > 0.0) {
        modified_velocity.z = min(velocity.z, 0.0); // Don't move further forward
    } else if (position.z < -half_world + boundary_margin && velocity.z < 0.0) {
        modified_velocity.z = max(velocity.z, 0.0); // Don't move further back
    }
    
    return modified_velocity;
}

/// Main compute shader entry point.
///
/// Each thread processes one cell and updates its linear and angular
/// velocities based on accelerations. The shader applies damping,
/// velocity limits, and boundary handling for stable motion.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's velocity update
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Velocity Update Pipeline
/// 1. Read current velocities and accelerations
/// 2. Perform Verlet integration for new velocities
/// 3. Apply velocity damping for realistic motion
/// 4. Handle boundary velocity constraints
/// 5. Apply velocity limits for numerical stability
/// 6. Update angular velocities similarly
/// 7. Store updated velocities and previous accelerations
///
/// ## Numerical Stability
/// - Velocity validation prevents NaN/infinite propagation
/// - Velocity clamping limits extreme speeds
/// - Damping reduces oscillations and instability
/// - Boundary handling prevents escape from simulation
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond the active cell count
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Read current cell state
    let pos_and_mass = position_and_mass[cell_index];
    let position = pos_and_mass.xyz;
    let mass = pos_and_mass.w;
    let current_velocity = velocity[cell_index].xyz;
    let current_acceleration = acceleration[cell_index].xyz;
    let previous_acceleration = prev_acceleration[cell_index].xyz;
    let current_angular_velocity = angular_velocity[cell_index].xyz;
    let current_angular_acceleration = angular_acceleration[cell_index].xyz;
    let previous_angular_acceleration = prev_angular_acceleration[cell_index].xyz;
    
    // Skip velocity update for massless cells
    if (mass < 0.001) {
        return;
    }
    
    // Handle dragged cells - they have modified velocity behavior
    if (physics_params.dragged_cell_index == i32(cell_index)) {
        // Dragged cells have their velocity controlled by UI
        // Apply strong damping to make them more controllable
        let damped_velocity = apply_velocity_damping(current_velocity, 5.0, physics_params.delta_time);
        let damped_angular_velocity = apply_angular_damping(current_angular_velocity, 10.0, physics_params.delta_time);
        
        velocity[cell_index] = vec4<f32>(damped_velocity, 0.0);
        angular_velocity[cell_index] = vec4<f32>(damped_angular_velocity, 0.0);
        
        // Store current accelerations as previous for next frame
        prev_acceleration[cell_index] = acceleration[cell_index];
        prev_angular_acceleration[cell_index] = angular_acceleration[cell_index];
        return;
    }
    
    // Validate current velocity
    if (!is_valid_velocity(current_velocity)) {
        // Reset invalid velocities to zero
        velocity[cell_index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        angular_velocity[cell_index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        
        // Store current accelerations as previous for next frame
        prev_acceleration[cell_index] = acceleration[cell_index];
        prev_angular_acceleration[cell_index] = angular_acceleration[cell_index];
        return;
    }
    
    // === Linear Velocity Update ===
    
    // Perform Verlet integration for velocity
    let dt = physics_params.delta_time;
    var new_velocity = verlet_integrate_velocity(
        current_velocity,
        current_acceleration,
        previous_acceleration,
        dt
    );
    
    // Apply velocity damping
    let velocity_damping = 0.1; // Light damping for realistic motion
    new_velocity = apply_velocity_damping(new_velocity, velocity_damping, dt);
    
    // Handle boundary velocity constraints
    let boundary_margin = physics_params.world_size * 0.05; // 5% margin
    new_velocity = handle_boundary_velocity(position, new_velocity, physics_params.world_size, boundary_margin);
    
    // Apply velocity limits for numerical stability
    let max_speed = physics_params.world_size * 2.0; // Maximum 2 world units per second
    new_velocity = clamp_velocity(new_velocity, max_speed);
    
    // Final validation
    if (!is_valid_velocity(new_velocity)) {
        new_velocity = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // === Angular Velocity Update ===
    
    // Perform Verlet integration for angular velocity
    var new_angular_velocity = verlet_integrate_velocity(
        current_angular_velocity,
        current_angular_acceleration,
        previous_angular_acceleration,
        dt
    );
    
    // Apply angular damping (stronger than linear damping)
    let angular_damping = 0.2;
    new_angular_velocity = apply_angular_damping(new_angular_velocity, angular_damping, dt);
    
    // Apply angular velocity limits
    let max_angular_speed = 10.0; // Maximum 10 rad/s
    new_angular_velocity = clamp_angular_velocity(new_angular_velocity, max_angular_speed);
    
    // Final validation for angular velocity
    if (!is_valid_velocity(new_angular_velocity)) {
        new_angular_velocity = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // === Store Updated Values ===
    
    // Store updated velocities
    velocity[cell_index] = vec4<f32>(new_velocity, 0.0);
    angular_velocity[cell_index] = vec4<f32>(new_angular_velocity, 0.0);
    
    // Store current accelerations as previous for next frame
    // This is needed for the Verlet integration in the next time step
    prev_acceleration[cell_index] = acceleration[cell_index];
    prev_angular_acceleration[cell_index] = angular_acceleration[cell_index];
}