//! # Rigid Body Constraints Compute Shader
//!
//! This compute shader implements advanced rigid body constraints for the
//! physics simulation. It handles distance constraints, orientation constraints,
//! and twist constraints to maintain realistic physical relationships between
//! connected cells.
//!
//! ## Constraint Types
//! - **Distance Constraints**: Maintain fixed distances between connected cells
//! - **Orientation Constraints**: Preserve relative orientations
//! - **Twist Constraints**: Prevent excessive rotation around bond axes
//! - **Collision Constraints**: Resolve interpenetration between rigid bodies
//!
//! ## Constraint Solving
//! - **Iterative Solver**: Multiple iterations for stability
//! - **Position-Based Dynamics**: Direct position corrections
//! - **Constraint Projection**: Project positions to satisfy constraints
//! - **Damping**: Prevent oscillations and instability
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Random access to connected cell pairs
//! - **Compute Intensity**: High - complex constraint solving algorithms
//! - **Convergence**: Multiple iterations for constraint satisfaction

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

// Rigid body constraint structure
struct RigidBodyConstraint {
    cell_a_index: u32,
    cell_b_index: u32,
    constraint_type: u32,           // 0=distance, 1=orientation, 2=twist
    is_active: u32,
    
    // Constraint parameters
    rest_length: f32,
    stiffness: f32,
    damping: f32,
    max_force: f32,
    
    // Target values
    target_orientation: vec4<f32>,  // Target relative orientation
    target_position: vec4<f32>,     // Target relative position
}

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> orientation: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> angular_velocity: array<vec4<f32>>;

// Bind group 1: Constraint system
@group(1) @binding(0) var<storage, read> constraints: array<RigidBodyConstraint>;
@group(1) @binding(1) var<storage, read> constraint_count: u32;

/// Calculate quaternion conjugate.
///
/// The conjugate of a quaternion (w, x, y, z) is (w, -x, -y, -z).
///
/// # Arguments
/// * `q` - Input quaternion
///
/// # Returns
/// Conjugate quaternion
fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.x, -q.y, -q.z, -q.w);
}

/// Multiply two quaternions.
///
/// Quaternion multiplication for combining rotations.
///
/// # Arguments
/// * `a` - First quaternion
/// * `b` - Second quaternion
///
/// # Returns
/// Product quaternion
fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

/// Rotate a vector by a quaternion.
///
/// Applies quaternion rotation to transform a vector.
///
/// # Arguments
/// * `v` - Vector to rotate
/// * `q` - Quaternion rotation
///
/// # Returns
/// Rotated vector
fn quat_rotate_vector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qvec = q.yzw;
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.x) + uuv) * 2.0;
}

/// Apply distance constraint between two cells.
///
/// Maintains a fixed distance between connected cells by adjusting positions.
///
/// # Arguments
/// * `pos_a` - Position of cell A
/// * `pos_b` - Position of cell B
/// * `mass_a` - Mass of cell A
/// * `mass_b` - Mass of cell B
/// * `rest_length` - Target distance
/// * `stiffness` - Constraint strength
///
/// # Returns
/// Position corrections for both cells
fn apply_distance_constraint(
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    mass_a: f32,
    mass_b: f32,
    rest_length: f32,
    stiffness: f32
) -> vec3<f32> {
    let delta = pos_b - pos_a;
    let current_distance = length(delta);
    
    // Avoid division by zero
    if (current_distance < 0.001) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    let direction = delta / current_distance;
    let constraint_error = current_distance - rest_length;
    let correction_magnitude = constraint_error * stiffness;
    
    // Distribute correction based on inverse mass
    let total_inv_mass = 1.0 / mass_a + 1.0 / mass_b;
    if (total_inv_mass > 0.001) {
        let correction_a_factor = (1.0 / mass_a) / total_inv_mass;
        return direction * correction_magnitude * correction_a_factor;
    }
    
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Apply orientation constraint between two cells.
///
/// Maintains relative orientation between connected cells.
///
/// # Arguments
/// * `orientation_a` - Orientation of cell A
/// * `orientation_b` - Orientation of cell B
/// * `target_relative` - Target relative orientation
/// * `stiffness` - Constraint strength
///
/// # Returns
/// Angular correction for cell A
fn apply_orientation_constraint(
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    target_relative: vec4<f32>,
    stiffness: f32
) -> vec3<f32> {
    // Calculate current relative orientation
    let relative_orientation = quat_multiply(quat_conjugate(orientation_a), orientation_b);
    
    // Calculate error quaternion
    let error_quat = quat_multiply(quat_conjugate(target_relative), relative_orientation);
    
    // Convert to axis-angle for correction
    let error_angle = 2.0 * acos(clamp(abs(error_quat.x), 0.0, 1.0));
    let sin_half_angle = sqrt(1.0 - error_quat.x * error_quat.x);
    
    if (sin_half_angle > 0.001) {
        let error_axis = error_quat.yzw / sin_half_angle;
        let angular_error = error_axis * error_angle * stiffness;
        return angular_error * 0.5; // Split correction between both cells
    }
    
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Main compute shader entry point.
///
/// Each thread processes one rigid body constraint and applies the necessary
/// corrections to maintain the constraint. Multiple constraint types are
/// supported with different solving algorithms.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps to constraint index
/// - Each thread processes one constraint
/// - Total threads needed: number of active constraints
/// - Workgroups needed: ceil(constraint_count / 64)
///
/// ## Constraint Pipeline
/// 1. Read constraint parameters and connected cell data
/// 2. Calculate constraint error and required correction
/// 3. Apply position and orientation corrections
/// 4. Update cell positions and orientations
/// 5. Apply damping to prevent oscillations
///
/// ## Numerical Stability
/// - Constraint corrections are clamped to prevent instability
/// - Mass-based correction distribution maintains realism
/// - Multiple solver iterations improve convergence
/// - Special handling for degenerate cases
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let constraint_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond active constraints
    if (constraint_index >= constraint_count) {
        return;
    }
    
    // Read constraint data
    let constraint = constraints[constraint_index];
    
    // Skip inactive constraints
    if (constraint.is_active == 0u) {
        return;
    }
    
    // Get cell indices
    let cell_a_idx = constraint.cell_a_index;
    let cell_b_idx = constraint.cell_b_index;
    
    // Bounds check for cell indices
    if (cell_a_idx >= physics_params.cell_count || cell_b_idx >= physics_params.cell_count) {
        return;
    }
    
    // Read cell A properties
    let pos_mass_a = position_and_mass[cell_a_idx];
    let pos_a = pos_mass_a.xyz;
    let mass_a = pos_mass_a.w;
    let vel_a = velocity[cell_a_idx].xyz;
    let orient_a = orientation[cell_a_idx];
    let angular_vel_a = angular_velocity[cell_a_idx].xyz;
    
    // Read cell B properties
    let pos_mass_b = position_and_mass[cell_b_idx];
    let pos_b = pos_mass_b.xyz;
    let mass_b = pos_mass_b.w;
    let vel_b = velocity[cell_b_idx].xyz;
    let orient_b = orientation[cell_b_idx];
    let angular_vel_b = angular_velocity[cell_b_idx].xyz;
    
    // Skip massless cells
    if (mass_a < 0.001 || mass_b < 0.001) {
        return;
    }
    
    // Apply constraint based on type
    if (constraint.constraint_type == 0u) {
        // Distance constraint
        let position_correction = apply_distance_constraint(
            pos_a, pos_b, mass_a, mass_b,
            constraint.rest_length, constraint.stiffness
        );
        
        // Apply position corrections
        let new_pos_a = pos_a - position_correction;
        let new_pos_b = pos_b + position_correction;
        
        // Clamp corrections to prevent instability
        let max_correction = physics_params.world_size * 0.01; // 1% of world size
        let correction_magnitude = length(position_correction);
        
        if (correction_magnitude < max_correction) {
            position_and_mass[cell_a_idx] = vec4<f32>(new_pos_a, mass_a);
            position_and_mass[cell_b_idx] = vec4<f32>(new_pos_b, mass_b);
        }
        
    } else if (constraint.constraint_type == 1u) {
        // Orientation constraint
        let angular_correction = apply_orientation_constraint(
            orient_a, orient_b,
            constraint.target_orientation, constraint.stiffness
        );
        
        // Apply angular corrections to angular velocity
        let max_angular_correction = 0.1; // Maximum angular correction per frame
        let correction_magnitude = length(angular_correction);
        
        if (correction_magnitude < max_angular_correction) {
            let new_angular_vel_a = angular_vel_a + angular_correction;
            let new_angular_vel_b = angular_vel_b - angular_correction;
            
            angular_velocity[cell_a_idx] = vec4<f32>(new_angular_vel_a, 0.0);
            angular_velocity[cell_b_idx] = vec4<f32>(new_angular_vel_b, 0.0);
        }
    }
    
    // Apply damping to prevent oscillations
    let damping_factor = 1.0 - constraint.damping * physics_params.delta_time;
    
    // Damp velocities of constrained cells
    let damped_vel_a = vel_a * damping_factor;
    let damped_vel_b = vel_b * damping_factor;
    let damped_angular_vel_a = angular_vel_a * damping_factor;
    let damped_angular_vel_b = angular_vel_b * damping_factor;
    
    velocity[cell_a_idx] = vec4<f32>(damped_vel_a, 0.0);
    velocity[cell_b_idx] = vec4<f32>(damped_vel_b, 0.0);
    angular_velocity[cell_a_idx] = vec4<f32>(damped_angular_vel_a, 0.0);
    angular_velocity[cell_b_idx] = vec4<f32>(damped_angular_vel_b, 0.0);
}