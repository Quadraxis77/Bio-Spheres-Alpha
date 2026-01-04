//! # Adhesion Physics Compute Shader
//!
//! This compute shader implements complete adhesion bond mechanics including
//! spring-damper forces, orientation constraints, twist constraints, and
//! bond breaking conditions. It processes all active adhesion connections
//! and applies forces to connected cells.
//!
//! ## Adhesion Mechanics
//! - **Linear Spring Forces**: Distance-based attraction/repulsion
//! - **Linear Damping**: Velocity-dependent damping for stability
//! - **Orientation Springs**: Rotational constraints between cells
//! - **Twist Constraints**: Prevents rotation around bond axis
//! - **Bond Breaking**: Removes bonds when forces exceed thresholds
//!
//! ## Force Calculation
//! - Reads adhesion connections and cell properties
//! - Calculates spring forces based on rest length and stiffness
//! - Applies orientation and twist constraints
//! - Accumulates forces and torques for both connected cells
//! - Handles bond breaking when force limits are exceeded
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Random access to cell arrays via adhesion indices
//! - **Compute Intensity**: High - complex vector and quaternion math
//! - **Cache Efficiency**: Moderate due to indirect memory access patterns

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
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    
    // UI interaction
    dragged_cell_index: i32,
    _padding1: vec3<f32>,
    
    // Padding to 256 bytes (using vec4<f32> for proper 16-byte alignment)
    _padding: array<vec4<f32>, 12>,
}

// GPU Mode adhesion settings structure (48 bytes)
struct GpuModeAdhesionSettings {
    can_break: i32,                         // 0 = false, 1 = true
    break_force: f32,                       // Force threshold for breaking
    rest_length: f32,                       // Natural bond length
    linear_spring_stiffness: f32,           // Spring constant for distance
    linear_spring_damping: f32,             // Damping coefficient
    orientation_spring_stiffness: f32,      // Rotational constraint stiffness
    orientation_spring_damping: f32,        // Rotational damping
    max_angular_deviation: f32,             // Maximum allowed angle deviation
    twist_constraint_stiffness: f32,        // Twist resistance stiffness
    twist_constraint_damping: f32,          // Twist damping
    enable_twist_constraint: i32,           // 0 = false, 1 = true
    _padding: i32,                          // Pad to 48 bytes
}

// GPU Mode structure for genome-based behavior
struct GpuMode {
    // Visual properties (64 bytes)
    color: vec4<f32>,
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    split_direction: vec4<f32>,
    
    // Child modes (8 bytes)
    child_modes: vec2<i32>,
    
    // Division properties (8 bytes)
    split_interval: f32,
    genome_offset: i32,
    
    // Adhesion settings (48 bytes)
    adhesion_settings: GpuModeAdhesionSettings,
    
    // Adhesion behavior (16 bytes)
    parent_make_adhesion: i32,
    child_a_keep_adhesion: i32,
    child_b_keep_adhesion: i32,
    max_adhesions: i32,
}

// Adhesion connection structure (96 bytes total)
struct GpuAdhesionConnection {
    cell_a_index: u32,                      // Index of first cell
    cell_b_index: u32,                      // Index of second cell
    mode_index: u32,                        // Mode index for settings lookup
    is_active: u32,                         // 1 = active, 0 = inactive
    
    zone_a: u32,                            // Zone classification for cell A
    zone_b: u32,                            // Zone classification for cell B
    _padding_zones: vec2<u32>,              // Padding for alignment
    
    anchor_direction_a: vec3<f32>,          // Anchor direction for cell A (local space)
    _padding_a: f32,                        // Padding for 16-byte alignment
    
    anchor_direction_b: vec3<f32>,          // Anchor direction for cell B (local space)
    _padding_b: f32,                        // Padding for 16-byte alignment
    
    twist_reference_a: vec4<f32>,           // Reference quaternion for cell A twist
    twist_reference_b: vec4<f32>,           // Reference quaternion for cell B twist
}

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acceleration: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> orientation: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> genome_orientation: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> angular_velocity: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> angular_acceleration: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(9) var<storage, read> genome_modes: array<GpuMode>;

// Bind group 1: Adhesion system
@group(1) @binding(0) var<storage, read_write> adhesion_connections: array<GpuAdhesionConnection>;
@group(1) @binding(1) var<storage, read> adhesion_indices: array<i32>; // 10 per cell
@group(1) @binding(2) var<storage, read> adhesion_counts: array<u32>;

/// Rotate a vector by a quaternion.
///
/// Applies quaternion rotation to transform a vector from one coordinate
/// system to another. Used for converting local anchor directions to world space.
///
/// # Arguments
/// * `v` - Vector to rotate
/// * `q` - Quaternion rotation (w, x, y, z)
///
/// # Returns
/// Rotated vector in new coordinate system
fn rotate_vector_by_quaternion(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    // Quaternion format: (w, x, y, z) where w is scalar part
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    
    // Quaternion rotation formula: v' = q * v * q^(-1)
    // Optimized version using cross products
    let qvec = vec3<f32>(x, y, z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    
    return v + ((uv * w) + uuv) * 2.0;
}

/// Calculate quaternion difference for orientation constraints.
///
/// Computes the relative rotation between two quaternions, which represents
/// how much one orientation differs from another. Used for orientation springs.
///
/// # Arguments
/// * `q1` - First quaternion
/// * `q2` - Second quaternion
///
/// # Returns
/// Quaternion representing the rotation from q1 to q2
fn quaternion_difference(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    // Calculate q2 * conjugate(q1)
    // conjugate(q1) = (w, -x, -y, -z)
    let q1_conj = vec4<f32>(q1.x, -q1.y, -q1.z, -q1.w);
    
    // Quaternion multiplication: q2 * q1_conj
    let w = q2.x * q1_conj.x - q2.y * q1_conj.y - q2.z * q1_conj.z - q2.w * q1_conj.w;
    let x = q2.x * q1_conj.y + q2.y * q1_conj.x + q2.z * q1_conj.w - q2.w * q1_conj.z;
    let y = q2.x * q1_conj.z - q2.y * q1_conj.w + q2.z * q1_conj.x + q2.w * q1_conj.y;
    let z = q2.x * q1_conj.w + q2.y * q1_conj.z - q2.z * q1_conj.y + q2.w * q1_conj.x;
    
    return vec4<f32>(w, x, y, z);
}

/// Convert quaternion to axis-angle representation.
///
/// Extracts the rotation axis and angle from a quaternion. Used for
/// calculating torques from orientation differences.
///
/// # Arguments
/// * `q` - Input quaternion (w, x, y, z)
///
/// # Returns
/// Tuple of (axis, angle) where axis is normalized and angle is in radians
fn quaternion_to_axis_angle(q: vec4<f32>) -> vec4<f32> {
    // Normalize quaternion
    let quat = normalize(q);
    let w = quat.x;
    let xyz = quat.yzw;
    
    // Calculate angle
    let angle = 2.0 * acos(clamp(abs(w), 0.0, 1.0));
    
    // Calculate axis
    let sin_half_angle = sqrt(1.0 - w * w);
    var axis = vec3<f32>(0.0, 1.0, 0.0); // Default axis
    
    if (sin_half_angle > 0.001) {
        axis = xyz / sin_half_angle;
    }
    
    return vec4<f32>(axis, angle);
}

/// Calculate linear spring force between two cells.
///
/// Implements Hooke's law with damping for adhesion bonds. The force
/// attracts cells toward the rest length and damps relative motion.
///
/// # Arguments
/// * `pos_a` - Position of cell A
/// * `pos_b` - Position of cell B
/// * `vel_a` - Velocity of cell A
/// * `vel_b` - Velocity of cell B
/// * `settings` - Adhesion settings from genome mode
///
/// # Returns
/// Force vector to apply to cell A (apply negative to cell B)
fn calculate_linear_spring_force(
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    vel_a: vec3<f32>,
    vel_b: vec3<f32>,
    settings: GpuModeAdhesionSettings
) -> vec3<f32> {
    // Calculate bond vector and current length
    let bond_vector = pos_b - pos_a;
    let current_length = length(bond_vector);
    
    // Avoid division by zero
    if (current_length < 0.001) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    let bond_direction = bond_vector / current_length;
    
    // Calculate spring force (Hooke's law)
    let length_difference = current_length - settings.rest_length;
    let spring_force_magnitude = settings.linear_spring_stiffness * length_difference;
    let spring_force = bond_direction * spring_force_magnitude;
    
    // Calculate damping force
    let relative_velocity = vel_b - vel_a;
    let relative_velocity_along_bond = dot(relative_velocity, bond_direction);
    let damping_force = bond_direction * (settings.linear_spring_damping * relative_velocity_along_bond);
    
    // Total force (spring + damping)
    return spring_force + damping_force;
}

/// Calculate orientation spring torque between two cells.
///
/// Applies rotational constraints to maintain relative orientations between
/// connected cells. This creates realistic adhesion behavior where cells
/// try to maintain their relative orientations.
///
/// # Arguments
/// * `orient_a` - Orientation of cell A
/// * `orient_b` - Orientation of cell B
/// * `anchor_a` - Anchor direction for cell A (local space)
/// * `anchor_b` - Anchor direction for cell B (local space)
/// * `angular_vel_a` - Angular velocity of cell A
/// * `angular_vel_b` - Angular velocity of cell B
/// * `settings` - Adhesion settings from genome mode
///
/// # Returns
/// Torque vector to apply to cell A (apply negative to cell B)
fn calculate_orientation_spring_torque(
    orient_a: vec4<f32>,
    orient_b: vec4<f32>,
    anchor_a: vec3<f32>,
    anchor_b: vec3<f32>,
    angular_vel_a: vec3<f32>,
    angular_vel_b: vec3<f32>,
    settings: GpuModeAdhesionSettings
) -> vec3<f32> {
    // Transform anchor directions to world space
    let world_anchor_a = rotate_vector_by_quaternion(anchor_a, orient_a);
    let world_anchor_b = rotate_vector_by_quaternion(anchor_b, orient_b);
    
    // Calculate desired relative orientation
    // Anchors should point toward each other for proper alignment
    let desired_direction_a = -world_anchor_b;
    let current_direction_a = world_anchor_a;
    
    // Calculate rotation axis and angle for correction
    let correction_axis = cross(current_direction_a, desired_direction_a);
    let correction_angle = acos(clamp(dot(current_direction_a, desired_direction_a), -1.0, 1.0));
    
    // Check if correction is needed
    if (correction_angle < 0.001 || length(correction_axis) < 0.001) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Normalize correction axis
    let normalized_correction_axis = normalize(correction_axis);
    
    // Apply maximum angular deviation limit
    let limited_angle = min(correction_angle, settings.max_angular_deviation);
    
    // Calculate spring torque
    let spring_torque_magnitude = settings.orientation_spring_stiffness * limited_angle;
    let spring_torque = normalized_correction_axis * spring_torque_magnitude;
    
    // Calculate damping torque
    let relative_angular_velocity = angular_vel_b - angular_vel_a;
    let damping_torque = relative_angular_velocity * settings.orientation_spring_damping;
    
    // Total torque (spring + damping)
    return spring_torque - damping_torque;
}

/// Calculate twist constraint torque to prevent rotation around bond axis.
///
/// Prevents cells from rotating around the bond axis by comparing current
/// orientations with reference orientations stored when the bond was created.
///
/// # Arguments
/// * `orient_a` - Current orientation of cell A
/// * `orient_b` - Current orientation of cell B
/// * `twist_ref_a` - Reference orientation for cell A (from bond creation)
/// * `twist_ref_b` - Reference orientation for cell B (from bond creation)
/// * `bond_direction` - Direction of the bond (normalized)
/// * `angular_vel_a` - Angular velocity of cell A
/// * `angular_vel_b` - Angular velocity of cell B
/// * `settings` - Adhesion settings from genome mode
///
/// # Returns
/// Torque vector to apply to cell A (apply negative to cell B)
fn calculate_twist_constraint_torque(
    orient_a: vec4<f32>,
    orient_b: vec4<f32>,
    twist_ref_a: vec4<f32>,
    twist_ref_b: vec4<f32>,
    bond_direction: vec3<f32>,
    angular_vel_a: vec3<f32>,
    angular_vel_b: vec3<f32>,
    settings: GpuModeAdhesionSettings
) -> vec3<f32> {
    // Skip if twist constraint is disabled
    if (settings.enable_twist_constraint == 0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Calculate current relative orientation
    let current_relative = quaternion_difference(orient_a, orient_b);
    
    // Calculate reference relative orientation
    let reference_relative = quaternion_difference(twist_ref_a, twist_ref_b);
    
    // Calculate twist difference
    let twist_difference = quaternion_difference(reference_relative, current_relative);
    
    // Convert to axis-angle representation
    let axis_angle = quaternion_to_axis_angle(twist_difference);
    let twist_axis = axis_angle.xyz;
    let twist_angle = axis_angle.w;
    
    // Project twist axis onto bond direction to isolate twist component
    let twist_component = dot(twist_axis, bond_direction);
    let twist_torque_axis = bond_direction * twist_component;
    
    // Calculate spring torque for twist constraint
    let spring_torque_magnitude = settings.twist_constraint_stiffness * twist_angle;
    let spring_torque = twist_torque_axis * spring_torque_magnitude;
    
    // Calculate damping torque for twist
    let relative_angular_velocity = angular_vel_b - angular_vel_a;
    let twist_velocity_component = dot(relative_angular_velocity, bond_direction);
    let damping_torque = bond_direction * (settings.twist_constraint_damping * twist_velocity_component);
    
    // Total twist torque (spring + damping)
    return spring_torque - damping_torque;
}

/// Check if adhesion bond should break due to excessive force.
///
/// Evaluates whether the current force on an adhesion bond exceeds the
/// breaking threshold specified in the genome mode settings.
///
/// # Arguments
/// * `force_magnitude` - Current force magnitude on the bond
/// * `settings` - Adhesion settings from genome mode
///
/// # Returns
/// True if bond should break, false otherwise
fn should_break_bond(force_magnitude: f32, settings: GpuModeAdhesionSettings) -> bool {
    return settings.can_break != 0 && force_magnitude > settings.break_force;
}

/// Main compute shader entry point for adhesion physics.
///
/// Each thread processes one adhesion connection and calculates the forces
/// and torques between the connected cells. The forces are accumulated
/// into the acceleration buffers for both cells.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps to adhesion connection index
/// - Each thread processes one adhesion connection
/// - Total threads needed: number of active adhesion connections
/// - Workgroups needed: ceil(active_connections / 64)
///
/// ## Force Processing Pipeline
/// 1. Read adhesion connection and validate it's active
/// 2. Read properties of both connected cells
/// 3. Calculate linear spring forces (attraction/repulsion)
/// 4. Calculate orientation spring torques (rotational constraints)
/// 5. Calculate twist constraint torques (prevent axis rotation)
/// 6. Check for bond breaking conditions
/// 7. Accumulate forces and torques into cell acceleration buffers
/// 8. Mark bonds as inactive if they should break
///
/// ## Numerical Stability
/// - Force and torque clamping prevents extreme values
/// - Damping terms reduce oscillations
/// - Special handling for edge cases (zero distance, parallel orientations)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let connection_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond active connections
    // Note: This assumes the shader is dispatched with the correct number of workgroups
    // based on the actual number of active adhesion connections
    if (connection_index >= arrayLength(&adhesion_connections)) {
        return;
    }
    
    // Read adhesion connection
    let connection = adhesion_connections[connection_index];
    
    // Skip inactive connections
    if (connection.is_active == 0u) {
        return;
    }
    
    // Get cell indices
    let cell_a_idx = connection.cell_a_index;
    let cell_b_idx = connection.cell_b_index;
    
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
    let mode_idx_a = mode_indices[cell_a_idx];
    
    // Read cell B properties
    let pos_mass_b = position_and_mass[cell_b_idx];
    let pos_b = pos_mass_b.xyz;
    let mass_b = pos_mass_b.w;
    let vel_b = velocity[cell_b_idx].xyz;
    let orient_b = orientation[cell_b_idx];
    let angular_vel_b = angular_velocity[cell_b_idx].xyz;
    let mode_idx_b = mode_indices[cell_b_idx];
    
    // Get adhesion settings from genome mode
    let mode = genome_modes[connection.mode_index];
    let settings = mode.adhesion_settings;
    
    // === Calculate Linear Spring Forces ===
    let linear_force = calculate_linear_spring_force(
        pos_a, pos_b, vel_a, vel_b, settings
    );
    
    // === Calculate Orientation Spring Torques ===
    let orientation_torque = calculate_orientation_spring_torque(
        orient_a, orient_b,
        connection.anchor_direction_a, connection.anchor_direction_b,
        angular_vel_a, angular_vel_b,
        settings
    );
    
    // === Calculate Twist Constraint Torques ===
    let bond_vector = pos_b - pos_a;
    let bond_length = length(bond_vector);
    var twist_torque = vec3<f32>(0.0, 0.0, 0.0);
    
    if (bond_length > 0.001) {
        let bond_direction = bond_vector / bond_length;
        twist_torque = calculate_twist_constraint_torque(
            orient_a, orient_b,
            connection.twist_reference_a, connection.twist_reference_b,
            bond_direction,
            angular_vel_a, angular_vel_b,
            settings
        );
    }
    
    // === Force and Torque Clamping ===
    let max_force = 1000.0; // Maximum force per adhesion
    let max_torque = 500.0;  // Maximum torque per adhesion
    
    let clamped_linear_force = clamp(linear_force, vec3<f32>(-max_force), vec3<f32>(max_force));
    let clamped_orientation_torque = clamp(orientation_torque, vec3<f32>(-max_torque), vec3<f32>(max_torque));
    let clamped_twist_torque = clamp(twist_torque, vec3<f32>(-max_torque), vec3<f32>(max_torque));
    
    // === Check Bond Breaking ===
    let total_force_magnitude = length(clamped_linear_force);
    if (should_break_bond(total_force_magnitude, settings)) {
        // Mark connection as inactive (break the bond)
        adhesion_connections[connection_index].is_active = 0u;
        return; // Don't apply forces if bond is broken
    }
    
    // === Apply Forces to Cell A ===
    if (mass_a > 0.001) {
        let acceleration_a = clamped_linear_force / mass_a;
        
        // Atomic add to acceleration buffer (approximated with regular add for now)
        // Note: In a real implementation, this would need atomic operations
        // or careful synchronization to handle multiple connections per cell
        let current_accel_a = acceleration[cell_a_idx].xyz;
        acceleration[cell_a_idx] = vec4<f32>(current_accel_a + acceleration_a, 0.0);
        
        // Apply torques (assuming unit moment of inertia for simplicity)
        let total_torque_a = clamped_orientation_torque + clamped_twist_torque;
        let current_angular_accel_a = angular_acceleration[cell_a_idx].xyz;
        angular_acceleration[cell_a_idx] = vec4<f32>(current_angular_accel_a + total_torque_a, 0.0);
    }
    
    // === Apply Forces to Cell B (Newton's third law) ===
    if (mass_b > 0.001) {
        let acceleration_b = -clamped_linear_force / mass_b;
        
        // Atomic add to acceleration buffer
        let current_accel_b = acceleration[cell_b_idx].xyz;
        acceleration[cell_b_idx] = vec4<f32>(current_accel_b + acceleration_b, 0.0);
        
        // Apply opposite torques
        let total_torque_b = -(clamped_orientation_torque + clamped_twist_torque);
        let current_angular_accel_b = angular_acceleration[cell_b_idx].xyz;
        angular_acceleration[cell_b_idx] = vec4<f32>(current_angular_accel_b + total_torque_b, 0.0);
    }
}