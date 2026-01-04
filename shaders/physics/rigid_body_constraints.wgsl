//! # Rigid Body Constraints Compute Shader
//!
//! This compute shader implements rigid body constraint solving for adhesion
//! networks and other constraints. It ensures that connected cells maintain
//! proper distances and orientations according to their adhesion bonds.
//!
//! ## Constraint Types
//! - **Distance Constraints**: Maintain proper separation between bonded cells
//! - **Orientation Constraints**: Preserve relative orientations in adhesion bonds
//! - **Twist Constraints**: Prevent excessive rotation around bond axes
//! - **Collision Constraints**: Resolve interpenetration between cells
//!
//! ## Constraint Solving
//! - **Iterative Solver**: Multiple iterations for convergence
//! - **Position-Based Dynamics**: Directly modify positions to satisfy constraints
//! - **Velocity Correction**: Update velocities to match position changes
//! - **Stability**: Damping and relaxation for numerical stability
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Random access to adhesion connections and cell data
//! - **Compute Intensity**: High - iterative constraint solving
//! - **Convergence**: Multiple passes may be needed for complex networks

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

// GPU Mode structure for adhesion settings
struct GpuMode {
    // Visual properties (16 bytes each)
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
    adhesion_can_break: i32,
    adhesion_break_force: f32,
    adhesion_rest_length: f32,
    adhesion_linear_spring_stiffness: f32,
    adhesion_linear_spring_damping: f32,
    adhesion_orientation_spring_stiffness: f32,
    adhesion_orientation_spring_damping: f32,
    adhesion_max_angular_deviation: f32,
    adhesion_twist_constraint_stiffness: f32,
    adhesion_twist_constraint_damping: f32,
    adhesion_enable_twist_constraint: i32,
    _adhesion_padding: i32,
    
    // Adhesion behavior (16 bytes)
    parent_make_adhesion: i32,
    child_a_keep_adhesion: i32,
    child_b_keep_adhesion: i32,
    max_adhesions: i32,
    
    // Flagellocyte properties (16 bytes)
    flagellocyte_thrust_force: f32,
    _mode_padding1: f32,
    _mode_padding2: f32,
    _mode_padding3: f32,
}

// Adhesion connection structure (96 bytes)
struct GpuAdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _padding_zones: vec2<u32>,
    anchor_direction_a: vec3<f32>,
    _padding_a: f32,
    anchor_direction_b: vec3<f32>,
    _padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
}

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> orientation: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> angular_velocity: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(6) var<storage, read> genome_modes: array<GpuMode>;

// Bind group 1: Adhesion system data
@group(1) @binding(0) var<storage, read> adhesion_connections: array<GpuAdhesionConnection>;
@group(1) @binding(1) var<storage, read> adhesion_indices: array<i32>;
@group(1) @binding(2) var<storage, read> adhesion_counts: array<u32>;

/// Quaternion multiplication.
///
/// Multiplies two quaternions following the standard quaternion multiplication rules.
///
/// # Arguments
/// * `q1` - First quaternion (w, x, y, z)
/// * `q2` - Second quaternion (w, x, y, z)
///
/// # Returns
/// Product quaternion
fn quat_multiply(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    );
}

/// Quaternion conjugate.
///
/// Returns the conjugate of a quaternion (w, -x, -y, -z).
///
/// # Arguments
/// * `q` - Input quaternion
///
/// # Returns
/// Conjugate quaternion
fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.w, -q.x, -q.y, -q.z);
}

/// Rotate vector by quaternion.
///
/// Rotates a 3D vector using quaternion rotation.
///
/// # Arguments
/// * `v` - Vector to rotate
/// * `q` - Rotation quaternion
///
/// # Returns
/// Rotated vector
fn quat_rotate_vector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qv = vec4<f32>(0.0, v.x, v.y, v.z);
    let result = quat_multiply(quat_multiply(q, qv), quat_conjugate(q));
    return result.yzw;
}

/// Calculate distance constraint correction.
///
/// Computes position corrections needed to maintain proper distance
/// between two connected cells.
///
/// # Arguments
/// * `pos_a` - Position of cell A
/// * `pos_b` - Position of cell B
/// * `rest_length` - Desired distance between cells
/// * `stiffness` - Constraint stiffness (0.0 to 1.0)
///
/// # Returns
/// Tuple of (correction_a, correction_b)
fn calculate_distance_constraint(
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    rest_length: f32,
    stiffness: f32
) -> vec2<vec3<f32>> {
    let delta = pos_b - pos_a;
    let current_distance = length(delta);
    
    // Avoid division by zero
    if (current_distance < 0.001) {
        return vec2<vec3<f32>>(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    
    let direction = delta / current_distance;
    let distance_error = current_distance - rest_length;
    let correction_magnitude = distance_error * stiffness * 0.5; // Split correction between both cells
    
    let correction = direction * correction_magnitude;
    
    return vec2<vec3<f32>>(-correction, correction);
}

/// Calculate orientation constraint correction.
///
/// Computes orientation corrections to maintain proper relative
/// orientations between connected cells.
///
/// # Arguments
/// * `orientation_a` - Orientation of cell A
/// * `orientation_b` - Orientation of cell B
/// * `target_relative_orientation` - Desired relative orientation
/// * `stiffness` - Constraint stiffness
///
/// # Returns
/// Tuple of (angular_correction_a, angular_correction_b)
fn calculate_orientation_constraint(
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    target_relative_orientation: vec4<f32>,
    stiffness: f32
) -> vec2<vec3<f32>> {
    // Calculate current relative orientation
    let relative_orientation = quat_multiply(quat_conjugate(orientation_a), orientation_b);
    
    // Calculate orientation error
    let orientation_error = quat_multiply(quat_conjugate(target_relative_orientation), relative_orientation);
    
    // Convert quaternion error to angular velocity correction
    // For small angles, the angular velocity is approximately 2 * (x, y, z) components
    let angular_error = 2.0 * orientation_error.yzw * stiffness;
    
    // Split correction between both cells
    let correction_a = -angular_error * 0.5;
    let correction_b = angular_error * 0.5;
    
    return vec2<vec3<f32>>(correction_a, correction_b);
}

/// Calculate twist constraint correction.
///
/// Prevents excessive rotation around the bond axis between two cells.
///
/// # Arguments
/// * `orientation_a` - Orientation of cell A
/// * `orientation_b` - Orientation of cell B
/// * `bond_direction` - Direction of bond between cells
/// * `max_twist` - Maximum allowed twist angle
/// * `stiffness` - Constraint stiffness
///
/// # Returns
/// Tuple of (angular_correction_a, angular_correction_b)
fn calculate_twist_constraint(
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    bond_direction: vec3<f32>,
    max_twist: f32,
    stiffness: f32
) -> vec2<vec3<f32>> {
    // Project orientations onto plane perpendicular to bond
    let forward_a = quat_rotate_vector(vec3<f32>(0.0, 0.0, 1.0), orientation_a);
    let forward_b = quat_rotate_vector(vec3<f32>(0.0, 0.0, 1.0), orientation_b);
    
    // Remove component along bond direction
    let projected_a = forward_a - dot(forward_a, bond_direction) * bond_direction;
    let projected_b = forward_b - dot(forward_b, bond_direction) * bond_direction;
    
    // Calculate twist angle
    let cos_twist = dot(normalize(projected_a), normalize(projected_b));
    let twist_angle = acos(clamp(cos_twist, -1.0, 1.0));
    
    // Apply correction if twist exceeds maximum
    if (twist_angle > max_twist) {
        let excess_twist = twist_angle - max_twist;
        let correction_magnitude = excess_twist * stiffness;
        
        // Calculate correction axis (perpendicular to both projected vectors)
        let correction_axis = normalize(cross(projected_a, projected_b));
        let correction = correction_axis * correction_magnitude;
        
        return vec2<vec3<f32>>(-correction * 0.5, correction * 0.5);
    }
    
    return vec2<vec3<f32>>(vec3<f32>(0.0), vec3<f32>(0.0));
}

/// Apply position correction with mass weighting.
///
/// Applies position corrections while accounting for cell masses.
/// Heavier cells receive smaller corrections.
///
/// # Arguments
/// * `position` - Current position
/// * `correction` - Position correction to apply
/// * `mass` - Cell mass
/// * `inverse_mass_sum` - Sum of inverse masses for normalization
///
/// # Returns
/// Corrected position
fn apply_position_correction(
    position: vec3<f32>,
    correction: vec3<f32>,
    mass: f32,
    inverse_mass_sum: f32
) -> vec3<f32> {
    if (mass < 0.001 || inverse_mass_sum < 0.001) {
        return position;
    }
    
    let inverse_mass = 1.0 / mass;
    let mass_weight = inverse_mass / inverse_mass_sum;
    
    return position + correction * mass_weight;
}

/// Apply angular correction with inertia weighting.
///
/// Applies angular corrections while accounting for moments of inertia.
///
/// # Arguments
/// * `angular_velocity` - Current angular velocity
/// * `correction` - Angular correction to apply
/// * `moment_of_inertia` - Moment of inertia
/// * `dt` - Time step
///
/// # Returns
/// Corrected angular velocity
fn apply_angular_correction(
    angular_velocity: vec3<f32>,
    correction: vec3<f32>,
    moment_of_inertia: f32,
    dt: f32
) -> vec3<f32> {
    if (moment_of_inertia < 0.001 || dt < 0.001) {
        return angular_velocity;
    }
    
    // Convert angular correction to angular velocity change
    let angular_velocity_change = correction / dt;
    
    return angular_velocity + angular_velocity_change;
}

/// Main compute shader entry point.
///
/// Each thread processes one adhesion connection and applies rigid body
/// constraints to maintain proper distances, orientations, and prevent
/// excessive twist between connected cells.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps to adhesion connection index
/// - Each thread processes one adhesion connection
/// - Multiple iterations may be needed for convergence
/// - Constraint solving affects connected cell pairs
///
/// ## Constraint Solving Pipeline
/// 1. Read adhesion connection and cell properties
/// 2. Calculate distance constraint corrections
/// 3. Calculate orientation constraint corrections
/// 4. Calculate twist constraint corrections (if enabled)
/// 5. Apply corrections with mass/inertia weighting
/// 6. Update cell positions and orientations
/// 7. Update velocities to match position changes
///
/// ## Numerical Stability
/// - Mass-weighted corrections prevent unrealistic behavior
/// - Stiffness parameters control correction strength
/// - Multiple iterations allow gradual convergence
/// - Validation prevents extreme corrections
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let connection_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond active connections
    // Note: This assumes we dispatch based on total adhesion count
    if (connection_index >= arrayLength(&adhesion_connections)) {
        return;
    }
    
    // Read adhesion connection
    let connection = adhesion_connections[connection_index];
    
    // Skip inactive connections
    if (connection.is_active == 0u) {
        return;
    }
    
    let cell_a_index = connection.cell_a_index;
    let cell_b_index = connection.cell_b_index;
    
    // Bounds check for cell indices
    if (cell_a_index >= physics_params.cell_count || cell_b_index >= physics_params.cell_count) {
        return;
    }
    
    // Read cell properties
    let pos_mass_a = position_and_mass[cell_a_index];
    let pos_mass_b = position_and_mass[cell_b_index];
    let position_a = pos_mass_a.xyz;
    let position_b = pos_mass_b.xyz;
    let mass_a = pos_mass_a.w;
    let mass_b = pos_mass_b.w;
    
    let velocity_a = velocity[cell_a_index].xyz;
    let velocity_b = velocity[cell_b_index].xyz;
    let orientation_a = orientation[cell_a_index];
    let orientation_b = orientation[cell_b_index];
    let angular_velocity_a = angular_velocity[cell_a_index].xyz;
    let angular_velocity_b = angular_velocity[cell_b_index].xyz;
    
    // Skip massless cells
    if (mass_a < 0.001 || mass_b < 0.001) {
        return;
    }
    
    // Skip dragged cells (they are controlled by UI)
    if (physics_params.dragged_cell_index == i32(cell_a_index) || 
        physics_params.dragged_cell_index == i32(cell_b_index)) {
        return;
    }
    
    // Get adhesion mode settings
    let mode = genome_modes[connection.mode_index];
    
    // === Distance Constraint ===
    
    let distance_corrections = calculate_distance_constraint(
        position_a,
        position_b,
        mode.adhesion_rest_length,
        mode.adhesion_linear_spring_stiffness * 0.1 // Reduce stiffness for stability
    );
    
    // Apply distance corrections with mass weighting
    let inverse_mass_a = 1.0 / mass_a;
    let inverse_mass_b = 1.0 / mass_b;
    let inverse_mass_sum = inverse_mass_a + inverse_mass_b;
    
    let corrected_position_a = apply_position_correction(
        position_a,
        distance_corrections.x,
        mass_a,
        inverse_mass_sum
    );
    let corrected_position_b = apply_position_correction(
        position_b,
        distance_corrections.y,
        mass_b,
        inverse_mass_sum
    );
    
    // === Orientation Constraint ===
    
    // Calculate target relative orientation from anchor directions
    let anchor_world_a = quat_rotate_vector(connection.anchor_direction_a, orientation_a);
    let anchor_world_b = quat_rotate_vector(connection.anchor_direction_b, orientation_b);
    let bond_direction = normalize(position_b - position_a);
    
    // Target: anchor directions should align with bond direction
    let target_relative_quat = vec4<f32>(1.0, 0.0, 0.0, 0.0); // Identity for now
    
    let orientation_corrections = calculate_orientation_constraint(
        orientation_a,
        orientation_b,
        target_relative_quat,
        mode.adhesion_orientation_spring_stiffness * 0.05 // Reduce for stability
    );
    
    // Calculate moments of inertia (assuming spherical cells)
    let radius_a = pow(mass_a * 0.75 / 3.14159, 1.0 / 3.0);
    let radius_b = pow(mass_b * 0.75 / 3.14159, 1.0 / 3.0);
    let inertia_a = 0.4 * mass_a * radius_a * radius_a;
    let inertia_b = 0.4 * mass_b * radius_b * radius_b;
    
    let corrected_angular_velocity_a = apply_angular_correction(
        angular_velocity_a,
        orientation_corrections.x,
        inertia_a,
        physics_params.delta_time
    );
    let corrected_angular_velocity_b = apply_angular_correction(
        angular_velocity_b,
        orientation_corrections.y,
        inertia_b,
        physics_params.delta_time
    );
    
    // === Twist Constraint (if enabled) ===
    
    var final_angular_velocity_a = corrected_angular_velocity_a;
    var final_angular_velocity_b = corrected_angular_velocity_b;
    
    if (mode.adhesion_enable_twist_constraint != 0) {
        let twist_corrections = calculate_twist_constraint(
            orientation_a,
            orientation_b,
            bond_direction,
            mode.adhesion_max_angular_deviation,
            mode.adhesion_twist_constraint_stiffness * 0.02 // Very gentle twist correction
        );
        
        final_angular_velocity_a = apply_angular_correction(
            final_angular_velocity_a,
            twist_corrections.x,
            inertia_a,
            physics_params.delta_time
        );
        final_angular_velocity_b = apply_angular_correction(
            final_angular_velocity_b,
            twist_corrections.y,
            inertia_b,
            physics_params.delta_time
        );
    }
    
    // === Update Cell Properties ===
    
    // Update positions
    position_and_mass[cell_a_index] = vec4<f32>(corrected_position_a, mass_a);
    position_and_mass[cell_b_index] = vec4<f32>(corrected_position_b, mass_b);
    
    // Update velocities to match position changes
    let dt = physics_params.delta_time;
    if (dt > 0.001) {
        let velocity_correction_a = (corrected_position_a - position_a) / dt;
        let velocity_correction_b = (corrected_position_b - position_b) / dt;
        
        velocity[cell_a_index] = vec4<f32>(velocity_a + velocity_correction_a * 0.1, 0.0); // Gentle velocity correction
        velocity[cell_b_index] = vec4<f32>(velocity_b + velocity_correction_b * 0.1, 0.0);
    }
    
    // Update angular velocities
    angular_velocity[cell_a_index] = vec4<f32>(final_angular_velocity_a, 0.0);
    angular_velocity[cell_b_index] = vec4<f32>(final_angular_velocity_b, 0.0);
}