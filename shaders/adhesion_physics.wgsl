// Adhesion Physics Compute Shader
// Applies spring-damper forces between adhered cells
// Workgroup size: 256 threads for adhesion operations
//
// Physics model (matching reference):
// 1. Linear spring force: F = k * (distance - rest_length) * direction
// 2. Linear damping: F -= damping * relative_velocity
// 3. Orientation spring torque: aligns anchor directions with adhesion axis
// 4. Twist constraint: prevents rotation around adhesion axis
//
// Deterministic execution:
// - Each thread processes one adhesion connection
// - Forces are accumulated atomically to cell accumulators
// - No race conditions due to atomic operations

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

// Adhesion settings (48 bytes, matching reference GPUModeAdhesionSettings)
struct AdhesionSettings {
    can_break: i32,
    break_force: f32,
    rest_length: f32,
    linear_spring_stiffness: f32,
    linear_spring_damping: f32,
    orientation_spring_stiffness: f32,
    orientation_spring_damping: f32,
    max_angular_deviation: f32,
    twist_constraint_stiffness: f32,
    twist_constraint_damping: f32,
    enable_twist_constraint: i32,
    _padding: i32,
}

// Adhesion connection structure (96 bytes)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    anchor_direction_a: vec3<f32>,
    padding_a: f32,
    anchor_direction_b: vec3<f32>,
    padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    _padding: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Adhesion bind group (group 1)
@group(1) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(1)
var<storage, read> adhesion_settings: array<AdhesionSettings>;

@group(1) @binding(2)
var<storage, read_write> adhesion_counts: array<u32>;

// Rotations bind group (group 2)
@group(2) @binding(0)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(2) @binding(1)
var<storage, read> angular_velocities: array<vec4<f32>>;

// Force accumulation buffers (group 3) - using atomic for multi-adhesion accumulation
@group(3) @binding(0)
var<storage, read_write> force_accum_x: array<atomic<i32>>;

@group(3) @binding(1)
var<storage, read_write> force_accum_y: array<atomic<i32>>;

@group(3) @binding(2)
var<storage, read_write> force_accum_z: array<atomic<i32>>;

@group(3) @binding(3)
var<storage, read_write> torque_accum_x: array<atomic<i32>>;

@group(3) @binding(4)
var<storage, read_write> torque_accum_y: array<atomic<i32>>;

@group(3) @binding(5)
var<storage, read_write> torque_accum_z: array<atomic<i32>>;

const PI: f32 = 3.14159265359;

// Calculate radius from mass (assuming density = 1)
fn calculate_radius_from_mass(mass: f32) -> f32 {
    let volume = mass / 1.0;
    return pow(volume * 3.0 / (4.0 * PI), 1.0 / 3.0);
}

// Quaternion multiplication
fn quat_multiply(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Quaternion conjugate
fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// Quaternion inverse
fn quat_inverse(q: vec4<f32>) -> vec4<f32> {
    let norm = dot(q, q);
    if (norm > 0.0) {
        return quat_conjugate(q) / norm;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Rotate vector by quaternion
fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

// Convert quaternion to axis-angle
fn quat_to_axis_angle(q: vec4<f32>) -> vec4<f32> {
    let angle = 2.0 * acos(clamp(q.w, -1.0, 1.0));
    var axis: vec3<f32>;
    if (angle < 0.001) {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        axis = normalize(q.xyz / sin(angle * 0.5));
    }
    return vec4<f32>(axis, angle);
}

// Deterministic quaternion from two vectors
fn quat_from_two_vectors(from_vec: vec3<f32>, to: vec3<f32>) -> vec4<f32> {
    let v1 = normalize(from_vec);
    let v2 = normalize(to);
    
    let cos_angle = dot(v1, v2);
    
    // Vectors are already aligned
    if (cos_angle > 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    // Vectors are opposite - use deterministic perpendicular axis
    if (cos_angle < -0.9999) {
        var axis: vec3<f32>;
        if (abs(v1.x) < abs(v1.y) && abs(v1.x) < abs(v1.z)) {
            axis = normalize(vec3<f32>(0.0, -v1.z, v1.y));
        } else if (abs(v1.y) < abs(v1.z)) {
            axis = normalize(vec3<f32>(-v1.z, 0.0, v1.x));
        } else {
            axis = normalize(vec3<f32>(-v1.y, v1.x, 0.0));
        }
        return vec4<f32>(axis, 0.0);
    }
    
    // General case: half-way quaternion method
    let halfway = normalize(v1 + v2);
    let axis = vec3<f32>(
        v1.y * halfway.z - v1.z * halfway.y,
        v1.z * halfway.x - v1.x * halfway.z,
        v1.x * halfway.y - v1.y * halfway.x
    );
    let w = dot(v1, halfway);
    
    return normalize(vec4<f32>(axis, w));
}

// Compute adhesion forces between two cells
fn compute_adhesion_forces(
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    vel_a: vec3<f32>,
    vel_b: vec3<f32>,
    mass_a: f32,
    mass_b: f32,
    rot_a: vec4<f32>,
    rot_b: vec4<f32>,
    ang_vel_a: vec3<f32>,
    ang_vel_b: vec3<f32>,
    connection: AdhesionConnection,
    settings: AdhesionSettings,
) -> array<vec4<f32>, 4> {
    // Output: [force_a, torque_a, force_b, torque_b]
    var force_a = vec3<f32>(0.0);
    var torque_a = vec3<f32>(0.0);
    var force_b = vec3<f32>(0.0);
    var torque_b = vec3<f32>(0.0);
    
    // Connection vector from A to B
    let delta_pos = pos_b - pos_a;
    let dist = length(delta_pos);
    if (dist < 0.0001) {
        return array<vec4<f32>, 4>(
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
        );
    }
    
    let adhesion_dir = delta_pos / dist;
    let rest_length = settings.rest_length;
    
    // Linear spring force
    let force_mag = settings.linear_spring_stiffness * (dist - rest_length);
    let spring_force = adhesion_dir * force_mag;
    
    // Linear damping - oppose relative motion
    let rel_vel = vel_b - vel_a;
    let damp_mag = 1.0 - settings.linear_spring_damping * dot(rel_vel, adhesion_dir);
    let damping_force = -adhesion_dir * damp_mag;
    
    force_a += spring_force + damping_force;
    force_b -= spring_force + damping_force;
    
    // Transform anchor directions to world space
    var anchor_a: vec3<f32>;
    var anchor_b: vec3<f32>;
    
    if (length(connection.anchor_direction_a) < 0.001 && length(connection.anchor_direction_b) < 0.001) {
        // Fallback: use default directions
        anchor_a = vec3<f32>(1.0, 0.0, 0.0);
        anchor_b = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        anchor_a = rotate_vector_by_quat(connection.anchor_direction_a, rot_a);
        anchor_b = rotate_vector_by_quat(connection.anchor_direction_b, rot_b);
    }
    
    // Orientation spring for cell A
    let axis_a = cross(anchor_a, adhesion_dir);
    let sin_a = length(axis_a);
    let cos_a = dot(anchor_a, adhesion_dir);
    let angle_a = atan2(sin_a, cos_a);
    if (sin_a > 0.0001) {
        let norm_axis_a = normalize(axis_a);
        let spring_torque_a = norm_axis_a * angle_a * settings.orientation_spring_stiffness;
        let damping_torque_a = -norm_axis_a * dot(ang_vel_a, norm_axis_a) * settings.orientation_spring_damping;
        torque_a += spring_torque_a + damping_torque_a;
    }
    
    // Orientation spring for cell B
    let axis_b = cross(anchor_b, -adhesion_dir);
    let sin_b = length(axis_b);
    let cos_b = dot(anchor_b, -adhesion_dir);
    let angle_b = atan2(sin_b, cos_b);
    if (sin_b > 0.0001) {
        let norm_axis_b = normalize(axis_b);
        let spring_torque_b = norm_axis_b * angle_b * settings.orientation_spring_stiffness;
        let damping_torque_b = -norm_axis_b * dot(ang_vel_b, norm_axis_b) * settings.orientation_spring_damping;
        torque_b += spring_torque_b + damping_torque_b;
    }
    
    // Twist constraints
    if (settings.enable_twist_constraint != 0 &&
        length(connection.twist_reference_a) > 0.001 &&
        length(connection.twist_reference_b) > 0.001) {
        
        let adhesion_axis = normalize(delta_pos);
        
        // Calculate target orientations
        let current_anchor_a = rotate_vector_by_quat(connection.anchor_direction_a, rot_a);
        let current_anchor_b = rotate_vector_by_quat(connection.anchor_direction_b, rot_b);
        
        let target_anchor_a = adhesion_axis;
        let target_anchor_b = -adhesion_axis;
        
        let alignment_rot_a = quat_from_two_vectors(current_anchor_a, target_anchor_a);
        let alignment_rot_b = quat_from_two_vectors(current_anchor_b, target_anchor_b);
        
        let target_orientation_a = quat_multiply(alignment_rot_a, connection.twist_reference_a);
        let target_orientation_b = quat_multiply(alignment_rot_b, connection.twist_reference_b);
        
        let correction_rot_a = quat_multiply(target_orientation_a, quat_conjugate(rot_a));
        let correction_rot_b = quat_multiply(target_orientation_b, quat_conjugate(rot_b));
        
        let axis_angle_a = quat_to_axis_angle(correction_rot_a);
        let axis_angle_b = quat_to_axis_angle(correction_rot_b);
        
        // Project correction onto adhesion axis for twist component
        var twist_correction_a = axis_angle_a.w * dot(axis_angle_a.xyz, adhesion_axis);
        var twist_correction_b = axis_angle_b.w * dot(axis_angle_b.xyz, adhesion_axis);
        
        // Clamp to prevent excessive torques
        twist_correction_a = clamp(twist_correction_a, -1.57, 1.57);
        twist_correction_b = clamp(twist_correction_b, -1.57, 1.57);
        
        // Apply twist torque (reduced strength for stability - matches reference)
        let twist_torque_a = adhesion_axis * twist_correction_a * settings.twist_constraint_stiffness * 0.05;
        let twist_torque_b = adhesion_axis * twist_correction_b * settings.twist_constraint_stiffness * 0.05;
        
        // Twist damping (stronger damping for stability - matches reference)
        let angular_vel_a_proj = dot(ang_vel_a, adhesion_axis);
        let angular_vel_b_proj = dot(ang_vel_b, adhesion_axis);
        let relative_angular_vel = angular_vel_a_proj - angular_vel_b_proj;
        
        let twist_damping_a = -adhesion_axis * relative_angular_vel * settings.twist_constraint_damping * 0.6;
        let twist_damping_b = adhesion_axis * relative_angular_vel * settings.twist_constraint_damping * 0.6;
        
        torque_a += twist_torque_a + twist_damping_a;
        torque_b += twist_torque_b + twist_damping_b;
    }
    
    // Apply tangential forces from torques to maintain organism shape
    // IMPROVED: Use balanced tangential forces that conserve momentum
    // 
    // The issue with the original implementation was that it applied:
    //   force_a += (-delta_pos).cross(torque_b)
    //   force_b += delta_pos.cross(torque_a)
    // 
    // This creates unbalanced forces when torques differ, causing phantom drift.
    // 
    // The fix: Apply equal and opposite tangential forces based on the TOTAL torque
    // that would be needed to maintain the constraint. This ensures momentum conservation.
    
    // Calculate the total corrective torque (sum of both cells' torques)
    let total_torque = torque_a + torque_b;
    
    // Calculate tangential force that would create this torque
    // F_tangential = torque × r / |r|²
    // This ensures equal and opposite forces on both cells
    let r_squared = dist * dist;
    if (r_squared > 0.0001) {
        let tangential_force = cross(total_torque, delta_pos) / r_squared;
        
        // Apply equal and opposite tangential forces
        // This maintains shape while conserving momentum
        force_a += tangential_force;
        force_b -= tangential_force;
    }
    
    return array<vec4<f32>, 4>(
        vec4<f32>(force_a, 0.0),
        vec4<f32>(torque_a, 0.0),
        vec4<f32>(force_b, 0.0),
        vec4<f32>(torque_b, 0.0)
    );
}

// Fixed-point scale for atomic accumulation (allows ~0.001 precision)
const FIXED_POINT_SCALE: f32 = 1000.0;

// Convert float to fixed-point i32 for atomic operations
fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

// Atomically add force to a cell's accumulator
fn atomic_add_force(cell_idx: u32, force: vec3<f32>) {
    atomicAdd(&force_accum_x[cell_idx], float_to_fixed(force.x));
    atomicAdd(&force_accum_y[cell_idx], float_to_fixed(force.y));
    atomicAdd(&force_accum_z[cell_idx], float_to_fixed(force.z));
}

// Atomically add torque to a cell's accumulator
fn atomic_add_torque(cell_idx: u32, torque: vec3<f32>) {
    atomicAdd(&torque_accum_x[cell_idx], float_to_fixed(torque.x));
    atomicAdd(&torque_accum_y[cell_idx], float_to_fixed(torque.y));
    atomicAdd(&torque_accum_z[cell_idx], float_to_fixed(torque.z));
}

// Process adhesions PER-ADHESION with atomic accumulation
// Each thread handles ONE adhesion and atomically adds forces to both cells
// This allows multiple adhesions per cell without race conditions
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let adhesion_idx = global_id.x;
    
    // Get adhesion count from adhesion_counts[0]
    let adhesion_count = adhesion_counts[0];
    if (adhesion_idx >= adhesion_count) {
        return;
    }
    
    let connection = adhesion_connections[adhesion_idx];
    
    // Skip inactive connections
    if (connection.is_active == 0u) {
        return;
    }
    
    let cell_a_idx = connection.cell_a_index;
    let cell_b_idx = connection.cell_b_index;
    
    // Get cell count for validation
    let cell_count = cell_count_buffer[0];
    if (cell_a_idx >= cell_count || cell_b_idx >= cell_count) {
        return;
    }
    
    // Load cell A data
    let pos_a = positions_in[cell_a_idx].xyz;
    let mass_a = positions_in[cell_a_idx].w;
    let vel_a = velocities_in[cell_a_idx].xyz;
    let rot_a = rotations_in[cell_a_idx];
    let ang_vel_a = angular_velocities[cell_a_idx].xyz;
    
    // Load cell B data
    let pos_b = positions_in[cell_b_idx].xyz;
    let mass_b = positions_in[cell_b_idx].w;
    let vel_b = velocities_in[cell_b_idx].xyz;
    let rot_b = rotations_in[cell_b_idx];
    let ang_vel_b = angular_velocities[cell_b_idx].xyz;
    
    // Load adhesion settings for this mode
    let settings = adhesion_settings[connection.mode_index];
    
    // Compute forces for both cells
    let forces = compute_adhesion_forces(
        pos_a, pos_b, vel_a, vel_b, mass_a, mass_b,
        rot_a, rot_b, ang_vel_a, ang_vel_b,
        connection, settings
    );
    
    // forces[0] = force on cell A, forces[1] = torque on cell A
    // forces[2] = force on cell B, forces[3] = torque on cell B
    
    // Atomically accumulate forces to both cells
    atomic_add_force(cell_a_idx, forces[0].xyz);
    atomic_add_torque(cell_a_idx, forces[1].xyz);
    atomic_add_force(cell_b_idx, forces[2].xyz);
    atomic_add_torque(cell_b_idx, forces[3].xyz);
}
