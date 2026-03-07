// Adhesion Constraint Sub-step Shader
// Runs N additional iterations AFTER the main physics pass to stiffen joints.
// Each thread handles ONE cell, reads/writes positions_out & velocities_out directly.
// No atomic accumulators needed — each cell is processed by exactly one thread.
//
// Bind groups reuse existing layouts:
//   Group 0: physics (params, pos_in, vel_in, pos_out, vel_out, cell_count)
//   Group 1: adhesion (connections, settings, counts, cell_adhesion_indices)
//   Group 2: rotations (rot_in, ang_vel_in, rot_out, ang_vel_out)

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

struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _align_pad: vec2<u32>,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
}

// Group 0: Physics (reuses existing physics bind group)
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

// Group 1: Adhesion data (reuses existing adhesion bind group)
@group(1) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(1)
var<storage, read> adhesion_settings: array<AdhesionSettings>;

@group(1) @binding(2)
var<storage, read> adhesion_counts: array<u32>;

@group(1) @binding(3)
var<storage, read> cell_adhesion_indices: array<i32>;

// Group 2: Rotations (reuses existing rotations bind group)
@group(2) @binding(0)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(2) @binding(1)
var<storage, read> angular_velocities_in: array<vec4<f32>>;

@group(2) @binding(2)
var<storage, read_write> rotations_out: array<vec4<f32>>;

@group(2) @binding(3)
var<storage, read_write> angular_velocities_out: array<vec4<f32>>;

// Genome orientations - pure genome-derived orientations (no physics perturbation)
// Used for anchor direction transformation so structures are genome-pure
@group(2) @binding(4)
var<storage, read> genome_orientations: array<vec4<f32>>;

const PI: f32 = 3.14159265359;
const MAX_ADHESIONS_PER_CELL: u32 = 20u;

fn quat_multiply(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

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

fn quat_from_two_vectors(from_vec: vec3<f32>, to: vec3<f32>) -> vec4<f32> {
    let v1 = normalize(from_vec);
    let v2 = normalize(to);
    let cos_angle = dot(v1, v2);
    
    if (cos_angle > 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
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
    
    let halfway = normalize(v1 + v2);
    let axis = vec3<f32>(
        v1.y * halfway.z - v1.z * halfway.y,
        v1.z * halfway.x - v1.x * halfway.z,
        v1.x * halfway.y - v1.y * halfway.x
    );
    let w = dot(v1, halfway);
    return normalize(vec4<f32>(axis, w));
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Compute adhesion forces for one cell from one connection.
// Same math as adhesion_physics.wgsl — kept in sync.
fn compute_substep_forces(
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    vel_a: vec3<f32>,
    vel_b: vec3<f32>,
    rot_a: vec4<f32>,
    rot_b: vec4<f32>,
    ang_vel_a: vec3<f32>,
    ang_vel_b: vec3<f32>,
    connection: AdhesionConnection,
    settings: AdhesionSettings,
    is_cell_a: bool,
) -> array<vec3<f32>, 2> {
    var force = vec3<f32>(0.0);
    var torque = vec3<f32>(0.0);

    let delta_pos = pos_b - pos_a;
    let dist = length(delta_pos);
    if (dist < 0.0001) {
        return array<vec3<f32>, 2>(force, torque);
    }

    let adhesion_dir = delta_pos / dist;
    let rest_length = settings.rest_length;

    // Transform anchor directions to world space using GENOME orientations
    // so that adhesion structure is genome-pure and deterministic.
    // Physics rotations are still used for twist constraint correction below.
    var anchor_a: vec3<f32>;
    var anchor_b: vec3<f32>;
    let genome_rot_a = genome_orientations[connection.cell_a_index];
    let genome_rot_b = genome_orientations[connection.cell_b_index];

    if (length(connection.anchor_direction_a.xyz) < 0.001 && length(connection.anchor_direction_b.xyz) < 0.001) {
        anchor_a = vec3<f32>(1.0, 0.0, 0.0);
        anchor_b = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        anchor_a = rotate_vector_by_quat(connection.anchor_direction_a.xyz, genome_rot_a);
        anchor_b = rotate_vector_by_quat(connection.anchor_direction_b.xyz, genome_rot_b);
    }

    // Geometric spring force: anchor-defined target positions
    let target_b = pos_a + anchor_a * rest_length;
    let target_a = pos_b + anchor_b * rest_length;
    let error_a = target_a - pos_a;
    let error_b = target_b - pos_b;
    let geo_force_on_a = (error_a - error_b) * 0.5 * settings.linear_spring_stiffness;

    // Linear damping
    let rel_vel = vel_b - vel_a;
    let damp_mag = 1.0 - settings.linear_spring_damping * dot(rel_vel, adhesion_dir);
    let damping_force = -adhesion_dir * damp_mag;

    if (is_cell_a) {
        force += geo_force_on_a + damping_force;
    } else {
        force -= geo_force_on_a + damping_force;
    }

    var torque_a = vec3<f32>(0.0);
    var torque_b = vec3<f32>(0.0);

    // Orientation spring for cell A - aligns anchor toward bonded neighbor
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

    // Twist constraints - uses physics rotations for current anchor detection
    if (settings.enable_twist_constraint != 0 &&
        length(connection.twist_reference_a) > 0.001 &&
        length(connection.twist_reference_b) > 0.001) {

        let adhesion_axis = normalize(delta_pos);

        let current_anchor_a = anchor_a;
        let current_anchor_b = anchor_b;

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

        var twist_correction_a = axis_angle_a.w * dot(axis_angle_a.xyz, adhesion_axis);
        var twist_correction_b = axis_angle_b.w * dot(axis_angle_b.xyz, adhesion_axis);

        twist_correction_a = clamp(twist_correction_a, -1.57, 1.57);
        twist_correction_b = clamp(twist_correction_b, -1.57, 1.57);

        let twist_torque_a = adhesion_axis * twist_correction_a * settings.twist_constraint_stiffness;
        let twist_torque_b = adhesion_axis * twist_correction_b * settings.twist_constraint_stiffness;

        let angular_vel_a_proj = dot(ang_vel_a, adhesion_axis);
        let angular_vel_b_proj = dot(ang_vel_b, adhesion_axis);
        let relative_angular_vel = angular_vel_a_proj - angular_vel_b_proj;

        let twist_damping_a = -adhesion_axis * relative_angular_vel * settings.twist_constraint_damping;
        let twist_damping_b = adhesion_axis * relative_angular_vel * settings.twist_constraint_damping;

        torque_a += twist_torque_a + twist_damping_a;
        torque_b += twist_torque_b + twist_damping_b;
    }

    // Apply tangential forces from torques to maintain organism shape.
    // F_tangential = total_torque × delta_pos / |delta_pos|²
    // Equal and opposite on both cells to conserve momentum (matches CPU reference).
    let total_torque_val = torque_a + torque_b;
    let r_squared = dot(delta_pos, delta_pos);
    if (r_squared > 0.0001) {
        let tangential_force = cross(total_torque_val, delta_pos) / r_squared;
        if (is_cell_a) {
            force += tangential_force;
        } else {
            force -= tangential_force;
        }
    }

    if (is_cell_a) {
        torque = torque_a;
    } else {
        torque = torque_b;
    }

    return array<vec3<f32>, 2>(force, torque);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Read from OUTPUT buffers (latest state after position_update + velocity_update)
    let my_pos = positions_out[cell_idx].xyz;
    let my_mass = positions_out[cell_idx].w;
    let my_vel = velocities_out[cell_idx].xyz;
    let my_rot = rotations_out[cell_idx];
    let my_ang_vel = angular_velocities_out[cell_idx].xyz;

    if (my_mass < 0.5) {
        return;
    }

    var total_force = vec3<f32>(0.0);
    var total_torque = vec3<f32>(0.0);

    let adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let total_adhesion_count = adhesion_counts[0];

    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let adhesion_idx_signed = cell_adhesion_indices[adhesion_base + i];

        if (adhesion_idx_signed < 0) {
            continue;
        }

        let adhesion_idx = u32(adhesion_idx_signed);

        if (adhesion_idx >= total_adhesion_count) {
            continue;
        }

        let connection = adhesion_connections[adhesion_idx];

        if (connection.is_active == 0u) {
            continue;
        }

        if (connection.cell_a_index >= cell_count || connection.cell_b_index >= cell_count) {
            continue;
        }

        let is_cell_a = (connection.cell_a_index == cell_idx);
        let is_cell_b = (connection.cell_b_index == cell_idx);
        if (!is_cell_a && !is_cell_b) {
            continue;
        }

        let other_idx = select(connection.cell_a_index, connection.cell_b_index, is_cell_a);

        if (other_idx >= cell_count) {
            continue;
        }

        // Read other cell from OUTPUT buffers (Jacobi: may read stale data within one dispatch)
        let other_pos = positions_out[other_idx].xyz;
        let other_vel = velocities_out[other_idx].xyz;
        let other_rot = rotations_out[other_idx];
        let other_ang_vel = angular_velocities_out[other_idx].xyz;

        let settings = adhesion_settings[connection.mode_index];

        var pos_a: vec3<f32>;
        var pos_b: vec3<f32>;
        var vel_a: vec3<f32>;
        var vel_b: vec3<f32>;
        var rot_a: vec4<f32>;
        var rot_b: vec4<f32>;
        var ang_vel_a: vec3<f32>;
        var ang_vel_b: vec3<f32>;

        if (is_cell_a) {
            pos_a = my_pos; pos_b = other_pos;
            vel_a = my_vel; vel_b = other_vel;
            rot_a = my_rot; rot_b = other_rot;
            ang_vel_a = my_ang_vel; ang_vel_b = other_ang_vel;
        } else {
            pos_a = other_pos; pos_b = my_pos;
            vel_a = other_vel; vel_b = my_vel;
            rot_a = other_rot; rot_b = my_rot;
            ang_vel_a = other_ang_vel; ang_vel_b = my_ang_vel;
        }

        let result = compute_substep_forces(
            pos_a, pos_b, vel_a, vel_b,
            rot_a, rot_b, ang_vel_a, ang_vel_b,
            connection, settings, is_cell_a
        );

        total_force += result[0];
        total_torque += result[1];
    }

    // Clamp forces and torques
    let max_force = 5000.0;
    let max_torque = 500.0;
    let force_mag = length(total_force);
    let torque_mag = length(total_torque);

    if (force_mag > max_force) {
        total_force = total_force * (max_force / force_mag);
    }
    if (torque_mag > max_torque) {
        total_torque = total_torque * (max_torque / torque_mag);
    }

    // Integrate directly — each iteration uses full dt for maximum stiffness
    let dt = params.delta_time;
    let safe_mass = max(my_mass, 0.001);
    let acceleration = total_force / safe_mass;

    var new_vel = my_vel + acceleration * dt;

    // Clamp velocity
    let speed = length(new_vel);
    if (speed > 500.0) {
        new_vel = (new_vel / speed) * 500.0;
    }

    let new_pos = my_pos + (new_vel - my_vel) * dt;

    // Angular integration
    let radius = calculate_radius_from_mass(my_mass);
    let moment_of_inertia = 0.4 * my_mass * radius * radius;
    var new_ang_vel = my_ang_vel;
    if (moment_of_inertia > 0.0001) {
        let angular_acceleration = total_torque / moment_of_inertia;
        new_ang_vel = my_ang_vel + angular_acceleration * dt;
    }

    // Integrate rotation
    var new_rot = my_rot;
    let ang_vel_mag = length(new_ang_vel);
    if (ang_vel_mag > 0.0001) {
        let angle = ang_vel_mag * dt;
        let axis = new_ang_vel / ang_vel_mag;
        let half_angle = angle * 0.5;
        let sin_half = sin(half_angle);
        let cos_half = cos(half_angle);
        let delta_rotation = vec4<f32>(axis * sin_half, cos_half);
        new_rot = normalize(quat_multiply(delta_rotation, my_rot));
    }

    // Write corrected state back to output buffers
    positions_out[cell_idx] = vec4<f32>(new_pos, my_mass);
    velocities_out[cell_idx] = vec4<f32>(new_vel, 0.0);
    rotations_out[cell_idx] = new_rot;
    angular_velocities_out[cell_idx] = vec4<f32>(new_ang_vel, 0.0);
}
