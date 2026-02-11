// Adhesion Physics Compute Shader — PBD (Position-Based Dynamics)
//
// Each thread handles ONE cell and accumulates position/rotation corrections
// from all its adhesion bonds. Corrections are written via atomic fixed-point
// accumulators so that multiple bonds per cell compose correctly.
//
// Three constraint passes are fused into one per-cell loop:
//   1. Distance constraint — keeps bonded cells at target distance
//   2. Hinge spring — corrects orientation based on bond angle deviation
//   3. Twist constraint — hardcoded PBD twist correction using anchor refs
//
// Bond breaking is signalled by writing to the force_accum buffers with a
// sentinel value (the host reads back and removes broken bonds).
//
// Workgroup size: 256 threads for optimal GPU occupancy

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

// Adhesion settings (PBD-based, 16 bytes)
struct AdhesionSettings {
    can_break: i32,
    adhesin_length: f32,
    adhesin_stretch: f32,
    stiffness: f32,
}

// Adhesion connection structure (104 bytes matching Rust GpuAdhesionConnection)
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
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(1)
var<storage, read> adhesion_settings: array<AdhesionSettings>;

@group(1) @binding(2)
var<storage, read> adhesion_counts: array<u32>;

// Cell adhesion indices - 20 indices per cell (group 1 continued)
@group(1) @binding(3)
var<storage, read> cell_adhesion_indices: array<i32>;

// Rotations bind group (group 2)
@group(2) @binding(0)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(2) @binding(1)
var<storage, read> angular_velocities_in: array<vec4<f32>>;

@group(2) @binding(2)
var<storage, read_write> rotations_out: array<vec4<f32>>;

@group(2) @binding(3)
var<storage, read_write> angular_velocities_out: array<vec4<f32>>;

// PBD position correction accumulators (group 3) — atomic i32 fixed-point
@group(3) @binding(0)
var<storage, read_write> pbd_pos_x: array<atomic<i32>>;

@group(3) @binding(1)
var<storage, read_write> pbd_pos_y: array<atomic<i32>>;

@group(3) @binding(2)
var<storage, read_write> pbd_pos_z: array<atomic<i32>>;

// PBD rotation correction accumulators (axis*angle, fixed-point)
@group(3) @binding(3)
var<storage, read_write> pbd_rot_x: array<atomic<i32>>;

@group(3) @binding(4)
var<storage, read_write> pbd_rot_y: array<atomic<i32>>;

@group(3) @binding(5)
var<storage, read_write> pbd_rot_z: array<atomic<i32>>;

const FIXED_POINT_SCALE: f32 = 1000.0;
const PI: f32 = 3.14159265359;
const MAX_ADHESIONS_PER_CELL: u32 = 20u;

// PBD solver constants (matching CPU)
const MAX_PBD_CORRECTION: f32 = 8.0;
const MAX_HINGE_SPRING: f32 = 8.0;
const HINGE_CORRECTION_RATE: f32 = 0.8;
const TWIST_CORRECTION_RATE: f32 = 0.2;
const MAX_TWIST_CORRECTION: f32 = 0.5;
const PBD_ITERATIONS: f32 = 8.0; // Must match dispatch loop count in gpu_scene_integration.rs

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
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

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// Rotate vector by quaternion
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
    let from_len = length(from_vec);
    let to_len = length(to);
    if (from_len < 0.0001 || to_len < 0.0001) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let v1 = from_vec / from_len;
    let v2 = to / to_len;
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

// Convert float to fixed-point i32 for atomic accumulation
fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

// PBD adhesion solver — per-cell
// Each thread accumulates position and rotation corrections from all bonds,
// then writes them to atomic accumulators. The position_update shader applies them.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Load this cell's data from positions_out (updated each PBD iteration)
    let my_pos = positions_out[cell_idx].xyz;
    let my_mass = positions_out[cell_idx].w;
    let my_rot = rotations_in[cell_idx];
    let my_radius = calculate_radius_from_mass(my_mass);
    let my_inv_mass = 1.0 / max(my_mass, 0.001);

    // Accumulate corrections
    var pos_correction = vec3<f32>(0.0);
    var rot_correction = vec3<f32>(0.0); // axis * angle accumulator

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

        let other_pos = positions_out[other_idx].xyz;
        let other_mass = positions_out[other_idx].w;
        let other_rot = rotations_in[other_idx];
        let other_radius = calculate_radius_from_mass(other_mass);
        let other_inv_mass = 1.0 / max(other_mass, 0.001);

        let settings = adhesion_settings[connection.mode_index];

        // =====================================================================
        // DISTANCE CONSTRAINT
        // =====================================================================
        // Always compute delta as B - A (matching CPU) regardless of which cell we are
        let pos_a = select(other_pos, my_pos, is_cell_a);
        let pos_b = select(my_pos, other_pos, is_cell_a);
        let delta = pos_b - pos_a;
        let dist = length(delta);
        let sum_radii = my_radius + other_radius;
        let target_dist = max(sum_radii * (1.0 + settings.adhesin_length), sum_radii * 0.1);
        let error = dist - target_dist;
        let softness = 1.0 - settings.adhesin_stretch * 0.8;

        var normal: vec3<f32>;
        if (dist < 0.001) {
            if (abs(error) < 0.001) {
                continue;
            }
            let anchor_world = rotate_vector_by_quat(connection.anchor_direction_a.xyz, my_rot);
            if (length(anchor_world) > 0.001) {
                normal = normalize(anchor_world);
            } else {
                normal = vec3<f32>(1.0, 0.0, 0.0);
            }
        } else {
            normal = delta / dist;
        }

        // normal always points A→B (matching CPU)
        let correction = clamp(error * softness, -MAX_PBD_CORRECTION, MAX_PBD_CORRECTION);
        let w_total = my_inv_mass + other_inv_mass;

        if (w_total > 1e-10) {
            let s = correction / w_total;
            // CPU: positions[idx_a] += normal * s * inv_m1
            //      positions[idx_b] -= normal * s * inv_m2
            if (is_cell_a) {
                pos_correction += normal * s * my_inv_mass;
            } else {
                pos_correction -= normal * s * my_inv_mass;
            }
        }

        // =====================================================================
        // HINGE SPRING (orientation + perpendicular lever)
        // =====================================================================
        if (settings.stiffness > 0.0 && dist > 0.001) {
            let bond_dir = delta / dist;
            let correction_strength = HINGE_CORRECTION_RATE * settings.stiffness;
            let total_inv_m = my_inv_mass + other_inv_mass;

            // Hinge for this cell
            var my_anchor_dir: vec3<f32>;
            var my_target_bond: vec3<f32>;
            if (is_cell_a) {
                my_anchor_dir = connection.anchor_direction_a.xyz;
                my_target_bond = bond_dir;
            } else {
                my_anchor_dir = connection.anchor_direction_b.xyz;
                my_target_bond = -bond_dir;
            }

            let local_bond = rotate_vector_by_quat(my_target_bond, quat_conjugate(my_rot));

            if (length(my_anchor_dir) > 0.001) {
                let cross_v = cross(my_anchor_dir, local_bond);
                let sin_v = length(cross_v);
                let cos_v = dot(my_anchor_dir, local_bond);
                let dev_angle = atan2(sin_v, cos_v);

                if (sin_v > 0.0001) {
                    let axis_local = cross_v / sin_v;
                    let corr_angle = dev_angle * correction_strength;

                    // Rotation correction (in local space, will be applied as world-space axis*angle)
                    // Divided by PBD_ITERATIONS because rotations_in is stale across iterations
                    // (matching CPU which applies hinge once after distance iterations)
                    let axis_world = rotate_vector_by_quat(axis_local, my_rot);
                    rot_correction += axis_world * corr_angle / PBD_ITERATIONS;

                    // Perpendicular translational lever
                    if (total_inv_m > 1e-10) {
                        let perp = cross(axis_world, my_target_bond);
                        if (length(perp) > 0.001) {
                            let perp_n = normalize(perp);
                            let trans = clamp(dev_angle * correction_strength * dist, -MAX_HINGE_SPRING, MAX_HINGE_SPRING);
                            pos_correction += perp_n * trans * (my_inv_mass / total_inv_m) / PBD_ITERATIONS;
                        }
                    }
                }
            }
        }

        // =====================================================================
        // TWIST CONSTRAINT (hardcoded PBD)
        // =====================================================================
        if (dist > 0.001) {
            let bond_axis = delta / dist;

            var twist_ref: vec4<f32>;
            var my_anchor: vec3<f32>;
            var target_dir: vec3<f32>;
            if (is_cell_a) {
                twist_ref = connection.twist_reference_a;
                my_anchor = connection.anchor_direction_a.xyz;
                target_dir = bond_axis;
            } else {
                twist_ref = connection.twist_reference_b;
                my_anchor = connection.anchor_direction_b.xyz;
                target_dir = -bond_axis;
            }

            if (length(twist_ref) > 0.001) {
                let anchor_world = rotate_vector_by_quat(my_anchor, my_rot);
                let alignment_rot = quat_from_two_vectors(anchor_world, target_dir);
                let target_orientation = normalize(quat_multiply(alignment_rot, twist_ref));
                let correction_rot = normalize(quat_multiply(target_orientation, quat_conjugate(my_rot)));

                let aa = quat_to_axis_angle(correction_rot);
                let twist_amount = clamp(aa.w * dot(aa.xyz, bond_axis), -MAX_TWIST_CORRECTION, MAX_TWIST_CORRECTION);

                if (abs(twist_amount) > 0.0001) {
                    rot_correction += bond_axis * twist_amount * TWIST_CORRECTION_RATE / PBD_ITERATIONS;
                }
            }
        }
    }

    // Clamp total corrections
    let pos_mag = length(pos_correction);
    if (pos_mag > MAX_PBD_CORRECTION) {
        pos_correction = pos_correction * (MAX_PBD_CORRECTION / pos_mag);
    }
    let rot_mag = length(rot_correction);
    if (rot_mag > MAX_HINGE_SPRING) {
        rot_correction = rot_correction * (MAX_HINGE_SPRING / rot_mag);
    }

    // Write position corrections to PBD accumulators
    atomicAdd(&pbd_pos_x[cell_idx], float_to_fixed(pos_correction.x));
    atomicAdd(&pbd_pos_y[cell_idx], float_to_fixed(pos_correction.y));
    atomicAdd(&pbd_pos_z[cell_idx], float_to_fixed(pos_correction.z));

    // Write rotation corrections to PBD accumulators
    atomicAdd(&pbd_rot_x[cell_idx], float_to_fixed(rot_correction.x));
    atomicAdd(&pbd_rot_y[cell_idx], float_to_fixed(rot_correction.y));
    atomicAdd(&pbd_rot_z[cell_idx], float_to_fixed(rot_correction.z));
}
