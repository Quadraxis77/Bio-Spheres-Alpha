// Stage 4: Deduplicated cell collision broadphase.
//
// The spatial grid still preserves the full 3x3x3 logical neighborhood, but
// collision pairs are generated per occupied bucket:
// - all same-bucket pairs once
// - all pairs against 13 lexicographically-forward neighbor buckets
//
// This removes A->B / B->A duplicate checks without dropping edge- or
// corner-adjacent voxel collisions.

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

@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<u32>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

@group(1) @binding(4)
var<storage, read> stiffnesses: array<f32>;

@group(1) @binding(5)
var<storage, read> organism_labels: array<u32>;

@group(1) @binding(6)
var<storage, read_write> occupied_grid_cells: array<u32>;

@group(1) @binding(7)
var<storage, read_write> occupied_grid_count: array<atomic<u32>>;

@group(2) @binding(0)
var<storage, read_write> force_accum_x: array<atomic<i32>>;

@group(2) @binding(1)
var<storage, read_write> force_accum_y: array<atomic<i32>>;

@group(2) @binding(2)
var<storage, read_write> force_accum_z: array<atomic<i32>>;

@group(2) @binding(3)
var<storage, read_write> torque_accum_x: array<atomic<i32>>;

@group(2) @binding(4)
var<storage, read_write> torque_accum_y: array<atomic<i32>>;

@group(2) @binding(5)
var<storage, read_write> torque_accum_z: array<atomic<i32>>;

@group(2) @binding(6)
var<storage, read> rotations: array<vec4<f32>>;

@group(2) @binding(7)
var<storage, read> angular_velocities: array<vec4<f32>>;

struct GpuBoulder {
    position:         vec3<f32>,
    radius:           f32,
    velocity:         vec3<f32>,
    dead:             u32,
    seed:             u32,
    _pad:             array<u32, 3>,
    angular_velocity: vec4<f32>,
    orientation:      vec4<f32>,
}
@group(2) @binding(8) var<storage, read> boulder_state: array<GpuBoulder>;
@group(2) @binding(9) var<storage, read> boulder_count: array<u32>;
@group(2) @binding(10) var<storage, read_write> boulder_force_accum: array<atomic<i32>>;

const MAX_CELLS_PER_GRID: u32 = 16u;
const FIXED_POINT_SCALE: f32 = 1000.0;
const FRICTION_COEFF: f32 = 0.3;
const BOUNDARY_REDIRECT_FORCE: f32 = 15.0;
const BOUNDARY_MAX_REDIRECT_FORCE: f32 = 250.0;
const BOUNDARY_ALIGNMENT_TORQUE: f32 = 50.0;

const FORWARD_NEIGHBOR_OFFSETS_3D: array<vec3<i32>, 13> = array<vec3<i32>, 13>(
    vec3<i32>(0, 0, 1),
    vec3<i32>(0, 1, -1),
    vec3<i32>(0, 1, 0),
    vec3<i32>(0, 1, 1),
    vec3<i32>(1, -1, -1),
    vec3<i32>(1, -1, 0),
    vec3<i32>(1, -1, 1),
    vec3<i32>(1, 0, -1),
    vec3<i32>(1, 0, 0),
    vec3<i32>(1, 0, 1),
    vec3<i32>(1, 1, -1),
    vec3<i32>(1, 1, 0),
    vec3<i32>(1, 1, 1),
);

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn grid_coords_to_index(x: i32, y: i32, z: i32, grid_resolution: i32) -> u32 {
    return u32(x + y * grid_resolution + z * grid_resolution * grid_resolution);
}

fn grid_index_to_coords(grid_idx: u32, grid_resolution: i32) -> vec3<i32> {
    let res = grid_resolution;
    let z = i32(grid_idx) / (res * res);
    let y = (i32(grid_idx) - z * res * res) / res;
    let x = i32(grid_idx) - z * res * res - y * res;
    return vec3<i32>(x, y, z);
}

fn add_force(cell_idx: u32, force: vec3<f32>) {
    atomicAdd(&force_accum_x[cell_idx], i32(force.x * FIXED_POINT_SCALE));
    atomicAdd(&force_accum_y[cell_idx], i32(force.y * FIXED_POINT_SCALE));
    atomicAdd(&force_accum_z[cell_idx], i32(force.z * FIXED_POINT_SCALE));
}

fn add_torque(cell_idx: u32, torque: vec3<f32>) {
    atomicAdd(&torque_accum_x[cell_idx], i32(torque.x * FIXED_POINT_SCALE));
    atomicAdd(&torque_accum_y[cell_idx], i32(torque.y * FIXED_POINT_SCALE));
    atomicAdd(&torque_accum_z[cell_idx], i32(torque.z * FIXED_POINT_SCALE));
}

fn live_cell(cell_idx: u32) -> bool {
    return cell_idx < cell_count_buffer[0] && positions_in[cell_idx].w >= 0.5;
}

fn should_collide(a_idx: u32, b_idx: u32) -> bool {
    let a_organism_id = organism_labels[a_idx];
    let b_organism_id = organism_labels[b_idx];
    return !(a_organism_id != 0xFFFFFFFFu &&
             b_organism_id != 0xFFFFFFFFu &&
             a_organism_id == b_organism_id);
}

fn resolve_cell_pair(a_idx: u32, b_idx: u32) {
    if (a_idx == b_idx || !live_cell(a_idx) || !live_cell(b_idx) || !should_collide(a_idx, b_idx)) {
        return;
    }

    let pos_a = positions_in[a_idx].xyz;
    let pos_b = positions_in[b_idx].xyz;
    let mass_a = positions_in[a_idx].w;
    let mass_b = positions_in[b_idx].w;
    let radius_a = calculate_radius_from_mass(mass_a);
    let radius_b = calculate_radius_from_mass(mass_b);
    let delta = pos_a - pos_b;
    let dist_sq = dot(delta, delta);
    let min_dist = radius_a + radius_b;

    if (dist_sq >= min_dist * min_dist) {
        return;
    }

    let dist = sqrt(max(dist_sq, 0.0));
    let penetration = min_dist - dist;
    var normal: vec3<f32>;
    if (dist > 0.0001) {
        normal = delta / dist;
    } else {
        normal = select(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), a_idx > b_idx);
    }

    let stiffness_a = stiffnesses[a_idx];
    let stiffness_b = stiffnesses[b_idx];
    let combined_stiffness = (stiffness_a + stiffness_b) * 0.5;
    let vel_a = velocities_in[a_idx].xyz;
    let vel_b = velocities_in[b_idx].xyz;
    let relative_vel = vel_a - vel_b;
    let normal_damping = dot(relative_vel, normal) * 0.5;
    let normal_force_mag = penetration * combined_stiffness - normal_damping;
    let normal_force = normal * normal_force_mag;

    var force_a = normal_force;
    var force_b = -normal_force;
    var torque_a = vec3<f32>(0.0);
    var torque_b = vec3<f32>(0.0);

    let r_a = -normal * radius_a;
    let r_b = normal * radius_b;
    let omega_a = angular_velocities[a_idx].xyz;
    let omega_b = angular_velocities[b_idx].xyz;
    let v_contact_a = vel_a + cross(omega_a, r_a);
    let v_contact_b = vel_b + cross(omega_b, r_b);
    let v_slip = v_contact_a - v_contact_b;
    let v_slip_tangential = v_slip - dot(v_slip, normal) * normal;
    let slip_speed = length(v_slip_tangential);

    if (slip_speed > 0.0001) {
        let friction_dir = -v_slip_tangential / slip_speed;
        let friction_mag = min(
            FRICTION_COEFF * abs(normal_force_mag),
            slip_speed * combined_stiffness * 0.1
        );
        let friction_force = friction_dir * friction_mag;
        force_a += friction_force;
        force_b -= friction_force;
        torque_a += cross(r_a, friction_force);
        torque_b += cross(r_b, -friction_force);
    }

    add_force(a_idx, force_a);
    add_force(b_idx, force_b);
    add_torque(a_idx, torque_a);
    add_torque(b_idx, torque_b);
}

fn apply_single_cell_forces(cell_idx: u32) {
    if (!live_cell(cell_idx)) {
        return;
    }

    let pos = positions_in[cell_idx].xyz;
    let vel = velocities_in[cell_idx].xyz;
    let mass = positions_in[cell_idx].w;
    let radius = calculate_radius_from_mass(mass);
    let stiffness = stiffnesses[cell_idx];
    var force = vec3<f32>(0.0);
    var torque = vec3<f32>(0.0);

    let dist_from_center = length(pos);
    let boundary_radius = params.world_size * 0.5;
    let soft_zone = 5.0;
    // Begin boundary response when the cell surface enters the soft zone.
    // This is radius-aware without adding any samples or neighbor work.
    let cell_boundary_radius = max(boundary_radius - radius, 0.0);
    let soft_zone_start = max(cell_boundary_radius - soft_zone, 0.0);

    if (dist_from_center > soft_zone_start) {
        let penetration = (dist_from_center - soft_zone_start) / soft_zone;
        let clamped_pen = clamp(penetration, 0.0, 1.0);
        let safe_dist = max(dist_from_center, 0.001);
        let r_hat = pos / safe_dist;
        let normal = -r_hat;
        let boundary_force_mag = clamped_pen * clamped_pen * 500.0;
        force += normal * boundary_force_mag;

        let r_contact = r_hat * radius;
        let omega = angular_velocities[cell_idx].xyz;
        let v_contact = vel + cross(omega, r_contact);
        let v_tangent = v_contact - dot(v_contact, normal) * normal;
        let tangent_speed = length(v_tangent);
        if (tangent_speed > 0.0001 && clamped_pen > 0.0) {
            let redirect_force_mag = min(
                tangent_speed * clamped_pen * BOUNDARY_REDIRECT_FORCE,
                BOUNDARY_MAX_REDIRECT_FORCE
            );
            force += normal * redirect_force_mag;
        }
        if (tangent_speed > 0.0001 && boundary_force_mag > 0.0) {
            let friction_dir = -v_tangent / tangent_speed;
            let friction_mag = min(
                FRICTION_COEFF * boundary_force_mag,
                tangent_speed * stiffness * 0.1
            );
            let friction_force = friction_dir * friction_mag;
            force += friction_force;
            torque += cross(r_contact, friction_force);
        }

        if (clamped_pen > 0.0) {
            let rotation = rotations[cell_idx];
            let forward = quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));
            let rotation_axis = cross(forward, normal);
            let rotation_axis_length = length(rotation_axis);
            if (rotation_axis_length > 0.001) {
                let normalized_axis = rotation_axis / rotation_axis_length;
                let dot_product = clamp(dot(forward, normal), -1.0, 1.0);
                let angle = acos(dot_product);
                torque += normalized_axis * (BOUNDARY_ALIGNMENT_TORQUE * clamped_pen * angle);
            }
        }
    }

    let num_boulders = boulder_count[0];
    for (var bi = 0u; bi < num_boulders; bi++) {
        let bld = boulder_state[bi];
        if (bld.dead != 0u || bld.radius <= 0.0) { continue; }

        let delta = pos - bld.position;
        let dist_sq = dot(delta, delta);
        let min_dist = radius + bld.radius;

        if (dist_sq < min_dist * min_dist && dist_sq > 0.00000001) {
            let dist = sqrt(dist_sq);
            let penetration = min_dist - dist;
            let normal = delta / dist;
            let normal_force_mag = penetration * stiffness;
            force += normal * normal_force_mag;

            let reaction = -normal * normal_force_mag;
            atomicAdd(&boulder_force_accum[bi * 3u + 0u], i32(reaction.x * FIXED_POINT_SCALE));
            atomicAdd(&boulder_force_accum[bi * 3u + 1u], i32(reaction.y * FIXED_POINT_SCALE));
            atomicAdd(&boulder_force_accum[bi * 3u + 2u], i32(reaction.z * FIXED_POINT_SCALE));

            let r_a = -normal * radius;
            let omega_a = angular_velocities[cell_idx].xyz;
            let v_contact_a = vel + cross(omega_a, r_a);
            let v_slip_tangential = v_contact_a - dot(v_contact_a, normal) * normal;
            let slip_speed = length(v_slip_tangential);
            if (slip_speed > 0.0001) {
                let friction_dir = -v_slip_tangential / slip_speed;
                let friction_mag = min(
                    FRICTION_COEFF * abs(normal_force_mag),
                    slip_speed * stiffness * 0.1
                );
                let friction_force = friction_dir * friction_mag;
                force += friction_force;
                torque += cross(r_a, friction_force);
            }
        }
    }

    add_force(cell_idx, force);
    add_torque(cell_idx, torque);
}

fn process_same_bucket(grid_idx: u32, count: u32) {
    let base = grid_idx * MAX_CELLS_PER_GRID;
    for (var i = 0u; i < count; i++) {
        let a_idx = spatial_grid_cells[base + i];
        for (var j = i + 1u; j < count; j++) {
            resolve_cell_pair(a_idx, spatial_grid_cells[base + j]);
        }
    }
}

fn process_neighbor_bucket(grid_idx_a: u32, count_a: u32, grid_idx_b: u32, count_b: u32) {
    let base_a = grid_idx_a * MAX_CELLS_PER_GRID;
    let base_b = grid_idx_b * MAX_CELLS_PER_GRID;
    for (var i = 0u; i < count_a; i++) {
        let a_idx = spatial_grid_cells[base_a + i];
        for (var j = 0u; j < count_b; j++) {
            resolve_cell_pair(a_idx, spatial_grid_cells[base_b + j]);
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dispatch_idx = global_id.x;

    // Boundary and boulder response are per-cell operations, not broadphase
    // pair operations. Run them directly by cell index so crowded grid buckets
    // cannot silently drop cells beyond MAX_CELLS_PER_GRID and leave those cells
    // without any world-boundary force.
    if (dispatch_idx < cell_count_buffer[0]) {
        apply_single_cell_forces(dispatch_idx);
    }

    let occupied_idx = dispatch_idx;
    let occupied_count = atomicLoad(&occupied_grid_count[0]);
    if (occupied_idx >= occupied_count) {
        return;
    }

    let grid_idx = occupied_grid_cells[occupied_idx];
    let count = min(spatial_grid_counts[grid_idx], MAX_CELLS_PER_GRID);
    if (count == 0u) {
        return;
    }

    process_same_bucket(grid_idx, count);

    let coords = grid_index_to_coords(grid_idx, params.grid_resolution);
    for (var n = 0u; n < 13u; n++) {
        let offset = FORWARD_NEIGHBOR_OFFSETS_3D[n];
        let nx = coords.x + offset.x;
        let ny = coords.y + offset.y;
        let nz = coords.z + offset.z;
        if (nx < 0 || ny < 0 || nz < 0 ||
            nx >= params.grid_resolution ||
            ny >= params.grid_resolution ||
            nz >= params.grid_resolution) {
            continue;
        }

        let neighbor_grid_idx = grid_coords_to_index(nx, ny, nz, params.grid_resolution);
        let neighbor_count = min(spatial_grid_counts[neighbor_grid_idx], MAX_CELLS_PER_GRID);
        if (neighbor_count == 0u) {
            continue;
        }

        process_neighbor_bucket(grid_idx, count, neighbor_grid_idx, neighbor_count);
    }
}
