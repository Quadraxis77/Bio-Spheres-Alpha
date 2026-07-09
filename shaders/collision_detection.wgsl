// Stage 4: Deduplicated cell collision broadphase — per-pair parallel dispatch.
//
// High-density cluster performance comes from parallelising the O(n²) intra-bucket
// pair work across GPU threads instead of assigning the entire bucket to one thread.
//
// Dispatch strategy (three independent passes per thread, indexed by dispatch_idx):
//
// Pass A — Per-cell forces (boundary sphere, boulders):
//   dispatch_idx maps directly to cell_idx. O(1) per thread, always run.
//
// Pass B — Intra-bucket pairs (same-bucket broadphase):
//   Each occupied bucket with N cells has N*(N-1)/2 unique pairs.
//   dispatch_idx is reinterpreted as a flat pair index across all occupied buckets.
//   A prefix-sum over bucket pair counts would give exact mapping, but that requires
//   an extra pass. Instead: dispatch_idx maps to occupied_bucket, and within each
//   bucket we use triangle-number inversion to find (i, j).
//   For buckets <= MAX_CELLS_PER_GRID (16 cells): at most 120 pairs, all resolved
//   with full friction by one thread per cell-pair (inner loop kept but j > i only).
//   For overflow (dense) buckets: use existing phase-sampled resolve_cell_pair_dense.
//
// Pass C — Cross-bucket pairs (13 forward neighbors, rotation-grouped):
//   Same as before: one thread per occupied bucket, rotated across 3 frames.
//   This is already parallel enough since cross-bucket work is O(N_a × N_b) per
//   thread and buckets are sparsely occupied in normal scenes.
//
// Net improvement for dense clusters: intra-bucket work is now O(N²/threads)
// instead of O(N²) per single thread, giving near-linear GPU scaling with cell count.

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

@group(1) @binding(8)
var<storage, read_write> spatial_grid_overflow_cells: array<u32>;

@group(1) @binding(9)
var<storage, read_write> spatial_grid_overflow_grid_indices: array<u32>;

@group(1) @binding(10)
var<storage, read_write> spatial_grid_overflow_count: array<atomic<u32>>;

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
@group(2) @binding(11) var<storage, read_write> death_flags: array<u32>;
@group(2) @binding(12) var<storage, read> cell_adhesion_indices: array<i32>;

const MAX_CELLS_PER_GRID: u32 = 16u;
const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const FIXED_POINT_SCALE: f32 = 1000.0;
const FRICTION_COEFF: f32 = 0.3;
const BOUNDARY_REDIRECT_FORCE: f32 = 15.0;
const BOUNDARY_MAX_REDIRECT_FORCE: f32 = 250.0;
const BOUNDARY_ALIGNMENT_TORQUE: f32 = 50.0;
const OVERFLOW_DENSE_THRESHOLD: u32 = 16384u;
const OVERFLOW_EXTREME_THRESHOLD: u32 = 65536u;
const MEDIUM_BUCKET_THRESHOLD: u32 = 8u;
const DENSE_BUCKET_THRESHOLD: u32 = 12u;
const EXTREME_BUCKET_THRESHOLD: u32 = 32u;
const MEDIUM_BUCKET_SAMPLE_LIMIT: u32 = 8u;
const DENSE_BUCKET_SAMPLE_LIMIT: u32 = 6u;
const EXTREME_BUCKET_SAMPLE_LIMIT: u32 = 4u;
const OVERFLOW_PAIR_WINDOW: u32 = 16u;
const OVERCROWD_BUCKET_THRESHOLD: u32 = 16u;
const INVALID_ORGANISM_LABEL: u32 = 0xFFFFFFFFu;

// 13 forward neighbor offsets split into 3 rotation groups.
// Group 0 (frame % 3 == 0): offsets 0..3  (4 neighbors)
// Group 1 (frame % 3 == 1): offsets 4..7  (4 neighbors)
// Group 2 (frame % 3 == 2): offsets 8..12 (5 neighbors)
// Kept in the same lexicographic-forward order as before so pair deduplication
// is preserved across all frames: each pair (A→B) is always checked from A,
// never from B, regardless of which rotation group it falls in.
const FORWARD_NEIGHBOR_OFFSETS_3D: array<vec3<i32>, 13> = array<vec3<i32>, 13>(
    vec3<i32>(0, 0, 1),   // group 0
    vec3<i32>(0, 1, -1),  // group 0
    vec3<i32>(0, 1, 0),   // group 0
    vec3<i32>(0, 1, 1),   // group 0
    vec3<i32>(1, -1, -1), // group 1
    vec3<i32>(1, -1, 0),  // group 1
    vec3<i32>(1, -1, 1),  // group 1
    vec3<i32>(1, 0, -1),  // group 1
    vec3<i32>(1, 0, 0),   // group 2
    vec3<i32>(1, 0, 1),   // group 2
    vec3<i32>(1, 1, -1),  // group 2
    vec3<i32>(1, 1, 0),   // group 2
    vec3<i32>(1, 1, 1),   // group 2
);

// Rotation group start/end indices (inclusive start, exclusive end).
// group_start[g] = first neighbor index for group g.
// group_end[g]   = one-past-last neighbor index for group g.
const ROTATION_GROUP_START: array<u32, 3> = array<u32, 3>(0u, 4u, 8u);
const ROTATION_GROUP_END:   array<u32, 3> = array<u32, 3>(4u, 8u, 13u);

const FORWARD_FACE_NEIGHBOR_OFFSETS_3D: array<vec3<i32>, 3> = array<vec3<i32>, 3>(
    vec3<i32>(0, 0, 1),
    vec3<i32>(0, 1, 0),
    vec3<i32>(1, 0, 0),
);

const OVERFLOW_NEIGHBOR_OFFSETS_3D: array<vec3<i32>, 7> = array<vec3<i32>, 7>(
    vec3<i32>(0, 0, 0),
    vec3<i32>(-1, 0, 0),
    vec3<i32>(1, 0, 0),
    vec3<i32>(0, -1, 0),
    vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, -1),
    vec3<i32>(0, 0, 1),
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

fn collision_sample_count(raw_count: u32) -> u32 {
    let capped_count = min(raw_count, MAX_CELLS_PER_GRID);
    if (raw_count <= MAX_CELLS_PER_GRID) {
        return capped_count;
    }

    let medium_count = select(capped_count, MEDIUM_BUCKET_SAMPLE_LIMIT, raw_count > MEDIUM_BUCKET_THRESHOLD);
    let dense_count = select(medium_count, DENSE_BUCKET_SAMPLE_LIMIT, raw_count > DENSE_BUCKET_THRESHOLD);
    return select(dense_count, EXTREME_BUCKET_SAMPLE_LIMIT, raw_count > EXTREME_BUCKET_THRESHOLD);
}

fn organism_id(cell_idx: u32) -> u32 {
    return organism_labels[cell_idx];
}

fn add_force(cell_idx: u32, force: vec3<f32>) {
    if (dot(force, force) == 0.0) {
        return;
    }
    atomicAdd(&force_accum_x[cell_idx], i32(force.x * FIXED_POINT_SCALE));
    atomicAdd(&force_accum_y[cell_idx], i32(force.y * FIXED_POINT_SCALE));
    atomicAdd(&force_accum_z[cell_idx], i32(force.z * FIXED_POINT_SCALE));
}

fn add_torque(cell_idx: u32, torque: vec3<f32>) {
    if (dot(torque, torque) == 0.0) {
        return;
    }
    atomicAdd(&torque_accum_x[cell_idx], i32(torque.x * FIXED_POINT_SCALE));
    atomicAdd(&torque_accum_y[cell_idx], i32(torque.y * FIXED_POINT_SCALE));
    atomicAdd(&torque_accum_z[cell_idx], i32(torque.z * FIXED_POINT_SCALE));
}

fn live_cell(cell_idx: u32) -> bool {
    return cell_idx < cell_count_buffer[0] && death_flags[cell_idx] == 0u && positions_in[cell_idx].w >= 0.5;
}

fn has_adhesion_connection(cell_idx: u32) -> bool {
    let base = cell_idx * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        if (cell_adhesion_indices[base + i] >= 0) {
            return true;
        }
    }
    return false;
}

fn cull_overcrowded_overflow_cell(cell_idx: u32, grid_idx: u32) -> bool {
    if (spatial_grid_counts[grid_idx] <= OVERCROWD_BUCKET_THRESHOLD) {
        return false;
    }

    if (has_adhesion_connection(cell_idx)) {
        return false;
    }

    death_flags[cell_idx] = 1u;
    return true;
}

fn should_collide(a_idx: u32, b_idx: u32) -> bool {
    let a_organism_id = organism_id(a_idx);
    let b_organism_id = organism_id(b_idx);
    return !(a_organism_id != INVALID_ORGANISM_LABEL &&
             b_organism_id != INVALID_ORGANISM_LABEL &&
             a_organism_id == b_organism_id);
}

fn resolve_cell_pair(a_idx: u32, b_idx: u32) {
    if (a_idx == b_idx || !should_collide(a_idx, b_idx) || !live_cell(a_idx) || !live_cell(b_idx)) {
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

// Dense buckets are deliberately sampled. In that regime the expensive rolling
// friction path costs more than it helps: it adds six angular reads, cross
// products, and torque atomics for pairs that are only a representative subset
// of the real contacts. Keep the normal separation force so piles still breathe,
// but skip angular friction for sampled high-density contacts.
fn resolve_cell_pair_dense(a_idx: u32, b_idx: u32) {
    if (a_idx == b_idx || !should_collide(a_idx, b_idx) || !live_cell(a_idx) || !live_cell(b_idx)) {
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

    var normal: vec3<f32>;
    var dist: f32;
    if (dist_sq > 0.00000001) {
        let inv_dist = inverseSqrt(dist_sq);
        normal = delta * inv_dist;
        dist = dist_sq * inv_dist;
    } else {
        normal = select(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), a_idx > b_idx);
        dist = 0.0;
    }

    let penetration = min_dist - dist;
    let combined_stiffness = (stiffnesses[a_idx] + stiffnesses[b_idx]) * 0.5;
    let relative_vel = velocities_in[a_idx].xyz - velocities_in[b_idx].xyz;
    let normal_damping = dot(relative_vel, normal) * 0.5;
    let normal_force_mag = penetration * combined_stiffness - normal_damping;
    let normal_force = normal * normal_force_mag;

    add_force(a_idx, normal_force);
    add_force(b_idx, -normal_force);
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

fn process_same_bucket(grid_idx: u32, raw_count: u32) {
    let base = grid_idx * MAX_CELLS_PER_GRID;
    let count = min(raw_count, MAX_CELLS_PER_GRID);

    // Normal (non-overflow) bucket: full O(n²) with friction.
    // Called once per (bucket, pair) thread — see main() for dispatch.
    if (raw_count <= MAX_CELLS_PER_GRID) {
        for (var i = 0u; i < count; i++) {
            let a_idx = spatial_grid_cells[base + i];
            for (var j = i + 1u; j < count; j++) {
                resolve_cell_pair(a_idx, spatial_grid_cells[base + j]);
            }
        }
        return;
    }

    // Overflow bucket: phase-sampled dense path.
    let window = collision_sample_count(raw_count);
    let phase = u32(params.current_frame) % count;
    for (var i = 0u; i < count; i++) {
        let a_idx = spatial_grid_cells[base + i];
        for (var k = 1u; k <= window; k++) {
            let j = (i + k + phase) % count;
            if (j == i) { continue; }
            resolve_cell_pair_dense(a_idx, spatial_grid_cells[base + j]);
        }
    }
}

// Process one specific intra-bucket pair identified by linear pair index.
// pair_idx is in [0, count*(count-1)/2).
// Uses triangle number inversion: i = floor((sqrt(8*p+1)-1)/2), j = p - i*(i+1)/2 + i+1
fn process_bucket_pair(grid_idx: u32, raw_count: u32, pair_idx: u32) {
    let base = grid_idx * MAX_CELLS_PER_GRID;
    let count = min(raw_count, MAX_CELLS_PER_GRID);
    let max_pairs = count * (count - 1u) / 2u;
    if (pair_idx >= max_pairs) {
        return;
    }

    // Invert triangle number: find (i, j) from flat pair_idx.
    // i = floor((sqrt(8*pair_idx + 1) - 1) / 2)
    let p = pair_idx;
    let i = u32((sqrt(f32(8u * p + 1u)) - 1.0) * 0.5);
    let j = p - i * (i + 1u) / 2u + i + 1u;

    if (i >= count || j >= count) { return; }

    let a_idx = spatial_grid_cells[base + i];
    let b_idx = spatial_grid_cells[base + j];
    resolve_cell_pair(a_idx, b_idx);
}

fn process_neighbor_bucket(grid_idx_a: u32, raw_count_a: u32, grid_idx_b: u32, raw_count_b: u32) {
    let base_a = grid_idx_a * MAX_CELLS_PER_GRID;
    let base_b = grid_idx_b * MAX_CELLS_PER_GRID;
    let count_a = min(raw_count_a, MAX_CELLS_PER_GRID);
    let count_b = min(raw_count_b, MAX_CELLS_PER_GRID);
    let window_b = collision_sample_count(raw_count_b);

    for (var i = 0u; i < count_a; i++) {
        let a_idx = spatial_grid_cells[base_a + i];
        if (raw_count_b <= MAX_CELLS_PER_GRID) {
            for (var j = 0u; j < count_b; j++) {
                resolve_cell_pair(a_idx, spatial_grid_cells[base_b + j]);
            }
        } else {
            let start = (i + u32(params.current_frame)) % count_b;
            for (var k = 0u; k < window_b; k++) {
                let j = (start + k) % count_b;
                resolve_cell_pair_dense(a_idx, spatial_grid_cells[base_b + j]);
            }
        }
    }
}

fn process_overflow_cell(overflow_idx: u32, cell_idx: u32, grid_idx: u32) {
    let coords = grid_index_to_coords(grid_idx, params.grid_resolution);

    for (var n = 0u; n < 7u; n++) {
        let offset = OVERFLOW_NEIGHBOR_OFFSETS_3D[n];
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
        let raw_neighbor_count = spatial_grid_counts[neighbor_grid_idx];
        let neighbor_count = min(raw_neighbor_count, MAX_CELLS_PER_GRID);
        if (neighbor_count == 0u) {
            continue;
        }

        let window = collision_sample_count(raw_neighbor_count);
        let start = (overflow_idx + u32(params.current_frame)) % neighbor_count;
        let base = neighbor_grid_idx * MAX_CELLS_PER_GRID;
        if (raw_neighbor_count <= MAX_CELLS_PER_GRID) {
            for (var i = 0u; i < neighbor_count; i++) {
                resolve_cell_pair(cell_idx, spatial_grid_cells[base + i]);
            }
        } else {
            for (var k = 0u; k < window; k++) {
                let i = (start + k) % neighbor_count;
                resolve_cell_pair_dense(cell_idx, spatial_grid_cells[base + i]);
            }
        }
    }
}

fn process_overflow_local_pairs(overflow_idx: u32, cell_idx: u32, grid_idx: u32, overflow_count: u32) {
    let a = grid_index_to_coords(grid_idx, params.grid_resolution);
    let end_idx = min(overflow_idx + 1u + OVERFLOW_PAIR_WINDOW, overflow_count);
    for (var other_idx = overflow_idx + 1u; other_idx < end_idx; other_idx++) {
        let other_grid_idx = spatial_grid_overflow_grid_indices[other_idx];
        let b = grid_index_to_coords(other_grid_idx, params.grid_resolution);
        let delta = abs(a - b);
        if (delta.x <= 1 && delta.y <= 1 && delta.z <= 1) {
            resolve_cell_pair_dense(cell_idx, spatial_grid_overflow_cells[other_idx]);
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dispatch_idx = global_id.x;

    // -------------------------------------------------------------------------
    // Pass A: per-cell forces (boundary sphere + boulders).
    // One thread per cell — always runs regardless of grid state.
    // -------------------------------------------------------------------------
    if (dispatch_idx < cell_count_buffer[0]) {
        apply_single_cell_forces(dispatch_idx);
    }

    // -------------------------------------------------------------------------
    // Pass B: intra-bucket pairs — one thread per (bucket, pair).
    //
    // A bucket with N cells has N*(N-1)/2 unique pairs. MAX_CELLS_PER_GRID=16
    // gives at most 120 pairs per bucket. We pack (bucket_idx, pair_idx) into
    // dispatch_idx as:
    //   bucket_slot = dispatch_idx / MAX_PAIRS_PER_BUCKET
    //   pair_slot   = dispatch_idx % MAX_PAIRS_PER_BUCKET
    // where MAX_PAIRS_PER_BUCKET = 120 (= 16*15/2).
    //
    // This turns the previously serial O(n²) single-thread loop into a fully
    // parallel per-pair dispatch. For a 200-bucket dense cluster, we now
    // dispatch 200*120 = 24,000 threads instead of 200 threads each doing
    // 120 iterations sequentially.
    // -------------------------------------------------------------------------
    let MAX_PAIRS_PER_BUCKET: u32 = 120u; // 16*(16-1)/2
    let bucket_slot = dispatch_idx / MAX_PAIRS_PER_BUCKET;
    let pair_slot   = dispatch_idx % MAX_PAIRS_PER_BUCKET;

    let occupied_count = atomicLoad(&occupied_grid_count[0]);
    if (bucket_slot < occupied_count) {
        let grid_idx  = occupied_grid_cells[bucket_slot];
        let raw_count = spatial_grid_counts[grid_idx];

        if (raw_count <= MAX_CELLS_PER_GRID) {
            // Normal bucket: dispatch one thread per pair.
            process_bucket_pair(grid_idx, raw_count, pair_slot);
        } else {
            // Overflow (dense) bucket: only let pair_slot==0 run the phase-
            // sampled dense path so we don't repeat it 120 times per bucket.
            if (pair_slot == 0u) {
                process_same_bucket(grid_idx, raw_count);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pass C: cross-bucket neighbor pairs — one thread per occupied bucket.
    // Rotation-grouped across 3 frames to reduce per-frame work by ~3x.
    // -------------------------------------------------------------------------
    let bucket_idx = dispatch_idx;
    if (bucket_idx < occupied_count) {
        let grid_idx  = occupied_grid_cells[bucket_idx];
        let raw_count = spatial_grid_counts[grid_idx];
        let count     = min(raw_count, MAX_CELLS_PER_GRID);

        if (count > 0u) {
            let coords = grid_index_to_coords(grid_idx, params.grid_resolution);

            if (raw_count > MEDIUM_BUCKET_THRESHOLD) {
                // Dense bucket: all 3 face neighbors, every frame.
                for (var n = 0u; n < 3u; n++) {
                    let offset = FORWARD_FACE_NEIGHBOR_OFFSETS_3D[n];
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
                    let neighbor_count = spatial_grid_counts[neighbor_grid_idx];
                    if (neighbor_count == 0u) { continue; }
                    process_neighbor_bucket(grid_idx, raw_count, neighbor_grid_idx, neighbor_count);
                }
            } else {
                // Normal bucket: rotate through 3 groups of ~4-5 forward neighbors.
                let rotation_group = u32(params.current_frame) % 3u;
                let n_start = ROTATION_GROUP_START[rotation_group];
                let n_end   = ROTATION_GROUP_END[rotation_group];
                for (var n = n_start; n < n_end; n++) {
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
                    let neighbor_count = spatial_grid_counts[neighbor_grid_idx];
                    if (neighbor_count == 0u) { continue; }
                    process_neighbor_bucket(grid_idx, raw_count, neighbor_grid_idx, neighbor_count);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pass D: overflow list — unchanged from original.
    // -------------------------------------------------------------------------
    let overflow_idx = dispatch_idx;
    let overflow_count = atomicLoad(&spatial_grid_overflow_count[0]);
    let capped_overflow_count = min(overflow_count, params.cell_capacity);
    let dense_stride = select(1u, 2u, overflow_count > OVERFLOW_DENSE_THRESHOLD);
    let overflow_stride = select(dense_stride, 4u, overflow_count > OVERFLOW_EXTREME_THRESHOLD);
    let overflow_phase = (overflow_idx + u32(params.current_frame)) & (overflow_stride - 1u);
    if (overflow_idx < capped_overflow_count) {
        let overflow_cell     = spatial_grid_overflow_cells[overflow_idx];
        let overflow_grid_idx = spatial_grid_overflow_grid_indices[overflow_idx];
        if (cull_overcrowded_overflow_cell(overflow_cell, overflow_grid_idx)) {
            return;
        }
        if (overflow_phase != 0u) {
            return;
        }
        process_overflow_cell(overflow_idx, overflow_cell, overflow_grid_idx);
        process_overflow_local_pairs(overflow_idx, overflow_cell, overflow_grid_idx, capped_overflow_count);
    }
}
