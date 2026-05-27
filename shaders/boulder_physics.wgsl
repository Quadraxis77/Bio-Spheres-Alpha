// Boulder Physics Shader
//
// Per-boulder physics: gravity, world sphere boundary, cave SDF collision,
// boulder-boulder collision with rolling contact friction, velocity/rotation
// integration, death detection, and moss direction update.
//
// NOTE — architectural debt:
// This shader reimplements gravity, boundary forces, cave SDF collision, and
// velocity integration that already exist in position_update.wgsl and
// collision_detection.wgsl. The correct design would have been to place boulders
// in reserved slots within the existing cell physics buffers (position_and_mass,
// velocity, rotations, angular_velocities) and let the existing pipeline handle
// all of that for free. Only the moss-specific logic (death by moss depletion,
// moss direction update) and the rolling contact torque would have needed to be
// added as branches in the existing shaders.
//
// As implemented, boulders run in a separate compute pass after the cell pipeline,
// which means they cannot interact with cells via the spatial grid (cells don't
// collide with boulders). If boulders had been placed in cell slots, cell-boulder
// collision would have worked automatically through collision_detection.wgsl.
//
// Bind groups:
//   Group 0: physics params (reuses physics_layout — gravity, world_size, etc.)
//   Group 1: boulder buffers (state, moss, moss_dir, eat_dir_accum, count)
//   Group 2: cave params (reuses cave collision bind group)

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
    gravity_mode: u32,
    angular_damping: f32,
    solo_metabolism_multiplier: f32,
    // No explicit padding — WGSL uniform structs don't need to fill the full buffer size.
    // The GPU buffer is 256 bytes; the struct only declares the fields it uses.
}

struct GpuBoulder {
    position:         vec3<f32>,
    radius:           f32,
    velocity:         vec3<f32>,
    dead:             u32,
    seed:             u32,
    _pad:             array<u32, 3>,
    angular_velocity: vec4<f32>,  // xyz = ω rad/s, w = unused
    orientation:      vec4<f32>,  // quaternion (x,y,z,w)
}

struct CaveParams {
    world_center: vec3<f32>,
    world_radius: f32,
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
    grid_resolution: u32,
    triangle_count: u32,
    collision_enabled: u32,
    collision_stiffness: f32,
    collision_damping: f32,
    substeps: u32,
    _padding: f32,
    // 256-byte padding (matches cave_collision.wgsl exactly)
    _padding2: vec4<f32>, _padding3: vec4<f32>, _padding4: vec4<f32>, _padding5: vec4<f32>,
    _padding6: vec4<f32>, _padding7: vec4<f32>, _padding8: vec4<f32>, _padding9: vec4<f32>,
    _padding10: vec4<f32>, _padding11: vec4<f32>, _padding12: vec4<f32>, _padding13: vec4<f32>,
    _padding14: vec4<f32>, _padding15: vec4<f32>, _padding16: vec4<f32>, _padding17: vec4<f32>,
    _padding18: vec4<f32>, _padding19: vec4<f32>, _padding20: vec4<f32>, _padding21: vec4<f32>,
    _padding22: vec4<f32>, _padding23: vec4<f32>, _padding24: vec4<f32>, _padding25: vec4<f32>,
    _padding26: vec4<f32>, _padding27: vec4<f32>, _padding28: vec4<f32>, _padding29: vec4<f32>,
    _padding30: vec4<f32>, _padding31: vec4<f32>, _padding32: vec4<f32>, _padding33: vec4<f32>,
    _padding34: vec4<f32>, _padding35: vec4<f32>, _padding36: vec4<f32>, _padding37: vec4<f32>,
    _padding38: vec4<f32>, _padding39: vec4<f32>, _padding40: vec4<f32>, _padding41: vec4<f32>,
    _padding42: vec4<f32>, _padding43: vec4<f32>, _padding44: vec4<f32>, _padding45: vec4<f32>,
    _padding46: vec4<f32>, _padding47: vec4<f32>,
}

// ── Group 0: Physics params ───────────────────────────────────────────────────
@group(0) @binding(0) var<uniform> params: PhysicsParams;

// ── Group 1: Boulder buffers ──────────────────────────────────────────────────
@group(1) @binding(0) var<storage, read_write> boulder_state:    array<GpuBoulder>;
@group(1) @binding(1) var<storage, read_write> boulder_moss:     array<atomic<i32>>;
@group(1) @binding(2) var<storage, read_write> boulder_moss_dir: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> boulder_eat_dir:  array<atomic<i32>>; // 3 per boulder
@group(1) @binding(4) var<storage, read>       boulder_count:    array<u32>;
// Force accumulator written by collision_detection.wgsl — cells pushing the boulder.
// Read and applied here, then the buffer is cleared by DMA next frame.
@group(1) @binding(5) var<storage, read_write> boulder_force_accum: array<atomic<i32>>; // 3 per boulder

// Water detection — same bitfield used by cells in position_update.wgsl.
// When grid_resolution == 0 the fluid system is not active; skip water checks.
struct WaterGridParams {
    grid_resolution: u32,
    cell_size:        f32,
    grid_origin_x:    f32,
    grid_origin_y:    f32,
    grid_origin_z:    f32,
    buoyancy_multiplier: f32,
    water_viscosity:  f32,
    _pad1:            f32,
}
@group(1) @binding(6) var<uniform> water_params:   WaterGridParams;
@group(1) @binding(7) var<storage, read> water_bitfield: array<u32>;

// Boulder-specific buoyancy params: [gravity_multiplier_in_water, drag_coeff, 0, 0]
// Separate from water_params so boulder buoyancy can be tuned independently of cells.
struct BoulderBuoyancyParams {
    gravity_multiplier: f32,
    drag_coeff:         f32,
    _pad0:              f32,
    _pad1:              f32,
}
@group(1) @binding(8) var<uniform> buoyancy_params: BoulderBuoyancyParams;

// ── Group 2: Cave params ──────────────────────────────────────────────────────
@group(2) @binding(0) var<uniform> cave_params: CaveParams;

// ── Constants ─────────────────────────────────────────────────────────────────
const FIXED_POINT_SCALE: f32 = 1000.0;
// Very high mass — boulders are heavy rock. Cells can push them but it takes
// many cells or sustained contact. A single cell barely moves a boulder.
const BOULDER_MASS: f32 = 50.0;
const BOULDER_DAMPING: f32 = 0.85;      // Velocity retained per second (air drag)
const BOUNDARY_STIFFNESS: f32 = 800.0;
const DEATH_SHRINK_RATE: f32 = 2.0;
const EAT_DIR_SHIFT_RATE: f32 = 0.3;

// Collision restitution: fraction of normal velocity retained after bounce.
// 0.3 = fairly inelastic (rocks don't bounce much).
const RESTITUTION: f32 = 0.3;
// Coulomb friction coefficient for rock-on-rock contact.
// Higher = more grip, faster spin-up, quicker stop.
const ROLLING_FRICTION_COEFF: f32 = 0.7;
const ANGULAR_DAMPING: f32 = 0.70;  // Aggressive damping — kills unnatural spin quickly
// Maximum angular speed (rad/s). Prevents spin from exceeding what rolling would produce.
// For a sphere rolling without slip: omega_max = v / r. We use a generous multiple.
const MAX_ANGULAR_SPEED: f32 = 8.0;

// ── Cave SDF (copied from cave_collision.wgsl) ────────────────────────────────

fn hash1(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    var h = seed;
    h = h * 374761393u + u32(x);
    h = h * 668265263u + u32(y);
    h = h * 1274126177u + u32(z);
    h ^= h >> 13u;
    h = h * 1274126177u;
    h ^= h >> 16u;
    return f32(h) / f32(0xFFFFFFFFu);
}

fn smoothstep_c(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }

fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let ix = i32(floor(pos.x)); let iy = i32(floor(pos.y)); let iz = i32(floor(pos.z));
    let fx = pos.x - floor(pos.x); let fy = pos.y - floor(pos.y); let fz = pos.z - floor(pos.z);
    let ux = smoothstep_c(fx); let uy = smoothstep_c(fy); let uz = smoothstep_c(fz);
    let c000 = hash1(ix,   iy,   iz,   seed); let c100 = hash1(ix+1, iy,   iz,   seed);
    let c010 = hash1(ix,   iy+1, iz,   seed); let c110 = hash1(ix+1, iy+1, iz,   seed);
    let c001 = hash1(ix,   iy,   iz+1, seed); let c101 = hash1(ix+1, iy,   iz+1, seed);
    let c011 = hash1(ix,   iy+1, iz+1, seed); let c111 = hash1(ix+1, iy+1, iz+1, seed);
    let x00 = mix(c000, c100, ux); let x10 = mix(c010, c110, ux);
    let x01 = mix(c001, c101, ux); let x11 = mix(c011, c111, ux);
    let y0 = mix(x00, x10, uy);   let y1 = mix(x01, x11, uy);
    return mix(y0, y1, uz);
}

fn fbm(pos: vec3<f32>) -> f32 {
    var value = 0.0; var amplitude = 1.0; var frequency = 1.0; var max_value = 0.0;
    for (var i = 0u; i < cave_params.octaves; i++) {
        let sp = pos * frequency / cave_params.scale;
        value += amplitude * value_noise_3d(sp, cave_params.seed + i * 1337u);
        max_value += amplitude; amplitude *= cave_params.persistence; frequency *= 2.0;
    }
    return value / max_value;
}

fn warp_domain(pos: vec3<f32>) -> vec3<f32> {
    let ws = cave_params.scale * 0.5; let wstr = cave_params.smoothness * cave_params.scale;
    let wseed = cave_params.seed + 9999u;
    let wx = value_noise_3d(pos / ws, wseed) - 0.5;
    let wy = value_noise_3d(pos / ws + vec3<f32>(31.7, 47.3, 13.1), wseed) - 0.5;
    let wz = value_noise_3d(pos / ws + vec3<f32>(73.9, 19.4, 67.2), wseed) - 0.5;
    return pos + vec3<f32>(wx, wy, wz) * wstr;
}

fn sample_cave_density(pos: vec3<f32>) -> f32 {
    let dist = length(pos - cave_params.world_center);
    if (dist >= cave_params.world_radius) { return 1.0; }
    let warped = warp_domain(pos);
    let noise = fbm(warped);
    let thr = clamp(cave_params.density, 0.0, 1.0);
    if (noise > thr) { return cave_params.threshold + (noise - thr) / max(1.0 - thr, 0.001) * 0.5; }
    return cave_params.threshold - 0.5;
}

fn sdf_gradient(pos: vec3<f32>, h: f32) -> vec3<f32> {
    let dx = vec3<f32>(h, 0.0, 0.0); let dy = vec3<f32>(0.0, h, 0.0); let dz = vec3<f32>(0.0, 0.0, h);
    let gx = sample_cave_density(pos + dx) - sample_cave_density(pos - dx);
    let gy = sample_cave_density(pos + dy) - sample_cave_density(pos - dy);
    let gz = sample_cave_density(pos + dz) - sample_cave_density(pos - dz);
    let g = vec3<f32>(gx, gy, gz);
    let len = length(g);
    if (len < 0.0001) { return vec3<f32>(0.0, 1.0, 0.0); }
    return g / len;
}

// ── Gravity direction ─────────────────────────────────────────────────────────

fn gravity_vector(pos: vec3<f32>) -> vec3<f32> {
    let g = params.gravity;
    switch (params.gravity_mode) {
        case 0u: { return vec3<f32>(-g, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 0.0, -g); }
        case 3u: {
            let r = length(pos);
            if (r > 0.001) { return -(pos / r) * g; }
            return vec3<f32>(0.0, -g, 0.0);
        }
        default: { return vec3<f32>(0.0, -g, 0.0); }
    }
}

// ── Water detection ───────────────────────────────────────────────────────────
// Mirrors the is_in_water() function from position_update.wgsl.
// Returns true if the given world position is inside a water voxel.
const WATER_GRID_X_GROUPS: u32 = 4u; // 128 / 32

fn is_in_water(world_pos: vec3<f32>) -> bool {
    let res = water_params.grid_resolution;
    if (res == 0u) { return false; } // fluid system not active

    let grid_pos = vec3<f32>(
        (world_pos.x - water_params.grid_origin_x) / water_params.cell_size,
        (world_pos.y - water_params.grid_origin_y) / water_params.cell_size,
        (world_pos.z - water_params.grid_origin_z) / water_params.cell_size,
    );

    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return false;
    }

    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);
    let x_group    = gx / 32u;
    let bit_index  = gx % 32u;
    let bitfield_idx = x_group + gy * WATER_GRID_X_GROUPS + gz * WATER_GRID_X_GROUPS * res;
    let bits = water_bitfield[bitfield_idx];
    return (bits & (1u << bit_index)) != 0u;
}

// ── Quaternion helpers ────────────────────────────────────────────────────────

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}

// Integrate angular velocity into orientation quaternion.
// ω is in rad/s; dt in seconds.
fn integrate_rotation(q: vec4<f32>, omega: vec3<f32>, dt: f32) -> vec4<f32> {
    let angle = length(omega) * dt;
    if (angle < 0.0001) { return q; }
    let axis = omega / length(omega);
    let half = angle * 0.5;
    let dq = vec4<f32>(axis * sin(half), cos(half));
    return normalize(quat_mul(dq, q));
}

// ── Surface contact response ──────────────────────────────────────────────────

struct ContactResult {
    vel:   vec3<f32>,
    omega: vec3<f32>,
}

// Applies normal impulse (restitution) and rolling contact friction.
// outward_normal: points away from the surface into free space.
fn surface_contact(
    vel:    vec3<f32>,
    omega:  vec3<f32>,
    normal: vec3<f32>,
    radius: f32,
    dt:     f32,
) -> ContactResult {
    var v = vel;
    var w = omega;

    let r_contact = -normal * radius;
    let v_contact = v + cross(w, r_contact);
    let vn_mag = -dot(v_contact, normal);

    // Normal impulse
    if (vn_mag > 0.0) {
        let j_n = (1.0 + RESTITUTION) * vn_mag / (1.0 + 2.5);
        let impulse_n = normal * j_n;
        v += impulse_n;
        w += cross(r_contact, impulse_n) * 2.5 / (radius * radius);
    }

    // Tangential impulse (rolling friction)
    let v_contact2 = v + cross(w, r_contact);
    let vt = v_contact2 - dot(v_contact2, normal) * normal;
    let vt_len = length(vt);
    // Only apply friction above a meaningful slip threshold — prevents spin from
    // numerical noise when the boulder is nearly stationary on a surface.
    if (vt_len > 0.05) {
        let t_dir = vt / vt_len;
        let j_n_mag = max((1.0 + RESTITUTION) * max(-dot(v_contact, normal), 0.0) / (1.0 + 2.5), 0.0);
        let j_t_max = ROLLING_FRICTION_COEFF * j_n_mag;
        let j_t_needed = vt_len / (1.0 + 2.5);
        let j_t = min(j_t_needed, j_t_max);
        let impulse_t = -t_dir * j_t;
        v += impulse_t;
        w += cross(r_contact, impulse_t) * 2.5 / (radius * radius);
    }

    return ContactResult(v, w);
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = boulder_count[0];
    if (idx >= count) { return; }

    var b = boulder_state[idx];

    // Skip already-dead boulders that have fully shrunk
    if (b.dead != 0u && b.radius <= 0.0) { return; }

    let dt = params.delta_time;
    let world_radius = params.world_size * 0.5;

    // ── Death check ───────────────────────────────────────────────────────────
    // Only check moss on boulders that have actually been spawned (radius > 0).
    // Unspawned slots have radius = 0 and are skipped by the early-exit above,
    // but we guard here too to avoid marking them dead.
    let moss = atomicLoad(&boulder_moss[idx]);
    if (moss <= 0 && b.dead == 0u && b.radius > 0.0) {
        b.dead = 1u;
    }

    if (b.dead != 0u) {
        // Shrink radius toward 0 and write back
        b.radius = max(b.radius - DEATH_SHRINK_RATE * dt, 0.0);
        boulder_state[idx] = b;
        return;
    }

    // ── Gravity ───────────────────────────────────────────────────────────────
    // Reduce gravity when submerged — boulders are dense rock but still displace water.
    // Check the boulder center; for large boulders this is an approximation but
    // good enough given the 128³ grid resolution.
    let in_water = is_in_water(b.position);
    // Use buoyancy_params.gravity_multiplier — controlled by the UI slider.
    // 0.0 = full buoyancy (floats), 1.0 = no buoyancy (sinks at full gravity).
    let gravity_multiplier = select(1.0, buoyancy_params.gravity_multiplier, in_water);
    let grav = gravity_vector(b.position) * gravity_multiplier;
    b.velocity += grav * dt;

    // Viscous drag when submerged — scales with drag_coeff from buoyancy_params
    if (in_water && water_params.water_viscosity > 0.0) {
        let drag = water_params.water_viscosity * buoyancy_params.drag_coeff * b.radius;
        b.velocity -= b.velocity * drag * dt;
    }

    // ── Cell-push forces (accumulated by collision_detection.wgsl) ────────────
    let fx = f32(atomicLoad(&boulder_force_accum[idx * 3u + 0u])) / FIXED_POINT_SCALE;
    let fy = f32(atomicLoad(&boulder_force_accum[idx * 3u + 1u])) / FIXED_POINT_SCALE;
    let fz = f32(atomicLoad(&boulder_force_accum[idx * 3u + 2u])) / FIXED_POINT_SCALE;
    // F = ma → a = F/m. High BOULDER_MASS means cells barely move the boulder.
    b.velocity += vec3<f32>(fx, fy, fz) * (dt / BOULDER_MASS);

    // ── World sphere boundary ─────────────────────────────────────────────────
    let dist_from_center = length(b.position);
    let boundary_dist = dist_from_center + b.radius - world_radius;
    if (boundary_dist > 0.0) {
        let outward_normal = normalize(b.position);
        b.position -= outward_normal * boundary_dist;
        let cr = surface_contact(b.velocity, b.angular_velocity.xyz, -outward_normal, b.radius, dt);
        b.velocity = cr.vel;
        b.angular_velocity = vec4<f32>(cr.omega, 0.0);
    }

    // ── Cave SDF collision ────────────────────────────────────────────────────
    if (cave_params.collision_enabled != 0u) {
        let density = sample_cave_density(b.position);
        let open_threshold = cave_params.threshold - 0.5 - 0.2;
        if (density > open_threshold) {
            if (density > cave_params.threshold) {
                let grad_step = max(cave_params.scale * 0.1, b.radius * 0.5);
                let outward_normal = -sdf_gradient(b.position, grad_step);
                let penetration = (density - cave_params.threshold) * cave_params.scale;
                if (penetration > 0.0) {
                    // Correct position exactly to the surface — no extra offset,
                    // which would cause jitter when resting against a wall.
                    b.position += outward_normal * penetration;
                    let cr = surface_contact(b.velocity, b.angular_velocity.xyz, outward_normal, b.radius, dt);
                    b.velocity = cr.vel;
                    b.angular_velocity = vec4<f32>(cr.omega, 0.0);
                }
            }
        }
    }

    // ── Boulder-boulder collision ─────────────────────────────────────────────
    for (var j = 0u; j < count; j++) {
        if (j == idx) { continue; }
        let other = boulder_state[j];
        if (other.dead != 0u || other.radius <= 0.0) { continue; }

        let diff = b.position - other.position;
        let dist = length(diff);
        let min_dist = b.radius + other.radius;

        if (dist < min_dist && dist > 0.001) {
            let outward_normal = diff / dist;
            let penetration = min_dist - dist;
            b.position += outward_normal * penetration * 0.5;
            let cr = surface_contact(b.velocity, b.angular_velocity.xyz, outward_normal, b.radius, dt);
            b.velocity = cr.vel;
            b.angular_velocity = vec4<f32>(cr.omega, 0.0);
        }
    }

    // ── Velocity damping + position integration ───────────────────────────────
    b.velocity *= pow(BOULDER_DAMPING, dt);
    // Clamp angular speed before damping to prevent runaway spin
    let ang_speed = length(b.angular_velocity.xyz);
    if (ang_speed > MAX_ANGULAR_SPEED) {
        b.angular_velocity = vec4<f32>(b.angular_velocity.xyz * (MAX_ANGULAR_SPEED / ang_speed), 0.0);
    }
    b.angular_velocity *= pow(ANGULAR_DAMPING, dt);
    b.position += b.velocity * dt;

    // ── Orientation integration ───────────────────────────────────────────────
    b.orientation = integrate_rotation(b.orientation, b.angular_velocity.xyz, dt);

    // ── Moss direction update from eat accumulator ────────────────────────────
    let ex = f32(atomicLoad(&boulder_eat_dir[idx * 3u + 0u])) / FIXED_POINT_SCALE;
    let ey = f32(atomicLoad(&boulder_eat_dir[idx * 3u + 1u])) / FIXED_POINT_SCALE;
    let ez = f32(atomicLoad(&boulder_eat_dir[idx * 3u + 2u])) / FIXED_POINT_SCALE;
    let eat_len = length(vec3<f32>(ex, ey, ez));

    if (eat_len > 0.001) {
        let eat_dir = vec3<f32>(ex, ey, ez) / eat_len;
        var moss_dir = boulder_moss_dir[idx].xyz;
        // Shift moss_dir away from the eating direction
        moss_dir = normalize(moss_dir - eat_dir * EAT_DIR_SHIFT_RATE * dt * eat_len);
        boulder_moss_dir[idx] = vec4<f32>(moss_dir, 0.0);

        // Clear eat accumulator
        atomicStore(&boulder_eat_dir[idx * 3u + 0u], 0);
        atomicStore(&boulder_eat_dir[idx * 3u + 1u], 0);
        atomicStore(&boulder_eat_dir[idx * 3u + 2u], 0);
    }

    boulder_state[idx] = b;
}
