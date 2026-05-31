// Glueocyte Environment Adhesion Shader
//
// When a Glueocyte cell with glueocyte_env_adhesion enabled touches a solid surface
// (cave wall via SDF or the boundary sphere), it locks onto that world-space contact
// point. A spring force then pulls it back toward that anchor each frame.
//
// Anchor acquisition: first frame the cell is within contact_threshold of a surface.
// Anchor release: when the mode's env_adhesion flag is turned off.
//
// Group 0: Standard physics bind group (params, positions, velocities, cell_count)
// Group 1: Force accumulation + env_anchor buffer
// Group 2: Mode indices, mode_cell_types, glueocyte_env_adhesion_flags
// Group 3: Cave params (for SDF sampling)

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
    _padding2: vec4<f32>,
    _padding3: vec4<f32>,
    _padding4: vec4<f32>,
    _padding5: vec4<f32>,
    _padding6: vec4<f32>,
    _padding7: vec4<f32>,
    _padding8: vec4<f32>,
    _padding9: vec4<f32>,
    _padding10: vec4<f32>,
    _padding11: vec4<f32>,
    _padding12: vec4<f32>,
    _padding13: vec4<f32>,
    _padding14: vec4<f32>,
    _padding15: vec4<f32>,
    _padding16: vec4<f32>,
    _padding17: vec4<f32>,
    _padding18: vec4<f32>,
    _padding19: vec4<f32>,
    _padding20: vec4<f32>,
    _padding21: vec4<f32>,
    _padding22: vec4<f32>,
    _padding23: vec4<f32>,
    _padding24: vec4<f32>,
    _padding25: vec4<f32>,
    _padding26: vec4<f32>,
    _padding27: vec4<f32>,
    _padding28: vec4<f32>,
    _padding29: vec4<f32>,
    _padding30: vec4<f32>,
    _padding31: vec4<f32>,
    _padding32: vec4<f32>,
    _padding33: vec4<f32>,
    _padding34: vec4<f32>,
    _padding35: vec4<f32>,
    _padding36: vec4<f32>,
    _padding37: vec4<f32>,
    _padding38: vec4<f32>,
    _padding39: vec4<f32>,
    _padding40: vec4<f32>,
    _padding41: vec4<f32>,
    _padding42: vec4<f32>,
    _padding43: vec4<f32>,
    _padding44: vec4<f32>,
    _padding45: vec4<f32>,
    _padding46: vec4<f32>,
    _padding47: vec4<f32>,
}

// Group 0: Standard physics bind group (matches physics_layout in compute_pipelines.rs)
// binding 0: params (uniform), 1: positions_in (read), 2: velocities_in (read),
// 3: positions_out (read_write), 4: velocities_out (read_write), 5: cell_count (read_write)
@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> cell_count_buffer: array<u32>;

// Group 1: Force accumulation + per-cell env anchor
// anchor: xyz = world-space anchor position, w = 1.0 if active / 0.0 if inactive
@group(1) @binding(0) var<storage, read_write> force_accum_x: array<atomic<i32>>;
@group(1) @binding(1) var<storage, read_write> force_accum_y: array<atomic<i32>>;
@group(1) @binding(2) var<storage, read_write> force_accum_z: array<atomic<i32>>;
@group(1) @binding(3) var<storage, read_write> env_anchors: array<vec4<f32>>;

// Boulder state and count for glueocyte attachment.
// Glueocytes can anchor to boulder surfaces the same way they anchor to cave walls.
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
@group(1) @binding(4) var<storage, read> boulder_state: array<GpuBoulder>;
@group(1) @binding(5) var<storage, read> boulder_count: array<u32>;

// Group 2: Mode data
@group(2) @binding(0) var<storage, read> mode_indices: array<u32>;
@group(2) @binding(1) var<storage, read> mode_cell_types: array<u32>;
@group(2) @binding(2) var<storage, read> glueocyte_env_adhesion_flags: array<u32>;
@group(2) @binding(3) var<storage, read> glueocyte_boulder_adhesion_flags: array<u32>;
// Signal gate (shared with cell adhesion shader): 4 u32 per mode
// [0]=cell_adhesion_enabled, [1]=signal_channel (0xFFFFFFFF=always active), [2]=threshold bits, [3]=self_adhesion
// The signal gate (channel + threshold) applies to ALL glueocyte adhesion types.
@group(2) @binding(4) var<storage, read> glueocyte_cell_adhesion_flags: array<u32>;
// Per-cell signal values: 16 u32 per cell (lower 11 bits = value)
@group(2) @binding(5) var<storage, read> signal_flags: array<u32>;

// Group 3: Cave params for SDF sampling
@group(3) @binding(0) var<uniform> cave_params: CaveParams;

// ---- Cave SDF (duplicated from cave_collision.wgsl) ----

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

fn smoothstep_custom(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let ix = i32(floor(pos.x));
    let iy = i32(floor(pos.y));
    let iz = i32(floor(pos.z));
    let fx = pos.x - floor(pos.x);
    let fy = pos.y - floor(pos.y);
    let fz = pos.z - floor(pos.z);
    let ux = smoothstep_custom(fx);
    let uy = smoothstep_custom(fy);
    let uz = smoothstep_custom(fz);
    let c000 = hash1(ix,     iy,     iz,     seed);
    let c100 = hash1(ix + 1, iy,     iz,     seed);
    let c010 = hash1(ix,     iy + 1, iz,     seed);
    let c110 = hash1(ix + 1, iy + 1, iz,     seed);
    let c001 = hash1(ix,     iy,     iz + 1, seed);
    let c101 = hash1(ix + 1, iy,     iz + 1, seed);
    let c011 = hash1(ix,     iy + 1, iz + 1, seed);
    let c111 = hash1(ix + 1, iy + 1, iz + 1, seed);
    let x00 = mix(c000, c100, ux);
    let x10 = mix(c010, c110, ux);
    let x01 = mix(c001, c101, ux);
    let x11 = mix(c011, c111, ux);
    let y0 = mix(x00, x10, uy);
    let y1 = mix(x01, x11, uy);
    return mix(y0, y1, uz);
}

fn fbm(pos: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    for (var i = 0u; i < cave_params.octaves; i = i + 1u) {
        let sample_pos = pos * frequency / cave_params.scale;
        let octave_seed = cave_params.seed + i * 1337u;
        value = value + amplitude * value_noise_3d(sample_pos, octave_seed);
        max_value = max_value + amplitude;
        amplitude = amplitude * cave_params.persistence;
        frequency = frequency * 2.0;
    }
    return value / max_value;
}

fn warp_domain(pos: vec3<f32>) -> vec3<f32> {
    let warp_scale = cave_params.scale * 0.5;
    let warp_strength = cave_params.smoothness * cave_params.scale;
    let warp_seed = cave_params.seed + 9999u;
    let wx = value_noise_3d(pos / warp_scale, warp_seed) - 0.5;
    let wy = value_noise_3d(pos / warp_scale + vec3<f32>(31.7, 47.3, 13.1), warp_seed) - 0.5;
    let wz = value_noise_3d(pos / warp_scale + vec3<f32>(73.9, 19.4, 67.2), warp_seed) - 0.5;
    return vec3<f32>(pos.x + wx * warp_strength, pos.y + wy * warp_strength, pos.z + wz * warp_strength);
}

fn sample_cave_density(pos: vec3<f32>) -> f32 {
    let dist_from_center = length(pos - cave_params.world_center);
    let sphere_sdf = dist_from_center - cave_params.world_radius;
    if (sphere_sdf > 0.0) { return 1.0; }
    let warped_pos = warp_domain(pos);
    let noise = fbm(warped_pos);
    let cave_threshold = clamp(cave_params.density, 0.0, 1.0);
    if (noise > cave_threshold) {
        let wall_factor = (noise - cave_threshold) / max(1.0 - cave_threshold, 0.001);
        return cave_params.threshold + wall_factor * 0.5;
    } else {
        return cave_params.threshold - 0.5;
    }
}

// Returns true if pos is touching a solid surface (cave wall, boundary sphere, or boulder)
fn is_touching_surface(pos: vec3<f32>, contact_threshold: f32, mode_idx: u32) -> bool {
    let boundary_radius = cave_params.world_radius;
    let dist_from_center = length(pos - cave_params.world_center);

    // Boundary sphere contact
    if (dist_from_center >= boundary_radius - contact_threshold) {
        return true;
    }

    // Cave wall contact (only when caves are enabled)
    if (cave_params.collision_enabled != 0u) {
        let offsets = array<vec3<f32>, 6>(
            vec3<f32>( contact_threshold, 0.0, 0.0),
            vec3<f32>(-contact_threshold, 0.0, 0.0),
            vec3<f32>(0.0,  contact_threshold, 0.0),
            vec3<f32>(0.0, -contact_threshold, 0.0),
            vec3<f32>(0.0, 0.0,  contact_threshold),
            vec3<f32>(0.0, 0.0, -contact_threshold),
        );
        for (var i = 0u; i < 6u; i++) {
            let sample_pos = pos + offsets[i];
            let density = sample_cave_density(sample_pos);
            if (density > cave_params.threshold) {
                return true;
            }
        }
    }

    // Boulder surface contact — only when boulder adhesion is enabled for this mode
    let boulder_adhesion_enabled = mode_idx < arrayLength(&glueocyte_boulder_adhesion_flags)
        && glueocyte_boulder_adhesion_flags[mode_idx] != 0u;
    if (boulder_adhesion_enabled) {
        let num_boulders = boulder_count[0];
        for (var bi = 0u; bi < num_boulders; bi++) {
            let bld = boulder_state[bi];
            if (bld.dead != 0u || bld.radius <= 0.0) { continue; }
            let dist = length(pos - bld.position);
            if (dist <= bld.radius + contact_threshold) {
                return true;
            }
        }
    }

    return false;
}

const FIXED_POINT_SCALE: f32 = 1000.0;
const SPRING_STRENGTH: f32 = 80.0;   // Cave wall / boundary spring strength
const SPRING_DAMPING: f32 = 8.0;
const CONTACT_THRESHOLD: f32 = 3.0;
const GLUEOCYTE_CELL_TYPE: u32 = 6u;
const BREAK_FORCE: f32 = 500.0;
const SIGNAL_CHANNELS: u32 = 16u;

fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

// Returns true when the glueocyte's signal gate is satisfied (i.e. it is "active").
// Uses the same flag buffer as the cell adhesion shader.
// If no signal gate is configured (channel == 0xFFFFFFFF), always returns true.
// Bit 1 of flags slot [3] = invert: when set, active when sig < threshold.
fn is_signal_gate_active(mode_idx: u32, cell_idx: u32) -> bool {
    let base = mode_idx * 4u;
    if (base + 2u >= arrayLength(&glueocyte_cell_adhesion_flags)) { return true; }
    let channel = glueocyte_cell_adhesion_flags[base + 1u];
    if (channel == 0xFFFFFFFFu) { return true; } // no gate — always active
    let threshold_bits = glueocyte_cell_adhesion_flags[base + 2u];
    let threshold = bitcast<f32>(threshold_bits);
    let raw = signal_flags[cell_idx * SIGNAL_CHANNELS + clamp(channel, 0u, 7u)];
    let sig = f32(raw & 0x7FFu);
    let flags = glueocyte_cell_adhesion_flags[base + 3u];
    let invert = (flags & 2u) != 0u;
    return select(sig >= threshold, sig < threshold, invert);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) { return; }

    // Check mode and cell type
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types)) { return; }

    let cell_type = mode_cell_types[mode_idx];
    if (cell_type != GLUEOCYTE_CELL_TYPE) {
        return;
    }

    // Check if env adhesion is enabled for this mode
    let env_adhesion_enabled = mode_idx < arrayLength(&glueocyte_env_adhesion_flags)
        && glueocyte_env_adhesion_flags[mode_idx] != 0u;

    if (!env_adhesion_enabled) {
        env_anchors[cell_idx].w = 0.0;
        return;
    }

    let pos = positions_in[cell_idx].xyz;
    let vel = velocities_in[cell_idx].xyz;
    let anchor = env_anchors[cell_idx];
    let is_active = anchor.w > 0.5;

    // Signal gate: if the gate is inactive, release any existing anchor and skip.
    if (!is_signal_gate_active(mode_idx, cell_idx)) {
        env_anchors[cell_idx].w = 0.0;
        return;
    }

    // Acquire anchor on first surface contact
    if (!is_active && is_touching_surface(pos, CONTACT_THRESHOLD, mode_idx)) {
        env_anchors[cell_idx] = vec4<f32>(pos, 1.0);
    }

    // Apply spring force toward anchor
    if (is_active) {
        let anchor_pos = anchor.xyz;
        let delta = anchor_pos - pos;
        let spring_force = delta * SPRING_STRENGTH;

        // Break the anchor if spring force exceeds threshold
        let force_mag = length(spring_force);
        if (force_mag > BREAK_FORCE) {
            env_anchors[cell_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            return;
        }

        let damping_force = -vel * SPRING_DAMPING;
        let total_force = spring_force + damping_force;

        atomicAdd(&force_accum_x[cell_idx], float_to_fixed(total_force.x));
        atomicAdd(&force_accum_y[cell_idx], float_to_fixed(total_force.y));
        atomicAdd(&force_accum_z[cell_idx], float_to_fixed(total_force.z));
    }
}
