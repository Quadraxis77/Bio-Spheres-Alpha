// Compute shader to extract steam voxel positions from fluid state
// and write them to the particle instance buffer

struct SteamParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,
}

struct ExtractParams {
    grid_resolution: u32,
    cell_size: f32,
    max_particles: u32,
    time: f32,
    grid_origin: vec3<f32>,
    sun_brightness: f32, // Normalized sun brightness (0-1.2) for particle lighting
    gravity_mode: u32, // 0=X, 1=Y, 2=Z, 3=radial
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Atomic counter for particle count
struct ParticleCounter {
    count: atomic<u32>,
}

// Fluid state is just an array of u32 fluid types (0=empty, 1=water, 2=lava, 3=steam, 4=solid)
@group(0) @binding(0) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(1) var<storage, read_write> particles: array<SteamParticle>;
@group(0) @binding(2) var<storage, read_write> counter: ParticleCounter;
@group(0) @binding(3) var<uniform> params: ExtractParams;
// Light field intensity per voxel (0 = shadowed, 1 = lit) - particle lighting
// is baked into the instance color here, so steam and snow in caves or at
// night are dark instead of self-illuminated.
@group(0) @binding(4) var<storage, read> light_field: array<f32>;
// Water density field. Soot from thermal smoke stacks only appears when the
// glowing stack source is submerged.
@group(0) @binding(5) var<storage, read> water_density: array<f32>;
// Geothermal glow/source field. xyz = glow color, w = source strength.
@group(0) @binding(6) var<storage, read> geothermal_glow: array<vec4<f32>>;

// Hash function for generating pseudo-random values from cell ID
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

// Generate a random float in [0, 1) from a seed
fn random_float(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967296.0;
}

// Convert grid coordinates to world position with random offset within voxel bounds
fn grid_to_world(x: u32, y: u32, z: u32) -> vec3<f32> {
    // Create a unique seed for this voxel position
    let seed = hash_u32(x + y * 1009u + z * 1009u * 1009u);
    
    // Generate random offsets within [0, 1) range for each dimension
    let random_x = random_float(seed);
    let random_y = random_float(seed + 1u);
    let random_z = random_float(seed + 2u);
    
    return params.grid_origin + vec3<f32>(
        (f32(x) + random_x) * params.cell_size,
        (f32(y) + random_y) * params.cell_size,
        (f32(z) + random_z) * params.cell_size
    );
}

// Convert 3D grid coords to 1D index
fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn neighboring_water_count(gid: vec3<u32>) -> u32 {
    let res = params.grid_resolution;
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),
        vec3<i32>(-1, 0, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 1),
        vec3<i32>(0, 0, -1)
    );

    var count = 0u;
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(gid.x) + offsets[i].x;
        let ny = i32(gid.y) + offsets[i].y;
        let nz = i32(gid.z) + offsets[i].z;
        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            let n_idx = grid_index(u32(nx), u32(ny), u32(nz));
            if (fluid_state[n_idx] & 0x7u) == 1u {
                count++;
            }
        }
    }
    return count;
}

fn world_center() -> vec3<f32> {
    let diameter = f32(params.grid_resolution) * params.cell_size;
    return params.grid_origin + vec3<f32>(diameter * 0.5);
}

fn anti_gravity_dir(pos: vec3<f32>) -> vec3<f32> {
    switch (params.gravity_mode) {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 0.0, 1.0); }
        case 3u: {
            let radial = pos - world_center();
            let r = length(radial);
            if r > 0.001 {
                return radial / r;
            }
            return vec3<f32>(0.0, 1.0, 0.0);
        }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = params.grid_resolution;

    // Check bounds
    if global_id.x >= res || global_id.y >= res || global_id.z >= res {
        return;
    }

    let idx = grid_index(global_id.x, global_id.y, global_id.z);
    let state = fluid_state[idx];

    // Fluid type is stored in lower 16 bits
    let fluid_type = state & 0x7u;

    // A steam voxel trapped inside water is a simulation bubble, not visible
    // above-surface vapor. Surface steam generally has water below it, while
    // submerged steam is surrounded by water or dense water surface field.
    if fluid_type == 3u && (water_density[idx] > 0.55 || neighboring_water_count(global_id) >= 4u) {
        return;
    }

    let geo = geothermal_glow[idx];
    let geothermal_source = geo.w > 0.10;
    let underwater_stack = geothermal_source && water_density[idx] > 0.35;
    let open_air_stack = geothermal_source && water_density[idx] <= 0.12;
    let animated_seed = hash_u32(idx + u32(params.time * 17.0));
    let soot_seed = animated_seed;
    let soot_keep = (soot_seed & 3u) == 0u;
    let vapor_seed = hash_u32(idx + u32(params.time * 11.0) + 0x6d2b79f5u);
    let vapor_keep = (vapor_seed & 7u) == 0u;

    // Render steam (3) as soft wispy particles, snow (4) as small opaque
    // round white flakes, underwater geothermal stack glow as soot, and
    // open-air vent glow as short-lived visual-only rising steam wisps.
    if fluid_type != 3u && fluid_type != 4u &&
        !(underwater_stack && soot_keep) &&
        !(open_air_stack && vapor_keep) {
        return;
    }

    // Atomically get a particle slot
    let particle_idx = atomicAdd(&counter.count, 1u);

    // Check if we have room
    if particle_idx >= params.max_particles {
        return;
    }

    // Calculate world position
    let world_pos = grid_to_world(global_id.x, global_id.y, global_id.z);

    // Create particle
    var particle: SteamParticle;
    particle.position = world_pos;

    // Lit by the local light field and overall sun brightness - no ambient
    // self-illumination (dark caves and night render dark flakes/wisps).
    let light = clamp(light_field[idx] * params.sun_brightness, 0.0, 1.2);
    var particle_kind = 1.0;
    if open_air_stack && vapor_keep && fluid_type != 3u && fluid_type != 4u {
        particle_kind = 2.0;
        // Visual-only open-air vent vapor: sparse short-lived wisps that rise
        // out of the geothermal glow field without creating simulation steam.
        let phase = random_float(vapor_seed + 13u);
        let plume_up = anti_gravity_dir(world_pos);
        let side_seed = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(plume_up.y) < 0.92);
        let plume_right = normalize(cross(side_seed, plume_up));
        let plume_forward = normalize(cross(plume_up, plume_right));
        let rise = phase * params.cell_size * 5.5;
        let spread = params.cell_size * mix(0.35, 1.45, phase);
        let swirl_angle = random_float(vapor_seed + 19u) * 6.2831853 + params.time * mix(0.65, 1.25, random_float(vapor_seed + 23u));
        let radius = spread * random_float(vapor_seed + 29u);
        let lateral = (plume_right * cos(swirl_angle) + plume_forward * sin(swirl_angle)) * radius;
        let jitter_angle = random_float(vapor_seed + 31u) * 6.2831853;
        let jitter = (plume_right * cos(jitter_angle) + plume_forward * sin(jitter_angle)) *
            params.cell_size * 0.45 * random_float(vapor_seed + 37u);
        particle.position = world_pos + lateral + jitter + plume_up * rise;
        particle.size = params.cell_size * mix(1.2, 3.2, phase);
        let fade = sin(phase * 3.14159265);
        let brightness = clamp(max(light, 0.42) + length(geo.xyz) * 0.025, 0.35, 1.1);
        particle.color = vec4<f32>(vec3<f32>(0.9, 0.9, 0.95) * brightness, 0.025 * fade);
    } else if underwater_stack && soot_keep {
        particle_kind = 3.0;
        // Underwater thermal soot: dark mineral flecks that catch a little of
        // the stack glow and drift as a loose plume.
        let soot_jitter = vec3<f32>(
            random_float(soot_seed + 11u) - 0.5,
            random_float(soot_seed + 17u) - 0.5,
            random_float(soot_seed + 23u) - 0.5
        ) * params.cell_size * 0.85;
        particle.position = world_pos + soot_jitter;
        particle.size = params.cell_size * mix(0.35, 0.9, random_float(soot_seed + 31u));
        let glow_lift = clamp(length(geo.xyz) * 0.08, 0.0, 0.22);
        let soot_color = vec3<f32>(0.025, 0.020, 0.016) + geo.xyz * glow_lift;
        particle.color = vec4<f32>(soot_color, 0.16);
    } else if fluid_type == 4u {
        // Snow: small, round, opaque white flakes
        particle.size = params.cell_size * 0.6;
        particle.color = vec4<f32>(vec3<f32>(1.0, 1.0, 1.0) * light, 0.9);
    } else {
        particle.size = params.cell_size * 1.5;  // Slightly larger than voxel
        // Steam color: white/grey, wispy
        particle.color = vec4<f32>(vec3<f32>(0.9, 0.9, 0.95) * light, 0.01);

        // Gentle per-particle drift, independent of the underlying voxel's
        // own timing. Regular steam is re-extracted fresh from live voxel
        // occupancy every step with no persistent per-particle velocity, so
        // it otherwise looks perfectly static between the fluid sim's own
        // (now-staggered, but still discrete) rise jumps. A small continuous
        // wobble - unique phase/frequency per particle via a position hash -
        // makes each wisp read as independently adrift instead of frozen to
        // the grid, without moving it far enough to visibly leave its voxel.
        let drift_seed = hash_u32(idx ^ 0x51ed270bu);
        let drift_up = anti_gravity_dir(world_pos);
        let drift_side_seed = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(drift_up.y) < 0.92);
        let drift_side = normalize(cross(drift_side_seed, drift_up));
        let drift_fwd = normalize(cross(drift_up, drift_side));
        let drift_freq = mix(0.5, 1.3, random_float(drift_seed + 1u));
        let drift_phase = random_float(drift_seed + 2u) * 6.2831853;
        let drift_angle = random_float(drift_seed + 3u) * 6.2831853
            + params.time * mix(0.3, 0.7, random_float(drift_seed + 4u));
        let drift_amp = params.cell_size * mix(0.12, 0.32, random_float(drift_seed + 5u));
        let lateral_drift = (drift_side * cos(drift_angle) + drift_fwd * sin(drift_angle))
            * drift_amp * sin(params.time * drift_freq + drift_phase);
        let vertical_bob = drift_up * params.cell_size * 0.15
            * sin(params.time * drift_freq * 1.7 + drift_phase);
        particle.position = world_pos + lateral_drift + vertical_bob;
    }

    // Animation data (time offset based on position for variation)
    particle.animation = vec4<f32>(
        params.time + f32(global_id.x) * 0.1,
        particle_kind,
        f32(global_id.y) * 0.1,
        f32(global_id.z) * 0.1
    );

    particles[particle_idx] = particle;
}
