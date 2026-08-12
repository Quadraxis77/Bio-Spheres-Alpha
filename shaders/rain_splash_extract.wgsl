// Rain splash extract compute shader
//
// Two entry points, same ring-buffer pattern as death_extract.wgsl:
//   spawn_new     - scans the fluid grid for falling water about to land and
//                   spawns a splash ring particle at the impact point.
//   age_particles - advances particle lifetimes each frame, expiring old ones.
//
// "Impact" detection reuses the same "falling" classification the rain-audio
// system already uses (shaders/water_audio_summary.wgsl): a water voxel whose
// velocity is aligned with gravity. A falling voxel is an *impact* the moment
// its downstream neighbor (in the fall direction) is occupied - i.e. it's
// about to be blocked from falling further, so it's landing right there.
//
// Particle layout (must match RainSplashParticle in rain_splash_particles.rs
// and the vertex attributes in rain_splash_particles.wgsl, 64 bytes):
//   position:    vec3<f32>   (12 bytes)
//   size:        f32         ( 4 bytes)
//   color:       vec4<f32>   (16 bytes)
//   animation:   vec4<f32>   (16 bytes)  x=age, y=max_lifetime
//   orientation: vec4<f32>   (16 bytes)  xyz=surface normal at impact (opposite gravity)

struct SplashParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,
    orientation: vec4<f32>,
}

struct SplashParams {
    grid_resolution: u32,
    gravity_mode: u32,
    max_particles: u32,
    // Only every sample_stride-th voxel along each axis is scanned - see
    // SplashExtractParams::sample_stride on the Rust side. spawn_new
    // early-exits unless a voxel is actively falling water, so its real
    // cost (not just thread count) scales with how much water is falling,
    // spiking hardest during the heavy rain this effect exists for.
    sample_stride: u32,
    grid_origin: vec3<f32>,
    cell_size: f32,
    time: f32,
    delta_time: f32,
    water_alpha: f32,
    _pad2: u32,
}

// Matches WATER_COLOR in shaders/fluid/density_mesh.wgsl - the shader that
// actually renders the water mesh - so splashes read as "made of the same
// water" instead of a mismatched hardcoded tint.
const WATER_COLOR: vec3<f32> = vec3<f32>(0.2, 0.5, 0.9);

struct ParticleCounter {
    count: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: SplashParams;
@group(0) @binding(1) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(2) var<storage, read> water_velocity: array<u32>;
@group(0) @binding(3) var<storage, read_write> particles: array<SplashParticle>;
@group(0) @binding(4) var<storage, read_write> counter: ParticleCounter;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn decode_velocity_component(v: u32) -> f32 {
    if v == 1u { return 1.0; }
    if v == 2u { return -1.0; }
    return 0.0;
}

fn decode_water_velocity(packed: u32) -> vec3<f32> {
    if packed == 0u { return vec3<f32>(0.0); }
    return vec3<f32>(
        decode_velocity_component(packed & 3u),
        decode_velocity_component((packed >> 2u) & 3u),
        decode_velocity_component((packed >> 4u) & 3u)
    );
}

fn world_center() -> vec3<f32> {
    let diameter = f32(params.grid_resolution) * params.cell_size;
    return params.grid_origin + vec3<f32>(diameter * 0.5);
}

fn gravity_dir(world_pos: vec3<f32>) -> vec3<f32> {
    switch (params.gravity_mode) {
        case 0u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 3u: {
            let radial = world_center() - world_pos;
            let r = length(radial);
            if r > 0.001 {
                return radial / r;
            }
            return vec3<f32>(0.0, -1.0, 0.0);
        }
        default: { return vec3<f32>(0.0, -1.0, 0.0); }
    }
}

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

fn random_float(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967296.0;
}

// -- spawn_new ------------------------------------------------------------
@compute @workgroup_size(4, 4, 4)
fn spawn_new(@builtin(global_invocation_id) sample_id: vec3<u32>) {
    let res = params.grid_resolution;
    let stride = max(params.sample_stride, 1u);
    let gid = sample_id * stride;
    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    if (fluid_state[idx] & 0x7u) != 1u {
        return;
    }

    let velocity = decode_water_velocity(water_velocity[idx]);
    if dot(velocity, velocity) < 0.5 {
        return;
    }

    let world_pos = params.grid_origin + (vec3<f32>(gid) + vec3<f32>(0.5)) * params.cell_size;
    let g_dir = gravity_dir(world_pos);
    if dot(normalize(velocity), g_dir) < 0.65 {
        return; // not falling
    }

    // Step to the actual downstream neighbor using the voxel's own velocity,
    // not gravity_dir directly - gravity_dir is continuous (in radial mode it
    // points toward the world center from wherever this voxel is, not
    // necessarily axis-aligned), but movement in this sim is always a single
    // axis-aligned swap, so velocity's components are guaranteed to be exactly
    // -1/0/1 and give a valid single-voxel grid step.
    let step_vec = vec3<i32>(velocity);
    let next = vec3<i32>(gid) + step_vec;
    let res_i = i32(res);
    if next.x < 0 || next.y < 0 || next.z < 0 || next.x >= res_i || next.y >= res_i || next.z >= res_i {
        return;
    }
    // Only a genuine landing on settled, standing water counts - not open
    // air, not solid/lava/steam, and not water that's itself still moving
    // this step (falling in a wall trickle/waterfall, or still spreading
    // laterally right after landing). Requiring zero velocity on the next
    // cell is strict on purpose: anything looser (e.g. "not also falling")
    // still let a whole falling wall-trickle column register an impact at
    // every voxel along its length instead of only where it actually lands.
    let next_idx = grid_index(u32(next.x), u32(next.y), u32(next.z));
    if (fluid_state[next_idx] & 0x7u) != 1u {
        return;
    }
    let next_velocity = decode_water_velocity(water_velocity[next_idx]);
    if dot(next_velocity, next_velocity) > 0.001 {
        return;
    }

    // Throttle: only a slice of qualifying voxels spawn a ring this frame.
    // Net spawn rate = this roll's odds x the sample_stride scan sparsity
    // above (stride=2 already examines only 1/8 of voxels). Targets an
    // overall ~1/16 of the original (pre-fix) spawn rate to match the
    // MAX_PARTICLES cut from 1536 to 384 - keep the two in sync if either
    // changes, or the ring buffer either idles half-empty or saturates and
    // starts thrashing again.
    let time_bucket = u32(params.time * 1000.0);
    let roll = hash_u32(idx ^ time_bucket);
    if (roll & 0x1u) != 0u {
        return;
    }

    let raw_slot = atomicAdd(&counter.count, 1u);
    let slot = raw_slot % params.max_particles;

    // Spawn at the shared face between the falling voxel and the one it's
    // landing on, biased slightly toward the falling (still-air) side rather
    // than the exact face - there's no shared surface-height lookup between
    // this shader and the smoothed isosurface the water mesh actually uses,
    // so this is a grid-aligned approximation biased as a safety margin
    // against ending up visually under that mesh.
    let impact_pos = params.grid_origin + (vec3<f32>(gid) + vec3<f32>(0.5)) * params.cell_size
        + g_dir * (params.cell_size * 0.42);
    let lifetime = 0.32 + random_float(roll + 7919u) * 0.18;
    let alpha = params.water_alpha * (0.85 + random_float(roll + 104729u) * 0.15);

    var p: SplashParticle;
    p.position = impact_pos;
    p.size = params.cell_size * 0.9;
    p.color = vec4<f32>(WATER_COLOR, alpha);
    p.animation = vec4<f32>(0.0, lifetime, 0.0, 0.0);
    p.orientation = vec4<f32>(-g_dir, 0.0);
    particles[slot] = p;
}

// -- age_particles ----------------------------------------------------------
@compute @workgroup_size(256)
fn age_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles {
        return;
    }

    var p = particles[particle_idx];
    if p.animation.y <= 0.0 {
        return; // uninitialized slot
    }

    let age = p.animation.x + params.delta_time;
    let max_lifetime = p.animation.y;

    if age >= max_lifetime {
        p.size = 0.0;
        p.animation.x = max_lifetime;
    } else {
        p.animation.x = age;
    }

    particles[particle_idx] = p;
}
