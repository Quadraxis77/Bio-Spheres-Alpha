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

    let geo = geothermal_glow[idx];
    let underwater_stack = geo.w > 0.10 && water_density[idx] > 0.35;
    let soot_seed = hash_u32(idx + u32(params.time * 17.0));
    let soot_keep = (soot_seed & 3u) == 0u;

    // Render steam (3) as soft wispy particles, snow (4) as small opaque
    // round white flakes, and underwater geothermal stack glow as soot.
    if fluid_type != 3u && fluid_type != 4u && !(underwater_stack && soot_keep) {
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
    if underwater_stack && soot_keep {
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
