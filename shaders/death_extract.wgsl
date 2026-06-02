// Death particle extract compute shader
//
// Two entry points:
//   spawn_new     - scans cell arrays for newly-dead cells and spawns particles
//   age_particles - advances particle lifetimes each frame
//
// Particle layout (must match DeathParticle in death_particles.rs, 64 bytes):
//   position:  vec3<f32>   (12 bytes)
//   size:      f32         ( 4 bytes)
//   color:     vec4<f32>   (16 bytes)
//   animation: vec4<f32>   (16 bytes)  x=age, y=max_lifetime, z=vel_dir_x, w=vel_dir_y
//   velocity:  vec4<f32>   (16 bytes)  x=vel_dir_x, y=vel_dir_y, z=vel_dir_z, w=unused

struct DeathParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,  // x=age, y=max_lifetime, z=vel_dir_x, w=vel_dir_y
    velocity: vec4<f32>,   // x=vel_dir_x, y=vel_dir_y, z=vel_dir_z, w=unused
}

struct ExtractParams {
    cell_capacity: u32,
    max_particles: u32,
    delta_time: f32,
    time: f32,
}

struct ParticleCounter {
    count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read>       death_flags:       array<u32>;
@group(0) @binding(1) var<storage, read>       prev_death_flags:  array<u32>;
@group(0) @binding(2) var<storage, read>       position_and_mass: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particles:         array<DeathParticle>;
@group(0) @binding(4) var<storage, read_write> counter:           ParticleCounter;
@group(0) @binding(5) var<uniform>             params:            ExtractParams;

// -- Utilities ----------------------------------------------------------------

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

// Uniform random direction on the unit sphere
fn random_unit_vec3(seed: u32) -> vec3<f32> {
    let theta = random_float(seed)       * 6.28318530718;
    let phi   = acos(2.0 * random_float(seed + 1u) - 1.0);
    return vec3<f32>(
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    );
}

// -- spawn_new ----------------------------------------------------------------
// Detects cells that transitioned alive->dead this frame and emits tissue fragments.
//
// Visual intent: pale, semi-transparent fragments that drift slowly outward in a
// sphere and gradually fade - like dead cell membrane breaking apart in fluid.
@compute @workgroup_size(256)
fn spawn_new(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    if cell_idx >= params.cell_capacity {
        return;
    }

    let is_dead    = death_flags[cell_idx] == 1u;
    let was_dead   = prev_death_flags[cell_idx] == 1u;
    let newly_dead = is_dead && !was_dead;
    if !newly_dead { return; }

    let pos_mass = position_and_mass[cell_idx];
    let pos  = pos_mass.xyz;
    let mass = pos_mass.w;
    if mass <= 0.0 { return; }

    // Radius from mass: r = cbrt(3*mass / (4*pi))
    let radius = pow(mass * 0.23873241463, 0.33333333333);

    // More fragments than before - they're small and subtle so we need density
    let num_particles = 10u;
    let base_seed = hash_u32(cell_idx ^ u32(params.time * 1000.0));

    for (var i = 0u; i < num_particles; i++) {
        let raw_slot = atomicAdd(&counter.count, 1u);
        let slot = raw_slot % params.max_particles;

        let seed_i = base_seed + i * 11u;

        // Direction: uniform sphere - fragments scatter in all directions equally
        let vel_dir = random_unit_vec3(seed_i);

        // Speed: slow - fragments drift, not fly.
        // Range: 0.3-0.8 cell-radii per second.
        let speed = radius * (0.3 + random_float(seed_i + 2u) * 0.5);

        // Lifetime: long - tissue lingers before dissolving (1.5-3.5 s)
        let lifetime = 1.5 + random_float(seed_i + 3u) * 2.0;

        // Fragment size: small relative to cell, slight variation
        // Fragments are 15-35% of cell radius
        let sz = radius * (0.15 + random_float(seed_i + 4u) * 0.20);

        // Color: pale biological tones - grey-white, faint pink, muted beige
        // Vary hue slightly per particle for organic feel
        let hue_roll = random_float(seed_i + 5u);
        var r: f32;
        var g: f32;
        var b: f32;
        if hue_roll < 0.5 {
            // Grey-white membrane fragment
            let grey = 0.75 + random_float(seed_i + 6u) * 0.20;
            r = grey;
            g = grey * 0.95;
            b = grey * 0.92;
        } else if hue_roll < 0.80 {
            // Faint pink / cytoplasm
            r = 0.80 + random_float(seed_i + 6u) * 0.15;
            g = 0.60 + random_float(seed_i + 7u) * 0.10;
            b = 0.60 + random_float(seed_i + 8u) * 0.10;
        } else {
            // Muted beige / organelle remnant
            r = 0.72 + random_float(seed_i + 6u) * 0.12;
            g = 0.65 + random_float(seed_i + 7u) * 0.10;
            b = 0.50 + random_float(seed_i + 8u) * 0.10;
        }
        // Alpha: semi-transparent from birth
        let alpha = 0.45 + random_float(seed_i + 9u) * 0.30;

        // Store speed in velocity.w so the render shader can decelerate drift
        var p: DeathParticle;
        p.position  = pos;
        p.size      = sz;
        p.color     = vec4<f32>(r, g, b, alpha);
        p.animation = vec4<f32>(0.0, lifetime, vel_dir.x, vel_dir.y);
        p.velocity  = vec4<f32>(vel_dir.x, vel_dir.y, vel_dir.z, speed);
        particles[slot] = p;
    }
}

// -- age_particles ------------------------------------------------------------
@compute @workgroup_size(256)
fn age_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if particle_idx >= params.max_particles { return; }

    var p = particles[particle_idx];
    if p.animation.y <= 0.0 { return; } // uninitialized slot

    let age          = p.animation.x + params.delta_time;
    let max_lifetime = p.animation.y;

    if age >= max_lifetime {
        p.size        = 0.0;
        p.animation.x = max_lifetime;
    } else {
        p.animation.x = age;
    }

    particles[particle_idx] = p;
}
