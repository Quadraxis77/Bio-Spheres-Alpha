// Boulder Bubble Shader
//
// Two compute entry points + one render entry point.
//
// spawn_bubbles: runs per-boulder each frame. For each live boulder that is in
//   water and moving fast enough, emits a small burst of bubble particles into
//   the ring buffer. Uses a hash of (boulder_idx, frame) to randomize spawn
//   positions around the boulder surface.
//
// age_bubbles: runs per-particle each frame. Advances age, drifts position
//   upward (bubbles rise), and zeroes expired particles.
//
// vs_main / fs_main: renders live particles as camera-facing billboards.
//   Bubbles are small white circles that fade out as they age.

// ── Structs ───────────────────────────────────────────────────────────────────

struct BubbleParticle {
    position:    vec3<f32>,
    size:        f32,
    velocity:    vec3<f32>,
    age:         f32,   // seconds since spawn
    max_age:     f32,   // lifetime in seconds
    _pad:        vec3<f32>,
}

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

struct BubbleParams {
    delta_time:      f32,
    current_time:    f32,
    current_frame:   u32,
    max_particles:   u32,
    boulder_count:   u32,
    min_speed:       f32,
    emit_rate:       f32,
    burst_duration:  f32,
    gravity_mode:    u32,
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
    // Pad to 64 bytes — wgpu requires uniform buffers to be at least 64 bytes
    _pad3:           vec4<f32>,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
}

// ── Compute bindings ──────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            bubble_params:   BubbleParams;
@group(0) @binding(1) var<storage, read>      boulder_state:   array<GpuBoulder>;
@group(0) @binding(2) var<storage, read_write> particles:      array<BubbleParticle>;
@group(0) @binding(3) var<storage, read_write> counter:        array<atomic<u32>>;
@group(0) @binding(4) var<uniform>            water_params:    WaterGridParams;
@group(0) @binding(5) var<storage, read>      water_bitfield:  array<u32>;
// Per-boulder: 1 = was in water last frame, 0 = was not. Written by spawn_bubbles.
@group(0) @binding(6) var<storage, read_write> prev_in_water:  array<u32>;
// Per-boulder: seconds remaining in the entry burst. Written by spawn_bubbles.
@group(0) @binding(7) var<storage, read_write> entry_timer:    array<f32>;

// ── Anti-gravity direction ────────────────────────────────────────────────────
// Returns the direction bubbles should float (opposite of gravity).
fn anti_gravity_dir(pos: vec3<f32>) -> vec3<f32> {
    switch (bubble_params.gravity_mode) {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }  // gravity -X → float +X
        case 2u: { return vec3<f32>(0.0, 0.0, 1.0); }  // gravity -Z → float +Z
        case 3u: {
            // Radial: float outward from origin
            let r = length(pos);
            if (r > 0.001) { return pos / r; }
            return vec3<f32>(0.0, 1.0, 0.0);
        }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }  // gravity -Y → float +Y
    }
}

// ── Water detection ───────────────────────────────────────────────────────────

const WATER_GRID_X_GROUPS: u32 = 4u;

fn is_in_water(world_pos: vec3<f32>) -> bool {
    let res = water_params.grid_resolution;
    if (res == 0u) { return false; }
    let gp = vec3<f32>(
        (world_pos.x - water_params.grid_origin_x) / water_params.cell_size,
        (world_pos.y - water_params.grid_origin_y) / water_params.cell_size,
        (world_pos.z - water_params.grid_origin_z) / water_params.cell_size,
    );
    if (gp.x < 0.0 || gp.x >= f32(res) || gp.y < 0.0 || gp.y >= f32(res) || gp.z < 0.0 || gp.z >= f32(res)) {
        return false;
    }
    let gx = u32(gp.x); let gy = u32(gp.y); let gz = u32(gp.z);
    let x_group = gx / 32u;
    let bit    = gx % 32u;
    let idx    = x_group + gy * WATER_GRID_X_GROUPS + gz * WATER_GRID_X_GROUPS * res;
    return (water_bitfield[idx] & (1u << bit)) != 0u;
}

// ── Hash helpers ──────────────────────────────────────────────────────────────

fn hash_u32(v: u32) -> f32 {
    var h = v;
    h ^= h >> 16u; h = h * 0x45d9f3bu; h ^= h >> 16u;
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

fn rand3(seed: u32) -> vec3<f32> {
    return vec3<f32>(
        hash_u32(seed * 1664525u + 1013904223u) * 2.0 - 1.0,
        hash_u32(seed * 22695477u + 1664525u)  * 2.0 - 1.0,
        hash_u32(seed * 1013904223u + 22695477u) * 2.0 - 1.0,
    );
}

// ── Spawn pass ────────────────────────────────────────────────────────────────
// Dispatched per-boulder. Detects water entry, manages entry_timer, emits bubbles
// only during the burst window after the boulder enters water.

@compute @workgroup_size(64)
fn spawn_bubbles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bi = gid.x;
    if (bi >= bubble_params.boulder_count) { return; }

    let b = boulder_state[bi];
    if (b.dead != 0u || b.radius <= 0.0) {
        prev_in_water[bi] = 0u;
        entry_timer[bi]   = 0.0;
        return;
    }

    let currently_in_water = is_in_water(b.position);
    let was_in_water       = prev_in_water[bi] != 0u;

    // Update previous-frame water state
    prev_in_water[bi] = select(0u, 1u, currently_in_water);

    // Detect entry: was dry last frame, wet this frame → start burst timer
    // Scale burst by entry speed — gentle entries produce few/no bubbles,
    // fast plunges produce a large burst.
    if (currently_in_water && !was_in_water) {
        // Entry speed = component of velocity in the gravity direction (how fast it fell in)
        let grav_dir = -anti_gravity_dir(b.position); // direction gravity pulls
        let entry_speed = max(dot(b.velocity, grav_dir), 0.0);

        // Minimum entry speed to produce any bubbles at all
        let min_entry_speed = 3.0;
        if (entry_speed > min_entry_speed) {
            // Scale burst duration: 0 at min_entry_speed, burst_duration at 15+ units/sec
            let speed_factor = clamp((entry_speed - min_entry_speed) / 12.0, 0.0, 1.0);
            entry_timer[bi] = bubble_params.burst_duration * speed_factor;
        }
        // If entry_speed <= min_entry_speed, timer stays 0 → no bubbles
    }

    // Tick down the burst timer
    var timer = entry_timer[bi];
    if (timer > 0.0) {
        timer = max(timer - bubble_params.delta_time, 0.0);
        entry_timer[bi] = timer;
    }

    // Only emit during the burst window and while still in water
    if (timer <= 0.0 || !currently_in_water) { return; }

    let speed = length(b.velocity);
    if (speed < bubble_params.min_speed) { return; }

    // Emission rate scales with how early in the burst we are — dense at entry, sparse at end
    let burst_fraction = timer / bubble_params.burst_duration; // 1.0 at entry, 0.0 at end
    let effective_rate = bubble_params.emit_rate * burst_fraction;
    let prob = effective_rate * bubble_params.delta_time;
    let roll = hash_u32(bi * 2654435761u ^ bubble_params.current_frame * 1013904223u);
    if (roll > prob) { return; }

    // Allocate ring buffer slot
    let slot = atomicAdd(&counter[0], 1u) % bubble_params.max_particles;

    // Spawn on the trailing side of the boulder
    let trail_dir = select(-normalize(b.velocity), vec3<f32>(0.0, 1.0, 0.0), speed < 0.001);
    let rand_dir  = normalize(rand3(bi * 7919u ^ bubble_params.current_frame));
    let spawn_dir = normalize(trail_dir * 0.6 + rand_dir * 0.4);
    let spawn_pos = b.position + spawn_dir * b.radius;

    // Float against gravity with small random drift
    let up    = anti_gravity_dir(spawn_pos);
    let drift = rand3(bi * 1337u ^ bubble_params.current_frame * 7919u) * 0.4;
    let vel   = up * 2.5 + drift;

    let lifetime = 0.8 + hash_u32(bi ^ bubble_params.current_frame) * 0.8;
    let size     = b.radius * 0.07 + hash_u32(slot) * b.radius * 0.05;

    particles[slot] = BubbleParticle(
        spawn_pos, size, vel, 0.0, lifetime, vec3<f32>(0.0),
    );
}

// ── Age pass ──────────────────────────────────────────────────────────────────
// Advances age and position, zeroes expired particles or particles that left water.

@compute @workgroup_size(256)
fn age_bubbles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bubble_params.max_particles) { return; }

    var p = particles[idx];
    if (p.max_age <= 0.0) { return; } // empty slot

    p.age += bubble_params.delta_time;

    // Kill if expired OR if the bubble has left the water
    if (p.age >= p.max_age || !is_in_water(p.position)) {
        particles[idx] = BubbleParticle(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0, 0.0, vec3<f32>(0.0));
        return;
    }

    // Rise and drift — velocity direction is already set toward anti-gravity
    p.position += p.velocity * bubble_params.delta_time;
    // Gentle drag
    p.velocity *= pow(0.92, bubble_params.delta_time * 60.0);

    particles[idx] = p;
}

// ── Render ────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) size:     f32,
    @location(2) velocity: vec3<f32>,
    @location(3) age:      f32,
    @location(4) max_age:  f32,
    @location(5) _pad:     vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) alpha: f32,
}

const QUAD_INDICES: array<u32, 6> = array<u32, 6>(0u, 2u, 1u, 1u, 2u, 3u);

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    inst: VertexInput,
) -> VertexOutput {
    // Skip dead/empty particles by collapsing to a degenerate position
    var out: VertexOutput;
    if (inst.max_age <= 0.0 || inst.size <= 0.0) {
        out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.uv    = vec2<f32>(0.0);
        out.alpha = 0.0;
        return out;
    }

    let corner = QUAD_INDICES[vid];
    let offset = vec2<f32>(
        select(-1.0, 1.0, (corner & 1u) != 0u),
        select(-1.0, 1.0, (corner & 2u) != 0u),
    ) * inst.size * 0.5;

    let to_cam = normalize(camera.camera_pos - inst.position);
    let right  = normalize(cross(to_cam, vec3<f32>(0.0, 1.0, 0.0)));
    let up     = cross(right, to_cam);

    let world_pos = inst.position + right * offset.x + up * offset.y;
    out.clip_pos  = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv        = vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u),
    );

    // Fade in quickly, fade out toward end of life
    let t = inst.age / inst.max_age;
    let fade_in  = smoothstep(0.0, 0.1, t);
    let fade_out = 1.0 - smoothstep(0.7, 1.0, t);
    out.alpha = fade_in * fade_out;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    let circle = 1.0 - smoothstep(0.35, 0.5, d);
    if (circle < 0.01) { discard; }
    // White bubble with a slight blue tint, semi-transparent
    let color = vec3<f32>(0.85, 0.92, 1.0);
    return vec4<f32>(color, in.alpha * circle * 0.75);
}
