// Death particle rendering shader
//
// Visual intent: dead cell membrane fragments — irregular, translucent blobs
// that look like broken organelles or cytoplasm drifting in fluid.
//
// Shape technique: domain-warped signed distance field.
//   The UV coordinates are warped by layered sine noise before the distance
//   is computed. This produces genuinely irregular shapes with concavities —
//   not just a circle with bumpy edges.
//
// Lighting: rim-bright membrane effect — the edge of each fragment is slightly
//   brighter/more opaque than the interior, like a thin cell wall catching light.
//   The interior is semi-transparent with subtle internal density variation.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) size:      f32,
    @location(2) color:     vec4<f32>,  // rgb=color, a=base_alpha
    @location(3) animation: vec4<f32>,  // x=age, y=max_lifetime, z=vel_dir_x, w=vel_dir_y
    @location(4) velocity:  vec4<f32>,  // x=vel_dir_x, y=vel_dir_y, z=vel_dir_z, w=speed
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv:         vec2<f32>,
    @location(1) color:      vec4<f32>,
    @location(2) life_frac:  f32,
    // Per-instance random seed passed through for per-fragment noise variation
    @location(3) shape_seed: f32,
}

const QUAD_INDICES: array<u32, 6> = array<u32, 6>(0u, 2u, 1u, 1u, 2u, 3u);

// ── Vertex shader ─────────────────────────────────────────────────────────────

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    instance: VertexInput,
) -> VertexOutput {
    let age          = instance.animation.x;
    let max_lifetime = instance.animation.y;
    let life_frac    = clamp(age / max(max_lifetime, 0.001), 0.0, 1.0);

    // Decelerated drift: fast at first, slows as fluid resistance builds
    let vel_dir   = vec3<f32>(instance.animation.z, instance.animation.w, instance.velocity.z);
    let speed     = instance.velocity.w;
    let decel     = 1.0 - 0.65 * life_frac;
    let world_pos = instance.position + vel_dir * speed * age * decel;

    // Size: stable through most of life, dissolves in the last 30%
    let shrink       = 1.0 - smoothstep(0.70, 1.0, life_frac) * 0.85;
    let current_size = instance.size * shrink;

    // Billboard quad
    let corner = QUAD_INDICES[vertex_id];
    let quad_offset = vec2<f32>(
        select(-1.0, 1.0, (corner & 1u) != 0u),
        select(-1.0, 1.0, (corner & 2u) != 0u)
    ) * current_size * 0.5;

    let to_camera = normalize(camera.camera_pos - world_pos);
    let right     = normalize(cross(to_camera, vec3<f32>(0.0, 1.0, 0.0)));
    let up        = cross(right, to_camera);

    let vertex_world = world_pos + right * quad_offset.x + up * quad_offset.y;

    let uv = vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u)
    );

    // Derive a per-instance shape seed from the spawn position so each
    // particle has a unique, stable shape (not flickering each frame).
    // Use the fractional part of the world position components.
    let shape_seed = fract(instance.position.x * 13.7 + instance.position.y * 7.3 + instance.position.z * 19.1);

    var out: VertexOutput;
    out.clip_pos  = camera.view_proj * vec4<f32>(vertex_world, 1.0);
    out.uv        = uv;
    out.color     = instance.color;
    out.life_frac = life_frac;
    out.shape_seed = shape_seed;
    return out;
}

// ── Fragment utilities ────────────────────────────────────────────────────────

// Smooth hash — maps a 2D point to [0,1) pseudo-randomly
fn hash2(p: vec2<f32>) -> f32 {
    let k = vec2<f32>(127.1, 311.7);
    return fract(sin(dot(p, k)) * 43758.5453);
}

// Value noise: smooth interpolation between hashed grid corners
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep curve

    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two-octave domain warp: displaces UV before distance computation.
// This is what creates genuine concavities and irregular blob shapes.
fn warp(p: vec2<f32>, seed: f32, strength: f32) -> vec2<f32> {
    // Offset each octave by the seed so different particles have different shapes
    let s1 = seed * 6.2831;
    let s2 = seed * 3.1415 + 1.7;

    // First warp pass — large-scale deformation
    let wx1 = vnoise(p * 2.1 + vec2<f32>(s1, s1 + 1.3)) - 0.5;
    let wy1 = vnoise(p * 2.1 + vec2<f32>(s1 + 3.7, s1 + 5.1)) - 0.5;

    // Second warp pass — finer detail, applied to already-warped coords
    let p2  = p + vec2<f32>(wx1, wy1) * strength;
    let wx2 = vnoise(p2 * 3.8 + vec2<f32>(s2, s2 + 2.4)) - 0.5;
    let wy2 = vnoise(p2 * 3.8 + vec2<f32>(s2 + 4.1, s2 + 0.9)) - 0.5;

    return p + vec2<f32>(wx1 + wx2 * 0.5, wy1 + wy2 * 0.5) * strength;
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let centered = in.uv - vec2<f32>(0.5);

    // Domain warp strength: stronger = more irregular shape.
    // Slightly animated over lifetime so the fragment slowly deforms as it ages.
    let warp_strength = 0.22 + in.life_frac * 0.08;
    let warped = warp(centered * 2.0, in.shape_seed, warp_strength);

    // Distance in warped space — this is what makes the shape irregular
    let warped_dist = length(warped);

    // Outer edge: soft falloff in warped space
    // Radius ~0.85 in warped coords (the *2.0 scale above means 0.5 world → 1.0 warped)
    let outer_alpha = 1.0 - smoothstep(0.72, 0.95, warped_dist);

    if outer_alpha < 0.005 {
        discard;
    }

    // ── Membrane rim effect ───────────────────────────────────────────────────
    // Real cell fragments have a denser membrane at the edge.
    // Compute the unwarped distance for the rim so it follows the true boundary.
    let true_dist = length(centered);

    // Rim: a band near the outer edge that is more opaque
    // In unwarped space the fragment fits within radius ~0.45
    let rim_inner = 0.28;
    let rim_outer = 0.44;
    let rim       = smoothstep(rim_inner, rim_outer, true_dist)
                  * (1.0 - smoothstep(rim_outer, 0.50, true_dist));
    let rim_opacity = rim * 0.55; // how much extra opacity the rim adds

    // ── Interior translucency ─────────────────────────────────────────────────
    // The interior is semi-transparent with subtle density variation from noise.
    // This mimics cytoplasm — not uniformly filled.
    let interior_noise = vnoise(centered * 5.5 + vec2<f32>(in.shape_seed * 4.0, in.life_frac));
    // Interior is more transparent at center, denser mid-way out
    let interior_density = smoothstep(0.05, 0.30, true_dist) * (0.5 + interior_noise * 0.35);

    // ── Combine opacity layers ────────────────────────────────────────────────
    // Base: the warped shape boundary
    // + rim: membrane edge
    // + interior: cytoplasm density
    let combined_opacity = outer_alpha * (0.30 + interior_density * 0.45 + rim_opacity);

    // ── Fade over lifetime ────────────────────────────────────────────────────
    let fade = 1.0 - smoothstep(0.65, 1.0, in.life_frac);

    // ── Color variation across the fragment ──────────────────────────────────
    // Rim is slightly cooler/more saturated (membrane pigment)
    // Interior is slightly warmer/more washed out (cytoplasm)
    let rim_tint   = mix(in.color.rgb, in.color.rgb * vec3<f32>(0.88, 0.92, 1.0), rim * 0.4);
    let final_color = mix(rim_tint, in.color.rgb * 1.08, interior_density * 0.3);

    let final_alpha = in.color.a * combined_opacity * fade;

    return vec4<f32>(final_color, clamp(final_alpha, 0.0, 1.0));
}
