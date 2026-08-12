// Post-process shader: eye adaptation + contrast.
//
// Two entry points share the same bind group layout:
//   cs_adapt  — compute, runs first; samples the scene texture to update the
//               persistent exposure buffer toward the target.
//   fs_tonemap — fragment, runs second; applies exposure + contrast to each pixel.

struct Params {
    contrast: f32,
    adapt_speed: f32,  // fraction of gap to close per frame (0.01 = slow, 0.2 = fast)
    adapt_min: f32,    // minimum allowed exposure multiplier
    adapt_max: f32,    // maximum allowed exposure multiplier
    adapt_enabled: u32,
    time: f32,               // seconds since PostProcessRenderer was created, drives ripple animation
    underwater_fraction: f32, // 0-1 continuous - camera occupancy inside water
    boundary_fraction: f32,   // 0-1 continuous - proximity to / pressure against the world sphere
    water_crossing_pulse: f32,    // 0-1, decays from 1.0 on an actual water-surface crossing
    boundary_crossing_pulse: f32, // 0-1, decays from 1.0 on an actual world-sphere crossing
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Persistent single-float exposure value, updated by cs_adapt each frame.
@group(0) @binding(1) var<storage, read_write> exposure_buf: array<f32>;
@group(0) @binding(2) var scene_tex: texture_2d<f32>;
@group(0) @binding(3) var scene_samp: sampler;

// ── Perceptual luminance ────────────────────────────────────────────────────
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ── Adapt exposure compute ──────────────────────────────────────────────────
// One thread samples a 4×4 grid across the scene and exponentially smooths
// the exposure buffer toward the metered target (keeps middle grey at 0.18).
@compute @workgroup_size(1)
fn cs_adapt(@builtin(global_invocation_id) gid: vec3<u32>) {
    _ = gid;
    let dim = textureDimensions(scene_tex, 0);
    var avg = 0.0;
    for (var yi = 0u; yi < 4u; yi++) {
        for (var xi = 0u; xi < 4u; xi++) {
            let px = vec2<u32>(
                u32(f32(dim.x) * (f32(xi) + 0.5) * 0.25),
                u32(f32(dim.y) * (f32(yi) + 0.5) * 0.25),
            );
            avg += luma(textureLoad(scene_tex, px, 0).rgb);
        }
    }
    avg /= 16.0;

    // Target: expose so the average maps to middle grey (0.18).
    let target_exp = clamp(0.18 / max(avg, 0.001), params.adapt_min, params.adapt_max);
    let prev       = exposure_buf[0];
    exposure_buf[0] = mix(prev, target_exp, params.adapt_speed);
}

// ── Tonemap render ──────────────────────────────────────────────────────────
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_tonemap(@builtin(vertex_index) vi: u32) -> VOut {
    var out: VOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv  = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_tonemap(in: VOut) -> @location(0) vec4<f32> {
    var uv = in.uv;
    let to_center = uv - vec2<f32>(0.5, 0.5);
    let dist = length(to_center);
    let radial = to_center / max(dist, 0.001);

    // ── Lens distortion ────────────────────────────────────────────────────
    // Continuous, gentle warps while lingering near/inside a boundary; a
    // sharper radial ripple riding on top only while a crossing pulse is
    // still decaying, so the "whoosh" reads as a one-off event.
    let boundary_warp = params.boundary_fraction * sin(dist * 18.0 + params.time * 1.5) * 0.0015;
    let boundary_ripple =
        params.boundary_crossing_pulse * sin(dist * 40.0 - params.time * 14.0) * 0.006;
    let water_ripple = params.water_crossing_pulse * sin(dist * 34.0 - params.time * 16.0) * 0.008;

    uv += radial * (boundary_warp + boundary_ripple + water_ripple);

    // Underwater: a gentle, slow two-axis refraction wobble - the "looking
    // through water" cue. Kept low-amplitude and low-frequency on purpose:
    // a full-screen distortion that's too large or too fast reads as motion
    // sickness territory, not ambience, especially since it's active
    // continuously for as long as the camera stays submerged.
    let water_wobble = vec2<f32>(
        sin(uv.y * 6.0 + params.time * 0.3),
        sin(uv.x * 5.0 + params.time * 0.22 + 1.7),
    ) * params.underwater_fraction * 0.0015;
    uv += water_wobble;

    // Very subtle chromatic aberration - reads as "glass" near the world
    // sphere boundary, strongest right at the moment of crossing it.
    let aberration = params.boundary_fraction * 0.0025 + params.boundary_crossing_pulse * 0.004;
    var color: vec3<f32>;
    if aberration > 0.0001 {
        color.r = textureSample(scene_tex, scene_samp, uv + radial * aberration).r;
        color.g = textureSample(scene_tex, scene_samp, uv).g;
        color.b = textureSample(scene_tex, scene_samp, uv - radial * aberration).b;
    } else {
        color = textureSample(scene_tex, scene_samp, uv).rgb;
    }

    // Eye-adaptation exposure.
    if params.adapt_enabled != 0u {
        let e = clamp(exposure_buf[0], params.adapt_min, params.adapt_max);
        color *= e;
    }

    // Midpoint-pivot contrast. Values below the pivot move darker while values
    // above it move brighter, so the control increases separation instead of
    // dimming the whole image like a gamma power curve.
    if params.contrast != 1.0 {
        let pivot = vec3<f32>(0.5);
        color = max((color - pivot) * params.contrast + pivot, vec3<f32>(0.0));
    }

    // ── Underwater color grade ──────────────────────────────────────────────
    // Clearly-legible desaturation + cool blue-green push - should read
    // immediately as "in water", not need to be pointed out.
    if params.underwater_fraction > 0.0 {
        let grey = luma(color);
        let desaturated = mix(color, vec3<f32>(grey), 0.55 * params.underwater_fraction);
        let tinted = desaturated * vec3<f32>(0.55, 0.85, 1.15);
        color = mix(color, tinted, params.underwater_fraction * 0.85);
    }

    // ── World sphere glassy tint ────────────────────────────────────────────
    // Very slight cool brightening near the boundary - the "glass" material
    // already established for the sphere's echo/ambient audio.
    if params.boundary_fraction > 0.0 {
        let glassy = color * vec3<f32>(1.02, 1.03, 1.06) + vec3<f32>(0.012, 0.012, 0.016);
        color = mix(color, glassy, params.boundary_fraction * 0.35);
    }

    // ── Crossing flashes ────────────────────────────────────────────────────
    // Brief additive brighten on top of everything else, gone within
    // CROSSING_PULSE_DECAY_SECONDS.
    color += vec3<f32>(0.85, 0.92, 1.0) * params.boundary_crossing_pulse * 0.12;
    color += vec3<f32>(0.6, 0.85, 0.95) * params.water_crossing_pulse * 0.10;

    // Soft vignette - mostly a water-crossing cue (a lens getting momentarily
    // "wet"), with a faint constant presence while fully underwater.
    let vignette_strength = 0.22 * params.underwater_fraction + 0.18 * params.water_crossing_pulse;
    if vignette_strength > 0.0 {
        // 1.0 at center fading to 0.0 at the edges - smoothstep needs
        // edge0 < edge1 (reversed edges are implementation-defined), so ramp
        // up toward the edges first and invert rather than passing edges
        // backwards.
        let vignette = 1.0 - smoothstep(0.35, 0.85, dist);
        color *= mix(1.0, vignette, vignette_strength);
    }

    return vec4<f32>(color, 1.0);
}
