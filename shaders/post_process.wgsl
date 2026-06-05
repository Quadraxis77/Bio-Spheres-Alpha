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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

// Smooth filmic S-curve contrast (keeps blacks black and whites white).
fn contrast_curve(c: vec3<f32>, contrast: f32) -> vec3<f32> {
    // Pivot around mid-grey (0.5).  contrast = 1.0 → identity.
    let lifted = (c - 0.5) * contrast + 0.5;
    // Soft-clip using a smooth curve to avoid hard clamp artefacts.
    return vec3<f32>(
        1.0 / (1.0 + exp(-10.0 * (lifted.r - 0.5))) * 0.96 + 0.02,
        1.0 / (1.0 + exp(-10.0 * (lifted.g - 0.5))) * 0.96 + 0.02,
        1.0 / (1.0 + exp(-10.0 * (lifted.b - 0.5))) * 0.96 + 0.02,
    );
}

@fragment
fn fs_tonemap(in: VOut) -> @location(0) vec4<f32> {
    var color = textureSample(scene_tex, scene_samp, in.uv).rgb;

    // Apply eye-adaptation exposure.
    if params.adapt_enabled != 0u {
        let exp = clamp(exposure_buf[0], params.adapt_min, params.adapt_max);
        color *= exp;
    }

    // Apply contrast.
    if params.contrast != 1.0 {
        color = contrast_curve(color, params.contrast);
    }

    return vec4<f32>(color, 1.0);
}
