// Depth of Field Blur Shader
//
// Full-screen post-process that applies a variable-radius disc blur
// based on the circle of confusion (CoC) derived from scene depth.
// Objects at the focal distance appear sharp; objects closer or farther
// get progressively blurred.
//
// Uses a Poisson-disc sampling pattern for smooth, artifact-free bokeh.

struct CameraUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
}

struct DofParams {
    // Focal plane distance from camera (world units)
    focal_distance: f32,
    // Range around focal distance that stays sharp (world units)
    focal_range: f32,
    // Maximum blur radius in pixels (at full-res)
    max_blur_radius: f32,
    // Blur intensity multiplier
    blur_strength: f32,
    // Screen dimensions
    screen_width: f32,
    screen_height: f32,
    // Near/far clip for depth linearization
    near_clip: f32,
    far_clip: f32,
}

// Group 0: Camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Group 1: DoF data
@group(1) @binding(0)
var<uniform> dof_params: DofParams;
@group(1) @binding(1)
var scene_tex: texture_2d<f32>;
@group(1) @binding(2)
var scene_samp: sampler;
@group(1) @binding(3)
var depth_tex: texture_depth_2d;
@group(1) @binding(4)
var depth_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// Linearize reverse-Z depth buffer value to view-space distance
fn linearize_depth(d: f32) -> f32 {
    let near = dof_params.near_clip;
    let far = dof_params.far_clip;
    // Standard perspective depth: z_ndc = (far * (z - near)) / (z * (far - near))
    // Solving for z: z = near * far / (far - z_ndc * (far - near))
    return near * far / (far - d * (far - near));
}

// Compute circle of confusion (0 = sharp, 1 = max blur)
fn compute_coc(linear_depth: f32) -> f32 {
    let dist_from_focal = abs(linear_depth - dof_params.focal_distance);
    let half_range = dof_params.focal_range * 0.5;
    // Smooth transition from sharp to blurred
    let coc = smoothstep(0.0, 1.0, (dist_from_focal - half_range) / max(dof_params.focal_distance * 0.5, 0.1));
    return clamp(coc * dof_params.blur_strength, 0.0, 1.0);
}

// 16-sample Poisson disc for smooth bokeh
const POISSON_SAMPLES: u32 = 16u;
const POISSON_DISC: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = vec2<f32>(1.0 / dof_params.screen_width, 1.0 / dof_params.screen_height);

    // Sample center pixel
    let center_color = textureSample(scene_tex, scene_samp, in.uv);
    let center_depth_raw = textureSample(depth_tex, depth_samp, in.uv);
    let center_depth = linearize_depth(center_depth_raw);
    let center_coc = compute_coc(center_depth);

    // Early out if center pixel is sharp
    if (center_coc < 0.01) {
        return center_color;
    }

    // Blur radius in pixels, scaled by CoC
    let blur_radius = center_coc * dof_params.max_blur_radius;

    // Accumulate weighted samples using Poisson disc
    var color_accum = center_color.rgb;
    var weight_accum = 1.0;

    for (var i = 0u; i < POISSON_SAMPLES; i++) {
        let offset = POISSON_DISC[i] * blur_radius * texel_size;
        let sample_uv = in.uv + offset;

        // Clamp to valid UV range
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));

        let sample_color = textureSample(scene_tex, scene_samp, clamped_uv);
        let sample_depth_raw = textureSample(depth_tex, depth_samp, clamped_uv);
        let sample_depth = linearize_depth(sample_depth_raw);
        let sample_coc = compute_coc(sample_depth);

        // Weight: prefer blurring background over foreground to avoid halo artifacts
        // If the sample is sharper than center, reduce its contribution
        let w = smoothstep(0.0, 1.0, sample_coc);

        color_accum += sample_color.rgb * w;
        weight_accum += w;
    }

    let blurred = color_accum / weight_accum;

    // Mix between sharp and blurred based on center CoC
    let final_color = mix(center_color.rgb, blurred, center_coc);

    return vec4<f32>(final_color, 1.0);
}
