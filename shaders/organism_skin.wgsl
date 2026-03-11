// Organism Skin Render Shader
//
// Renders the isosurface extracted from the organism density field.
// Appearance: organic, translucent biological membrane with warm inner tones,
// cool rim highlights, and a subtle subsurface-scattering approximation.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

// Must match OrganismSkinParams in Rust (80 bytes = 5 × vec4)
struct SkinParams {
    // vec4 0
    base_r: f32, base_g: f32, base_b: f32, ambient: f32,
    // vec4 1
    diffuse: f32, specular: f32, shininess: f32, fresnel: f32,
    // vec4 2
    fresnel_power: f32, alpha: f32, time: f32, sss_strength: f32,
    // vec4 3
    sss_r: f32, sss_g: f32, sss_b: f32, rim_strength: f32,
    // vec4 4 — light direction (world space, pointing toward light) + padding
    light_dir_x: f32, light_dir_y: f32, light_dir_z: f32, _pad: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> skin: SkinParams;

struct VertexInput {
    @location(0) position:   vec3<f32>,
    @location(1) fluid_type: f32,   // unused by skin shader, present for buffer compat
    @location(2) normal:     vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       world_pos:   vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       view_dir:    vec3<f32>,
}

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos    = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.world_pos   = v.position;
    out.world_normal = normalize(v.normal);
    out.view_dir    = normalize(camera.camera_pos - v.position);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lighting helpers
// ─────────────────────────────────────────────────────────────────────────────

fn phong_diffuse(n: vec3<f32>, l: vec3<f32>) -> f32 {
    return max(dot(n, l), 0.0);
}

fn phong_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, shininess: f32) -> f32 {
    let r = reflect(-l, n);
    return pow(max(dot(r, v), 0.0), shininess);
}

// Schlick Fresnel approximation
fn fresnel_schlick(v: vec3<f32>, n: vec3<f32>, f0: f32) -> f32 {
    let cos_theta = clamp(dot(v, n), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

// Rim lighting: strong at grazing angles, zero facing the camera
fn rim_light(v: vec3<f32>, n: vec3<f32>) -> f32 {
    return pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 3.5);
}

// Cheap SSS approximation: thickness proxy from normal·view_dir (back-lit halo)
fn sss_approx(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, strength: f32) -> f32 {
    // Light transmitted through the surface appears where the normal faces away from camera
    let back_diffuse = max(-dot(n, l) + 0.3, 0.0);  // light bleeding from behind
    let view_factor  = pow(max(dot(v, -l), 0.0), 2.0);  // forward scatter
    return strength * back_diffuse * view_factor;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(in.view_dir);
    let l = normalize(vec3<f32>(skin.light_dir_x, skin.light_dir_y, skin.light_dir_z));

    let base_color = vec3<f32>(skin.base_r, skin.base_g, skin.base_b);
    let sss_color  = vec3<f32>(skin.sss_r,  skin.sss_g,  skin.sss_b);

    // ── Diffuse ──────────────────────────────────────────────────────────────
    let diff = phong_diffuse(n, l);

    // ── Specular ─────────────────────────────────────────────────────────────
    let spec = phong_specular(n, v, l, skin.shininess);

    // ── Fresnel rim ──────────────────────────────────────────────────────────
    let fres = fresnel_schlick(v, n, skin.fresnel);
    let rim  = rim_light(v, n) * skin.rim_strength;

    // ── Subsurface scattering approximation ──────────────────────────────────
    let sss = sss_approx(n, l, v, skin.sss_strength);

    // ── Combine ──────────────────────────────────────────────────────────────
    var color = base_color * (skin.ambient + skin.diffuse * diff)
              + base_color * pow(fres, skin.fresnel_power) * 0.4  // fresnel tint
              + vec3<f32>(1.0) * skin.specular * spec             // white specular
              + vec3<f32>(1.0, 1.0, 1.0) * rim                   // cool rim
              + sss_color * sss;                                   // warm sss bleed

    // ── Alpha: more opaque toward surface, translucent at glancing angles ────
    let alpha = clamp(skin.alpha + fres * 0.25, 0.0, 1.0);

    return vec4<f32>(color, alpha);
}
