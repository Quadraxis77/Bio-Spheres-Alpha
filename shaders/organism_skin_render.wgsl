// Organism Skin Render Shader — per-organism coloring
//
// Renders isosurface mesh extracted by organism_skin_surface_nets.wgsl.
// Each vertex carries an organism_id (in the fluid_type/organism_id field).
// The fragment shader hashes the organism_id to produce a unique hue per organism.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct SkinParams {
    // vec4 0
    base_r: f32, base_g: f32, base_b: f32, ambient: f32,
    // vec4 1
    diffuse: f32, specular: f32, shininess: f32, fresnel: f32,
    // vec4 2
    fresnel_power: f32, alpha: f32, time: f32, sss_strength: f32,
    // vec4 3
    sss_r: f32, sss_g: f32, sss_b: f32, rim_strength: f32,
    // vec4 4
    light_dir_x: f32, light_dir_y: f32, light_dir_z: f32, _pad: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> skin: SkinParams;

struct VertexInput {
    @location(0) position:    vec3<f32>,
    @location(1) organism_id: f32,
    @location(2) normal:      vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       view_dir:     vec3<f32>,
    @location(3)       org_id:       f32,
}

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.world_pos    = v.position;
    out.world_normal = normalize(v.normal);
    out.view_dir     = normalize(camera.camera_pos - v.position);
    out.org_id       = v.organism_id;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Organism-specific color from ID
// ─────────────────────────────────────────────────────────────────────────────

// Hash a u32 to a float in [0,1)
fn hash_f(x: u32) -> f32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFu) / 65536.0;
}

// Convert HSL to RGB (s and l in [0,1], h in [0,1])
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    var rgb = vec3<f32>(0.0);
    if hp < 1.0      { rgb = vec3<f32>(c, x, 0.0); }
    else if hp < 2.0 { rgb = vec3<f32>(x, c, 0.0); }
    else if hp < 3.0 { rgb = vec3<f32>(0.0, c, x); }
    else if hp < 4.0 { rgb = vec3<f32>(0.0, x, c); }
    else if hp < 5.0 { rgb = vec3<f32>(x, 0.0, c); }
    else             { rgb = vec3<f32>(c, 0.0, x); }
    let m = l - c * 0.5;
    return rgb + vec3<f32>(m);
}

// Generate a visually distinct color for an organism ID.
// Uses golden-ratio hue spacing for maximum separation.
fn organism_color(org_id: u32) -> vec3<f32> {
    let golden_ratio = 0.618033988749895;
    let hue = fract(f32(org_id) * golden_ratio);
    let sat = 0.5 + hash_f(org_id * 7u + 3u) * 0.3;  // 0.5–0.8
    let lit = 0.45 + hash_f(org_id * 13u + 7u) * 0.2; // 0.45–0.65
    return hsl_to_rgb(hue, sat, lit);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lighting (same as original organism_skin.wgsl)
// ─────────────────────────────────────────────────────────────────────────────

fn phong_diffuse(n: vec3<f32>, l: vec3<f32>) -> f32 {
    return max(dot(n, l), 0.0);
}

fn phong_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, shininess: f32) -> f32 {
    let r = reflect(-l, n);
    return pow(max(dot(r, v), 0.0), shininess);
}

fn fresnel_schlick(v: vec3<f32>, n: vec3<f32>, f0: f32) -> f32 {
    let cos_theta = clamp(dot(v, n), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn rim_light(v: vec3<f32>, n: vec3<f32>) -> f32 {
    return pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 3.5);
}

fn sss_approx(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, strength: f32) -> f32 {
    let back_diffuse = max(-dot(n, l) + 0.3, 0.0);
    let view_factor  = pow(max(dot(v, -l), 0.0), 2.0);
    return strength * back_diffuse * view_factor;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(in.view_dir);
    let l = normalize(vec3<f32>(skin.light_dir_x, skin.light_dir_y, skin.light_dir_z));

    let base_color = vec3<f32>(skin.base_r, skin.base_g, skin.base_b);

    let sss_color = vec3<f32>(skin.sss_r, skin.sss_g, skin.sss_b);

    let diff = phong_diffuse(n, l);
    let spec = phong_specular(n, v, l, skin.shininess);
    let fres = fresnel_schlick(v, n, skin.fresnel);
    let rim  = rim_light(v, n) * skin.rim_strength;
    let sss  = sss_approx(n, l, v, skin.sss_strength);

    var color = base_color * (skin.ambient + skin.diffuse * diff)
              + base_color * pow(fres, skin.fresnel_power) * 0.4
              + vec3<f32>(1.0) * skin.specular * spec
              + vec3<f32>(1.0, 1.0, 1.0) * rim
              + sss_color * sss;

    let alpha = clamp(skin.alpha + fres * 0.25, 0.0, 1.0);

    return vec4<f32>(color, alpha);
}
