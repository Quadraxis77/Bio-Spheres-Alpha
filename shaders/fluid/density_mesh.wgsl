// Simple density mesh rendering shader
// Renders isosurface extracted by GPU surface nets
// With shadow field sampling from light field system

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct RenderParams {
    base_color: vec3<f32>,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    fresnel: f32,
    fresnel_power: f32,
    rim: f32,
    reflection: f32,
    alpha: f32,
    time: f32,
    wave_height: f32,
    wave_speed: f32,
    noise_scale: f32,
    noise_octaves: f32,
    noise_lacunarity: f32,
    noise_persistence: f32,
    reflection_brightness: f32,
    light_dir: vec3<f32>,
    waterline_alpha: f32,
    gravity: vec3<f32>,
    gravity_mode: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> params: RenderParams;

// Shadow field data (from light field system)
struct ShadowFieldParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    shadow_strength: f32,
    shadow_enabled: u32,
    shadow_quality: f32,
    caustic_intensity: f32,
    caustic_scale: f32,
    caustic_speed: f32,
    time: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    moss_parallax_depth: f32,
    moss_scale: f32,
    // Moss appearance parameters
    moss_noise_type: u32,
    moss_noise_frequency: f32,
    moss_noise_lacunarity: f32,
    moss_height_sharpness_low: f32,
    moss_height_sharpness_high: f32,
    moss_bump_strength: f32,
    moss_color_dark_r: f32,
    moss_color_dark_g: f32,
    moss_color_dark_b: f32,
    moss_color_bright_r: f32,
    moss_color_bright_g: f32,
    moss_color_bright_b: f32,
    _pad_moss_0: f32,
    _pad_moss_1: f32,
}

@group(1) @binding(0) var<uniform> shadow_params: ShadowFieldParams;
@group(1) @binding(1) var light_field_tex: texture_3d<f32>;
@group(1) @binding(2) var light_color_field_tex: texture_3d<f32>;
@group(1) @binding(3) var light_field_sampler: sampler;

// Environment cubemap for reflections
@group(2) @binding(0) var env_cubemap: texture_cube<f32>;
@group(2) @binding(1) var env_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) fluid_type: f32,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_dir: vec3<f32>,
    @location(3) fluid_type: f32,
    @location(4) geometry_normal: vec3<f32>,  // unperturbed normal for reflection
}

// --- Shadow field sampling ---
fn world_to_light_uvw(world_pos: vec3<f32>) -> vec3<f32> {
    let res = f32(shadow_params.grid_resolution);
    return vec3<f32>(
        (world_pos.x - shadow_params.grid_origin_x) / (shadow_params.cell_size * res),
        (world_pos.y - shadow_params.grid_origin_y) / (shadow_params.cell_size * res),
        (world_pos.z - shadow_params.grid_origin_z) / (shadow_params.cell_size * res),
    );
}

fn light_uvw_in_bounds(uvw: vec3<f32>) -> bool {
    return all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0));
}

fn sample_light_field(world_pos: vec3<f32>) -> f32 {
    if (shadow_params.shadow_enabled == 0u) {
        return 1.0;
    }
    let uvw = world_to_light_uvw(world_pos);
    if (!light_uvw_in_bounds(uvw)) {
        return 1.0;
    }
    return textureSampleLevel(light_field_tex, light_field_sampler, uvw, 0.0).r;
}

fn sample_light_color_field(world_pos: vec3<f32>) -> vec3<f32> {
    let fallback = vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);
    if (shadow_params.shadow_enabled == 0u) {
        return fallback;
    }
    let uvw = world_to_light_uvw(world_pos);
    if (!light_uvw_in_bounds(uvw)) {
        return fallback;
    }
    let sample = textureSampleLevel(light_color_field_tex, light_field_sampler, uvw, 0.0);
    if (sample.w <= 0.0001) {
        return fallback;
    }
    return sample.rgb;
}
// --- 3D Perlin noise implementation ---

// Hash function for gradient lookup
fn hash3(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123) * 2.0 - 1.0;
}

// Quintic smoothstep for smoother interpolation (avoids second-derivative discontinuities)
fn quintic(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Quintic derivative for analytical gradient
fn quintic_deriv(t: vec3<f32>) -> vec3<f32> {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

// 3D Perlin noise returning value and analytical gradient
fn perlin_noise_3d(p: vec3<f32>) -> vec4<f32> {
    let pi = floor(p);
    let pf = p - pi;

    let u = quintic(pf);
    let du = quintic_deriv(pf);

    // 8 corner gradients
    let g000 = hash3(pi + vec3<f32>(0.0, 0.0, 0.0));
    let g100 = hash3(pi + vec3<f32>(1.0, 0.0, 0.0));
    let g010 = hash3(pi + vec3<f32>(0.0, 1.0, 0.0));
    let g110 = hash3(pi + vec3<f32>(1.0, 1.0, 0.0));
    let g001 = hash3(pi + vec3<f32>(0.0, 0.0, 1.0));
    let g101 = hash3(pi + vec3<f32>(1.0, 0.0, 1.0));
    let g011 = hash3(pi + vec3<f32>(0.0, 1.0, 1.0));
    let g111 = hash3(pi + vec3<f32>(1.0, 1.0, 1.0));

    // Dot products with distance vectors
    let d000 = dot(g000, pf - vec3<f32>(0.0, 0.0, 0.0));
    let d100 = dot(g100, pf - vec3<f32>(1.0, 0.0, 0.0));
    let d010 = dot(g010, pf - vec3<f32>(0.0, 1.0, 0.0));
    let d110 = dot(g110, pf - vec3<f32>(1.0, 1.0, 0.0));
    let d001 = dot(g001, pf - vec3<f32>(0.0, 0.0, 1.0));
    let d101 = dot(g101, pf - vec3<f32>(1.0, 0.0, 1.0));
    let d011 = dot(g011, pf - vec3<f32>(0.0, 1.0, 1.0));
    let d111 = dot(g111, pf - vec3<f32>(1.0, 1.0, 1.0));

    // Trilinear interpolation
    let k0 = d000;
    let k1 = d100 - d000;
    let k2 = d010 - d000;
    let k3 = d001 - d000;
    let k4 = d000 - d100 - d010 + d110;
    let k5 = d000 - d010 - d001 + d011;
    let k6 = d000 - d100 - d001 + d101;
    let k7 = -d000 + d100 + d010 - d110 + d001 - d101 - d011 + d111;

    let value = k0 + k1 * u.x + k2 * u.y + k3 * u.z
              + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x
              + k7 * u.x * u.y * u.z;

    // Analytical gradient
    let grad = vec3<f32>(
        du.x * (k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z),
        du.y * (k2 + k5 * u.z + k4 * u.x + k7 * u.x * u.z),
        du.z * (k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y)
    );

    return vec4<f32>(grad, value);
}

// Fractal Brownian Motion with analytical gradient (multi-octave Perlin noise)
fn fbm_3d(p: vec3<f32>, octaves: i32, lacunarity: f32, persistence: f32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_amp = 0.0;

    for (var i = 0; i < octaves; i++) {
        let n = perlin_noise_3d(p * frequency);
        sum += n * amplitude;
        max_amp += amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }

    return sum / max_amp;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let n = normalize(in.normal);
    let wp = in.position;
    let t = params.time * params.wave_speed;
    let height = params.wave_height;
    let scale = params.noise_scale;
    let octaves = i32(params.noise_octaves);
    let lac = params.noise_lacunarity;
    let pers = params.noise_persistence;
    
    // Sample 3D Perlin noise with time-animated coordinates
    // The noise input spans multiple voxels based on noise_scale
    // (ice no longer flows through this shader - it has its own mesh, see ice_mesh.wgsl)
    let noise_pos = wp * scale + vec3<f32>(t * 0.3, t * 0.17, t * 0.23);
    let noise = fbm_3d(noise_pos, octaves, lac, pers);
    let displacement = noise.w * height;

    let displaced_pos = wp + n * displacement;

    // Use analytical gradient from noise to perturb the normal
    // Scale gradient by wave_height and noise_scale for correct magnitude
    let grad_scale = height * scale;
    let noise_grad = noise.xyz * grad_scale;
    let perturbed_normal = normalize(n - noise_grad + n * dot(noise_grad, n));

    out.world_position = displaced_pos;
    out.world_normal = perturbed_normal;
    out.geometry_normal = n;
    out.view_dir = normalize(camera.camera_pos - displaced_pos);
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    out.fluid_type = in.fluid_type;
    
    return out;
}

// Fluid type colors (ice renders via its own shader, see ice_mesh.wgsl)
const WATER_COLOR: vec3<f32> = vec3<f32>(0.2, 0.5, 0.9);
const STEAM_COLOR: vec3<f32> = vec3<f32>(0.9, 0.92, 0.95);

fn get_fluid_color(fluid_type: f32) -> vec3<f32> {
    let ft = u32(fluid_type + 0.5);
    if ft == 1u { return WATER_COLOR; }
    if ft == 3u { return STEAM_COLOR; }
    return vec3<f32>(0.5, 0.5, 0.5); // Default gray
}

fn get_fluid_alpha(fluid_type: f32, base_alpha: f32) -> f32 {
    let ft = u32(fluid_type + 0.5);
    if ft == 1u { return base_alpha; } // Water - use user setting
    if ft == 3u { return 1.0; } // Steam - fully opaque
    return base_alpha;
}

struct FragmentOutput {
    @location(0) accum: vec4<f32>,   // WBOIT accumulation (premultiplied color * weight, weight)
    @location(1) revealage: vec4<f32>, // WBOIT revealage (1 - alpha product)
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(in.view_dir);
    
    // Get base color from fluid type
    let base_color = get_fluid_color(in.fluid_type);
    let alpha = get_fluid_alpha(in.fluid_type, params.alpha);
    let ft = u32(in.fluid_type + 0.5);
    
    // Use shadow field light direction if available, fall back to render params
    let light_dir = normalize(vec3<f32>(shadow_params.light_dir_x, shadow_params.light_dir_y, shadow_params.light_dir_z));
    
    // Sample shadow field at the water surface position
    // Offset slightly along normal to avoid self-shadowing artifacts
    let offset_distance = shadow_params.cell_size * 2.0;
    let shadow_sample_pos = in.world_position + normal * offset_distance;
    let light_value = sample_light_field(shadow_sample_pos);
    let shadow = mix(1.0, light_value, shadow_params.shadow_strength);
    let sun_color = sample_light_color_field(shadow_sample_pos);
    let sampled_light = sun_color * shadow;
    let sampled_light_strength = clamp(max(max(sampled_light.r, sampled_light.g), sampled_light.b), 0.0, 1.0);
    
    // Diffuse (Lambert)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * params.diffuse;
    
    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let specular = pow(n_dot_h, params.shininess) * params.specular;
    
    // Fresnel effect (unused - kept as variable for rim light compatibility)
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), params.fresnel_power) * params.fresnel;
    
    // Rim lighting
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0) * params.rim;
    
    // Combine lighting with shadow applied to direct illumination
    let ambient_light = select(params.ambient, 0.0, ft == 1u);
    let direct_light = diffuse * shadow;
    let lighting = ambient_light + direct_light;
    var final_color = base_color * sun_color * lighting;
    
    // Add specular (also shadowed - no specular highlight in shadow)
    final_color += sun_color * specular * shadow;
    
    // Sample environment cubemap for mirror reflection
    // Use geometry normal (not wave-perturbed) for stable, coherent reflections
    let reflect_normal = normalize(in.geometry_normal);
    let reflect_dir = reflect(-view_dir, reflect_normal);
    let env_color = textureSample(env_cubemap, env_sampler, reflect_dir).rgb;
    // Boost reflection brightness - cave walls are dark, need to be visible in reflection
    let boosted_env = env_color * params.reflection_brightness * sampled_light_strength;
    // Waterline: tris facing against gravity (upward normals) are the water surface.
    // Faces below the waterline (sides + bottom) get transparent, non-reflective treatment.
    // Radial gravity: reflective faces are those aligned with the radial direction
    // (pointing toward or away from the origin), so use abs(dot).
    // Axial gravity: reflective faces are those pointing against gravity (upward).
    let face_up = select(
        dot(normal, normalize(params.gravity)),
        abs(dot(normal, normalize(in.world_position))),
        params.gravity_mode == 3u,
    );
    let submerged = smoothstep(0.15, -0.15, face_up); // 0 at waterline/top, 1 below

    // Reflections: full on waterline surface, none below.
    let reflection_contrib = mix(params.reflection, 0.0, submerged);
    final_color = mix(final_color, boosted_env, reflection_contrib);

    // Rim light: suppressed below waterline.
    let rim_color = mix(vec3<f32>(0.7, 0.85, 1.0), base_color, 0.5);
    final_color += rim_color * rim * sampled_light_strength * (1.0 - submerged);

    // Subsurface scattering (water only, attenuated by shadow).
    if ft == 1u {
        let sss = pow(max(dot(view_dir, -light_dir), 0.0), 2.0) * 0.15;
        final_color += vec3<f32>(0.0, 0.4, 0.6) * sss * sampled_light_strength;
    }

    // Alpha: waterline/above follows the user transparency setting exactly.
    // Reflection changes color/specular response, not coverage/opacity.
    let surface_alpha = alpha;
    let final_alpha = mix(surface_alpha, params.waterline_alpha, submerged);

    // WBOIT weight function (McGuire & Bavoil 2013)
    let depth = in.clip_position.z;
    let w = final_alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(depth / 200.0, 4.0))));

    var out: FragmentOutput;
    out.accum = vec4<f32>(final_color * w, w);
    out.revealage = vec4<f32>(final_alpha, 0.0, 0.0, final_alpha);
    return out;
}
