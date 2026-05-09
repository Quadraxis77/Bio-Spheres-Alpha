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
    _pad: f32,
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
@group(1) @binding(1) var<storage, read> light_field: array<f32>;

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

// Sample light field at world position with trilinear interpolation
fn sample_light_field(world_pos: vec3<f32>) -> f32 {
    if (shadow_params.shadow_enabled == 0u) {
        return 1.0;
    }
    let res = shadow_params.grid_resolution;
    let fres = f32(res);

    let gx = (world_pos.x - shadow_params.grid_origin_x) / shadow_params.cell_size - 0.5;
    let gy = (world_pos.y - shadow_params.grid_origin_y) / shadow_params.cell_size - 0.5;
    let gz = (world_pos.z - shadow_params.grid_origin_z) / shadow_params.cell_size - 0.5;

    if (gx < -0.5 || gx >= fres - 0.5 ||
        gy < -0.5 || gy >= fres - 0.5 ||
        gz < -0.5 || gz >= fres - 0.5) {
        return 1.0;
    }

    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let iz = i32(floor(gz));
    let fx = gx - floor(gx);
    let fy = gy - floor(gy);
    let fz = gz - floor(gz);

    let ires = i32(res);
    let x0 = u32(clamp(ix, 0, ires - 1));
    let x1 = u32(clamp(ix + 1, 0, ires - 1));
    let y0 = u32(clamp(iy, 0, ires - 1));
    let y1 = u32(clamp(iy + 1, 0, ires - 1));
    let z0 = u32(clamp(iz, 0, ires - 1));
    let z1 = u32(clamp(iz + 1, 0, ires - 1));

    let c000 = light_field[x0 + y0 * res + z0 * res * res];
    let c100 = light_field[x1 + y0 * res + z0 * res * res];
    let c010 = light_field[x0 + y1 * res + z0 * res * res];
    let c110 = light_field[x1 + y1 * res + z0 * res * res];
    let c001 = light_field[x0 + y0 * res + z1 * res * res];
    let c101 = light_field[x1 + y0 * res + z1 * res * res];
    let c011 = light_field[x0 + y1 * res + z1 * res * res];
    let c111 = light_field[x1 + y1 * res + z1 * res * res];

    let c00 = mix(c000, c100, fx);
    let c10 = mix(c010, c110, fx);
    let c01 = mix(c001, c101, fx);
    let c11 = mix(c011, c111, fx);
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);

    return mix(c0, c1, fz);
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

// Fluid type colors
const WATER_COLOR: vec3<f32> = vec3<f32>(0.2, 0.5, 0.9);
const LAVA_COLOR: vec3<f32> = vec3<f32>(1.0, 0.3, 0.05);
const STEAM_COLOR: vec3<f32> = vec3<f32>(0.9, 0.92, 0.95);

fn get_fluid_color(fluid_type: f32) -> vec3<f32> {
    let ft = u32(fluid_type + 0.5);
    if ft == 1u { return WATER_COLOR; }
    if ft == 2u { return LAVA_COLOR; }
    if ft == 3u { return STEAM_COLOR; }
    return vec3<f32>(0.5, 0.5, 0.5); // Default gray
}

fn get_fluid_alpha(fluid_type: f32, base_alpha: f32) -> f32 {
    let ft = u32(fluid_type + 0.5);
    if ft == 1u { return base_alpha; } // Water - use user setting
    if ft == 2u { return min(base_alpha + 0.1, 1.0); } // Lava - slightly more opaque
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
    
    // Use shadow field light direction if available, fall back to render params
    let light_dir = normalize(vec3<f32>(shadow_params.light_dir_x, shadow_params.light_dir_y, shadow_params.light_dir_z));
    let sun_color = vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);
    
    // Sample shadow field at the water surface position
    // Offset slightly along normal to avoid self-shadowing artifacts
    let offset_distance = shadow_params.cell_size * 2.0;
    let shadow_sample_pos = in.world_position + normal * offset_distance;
    let light_value = sample_light_field(shadow_sample_pos);
    let shadow = mix(1.0, light_value, shadow_params.shadow_strength);
    
    // Diffuse (Lambert)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * params.diffuse;
    
    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let specular = pow(n_dot_h, params.shininess) * params.specular;
    
    // Fresnel effect (unused — kept as variable for rim light compatibility)
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), params.fresnel_power) * params.fresnel;
    
    // Rim lighting
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0) * params.rim;
    
    // Combine lighting with shadow applied to direct illumination
    let ambient_light = params.ambient;
    let direct_light = diffuse * shadow;
    let lighting = ambient_light + direct_light;
    var final_color = base_color * sun_color * lighting;
    
    // Add specular (also shadowed — no specular highlight in shadow)
    final_color += sun_color * specular * shadow;
    
    // Lava emissive glow (unaffected by shadow)
    let ft = u32(in.fluid_type + 0.5);
    if ft == 2u {
        final_color += LAVA_COLOR * 0.5;
    }
    
    // Sample environment cubemap for mirror reflection
    // Use geometry normal (not wave-perturbed) for stable, coherent reflections
    let reflect_normal = normalize(in.geometry_normal);
    let reflect_dir = reflect(-view_dir, reflect_normal);
    let env_color = textureSample(env_cubemap, env_sampler, reflect_dir).rgb;
    // Boost reflection brightness — cave walls are dark, need to be visible in reflection
    let boosted_env = env_color * params.reflection_brightness;
    // Mirror blend: at reflection=1.0, surface is pure cubemap; at 0.0, pure lit surface
    final_color = mix(final_color, boosted_env, params.reflection);
    
    // Rim light (colored by fluid type, attenuated by shadow)
    let rim_color = mix(vec3<f32>(0.7, 0.85, 1.0), base_color, 0.5);
    final_color += rim_color * rim * mix(0.3, 1.0, shadow);
    
    // Subsurface scattering (water only, attenuated by shadow)
    if ft == 1u {
        let sss = pow(max(dot(view_dir, -light_dir), 0.0), 2.0) * 0.15;
        final_color += vec3<f32>(0.0, 0.4, 0.6) * sss * shadow;
    }
    
    // Mirror reflection makes the surface opaque
    let final_alpha = clamp(alpha + params.reflection, alpha, 1.0);

    // WBOIT weight function (McGuire & Bavoil 2013)
    let depth = in.clip_position.z;
    let w = final_alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(depth / 200.0, 4.0))));

    var out: FragmentOutput;
    out.accum = vec4<f32>(final_color * w, w);
    out.revealage = vec4<f32>(final_alpha, 0.0, 0.0, final_alpha);
    return out;
}
