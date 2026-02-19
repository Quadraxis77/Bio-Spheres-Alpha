// Simple density mesh rendering shader
// Renders isosurface extracted by GPU surface nets

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
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> params: RenderParams;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(in.view_dir);
    
    // Get base color from fluid type
    let base_color = get_fluid_color(in.fluid_type);
    let alpha = get_fluid_alpha(in.fluid_type, params.alpha);
    
    // Light direction
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // Diffuse (Lambert)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * params.diffuse;
    
    // Specular (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let specular = pow(n_dot_h, params.shininess) * params.specular;
    
    // Fresnel effect
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), params.fresnel_power) * params.fresnel;
    
    // Rim lighting
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0) * params.rim;
    
    // Combine lighting
    let lighting = params.ambient + diffuse;
    var final_color = base_color * lighting;
    
    // Add specular
    final_color += vec3<f32>(1.0) * specular;
    
    // Lava emissive glow
    let ft = u32(in.fluid_type + 0.5);
    if ft == 2u {
        // Lava glows from within
        final_color += LAVA_COLOR * 0.5;
    }
    
    // Add reflection approximation
    let sky_color = vec3<f32>(0.6, 0.8, 1.0);
    let reflect_dir = reflect(-view_dir, normal);
    let sky_factor = max(reflect_dir.y, 0.0);
    let reflection = mix(vec3<f32>(0.9, 0.95, 1.0), sky_color, sky_factor);
    final_color = mix(final_color, reflection, fresnel * params.reflection);
    
    // Rim light (colored by fluid type)
    let rim_color = mix(vec3<f32>(0.7, 0.85, 1.0), base_color, 0.5);
    final_color += rim_color * rim;
    
    // Subsurface scattering (water only)
    if ft == 1u {
        let sss = pow(max(dot(view_dir, -light_dir), 0.0), 2.0) * 0.15;
        final_color += vec3<f32>(0.0, 0.4, 0.6) * sss;
    }
    
    return vec4<f32>(final_color, alpha);
}
