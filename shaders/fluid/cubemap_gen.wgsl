// Procedural cubemap generation compute shader
// Generates a static environment cubemap for water reflections
// Each face is 256×256 pixels, written to a 2D texture array with 6 layers

struct CubemapParams {
    light_dir: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    sun_angular_radius: f32,
    face_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: CubemapParams;
@group(0) @binding(1) var output_texture: texture_storage_2d_array<rgba16float, write>;

// Convert face index + UV to a world-space direction
fn face_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    // Map UV from [0,1] to [-1,1]
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    // Standard cubemap face directions (OpenGL convention)
    switch face {
        // +X
        case 0u: { return normalize(vec3<f32>( 1.0, -v,   -u)); }
        // -X
        case 1u: { return normalize(vec3<f32>(-1.0, -v,    u)); }
        // +Y
        case 2u: { return normalize(vec3<f32>( u,    1.0,  v)); }
        // -Y
        case 3u: { return normalize(vec3<f32>( u,   -1.0, -v)); }
        // +Z
        case 4u: { return normalize(vec3<f32>( u,   -v,    1.0)); }
        // -Z
        case 5u: { return normalize(vec3<f32>(-u,   -v,   -1.0)); }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

// Procedural sky color based on direction
fn sky_color(dir: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    let up = dir.y;

    // Deep ocean blue at the bottom, lighter blue at horizon, pale blue at zenith
    let deep_color = vec3<f32>(0.02, 0.05, 0.15);
    let horizon_color = vec3<f32>(0.35, 0.55, 0.75);
    let zenith_color = vec3<f32>(0.15, 0.35, 0.65);

    var color: vec3<f32>;
    if up < 0.0 {
        // Below horizon — dark ocean/abyss
        let t = clamp(-up, 0.0, 1.0);
        color = mix(horizon_color * 0.4, deep_color, t * t);
    } else {
        // Above horizon — sky gradient
        let t = clamp(up, 0.0, 1.0);
        color = mix(horizon_color, zenith_color, sqrt(t));
    }

    // Atmospheric scattering near sun
    let sun_dot = max(dot(dir, light_dir), 0.0);
    let scatter = pow(sun_dot, 8.0) * 0.3;
    let scatter_color = vec3<f32>(1.0, 0.85, 0.6);
    color += scatter_color * scatter;

    // Horizon glow
    let horizon_glow = exp(-abs(up) * 8.0) * 0.15;
    let glow_tint = mix(vec3<f32>(0.6, 0.7, 0.85), vec3<f32>(1.0, 0.8, 0.5), sun_dot * sun_dot);
    color += glow_tint * horizon_glow;

    return color;
}

// Sun disc and corona
fn sun_contribution(dir: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    let cos_angle = dot(dir, light_dir);
    let sun_radius = params.sun_angular_radius;

    // Sun disc
    let disc = smoothstep(cos(sun_radius * 1.5), cos(sun_radius * 0.5), cos_angle);
    let sun = params.sun_color * disc * params.sun_intensity;

    // Corona
    let corona_angle = acos(clamp(cos_angle, -1.0, 1.0));
    let corona = exp(-corona_angle * corona_angle / (sun_radius * sun_radius * 16.0));
    let corona_color = params.sun_color * corona * params.sun_intensity * 0.3;

    return sun + corona_color;
}

@compute @workgroup_size(8, 8, 1)
fn generate_cubemap(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let face = global_id.z;
    let pixel = vec2<u32>(global_id.x, global_id.y);
    let size = params.face_size;

    if pixel.x >= size || pixel.y >= size || face >= 6u {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / f32(size);
    let dir = face_uv_to_direction(face, uv);
    let light_dir = normalize(params.light_dir);

    var color = sky_color(dir, light_dir);
    color += sun_contribution(dir, light_dir);

    // Tone mapping (simple Reinhard)
    color = color / (color + vec3<f32>(1.0));

    textureStore(output_texture, vec2<i32>(pixel), i32(face), vec4<f32>(color, 1.0));
}
