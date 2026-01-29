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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    out.world_position = in.position;
    out.world_normal = normalize(in.normal);
    out.view_dir = normalize(camera.camera_pos - in.position);
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
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
