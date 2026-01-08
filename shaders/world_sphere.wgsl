// World Sphere Shader
// Renders a transparent boundary sphere with proper lighting
// Uses front-face culling to render the inside of the sphere
// Uses the same directional lighting as the cell renderer

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

struct LightingUniform {
    light_direction: vec3<f32>,  // Direction TO the light (normalized)
    _padding1: f32,
    light_color: vec3<f32>,
    _padding2: f32,
    ambient_color: vec3<f32>,
    _padding3: f32,
};

struct WorldSphereParams {
    // Base color (RGBA) - sRGB color space
    base_color: vec4<f32>,
    // Emissive color (RGB, linear) + unused
    emissive: vec4<f32>,
    // Material properties
    radius: f32,
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    // Padding
    _padding: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> lighting: LightingUniform;

@group(1) @binding(0)
var<uniform> params: WorldSphereParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Scale vertex by sphere radius
    let world_pos = in.position * params.radius;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_position = world_pos;
    // Invert normal for inside rendering (front-face culling)
    out.world_normal = -normalize(in.normal);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.camera_pos - in.world_position);
    
    // Use scene lighting (same as cell renderer)
    let light_dir = normalize(lighting.light_direction);
    let light_color = lighting.light_color;
    let ambient_color = lighting.ambient_color;
    
    // Base color (already in sRGB, keep as-is for now)
    let base_color = params.base_color.rgb;
    let alpha = params.base_color.a;
    
    // Diffuse lighting (Lambert) - same as cell shader
    let n_dot_l = max(0.0, dot(normal, light_dir));
    let diffuse = n_dot_l;
    
    // Specular (Blinn-Phong)
    let half_vec = normalize(light_dir + view_dir);
    let n_dot_h = max(0.0, dot(normal, half_vec));
    let roughness = params.perceptual_roughness;
    let specular_power = mix(128.0, 8.0, roughness); // Higher power = sharper highlights
    let specular = pow(n_dot_h, specular_power) * (1.0 - roughness) * 0.5;
    
    // Fresnel rim effect (view-dependent)
    let n_dot_v = max(0.0, dot(normal, view_dir));
    let fresnel = pow(1.0 - n_dot_v, 3.0) * params.reflectance;
    
    // Combine lighting
    var color = base_color * ambient_color; // Ambient
    color += base_color * diffuse * light_color; // Diffuse
    color += light_color * specular; // Specular
    color += base_color * fresnel * 0.5; // Fresnel rim
    
    // Add emissive
    color += params.emissive.rgb;
    
    // Edge glow - stronger emissive at grazing angles
    let edge_factor = 1.0 - n_dot_v;
    color += params.emissive.rgb * edge_factor * edge_factor * 2.0;
    
    // Alpha with edge enhancement
    var final_alpha = alpha;
    final_alpha += edge_factor * edge_factor * 0.3;
    final_alpha = clamp(final_alpha, 0.0, 0.85);
    
    return vec4<f32>(color, final_alpha);
}
