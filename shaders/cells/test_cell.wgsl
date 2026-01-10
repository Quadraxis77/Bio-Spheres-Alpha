// Test Cell Appearance Shader
//
// Basic cell rendering shader for Test cell type.
// Renders cells as camera-facing billboards with sphere-like shading.

// Camera uniform
struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

// Lighting uniform
struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> lighting: Lighting;

// Instance data from vertex buffer
struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
    @location(3) visual_params: vec4<f32>, // specular, power, fresnel, emissive
    @location(4) rotation: vec4<f32>,       // quaternion (unused for billboards)
    @location(5) type_data_0: vec4<f32>,    // reserved for type-specific data
    @location(6) type_data_1: vec4<f32>,    // reserved for type-specific data
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) visual_params: vec4<f32>,
    @location(5) uv: vec2<f32>,
}

// Billboard quad vertices (triangle strip)
const QUAD_VERTICES: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Get quad vertex position
    let quad_pos = QUAD_VERTICES[vertex_index];
    
    // Calculate billboard vectors (camera-facing)
    let to_camera = normalize(camera.camera_pos - instance.position);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    let up = cross(to_camera, right);
    
    // Scale by radius and offset from center
    let world_offset = (right * quad_pos.x + up * quad_pos.y) * instance.radius;
    let world_pos = instance.position + world_offset;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.center = instance.position;
    out.radius = instance.radius;
    out.color = instance.color;
    out.visual_params = instance.visual_params;
    out.uv = quad_pos * 0.5 + 0.5;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Ray-sphere intersection for proper sphere shading
    let ray_origin = camera.camera_pos;
    let ray_dir = normalize(in.world_pos - ray_origin);
    
    let oc = ray_origin - in.center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - in.radius * in.radius;
    let discriminant = b * b - 4.0 * a * c;
    
    // Discard if ray misses sphere
    if (discriminant < 0.0) {
        discard;
    }
    
    // Calculate intersection point and normal
    let t = (-b - sqrt(discriminant)) / (2.0 * a);
    let hit_point = ray_origin + ray_dir * t;
    let normal = normalize(hit_point - in.center);
    
    // Extract visual parameters
    let specular_strength = in.visual_params.x;
    let specular_power = in.visual_params.y;
    let fresnel_strength = in.visual_params.z;
    let emissive = in.visual_params.w;
    
    // Lighting calculation
    let n_dot_l = max(dot(normal, -lighting.light_dir), 0.0);
    let diffuse = n_dot_l * lighting.light_color;
    
    // Specular (Blinn-Phong)
    let view_dir = normalize(camera.camera_pos - hit_point);
    let half_dir = normalize(-lighting.light_dir + view_dir);
    let spec = pow(max(dot(normal, half_dir), 0.0), specular_power) * specular_strength;
    let specular = spec * lighting.light_color;
    
    // Fresnel rim lighting
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * fresnel_strength;
    
    // Combine lighting
    let ambient_color = in.color.rgb * lighting.ambient;
    let lit_color = in.color.rgb * diffuse + specular + fresnel * in.color.rgb;
    let final_color = ambient_color + lit_color + in.color.rgb * emissive;
    
    return vec4<f32>(final_color, in.color.a);
}
