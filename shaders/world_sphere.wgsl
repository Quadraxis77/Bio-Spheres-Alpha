// World Sphere Shader
// Matches reference implementation from Biospheres-Master
// Simple lighting with ambient + diffuse, distance fade

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

struct WorldSphereParams {
    // Sphere color (RGB)
    sphere_color: vec3<f32>,
    transparency: f32,
    // Sphere radius
    radius: f32,
    // Distance fade
    fade_start_distance: f32,
    fade_end_distance: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> params: WorldSphereParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frag_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Scale vertex by sphere radius
    let world_pos = in.position * params.radius;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.frag_pos = world_pos;
    out.normal = in.normal;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance-based fade (matching reference)
    let distance_from_camera = length(in.frag_pos - camera.camera_pos);
    var fade_factor = 1.0;
    
    if (distance_from_camera > params.fade_start_distance) {
        fade_factor = 1.0 - clamp(
            (distance_from_camera - params.fade_start_distance) / 
            (params.fade_end_distance - params.fade_start_distance), 
            0.0, 1.0
        );
    }
    
    // Simple lighting - opposite direction from cells so sphere glows toward light source
    // Cells use light_dir (0.5, 1.0, 0.3), so sphere uses (-0.5, -1.0, -0.3)
    // Flip normal since we're viewing from inside the sphere (back faces)
    let norm = normalize(-in.normal);
    let light_dir = normalize(vec3<f32>(-0.5, -1.0, -0.3));
    
    // Ambient + diffuse lighting
    let ambient = 0.3;
    let diffuse = max(dot(norm, light_dir), 0.0);
    let light_factor = ambient + diffuse * 0.7;
    
    let result = light_factor * params.sphere_color;
    
    // Apply distance fade and transparency
    let final_alpha = params.transparency * fade_factor;
    
    return vec4<f32>(result, final_alpha);
}
