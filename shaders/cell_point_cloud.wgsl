// Ultra-simple point cloud shader for maximum performance cell rendering
// Renders cells as flat colored circles without lighting, noise, or sphere intersection

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
}

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) visual_params: vec4<f32>,  // unused but must match layout
    @location(5) membrane_params: vec4<f32>, // unused but must match layout
    @location(6) rotation: vec4<f32>,        // unused but must match layout
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Billboard: face camera
    let to_camera = normalize(camera.camera_pos - instance.position);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    let up = cross(to_camera, right);
    
    // Scale quad by radius
    let world_pos = instance.position 
        + right * vertex.quad_pos.x * instance.radius 
        + up * vertex.quad_pos.y * instance.radius;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vertex.quad_pos;
    out.color = instance.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple circle test - discard outside radius
    let dist_sq = dot(in.uv, in.uv);
    if (dist_sq > 1.0) {
        discard;
    }
    
    // Simple shading: darken edges slightly for depth perception
    let edge_factor = 1.0 - dist_sq * 0.3;
    let final_color = in.color.rgb * edge_factor;
    
    return vec4<f32>(final_color, in.color.a);
}
