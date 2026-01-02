// Split ring shader for Bio-Spheres
// Renders flat rings around cells to show split plane direction
// Uses instancing to render rings for multiple cells in one draw call

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    // Per-vertex attributes
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Per-instance attributes (transform matrix as 4 vec4s + color)
    @location(2) transform_col0: vec4<f32>,
    @location(3) transform_col1: vec4<f32>,
    @location(4) transform_col2: vec4<f32>,
    @location(5) transform_col3: vec4<f32>,
    @location(6) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) local_pos: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Reconstruct transform matrix from instance attributes
    let transform = mat4x4<f32>(
        vertex.transform_col0,
        vertex.transform_col1,
        vertex.transform_col2,
        vertex.transform_col3
    );
    
    // The vertex positions are already in the correct ring shape
    // Just apply the transform directly
    let world_pos = transform * vec4<f32>(vertex.position, 1.0);
    
    // Project to clip space
    out.clip_position = camera.view_proj * world_pos;
    
    // Pass world position and normal for lighting
    out.world_pos = world_pos.xyz;
    out.world_normal = normalize((transform * vec4<f32>(vertex.normal, 0.0)).xyz);
    
    // Pass color from instance
    out.color = vertex.color;
    
    // Pass local position for effects
    out.local_pos = vertex.position;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(in.world_normal, light_dir), 0.3);
    
    // Apply lighting to color
    var final_color = in.color.rgb * diffuse;
    
    // Add slight transparency
    let alpha = in.color.a * 0.8;
    
    return vec4<f32>(final_color, alpha);
}
