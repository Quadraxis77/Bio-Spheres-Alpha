// Voxel instance rendering shader
// Renders voxels as instanced cubes with color-coded types

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(1) instance_position: vec3<f32>,
    @location(2) voxel_type: u32,
    @location(3) color: vec4<f32>,
    @location(4) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Scale vertex by voxel size and translate to instance position
    let world_pos = vertex.position * instance.size + instance.instance_position;
    out.world_position = world_pos;
    
    // Simple normal calculation (cube face normals)
    // This is approximate - proper normals would require per-face data
    out.world_normal = normalize(vertex.position);
    
    // Transform to clip space
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Pass through color
    out.color = instance.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ambient = 0.3;
    let diffuse = max(dot(in.world_normal, light_dir), 0.0) * 0.7;
    let lighting = ambient + diffuse;
    
    // Apply lighting to color
    var final_color = in.color;
    final_color = vec4<f32>(final_color.rgb * lighting, final_color.a);
    
    return final_color;
}
