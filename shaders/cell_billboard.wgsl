// Cell billboard shader - renders cells as camera-facing quads

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,  // Quad corner position (-1 to 1)
}

struct InstanceInput {
    @location(1) position: vec3<f32>,   // Cell world position
    @location(2) radius: f32,           // Cell radius
    @location(3) color: vec4<f32>,      // Cell color (RGBA)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,         // UV coordinates for circle rendering
    @location(1) color: vec4<f32>,      // Cell color
    @location(2) world_pos: vec3<f32>,  // World position for depth
}

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate billboard vectors (camera-facing)
    let to_camera = normalize(camera.camera_pos - instance.position);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, to_camera));
    let billboard_up = cross(to_camera, right);
    
    // Scale quad by cell radius
    let offset = (right * vertex.quad_pos.x + billboard_up * vertex.quad_pos.y) * instance.radius;
    
    // Calculate world position
    let world_pos = instance.position + offset;
    out.world_pos = world_pos;
    
    // Transform to clip space
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Pass through UV coordinates (from -1,1 to 0,1 range)
    out.uv = vertex.quad_pos * 0.5 + 0.5;
    
    // Pass through color
    out.color = instance.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert UV to centered coordinates (-1 to 1)
    let centered_uv = (in.uv - 0.5) * 2.0;
    let dist = length(centered_uv);
    
    // Discard pixels outside the circle
    if (dist > 1.0) {
        discard;
    }
    
    // Calculate sphere normal for lighting
    let z = sqrt(1.0 - dist * dist);
    let normal = normalize(vec3<f32>(centered_uv.x, centered_uv.y, z));
    
    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(normal, light_dir), 0.0);
    
    // Ambient + diffuse lighting
    let ambient = 0.3;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    
    // Apply lighting to color
    var final_color = in.color;
    final_color = vec4<f32>(final_color.rgb * lighting, final_color.a);
    
    // Add subtle rim lighting for depth perception
    let rim = pow(1.0 - dist, 2.0) * 0.2;
    final_color = vec4<f32>(final_color.rgb + rim, final_color.a);
    
    return final_color;
}
