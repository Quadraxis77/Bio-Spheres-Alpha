// Cell billboard shader - renders cells as camera-facing quads with sphere lighting

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

struct LightingUniform {
    light_direction: vec3<f32>,    // Direction TO the light (normalized)
    light_color: vec3<f32>,        // Light color and intensity
    ambient_color: vec3<f32>,      // Ambient light color
    _padding: f32,                 // Alignment padding
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> lighting: LightingUniform;

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
    @location(3) billboard_right: vec3<f32>,  // Billboard right vector
    @location(4) billboard_up: vec3<f32>,     // Billboard up vector
    @location(5) billboard_forward: vec3<f32>, // Billboard forward vector (to camera)
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
    
    // Pass billboard basis vectors for normal calculation in fragment shader
    out.billboard_right = right;
    out.billboard_up = billboard_up;
    out.billboard_forward = to_camera;
    
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
    
    // Calculate sphere normal in billboard space
    // For a sphere: x² + y² + z² = 1, so z = sqrt(1 - x² - y²)
    let z = sqrt(1.0 - dist * dist);
    let normal_billboard = vec3<f32>(centered_uv.x, centered_uv.y, z);
    
    // Transform light direction from world space to billboard space
    // This is the inverse of transforming normals to world space
    let light_billboard = vec3<f32>(
        dot(lighting.light_direction, in.billboard_right),
        dot(lighting.light_direction, in.billboard_up),
        dot(lighting.light_direction, in.billboard_forward)
    );
    let light_billboard_normalized = normalize(light_billboard);
    
    // Calculate lighting in billboard space
    let diffuse = max(dot(normal_billboard, light_billboard_normalized), 0.0);
    
    // Combine ambient and diffuse lighting
    let lighting_factor = lighting.ambient_color + lighting.light_color * diffuse;
    
    // Apply lighting to cell color
    let lit_color = in.color.rgb * lighting_factor;
    
    // Add specular highlight in billboard space
    let view_billboard = vec3<f32>(0.0, 0.0, 1.0); // View direction in billboard space is always +Z
    let reflect_dir = reflect(-light_billboard_normalized, normal_billboard);
    let spec = pow(max(dot(view_billboard, reflect_dir), 0.0), 32.0) * 0.3;
    let final_color = lit_color + spec;
    
    return vec4<f32>(final_color, in.color.a);
}
