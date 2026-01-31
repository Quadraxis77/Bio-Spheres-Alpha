// Nutrient particle rendering shader
// Renders camera-facing triangles

struct NutrientParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,
};

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) particle_size: f32,
};

// Generate camera-facing triangle vertices with random rotation
fn get_triangle_vertex(index: u32, particle_pos: vec3<f32>, particle_size: f32, rotation: f32) -> vec3<f32> {
    // Triangle vertices in local space (pointing up)
    let triangle_vertices = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),    // Top
        vec2<f32>(-0.866, -0.5), // Bottom left
        vec2<f32>(0.866, -0.5)   // Bottom right
    );
    
    let vertex = triangle_vertices[index];
    
    // Apply random rotation to the vertex
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    let rotated_vertex = vec2<f32>(
        vertex.x * cos_r - vertex.y * sin_r,
        vertex.x * sin_r + vertex.y * cos_r
    );
    
    // Calculate view-aligned triangle (billboard)
    let to_camera = normalize(camera.camera_pos - particle_pos);
    
    // Use a stable up vector that works for all camera angles
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(to_camera, world_up));
    let up = cross(right, to_camera);
    
    // Scale by particle size and create camera-facing triangle
    let scaled_vertex = rotated_vertex * particle_size;
    return particle_pos + right * scaled_vertex.x + up * scaled_vertex.y;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_id: u32,
    @builtin(vertex_index) vertex_id: u32,
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) animation: vec4<f32>
) -> VertexOutput {
    var output: VertexOutput;
    
    // Get triangle vertex position with random rotation
    let world_pos = get_triangle_vertex(vertex_id, position, size, animation.z);
    
    // Transform to clip space
    output.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    // Pass through color (no animation for now)
    output.color = color;
    
    // UV coordinates for triangle (for sharp edges, we'll use them differently)
    let triangle_uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.5, 0.0),    // Top
        vec2<f32>(0.0, 1.0),    // Bottom left
        vec2<f32>(1.0, 1.0)     // Bottom right
    );
    output.uv = triangle_uvs[vertex_id];
    
    output.particle_size = size;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sharp triangle - no circular falloff
    // Just return the color with full opacity
    return input.color;
}
