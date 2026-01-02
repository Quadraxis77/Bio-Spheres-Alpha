// Cell billboard depth-only shader
// Writes sphere depth for occlusion testing, no color output

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Lighting uniform still needed for bind group compatibility
struct LightingUniform {
    light_direction: vec3<f32>,
    light_color: vec3<f32>,
    ambient_color: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(1)
var<uniform> lighting: LightingUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
}

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) visual_params: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) billboard_right: vec3<f32>,
    @location(2) billboard_up: vec3<f32>,
    @location(3) billboard_forward: vec3<f32>,
    @location(4) cell_center: vec3<f32>,
    @location(5) @interpolate(flat) cell_radius: f32,
}

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    
    let to_camera = normalize(camera.camera_pos - instance.position);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, to_camera));
    let billboard_up = cross(to_camera, right);
    
    let offset = (right * vertex.quad_pos.x + billboard_up * vertex.quad_pos.y) * instance.radius;
    let world_pos = instance.position + offset;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vertex.quad_pos * 0.5 + 0.5;
    out.billboard_right = right;
    out.billboard_up = billboard_up;
    out.billboard_forward = to_camera;
    out.cell_center = instance.position;
    out.cell_radius = instance.radius;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let centered_uv = (in.uv - 0.5) * 2.0;
    let dist = length(centered_uv);
    
    if (dist > 1.0) {
        discard;
    }
    
    let z = sqrt(1.0 - dist * dist);
    
    let sphere_surface_world = in.cell_center 
        + in.billboard_right * centered_uv.x * in.cell_radius
        + in.billboard_up * centered_uv.y * in.cell_radius
        + in.billboard_forward * z * in.cell_radius;
    
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_surface_world, 1.0);
    return sphere_clip.z / sphere_clip.w;
}
