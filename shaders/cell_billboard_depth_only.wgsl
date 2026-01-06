// Minimal depth-only shader for cell billboard depth pre-pass
// This shader does the absolute minimum work needed to write correct sphere depth
// NO lighting, NO noise, NO color calculations - just depth
//
// The key optimization: this shader uses discard, but since it's depth-only,
// the GPU can still use early-Z for the subsequent color pass which uses
// a different shader without discard.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Lighting uniform needed for bind group compatibility (not used)
struct LightingUniform {
    light_direction: vec3<f32>,
    light_color: vec3<f32>,
    ambient_color: vec3<f32>,
    time: f32,
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
    @location(5) membrane_params: vec4<f32>,
    @location(6) rotation: vec4<f32>,
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
    
    // Calculate billboard vectors (camera-facing)
    let to_camera = normalize(camera.camera_pos - instance.position);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, to_camera));
    let billboard_up = cross(to_camera, right);
    
    // Scale quad by cell radius
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
    // UV to centered coordinates [-0.5, 0.5]
    let centered = in.uv - 0.5;
    let dist_2d = length(centered);
    let radius = 0.5;
    
    // Discard outside circle - this is the only discard in the depth pass
    if (dist_2d > radius) {
        discard;
    }
    
    // Calculate sphere normal for depth
    let local_x = centered.x / radius;
    let local_y = centered.y / radius;
    let r2 = local_x * local_x + local_y * local_y;
    let local_z = sqrt(max(0.0, 1.0 - r2));
    
    let local_normal = vec3<f32>(local_x, local_y, local_z);
    
    // Transform local normal to world space using billboard basis
    let world_normal = normalize(
        local_normal.x * in.billboard_right +
        local_normal.y * in.billboard_up +
        local_normal.z * in.billboard_forward
    );
    
    // Calculate sphere surface position for depth
    let sphere_surface_world = in.cell_center + world_normal * in.cell_radius;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_surface_world, 1.0);
    
    return sphere_clip.z / sphere_clip.w;
}
