// Orientation gizmo shader for Bio-Spheres
// Renders 3D axis lines at cell position showing cell orientation
// Lines are occluded by an imaginary sphere at the cell center

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct GizmoUniform {
    transform: mat4x4<f32>,      // Cell position + rotation + scale
    params: vec4<f32>,           // size, opacity, _padding, _padding
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> gizmo: GizmoUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) cell_center: vec3<f32>,
    @location(3) cell_radius: f32,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform vertex position by gizmo transform (position + rotation + scale)
    let world_pos = gizmo.transform * vec4<f32>(vertex.position, 1.0);
    
    // Project to clip space using camera view-projection matrix
    out.clip_position = camera.view_proj * world_pos;
    
    // Apply opacity to color
    out.color = vec4<f32>(vertex.color.rgb, vertex.color.a * gizmo.params.y);
    
    // Pass world position for sphere occlusion test
    out.world_pos = world_pos.xyz;
    
    // Extract cell center from transform matrix (translation component)
    out.cell_center = vec3<f32>(gizmo.transform[3][0], gizmo.transform[3][1], gizmo.transform[3][2]);
    
    // Cell radius is the scale factor
    out.cell_radius = gizmo.params.x;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Perform sphere occlusion test
    // Ray from camera to fragment world position
    let ray_origin = camera.camera_pos;
    let ray_dir = normalize(in.world_pos - ray_origin);
    
    // Sphere center and radius
    let sphere_center = in.cell_center;
    let sphere_radius = in.cell_radius;
    
    // Ray-sphere intersection test
    let oc = ray_origin - sphere_center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    // If ray intersects sphere, check if fragment is behind the sphere
    if (discriminant >= 0.0) {
        let t1 = (-b - sqrt(discriminant)) / (2.0 * a);
        let t2 = (-b + sqrt(discriminant)) / (2.0 * a);
        
        // Distance from camera to fragment
        let fragment_distance = length(in.world_pos - ray_origin);
        
        // If fragment is behind the near intersection point, it's occluded
        if (t1 > 0.0 && fragment_distance > t1) {
            discard;
        }
    }
    
    return in.color;
}