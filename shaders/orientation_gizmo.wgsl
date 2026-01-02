// Orientation gizmo shader for Bio-Spheres
// Renders 3D axis lines at cell position showing cell orientation
// Lines are occluded by an imaginary sphere at the cell center
// Uses instancing to render gizmos for multiple cells in one draw call

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
    @location(1) color: vec4<f32>,
    // Per-instance attributes (transform matrix as 4 vec4s + params)
    @location(2) transform_col0: vec4<f32>,
    @location(3) transform_col1: vec4<f32>,
    @location(4) transform_col2: vec4<f32>,
    @location(5) transform_col3: vec4<f32>,
    @location(6) params: vec4<f32>,  // cell_radius, opacity, _padding, _padding
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) cell_center: vec3<f32>,
    @location(3) @interpolate(flat) cell_radius: f32,
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
    
    // Transform vertex position by gizmo transform (position + rotation + scale)
    let world_pos = transform * vec4<f32>(vertex.position, 1.0);
    
    // Project to clip space using camera view-projection matrix
    out.clip_position = camera.view_proj * world_pos;
    
    // Apply opacity to color
    out.color = vec4<f32>(vertex.color.rgb, vertex.color.a * vertex.params.y);
    
    // Pass world position for sphere occlusion test
    out.world_pos = world_pos.xyz;
    
    // Extract cell center from transform matrix (translation component = column 3)
    out.cell_center = vertex.transform_col3.xyz;
    
    // Cell radius is the first param
    out.cell_radius = vertex.params.x;
    
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