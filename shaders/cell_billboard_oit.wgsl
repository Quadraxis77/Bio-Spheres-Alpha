// Cell billboard shader with Weighted Blended Order-Independent Transparency (WBOIT)
// Accumulation pass - outputs to two render targets for later compositing

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

struct LightingUniform {
    light_direction: vec3<f32>,
    light_color: vec3<f32>,
    ambient_color: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

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
    @location(1) color: vec4<f32>,
    @location(2) billboard_right: vec3<f32>,
    @location(3) billboard_up: vec3<f32>,
    @location(4) billboard_forward: vec3<f32>,
    @location(5) cell_center: vec3<f32>,
    @location(6) @interpolate(flat) cell_radius: f32,
    @location(7) @interpolate(flat) visual_params: vec4<f32>,
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
    out.color = instance.color;
    out.billboard_right = right;
    out.billboard_up = billboard_up;
    out.billboard_forward = to_camera;
    out.cell_center = instance.position;
    out.cell_radius = instance.radius;
    out.visual_params = instance.visual_params;
    
    return out;
}

// WBOIT outputs: accumulation (RGB * A * weight, A * weight) and revealage (1 - A)
struct FragmentOutput {
    @location(0) accum: vec4<f32>,    // Premultiplied color accumulation
    @location(1) revealage: f32,       // Product of (1 - alpha)
}

// Weight function for WBOIT - emphasizes closer fragments
fn calculate_weight(z: f32, alpha: f32) -> f32 {
    // Simplified weight - just use 1.0 for now to debug
    return 1.0;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    let specular_strength = in.visual_params.x;
    let specular_power = in.visual_params.y;
    let fresnel_strength = in.visual_params.z;
    let emissive = in.visual_params.w;
    
    let centered_uv = (in.uv - 0.5) * 2.0;
    let dist = length(centered_uv);
    
    if (dist > 1.0) {
        discard;
    }
    
    let z = sqrt(1.0 - dist * dist);
    let normal_billboard = vec3<f32>(centered_uv.x, centered_uv.y, z);
    
    // Calculate sphere depth for weight calculation
    let sphere_surface_world = in.cell_center 
        + in.billboard_right * centered_uv.x * in.cell_radius
        + in.billboard_up * centered_uv.y * in.cell_radius
        + in.billboard_forward * z * in.cell_radius;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_surface_world, 1.0);
    let linear_depth = sphere_clip.w; // Use linear depth for weight
    
    // Lighting calculation
    let light_billboard = vec3<f32>(
        dot(lighting.light_direction, in.billboard_right),
        dot(lighting.light_direction, in.billboard_up),
        dot(lighting.light_direction, in.billboard_forward)
    );
    let light_billboard_normalized = normalize(light_billboard);
    let diffuse = max(dot(normal_billboard, light_billboard_normalized), 0.0);
    let lighting_factor = lighting.ambient_color + lighting.light_color * diffuse;
    let lit_color = in.color.rgb * lighting_factor;
    
    // Specular
    let view_billboard = vec3<f32>(0.0, 0.0, 1.0);
    let reflect_dir = reflect(-light_billboard_normalized, normal_billboard);
    let spec = pow(max(dot(view_billboard, reflect_dir), 0.0), specular_power) * specular_strength;
    
    // Fresnel rim lighting
    let fresnel = pow(1.0 - z, 3.0) * fresnel_strength;
    let rim_color = in.color.rgb * fresnel;
    
    // Emissive
    let emissive_color = in.color.rgb * emissive;
    
    // Final color
    let final_color = lit_color + spec + rim_color + emissive_color;
    let alpha = in.color.a;
    
    // Calculate WBOIT weight
    let weight = calculate_weight(linear_depth / 100.0, alpha); // Normalize depth
    
    // Output accumulation: premultiplied color * weight
    out.accum = vec4<f32>(final_color * alpha * weight, alpha * weight);
    
    // Output revealage: product of (1 - alpha)
    out.revealage = alpha;
    
    return out;
}
