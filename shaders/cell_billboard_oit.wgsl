// Cell billboard shader with Weighted Blended Order-Independent Transparency (WBOIT)
// Accumulation pass - outputs to two render targets for later compositing
// SIMPLIFIED VERSION: No noise, minimal lighting for performance

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

struct LightingUniform {
    light_direction: vec3<f32>,
    light_color: vec3<f32>,
    ambient_color: vec3<f32>,
    time: f32,
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
    @location(5) membrane_params: vec4<f32>,
    @location(6) rotation: vec4<f32>,
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
    
    return out;
}

// WBOIT output structure
struct WboitOutput {
    @location(0) accumulation: vec4<f32>,  // Weighted color accumulation
    @location(1) revealage: f32,           // Transparency coverage
}

@fragment
fn fs_main(in: VertexOutput) -> WboitOutput {
    var out: WboitOutput;
    
    // UV to centered coordinates
    let centered = in.uv - 0.5;
    let dist_2d = length(centered);
    let radius = 0.5;
    
    // Discard outside circle
    if (dist_2d > radius) {
        discard;
    }
    
    // Calculate sphere coordinates
    let local_x = centered.x / radius;
    let local_y = centered.y / radius;
    let r2 = local_x * local_x + local_y * local_y;
    let local_z = sqrt(max(0.0, 1.0 - r2));
    
    let local_normal = vec3<f32>(local_x, local_y, local_z);
    let world_normal = normalize(
        local_normal.x * in.billboard_right +
        local_normal.y * in.billboard_up +
        local_normal.z * in.billboard_forward
    );
    
    // ============== NUCLEUS RENDERING ==============
    let nucleus_size = 0.55;
    let nucleus_r2 = r2 / (nucleus_size * nucleus_size);
    let is_nucleus = nucleus_r2 < 1.0 && nucleus_size > 0.01;
    
    // Simple lighting without noise
    let light_dir = normalize(lighting.light_direction);
    let diffuse = max(dot(world_normal, light_dir), 0.0);
    
    let base_color = in.color.rgb;
    let lit_color = base_color * (lighting.ambient_color + lighting.light_color * diffuse);
    
    // Nucleus rendering (simple, no noise)
    var final_color = lit_color;
    if (is_nucleus) {
        // Nucleus is darker version of cell color
        let nucleus_color = base_color * 0.6;
        let nucleus_lit = nucleus_color * (lighting.ambient_color + lighting.light_color * diffuse * 0.8);
        
        // Smooth edge transition for nucleus
        let nuc_edge_dist = sqrt(nucleus_r2);
        let nuc_edge_softness = smoothstep(0.85, 1.0, nuc_edge_dist);
        let nucleus_alpha = 1.0 - nuc_edge_softness;
        
        final_color = mix(lit_color, nucleus_lit, nucleus_alpha);
    }
    
    let alpha = in.color.a;
    
    // WBOIT weight calculation
    // Use depth-based weighting to approximate correct blending order
    let z = in.clip_position.z;
    let weight = max(min(1.0, max(max(lit_color.r, lit_color.g), lit_color.b) * alpha), alpha) * 
                 clamp(0.03 / (1e-5 + pow(z / 200.0, 4.0)), 1e-2, 3e3);
    
    // Output weighted color and revealage
    out.accumulation = vec4<f32>(final_color * alpha * weight, alpha);
    out.revealage = alpha;
    
    return out;
}