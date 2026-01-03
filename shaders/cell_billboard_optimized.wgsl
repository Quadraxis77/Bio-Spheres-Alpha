// Optimized cell billboard shader with reduced overdraw and improved performance
// Key optimizations:
// 1. Early discard for pixels outside sphere
// 2. Adaptive noise quality based on screen size
// 3. Simplified lighting for distant cells
// 4. Reduced noise sampling

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
    @location(1) cell_pos: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) visual_params: vec4<f32>,
    @location(5) membrane_params: vec4<f32>,
    @location(6) rotation: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) cell_center: vec3<f32>,
    @location(3) cell_radius: f32,
    @location(4) billboard_right: vec3<f32>,
    @location(5) billboard_up: vec3<f32>,
    @location(6) billboard_forward: vec3<f32>,
    @location(7) visual_params: vec4<f32>,
    @location(8) membrane_params: vec4<f32>,
    @location(9) cell_rotation: vec4<f32>,
    @location(10) screen_size: f32, // NEW: Screen-space size for LOD
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate billboard vectors
    let to_camera = normalize(camera.camera_pos - in.cell_pos);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    let up = cross(to_camera, right);
    
    // Calculate screen-space size for LOD (approximate)
    let dist_to_camera = length(camera.camera_pos - in.cell_pos);
    let screen_size = in.radius / max(dist_to_camera, 0.1);
    
    // Scale quad by radius
    let world_offset = (right * in.quad_pos.x + up * in.quad_pos.y) * in.radius;
    let world_pos = in.cell_pos + world_offset;
    
    out.position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = in.quad_pos * 0.5 + 0.5;
    out.color = in.color;
    out.cell_center = in.cell_pos;
    out.cell_radius = in.radius;
    out.billboard_right = right;
    out.billboard_up = up;
    out.billboard_forward = to_camera;
    out.visual_params = in.visual_params;
    out.membrane_params = in.membrane_params;
    out.cell_rotation = in.rotation;
    out.screen_size = screen_size;
    
    return out;
}

// Fast hash function for noise
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Simplified noise for distant cells
fn simple_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // Cubic interpolation
    
    let n000 = hash31(i);
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    
    let n00 = mix(n000, n100, u.x);
    let n10 = mix(n010, n110, u.x);
    
    return mix(n00, n10, u.y) * 2.0 - 1.0;
}

// Quaternion rotation functions
fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let q_conj = vec4<f32>(-q.x, -q.y, -q.z, q.w);
    let qv = vec3<f32>(q_conj.x, q_conj.y, q_conj.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q_conj.w) + uuv) * 2.0;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Extract parameters
    let specular_strength = in.visual_params.x;
    let specular_power = in.visual_params.y;
    let fresnel_strength = in.visual_params.z;
    let emissive = in.visual_params.w;
    let noise_scale = in.membrane_params.x;
    let noise_strength = in.membrane_params.y;
    let noise_speed = in.membrane_params.z;
    let anim_offset = in.membrane_params.w;
    
    // UV to centered coordinates [-0.5, 0.5]
    let centered = in.uv - 0.5;
    let dist_2d = length(centered);
    let radius = 0.5;
    
    // EARLY DISCARD: Outside circle (major overdraw reduction)
    if (dist_2d > radius) {
        discard;
    }
    
    // Calculate sphere normal and depth
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
    
    // Calculate sphere surface position for depth
    let sphere_surface_world = in.cell_center + world_normal * in.cell_radius;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_surface_world, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;
    
    // LOD-based rendering: Skip expensive effects for small/distant cells
    let is_large_cell = in.screen_size > 0.02; // ~2% of screen height
    let is_medium_cell = in.screen_size > 0.008; // ~0.8% of screen height
    
    // Lighting setup
    let light_dir = normalize(lighting.light_direction);
    let view_dir = in.billboard_forward;
    
    var final_normal = world_normal;
    
    // Membrane noise - only for reasonably sized cells
    if (is_medium_cell && noise_strength > 0.001) {
        let local_noise_pos = quat_rotate_inverse(in.cell_rotation, world_normal);
        let anim_time = lighting.time * noise_speed + anim_offset;
        
        // Use simplified noise for medium cells, skip for small cells
        let noise_sample = simple_noise(local_noise_pos * noise_scale + vec3<f32>(0.0, 0.0, anim_time));
        
        // Reduce noise strength based on screen size
        let effective_noise_strength = noise_strength * saturate(in.screen_size * 50.0);
        
        // Simple normal perturbation (much faster than full gradient calculation)
        let noise_offset = noise_sample * effective_noise_strength;
        final_normal = normalize(world_normal + in.billboard_right * noise_offset * 0.3);
    }
    
    // Nucleus rendering - only for large cells
    var nucleus_contrib = vec3<f32>(0.0);
    var nucleus_alpha = 0.0;
    
    if (is_large_cell) {
        let nucleus_size = 0.55;
        let nucleus_r2 = r2 / (nucleus_size * nucleus_size);
        
        if (nucleus_r2 < 1.0) {
            // Simplified nucleus lighting (no noise for performance)
            let nuc_ndot_l = max(0.0, dot(world_normal, light_dir));
            let nucleus_color = in.color.rgb * 0.4;
            nucleus_contrib = nucleus_color * (0.2 + nuc_ndot_l * 0.8);
            
            let nuc_edge_dist = sqrt(nucleus_r2);
            nucleus_alpha = 1.0 - smoothstep(0.85, 1.0, nuc_edge_dist);
        }
    }
    
    // Membrane lighting
    let ndot_l = max(0.0, dot(final_normal, light_dir));
    let ambient = 0.12;
    let diffuse = ndot_l;
    
    // Simplified lighting for small cells
    var membrane_color: vec3<f32>;
    if (is_medium_cell) {
        // Full lighting for medium+ cells
        let half_vec = normalize(light_dir + view_dir);
        let ndot_h = max(0.0, dot(final_normal, half_vec));
        let specular = pow(ndot_h, specular_power) * specular_strength;
        let fresnel = pow(1.0 - max(0.0, dot(final_normal, view_dir)), 3.0) * fresnel_strength;
        
        membrane_color = in.color.rgb * (ambient + diffuse * 0.8)
            + vec3<f32>(specular)
            + in.color.rgb * fresnel * 1.5
            + in.color.rgb * emissive;
    } else {
        // Simplified lighting for small cells
        membrane_color = in.color.rgb * (ambient + diffuse * 0.9) + in.color.rgb * emissive;
    }
    
    // Membrane transparency
    let user_membrane_opacity = in.color.a;
    let center_opacity = 0.2 * user_membrane_opacity;
    let edge_opacity = 0.75 * user_membrane_opacity;
    let edge_factor = sqrt(r2);
    let membrane_alpha = mix(center_opacity, edge_opacity, pow(edge_factor, 1.2));
    
    // Composite layers
    let bg_color = in.color.rgb * 0.3;
    var cell_color = bg_color;
    
    // Add nucleus if present
    if (nucleus_alpha > 0.0) {
        cell_color = mix(bg_color, nucleus_contrib, nucleus_alpha);
    }
    
    // Add membrane
    let final_color = mix(cell_color, membrane_color, membrane_alpha);
    
    // Always fully opaque
    out.color = vec4<f32>(final_color, 1.0);
    return out;
}