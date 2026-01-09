// Cell billboard shader - renders cells as camera-facing quads with sphere lighting
// Uses frag_depth to write correct sphere surface depth for proper occlusion
// Includes Perlin noise for organic cell membrane effect and nucleus rendering

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

struct LightingUniform {
    light_direction: vec3<f32>,    // Direction TO the light (normalized)
    light_color: vec3<f32>,        // Light color and intensity
    ambient_color: vec3<f32>,      // Ambient light color
    time: f32,                     // Simulation time for animations
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> lighting: LightingUniform;

// ============================================================================
// Quaternion Utilities
// ============================================================================

// Rotate a vector by a quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

// Inverse quaternion rotation (conjugate for unit quaternions)
fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let q_conj = vec4<f32>(-q.x, -q.y, -q.z, q.w);
    return quat_rotate(q_conj, v);
}

// ============================================================================
// Vertex/Fragment Shader
// ============================================================================

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
}

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) visual_params: vec4<f32>,  // x: specular_strength, y: specular_power, z: fresnel_strength, w: emissive
    @location(5) membrane_params: vec4<f32>, // x: noise_scale, y: noise_strength, z: noise_anim_speed, w: unused
    @location(6) rotation: vec4<f32>,        // Quaternion (x, y, z, w) for cell orientation
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
    @location(8) @interpolate(flat) membrane_params: vec4<f32>,
    @location(9) @interpolate(flat) instance_seed: f32,
    @location(10) @interpolate(flat) cell_rotation: vec4<f32>,
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
    out.color = instance.color;
    out.billboard_right = right;
    out.billboard_up = billboard_up;
    out.billboard_forward = to_camera;
    out.cell_center = instance.position;
    out.cell_radius = instance.radius;
    out.visual_params = instance.visual_params;
    out.membrane_params = instance.membrane_params;
    out.cell_rotation = instance.rotation;
    
    // Create unique seed per instance for noise variation
    out.instance_seed = fract(sin(dot(instance.position.xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Unpack parameters
    let specular_strength = in.visual_params.x;
    let specular_power = in.visual_params.y;
    let fresnel_strength = in.visual_params.z;
    let emissive = in.visual_params.w;
    let noise_scale = in.membrane_params.x;
    let noise_strength = in.membrane_params.y;
    let noise_speed = in.membrane_params.z;
    
    // UV to centered coordinates [-0.5, 0.5]
    let centered = in.uv - 0.5;
    let dist_2d = length(centered);
    let radius = 0.5;  // Full billboard size
    
    // Discard outside circle
    if (dist_2d > radius) {
        discard;
    }
    
    // ============== SPHERICAL NORMAL CALCULATION ==============
    let local_x = centered.x / radius;
    let local_y = centered.y / radius;  // Don't flip Y - causes noise scroll inversion
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
    out.depth = sphere_clip.z / sphere_clip.w;
    
    // ============== SIMPLIFIED RENDERING (NO NOISE) ==============
    // Use clean world normal without any noise perturbation
    var perturbed_normal = world_normal;
    
    // ============== NUCLEUS RENDERING ==============
    let nucleus_size = 0.55;
    let nucleus_r2 = r2 / (nucleus_size * nucleus_size);
    let is_nucleus = nucleus_r2 < 1.0 && nucleus_size > 0.01;
    
    // ============== SIMPLIFIED LIGHTING ==============
    let light_dir = normalize(lighting.light_direction);
    
    // Simple diffuse lighting only
    let diffuse = max(dot(perturbed_normal, light_dir), 0.0);
    
    // Base color with simple lighting
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
    
    // ============== SIMPLE OUTPUT (FULLY OPAQUE) ==============
    out.color = vec4<f32>(final_color, 1.0);  // Always fully opaque
    return out;
}
