// Color-only shader for cell billboard rendering after depth pre-pass
// This shader does NOT use discard - it relies on the depth buffer from the pre-pass
// to reject pixels outside the sphere. This enables early-Z optimization.
//
// Key optimization: By not using discard, the GPU can perform early-Z testing
// and reject fragments before running the expensive fragment shader.

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

// Quaternion rotation functions
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let q_conj = vec4<f32>(-q.x, -q.y, -q.z, q.w);
    return quat_rotate(q_conj, v);
}

// Fast hash function for noise
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Quintic interpolation curve
fn quintic(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3D Value noise
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = quintic(f);
    
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    
    let n00 = mix(n000, n100, u.x);
    let n01 = mix(n001, n101, u.x);
    let n10 = mix(n010, n110, u.x);
    let n11 = mix(n011, n111, u.x);
    let n0 = mix(n00, n10, u.y);
    let n1 = mix(n01, n11, u.y);
    
    return mix(n0, n1, u.z) * 2.0 - 1.0;
}

// Fast normal perturbation
fn perturb_normal_fast(normal: vec3<f32>, sphere_pos: vec3<f32>, scale: f32, strength: f32, time: f32) -> vec3<f32> {
    if (strength <= 0.001 || scale <= 0.0) {
        return normal;
    }
    
    let eps = 0.02;
    let animated_pos = sphere_pos + vec3<f32>(0.0, 0.0, time);
    let scaled_pos = animated_pos * scale;
    
    let base = value_noise_3d(scaled_pos);
    let nx = value_noise_3d(scaled_pos + vec3<f32>(eps, 0.0, 0.0)) - base;
    let ny = value_noise_3d(scaled_pos + vec3<f32>(0.0, eps, 0.0)) - base;
    let nz = value_noise_3d(scaled_pos + vec3<f32>(0.0, 0.0, eps)) - base;
    
    let noise_gradient = vec3<f32>(nx, ny, nz) / eps;
    let perturbation = noise_gradient * strength;
    let tangent_perturbation = perturbation - normal * dot(perturbation, normal);
    
    return normalize(normal + tangent_perturbation);
}

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
    @location(7) @interpolate(flat) visual_params: vec4<f32>,
    @location(8) @interpolate(flat) membrane_params: vec4<f32>,
    @location(9) @interpolate(flat) cell_rotation: vec4<f32>,
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
    out.membrane_params = instance.membrane_params;
    out.cell_rotation = instance.rotation;
    
    return out;
}

// Fragment output with explicit depth to match the depth pre-pass exactly
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
    
    // UV to centered coordinates
    let centered = in.uv - 0.5;
    let dist_2d = length(centered);
    let radius = 0.5;
    
    // Discard pixels outside the circle
    // This is necessary because depth testing alone can't reject these pixels
    // when they haven't been written to the depth buffer
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
    
    // Calculate sphere surface position for depth (must match depth pre-pass exactly)
    let sphere_surface_world = in.cell_center + world_normal * in.cell_radius;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_surface_world, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;
    
    // LOD based on screen size
    let dist_to_camera = length(in.cell_center - camera.camera_pos);
    let screen_size = in.cell_radius / max(dist_to_camera, 0.1);
    
    // Membrane noise perturbation (skip for small/distant cells)
    var perturbed_normal = world_normal;
    var effective_noise_strength = 0.0;
    if (screen_size > 0.005 && noise_strength > 0.001) {
        let lod_factor = saturate(1.0 - screen_size * 1.5);
        effective_noise_strength = noise_strength * lod_factor;
        
        if (effective_noise_strength > 0.001) {
            let local_noise_pos = quat_rotate_inverse(in.cell_rotation, world_normal);
            let anim_offset = in.membrane_params.w;
            let anim_time = lighting.time * noise_speed + anim_offset;
            
            perturbed_normal = perturb_normal_fast(
                world_normal,
                local_noise_pos,
                noise_scale,
                effective_noise_strength,
                anim_time
            );
        }
    }
    
    // Nucleus parameters
    let nucleus_size = 0.55;
    let nucleus_r2 = r2 / (nucleus_size * nucleus_size);
    let is_nucleus = nucleus_r2 < 1.0 && nucleus_size > 0.01;
    
    // Lighting setup
    let light_dir = normalize(lighting.light_direction);
    let view_dir = in.billboard_forward;
    let half_vec = normalize(light_dir + view_dir);
    
    // Nucleus rendering (only for visible cells)
    var nucleus_contrib = vec3<f32>(0.0);
    var nucleus_alpha = 0.0;
    
    if (is_nucleus && screen_size > 0.01) {
        let nuc_local_x = local_x / nucleus_size;
        let nuc_local_y = local_y / nucleus_size;
        let nuc_local_z = sqrt(max(0.0, 1.0 - nucleus_r2));
        let nuc_local_normal = vec3<f32>(nuc_local_x, nuc_local_y, nuc_local_z);
        
        let nuc_world_normal = normalize(
            nuc_local_normal.x * in.billboard_right +
            nuc_local_normal.y * in.billboard_up +
            nuc_local_normal.z * in.billboard_forward
        );
        
        var nuc_perturbed_normal = nuc_world_normal;
        if (effective_noise_strength > 0.001 && screen_size > 0.03) {
            let nuc_local_noise_pos = quat_rotate_inverse(in.cell_rotation, nuc_world_normal);
            let anim_offset = in.membrane_params.w;
            let anim_time = lighting.time * noise_speed + anim_offset;
            
            nuc_perturbed_normal = perturb_normal_fast(
                nuc_world_normal,
                nuc_local_noise_pos,
                noise_scale * 1.5,
                noise_strength * 0.7,
                anim_time * 0.5
            );
        }
        
        let nuc_ndot_l = max(0.0, dot(nuc_perturbed_normal, light_dir));
        let nuc_ambient = 0.2;
        let nuc_diffuse = nuc_ndot_l;
        let nuc_ndot_h = max(0.0, dot(nuc_perturbed_normal, half_vec));
        let nuc_specular = pow(nuc_ndot_h, 32.0) * 0.4;
        
        let nucleus_color = in.color.rgb * 0.4;
        nucleus_contrib = nucleus_color * (nuc_ambient + nuc_diffuse * 0.8) + vec3<f32>(nuc_specular * 0.5);
        
        let nuc_edge_dist = sqrt(nucleus_r2);
        let nuc_edge_softness = smoothstep(0.85, 1.0, nuc_edge_dist);
        nucleus_alpha = 1.0 - nuc_edge_softness;
    }
    
    // Membrane lighting
    let ndot_l = max(0.0, dot(perturbed_normal, light_dir));
    let ambient = 0.12;
    let diffuse = ndot_l;
    let ndot_h = max(0.0, dot(perturbed_normal, half_vec));
    let specular = pow(ndot_h, specular_power) * specular_strength;
    let fresnel = pow(1.0 - max(0.0, dot(perturbed_normal, view_dir)), 3.0) * fresnel_strength;
    let subsurface = (diffuse * 0.7 + 0.3) * 0.15;
    
    let base_color = in.color.rgb;
    let inner_glow = base_color * 1.2 * subsurface;
    
    let membrane_color = base_color * (ambient + diffuse * 0.8)
        + vec3<f32>(specular)
        + base_color * fresnel * 1.5
        + inner_glow
        + base_color * emissive;
    
    // Membrane transparency
    let user_membrane_opacity = in.color.a;
    let center_opacity = 0.2 * user_membrane_opacity;
    let edge_opacity = 0.75 * user_membrane_opacity;
    let edge_factor = sqrt(r2);
    let membrane_alpha = mix(center_opacity, edge_opacity, pow(edge_factor, 1.2));
    
    // Composite layers
    let bg_color = in.color.rgb * 0.3;
    var cell_color = bg_color;
    if (is_nucleus) {
        cell_color = mix(bg_color, nucleus_contrib, nucleus_alpha);
    }
    let final_color = mix(cell_color, membrane_color, membrane_alpha);
    
    out.color = vec4<f32>(final_color, 1.0);
    return out;
}
