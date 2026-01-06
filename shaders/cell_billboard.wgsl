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
// Fast Value Noise Implementation (much faster than Perlin)
// ============================================================================

// Fast hash function for 3D coordinates - improved version
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Quintic interpolation curve (smoother than cubic, reduces banding)
fn quintic(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3D Value noise with quintic interpolation for smoother results
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Quintic interpolation curve (same as Perlin's improved noise)
    // This significantly reduces banding artifacts
    let u = quintic(f);
    
    // Hash the 8 corners of the cube
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    
    // Trilinear interpolation
    let n00 = mix(n000, n100, u.x);
    let n01 = mix(n001, n101, u.x);
    let n10 = mix(n010, n110, u.x);
    let n11 = mix(n011, n111, u.x);
    let n0 = mix(n00, n10, u.y);
    let n1 = mix(n01, n11, u.y);
    
    return mix(n0, n1, u.z) * 2.0 - 1.0;  // Map to [-1, 1]
}

// Fractal Brownian Motion - single octave for performance
fn fbm(p: vec3<f32>) -> f32 {
    return value_noise_3d(p);
}

// Fast normal perturbation using 4 value noise samples for proper gradient
// Value noise is ~5-10x faster than Perlin, so 4 samples is still fast
fn perturb_normal_fast(normal: vec3<f32>, sphere_pos: vec3<f32>, scale: f32, strength: f32, time: f32) -> vec3<f32> {
    if (strength <= 0.001 || scale <= 0.0) {
        return normal;
    }
    
    let eps = 0.02;
    let animated_pos = sphere_pos + vec3<f32>(0.0, 0.0, time);
    let scaled_pos = animated_pos * scale;
    
    // Compute gradient with 4 samples (base + 3 offsets)
    let base = value_noise_3d(scaled_pos);
    let nx = value_noise_3d(scaled_pos + vec3<f32>(eps, 0.0, 0.0)) - base;
    let ny = value_noise_3d(scaled_pos + vec3<f32>(0.0, eps, 0.0)) - base;
    let nz = value_noise_3d(scaled_pos + vec3<f32>(0.0, 0.0, eps)) - base;
    
    let noise_gradient = vec3<f32>(nx, ny, nz) / eps;
    
    // Project perturbation onto tangent plane
    let perturbation = noise_gradient * strength;
    let tangent_perturbation = perturbation - normal * dot(perturbation, normal);
    
    return normalize(normal + tangent_perturbation);
}

// Alias for compatibility
fn perturb_normal(normal: vec3<f32>, sphere_pos: vec3<f32>, scale: f32, strength: f32, time: f32) -> vec3<f32> {
    return perturb_normal_fast(normal, sphere_pos, scale, strength, time);
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
    
    // ============== MEMBRANE NOISE PERTURBATION ==============
    // Transform the world normal by the inverse of the cell's rotation to get local-space normal
    // This makes the noise texture rotate with the cell
    let local_noise_pos = quat_rotate_inverse(in.cell_rotation, world_normal);
    
    // Add random texture offset based on instance seed to make each cell unique
    let texture_offset = vec3<f32>(
        sin(in.instance_seed * 12.9898) * 43758.5453,
        sin(in.instance_seed * 78.233) * 43758.5453,
        sin(in.instance_seed * 37.719) * 43758.5453
    );
    let offset_noise_pos = local_noise_pos + texture_offset;
    
    // Use stable animation offset from cell ID (membrane_params.w) so splits don't cause jumps
    let anim_offset = in.membrane_params.w;
    let anim_time = lighting.time * noise_speed + anim_offset;
    
    // LOD: Aggressively reduce noise for performance
    // screen_size approximates how large the cell appears on screen
    let dist_to_camera = length(in.cell_center - camera.camera_pos);
    let screen_size = in.cell_radius / max(dist_to_camera, 0.1);
    
    // Skip noise entirely for small/distant cells (huge performance win)
    // Threshold of 0.02 means cells covering less than ~2% of screen height skip noise
    var perturbed_normal = world_normal;
    var effective_noise_strength = 0.0;
    if (screen_size > 0.005 && noise_strength > 0.001) {
        // Fade out noise as cells get larger (fill-rate optimization)
        let lod_factor = saturate(1.0 - screen_size * 1.5);
        effective_noise_strength = noise_strength * lod_factor;
        
        if (effective_noise_strength > 0.001) {
            // Use fast single-sample noise for membrane (4x faster)
            perturbed_normal = perturb_normal_fast(
                world_normal,
                offset_noise_pos,  // Use offset position for unique texture per cell
                noise_scale,
                effective_noise_strength,
                anim_time
            );
        }
    }
    
    // ============== NUCLEUS PARAMETERS ==============
    let nucleus_size = 0.55;  // Could be made configurable
    let nucleus_r2 = r2 / (nucleus_size * nucleus_size);
    let is_nucleus = nucleus_r2 < 1.0 && nucleus_size > 0.01;
    
    // ============== LIGHTING SETUP ==============
    let light_dir = normalize(lighting.light_direction);
    let view_dir = in.billboard_forward;
    let half_vec = normalize(light_dir + view_dir);
    
    // ============== NUCLEUS RENDERING ==============
    var nucleus_contrib = vec3<f32>(0.0);
    var nucleus_alpha = 0.0;
    
    // Only render nucleus detail for cells that are reasonably visible
    // Skip nucleus entirely for very small/distant cells
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
        
        // Nucleus noise - use same approach as membrane with unique offset
        var nuc_perturbed_normal = nuc_world_normal;
        if (effective_noise_strength > 0.001 && screen_size > 0.03) {
            // Same coordinate system as membrane - just the rotated normal with offset
            let nuc_local_noise_pos = quat_rotate_inverse(in.cell_rotation, nuc_world_normal);
            let nuc_offset_noise_pos = nuc_local_noise_pos + texture_offset;
            
            nuc_perturbed_normal = perturb_normal_fast(
                nuc_world_normal,
                nuc_offset_noise_pos,  // Use offset position for unique nucleus texture
                noise_scale * 1.5,
                noise_strength * 0.7,
                anim_time * 0.5
            );
        }
        
        // Nucleus lighting
        let nuc_ndot_l = max(0.0, dot(nuc_perturbed_normal, light_dir));
        let nuc_ambient = 0.2;
        let nuc_diffuse = nuc_ndot_l;
        let nuc_ndot_h = max(0.0, dot(nuc_perturbed_normal, half_vec));
        let nuc_specular = pow(nuc_ndot_h, 32.0) * 0.4;
        
        // Nucleus color - darker version of cell color
        let nucleus_color = in.color.rgb * 0.4;
        nucleus_contrib = nucleus_color * (nuc_ambient + nuc_diffuse * 0.8) + vec3<f32>(nuc_specular * 0.5);
        
        // Nucleus edge softness
        let nuc_edge_dist = sqrt(nucleus_r2);
        let nuc_edge_softness = smoothstep(0.85, 1.0, nuc_edge_dist);
        nucleus_alpha = 1.0 - nuc_edge_softness;
    }
    
    // ============== MEMBRANE LIGHTING ==============
    let ndot_l = max(0.0, dot(perturbed_normal, light_dir));
    let ambient = 0.12;
    let diffuse = ndot_l;
    
    // Specular (Blinn-Phong)
    let ndot_h = max(0.0, dot(perturbed_normal, half_vec));
    let specular = pow(ndot_h, specular_power) * specular_strength;
    
    // Fresnel rim lighting
    let fresnel = pow(1.0 - max(0.0, dot(perturbed_normal, view_dir)), 3.0) * fresnel_strength;
    
    // Subsurface approximation
    let subsurface = (diffuse * 0.7 + 0.3) * 0.15;
    
    let base_color = in.color.rgb;
    let inner_glow = base_color * 1.2 * subsurface;
    
    let membrane_color = base_color * (ambient + diffuse * 0.8)
        + vec3<f32>(specular)
        + base_color * fresnel * 1.5
        + inner_glow
        + base_color * emissive;
    
    // Membrane transparency - use user opacity setting scaled by edge factor
    // User opacity only affects how visible the membrane is over the cytoplasm/nucleus
    let user_membrane_opacity = in.color.a;
    let center_opacity = 0.2 * user_membrane_opacity;
    let edge_opacity = 0.75 * user_membrane_opacity;
    let edge_factor = sqrt(r2);
    let membrane_alpha = mix(center_opacity, edge_opacity, pow(edge_factor, 1.2));
    
    // ============== COMPOSITE ==============
    // Layer 1: Background (cytoplasm) - always opaque
    let bg_color = in.color.rgb * 0.3;
    
    // Layer 2: Nucleus over background - always opaque
    var cell_color = bg_color;
    if (is_nucleus) {
        cell_color = mix(bg_color, nucleus_contrib, nucleus_alpha);
    }
    
    // Layer 3: Membrane over everything - affected by user opacity
    let final_color = mix(cell_color, membrane_color, membrane_alpha);
    
    // Cell is always fully opaque - membrane opacity only affects internal compositing
    out.color = vec4<f32>(final_color, 1.0);
    return out;
}
