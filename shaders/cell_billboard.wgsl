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
// Perlin Noise Implementation (optimized for performance)
// ============================================================================

fn permute(x: vec4<f32>) -> vec4<f32> {
    return ((x * 34.0 + 1.0) * x) % 289.0;
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn fade(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Classic 3D Perlin noise
fn perlin_noise_3d(P: vec3<f32>) -> f32 {
    var Pi0 = floor(P);
    var Pi1 = Pi0 + vec3<f32>(1.0);
    Pi0 = Pi0 % 289.0;
    Pi1 = Pi1 % 289.0;
    let Pf0 = fract(P);
    let Pf1 = Pf0 - vec3<f32>(1.0);
    
    let ix = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    let iy = vec4<f32>(Pi0.yy, Pi1.yy);
    let iz0 = vec4<f32>(Pi0.z);
    let iz1 = vec4<f32>(Pi1.z);
    
    let ixy = permute(permute(ix) + iy);
    let ixy0 = permute(ixy + iz0);
    let ixy1 = permute(ixy + iz1);
    
    var gx0 = ixy0 / 7.0;
    var gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    let gz0 = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
    let sz0 = step(gz0, vec4<f32>(0.0));
    gx0 = gx0 - sz0 * (step(vec4<f32>(0.0), gx0) - 0.5);
    gy0 = gy0 - sz0 * (step(vec4<f32>(0.0), gy0) - 0.5);
    
    var gx1 = ixy1 / 7.0;
    var gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    let gz1 = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
    let sz1 = step(gz1, vec4<f32>(0.0));
    gx1 = gx1 - sz1 * (step(vec4<f32>(0.0), gx1) - 0.5);
    gy1 = gy1 - sz1 * (step(vec4<f32>(0.0), gy1) - 0.5);
    
    var g000 = vec3<f32>(gx0.x, gy0.x, gz0.x);
    var g100 = vec3<f32>(gx0.y, gy0.y, gz0.y);
    var g010 = vec3<f32>(gx0.z, gy0.z, gz0.z);
    var g110 = vec3<f32>(gx0.w, gy0.w, gz0.w);
    var g001 = vec3<f32>(gx1.x, gy1.x, gz1.x);
    var g101 = vec3<f32>(gx1.y, gy1.y, gz1.y);
    var g011 = vec3<f32>(gx1.z, gy1.z, gz1.z);
    var g111 = vec3<f32>(gx1.w, gy1.w, gz1.w);
    
    let norm0 = taylor_inv_sqrt(vec4<f32>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 = g000 * norm0.x;
    g010 = g010 * norm0.y;
    g100 = g100 * norm0.z;
    g110 = g110 * norm0.w;
    let norm1 = taylor_inv_sqrt(vec4<f32>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 = g001 * norm1.x;
    g011 = g011 * norm1.y;
    g101 = g101 * norm1.z;
    g111 = g111 * norm1.w;
    
    let n000 = dot(g000, Pf0);
    let n100 = dot(g100, vec3<f32>(Pf1.x, Pf0.yz));
    let n010 = dot(g010, vec3<f32>(Pf0.x, Pf1.y, Pf0.z));
    let n110 = dot(g110, vec3<f32>(Pf1.xy, Pf0.z));
    let n001 = dot(g001, vec3<f32>(Pf0.xy, Pf1.z));
    let n101 = dot(g101, vec3<f32>(Pf1.x, Pf0.y, Pf1.z));
    let n011 = dot(g011, vec3<f32>(Pf0.x, Pf1.yz));
    let n111 = dot(g111, Pf1);
    
    let fade_xyz = fade(Pf0);
    let n_z = mix(vec4<f32>(n000, n100, n010, n110), vec4<f32>(n001, n101, n011, n111), fade_xyz.z);
    let n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    
    return 2.2 * n_xyz;
}

// Fractal Brownian Motion - single octave for performance
fn fbm(p: vec3<f32>) -> f32 {
    return perlin_noise_3d(p);
}

// Calculate perturbed normal using noise gradient - forward differences for performance
fn perturb_normal(normal: vec3<f32>, sphere_pos: vec3<f32>, scale: f32, strength: f32, time: f32) -> vec3<f32> {
    if (strength <= 0.0 || scale <= 0.0) {
        return normal;
    }
    
    let eps = 0.02;
    let animated_pos = sphere_pos + vec3<f32>(0.0, 0.0, time);
    let scaled_pos = animated_pos * scale;
    
    // Base sample + 3 offset samples (forward differences)
    let base = fbm(scaled_pos);
    let nx = fbm(scaled_pos + vec3<f32>(eps, 0.0, 0.0)) - base;
    let ny = fbm(scaled_pos + vec3<f32>(0.0, eps, 0.0)) - base;
    let nz = fbm(scaled_pos + vec3<f32>(0.0, 0.0, eps)) - base;
    
    let noise_gradient = vec3<f32>(nx, ny, nz) / eps;
    
    // Project perturbation onto tangent plane
    let perturbation = noise_gradient * strength;
    let tangent_perturbation = perturbation - normal * dot(perturbation, normal);
    
    return normalize(normal + tangent_perturbation);
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
    // Use stable animation offset from cell ID (membrane_params.w) so splits don't cause jumps
    let anim_offset = in.membrane_params.w;
    let anim_time = lighting.time * noise_speed + anim_offset;
    
    // LOD: Reduce noise strength for cells that cover many pixels (large on screen)
    // This improves fill-rate performance for close/large cells
    let screen_size = in.cell_radius / max(length(in.cell_center - camera.camera_pos), 0.1);
    let lod_factor = saturate(1.0 - screen_size * 2.0);  // Fade out noise when cell is large on screen
    let effective_noise_strength = noise_strength * lod_factor;
    
    let perturbed_normal = perturb_normal(
        world_normal,
        local_noise_pos,
        noise_scale,
        effective_noise_strength,
        anim_time
    );
    
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
    
    if (is_nucleus) {
        let nuc_local_x = local_x / nucleus_size;
        let nuc_local_y = local_y / nucleus_size;
        let nuc_local_z = sqrt(max(0.0, 1.0 - nucleus_r2));
        let nuc_local_normal = vec3<f32>(nuc_local_x, nuc_local_y, nuc_local_z);
        
        let nuc_world_normal = normalize(
            nuc_local_normal.x * in.billboard_right +
            nuc_local_normal.y * in.billboard_up +
            nuc_local_normal.z * in.billboard_forward
        );
        
        // Nucleus noise - use local-space position with offset for variation
        let nuc_local_noise_pos = quat_rotate_inverse(in.cell_rotation, nuc_world_normal);
        let nuc_local_surface = nuc_local_noise_pos * nucleus_size + vec3<f32>(50.0);
        let nuc_perturbed_normal = perturb_normal(
            nuc_world_normal,
            nuc_local_surface,
            noise_scale * 1.5,
            noise_strength * 0.7,
            anim_time * 0.5
        );
        
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
