// Cave System Rendering Shader
// Renders procedurally generated cave mesh with lighting

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct CaveParams {
    world_center: vec3<f32>,
    world_radius: f32,
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
    grid_resolution: u32,
    triangle_count: u32,
    collision_enabled: u32,
    collision_stiffness: f32,
    collision_damping: f32,
    substeps: u32,
    _padding: f32,
    // Total: 17 * 4 = 68 bytes, need padding to 256 bytes
    _padding2: vec4<f32>,
    _padding3: vec4<f32>,
    _padding4: vec4<f32>,
    _padding5: vec4<f32>,
    _padding6: vec4<f32>,
    _padding7: vec4<f32>,
    _padding8: vec4<f32>,
    _padding9: vec4<f32>,
    _padding10: vec4<f32>,
    _padding11: vec4<f32>,
    _padding12: vec4<f32>,
    _padding13: vec4<f32>,
    _padding14: vec4<f32>,
    _padding15: vec4<f32>,
    _padding16: vec4<f32>,
    _padding17: vec4<f32>,
    _padding18: vec4<f32>,
    _padding19: vec4<f32>,
    _padding20: vec4<f32>,
    _padding21: vec4<f32>,
    _padding22: vec4<f32>,
    _padding23: vec4<f32>,
    _padding24: vec4<f32>,
    _padding25: vec4<f32>,
    _padding26: vec4<f32>,
    _padding27: vec4<f32>,
    _padding28: vec4<f32>,
    _padding29: vec4<f32>,
    _padding30: vec4<f32>,
    _padding31: vec4<f32>,
    _padding32: vec4<f32>,
    _padding33: vec4<f32>,
    _padding34: vec4<f32>,
    _padding35: vec4<f32>,
    _padding36: vec4<f32>,
    _padding37: vec4<f32>,
    _padding38: vec4<f32>,
    _padding39: vec4<f32>,
    _padding40: vec4<f32>,
    _padding41: vec4<f32>,
    _padding42: vec4<f32>,
    _padding43: vec4<f32>,
    _padding44: vec4<f32>,
    _padding45: vec4<f32>,
    _padding46: vec4<f32>,
    _padding47: vec4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> cave_params: CaveParams;

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    out.world_position = vertex.position;
    out.normal = vertex.normal;
    out.uv = vertex.uv;
    out.clip_position = camera.view_proj * vec4<f32>(vertex.position, 1.0);
    
    return out;
}

// Simple hash function for procedural texture
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth noise for cave texture
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));
    
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Triplanar texture sampling for cave walls
// Prevents stretching from multiple viewing angles
fn sample_triplanar_texture(world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    // Determine which plane to sample from based on normal
    let abs_normal = abs(normal);
    let raw_weights = abs_normal * abs_normal; // Square for proper weighting
    let total_weight = raw_weights.x + raw_weights.y + raw_weights.z;
    let normalized_weights = raw_weights / total_weight;
    
    // Sample from three planes
    let uv_xz = world_pos.xz * 0.05;  // Scale for texture density
    let uv_xy = world_pos.xy * 0.05;
    let uv_yz = world_pos.yz * 0.05;
    
    // Generate texture for each plane
    let tex_xz = noise(uv_xz);
    let tex_xy = noise(uv_xy);
    let tex_yz = noise(uv_yz);
    
    // Blend based on normal direction
    return tex_xz * normalized_weights.x + tex_xy * normalized_weights.y + tex_yz * normalized_weights.z;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize normal
    let N = normalize(in.normal);
    
    // Light direction (from above and slightly to the side)
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.2));
    
    // View direction
    let V = normalize(camera.camera_pos - in.world_position);
    
    // Diffuse lighting
    let diffuse = max(dot(N, light_dir), 0.0);
    
    // Specular lighting (Blinn-Phong)
    let H = normalize(light_dir + V);
    let specular = pow(max(dot(N, H), 0.0), 32.0);
    
    // Triplanar texture sampling prevents stretching
    let texture_value = sample_triplanar_texture(in.world_position, N);
    
    // Ambient occlusion approximation
    let ao = 0.5 + 0.5 * texture_value;
    
    // Base cave color (dark gray-brown)
    let base_color = vec3<f32>(0.15, 0.12, 0.10);
    
    // Add color variation based on triplanar texture
    let color_variation = texture_value * 0.15;
    let cave_color = base_color + vec3<f32>(color_variation);
    
    // Combine lighting
    let ambient = vec3<f32>(0.1) * ao;
    let final_color = cave_color * (ambient + diffuse * 0.7) + vec3<f32>(specular * 0.3);
    
    // Completely opaque walls
    let alpha = 1.0;
    
    return vec4<f32>(final_color, alpha);
}
