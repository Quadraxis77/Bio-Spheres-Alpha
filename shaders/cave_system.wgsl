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

// Shadow field data (from light field system)
struct ShadowFieldParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    shadow_strength: f32,
    shadow_enabled: u32,
    shadow_quality: f32,
}

@group(2) @binding(0) var<uniform> shadow_params: ShadowFieldParams;
@group(2) @binding(1) var<storage, read> light_field: array<f32>;

// Sample light field at world position with trilinear interpolation
fn sample_light_field(world_pos: vec3<f32>) -> f32 {
    if (shadow_params.shadow_enabled == 0u) {
        return 1.0;
    }
    let res = shadow_params.grid_resolution;
    let fres = f32(res);
    
    let gx = (world_pos.x - shadow_params.grid_origin_x) / shadow_params.cell_size - 0.5;
    let gy = (world_pos.y - shadow_params.grid_origin_y) / shadow_params.cell_size - 0.5;
    let gz = (world_pos.z - shadow_params.grid_origin_z) / shadow_params.cell_size - 0.5;
    
    if (gx < -0.5 || gx >= fres - 0.5 ||
        gy < -0.5 || gy >= fres - 0.5 ||
        gz < -0.5 || gz >= fres - 0.5) {
        return 1.0;
    }
    
    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let iz = i32(floor(gz));
    let fx = gx - floor(gx);
    let fy = gy - floor(gy);
    let fz = gz - floor(gz);
    
    let ires = i32(res);
    let x0 = u32(clamp(ix, 0, ires - 1));
    let x1 = u32(clamp(ix + 1, 0, ires - 1));
    let y0 = u32(clamp(iy, 0, ires - 1));
    let y1 = u32(clamp(iy + 1, 0, ires - 1));
    let z0 = u32(clamp(iz, 0, ires - 1));
    let z1 = u32(clamp(iz + 1, 0, ires - 1));
    
    let c000 = light_field[x0 + y0 * res + z0 * res * res];
    let c100 = light_field[x1 + y0 * res + z0 * res * res];
    let c010 = light_field[x0 + y1 * res + z0 * res * res];
    let c110 = light_field[x1 + y1 * res + z0 * res * res];
    let c001 = light_field[x0 + y0 * res + z1 * res * res];
    let c101 = light_field[x1 + y0 * res + z1 * res * res];
    let c011 = light_field[x0 + y1 * res + z1 * res * res];
    let c111 = light_field[x1 + y1 * res + z1 * res * res];
    
    let c00 = mix(c000, c100, fx);
    let c10 = mix(c010, c110, fx);
    let c01 = mix(c001, c101, fx);
    let c11 = mix(c011, c111, fx);
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    
    return mix(c0, c1, fz);
}

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
    
    // Calculate distance from world center for outer layer detection
    let dist_from_center = length(in.world_position - cave_params.world_center);
    let normalized_dist = dist_from_center / cave_params.world_radius;
    
    // Check if we're in bottom quarter (negative Y) with noise variation
    let world_pos_normalized = (in.world_position - cave_params.world_center) / cave_params.world_radius;
    let hemisphere_factor = -world_pos_normalized.y; // Positive for bottom hemisphere
    
    // Add noise variation to create irregular boundary between quarters
    let noise_scale = 8.0;
    let boundary_variation = sample_triplanar_texture(in.world_position * noise_scale, normalize(in.world_position)) * 0.3 - 0.15;
    let hemisphere_threshold = 0.5 + boundary_variation; // 0.5 = bottom quarter (50% of hemisphere)
    
    // Only apply red coloring to bottom quarter with variation
    let is_bottom_quarter = hemisphere_factor > hemisphere_threshold;
    
    // Base cave color (dark gray-brown)
    let base_color = vec3<f32>(0.15, 0.12, 0.10);
    
    // Outer layer reddish color
    let outer_color = vec3<f32>(0.6, 0.15, 0.1); // Reddish brown
    
    // Blend factor for outer layer (starts at 0.9 of world radius, full at 1.0)
    let outer_blend_start = 0.9;
    let outer_blend_factor = smoothstep(outer_blend_start, 1.0, normalized_dist);
    
    // Only apply red color if in bottom quarter AND in outer layer
    let red_factor = outer_blend_factor * select(0.0, 1.0, is_bottom_quarter);
    
    // Add color variation based on triplanar texture
    let color_variation = texture_value * 0.15;
    let cave_color = base_color + vec3<f32>(color_variation);
    
    // Blend with outer layer color only in bottom hemisphere
    let final_base_color = mix(cave_color, outer_color, red_factor);
    
    // Combine lighting
    // Offset sample position along normal to avoid self-shadowing at cave surface
    // Higher quality = larger offset to avoid self-shadowing artifacts
    let offset_distance = mix(3.0, 6.0, shadow_params.shadow_quality) * shadow_params.cell_size;
    let shadow_sample_pos = in.world_position + N * offset_distance;
    // Clamp sample position to stay within valid light field grid bounds
    let grid_size = shadow_params.cell_size * f32(shadow_params.grid_resolution);
    let grid_min = vec3<f32>(shadow_params.grid_origin_x, shadow_params.grid_origin_y, shadow_params.grid_origin_z);
    let grid_max = grid_min + vec3<f32>(grid_size, grid_size, grid_size);
    let clamped_pos = clamp(shadow_sample_pos, grid_min, grid_max);
    let shadow = mix(1.0, sample_light_field(clamped_pos), shadow_params.shadow_strength);
    let ambient = vec3<f32>(0.1) * ao;
    let final_color = final_base_color * (ambient + diffuse * 0.7 * shadow) + vec3<f32>(specular * 0.3 * shadow);
    
    // Completely opaque walls
    let alpha = 1.0;
    
    return vec4<f32>(final_color, alpha);
}
