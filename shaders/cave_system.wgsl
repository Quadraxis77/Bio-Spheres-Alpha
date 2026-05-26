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
    caustic_intensity: f32,
    caustic_scale: f32,
    caustic_speed: f32,
    time: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    moss_parallax_depth: f32,
    moss_scale: f32,
    // Moss appearance parameters
    moss_noise_type: u32,
    moss_noise_frequency: f32,
    moss_noise_lacunarity: f32,
    moss_height_sharpness_low: f32,
    moss_height_sharpness_high: f32,
    moss_bump_strength: f32,
    moss_color_dark_r: f32,
    moss_color_dark_g: f32,
    moss_color_dark_b: f32,
    moss_color_bright_r: f32,
    moss_color_bright_g: f32,
    moss_color_bright_b: f32,
    _pad_moss_0: f32,
    _pad_moss_1: f32,
}

@group(2) @binding(0) var<uniform> shadow_params: ShadowFieldParams;
@group(2) @binding(1) var<storage, read> light_field: array<f32>;
@group(2) @binding(2) var<storage, read> water_density: array<f32>;
@group(2) @binding(3) var<storage, read> moss_density: array<f32>;

// Sample light field at world position.
// Uses nearest-neighbor when close (cam_dist < 40) — the light field varies over
// 30+ world-unit scales so the 1-voxel interpolation error is invisible up close,
// and this saves 7 storage-buffer reads (24 bytes each) per fragment.
// At distance >= 40 we do full trilinear to keep smooth shadow gradients.
fn sample_light_field_lod(world_pos: vec3<f32>, cam_dist: f32) -> f32 {
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

    let ires = i32(res);

    if (cam_dist < 40.0) {
        // Nearest-neighbor: 1 read instead of 8
        let ix = u32(clamp(i32(round(gx)), 0, ires - 1));
        let iy = u32(clamp(i32(round(gy)), 0, ires - 1));
        let iz = u32(clamp(i32(round(gz)), 0, ires - 1));
        return light_field[ix + iy * res + iz * res * res];
    }

    // Trilinear: smooth shadow gradients at range
    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let iz = i32(floor(gz));
    let fx = gx - floor(gx);
    let fy = gy - floor(gy);
    let fz = gz - floor(gz);

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
    let c0  = mix(c00, c10, fy);
    let c1  = mix(c01, c11, fy);
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

// ============================================================
// Smooth water density sampling (trilinear interpolation)
// ============================================================
fn sample_water_density(world_pos: vec3<f32>) -> f32 {
    let res = shadow_params.grid_resolution;
    let fres = f32(res);
    let grid_origin = vec3<f32>(shadow_params.grid_origin_x, shadow_params.grid_origin_y, shadow_params.grid_origin_z);

    let gx = (world_pos.x - grid_origin.x) / shadow_params.cell_size;
    let gy = (world_pos.y - grid_origin.y) / shadow_params.cell_size;
    let gz = (world_pos.z - grid_origin.z) / shadow_params.cell_size;

    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let iz = i32(floor(gz));
    let fx = gx - floor(gx);
    let fy = gy - floor(gy);
    let fz = gz - floor(gz);

    let ires = i32(res);
    if (ix < 0 || ix >= ires - 1 || iy < 0 || iy >= ires - 1 || iz < 0 || iz >= ires - 1) {
        return 0.0;
    }

    let x0 = u32(ix);
    let x1 = u32(ix + 1);
    let y0 = u32(iy);
    let y1 = u32(iy + 1);
    let z0 = u32(iz);
    let z1 = u32(iz + 1);

    let d000 = water_density[x0 + y0 * res + z0 * res * res];
    let d100 = water_density[x1 + y0 * res + z0 * res * res];
    let d010 = water_density[x0 + y1 * res + z0 * res * res];
    let d110 = water_density[x1 + y1 * res + z0 * res * res];
    let d001 = water_density[x0 + y0 * res + z1 * res * res];
    let d101 = water_density[x1 + y0 * res + z1 * res * res];
    let d011 = water_density[x0 + y1 * res + z1 * res * res];
    let d111 = water_density[x1 + y1 * res + z1 * res * res];

    let d00 = mix(d000, d100, fx);
    let d10 = mix(d010, d110, fx);
    let d01 = mix(d001, d101, fx);
    let d11 = mix(d011, d111, fx);
    let d0 = mix(d00, d10, fy);
    let d1 = mix(d01, d11, fy);

    return mix(d0, d1, fz);
}

// ============================================================
// Procedural caustics (iterative coordinate distortion)
// Based on the well-known Shadertoy water caustic technique.
// Bright lines form where sin/cos approach zero through
// iterative warping, producing realistic caustic networks.
// ============================================================

fn caustic_single(uv: vec2<f32>, time: f32) -> f32 {
    // Use continuous UV directly (no fract/tiling).
    // Offset into a range where the formula behaves well.
    var p = uv * 6.28318 - vec2<f32>(250.0, 250.0);
    var i = p;
    var c = 1.0;
    let inten = 0.005;

    for (var n = 0; n < 5; n++) {
        let t = time * (1.0 - 3.5 / f32(n + 1));
        i = p + vec2<f32>(
            cos(t - i.x) + sin(t + i.y),
            sin(t - i.y) + cos(t + i.x)
        );
        c += 1.0 / length(vec2<f32>(
            p.x / (sin(i.x + t) / inten),
            p.y / (cos(i.y + t) / inten)
        ));
    }

    c = c / 5.0;
    c = 1.17 - pow(c, 1.4);
    return pow(abs(c), 8.0);
}

fn caustic_pattern(world_pos: vec3<f32>, normal: vec3<f32>, time: f32) -> f32 {
    let scale = shadow_params.caustic_scale;
    let speed = shadow_params.caustic_speed;
    let t = time * speed;

    // Triplanar projection to avoid stretching
    let abs_n = abs(normal);
    var uv: vec2<f32>;
    if (abs_n.y > abs_n.x && abs_n.y > abs_n.z) {
        uv = world_pos.xz;
    } else if (abs_n.x > abs_n.z) {
        uv = world_pos.yz;
    } else {
        uv = world_pos.xy;
    }

    let p = uv * scale * 0.05;

    // Two layers at slightly different scales/times for richer look
    let c1 = caustic_single(p, t);
    let c2 = caustic_single(p * 0.8 + vec2<f32>(1.7, 3.2), t * 0.7 + 5.0);

    return min((c1 + c2) * 0.5, 1.0);
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

// Moss density sampling — trilinear interpolation from 128³ grid.
fn sample_moss_density(world_pos: vec3<f32>) -> f32 {
    let res = shadow_params.grid_resolution;
    let grid_origin = vec3<f32>(shadow_params.grid_origin_x, shadow_params.grid_origin_y, shadow_params.grid_origin_z);

    let gx = (world_pos.x - grid_origin.x) / shadow_params.cell_size;
    let gy = (world_pos.y - grid_origin.y) / shadow_params.cell_size;
    let gz = (world_pos.z - grid_origin.z) / shadow_params.cell_size;

    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let iz = i32(floor(gz));
    let fx = gx - floor(gx);
    let fy = gy - floor(gy);
    let fz = gz - floor(gz);

    let ires = i32(res);
    if (ix < 0 || ix >= ires - 1 || iy < 0 || iy >= ires - 1 || iz < 0 || iz >= ires - 1) {
        return 0.0;
    }

    let x0 = u32(ix); let x1 = u32(ix + 1);
    let y0 = u32(iy); let y1 = u32(iy + 1);
    let z0 = u32(iz); let z1 = u32(iz + 1);

    let m000 = moss_density[x0 + y0 * res + z0 * res * res];
    let m100 = moss_density[x1 + y0 * res + z0 * res * res];
    let m010 = moss_density[x0 + y1 * res + z0 * res * res];
    let m110 = moss_density[x1 + y1 * res + z0 * res * res];
    let m001 = moss_density[x0 + y0 * res + z1 * res * res];
    let m101 = moss_density[x1 + y0 * res + z1 * res * res];
    let m011 = moss_density[x0 + y1 * res + z1 * res * res];
    let m111 = moss_density[x1 + y1 * res + z1 * res * res];

    let m00 = mix(m000, m100, fx);
    let m10 = mix(m010, m110, fx);
    let m01 = mix(m001, m101, fx);
    let m11 = mix(m011, m111, fx);
    let m0  = mix(m00, m10, fy);
    let m1  = mix(m01, m11, fy);
    return mix(m0, m1, fz);
}

// ============================================================
// Procedural moss height map for parallax occlusion mapping
// Returns height in [0, 1] where 1 = tallest moss tuft
//
// octaves: controls quality vs cost
//   3 = full (close range)
//   2 = mid range
//   1 = far range (single octave, very cheap)
// ============================================================

// Worley (cellular) noise for moss
fn worley_noise(uv: vec2<f32>) -> f32 {
    let i = floor(uv);
    let f = fract(uv);
    var min_dist = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let cell_offset = i + neighbor;
            let point = vec2<f32>(
                fract(sin(dot(cell_offset, vec2<f32>(127.1, 311.7))) * 43758.5453),
                fract(sin(dot(cell_offset, vec2<f32>(269.5, 183.3))) * 43758.5453)
            );
            let diff = neighbor + point - f;
            let dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }
    return min_dist;
}

// ============================================================
// Moss height — single octave only (LOD removed)
// ============================================================
fn moss_height_lod(uv: vec2<f32>, octaves: i32) -> f32 {
    let freq = shadow_params.moss_noise_frequency;
    let noise_type = shadow_params.moss_noise_type;
    var h: f32;
    if (noise_type == 1u) {
        h = 1.0 - worley_noise(uv * freq);
    } else if (noise_type == 2u) {
        h = 1.0 - abs(noise(uv * freq) * 2.0 - 1.0);
    } else {
        h = noise(uv * freq);
    }
    h = smoothstep(shadow_params.moss_height_sharpness_low, shadow_params.moss_height_sharpness_high, h);
    return h;
}

// Full-quality wrapper kept for reference — all call sites use moss_height_lod directly.

// Moss color: varies from dark base to bright tips with per-position variation
fn moss_color(uv: vec2<f32>, height_val: f32) -> vec3<f32> {
    // Colors from uniform parameters
    let dark_moss = vec3<f32>(shadow_params.moss_color_dark_r, shadow_params.moss_color_dark_g, shadow_params.moss_color_dark_b);
    let bright_moss = vec3<f32>(shadow_params.moss_color_bright_r, shadow_params.moss_color_bright_g, shadow_params.moss_color_bright_b);
    // Mid color is average of dark and bright
    let mid_moss = mix(dark_moss, bright_moss, 0.4);

    // Height-based gradient: dark at base, bright at tips
    var col = mix(dark_moss, mid_moss, smoothstep(0.0, 0.4, height_val));
    col = mix(col, bright_moss, smoothstep(0.4, 0.9, height_val));

    // Per-position hue variation (some patches more yellow, some more blue-green)
    let hue_var = noise(uv * 5.0 + 42.0);
    let yellow_shift = vec3<f32>(0.04, 0.02, -0.02) * (hue_var - 0.5);
    col += yellow_shift;

    return col;
}

fn get_triplanar_uv(world_pos: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let abs_n = abs(normal);
    let scale = shadow_params.moss_scale;
    if (abs_n.y > abs_n.x && abs_n.y > abs_n.z) {
        return world_pos.xz * scale;
    } else if (abs_n.x > abs_n.z) {
        return world_pos.yz * scale;
    } else {
        return world_pos.xy * scale;
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize normal
    var N = normalize(in.normal);
    
    // Light direction from uniform (matches sun direction)
    let light_dir = normalize(vec3<f32>(shadow_params.light_dir_x, shadow_params.light_dir_y, shadow_params.light_dir_z));
    
    // View direction
    let V = normalize(camera.camera_pos - in.world_position);
    
    // Triplanar texture sampling prevents stretching
    let texture_value = sample_triplanar_texture(in.world_position, N);
    
    // Ambient occlusion approximation
    var ao = 0.5 + 0.5 * texture_value;
    
    // Calculate distance from world center for outer layer detection
    let dist_from_center = length(in.world_position - cave_params.world_center);
    let normalized_dist = dist_from_center / cave_params.world_radius;
    
    // Check if we're in bottom quarter (negative Y) with noise variation
    let world_pos_normalized = (in.world_position - cave_params.world_center) / cave_params.world_radius;
    let hemisphere_factor = -world_pos_normalized.y; // Positive for bottom hemisphere
    
    // Add noise variation to create irregular boundary between quarters
    let noise_scale = 8.0;
    let boundary_variation = sample_triplanar_texture(in.world_position * noise_scale, normalize(in.world_position)) * 0.3 - 0.15;
    let hemisphere_threshold = 0.5 + boundary_variation;
    
    // Only apply red coloring to bottom quarter with variation
    let is_bottom_quarter = hemisphere_factor > hemisphere_threshold;
    
    // Base cave color (dark gray-brown)
    let base_color = vec3<f32>(0.15, 0.12, 0.10);
    
    // Outer layer reddish color
    let outer_color = vec3<f32>(0.6, 0.15, 0.1);
    
    // Blend factor for outer layer
    let outer_blend_start = 0.9;
    let outer_blend_factor = smoothstep(outer_blend_start, 1.0, normalized_dist);
    let red_factor = outer_blend_factor * select(0.0, 1.0, is_bottom_quarter);
    
    // Add color variation based on triplanar texture
    let color_variation = texture_value * 0.15;
    let cave_color = base_color + vec3<f32>(color_variation);
    
    // Blend with outer layer color only in bottom hemisphere
    var final_base_color = mix(cave_color, outer_color, red_factor);

    // ── Moss rendering (flat, no POM, no normals, 1 noise octave) ──────────
    let moss_sample_pos = in.world_position + N * shadow_params.cell_size * 0.5;
    let cam_dist = length(camera.camera_pos - in.world_position);
    let moss_amount = sample_moss_density(moss_sample_pos);

    var specular_strength = 1.0;

    if (moss_amount > 0.01) {
        let base_uv = get_triplanar_uv(in.world_position, N);
        let h       = moss_height_lod(base_uv, 1);
        let m_color = moss_color(base_uv, h);

        let color_blend = moss_amount * smoothstep(0.1, 0.5, h);
        final_base_color = mix(final_base_color, m_color, color_blend);

        specular_strength = mix(1.0, 0.1, moss_amount);

        let moss_ao = mix(0.4, 1.0, h);
        ao *= mix(1.0, moss_ao, moss_amount);
    }

    // Diffuse lighting (using potentially moss-perturbed normal)
    let diffuse = max(dot(N, light_dir), 0.0);
    
    // Specular lighting (Blinn-Phong, reduced on mossy surfaces)
    let H = normalize(light_dir + V);
    let specular = pow(max(dot(N, H), 0.0), 32.0) * specular_strength;
    
    // Combine lighting
    let offset_distance = mix(3.0, 6.0, shadow_params.shadow_quality) * shadow_params.cell_size;
    let shadow_sample_pos = in.world_position + N * offset_distance;
    let grid_size = shadow_params.cell_size * f32(shadow_params.grid_resolution);
    let grid_min = vec3<f32>(shadow_params.grid_origin_x, shadow_params.grid_origin_y, shadow_params.grid_origin_z);
    let grid_max = grid_min + vec3<f32>(grid_size, grid_size, grid_size);
    let clamped_pos = clamp(shadow_sample_pos, grid_min, grid_max);
    let shadow = mix(1.0, sample_light_field_lod(clamped_pos, cam_dist), shadow_params.shadow_strength);
    let sun_color = vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);
    let ambient = vec3<f32>(0.1) * ao;
    var final_color = final_base_color * (ambient + sun_color * diffuse * 0.7 * shadow) + sun_color * vec3<f32>(specular * 0.3 * shadow);
    
    // Apply underwater caustics on lit surfaces
    let water_check_pos = in.world_position + N * shadow_params.cell_size * 1.5;
    let density = sample_water_density(water_check_pos);
    if (shadow_params.caustic_intensity > 0.0 && shadow > 0.3 && density > 0.01) {
        let caustic = caustic_pattern(in.world_position, N, shadow_params.time);
        let density_fade = smoothstep(0.0, 0.5, density);
        let caustic_tint = sun_color * vec3<f32>(0.7, 0.85, 1.0);
        final_color += caustic_tint * caustic * shadow_params.caustic_intensity * shadow * density_fade * 0.5;
    }
    
    // Completely opaque walls
    let alpha = 1.0;
    
    return vec4<f32>(final_color, alpha);
}
