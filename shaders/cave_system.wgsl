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
    geothermal_enabled: u32,
    geothermal_count: u32,
    geothermal_placement_mode: u32,
    geothermal_lower_hemisphere: u32,
    geothermal_gravity_mode: u32,
    geothermal_gravity: f32,
    geothermal_length: f32,
    geothermal_width: f32,
    geothermal_depth: f32,
    geothermal_back_margin: f32,
    geothermal_top_margin: f32,
    geothermal_heat_output: f32,
    geothermal_heat_radius: f32,
    geothermal_glow_strength: f32,
    geothermal_glow_radius: f32,
    geothermal_glow_color: vec3<f32>,
    isolated_chunk_cull_volume: f32,
    mesh_smoothing_iterations: u32,
    mesh_smoothing_factor: f32,
    mesh_smooth_normals: u32,
    appearance: u32,
    rock_dark_r: f32,
    rock_dark_g: f32,
    rock_dark_b: f32,
    rock_layer_scale: f32,
    rock_cool_r: f32,
    rock_cool_g: f32,
    rock_cool_b: f32,
    rock_warp_strength: f32,
    rock_warm_r: f32,
    rock_warm_g: f32,
    rock_warm_b: f32,
    rock_fine_band_strength: f32,
    rock_pale_r: f32,
    rock_pale_g: f32,
    rock_pale_b: f32,
    rock_cool_mottle_strength: f32,
    rock_grain_strength: f32,
    rock_patch_contrast: f32,
    rock_seam_darkening: f32,
    rock_wall_line_strength: f32,
    rock_min_color: f32,
    rock_max_color: f32,
    rock_ambient_strength: f32,
    rock_diffuse_strength: f32,
    rock_specular_strength: f32,
    rock_specular_power: f32,
    rock_texture_scale: f32,
    rock_coarse_frequency: f32,
    rock_fine_frequency: f32,
    rock_seam_frequency: f32,
    rock_fine_noise_scale: f32,
    rock_fine_noise_strength: f32,
    rock_seam_noise_scale: f32,
    rock_seam_noise_strength: f32,
    rock_coarse_band_low: f32,
    rock_coarse_band_high: f32,
    rock_fine_band_low: f32,
    rock_fine_band_high: f32,
    rock_seam_low: f32,
    rock_seam_high: f32,
    rock_geometry_conform: f32,
    rock_parallax_depth: f32,
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
@group(2) @binding(1) var light_field_tex: texture_3d<f32>;
@group(2) @binding(2) var light_color_field_tex: texture_3d<f32>;
@group(2) @binding(3) var light_field_sampler: sampler;
@group(2) @binding(4) var<storage, read> water_density: array<f32>;
@group(2) @binding(5) var<storage, read> moss_density: array<f32>;

fn world_to_light_uvw(world_pos: vec3<f32>) -> vec3<f32> {
    let res = f32(shadow_params.grid_resolution);
    return vec3<f32>(
        (world_pos.x - shadow_params.grid_origin_x) / (shadow_params.cell_size * res),
        (world_pos.y - shadow_params.grid_origin_y) / (shadow_params.cell_size * res),
        (world_pos.z - shadow_params.grid_origin_z) / (shadow_params.cell_size * res),
    );
}

fn light_uvw_in_bounds(uvw: vec3<f32>) -> bool {
    return all(uvw >= vec3<f32>(0.0)) && all(uvw <= vec3<f32>(1.0));
}

fn sample_light_field_lod(world_pos: vec3<f32>) -> f32 {
    if (shadow_params.shadow_enabled == 0u) {
        return 1.0;
    }
    let uvw = world_to_light_uvw(world_pos);
    if (!light_uvw_in_bounds(uvw)) {
        return 1.0;
    }
    return textureSampleLevel(light_field_tex, light_field_sampler, uvw, 0.0).r;
}

fn sample_light_color_field(world_pos: vec3<f32>) -> vec3<f32> {
    let fallback = vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);
    if (shadow_params.shadow_enabled == 0u) {
        return fallback;
    }
    let uvw = world_to_light_uvw(world_pos);
    if (!light_uvw_in_bounds(uvw)) {
        return fallback;
    }
    let sample = textureSampleLevel(light_color_field_tex, light_field_sampler, uvw, 0.0);
    if (sample.w <= 0.0001) {
        return fallback;
    }
    return sample.rgb;
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
    let texture_scale = cave_params.rock_texture_scale;
    let uv_xz = world_pos.xz * texture_scale;
    let uv_xy = world_pos.xy * texture_scale;
    let uv_yz = world_pos.yz * texture_scale;
    
    // Generate texture for each plane
    let tex_xz = noise(uv_xz);
    let tex_xy = noise(uv_xy);
    let tex_yz = noise(uv_yz);
    
    // Blend based on normal direction
    return tex_xz * normalized_weights.x + tex_xy * normalized_weights.y + tex_yz * normalized_weights.z;
}

fn layered_rock_color(world_pos: vec3<f32>, normal: vec3<f32>, texture_value: f32) -> vec3<f32> {
    // Layer mostly by world height, but warp the coordinate so strata bend and
    // pinch instead of forming perfect latitude rings around the cave volume.
    let broad_warp = noise(world_pos.xz * 0.018)
                   + noise(world_pos.yz * 0.014 + vec2<f32>(17.0, 5.0)) * 0.7
                   + noise(world_pos.xy * 0.011 + vec2<f32>(4.0, 23.0)) * 0.45;
    let conform_axis = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), abs(normal), cave_params.rock_geometry_conform));
    let conform_height = dot(world_pos, conform_axis);
    let layer_coord = conform_height * cave_params.rock_layer_scale
                    + broad_warp * cave_params.rock_warp_strength;

    let coarse_wave = sin(layer_coord * cave_params.rock_coarse_frequency);
    let fine_wave = sin(
        layer_coord * cave_params.rock_fine_frequency
        + noise(world_pos.xz * cave_params.rock_fine_noise_scale) * cave_params.rock_fine_noise_strength
    );
    let seam_wave = sin(
        layer_coord * cave_params.rock_seam_frequency
        + noise(world_pos.yz * cave_params.rock_seam_noise_scale) * cave_params.rock_seam_noise_strength
    );

    let coarse_band = smoothstep(cave_params.rock_coarse_band_low, cave_params.rock_coarse_band_high, coarse_wave);
    let fine_band = smoothstep(cave_params.rock_fine_band_low, cave_params.rock_fine_band_high, fine_wave);
    let dark_seam = smoothstep(cave_params.rock_seam_low, cave_params.rock_seam_high, abs(seam_wave));

    let dark_rock = vec3<f32>(cave_params.rock_dark_r, cave_params.rock_dark_g, cave_params.rock_dark_b);
    let cool_slate = vec3<f32>(cave_params.rock_cool_r, cave_params.rock_cool_g, cave_params.rock_cool_b);
    let warm_shale = vec3<f32>(cave_params.rock_warm_r, cave_params.rock_warm_g, cave_params.rock_warm_b);
    let pale_silt = vec3<f32>(cave_params.rock_pale_r, cave_params.rock_pale_g, cave_params.rock_pale_b);

    var color = mix(dark_rock, warm_shale, coarse_band);
    color = mix(color, pale_silt, fine_band * cave_params.rock_fine_band_strength);
    color = mix(
        color,
        cool_slate,
        noise(world_pos.xy * 0.025 + vec2<f32>(11.0, 31.0)) * cave_params.rock_cool_mottle_strength
    );

    let grain = texture_value - 0.5;
    let patch_noise = noise(world_pos.xz * 0.032 + vec2<f32>(19.0, 7.0));
    color += vec3<f32>(grain * cave_params.rock_grain_strength);
    color *= mix(
        1.0 - cave_params.rock_patch_contrast,
        1.0 + cave_params.rock_patch_contrast,
        patch_noise
    );
    color *= 1.0 - dark_seam * cave_params.rock_seam_darkening;

    // Steeper walls show crisper sediment lines; flatter caps keep a subtler,
    // more scuffed look so the pattern does not become overly graphic.
    let wall_factor = 1.0 - abs(normal.y);
    color = mix(
        color * 0.92 + vec3<f32>(0.025),
        color,
        wall_factor * cave_params.rock_wall_line_strength + 0.35
    );

    return clamp(color, vec3<f32>(cave_params.rock_min_color), vec3<f32>(cave_params.rock_max_color));
}

fn lava_surface_uv(world_pos: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let abs_n = abs(normal);
    if (abs_n.y > abs_n.x && abs_n.y > abs_n.z) {
        return world_pos.xz;
    } else if (abs_n.x > abs_n.z) {
        return world_pos.zy;
    }
    return world_pos.xy;
}

fn lava_plate_edge(uv: vec2<f32>) -> f32 {
    let cell = floor(uv);
    let local = fract(uv);
    var nearest = 10.0;
    var second = 10.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y));
            let c = cell + offset;
            let p = vec2<f32>(
                hash(c + vec2<f32>(17.0, 41.0)),
                hash(c + vec2<f32>(73.0, 19.0))
            ) * 0.68 + vec2<f32>(0.16);
            let d = length(offset + p - local);
            if (d < nearest) {
                second = nearest;
                nearest = d;
            } else if (d < second) {
                second = d;
            }
        }
    }

    return second - nearest;
}

fn lava_tube_color(world_pos: vec3<f32>, normal: vec3<f32>, texture_value: f32) -> vec3<f32> {
    let basalt_black = vec3<f32>(cave_params.rock_dark_r, cave_params.rock_dark_g, cave_params.rock_dark_b);
    let blue_glass = vec3<f32>(cave_params.rock_cool_r, cave_params.rock_cool_g, cave_params.rock_cool_b);
    let iron_skin = vec3<f32>(cave_params.rock_warm_r, cave_params.rock_warm_g, cave_params.rock_warm_b);
    let ash_dust = vec3<f32>(cave_params.rock_pale_r, cave_params.rock_pale_g, cave_params.rock_pale_b);
    let ash_gray = mix(blue_glass, vec3<f32>(0.190, 0.175, 0.150), 0.38);

    let center = vec3<f32>(
        cave_params.world_center.x,
        cave_params.world_center.y,
        cave_params.world_center.z
    );
    let radial = normalize(world_pos - center + vec3<f32>(0.001, 0.002, 0.003));
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var flow_axis = cross(up, radial);
    if (dot(flow_axis, flow_axis) < 0.001) {
        flow_axis = cross(vec3<f32>(1.0, 0.0, 0.0), radial);
    }
    flow_axis = normalize(flow_axis);
    let axial = dot(world_pos, flow_axis);
    let surface_uv = lava_surface_uv(world_pos, normal);
    let plate_scale = max(cave_params.rock_coarse_frequency * 0.018, 0.08);
    let warped_uv = surface_uv * plate_scale
                  + vec2<f32>(
                      noise(surface_uv * cave_params.rock_fine_noise_scale + vec2<f32>(11.0, 29.0)),
                      noise(surface_uv.yx * cave_params.rock_fine_noise_scale + vec2<f32>(53.0, 7.0))
                    ) * cave_params.rock_warp_strength * 0.12;

    let plate_gap = lava_plate_edge(warped_uv);
    let crack_thin = cave_params.rock_seam_low * 0.080 + 0.006;
    let crack_thick = max(crack_thin + 0.008, cave_params.rock_seam_high * 0.120 + 0.012);
    let plate_crack = 1.0 - smoothstep(crack_thin, crack_thick, plate_gap);
    let dark_joint = plate_crack * cave_params.rock_seam_darkening;

    let plate_id = floor(warped_uv);
    let plate_tone = noise(plate_id * 0.71 + vec2<f32>(3.0, 37.0));
    let scuff = noise(surface_uv * max(cave_params.rock_texture_scale, 0.005) + vec2<f32>(5.0, 37.0));
    let glass_sheen = smoothstep(0.52, 0.94, scuff) * cave_params.rock_grain_strength;

    let blister_noise = noise(plate_id * 1.37 + vec2<f32>(19.0, 11.0));
    let blister_low = clamp(cave_params.rock_fine_band_low * 0.5 + 0.5, 0.0, 0.98);
    let blister_high = max(blister_low + 0.02, clamp(cave_params.rock_fine_band_high * 0.5 + 0.5, 0.02, 1.0));
    let blister = smoothstep(blister_low, blister_high, blister_noise)
                * smoothstep(0.35, 0.90, scuff)
                * cave_params.rock_cool_mottle_strength;

    let wall = 1.0 - abs(normal.y);
    let ceiling = smoothstep(0.35, 0.95, normal.y);
    let floor = smoothstep(0.35, 0.95, -normal.y);
    let tube_vignette = wall * 0.12 + ceiling * 0.08 - floor * 0.04;
    let axial_polish = smoothstep(0.25, 0.85, noise(vec2<f32>(axial * 0.035, dot(world_pos, radial) * 0.025) + vec2<f32>(43.0, 17.0)));
    let cooled_drip = ceiling * smoothstep(0.72, 0.96, noise(surface_uv * 0.030 + vec2<f32>(83.0, 12.0))) * 0.20;
    let ash_deposit = floor
                    * smoothstep(0.45, 0.88, noise(surface_uv * 0.045 + vec2<f32>(101.0, 23.0)))
                    * cave_params.rock_patch_contrast;
    let shelf_scuff = wall
                    * smoothstep(0.60, 0.93, noise(vec2<f32>(axial * 0.055, world_pos.y * 0.038) + vec2<f32>(9.0, 67.0)))
                    * cave_params.rock_wall_line_strength;

    var color = mix(basalt_black, blue_glass, texture_value * 0.28 + glass_sheen + axial_polish * 0.08);
    color = mix(color, ash_gray, plate_tone * cave_params.rock_fine_band_strength * 0.45);
    color = mix(color, iron_skin, blister * 0.38);
    color *= 1.0 - dark_joint * 0.58;
    color += vec3<f32>(tube_vignette + cooled_drip);
    color = mix(color, ash_dust, ash_deposit * 0.32 + shelf_scuff * 0.18 + cooled_drip * 0.14);

    return clamp(color, vec3<f32>(cave_params.rock_min_color), vec3<f32>(cave_params.rock_max_color * 1.8));
}

fn rock_relief_normal(world_pos: vec3<f32>, normal: vec3<f32>, relief_depth: f32) -> vec3<f32> {
    if (relief_depth <= 0.001) {
        return normal;
    }

    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) < 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = normalize(cross(normal, tangent));
    let eps = max(0.5, 0.03 / max(cave_params.rock_texture_scale, 0.001));

    let h_l = sample_triplanar_texture(world_pos - tangent * eps, normal);
    let h_r = sample_triplanar_texture(world_pos + tangent * eps, normal);
    let h_d = sample_triplanar_texture(world_pos - bitangent * eps, normal);
    let h_u = sample_triplanar_texture(world_pos + bitangent * eps, normal);
    let relief = clamp(relief_depth * cave_params.rock_texture_scale * 0.75, 0.0, 1.5);
    let gradient = tangent * (h_r - h_l) + bitangent * (h_u - h_d);

    return normalize(normal - gradient * relief);
}

// Moss density sampling - trilinear interpolation from 128^3 grid.
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
// Moss height - multi-octave for better detail
// ============================================================
fn moss_height_lod(uv: vec2<f32>, octaves: i32) -> f32 {
    let freq = shadow_params.moss_noise_frequency;
    let lac  = shadow_params.moss_noise_lacunarity;
    let noise_type = shadow_params.moss_noise_type;

    // Base octave
    var h: f32;
    if (noise_type == 1u) {
        h = 1.0 - worley_noise(uv * freq);
    } else if (noise_type == 2u) {
        h = 1.0 - abs(noise(uv * freq) * 2.0 - 1.0);
    } else {
        h = noise(uv * freq);
    }

    // Second octave - finer detail at 2x frequency, half amplitude
    let h2 = noise(uv * freq * lac) * 0.5;

    // Third octave - micro detail at 4x frequency, quarter amplitude
    let h3 = noise(uv * freq * lac * lac) * 0.25;

    // Combine: base shape + fine detail + micro detail
    h = (h + h2 + h3) / 1.75; // normalize back to [0,1] range

    h = smoothstep(shadow_params.moss_height_sharpness_low, shadow_params.moss_height_sharpness_high, h);
    return h;
}

// Full-quality wrapper kept for reference - all call sites use moss_height_lod directly.

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
    
    // Triplanar texture sampling prevents stretching. Rock depth is expressed
    // as stable relief-normal perturbation, not a view-space world-position
    // offset; that avoids screen-center warping as the camera moves.
    let texture_value = sample_triplanar_texture(in.world_position, N);
    N = rock_relief_normal(in.world_position, N, cave_params.rock_parallax_depth);
    
    // Ambient occlusion approximation
    var ao = 0.5 + 0.5 * texture_value;
    
    var final_base_color: vec3<f32>;
    if (cave_params.appearance == 1u) {
        final_base_color = lava_tube_color(in.world_position, N, texture_value);
    } else {
        final_base_color = layered_rock_color(in.world_position, N, texture_value);
    }

    // -- Moss rendering with parallax and normal perturbation --------------
    let moss_sample_pos = in.world_position + N * shadow_params.cell_size * 0.5;
    let cam_dist = length(camera.camera_pos - in.world_position);
    let moss_amount = sample_moss_density(moss_sample_pos);

    var specular_strength = 1.0;
    if (moss_amount > 0.01) {
        let base_uv = get_triplanar_uv(in.world_position, N);

        // -- Parallax offset -----------------------------------------------
        // Shift UV in the view direction projected onto the surface plane.
        // This makes moss tufts appear to have real depth.
        var pom_uv = base_uv;
        let parallax_depth = shadow_params.moss_parallax_depth;
        if (parallax_depth > 0.001) {
            // Project view vector onto the surface tangent plane
            let view_tangent = V - dot(V, N) * N;
            // Use triplanar dominant axis to get a 2D tangent direction
            let abs_n = abs(N);
            var tangent_2d: vec2<f32>;
            if (abs_n.y > abs_n.x && abs_n.y > abs_n.z) {
                tangent_2d = vec2<f32>(dot(view_tangent, vec3<f32>(1.0, 0.0, 0.0)),
                                       dot(view_tangent, vec3<f32>(0.0, 0.0, 1.0)));
            } else if (abs_n.x > abs_n.z) {
                tangent_2d = vec2<f32>(dot(view_tangent, vec3<f32>(0.0, 1.0, 0.0)),
                                       dot(view_tangent, vec3<f32>(0.0, 0.0, 1.0)));
            } else {
                tangent_2d = vec2<f32>(dot(view_tangent, vec3<f32>(1.0, 0.0, 0.0)),
                                       dot(view_tangent, vec3<f32>(0.0, 1.0, 0.0)));
            }
            // Sample height at base UV, offset proportional to (1 - height)
            let h0 = moss_height_lod(base_uv, 1);
            pom_uv = base_uv + tangent_2d * shadow_params.moss_scale * parallax_depth * (1.0 - h0);
        }

        let h       = moss_height_lod(pom_uv, 1);
        let m_color = moss_color(pom_uv, h);

        // -- Normal perturbation from height gradient ----------------------
        // Finite-difference gradient of the height map perturbs the shading normal,
        // giving moss tufts a 3D bumpy appearance.
        let bump = shadow_params.moss_bump_strength * moss_amount;
        if (bump > 0.001) {
            let eps = 0.5 / shadow_params.moss_noise_frequency;
            let hx = moss_height_lod(pom_uv + vec2<f32>(eps, 0.0), 1)
                   - moss_height_lod(pom_uv - vec2<f32>(eps, 0.0), 1);
            let hy = moss_height_lod(pom_uv + vec2<f32>(0.0, eps), 1)
                   - moss_height_lod(pom_uv - vec2<f32>(0.0, eps), 1);
            // Build a perturbed normal in surface space and blend with geometric normal
            let abs_n = abs(N);
            var perturb: vec3<f32>;
            if (abs_n.y > abs_n.x && abs_n.y > abs_n.z) {
                perturb = vec3<f32>(hx, 0.0, hy);
            } else if (abs_n.x > abs_n.z) {
                perturb = vec3<f32>(0.0, hx, hy);
            } else {
                perturb = vec3<f32>(hx, hy, 0.0);
            }
            N = normalize(N + perturb * bump);
        }

        // Blend moss color: moss_amount drives coverage, h modulates shade within moss
        let color_blend = moss_amount * smoothstep(0.0, 0.3, h);
        final_base_color = mix(final_base_color, m_color, color_blend);

        specular_strength = mix(1.0, 0.1, moss_amount);

        let moss_ao = mix(0.4, 1.0, h);
        ao *= mix(1.0, moss_ao, moss_amount);
    }

    // Diffuse lighting (using potentially moss-perturbed normal)
    let diffuse = max(dot(N, light_dir), 0.0);
    
    // Specular lighting (Blinn-Phong, reduced on mossy surfaces)
    let H = normalize(light_dir + V);
    let specular = pow(max(dot(N, H), 0.0), cave_params.rock_specular_power) * specular_strength;
    
    // Combine lighting
    let offset_distance = mix(3.0, 6.0, shadow_params.shadow_quality) * shadow_params.cell_size;
    let shadow_sample_pos = in.world_position + N * offset_distance;
    let grid_size = shadow_params.cell_size * f32(shadow_params.grid_resolution);
    let grid_min = vec3<f32>(shadow_params.grid_origin_x, shadow_params.grid_origin_y, shadow_params.grid_origin_z);
    let grid_max = grid_min + vec3<f32>(grid_size, grid_size, grid_size);
    // Surfaces outside the light field grid (e.g. the world-boundary caps,
    // which extend slightly past world_radius) have no occlusion data.
    // Clamping their position onto the grid's outer shell would sample the
    // boundary-solid voxels and incorrectly darken them, so treat them as
    // fully sunlit instead - matching how the compute shader lights voxels
    // outside the world sphere.
    var shadow = 1.0;
    var sun_color = vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);
    if (all(in.world_position >= grid_min) && all(in.world_position <= grid_max)) {
        let clamped_pos = clamp(shadow_sample_pos, grid_min, grid_max);
        shadow = mix(1.0, sample_light_field_lod(clamped_pos), shadow_params.shadow_strength);
        sun_color = sample_light_color_field(clamped_pos);
    }
    // Ambient scales with sun_color so cave walls go pitch black when sun is off.
    let ambient = sun_color * cave_params.rock_ambient_strength * ao;
    var final_color = final_base_color * (ambient + sun_color * diffuse * cave_params.rock_diffuse_strength * shadow)
        + sun_color * vec3<f32>(specular * cave_params.rock_specular_strength * shadow);
    
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
