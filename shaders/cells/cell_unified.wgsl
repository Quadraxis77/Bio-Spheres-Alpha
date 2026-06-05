// Unified Cell Shader - 3-Layer Procedural Rendering
//
// All cell types rendered with a single pipeline:
//   Layer 1 (back):   Opaque cytoplasm fill
//   Layer 2 (middle): Type-specific organelle/internal patterns
//   Layer 3 (front):  Semi-transparent membrane with specular/fresnel
//
// Cell type is read from type_data_1.w to select the pattern function.
// Patterns are computed in cell-local 3D space (rotation-aware) so they
// rotate with the cell and show different cross-sections from different angles.

struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    lod_scale_factor: f32,
    lod_threshold_low: f32,
    lod_threshold_medium: f32,
    lod_threshold_high: f32,
}

struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    outline_width: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var hex_bake_texture: texture_2d<f32>;
@group(0) @binding(3) var hex_bake_sampler: sampler;

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

@group(1) @binding(0) var<uniform> shadow_params: ShadowFieldParams;
@group(1) @binding(1) var<storage, read> light_field: array<f32>;
@group(1) @binding(2) var<storage, read> light_color_field: array<vec4<f32>>;

// Sample light field at world position with optimized bilinear + linear interpolation
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
    
    // Optimized: Sample only 4 corners for bilinear, then linear blend between Z slices
    // This reduces memory reads from 8 to 4 + 1 = 5 (37.5% reduction)
    let c00 = light_field[x0 + y0 * res + z0 * res * res];
    let c10 = light_field[x1 + y0 * res + z0 * res * res];
    let c01 = light_field[x0 + y1 * res + z0 * res * res];
    let c11 = light_field[x1 + y1 * res + z0 * res * res];
    
    // Bilinear interpolation on Z slice 0
    let c0 = mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
    
    // For performance, only sample second Z slice if needed (when fz > 0.1)
    var c1 = c0; // Default to slice 0 value
    if (fz > 0.1) {
        let c001 = light_field[x0 + y0 * res + z1 * res * res];
        let c101 = light_field[x1 + y0 * res + z1 * res * res];
        let c011 = light_field[x0 + y1 * res + z1 * res * res];
        let c111 = light_field[x1 + y1 * res + z1 * res * res];
        
        // Bilinear interpolation on Z slice 1
        c1 = mix(mix(c001, c101, fx), mix(c011, c111, fx), fy);
    }
    
    // Linear blend between Z slices
    return mix(c0, c1, fz);
}

struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
    @location(3) visual_params: vec4<f32>,
    @location(4) rotation: vec4<f32>,
    @location(5) type_data_0: vec4<f32>,
    @location(6) type_data_1: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) center: vec3<f32>,
    @location(3) radius: f32,
    @location(4) visual_params: vec4<f32>,
    @location(5) cam_right: vec3<f32>,
    @location(6) cam_up: vec3<f32>,
    @location(7) to_camera: vec3<f32>,
    @location(8) @interpolate(flat) lod_level: f32,
    @location(9) @interpolate(flat) cell_type_and_debug: vec2<f32>,
    @location(10) @interpolate(flat) rotation: vec4<f32>,
    @location(11) @interpolate(flat) type_data_0: vec4<f32>,
    @location(12) @interpolate(flat) type_data_1: vec4<f32>,
    @location(13) @interpolate(flat) instance_index: u32,
}

const QUAD_POSITIONS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

const PI: f32 = 3.14159265359;

// ============================================================================
// Quaternion helpers
// ============================================================================

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let q_conj = vec4<f32>(-q.x, -q.y, -q.z, q.w);
    return quat_rotate(q_conj, v);
}

// ============================================================================
// Vertex Shader
// ============================================================================

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let quad_pos = QUAD_POSITIONS[vertex_index];

    // Billboard facing camera
    let to_camera = normalize(camera.camera_pos - instance.position);
    var right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    var up = cross(to_camera, right);

    if (length(right) < 0.001) {
        right = vec3<f32>(1.0, 0.0, 0.0);
        up = vec3<f32>(0.0, 0.0, 1.0);
    }

    // Calculate LOD based on screen-space size
    let camera_distance = max(length(instance.position - camera.camera_pos), 1.0);
    let screen_radius = (instance.radius / camera_distance) * camera.lod_scale_factor;

    var lod_level: u32;
    if (screen_radius < camera.lod_threshold_low) {
        lod_level = 0u;
    } else if (screen_radius < camera.lod_threshold_medium) {
        lod_level = 1u;
    } else if (screen_radius < camera.lod_threshold_high) {
        lod_level = 2u;
    } else {
        lod_level = 3u;
    }

    // Devorocyte (type 11): expand billboard to accommodate spikes extending beyond the sphere.
    // Spikes reach 0.75 radii beyond the surface, so we need radius * (1.1 + 0.75) = 1.85.
    let cell_type_vs = u32(round(instance.type_data_1.w));
    let billboard_scale = select(1.1, 1.85, cell_type_vs == 11u);
    let world_size_scaled = instance.radius * billboard_scale;
    let world_pos = instance.position + right * quad_pos.x * world_size_scaled + up * quad_pos.y * world_size_scaled;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    // Scale UV so that local_pos = uv*2-1 correctly maps to sphere-radius units.
    // For the expanded Devorocyte billboard (1.55x), quad_pos=1 corresponds to 1.409 sphere radii.
    // We encode this by scaling uv: local_pos = (uv*2-1) will then range to billboard_scale/1.1.
    // Standard cells: billboard_scale=1.1, ratio=1.0, uv=quad_pos*0.5+0.5 -> local_pos=quad_pos  [-1,1].
    // Devorocyte:     billboard_scale=1.85, ratio~=1.682, uv scaled -> local_pos  [-1.682, 1.682].
    let uv_scale = billboard_scale / 1.1;
    out.uv = quad_pos * uv_scale * 0.5 + 0.5;
    out.color = instance.color;
    out.center = instance.position;
    out.radius = instance.radius;
    out.visual_params = instance.visual_params;
    out.cam_right = right;
    out.cam_up = up;
    out.to_camera = to_camera;
    out.lod_level = f32(lod_level);
    out.cell_type_and_debug = vec2<f32>(instance.type_data_1.w, instance.type_data_1.z);
    out.rotation = instance.rotation;
    out.type_data_0 = instance.type_data_0;
    out.type_data_1 = instance.type_data_1;
    out.instance_index = instance_index;

    return out;
}

// ============================================================================
// Noise & Pattern Primitives
// ============================================================================

// Hash-based pseudo-random (fast, no texture needed)
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// Smooth value noise (3D)
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep

    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    let n00 = mix(n000, n100, u.x);
    let n10 = mix(n010, n110, u.x);
    let n01 = mix(n001, n101, u.x);
    let n11 = mix(n011, n111, u.x);

    let n0 = mix(n00, n10, u.y);
    let n1 = mix(n01, n11, u.y);

    return mix(n0, n1, u.z);
}

// 3D Voronoi distance (returns distance to nearest cell center)
fn voronoi_3d(p: vec3<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);

    var min_dist = 1.0;
    var second_dist = 1.0;

    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor = vec3<f32>(f32(x), f32(y), f32(z));
                let cell_center = hash33(i + neighbor);
                let diff = neighbor + cell_center - f;
                let d = dot(diff, diff);
                if (d < min_dist) {
                    second_dist = min_dist;
                    min_dist = d;
                } else if (d < second_dist) {
                    second_dist = d;
                }
            }
        }
    }

    return vec2<f32>(sqrt(min_dist), sqrt(second_dist));
}

// Concentric rings pattern
fn rings_pattern(p: vec3<f32>, frequency: f32, sharpness: f32) -> f32 {
    let r = length(p);
    let wave = sin(r * frequency * PI * 2.0);
    return smoothstep(-sharpness, sharpness, wave);
}

// Spots pattern (thresholded Voronoi)
fn spots_pattern(p: vec3<f32>, scale: f32, threshold: f32) -> f32 {
    let v = voronoi_3d(p * scale);
    return smoothstep(threshold - 0.05, threshold + 0.05, v.x);
}

// Streaks pattern (elongated along one axis)
fn streaks_pattern(p: vec3<f32>, frequency: f32, elongation: f32) -> f32 {
    let stretched = vec3<f32>(p.x, p.y * elongation, p.z);
    let n = value_noise_3d(stretched * frequency);
    return n;
}

// Blob pattern (large smooth noise features)
fn blob_pattern(p: vec3<f32>, scale: f32) -> f32 {
    let n1 = value_noise_3d(p * scale);
    let n2 = value_noise_3d(p * scale * 2.0 + 5.0) * 0.5;
    return clamp(n1 + n2 * 0.5, 0.0, 1.0);
}

// ============================================================================
// Icosphere Hex Ridge System (for photocyte nucleus)
// ============================================================================
// Proper icosahedron-based hex pattern. The Voronoi diagram of the 42 vertices
// of a frequency-2 geodesic sphere (12 original ico verts + 30 edge midpoints)
// produces 12 pentagons and 20 hexagons - a classic soccer ball / icosphere.
//
// type_data_0 per-instance params:
//   .x = subdivision    (1 or 2, default 2) - 1 = 12 pentagons, 2 = 12 pent + 20 hex
//   .y = ridge_width    (0.01-0.5, default 0.12)
//   .z = meander        (0.0-0.3, default 0.08)
//   .w = ridge_strength (0.0-0.5, default 0.15)
const MEANDER_SCALE: f32 = 8.0;

fn noise3(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        value_noise_3d(p + vec3<f32>(7.3, 0.0, 0.0)),
        value_noise_3d(p + vec3<f32>(0.0, 13.7, 0.0)),
        value_noise_3d(p + vec3<f32>(0.0, 0.0, 23.1))
    ) * 2.0 - 1.0;
}

const ICO_PHI: f32 = 1.618033988749895;

// 12 icosahedron base vertices
fn ico_base_vertex(idx: u32) -> vec3<f32> {
    switch (idx) {
        case 0u:  { return normalize(vec3<f32>( 0.0,  1.0,  ICO_PHI)); }
        case 1u:  { return normalize(vec3<f32>( 0.0, -1.0,  ICO_PHI)); }
        case 2u:  { return normalize(vec3<f32>( 0.0,  1.0, -ICO_PHI)); }
        case 3u:  { return normalize(vec3<f32>( 0.0, -1.0, -ICO_PHI)); }
        case 4u:  { return normalize(vec3<f32>( 1.0,  ICO_PHI, 0.0)); }
        case 5u:  { return normalize(vec3<f32>(-1.0,  ICO_PHI, 0.0)); }
        case 6u:  { return normalize(vec3<f32>( 1.0, -ICO_PHI, 0.0)); }
        case 7u:  { return normalize(vec3<f32>(-1.0, -ICO_PHI, 0.0)); }
        case 8u:  { return normalize(vec3<f32>( ICO_PHI, 0.0,  1.0)); }
        case 9u:  { return normalize(vec3<f32>(-ICO_PHI, 0.0,  1.0)); }
        case 10u: { return normalize(vec3<f32>( ICO_PHI, 0.0, -1.0)); }
        case 11u: { return normalize(vec3<f32>(-ICO_PHI, 0.0, -1.0)); }
        default:  { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

// 20 icosahedron faces (3 vertex indices each), packed as vec3<u32>
fn ico_face(idx: u32) -> vec3<u32> {
    switch (idx) {
        case 0u:  { return vec3<u32>(0, 1, 8); }
        case 1u:  { return vec3<u32>(0, 8, 4); }
        case 2u:  { return vec3<u32>(0, 4, 5); }
        case 3u:  { return vec3<u32>(0, 5, 9); }
        case 4u:  { return vec3<u32>(0, 9, 1); }
        case 5u:  { return vec3<u32>(1, 6, 8); }
        case 6u:  { return vec3<u32>(8, 6, 10); }
        case 7u:  { return vec3<u32>(8, 10, 4); }
        case 8u:  { return vec3<u32>(4, 10, 2); }
        case 9u:  { return vec3<u32>(4, 2, 5); }
        case 10u: { return vec3<u32>(5, 2, 11); }
        case 11u: { return vec3<u32>(5, 11, 9); }
        case 12u: { return vec3<u32>(9, 11, 7); }
        case 13u: { return vec3<u32>(9, 7, 1); }
        case 14u: { return vec3<u32>(1, 7, 6); }
        case 15u: { return vec3<u32>(3, 6, 7); }
        case 16u: { return vec3<u32>(3, 10, 6); }
        case 17u: { return vec3<u32>(3, 2, 10); }
        case 18u: { return vec3<u32>(3, 11, 2); }
        case 19u: { return vec3<u32>(3, 7, 11); }
        default:  { return vec3<u32>(0, 1, 2); }
    }
}

// Result from the hex triplet pattern
struct HexTripletResult {
    edge_dist: f32,   // Voronoi edge distance (d2 - d1)
    is_hex: bool,     // true if point is in a hex region (not a pentagon/vertex region)
}

// Convert a unit sphere direction to equirectangular UV coordinates.
fn dir_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let theta = atan2(dir.z, dir.x); // -pi..pi
    let phi = acos(clamp(dir.y, -1.0, 1.0)); // 0..pi
    let u = (theta / (2.0 * PI)) + 0.5; // 0..1
    let v = phi / PI; // 0..1
    return vec2<f32>(u, v);
}

// Sample the pre-baked hex pattern texture. Applies meander distortion before lookup.
// Returns HexTripletResult from a single texture sample - no Voronoi math at runtime.
fn hex_triplet(dir: vec3<f32>, meander_amount: f32) -> HexTripletResult {
    let meander = noise3(dir * MEANDER_SCALE) * meander_amount;
    let p = normalize(dir + meander);
    let uv = dir_to_equirect_uv(p);
    let sample = textureSampleLevel(hex_bake_texture, hex_bake_sampler, uv, 0.0);

    var result: HexTripletResult;
    result.edge_dist = sample.r;
    result.is_hex = (sample.g > 0.5);
    return result;
}

// Gaussian ridge: peaks at edge (edge_dist~=0), zero at face centers.
fn geodesic_ridge(edge_dist: f32, ridge_width: f32) -> f32 {
    return exp(-(edge_dist * edge_dist) / (ridge_width * ridge_width));
}

// ============================================================================
// Per-Type Internal Pattern Functions
// ============================================================================
// Each returns a vec3: (pattern_value, color_shift, unused)
// pattern_value: 0 = pure cytoplasm, 1 = full organelle coverage
// color_shift: how much to shift the color (positive = lighter, negative = darker)

// Type 0: Test Cell - Concentric rings pattern, configurable.
// type_data_0: x=ring_frequency, y=ring_sharpness, z=ring_brightness, w=unused
fn internals_test(p: vec3<f32>, r: f32, type_data_0: vec4<f32>) -> vec3<f32> {
    let freq      = clamp(type_data_0.x, 1.0, 20.0);
    let sharpness = clamp(type_data_0.y, 0.01, 1.0);
    let brightness = clamp(type_data_0.z, 0.0, 1.0);
    if (brightness < 0.01) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let ring = rings_pattern(p, freq, sharpness);
    return vec3<f32>(ring * brightness * 0.5, ring * brightness * 0.2 - 0.1, 0.0);
}

// Type 10: Embryocyte - Egg-like appearance.
// A warm amber yolk sphere visible through a translucent albumen interior.
// type_data_0: x=yolk_radius (0.3-0.7), y=yolk_offset_y (-0.3-0.0), z=yolk_brightness (0.5-1.5), w=unused
fn internals_embryocyte(p: vec3<f32>, r: f32, type_data_0: vec4<f32>) -> vec3<f32> {
    let yolk_radius    = clamp(type_data_0.x, 0.3, 0.7);
    let yolk_offset_y  = clamp(type_data_0.y, -0.35, 0.0);
    let yolk_bright    = clamp(type_data_0.z, 0.5, 1.5);

    // Yolk sphere: warm amber, offset slightly downward in cell-local space
    let yolk_center = vec3<f32>(0.0, yolk_offset_y, 0.0);
    let yolk_dist = length(p - yolk_center);

    // Soft yolk edge with a bright highlight toward the top
    let yolk_mask = smoothstep(yolk_radius + 0.04, yolk_radius - 0.06, yolk_dist);

    // Subtle internal gradient: brighter at top of yolk (light from above)
    let yolk_gradient = 0.7 + 0.3 * clamp((p.y - yolk_offset_y) / yolk_radius + 0.5, 0.0, 1.0);

    // Albumen haze: faint milky glow between yolk and membrane
    let albumen = smoothstep(0.9, 0.5, r) * 0.12;

    let pattern = yolk_mask * yolk_bright * yolk_gradient + albumen;
    // Yolk shifts color warm (positive = lighter/warmer via base_color tint)
    let color_shift = yolk_mask * 0.35 * yolk_bright;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 13: Gametocyte - Translucent reproductive cell with a glowing central nucleus.
// A bright pulsing nucleus is visible deep inside a lightly pigmented, softly-lit cytoplasm.
// type_data_0: x=merge_range (unused for visuals), y=unused, z=unused, w=unused
// Visual parameters come from CellTypeVisuals (param_a=pulse_speed, param_b=nucleus_glow).
fn internals_gametocyte(p: vec3<f32>, r: f32, current_time: f32) -> vec3<f32> {
    // Half-circle nucleus: radial mask clipped to the +Y hemisphere
    let nuc_dist = length(p);
    let radial_mask = smoothstep(0.38, 0.20, nuc_dist);

    // Clip to half: smooth fade across the equator so the flat edge isn't a hard seam
    let half_mask = smoothstep(-0.06, 0.06, p.y);

    let nuc_mask = radial_mask * half_mask;

    // Slow pulsing glow
    let pulse = 0.7 + 0.3 * sin(current_time * 1.5);
    let nuc_brightness = nuc_mask * pulse * 1.4;

    // Faint second ring for depth effect (also clipped to half)
    let ring = smoothstep(0.55, 0.48, nuc_dist) * smoothstep(0.20, 0.30, nuc_dist) * half_mask * 0.18;

    let pattern = nuc_brightness + ring;
    let color_shift = nuc_mask * 0.25;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 15: Memorocyte - Leaky-integrator memory cell.
// Concentric rings drift inward (memory charging) and fade (decaying).
// A dim ambient glow pulses slowly to suggest an analogue charge level.
fn internals_memorocyte(p: vec3<f32>, r: f32, current_time: f32) -> vec3<f32> {
    let dist = length(p);

    let ring_width = 0.06;
    let ring_speed = 0.35;
    let anim = fract(current_time * ring_speed);

    var ring_total = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let base_r = 0.75 - f32(i) * 0.155;
        let ring_r = base_r - anim * 0.12 * (1.0 + f32(i) * 0.3);
        let ring_mask = smoothstep(ring_width, 0.0, abs(dist - ring_r));
        let brightness = 0.55 + f32(i) * 0.12;
        ring_total += ring_mask * brightness;
    }

    let charge_pulse = 0.3 + 0.15 * sin(current_time * 0.7);
    let ambient_glow = smoothstep(0.75, 0.0, dist) * charge_pulse;
    let pattern = clamp(ring_total + ambient_glow, 0.0, 1.4);
    let color_shift = ring_total * 0.12 + ambient_glow * 0.08;
    return vec3<f32>(pattern, color_shift, 0.0);
}

fn light_color_load(ix: i32, iy: i32, iz: i32) -> vec4<f32> {
    let res = shadow_params.grid_resolution;
    let ires = i32(res);
    if (ix < 0 || ix >= ires || iy < 0 || iy >= ires || iz < 0 || iz >= ires) {
        return vec4<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b, 0.0);
    }
    return light_color_field[u32(ix) + u32(iy) * res + u32(iz) * res * res];
}

fn sample_light_color_field(world_pos: vec3<f32>) -> vec3<f32> {
    if (shadow_params.shadow_enabled == 0u) {
        return vec3<f32>(1.0, 1.0, 1.0);
    }

    let gx = (world_pos.x - shadow_params.grid_origin_x) / shadow_params.cell_size - 0.5;
    let gy = (world_pos.y - shadow_params.grid_origin_y) / shadow_params.cell_size - 0.5;
    let gz = (world_pos.z - shadow_params.grid_origin_z) / shadow_params.cell_size - 0.5;

    let x0 = i32(floor(gx)); let x1 = x0 + 1;
    let y0 = i32(floor(gy)); let y1 = y0 + 1;
    let z0 = i32(floor(gz)); let z1 = z0 + 1;
    let fx = fract(gx); let fy = fract(gy); let fz = fract(gz);

    let c000 = light_color_load(x0,y0,z0);
    let c100 = light_color_load(x1,y0,z0);
    let c010 = light_color_load(x0,y1,z0);
    let c110 = light_color_load(x1,y1,z0);
    let c001 = light_color_load(x0,y0,z1);
    let c101 = light_color_load(x1,y0,z1);
    let c011 = light_color_load(x0,y1,z1);
    let c111 = light_color_load(x1,y1,z1);

    let w000 = (1.0 - fx) * (1.0 - fy) * (1.0 - fz);
    let w100 = fx * (1.0 - fy) * (1.0 - fz);
    let w010 = (1.0 - fx) * fy * (1.0 - fz);
    let w110 = fx * fy * (1.0 - fz);
    let w001 = (1.0 - fx) * (1.0 - fy) * fz;
    let w101 = fx * (1.0 - fy) * fz;
    let w011 = (1.0 - fx) * fy * fz;
    let w111 = fx * fy * fz;

    let local_weight =
        c000.w * w000 + c100.w * w100 + c010.w * w010 + c110.w * w110 +
        c001.w * w001 + c101.w * w101 + c011.w * w011 + c111.w * w111;
    if (local_weight > 0.0001) {
        let local_color =
            c000.rgb * c000.w * w000 + c100.rgb * c100.w * w100 +
            c010.rgb * c010.w * w010 + c110.rgb * c110.w * w110 +
            c001.rgb * c001.w * w001 + c101.rgb * c101.w * w101 +
            c011.rgb * c011.w * w011 + c111.rgb * c111.w * w111;
        return local_color / local_weight;
    }

    return vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);
}

// Type 16: Luminocyte - signal-reactive lantern body.
// type_data_0: x=band_frequency, y=band_width, z=core_glow, w=color_shift
fn internals_luminocyte(p: vec3<f32>, r: f32, current_time: f32, type_data_0: vec4<f32>) -> vec3<f32> {
    let ring_freq  = max(type_data_0.x, 1.0);
    let ring_width = clamp(type_data_0.y, 0.03, 0.45);
    let core_glow  = clamp(type_data_0.z, 0.0, 2.0);
    let color_shift = clamp(type_data_0.w, 0.0, 1.0);

    let radial = length(p);

    // Radial core glow — bright centre fading outward.
    let core = 1.0 - smoothstep(0.0, 0.55, radial);

    // Concentric rings that pulse outward from the centre.
    let ring_phase = radial * ring_freq - current_time * 2.0;
    let ring_raw   = 0.5 + 0.5 * sin(ring_phase * 6.2831853);
    let rings      = 1.0 - smoothstep(ring_width, ring_width + 0.12, 1.0 - ring_raw);
    // Rings live in the mid-shell zone, not in the bright core or at the edge.
    let ring_zone  = smoothstep(0.15, 0.40, radial) * (1.0 - smoothstep(0.70, 0.92, radial));

    let light = clamp(core * core_glow + rings * ring_zone * 0.9, 0.0, 1.8);
    return vec3<f32>(light, color_shift * rings * ring_zone, 0.0);
}

// Type 14: Cognocyte - Signal-processing cell with a pulsing computational core
// and three pairs of radiating signal traces (one per axis).
// The traces pulse with 120-degree phase offsets suggesting active signal routing.
fn internals_cognocyte(p: vec3<f32>, r: f32, current_time: f32) -> vec3<f32> {
    let core_dist = length(p);
    let core_mask = smoothstep(0.24, 0.10, core_dist);

    let fast = sin(current_time * 6.2);
    let slow = 0.55 + 0.45 * sin(current_time * 1.9 + 0.4);
    let core_pulse = slow * (0.65 + 0.35 * fast);
    let core_bright = core_mask * core_pulse * 2.0;

    let tw      = 0.07;
    let start_r = 0.22;
    let end_r   = 0.82;

    let dist_from_x_axis = length(p.yz);
    let along_x          = abs(p.x);
    let trace_x = smoothstep(tw, tw * 0.2, dist_from_x_axis)
                * smoothstep(start_r, start_r + 0.08, along_x)
                * smoothstep(end_r,   end_r   - 0.10, along_x);

    let dist_from_y_axis = length(p.xz);
    let along_y          = abs(p.y);
    let trace_y = smoothstep(tw, tw * 0.2, dist_from_y_axis)
                * smoothstep(start_r, start_r + 0.08, along_y)
                * smoothstep(end_r,   end_r   - 0.10, along_y);

    let dist_from_z_axis = length(p.xy);
    let along_z          = abs(p.z);
    let trace_z = smoothstep(tw, tw * 0.2, dist_from_z_axis)
                * smoothstep(start_r, start_r + 0.08, along_z)
                * smoothstep(end_r,   end_r   - 0.10, along_z);

    let t    = current_time * 3.8;
    let p120 = 2.09439510;
    let px = 0.50 + 0.50 * sin(t);
    let py = 0.50 + 0.50 * sin(t + p120);
    let pz = 0.50 + 0.50 * sin(t + 2.0 * p120);

    let traces = trace_x * px + trace_y * py + trace_z * pz;
    let pattern     = clamp(core_bright + traces * 0.80, 0.0, 1.5);
    let color_shift = core_mask * 0.50 + traces * 0.20;
    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 7: Oculocyte - handled inline in fs_main (needs surface direction, not interior pos).

// Type 1: Flagellocyte - Same as test cell (tail is rendered separately).
fn internals_flagellocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Type 2: Phagocyte - nucleus sphere visible through cytoplasm.
// type_data_0: x=nucleus_radius (0.1-0.5), y=nucleus_darkness (0.1-0.8),
//              z=nucleus_sharpness (0.01-0.15), w=unused
fn internals_phagocyte(p: vec3<f32>, r: f32, type_data_0: vec4<f32>) -> vec3<f32> {
    let nuc_radius    = clamp(type_data_0.x, 0.1, 0.5);
    let nuc_darkness  = clamp(type_data_0.y, 0.1, 0.8);
    let nuc_sharpness = clamp(type_data_0.z, 0.01, 0.15);
    let nucleus_r = length(p);
    let nucleus = smoothstep(nuc_radius + nuc_sharpness, nuc_radius - nuc_sharpness, nucleus_r);

    let pattern = nucleus * 0.6;
    let color_shift = nucleus * -nuc_darkness;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 3: Photocyte - internals handled as inner sphere in fragment shader.
fn internals_photocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Type 4: Lipocyte - Oily looking internals.
// Large blobby fat droplets with bright highlights and dark boundaries.
// type_data_0: x=droplet_scale (1.0-8.0), y=droplet_threshold (0.2-0.6),
//              z=boundary_sharpness (0.05-0.3), w=brightness (0.3-1.0)
fn internals_lipocyte(p: vec3<f32>, r: f32, type_data_0: vec4<f32>) -> vec3<f32> {
    let droplet_scale  = clamp(type_data_0.x, 1.0, 8.0);
    let droplet_thresh = clamp(type_data_0.y, 0.2, 0.6);
    let boundary_sharp = clamp(type_data_0.z, 0.05, 0.3);
    let brightness     = clamp(type_data_0.w, 0.3, 1.0);

    let n1 = value_noise_3d(p * droplet_scale + 50.0);
    let n2 = value_noise_3d(p * droplet_scale * 1.67 + vec3<f32>(n1 * 2.0, 0.0, 0.0) + 70.0);
    let oily = n1 * 0.6 + n2 * 0.4;

    let droplet  = smoothstep(droplet_thresh - 0.05, droplet_thresh + 0.2, oily);
    let boundary = 1.0 - smoothstep(0.0, boundary_sharp, abs(oily - droplet_thresh));

    let pattern     = max(droplet * 0.85, boundary * 0.7) * brightness;
    let color_shift = (droplet * 0.5 - boundary * 0.35) * brightness;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 6: Glueocyte - Domain-warped Voronoi slime pattern.
// voro_scale: cell density (higher = more cells)
// border_width: thickness of the dark border between cells (0..1)
// meander: domain warp strength - how much borders wiggle
// border_dark: darkness of the border groove (0..1)
fn internals_glueocyte(surf: vec3<f32>, voro_scale: f32, border_width: f32, meander: f32, border_dark: f32, t: f32, cell_seed: f32, anim_speed: f32) -> vec3<f32> {
    // Per-cell random phase and speed offsets so each cell animates differently
    let phase  = cell_seed * 6.2831853;                        // 0..2pi offset
    let speed  = anim_speed * (1.0 + cell_seed * 0.4);        // 20% variation per cell
    let anim   = t * speed + phase;

    // Domain warp: distort the lookup position with noise so borders meander organically.
    // Animate the warp offset over time for a slow flowing/breathing motion.
    let warp_freq = voro_scale * 1.3;
    let wx = value_noise_3d(surf * warp_freq + vec3<f32>(1.7 + sin(anim * 0.7), 9.2, 3.4));
    let wy = value_noise_3d(surf * warp_freq + vec3<f32>(8.3, 2.8 + cos(anim * 0.5), 5.1));
    let wz = value_noise_3d(surf * warp_freq + vec3<f32>(4.1, 6.7, 1.9 + sin(anim * 0.6 + 1.1)));
    let warped = surf + vec3<f32>(wx - 0.5, wy - 0.5, wz - 0.5) * meander * 2.0;

    // Voronoi: find nearest and second-nearest feature points
    let sp = warped * voro_scale;
    let ip = floor(sp);
    var d1 = 1e9;
    var d2 = 1e9;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let cell = ip + vec3<f32>(f32(dx), f32(dy), f32(dz));
                let h1 = fract(sin(dot(cell, vec3<f32>(127.1, 311.7,  74.7))) * 43758.5453);
                let h2 = fract(sin(dot(cell, vec3<f32>(269.5, 183.3, 246.1))) * 43758.5453);
                let h3 = fract(sin(dot(cell, vec3<f32>(113.5, 271.9, 124.6))) * 43758.5453);
                let pt = cell + vec3<f32>(h1, h2, h3);
                let d = distance(sp, pt);
                if (d < d1) { d2 = d1; d1 = d; }
                else if (d < d2) { d2 = d; }
            }
        }
    }

    // Border = distance to the edge between two cells (0 at border, positive inside)
    let edge = d2 - d1;

    // Dark border groove
    let groove = 1.0 - smoothstep(0.0, border_width, edge);
    // Cell interior: slightly raised/bright toward center
    let interior = smoothstep(border_width * 0.5, border_width * 2.0, edge);
    // Wet gloss pooling at cell centers
    let gloss = pow(interior, 3.0) * 0.6;

    let pattern = groove * border_dark + interior * 0.4 + gloss * 0.35;
    let color_shift = gloss * 0.45 - groove * border_dark * 0.5;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 5: Buoyocyte - Gas bubbles scattered inside a hollow cell.
// 7 bubbles of varying sizes. Since p is in rotation-aware cell-local space,
// the bubbles rotate with the cell automatically.
// type_data_0: x=bubble_scale (0.5-2.0), y=rotation_speed (0.1-2.0),
//              z=wall_brightness (0.3-1.0), w=gas_brightness (0.5-1.5)
fn gas_bubble_sdf(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    // Perfect sphere - no noise to avoid breathing effect
    return length(p - center) - radius;
}

fn internals_buoyocyte(p: vec3<f32>, r: f32, time: f32, cell_index: u32, type_data_0: vec4<f32>) -> vec3<f32> {
    let bubble_scale    = clamp(type_data_0.x, 0.5, 2.0);
    let rot_speed_mult  = clamp(type_data_0.y, 0.1, 2.0);
    let wall_bright     = clamp(type_data_0.z, 0.3, 1.0);
    let gas_bright      = clamp(type_data_0.w, 0.5, 1.5);

    // Pseudo-random seed from cell index
    let seed  = fract(f32(cell_index) * 12.9898);
    let seed2 = fract(f32(cell_index) * 78.233);

    // Randomized rotation speed and phase per cell
    let rotation_speed = rot_speed_mult * (0.3 + seed * 0.4);
    let phase_offset = seed2 * 6.283;
    let angle = time * rotation_speed + phase_offset;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    
    // Randomized bubble positions (offsets from base positions)
    let offset0 = vec3<f32>(seed * 0.1, seed2 * 0.1, (seed + seed2) * 0.1);
    let offset1 = vec3<f32>(seed2 * 0.15, (seed + 0.5) * 0.1, seed * 0.12);
    let offset2 = vec3<f32>((seed + 0.3) * 0.12, seed2 * 0.08, (seed2 + 0.7) * 0.15);
    let offset3 = vec3<f32>((seed2 + 0.2) * 0.1, seed * 0.13, (seed + 0.4) * 0.11);
    let offset4 = vec3<f32>(seed * 0.14, (seed2 + 0.6) * 0.09, (seed + 0.8) * 0.1);
    let offset5 = vec3<f32>((seed + 0.1) * 0.11, seed2 * 0.12, seed * 0.08);
    let offset6 = vec3<f32>((seed2 + 0.4) * 0.09, (seed + 0.2) * 0.11, seed2 * 0.13);
    
    // Base positions + random offsets, then rotate
    let base0 = vec3<f32>( 0.00,  0.25,  0.00) + offset0;
    let base1 = vec3<f32>(-0.28,  0.05,  0.15) + offset1;
    let base2 = vec3<f32>( 0.22, -0.10, -0.20) + offset2;
    let base3 = vec3<f32>( 0.05, -0.25,  0.22) + offset3;
    let base4 = vec3<f32>(-0.18, -0.22, -0.12) + offset4;
    let base5 = vec3<f32>( 0.30,  0.18, -0.05) + offset5;
    let base6 = vec3<f32>(-0.10,  0.10, -0.30) + offset6;
    
    // Inline rotation: rotate randomized positions around Y axis
    let rot0 = vec3<f32>(base0.x * cos_a - base0.z * sin_a, base0.y, base0.x * sin_a + base0.z * cos_a);
    let rot1 = vec3<f32>(base1.x * cos_a - base1.z * sin_a, base1.y, base1.x * sin_a + base1.z * cos_a);
    let rot2 = vec3<f32>(base2.x * cos_a - base2.z * sin_a, base2.y, base2.x * sin_a + base2.z * cos_a);
    let rot3 = vec3<f32>(base3.x * cos_a - base3.z * sin_a, base3.y, base3.x * sin_a + base3.z * cos_a);
    let rot4 = vec3<f32>(base4.x * cos_a - base4.z * sin_a, base4.y, base4.x * sin_a + base4.z * cos_a);
    let rot5 = vec3<f32>(base5.x * cos_a - base5.z * sin_a, base5.y, base5.x * sin_a + base5.z * cos_a);
    let rot6 = vec3<f32>(base6.x * cos_a - base6.z * sin_a, base6.y, base6.x * sin_a + base6.z * cos_a);
    
    // 7 gas bubbles at rotating positions with varying sizes, scaled by bubble_scale
    let d0 = gas_bubble_sdf(p, rot0, 0.30 * bubble_scale);
    let d1 = gas_bubble_sdf(p, rot1, 0.22 * bubble_scale);
    let d2 = gas_bubble_sdf(p, rot2, 0.20 * bubble_scale);
    let d3 = gas_bubble_sdf(p, rot3, 0.18 * bubble_scale);
    let d4 = gas_bubble_sdf(p, rot4, 0.16 * bubble_scale);
    let d5 = gas_bubble_sdf(p, rot5, 0.14 * bubble_scale);
    let d6 = gas_bubble_sdf(p, rot6, 0.12 * bubble_scale);

    // Soft masks (negative SDF = inside) - wider soft edges for better visibility
    let b0 = 1.0 - smoothstep(-0.04, 0.02, d0);
    let b1 = 1.0 - smoothstep(-0.04, 0.02, d1);
    let b2 = 1.0 - smoothstep(-0.04, 0.02, d2);
    let b3 = 1.0 - smoothstep(-0.04, 0.02, d3);
    let b4 = 1.0 - smoothstep(-0.04, 0.02, d4);
    let b5 = 1.0 - smoothstep(-0.04, 0.02, d5);
    let b6 = 1.0 - smoothstep(-0.04, 0.02, d6);

    // Union of all bubbles
    let gas = max(b0, max(b1, max(b2, max(b3, max(b4, max(b5, b6))))));

    // Membrane walls (bright ring at each bubble surface)
    let w0 = exp(-200.0 * d0 * d0) * step(-0.1, -d0);
    let w1 = exp(-200.0 * d1 * d1) * step(-0.1, -d1);
    let w2 = exp(-200.0 * d2 * d2) * step(-0.1, -d2);
    let w3 = exp(-200.0 * d3 * d3) * step(-0.1, -d3);
    let w4 = exp(-200.0 * d4 * d4) * step(-0.1, -d4);
    let w5 = exp(-200.0 * d5 * d5) * step(-0.1, -d5);
    let w6 = exp(-200.0 * d6 * d6) * step(-0.1, -d6);
    let walls = max(w0, max(w1, max(w2, max(w3, max(w4, max(w5, w6))))));

    // Specular highlight
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.2));
    let spec = pow(max(0.0, dot(normalize(p), light_dir)), 10.0) * gas;

    // Compose
    let pattern = (1.0 - gas) * 0.08
               + gas * 0.90 * gas_bright
               + walls * 0.80 * wall_bright
               + spec * 0.3;

    let color_shift = gas * 0.60 * gas_bright
                    + walls * 0.40 * wall_bright
                    + spec * 0.2;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 9: Myocyte - Peripheral flattened nuclei (syncytial cell with multiple nuclei).
// Nuclei are elongated ellipsoids near the membrane, compressed along the fiber axis (local Z).
// type_data_0: x=line_freq, y=bulge_strength, z=warp_amt, w=sarc_freq_packed
fn internals_myocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    // Two nuclei offset toward the membrane in XY, flattened along Z (fiber axis)
    let n1c = vec3<f32>( 0.52,  0.32,  0.05);
    let n2c = vec3<f32>(-0.50, -0.30,  0.10);

    // Flattened ellipsoid SDF: scale Z up to squash it, keeping XY round
    let p1  = p - n1c;
    let p2  = p - n2c;
    let e1  = length(vec3<f32>(p1.x * 0.90, p1.y * 1.05, p1.z * 2.30)) - 0.115;
    let e2  = length(vec3<f32>(p2.x * 1.00, p2.y * 0.88, p2.z * 2.50)) - 0.100;

    let m1 = smoothstep( 0.025, -0.008, e1);
    let m2 = smoothstep( 0.025, -0.008, e2);
    let nucleus = max(m1, m2);

    // Nuclei are significantly darker and slightly violet-tinted (heterochromatin)
    let color_shift = nucleus * -0.58;
    let pattern     = nucleus * 0.80;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// ============================================================================
// Pattern Dispatcher
// ============================================================================

fn get_internals(cell_type: u32, p: vec3<f32>, r: f32, cell_index: u32, type_data_0: vec4<f32>) -> vec3<f32> {
    switch (cell_type) {
        case 0u: { return internals_test(p, r, type_data_0); }
        case 1u: { return internals_flagellocyte(p, r); }
        case 2u: { return internals_phagocyte(p, r, type_data_0); }
        case 3u: { return internals_photocyte(p, r); }
        case 4u: { return internals_lipocyte(p, r, type_data_0); }
        case 5u: { return internals_buoyocyte(p, r, camera.time, cell_index, type_data_0); }
        case 6u: { return vec3<f32>(0.0); } // handled inline in fs_main using surface dir
        case 7u: { return vec3<f32>(0.0); } // Oculocyte: handled inline in fs_main using surface dir
        case 8u: { return vec3<f32>(0.0); } // Ciliocyte: handled inline in fs_main
        case 9u: { return internals_myocyte(p, r); } // Peripheral nuclei; surface pattern handled inline
        case 10u: { return internals_embryocyte(p, r, type_data_0); }
        case 13u: { return internals_gametocyte(p, r, camera.time); }
        case 14u: { return internals_cognocyte(p, r, camera.time); }
        case 15u: { return internals_memorocyte(p, r, camera.time); }
        case 16u: { return internals_luminocyte(p, r, camera.time, type_data_0); }
        default: { return internals_test(p, r, type_data_0); }
    }
}

// ============================================================================
// Membrane Properties Per Type
// ============================================================================

struct MembraneParams {
    thickness: f32,      // 0.0 - 0.3 (how thick the membrane shell appears)
    opacity: f32,        // 0.0 - 1.0 (how opaque the membrane is)
    rim_power: f32,      // 1.0 - 5.0 (how sharp the rim darkening is)
    color_darken: f32,   // 0.0 - 1.0 (how much darker the membrane is vs cytoplasm)
}

fn get_membrane_params(cell_type: u32) -> MembraneParams {
    var m: MembraneParams;
    switch (cell_type) {
        case 0u: { // Test - standard membrane
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 1u: { // Flagellocyte - same as test
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 2u: { // Phagocyte - standard membrane
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 3u: { // Photocyte - translucent hex membrane over solar panel internals
            m.thickness = 0.05;
            m.opacity = 0.35;
            m.rim_power = 2.0;
            m.color_darken = 0.2;
        }
        case 4u: { // Lipocyte - translucent to show oily internals
            m.thickness = 0.04;
            m.opacity = 0.3;
            m.rim_power = 2.0;
            m.color_darken = 0.15;
        }
        case 5u: { // Buoyocyte - thick membrane for gas bladder
            m.thickness = 0.08;
            m.opacity = 0.4;
            m.rim_power = 3.0;
            m.color_darken = 0.3;
        }
        case 6u: { // Glueocyte - sticky cell, thin membrane
            m.thickness = 0.05;
            m.opacity = 0.45;
            m.rim_power = 2.5;
            m.color_darken = 0.2;
        }
        case 7u: { // Oculocyte - glossy cornea, high specular
            m.thickness = 0.04;
            m.opacity = 0.35;
            m.rim_power = 1.8;
            m.color_darken = 0.1;
        }
        case 8u: { // Ciliocyte - standard membrane, rings are normal perturbation only
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 9u: { // Myocyte - semi-translucent membrane reveals fiber+sarcomere texture beneath
            m.thickness = 0.055;
            m.opacity = 0.42;
            m.rim_power = 2.5;
            m.color_darken = 0.22;
        }
        case 10u: { // Embryocyte - thick chalky eggshell membrane, matte and opaque
            m.thickness = 0.10;
            m.opacity = 0.65;
            m.rim_power = 3.5;
            m.color_darken = 0.35;
        }
        case 13u: { // Gametocyte - thin, translucent membrane with strong fresnel rim
            m.thickness = 0.04;
            m.opacity = 0.35;
            m.rim_power = 1.8;
            m.color_darken = 0.15;
        }
        case 14u: { // Cognocyte - moderately translucent to reveal internal traces
            m.thickness = 0.055;
            m.opacity = 0.40;
            m.rim_power = 3.2;
            m.color_darken = 0.28;
        }
        case 15u: { // Memorocyte - very translucent glassy shell, rings visible within
            m.thickness = 0.04;
            m.opacity = 0.28;
            m.rim_power = 2.0;
            m.color_darken = 0.12;
        }
        case 16u: { // Luminocyte - clear bright membrane with strong fresnel glow
            m.thickness = 0.035;
            m.opacity = 0.30;
            m.rim_power = 1.7;
            m.color_darken = 0.05;
        }
        default: {
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
    }
    return m;
}

// ============================================================================
// Fragment Shader - Cheap 3-Layer Analytical Compositing
//
// No ray marching. Three sample points on the same view ray through the sphere:
//   1. Back wall  (z = -sqrt(1-r^2)) -> opaque cytoplasm background
//   2. Midpoint   (z = 0 plane)     -> organelle pattern sampled at interior 3D pos
//   3. Front wall (z = +sqrt(1-r^2)) -> semi-transparent membrane + specular/fresnel
//
// Composited back-to-front in a single pass.
// ============================================================================

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let lod = u32(in.lod_level);
    let cell_type = u32(round(in.cell_type_and_debug.x));
    let debug_colors = in.cell_type_and_debug.y;

    // Billboard-space position [-1, 1] for sphere, potentially larger for Devorocyte spikes
    let local_pos = in.uv * 2.0 - 1.0;
    let r2 = dot(local_pos, local_pos);

    // For non-Devorocyte cells, discard outside the sphere as usual.
    // For Devorocyte (type 11), the billboard is expanded - pixels outside r2>1 may be spikes.
    let is_devorocyte = (cell_type == 11u);
    if (r2 > 1.0 && !is_devorocyte) {
        discard;
    }

    // Anti-aliased edge
    var aa_width: f32;
    switch (lod) {
        case 0u: { aa_width = 0.06; }
        case 1u: { aa_width = 0.03; }
        case 2u: { aa_width = 0.015; }
        case 3u: { aa_width = 0.008; }
        default: { aa_width = 0.03; }
    }
    let r = sqrt(r2);
    let edge_alpha = smoothstep(1.0, 1.0 - aa_width, r);

    // ====================================================================
    // Three sample points along the view ray through the unit sphere
    // ====================================================================
    let z_front = sqrt(max(0.0, 1.0 - r2));

    // Billboard-space positions for each layer
    let front_pos = vec3<f32>(local_pos.x, local_pos.y, z_front);   // front shell
    let mid_pos   = vec3<f32>(local_pos.x, local_pos.y, 0.0);       // interior midplane

    // World-space front normal (for membrane lighting)
    let world_normal_front = normalize(
        in.cam_right * front_pos.x +
        in.cam_up * front_pos.y +
        in.to_camera * front_pos.z
    );

    // Transform midpoint to cell-local 3D space (rotation-aware)
    // mid_pos is (x, y, 0) in billboard space - a flat cross-section through the sphere.
    // Do NOT normalize: we want the actual interior position, not a direction on the shell.
    let world_mid_vec = in.cam_right * mid_pos.x + in.cam_up * mid_pos.y;
    let interior_pos = quat_rotate_inverse(in.rotation, world_mid_vec);

    let base_color = in.color.rgb;
    let membrane = get_membrane_params(cell_type);
    let light_dir = normalize(lighting.light_dir);
    let view_dir = in.to_camera;

    // ====================================================================
    // Layer 1+2: Base cytoplasm + interior organelles
    // ====================================================================
    // Start with a uniform base color (no separate back-wall lighting to avoid disc artifact)
    var interior_result = base_color * 0.75;

    {
        // Get cell index from instance data (assuming it's packed in w component)
        let cell_index = u32(in.instance_index);
        let internals = get_internals(cell_type, interior_pos, r, cell_index, in.type_data_0);
        let pattern_value = internals.x;
        let color_shift = internals.y;

        // Organelle color: shifted version of base color
        let organelle_color = base_color * (0.75 + color_shift);

        // Composite organelles over base cytoplasm
        interior_result = mix(interior_result, organelle_color, pattern_value);
    }

    // ====================================================================
    // Glueocyte (type 6): Domain-warped Voronoi slime (LOD >= 1)
    // type_data_0: x=voro_scale, y=border_width, z=meander, w=border_dark
    // ====================================================================
    if (cell_type == 6u) {
        let surf_local = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));
        let cell_seed = fract(f32(in.instance_index) * 0.6180339887); // golden ratio hash for per-cell variation
        let anim_speed = in.type_data_1.y; // membrane_noise_speed -> animation speed multiplier
        let slime = internals_glueocyte(
            surf_local,
            in.type_data_0.x,  // voro_scale   (goldberg_scale)
            in.type_data_0.y,  // border_width  (goldberg_ridge_width)
            in.type_data_0.z,  // meander       (goldberg_meander)
            in.type_data_0.w,  // border_dark   (goldberg_ridge_strength)
            camera.time,
            cell_seed,
            anim_speed,
        );
        let slime_color = base_color * (0.75 + slime.y);
        interior_result = mix(interior_result, slime_color, slime.x);
    }

    // ====================================================================
    // Oculocyte (type 7): Forward-facing eye (LOD >= 1)
    // The eye is centered on the cell's forward axis (local +Z).
    // type_data_0: x=pupil_size, y=iris_freq, z=iris_texture, w=pupil_dark
    // ====================================================================
    if (cell_type == 7u) {
        // Surface direction in cell-local space (rotation-aware, on the front shell)
        let surf_local = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));

        let pupil_size   = clamp(in.type_data_0.x, 0.1, 0.45);
        let iris_freq    = clamp(in.type_data_0.y, 2.0, 16.0);
        let iris_texture = clamp(in.type_data_0.z, 0.0, 0.6);
        let pupil_dark   = clamp(in.type_data_0.w, 0.5, 1.0);

        // Dot product with forward axis: 1.0 = dead center, 0.0 = equator, -1.0 = back
        let fwd_dot = surf_local.z; // local +Z is the gaze direction

        // Angular distance from the forward pole (0 = center, 1 = equator)
        let ang_dist = sqrt(max(0.0, 1.0 - fwd_dot * fwd_dot));

        // Iris outer radius in angular space
        let iris_outer = pupil_size + 0.38;

        // Pupil: dark disc at the forward pole
        let pupil_mask = smoothstep(pupil_size + 0.015, pupil_size - 0.02, ang_dist);

        // Iris: annular band around the pupil
        let iris_mask = smoothstep(iris_outer + 0.02, iris_outer - 0.02, ang_dist)
                      * (1.0 - pupil_mask);

        // Sclera: bright area outside the iris on the front hemisphere
        let sclera_mask = smoothstep(iris_outer + 0.06, iris_outer + 0.01, ang_dist)
                        * smoothstep(0.0, 0.12, fwd_dot)
                        * (1.0 - iris_mask) * (1.0 - pupil_mask);

        // Iris radial striations: spokes from the pupil center
        let angle = atan2(surf_local.y, surf_local.x);
        let stria_a = sin(angle * iris_freq) * 0.5 + 0.5;
        let stria_b = sin(angle * iris_freq * 1.618 + 0.9) * 0.5 + 0.5;
        let iris_detail = mix(stria_a, stria_b, iris_texture) * iris_mask;

        // Cornea specular: bright highlight near the forward pole
        let cornea_spec = pow(max(0.0, fwd_dot), 12.0) * 0.5 * (1.0 - pupil_mask);

        // Compose over interior
        // Sclera: bright white
        let sclera_color = vec3<f32>(0.95, 0.95, 0.92) * (0.8 + 0.2 * fwd_dot);
        // Iris: base_color tinted with striation detail
        let iris_color = base_color * (0.6 + iris_detail * 0.6);
        // Pupil: very dark
        let pupil_color = base_color * (1.0 - pupil_dark);

        var eye_color = interior_result;
        eye_color = mix(eye_color, sclera_color, sclera_mask * 0.85);
        eye_color = mix(eye_color, iris_color,   iris_mask   * 0.9);
        eye_color = mix(eye_color, pupil_color,  pupil_mask  * 0.95);
        eye_color = eye_color + vec3<f32>(cornea_spec);

        interior_result = eye_color;
    }

    // ====================================================================
    // Myocyte (type 9): Sarcomere striations + longitudinal myofibril bundles (LOD >= 1)
    //
    // Two overlapping patterns produce the classic striated-muscle appearance:
    //   1. Longitudinal myofibrils - azimuthal (around Y) cosine bands; each fiber
    //      looks like a convex cylinder from the normal perturbation below.
    //      Fibers converge at the Y poles (forward / rear of the cell).
    //   2. Sarcomere cross-striations - periodic transverse bands along local Y:
    //        Z-disc (very dark) -> I-band (light, actin) -> A-band (dark, myosin+actin)
    //        -> H-zone (medium, myosin only) -> M-line (dark seam) -> H-zone -> A-band
    //        -> I-band -> Z-disc (next sarcomere)
    //
    // type_data_0: x=line_freq (fiber count), y=bulge_strength, z=warp_amt,
    // ====================================================================
    // Myocyte (type 9): Longitudinal fiber bundles converging at forward/rear poles
    // Lines run like meridians on a globe - converging at local +Z (forward)
    // and local -Z (rear). type_data_0: x=line_freq, y=bulge_str, z=warp, w=unused
    // ====================================================================
    if (cell_type == 9u) {
        let surf_local = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));

        let line_freq = clamp(in.type_data_0.x, 4.0, 20.0);
        let warp_amt  = clamp(in.type_data_0.z, 0.0, 0.30);

        // Angle around local Z - lines of constant angle are meridians
        // converging at the forward (+Z) and rear (-Z) poles
        let angle = atan2(surf_local.y, surf_local.x);
        let wn1 = value_noise_3d(surf_local * 4.5 + vec3<f32>(2.3, 8.1, 0.0));
        let wn2 = value_noise_3d(surf_local * 9.0 + vec3<f32>(5.1, 1.7, 3.3)) * 0.5;
        let warp_n = (wn1 + wn2 - 0.75) * 2.0;
        let angle_warped = angle + warp_n * warp_amt;

        // Squared cosine: wide bright crowns, narrow dark grooves
        let fiber_cos    = cos(angle_warped * line_freq);
        let fiber_t      = fiber_cos * 0.5 + 0.5;
        let fiber_bright = fiber_t * fiber_t;

        // Groove = 0.6, crown = 1.0
        interior_result = interior_result * (0.60 + fiber_bright * 0.40);
    }

    // ====================================================================
    // Ciliocyte (type 8): Rolling ring color ripple (LOD >= 1)
    // type_data_0: x=effective_speed, y=ring_frequency, z=ring_depth, w=ring_speed
    // Rings scroll along local Z axis proportional to effective_speed.
    // Static when effective_speed is 0.
    // ====================================================================
    if (cell_type == 8u) {
        // Get local-space direction on sphere surface (rotation-aware)
        let cilia_local_dir = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));

        // Read ring parameters from type_data
        let effective_speed = in.type_data_0.x;
        let ring_freq = in.type_data_0.y;
        let ring_depth = in.type_data_0.z;
        let ring_speed = in.type_data_0.w;

        // Ring phase: position along forward axis (local Z) + time scroll
        // When effective_speed is 0, time term vanishes -> static rings
        let phase = cilia_local_dir.z * ring_freq - camera.time * ring_speed * effective_speed;

        // Sinusoidal ring wave in [-1, 1]
        let ring_wave = sin(phase * 6.283185);

        // t in [0, 1]: 0 = trough, 1 = peak
        let t = ring_wave * 0.5 + 0.5;

        // Trough color: darker, slightly warm/purple (compressed cilia)
        let trough_color = base_color * (1.0 - ring_depth * 0.6)
                         * vec3<f32>(0.85, 0.80, 1.0);

        // Peak color: brighter, slightly cool/cyan (extended cilia catching light)
        let peak_color   = base_color * (1.0 + ring_depth * 0.45)
                         * vec3<f32>(1.0, 1.05, 1.15);

        // Smooth blend between trough and peak
        interior_result = mix(trough_color, peak_color, smoothstep(0.0, 1.0, t));
    }

    // ====================================================================
    // Photocyte (type 3): Inner hex-patterned nucleus sphere (front + back)
    // ====================================================================
    if (cell_type == 3u) {
        let nucleus_radius = in.type_data_1.x;
        let inner_r2 = nucleus_radius * nucleus_radius;
        let hit_r2 = r2;
        if (hit_r2 < inner_r2) {
            let nz = sqrt(inner_r2 - hit_r2);
            let ridge_width = in.type_data_0.y;
            let meander = in.type_data_0.z;
            let ridge_strength = in.type_data_0.w;
            let nr = sqrt(hit_r2) / nucleus_radius;
            let nucleus_edge = smoothstep(1.0, 0.97, nr);
            let n_half = normalize(-light_dir + view_dir);

            // --- Back face first (behind, seen through front gaps) ---
            let back_billboard = vec3<f32>(local_pos.x, local_pos.y, -nz);
            let back_world_normal = normalize(
                in.cam_right * back_billboard.x +
                in.cam_up * back_billboard.y +
                in.to_camera * back_billboard.z
            );
            let back_local_dir = normalize(quat_rotate_inverse(in.rotation, back_world_normal));
            let back_hex = hex_triplet(back_local_dir, meander);

            if (back_hex.is_hex) {
                let back_h_c = geodesic_ridge(back_hex.edge_dist, ridge_width);
                let back_face_interior = smoothstep(0.3, 0.7, 1.0 - back_h_c);

                let back_lit_normal = -back_world_normal;
                let back_ndotl = max(dot(back_lit_normal, -light_dir), 0.0);
                let back_lit = 0.25 + 0.5 * back_ndotl;
                let back_color = base_color * back_lit * 0.9;

                interior_result = mix(interior_result, back_color, back_face_interior * nucleus_edge);
            }

            // --- Front face (on top) ---
            let front_billboard = vec3<f32>(local_pos.x, local_pos.y, nz);
            let front_world_normal = normalize(
                in.cam_right * front_billboard.x +
                in.cam_up * front_billboard.y +
                in.to_camera * front_billboard.z
            );
            let front_local_dir = normalize(quat_rotate_inverse(in.rotation, front_world_normal));
            let front_hex = hex_triplet(front_local_dir, meander);

            if (front_hex.is_hex) {
                let front_h_c = geodesic_ridge(front_hex.edge_dist, ridge_width);
                let front_face_interior = smoothstep(0.3, 0.7, 1.0 - front_h_c);

                // Use edge_dist to darken edges slightly instead of expensive gradient normal perturbation
                let edge_darken = 1.0 - front_h_c * ridge_strength * 2.0;

                let front_ndotl = max(dot(front_world_normal, -light_dir), 0.0);
                let front_lit = (0.3 + 0.7 * front_ndotl) * edge_darken;
                let front_spec = pow(max(dot(front_world_normal, n_half), 0.0), 20.0) * 0.4;
                let front_color = base_color * front_lit * 1.2 + vec3<f32>(front_spec);

                interior_result = mix(interior_result, front_color, front_face_interior * nucleus_edge);
            }
        }
    }

    // ====================================================================
    // Layer 3 (FRONT): Semi-transparent membrane
    // ====================================================================
    // Membrane opacity: thin at center (see through), gently thicker at edges.
    // Use sqrt falloff instead of 1/z to avoid the rim going fully opaque.
    let rim_factor = 1.0 - z_front; // 0 at center, 1 at edge
    var membrane_alpha = (membrane.thickness + rim_factor * rim_factor * 0.3) * membrane.opacity;

    // Membrane color (darker than base)
    var membrane_color = base_color * (1.0 - membrane.color_darken);

    // Membrane normal (no per-type perturbation currently)
    var perturbed_normal = world_normal_front;

    // Ciliocyte (type 8): Perturb membrane normal to create rolling ridge highlights
    if (cell_type == 8u) {
        let cilia_norm_local_dir = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));

        let eff_speed = in.type_data_0.x;
        let r_freq = in.type_data_0.y;
        let r_depth = in.type_data_0.z;
        let r_speed = in.type_data_0.w;

        let norm_phase = cilia_norm_local_dir.z * r_freq - camera.time * r_speed * eff_speed;
        let norm_ring_wave = sin(norm_phase * 6.283185);

        // Perturb normal along the cell's world-space forward axis
        let forward_world = quat_rotate(in.rotation, vec3<f32>(0.0, 0.0, 1.0));
        let perturb_strength = r_depth * norm_ring_wave;
        perturbed_normal = normalize(perturbed_normal + forward_world * perturb_strength);
    }

    // Myocyte (type 9): Normal perturbation for both fiber bundles and sarcomere ridges
    if (cell_type == 9u) {
        let myo_surf   = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));
        let line_freq  = clamp(in.type_data_0.x, 4.0, 20.0);
        let bulge_str  = clamp(in.type_data_0.y, 0.1, 0.90);
        let warp_amt   = clamp(in.type_data_0.z, 0.0, 0.30);

        // -- Fiber bundle normals (azimuthal bumps around local forward Z) --
        // Matches the color pass: atan2(y, x) orbits around local +Z so fibers
        // converge at the cell's own forward and rear poles.
        let angle = atan2(myo_surf.y, myo_surf.x);
        let wn1 = value_noise_3d(myo_surf * 4.5 + vec3<f32>(2.3, 8.1, 0.0));
        let wn2 = value_noise_3d(myo_surf * 9.0 + vec3<f32>(5.1, 1.7, 3.3)) * 0.5;
        let warp_n = (wn1 + wn2 - 0.75) * 2.0;
        let angle_warped = angle + warp_n * warp_amt;

        let fiber_slope = -sin(angle_warped * line_freq) * line_freq;
        // Tangent sweeps around local Z, transformed to world space.
        // Safe: when myo_surf is near the Z poles, xy_len -> 0; clamp to avoid NaN.
        let xy_len = length(myo_surf.xy);
        let tangent_local = select(
            normalize(vec3<f32>(-myo_surf.y, myo_surf.x, 0.0)),
            vec3<f32>(1.0, 0.0, 0.0),  // fallback at poles
            xy_len < 0.001
        );
        let tangent_world = quat_rotate(in.rotation, tangent_local);
        perturbed_normal = normalize(perturbed_normal + tangent_world * fiber_slope * bulge_str * 0.10);
    }

    // ====================================================================
    // Cognocyte (type 14): Angular circuit-board surface facets (LOD >= 1)
    // ====================================================================
    if (cell_type == 14u) {
        let surf_local = normalize(quat_rotate_inverse(in.rotation,
            in.cam_right * front_pos.x + in.cam_up * front_pos.y + in.to_camera * front_pos.z));

        let lat = acos(clamp(surf_local.z, -1.0, 1.0));
        let lon = atan2(surf_local.y, surf_local.x);

        let lat_bands = 6.0;
        let lon_segs  = 8.0;
        let lat_cell  = floor(lat / 3.14159265 * lat_bands);
        let lon_cell  = floor((lon + 3.14159265) / 6.28318530 * lon_segs);

        let lat_frac = fract(lat / 3.14159265 * lat_bands);
        let lon_frac = fract((lon + 3.14159265) / 6.28318530 * lon_segs);

        let edge_w    = 0.065;
        let lat_edge  = smoothstep(edge_w, 0.0, lat_frac) + smoothstep(1.0 - edge_w, 1.0, lat_frac);
        let lon_edge  = smoothstep(edge_w, 0.0, lon_frac) + smoothstep(1.0 - edge_w, 1.0, lon_frac);
        let edge_mask = clamp(lat_edge + lon_edge, 0.0, 1.0);

        let facet_id  = lat_cell * lon_segs + lon_cell;
        let facet_h   = fract(facet_id * 0.61803398 + 0.41421356);
        let facet_dim = 0.82 + 0.18 * facet_h;

        let t_pulse  = camera.time * 3.8;
        let ring_lat = fract(t_pulse / 6.28318530) * 3.14159265;
        let ring_mask = smoothstep(0.18, 0.0, abs(lat - ring_lat)) * 0.45;

        let edge_color  = base_color * 1.55;
        let facet_color = interior_result * facet_dim;
        let ring_color  = base_color * 1.8;

        var cogno_surf = mix(facet_color, edge_color, edge_mask * 0.70);
        cogno_surf     = mix(cogno_surf,  ring_color,  ring_mask);
        interior_result = cogno_surf;
    }

    // Composite membrane over interior
    var composited = mix(interior_result, membrane_color, membrane_alpha);

    // ====================================================================
    // Membrane surface lighting (specular + fresnel on front shell only)
    // ====================================================================
    // Sample shadow from light field, offset toward light to avoid self-occlusion
    // lighting.light_dir points FROM light, so negate to go TOWARD light
    let offset_distance = mix(3.0, 6.0, shadow_params.shadow_quality) * shadow_params.cell_size;
    let shadow_sample_pos = in.center - normalize(lighting.light_dir) * offset_distance;
    // Clamp sample position to stay within valid light field grid bounds
    let grid_size = shadow_params.cell_size * f32(shadow_params.grid_resolution);
    let grid_min = vec3<f32>(shadow_params.grid_origin_x, shadow_params.grid_origin_y, shadow_params.grid_origin_z);
    let grid_max = grid_min + vec3<f32>(grid_size, grid_size, grid_size);
    let clamped_pos = clamp(shadow_sample_pos, grid_min, grid_max);
    let shadow = mix(1.0, sample_light_field(clamped_pos), shadow_params.shadow_strength);
    let local_light_color = sample_light_color_field(clamped_pos);
    let front_ndotl = max(dot(perturbed_normal, -light_dir), 0.0);
    let front_diffuse = front_ndotl * local_light_color * shadow;

    // Apply diffuse to the composited result.
    // ambient doubles as the shadow floor so both scale together with sun intensity.
    let unlit_composited = composited;
    composited = composited * (lighting.ambient + (1.0 - lighting.ambient) * front_diffuse);

    // Specular highlight on membrane surface (uses perturbed normal for ridge highlights)
    let half_vec = normalize(-light_dir + view_dir);
    let spec = pow(max(dot(perturbed_normal, half_vec), 0.0), in.visual_params.y);
    let specular = spec * in.visual_params.x * local_light_color * shadow;

    // Fresnel rim (membrane reflection)
    let fresnel = pow(1.0 - max(dot(perturbed_normal, view_dir), 0.0), 3.0);
    let fresnel_contribution = fresnel * in.visual_params.z;

    // Subsurface scattering (light bleeding through from behind)
    let sss = max(dot(world_normal_front, light_dir), 0.0) * 0.12;
    let sss_color = base_color * sss;

    var final_color = composited + specular + fresnel_contribution + sss_color
                    + unlit_composited * in.visual_params.w; // emissive: uses pre-lighting color so cells glow in darkness

    // Cel-shaded outline: dark band at the silhouette edge.
    // Emissive cells use a dark-tinted version of their own colour instead of pure black,
    // so bright luminocytes don't get an ugly hard black ring.
    if (lighting.outline_width > 0.0) {
        let outline_inner = 1.0 - lighting.outline_width;
        let aa = fwidth(r) * 1.5;
        let outline = smoothstep(outline_inner - aa, outline_inner + aa, r);
        // Fade outline toward the cell's own dark colour on emissive cells.
        let emissive = in.visual_params.w;
        let outline_color = mix(
            vec3<f32>(0.0),                     // black for non-emissive
            unlit_composited * 0.08,            // dark cell tint for emissive
            clamp(emissive * 0.4, 0.0, 1.0),
        );
        final_color = mix(final_color, outline_color, outline);
    }

    // Yellow outline for selected-mode cells (type_data_1.z == 1.0)
    let highlight_flag = in.type_data_1.z;
    if (highlight_flag > 0.5) {
        let yellow_width = max(lighting.outline_width, 0.08);
        let yellow_inner = 1.0 - yellow_width;
        let aa2 = fwidth(r) * 1.5;
        let yellow_outline = smoothstep(yellow_inner - aa2, yellow_inner + aa2, r);
        final_color = mix(final_color, vec3<f32>(1.0, 1.0, 0.0), yellow_outline);
    }

    // ====================================================================
    // Devorocyte (type 11): Cone spike rendering
    // type_data_0: x=spike_height (0.3-1.5), y=spike_sharpness (0.95-0.999),
    //              z=spike_embed (0.05-0.25), w=tip_fade (0.0-1.0)
    // ====================================================================
    if (cell_type == 11u) {
        let spike_height  = clamp(in.type_data_0.x, 0.3, 1.5);
        let cone_half_cos = clamp(in.type_data_0.y, 0.95, 0.999);
        let spike_embed   = clamp(in.type_data_0.z, 0.05, 0.25);
        let tip_fade_str  = clamp(in.type_data_0.w, 0.0, 1.0);

        let pixel_world   = in.center
                          + in.cam_right * local_pos.x * in.radius
                          + in.cam_up    * local_pos.y * in.radius;
        let ray_dir_world = normalize(pixel_world - camera.camera_pos);

        let cam_local_unscaled = quat_rotate_inverse(in.rotation, camera.camera_pos - in.center);
        let ray_o = cam_local_unscaled / in.radius;
        let ray_d = normalize(quat_rotate_inverse(in.rotation, ray_dir_world));

        // t at which the ray hits the sphere front surface in cell-local space.
        // Solve |ray_o + t*ray_d|^2 = 1. Take the larger positive root (front face).
        let sph_b    = dot(ray_o, ray_d);
        let sph_c    = dot(ray_o, ray_o) - 1.0;
        let sph_disc = sph_b * sph_b - sph_c;
        var t_sphere_front = 1e9;
        if (sph_disc >= 0.0) {
            let t_sf = -sph_b + sqrt(sph_disc);
            if (t_sf > 0.001) { t_sphere_front = t_sf; }
        }

        let spike_dirs = array<vec3<f32>, 20>(
            normalize(vec3<f32>( 0.000,  1.000,  0.000)),
            normalize(vec3<f32>( 0.894,  0.800, -0.000)),
            normalize(vec3<f32>(-0.588,  0.600,  0.809)),
            normalize(vec3<f32>(-0.588,  0.400, -0.809)),
            normalize(vec3<f32>( 0.951,  0.200,  0.309)),
            normalize(vec3<f32>( 0.000,  0.000, -1.000)),
            normalize(vec3<f32>(-0.951,  0.200,  0.309)),
            normalize(vec3<f32>( 0.588, -0.200,  0.809)),
            normalize(vec3<f32>( 0.588, -0.200, -0.809)),
            normalize(vec3<f32>(-0.951, -0.200, -0.309)),
            normalize(vec3<f32>( 0.000, -0.200,  1.000)),
            normalize(vec3<f32>( 0.951, -0.400, -0.309)),
            normalize(vec3<f32>(-0.588, -0.400,  0.809)),
            normalize(vec3<f32>(-0.000, -0.600, -1.000)),
            normalize(vec3<f32>( 0.588, -0.600,  0.809)),
            normalize(vec3<f32>( 0.951, -0.600, -0.309)),
            normalize(vec3<f32>(-0.951, -0.600,  0.309)),
            normalize(vec3<f32>( 0.000, -0.800,  0.000)),
            normalize(vec3<f32>( 0.588, -0.800, -0.809)),
            normalize(vec3<f32>(-0.588, -1.000,  0.000))
        );

        let spike_total    = spike_height + spike_embed;
        let cone_half_cos2 = cone_half_cos * cone_half_cos;

        var best_t    = 1e9;
        var best_norm = vec3<f32>(0.0, 1.0, 0.0);
        var hit_spike = false;

        for (var si = 0u; si < 20u; si++) {
            let spike_dir = spike_dirs[si];
            let apex      = spike_dir * (1.0 + spike_height);
            let cone_axis = -spike_dir;

            let oc = ray_o - apex;
            let dv = dot(ray_d, cone_axis);
            let ov = dot(oc, cone_axis);

            let a    = dv * dv - cone_half_cos2;
            let b    = 2.0 * (dv * ov - cone_half_cos2 * dot(oc, ray_d));
            let c    = ov * ov - cone_half_cos2 * dot(oc, oc);
            let disc = b * b - 4.0 * a * c;
            if (disc < 0.0 || abs(a) < 1e-6) { continue; }

            let sq = sqrt(disc);
            let t0 = (-b - sq) / (2.0 * a);
            let t1 = (-b + sq) / (2.0 * a);

            var cone_t = -1.0;
            var cone_n = vec3<f32>(0.0);

            if (t0 > 0.001 && t0 < best_t) {
                let hp    = ray_o + ray_d * t0;
                let along = dot(hp - apex, cone_axis);
                if (along >= 0.0 && along <= spike_total) {
                    let n = normalize((hp - apex) - cone_axis * (along / cone_half_cos2));
                    if (dot(n, -ray_d) >= 0.0) { cone_t = t0; cone_n = n; }
                }
            }
            if (cone_t < 0.0 && t1 > 0.001 && t1 < best_t) {
                let hp    = ray_o + ray_d * t1;
                let along = dot(hp - apex, cone_axis);
                if (along >= 0.0 && along <= spike_total) {
                    let n = normalize((hp - apex) - cone_axis * (along / cone_half_cos2));
                    if (dot(n, -ray_d) >= 0.0) { cone_t = t1; cone_n = n; }
                }
            }

            if (cone_t > 0.0) { best_t = cone_t; best_norm = cone_n; hit_spike = true; }
        }

        // Spike wins if it hit AND is closer to the camera than the sphere front surface.
        if (hit_spike && best_t < t_sphere_front) {
            let spike_world_normal = normalize(quat_rotate(in.rotation, best_norm));
            let hit_local = ray_o + ray_d * best_t;
            let hit_world = in.center + quat_rotate(in.rotation, hit_local) * in.radius;

            let spike_ndotl = max(dot(spike_world_normal, -light_dir), 0.0);
            let spike_half  = normalize(-light_dir + view_dir);
            let spike_spec  = pow(max(dot(spike_world_normal, spike_half), 0.0), 40.0) * 0.8;

            let tip_fade    = clamp((length(hit_local) - 1.0) / spike_height, 0.0, 1.0);
            let spike_color = base_color * (0.8 - tip_fade * tip_fade_str * 0.5)
                            * (0.2 + 0.8 * spike_ndotl)
                            + vec3<f32>(spike_spec);

            let spike_clip = camera.view_proj * vec4<f32>(hit_world, 1.0);
            out.depth = spike_clip.z / spike_clip.w;
            out.color = vec4<f32>(spike_color, in.color.a);
            return out;
        }

        // No spike closer than the sphere - discard if we're in the expanded region.
        if (r2 > 1.0) { discard; }
        // Otherwise fall through to normal sphere shading.
    }


    let sphere_offset = z_front * in.radius;
    let sphere_world_pos = in.center + in.to_camera * sphere_offset;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_world_pos, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;

    var output_color: vec3<f32>;
    if (debug_colors > 0.5 && highlight_flag < 0.5) {
        var debug_color: vec3<f32>;
        switch (lod) {
            case 0u: { debug_color = vec3<f32>(1.0, 0.2, 0.2); }
            case 1u: { debug_color = vec3<f32>(0.2, 1.0, 0.2); }
            case 2u: { debug_color = vec3<f32>(0.2, 0.2, 1.0); }
            case 3u: { debug_color = vec3<f32>(1.0, 1.0, 0.2); }
            default: { debug_color = vec3<f32>(1.0, 0.2, 1.0); }
        }
        output_color = mix(final_color, debug_color, 0.35);
    } else {
        output_color = final_color;
    }

    out.color = vec4<f32>(output_color, in.color.a * edge_alpha);
    return out;
}

// -- Depth-only prepass entry point -------------------------------------------
// Runs the minimum work needed to write correct sphere depth:
//   - discard pixels outside the circle
//   - project the sphere front face to clip space and write its depth
// No lighting, no internals, no color output.
// The color pass that follows uses LessEqual + no depth write to eliminate
// all overdraw - every hidden fragment is rejected before its shader runs.
struct DepthOnlyOutput {
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_depth(in: VertexOutput) -> DepthOnlyOutput {
    let local_pos = in.uv * 2.0 - 1.0;
    let r2 = dot(local_pos, local_pos);

    if (r2 > 1.0) {
        // Always discard outside the sphere, including devorocytes.
        // Spike depth for the expanded region is written by fs_main, not the prepass.
        // Without this, the prepass would write cell-center depth (z_front=0) for the
        // entire expanded quad, incorrectly occluding everything behind the cell in the
        // shared depth buffer.
        discard;
    }

    let z_front = sqrt(max(1.0 - r2, 0.0));
    let sphere_world_pos = in.center + in.to_camera * (z_front * in.radius);
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_world_pos, 1.0);

    var out: DepthOnlyOutput;
    out.depth = sphere_clip.z / sphere_clip.w;
    return out;
}
