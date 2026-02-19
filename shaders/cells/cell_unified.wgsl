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
}

@group(1) @binding(0) var<uniform> shadow_params: ShadowFieldParams;
@group(1) @binding(1) var<storage, read> light_field: array<f32>;

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

    let world_size = instance.radius * 1.1;
    let world_pos = instance.position + right * quad_pos.x * world_size + up * quad_pos.y * world_size;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = quad_pos * 0.5 + 0.5;
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
// produces 12 pentagons and 20 hexagons — a classic soccer ball / icosphere.
//
// type_data_0 per-instance params:
//   .x = subdivision    (1 or 2, default 2) — 1 = 12 pentagons, 2 = 12 pent + 20 hex
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
    let theta = atan2(dir.z, dir.x); // -π..π
    let phi = acos(clamp(dir.y, -1.0, 1.0)); // 0..π
    let u = (theta / (2.0 * PI)) + 0.5; // 0..1
    let v = phi / PI; // 0..1
    return vec2<f32>(u, v);
}

// Sample the pre-baked hex pattern texture. Applies meander distortion before lookup.
// Returns HexTripletResult from a single texture sample — no Voronoi math at runtime.
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

// Gaussian ridge: peaks at edge (edge_dist≈0), zero at face centers.
fn geodesic_ridge(edge_dist: f32, ridge_width: f32) -> f32 {
    return exp(-(edge_dist * edge_dist) / (ridge_width * ridge_width));
}

// ============================================================================
// Per-Type Internal Pattern Functions
// ============================================================================
// Each returns a vec3: (pattern_value, color_shift, unused)
// pattern_value: 0 = pure cytoplasm, 1 = full organelle coverage
// color_shift: how much to shift the color (positive = lighter, negative = darker)

// Type 0: Test Cell — No special effects. Plain sphere.
fn internals_test(p: vec3<f32>, r: f32) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Type 1: Flagellocyte — Same as test cell (tail is rendered separately).
fn internals_flagellocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Type 2: Phagocyte — Simple cell with a visible nucleus.
fn internals_phagocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    let nucleus_r = length(p);
    let nucleus = smoothstep(0.3, 0.2, nucleus_r);

    let pattern = nucleus * 0.6;
    let color_shift = nucleus * -0.4; // nucleus is darker

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 3: Photocyte — internals handled as inner sphere in fragment shader.
fn internals_photocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Type 4: Lipocyte — Oily looking internals.
// Large blobby fat droplets with bright highlights and dark boundaries.
fn internals_lipocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    // Layered smooth noise for oily, blobby look
    let n1 = value_noise_3d(p * 3.0 + 50.0);
    let n2 = value_noise_3d(p * 5.0 + vec3<f32>(n1 * 2.0, 0.0, 0.0) + 70.0);
    let oily = n1 * 0.6 + n2 * 0.4;

    // Fat droplets: bright rounded blobs
    let droplet = smoothstep(0.35, 0.6, oily);
    // Dark boundaries between droplets
    let boundary = 1.0 - smoothstep(0.0, 0.15, abs(oily - 0.45));

    let pattern = max(droplet * 0.85, boundary * 0.7);
    // Droplets are brighter (yellowish sheen), boundaries are darker
    let color_shift = droplet * 0.5 - boundary * 0.35;

    return vec3<f32>(pattern, color_shift, 0.0);
}

// Type 5: Buoyocyte — Hollow interior with gas bladder organelle.
// Features a thick-walled hollow cell with a biconvex (contact lens-shaped)
// gas organelle that provides buoyancy. The organelle appears as a translucent
// bubble with highlights and refraction effects.
fn internals_buoyocyte(p: vec3<f32>, r: f32) -> vec3<f32> {
    // Hollow cell: empty center region
    let hollow_radius = 0.25;  // Inner hollow cavity
    let hollow = smoothstep(hollow_radius - 0.05, hollow_radius + 0.05, r);
    
    // Contact lens-shaped gas organelle (biconvex lens)
    // The organelle sits in the upper hemisphere and has a distinctive lens shape
    let organelle_center = vec3<f32>(0.0, 0.15, 0.0);  // Positioned in upper half
    
    // Transform to organelle-local space
    let op = p - organelle_center;
    let or = length(op);
    
    // Biconvex lens shape: thicker in center, thinner at edges
    // Using two spherical surfaces to create the lens profile
    let lens_radius = 0.35;  // Overall lens radius
    let lens_thickness = 0.12;  // Maximum thickness at center
    
    // Distance from lens center along Y axis determines thickness
    let y_dist = abs(op.y);
    let xz_dist = sqrt(op.x * op.x + op.z * op.z);
    
    // Lens equation: thicker at center, curved edges
    let lens_profile = lens_thickness * (1.0 - (y_dist / lens_radius) * (y_dist / lens_radius));
    let lens_edge = smoothstep(lens_radius - 0.05, lens_radius + 0.05, xz_dist);
    
    // Combine radial and vertical constraints for lens shape
    let lens_shape = (1.0 - lens_edge) * (1.0 - smoothstep(0.0, lens_thickness, lens_profile));
    
    // Gas bubble effect: translucent with bright highlights
    let gas_bubble = lens_shape;
    
    // Add refraction-like highlights on the gas surface
    let highlight_angle = dot(normalize(op), normalize(vec3<f32>(0.3, 0.7, 0.2)));
    let highlight = pow(max(0.0, highlight_angle), 8.0) * gas_bubble;
    
    // Combine effects
    // 1. Hollow region (empty/dark)
    // 2. Gas organelle (bright, translucent)
    // 3. Cytoplasm (medium brightness)
    let cytoplasm = 1.0 - hollow - gas_bubble * 0.7;
    let pattern = max(cytoplasm * 0.3, gas_bubble * 0.8 + highlight);
    
    // Color shifts: gas appears lighter/brighter, hollow appears darker
    let color_shift = gas_bubble * 0.6 - hollow * 0.4 + highlight * 0.3;
    
    return vec3<f32>(pattern, color_shift, 0.0);
}

// ============================================================================
// Pattern Dispatcher
// ============================================================================

fn get_internals(cell_type: u32, p: vec3<f32>, r: f32) -> vec3<f32> {
    switch (cell_type) {
        case 0u: { return internals_test(p, r); }
        case 1u: { return internals_flagellocyte(p, r); }
        case 2u: { return internals_phagocyte(p, r); }
        case 3u: { return internals_photocyte(p, r); }
        case 4u: { return internals_lipocyte(p, r); }
        case 5u: { return internals_buoyocyte(p, r); }
        default: { return internals_test(p, r); }
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
        case 0u: { // Test — standard membrane
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 1u: { // Flagellocyte — same as test
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 2u: { // Phagocyte — standard membrane
            m.thickness = 0.06;
            m.opacity = 0.5;
            m.rim_power = 2.5;
            m.color_darken = 0.25;
        }
        case 3u: { // Photocyte — translucent hex membrane over solar panel internals
            m.thickness = 0.05;
            m.opacity = 0.35;
            m.rim_power = 2.0;
            m.color_darken = 0.2;
        }
        case 4u: { // Lipocyte — translucent to show oily internals
            m.thickness = 0.04;
            m.opacity = 0.3;
            m.rim_power = 2.0;
            m.color_darken = 0.15;
        }
        case 5u: { // Buoyocyte — thick membrane for gas bladder
            m.thickness = 0.08;
            m.opacity = 0.4;
            m.rim_power = 3.0;
            m.color_darken = 0.3;
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
//   1. Back wall  (z = -sqrt(1-r²)) → opaque cytoplasm background
//   2. Midpoint   (z = 0 plane)     → organelle pattern sampled at interior 3D pos
//   3. Front wall (z = +sqrt(1-r²)) → semi-transparent membrane + specular/fresnel
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

    // Billboard-space position [-1, 1]
    let local_pos = in.uv * 2.0 - 1.0;
    let r2 = dot(local_pos, local_pos);

    if (r2 > 1.0) {
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
    // mid_pos is (x, y, 0) in billboard space — a flat cross-section through the sphere.
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

    if (lod >= 1u) {
        let internals = get_internals(cell_type, interior_pos, r);
        let pattern_value = internals.x;
        let color_shift = internals.y;

        // Organelle color: shifted version of base color
        let organelle_color = base_color * (0.75 + color_shift);

        // Composite organelles over base cytoplasm
        interior_result = mix(interior_result, organelle_color, pattern_value);
    }

    // ====================================================================
    // Photocyte (type 3): Inner hex-patterned nucleus sphere (front + back)
    // ====================================================================
    if (cell_type == 3u && lod >= 1u) {
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
    let front_ndotl = max(dot(perturbed_normal, -light_dir), 0.0);
    let front_diffuse = front_ndotl * lighting.light_color * shadow;

    // Apply diffuse to the composited result
    composited = composited * (0.3 + 0.7 * (lighting.ambient + front_diffuse));

    // Specular highlight on membrane surface (uses perturbed normal for ridge highlights)
    let half_vec = normalize(-light_dir + view_dir);
    let spec = pow(max(dot(perturbed_normal, half_vec), 0.0), in.visual_params.y);
    let specular = spec * in.visual_params.x * lighting.light_color * shadow;

    // Fresnel rim (membrane reflection)
    let fresnel = pow(1.0 - max(dot(perturbed_normal, view_dir), 0.0), 3.0);
    let fresnel_contribution = fresnel * in.visual_params.z;

    // Subsurface scattering (light bleeding through from behind)
    let sss = max(dot(world_normal_front, light_dir), 0.0) * 0.12;
    let sss_color = base_color * sss;

    var final_color = composited + specular + fresnel_contribution + sss_color
                    + composited * in.visual_params.w; // emissive

    // Cel-shaded black outline: hard black band at the silhouette edge
    if (lighting.outline_width > 0.0) {
        let aa = fwidth(z_front);
        let outline = smoothstep(lighting.outline_width - aa, lighting.outline_width + aa, z_front);
        final_color = mix(vec3<f32>(0.0, 0.0, 0.0), final_color, outline);
    }

    // ====================================================================
    // Sphere depth (front surface)
    // ====================================================================
    let sphere_offset = z_front * in.radius;
    let sphere_world_pos = in.center + in.to_camera * sphere_offset;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_world_pos, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;

    // ====================================================================
    // Debug LOD colors
    // ====================================================================
    var output_color: vec3<f32>;
    if (debug_colors > 0.5) {
        switch (lod) {
            case 0u: { output_color = vec3<f32>(1.0, 0.2, 0.2); }
            case 1u: { output_color = vec3<f32>(0.2, 1.0, 0.2); }
            case 2u: { output_color = vec3<f32>(0.2, 0.2, 1.0); }
            case 3u: { output_color = vec3<f32>(1.0, 1.0, 0.2); }
            default: { output_color = vec3<f32>(1.0, 0.2, 1.0); }
        }
    } else {
        output_color = final_color;
    }

    out.color = vec4<f32>(output_color, in.color.a * edge_alpha);
    return out;
}
