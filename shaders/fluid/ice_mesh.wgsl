// Ice mesh rendering shader
// Renders the ice-only surface-nets isosurface as a rigid, deeply faceted,
// semitransparent crystalline volume inside the shared WBOIT pass.
//
// Group LAYOUTS intentionally match density_mesh.wgsl (uniform+uniform,
// shadow field, cubemap) so the surface nets renderer reuses its layouts -
// but ice binds its own IceRenderParams buffer at group(0) binding(1).

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

// Ice appearance, driven by the Fluid Settings UI (see IceAppearanceSettings).
struct IceRenderParams {
    surface_color: vec3<f32>,
    // Facets per world unit (lower = larger crystal faces).
    facet_scale: f32,
    deep_color: vec3<f32>,
    // How far each crystal facet plane tilts off the smooth surface, and how
    // far vertices are displaced to land on it (0 = no displacement, smooth).
    displacement_strength: f32,
    // Fraction of the flat facet normal blended into diffuse shading (0 =
    // smooth shading only, 1 = full per-face diffuse patches).
    facet_diffuse: f32,
    // Blinn-Phong exponent for the per-face glint.
    glint_shininess: f32,
    // Glint intensity multiplier.
    glint_strength: f32,
    // Base opacity (fresnel and glancing facets add on top).
    alpha_base: f32,
    // Environment reflection brightness multiplier.
    reflection_brightness: f32,
    // How much fresnel-weighted env reflection mixes in at grazing angles.
    fresnel_reflection: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> params: IceRenderParams;

// Shadow field data (from light field system) - must match density_mesh.wgsl
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

// Environment cubemap for reflections
@group(2) @binding(0) var env_cubemap: texture_cube<f32>;
@group(2) @binding(1) var env_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) fluid_type: f32,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_dir: vec3<f32>,
    // Geometric normal of this vertex's facet plane, computed in the vertex
    // shader. Screen-space derivatives (dpdx/dpdy) degenerate for
    // near-horizontal facets viewed from directly above (the tiny vertical
    // displacement gets swamped by the much larger x/z screen derivatives,
    // so the cross product collapses back to the smooth normal) - a
    // top-down view of a flat ice sheet showed no faceting at all. Carrying
    // the true per-facet plane normal from the vertex shader is
    // view-independent and fixes that.
    //
    // Smoothly interpolated (not flat): each vertex's facet_normal can
    // belong to a different Voronoi cell, so flat interpolation (constant
    // per triangle, picked from one provoking vertex) drew a hard-edged grid
    // of differently-shaded triangles - "a bunch of squares". Interpolating
    // and renormalizing blends between neighboring cells' facet normals
    // across each triangle, turning those hard edges into soft gradients.
    @location(3) facet_normal: vec3<f32>,
}

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

// Crystal facet field: world space is partitioned into irregular Voronoi
// regions ("crystal faces"). Each region gets its own flat plane, and
// vertices are displaced (along the smooth surface normal) onto that plane -
// real faceted geometry, not a fragment-shader normal trick. Facet size and
// depth come from IceRenderParams (Fluid Settings UI).

fn hash3_vec(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453123);
}

fn cell_order_key(cell: vec3<f32>) -> f32 {
    return dot(cell, vec3<f32>(1.0, 4096.0, 16777216.0));
}

// Returns the integer cell coordinates of the two nearest Voronoi crystal
// feature points to `p` (already scaled by facet_scale), along with the
// (non-squared) distances to each. `cell_a`/`da` is the nearest, `cell_b`/`db`
// the second-nearest. Used by facet_displace to blend smoothly between two
// adjacent crystal faces near their shared boundary instead of snapping
// discontinuously from one face's plane to the other.
struct VoronoiPair {
    cell_a: vec3<f32>,
    cell_b: vec3<f32>,
    da: f32,
    db: f32,
}

fn voronoi_pair(p: vec3<f32>) -> VoronoiPair {
    let base = floor(p);
    var best_d = 1e9;
    var second_d = 1e9;
    var best_cell = base;
    var second_cell = base;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            for (var z = -1; z <= 1; z++) {
                let cell = base + vec3<f32>(f32(x), f32(y), f32(z));
                let feature = cell + hash3_vec(cell);
                let diff = p - feature;
                let d = length(diff);
                if (d < best_d) {
                    second_d = best_d;
                    second_cell = best_cell;
                    best_d = d;
                    best_cell = cell;
                } else if (d < second_d) {
                    second_d = d;
                    second_cell = cell;
                }
            }
        }
    }
    return VoronoiPair(best_cell, second_cell, best_d, second_d);
}

// Builds the random per-cell facet normal: a fixed lean (1.0x smooth_normal)
// plus a random tilt confined to the plane PERPENDICULAR to smooth_normal.
// This guarantees dot(smooth_normal, facet_normal) >= ~0.5 (max ~60 degrees
// off smooth_normal) for every random tilt and every surface orientation, so
// the plane is never near-degenerate relative to the displacement direction
// (see facet_displace for the history behind this formula).
fn facet_plane_normal(cell: vec3<f32>, smooth_normal: vec3<f32>) -> vec3<f32> {
    let tilt = hash3_vec(cell + vec3<f32>(17.31, 47.7, 93.1)) * 2.0 - 1.0;
    let tilt_perp = tilt - smooth_normal * dot(tilt, smooth_normal);
    return normalize(smooth_normal * 1.0 + tilt_perp);
}

// Projects `pos` onto the flat facet plane of its Voronoi cell, moving along
// the smooth surface normal, then blends toward that projected position by
// displacement_strength. The plane passes through the cell's feature point
// with a fixed per-cell random normal (independent of displacement_strength)
// - so every vertex in the same crystal face shares the same plane, giving a
// genuinely flat facet. displacement_strength is a linear blend factor: 0 =
// untouched smooth surface, 1 = vertices land exactly on the facet plane,
// >1 = pushed past it for deeper, sharper facets. Keeping the plane itself
// independent of displacement_strength means small slider changes produce
// proportionally small displacement changes, instead of snapping straight
// to the fully-projected position the moment displacement_strength > 0.
struct FacetResult {
    pos: vec3<f32>,
    normal: vec3<f32>,
}

struct OrderedVoronoiPair {
    cell_a: vec3<f32>,
    cell_b: vec3<f32>,
    dist_a: f32,
    dist_b: f32,
}

fn ordered_voronoi_pair(p: vec3<f32>) -> OrderedVoronoiPair {
    let pair = voronoi_pair(p);
    var cell_a = pair.cell_a;
    var cell_b = pair.cell_b;
    var dist_a = pair.da;
    var dist_b = pair.db;
    if (cell_order_key(cell_a) > cell_order_key(cell_b)) {
        let old_cell_a = cell_a;
        let old_dist_a = dist_a;
        cell_a = cell_b;
        dist_a = dist_b;
        cell_b = old_cell_a;
        dist_b = old_dist_a;
    }
    return OrderedVoronoiPair(cell_a, cell_b, dist_a, dist_b);
}

fn facet_blend_weight(dist_a: f32, dist_b: f32) -> f32 {
    const BLEND_WIDTH: f32 = 0.075;
    var w = smoothstep(-BLEND_WIDTH, BLEND_WIDTH, dist_a - dist_b);
    return w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
}

fn facet_shading_normal(pos: vec3<f32>, smooth_normal: vec3<f32>) -> vec3<f32> {
    let scale = max(params.facet_scale, 0.001);
    let pair = ordered_voronoi_pair(pos * scale);
    let normal_a = facet_plane_normal(pair.cell_a, smooth_normal);
    let normal_b = facet_plane_normal(pair.cell_b, smooth_normal);
    let w = facet_blend_weight(pair.dist_a, pair.dist_b);
    return normalize(mix(normal_a, normal_b, w));
}

fn facet_displace(pos: vec3<f32>, smooth_normal: vec3<f32>) -> FacetResult {
    if params.displacement_strength <= 0.0 {
        return FacetResult(pos, smooth_normal);
    }

    let scale = max(params.facet_scale, 0.001);
    let pair = ordered_voronoi_pair(pos * scale);
    let max_disp = 1.0 / scale;

    // Each vertex picks its own nearest/second-nearest cell pair. Sort that
    // pair into a stable order before blending; otherwise the two sides of a
    // boundary both call their own nearest cell "A", so the interpolation
    // restarts at zero immediately after the boundary and creates a jagged
    // snap between facets.
    //
    // The transition band has to be wider than a single surface-nets triangle
    // in scaled facet space. When it is too narrow, the Voronoi boundary is
    // only sampled at mesh vertices and turns into a saw-toothed crack between
    // otherwise broad crystal faces.
    let normal_a = facet_plane_normal(pair.cell_a, smooth_normal);
    let normal_b = facet_plane_normal(pair.cell_b, smooth_normal);
    let feature_a = (pair.cell_a + hash3_vec(pair.cell_a)) / scale;
    let feature_b = (pair.cell_b + hash3_vec(pair.cell_b)) / scale;

    let t_a = clamp(dot(feature_a - pos, normal_a) / dot(smooth_normal, normal_a), -max_disp, max_disp);
    let t_b = clamp(dot(feature_b - pos, normal_b) / dot(smooth_normal, normal_b), -max_disp, max_disp);

    let w = facet_blend_weight(pair.dist_a, pair.dist_b);
    let t = mix(t_a, t_b, w) * params.displacement_strength;
    let facet_normal = normalize(mix(normal_a, normal_b, w));
    return FacetResult(pos + smooth_normal * t, facet_normal);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let smooth_normal = normalize(in.normal);
    let facet = facet_displace(in.position, smooth_normal);

    out.world_position = facet.pos;
    out.world_normal = smooth_normal;
    out.facet_normal = facet.normal;
    out.view_dir = normalize(camera.camera_pos - facet.pos);
    out.clip_position = camera.view_proj * vec4<f32>(facet.pos, 1.0);

    return out;
}

struct ShadeResult {
    color: vec3<f32>,
    fresnel: f32,
    facing: f32,
}

// Deeply faceted: vertices were displaced onto flat per-Voronoi-cell planes
// in the vertex shader, so each triangle is genuinely planar. Uses the
// per-facet plane normal carried (flat) from the vertex shader - this drives
// SPECULAR almost exclusively; body color and base diffuse come from the
// smooth normal, so facets read as glassy faces catching the light
// differently, not as patches of different paint (full facet diffuse looked
// like a patchwork quilt). Screen-space derivatives (dpdx/dpdy) were used
// previously, but degenerate to the smooth normal for near-horizontal
// facets viewed from directly above.
fn shade_ice(in: VertexOutput) -> ShadeResult {
    let smooth_normal = normalize(in.world_normal);
    let view_dir = normalize(in.view_dir);
    let facet_normal = facet_shading_normal(in.world_position, smooth_normal);

    let light_dir = normalize(vec3<f32>(shadow_params.light_dir_x, shadow_params.light_dir_y, shadow_params.light_dir_z));

    // Shadow field sample, offset along the normal to avoid self-shadowing.
    let shadow_sample_pos = in.world_position + smooth_normal * shadow_params.cell_size * 2.0;
    let light_value = sample_light_field(shadow_sample_pos);
    let shadow = mix(1.0, light_value, shadow_params.shadow_strength);
    let sun_color = vec3<f32>(shadow_params.sun_color_r, shadow_params.sun_color_g, shadow_params.sun_color_b);

    // Body color: continuous across the surface (smooth normal) - pale
    // facing the viewer, deep interior blue at glancing angles.
    let facing = max(dot(smooth_normal, view_dir), 0.0);
    let body_color = mix(params.deep_color, params.surface_color, facing);

    // Lambert diffuse with a generous ambient floor - ice scatters light
    // internally, so it never goes fully black in shadow. Mostly smooth,
    // with a subtle (20%) facet contribution so faces still shade apart.
    let n_dot_l_smooth = max(dot(smooth_normal, light_dir), 0.0);
    let n_dot_l_facet = max(dot(facet_normal, light_dir), 0.0);
    let n_dot_l = mix(n_dot_l_smooth, n_dot_l_facet, params.facet_diffuse);
    let lighting = 0.35 + n_dot_l * 0.75 * shadow;
    var final_color = body_color * sun_color * lighting;

    // Sharp Blinn-Phong glint off the FACET normal - this is where the
    // crystal faces live: each face flashes at its own view/light angles.
    // Per-facet brightness variance from a hash of the flat normal direction
    // (constant across the facet, since the normal is constant per facet).
    let half_dir = normalize(light_dir + view_dir);
    let glint = pow(max(dot(facet_normal, half_dir), 0.0), max(params.glint_shininess, 1.0));
    let facet_variance = fract(sin(dot(facet_normal, vec3<f32>(12.9898, 78.233, 45.164))) * 43758.5453);
    let glint_strength = (0.5 + 0.5 * facet_variance) * params.glint_strength;
    final_color += sun_color * glint * glint_strength * shadow;

    // Fresnel-weighted environment reflection off the smooth geometry normal.
    let fresnel = pow(1.0 - max(dot(smooth_normal, view_dir), 0.0), 3.0);
    let reflect_dir = reflect(-view_dir, smooth_normal);
    let env_color = textureSample(env_cubemap, env_sampler, reflect_dir).rgb;
    final_color = mix(
        final_color,
        env_color * params.reflection_brightness,
        fresnel * params.fresnel_reflection,
    );

    return ShadeResult(final_color, fresnel, facing);
}

// Renders straight into the scene's color+depth targets with depth write
// enabled and ordinary alpha blending - a rigid solid, not an OIT fluid. No
// WBOIT: ice is a single front-facing layer per pixel (back faces culled),
// so there's no order-independence to gain, and WBOIT's weighted-average
// compositing made near-opaque ice look "ice-tinted" by whatever was behind
// it (and overflowed Rgba16Float at high alpha with HDR glints/reflections).
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let shaded = shade_ice(in);

    // Mostly opaque with just a hint of translucency: ice is a dense solid -
    // a low alpha here reads as ghostly rather than icy. Grazing facets
    // (fresnel) and interior-facing facets push toward fully solid.
    let alpha = clamp(params.alpha_base + shaded.fresnel * 0.12 + (1.0 - shaded.facing) * 0.06, 0.0, 1.0);

    return vec4<f32>(shaded.color, alpha);
}
