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
    // shader and flat-interpolated. Screen-space derivatives (dpdx/dpdy)
    // degenerate for near-horizontal facets viewed from directly above (the
    // tiny vertical displacement gets swamped by the much larger x/z screen
    // derivatives, so the cross product collapses back to the smooth
    // normal) - a top-down view of a flat ice sheet showed no faceting at
    // all. Carrying the true per-facet plane normal from the vertex shader
    // is view-independent and fixes that.
    @location(3) @interpolate(flat) facet_normal: vec3<f32>,
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

// Returns the integer cell coordinate of the nearest Voronoi crystal feature
// point in `p` (already scaled by facet_scale) - constant across each
// irregular polygonal region, with sharp boundaries between regions.
fn voronoi_cell(p: vec3<f32>) -> vec3<f32> {
    let base = floor(p);
    var best_d = 1e9;
    var best_cell = base;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            for (var z = -1; z <= 1; z++) {
                let cell = base + vec3<f32>(f32(x), f32(y), f32(z));
                let feature = cell + hash3_vec(cell);
                let diff = p - feature;
                let d = dot(diff, diff);
                if (d < best_d) {
                    best_d = d;
                    best_cell = cell;
                }
            }
        }
    }
    return best_cell;
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

fn facet_displace(pos: vec3<f32>, smooth_normal: vec3<f32>) -> FacetResult {
    if params.displacement_strength <= 0.0 {
        return FacetResult(pos, smooth_normal);
    }

    let scale = max(params.facet_scale, 0.001);
    let p = pos * scale;
    let cell = voronoi_cell(p);
    let feature_world = (cell + hash3_vec(cell)) / scale;

    let tilt = hash3_vec(cell + vec3<f32>(17.31, 47.7, 93.1)) * 2.0 - 1.0;
    let facet_normal = normalize(smooth_normal + tilt);

    // Skip near-degenerate planes (facet normal nearly perpendicular to the
    // direction we're displacing along) rather than risk a huge spike.
    let denom = dot(smooth_normal, facet_normal);
    if abs(denom) < 0.2 {
        return FacetResult(pos, smooth_normal);
    }

    let t_plane = dot(feature_world - pos, facet_normal) / denom;
    let max_disp = 1.0 / scale;
    let t = clamp(t_plane, -max_disp, max_disp) * params.displacement_strength;
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

struct FragmentOutput {
    @location(0) accum: vec4<f32>,   // WBOIT accumulation (premultiplied color * weight, weight)
    @location(1) revealage: vec4<f32>, // WBOIT revealage (1 - alpha product)
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // Deeply faceted: vertices were displaced onto flat per-Voronoi-cell
    // planes in the vertex shader, so each triangle is genuinely planar.
    // Use the per-facet plane normal carried (flat) from the vertex shader -
    // this drives SPECULAR almost exclusively; body color and base diffuse
    // come from the smooth normal, so facets read as glassy faces catching
    // the light differently, not as patches of different paint (full facet
    // diffuse looked like a patchwork quilt). Screen-space derivatives
    // (dpdx/dpdy) were used previously, but degenerate to the smooth normal
    // for near-horizontal facets viewed from directly above.
    let smooth_normal = normalize(in.world_normal);
    let view_dir = normalize(in.view_dir);
    let facet_normal = normalize(in.facet_normal);

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

    // Mostly opaque with a hint of translucency: ice is a dense solid, and
    // WBOIT's weighted average already softens it against whatever is behind
    // - a low alpha here reads as ghostly rather than icy. Grazing facets
    // (fresnel) and interior-facing facets push toward fully solid.
    let alpha = clamp(params.alpha_base + fresnel * 0.12 + (1.0 - facing) * 0.06, 0.0, 0.97);

    // WBOIT weight function (McGuire & Bavoil 2013) - same shape as
    // density_mesh so ice and water composite order-independently, but
    // boosted so the dense solid dominates the weighted average where its
    // fragments coincide with the water surface or sit in front of it.
    const ICE_WEIGHT_BOOST: f32 = 6.0;
    let depth = in.clip_position.z;
    let w = ICE_WEIGHT_BOOST * alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(depth / 200.0, 4.0))));

    var out: FragmentOutput;
    out.accum = vec4<f32>(final_color * w, w);
    out.revealage = vec4<f32>(alpha, 0.0, 0.0, alpha);
    return out;
}
