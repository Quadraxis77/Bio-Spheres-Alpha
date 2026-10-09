// Procedural Sun Shader
// Full-screen analytical rendering of a finite world-space sun sphere.
// Surface and corona detail are anchored in world space, independent of head roll.
// Surface detail, limb darkening, a finite corona, and scene-depth occlusion.
// Rendered as a full-screen triangle without camera-facing geometry.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_rot_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

struct SunParams {
    // Light direction (normalized, pointing toward sun)
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    // Sun angular radius in radians (visual size)
    sun_angular_radius: f32,
    // Sun color (RGB)
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    // Sun intensity multiplier
    sun_intensity: f32,
    // Corona parameters
    corona_radius: f32,    // How far corona extends (multiplier of sun radius)
    corona_intensity: f32, // Brightness of corona
    corona_falloff: f32,   // How quickly corona fades
    // Lens flare parameters
    flare_intensity: f32,  // Overall lens flare brightness
    flare_ghost_count: f32, // Number of lens ghosts (as f32 for uniform compat)
    flare_ghost_dispersal: f32, // Spacing between ghosts
    flare_halo_radius: f32, // Radius of the lens halo ring
    // Sun ray parameters
    ray_intensity: f32,    // Brightness of god rays from sun
    ray_count: f32,        // Number of distinct ray beams
    ray_falloff: f32,      // How quickly rays fade with distance
    // Eclipse occlusion (0.0 = fully eclipsed, 1.0 = fully visible)
    eclipse_factor: f32,
    // Screen dimensions
    screen_width: f32,
    screen_height: f32,
    // Solar flare parameters
    flare_speed: f32,
    sunspot_scale: f32,
    // Additional flare settings
    starburst_intensity: f32,  // Brightness of diffraction spikes
    starburst_points: f32,     // Number of starburst spike points
    starburst_falloff: f32,    // How quickly starburst fades
    streak_intensity: f32,     // Anamorphic streak brightness
    streak_width: f32,         // Vertical tightness of streak
    ghost_size: f32,           // Base size of lens ghosts
    chromatic_aberration: f32, // Color separation in ghosts/halo
    prominence_intensity: f32, // Solar flare/prominence brightness
    glow_intensity: f32,       // Soft bloom glow around sun
    prominence_extent: f32,    // How far prominences reach (falloff rate)
    // Orbit ring gizmo
    orbit_axis_x: f32,
    orbit_axis_y: f32,
    orbit_axis_z: f32,
    orbit_ring_opacity: f32,
    orbit_world_radius: f32,
    sun_distance: f32,
    _pad1: f32,
    _pad2: f32,
}

// Group 0: Camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Group 1: Sun parameters and depth
@group(1) @binding(0)
var<uniform> sun_params: SunParams;

@group(1) @binding(1)
var depth_texture: texture_depth_2d;

@group(1) @binding(2)
var depth_sampler: sampler;

// Vertex output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ============================================================
// Noise functions for procedural sun surface
// ============================================================

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

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

fn fbm3d(p_in: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var p = p_in;
    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise3d(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// ============================================================
// Sun surface rendering
// ============================================================

// The normal is a point on the same world-space globe for both eyes. No
// screen UVs or camera-facing basis enter the surface's procedural coordinates.
fn sun_surface(normal: vec3<f32>, ray_dir: vec3<f32>, time: f32) -> vec3<f32> {
    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);
    let mu = abs(dot(normal, -ray_dir));
    let limb = 0.18 + 0.82 * pow(mu, 0.6);
    let drift = vec3<f32>(time * 0.025, time * 0.012, -time * 0.018);
    let gran = fbm3d(normal * 35.0 + drift, 3);
    let spots = smoothstep(0.52, 0.7, fbm3d(normal * sun_params.sunspot_scale + drift * 0.3, 4));
    let surface = sun_color * limb * (0.85 + gran * 0.3) * (1.0 - spots * 0.75);
    return surface * sun_params.sun_intensity;
}

// A finite coronal shell. Its path length and density are in world space;
// head rotation cannot rotate the streamers around the star.
fn sphere_corona(closest: vec3<f32>, dist: f32, time: f32) -> vec3<f32> {
    if (dist < 1.0 || dist >= sun_params.corona_radius) { return vec3<f32>(0.0); }
    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);
    let extent = max(sun_params.corona_radius - 1.0, 0.001);
    let radial = pow(max(1.0 - (dist - 1.0) / extent, 0.0), sun_params.corona_falloff);
    let shell_path = sqrt(max(sun_params.corona_radius * sun_params.corona_radius - dist * dist, 0.0)) / sun_params.corona_radius;
    let density = fbm3d(normalize(closest) * 5.0 + vec3<f32>(time * 0.03, 0.0, time * 0.01), 3);
    return sun_color * radial * shell_path * (0.15 + density * 0.35)
        * sun_params.corona_intensity * sun_params.sun_intensity;
}

// ============================================================
// Main fragment shader
// ============================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let time = camera.time;

    let light_dir = normalize(vec3<f32>(sun_params.light_dir_x, sun_params.light_dir_y, sun_params.light_dir_z));
    // Center and radius depend only on world settings, never on either eye.
    let sun_center = light_dir * sun_params.sun_distance;
    let sun_radius = sun_params.sun_distance * sin(sun_params.sun_angular_radius);
    let ndc_pos = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // A near-plane point avoids subtracting huge far-plane coordinates, and
    // works with asymmetric OpenXR and off-axis preview projections.
    let near_h = camera.inv_view_rot_proj * vec4<f32>(ndc_pos, 0.0, 1.0);
    let ray_dir = normalize(near_h.xyz / near_h.w);
    let to_center = (sun_center - camera.camera_pos) / sun_radius;
    let along = dot(to_center, ray_dir);
    let closest = ray_dir * along - to_center;
    let dist_in_radii = length(closest);
    let edge_width = max(fwidth(dist_in_radii), 0.0001);
    let root = sqrt(max(1.0 - dist_in_radii * dist_in_radii, 0.0));
    let near_t = along - root;
    let hit_t = select(along + root, near_t, near_t > 0.0) * sun_radius;
    let inside_sun = dot(to_center, to_center) < 1.0;
    let sun_in_front = along > 0.0 || inside_sun;

    // Compare actual ray distances, so foreground terrain cuts the silhouette
    // and geometry behind the finite sun does not incorrectly hide it.
    let depth_size = textureDimensions(depth_texture);
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(depth_size)), vec2<i32>(0), vec2<i32>(depth_size) - vec2<i32>(1));
    let pixel_depth = textureLoad(depth_texture, pixel, 0);
    var scene_t = 1e30;
    if (pixel_depth < 1.0) {
        let geometry_h = camera.inv_view_rot_proj * vec4<f32>(ndc_pos, pixel_depth, 1.0);
        scene_t = max(dot(geometry_h.xyz / geometry_h.w, ray_dir), 0.0);
    }
    var final_color = vec3<f32>(0.0);
    var final_alpha = 0.0;
    let effect_t = select(along * sun_radius, hit_t, dist_in_radii <= 1.0);
    if (sun_in_front && effect_t < scene_t) {
        let coverage = 1.0 - smoothstep(1.0 - edge_width, 1.0 + edge_width, dist_in_radii);
        if (coverage > 0.0) {
            let normal = normalize(ray_dir * (hit_t / sun_radius) - to_center);
            final_color = sun_surface(normal, ray_dir, time) * 8.0;
            final_alpha = coverage;
        }
        let corona_color = sphere_corona(closest, dist_in_radii, time) * 8.0;
        final_color += corona_color;
        final_alpha = max(final_alpha, min(length(corona_color) * 0.3, 0.85));
        final_color *= sun_params.eclipse_factor;
        final_alpha *= sun_params.eclipse_factor;
    }

// ── Orbit ring gizmo ───────────────────────────────────────────────────────
// Renders the sun's orbital path as a circle of radius R centered at
// sun_plane * R in the orbit plane.  The circle passes through the world
// origin (the world sphere's position on the orbit).
//
// Finite world-space intersection — not a skybox effect.

if (sun_params.orbit_ring_opacity > 0.001) {
    // Reuse the translation-free world ray from the sphere intersection.

    // Direction toward the sun.
    let sun_dir = normalize(vec3<f32>(
        sun_params.light_dir_x,
        sun_params.light_dir_y,
        sun_params.light_dir_z
    ));

    // Orbit plane normal.
    let orbit_axis = normalize(vec3<f32>(
        sun_params.orbit_axis_x,
        sun_params.orbit_axis_y,
        sun_params.orbit_axis_z
    ));

    // Project sun direction into the orbit plane.
    var sun_plane = sun_dir - orbit_axis * dot(sun_dir, orbit_axis);

    if (dot(sun_plane, sun_plane) < 0.000001) {
        var fallback = vec3<f32>(1.0, 0.0, 0.0);

        if (abs(dot(fallback, orbit_axis)) > 0.95) {
            fallback = vec3<f32>(0.0, 0.0, 1.0);
        }

        sun_plane = fallback - orbit_axis * dot(fallback, orbit_axis);
    }

    sun_plane = normalize(sun_plane);

    // Arbitrary large visual orbit radius.
    let orbit_radius = 100000.0;

    // Circle center in world space.
    let center = sun_plane * orbit_radius;

    // Intersect view ray with the orbit plane through world origin.
    let denom = dot(ray_dir, orbit_axis);
    let numer = -dot(camera.camera_pos, orbit_axis);

    if (abs(denom) > 0.000001) {
        let t = numer / denom;

        if (t > 0.0) {
            let hit = camera.camera_pos + ray_dir * t;

            // Discard pixels whose hit point falls outside the orbit circle.
            let hit_dist = length(hit - center);
            if (hit_dist <= orbit_radius) {
                let dist_n = (hit_dist - orbit_radius) / orbit_radius;

                // Analytical screen-space derivative: one pixel subtends
                // t / screen_height world units at distance t, divided by
                // abs(denom) to account for the oblique plane intersection.
                // This is stable at any distance unlike fwidth.
                let world_per_pixel = t / (sun_params.screen_height * max(abs(denom), 0.000001));
                let pixel_width = world_per_pixel / orbit_radius;

                let core_pixels = 1.75;
                let glow_pixels = 5.0;

                let core = smoothstep(core_pixels * pixel_width, 0.0, abs(dist_n));
                let glow = smoothstep(glow_pixels * pixel_width, core_pixels * pixel_width, abs(dist_n)) * 0.25;

                let ring = (core + glow) * sun_params.orbit_ring_opacity;

                final_color += vec3<f32>(0.0, 0.8, 1.0) * ring * 5.0;
                final_alpha = max(final_alpha, ring);
            }
        }
    }
}

    // Tone map to prevent harsh clipping
    // Compress brightness without washing the selected sun colour toward white.
    let peak = max(final_color.r, max(final_color.g, final_color.b));
    final_color = final_color / (1.0 + peak);

    return vec4<f32>(final_color, saturate(final_alpha));
}
