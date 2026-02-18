// Procedural Sun Shader
// Full-screen post-process that renders a procedural sun at infinite distance
// along the light direction vector. Features:
//   1. Animated sun disk with sunspots and solar flares
//   2. Corona glow with animated tendrils
//   3. Volumetric sun rays (radial light shafts)
//   4. Lens flare effects (ghosts, halo, starburst)
//   5. Eclipse support: depth buffer occlusion fades all effects
//
// Rendered as a full-screen triangle pass composited over the scene.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
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

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i + vec2<f32>(0.0, 0.0));
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2d(p_in: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var p = p_in;
    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise2d(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
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

// Generate sunspot pattern
fn sunspots(uv: vec2<f32>, time: f32) -> f32 {
    let scale = sun_params.sunspot_scale;
    
    // Simple Perlin noise that evolves over time - faster movement
    let p = vec3<f32>(uv.x * scale, uv.y * scale, time * 0.3);
    let noise = fbm3d(p, 4);
    
    // Convert noise to sunspots (dark regions) - make them much darker
    let threshold = 0.5;
    let spot = smoothstep(threshold, threshold + 0.2, noise);
    return spot * 0.8; // Much darker
}

// Generate solar flare / prominence shapes
fn solar_flares(angle: f32, dist: f32, time: f32) -> f32 {
    let speed = sun_params.flare_speed;

    // Use sin/cos of angle to avoid seam at atan2 discontinuity (±π)
    let ca = cos(angle);
    let sa = sin(angle);

    // Multiple flare tendrils at different angular positions
    var flare = 0.0;

    // Large slow-moving flares (use sin/cos pairs for seamless wrapping)
    let n1 = fbm2d(vec2<f32>(ca * 2.0 + sa * 1.5 + time * speed * 0.3, dist * 3.0 + time * speed * 0.1), 4);
    let n2 = fbm2d(vec2<f32>(sa * 2.5 - ca * 1.8 - time * speed * 0.2, dist * 2.0 - time * speed * 0.15), 3);
    flare += max(0.0, n1 - 0.35) * 2.0;
    flare += max(0.0, n2 - 0.4) * 1.5;

    // Small fast flickering flares
    let n3 = noise2d(vec2<f32>(ca * 4.0 + sa * 3.0 + time * speed * 2.0, dist * 5.0));
    flare += max(0.0, n3 - 0.6) * 0.8;

    // Fade with distance from sun edge
    let edge_dist = dist - 1.0; // 0 at sun edge, positive outside
    if (edge_dist < 0.0) {
        return 0.0; // Inside sun disk
    }
    // If extent is 0, no prominences (avoid division by zero)
    if (sun_params.prominence_extent <= 0.0) {
        return 0.0;
    }
    let falloff = exp(-edge_dist / sun_params.prominence_extent);
    return flare * falloff * sun_params.prominence_intensity;
}

// Render the sun disk with surface detail
fn sun_disk(sun_uv: vec2<f32>, dist_from_center: f32, time: f32) -> vec3<f32> {
    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);

    if (dist_from_center > 1.0) {
        return vec3<f32>(0.0);
    }

    // Base sun color with limb darkening
    let limb = 1.0 - dist_from_center * dist_from_center;
    let limb_color = mix(sun_color * 0.6, sun_color, sqrt(limb));

    // Granulation (convection cells on sun surface)
    let gran = fbm2d(sun_uv * 20.0 + vec2<f32>(time * 0.05, time * 0.03), 3);
    let granulation = mix(0.95, 1.05, gran);

    // Sunspots (dark regions)
    let spots = sunspots(sun_uv, time);

    // Combine
    var color = limb_color * granulation * (1.0 - spots);

    // Bright edge glow (chromosphere)
    let edge = smoothstep(0.85, 1.0, dist_from_center);
    color += sun_color * edge * 0.3;

    return color * sun_params.sun_intensity;
}

// ============================================================
// Corona rendering
// ============================================================

fn corona(angle: f32, dist: f32, time: f32) -> vec3<f32> {
    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);

    if (dist < 0.95) {
        return vec3<f32>(0.0); // Inside sun disk
    }

    // Radial falloff
    let r = dist - 1.0;
    if (r < 0.0) {
        // Thin chromosphere ring at the very edge
        let edge_glow = smoothstep(1.0, 0.95, dist) * 2.0;
        return sun_color * edge_glow * sun_params.corona_intensity;
    }

    let corona_extent = sun_params.corona_radius - 1.0;
    if (r > corona_extent) {
        return vec3<f32>(0.0);
    }

    // Base radial falloff
    let radial = pow(1.0 - r / corona_extent, sun_params.corona_falloff);

    // Streamer structure (elongated features)
    let streamer_noise = fbm2d(vec2<f32>(angle * 4.0 + time * 0.1, r * 2.0 + time * 0.05), 4);
    let streamers = 0.5 + 0.5 * streamer_noise;

    // Fine detail
    let detail = fbm2d(vec2<f32>(angle * 12.0 - time * 0.3, r * 8.0 + time * 0.1), 3);
    let fine = 0.7 + 0.3 * detail;

    // Coronal loops (arch-like structures)
    let loop_noise = noise2d(vec2<f32>(angle * 6.0 + time * 0.08, 0.0));
    let loop_shape = exp(-pow((r - 0.2 * loop_noise) * 5.0, 2.0));

    let intensity = radial * streamers * fine + loop_shape * 0.3 * radial;

    // Corona color shifts slightly toward white at outer edges
    let corona_color = mix(sun_color, vec3<f32>(1.0, 0.95, 0.9), r / corona_extent * 0.5);

    return corona_color * intensity * sun_params.corona_intensity;
}

// ============================================================
// Sun rays (radial light shafts)
// ============================================================

fn sun_rays(uv_from_sun: vec2<f32>, time: f32) -> f32 {
    let dist = length(uv_from_sun);
    if (dist < 0.001 || dist > 1.0) {
        return 0.0;
    }

    let angle = atan2(uv_from_sun.y, uv_from_sun.x);
    let ray_count = sun_params.ray_count;

    // Sharp static rays - no rotation
    var rays = 0.0;
    // Primary rays (sharp spikes)
    let r1 = sin(angle * ray_count) * 0.5 + 0.5;
    rays += pow(r1, 12.0);
    // Secondary rays (half-count, offset, thinner)
    let r2 = sin(angle * ray_count * 0.5 + 0.7854) * 0.5 + 0.5;
    rays += pow(r2, 16.0) * 0.4;
    // Subtle intensity variation along rays (static)
    let noise_mod = fbm2d(vec2<f32>(angle * 2.0, dist * 3.0), 2);
    rays *= 0.8 + 0.2 * noise_mod;

    // Radial falloff
    let falloff = exp(-dist * sun_params.ray_falloff);

    return rays * falloff * sun_params.ray_intensity;
}

// ============================================================
// Lens flare effects
// ============================================================

// Single lens ghost (circular bright spot)
fn lens_ghost(uv: vec2<f32>, pos: vec2<f32>, radius: f32, color: vec3<f32>) -> vec3<f32> {
    let d = length(uv - pos);
    let intensity = smoothstep(radius, radius * 0.3, d);
    // Ring enhancement
    let ring = smoothstep(radius * 0.8, radius * 0.7, d) * smoothstep(radius * 0.5, radius * 0.6, d);
    return color * (intensity * 0.3 + ring * 0.7);
}

// Anamorphic flare streak (horizontal light streak)
fn anamorphic_streak(uv: vec2<f32>, sun_screen: vec2<f32>) -> vec3<f32> {
    let dy = abs(uv.y - sun_screen.y);
    let dx = abs(uv.x - sun_screen.x);
    let streak = exp(-dy * sun_params.streak_width) * exp(-dx * 2.0);
    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);
    // Chromatic dispersion
    let r = streak * 1.0;
    let g = streak * 0.8;
    let b = streak * 0.5;
    return vec3<f32>(r, g, b) * sun_color * sun_params.streak_intensity;
}

// Starburst pattern (diffraction spikes)
fn starburst(uv_from_sun: vec2<f32>, time: f32) -> f32 {
    let dist = length(uv_from_sun);
    if (dist < 0.001) {
        return 1.0;
    }

    let angle = atan2(uv_from_sun.y, uv_from_sun.x);

    // Primary star points
    var burst = 0.0;
    let spike_count = sun_params.starburst_points;
    let spike = pow(abs(cos(angle * spike_count * 0.5)), 64.0);
    burst += spike;

    // Finer secondary star rotated
    let spike2 = pow(abs(cos((angle + 0.3927) * 2.0)), 128.0);
    burst += spike2 * 0.5;

    let falloff = exp(-dist * sun_params.starburst_falloff);
    return burst * falloff;
}

// Full lens flare system
fn lens_flare(uv: vec2<f32>, sun_screen: vec2<f32>, time: f32) -> vec3<f32> {
    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);
    var flare = vec3<f32>(0.0);

    // Vector from screen center to sun
    let center = vec2<f32>(0.5, 0.5);
    let sun_vec = sun_screen - center;
    let ghost_count = i32(sun_params.flare_ghost_count);

    // Lens ghosts (reflections along the sun-center axis)
    for (var i = 0; i < ghost_count; i++) {
        let t = sun_params.flare_ghost_dispersal * (f32(i + 1) / f32(ghost_count));
        let ghost_pos = center - sun_vec * t;

        // Vary ghost size and color
        let size = sun_params.ghost_size + sun_params.ghost_size * 0.75 * f32(i);
        let hue_shift = f32(i) * sun_params.chromatic_aberration;
        let ghost_color = vec3<f32>(
            sun_color.r * (1.0 - hue_shift * 0.3),
            sun_color.g * (1.0 + hue_shift * 0.1),
            sun_color.b * (1.0 + hue_shift * 0.5),
        );
        flare += lens_ghost(uv, ghost_pos, size, ghost_color) * 0.15;
    }

    // Halo ring around sun position
    let halo_dist = length(uv - sun_screen);
    let halo_ring = smoothstep(sun_params.flare_halo_radius + 0.01, sun_params.flare_halo_radius, halo_dist)
                  * smoothstep(sun_params.flare_halo_radius - 0.03, sun_params.flare_halo_radius - 0.01, halo_dist);
    // Rainbow-ish halo
    let halo_angle = atan2(uv.y - sun_screen.y, uv.x - sun_screen.x);
    let halo_color = vec3<f32>(
        0.7 + 0.3 * sin(halo_angle * 2.0),
        0.7 + 0.3 * sin(halo_angle * 2.0 + 2.094),
        0.7 + 0.3 * sin(halo_angle * 2.0 + 4.189),
    );
    flare += halo_color * halo_ring * 0.2;

    // Anamorphic streak
    flare += anamorphic_streak(uv, sun_screen);

    // Starburst at sun position
    let uv_from_sun = uv - sun_screen;
    let burst = starburst(uv_from_sun, time);
    flare += sun_color * burst * sun_params.starburst_intensity;

    return flare * sun_params.flare_intensity;
}

// ============================================================
// Eclipse detection
// ============================================================

// Sample depth buffer around sun screen position to determine eclipse factor
fn compute_eclipse(sun_screen_uv: vec2<f32>) -> f32 {
    // If sun is off-screen, treat as partially eclipsed based on distance to edge
    if (sun_screen_uv.x < -0.1 || sun_screen_uv.x > 1.1 ||
        sun_screen_uv.y < -0.1 || sun_screen_uv.y > 1.1) {
        return 0.0;
    }

    // Sample depth at multiple points around the sun center
    // to get smooth eclipse transitions
    let texel_size = vec2<f32>(1.0 / sun_params.screen_width, 1.0 / sun_params.screen_height);
    let sample_radius = sun_params.sun_angular_radius * 15.0; // Sample area around sun

    var visible_samples = 0.0;
    let total_samples = 16.0;

    // Poisson-disk-like sampling pattern
    let offsets = array<vec2<f32>, 16>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.25, 0.0),
        vec2<f32>(-0.25, 0.0),
        vec2<f32>(0.0, 0.25),
        vec2<f32>(0.0, -0.25),
        vec2<f32>(0.18, 0.18),
        vec2<f32>(-0.18, 0.18),
        vec2<f32>(0.18, -0.18),
        vec2<f32>(-0.18, -0.18),
        vec2<f32>(0.5, 0.0),
        vec2<f32>(-0.5, 0.0),
        vec2<f32>(0.0, 0.5),
        vec2<f32>(0.0, -0.5),
        vec2<f32>(0.35, 0.35),
        vec2<f32>(-0.35, 0.35),
        vec2<f32>(0.35, -0.35),
    );

    for (var i = 0; i < 16; i++) {
        let sample_uv = sun_screen_uv + offsets[i] * sample_radius;
        // Clamp to screen bounds
        let clamped_uv = clamp(sample_uv, vec2<f32>(0.001), vec2<f32>(0.999));
        let depth = textureSample(depth_texture, depth_sampler, clamped_uv);

        // If depth is very close to 1.0 (far plane), nothing is occluding
        if (depth > 0.99999) {
            visible_samples += 1.0;
        }
    }

    return visible_samples / total_samples;
}

// ============================================================
// Main fragment shader
// ============================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let time = camera.time;

    // Get light direction (points toward sun)
    let light_dir = normalize(vec3<f32>(sun_params.light_dir_x, sun_params.light_dir_y, sun_params.light_dir_z));

    // Project sun position to screen space
    // Sun is at infinite distance, so we project the direction vector
    let far_sun_pos = camera.camera_pos + light_dir * 500.0; // Far enough to be "at infinity"
    let clip_pos = camera.view_proj * vec4<f32>(far_sun_pos, 1.0);

    // Check if sun is behind camera
    if (clip_pos.w <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let ndc = clip_pos.xyz / clip_pos.w;
    // Convert NDC to UV (flip Y for wgpu)
    let sun_screen = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    // Compute eclipse factor from depth buffer
    let eclipse = compute_eclipse(sun_screen) * sun_params.eclipse_factor;

    if (eclipse < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // UV relative to sun center
    let aspect = sun_params.screen_width / sun_params.screen_height;
    var uv_from_sun = uv - sun_screen;
    uv_from_sun.x *= aspect; // Correct for aspect ratio

    // Distance from sun center in screen space
    let screen_dist = length(uv_from_sun);

    // Convert to sun-radius-relative distance
    let sun_radius_screen = sun_params.sun_angular_radius;
    let dist_in_radii = screen_dist / sun_radius_screen;

    // Angle from sun center
    let angle = atan2(uv_from_sun.y, uv_from_sun.x);

    // Sun UV (normalized to sun disk, -1 to 1)
    let sun_uv = uv_from_sun / sun_radius_screen;

    let sun_color = vec3<f32>(sun_params.sun_color_r, sun_params.sun_color_g, sun_params.sun_color_b);

    // Per-pixel depth check: sample depth at this fragment's UV
    // If geometry exists at this pixel (depth < 1.0), skip sun/corona/glow
    let pixel_depth = textureSample(depth_texture, depth_sampler, uv);
    let has_geometry = pixel_depth < 0.99999;

    var final_color = vec3<f32>(0.0);
    var final_alpha = 0.0;

    // Sun-position effects only render where there is NO geometry
    if (!has_geometry) {
        // 1. Sun disk with surface detail
        let disk = sun_disk(sun_uv, dist_in_radii, time);
        final_color += disk;
        if (dist_in_radii < 1.0) {
            final_alpha = 1.0;
        }

        // 2. Solar flares removed
        // Note: solar_flares function still exists but is not called

        // 3. Corona
        let corona_color = corona(angle, dist_in_radii, time);
        final_color += corona_color;
        final_alpha = max(final_alpha, length(corona_color) * 0.5);

        // 4. Sun rays (volumetric-style radial beams)
        let rays = sun_rays(uv_from_sun / sun_radius_screen * 0.15, time);
        final_color += sun_color * rays * 0.5;
        final_alpha = max(final_alpha, rays * 0.3);

        // 5. Soft glow around sun (large-scale bloom approximation)
        let glow_falloff = exp(-dist_in_radii * 0.8);
        let glow = sun_color * glow_falloff * sun_params.glow_intensity * sun_params.sun_intensity;
        final_color += glow;
        final_alpha = max(final_alpha, glow_falloff * 0.1);
    }

    // 6. Lens flare removed
    // Note: lens_flare function still exists but is not called

    // Apply eclipse factor
    final_color *= eclipse;
    final_alpha *= eclipse;

    // Tone map to prevent harsh clipping
    final_color = final_color / (1.0 + final_color);

    return vec4<f32>(final_color, saturate(final_alpha));
}
