// Preview Scene Skybox
//
// Dark gradient background with a perspective grid floor.
// Colors are driven by theme params uploaded each frame.

struct CameraUniforms {
    // Rotation-only inverse view-proj (no translation = no shaking)
    inv_view_rot_proj: mat4x4<f32>,
    // Full view-proj for projecting the grid floor
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

// Theme-driven color params — uploaded from the active UI palette each frame.
struct SkyboxThemeParams {
    zenith_color:  vec3<f32>,  // sky color at the top
    _pad0:         f32,
    horizon_color: vec3<f32>,  // sky color at the horizon
    _pad1:         f32,
    glow_color:    vec3<f32>,  // horizon glow tint
    _pad2:         f32,
    grid_color:    vec3<f32>,  // perspective grid line color
    grid_opacity:  f32,        // grid brightness multiplier
    floor_color:   vec3<f32>,  // solid floor background color (between grid lines)
    _pad3:         f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> theme:  SkyboxThemeParams;

// ── Vertex shader ─────────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32((vi >> 1u) & 1u) * 4 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 1.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Smooth grid lines: returns 1.0 on a line, 0.0 between lines
fn grid_line(v: f32, line_width: f32) -> f32 {
    let f = abs(fract(v - 0.5) - 0.5);
    return 1.0 - smoothstep(0.0, line_width, f);
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ── Reconstruct view direction (rotation-only, no shaking) ────────────────
    let ndc    = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let wh     = camera.inv_view_rot_proj * ndc;
    let dir    = normalize(wh.xyz / wh.w);

    // ── Sky gradient ──────────────────────────────────────────────────────────
    // dir.y: +1 = straight up, -1 = straight down, 0 = horizon
    let horizon_t = clamp(dir.y * 1.5 + 0.1, 0.0, 1.0);
    let horizon_smooth = horizon_t * horizon_t * (3.0 - 2.0 * horizon_t);

    var sky = mix(theme.horizon_color, theme.zenith_color, horizon_smooth);

    // Faint glow near the horizon
    let glow = max(0.0, 1.0 - abs(dir.y) * 4.0);
    sky += theme.glow_color * glow * glow;

    // ── Perspective grid floor ────────────────────────────────────────────────
    // Only draw grid where the ray hits the floor plane (y < 0 direction)
    var color = sky;

    if (dir.y < -0.001) {
        let floor_world_y = -28.0;
        let t = (floor_world_y - camera.camera_pos.y) / dir.y;

        if (t > 0.0 && t < 800.0) {
            let hit = camera.camera_pos + dir * t;

            let grid_scale = 6.0;
            let gx = grid_line(hit.x / grid_scale, 0.02);
            let gz = grid_line(hit.z / grid_scale, 0.02);
            let line = max(gx, gz);

            // Fade with distance
            let fade = exp(-t * 0.008) * clamp(1.0 - t * 0.001, 0.0, 1.0);

            // Replace sky with the dark floor background, then add grid lines on top.
            // floor_alpha fades the floor in with distance so it blends into the horizon.
            let floor_alpha = clamp(fade * 2.0, 0.0, 1.0);
            let floor_base = mix(sky, theme.floor_color, floor_alpha);
            let grid_contrib = theme.grid_color * line * fade * theme.grid_opacity;
            color = floor_base + grid_contrib;
        }
    }

    return vec4<f32>(color, 1.0);
}
