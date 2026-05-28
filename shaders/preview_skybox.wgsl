// Preview Scene Skybox
//
// Dark teal/navy gradient background with a perspective grid floor,
// matching the Bio-Spheres Lab genome editor aesthetic.
//
// Background: deep navy at top, dark teal toward horizon
// Grid: faint cyan perspective grid on the floor plane

struct CameraUniforms {
    // Rotation-only inverse view-proj (no translation = no shaking)
    inv_view_rot_proj: mat4x4<f32>,
    // Full view-proj for projecting the grid floor
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

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
    // Map to 0 (horizon) → 1 (zenith) for the upper hemisphere
    let horizon_t = clamp(dir.y * 1.5 + 0.1, 0.0, 1.0);
    let horizon_smooth = horizon_t * horizon_t * (3.0 - 2.0 * horizon_t);

    // Deep navy at zenith, dark teal at horizon
    let zenith_color  = vec3<f32>(0.012, 0.020, 0.055); // #030514 deep navy
    let horizon_color = vec3<f32>(0.018, 0.065, 0.085); // #041117 dark teal
    var sky = mix(horizon_color, zenith_color, horizon_smooth);

    // Very faint teal glow near the horizon
    let glow = max(0.0, 1.0 - abs(dir.y) * 4.0);
    sky += vec3<f32>(0.0, 0.04, 0.06) * glow * glow;

    // ── Perspective grid floor ────────────────────────────────────────────────
    // Only draw grid where the ray hits the floor plane (y < 0 direction)
    var grid_contrib = vec3<f32>(0.0);

    if (dir.y < -0.001) {
        // Ray-plane intersection: floor at fixed world y = FLOOR_WORLD_Y
        // Placed well below the origin so organisms don't clip through it.
        let floor_world_y = -28.0;
        let t = (floor_world_y - camera.camera_pos.y) / dir.y;

        if (t > 0.0 && t < 800.0) {
            let hit = camera.camera_pos + dir * t;

            // Grid spacing — larger value = fewer, more spread-out lines
            let grid_scale = 6.0;
            let gx = grid_line(hit.x / grid_scale, 0.02);
            let gz = grid_line(hit.z / grid_scale, 0.02);
            let line = max(gx, gz);

            // Fade with distance
            let dist = t;
            let fade = exp(-dist * 0.008) * clamp(1.0 - dist * 0.001, 0.0, 1.0);

            // Cyan grid color matching the UI aesthetic
            let grid_color = vec3<f32>(0.05, 0.55, 0.65);
            grid_contrib = grid_color * line * fade * 0.35;
        }
    }

    var color = sky + grid_contrib;
    return vec4<f32>(color, 1.0);
}
