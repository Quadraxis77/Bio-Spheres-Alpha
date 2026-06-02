// Water particle rendering shader
// Renders small, prominent water particles as billboarded quads
// Moving particles are elongated along the gravity axis to simulate falling droplets

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct WaterRenderParams {
    gravity_mode: u32,  // 0=X, 1=Y, 2=Z, 3=radial
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> render_params: WaterRenderParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) animation: vec4<f32>,  // x=time_offset, yzw=velocity direction
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) animation: vec4<f32>,
    @location(4) elongation: f32,  // How much the particle is elongated (0=circle, >0=stretched)
}

// Quad vertex indices for two triangles with CCW winding
const QUAD_INDICES: array<u32, 6> = array<u32, 6>(0u, 2u, 1u, 1u, 2u, 3u);

// Get the gravity axis direction based on gravity_mode
fn get_gravity_axis() -> vec3<f32> {
    if render_params.gravity_mode == 0u {
        return vec3<f32>(1.0, 0.0, 0.0);  // X axis
    } else if render_params.gravity_mode == 2u {
        return vec3<f32>(0.0, 0.0, 1.0);  // Z axis
    }
    // Default Y axis (mode 1) and radial (mode 3 falls back to Y)
    return vec3<f32>(0.0, 1.0, 0.0);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    instance: VertexInput,
) -> VertexOutput {
    // Extract velocity from animation.yzw
    let velocity = vec3<f32>(instance.animation.y, instance.animation.z, instance.animation.w);
    let speed = length(velocity);
    let is_moving = speed > 0.1;

    // Sub-voxel lerp: smoothly slide the particle toward the next voxel position.
    // The fluid sim moves voxels discretely; we animate the visual position continuously
    // by offsetting along the velocity direction by a fraction of one cell_size.
    // step_freq controls how many voxel-steps per second the animation assumes.
    // At ~8 steps/sec the particle glides one full cell every ~0.125s, which matches
    // typical fluid sim throughput at 60fps with 4 sub-steps.
    let step_freq = 8.0;
    let phase = fract(instance.animation.x * step_freq);  // 0->1 within current step
    let cell_size = instance.size / 0.7;  // recover approximate cell_size from particle size
    // Lerp moving particles toward next voxel. For stationary ones (zero velocity),
    // add a tiny time-based wobble so they don't appear frozen in place.
    var lerp_offset: vec3<f32>;
    if is_moving {
        lerp_offset = velocity * phase * cell_size;
    } else {
        // Sub-pixel drift: oscillate gently so the particle doesn't look glued to a grid point
        let t = instance.animation.x;
        let drift_scale = cell_size * 0.08;
        lerp_offset = vec3<f32>(
            sin(t * 3.7 + instance.position.x) * drift_scale,
            sin(t * 2.9 + instance.position.y) * drift_scale,
            sin(t * 4.1 + instance.position.z) * drift_scale
        );
    }

    let world_pos = instance.position + lerp_offset;
    // Scale size down for stationary/slow particles - rain drops are larger,
    // circular stationary ones should be smaller so they don't dominate.
    let size_scale = mix(0.35, 1.0, smoothstep(0.0, 0.5, speed));
    let size = instance.size * size_scale;

    // Determine elongation factor for teardrop shape when moving.
    // Use a soft speed threshold so the elongated shape lingers briefly after
    // the particle slows - prevents snapping instantly to a circle.
    // speed is 0 (stationary), ~1.0 (cardinal move), or ~1.7 (diagonal).
    // We map speed through a smooth curve and blend with a slow decay driven
    // by the time phase so the shape fades over ~0.3s rather than one frame.
    let speed_factor = smoothstep(0.05, 0.8, speed);
    // Decay: fract(time * decay_freq) gives a 0->1 ramp; when speed is zero the
    // elongation rides this ramp down to 0 over one cycle (~0.25s at 4Hz).
    let decay_phase = fract(instance.animation.x * 4.0);
    let lingering = speed_factor + (1.0 - speed_factor) * (1.0 - decay_phase);
    let elongation_factor = lingering * 1.8;

    // Get corner index for this vertex
    let corner = QUAD_INDICES[vertex_id];
    let quad_offset = vec2<f32>(
        select(-1.0, 1.0, (corner & 1u) != 0u),
        select(-1.0, 1.0, (corner & 2u) != 0u)
    ) * size * 0.5;

    // Calculate billboard axes
    let to_camera = normalize(camera.camera_pos - world_pos);

    // For moving particles: elongate along gravity axis projected onto the billboard plane
    var right: vec3<f32>;
    var up: vec3<f32>;

    if elongation_factor > 0.05 {
        // Get gravity axis and use it as the stretch direction
        var stretch_dir: vec3<f32>;
        if render_params.gravity_mode == 3u {
            // Radial mode: stretch toward origin (falling inward)
            stretch_dir = normalize(world_pos);
        } else {
            stretch_dir = get_gravity_axis();
        }

        // Project stretch direction onto the billboard plane (perpendicular to view)
        let stretch_on_plane = stretch_dir - dot(stretch_dir, to_camera) * to_camera;
        let stretch_len = length(stretch_on_plane);

        if stretch_len > 0.001 {
            // Use the projected gravity axis as "up" for the billboard
            up = normalize(stretch_on_plane);
            right = cross(up, to_camera);
        } else {
            // Gravity axis is parallel to view direction, fall back to standard billboard
            right = normalize(cross(to_camera, vec3<f32>(0.0, 1.0, 0.0)));
            up = cross(right, to_camera);
        }

        // Apply elongation: stretch along the gravity axis (up), compress perpendicular (right)
        let stretch_y = 1.0 + elongation_factor;  // Stretch along gravity
        let compress_x = 1.0 / (1.0 + elongation_factor * 0.3);  // Slight compression perpendicular

        let vertex_world = world_pos 
            + right * quad_offset.x * compress_x 
            + up * quad_offset.y * stretch_y;

        let clip_pos = camera.view_proj * vec4<f32>(vertex_world, 1.0);

        let uv = vec2<f32>(
            select(0.0, 1.0, (corner & 1u) != 0u),
            select(0.0, 1.0, (corner & 2u) != 0u)
        );

        var out: VertexOutput;
        out.clip_pos = clip_pos;
        out.uv = uv;
        out.color = instance.color;
        out.world_pos = world_pos;
        out.animation = instance.animation;
        out.elongation = elongation_factor;
        return out;
    } else {
        // Standard circular billboard for stationary/slow particles - apply lerp offset too
        right = normalize(cross(to_camera, vec3<f32>(0.0, 1.0, 0.0)));
        up = cross(right, to_camera);

        let vertex_world = world_pos + right * quad_offset.x + up * quad_offset.y;
        let clip_pos = camera.view_proj * vec4<f32>(vertex_world, 1.0);

        let uv = vec2<f32>(
            select(0.0, 1.0, (corner & 1u) != 0u),
            select(0.0, 1.0, (corner & 2u) != 0u)
        );

        var out: VertexOutput;
        out.clip_pos = clip_pos;
        out.uv = uv;
        out.color = instance.color;
        out.world_pos = world_pos;
        out.animation = instance.animation;
        out.elongation = 0.0;
        return out;
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv - vec2<f32>(0.5);  // center at (0,0), range -0.5..0.5

    var alpha: f32;

    if in.elongation > 0.1 {
        // Teardrop shape for falling droplets.
        // The vertex shader stretches along the gravity axis, so uv.y maps to that axis.
        // uv.y > 0 = tip (pointed, against motion), uv.y < 0 = belly (round).
        var dist: f32;
        if uv.y > 0.0 {
            // Tip: narrow parabolic cone - squash x to create a sharp point
            dist = length(vec2<f32>(uv.x * 2.8, uv.y));
        } else {
            // Belly: standard circle
            dist = length(uv) / 0.38;
        }
        alpha = 1.0 - smoothstep(0.38, 0.52, dist);

        // Specular highlight - small bright spot on the belly
        let hl = length(uv - vec2<f32>(-0.09, -0.11));
        alpha = max(alpha, (1.0 - smoothstep(0.0, 0.11, hl)) * 0.65);
    } else {
        // Stationary droplet - round with a crisp edge and a highlight
        let dist = length(uv);
        alpha = 1.0 - smoothstep(0.35, 0.48, dist);

        let hl = length(uv - vec2<f32>(-0.09, -0.09));
        alpha = max(alpha, (1.0 - smoothstep(0.0, 0.09, hl)) * 0.55);
    }

    if alpha < 0.01 {
        discard;
    }

    let shimmer = sin(in.animation.x * 3.0) * 0.05 + 0.95;
    return vec4<f32>(in.color.rgb * shimmer, in.color.a * alpha);
}
