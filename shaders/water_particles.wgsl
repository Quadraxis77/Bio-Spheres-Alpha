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
    let world_pos = instance.position;
    let size = instance.size;

    // Extract velocity from animation.yzw
    let velocity = vec3<f32>(instance.animation.y, instance.animation.z, instance.animation.w);
    let speed = length(velocity);

    // Determine elongation factor based on whether the particle is moving
    // speed is 0 or ~1.0 (normalized direction) or ~1.7 (diagonal)
    let is_moving = speed > 0.1;
    let elongation_factor = select(0.0, 1.8, is_moving);  // 1.8x stretch when moving

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

    if is_moving {
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
        // Standard circular billboard for stationary particles
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
    let centered_uv = in.uv - vec2<f32>(0.5);

    // For elongated particles, use an ellipse shape instead of a circle
    // Scale the UV so the ellipse fits within the stretched quad
    var dist: f32;
    if in.elongation > 0.1 {
        // Elliptical shape: compress distance check along stretch axis
        // This makes the particle look like an elongated teardrop
        let stretch_y = 1.0 + in.elongation;
        let compress_x = 1.0 / (1.0 + in.elongation * 0.3);
        let scaled_uv = vec2<f32>(
            centered_uv.x / compress_x,
            centered_uv.y / stretch_y
        );
        dist = length(scaled_uv);
    } else {
        // Standard circular particle
        dist = length(centered_uv);
    }

    // Water droplets should have more defined edges than steam
    let circle_alpha = 1.0 - smoothstep(0.2, 0.4, dist);
    
    // Add a subtle shimmer effect
    let shimmer = sin(in.animation.x * 3.0) * 0.1 + 0.9;
    
    // Discard pixels outside the shape
    if circle_alpha < 0.01 {
        discard;
    }

    // Apply shimmer to color brightness
    let final_color = in.color.rgb * shimmer;
    
    return vec4<f32>(final_color, in.color.a * circle_alpha);
}
