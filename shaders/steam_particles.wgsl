// Steam particle rendering shader
// Renders billboarded quads for steam particles

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
    @location(3) animation: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

// Quad vertex indices for two triangles with CCW winding
const QUAD_INDICES: array<u32, 6> = array<u32, 6>(0u, 2u, 1u, 1u, 2u, 3u);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    instance: VertexInput,
) -> VertexOutput {
    let world_pos = instance.position;
    let size = instance.size;

    // Get corner index for this vertex
    let corner = QUAD_INDICES[vertex_id];
    let quad_offset = vec2<f32>(
        select(-1.0, 1.0, (corner & 1u) != 0u),
        select(-1.0, 1.0, (corner & 2u) != 0u)
    ) * size * 0.5;

    // Calculate view-aligned quad (billboard)
    let to_camera = normalize(camera.camera_pos - world_pos);
    let right = normalize(cross(to_camera, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, to_camera);

    let vertex_world = world_pos + right * quad_offset.x + up * quad_offset.y;

    // Transform to clip space
    let clip_pos = camera.view_proj * vec4<f32>(vertex_world, 1.0);

    // Calculate UV coordinates (0 to 1 range)
    let uv = vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u)
    );

    var out: VertexOutput;
    out.clip_pos = clip_pos;
    out.uv = uv;
    out.color = instance.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create soft circular particle
    let center_dist = length(in.uv - vec2<f32>(0.5));
    let circle_alpha = 1.0 - smoothstep(0.3, 0.5, center_dist);

    // Discard pixels outside the circle
    if circle_alpha < 0.01 {
        discard;
    }

    return vec4<f32>(in.color.rgb, in.color.a * circle_alpha);
}
