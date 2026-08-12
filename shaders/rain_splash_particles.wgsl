// Rain splash rendering shader
//
// Unlike the camera-facing billboard particles (water/steam/nutrient), a
// splash ring lies flat on the impact surface - oriented by its stored
// surface normal, not the camera - and expands outward while fading, like a
// ripple spreading from where a raindrop landed.

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
    @location(3) animation: vec4<f32>,   // x=age, y=max_lifetime
    @location(4) orientation: vec4<f32>, // xyz=surface normal at impact
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) progress: f32,
}

const QUAD_INDICES: array<u32, 6> = array<u32, 6>(0u, 2u, 1u, 1u, 2u, 3u);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    instance: VertexInput,
) -> VertexOutput {
    let progress = clamp(instance.animation.x / max(instance.animation.y, 0.001), 0.0, 1.0);

    // Ring expands outward over its lifetime.
    let ring_size = instance.size * mix(0.2, 1.0, progress);

    var normal = instance.orientation.xyz;
    if dot(normal, normal) < 0.01 {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    }
    normal = normalize(normal);

    // Flat tangent basis on the impact surface (perpendicular to normal) -
    // not billboarded toward the camera, since a splash ring lies on the water.
    var tangent = cross(normal, vec3<f32>(0.0, 0.0, 1.0));
    if dot(tangent, tangent) < 0.01 {
        tangent = cross(normal, vec3<f32>(1.0, 0.0, 0.0));
    }
    tangent = normalize(tangent);
    let bitangent = cross(normal, tangent);

    let corner = QUAD_INDICES[vertex_id];
    let quad_offset = vec2<f32>(
        select(-1.0, 1.0, (corner & 1u) != 0u),
        select(-1.0, 1.0, (corner & 2u) != 0u)
    ) * ring_size;

    // Lift slightly off the surface along the normal to avoid z-fighting with
    // the water mesh underneath.
    let world_pos = instance.position
        + normal * (instance.size * 0.03)
        + tangent * quad_offset.x
        + bitangent * quad_offset.y;

    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    let uv = vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u)
    );

    var out: VertexOutput;
    out.clip_pos = clip_pos;
    out.uv = uv;
    out.color = instance.color;
    out.progress = progress;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_c = (in.uv - vec2<f32>(0.5)) * 2.0; // -1..1
    let dist = length(uv_c);
    if dist > 1.0 {
        discard;
    }

    // Thin ring that expands outward from the impact point and fades as it grows.
    let ring_radius = mix(0.12, 0.92, in.progress);
    let ring_width = 0.24;
    let ring = 1.0 - smoothstep(0.0, ring_width, abs(dist - ring_radius));

    let fade = 1.0 - in.progress;
    let alpha = ring * fade * in.color.a;
    if alpha < 0.01 {
        discard;
    }

    return vec4<f32>(in.color.rgb, alpha);
}
