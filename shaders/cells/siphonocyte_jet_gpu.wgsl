struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    partition_offset: u32,
    gravity_mode: u32,
    gravity: f32,
    _padding: f32,
}

struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _padding: f32,
}

struct CellInstance {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    visual_params: vec4<f32>,
    rotation: vec4<f32>,
    type_data_0: vec4<f32>,
    type_data_1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<storage, read> cell_instances: array<CellInstance>;

struct VertexInput {
    @location(0) data: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) style: f32,
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn safe_perp(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let c = cross(a, b);
    if (length(c) > 0.001) {
        return normalize(c);
    }
    return normalize(cross(a, vec3<f32>(0.0, 1.0, 0.0)));
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let instance = cell_instances[camera.partition_offset + instance_index];
    let cell_type = u32(round(instance.type_data_1.w));
    let style = instance.type_data_1.x;
    if (cell_type != 17u || style < 0.5) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 0.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.style = 0.0;
        return out;
    }

    let is_steam = style > 1.5;
    let activity = clamp(instance.type_data_0.w, 0.0, 1.0);
    let t = fract(in.data.z - camera.time * mix(2.2, 0.8, select(0.0, 1.0, is_steam)));
    let seed = in.data.w;
    let corner = in.data.xy;

    let jet_dir = normalize(quat_rotate(instance.rotation, vec3<f32>(0.0, 0.0, -1.0)));
    let start = instance.position + jet_dir * instance.radius * 1.04;
    let to_cam = normalize(camera.camera_pos - start);
    let right = safe_perp(jet_dir, to_cam);
    let up = normalize(cross(right, jet_dir));

    let length_scale = instance.radius * mix(1.8, 3.2, activity) * select(1.0, 1.35, is_steam);
    let spread = instance.radius * mix(0.05, 0.22, t) * select(0.65, 1.8, is_steam);
    let phase = seed * 6.28318 + camera.time * mix(18.0, 5.0, select(0.0, 1.0, is_steam));
    let swirl = right * sin(phase + t * 11.0) + up * cos(phase * 0.7 + t * 9.0);
    let center = start + jet_dir * (t * length_scale) + swirl * spread * (0.35 + seed);
    let size = instance.radius * mix(0.045, 0.105, seed) * mix(1.0, 1.8, t) * select(0.75, 1.65, is_steam);

    let world_position = center + (right * corner.x + up * corner.y) * size;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.uv = corner;
    out.style = style;
    let alpha = mix(0.82, 0.18, t) * select(1.0, 0.72, is_steam);
    let water_color = vec4<f32>(0.26, 0.72, 1.0, alpha);
    let steam_color = vec4<f32>(0.86, 0.91, 0.92, alpha * 0.72);
    out.color = select(water_color, steam_color, is_steam);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    let soft = smoothstep(1.0, 0.18, r);
    let core = select(smoothstep(0.72, 0.15, r), smoothstep(1.0, 0.0, r), in.style > 1.5);
    return vec4<f32>(in.color.rgb * (0.65 + core * 0.45), in.color.a * soft);
}
