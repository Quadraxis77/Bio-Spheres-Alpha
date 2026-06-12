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
    @location(0) color: vec4<f32>,
    @location(1) shade: f32,
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let instance = cell_instances[camera.partition_offset + instance_index];
    let cell_type = u32(round(instance.type_data_1.w));
    if (cell_type != 17u) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 0.0);
        out.color = vec4<f32>(0.0);
        out.shade = 0.0;
        return out;
    }

    let radial = clamp(in.data.z, 0.0, 1.0);
    let dir = vec3<f32>(in.data.x, in.data.y, 0.0);
    let base_radius = clamp(instance.type_data_0.x, 0.08, 0.65);
    let darkness = clamp(instance.type_data_0.y, 0.0, 1.0);
    let rim_brightness = clamp(instance.type_data_0.z, 0.0, 1.5);
    let nozzle_height = clamp(instance.type_data_0.w, 0.0, 0.55);
    let activity = clamp(instance.type_data_1.y, 0.0, 1.0);
    let pulse = 0.5 + 0.5 * sin(camera.time * mix(2.2, 8.0, activity));

    let throat_radius = base_radius * mix(0.34, 0.72, activity) * (1.0 + pulse * activity * 0.06);
    let outer_radius = base_radius * mix(1.55, 1.95, nozzle_height);
    let band_radius = mix(throat_radius, outer_radius, radial);
    let angle = atan2(in.data.y, in.data.x);
    let rock_ripple = sin(angle * 7.0 + radial * 5.0) * 0.015 * rim_brightness;
    let lip = smoothstep(0.28, 0.52, radial) * (1.0 - smoothstep(0.68, 0.92, radial));
    let skirt = smoothstep(0.72, 1.0, radial);
    let inner_slope = (1.0 - radial) * 0.10;
    let cone_height = nozzle_height * (lip * 0.95 + inner_slope) + rock_ripple * lip;
    let embedded_base = skirt * mix(0.025, 0.075, nozzle_height);
    let local = vec3<f32>(
        dir.x * band_radius,
        dir.y * band_radius,
        -1.0 - cone_height + embedded_base
    ) * instance.radius;

    let world_position = instance.position + quat_rotate(instance.rotation, local);
    let local_normal = normalize(vec3<f32>(dir.xy * (0.22 + cone_height * 2.4), -0.85 + lip * 0.45));
    let world_normal = normalize(quat_rotate(instance.rotation, local_normal));
    let light = max(dot(world_normal, -normalize(lighting.light_dir)), 0.0);

    let rock = mix(instance.color.rgb * 0.70, vec3<f32>(0.42, 0.30, 0.26), 0.58);
    let inner_dark = (1.0 - radial) * darkness;
    let rim = lip * rim_brightness;
    let contact_shadow = skirt * (1.0 - smoothstep(0.92, 1.0, radial));
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.color = vec4<f32>(
        rock + rim * vec3<f32>(0.18, 0.13, 0.08)
            - inner_dark * vec3<f32>(0.32, 0.25, 0.22)
            - contact_shadow * vec3<f32>(0.07, 0.055, 0.045),
        instance.color.a
    );
    out.shade = lighting.ambient + light * 0.75;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb * in.shade, in.color.a);
}
