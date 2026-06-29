struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;

struct VertexInput {
    @location(0) data0: vec4<f32>,
    @location(1) data1: vec4<f32>,
    @location(2) cell_position: vec3<f32>,
    @location(3) cell_radius: f32,
    @location(4) rotation: vec4<f32>,
    @location(5) color: vec4<f32>,
    @location(6) feather_params: vec4<f32>,
    @location(7) frozen: f32,
    @location(8) phase_offset: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) feather_t: f32,
    @location(2) light: f32,
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return quat_rotate(vec4<f32>(-q.x, -q.y, -q.z, q.w), v);
}

fn safe_perpendicular_axis(n: vec3<f32>) -> vec3<f32> {
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    return normalize(cross(n, helper));
}

fn stroke_point(base_dir: vec3<f32>, side_dir: vec3<f32>, stroke_axis: vec3<f32>, line_kind: f32, root_t: f32, barb_side: f32, t: f32, phase: f32, length_scale: f32) -> vec3<f32> {
    // Dramatic plumocyte feather cycle:
    // 1. Power stroke begins high above the equator at full extension.
    // 2. Feather sweeps far below the equator while still fully extended.
    // 3. Only after the power stroke does it curl inward to reduce drag.
    // 4. It rolls upward while tucked, then opens into the next full extension.
    let pi = 3.14159265359;

    var lift = 0.0;
    var reach_mul = 1.0;
    var tuck = 0.0;
    var roll = 0.0;
    var openness = 1.0;

    if (phase < 0.50) {
        // POWER STROKE: stay long while moving from high-above to low-below.
        let u = smoothstep(0.0, 1.0, phase / 0.50);
        lift = mix(1.65, -1.75, u);
        reach_mul = mix(1.62, 1.54, u);
        tuck = 0.0;
        roll = 0.0;
        openness = 1.25;
    } else if (phase < 0.74) {
        // COLLAPSE: after the downstroke is finished, curl inward to reduce drag.
        let u = smoothstep(0.0, 1.0, (phase - 0.50) / 0.24);
        lift = mix(-1.75, -0.45, u);
        reach_mul = mix(1.54, 0.34, u);
        tuck = mix(0.0, 1.45, u);
        roll = mix(0.0, 0.95, u);
        openness = mix(1.25, 0.12, u);
    } else {
        // RECOVERY / ROLL-UP: rise while tucked, then reopen high above the equator.
        let u = smoothstep(0.0, 1.0, (phase - 0.74) / 0.26);
        lift = mix(-0.45, 1.65, u);
        reach_mul = mix(0.34, 1.62, u);
        tuck = mix(1.45, 0.0, u);
        roll = mix(0.95, 0.0, u);
        openness = mix(0.12, 1.25, u);
    }

    let reach = length_scale * 1.50 * reach_mul;
    let start = 1.02;
    let distal = t * t;
    let arch_profile = mix(0.12, 1.0, distal);
    let curl_profile = sin(t * pi) * mix(0.10, 1.0, t);

    let extension = base_dir * (start + t * reach);
    let vertical_motion = stroke_axis * lift * arch_profile;
    let inward_curl = (-base_dir * 0.95 + stroke_axis * 0.32) * tuck * curl_profile;
    let roll_offset = side_dir * roll * distal * 0.18;

    let centerline = extension + vertical_motion + inward_curl + roll_offset;

    if (line_kind > 0.5) {
        let root_distal = root_t * root_t;
        let root_arch_profile = mix(0.12, 1.0, root_distal);
        let root_curl_profile = sin(root_t * pi) * mix(0.10, 1.0, root_t);

        let root_extension = base_dir * (start + root_t * reach);
        let root_vertical_motion = stroke_axis * lift * root_arch_profile;
        let root_inward_curl = (-base_dir * 0.95 + stroke_axis * 0.32) * tuck * root_curl_profile;
        let root_roll_offset = side_dir * roll * root_distal * 0.18;
        let root = root_extension + root_vertical_motion + root_inward_curl + root_roll_offset;

        let tuck01 = clamp(tuck / 1.45, 0.0, 1.0);
        let barb_len = reach * 0.32 * (1.0 - root_t * 0.22) * mix(1.15, 0.42, tuck01);
        let barb_splay = mix(1.55, 0.10, tuck01) * openness;
        let barb_dir = normalize(
            base_dir * 0.16 +
            side_dir * barb_side * barb_splay +
            stroke_axis * (0.25 + lift * 0.12 + tuck * 0.20)
        );

        return root + barb_dir * barb_len * t;
    }

    return centerline;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let feather_index = u32(round(in.data0.x));
    let line_kind = in.data0.y;
    let t = clamp(in.data0.z, 0.0, 1.0);
    let ribbon_side = in.data0.w;
    let root_t = in.data1.x;
    let barb_side = in.data1.y;

    let feather_length = clamp(in.feather_params.x, 0.2, 2.8);
    let feather_width = clamp(in.feather_params.y, 0.025, 0.22);
    let brightness = clamp(in.feather_params.z, 0.0, 1.8);
    let stroke_speed = clamp(in.feather_params.w, 0.0, 8.0);
    let frozen = in.frozen >= 0.5;
    let phase_offset = fract(in.phase_offset);

    let preview_down_world = vec3<f32>(0.0, -1.0, 0.0);
    let equator_normal = normalize(quat_rotate_inverse(in.rotation, preview_down_world));
    let axis_a = safe_perpendicular_axis(equator_normal);
    let axis_b = normalize(cross(equator_normal, axis_a));

    let angle = f32(feather_index) * 0.78539816339;
    let base_dir = normalize(axis_a * cos(angle) + axis_b * sin(angle));
    let side_dir = normalize(cross(equator_normal, base_dir));
    let stroke_axis = equator_normal;

    let parity = f32(feather_index & 1u);
    let phase = fract(select(camera.time * stroke_speed + phase_offset, 0.0, frozen) + parity * 0.5);

    let p = stroke_point(base_dir, side_dir, stroke_axis, line_kind, root_t, barb_side, t, phase, feather_length);
    let p_next = stroke_point(base_dir, side_dir, stroke_axis, line_kind, root_t, barb_side, min(t + 0.03, 1.0), phase, feather_length);

    let tangent = normalize(p_next - p);
    let width = feather_width * in.cell_radius * mix(0.55, 0.22, t) * select(1.0, 0.72, line_kind > 0.5);
    let normal_offset = normalize(cross(tangent, equator_normal)) * ribbon_side * width;

    let local_position = p * in.cell_radius + normal_offset;
    let world_position = in.cell_position + quat_rotate(in.rotation, local_position);

    let world_normal = normalize(quat_rotate(in.rotation, equator_normal));
    let ndotl = max(dot(world_normal, -normalize(lighting.light_dir)), 0.0);

    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.color = vec4<f32>(mix(in.color.rgb * 0.78, vec3<f32>(0.62, 0.95, 0.82), 0.35), in.color.a);
    out.feather_t = t;
    out.light = lighting.ambient + ndotl * 0.65;
    out.color = vec4<f32>(out.color.rgb * (0.55 + brightness * 0.45), out.color.a);
    out.color.a *= select(0.92, 0.58, frozen) * mix(0.95, 0.45, t);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tip = mix(1.0, 0.68, in.feather_t);
    return vec4<f32>(in.color.rgb * in.light * tip, in.color.a);
}
