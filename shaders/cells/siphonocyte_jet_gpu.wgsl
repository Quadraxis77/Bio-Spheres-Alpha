struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    partition_offset: u32,
    gravity_mode: u32,
    gravity: f32,
    _padding: f32,
}

struct SiphonJetParams {
    delta_time: f32,
    current_time: f32,
    current_frame: u32,
    max_particles: u32,
    cell_capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: vec4<f32>,
    _pad4: vec4<f32>,
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

struct SiphonJetParticle {
    position: vec3<f32>,
    size: f32,
    velocity: vec3<f32>,
    age: f32,
    max_age: f32,
    style: f32,
    seed: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> jet_params: SiphonJetParams;
@group(0) @binding(1) var<storage, read> cell_instances: array<CellInstance>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particles: array<SiphonJetParticle>;
@group(0) @binding(4) var<storage, read_write> ring_counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> build_counters: array<u32>;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn hash_u32(v: u32) -> f32 {
    var h = v;
    h ^= h >> 16u;
    h = h * 0x45d9f3bu;
    h ^= h >> 16u;
    return f32(h & 0x00ffffffu) / 16777216.0;
}

fn rand3(seed: u32) -> vec3<f32> {
    return vec3<f32>(
        hash_u32(seed * 1664525u + 1013904223u) * 2.0 - 1.0,
        hash_u32(seed * 22695477u + 1664525u) * 2.0 - 1.0,
        hash_u32(seed * 1013904223u + 22695477u) * 2.0 - 1.0,
    );
}

@compute @workgroup_size(64)
fn spawn_jets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let emission_lanes = 4u;
    let idx = gid.x / emission_lanes;
    let lane = gid.x % emission_lanes;
    let siphon_count = build_counters[4u + 17u];
    if (gid.x >= siphon_count * emission_lanes || idx >= jet_params.cell_capacity) { return; }

    let instance = cell_instances[idx];
    let style = floor(instance.type_data_1.x + 0.001);
    if (style < 0.5) { return; }

    let activity = clamp(instance.type_data_1.y, 0.0, 1.0);
    let seed = idx * 747796405u ^ lane * 1597334677u ^ jet_params.current_frame * 2891336453u;
    let emit_rate = mix(26.0, 76.0, activity);
    if (hash_u32(seed) > emit_rate * jet_params.delta_time) { return; }

    let slot = atomicAdd(&ring_counter[0], 1u) % jet_params.max_particles;
    let is_steam = style > 1.5;
    let jet_dir = normalize(quat_rotate(instance.rotation, vec3<f32>(0.0, 0.0, -1.0)));
    let start = instance.position + jet_dir * instance.radius * 1.08;
    let jitter = normalize(rand3(seed ^ 0xa511e9b3u));
    let sideways = normalize(jitter - jet_dir * dot(jitter, jet_dir));
    let spread = mix(0.015, 0.09, hash_u32(seed ^ 0x68bc21ebu)) * select(0.9, 1.35, is_steam);
    let spawn_pos = start + sideways * instance.radius * spread;

    let speed = instance.radius * mix(5.4, 10.5, activity) * select(1.0, 0.7, is_steam);
    let drift = (sideways * mix(0.4, 1.8, hash_u32(seed ^ 0x51ed270bu)) + rand3(seed ^ 0x9e3779b9u) * 0.35)
        * instance.radius * select(0.7, 1.1, is_steam);
    let velocity = jet_dir * speed + drift;
    let lifetime = mix(0.46, 1.05, hash_u32(seed ^ 0x85ebca6bu)) * select(1.0, 1.35, is_steam);
    let size = instance.radius * mix(0.055, 0.16, hash_u32(seed ^ 0xc2b2ae35u)) * select(0.9, 1.65, is_steam);

    particles[slot] = SiphonJetParticle(
        spawn_pos,
        size,
        velocity,
        0.0,
        lifetime,
        style,
        hash_u32(seed ^ 0x27d4eb2du),
        0.0,
    );
}

@compute @workgroup_size(256)
fn age_jets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= jet_params.max_particles) { return; }

    var p = particles[idx];
    if (p.max_age <= 0.0) { return; }

    p.age += jet_params.delta_time;
    if (p.age >= p.max_age) {
        particles[idx] = SiphonJetParticle(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    let t = clamp(p.age / p.max_age, 0.0, 1.0);
    let turbulence = rand3(idx * 747796405u ^ jet_params.current_frame * 1597334677u)
        * mix(0.15, 1.05, t) * p.size;
    p.position += (p.velocity + turbulence) * jet_params.delta_time;
    p.velocity *= pow(mix(0.91, 0.80, t), jet_params.delta_time * 60.0);
    particles[idx] = p;
}

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) size: f32,
    @location(2) velocity: vec3<f32>,
    @location(3) age: f32,
    @location(4) max_age: f32,
    @location(5) style: f32,
    @location(6) seed: f32,
    @location(7) _pad: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) style: f32,
}

const QUAD_INDICES: array<u32, 6> = array<u32, 6>(0u, 2u, 1u, 1u, 2u, 3u);

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    if (inst.max_age <= 0.0 || inst.size <= 0.0) {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.style = 0.0;
        return out;
    }

    let corner = QUAD_INDICES[vid];
    let uv = vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u),
    );
    let t = clamp(inst.age / inst.max_age, 0.0, 1.0);
    let plume_size = inst.size * mix(0.45, 3.1, smoothstep(0.0, 1.0, t));
    let offset = (uv * 2.0 - vec2<f32>(1.0)) * plume_size;
    let to_cam = normalize(camera.camera_pos - inst.position);
    var right = cross(to_cam, vec3<f32>(0.0, 1.0, 0.0));
    if (length(right) < 0.001) {
        right = cross(to_cam, vec3<f32>(1.0, 0.0, 0.0));
    }
    right = normalize(right);
    let up = normalize(cross(right, to_cam));
    let world_position = inst.position + right * offset.x + up * offset.y;

    let alpha = smoothstep(0.0, 0.04, t) * (1.0 - smoothstep(0.72, 1.0, t));
    let is_steam = inst.style > 1.5;
    let hot_core = vec3<f32>(0.88, 0.98, 1.0);
    let blue_flame = vec3<f32>(0.18, 0.62, 1.0);
    let smoke = select(vec3<f32>(0.28, 0.52, 0.68), vec3<f32>(0.72, 0.76, 0.76), is_steam);
    let plume_color = mix(mix(hot_core, blue_flame, smoothstep(0.05, 0.34, t)), smoke, smoothstep(0.46, 1.0, t));
    let plume_alpha = alpha * mix(0.95, 0.38, smoothstep(0.45, 1.0, t)) * select(1.0, 0.82, is_steam);

    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.uv = uv * 2.0 - vec2<f32>(1.0);
    out.color = vec4<f32>(plume_color, plume_alpha);
    out.style = inst.style;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    let soft = smoothstep(1.0, 0.18, r);
    let core = select(smoothstep(0.72, 0.15, r), smoothstep(1.0, 0.0, r), in.style > 1.5);
    return vec4<f32>(in.color.rgb * (0.65 + core * 0.45), in.color.a * soft);
}
