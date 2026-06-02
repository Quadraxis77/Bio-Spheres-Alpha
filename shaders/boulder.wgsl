// Boulder Render Shader
//
// Reads boulder state directly from GPU storage buffers - no CPU upload needed.
// The vertex shader indexes into boulder_state by instance_index.
// Dead boulders are culled by outputting a degenerate triangle (all vertices at
// the same clip position) so the rasterizer discards them with zero cost.
//
// Moss is purely visual - FBM noise in boulder-local space creates patchy coverage
// with natural bare spots. No simulation data (nutrients, consumption) affects it.
// The pattern is fixed to the boulder surface and rotates with the orientation
// quaternion so it stays put as the rock spins.

struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    lod_scale_factor: f32,
    lod_threshold_low: f32,
    lod_threshold_medium: f32,
    lod_threshold_high: f32,
}

struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    outline_width: f32,
}

// Must match GpuBoulder in boulder_buffers.rs exactly (80 bytes).
struct GpuBoulder {
    position:         vec3<f32>,
    radius:           f32,
    velocity:         vec3<f32>,
    dead:             u32,
    seed:             u32,
    _pad:             array<u32, 3>,
    angular_velocity: vec4<f32>,  // xyz = omega rad/s, w = unused
    orientation:      vec4<f32>,  // quaternion (x,y,z,w)
}

@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;

// Boulder data read directly from GPU physics buffers - always current.
@group(1) @binding(0) var<storage, read> boulder_state: array<GpuBoulder>;
@group(1) @binding(1) var<storage, read> boulder_count: array<u32>;

// -- Vertex output -------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) seed: u32,
    @location(3) @interpolate(flat) boulder_center: vec3<f32>,
    @location(4) @interpolate(flat) boulder_orient: vec4<f32>,
    @location(5) @interpolate(flat) boulder_radius: f32,
}

// -- Hash helpers --------------------------------------------------------------

fn hash_u32(v: u32) -> f32 {
    var h = v;
    h ^= h >> 16u;
    h  = h * 0x45d9f3bu;
    h ^= h >> 16u;
    return f32(h & 0xFFFFFFu) / 16777216.0;
}

// Per-vertex displacement: signed, so vertices can move in OR out.
// Range: [-0.25, +0.45] - asymmetric so boulders are more convex than concave.
fn vertex_displacement(vert_id: u32, seed: u32) -> f32 {
    let combined = vert_id * 2654435761u ^ seed * 1013904223u;
    let raw = hash_u32(combined); // [0, 1)
    // Map [0,1) -> [-0.25, 0.45]: inward up to 25%, outward up to 45%
    return raw * 0.70 - 0.25;
}

// 3D value noise for moss pattern - cheap, no texture needed.
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    // 8 corner hashes
    let h000 = hash_u32(u32(i.x * 1.0 + i.y * 57.0 + i.z * 113.0) & 0xFFFFFFu);
    let h100 = hash_u32(u32((i.x+1.0) * 1.0 + i.y * 57.0 + i.z * 113.0) & 0xFFFFFFu);
    let h010 = hash_u32(u32(i.x * 1.0 + (i.y+1.0) * 57.0 + i.z * 113.0) & 0xFFFFFFu);
    let h110 = hash_u32(u32((i.x+1.0) * 1.0 + (i.y+1.0) * 57.0 + i.z * 113.0) & 0xFFFFFFu);
    let h001 = hash_u32(u32(i.x * 1.0 + i.y * 57.0 + (i.z+1.0) * 113.0) & 0xFFFFFFu);
    let h101 = hash_u32(u32((i.x+1.0) * 1.0 + i.y * 57.0 + (i.z+1.0) * 113.0) & 0xFFFFFFu);
    let h011 = hash_u32(u32(i.x * 1.0 + (i.y+1.0) * 57.0 + (i.z+1.0) * 113.0) & 0xFFFFFFu);
    let h111 = hash_u32(u32((i.x+1.0) * 1.0 + (i.y+1.0) * 57.0 + (i.z+1.0) * 113.0) & 0xFFFFFFu);

    let x00 = mix(h000, h100, u.x);
    let x10 = mix(h010, h110, u.x);
    let x01 = mix(h001, h101, u.x);
    let x11 = mix(h011, h111, u.x);
    let y0  = mix(x00, x10, u.y);
    let y1  = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

// -- Icosphere geometry --------------------------------------------------------

const PHI: f32 = 1.6180339887498948482;

fn ico_vertex(i: u32) -> vec3<f32> {
    switch (i) {
        case 0u:  { return normalize(vec3<f32>( 0.0,  1.0,  PHI)); }
        case 1u:  { return normalize(vec3<f32>( 0.0, -1.0,  PHI)); }
        case 2u:  { return normalize(vec3<f32>( 0.0,  1.0, -PHI)); }
        case 3u:  { return normalize(vec3<f32>( 0.0, -1.0, -PHI)); }
        case 4u:  { return normalize(vec3<f32>( 1.0,  PHI,  0.0)); }
        case 5u:  { return normalize(vec3<f32>(-1.0,  PHI,  0.0)); }
        case 6u:  { return normalize(vec3<f32>( 1.0, -PHI,  0.0)); }
        case 7u:  { return normalize(vec3<f32>(-1.0, -PHI,  0.0)); }
        case 8u:  { return normalize(vec3<f32>( PHI,  0.0,  1.0)); }
        case 9u:  { return normalize(vec3<f32>(-PHI,  0.0,  1.0)); }
        case 10u: { return normalize(vec3<f32>( PHI,  0.0, -1.0)); }
        case 11u: { return normalize(vec3<f32>(-PHI,  0.0, -1.0)); }
        default:  { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

fn ico_face(f: u32) -> vec3<u32> {
    switch (f) {
        case 0u:  { return vec3<u32>(0u,  1u,  8u);  }
        case 1u:  { return vec3<u32>(0u,  8u,  4u);  }
        case 2u:  { return vec3<u32>(0u,  4u,  5u);  }
        case 3u:  { return vec3<u32>(0u,  5u,  9u);  }
        case 4u:  { return vec3<u32>(0u,  9u,  1u);  }
        case 5u:  { return vec3<u32>(1u,  6u,  8u);  }
        case 6u:  { return vec3<u32>(8u,  6u,  10u); }
        case 7u:  { return vec3<u32>(8u,  10u, 4u);  }
        case 8u:  { return vec3<u32>(4u,  10u, 2u);  }
        case 9u:  { return vec3<u32>(4u,  2u,  5u);  }
        case 10u: { return vec3<u32>(5u,  2u,  11u); }
        case 11u: { return vec3<u32>(5u,  11u, 9u);  }
        case 12u: { return vec3<u32>(9u,  11u, 7u);  }
        case 13u: { return vec3<u32>(9u,  7u,  1u);  }
        case 14u: { return vec3<u32>(1u,  7u,  6u);  }
        case 15u: { return vec3<u32>(3u,  6u,  7u);  }
        case 16u: { return vec3<u32>(3u,  10u, 6u);  }
        case 17u: { return vec3<u32>(3u,  2u,  10u); }
        case 18u: { return vec3<u32>(3u,  11u, 2u);  }
        case 19u: { return vec3<u32>(3u,  7u,  11u); }
        default:  { return vec3<u32>(0u,  1u,  2u);  }
    }
}

// -- Quaternion rotation -------------------------------------------------------
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + (uv * q.w + uuv) * 2.0;
}

// Rotate by the conjugate (inverse for unit quaternions).
fn quat_rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return quat_rotate(vec4<f32>(-q.xyz, q.w), v);
}

// Displace a sphere-direction by the per-vertex hash and return the world position.
// The direction is first rotated by the boulder's orientation quaternion so the
// rock shape spins with the boulder.
fn displaced_pos(dir: vec3<f32>, vert_id: u32, seed: u32, center: vec3<f32>, radius: f32, orientation: vec4<f32>) -> vec3<f32> {
    let disp = vertex_displacement(vert_id, seed);
    // Rotate the base direction by the boulder's current orientation
    let rotated_dir = quat_rotate(orientation, dir);
    return center + rotated_dir * radius * (1.0 + disp);
}

// -- Vertex shader -------------------------------------------------------------

@vertex
fn vs_main(
    @builtin(vertex_index)   vertex_index:   u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let b = boulder_state[instance_index];

    // Cull dead or unspawned boulders by outputting a degenerate triangle.
    // All three vertices collapse to the same clip position -> rasterizer discards.
    var out: VertexOutput;
    if (b.dead != 0u || b.radius <= 0.0) {
        out.clip_pos      = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.world_pos     = vec3<f32>(0.0);
        out.world_normal  = vec3<f32>(0.0, 1.0, 0.0);
        out.seed          = 0u;
        out.boulder_center = vec3<f32>(0.0);
        out.boulder_orient = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.boulder_radius = 0.0;
        return out;
    }

    let face_idx = vertex_index / 12u;
    let sub_tri  = (vertex_index % 12u) / 3u;
    let corner   = vertex_index % 3u;

    let face = ico_face(face_idx);
    let v0 = ico_vertex(face.x);
    let v1 = ico_vertex(face.y);
    let v2 = ico_vertex(face.z);
    let m01 = normalize(v0 + v1);
    let m12 = normalize(v1 + v2);
    let m02 = normalize(v0 + v2);

    // Symmetric edge IDs so adjacent faces agree on shared midpoints.
    let id_v0  = face.x;
    let id_v1  = face.y;
    let id_v2  = face.z;
    let id_m01 = min(face.x, face.y) * 12u + max(face.x, face.y);
    let id_m12 = min(face.y, face.z) * 12u + max(face.y, face.z);
    let id_m02 = min(face.x, face.z) * 12u + max(face.x, face.z);

    // Compute all three displaced positions for this sub-triangle so we can
    // derive a flat face normal from the actual geometry.
    var dir_a: vec3<f32>; var id_a: u32;
    var dir_b: vec3<f32>; var id_b: u32;
    var dir_c: vec3<f32>; var id_c: u32;
    switch (sub_tri) {
        case 0u: { dir_a = v0;  id_a = id_v0;  dir_b = m01; id_b = id_m01; dir_c = m02; id_c = id_m02; }
        case 1u: { dir_a = m01; id_a = id_m01; dir_b = v1;  id_b = id_v1;  dir_c = m12; id_c = id_m12; }
        case 2u: { dir_a = m02; id_a = id_m02; dir_b = m12; id_b = id_m12; dir_c = v2;  id_c = id_v2;  }
        default: { dir_a = m01; id_a = id_m01; dir_b = m12; id_b = id_m12; dir_c = m02; id_c = id_m02; }
    }

    let pa = displaced_pos(dir_a, id_a, b.seed, b.position, b.radius, b.orientation);
    let pb = displaced_pos(dir_b, id_b, b.seed, b.position, b.radius, b.orientation);
    let pc = displaced_pos(dir_c, id_c, b.seed, b.position, b.radius, b.orientation);

    // Flat face normal from actual displaced geometry - gives hard faceted look.
    let flat_normal = normalize(cross(pb - pa, pc - pa));

    // Select this vertex's world position.
    var world_pos: vec3<f32>;
    switch (corner) {
        case 0u: { world_pos = pa; }
        case 1u: { world_pos = pb; }
        default: { world_pos = pc; }
    }

    out.clip_pos       = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos      = world_pos;
    out.world_normal   = flat_normal;
    out.seed           = b.seed;
    out.boulder_center = b.position;
    out.boulder_orient = b.orientation;
    out.boulder_radius = b.radius;
    return out;
}

// -- Fragment shader -----------------------------------------------------------

const ROCK_COLOR_DARK:   vec3<f32> = vec3<f32>(0.18, 0.16, 0.14);
const ROCK_COLOR_LIGHT:  vec3<f32> = vec3<f32>(0.45, 0.40, 0.35);
const MOSS_COLOR_DARK:   vec3<f32> = vec3<f32>(0.04, 0.12, 0.03);
const MOSS_COLOR_BRIGHT: vec3<f32> = vec3<f32>(0.16, 0.40, 0.10);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(-lighting.light_dir);
    let v = normalize(camera.camera_pos - in.world_pos);

    // Diffuse + ambient
    let ndotl   = max(dot(n, l), 0.0);
    let diffuse = ndotl * lighting.light_color;
    let ambient = vec3<f32>(lighting.ambient);

    // Blinn-Phong specular - rock is rough, keep it subtle
    let h    = normalize(l + v);
    let spec = pow(max(dot(n, h), 0.0), 12.0) * 0.06;

    // Rock base color: vary per face using the flat normal for a stone-like look
    let face_var   = abs(dot(n, normalize(vec3<f32>(0.577, 0.577, 0.577))));
    let rock_color = mix(ROCK_COLOR_DARK, ROCK_COLOR_LIGHT, face_var);

    // -- Moss coverage ---------------------------------------------------------
    // Transform the fragment's world position into boulder-local space so the
    // noise pattern is fixed to the surface and rotates with the boulder.
    let local_pos = quat_rotate_inv(in.boulder_orient, in.world_pos - in.boulder_center);

    // Normalise by radius so patch scale is consistent across boulder sizes.
    // The seed offsets the noise domain so each boulder has a unique pattern.
    let seed_offset = vec3<f32>(
        hash_u32(in.seed)             * 31.7,
        hash_u32(in.seed + 1u)        * 29.3,
        hash_u32(in.seed + 2u)        * 37.1,
    );
    let noise_pos = local_pos / max(in.boulder_radius, 0.5) + seed_offset;

    // Four octaves of FBM - large patches from low frequency, fine detail from high.
    let n1 = noise3(noise_pos * 0.9);
    let n2 = noise3(noise_pos * 2.1  + vec3<f32>(17.3,  5.7, 11.1)) * 0.50;
    let n3 = noise3(noise_pos * 4.7  + vec3<f32>( 3.7, 22.4,  8.9)) * 0.25;
    let n4 = noise3(noise_pos * 10.3 + vec3<f32>(31.1,  7.3, 19.5)) * 0.12;
    let raw_noise = (n1 + n2 + n3 + n4) / 1.87;

    // Fixed threshold creates ~50% coverage with natural patchy bare spots.
    // smoothstep width of 0.08 gives soft patch edges.
    let moss_mask = smoothstep(0.42, 0.58, raw_noise);

    // Two-tone moss colour from a finer noise layer - avoids flat green.
    let detail_noise = noise3(noise_pos * 8.5 + vec3<f32>(5.1, 13.7, 2.3));
    let moss_color   = mix(MOSS_COLOR_DARK, MOSS_COLOR_BRIGHT, detail_noise);

    // Blend rock and moss
    let base_color = mix(rock_color, moss_color, moss_mask);

    let lit = base_color * (ambient + diffuse) + vec3<f32>(spec);
    return vec4<f32>(lit, 1.0);
}
