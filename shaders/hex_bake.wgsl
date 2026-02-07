// Compute shader: bake Goldberg hex triplet pattern into equirectangular texture.
// Output: RG texture where R = edge_dist (Voronoi d2-d1), G = is_hex (1.0 or 0.0)
// The texture is sampled by cell_unified.wgsl using spherical direction → UV mapping.

const ICO_PHI: f32 = 1.618033988749895;
const PI: f32 = 3.14159265359;

fn ico_base_vertex(idx: u32) -> vec3<f32> {
    switch (idx) {
        case 0u:  { return normalize(vec3<f32>( 0.0,  1.0,  ICO_PHI)); }
        case 1u:  { return normalize(vec3<f32>( 0.0, -1.0,  ICO_PHI)); }
        case 2u:  { return normalize(vec3<f32>( 0.0,  1.0, -ICO_PHI)); }
        case 3u:  { return normalize(vec3<f32>( 0.0, -1.0, -ICO_PHI)); }
        case 4u:  { return normalize(vec3<f32>( 1.0,  ICO_PHI, 0.0)); }
        case 5u:  { return normalize(vec3<f32>(-1.0,  ICO_PHI, 0.0)); }
        case 6u:  { return normalize(vec3<f32>( 1.0, -ICO_PHI, 0.0)); }
        case 7u:  { return normalize(vec3<f32>(-1.0, -ICO_PHI, 0.0)); }
        case 8u:  { return normalize(vec3<f32>( ICO_PHI, 0.0,  1.0)); }
        case 9u:  { return normalize(vec3<f32>(-ICO_PHI, 0.0,  1.0)); }
        case 10u: { return normalize(vec3<f32>( ICO_PHI, 0.0, -1.0)); }
        case 11u: { return normalize(vec3<f32>(-ICO_PHI, 0.0, -1.0)); }
        default:  { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

fn ico_face(idx: u32) -> vec3<u32> {
    switch (idx) {
        case 0u:  { return vec3<u32>(0, 1, 8); }
        case 1u:  { return vec3<u32>(0, 8, 4); }
        case 2u:  { return vec3<u32>(0, 4, 5); }
        case 3u:  { return vec3<u32>(0, 5, 9); }
        case 4u:  { return vec3<u32>(0, 9, 1); }
        case 5u:  { return vec3<u32>(1, 6, 8); }
        case 6u:  { return vec3<u32>(8, 6, 10); }
        case 7u:  { return vec3<u32>(8, 10, 4); }
        case 8u:  { return vec3<u32>(4, 10, 2); }
        case 9u:  { return vec3<u32>(4, 2, 5); }
        case 10u: { return vec3<u32>(5, 2, 11); }
        case 11u: { return vec3<u32>(5, 11, 9); }
        case 12u: { return vec3<u32>(9, 11, 7); }
        case 13u: { return vec3<u32>(9, 7, 1); }
        case 14u: { return vec3<u32>(1, 7, 6); }
        case 15u: { return vec3<u32>(3, 6, 7); }
        case 16u: { return vec3<u32>(3, 10, 6); }
        case 17u: { return vec3<u32>(3, 2, 10); }
        case 18u: { return vec3<u32>(3, 11, 2); }
        case 19u: { return vec3<u32>(3, 7, 11); }
        default:  { return vec3<u32>(0, 1, 2); }
    }
}

// Compute hex triplet pattern for a direction on the unit sphere.
// Returns vec2(edge_dist, is_hex).
fn compute_hex_pattern(p: vec3<f32>) -> vec2<f32> {
    // Find nearest ico vertex
    var best_vert = 0u;
    var best_vdot = -2.0;
    for (var v = 0u; v < 12u; v++) {
        let d = dot(p, ico_base_vertex(v));
        if (d > best_vdot) {
            best_vdot = d;
            best_vert = v;
        }
    }

    var d1 = 10.0;
    var d2 = 10.0;
    var nearest_is_corner = false;

    // Check faces sharing nearest vertex
    for (var face = 0u; face < 20u; face++) {
        let fi = ico_face(face);
        if (fi.x != best_vert && fi.y != best_vert && fi.z != best_vert) {
            continue;
        }

        let va = ico_base_vertex(fi.x);
        let vb = ico_base_vertex(fi.y);
        let vc = ico_base_vertex(fi.z);

        for (var i = 0; i <= 4; i++) {
            for (var j = 0; j <= 4 - i; j++) {
                let k = 4 - i - j;
                let raw = va * f32(i) + vb * f32(j) + vc * f32(k);
                let dp = dot(p, raw);
                let inv_len = inverseSqrt(dot(raw, raw));
                let cos_dist = 1.0 - dp * inv_len;

                if (cos_dist < d1) {
                    d2 = d1;
                    d1 = cos_dist;
                    nearest_is_corner = (i == 4 || j == 4 || k == 4);
                } else if (cos_dist < d2) {
                    d2 = cos_dist;
                }
            }
        }
    }

    let edge_dist = d2 - d1;
    let is_hex = select(0.0, 1.0, !nearest_is_corner);
    return vec2<f32>(edge_dist, is_hex);
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Equirectangular: UV → spherical direction
    let u = (f32(gid.x) + 0.5) / f32(dims.x); // 0..1
    let v = (f32(gid.y) + 0.5) / f32(dims.y); // 0..1

    let theta = u * 2.0 * PI;        // longitude: 0..2π
    let phi = v * PI;                  // latitude: 0..π (top to bottom)

    let sin_phi = sin(phi);
    let dir = vec3<f32>(
        sin_phi * cos(theta),
        cos(phi),
        sin_phi * sin(theta)
    );

    let result = compute_hex_pattern(dir);

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(result.x, result.y, 0.0, 0.0));
}
