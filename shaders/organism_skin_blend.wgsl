// Organism Skin Temporal Density Blend
//
// Blends current frame density with previous frame for smooth skin motion.
// For each slot, if the organism ID matches the previous frame, lerp the density.
// If the organism changed (new organism appeared or old one left), snap immediately.
//
// Runs after normalize_density, before surface nets extraction.

struct BlendParams {
    total_voxels: u32,
    blend_factor: f32,  // 0.0 = keep previous, 1.0 = use new (typical: 0.3–0.5)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BlendParams;

// Current frame density (read-write: we blend in-place)
@group(0) @binding(1)  var<storage, read_write> density_0: array<f32>;
@group(0) @binding(2)  var<storage, read_write> density_1: array<f32>;
@group(0) @binding(3)  var<storage, read_write> density_2: array<f32>;
@group(0) @binding(4)  var<storage, read_write> density_3: array<f32>;

// Current frame organism IDs (read-only for matching)
@group(0) @binding(5)  var<storage, read> org_id_0: array<u32>;
@group(0) @binding(6)  var<storage, read> org_id_1: array<u32>;
@group(0) @binding(7)  var<storage, read> org_id_2: array<u32>;
@group(0) @binding(8)  var<storage, read> org_id_3: array<u32>;

// Previous frame density (read-write: updated after blend)
@group(0) @binding(9)  var<storage, read_write> prev_density_0: array<f32>;
@group(0) @binding(10) var<storage, read_write> prev_density_1: array<f32>;
@group(0) @binding(11) var<storage, read_write> prev_density_2: array<f32>;
@group(0) @binding(12) var<storage, read_write> prev_density_3: array<f32>;

// Previous frame organism IDs (read-write: updated after blend)
@group(0) @binding(13) var<storage, read_write> prev_org_id_0: array<u32>;
@group(0) @binding(14) var<storage, read_write> prev_org_id_1: array<u32>;
@group(0) @binding(15) var<storage, read_write> prev_org_id_2: array<u32>;
@group(0) @binding(16) var<storage, read_write> prev_org_id_3: array<u32>;

// Find the previous density for a given organism in any of the 4 prev slots.
// Returns the previous density if the organism was present, 0.0 otherwise.
fn find_prev_density(idx: u32, org: u32) -> f32 {
    if prev_org_id_0[idx] == org { return prev_density_0[idx]; }
    if prev_org_id_1[idx] == org { return prev_density_1[idx]; }
    if prev_org_id_2[idx] == org { return prev_density_2[idx]; }
    if prev_org_id_3[idx] == org { return prev_density_3[idx]; }
    return 0.0;
}

@compute @workgroup_size(256, 1, 1)
fn blend_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.total_voxels { return; }
    let idx = gid.x;
    let alpha = params.blend_factor;

    // Slot 0
    let cur_org_0 = org_id_0[idx];
    let cur_den_0 = density_0[idx];
    if cur_org_0 != 0u {
        let prev = find_prev_density(idx, cur_org_0);
        density_0[idx] = mix(prev, cur_den_0, alpha);
    }
    prev_density_0[idx] = density_0[idx];
    prev_org_id_0[idx] = cur_org_0;

    // Slot 1
    let cur_org_1 = org_id_1[idx];
    let cur_den_1 = density_1[idx];
    if cur_org_1 != 0u {
        let prev = find_prev_density(idx, cur_org_1);
        density_1[idx] = mix(prev, cur_den_1, alpha);
    }
    prev_density_1[idx] = density_1[idx];
    prev_org_id_1[idx] = cur_org_1;

    // Slot 2
    let cur_org_2 = org_id_2[idx];
    let cur_den_2 = density_2[idx];
    if cur_org_2 != 0u {
        let prev = find_prev_density(idx, cur_org_2);
        density_2[idx] = mix(prev, cur_den_2, alpha);
    }
    prev_density_2[idx] = density_2[idx];
    prev_org_id_2[idx] = cur_org_2;

    // Slot 3
    let cur_org_3 = org_id_3[idx];
    let cur_den_3 = density_3[idx];
    if cur_org_3 != 0u {
        let prev = find_prev_density(idx, cur_org_3);
        density_3[idx] = mix(prev, cur_den_3, alpha);
    }
    prev_density_3[idx] = density_3[idx];
    prev_org_id_3[idx] = cur_org_3;
}
