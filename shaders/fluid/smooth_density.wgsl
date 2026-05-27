// Density smoothing compute shader
// Applies 3x3x3 spatial box blur + temporal exponential moving average
// This produces a stable, smooth density field from rapidly-changing voxel data

struct SmoothParams {
    grid_resolution: u32,
    blend_factor: f32,  // 0.0 = keep previous, 1.0 = use new blurred instantly
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: SmoothParams;
@group(0) @binding(1) var<storage, read> raw_density: array<f32>;
@group(0) @binding(2) var<storage, read> prev_smoothed: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

@compute @workgroup_size(4, 4, 4)
fn smooth_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if (gid.x >= res || gid.y >= res || gid.z >= res) {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);

    // Early-out: skip the expensive 27-sample blur for voxels that are empty now
    // and have no recent history. These dominate the grid at high resolutions.
    let raw_center = raw_density[idx];
    let prev_center = prev_smoothed[idx];
    if raw_center < 0.001 && prev_center < 0.001 {
        output[idx] = 0.0;
        return;
    }

    // 3x3x3 spatial box blur of raw density
    var sum = 0.0;
    var count = 0.0;
    let ires = i32(res);
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nx = i32(gid.x) + dx;
                let ny = i32(gid.y) + dy;
                let nz = i32(gid.z) + dz;
                if (nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires) {
                    sum += raw_density[grid_index(u32(nx), u32(ny), u32(nz))];
                    count += 1.0;
                }
            }
        }
    }
    let blurred = sum / count;

    // Temporal blend: asymmetric exponential moving average.
    // Decay fast (high blend) when density drops — removes ghost mesh left by moving steam.
    // Grow slowly (low blend) when density rises — keeps pools and puddles stable.
    let prev = prev_smoothed[idx];
    let decay_blend  = 0.6;  // fast decay: ghost mesh from steam clears in ~2 frames
    let growth_blend = 0.12; // slow growth: water surface stays smooth and stable
    let blend = select(growth_blend, decay_blend, blurred < prev);
    output[idx] = mix(prev, blurred, blend);
}
