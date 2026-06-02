// Organism Skin Spatial Smoothing - K=4 per-slot 3x3x3 box blur
//
// Each slot's density is blurred independently as a plain scalar field,
// exactly like the water smooth_density shader. This spreads density into
// neighboring empty voxels at organism boundaries, producing a smooth surface.
//
// Also propagates organism IDs to empty voxels that gain density from the blur,
// so surface nets can assign the correct color.
//
// Runs after normalize_density, before temporal blend.

struct SmoothParams {
    grid_resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SmoothParams;

// Input: normalized density per slot (from normalize_density pass)
@group(0) @binding(1)  var<storage, read> density_in_0: array<f32>;
@group(0) @binding(2)  var<storage, read> density_in_1: array<f32>;
@group(0) @binding(3)  var<storage, read> density_in_2: array<f32>;
@group(0) @binding(4)  var<storage, read> density_in_3: array<f32>;

// Input: organism IDs per slot
@group(0) @binding(5)  var<storage, read> org_in_0: array<u32>;
@group(0) @binding(6)  var<storage, read> org_in_1: array<u32>;
@group(0) @binding(7)  var<storage, read> org_in_2: array<u32>;
@group(0) @binding(8)  var<storage, read> org_in_3: array<u32>;

// Output: smoothed density per slot
@group(0) @binding(9)  var<storage, read_write> density_out_0: array<f32>;
@group(0) @binding(10) var<storage, read_write> density_out_1: array<f32>;
@group(0) @binding(11) var<storage, read_write> density_out_2: array<f32>;
@group(0) @binding(12) var<storage, read_write> density_out_3: array<f32>;

// Output: propagated organism IDs per slot
@group(0) @binding(13) var<storage, read_write> org_out_0: array<u32>;
@group(0) @binding(14) var<storage, read_write> org_out_1: array<u32>;
@group(0) @binding(15) var<storage, read_write> org_out_2: array<u32>;
@group(0) @binding(16) var<storage, read_write> org_out_3: array<u32>;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

@compute @workgroup_size(4, 4, 4)
fn smooth_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let ires = i32(res);
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);

    // Read center densities for early-out
    let c0 = density_in_0[idx];
    let c1 = density_in_1[idx];
    let c2 = density_in_2[idx];
    let c3 = density_in_3[idx];

    // Fast early-out: if center is empty, check if ANY neighbor has density.
    if c0 == 0.0 && c1 == 0.0 && c2 == 0.0 && c3 == 0.0 {
        var any_neighbor = false;
        for (var dz: i32 = -1; dz <= 1; dz++) {
            if any_neighbor { break; }
            for (var dy: i32 = -1; dy <= 1; dy++) {
                if any_neighbor { break; }
                for (var dx: i32 = -1; dx <= 1; dx++) {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nx = ix + dx; let ny = iy + dy; let nz = iz + dz;
                    if nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires {
                        let ni = grid_index(u32(nx), u32(ny), u32(nz));
                        if density_in_0[ni] != 0.0 || density_in_1[ni] != 0.0 ||
                           density_in_2[ni] != 0.0 || density_in_3[ni] != 0.0 {
                            any_neighbor = true;
                            break;
                        }
                    }
                }
            }
        }
        if !any_neighbor {
            density_out_0[idx] = 0.0;
            density_out_1[idx] = 0.0;
            density_out_2[idx] = 0.0;
            density_out_3[idx] = 0.0;
            org_out_0[idx] = 0u;
            org_out_1[idx] = 0u;
            org_out_2[idx] = 0u;
            org_out_3[idx] = 0u;
            return;
        }
    }

    // 3x3x3 box blur each slot independently as a plain scalar field.
    var sum0 = 0.0; var sum1 = 0.0; var sum2 = 0.0; var sum3 = 0.0;
    var count = 0.0;
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nx = ix + dx; let ny = iy + dy; let nz = iz + dz;
                if nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires {
                    let ni = grid_index(u32(nx), u32(ny), u32(nz));
                    sum0 += density_in_0[ni];
                    sum1 += density_in_1[ni];
                    sum2 += density_in_2[ni];
                    sum3 += density_in_3[ni];
                    count += 1.0;
                }
            }
        }
    }

    let d0 = sum0 / count;
    let d1 = sum1 / count;
    let d2 = sum2 / count;
    let d3 = sum3 / count;

    density_out_0[idx] = d0;
    density_out_1[idx] = d1;
    density_out_2[idx] = d2;
    density_out_3[idx] = d3;

    // Propagate org IDs: keep center org if present, otherwise find from neighbors.
    let o0 = org_in_0[idx];
    let o1 = org_in_1[idx];
    let o2 = org_in_2[idx];
    let o3 = org_in_3[idx];

    // Slot 0
    if d0 > 0.0 && o0 == 0u {
        var found = 0u;
        for (var dz: i32 = -1; dz <= 1; dz++) {
            if found != 0u { break; }
            for (var dy: i32 = -1; dy <= 1; dy++) {
                if found != 0u { break; }
                for (var dx: i32 = -1; dx <= 1; dx++) {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nx = ix + dx; let ny = iy + dy; let nz = iz + dz;
                    if nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires {
                        let v = org_in_0[grid_index(u32(nx), u32(ny), u32(nz))];
                        if v != 0u { found = v; break; }
                    }
                }
            }
        }
        org_out_0[idx] = found;
    } else {
        org_out_0[idx] = o0;
    }

    // Slot 1
    if d1 > 0.0 && o1 == 0u {
        var found = 0u;
        for (var dz: i32 = -1; dz <= 1; dz++) {
            if found != 0u { break; }
            for (var dy: i32 = -1; dy <= 1; dy++) {
                if found != 0u { break; }
                for (var dx: i32 = -1; dx <= 1; dx++) {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nx = ix + dx; let ny = iy + dy; let nz = iz + dz;
                    if nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires {
                        let v = org_in_1[grid_index(u32(nx), u32(ny), u32(nz))];
                        if v != 0u { found = v; break; }
                    }
                }
            }
        }
        org_out_1[idx] = found;
    } else {
        org_out_1[idx] = o1;
    }

    // Slot 2
    if d2 > 0.0 && o2 == 0u {
        var found = 0u;
        for (var dz: i32 = -1; dz <= 1; dz++) {
            if found != 0u { break; }
            for (var dy: i32 = -1; dy <= 1; dy++) {
                if found != 0u { break; }
                for (var dx: i32 = -1; dx <= 1; dx++) {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nx = ix + dx; let ny = iy + dy; let nz = iz + dz;
                    if nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires {
                        let v = org_in_2[grid_index(u32(nx), u32(ny), u32(nz))];
                        if v != 0u { found = v; break; }
                    }
                }
            }
        }
        org_out_2[idx] = found;
    } else {
        org_out_2[idx] = o2;
    }

    // Slot 3
    if d3 > 0.0 && o3 == 0u {
        var found = 0u;
        for (var dz: i32 = -1; dz <= 1; dz++) {
            if found != 0u { break; }
            for (var dy: i32 = -1; dy <= 1; dy++) {
                if found != 0u { break; }
                for (var dx: i32 = -1; dx <= 1; dx++) {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let nx = ix + dx; let ny = iy + dy; let nz = iz + dz;
                    if nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires {
                        let v = org_in_3[grid_index(u32(nx), u32(ny), u32(nz))];
                        if v != 0u { found = v; break; }
                    }
                }
            }
        }
        org_out_3[idx] = found;
    } else {
        org_out_3[idx] = o3;
    }
}
