// Organism Skin Density - K=4 Overlapping Slots
//
// Each voxel has four slots. Each slot holds an organism_id (u32) and a
// fixed-point density accumulator (atomic i32). Cells claim a slot via CAS
// and accumulate density via atomicAdd. Four organisms can overlap at any
// voxel; a fifth is dropped.
//
// Three entry points (same bind group):
//   clear_density      - zero all slot data
//   generate_density   - per-cell metaball splatting into slots
//   normalize_density  - convert fixed-point i32 -> f32 density per slot

struct OrganismDensityParams {
    grid_origin: vec3<f32>,
    cell_size: f32,
    grid_resolution: u32,
    skin_radius_scale: f32,
    max_cells: u32,
    min_cells_for_skin: u32,
}

@group(0) @binding(0) var<uniform>            params: OrganismDensityParams;
@group(0) @binding(1) var<storage, read>      position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>      death_flags: array<u32>;
@group(0) @binding(3) var<storage, read>      cell_count: array<u32>;

// K=4 slot buffers - organism IDs (atomic u32)
@group(0) @binding(4)  var<storage, read_write> slot_org_0: array<atomic<u32>>;
@group(0) @binding(5)  var<storage, read_write> slot_org_1: array<atomic<u32>>;
@group(0) @binding(6)  var<storage, read_write> slot_org_2: array<atomic<u32>>;
@group(0) @binding(7)  var<storage, read_write> slot_org_3: array<atomic<u32>>;

// K=4 slot buffers - fixed-point density accumulators (atomic i32)
@group(0) @binding(8)  var<storage, read_write> slot_density_0: array<atomic<i32>>;
@group(0) @binding(9)  var<storage, read_write> slot_density_1: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> slot_density_2: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> slot_density_3: array<atomic<i32>>;

// K=4 normalized f32 density output (written by normalize pass, read by surface nets)
@group(0) @binding(12) var<storage, read_write> density_out_0: array<f32>;
@group(0) @binding(13) var<storage, read_write> density_out_1: array<f32>;
@group(0) @binding(14) var<storage, read_write> density_out_2: array<f32>;
@group(0) @binding(15) var<storage, read_write> density_out_3: array<f32>;

// K=4 organism ID output (non-atomic copy for surface nets to read)
@group(0) @binding(16) var<storage, read_write> org_id_out_0: array<u32>;
@group(0) @binding(17) var<storage, read_write> org_id_out_1: array<u32>;
@group(0) @binding(18) var<storage, read_write> org_id_out_2: array<u32>;
@group(0) @binding(19) var<storage, read_write> org_id_out_3: array<u32>;

// Per-cell organism skin ID (0 = no skin, >0 = 16-bit organism ID)
@group(0) @binding(20) var<storage, read>      cell_skin_id: array<u32>;

const FIXED_SCALE: f32 = 32768.0;
const MAX_VOXEL_RADIUS: i32 = 8;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    let r = params.grid_resolution;
    return x + y * r + z * r * r;
}

// Wyvill metaball kernel: f(r) = (1 - r^2/R^2)^3
// Peaks at 1.0 at the cell centre, falls to 0.0 at r = R.
// Much sharper than smoothstep near the surface - the iso level sits
// close to the cell boundary for both isolated cells and dense clusters.
// r2_over_R2 = (dist/radius)^2  in [0, 1]
fn metaball_kernel(r2_over_R2: f32) -> f32 {
    let x = 1.0 - r2_over_R2;
    return x * x * x;
}

// Slot claiming is inlined in generate_density because WGSL (naga) does not
// allow passing storage pointers to functions.

// -----------------------------------------------------------------------------
// Pass 1: clear all slot data
// -----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn clear_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    if gid.x >= total { return; }

    atomicStore(&slot_org_0[gid.x], 0u);
    atomicStore(&slot_org_1[gid.x], 0u);
    atomicStore(&slot_org_2[gid.x], 0u);
    atomicStore(&slot_org_3[gid.x], 0u);
    atomicStore(&slot_density_0[gid.x], 0);
    atomicStore(&slot_density_1[gid.x], 0);
    atomicStore(&slot_density_2[gid.x], 0);
    atomicStore(&slot_density_3[gid.x], 0);
}

// -----------------------------------------------------------------------------
// Pass 2: per-cell metaball splatting with K=4 slot claiming
// -----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn generate_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    let total_cells = cell_count[0];
    if cell_idx >= total_cells { return; }
    if death_flags[cell_idx] != 0u { return; }

    let pos_mass = position_and_mass[cell_idx];
    let pos      = pos_mass.xyz;
    let mass     = pos_mass.w;
    if mass <= 0.0 { return; }

    let skin_id = cell_skin_id[cell_idx];
    if skin_id == 0u { return; }

    // skin_radius_scale is now the tight skin offset in world units added on top
    // of the cell radius.  A value of 1.2-1.5 gives a snug skin; 2.0 gives a
    // slightly looser one.  The Wyvill kernel peaks at 1.0 at the cell centre
    // and reaches 0.0 at influence_radius, so the iso level of ~0.5 sits
    // roughly at the cell surface for any cluster density.
    let cell_radius      = clamp(mass, 0.5, 2.0);
    let influence_radius = cell_radius * params.skin_radius_scale;
    let cell_size        = params.cell_size;
    let grid_origin      = params.grid_origin;
    let res              = i32(params.grid_resolution);

    let local = pos - grid_origin;
    let cx    = i32(local.x / cell_size);
    let cy    = i32(local.y / cell_size);
    let cz    = i32(local.z / cell_size);

    let vr = min(MAX_VOXEL_RADIUS, i32(influence_radius / cell_size) + 1);
    let vmin = max(vec3<i32>(0),       vec3<i32>(cx - vr, cy - vr, cz - vr));
    let vmax = min(vec3<i32>(res - 1), vec3<i32>(cx + vr, cy + vr, cz + vr));

    let radius_sq     = influence_radius * influence_radius;
    let inv_radius_sq = 1.0 / radius_sq;

    for (var vz = vmin.z; vz <= vmax.z; vz++) {
        let wz = grid_origin.z + f32(vz) * cell_size - pos.z;
        let wz2 = wz * wz;
        if wz2 >= radius_sq { continue; }

        for (var vy = vmin.y; vy <= vmax.y; vy++) {
            let wy = grid_origin.y + f32(vy) * cell_size - pos.y;
            let wyz2 = wz2 + wy * wy;
            if wyz2 >= radius_sq { continue; }

            for (var vx = vmin.x; vx <= vmax.x; vx++) {
                let wx = grid_origin.x + f32(vx) * cell_size - pos.x;
                let dist_sq = wyz2 + wx * wx;
                if dist_sq >= radius_sq { continue; }

                // Wyvill kernel: (1 - dist^2/R^2)^3  - peaks at 1.0 at centre,
                // 0.0 at the influence boundary.  No sqrt needed.
                let contrib = metaball_kernel(dist_sq * inv_radius_sq);
                let fixed   = i32(contrib * FIXED_SCALE);
                if fixed <= 0 { continue; }

                let idx = voxel_index(u32(vx), u32(vy), u32(vz));

                // Slot 0
                var claimed = false;
                {
                    let org0 = atomicLoad(&slot_org_0[idx]);
                    if org0 == skin_id {
                        atomicAdd(&slot_density_0[idx], fixed);
                        claimed = true;
                    } else if org0 == 0u {
                        let cas0 = atomicCompareExchangeWeak(&slot_org_0[idx], 0u, skin_id);
                        if cas0.exchanged || cas0.old_value == skin_id {
                            atomicAdd(&slot_density_0[idx], fixed);
                            claimed = true;
                        }
                    }
                }
                if claimed { continue; }

                // Slot 1
                {
                    let org1 = atomicLoad(&slot_org_1[idx]);
                    if org1 == skin_id {
                        atomicAdd(&slot_density_1[idx], fixed);
                        claimed = true;
                    } else if org1 == 0u {
                        let cas1 = atomicCompareExchangeWeak(&slot_org_1[idx], 0u, skin_id);
                        if cas1.exchanged || cas1.old_value == skin_id {
                            atomicAdd(&slot_density_1[idx], fixed);
                            claimed = true;
                        }
                    }
                }
                if claimed { continue; }

                // Slot 2
                {
                    let org2 = atomicLoad(&slot_org_2[idx]);
                    if org2 == skin_id {
                        atomicAdd(&slot_density_2[idx], fixed);
                        claimed = true;
                    } else if org2 == 0u {
                        let cas2 = atomicCompareExchangeWeak(&slot_org_2[idx], 0u, skin_id);
                        if cas2.exchanged || cas2.old_value == skin_id {
                            atomicAdd(&slot_density_2[idx], fixed);
                            claimed = true;
                        }
                    }
                }
                if claimed { continue; }

                // Slot 3
                {
                    let org3 = atomicLoad(&slot_org_3[idx]);
                    if org3 == skin_id {
                        atomicAdd(&slot_density_3[idx], fixed);
                        claimed = true;
                    } else if org3 == 0u {
                        let cas3 = atomicCompareExchangeWeak(&slot_org_3[idx], 0u, skin_id);
                        if cas3.exchanged || cas3.old_value == skin_id {
                            atomicAdd(&slot_density_3[idx], fixed);
                            claimed = true;
                        }
                    }
                }
                // All 4 slots taken by other organisms - drop this contribution.
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Pass 3: normalize fixed-point -> f32 and copy org IDs for surface nets
// -----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn normalize_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    if gid.x >= total { return; }

    let idx = gid.x;

    // Early-out: check if any slot has data before doing expensive atomic loads on all 8 accumulators.
    // The org ID slots are only non-zero if a cell claimed them during generate_density.
    let org0 = atomicLoad(&slot_org_0[idx]);
    let org1 = atomicLoad(&slot_org_1[idx]);
    let org2 = atomicLoad(&slot_org_2[idx]);
    let org3 = atomicLoad(&slot_org_3[idx]);

    if (org0 | org1 | org2 | org3) == 0u {
        // All slots empty - write zeros and skip
        density_out_0[idx] = 0.0;
        density_out_1[idx] = 0.0;
        density_out_2[idx] = 0.0;
        density_out_3[idx] = 0.0;
        org_id_out_0[idx] = 0u;
        org_id_out_1[idx] = 0u;
        org_id_out_2[idx] = 0u;
        org_id_out_3[idx] = 0u;
        return;
    }

    if org0 != 0u {
        density_out_0[idx] = f32(atomicLoad(&slot_density_0[idx])) / FIXED_SCALE;
        org_id_out_0[idx] = org0;
    } else {
        density_out_0[idx] = 0.0;
        org_id_out_0[idx] = 0u;
    }

    if org1 != 0u {
        density_out_1[idx] = f32(atomicLoad(&slot_density_1[idx])) / FIXED_SCALE;
        org_id_out_1[idx] = org1;
    } else {
        density_out_1[idx] = 0.0;
        org_id_out_1[idx] = 0u;
    }

    if org2 != 0u {
        density_out_2[idx] = f32(atomicLoad(&slot_density_2[idx])) / FIXED_SCALE;
        org_id_out_2[idx] = org2;
    } else {
        density_out_2[idx] = 0.0;
        org_id_out_2[idx] = 0u;
    }

    if org3 != 0u {
        density_out_3[idx] = f32(atomicLoad(&slot_density_3[idx])) / FIXED_SCALE;
        org_id_out_3[idx] = org3;
    } else {
        density_out_3[idx] = 0.0;
        org_id_out_3[idx] = 0u;
    }
}
