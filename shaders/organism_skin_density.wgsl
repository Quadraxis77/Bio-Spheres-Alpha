// Organism Skin Density — K=2 Overlapping Slots
//
// Each voxel has two slots. Each slot holds an organism_id (u32) and a
// fixed-point density accumulator (atomic i32). Cells claim a slot via CAS
// and accumulate density via atomicAdd. Two organisms can overlap at any
// voxel; a third is dropped.
//
// Three entry points (same bind group):
//   clear_density      — zero all slot data
//   generate_density   — per-cell metaball splatting into slots
//   normalize_density  — convert fixed-point i32 → f32 density per slot

struct OrganismDensityParams {
    grid_origin: vec3<f32>,
    cell_size: f32,
    grid_resolution: u32,
    skin_radius_scale: f32,
    max_cells: u32,
    min_cells_for_skin: u32,  // organisms with fewer cells get no skin (typically 4)
}

@group(0) @binding(0) var<uniform>            params: OrganismDensityParams;
@group(0) @binding(1) var<storage, read>      position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>      death_flags: array<u32>;
@group(0) @binding(3) var<storage, read>      cell_count: array<u32>;

// K=2 slot buffers — organism IDs
@group(0) @binding(4) var<storage, read_write> slot_org_0: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> slot_org_1: array<atomic<u32>>;

// K=2 slot buffers — fixed-point density accumulators
@group(0) @binding(6) var<storage, read_write> slot_density_0: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> slot_density_1: array<atomic<i32>>;

// Normalized f32 density output (written by normalize pass, read by surface nets)
@group(0) @binding(8) var<storage, read_write> density_out_0: array<f32>;
@group(0) @binding(9) var<storage, read_write> density_out_1: array<f32>;

// Organism ID output (non-atomic copy for surface nets to read)
@group(0) @binding(10) var<storage, read_write> org_id_out_0: array<u32>;
@group(0) @binding(11) var<storage, read_write> org_id_out_1: array<u32>;

// Per-cell organism skin ID (0 = no skin, >0 = 16-bit organism ID)
// Written by the organism_skin_count pass.
@group(0) @binding(12) var<storage, read>      cell_skin_id: array<u32>;

const FIXED_SCALE: f32 = 32768.0;
const MAX_VOXEL_RADIUS: i32 = 28;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    let r = params.grid_resolution;
    return x + y * r + z * r * r;
}

fn metaball_kernel(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: clear all slot data
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn clear_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    if gid.x >= total { return; }

    atomicStore(&slot_org_0[gid.x], 0u);
    atomicStore(&slot_org_1[gid.x], 0u);
    atomicStore(&slot_density_0[gid.x], 0);
    atomicStore(&slot_density_1[gid.x], 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: per-cell metaball splatting with K=2 slot claiming
// ─────────────────────────────────────────────────────────────────────────────
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

    // Read pre-computed skin ID (0 = no skin for this cell)
    let skin_id = cell_skin_id[cell_idx];
    if skin_id == 0u { return; }

    let influence_radius = clamp(mass, 0.5, 2.0) * params.skin_radius_scale;
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

    let inv_radius = 1.0 / influence_radius;

    for (var vz = vmin.z; vz <= vmax.z; vz++) {
        for (var vy = vmin.y; vy <= vmax.y; vy++) {
            for (var vx = vmin.x; vx <= vmax.x; vx++) {
                let voxel_ws = grid_origin
                    + vec3<f32>(f32(vx), f32(vy), f32(vz)) * cell_size;

                let dist = length(voxel_ws - pos);
                if dist >= influence_radius { continue; }

                let t       = 1.0 - dist * inv_radius;
                let contrib = metaball_kernel(t);
                let fixed   = i32(contrib * FIXED_SCALE);
                if fixed <= 0 { continue; }

                let idx = voxel_index(u32(vx), u32(vy), u32(vz));

                // Try to claim or accumulate into slot 0
                let org0 = atomicLoad(&slot_org_0[idx]);
                if org0 == skin_id {
                    // Our organism already owns slot 0 — accumulate
                    atomicAdd(&slot_density_0[idx], fixed);
                    continue;
                }
                if org0 == 0u {
                    // Slot 0 is empty — try to claim it
                    let cas = atomicCompareExchangeWeak(&slot_org_0[idx], 0u, skin_id);
                    if cas.exchanged || cas.old_value == skin_id {
                        atomicAdd(&slot_density_0[idx], fixed);
                        continue;
                    }
                    // CAS failed — another organism claimed slot 0 between our load and CAS.
                    // Fall through to try slot 1.
                }

                // Try to claim or accumulate into slot 1
                let org1 = atomicLoad(&slot_org_1[idx]);
                if org1 == skin_id {
                    atomicAdd(&slot_density_1[idx], fixed);
                    continue;
                }
                if org1 == 0u {
                    let cas = atomicCompareExchangeWeak(&slot_org_1[idx], 0u, skin_id);
                    if cas.exchanged || cas.old_value == skin_id {
                        atomicAdd(&slot_density_1[idx], fixed);
                        continue;
                    }
                }

                // Both slots taken by other organisms — drop this contribution.
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3: normalize fixed-point → f32 and copy org IDs for surface nets
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(4, 4, 4)
fn normalize_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    let idx = voxel_index(gid.x, gid.y, gid.z);

    // Slot 0
    let raw0 = atomicLoad(&slot_density_0[idx]);
    density_out_0[idx] = f32(raw0) / FIXED_SCALE;
    org_id_out_0[idx] = atomicLoad(&slot_org_0[idx]);

    // Slot 1
    let raw1 = atomicLoad(&slot_density_1[idx]);
    density_out_1[idx] = f32(raw1) / FIXED_SCALE;
    org_id_out_1[idx] = atomicLoad(&slot_org_1[idx]);
}
