// Organism Shrink-Wrap Skin
//
// Icosphere mesh per organism that contracts onto cell surfaces each frame.
// Uses 5 passes:
//   1. clear_org_state   - zero accumulators in org_state
//   2. accumulate_cells  - one thread per cell, atomically adds to org centroid
//   3. finalize_orgs     - divide centroid by count, place icosphere vertices
//   4. shrink_step       - move each vertex toward nearest cell surface
//   5. smooth_step       - Laplacian smooth + recompute normals
//
// Icosphere (3 subdivisions): 642 vertices, 1280 triangles.
// Vertex positions are stored in ico_unit_positions buffer (uploaded from CPU).

const VERTS_PER_ORG:      u32 = 642u;
const TRIS_PER_ORG:       u32 = 1280u;
const MAX_ORGANISMS:      u32 = 512u;
const DEAD_LABEL:         u32 = 0xFFFFFFFFu;
const MAX_CELLS_PER_GRID: u32 = 16u;
const FIXED_SCALE:        f32 = 1024.0; // fixed-point scale for centroid accumulation

struct ShrinkParams {
    world_size:      f32,
    grid_cell_size:  f32,
    grid_resolution: i32,
    cell_count:      u32,
    skin_offset:     f32,   // gap between skin and cell surface (world units)
    shrink_speed:    f32,   // fraction of gap to close per step (0.1-0.3)
    smooth_factor:   f32,   // Laplacian blend weight (0.0-0.5)
    min_cells:       u32,   // minimum cells for an organism to get a skin
}

struct SkinVertex {
    position:    vec3<f32>,
    organism_id: f32,
    normal:      vec3<f32>,
    _pad:        f32,
}

// Accumulator for centroid (fixed-point i32 x FIXED_SCALE) + bounding radius (f32 via atomic)
// Layout: [sum_x_fixed: i32, sum_y_fixed: i32, sum_z_fixed: i32, cell_count: u32,
//          max_dist_sq_fixed: i32 (xFIXED_SCALE), is_used: u32, _pad0: u32, _pad1: u32]
// Total: 32 bytes
struct OrgAccum {
    sum_x:       atomic<i32>,
    sum_y:       atomic<i32>,
    sum_z:       atomic<i32>,
    cell_count:  atomic<u32>,
    max_dist_sq: atomic<i32>,  // fixed-point xFIXED_SCALE
    is_used:     u32,
    _pad0:       u32,
    _pad1:       u32,
}

struct OrgState {
    centroid:   vec3<f32>,
    radius:     f32,
    skin_id:    u32,
    cell_count: u32,
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0)  var<uniform>             params:              ShrinkParams;
@group(0) @binding(1)  var<storage, read>       position_and_mass:   array<vec4<f32>>;
@group(0) @binding(2)  var<storage, read>       death_flags:         array<u32>;
@group(0) @binding(3)  var<storage, read>       cell_count_buf:      array<u32>;
@group(0) @binding(4)  var<storage, read>       label_buffer:        array<u32>;
@group(0) @binding(5)  var<storage, read>       spatial_grid_counts: array<u32>;
@group(0) @binding(6)  var<storage, read>       spatial_grid_cells:  array<u32>;
@group(0) @binding(7)  var<storage, read_write> org_accum:           array<OrgAccum>;
@group(0) @binding(8)  var<storage, read_write> org_state:           array<OrgState>;
@group(0) @binding(9)  var<storage, read_write> vertices:            array<SkinVertex>;
// Stable organism ID per cell - use this instead of raw label_buffer.
// Assigned by organism_stable_id.wgsl; persists across label changes.
// 0 = no skin, 1-512 = organism slot index + 1.
@group(0) @binding(10) var<storage, read>       stable_id_per_cell:  array<u32>;
// Unit-sphere icosphere vertex positions (162 verts, uploaded from CPU).
// Used by finalize_orgs to place the initial sphere.
@group(0) @binding(11) var<storage, read>       ico_unit_positions:  array<vec4<f32>>;

// -- Helpers -------------------------------------------------------------------

// stable_id_per_cell[ci] is already the skin slot in [1, MAX_ORGANISMS].
// 0 means no skin. This replaces the old label_to_skin_id hash.
fn cell_skin_id(ci: u32) -> u32 {
    return stable_id_per_cell[ci];
}

fn grid_idx_from_pos(pos: vec3<f32>) -> u32 {
    let half = params.world_size * 0.5;
    let gp   = (pos + half) / params.grid_cell_size;
    let res  = params.grid_resolution;
    let gx   = clamp(i32(gp.x), 0, res - 1);
    let gy   = clamp(i32(gp.y), 0, res - 1);
    let gz   = clamp(i32(gp.z), 0, res - 1);
    return u32(gx + gy * res + gz * res * res);
}

// Query spatial grid for nearest cell belonging to skin_id.
// Returns vec4(cell_pos, cell_radius) or vec4(0,0,0,0) if not found.
fn nearest_cell(pos: vec3<f32>, skin_id: u32) -> vec4<f32> {
    let half = params.world_size * 0.5;
    let gp   = (pos + half) / params.grid_cell_size;
    let res  = params.grid_resolution;
    let bx   = i32(gp.x);
    let by   = i32(gp.y);
    let bz   = i32(gp.z);

    var best_dist_sq = 1e30;
    var best_pos     = vec3<f32>(0.0);
    var best_radius  = 0.0;

    for (var dz: i32 = -2; dz <= 2; dz++) {
        for (var dy: i32 = -2; dy <= 2; dy++) {
            for (var dx: i32 = -2; dx <= 2; dx++) {
                let gx = bx + dx; let gy = by + dy; let gz = bz + dz;
                if gx < 0 || gx >= res || gy < 0 || gy >= res || gz < 0 || gz >= res { continue; }
                let gi    = u32(gx + gy * res + gz * res * res);
                let count = min(spatial_grid_counts[gi], MAX_CELLS_PER_GRID);
                for (var s: u32 = 0u; s < count; s++) {
                    let ci = spatial_grid_cells[gi * MAX_CELLS_PER_GRID + s];
                    if death_flags[ci] != 0u { continue; }
                    if cell_skin_id(ci) != skin_id { continue; }
                    let pm      = position_and_mass[ci];
                    let d       = pm.xyz - pos;
                    let dist_sq = dot(d, d);
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_pos     = pm.xyz;
                        best_radius  = clamp(pm.w, 0.5, 2.0);
                    }
                }
            }
        }
    }
    return vec4<f32>(best_pos, best_radius);
}

// -- Icosphere vertex positions - read from buffer (162 verts, 2 subdivisions) -
// ico_unit_positions[v].xyz is the unit-sphere position for vertex v.
// Uploaded from CPU by build_icosphere(2) in organism_skin.rs.
fn ico_pos(v: u32) -> vec3<f32> {
    return ico_unit_positions[v].xyz;
}

// -- Pass 1: clear_org_state ---------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn clear_org_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= MAX_ORGANISMS { return; }
    atomicStore(&org_accum[i].sum_x,       0);
    atomicStore(&org_accum[i].sum_y,       0);
    atomicStore(&org_accum[i].sum_z,       0);
    atomicStore(&org_accum[i].cell_count,  0u);
    atomicStore(&org_accum[i].max_dist_sq, 0);
    org_accum[i].is_used = 0u;
    org_state[i].skin_id    = 0u;
    org_state[i].cell_count = 0u;
}

// -- Pass 2: accumulate_cells --------------------------------------------------
// One thread per cell. Adds cell position to its organism's accumulator.
@compute @workgroup_size(256, 1, 1)
fn accumulate_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    if ci >= cell_count_buf[0] { return; }
    if death_flags[ci] != 0u { return; }

    let skin_id = stable_id_per_cell[ci];
    if skin_id == 0u { return; }
    // slot is 0-based index into org_accum/org_state arrays
    let slot = skin_id - 1u;

    let pm = position_and_mass[ci];
    atomicAdd(&org_accum[slot].sum_x,      i32(pm.x * FIXED_SCALE));
    atomicAdd(&org_accum[slot].sum_y,      i32(pm.y * FIXED_SCALE));
    atomicAdd(&org_accum[slot].sum_z,      i32(pm.z * FIXED_SCALE));
    atomicAdd(&org_accum[slot].cell_count, 1u);
}

// -- Pass 3: finalize_orgs -----------------------------------------------------
// One thread per organism slot. Computes centroid, then scans cells again
// to find bounding radius, then places icosphere.
// NOTE: bounding radius scan is done per-org by re-reading the cell list.
// This is O(cells) per org but only runs for active orgs and is fast in practice.
@compute @workgroup_size(64, 1, 1)
fn finalize_orgs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if slot >= MAX_ORGANISMS { return; }

    let cnt = atomicLoad(&org_accum[slot].cell_count);
    if cnt < params.min_cells {
        org_state[slot].skin_id    = 0u;
        org_state[slot].cell_count = 0u;
        // Zero vertices
        let base = slot * VERTS_PER_ORG;
        for (var v = 0u; v < VERTS_PER_ORG; v++) {
            vertices[base + v].organism_id = 0.0;
            vertices[base + v].position    = vec3<f32>(0.0);
        }
        return;
    }

    let inv_cnt  = 1.0 / f32(cnt);
    let centroid = vec3<f32>(
        f32(atomicLoad(&org_accum[slot].sum_x)) / FIXED_SCALE * inv_cnt,
        f32(atomicLoad(&org_accum[slot].sum_y)) / FIXED_SCALE * inv_cnt,
        f32(atomicLoad(&org_accum[slot].sum_z)) / FIXED_SCALE * inv_cnt,
    );

    // Compute bounding radius by scanning cells
    let skin_id      = slot + 1u;
    let total_cells  = cell_count_buf[0];
    var max_dist_sq  = 0.0;
    for (var ci = 0u; ci < total_cells; ci++) {
        if death_flags[ci] != 0u { continue; }
        if stable_id_per_cell[ci] != skin_id { continue; }
        let pm      = position_and_mass[ci];
        let r       = clamp(pm.w, 0.5, 2.0);
        let d       = pm.xyz - centroid;
        let dist_sq = dot(d, d) + r * r;
        if dist_sq > max_dist_sq { max_dist_sq = dist_sq; }
    }

    let bounding_r = sqrt(max_dist_sq) + params.skin_offset + 0.5;

    org_state[slot].centroid   = centroid;
    org_state[slot].radius     = bounding_r;
    org_state[slot].skin_id    = skin_id;
    org_state[slot].cell_count = cnt;

    // Place icosphere vertices on bounding sphere
    let base = slot * VERTS_PER_ORG;
    for (var v = 0u; v < VERTS_PER_ORG; v++) {
        let dir = ico_pos(v);
        vertices[base + v].position    = centroid + dir * bounding_r;
        vertices[base + v].organism_id = f32(skin_id);
        vertices[base + v].normal      = dir;
    }
}

// -- Pass 4: shrink_step -------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn shrink_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vert_idx = gid.x;
    if vert_idx >= MAX_ORGANISMS * VERTS_PER_ORG { return; }

    let slot  = vert_idx / VERTS_PER_ORG;
    let state = org_state[slot];
    if state.skin_id == 0u { return; }

    var pos    = vertices[vert_idx].position;
    let result = nearest_cell(pos, state.skin_id);
    let cell_r = result.w;

    if cell_r > 0.0 {
        let to_cell  = result.xyz - pos;
        let dist     = length(to_cell);
        let target_d = cell_r + params.skin_offset;
        if dist > target_d + 0.001 {
            pos += normalize(to_cell) * (dist - target_d) * params.shrink_speed;
        }
    } else {
        // No nearby cell - drift toward centroid
        let to_c = state.centroid - pos;
        let d    = length(to_c);
        if d > 0.001 { pos += normalize(to_c) * d * 0.05; }
    }

    vertices[vert_idx].position = pos;
}

// -- Icosphere adjacency (42 vertices, each has 5 or 6 neighbours) ------------
// -- Pass 5: smooth_step -------------------------------------------------------
// Laplacian smooth using the 6 nearest neighbours by unit-sphere angular distance.
// This works for any icosphere resolution without a hardcoded adjacency table.
@compute @workgroup_size(64, 1, 1)
fn smooth_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vert_idx = gid.x;
    if vert_idx >= MAX_ORGANISMS * VERTS_PER_ORG { return; }

    let slot    = vert_idx / VERTS_PER_ORG;
    let local_v = vert_idx % VERTS_PER_ORG;
    let state   = org_state[slot];
    if state.skin_id == 0u { return; }

    let base     = slot * VERTS_PER_ORG;
    let pos      = vertices[vert_idx].position;
    let unit_dir = ico_unit_positions[local_v].xyz; // unit-sphere direction for this vertex

    // Find the 6 nearest neighbours by dot product on the unit sphere.
    // Neighbours are vertices whose unit directions are closest to ours.
    // We scan all VERTS_PER_ORG vertices - expensive but correct for any resolution.
    // With 162 verts this is 162 comparisons per thread, which is fast.
    var top6_dot: array<f32, 6>;
    var top6_idx: array<u32, 6>;
    for (var k = 0u; k < 6u; k++) { top6_dot[k] = -2.0; top6_idx[k] = local_v; }

    for (var j = 0u; j < VERTS_PER_ORG; j++) {
        if j == local_v { continue; }
        let d = dot(unit_dir, ico_unit_positions[j].xyz);
        // Insert into top-6 if larger than current minimum
        var min_k = 0u;
        for (var k = 1u; k < 6u; k++) {
            if top6_dot[k] < top6_dot[min_k] { min_k = k; }
        }
        if d > top6_dot[min_k] {
            top6_dot[min_k] = d;
            top6_idx[min_k] = j;
        }
    }

    var avg = vec3<f32>(0.0);
    for (var k = 0u; k < 6u; k++) {
        avg += vertices[base + top6_idx[k]].position;
    }
    avg /= 6.0;

    var new_pos = mix(pos, avg, params.smooth_factor);

    // Re-project: don't let smooth push vertex inside a cell
    let result = nearest_cell(new_pos, state.skin_id);
    if result.w > 0.0 {
        let to_cell  = result.xyz - new_pos;
        let dist     = length(to_cell);
        let min_dist = result.w + params.skin_offset;
        if dist < min_dist && dist > 0.001 {
            new_pos = result.xyz - normalize(to_cell) * min_dist;
        }
    }

    // Outward normal from centroid
    let from_c = new_pos - state.centroid;
    let len    = length(from_c);
    let normal = select(vec3<f32>(0.0, 1.0, 0.0), from_c / len, len > 0.001);

    vertices[vert_idx].position = new_pos;
    vertices[vert_idx].normal   = normal;
}
