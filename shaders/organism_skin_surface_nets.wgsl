// Organism Skin Surface Nets — K=4 overlapping organism extraction
//
// For each voxel cell, finds up to 4 unique organism IDs across the 8 corners
// and generates an independent isosurface vertex for each organism present.
//
// Uses 4 vertex maps (one per organism slot) so vertex indices are full 32-bit.

struct SurfaceNetsParams {
    grid_resolution: u32,
    iso_level: f32,
    cell_size: f32,
    max_vertices: u32,

    grid_origin: vec3<f32>,
    max_indices: u32,

    density_resolution: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
}

struct Vertex {
    position: vec3<f32>,
    organism_id: f32,
    normal: vec3<f32>,
    _pad1: f32,
}

struct Counter {
    vertex_count: atomic<u32>,
    index_count: atomic<u32>,
}

struct IndirectDraw {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: SurfaceNetsParams;

// K=4 density slots
@group(0) @binding(1)  var<storage, read> density_0: array<f32>;
@group(0) @binding(2)  var<storage, read> density_1: array<f32>;
@group(0) @binding(3)  var<storage, read> density_2: array<f32>;
@group(0) @binding(4)  var<storage, read> density_3: array<f32>;

// K=4 organism ID slots
@group(0) @binding(5)  var<storage, read> org_id_0: array<u32>;
@group(0) @binding(6)  var<storage, read> org_id_1: array<u32>;
@group(0) @binding(7)  var<storage, read> org_id_2: array<u32>;
@group(0) @binding(8)  var<storage, read> org_id_3: array<u32>;

// Output mesh
@group(0) @binding(9)  var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(10) var<storage, read_write> indices: array<u32>;

// 4 vertex maps — one per organism slot at each voxel cell
@group(0) @binding(11) var<storage, read_write> vertex_map_0: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> vertex_map_1: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> vertex_map_2: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> vertex_map_3: array<atomic<u32>>;

@group(0) @binding(15) var<storage, read_write> counters: Counter;
@group(0) @binding(16) var<storage, read_write> indirect_draw: IndirectDraw;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn grid_to_world(x: f32, y: f32, z: f32) -> vec3<f32> {
    return params.grid_origin + vec3<f32>(x, y, z) * params.cell_size;
}

fn density_index(dx: u32, dy: u32, dz: u32) -> u32 {
    let dres = params.density_resolution;
    return dx + dy * dres + dz * dres * dres;
}

// Sample density for a specific organism at a padded-grid coordinate.
fn sample_organism_density(x: i32, y: i32, z: i32, target_org: u32) -> f32 {
    let dres = i32(params.density_resolution);
    let dx = x - 1; let dy = y - 1; let dz = z - 1;
    if dx < 0 || dx >= dres || dy < 0 || dy >= dres || dz < 0 || dz >= dres {
        return 0.0;
    }
    let idx = density_index(u32(dx), u32(dy), u32(dz));
    if org_id_0[idx] == target_org { return density_0[idx]; }
    if org_id_1[idx] == target_org { return density_1[idx]; }
    if org_id_2[idx] == target_org { return density_2[idx]; }
    if org_id_3[idx] == target_org { return density_3[idx]; }
    return 0.0;
}

// Check if any density exists at a padded-grid voxel (any slot)
fn has_any_density_at(x: i32, y: i32, z: i32) -> bool {
    let dres = i32(params.density_resolution);
    let dx = x - 1; let dy = y - 1; let dz = z - 1;
    if dx < 0 || dx >= dres || dy < 0 || dy >= dres || dz < 0 || dz >= dres { return false; }
    let idx = density_index(u32(dx), u32(dy), u32(dz));
    return density_0[idx] > 0.0 || density_1[idx] > 0.0
        || density_2[idx] > 0.0 || density_3[idx] > 0.0;
}

// Collect unique organism IDs at a density voxel (up to 4)
fn collect_orgs_at(x: i32, y: i32, z: i32,
                   orgs: ptr<function, array<u32, 4>>,
                   count: ptr<function, u32>) {
    let dres = i32(params.density_resolution);
    let dx = x - 1; let dy = y - 1; let dz = z - 1;
    if dx < 0 || dx >= dres || dy < 0 || dy >= dres || dz < 0 || dz >= dres { return; }
    let idx = density_index(u32(dx), u32(dy), u32(dz));

    var ids = array<u32, 4>(org_id_0[idx], org_id_1[idx], org_id_2[idx], org_id_3[idx]);
    for (var s = 0u; s < 4u; s++) {
        let oid = ids[s];
        if oid == 0u { continue; }
        if *count >= 4u { return; }
        var found = false;
        for (var j = 0u; j < *count; j++) {
            if (*orgs)[j] == oid { found = true; break; }
        }
        if !found {
            (*orgs)[*count] = oid;
            *count += 1u;
        }
    }
}

const CORNER_OFFSETS: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
    vec3<i32>(0,0,0), vec3<i32>(1,0,0), vec3<i32>(0,1,0), vec3<i32>(1,1,0),
    vec3<i32>(0,0,1), vec3<i32>(1,0,1), vec3<i32>(0,1,1), vec3<i32>(1,1,1)
);

const EDGE_PAIRS: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u,1u), vec2<u32>(2u,3u), vec2<u32>(4u,5u), vec2<u32>(6u,7u),
    vec2<u32>(0u,2u), vec2<u32>(1u,3u), vec2<u32>(4u,6u), vec2<u32>(5u,7u),
    vec2<u32>(0u,4u), vec2<u32>(1u,5u), vec2<u32>(2u,6u), vec2<u32>(3u,7u)
);

// Store vertex index into the appropriate vertex map slot
fn store_vertex_map(cell_idx: u32, org_i: u32, vertex_idx: u32) {
    let encoded = vertex_idx + 1u;
    switch org_i {
        case 0u: { atomicStore(&vertex_map_0[cell_idx], encoded); }
        case 1u: { atomicStore(&vertex_map_1[cell_idx], encoded); }
        case 2u: { atomicStore(&vertex_map_2[cell_idx], encoded); }
        case 3u: { atomicStore(&vertex_map_3[cell_idx], encoded); }
        default: {}
    }
}

// Load vertex map slot
fn load_vertex_map(cell_idx: u32, slot: u32) -> u32 {
    switch slot {
        case 0u: { return atomicLoad(&vertex_map_0[cell_idx]); }
        case 1u: { return atomicLoad(&vertex_map_1[cell_idx]); }
        case 2u: { return atomicLoad(&vertex_map_2[cell_idx]); }
        case 3u: { return atomicLoad(&vertex_map_3[cell_idx]); }
        default: { return 0u; }
    }
}

// Find the vertex index for a given organism at a neighboring cell.
fn find_vertex_for_org(neighbor_cell_idx: u32, target_org: u32) -> u32 {
    for (var s = 0u; s < 4u; s++) {
        let vm = load_vertex_map(neighbor_cell_idx, s);
        if vm != 0u {
            let vi = vm - 1u;
            if u32(vertices[vi].organism_id) == target_org { return vm; }
        }
    }
    return 0u;
}

fn emit_quad(v0: u32, v1: u32, v2: u32, v3: u32, inside: bool) {
    let idx_base = atomicAdd(&counters.index_count, 6u);
    if idx_base + 5u >= params.max_indices { return; }
    if inside {
        indices[idx_base+0u]=v0; indices[idx_base+1u]=v1; indices[idx_base+2u]=v2;
        indices[idx_base+3u]=v0; indices[idx_base+4u]=v2; indices[idx_base+5u]=v3;
    } else {
        indices[idx_base+0u]=v0; indices[idx_base+1u]=v2; indices[idx_base+2u]=v1;
        indices[idx_base+3u]=v0; indices[idx_base+4u]=v3; indices[idx_base+5u]=v2;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: Generate vertices
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(4, 4, 4)
fn generate_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    let cell_idx = grid_index(gid.x, gid.y, gid.z);
    let iso = params.iso_level;
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);

    // Early-out: skip if no density at center voxel (fast path for vast empty space)
    let dres = i32(params.density_resolution);
    let cdx = ix - 1; let cdy = iy - 1; let cdz = iz - 1;
    var has_density = false;
    if cdx >= 0 && cdx < dres && cdy >= 0 && cdy < dres && cdz >= 0 && cdz < dres {
        let cidx = density_index(u32(cdx), u32(cdy), u32(cdz));
        has_density = density_0[cidx] > 0.0 || density_1[cidx] > 0.0
                   || density_2[cidx] > 0.0 || density_3[cidx] > 0.0;
    }
    // If center has no density, check remaining corners
    if !has_density {
        for (var i = 0u; i < 8u && !has_density; i++) {
            let off = CORNER_OFFSETS[i];
            if has_any_density_at(ix + off.x, iy + off.y, iz + off.z) {
                has_density = true;
            }
        }
    }

    // Clear all 4 vertex map slots
    atomicStore(&vertex_map_0[cell_idx], 0u);
    atomicStore(&vertex_map_1[cell_idx], 0u);
    atomicStore(&vertex_map_2[cell_idx], 0u);
    atomicStore(&vertex_map_3[cell_idx], 0u);

    if !has_density { return; }

    // Collect unique organism IDs across all 8 corners (up to 4)
    var unique_orgs: array<u32, 4>;
    var num_unique = 0u;
    for (var i = 0u; i < 8u; i++) {
        let off = CORNER_OFFSETS[i];
        collect_orgs_at(ix + off.x, iy + off.y, iz + off.z,
                        &unique_orgs, &num_unique);
        if num_unique >= 4u { break; }
    }
    if num_unique == 0u { return; }

    // For each unique organism, check for isosurface crossing and emit vertex
    for (var org_i = 0u; org_i < num_unique; org_i++) {
        let target_org = unique_orgs[org_i];

        var corners: array<f32, 8>;
        for (var i = 0u; i < 8u; i++) {
            let off = CORNER_OFFSETS[i];
            corners[i] = sample_organism_density(ix + off.x, iy + off.y, iz + off.z, target_org);
        }

        var inside_count = 0u;
        for (var i = 0u; i < 8u; i++) {
            if corners[i] >= iso { inside_count++; }
        }
        if inside_count == 0u || inside_count == 8u { continue; }

        // Find edge crossings
        var sum = vec3<f32>(0.0);
        var edge_count = 0u;
        for (var e = 0u; e < 12u; e++) {
            let pair = EDGE_PAIRS[e];
            let v0 = corners[pair.x];
            let v1 = corners[pair.y];
            if (v0 >= iso) != (v1 >= iso) {
                var t = 0.5;
                if abs(v1 - v0) > 1e-6 { t = clamp((iso - v0) / (v1 - v0), 0.0, 1.0); }
                let p0 = vec3<f32>(CORNER_OFFSETS[pair.x]);
                let p1 = vec3<f32>(CORNER_OFFSETS[pair.y]);
                sum += p0 + (p1 - p0) * t;
                edge_count++;
            }
        }
        if edge_count == 0u { continue; }

        let vertex_idx = atomicAdd(&counters.vertex_count, 1u);
        if vertex_idx >= params.max_vertices { continue; }

        let local_pos = sum / f32(edge_count);
        let world_pos = grid_to_world(
            f32(gid.x) + local_pos.x,
            f32(gid.y) + local_pos.y,
            f32(gid.z) + local_pos.z);

        let grad_x = (corners[1] + corners[3] + corners[5] + corners[7])
                   - (corners[0] + corners[2] + corners[4] + corners[6]);
        let grad_y = (corners[2] + corners[3] + corners[6] + corners[7])
                   - (corners[0] + corners[1] + corners[4] + corners[5]);
        let grad_z = (corners[4] + corners[5] + corners[6] + corners[7])
                   - (corners[0] + corners[1] + corners[2] + corners[3]);
        let gradient = vec3<f32>(-grad_x, -grad_y, -grad_z);
        var normal = normalize(gradient);
        if dot(gradient, gradient) < 1e-12 { normal = vec3<f32>(0.0, 1.0, 0.0); }

        vertices[vertex_idx] = Vertex(world_pos, f32(target_org), normal, 0.0);
        store_vertex_map(cell_idx, org_i, vertex_idx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: Generate indices
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(4, 4, 4)
fn generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    let cell_idx = grid_index(gid.x, gid.y, gid.z);
    let iso = params.iso_level;
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);

    // Process all 4 vertex map slots at this cell
    for (var slot = 0u; slot < 4u; slot++) {
        let encoded = load_vertex_map(cell_idx, slot);
        if encoded == 0u { continue; }

        let v0_idx = encoded - 1u;
        let target_org = u32(vertices[v0_idx].organism_id);
        if target_org == 0u { continue; }

        let corner0 = sample_organism_density(ix, iy, iz, target_org);
        let corner0_inside = corner0 >= iso;

        // X edge
        let corner1 = sample_organism_density(ix + 1, iy, iz, target_org);
        if (corner0 >= iso) != (corner1 >= iso) {
            if gid.y > 0u && gid.z > 0u {
                let v1 = find_vertex_for_org(grid_index(gid.x, gid.y - 1u, gid.z), target_org);
                let v2 = find_vertex_for_org(grid_index(gid.x, gid.y - 1u, gid.z - 1u), target_org);
                let v3 = find_vertex_for_org(grid_index(gid.x, gid.y, gid.z - 1u), target_org);
                if v1 != 0u && v2 != 0u && v3 != 0u {
                    emit_quad(v0_idx, v1 - 1u, v2 - 1u, v3 - 1u, corner0_inside);
                }
            }
        }

        // Y edge
        let corner2 = sample_organism_density(ix, iy + 1, iz, target_org);
        if (corner0 >= iso) != (corner2 >= iso) {
            if gid.x > 0u && gid.z > 0u {
                let v1 = find_vertex_for_org(grid_index(gid.x, gid.y, gid.z - 1u), target_org);
                let v2 = find_vertex_for_org(grid_index(gid.x - 1u, gid.y, gid.z - 1u), target_org);
                let v3 = find_vertex_for_org(grid_index(gid.x - 1u, gid.y, gid.z), target_org);
                if v1 != 0u && v2 != 0u && v3 != 0u {
                    emit_quad(v0_idx, v1 - 1u, v2 - 1u, v3 - 1u, corner0_inside);
                }
            }
        }
        // Z edge
        let corner4 = sample_organism_density(ix, iy, iz + 1, target_org);
        if (corner0 >= iso) != (corner4 >= iso) {
            if gid.x > 0u && gid.y > 0u {
                let v1 = find_vertex_for_org(grid_index(gid.x - 1u, gid.y, gid.z), target_org);
                let v2 = find_vertex_for_org(grid_index(gid.x - 1u, gid.y - 1u, gid.z), target_org);
                let v3 = find_vertex_for_org(grid_index(gid.x, gid.y - 1u, gid.z), target_org);
                if v1 != 0u && v2 != 0u && v3 != 0u {
                    emit_quad(v0_idx, v1 - 1u, v2 - 1u, v3 - 1u, corner0_inside);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3: Reset counters and indirect draw buffer
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(1, 1, 1)
fn reset_counters() {
    atomicStore(&counters.vertex_count, 0u);
    atomicStore(&counters.index_count, 0u);
    indirect_draw.index_count = 0u;
    indirect_draw.instance_count = 1u;
    indirect_draw.first_index = 0u;
    indirect_draw.base_vertex = 0i;
    indirect_draw.first_instance = 0u;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4: Finalize indirect draw buffer from counters
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(1, 1, 1)
fn finalize_indirect() {
    let vc = atomicLoad(&counters.vertex_count);
    let ic = atomicLoad(&counters.index_count);
    indirect_draw.index_count = min(ic, params.max_indices);
    indirect_draw.instance_count = 1u;
    indirect_draw.first_index = 0u;
    indirect_draw.base_vertex = 0i;
    indirect_draw.first_instance = 0u;
}
