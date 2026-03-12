// Organism Skin Surface Nets — K=2 overlapping organism extraction
//
// Fork of surface_nets_gpu.wgsl that reads from dual-slot density/org_id buffers.
// For each voxel cell, finds up to 2 unique organism IDs across the 8 corners
// and generates an independent isosurface vertex for each organism present.
//
// Uses two vertex maps (one per organism slot) so vertex indices are full 32-bit.
// The vertex organism_id field carries the skin ID for per-organism coloring.

struct SurfaceNetsParams {
    grid_resolution: u32,   // Padded processing grid (e.g. 130)
    iso_level: f32,
    cell_size: f32,
    max_vertices: u32,

    grid_origin: vec3<f32>,
    max_indices: u32,

    density_resolution: u32, // Actual density data size (e.g. 128)
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
}

struct Vertex {
    position: vec3<f32>,
    organism_id: f32,  // organism skin ID (as f32 for vertex attribute compat)
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

// K=2 density slots (f32, written by normalize pass)
@group(0) @binding(1) var<storage, read> density_0: array<f32>;
@group(0) @binding(2) var<storage, read> density_1: array<f32>;

// K=2 organism ID slots (u32, written by normalize pass)
@group(0) @binding(3) var<storage, read> org_id_0: array<u32>;
@group(0) @binding(4) var<storage, read> org_id_1: array<u32>;

// Output mesh
@group(0) @binding(5) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(6) var<storage, read_write> indices: array<u32>;

// Two vertex maps — one per organism slot at each voxel cell.
// Each stores (vertex_index + 1), with 0 meaning "no vertex".
@group(0) @binding(7) var<storage, read_write> vertex_map_0: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> vertex_map_1: array<atomic<u32>>;

@group(0) @binding(9) var<storage, read_write> counters: Counter;
@group(0) @binding(10) var<storage, read_write> indirect_draw: IndirectDraw;

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

fn is_in_density_bounds(x: i32, y: i32, z: i32) -> bool {
    let dres = i32(params.density_resolution);
    let dx = x - 1;
    let dy = y - 1;
    let dz = z - 1;
    return dx >= 0 && dx < dres && dy >= 0 && dy < dres && dz >= 0 && dz < dres;
}

// Sample density for a specific organism at a padded-grid coordinate.
fn sample_organism_density(x: i32, y: i32, z: i32, target_org: u32) -> f32 {
    let dres = i32(params.density_resolution);
    let dx = x - 1;
    let dy = y - 1;
    let dz = z - 1;
    if dx < 0 || dx >= dres || dy < 0 || dy >= dres || dz < 0 || dz >= dres {
        return 0.0;
    }
    let idx = density_index(u32(dx), u32(dy), u32(dz));
    if org_id_0[idx] == target_org { return density_0[idx]; }
    if org_id_1[idx] == target_org { return density_1[idx]; }
    return 0.0;
}

// Sample density for a specific organism with 3x3x3 Gaussian smoothing.
fn sample_organism_density_filtered(x: i32, y: i32, z: i32, target_org: u32) -> f32 {
    if !is_in_density_bounds(x, y, z) { return 0.0; }
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var ddx = -1; ddx <= 1; ddx++) {
        for (var ddy = -1; ddy <= 1; ddy++) {
            for (var ddz = -1; ddz <= 1; ddz++) {
                let dist_sq = f32(ddx * ddx + ddy * ddy + ddz * ddz);
                let weight = exp(-dist_sq * 0.8);
                sum += sample_organism_density(x + ddx, y + ddy, z + ddz, target_org) * weight;
                weight_sum += weight;
            }
        }
    }
    return sum / weight_sum;
}

// Check if any density exists at a padded-grid voxel (either slot)
fn has_any_density_at(x: i32, y: i32, z: i32) -> bool {
    let dres = i32(params.density_resolution);
    let dx = x - 1; let dy = y - 1; let dz = z - 1;
    if dx < 0 || dx >= dres || dy < 0 || dy >= dres || dz < 0 || dz >= dres { return false; }
    let idx = density_index(u32(dx), u32(dy), u32(dz));
    return density_0[idx] > 0.0 || density_1[idx] > 0.0;
}

// Collect unique organism IDs at a density voxel (up to 2)
fn get_orgs_at(x: i32, y: i32, z: i32, out_orgs: ptr<function, array<u32, 2>>, out_count: ptr<function, u32>) {
    let dres = i32(params.density_resolution);
    let dx = x - 1; let dy = y - 1; let dz = z - 1;
    if dx < 0 || dx >= dres || dy < 0 || dy >= dres || dz < 0 || dz >= dres { return; }
    let idx = density_index(u32(dx), u32(dy), u32(dz));
    let o0 = org_id_0[idx];
    let o1 = org_id_1[idx];
    if o0 != 0u {
        (*out_orgs)[*out_count] = o0;
        *out_count += 1u;
    }
    if o1 != 0u && o1 != o0 && *out_count < 2u {
        (*out_orgs)[*out_count] = o1;
        *out_count += 1u;
    }
}

const CORNER_OFFSETS: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1)
);

const EDGE_PAIRS: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),
    vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u)
);

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

    // Early-out: check 8 corners for any density
    var has_density = false;
    for (var i = 0u; i < 8u && !has_density; i++) {
        let off = CORNER_OFFSETS[i];
        if has_any_density_at(ix + off.x, iy + off.y, iz + off.z) {
            has_density = true;
        }
    }
    if !has_density {
        atomicStore(&vertex_map_0[cell_idx], 0u);
        atomicStore(&vertex_map_1[cell_idx], 0u);
        return;
    }

    // Collect unique organism IDs across all 8 corners
    var unique_orgs: array<u32, 2>;
    var num_unique = 0u;

    for (var i = 0u; i < 8u; i++) {
        let off = CORNER_OFFSETS[i];
        let cx = ix + off.x;
        let cy = iy + off.y;
        let cz = iz + off.z;

        let dres = i32(params.density_resolution);
        let ddx = cx - 1; let ddy = cy - 1; let ddz = cz - 1;
        if ddx < 0 || ddx >= dres || ddy < 0 || ddy >= dres || ddz < 0 || ddz >= dres { continue; }
        let didx = density_index(u32(ddx), u32(ddy), u32(ddz));

        let o0 = org_id_0[didx];
        let o1 = org_id_1[didx];

        if o0 != 0u && num_unique < 2u {
            var found = false;
            for (var j = 0u; j < num_unique; j++) { if unique_orgs[j] == o0 { found = true; break; } }
            if !found { unique_orgs[num_unique] = o0; num_unique++; }
        }
        if o1 != 0u && num_unique < 2u {
            var found = false;
            for (var j = 0u; j < num_unique; j++) { if unique_orgs[j] == o1 { found = true; break; } }
            if !found { unique_orgs[num_unique] = o1; num_unique++; }
        }
    }

    // Default: no vertices
    atomicStore(&vertex_map_0[cell_idx], 0u);
    atomicStore(&vertex_map_1[cell_idx], 0u);

    if num_unique == 0u { return; }

    // For each unique organism, check for isosurface crossing and emit vertex
    for (var org_i = 0u; org_i < num_unique; org_i++) {
        let target_org = unique_orgs[org_i];

        // Sample 8 corners for this organism
        var corners: array<f32, 8>;
        for (var i = 0u; i < 8u; i++) {
            let off = CORNER_OFFSETS[i];
            corners[i] = sample_organism_density_filtered(ix + off.x, iy + off.y, iz + off.z, target_org);
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
        let world_pos = grid_to_world(f32(gid.x) + local_pos.x, f32(gid.y) + local_pos.y, f32(gid.z) + local_pos.z);

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

        // Store in the appropriate vertex map slot
        if org_i == 0u {
            atomicStore(&vertex_map_0[cell_idx], vertex_idx + 1u);
        } else {
            atomicStore(&vertex_map_1[cell_idx], vertex_idx + 1u);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: Generate indices
// For each organism vertex at this cell, connect to same-organism vertices
// at neighboring cells.
// ─────────────────────────────────────────────────────────────────────────────

// Find the vertex index for a given organism at a neighboring cell.
// Checks both vertex maps and returns the one whose organism_id matches.
// Returns 0 if no matching vertex exists.
fn find_vertex_for_org(neighbor_cell_idx: u32, target_org: u32) -> u32 {
    let vm0 = atomicLoad(&vertex_map_0[neighbor_cell_idx]);
    if vm0 != 0u {
        let vi = vm0 - 1u;
        if u32(vertices[vi].organism_id) == target_org { return vm0; }
    }
    let vm1 = atomicLoad(&vertex_map_1[neighbor_cell_idx]);
    if vm1 != 0u {
        let vi = vm1 - 1u;
        if u32(vertices[vi].organism_id) == target_org { return vm1; }
    }
    return 0u;
}

fn emit_quad(v0: u32, v1: u32, v2: u32, v3: u32, inside: bool) {
    let idx_base = atomicAdd(&counters.index_count, 6u);
    if idx_base + 5u >= params.max_indices { return; }
    if inside {
        indices[idx_base + 0u] = v0; indices[idx_base + 1u] = v1; indices[idx_base + 2u] = v2;
        indices[idx_base + 3u] = v0; indices[idx_base + 4u] = v2; indices[idx_base + 5u] = v3;
    } else {
        indices[idx_base + 0u] = v0; indices[idx_base + 1u] = v2; indices[idx_base + 2u] = v1;
        indices[idx_base + 3u] = v0; indices[idx_base + 4u] = v3; indices[idx_base + 5u] = v2;
    }
}

@compute @workgroup_size(4, 4, 4)
fn generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    let cell_idx = grid_index(gid.x, gid.y, gid.z);
    let iso = params.iso_level;
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);

    // Process both vertex map slots at this cell
    for (var slot = 0u; slot < 2u; slot++) {
        let encoded = select(atomicLoad(&vertex_map_0[cell_idx]),
                             atomicLoad(&vertex_map_1[cell_idx]),
                             slot == 1u);
        if encoded == 0u { continue; }

        let v0_idx = encoded - 1u;
        let target_org = u32(vertices[v0_idx].organism_id);
        if target_org == 0u { continue; }

        let corner0 = sample_organism_density_filtered(ix, iy, iz, target_org);
        let corner0_inside = corner0 >= iso;

        // X edge
        let corner1 = sample_organism_density_filtered(ix + 1, iy, iz, target_org);
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
        let corner2 = sample_organism_density_filtered(ix, iy + 1, iz, target_org);
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
        let corner4 = sample_organism_density_filtered(ix, iy, iz + 1, target_org);
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
// Reset / finalize
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(1)
fn reset_counters() {
    atomicStore(&counters.vertex_count, 0u);
    atomicStore(&counters.index_count, 0u);
    indirect_draw.index_count = 0u;
    indirect_draw.instance_count = 1u;
    indirect_draw.first_index = 0u;
    indirect_draw.base_vertex = 0i;
    indirect_draw.first_instance = 0u;
}

@compute @workgroup_size(1)
fn finalize_indirect() {
    indirect_draw.index_count = min(atomicLoad(&counters.index_count), params.max_indices);
}
