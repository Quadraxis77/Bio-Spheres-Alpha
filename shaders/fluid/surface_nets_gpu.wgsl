// GPU Surface Nets - Isosurface extraction from density voxels
// Two-pass algorithm:
// Pass 1: Generate vertices at cells containing isosurface
// Pass 2: Generate quads connecting adjacent surface cells

struct SurfaceNetsParams {
    grid_resolution: u32,
    iso_level: f32,
    cell_size: f32,
    max_vertices: u32,
    
    grid_origin: vec3<f32>,
    max_indices: u32,
}

struct Vertex {
    position: vec3<f32>,
    fluid_type: f32,  // 0=empty, 1=water, 2=lava, 3=steam
    normal: vec3<f32>,
    _pad1: f32,
}

struct Counter {
    vertex_count: atomic<u32>,
    index_count: atomic<u32>,
}

// Indirect draw buffer format for draw_indexed_indirect
struct IndirectDraw {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: SurfaceNetsParams;
@group(0) @binding(1) var<storage, read> density: array<f32>;
@group(0) @binding(2) var<storage, read> fluid_types: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> vertex_map: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> counters: Counter;
@group(0) @binding(7) var<storage, read_write> indirect_draw: IndirectDraw;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn grid_to_world(x: f32, y: f32, z: f32) -> vec3<f32> {
    return params.grid_origin + vec3<f32>(x, y, z) * params.cell_size;
}

fn sample_density_clamped(x: i32, y: i32, z: i32) -> f32 {
    let res = i32(params.grid_resolution);
    let cx = clamp(x, 0, res - 1);
    let cy = clamp(y, 0, res - 1);
    let cz = clamp(z, 0, res - 1);
    return density[grid_index(u32(cx), u32(cy), u32(cz))];
}

// Corner offsets for the 8 corners of a cell
const CORNER_OFFSETS: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1)
);

// Edge pairs (corner indices) and their directions
const EDGE_PAIRS: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(4u, 5u), vec2<u32>(6u, 7u),  // X edges
    vec2<u32>(0u, 2u), vec2<u32>(1u, 3u), vec2<u32>(4u, 6u), vec2<u32>(5u, 7u),  // Y edges
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u)   // Z edges
);

// Pass 1: Generate vertices for cells containing isosurface
@compute @workgroup_size(4, 4, 4)
fn generate_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res - 1u || gid.y >= res - 1u || gid.z >= res - 1u {
        return;
    }
    
    let cell_idx = grid_index(gid.x, gid.y, gid.z);
    let iso = params.iso_level;
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);
    
    // Sample 8 corners
    var corners: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let off = CORNER_OFFSETS[i];
        corners[i] = sample_density_clamped(ix + off.x, iy + off.y, iz + off.z);
    }
    
    // Count inside/outside corners
    var inside_count = 0u;
    for (var i = 0u; i < 8u; i++) {
        if corners[i] >= iso {
            inside_count++;
        }
    }
    
    // No surface if all same side
    if inside_count == 0u || inside_count == 8u {
        atomicStore(&vertex_map[cell_idx], 0u);
        return;
    }
    
    // Find edge crossings and average positions
    var sum = vec3<f32>(0.0);
    var count = 0u;
    
    for (var e = 0u; e < 12u; e++) {
        let pair = EDGE_PAIRS[e];
        let v0 = corners[pair.x];
        let v1 = corners[pair.y];
        
        // Edge crosses isosurface?
        if (v0 >= iso) != (v1 >= iso) {
            var t = 0.5;
            if abs(v1 - v0) > 1e-6 {
                t = clamp((iso - v0) / (v1 - v0), 0.0, 1.0);
            }
            
            let p0 = vec3<f32>(CORNER_OFFSETS[pair.x]);
            let p1 = vec3<f32>(CORNER_OFFSETS[pair.y]);
            sum += p0 + (p1 - p0) * t;
            count++;
        }
    }
    
    if count == 0u {
        atomicStore(&vertex_map[cell_idx], 0u);
        return;
    }
    
    // Allocate vertex
    let vertex_idx = atomicAdd(&counters.vertex_count, 1u);
    if vertex_idx >= params.max_vertices {
        atomicStore(&vertex_map[cell_idx], 0u);
        return;
    }
    
    // Store vertex index + 1 (0 means no vertex)
    atomicStore(&vertex_map[cell_idx], vertex_idx + 1u);
    
    // Compute vertex position
    let local_pos = sum / f32(count);
    let world_pos = grid_to_world(f32(gid.x) + local_pos.x, f32(gid.y) + local_pos.y, f32(gid.z) + local_pos.z);
    
    // Compute normal from density gradient (central differences)
    let gx = sample_density_clamped(ix + 1, iy, iz) - sample_density_clamped(ix - 1, iy, iz);
    let gy = sample_density_clamped(ix, iy + 1, iz) - sample_density_clamped(ix, iy - 1, iz);
    let gz = sample_density_clamped(ix, iy, iz + 1) - sample_density_clamped(ix, iy, iz - 1);
    var normal = normalize(vec3<f32>(-gx, -gy, -gz));
    if length(vec3<f32>(gx, gy, gz)) < 1e-6 {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    }
    
    // Get fluid type from this cell
    let ft = f32(fluid_types[cell_idx]);
    
    vertices[vertex_idx] = Vertex(world_pos, ft, normal, 0.0);
}

// Pass 2: Generate quads connecting adjacent surface cells
@compute @workgroup_size(4, 4, 4)
fn generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res - 1u || gid.y >= res - 1u || gid.z >= res - 1u {
        return;
    }
    
    let cell_idx = grid_index(gid.x, gid.y, gid.z);
    let v0_idx = atomicLoad(&vertex_map[cell_idx]);
    if v0_idx == 0u {
        return;
    }
    
    let iso = params.iso_level;
    let ix = i32(gid.x);
    let iy = i32(gid.y);
    let iz = i32(gid.z);
    
    // Sample corner 0 to determine winding
    let corner0 = sample_density_clamped(ix, iy, iz);
    let corner0_inside = corner0 >= iso;
    
    // Check X edge (0-1)
    let corner1 = sample_density_clamped(ix + 1, iy, iz);
    if (corner0 >= iso) != (corner1 >= iso) {
        if gid.y > 0u && gid.z > 0u {
            let v1_idx = atomicLoad(&vertex_map[grid_index(gid.x, gid.y - 1u, gid.z)]);
            let v2_idx = atomicLoad(&vertex_map[grid_index(gid.x, gid.y - 1u, gid.z - 1u)]);
            let v3_idx = atomicLoad(&vertex_map[grid_index(gid.x, gid.y, gid.z - 1u)]);
            
            if v1_idx != 0u && v2_idx != 0u && v3_idx != 0u {
                let idx_base = atomicAdd(&counters.index_count, 6u);
                if idx_base + 5u < params.max_indices {
                    let i0 = v0_idx - 1u;
                    let i1 = v1_idx - 1u;
                    let i2 = v2_idx - 1u;
                    let i3 = v3_idx - 1u;
                    
                    if corner0_inside {
                        indices[idx_base + 0u] = i0;
                        indices[idx_base + 1u] = i1;
                        indices[idx_base + 2u] = i2;
                        indices[idx_base + 3u] = i0;
                        indices[idx_base + 4u] = i2;
                        indices[idx_base + 5u] = i3;
                    } else {
                        indices[idx_base + 0u] = i0;
                        indices[idx_base + 1u] = i2;
                        indices[idx_base + 2u] = i1;
                        indices[idx_base + 3u] = i0;
                        indices[idx_base + 4u] = i3;
                        indices[idx_base + 5u] = i2;
                    }
                }
            }
        }
    }
    
    // Check Y edge (0-2)
    let corner2 = sample_density_clamped(ix, iy + 1, iz);
    if (corner0 >= iso) != (corner2 >= iso) {
        if gid.x > 0u && gid.z > 0u {
            let v1_idx = atomicLoad(&vertex_map[grid_index(gid.x, gid.y, gid.z - 1u)]);
            let v2_idx = atomicLoad(&vertex_map[grid_index(gid.x - 1u, gid.y, gid.z - 1u)]);
            let v3_idx = atomicLoad(&vertex_map[grid_index(gid.x - 1u, gid.y, gid.z)]);
            
            if v1_idx != 0u && v2_idx != 0u && v3_idx != 0u {
                let idx_base = atomicAdd(&counters.index_count, 6u);
                if idx_base + 5u < params.max_indices {
                    let i0 = v0_idx - 1u;
                    let i1 = v1_idx - 1u;
                    let i2 = v2_idx - 1u;
                    let i3 = v3_idx - 1u;
                    
                    if corner0_inside {
                        indices[idx_base + 0u] = i0;
                        indices[idx_base + 1u] = i1;
                        indices[idx_base + 2u] = i2;
                        indices[idx_base + 3u] = i0;
                        indices[idx_base + 4u] = i2;
                        indices[idx_base + 5u] = i3;
                    } else {
                        indices[idx_base + 0u] = i0;
                        indices[idx_base + 1u] = i2;
                        indices[idx_base + 2u] = i1;
                        indices[idx_base + 3u] = i0;
                        indices[idx_base + 4u] = i3;
                        indices[idx_base + 5u] = i2;
                    }
                }
            }
        }
    }
    
    // Check Z edge (0-4)
    let corner4 = sample_density_clamped(ix, iy, iz + 1);
    if (corner0 >= iso) != (corner4 >= iso) {
        if gid.x > 0u && gid.y > 0u {
            let v1_idx = atomicLoad(&vertex_map[grid_index(gid.x - 1u, gid.y, gid.z)]);
            let v2_idx = atomicLoad(&vertex_map[grid_index(gid.x - 1u, gid.y - 1u, gid.z)]);
            let v3_idx = atomicLoad(&vertex_map[grid_index(gid.x, gid.y - 1u, gid.z)]);
            
            if v1_idx != 0u && v2_idx != 0u && v3_idx != 0u {
                let idx_base = atomicAdd(&counters.index_count, 6u);
                if idx_base + 5u < params.max_indices {
                    let i0 = v0_idx - 1u;
                    let i1 = v1_idx - 1u;
                    let i2 = v2_idx - 1u;
                    let i3 = v3_idx - 1u;
                    
                    if corner0_inside {
                        indices[idx_base + 0u] = i0;
                        indices[idx_base + 1u] = i1;
                        indices[idx_base + 2u] = i2;
                        indices[idx_base + 3u] = i0;
                        indices[idx_base + 4u] = i2;
                        indices[idx_base + 5u] = i3;
                    } else {
                        indices[idx_base + 0u] = i0;
                        indices[idx_base + 1u] = i2;
                        indices[idx_base + 2u] = i1;
                        indices[idx_base + 3u] = i0;
                        indices[idx_base + 4u] = i3;
                        indices[idx_base + 5u] = i2;
                    }
                }
            }
        }
    }
}

// Reset counters before each extraction
@compute @workgroup_size(1)
fn reset_counters() {
    atomicStore(&counters.vertex_count, 0u);
    atomicStore(&counters.index_count, 0u);
    // Initialize indirect draw buffer
    indirect_draw.index_count = 0u;
    indirect_draw.instance_count = 1u;
    indirect_draw.first_index = 0u;
    indirect_draw.base_vertex = 0i;
    indirect_draw.first_instance = 0u;
}

// Finalize indirect draw buffer after index generation
@compute @workgroup_size(1)
fn finalize_indirect() {
    indirect_draw.index_count = min(atomicLoad(&counters.index_count), params.max_indices);
}
