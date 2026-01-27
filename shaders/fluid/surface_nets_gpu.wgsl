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
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> vertex_map: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> counters: Counter;
@group(0) @binding(8) var<storage, read_write> indirect_draw: IndirectDraw;

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

// Check if a position is solid (from solid mask)
fn is_solid(x: u32, y: u32, z: u32) -> bool {
    let idx = grid_index(x, y, z);
    return solid_mask[idx] == 1u;
}

// Check if a position is at world boundary
fn is_at_boundary(x: i32, y: i32, z: i32) -> bool {
    let res = i32(params.grid_resolution);
    return x < 0 || x >= res || y < 0 || y >= res || z < 0 || z >= res;
}

// GREEDY APPROACH: Only create surface if water touches empty space (not solid or boundary)
fn should_create_water_surface(x: i32, y: i32, z: i32) -> bool {
    // First check if this voxel actually contains water
    if x < 0 || y < 0 || z < 0 {
        return false;
    }
    
    let res = i32(params.grid_resolution);
    if x >= res || y >= res || z >= res {
        return false;
    }
    
    let idx = grid_index(u32(x), u32(y), u32(z));
    let water_density = density[idx];
    
    // Must have water to create surface
    if water_density <= params.iso_level {
        return false;
    }
    
    // Check all 6 neighbors - if ANY is empty space, create surface
    // If ALL neighbors are either water or solid/boundary, DON'T create surface
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1)   // -Z
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = x + offsets[i].x;
        let ny = y + offsets[i].y;
        let nz = z + offsets[i].z;
        
        // If neighbor is outside boundary, this water touches world boundary - NO SURFACE
        if is_at_boundary(nx, ny, nz) {
            continue; // Skip this neighbor, don't create surface for boundary contact
        }
        
        // If neighbor is solid, water touches solid - NO SURFACE
        if is_solid(u32(nx), u32(ny), u32(nz)) {
            continue; // Skip this neighbor, don't create surface for solid contact
        }
        
        // If neighbor is empty (no water), water touches air - CREATE SURFACE
        let neighbor_idx = grid_index(u32(nx), u32(ny), u32(nz));
        let neighbor_density = density[neighbor_idx];
        
        if neighbor_density <= params.iso_level {
            return true; // Water touches empty space - create surface!
        }
    }
    
    // All neighbors are either water, solid, or boundary - NO SURFACE
    return false;
}

// Multi-voxel averaging kernel for smoothing density field
fn sample_density_averaged(x: i32, y: i32, z: i32, kernel_size: i32) -> f32 {
    var sum = 0.0;
    var count = 0;
    let res = i32(params.grid_resolution);
    
    // Sample surrounding voxels in a cubic kernel
    for (var dx = -kernel_size; dx <= kernel_size; dx++) {
        for (var dy = -kernel_size; dy <= kernel_size; dy++) {
            for (var dz = -kernel_size; dz <= kernel_size; dz++) {
                let nx = clamp(x + dx, 0, res - 1);
                let ny = clamp(y + dy, 0, res - 1);
                let nz = clamp(z + dz, 0, res - 1);
                
                // Weight by distance (Gaussian-like falloff)
                let dist_sq = f32(dx * dx + dy * dy + dz * dz);
                let weight = exp(-dist_sq * 0.2); // Reduced weight for gentler smoothing
                
                sum += sample_density_clamped(nx, ny, nz) * weight;
                count += 1;
            }
        }
    }
    
    return sum / f32(count);
}

// Height-based filtering to ignore single voxel differences
fn should_create_surface(corners: array<f32, 8>, iso: f32) -> bool {
    // Count significant height differences (more than 1 voxel)
    var significant_changes = 0;
    var prev_inside = corners[0] >= iso;
    
    for (var i = 1u; i < 8u; i++) {
        let current_inside = corners[i] >= iso;
        if prev_inside != current_inside {
            significant_changes++;
        }
        prev_inside = current_inside;
    }
    
    // Only create surface if there are significant height changes
    // This filters out single voxel variations
    return significant_changes >= 1; // Reduced threshold to allow more surfaces
}

// Sample density with aggressive smoothing to eliminate fine details
fn sample_density_height_filtered(x: i32, y: i32, z: i32) -> f32 {
    // Use a 3x3x3 kernel for strong averaging
    var sum = 0.0;
    var weight_sum = 0.0;
    let res = i32(params.grid_resolution);
    
    // Sample surrounding voxels in a 3x3x3 kernel
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let nx = clamp(x + dx, 0, res - 1);
                let ny = clamp(y + dy, 0, res - 1);
                let nz = clamp(z + dz, 0, res - 1);
                
                // Strong Gaussian weighting for aggressive smoothing
                let dist_sq = f32(dx * dx + dy * dy + dz * dz);
                let weight = exp(-dist_sq * 0.8); // Strong falloff for more smoothing
                
                sum += sample_density_clamped(nx, ny, nz) * weight;
                weight_sum += weight;
            }
        }
    }
    
    return sum / weight_sum;
}

// Smoothed trilinear interpolation
fn sample_density_trilinear_smooth(world_pos: vec3<f32>) -> f32 {
    // Convert world position to grid coordinates
    let grid_pos = (world_pos - params.grid_origin) / params.cell_size;
    let grid_x = grid_pos.x - 0.5;
    let grid_y = grid_pos.y - 0.5;
    let grid_z = grid_pos.z - 0.5;
    
    // Get integer grid coordinates and fractional parts
    let x0 = i32(floor(grid_x));
    let y0 = i32(floor(grid_y));
    let z0 = i32(floor(grid_z));
    let fx = fract(grid_x);
    let fy = fract(grid_y);
    let fz = fract(grid_z);
    
    // Sample 8 corners with height filtering
    let c000 = sample_density_height_filtered(x0, y0, z0);
    let c100 = sample_density_height_filtered(x0 + 1, y0, z0);
    let c010 = sample_density_height_filtered(x0, y0 + 1, z0);
    let c110 = sample_density_height_filtered(x0 + 1, y0 + 1, z0);
    let c001 = sample_density_height_filtered(x0, y0, z0 + 1);
    let c101 = sample_density_height_filtered(x0 + 1, y0, z0 + 1);
    let c011 = sample_density_height_filtered(x0, y0 + 1, z0 + 1);
    let c111 = sample_density_height_filtered(x0 + 1, y0 + 1, z0 + 1);
    
    // Trilinear interpolation
    let c00 = mix(c000, c100, fx);
    let c01 = mix(c001, c101, fx);
    let c10 = mix(c010, c110, fx);
    let c11 = mix(c011, c111, fx);
    
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    
    return mix(c0, c1, fz);
}

// Improved gradient calculation with smoothed trilinear sampling
fn sample_gradient_trilinear(world_pos: vec3<f32>) -> vec3<f32> {
    let epsilon = params.cell_size * 0.1; // Small offset for gradient
    
    let dx = (sample_density_trilinear_smooth(world_pos + vec3<f32>(epsilon, 0.0, 0.0)) - 
             sample_density_trilinear_smooth(world_pos - vec3<f32>(epsilon, 0.0, 0.0))) / (2.0 * epsilon);
    let dy = (sample_density_trilinear_smooth(world_pos + vec3<f32>(0.0, epsilon, 0.0)) - 
             sample_density_trilinear_smooth(world_pos - vec3<f32>(0.0, epsilon, 0.0))) / (2.0 * epsilon);
    let dz = (sample_density_trilinear_smooth(world_pos + vec3<f32>(0.0, 0.0, epsilon)) - 
             sample_density_trilinear_smooth(world_pos - vec3<f32>(0.0, 0.0, epsilon))) / (2.0 * epsilon);
    
    return vec3<f32>(-dx, -dy, -dz);
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
    
    // GREEDY APPROACH: Check if any corner of this cell should create a surface
    var should_create_surface = false;
    for (var cx = 0; cx < 2; cx++) {
        for (var cy = 0; cy < 2; cy++) {
            for (var cz = 0; cz < 2; cz++) {
                if should_create_water_surface(ix + cx, iy + cy, iz + cz) {
                    should_create_surface = true;
                    break;
                }
            }
            if should_create_surface { break; }
        }
        if should_create_surface { break; }
    }
    
    if !should_create_surface {
        atomicStore(&vertex_map[cell_idx], 0u);
        return;
    }
    
    // Sample 8 corners with aggressive smoothing
    var corners: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let off = CORNER_OFFSETS[i];
        corners[i] = sample_density_height_filtered(ix + off.x, iy + off.y, iz + off.z);
    }
    
    // Count inside/outside corners (standard approach)
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
    
    // Find edge crossings with improved subvoxel precision
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
            
            // Use subvoxel positioning for smoother edges
            let edge_pos = p0 + (p1 - p0) * t;
            sum += edge_pos;
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
    
    // Compute vertex position with subvoxel precision
    let local_pos = sum / f32(count);
    let world_pos = grid_to_world(f32(gid.x) + local_pos.x, f32(gid.y) + local_pos.y, f32(gid.z) + local_pos.z);
    
    // Compute smooth normal using trilinear gradient
    let gradient = sample_gradient_trilinear(world_pos);
    var normal = normalize(gradient);
    if length(gradient) < 1e-6 {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    }
    
    // Force water type for consistent color since we're only rendering water surface
    let ft = 1.0;
    
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
    
    // Sample corner 0 to determine winding (with height filtering)
    let corner0 = sample_density_height_filtered(ix, iy, iz);
    let corner0_inside = corner0 >= iso;
    
    // Check X edge (0-1)
    let corner1 = sample_density_height_filtered(ix + 1, iy, iz);
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
    let corner2 = sample_density_height_filtered(ix, iy + 1, iz);
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
    let corner4 = sample_density_height_filtered(ix, iy, iz + 1);
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
