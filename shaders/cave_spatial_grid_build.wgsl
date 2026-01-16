// Cave Spatial Grid Builder
// Builds spatial grid for fast triangle lookup during collision detection

struct CaveParams {
    world_center: vec3<f32>,
    world_radius: f32,
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
    grid_resolution: u32,
    triangle_count: u32,
    collision_enabled: u32,
    collision_stiffness: f32,
    collision_damping: f32,
    substeps: u32,
    _padding: f32,
    // Padding to 256 bytes (16-byte aligned)
    _padding2: vec4<f32>,
    _padding3: vec4<f32>,
    _padding4: vec4<f32>,
    _padding5: vec4<f32>,
    _padding6: vec4<f32>,
    _padding7: vec4<f32>,
    _padding8: vec4<f32>,
    _padding9: vec4<f32>,
    _padding10: vec4<f32>,
    _padding11: vec4<f32>,
    _padding12: vec4<f32>,
    _padding13: vec4<f32>,
    _padding14: vec4<f32>,
    _padding15: vec4<f32>,
    _padding16: vec4<f32>,
    _padding17: vec4<f32>,
    _padding18: vec4<f32>,
    _padding19: vec4<f32>,
    _padding20: vec4<f32>,
    _padding21: vec4<f32>,
    _padding22: vec4<f32>,
    _padding23: vec4<f32>,
    _padding24: vec4<f32>,
    _padding25: vec4<f32>,
    _padding26: vec4<f32>,
    _padding27: vec4<f32>,
    _padding28: vec4<f32>,
    _padding29: vec4<f32>,
    _padding30: vec4<f32>,
    _padding31: vec4<f32>,
    _padding32: vec4<f32>,
    _padding33: vec4<f32>,
    _padding34: vec4<f32>,
    _padding35: vec4<f32>,
    _padding36: vec4<f32>,
    _padding37: vec4<f32>,
    _padding38: vec4<f32>,
    _padding39: vec4<f32>,
    _padding40: vec4<f32>,
    _padding41: vec4<f32>,
    _padding42: vec4<f32>,
    _padding43: vec4<f32>,
    _padding44: vec4<f32>,
    _padding45: vec4<f32>,
    _padding46: vec4<f32>,
    _padding47: vec4<f32>,
}

struct CaveVertex {
    position: vec3<f32>,
    _padding: f32,
}

struct CaveTriangle {
    v0: u32,
    v1: u32,
    v2: u32,
    _padding: u32,
}

struct CaveSpatialCell {
    triangle_count: atomic<u32>,
    triangle_indices: array<u32, 16>, // Max 16 triangles per cell
}

@group(0) @binding(0) var<uniform> cave_params: CaveParams;
@group(0) @binding(1) var<storage, read> cave_vertices: array<CaveVertex>;
@group(0) @binding(2) var<storage, read> cave_triangles: array<CaveTriangle>;
@group(0) @binding(3) var<storage, read_write> cave_spatial_grid: array<CaveSpatialCell>;

// Get AABB of triangle
fn get_triangle_aabb(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec4<f32> {
    let min_pos = min(min(v0, v1), v2);
    let max_pos = max(max(v0, v1), v2);
    return vec4<f32>(min_pos, max_pos.x); // Pack into vec4 for simplicity
}

// Convert world position to grid cell index
fn world_to_grid(pos: vec3<f32>) -> vec3<i32> {
    let normalized = (pos - cave_params.world_center + vec3<f32>(cave_params.world_radius)) / (cave_params.world_radius * 2.0);
    let grid_pos = vec3<i32>(floor(normalized * f32(cave_params.grid_resolution)));
    return clamp(grid_pos, vec3<i32>(0), vec3<i32>(i32(cave_params.grid_resolution) - 1));
}

// Convert grid cell to linear index
fn grid_to_index(grid_pos: vec3<i32>) -> i32 {
    let res = i32(cave_params.grid_resolution);
    return grid_pos.x + grid_pos.y * res + grid_pos.z * res * res;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tri_idx = global_id.x;
    
    if (tri_idx >= cave_params.triangle_count) {
        return;
    }
    
    // Get triangle vertices
    let tri = cave_triangles[tri_idx];
    let v0 = cave_vertices[tri.v0].position;
    let v1 = cave_vertices[tri.v1].position;
    let v2 = cave_vertices[tri.v2].position;
    
    // Get triangle AABB
    let aabb_min = min(min(v0, v1), v2);
    let aabb_max = max(max(v0, v1), v2);
    
    // Convert AABB to grid cells
    let grid_min = world_to_grid(aabb_min);
    let grid_max = world_to_grid(aabb_max);
    
    // Insert triangle into all overlapping grid cells
    for (var x = grid_min.x; x <= grid_max.x; x = x + 1) {
        for (var y = grid_min.y; y <= grid_max.y; y = y + 1) {
            for (var z = grid_min.z; z <= grid_max.z; z = z + 1) {
                let grid_pos = vec3<i32>(x, y, z);
                let cell_idx = grid_to_index(grid_pos);
                
                // Atomically add triangle to cell
                let count = atomicAdd(&cave_spatial_grid[cell_idx].triangle_count, 1u);
                
                // Only store if there's space (max 16 triangles per cell)
                if (count < 16u) {
                    cave_spatial_grid[cell_idx].triangle_indices[count] = tri_idx;
                }
            }
        }
    }
}
