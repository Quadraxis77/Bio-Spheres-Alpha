// Light Field Compute Shader
// Computes per-voxel light intensity by ray marching from each voxel toward the light source.
// Occlusion comes from:
//   1. Cave solid mask (static geometry)
//   2. Cell occupancy grid (dynamic - cells block light)
// Output: light_field buffer with f32 intensity per voxel (0.0 = fully shadowed, 1.0 = fully lit)

struct LightFieldParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    world_radius: f32,
    // Light source (directional light defined by direction toward light)
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    // Number of ray march steps (higher = more accurate shadows, more expensive)
    max_steps: u32,
    // Step size multiplier (1.0 = one voxel per step)
    step_size: f32,
    // Absorption per solid voxel hit (controls shadow softness)
    absorption_solid: f32,
    // Absorption per cell-occupied voxel (cells partially block light)
    absorption_cell: f32,
    // Ambient light floor (minimum light even in full shadow)
    ambient_floor: f32,
    // Scattering coefficient for volumetric fog
    scattering_coefficient: f32,
    // Current time for animation
    time: f32,
}

// Group 0: Parameters
@group(0) @binding(0)
var<uniform> params: LightFieldParams;

// Group 0: Solid mask from cave system (1 = solid, 0 = empty)
@group(0) @binding(1)
var<storage, read> solid_mask: array<u32>;

// Group 0: Cell occupancy grid (atomic u32 count of cells per voxel)
// Built by a separate pass that bins cells into voxels
@group(0) @binding(2)
var<storage, read> cell_occupancy: array<u32>;

// Group 0: Output light field (f32 intensity per voxel)
@group(0) @binding(3)
var<storage, read_write> light_field: array<f32>;

// Convert 3D grid coordinates to linear index
fn grid_to_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

// Convert linear index to 3D grid coordinates
fn index_to_grid(idx: u32) -> vec3<u32> {
    let res = params.grid_resolution;
    let z = idx / (res * res);
    let y = (idx - z * res * res) / res;
    let x = idx - z * res * res - y * res;
    return vec3<u32>(x, y, z);
}

// Convert grid coordinates to world position (voxel center)
fn grid_to_world(gx: u32, gy: u32, gz: u32) -> vec3<f32> {
    return vec3<f32>(
        params.grid_origin_x + (f32(gx) + 0.5) * params.cell_size,
        params.grid_origin_y + (f32(gy) + 0.5) * params.cell_size,
        params.grid_origin_z + (f32(gz) + 0.5) * params.cell_size,
    );
}

// Check if grid coordinates are within bounds
fn is_in_bounds(x: i32, y: i32, z: i32) -> bool {
    let res = i32(params.grid_resolution);
    return x >= 0 && x < res && y >= 0 && y < res && z >= 0 && z < res;
}

// Sample solid mask at grid position (returns true if solid or out of bounds)
fn is_solid(x: i32, y: i32, z: i32) -> bool {
    if (!is_in_bounds(x, y, z)) {
        return true; // Out of bounds = solid (world boundary)
    }
    let idx = grid_to_index(u32(x), u32(y), u32(z));
    return solid_mask[idx] != 0u;
}

// Sample cell occupancy at grid position (returns number of cells in voxel)
fn cell_count_at(x: i32, y: i32, z: i32) -> u32 {
    if (!is_in_bounds(x, y, z)) {
        return 0u;
    }
    let idx = grid_to_index(u32(x), u32(y), u32(z));
    return cell_occupancy[idx];
}

// Check if a grid-space position is inside the world sphere
fn is_inside_sphere(gx: f32, gy: f32, gz: f32) -> bool {
    let fres = f32(params.grid_resolution);
    let half = fres * 0.5;
    let dx = gx - half;
    let dy = gy - half;
    let dz = gz - half;
    // Sphere radius in grid units = grid_resolution / 2
    return (dx * dx + dy * dy + dz * dz) <= (half * half);
}

// Ray march from voxel toward light source, accumulating occlusion
fn compute_light_at_voxel(gx: u32, gy: u32, gz: u32) -> f32 {
    let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));
    
    // Start from voxel center, march in light direction
    var pos = vec3<f32>(f32(gx) + 0.5, f32(gy) + 0.5, f32(gz) + 0.5);
    let step = light_dir * params.step_size;
    
    var transmittance = 1.0;
    
    for (var i = 0u; i < params.max_steps; i++) {
        pos += step;
        
        // Convert to integer grid coords
        let ix = i32(floor(pos.x));
        let iy = i32(floor(pos.y));
        let iz = i32(floor(pos.z));
        
        // If we've left the grid, we've reached open sky - stop
        if (!is_in_bounds(ix, iy, iz)) {
            break;
        }
        
        // If we've exited the world sphere, we've reached open sky - stop
        // (solid_mask marks outside-sphere voxels as solid for fluid,
        //  but they shouldn't cast light shadows)
        if (!is_inside_sphere(pos.x, pos.y, pos.z)) {
            break;
        }
        
        // Check solid (cave walls) - these fully block light
        if (is_solid(ix, iy, iz)) {
            transmittance *= exp(-params.absorption_solid);
        }
        
        // Check cell occupancy - cells fully block light
        let cells = cell_count_at(ix, iy, iz);
        if (cells > 0u) {
            transmittance *= exp(-params.absorption_cell * f32(cells));
            // Any cell presence is a heavy occluder
            if (transmittance < 0.01) {
                transmittance = 0.0;
                break;
            }
        }
        
        // Early exit if light is effectively blocked
        if (transmittance < 0.01) {
            transmittance = 0.0;
            break;
        }
    }
    
    // Apply ambient floor - even fully shadowed areas get some light
    return max(transmittance, params.ambient_floor);
}

@compute @workgroup_size(64)
fn compute_light_field(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_voxels = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_voxels) {
        return;
    }
    
    let grid_pos = index_to_grid(idx);
    
    // Voxels outside the world sphere get full light (open sky)
    // solid_mask marks these as solid for fluid containment, but they aren't occluders
    let grid_pos_f = vec3<f32>(f32(grid_pos.x) + 0.5, f32(grid_pos.y) + 0.5, f32(grid_pos.z) + 0.5);
    if (!is_inside_sphere(grid_pos_f.x, grid_pos_f.y, grid_pos_f.z)) {
        light_field[idx] = 1.0;
        return;
    }
    
    // Solid voxels inside the sphere get zero light (they are inside cave walls)
    if (solid_mask[idx] != 0u) {
        light_field[idx] = 0.0;
        return;
    }
    
    // Ray march toward light to compute intensity
    let intensity = compute_light_at_voxel(grid_pos.x, grid_pos.y, grid_pos.z);
    light_field[idx] = intensity;
}

// === Cell Occupancy Grid Builder ===
// Separate entry point to build cell_occupancy from cell positions
// This clears the grid first, then each cell atomically increments its voxel

struct OccupancyParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> occupancy_params: OccupancyParams;

@group(0) @binding(1)
var<storage, read> cell_positions: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> cell_count_buf: array<u32>;

@group(0) @binding(3)
var<storage, read_write> cell_occupancy_out: array<atomic<u32>>;

fn world_to_grid(world_pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor((world_pos.x - occupancy_params.grid_origin_x) / occupancy_params.cell_size)),
        i32(floor((world_pos.y - occupancy_params.grid_origin_y) / occupancy_params.cell_size)),
        i32(floor((world_pos.z - occupancy_params.grid_origin_z) / occupancy_params.cell_size)),
    );
}

@compute @workgroup_size(256)
fn build_cell_occupancy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buf[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    let pos = cell_positions[cell_idx].xyz;
    let grid_pos = world_to_grid(pos);
    let res = i32(occupancy_params.grid_resolution);
    
    // Bounds check
    if (grid_pos.x < 0 || grid_pos.x >= res ||
        grid_pos.y < 0 || grid_pos.y >= res ||
        grid_pos.z < 0 || grid_pos.z >= res) {
        return;
    }
    
    let voxel_idx = u32(grid_pos.x) + u32(grid_pos.y) * u32(res) + u32(grid_pos.z) * u32(res) * u32(res);
    atomicAdd(&cell_occupancy_out[voxel_idx], 1u);
}

@compute @workgroup_size(256)
fn clear_occupancy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = occupancy_params.grid_resolution * occupancy_params.grid_resolution * occupancy_params.grid_resolution;
    if (idx >= total) {
        return;
    }
    atomicStore(&cell_occupancy_out[idx], 0u);
}
