// Stage 4: Collision detection using O(1) spatial grid neighbor lookup
// Only processes cells in neighboring 3x3x3 grid cells using spatial_grid_cells
// Workgroup size: 64 threads for cell operations
//
// Algorithm:
// 1. Get this cell's grid coordinates
// 2. For each of 27 neighbor grid cells (3x3x3):
//    - Calculate neighbor grid index
//    - Read count from spatial_grid_counts[neighbor_grid_idx]
//    - Iterate spatial_grid_cells[grid_base + 0..count] for O(1) lookup
//    - Compute collision for each neighbor cell
//
// This is O(27 * max_cells_per_grid) = O(27 * 32) = O(864) per cell,
// which is constant time regardless of total cell count.

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<u32>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

// Sorted cell indices by grid cell (16 cells per grid cell)
@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

const GRID_RESOLUTION: i32 = 64;
const MAX_CELLS_PER_GRID: u32 = 16u;
const PI: f32 = 3.14159265359;

fn calculate_radius_from_mass(mass: f32) -> f32 {
    let volume = mass / 1.0;
    return pow(volume * 3.0 / (4.0 * PI), 1.0 / 3.0);
}

fn grid_coords_to_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION);
}

fn grid_index_to_coords(grid_idx: u32) -> vec3<i32> {
    let res = GRID_RESOLUTION;
    let z = i32(grid_idx) / (res * res);
    let y = (i32(grid_idx) - z * res * res) / res;
    let x = i32(grid_idx) - z * res * res - y * res;
    return vec3<i32>(x, y, z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    if (cell_idx >= params.cell_count) {
        return;
    }
    
    let pos = positions_in[cell_idx].xyz;
    let mass = positions_in[cell_idx].w;
    let vel = velocities_in[cell_idx].xyz;
    let radius = calculate_radius_from_mass(mass);
    let my_grid_idx = cell_grid_indices[cell_idx];
    let my_grid_coords = grid_index_to_coords(my_grid_idx);
    
    var force = vec3<f32>(0.0);
    
    // Boundary forces - soft spherical boundary
    let dist_from_center = length(pos);
    let boundary_radius = params.world_size * 0.5;
    
    if (dist_from_center > 0.001) {
        let soft_zone = 5.0;
        let soft_zone_start = boundary_radius - soft_zone;
        
        if (dist_from_center > soft_zone_start) {
            let penetration = (dist_from_center - soft_zone_start) / soft_zone;
            let clamped_pen = clamp(penetration, 0.0, 1.0);
            let normal = -pos / dist_from_center;
            force += normal * clamped_pen * clamped_pen * 500.0;
        }
    }
    
    // Cell-cell collision using O(1) spatial grid lookup
    // Iterate through 27 neighbor grid cells (3x3x3)
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nx = my_grid_coords.x + dx;
                let ny = my_grid_coords.y + dy;
                let nz = my_grid_coords.z + dz;
                
                // Skip out-of-bounds grid cells
                if (nx < 0 || nx >= GRID_RESOLUTION ||
                    ny < 0 || ny >= GRID_RESOLUTION ||
                    nz < 0 || nz >= GRID_RESOLUTION) {
                    continue;
                }
                
                let neighbor_grid_idx = grid_coords_to_index(nx, ny, nz);
                let cell_count_in_grid = spatial_grid_counts[neighbor_grid_idx];
                
                if (cell_count_in_grid == 0u) {
                    continue;
                }
                
                // O(1) lookup: iterate only cells in this grid cell via spatial_grid_cells
                let grid_base_offset = neighbor_grid_idx * MAX_CELLS_PER_GRID;
                
                for (var i = 0u; i < cell_count_in_grid; i++) {
                    let other_idx = spatial_grid_cells[grid_base_offset + i];
                    
                    if (other_idx == cell_idx) {
                        continue;
                    }
                    
                    let other_pos = positions_in[other_idx].xyz;
                    let other_mass = positions_in[other_idx].w;
                    let other_radius = calculate_radius_from_mass(other_mass);
                    
                    let delta = pos - other_pos;
                    let dist = length(delta);
                    let min_dist = radius + other_radius;
                    
                    if (dist < min_dist) {
                        let penetration = min_dist - dist;
                        
                        // Handle cells at same position - use deterministic separation direction
                        var normal: vec3<f32>;
                        if (dist > 0.0001) {
                            normal = delta / dist;
                        } else {
                            // Cells at same position - use cell index to determine separation direction
                            // This ensures deterministic behavior
                            if (cell_idx > other_idx) {
                                normal = vec3<f32>(1.0, 0.0, 0.0);
                            } else {
                                normal = vec3<f32>(-1.0, 0.0, 0.0);
                            }
                        }
                        
                        // Spring force with damping
                        let other_vel = velocities_in[other_idx].xyz;
                        let relative_vel = vel - other_vel;
                        let damping = dot(relative_vel, normal) * 0.5;
                        
                        force += normal * (penetration * params.boundary_stiffness - damping);
                    }
                }
            }
        }
    }
    
    // No gravity in this simulation (cells float in fluid)
    // force.y -= params.gravity * mass;
    
    // Calculate new velocity from force
    let acceleration = force / mass;
    let new_vel = vel + acceleration * params.delta_time * params.acceleration_damping;
    
    // Write to output buffers
    positions_out[cell_idx] = vec4<f32>(pos, mass);
    velocities_out[cell_idx] = vec4<f32>(new_vel, 0.0);
}
