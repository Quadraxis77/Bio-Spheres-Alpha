// Stage 4: Collision detection using O(1) spatial grid neighbor lookup
// Only processes cells in neighboring 3x3x3 grid cells using spatial_grid_cells
// Workgroup size: 256 threads for optimal GPU occupancy
//
// Optimizations applied:
// - Pre-computed 27 neighbor grid indices (reduces redundant calculations)
// - Branchless boundary checks using clamp + select (reduces warp divergence)
// - 256-thread workgroups (8 warps = optimal GPU scheduling)
//
// Algorithm:
// 1. Get this cell's grid coordinates
// 2. Pre-compute all 27 neighbor grid indices and counts
// 3. For each neighbor grid cell:
//    - Read count from pre-computed array
//    - Iterate spatial_grid_cells[grid_base + 0..count] for O(1) lookup
//    - Compute collision for each neighbor cell
//
// This is O(27 * max_cells_per_grid) = O(27 * 16) = O(432) per cell,
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
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
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

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<u32>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

// Sorted cell indices by grid cell (16 cells per grid cell)
@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

// Per-cell membrane stiffness from genome mode
@group(1) @binding(4)
var<storage, read> stiffnesses: array<f32>;

// Force accumulation buffers (group 2) - atomic i32 for multi-adhesion accumulation
@group(2) @binding(0)
var<storage, read_write> force_accum_x: array<atomic<i32>>;

@group(2) @binding(1)
var<storage, read_write> force_accum_y: array<atomic<i32>>;

@group(2) @binding(2)
var<storage, read_write> force_accum_z: array<atomic<i32>>;

const GRID_RESOLUTION: i32 = 64;
const MAX_CELLS_PER_GRID: u32 = 16u;
const PI: f32 = 3.14159265359;
const FIXED_POINT_SCALE: f32 = 1000.0;

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let pos = positions_in[cell_idx].xyz;
    let mass = positions_in[cell_idx].w;
    let vel = velocities_in[cell_idx].xyz;
    let radius = calculate_radius_from_mass(mass);
    let my_stiffness = stiffnesses[cell_idx];
    let my_grid_idx = cell_grid_indices[cell_idx];
    let my_grid_coords = grid_index_to_coords(my_grid_idx);
    
    var force = vec3<f32>(0.0);
    
    // Boundary forces - soft spherical boundary (branchless)
    let dist_from_center = length(pos);
    let boundary_radius = params.world_size * 0.5;
    let soft_zone = 5.0;
    let soft_zone_start = boundary_radius - soft_zone;
    let penetration = (dist_from_center - soft_zone_start) / soft_zone;
    let clamped_pen = clamp(penetration, 0.0, 1.0);
    // Use select to avoid branch - safe_dist avoids division by zero
    let safe_dist = max(dist_from_center, 0.001);
    let normal = -pos / safe_dist;
    // Only apply force when penetration > 0 (clamped_pen handles this naturally)
    force += normal * clamped_pen * clamped_pen * 500.0;
    
    // Pre-compute all 27 neighbor grid indices and counts to reduce redundant calculations
    // and improve memory access patterns
    var neighbor_indices: array<u32, 27>;
    var neighbor_counts: array<u32, 27>;
    var n_idx = 0u;
    
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                // Use clamp instead of branch for boundary check
                let nx = clamp(my_grid_coords.x + dx, 0, GRID_RESOLUTION - 1);
                let ny = clamp(my_grid_coords.y + dy, 0, GRID_RESOLUTION - 1);
                let nz = clamp(my_grid_coords.z + dz, 0, GRID_RESOLUTION - 1);
                
                // Check if this is actually a valid neighbor (not clamped)
                let is_valid = (my_grid_coords.x + dx >= 0) && (my_grid_coords.x + dx < GRID_RESOLUTION) &&
                               (my_grid_coords.y + dy >= 0) && (my_grid_coords.y + dy < GRID_RESOLUTION) &&
                               (my_grid_coords.z + dz >= 0) && (my_grid_coords.z + dz < GRID_RESOLUTION);
                
                let neighbor_grid_idx = grid_coords_to_index(nx, ny, nz);
                neighbor_indices[n_idx] = neighbor_grid_idx;
                // Use select to zero out invalid neighbors (branchless)
                neighbor_counts[n_idx] = select(0u, min(spatial_grid_counts[neighbor_grid_idx], MAX_CELLS_PER_GRID), is_valid);
                n_idx++;
            }
        }
    }
    
    // Cell-cell collision using pre-computed neighbor data
    for (var n = 0u; n < 27u; n++) {
        let cell_count_in_grid = neighbor_counts[n];
        if (cell_count_in_grid == 0u) {
            continue;
        }
        
        let grid_base_offset = neighbor_indices[n] * MAX_CELLS_PER_GRID;
        
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
                let coll_penetration = min_dist - dist;
                
                // Handle cells at same position - use deterministic separation direction
                var coll_normal: vec3<f32>;
                if (dist > 0.0001) {
                    coll_normal = delta / dist;
                } else {
                    // Cells at same position - use cell index to determine separation direction
                    // This ensures deterministic behavior
                    if (cell_idx > other_idx) {
                        coll_normal = vec3<f32>(1.0, 0.0, 0.0);
                    } else {
                        coll_normal = vec3<f32>(-1.0, 0.0, 0.0);
                    }
                }
                
                // Use average of both cells' membrane stiffness for collision response
                let other_stiffness = stiffnesses[other_idx];
                let combined_stiffness = (my_stiffness + other_stiffness) * 0.5;
                
                // Spring force with damping
                let other_vel = velocities_in[other_idx].xyz;
                let relative_vel = vel - other_vel;
                let damping = dot(relative_vel, coll_normal) * 0.5;
                
                force += coll_normal * (coll_penetration * combined_stiffness - damping);
            }
        }
    }
    
    // No gravity in this simulation (cells float in fluid)
    // force.y -= params.gravity * mass;
    
    // Accumulate forces to force buffer using atomics (matching CPU pipeline)
    // Forces will be integrated later in position_update shader using Verlet integration
    // This ensures collision and adhesion forces are combined before integration
    atomicAdd(&force_accum_x[cell_idx], i32(force.x * FIXED_POINT_SCALE));
    atomicAdd(&force_accum_y[cell_idx], i32(force.y * FIXED_POINT_SCALE));
    atomicAdd(&force_accum_z[cell_idx], i32(force.z * FIXED_POINT_SCALE));
    
    // Copy position and velocity to output (no modification here)
    // Velocity will be updated in position_update after all forces are accumulated
    positions_out[cell_idx] = vec4<f32>(pos, mass);
    velocities_out[cell_idx] = vec4<f32>(vel, 0.0);
}
