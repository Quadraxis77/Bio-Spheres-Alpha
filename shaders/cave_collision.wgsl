// Cave Collision SDF Shader
// 
// Implements GPU-based SDF collision detection using Voronoi-based cave generation.
// Uses 3D Voronoi cells with random wall thresholds for clear wall/air regions.
//
// Voronoi cave approach:
// - Each Voronoi cell has a random "wall threshold" (0-1)
// - If wall_density > wall_threshold, the cell is a CAVE OBSTACLE (solid)
// - Caves are SOLID OBSTACLES that cells must avoid
// - Cells stay in OPEN SPACE (non-cave cells)
// - When in a cave cell, cells are pushed OUT into open space

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

@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0) var<uniform> cave_params: CaveParams;

// Constants
const EPSILON: f32 = 0.0001;

// Hash function for Voronoi cell generation
fn hash3(x: i32, y: i32, z: i32, seed: u32) -> vec3<f32> {
    var h = seed;
    h = h * 374761393u + u32(x);
    h = h * 668265263u + u32(y);
    h = h * 1274126177u + u32(z);
    h ^= h >> 13u;
    h = h * 1274126177u;
    h ^= h >> 16u;
    
    // Generate 3 random values
    let x_rand = f32(h) / f32(0xFFFFFFFFu);
    h = h * 1664525u + 1013904223u;
    let y_rand = f32(h) / f32(0xFFFFFFFFu);
    h = h * 1664525u + 1013904223u;
    let z_rand = f32(h) / f32(0xFFFFFFFFu);
    
    return vec3<f32>(x_rand, y_rand, z_rand);
}

// Hash function for single value (wall threshold)
fn hash1(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    var h = seed + 12345u; // Different seed for threshold
    h = h * 374761393u + u32(x);
    h = h * 668265263u + u32(y);
    h = h * 1274126177u + u32(z);
    h ^= h >> 13u;
    h = h * 1274126177u;
    h ^= h >> 16u;
    
    return f32(h) / f32(0xFFFFFFFFu);
}

// 3D Voronoi noise - returns (distance to nearest point, is_current_cell_wall)
fn voronoi_cave(pos: vec3<f32>) -> vec2<f32> {
    let cell_size = cave_params.scale;
    let scaled_pos = pos / cell_size;
    
    // Current cell
    let cell = vec3<i32>(floor(scaled_pos));
    let local_pos = fract(scaled_pos);
    
    var min_dist = 10000.0;
    var closest_is_wall = 0.0;
    
    // Check neighboring cells (3x3x3 grid)
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor = cell + vec3<i32>(dx, dy, dz);
                
                // Get random point within this cell
                let rand = hash3(neighbor.x, neighbor.y, neighbor.z, cave_params.seed);
                let point = vec3<f32>(f32(dx), f32(dy), f32(dz)) + rand;
                
                // Distance to this Voronoi point
                let diff = point - local_pos;
                let dist = length(diff);
                
                // Check if this cell is a wall (cave obstacle)
                let wall_threshold = hash1(neighbor.x, neighbor.y, neighbor.z, cave_params.seed);
                let cell_is_wall = cave_params.density > wall_threshold;
                
                // Track closest cell (wall or not)
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_is_wall = f32(cell_is_wall);
                }
            }
        }
    }
    
    // Convert distance back to world space
    min_dist = min_dist * cell_size;
    
    return vec2<f32>(min_dist, closest_is_wall);
}

// Sample cave density using Voronoi-based walls
fn sample_cave_density(pos: vec3<f32>) -> f32 {
    // Distance from world center (spherical constraint)
    let dist_from_center = length(pos - cave_params.world_center);
    let sphere_sdf = dist_from_center - cave_params.world_radius;
    
    // Outside sphere = solid (high density)
    if (sphere_sdf > 0.0) {
        return 1.0;
    }
    
    // Inside sphere: use Voronoi-based cave system
    let voronoi = voronoi_cave(pos);
    let dist_to_wall = voronoi.x;
    let has_nearby_wall = voronoi.y;
    
    // If no walls nearby, it's open air (cave space)
    if (has_nearby_wall < 0.5) {
        return cave_params.threshold - 1.0; // Well below threshold = air
    }
    
    // Distance-based density relative to threshold
    let wall_thickness = cave_params.scale * 0.5; // Half the cell size
    
    if (dist_to_wall < wall_thickness) {
        // Inside wall region (above threshold = solid)
        return cave_params.threshold + (1.0 - dist_to_wall / wall_thickness) * 0.5;
    } else {
        // Outside wall region (below threshold = air/cave)
        return cave_params.threshold - 0.5;
    }
}

// Compute SDF gradient (normal) using central differences
fn compute_sdf_gradient(pos: vec3<f32>, h: f32) -> vec3<f32> {
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    
    let grad_x = sample_cave_density(pos + dx) - sample_cave_density(pos - dx);
    let grad_y = sample_cave_density(pos + dy) - sample_cave_density(pos - dy);
    let grad_z = sample_cave_density(pos + dz) - sample_cave_density(pos - dz);
    
    let grad = vec3<f32>(grad_x, grad_y, grad_z);
    let len = length(grad);
    
    // Avoid division by zero
    if (len < 0.0001) {
        return vec3<f32>(0.0, 1.0, 0.0); // Default up direction
    }
    
    return grad / len;
}

// Find nearest Voronoi point (returns point position and whether it's a wall)
fn find_nearest_voronoi_point(pos: vec3<f32>) -> vec4<f32> {
    let cell_size = cave_params.scale;
    let scaled_pos = pos / cell_size;
    let cell = vec3<i32>(floor(scaled_pos));
    let local_pos = fract(scaled_pos);
    
    var min_dist = 10000.0;
    var nearest_point = vec3<f32>(0.0);
    var is_wall = 0.0;
    
    // Check 3x3x3 neighborhood
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor = cell + vec3<i32>(dx, dy, dz);
                let rand = hash3(neighbor.x, neighbor.y, neighbor.z, cave_params.seed);
                let point = vec3<f32>(f32(dx), f32(dy), f32(dz)) + rand;
                let diff = point - local_pos;
                let dist = length(diff);
                
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_point = (vec3<f32>(f32(neighbor.x), f32(neighbor.y), f32(neighbor.z)) + rand) * cell_size;
                    
                    let wall_threshold = hash1(neighbor.x, neighbor.y, neighbor.z, cave_params.seed);
                    is_wall = f32(cave_params.density > wall_threshold);
                }
            }
        }
    }
    
    return vec4<f32>(nearest_point, is_wall);
}

// Solve SDF-based collision constraint
fn solve_cave_collision(cell_idx: u32, pos: vec3<f32>, radius: f32, mass: f32, dt: f32) -> vec3<f32> {
    if (cave_params.collision_enabled == 0u) {
        return pos;
    }
    
    var corrected_pos = pos;
    
    // Multiple substeps for stability
    for (var substep = 0u; substep < cave_params.substeps; substep = substep + 1u) {
        // Find which Voronoi cell we're in
        let voronoi_data = find_nearest_voronoi_point(corrected_pos);
        let voronoi_point = voronoi_data.xyz;
        let is_in_wall_cell = voronoi_data.w > 0.5;
        
        // If we're in a wall cell, push out
        if (is_in_wall_cell) {
            // Normal points from Voronoi center toward cell position (outward)
            let to_cell = corrected_pos - voronoi_point;
            let dist = length(to_cell);
            
            if (dist > 0.001) {
                let normal = to_cell / dist;
                
                // Push cell to the edge of the Voronoi cell + radius
                let target_dist = cave_params.scale * 0.5 + radius;
                let penetration = target_dist - dist;
                
                if (penetration > 0.0) {
                    // Push cell outward
                    let stiffness_factor = cave_params.collision_stiffness / 1000.0;
                    let correction = normal * penetration * stiffness_factor;
                    
                    corrected_pos = corrected_pos + correction;
                    
                    // Apply velocity damping
                    let vel = velocities[cell_idx].xyz;
                    let vel_normal_mag = dot(vel, normal);
                    if (vel_normal_mag < 0.0) {
                        // Remove velocity toward wall center
                        let vel_tangent = vel - vel_normal_mag * normal;
                        velocities[cell_idx] = vec4<f32>(vel_tangent * (1.0 - cave_params.collision_damping), velocities[cell_idx].w);
                    }
                }
            }
        }
        // If in open cell (not wall), no collision
    }
    
    return corrected_pos;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (idx >= cell_count) {
        return;
    }
    
    // Read position and mass (already updated by position_update)
    let pos_mass = positions[idx];
    let pos = pos_mass.xyz;
    let mass = pos_mass.w;
    
    // Skip dead cells
    if (mass <= 0.0) {
        // Keep the dead cell data as-is
        return;
    }
    
    // Calculate radius from mass (mass = 4/3 * pi * r^3)
    let visual_radius = pow(mass * 0.75 / 3.14159265359, 1.0 / 3.0);
    
    // Use visual radius for collision (SDF gives us exact distance)
    let collision_radius = visual_radius;
    
    // Solve cave collision with SDF
    let corrected_pos = solve_cave_collision(idx, pos, collision_radius, mass, params.delta_time);
    
    // Write corrected position back (in-place update)
    positions[idx] = vec4<f32>(corrected_pos, mass);
}
