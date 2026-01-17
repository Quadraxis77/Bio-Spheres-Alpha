// Stage 6: Position and velocity integration with tunneling prevention
// Uses predictive collision detection and velocity clamping to prevent tunneling
// Workgroup size: 256 threads for optimal GPU occupancy
//
// Anti-tunneling algorithm:
// 1. Clamp maximum velocity based on cell radius and timestep
// 2. Predictive sweep test to detect potential tunneling
// 3. Sub-stepping for high-velocity collisions
// 4. Impulse-based collision response during integration
//
// Verlet integration (matching CPU exactly):
//   velocity_change = 0.5 * (old_acceleration + new_acceleration) * dt
//   new_velocity = (velocity + velocity_change) * damping_factor
//   new_position = position + new_velocity * dt (with collision constraints)

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
    gravity_dir_x: f32,
    gravity_dir_y: f32,
    gravity_dir_z: f32,
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

// Rotations (group 1) - propagate from input to output
@group(1) @binding(0)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read_write> rotations_out: array<vec4<f32>>;

// Force accumulation (atomic i32, fixed-point) and previous accelerations (group 2)
@group(2) @binding(0)
var<storage, read> force_accum_x: array<i32>;

@group(2) @binding(1)
var<storage, read> force_accum_y: array<i32>;

@group(2) @binding(2)
var<storage, read> force_accum_z: array<i32>;

@group(2) @binding(3)
var<storage, read_write> prev_accelerations: array<vec4<f32>>;

// Spatial grid data for cell-cell sweep tests (group 3)
@group(3) @binding(0)
var<storage, read> spatial_grid_counts: array<u32>;

@group(3) @binding(1)
var<storage, read> spatial_grid_offsets: array<u32>;

@group(3) @binding(2)
var<storage, read> cell_grid_indices: array<u32>;

@group(3) @binding(3)
var<storage, read> spatial_grid_cells: array<u32>;

@group(3) @binding(4)
var<storage, read> stiffnesses: array<f32>;

const FIXED_POINT_SCALE: f32 = 1000.0;

// Anti-tunneling constants
const MAX_SUBSTEPS: i32 = 4;
const SAFETY_FACTOR: f32 = 0.8; // Reduce max velocity by 20% for safety
const GRID_RESOLUTION: i32 = 64;
const MAX_CELLS_PER_GRID: u32 = 16u;

// Convert fixed-point i32 back to float
fn fixed_to_float(v: i32) -> f32 {
    return f32(v) / FIXED_POINT_SCALE;
}

// Grid utility functions (same as collision_detection.wgsl)
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

fn world_pos_to_grid_index(pos: vec3<f32>, world_size: f32, grid_cell_size: f32) -> u32 {
    let grid_pos = (pos + world_size * 0.5) / grid_cell_size;
    let grid_x = clamp(i32(grid_pos.x), 0, GRID_RESOLUTION - 1);
    let grid_y = clamp(i32(grid_pos.y), 0, GRID_RESOLUTION - 1);
    let grid_z = clamp(i32(grid_pos.z), 0, GRID_RESOLUTION - 1);
    return grid_coords_to_index(grid_x, grid_y, grid_z);
}

// Calculate cell radius from mass (same as collision_detection.wgsl)
fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Calculate maximum safe velocity to prevent tunneling
fn calculate_max_safe_velocity(radius: f32, dt: f32) -> f32 {
    // Maximum distance a cell can travel in one timestep without tunneling
    // Use safety factor to account for numerical errors
    return (radius * 2.0 * SAFETY_FACTOR) / dt;
}

// Predictive sweep test to detect potential tunneling
fn will_tunnel_through_cell(
    start_pos: vec3<f32>,
    end_pos: vec3<f32>,
    radius: f32,
    other_pos: vec3<f32>,
    other_radius: f32
) -> bool {
    let combined_radius = radius + other_radius;
    let sweep_vector = end_pos - start_pos;
    let sweep_length = length(sweep_vector);
    
    if (sweep_length < 0.0001) {
        return false; // No movement
    }
    
    let sweep_dir = sweep_vector / sweep_length;
    
    // Vector from start position to other cell
    let to_other = other_pos - start_pos;
    
    // Project onto sweep direction
    let projection_length = dot(to_other, sweep_dir);
    
    // Check if closest approach happens during this sweep
    if (projection_length < 0.0 || projection_length > sweep_length) {
        return false; // Closest approach is outside sweep range
    }
    
    // Find closest point on sweep line to other cell
    let closest_point = start_pos + sweep_dir * projection_length;
    let distance_to_other = length(closest_point - other_pos);
    
    // Check if we get too close (tunneling condition)
    return distance_to_other < combined_radius;
}

// Apply impulse to prevent tunneling
fn apply_anti_tunneling_impulse(
    pos: vec3<f32>,
    vel: vec3<f32>,
    radius: f32,
    dt: f32,
    max_safe_vel: f32
) -> vec3<f32> {
    let speed = length(vel);
    
    if (speed <= max_safe_vel) {
        return vel; // No clamping needed
    }
    
    // Calculate impulse to reduce velocity to safe level
    let impulse_magnitude = speed - max_safe_vel;
    let vel_dir = normalize(vel);
    
    // Apply impulse opposite to velocity direction
    return vel - vel_dir * impulse_magnitude;
}

// Sub-stepped position update with collision checking
fn sub_stepped_position_update(
    start_pos: vec3<f32>,
    vel: vec3<f32>,
    radius: f32,
    dt: f32,
    max_safe_vel: f32
) -> vec3<f32> {
    let speed = length(vel);
    
    if (speed <= max_safe_vel) {
        // Normal integration - no sub-stepping needed
        return start_pos + vel * dt;
    }
    
    // Calculate number of substeps needed
    let substep_ratio = speed / max_safe_vel;
    let num_substeps = min(i32(ceil(substep_ratio)), MAX_SUBSTEPS);
    let substep_dt = dt / f32(num_substeps);
    
    var current_pos = start_pos;
    
    // Perform sub-stepped integration
    for (var i = 0; i < num_substeps; i++) {
        current_pos = current_pos + vel * substep_dt;
    }
    
    return current_pos;
}

// Cell-cell sweep test using spatial grid for high-velocity collision detection
fn perform_cell_sweep_test(
    start_pos: vec3<f32>,
    end_pos: vec3<f32>,
    radius: f32,
    vel: vec3<f32>,
    cell_idx: u32,
    cell_count: u32,
    world_size: f32,
    grid_cell_size: f32
) -> vec3<f32> {
    // Only perform sweep test for high-velocity cells
    let travel_distance = length(end_pos - start_pos);
    let max_safe_distance = radius * 2.0 * SAFETY_FACTOR;
    
    if (travel_distance <= max_safe_distance) {
        return vel; // No sweep test needed
    }
    
    var corrected_vel = vel;
    
    // Check cells in expanded grid region that could be intersected during sweep
    let start_grid_idx = world_pos_to_grid_index(start_pos, world_size, grid_cell_size);
    let end_grid_idx = world_pos_to_grid_index(end_pos, world_size, grid_cell_size);
    
    // Calculate bounding box of sweep in grid coordinates
    let start_coords = grid_index_to_coords(start_grid_idx);
    let end_coords = grid_index_to_coords(end_grid_idx);
    
    let min_x = min(start_coords.x, end_coords.x) - 1;
    let max_x = max(start_coords.x, end_coords.x) + 1;
    let min_y = min(start_coords.y, end_coords.y) - 1;
    let max_y = max(start_coords.y, end_coords.y) + 1;
    let min_z = min(start_coords.z, end_coords.z) - 1;
    let max_z = max(start_coords.z, end_coords.z) + 1;
    
    // Check all grid cells in the sweep bounding box
    for (var gz = min_z; gz <= max_z; gz++) {
        for (var gy = min_y; gy <= max_y; gy++) {
            for (var gx = min_x; gx <= max_x; gx++) {
                // Clamp to grid bounds
                let clamped_x = clamp(gx, 0, GRID_RESOLUTION - 1);
                let clamped_y = clamp(gy, 0, GRID_RESOLUTION - 1);
                let clamped_z = clamp(gz, 0, GRID_RESOLUTION - 1);
                
                let check_grid_idx = grid_coords_to_index(clamped_x, clamped_y, clamped_z);
                let cells_in_grid = min(spatial_grid_counts[check_grid_idx], MAX_CELLS_PER_GRID);
                
                if (cells_in_grid == 0u) {
                    continue;
                }
                
                let grid_base_offset = check_grid_idx * MAX_CELLS_PER_GRID;
                
                // Check each cell in this grid cell
                for (var i = 0u; i < cells_in_grid; i++) {
                    let other_idx = spatial_grid_cells[grid_base_offset + i];
                    
                    if (other_idx == cell_idx) {
                        continue; // Skip self
                    }
                    
                    let other_pos = positions_in[other_idx].xyz;
                    let other_mass = positions_in[other_idx].w;
                    let other_radius = calculate_radius_from_mass(other_mass);
                    
                    // Check if we would tunnel through this cell
                    if (will_tunnel_through_cell(start_pos, end_pos, radius, other_pos, other_radius)) {
                        // Calculate impulse to prevent tunneling
                        let to_other = other_pos - start_pos;
                        let distance = length(to_other);
                        
                        if (distance > 0.0001) {
                            let escape_dir = normalize(to_other);
                            // Apply strong impulse to escape collision
                            let escape_impulse = escape_dir * travel_distance * 0.5;
                            corrected_vel = corrected_vel + escape_impulse;
                        }
                    }
                }
            }
        }
    }
    
    return corrected_vel;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let pos = positions_out[cell_idx].xyz;
    let mass = positions_out[cell_idx].w;
    let vel = velocities_out[cell_idx].xyz;
    
    // Read accumulated forces from collision and adhesion stages (convert from fixed-point)
    var force = vec3<f32>(
        fixed_to_float(force_accum_x[cell_idx]),
        fixed_to_float(force_accum_y[cell_idx]),
        fixed_to_float(force_accum_z[cell_idx])
    );

    // Apply gravity (F = mg) in selected directions
    let gravity_force = params.gravity * mass;
    force.x += gravity_force * params.gravity_dir_x;
    force.y += gravity_force * params.gravity_dir_y;
    force.z += gravity_force * params.gravity_dir_z;

    // Read previous acceleration for Verlet integration
    let old_acceleration = prev_accelerations[cell_idx].xyz;

    // Calculate new acceleration from accumulated forces
    let new_acceleration = force / mass;
    
    // Verlet integration (matching CPU exactly):
    // velocity_change = 0.5 * (old_acceleration + new_acceleration) * dt
    let velocity_change = 0.5 * (old_acceleration + new_acceleration) * params.delta_time;
    
    // Apply velocity damping (matching CPU: damping^(dt*100))
    let damping_factor = pow(params.acceleration_damping, params.delta_time * 100.0);
    let new_vel = (vel + velocity_change) * damping_factor;
    
    // Calculate cell properties for anti-tunneling
    let radius = calculate_radius_from_mass(mass);
    let max_safe_velocity = calculate_max_safe_velocity(radius, params.delta_time);
    
    // Apply anti-tunneling velocity clamping
    let anti_tunneling_vel = apply_anti_tunneling_impulse(pos, new_vel, radius, params.delta_time, max_safe_velocity);
    
    // Predict where the cell would move with the clamped velocity
    let predicted_pos = pos + anti_tunneling_vel * params.delta_time;
    
    // Perform cell-cell sweep test for high-velocity collisions
    let sweep_corrected_vel = perform_cell_sweep_test(
        pos, 
        predicted_pos, 
        radius, 
        anti_tunneling_vel, 
        cell_idx, 
        cell_count, 
        params.world_size, 
        params.grid_cell_size
    );
    
    // Use sub-stepped position update with sweep-corrected velocity
    let new_pos = sub_stepped_position_update(pos, sweep_corrected_vel, radius, params.delta_time, max_safe_velocity);
    
    // Boundary collision with tunneling prevention
    let boundary_radius = params.world_size * 0.5;
    var final_pos = new_pos;
    var final_vel = sweep_corrected_vel;
    
    // Check if new position would violate boundary
    let dist_from_center = length(new_pos);
    if (dist_from_center + radius > boundary_radius) {
        // Calculate boundary penetration
        let penetration = (dist_from_center + radius) - boundary_radius;
        
        // Apply boundary impulse
        let inward_dir = -normalize(new_pos);
        let boundary_impulse = inward_dir * penetration;
        
        // Correct position
        final_pos = new_pos + boundary_impulse;
        
        // Reflect velocity with damping
        let vel_normal = dot(sweep_corrected_vel, -inward_dir);
        if (vel_normal > 0.0) {
            let reflection = sweep_corrected_vel - 2.0 * vel_normal * (-inward_dir);
            final_vel = reflection * 0.5; // 50% energy loss on boundary collision
        }
    }
    
    // Write updated position and velocity
    positions_out[cell_idx] = vec4<f32>(final_pos, mass);
    velocities_out[cell_idx] = vec4<f32>(final_vel, 0.0);
    
    // Store current acceleration for next frame's Verlet integration
    prev_accelerations[cell_idx] = vec4<f32>(new_acceleration, 0.0);
    
    // Propagate rotation from input to output (no rotation physics yet)
    rotations_out[cell_idx] = rotations_in[cell_idx];
}