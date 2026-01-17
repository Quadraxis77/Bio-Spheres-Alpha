// Stage 6: Position and velocity integration
// Simple Verlet integration for maximum performance
// Workgroup size: 256 threads for optimal GPU occupancy
//
// Verlet integration (matching CPU exactly):
//   velocity_change = 0.5 * (old_acceleration + new_acceleration) * dt
//   new_velocity = (velocity + velocity_change) * damping_factor
//   new_position = position + new_velocity * dt

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

const FIXED_POINT_SCALE: f32 = 1000.0;

// Convert fixed-point i32 back to float
fn fixed_to_float(v: i32) -> f32 {
    return f32(v) / FIXED_POINT_SCALE;
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
    
    // Simple position update
    let new_pos = pos + new_vel * params.delta_time;
    
    // Simple boundary collision
    let boundary_radius = params.world_size * 0.5;
    var final_pos = new_pos;
    var final_vel = new_vel;
    
    // Check if new position would violate boundary
    let dist_from_center = length(new_pos);
    if (dist_from_center > boundary_radius) {
        // Push back inside boundary
        final_pos = normalize(new_pos) * boundary_radius * 0.99;
        
        // Reflect velocity with damping
        let inward_dir = -normalize(new_pos);
        let vel_normal = dot(new_vel, inward_dir);
        if (vel_normal < 0.0) {
            let reflection = new_vel - 2.0 * vel_normal * inward_dir;
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
