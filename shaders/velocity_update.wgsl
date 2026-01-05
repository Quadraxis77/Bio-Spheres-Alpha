// Stage 6: Velocity integration with damping
// Workgroup size: 64 threads for cell operations

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let vel = velocities_out[cell_idx].xyz;
    
    // Apply velocity damping
    let damping_factor = pow(params.acceleration_damping, params.delta_time * 100.0);
    let new_vel = vel * damping_factor;
    
    // Clamp velocity to prevent instability
    let max_speed = 100.0;
    let speed = length(new_vel);
    var final_vel = new_vel;
    
    if (speed > max_speed) {
        final_vel = normalize(new_vel) * max_speed;
    }
    
    // Write updated velocity
    velocities_out[cell_idx] = vec4<f32>(final_vel, 0.0);
}