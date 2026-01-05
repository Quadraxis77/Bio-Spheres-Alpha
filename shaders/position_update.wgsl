// Stage 5: Position integration using Verlet integration
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    if (cell_idx >= params.cell_count) {
        return;
    }
    
    let pos = positions_out[cell_idx].xyz;
    let mass = positions_out[cell_idx].w;
    let vel = velocities_out[cell_idx].xyz;
    
    // Verlet integration: x(t+dt) = x(t) + v(t)*dt
    let new_pos = pos + vel * params.delta_time;
    
    // Clamp to boundary
    let boundary_radius = params.world_size * 0.5;
    let dist = length(new_pos);
    var final_pos = new_pos;
    
    if (dist > boundary_radius) {
        final_pos = normalize(new_pos) * boundary_radius;
    }
    
    // Write updated position
    positions_out[cell_idx] = vec4<f32>(final_pos, mass);
}