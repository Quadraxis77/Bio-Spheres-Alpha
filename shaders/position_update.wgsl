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

// Rotations (group 1) - propagate from input to output
@group(1) @binding(0)
var<storage, read> rotations_in: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read_write> rotations_out: array<vec4<f32>>;

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
    
    // Propagate rotation from input to output (no rotation physics yet)
    rotations_out[cell_idx] = rotations_in[cell_idx];
}