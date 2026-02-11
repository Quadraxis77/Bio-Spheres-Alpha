// Apply PBD corrections to positions_out and clear accumulators
// Dispatched after each adhesion iteration so the next iteration sees updated positions.
// Workgroup size: 256 threads for optimal GPU occupancy

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

@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// PBD position correction accumulators (group 1)
@group(1) @binding(0)
var<storage, read_write> pbd_pos_x: array<atomic<i32>>;

@group(1) @binding(1)
var<storage, read_write> pbd_pos_y: array<atomic<i32>>;

@group(1) @binding(2)
var<storage, read_write> pbd_pos_z: array<atomic<i32>>;

// PBD rotation correction accumulators
@group(1) @binding(3)
var<storage, read_write> pbd_rot_x: array<atomic<i32>>;

@group(1) @binding(4)
var<storage, read_write> pbd_rot_y: array<atomic<i32>>;

@group(1) @binding(5)
var<storage, read_write> pbd_rot_z: array<atomic<i32>>;

const FIXED_POINT_SCALE: f32 = 1000.0;

fn fixed_to_float(v: i32) -> f32 {
    return f32(v) / FIXED_POINT_SCALE;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Read and clear PBD position corrections atomically
    let dx = atomicExchange(&pbd_pos_x[cell_idx], 0);
    let dy = atomicExchange(&pbd_pos_y[cell_idx], 0);
    let dz = atomicExchange(&pbd_pos_z[cell_idx], 0);

    // Read and clear PBD rotation corrections atomically
    let rx = atomicExchange(&pbd_rot_x[cell_idx], 0);
    let ry = atomicExchange(&pbd_rot_y[cell_idx], 0);
    let rz = atomicExchange(&pbd_rot_z[cell_idx], 0);

    // Apply position corrections to positions_out
    let pos = positions_out[cell_idx];
    let correction = vec3<f32>(
        fixed_to_float(dx),
        fixed_to_float(dy),
        fixed_to_float(dz)
    );
    positions_out[cell_idx] = vec4<f32>(pos.xyz + correction, pos.w);

    // Rotation corrections: re-accumulate so velocity_update can apply them once at end.
    atomicAdd(&pbd_rot_x[cell_idx], rx);
    atomicAdd(&pbd_rot_y[cell_idx], ry);
    atomicAdd(&pbd_rot_z[cell_idx], rz);
}
