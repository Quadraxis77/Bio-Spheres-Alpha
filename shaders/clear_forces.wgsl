// Stage 3.5: Clear force accumulation buffers
// Must run before collision detection and adhesion physics
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

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Force accumulation buffers (group 1) - atomic i32 for multi-adhesion accumulation
@group(1) @binding(0)
var<storage, read_write> force_accum_x: array<atomic<i32>>;

@group(1) @binding(1)
var<storage, read_write> force_accum_y: array<atomic<i32>>;

@group(1) @binding(2)
var<storage, read_write> force_accum_z: array<atomic<i32>>;

@group(1) @binding(3)
var<storage, read_write> torque_accum_x: array<atomic<i32>>;

@group(1) @binding(4)
var<storage, read_write> torque_accum_y: array<atomic<i32>>;

@group(1) @binding(5)
var<storage, read_write> torque_accum_z: array<atomic<i32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Clear force and torque accumulators to zero
    atomicStore(&force_accum_x[cell_idx], 0);
    atomicStore(&force_accum_y[cell_idx], 0);
    atomicStore(&force_accum_z[cell_idx], 0);
    atomicStore(&torque_accum_x[cell_idx], 0);
    atomicStore(&torque_accum_y[cell_idx], 0);
    atomicStore(&torque_accum_z[cell_idx], 0);
}
