// Swim Force Shader - Applies thrust force for Flagellocyte cells
//
// This shader adds swim force to the force accumulation buffers based on:
// - Cell type (only Flagellocytes have swim force)
// - Mode's swim_force setting (0.0 - 1.0)
// - Cell's rotation (forward direction = +Z in local space)
//
// Must run after clear_forces and before position_update.
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

// Mode properties (per-mode settings from genome)
struct ModeProperties {
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    membrane_stiffness: f32,
    split_interval: f32,
    split_mass: f32,
    swim_force: f32,
    _pad0: f32,
    _pad1: f32,
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

// Force accumulation buffers (group 1) - atomic i32 for multi-source accumulation
@group(1) @binding(0)
var<storage, read_write> force_accum_x: array<atomic<i32>>;

@group(1) @binding(1)
var<storage, read_write> force_accum_y: array<atomic<i32>>;

@group(1) @binding(2)
var<storage, read_write> force_accum_z: array<atomic<i32>>;

// Rotations for determining forward direction
@group(1) @binding(3)
var<storage, read> rotations: array<vec4<f32>>;

// Cell data (group 2)
@group(2) @binding(0)
var<storage, read> mode_indices: array<u32>;

@group(2) @binding(1)
var<storage, read> cell_types: array<u32>;

@group(2) @binding(2)
var<storage, read> mode_properties: array<ModeProperties>;

const FIXED_POINT_SCALE: f32 = 1000.0;
const SWIM_FORCE_MULTIPLIER: f32 = 50.0; // Scale swim_force (0-1) to actual force magnitude

// Convert float to fixed-point i32 for atomic accumulation
fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

// Rotate a vector by a quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Check if thrust force is enabled globally
    if (params.enable_thrust_force == 0) {
        return;
    }
    
    // Get cell type - only Flagellocytes (type 1) have swim force
    let cell_type = cell_types[cell_idx];
    if (cell_type != 1u) {
        return;
    }
    
    // Get mode properties
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_properties)) {
        return;
    }
    let mode = mode_properties[mode_idx];
    
    // Skip if no swim force configured
    if (mode.swim_force <= 0.0) {
        return;
    }
    
    // Get cell rotation and calculate forward direction
    // Forward is +Z in local space (matching preview physics)
    let rotation = rotations[cell_idx];
    let forward = quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));
    
    // Calculate swim force vector
    let force_magnitude = mode.swim_force * SWIM_FORCE_MULTIPLIER;
    let swim_force = forward * force_magnitude;
    
    // Add to force accumulation buffers (atomic for thread safety)
    atomicAdd(&force_accum_x[cell_idx], float_to_fixed(swim_force.x));
    atomicAdd(&force_accum_y[cell_idx], float_to_fixed(swim_force.y));
    atomicAdd(&force_accum_z[cell_idx], float_to_fixed(swim_force.z));
}
