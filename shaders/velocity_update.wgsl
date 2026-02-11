// Stage 7: Angular velocity integration
// Linear velocity is now handled in position_update with Verlet integration
// This shader handles angular velocity damping and rotation integration
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

// Torque accumulation (atomic i32, fixed-point) and angular velocities (group 1)
@group(1) @binding(0)
var<storage, read> torque_accum_x: array<i32>;

@group(1) @binding(1)
var<storage, read> torque_accum_y: array<i32>;

@group(1) @binding(2)
var<storage, read> torque_accum_z: array<i32>;

@group(1) @binding(3)
var<storage, read_write> angular_velocities: array<vec4<f32>>;

@group(1) @binding(4)
var<storage, read_write> rotations: array<vec4<f32>>;

// PBD rotation corrections from adhesion physics (group 1, bindings 5-7)
@group(1) @binding(5)
var<storage, read> pbd_rot_x: array<i32>;

@group(1) @binding(6)
var<storage, read> pbd_rot_y: array<i32>;

@group(1) @binding(7)
var<storage, read> pbd_rot_z: array<i32>;

const PI: f32 = 3.14159265359;
const FIXED_POINT_SCALE: f32 = 1000.0;

// Convert fixed-point i32 back to float
fn fixed_to_float(v: i32) -> f32 {
    return f32(v) / FIXED_POINT_SCALE;
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Quaternion multiplication
fn quat_multiply(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    let mass = positions_out[cell_idx].w;
    let radius = calculate_radius_from_mass(mass);
    
    // Read accumulated torque from adhesion physics (convert from fixed-point)
    let torque = vec3<f32>(
        fixed_to_float(torque_accum_x[cell_idx]),
        fixed_to_float(torque_accum_y[cell_idx]),
        fixed_to_float(torque_accum_z[cell_idx])
    );
    
    // Calculate moment of inertia (solid sphere: I = 2/5 * m * r^2)
    let moment_of_inertia = 0.4 * mass * radius * radius;
    
    // Calculate angular acceleration from torque
    var angular_acceleration = vec3<f32>(0.0);
    if (moment_of_inertia > 0.0001) {
        angular_acceleration = torque / moment_of_inertia;
    }
    
    // Read current angular velocity
    let ang_vel = angular_velocities[cell_idx].xyz;
    
    // Apply angular damping (matching CPU: damping^(dt*100))
    let angular_damping_factor = pow(params.acceleration_damping, params.delta_time * 100.0);
    let new_ang_vel = (ang_vel + angular_acceleration * params.delta_time) * angular_damping_factor;
    
    // Write updated angular velocity
    angular_velocities[cell_idx] = vec4<f32>(new_ang_vel, 0.0);
    
    // Integrate rotation using angular velocity
    var current_rot = rotations[cell_idx];
    let ang_vel_mag = length(new_ang_vel);
    if (ang_vel_mag > 0.0001) {
        let angle = ang_vel_mag * params.delta_time;
        let axis = new_ang_vel / ang_vel_mag;
        
        // Create rotation quaternion from axis-angle
        let half_angle = angle * 0.5;
        let sin_half = sin(half_angle);
        let cos_half = cos(half_angle);
        let delta_rotation = vec4<f32>(axis * sin_half, cos_half);
        
        // Apply rotation: new_rot = delta_rot * current_rot
        current_rot = quat_multiply(delta_rotation, current_rot);
    }
    
    // Apply PBD rotation corrections directly (axis * angle vector)
    let pbd_rot = vec3<f32>(
        fixed_to_float(pbd_rot_x[cell_idx]),
        fixed_to_float(pbd_rot_y[cell_idx]),
        fixed_to_float(pbd_rot_z[cell_idx])
    );
    let pbd_rot_mag = length(pbd_rot);
    if (pbd_rot_mag > 0.0001) {
        let pbd_axis = pbd_rot / pbd_rot_mag;
        let pbd_half = pbd_rot_mag * 0.5;
        let pbd_delta = vec4<f32>(pbd_axis * sin(pbd_half), cos(pbd_half));
        current_rot = quat_multiply(pbd_delta, current_rot);
    }
    
    // Normalize to prevent drift
    rotations[cell_idx] = normalize(current_rot);
}