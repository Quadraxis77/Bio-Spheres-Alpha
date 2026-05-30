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
    _pad0: f32,           // gravity_mode (unused here)
    angular_damping: f32, // fraction of angular velocity retained per second (independent of linear)
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

// Genome orientations — synced from rotations each frame so the geometric spring
// target tracks the creature's actual orientation rather than its birth orientation.
@group(1) @binding(5)
var<storage, read_write> genome_orientations: array<vec4<f32>>;

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

    // Skip dead cells - clear angular velocity so recycled slots start clean
    if (mass < 0.5) {
        angular_velocities[cell_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

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
    
    // Apply angular damping: separate coefficient from linear damping so structures
    // can spin freely while still gradually losing rotational energy to drag.
    // Multiply exponent by 100.0 to match the CPU path (integrate_angular_velocities
    // uses powf(dt * 100.0)). Without this, GPU damping is ~100x weaker than CPU
    // because dt is ~0.016, making pow(0.95, 0.016) ≈ 0.9992 instead of the
    // intended pow(0.95, 1.6) ≈ 0.92.
    let ang_damp = select(params.angular_damping, 0.94, params.angular_damping < 0.001);
    let angular_damping_factor = pow(ang_damp, params.delta_time * 100.0);
    let new_ang_vel = (ang_vel + angular_acceleration * params.delta_time) * angular_damping_factor;
    
    // Write updated angular velocity
    angular_velocities[cell_idx] = vec4<f32>(new_ang_vel, 0.0);
    
    // Integrate rotation using angular velocity
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
        let current_rot = rotations[cell_idx];
        let new_rot = quat_multiply(delta_rotation, current_rot);
        
        // Normalize to prevent drift
        rotations[cell_idx] = normalize(new_rot);
    }

    // Sync genome_orientations from rotations so the adhesion geometric spring
    // target tracks the creature's actual orientation. Without this, genome_orientations
    // stays frozen at birth and the spring locks the creature to its spawn orientation.
    genome_orientations[cell_idx] = rotations[cell_idx];
}