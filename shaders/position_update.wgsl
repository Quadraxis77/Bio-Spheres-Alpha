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

// Water grid parameters for buoyancy (must match WaterGridParams in Rust)
struct WaterGridParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    buoyancy_multiplier: f32,
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

// Water bitfield for buoyancy (group 2, bindings 4-5) - 32x compressed water detection
// Each u32 contains 32 voxels packed as bits (1 = water, 0 = not water)
@group(2) @binding(4)
var<uniform> water_params: WaterGridParams;

@group(2) @binding(5)
var<storage, read> water_bitfield: array<u32>;

// PBD position corrections from adhesion physics (group 2, bindings 6-8)
@group(2) @binding(6)
var<storage, read> pbd_pos_x: array<i32>;

@group(2) @binding(7)
var<storage, read> pbd_pos_y: array<i32>;

@group(2) @binding(8)
var<storage, read> pbd_pos_z: array<i32>;

// Cell type behavior flags for parameterized shader logic
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    applies_buoyancy: u32,
    _padding: array<u32, 9>,
}

// Spatial grid + cell type data (group 3)
@group(3) @binding(5)
var<storage, read> mode_indices: array<u32>;

@group(3) @binding(6)
var<storage, read> mode_cell_types: array<u32>;

@group(3) @binding(7)
var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// Mode properties (per-mode settings from genome)
// Layout: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, split_ratio, buoyancy_force, padding]
// Total: 12 floats = 48 bytes per mode
struct ModeProperties {
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    membrane_stiffness: f32,
    split_interval: f32,
    split_mass: f32,
    nutrient_priority: f32,
    swim_force: f32,
    prioritize_when_low: f32,
    max_splits: f32,
    split_ratio: f32,
    buoyancy_force: f32,
    _pad0: f32,
}

@group(3) @binding(8)
var<storage, read> mode_properties: array<ModeProperties>;

const FIXED_POINT_SCALE: f32 = 1000.0;
const WATER_GRID_X_GROUPS: u32 = 4u;  // 128 / 32 = 4 u32s per row

// Archimedes buoyancy constants
// F_buoyancy = ρ_fluid × V_cell × g × buoyancy_force (opposing gravity)
// ρ_fluid = density of water in simulation units
// V_cell = (4/3)π r³ where r is derived from mass (assuming unit density cells)
// buoyancy_force = per-mode slider (0.0 to 1.0) controlling buoyancy strength
const WATER_DENSITY: f32 = 1.0;
const PI: f32 = 3.14159265;
const BUOYANCY_BASE_SCALE: f32 = 4.0;  // Base amplification, scaled by per-mode buoyancy_force

// Convert fixed-point i32 back to float
fn fixed_to_float(v: i32) -> f32 {
    return f32(v) / FIXED_POINT_SCALE;
}

// Check if a world position is inside water using the compressed bitfield
// Returns true if the cell is in a water voxel
fn is_in_water(world_pos: vec3<f32>) -> bool {
    let res = water_params.grid_resolution;

    // Convert world position to grid coordinates
    let grid_pos = vec3<f32>(
        (world_pos.x - water_params.grid_origin_x) / water_params.cell_size,
        (world_pos.y - water_params.grid_origin_y) / water_params.cell_size,
        (world_pos.z - water_params.grid_origin_z) / water_params.cell_size
    );

    // Bounds check - outside grid means not in water
    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return false;
    }

    // Convert to integer grid coordinates
    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);

    // Calculate bitfield index
    // Bitfield layout: each u32 contains 32 consecutive voxels along X axis
    // bitfield_idx = (x / 32) + y * 4 + z * 4 * 128
    let x_group = gx / 32u;
    let bit_index = gx % 32u;
    let bitfield_idx = x_group + gy * WATER_GRID_X_GROUPS + gz * WATER_GRID_X_GROUPS * res;

    // Read the u32 containing this voxel and extract the bit
    let bits = water_bitfield[bitfield_idx];
    return (bits & (1u << bit_index)) != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Read from positions_out which has PBD corrections already applied by iterative adhesion loop
    let pos = positions_out[cell_idx].xyz;
    let mass = positions_out[cell_idx].w;
    let vel = velocities_in[cell_idx].xyz;
    
    // Read accumulated forces from collision and adhesion stages (convert from fixed-point)
    var force = vec3<f32>(
        fixed_to_float(force_accum_x[cell_idx]),
        fixed_to_float(force_accum_y[cell_idx]),
        fixed_to_float(force_accum_z[cell_idx])
    );

    // Check if cell is in water - if so, reduce gravity by 95%
    let in_water = is_in_water(pos);
    var gravity_multiplier = 1.0;
    if (in_water) {
        // Reduce gravity to 5% when in water (95% less influenced)
        gravity_multiplier = 0.05;
    }

    // Apply gravity (F = mg) in selected directions, reduced if in water
    let gravity_force = params.gravity * mass * gravity_multiplier;
    force.x += gravity_force * params.gravity_dir_x;
    force.y += gravity_force * params.gravity_dir_y;
    force.z += gravity_force * params.gravity_dir_z;

    // Archimedes buoyancy for Buoyocyte cells submerged in water
    // F_buoyancy = ρ_fluid × V_cell × g (opposing gravity direction)
    if (in_water) {
        let mode_idx = mode_indices[cell_idx];
        let cell_type = mode_cell_types[mode_idx];
        let behavior = type_behaviors[cell_type];
        if (behavior.applies_buoyancy == 1u) {
            let mode = mode_properties[mode_idx];
            if (mode.buoyancy_force > 0.0) {
                // Cell radius from mass assuming unit-density sphere: m = (4/3)πr³ → r = (3m/4π)^(1/3)
                let radius = pow(3.0 * mass / (4.0 * PI), 1.0 / 3.0);
                let volume = (4.0 / 3.0) * PI * radius * radius * radius;
                // Buoyant force magnitude = ρ_fluid × V × |g| × base_scale × buoyancy_force
                let buoyancy_magnitude = WATER_DENSITY * volume * abs(params.gravity) * BUOYANCY_BASE_SCALE * mode.buoyancy_force;
                // Apply buoyancy opposing gravity direction
                // gravity is negative (e.g. -9.8), gravity_dir is positive mask (0 or 1)
                // gravity pushes down: force += gravity * mass * dir (negative)
                // buoyancy must push up: force -= gravity_sign * magnitude * dir
                let gravity_sign = sign(params.gravity);
                force.x -= gravity_sign * buoyancy_magnitude * params.gravity_dir_x;
                force.y -= gravity_sign * buoyancy_magnitude * params.gravity_dir_y;
                force.z -= gravity_sign * buoyancy_magnitude * params.gravity_dir_z;
            }
        }
    }

    // Read previous acceleration for Verlet integration
    let old_acceleration = prev_accelerations[cell_idx].xyz;

    // Calculate new acceleration from accumulated forces
    let new_acceleration = force / mass;
    
    // Verlet integration (matching CPU exactly):
    // velocity_change = 0.5 * (old_acceleration + new_acceleration) * dt
    let velocity_change = 0.5 * (old_acceleration + new_acceleration) * params.delta_time;
    
    // Apply velocity damping (matching CPU: damping^(dt*100))
    let damping_factor = pow(params.acceleration_damping, params.delta_time * 100.0);
    var new_vel = (vel + velocity_change) * damping_factor;
    
    // Clamp velocity to max 10 units/second
    let speed = length(new_vel);
    if (speed > 10.0) {
        new_vel = (new_vel / speed) * 10.0;
    }
    
    // Simple position update
    let new_pos = pos + new_vel * params.delta_time;
    
    // Smooth boundary collision with lerping to prevent teleporting
    let boundary_radius = params.world_size * 0.5;
    var final_pos = new_pos;
    var final_vel = new_vel;
    
    // Check if new position would violate boundary
    let dist_from_center = length(new_pos);
    if (dist_from_center > boundary_radius) {
        // Calculate penetration depth
        let penetration = dist_from_center - boundary_radius;
        
        // Smooth lerp factor based on penetration (0.0 = no correction, 1.0 = full correction)
        // Use a soft lerp that increases gradually with penetration depth
        let max_penetration = 5.0; // Maximum penetration for full correction
        let lerp_factor = clamp(penetration / max_penetration, 0.0, 1.0);
        let smooth_lerp = lerp_factor * lerp_factor; // Quadratic for smoother transition
        
        // Calculate target position (just inside boundary)
        let target_pos = normalize(new_pos) * boundary_radius * 0.99;
        
        // Smoothly lerp current position toward target position
        final_pos = mix(new_pos, target_pos, smooth_lerp);
        
        // Apply gentle velocity damping instead of hard reflection
        let inward_dir = -normalize(new_pos);
        let vel_normal = dot(new_vel, inward_dir);
        
        if (vel_normal < 0.0) {
            // Reduce outward velocity smoothly based on penetration
            let damping_factor = 1.0 - (smooth_lerp * 0.8); // Max 80% reduction
            let vel_tangent = new_vel - vel_normal * inward_dir;
            let vel_normal_damped = vel_normal * damping_factor;
            final_vel = vel_tangent + vel_normal_damped * inward_dir;
        }
    }
    
    // PBD position corrections are already applied to positions_out by the iterative
    // adhesion loop (apply_pbd shader). No need to read PBD accumulators here.
    
    // Write updated position and velocity
    positions_out[cell_idx] = vec4<f32>(final_pos, mass);
    velocities_out[cell_idx] = vec4<f32>(final_vel, 0.0);
    
    // Store current acceleration for next frame's Verlet integration
    prev_accelerations[cell_idx] = vec4<f32>(new_acceleration, 0.0);
    
    // Propagate rotation from input to output (no rotation physics yet)
    rotations_out[cell_idx] = rotations_in[cell_idx];
}
