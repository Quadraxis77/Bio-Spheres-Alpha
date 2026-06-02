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
    gravity_mode: u32,
    _pad1: f32,
    _pad2: f32,
}

// mode_properties layout across 5 sub-buffers (v0-v4), 1 vec4 per mode each:
// v0: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
// v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
// v2: [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
// v3: [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
// v4: [max_adhesions, mode_a_after_splits, mode_b_after_splits, buoyancy_force]

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
var<storage, read> cell_types: array<u32>;  // DEPRECATED - use mode_cell_types instead

// mode_properties split into 5 vec4 sub-buffers (bindings 2-6)
@group(2) @binding(2)
var<storage, read> mode_properties_v0: array<vec4<f32>>;
@group(2) @binding(3)
var<storage, read> mode_properties_v1: array<vec4<f32>>;
@group(2) @binding(4)
var<storage, read> mode_properties_v2: array<vec4<f32>>;
@group(2) @binding(5)
var<storage, read> mode_properties_v3: array<vec4<f32>>;
@group(2) @binding(6)
var<storage, read> mode_properties_v4: array<vec4<f32>>;

// Mode cell types lookup table
@group(2) @binding(7)
var<storage, read> mode_cell_types: array<u32>;

// Cell type behavior flags
@group(2) @binding(8)
var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// Signal flags buffer (read-only)
@group(2) @binding(9)
var<storage, read> signal_flags: array<atomic<u32>>;

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

fn anti_gravity_direction(pos: vec3<f32>) -> vec3<f32> {
    if (params.gravity_mode == 3u) {
        let r = length(pos);
        if (r > 0.001) {
            return pos / r;
        }
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    let sign = select(1.0, -1.0, params.gravity < 0.0);
    if (params.gravity_mode == 0u) {
        return vec3<f32>(sign, 0.0, 0.0);
    }
    if (params.gravity_mode == 2u) {
        return vec3<f32>(0.0, 0.0, sign);
    }
    return vec3<f32>(0.0, sign, 0.0);
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
    
    // Get mode index first, then derive cell type from mode
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_properties_v1)) {
        return;
    }
    
    // Derive cell type from mode (always up-to-date with genome settings)
    let cell_type = mode_cell_types[mode_idx];

    // Check if this cell type applies swim force using behavior flags
    let behavior = type_behaviors[cell_type];
    if (behavior.applies_swim_force == 0u) {
        return;
    }
    
    // Get mode properties from split sub-buffers
    // v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
    // v2: [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
    // v3: [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
    let mode_v1 = mode_properties_v1[mode_idx];
    let mode_v2 = mode_properties_v2[mode_idx];
    let mode_v3 = mode_properties_v3[mode_idx];

    // Determine effective swim speed
    var effective_speed: f32;
    if (mode_v3.z >= 0.5) { // flagellocyte_use_signal
        // Signal-based mode: read from the specific channel
        // mode_v2.z = flagellocyte_signal_channel
        let sig_channel = clamp(u32(mode_v2.z), 0u, 7u);
        let raw_signal = atomicLoad(&signal_flags[cell_idx * 16u + sig_channel]);
        let signal_value = f32(raw_signal & 2047u); // Extract value component
        if (signal_value >= mode_v3.y) { // flagellocyte_threshold_c
            effective_speed = mode_v3.x; // flagellocyte_speed_b
        } else {
            effective_speed = mode_v2.w; // flagellocyte_speed_a
        }
    } else {
        // Fixed speed mode
        effective_speed = mode_v1.z; // swim_force
    }
    
    // Skip if no effective swim speed
    if (effective_speed <= 0.0) {
        return;
    }
    
    // Get cell rotation and calculate forward direction
    // Forward is +Z in local space (matching preview physics)
    let rotation = rotations[cell_idx];
    let forward = quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));
    
    // Calculate swim force vector.
    // Traction falloff: force tapers as the cell approaches its target speed.
    // This prevents many aligned flagellocytes from summing to unbounded aggregate
    // force - each cell contributes less the faster the body is already moving.
    let vel = velocities_in[cell_idx].xyz;
    let speed_along_forward = dot(vel, forward);
    let target_speed = effective_speed * 8.0;
    let traction = clamp(1.0 - speed_along_forward / max(target_speed, 0.001), 0.0, 1.0);
    let force_magnitude = effective_speed * SWIM_FORCE_MULTIPLIER * traction;
    let swim_force = forward * force_magnitude;
    
    // Add to force accumulation buffers (atomic for thread safety)
    atomicAdd(&force_accum_x[cell_idx], float_to_fixed(swim_force.x));
    atomicAdd(&force_accum_y[cell_idx], float_to_fixed(swim_force.y));
    atomicAdd(&force_accum_z[cell_idx], float_to_fixed(swim_force.z));
}

// Buoyancy pass - separate entry point so it runs for all cells regardless of swim force
@compute @workgroup_size(256)
fn buoyancy_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) { return; }

    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_properties_v4)) { return; }

    let cell_type = mode_cell_types[mode_idx];
    let behavior = type_behaviors[cell_type];
    if (behavior.applies_buoyancy == 0u) { return; }

    // v4.w = buoyancy_force
    let buoyancy_force = mode_properties_v4[mode_idx].w;
    if (buoyancy_force <= 0.0) { return; }

    // Apply force opposite the configured gravity axis.
    // Traction falloff: buoyancy force tapers as the cell approaches its terminal
    // rise speed, preventing large buoyocyte blobs from accumulating unlimited
    // upward force that would fling them through the boundary.
    const BUOYANCY_MULTIPLIER: f32 = 120.0;
    const BUOYANCY_TERMINAL_SPEED: f32 = 12.0;
    let pos = positions_in[cell_idx].xyz;
    let up = anti_gravity_direction(pos);
    let velocity_along_up = dot(velocities_in[cell_idx].xyz, up);
    let traction = clamp(1.0 - velocity_along_up / max(buoyancy_force * BUOYANCY_TERMINAL_SPEED, 0.001), 0.0, 1.0);
    let force = up * (buoyancy_force * BUOYANCY_MULTIPLIER * traction);
    atomicAdd(&force_accum_x[cell_idx], float_to_fixed(force.x));
    atomicAdd(&force_accum_y[cell_idx], float_to_fixed(force.y));
    atomicAdd(&force_accum_z[cell_idx], float_to_fixed(force.z));
}
