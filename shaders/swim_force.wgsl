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

// Water grid parameters for checking whether flagella have a medium to push against.
// Must match WaterGridParams in Rust and position_update.wgsl.
struct WaterGridParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    buoyancy_multiplier: f32,
    water_viscosity: f32,
    _pad1: f32,
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

@group(1) @binding(4)
var<storage, read_write> torque_accum_x: array<atomic<i32>>;

@group(1) @binding(5)
var<storage, read_write> torque_accum_y: array<atomic<i32>>;

@group(1) @binding(6)
var<storage, read_write> torque_accum_z: array<atomic<i32>>;

@group(1) @binding(7)
var<storage, read_write> angular_velocities: array<vec4<f32>>;

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

@group(2) @binding(10)
var<uniform> water_params: WaterGridParams;

@group(2) @binding(11)
var<storage, read> water_bitfield: array<u32>;

@group(2) @binding(12)
var<storage, read> mode_properties_v14: array<vec4<f32>>;

@group(2) @binding(13)
var<storage, read> mode_properties_v15: array<vec4<f32>>;

@group(2) @binding(14)
var<storage, read_write> cell_water: array<f32>;

@group(2) @binding(15)
var<storage, read_write> cell_heat_energy: array<f32>;

@group(2) @binding(16)
var<storage, read> cell_thermal_state: array<u32>;

const FIXED_POINT_SCALE: f32 = 1000.0;
const SWIM_FORCE_MULTIPLIER: f32 = 50.0; // Scale swim_force (0-1) to actual force magnitude
const WATER_GRID_X_GROUPS: u32 = 4u;  // 128 / 32 = 4 u32s per row
const CELL_TYPE_SIPHONOCYTE: u32 = 17u;
const CELL_TYPE_PLUMOCYTE: u32 = 18u;
const SIPHON_WATER_CAPACITY: f32 = 1.5;
const SIPHON_MIN_IMPULSE_WATER: f32 = 0.01;
const SIPHON_DRY_EFFICIENCY: f32 = 0.50;
const STEAM_SAFE_FLOOR_TEMP: f32 = 115.0;
const THERMAL_STATE_FROZEN: u32 = 1u;

// Convert float to fixed-point i32 for atomic accumulation
fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

fn water_voxel_at(world_pos: vec3<f32>) -> bool {
    let res = water_params.grid_resolution;
    if (res == 0u) {
        return false;
    }

    let grid_pos = vec3<f32>(
        (world_pos.x - water_params.grid_origin_x) / water_params.cell_size,
        (world_pos.y - water_params.grid_origin_y) / water_params.cell_size,
        (world_pos.z - water_params.grid_origin_z) / water_params.cell_size
    );

    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return false;
    }

    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);

    let x_group = gx / 32u;
    let bit_index = gx % 32u;
    let bitfield_idx = x_group + gy * WATER_GRID_X_GROUPS + gz * WATER_GRID_X_GROUPS * res;

    let bits = water_bitfield[bitfield_idx];
    return (bits & (1u << bit_index)) != 0u;
}

fn is_in_water(world_pos: vec3<f32>) -> bool {
    if (water_voxel_at(world_pos)) {
        return true;
    }

    let boundary_radius = params.world_size * 0.5;
    let dist_from_center = length(world_pos);
    if (dist_from_center > 0.0001 &&
        dist_from_center >= boundary_radius - water_params.cell_size * 2.0 &&
        dist_from_center <= boundary_radius + water_params.cell_size) {
        let inward_pos = world_pos - (world_pos / dist_from_center) * water_params.cell_size;
        return water_voxel_at(inward_pos);
    }

    return false;
}

// Rotate a vector by a quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn temperature_for(heat_energy: f32, water: f32) -> f32 {
    return heat_energy / max(1.0 + water, 0.001);
}

fn heat_for_temperature(temp: f32, water: f32) -> f32 {
    return temp * max(1.0 + water, 0.001);
}

fn unpack_signal_channel(packed: f32) -> u32 {
    return u32(clamp(packed, 0.0, 262143.0)) & 15u;
}

fn unpack_siphon_invert(packed: f32) -> bool {
    return (u32(clamp(packed, 0.0, 262143.0)) & 16u) != 0u;
}

fn unpack_siphon_mode(packed: f32) -> u32 {
    return min((u32(clamp(packed, 0.0, 262143.0)) / 32u) & 3u, 3u);
}

fn unpack_siphon_threshold(packed: f32) -> f32 {
    return f32(u32(clamp(packed, 0.0, 262143.0)) / 128u);
}

fn siphon_signal_active(cell_idx: u32, props: vec4<f32>) -> bool {
    let channel = unpack_signal_channel(props.w);
    let raw_signal = atomicLoad(&signal_flags[cell_idx * 16u + channel]);
    let signal_value = f32(raw_signal & 2047u);
    let above = signal_value >= unpack_siphon_threshold(props.w);
    return select(above, !above, unpack_siphon_invert(props.w));
}

fn siphon_should_expel(cell_idx: u32, mode_idx: u32, props: vec4<f32>) -> bool {
    let mode = unpack_siphon_mode(props.w);
    if (mode == 2u) {
        return false;
    }
    let impulse = max(props.z, 0.0);
    let stroke_phase = fract(params.current_time * mix(0.85, 2.4, clamp(impulse / 3.0, 0.0, 1.0)) + f32(cell_idx & 1023u) * 0.013);
    let expel_stroke = smoothstep(0.54, 0.64, stroke_phase) * (1.0 - smoothstep(0.82, 0.96, stroke_phase));
    if (mode == 0u || mode == 1u) {
        if (expel_stroke <= 0.05) {
            return false;
        }
    }
    if (mode == 0u) {
        return true;
    }
    if (mode == 1u) {
        return siphon_signal_active(cell_idx, props);
    }
    if (mode == 3u) {
        return siphon_signal_active(cell_idx, props);
    }
    return false;
}

fn apply_siphon_force(cell_idx: u32, mode_idx: u32) {
    if (mode_idx >= arrayLength(&mode_properties_v14)) {
        return;
    }

    let props = mode_properties_v14[mode_idx];
    if (!siphon_should_expel(cell_idx, mode_idx, props)) {
        return;
    }

    let water = clamp(cell_water[cell_idx], 0.0, SIPHON_WATER_CAPACITY);
    let submerged = is_in_water(positions_in[cell_idx].xyz);
    let can_spend_water = submerged && water >= SIPHON_MIN_IMPULSE_WATER;

    let rotation = rotations[cell_idx];
    let forward = quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));
    let expel_rate = max(props.y, 0.0);
    let impulse = max(props.z, 0.0);
    let dt = max(params.delta_time, 0.0);

    var water_cost = expel_rate * dt;
    var impulse_mult = 1.0;

    var reserve_efficiency = SIPHON_DRY_EFFICIENCY;
    if (can_spend_water) {
        reserve_efficiency = 1.0;

        // Steam assist spends only excess heat above the safe floor and never cools
        // below that floor. It changes internal reserves only; no steam voxels exist.
        let heat = max(cell_heat_energy[cell_idx], 0.0);
        let temp = temperature_for(heat, water);
        var remaining_heat = heat;
        if (temp > 130.0) {
            let safe_heat = heat_for_temperature(STEAM_SAFE_FLOOR_TEMP, water);
            let available_heat = max(heat - safe_heat, 0.0);
            let heat_spend = min(available_heat, impulse * 6.0 * dt);
            if (heat_spend > 0.0) {
                remaining_heat = heat - heat_spend;
                water_cost *= 0.55;
                impulse_mult = 1.35;
            }
        }

        let spent = min(water, water_cost);
        reserve_efficiency = clamp(spent / max(water_cost, 0.001), SIPHON_DRY_EFFICIENCY, 1.0);
        let remaining_water = water - spent;

        // Expelled water carries its share of the cell's heat. Previously water
        // mass was removed while nearly all heat stayed behind, so repeated siphon
        // strokes concentrated temperature until the cell entered heat shock.
        // Preserve the post-steam-assist temperature across the mass change.
        let post_assist_temp = temperature_for(remaining_heat, water);
        cell_water[cell_idx] = remaining_water;
        cell_heat_energy[cell_idx] =
            heat_for_temperature(post_assist_temp, remaining_water);
    }

    let force_magnitude = impulse * 80.0 * impulse_mult * reserve_efficiency;
    let force = forward * force_magnitude;
    atomicAdd(&force_accum_x[cell_idx], float_to_fixed(force.x));
    atomicAdd(&force_accum_y[cell_idx], float_to_fixed(force.y));
    atomicAdd(&force_accum_z[cell_idx], float_to_fixed(force.z));
}

fn apply_plumocyte_force(cell_idx: u32, mode_idx: u32) {
    if (mode_idx >= arrayLength(&mode_properties_v15)) {
        return;
    }
    if (cell_idx < arrayLength(&cell_thermal_state) && cell_thermal_state[cell_idx] <= THERMAL_STATE_FROZEN) {
        return;
    }
    let props = mode_properties_v15[mode_idx];
    let drag_mult = max(props.x, 0.0);
    let rotation_resistance = max(props.y, 0.0);
    if (drag_mult <= 0.001 && rotation_resistance <= 0.001) {
        return;
    }

    if (drag_mult > 0.001) {
        let velocity = velocities_in[cell_idx].xyz;
        let anti_gravity = anti_gravity_direction(positions_in[cell_idx].xyz);
        let fall_speed = dot(velocity, -anti_gravity);
        if (fall_speed > 0.001) {
            let drag = anti_gravity * fall_speed * drag_mult * 18.0;
            atomicAdd(&force_accum_x[cell_idx], float_to_fixed(drag.x));
            atomicAdd(&force_accum_y[cell_idx], float_to_fixed(drag.y));
            atomicAdd(&force_accum_z[cell_idx], float_to_fixed(drag.z));
        }
    }

    if (rotation_resistance > 0.001 && cell_idx < arrayLength(&angular_velocities)) {
        let omega = angular_velocities[cell_idx].xyz;
        let spin_speed = length(omega);
        if (spin_speed > 0.001) {
            let torque = -omega * rotation_resistance * 12.0;
            atomicAdd(&torque_accum_x[cell_idx], float_to_fixed(torque.x));
            atomicAdd(&torque_accum_y[cell_idx], float_to_fixed(torque.y));
            atomicAdd(&torque_accum_z[cell_idx], float_to_fixed(torque.z));
        }
    }
}

fn plumocyte_is_frozen(cell_idx: u32) -> bool {
    return cell_idx < arrayLength(&cell_thermal_state) && cell_thermal_state[cell_idx] <= THERMAL_STATE_FROZEN;
}

fn plumocyte_rotation_retain(rotation_resistance: f32) -> f32 {
    return exp(-rotation_resistance * 6.0 * max(params.delta_time, 0.0));
}

fn anti_gravity_direction(pos: vec3<f32>) -> vec3<f32> {
    if (params.gravity_mode == 3u) {
        let r = length(pos);
        if (r > 0.001) {
            return (pos / r) * select(1.0, -1.0, params.gravity < 0.0);
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
    if (mode_idx >= arrayLength(&mode_cell_types) || mode_idx >= arrayLength(&mode_properties_v1)) {
        return;
    }
    
    // Derive cell type from mode (always up-to-date with genome settings)
    let cell_type = mode_cell_types[mode_idx];

    if (cell_type == CELL_TYPE_SIPHONOCYTE) {
        apply_siphon_force(cell_idx, mode_idx);
        return;
    }

    if (cell_type == CELL_TYPE_PLUMOCYTE) {
        apply_plumocyte_force(cell_idx, mode_idx);
        return;
    }

    // Check if this cell type applies swim force using behavior flags
    let behavior = type_behaviors[cell_type];
    if (behavior.applies_swim_force == 0u) {
        return;
    }

    if (!is_in_water(positions_in[cell_idx].xyz)) {
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

// Final angular damping pass for Plumocytes. The main swim force pass contributes
// damping torque before velocity_update, but adhesion/cave substeps can directly
// rewrite angular velocity later in the frame. This pass runs after those
// corrections so the user-facing setting remains effective in GPU scenes.
@compute @workgroup_size(256)
fn plumocyte_rotation_damping_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }

    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types) || mode_idx >= arrayLength(&mode_properties_v15)) {
        return;
    }
    if (mode_cell_types[mode_idx] != CELL_TYPE_PLUMOCYTE || plumocyte_is_frozen(cell_idx)) {
        return;
    }

    let rotation_resistance = max(mode_properties_v15[mode_idx].y, 0.0);
    if (rotation_resistance <= 0.001 || cell_idx >= arrayLength(&angular_velocities)) {
        return;
    }

    let retain = plumocyte_rotation_retain(rotation_resistance);
    let omega = angular_velocities[cell_idx].xyz * retain;
    angular_velocities[cell_idx] = vec4<f32>(omega, 0.0);
}

// Buoyancy pass - separate entry point so it runs for all cells regardless of swim force
@compute @workgroup_size(256)
fn buoyancy_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) { return; }

    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types) || mode_idx >= arrayLength(&mode_properties_v4)) { return; }

    let cell_type = mode_cell_types[mode_idx];
    let behavior = type_behaviors[cell_type];
    if (behavior.applies_buoyancy == 0u) { return; }

    // Buoyocytes generate lift only while submerged. With no fluid grid, or
    // while the cell is in air, they behave like ordinary cells under gravity.
    let pos = positions_in[cell_idx].xyz;
    if (!is_in_water(pos)) { return; }

    // v4.w = buoyancy_force
    let buoyancy_force = mode_properties_v4[mode_idx].w;
    if (buoyancy_force <= 0.0) { return; }

    // Apply force opposite the configured gravity axis.
    // Traction falloff: buoyancy force tapers as the cell approaches its terminal
    // rise speed, preventing large buoyocyte blobs from accumulating unlimited
    // upward force that would fling them through the boundary.
    const BUOYANCY_MULTIPLIER: f32 = 120.0;
    const BUOYANCY_TERMINAL_SPEED: f32 = 12.0;
    let up = anti_gravity_direction(pos);
    let velocity_along_up = dot(velocities_in[cell_idx].xyz, up);
    let traction = clamp(1.0 - velocity_along_up / max(buoyancy_force * BUOYANCY_TERMINAL_SPEED, 0.001), 0.0, 1.0);
    let force = up * (buoyancy_force * BUOYANCY_MULTIPLIER * traction);
    atomicAdd(&force_accum_x[cell_idx], float_to_fixed(force.x));
    atomicAdd(&force_accum_y[cell_idx], float_to_fixed(force.y));
    atomicAdd(&force_accum_z[cell_idx], float_to_fixed(force.z));
}
