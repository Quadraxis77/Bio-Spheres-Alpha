// Hidden per-cell water/heat physiology.
// Runs at a low cadence. Each invocation gathers from adhesion-connected neighbors
// and writes only its own "next" physiology slot.

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
    angular_damping: f32,
    solo_metabolism_multiplier: f32,
};

struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    bond_flags: u32,
    _align_pad1: u32,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
};

struct WaterGridParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    buoyancy_multiplier: f32,
    water_viscosity: f32,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read_write> cell_water: array<f32>;
@group(1) @binding(1)
var<storage, read_write> cell_heat_energy: array<f32>;
@group(1) @binding(2)
var<storage, read_write> cell_cached_temperature: array<f32>;
@group(1) @binding(3)
var<storage, read_write> cell_thermal_state: array<u32>;
@group(1) @binding(4)
var<storage, read_write> cell_water_next: array<f32>;
@group(1) @binding(5)
var<storage, read_write> cell_heat_energy_next: array<f32>;
@group(1) @binding(6)
var<storage, read_write> cell_cached_temperature_next: array<f32>;
@group(1) @binding(7)
var<storage, read_write> cell_thermal_state_next: array<u32>;
@group(1) @binding(8)
var<storage, read_write> prev_muscle_contraction: array<f32>;
@group(1) @binding(9)
var<storage, read_write> muscle_contraction: array<f32>;

@group(2) @binding(0)
var<storage, read> death_flags: array<u32>;
@group(2) @binding(1)
var<storage, read> mode_indices: array<u32>;
@group(2) @binding(2)
var<storage, read> mode_cell_types: array<u32>;

@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;
@group(3) @binding(4)
var<storage, read> adhesion_counts: array<u32>;
@group(3) @binding(5)
var<storage, read> cell_adhesion_indices: array<i32>;

@group(2) @binding(3)
var<uniform> water_params: WaterGridParams;
@group(2) @binding(4)
var<storage, read> fluid_state: array<atomic<u32>>;
@group(2) @binding(5)
var<storage, read> voxel_temperature: array<atomic<u32>>;
@group(2) @binding(6)
var<storage, read> geothermal_heat: array<f32>;

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const BOND_FLAG_BARRIER_BALL: u32 = 2u;

const CELL_TYPE_MYOCYTE: u32 = 9u;
const CELL_TYPE_VASCULOCYTE: u32 = 12u;

const STATE_DEEP_FROZEN: u32 = 0u;
const STATE_FROZEN: u32 = 1u;
const STATE_CHILLED: u32 = 2u;
const STATE_STABLE_COOL: u32 = 3u;
const STATE_IDEAL: u32 = 4u;
const STATE_WARM: u32 = 5u;
const STATE_HOT_SAFE: u32 = 6u;
const STATE_OVERHEATED: u32 = 7u;
const STATE_HEAT_SHOCK: u32 = 8u;
const STATE_CRITICAL: u32 = 9u;

const PHYSIOLOGY_TICK_SCALE: f32 = 4.0;
const DRY_THERMAL_MASS: f32 = 1.0;
const WATER_THERMAL_MASS_FACTOR: f32 = 1.0;
const BASE_ENV_TEMP: f32 = 105.0;
const TEMP_DEADBAND: f32 = 0.25;
const WATER_DEADBAND: f32 = 0.0025;
const VOXEL_TEMP_MIN_C: f32 = -50.0;
const VOXEL_TEMP_MAX_C: f32 = 150.0;
const VOXEL_TEMP_FP: f32 = 256.0;
const FLUID_TYPE_MASK: u32 = 0x7u;

struct VoxelEnvironment {
    valid: bool,
    fluid_type: u32,
    fill_fraction: f32,
    temp_internal: f32,
};

fn cell_type_for(cell_idx: u32) -> u32 {
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types)) {
        return 0xFFFFFFFFu;
    }
    return mode_cell_types[mode_idx];
}

fn water_capacity(cell_type: u32) -> f32 {
    if (cell_type == CELL_TYPE_VASCULOCYTE) {
        return 10.0;
    }
    return 1.0;
}

fn water_transfer_mult(cell_type: u32) -> f32 {
    if (cell_type == CELL_TYPE_VASCULOCYTE) {
        return 4.0;
    }
    return 1.0;
}

fn heat_transfer_mult(cell_type: u32) -> f32 {
    if (cell_type == CELL_TYPE_VASCULOCYTE) {
        return 4.0;
    }
    return 1.0;
}

fn thermal_mass_for(water: f32) -> f32 {
    return max(DRY_THERMAL_MASS + water * WATER_THERMAL_MASS_FACTOR, 0.001);
}

fn temperature_for(heat_energy: f32, water: f32) -> f32 {
    return heat_energy / thermal_mass_for(water);
}

fn heat_for_temperature(temp: f32, water: f32) -> f32 {
    return temp * thermal_mass_for(water);
}

fn celsius_to_internal_temp(temp_c: f32) -> f32 {
    return clamp((temp_c - VOXEL_TEMP_MIN_C) / (VOXEL_TEMP_MAX_C - VOXEL_TEMP_MIN_C) * 255.0, 0.0, 255.0);
}

fn read_voxel_environment(world_pos: vec3<f32>) -> VoxelEnvironment {
    let res = water_params.grid_resolution;
    if (res == 0u) {
        return VoxelEnvironment(false, 0u, 0.0, BASE_ENV_TEMP);
    }

    let grid_pos = vec3<f32>(
        (world_pos.x - water_params.grid_origin_x) / water_params.cell_size,
        (world_pos.y - water_params.grid_origin_y) / water_params.cell_size,
        (world_pos.z - water_params.grid_origin_z) / water_params.cell_size
    );

    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return VoxelEnvironment(false, 0u, 0.0, BASE_ENV_TEMP);
    }

    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);
    let voxel_idx = gx + gy * res + gz * res * res;

    let state = atomicLoad(&fluid_state[voxel_idx]);
    let fluid_type = state & FLUID_TYPE_MASK;
    let fill_fraction = f32((state >> 16u) & 0xFFFFu) / 65535.0;

    var temp_internal = BASE_ENV_TEMP;
    let raw_temp = atomicLoad(&voxel_temperature[voxel_idx]);
    if (raw_temp != 0u) {
        let temp_c = VOXEL_TEMP_MIN_C + f32(raw_temp) / VOXEL_TEMP_FP + geothermal_heat[voxel_idx] * 0.15;
        temp_internal = celsius_to_internal_temp(temp_c);
    } else if (geothermal_heat[voxel_idx] > 0.0) {
        temp_internal = celsius_to_internal_temp(20.0 + geothermal_heat[voxel_idx] * 0.15);
    }

    return VoxelEnvironment(true, fluid_type, fill_fraction, temp_internal);
}

fn is_frozen_state(state: u32) -> bool {
    return state <= STATE_FROZEN;
}

fn classify_state(temp: f32, old_state: u32) -> u32 {
    if (old_state == STATE_FROZEN && temp < 65.0) {
        return STATE_FROZEN;
    }
    if (old_state == STATE_HEAT_SHOCK && temp > 145.0) {
        return STATE_HEAT_SHOCK;
    }
    if (temp <= 45.0) { return STATE_DEEP_FROZEN; }
    if (temp < 60.0) { return STATE_FROZEN; }
    if (temp < 85.0) { return STATE_CHILLED; }
    if (temp < 100.0) { return STATE_STABLE_COOL; }
    if (temp < 115.0) { return STATE_IDEAL; }
    if (temp < 130.0) { return STATE_WARM; }
    if (temp < 140.0) { return STATE_HOT_SAFE; }
    if (temp < 150.0) { return STATE_OVERHEATED; }
    if (temp < 160.0) { return STATE_HEAT_SHOCK; }
    return STATE_CRITICAL;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }

    if (death_flags[cell_idx] == 1u) {
        cell_water_next[cell_idx] = 0.0;
        cell_heat_energy_next[cell_idx] = 0.0;
        cell_cached_temperature_next[cell_idx] = BASE_ENV_TEMP;
        cell_thermal_state_next[cell_idx] = STATE_IDEAL;
        prev_muscle_contraction[cell_idx] = 0.0;
        return;
    }

    let dt = max(params.delta_time * PHYSIOLOGY_TICK_SCALE, 0.0);
    let cell_type = cell_type_for(cell_idx);
    let capacity = water_capacity(cell_type);
    let old_water = clamp(cell_water[cell_idx], 0.0, capacity);
    let old_heat = max(cell_heat_energy[cell_idx], 0.0);
    let old_state = cell_thermal_state[cell_idx];
    let old_temp = temperature_for(old_heat, old_water);

    var water_delta = 0.0;
    var heat_delta = 0.0;

    let env = read_voxel_environment(positions[cell_idx].xyz);

    // Passive environment coupling. The cell samples only the voxel it occupies
    // and never writes moisture or heat back into the fluid grid.
    if (env.valid) {
        var water_target_fraction = 0.45;
        var water_exchange = 0.006;
        if (env.fluid_type == 1u) {
            water_target_fraction = env.fill_fraction;
            water_exchange = 0.08;
        } else if (env.fluid_type == 3u) {
            water_target_fraction = env.fill_fraction * 0.2;
            water_exchange = 0.03;
        } else if (env.fluid_type == 2u || env.fluid_type == 4u) {
            water_target_fraction = 0.0;
            water_exchange = 0.01;
        }
        water_delta += water_exchange * (water_target_fraction * capacity - old_water) * dt;
    } else {
        water_delta += 0.015 * (capacity - old_water) * dt;
    }
    let env_exchange = select(0.04, 0.02, cell_type == CELL_TYPE_VASCULOCYTE);
    let env_temp = select(BASE_ENV_TEMP, env.temp_internal, env.valid);
    heat_delta += (env_temp - old_temp) * env_exchange * dt;

    // Heat stress slowly dehydrates cells; Plumocyte/Siphonocyte specialization is later.
    if (old_temp > 140.0) {
        water_delta -= (old_temp - 140.0) * 0.0015 * dt;
    }

    // Myocyte heat is generated by contraction change, not by static posture.
    if (cell_type == CELL_TYPE_MYOCYTE) {
        let current_contraction = clamp(muscle_contraction[cell_idx], 0.0, 1.0);
        let previous_contraction = clamp(prev_muscle_contraction[cell_idx], 0.0, 1.0);
        let contraction_delta = abs(current_contraction - previous_contraction);
        heat_delta += contraction_delta * 18.0;
        prev_muscle_contraction[cell_idx] = current_contraction;
    } else {
        prev_muscle_contraction[cell_idx] = 0.0;
    }

    let adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let total_adhesions = adhesion_counts[0];
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let adhesion_idx_signed = cell_adhesion_indices[adhesion_base + i];
        if (adhesion_idx_signed < 0) {
            continue;
        }
        let adhesion_idx = u32(adhesion_idx_signed);
        if (adhesion_idx >= total_adhesions) {
            continue;
        }

        let connection = adhesion_connections[adhesion_idx];
        if (connection.is_active == 0u || (connection.bond_flags & BOND_FLAG_BARRIER_BALL) != 0u) {
            continue;
        }

        var neighbor_idx: u32;
        if (connection.cell_a_index == cell_idx) {
            neighbor_idx = connection.cell_b_index;
        } else if (connection.cell_b_index == cell_idx) {
            neighbor_idx = connection.cell_a_index;
        } else {
            continue;
        }

        if (neighbor_idx >= cell_count || death_flags[neighbor_idx] == 1u) {
            continue;
        }

        let neighbor_type = cell_type_for(neighbor_idx);
        let neighbor_capacity = water_capacity(neighbor_type);
        let neighbor_water = clamp(cell_water[neighbor_idx], 0.0, neighbor_capacity);
        let neighbor_heat = max(cell_heat_energy[neighbor_idx], 0.0);
        let neighbor_temp = temperature_for(neighbor_heat, neighbor_water);

        let frozen_mult = select(1.0, 0.15, is_frozen_state(old_state) || is_frozen_state(cell_thermal_state[neighbor_idx]));
        let self_pressure = old_water / max(capacity, 0.001);
        let neighbor_pressure = neighbor_water / max(neighbor_capacity, 0.001);
        let pressure_diff = neighbor_pressure - self_pressure;
        if (abs(pressure_diff) > WATER_DEADBAND) {
            let conductance = 0.8 * sqrt(water_transfer_mult(cell_type) * water_transfer_mult(neighbor_type)) * frozen_mult;
            water_delta += pressure_diff * conductance * dt;
        }

        let temp_diff = neighbor_temp - old_temp;
        if (abs(temp_diff) > TEMP_DEADBAND) {
            let conductance = 0.65 * sqrt(heat_transfer_mult(cell_type) * heat_transfer_mult(neighbor_type));
            let heat_frozen_mult = select(1.0, 0.45, is_frozen_state(old_state) || is_frozen_state(cell_thermal_state[neighbor_idx]));
            heat_delta += temp_diff * conductance * heat_frozen_mult * dt;
        }
    }

    let next_water = clamp(old_water + water_delta, 0.0, capacity);
    let min_heat = heat_for_temperature(20.0, next_water);
    let max_heat = heat_for_temperature(180.0, next_water);
    let next_heat = clamp(old_heat + heat_delta, min_heat, max_heat);
    let next_temp = temperature_for(next_heat, next_water);
    let next_state = classify_state(next_temp, old_state);

    cell_water_next[cell_idx] = next_water;
    cell_heat_energy_next[cell_idx] = next_heat;
    cell_cached_temperature_next[cell_idx] = next_temp;
    cell_thermal_state_next[cell_idx] = next_state;
}
