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
var<storage, read> cell_water: array<f32>;
@group(1) @binding(1)
var<storage, read> cell_heat_energy: array<f32>;
@group(1) @binding(2)
var<storage, read> cell_cached_temperature: array<f32>;
@group(1) @binding(3)
var<storage, read> cell_thermal_state: array<u32>;
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
@group(1) @binding(10)
var<storage, read_write> cell_water_delta: array<atomic<i32>>;
@group(1) @binding(11)
var<storage, read_write> cell_heat_delta: array<atomic<i32>>;

@group(2) @binding(0)
var<storage, read> death_flags: array<u32>;
@group(2) @binding(1)
var<storage, read> mode_indices: array<u32>;
@group(2) @binding(2)
var<storage, read> mode_cell_types: array<u32>;

@group(2) @binding(3)
var<uniform> water_params: WaterGridParams;
@group(2) @binding(4)
var<storage, read> fluid_state: array<atomic<u32>>;
@group(2) @binding(5)
var<storage, read> voxel_temperature: array<atomic<u32>>;
@group(2) @binding(6)
var<storage, read> geothermal_heat: array<f32>;
@group(2) @binding(7)
var<storage, read> mode_properties_v14: array<vec4<f32>>;
@group(2) @binding(8)
var<storage, read> mode_properties_v15: array<vec4<f32>>;
@group(2) @binding(9)
var<storage, read> signal_flags: array<atomic<u32>>;
@group(2) @binding(10)
var<storage, read> adhesion_settings_v0: array<vec4<f32>>;

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const BOND_FLAG_BARRIER_BALL: u32 = 2u;

const CELL_TYPE_MYOCYTE: u32 = 9u;
const CELL_TYPE_VASCULOCYTE: u32 = 12u;
const CELL_TYPE_SIPHONOCYTE: u32 = 17u;
const CELL_TYPE_PLUMOCYTE: u32 = 18u;

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
const WATER_MASS_TEMP_SHOCK_DAMPING: f32 = 0.75;
const BASE_ENV_TEMP: f32 = 105.0;
const TEMP_DEADBAND: f32 = 0.25;
const WATER_DEADBAND: f32 = 0.0025;
const HOT_DRY_AIR_START_TEMP: f32 = 120.0;
const HOT_DRY_AIR_FULL_TEMP: f32 = 170.0;
const HOT_DRY_AIR_TARGET_FRACTION: f32 = 0.18;
const HOT_DRY_AIR_MAX_LOSS_RATE: f32 = 0.014;
const SHORT_BOND_REST_LENGTH: f32 = 0.75;
const LONG_BOND_REST_LENGTH: f32 = 1.6;
const SHORT_BOND_ENV_SHIELD: f32 = 0.18;
const LONG_BOND_HEAT_EXPULSION_RATE: f32 = 0.018;
const MYOCYTE_CONTRACTION_HEAT: f32 = 36.0;
const MYOCYTE_VIBRATION_HEAT: f32 = 5.0;
const COMFORT_LOW_TEMP: f32 = 100.0;
const EXTREME_HEAT_START_TEMP: f32 = 140.0;
const EXTREME_HEAT_FULL_TEMP: f32 = 170.0;
const VOXEL_TEMP_MIN_C: f32 = -50.0;
const VOXEL_TEMP_MAX_C: f32 = 150.0;
const VOXEL_TEMP_FP: f32 = 256.0;
const FLUID_TYPE_MASK: u32 = 0x7u;
const DELTA_FIXED_SCALE: f32 = 65536.0;

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
    if (cell_type == CELL_TYPE_SIPHONOCYTE) {
        return 1.5;
    }
    return 1.0;
}

fn water_transfer_mult(cell_type: u32) -> f32 {
    if (cell_type == CELL_TYPE_VASCULOCYTE) {
        return 4.0;
    }
    if (cell_type == CELL_TYPE_SIPHONOCYTE) {
        return 4.0;
    }
    return 1.0;
}

fn heat_transfer_mult(cell_type: u32) -> f32 {
    if (cell_type == CELL_TYPE_VASCULOCYTE) {
        return 4.0;
    }
    if (cell_type == CELL_TYPE_SIPHONOCYTE) {
        return 3.0;
    }
    if (cell_type == CELL_TYPE_PLUMOCYTE) {
        return 1.35;
    }
    return 1.0;
}

fn water_pair_conductance(cell_type: u32, neighbor_type: u32) -> f32 {
    let self_siphon = cell_type == CELL_TYPE_SIPHONOCYTE;
    let neighbor_siphon = neighbor_type == CELL_TYPE_SIPHONOCYTE;
    let self_vascular = cell_type == CELL_TYPE_VASCULOCYTE;
    let neighbor_vascular = neighbor_type == CELL_TYPE_VASCULOCYTE;

    if ((self_siphon && neighbor_vascular) || (neighbor_siphon && self_vascular)) {
        return 4.0;
    }
    if (self_siphon || neighbor_siphon) {
        return 0.0;
    }
    return 0.8 * sqrt(water_transfer_mult(cell_type) * water_transfer_mult(neighbor_type));
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

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / DELTA_FIXED_SCALE;
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

fn siphon_intake_amount(siphon_idx: u32, mode_idx: u32, dt: f32, env: VoxelEnvironment) -> f32 {
    var intake_rate = 1.0;
    var impulse = 0.0;
    var mode = 0u;
    var signal_active = false;
    if (mode_idx < arrayLength(&mode_properties_v14)) {
        let props = mode_properties_v14[mode_idx];
        intake_rate = max(props.x, 0.0);
        impulse = max(props.z, 0.0);
        let packed = u32(clamp(props.w, 0.0, 262143.0));
        mode = min((packed / 32u) & 3u, 3u);
        let channel = packed & 15u;
        let invert = (packed & 16u) != 0u;
        let threshold = f32(packed / 128u);
        let raw_signal = atomicLoad(&signal_flags[siphon_idx * 16u + channel]);
        let above = f32(raw_signal & 2047u) >= threshold;
        signal_active = select(above, !above, invert);
    }

    var stroke_gate = 1.0;
    if (mode == 0u || mode == 1u) {
        let stroke_phase = fract(params.current_time * mix(0.85, 2.4, clamp(impulse / 3.0, 0.0, 1.0)) + f32(siphon_idx & 1023u) * 0.013);
        stroke_gate = smoothstep(0.05, 0.18, stroke_phase) * (1.0 - smoothstep(0.36, 0.48, stroke_phase));
        if (stroke_gate <= 0.001) {
            return 0.0;
        }
        if (mode == 1u && !signal_active) {
            return 0.0;
        }
    } else if (mode == 2u) {
        if (!signal_active) {
            return 0.0;
        }
    } else {
        return 0.0;
    }

    if (!env.valid) {
        return 0.0;
    }

    var intake_source = 0.0;
    if (env.fluid_type == 1u) {
        intake_source = env.fill_fraction;
    } else if (env.fluid_type == 3u) {
        intake_source = env.fill_fraction * 0.25;
    } else if (env.fluid_type == 2u || env.fluid_type == 4u) {
        intake_source = env.fill_fraction * 0.08;
    }
    return intake_source * intake_rate * mix(0.45, 1.25, stroke_gate) * dt;
}

fn is_frozen_state(state: u32) -> bool {
    return state <= STATE_FROZEN;
}

fn classify_state(temp: f32, old_state: u32, entombed_in_ice: bool) -> u32 {
    if (entombed_in_ice && old_state == STATE_FROZEN && temp < 65.0) {
        return STATE_FROZEN;
    }
    if (old_state == STATE_HEAT_SHOCK && temp > 145.0) {
        return STATE_HEAT_SHOCK;
    }
    if (entombed_in_ice && temp <= 45.0) { return STATE_DEEP_FROZEN; }
    if (entombed_in_ice && temp < 60.0) { return STATE_FROZEN; }
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
        atomicStore(&cell_water_delta[cell_idx], 0);
        atomicStore(&cell_heat_delta[cell_idx], 0);
        return;
    }

    let dt = max(params.delta_time * PHYSIOLOGY_TICK_SCALE, 0.0);
    let cell_type = cell_type_for(cell_idx);
    let mode_idx = mode_indices[cell_idx];
    let capacity = water_capacity(cell_type);
    let old_water = clamp(cell_water[cell_idx], 0.0, capacity);
    let old_heat = max(cell_heat_energy[cell_idx], 0.0);
    let old_state = cell_thermal_state[cell_idx];
    let old_temp = temperature_for(old_heat, old_water);

    var water_delta = 0.0;
    var heat_delta = 0.0;
    water_delta += fixed_to_float(atomicLoad(&cell_water_delta[cell_idx]));
    heat_delta += fixed_to_float(atomicLoad(&cell_heat_delta[cell_idx]));
    atomicStore(&cell_water_delta[cell_idx], 0);
    atomicStore(&cell_heat_delta[cell_idx], 0);

    let env = read_voxel_environment(positions[cell_idx].xyz);
    var exposure_mult = 1.0;
    if (cell_type == CELL_TYPE_PLUMOCYTE && mode_idx < arrayLength(&mode_properties_v15)) {
        let plum = mode_properties_v15[mode_idx];
        exposure_mult += clamp(plum.x, 0.0, 1.0) * max(plum.w, 0.0);
    }

    // Passive environment coupling. The cell samples only the voxel it occupies
    // and never writes moisture or heat back into the fluid grid.
    if (cell_type == CELL_TYPE_SIPHONOCYTE) {
        water_delta += siphon_intake_amount(cell_idx, mode_idx, dt, env);
    } else if (cell_type != CELL_TYPE_VASCULOCYTE && env.valid) {
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
        if (env.fluid_type == 0u && env.temp_internal > HOT_DRY_AIR_START_TEMP) {
            let hot_dry_factor = clamp(
                (env.temp_internal - HOT_DRY_AIR_START_TEMP) / (HOT_DRY_AIR_FULL_TEMP - HOT_DRY_AIR_START_TEMP),
                0.0,
                1.0
            );
            let fill_fraction = old_water / max(capacity, 0.001);
            water_target_fraction = mix(water_target_fraction, HOT_DRY_AIR_TARGET_FRACTION, hot_dry_factor);
            water_exchange = mix(water_exchange, 0.012, hot_dry_factor);
            water_delta -= HOT_DRY_AIR_MAX_LOSS_RATE * hot_dry_factor * fill_fraction * dt * exposure_mult;
        }
        water_delta += water_exchange * exposure_mult * (water_target_fraction * capacity - old_water) * dt;
    } else {
        if (cell_type != CELL_TYPE_SIPHONOCYTE && cell_type != CELL_TYPE_VASCULOCYTE) {
            water_delta += 0.015 * (capacity - old_water) * dt;
        }
    }
    let env_exchange = select(0.04, 0.02, cell_type == CELL_TYPE_VASCULOCYTE);
    let env_temp = select(BASE_ENV_TEMP, env.temp_internal, env.valid);

    // Heat stress slowly dehydrates cells; Plumocyte/Siphonocyte specialization is later.
    if (old_temp > 140.0) {
        water_delta -= (old_temp - 140.0) * 0.0015 * dt * exposure_mult;
    }

    // Myocyte heat is generated by contraction change, not by static posture.
    if (cell_type == CELL_TYPE_MYOCYTE) {
        let current_contraction = clamp(muscle_contraction[cell_idx], 0.0, 1.0);
        let previous_contraction = clamp(prev_muscle_contraction[cell_idx], 0.0, 1.0);
        let contraction_delta = abs(current_contraction - previous_contraction);
        let contraction_speed = contraction_delta / max(dt, 0.001);
        heat_delta += contraction_delta * (MYOCYTE_CONTRACTION_HEAT + contraction_speed * MYOCYTE_VIBRATION_HEAT);
        prev_muscle_contraction[cell_idx] = current_contraction;
    } else {
        prev_muscle_contraction[cell_idx] = 0.0;
    }

    let base_env_exchange = env_exchange;
    heat_delta += max(env_temp - old_temp, 0.0) * base_env_exchange * exposure_mult * dt;

    let next_water = clamp(old_water + water_delta, 0.0, capacity);
    // Water changes alter the cell's thermal mass, so dry cells still swing
    // faster than hydrated cells. Dampen that mass shock so rehydration does
    // not instantly crash body temperature toward freezing.
    let water_adjusted_heat = mix(old_heat, heat_for_temperature(old_temp, next_water), WATER_MASS_TEMP_SHOCK_DAMPING);
    let min_heat = heat_for_temperature(20.0, next_water);
    let max_heat = heat_for_temperature(180.0, next_water);
    let next_heat = clamp(water_adjusted_heat + heat_delta, min_heat, max_heat);
    let next_temp = temperature_for(next_heat, next_water);
    let entombed_in_ice = env.valid && env.fluid_type == 2u && env.fill_fraction > 0.5;
    let next_state = classify_state(next_temp, old_state, entombed_in_ice);

    cell_water_next[cell_idx] = next_water;
    cell_heat_energy_next[cell_idx] = next_heat;
    cell_cached_temperature_next[cell_idx] = next_temp;
    cell_thermal_state_next[cell_idx] = next_state;
}
