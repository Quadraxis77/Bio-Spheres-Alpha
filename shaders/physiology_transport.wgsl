// Pairwise water/heat exchange over adhesion links.
// This mirrors nutrient transport's sender-side pattern: each active connection is
// processed once from cell_a, then equal/opposite fixed-point deltas are atomically
// accumulated for the apply/update pass.

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

@group(0) @binding(0)
var<uniform> params: PhysicsParams;
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read> cell_water: array<f32>;
@group(1) @binding(1)
var<storage, read> cell_heat_energy: array<f32>;
@group(1) @binding(3)
var<storage, read> cell_thermal_state: array<u32>;
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

@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;
@group(3) @binding(4)
var<storage, read> adhesion_counts: array<u32>;
@group(3) @binding(5)
var<storage, read> cell_adhesion_indices: array<i32>;

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const BOND_FLAG_BARRIER_BALL: u32 = 2u;
const CELL_TYPE_VASCULOCYTE: u32 = 12u;
const CELL_TYPE_SIPHONOCYTE: u32 = 17u;
const STATE_FROZEN: u32 = 1u;
const PHYSIOLOGY_TICK_SCALE: f32 = 4.0;
const DRY_THERMAL_MASS: f32 = 1.0;
const WATER_THERMAL_MASS_FACTOR: f32 = 1.0;
const WATER_DEADBAND: f32 = 0.0025;
const TEMP_DEADBAND: f32 = 0.25;
const DELTA_FIXED_SCALE: f32 = 65536.0;

fn float_to_fixed(value: f32) -> i32 {
    return i32(clamp(value * DELTA_FIXED_SCALE, -2147483000.0, 2147483000.0));
}

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

fn is_frozen_state(state: u32) -> bool {
    return state <= STATE_FROZEN;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count || death_flags[cell_idx] == 1u) {
        return;
    }

    let cell_type = cell_type_for(cell_idx);
    let capacity = water_capacity(cell_type);
    let old_water = clamp(cell_water[cell_idx], 0.0, capacity);
    let old_heat = max(cell_heat_energy[cell_idx], 0.0);
    let old_temp = temperature_for(old_heat, old_water);
    let old_state = cell_thermal_state[cell_idx];
    let self_pressure = old_water / max(capacity, 0.001);
    let dt = max(params.delta_time * PHYSIOLOGY_TICK_SCALE, 0.0);

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
        if (connection.is_active == 0u ||
            connection.cell_a_index != cell_idx ||
            (connection.bond_flags & BOND_FLAG_BARRIER_BALL) != 0u) {
            continue;
        }

        let neighbor_idx = connection.cell_b_index;
        if (neighbor_idx >= cell_count || death_flags[neighbor_idx] == 1u) {
            continue;
        }

        let neighbor_type = cell_type_for(neighbor_idx);
        let neighbor_capacity = water_capacity(neighbor_type);
        let neighbor_water = clamp(cell_water[neighbor_idx], 0.0, neighbor_capacity);
        let neighbor_heat = max(cell_heat_energy[neighbor_idx], 0.0);
        let neighbor_temp = temperature_for(neighbor_heat, neighbor_water);
        let neighbor_state = cell_thermal_state[neighbor_idx];
        let frozen_mult = select(1.0, 0.15, is_frozen_state(old_state) || is_frozen_state(neighbor_state));

        let neighbor_pressure = neighbor_water / max(neighbor_capacity, 0.001);
        let pressure_diff = neighbor_pressure - self_pressure;
        if (abs(pressure_diff) > WATER_DEADBAND) {
            let conductance = water_pair_conductance(cell_type, neighbor_type) * frozen_mult;
            let water_delta = pressure_diff * conductance * dt;
            let fixed = float_to_fixed(water_delta);
            atomicAdd(&cell_water_delta[cell_idx], fixed);
            atomicAdd(&cell_water_delta[neighbor_idx], -fixed);
        }

        let temp_diff = neighbor_temp - old_temp;
        if (abs(temp_diff) > TEMP_DEADBAND) {
            let conductance = 0.65 * sqrt(heat_transfer_mult(cell_type) * heat_transfer_mult(neighbor_type));
            let heat_frozen_mult = select(1.0, 0.45, is_frozen_state(old_state) || is_frozen_state(neighbor_state));
            let heat_delta = temp_diff * conductance * heat_frozen_mult * dt;
            let fixed = float_to_fixed(heat_delta);
            atomicAdd(&cell_heat_delta[cell_idx], fixed);
            atomicAdd(&cell_heat_delta[neighbor_idx], -fixed);
        }
    }
}
