// Signal Sense Compute Shader
// Two-phase signal emission:
//   Phase 1: Oculocyte cells detect targets along a forward ray -> emit on channels 0-7
//   Phase 2: ALL cells with regulation_emit_channel 8-15 -> emit unconditionally
//
// 16 channels per cell: signal_flags[cell_idx * 16 + channel]
// Each channel is a packed u32: bit 24 = source flag, bits 11-23 = scaled travel budget, bits 0-10 = value
//
// sense_type is now a bitmask: bit0=Cell, bit1=Food, bit2=Light, bit3=Wall/Cave, bit4=Self, bit5=Mossrock
// Multiple bits can be set; the oculocyte fires if ANY enabled sense type detects a target.
// bit3 (Wall/Cave) detects: world boundary sphere + solid cave voxels + water surface isosurface.

const OCULOCYTE_TYPE: u32 = 7u;
const LIGHT_THRESHOLD: f32 = 0.1;
const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_VALUE_MASK: u32 = 2047u;
const SIGNAL_HOP_SHIFT: u32 = 11u;
const SIGNAL_SOURCE_FLAG: u32 = 1u << 24u;
const SIGNAL_BUDGET_SCALE: u32 = 4u;
const THERMAL_STATE_CRITICAL: u32 = 9u;

@group(0) @binding(0)
var<storage, read_write> signal_flags: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read> positions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> rotations: array<vec4<f32>>;

@group(1) @binding(2)
var<storage, read> mode_indices: array<u32>;

@group(1) @binding(3)
var<storage, read> mode_cell_types: array<u32>;

// Oculocyte params: [sense_type(u32), ray_length_bits(u32), signal_hops(u32), signal_channel(u32)]
@group(1) @binding(4)
var<storage, read> oculocyte_params: array<vec4<u32>>;

// Regulation params: [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding(u32)]
@group(1) @binding(5)
var<storage, read> regulation_params: array<vec4<u32>>;

// Oculocyte signal values: one f32 per mode - the value emitted when target is detected.
// Kept separate from oculocyte_params to preserve the mutation system's vec4<u32> stride.
@group(1) @binding(6)
var<storage, read> oculocyte_signal_values: array<f32>;

// Oculocyte light filters: [target_r, target_g, target_b, tolerance] per mode.
@group(1) @binding(7)
var<storage, read> oculocyte_light_filters: array<vec4<f32>>;

@group(1) @binding(8)
var<storage, read> cell_thermal_state: array<u32>;

// World data for barrier, food, and light sensing
struct SignalSenseWorldParams {
    boundary_radius: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(2) @binding(0)
var<uniform> world_params: SignalSenseWorldParams;

@group(2) @binding(1)
var<storage, read> nutrient_voxels: array<u32>;

@group(2) @binding(2)
var<storage, read> light_field: array<f32>;

@group(2) @binding(3)
var<storage, read> solid_mask: array<u32>;

// binding 4: density field from surface nets - used for water surface detection
// Values are fluid density per voxel (f32); the isosurface threshold is 0.5
@group(2) @binding(4)
var<storage, read> density_field: array<f32>;

// binding 5: boulder state - for sense_type 5 (Boulder detection)
struct SenseBoulder {
    position: vec3<f32>,
    radius:   f32,
    velocity: vec3<f32>,
    dead:     u32,
    // remaining fields not needed for sensing
}
@group(2) @binding(5) var<storage, read> boulder_state_sense: array<SenseBoulder>;
@group(2) @binding(6) var<storage, read> boulder_count_sense: array<u32>;
@group(2) @binding(7) var<storage, read> light_color_field: array<vec4<f32>>;

// Shared cell spatial grid. This is built by the physics step and reused here
// so oculocyte cell sensing does not scan the whole population.
@group(3) @binding(0) var<storage, read> spatial_grid_counts: array<u32>;
@group(3) @binding(1) var<storage, read> spatial_grid_cells: array<u32>;
@group(3) @binding(2) var<storage, read> cell_grid_indices: array<u32>;

// Rotate vector by quaternion: q * v * q^-1
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

const CELL_SPATIAL_GRID_RESOLUTION: i32 = 128;
const CELL_SPATIAL_GRID_MAX_CELLS: u32 = 16u;
const INVALID_GRID_INDEX: u32 = 0xFFFFFFFFu;

fn cell_grid_index_to_coords(grid_idx: u32) -> vec3<i32> {
    let res = CELL_SPATIAL_GRID_RESOLUTION;
    let z = i32(grid_idx) / (res * res);
    let y = (i32(grid_idx) - z * res * res) / res;
    let x = i32(grid_idx) - z * res * res - y * res;
    return vec3<i32>(x, y, z);
}

fn cell_grid_coords_to_index(cell_i: vec3<i32>) -> u32 {
    let res = CELL_SPATIAL_GRID_RESOLUTION;
    return u32(cell_i.x + cell_i.y * res + cell_i.z * res * res);
}

fn world_pos_to_cell_grid_index(pos: vec3<f32>, world_size: f32, grid_cell_size: f32) -> u32 {
    let grid_pos = (pos + world_size * 0.5) / grid_cell_size;
    let grid_x = clamp(i32(grid_pos.x), 0, CELL_SPATIAL_GRID_RESOLUTION - 1);
    let grid_y = clamp(i32(grid_pos.y), 0, CELL_SPATIAL_GRID_RESOLUTION - 1);
    let grid_z = clamp(i32(grid_pos.z), 0, CELL_SPATIAL_GRID_RESOLUTION - 1);
    return cell_grid_coords_to_index(vec3<i32>(grid_x, grid_y, grid_z));
}

fn ray_hits_cell_sphere(candidate_idx: u32, idx: u32, my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32) -> bool {
    if (candidate_idx == idx) { return false; }
    let other = positions[candidate_idx];
    if (other.w < 0.5) { return false; }
    let oc = other.xyz - my_pos;
    let tca = dot(oc, forward);
    if (tca < 0.0 || tca > ray_length) { return false; }
    // positions[i].w stores mass; derive radius as clamp(mass, 0.5, 2.0)
    let r = clamp(other.w, 0.5, 2.0);
    let dist_sq = dot(oc, oc) - tca * tca;
    return dist_sq <= r * r;
}

// sense_type 0: ray-vs-sphere test against cells in spatial-grid buckets crossed by the ray.
fn sense_cells(idx: u32, my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32, cell_count: u32) -> bool {
    let world_size = world_params.boundary_radius * 2.0;
    let cs = world_size / f32(CELL_SPATIAL_GRID_RESOLUTION);
    if (cs <= 0.0) { return false; }

    let grid_origin = vec3<f32>(-world_params.boundary_radius);
    var cell_i = cell_grid_index_to_coords(world_pos_to_cell_grid_index(my_pos, world_size, cs));

    let step = vec3<i32>(
        select(-1, 1, forward.x >= 0.0),
        select(-1, 1, forward.y >= 0.0),
        select(-1, 1, forward.z >= 0.0),
    );
    let step_f = vec3<f32>(f32(step.x), f32(step.y), f32(step.z));
    let next_boundary = (vec3<f32>(cell_i) + max(step_f, vec3(0.0))) * cs + grid_origin;
    var t_max = abs((next_boundary - my_pos) / forward);
    if (abs(forward.x) < 1e-6) { t_max.x = 1e30; }
    if (abs(forward.y) < 1e-6) { t_max.y = 1e30; }
    if (abs(forward.z) < 1e-6) { t_max.z = 1e30; }
    var t_delta = abs(vec3(cs) / forward);
    if (abs(forward.x) < 1e-6) { t_delta.x = 1e30; }
    if (abs(forward.y) < 1e-6) { t_delta.y = 1e30; }
    if (abs(forward.z) < 1e-6) { t_delta.z = 1e30; }

    var t = 0.0;
    for (var iter = 0; iter < 512; iter++) {
        if (t > ray_length) { break; }
        if (any(cell_i < vec3<i32>(0)) || any(cell_i >= vec3<i32>(CELL_SPATIAL_GRID_RESOLUTION))) { break; }

        let grid_idx = cell_grid_coords_to_index(cell_i);
        let count = min(spatial_grid_counts[grid_idx], CELL_SPATIAL_GRID_MAX_CELLS);
        let base = grid_idx * CELL_SPATIAL_GRID_MAX_CELLS;
        for (var i = 0u; i < count; i++) {
            let candidate_idx = spatial_grid_cells[base + i];
            if (candidate_idx < cell_count && ray_hits_cell_sphere(candidate_idx, idx, my_pos, forward, ray_length)) {
                return true;
            }
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t = t_max.x; t_max.x += t_delta.x; cell_i.x += step.x;
        } else if (t_max.y < t_max.z) {
            t = t_max.y; t_max.y += t_delta.y; cell_i.y += step.y;
        } else {
            t = t_max.z; t_max.z += t_delta.z; cell_i.z += step.z;
        }
    }
    return false;
}

// DDA ray march through a voxel grid along the forward ray.
// Returns true if a non-zero nutrient voxel is hit within ray_length.
fn dda_march_food(my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32) -> bool {
    let res = i32(world_params.grid_resolution);
    if (res <= 0) { return false; }
    let grid_origin = vec3<f32>(world_params.grid_origin_x, world_params.grid_origin_y, world_params.grid_origin_z);
    let cs = world_params.cell_size;

    // Starting voxel
    var cell_i = vec3<i32>(floor((my_pos - grid_origin) / cs));

    // Step direction and initial t values
    let step = vec3<i32>(
        select(-1, 1, forward.x >= 0.0),
        select(-1, 1, forward.y >= 0.0),
        select(-1, 1, forward.z >= 0.0),
    );
    let step_f = vec3<f32>(f32(step.x), f32(step.y), f32(step.z));

    // t at which ray crosses next voxel boundary in each axis
    let next_boundary = (vec3<f32>(cell_i) + max(step_f, vec3(0.0))) * cs + grid_origin;
    var t_max = abs((next_boundary - my_pos) / forward);
    // Handle zero-direction components
    if (abs(forward.x) < 1e-6) { t_max.x = 1e30; }
    if (abs(forward.y) < 1e-6) { t_max.y = 1e30; }
    if (abs(forward.z) < 1e-6) { t_max.z = 1e30; }

    let t_delta = abs(vec3(cs) / forward);

    var t = 0.0;
    for (var iter = 0; iter < 512; iter++) {
        if (t > ray_length) { break; }
        // Bounds check
        if (any(cell_i < vec3<i32>(0)) || any(cell_i >= vec3<i32>(res))) { break; }

        let vi = u32(cell_i.x + cell_i.y * res + cell_i.z * res * res);
        if (nutrient_voxels[vi] == 1u) { return true; }  // Only sense active nutrients (1), not consumed (2)

        // Advance to next voxel
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t = t_max.x; t_max.x += t_delta.x; cell_i.x += step.x;
        } else if (t_max.y < t_max.z) {
            t = t_max.y; t_max.y += t_delta.y; cell_i.y += step.y;
        } else {
            t = t_max.z; t_max.z += t_delta.z; cell_i.z += step.z;
        }
    }
    return false;
}

// sense_type 2: DDA ray march for lit voxels matching the mode's light color filter.
fn dda_march_light(my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32, mode_idx: u32) -> bool {
    let res = i32(world_params.grid_resolution);
    if (res <= 0) { return false; }
    let grid_origin = vec3<f32>(world_params.grid_origin_x, world_params.grid_origin_y, world_params.grid_origin_z);
    let cs = world_params.cell_size;

    var cell_i = vec3<i32>(floor((my_pos - grid_origin) / cs));

    let step = vec3<i32>(
        select(-1, 1, forward.x >= 0.0),
        select(-1, 1, forward.y >= 0.0),
        select(-1, 1, forward.z >= 0.0),
    );
    let step_f = vec3<f32>(f32(step.x), f32(step.y), f32(step.z));

    let next_boundary = (vec3<f32>(cell_i) + max(step_f, vec3(0.0))) * cs + grid_origin;
    var t_max = abs((next_boundary - my_pos) / forward);
    if (abs(forward.x) < 1e-6) { t_max.x = 1e30; }
    if (abs(forward.y) < 1e-6) { t_max.y = 1e30; }
    if (abs(forward.z) < 1e-6) { t_max.z = 1e30; }

    let t_delta = abs(vec3(cs) / forward);

    var t = 0.0;
    for (var iter = 0; iter < 512; iter++) {
        if (t > ray_length) { break; }
        if (any(cell_i < vec3<i32>(0)) || any(cell_i >= vec3<i32>(res))) { break; }

        let vi = u32(cell_i.x + cell_i.y * res + cell_i.z * res * res);
        if (light_field[vi] > LIGHT_THRESHOLD) {
            let light_filter = oculocyte_light_filters[mode_idx];
            let tolerance = max(light_filter.a, 0.001);
            let light_color = light_color_field[vi].rgb;
            if (length(light_color - light_filter.rgb) <= tolerance) {
                return true;
            }
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t = t_max.x; t_max.x += t_delta.x; cell_i.x += step.x;
        } else if (t_max.y < t_max.z) {
            t = t_max.y; t_max.y += t_delta.y; cell_i.y += step.y;
        } else {
            t = t_max.z; t_max.z += t_delta.z; cell_i.z += step.z;
        }
    }
    return false;
}

// sense_type 3: ray-sphere intersection against world boundary, then DDA for solid voxels
// and water surface isosurface - all physical barriers an organism would bump into.
const WATER_ISO_LEVEL: f32 = 0.5;

fn sense_barrier(my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32) -> bool {
    // Check world boundary sphere
    let r = world_params.boundary_radius;
    if (r > 0.0) {
        let b = 2.0 * dot(my_pos, forward);
        let c = dot(my_pos, my_pos) - r * r;
        let discriminant = b * b - 4.0 * c;
        if (discriminant >= 0.0) {
            let sqrt_d = sqrt(discriminant);
            let t1 = (-b - sqrt_d) * 0.5;
            let t2 = (-b + sqrt_d) * 0.5;
            var t_hit = t1;
            if (t_hit <= 0.0) { t_hit = t2; }
            if (t_hit > 0.0 && t_hit <= ray_length) { return true; }
        }
    }

    // DDA march for solid cave voxels and water surface isosurface.
    let res = i32(world_params.grid_resolution);
    if (res <= 0) { return false; }
    let grid_origin = vec3<f32>(world_params.grid_origin_x, world_params.grid_origin_y, world_params.grid_origin_z);
    let cs = world_params.cell_size;
    let inv_cs = 1.0 / cs;

    var cell_i = vec3<i32>(floor((my_pos - grid_origin) * inv_cs));

    let step = vec3<i32>(
        select(-1, 1, forward.x >= 0.0),
        select(-1, 1, forward.y >= 0.0),
        select(-1, 1, forward.z >= 0.0),
    );
    let step_f = vec3<f32>(f32(step.x), f32(step.y), f32(step.z));

    let next_boundary = (vec3<f32>(cell_i) + max(step_f, vec3(0.0))) * cs + grid_origin;
    var t_max = abs((next_boundary - my_pos) / forward);
    if (abs(forward.x) < 1e-6) { t_max.x = 1e30; }
    if (abs(forward.y) < 1e-6) { t_max.y = 1e30; }
    if (abs(forward.z) < 1e-6) { t_max.z = 1e30; }

    let t_delta = abs(vec3(cs) / forward);

    // Record whether the starting voxel is inside fluid so we can detect
    // crossing the surface in either direction (from above or below).
    var in_fluid = false;
    if (all(cell_i >= vec3<i32>(0)) && all(cell_i < vec3<i32>(res))) {
        let vi0 = u32(cell_i.x + cell_i.y * res + cell_i.z * res * res);
        in_fluid = density_field[vi0] >= WATER_ISO_LEVEL;
    }

    var t = 0.0;
    for (var iter = 0; iter < 512; iter++) {
        if (t > ray_length) { break; }
        if (any(cell_i < vec3<i32>(0)) || any(cell_i >= vec3<i32>(res))) { break; }

        let vi = u32(cell_i.x + cell_i.y * res + cell_i.z * res * res);

        // Solid cave wall
        if (solid_mask[vi] != 0u) { return true; }

        // Water surface: detect the phase transition (entering or exiting fluid)
        let d = density_field[vi];
        if (!in_fluid && d >= WATER_ISO_LEVEL) { return true; }
        if (in_fluid && d < WATER_ISO_LEVEL) { return true; }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            t = t_max.x; t_max.x += t_delta.x; cell_i.x += step.x;
        } else if (t_max.y < t_max.z) {
            t = t_max.y; t_max.y += t_delta.y; cell_i.y += step.y;
        } else {
            t = t_max.z; t_max.z += t_delta.z; cell_i.z += step.z;
        }
    }
    return false;
}

// sense_type 5: ray-vs-sphere test against all live boulders.
// Same pattern as sense_cells but reads from boulder_state_sense.
fn sense_boulders(my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32) -> bool {
    let count = boulder_count_sense[0];
    for (var i = 0u; i < count; i++) {
        let b = boulder_state_sense[i];
        if (b.dead != 0u || b.radius <= 0.0) { continue; }
        let oc = b.position - my_pos;
        let tca = dot(oc, forward);
        if (tca < 0.0 || tca > ray_length) { continue; }
        let dist_sq = dot(oc, oc) - tca * tca;
        if (dist_sq <= b.radius * b.radius) { return true; }
    }
    return false;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }

    let mode_idx = mode_indices[idx];
    if (mode_idx >= arrayLength(&mode_cell_types)) { return; }
    let cell_type = mode_cell_types[mode_idx];

    // === Phase 1: Oculocyte sensing (channels 0-7) ===
    if (cell_type == OCULOCYTE_TYPE) {
        // Read oculocyte parameters for this mode
        let params = oculocyte_params[mode_idx];
        let sense_mask = params.x; // bitmask: bit0=Cell, bit1=Food, bit2=Light, bit3=Barrier, bit4=Self, bit5=Mossrock
        let ray_length = bitcast<f32>(params.y);
        let signal_hops = params.z;
        let signal_channel = min(params.w, 7u); // Clamp oculocyte to channels 0-7

        // Skip if ray_length is effectively zero and no always-fire bits are set
        if (ray_length >= 0.01 || (sense_mask & 16u) != 0u) {
            let my_pos = positions[idx].xyz;

            // Forward direction from cell's orientation quaternion
            let rot = rotations[idx];
            let forward = quat_rotate(rot, vec3<f32>(0.0, 0.0, 1.0));

            // Check each enabled sense type independently; fire if any detects a target.
            var detected = false;
            if (!detected && (sense_mask & 1u) != 0u)  { detected = sense_cells(idx, my_pos, forward, ray_length, cell_count); }
            if (!detected && (sense_mask & 2u) != 0u)  { detected = dda_march_food(my_pos, forward, ray_length); }
            if (!detected && (sense_mask & 4u) != 0u)  { detected = dda_march_light(my_pos, forward, ray_length, mode_idx); }
            if (!detected && (sense_mask & 8u) != 0u)  { detected = sense_barrier(my_pos, forward, ray_length); }
            if (!detected && (sense_mask & 16u) != 0u) { detected = true; } // Self: always fires
            if (!detected && (sense_mask & 32u) != 0u) { detected = sense_boulders(my_pos, forward, ray_length); }

            if (detected) {
                // Write signal to the correct channel slot.
                // Format: bit24 = source flag, bits 11-23 = scaled travel budget, bits 0-10 = value.
                // Use atomicStore - the clear pass already zeroed this slot, and only one
                // oculocyte thread owns each (cell, channel) pair, so there is no contention.
                // atomicAdd on a packed bitfield would corrupt hops/value/source-flag if two
                // threads ever wrote the same slot concurrently.
                let raw_value = oculocyte_signal_values[mode_idx];
                let clamped_value = min(u32(max(raw_value, 0.0)), SIGNAL_VALUE_MASK);
                let emit_value = select(1u, clamped_value, clamped_value > 0u); // Ensure at least 1 so propagation fires
                let signal_budget = signal_hops * SIGNAL_BUDGET_SCALE;
                let signal_value = SIGNAL_SOURCE_FLAG | (signal_budget << SIGNAL_HOP_SHIFT) | emit_value;
                atomicStore(&signal_flags[idx * SIGNAL_CHANNELS + signal_channel], signal_value);
            }
        }
    }

    // Critical heat saturates only sensory channels 0-7. Regulation channels 8-15
    // are intentionally left untouched so thermal crisis does not impersonate
    // authored regulation emissions.
    if (cell_thermal_state[idx] == THERMAL_STATE_CRITICAL) {
        let critical_signal = SIGNAL_SOURCE_FLAG | SIGNAL_VALUE_MASK;
        let base = idx * SIGNAL_CHANNELS;
        for (var ch = 0u; ch < 8u; ch++) {
            atomicStore(&signal_flags[base + ch], critical_signal);
        }
    }

    // === Phase 2: Regulation emission (channels 8-15, any cell type) ===
    let reg_params = regulation_params[mode_idx];
    let reg_channel = reg_params.x;
    // reg_channel == 0xFFFFFFFF means disabled (stored as -1 cast to u32)
    if (reg_channel >= 8u && reg_channel <= 15u) {
        let reg_value_bits = reg_params.y;
        let reg_hops = reg_params.z;
        let reg_value = bitcast<f32>(reg_value_bits);

        // Clamp value to 11-bit range (0-2047)
        let clamped_value = min(u32(max(reg_value, 0.0)), SIGNAL_VALUE_MASK);

        if (clamped_value > 0u && reg_hops > 0u) {
            // Source flag + hops + value.
            // Use atomicStore - the clear pass already zeroed this slot, and each cell
            // owns exactly one regulation channel, so there is no write contention.
            let signal_budget = reg_hops * SIGNAL_BUDGET_SCALE;
            let signal_packed = SIGNAL_SOURCE_FLAG | (signal_budget << SIGNAL_HOP_SHIFT) | clamped_value;
            atomicStore(&signal_flags[idx * SIGNAL_CHANNELS + reg_channel], signal_packed);
        }
    }
}
