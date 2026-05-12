// Signal Sense Compute Shader
// Two-phase signal emission:
//   Phase 1: Oculocyte cells detect targets along a forward ray → emit on channels 0-7
//   Phase 2: ALL cells with regulation_emit_channel 8-15 → emit unconditionally
//
// 16 channels per cell: signal_flags[cell_idx * 16 + channel]
// Each channel is a packed u32: bits 16+ = direction flag, bits 11-15 = hops, bits 0-10 = value
//
// sense_type 0 = Cell (ray-vs-sphere test against each cell)
// sense_type 1 = Food (DDA ray march through nutrient voxels)
// sense_type 2 = Light (DDA ray march through light voxels)
// sense_type 3 = Barrier (ray-sphere vs world boundary + DDA for solid voxels + water surface isosurface)
// sense_type 4 = Self (always detects)

const OCULOCYTE_TYPE: u32 = 7u;
const LIGHT_THRESHOLD: f32 = 0.1;
const SIGNAL_CHANNELS: u32 = 16u;

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

// binding 4: density field from surface nets — used for water surface detection
// Values are fluid density per voxel (f32); the isosurface threshold is 0.5
@group(2) @binding(4)
var<storage, read> density_field: array<f32>;

// Rotate vector by quaternion: q * v * q^-1
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// sense_type 0: ray-vs-sphere test against every cell. Exits on first hit.
fn sense_cells(idx: u32, my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32, cell_count: u32) -> bool {
    for (var i = 0u; i < cell_count; i++) {
        if (i == idx) { continue; }
        let oc = positions[i].xyz - my_pos;
        let tca = dot(oc, forward);
        if (tca < 0.0 || tca > ray_length) { continue; }
        // positions[i].w stores mass; derive radius as clamp(mass, 0.5, 2.0)
        let r = clamp(positions[i].w, 0.5, 2.0);
        let dist_sq = dot(oc, oc) - tca * tca;
        if (dist_sq <= r * r) { return true; }
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

// sense_type 2: DDA ray march for lit voxels
fn dda_march_light(my_pos: vec3<f32>, forward: vec3<f32>, ray_length: f32) -> bool {
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
        if (light_field[vi] > LIGHT_THRESHOLD) { return true; }

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
// and water surface isosurface — all physical barriers an organism would bump into.
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }

    let mode_idx = mode_indices[idx];
    let cell_type = mode_cell_types[mode_idx];

    // === Phase 1: Oculocyte sensing (channels 0-7) ===
    if (cell_type == OCULOCYTE_TYPE) {
        // Read oculocyte parameters for this mode
        let params = oculocyte_params[mode_idx];
        let sense_type = params.x;
        let ray_length = bitcast<f32>(params.y);
        let signal_hops = params.z;
        let signal_channel = min(params.w, 7u); // Clamp oculocyte to channels 0-7

        // Skip if ray_length is effectively zero
        if (ray_length >= 0.01) {
            let my_pos = positions[idx].xyz;

            // Forward direction from cell's orientation quaternion
            let rot = rotations[idx];
            let forward = quat_rotate(rot, vec3<f32>(0.0, 0.0, 1.0));

            var detected = false;

            switch (sense_type) {
                case 0u: { detected = sense_cells(idx, my_pos, forward, ray_length, cell_count); }
                case 1u: { detected = dda_march_food(my_pos, forward, ray_length); }
                case 2u: { detected = dda_march_light(my_pos, forward, ray_length); }
                case 3u: { detected = sense_barrier(my_pos, forward, ray_length); }
                case 4u: { detected = true; } // Self-sense: always fires
                default: {}
            }

            if (detected) {
                // Write signal to the correct channel slot.
                // Format: bit16 = source flag, bits 11-15 = hops, bits 0-10 = value
                // Use atomicStore — the clear pass already zeroed this slot, and only one
                // oculocyte thread owns each (cell, channel) pair, so there is no contention.
                // atomicAdd on a packed bitfield would corrupt hops/value/source-flag if two
                // threads ever wrote the same slot concurrently.
                let signal_value = (1u << 16u) | (signal_hops << 11u) | 20u; // Source flag + hops + base value
                atomicStore(&signal_flags[idx * SIGNAL_CHANNELS + signal_channel], signal_value);
            }
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
        let clamped_value = min(u32(max(reg_value, 0.0)), 2047u);

        if (clamped_value > 0u && reg_hops > 0u) {
            // Source flag + hops + value.
            // Use atomicStore — the clear pass already zeroed this slot, and each cell
            // owns exactly one regulation channel, so there is no write contention.
            let signal_packed = (1u << 16u) | (reg_hops << 11u) | clamped_value;
            atomicStore(&signal_flags[idx * SIGNAL_CHANNELS + reg_channel], signal_packed);
        }
    }
}
