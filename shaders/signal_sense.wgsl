// Signal Sense Compute Shader
// Oculocyte cells detect targets along a single forward ray.
// Exits early on first hit for all sense types.
//
// sense_type 0 = Cell (ray-vs-sphere test against each cell)
// sense_type 1 = Food (DDA ray march through nutrient voxels)
// sense_type 2 = Light (DDA ray march through light voxels)
// sense_type 3 = Barrier (ray-sphere vs world boundary + DDA for solid voxels)

const OCULOCYTE_TYPE: u32 = 7u;
const LIGHT_THRESHOLD: f32 = 0.1;

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

@group(1) @binding(4)
var<storage, read> oculocyte_params: array<vec4<u32>>;

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
// Returns the flat voxel index of the first hit, or -1 if no hit within ray_length.
// check_fn is inlined by the caller — we return the voxel index and let the caller check the buffer.
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
        if (nutrient_voxels[vi] != 0u) { return true; }

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

    // DDA march for solid cave voxels
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

    var t = 0.0;
    for (var iter = 0; iter < 512; iter++) {
        if (t > ray_length) { break; }
        if (any(cell_i < vec3<i32>(0)) || any(cell_i >= vec3<i32>(res))) { break; }

        let vi = u32(cell_i.x + cell_i.y * res + cell_i.z * res * res);
        if (solid_mask[vi] != 0u) { return true; }

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

    // Only oculocytes perform sensing
    let mode_idx = mode_indices[idx];
    let cell_type = mode_cell_types[mode_idx];
    if (cell_type != OCULOCYTE_TYPE) { return; }

    // Read oculocyte parameters for this mode
    let params = oculocyte_params[mode_idx];
    let sense_type = params.x;
    let ray_length = bitcast<f32>(params.y);
    let signal_hops = params.z;

    // Skip if ray_length is effectively zero
    if (ray_length < 0.01) { return; }

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
        default: {}
    }

    if (detected) {
        // Write signal value with flow direction encoded
        // Format: DDDDDHHHHHVVVVVVVVVVVV (5 bits direction flag, 5 bits hops, 11 bits value)
        // Direction flag: 1 = signal source, 0 = propagated signal
        let signal_value = (1u << 16u) | (signal_hops << 11u) | 20u; // Source flag + hops + base value
        atomicAdd(&signal_flags[idx], signal_value);
    }
}
