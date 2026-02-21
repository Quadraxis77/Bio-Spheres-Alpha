// Signal Sense Compute Shader
// Oculocyte cells detect nearby targets within a directional cone (field of view).
// FOV curve: quadratic — range=25 -> ~150° total, range=50 -> ~10° total
// Matches CPU signal_system.rs behavior.
//
// Detection method: AABB cone scan — iterate voxels in the bounding sphere,
// check data first (cheap buffer read), then cone geometry only on non-empty voxels.
// This is efficient because most voxels are empty, so the expensive cone test is rarely run.
//
// sense_type 0 = Cell (detect any nearby cell in forward cone)
// sense_type 1 = Food (detect nutrient voxels in forward cone)
// sense_type 2 = Light (detect lit voxels in forward cone)
// sense_type 3 = Barrier (world boundary intersection + cave solid voxels in cone)

const OCULOCYTE_TYPE: u32 = 7u;
const LIGHT_THRESHOLD: f32 = 0.1;

@group(0) @binding(0)
var<storage, read_write> signal_flags: array<u32>;

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

// Check if a point is inside the sensing cone (within range AND within FOV angle)
fn point_in_cone(my_pos: vec3<f32>, forward: vec3<f32>, sense_range_sq: f32, cos_half_fov: f32, point: vec3<f32>) -> bool {
    let diff = point - my_pos;
    let dist_sq = dot(diff, diff);
    if (dist_sq > sense_range_sq || dist_sq < 0.001) { return false; }
    return dot(forward, diff * inverseSqrt(dist_sq)) >= cos_half_fov;
}

// sense_type 0: detect any nearby cell within forward cone
fn sense_cells(idx: u32, my_pos: vec3<f32>, forward: vec3<f32>, sense_range_sq: f32, cos_half_fov: f32, cell_count: u32) -> bool {
    for (var i = 0u; i < cell_count; i++) {
        if (i == idx) { continue; }

        let other_pos = positions[i].xyz;
        let diff = other_pos - my_pos;
        let dist_sq = dot(diff, diff);

        if (dist_sq > sense_range_sq || dist_sq < 0.001) { continue; }

        let dir = normalize(diff);
        if (dot(forward, dir) >= cos_half_fov) {
            return true;
        }
    }
    return false;
}

// sense_type 1: detect nutrient voxels within the FOV cone
// AABB scan: iterate bounding sphere voxels, check data first, then cone geometry
fn sense_food(my_pos: vec3<f32>, forward: vec3<f32>, sense_range: f32, sense_range_sq: f32, cos_half_fov: f32) -> bool {
    let res = i32(world_params.grid_resolution);
    if (res <= 0) { return false; }
    let grid_origin = vec3<f32>(world_params.grid_origin_x, world_params.grid_origin_y, world_params.grid_origin_z);
    let cs = world_params.cell_size;

    // Compute AABB of bounding sphere in grid coordinates
    let min_g = clamp(vec3<i32>(floor((my_pos - vec3(sense_range) - grid_origin) / cs)), vec3<i32>(0), vec3<i32>(res - 1));
    let max_g = clamp(vec3<i32>(floor((my_pos + vec3(sense_range) - grid_origin) / cs)), vec3<i32>(0), vec3<i32>(res - 1));

    for (var gz = min_g.z; gz <= max_g.z; gz++) {
        for (var gy = min_g.y; gy <= max_g.y; gy++) {
            for (var gx = min_g.x; gx <= max_g.x; gx++) {
                let vi = gx + gy * res + gz * res * res;
                // Check data first (cheap) — skip empty voxels
                if (nutrient_voxels[u32(vi)] != 0u) {
                    let vc = grid_origin + (vec3<f32>(f32(gx), f32(gy), f32(gz)) + 0.5) * cs;
                    if (point_in_cone(my_pos, forward, sense_range_sq, cos_half_fov, vc)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// sense_type 2: detect lit voxels within the FOV cone
fn sense_light(my_pos: vec3<f32>, forward: vec3<f32>, sense_range: f32, sense_range_sq: f32, cos_half_fov: f32) -> bool {
    let res = i32(world_params.grid_resolution);
    if (res <= 0) { return false; }
    let grid_origin = vec3<f32>(world_params.grid_origin_x, world_params.grid_origin_y, world_params.grid_origin_z);
    let cs = world_params.cell_size;

    let min_g = clamp(vec3<i32>(floor((my_pos - vec3(sense_range) - grid_origin) / cs)), vec3<i32>(0), vec3<i32>(res - 1));
    let max_g = clamp(vec3<i32>(floor((my_pos + vec3(sense_range) - grid_origin) / cs)), vec3<i32>(0), vec3<i32>(res - 1));

    for (var gz = min_g.z; gz <= max_g.z; gz++) {
        for (var gy = min_g.y; gy <= max_g.y; gy++) {
            for (var gx = min_g.x; gx <= max_g.x; gx++) {
                let vi = gx + gy * res + gz * res * res;
                if (light_field[u32(vi)] > LIGHT_THRESHOLD) {
                    let vc = grid_origin + (vec3<f32>(f32(gx), f32(gy), f32(gz)) + 0.5) * cs;
                    if (point_in_cone(my_pos, forward, sense_range_sq, cos_half_fov, vc)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// sense_type 3: detect barriers within the FOV cone
// Checks world boundary (sphere intersection) + cave solid voxels (AABB scan)
fn sense_barrier(my_pos: vec3<f32>, forward: vec3<f32>, sense_range: f32, sense_range_sq: f32, cos_half_fov: f32) -> bool {
    // Check world boundary: is the nearest boundary point within the cone?
    let r = world_params.boundary_radius;
    if (r > 0.0) {
        let dist_to_boundary = r - length(my_pos);
        if (dist_to_boundary > 0.0 && dist_to_boundary <= sense_range) {
            // Nearest boundary point is along the radial outward direction
            let outward = normalize(my_pos);
            if (dot(forward, outward) >= cos_half_fov) {
                return true;
            }
            // Also check forward ray intersection for non-radial hits
            let b = 2.0 * dot(my_pos, forward);
            let c = dot(my_pos, my_pos) - r * r;
            let discriminant = b * b - 4.0 * c;
            if (discriminant >= 0.0) {
                let sqrt_d = sqrt(discriminant);
                let t1 = (-b - sqrt_d) * 0.5;
                let t2 = (-b + sqrt_d) * 0.5;
                var t_hit = t1;
                if (t_hit <= 0.0) { t_hit = t2; }
                if (t_hit > 0.0 && t_hit <= sense_range) {
                    return true;
                }
            }
        }
    }

    // Check cave solid voxels within the FOV cone
    let res = i32(world_params.grid_resolution);
    if (res <= 0) { return false; }
    let grid_origin = vec3<f32>(world_params.grid_origin_x, world_params.grid_origin_y, world_params.grid_origin_z);
    let cs = world_params.cell_size;

    let min_g = clamp(vec3<i32>(floor((my_pos - vec3(sense_range) - grid_origin) / cs)), vec3<i32>(0), vec3<i32>(res - 1));
    let max_g = clamp(vec3<i32>(floor((my_pos + vec3(sense_range) - grid_origin) / cs)), vec3<i32>(0), vec3<i32>(res - 1));

    for (var gz = min_g.z; gz <= max_g.z; gz++) {
        for (var gy = min_g.y; gy <= max_g.y; gy++) {
            for (var gx = min_g.x; gx <= max_g.x; gx++) {
                let vi = gx + gy * res + gz * res * res;
                if (solid_mask[u32(vi)] != 0u) {
                    let vc = grid_origin + (vec3<f32>(f32(gx), f32(gy), f32(gz)) + 0.5) * cs;
                    if (point_in_cone(my_pos, forward, sense_range_sq, cos_half_fov, vc)) {
                        return true;
                    }
                }
            }
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
    let sense_range = bitcast<f32>(params.y);
    let signal_hops = params.z;

    // Skip if sense_range is effectively zero
    if (sense_range < 0.01) { return; }

    let my_pos = positions[idx].xyz;
    let sense_range_sq = sense_range * sense_range;

    // Forward direction from cell's orientation quaternion
    let rot = rotations[idx];
    let forward = quat_rotate(rot, vec3<f32>(0.0, 0.0, 1.0));

    // FOV curve: quadratic so range drops faster as FOV widens (matches CPU)
    let t = clamp(sense_range / 50.0, 0.0, 0.998);
    let cos_half_fov = t * t;

    var detected = false;

    switch (sense_type) {
        case 0u: { detected = sense_cells(idx, my_pos, forward, sense_range_sq, cos_half_fov, cell_count); }
        case 1u: { detected = sense_food(my_pos, forward, sense_range, sense_range_sq, cos_half_fov); }
        case 2u: { detected = sense_light(my_pos, forward, sense_range, sense_range_sq, cos_half_fov); }
        case 3u: { detected = sense_barrier(my_pos, forward, sense_range, sense_range_sq, cos_half_fov); }
        default: {}
    }

    if (detected) {
        // Write signal value: hops + 1 (value > 0 means "has signal", value > 1 means "propagates further")
        signal_flags[idx] = signal_hops + 1u;
    }
}
