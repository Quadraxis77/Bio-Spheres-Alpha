// Moss Growth / Erosion / Decay Compute Shader
//
// Runs once per physics step over the 128³ voxel grid.
// Growth conditions:
//   - Solid-adjacent (cave surface)
//   - Lit above threshold
//   - Not submerged
//   - Within water proximity range (~10 voxels), with smooth falloff
//   - Passes a noise-based coverage test (patchy, organic distribution)
//
// Water flow and submersion erode existing moss.

struct MossParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    growth_rate: f32,        // moss units per second at full light
    erosion_rate: f32,       // moss units per second under flowing water
    decay_rate: f32,         // moss units per second when conditions not met
    min_light: f32,          // minimum light_field value for growth
    delta_time: f32,
    world_radius: f32,
    water_radius: f32,       // max voxel distance to search for water
    wetness_evaporation: f32, // how fast wetness dries out (units per second)
    wetness_radius: f32,     // how far wetness spreads from source (voxels)
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: MossParams;
@group(0) @binding(1) var<storage, read_write> moss_density: array<f32>;
@group(0) @binding(2) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(3) var<storage, read> light_field: array<f32>;
@group(0) @binding(4) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(5) var<storage, read> water_velocity: array<u32>;
@group(0) @binding(6) var<storage, read_write> wetness: array<f32>;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn is_solid(x: i32, y: i32, z: i32) -> bool {
    let res = i32(params.grid_resolution);
    if (x < 0 || x >= res || y < 0 || y >= res || z < 0 || z >= res) {
        return true;
    }
    return solid_mask[voxel_index(u32(x), u32(y), u32(z))] == 1u;
}

fn is_water_at(x: i32, y: i32, z: i32) -> bool {
    let res = i32(params.grid_resolution);
    if (x < 0 || x >= res || y < 0 || y >= res || z < 0 || z >= res) {
        return false;
    }
    let state = fluid_state[voxel_index(u32(x), u32(y), u32(z))];
    return (state & 0xFFFFu) == 1u;
}

fn is_steam_at(x: i32, y: i32, z: i32) -> bool {
    let res = i32(params.grid_resolution);
    if (x < 0 || x >= res || y < 0 || y >= res || z < 0 || z >= res) {
        return false;
    }
    let state = fluid_state[voxel_index(u32(x), u32(y), u32(z))];
    return (state & 0xFFFFu) == 3u;
}

fn is_moisture_source(x: i32, y: i32, z: i32) -> bool {
    return is_water_at(x, y, z) || is_steam_at(x, y, z);
}

fn has_solid_neighbor(x: i32, y: i32, z: i32) -> bool {
    return is_solid(x - 1, y, z) || is_solid(x + 1, y, z)
        || is_solid(x, y - 1, z) || is_solid(x, y + 1, z)
        || is_solid(x, y, z - 1) || is_solid(x, y, z + 1);
}

// ============================================================
// Water proximity: checks a wide area around the voxel
// Returns a value in [0, 1] where 1 = right next to water, 0 = far away
// Samples along 26 directions (6 faces + 12 edges + 8 corners)
// at 5 steps scaled by params.water_radius
// ============================================================
fn check_water_in_direction(x: i32, y: i32, z: i32, dx: i32, dy: i32, dz: i32) -> f32 {
    let r = params.water_radius;
    // 5 steps at fractions of the radius: ~7%, 20%, 40%, 67%, 100%
    let d1 = max(i32(r * 0.07), 1);
    let d2 = max(i32(r * 0.2), 2);
    let d3 = max(i32(r * 0.4), 3);
    let d4 = max(i32(r * 0.67), 4);
    let d5 = i32(r);

    if (is_water_at(x + dx * d1, y + dy * d1, z + dz * d1)) { return 1.0; }
    if (is_water_at(x + dx * d2, y + dy * d2, z + dz * d2)) { return 0.8; }
    if (is_water_at(x + dx * d3, y + dy * d3, z + dz * d3)) { return 0.6; }
    if (is_water_at(x + dx * d4, y + dy * d4, z + dz * d4)) { return 0.4; }
    if (is_water_at(x + dx * d5, y + dy * d5, z + dz * d5)) { return 0.2; }
    return 0.0;
}

fn water_proximity(x: i32, y: i32, z: i32) -> f32 {
    var best = 0.0;

    // 6 face directions
    best = max(best, check_water_in_direction(x, y, z, 1, 0, 0));
    best = max(best, check_water_in_direction(x, y, z, -1, 0, 0));
    best = max(best, check_water_in_direction(x, y, z, 0, 1, 0));
    best = max(best, check_water_in_direction(x, y, z, 0, -1, 0));
    best = max(best, check_water_in_direction(x, y, z, 0, 0, 1));
    best = max(best, check_water_in_direction(x, y, z, 0, 0, -1));
    if (best >= 0.8) { return best; } // Early out if close water found

    // 12 edge directions
    best = max(best, check_water_in_direction(x, y, z, 1, 1, 0));
    best = max(best, check_water_in_direction(x, y, z, 1, -1, 0));
    best = max(best, check_water_in_direction(x, y, z, -1, 1, 0));
    best = max(best, check_water_in_direction(x, y, z, -1, -1, 0));
    best = max(best, check_water_in_direction(x, y, z, 1, 0, 1));
    best = max(best, check_water_in_direction(x, y, z, 1, 0, -1));
    best = max(best, check_water_in_direction(x, y, z, -1, 0, 1));
    best = max(best, check_water_in_direction(x, y, z, -1, 0, -1));
    best = max(best, check_water_in_direction(x, y, z, 0, 1, 1));
    best = max(best, check_water_in_direction(x, y, z, 0, 1, -1));
    best = max(best, check_water_in_direction(x, y, z, 0, -1, 1));
    best = max(best, check_water_in_direction(x, y, z, 0, -1, -1));
    if (best >= 0.6) { return best; }

    // 8 corner directions
    best = max(best, check_water_in_direction(x, y, z, 1, 1, 1));
    best = max(best, check_water_in_direction(x, y, z, 1, 1, -1));
    best = max(best, check_water_in_direction(x, y, z, 1, -1, 1));
    best = max(best, check_water_in_direction(x, y, z, 1, -1, -1));
    best = max(best, check_water_in_direction(x, y, z, -1, 1, 1));
    best = max(best, check_water_in_direction(x, y, z, -1, 1, -1));
    best = max(best, check_water_in_direction(x, y, z, -1, -1, 1));
    best = max(best, check_water_in_direction(x, y, z, -1, -1, -1));

    return best;
}

fn has_water_flow(idx: u32) -> bool {
    return water_velocity[idx] != 0u;
}

// ============================================================
// 3D hash for coverage noise (deterministic, no time dependency)
// ============================================================
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let c000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));

    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

// Multi-octave coverage noise: returns [0, 1]
// Large-scale patches with fine detail at edges
fn coverage_noise(world_pos: vec3<f32>) -> f32 {
    let scale = 0.08; // ~12 voxels per noise period at 128 grid
    let p = world_pos * scale;
    var n = noise3d(p) * 0.6;
    n += noise3d(p * 2.3 + 7.1) * 0.25;
    n += noise3d(p * 5.7 + 13.3) * 0.15;
    return n;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = params.grid_resolution;
    if (global_id.x >= res || global_id.y >= res || global_id.z >= res) {
        return;
    }

    let idx = voxel_index(global_id.x, global_id.y, global_id.z);
    let ix = i32(global_id.x);
    let iy = i32(global_id.y);
    let iz = i32(global_id.z);

    let dt = params.delta_time;

    // ============================================================
    // Wetness update: water or steam contact leaves a moisture memory
    // ============================================================
    var current_wetness = wetness[idx];
    let has_water_here = is_water_at(ix, iy, iz);
    let has_steam_here = is_steam_at(ix, iy, iz);
    let has_flow = has_water_flow(idx);

    if (has_water_here || has_flow) {
        // Direct water contact: fully wet
        current_wetness = 1.0;
    } else if (has_steam_here) {
        // Steam condensation: adds moisture but not as strongly as liquid water
        current_wetness = max(current_wetness, 0.5);
        current_wetness = min(current_wetness + 0.3 * dt, 1.0);
    } else {
        // Check if any immediate neighbor has water or steam (splashing/condensation)
        let neighbor_wet = is_moisture_source(ix - 1, iy, iz) || is_moisture_source(ix + 1, iy, iz)
                        || is_moisture_source(ix, iy - 1, iz) || is_moisture_source(ix, iy + 1, iz)
                        || is_moisture_source(ix, iy, iz - 1) || is_moisture_source(ix, iy, iz + 1);
        if (neighbor_wet) {
            current_wetness = max(current_wetness, 0.7);
        } else {
            // Evaporate over time
            current_wetness = current_wetness - params.wetness_evaporation * dt;
        }
    }
    current_wetness = clamp(current_wetness, 0.0, 1.0);
    wetness[idx] = current_wetness;

    // ============================================================
    // Moss growth logic
    // ============================================================
    var current_moss = moss_density[idx];

    // Solid voxels cannot have moss
    if (is_solid(ix, iy, iz)) {
        moss_density[idx] = 0.0;
        return;
    }

    // Must be a cave surface voxel
    let on_surface = has_solid_neighbor(ix, iy, iz);

    // Submerged check (only liquid water, not steam)
    let submerged = has_water_here;

    // Light intensity
    let light = light_field[idx];

    // Moisture source: combine real-time water proximity with wetness memory
    // Either one is sufficient for moss to grow
    let w_prox = water_proximity(ix, iy, iz);

    // Effective moisture: max of real-time proximity and wetness memory
    // Wetness acts as a nucleation point — once wet, moss can start even after water leaves
    let moisture = max(w_prox, current_wetness);

    // Water flow erosion check
    let flowing_water = has_flow;

    // Coverage noise
    let world_pos = vec3<f32>(
        f32(global_id.x) * params.cell_size + params.grid_origin_x,
        f32(global_id.y) * params.cell_size + params.grid_origin_y,
        f32(global_id.z) * params.cell_size + params.grid_origin_z,
    );
    let coverage = coverage_noise(world_pos);

    // Coverage threshold: moisture lowers the threshold
    let coverage_threshold = 1.1 - moisture * 0.9;

    // Growth conditions
    let can_grow = on_surface && !submerged && (light > params.min_light)
                && (moisture > 0.0) && (coverage > coverage_threshold);

    if (can_grow) {
        let light_factor = smoothstep(params.min_light, params.min_light + 0.2, light);
        let growth_target = light_factor * moisture;
        current_moss = current_moss + params.growth_rate * light_factor * moisture * dt;
        current_moss = min(current_moss, growth_target);
    } else {
        current_moss = current_moss - params.decay_rate * dt;
    }

    // Water flow erosion
    if (flowing_water && current_moss > 0.0) {
        current_moss = current_moss - params.erosion_rate * dt;
    }

    // Submerged moss erodes faster
    if (submerged && current_moss > 0.0) {
        current_moss = current_moss - params.erosion_rate * 2.0 * dt;
    }

    moss_density[idx] = clamp(current_moss, 0.0, 1.0);
}
