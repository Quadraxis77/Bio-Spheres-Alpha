// Moss Growth / Erosion / Decay Compute Shader
//
// Runs once per frame over the 128^3 voxel grid.
// Growth conditions:
//   - Solid-adjacent (cave surface)
//   - Lit above threshold
//   - Not submerged
//   - Moist (wetness buffer > 0, propagated from water each frame)
//   - Passes a per-voxel hash coverage test (patchy distribution)
//
// Water flow and submersion erode existing moss.

struct MossParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    growth_rate: f32,
    erosion_rate: f32,
    decay_rate: f32,
    min_light: f32,
    delta_time: f32,
    world_radius: f32,
    water_radius: f32,       // wetness diffusion spread strength (0-1 mapped from 0-25)
    wetness_evaporation: f32,
    slice_index: u32,        // unused, kept for struct compat
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

fn fluid_type_at(x: i32, y: i32, z: i32) -> u32 {
    let res = i32(params.grid_resolution);
    if (x < 0 || x >= res || y < 0 || y >= res || z < 0 || z >= res) {
        return 0u;
    }
    return fluid_state[voxel_index(u32(x), u32(y), u32(z))] & 0xFFFFu;
}

fn is_water_at(x: i32, y: i32, z: i32) -> bool {
    return fluid_type_at(x, y, z) == 1u;
}

fn has_solid_neighbor(x: i32, y: i32, z: i32) -> bool {
    return is_solid(x - 1, y, z) || is_solid(x + 1, y, z)
        || is_solid(x, y - 1, z) || is_solid(x, y + 1, z)
        || is_solid(x, y, z - 1) || is_solid(x, y, z + 1);
}

fn has_water_flow(idx: u32) -> bool {
    return water_velocity[idx] != 0u;
}

// Cheap per-voxel coverage hash - replaces 3-octave noise (24 hash calls -> 1).
// Returns a stable value in [0,1] for each voxel position.
// The low-frequency variation comes from the voxel coordinates themselves;
// the result is visually similar to noise at the scale of the moss grid.
fn voxel_coverage(x: u32, y: u32, z: u32) -> f32 {
    // Mix the three coordinates into a single hash
    var h = x * 1664525u + y * 1013904223u + z * 22695477u;
    h = h ^ (h >> 16u);
    h = h * 2246822519u;
    h = h ^ (h >> 13u);
    h = h * 3266489917u;
    h = h ^ (h >> 16u);
    return f32(h) / 4294967295.0; // normalize to [0,1]
}

// Wetness diffusion: read 6 face-neighbors and propagate the maximum inward.
fn diffuse_wetness(x: i32, y: i32, z: i32) -> f32 {
    let res = i32(params.grid_resolution);
    var best = 0.0;
    if (x > 0)       { best = max(best, wetness[voxel_index(u32(x - 1), u32(y), u32(z))]); }
    if (x < res - 1) { best = max(best, wetness[voxel_index(u32(x + 1), u32(y), u32(z))]); }
    if (y > 0)       { best = max(best, wetness[voxel_index(u32(x), u32(y - 1), u32(z))]); }
    if (y < res - 1) { best = max(best, wetness[voxel_index(u32(x), u32(y + 1), u32(z))]); }
    if (z > 0)       { best = max(best, wetness[voxel_index(u32(x), u32(y), u32(z - 1))]); }
    if (z < res - 1) { best = max(best, wetness[voxel_index(u32(x), u32(y), u32(z + 1))]); }
    let spread = clamp(params.water_radius / 25.0, 0.5, 0.99);
    return best * spread;
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

    // -- Wetness update ------------------------------------------
    var current_wetness = wetness[idx];
    let has_water_here = is_water_at(ix, iy, iz);
    let has_flow = has_water_flow(idx);
    let fluid_type = fluid_type_at(ix, iy, iz);
    let has_steam_here = (fluid_type == 3u);

    if (has_water_here || has_flow) {
        current_wetness = 1.0;
    } else if (has_steam_here) {
        current_wetness = max(current_wetness, 0.5);
        current_wetness = min(current_wetness + 0.3 * dt, 1.0);
    } else {
        let neighbor_wetness = diffuse_wetness(ix, iy, iz);
        current_wetness = max(current_wetness, neighbor_wetness);
        current_wetness = current_wetness - params.wetness_evaporation * dt;
    }
    current_wetness = clamp(current_wetness, 0.0, 1.0);
    wetness[idx] = current_wetness;

    // -- Moss growth logic ----------------------------------------
    var current_moss = moss_density[idx];

    if (is_solid(ix, iy, iz)) {
        moss_density[idx] = 0.0;
        return;
    }

    let on_surface  = has_solid_neighbor(ix, iy, iz);
    let submerged   = has_water_here;
    let light       = light_field[idx];
    let moisture    = current_wetness;
    let flowing_water = has_flow;

    // Per-voxel coverage: cheap hash replaces 3-octave noise
    let coverage = voxel_coverage(global_id.x, global_id.y, global_id.z);
    let coverage_threshold = 1.1 - moisture * 0.9;

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

    if (flowing_water && current_moss > 0.0) {
        current_moss = current_moss - params.erosion_rate * dt;
    }
    if (submerged && current_moss > 0.0) {
        current_moss = current_moss - params.erosion_rate * 2.0 * dt;
    }

    moss_density[idx] = clamp(current_moss, 0.0, 1.0);
}
