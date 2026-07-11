// Light Field Compute Shader
// Computes per-voxel light intensity by ray marching from each voxel toward the light source.
// Occlusion comes from:
//   1. Cave solid mask (static geometry)
//   2. Cell occupancy grid (dynamic - cells block light)
// Output: light_field buffer with f32 intensity per voxel (0.0 = fully shadowed, 1.0 = fully lit)

struct LightFieldParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    world_radius: f32,
    // Light source (directional light defined by direction toward light)
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    // Number of ray march steps (higher = more accurate shadows, more expensive)
    max_steps: u32,
    // Step size multiplier (1.0 = one voxel per step)
    step_size: f32,
    // Absorption per solid voxel hit (controls shadow softness)
    absorption_solid: f32,
    // Absorption per cell-occupied voxel (cells partially block light)
    absorption_cell: f32,
    // Ambient light floor (minimum light even in full shadow)
    ambient_floor: f32,
    // Scattering coefficient for volumetric fog
    scattering_coefficient: f32,
    // Current time for animation
    time: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    water_light_attenuation: f32,
}

// Group 0: Parameters
@group(0) @binding(0)
var<uniform> params: LightFieldParams;

// Group 0: Solid mask from cave system (1 = solid, 0 = empty)
@group(0) @binding(1)
var<storage, read> solid_mask: array<u32>;

// Group 0: Cell occupancy grid (atomic u32 count of cells per voxel)
// Built by a separate pass that bins cells into voxels
@group(0) @binding(2)
var<storage, read> cell_occupancy: array<u32>;

// Group 0: Output light field (f32 intensity per voxel)
@group(0) @binding(3)
var<storage, read_write> light_field: array<f32>;

// Group 0: RGB color associated with the local light field.
// xyz = resolved light color, w = local emitter weight.
@group(0) @binding(4)
var<storage, read_write> light_color_field: array<vec4<f32>>;

// Group 0: Water density field. Values near 1.0 mean this voxel contains water.
@group(0) @binding(5)
var<storage, read> water_density: array<f32>;

// Group 0: Atmospheric humidity field from the fluid simulator, fixed-point
// (value = humidity_0_255 * HUMIDITY_FIXED_POINT). Higher humidity scatters
// more light, gently dimming the light field (fog).
@group(0) @binding(6)
var<storage, read> humidity: array<u32>;

const HUMIDITY_FIXED_POINT: f32 = 256.0;
// Per-voxel humidity contribution to light attenuation along the ray.
const HUMIDITY_LIGHT_ATTENUATION: f32 = 0.002;
// Never let humidity alone reduce transmittance below this fraction -
// fog dims light but shouldn't black it out on its own.
const HUMIDITY_ATTENUATION_FLOOR: f32 = 0.4;

// Group 0: Ice density field (f32 per voxel, 1.0 = ice). Ice is translucent
// but scatters more than liquid water, so it attenuates the ray more
// strongly per voxel - water under an ice sheet sits in cold shade.
@group(0) @binding(7)
var<storage, read> ice_density: array<f32>;

// Prebaked local glow from geothermal crevice low points. xyz is RGB radiance,
// w is source strength. Medium tinting is applied per voxel below.
@group(0) @binding(8)
var<storage, read> geothermal_glow: array<vec4<f32>>;

// Full sunlight is 1.0. Photocytes gain 20 nutrients/sec at that level.
// A default photocyte needs 50 nutrients/sec to replace itself
// (split_mass 1.5 over split_interval 1.0), so vents are capped at 1.5x
// replacement: 75 nutrients/sec, or 3.75 light units.
const GEOTHERMAL_PHOTOCYTE_LIGHT_VALUE: f32 = 3.75;

// Per-voxel absorption coefficient for ice (liquid water uses 0.055).
const ICE_ABSORPTION: f32 = 0.12;

// Convert 3D grid coordinates to linear index
fn grid_to_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

// Convert linear index to 3D grid coordinates
fn index_to_grid(idx: u32) -> vec3<u32> {
    let res = params.grid_resolution;
    let z = idx / (res * res);
    let y = (idx - z * res * res) / res;
    let x = idx - z * res * res - y * res;
    return vec3<u32>(x, y, z);
}

// Convert grid coordinates to world position (voxel center)
fn grid_to_world(gx: u32, gy: u32, gz: u32) -> vec3<f32> {
    return vec3<f32>(
        params.grid_origin_x + (f32(gx) + 0.5) * params.cell_size,
        params.grid_origin_y + (f32(gy) + 0.5) * params.cell_size,
        params.grid_origin_z + (f32(gz) + 0.5) * params.cell_size,
    );
}

// Check if grid coordinates are within bounds
fn is_in_bounds(x: i32, y: i32, z: i32) -> bool {
    let res = i32(params.grid_resolution);
    return x >= 0 && x < res && y >= 0 && y < res && z >= 0 && z < res;
}

// Sample solid mask at grid position (returns true if solid or out of bounds)
fn is_solid(x: i32, y: i32, z: i32) -> bool {
    if (!is_in_bounds(x, y, z)) {
        return true; // Out of bounds = solid (world boundary)
    }
    let idx = grid_to_index(u32(x), u32(y), u32(z));
    return solid_mask[idx] != 0u;
}

// Check if a grid-space position is inside the world sphere
fn is_inside_sphere(gx: f32, gy: f32, gz: f32) -> bool {
    let fres = f32(params.grid_resolution);
    let half = fres * 0.5;
    let dx = gx - half;
    let dy = gy - half;
    let dz = gz - half;
    // Sphere radius in grid units = grid_resolution / 2
    return (dx * dx + dy * dy + dz * dz) <= (half * half);
}

fn geothermal_strength_at(x: i32, y: i32, z: i32) -> f32 {
    if (!is_in_bounds(x, y, z)) {
        return 0.0;
    }
    let idx = grid_to_index(u32(x), u32(y), u32(z));
    return geothermal_glow[idx].w;
}

fn geothermal_source_dir(gx: u32, gy: u32, gz: u32) -> vec3<f32> {
    let x = i32(gx);
    let y = i32(gy);
    let z = i32(gz);
    let grad = vec3<f32>(
        geothermal_strength_at(x + 1, y, z) - geothermal_strength_at(x - 1, y, z),
        geothermal_strength_at(x, y + 1, z) - geothermal_strength_at(x, y - 1, z),
        geothermal_strength_at(x, y, z + 1) - geothermal_strength_at(x, y, z - 1),
    );

    let len_sq = dot(grad, grad);
    if (len_sq <= 0.000001) {
        return vec3<f32>(0.0);
    }
    return grad * inverseSqrt(len_sq);
}

fn geothermal_transmittance(gx: u32, gy: u32, gz: u32, local_strength: f32) -> f32 {
    let source_dir = geothermal_source_dir(gx, gy, gz);
    if (dot(source_dir, source_dir) <= 0.000001) {
        return 1.0;
    }

    var pos = vec3<f32>(f32(gx) + 0.5, f32(gy) + 0.5, f32(gz) + 0.5);
    let step = source_dir * params.step_size;
    var transmittance = 1.0;
    var consecutive_solid = 0u;
    var strongest_seen = local_strength;

    for (var i = 0u; i < params.max_steps; i++) {
        pos += step;

        let ix = i32(floor(pos.x));
        let iy = i32(floor(pos.y));
        let iz = i32(floor(pos.z));

        if (!is_in_bounds(ix, iy, iz) || !is_inside_sphere(pos.x, pos.y, pos.z)) {
            break;
        }

        let sample_idx = grid_to_index(u32(ix), u32(iy), u32(iz));
        let sample_strength = geothermal_glow[sample_idx].w;
        if (sample_strength <= 0.001 && strongest_seen > local_strength * 1.1) {
            break;
        }
        strongest_seen = max(strongest_seen, sample_strength);

        let solid_here = solid_mask[sample_idx] != 0u && !is_near_sphere_boundary(pos.x, pos.y, pos.z);
        if (solid_here) {
            consecutive_solid += 1u;
            if (consecutive_solid >= 2u) {
                let x = params.absorption_solid;
                transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
            }
        } else {
            consecutive_solid = 0u;
        }

        let cells = cell_occupancy[sample_idx];
        if (cells > 0u) {
            let x = params.absorption_cell * f32(cells);
            transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
        }

        let water_amount = clamp(water_density[sample_idx], 0.0, 1.0);
        if (water_amount > 0.01) {
            let x = 0.055 * water_amount * params.step_size;
            transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
        }

        let ice_amount = clamp(ice_density[sample_idx], 0.0, 1.0);
        if (ice_amount > 0.01) {
            let x = ICE_ABSORPTION * ice_amount * params.step_size;
            transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
        }

        if (transmittance < 0.05) {
            return 0.0;
        }
    }

    return transmittance;
}

// Check if a grid-space position is near the sphere boundary (outer shell)
// Solids in this zone are the world boundary, not interior cave walls
fn is_near_sphere_boundary(gx: f32, gy: f32, gz: f32) -> bool {
    let fres = f32(params.grid_resolution);
    let half = fres * 0.5;
    let dx = gx - half;
    let dy = gy - half;
    let dz = gz - half;
    let dist_sq = dx * dx + dy * dy + dz * dz;
    // Within outer 8% of sphere radius = world boundary shell
    let inner_radius = half * 0.92;
    return dist_sq > (inner_radius * inner_radius);
}

// Ray march from voxel toward light source, accumulating occlusion.
// Returns vec2(transmittance, water_column) where water_column is the
// total integrated water density along the ray — used for chromatic tinting.
fn compute_light_at_voxel(gx: u32, gy: u32, gz: u32) -> vec2<f32> {
    // CPU-side LightFieldSystem::set_light_dir stores this normalized.
    let light_dir = vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z);

    // Start from voxel center, march in light direction
    var pos = vec3<f32>(f32(gx) + 0.5, f32(gy) + 0.5, f32(gz) + 0.5);
    let step = light_dir * params.step_size;

    var transmittance = 1.0;
    var water_column = 0.0; // integrated water density along the ray
    var humidity_column = 0.0; // integrated atmospheric humidity along the ray
    var consecutive_solid = 0u;

    for (var i = 0u; i < params.max_steps; i++) {
        pos += step;

        // Convert to integer grid coords
        let ix = i32(floor(pos.x));
        let iy = i32(floor(pos.y));
        let iz = i32(floor(pos.z));

        // If we've left the grid, we've reached open sky - stop
        if (!is_in_bounds(ix, iy, iz)) {
            break;
        }

        // If we've exited the world sphere, we've reached open sky - stop
        if (!is_inside_sphere(pos.x, pos.y, pos.z)) {
            break;
        }

        let sample_idx = grid_to_index(u32(ix), u32(iy), u32(iz));

        // Check solid (cave walls and the outer sphere shell). Boundary-shell
        // solids must still occlude sunlight; otherwise rays can leak through
        // the floor/edge of the world sphere and create a false lit rim when
        // the sun is below the sphere. Empty boundary openings remain passable.
        let solid_here = solid_mask[sample_idx] != 0u;
        if (solid_here) {
            let boundary_shell_solid = is_near_sphere_boundary(pos.x, pos.y, pos.z);
            if (boundary_shell_solid) {
                transmittance = 0.0;
                break;
            }

            consecutive_solid += 1u;
            // Only apply absorption after 2+ consecutive interior solid voxels (actual wall)
            if (consecutive_solid >= 2u) {
                // Fast exp approximation: exp(-x) ~= 1 / (1 + x + x^2/2)
                let x = params.absorption_solid;
                transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
            }
        } else {
            consecutive_solid = 0u;
        }

        // Check cell occupancy - cells block light
        let cells = cell_occupancy[sample_idx];
        if (cells > 0u) {
            // Fast exp approximation for multiple cells
            let x = params.absorption_cell * f32(cells);
            transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
        }

        let water_amount = clamp(water_density[sample_idx], 0.0, 1.0);
        if (water_amount > 0.01) {
            // Sunlight loses energy quickly with depth, but still fades
            // continuously so caustics weaken instead of ending abruptly.
            let x = 0.115 * params.water_light_attenuation * water_amount * params.step_size;
            transmittance *= 1.0 / (1.0 + x + x * x * 0.5);
            // Accumulate water column for downstream chromatic tinting
            water_column += water_amount * params.step_size;
        }

        // Ice: translucent but a stronger scatterer than liquid water - an
        // ice sheet dims the pool beneath it, which also slows melting from
        // solar heating (the thermal model reads this light field).
        let ice_amount = clamp(ice_density[sample_idx], 0.0, 1.0);
        if (ice_amount > 0.01) {
            let xi = ICE_ABSORPTION * ice_amount * params.step_size;
            transmittance *= 1.0 / (1.0 + xi + xi * xi * 0.5);
            // Ice tints downstream light like water does.
            water_column += ice_amount * params.step_size;
        }

        // Accumulate atmospheric humidity along the ray (fog/mist scattering)
        let humidity_amount = f32(humidity[sample_idx]) / HUMIDITY_FIXED_POINT;
        humidity_column += humidity_amount * params.step_size;

        // Early exit if light is effectively blocked
        if (transmittance < 0.05) {
            transmittance = 0.0;
            break;
        }
    }

    // Humidity gently dims light (fog), capped so it never fully blacks out a voxel on its own
    let humidity_atten = max(1.0 - humidity_column * HUMIDITY_LIGHT_ATTENUATION, HUMIDITY_ATTENUATION_FLOOR);
    transmittance *= humidity_atten;

    // Apply ambient floor - even fully shadowed areas get some light
    return vec2<f32>(max(transmittance, params.ambient_floor), water_column);
}

@compute @workgroup_size(64)
fn compute_light_field(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_voxels = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_voxels) {
        return;
    }
    
    let grid_pos = index_to_grid(idx);
    
    // Voxels outside the world sphere get full light (open sky)
    // solid_mask marks these as solid for fluid containment, but they aren't occluders
    let grid_pos_f = vec3<f32>(f32(grid_pos.x) + 0.5, f32(grid_pos.y) + 0.5, f32(grid_pos.z) + 0.5);
    if (!is_inside_sphere(grid_pos_f.x, grid_pos_f.y, grid_pos_f.z)) {
        light_field[idx] = 1.0;
        let geo = geothermal_glow[idx];
        light_color_field[idx] = vec4<f32>(vec3<f32>(params.sun_color_r, params.sun_color_g, params.sun_color_b) + geo.xyz, geo.w);
        return;
    }
    
    // Solid voxels inside the sphere: check if part of actual wall
    if (solid_mask[idx] != 0u) {
        // Quick check: only do expensive neighbor check near sphere boundary
        // Deep interior solids are cave walls and should be dark
        if (is_near_sphere_boundary(grid_pos_f.x, grid_pos_f.y, grid_pos_f.z)) {
            // Near boundary: check if isolated noise voxel
            let gx = i32(grid_pos.x);
            let gy = i32(grid_pos.y);
            let gz = i32(grid_pos.z);
            var solid_neighbors = 0u;
            if (is_solid(gx + 1, gy, gz)) { solid_neighbors += 1u; }
            if (is_solid(gx - 1, gy, gz)) { solid_neighbors += 1u; }
            if (is_solid(gx, gy + 1, gz)) { solid_neighbors += 1u; }
            if (is_solid(gx, gy - 1, gz)) { solid_neighbors += 1u; }
            if (is_solid(gx, gy, gz + 1)) { solid_neighbors += 1u; }
            if (is_solid(gx, gy, gz - 1)) { solid_neighbors += 1u; }
            if (solid_neighbors >= 3u) {
                light_field[idx] = 0.0;
                light_color_field[idx] = geothermal_glow[idx];
                return;
            }
        } else {
            // Deep interior solid = actual wall, skip expensive neighbor check
            light_field[idx] = 0.0;
            light_color_field[idx] = geothermal_glow[idx];
            return;
        }
    }
    
    // Ray march toward light to compute intensity and accumulated water depth
    let result = compute_light_at_voxel(grid_pos.x, grid_pos.y, grid_pos.z);
    let intensity = result.x;
    let water_column = result.y;
    light_field[idx] = intensity;

    let sun_color = vec3<f32>(params.sun_color_r, params.sun_color_g, params.sun_color_b);

    // Chromatic absorption: water filters out red then green first, then
    // attenuates blue at depth so deep water becomes genuinely darker.
    // Uses the same fast exp approximation as transmittance: 1/(1 + x + x²/2).
    let wc = water_column * params.water_light_attenuation;
    let r_abs = wc * 4.4;
    let g_abs = wc * 1.35;
    let b_abs = wc * 0.28;
    let r_atten = 1.0 / (1.0 + r_abs + r_abs * r_abs * 0.5);
    let g_atten = 1.0 / (1.0 + g_abs + g_abs * g_abs * 0.5);
    let b_atten = 1.0 / (1.0 + b_abs + b_abs * b_abs * 0.5);
    let ray_tint = vec3<f32>(r_atten, g_atten, b_atten);

    // Additional tint for voxels that are themselves submerged
    let voxel_water = clamp(water_density[idx], 0.0, 1.0);
    let voxel_tint = mix(vec3<f32>(1.0), vec3<f32>(0.56, 0.76, 0.90), voxel_water * 0.55);

    let geo = geothermal_glow[idx];
    let submerged = smoothstep(0.35, 0.85, voxel_water);
    let geo_tint = mix(vec3<f32>(1.0), vec3<f32>(0.92, 0.86, 0.78), submerged * 0.22);
    let ice_amount_here = clamp(ice_density[idx], 0.0, 1.0);
    let ice_tint = mix(vec3<f32>(1.0), vec3<f32>(0.90, 0.88, 0.86), ice_amount_here * 0.18);
    let geo_transmittance = geothermal_transmittance(grid_pos.x, grid_pos.y, grid_pos.z, geo.w);
    let local_glow = geo.xyz * geo_tint * ice_tint * geo_transmittance;
    let local_weight = clamp(geo.w * geo_transmittance, 0.0, GEOTHERMAL_PHOTOCYTE_LIGHT_VALUE);
    light_field[idx] = max(intensity, local_weight);
    let sunlight_color = sun_color * ray_tint * voxel_tint;
    let local_blend = local_weight / max(intensity + local_weight, 0.001);
    let resolved_color = mix(sunlight_color, local_glow, local_blend);
    light_color_field[idx] = vec4<f32>(resolved_color, local_weight);
}

// === Cell Occupancy Grid Builder ===
// Separate entry point to build cell_occupancy from cell positions
// This clears the grid first, then each cell atomically increments its voxel

struct OccupancyParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> occupancy_params: OccupancyParams;

@group(0) @binding(1)
var<storage, read> cell_positions: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> cell_count_buf: array<u32>;

@group(0) @binding(3)
var<storage, read_write> cell_occupancy_out: array<atomic<u32>>;

fn world_to_grid(world_pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor((world_pos.x - occupancy_params.grid_origin_x) / occupancy_params.cell_size)),
        i32(floor((world_pos.y - occupancy_params.grid_origin_y) / occupancy_params.cell_size)),
        i32(floor((world_pos.z - occupancy_params.grid_origin_z) / occupancy_params.cell_size)),
    );
}

@compute @workgroup_size(256)
fn build_cell_occupancy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buf[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    let pos = cell_positions[cell_idx].xyz;
    let grid_pos = world_to_grid(pos);
    let res = i32(occupancy_params.grid_resolution);
    
    // Bounds check
    if (grid_pos.x < 0 || grid_pos.x >= res ||
        grid_pos.y < 0 || grid_pos.y >= res ||
        grid_pos.z < 0 || grid_pos.z >= res) {
        return;
    }
    
    let voxel_idx = u32(grid_pos.x) + u32(grid_pos.y) * u32(res) + u32(grid_pos.z) * u32(res) * u32(res);
    atomicAdd(&cell_occupancy_out[voxel_idx], 1u);
}

@compute @workgroup_size(256)
fn clear_occupancy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = occupancy_params.grid_resolution * occupancy_params.grid_resolution * occupancy_params.grid_resolution;
    if (idx >= total) {
        return;
    }
    atomicStore(&cell_occupancy_out[idx], 0u);
}
