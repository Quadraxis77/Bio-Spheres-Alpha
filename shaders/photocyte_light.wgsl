// Photocyte Light Consumption Shader
// Photocytes gain mass based on light intensity at their position.
// Reads from the pre-computed light field buffer.
// Similar structure to phagocyte_consume.wgsl but uses light instead of nutrient voxels.

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
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct PhotocyteParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    // Mass gained per second at full light intensity
    mass_per_second_full_light: f32,
    // Minimum light intensity to gain any mass (threshold)
    min_light_threshold: f32,
    _pad0: f32,
}

// Physics bind group (group 0)
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec4<f32>>;  // xyz = position, w = mass

@group(0) @binding(2)
var<storage, read> cell_count_buffer: array<u32>;

// Photocyte system bind group (group 1)
@group(1) @binding(0)
var<uniform> photocyte_params: PhotocyteParams;

@group(1) @binding(1)
var<storage, read_write> light_field: array<f32>;  // Per-voxel light intensity (0.0+)

@group(1) @binding(2)
var<storage, read> cell_types: array<u32>;  // Per-cell type ID

// Nutrients buffer (read-write, fixed-point i32)
@group(1) @binding(3)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Split nutrient thresholds per cell (nutrient cap = 2x threshold)
@group(1) @binding(4)
var<storage, read> split_nutrient_thresholds: array<f32>;

// Death flags to skip dead cells
@group(1) @binding(5)
var<storage, read> death_flags: array<u32>;

@group(1) @binding(6)
var<storage, read> mode_indices: array<u32>;

@group(1) @binding(7)
var<storage, read> mode_properties_v7: array<vec4<f32>>;

@group(1) @binding(8)
var<storage, read> mode_emissive: array<vec4<f32>>;

@group(1) @binding(9)
var<storage, read_write> signal_flags: array<atomic<u32>>;

@group(1) @binding(10)
var<storage, read> mode_colors: array<vec4<f32>>;

@group(1) @binding(11)
var<storage, read_write> light_color_accum: array<atomic<u32>>;

@group(1) @binding(12)
var<storage, read_write> light_color_field: array<vec4<f32>>;

// Photocyte cell type constant
const PHOTOCYTE_TYPE: u32 = 3u;
const LUMINOCYTE_TYPE: u32 = 16u;
const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_VALUE_MASK: u32 = 2047u;
// Luminocyte energy cost per second at full brightness.
// Low enough for a decent glow to be affordable; the proportional photocyte
// gain below prevents luminocyte light from becoming free energy.
const LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND: f32 = 6.0;
const COLOR_ACCUM_SCALE: f32 = 4096.0;

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

// Convert world position to voxel grid index
fn world_to_voxel_index(world_pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<f32>(
        (world_pos.x - photocyte_params.grid_origin_x) / photocyte_params.cell_size,
        (world_pos.y - photocyte_params.grid_origin_y) / photocyte_params.cell_size,
        (world_pos.z - photocyte_params.grid_origin_z) / photocyte_params.cell_size,
    );
    
    let res = photocyte_params.grid_resolution;
    
    // Bounds check
    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return 0xFFFFFFFFu;
    }
    
    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);
    
    return gx + gy * res + gz * res * res;
}

// Sample light intensity at a world position with nearest neighbor (much faster)
fn sample_light(world_pos: vec3<f32>) -> f32 {
    let res = photocyte_params.grid_resolution;
    
    // Convert to grid-space coordinates
    let gx = (world_pos.x - photocyte_params.grid_origin_x) / photocyte_params.cell_size;
    let gy = (world_pos.y - photocyte_params.grid_origin_y) / photocyte_params.cell_size;
    let gz = (world_pos.z - photocyte_params.grid_origin_z) / photocyte_params.cell_size;
    
    // Round to nearest voxel
    let ix = i32(round(gx));
    let iy = i32(round(gy));
    let iz = i32(round(gz));
    
    let ires = i32(res);
    
    // Bounds check and sample single voxel
    if (ix < 0 || ix >= ires || iy < 0 || iy >= ires || iz < 0 || iz >= ires) {
        return 0.0; // Out of bounds = no light
    }
    
    let idx = u32(ix) + u32(iy) * res + u32(iz) * res * res;
    return light_field[idx];
}

fn signal_value(cell_idx: u32, channel: u32) -> f32 {
    let packed = atomicLoad(&signal_flags[cell_idx * SIGNAL_CHANNELS + min(channel, SIGNAL_CHANNELS - 1u)]);
    return f32(packed & SIGNAL_VALUE_MASK);
}

fn emit_luminocyte_light(cell_idx: u32, pos: vec3<f32>) {
    if (cell_idx >= arrayLength(&mode_indices)) {
        return;
    }

    let mode_idx = mode_indices[cell_idx];
    var signal_channel = 0u;
    var threshold = 1.0;
    var dim_level = 0.15;
    var bright_level = 1.0;
    var invert = false;
    if (mode_idx < arrayLength(&mode_properties_v7)) {
        let control = mode_properties_v7[mode_idx];
        invert = control.x >= 0.5;
        signal_channel = u32(clamp(control.z, 0.0, 15.0));
        threshold = control.w;
    }
    if (mode_idx < arrayLength(&mode_emissive)) {
        let raw = mode_emissive[mode_idx].x;
        bright_level = select(0.5, raw, raw > 0.001); // default brightness when unset
        dim_level = bright_level * 0.15;
    }

    let incoming = signal_value(cell_idx, signal_channel);
    let above = incoming >= threshold;
    let brightness = select(dim_level, bright_level, above != invert);
    if (brightness <= 0.001) {
        return;
    }

    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    // Smooth ramp: zero at 1 nutrient, full brightness above ~10. Eliminates hard on/off flicker.
    let nutrient_factor = smoothstep(1.0, 10.0, current_nutrients);
    let effective_brightness = brightness * nutrient_factor;
    if (effective_brightness <= 0.001) {
        return;
    }

    let requested_cost = effective_brightness * LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND * params.delta_time;
    let paid_cost = min(requested_cost, max(current_nutrients - 1.0, 0.0));
    if (paid_cost > 0.0) {
        atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(paid_cost));
    }
    var emit_color = vec3<f32>(1.0, 1.0, 1.0);
    if (mode_idx < arrayLength(&mode_colors)) {
        emit_color = clamp(mode_colors[mode_idx].xyz, vec3<f32>(0.0), vec3<f32>(4.0));
    }

    let res = i32(photocyte_params.grid_resolution);
    // Compute grid position directly — avoids round-trip through flat index.
    let gx = (pos.x - photocyte_params.grid_origin_x) / photocyte_params.cell_size;
    let gy = (pos.y - photocyte_params.grid_origin_y) / photocyte_params.cell_size;
    let gz = (pos.z - photocyte_params.grid_origin_z) / photocyte_params.cell_size;
    if (gx < 0.0 || gx >= f32(res) || gy < 0.0 || gy >= f32(res) || gz < 0.0 || gz >= f32(res)) {
        return;
    }
    let cx = i32(gx);
    let cy = i32(gy);
    let cz = i32(gz);
    // Sub-voxel offset of the emitter from the integer grid position.
    // Matches the original emitter_grid_pos convention (voxel centers at integer coords).
    let fx = f32(cx) - gx + 0.5;
    let fy = f32(cy) - gy + 0.5;
    let fz = f32(cz) - gz + 0.5;

    let emit_radius: i32 = 4;
    let emit_radius_f = f32(emit_radius);
    let emit_radius_sq = f32(emit_radius * emit_radius);
    for (var dz = -emit_radius; dz <= emit_radius; dz++) {
        let z = cz + dz;
        if (z < 0 || z >= res) {
            continue;
        }
        let vz = f32(dz) + fz;
        let vz_sq = vz * vz;
        if (vz_sq >= emit_radius_sq) { continue; }
        let ryz = emit_radius_sq - vz_sq;

        for (var dy = -emit_radius; dy <= emit_radius; dy++) {
            let y = cy + dy;
            if (y < 0 || y >= res) {
                continue;
            }
            let vy = f32(dy) + fy;
            let vy_sq = vy * vy;
            let rx = ryz - vy_sq;
            if (rx < 0.0) { continue; }
            // Tight dx range: only iterate voxels whose center lies within the sphere.
            let half_span = sqrt(rx);
            let dx_min = i32(ceil(-half_span - fx));
            let dx_max = i32(floor(half_span - fx));

            let yz_offset = u32(y) * photocyte_params.grid_resolution + u32(z) * photocyte_params.grid_resolution * photocyte_params.grid_resolution;
            for (var dx = max(dx_min, -emit_radius); dx <= min(dx_max, emit_radius); dx++) {
                let x = cx + dx;
                if (x < 0 || x >= res) {
                    continue;
                }
                let idx = u32(x) + yz_offset;
                let existing_light = light_field[idx];
                let darkness = max(0.0, 1.0 - existing_light);
                // light_field == 0.0 means solid rock. Reject it before the
                // distance/falloff math so luminocytes do less work in caves.
                if (existing_light < 0.001 || darkness <= 0.0) {
                    continue;
                }
                let vx = f32(dx) + fx;
                let dist = sqrt(vx * vx + vy_sq + vz_sq);
                // Center on the actual cell position instead of the containing voxel
                // so luminocyte light does not stamp visible grid blocks.
                let edge = 1.0 - smoothstep(0.0, 1.0, dist / emit_radius_f);
                let falloff = edge * edge * (1.0 + 0.45 * (1.0 - dist / emit_radius_f)) + 0.04;
                // Scale contribution by how dark the voxel already is so luminocyte
                // light fills shadow but gets washed out when the sun is bright.
                let local_light = effective_brightness * falloff * darkness;
                let weight = u32(max(local_light * COLOR_ACCUM_SCALE, 0.0));
                if (weight > 0u) {
                    let base = idx * 4u;
                    atomicAdd(&light_color_accum[base + 0u], u32(emit_color.r * f32(weight)));
                    atomicAdd(&light_color_accum[base + 1u], u32(emit_color.g * f32(weight)));
                    atomicAdd(&light_color_accum[base + 2u], u32(emit_color.b * f32(weight)));
                    atomicAdd(&light_color_accum[base + 3u], weight);
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn resolve_luminocyte_light_color(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = photocyte_params.grid_resolution * photocyte_params.grid_resolution * photocyte_params.grid_resolution;
    if (idx >= total) {
        return;
    }

    let base = idx * 4u;
    let weight = atomicLoad(&light_color_accum[base + 3u]);
    // light_color_field is DMA-cleared before this pass, so skip writing zeros.
    if (weight == 0u) {
        return;
    }

    light_field[idx] += f32(weight) / COLOR_ACCUM_SCALE;

    let inv_weight = 1.0 / f32(weight);
    let color = vec3<f32>(
        f32(atomicLoad(&light_color_accum[base + 0u])) * inv_weight,
        f32(atomicLoad(&light_color_accum[base + 1u])) * inv_weight,
        f32(atomicLoad(&light_color_accum[base + 2u])) * inv_weight,
    );
    light_color_field[idx] = vec4<f32>(color, f32(weight) / COLOR_ACCUM_SCALE);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Photocytes gain nutrients from light; luminocytes inject local light.
    let cell_type = cell_types[cell_idx];
    if (cell_type != PHOTOCYTE_TYPE && cell_type != LUMINOCYTE_TYPE) {
        return;
    }
    
    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }
    
    // Get cell position
    let pos = positions[cell_idx].xyz;

    if (cell_type == LUMINOCYTE_TYPE) {
        emit_luminocyte_light(cell_idx, pos);
        return;
    }

    // Read current nutrients from nutrients_buffer (fixed-point i32)
    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    
    // Skip cells with very low nutrients (would be marked dead)
    if (current_nutrients < 1.0) {
        return;
    }
    
    // Nutrient cap: 2x split_nutrient_threshold.
    // Cap at 200 before doubling so the "never split" sentinel (threshold > 100)
    // doesn't inflate the cap to an absurd value.
    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;
    // The actual nutrient gain rate comes from mass_per_second_full_light
    // (which is base_rate * sun_intensity, set on the CPU).
    let light_intensity = sample_light(pos);
    let in_light = light_intensity >= photocyte_params.min_light_threshold;

    // Convert mass_per_second to nutrients per second (multiply by 100).
    // Scale gain proportionally to light intensity so a faint luminocyte glow
    // cannot grant the same full-rate energy as direct sunlight.
    let nutrient_rate = photocyte_params.mass_per_second_full_light * 100.0
                        * clamp(light_intensity, 0.0, 1.0);

    if (in_light) {
        // In light: gain nutrients scaled by intensity.
        // Use atomicAdd of the delta - atomicStore would overwrite concurrent writes
        // from nutrient_transport running in the same frame.
        let nutrient_gain = min(nutrient_rate * params.delta_time, max(max_nutrients - current_nutrients, 0.0));
        if (nutrient_gain > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(nutrient_gain));
        }
    } else {
        // In shadow: lose nutrients (half the gain rate).
        // Subtract via atomicAdd of a negative delta; clamp so we don't go below 1.0.
        let nutrient_loss = min(nutrient_rate * 0.5 * params.delta_time, max(current_nutrients - 1.0, 0.0));
        if (nutrient_loss > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(nutrient_loss));
        }
    }
}
