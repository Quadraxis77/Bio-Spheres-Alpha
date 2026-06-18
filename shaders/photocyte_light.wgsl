// Photocyte Light Consumption Shader
// Photocytes gain mass based on light intensity at their position.
// Reads from the pre-computed light field buffer.
//
// Luminocytes write their glow state to glow_flags (plain store, no voxel scatter).
// Photocyte detection of luminocyte light is handled by sense_luminocyte.wgsl.

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
var<storage, read> light_field: array<f32>;  // Per-voxel light intensity (read-only)

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

// Luminocyte glow flags: vec4(color.rgb, brightness). Written by luminocytes each frame.
// Read by sense_luminocyte.wgsl to give nearby photocytes nutrient gain.
// Buffer is DMA-cleared before this dispatch so dead/off luminocytes read as zero.
@group(1) @binding(11)
var<storage, read_write> glow_flags: array<vec4<f32>>;

// Photocyte cell type constant
const PHOTOCYTE_TYPE: u32 = 3u;
const LUMINOCYTE_TYPE: u32 = 16u;
const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_VALUE_MASK: u32 = 2047u;
// Luminocyte energy cost per second at full brightness.
const LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND: f32 = 6.0;

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

// Sample light intensity from the voxel occupied by the cell.
fn sample_light(world_pos: vec3<f32>) -> f32 {
    let res = photocyte_params.grid_resolution;

    let gx = (world_pos.x - photocyte_params.grid_origin_x) / photocyte_params.cell_size;
    let gy = (world_pos.y - photocyte_params.grid_origin_y) / photocyte_params.cell_size;
    let gz = (world_pos.z - photocyte_params.grid_origin_z) / photocyte_params.cell_size;

    let ix = i32(floor(gx));
    let iy = i32(floor(gy));
    let iz = i32(floor(gz));

    let ires = i32(res);

    if (ix < 0 || ix >= ires || iy < 0 || iy >= ires || iz < 0 || iz >= ires) {
        return 0.0;
    }

    let idx = u32(ix) + u32(iy) * res + u32(iz) * res * res;
    return light_field[idx];
}

fn signal_value(cell_idx: u32, channel: u32) -> f32 {
    let packed = atomicLoad(&signal_flags[cell_idx * SIGNAL_CHANNELS + min(channel, SIGNAL_CHANNELS - 1u)]);
    return f32(packed & SIGNAL_VALUE_MASK);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    let cell_type = cell_types[cell_idx];
    if (cell_type != PHOTOCYTE_TYPE && cell_type != LUMINOCYTE_TYPE) {
        return;
    }

    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }

    let pos = positions[cell_idx].xyz;

    if (cell_type == LUMINOCYTE_TYPE) {
        // Compute effective brightness from signal/nutrients (same logic as before)
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
            bright_level = select(0.5, raw, raw > 0.001);
            dim_level = bright_level * 0.15;
        }

        let incoming = signal_value(cell_idx, signal_channel);
        let above = incoming >= threshold;
        let brightness = select(dim_level, bright_level, above != invert);
        if (brightness <= 0.001) {
            // glow_flags[cell_idx] is already 0 (DMA-cleared before this dispatch)
            return;
        }

        let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
        let nutrient_factor = smoothstep(1.0, 10.0, current_nutrients);
        let effective_brightness = brightness * nutrient_factor;
        if (effective_brightness <= 0.001) {
            return;
        }

        var emit_color = vec3<f32>(1.0, 1.0, 1.0);
        if (mode_idx < arrayLength(&mode_colors)) {
            emit_color = clamp(mode_colors[mode_idx].xyz, vec3<f32>(0.0), vec3<f32>(4.0));
        }

        // Plain store — no voxel scatter, no atomics. Buffer was DMA-cleared this frame.
        glow_flags[cell_idx] = vec4<f32>(emit_color, effective_brightness);

        // Deduct nutrients
        let requested_cost = effective_brightness * LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND * params.delta_time;
        let paid_cost = min(requested_cost, max(current_nutrients - 1.0, 0.0));
        if (paid_cost > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(paid_cost));
        }
        return;
    }

    // Photocyte: gain/lose nutrients based on ambient sunlight
    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));

    if (current_nutrients < 1.0) {
        return;
    }

    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;
    let light_intensity = sample_light(pos);
    let in_light = light_intensity >= photocyte_params.min_light_threshold;

    let nutrient_rate = photocyte_params.mass_per_second_full_light * 100.0
                        * clamp(light_intensity, 0.0, 1.0);

    if (in_light) {
        let nutrient_gain = min(nutrient_rate * params.delta_time, max(max_nutrients - current_nutrients, 0.0));
        if (nutrient_gain > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(nutrient_gain));
        }
    } else {
        let nutrient_loss = min(nutrient_rate * 0.5 * params.delta_time, max(current_nutrients - 1.0, 0.0));
        if (nutrient_loss > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(nutrient_loss));
        }
    }
}
