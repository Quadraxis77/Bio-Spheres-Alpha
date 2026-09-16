// Photocyte Light Consumption Shader
// Photocytes gain mass based on light intensity at their position.
// Reads from the pre-computed light field buffer.
//
// Luminocytes write their glow state to glow_flags. The render-frame emission
// pass scatters it into this same field before the next physics step.

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
    // Mass gained per second at full sunlight intensity
    mass_per_second_full_light: f32,
    // Mass gained per second from geothermal light. This intentionally does
    // not receive the sun brightness multiplier.
    geothermal_mass_per_second_full_light: f32,
    // Minimum light intensity to gain any mass (threshold)
    min_light_threshold: f32,
    ambient_floor: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
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
// Buffer is DMA-cleared before this dispatch so dead/off luminocytes read as zero.
@group(1) @binding(11)
var<storage, read_write> glow_flags: array<vec4<f32>>;

// xyz = blended radiative color, w = total local-source intensity. Luminocyte
// radiance is resolved here before the next physics step, after cave occlusion.
@group(1) @binding(12)
var<storage, read> light_color_field: array<vec4<f32>>;

struct Emission {
    r: atomic<u32>,
    g: atomic<u32>,
    b: atomic<u32>,
    strength: atomic<u32>,
}
@group(1) @binding(13)
var<storage, read_write> luminocyte_emission: array<Emission>;

// Photocyte cell type constant
const PHOTOCYTE_TYPE: u32 = 3u;
const LUMINOCYTE_TYPE: u32 = 16u;
const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_VALUE_MASK: u32 = 2047u;

fn decode_signal(raw: u32) -> f32 {
    return f32(bitcast<i32>((raw & SIGNAL_VALUE_MASK) << 21u) >> 21u);
}
fn listener_active(value: f32, threshold: f32, response_mode: u32, invert: bool) -> bool {
    var response = max(value, 0.0);
    if (response_mode == 1u) { response = max(-value, 0.0); }
    if (response_mode == 2u) { response = abs(value); }
    let normal = response > 0.0 && response >= max(threshold, 0.0);
    return select(normal, !normal, invert);
}
// Luminocyte energy cost per second at full brightness.
const LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND: f32 = 6.0;
// A photocyte converts at most 5 nutrients/sec from one unit of luminocyte
// radiance. A luminocyte pays 6 nutrients/sec for that unit before falloff and
// occlusion, so a single receiving photocyte cannot recover the emitter's cost.
const LUMINOCYTE_PHOTOCYTE_NUTRIENTS_PER_LIGHT_SECOND: f32 = 5.0;
// Photocytes below the direct-light threshold should starve at a stable rate.
// Do not scale this by sampled sunlight; true darkness would otherwise drain
// nothing and blocked cells could survive indefinitely.
const PHOTOCYTE_SHADE_LOSS_RATE: f32 = 10.0;
// Full sunlight is 1.0. Geothermal vents are allowed to be worth 1.5x the
// default photocyte replacement rate: 75 nutrients/sec / 20 nutrients/sec.
const GEOTHERMAL_PHOTOCYTE_LIGHT_VALUE: f32 = 3.75;

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;
const LUMINOCYTE_FIELD_FIXED_POINT_SCALE: f32 = 1024.0;

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

fn light_index(world_pos: vec3<f32>) -> u32 {
    let res = photocyte_params.grid_resolution;
    let p = vec3<i32>(floor((world_pos - vec3<f32>(
        photocyte_params.grid_origin_x,
        photocyte_params.grid_origin_y,
        photocyte_params.grid_origin_z,
    )) / photocyte_params.cell_size));
    let ires = i32(res);
    if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(ires))) {
        return 0xffffffffu;
    }
    return u32(p.x) + u32(p.y) * res + u32(p.z) * res * res;
}

// Returns (total radiance, local-source radiance) from the shared field.
fn sample_light(world_pos: vec3<f32>) -> vec2<f32> {
    let idx = light_index(world_pos);
    if (idx == 0xffffffffu) {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(light_field[idx], max(light_color_field[idx].w, 0.0));
}

fn sample_luminocyte_intensity(world_pos: vec3<f32>) -> f32 {
    let idx = light_index(world_pos);
    if (idx == 0xffffffffu) { return 0.0; }
    return f32(atomicLoad(&luminocyte_emission[idx].strength))
        / LUMINOCYTE_FIELD_FIXED_POINT_SCALE;
}

// The occupancy field includes this cell. Sampling only its center can
// mistake its own opaque voxel footprint for shade. Probe the sun-facing
// surface beyond that footprint, while retaining local geothermal exposure.
fn sample_photocyte_light(pos: vec3<f32>, mass: f32) -> vec2<f32> {
    let toward_sun = vec3<f32>(photocyte_params.light_dir_x, photocyte_params.light_dir_y, photocyte_params.light_dir_z);
    let radius = clamp(mass, 0.5, 2.0);
    let surface = pos + toward_sun * (radius + photocyte_params.cell_size);
    let center = sample_light(pos);
    let sun_surface = sample_light(surface);
    // Local radiance is sampled at the cell itself. Sunlight probes the exposed
    // surface so the cell's own occupancy voxel cannot shadow it.
    let center_sun = max(center.x - center.y, 0.0);
    let surface_sun = max(sun_surface.x - sun_surface.y, 0.0);
    return vec2<f32>(max(center_sun, surface_sun), center.y);
}

fn signal_value(cell_idx: u32, channel: u32) -> f32 {
    let packed = atomicLoad(&signal_flags[cell_idx * SIGNAL_CHANNELS + min(channel, SIGNAL_CHANNELS - 1u)]);
    return decode_signal(packed);
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
    if (death_flags[cell_idx] != 0u) {
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
        var response_mode = 0u;
        if (mode_idx < arrayLength(&mode_properties_v7)) {
            let control = mode_properties_v7[mode_idx];
            invert = control.x >= 0.5;
            response_mode = min(u32(max(control.y, 0.0)), 2u);
            signal_channel = u32(clamp(control.z, 0.0, 15.0));
            threshold = control.w;
        }
        if (mode_idx < arrayLength(&mode_emissive)) {
            let raw = mode_emissive[mode_idx].x;
            bright_level = select(0.5, raw, raw > 0.001);
            dim_level = bright_level * 0.15;
        }

        let incoming = signal_value(cell_idx, signal_channel);
        let listener_on = listener_active(incoming, threshold, response_mode, invert);
        let brightness = select(dim_level, bright_level, listener_on);
        if (brightness <= 0.001) {
            // glow_flags[cell_idx] is already 0 (DMA-cleared before this dispatch)
            return;
        }

        let current_nutrients_fixed = atomicLoad(&nutrients_buffer[cell_idx]);
        let current_nutrients = fixed_to_float(current_nutrients_fixed);
        let nutrient_factor = smoothstep(1.0, 10.0, current_nutrients);
        let effective_brightness = brightness * nutrient_factor;
        if (effective_brightness <= 0.001) {
            return;
        }

        var emit_color = vec3<f32>(1.0, 1.0, 1.0);
        if (mode_idx < arrayLength(&mode_colors)) {
            emit_color = clamp(mode_colors[mode_idx].xyz, vec3<f32>(0.0), vec3<f32>(4.0));
        }

        // Pay before emitting. Near starvation, the actual brightness is reduced
        // to exactly what the cell could afford, so unpaid light cannot enter
        // the radiative field.
        let requested_cost = effective_brightness * LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND * params.delta_time;
        // Nutrients use thousandths. Round the payment up, then derive emitted
        // brightness from what was actually paid. This prevents sub-unit light
        // from escaping for free at very small time steps or brightness values.
        let requested_cost_fixed = i32(ceil(requested_cost * FIXED_POINT_SCALE));
        let available_cost_fixed = max(current_nutrients_fixed - i32(FIXED_POINT_SCALE), 0);
        let paid_cost_fixed = min(requested_cost_fixed, available_cost_fixed);
        if (paid_cost_fixed > 0) {
            atomicAdd(&nutrients_buffer[cell_idx], -paid_cost_fixed);
            let paid_cost = fixed_to_float(paid_cost_fixed);
            let paid_brightness = min(
                effective_brightness,
                paid_cost / max(LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND * params.delta_time, 0.000001),
            );
            // Plain store — voxel scatter happens once after the physics steps.
            glow_flags[cell_idx] = vec4<f32>(emit_color, paid_brightness);
        }
        return;
    }

    // Photocyte: consume the same occluded radiative field used by rendering.
    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));

    if (current_nutrients < 1.0) {
        return;
    }

    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;
    let radiance = sample_photocyte_light(pos, positions[cell_idx].w);

    let ambient_floor = clamp(photocyte_params.ambient_floor, 0.0, 0.95);
    let direct_sun = clamp(
        (radiance.x - ambient_floor) / max(1.0 - ambient_floor, 0.001),
        0.0,
        1.0
    );
    let luminocyte_light = sample_luminocyte_intensity(pos);
    let geothermal_light = clamp(
        max(radiance.y - luminocyte_light, 0.0),
        0.0,
        GEOTHERMAL_PHOTOCYTE_LIGHT_VALUE,
    );
    let usable_light = direct_sun + geothermal_light + luminocyte_light;

    // Sun, geothermal glow, and luminocyte radiance add. Luminocyte food uses
    // its own conversion coefficient so it stays below the emitter's cost.
    let nutrient_rate = photocyte_params.mass_per_second_full_light * 100.0 * direct_sun
        + photocyte_params.geothermal_mass_per_second_full_light * 100.0 * geothermal_light
        + LUMINOCYTE_PHOTOCYTE_NUTRIENTS_PER_LIGHT_SECOND * luminocyte_light;
    let capacity = max(max_nutrients - current_nutrients, 0.0);
    let nutrient_gain = min(nutrient_rate * params.delta_time, capacity);

    if (nutrient_gain > 0.0) {
        atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(nutrient_gain));
    } else if (usable_light < photocyte_params.min_light_threshold) {
        let nutrient_loss = min(PHOTOCYTE_SHADE_LOSS_RATE * params.delta_time, max(current_nutrients - 1.0, 0.0));
        if (nutrient_loss > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(nutrient_loss));
        }
    }
}
