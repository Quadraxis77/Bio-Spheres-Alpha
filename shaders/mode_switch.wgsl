// Mode Switch Compute Shader
// Signal-conditional mode switching: changes a cell's mode_index without division.
// Runs once per frame after the signal system has written signal_flags.
//
// On switch, resets split_counts to 0 and copies the new mode's cached per-cell
// settings so switched cells immediately behave like their new mode.
//
// Also writes current_time into mode_switch_time[cell_idx] whenever a switch occurs.
// adhesion_physics.wgsl reads this to extend the bond-break grace period.

const SIGNAL_CHANNELS: u32 = 16u;

struct PhysicsParamsMin {
    delta_time:    f32,
    current_time:  f32,
    current_frame: i32,
    cell_count:    u32,
}

@group(0) @binding(0)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(0) @binding(1)
var<storage, read> death_flags: array<u32>;

@group(0) @binding(2)
var<storage, read_write> mode_indices: array<u32>;

@group(0) @binding(3)
var<storage, read> mode_cell_types: array<u32>;

@group(0) @binding(4)
var<storage, read_write> mode_switch_time: array<f32>;

@group(0) @binding(5)
var<uniform> params: PhysicsParamsMin;

@group(0) @binding(6)
var<storage, read_write> split_counts: array<u32>;

@group(0) @binding(7)
var<storage, read_write> max_splits: array<u32>;

@group(0) @binding(8)
var<storage, read_write> split_intervals: array<f32>;

@group(0) @binding(9)
var<storage, read_write> split_nutrient_thresholds: array<f32>;

@group(0) @binding(10)
var<storage, read_write> nutrient_gain_rates: array<f32>;

@group(0) @binding(11)
var<storage, read_write> max_cell_sizes: array<f32>;

@group(0) @binding(12)
var<storage, read_write> stiffnesses: array<f32>;

@group(0) @binding(13)
var<storage, read_write> cell_types: array<u32>;

// Per-cell signal flags (packed u32: bits 0-10 = value, bits 11-15 = hops, bit 16 = source flag)
@group(1) @binding(0)
var<storage, read> signal_flags: array<u32>;

// signal_settings_v3: [child_b_mode_above, child_b_mode_below, mode_switch_channel(f32), mode_switch_threshold]
@group(1) @binding(1)
var<storage, read> signal_settings_v3: array<vec4<f32>>;

// signal_settings_v4: [mode_switch_target(f32), mode_switch_invert(f32), 0, 0]
@group(1) @binding(2)
var<storage, read> signal_settings_v4: array<vec4<f32>>;

// Per-mode regulation emission params: [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding]
// Used to detect self-trigger: skip mode switch if this cell is itself emitting on the switch channel.
@group(1) @binding(3)
var<storage, read> regulation_params: array<vec4<u32>>;

// Per-mode properties (indexed by global mode index)
// v0: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
@group(2) @binding(0)
var<storage, read> mode_properties_v0: array<vec4<f32>>;

// v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
@group(2) @binding(1)
var<storage, read> mode_properties_v1: array<vec4<f32>>;

// v2: [max_splits (-1.0 = infinite), split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
@group(2) @binding(2)
var<storage, read> mode_properties_v2: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) { return; }

    // Skip dead cells
    if (death_flags[cell_idx] == 1u) { return; }

    let mode_idx = mode_indices[cell_idx];

    // Bounds check
    if (mode_idx >= arrayLength(&signal_settings_v3)) { return; }

    let ss_v3 = signal_settings_v3[mode_idx];
    let ss_v4 = signal_settings_v4[mode_idx];

    // ss_v3.z = mode_switch_signal_channel (f32, -1 = disabled, 8-15 = active)
    let switch_channel = ss_v3.z;
    if (switch_channel < 8.0) { return; } // Disabled or oculocyte channel - skip

    let switch_threshold = ss_v3.w;
    let switch_target    = ss_v4.x; // absolute mode index (f32)
    let switch_invert    = ss_v4.y; // 0 = normal, 1 = inverted

    // Validate target mode
    if (switch_target < 0.0) { return; }
    let target_mode_idx = u32(switch_target);
    if (target_mode_idx >= arrayLength(&mode_cell_types)) { return; }

    // Don't switch if already in the target mode
    if (mode_idx == target_mode_idx) { return; }

    // Read signal value from packed u32 (lower 11 bits)
    let ch = clamp(u32(switch_channel), 8u, 15u);
    let raw_signal = signal_flags[cell_idx * SIGNAL_CHANNELS + ch];
    let signal_value = f32(raw_signal & 0x7FFu);

    // Self-trigger guard: if this cell itself emits on the switch channel, its own emission
    // populates its signal slot every frame and would fire the switch unconditionally.
    // Skip if the only source of signal could be the cell's own regulation emit.
    // We detect this by checking: is the cell an emitter on this channel AND the signal has
    // the source flag set (bit 16), meaning it originated here rather than arriving from a neighbor.
    if (mode_idx < arrayLength(&regulation_params)) {
        let reg_ch = regulation_params[mode_idx].x;
        if (reg_ch == ch) {
            // Cell emits on the same channel it watches for mode switching.
            // Require the source flag (bit 16) to be CLEAR — meaning the signal arrived
            // from at least one propagation hop away (an external source).
            // If bit 16 is set, the signal is from this cell's own sense pass; ignore it.
            if ((raw_signal & (1u << 16u)) != 0u) { return; }
        }
    }

    // Compare against threshold
    let above = signal_value >= switch_threshold;
    let should_switch = select(above, !above, switch_invert > 0.5);

    if (!should_switch) { return; }

    // === Perform the switch ===
    mode_indices[cell_idx] = target_mode_idx;
    mode_switch_time[cell_idx] = params.current_time;

    // Reset split count so the new mode starts fresh.
    split_counts[cell_idx] = 0u;

    // Copy new mode's cached settings into per-cell buffers.
    // Without this the cell keeps old mode settings until it divides.
    if (target_mode_idx < arrayLength(&mode_properties_v0)) {
        let v0 = mode_properties_v0[target_mode_idx];
        let v1 = mode_properties_v1[target_mode_idx];
        let v2 = mode_properties_v2[target_mode_idx];
        let target_cell_type = mode_cell_types[target_mode_idx];

        let nutrient_rate = select(0.0, v0.x, target_cell_type == 0u);
        nutrient_gain_rates[cell_idx]        = nutrient_rate;
        max_cell_sizes[cell_idx]             = v0.y;
        stiffnesses[cell_idx]                = v0.z;
        cell_types[cell_idx]                 = target_cell_type;
        split_intervals[cell_idx]            = v0.w; // split_interval
        split_nutrient_thresholds[cell_idx]  = (v1.x - 1.0) * 100.0; // split_mass -> nutrient threshold

        // max_splits: -1.0 in genome = infinite = 0xFFFFFFFF in GPU buffer
        let raw_max = v2.x;
        max_splits[cell_idx] = select(u32(raw_max), 0xFFFFFFFFu, raw_max < 0.0);
    }
}
