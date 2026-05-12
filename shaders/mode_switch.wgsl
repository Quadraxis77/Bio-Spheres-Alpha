// Mode Switch Compute Shader
// Signal-conditional mode switching: changes a cell's mode_index without division.
// Runs once per frame after the signal system has written signal_flags.
//
// Condition: if mode has mode_switch_signal_channel >= 8, read that channel's signal.
//   - If signal >= threshold (or inverted: signal < threshold), switch to mode_switch_target.
//   - mode_switch_target is an absolute GPU mode index (already remapped by sync_signal_settings).
//
// signal_settings_v3: [child_b_mode_above, child_b_mode_below, mode_switch_channel, mode_switch_threshold]
// signal_settings_v4: [mode_switch_target, mode_switch_invert, 0, 0]

const SIGNAL_CHANNELS: u32 = 16u;

@group(0) @binding(0)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(0) @binding(1)
var<storage, read> death_flags: array<u32>;

@group(0) @binding(2)
var<storage, read_write> mode_indices: array<u32>;

@group(0) @binding(3)
var<storage, read> mode_cell_types: array<u32>;

// Per-cell signal flags (packed u32: bits 0-10 = value, bits 11-15 = hops, bit 16 = source flag)
@group(1) @binding(0)
var<storage, read> signal_flags: array<u32>;

// signal_settings_v3: [child_b_mode_above, child_b_mode_below, mode_switch_channel(f32), mode_switch_threshold]
@group(1) @binding(1)
var<storage, read> signal_settings_v3: array<vec4<f32>>;

// signal_settings_v4: [mode_switch_target(f32), mode_switch_invert(f32), 0, 0]
@group(1) @binding(2)
var<storage, read> signal_settings_v4: array<vec4<f32>>;

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
    if (switch_channel < 8.0) { return; } // Disabled or oculocyte channel — skip

    let switch_threshold = ss_v3.w;
    let switch_target    = ss_v4.x; // absolute mode index (f32)
    let switch_invert    = ss_v4.y; // 0 = normal, 1 = inverted

    // Validate target mode
    if (switch_target < 0.0) { return; }
    let target_mode_idx = u32(switch_target);
    if (target_mode_idx >= arrayLength(&mode_cell_types)) { return; }

    // Read signal value from packed u32 (lower 11 bits)
    let ch = clamp(u32(switch_channel), 8u, 15u);
    let raw_signal = signal_flags[cell_idx * SIGNAL_CHANNELS + ch];
    let signal_value = f32(raw_signal & 0x7FFu);

    // Compare against threshold (matches CPU: unwrap_or(0.0) >= threshold)
    let above = signal_value >= switch_threshold;
    let should_switch = select(above, !above, switch_invert > 0.5);

    if (should_switch) {
        mode_indices[cell_idx] = target_mode_idx;
    }
}
