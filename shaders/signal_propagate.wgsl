// Signal Propagate Compute Shader
// Pull-based signal propagation through adhesion connections.
// Each cell reads its adhesion neighbors' signal values and updates itself.
// Dispatch once per hop iteration (max_hops times total).
//
// 16 channels per cell: signal_flags[cell_idx * 16 + channel]
// Each channel is independently propagated.

struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _align_pad: vec2<u32>,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
}

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_ATTENUATION_PER_HOP: f32 = 0.8; // 80% signal strength retained per hop

@group(0) @binding(0)
var<storage, read_write> signal_flags: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(1)
var<storage, read> cell_adhesion_indices: array<i32>;

@group(1) @binding(2)
var<storage, read> mode_indices: array<u32>;

@group(1) @binding(3)
var<storage, read> mode_cell_types: array<u32>;

const OCULOCYTE_TYPE: u32 = 7u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }

    // Signal senders (oculocytes) are immune to receiving signals on oculocyte channels (0-7).
    // However, they CAN receive regulation signals (8-15) from other cells.
    let mode_idx = mode_indices[idx];
    let cell_type = mode_cell_types[mode_idx];
    let is_oculocyte = (cell_type == OCULOCYTE_TYPE);

    let my_base = idx * SIGNAL_CHANNELS;

    // Propagate each channel independently
    for (var ch = 0u; ch < SIGNAL_CHANNELS; ch++) {
        // Oculocytes are immune to receiving on channels 0-7 (their own sensing channels)
        if (is_oculocyte && ch < 8u) { continue; }

        // Only propagate to channels that have not yet received a signal this frame.
        let current_signal = atomicLoad(&signal_flags[my_base + ch]);
        if (current_signal != 0u) { continue; }

        // Find the best neighbor for this channel: highest remaining hops
        var best_hops = 0u;
        var best_value = 0u;
        let adhesion_base = idx * MAX_ADHESIONS_PER_CELL;

        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            let adh_idx = cell_adhesion_indices[adhesion_base + i];
            if (adh_idx < 0) { continue; }

            let conn = adhesion_connections[u32(adh_idx)];
            if (conn.is_active == 0u) { continue; }

            // Find neighbor cell index
            var neighbor: u32;
            if (conn.cell_a_index == idx) {
                neighbor = conn.cell_b_index;
            } else {
                neighbor = conn.cell_a_index;
            }

            if (neighbor >= cell_count) { continue; }

            let neighbor_signal = atomicLoad(&signal_flags[neighbor * SIGNAL_CHANNELS + ch]);

            // Decode hop count and value from signal
            // Format: DDDDDHHHHHVVVVVVVVVVVV (5 bits direction flag, 5 bits hops, 11 bits value)
            let neighbor_hops = (neighbor_signal >> 11u) & 31u;
            let neighbor_value = neighbor_signal & 2047u;

            // Keep hops and value from the same neighbor so they stay semantically paired.
            if (neighbor_hops > 0u && neighbor_value > 0u) {
                if (neighbor_hops > best_hops || (neighbor_hops == best_hops && neighbor_value > best_value)) {
                    best_hops = neighbor_hops;
                    best_value = neighbor_value;
                }
            }
        }

        // If a propagatable neighbor was found, write signal to this cell (once only).
        if (best_value > 0u && best_hops > 0u) {
            let attenuated_value = best_value * u32(SIGNAL_ATTENUATION_PER_HOP * 1000.0) / 1000u;
            let attenuated = max(1u, attenuated_value);

            let new_hops = best_hops - 1u;
            let encoded_signal = (new_hops << 11u) | attenuated;

            atomicStore(&signal_flags[my_base + ch], encoded_signal);
        }
    }
}
