// Signal Propagate Compute Shader
// Pull-based signal propagation through adhesion connections.
// Each cell reads its adhesion neighbors' signal values from signal_flags (read-only this
// dispatch) and writes its own propagated value to signal_flags_next (write-only).
// After each dispatch the caller copies signal_flags_next -> signal_flags so the next
// hop iteration sees the freshly propagated values.
//
// This double-buffer design eliminates the read-write hazard that existed when both
// reads and writes targeted the same buffer: without it, whether a cell received a
// signal depended on GPU thread scheduling order, and relay cells (those that already
// had a signal) could never forward it to their own neighbors.
//
// 16 channels per cell: signal_flags[cell_idx * 16 + channel]
// Each channel is independently propagated.
// Signal word format: bit16 = source flag, bits 11-15 = hops remaining, bits 0-10 = value
//
// Summation semantics: contributions from multiple neighbors are SUMMED (clamped to 2047),
// and hops is the MAX across all contributing neighbors. This enables quorum-sensing patterns
// where a cell crossing a threshold only when many neighbours are emitting.

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
const SIGNAL_ATTENUATION_PER_HOP: f32 = 0.5; // 50% signal strength retained per hop (matches CPU)

// Group 0: signal buffers + cell count
// binding 0: signal_flags      - read-only source for this hop (previous state)
// binding 1: cell_count_buffer - live cell count
// binding 2: signal_flags_next - write-only destination for this hop
@group(0) @binding(0)
var<storage, read> signal_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> cell_count_buffer: array<u32>;

@group(0) @binding(2)
var<storage, read_write> signal_flags_next: array<u32>;

// Group 1: adhesion topology + cell type data
@group(1) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(1)
var<storage, read> cell_adhesion_indices: array<i32>;

@group(1) @binding(2)
var<storage, read> mode_indices: array<u32>;

@group(1) @binding(3)
var<storage, read> mode_cell_types: array<u32>;

const OCULOCYTE_TYPE: u32 = 7u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }

    let mode_idx = mode_indices[idx];
    let cell_type = mode_cell_types[mode_idx];
    // Oculocytes are hard stops: they never receive signals on any channel.
    // This matches the CPU BFS in signal_system.rs where is_signal_sender() returns
    // true for oculocytes and they are skipped entirely (not written, not enqueued).
    let is_oculocyte = (cell_type == OCULOCYTE_TYPE);

    let my_base = idx * SIGNAL_CHANNELS;

    // Propagate each channel independently.
    for (var ch = 0u; ch < SIGNAL_CHANNELS; ch++) {
        // Oculocytes are hard stops on all channels - they never relay signals.
        if (is_oculocyte) {
            // Copy the source signal unchanged so the emitter's own value persists
            // into signal_flags_next (the caller will copy next -> current after this pass).
            signal_flags_next[my_base + ch] = signal_flags[my_base + ch];
            continue;
        }

        // Start with whatever this cell already has (from the sense pass or a prior hop).
        let own_signal = signal_flags[my_base + ch];
        let own_hops  = (own_signal >> 11u) & 31u;
        let own_value = own_signal & 2047u;

        // Accumulate contributions from all neighbors:
        //   summed_value  – sum of each neighbor's attenuated value (clamped to 2047)
        //   best_hops     – max hops across contributing neighbors (determines reach)
        //   any_source    – true if any contributing neighbor is a direct emitter (bit 16)
        //
        // The cell's own current signal is included as the starting sum so that a cell
        // which is itself an emitter (or already accumulated signal) carries it forward.
        var summed_value: u32 = own_value;
        var best_hops:    u32 = own_hops;
        var any_source:   u32 = own_signal & (1u << 16u);
        let adhesion_base = idx * MAX_ADHESIONS_PER_CELL;

        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            let adh_idx = cell_adhesion_indices[adhesion_base + i];
            if (adh_idx < 0) { continue; }

            let conn = adhesion_connections[u32(adh_idx)];
            if (conn.is_active == 0u) { continue; }

            // Resolve neighbor cell index.
            var neighbor: u32;
            if (conn.cell_a_index == idx) {
                neighbor = conn.cell_b_index;
            } else {
                neighbor = conn.cell_a_index;
            }

            if (neighbor >= cell_count) { continue; }

            let neighbor_signal = signal_flags[neighbor * SIGNAL_CHANNELS + ch];
            // Decode signal word: bit16 = source flag, bits 11-15 = hops, bits 0-10 = value
            let neighbor_hops        = (neighbor_signal >> 11u) & 31u;
            let neighbor_value       = neighbor_signal & 2047u;
            let neighbor_source_flag = neighbor_signal & (1u << 16u);

            // Only accept contributions from neighbors with signal remaining to relay.
            if (neighbor_hops > 0u && neighbor_value > 0u) {
                // Attenuate: direct sources pass full strength; relays lose 50% per hop.
                let contrib = select(
                    max(1u, neighbor_value * u32(SIGNAL_ATTENUATION_PER_HOP * 1000.0) / 1000u),
                    neighbor_value,
                    neighbor_source_flag != 0u
                );
                // Sum contributions; clamp to 11-bit max.
                summed_value = min(summed_value + contrib, 2047u);
                // Track max hops so the furthest-reaching signal governs propagation distance.
                if (neighbor_hops > best_hops) {
                    best_hops = neighbor_hops;
                }
                // If any contributor is a direct source, mark result as source-adjacent.
                any_source = any_source | neighbor_source_flag;
            }
        }

        // Determine what to write to signal_flags_next for this cell/channel.
        if (summed_value > 0u && best_hops > 0u) {
            // Decrement hop count; clear source flag so downstream cells attenuate normally.
            let new_hops = best_hops - 1u;
            let encoded  = (new_hops << 11u) | summed_value;
            signal_flags_next[my_base + ch] = encoded;
        } else {
            // No propagatable signal - preserve whatever this cell already had
            // (e.g. its own emission from the sense pass, or zero after clear).
            signal_flags_next[my_base + ch] = own_signal;
        }
    }
}
