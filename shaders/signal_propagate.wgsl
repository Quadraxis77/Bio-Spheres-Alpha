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
// Signal word format: bit24 = source flag, bits 11-23 = scaled travel budget, bits 0-10 = value
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
    bond_flags: u32,
    _align_pad1: u32,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
}

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_VALUE_MASK: u32 = 2047u;
const SIGNAL_HOP_SHIFT: u32 = 11u;
const SIGNAL_HOP_MASK: u32 = 8191u;
const SIGNAL_SOURCE_FLAG: u32 = 1u << 24u;
const SIGNAL_NORMAL_COST: u32 = 4u;
const SIGNAL_ROAD_COST: u32 = 1u;
const SIGNAL_LOSS_PER_POINT: f32 = 0.05;

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
const VASCULOCYTE_TYPE: u32 = 12u;
const BOND_FLAG_BARRIER_BALL: u32 = 2u;

// [nutrient_transport, nutrient_exchange, signal_transport, signal_exchange]
@group(1) @binding(4)
var<storage, read> mode_properties_v12: array<vec4<f32>>;

fn is_signal_transport_vascular(mode_idx: u32, cell_type: u32) -> bool {
    if (cell_type != VASCULOCYTE_TYPE || mode_idx >= arrayLength(&mode_properties_v12)) {
        return false;
    }
    return mode_properties_v12[mode_idx].z > 0.5;
}

fn is_signal_exchange_vascular(mode_idx: u32, cell_type: u32) -> bool {
    if (cell_type != VASCULOCYTE_TYPE || mode_idx >= arrayLength(&mode_properties_v12)) {
        return false;
    }
    return mode_properties_v12[mode_idx].w > 0.5;
}

fn can_signal_cross(
    from_mode: u32,
    from_type: u32,
    to_mode: u32,
    to_type: u32,
) -> bool {
    let from_vascular = from_type == VASCULOCYTE_TYPE;
    let to_vascular = to_type == VASCULOCYTE_TYPE;
    if (from_vascular && to_vascular) {
        return is_signal_transport_vascular(from_mode, from_type) &&
               is_signal_transport_vascular(to_mode, to_type);
    }
    if (from_vascular && !to_vascular) {
        return is_signal_exchange_vascular(from_mode, from_type);
    }
    if (!from_vascular && to_vascular) {
        return is_signal_exchange_vascular(to_mode, to_type);
    }
    return true;
}

fn signal_edge_cost(
    from_mode: u32,
    from_type: u32,
    to_mode: u32,
    to_type: u32,
) -> u32 {
    if (from_type == VASCULOCYTE_TYPE && to_type == VASCULOCYTE_TYPE &&
        is_signal_transport_vascular(from_mode, from_type) &&
        is_signal_transport_vascular(to_mode, to_type)) {
        return SIGNAL_ROAD_COST;
    }
    return SIGNAL_NORMAL_COST;
}

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
        let own_hops  = (own_signal >> SIGNAL_HOP_SHIFT) & SIGNAL_HOP_MASK;
        let own_value = own_signal & SIGNAL_VALUE_MASK;

        // Accumulate contributions from all neighbors:
        //   summed_value  – sum of each neighbor's attenuated value (clamped to 2047)
        //   best_hops     – max hops across contributing neighbors (determines reach)
        //   any_source    - true if any contributing neighbor is a direct emitter (bit 24)
        //
        // The cell's own current signal is included as the starting sum so that a cell
        // which is itself an emitter (or already accumulated signal) carries it forward.
        var summed_value: u32 = own_value;
        var best_hops:    u32 = own_hops;
        var any_source:   u32 = own_signal & SIGNAL_SOURCE_FLAG;
        let adhesion_base = idx * MAX_ADHESIONS_PER_CELL;

        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            let adh_idx = cell_adhesion_indices[adhesion_base + i];
            if (adh_idx < 0) { continue; }

            let conn = adhesion_connections[u32(adh_idx)];
            if (conn.is_active == 0u) { continue; }
            if ((conn.bond_flags & BOND_FLAG_BARRIER_BALL) != 0u) { continue; }

            // Resolve neighbor cell index.
            var neighbor: u32;
            if (conn.cell_a_index == idx) {
                neighbor = conn.cell_b_index;
            } else {
                neighbor = conn.cell_a_index;
            }

            if (neighbor >= cell_count) { continue; }

            let neighbor_signal = signal_flags[neighbor * SIGNAL_CHANNELS + ch];
            let neighbor_mode = mode_indices[neighbor];
            if (neighbor_mode >= arrayLength(&mode_cell_types)) { continue; }
            let neighbor_type = mode_cell_types[neighbor_mode];
            if (!can_signal_cross(neighbor_mode, neighbor_type, mode_idx, cell_type)) { continue; }

            let edge_cost = signal_edge_cost(neighbor_mode, neighbor_type, mode_idx, cell_type);

            // Decode signal word.
            let neighbor_hops        = (neighbor_signal >> SIGNAL_HOP_SHIFT) & SIGNAL_HOP_MASK;
            let neighbor_value       = neighbor_signal & SIGNAL_VALUE_MASK;
            let neighbor_source_flag = neighbor_signal & SIGNAL_SOURCE_FLAG;

            // Only accept contributions from neighbors with signal remaining to relay.
            if (neighbor_hops >= edge_cost && neighbor_value > 0u) {
                let loss = SIGNAL_LOSS_PER_POINT * (f32(edge_cost) / f32(SIGNAL_NORMAL_COST));
                let retain = clamp(1.0 - loss, 0.0, 1.0);
                let contrib = max(1u, u32(f32(neighbor_value) * retain));
                // Sum contributions; clamp to 11-bit max.
                summed_value = min(summed_value + contrib, SIGNAL_VALUE_MASK);
                // Track max remaining budget so the furthest-reaching signal governs propagation distance.
                let remaining_hops = neighbor_hops - edge_cost;
                if (remaining_hops > best_hops) {
                    best_hops = remaining_hops;
                }
                // If any contributor is a direct source, mark result as source-adjacent.
                any_source = any_source | neighbor_source_flag;
            }
        }

        // Determine what to write to signal_flags_next for this cell/channel.
        if (summed_value > 0u && best_hops > 0u) {
            let encoded  = (best_hops << SIGNAL_HOP_SHIFT) | summed_value;
            signal_flags_next[my_base + ch] = encoded;
        } else {
            // No propagatable signal - preserve whatever this cell already had
            // (e.g. its own emission from the sense pass, or zero after clear).
            signal_flags_next[my_base + ch] = own_signal;
        }
    }
}
