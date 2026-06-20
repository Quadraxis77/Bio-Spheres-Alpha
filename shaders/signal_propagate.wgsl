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
// and hops is the MAX across all contributing neighbors. Same-mode regulation emitters also
// form a deterministic accumulation chain. The host runs this shader in both index
// directions and invokes `combine_sweeps` so emitters receive contributions from both sides.
// Re-basing each emitter on its authored value prevents feedback growth.

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
const PROPAGATION_DIRECTION: u32 = 0u;

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

@group(0) @binding(3)
var<storage, read> signal_flags_forward: array<u32>;

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

// [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding(u32)]
@group(1) @binding(5)
var<storage, read> regulation_params: array<vec4<u32>>;

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

    // The reverse sweep exists only for equal-budget regulation-source chains,
    // which occupy channels 8-15. Sensory channels have strictly decreasing
    // budgets and are already complete after the forward sweep.
    let first_channel = select(0u, 8u, PROPAGATION_DIRECTION == 1u);

    // Propagate each relevant channel independently.
    for (var ch = first_channel; ch < SIGNAL_CHANNELS; ch++) {
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
        let own_source_flag = own_signal & SIGNAL_SOURCE_FLAG;
        let own_regulation = regulation_params[mode_idx];
        let is_regulation_source =
            own_source_flag != 0u &&
            own_regulation.x == ch &&
            ch >= 8u &&
            ch <= 15u;

        // Recompute from direct emission and strictly-upstream neighbors.
        //   summed_value  – sum of each neighbor's attenuated value (clamped to 2047)
        //   best_hops     – max hops across contributing neighbors (determines reach)
        //
        // Propagated values are not reused as a base; doing so creates feedback growth
        // on every dispatch and quickly saturates bonded loops at 2047.
        // Regulation emitters are re-based on their authored value. Their packed
        // signal may contain an upstream sum from the previous pass, which must not
        // be treated as a fresh local emission or it would grow on every iteration.
        var direct_value = own_value;
        if (is_regulation_source) {
            direct_value = min(
                u32(max(bitcast<f32>(own_regulation.y), 0.0)),
                SIGNAL_VALUE_MASK,
            );
        }
        var summed_value: u32 = select(0u, direct_value, own_source_flag != 0u);
        var best_hops:    u32 = own_hops;
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
            let neighbor_regulation = regulation_params[neighbor_mode];

            // Equal-budget sources need a deterministic acyclic sweep. The host runs
            // both directions and combines them afterward.
            let neighbor_is_upstream = select(neighbor < idx, neighbor > idx, PROPAGATION_DIRECTION == 1u);
            let same_mode_source_upstream =
                is_regulation_source &&
                neighbor_source_flag != 0u &&
                neighbor_mode == mode_idx &&
                neighbor_regulation.x == ch &&
                neighbor_is_upstream;

            // Only accept contributions from neighbors with signal remaining to relay.
            if (neighbor_hops >= edge_cost &&
                (neighbor_hops > own_hops || same_mode_source_upstream) &&
                neighbor_value > 0u) {
                let loss = SIGNAL_LOSS_PER_POINT * (f32(edge_cost) / f32(SIGNAL_NORMAL_COST));
                let retain = clamp(1.0 - loss, 0.0, 1.0);
                var contrib = max(1u, u32(f32(neighbor_value) * retain));

                // The first hop from a source is lossless. Regulation emitters may
                // also contain accumulated incoming traffic, so restore attenuation
                // only for their authored local component; the incoming portion
                // continues to degrade normally. Oculocytes never accept incoming
                // signals, so their whole value is the direct component.
                if (neighbor_source_flag != 0u) {
                    var direct_value = neighbor_value;
                    if (neighbor_regulation.x == ch && ch >= 8u && ch <= 15u) {
                        direct_value = min(
                            u32(max(bitcast<f32>(neighbor_regulation.y), 0.0)),
                            SIGNAL_VALUE_MASK,
                        );
                    }
                    let direct_component = min(direct_value, neighbor_value);
                    let restored = u32(f32(direct_component) * (1.0 - retain));
                    contrib = min(contrib + restored, SIGNAL_VALUE_MASK);
                }
                // Sum contributions; clamp to 11-bit max.
                summed_value = min(summed_value + contrib, SIGNAL_VALUE_MASK);
                // Track max remaining budget so the furthest-reaching signal governs propagation distance.
                let remaining_hops = neighbor_hops - edge_cost;
                if (remaining_hops > best_hops) {
                    best_hops = remaining_hops;
                }
            }
        }

        // Determine what to write to signal_flags_next for this cell/channel.
        if (summed_value > 0u) {
            let encoded  = own_source_flag | (best_hops << SIGNAL_HOP_SHIFT) | summed_value;
            signal_flags_next[my_base + ch] = encoded;
        } else {
            // No propagatable signal - preserve whatever this cell already had
            // (e.g. its own emission from the sense pass, or zero after clear).
            signal_flags_next[my_base + ch] = own_signal;
        }
    }
}

@compute @workgroup_size(256)
fn combine_sweeps(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }

    let mode_idx = mode_indices[idx];
    let regulation = regulation_params[mode_idx];
    let base = idx * SIGNAL_CHANNELS;

    for (var ch = 0u; ch < SIGNAL_CHANNELS; ch++) {
        let forward = signal_flags_forward[base + ch];
        let reverse = signal_flags[base + ch];
        let forward_value = forward & SIGNAL_VALUE_MASK;
        let reverse_value = reverse & SIGNAL_VALUE_MASK;
        let source_flag = (forward | reverse) & SIGNAL_SOURCE_FLAG;
        let forward_hops = (forward >> SIGNAL_HOP_SHIFT) & SIGNAL_HOP_MASK;
        let reverse_hops = (reverse >> SIGNAL_HOP_SHIFT) & SIGNAL_HOP_MASK;
        let best_hops = max(forward_hops, reverse_hops);

        let is_regulation_source =
            source_flag != 0u &&
            regulation.x == ch &&
            ch >= 8u &&
            ch <= 15u;

        var combined_value = max(forward_value, reverse_value);
        if (is_regulation_source) {
            let direct_value = min(
                u32(max(bitcast<f32>(regulation.y), 0.0)),
                SIGNAL_VALUE_MASK,
            );
            combined_value = min(
                forward_value + reverse_value - min(direct_value, min(forward_value, reverse_value)),
                SIGNAL_VALUE_MASK,
            );
        }

        signal_flags_next[base + ch] =
            source_flag | (best_hops << SIGNAL_HOP_SHIFT) | combined_value;
    }
}
