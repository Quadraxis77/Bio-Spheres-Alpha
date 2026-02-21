// Signal Propagate Compute Shader
// Pull-based signal propagation through adhesion connections.
// Each cell reads its adhesion neighbors' signal values and updates itself.
// Dispatch once per hop iteration (max_hops times total).

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

@group(0) @binding(0)
var<storage, read_write> signal_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(1)
var<storage, read> cell_adhesion_indices: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }

    // Find maximum signal value among adhesion neighbors
    var max_neighbor_signal = 0u;
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

        let neighbor_signal = signal_flags[neighbor];
        max_neighbor_signal = max(max_neighbor_signal, neighbor_signal);
    }

    // If any neighbor has a propagatable signal (> 1), update this cell
    // Decrement by 1 for each hop
    if (max_neighbor_signal > 1u) {
        let propagated = max_neighbor_signal - 1u;
        // Only update if propagated value is higher than current
        // (preserves oculocyte source values which may be higher)
        let current = signal_flags[idx];
        if (propagated > current) {
            signal_flags[idx] = propagated;
        }
    }
}
