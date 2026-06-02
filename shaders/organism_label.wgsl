// Organism Label Shader
//
// Assigns each live cell a label equal to the minimum cell index in its connected
// component (determined by active adhesion bonds).
//
// ## Algorithm: continuous flood fill (one hop per frame)
//
// Each frame, every live cell looks at all its bonded neighbors and adopts the
// minimum label it sees (including its own). Over successive frames the minimum
// label floods through the entire connected component, regardless of organism size.
//
// No fixed iteration count, no convergence detection, no size limit.
//
// Correctness properties:
//   - A freshly-divided cell pair gets the correct shared label within a few frames
//     as the minimum propagates through the new bond.
//   - When a cell dies its neighbors stop propagating through it (dead cells hold
//     DEAD_LABEL). The remaining component re-floods to the new minimum within
//     a few frames.
//   - Labels are eventually consistent, not frame-perfect. The throttled period
//     (currently every 60 frames) means a full reset happens once per second;
//     between resets the continuous single-hop pass keeps labels fresh.
//
// ## Reset pass (init_labels)
//
// Periodically (controlled by run_init in LabelState) each cell resets its label
// to its own index. This clears stale labels from dead organisms and prevents
// label "fossils" from persisting indefinitely. After a reset, the flood fill
// re-converges within diameter(organism) frames.
//
// ## Size counting
//
// After the hook pass, clear/accumulate/broadcast passes compute per-cell organism
// size for the Kleiber metabolic discount. These run every frame alongside the hook.

// -- Constants ----------------------------------------------------------------

const MAX_ADHESIONS_PER_CELL: u32 = 20u;

const DEAD_LABEL: u32 = 0xFFFFFFFFu;

// Adhesion connection structure (104 bytes matching Rust GpuAdhesionConnection)
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

// -- Bind group 0 -------------------------------------------------------------

struct LabelState {
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    run_init: u32,  // 1 = reset labels to cell index this frame, 0 = skip reset
    run_hc:   u32,  // always 1 (hook pass runs every frame)
    _pad4:    u32,
    _pad5:    u32,
}

@group(0) @binding(0) var<storage, read_write> label_state:              LabelState;
@group(0) @binding(1) var<storage, read>       cell_count_buffer:        array<u32>;
@group(0) @binding(2) var<storage, read>       adhesion_counts:          array<u32>; // [0]=total [1]=live
@group(0) @binding(3) var<storage, read_write> label_buffer:             array<atomic<u32>>;
@group(0) @binding(4) var<storage, read>       death_flags:              array<u32>;
@group(0) @binding(5) var<storage, read>       cell_adhesion_indices:    array<i32>;
@group(0) @binding(6) var<storage, read>       adhesion_connections:     array<AdhesionConnection>;
// Per-organism cell count. Indexed by root cell index (= label value).
// Only the root slot is meaningful; all other slots are 0 after the count pass.
@group(0) @binding(7) var<storage, read_write> organism_size_buffer:     array<atomic<u32>>;

// -- label_controller ---------------------------------------------------------
// Sets run_hc = 1 unconditionally. run_init is set by the Rust side before
// each periodic reset; the controller preserves it so init_labels can read it.

@compute @workgroup_size(1, 1, 1)
fn label_controller() {
    label_state.run_hc = 1u;
    // run_init is written by the Rust side via queue.write_buffer before this pass.
    // We don't touch it here so init_labels sees the correct value.
}

// -- init_labels --------------------------------------------------------------
// Periodic reset: label[i] = i for live cells, DEAD_LABEL for dead.
// Only runs when run_init == 1 (set by Rust side on the reset frame).
// After a reset the flood fill re-converges within diameter(organism) frames.

@compute @workgroup_size(256, 1, 1)
fn init_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if label_state.run_init == 0u { return; }

    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }

    if death_flags[i] != 0u {
        atomicStore(&label_buffer[i], DEAD_LABEL);
    } else {
        atomicStore(&label_buffer[i], i);
    }
}

// -- hook_labels --------------------------------------------------------------
// Continuous flood fill: each cell adopts the minimum label among itself and
// all its bonded live neighbors. One dispatch per frame propagates the minimum
// label one hop further through each connected component.

@compute @workgroup_size(256, 1, 1)
fn hook_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }
    if death_flags[i] != 0u {
        // Dead cell: stamp DEAD_LABEL so neighbors stop propagating through it.
        atomicStore(&label_buffer[i], DEAD_LABEL);
        return;
    }

    var my_label = atomicLoad(&label_buffer[i]);
    if my_label == DEAD_LABEL {
        // Isolated live cell with stale DEAD_LABEL (e.g. just after a reset that
        // hasn't run init yet). Seed it with its own index so it can participate.
        my_label = i;
    }

    let base = i * MAX_ADHESIONS_PER_CELL;

    for (var s = 0u; s < MAX_ADHESIONS_PER_CELL; s++) {
        let slot = cell_adhesion_indices[base + s];
        if slot < 0 { continue; }

        let connection = adhesion_connections[u32(slot)];
        if connection.is_active == 0u { continue; }

        let cell_a = connection.cell_a_index;
        let cell_b = connection.cell_b_index;
        if (cell_a != i && cell_b != i) { continue; }
        let nb = select(cell_a, cell_b, cell_a == i);

        // Skip dead neighbors - don't propagate through them.
        if death_flags[nb] != 0u { continue; }

        let nb_label = atomicLoad(&label_buffer[nb]);
        if nb_label != DEAD_LABEL {
            my_label = min(my_label, nb_label);
        }
    }

    atomicMin(&label_buffer[i], my_label);
}

// -- clear_organism_sizes ------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn clear_organism_sizes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }
    atomicStore(&organism_size_buffer[i], 0u);
}

// -- count_organism_sizes ------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn count_organism_sizes_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }

    let lbl = atomicLoad(&label_buffer[i]);
    if lbl == DEAD_LABEL { return; }
    if lbl >= cell_count_buffer[0] { return; }

    atomicAdd(&organism_size_buffer[lbl], 1u);
}

@compute @workgroup_size(256, 1, 1)
fn count_organism_sizes_broadcast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }

    let lbl = atomicLoad(&label_buffer[i]);
    if lbl == DEAD_LABEL {
        atomicStore(&organism_size_buffer[i], 1u);
        return;
    }
    if lbl >= cell_count_buffer[0] { return; }

    let org_size = atomicLoad(&organism_size_buffer[lbl]);
    atomicStore(&organism_size_buffer[i], org_size);
}
