// Organism Label Shader
//
// Assigns each live cell a label equal to the minimum cell index in its connected
// component (determined by active adhesion bonds). Runs as a self-throttled GPU
// state machine: the controller detects topology changes (cell count or live bond
// count changes), writes run_init / run_hc flags into label_state, and the
// subsequent dispatches early-exit when those flags are 0.
//
// NOTE: Indirect dispatch was intentionally avoided. The command processor reads
// indirect args before prior storage writes are guaranteed visible. Writing flags
// into label_state (storage) and reading them in subsequent shaders IS correctly
// ordered within a single compute pass.
//
// Algorithm: parallel pointer-jumping union-find
//   init:     label[i] = i  (dead cells get 0xFFFFFFFF sentinel)
//   hook:     for each cell, atomicMin with all active neighbours' labels
//   compress: label[i] = label[label[i]]  (one level of path compression)
//
// Unique within-organism ID (zero extra cost after convergence):
//   organism_root = label[cell_i]          // same for all cells in organism
//   local_id      = cell_i - organism_root // unique, non-negative, sparse

// ── Constants ────────────────────────────────────────────────────────────────

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

// ── Bind group 0 ─────────────────────────────────────────────────────────────

struct LabelState {
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    run_init: u32,  // always 1
    run_hc:   u32,  // always 1
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

// ── label_controller ─────────────────────────────────────────────────────────
// Runs every frame unconditionally. Labels are recomputed from scratch each
// frame so GPU-created bonds (from division) are always reflected immediately.
// The hook+compress passes are cheap enough to run every frame.

@compute @workgroup_size(1, 1, 1)
fn label_controller() {
    label_state.run_init = 1u;
    label_state.run_hc   = 1u;
}

// ── init_labels ──────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn init_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Storage read ordered after controller's write — safe within the same pass.
    if label_state.run_init == 0u { return; }

    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }

    if death_flags[i] != 0u {
        atomicStore(&label_buffer[i], DEAD_LABEL);
    } else {
        atomicStore(&label_buffer[i], i);
    }
}

// ── hook_labels ──────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn hook_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if label_state.run_hc == 0u { return; }

    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }
    if death_flags[i] != 0u { return; }

    var my_label = atomicLoad(&label_buffer[i]);
    if my_label == DEAD_LABEL { return; }

    let base = i * MAX_ADHESIONS_PER_CELL;

    for (var s = 0u; s < MAX_ADHESIONS_PER_CELL; s++) {
        let slot = cell_adhesion_indices[base + s];
        if slot < 0 { continue; }

        let connection = adhesion_connections[u32(slot)];
        if connection.is_active == 0u { continue; }

        let cell_a = connection.cell_a_index;
        let cell_b = connection.cell_b_index;
        // Validate this bond actually involves cell i — guards against stale slot indices
        // left over from a previous cell that occupied this slot.
        if (cell_a != i && cell_b != i) { continue; }
        // Select the neighbour: if cell_a == i, neighbour is cell_b, else cell_a.
        let nb = select(cell_a, cell_b, cell_a == i);

        let nb_label = atomicLoad(&label_buffer[nb]);
        if nb_label != DEAD_LABEL {
            my_label = min(my_label, nb_label);
        }
    }

    atomicMin(&label_buffer[i], my_label);
}

// ── compress_labels ───────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn compress_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if label_state.run_hc == 0u { return; }

    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }

    let lbl = atomicLoad(&label_buffer[i]);
    if lbl == DEAD_LABEL { return; }

    if lbl < cell_count_buffer[0] {
        let parent = atomicLoad(&label_buffer[lbl]);
        if parent != DEAD_LABEL {
            atomicStore(&label_buffer[i], parent);
        }
    }
}

// ── clear_organism_sizes ──────────────────────────────────────────────────────
// Zeroes the organism_size_buffer before the count pass.
// Must run every frame (not just on topology change) so stale counts from
// dead organisms are cleared even when labels are not recomputed.

@compute @workgroup_size(256, 1, 1)
fn clear_organism_sizes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cell_count_buffer[0] { return; }
    atomicStore(&organism_size_buffer[i], 0u);
}

// ── count_organism_sizes ──────────────────────────────────────────────────────
// Two-pass approach:
//   Pass A (this shader, first invocation): each live cell atomicAdds 1 into
//           organism_size_buffer[root_label], accumulating the true organism count.
//   Pass B (second invocation of same entry, after a barrier — handled by running
//           the shader twice): each live cell reads organism_size_buffer[root_label]
//           and writes it to organism_size_buffer[cell_i], so every cell slot holds
//           its organism's size directly. Consumers can then index by cell_idx with
//           no label lookup.
//
// Because WGSL has no cross-workgroup barrier, we split into two separate entry
// points dispatched sequentially in the same compute pass.

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
        // Dead/isolated cells get size 1 so they still pay full metabolism
        atomicStore(&organism_size_buffer[i], 1u);
        return;
    }
    if lbl >= cell_count_buffer[0] { return; }

    // Read the accumulated count at the root and copy it to this cell's slot.
    // After this pass, organism_size_buffer[cell_i] == organism size for all live cells.
    let org_size = atomicLoad(&organism_size_buffer[lbl]);
    atomicStore(&organism_size_buffer[i], org_size);
}
