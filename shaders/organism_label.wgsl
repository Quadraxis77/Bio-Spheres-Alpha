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

// Total hook+compress pairs executed per convergence cycle (2 per frame).
// 12 pairs handles linear chains of up to 2^12 = 4096 cells.
const LABEL_ITERATIONS: u32 = 12u;

// Minimum frames between convergence cycles (cooldown / debounce).
// ~30 frames ≈ 0.5 s at 60 fps.
const COOLDOWN_FRAMES: u32 = 30u;

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
    iteration:         u32,  // remaining hook+compress pairs (0 = idle)
    frames_since_init: u32,  // frames elapsed since last init (cooldown counter)
    last_cell_count:   u32,  // cell count observed at last trigger
    last_bond_count:   u32,  // live bond count observed at last trigger
    run_init:          u32,  // 1 = init_labels should execute this frame
    run_hc:            u32,  // 1 = hook+compress should execute this frame
    _pad0:             u32,
    _pad1:             u32,
}

@group(0) @binding(0) var<storage, read_write> label_state:              LabelState;
@group(0) @binding(1) var<storage, read>       cell_count_buffer:        array<u32>;
@group(0) @binding(2) var<storage, read>       adhesion_counts:          array<u32>; // [0]=total [1]=live
@group(0) @binding(3) var<storage, read_write> label_buffer:             array<atomic<u32>>;
@group(0) @binding(4) var<storage, read>       death_flags:              array<u32>;
@group(0) @binding(5) var<storage, read>       cell_adhesion_indices:    array<i32>;
@group(0) @binding(6) var<storage, read> adhesion_connections: array<AdhesionConnection>;

// ── label_controller ─────────────────────────────────────────────────────────
// Single thread. Updates the state machine and writes run_init / run_hc flags
// that the subsequent dispatches read (ordering guaranteed within the same pass).

@compute @workgroup_size(1, 1, 1)
fn label_controller() {
    let cells_now = cell_count_buffer[0];
    let bonds_now = adhesion_counts[1];

    label_state.frames_since_init += 1u;

    if label_state.iteration > 0u {
        // Mid-convergence: run hook+compress, skip init.
        label_state.run_init = 0u;
        label_state.run_hc   = 1u;
        label_state.iteration -= 1u;

    } else {
        let cells_changed = cells_now != label_state.last_cell_count;
        let bonds_changed = bonds_now != label_state.last_bond_count;

        if (cells_changed || bonds_changed) && label_state.frames_since_init >= COOLDOWN_FRAMES {
            // Kick a new convergence cycle.
            label_state.last_cell_count   = cells_now;
            label_state.last_bond_count   = bonds_now;
            label_state.frames_since_init = 0u;
            // -1: one pair runs this frame alongside init.
            label_state.iteration         = LABEL_ITERATIONS - 1u;
            label_state.run_init          = 1u;
            label_state.run_hc            = 1u;
        } else {
            // Idle.
            label_state.run_init = 0u;
            label_state.run_hc   = 0u;
        }
    }
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
