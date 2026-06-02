// Organism Stable ID Assignment
//
// Runs after organism_label.wgsl has converged.  Maps each root label (volatile -
// changes when the minimum-index cell dies) to a stable sequential ID that persists
// across label changes.
//
// ## Buffers
//   label_buffer[cell_i]          - root label for cell i (from organism_label.wgsl)
//   organism_size_buffer[label]   - cell count for root label (0 = dead organism)
//   stable_id_map[label]          - label -> stable ID (0 = unassigned)
//   stable_id_counter[0]          - next available stable ID (atomic, starts at 1)
//   stable_id_per_cell[cell_i]    - output: stable ID for each cell (0 = no skin)
//
// ## Algorithm (two passes)
//
// Pass 1 (assign_stable_ids): one thread per cell slot.
//   For each live cell whose root label has no stable ID yet:
//     CAS stable_id_map[root_label] from 0 -> next_id (atomicAdd counter).
//   This is safe because multiple threads racing on the same root all try to
//   claim slot 0; only one wins the CAS, the rest read the winner's ID.
//
// Pass 2 (broadcast_stable_ids): one thread per cell slot.
//   stable_id_per_cell[cell_i] = stable_id_map[label_buffer[cell_i]]
//
// ## Stable ID recycling
//   Dead organisms (size = 0 at their root slot) have their stable_id_map entry
//   cleared to 0 in pass 1, making the slot available for reuse.
//   The counter wraps at MAX_STABLE_ID (512) so IDs are always in [1, 512].

const DEAD_LABEL:    u32 = 0xFFFFFFFFu;
const MAX_STABLE_ID: u32 = 512u;
const MIN_CELLS:     u32 = 4u;

@group(0) @binding(0) var<storage, read>       label_buffer:        array<u32>;
@group(0) @binding(1) var<storage, read>       organism_size_buffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> stable_id_map:       array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> stable_id_counter:   array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> stable_id_per_cell:  array<u32>;
@group(0) @binding(5) var<storage, read>       cell_count_buf:      array<u32>;
@group(0) @binding(6) var<storage, read>       death_flags:         array<u32>;

// -- Pass 1: assign stable IDs to root labels ---------------------------------
// One thread per cell. Only the thread whose cell_i == root_label does the work
// (i.e. only root cells assign IDs), avoiding races on the CAS.
@compute @workgroup_size(256, 1, 1)
fn assign_stable_ids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_i = gid.x;
    if cell_i >= cell_count_buf[0] { return; }
    if death_flags[cell_i] != 0u { return; }

    let root = label_buffer[cell_i];
    if root == DEAD_LABEL { return; }

    // Only the root cell (cell_i == root) manages the stable ID for this organism.
    // This eliminates all races - exactly one thread per organism runs this block.
    if cell_i != root { return; }

    let size = organism_size_buffer[root];

    if size < MIN_CELLS {
        // Organism too small - clear any existing stable ID so the slot is recycled.
        atomicStore(&stable_id_map[root], 0u);
        return;
    }

    // If this root already has a stable ID, keep it.
    let existing = atomicLoad(&stable_id_map[root]);
    if existing != 0u { return; }

    // Assign the next available stable ID (wrapping in [1, MAX_STABLE_ID]).
    // atomicAdd returns the old value; we add 1 and wrap.
    let raw = atomicAdd(&stable_id_counter[0], 1u);
    let new_id = (raw % MAX_STABLE_ID) + 1u;
    atomicStore(&stable_id_map[root], new_id);
}

// -- Pass 2: broadcast stable IDs to every cell -------------------------------
@compute @workgroup_size(256, 1, 1)
fn broadcast_stable_ids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_i = gid.x;
    if cell_i >= cell_count_buf[0] { return; }

    if death_flags[cell_i] != 0u {
        stable_id_per_cell[cell_i] = 0u;
        return;
    }

    let root = label_buffer[cell_i];
    if root == DEAD_LABEL {
        stable_id_per_cell[cell_i] = 0u;
        return;
    }

    stable_id_per_cell[cell_i] = atomicLoad(&stable_id_map[root]);
}
