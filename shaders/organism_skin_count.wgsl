// Organism Skin Count - determine which cells get skins
//
// Two passes:
//   1. count_organisms  - per-cell: atomicAdd into a histogram keyed by organism label
//   2. assign_skin_ids  - per-cell: read histogram, assign 16-bit skin ID if count >= threshold
//
// The histogram is a fixed-size buffer indexed by (organism_label % HISTOGRAM_SIZE).
// Collisions are possible but rare since labels are sparse cell indices.
// A collision means two organisms share a skin ID - they merge visually, which is
// acceptable as a rare edge case.

struct SkinCountParams {
    min_cells: u32,       // minimum cells for an organism to get a skin (e.g. 4)
    histogram_size: u32,  // size of the histogram buffer (e.g. 65536)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>            params: SkinCountParams;
@group(0) @binding(1) var<storage, read>      cell_count: array<u32>;
@group(0) @binding(2) var<storage, read>      death_flags: array<u32>;
@group(0) @binding(3) var<storage, read>      label_buffer: array<u32>;  // organism labels from union-find
@group(0) @binding(4) var<storage, read_write> histogram: array<atomic<u32>>;  // cell count per organism hash
@group(0) @binding(5) var<storage, read_write> cell_skin_id: array<u32>;       // output: 0 or skin_id per cell
@group(0) @binding(6) var<storage, read_write> skinned_cell_counter: array<atomic<u32>>;  // [0]: count of cells with skin

const DEAD_LABEL: u32 = 0xFFFFFFFFu;

// Hash an organism label to a histogram bin.
// Bin 0 is reserved for "no skin", so valid bins are [1, histogram_size).
fn label_to_bin(label: u32) -> u32 {
    // Simple modular hash into [1, histogram_size)
    return (label % (params.histogram_size - 1u)) + 1u;
}

// -----------------------------------------------------------------------------
// Pass 0: clear histogram
// -----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn clear_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.histogram_size { return; }
    atomicStore(&histogram[gid.x], 0u);
    // Clear the skinned cell counter (only thread 0)
    if gid.x == 0u {
        atomicStore(&skinned_cell_counter[0], 0u);
    }
}

// -----------------------------------------------------------------------------
// Pass 1: count cells per organism
// -----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn count_organisms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if cell_idx >= cell_count[0] { return; }
    if death_flags[cell_idx] != 0u { return; }

    let label = label_buffer[cell_idx];
    if label == DEAD_LABEL { return; }

    let bin = label_to_bin(label);
    atomicAdd(&histogram[bin], 1u);
}

// -----------------------------------------------------------------------------
// Pass 2: assign skin IDs based on organism size
// -----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn assign_skin_ids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if cell_idx >= cell_count[0] { return; }

    // Dead cells get no skin
    if death_flags[cell_idx] != 0u {
        cell_skin_id[cell_idx] = 0u;
        return;
    }

    let label = label_buffer[cell_idx];
    if label == DEAD_LABEL {
        cell_skin_id[cell_idx] = 0u;
        return;
    }

    let bin = label_to_bin(label);
    let count = atomicLoad(&histogram[bin]);

    if count >= params.min_cells {
        // This organism is large enough - use the histogram bin as the skin ID.
        // The bin is in [1, histogram_size), which fits in 16 bits for histogram_size <= 65536.
        cell_skin_id[cell_idx] = bin;
        atomicAdd(&skinned_cell_counter[0], 1u);
    } else {
        cell_skin_id[cell_idx] = 0u;
    }
}
