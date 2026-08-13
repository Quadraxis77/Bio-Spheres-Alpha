// Phase 1 topology-general benchmark ABI validation.
// This pipeline deliberately performs no signal propagation yet. It validates
// the exact flattened metadata that the local-up/macro/local-down kernels will
// consume, through actual wgpu pipeline creation and GPU buffer access.

struct Params {
    cell_count: u32,
    microtree_count: u32,
    generation: u32,
    block_size: u32,
    node_list_count: u32,
    child_list_count: u32,
    depth_offset_count: u32,
    depth_microtree_count: u32,
}

struct CellTopology {
    parent_cell: u32,
    microtree_id: u32,
    local_index: u32,
    role_flags: u32,
    generation: u32,
    parent_retention: f32,
    local_depth: u32,
    padding: u32,
}

struct MicrotreeTopology {
    node_offset: u32,
    node_count: u32,
    parent_microtree: u32,
    attachment_node: u32,
    external_parent_cell: u32,
    child_boundary_offset: u32,
    child_boundary_count: u32,
    generation: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<CellTopology>;
@group(0) @binding(2) var<storage, read> microtrees: array<MicrotreeTopology>;
@group(0) @binding(3) var<storage, read> node_list: array<u32>;
@group(0) @binding(4) var<storage, read> child_microtrees: array<u32>;
@group(0) @binding(5) var<storage, read> depth_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> depth_microtrees: array<u32>;
// invalid cells, invalid microtrees, invalid depth metadata, checked records
@group(0) @binding(7) var<storage, read_write> diagnostics: array<atomic<u32>, 4>;

const INVALID: u32 = 0xffffffffu;
const VALID_ROLE_MASK: u32 = 0x7u;

@compute @workgroup_size(256)
fn validate_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_index = gid.x;
    if (list_index >= params.node_list_count) { return; }
    atomicAdd(&diagnostics[3], 1u);

    let node = node_list[list_index];
    var invalid = node >= params.cell_count;
    if (!invalid) {
        let cell = cells[node];
        invalid = cell.microtree_id >= params.microtree_count
            || cell.generation != params.generation
            || countOneBits(cell.role_flags & VALID_ROLE_MASK) != 1u;
        if (!invalid) {
            let microtree = microtrees[cell.microtree_id];
            invalid = cell.local_index >= microtree.node_count
                || cell.local_depth >= microtree.node_count
                || microtree.node_offset + cell.local_index >= params.node_list_count
                || node_list[microtree.node_offset + cell.local_index] != node;
            if (!invalid && cell.parent_cell != INVALID
                && cells[cell.parent_cell].microtree_id == cell.microtree_id) {
                invalid = cell.local_depth != cells[cell.parent_cell].local_depth + 1u;
            }
        }
    }
    if (invalid) { atomicAdd(&diagnostics[0], 1u); }
}

@compute @workgroup_size(256)
fn validate_microtrees(@builtin(global_invocation_id) gid: vec3<u32>) {
    let microtree_id = gid.x;
    if (microtree_id >= params.microtree_count) { return; }
    let microtree = microtrees[microtree_id];
    var invalid = microtree.node_count == 0u
        || microtree.node_count > params.block_size
        || microtree.node_offset + microtree.node_count > params.node_list_count
        || microtree.child_boundary_offset + microtree.child_boundary_count > params.child_list_count
        || microtree.generation != params.generation;
    if (!invalid) {
        invalid = node_list[microtree.node_offset] != microtree.attachment_node;
    }
    if (!invalid && microtree.parent_microtree == INVALID) {
        invalid = microtree.external_parent_cell != INVALID;
    }
    if (!invalid && microtree.parent_microtree != INVALID) {
        invalid = microtree.parent_microtree >= microtree_id
            || microtree.external_parent_cell >= params.cell_count
            || cells[microtree.attachment_node].parent_cell != microtree.external_parent_cell;
    }
    if (!invalid) {
        let child_end = microtree.child_boundary_offset + microtree.child_boundary_count;
        var child_index = microtree.child_boundary_offset;
        loop {
            if (child_index >= child_end) { break; }
            let child = child_microtrees[child_index];
            if (child >= params.microtree_count || microtrees[child].parent_microtree != microtree_id) {
                invalid = true;
                break;
            }
            child_index += 1u;
        }
    }
    if (invalid) { atomicAdd(&diagnostics[1], 1u); }
}

@compute @workgroup_size(256)
fn validate_depths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.depth_microtree_count) { return; }
    var invalid = params.depth_offset_count == 0u
        || depth_offsets[0] != 0u
        || depth_offsets[params.depth_offset_count - 1u] != params.depth_microtree_count;
    let microtree_id = depth_microtrees[index];
    invalid = invalid || microtree_id >= params.microtree_count;
    if (invalid) { atomicAdd(&diagnostics[2], 1u); }
}
