// Topology-general Phase 1 microtree propagation candidate.
// Microtrees are depth-contiguous and contain at most BLOCK_SIZE cells.

struct Params {
    cell_count: u32,
    microtree_count: u32,
    microtree_start: u32,
    microtree_dispatch_count: u32,
    channel_group: u32,
    block_size: u32,
    generation: u32,
    padding: u32,
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

const BLOCK_SIZE: u32 = 64u;
const INVALID: u32 = 0xffffffffu;
var<workgroup> shared_value: array<vec4<f32>, 64>;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cells: array<CellTopology>;
@group(0) @binding(3) var<storage, read> microtrees: array<MicrotreeTopology>;
@group(0) @binding(4) var<storage, read> node_list: array<u32>;
// Upward child-boundary aggregate during local_up; per-cell from-parent value
// during local_down. Lifetimes do not overlap.
@group(0) @binding(5) var<storage, read_write> boundary_or_down: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> subtree: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> microtree_up: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> finalized: array<vec4<f32>>;
const CACHED_LAYOUT_MAGIC: u32 = 0x43414348u;

fn source_at(cell: u32) -> vec4<f32> {
    let group = select(params.channel_group, workgroup_group, params.channel_group == INVALID);
    return clamp(
        sources[cell * 4u + group],
        vec4<f32>(-1000.0),
        vec4<f32>(1000.0),
    );
}

var<private> workgroup_group: u32;

fn value_index(cell: u32) -> u32 {
    return cell * 4u + select(params.channel_group, workgroup_group, params.channel_group == INVALID);
}

fn microtree_value_index(microtree: u32) -> u32 {
    return microtree * 4u + select(params.channel_group, workgroup_group, params.channel_group == INVALID);
}

@compute @workgroup_size(64)
fn local_up(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    workgroup_group = workgroup.z;
    let relative = workgroup.x;
    if (relative >= params.microtree_dispatch_count) { return; }
    let microtree_id = params.microtree_start + relative;
    let microtree = microtrees[microtree_id];
    let live = lane < microtree.node_count;
    var node = INVALID;
    if (live) {
        node = node_list[microtree.node_offset + lane];
        shared_value[lane] = source_at(node) + boundary_or_down[value_index(node)];
    } else {
        shared_value[lane] = vec4<f32>(0.0);
    }
    workgroupBarrier();

    var remaining = select(BLOCK_SIZE, microtree.child_boundary_offset, params.padding == CACHED_LAYOUT_MAGIC);
    loop {
        if (remaining == 0u) { break; }
        let depth = remaining - 1u;
        if (live && cells[node].local_depth == depth) {
            var aggregate = source_at(node) + boundary_or_down[value_index(node)];
            if (params.padding == CACHED_LAYOUT_MAGIC) {
                for (var child_index = 0u; child_index < cells[node].padding; child_index++) {
                    let child = node_list[cells[node].role_flags + child_index];
                    aggregate += cells[child].parent_retention * shared_value[cells[child].local_index];
                }
            } else {
                for (var child_lane = 0u; child_lane < microtree.node_count; child_lane++) {
                    let child = node_list[microtree.node_offset + child_lane];
                    if (cells[child].parent_cell == node) {
                        aggregate += cells[child].parent_retention * shared_value[child_lane];
                    }
                }
            }
            shared_value[lane] = aggregate;
        }
        workgroupBarrier();
        remaining -= 1u;
    }

    if (live) {
        subtree[value_index(node)] = shared_value[lane];
    }
    if (lane == 0u) {
        let attachment = microtree.attachment_node;
        if (microtree.external_parent_cell == INVALID) {
            microtree_up[microtree_value_index(microtree_id)] = shared_value[0];
        } else {
            microtree_up[microtree_value_index(microtree_id)] =
                cells[attachment].parent_retention * shared_value[0];
        }
    }
}

@compute @workgroup_size(64)
fn local_down(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    workgroup_group = workgroup.z;
    let relative = workgroup.x;
    if (relative >= params.microtree_dispatch_count) { return; }
    let microtree_id = params.microtree_start + relative;
    let microtree = microtrees[microtree_id];
    let live = lane < microtree.node_count;
    var node = INVALID;
    if (live) {
        node = node_list[microtree.node_offset + lane];
        shared_value[lane] = select(
            vec4<f32>(0.0),
            boundary_or_down[value_index(node)],
            cells[node].local_depth == 0u,
        );
    } else {
        shared_value[lane] = vec4<f32>(0.0);
    }
    workgroupBarrier();

    let depth_count = select(BLOCK_SIZE, microtree.child_boundary_offset, params.padding == CACHED_LAYOUT_MAGIC);
    for (var depth = 0u; depth < depth_count; depth++) {
        if (live && cells[node].local_depth == depth && depth > 0u) {
            let parent = cells[node].parent_cell;
            let parent_lane = cells[parent].local_index;
            let own_up = cells[node].parent_retention * subtree[value_index(node)];
            shared_value[lane] = cells[node].parent_retention
                * (shared_value[parent_lane] + subtree[value_index(parent)] - own_up);
        }
        workgroupBarrier();
    }

    if (live) {
        let incoming = shared_value[lane];
        boundary_or_down[value_index(node)] = incoming;
        finalized[value_index(node)] = incoming + subtree[value_index(node)] - source_at(node);
    }
}

// Runs once per child microtree after its parent depth has completed local_down.
@compute @workgroup_size(256)
fn write_child_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    workgroup_group = gid.z;
    if (gid.x >= params.microtree_dispatch_count) { return; }
    let child_microtree = params.microtree_start + gid.x;
    let child = microtrees[child_microtree];
    let attachment = child.attachment_node;
    let parent = child.external_parent_cell;
    if (parent == INVALID) { return; }
    boundary_or_down[value_index(attachment)] = cells[attachment].parent_retention
        * (boundary_or_down[value_index(parent)] + subtree[value_index(parent)]
            - microtree_up[microtree_value_index(child_microtree)]);
}
