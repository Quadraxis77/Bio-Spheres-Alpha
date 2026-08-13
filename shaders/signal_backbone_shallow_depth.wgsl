// O(N) deterministic solve for shallow arbitrary forests. Each invocation owns
// one node and accumulates its already-complete children in stable cell order.

struct Params {
    cell_count: u32,
    node_offset: u32,
    node_count: u32,
    padding: u32,
}

struct Topology {
    child_offset: u32,
    child_count: u32,
}

struct Child {
    node: u32,
    retention: f32,
}

const INVALID: u32 = 0xffffffffu;
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> topology: array<Topology>;
@group(0) @binding(3) var<storage, read> children: array<Child>;
@group(0) @binding(4) var<storage, read> depth_nodes: array<u32>;
@group(0) @binding(5) var<storage, read_write> subtree: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> down: array<vec4<f32>>;

fn index(cell: u32, group: u32) -> u32 { return cell * 4u + group; }
fn source(cell: u32, group: u32) -> vec4<f32> {
    return clamp(sources[index(cell, group)], vec4<f32>(-1000.0), vec4<f32>(1000.0));
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.cell_count || gid.z >= 4u) { return; }
    subtree[index(gid.x, gid.z)] = source(gid.x, gid.z);
    down[index(gid.x, gid.z)] = vec4<f32>(0.0);
}

@compute @workgroup_size(256)
fn upward(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.node_count || gid.z >= 4u) { return; }
    let node = depth_nodes[params.node_offset + gid.x];
    let info = topology[node];
    var aggregate = subtree[index(node, gid.z)];
    for (var child_index = 0u; child_index < info.child_count; child_index++) {
        let child = children[info.child_offset + child_index];
        aggregate += child.retention * subtree[index(child.node, gid.z)];
    }
    subtree[index(node, gid.z)] = aggregate;
}

@compute @workgroup_size(256)
fn downward(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.node_count || gid.z >= 4u) { return; }
    let node = depth_nodes[params.node_offset + gid.x];
    let info = topology[node];
    let incoming = down[index(node, gid.z)];
    let node_subtree = subtree[index(node, gid.z)];
    for (var child_index = 0u; child_index < info.child_count; child_index++) {
        let child = children[info.child_offset + child_index];
        let own_up = child.retention * subtree[index(child.node, gid.z)];
        down[index(child.node, gid.z)] = child.retention * (incoming + node_subtree - own_up);
    }
    // Children now hold everything they need, so the subtree slot can become
    // the finalized field in place without another full-size buffer.
    subtree[index(node, gid.z)] = incoming + node_subtree - source(node, gid.z);
}
