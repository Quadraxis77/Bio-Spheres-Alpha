struct FollowParams { cell: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
struct FollowResult { center: vec4<f32>, root: u32, count: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<uniform> params: FollowParams;
@group(0) @binding(4) var<storage, read_write> partials: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> result: FollowResult;
var<workgroup> sums: array<vec4<f32>, 256>;
fn root_label() -> u32 {
    if (params.cell >= min(counts[0], arrayLength(&labels))) { return 0xffffffffu; }
    return select(labels[params.cell], params.cell, labels[params.cell] == 0xffffffffu);
}
fn reduce(lane: u32) {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
}
@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32,
          @builtin(workgroup_id) group: vec3<u32>) {
    let i = gid.x;
    sums[lane] = vec4<f32>(0.0);
    let root = root_label();
    if (root != 0xffffffffu && i < min(counts[0], arrayLength(&positions)) && i < arrayLength(&labels)) {
        if (labels[i] == root && positions[i].w > 0.0) {
            sums[lane] = vec4<f32>(positions[i].xyz, 1.0);
        }
    }
    reduce(lane);
    if (lane == 0u) { partials[group.x] = sums[0]; }
}
@compute @workgroup_size(256)
fn finish(@builtin(local_invocation_index) lane: u32) {
    var total = vec4<f32>(0.0);
    for (var i = lane; i < arrayLength(&partials); i += 256u) { total += partials[i]; }
    sums[lane] = total;
    reduce(lane);
    if (lane == 0u) {
        result.center = vec4<f32>(sums[0].xyz / max(sums[0].w, 1.0), 0.0);
        result.root = root_label();
        result.count = u32(sums[0].w);
    }
}
