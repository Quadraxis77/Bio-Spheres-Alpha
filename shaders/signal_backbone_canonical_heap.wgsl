// Fast canonical-heap representation for broad shallow trees. Phase 4 may
// remap arbitrary rebuilt components into this cache order.
struct Params { cell_count: u32, first: u32, count: u32, padding: u32 }
struct ValueParams {
    cell_count: u32, signal_tick: u32, active_group_mask: u32, full_channel_cost_fixed: f32,
    signal_time: f32, tick_seconds: f32, padding0: u32, padding1: u32,
}
struct SourceMeta { identity: u32, flags: u32, requested_absolute: f32, padding: u32 }
const CELL_LIVE: u32 = 1u;
const CELL_CRITICAL_HEAT: u32 = 2u;
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> retention: array<f32>;
@group(0) @binding(3) var<storage, read_write> field: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> finalized: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> source_meta: array<SourceMeta>;
@group(0) @binding(7) var<storage, read_write> packed_public: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(9) var<uniform> value_params: ValueParams;
fn at(cell: u32, group: u32) -> u32 { return group * params.cell_count + cell; }
fn down_at(cell: u32, group: u32) -> u32 { return cell * 4u + group; }
fn source(cell: u32, group: u32) -> vec4<f32> {
    return clamp(sources[cell * 4u + group], vec4<f32>(-1000.0), vec4<f32>(1000.0));
}
fn aggregate(cell: u32, group: u32) -> vec4<f32> {
    // A leaf subtree is exactly its source, so no initialization copy is needed.
    if (cell * 2u + 1u >= params.cell_count) { return source(cell, group); }
    return field[at(cell, group)];
}
fn heat_sign(identity: u32, channel: u32, tick: u32) -> f32 {
    var h = identity ^ (channel * 0x9e3779b9u) ^ (tick * 0x85ebca6bu);
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    return select(-1000.0, 1000.0, (h & 1u) != 0u);
}
fn pack_group(value: vec4<f32>, heat: vec4<f32>, saturated: vec4<bool>) -> vec4<u32> {
    let integer = vec4<i32>(round(value));
    return (bitcast<vec4<u32>>(integer) & vec4<u32>(0x7ffu))
        | select(vec4<u32>(0u), vec4<u32>(1u << 11u), heat != vec4<f32>(0.0))
        | select(vec4<u32>(0u), vec4<u32>(1u << 12u), saturated);
}
@compute @workgroup_size(256)
fn upward(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count || gid.z >= 4u) { return; }
    let node = params.first + gid.x;
    let left = node * 2u + 1u;
    let right = left + 1u;
    var sum = source(node, gid.z);
    if (left < params.cell_count) { sum += retention[left] * aggregate(left, gid.z); }
    if (right < params.cell_count) { sum += retention[right] * aggregate(right, gid.z); }
    field[at(node, gid.z)] = sum;
}
fn process_downward(node: u32, group: u32) {
    let left = node * 2u + 1u;
    let right = left + 1u;
    var incoming = vec4<f32>(0.0);
    if (node != 0u) { incoming = finalized[down_at(node, group)]; }
    let total = aggregate(node, group);
    var left_aggregate = vec4<f32>(0.0);
    var right_aggregate = vec4<f32>(0.0);
    if (left < params.cell_count) { left_aggregate = aggregate(left, group); }
    if (right < params.cell_count) { right_aggregate = aggregate(right, group); }
    if (left < params.cell_count) {
        finalized[down_at(left, group)] = retention[left] * (incoming + total - retention[left] * left_aggregate);
    }
    if (right < params.cell_count) {
        finalized[down_at(right, group)] = retention[right] * (incoming + total - retention[right] * right_aggregate);
    }
    var heat = vec4<f32>(0.0);
    let source_info = source_meta[node];
    if ((source_info.flags & (CELL_LIVE | CELL_CRITICAL_HEAT)) == (CELL_LIVE | CELL_CRITICAL_HEAT)) {
        for (var heat_lane = 0u; heat_lane < 4u; heat_lane++) {
            heat[heat_lane] = heat_sign(source_info.identity, group * 4u + heat_lane, value_params.signal_tick);
        }
    }
    // Downward messages reuse the final-output slot and are replaced in place
    // when that node is evaluated, avoiding a third full-size field.
    var raw = incoming + heat;
    if (left < params.cell_count) { raw += retention[left] * left_aggregate; }
    if (right < params.cell_count) { raw += retention[right] * right_aggregate; }
    let value = clamp(raw, vec4<f32>(-1000.0), vec4<f32>(1000.0));
    let saturated = select(raw > vec4<f32>(1000.0), vec4<bool>(true), raw < vec4<f32>(-1000.0));
    finalized[node * 4u + group] = value;
    packed_public[node * 4u + group] = pack_group(value, heat, saturated);
}

@compute @workgroup_size(256)
fn upward_top(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group_id: vec3<u32>) {
    let group = group_id.z;
    for (var depth = 9u; depth > 0u; depth--) {
        let level = depth - 1u;
        let first = (1u << level) - 1u;
        let count = min(1u << level, params.cell_count - min(first, params.cell_count));
        if (lane < count) {
            let node = first + lane;
            let left = node * 2u + 1u;
            let right = left + 1u;
            var sum = source(node, group);
            if (left < params.cell_count) { sum += retention[left] * aggregate(left, group); }
            if (right < params.cell_count) { sum += retention[right] * aggregate(right, group); }
            field[at(node, group)] = sum;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn downward_top(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group_id: vec3<u32>) {
    let group = group_id.z;
    for (var level = 0u; level < 9u; level++) {
        let first = (1u << level) - 1u;
        let count = min(1u << level, params.cell_count - min(first, params.cell_count));
        if (lane < count) { process_downward(first + lane, group); }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn downward(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.count && gid.z < 4u) {
        process_downward(params.first + gid.x, gid.z);
    }
}
