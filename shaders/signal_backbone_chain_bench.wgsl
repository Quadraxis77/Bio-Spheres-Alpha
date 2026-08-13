// Standalone Phase 1 blocked candidate for a cached chain.
// One workgroup solves each <=128-cell microtree locally. Only the much smaller
// boundary forest uses pointer jumping, so global work is O(N + N/B log(N/B)).

struct Params {
    cell_count: u32,
    block_count: u32,
    stride: u32,
    channel_group: u32,
}
struct ValueParams {
    cell_count: u32, signal_tick: u32, active_group_mask: u32, full_channel_cost_fixed: f32,
    signal_time: f32, tick_seconds: f32, padding0: u32, padding1: u32,
}
struct SourceMeta { identity: u32, flags: u32, requested_absolute: f32, padding: u32 }
const CELL_LIVE: u32 = 1u;
const CELL_CRITICAL_HEAT: u32 = 2u;

struct DirectionPair {
    left: vec4<f32>,
    right: vec4<f32>,
}

fn compensated_add(sum: vec4<f32>, error: vec4<f32>, value: vec4<f32>) -> DirectionPair {
    let next = sum + value;
    let correction = select(
        (value - next) + sum,
        (sum - next) + value,
        abs(sum) >= abs(value),
    );
    var result: DirectionPair;
    result.left = next;
    result.right = error + correction;
    return result;
}

fn retained_pair(high: vec4<f32>, low: vec4<f32>, retention: f32) -> DirectionPair {
    let product = retention * high;
    var result: DirectionPair;
    result.left = product;
    result.right = fma(vec4<f32>(retention), high, -product) + retention * low;
    return result;
}

fn balanced_scratch_index(node: u32, group: u32) -> u32 {
    return select(node, node + group * params.cell_count, BALANCED_COMPENSATED || BALANCED_GROUPED);
}

const BLOCK_SIZE: u32 = 128u;
const BALANCED_COMPENSATED: bool = true;
const BALANCED_GROUPED: bool = false;
var<workgroup> shared_left_coefficient: array<f32, 128>;
var<workgroup> shared_right_coefficient: array<f32, 128>;
var<workgroup> shared_left_bias: array<vec4<f32>, 128>;
var<workgroup> shared_right_bias: array<vec4<f32>, 128>;
var<workgroup> star_sum: array<vec4<f32>, 256>;
var<workgroup> gameplay_up: array<vec4<f32>, 64>;
var<workgroup> gameplay_down: array<vec4<f32>, 64>;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
// retention[i] is the edge retention between cells i-1 and i; index 0 is unused.
@group(0) @binding(2) var<storage, read> edge_retention: array<f32>;
@group(0) @binding(3) var<storage, read> coeff_in: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> coeff_out: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> bias_in: array<DirectionPair>;
@group(0) @binding(6) var<storage, read_write> bias_out: array<DirectionPair>;
@group(0) @binding(7) var<storage, read_write> finalized: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> tree_down: array<vec4<f32>>;
@group(0) @binding(9) var<uniform> value_params: ValueParams;
@group(0) @binding(10) var<storage, read> source_meta: array<SourceMeta>;
@group(0) @binding(11) var<storage, read_write> packed_public: array<vec4<u32>>;

fn heat_sign(identity: u32, channel: u32, tick: u32) -> f32 {
    var h = identity ^ (channel * 0x9e3779b9u) ^ (tick * 0x85ebca6bu);
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    return select(-1000.0, 1000.0, (h & 1u) != 0u);
}
fn publish_path(cell: u32, group: u32, received: vec4<f32>) {
    var heat = vec4<f32>(0.0);
    let source_info = source_meta[cell];
    if ((source_info.flags & (CELL_LIVE | CELL_CRITICAL_HEAT)) == (CELL_LIVE | CELL_CRITICAL_HEAT)) {
        for (var lane = 0u; lane < 4u; lane++) {
            heat[lane] = heat_sign(source_info.identity, group * 4u + lane, value_params.signal_tick);
        }
    }
    let raw = received + heat;
    let value = clamp(raw, vec4<f32>(-1000.0), vec4<f32>(1000.0));
    let saturated = select(raw > vec4<f32>(1000.0), vec4<bool>(true), raw < vec4<f32>(-1000.0));
    finalized[cell * 4u + group] = value;
    packed_public[cell * 4u + group] = (bitcast<vec4<u32>>(vec4<i32>(round(value))) & vec4<u32>(0x7ffu))
        | select(vec4<u32>(0u), vec4<u32>(1u << 11u), heat != vec4<f32>(0.0))
        | select(vec4<u32>(0u), vec4<u32>(1u << 12u), saturated);
}

fn source_at(cell: u32) -> vec4<f32> {
    return clamp(sources[cell * 4u + params.channel_group], vec4<f32>(-1000.0), vec4<f32>(1000.0));
}

fn source_at_group(cell: u32, group: u32) -> vec4<f32> {
    return clamp(sources[cell * 4u + group], vec4<f32>(-1000.0), vec4<f32>(1000.0));
}

// Produce each microtree's affine boundary transforms. For example, the left
// transform maps the incoming message at its first cell to the message entering
// the next microtree: outgoing = coefficient * incoming + bias.
@compute @workgroup_size(128)
fn summarize_block(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let block = workgroup.x;
    if (block >= params.block_count) { return; }
    let start = block * BLOCK_SIZE;
    let end = min(start + BLOCK_SIZE, params.cell_count);
    let count = end - start;
    let lane_is_live = lane < count;

    var left_coefficient = 1.0;
    var left_bias = vec4<f32>(0.0);
    if (lane_is_live) {
        let cell = start + lane;
        if (cell + 1u < params.cell_count) {
            left_coefficient = edge_retention[cell + 1u];
            left_bias = left_coefficient * source_at(cell);
        }
    }
    var right_coefficient = 1.0;
    var right_bias = vec4<f32>(0.0);
    if (lane_is_live) {
        let cell = end - 1u - lane;
        if (cell > 0u) {
            right_coefficient = edge_retention[cell];
            right_bias = right_coefficient * source_at(cell);
        }
    }
    shared_left_coefficient[lane] = left_coefficient;
    shared_left_bias[lane] = left_bias;
    shared_right_coefficient[lane] = right_coefficient;
    shared_right_bias[lane] = right_bias;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if (offset >= BLOCK_SIZE) { break; }
        let own_left_coefficient = shared_left_coefficient[lane];
        let own_left_bias = shared_left_bias[lane];
        let own_right_coefficient = shared_right_coefficient[lane];
        let own_right_bias = shared_right_bias[lane];
        var previous_left_coefficient = 1.0;
        var previous_left_bias = vec4<f32>(0.0);
        var previous_right_coefficient = 1.0;
        var previous_right_bias = vec4<f32>(0.0);
        if (lane >= offset) {
            previous_left_coefficient = shared_left_coefficient[lane - offset];
            previous_left_bias = shared_left_bias[lane - offset];
            previous_right_coefficient = shared_right_coefficient[lane - offset];
            previous_right_bias = shared_right_bias[lane - offset];
        }
        workgroupBarrier();
        if (lane >= offset) {
            shared_left_coefficient[lane] = own_left_coefficient * previous_left_coefficient;
            shared_left_bias[lane] = own_left_coefficient * previous_left_bias + own_left_bias;
            shared_right_coefficient[lane] = own_right_coefficient * previous_right_coefficient;
            shared_right_bias[lane] = own_right_coefficient * previous_right_bias + own_right_bias;
        }
        workgroupBarrier();
        offset *= 2u;
    }

    if (lane == BLOCK_SIZE - 1u) {
        coeff_out[block] = vec2<f32>(shared_left_coefficient[lane], shared_right_coefficient[lane]);
        var biases: DirectionPair;
        biases.left = shared_left_bias[lane];
        biases.right = shared_right_bias[lane];
        bias_out[block] = biases;
    }
}

@compute @workgroup_size(256)
fn scan_macro(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    if (block >= params.block_count) { return; }
    let current_coefficient = coeff_in[block];
    let current_bias = bias_in[block];
    var next_coefficient = current_coefficient;
    var next_bias = current_bias;

    if (block >= params.stride) {
        let previous = block - params.stride;
        next_coefficient.x = current_coefficient.x * coeff_in[previous].x;
        next_bias.left = current_coefficient.x * bias_in[previous].left + current_bias.left;
    }
    if (block + params.stride < params.block_count) {
        let following = block + params.stride;
        next_coefficient.y = current_coefficient.y * coeff_in[following].y;
        next_bias.right = current_coefficient.y * bias_in[following].right + current_bias.right;
    }
    coeff_out[block] = next_coefficient;
    bias_out[block] = next_bias;
}

// Rebase at the microtree boundary and solve all local directed messages. The
// shared array keeps only the left-to-right half; the reverse recurrence can
// finalize each cell immediately while walking right-to-left.
@compute @workgroup_size(128)
fn finalize_block(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let block = workgroup.x;
    if (block >= params.block_count) { return; }
    let start = block * BLOCK_SIZE;
    let end = min(start + BLOCK_SIZE, params.cell_count);
    let count = end - start;

    let lane_is_live = lane < count;
    var incoming_left = vec4<f32>(0.0);
    if (block > 0u) {
        incoming_left = bias_in[block - 1u].left;
    }
    var incoming_right = vec4<f32>(0.0);
    if (block + 1u < params.block_count) {
        incoming_right = bias_in[block + 1u].right;
    }

    var left_coefficient = 1.0;
    var left_bias = vec4<f32>(0.0);
    var right_coefficient = 1.0;
    var right_bias = vec4<f32>(0.0);
    if (lane_is_live) {
        let cell = start + lane;
        if (lane > 0u) {
            left_coefficient = edge_retention[cell];
            left_bias = left_coefficient * source_at(cell - 1u);
        }
        let reverse_lane = count - 1u - lane;
        let reverse_cell = start + reverse_lane;
        if (reverse_lane + 1u < count) {
            right_coefficient = edge_retention[reverse_cell + 1u];
            right_bias = right_coefficient * source_at(reverse_cell + 1u);
        }
    }
    shared_left_coefficient[lane] = left_coefficient;
    shared_left_bias[lane] = left_bias;
    shared_right_coefficient[lane] = right_coefficient;
    shared_right_bias[lane] = right_bias;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if (offset >= BLOCK_SIZE) { break; }
        let own_left_coefficient = shared_left_coefficient[lane];
        let own_left_bias = shared_left_bias[lane];
        let own_right_coefficient = shared_right_coefficient[lane];
        let own_right_bias = shared_right_bias[lane];
        var previous_left_coefficient = 1.0;
        var previous_left_bias = vec4<f32>(0.0);
        var previous_right_coefficient = 1.0;
        var previous_right_bias = vec4<f32>(0.0);
        if (lane >= offset) {
            previous_left_coefficient = shared_left_coefficient[lane - offset];
            previous_left_bias = shared_left_bias[lane - offset];
            previous_right_coefficient = shared_right_coefficient[lane - offset];
            previous_right_bias = shared_right_bias[lane - offset];
        }
        workgroupBarrier();
        if (lane >= offset) {
            shared_left_coefficient[lane] = own_left_coefficient * previous_left_coefficient;
            shared_left_bias[lane] = own_left_coefficient * previous_left_bias + own_left_bias;
            shared_right_coefficient[lane] = own_right_coefficient * previous_right_coefficient;
            shared_right_bias[lane] = own_right_coefficient * previous_right_bias + own_right_bias;
        }
        workgroupBarrier();
        offset *= 2u;
    }

    if (lane_is_live) {
        let reverse_lane = count - 1u - lane;
        let received_left = shared_left_coefficient[lane] * incoming_left + shared_left_bias[lane];
        let received_right = shared_right_coefficient[reverse_lane] * incoming_right
            + shared_right_bias[reverse_lane];
        let cell = start + lane;
        publish_path(cell, params.channel_group, received_left + received_right);
    }
}

// Star strategy: reduce all local sources hierarchically, then apply the exact
// no-self formula for the center and every leaf. `bias.left` is reusable
// reduction scratch; `params.block_count` is the current reduction input count.
@compute @workgroup_size(256)
fn star_partial(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) workgroup: vec3<u32>) {
    var value = vec4<f32>(0.0);
    if (gid.x < params.cell_count) {
        value = source_at(gid.x);
    }
    star_sum[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            star_sum[lane] += star_sum[lane + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (lane == 0u) {
        var partial: DirectionPair;
        partial.left = star_sum[0];
        partial.right = vec4<f32>(0.0);
        bias_out[workgroup.x] = partial;
    }
}

@compute @workgroup_size(256)
fn star_reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) workgroup: vec3<u32>) {
    var value = vec4<f32>(0.0);
    if (gid.x < params.block_count) {
        value = bias_in[gid.x].left;
    }
    star_sum[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            star_sum[lane] += star_sum[lane + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (lane == 0u) {
        var partial: DirectionPair;
        partial.left = star_sum[0];
        partial.right = vec4<f32>(0.0);
        bias_out[workgroup.x] = partial;
    }
}

@compute @workgroup_size(256)
fn star_finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.cell_count) { return; }
    let retention = 0.95;
    let total_source = bias_in[0].left;
    let own_source = source_at(cell);
    let center_source = source_at(0u);
    var received: vec4<f32>;
    if (cell == 0u) {
        received = retention * (total_source - own_source);
    } else {
        received = retention * center_source
            + retention * retention * (total_source - center_source - own_source);
    }
    finalized[cell * 4u + params.channel_group] = received;
}

// Balanced binary-tree strategy. Synthetic node IDs use heap order, so each
// depth is contiguous. `block_count` is the first node and `stride` the count.
@compute @workgroup_size(256)
fn balanced_initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if (node >= params.cell_count) { return; }
    let group = select(params.channel_group, gid.z, params.channel_group == 0xffffffffu);
    finalized[node * 4u + group] = source_at_group(node, group);
    tree_down[balanced_scratch_index(node, group)] = vec4<f32>(0.0);
}

@compute @workgroup_size(256)
fn balanced_up(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.stride) { return; }
    let node = params.block_count + gid.x;
    if (node >= params.cell_count) { return; }
    let group = select(params.channel_group, gid.z, params.channel_group == 0xffffffffu);
    let left = node * 2u + 1u;
    let right = left + 1u;
    var aggregate = source_at_group(node, group);
    var aggregate_error = vec4<f32>(0.0);
    if (left < params.cell_count) {
        if (BALANCED_COMPENSATED) {
            let retained = retained_pair(
                finalized[left * 4u + group],
                tree_down[balanced_scratch_index(left, group)],
                0.95,
            );
            var combined = compensated_add(aggregate, aggregate_error, retained.left);
            aggregate = combined.left;
            aggregate_error = combined.right + retained.right;
        } else {
            aggregate += 0.95 * finalized[left * 4u + group];
        }
    }
    if (right < params.cell_count) {
        if (BALANCED_COMPENSATED) {
            let retained = retained_pair(
                finalized[right * 4u + group],
                tree_down[balanced_scratch_index(right, group)],
                0.95,
            );
            var combined = compensated_add(aggregate, aggregate_error, retained.left);
            aggregate = combined.left;
            aggregate_error = combined.right + retained.right;
        } else {
            aggregate += 0.95 * finalized[right * 4u + group];
        }
    }
    finalized[node * 4u + group] = aggregate;
    tree_down[balanced_scratch_index(node, group)] = aggregate_error;
}

@compute @workgroup_size(256)
fn balanced_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.stride) { return; }
    let node = params.block_count + gid.x;
    if (node >= params.cell_count) { return; }
    let group = select(params.channel_group, gid.z, params.channel_group == 0xffffffffu);
    let left = node * 2u + 1u;
    let right = left + 1u;
    var left_message: DirectionPair;
    left_message.left = vec4<f32>(0.0);
    left_message.right = vec4<f32>(0.0);
    var right_message: DirectionPair;
    right_message.left = vec4<f32>(0.0);
    right_message.right = vec4<f32>(0.0);
    if (left < params.cell_count) {
        if (BALANCED_COMPENSATED) {
            left_message = retained_pair(
                finalized[left * 4u + group],
                tree_down[balanced_scratch_index(left, group)],
                0.95,
            );
        } else {
            left_message.left = 0.95 * finalized[left * 4u + group];
        }
    }
    if (right < params.cell_count) {
        if (BALANCED_COMPENSATED) {
            right_message = retained_pair(
                finalized[right * 4u + group],
                tree_down[balanced_scratch_index(right, group)],
                0.95,
            );
        } else {
            right_message.left = 0.95 * finalized[right * 4u + group];
        }
    }
    var incoming: DirectionPair;
    incoming.left = select(tree_down[balanced_scratch_index(node, group)], vec4<f32>(0.0), node == 0u);
    incoming.right = select(
        finalized[node * 4u + group],
        vec4<f32>(0.0),
        node == 0u,
    );
    if (BALANCED_COMPENSATED) {
        if (left < params.cell_count) {
            var sum = compensated_add(source_at_group(node, group), vec4<f32>(0.0), incoming.left);
            sum = compensated_add(sum.left, sum.right, incoming.right);
            sum = compensated_add(sum.left, sum.right, right_message.left);
            sum = compensated_add(sum.left, sum.right, right_message.right);
            let outgoing = retained_pair(sum.left, sum.right, 0.95);
            tree_down[balanced_scratch_index(left, group)] = outgoing.left;
            finalized[left * 4u + group] = outgoing.right;
        }
        if (right < params.cell_count) {
            var sum = compensated_add(source_at_group(node, group), vec4<f32>(0.0), incoming.left);
            sum = compensated_add(sum.left, sum.right, incoming.right);
            sum = compensated_add(sum.left, sum.right, left_message.left);
            sum = compensated_add(sum.left, sum.right, left_message.right);
            let outgoing = retained_pair(sum.left, sum.right, 0.95);
            tree_down[balanced_scratch_index(right, group)] = outgoing.left;
            finalized[right * 4u + group] = outgoing.right;
        }
        var received = compensated_add(incoming.left, incoming.right, left_message.left);
        received = compensated_add(received.left, received.right, left_message.right);
        received = compensated_add(received.left, received.right, right_message.left);
        received = compensated_add(received.left, received.right, right_message.right);
        finalized[node * 4u + group] = clamp(
            received.left + received.right,
            vec4<f32>(-1000.0),
            vec4<f32>(1000.0),
        );
    } else {
        if (left < params.cell_count) {
            tree_down[balanced_scratch_index(left, group)] = 0.95 * (source_at_group(node, group) + incoming.left + right_message.left);
        }
        if (right < params.cell_count) {
            tree_down[balanced_scratch_index(right, group)] = 0.95 * (source_at_group(node, group) + incoming.left + left_message.left);
        }
        finalized[node * 4u + group] = clamp(
            incoming.left + left_message.left + right_message.left,
            vec4<f32>(-1000.0),
            vec4<f32>(1000.0),
        );
    }
}

fn gameplay_edge_retention(local_child: u32) -> f32 {
    return select(0.95, 0.9875, local_child % 5u == 0u);
}

// Representative mixed scene: deterministic independent 37-cell balanced
// organisms, with every fifth child edge modeled as a vascular road.
@compute @workgroup_size(64)
fn gameplay_solve(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let base = workgroup.x * 37u;
    if (base >= params.cell_count) { return; }
    let count = min(37u, params.cell_count - base);
    if (lane < count) {
        gameplay_up[lane] = source_at(base + lane);
        gameplay_down[lane] = vec4<f32>(0.0);
    } else {
        gameplay_up[lane] = vec4<f32>(0.0);
        gameplay_down[lane] = vec4<f32>(0.0);
    }
    workgroupBarrier();

    var depth = 5i;
    loop {
        let first = (1u << u32(depth)) - 1u;
        let next = (1u << u32(depth + 1i)) - 1u;
        if (lane >= first && lane < next && lane < count) {
            let left = lane * 2u + 1u;
            let right = left + 1u;
            var aggregate = source_at(base + lane);
            if (left < count) {
                aggregate += gameplay_edge_retention(left) * gameplay_up[left];
            }
            if (right < count) {
                aggregate += gameplay_edge_retention(right) * gameplay_up[right];
            }
            gameplay_up[lane] = aggregate;
        }
        workgroupBarrier();
        if (depth == 0i) { break; }
        depth -= 1i;
    }

    depth = 0i;
    loop {
        let first = (1u << u32(depth)) - 1u;
        let next = (1u << u32(depth + 1i)) - 1u;
        if (lane >= first && lane < next && lane < count) {
            let left = lane * 2u + 1u;
            let right = left + 1u;
            var left_message = vec4<f32>(0.0);
            var right_message = vec4<f32>(0.0);
            if (left < count) {
                left_message = gameplay_edge_retention(left) * gameplay_up[left];
            }
            if (right < count) {
                right_message = gameplay_edge_retention(right) * gameplay_up[right];
            }
            let incoming = gameplay_down[lane];
            if (left < count) {
                gameplay_down[left] = gameplay_edge_retention(left)
                    * (source_at(base + lane) + incoming + right_message);
            }
            if (right < count) {
                gameplay_down[right] = gameplay_edge_retention(right)
                    * (source_at(base + lane) + incoming + left_message);
            }
            finalized[(base + lane) * 4u + params.channel_group] = incoming + left_message + right_message;
        }
        workgroupBarrier();
        if (depth == 5i) { break; }
        depth += 1i;
    }
}

@compute @workgroup_size(256)
fn cognocyte_bench(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.cell_count) { return; }
    let group0 = finalized[cell * 4u];
    let result = group0.x * group0.y / 1000.0;
    tree_down[cell].y = clamp(result, -1000.0, 1000.0);
}

@compute @workgroup_size(256)
fn memorocyte_bench(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.cell_count) { return; }
    let input = finalized[cell * 4u].x;
    let previous = tree_down[cell].x;
    // Fixed 15 Hz benchmark tick with configured rate 0.5.
    let effective_rate = 1.0 - pow(0.5, 1.0 / 15.0);
    let next = clamp(previous + (input - previous) * effective_rate, -1000.0, 1000.0);
    tree_down[cell] = vec4<f32>(next, next, 0.0, 0.0);
}
