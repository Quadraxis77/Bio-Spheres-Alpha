// Phase 3 cached-signal value stages. Topology propagation is deliberately a
// separate pipeline so it can be timed independently and replaced without
// changing source, processor, or packed-public semantics.

struct Params {
    cell_count: u32,
    signal_tick: u32,
    active_group_mask: u32,
    full_channel_cost_fixed: f32,
    signal_time: f32,
    tick_seconds: f32,
    padding0: u32,
    padding1: u32,
}

struct ProcessorConfig {
    packed: u32,
    generation: u32,
    rate: f32,
    phase: f32,
    strength: f32,
}

struct ProcessorState {
    memory: f32,
    output: f32,
    output_channel: u32,
    generation: u32,
}

struct SourceMeta {
    identity: u32,
    flags: u32,
    requested_absolute: f32,
    padding: u32,
}

const CELL_LIVE: u32 = 1u;
const CELL_CRITICAL_HEAT: u32 = 2u;
const PROCESSOR_COGNOCYTE: u32 = 1u;
const PROCESSOR_MEMOROCYTE: u32 = 2u;
const TAU: f32 = 6.283185307179586;

// Group 0 ABI is entry-point-specific; every pipeline uses an explicit layout.
@group(0) @binding(0) var<uniform> params: Params;

// Source stage.
@group(0) @binding(1) var<storage, read> requested_values: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> source_meta: array<SourceMeta>;
@group(0) @binding(3) var<storage, read> source_processor_state: array<ProcessorState>;
@group(0) @binding(4) var<storage, read_write> nutrients: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> source_diagnostics: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> source_values: array<vec4<f32>>;

fn heat_sign(identity: u32, channel: u32, tick: u32) -> f32 {
    // Fully specified integer hash: stable across CPU/GPU vendors and
    // independent for each cell/channel/tick tuple.
    var h = identity ^ (channel * 0x9e3779b9u) ^ (tick * 0x85ebca6bu);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return select(-1000.0, 1000.0, (h & 1u) != 0u);
}

fn processor_source(cell: u32, group: u32) -> vec4<f32> {
    let state = source_processor_state[cell];
    if (state.output == 0.0 || state.output_channel / 4u != group) {
        return vec4<f32>(0.0);
    }
    var value = vec4<f32>(0.0);
    value[state.output_channel & 3u] = clamp(state.output, -1000.0, 1000.0);
    return value;
}

@compute @workgroup_size(256)
fn evaluate_sources(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.cell_count) { return; }
    let source_info = source_meta[cell];
    let flags = source_info.flags;
    let live = (flags & CELL_LIVE) != 0u;
    let critical = live && (flags & CELL_CRITICAL_HEAT) != 0u;
    let state = source_processor_state[cell];
    let processor_abs = select(0.0, abs(clamp(state.output, -1000.0, 1000.0)), live);
    let ordinary_abs = max(source_info.requested_absolute, 0.0) + processor_abs;
    let ordinary_cost = ordinary_abs * params.full_channel_cost_fixed / 1000.0;
    let heat_cost = select(0.0, 16.0 * params.full_channel_cost_fixed, critical);
    let available = max(atomicLoad(&nutrients[cell]), 0);
    let paid_heat = min(available, i32(ceil(heat_cost)));
    var remaining = available - paid_heat;
    var funding = 1.0;
    if (critical) {
        // Pathological output is never browned out; ordinary output on the
        // same cell is suppressed after the heat charge.
        funding = 0.0;
    } else if (ordinary_cost > 0.0) {
        funding = clamp(f32(remaining) / ordinary_cost, 0.0, 1.0);
    }
    let paid_ordinary = min(remaining, i32(ceil(ordinary_cost * funding)));
    remaining -= paid_ordinary;
    atomicStore(&nutrients[cell], remaining);

    for (var group = 0u; group < 4u; group++) {
        let offset = cell * 4u + group;
        if (!live || (params.active_group_mask & (1u << group)) == 0u) {
            source_values[offset] = vec4<f32>(0.0);
            continue;
        }
        var heat = vec4<f32>(0.0);
        if (critical) {
            for (var lane = 0u; lane < 4u; lane++) {
                heat[lane] = heat_sign(source_info.identity, group * 4u + lane, params.signal_tick);
            }
        }
        // Do not clamp here: cancellation and saturation occur only after the
        // complete tree accumulation.
        source_values[offset] = (requested_values[offset] + processor_source(cell, group)) * funding + heat;
    }
    if (critical) { atomicAdd(&source_diagnostics[0], 1u); }
}

// Finalize/publication stage. Bindings deliberately alias source-stage numbers
// with different types under a separate explicit layout.
@group(0) @binding(1) var<storage, read_write> publish_finalized: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> publish_source_meta: array<SourceMeta>;
@group(0) @binding(3) var<storage, read_write> packed_public: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> publish_diagnostics: array<atomic<u32>>;

fn pack_signed(value: f32, pathological: bool, saturated: bool) -> u32 {
    let integer = i32(round(clamp(value, -1000.0, 1000.0)));
    let payload = bitcast<u32>(integer) & 0x7ffu;
    return payload | select(0u, 1u << 11u, pathological) | select(0u, 1u << 12u, saturated);
}

fn pack_signed_group(value: vec4<f32>, heat: vec4<f32>, saturated: vec4<bool>) -> vec4<u32> {
    let integer = vec4<i32>(round(clamp(value, vec4<f32>(-1000.0), vec4<f32>(1000.0))));
    let payload = bitcast<vec4<u32>>(integer) & vec4<u32>(0x7ffu);
    return payload
        | select(vec4<u32>(0u), vec4<u32>(1u << 11u), heat != vec4<f32>(0.0))
        | select(vec4<u32>(0u), vec4<u32>(1u << 12u), saturated);
}

@compute @workgroup_size(256)
fn finalize_and_publish(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let cell = gid.x;
    let group = gid.z;
    if (cell < params.cell_count && group < 4u) {
        let offset = cell * 4u + group;
        var heat = vec4<f32>(0.0);
        let source_info = publish_source_meta[cell];
        if ((source_info.flags & (CELL_LIVE | CELL_CRITICAL_HEAT)) == (CELL_LIVE | CELL_CRITICAL_HEAT)) {
            for (var lane = 0u; lane < 4u; lane++) {
                heat[lane] = heat_sign(source_info.identity, group * 4u + lane, params.signal_tick);
            }
        }
        let raw = publish_finalized[offset] + heat;
        let value = clamp(raw, vec4<f32>(-1000.0), vec4<f32>(1000.0));
        publish_finalized[offset] = value;
        let saturated = select(
            raw > vec4<f32>(1000.0),
            vec4<bool>(true),
            raw < vec4<f32>(-1000.0),
        );
        packed_public[offset] = pack_signed_group(value, heat, saturated);
    }
}

// Processor stage.
@group(0) @binding(1) var<storage, read> processor_field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> processor_source_meta: array<SourceMeta>;
@group(0) @binding(3) var<storage, read> processor_configs: array<ProcessorConfig>;
@group(0) @binding(4) var<storage, read_write> processor_states: array<ProcessorState>;
@group(0) @binding(6) var<storage, read_write> processor_diagnostics: array<atomic<u32>>;

fn channel_value(cell: u32, channel: u32) -> f32 {
    return processor_field[cell * 4u + min(channel, 15u) / 4u][channel & 3u];
}

fn cognocyte(op: u32, a: f32, b: f32, config: ProcessorConfig) -> f32 {
    if (op == 14u) {
        let sine = sin((config.rate * params.signal_time + config.phase) * TAU);
        let polarity = (config.packed >> 21u) & 3u;
        let normalized = select(max(sine, 0.0), sine * 0.5 + 0.5, polarity == 2u);
        let magnitude = abs(config.strength);
        if (polarity == 1u) { return -normalized * magnitude; }
        if (polarity == 2u) { return (normalized * 2.0 - 1.0) * magnitude; }
        return normalized * magnitude;
    }
    if (op == 15u) {
        let phase = fract(config.rate * params.signal_time + config.phase);
        let magnitude = abs(config.strength);
        let polarity = (config.packed >> 21u) & 3u;
        if (polarity == 1u) { return -phase * magnitude; }
        if (polarity == 2u) { return (phase * 2.0 - 1.0) * magnitude; }
        return phase * magnitude;
    }
    let unary = op == 12u || (op >= 16u && op <= 19u);
    if (!unary && (a == 0.0 || b == 0.0)) { return 0.0; }
    switch op {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b / 1000.0; }
        case 3u: { return select(a * 1000.0 / b, 0.0, abs(b) <= 0.1); }
        case 4u: { return min(a, b); }
        case 5u: { return max(a, b); }
        case 6u: { return (a + b) * 0.5; }
        case 7u: { return select(0.0, 1000.0, a > b); }
        case 8u: { return select(0.0, 1000.0, a < b); }
        case 9u: { return select(0.0, 1000.0, abs(a - b) <= 0.1); }
        case 10u: { return select(0.0, 1000.0, a > 0.0 && b > 0.0); }
        case 11u: { return select(0.0, 1000.0, a > 0.0 || b > 0.0); }
        case 12u: { return select(1000.0, 0.0, a > 0.0); }
        case 13u: { return select(0.0, b, a > 0.0); }
        case 16u: { return abs(a); }
        case 17u: { return -a; }
        case 18u: { return max(a, 0.0); }
        case 19u: { return min(a, 0.0); }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn evaluate_processors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.cell_count) { return; }
    let config = processor_configs[cell];
    let kind = config.packed & 0xfu;
    let operation = (config.packed >> 4u) & 0x1fu;
    let input_a = (config.packed >> 9u) & 0xfu;
    let input_b = (config.packed >> 13u) & 0xfu;
    let output_channel = (config.packed >> 17u) & 0xfu;
    var previous = processor_states[cell];
    var next = ProcessorState(0.0, 0.0, output_channel, config.generation);
    if ((processor_source_meta[cell].flags & CELL_LIVE) == 0u || kind == 0u) {
        processor_states[cell] = next;
        return;
    }
    if (previous.generation != config.generation) {
        previous = next;
    }
    let a = channel_value(cell, input_a);
    let b = channel_value(cell, input_b);
    var output = 0.0;
    if (kind == PROCESSOR_COGNOCYTE) {
        output = cognocyte(operation, a, b, config);
    } else if (kind == PROCESSOR_MEMOROCYTE) {
        let rate = clamp(config.rate, 0.0, 1.0);
        let effective_rate = 1.0 - pow(1.0 - rate, params.tick_seconds);
        next.memory = previous.memory + (a - previous.memory) * effective_rate;
        output = next.memory;
    }
    if (output != output || abs(output) > 3.402823e38) {
        atomicAdd(&processor_diagnostics[5], 1u);
        output = 0.0;
    }
    next.memory = clamp(next.memory, -1000.0, 1000.0);
    next.output = clamp(output, -1000.0, 1000.0);
    // Each invocation reads and writes only its own prior state after source
    // evaluation has finished, so an in-place commit preserves tick latency.
    processor_states[cell] = next;
}
