// Bounded deterministic reduction of child-microtree upward messages.

struct Params {
    chunk_count: u32,
    input_kind: u32,
    padding0: u32,
    padding1: u32,
}

struct Chunk {
    input_offset: u32,
    input_count: u32,
    output_slot: u32,
    target_parent_cell: u32,
    final_output: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

var<workgroup> partial: array<vec4<f32>, 256>;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_indices: array<u32>;
@group(0) @binding(2) var<storage, read> chunks: array<Chunk>;
@group(0) @binding(3) var<storage, read> microtree_up: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> previous_partial: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> next_partial: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> boundary_sum: array<vec4<f32>>;

@compute @workgroup_size(256)
fn reduce_boundaries(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let chunk_id = workgroup.x;
    if (chunk_id >= params.chunk_count) { return; }
    let chunk = chunks[chunk_id];
    let group = workgroup.z;
    var value = vec4<f32>(0.0);
    if (lane < chunk.input_count) {
        let input_id = input_indices[chunk.input_offset + lane];
        if (params.input_kind == 0u) {
            value = microtree_up[input_id * 4u + group];
        } else {
            value = previous_partial[input_id * 4u + group];
        }
    }
    partial[lane] = value;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (lane == 0u) {
        if (chunk.final_output != 0u) {
            // Attachment sources may already have initialized this relay.
            // Reduction passes are serialized by depth, so this remains a
            // single deterministic writer for the target in this pass.
            boundary_sum[chunk.target_parent_cell * 4u + group] += partial[0];
        } else {
            next_partial[chunk.output_slot * 4u + group] = partial[0];
        }
    }
}
