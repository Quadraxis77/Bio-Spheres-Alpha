// Inject source-only cell emissions at their immutable relay attachment.
// One invocation owns each relay, preserving deterministic attachment order.

struct Params {
    cell_count: u32,
    channel_group: u32,
    padding0: u32,
    padding1: u32,
}

struct AttachmentRange {
    offset: u32,
    count: u32,
}

struct Attachment {
    source: u32,
    retention: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ranges: array<AttachmentRange>;
@group(0) @binding(3) var<storage, read> attachments: array<Attachment>;
@group(0) @binding(4) var<storage, read_write> boundary_sum: array<vec4<f32>>;

@compute @workgroup_size(256)
fn inject_attachments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let relay = gid.x;
    if (relay >= params.cell_count) { return; }
    let range = ranges[relay];
    var sum = vec4<f32>(0.0);
    for (var index = 0u; index < range.count; index++) {
        let attachment = attachments[range.offset + index];
        sum += attachment.retention * clamp(
            sources[attachment.source * 4u + params.channel_group],
            vec4<f32>(-1000.0),
            vec4<f32>(1000.0),
        );
    }
    boundary_sum[relay * 4u + params.channel_group] = sum;
}
