struct Params {
    cell_count: u32,
    operation: u32,
    padding0: u32,
    padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> finalized: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> processor_state: array<vec4<f32>>;

@compute @workgroup_size(256)
fn process_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= params.cell_count) { return; }
    let input = finalized[cell * 4u];
    if (params.operation == 1u) {
        // Representative normalized signed Cognocyte Multiply. The complete
        // operation matrix is covered by the authoritative CPU oracle.
        processor_state[cell] = vec4<f32>(
            clamp(input.x * input.y / 1000.0, -1000.0, 1000.0),
            0.0, 0.0, 0.0,
        );
    } else {
        // Memorocyte configured rate 0.5 at the fixed 15 Hz signal tick.
        let effective_rate = 1.0 - pow(0.5, 1.0 / 15.0);
        let previous = processor_state[cell].x;
        let next = clamp(previous + (input.x - previous) * effective_rate, -1000.0, 1000.0);
        processor_state[cell] = vec4<f32>(next, 0.0, 0.0, 0.0);
    }
}

