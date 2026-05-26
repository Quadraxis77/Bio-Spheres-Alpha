// Signal Clear Compute Shader
// Clears per-cell signal flags to zero at the start of each frame.
// 16 channels per cell (channels 0-7 oculocyte, 8-15 regulation).

const SIGNAL_CHANNELS: u32 = 16u;

@group(0) @binding(0)
var<storage, read_write> signal_flags: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> cell_count_buffer: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (idx >= cell_count) { return; }
    let base = idx * SIGNAL_CHANNELS;
    for (var ch = 0u; ch < SIGNAL_CHANNELS; ch++) {
        atomicStore(&signal_flags[base + ch], 0u);
    }
}
