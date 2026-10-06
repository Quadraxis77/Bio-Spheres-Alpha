// Derive per-cell caches from the restored GPU genome modes, including genomes
// absent from the CPU-authored list. Used only when loading a saved world.
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read> modes: array<u32>;
@group(0) @binding(2) var<storage, read> types: array<u32>;
@group(0) @binding(3) var<storage, read> v0: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> v2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> cell_types: array<u32>;
@group(0) @binding(6) var<storage, read_write> max_splits: array<u32>;
@group(0) @binding(7) var<storage, read_write> gain: array<f32>;
@group(0) @binding(8) var<storage, read_write> size: array<f32>;
@group(0) @binding(9) var<storage, read_write> stiffness: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= counts[0]) { return; }
    let mode = modes[i];
    if (mode >= arrayLength(&types)) { return; }
    cell_types[i] = types[mode];
    max_splits[i] = select(u32(max(v2[mode].x, 0.0)), 0xffffffffu, v2[mode].x < 0.0);
    gain[i] = select(v0[mode].x, 0.0, types[mode] == 1u);
    size[i] = v0[mode].y;
    stiffness[i] = v0[mode].z;
}
