@group(0) @binding(0) var<storage, read> emission: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4<f32>>;
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&colors) { return; }
    let local = vec4<f32>(emission[id.x]) / 1024.0;
    if local.w <= 0.0 { return; }
    let old = colors[id.x];
    let weight = old.w + local.w;
    let blend = local.w / (1.0 + weight);
    let color = mix(old.xyz, local.xyz / max(local.w, 0.001), blend);
    colors[id.x] = vec4<f32>(color, weight);
}
