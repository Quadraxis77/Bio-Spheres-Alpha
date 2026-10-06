@group(0) @binding(0) var<storage, read> emission: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> intensity: array<f32>;
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&colors) { return; }
    let local = vec4<f32>(emission[id.x]) / 1024.0;
    if local.w <= 0.0 { return; }
    // Add luminocyte radiance to the same scalar field consumed by surfaces,
    // volumetric scattering, climate, and photocytes. `local.w` is already the
    // sum of all visible emitters at this voxel, so overlapping lights blend.
    let old_intensity = max(intensity[id.x], 0.0);
    let combined_intensity = min(old_intensity + local.w, 16.0);
    let local_color = local.xyz / max(local.w, 0.001);
    let blend = local.w / max(combined_intensity, 0.001);
    let old = colors[id.x];
    colors[id.x] = vec4<f32>(mix(old.xyz, local_color, blend), old.w + local.w);
    intensity[id.x] = combined_intensity;
}
