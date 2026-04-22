// Depth of Field Composite Shader
// Upscales the half-resolution DoF texture back to full resolution
// using bilinear filtering. Writes directly (no alpha blend needed
// since DoF replaces the entire scene color).

@group(0) @binding(0) var dof_tex: texture_2d<f32>;
@group(0) @binding(1) var dof_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(dof_tex, dof_samp, in.uv);
}
