// Volumetric Fog Composite Shader
// Upscales the half-resolution fog texture back to full resolution.
// Uses a 9-tap weighted blur to hide ray-march grain before blending.

@group(0) @binding(0) var fog_tex: texture_2d<f32>;
@group(0) @binding(1) var fog_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

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
    // One texel in the half-res fog texture.
    let dim = vec2<f32>(textureDimensions(fog_tex, 0));
    let d = 1.0 / dim;

    // 9-tap tent filter (3x3 with bilinear samples at 0.5-texel offsets).
    // Sampling at ±0.5 texels lets bilinear HW average adjacent pixels for free,
    // giving the equivalent of a wider kernel at 9-sample cost.
    let h = d * 0.5; // half-texel step
    let uv = in.uv;

    var col = vec4<f32>(0.0);
    // Corner samples (bilinear at half-texel → each covers a 2x2 area of half-res pixels)
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>(-h.x, -h.y)) * 0.0625;
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>( h.x, -h.y)) * 0.0625;
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>(-h.x,  h.y)) * 0.0625;
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>( h.x,  h.y)) * 0.0625;
    // Edge samples at 1.5-texel distance (each bilinear tap spans 2 pixels)
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>(-d.x * 1.5,  0.0)) * 0.125;
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>( d.x * 1.5,  0.0)) * 0.125;
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>( 0.0, -d.y * 1.5)) * 0.125;
    col += textureSample(fog_tex, fog_samp, uv + vec2<f32>( 0.0,  d.y * 1.5)) * 0.125;
    // Center (full weight)
    col += textureSample(fog_tex, fog_samp, uv) * 0.5;

    return col;
}
