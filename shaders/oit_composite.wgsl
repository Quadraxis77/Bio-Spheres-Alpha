// OIT Composite shader - combines accumulation and revealage textures
// Renders a fullscreen quad to composite transparent fragments over opaque background

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Fullscreen triangle (3 vertices cover entire screen)
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    
    return out;
}

@group(0) @binding(0)
var accum_texture: texture_2d<f32>;

@group(0) @binding(1)
var revealage_texture: texture_2d<f32>;

@group(0) @binding(2)
var tex_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let accum = textureSample(accum_texture, tex_sampler, in.uv);
    let revealage = textureSample(revealage_texture, tex_sampler, in.uv).r;
    
    // If no transparent fragments, return transparent
    if (accum.a < 0.00001) {
        discard;
    }
    
    // Reconstruct average color: accum.rgb / accum.a
    let avg_color = accum.rgb / max(accum.a, 0.00001);
    
    // Final alpha is 1 - revealage (product of all (1-alpha))
    let final_alpha = 1.0 - revealage;
    
    // Output premultiplied color for PREMULTIPLIED_ALPHA_BLENDING
    return vec4<f32>(avg_color * final_alpha, final_alpha);
}
