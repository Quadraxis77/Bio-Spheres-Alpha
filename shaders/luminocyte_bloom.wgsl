// Luminocyte Bloom Shader
// Additive gaussian halo sprites for glowing luminocytes.
// Each quad's clip-space Z is set to the luminocyte's actual depth so the
// GPU hardware depth test handles occlusion — same mechanism that hides cells
// behind cave walls. depth_write is off so bloom doesn't pollute the depth buffer.

struct BloomCamera {
    view_proj: mat4x4<f32>,
    aspect_ratio: f32,
    bloom_radius: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> camera: BloomCamera;
@group(0) @binding(1) var<storage, read> glow_flags: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cell_count_buffer: array<u32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) brightness: f32,
}

const CORNERS = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
);

fn degenerate(out: ptr<function, VertexOut>) {
    // Place behind the far plane so the depth test always fails → nothing drawn.
    (*out).position  = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    (*out).uv        = vec2<f32>(0.0);
    (*out).color     = vec3<f32>(0.0);
    (*out).brightness = 0.0;
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vi:   u32,
    @builtin(instance_index) inst: u32,
) -> VertexOut {
    var out: VertexOut;

    if (inst >= cell_count_buffer[0] || glow_flags[inst].w <= 0.001) {
        degenerate(&out); return out;
    }

    let world_pos = positions[inst].xyz;
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    if (clip.w <= 0.001) { degenerate(&out); return out; }

    let ndc    = clip.xy / clip.w;
    let corner = CORNERS[vi];
    let offset = corner * vec2<f32>(camera.bloom_radius / camera.aspect_ratio, camera.bloom_radius);

    // Set clip-space Z to the luminocyte's actual depth so the hardware depth
    // test compares each halo fragment against scene geometry at that pixel.
    // clip.w = 1.0 here, so lum_ndc_z = clip.z / clip.w directly.
    let lum_ndc_z = clip.z / clip.w;

    out.position  = vec4<f32>(ndc + offset, lum_ndc_z, 1.0);
    out.uv        = corner;
    out.color     = glow_flags[inst].rgb;
    out.brightness = glow_flags[inst].w;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dist_sq = dot(in.uv, in.uv);
    if (dist_sq >= 1.0) { discard; }

    let falloff   = exp(-4.5 * dist_sq);
    let intensity = in.brightness * falloff;
    return vec4<f32>(in.color * intensity, intensity);
}
