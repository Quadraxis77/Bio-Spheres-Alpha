// Adhesion line shader - outlined quads
// Renders adhesion connections as camera-facing quads with:
//   - Center colored by zone classification (green/blue/red)
//   - Outline colored by signal state (black=no signal, yellow=has signal)

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) zone_color: vec4<f32>,
    @location(2) signal_color: vec4<f32>,
    @location(3) edge_factor: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) zone_color: vec4<f32>,
    @location(1) signal_color: vec4<f32>,
    @location(2) edge_factor: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.zone_color = in.zone_color;
    out.signal_color = in.signal_color;
    out.edge_factor = in.edge_factor;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // abs(edge_factor): 0.0 at center, 1.0 at edges
    let t = abs(in.edge_factor);
    // Outer ~35% is outline (signal color), inner ~65% is zone color
    let blend = smoothstep(0.5, 0.8, t);
    return mix(in.zone_color, in.signal_color, blend);
}
