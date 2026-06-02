// GPU Adhesion Line Shader - Outlined Billboard Quads
// Renders adhesion connections directly from GPU buffers (no CPU readback)
// Each connection = 12 vertices (2 half-segments x 2 triangles x 3 verts)
// Center colored by zone classification, outline by signal state

const LINE_HALF_WIDTH: f32 = 0.04;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

// Adhesion connection structure (104 bytes matching Rust GpuAdhesionConnection)
// IMPORTANT: Use vec4 for anchor directions because vec3 has 16-byte alignment in WGSL
// which would cause layout mismatch with Rust's [f32; 3] + f32 padding
struct AdhesionConnection {
    cell_a_index: u32,          // offset 0
    cell_b_index: u32,          // offset 4
    mode_index: u32,            // offset 8
    is_active: u32,             // offset 12
    zone_a: u32,                // offset 16
    zone_b: u32,                // offset 20
    _align_pad: vec2<u32>,      // offset 24-31 (8 bytes)
    anchor_direction_a: vec4<f32>,  // offset 32-47 (xyz = direction, w = padding)
    anchor_direction_b: vec4<f32>,  // offset 48-63 (xyz = direction, w = padding)
    twist_reference_a: vec4<f32>,   // offset 64-79
    twist_reference_b: vec4<f32>,   // offset 80-95
    birth_time: f32,                // offset 96-99
    _pad: u32,                      // offset 100-103
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> positions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(1) @binding(2)
var<storage, read> adhesion_counts: array<u32>;

@group(1) @binding(3)
var<storage, read> cell_count_buffer: array<u32>;

@group(1) @binding(4)
var<storage, read> signal_flags: array<atomic<u32>>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) zone_color: vec4<f32>,
    @location(1) signal_color: vec4<f32>,
    @location(2) edge_factor: f32,
}

// Zone colors (matching reference implementation)
fn get_zone_color(zone: u32) -> vec4<f32> {
    switch (zone) {
        case 0u: { return vec4<f32>(0.0, 1.0, 0.0, 0.8); } // Zone A - Green
        case 1u: { return vec4<f32>(0.0, 0.0, 1.0, 0.8); } // Zone B - Blue
        default: { return vec4<f32>(1.0, 0.0, 0.0, 0.8); } // Zone C - Red
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // 12 vertices per instance: 2 half-segments x 2 triangles x 3 verts
    // Half-segment 0 (verts 0-5): A -> midpoint, zone_a color
    // Half-segment 1 (verts 6-11): midpoint -> B, zone_b color
    
    let adhesion_count = adhesion_counts[0];
    let cell_count = cell_count_buffer[0];
    
    // Check if this instance is valid
    if (instance_index >= adhesion_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.zone_color = vec4<f32>(0.0);
        out.signal_color = vec4<f32>(0.0);
        out.edge_factor = 0.0;
        return out;
    }
    
    let connection = adhesion_connections[instance_index];
    
    // Skip inactive connections
    if (connection.is_active == 0u) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.zone_color = vec4<f32>(0.0);
        out.signal_color = vec4<f32>(0.0);
        out.edge_factor = 0.0;
        return out;
    }
    
    // Validate cell indices
    if (connection.cell_a_index >= cell_count || connection.cell_b_index >= cell_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.zone_color = vec4<f32>(0.0);
        out.signal_color = vec4<f32>(0.0);
        out.edge_factor = 0.0;
        return out;
    }
    
    // Get cell positions
    let pos_a = positions[connection.cell_a_index].xyz;
    let pos_b = positions[connection.cell_b_index].xyz;
    let midpoint = (pos_a + pos_b) * 0.5;
    
    // Signal outline color: yellow only when signal actually flowed through this bond.
    // Both endpoints having signal is not sufficient - two 1-hop neighbours of the same
    // source both have signal but the connection between them was never traversed.
    // Instead, signal flowed along this bond only if the hop counts differ by exactly 1
    // (the upstream cell has one more remaining hop than the downstream cell).
    // With 16 channels, check ALL channels and highlight if ANY channel has flow.
    var signal_flowed_through = false;
    for (var ch = 0u; ch < 16u; ch++) {
        let signal_a = atomicLoad(&signal_flags[connection.cell_a_index * 16u + ch]);
        let signal_b = atomicLoad(&signal_flags[connection.cell_b_index * 16u + ch]);

        // Decode hop counts (bits 11-15) and values (bits 0-10)
        let hops_a = (signal_a >> 11u) & 31u;
        let hops_b = (signal_b >> 11u) & 31u;
        let value_a = signal_a & 2047u;
        let value_b = signal_b & 2047u;

        // Signal flowed along this bond if both ends have signal and their hop counts
        // differ by exactly 1 (one is exactly one step upstream of the other).
        let both_have_signal = (value_a != 0u) && (value_b != 0u);
        let hops_differ_by_one = (hops_a == hops_b + 1u) || (hops_b == hops_a + 1u);
        if (both_have_signal && hops_differ_by_one) {
            signal_flowed_through = true;
            break;
        }
    }
    
    var sig_color: vec4<f32>;
    if (signal_flowed_through) {
        sig_color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // Bright yellow
    } else {
        sig_color = vec4<f32>(0.0, 0.0, 0.0, 0.6); // Black
    }
    
    // Compute billboard perpendicular direction
    let line_dir = normalize(pos_b - pos_a);
    let view_dir = normalize(camera.camera_pos - midpoint);
    var perp = cross(line_dir, view_dir);
    let perp_len = length(perp);
    if (perp_len < 0.001) {
        perp = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        perp = perp / perp_len;
    }
    
    // Determine which half-segment and which triangle vertex
    let half_seg = vertex_index / 6u;  // 0 = A->mid, 1 = mid->B
    let local_vert = vertex_index % 6u; // 0-5 within the half-segment
    
    // Endpoints for this half-segment
    var seg_start: vec3<f32>;
    var seg_end: vec3<f32>;
    var zone_col: vec4<f32>;
    
    if (half_seg == 0u) {
        seg_start = pos_a;
        seg_end = midpoint;
        zone_col = get_zone_color(connection.zone_a);
    } else {
        seg_start = midpoint;
        seg_end = pos_b;
        zone_col = get_zone_color(connection.zone_b);
    }
    
    // Quad corners:
    // v0 = start + perp * hw  (edge=+1)
    // v1 = start - perp * hw  (edge=-1)
    // v2 = end   + perp * hw  (edge=+1)
    // v3 = end   - perp * hw  (edge=-1)
    // Triangle 1: v0, v1, v2 -> local_vert 0, 1, 2
    // Triangle 2: v1, v3, v2 -> local_vert 3, 4, 5
    
    var world_pos: vec3<f32>;
    var edge_f: f32;
    
    switch (local_vert) {
        case 0u: { // v0
            world_pos = seg_start + perp * LINE_HALF_WIDTH;
            edge_f = 1.0;
        }
        case 1u: { // v1
            world_pos = seg_start - perp * LINE_HALF_WIDTH;
            edge_f = -1.0;
        }
        case 2u: { // v2
            world_pos = seg_end + perp * LINE_HALF_WIDTH;
            edge_f = 1.0;
        }
        case 3u: { // v1 (second triangle)
            world_pos = seg_start - perp * LINE_HALF_WIDTH;
            edge_f = -1.0;
        }
        case 4u: { // v3
            world_pos = seg_end - perp * LINE_HALF_WIDTH;
            edge_f = -1.0;
        }
        default: { // v2 (second triangle)
            world_pos = seg_end + perp * LINE_HALF_WIDTH;
            edge_f = 1.0;
        }
    }
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.zone_color = zone_col;
    out.signal_color = sig_color;
    out.edge_factor = edge_f;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // abs(edge_factor): 0.0 at center, 1.0 at edges
    let t = abs(in.edge_factor);
    // Outer 50% is outline (signal color), inner 50% is zone color
    // This makes signal visualization much more prominent
    let blend = smoothstep(0.25, 0.75, t);
    return mix(in.zone_color, in.signal_color, blend);
}
