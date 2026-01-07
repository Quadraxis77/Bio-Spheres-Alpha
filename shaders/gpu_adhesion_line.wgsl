// GPU Adhesion Line Shader
// Renders adhesion connections directly from GPU buffers (no CPU readback)
// Each adhesion connection generates 4 vertices (2 line segments)

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

// Adhesion connection structure (must match Rust/compute shader)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    anchor_direction_a: vec3<f32>,
    padding_a: f32,
    anchor_direction_b: vec3<f32>,
    padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    _padding: vec2<u32>,
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Zone colors (matching CPU implementation)
fn get_zone_color(zone: u32) -> vec4<f32> {
    switch (zone) {
        case 0u: { return vec4<f32>(1.0, 0.3, 0.3, 0.8); } // Zone A - Red
        case 1u: { return vec4<f32>(0.3, 1.0, 0.3, 0.8); } // Zone B - Green
        default: { return vec4<f32>(0.3, 0.3, 1.0, 0.8); } // Zone C - Blue
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Each instance is one adhesion connection
    // Each connection has 4 vertices: (A, mid), (mid, B)
    // vertex_index 0,1 = segment A->mid, vertex_index 2,3 = segment mid->B
    
    let adhesion_count = adhesion_counts[0];
    let cell_count = cell_count_buffer[0];
    
    // Check if this instance is valid
    if (instance_index >= adhesion_count) {
        // Output degenerate vertex (will be clipped)
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    
    let connection = adhesion_connections[instance_index];
    
    // Skip inactive connections
    if (connection.is_active == 0u) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    
    // Validate cell indices
    if (connection.cell_a_index >= cell_count || connection.cell_b_index >= cell_count) {
        out.clip_position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    
    // Get cell positions
    let pos_a = positions[connection.cell_a_index].xyz;
    let pos_b = positions[connection.cell_b_index].xyz;
    let midpoint = (pos_a + pos_b) * 0.5;
    
    // Determine position and color based on vertex index
    var world_pos: vec3<f32>;
    var color: vec4<f32>;
    
    switch (vertex_index) {
        case 0u: {
            // Segment 1 start: Cell A
            world_pos = pos_a;
            color = get_zone_color(connection.zone_a);
        }
        case 1u: {
            // Segment 1 end: Midpoint (Zone A color)
            world_pos = midpoint;
            color = get_zone_color(connection.zone_a);
        }
        case 2u: {
            // Segment 2 start: Midpoint (Zone B color)
            world_pos = midpoint;
            color = get_zone_color(connection.zone_b);
        }
        default: {
            // Segment 2 end: Cell B
            world_pos = pos_b;
            color = get_zone_color(connection.zone_b);
        }
    }
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
