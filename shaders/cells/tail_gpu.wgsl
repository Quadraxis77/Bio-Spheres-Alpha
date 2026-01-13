// Flagellocyte Tail Shader (GPU Instance Buffer Version)
//
// Reads cell instance data from storage buffer.
// All cells are written contiguously at [0, total_count).
// The shader filters for flagellocytes (cell_type == 1) and outputs
// degenerate triangles for non-flagellocytes.

struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _padding: f32,
}

// CellInstance structure (96 bytes, matches Rust struct)
struct CellInstance {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    visual_params: vec4<f32>,
    rotation: vec4<f32>,
    type_data_0: vec4<f32>,  // tail_length, tail_thickness, tail_amplitude, tail_frequency
    type_data_1: vec4<f32>,  // tail_speed, tail_taper, tail_segments, cell_type
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<storage, read> cell_instances: array<CellInstance>;

struct VertexInput {
    @location(0) local_pos: vec3<f32>,
    @location(1) local_normal: vec3<f32>,
    @location(2) t: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) t: f32,
}

// Rotate vector by quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

@vertex
fn vs_main(
    in: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Read instance data from storage buffer
    // instance_index already accounts for first_instance from indirect buffer
    let instance = cell_instances[instance_index];
    
    // Check if this is a flagellocyte (cell_type = 1)
    // cell_type is stored in type_data_1.w
    let cell_type = u32(round(instance.type_data_1.w));
    if (cell_type != 1u) {
        // Not a flagellocyte - output vertex outside clip volume to discard
        // Using w=0 with large xyz ensures the vertex is clipped away
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 0.0);
        out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.t = 0.0;
        return out;
    }
    
    // Extract tail parameters from type_data
    let tail_length = instance.type_data_0.x;
    let tail_thickness = instance.type_data_0.y;
    let tail_amplitude = instance.type_data_0.z;
    let tail_frequency = instance.type_data_0.w;
    let tail_speed = instance.type_data_1.x;
    let tail_taper = instance.type_data_1.y;
    
    let t = in.t;
    let time = camera.time;
    
    // Taper: thickness decreases along length
    let taper_factor = 1.0 - t * tail_taper;
    // Use a minimum thickness to ensure visibility
    let thickness = max(tail_thickness * taper_factor * instance.radius, 0.05);
    
    // Scale tail length with cell radius
    let scaled_tail_length = tail_length * instance.radius;
    let scaled_amplitude = tail_amplitude * instance.radius;
    
    // Helix parameters - amplitude ramps up from 0 at attachment to full at ~20% along tail
    let helix_angle = t * tail_frequency * 6.28318 + time * tail_speed;
    let amplitude_ramp = smoothstep(0.0, 0.2, t); // Smooth ramp from 0 to 1 over first 20%
    let helix_radius = scaled_amplitude * amplitude_ramp * (1.0 - t * 0.3); // Amplitude ramps up then tapers
    
    // Position along helix spine (local space, extending in -Z from cell back)
    // Small offset (5% of radius) to prevent clipping with sphere surface
    let attachment_offset = instance.radius * 0.05;
    let spine_x = cos(helix_angle) * helix_radius;
    let spine_y = sin(helix_angle) * helix_radius;
    let spine_z = -instance.radius - attachment_offset - t * scaled_tail_length; // Start slightly behind cell
    
    // Calculate tangent along helix for tube orientation
    let dt = 0.01;
    let next_t = t + dt;
    let next_angle = next_t * tail_frequency * 6.28318 + time * tail_speed;
    let next_amplitude_ramp = smoothstep(0.0, 0.2, next_t);
    let next_radius = scaled_amplitude * next_amplitude_ramp * (1.0 - next_t * 0.3);
    let next_x = cos(next_angle) * next_radius;
    let next_y = sin(next_angle) * next_radius;
    let next_z = -instance.radius - attachment_offset - next_t * scaled_tail_length;
    
    let tangent = normalize(vec3<f32>(next_x - spine_x, next_y - spine_y, next_z - spine_z));
    
    // Build local coordinate frame around spine
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(tangent, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up, tangent));
    let local_up = cross(tangent, right);
    
    // Position vertex on tube surface
    let tube_offset = right * in.local_pos.x * thickness + local_up * in.local_pos.y * thickness;
    let local_position = vec3<f32>(spine_x, spine_y, spine_z) + tube_offset;
    
    // Transform by cell rotation and position
    let world_position = instance.position + quat_rotate(instance.rotation, local_position);
    
    // Transform normal
    let local_normal = right * in.local_normal.x + local_up * in.local_normal.y;
    let world_normal = quat_rotate(instance.rotation, local_normal);
    
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_normal = normalize(world_normal);
    out.color = instance.color;
    out.t = t;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple diffuse lighting
    let light_dir = normalize(lighting.light_dir);
    let ndotl = max(dot(in.world_normal, -light_dir), 0.0);
    let diffuse = ndotl * lighting.light_color;
    
    // Slight darkening toward tip
    let tip_darken = 1.0 - in.t * 0.3;
    
    let final_color = in.color.rgb * (lighting.ambient + diffuse) * tip_darken;
    
    return vec4<f32>(final_color, in.color.a);
}
