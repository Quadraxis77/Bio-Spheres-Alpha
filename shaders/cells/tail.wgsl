// Flagellocyte Tail Shader - Instanced 3D Helical Tube
//
// Renders tails as helical tubes that rotate with cell orientation.
// The helix shape is computed in the vertex shader from instance parameters.

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

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;

struct VertexInput {
    // Per-vertex
    @location(0) local_pos: vec3<f32>,   // Position on unit circle + t along length
    @location(1) local_normal: vec3<f32>, // Normal on unit circle
    @location(2) t: f32,                  // Parameter along helix (0-1)
    
    // Per-instance
    @location(3) cell_position: vec3<f32>,
    @location(4) cell_radius: f32,
    @location(5) rotation: vec4<f32>,     // Quaternion (xyzw)
    @location(6) color: vec4<f32>,
    @location(7) tail_params: vec4<f32>,  // length, thickness, amplitude, frequency
    @location(8) tail_params2: vec4<f32>, // speed, taper, time, pad
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
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let tail_length = in.tail_params.x;
    let tail_thickness = in.tail_params.y;
    let tail_amplitude = in.tail_params.z;
    let tail_frequency = in.tail_params.w;
    let tail_speed = in.tail_params2.x;
    let tail_taper = in.tail_params2.y;
    let time = in.tail_params2.z;
    
    let t = in.t;
    
    // Taper: thickness decreases along length
    let taper_factor = 1.0 - t * tail_taper;
    let thickness = tail_thickness * taper_factor * in.cell_radius; // Scale with cell size
    
    // Scale tail length with cell radius
    let scaled_tail_length = tail_length * in.cell_radius;
    let scaled_amplitude = tail_amplitude * in.cell_radius;
    
    // Helix parameters - amplitude ramps up from 0 at attachment to full at ~20% along tail
    let helix_angle = t * tail_frequency * 6.28318 + time * tail_speed;
    let amplitude_ramp = smoothstep(0.0, 0.2, t); // Smooth ramp from 0 to 1 over first 20%
    let helix_radius = scaled_amplitude * amplitude_ramp * (1.0 - t * 0.3); // Amplitude ramps up then tapers
    
    // Position along helix spine (local space, extending in -Z from cell back)
    // Small offset (5% of radius) to prevent clipping with sphere surface
    let attachment_offset = in.cell_radius * 0.05;
    let spine_x = cos(helix_angle) * helix_radius;
    let spine_y = sin(helix_angle) * helix_radius;
    let spine_z = -in.cell_radius - attachment_offset - t * scaled_tail_length; // Start slightly behind cell
    
    // Calculate tangent along helix for tube orientation
    let dt = 0.01;
    let next_t = t + dt;
    let next_angle = next_t * tail_frequency * 6.28318 + time * tail_speed;
    let next_amplitude_ramp = smoothstep(0.0, 0.2, next_t);
    let next_radius = scaled_amplitude * next_amplitude_ramp * (1.0 - next_t * 0.3);
    let next_x = cos(next_angle) * next_radius;
    let next_y = sin(next_angle) * next_radius;
    let next_z = -in.cell_radius - attachment_offset - next_t * scaled_tail_length;
    
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
    let world_position = in.cell_position + quat_rotate(in.rotation, local_position);
    
    // Transform normal
    let local_normal = right * in.local_normal.x + local_up * in.local_normal.y;
    let world_normal = quat_rotate(in.rotation, local_normal);
    
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_normal = normalize(world_normal);
    out.color = in.color;
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
