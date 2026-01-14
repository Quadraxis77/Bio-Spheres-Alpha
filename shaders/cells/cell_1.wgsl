// Procedural Circle Shader - Flagellocyte (Type 1)
// GPU Instanced with Fixed LOD Sizes: 32, 64, 128, 256 pixels

struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    lod_scale_factor: f32,
    lod_threshold_low: f32,
    lod_threshold_medium: f32,
    lod_threshold_high: f32,
}

struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;

struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
    @location(3) visual_params: vec4<f32>,
    @location(4) rotation: vec4<f32>,
    @location(5) type_data_0: vec4<f32>,
    @location(6) type_data_1: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) center: vec3<f32>,
    @location(3) radius: f32,
    @location(4) visual_params: vec4<f32>,
    @location(5) cam_right: vec3<f32>,
    @location(6) cam_up: vec3<f32>,
    @location(7) to_camera: vec3<f32>,
    @location(8) lod_level: f32,
    @location(9) debug_colors: f32,
}

const QUAD_POSITIONS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    let quad_pos = QUAD_POSITIONS[vertex_index];
    
    // Billboard facing camera
    let to_camera = normalize(camera.camera_pos - instance.position);
    var right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_camera));
    var up = cross(to_camera, right);
    
    if (length(right) < 0.001) {
        right = vec3<f32>(1.0, 0.0, 0.0);
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    
    // Calculate LOD based on screen-space size
    let camera_distance = max(length(instance.position - camera.camera_pos), 1.0);
    let screen_radius = (instance.radius / camera_distance) * camera.lod_scale_factor;
    
    // Determine LOD level
    var lod_level: u32;
    if (screen_radius < camera.lod_threshold_low) {
        lod_level = 0u;  // 32 pixels
    } else if (screen_radius < camera.lod_threshold_medium) {
        lod_level = 1u;  // 64 pixels
    } else if (screen_radius < camera.lod_threshold_high) {
        lod_level = 2u;  // 128 pixels
    } else {
        lod_level = 3u;  // 256 pixels
    }
    
    // Use actual instance radius for world size
    let world_size = instance.radius * 1.1;  // Slight oversizing for antialiasing
    
    // Create billboard quad at actual cell size
    let world_pos = instance.position + right * quad_pos.x * world_size + up * quad_pos.y * world_size;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = quad_pos * 0.5 + 0.5;
    out.color = instance.color;
    out.center = instance.position;
    out.radius = instance.radius;  // Use actual radius for sphere calculations
    out.visual_params = instance.visual_params;
    out.cam_right = right;
    out.cam_up = up;
    out.to_camera = to_camera;
    out.lod_level = f32(lod_level);
    out.debug_colors = instance.type_data_1.z;  // Debug colors flag from instance data
    
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

fn circle_pattern(uv: vec2<f32>, lod: u32) -> f32 {
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(uv - center);
    let radius = 0.5;
    
    var aa_width: f32;
    switch (lod) {
        case 0u: { aa_width = 0.02; }
        case 1u: { aa_width = 0.01; }
        case 2u: { aa_width = 0.005; }
        case 3u: { aa_width = 0.0025; }
        default: { aa_width = 0.01; }
    }
    
    return smoothstep(radius + aa_width, radius - aa_width, dist);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    let lod = u32(in.lod_level);
    let local_pos = in.uv * 2.0 - 1.0;
    let r2 = dot(local_pos, local_pos);
    
    if (r2 > 1.0) {
        discard;
    }
    
    let circle_alpha = circle_pattern(in.uv, lod);
    if (circle_alpha < 0.01) {
        discard;
    }
    
    // Sphere surface point
    let z = sqrt(max(0.0, 1.0 - r2));
    let billboard_normal = vec3<f32>(local_pos.x, local_pos.y, z);
    
    let world_normal = normalize(
        in.cam_right * billboard_normal.x +
        in.cam_up * billboard_normal.y +
        in.to_camera * billboard_normal.z
    );
    
    // Sphere depth
    let sphere_offset = z * in.radius;
    let sphere_world_pos = in.center + in.to_camera * sphere_offset;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_world_pos, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;
    
    // Lighting
    let light_dir = normalize(lighting.light_dir);
    let ndotl = max(dot(world_normal, -light_dir), 0.0);
    
    let view_dir = in.to_camera;
    let half_vec = normalize(-light_dir + view_dir);
    let spec = pow(max(dot(world_normal, half_vec), 0.0), in.visual_params.y);
    let specular = spec * in.visual_params.x * lighting.light_color;
    
    let fresnel = pow(1.0 - max(dot(world_normal, view_dir), 0.0), 3.0);
    let fresnel_color = fresnel * in.visual_params.z;
    
    let diffuse = ndotl * lighting.light_color;
    let lit_color = in.color.rgb * (lighting.ambient + diffuse) + specular + fresnel_color;
    let final_color = lit_color + in.color.rgb * in.visual_params.w;
    
    // Apply LOD debug colors if enabled
    var output_color: vec3<f32>;
    if (in.debug_colors > 0.5) {
        // Show LOD level as color: Red=32, Green=64, Blue=128, Yellow=256
        switch (lod) {
            case 0u: { output_color = vec3<f32>(1.0, 0.2, 0.2); }  // Red - 32 pixels
            case 1u: { output_color = vec3<f32>(0.2, 1.0, 0.2); }  // Green - 64 pixels
            case 2u: { output_color = vec3<f32>(0.2, 0.2, 1.0); }  // Blue - 128 pixels
            case 3u: { output_color = vec3<f32>(1.0, 1.0, 0.2); }  // Yellow - 256 pixels
            default: { output_color = vec3<f32>(1.0, 0.2, 1.0); }  // Magenta - error
        }
    } else {
        output_color = final_color;
    }
    
    out.color = vec4<f32>(output_color, in.color.a * circle_alpha);
    return out;
}
