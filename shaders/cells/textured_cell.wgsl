// Cell Body Shader - Textured Billboard with Equirectangular Sphere Mapping
// 
// Billboards that sample from equirectangular textures mapped to a sphere.
// The texture rotates with the cell's orientation quaternion.
// LOD-based atlas: 4 LODs horizontally, 2 cell types vertically.

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
@group(0) @binding(2) var cell_atlas: texture_2d<f32>;
@group(0) @binding(3) var cell_sampler: sampler;

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
    @location(4) cell_type: f32,
    @location(5) debug_colors: f32,
    @location(6) rotation: vec4<f32>,
    @location(7) cam_right: vec3<f32>,
    @location(8) cam_up: vec3<f32>,
    @location(9) to_camera: vec3<f32>,
}

const QUAD_POSITIONS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

const PI: f32 = 3.14159265359;

// Rotate vector by quaternion (q = xyzw format)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

// Inverse quaternion rotation (conjugate)
fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let q_conj = vec4<f32>(-q.x, -q.y, -q.z, q.w);
    return quat_rotate(q_conj, v);
}

// Convert 3D direction to equirectangular UV [0,1]
fn sphere_to_equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    // Negate z to flip front/back (rotate 180 degrees around Y axis)
    let u = atan2(dir.x, -dir.z) / (2.0 * PI) + 0.5;
    let v = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    return vec2<f32>(u, 1.0 - v);
}

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
    
    let scale = instance.radius * 1.1;
    let world_pos = instance.position + right * quad_pos.x * scale + up * quad_pos.y * scale;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = quad_pos * 0.5 + 0.5;
    out.color = instance.color;
    out.center = instance.position;
    out.radius = instance.radius;
    out.cell_type = instance.type_data_1.w;
    out.debug_colors = instance.type_data_1.z;
    out.rotation = instance.rotation;
    out.cam_right = right;
    out.cam_up = up;
    out.to_camera = to_camera;
    
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Billboard-space position [-1, 1]
    let local_pos = in.uv * 2.0 - 1.0;
    let r2 = dot(local_pos, local_pos);
    
    // Discard outside circle
    if (r2 > 1.0) {
        discard;
    }
    
    // Sphere surface point (z toward camera)
    let z = sqrt(max(0.0, 1.0 - r2));
    let billboard_normal = vec3<f32>(local_pos.x, local_pos.y, z);
    
    // Transform to world space
    let world_normal = normalize(
        in.cam_right * billboard_normal.x +
        in.cam_up * billboard_normal.y +
        in.to_camera * billboard_normal.z
    );
    
    // Transform to cell's local space for texture lookup
    let local_sphere_point = quat_rotate_inverse(in.rotation, world_normal);
    
    // Convert to equirectangular UV [0,1]
    let sphere_uv = sphere_to_equirect_uv(local_sphere_point);
    
    // LOD selection based on screen size
    let clip_pos = camera.view_proj * vec4<f32>(in.center, 1.0);
    let ndc_radius = in.radius / clip_pos.w;
    let screen_radius = ndc_radius * camera.lod_scale_factor;
    
    var lod: u32;
    if (screen_radius < camera.lod_threshold_low) {
        lod = 0u;
    } else if (screen_radius < camera.lod_threshold_medium) {
        lod = 1u;
    } else if (screen_radius < camera.lod_threshold_high) {
        lod = 2u;
    } else {
        lod = 3u;
    }
    
    // Atlas layout: 1024x256
    // 4 LOD slots horizontally (256px each), 2 cell types vertically (128px each)
    // Each LOD texture fills the full slot width for proper U wrapping
    // LOD sizes: 256x16, 256x32, 256x64, 256x128 (width always 256 for wrap)
    
    let cell_type = u32(round(in.cell_type));
    let slot_width = 256.0;
    let slot_height = 128.0;
    let atlas_width = 1024.0;
    let atlas_height = 256.0;
    
    // Texture height varies by LOD, width is always full slot for wrapping
    let texture_height = slot_height / pow(2.0, f32(3u - lod));
    
    // Padding only in Y to center vertically
    let pad_y = (slot_height - texture_height) * 0.5;
    
    // Slot position
    let slot_x = f32(lod) * slot_width;
    let slot_y = f32(cell_type) * slot_height;
    
    // Wrap U coordinate manually (fract gives [0,1) range)
    let wrapped_u = fract(sphere_uv.x);
    
    // Inset UV slightly to avoid sampling across slot boundaries
    // This prevents the gap at the texture seam
    let texel_size_u = 1.0 / slot_width;
    let texel_size_v = 1.0 / texture_height;
    let inset_u = wrapped_u * (1.0 - 2.0 * texel_size_u) + texel_size_u;
    let inset_v = sphere_uv.y * (1.0 - 2.0 * texel_size_v) + texel_size_v;
    
    // Final atlas UV - U spans full slot width, V is centered
    let atlas_uv = vec2<f32>(
        (slot_x + inset_u * slot_width) / atlas_width,
        (slot_y + pad_y + inset_v * texture_height) / atlas_height
    );
    
    // Sample texture
    let tex_color = textureSample(cell_atlas, cell_sampler, atlas_uv);
    
    // Sphere depth
    let sphere_offset = z * in.radius;
    let sphere_world_pos = in.center + in.to_camera * sphere_offset;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_world_pos, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;
    
    // Lighting
    let light_dir = normalize(lighting.light_dir);
    let ndotl = max(dot(world_normal, -light_dir), 0.0);
    let diffuse = ndotl * lighting.light_color;
    
    // Color output
    var base_color: vec3<f32>;
    if (in.debug_colors > 0.5) {
        switch (lod) {
            case 0u: { base_color = vec3<f32>(1.0, 0.2, 0.2); }
            case 1u: { base_color = vec3<f32>(0.2, 1.0, 0.2); }
            case 2u: { base_color = vec3<f32>(0.2, 0.2, 1.0); }
            case 3u: { base_color = vec3<f32>(1.0, 1.0, 0.2); }
            default: { base_color = vec3<f32>(1.0, 0.2, 1.0); }
        }
    } else {
        base_color = tex_color.rgb * in.color.rgb;
    }
    
    let final_color = base_color * (lighting.ambient + diffuse);
    out.color = vec4<f32>(final_color, in.color.a);
    return out;
}
