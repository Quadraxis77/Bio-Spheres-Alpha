// Cell Body Shader - Textured Billboard
// 
// Simple flat textured billboards with LOD-based texture atlas sampling.
// Nearest-neighbor filtering for pixelated edges (debug), switch to linear later.

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

// Quad vertices: two triangles forming a square
const QUAD_POSITIONS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

// Rotate vector by quaternion (q = xyzw format from instance)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qvec = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
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
    
    // Handle looking straight up/down
    if (length(right) < 0.001) {
        right = vec3<f32>(1.0, 0.0, 0.0);
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    
    // Scale billboard to cell radius
    let scale = instance.radius * 1.1; // Slight margin for texture edge
    let world_pos = instance.position + right * quad_pos.x * scale + up * quad_pos.y * scale;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = quad_pos * 0.5 + 0.5; // Convert from [-1,1] to [0,1]
    out.color = instance.color;
    out.center = instance.position;
    out.radius = instance.radius;
    out.cell_type = instance.type_data_1.w; // Cell type in last component
    out.debug_colors = instance.type_data_1.z; // Debug colors flag
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
    
    // Calculate screen-space size for LOD selection
    // clip_pos.w is the depth in view space, so radius/w gives NDC size
    // Multiply by scale factor to convert to a comparable threshold value
    let clip_pos = camera.view_proj * vec4<f32>(in.center, 1.0);
    let ndc_radius = in.radius / clip_pos.w;
    let screen_radius = ndc_radius * camera.lod_scale_factor;
    
    // Select LOD: 0=32px, 1=64px, 2=128px, 3=256px
    // Higher screen_radius = closer/larger = higher LOD
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
    
    // Atlas layout: 1024x512
    // Columns: LOD 0-3 (each 256px wide)
    // Rows: Cell type 0-1 (each 256px tall)
    // Actual texture centered within each 256x256 slot
    
    let cell_type = u32(round(in.cell_type));
    let slot_size = 256.0;
    let atlas_width = 1024.0;
    let atlas_height = 512.0;
    
    // Texture sizes: 32, 64, 128, 256
    let texture_size = slot_size / pow(2.0, f32(3u - lod));
    let padding = (slot_size - texture_size) * 0.5;
    
    // Calculate atlas UV
    let slot_x = f32(lod) * slot_size;
    let slot_y = f32(cell_type) * slot_size;
    
    let atlas_uv = vec2<f32>(
        (slot_x + padding + in.uv.x * texture_size) / atlas_width,
        (slot_y + padding + in.uv.y * texture_size) / atlas_height
    );
    
    // Sample texture
    let tex_color = textureSample(cell_atlas, cell_sampler, atlas_uv);
    
    // Discard transparent pixels
    if (tex_color.a < 0.5) {
        discard;
    }
    
    // Calculate sphere normal from UV (billboard-space)
    let local_pos = in.uv * 2.0 - 1.0; // Convert [0,1] to [-1,1]
    let r2 = dot(local_pos, local_pos);
    
    // Derive Z from sphere equation: x² + y² + z² = 1
    let z = sqrt(max(0.0, 1.0 - r2));
    
    // Calculate correct sphere depth
    // The sphere surface point is offset from the billboard by z * radius toward the camera
    let sphere_offset = z * in.radius; // How much closer the sphere surface is
    let sphere_world_pos = in.center + in.to_camera * sphere_offset;
    let sphere_clip = camera.view_proj * vec4<f32>(sphere_world_pos, 1.0);
    out.depth = sphere_clip.z / sphere_clip.w;
    
    // Local normal in billboard space (pointing toward camera)
    let billboard_normal = vec3<f32>(local_pos.x, local_pos.y, z);
    
    // Transform to world space using camera basis vectors
    // No cell rotation - sphere normals are view-dependent only
    let world_normal = normalize(
        in.cam_right * billboard_normal.x +
        in.cam_up * billboard_normal.y +
        cross(in.cam_right, in.cam_up) * billboard_normal.z
    );
    
    // Simple diffuse lighting
    let light_dir = normalize(lighting.light_dir);
    let ndotl = max(dot(world_normal, -light_dir), 0.0);
    let diffuse = ndotl * lighting.light_color;
    
    // Debug colors by LOD level
    var base_color: vec3<f32>;
    if (in.debug_colors > 0.5) {
        switch (lod) {
            case 0u: { base_color = vec3<f32>(1.0, 0.2, 0.2); } // Red - 32px
            case 1u: { base_color = vec3<f32>(0.2, 1.0, 0.2); } // Green - 64px
            case 2u: { base_color = vec3<f32>(0.2, 0.2, 1.0); } // Blue - 128px
            case 3u: { base_color = vec3<f32>(1.0, 1.0, 0.2); } // Yellow - 256px
            default: { base_color = vec3<f32>(1.0, 0.2, 1.0); } // Magenta - fallback
        }
    } else {
        // Apply instance color tint
        base_color = tex_color.rgb * in.color.rgb;
    }
    
    // Apply lighting
    let final_color = base_color * (lighting.ambient + diffuse);
    
    out.color = vec4<f32>(final_color, in.color.a);
    return out;
}
