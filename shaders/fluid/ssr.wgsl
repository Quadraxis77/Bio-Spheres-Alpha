// Screen Space Reflections (SSR) for fluid surfaces
// Uses ray marching in screen space to find reflections

struct SSRParams {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    max_distance: f32,
    resolution: vec2<f32>,
    thickness: f32,
    max_steps: u32,
    stride: f32,
    jitter: f32,
    fade_start: f32,
    fade_end: f32,
    intensity: f32,
}

@group(0) @binding(0)
var<uniform> params: SSRParams;

@group(0) @binding(1)
var color_texture: texture_2d<f32>;

@group(0) @binding(2)
var depth_texture: texture_2d<f32>;

@group(0) @binding(3)
var normal_texture: texture_2d<f32>;

@group(0) @binding(4)
var linear_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen quad vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate full-screen triangle
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    
    return out;
}

// Reconstruct world position from depth
fn world_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world = params.inv_view_proj * ndc;
    return world.xyz / world.w;
}

// Project world position to screen UV
fn project_to_screen(world_pos: vec3<f32>) -> vec3<f32> {
    let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>((ndc.xy + 1.0) * 0.5, ndc.z);
}

// Linear depth from depth buffer value
fn linearize_depth(depth: f32) -> f32 {
    let near = 0.1;
    let far = 1000.0;
    return near * far / (far - depth * (far - near));
}

// Ray march in screen space to find reflection
fn ray_march(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    start_uv: vec2<f32>,
) -> vec4<f32> {
    var ray_pos = ray_origin;
    var prev_uv = start_uv;
    
    let step_size = params.stride;
    let max_steps = params.max_steps;
    
    for (var i = 0u; i < max_steps; i++) {
        ray_pos += ray_dir * step_size;
        
        // Project to screen
        let screen = project_to_screen(ray_pos);
        let uv = screen.xy;
        
        // Check bounds
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        
        // Sample depth
        let sampled_depth = textureSample(depth_texture, linear_sampler, uv).r;
        let scene_pos = world_pos_from_depth(uv, sampled_depth);
        
        // Check intersection
        let ray_depth = length(ray_pos - params.camera_pos);
        let scene_depth = length(scene_pos - params.camera_pos);
        
        if (ray_depth > scene_depth && ray_depth < scene_depth + params.thickness) {
            // Hit! Sample color
            let color = textureSample(color_texture, linear_sampler, uv).rgb;
            
            // Fade based on distance
            let dist = f32(i) / f32(max_steps);
            let fade = 1.0 - smoothstep(params.fade_start, params.fade_end, dist);
            
            // Edge fade (fade at screen edges)
            let edge_fade_x = 1.0 - pow(abs(uv.x * 2.0 - 1.0), 4.0);
            let edge_fade_y = 1.0 - pow(abs(uv.y * 2.0 - 1.0), 4.0);
            let edge_fade = edge_fade_x * edge_fade_y;
            
            return vec4<f32>(color, fade * edge_fade * params.intensity);
        }
        
        prev_uv = uv;
    }
    
    // No hit
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    
    // Sample original color
    let original_color = textureSample(color_texture, linear_sampler, uv);
    
    // Sample normal (check if this is a reflective surface)
    let normal_sample = textureSample(normal_texture, linear_sampler, uv);
    let normal = normal_sample.xyz * 2.0 - 1.0;
    let reflectivity = normal_sample.a; // Store reflectivity in alpha
    
    // Skip non-reflective surfaces
    if (reflectivity < 0.01) {
        return original_color;
    }
    
    // Get world position
    let depth = textureSample(depth_texture, linear_sampler, uv).r;
    let world_pos = world_pos_from_depth(uv, depth);
    
    // Calculate reflection direction
    let view_dir = normalize(world_pos - params.camera_pos);
    let reflect_dir = reflect(view_dir, normalize(normal));
    
    // Ray march
    let reflection = ray_march(world_pos + reflect_dir * 0.1, reflect_dir, uv);
    
    // Blend reflection with original
    let final_color = mix(original_color.rgb, reflection.rgb, reflection.a * reflectivity);
    
    return vec4<f32>(final_color, original_color.a);
}
