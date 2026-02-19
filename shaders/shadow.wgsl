// Shadow Mapping Shaders
// Renders depth from light's perspective and samples shadow maps in other shaders

struct ShadowCamera {
    light_space_matrix: mat4x4<f32>,
    light_dir: vec3<f32>,
    shadow_map_size: u32,
    near_plane: f32,
    far_plane: f32,
};

@group(0) @binding(0)
var<uniform> shadow_camera: ShadowCamera;

@group(0) @binding(1)
var shadow_sampler: sampler_comparison;

@group(0) @binding(2)
var shadow_map: texture_depth_2d;

// Vertex shader for rendering shadow map
@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return shadow_camera.light_space_matrix * vec4<f32>(position, 1.0);
}

// Fragment shader for shadow map (no color output, just depth)
@fragment
fn fs_main() -> @builtin(frag_depth) f32 {
    return gl_FragCoord.z;
}

// Function to calculate shadow factor in other shaders
fn calculate_shadow(world_pos: vec3<f32>) -> f32 {
    // Transform world position to light space
    let light_space_pos = shadow_camera.light_space_matrix * vec4<f32>(world_pos, 1.0);
    
    // Perform perspective divide
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert to texture coordinates (flip Y for DirectX style)
    let shadow_uv = vec2<f32>(
        ndc.x * 0.5 + 0.5,
        1.0 - (ndc.y * 0.5 + 0.5)
    );
    
    // Check if within shadow map bounds
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || 
        shadow_uv.y < 0.0 || shadow_uv.y > 1.0 ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0; // Outside shadow map = fully lit
    }
    
    // Sample shadow map with PCF filtering for softer edges
    let shadow_depth = textureSampleCompare(
        shadow_map, 
        shadow_sampler, 
        shadow_uv, 
        ndc.z
    );
    
    return shadow_depth;
}

// Function to calculate shadow with PCF filtering
fn calculate_shadow_pcf(world_pos: vec3<f32>) -> f32 {
    let light_space_pos = shadow_camera.light_space_matrix * vec4<f32>(world_pos, 1.0);
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    let shadow_uv = vec2<f32>(
        ndc.x * 0.5 + 0.5,
        1.0 - (ndc.y * 0.5 + 0.5)
    );
    
    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || 
        shadow_uv.y < 0.0 || shadow_uv.y > 1.0 ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    
    // PCF (Percentage Closer Filtering) for softer shadows
    let texel_size = 1.0 / f32(shadow_camera.shadow_map_size);
    var shadow = 0.0;
    
    // 2x2 PCF (4 samples instead of 9 for better performance)
    for (let x = 0; x <= 1; x++) {
        for (let y = 0; y <= 1; y++) {
            let offset = vec2<f32>(f32(x) - 0.5, f32(y) - 0.5) * texel_size;
            let depth = textureSampleCompare(
                shadow_map, 
                shadow_sampler, 
                shadow_uv + offset, 
                ndc.z
            );
            shadow += depth;
        }
    }
    
    return shadow * 0.25; // Average of 4 samples
}
