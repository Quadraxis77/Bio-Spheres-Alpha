// Hi-Z copy depth shader
// Copies depth buffer (texture_depth_2d) to Hi-Z mip 0 (r32float)

struct HizParams {
    src_width: u32,
    src_height: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: HizParams;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var dst_texture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    
    // Get texture dimensions
    let dims = textureDimensions(depth_texture);
    
    // Early out if outside texture bounds
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    // Load depth value and write to Hi-Z
    let depth = textureLoad(depth_texture, coord, 0);
    textureStore(dst_texture, coord, vec4<f32>(depth, 0.0, 0.0, 1.0));
}
