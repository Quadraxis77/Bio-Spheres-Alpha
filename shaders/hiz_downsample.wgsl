// Hi-Z downsample shader
// Downsamples Hi-Z mip N to mip N+1 by taking MAX of 2x2 block
// Using MAX depth for occlusion culling: a cell is occluded only if it's behind
// the FARTHEST thing in that region (conservative for the occludee)

struct HizParams {
    src_width: u32,
    src_height: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: HizParams;
@group(0) @binding(1) var src_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = vec2<i32>(global_id.xy);
    
    // Calculate destination dimensions
    let dst_width = max(params.src_width / 2u, 1u);
    let dst_height = max(params.src_height / 2u, 1u);
    
    // Early out if outside destination bounds
    if (global_id.x >= dst_width || global_id.y >= dst_height) {
        return;
    }
    
    // Sample 2x2 block from source
    let src_coord = dst_coord * 2;
    
    let d00 = textureLoad(src_texture, src_coord + vec2<i32>(0, 0), 0).r;
    let d10 = textureLoad(src_texture, src_coord + vec2<i32>(1, 0), 0).r;
    let d01 = textureLoad(src_texture, src_coord + vec2<i32>(0, 1), 0).r;
    let d11 = textureLoad(src_texture, src_coord + vec2<i32>(1, 1), 0).r;
    
    // Take maximum depth (farthest from camera)
    // A cell is occluded only if it's behind the farthest thing in this region
    // This is conservative: we only cull if we're SURE the cell is fully occluded
    let max_depth = max(max(d00, d10), max(d01, d11));
    
    textureStore(dst_texture, dst_coord, vec4<f32>(max_depth, 0.0, 0.0, 1.0));
}
