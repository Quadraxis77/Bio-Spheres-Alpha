const GRID_RESOLUTION: u32 = 128u;
const TOTAL_VOXELS: u32 = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;

@group(0) @binding(0) var<storage, read> light_field: array<f32>;
@group(0) @binding(1) var<storage, read> light_color_field: array<vec4<f32>>;
@group(0) @binding(2) var light_field_tex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var light_color_tex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(64)
fn pack_light_field(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= TOTAL_VOXELS) {
        return;
    }

    let x = idx % GRID_RESOLUTION;
    let y = (idx / GRID_RESOLUTION) % GRID_RESOLUTION;
    let z = idx / (GRID_RESOLUTION * GRID_RESOLUTION);
    let coord = vec3<i32>(i32(x), i32(y), i32(z));

    textureStore(light_field_tex, coord, vec4<f32>(light_field[idx], 0.0, 0.0, 1.0));
    textureStore(light_color_tex, coord, light_color_field[idx]);
}
