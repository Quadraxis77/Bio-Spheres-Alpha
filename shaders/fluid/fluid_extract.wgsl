// Extract density from fluid state for rendering
// Both fluid simulation and surface nets use 128³ grid with 128-stride indexing

struct ExtractParams {
    grid_resolution: u32,  // Fluid simulation resolution (128)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> extract_params: ExtractParams;
@group(0) @binding(1) var<storage, read> fluid_state: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> density_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> fluid_type_out: array<u32>;

fn get_fluid_type(state: u32) -> u32 {
    return state & 0xFFFFu;
}

fn sample_fluid_state(x: u32, y: u32, z: u32) -> u32 {
    let res = extract_params.grid_resolution;
    if x >= res || y >= res || z >= res {
        return 0u; // Out of bounds = empty
    }
    let idx = x + y * res + z * res * res;
    return atomicLoad(&fluid_state[idx]);
}

@compute @workgroup_size(4, 4, 4)
fn extract_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = extract_params.grid_resolution;  // 128

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    // Output index in 128³ grid
    let out_idx = gid.x + gid.y * res + gid.z * res * res;
    
    // Sample from 128³ simulation grid (direct 1:1 mapping)
    let sample_x = gid.x;
    let sample_y = gid.y;
    let sample_z = gid.z;
    
    let state = sample_fluid_state(sample_x, sample_y, sample_z);
    let fluid_type = get_fluid_type(state);
    let fill = f32((state >> 16u) & 0xFFFFu) / 65535.0;

    if (fluid_type == 1u || fluid_type == 2u) && fill > 0.0 {
        density_out[out_idx] = fill;
    } else {
        density_out[out_idx] = 0.0;
    }

    fluid_type_out[out_idx] = fluid_type;
}
