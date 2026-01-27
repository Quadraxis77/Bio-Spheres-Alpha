// Water Bitfield Compression Shader
// Compresses 128³ voxel grid into bitfield for fast water detection
// Each u32 contains 32 consecutive voxels along X axis
//
// Input: 2,097,152 voxels (8MB as u32)
// Output: 65,536 u32 bitfield (256KB) - 32x smaller
//
// Bit layout: bit i represents voxel at x = (base_x + i)
// where base_x = (u32_index % 4) * 32

struct BitfieldParams {
    grid_resolution: u32,  // 128
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: BitfieldParams;
@group(0) @binding(1) var<storage, read> voxels: array<u32>;
@group(0) @binding(2) var<storage, read_write> water_bitfield: array<atomic<u32>>;

// Extract fluid type from voxel state (lower 16 bits)
fn get_fluid_type(state: u32) -> u32 {
    return state & 0xFFFFu;
}

// Each thread processes one u32 in the bitfield (32 voxels)
@compute @workgroup_size(256)
fn generate_water_bitfield(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bitfield_idx = global_id.x;
    let res = params.grid_resolution;  // 128
    let x_groups = res / 32u;  // 4 (number of u32s per row)

    // Total bitfield size: 128/32 * 128 * 128 = 65536
    let total_bitfield_size = x_groups * res * res;
    if (bitfield_idx >= total_bitfield_size) {
        return;
    }

    // Decode which 32-voxel group this is
    // bitfield_idx = x_group + y * x_groups + z * x_groups * res
    let x_group = bitfield_idx % x_groups;
    let y = (bitfield_idx / x_groups) % res;
    let z = bitfield_idx / (x_groups * res);

    let base_x = x_group * 32u;

    // Pack 32 voxels into one u32
    var bits = 0u;
    for (var i = 0u; i < 32u; i++) {
        let x = base_x + i;
        let voxel_idx = x + y * res + z * res * res;
        let voxel_state = voxels[voxel_idx];
        let fluid_type = get_fluid_type(voxel_state);

        // Set bit if this voxel contains water (type == 1)
        if (fluid_type == 1u) {
            bits = bits | (1u << i);
        }
    }

    // Write compressed result
    atomicStore(&water_bitfield[bitfield_idx], bits);
}

// Clear the bitfield (for initialization)
@compute @workgroup_size(256)
fn clear_water_bitfield(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bitfield_idx = global_id.x;
    let res = params.grid_resolution;
    let x_groups = res / 32u;
    let total_bitfield_size = x_groups * res * res;

    if (bitfield_idx >= total_bitfield_size) {
        return;
    }

    atomicStore(&water_bitfield[bitfield_idx], 0u);
}
