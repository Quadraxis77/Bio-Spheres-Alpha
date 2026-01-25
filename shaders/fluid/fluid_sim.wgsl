// GPU Fluid Simulation - Pair-based swapping
// 6 directional passes (±X, ±Y, ±Z), each with 2 checkered phases
// Simple rule: swap neighbors unless it's air-above-water (anti-gravity)

struct FluidParams {
    grid_resolution: u32,
    world_radius: f32,
    cell_size: f32,
    direction: u32,  // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

    grid_origin: vec3<f32>,
    phase: u32,  // 0 or 1 for checkered

    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read_write> voxels: array<u32>;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn get_fluid_type(state: u32) -> u32 {
    return state & 0xFFFFu;
}

fn grid_to_world(x: u32, y: u32, z: u32) -> vec3<f32> {
    return params.grid_origin + vec3<f32>(
        f32(x) + 0.5,
        f32(y) + 0.5,
        f32(z) + 0.5
    ) * params.cell_size;
}

fn is_in_bounds(pos: vec3<f32>) -> bool {
    return length(pos) < params.world_radius * 0.95;
}

// Direction offsets: +X, -X, +Y, -Y, +Z, -Z
fn get_offset(dir: u32) -> vec3<i32> {
    switch dir {
        case 0u: { return vec3<i32>(1, 0, 0); }   // +X
        case 1u: { return vec3<i32>(-1, 0, 0); }  // -X
        case 2u: { return vec3<i32>(0, 1, 0); }   // +Y (up)
        case 3u: { return vec3<i32>(0, -1, 0); }  // -Y (down/gravity)
        case 4u: { return vec3<i32>(0, 0, 1); }   // +Z
        default: { return vec3<i32>(0, 0, -1); }  // -Z
    }
}

// Get the coordinate used for checkering based on direction
fn get_checker_coord(pos: vec3<u32>, dir: u32) -> u32 {
    switch dir {
        case 0u, 1u: { return pos.x; }  // X direction: checker on X
        case 2u, 3u: { return pos.y; }  // Y direction: checker on Y
        default: { return pos.z; }       // Z direction: checker on Z
    }
}

// Main swap pass - process pairs in given direction
@compute @workgroup_size(4, 4, 4)
fn fluid_swap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    // Checkered processing: only process cells matching current phase
    let checker = get_checker_coord(gid, params.direction);
    if (checker % 2u) != params.phase {
        return;
    }

    // Get neighbor position
    let offset = get_offset(params.direction);
    let nx = i32(gid.x) + offset.x;
    let ny = i32(gid.y) + offset.y;
    let nz = i32(gid.z) + offset.z;

    // Bounds check
    if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
        return;
    }

    let idx_a = grid_index(gid.x, gid.y, gid.z);
    let idx_b = grid_index(u32(nx), u32(ny), u32(nz));

    let state_a = voxels[idx_a];
    let state_b = voxels[idx_b];

    let type_a = get_fluid_type(state_a);
    let type_b = get_fluid_type(state_b);

    // Only consider swaps involving water (1) and empty (0)
    let a_is_water = type_a == 1u;
    let b_is_water = type_b == 1u;
    let a_is_empty = type_a == 0u;
    let b_is_empty = type_b == 0u;

    // Skip if both same (both water or both empty) or neither water/empty
    if !((a_is_water && b_is_empty) || (a_is_empty && b_is_water)) {
        return;
    }

    // For Y direction (gravity): only swap if water is above empty
    // direction 3 = -Y (looking down), so cell A is above cell B
    // direction 2 = +Y (looking up), so cell A is below cell B
    if params.direction == 3u {
        // -Y pass: A is current cell, B is below
        // Swap if A has water and B is empty (water falls)
        if !(a_is_water && b_is_empty) {
            return;
        }
    } else if params.direction == 2u {
        // +Y pass: A is current cell, B is above
        // Swap if A is empty and B has water (water falls from above)
        if !(a_is_empty && b_is_water) {
            return;
        }
    }
    // For X/Z directions: always allow swap (spreading)

    // Check world boundaries for both cells
    let world_a = grid_to_world(gid.x, gid.y, gid.z);
    let world_b = grid_to_world(u32(nx), u32(ny), u32(nz));
    if !is_in_bounds(world_a) || !is_in_bounds(world_b) {
        return;
    }

    // Swap!
    voxels[idx_a] = state_b;
    voxels[idx_b] = state_a;
}

// Initialize a sphere of water
@compute @workgroup_size(4, 4, 4)
fn fluid_init_sphere(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);

    // Sphere center at (0, world_radius * 0.5, 0), radius = world_radius * 0.45
    let sphere_center = vec3<f32>(0.0, params.world_radius * 0.5, 0.0);
    let sphere_radius = params.world_radius * 0.45;

    let dist = length(world_pos - sphere_center);

    if dist < sphere_radius && is_in_bounds(world_pos) {
        // Water: type=1, fill=1.0 -> packed as (65535 << 16) | 1
        voxels[idx] = (65535u << 16u) | 1u;
    } else {
        voxels[idx] = 0u;
    }
}

// Clear all fluid
@compute @workgroup_size(4, 4, 4)
fn fluid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    voxels[idx] = 0u;
}

// === Density extraction for Surface Nets ===

struct ExtractParams {
    grid_resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> extract_params: ExtractParams;
@group(0) @binding(1) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(2) var<storage, read_write> density_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> fluid_type_out: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn extract_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = extract_params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = gid.x + gid.y * res + gid.z * res * res;
    let state = fluid_state[idx];

    let fluid_type = state & 0xFFFFu;
    let fill = f32((state >> 16u) & 0xFFFFu) / 65535.0;

    if fluid_type == 1u && fill > 0.0 {
        density_out[idx] = fill;
    } else {
        density_out[idx] = 0.0;
    }

    fluid_type_out[idx] = fluid_type;
}
