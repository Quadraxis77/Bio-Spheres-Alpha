// Moss Consumption Shader
//
// Phagocytes eat moss on contact — instantly consuming whatever is at their voxel.
// After consumption, the voxel is set to a negative value representing a "grazed"
// cooldown. The growth shader treats negative values as a timer that ticks back
// toward zero before moss can regrow, preventing immediate regrowth.
//
// Negative moss_density meaning:
//   -1.0 = just grazed (full cooldown)
//    0.0 = cooldown expired, ready to regrow
//   >0.0 = active moss

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct MossConsumeParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    nutrient_per_moss: f32,   // nutrients gained per unit of moss consumed
    graze_cooldown: f32,      // seconds before moss can regrow after being eaten
    _pad0: f32,
}

// Group 0: Physics data
@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cell_count_buffer: array<u32>;

// Group 1: Moss + cell data
@group(1) @binding(0) var<uniform> moss_params: MossConsumeParams;
@group(1) @binding(1) var<storage, read_write> moss_density: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> cell_types: array<u32>;
@group(1) @binding(3) var<storage, read_write> nutrients_buffer: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read> split_nutrient_thresholds: array<f32>;
@group(1) @binding(5) var<storage, read> death_flags: array<u32>;

const PHAGOCYTE_TYPE: u32 = 2u;
const FIXED_POINT_SCALE: f32 = 1000.0;

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

fn world_to_voxel_index(world_pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<f32>(
        (world_pos.x - moss_params.grid_origin_x) / moss_params.cell_size,
        (world_pos.y - moss_params.grid_origin_y) / moss_params.cell_size,
        (world_pos.z - moss_params.grid_origin_z) / moss_params.cell_size
    );

    let res = moss_params.grid_resolution;
    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return 0xFFFFFFFFu;
    }

    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);
    return gx + gy * res + gz * res * res;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Only phagocytes consume moss
    if (cell_types[cell_idx] != PHAGOCYTE_TYPE) {
        return;
    }

    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }

    let pos = positions[cell_idx].xyz;
    let voxel_idx = world_to_voxel_index(pos);
    if (voxel_idx == 0xFFFFFFFFu) {
        return;
    }

    // Check this voxel and all 6 face neighbors for moss
    // Cells sit in air voxels adjacent to the mossy surface voxels
    let res = moss_params.grid_resolution;
    let gp = vec3<f32>(
        (pos.x - moss_params.grid_origin_x) / moss_params.cell_size,
        (pos.y - moss_params.grid_origin_y) / moss_params.cell_size,
        (pos.z - moss_params.grid_origin_z) / moss_params.cell_size
    );
    let ix = i32(gp.x);
    let iy = i32(gp.y);
    let iz = i32(gp.z);
    let ires = i32(res);

    // Find the best voxel with moss (check self + 6 neighbors)
    var best_idx = voxel_idx;
    var best_moss = bitcast<f32>(atomicLoad(&moss_density[voxel_idx]));

    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );

    for (var i = 0; i < 6; i++) {
        let nx = ix + offsets[i].x;
        let ny = iy + offsets[i].y;
        let nz = iz + offsets[i].z;
        if (nx >= 0 && nx < ires && ny >= 0 && ny < ires && nz >= 0 && nz < ires) {
            let ni = u32(nx) + u32(ny) * res + u32(nz) * res * res;
            let nm = bitcast<f32>(atomicLoad(&moss_density[ni]));
            if (nm > best_moss) {
                best_moss = nm;
                best_idx = ni;
            }
        }
    }

    // Nothing to eat in this voxel or neighbors
    if (best_moss <= 0.0) {
        return;
    }

    // Check nutrient cap
    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    let max_nutrients = split_nutrient_thresholds[cell_idx] * 2.0;
    if (current_nutrients >= max_nutrients) {
        return;
    }

    // Eat all the moss at the best voxel: atomically set to zero
    let zero_bits = bitcast<u32>(0.0);
    let old_bits = atomicExchange(&moss_density[best_idx], zero_bits);
    let old_moss = bitcast<f32>(old_bits);

    // Another thread might have eaten it between our load and exchange
    if (old_moss <= 0.0) {
        return;
    }

    // Add nutrients proportional to moss consumed
    let nutrient_gain = old_moss * moss_params.nutrient_per_moss;
    let new_nutrients = min(current_nutrients + nutrient_gain, max_nutrients);
    atomicStore(&nutrients_buffer[cell_idx], float_to_fixed(new_nutrients));
}
