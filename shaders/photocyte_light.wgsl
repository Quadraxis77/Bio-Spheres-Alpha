// Photocyte Light Consumption Shader
// Photocytes gain mass based on light intensity at their position.
// Reads from the pre-computed light field buffer.
// Similar structure to phagocyte_consume.wgsl but uses light instead of nutrient voxels.

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

struct PhotocyteParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    // Mass gained per second at full light intensity
    mass_per_second_full_light: f32,
    // Minimum light intensity to gain any mass (threshold)
    min_light_threshold: f32,
    _pad0: f32,
}

// Physics bind group (group 0)
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec4<f32>>;  // xyz = position, w = mass

@group(0) @binding(2)
var<storage, read> cell_count_buffer: array<u32>;

// Photocyte system bind group (group 1)
@group(1) @binding(0)
var<uniform> photocyte_params: PhotocyteParams;

@group(1) @binding(1)
var<storage, read> light_field: array<f32>;  // Per-voxel light intensity (0.0-1.0)

@group(1) @binding(2)
var<storage, read> cell_types: array<u32>;  // Per-cell type ID

// Nutrients buffer (read-write, fixed-point i32)
@group(1) @binding(3)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Split nutrient thresholds per cell (nutrient cap = 2x threshold)
@group(1) @binding(4)
var<storage, read> split_nutrient_thresholds: array<f32>;

// Death flags to skip dead cells
@group(1) @binding(5)
var<storage, read> death_flags: array<u32>;

// Photocyte cell type constant
const PHOTOCYTE_TYPE: u32 = 3u;

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

// Convert world position to voxel grid index
fn world_to_voxel_index(world_pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<f32>(
        (world_pos.x - photocyte_params.grid_origin_x) / photocyte_params.cell_size,
        (world_pos.y - photocyte_params.grid_origin_y) / photocyte_params.cell_size,
        (world_pos.z - photocyte_params.grid_origin_z) / photocyte_params.cell_size,
    );
    
    let res = photocyte_params.grid_resolution;
    
    // Bounds check
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

// Sample light intensity at a world position with nearest neighbor (much faster)
fn sample_light(world_pos: vec3<f32>) -> f32 {
    let res = photocyte_params.grid_resolution;
    
    // Convert to grid-space coordinates
    let gx = (world_pos.x - photocyte_params.grid_origin_x) / photocyte_params.cell_size;
    let gy = (world_pos.y - photocyte_params.grid_origin_y) / photocyte_params.cell_size;
    let gz = (world_pos.z - photocyte_params.grid_origin_z) / photocyte_params.cell_size;
    
    // Round to nearest voxel
    let ix = i32(round(gx));
    let iy = i32(round(gy));
    let iz = i32(round(gz));
    
    let ires = i32(res);
    
    // Bounds check and sample single voxel
    if (ix < 0 || ix >= ires || iy < 0 || iy >= ires || iz < 0 || iz >= ires) {
        return 0.0; // Out of bounds = no light
    }
    
    let idx = u32(ix) + u32(iy) * res + u32(iz) * res * res;
    return light_field[idx];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Only photocytes gain nutrients from light
    let cell_type = cell_types[cell_idx];
    if (cell_type != PHOTOCYTE_TYPE) {
        return;
    }
    
    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }
    
    // Get cell position
    let pos = positions[cell_idx].xyz;

    // Read current nutrients from nutrients_buffer (fixed-point i32)
    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    
    // Skip cells with very low nutrients (would be marked dead)
    if (current_nutrients < 1.0) {
        return;
    }
    
    // Nutrient cap: 2x split_nutrient_threshold
    let max_nutrients = split_nutrient_thresholds[cell_idx] * 2.0;
    
    // Sample light field — used only as a binary shadow/occluded test.
    // The actual nutrient gain rate comes from mass_per_second_full_light
    // (which is base_rate * sun_intensity, set on the CPU).
    let light_intensity = sample_light(pos);
    let in_light = light_intensity >= photocyte_params.min_light_threshold;
    
    // Convert mass_per_second to nutrients per second (multiply by 100)
    let nutrient_rate = photocyte_params.mass_per_second_full_light * 100.0;
    
    if (in_light) {
        // In light: gain nutrients at the full rate from the brightness setting.
        // Use atomicAdd of the delta — atomicStore would overwrite concurrent writes
        // from nutrient_transport running in the same frame.
        let nutrient_gain = min(nutrient_rate * params.delta_time, max(max_nutrients - current_nutrients, 0.0));
        if (nutrient_gain > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(nutrient_gain));
        }
    } else {
        // In shadow: lose nutrients (half the gain rate).
        // Subtract via atomicAdd of a negative delta; clamp so we don't go below 1.0.
        let nutrient_loss = min(nutrient_rate * 0.5 * params.delta_time, max(current_nutrients - 1.0, 0.0));
        if (nutrient_loss > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(nutrient_loss));
        }
    }
}
