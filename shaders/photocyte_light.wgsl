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

@group(1) @binding(3)
var<storage, read> split_masses: array<f32>;  // Per-cell split mass threshold (mass cap = 2x)

// Photocyte cell type constant
const PHOTOCYTE_TYPE: u32 = 3u;

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
        return 1.0; // Out of bounds = full light
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
    
    // Only photocytes gain mass from light
    let cell_type = cell_types[cell_idx];
    if (cell_type != PHOTOCYTE_TYPE) {
        return;
    }
    
    // Get cell position and current mass
    let pos_mass = positions[cell_idx];
    let pos = pos_mass.xyz;
    let current_mass = pos_mass.w;

    // Skip dead/dying cells - do not resurrect cells that death_scan should remove
    if (current_mass < 0.5) {
        return;
    }
    
    // Mass cap: 2x split_mass
    let max_mass = split_masses[cell_idx] * 2.0;
    
    // Sample light intensity at cell position (trilinear interpolated)
    let light_intensity = sample_light(pos);
    
    var new_mass = current_mass;
    
    if (light_intensity >= photocyte_params.min_light_threshold) {
        // In light: gain mass proportional to light intensity
        let effective_intensity = (light_intensity - photocyte_params.min_light_threshold) 
                                / (1.0 - photocyte_params.min_light_threshold);
        let mass_gain = photocyte_params.mass_per_second_full_light * effective_intensity * params.delta_time;
        new_mass = min(current_mass + mass_gain, max_mass);
    } else {
        // In shadow: metabolic cost — photocytes slowly lose mass without light
        // Loss rate is half of max gain rate
        let mass_loss = photocyte_params.mass_per_second_full_light * 0.5 * params.delta_time;
        new_mass = max(current_mass - mass_loss, 0.1);  // Don't go below minimum viable mass
    }
    
    // Only write if mass actually changed
    if (new_mass != current_mass) {
        positions[cell_idx] = vec4<f32>(pos, new_mass);
    }
}
