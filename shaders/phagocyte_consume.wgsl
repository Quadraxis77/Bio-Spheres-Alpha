// Phagocyte Nutrient Consumption Shader
// Phagocytes in water voxels with nutrients consume them and gain mass
// Nutrients are stored in a separate buffer (1 = has nutrient, 0 = empty)

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

struct NutrientParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    mass_per_nutrient: f32,  // How much mass a phagocyte gains per nutrient consumed
    _pad0: f32,
    _pad1: f32,
}

// Physics bind group (group 0) - simplified to only what we need
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec4<f32>>;  // Read position, write mass

@group(0) @binding(2)
var<storage, read> cell_count_buffer: array<u32>;

// Nutrient system bind group (group 1)
@group(1) @binding(0)
var<uniform> nutrient_params: NutrientParams;

@group(1) @binding(1)
var<storage, read> fluid_state: array<u32>;

@group(1) @binding(2)
var<storage, read_write> nutrient_voxels: array<atomic<u32>>;

// Cell type buffer (group 1, binding 3) - to identify phagocytes
@group(1) @binding(3)
var<storage, read> cell_types: array<u32>;

// Split mass thresholds per cell (mass cap = 2x split_mass) - group 1, binding 4
@group(1) @binding(4)
var<storage, read> split_masses: array<f32>;

// Death flags - skip dead cells
@group(1) @binding(5)
var<storage, read> death_flags: array<u32>;

// Phagocyte cell type constant
const PHAGOCYTE_TYPE: u32 = 2u;
const DEATH_MASS_THRESHOLD: f32 = 0.5;

// Convert world position to voxel grid index
fn world_to_voxel_index(world_pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<f32>(
        (world_pos.x - nutrient_params.grid_origin_x) / nutrient_params.cell_size,
        (world_pos.y - nutrient_params.grid_origin_y) / nutrient_params.cell_size,
        (world_pos.z - nutrient_params.grid_origin_z) / nutrient_params.cell_size
    );
    
    let res = nutrient_params.grid_resolution;
    
    // Bounds check
    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return 0xFFFFFFFFu;  // Invalid index
    }
    
    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);
    
    return gx + gy * res + gz * res * res;
}

// Check if voxel contains water (fluid type 1)
fn is_water_voxel(voxel_index: u32) -> bool {
    if (voxel_index == 0xFFFFFFFFu) {
        return false;
    }
    let state = fluid_state[voxel_index];
    let fluid_type = state & 0xFFFFu;
    return fluid_type == 1u;
}

// Unpack spawn_time (tenths) from packed nutrient value
fn unpack_spawn_time(packed: u32) -> u32 {
    return packed & 0xFFFFu;
}

// Unpack lifetime (tenths) from packed nutrient value
fn unpack_lifetime(packed: u32) -> u32 {
    return (packed >> 16u) & 0xFFFFu;
}

// Get current time in tenths of a second (wrapping u16)
fn current_time_tenths_phago() -> u32 {
    return u32(params.current_time * 10.0) & 0xFFFFu;
}

// Check if voxel has a nutrient (non-zero means the populate shader kept it alive)
fn has_nutrient(voxel_index: u32) -> bool {
    if (voxel_index == 0xFFFFFFFFu) {
        return false;
    }
    return atomicLoad(&nutrient_voxels[voxel_index]) != 0u;
}

// Try to consume nutrient from voxel (atomic exchange to 0)
// Returns true if successfully consumed a nutrient
fn try_consume_nutrient(voxel_index: u32) -> bool {
    if (voxel_index == 0xFFFFFFFFu) {
        return false;
    }
    // Atomically swap to 0 — if we got a non-zero value, we consumed it
    let old_value = atomicExchange(&nutrient_voxels[voxel_index], 0u);
    return old_value != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }
    
    // Only phagocytes can consume nutrients
    let cell_type = cell_types[cell_idx];
    if (cell_type != PHAGOCYTE_TYPE) {
        return;
    }
    
    // Get cell position and current mass
    let pos_mass = positions[cell_idx];
    let pos = pos_mass.xyz;
    let current_mass = pos_mass.w;
    
    // Mass cap: 2x split_mass
    let max_mass = split_masses[cell_idx] * 2.0;
    
    // Don't consume if already at max mass
    if (current_mass >= max_mass) {
        return;
    }
    
    // Find which voxel this cell is in
    let voxel_index = world_to_voxel_index(pos);
    
    // Check if in a water voxel with nutrient
    if (!is_water_voxel(voxel_index)) {
        return;
    }
    
    // Try to consume the nutrient (atomic operation)
    if (try_consume_nutrient(voxel_index)) {
        // Successfully consumed nutrient - add mass
        let new_mass = min(current_mass + nutrient_params.mass_per_nutrient, max_mass);
        positions[cell_idx] = vec4<f32>(pos, new_mass);
    }
}
