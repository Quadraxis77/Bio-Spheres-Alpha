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

// Nutrients buffer (read-write, fixed-point i32)
@group(1) @binding(4)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Split nutrient thresholds per cell (nutrient cap = 2x threshold)
@group(1) @binding(5)
var<storage, read> split_nutrient_thresholds: array<f32>;

// Death flags to skip dead cells
@group(1) @binding(6)
var<storage, read> death_flags: array<u32>;

// Phagocyte cell type constant
const PHAGOCYTE_TYPE: u32 = 2u;

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

// Check if voxel has a nutrient (atomically read)
fn has_nutrient(voxel_index: u32) -> bool {
    if (voxel_index == 0xFFFFFFFFu) {
        return false;
    }
    return atomicLoad(&nutrient_voxels[voxel_index]) == 1u;
}

// Try to consume nutrient from voxel (atomic compare-exchange)
// Returns true if successfully consumed
// Sets voxel to 2 (consumed/depleted) instead of 0 (empty) so the populate
// shader won't immediately refill it — the nutrient stays gone until the
// noise pattern drifts away and resets the voxel back to 0.
fn try_consume_nutrient(voxel_index: u32) -> bool {
    if (voxel_index == 0xFFFFFFFFu) {
        return false;
    }
    // Try to atomically change from 1 (has nutrient) to 2 (consumed/depleted)
    let result = atomicCompareExchangeWeak(&nutrient_voxels[voxel_index], 1u, 2u);
    return result.exchanged;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Only phagocytes can consume nutrients
    let cell_type = cell_types[cell_idx];
    if (cell_type != PHAGOCYTE_TYPE) {
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
    
    // Nutrient cap: 2x split_nutrient_threshold.
    // Cap at 200 before doubling so the "never split" sentinel (threshold > 100)
    // doesn't inflate the cap to an absurd value.
    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;
    if (current_nutrients >= max_nutrients) {
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
        // Successfully consumed nutrient - add nutrients via atomicAdd, NOT atomicStore.
        //
        // Using atomicStore here would be a race: we read current_nutrients at the top of
        // the shader, then nutrient_transport (which runs in the same frame) may have
        // already done atomicAdd writes to this cell's slot. An atomicStore would silently
        // overwrite those writes, permanently destroying transported nutrients.
        //
        // Instead, clamp the gain so we don't exceed max_nutrients. We compute the
        // headroom from the stale read — this is a slight over-estimate if transport
        // added nutrients between the read and now, but the worst case is a tiny
        // overshoot that the next frame's cap check will correct. That is far better
        // than the previous behaviour of discarding all transport income.
        let nutrient_gain = nutrient_params.mass_per_nutrient * 100.0;
        let headroom = max(max_nutrients - current_nutrients, 0.0);
        let clamped_gain = min(nutrient_gain, headroom);
        if (clamped_gain > 0.0) {
            atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(clamped_gain));
        }
    }
}
