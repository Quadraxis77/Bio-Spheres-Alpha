// Mass Accumulation Shader
// Phase 6: Chemical & Nutrient Systems - mass_accum stage (nutrient growth only)
// Workgroup size: 256 threads for optimal GPU occupancy
//
// This shader handles ONLY nutrient growth based on nutrient_gain_rate.
// Nutrient consumption and transport are handled by the nutrient_transport shader.
//
// Mass growth formula:
//   new_mass = old_mass + nutrient_gain_rate * delta_time
//   capped at 2x split_mass (storage capacity for division)
//
// The nutrient_gain_rate is set per-cell from the genome mode's nutrient_gain_rate
// field (default: 0.2 mass per second for Test cells, 0.0 for others).

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

// Physics bind group (group 0) - standard 6-binding layout
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Mass accumulation bind group (group 1) - nutrient gain rates per cell
@group(1) @binding(0)
var<storage, read> nutrient_gain_rates: array<f32>;

// Split nutrient thresholds per cell (derived from split_mass)
@group(1) @binding(1)
var<storage, read> split_nutrient_thresholds: array<f32>;

// Death flags to skip dead cells
@group(1) @binding(2)
var<storage, read> death_flags: array<u32>;

// Cell types per mode
@group(1) @binding(3)
var<storage, read> mode_cell_types: array<u32>;

// Mode indices per cell
@group(1) @binding(4)
var<storage, read> mode_indices: array<u32>;

// Nutrients buffer: fixed-point i32 with scale 1000 (100.0 nutrients = 100000)
@group(1) @binding(5)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Skip dead cells - they're waiting to be recycled via ring buffer
    if (death_flags[cell_idx] == 1u) {
        return;
    }
    
    // Get cell type from mode
    let mode_idx = mode_indices[cell_idx];
    var cell_type = 0xFFFFFFFFu; // Invalid marker
    var mode_valid = false;
    if (mode_idx < arrayLength(&mode_cell_types)) {
        cell_type = mode_cell_types[mode_idx];
        mode_valid = true;
    }
    
    // Only Test cells (cell_type == 0) with valid mode indices auto-gain nutrients on GPU
    // Phagocytes and Photocytes use specialized shaders (phagocyte_consume, photocyte_light)
    // Cells with invalid mode indices should NOT auto-gain (prevents immortal grey cells)
    let auto_gain_cell = mode_valid && cell_type == 0u;
    if (!auto_gain_cell) {
        return;
    }
    
    // Read current nutrients from nutrients_buffer (fixed-point i32)
    let current_nutrients_fixed = atomicLoad(&nutrients_buffer[cell_idx]);
    let current_nutrients = fixed_to_float(current_nutrients_fixed);
    
    // Read per-cell nutrient gain rate and split threshold
    let nutrient_gain_rate = nutrient_gain_rates[cell_idx];
    let split_nutrient_threshold = split_nutrient_thresholds[cell_idx];
    
    // Nutrient cap: 2x split_nutrient_threshold (allows storage for division plus buffer)
    let max_nutrients = split_nutrient_threshold * 2.0;
    
    // Only grow if below nutrient cap
    if (current_nutrients < max_nutrients) {
        // Calculate nutrient increase: nutrients += nutrient_gain_rate * delta_time
        let nutrient_increase = nutrient_gain_rate * params.delta_time;
        // Cap the increase so we don't exceed max_nutrients
        let capped_increase = min(nutrient_increase, max(max_nutrients - current_nutrients, 0.0));
        
        // Use atomicAdd (not atomicStore) to avoid overwriting concurrent writes
        atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(capped_increase));
    }
}
