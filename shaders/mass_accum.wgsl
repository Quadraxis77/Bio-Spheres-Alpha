// Mass Accumulation Shader
// Phase 7: Energy & Mass Transfer - mass_accum stage
// Workgroup size: 256 threads for optimal GPU occupancy
//
// This shader increases cell mass over time based on the nutrient_gain_rate
// from the cell's genome mode. This is the proper implementation that reads
// per-cell nutrient gain rates rather than using a hardcoded value.
//
// Mass growth formula:
//   new_mass = old_mass + nutrient_gain_rate * delta_time
//
// The nutrient_gain_rate is set per-cell from the genome mode's nutrient_gain_rate
// field (default: 0.2 mass per second).

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Read current position and mass from output buffer (after physics integration)
    let pos_mass = positions_out[cell_idx];
    let current_mass = pos_mass.w;
    
    // Read per-cell nutrient gain rate from genome mode
    let nutrient_gain_rate = nutrient_gain_rates[cell_idx];
    
    // Calculate mass increase: mass += nutrient_gain_rate * delta_time
    let mass_increase = nutrient_gain_rate * params.delta_time;
    let new_mass = current_mass + mass_increase;
    
    // Write updated mass back (position unchanged)
    positions_out[cell_idx] = vec4<f32>(pos_mass.xyz, new_mass);
}
