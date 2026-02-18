// Nutrient Apply Shader
// Phase 6: Chemical & Nutrient Systems - mass delta application stage
// Workgroup size: 256 threads for optimal GPU occupancy
//
// This shader applies the accumulated mass deltas from the nutrient_transport shader.
// It runs as a SEPARATE dispatch to guarantee all atomic accumulations from
// nutrient_transport have completed across ALL workgroups before any thread
// reads the final delta value.
//
// Without this separation, a race condition exists: cell_b's workgroup could
// read and apply its mass_delta before cell_a's workgroup writes the transport
// delta to mass_deltas[cell_b], causing asymmetric nutrient flow.

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

// Nutrient apply bind group (group 1)
@group(1) @binding(0)
var<storage, read_write> mass_deltas: array<atomic<i32>>;

@group(1) @binding(1)
var<storage, read> death_flags: array<u32>;

// Fixed-point conversion (matching nutrient_transport.wgsl)
const FIXED_POINT_SCALE: f32 = 1000.0;

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
    
    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }
    
    // Read final accumulated mass delta from atomic buffer
    // All nutrient_transport workgroups have completed by now (separate dispatch)
    let final_mass_delta = fixed_to_float(atomicLoad(&mass_deltas[cell_idx]));
    
    // Skip if no change
    if (final_mass_delta == 0.0) {
        return;
    }
    
    // Apply delta to mass
    let pos_mass = positions_out[cell_idx];
    let final_mass = pos_mass.w + final_mass_delta;
    
    // Update mass in positions buffer (clamp to >= 0)
    positions_out[cell_idx] = vec4<f32>(pos_mass.xyz, max(final_mass, 0.0));
}
