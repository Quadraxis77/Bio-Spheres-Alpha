// Nutrient Transport Shader
// Phase 6: Chemical & Nutrient Systems - nutrient transport stage
// Workgroup size: 256 threads for optimal GPU occupancy
//
// This shader implements the nutrient distribution system from BioSpheres-Q reference:
// 1. Nutrient consumption for active cells (Flagellocytes with swim force)
// 2. Nutrient transport between adhesion-connected cells based on priority ratios
// 3. Cell death detection from starvation (mass < 0.5 threshold)
//
// Transport Algorithm:
// - Nutrients flow to establish equilibrium: mass_a / mass_b = priority_a / priority_b
// - Flow is driven by "pressure" differences: pressure = mass / priority
// - Cells with low mass get temporary priority boost (10x) when below danger threshold (0.6)
// - Transport rate: 0.5 (matches reference implementation)

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

// Nutrient system bind group (group 1)
@group(1) @binding(0)
var<storage, read> nutrient_gain_rates: array<f32>;

@group(1) @binding(1)
var<storage, read> max_cell_sizes: array<f32>;

@group(1) @binding(2)
var<storage, read> mode_indices: array<u32>;

@group(1) @binding(3)
var<storage, read> genome_ids: array<u32>;

// Adhesion system bind group (group 2)
@group(2) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(2) @binding(1)
var<storage, read> adhesion_indices: array<array<i32, 20>>;

// Nutrient transport bind group (group 3) - mass deltas and mode properties
@group(3) @binding(0)
var<storage, read_write> mass_deltas: array<atomic<i32>>;

@group(3) @binding(1)
var<storage, read_write> death_flags: array<u32>;

@group(3) @binding(2)
var<storage, read> mode_properties: array<ModeProperties>;

// Adhesion connection structure (exactly 96 bytes, matching reference)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    anchor_direction_a: vec3<f32>,
    _pad_a: f32,
    anchor_direction_b: vec3<f32>,
    _pad_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
}

// Mode properties structure (32 bytes per mode)
struct ModeProperties {
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    membrane_stiffness: f32,
    split_interval: f32,
    split_mass: f32,
    nutrient_priority: f32,
    swim_force: f32,
    prioritize_when_low: f32,  // 1.0 = true, 0.0 = false
}

// Constants matching reference implementation
const MIN_CELL_MASS: f32 = 0.5;
const DANGER_THRESHOLD: f32 = 0.6;
const PRIORITY_BOOST: f32 = 10.0;
const TRANSPORT_RATE: f32 = 0.5;
const SWIM_CONSUMPTION_RATE: f32 = 0.2;  // 0.2 mass per second at full swim force

// Fixed-point conversion for atomic operations (matching other shaders)
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
    
    // Initialize mass delta to 0 (atomic)
    atomicStore(&mass_deltas[cell_idx], 0);
    
    // Read current mass from positions buffer
    let current_mass = positions_out[cell_idx].w;
    
    // Step 1: Nutrient consumption for Flagellocytes with swim force
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx < arrayLength(&mode_properties)) {
        let mode = mode_properties[mode_idx];
        
        // Consume mass proportional to swim force (Flagellocytes only)
        if (mode.swim_force > 0.0) {
            let mass_loss = mode.swim_force * SWIM_CONSUMPTION_RATE * params.delta_time;
            let consumption_fixed = float_to_fixed(-mass_loss);
            atomicAdd(&mass_deltas[cell_idx], consumption_fixed);
        }
    }
    
    // Step 2: Check for cell death from starvation (preliminary check)
    // We'll do a final check after transport
    let preliminary_mass_delta = fixed_to_float(atomicLoad(&mass_deltas[cell_idx]));
    let preliminary_final_mass = current_mass + preliminary_mass_delta;
    
    if (preliminary_final_mass < MIN_CELL_MASS) {
        death_flags[cell_idx] = 1u;  // Mark for death
        // CRITICAL: Write the reduced mass before returning so lifecycle_death_scan sees it
        let pos = positions_out[cell_idx].xyz;
        positions_out[cell_idx] = vec4<f32>(pos, max(preliminary_final_mass, 0.0));
        return;  // Skip nutrient transport for dying cells
    } else {
        death_flags[cell_idx] = 0u;  // Still alive (preliminary)
    }
    
    // Step 3: Nutrient transport between adhesion-connected cells
    // Process adhesions where this cell is cell_a (to avoid double processing)
    let adhesion_list = adhesion_indices[cell_idx];
    
    for (var i = 0; i < 20; i++) {
        let adhesion_idx = adhesion_list[i];
        if (adhesion_idx < 0 || adhesion_idx >= i32(arrayLength(&adhesion_connections))) {
            continue;
        }
        
        let adhesion = adhesion_connections[adhesion_idx];
        if (adhesion.is_active == 0u) {
            continue;
        }
        
        // Only process if this cell is cell_a (avoid double processing)
        if (adhesion.cell_a_index != cell_idx) {
            continue;
        }
        
        let cell_b_idx = adhesion.cell_b_index;
        if (cell_b_idx >= cell_count) {
            continue;
        }
        
        // Get masses (use original mass for both cells since we'll apply deltas atomically)
        let mass_a = current_mass;
        let mass_b = positions_out[cell_b_idx].w;
        
        // Get mode properties for both cells
        let mode_a_idx = mode_indices[cell_idx];
        let mode_b_idx = mode_indices[cell_b_idx];
        
        if (mode_a_idx >= arrayLength(&mode_properties) || 
            mode_b_idx >= arrayLength(&mode_properties)) {
            continue;
        }
        
        let mode_a = mode_properties[mode_a_idx];
        let mode_b = mode_properties[mode_b_idx];
        
        // Get base priorities
        let base_priority_a = mode_a.nutrient_priority;
        let base_priority_b = mode_b.nutrient_priority;
        
        // Apply temporary priority boost when cells are dangerously low on nutrients
        let prioritize_a = mode_a.prioritize_when_low > 0.5;
        let prioritize_b = mode_b.prioritize_when_low > 0.5;
        
        let priority_a = select(base_priority_a, base_priority_a * PRIORITY_BOOST, 
                               prioritize_a && mass_a < DANGER_THRESHOLD);
        let priority_b = select(base_priority_b, base_priority_b * PRIORITY_BOOST,
                               prioritize_b && mass_b < DANGER_THRESHOLD);
        
        // Calculate equilibrium-based nutrient flow
        // At equilibrium: mass_a / mass_b = priority_a / priority_b
        // Flow is driven by "pressure" difference: pressure = mass / priority
        let pressure_a = mass_a / priority_a;
        let pressure_b = mass_b / priority_b;
        let pressure_diff = pressure_a - pressure_b;
        
        // Calculate mass transfer (positive = A -> B, negative = B -> A)
        let mass_transfer = pressure_diff * TRANSPORT_RATE * params.delta_time;
        
        // Apply transfer with minimum thresholds
        let min_mass_a = select(0.0, 0.1, prioritize_a);
        let min_mass_b = select(0.0, 0.1, prioritize_b);
        
        let actual_transfer = select(
            max(mass_transfer, -(mass_b - min_mass_b)),  // B -> A: limit by B's mass
            min(mass_transfer, mass_a - min_mass_a),     // A -> B: limit by A's mass
            mass_transfer > 0.0
        );
        
        // Apply transfer using atomic operations (thread-safe)
        let transfer_fixed = float_to_fixed(actual_transfer);
        
        // Atomically subtract from cell A and add to cell B
        atomicAdd(&mass_deltas[cell_idx], -transfer_fixed);
        atomicAdd(&mass_deltas[cell_b_idx], transfer_fixed);
    }
    
    // Step 4: Apply consumption and transport results
    // Read final accumulated mass delta from atomic buffer
    let final_mass_delta = fixed_to_float(atomicLoad(&mass_deltas[cell_idx]));
    let final_mass = current_mass + final_mass_delta;
    
    // Final death check after transport
    if (final_mass < MIN_CELL_MASS) {
        death_flags[cell_idx] = 1u;
    }
    
    // Update mass in positions buffer
    let pos = positions_out[cell_idx].xyz;
    positions_out[cell_idx] = vec4<f32>(pos, max(final_mass, 0.0));
}