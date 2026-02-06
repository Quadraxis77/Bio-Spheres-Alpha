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
// Matches adhesion_layout: binding 0 = connections, 1 = settings, 2 = counts, 3 = cell_adhesion_indices
@group(2) @binding(0)
var<storage, read> adhesion_connections: array<AdhesionConnection>;

@group(2) @binding(1)
var<storage, read> adhesion_settings: array<vec4<f32>>;  // Not used but must match layout

@group(2) @binding(2)
var<storage, read> adhesion_counts: array<u32>;  // Not used but must match layout

@group(2) @binding(3)
var<storage, read> adhesion_indices: array<array<i32, 20>>;  // MAX_ADHESIONS_PER_CELL = 20

// Nutrient transport bind group (group 3) - mass deltas and mode properties
@group(3) @binding(0)
var<storage, read_write> mass_deltas: array<atomic<i32>>;

@group(3) @binding(1)
var<storage, read_write> death_flags: array<u32>;

@group(3) @binding(2)
var<storage, read> mode_properties: array<ModeProperties>;

@group(3) @binding(3)
var<storage, read> split_ready_frame: array<i32>;

@group(3) @binding(4)
var<storage, read> mode_cell_types: array<u32>;

// Adhesion connection structure (exactly 104 bytes, matching Rust GpuAdhesionConnection)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _align_pad: vec2<u32>,            // Padding to align anchor_direction_a
    anchor_direction_a: vec4<f32>,    // xyz = direction, w = padding
    anchor_direction_b: vec4<f32>,    // xyz = direction, w = padding
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    _padding: vec2<u32>,              // Final padding to match 104 bytes
}

// Mode properties structure (48 bytes per mode)
// Layout: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, padding x3]
struct ModeProperties {
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    membrane_stiffness: f32,
    split_interval: f32,
    split_mass: f32,
    nutrient_priority: f32,
    swim_force: f32,
    prioritize_when_low: f32,  // 1.0 = true, 0.0 = false
    max_splits: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Constants matching reference implementation
const MIN_CELL_MASS: f32 = 0.5;
const DANGER_THRESHOLD: f32 = 0.6;
const PRIORITY_BOOST: f32 = 10.0;
const TRANSPORT_RATE: f32 = 0.5;
const BASE_METABOLISM_RATE: f32 = 0.05;  // Base metabolic cost per second for all cells
const SWIM_CONSUMPTION_RATE: f32 = 0.2;  // Additional 0.2 mass per second at full swim force
const DEFER_FRAMES: i32 = 32;  // 0.5 seconds at 64 FPS = 32 frames

// Fixed-point conversion for atomic operations (matching other shaders)
const FIXED_POINT_SCALE: f32 = 1000.0;

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

// Check if a cell should be blocked from nutrient transfer due to split attempt delay
fn is_cell_blocked_from_nutrients(cell_idx: u32) -> bool {
    let ready_frame = split_ready_frame[cell_idx];
    if (ready_frame < 0) {
        return false; // Cell is not ready to split
    }
    
    let frames_since_ready = params.current_frame - ready_frame;
    return frames_since_ready < DEFER_FRAMES;
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
    
    // NOTE: mass_deltas is cleared by CPU before this shader runs (encoder.clear_buffer)
    // This prevents a race condition where atomicStore would overwrite transfers from other cells
    
    // Read current mass from positions buffer
    let current_mass = positions_out[cell_idx].w;
    
    // Step 1: Nutrient consumption - all cells except Test cells have base metabolism
    let mode_idx = mode_indices[cell_idx];

    // Get cell type from mode
    var cell_type = 0u;
    if (mode_idx < arrayLength(&mode_cell_types)) {
        cell_type = mode_cell_types[mode_idx];
    }

    // Test cells (cell_type 0) have no metabolism - they auto-gain from nutrient_gain_rate
    // All other cells have base metabolism
    if (cell_type != 0u && mode_idx < arrayLength(&mode_properties)) {
        let mode = mode_properties[mode_idx];

        // Base metabolism: consume nutrients to stay alive
        var mass_loss = BASE_METABOLISM_RATE * params.delta_time;

        // Additional consumption from swim force (Flagellocytes only)
        if (mode.swim_force > 0.0) {
            mass_loss += mode.swim_force * SWIM_CONSUMPTION_RATE * params.delta_time;
        }

        let consumption_fixed = float_to_fixed(-mass_loss);
        atomicAdd(&mass_deltas[cell_idx], consumption_fixed);
    }
    
    // NOTE: No early death check here - we must let transport happen first
    // so starving cells can receive nutrients from neighbors before dying
    
    // Step 2: Nutrient transport between adhesion-connected cells
    // Process adhesions where this cell is cell_a (to avoid double processing)
    let adhesion_list = adhesion_indices[cell_idx];
    
    for (var i = 0; i < 20; i++) {  // MAX_ADHESIONS_PER_CELL = 20
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
        
        // Check if either cell should be blocked from nutrient transfer due to split attempt delay
        if (is_cell_blocked_from_nutrients(cell_idx) || is_cell_blocked_from_nutrients(cell_b_idx)) {
            continue; // Skip nutrient transfer for blocked cells
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
    
    // NOTE: Do NOT set death_flags here - lifecycle death_scan checks mass and handles it
    // This ensures proper ring buffer slot recycling
    
    // Update mass in positions buffer
    let pos = positions_out[cell_idx].xyz;
    positions_out[cell_idx] = vec4<f32>(pos, max(final_mass, 0.0));
}