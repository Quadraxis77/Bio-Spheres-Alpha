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
// - Transport rate: 20.0 nutrients/sec

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
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

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

// Nutrients buffer: fixed-point i32 with scale 1000 (100.0 nutrients = 100000)
@group(3) @binding(5)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Split nutrient thresholds (derived from split_mass: threshold = (split_mass - 1.0) * 100.0)
@group(3) @binding(6)
var<storage, read> split_nutrient_thresholds: array<f32>;

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
    birth_time: f32,                  // offset 96-99
    _pad: u32,                        // offset 100-103
}

// Mode properties structure (80 bytes per mode = 20 floats)
struct ModeProperties {
    nutrient_gain_rate: f32,        // index 0
    max_cell_size: f32,             // index 1
    membrane_stiffness: f32,        // index 2
    split_interval: f32,            // index 3
    split_mass: f32,                // index 4
    nutrient_priority: f32,         // index 5
    swim_force: f32,                // index 6
    prioritize_when_low: f32,       // index 7: 1.0 = true, 0.0 = false
    max_splits: f32,                // index 8
    split_ratio: f32,               // index 9
    flagellocyte_signal_channel: f32, // index 10
    flagellocyte_speed_a: f32,      // index 11
    flagellocyte_speed_b: f32,      // index 12
    flagellocyte_threshold_c: f32,  // index 13
    flagellocyte_use_signal: f32,   // index 14: 1.0 = signal mode, 0.0 = fixed mode
    min_adhesions: f32,             // index 15
    max_adhesions: f32,             // index 16
    _pad17: f32,                    // index 17
    _pad18: f32,                    // index 18
    _pad19: f32,                    // index 19
}

// Constants matching reference implementation
const MIN_CELL_MASS: f32 = 0.5;
const MIN_NUTRIENTS: f32 = 1.0;  // Death threshold: nutrients < 1.0
const DANGER_THRESHOLD: f32 = 0.6;
const DANGER_NUTRIENTS: f32 = 60.0;  // 0.6 * 100 = 60 nutrients danger threshold
const PRIORITY_BOOST: f32 = 10.0;
const TRANSPORT_RATE: f32 = 20.0;  // Max nutrients/sec total outflow per cell across ALL connections
const LERP_SPEED: f32 = 999.0;     // Smoothing factor (per second); lower = smoother, less oscillation
const BASE_METABOLISM_RATE: f32 = 1.0;  // Base metabolic cost in nutrients/sec for non-auto-gain cells
const SWIM_CONSUMPTION_RATE: f32 = 2.0;  // 2 nutrients/sec at swim_force=1, 6/sec at swim_force=3
const DEFER_FRAMES: i32 = 6;  // ~0.1 seconds at 64 FPS = 6 frames

// Fixed-point conversion for atomic operations (matching other shaders)
const FIXED_POINT_SCALE: f32 = 1000.0;

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

// Derive mass from nutrients: mass = 1.0 + nutrients / 100.0
fn nutrients_to_mass(nutrients: f32) -> f32 {
    return 1.0 + nutrients / 100.0;
}

// Derive nutrients from mass: nutrients = (mass - 1.0) * 100.0
fn mass_to_nutrients(mass: f32) -> f32 {
    return (mass - 1.0) * 100.0;
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
    
    // Read current nutrients from nutrients_buffer (fixed-point i32)
    let current_nutrients_fixed = atomicLoad(&nutrients_buffer[cell_idx]);
    let current_nutrients = fixed_to_float(current_nutrients_fixed);
    
    // Step 1: Nutrient consumption - all cells except Test cells have base metabolism
    let mode_idx = mode_indices[cell_idx];

    // Get cell type from mode
    var cell_type = 0u;
    if (mode_idx < arrayLength(&mode_cell_types)) {
        cell_type = mode_cell_types[mode_idx];
    }

    // Test cells (cell_type 0) have no metabolism - they auto-gain from nutrient_gain_rate
    // Phagocytes (cell_type 2) and Photocytes (cell_type 3) use specialized shaders for gain
    // but still need base metabolism to starve when not consuming/absorbing
    // All other cells have base metabolism
    let auto_gain_cell = cell_type == 0u;
    if (!auto_gain_cell && mode_idx < arrayLength(&mode_properties)) {
        let mode = mode_properties[mode_idx];

        // Base metabolism: consume nutrients to stay alive (1.0 nutrients/sec)
        var nutrient_loss = BASE_METABOLISM_RATE * params.delta_time;

        // Additional consumption from swim force (Flagellocytes only)
        var effective_swim_speed = 0.0;
        if (mode.flagellocyte_use_signal >= 0.5) {
            // Signal-based: use speed_a as the active speed (conservative estimate for consumption)
            effective_swim_speed = max(mode.flagellocyte_speed_a, mode.flagellocyte_speed_b);
        } else {
            effective_swim_speed = mode.swim_force;
        }
        if (effective_swim_speed > 0.0) {
            nutrient_loss += effective_swim_speed * SWIM_CONSUMPTION_RATE * params.delta_time;
        }

        // Apply nutrient consumption directly to nutrients_buffer
        let new_nutrients = max(current_nutrients - nutrient_loss, 0.0);
        atomicStore(&nutrients_buffer[cell_idx], float_to_fixed(new_nutrients));
    }
    
    // NOTE: No early death check here - we must let transport happen first
    // so starving cells can receive nutrients from neighbors before dying
    
    // Step 2: Nutrient transport between adhesion-connected cells.
    // Total outflow is capped at TRANSPORT_RATE nutrients/sec across ALL connections.
    // Transfer is lerp-smoothed to prevent oscillation.
    //
    // Strategy: count active outgoing connections first, then divide TRANSPORT_RATE
    // equally among them so the total never exceeds the cap.
    let adhesion_list = adhesion_indices[cell_idx];
    let mode_a_idx = mode_indices[cell_idx];

    // Count active connections where this cell is cell_a (we only process those)
    var active_conn_count = 0u;
    for (var i = 0; i < 20; i++) {
        let adhesion_idx = adhesion_list[i];
        if (adhesion_idx < 0 || adhesion_idx >= i32(arrayLength(&adhesion_connections))) {
            continue;
        }
        let adhesion = adhesion_connections[adhesion_idx];
        if (adhesion.is_active == 0u || adhesion.cell_a_index != cell_idx) {
            continue;
        }
        let cell_b_idx = adhesion.cell_b_index;
        if (cell_b_idx >= cell_count || death_flags[cell_b_idx] == 1u) {
            continue;
        }
        // Only block the sending cell (cell_a = cell_idx) if it is split-deferred.
        // Never block a cell from *receiving* — a starving receiver must always be reachable.
        if (is_cell_blocked_from_nutrients(cell_idx)) {
            continue;
        }
        active_conn_count++;
    }

    // Rate per connection: TRANSPORT_RATE / conn_count so total stays bounded
    let rate_per_conn = select(TRANSPORT_RATE, TRANSPORT_RATE / f32(active_conn_count), active_conn_count > 1u);
    // Lerp factor: fraction of the gap to close this step
    let lerp_t = min(LERP_SPEED * params.delta_time, 1.0);

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
        
        // Only block the sending cell (cell_a = cell_idx) if it is split-deferred.
        // cell_b may be split-deferred and still legitimately receive nutrients.
        if (is_cell_blocked_from_nutrients(cell_idx)) {
            continue;
        }

        if (death_flags[cell_b_idx] == 1u) {
            continue;
        }
        
        // Get nutrients for both cells (use atomic load for thread-safety)
        let nutrients_a = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
        let nutrients_b = fixed_to_float(atomicLoad(&nutrients_buffer[cell_b_idx]));
        
        // Get mode properties for both cells
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
                               prioritize_a && nutrients_a < DANGER_NUTRIENTS);
        let priority_b = select(base_priority_b, base_priority_b * PRIORITY_BOOST,
                               prioritize_b && nutrients_b < DANGER_NUTRIENTS);
        
        // Pressure difference drives flow: clamp to [-rate_per_conn, rate_per_conn] nutrients/sec.
        // When priorities are equal, any nonzero difference drives full flow up to the cap.
        let pressure_diff = nutrients_a / priority_a - nutrients_b / priority_b;
        let desired_rate = clamp(pressure_diff, -rate_per_conn, rate_per_conn);

        // Lerp-smoothed transfer this timestep
        let nutrient_transfer = desired_rate * lerp_t * params.delta_time;
        
        // Apply transfer with min/max thresholds
        let min_nutrients_a = select(0.0, 10.0, prioritize_a);
        let min_nutrients_b = select(0.0, 10.0, prioritize_b);
        let max_nutrients_a = split_nutrient_thresholds[cell_idx] * 2.0;
        let max_nutrients_b = split_nutrient_thresholds[cell_b_idx] * 2.0;

        var actual_transfer: f32;
        if (nutrient_transfer > 0.0) {
            let can_give = max(nutrients_a - min_nutrients_a, 0.0);
            let can_recv = max(max_nutrients_b - nutrients_b, 0.0);
            actual_transfer = min(nutrient_transfer, min(can_give, can_recv));
        } else {
            let can_give = max(nutrients_b - min_nutrients_b, 0.0);
            let can_recv = max(max_nutrients_a - nutrients_a, 0.0);
            actual_transfer = max(nutrient_transfer, -min(can_give, can_recv));
        }
        
        // Apply transfer using atomic operations (thread-safe)
        let transfer_fixed = float_to_fixed(actual_transfer);
        
        // Atomically subtract from cell A and add to cell B
        atomicAdd(&nutrients_buffer[cell_idx], -transfer_fixed);
        atomicAdd(&nutrients_buffer[cell_b_idx], transfer_fixed);
    }
    
    // NOTE: Death detection is handled by a separate shader that checks nutrients < MIN_NUTRIENTS
    // and sets death_flags accordingly.
}