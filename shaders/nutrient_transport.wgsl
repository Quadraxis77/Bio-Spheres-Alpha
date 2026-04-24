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
// - Cells with low nutrients get temporary priority boost (10x) when below 10.0 nutrients
// - Transport rate: 30.0 nutrients/sec

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
    solo_metabolism_multiplier: f32,
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
// Matches adhesion_layout: 0=connections, 1=settings_v0, 2=settings_v1, 3=settings_v2, 4=counts, 5=cell_adhesion_indices
@group(2) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(2) @binding(1)
var<storage, read> adhesion_settings_v0: array<vec4<f32>>;  // Not used but must match layout

@group(2) @binding(2)
var<storage, read> adhesion_settings_v1: array<vec4<f32>>;  // Not used but must match layout

@group(2) @binding(3)
var<storage, read> adhesion_settings_v2: array<vec4<f32>>;  // Not used but must match layout

@group(2) @binding(4)
var<storage, read> adhesion_counts: array<u32>;  // Not used but must match layout

@group(2) @binding(5)
var<storage, read> adhesion_indices: array<array<i32, 20>>;  // MAX_ADHESIONS_PER_CELL = 20

// Nutrient transport bind group (group 3) - mass deltas and mode properties
@group(3) @binding(0)
var<storage, read_write> mass_deltas: array<atomic<i32>>;

@group(3) @binding(1)
var<storage, read_write> death_flags: array<u32>;

// mode_properties split into 5 vec4 sub-buffers (bindings 2-6)
@group(3) @binding(2)
var<storage, read> mode_properties_v0: array<vec4<f32>>; // [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
@group(3) @binding(3)
var<storage, read> mode_properties_v1: array<vec4<f32>>; // [split_mass, nutrient_priority, swim_force, prioritize_when_low]
@group(3) @binding(4)
var<storage, read> mode_properties_v2: array<vec4<f32>>; // [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
@group(3) @binding(5)
var<storage, read> mode_properties_v3: array<vec4<f32>>; // [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
@group(3) @binding(6)
var<storage, read> mode_properties_v4: array<vec4<f32>>; // [max_adhesions, mode_a_after_splits, mode_b_after_splits, padding]

@group(3) @binding(7)
var<storage, read> split_ready_frame: array<i32>;

@group(3) @binding(8)
var<storage, read> mode_cell_types: array<u32>;

// Nutrients buffer: fixed-point i32 with scale 1000 (100.0 nutrients = 100000)
@group(3) @binding(9)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Split nutrient thresholds (derived from split_mass: threshold = (split_mass - 1.0) * 100.0)
@group(3) @binding(10)
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

// Constants matching reference implementation
const MIN_CELL_MASS: f32 = 0.5;
const MIN_NUTRIENTS: f32 = 1.0;  // Death threshold: nutrients < 1.0
const DANGER_NUTRIENTS: f32 = 10.0;  // Priority boost threshold in nutrients (matches preview)
const PRIORITY_BOOST: f32 = 10.0;
const TRANSPORT_RATE: f32 = 30.0;  // Max nutrients/sec total outflow per cell across ALL connections
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
    // IMPORTANT: Cells with invalid mode indices (out of bounds) should NOT default to Test cell behavior.
    // They must have metabolism to prevent immortal grey cells from mutations.
    var cell_type = 0xFFFFFFFFu; // Invalid marker
    var mode_valid = false;
    if (mode_idx < arrayLength(&mode_cell_types)) {
        cell_type = mode_cell_types[mode_idx];
        mode_valid = true;
    }

    // Test cells (cell_type 0) have no metabolism - they auto-gain from nutrient_gain_rate
    // Phagocytes (cell_type 2) and Photocytes (cell_type 3) use specialized shaders for gain
    // but still need base metabolism to starve when not consuming/absorbing
    // All other cells (including invalid mode cells) have base metabolism
    let auto_gain_cell = mode_valid && cell_type == 0u;
    if (!auto_gain_cell && mode_idx < arrayLength(&mode_properties_v0)) {
        let mode_v1 = mode_properties_v1[mode_idx];

        // Base metabolism: consume nutrients to stay alive (1.0 nutrients/sec)
        let mode_v3 = mode_properties_v3[mode_idx];
        var nutrient_loss = BASE_METABOLISM_RATE * params.delta_time;

        // Additional consumption from swim force (Flagellocytes only)
        let swim_force = mode_v1.z; // mode_properties_v1.z = swim_force
        let flagellocyte_use_signal = mode_v3.z; // v3.z = flagellocyte_use_signal
        var effective_swim_speed = 0.0;
        if (flagellocyte_use_signal > 0.5) {
            // Signal-based: use max of speed_a and speed_b for conservative consumption estimate
            effective_swim_speed = max(mode_properties_v2[mode_idx].w, mode_properties_v3[mode_idx].x);
        } else {
            effective_swim_speed = swim_force;
        }
        if (effective_swim_speed > 0.0) {
            nutrient_loss += effective_swim_speed * SWIM_CONSUMPTION_RATE * params.delta_time;
        }

        // Solo cell metabolism penalty: cells with fewer adhesion connections burn
        // nutrients faster. Multiplier interpolates from solo_metabolism_multiplier
        // (at 0 connections) to 1.0 (at 3+ connections).
        // When solo_metabolism_multiplier == 1.0, the feature is effectively disabled.
        if (params.solo_metabolism_multiplier > 1.0) {
            let adhesion_list = adhesion_indices[cell_idx];
            var active_adhesion_count = 0u;
            for (var j = 0; j < 20; j++) {
                let adh_idx = adhesion_list[j];
                if (adh_idx >= 0 && adh_idx < i32(arrayLength(&adhesion_connections))) {
                    if (adhesion_connections[adh_idx].is_active != 0u) {
                        active_adhesion_count++;
                    }
                }
            }
            // Gradient: 0 connections = full multiplier, 1-2 = partial, 3+ = no penalty
            let t = min(f32(active_adhesion_count), 3.0) / 3.0;
            let metabolism_scale = mix(params.solo_metabolism_multiplier, 1.0, t);
            nutrient_loss *= metabolism_scale;
        }

        // Apply nutrient consumption as an atomic delta (NOT atomicStore!)
        // atomicStore here would race with atomicAdd from neighbor transport threads,
        // overwriting any nutrients transferred to this cell during the same dispatch.
        let safe_loss = min(nutrient_loss, max(current_nutrients, 0.0));
        atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(safe_loss));
    }
    
    // NOTE: No early death check here - we must let transport happen first
    // so starving cells can receive nutrients from neighbors before dying
    
    // Step 2: Nutrient transport between adhesion-connected cells.
    // Total outflow is capped at TRANSPORT_RATE nutrients/sec across ALL connections.
    // Transfer is lerp-smoothed to prevent oscillation.
    //
    // Two-pass approach (matches preview):
    // Pass 1: compute desired flow per connection, sum total outflow for this cell
    // Pass 2: proportionally scale flows so total stays within TRANSPORT_RATE, then apply
    let adhesion_list = adhesion_indices[cell_idx];
    let mode_a_idx = mode_indices[cell_idx];

    // Snapshot nutrients for this cell once (matches preview's single-snapshot semantics)
    let nutrients_a_snap = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));

    // Pass 1: compute desired flows and accumulate total outflow for this cell
    var total_out: f32 = 0.0;
    var desired_arr: array<f32, 20>;
    var cellb_arr: array<u32, 20>;
    var nutrb_arr: array<f32, 20>;
    var prioa_arr: array<u32, 20>;
    var priob_arr: array<u32, 20>;
    var valid_arr: array<u32, 20>;

    for (var i = 0; i < 20; i++) {
        valid_arr[i] = 0u;
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

        let mode_b_idx = mode_indices[cell_b_idx];
        if (mode_a_idx >= arrayLength(&mode_properties_v0) ||
            mode_b_idx >= arrayLength(&mode_properties_v0)) {
            continue;
        }

        let nutrients_b = fixed_to_float(atomicLoad(&nutrients_buffer[cell_b_idx]));
        let mode_a_v1 = mode_properties_v1[mode_a_idx];
        let mode_b_v1 = mode_properties_v1[mode_b_idx];
        // v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
        let prioritize_a = mode_a_v1.w > 0.5; // prioritize_when_low
        let prioritize_b = mode_b_v1.w > 0.5; // prioritize_when_low

        let priority_a = select(mode_a_v1.y, mode_a_v1.y * PRIORITY_BOOST,
                               prioritize_a && nutrients_a_snap < DANGER_NUTRIENTS);
        let priority_b = select(mode_b_v1.y, mode_b_v1.y * PRIORITY_BOOST,
                               prioritize_b && nutrients_b < DANGER_NUTRIENTS);

        let pressure_diff = nutrients_a_snap / priority_a - nutrients_b / priority_b;
        let desired = clamp(pressure_diff, -TRANSPORT_RATE, TRANSPORT_RATE);

        desired_arr[i] = desired;
        cellb_arr[i] = cell_b_idx;
        nutrb_arr[i] = nutrients_b;
        prioa_arr[i] = select(0u, 1u, prioritize_a);
        priob_arr[i] = select(0u, 1u, prioritize_b);
        valid_arr[i] = 1u;

        if (desired > 0.0) {
            total_out += desired;
        }
    }

    // Pass 2: apply proportionally scaled transfers (matches preview)
    let lerp_t = min(LERP_SPEED * params.delta_time, 1.0);

    for (var i = 0; i < 20; i++) {
        if (valid_arr[i] == 0u) {
            continue;
        }

        let desired = desired_arr[i];
        let cell_b_idx = cellb_arr[i];
        let nutrients_b = nutrb_arr[i];
        let prioritize_a = prioa_arr[i] == 1u;
        let prioritize_b = priob_arr[i] == 1u;

        // Scale outflow so the sending cell never exceeds TRANSPORT_RATE total.
        // Proportionally reduces each connection's flow based on total desired outflow.
        // For negative flows (cell_b sends), we cannot compute cell_b's total_out
        // on GPU in a single pass, so we don't scale (matches preview when total_out <= TRANSPORT_RATE).
        var scale: f32 = 1.0;
        if (desired > 0.0 && total_out > TRANSPORT_RATE) {
            scale = TRANSPORT_RATE / total_out;
        }

        // Target transfer this step (nutrients), lerped toward desired
        let target_per_sec = desired * scale;
        let nutrient_transfer = target_per_sec * lerp_t * params.delta_time;

        // Clamp by available nutrients and receiver capacity
        let min_nutrients_a = select(0.0, 10.0, prioritize_a);
        let min_nutrients_b = select(0.0, 10.0, prioritize_b);
        let max_nutrients_a = split_nutrient_thresholds[cell_idx] * 2.0;
        let max_nutrients_b = split_nutrient_thresholds[cell_b_idx] * 2.0;

        var actual_transfer: f32;
        if (nutrient_transfer > 0.0) {
            let can_give = max(nutrients_a_snap - min_nutrients_a, 0.0);
            let can_recv = max(max_nutrients_b - nutrients_b, 0.0);
            actual_transfer = min(nutrient_transfer, min(can_give, can_recv));
        } else {
            let can_give = max(nutrients_b - min_nutrients_b, 0.0);
            let can_recv = max(max_nutrients_a - nutrients_a_snap, 0.0);
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