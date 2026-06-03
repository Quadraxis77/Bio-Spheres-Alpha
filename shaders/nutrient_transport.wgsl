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
//
// Vasculocyte transport rules (cell_type == 12):
// - Vascular-to-vascular connections use VASCULAR_TRANSPORT_RATE (5x normal) for
//   high-throughput conduit behaviour across large body structures.
// - Vasculocyte nutrient transport and exchange are separate mode settings.
// - Transport-enabled vasculocyte pairs use VASCULAR_TRANSPORT_RATE as internal pipes.
// - Exchange ports move nutrients between pipe and non-vascular tissue in both directions.
// - Sealed pipes have strongly reduced exchange with non-vascular neighbors.
// - Physical compression between any two cells boosts their shared connection's
//   transport rate by up to PUMP_AMPLIFICATION * compression_ratio. This amplifies
//   flow through cells that are being squeezed (e.g. Lipocytes compressed by Myocytes).
// - Myocytes (cell_type 9) act as directional nutrient heart valves. Their velocity
//   direction defines the pump axis. Connections in the forward direction get a
//   positive pressure bias (pushing nutrients downstream); backward connections get a
//   negative bias (valve effect, resisting backflow). Pump strength scales with speed,
//   so myocytes must stay in motion (via swim force or body dynamics) to sustain pumping.

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

// Embryocyte reserve buffer (one atomic<u32> per cell, capped at 65535).
// Incoming nutrients are redirected here for Embryocytes (cell_type == 10).
@group(3) @binding(11)
var<storage, read_write> embryocyte_reserves: array<atomic<u32>>;

// Mode properties v9: per-mode Embryocyte trigger params (not used here, kept for layout).
// [use_timer as f32, release_timer, use_threshold as f32, threshold_value as f32]
@group(3) @binding(12)
var<storage, read> mode_properties_v9: array<vec4<f32>>;

// Mode properties v12: per-mode Vasculocyte params.
// [nutrient_transport, nutrient_exchange, signal_transport, signal_exchange]
@group(3) @binding(13)
var<storage, read> mode_properties_v12: array<vec4<f32>>;

// Organism size buffer: organism_size_buffer[cell_i] == cell count for that organism.
// Populated each frame by the clear/accumulate/broadcast passes in organism_label.wgsl.
// Consumers index directly by cell_idx - no label lookup needed.
@group(3) @binding(14)
var<storage, read> organism_size_buffer: array<u32>;

struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    bond_flags: u32,
    _align_pad1: u32,                 // Padding to align anchor_direction_a
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
const TRANSPORT_RATE: f32 = 100.0;  // Max nutrients/sec total outflow per cell across ALL connections
                                    // Raised from 30 so non-vascular organisms can diffuse nutrients
                                    // through the body. Vasculocytes still get 5x (500/sec highway).
const LERP_SPEED: f32 = 999.0;     // Smoothing factor (per second); lower = smoother, less oscillation
const BASE_METABOLISM_RATE: f32 = 1.0;  // Base metabolic cost in nutrients/sec for non-auto-gain cells
const SWIM_CONSUMPTION_RATE: f32 = 2.0;  // 2 nutrients/sec at swim_force=1, 6/sec at swim_force=3
const DEFER_FRAMES: i32 = 6;  // ~0.1 seconds at 64 FPS = 6 frames

// Vasculocyte transport constants
const VASCULOCYTE_CELL_TYPE: u32 = 12u;
const VASCULAR_TRANSPORT_RATE: f32 = 500.0; // 5x normal: vascular-to-vascular highway rate
const VASCULAR_SEAL_FACTOR: f32 = 0.05;     // Only 5% of normal rate leaks through a sealed vasculocyte
const BOND_FLAG_BARRIER_BALL: u32 = 2u;
const PUMP_AMPLIFICATION: f32 = 8.0;        // Compression multiplier for myocyte-driven pumping

// Myocyte directional pump constants
// Myocytes act as nutrient heart valves: their velocity direction defines the pump axis.
// Connections aligned with the forward axis receive a positive pressure bias (pushing nutrients
// forward); connections on the backward side receive a negative bias (valve effect, resisting
// backflow). Pump strength scales linearly with speed, clamped to 1.0.
const MYOCYTE_CELL_TYPE: u32 = 9u;
const MYOCYTE_PUMP_FORCE: f32 = 20.0;      // Max pressure bias (nutrients) added per unit alignment
const MYOCYTE_PUMP_SPEED_SCALE: f32 = 3.0; // Speed units that map to full pump strength (1.0)

// Fixed-point conversion for atomic operations (matching other shaders)
const FIXED_POINT_SCALE: f32 = 1000.0;

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

// Radius from mass, matching collision_detection.wgsl
fn cell_radius(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Compression ratio for an adhesion connection [0, 1].
// Returns 0 when cells are at or beyond their natural separation,
// positive when they are squeezed closer together (e.g. by a myocyte).
fn adhesion_compression(pos_a: vec3<f32>, mass_a: f32, pos_b: vec3<f32>, mass_b: f32) -> f32 {
    let natural_dist = cell_radius(mass_a) + cell_radius(mass_b);
    let actual_dist  = length(pos_b - pos_a);
    return max(0.0, (natural_dist - actual_dist) / natural_dist);
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
    // Embryocytes (cell_type 10) skip ALL normal metabolism - energy comes only from reserve
    // All other cells (including invalid mode cells) have base metabolism
    let is_embryocyte = mode_valid && cell_type == 10u;
    let auto_gain_cell = mode_valid && cell_type == 0u;
    if (!auto_gain_cell && !is_embryocyte && mode_idx < arrayLength(&mode_properties_v0)) {
        let mode_v1 = mode_properties_v1[mode_idx];

        // Base metabolism: consume nutrients to stay alive (1.0 nutrients/sec)
        let mode_v3 = mode_properties_v3[mode_idx];
        var nutrient_loss = BASE_METABOLISM_RATE * params.delta_time;

        // Kleiber's Law metabolic discount: larger organisms burn fewer nutrients per cell.
        // Discount = 1 / size^0.25, capped so organisms above KLEIBER_CAP cells get
        // the maximum discount. Solo cells (size == 1) pay full rate.
        // Cap at 100 cells -> discount = 1/100^0.25 ~= 0.316 (saves ~68% per cell).
        const KLEIBER_CAP: f32 = 100.0;
        let org_size = f32(max(organism_size_buffer[cell_idx], 1u));
        let capped_size = min(org_size, KLEIBER_CAP);
        let kleiber_discount = 1.0 / pow(capped_size, 0.25);
        nutrient_loss *= kleiber_discount;

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

        // Reserve-first metabolism: burn reserve before touching regular nutrients.
        // Reserve is stored x1000 (fixed-point), so burn float_to_fixed(nutrient_loss)
        // milli-units per tick - exact, no truncation at sub-unit drain rates.
        let cur_reserve = atomicLoad(&embryocyte_reserves[cell_idx]);
        var remaining_loss = nutrient_loss;
        if (cur_reserve > 0u) {
            let loss_fixed = u32(float_to_fixed(nutrient_loss));
            let reserve_burned = min(loss_fixed, cur_reserve);
            atomicSub(&embryocyte_reserves[cell_idx], reserve_burned);
            remaining_loss = max(nutrient_loss - f32(reserve_burned) / FIXED_POINT_SCALE, 0.0);

            // Also convert reserve into the nutrient pool when nutrients are below the
            // split threshold. This lets cells with inherited reserve actually grow and
            // divide, not just survive longer. Convert at up to 20/sec, capped by
            // how much space is left below the split threshold.
            const RESERVE_TO_NUTRIENT_RATE: f32 = 20.0;
            let split_threshold = min(split_nutrient_thresholds[cell_idx], 200.0);
            let nutrient_headroom = max(split_threshold - current_nutrients, 0.0);
            let updated_reserve = atomicLoad(&embryocyte_reserves[cell_idx]);
            if (nutrient_headroom > 0.0 && updated_reserve > 0u) {
                let convert_amount = min(RESERVE_TO_NUTRIENT_RATE * params.delta_time, nutrient_headroom);
                let convert_fixed = u32(float_to_fixed(convert_amount));
                let actual_convert = min(convert_fixed, updated_reserve);
                atomicSub(&embryocyte_reserves[cell_idx], actual_convert);
                // Add converted nutrients (net of remaining metabolic loss)
                let nutrients_gained = f32(actual_convert) / FIXED_POINT_SCALE;
                let net_gain = nutrients_gained - remaining_loss;
                if (net_gain > 0.0) {
                    atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(net_gain));
                } else {
                    let safe_loss = min(-net_gain, max(current_nutrients, 0.0));
                    atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(safe_loss));
                }
                remaining_loss = 0.0; // already handled above
            }
        }

        // Apply nutrient consumption as an atomic delta (NOT atomicStore!)
        // atomicStore here would race with atomicAdd from neighbor transport threads,
        // overwriting any nutrients transferred to this cell during the same dispatch.
        if (remaining_loss > 0.0) {
            let safe_loss = min(remaining_loss, max(current_nutrients, 0.0));
            atomicAdd(&nutrients_buffer[cell_idx], -float_to_fixed(safe_loss));
        }
    }
    
    // NOTE: No early death check here - we must let transport happen first
    // so starving cells can receive nutrients from neighbors before dying
    
    // Step 2: Nutrient transport between adhesion-connected cells.
    // Total outflow is capped at TRANSPORT_RATE nutrients/sec across ALL connections.
    // Transfer is lerp-smoothed to prevent oscillation.
    //
    // Embryocyte rules (cell_type == 10):
    //   - Embryocytes NEVER send nutrients out -> skip all connections if cell_a is Embryocyte.
    //   - Incoming nutrients to an Embryocyte go to its reserve buffer, not nutrients_buffer.
    //
    // Two-pass approach (matches preview):
    // Pass 1: compute desired flow per connection, sum total outflow for this cell
    // Pass 2: proportionally scale flows so total stays within TRANSPORT_RATE, then apply
    let adhesion_list = adhesion_indices[cell_idx];
    let mode_a_idx = mode_indices[cell_idx];

    // Embryocytes never send - skip the entire transport loop for them.
    if (is_embryocyte) {
        return;
    }

    // Snapshot nutrients for this cell once (matches preview's single-snapshot semantics)
    let nutrients_a_snap = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));

    // Determine if cell_a is a vasculocyte and whether it is an outlet.
    let cell_a_is_vascular = mode_valid && cell_type == VASCULOCYTE_CELL_TYPE;
    var cell_a_nutrient_transport = false;
    var cell_a_nutrient_exchange = false;
    if (cell_a_is_vascular && mode_a_idx < arrayLength(&mode_properties_v12)) {
        let vascular_a = mode_properties_v12[mode_a_idx];
        cell_a_nutrient_transport = vascular_a.x > 0.5;
        cell_a_nutrient_exchange = vascular_a.y > 0.5;
    }

    // --- Myocyte directional pump setup ---
    // The myocyte's current velocity defines its pump axis and strength. A fast-moving
    // myocyte pumps hard; a stationary myocyte applies no directional bias. This means
    // players can use swim force or physical contact to keep myocytes in motion and
    // sustain pumping. The velocity direction rotates with the organism naturally.
    let cell_a_is_myocyte = mode_valid && cell_type == MYOCYTE_CELL_TYPE;
    var myocyte_pump_axis: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var myocyte_pump_strength: f32 = 0.0;
    if (cell_a_is_myocyte) {
        let vel = velocities_in[cell_idx].xyz;
        let speed = length(vel);
        if (speed > 0.01) {
            myocyte_pump_axis    = vel / speed; // normalized pump direction
            myocyte_pump_strength = clamp(speed * MYOCYTE_PUMP_SPEED_SCALE, 0.0, 1.0);
        }
    }

    // Cache position and mass of cell_a for compression calculations.
    let pos_a  = positions_in[cell_idx].xyz;
    let mass_a = positions_in[cell_idx].w;

    // Pre-scan: find the maximum compression cell_a is experiencing from ANY neighbor.
    // This is what makes the myocyte+lipocyte pump work correctly: a myocyte squeezing
    // a lipocyte raises the lipocyte's max_compression_a, which then boosts the
    // lipocyte's outflow rate to ALL of its other connections (e.g. vasculocytes).
    // Without this, the compression boost would only apply to the myocyte<->lipocyte
    // connection itself, not to the lipocyte's downstream connections.
    var max_compression_a: f32 = 0.0;
    for (var i = 0; i < 20; i++) {
        let adh_idx = adhesion_list[i];
        if (adh_idx < 0 || adh_idx >= i32(arrayLength(&adhesion_connections))) {
            continue;
        }
        let adh = adhesion_connections[adh_idx];
        if (adh.is_active == 0u) {
            continue;
        }
        // Check both directions: cell_a may appear as either endpoint.
        var neighbor_idx: u32;
        if (adh.cell_a_index == cell_idx) {
            neighbor_idx = adh.cell_b_index;
        } else if (adh.cell_b_index == cell_idx) {
            neighbor_idx = adh.cell_a_index;
        } else {
            continue;
        }
        if (neighbor_idx >= cell_count || death_flags[neighbor_idx] == 1u) {
            continue;
        }
        let pos_n  = positions_in[neighbor_idx].xyz;
        let mass_n = positions_in[neighbor_idx].w;
        let comp   = adhesion_compression(pos_a, mass_a, pos_n, mass_n);
        max_compression_a = max(max_compression_a, comp);
    }
    // Global outflow multiplier for this cell - applied to all connections uniformly.
    let cell_pump_mult = 1.0 + max_compression_a * PUMP_AMPLIFICATION;

    // Pass 1: compute desired flows and accumulate total outflow for this cell
    var total_out: f32 = 0.0;
    var desired_arr: array<f32, 20>;
    var rate_arr: array<f32, 20>;   // effective max rate per connection
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
        if ((adhesion.bond_flags & BOND_FLAG_BARRIER_BALL) != 0u) {
            continue;
        }
        let cell_b_idx = adhesion.cell_b_index;
        if (cell_b_idx >= cell_count || death_flags[cell_b_idx] == 1u) {
            continue;
        }
        // Only block the sending cell (cell_a = cell_idx) if it is split-deferred.
        // Never block a cell from *receiving* - a starving receiver must always be reachable.
        if (is_cell_blocked_from_nutrients(cell_idx)) {
            continue;
        }

        let mode_b_idx = mode_indices[cell_b_idx];
        if (mode_a_idx >= arrayLength(&mode_properties_v0) ||
            mode_b_idx >= arrayLength(&mode_properties_v0)) {
            continue;
        }

        // Also skip if cell_b is split-deferred - don't let nutrients flow into a cell
        // that is about to divide, so both cells in a deferred pair freeze at the same
        // nutrient level and divide with symmetric children.
        if (is_cell_blocked_from_nutrients(cell_b_idx)) {
            continue;
        }

        // Determine if cell_b is a vasculocyte.
        var cell_b_type = 0xFFFFFFFFu;
        if (mode_b_idx < arrayLength(&mode_cell_types)) {
            cell_b_type = mode_cell_types[mode_b_idx];
        }
        let cell_b_is_vascular = cell_b_type == VASCULOCYTE_CELL_TYPE;
        var cell_b_nutrient_transport = false;
        var cell_b_nutrient_exchange = false;
        if (cell_b_is_vascular && mode_b_idx < arrayLength(&mode_properties_v12)) {
            let vascular_b = mode_properties_v12[mode_b_idx];
            cell_b_nutrient_transport = vascular_b.x > 0.5;
            cell_b_nutrient_exchange = vascular_b.y > 0.5;
        }

        // --- Effective transport rate for this connection ---
        // 1. Transport-enabled vascular-to-vascular: 5x highway rate.
        // 2. Vascular sealed to/from non-vascular: nearly blocked.
        // 3. Vascular exchange port to/from non-vascular: normal rate.
        // 4. Non-vascular to non-vascular: normal rate.
        // 6. Embryocyte receiver: rate scaled by nutrient_priority so high-priority
        //    embryocytes fill their reserve much faster than the base 30/sec.
        var effective_rate: f32 = TRANSPORT_RATE;
        if (cell_a_is_vascular && cell_b_is_vascular) {
            if (cell_a_nutrient_transport && cell_b_nutrient_transport) {
                effective_rate = VASCULAR_TRANSPORT_RATE;
            } else {
                effective_rate = TRANSPORT_RATE * VASCULAR_SEAL_FACTOR;
            }
        } else if (cell_a_is_vascular && !cell_b_is_vascular) {
            if (!cell_a_nutrient_exchange) {
                effective_rate = TRANSPORT_RATE * VASCULAR_SEAL_FACTOR;
            }
        } else if (!cell_a_is_vascular && cell_b_is_vascular) {
            if (!cell_b_nutrient_exchange) {
                effective_rate = TRANSPORT_RATE * VASCULAR_SEAL_FACTOR;
            }
        }

        // Embryocyte receivers: scale rate by their nutrient_priority so the setting
        // actually controls how fast the reserve fills. Priority 4.0 -> 4x base rate.
        let cell_b_is_embryocyte_check = cell_b_type == 10u;
        if (cell_b_is_embryocyte_check && mode_b_idx < arrayLength(&mode_properties_v1)) {
            let embryo_priority = mode_properties_v1[mode_b_idx].y;
            effective_rate *= max(embryo_priority, 1.0);
        }

        // --- Compression-driven pump boost ---
        // Apply the cell-level pump multiplier derived from the pre-scan.
        // If any neighbor (e.g. a myocyte) is compressing cell_a, all of cell_a's
        // outgoing connections benefit - so a lipocyte squeezed by a myocyte pushes
        // nutrients faster into every adjacent vasculocyte, not just back into the myocyte.
        effective_rate *= cell_pump_mult;

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

        // For embryocyte receivers, bypass the pressure-equilibrium formula entirely.
        // The embryocyte is a pure sink - the sender should push at the full effective_rate
        // as long as it has nutrients. The pressure-diff formula caps flow at
        // nutrients_a / priority_a (e.g. 20/3.5 ~= 5.7/sec), which is far too slow.
        // Instead, use effective_rate directly as desired so the sender drains into
        // the embryocyte as fast as the rate cap allows.
        var desired: f32;
        if (cell_b_type == 10u) {
            // Embryocyte receiver: always push at full effective_rate (A->B)
            desired = effective_rate;
        } else {
            // --- Myocyte directional pump bias ---
            var pump_pressure_bias: f32 = 0.0;
            if (cell_a_is_myocyte && myocyte_pump_strength > 0.0) {
                let pos_b    = positions_in[cell_b_idx].xyz;
                let dir_to_b = pos_b - pos_a;
                let dist     = length(dir_to_b);
                if (dist > 0.001) {
                    let axis_dot = dot(dir_to_b / dist, myocyte_pump_axis);
                    pump_pressure_bias = MYOCYTE_PUMP_FORCE * myocyte_pump_strength * axis_dot;
                }
            }
            let pressure_diff = nutrients_a_snap / priority_a - nutrients_b / priority_b + pump_pressure_bias;
            desired = clamp(pressure_diff, -effective_rate, effective_rate);
        }

        desired_arr[i] = desired;
        rate_arr[i]    = effective_rate;
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

        // Scale outflow so the sending cell's total outflow stays within budget.
        // Use the per-connection effective_rate as the cap (already accounts for
        // vascular highway rate and compression boost).
        let conn_rate = rate_arr[i];
        var scale: f32 = 1.0;
        if (desired > 0.0 && total_out > conn_rate) {
            scale = conn_rate / total_out;
        }

        // Target transfer this step (nutrients), lerped toward desired
        let target_per_sec = desired * scale;
        let nutrient_transfer = target_per_sec * lerp_t * params.delta_time;

        // Determine if cell_b is an Embryocyte - incoming goes to reserve, not nutrients.
        let mode_b_idx = mode_indices[cell_b_idx];
        var cell_b_is_embryocyte = false;
        if (mode_b_idx < arrayLength(&mode_cell_types)) {
            cell_b_is_embryocyte = mode_cell_types[mode_b_idx] == 10u;
        }

        // Clamp by available nutrients and receiver capacity
        let min_nutrients_a = select(0.0, 10.0, prioritize_a);
        let min_nutrients_b = select(0.0, 10.0, prioritize_b);
        // Cap at 200 before doubling so the "never split" sentinel (threshold > 100)
        // doesn't inflate the nutrient cap to an absurd value.
        let max_nutrients_a = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;

        // For Embryocyte receivers: capacity = space left in reserve (x1000 fixed-point)
        var max_recv_b: f32;
        if (cell_b_is_embryocyte) {
            let cur_reserve_b = atomicLoad(&embryocyte_reserves[cell_b_idx]);
            max_recv_b = f32(65535000u - min(cur_reserve_b, 65535000u)) / FIXED_POINT_SCALE;
        } else {
            max_recv_b = min(split_nutrient_thresholds[cell_b_idx], 200.0) * 2.0;
        }

        var actual_transfer: f32;
        if (nutrient_transfer > 0.0) {
            let can_give = max(nutrients_a_snap - min_nutrients_a, 0.0);
            let can_recv = max(max_recv_b - nutrients_b, 0.0);
            actual_transfer = min(nutrient_transfer, min(can_give, can_recv));
        } else {
            // Reverse flow (B->A): Embryocyte cell_b cannot send, so block
            if (cell_b_is_embryocyte) {
                actual_transfer = 0.0;
            } else {
                let can_give = max(nutrients_b - min_nutrients_b, 0.0);
                let can_recv = max(max_nutrients_a - nutrients_a_snap, 0.0);
                actual_transfer = max(nutrient_transfer, -min(can_give, can_recv));
            }
        }

        // Apply transfer using atomic operations (thread-safe).
        // If cell_b is an Embryocyte, incoming nutrients go to its reserve buffer.
        if (actual_transfer > 0.0 && cell_b_is_embryocyte) {
            // A->B: deduct from A's nutrients, add to B's reserve (x1000 fixed-point)
            let transfer_fixed = float_to_fixed(actual_transfer);
            atomicAdd(&nutrients_buffer[cell_idx], -transfer_fixed);
            let reserve_gain = u32(float_to_fixed(actual_transfer));
            atomicAdd(&embryocyte_reserves[cell_b_idx], reserve_gain);
        } else {
            // Normal transfer: atomically subtract from cell A, add to cell B
            let transfer_fixed = float_to_fixed(actual_transfer);
            atomicAdd(&nutrients_buffer[cell_idx], -transfer_fixed);
            atomicAdd(&nutrients_buffer[cell_b_idx], transfer_fixed);
        }
    }
    
    // NOTE: Death detection is handled by a separate shader that checks nutrients < MIN_NUTRIENTS
    // and sets death_flags accordingly.
}
