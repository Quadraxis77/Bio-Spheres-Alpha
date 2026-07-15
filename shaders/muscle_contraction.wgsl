// Muscle Contraction Shader - Writes per-cell contraction values for Myocyte cells
//
// Each myocyte computes its contraction amount (0.0 = relaxed, 1.0 = fully contracted)
// and writes it to a per-cell buffer. The adhesion_physics shader then reads each cell's
// contraction value to scale its own half of the bond:
//   cell_a_reach = rest_length * (1.0 - contraction_a)
//   cell_b_reach = rest_length * (1.0 - contraction_b)
//
// This means each myocyte only controls its own side of the adhesion bond.
// Two opposing myocytes on the same bond both pull inward independently.
//
// In Pulse mode: oscillates between contracted and relaxed using a sine wave timer.
// Pulse A contracts when sin(time * rate * 2pi) >= 0, Pulse B when < 0.
// In Signal mode: reads a signal channel and contracts based on threshold.
//
// Must run BEFORE adhesion_physics.wgsl.
// Workgroup size: 256 threads for optimal GPU occupancy

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

// Cell type behavior flags for parameterized shader logic
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    applies_buoyancy: u32,
    applies_cilia_force: u32,
    applies_muscle_contraction: u32,
    _padding: array<u32, 7>,
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(1)
var<storage, read_write> cell_count_buffer: array<u32>;

// Cell data (group 1)
@group(1) @binding(0)
var<storage, read> mode_indices: array<u32>;

// Mode cell types lookup table
@group(1) @binding(1)
var<storage, read> mode_cell_types: array<u32>;

// Cell type behavior flags
@group(1) @binding(2)
var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// Signal flags buffer (read-only) - 16 channels per cell
@group(1) @binding(3)
var<storage, read> signal_flags: array<atomic<u32>>;

// Myocyte mode properties
// v7: [myocyte_contraction, myocyte_use_signal, myocyte_signal_channel, myocyte_threshold]
// v8: [myocyte_contraction_above, myocyte_contraction_below, myocyte_pulse_rate, myocyte_pulse_phase]
@group(1) @binding(4)
var<storage, read> mode_properties_v7: array<vec4<f32>>;

@group(1) @binding(5)
var<storage, read> mode_properties_v8: array<vec4<f32>>;

// v11: [devorocyte_consume_range, devorocyte_consume_rate, myocyte_grip_contracted, myocyte_grip_extended]
@group(1) @binding(6)
var<storage, read> mode_properties_v11: array<vec4<f32>>;

// Per-cell outputs (group 2)
// Binding 0: contraction value (0.0 = relaxed, 1.0 = fully contracted)
@group(2) @binding(0)
var<storage, read_write> muscle_contraction_out: array<f32>;

// Binding 1: grip (friction drag) value — mix(grip_extended, grip_contracted, contraction)
// Read by position_update shader to apply medium-scaled friction.
@group(2) @binding(1)
var<storage, read_write> cell_grip_out: array<f32>;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // Get mode index and derive cell type
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types) || mode_idx >= arrayLength(&mode_properties_v7)) {
        // Non-myocyte: write 0.0 (relaxed, no grip)
        muscle_contraction_out[cell_idx] = 0.0;
        cell_grip_out[cell_idx] = 0.0;
        return;
    }

    let cell_type = mode_cell_types[mode_idx];

    // Check if this cell type applies muscle contraction
    let behavior = type_behaviors[cell_type];
    if (behavior.applies_muscle_contraction == 0u) {
        // Non-myocyte: write 0.0 (relaxed, no grip)
        muscle_contraction_out[cell_idx] = 0.0;
        cell_grip_out[cell_idx] = 0.0;
        return;
    }
    
    // Load myocyte parameters for this mode
    let myo_v7 = mode_properties_v7[mode_idx];
    let myo_v8 = mode_properties_v8[mode_idx];
    
    let fixed_contraction = myo_v7.x;  // myocyte_contraction (used in pulse mode)
    let use_signal = myo_v7.y;         // myocyte_use_signal (0.0 or 1.0)
    let signal_channel = myo_v7.z;     // myocyte_signal_channel
    let threshold = myo_v7.w;          // myocyte_threshold
    let contraction_above = myo_v8.x;  // myocyte_contraction_above
    let contraction_below = myo_v8.y;  // myocyte_contraction_below
    let pulse_rate = myo_v8.z;         // myocyte_pulse_rate (cycles per second)
    let pulse_phase = myo_v8.w;        // myocyte_pulse_phase (0.0 = Pulse A, 1.0 = Pulse B)
    
    // Determine effective contraction amount
    var contraction: f32;
    if (use_signal >= 0.5) {
        // Signal-based mode: read from the specific channel
        let sig_channel = clamp(u32(signal_channel), 0u, 15u);
        let raw_signal = atomicLoad(&signal_flags[cell_idx * 16u + sig_channel]);
        let signal_value = f32(raw_signal & 2047u); // Extract value component
        
        if (signal_value >= threshold) {
            contraction = contraction_above;
        } else {
            contraction = contraction_below;
        }
    } else {
        // Phased timer mode: oscillate based on current_time
        // sin(time * rate * 2pi) produces a wave from -1 to +1
        // Pulse A contracts when sin >= 0, Pulse B contracts when sin < 0
        let wave = sin(params.current_time * pulse_rate * 2.0 * PI);
        let is_active_phase = select((wave < 0.0), (wave >= 0.0), pulse_phase < 0.5);
        
        if (is_active_phase) {
            contraction = fixed_contraction;
        } else {
            contraction = 0.0;
        }
    }
    
    // Write per-cell contraction value (clamped to valid range)
    let clamped = clamp(contraction, 0.0, 1.0);
    muscle_contraction_out[cell_idx] = clamped;

    // Compute grip from contraction phase.
    // grip_contracted = drag when fully contracted, grip_extended = drag when fully extended.
    // Interpolating between them gives smooth grip variation through the pulse cycle.
    let grip_contracted = mode_properties_v11[mode_idx].z;
    let grip_extended   = mode_properties_v11[mode_idx].w;
    cell_grip_out[cell_idx] = mix(grip_extended, grip_contracted, clamped);
}
