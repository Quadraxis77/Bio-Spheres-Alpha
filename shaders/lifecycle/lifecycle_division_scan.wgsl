//! # Lifecycle Division Scan Compute Shader
//!
//! This shader implements the first phase of division slot assignment by scanning
//! all cells to identify those ready to divide and calculating their slot reservation
//! needs. It prepares data for prefix-sum slot allocation.
//!
//! ## Functionality
//! - **Division Detection**: Identify cells ready to divide based on mass and timing
//! - **Slot Calculation**: Calculate how many slots each dividing cell needs
//! - **Reservation Setup**: Prepare reservation data for prefix-sum allocation
//! - **Deterministic Processing**: Ensure consistent division detection across frames
//!
//! ## Requirements Addressed
//! - 6.1: Division mechanics using GPU compute shaders exclusively
//! - 6.2: Split_ready_frame tracking using GPU buffers and compute shaders
//! - 6.4: Division attempts using GPU compute shaders without CPU synchronization

// Physics parameters and configuration
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
    dragged_cell_index: i32,
    _padding1: vec3<f32>,
    _padding: array<f32, 48>,
}

// Bind group 0: Physics parameters and cell data
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> split_ready_frame: array<i32>;
@group(0) @binding(3) var<storage, read> split_masses: array<f32>;
@group(0) @binding(4) var<storage, read> birth_times: array<f32>;
@group(0) @binding(5) var<storage, read> split_intervals: array<f32>;
@group(0) @binding(6) var<storage, read> adhesion_counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> division_candidates: array<u32>;
@group(0) @binding(8) var<storage, read_write> division_reservations: array<u32>;

// Division constants
const MIN_DIVISION_MASS: f32 = 2.0;           // Minimum mass required for division
const MIN_DIVISION_AGE: f32 = 5.0;            // Minimum age before division allowed
const ADHESION_INHERITANCE_FACTOR: f32 = 0.7; // Fraction of adhesions inherited by children

/// Check if cell is ready to divide based on all criteria
fn is_ready_to_divide(
    current_mass: f32,
    split_mass: f32,
    birth_time: f32,
    split_interval: f32,
    split_ready_frame: i32,
    current_frame: i32
) -> bool {
    // Check mass requirement
    let mass_ready = current_mass >= split_mass && current_mass >= MIN_DIVISION_MASS;
    
    // Check age requirement
    let age = physics_params.current_time - birth_time;
    let age_ready = age >= split_interval && age >= MIN_DIVISION_AGE;
    
    // Check if split ready frame has been set and reached
    let timing_ready = split_ready_frame >= 0 && current_frame >= split_ready_frame;
    
    return mass_ready && age_ready && timing_ready;
}

/// Calculate how many slots a dividing cell needs
/// This includes: 1 slot for Child B + slots for inherited adhesions
fn calculate_slot_reservation(adhesion_count: u32) -> u32 {
    // Child B needs 1 slot
    var slots_needed = 1u;
    
    // Calculate adhesion inheritance slots
    // Each child inherits approximately 70% of parent's adhesions
    let inherited_adhesions = u32(f32(adhesion_count) * ADHESION_INHERITANCE_FACTOR);
    
    // Child A stays in parent's slot, Child B needs slots for its adhesions
    slots_needed += inherited_adhesions;
    
    return slots_needed;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Load cell data
    let pos_mass = position_and_mass[cell_index];
    let current_mass = pos_mass.w;
    let split_mass = split_masses[cell_index];
    let birth_time = birth_times[cell_index];
    let split_interval = split_intervals[cell_index];
    let ready_frame = split_ready_frame[cell_index];
    let adhesion_count = adhesion_counts[cell_index];
    
    // Check if cell is ready to divide
    let ready_to_divide = is_ready_to_divide(
        current_mass,
        split_mass,
        birth_time,
        split_interval,
        ready_frame,
        physics_params.current_frame
    );
    
    if (ready_to_divide) {
        // Mark as division candidate
        division_candidates[cell_index] = 1u;
        
        // Calculate slot reservation needs
        let slots_needed = calculate_slot_reservation(adhesion_count);
        division_reservations[cell_index] = slots_needed;
    } else {
        // Not ready to divide
        division_candidates[cell_index] = 0u;
        division_reservations[cell_index] = 0u;
    }
}