//! # Nutrient System Compute Shader
//!
//! This shader implements the complete nutrient system for cell metabolism including
//! nutrient gain, consumption, flow calculations with pressure-based equilibrium,
//! priority-based distribution, and cell death from nutrient depletion.
//!
//! ## Functionality
//! - **Nutrient Gain**: Automatic gain for Test cells, consumption for Flagellocytes
//! - **Nutrient Flow**: Pressure-based equilibrium between connected cells via adhesions
//! - **Priority Distribution**: Priority-based nutrient allocation with temporary boosts
//! - **Cell Death**: Handle cell death from nutrient depletion
//! - **Mass Updates**: Update cell radii based on mass changes from nutrients
//!
//! ## Requirements Addressed
//! - 13.3: Nutrient flow calculations with pressure-based equilibrium
//! - 13.4: Priority-based nutrient distribution
//! - 13.5: Temporary priority boosts using GPU-based threshold calculations
//! - 13.7: Cell death from nutrient depletion

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

// GPU Mode structure for nutrient priorities
struct GpuMode {
    color: vec4<f32>,
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    split_direction: vec4<f32>,
    child_modes: vec2<i32>,
    split_interval: f32,
    genome_offset: i32,
    // Adhesion settings (48 bytes)
    adhesion_can_break: i32,
    adhesion_break_force: f32,
    adhesion_rest_length: f32,
    adhesion_linear_spring_stiffness: f32,
    adhesion_linear_spring_damping: f32,
    adhesion_orientation_spring_stiffness: f32,
    adhesion_orientation_spring_damping: f32,
    adhesion_max_angular_deviation: f32,
    adhesion_twist_constraint_stiffness: f32,
    adhesion_twist_constraint_damping: f32,
    adhesion_enable_twist_constraint: i32,
    adhesion_padding: i32,
    // Adhesion behavior (16 bytes)
    parent_make_adhesion: i32,
    child_a_keep_adhesion: i32,
    child_b_keep_adhesion: i32,
    max_adhesions: i32,
}

// Adhesion connection structure (96 bytes)
struct GpuAdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _padding_zones: vec2<u32>,
    anchor_direction_a: vec3<f32>,
    _padding_a: f32,
    anchor_direction_b: vec3<f32>,
    _padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
}

// Bind group 0: Physics parameters and cell data
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nitrates: array<f32>;
@group(0) @binding(4) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(5) var<storage, read> genome_modes: array<GpuMode>;
@group(0) @binding(6) var<storage, read> genome_ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> split_ready_frame: array<i32>;
@group(0) @binding(8) var<storage, read_write> death_flags: array<u32>;

// Bind group 1: Adhesion system data
@group(1) @binding(0) var<storage, read> adhesion_connections: array<GpuAdhesionConnection>;
@group(1) @binding(1) var<storage, read> adhesion_indices: array<i32>;
@group(1) @binding(2) var<storage, read> adhesion_counts: array<u32>;

// Nutrient system constants
const NUTRIENT_GAIN_RATE: f32 = 0.2;              // Base nutrient gain per second
const NUTRIENT_CONSUMPTION_RATE: f32 = 0.1;       // Consumption per unit swim force
const NUTRIENT_FLOW_RATE: f32 = 0.5;              // Flow rate between connected cells
const NUTRIENT_DEATH_THRESHOLD: f32 = -1.0;       // Death when nutrients drop below this
const NUTRIENT_STORAGE_CAP: f32 = 10.0;           // Maximum nutrient storage
const PRIORITY_BOOST_THRESHOLD: f32 = 2.0;        // Threshold for priority boost
const PRIORITY_BOOST_MULTIPLIER: f32 = 2.0;       // Multiplier for priority boost
const MASS_TO_RADIUS_FACTOR: f32 = 0.62035;       // (3/(4*π))^(1/3) for sphere volume
const MIN_CELL_RADIUS: f32 = 0.5;                 // Minimum cell radius
const MAX_CELL_RADIUS: f32 = 3.0;                 // Maximum cell radius

// Cell type constants
const CELL_TYPE_TEST: i32 = 0;        // Test cells with automatic nutrient gain
const CELL_TYPE_FLAGELLOCYTE: i32 = 1; // Flagellocyte cells with swim forces

/// Get cell type from genome data
fn get_cell_type(genome_id: u32, mode_index: i32) -> i32 {
    // For now, determine cell type based on genome ID
    // This will be expanded when genome system is fully integrated
    if (genome_id % 2u == 0u) {
        return CELL_TYPE_TEST;
    } else {
        return CELL_TYPE_FLAGELLOCYTE;
    }
}

/// Calculate base nutrient gain for a cell based on its type
fn calculate_base_nutrient_gain(cell_type: i32, velocity_magnitude: f32) -> f32 {
    var gain = 0.0;
    
    // Test cells get automatic nutrient gain
    if (cell_type == CELL_TYPE_TEST) {
        gain = NUTRIENT_GAIN_RATE;
    }
    
    // Flagellocyte cells consume nutrients based on movement
    if (cell_type == CELL_TYPE_FLAGELLOCYTE) {
        let consumption = velocity_magnitude * NUTRIENT_CONSUMPTION_RATE;
        gain = -consumption; // Negative gain (consumption)
    }
    
    return gain;
}

/// Calculate nutrient priority for a cell
fn calculate_nutrient_priority(
    cell_type: i32,
    current_nitrates: f32,
    mode_index: i32,
    genome_modes: ptr<storage, array<GpuMode>, read>
) -> f32 {
    var priority = 1.0; // Base priority
    
    // Higher priority for cells with low nutrients
    if (current_nitrates < PRIORITY_BOOST_THRESHOLD) {
        priority *= PRIORITY_BOOST_MULTIPLIER;
    }
    
    // Cell type specific priorities
    if (cell_type == CELL_TYPE_FLAGELLOCYTE) {
        priority *= 1.5; // Flagellocytes have higher priority
    }
    
    // TODO: Add genome-specific nutrient_priority when genome system is integrated
    // if (mode_index >= 0 && mode_index < arrayLength(genome_modes)) {
    //     let mode = genome_modes[mode_index];
    //     priority *= mode.nutrient_priority;
    // }
    
    return priority;
}

/// Calculate nutrient flow between two connected cells
fn calculate_nutrient_flow(
    nitrates_a: f32,
    nitrates_b: f32,
    priority_a: f32,
    priority_b: f32
) -> f32 {
    // Pressure-based equilibrium with priority weighting
    let pressure_diff = (nitrates_a / priority_a) - (nitrates_b / priority_b);
    let flow = pressure_diff * NUTRIENT_FLOW_RATE;
    
    // Limit flow to prevent negative nutrients
    let max_flow_from_a = max(0.0, nitrates_a);
    let max_flow_to_a = max(0.0, nitrates_b);
    
    return clamp(flow, -max_flow_to_a, max_flow_from_a);
}

/// Update cell radius based on mass
fn calculate_radius_from_mass(mass: f32) -> f32 {
    let radius = pow(mass * MASS_TO_RADIUS_FACTOR, 1.0 / 3.0);
    return clamp(radius, MIN_CELL_RADIUS, MAX_CELL_RADIUS);
}

/// Check if cell should die from nutrient depletion
fn should_cell_die(nitrates: f32) -> bool {
    return nitrates <= NUTRIENT_DEATH_THRESHOLD;
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
    let velocity_vec = velocity[cell_index].xyz;
    let velocity_magnitude = length(velocity_vec);
    let current_nitrates = nitrates[cell_index];
    let mode_index = mode_indices[cell_index];
    let genome_id = genome_ids[cell_index];
    let adhesion_count = adhesion_counts[cell_index];
    let current_split_ready_frame = split_ready_frame[cell_index];
    
    // Determine cell type
    let cell_type = get_cell_type(genome_id, mode_index);
    
    // Calculate base nutrient gain/consumption
    let base_gain = calculate_base_nutrient_gain(cell_type, velocity_magnitude);
    
    // Calculate nutrient priority for this cell
    let priority = calculate_nutrient_priority(cell_type, current_nitrates, mode_index, &genome_modes);
    
    // Calculate nutrient flow from adhesion connections
    var nutrient_flow = 0.0;
    
    // Process all adhesion connections for this cell
    let adhesion_start_index = cell_index * 10u; // 10 adhesions per cell max
    for (var i = 0u; i < 10u; i++) {
        let adhesion_index = adhesion_indices[adhesion_start_index + i];
        
        // Check if this adhesion slot is active
        if (adhesion_index >= 0) {
            let connection = adhesion_connections[adhesion_index];
            
            // Check if connection is active
            if (connection.is_active != 0u) {
                var other_cell_index: u32;
                
                // Determine which cell is the other one
                if (connection.cell_a_index == cell_index) {
                    other_cell_index = connection.cell_b_index;
                } else if (connection.cell_b_index == cell_index) {
                    other_cell_index = connection.cell_a_index;
                } else {
                    continue; // This connection doesn't involve our cell
                }
                
                // Bounds check for other cell
                if (other_cell_index >= physics_params.cell_count) {
                    continue;
                }
                
                // Get other cell's nutrient data
                let other_nitrates = nitrates[other_cell_index];
                let other_mode_index = mode_indices[other_cell_index];
                let other_genome_id = genome_ids[other_cell_index];
                let other_cell_type = get_cell_type(other_genome_id, other_mode_index);
                let other_priority = calculate_nutrient_priority(other_cell_type, other_nitrates, other_mode_index, &genome_modes);
                
                // Calculate flow between cells
                let flow = calculate_nutrient_flow(current_nitrates, other_nitrates, priority, other_priority);
                nutrient_flow -= flow; // Negative because flow is from our perspective
            }
        }
    }
    
    // Apply nutrient changes
    var new_nitrates = current_nitrates;
    
    // Apply base gain/consumption
    new_nitrates += base_gain * physics_params.delta_time;
    
    // Apply nutrient flow from adhesions
    new_nitrates += nutrient_flow * physics_params.delta_time;
    
    // Clamp to storage limits
    new_nitrates = clamp(new_nitrates, NUTRIENT_DEATH_THRESHOLD - 1.0, NUTRIENT_STORAGE_CAP);
    
    // Update nitrates
    nitrates[cell_index] = new_nitrates;
    
    // Check for cell death from nutrient depletion
    if (should_cell_die(new_nitrates)) {
        death_flags[cell_index] = 1u; // Mark for death
        split_ready_frame[cell_index] = -1; // Cancel any pending division
    }
    
    // Defer nutrient transfer for dividing cells
    // If cell is ready to divide, don't allow nutrient flow to prevent mass changes
    if (current_split_ready_frame >= 0 && physics_params.current_frame >= current_split_ready_frame) {
        // Cell is dividing, freeze nutrient changes
        nitrates[cell_index] = current_nitrates;
    }
    
    // Update cell mass and radius based on nutrient levels
    // Mass increases with nutrient storage (simplified model)
    let nutrient_mass_contribution = max(0.0, new_nitrates * 0.1);
    let base_mass = 1.0; // Base cell mass
    let new_mass = base_mass + nutrient_mass_contribution;
    let new_radius = calculate_radius_from_mass(new_mass);
    
    // Update position_and_mass buffer with new mass
    position_and_mass[cell_index] = vec4<f32>(pos_mass.xyz, new_mass);
}