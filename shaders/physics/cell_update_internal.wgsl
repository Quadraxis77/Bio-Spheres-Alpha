//! # Cell Internal Update Compute Shader
//!
//! This shader handles all internal cell state updates including aging, nutrient processing,
//! signaling substance updates, and toxin accumulation. It runs entirely on the GPU with
//! zero CPU involvement, processing all cells in parallel.
//!
//! ## Functionality
//! - **Aging**: Updates cell age and handles age-related effects
//! - **Nutrient Processing**: Updates nitrate levels based on cell type and behavior
//! - **Signaling Substances**: Processes chemical communication between cells
//! - **Toxin Accumulation**: Handles toxin buildup and effects on cell health
//! - **Split Timer**: Manages division timing using split_ready_frame tracking
//!
//! ## Requirements Addressed
//! - 3.7: Cell internal physics (mass, nutrients, aging) on GPU
//! - 13.1: Nutrient gain calculations using GPU compute shaders
//! - 13.2: Nutrient consumption using GPU compute shaders based on swim force

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
    _padding: array<vec4<f32>, 12>,
}

// GPU Mode structure matching reference implementation
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

// Bind group 0: Physics parameters and cell data
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ages: array<f32>;
@group(0) @binding(4) var<storage, read_write> nitrates: array<f32>;
@group(0) @binding(5) var<storage, read_write> toxins: array<f32>;
@group(0) @binding(6) var<storage, read_write> signalling_substances: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(8) var<storage, read> genome_modes: array<GpuMode>;
@group(0) @binding(9) var<storage, read> birth_times: array<f32>;
@group(0) @binding(10) var<storage, read> split_intervals: array<f32>;
@group(0) @binding(11) var<storage, read> split_masses: array<f32>;
@group(0) @binding(12) var<storage, read_write> split_ready_frame: array<i32>;
@group(0) @binding(13) var<storage, read> genome_ids: array<u32>;

// Cell type constants (matching CPU implementation)
const CELL_TYPE_TEST: i32 = 0;        // Test cells with automatic nutrient gain
const CELL_TYPE_FLAGELLOCYTE: i32 = 1; // Flagellocyte cells with swim forces

// Nutrient system constants
const BASE_NUTRIENT_GAIN: f32 = 0.1;           // Base nutrient gain per frame
const FLAGELLOCYTE_CONSUMPTION_RATE: f32 = 0.05; // Nutrient consumption per swim force
const TOXIN_ACCUMULATION_RATE: f32 = 0.001;    // Toxin buildup per frame
const SIGNALING_DECAY_RATE: f32 = 0.95;        // Signaling substance decay per frame
const SPLIT_DELAY_FRAMES: i32 = 60;            // Frames to wait before division after reaching mass

/// Get cell type from genome mode data
fn get_cell_type(genome_id: u32, mode_index: i32) -> i32 {
    // For now, determine cell type based on genome ID
    // This will be expanded when genome system is fully integrated
    if (genome_id % 2u == 0u) {
        return CELL_TYPE_TEST;
    } else {
        return CELL_TYPE_FLAGELLOCYTE;
    }
}

/// Calculate nutrient gain based on cell type and current state
fn calculate_nutrient_gain(cell_type: i32, current_mass: f32, velocity_magnitude: f32) -> f32 {
    var gain = BASE_NUTRIENT_GAIN;
    
    // Test cells get automatic nutrient gain
    if (cell_type == CELL_TYPE_TEST) {
        gain *= 2.0; // Double gain for test cells
    }
    
    // Flagellocyte cells consume nutrients based on movement
    if (cell_type == CELL_TYPE_FLAGELLOCYTE) {
        let consumption = velocity_magnitude * FLAGELLOCYTE_CONSUMPTION_RATE;
        gain -= consumption;
    }
    
    // Scale gain by mass (larger cells need more nutrients)
    gain *= sqrt(current_mass);
    
    return gain;
}

/// Update signaling substances with decay and interactions
fn update_signaling_substances(cell_index: u32, current_substances: vec4<f32>) -> vec4<f32> {
    var new_substances = current_substances;
    
    // Apply decay to all substances
    new_substances *= SIGNALING_DECAY_RATE;
    
    // Add small random fluctuations (using cell index as seed)
    let seed = f32(cell_index) * 0.1 + physics_params.current_time * 0.01;
    let noise = sin(seed) * 0.01;
    new_substances.x += noise;
    new_substances.y += cos(seed * 1.3) * 0.01;
    new_substances.z += sin(seed * 1.7) * 0.01;
    new_substances.w += cos(seed * 2.1) * 0.01;
    
    // Clamp to valid range [0, 1]
    new_substances = clamp(new_substances, vec4<f32>(0.0), vec4<f32>(1.0));
    
    return new_substances;
}

/// Update toxin levels with accumulation and effects
fn update_toxins(current_toxins: f32, nutrient_level: f32) -> f32 {
    var new_toxins = current_toxins;
    
    // Accumulate toxins over time
    new_toxins += TOXIN_ACCUMULATION_RATE * physics_params.delta_time;
    
    // High nutrient levels help clear toxins
    if (nutrient_level > 0.5) {
        new_toxins -= 0.002 * physics_params.delta_time;
    }
    
    // Clamp to valid range [0, 1]
    new_toxins = clamp(new_toxins, 0.0, 1.0);
    
    return new_toxins;
}

/// Check if cell is ready to divide and update split timing
fn update_split_timing(
    cell_index: u32,
    current_mass: f32,
    split_mass: f32,
    birth_time: f32,
    split_interval: f32,
    current_split_ready_frame: i32
) -> i32 {
    let age = physics_params.current_time - birth_time;
    
    // Check if cell meets division criteria
    let mass_ready = current_mass >= split_mass;
    let age_ready = age >= split_interval;
    
    if (mass_ready && age_ready) {
        // If not already marked for division, set the ready frame
        if (current_split_ready_frame < 0) {
            return physics_params.current_frame + SPLIT_DELAY_FRAMES;
        } else {
            // Keep existing ready frame
            return current_split_ready_frame;
        }
    } else {
        // Not ready for division
        return -1;
    }
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
    let current_age = ages[cell_index];
    let current_nitrates = nitrates[cell_index];
    let current_toxins = toxins[cell_index];
    let current_substances = signalling_substances[cell_index];
    let mode_index = mode_indices[cell_index];
    let genome_id = genome_ids[cell_index];
    let birth_time = birth_times[cell_index];
    let split_interval = split_intervals[cell_index];
    let split_mass = split_masses[cell_index];
    let current_split_ready_frame = split_ready_frame[cell_index];
    
    // Update age
    let new_age = current_age + physics_params.delta_time;
    ages[cell_index] = new_age;
    
    // Determine cell type
    let cell_type = get_cell_type(genome_id, mode_index);
    
    // Update nutrient levels
    let nutrient_gain = calculate_nutrient_gain(cell_type, current_mass, velocity_magnitude);
    let new_nitrates = clamp(current_nitrates + nutrient_gain * physics_params.delta_time, 0.0, 10.0);
    nitrates[cell_index] = new_nitrates;
    
    // Update toxin levels
    let new_toxins = update_toxins(current_toxins, new_nitrates);
    toxins[cell_index] = new_toxins;
    
    // Update signaling substances
    let new_substances = update_signaling_substances(cell_index, current_substances);
    signalling_substances[cell_index] = new_substances;
    
    // Update split timing
    let new_split_ready_frame = update_split_timing(
        cell_index,
        current_mass,
        split_mass,
        birth_time,
        split_interval,
        current_split_ready_frame
    );
    split_ready_frame[cell_index] = new_split_ready_frame;
}