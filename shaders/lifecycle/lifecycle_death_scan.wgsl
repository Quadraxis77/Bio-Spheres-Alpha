//! # Lifecycle Death Scan Compute Shader
//!
//! This shader implements the first phase of prefix-sum lifecycle management by
//! scanning all cells to identify those ready to die. It sets death flags based
//! on various death criteria including nutrient depletion, toxin levels, and age.
//!
//! ## Functionality
//! - **Death Detection**: Identify cells that meet death criteria
//! - **Flag Setting**: Set death flags for prefix-sum compaction
//! - **Adhesion Cleanup Preparation**: Prepare for adhesion cleanup
//! - **Deterministic Processing**: Ensure consistent death detection across frames
//!
//! ## Requirements Addressed
//! - 6.6: Cell aging and death conditions using GPU compute shaders
//! - 6.7: Cell removal and state compaction using GPU compute shaders

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

// Bind group 0: Physics parameters and cell data
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ages: array<f32>;
@group(0) @binding(3) var<storage, read> nitrates: array<f32>;
@group(0) @binding(4) var<storage, read> toxins: array<f32>;
@group(0) @binding(5) var<storage, read> birth_times: array<f32>;
@group(0) @binding(6) var<storage, read_write> death_flags: array<u32>;

// Death criteria constants
const MAX_CELL_AGE: f32 = 300.0;              // Maximum age before natural death
const NUTRIENT_DEATH_THRESHOLD: f32 = -1.0;   // Death when nutrients drop below this
const TOXIN_DEATH_THRESHOLD: f32 = 0.9;       // Death when toxins exceed this
const MIN_MASS_THRESHOLD: f32 = 0.1;          // Death when mass drops below this

/// Check if cell should die from age
fn should_die_from_age(age: f32) -> bool {
    return age >= MAX_CELL_AGE;
}

/// Check if cell should die from nutrient depletion
fn should_die_from_nutrients(nitrates: f32) -> bool {
    return nitrates <= NUTRIENT_DEATH_THRESHOLD;
}

/// Check if cell should die from toxin accumulation
fn should_die_from_toxins(toxins: f32) -> bool {
    return toxins >= TOXIN_DEATH_THRESHOLD;
}

/// Check if cell should die from insufficient mass
fn should_die_from_mass(mass: f32) -> bool {
    return mass <= MIN_MASS_THRESHOLD;
}

/// Comprehensive death check for a cell
fn should_cell_die(
    age: f32,
    nitrates: f32,
    toxins: f32,
    mass: f32,
    birth_time: f32
) -> bool {
    // Calculate actual age from birth time
    let actual_age = physics_params.current_time - birth_time;
    
    // Check all death criteria
    let age_death = should_die_from_age(actual_age);
    let nutrient_death = should_die_from_nutrients(nitrates);
    let toxin_death = should_die_from_toxins(toxins);
    let mass_death = should_die_from_mass(mass);
    
    return age_death || nutrient_death || toxin_death || mass_death;
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
    let mass = pos_mass.w;
    let age = ages[cell_index];
    let nitrates_level = nitrates[cell_index];
    let toxin_level = toxins[cell_index];
    let birth_time = birth_times[cell_index];
    
    // Check if cell should die
    let should_die = should_cell_die(age, nitrates_level, toxin_level, mass, birth_time);
    
    // Set death flag for prefix-sum compaction
    if (should_die) {
        death_flags[cell_index] = 1u;
    } else {
        death_flags[cell_index] = 0u;
    }
}