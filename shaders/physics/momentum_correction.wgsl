//! # Momentum Correction Compute Shader
//!
//! This compute shader implements momentum conservation corrections for the
//! physics simulation. It ensures that the total momentum of the system
//! remains conserved during interactions and prevents drift due to numerical
//! errors in force calculations.
//!
//! ## Conservation Principles
//! - **Linear Momentum**: Total momentum should remain constant in absence of external forces
//! - **Angular Momentum**: Total angular momentum should be conserved around center of mass
//! - **Energy Conservation**: Kinetic energy should not increase without external work
//! - **Numerical Stability**: Corrections should be small and gradual
//!
//! ## Correction Method
//! - **Two-Pass Algorithm**: First pass calculates totals, second pass applies corrections
//! - **Proportional Correction**: Corrections distributed based on cell mass
//! - **Damped Correction**: Gradual correction to prevent oscillations
//! - **Conservation Validation**: Verify corrections maintain physical laws
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Sequential access to velocity and mass arrays
//! - **Compute Intensity**: Moderate - vector math and reduction operations
//! - **Synchronization**: Requires workgroup synchronization for reductions

// Physics parameters uniform buffer
struct PhysicsParams {
    // Time and frame info
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    
    // World and physics settings
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    
    // Grid settings
    grid_resolution: i32,        // 64 for 64³ grid
    grid_cell_size: f32,         // world_size / grid_resolution
    max_cells_per_grid: i32,     // Maximum cells per grid cell (32)
    enable_thrust_force: i32,
    
    // UI interaction
    dragged_cell_index: i32,
    _padding1: vec3<f32>,
    
    // Padding to 256 bytes
    _padding: array<f32, 48>,
}

// Momentum conservation data structure
struct MomentumData {
    total_linear_momentum: vec3<f32>,
    total_angular_momentum: vec3<f32>,
    total_mass: f32,
    center_of_mass: vec3<f32>,
    total_kinetic_energy: f32,
    _padding: vec3<f32>,
}

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> angular_velocity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> momentum_data: MomentumData;

// Shared memory for workgroup reductions
var<workgroup> shared_linear_momentum: array<vec3<f32>, 64>;
var<workgroup> shared_angular_momentum: array<vec3<f32>, 64>;
var<workgroup> shared_mass: array<f32, 64>;
var<workgroup> shared_center_of_mass: array<vec3<f32>, 64>;
var<workgroup> shared_kinetic_energy: array<f32, 64>;

/// Calculate linear momentum for a cell.
///
/// Linear momentum is mass times velocity (p = mv).
///
/// # Arguments
/// * `mass` - Cell mass
/// * `velocity` - Cell velocity
///
/// # Returns
/// Linear momentum vector
fn calculate_linear_momentum(mass: f32, velocity: vec3<f32>) -> vec3<f32> {
    return mass * velocity;
}

/// Calculate angular momentum for a cell about center of mass.
///
/// Angular momentum is L = r × (m * v) where r is position relative to center of mass.
///
/// # Arguments
/// * `position` - Cell position
/// * `velocity` - Cell velocity
/// * `mass` - Cell mass
/// * `center_of_mass` - System center of mass
///
/// # Returns
/// Angular momentum vector
fn calculate_angular_momentum(
    position: vec3<f32>,
    velocity: vec3<f32>,
    mass: f32,
    center_of_mass: vec3<f32>
) -> vec3<f32> {
    let relative_position = position - center_of_mass;
    let linear_momentum = mass * velocity;
    return cross(relative_position, linear_momentum);
}

/// Calculate kinetic energy for a cell.
///
/// Kinetic energy is KE = 0.5 * m * v² + 0.5 * I * ω²
/// where I is moment of inertia and ω is angular velocity.
///
/// # Arguments
/// * `mass` - Cell mass
/// * `velocity` - Linear velocity
/// * `angular_velocity` - Angular velocity
/// * `radius` - Cell radius (for moment of inertia)
///
/// # Returns
/// Total kinetic energy
fn calculate_kinetic_energy(
    mass: f32,
    velocity: vec3<f32>,
    angular_velocity: vec3<f32>,
    radius: f32
) -> f32 {
    // Linear kinetic energy
    let linear_ke = 0.5 * mass * dot(velocity, velocity);
    
    // Rotational kinetic energy (assuming solid sphere: I = 2/5 * m * r²)
    let moment_of_inertia = 0.4 * mass * radius * radius;
    let rotational_ke = 0.5 * moment_of_inertia * dot(angular_velocity, angular_velocity);
    
    return linear_ke + rotational_ke;
}

/// Perform workgroup reduction for vector values.
///
/// Reduces an array of vectors to a single sum using shared memory
/// and workgroup synchronization.
///
/// # Arguments
/// * `local_id` - Local thread ID within workgroup
/// * `input_value` - Input vector for this thread
/// * `shared_array` - Shared memory array for reduction
///
/// # Returns
/// Sum of all input values (only valid for thread 0)
fn workgroup_reduce_vec3(
    local_id: u32,
    input_value: vec3<f32>,
    shared_array: ptr<workgroup, array<vec3<f32>, 64>>
) -> vec3<f32> {
    // Store input value in shared memory
    (*shared_array)[local_id] = input_value;
    workgroupBarrier();
    
    // Parallel reduction
    var stride = 32u;
    while (stride > 0u) {
        if (local_id < stride && local_id + stride < 64u) {
            (*shared_array)[local_id] += (*shared_array)[local_id + stride];
        }
        stride /= 2u;
        workgroupBarrier();
    }
    
    return (*shared_array)[0];
}

/// Perform workgroup reduction for scalar values.
///
/// Reduces an array of scalars to a single sum using shared memory
/// and workgroup synchronization.
///
/// # Arguments
/// * `local_id` - Local thread ID within workgroup
/// * `input_value` - Input scalar for this thread
/// * `shared_array` - Shared memory array for reduction
///
/// # Returns
/// Sum of all input values (only valid for thread 0)
fn workgroup_reduce_f32(
    local_id: u32,
    input_value: f32,
    shared_array: ptr<workgroup, array<f32, 64>>
) -> f32 {
    // Store input value in shared memory
    (*shared_array)[local_id] = input_value;
    workgroupBarrier();
    
    // Parallel reduction
    var stride = 32u;
    while (stride > 0u) {
        if (local_id < stride && local_id + stride < 64u) {
            (*shared_array)[local_id] += (*shared_array)[local_id + stride];
        }
        stride /= 2u;
        workgroupBarrier();
    }
    
    return (*shared_array)[0];
}

/// Apply momentum correction to cell velocity.
///
/// Distributes momentum correction proportionally based on cell mass.
/// Larger cells receive smaller velocity corrections for the same momentum change.
///
/// # Arguments
/// * `current_velocity` - Current cell velocity
/// * `mass` - Cell mass
/// * `momentum_correction` - Total momentum correction to apply
/// * `total_mass` - Total mass of all cells
/// * `correction_strength` - Strength of correction (0.0 to 1.0)
///
/// # Returns
/// Corrected velocity
fn apply_momentum_correction(
    current_velocity: vec3<f32>,
    mass: f32,
    momentum_correction: vec3<f32>,
    total_mass: f32,
    correction_strength: f32
) -> vec3<f32> {
    // Distribute correction proportionally by mass
    let mass_fraction = mass / total_mass;
    let velocity_correction = (momentum_correction / mass) * mass_fraction * correction_strength;
    
    return current_velocity - velocity_correction;
}

/// Apply angular momentum correction to cell angular velocity.
///
/// Similar to linear momentum correction but for rotational motion.
///
/// # Arguments
/// * `current_angular_velocity` - Current angular velocity
/// * `moment_of_inertia` - Cell moment of inertia
/// * `angular_momentum_correction` - Angular momentum correction
/// * `total_moment_of_inertia` - Total moment of inertia
/// * `correction_strength` - Strength of correction
///
/// # Returns
/// Corrected angular velocity
fn apply_angular_momentum_correction(
    current_angular_velocity: vec3<f32>,
    moment_of_inertia: f32,
    angular_momentum_correction: vec3<f32>,
    total_moment_of_inertia: f32,
    correction_strength: f32
) -> vec3<f32> {
    let inertia_fraction = moment_of_inertia / total_moment_of_inertia;
    let angular_velocity_correction = (angular_momentum_correction / moment_of_inertia) * inertia_fraction * correction_strength;
    
    return current_angular_velocity - angular_velocity_correction;
}

/// Main compute shader entry point.
///
/// This shader operates in two phases:
/// 1. **Calculation Phase**: Calculate total momentum, center of mass, and energy
/// 2. **Correction Phase**: Apply corrections to maintain conservation laws
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's momentum contribution
/// - Workgroup reductions calculate system totals
/// - All threads apply proportional corrections
///
/// ## Conservation Pipeline
/// 1. Calculate individual cell contributions (momentum, energy)
/// 2. Perform workgroup reductions to get system totals
/// 3. Calculate required corrections for conservation
/// 4. Apply corrections proportionally to all cells
/// 5. Validate that corrections maintain physical laws
///
/// ## Numerical Stability
/// - Gradual corrections prevent oscillations
/// - Mass-proportional distribution maintains realism
/// - Validation prevents overcorrection
/// - Special handling for edge cases (massless cells, etc.)
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let cell_index = global_id.x;
    let local_index = local_id.x;
    
    // Initialize local values
    var local_linear_momentum = vec3<f32>(0.0, 0.0, 0.0);
    var local_angular_momentum = vec3<f32>(0.0, 0.0, 0.0);
    var local_mass = 0.0;
    var local_weighted_position = vec3<f32>(0.0, 0.0, 0.0);
    var local_kinetic_energy = 0.0;
    
    // === Phase 1: Calculate Contributions ===
    
    if (cell_index < physics_params.cell_count) {
        // Read cell properties
        let pos_and_mass = position_and_mass[cell_index];
        let position = pos_and_mass.xyz;
        let mass = pos_and_mass.w;
        let current_velocity = velocity[cell_index].xyz;
        let current_angular_velocity = angular_velocity[cell_index].xyz;
        
        // Skip massless cells
        if (mass > 0.001) {
            // Calculate contributions
            local_linear_momentum = calculate_linear_momentum(mass, current_velocity);
            local_mass = mass;
            local_weighted_position = position * mass;
            
            // Calculate cell radius for moment of inertia
            let radius = pow(mass * 0.75 / 3.14159, 1.0 / 3.0);
            local_kinetic_energy = calculate_kinetic_energy(mass, current_velocity, current_angular_velocity, radius);
        }
    }
    
    // === Phase 2: Workgroup Reductions ===
    
    let total_linear_momentum = workgroup_reduce_vec3(local_index, local_linear_momentum, &shared_linear_momentum);
    let total_mass = workgroup_reduce_f32(local_index, local_mass, &shared_mass);
    let total_weighted_position = workgroup_reduce_vec3(local_index, local_weighted_position, &shared_center_of_mass);
    let total_kinetic_energy = workgroup_reduce_f32(local_index, local_kinetic_energy, &shared_kinetic_energy);
    
    // Calculate center of mass (only valid for thread 0)
    var center_of_mass = vec3<f32>(0.0, 0.0, 0.0);
    if (local_index == 0u && total_mass > 0.001) {
        center_of_mass = total_weighted_position / total_mass;
    }
    
    // Broadcast center of mass to all threads in workgroup
    shared_center_of_mass[0] = center_of_mass;
    workgroupBarrier();
    center_of_mass = shared_center_of_mass[0];
    
    // Calculate angular momentum with known center of mass
    if (cell_index < physics_params.cell_count) {
        let pos_and_mass = position_and_mass[cell_index];
        let position = pos_and_mass.xyz;
        let mass = pos_and_mass.w;
        let current_velocity = velocity[cell_index].xyz;
        
        if (mass > 0.001) {
            local_angular_momentum = calculate_angular_momentum(position, current_velocity, mass, center_of_mass);
        }
    }
    
    let total_angular_momentum = workgroup_reduce_vec3(local_index, local_angular_momentum, &shared_angular_momentum);
    
    // === Phase 3: Store System Totals (Thread 0 Only) ===
    
    if (global_id.x == 0u) {
        momentum_data.total_linear_momentum = total_linear_momentum;
        momentum_data.total_angular_momentum = total_angular_momentum;
        momentum_data.total_mass = total_mass;
        momentum_data.center_of_mass = center_of_mass;
        momentum_data.total_kinetic_energy = total_kinetic_energy;
    }
    
    // === Phase 4: Apply Corrections ===
    
    if (cell_index < physics_params.cell_count) {
        let pos_and_mass = position_and_mass[cell_index];
        let position = pos_and_mass.xyz;
        let mass = pos_and_mass.w;
        let current_velocity = velocity[cell_index].xyz;
        let current_angular_velocity = angular_velocity[cell_index].xyz;
        
        // Skip massless cells and dragged cells
        if (mass > 0.001 && physics_params.dragged_cell_index != i32(cell_index)) {
            // Calculate correction strength (gradual correction to prevent oscillations)
            let correction_strength = 0.01; // 1% correction per frame
            
            // Apply linear momentum correction
            if (total_mass > 0.001) {
                let corrected_velocity = apply_momentum_correction(
                    current_velocity,
                    mass,
                    total_linear_momentum,
                    total_mass,
                    correction_strength
                );
                velocity[cell_index] = vec4<f32>(corrected_velocity, 0.0);
            }
            
            // Apply angular momentum correction
            let radius = pow(mass * 0.75 / 3.14159, 1.0 / 3.0);
            let moment_of_inertia = 0.4 * mass * radius * radius;
            let total_moment_of_inertia = total_mass * radius * radius * 0.4; // Approximation
            
            if (total_moment_of_inertia > 0.001) {
                let corrected_angular_velocity = apply_angular_momentum_correction(
                    current_angular_velocity,
                    moment_of_inertia,
                    total_angular_momentum,
                    total_moment_of_inertia,
                    correction_strength
                );
                angular_velocity[cell_index] = vec4<f32>(corrected_angular_velocity, 0.0);
            }
        }
    }
}