//! # Force Calculation Compute Shader
//!
//! This compute shader implements additional force calculations that complement
//! collision detection. It focuses on boundary forces, swim forces, and other
//! non-collision forces that affect cell motion.
//!
//! ## Force Types
//! - **Boundary Forces**: Exponential repulsion near simulation boundaries
//! - **Swim Forces**: Directional thrust for Flagellocyte cells
//! - **Gravity Forces**: Downward acceleration based on cell mass
//! - **Drag Forces**: Velocity-dependent resistance for realistic motion
//!
//! ## Force Accumulation
//! - Reads existing acceleration from collision detection
//! - Adds additional forces to the acceleration buffer
//! - Applies force clamping for numerical stability
//! - Handles special cases (dragged cells, disabled forces)
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Sequential access to cell property arrays
//! - **Compute Intensity**: Moderate - mostly vector math operations
//! - **Cache Efficiency**: Good spatial locality in array access

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

// GPU Mode structure for genome-based behavior
struct GpuMode {
    // Visual properties (16 bytes each)
    color: vec4<f32>,
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    split_direction: vec4<f32>,
    
    // Child modes (8 bytes)
    child_modes: vec2<i32>,
    
    // Division properties (8 bytes)
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
    _adhesion_padding: i32,
    
    // Adhesion behavior (16 bytes)
    parent_make_adhesion: i32,
    child_a_keep_adhesion: i32,
    child_b_keep_adhesion: i32,
    max_adhesions: i32,
    
    // Flagellocyte properties (16 bytes)
    flagellocyte_thrust_force: f32,
    cell_type: i32,              // 0 = Test, 1 = Flagellocyte
    nutrient_consumption_rate: f32,
    _mode_padding3: f32,
}

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acceleration: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> orientation: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(6) var<storage, read> genome_modes: array<GpuMode>;
@group(0) @binding(7) var<storage, read> nitrates: array<f32>;

/// Calculate boundary force to keep cell within simulation bounds.
///
/// Uses exponential force increase near boundaries for smooth containment.
/// The force magnitude increases quadratically as cells approach boundaries.
///
/// # Arguments
/// * `position` - Cell position in world space
/// * `world_size` - Size of simulation world
/// * `boundary_stiffness` - Strength of boundary forces
///
/// # Returns
/// Force vector to push cell away from boundaries
fn calculate_boundary_force(position: vec3<f32>, world_size: f32, boundary_stiffness: f32) -> vec3<f32> {
    let half_world = world_size * 0.5;
    let boundary_margin = world_size * 0.1; // 10% margin for force activation
    var force = vec3<f32>(0.0, 0.0, 0.0);
    
    // X boundaries (left and right walls)
    if (position.x > half_world - boundary_margin) {
        let penetration = position.x - (half_world - boundary_margin);
        let normalized_penetration = penetration / boundary_margin;
        force.x -= boundary_stiffness * normalized_penetration * normalized_penetration;
    } else if (position.x < -half_world + boundary_margin) {
        let penetration = (-half_world + boundary_margin) - position.x;
        let normalized_penetration = penetration / boundary_margin;
        force.x += boundary_stiffness * normalized_penetration * normalized_penetration;
    }
    
    // Y boundaries (top and bottom walls)
    if (position.y > half_world - boundary_margin) {
        let penetration = position.y - (half_world - boundary_margin);
        let normalized_penetration = penetration / boundary_margin;
        force.y -= boundary_stiffness * normalized_penetration * normalized_penetration;
    } else if (position.y < -half_world + boundary_margin) {
        let penetration = (-half_world + boundary_margin) - position.y;
        let normalized_penetration = penetration / boundary_margin;
        force.y += boundary_stiffness * normalized_penetration * normalized_penetration;
    }
    
    // Z boundaries (front and back walls)
    if (position.z > half_world - boundary_margin) {
        let penetration = position.z - (half_world - boundary_margin);
        let normalized_penetration = penetration / boundary_margin;
        force.z -= boundary_stiffness * normalized_penetration * normalized_penetration;
    } else if (position.z < -half_world + boundary_margin) {
        let penetration = (-half_world + boundary_margin) - position.z;
        let normalized_penetration = penetration / boundary_margin;
        force.z += boundary_stiffness * normalized_penetration * normalized_penetration;
    }
    
    return force;
}

/// Calculate swim force for Flagellocyte cells.
///
/// Applies directional thrust based on cell orientation and genome settings.
/// The thrust force is modulated by nutrient levels - cells with low nutrients
/// have reduced swimming ability.
///
/// # Arguments
/// * `orientation` - Cell orientation quaternion (w, x, y, z)
/// * `mode` - Genome mode with thrust settings
/// * `nutrient_level` - Current nutrient level (affects thrust strength)
///
/// # Returns
/// Force vector in cell's forward direction
fn calculate_swim_force(orientation: vec4<f32>, mode: GpuMode, nutrient_level: f32) -> vec3<f32> {
    // Only Flagellocyte cells (cell_type == 1) can swim
    if (mode.cell_type != 1 || mode.flagellocyte_thrust_force <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Convert quaternion to forward direction vector
    // Quaternion format: (w, x, y, z) where w is the scalar part
    let quat = orientation;
    let w = quat.x; // w component
    let x = quat.y; // x component  
    let y = quat.z; // y component
    let z = quat.w; // z component
    
    // Rotate forward vector (0, 0, 1) by quaternion
    // This gives the cell's forward direction in world space
    let forward = vec3<f32>(
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y)
    );
    
    // Normalize to ensure unit vector
    let forward_normalized = normalize(forward);
    
    // Modulate thrust by nutrient level
    // Cells with low nutrients have reduced swimming ability
    let nutrient_factor = clamp(nutrient_level / 100.0, 0.1, 1.0); // Assume 100 is full nutrients
    let effective_thrust = mode.flagellocyte_thrust_force * nutrient_factor;
    
    return forward_normalized * effective_thrust;
}

/// Calculate drag force based on velocity.
///
/// Applies velocity-dependent resistance to simulate fluid drag.
/// The drag force opposes motion and increases with velocity squared.
///
/// # Arguments
/// * `velocity` - Current cell velocity
/// * `radius` - Cell radius (affects drag coefficient)
///
/// # Returns
/// Drag force vector opposing motion
fn calculate_drag_force(velocity: vec3<f32>, radius: f32) -> vec3<f32> {
    let speed = length(velocity);
    
    // No drag for stationary cells
    if (speed < 0.001) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Drag coefficient based on cell size (larger cells have more drag)
    let drag_coefficient = 0.1 * radius * radius;
    
    // Quadratic drag: F = -k * v * |v|
    let drag_magnitude = drag_coefficient * speed * speed;
    let drag_direction = -normalize(velocity);
    
    return drag_direction * drag_magnitude;
}

/// Calculate buoyancy force for cells in fluid.
///
/// Simulates buoyancy effects based on cell density relative to fluid.
/// Cells with lower density than fluid experience upward buoyancy.
///
/// # Arguments
/// * `mass` - Cell mass
/// * `radius` - Cell radius
///
/// # Returns
/// Buoyancy force vector (typically upward)
fn calculate_buoyancy_force(mass: f32, radius: f32) -> vec3<f32> {
    // Calculate cell volume (assuming spherical)
    let volume = (4.0 / 3.0) * 3.14159 * radius * radius * radius;
    
    // Calculate cell density
    let cell_density = mass / volume;
    
    // Assume fluid density (water-like)
    let fluid_density = 1000.0; // kg/m³
    
    // Buoyancy force: F = (fluid_density - cell_density) * volume * gravity
    let density_difference = fluid_density - cell_density;
    let buoyancy_magnitude = density_difference * volume * 9.81; // Standard gravity
    
    // Buoyancy acts upward
    return vec3<f32>(0.0, buoyancy_magnitude, 0.0);
}

/// Apply force clamping for numerical stability.
///
/// Prevents extreme forces that could cause numerical instability or
/// unrealistic cell behavior. Uses both magnitude and component limits.
///
/// # Arguments
/// * `force` - Input force vector
/// * `max_magnitude` - Maximum allowed force magnitude
///
/// # Returns
/// Clamped force vector
fn clamp_force(force: vec3<f32>, max_magnitude: f32) -> vec3<f32> {
    let magnitude = length(force);
    
    // Clamp magnitude if too large
    if (magnitude > max_magnitude) {
        return normalize(force) * max_magnitude;
    }
    
    // Also clamp individual components to prevent extreme values
    let max_component = max_magnitude * 0.8; // Allow some headroom
    return vec3<f32>(
        clamp(force.x, -max_component, max_component),
        clamp(force.y, -max_component, max_component),
        clamp(force.z, -max_component, max_component)
    );
}

/// Main compute shader entry point.
///
/// Each thread processes one cell and calculates additional forces beyond
/// collision detection. These forces are added to the existing acceleration
/// from collision detection to create the complete force picture.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's additional forces
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Force Processing Pipeline
/// 1. Read existing acceleration from collision detection
/// 2. Calculate boundary forces to contain cells
/// 3. Calculate swim forces for Flagellocyte cells
/// 4. Calculate environmental forces (drag, buoyancy)
/// 5. Apply gravity based on cell mass
/// 6. Accumulate all forces and apply clamping
/// 7. Handle special cases (dragged cells, disabled forces)
///
/// ## Numerical Stability
/// - Force clamping prevents extreme accelerations
/// - Acceleration damping reduces oscillations
/// - Special handling for edge cases (zero mass, infinite forces)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond the active cell count
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Read cell properties
    let pos_and_mass = position_and_mass[cell_index];
    let position = pos_and_mass.xyz;
    let mass = pos_and_mass.w;
    let velocity = velocity[cell_index].xyz;
    let cell_orientation = orientation[cell_index];
    let mode_index = mode_indices[cell_index];
    let nutrient_level = nitrates[cell_index];
    
    // Calculate cell radius from mass (assuming spherical cells)
    let radius = pow(mass * 0.75 / 3.14159, 1.0 / 3.0); // Volume to radius conversion
    
    // Get genome mode for behavior settings
    let mode = genome_modes[mode_index];
    
    // Read existing acceleration from collision detection
    var total_acceleration = acceleration[cell_index].xyz;
    
    // Initialize additional force accumulator
    var additional_force = vec3<f32>(0.0, 0.0, 0.0);
    
    // === Boundary Forces ===
    let boundary_force = calculate_boundary_force(
        position,
        physics_params.world_size,
        physics_params.boundary_stiffness
    );
    additional_force += boundary_force;
    
    // === Swim Forces (Flagellocyte cells only) ===
    if (physics_params.enable_thrust_force != 0) {
        let swim_force = calculate_swim_force(cell_orientation, mode, nutrient_level);
        additional_force += swim_force;
    }
    
    // === Environmental Forces ===
    
    // Drag force (velocity-dependent resistance)
    let drag_force = calculate_drag_force(velocity, radius);
    additional_force += drag_force;
    
    // Buoyancy force (density-dependent upward force)
    let buoyancy_force = calculate_buoyancy_force(mass, radius);
    additional_force += buoyancy_force * 0.1; // Scale down for subtle effect
    
    // === Gravity ===
    // Apply gravitational acceleration (F = mg)
    additional_force.y -= physics_params.gravity * mass;
    
    // === Force Clamping for Numerical Stability ===
    let max_force = 5000.0; // Maximum force magnitude per cell
    additional_force = clamp_force(additional_force, max_force);
    
    // === Handle Special Cases ===
    
    // Dragged cells have modified physics
    if (physics_params.dragged_cell_index == i32(cell_index)) {
        // Reduce all forces for dragged cells to make them more controllable
        additional_force *= 0.2;
        // Also reduce existing collision forces
        total_acceleration *= 0.5;
    }
    
    // Prevent division by zero for massless cells
    if (mass < 0.001) {
        additional_force = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // === Convert Additional Forces to Acceleration ===
    let additional_acceleration = additional_force / mass;
    
    // === Apply Acceleration Damping ===
    // Damping reduces oscillations and improves stability
    let damping_factor = 1.0 - physics_params.acceleration_damping * physics_params.delta_time;
    let damped_additional_acceleration = additional_acceleration * damping_factor;
    
    // === Accumulate Total Acceleration ===
    total_acceleration += damped_additional_acceleration;
    
    // === Final Acceleration Clamping ===
    let max_acceleration = 1000.0; // Maximum total acceleration magnitude
    let acceleration_magnitude = length(total_acceleration);
    if (acceleration_magnitude > max_acceleration) {
        total_acceleration = normalize(total_acceleration) * max_acceleration;
    }
    
    // === Store Final Acceleration ===
    acceleration[cell_index] = vec4<f32>(total_acceleration, 0.0);
}