//! # Cell Physics Spatial Compute Shader
//!
//! This compute shader implements collision detection and force calculation using
//! spatial grid acceleration. It processes cell-cell collisions efficiently by
//! only checking cells within the same or neighboring grid cells.
//!
//! ## Algorithm
//! 1. Each thread processes one cell
//! 2. Find the cell's grid position and neighboring grid cells
//! 3. Iterate through all cells in neighboring grid cells
//! 4. Calculate collision forces for overlapping cells
//! 5. Apply repulsion forces to separate overlapping cells
//!
//! ## Collision Detection
//! - Uses spatial grid for O(n) collision detection instead of O(n²)
//! - Checks 27 neighboring grid cells (3x3x3 neighborhood)
//! - Calculates sphere-sphere collision with radius-based overlap
//! - Applies spring-damper forces for realistic collision response
//!
//! ## Force Calculation
//! - **Collision Forces**: Repulsion between overlapping cells
//! - **Boundary Forces**: Keep cells within simulation bounds
//! - **Swim Forces**: Directional thrust for Flagellocyte cells
//! - **Force Clamping**: Numerical stability through force limits
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Spatial locality through grid-based iteration
//! - **Cache Efficiency**: Neighboring cells likely in same cache lines
//! - **Scalability**: O(n) complexity instead of O(n²) brute force

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
    
    // Padding to 256 bytes (using vec4<f32> for proper 16-byte alignment)
    _padding: array<vec4<f32>, 12>,
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
    _mode_padding1: f32,
    _mode_padding2: f32,
    _mode_padding3: f32,
}

// Bind group 0: Physics parameters and simulation buffers
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acceleration: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> orientation: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(6) var<storage, read> genome_modes: array<GpuMode>;

// Bind group 1: Spatial grid buffers
@group(1) @binding(0) var<storage, read> spatial_grid_counts: array<u32>;
@group(1) @binding(1) var<storage, read> spatial_grid_offsets: array<u32>;
@group(1) @binding(2) var<storage, read> spatial_grid_indices: array<u32>;
@group(1) @binding(3) var<storage, read> spatial_grid_assignments: array<u32>;

/// Convert world position to grid coordinates.
///
/// # Arguments
/// * `world_pos` - Position in world space
/// * `world_size` - Size of the simulation world
/// * `grid_resolution` - Number of grid cells per dimension
///
/// # Returns
/// Grid coordinates as vec3<i32> clamped to valid range
fn world_to_grid(world_pos: vec3<f32>, world_size: f32, grid_resolution: i32) -> vec3<i32> {
    let half_world = world_size * 0.5;
    let normalized_pos = (world_pos + vec3<f32>(half_world)) / world_size;
    let grid_pos = normalized_pos * f32(grid_resolution);
    
    return vec3<i32>(
        clamp(i32(grid_pos.x), 0, grid_resolution - 1),
        clamp(i32(grid_pos.y), 0, grid_resolution - 1),
        clamp(i32(grid_pos.z), 0, grid_resolution - 1)
    );
}

/// Convert 3D grid coordinates to linear grid index.
///
/// # Arguments
/// * `grid_coords` - 3D grid coordinates
/// * `grid_resolution` - Number of grid cells per dimension
///
/// # Returns
/// Linear grid index for array access
fn grid_coords_to_index(grid_coords: vec3<i32>, grid_resolution: i32) -> u32 {
    return u32(grid_coords.x + grid_coords.y * grid_resolution + grid_coords.z * grid_resolution * grid_resolution);
}

/// Calculate collision force between two cells.
///
/// Uses spring-damper mechanics to generate realistic collision response.
/// The force magnitude depends on overlap amount and relative velocity.
///
/// # Arguments
/// * `pos_a` - Position of first cell
/// * `pos_b` - Position of second cell
/// * `vel_a` - Velocity of first cell
/// * `vel_b` - Velocity of second cell
/// * `radius_a` - Radius of first cell
/// * `radius_b` - Radius of second cell
///
/// # Returns
/// Force vector to apply to first cell (negate for second cell)
fn calculate_collision_force(
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    vel_a: vec3<f32>,
    vel_b: vec3<f32>,
    radius_a: f32,
    radius_b: f32
) -> vec3<f32> {
    let delta_pos = pos_a - pos_b;
    let distance = length(delta_pos);
    let min_distance = radius_a + radius_b;
    
    // No collision if cells don't overlap
    if (distance >= min_distance || distance < 0.001) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Calculate overlap and normal direction
    let overlap = min_distance - distance;
    let normal = normalize(delta_pos);
    
    // Spring force proportional to overlap
    let spring_constant = 1000.0; // Adjustable collision stiffness
    let spring_force = spring_constant * overlap;
    
    // Damping force proportional to relative velocity
    let relative_velocity = vel_a - vel_b;
    let normal_velocity = dot(relative_velocity, normal);
    let damping_constant = 50.0; // Adjustable collision damping
    let damping_force = damping_constant * normal_velocity;
    
    // Total collision force
    let total_force_magnitude = spring_force + damping_force;
    return normal * total_force_magnitude;
}

/// Calculate boundary force to keep cell within simulation bounds.
///
/// Applies repulsion forces when cells approach the world boundaries.
/// Uses exponential force increase near boundaries for smooth containment.
///
/// # Arguments
/// * `position` - Cell position
/// * `world_size` - Size of simulation world
/// * `boundary_stiffness` - Strength of boundary forces
///
/// # Returns
/// Force vector to push cell away from boundaries
fn calculate_boundary_force(position: vec3<f32>, world_size: f32, boundary_stiffness: f32) -> vec3<f32> {
    let half_world = world_size * 0.5;
    let boundary_margin = world_size * 0.05; // 5% margin for force activation
    var force = vec3<f32>(0.0, 0.0, 0.0);
    
    // X boundaries
    if (position.x > half_world - boundary_margin) {
        let penetration = position.x - (half_world - boundary_margin);
        force.x -= boundary_stiffness * penetration * penetration;
    } else if (position.x < -half_world + boundary_margin) {
        let penetration = (-half_world + boundary_margin) - position.x;
        force.x += boundary_stiffness * penetration * penetration;
    }
    
    // Y boundaries
    if (position.y > half_world - boundary_margin) {
        let penetration = position.y - (half_world - boundary_margin);
        force.y -= boundary_stiffness * penetration * penetration;
    } else if (position.y < -half_world + boundary_margin) {
        let penetration = (-half_world + boundary_margin) - position.y;
        force.y += boundary_stiffness * penetration * penetration;
    }
    
    // Z boundaries
    if (position.z > half_world - boundary_margin) {
        let penetration = position.z - (half_world - boundary_margin);
        force.z -= boundary_stiffness * penetration * penetration;
    } else if (position.z < -half_world + boundary_margin) {
        let penetration = (-half_world + boundary_margin) - position.z;
        force.z += boundary_stiffness * penetration * penetration;
    }
    
    return force;
}

/// Calculate swim force for Flagellocyte cells.
///
/// Applies directional thrust based on cell orientation and genome settings.
/// Only active for cells with flagellocyte_thrust_force > 0.
///
/// # Arguments
/// * `orientation` - Cell orientation quaternion
/// * `mode` - Genome mode with thrust settings
///
/// # Returns
/// Force vector in cell's forward direction
fn calculate_swim_force(orientation: vec4<f32>, mode: GpuMode) -> vec3<f32> {
    if (mode.flagellocyte_thrust_force <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Convert quaternion to forward direction
    // Forward direction is typically (0, 0, 1) rotated by quaternion
    let quat = orientation; // (w, x, y, z)
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;
    
    // Rotate forward vector (0, 0, 1) by quaternion
    let forward = vec3<f32>(
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y)
    );
    
    return normalize(forward) * mode.flagellocyte_thrust_force;
}

/// Main compute shader entry point.
///
/// Each thread processes one cell and calculates all forces acting on it:
/// collision forces from nearby cells, boundary forces, and swim forces.
/// Uses spatial grid acceleration for efficient collision detection.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's force calculation
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Spatial Grid Traversal
/// - Checks 27 neighboring grid cells (3x3x3 neighborhood)
/// - Only processes cells within collision range
/// - Avoids O(n²) brute force collision detection
/// - Maintains spatial locality for cache efficiency
///
/// ## Force Accumulation
/// - Accumulates all forces into acceleration buffer
/// - Applies force clamping for numerical stability
/// - Handles special cases (dragged cells, boundary conditions)
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
    let cell_velocity = velocity[cell_index].xyz;
    let cell_orientation = orientation[cell_index];
    let mode_index = mode_indices[cell_index];
    
    // Calculate cell radius from mass (assuming spherical cells)
    let radius = pow(mass * 0.75 / 3.14159, 1.0 / 3.0); // Volume to radius conversion
    
    // Get genome mode for behavior settings
    let mode = genome_modes[mode_index];
    
    // Initialize force accumulator
    var total_force = vec3<f32>(0.0, 0.0, 0.0);
    
    // === Collision Detection Using Spatial Grid ===
    
    // Find this cell's grid position
    let grid_coords = world_to_grid(position, physics_params.world_size, physics_params.grid_resolution);
    
    // Check all 27 neighboring grid cells (3x3x3 neighborhood)
    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dz: i32 = -1; dz <= 1; dz++) {
                let neighbor_coords = grid_coords + vec3<i32>(dx, dy, dz);
                
                // Skip if neighbor is outside grid bounds
                if (neighbor_coords.x < 0 || neighbor_coords.x >= physics_params.grid_resolution ||
                    neighbor_coords.y < 0 || neighbor_coords.y >= physics_params.grid_resolution ||
                    neighbor_coords.z < 0 || neighbor_coords.z >= physics_params.grid_resolution) {
                    continue;
                }
                
                let neighbor_grid_index = grid_coords_to_index(neighbor_coords, physics_params.grid_resolution);
                let cell_count_in_grid = spatial_grid_counts[neighbor_grid_index];
                let grid_start_offset = spatial_grid_offsets[neighbor_grid_index];
                
                // Iterate through all cells in this grid cell
                for (var i: u32 = 0u; i < cell_count_in_grid; i++) {
                    let other_cell_index = spatial_grid_indices[grid_start_offset + i];
                    
                    // Skip self-collision
                    if (other_cell_index == cell_index) {
                        continue;
                    }
                    
                    // Read other cell properties
                    let other_pos_and_mass = position_and_mass[other_cell_index];
                    let other_position = other_pos_and_mass.xyz;
                    let other_mass = other_pos_and_mass.w;
                    let other_velocity = velocity[other_cell_index].xyz;
                    let other_radius = pow(other_mass * 0.75 / 3.14159, 1.0 / 3.0);
                    
                    // Calculate collision force
                    let collision_force = calculate_collision_force(
                        position, other_position,
                        cell_velocity, other_velocity,
                        radius, other_radius
                    );
                    
                    total_force += collision_force;
                }
            }
        }
    }
    
    // === Boundary Forces ===
    let boundary_force = calculate_boundary_force(
        position,
        physics_params.world_size,
        physics_params.boundary_stiffness
    );
    total_force += boundary_force;
    
    // === Swim Forces (Flagellocyte cells) ===
    if (physics_params.enable_thrust_force != 0) {
        let swim_force = calculate_swim_force(cell_orientation, mode);
        total_force += swim_force;
    }
    
    // === Gravity ===
    total_force.y -= physics_params.gravity * mass;
    
    // === Force Clamping for Numerical Stability ===
    let max_force = 10000.0; // Adjustable maximum force magnitude
    let force_magnitude = length(total_force);
    if (force_magnitude > max_force) {
        total_force = normalize(total_force) * max_force;
    }
    
    // === Handle Dragged Cell (UI Interaction) ===
    if (physics_params.dragged_cell_index == i32(cell_index)) {
        // Dragged cells have reduced physics forces
        total_force *= 0.1;
    }
    
    // === Convert Force to Acceleration ===
    let cell_acceleration = total_force / mass;
    
    // Apply acceleration damping for stability
    let damped_acceleration = cell_acceleration * (1.0 - physics_params.acceleration_damping * physics_params.delta_time);
    
    // Store final acceleration
    acceleration[cell_index] = vec4<f32>(damped_acceleration, 0.0);
}