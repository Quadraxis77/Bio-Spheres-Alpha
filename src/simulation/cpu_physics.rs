// CPU-based physics simulation
// Matches the BioSpheres-Q reference implementation

use glam::{Vec3, Quat, IVec3, UVec3};
use std::collections::HashMap;
use crate::simulation::canonical_state::{CanonicalState, DivisionEvent};
use crate::simulation::physics_config::PhysicsConfig;
use crate::genome::Genome;
use crate::cell::division;

// ============================================================================
// Spatial Grid
// ============================================================================

/// Deterministic spatial grid using fixed-size arrays and prefix-sum algorithm
/// This provides O(1) cell lookups with zero allocations per tick
#[derive(Clone)]
pub struct DeterministicSpatialGrid {
    pub grid_dimensions: UVec3,
    pub world_size: f32,
    pub cell_size: f32,
    pub sphere_radius: f32,

    /// Pre-computed active cells (within sphere)
    pub active_cells: Vec<IVec3>,

    /// HashMap for O(1) lookup of active cell index
    pub active_cell_map: HashMap<IVec3, usize>,

    /// Cell contents (flat array with prefix sums)
    pub cell_contents: Vec<usize>,

    /// Prefix sum offsets for each active grid cell
    pub cell_offsets: Vec<usize>,

    /// Counts per grid cell (used during rebuild)
    pub cell_counts: Vec<usize>,

    /// Track which grid cells were used in last rebuild
    pub used_grid_cells: Vec<usize>,
}

impl DeterministicSpatialGrid {
    /// Create a new deterministic spatial grid
    pub fn new(grid_dim: u32, world_size: f32, sphere_radius: f32) -> Self {
        Self::with_capacity(grid_dim, world_size, sphere_radius, 10_000)
    }
    
    /// Create a new deterministic spatial grid with specified cell capacity
    pub fn with_capacity(grid_dim: u32, world_size: f32, sphere_radius: f32, max_cells: usize) -> Self {
        let grid_dimensions = UVec3::splat(grid_dim);
        let cell_size = world_size / grid_dim as f32;

        // Precompute active grid cells
        let active_cells = Self::precompute_active_cells(grid_dimensions);
        let active_count = active_cells.len();

        // Build HashMap for O(1) lookups
        let mut active_cell_map = HashMap::new();
        for (idx, &coord) in active_cells.iter().enumerate() {
            active_cell_map.insert(coord, idx);
        }

        Self {
            grid_dimensions,
            world_size,
            cell_size,
            sphere_radius,
            active_cells,
            active_cell_map,
            cell_contents: vec![0; max_cells],
            cell_offsets: vec![0; active_count],
            cell_counts: vec![0; active_count],
            used_grid_cells: Vec::with_capacity(max_cells),
        }
    }

    /// Precompute which grid cells are active
    fn precompute_active_cells(grid_dimensions: UVec3) -> Vec<IVec3> {
        let mut active_cells = Vec::new();

        // Include all grid cells
        for x in 0..grid_dimensions.x as i32 {
            for y in 0..grid_dimensions.y as i32 {
                for z in 0..grid_dimensions.z as i32 {
                    active_cells.push(IVec3::new(x, y, z));
                }
            }
        }

        active_cells
    }

    /// Convert world position to grid coordinates
    fn world_to_grid(&self, position: Vec3) -> IVec3 {
        let offset_position = position + Vec3::splat(self.world_size / 2.0);
        let grid_pos = offset_position / self.cell_size;
        
        let max_coord = (self.grid_dimensions.x - 1) as i32;
        IVec3::new(
            (grid_pos.x as i32).clamp(0, max_coord),
            (grid_pos.y as i32).clamp(0, max_coord),
            (grid_pos.z as i32).clamp(0, max_coord),
        )
    }

    /// Find the index of a grid coordinate in the active_cells array
    fn active_cell_index(&self, grid_coord: IVec3) -> Option<usize> {
        self.active_cell_map.get(&grid_coord).copied()
    }

    /// Rebuild the spatial grid using prefix sum algorithm (parallel for large cell counts)
    pub fn rebuild(&mut self, positions: &[Vec3], cell_count: usize) {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        // Clear only previously used counts
        for &idx in &self.used_grid_cells {
            self.cell_counts[idx] = 0;
        }
        self.used_grid_cells.clear();

        // Use parallel counting for large cell counts (>500 cells)
        if cell_count > 500 {
            // Parallel counting with atomic operations
            let atomic_counts: Vec<AtomicUsize> = (0..self.active_cells.len())
                .map(|_| AtomicUsize::new(0))
                .collect();
            
            (0..cell_count).into_par_iter().for_each(|i| {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    atomic_counts[idx].fetch_add(1, Ordering::Relaxed);
                }
            });
            
            // Convert atomic counts and track used cells
            for (idx, atomic_count) in atomic_counts.iter().enumerate() {
                let count = atomic_count.load(Ordering::Relaxed);
                if count > 0 {
                    self.cell_counts[idx] = count;
                    self.used_grid_cells.push(idx);
                }
            }
        } else {
            // Sequential counting for small cell counts
            for i in 0..cell_count {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    if self.cell_counts[idx] == 0 {
                        self.used_grid_cells.push(idx);
                    }
                    self.cell_counts[idx] += 1;
                }
            }
        }

        // Compute offsets using prefix sum
        let mut offset = 0;
        for &idx in &self.used_grid_cells {
            self.cell_offsets[idx] = offset;
            offset += self.cell_counts[idx];
        }

        // Reset counts for insertion phase
        for &idx in &self.used_grid_cells {
            self.cell_counts[idx] = 0;
        }

        // Parallel insertion for large cell counts
        if cell_count > 500 {
            let atomic_offsets: Vec<AtomicUsize> = self.cell_offsets
                .iter()
                .map(|&offset| AtomicUsize::new(offset))
                .collect();
            
            (0..cell_count).into_par_iter().for_each(|i| {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    let insert_pos = atomic_offsets[idx].fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        // Safe because each thread writes to a unique position
                        let ptr = self.cell_contents.as_ptr() as *mut usize;
                        *ptr.add(insert_pos) = i;
                    }
                }
            });
            
            // Update counts from atomic offsets
            for (idx, atomic_offset) in atomic_offsets.iter().enumerate() {
                let final_offset = atomic_offset.load(Ordering::Relaxed);
                self.cell_counts[idx] = final_offset - self.cell_offsets[idx];
            }
        } else {
            // Sequential insertion for small cell counts
            for i in 0..cell_count {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    let insert_pos = self.cell_offsets[idx] + self.cell_counts[idx];
                    self.cell_contents[insert_pos] = i;
                    self.cell_counts[idx] += 1;
                }
            }
        }
    }

    /// Get a slice of cell indices in a specific grid cell
    pub fn get_cell_contents(&self, grid_idx: usize) -> &[usize] {
        let start = self.cell_offsets[grid_idx];
        let count = self.cell_counts[grid_idx];
        &self.cell_contents[start..start + count]
    }
}

// ============================================================================
// Collision Detection
// ============================================================================

/// Collision pair between two cells
#[derive(Clone, Copy, Debug)]
pub struct CollisionPair {
    pub index_a: usize,
    pub index_b: usize,
    pub overlap: f32,
    pub normal: Vec3,
}

/// Forward neighbors for half-space optimization (13 neighbors instead of 27)
const FORWARD_NEIGHBORS: [IVec3; 13] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 1, 0), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0),
    IVec3::new(-1, -1, 1), IVec3::new(0, -1, 1), IVec3::new(1, -1, 1),
    IVec3::new(-1, 0, 1), IVec3::new(0, 0, 1), IVec3::new(1, 0, 1),
    IVec3::new(-1, 1, 1), IVec3::new(0, 1, 1), IVec3::new(1, 1, 1),
];

/// Detect collisions using the deterministic spatial grid - Single-threaded version
pub fn detect_collisions_st(
    state: &CanonicalState,
) -> Vec<CollisionPair> {
    let mut collision_pairs = Vec::new();

    // Iterate through only the grid cells that contain simulation cells
    for &grid_idx in &state.spatial_grid.used_grid_cells {
        let grid_coord = state.spatial_grid.active_cells[grid_idx];
        let cells_in_grid = state.spatial_grid.get_cell_contents(grid_idx);

        // Check collisions within the same grid cell
        for i in 0..cells_in_grid.len() {
            let idx_a = cells_in_grid[i];
            for j in (i + 1)..cells_in_grid.len() {
                let idx_b = cells_in_grid[j];

                let delta = state.positions[idx_b] - state.positions[idx_a];
                let distance = delta.length();
                let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                if distance < combined_radius {
                    // Skip collision if cells are in the same organism
                    // (adhesion forces handle their interaction instead)
                    if are_cells_in_same_organism(state, idx_a, idx_b) {
                        continue;
                    }
                    
                    let overlap = combined_radius - distance;
                    let normal = if distance > 0.0001 {
                        delta / distance
                    } else {
                        Vec3::X
                    };

                    collision_pairs.push(CollisionPair {
                        index_a: idx_a,
                        index_b: idx_b,
                        overlap,
                        normal,
                    });
                }
            }
        }

        // Check forward neighbors to avoid duplicate checks
        for &offset in &FORWARD_NEIGHBORS {
            let neighbor_coord = grid_coord + offset;

            let Some(neighbor_idx) = state.spatial_grid.active_cell_index(neighbor_coord) else {
                continue;
            };

            let neighbor_cells = state.spatial_grid.get_cell_contents(neighbor_idx);

            for &idx_a in cells_in_grid {
                for &idx_b in neighbor_cells {
                    let delta = state.positions[idx_b] - state.positions[idx_a];
                    let distance = delta.length();
                    let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                    if distance < combined_radius {
                        // Skip collision if cells are in the same organism
                        // (adhesion forces handle their interaction instead)
                        if are_cells_in_same_organism(state, idx_a, idx_b) {
                            continue;
                        }
                        
                        let overlap = combined_radius - distance;
                        let normal = if distance > 0.0001 {
                            delta / distance
                        } else {
                            Vec3::X
                        };

                        collision_pairs.push(CollisionPair {
                            index_a: idx_a,
                            index_b: idx_b,
                            overlap,
                            normal,
                        });
                    }
                }
            }
        }
    }

    // Sort collision pairs by indices to maintain deterministic ordering
    collision_pairs.sort_unstable_by_key(|pair| (pair.index_a, pair.index_b));

    collision_pairs
}

/// Detect collisions using the deterministic spatial grid - Multithreaded version
/// Uses parallel iteration over grid cells for improved performance with large cell counts.
pub fn detect_collisions_parallel(
    state: &CanonicalState,
) -> Vec<CollisionPair> {
    use rayon::prelude::*;
    
    // Process each grid cell in parallel
    let mut collision_pairs: Vec<CollisionPair> = state.spatial_grid.used_grid_cells
        .par_iter()
        .flat_map(|&grid_idx| {
            let mut local_pairs = Vec::new();
            let grid_coord = state.spatial_grid.active_cells[grid_idx];
            let cells_in_grid = state.spatial_grid.get_cell_contents(grid_idx);

            // Check collisions within the same grid cell
            for i in 0..cells_in_grid.len() {
                let idx_a = cells_in_grid[i];
                for j in (i + 1)..cells_in_grid.len() {
                    let idx_b = cells_in_grid[j];

                    let delta = state.positions[idx_b] - state.positions[idx_a];
                    let distance = delta.length();
                    let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                    if distance < combined_radius {
                        // Skip collision if cells are in the same organism
                        if are_cells_in_same_organism(state, idx_a, idx_b) {
                            continue;
                        }
                        
                        let overlap = combined_radius - distance;
                        let normal = if distance > 0.0001 {
                            delta / distance
                        } else {
                            Vec3::X
                        };

                        local_pairs.push(CollisionPair {
                            index_a: idx_a,
                            index_b: idx_b,
                            overlap,
                            normal,
                        });
                    }
                }
            }

            // Check forward neighbors
            for &offset in &FORWARD_NEIGHBORS {
                let neighbor_coord = grid_coord + offset;

                let Some(neighbor_idx) = state.spatial_grid.active_cell_index(neighbor_coord) else {
                    continue;
                };

                let neighbor_cells = state.spatial_grid.get_cell_contents(neighbor_idx);

                for &idx_a in cells_in_grid {
                    for &idx_b in neighbor_cells {
                        let delta = state.positions[idx_b] - state.positions[idx_a];
                        let distance = delta.length();
                        let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                        if distance < combined_radius {
                            if are_cells_in_same_organism(state, idx_a, idx_b) {
                                continue;
                            }
                            
                            let overlap = combined_radius - distance;
                            let normal = if distance > 0.0001 {
                                delta / distance
                            } else {
                                Vec3::X
                            };

                            local_pairs.push(CollisionPair {
                                index_a: idx_a,
                                index_b: idx_b,
                                overlap,
                                normal,
                            });
                        }
                    }
                }
            }
            
            local_pairs
        })
        .collect();

    // Sort for deterministic ordering
    collision_pairs.sort_unstable_by_key(|pair| (pair.index_a, pair.index_b));

    collision_pairs
}

/// Check if two cells are in the same organism (connected via adhesions)
/// Cells in the same organism rely on adhesion forces for their interactions
#[inline]
fn are_cells_in_same_organism(state: &CanonicalState, cell_a: usize, cell_b: usize) -> bool {
    state.adhesion_manager.are_cells_connected(&state.adhesion_connections, cell_a, cell_b)
}

// ============================================================================
// Force Computation
// ============================================================================

/// Compute collision forces from detected collision pairs
pub fn compute_collision_forces(
    state: &mut CanonicalState,
    collision_pairs: &[CollisionPair],
    config: &PhysicsConfig,
) {
    // Clear all forces and torques
    for i in 0..state.cell_count {
        state.forces[i] = Vec3::ZERO;
        state.torques[i] = Vec3::ZERO;
    }

    // Process each collision pair
    for pair in collision_pairs {
        let idx_a = pair.index_a;
        let idx_b = pair.index_b;

        let stiffness_a = state.stiffnesses[idx_a];
        let stiffness_b = state.stiffnesses[idx_b];

        // Compute combined stiffness using harmonic mean
        let combined_stiffness = if stiffness_a > 0.0 && stiffness_b > 0.0 {
            (stiffness_a * stiffness_b) / (stiffness_a + stiffness_b)
        } else if stiffness_a > 0.0 {
            stiffness_a
        } else if stiffness_b > 0.0 {
            stiffness_b
        } else {
            config.default_stiffness
        };

        // Calculate spring force
        let spring_force_magnitude = combined_stiffness * pair.overlap;

        // Calculate relative velocity along collision normal
        let relative_velocity = state.velocities[idx_b] - state.velocities[idx_a];
        let relative_velocity_normal = relative_velocity.dot(pair.normal);

        // Calculate damping force
        let damping_force_magnitude = -config.damping * relative_velocity_normal;

        // Total force magnitude
        let total_force_magnitude = spring_force_magnitude + damping_force_magnitude;

        // Clamp force magnitude
        let max_force = 10000.0;
        let clamped_force_magnitude = total_force_magnitude.clamp(-max_force, max_force);
        let force = clamped_force_magnitude * pair.normal;

        // Apply equal and opposite forces
        state.forces[idx_b] += force;
        state.forces[idx_a] -= force;

        // Rolling friction torque
        if config.friction_coefficient > 0.0 && pair.overlap > 0.0 {
            let contact_offset_a = pair.normal * state.radii[idx_a];
            let contact_offset_b = -pair.normal * state.radii[idx_b];

            let vel_at_contact_a = state.velocities[idx_a] +
                state.angular_velocities[idx_a].cross(contact_offset_a);
            let vel_at_contact_b = state.velocities[idx_b] +
                state.angular_velocities[idx_b].cross(contact_offset_b);

            let relative_vel_at_contact = vel_at_contact_b - vel_at_contact_a;
            let tangential_velocity = relative_vel_at_contact - pair.normal * relative_vel_at_contact.dot(pair.normal);
            let tangential_speed = tangential_velocity.length();

            if tangential_speed > 0.0001 {
                let tangent_direction = tangential_velocity / tangential_speed;
                let max_friction_torque = config.friction_coefficient * clamped_force_magnitude.abs();

                let torque_axis_a = contact_offset_a.cross(tangent_direction);
                let torque_axis_b = contact_offset_b.cross(tangent_direction);

                let torque_magnitude = (tangential_speed * state.radii[idx_a]).min(max_friction_torque);

                let resistance_torque_a = -torque_axis_a.normalize_or_zero() * torque_magnitude;
                let resistance_torque_b = -torque_axis_b.normalize_or_zero() * torque_magnitude;

                state.torques[idx_a] += resistance_torque_a;
                state.torques[idx_b] += resistance_torque_b;
            }
        }
    }
}

// ============================================================================
// Boundary Forces
// ============================================================================

/// Apply boundary forces that push cells toward center near the spherical boundary
/// 
/// Creates a smooth inward force that increases as cells approach the boundary.
/// The force activates in a "soft zone" near the boundary and becomes stronger closer to the edge.
/// Also applies torque to rotate cells to face inward.
pub fn apply_boundary_forces(
    state: &mut CanonicalState,
    config: &PhysicsConfig,
) {
    // Boundary force parameters
    let boundary_radius = config.sphere_radius;
    let soft_zone_thickness = 5.0; // Start applying force 5 units before boundary
    let soft_zone_start = boundary_radius - soft_zone_thickness;
    let max_boundary_force = 500.0; // Maximum inward force at the boundary

    for i in 0..state.cell_count {
        let distance_from_origin = state.positions[i].length();

        // Skip if at origin
        if distance_from_origin < 0.0001 {
            continue;
        }

        let r_hat = state.positions[i] / distance_from_origin;

        // Apply smooth inward force in the soft zone
        if distance_from_origin > soft_zone_start {
            // Calculate how far into the soft zone (0.0 at start, 1.0 at boundary)
            let penetration = (distance_from_origin - soft_zone_start) / soft_zone_thickness;
            let penetration_clamped = penetration.clamp(0.0, 1.0);

            // Quadratic force curve: gentle at first, strong near boundary
            let force_magnitude = max_boundary_force * penetration_clamped * penetration_clamped;

            // Apply inward force (negative radial direction)
            let inward_force = -r_hat * force_magnitude;
            // CRITICAL: Use hardcoded 0.016 to match reference implementation exactly
            state.velocities[i] += inward_force * 0.016;

            // Apply torque to rotate cell to face inward
            // Get the cell's forward direction (local +Z axis)
            let forward = state.rotations[i] * Vec3::Z;

            // Calculate desired direction (toward center)
            let desired_direction = -r_hat;

            // Calculate rotation axis (cross product of current forward and desired direction)
            let rotation_axis = forward.cross(desired_direction);
            let rotation_axis_length = rotation_axis.length();

            // Only apply torque if there's a meaningful rotation needed
            if rotation_axis_length > 0.001 {
                let normalized_axis = rotation_axis / rotation_axis_length;

                // Calculate angle between current and desired direction
                let dot_product = forward.dot(desired_direction).clamp(-1.0, 1.0);
                let angle = dot_product.acos();

                // Torque magnitude increases with penetration and angle
                let torque_strength = 50.0 * penetration_clamped * angle;

                // Apply torque to rotate toward center
                state.torques[i] += normalized_axis * torque_strength;
            }
        }

        // Hard clamp: if somehow past boundary, push back and reverse velocity
        if distance_from_origin > boundary_radius {
            state.positions[i] = r_hat * boundary_radius;

            // Reverse any outward velocity component
            let radial_velocity = state.velocities[i].dot(r_hat);
            if radial_velocity > 0.0 {
                state.velocities[i] -= r_hat * radial_velocity * 2.0; // Remove and reverse
            }
        }
    }
}

// ============================================================================
// Integration Functions
// ============================================================================

/// Verlet integration position update
pub fn verlet_integrate_positions(
    positions: &mut [Vec3],
    velocities: &[Vec3],
    accelerations: &[Vec3],
    dt: f32,
) {
    let dt_sq = dt * dt;
    for i in 0..positions.len() {
        if velocities[i].is_finite() && accelerations[i].is_finite() {
            positions[i] += velocities[i] * dt + 0.5 * accelerations[i] * dt_sq;
        }
    }
}

/// Verlet integration velocity update
pub fn verlet_integrate_velocities(
    velocities: &mut [Vec3],
    accelerations: &mut [Vec3],
    prev_accelerations: &mut [Vec3],
    forces: &[Vec3],
    masses: &[f32],
    dt: f32,
    velocity_damping: f32,
) {
    let velocity_damping_factor = velocity_damping.powf(dt * 100.0);

    for i in 0..velocities.len() {
        if masses[i] <= 0.0 || !masses[i].is_finite() {
            continue;
        }
        
        let old_acceleration = accelerations[i];
        let new_acceleration = forces[i] / masses[i];

        if new_acceleration.is_finite() {
            let velocity_change = 0.5 * (old_acceleration + new_acceleration) * dt;
            velocities[i] = (velocities[i] + velocity_change) * velocity_damping_factor;
            accelerations[i] = new_acceleration;
            prev_accelerations[i] = old_acceleration;
        }
    }
}

/// Update rotations from angular velocities
pub fn integrate_rotations(
    rotations: &mut [Quat],
    angular_velocities: &[Vec3],
    dt: f32,
) {
    for i in 0..rotations.len() {
        let ang_vel = angular_velocities[i];
        if ang_vel.length_squared() > 0.0001 {
            let angle = ang_vel.length() * dt;
            let axis = ang_vel.normalize();
            let delta_rotation = Quat::from_axis_angle(axis, angle);
            rotations[i] = (delta_rotation * rotations[i]).normalize();
        }
    }
}

/// Update angular velocities from torques (SoA version) - Single-threaded
pub fn integrate_angular_velocities(
    angular_velocities: &mut [Vec3],
    torques: &[Vec3],
    radii: &[f32],
    masses: &[f32],
    dt: f32,
    angular_damping: f32,
) {
    let angular_damping_factor = angular_damping.powf(dt * 100.0);

    for i in 0..angular_velocities.len() {
        if masses[i] <= 0.0 || !masses[i].is_finite() || radii[i] <= 0.0 {
            continue;
        }

        // Moment of inertia for a sphere: I = (2/5) * m * r²
        let moment_of_inertia = 0.4 * masses[i] * radii[i] * radii[i];

        if moment_of_inertia > 0.0 {
            let angular_acceleration = torques[i] / moment_of_inertia;
            angular_velocities[i] = (angular_velocities[i] + angular_acceleration * dt) * angular_damping_factor;
        }
    }
}

// ============================================================================
// Nutrient Growth
// ============================================================================

/// Update nutrient growth for cells
pub fn update_nutrient_growth(
    state: &mut CanonicalState,
    genome: &Genome,
    dt: f32,
) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Cells gain mass over time based on their mode's nutrient_gain_rate
            let mass_gain = mode.nutrient_gain_rate * dt;
            state.masses[i] += mass_gain;
            
            // Update radius based on mass (radius = mass, clamped to max_cell_size)
            state.radii[i] = state.masses[i].min(mode.max_cell_size).clamp(0.5, 2.0);
        }
    }
}

// ============================================================================
// Main Physics Step
// ============================================================================

/// Single-threaded physics step function
pub fn physics_step_st(
    state: &mut CanonicalState,
    config: &PhysicsConfig,
) {
    let dt = config.fixed_timestep;

    // 1. Verlet integration (position update)
    verlet_integrate_positions(
        &mut state.positions[..state.cell_count],
        &state.velocities[..state.cell_count],
        &state.accelerations[..state.cell_count],
        dt,
    );

    // 2. Update rotations from angular velocities
    integrate_rotations(
        &mut state.rotations[..state.cell_count],
        &state.angular_velocities[..state.cell_count],
        dt,
    );

    // 3. Update spatial partitioning
    state.spatial_grid.rebuild(&state.positions, state.cell_count);

    // 4. Detect collisions
    let collisions = detect_collisions_st(state);

    // 5. Compute forces and torques
    compute_collision_forces(state, &collisions, config);

    // 6. Apply boundary conditions
    apply_boundary_forces(state, config);

    // 7. Verlet integration (velocity update)
    verlet_integrate_velocities(
        &mut state.velocities[..state.cell_count],
        &mut state.accelerations[..state.cell_count],
        &mut state.prev_accelerations[..state.cell_count],
        &state.forces[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.velocity_damping,
    );

    // 8. Update angular velocities from torques
    integrate_angular_velocities(
        &mut state.angular_velocities[..state.cell_count],
        &state.torques[..state.cell_count],
        &state.radii[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.angular_damping,
    );
}

/// Physics step with genome-based features (nutrient growth, division)
pub fn physics_step_with_genome(
    state: &mut CanonicalState,
    genome: &Genome,
    config: &PhysicsConfig,
    current_time: f32,
) -> Vec<DivisionEvent> {
    let dt = config.fixed_timestep;

    // 1. Verlet integration (position update)
    verlet_integrate_positions(
        &mut state.positions[..state.cell_count],
        &state.velocities[..state.cell_count],
        &state.accelerations[..state.cell_count],
        dt,
    );

    // 2. Update rotations from angular velocities
    integrate_rotations(
        &mut state.rotations[..state.cell_count],
        &state.angular_velocities[..state.cell_count],
        dt,
    );

    // 3. Update spatial partitioning
    state.spatial_grid.rebuild(&state.positions, state.cell_count);

    // 4. Detect collisions (use parallel version for better performance)
    let collisions = detect_collisions_parallel(state);

    // 5. Compute forces and torques
    compute_collision_forces(state, &collisions, config);

    // 5.5. Compute adhesion forces
    // Update adhesion settings cache if genome changed
    state.update_adhesion_settings_cache(genome);
    
    // Apply adhesion forces using parallel version for better performance
    if state.adhesion_connections.active_count > 0 && !state.cached_adhesion_settings.is_empty() {
        crate::cell::compute_adhesion_forces_parallel(
            &state.adhesion_connections,
            &state.positions[..state.cell_count],
            &state.velocities[..state.cell_count],
            &state.rotations[..state.cell_count],
            &state.angular_velocities[..state.cell_count],
            &state.masses[..state.cell_count],
            &state.cached_adhesion_settings,
            &mut state.forces[..state.cell_count],
            &mut state.torques[..state.cell_count],
        );
    }

    // 6. Apply boundary conditions
    apply_boundary_forces(state, config);

    // 7. Verlet integration (velocity update)
    verlet_integrate_velocities(
        &mut state.velocities[..state.cell_count],
        &mut state.accelerations[..state.cell_count],
        &mut state.prev_accelerations[..state.cell_count],
        &state.forces[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.velocity_damping,
    );

    // 8. Update angular velocities from torques
    integrate_angular_velocities(
        &mut state.angular_velocities[..state.cell_count],
        &state.torques[..state.cell_count],
        &state.radii[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.angular_damping,
    );

    // 9. Update nutrient growth
    update_nutrient_growth(state, genome, dt);

    // 10. Run division step
    let max_cells = state.capacity;
    let rng_seed = 12345;
    division::division_step(state, genome, current_time, max_cells, rng_seed)
}