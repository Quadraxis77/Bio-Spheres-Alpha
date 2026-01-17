//! Preview Physics - CPU-based physics for preview scene
//! 
//! This module provides CPU physics for the preview scene which needs
//! time scrubbing and checkpoint support. The GPU scene uses pure GPU physics.

use glam::{Vec3, Quat};
use crate::simulation::canonical_state::{CanonicalState, DivisionEvent};
use crate::simulation::physics_config::PhysicsConfig;
use crate::genome::Genome;
use crate::cell::division;

/// Collision pair between two cells
#[derive(Clone, Copy, Debug)]
pub struct CollisionPair {
    pub index_a: usize,
    pub index_b: usize,
    pub overlap: f32,
    pub normal: Vec3,
}

/// Forward neighbors for half-space optimization
const FORWARD_NEIGHBORS: [glam::IVec3; 13] = [
    glam::IVec3::new(1, 0, 0),
    glam::IVec3::new(-1, 1, 0), glam::IVec3::new(0, 1, 0), glam::IVec3::new(1, 1, 0),
    glam::IVec3::new(-1, -1, 1), glam::IVec3::new(0, -1, 1), glam::IVec3::new(1, -1, 1),
    glam::IVec3::new(-1, 0, 1), glam::IVec3::new(0, 0, 1), glam::IVec3::new(1, 0, 1),
    glam::IVec3::new(-1, 1, 1), glam::IVec3::new(0, 1, 1), glam::IVec3::new(1, 1, 1),
];

/// Check if two cells are connected via adhesions
#[inline]
fn are_cells_in_same_organism(state: &CanonicalState, cell_a: usize, cell_b: usize) -> bool {
    state.adhesion_manager.are_cells_connected(&state.adhesion_connections, cell_a, cell_b)
}

/// Detect collisions using the spatial grid
pub fn detect_collisions(state: &CanonicalState) -> Vec<CollisionPair> {
    let mut collision_pairs = Vec::new();

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
                    if are_cells_in_same_organism(state, idx_a, idx_b) {
                        continue;
                    }
                    
                    let overlap = combined_radius - distance;
                    let normal = if distance > 0.0001 { delta / distance } else { Vec3::X };

                    collision_pairs.push(CollisionPair {
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
            let Some(neighbor_idx) = state.spatial_grid.active_cell_index(neighbor_coord) else { continue };
            let neighbor_cells = state.spatial_grid.get_cell_contents(neighbor_idx);

            for &idx_a in cells_in_grid {
                for &idx_b in neighbor_cells {
                    let delta = state.positions[idx_b] - state.positions[idx_a];
                    let distance = delta.length();
                    let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                    if distance < combined_radius {
                        if are_cells_in_same_organism(state, idx_a, idx_b) { continue; }
                        
                        let overlap = combined_radius - distance;
                        let normal = if distance > 0.0001 { delta / distance } else { Vec3::X };

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

    collision_pairs.sort_unstable_by_key(|pair| (pair.index_a, pair.index_b));
    collision_pairs
}

/// Compute collision forces
pub fn compute_collision_forces(state: &mut CanonicalState, collision_pairs: &[CollisionPair], config: &PhysicsConfig) {
    for i in 0..state.cell_count {
        state.forces[i] = Vec3::ZERO;
        state.torques[i] = Vec3::ZERO;
    }

    for pair in collision_pairs {
        let idx_a = pair.index_a;
        let idx_b = pair.index_b;

        let stiffness_a = state.stiffnesses[idx_a];
        let stiffness_b = state.stiffnesses[idx_b];

        let combined_stiffness = if stiffness_a > 0.0 && stiffness_b > 0.0 {
            (stiffness_a * stiffness_b) / (stiffness_a + stiffness_b)
        } else if stiffness_a > 0.0 {
            stiffness_a
        } else {
            stiffness_b
        };

        let spring_force_magnitude = combined_stiffness * pair.overlap;
        let relative_velocity = state.velocities[idx_b] - state.velocities[idx_a];
        let relative_velocity_normal = relative_velocity.dot(pair.normal);
        let damping_force_magnitude = -config.damping * relative_velocity_normal;
        let total_force_magnitude = (spring_force_magnitude + damping_force_magnitude).clamp(-10000.0, 10000.0);
        let force = total_force_magnitude * pair.normal;

        state.forces[idx_b] += force;
        state.forces[idx_a] -= force;
    }
}

/// Apply boundary forces
pub fn apply_boundary_forces(state: &mut CanonicalState, config: &PhysicsConfig) {
    let boundary_radius = config.sphere_radius;
    let soft_zone_thickness = 5.0;
    let soft_zone_start = boundary_radius - soft_zone_thickness;
    let max_boundary_force = 500.0;

    for i in 0..state.cell_count {
        let distance_from_origin = state.positions[i].length();
        if distance_from_origin < 0.0001 { continue; }

        let r_hat = state.positions[i] / distance_from_origin;

        if distance_from_origin > soft_zone_start {
            let penetration = ((distance_from_origin - soft_zone_start) / soft_zone_thickness).clamp(0.0, 1.0);
            let force_magnitude = max_boundary_force * penetration * penetration;
            let inward_force = -r_hat * force_magnitude;
            state.velocities[i] += inward_force * 0.016;
        }

        if distance_from_origin > boundary_radius {
            state.positions[i] = r_hat * boundary_radius;
            let radial_velocity = state.velocities[i].dot(r_hat);
            if radial_velocity > 0.0 {
                // Extremely aggressive energy loss (coefficient of restitution = 0.02)
                // Cells lose 98% of their energy on boundary impact
                state.velocities[i] -= r_hat * radial_velocity * 1.98;
            }
        }
    }
}

/// Verlet position integration
pub fn verlet_integrate_positions(positions: &mut [Vec3], velocities: &[Vec3], accelerations: &[Vec3], dt: f32) {
    let dt_sq = dt * dt;
    for i in 0..positions.len() {
        if velocities[i].is_finite() && accelerations[i].is_finite() {
            positions[i] += velocities[i] * dt + 0.5 * accelerations[i] * dt_sq;
        }
    }
}

/// Verlet velocity integration
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
        if masses[i] <= 0.0 || !masses[i].is_finite() { continue; }
        
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

/// Integrate rotations
pub fn integrate_rotations(rotations: &mut [Quat], angular_velocities: &[Vec3], dt: f32) {
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

/// Integrate angular velocities
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
        if masses[i] <= 0.0 || !masses[i].is_finite() || radii[i] <= 0.0 { continue; }

        let moment_of_inertia = 0.4 * masses[i] * radii[i] * radii[i];
        if moment_of_inertia > 0.0 {
            let angular_acceleration = torques[i] / moment_of_inertia;
            angular_velocities[i] = (angular_velocities[i] + angular_acceleration * dt) * angular_damping_factor;
        }
    }
}

/// Apply swim forces for Flagellocyte cells (cell_type == 1)
/// 
/// Flagellocytes apply a forward thrust force in their orientation direction.
/// The thrust magnitude is determined by the swim_force parameter in ModeSettings.
/// 
/// # Arguments
/// * `state` - The canonical simulation state containing cell data
/// * `genome` - The genome containing mode settings with swim_force values
/// 
/// # Physics
/// - Forward direction is derived from the cell's rotation quaternion (local +Z axis)
/// - Thrust force = forward * swim_force * 120.0 (scaled for physics simulation)
/// - Only applies to cells with cell_type == 1 (Flagellocyte) and swim_force > 0.0
pub fn apply_swim_forces(state: &mut CanonicalState, genome: &Genome) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Only apply swim force to Flagellocyte cells (cell_type == 1)
            if mode.cell_type == 1 && mode.swim_force > 0.0 {
                // Get forward direction from cell's rotation (local +Z axis)
                let forward = state.rotations[i] * Vec3::Z;
                
                // Apply thrust force in forward direction
                // Scale by 120.0 to make the force meaningful in the physics simulation
                let thrust_force = forward * mode.swim_force * 120.0;
                state.forces[i] += thrust_force;
            }
        }
    }
}

/// Update nutrient growth
/// Note: Flagellocytes (cell_type == 1) don't generate their own nutrients -
/// they must receive nutrients through adhesion connections from other cells.
pub fn update_nutrient_growth(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Flagellocytes don't generate their own nutrients
            if mode.cell_type == 1 {
                continue;
            }
            
            let current_mass = state.masses[i];
            let max_mass = mode.max_cell_size;
            
            if current_mass < max_mass {
                let mass_gain = mode.nutrient_gain_rate * dt;
                if mass_gain > 0.0 {
                    let new_mass = (current_mass + mass_gain).min(max_mass);
                    if new_mass != current_mass {
                        state.masses[i] = new_mass;
                        let new_radius = new_mass.clamp(0.5, 2.0);
                        if new_radius != state.radii[i] {
                            state.radii[i] = new_radius;
                            state.masses_changed = true;
                        }
                    }
                }
            }
        }
    }
}

/// Transport nutrients between adhesion-connected cells based on priority ratios.
/// 
/// Nutrients flow to establish equilibrium: mass_a / mass_b = priority_a / priority_b
/// Flow is driven by "pressure" differences: pressure = mass / priority
/// Cells with low mass get temporary priority boost when below danger threshold.
/// 
/// This allows Flagellocytes (which consume nutrients for swimming) to receive
/// nutrients from connected cells with higher nutrient_gain_rate.
pub fn transport_nutrients_through_adhesions(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    const DANGER_THRESHOLD: f32 = 0.6;
    const PRIORITY_BOOST: f32 = 10.0;
    const TRANSPORT_RATE: f32 = 0.5;
    
    // Collect mass deltas to apply atomically after processing all adhesions
    let mut mass_deltas = vec![0.0f32; state.cell_count];
    
    // Process each active adhesion connection (SoA layout)
    let connections = &state.adhesion_connections;
    for i in 0..connections.is_active.len() {
        if connections.is_active[i] == 0 {
            continue;
        }
        
        let cell_a = connections.cell_a_index[i];
        let cell_b = connections.cell_b_index[i];
        
        if cell_a >= state.cell_count || cell_b >= state.cell_count {
            continue;
        }
        
        // Get mode settings for both cells
        let mode_a_idx = state.mode_indices[cell_a];
        let mode_b_idx = state.mode_indices[cell_b];
        
        let mode_a = match genome.modes.get(mode_a_idx) {
            Some(m) => m,
            None => continue,
        };
        let mode_b = match genome.modes.get(mode_b_idx) {
            Some(m) => m,
            None => continue,
        };
        
        let mass_a = state.masses[cell_a];
        let mass_b = state.masses[cell_b];
        
        // Get base priorities
        let base_priority_a = mode_a.nutrient_priority;
        let base_priority_b = mode_b.nutrient_priority;
        
        // Apply temporary priority boost when cells are dangerously low on nutrients
        let priority_a = if mode_a.prioritize_when_low && mass_a < DANGER_THRESHOLD {
            base_priority_a * PRIORITY_BOOST
        } else {
            base_priority_a
        };
        let priority_b = if mode_b.prioritize_when_low && mass_b < DANGER_THRESHOLD {
            base_priority_b * PRIORITY_BOOST
        } else {
            base_priority_b
        };
        
        // Calculate equilibrium-based nutrient flow
        // At equilibrium: mass_a / mass_b = priority_a / priority_b
        // Flow is driven by "pressure" difference: pressure = mass / priority
        let pressure_a = mass_a / priority_a;
        let pressure_b = mass_b / priority_b;
        let pressure_diff = pressure_a - pressure_b;
        
        // Calculate mass transfer (positive = A -> B, negative = B -> A)
        let mass_transfer = pressure_diff * TRANSPORT_RATE * dt;
        
        // Apply transfer with minimum thresholds
        let min_mass_a = if mode_a.prioritize_when_low { 0.1 } else { 0.0 };
        let min_mass_b = if mode_b.prioritize_when_low { 0.1 } else { 0.0 };
        
        let actual_transfer = if mass_transfer > 0.0 {
            // A -> B: limit by A's available mass
            mass_transfer.min(mass_a - min_mass_a)
        } else {
            // B -> A: limit by B's available mass
            mass_transfer.max(-(mass_b - min_mass_b))
        };
        
        // Accumulate deltas
        mass_deltas[cell_a] -= actual_transfer;
        mass_deltas[cell_b] += actual_transfer;
    }
    
    // Apply accumulated mass deltas
    for i in 0..state.cell_count {
        if mass_deltas[i] != 0.0 {
            let new_mass = (state.masses[i] + mass_deltas[i]).max(0.0);
            if new_mass != state.masses[i] {
                state.masses[i] = new_mass;
                let new_radius = new_mass.clamp(0.5, 2.0);
                if new_radius != state.radii[i] {
                    state.radii[i] = new_radius;
                    state.masses_changed = true;
                }
            }
        }
    }
}

/// Consume nutrients for Flagellocyte cells based on swim force.
/// 
/// Flagellocytes (cell_type == 1) consume mass proportional to their swim force.
/// The consumption rate is fixed (not user-adjustable) - faster swimming costs more nutrients.
/// 
/// # Arguments
/// * `state` - The canonical simulation state containing cell data
/// * `genome` - The genome containing mode settings with swim_force values
/// * `dt` - Delta time for this physics step
/// 
/// # Returns
/// A vector of cell indices that died (mass < MIN_CELL_MASS threshold).
/// These cells should be removed from the simulation.
/// 
/// # Physics
/// - Consumption rate: swim_force * 0.2 * dt (0.2 mass per second at full swim force)
/// - Death threshold: mass < 0.5
/// - Radius is updated based on remaining mass (clamped to 0.5-2.0 range)
/// - Only applies to cells with cell_type == 1 (Flagellocyte) and swim_force > 0.0
pub fn consume_swim_nutrients(
    state: &mut CanonicalState,
    genome: &Genome,
    dt: f32,
) -> Vec<usize> {
    const MIN_CELL_MASS: f32 = 0.5;
    // Fixed consumption rate - NOT adjustable by user
    // This creates a direct tradeoff: faster swimming = higher nutrient cost
    const CONSUMPTION_RATE: f32 = 0.2; // mass per second at full swim force
    
    let mut cells_to_remove = Vec::new();
    
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Only apply nutrient consumption to Flagellocyte cells (cell_type == 1)
            if mode.cell_type == 1 && mode.swim_force > 0.0 {
                // Consume mass proportional to swim force (automatic, not adjustable)
                let mass_loss = mode.swim_force * CONSUMPTION_RATE * dt;
                state.masses[i] -= mass_loss;
                
                // Check if cell has died (below minimum mass threshold)
                if state.masses[i] < MIN_CELL_MASS {
                    cells_to_remove.push(i);
                    continue;
                }
                
                // Update radius based on new mass
                let max_size = mode.max_cell_size;
                let target_radius = state.masses[i].min(max_size);
                let new_radius = target_radius.clamp(0.5, 2.0);
                if new_radius != state.radii[i] {
                    state.radii[i] = new_radius;
                    state.masses_changed = true;
                }
            }
        }
    }
    
    cells_to_remove
}

/// Physics step with genome support
pub fn physics_step_with_genome(
    state: &mut CanonicalState,
    genome: &Genome,
    config: &PhysicsConfig,
    current_time: f32,
) -> Vec<DivisionEvent> {
    let dt = config.fixed_timestep;

    verlet_integrate_positions(
        &mut state.positions[..state.cell_count],
        &state.velocities[..state.cell_count],
        &state.accelerations[..state.cell_count],
        dt,
    );

    integrate_rotations(
        &mut state.rotations[..state.cell_count],
        &state.angular_velocities[..state.cell_count],
        dt,
    );

    state.spatial_grid.rebuild(&state.positions, state.cell_count);

    let collisions = detect_collisions(state);
    compute_collision_forces(state, &collisions, config);

    state.update_adhesion_settings_cache(genome);
    state.update_membrane_stiffness_from_genome(genome);
    
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

    // Skip swim forces in preview mode - flagellocyte thrust is GPU-only
    // apply_swim_forces(state, genome);

    apply_boundary_forces(state, config);

    verlet_integrate_velocities(
        &mut state.velocities[..state.cell_count],
        &mut state.accelerations[..state.cell_count],
        &mut state.prev_accelerations[..state.cell_count],
        &state.forces[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.velocity_damping,
    );

    integrate_angular_velocities(
        &mut state.angular_velocities[..state.cell_count],
        &state.torques[..state.cell_count],
        &state.radii[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.angular_damping,
    );

    update_nutrient_growth(state, genome, dt);
    
    // Transport nutrients between adhesion-connected cells
    // This allows Flagellocytes to receive nutrients from connected cells
    transport_nutrients_through_adhesions(state, genome, dt);
    
    // Consume nutrients for Flagellocyte cells swimming
    // This must happen after nutrient growth so cells can potentially recover
    let dead_cells = consume_swim_nutrients(state, genome, dt);
    
    // Remove dead cells (those that ran out of nutrients while swimming)
    if !dead_cells.is_empty() {
        state.remove_cells(&dead_cells);
    }

    let max_cells = state.capacity;
    let rng_seed = 12345;
    division::division_step(state, genome, current_time, max_cells, rng_seed)
}