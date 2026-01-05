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
                state.velocities[i] -= r_hat * radial_velocity * 2.0;
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

/// Update nutrient growth
pub fn update_nutrient_growth(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
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

    let max_cells = state.capacity;
    let rng_seed = 12345;
    division::division_step(state, genome, current_time, max_cells, rng_seed)
}