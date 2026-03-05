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
pub fn apply_buoyancy_forces(state: &mut CanonicalState, genome: &Genome) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            if mode.cell_type == 5 && mode.buoyancy_force > 0.0 {
                let upward_force = Vec3::Y * mode.buoyancy_force * 120.0;
                state.forces[i] += upward_force;
            }
        }
    }
}
pub fn apply_swim_forces(state: &mut CanonicalState, genome: &Genome) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Only apply swim force to Flagellocyte cells (cell_type == 1)
            if mode.cell_type == 1 {
                // Determine effective swim speed
                let effective_speed = if mode.flagellocyte_use_signal {
                    let channel = mode.flagellocyte_signal_channel.clamp(0, 15) as usize;
                    let signal_value = state.signal_channels[i * 16 + channel].unwrap_or(0.0);
                    if signal_value >= mode.flagellocyte_threshold_c {
                        mode.flagellocyte_speed_b
                    } else {
                        mode.flagellocyte_speed_a
                    }
                } else {
                    mode.swim_force
                };

                if effective_speed <= 0.0 { continue; }

                // Get forward direction from cell's rotation (local +Z axis)
                let forward = state.rotations[i] * Vec3::Z;
                
                // Apply thrust force in forward direction
                // Scale by 120.0 to make the force meaningful in the physics simulation
                let thrust_force = forward * effective_speed * 120.0;
                state.forces[i] += thrust_force;
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
/// - Consumption rate: swim_force * 5.0 * dt (5 nutrients per second at full swim force)
/// - Death threshold: nutrients < 1.0
/// - Mass and radius are derived from nutrients: mass = 1.0 + nutrients/100.0
/// - Only applies to cells with cell_type == 1 (Flagellocyte) and swim_force > 0.0
pub fn consume_swim_nutrients(
    state: &mut CanonicalState,
    genome: &Genome,
    dt: f32,
) -> Vec<usize> {
    const DEATH_NUTRIENT_THRESHOLD: f32 = 1.0;
    // Fixed consumption rate - NOT adjustable by user
    // This creates a direct tradeoff: faster swimming = higher nutrient cost
    // Rate: 1.0 nutrients/sec at swim_force=1.0, 3.0/sec at swim_force=3.0
    const CONSUMPTION_RATE: f32 = 1.0;
    
    let mut cells_to_remove = Vec::new();
    
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Only apply nutrient consumption to Flagellocyte cells (cell_type == 1)
            if mode.cell_type == 1 {
                // Determine effective swim speed
                let effective_speed = if mode.flagellocyte_use_signal {
                    let channel = mode.flagellocyte_signal_channel.clamp(0, 15) as usize;
                    let signal_value = state.signal_channels[i * 16 + channel].unwrap_or(0.0);
                    if signal_value >= mode.flagellocyte_threshold_c {
                        mode.flagellocyte_speed_b
                    } else {
                        mode.flagellocyte_speed_a
                    }
                } else {
                    mode.swim_force
                };
                if effective_speed <= 0.0 { continue; }

                // Consume nutrients proportional to effective speed
                let nutrient_loss = effective_speed * CONSUMPTION_RATE * dt;
                state.nutrients[i] = (state.nutrients[i] - nutrient_loss).max(0.0);
                
                // Check if cell has died (below death nutrient threshold)
                if state.nutrients[i] < DEATH_NUTRIENT_THRESHOLD {
                    cells_to_remove.push(i);
                    continue;
                }
                
                // Derive mass and radius from nutrients
                let new_mass = 1.0 + state.nutrients[i] / 100.0;
                let new_radius = new_mass.min(mode.max_cell_size).clamp(0.5, 2.0);
                if new_mass != state.masses[i] {
                    state.masses[i] = new_mass;
                    state.masses_changed = true;
                }
                if new_radius != state.radii[i] {
                    state.radii[i] = new_radius;
                    state.masses_changed = true;
                }
            }
        }
    }
    
    cells_to_remove
}

/// Note: Flagellocytes (cell_type == 1) don't generate their own nutrients -
/// they must receive nutrients through adhesion connections from other cells.
/// Mass and radius are derived from nutrients: mass = 1.0 + nutrients/100.0
pub fn update_nutrient_growth(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    const BASE_METABOLISM_RATE: f32 = 1.0; // nutrients/sec drain for non-auto-gain cells
    const AUTO_GAIN_RATE: f32 = 20.0; // nutrients/sec for Test, Phagocyte, Photocyte
    const OCULOCYTE_SENSE_CONSUMPTION_RATE: f32 = 0.08; // nutrients/sec per unit sense_range (default 25 range = 2.0/sec)
    
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            let can_auto_gain = mode.cell_type == 0  // Test
                             || mode.cell_type == 2  // Phagocyte
                             || mode.cell_type == 3; // Photocyte
            let is_oculocyte = mode.cell_type == 7;

            let current_nutrients = state.nutrients[i];
            let split_nutrient_threshold = state.split_nutrient_thresholds[i];

            if can_auto_gain {
                // Auto-gain nutrients up to split threshold * 2
                // Test cells: pure auto-gain, no drain (matches mass_accum.wgsl)
                // Phagocyte/Photocyte: gain - base drain (matches GPU: gain via specialized shader, drain via nutrient_transport.wgsl)
                let is_test_cell = mode.cell_type == 0;
                let max_nutrients = split_nutrient_threshold * 2.0;
                let net_gain = if is_test_cell {
                    AUTO_GAIN_RATE * dt
                } else {
                    // Phagocyte/Photocyte: gain 20/sec, drain 1/sec = net 19/sec (matches GPU scene)
                    (AUTO_GAIN_RATE - BASE_METABOLISM_RATE) * dt
                };
                if current_nutrients < max_nutrients {
                    let new_nutrients = (current_nutrients + net_gain).min(max_nutrients).max(0.0);
                    if new_nutrients != current_nutrients {
                        state.nutrients[i] = new_nutrients;
                        // Derive mass and radius from nutrients
                        let new_mass = 1.0 + new_nutrients / 100.0;
                        let new_radius = new_mass.min(mode.max_cell_size).clamp(0.5, 2.0);
                        state.masses[i] = new_mass;
                        if new_radius != state.radii[i] {
                            state.radii[i] = new_radius;
                            state.masses_changed = true;
                        }
                    }
                }
            } else {
                // Non-auto-gain cells lose nutrients to metabolism
                // Base drain for all non-auto-gain cells
                let base_loss = BASE_METABOLISM_RATE * dt;
                // Oculocytes have additional sense range consumption
                let sense_loss = if is_oculocyte {
                    mode.oculocyte_ray_length * OCULOCYTE_SENSE_CONSUMPTION_RATE * dt
                } else {
                    0.0
                };
                let total_loss = base_loss + sense_loss;
                let new_nutrients = (current_nutrients - total_loss).max(0.0);
                if new_nutrients != current_nutrients {
                    state.nutrients[i] = new_nutrients;
                    // Derive mass and radius from nutrients
                    let new_mass = 1.0 + new_nutrients / 100.0;
                    state.masses[i] = new_mass;
                    let new_radius = new_mass.min(mode.max_cell_size).clamp(0.5, 2.0);
                    if new_radius != state.radii[i] {
                        state.radii[i] = new_radius;
                        state.masses_changed = true;
                    }
                }
            }
        }
    }
}

/// Transport nutrients between adhesion-connected cells based on priority ratios.
/// Total outflow per cell is capped at TRANSPORT_RATE nutrients/sec regardless of connection count.
/// Transfer is lerped (smoothed) to prevent oscillation.
pub fn transport_nutrients_through_adhesions(state: &mut CanonicalState, genome: &Genome, dt: f32) {
        const PRIORITY_BOOST: f32 = 10.0;
    /// Max nutrients/sec a cell can send OR receive in total across all connections.
    const TRANSPORT_RATE: f32 = 30.0;
    /// Lerp factor: how quickly flow tracks the pressure-diff target (per second).
    /// Lower = smoother/slower response, higher = faster but more oscillation risk.
    const LERP_SPEED: f32 = 999.0;

    let n = state.cell_count;

    // Pass 1: compute desired (uncapped) flow for each connection.
    // Positive = nutrients flow from cell_a to cell_b.
    struct ConnFlow {
        conn_idx: usize,
        cell_a: usize,
        cell_b: usize,
        desired: f32, // nutrients/sec, uncapped
    }
    let mut conn_flows: Vec<ConnFlow> = Vec::new();

    let connections = &state.adhesion_connections;
    for i in 0..connections.is_active.len() {
        if connections.is_active[i] == 0 {
            continue;
        }
        let cell_a = connections.cell_a_index[i];
        let cell_b = connections.cell_b_index[i];
        if cell_a >= n || cell_b >= n {
            continue;
        }
        let mode_a = match genome.modes.get(state.mode_indices[cell_a]) {
            Some(m) => m,
            None => continue,
        };
        let mode_b = match genome.modes.get(state.mode_indices[cell_b]) {
            Some(m) => m,
            None => continue,
        };

        let nutrients_a = state.nutrients[cell_a];
        let nutrients_b = state.nutrients[cell_b];

        let priority_a = if mode_a.prioritize_when_low && nutrients_a < 10.0 {
            mode_a.nutrient_priority * PRIORITY_BOOST
        } else {
            mode_a.nutrient_priority
        };
        let priority_b = if mode_b.prioritize_when_low && nutrients_b < 10.0 {
            mode_b.nutrient_priority * PRIORITY_BOOST
        } else {
            mode_b.nutrient_priority
        };

        let pressure_diff = nutrients_a / priority_a - nutrients_b / priority_b;
        let desired = pressure_diff.clamp(-TRANSPORT_RATE, TRANSPORT_RATE);

        conn_flows.push(ConnFlow { conn_idx: i, cell_a, cell_b, desired });
    }

    // Pass 2: sum total desired outflow per cell to compute scaling factors.
    // Each cell's total outflow is capped at TRANSPORT_RATE nutrients/sec.
    let mut total_out = vec![0.0f32; n];
    for cf in &conn_flows {
        if cf.desired > 0.0 {
            total_out[cf.cell_a] += cf.desired;
        } else if cf.desired < 0.0 {
            total_out[cf.cell_b] += -cf.desired;
        }
    }

    // Pass 3: apply lerped, scaled transfers with transport-rate-adjusted pressure.
    let mut nutrient_deltas = vec![0.0f32; n];
    let lerp_t = (LERP_SPEED * dt).min(1.0);

    for cf in &conn_flows {
        let cell_a = cf.cell_a;
        let cell_b = cf.cell_b;

        // Scale outflow so the sending cell never exceeds TRANSPORT_RATE total.
        // This proportionally reduces each connection's flow so that the sender's
        // combined outflow across all connections stays within TRANSPORT_RATE.
        // No separate saturation factor is applied — that would double-throttle and
        // zero out flow entirely when total_out >= TRANSPORT_RATE, starving receivers.
        let scale = if cf.desired > 0.0 && total_out[cell_a] > TRANSPORT_RATE {
            TRANSPORT_RATE / total_out[cell_a]
        } else if cf.desired < 0.0 && total_out[cell_b] > TRANSPORT_RATE {
            TRANSPORT_RATE / total_out[cell_b]
        } else {
            1.0
        };

        // Target transfer this step (nutrients), lerped toward desired
        let target_per_sec = cf.desired * scale;
        let transfer = target_per_sec * lerp_t * dt;

        // Clamp by available nutrients and receiver capacity
        let mode_a = genome.modes.get(state.mode_indices[cell_a]).unwrap();
        let mode_b = genome.modes.get(state.mode_indices[cell_b]).unwrap();
        let min_a = if mode_a.prioritize_when_low { 10.0 } else { 0.0 };
        let min_b = if mode_b.prioritize_when_low { 10.0 } else { 0.0 };
        let max_a = state.split_nutrient_thresholds[cell_a] * 2.0;
        let max_b = state.split_nutrient_thresholds[cell_b] * 2.0;

        let actual = if transfer > 0.0 {
            let can_give = (state.nutrients[cell_a] - min_a).max(0.0);
            let can_recv = (max_b - state.nutrients[cell_b]).max(0.0);
            transfer.min(can_give).min(can_recv)
        } else {
            let can_give = (state.nutrients[cell_b] - min_b).max(0.0);
            let can_recv = (max_a - state.nutrients[cell_a]).max(0.0);
            transfer.max(-(can_give.min(can_recv)))
        };

        nutrient_deltas[cell_a] -= actual;
        nutrient_deltas[cell_b] += actual;

        // Store actual flow rate (nutrients/sec, positive = A→B) for display
        if dt > 0.0 {
            state.adhesion_connections.connection_flow_rates[cf.conn_idx] = actual / dt;
        }
    }

    for i in 0..state.cell_count {
        if nutrient_deltas[i] != 0.0 {
            let new_nutrients = (state.nutrients[i] + nutrient_deltas[i]).max(0.0);
            if new_nutrients != state.nutrients[i] {
                state.nutrients[i] = new_nutrients;
                // Derive mass from nutrients: mass = 1.0 + nutrients/100.0
                let new_mass = 1.0 + new_nutrients / 100.0;
                state.masses[i] = new_mass;
                let max_size = genome.modes.get(state.mode_indices[i])
                    .map(|m| m.max_cell_size)
                    .unwrap_or(2.0);
                let new_radius = new_mass.min(max_size).clamp(0.5, 2.0);
                if new_radius != state.radii[i] {
                    state.radii[i] = new_radius;
                    state.masses_changed = true;
                }
            }
        }
    }
}

/// Form adhesion bonds for Glueocyte cells on contact with other cells.
///
/// Any Glueocyte (cell_type == 6) that is currently overlapping another cell
/// will attempt to form an adhesion bond using its configured adhesion_settings,
/// subject to the cell's max_adhesions limit and the global adhesion capacity.
pub fn form_glueocyte_contact_bonds(state: &mut CanonicalState, genome: &Genome, current_time: f32) {
    let collision_pairs = detect_collisions(state);

    for pair in collision_pairs {
        let idx_a = pair.index_a;
        let idx_b = pair.index_b;

        if state.adhesion_manager.are_cells_connected(&state.adhesion_connections, idx_a, idx_b) {
            continue;
        }

        // Skip if either cell still has sister immunity toward the other
        let id_b = state.cell_ids[idx_b];
        let id_a = state.cell_ids[idx_a];
        if state.sister_cell_id[idx_a] == id_b && current_time < state.sister_expiry[idx_a] {
            continue;
        }
        if state.sister_cell_id[idx_b] == id_a && current_time < state.sister_expiry[idx_b] {
            continue;
        }

        let mode_a = state.mode_indices[idx_a];
        let mode_b = state.mode_indices[idx_b];

        let is_glue_a = genome.modes.get(mode_a).map(|m| m.cell_type == 6).unwrap_or(false);
        let is_glue_b = genome.modes.get(mode_b).map(|m| m.cell_type == 6).unwrap_or(false);

        if !is_glue_a && !is_glue_b {
            continue;
        }

        // Skip if the Glueocyte(s) in this pair have cell adhesion disabled
        let cell_adhesion_a = genome.modes.get(mode_a).map(|m| !is_glue_a || m.glueocyte_cell_adhesion).unwrap_or(true);
        let cell_adhesion_b = genome.modes.get(mode_b).map(|m| !is_glue_b || m.glueocyte_cell_adhesion).unwrap_or(true);
        if !cell_adhesion_a || !cell_adhesion_b {
            continue;
        }

        let adhesions_a = state.adhesion_manager.count_active_adhesions(idx_a);
        let adhesions_b = state.adhesion_manager.count_active_adhesions(idx_b);

        let max_a = genome.modes.get(mode_a).map(|m| m.max_adhesions as usize).unwrap_or(10);
        let max_b = genome.modes.get(mode_b).map(|m| m.max_adhesions as usize).unwrap_or(10);

        if adhesions_a >= max_a || adhesions_b >= max_b {
            continue;
        }

        let dir_a_to_b = (state.positions[idx_b] - state.positions[idx_a]).normalize_or(glam::Vec3::X);
        let dir_b_to_a = -dir_a_to_b;

        let anchor_a = state.genome_orientations[idx_a].inverse() * dir_a_to_b;
        let anchor_b = state.genome_orientations[idx_b].inverse() * dir_b_to_a;

        let split_dir_a = genome.modes.get(mode_a).map(|m| {
            let pitch = m.parent_split_direction.x.to_radians();
            let yaw = m.parent_split_direction.y.to_radians();
            glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0) * glam::Vec3::Z
        }).unwrap_or(glam::Vec3::Z);

        let split_dir_b = genome.modes.get(mode_b).map(|m| {
            let pitch = m.parent_split_direction.x.to_radians();
            let yaw = m.parent_split_direction.y.to_radians();
            glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0) * glam::Vec3::Z
        }).unwrap_or(glam::Vec3::Z);

        let split_ratio_a = genome.modes.get(mode_a).map(|m| m.split_ratio).unwrap_or(0.5);
        let split_ratio_b = genome.modes.get(mode_b).map(|m| m.split_ratio).unwrap_or(0.5);

        let _ = state.adhesion_manager.add_adhesion_with_directions(
            &mut state.adhesion_connections,
            idx_a,
            idx_b,
            mode_a,
            anchor_a,
            anchor_b,
            split_dir_a,
            split_dir_b,
            state.genome_orientations[idx_a],
            state.genome_orientations[idx_b],
            split_ratio_a,
            split_ratio_b,
            current_time,
        );
    }
}

/// Physics step with genome support
pub fn physics_step_with_genome(
    state: &mut CanonicalState,
    genome: &Genome,
    config: &PhysicsConfig,
    current_time: f32,
    test_signals: Option<&[crate::simulation::signal_system::SignalEmission]>,
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
        let bonds_to_break = crate::cell::compute_adhesion_forces_parallel(
            &state.adhesion_connections,
            &state.positions[..state.cell_count],
            &state.velocities[..state.cell_count],
            &state.rotations[..state.cell_count],
            &state.angular_velocities[..state.cell_count],
            &state.masses[..state.cell_count],
            &state.genome_orientations[..state.cell_count],
            &state.cached_adhesion_settings,
            &mut state.forces[..state.cell_count],
            &mut state.torques[..state.cell_count],
            current_time,
        );
        for conn_idx in bonds_to_break {
            state.adhesion_manager.remove_adhesion(&mut state.adhesion_connections, conn_idx);
        }
    }

    // Skip swim forces in preview mode - flagellocyte thrust is GPU-only
    // apply_swim_forces(state, genome);

    // Skip buoyancy forces in preview mode - buoyocyte lift is GPU-only
    // apply_buoyancy_forces(state, genome);

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

    // Adhesion constraint sub-stepping (matches GPU Stage 7.5)
    // Each iteration re-evaluates adhesion forces against latest positions and
    // applies corrections directly. Dramatically increases effective joint stiffness.
    if config.constraint_iterations > 0
        && state.adhesion_connections.active_count > 0
        && !state.cached_adhesion_settings.is_empty()
    {
        let cell_count = state.cell_count;
        for _ in 0..config.constraint_iterations {
            crate::cell::compute_adhesion_substep(
                &state.adhesion_connections,
                &mut state.positions[..cell_count],
                &mut state.velocities[..cell_count],
                &mut state.rotations[..cell_count],
                &mut state.angular_velocities[..cell_count],
                &state.masses[..cell_count],
                &state.genome_orientations[..cell_count],
                &state.cached_adhesion_settings,
                cell_count,
                dt,
            );
        }
    }

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

    // Remove any cells that starved (nutrients < 1.0), matching GPU death threshold.
    // This catches non-auto-gain cells (Buoyocyte, Glueocyte, Lipocyte, Flagellocyte)
    // that lost nutrients via passive drain and weren't rescued by nutrient transport.
    const DEATH_NUTRIENT_THRESHOLD: f32 = 1.0;
    let starved: Vec<usize> = (0..state.cell_count)
        .filter(|&i| state.nutrients[i] < DEATH_NUTRIENT_THRESHOLD)
        .collect();
    if !starved.is_empty() {
        state.remove_cells(&starved);
    }

    // Form contact adhesion bonds for Glueocyte cells
    form_glueocyte_contact_bonds(state, genome, current_time);

    // Run signal system (oculocyte sensing + BFS propagation)
    let boundary_radius = config.sphere_radius;
    crate::simulation::signal_system::run_signal_system(state, genome, boundary_radius);

    // Apply persistent test signals (if any) after normal signal system
    if let Some(test_signals) = test_signals {
        if !test_signals.is_empty() {
            // Clear signals before applying test signals to avoid accumulation
            crate::simulation::signal_system::clear_all_signals(state);
            crate::simulation::signal_system::propagate_test_signals(state, genome, test_signals.to_vec());
        }
    }

    let max_cells = state.capacity;
    let rng_seed = 12345;
    division::division_step(state, genome, current_time, max_cells, rng_seed)
}