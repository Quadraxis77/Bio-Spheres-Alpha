//! Preview Physics - CPU-based physics for preview scene
//!
//! This module provides CPU physics for the preview scene which needs
//! time scrubbing and checkpoint support. The GPU scene uses pure GPU physics.

use crate::cell::division;
use crate::genome::Genome;
use crate::simulation::canonical_state::{CanonicalState, DivisionEvent};
use crate::simulation::physics_config::PhysicsConfig;
use glam::{Quat, Vec3};

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
    glam::IVec3::new(-1, 1, 0),
    glam::IVec3::new(0, 1, 0),
    glam::IVec3::new(1, 1, 0),
    glam::IVec3::new(-1, -1, 1),
    glam::IVec3::new(0, -1, 1),
    glam::IVec3::new(1, -1, 1),
    glam::IVec3::new(-1, 0, 1),
    glam::IVec3::new(0, 0, 1),
    glam::IVec3::new(1, 0, 1),
    glam::IVec3::new(-1, 1, 1),
    glam::IVec3::new(0, 1, 1),
    glam::IVec3::new(1, 1, 1),
];

/// Check if two cells share a normal developmental adhesion.
#[inline]
fn are_cells_in_same_organism(state: &CanonicalState, cell_a: usize, cell_b: usize) -> bool {
    if cell_a >= state.adhesion_manager.cell_adhesion_indices.len() {
        return false;
    }

    let connections = &state.adhesion_connections;
    for &conn_idx in &state.adhesion_manager.cell_adhesion_indices[cell_a] {
        if conn_idx < 0 {
            continue;
        }
        let conn_idx = conn_idx as usize;
        if conn_idx >= connections.active_count || connections.is_active[conn_idx] == 0 {
            continue;
        }
        if (connections.bond_flags[conn_idx] & crate::cell::adhesion::BOND_FLAG_BARRIER_BALL) != 0 {
            continue;
        }
        let other = if connections.cell_a_index[conn_idx] == cell_a {
            connections.cell_b_index[conn_idx]
        } else {
            connections.cell_a_index[conn_idx]
        };

        if other == cell_b {
            return true;
        }
    }
    false
}

fn find_scaffold_connection_between(
    state: &CanonicalState,
    cell_a: usize,
    cell_b: usize,
) -> Option<usize> {
    if cell_a >= state.adhesion_manager.cell_adhesion_indices.len() {
        return None;
    }

    let connections = &state.adhesion_connections;
    for &conn_idx in &state.adhesion_manager.cell_adhesion_indices[cell_a] {
        if conn_idx < 0 {
            continue;
        }
        let conn_idx = conn_idx as usize;
        if conn_idx >= connections.active_count || connections.is_active[conn_idx] == 0 {
            continue;
        }
        if (connections.bond_flags[conn_idx] & crate::cell::adhesion::BOND_FLAG_BARRIER_BALL) == 0 {
            continue;
        }

        let ca = connections.cell_a_index[conn_idx];
        let cb = connections.cell_b_index[conn_idx];
        if (ca == cell_a && cb == cell_b) || (ca == cell_b && cb == cell_a) {
            return Some(conn_idx);
        }
    }

    None
}

fn create_or_update_scaffold_bond(
    state: &mut CanonicalState,
    ca: usize,
    cb: usize,
    rule_id: u32,
    rest_length: f32,
    current_time: f32,
) {
    use crate::cell::adhesion::BOND_FLAG_BARRIER_BALL;

    if let Some(conn_idx) = find_scaffold_connection_between(state, ca, cb) {
        state.adhesion_connections.scaffold_rule_id[conn_idx] = rule_id;
        state.adhesion_connections.rest_length_overrides[conn_idx] = rest_length.max(0.001);
        return;
    }

    if state
        .adhesion_manager
        .find_connection_between(&state.adhesion_connections, ca, cb)
        .is_some()
    {
        return;
    }

    let mode_index = state.mode_indices[ca];
    let conn_idx = state.adhesion_manager.add_ball_joint_with_rest_length(
        &mut state.adhesion_connections,
        ca,
        cb,
        mode_index,
        current_time,
        BOND_FLAG_BARRIER_BALL,
        rest_length,
    );
    if let Some(idx) = conn_idx {
        state.adhesion_connections.scaffold_rule_id[idx] = rule_id;
    }
}

/// Sync developmental organism scopes to current developmental adhesion components.
///
/// Structural barrier-ball bonds (glueocyte/scaffold mechanics) do not define
/// ancestry and must not merge organism IDs.
pub fn sync_development_organism_ids_from_adhesions(state: &mut CanonicalState) {
    let n = state.cell_count;
    if n == 0 {
        return;
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for conn_idx in 0..state.adhesion_connections.active_count {
        if state.adhesion_connections.is_active[conn_idx] == 0 {
            continue;
        }
        if (state.adhesion_connections.bond_flags[conn_idx]
            & crate::cell::adhesion::BOND_FLAG_BARRIER_BALL)
            != 0
        {
            continue;
        }
        let a = state.adhesion_connections.cell_a_index[conn_idx];
        let b = state.adhesion_connections.cell_b_index[conn_idx];
        if a < n && b < n {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }

    let mut visited = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    let mut max_organism_id = state.next_organism_id.saturating_sub(1);

    for start in 0..n {
        if visited[start] {
            continue;
        }

        visited[start] = true;
        queue.push_back(start);
        let mut members = Vec::new();
        let mut min_cell_id = state.cell_ids[start];

        while let Some(cell) = queue.pop_front() {
            members.push(cell);
            min_cell_id = min_cell_id.min(state.cell_ids[cell]);
            for &neighbor in &adjacency[cell] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        let organism_id = min_cell_id.saturating_add(1).max(1);
        max_organism_id = max_organism_id.max(organism_id);
        for cell in members {
            state.organism_ids[cell] = organism_id;
        }
    }

    state.next_organism_id = state
        .next_organism_id
        .max(max_organism_id.saturating_add(1));
}

/// Instantiate and maintain scaffold bonds defined by a genome's `scaffold_rules`.
///
/// - Creates barrier-ball bonds for all same-organism cell pairs that match a rule.
/// - Updates `rest_length_overrides` when they drift from the rule.
/// - Removes bonds whose `scaffold_rule_id` no longer corresponds to any rule.
///
/// Must be called after `sync_development_organism_ids_from_adhesions`.
pub fn resolve_scaffold_rules(
    state: &mut CanonicalState,
    genome: &Genome,
    genome_idx: usize,
    current_time: f32,
) {
    use std::collections::HashSet;

    let live_rule_ids: HashSet<u32> = genome.scaffold_rules.iter().map(|r| r.id).collect();

    // --- pass 1: remove bonds whose rule was deleted ---
    let active_count = state.adhesion_connections.active_count;
    let mut to_remove: Vec<usize> = Vec::new();
    for conn_idx in 0..active_count {
        if state.adhesion_connections.is_active[conn_idx] == 0 {
            continue;
        }
        let rule_id = state.adhesion_connections.scaffold_rule_id[conn_idx];
        if rule_id != 0 && !live_rule_ids.contains(&rule_id) {
            to_remove.push(conn_idx);
        }
    }
    for conn_idx in to_remove.into_iter().rev() {
        state
            .adhesion_manager
            .remove_adhesion(&mut state.adhesion_connections, conn_idx);
    }

    // --- pass 2: create / update bonds from live rules ---
    for rule in &genome.scaffold_rules {
        let is_structural = matches!(
            &rule.endpoint_a,
            crate::genome::CellAddressSelector::ByLineageHashOrMode { .. }
                | crate::genome::CellAddressSelector::ByOrganismCellId(_)
        ) || matches!(
            &rule.endpoint_b,
            crate::genome::CellAddressSelector::ByLineageHashOrMode { .. }
                | crate::genome::CellAddressSelector::ByOrganismCellId(_)
        );

        if is_structural {
            // Specific bond: exactly ONE bond per organism instance.
            // Find the living lineage-hash representative for each endpoint within
            // each organism, then create/maintain that single bond.
            let mut organism_ids: Vec<u32> = (0..state.cell_count)
                .filter(|&i| state.genome_ids[i] == genome_idx && state.organism_ids[i] != 0)
                .map(|i| state.organism_ids[i])
                .collect();
            organism_ids.sort_unstable();
            organism_ids.dedup();

            for org_id in organism_ids {
                let Some(ca) =
                    find_structural_match_in_org(state, genome_idx, org_id, &rule.endpoint_a)
                else {
                    continue;
                };
                let Some(cb) =
                    find_structural_match_in_org(state, genome_idx, org_id, &rule.endpoint_b)
                else {
                    continue;
                };
                if ca == cb {
                    continue;
                }

                create_or_update_scaffold_bond(
                    state,
                    ca,
                    cb,
                    rule.id,
                    rule.rest_length,
                    current_time,
                );
            }
        } else {
            // ── Pattern rule (ByModeIndex etc.): nearest-neighbour per source, both passes ──
            let cells_a: Vec<usize> = (0..state.cell_count)
                .filter(|&i| {
                    state.genome_ids[i] == genome_idx
                        && selector_matches(state, i, &rule.endpoint_a)
                })
                .collect();
            let cells_b: Vec<usize> = (0..state.cell_count)
                .filter(|&i| {
                    state.genome_ids[i] == genome_idx
                        && selector_matches(state, i, &rule.endpoint_b)
                })
                .collect();

            if rule.endpoint_a == rule.endpoint_b {
                let mut organism_ids: Vec<u32> =
                    cells_a.iter().map(|&i| state.organism_ids[i]).collect();
                organism_ids.sort_unstable();
                organism_ids.dedup();

                for org_id in organism_ids {
                    if org_id == 0 {
                        continue;
                    }
                    let mut cells: Vec<usize> = cells_a
                        .iter()
                        .copied()
                        .filter(|&i| state.organism_ids[i] == org_id)
                        .collect();
                    if cells.len() < 2 {
                        continue;
                    }
                    cells.sort_by_key(|&i| {
                        (
                            state.organism_cell_ids[i],
                            state.lineage_depths[i],
                            state.cell_ids[i],
                            i,
                        )
                    });

                    for idx in 0..cells.len() {
                        let ca = cells[idx];
                        let cb = cells[(idx + 1) % cells.len()];
                        if ca == cb {
                            continue;
                        }
                        create_or_update_scaffold_bond(
                            state,
                            ca,
                            cb,
                            rule.id,
                            rule.rest_length,
                            current_time,
                        );
                    }
                }
                continue;
            }

            for pass in 0..2u8 {
                let (sources, targets) = if pass == 0 {
                    (&cells_a, &cells_b)
                } else {
                    (&cells_b, &cells_a)
                };

                for &ca in sources {
                    let best_cb = targets
                        .iter()
                        .copied()
                        .filter(|&cb| cb != ca && state.organism_ids[ca] == state.organism_ids[cb])
                        .min_by(|&cb1, &cb2| {
                            let preferred_delta = rule.preferred_generation_delta.unsigned_abs();
                            let generation_error = |cb: usize| {
                                state.lineage_depths[ca]
                                    .abs_diff(state.lineage_depths[cb])
                                    .abs_diff(preferred_delta)
                            };
                            generation_error(cb1)
                                .cmp(&generation_error(cb2))
                                .then_with(|| {
                                    state.organism_cell_ids[cb1].cmp(&state.organism_cell_ids[cb2])
                                })
                                .then_with(|| cb1.cmp(&cb2))
                        });

                    let Some(cb) = best_cb else { continue };

                    create_or_update_scaffold_bond(
                        state,
                        ca,
                        cb,
                        rule.id,
                        rule.rest_length,
                        current_time,
                    );
                }
            }
        }
    }
}

pub(crate) fn selector_matches(
    state: &CanonicalState,
    cell_idx: usize,
    selector: &crate::genome::CellAddressSelector,
) -> bool {
    match selector {
        crate::genome::CellAddressSelector::AnyCell => true,
        crate::genome::CellAddressSelector::ByModeIndex(m) => state.mode_indices[cell_idx] == *m,
        crate::genome::CellAddressSelector::ByOrganismCellId(id) => {
            state.organism_cell_ids[cell_idx] == *id
        }
        crate::genome::CellAddressSelector::ByLineageHash(h) => {
            state.lineage_hashes[cell_idx] == *h
        }
        crate::genome::CellAddressSelector::ByLineageHashOrMode {
            lineage_hash,
            mode_index,
            ..
        } => {
            state.lineage_hashes[cell_idx] == *lineage_hash
                || state.mode_indices[cell_idx] == *mode_index
        }
    }
}

/// Find the single living representative of a `ByLineageHashOrMode` selector
/// within a specific organism.
///
/// Priority:
///   1. Exact lineage-hash match (the original cell is still alive).
///   2. BFS through the division log following `preferred_branch_slot` at each
///      generation — stops at the first still-living descendant on that branch.
///   3. Mode-only fallback within the organism if the lineage is unresolvable.
///
/// Returns exactly one cell index, or `None`.
fn find_structural_match_in_org(
    state: &CanonicalState,
    genome_idx: usize,
    organism_id: u32,
    selector: &crate::genome::CellAddressSelector,
) -> Option<usize> {
    use crate::genome::CellAddressSelector;
    if let CellAddressSelector::ByOrganismCellId(id) = selector {
        if *id == 0 {
            return None;
        }
        return (0..state.cell_count).find(|&i| {
            state.genome_ids[i] == genome_idx
                && state.organism_ids[i] == organism_id
                && state.organism_cell_ids[i] == *id
        });
    }

    let CellAddressSelector::ByLineageHashOrMode {
        lineage_hash,
        mode_index,
        preferred_branch_slot,
    } = selector
    else {
        return None;
    };

    // 1. Exact lineage hash within this organism.
    for i in 0..state.cell_count {
        if state.genome_ids[i] == genome_idx
            && state.organism_ids[i] == organism_id
            && state.lineage_hashes[i] == *lineage_hash
        {
            return Some(i);
        }
    }

    // 2. BFS through the division log following the preferred branch slot.
    let mut parent_to_live: std::collections::HashMap<u64, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..state.cell_count {
        if state.genome_ids[i] == genome_idx && state.organism_ids[i] == organism_id {
            parent_to_live
                .entry(state.parent_lineage_hashes[i])
                .or_default()
                .push(i);
        }
    }

    let mut frontier: Vec<u64> = vec![*lineage_hash];
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    const MAX_GENERATIONS: usize = 24;

    for _ in 0..MAX_GENERATIONS {
        if frontier.is_empty() {
            break;
        }
        let mut next_frontier: Vec<u64> = Vec::new();
        let mut preferred_result: Option<usize> = None;
        let mut any_result: Option<usize> = None;

        for ancestor_hash in frontier.drain(..) {
            if !visited.insert(ancestor_hash) {
                continue;
            }
            if let Some(children) = parent_to_live.get(&ancestor_hash) {
                for &ci in children {
                    if state.lineage_branch_slots[ci] == *preferred_branch_slot {
                        preferred_result = Some(ci);
                    } else if any_result.is_none() {
                        any_result = Some(ci);
                    }
                    next_frontier.push(state.lineage_hashes[ci]);
                }
            }
            if let Some(&(child_a_hash, child_b_hash)) = state.division_log.get(&ancestor_hash) {
                next_frontier.push(child_a_hash);
                next_frontier.push(child_b_hash);
            }
        }

        if let Some(result) = preferred_result.or(any_result) {
            return Some(result);
        }
        frontier = next_frontier;
    }

    // 3. Mode-only fallback within this organism.
    (0..state.cell_count).find(|&i| {
        state.genome_ids[i] == genome_idx
            && state.organism_ids[i] == organism_id
            && state.mode_indices[i] == *mode_index
    })
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
pub fn compute_collision_forces(
    state: &mut CanonicalState,
    collision_pairs: &[CollisionPair],
    config: &PhysicsConfig,
) {
    for i in 0..state.cell_count {
        state.forces[i] = Vec3::ZERO;
        state.torques[i] = Vec3::ZERO;
    }

    const FRICTION_COEFF: f32 = 0.3;

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
        let total_force_magnitude =
            (spring_force_magnitude + damping_force_magnitude).clamp(-10000.0, 10000.0);
        let force = total_force_magnitude * pair.normal;

        state.forces[idx_b] += force;
        state.forces[idx_a] -= force;

        // ---- Rolling / sliding friction ----
        // Contact point vectors from each cell centre to the contact point.
        // pair.normal points from A toward B, so the contact point is on A's surface.
        let radius_a = state.radii[idx_a];
        let radius_b = state.radii[idx_b];
        let r_a = -pair.normal * radius_a; // from A centre to contact
        let r_b = pair.normal * radius_b; // from B centre to contact

        // Surface velocity at the contact point for each cell:
        //   v_contact = v_linear + omega x r
        let omega_a = state.angular_velocities[idx_a];
        let omega_b = state.angular_velocities[idx_b];
        let v_contact_a = state.velocities[idx_a] + omega_a.cross(r_a);
        let v_contact_b = state.velocities[idx_b] + omega_b.cross(r_b);

        // Relative slip velocity at the contact point, tangential component only
        let v_slip = v_contact_a - v_contact_b;
        let v_slip_tangential = v_slip - v_slip.dot(pair.normal) * pair.normal;
        let slip_speed = v_slip_tangential.length();

        if slip_speed > 0.0001 {
            let friction_dir = -v_slip_tangential / slip_speed;
            let friction_mag = (FRICTION_COEFF * total_force_magnitude.abs())
                .min(slip_speed * combined_stiffness * 0.1);
            let friction_force = friction_dir * friction_mag;

            // Apply tangential friction force to both cells
            state.forces[idx_a] += friction_force;
            state.forces[idx_b] -= friction_force;

            // Torque on each cell: tau = r x F_friction
            state.torques[idx_a] += r_a.cross(friction_force);
            state.torques[idx_b] += r_b.cross(-friction_force);
        }
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
        if distance_from_origin < 0.0001 {
            continue;
        }

        let r_hat = state.positions[i] / distance_from_origin;

        if distance_from_origin > soft_zone_start {
            let penetration =
                ((distance_from_origin - soft_zone_start) / soft_zone_thickness).clamp(0.0, 1.0);
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
        if masses[i] <= 0.0 || !masses[i].is_finite() || radii[i] <= 0.0 {
            continue;
        }

        let moment_of_inertia = 0.4 * masses[i] * radii[i] * radii[i];
        if moment_of_inertia > 0.0 {
            let angular_acceleration = torques[i] / moment_of_inertia;
            angular_velocities[i] =
                (angular_velocities[i] + angular_acceleration * dt) * angular_damping_factor;
        }
    }
}

/// Apply myocyte contraction: compute per-cell contraction values for Myocyte cells (cell_type == 9)
///
/// Each myocyte computes its contraction amount and stores it in state.muscle_contractions.
/// The adhesion force computation then uses these per-cell values to scale each cell's
/// half of the bond independently.
///
/// In pulse mode: oscillates between contracted and relaxed using a sine wave timer.
/// Pulse A contracts when sin(time * rate * 2pi) >= 0, Pulse B when < 0.
/// In signal mode: reads a signal channel and contracts based on threshold.
pub fn apply_myocyte_contraction(state: &mut CanonicalState, genome: &Genome, current_time: f32) {
    // Write per-cell contraction values. Non-myocyte cells get 0.0 (relaxed).
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        let Some(mode) = genome.modes.get(mode_index) else {
            state.muscle_contractions[i] = 0.0;
            continue;
        };

        // Only Myocyte cells (cell_type == 9) get non-zero contraction
        if mode.cell_type != 9 {
            state.muscle_contractions[i] = 0.0;
            continue;
        }

        // Determine effective contraction amount
        let contraction = if mode.myocyte_use_signal {
            // Signal-based mode: read from the specific channel
            let channel = (mode.myocyte_signal_channel as usize).min(15);
            let signal_value = state
                .signal_channels
                .get(i * 16 + channel)
                .copied()
                .flatten()
                .unwrap_or(0.0);
            if signal_value >= mode.myocyte_threshold {
                mode.myocyte_contraction_above
            } else {
                mode.myocyte_contraction_below
            }
        } else {
            // Phased timer mode: oscillate based on current_time
            let wave = (current_time * mode.myocyte_pulse_rate * 2.0 * std::f32::consts::PI).sin();
            let is_active_phase = if mode.myocyte_pulse_phase == 0 {
                wave >= 0.0 // Pulse A
            } else {
                wave < 0.0 // Pulse B
            };
            if is_active_phase {
                mode.myocyte_contraction
            } else {
                0.0
            }
        };

        state.muscle_contractions[i] = contraction.clamp(0.0, 1.0);
    }
}

/// Apply myocyte peristaltic grip as friction forces (CPU preview).
///
/// Uses the per-cell contraction value already computed by `apply_myocyte_contraction`.
/// Grip force = -velocity * (1 - exp(-grip * medium_scale)) * mass / dt, equivalent to
/// the exponential velocity-damping used in the GPU position_update shader.
pub fn apply_myocyte_grip(state: &mut CanonicalState, genome: &Genome, dt: f32, in_water: &[bool]) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        let Some(mode) = genome.modes.get(mode_index) else {
            continue;
        };
        if mode.cell_type != 9 {
            continue; // only myocytes
        }
        let grip_contracted = mode.myocyte_grip_contracted;
        let grip_extended = mode.myocyte_grip_extended;
        if grip_contracted <= 0.0 && grip_extended <= 0.0 {
            continue;
        }

        let contraction = state.muscle_contractions[i];
        let grip = grip_extended + (grip_contracted - grip_extended) * contraction;
        if grip <= 0.001 {
            continue;
        }

        let medium_scale = if in_water.get(i).copied().unwrap_or(false) {
            1.0_f32
        } else {
            0.05_f32 // air/substrate — gravity dominates, minor grip only
        };
        let effective_grip = grip * medium_scale;
        let retain = (-effective_grip * dt).exp();

        // Apply as a force: F = vel * (retain - 1) * mass / dt
        // This will cause velocity to be multiplied by `retain` after integration.
        let vel = state.velocities[i];
        let mass = state.masses[i].max(0.001);
        state.forces[i] += vel * (retain - 1.0) * (mass / dt);
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
                    let channel = mode.flagellocyte_signal_channel.clamp(0, 7) as usize;
                    let signal_value = state.signal_channels[i * 16 + channel].unwrap_or(0.0);
                    if signal_value >= mode.flagellocyte_threshold_c {
                        mode.flagellocyte_speed_b
                    } else {
                        mode.flagellocyte_speed_a
                    }
                } else {
                    mode.swim_force
                };

                if effective_speed <= 0.0 {
                    continue;
                }

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
pub fn consume_swim_nutrients(state: &mut CanonicalState, genome: &Genome, dt: f32) -> Vec<usize> {
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
                    let channel = mode.flagellocyte_signal_channel.clamp(0, 7) as usize;
                    let signal_value = state.signal_channels[i * 16 + channel].unwrap_or(0.0);
                    if signal_value >= mode.flagellocyte_threshold_c {
                        mode.flagellocyte_speed_b
                    } else {
                        mode.flagellocyte_speed_a
                    }
                } else {
                    mode.swim_force
                };
                if effective_speed <= 0.0 {
                    continue;
                }

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
///
/// Embryocytes (cell_type == 10) skip this function entirely - their energy
/// comes exclusively from the reserve field (see `update_embryocyte_reserve_burn`
/// and `transport_nutrients_through_adhesions`).
///
/// Non-Embryocyte cells that have a non-zero reserve burn from reserve first:
/// their metabolic drain is satisfied from reserve before touching normal nutrients.
pub fn update_nutrient_growth(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    const BASE_METABOLISM_RATE: f32 = 1.0; // nutrients/sec drain for non-auto-gain cells
    const AUTO_GAIN_RATE: f32 = 20.0; // nutrients/sec for Test, Phagocyte, Photocyte
    const OCULOCYTE_SENSE_CONSUMPTION_RATE: f32 = 0.08; // nutrients/sec per unit sense_range (default 25 range = 2.0/sec)

    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Embryocytes skip normal metabolism entirely - reserve-only energy system
            if mode.cell_type == 10 {
                continue;
            }

            let can_auto_gain = mode.cell_type == 0  // Test
                             || mode.cell_type == 2  // Phagocyte
                             || mode.cell_type == 3  // Photocyte
                             || mode.cell_type == 11; // Devorocyte (simulates predation as auto-gain in preview)
            let is_oculocyte = mode.cell_type == 7;

            let current_nutrients = state.nutrients[i];
            let split_nutrient_threshold = state.split_nutrient_thresholds[i];

            if can_auto_gain {
                // Auto-gain nutrients up to split threshold * 2
                // Test cells: pure auto-gain, no drain (matches mass_accum.wgsl)
                // Phagocyte/Photocyte: gain - base drain (matches GPU: gain via specialized shader, drain via nutrient_transport.wgsl)
                let is_test_cell = mode.cell_type == 0;
                // Cap threshold at 200 before doubling so the "never split" sentinel
                // (threshold > 100 for normal cells, > 200 for lipocytes) doesn't inflate
                // the nutrient cap to an absurd value.
                let max_nutrients = split_nutrient_threshold.min(200.0) * 2.0;
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
                // Non-auto-gain cells lose nutrients to metabolism.
                // If a reserve is present, burn it first to offset the drain before
                // touching normal nutrients (reserve-first metabolism for descendants
                // of Embryocytes).
                let base_loss = BASE_METABOLISM_RATE * dt;
                let sense_loss = if is_oculocyte {
                    mode.oculocyte_ray_length * OCULOCYTE_SENSE_CONSUMPTION_RATE * dt
                } else {
                    0.0
                };
                let total_loss = base_loss + sense_loss;

                // Attempt to cover drain from reserve before nutrients
                let reserve = state.reserves[i];
                if reserve > 0 {
                    // Reserve covers the full drain 1:1. Reserve is stored x1000 (fixed-point)
                    // so burn total_loss * 1000 milli-units, then apply any remainder to nutrients.
                    let reserve_burned = ((total_loss * 1000.0) as u32).min(reserve);
                    state.reserves[i] = reserve.saturating_sub(reserve_burned);
                    let covered = reserve_burned as f32 / 1000.0;
                    let remaining_loss = (total_loss - covered).max(0.0);

                    // Also convert reserve into the nutrient pool when nutrients are low.
                    // This lets cells with inherited reserve actually grow and divide,
                    // not just survive longer. Convert at up to 20/sec, capped by
                    // how much space is left below the split threshold.
                    const RESERVE_TO_NUTRIENT_RATE: f32 = 20.0;
                    let split_threshold = state.split_nutrient_thresholds[i].min(200.0);
                    let nutrient_headroom = (split_threshold - current_nutrients).max(0.0);
                    if nutrient_headroom > 0.0 && state.reserves[i] > 0 {
                        let convert_amount = (RESERVE_TO_NUTRIENT_RATE * dt).min(nutrient_headroom);
                        let convert_fixed = (convert_amount * 1000.0) as u32;
                        let actual_convert = convert_fixed.min(state.reserves[i]);
                        state.reserves[i] = state.reserves[i].saturating_sub(actual_convert);
                        let nutrients_gained = actual_convert as f32 / 1000.0;
                        let new_nutrients =
                            (current_nutrients + nutrients_gained - remaining_loss).max(0.0);
                        if new_nutrients != current_nutrients {
                            state.nutrients[i] = new_nutrients;
                            let new_mass = 1.0 + new_nutrients / 100.0;
                            state.masses[i] = new_mass;
                            let new_radius = new_mass.min(mode.max_cell_size).clamp(0.5, 2.0);
                            if new_radius != state.radii[i] {
                                state.radii[i] = new_radius;
                                state.masses_changed = true;
                            }
                        }
                    } else {
                        let new_nutrients = (current_nutrients - remaining_loss).max(0.0);
                        if new_nutrients != current_nutrients {
                            state.nutrients[i] = new_nutrients;
                            let new_mass = 1.0 + new_nutrients / 100.0;
                            state.masses[i] = new_mass;
                            let new_radius = new_mass.min(mode.max_cell_size).clamp(0.5, 2.0);
                            if new_radius != state.radii[i] {
                                state.radii[i] = new_radius;
                                state.masses_changed = true;
                            }
                        }
                    }
                } else {
                    // No reserve: standard drain
                    let new_nutrients = (current_nutrients - total_loss).max(0.0);
                    if new_nutrients != current_nutrients {
                        state.nutrients[i] = new_nutrients;
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
}

/// Burn Embryocyte reserve for free (detached) Embryocytes at 10 units/sec,
/// and tick the accumulation timer for attached Embryocytes.
///
/// - **Attached** (>=1 active adhesion): increment `embryocyte_timers[i]` by `dt`.
/// - **Free** (no adhesions): burn `reserve` at 10 units/sec.
///
/// This runs after `transport_nutrients_through_adhesions` so reserve
/// has already been topped up by nutrient transport this tick.
pub fn update_embryocyte_reserve_burn(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    const RESERVE_BURN_RATE: f32 = 10.0; // units/sec when free

    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        let Some(mode) = genome.modes.get(mode_index) else {
            continue;
        };
        if mode.cell_type != 10 {
            continue;
        }

        let adhesion_count = state
            .adhesion_manager
            .count_active_adhesions(i, &state.adhesion_connections);
        if adhesion_count > 0 {
            // Attached: tick the accumulation timer
            state.embryocyte_timers[i] += dt;
        } else {
            // Free: burn reserve (stored x1000 fixed-point, so burn rate * 1000)
            let burn = (RESERVE_BURN_RATE * dt * 1000.0) as u32;
            state.reserves[i] = state.reserves[i].saturating_sub(burn);
            // Reset timer (will restart when re-attached)
            state.embryocyte_timers[i] = 0.0;
        }
    }
}

/// Check AND-logic release triggers for attached Embryocytes.
///
/// All *enabled* triggers must be satisfied simultaneously for release to occur.
/// When triggered, all adhesions for the cell are dropped (it becomes free).
///
/// Triggers:
/// - **Timer**: `embryocyte_timers[i] >= mode.embryocyte_release_timer`
/// - **Threshold**: `reserves[i] >= mode.embryocyte_threshold_value`
/// - **Signal**: signal on `embryocyte_signal_channel` >= `embryocyte_signal_value`
///
/// If no triggers are enabled, the cell never self-releases (manual-only).
pub fn check_embryocyte_release_triggers(state: &mut CanonicalState, genome: &Genome) {
    let mut cells_to_release: Vec<usize> = Vec::new();

    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        let Some(mode) = genome.modes.get(mode_index) else {
            continue;
        };
        if mode.cell_type != 10 {
            continue;
        }

        // Must have at least one adhesion to "release" from
        if state
            .adhesion_manager
            .count_active_adhesions(i, &state.adhesion_connections)
            == 0
        {
            continue;
        }

        // If no triggers enabled, skip (never self-releases)
        let any_enabled = mode.embryocyte_use_timer
            || mode.embryocyte_use_threshold
            || mode.embryocyte_use_signal;
        if !any_enabled {
            continue;
        }

        // AND logic: all enabled triggers must be satisfied
        let mut all_satisfied = true;

        if mode.embryocyte_use_timer {
            if state.embryocyte_timers[i] < mode.embryocyte_release_timer {
                all_satisfied = false;
            }
        }

        if all_satisfied && mode.embryocyte_use_threshold {
            if state.reserves[i] / 1000 < mode.embryocyte_threshold_value {
                all_satisfied = false;
            }
        }

        if all_satisfied && mode.embryocyte_use_signal {
            let channel = (mode.embryocyte_signal_channel as usize).min(15);
            let signal_val = state
                .signal_channels
                .get(i * 16 + channel)
                .copied()
                .flatten()
                .unwrap_or(0.0);
            if signal_val < mode.embryocyte_signal_value {
                all_satisfied = false;
            }
        }

        if all_satisfied {
            cells_to_release.push(i);
        }
    }

    // Drop all adhesions for triggered cells
    for i in cells_to_release {
        state
            .adhesion_manager
            .remove_all_connections_for_cell(&mut state.adhesion_connections, i);
        // Reset accumulation timer
        state.embryocyte_timers[i] = 0.0;
    }
}

/// Transport nutrients between adhesion-connected cells based on priority ratios.
/// Total outflow per cell is capped at TRANSPORT_RATE nutrients/sec regardless of connection count.
/// Transfer is lerped (smoothed) to prevent oscillation.
pub fn transport_nutrients_through_adhesions(state: &mut CanonicalState, genome: &Genome, dt: f32) {
    const PRIORITY_BOOST: f32 = 10.0;
    /// Max nutrients/sec a cell can send OR receive in total across all connections.
    /// Raised from 30 to 100 so non-vascular organisms can diffuse nutrients through
    /// the body without needing vasculocytes. The pressure gradient (nutrient imbalance
    /// between cells) still drives flow direction; this cap just stops being the bottleneck.
    const TRANSPORT_RATE: f32 = 100.0;
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
        desired: f32,  // nutrients/sec, clamped to rate_cap
        rate_cap: f32, // per-connection rate cap (higher for embryocyte connections)
    }
    let mut conn_flows: Vec<ConnFlow> = Vec::new();

    let connections = &state.adhesion_connections;
    for i in 0..connections.is_active.len() {
        if connections.is_active[i] == 0 {
            continue;
        }
        if (connections.bond_flags[i] & crate::cell::adhesion::BOND_FLAG_BARRIER_BALL) != 0 {
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

        let is_embryo_a_pass1 = mode_a.cell_type == 10;
        let is_embryo_b_pass1 = mode_b.cell_type == 10;

        // Embryocyte fill rate: scale the rate cap by priority so high-priority
        // embryocytes can receive faster than the base 30/sec.
        // Embryocytes are always a pure sink - treat their "pressure" as 0 so the
        // sender always pushes toward them as long as it has nutrients. The rate cap
        // (not the pressure diff) is what limits fill speed.
        let embryo_rate_cap = if is_embryo_b_pass1 {
            TRANSPORT_RATE * mode_b.nutrient_priority.max(1.0)
        } else if is_embryo_a_pass1 {
            TRANSPORT_RATE * mode_a.nutrient_priority.max(1.0)
        } else {
            TRANSPORT_RATE
        };

        // For embryocyte receivers, bypass the pressure-equilibrium formula entirely.
        // The embryocyte is a pure sink - the sender should push at the full rate cap
        // as long as it has nutrients. The pressure-diff formula caps flow at
        // nutrients_a / priority_a (e.g. 20/3.5 ~= 5.7/sec), which is far too slow.
        // Instead, use the full embryo_rate_cap as desired so the sender drains into
        // the embryocyte as fast as the rate cap allows.
        let effective_nutrients_b = if is_embryo_b_pass1 { 0.0 } else { nutrients_b };
        let effective_nutrients_a = if is_embryo_a_pass1 { 0.0 } else { nutrients_a };

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

        let desired = if is_embryo_b_pass1 {
            // Embryocyte receiver: always push at full rate cap (A->B direction)
            embryo_rate_cap
        } else if is_embryo_a_pass1 {
            // Embryocyte sender: will be blocked later, but set negative for direction
            -embryo_rate_cap
        } else {
            let pressure_diff =
                effective_nutrients_a / priority_a - effective_nutrients_b / priority_b;
            pressure_diff.clamp(-embryo_rate_cap, embryo_rate_cap)
        };

        conn_flows.push(ConnFlow {
            conn_idx: i,
            cell_a,
            cell_b,
            desired,
            rate_cap: embryo_rate_cap,
        });
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
    // Snapshot all nutrient values before applying any transfers so the order of
    // connection processing doesn't affect the result (matches GPU single-snapshot semantics).
    let nutrients_snap: Vec<f32> = state.nutrients[..n].to_vec();
    let mut nutrient_deltas = vec![0.0f32; n];
    let lerp_t = (LERP_SPEED * dt).min(1.0);

    for cf in &conn_flows {
        let cell_a = cf.cell_a;
        let cell_b = cf.cell_b;

        // Scale outflow so the sending cell never exceeds its connection's rate cap total.
        let scale = if cf.desired > 0.0 && total_out[cell_a] > cf.rate_cap {
            cf.rate_cap / total_out[cell_a]
        } else if cf.desired < 0.0 && total_out[cell_b] > cf.rate_cap {
            cf.rate_cap / total_out[cell_b]
        } else {
            1.0
        };

        // Target transfer this step (nutrients), lerped toward desired
        let target_per_sec = cf.desired * scale;
        let transfer = target_per_sec * lerp_t * dt;

        // Clamp by available nutrients and receiver capacity
        let mode_a = genome.modes.get(state.mode_indices[cell_a]).unwrap();
        let mode_b = genome.modes.get(state.mode_indices[cell_b]).unwrap();
        let min_a = if mode_a.prioritize_when_low {
            10.0
        } else {
            0.0
        };
        let min_b = if mode_b.prioritize_when_low {
            10.0
        } else {
            0.0
        };
        let max_a = state.split_nutrient_thresholds[cell_a].min(200.0) * 2.0;
        let max_b = state.split_nutrient_thresholds[cell_b].min(200.0) * 2.0;

        let is_embryo_a = mode_a.cell_type == 10;
        let is_embryo_b = mode_b.cell_type == 10;

        let transfer_blocked = (transfer > 0.0 && is_embryo_a) || (transfer < 0.0 && is_embryo_b);
        if transfer_blocked {
            if dt > 0.0 {
                state.adhesion_connections.connection_flow_rates[cf.conn_idx] = 0.0;
            }
            continue;
        }

        // Use snapshotted nutrients for can_give/can_recv so connection order doesn't matter
        let snap_a = nutrients_snap[cell_a];
        let snap_b = nutrients_snap[cell_b];

        let actual = if transfer > 0.0 {
            let can_give = (snap_a - min_a).max(0.0);
            let can_recv = if is_embryo_b {
                let reserve_space =
                    65_535_000u32.saturating_sub(state.reserves[cell_b]) as f32 / 1000.0;
                reserve_space
            } else {
                (max_b - snap_b).max(0.0)
            };
            transfer.min(can_give).min(can_recv)
        } else {
            let can_give = (snap_b - min_b).max(0.0);
            let can_recv = if is_embryo_a {
                let reserve_space =
                    65_535_000u32.saturating_sub(state.reserves[cell_a]) as f32 / 1000.0;
                reserve_space
            } else {
                (max_a - snap_a).max(0.0)
            };
            transfer.max(-(can_give.min(can_recv)))
        };

        // Route the transfer: if the receiver is an Embryocyte, add to reserve instead of nutrients.
        if transfer > 0.0 && is_embryo_b {
            // A->B, B is Embryocyte: add to B's reserve (x1000 fixed-point)
            let gained = (actual.max(0.0) * 1000.0) as u32;
            state.reserves[cell_b] = state.reserves[cell_b]
                .saturating_add(gained)
                .min(65_535_000);
            nutrient_deltas[cell_a] -= actual;
            // Don't touch nutrient_deltas[cell_b] - reserve was updated directly
        } else if transfer < 0.0 && is_embryo_a {
            // B->A, A is Embryocyte: add to A's reserve (x1000 fixed-point)
            let gained = ((-actual).max(0.0) * 1000.0) as u32;
            state.reserves[cell_a] = state.reserves[cell_a]
                .saturating_add(gained)
                .min(65_535_000);
            nutrient_deltas[cell_b] -= -actual;
            // Don't touch nutrient_deltas[cell_a]
        } else {
            nutrient_deltas[cell_a] -= actual;
            nutrient_deltas[cell_b] += actual;
        }

        // Store actual flow rate (nutrients/sec, positive = A->B) for display
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
                let max_size = genome
                    .modes
                    .get(state.mode_indices[i])
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
pub fn form_glueocyte_contact_bonds(
    state: &mut CanonicalState,
    genome: &Genome,
    current_time: f32,
) {
    let collision_pairs = detect_collisions(state);

    for pair in collision_pairs {
        let idx_a = pair.index_a;
        let idx_b = pair.index_b;
        if idx_a >= state.cell_count || idx_b >= state.cell_count {
            continue;
        }

        if state
            .adhesion_manager
            .are_cells_connected(&state.adhesion_connections, idx_a, idx_b)
        {
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

        let is_glue_a = genome
            .modes
            .get(mode_a)
            .map(|m| m.cell_type == 6)
            .unwrap_or(false);
        let is_glue_b = genome
            .modes
            .get(mode_b)
            .map(|m| m.cell_type == 6)
            .unwrap_or(false);

        if !is_glue_a && !is_glue_b {
            continue;
        }

        // Skip if the Glueocyte(s) in this pair have cell adhesion disabled
        let cell_adhesion_a = genome
            .modes
            .get(mode_a)
            .map(|m| !is_glue_a || m.glueocyte_cell_adhesion)
            .unwrap_or(true);
        let cell_adhesion_b = genome
            .modes
            .get(mode_b)
            .map(|m| !is_glue_b || m.glueocyte_cell_adhesion)
            .unwrap_or(true);
        if !cell_adhesion_a || !cell_adhesion_b {
            continue;
        }

        // Skip if either glueocyte has self-adhesion disabled.
        // In preview mode all cells belong to the same organism, so self-adhesion
        // off means no cell-to-cell bonding at all in this context.
        let self_adhesion_a = genome
            .modes
            .get(mode_a)
            .map(|m| !is_glue_a || m.glueocyte_self_adhesion)
            .unwrap_or(true);
        let self_adhesion_b = genome
            .modes
            .get(mode_b)
            .map(|m| !is_glue_b || m.glueocyte_self_adhesion)
            .unwrap_or(true);
        if !self_adhesion_a || !self_adhesion_b {
            continue;
        }

        let adhesions_a = state
            .adhesion_manager
            .count_active_adhesions(idx_a, &state.adhesion_connections);
        let adhesions_b = state
            .adhesion_manager
            .count_active_adhesions(idx_b, &state.adhesion_connections);

        let max_a = genome
            .modes
            .get(mode_a)
            .map(|m| m.max_adhesions as usize)
            .unwrap_or(10);
        let max_b = genome
            .modes
            .get(mode_b)
            .map(|m| m.max_adhesions as usize)
            .unwrap_or(10);

        if adhesions_a >= max_a || adhesions_b >= max_b {
            continue;
        }

        let glue_mode = if is_glue_a { mode_a } else { mode_b };
        let _ = state.adhesion_manager.add_ball_joint(
            &mut state.adhesion_connections,
            idx_a,
            idx_b,
            glue_mode,
            current_time,
            crate::cell::adhesion::BOND_FLAG_GLUEOCYTE
                | crate::cell::adhesion::BOND_FLAG_BARRIER_BALL,
        );
    }
}

/// Kill any organism that is numerically exploding due to conflicting adhesion constraints.
///
/// Detection: after velocity integration + constraint substeps, scan every cell.
/// A cell is "frantic" if its speed exceeds `FRENZY_SPEED_THRESHOLD`. If the fraction
/// of frantic cells in a connected organism exceeds `FRENZY_FRACTION_THRESHOLD`, the
/// entire organism is removed.
///
/// Connected components are found via BFS over active adhesion connections - the same
/// topology the adhesion solver uses, so the kill boundary matches the physical problem.
///
/// Isolated single cells are never killed by this check (they can legitimately move fast).
pub fn kill_frenzied_organisms(state: &mut CanonicalState) {
    // Speed above which a cell is considered "frantic".
    // Normal adhesion-driven motion tops out around 20-30 units/sec.
    // Exploding constraint fights routinely hit 200-1000+.
    const FRENZY_SPEED_THRESHOLD: f32 = 150.0;
    // Fraction of an organism's cells that must be frantic before the whole thing dies.
    // 0.5 = majority vote: avoids killing organisms where one cell briefly spikes.
    const FRENZY_FRACTION_THRESHOLD: f32 = 0.5;
    // Minimum organism size to apply the check - don't kill isolated single cells.
    const MIN_ORGANISM_SIZE: usize = 2;

    let n = state.cell_count;
    if n == 0 {
        return;
    }

    // --- Step 1: mark frantic cells ---
    let mut is_frantic = vec![false; n];
    for i in 0..n {
        if state.velocities[i].length_squared() > FRENZY_SPEED_THRESHOLD * FRENZY_SPEED_THRESHOLD {
            is_frantic[i] = true;
        }
    }

    // Fast-exit: if no frantic cells, nothing to do.
    if !is_frantic.iter().any(|&f| f) {
        return;
    }

    // --- Step 2: BFS connected-component labeling over active adhesions ---
    // component[i] = root index of cell i's organism (-1 = unvisited)
    let mut component = vec![usize::MAX; n];
    // Build adjacency from active adhesion connections (undirected).
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for idx in 0..state.adhesion_connections.is_active.len() {
        if state.adhesion_connections.is_active[idx] == 0 {
            continue;
        }
        let a = state.adhesion_connections.cell_a_index[idx];
        let b = state.adhesion_connections.cell_b_index[idx];
        if a < n && b < n {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    let mut organisms: Vec<Vec<usize>> = Vec::new(); // list of components (each = cell indices)
    let mut queue = std::collections::VecDeque::new();

    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        // BFS from `start`
        let org_id = organisms.len();
        let mut members = Vec::new();
        component[start] = org_id;
        queue.push_back(start);
        while let Some(cell) = queue.pop_front() {
            members.push(cell);
            for &neighbor in &adj[cell] {
                if component[neighbor] == usize::MAX {
                    component[neighbor] = org_id;
                    queue.push_back(neighbor);
                }
            }
        }
        organisms.push(members);
    }

    // --- Step 3: check each organism and collect cells to kill ---
    let mut to_kill: Vec<usize> = Vec::new();
    for members in &organisms {
        if members.len() < MIN_ORGANISM_SIZE {
            continue;
        }
        let frantic_count = members.iter().filter(|&&i| is_frantic[i]).count();
        let fraction = frantic_count as f32 / members.len() as f32;
        if fraction >= FRENZY_FRACTION_THRESHOLD {
            to_kill.extend_from_slice(members);
        }
    }

    if !to_kill.is_empty() {
        state.remove_cells(&to_kill);
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

    state
        .spatial_grid
        .rebuild(&state.positions, state.cell_count);

    let collisions = detect_collisions(state);
    compute_collision_forces(state, &collisions, config);

    state.update_adhesion_settings_cache(genome);
    state.update_membrane_stiffness_from_genome(genome);

    // Apply myocyte contraction: temporarily modify rest lengths for myocyte cells
    apply_myocyte_contraction(state, genome, current_time);

    if state.adhesion_connections.active_count > 0 && !state.cached_adhesion_settings.is_empty() {
        // Apply adhesion expansion: override rest_length to the genome maximum (5.0)
        // when the tool is active, so all bonds appear fully stretched.
        let expanded_settings: Vec<crate::genome::AdhesionSettings>;
        let effective_settings: &[crate::genome::AdhesionSettings] =
            if state.adhesion_expansion_active {
                const ADHESION_MAX_REST_LENGTH: f32 = 5.0;
                expanded_settings = state
                    .cached_adhesion_settings
                    .iter()
                    .map(|s| {
                        let mut s2 = s.clone();
                        s2.rest_length = ADHESION_MAX_REST_LENGTH;
                        s2
                    })
                    .collect();
                &expanded_settings
            } else {
                &state.cached_adhesion_settings
            };

        let bonds_to_break = crate::cell::compute_adhesion_forces_parallel(
            &state.adhesion_connections,
            &state.positions[..state.cell_count],
            &state.velocities[..state.cell_count],
            &state.rotations[..state.cell_count],
            &state.angular_velocities[..state.cell_count],
            &state.masses[..state.cell_count],
            &state.genome_orientations[..state.cell_count],
            effective_settings,
            &mut state.forces[..state.cell_count],
            &mut state.torques[..state.cell_count],
            current_time,
            &state.muscle_contractions[..state.cell_count],
            dt,
        );
        for conn_idx in bonds_to_break {
            state
                .adhesion_manager
                .remove_adhesion(&mut state.adhesion_connections, conn_idx);
        }
    }

    // Skip swim forces in preview mode - flagellocyte thrust is GPU-only
    // apply_swim_forces(state, genome);

    // Skip buoyancy forces in preview mode - buoyocyte rising is GPU-only
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
        // Reuse the same expansion logic for substep iterations
        let expanded_settings_sub: Vec<crate::genome::AdhesionSettings>;
        let effective_settings_sub: &[crate::genome::AdhesionSettings] =
            if state.adhesion_expansion_active {
                const ADHESION_MAX_REST_LENGTH: f32 = 5.0;
                expanded_settings_sub = state
                    .cached_adhesion_settings
                    .iter()
                    .map(|s| {
                        let mut s2 = s.clone();
                        s2.rest_length = ADHESION_MAX_REST_LENGTH;
                        s2
                    })
                    .collect();
                &expanded_settings_sub
            } else {
                &state.cached_adhesion_settings
            };
        for _ in 0..config.constraint_iterations {
            crate::cell::compute_adhesion_substep(
                &state.adhesion_connections,
                &mut state.positions[..cell_count],
                &mut state.velocities[..cell_count],
                &mut state.rotations[..cell_count],
                &mut state.angular_velocities[..cell_count],
                &state.masses[..cell_count],
                &state.genome_orientations[..cell_count],
                effective_settings_sub,
                cell_count,
                dt,
                &state.muscle_contractions[..cell_count],
                config.angular_damping,
            );
        }
    }

    update_nutrient_growth(state, genome, dt);

    // Kill organisms that are numerically exploding from conflicting adhesion constraints.
    // Runs after constraint substeps so velocities reflect the worst-case instability.
    kill_frenzied_organisms(state);

    // Transport nutrients between adhesion-connected cells.
    // Embryocytes: incoming redirected to reserve, outgoing blocked.
    transport_nutrients_through_adhesions(state, genome, dt);

    // Tick Embryocyte timers (attached) and burn reserve (free).
    // Runs after transport so reserve reflects this tick's incoming nutrients.
    update_embryocyte_reserve_burn(state, genome, dt);

    // Check AND-logic release triggers for attached Embryocytes.
    check_embryocyte_release_triggers(state, genome);

    // Consume nutrients for Flagellocyte cells swimming
    // This must happen after nutrient growth so cells can potentially recover
    let dead_cells = consume_swim_nutrients(state, genome, dt);

    // Remove dead cells (those that ran out of nutrients while swimming)
    if !dead_cells.is_empty() {
        state.remove_cells(&dead_cells);
    }

    // Remove any cells that starved, matching GPU death threshold.
    // - Standard cells (non-Embryocyte): die when nutrients < 1.0 AND reserve == 0.
    //   A non-zero reserve extends life (reserve burns first in update_nutrient_growth).
    // - Embryocytes (cell_type == 10): die when reserve == 0 (nutrients are irrelevant).
    const DEATH_NUTRIENT_THRESHOLD: f32 = 1.0;
    let starved: Vec<usize> = (0..state.cell_count)
        .filter(|&i| {
            let mode_index = state.mode_indices[i];
            let is_embryocyte = genome
                .modes
                .get(mode_index)
                .map(|m| m.cell_type == 10)
                .unwrap_or(false);
            if is_embryocyte {
                state.reserves[i] == 0
            } else {
                state.nutrients[i] < DEATH_NUTRIENT_THRESHOLD && state.reserves[i] == 0
            }
        })
        .collect();
    if !starved.is_empty() {
        state.remove_cells(&starved);
    }

    // Form contact adhesion bonds for Glueocyte cells
    form_glueocyte_contact_bonds(state, genome, current_time);

    // Run signal system (oculocyte sensing + BFS propagation)
    let boundary_radius = config.sphere_radius;
    crate::simulation::signal_system::run_signal_system(
        state,
        genome,
        boundary_radius,
        dt,
        current_time,
    );

    // Apply persistent test signals (if any) after normal signal system.
    // Do NOT clear signals first - regulation signals (channels 8-15) must remain intact.
    // run_signal_system already called clear_all_signals at the start of this step, so
    // there is no cross-step accumulation. Test signals simply add on top of the
    // normally-computed oculocyte and regulation signals.
    if let Some(test_signals) = test_signals {
        if !test_signals.is_empty() {
            crate::simulation::signal_system::propagate_test_signals(
                state,
                genome,
                test_signals.to_vec(),
            );
        }
    }

    let max_cells = state.capacity;
    let rng_seed = 12345;

    // Stemocyte cycle completion: read one developmental gradient and either
    // differentiate in place, enter apoptosis, or remain a Stemocyte so the
    // normal division path below can run.
    let mut stemocyte_deaths = Vec::new();
    let mut stemocyte_switches = Vec::new();
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        let Some(mode) = genome.modes.get(mode_index) else {
            continue;
        };
        if mode.cell_type != crate::cell::CellType::Stemocyte as i32 {
            continue;
        }
        let age = current_time - state.birth_times[i];
        if age < state.split_intervals[i] {
            continue;
        }

        let channel = mode.stemocyte_signal_channel.clamp(8, 15) as usize;
        let signal_value = state.signal_channels[i * 16 + channel].unwrap_or(0.0);
        match mode.stemocyte_outcome_for_signal(signal_value) {
            -2 => stemocyte_deaths.push(i),
            target if target >= 0 && (target as usize) < genome.modes.len() => {
                stemocyte_switches.push((i, target as usize));
            }
            _ => {}
        }
    }
    for (cell_index, target) in stemocyte_switches {
        state.mode_indices[cell_index] = target;
        state.split_counts[cell_index] = 0;
        state.birth_times[cell_index] = current_time;
        let new_mode = &genome.modes[target];
        state.split_intervals[cell_index] = new_mode.split_interval;
        state.split_nutrient_thresholds[cell_index] = (new_mode.split_mass - 1.0) * 100.0;
        state.stiffnesses[cell_index] = new_mode.membrane_stiffness;
    }
    if !stemocyte_deaths.is_empty() {
        state.remove_cells(&stemocyte_deaths);
    }

    // Signal-conditional apoptosis: kill cells whose signal meets the apoptosis condition
    let mut apoptosis_cells: Vec<usize> = Vec::new();
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            if mode.apoptosis_signal_channel >= 8 && (mode.apoptosis_signal_channel as usize) <= 15
            {
                let ch = mode.apoptosis_signal_channel as usize;
                let signal_val = state.signal_channels[i * 16 + ch].unwrap_or(0.0);
                let above = signal_val >= mode.apoptosis_signal_threshold;
                let should_die = if mode.apoptosis_signal_invert {
                    !above
                } else {
                    above
                };
                if should_die {
                    apoptosis_cells.push(i);
                }
            }
        }
    }
    if !apoptosis_cells.is_empty() {
        state.remove_cells(&apoptosis_cells);
    }

    // Signal-conditional mode switching: change cell mode without division
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            if mode.mode_switch_signal_channel >= 8
                && (mode.mode_switch_signal_channel as usize) <= 15
                && mode.mode_switch_target >= 0
            {
                let ch = mode.mode_switch_signal_channel as usize;
                let signal_val = state.signal_channels[i * 16 + ch].unwrap_or(0.0);
                let above = signal_val >= mode.mode_switch_signal_threshold;
                let should_switch = if mode.mode_switch_invert {
                    !above
                } else {
                    above
                };
                if should_switch {
                    let target = mode.mode_switch_target as usize;
                    if target < genome.modes.len() {
                        state.mode_indices[i] = target;
                        // Reset split count and copy new mode's per-cell settings,
                        // mirroring what mode_switch.wgsl does on the GPU side.
                        state.split_counts[i] = 0;
                        let new_mode = &genome.modes[target];
                        state.split_intervals[i] = new_mode.split_interval;
                        state.split_nutrient_thresholds[i] = (new_mode.split_mass - 1.0) * 100.0;
                        state.stiffnesses[i] = new_mode.membrane_stiffness;
                    }
                }
            }
        }
    }

    let division_events = division::division_step(state, genome, current_time, max_cells, rng_seed);
    sync_development_organism_ids_from_adhesions(state);
    // Resolve scaffold rules every step so bonds form/persist through resim.
    // genome_idx = 0: preview scene uses a single genome.
    if !genome.scaffold_rules.is_empty() {
        resolve_scaffold_rules(state, genome, 0, current_time);
    }
    division_events
}
