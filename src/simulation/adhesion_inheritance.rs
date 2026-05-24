use glam::{Vec3, Quat, EulerRot};
use crate::cell::{AdhesionZone, classify_bond_direction, MAX_ADHESIONS_PER_CELL};
use crate::simulation::canonical_state::CanonicalState;
use crate::genome::Genome;

/// Handle adhesion inheritance during cell division.
///
/// Matches GPU `lifecycle_division_execute_ring.wgsl` exactly:
/// - Connections are modified IN-PLACE for single-child inheritance (Zone A/B)
/// - Only Zone C (both children) allocates a duplicate connection
/// - Neighbor adhesion indices are never polluted with stale references
/// - Original mode_index is preserved (not changed to child's mode)
/// - Keep flags always use normal child_a/b.keep_adhesion (not after_split variants)
///
/// Zone inheritance rules:
/// - Zone A: Inherit to child B (adhesions pointing opposite to split direction)
/// - Zone B: Inherit to child A (adhesions pointing same as split direction)
/// - Zone C: Inherit to both children (modify original for A, duplicate for B)
pub fn inherit_adhesions_on_division(
    state: &mut CanonicalState,
    genome: &Genome,
    parent_mode_idx: usize,
    child_a_idx: usize,
    child_b_idx: usize,
    parent_genome_orientation: Quat,
    current_time: f32,
    parent_split_count: i32,
    parent_radius: f32,
) {
    let parent_mode = match genome.modes.get(parent_mode_idx) {
        Some(mode) => mode,
        None => return,
    };

    // Use after_split keep flags when this division triggers the max_splits transition,
    // otherwise use the normal keep flags. This is what detaches the egg: mode 0 has
    // child_a_after_split_keep_adhesion = false so the egg spawns free.
    let will_reach_max_splits = parent_mode.max_splits >= 0
        && (parent_split_count + 1) >= parent_mode.max_splits;
    let child_a_keep = if will_reach_max_splits {
        parent_mode.child_a_after_split_keep_adhesion
    } else {
        parent_mode.child_a.keep_adhesion
    };
    let child_b_keep = if will_reach_max_splits {
        parent_mode.child_b_after_split_keep_adhesion
    } else {
        parent_mode.child_b.keep_adhesion
    };

    // CRITICAL: Collect parent's adhesion connections BEFORE clearing indices
    // (child A reuses parent index, so clearing would lose the references)
    let mut parent_connections = Vec::new();
    for slot_idx in 0..MAX_ADHESIONS_PER_CELL {
        let connection_idx = state.adhesion_manager.cell_adhesion_indices[child_a_idx][slot_idx];
        if connection_idx >= 0 {
            parent_connections.push(connection_idx as usize);
        }
    }

    // Clear adhesion indices for both children (rebuilt below)
    state.adhesion_manager.init_cell_adhesion_indices(child_a_idx);
    state.adhesion_manager.init_cell_adhesion_indices(child_b_idx);

    // If neither child keeps adhesions, deactivate all old connections,
    // clean up neighbor references, and return early (matches GPU lines 658-679).
    if !child_a_keep && !child_b_keep {
        for &connection_idx in &parent_connections {
            if connection_idx >= state.adhesion_connections.active_count {
                continue;
            }
            if state.adhesion_connections.is_active[connection_idx] == 0 {
                continue;
            }
            let cell_a = state.adhesion_connections.cell_a_index[connection_idx];
            let cell_b = state.adhesion_connections.cell_b_index[connection_idx];
            let neighbor_idx = if cell_a == child_a_idx { cell_b }
                else if cell_b == child_a_idx { cell_a }
                else { continue };

            state.adhesion_connections.is_active[connection_idx] = 0;
            state.adhesion_manager.remove_adhesion_index(neighbor_idx, connection_idx as i32);
        }
        return;
    }

    // Geometric parameters from genome
    let pitch = parent_mode.parent_split_direction.x.to_radians();
    let yaw = parent_mode.parent_split_direction.y.to_radians();
    let split_rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    let split_dir_local = split_rotation * Vec3::Z;
    let split_dir_normalized = split_dir_local.normalize_or_zero();
    let child_offset = parent_radius * 0.25;
    let child_a_pos_parent_frame = split_dir_normalized * child_offset;
    let child_b_pos_parent_frame = -split_dir_normalized * child_offset;

    // Match GPU: use will_reach_max_splits for orientation selection (already computed above)
    let child_a_orientation_final = if will_reach_max_splits {
        parent_mode.child_a_after_split_orientation
    } else {
        parent_mode.child_a.orientation
    };
    let child_b_orientation_final = if will_reach_max_splits {
        parent_mode.child_b_after_split_orientation
    } else {
        parent_mode.child_b.orientation
    };
    let child_a_orientation_delta = split_rotation * child_a_orientation_final;
    let child_b_orientation_delta = split_rotation * child_b_orientation_final;

    // Child genome orientations (for twist references)
    let child_a_genome_orientation = state.genome_orientations[child_a_idx];
    let child_b_genome_orientation = state.genome_orientations[child_b_idx];

    // Split ratios for zone classification on child side
    let child_a_mode_idx = state.mode_indices[child_a_idx];
    let child_b_mode_idx = state.mode_indices[child_b_idx];
    let child_a_split_ratio = genome.modes.get(child_a_mode_idx)
        .map(|m| m.split_ratio).unwrap_or(0.5);
    let child_b_split_ratio = genome.modes.get(child_b_mode_idx)
        .map(|m| m.split_ratio).unwrap_or(0.5);
    let parent_split_ratio = parent_mode.split_ratio;

    // Track adhesion counts for each child
    let mut child_a_adhesion_count: usize = 0;
    let mut child_b_adhesion_count: usize = 0;

    // Process inherited adhesions (matches GPU lines 722-943)
    for &connection_idx in &parent_connections {
        if connection_idx >= state.adhesion_connections.active_count {
            continue;
        }
        if state.adhesion_connections.is_active[connection_idx] == 0 {
            continue;
        }

        let cell_a = state.adhesion_connections.cell_a_index[connection_idx];
        let cell_b = state.adhesion_connections.cell_b_index[connection_idx];
        let (neighbor_idx, parent_is_a) = if cell_a == child_a_idx {
            (cell_b, true)
        } else if cell_b == child_a_idx {
            (cell_a, false)
        } else {
            continue;
        };

        // Parent's anchor direction for zone classification
        let parent_anchor_dir = if parent_is_a {
            state.adhesion_connections.anchor_direction_a[connection_idx]
        } else {
            state.adhesion_connections.anchor_direction_b[connection_idx]
        };

        let zone = classify_bond_direction(parent_anchor_dir, split_dir_local, parent_split_ratio);

        // Determine which children inherit (matches GPU lines 766-776)
        let (give_to_a, give_to_b) = match zone {
            AdhesionZone::ZoneA => (false, child_b_keep),
            AdhesionZone::ZoneB => (child_a_keep, false),
            AdhesionZone::ZoneC => (child_a_keep, child_b_keep),
        };

        if !give_to_a && !give_to_b {
            // Neither child inherits — deactivate and clean up neighbor reference
            state.adhesion_connections.is_active[connection_idx] = 0;
            state.adhesion_manager.remove_adhesion_index(neighbor_idx, connection_idx as i32);
            continue;
        }

        // Neighbor position in parent frame: use rest_length as the distance, matching
        // the geometric spring equilibrium. The spring enforces equilibrium at rest_length
        // (displacement = (anchor_a - anchor_b) * L/2, and since anchors are antiparallel,
        // |displacement| = rest_length). Using rest_length + radii here would estimate the
        // neighbor at the wrong distance, causing anchor directions to point slightly off and
        // producing variable bond lengths across inherited bonds.
        let conn_rest_length = genome.modes.get(state.adhesion_connections.mode_index[connection_idx])
            .map(|m| m.adhesion_settings.rest_length)
            .unwrap_or(1.0);
        let neighbor_pos_parent_frame = parent_anchor_dir * conn_rest_length;

        // Relative rotation for neighbor anchor calculation (genome-pure)
        let neighbor_genome_orientation = state.genome_orientations[neighbor_idx];
        let relative_rotation = neighbor_genome_orientation.inverse() * parent_genome_orientation;

        if give_to_a && !give_to_b {
            // Only child A inherits — modify connection IN PLACE (matches GPU lines 796-822)
            // Cell indices already correct since child A reuses parent slot
            let child_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame, neighbor_pos_parent_frame, child_a_orientation_delta,
            );
            // Neighbor anchor must be exactly antiparallel to child anchor in world space
            // so the geometric spring equilibrium lands at exactly rest_length.
            // child_anchor is in child's local space; rotate to world via child_orientation_delta,
            // negate, then rotate into neighbor's local space via relative_rotation.
            let child_anchor_world = child_a_orientation_delta * child_anchor;
            let neighbor_anchor = (relative_rotation * (-child_anchor_world)).normalize();

            if parent_is_a {
                state.adhesion_connections.anchor_direction_a[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_b[connection_idx] = neighbor_anchor;
                state.adhesion_connections.twist_reference_a[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_a[connection_idx] =
                    classify_bond_direction(child_anchor, split_dir_local, child_a_split_ratio) as u8;
            } else {
                state.adhesion_connections.anchor_direction_b[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_a[connection_idx] = neighbor_anchor;
                state.adhesion_connections.twist_reference_b[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_b[connection_idx] =
                    classify_bond_direction(child_anchor, split_dir_local, child_a_split_ratio) as u8;
            }

            // Add to child A's adhesion indices
            if child_a_adhesion_count < MAX_ADHESIONS_PER_CELL {
                state.adhesion_manager.cell_adhesion_indices[child_a_idx][child_a_adhesion_count] = connection_idx as i32;
                child_a_adhesion_count += 1;
            }

        } else if give_to_b && !give_to_a {
            // Only child B inherits — modify connection IN PLACE (matches GPU lines 824-852)
            // Update cell index from parent slot to child B
            let child_anchor = calculate_child_anchor_direction(
                child_b_pos_parent_frame, neighbor_pos_parent_frame, child_b_orientation_delta,
            );
            // Enforce antiparallel anchors for correct rest_length equilibrium.
            let child_anchor_world = child_b_orientation_delta * child_anchor;
            let neighbor_anchor = (relative_rotation * (-child_anchor_world)).normalize();

            if parent_is_a {
                state.adhesion_connections.cell_a_index[connection_idx] = child_b_idx;
                state.adhesion_connections.anchor_direction_a[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_b[connection_idx] = neighbor_anchor;
                state.adhesion_connections.twist_reference_a[connection_idx] = child_b_genome_orientation;
                state.adhesion_connections.zone_a[connection_idx] =
                    classify_bond_direction(child_anchor, split_dir_local, child_b_split_ratio) as u8;
            } else {
                state.adhesion_connections.cell_b_index[connection_idx] = child_b_idx;
                state.adhesion_connections.anchor_direction_b[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_a[connection_idx] = neighbor_anchor;
                state.adhesion_connections.twist_reference_b[connection_idx] = child_b_genome_orientation;
                state.adhesion_connections.zone_b[connection_idx] =
                    classify_bond_direction(child_anchor, split_dir_local, child_b_split_ratio) as u8;
            }

            // Add to child B's adhesion indices
            if child_b_adhesion_count < MAX_ADHESIONS_PER_CELL {
                state.adhesion_manager.cell_adhesion_indices[child_b_idx][child_b_adhesion_count] = connection_idx as i32;
                child_b_adhesion_count += 1;
            }

        } else {
            // Zone C: Both inherit — modify original for child A, duplicate for child B
            // (matches GPU lines 854-942)

            // --- Child A gets the original connection (modified in place) ---
            let child_a_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame, neighbor_pos_parent_frame, child_a_orientation_delta,
            );
            // Enforce antiparallel anchors for correct rest_length equilibrium.
            let child_a_anchor_world = child_a_orientation_delta * child_a_anchor;
            let neighbor_anchor_to_a = (relative_rotation * (-child_a_anchor_world)).normalize();

            if parent_is_a {
                state.adhesion_connections.anchor_direction_a[connection_idx] = child_a_anchor;
                state.adhesion_connections.anchor_direction_b[connection_idx] = neighbor_anchor_to_a;
                state.adhesion_connections.twist_reference_a[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_a[connection_idx] =
                    classify_bond_direction(child_a_anchor, split_dir_local, child_a_split_ratio) as u8;
            } else {
                state.adhesion_connections.anchor_direction_b[connection_idx] = child_a_anchor;
                state.adhesion_connections.anchor_direction_a[connection_idx] = neighbor_anchor_to_a;
                state.adhesion_connections.twist_reference_b[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_b[connection_idx] =
                    classify_bond_direction(child_a_anchor, split_dir_local, child_a_split_ratio) as u8;
            }

            if child_a_adhesion_count < MAX_ADHESIONS_PER_CELL {
                state.adhesion_manager.cell_adhesion_indices[child_a_idx][child_a_adhesion_count] = connection_idx as i32;
                child_a_adhesion_count += 1;
            }

            // --- Child B gets a duplicate connection ---
            let child_b_anchor = calculate_child_anchor_direction(
                child_b_pos_parent_frame, neighbor_pos_parent_frame, child_b_orientation_delta,
            );
            // Enforce antiparallel anchors for correct rest_length equilibrium.
            let child_b_anchor_world = child_b_orientation_delta * child_b_anchor;
            let neighbor_anchor_to_b = (relative_rotation * (-child_b_anchor_world)).normalize();

            // Find a free connection slot for the duplicate
            let dup_idx = find_free_connection_slot(&state.adhesion_connections);
            if let Some(dup_idx) = dup_idx {
                let neighbor_mode_idx = state.mode_indices[neighbor_idx];
                let neighbor_split_ratio = genome.modes.get(neighbor_mode_idx)
                    .map(|m| m.split_ratio).unwrap_or(0.5);

                // Keep original mode_index (matches GPU — connection.mode_index is never changed)
                let original_mode_index = state.adhesion_connections.mode_index[connection_idx];

                if parent_is_a {
                    state.adhesion_connections.cell_a_index[dup_idx] = child_b_idx;
                    state.adhesion_connections.cell_b_index[dup_idx] = neighbor_idx;
                    state.adhesion_connections.anchor_direction_a[dup_idx] = child_b_anchor;
                    state.adhesion_connections.anchor_direction_b[dup_idx] = neighbor_anchor_to_b;
                    state.adhesion_connections.twist_reference_a[dup_idx] = child_b_genome_orientation;
                    state.adhesion_connections.twist_reference_b[dup_idx] = neighbor_genome_orientation;
                    state.adhesion_connections.zone_a[dup_idx] =
                        classify_bond_direction(child_b_anchor, split_dir_local, child_b_split_ratio) as u8;
                    state.adhesion_connections.zone_b[dup_idx] =
                        classify_bond_direction(neighbor_anchor_to_b, split_dir_local, neighbor_split_ratio) as u8;
                } else {
                    state.adhesion_connections.cell_a_index[dup_idx] = neighbor_idx;
                    state.adhesion_connections.cell_b_index[dup_idx] = child_b_idx;
                    state.adhesion_connections.anchor_direction_a[dup_idx] = neighbor_anchor_to_b;
                    state.adhesion_connections.anchor_direction_b[dup_idx] = child_b_anchor;
                    state.adhesion_connections.twist_reference_a[dup_idx] = neighbor_genome_orientation;
                    state.adhesion_connections.twist_reference_b[dup_idx] = child_b_genome_orientation;
                    state.adhesion_connections.zone_a[dup_idx] =
                        classify_bond_direction(neighbor_anchor_to_b, split_dir_local, neighbor_split_ratio) as u8;
                    state.adhesion_connections.zone_b[dup_idx] =
                        classify_bond_direction(child_b_anchor, split_dir_local, child_b_split_ratio) as u8;
                }

                state.adhesion_connections.mode_index[dup_idx] = original_mode_index;
                state.adhesion_connections.is_active[dup_idx] = 1;
                state.adhesion_connections.birth_time[dup_idx] = current_time;

                if dup_idx >= state.adhesion_connections.active_count {
                    state.adhesion_connections.active_count = dup_idx + 1;
                }

                // Add to child B's adhesion indices
                if child_b_adhesion_count < MAX_ADHESIONS_PER_CELL {
                    state.adhesion_manager.cell_adhesion_indices[child_b_idx][child_b_adhesion_count] = dup_idx as i32;
                    child_b_adhesion_count += 1;
                }

                // Add duplicate to neighbor's adhesion indices (matches GPU lines 934-940)
                for slot in state.adhesion_manager.cell_adhesion_indices[neighbor_idx].iter_mut() {
                    if *slot < 0 {
                        *slot = dup_idx as i32;
                        break;
                    }
                }
            }
        }
    }
}

/// Calculate child anchor direction in child's local frame (matches GPU calculate_child_anchor_direction)
fn calculate_child_anchor_direction(
    child_pos_parent_frame: Vec3,
    neighbor_pos_parent_frame: Vec3,
    child_orientation_delta: Quat,
) -> Vec3 {
    let direction_to_neighbor = (neighbor_pos_parent_frame - child_pos_parent_frame).normalize_or_zero();
    let inv_delta = child_orientation_delta.inverse();
    (inv_delta * direction_to_neighbor).normalize()
}

/// Find a free connection slot (matches GPU allocate_adhesion_slot fallback)
fn find_free_connection_slot(connections: &crate::cell::adhesion::AdhesionConnections) -> Option<usize> {
    for i in 0..connections.cell_a_index.len() {
        if i >= connections.active_count || connections.is_active[i] == 0 {
            return Some(i);
        }
    }
    None
}
