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
    _parent_radius: f32,
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

    // Geometric parameters from genome — only needed for zone classification
    let pitch = parent_mode.parent_split_direction.x.to_radians();
    let yaw = parent_mode.parent_split_direction.y.to_radians();
    let split_rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    let split_dir_local = split_rotation * Vec3::Z;

    // Child genome orientations (already written to state by division.rs before this is called)
    let child_a_genome_orientation = state.genome_orientations[child_a_idx];
    let child_b_genome_orientation = state.genome_orientations[child_b_idx];

    // Transforms to re-express a parent-local anchor in each child's local space.
    // parent_anchor_world = parent_genome_orientation * parent_anchor_local
    // child_anchor_local  = child_genome_orientation.inverse() * parent_anchor_world
    let parent_to_child_a = child_a_genome_orientation.inverse() * parent_genome_orientation;
    let parent_to_child_b = child_b_genome_orientation.inverse() * parent_genome_orientation;

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

        // Preserve the neighbor's existing anchor direction — the neighbor hasn't moved
        // or split, so its anchor is still correct. Only the dividing cell's side needs
        // to be re-expressed in the child's local coordinate frame.
        let existing_neighbor_anchor = if parent_is_a {
            state.adhesion_connections.anchor_direction_b[connection_idx]
        } else {
            state.adhesion_connections.anchor_direction_a[connection_idx]
        };

        if give_to_a && !give_to_b {
            // Re-express parent's anchor in child A's local space (exact, no estimation)
            let child_anchor = (parent_to_child_a * parent_anchor_dir).normalize();

            if parent_is_a {
                state.adhesion_connections.anchor_direction_a[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_b[connection_idx] = existing_neighbor_anchor;
                state.adhesion_connections.twist_reference_a[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_a[connection_idx] =
                    classify_bond_direction(child_anchor, split_dir_local, child_a_split_ratio) as u8;
            } else {
                state.adhesion_connections.anchor_direction_b[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_a[connection_idx] = existing_neighbor_anchor;
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
            // Re-express parent's anchor in child B's local space (exact, no estimation)
            let child_anchor = (parent_to_child_b * parent_anchor_dir).normalize();

            if parent_is_a {
                state.adhesion_connections.cell_a_index[connection_idx] = child_b_idx;
                state.adhesion_connections.anchor_direction_a[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_b[connection_idx] = existing_neighbor_anchor;
                state.adhesion_connections.twist_reference_a[connection_idx] = child_b_genome_orientation;
                state.adhesion_connections.zone_a[connection_idx] =
                    classify_bond_direction(child_anchor, split_dir_local, child_b_split_ratio) as u8;
            } else {
                state.adhesion_connections.cell_b_index[connection_idx] = child_b_idx;
                state.adhesion_connections.anchor_direction_b[connection_idx] = child_anchor;
                state.adhesion_connections.anchor_direction_a[connection_idx] = existing_neighbor_anchor;
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

            // Re-express parent's anchor in each child's local space (exact)
            let child_a_anchor = (parent_to_child_a * parent_anchor_dir).normalize();
            let child_b_anchor = (parent_to_child_b * parent_anchor_dir).normalize();

            // --- Child A gets the original connection (modified in place) ---
            if parent_is_a {
                state.adhesion_connections.anchor_direction_a[connection_idx] = child_a_anchor;
                state.adhesion_connections.anchor_direction_b[connection_idx] = existing_neighbor_anchor;
                state.adhesion_connections.twist_reference_a[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_a[connection_idx] =
                    classify_bond_direction(child_a_anchor, split_dir_local, child_a_split_ratio) as u8;
            } else {
                state.adhesion_connections.anchor_direction_b[connection_idx] = child_a_anchor;
                state.adhesion_connections.anchor_direction_a[connection_idx] = existing_neighbor_anchor;
                state.adhesion_connections.twist_reference_b[connection_idx] = child_a_genome_orientation;
                state.adhesion_connections.zone_b[connection_idx] =
                    classify_bond_direction(child_a_anchor, split_dir_local, child_a_split_ratio) as u8;
            }

            if child_a_adhesion_count < MAX_ADHESIONS_PER_CELL {
                state.adhesion_manager.cell_adhesion_indices[child_a_idx][child_a_adhesion_count] = connection_idx as i32;
                child_a_adhesion_count += 1;
            }

            // --- Child B gets a duplicate connection ---
            let dup_idx = find_free_connection_slot(&state.adhesion_connections);
            if let Some(dup_idx) = dup_idx {
                let original_mode_index = state.adhesion_connections.mode_index[connection_idx];
                let neighbor_mode_idx = state.mode_indices[neighbor_idx];
                let neighbor_split_ratio = genome.modes.get(neighbor_mode_idx)
                    .map(|m| m.split_ratio).unwrap_or(0.5);

                if parent_is_a {
                    state.adhesion_connections.cell_a_index[dup_idx] = child_b_idx;
                    state.adhesion_connections.cell_b_index[dup_idx] = neighbor_idx;
                    state.adhesion_connections.anchor_direction_a[dup_idx] = child_b_anchor;
                    state.adhesion_connections.anchor_direction_b[dup_idx] = existing_neighbor_anchor;
                    state.adhesion_connections.twist_reference_a[dup_idx] = child_b_genome_orientation;
                    state.adhesion_connections.twist_reference_b[dup_idx] = state.genome_orientations[neighbor_idx];
                    state.adhesion_connections.zone_a[dup_idx] =
                        classify_bond_direction(child_b_anchor, split_dir_local, child_b_split_ratio) as u8;
                    state.adhesion_connections.zone_b[dup_idx] =
                        classify_bond_direction(existing_neighbor_anchor, split_dir_local, neighbor_split_ratio) as u8;
                } else {
                    state.adhesion_connections.cell_a_index[dup_idx] = neighbor_idx;
                    state.adhesion_connections.cell_b_index[dup_idx] = child_b_idx;
                    state.adhesion_connections.anchor_direction_a[dup_idx] = existing_neighbor_anchor;
                    state.adhesion_connections.anchor_direction_b[dup_idx] = child_b_anchor;
                    state.adhesion_connections.twist_reference_a[dup_idx] = state.genome_orientations[neighbor_idx];
                    state.adhesion_connections.twist_reference_b[dup_idx] = child_b_genome_orientation;
                    state.adhesion_connections.zone_a[dup_idx] =
                        classify_bond_direction(existing_neighbor_anchor, split_dir_local, neighbor_split_ratio) as u8;
                    state.adhesion_connections.zone_b[dup_idx] =
                        classify_bond_direction(child_b_anchor, split_dir_local, child_b_split_ratio) as u8;
                }

                state.adhesion_connections.mode_index[dup_idx] = original_mode_index;
                state.adhesion_connections.is_active[dup_idx] = 1;
                state.adhesion_connections.birth_time[dup_idx] = current_time;

                if dup_idx >= state.adhesion_connections.active_count {
                    state.adhesion_connections.active_count = dup_idx + 1;
                }

                if child_b_adhesion_count < MAX_ADHESIONS_PER_CELL {
                    state.adhesion_manager.cell_adhesion_indices[child_b_idx][child_b_adhesion_count] = dup_idx as i32;
                    child_b_adhesion_count += 1;
                }

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

/// Find a free connection slot (matches GPU allocate_adhesion_slot fallback)
fn find_free_connection_slot(connections: &crate::cell::adhesion::AdhesionConnections) -> Option<usize> {
    for i in 0..connections.cell_a_index.len() {
        if i >= connections.active_count || connections.is_active[i] == 0 {
            return Some(i);
        }
    }
    None
}
