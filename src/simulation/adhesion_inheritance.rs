use glam::{Vec3, Quat, EulerRot};
use crate::cell::{AdhesionZone, classify_bond_direction, MAX_ADHESIONS_PER_CELL};
use crate::simulation::canonical_state::CanonicalState;
use crate::genome::Genome;

/// Handle adhesion inheritance during cell division
/// 
/// This function processes all adhesions from the parent cell and determines
/// which child(ren) should inherit each connection based on zone classification.
/// 
/// Zone inheritance rules:
/// - Zone A: Inherit to child B (adhesions pointing opposite to split direction)
/// - Zone B: Inherit to child A (adhesions pointing same as split direction)
/// - Zone C: Inherit to both children (adhesions in equatorial band)
/// 
/// CRITICAL: parent_genome_orientation must be the parent's orientation BEFORE division,
/// since child A overwrites the parent's slot and changes the genome orientation.
pub fn inherit_adhesions_on_division(
    state: &mut CanonicalState,
    genome: &Genome,
    parent_mode_idx: usize,
    child_a_idx: usize,
    child_b_idx: usize,
    parent_genome_orientation: Quat,
) {
    // Get parent mode settings
    let parent_mode = match genome.modes.get(parent_mode_idx) {
        Some(mode) => mode,
        None => return, // Invalid mode
    };
    
    // Check if children keep adhesions
    let child_a_keep = parent_mode.child_a.keep_adhesion;
    let child_b_keep = parent_mode.child_b.keep_adhesion;
    
    log::debug!(
        "inherit_adhesions_on_division: parent_mode={}, child_a_idx={}, child_b_idx={}, child_a_keep={}, child_b_keep={}",
        parent_mode_idx, child_a_idx, child_b_idx, child_a_keep, child_b_keep
    );
    
    // CRITICAL: Collect parent's adhesion connections BEFORE initializing child indices
    // (since child A reuses parent index, initializing would clear the connections)
    let mut parent_connections = Vec::new();
    for slot_idx in 0..MAX_ADHESIONS_PER_CELL {
        let connection_idx = state.adhesion_manager.cell_adhesion_indices[child_a_idx][slot_idx];
        if connection_idx >= 0 {
            parent_connections.push(connection_idx as usize);
        }
    }
    
    // Initialize adhesion indices for child cells (matches C++ Requirement 10.4)
    // This clears the parent's old adhesion indices
    // MUST happen AFTER collecting parent connections
    state.adhesion_manager.init_cell_adhesion_indices(child_a_idx);
    state.adhesion_manager.init_cell_adhesion_indices(child_b_idx);
    
    // Get parent properties
    let parent_radius = state.radii[child_a_idx];
    
    // Calculate split direction from parent mode (in local space)
    let pitch = parent_mode.parent_split_direction.x.to_radians();
    let yaw = parent_mode.parent_split_direction.y.to_radians();
    let split_direction_local = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z;
    
    // Extract split direction and offset for geometric calculations (matching C++)
    let split_magnitude = split_direction_local.length();
    let split_dir_parent = if split_magnitude < 0.0001 {
        Vec3::Z
    } else {
        split_direction_local / split_magnitude
    };
    let split_offset_magnitude = if split_magnitude < 0.0001 {
        0.0
    } else {
        split_magnitude * 0.5
    };
    
    // Process each parent connection
    for &connection_idx in &parent_connections {
        if connection_idx >= state.adhesion_connections.active_count {
            continue;
        }
        
        if state.adhesion_connections.is_active[connection_idx] == 0 {
            continue;
        }
        
        let cell_a_idx = state.adhesion_connections.cell_a_index[connection_idx];
        let cell_b_idx_conn = state.adhesion_connections.cell_b_index[connection_idx];
        
        let (neighbor_idx, parent_is_a) = if cell_a_idx == child_a_idx {
            (cell_b_idx_conn, true)
        } else if cell_b_idx_conn == child_a_idx {
            (cell_a_idx, false)
        } else {
            continue;
        };
        
        let (parent_anchor_direction, neighbor_anchor_direction) = if parent_is_a {
            (
                state.adhesion_connections.anchor_direction_a[connection_idx],
                state.adhesion_connections.anchor_direction_b[connection_idx],
            )
        } else {
            (
                state.adhesion_connections.anchor_direction_b[connection_idx],
                state.adhesion_connections.anchor_direction_a[connection_idx],
            )
        };
        
        let zone = classify_bond_direction(parent_anchor_direction, split_direction_local);
        
        log::debug!(
            "  Connection {}: neighbor={}, zone={:?}, parent_anchor={:?}, split_dir={:?}",
            connection_idx, neighbor_idx, zone, parent_anchor_direction, split_direction_local
        );
        
        match zone {
            AdhesionZone::ZoneA if child_b_keep => {
                log::debug!("    -> Inheriting to child B (idx {})", child_b_idx);
                create_inherited_adhesion(
                    state,
                    genome,
                    child_b_idx,
                    neighbor_idx,
                    parent_mode_idx,
                    parent_is_a,
                    parent_genome_orientation,
                    parent_anchor_direction,
                    neighbor_anchor_direction,
                    parent_radius,
                    state.radii[neighbor_idx],
                    parent_mode.child_b.orientation,
                    split_offset_magnitude,
                    split_dir_parent,
                    false,
                );
            }
            AdhesionZone::ZoneB if child_a_keep => {
                log::debug!("    -> Inheriting to child A (idx {})", child_a_idx);
                create_inherited_adhesion(
                    state,
                    genome,
                    child_a_idx,
                    neighbor_idx,
                    parent_mode_idx,
                    parent_is_a,
                    parent_genome_orientation,
                    parent_anchor_direction,
                    neighbor_anchor_direction,
                    parent_radius,
                    state.radii[neighbor_idx],
                    parent_mode.child_a.orientation,
                    split_offset_magnitude,
                    split_dir_parent,
                    true,
                );
            }
            AdhesionZone::ZoneC => {
                log::debug!("    -> Zone C: child_a_keep={}, child_b_keep={}", child_a_keep, child_b_keep);
                if child_b_keep {
                    log::debug!("      -> Inheriting to child B (idx {})", child_b_idx);
                    create_inherited_adhesion(
                        state,
                        genome,
                        child_b_idx,
                        neighbor_idx,
                        parent_mode_idx,
                        parent_is_a,
                        parent_genome_orientation,
                        parent_anchor_direction,
                        neighbor_anchor_direction,
                        parent_radius,
                        state.radii[neighbor_idx],
                        parent_mode.child_b.orientation,
                        split_offset_magnitude,
                        split_dir_parent,
                        false,
                    );
                }
                if child_a_keep {
                    log::debug!("      -> Inheriting to child A (idx {})", child_a_idx);
                    create_inherited_adhesion(
                        state,
                        genome,
                        child_a_idx,
                        neighbor_idx,
                        parent_mode_idx,
                        parent_is_a,
                        parent_genome_orientation,
                        parent_anchor_direction,
                        neighbor_anchor_direction,
                        parent_radius,
                        state.radii[neighbor_idx],
                        parent_mode.child_a.orientation,
                        split_offset_magnitude,
                        split_dir_parent,
                        true,
                    );
                }
            }
            AdhesionZone::ZoneA => {
                log::debug!("    -> Zone A but child_b_keep=false, skipping inheritance");
            }
            AdhesionZone::ZoneB => {
                log::debug!("    -> Zone B but child_a_keep=false, skipping inheritance");
            }
        }
        
        state.adhesion_connections.is_active[connection_idx] = 0;
    }
}

/// Create an inherited adhesion connection from parent to child
#[allow(clippy::too_many_arguments)]
fn create_inherited_adhesion(
    state: &mut CanonicalState,
    genome: &Genome,
    child_idx: usize,
    neighbor_idx: usize,
    _mode_index: usize,
    parent_was_a: bool,
    parent_genome_orientation: Quat,
    parent_anchor_direction: Vec3,
    _neighbor_anchor_direction: Vec3,
    _parent_radius: f32,
    _neighbor_radius: f32,
    child_orientation_delta: Quat,
    split_offset_magnitude: f32,
    split_dir_parent: Vec3,
    is_child_a: bool,
) {
    // Get child mode to use its adhesion settings (not parent's)
    let child_mode_idx = state.mode_indices[child_idx];
    let child_mode = match genome.modes.get(child_mode_idx) {
        Some(mode) => mode,
        None => return, // Invalid mode
    };
    
    // Get rest offset from child's mode (not parent's)
    // HARDCODED RADIUS: Use fixed radius value (1.0) to ensure adhesion is completely independent of cell growth
    const FIXED_RADIUS: f32 = 1.0;
    let sum_radii = FIXED_RADIUS + FIXED_RADIUS;
    let center_to_center_dist = (sum_radii * (1.0 + child_mode.adhesion_settings.adhesin_length)).max(sum_radii * 0.1);
    
    // Calculate positions in parent frame for geometric anchor placement (MATCHES C++)
    let child_pos_parent_frame = if is_child_a {
        split_dir_parent * split_offset_magnitude  // Child A at +offset
    } else {
        -split_dir_parent * split_offset_magnitude  // Child B at -offset
    };
    let neighbor_pos_parent_frame = parent_anchor_direction * center_to_center_dist;
    
    // Child anchor: direction from child to neighbor, transformed by genome orientation
    let direction_to_neighbor_parent_frame = (neighbor_pos_parent_frame - child_pos_parent_frame).normalize();
    let child_anchor_direction = (child_orientation_delta.inverse() * direction_to_neighbor_parent_frame).normalize();
    
    // Neighbor anchor: direction from neighbor to child, transformed to neighbor's frame
    let direction_to_child_parent_frame = (child_pos_parent_frame - neighbor_pos_parent_frame).normalize();
    let neighbor_genome_orientation = state.genome_orientations[neighbor_idx];
    let relative_rotation = neighbor_genome_orientation.inverse() * parent_genome_orientation;
    let neighbor_anchor_direction = (relative_rotation * direction_to_child_parent_frame).normalize();
    
    // Get child and neighbor mode indices for zone classification
    let child_mode_idx = state.mode_indices[child_idx];
    let neighbor_mode_idx = state.mode_indices[neighbor_idx];
    
    // Get split directions from each cell's mode
    let child_mode = genome.modes.get(child_mode_idx);
    let neighbor_mode = genome.modes.get(neighbor_mode_idx);
    
    let child_split_dir = if let Some(mode) = child_mode {
        let pitch = mode.parent_split_direction.x.to_radians();
        let yaw = mode.parent_split_direction.y.to_radians();
        Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z
    } else {
        Vec3::Z
    };
    
    let neighbor_split_dir = if let Some(mode) = neighbor_mode {
        let pitch = mode.parent_split_direction.x.to_radians();
        let yaw = mode.parent_split_direction.y.to_radians();
        Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z
    } else {
        Vec3::Z
    };
    
    // Get genome orientations for twist references
    let child_genome_orientation = state.genome_orientations[child_idx];
    let neighbor_genome_orientation_for_twist = state.genome_orientations[neighbor_idx];
    
    // Preserve original side assignment: if neighbor was originally cellA, keep them as cellA
    // Use child's mode index for the new adhesion (not parent's)
    let result = if parent_was_a {
        // Parent was cellA, neighbor was cellB, so neighbor becomes cellB
        state.adhesion_manager.add_adhesion_with_directions(
            &mut state.adhesion_connections,
            child_idx,
            neighbor_idx,
            child_mode_idx,
            child_anchor_direction,
            neighbor_anchor_direction,
            child_split_dir,
            neighbor_split_dir,
            child_genome_orientation,
            neighbor_genome_orientation_for_twist,
        )
    } else {
        // Parent was cellB, neighbor was cellA, so neighbor becomes cellA
        state.adhesion_manager.add_adhesion_with_directions(
            &mut state.adhesion_connections,
            neighbor_idx,
            child_idx,
            child_mode_idx,
            neighbor_anchor_direction,
            child_anchor_direction,
            neighbor_split_dir,
            child_split_dir,
            neighbor_genome_orientation_for_twist,
            child_genome_orientation,
        )
    };
    
    let _ = result; // Suppress unused warning
}
