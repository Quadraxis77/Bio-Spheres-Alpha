// Cell division logic

use glam::{Vec3, Quat, EulerRot};
use crate::simulation::canonical_state::{CanonicalState, DivisionEvent};
use crate::simulation::adhesion_inheritance::inherit_adhesions_on_division;
use crate::genome::Genome;

/// Deterministic pseudo-random rotation for cell division
/// Generates small rotation perturbations (0.001 radians) for visual variety
fn pseudo_random_rotation(cell_id: u32, rng_seed: u64) -> Quat {
    // Simple deterministic hash for pseudo-randomness
    let hash = ((cell_id as u64).wrapping_mul(2654435761).wrapping_add(rng_seed)) % 1000;
    let angle = (hash as f32 / 1000.0) * 0.001; // 0.001 radian max perturbation
    
    // Random axis
    let axis_hash = hash.wrapping_mul(7919) % 3;
    let axis = match axis_hash {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    };
    
    Quat::from_axis_angle(axis, angle)
}

/// Cell division step - processes all cells ready to divide
/// Returns vector of DivisionEvent describing which divisions occurred
pub fn division_step(
    state: &mut CanonicalState,
    genome: &Genome,
    current_time: f32,
    max_cells: usize,
    rng_seed: u64,
) -> Vec<DivisionEvent> {
    
    // Early exit if at capacity
    if state.cell_count >= max_cells {
        return Vec::new();
    }
    
    // Multi-pass split system: Instead of deferring splits to the next tick,
    // we perform multiple passes within this tick. Each pass handles splits
    // that don't conflict with each other, then we move to the next pass.
    
    // Use pre-allocated buffers to avoid per-frame allocations
    state.division_events_buffer.clear();
    // Clear only the portion of already_split we need (avoid touching entire capacity)
    for i in 0..state.cell_count {
        state.already_split_buffer[i] = false;
    }
    
    // Maximum number of passes to prevent infinite loops
    const MAX_PASSES: usize = 10;
    
    for _pass in 0..MAX_PASSES {
        // Find cells ready to divide in this pass
        state.divisions_to_process_buffer.clear();
        for i in 0..state.cell_count {
            // Skip cells that already split in a previous pass
            if state.already_split_buffer[i] {
                continue;
            }
            
            let cell_age = current_time - state.birth_times[i];
            let mode_index = state.mode_indices[i];
            let mode = genome.modes.get(mode_index);
            
            // Check max_splits limit (-1 means infinite)
            let can_split_by_count = if let Some(m) = mode {
                m.max_splits < 0 || state.split_counts[i] < m.max_splits
            } else {
                true
            };
            
            // Skip adhesion checks since we're not implementing adhesions
            let can_split_by_adhesions = true;
            
            // Check mass threshold - cells must have enough mass to split (using per-cell split_mass)
            let can_split_by_mass = state.masses[i] >= state.split_masses[i];
            
            // Check time threshold - cells must be old enough to split
            // Flagellocytes (cell_type == 1) split based on mass only, ignore split_interval
            let is_flagellocyte = mode.map(|m| m.cell_type == 1).unwrap_or(false);
            let can_split_by_time = is_flagellocyte || cell_age >= state.split_intervals[i];
            
            // Cell can split if ALL conditions are met
            // Note: split_intervals[i] <= 59.0 check only applies to non-Flagellocytes
            let split_interval_valid = is_flagellocyte || state.split_intervals[i] <= 59.0;
            if can_split_by_count && can_split_by_adhesions && can_split_by_mass && can_split_by_time && split_interval_valid {
                state.divisions_to_process_buffer.push(i);
            }
        }
        
        // If no cells ready to split, we're done
        if state.divisions_to_process_buffer.is_empty() {
            break;
        }
        
        // Since we're skipping adhesions, process all divisions without filtering
        state.filtered_divisions_buffer.clear();
        state.filtered_divisions_buffer.extend_from_slice(&state.divisions_to_process_buffer);
        
        // If no divisions can proceed in this pass, we're done
        if state.filtered_divisions_buffer.is_empty() {
            break;
        }

        // Collect division data before modifying state
        struct DivisionData {
            parent_idx: usize,
            parent_mode_idx: usize,
            child_a_slot: usize,
            child_b_slot: usize,
            parent_velocity: Vec3,
            parent_genome_id: usize,
            parent_stiffness: f32,
            #[allow(dead_code)]
            parent_split_count: i32,
            parent_genome_orientation: Quat,
            child_a_pos: Vec3,
            child_b_pos: Vec3,
            child_a_orientation: Quat,
            child_b_orientation: Quat,
            child_a_genome_orientation: Quat,
            child_b_genome_orientation: Quat,
            child_a_mode_idx: usize,
            child_b_mode_idx: usize,
            child_a_mass: f32,
            child_b_mass: f32,
            child_a_radius: f32,
            child_b_radius: f32,
            child_a_split_interval: f32,
            child_b_split_interval: f32,
            child_a_split_mass_threshold: f32,
            child_b_split_mass_threshold: f32,
            child_a_split_count: i32,
            child_b_split_count: i32,
            split_direction_local: Vec3,
        }
        
        let mut division_data_list = Vec::new();
        let mut pass_division_events = Vec::new();

        // Calculate available slots for children
        // Child A reuses parent index (matches reference), Child B gets new slot
        let mut next_available_slot = state.cell_count;

        // Process each division and collect data
        for i in 0..state.filtered_divisions_buffer.len() {
            let parent_idx = state.filtered_divisions_buffer[i];
            // Check if we have space for 1 more cell (child B)
            if next_available_slot >= state.capacity {
                break;
            }

            // Mark this cell as having split in this pass
            state.already_split_buffer[parent_idx] = true;

            // Child A reuses parent index, Child B gets new slot
            let child_a_slot = parent_idx;
            let child_b_slot = next_available_slot;
            next_available_slot += 1;
            
            let mode_index = state.mode_indices[parent_idx];
            let mode = genome.modes.get(mode_index);
            
            if let Some(mode) = mode {
                // Save parent properties
                let parent_position = state.positions[parent_idx];
                let parent_velocity = state.velocities[parent_idx];
                let parent_rotation = state.rotations[parent_idx];
                let parent_genome_orientation = state.genome_orientations[parent_idx];
                let parent_radius = state.radii[parent_idx];
                let parent_mass = state.masses[parent_idx];
                let parent_genome_id = state.genome_ids[parent_idx];
                let parent_stiffness = state.stiffnesses[parent_idx];
                let parent_split_count = state.split_counts[parent_idx];
                
                // Calculate split direction using physics rotation (for positioning)
                let pitch = mode.parent_split_direction.x.to_radians();
                let yaw = mode.parent_split_direction.y.to_radians();
                let split_direction = parent_rotation
                    * Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
                    * Vec3::Z;
                
                // 75% overlap means centers are 25% of combined diameter apart
                // Match reference convention: Child A at +offset, Child B at -offset
                let offset_distance = parent_radius * 0.25;
                let child_a_pos = parent_position + split_direction * offset_distance;
                let child_b_pos = parent_position - split_direction * offset_distance;
                
                // Get child mode indices
                // Check if children will reach max_splits after this division
                let will_reach_max_splits = mode.max_splits >= 0 && (parent_split_count + 1) >= mode.max_splits;
                
                // If max_splits is reached and mode_a_after_splits is set, use that mode for Child A
                let child_a_mode_idx = if will_reach_max_splits && mode.mode_a_after_splits >= 0 {
                    mode.mode_a_after_splits.max(0) as usize
                } else {
                    mode.child_a.mode_number.max(0) as usize
                };
                // If max_splits is reached and mode_b_after_splits is set, use that mode for Child B
                let child_b_mode_idx = if will_reach_max_splits && mode.mode_b_after_splits >= 0 {
                    mode.mode_b_after_splits.max(0) as usize
                } else {
                    mode.child_b.mode_number.max(0) as usize
                };
                
                // Determine split counts: reset to 0 if mode changes, otherwise inherit parent's count + 1
                let child_a_split_count = if child_a_mode_idx != mode_index {
                    0
                } else {
                    parent_split_count + 1
                };
                let child_b_split_count = if child_b_mode_idx != mode_index {
                    0
                } else {
                    parent_split_count + 1
                };
                
                // Get child properties
                let child_a_mode = genome.modes.get(child_a_mode_idx);
                let child_b_mode = genome.modes.get(child_b_mode_idx);
                
                // Split parent's mass according to split_ratio
                // split_ratio determines what fraction goes to Child A (0.0 to 1.0)
                let split_ratio = mode.split_ratio.clamp(0.0, 1.0);
                let child_a_mass = parent_mass * split_ratio;
                let child_b_mass = parent_mass * (1.0 - split_ratio);
                
                // Calculate child radii based on their masses
                let child_a_radius = if let Some(m) = child_a_mode {
                    child_a_mass.min(m.max_cell_size).clamp(0.5, 2.0)
                } else {
                    child_a_mass.clamp(0.5, 2.0)
                };
                
                let child_b_radius = if let Some(m) = child_b_mode {
                    child_b_mass.min(m.max_cell_size).clamp(0.5, 2.0)
                } else {
                    child_b_mass.clamp(0.5, 2.0)
                };
                
                // Get split intervals (potentially randomized from range)
                // Use parent cell_id + tick for deterministic randomness
                let parent_cell_id = state.cell_ids[parent_idx];
                let tick = (current_time * 60.0) as u64; // Approximate tick from time
                
                let child_a_split_interval = if let Some(m) = child_a_mode {
                    m.get_split_interval(parent_cell_id, tick, rng_seed)
                } else {
                    5.0
                };
                
                let child_b_split_interval = if let Some(m) = child_b_mode {
                    m.get_split_interval(parent_cell_id + 1, tick, rng_seed)
                } else {
                    5.0
                };
                
                // Get split mass thresholds (potentially randomized from range)
                let child_a_split_mass_threshold = if let Some(m) = child_a_mode {
                    m.get_split_mass(parent_cell_id, tick, rng_seed)
                } else {
                    1.5
                };
                
                let child_b_split_mass_threshold = if let Some(m) = child_b_mode {
                    m.get_split_mass(parent_cell_id + 1, tick, rng_seed)
                } else {
                    1.5
                };
                
                // Use parent's GENOME orientation for child genome orientations
                // This ensures genome orientations stay fixed and don't inherit physics rotation
                let child_a_genome_orientation = parent_genome_orientation * mode.child_a.orientation;
                let child_b_genome_orientation = parent_genome_orientation * mode.child_b.orientation;
                
                // Physics rotations inherit from parent's physics rotation + child orientation delta
                // This preserves the parent's spin while applying the genome-specified orientation change
                let child_a_orientation = parent_rotation * mode.child_a.orientation;
                let child_b_orientation = parent_rotation * mode.child_b.orientation;
                
                division_data_list.push(DivisionData {
                    parent_idx,
                    parent_mode_idx: mode_index,
                    child_a_slot,
                    child_b_slot,
                    parent_velocity,
                    parent_genome_id,
                    parent_stiffness,
                    parent_split_count,
                    parent_genome_orientation,
                    child_a_pos,
                    child_b_pos,
                    child_a_orientation,
                    child_b_orientation,
                    child_a_genome_orientation,
                    child_b_genome_orientation,
                    child_a_mode_idx,
                    child_b_mode_idx,
                    child_a_mass,
                    child_b_mass,
                    child_a_radius,
                    child_b_radius,
                    child_a_split_interval,
                    child_b_split_interval,
                    child_a_split_mass_threshold,
                    child_b_split_mass_threshold,
                    child_a_split_count,
                    child_b_split_count,
                    split_direction_local: Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z,
                });
            }
        }
        
        // Now write all the children to their allocated slots
        for data in &division_data_list {
            // Children are born at current_time
            let child_birth_time = current_time;

            if data.child_a_slot < state.capacity {
                // Write child A
                let child_a_id = state.next_cell_id;
                state.cell_ids[data.child_a_slot] = child_a_id;
                state.next_cell_id += 1;
                state.positions[data.child_a_slot] = data.child_a_pos;
                state.prev_positions[data.child_a_slot] = data.child_a_pos;
                state.velocities[data.child_a_slot] = data.parent_velocity;
                state.masses[data.child_a_slot] = data.child_a_mass;
                state.radii[data.child_a_slot] = data.child_a_radius;
                state.genome_ids[data.child_a_slot] = data.parent_genome_id;
                state.mode_indices[data.child_a_slot] = data.child_a_mode_idx;

                // Apply pseudo-random rotation perturbation (0.001 radians)
                let random_rotation_a = pseudo_random_rotation(child_a_id, rng_seed);
                state.rotations[data.child_a_slot] = data.child_a_orientation * random_rotation_a;

                state.genome_orientations[data.child_a_slot] = data.child_a_genome_orientation;
                state.angular_velocities[data.child_a_slot] = Vec3::ZERO;
                state.forces[data.child_a_slot] = Vec3::ZERO;
                state.torques[data.child_a_slot] = Vec3::ZERO;
                state.accelerations[data.child_a_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_a_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_a_slot] = data.parent_stiffness;
                state.birth_times[data.child_a_slot] = child_birth_time;
                state.split_intervals[data.child_a_slot] = data.child_a_split_interval;
                state.split_masses[data.child_a_slot] = data.child_a_split_mass_threshold;
                state.split_counts[data.child_a_slot] = data.child_a_split_count;
            }

            if data.child_b_slot < state.capacity {
                // Write child B
                let child_b_id = state.next_cell_id;
                state.cell_ids[data.child_b_slot] = child_b_id;
                state.next_cell_id += 1;
                state.positions[data.child_b_slot] = data.child_b_pos;
                state.prev_positions[data.child_b_slot] = data.child_b_pos;
                state.velocities[data.child_b_slot] = data.parent_velocity;
                state.masses[data.child_b_slot] = data.child_b_mass;
                state.radii[data.child_b_slot] = data.child_b_radius;
                state.genome_ids[data.child_b_slot] = data.parent_genome_id;
                state.mode_indices[data.child_b_slot] = data.child_b_mode_idx;

                // Apply pseudo-random rotation perturbation (0.001 radians)
                let random_rotation_b = pseudo_random_rotation(child_b_id, rng_seed);
                state.rotations[data.child_b_slot] = data.child_b_orientation * random_rotation_b;

                state.genome_orientations[data.child_b_slot] = data.child_b_genome_orientation;
                state.angular_velocities[data.child_b_slot] = Vec3::ZERO;
                state.forces[data.child_b_slot] = Vec3::ZERO;
                state.torques[data.child_b_slot] = Vec3::ZERO;
                state.accelerations[data.child_b_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_b_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_b_slot] = data.parent_stiffness;
                state.birth_times[data.child_b_slot] = child_birth_time;
                state.split_intervals[data.child_b_slot] = data.child_b_split_interval;
                state.split_masses[data.child_b_slot] = data.child_b_split_mass_threshold;
                state.split_counts[data.child_b_slot] = data.child_b_split_count;
            }
            
            // Record the division event
            pass_division_events.push(DivisionEvent {
                parent_idx: data.parent_idx,
                child_a_idx: data.child_a_slot,
                child_b_idx: data.child_b_slot,
            });
            
            // Handle adhesion inheritance from parent to children
            // This must happen AFTER children are written but BEFORE new adhesions are created
            inherit_adhesions_on_division(
                state,
                genome,
                data.parent_mode_idx,
                data.child_a_slot,
                data.child_b_slot,
                data.parent_genome_orientation,
            );
            
            // Create adhesion between children if parent_make_adhesion is enabled
            // keep_adhesion only affects inheritance from parent, NOT child-to-child adhesion
            let parent_mode = genome.modes.get(data.parent_mode_idx);
            
            if let Some(mode) = parent_mode {
                // Only parent_make_adhesion controls child-to-child adhesion creation
                if mode.parent_make_adhesion {
                        // CRITICAL: Use split direction from parent's GENOME orientation (not world positions!)
                        // This ensures anchors stay aligned with the genome's intended split direction
                        // even if physics has moved the cells slightly
                        
                        // CRITICAL: Match C++ implementation exactly
                        // Direction vectors in parent's local frame:
                        // Child A is at +offset, child B is at -offset
                        // Child A points toward B (at -offset): -splitDirLocal
                        // Child B points toward A (at +offset): +splitDirLocal
                        // Transform to each child's local space using genome-derived orientation deltas
                        let direction_a_to_b_parent_local = -data.split_direction_local;
                        let direction_b_to_a_parent_local = data.split_direction_local;
                        
                        let anchor_direction_a = (mode.child_a.orientation.inverse() * direction_a_to_b_parent_local).normalize();
                        let anchor_direction_b = (mode.child_b.orientation.inverse() * direction_b_to_a_parent_local).normalize();
                        
                        // Get split directions for zone classification
                        let child_a_mode = genome.modes.get(data.child_a_mode_idx);
                        let child_b_mode = genome.modes.get(data.child_b_mode_idx);
                        
                        let child_a_split_dir = if let Some(m) = child_a_mode {
                            let pitch = m.parent_split_direction.x.to_radians();
                            let yaw = m.parent_split_direction.y.to_radians();
                            Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z
                        } else {
                            Vec3::Z
                        };
                        
                        let child_b_split_dir = if let Some(m) = child_b_mode {
                            let pitch = m.parent_split_direction.x.to_radians();
                            let yaw = m.parent_split_direction.y.to_radians();
                            Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z
                        } else {
                            Vec3::Z
                        };
                        
                        // Create child-to-child connection with parent's mode index
                        let _ = state.adhesion_manager.add_adhesion_with_directions(
                            &mut state.adhesion_connections,
                            data.child_a_slot,
                            data.child_b_slot,
                            data.parent_mode_idx,  // Use parent's mode index (matches reference)
                            anchor_direction_a,
                            anchor_direction_b,
                            child_a_split_dir,
                            child_b_split_dir,
                            data.child_a_genome_orientation,
                            data.child_b_genome_orientation,
                        );
                }
            }
        }
        
        // Update cell count (child B cells are already written)
        let new_cell_count = state.cell_count + division_data_list.len();
        state.cell_count = new_cell_count;
        
        // Add this pass's events to the pre-allocated buffer
        state.division_events_buffer.extend(pass_division_events);
        
        // Check if we're at capacity - if so, stop processing passes
        if state.cell_count >= max_cells {
            break;
        }
    } // End of multi-pass loop
    
    // Return a clone of the events
    state.division_events_buffer.clone()
}


/// Cell division step with multiple genomes support
/// Each cell looks up its genome via genome_id
pub fn division_step_multi(
    state: &mut CanonicalState,
    genomes: &[Genome],
    current_time: f32,
    max_cells: usize,
    rng_seed: u64,
) -> Vec<DivisionEvent> {
    if genomes.is_empty() {
        return Vec::new();
    }
    
    // Early exit if at capacity
    if state.cell_count >= max_cells {
        return Vec::new();
    }
    
    // Pre-compute genome mode offsets for global mode index calculation
    let genome_mode_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(genomes.len());
        let mut offset = 0usize;
        for genome in genomes {
            offsets.push(offset);
            offset += genome.modes.len();
        }
        offsets
    };
    
    // Use pre-allocated buffers to avoid per-frame allocations
    state.division_events_buffer.clear();
    for i in 0..state.cell_count {
        state.already_split_buffer[i] = false;
    }
    
    const MAX_PASSES: usize = 10;
    
    for _pass in 0..MAX_PASSES {
        // Find cells ready to divide in this pass
        state.divisions_to_process_buffer.clear();
        for i in 0..state.cell_count {
            if state.already_split_buffer[i] {
                continue;
            }
            
            // Get the genome for this cell
            let genome_id = state.genome_ids[i];
            let genome = match genomes.get(genome_id) {
                Some(g) => g,
                None => continue, // Skip cells with invalid genome_id
            };
            
            let cell_age = current_time - state.birth_times[i];
            let mode_index = state.mode_indices[i];
            let mode = genome.modes.get(mode_index);
            
            let can_split_by_count = if let Some(m) = mode {
                m.max_splits < 0 || state.split_counts[i] < m.max_splits
            } else {
                true
            };
            
            let can_split_by_adhesions = true;
            let can_split_by_mass = state.masses[i] >= state.split_masses[i];
            
            // Flagellocytes (cell_type == 1) split based on mass only, ignore split_interval
            let is_flagellocyte = mode.map(|m| m.cell_type == 1).unwrap_or(false);
            let can_split_by_time = is_flagellocyte || cell_age >= state.split_intervals[i];
            
            // Note: split_intervals[i] <= 59.0 check only applies to non-Flagellocytes
            let split_interval_valid = is_flagellocyte || state.split_intervals[i] <= 59.0;
            if can_split_by_count && can_split_by_adhesions && can_split_by_mass && can_split_by_time && split_interval_valid {
                state.divisions_to_process_buffer.push(i);
            }
        }
        
        if state.divisions_to_process_buffer.is_empty() {
            break;
        }
        
        state.filtered_divisions_buffer.clear();
        state.filtered_divisions_buffer.extend_from_slice(&state.divisions_to_process_buffer);
        
        if state.filtered_divisions_buffer.is_empty() {
            break;
        }

        struct DivisionData {
            parent_idx: usize,
            parent_mode_idx: usize,
            parent_genome_id: usize,
            child_a_slot: usize,
            child_b_slot: usize,
            parent_velocity: Vec3,
            parent_stiffness: f32,
            #[allow(dead_code)]
            parent_split_count: i32,
            parent_genome_orientation: Quat,
            child_a_pos: Vec3,
            child_b_pos: Vec3,
            child_a_orientation: Quat,
            child_b_orientation: Quat,
            child_a_genome_orientation: Quat,
            child_b_genome_orientation: Quat,
            child_a_mode_idx: usize,
            child_b_mode_idx: usize,
            child_a_mass: f32,
            child_b_mass: f32,
            child_a_radius: f32,
            child_b_radius: f32,
            child_a_split_interval: f32,
            child_b_split_interval: f32,
            child_a_split_mass_threshold: f32,
            child_b_split_mass_threshold: f32,
            child_a_split_count: i32,
            child_b_split_count: i32,
            split_direction_local: Vec3,
        }
        
        let mut division_data_list = Vec::new();
        let mut pass_division_events = Vec::new();
        let mut next_available_slot = state.cell_count;

        for i in 0..state.filtered_divisions_buffer.len() {
            let parent_idx = state.filtered_divisions_buffer[i];
            if next_available_slot >= state.capacity {
                break;
            }

            state.already_split_buffer[parent_idx] = true;

            let child_a_slot = parent_idx;
            let child_b_slot = next_available_slot;
            next_available_slot += 1;
            
            let genome_id = state.genome_ids[parent_idx];
            let genome = match genomes.get(genome_id) {
                Some(g) => g,
                None => continue,
            };
            
            let mode_index = state.mode_indices[parent_idx];
            let mode = match genome.modes.get(mode_index) {
                Some(m) => m,
                None => continue,
            };
            
            let parent_position = state.positions[parent_idx];
            let parent_velocity = state.velocities[parent_idx];
            let parent_rotation = state.rotations[parent_idx];
            let parent_genome_orientation = state.genome_orientations[parent_idx];
            let parent_radius = state.radii[parent_idx];
            let parent_mass = state.masses[parent_idx];
            let parent_stiffness = state.stiffnesses[parent_idx];
            let parent_split_count = state.split_counts[parent_idx];
            
            let pitch = mode.parent_split_direction.x.to_radians();
            let yaw = mode.parent_split_direction.y.to_radians();
            let split_direction = parent_rotation
                * Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
                * Vec3::Z;
            
            let offset_distance = parent_radius * 0.25;
            let child_a_pos = parent_position + split_direction * offset_distance;
            let child_b_pos = parent_position - split_direction * offset_distance;
            
            let will_reach_max_splits = mode.max_splits >= 0 && (parent_split_count + 1) >= mode.max_splits;
            
            let child_a_mode_idx = if will_reach_max_splits && mode.mode_a_after_splits >= 0 {
                mode.mode_a_after_splits.max(0) as usize
            } else {
                mode.child_a.mode_number.max(0) as usize
            };
            let child_b_mode_idx = if will_reach_max_splits && mode.mode_b_after_splits >= 0 {
                mode.mode_b_after_splits.max(0) as usize
            } else {
                mode.child_b.mode_number.max(0) as usize
            };
            
            let child_a_split_count = if child_a_mode_idx != mode_index { 0 } else { parent_split_count + 1 };
            let child_b_split_count = if child_b_mode_idx != mode_index { 0 } else { parent_split_count + 1 };
            
            let child_a_mode = genome.modes.get(child_a_mode_idx);
            let child_b_mode = genome.modes.get(child_b_mode_idx);
            
            let split_ratio = mode.split_ratio.clamp(0.0, 1.0);
            let child_a_mass = parent_mass * split_ratio;
            let child_b_mass = parent_mass * (1.0 - split_ratio);
            
            let child_a_radius = if let Some(m) = child_a_mode {
                child_a_mass.min(m.max_cell_size).clamp(0.5, 2.0)
            } else {
                child_a_mass.clamp(0.5, 2.0)
            };
            
            let child_b_radius = if let Some(m) = child_b_mode {
                child_b_mass.min(m.max_cell_size).clamp(0.5, 2.0)
            } else {
                child_b_mass.clamp(0.5, 2.0)
            };
            
            let parent_cell_id = state.cell_ids[parent_idx];
            let tick = (current_time * 60.0) as u64;
            
            let child_a_split_interval = if let Some(m) = child_a_mode {
                m.get_split_interval(parent_cell_id, tick, rng_seed)
            } else { 5.0 };
            
            let child_b_split_interval = if let Some(m) = child_b_mode {
                m.get_split_interval(parent_cell_id + 1, tick, rng_seed)
            } else { 5.0 };
            
            let child_a_split_mass_threshold = if let Some(m) = child_a_mode {
                m.get_split_mass(parent_cell_id, tick, rng_seed)
            } else { 1.5 };
            
            let child_b_split_mass_threshold = if let Some(m) = child_b_mode {
                m.get_split_mass(parent_cell_id + 1, tick, rng_seed)
            } else { 1.5 };
            
            let child_a_genome_orientation = parent_genome_orientation * mode.child_a.orientation;
            let child_b_genome_orientation = parent_genome_orientation * mode.child_b.orientation;
            let child_a_orientation = parent_rotation * mode.child_a.orientation;
            let child_b_orientation = parent_rotation * mode.child_b.orientation;
            
            division_data_list.push(DivisionData {
                parent_idx,
                parent_mode_idx: mode_index,
                parent_genome_id: genome_id,
                child_a_slot,
                child_b_slot,
                parent_velocity,
                parent_stiffness,
                parent_split_count,
                parent_genome_orientation,
                child_a_pos,
                child_b_pos,
                child_a_orientation,
                child_b_orientation,
                child_a_genome_orientation,
                child_b_genome_orientation,
                child_a_mode_idx,
                child_b_mode_idx,
                child_a_mass,
                child_b_mass,
                child_a_radius,
                child_b_radius,
                child_a_split_interval,
                child_b_split_interval,
                child_a_split_mass_threshold,
                child_b_split_mass_threshold,
                child_a_split_count,
                child_b_split_count,
                split_direction_local: Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z,
            });
        }
        
        // Write children to slots
        for data in &division_data_list {
            let child_birth_time = current_time;
            let genome = &genomes[data.parent_genome_id];

            if data.child_a_slot < state.capacity {
                let child_a_id = state.next_cell_id;
                state.cell_ids[data.child_a_slot] = child_a_id;
                state.next_cell_id += 1;
                state.positions[data.child_a_slot] = data.child_a_pos;
                state.prev_positions[data.child_a_slot] = data.child_a_pos;
                state.velocities[data.child_a_slot] = data.parent_velocity;
                state.masses[data.child_a_slot] = data.child_a_mass;
                state.radii[data.child_a_slot] = data.child_a_radius;
                state.genome_ids[data.child_a_slot] = data.parent_genome_id;
                state.mode_indices[data.child_a_slot] = data.child_a_mode_idx;
                let random_rotation_a = pseudo_random_rotation(child_a_id, rng_seed);
                state.rotations[data.child_a_slot] = data.child_a_orientation * random_rotation_a;
                state.genome_orientations[data.child_a_slot] = data.child_a_genome_orientation;
                state.angular_velocities[data.child_a_slot] = Vec3::ZERO;
                state.forces[data.child_a_slot] = Vec3::ZERO;
                state.torques[data.child_a_slot] = Vec3::ZERO;
                state.accelerations[data.child_a_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_a_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_a_slot] = data.parent_stiffness;
                state.birth_times[data.child_a_slot] = child_birth_time;
                state.split_intervals[data.child_a_slot] = data.child_a_split_interval;
                state.split_masses[data.child_a_slot] = data.child_a_split_mass_threshold;
                state.split_counts[data.child_a_slot] = data.child_a_split_count;
            }

            if data.child_b_slot < state.capacity {
                let child_b_id = state.next_cell_id;
                state.cell_ids[data.child_b_slot] = child_b_id;
                state.next_cell_id += 1;
                state.positions[data.child_b_slot] = data.child_b_pos;
                state.prev_positions[data.child_b_slot] = data.child_b_pos;
                state.velocities[data.child_b_slot] = data.parent_velocity;
                state.masses[data.child_b_slot] = data.child_b_mass;
                state.radii[data.child_b_slot] = data.child_b_radius;
                state.genome_ids[data.child_b_slot] = data.parent_genome_id;
                state.mode_indices[data.child_b_slot] = data.child_b_mode_idx;
                let random_rotation_b = pseudo_random_rotation(child_b_id, rng_seed);
                state.rotations[data.child_b_slot] = data.child_b_orientation * random_rotation_b;
                state.genome_orientations[data.child_b_slot] = data.child_b_genome_orientation;
                state.angular_velocities[data.child_b_slot] = Vec3::ZERO;
                state.forces[data.child_b_slot] = Vec3::ZERO;
                state.torques[data.child_b_slot] = Vec3::ZERO;
                state.accelerations[data.child_b_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_b_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_b_slot] = data.parent_stiffness;
                state.birth_times[data.child_b_slot] = child_birth_time;
                state.split_intervals[data.child_b_slot] = data.child_b_split_interval;
                state.split_masses[data.child_b_slot] = data.child_b_split_mass_threshold;
                state.split_counts[data.child_b_slot] = data.child_b_split_count;
            }
            
            pass_division_events.push(DivisionEvent {
                parent_idx: data.parent_idx,
                child_a_idx: data.child_a_slot,
                child_b_idx: data.child_b_slot,
            });
            
            // Handle adhesion inheritance
            inherit_adhesions_on_division(
                state,
                genome,
                data.parent_mode_idx,
                data.child_a_slot,
                data.child_b_slot,
                data.parent_genome_orientation,
            );
            
            // Create adhesion between children if parent_make_adhesion is enabled
            let parent_mode = genome.modes.get(data.parent_mode_idx);
            
            if let Some(mode) = parent_mode {
                if mode.parent_make_adhesion {
                    let direction_a_to_b_parent_local = -data.split_direction_local;
                    let direction_b_to_a_parent_local = data.split_direction_local;
                    
                    let anchor_direction_a = (mode.child_a.orientation.inverse() * direction_a_to_b_parent_local).normalize();
                    let anchor_direction_b = (mode.child_b.orientation.inverse() * direction_b_to_a_parent_local).normalize();
                    
                    let child_a_mode = genome.modes.get(data.child_a_mode_idx);
                    let child_b_mode = genome.modes.get(data.child_b_mode_idx);
                    
                    let child_a_split_dir = if let Some(m) = child_a_mode {
                        let pitch = m.parent_split_direction.x.to_radians();
                        let yaw = m.parent_split_direction.y.to_radians();
                        Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z
                    } else { Vec3::Z };
                    
                    let child_b_split_dir = if let Some(m) = child_b_mode {
                        let pitch = m.parent_split_direction.x.to_radians();
                        let yaw = m.parent_split_direction.y.to_radians();
                        Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z
                    } else { Vec3::Z };
                    
                    // Calculate global mode index for adhesion settings lookup
                    let global_mode_idx = genome_mode_offsets.get(data.parent_genome_id).copied().unwrap_or(0) + data.parent_mode_idx;
                    
                    let _ = state.adhesion_manager.add_adhesion_with_directions(
                        &mut state.adhesion_connections,
                        data.child_a_slot,
                        data.child_b_slot,
                        global_mode_idx,  // Use global mode index for multi-genome support
                        anchor_direction_a,
                        anchor_direction_b,
                        child_a_split_dir,
                        child_b_split_dir,
                        data.child_a_genome_orientation,
                        data.child_b_genome_orientation,
                    );
                }
            }
        }
        
        let new_cell_count = state.cell_count + division_data_list.len();
        state.cell_count = new_cell_count;
        state.division_events_buffer.extend(pass_division_events);
        
        if state.cell_count >= max_cells {
            break;
        }
    }
    
    state.division_events_buffer.clone()
}
