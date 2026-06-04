// Cell division logic

use crate::genome::Genome;
use crate::simulation::adhesion_inheritance::inherit_adhesions_on_division;
use crate::simulation::canonical_state::{CanonicalState, CellDevelopmentAddress, DivisionEvent};
use glam::{EulerRot, Quat, Vec3};

/// Deterministic pseudo-random rotation for cell division
/// Generates small rotation perturbations for visual variety
/// Uses u32 arithmetic to match the GPU shader implementation exactly
fn pseudo_random_rotation(cell_id: u32, rng_seed: u64) -> Quat {
    let seed = rng_seed as u32;
    let hash1 = cell_id.wrapping_mul(2654435761u32).wrapping_add(seed) % 1000000;
    let hash2 = cell_id
        .wrapping_mul(1597334677u32)
        .wrapping_add(seed.wrapping_mul(3))
        % 1000000;

    // Generate angle in range [0.001, 0.1] radians for more visible variety
    let angle = (hash1 as f32 / 1000000.0) * 0.099 + 0.001;

    // Generate random axis using both hashes for more variety
    let x = (hash2 as f32 / 1000000.0) * 2.0 - 1.0; // [-1, 1]
    let y = ((hash2.wrapping_mul(7) % 1000000) as f32 / 1000000.0) * 2.0 - 1.0; // [-1, 1]
    let z = ((hash2.wrapping_mul(13) % 1000000) as f32 / 1000000.0) * 2.0 - 1.0; // [-1, 1]

    let axis = Vec3::new(x, y, z).normalize_or_zero();

    // Fallback to a default axis if normalization fails (very unlikely)
    let final_axis = if axis.length_squared() > 0.001 {
        axis
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };

    Quat::from_axis_angle(final_axis, angle)
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

            // Check min/max adhesions against active connection count
            let can_split_by_adhesions = if let Some(m) = mode {
                let active = if m.min_adhesions > 0 || m.max_adhesions > 0 {
                    state
                        .adhesion_manager
                        .count_active_adhesions(i, &state.adhesion_connections)
                } else {
                    0
                };
                let min_ok = m.min_adhesions <= 0 || active >= m.min_adhesions as usize;
                let max_ok = m.max_adhesions <= 0 || active < m.max_adhesions as usize;
                min_ok && max_ok
            } else {
                true
            };

            // Check signal-conditional division gating
            let can_split_by_signal = if let Some(m) = mode {
                if m.division_signal_channel >= 8 && (m.division_signal_channel as usize) <= 15 {
                    let ch = m.division_signal_channel as usize;
                    let signal_val = state.signal_channels[i * 16 + ch].unwrap_or(0.0);
                    let above = signal_val >= m.division_signal_threshold;
                    if m.division_signal_invert {
                        !above
                    } else {
                        above
                    }
                } else {
                    true // disabled = no gating
                }
            } else {
                true
            };

            // Gametocytes never split. They detach by release triggers and reproduce
            // only by merging with a compatible Gametocyte.
            if mode.map(|m| m.cell_type == 13).unwrap_or(false) {
                continue;
            }

            // Embryocytes (cell_type == 10) use a timer-based split condition instead of
            // nutrients. They divide only when they have NO active connections AND their
            // age has reached split_interval. This models the embryocyte "hatching" when
            // it detaches from the organism.
            let is_embryocyte = mode.map(|m| m.cell_type == 10).unwrap_or(false);
            if is_embryocyte {
                let active_connections = state
                    .adhesion_manager
                    .count_active_adhesions(i, &state.adhesion_connections);
                let split_interval_valid = state.split_intervals[i] <= 59.0;
                // Only divide when free (no connections) and timer has elapsed
                if can_split_by_count
                    && active_connections == 0
                    && cell_age >= state.split_intervals[i]
                    && split_interval_valid
                {
                    state.divisions_to_process_buffer.push(i);
                }
                continue;
            }

            // Check nutrient threshold - cells must have enough nutrients to split
            // Values > 100 mean "never split" (UI sentinel: split_mass > 2.0 -> threshold > 100)
            // Lipocytes (cell_type 4) can have threshold up to 200
            // Reserve counts 1:1 toward the threshold - cells with reserve can split even
            // if their regular nutrients alone are below the threshold.
            let is_lipocyte = mode.map(|m| m.cell_type == 4).unwrap_or(false);
            let max_threshold = if is_lipocyte { 200.0 } else { 100.0 };
            let effective_nutrients = state.nutrients[i] + state.reserves[i] as f32 / 1000.0;
            let can_split_by_nutrients = state.split_nutrient_thresholds[i] <= max_threshold
                && effective_nutrients >= state.split_nutrient_thresholds[i];

            // Check time threshold - cells must be old enough to split
            let can_split_by_time = cell_age >= state.split_intervals[i];

            // Cell can split if ALL conditions are met
            let split_interval_valid = state.split_intervals[i] <= 59.0;
            if can_split_by_count
                && can_split_by_adhesions
                && can_split_by_signal
                && can_split_by_nutrients
                && can_split_by_time
                && split_interval_valid
            {
                state.divisions_to_process_buffer.push(i);
            }
        }

        // If no cells ready to split, we're done
        if state.divisions_to_process_buffer.is_empty() {
            break;
        }

        // Since we're skipping adhesions, process all divisions without filtering
        state.filtered_divisions_buffer.clear();
        state
            .filtered_divisions_buffer
            .extend_from_slice(&state.divisions_to_process_buffer);

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
            #[allow(dead_code)]
            parent_nutrients: f32,
            parent_radius: f32,
            parent_reserve: u32,
            parent_lineage_hash: u64,
            child_a_pos: Vec3,
            child_b_pos: Vec3,
            child_a_orientation: Quat,
            child_b_orientation: Quat,
            child_a_genome_orientation: Quat,
            child_b_genome_orientation: Quat,
            child_a_mode_idx: usize,
            child_b_mode_idx: usize,
            child_a_nutrients: f32,
            child_b_nutrients: f32,
            child_a_split_interval: f32,
            child_b_split_interval: f32,
            child_a_split_nutrient_threshold: f32,
            child_b_split_nutrient_threshold: f32,
            child_a_split_count: i32,
            child_b_split_count: i32,
            child_a_development_address: CellDevelopmentAddress,
            child_b_development_address: CellDevelopmentAddress,
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
                let parent_nutrients = state.nutrients[parent_idx];
                let parent_genome_id = state.genome_ids[parent_idx];
                let parent_stiffness = state.stiffnesses[parent_idx];
                let parent_split_count = state.split_counts[parent_idx];
                let parent_reserve = state.reserves[parent_idx];

                // Calculate split direction using parent's current rotation plus mode's split angle
                // This compounds the rotation each generation without needing to track state
                let pitch = mode.parent_split_direction.x.to_radians();
                let yaw = mode.parent_split_direction.y.to_radians();

                // Use genome orientation (not physics rotation) so the split axis is
                // stable across generations. Physics rotation drifts due to torques and
                // collisions; genome orientation is the pure accumulated split chain and
                // never drifts, which is what produces evenly-spaced spokes.
                let split_direction = parent_genome_orientation
                    * Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
                    * Vec3::Z;

                // Apply a tiny positional jitter to the child spawn offset so that
                // cells don't sit on a perfectly symmetric axis. This breaks the
                // degenerate case where every collision normal is identical, which
                // causes cells to stack in a line instead of sliding past each other.
                // The jitter is applied only to the offset direction, not to the
                // genome/physics orientations, so adhesion anchors are unaffected.
                let parent_cell_id = state.cell_ids[parent_idx];
                let jitter_rot = pseudo_random_rotation(parent_cell_id, rng_seed);
                let jittered_split_direction = (jitter_rot * split_direction).normalize();

                // 75% overlap means centers are 25% of combined diameter apart
                // Match reference convention: Child A at +offset, Child B at -offset
                let offset_distance = parent_radius * 0.25;
                let child_a_pos = parent_position + jittered_split_direction * offset_distance;
                let child_b_pos = parent_position - jittered_split_direction * offset_distance;

                // Get child mode indices
                // Check if children will reach max_splits after this division
                let will_reach_max_splits =
                    mode.max_splits >= 0 && (parent_split_count + 1) >= mode.max_splits;

                // If max_splits is reached and mode_a_after_splits is set, use that mode for Child A
                let mut child_a_mode_idx = if will_reach_max_splits && mode.mode_a_after_splits >= 0
                {
                    mode.mode_a_after_splits.max(0) as usize
                } else {
                    mode.child_a.mode_number.max(0) as usize
                };
                // If max_splits is reached and mode_b_after_splits is set, use that mode for Child B
                let mut child_b_mode_idx = if will_reach_max_splits && mode.mode_b_after_splits >= 0
                {
                    mode.mode_b_after_splits.max(0) as usize
                } else {
                    mode.child_b.mode_number.max(0) as usize
                };

                // Signal-conditional child mode routing: override child modes based on parent's signal state
                if mode.signal_child_a_channel >= 8 && (mode.signal_child_a_channel as usize) <= 15
                {
                    let ch = mode.signal_child_a_channel as usize;
                    let signal_val = state.signal_channels[parent_idx * 16 + ch].unwrap_or(0.0);
                    if signal_val >= mode.signal_child_a_threshold {
                        if mode.signal_child_a_mode_above >= 0 {
                            child_a_mode_idx = mode.signal_child_a_mode_above as usize;
                        }
                    } else {
                        if mode.signal_child_a_mode_below >= 0 {
                            child_a_mode_idx = mode.signal_child_a_mode_below as usize;
                        }
                    }
                }
                if mode.signal_child_b_channel >= 8 && (mode.signal_child_b_channel as usize) <= 15
                {
                    let ch = mode.signal_child_b_channel as usize;
                    let signal_val = state.signal_channels[parent_idx * 16 + ch].unwrap_or(0.0);
                    if signal_val >= mode.signal_child_b_threshold {
                        if mode.signal_child_b_mode_above >= 0 {
                            child_b_mode_idx = mode.signal_child_b_mode_above as usize;
                        }
                    } else {
                        if mode.signal_child_b_mode_below >= 0 {
                            child_b_mode_idx = mode.signal_child_b_mode_below as usize;
                        }
                    }
                }

                // Embryocyte rule: children of Embryocytes (cell_type == 10) cannot
                // themselves be Embryocytes (would create reserve-doubling chains).
                // Remap any Embryocyte child mode to mode 0 as a safety guard.
                if mode.cell_type == 10 {
                    if genome
                        .modes
                        .get(child_a_mode_idx)
                        .map(|m| m.cell_type == 10)
                        .unwrap_or(false)
                    {
                        child_a_mode_idx = 0;
                    }
                    if genome
                        .modes
                        .get(child_b_mode_idx)
                        .map(|m| m.cell_type == 10)
                        .unwrap_or(false)
                    {
                        child_b_mode_idx = 0;
                    }
                }

                // Determine split counts for children:
                // - Reset to 0 when the child's mode changes (new mode starts fresh).
                // - Reset to 0 when the after-splits routing fires AND routes to a
                //   DIFFERENT mode than the normal child mode - this is a deliberate
                //   lifecycle transition (e.g. stem cell shedding an egg then continuing
                //   to grow). Without the reset, the child that stays in the same mode
                //   via mode_b_after_splits would have split_count >= max_splits and
                //   immediately fail can_split_by_count on the very next tick.
                // - Setting mode_a/b_after_splits to the same mode as the normal child
                //   is identical to leaving it at default (-1) - no reset, cell stops.
                // - Otherwise increment. If max_splits is reached with no after-splits
                //   routing, the count stays >= max_splits and the cell stops dividing.
                let normal_child_a_mode = mode.child_a.mode_number.max(0) as usize;
                let normal_child_b_mode = mode.child_b.mode_number.max(0) as usize;
                let after_splits_a_fires = will_reach_max_splits
                    && mode.mode_a_after_splits >= 0
                    && mode.mode_a_after_splits as usize != normal_child_a_mode;
                let after_splits_b_fires = will_reach_max_splits
                    && mode.mode_b_after_splits >= 0
                    && mode.mode_b_after_splits as usize != normal_child_b_mode;
                let child_a_split_count = if child_a_mode_idx != mode_index || after_splits_a_fires
                {
                    0
                } else {
                    parent_split_count + 1
                };
                let child_b_split_count = if child_b_mode_idx != mode_index || after_splits_b_fires
                {
                    0
                } else {
                    parent_split_count + 1
                };

                // Get child properties
                let child_a_mode = genome.modes.get(child_a_mode_idx);
                let child_b_mode = genome.modes.get(child_b_mode_idx);

                // Split parent's nutrients according to split_ratio
                // split_ratio determines what fraction goes to Child A (0.0 to 1.0)
                let split_ratio = mode.split_ratio.clamp(0.0, 1.0);
                let child_a_nutrients = parent_nutrients * split_ratio;
                let child_b_nutrients = parent_nutrients * (1.0 - split_ratio);

                // Get split intervals (potentially randomized from range)
                // Use parent cell_id + tick for deterministic randomness
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

                // Get split mass thresholds and convert to nutrient thresholds
                // nutrient_threshold = (split_mass - 1.0) * 100.0
                let child_a_split_nutrient_threshold = if let Some(m) = child_a_mode {
                    let split_mass = m.get_split_mass(parent_cell_id, tick, rng_seed);
                    (split_mass - 1.0) * 100.0
                } else {
                    50.0 // Default: split_mass 1.5 -> 50 nutrients
                };

                let child_b_split_nutrient_threshold = if let Some(m) = child_b_mode {
                    let split_mass = m.get_split_mass(parent_cell_id + 1, tick, rng_seed);
                    (split_mass - 1.0) * 100.0
                } else {
                    50.0
                };

                // Calculate split rotation for both physics and genome orientations
                // This compounds the rotation each generation
                let split_rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);

                // Genome orientations also inherit split angle compounding for proper adhesion anchor positioning
                // This ensures adhesion angles are correctly calculated through generations
                // Use split orientations if max_splits is reached, otherwise use regular child orientations
                let child_a_orientation_for_genome = if will_reach_max_splits {
                    mode.child_a_after_split_orientation
                } else {
                    mode.child_a.orientation
                };
                let child_b_orientation_for_genome = if will_reach_max_splits {
                    mode.child_b_after_split_orientation
                } else {
                    mode.child_b.orientation
                };

                let child_a_genome_orientation =
                    parent_genome_orientation * split_rotation * child_a_orientation_for_genome;
                let child_b_genome_orientation =
                    parent_genome_orientation * split_rotation * child_b_orientation_for_genome;

                // Physics rotations inherit from parent's physics rotation + split angle + child orientation delta
                // This compounds the rotation each generation and applies genome-specified orientation
                let child_a_orientation =
                    parent_rotation * split_rotation * child_a_orientation_for_genome;
                let child_b_orientation =
                    parent_rotation * split_rotation * child_b_orientation_for_genome;
                let mut child_a_development_address = state.derive_division_development_address(
                    parent_idx,
                    mode_index,
                    child_a_mode_idx,
                    1,
                );
                let mut child_b_development_address = state.derive_division_development_address(
                    parent_idx,
                    mode_index,
                    child_b_mode_idx,
                    2,
                );
                let initial_mode_idx = genome.initial_mode.max(0) as usize;
                if mode_index != initial_mode_idx && child_a_mode_idx == initial_mode_idx {
                    child_a_development_address.organism_cell_id = 1;
                }
                if mode_index != initial_mode_idx && child_b_mode_idx == initial_mode_idx {
                    child_b_development_address.organism_cell_id = 1;
                }

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
                    parent_nutrients,
                    parent_radius,
                    parent_reserve,
                    parent_lineage_hash: state.lineage_hashes[parent_idx],
                    child_a_pos,
                    child_b_pos,
                    child_a_orientation,
                    child_b_orientation,
                    child_a_genome_orientation,
                    child_b_genome_orientation,
                    child_a_mode_idx,
                    child_b_mode_idx,
                    child_a_nutrients,
                    child_b_nutrients,
                    child_a_split_interval,
                    child_b_split_interval,
                    child_a_split_nutrient_threshold,
                    child_b_split_nutrient_threshold,
                    child_a_split_count,
                    child_b_split_count,
                    child_a_development_address,
                    child_b_development_address,
                });
            }
        }

        // Now write all the children to their allocated slots
        for data in &division_data_list {
            // Children are born at current_time
            let child_birth_time = current_time;

            // Reserve halving: each child gets half the parent's reserve (integer shift).
            let child_reserve = data.parent_reserve >> 1;

            if data.child_a_slot < state.capacity {
                // Write child A
                let child_a_id = state.next_cell_id;
                state.cell_ids[data.child_a_slot] = child_a_id;
                state.next_cell_id += 1;
                state.set_development_address(data.child_a_slot, data.child_a_development_address);
                state.parent_lineage_hashes[data.child_a_slot] = data.parent_lineage_hash;
                state.positions[data.child_a_slot] = data.child_a_pos;
                state.prev_positions[data.child_a_slot] = data.child_a_pos;
                state.velocities[data.child_a_slot] = data.parent_velocity;

                // Derive mass and radius from nutrients
                state.nutrients[data.child_a_slot] = data.child_a_nutrients;
                let child_a_mass = 1.0 + data.child_a_nutrients / 100.0;
                state.masses[data.child_a_slot] = child_a_mass;
                state.radii[data.child_a_slot] = child_a_mass.clamp(0.5, 2.0);

                state.genome_ids[data.child_a_slot] = data.parent_genome_id;
                state.mode_indices[data.child_a_slot] = data.child_a_mode_idx;

                // Physics rotation = genome rotation chain (no random perturbation for determinism)
                state.rotations[data.child_a_slot] = data.child_a_orientation;

                state.genome_orientations[data.child_a_slot] = data.child_a_genome_orientation;
                state.angular_velocities[data.child_a_slot] = Vec3::ZERO;
                state.forces[data.child_a_slot] = Vec3::ZERO;
                state.torques[data.child_a_slot] = Vec3::ZERO;
                state.accelerations[data.child_a_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_a_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_a_slot] = data.parent_stiffness;
                state.birth_times[data.child_a_slot] = child_birth_time;
                state.split_intervals[data.child_a_slot] = data.child_a_split_interval;
                state.split_nutrient_thresholds[data.child_a_slot] =
                    data.child_a_split_nutrient_threshold;
                state.split_counts[data.child_a_slot] = data.child_a_split_count;
                // Inherit halved reserve; reset Embryocyte accumulation timer
                state.reserves[data.child_a_slot] = child_reserve;
                state.embryocyte_timers[data.child_a_slot] = 0.0;
            }

            if data.child_b_slot < state.capacity {
                // Write child B
                let child_b_id = state.next_cell_id;
                state.cell_ids[data.child_b_slot] = child_b_id;
                state.next_cell_id += 1;
                state.set_development_address(data.child_b_slot, data.child_b_development_address);
                state.parent_lineage_hashes[data.child_b_slot] = data.parent_lineage_hash;
                state.positions[data.child_b_slot] = data.child_b_pos;
                state.prev_positions[data.child_b_slot] = data.child_b_pos;
                state.velocities[data.child_b_slot] = data.parent_velocity;

                // Derive mass and radius from nutrients
                state.nutrients[data.child_b_slot] = data.child_b_nutrients;
                let child_b_mass = 1.0 + data.child_b_nutrients / 100.0;
                state.masses[data.child_b_slot] = child_b_mass;
                state.radii[data.child_b_slot] = child_b_mass.clamp(0.5, 2.0);

                state.genome_ids[data.child_b_slot] = data.parent_genome_id;
                state.mode_indices[data.child_b_slot] = data.child_b_mode_idx;

                // Physics rotation = genome rotation chain (no random perturbation for determinism)
                state.rotations[data.child_b_slot] = data.child_b_orientation;

                state.genome_orientations[data.child_b_slot] = data.child_b_genome_orientation;
                state.angular_velocities[data.child_b_slot] = Vec3::ZERO;
                state.forces[data.child_b_slot] = Vec3::ZERO;
                state.torques[data.child_b_slot] = Vec3::ZERO;
                state.accelerations[data.child_b_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_b_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_b_slot] = data.parent_stiffness;
                state.birth_times[data.child_b_slot] = child_birth_time;
                state.split_intervals[data.child_b_slot] = data.child_b_split_interval;
                state.split_nutrient_thresholds[data.child_b_slot] =
                    data.child_b_split_nutrient_threshold;
                state.split_counts[data.child_b_slot] = data.child_b_split_count;
                // Inherit halved reserve; reset Embryocyte accumulation timer
                state.reserves[data.child_b_slot] = child_reserve;
                state.embryocyte_timers[data.child_b_slot] = 0.0;
            }

            // Record the division event
            pass_division_events.push(DivisionEvent {
                parent_idx: data.parent_idx,
                child_a_idx: data.child_a_slot,
                child_b_idx: data.child_b_slot,
            });

            // Log the division: parent_hash → (child_a_hash, child_b_hash)
            // Used by the scaffold resolver to trace descendants through dead ancestors.
            state.division_log.insert(
                data.parent_lineage_hash,
                (
                    data.child_a_development_address.lineage_hash,
                    data.child_b_development_address.lineage_hash,
                ),
            );

            // Handle adhesion inheritance from parent to children
            // This must happen AFTER children are written but BEFORE new adhesions are created
            inherit_adhesions_on_division(
                state,
                genome,
                data.parent_mode_idx,
                data.child_a_slot,
                data.child_b_slot,
                data.parent_genome_orientation,
                current_time,
                data.parent_split_count,
                data.parent_radius, // Use the saved pre-split parent radius, not the child's post-split radius
            );

            // Create sibling adhesion between children if parent_make_adhesion is enabled.
            // Normal keep_adhesion flags only affect inheritance; after-split keep flags
            // also suppress the sibling bond when the max-splits transition fires.
            let parent_mode = genome.modes.get(data.parent_mode_idx);

            if let Some(mode) = parent_mode {
                let will_reach_max_splits =
                    mode.max_splits >= 0 && (data.parent_split_count + 1) >= mode.max_splits;
                let after_split_sibling_allowed = !will_reach_max_splits
                    || (mode.child_a_after_split_keep_adhesion
                        && mode.child_b_after_split_keep_adhesion);

                if mode.parent_make_adhesion && after_split_sibling_allowed {
                    // Calculate anchor directions based on compounded genome orientations (matches Python reference)
                    // Python: angle1_relative = (spawn_direction + math.pi) - daughter1.arrow_direction
                    // Python: angle2_relative = spawn_direction - daughter2.arrow_direction

                    // Get the spawn direction from parent's genome orientation + split angle
                    let pitch = mode.parent_split_direction.x.to_radians();
                    let yaw = mode.parent_split_direction.y.to_radians();
                    let split_rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
                    let spawn_direction_quat = data.parent_genome_orientation * split_rotation;
                    let spawn_direction_vec = spawn_direction_quat * Vec3::Z;

                    // Daughter A anchor: points toward Daughter B (opposite spawn direction)
                    let direction_a_to_b = -spawn_direction_vec;
                    // Transform to Daughter A's local genome space (must use genome orientation,
                    // not physics orientation, so anchors stay consistent with inherited bonds
                    // which are also expressed in genome space)
                    let anchor_direction_a =
                        (data.child_a_genome_orientation.inverse() * direction_a_to_b).normalize();

                    // Daughter B anchor: points toward Daughter A (same as spawn direction)
                    let direction_b_to_a = spawn_direction_vec;
                    // Transform to Daughter B's local genome space
                    let anchor_direction_b =
                        (data.child_b_genome_orientation.inverse() * direction_b_to_a).normalize();

                    // Get split directions for zone classification
                    let child_a_mode = genome.modes.get(data.child_a_mode_idx);
                    let child_b_mode = genome.modes.get(data.child_b_mode_idx);

                    // Match GPU: use parent's split direction for sibling zone classification
                    let child_a_split_dir = split_rotation * Vec3::Z;
                    let child_b_split_dir = split_rotation * Vec3::Z;

                    // Get split ratios for zone classification
                    let child_a_split_ratio = child_a_mode.map(|m| m.split_ratio).unwrap_or(0.5);
                    let child_b_split_ratio = child_b_mode.map(|m| m.split_ratio).unwrap_or(0.5);

                    // Create child-to-child connection with parent's mode index.
                    // Disconnect any existing bond on child A whose anchor occupies the
                    // same direction as the new sibling bond (within ANCHOR_OVERLAP_COS).
                    // Both anchors are compared in child A's LOCAL frame so the test is
                    // independent of genome orientation. This matches the GPU shader.
                    {
                        let mut old_bond_to_remove: Option<usize> = None;
                        for slot in 0..crate::cell::MAX_ADHESIONS_PER_CELL {
                            let conn_idx = state.adhesion_manager.cell_adhesion_indices
                                [data.child_a_slot][slot];
                            if conn_idx < 0 {
                                continue;
                            }
                            let conn_idx = conn_idx as usize;
                            if conn_idx >= state.adhesion_connections.active_count {
                                continue;
                            }
                            if state.adhesion_connections.is_active[conn_idx] == 0 {
                                continue;
                            }
                            // This cell's (child A's) side anchor in its own local frame
                            let existing_anchor_local = if state.adhesion_connections.cell_a_index
                                [conn_idx]
                                == data.child_a_slot
                            {
                                state.adhesion_connections.anchor_direction_a[conn_idx]
                            } else {
                                state.adhesion_connections.anchor_direction_b[conn_idx]
                            };
                            if anchor_direction_a.dot(existing_anchor_local)
                                > crate::cell::ANCHOR_OVERLAP_COS
                            {
                                old_bond_to_remove = Some(conn_idx);
                                break;
                            }
                        }
                        if let Some(idx) = old_bond_to_remove {
                            state
                                .adhesion_manager
                                .remove_adhesion(&mut state.adhesion_connections, idx);
                        }
                    }

                    let _result = state.adhesion_manager.add_adhesion_with_directions(
                        &mut state.adhesion_connections,
                        data.child_a_slot,
                        data.child_b_slot,
                        data.parent_mode_idx, // Use parent's mode index (matches reference)
                        anchor_direction_a,
                        anchor_direction_b,
                        child_a_split_dir,
                        child_b_split_dir,
                        data.child_a_orientation,
                        data.child_b_orientation,
                        child_a_split_ratio,
                        child_b_split_ratio,
                        current_time,
                    );
                } else {
                    // parent_make_adhesion is off: grant 1-second sister immunity so
                    // Glueocyte children don't immediately bond to each other on contact.
                    let id_a = state.cell_ids[data.child_a_slot];
                    let id_b = state.cell_ids[data.child_b_slot];
                    let expiry = current_time + 1.0;
                    state.sister_cell_id[data.child_a_slot] = id_b;
                    state.sister_expiry[data.child_a_slot] = expiry;
                    state.sister_cell_id[data.child_b_slot] = id_a;
                    state.sister_expiry[data.child_b_slot] = expiry;
                }
            }
        }

        // Update cell count (child B cells are already written)
        let new_cell_count = state.cell_count + division_data_list.len();
        state.cell_count = new_cell_count;

        // Kill children born under minimum nutrients (1.0 = death threshold)
        const MIN_NUTRIENTS: f32 = 1.0;
        let mut dead_on_arrival = Vec::new();
        for data in &division_data_list {
            if data.child_b_nutrients < MIN_NUTRIENTS && data.child_b_slot < state.cell_count {
                dead_on_arrival.push(data.child_b_slot);
            }
            if data.child_a_nutrients < MIN_NUTRIENTS && data.child_a_slot < state.cell_count {
                dead_on_arrival.push(data.child_a_slot);
            }
        }
        if !dead_on_arrival.is_empty() {
            // Sort descending so swap-with-last removal doesn't invalidate earlier indices
            dead_on_arrival.sort_unstable_by(|a, b| b.cmp(a));
            dead_on_arrival.dedup();
            for idx in &dead_on_arrival {
                state.remove_cell(*idx);
            }
        }

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
            // Check nutrient threshold - cells must have enough nutrients to split
            // Values > 100 mean "never split" (UI sentinel: split_mass > 2.0 -> threshold > 100)
            // Lipocytes (cell_type 4) can have threshold up to 200
            // Reserve counts 1:1 toward the threshold - cells with reserve can split even
            // if their regular nutrients alone are below the threshold.
            let is_lipocyte = mode.map(|m| m.cell_type == 4).unwrap_or(false);
            let max_threshold = if is_lipocyte { 200.0 } else { 100.0 };
            let effective_nutrients = state.nutrients[i] + state.reserves[i] as f32 / 1000.0;
            let can_split_by_nutrients = state.split_nutrient_thresholds[i] <= max_threshold
                && effective_nutrients >= state.split_nutrient_thresholds[i];

            // Check time threshold - cells must be old enough to split
            let can_split_by_time = cell_age >= state.split_intervals[i];

            let split_interval_valid = state.split_intervals[i] <= 59.0;
            if can_split_by_count
                && can_split_by_adhesions
                && can_split_by_nutrients
                && can_split_by_time
                && split_interval_valid
            {
                state.divisions_to_process_buffer.push(i);
            }
        }

        if state.divisions_to_process_buffer.is_empty() {
            break;
        }

        state.filtered_divisions_buffer.clear();
        state
            .filtered_divisions_buffer
            .extend_from_slice(&state.divisions_to_process_buffer);

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
            #[allow(dead_code)]
            parent_nutrients: f32,
            parent_reserve: u32,
            parent_lineage_hash: u64,
            child_a_pos: Vec3,
            child_b_pos: Vec3,
            child_a_orientation: Quat,
            child_b_orientation: Quat,
            child_a_genome_orientation: Quat,
            child_b_genome_orientation: Quat,
            child_a_mode_idx: usize,
            child_b_mode_idx: usize,
            child_a_nutrients: f32,
            child_b_nutrients: f32,
            child_a_split_interval: f32,
            child_b_split_interval: f32,
            child_a_split_nutrient_threshold: f32,
            child_b_split_nutrient_threshold: f32,
            child_a_split_count: i32,
            child_b_split_count: i32,
            child_a_development_address: CellDevelopmentAddress,
            child_b_development_address: CellDevelopmentAddress,
            split_direction_local: Vec3,
            parent_radius: f32,
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
            let parent_nutrients = state.nutrients[parent_idx];
            let parent_stiffness = state.stiffnesses[parent_idx];
            let parent_split_count = state.split_counts[parent_idx];
            let parent_reserve = state.reserves[parent_idx];

            let pitch = mode.parent_split_direction.x.to_radians();
            let yaw = mode.parent_split_direction.y.to_radians();
            // Use genome orientation (not physics rotation) - same reasoning as the
            // primary division path above: genome orientation is drift-free.
            let split_direction = parent_genome_orientation
                * Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
                * Vec3::Z;

            let offset_distance = parent_radius * 0.25;
            let child_a_pos = parent_position + split_direction * offset_distance;
            let child_b_pos = parent_position - split_direction * offset_distance;

            let will_reach_max_splits =
                mode.max_splits >= 0 && (parent_split_count + 1) >= mode.max_splits;

            let mut child_a_mode_idx = if will_reach_max_splits && mode.mode_a_after_splits >= 0 {
                mode.mode_a_after_splits.max(0) as usize
            } else {
                mode.child_a.mode_number.max(0) as usize
            };
            let mut child_b_mode_idx = if will_reach_max_splits && mode.mode_b_after_splits >= 0 {
                mode.mode_b_after_splits.max(0) as usize
            } else {
                mode.child_b.mode_number.max(0) as usize
            };

            // Signal-conditional child mode routing: override child modes based on parent's signal state
            if mode.signal_child_a_channel >= 0 && (mode.signal_child_a_channel as usize) < 16 {
                let ch = mode.signal_child_a_channel as usize;
                let signal_val = state.signal_channels[parent_idx * 16 + ch].unwrap_or(0.0);
                if signal_val >= mode.signal_child_a_threshold {
                    if mode.signal_child_a_mode_above >= 0 {
                        child_a_mode_idx = mode.signal_child_a_mode_above as usize;
                    }
                } else {
                    if mode.signal_child_a_mode_below >= 0 {
                        child_a_mode_idx = mode.signal_child_a_mode_below as usize;
                    }
                }
            }
            if mode.signal_child_b_channel >= 0 && (mode.signal_child_b_channel as usize) < 16 {
                let ch = mode.signal_child_b_channel as usize;
                let signal_val = state.signal_channels[parent_idx * 16 + ch].unwrap_or(0.0);
                if signal_val >= mode.signal_child_b_threshold {
                    if mode.signal_child_b_mode_above >= 0 {
                        child_b_mode_idx = mode.signal_child_b_mode_above as usize;
                    }
                } else {
                    if mode.signal_child_b_mode_below >= 0 {
                        child_b_mode_idx = mode.signal_child_b_mode_below as usize;
                    }
                }
            }

            // Embryocyte rule: children of Embryocytes (cell_type == 10) cannot
            // themselves be Embryocytes (would create reserve-doubling chains).
            if mode.cell_type == 10 {
                if genome
                    .modes
                    .get(child_a_mode_idx)
                    .map(|m| m.cell_type == 10)
                    .unwrap_or(false)
                {
                    child_a_mode_idx = 0;
                }
                if genome
                    .modes
                    .get(child_b_mode_idx)
                    .map(|m| m.cell_type == 10)
                    .unwrap_or(false)
                {
                    child_b_mode_idx = 0;
                }
            }

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

            let child_a_mode = genome.modes.get(child_a_mode_idx);
            let child_b_mode = genome.modes.get(child_b_mode_idx);

            // Split parent's nutrients according to split_ratio
            let split_ratio = mode.split_ratio.clamp(0.0, 1.0);
            let child_a_nutrients = parent_nutrients * split_ratio;
            let child_b_nutrients = parent_nutrients * (1.0 - split_ratio);

            let parent_cell_id = state.cell_ids[parent_idx];
            let tick = (current_time * 60.0) as u64;

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

            // Convert split_mass to nutrient threshold: (split_mass - 1.0) * 100.0
            let child_a_split_nutrient_threshold = if let Some(m) = child_a_mode {
                let split_mass = m.get_split_mass(parent_cell_id, tick, rng_seed);
                (split_mass - 1.0) * 100.0
            } else {
                50.0
            };

            let child_b_split_nutrient_threshold = if let Some(m) = child_b_mode {
                let split_mass = m.get_split_mass(parent_cell_id + 1, tick, rng_seed);
                (split_mass - 1.0) * 100.0
            } else {
                50.0
            };

            // Match division_step: compound split_rotation into genome orientations so
            // adhesion anchor directions are computed in the correct frame across generations.
            let split_rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
            let child_a_orientation_for_genome = if will_reach_max_splits {
                mode.child_a_after_split_orientation
            } else {
                mode.child_a.orientation
            };
            let child_b_orientation_for_genome = if will_reach_max_splits {
                mode.child_b_after_split_orientation
            } else {
                mode.child_b.orientation
            };
            let child_a_genome_orientation =
                parent_genome_orientation * split_rotation * child_a_orientation_for_genome;
            let child_b_genome_orientation =
                parent_genome_orientation * split_rotation * child_b_orientation_for_genome;
            let child_a_orientation =
                parent_rotation * split_rotation * child_a_orientation_for_genome;
            let child_b_orientation =
                parent_rotation * split_rotation * child_b_orientation_for_genome;
            let mut child_a_development_address = state.derive_division_development_address(
                parent_idx,
                mode_index,
                child_a_mode_idx,
                1,
            );
            let mut child_b_development_address = state.derive_division_development_address(
                parent_idx,
                mode_index,
                child_b_mode_idx,
                2,
            );
            let initial_mode_idx = genome.initial_mode.max(0) as usize;
            if mode_index != initial_mode_idx && child_a_mode_idx == initial_mode_idx {
                child_a_development_address.organism_cell_id = 1;
            }
            if mode_index != initial_mode_idx && child_b_mode_idx == initial_mode_idx {
                child_b_development_address.organism_cell_id = 1;
            }

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
                parent_nutrients,
                parent_reserve,
                parent_lineage_hash: state.lineage_hashes[parent_idx],
                child_a_pos,
                child_b_pos,
                child_a_orientation,
                child_b_orientation,
                child_a_genome_orientation,
                child_b_genome_orientation,
                child_a_mode_idx,
                child_b_mode_idx,
                child_a_nutrients,
                child_b_nutrients,
                child_a_split_interval,
                child_b_split_interval,
                child_a_split_nutrient_threshold,
                child_b_split_nutrient_threshold,
                child_a_split_count,
                child_b_split_count,
                child_a_development_address,
                child_b_development_address,
                split_direction_local: Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0) * Vec3::Z,
                parent_radius,
            });
        }

        // Write children to slots
        for data in &division_data_list {
            let child_birth_time = current_time;
            let genome = &genomes[data.parent_genome_id];

            // Reserve halving: each child gets half the parent's reserve (integer shift).
            let child_reserve = data.parent_reserve >> 1;

            if data.child_a_slot < state.capacity {
                let child_a_id = state.next_cell_id;
                state.cell_ids[data.child_a_slot] = child_a_id;
                state.next_cell_id += 1;
                state.set_development_address(data.child_a_slot, data.child_a_development_address);
                state.parent_lineage_hashes[data.child_a_slot] = data.parent_lineage_hash;
                state.positions[data.child_a_slot] = data.child_a_pos;
                state.prev_positions[data.child_a_slot] = data.child_a_pos;
                state.velocities[data.child_a_slot] = data.parent_velocity;

                // Derive mass and radius from nutrients
                state.nutrients[data.child_a_slot] = data.child_a_nutrients;
                let child_a_mass = 1.0 + data.child_a_nutrients / 100.0;
                state.masses[data.child_a_slot] = child_a_mass;
                state.radii[data.child_a_slot] = child_a_mass.clamp(0.5, 2.0);

                state.genome_ids[data.child_a_slot] = data.parent_genome_id;
                state.mode_indices[data.child_a_slot] = data.child_a_mode_idx;
                state.rotations[data.child_a_slot] = data.child_a_orientation;
                state.genome_orientations[data.child_a_slot] = data.child_a_genome_orientation;
                state.angular_velocities[data.child_a_slot] = Vec3::ZERO;
                state.forces[data.child_a_slot] = Vec3::ZERO;
                state.torques[data.child_a_slot] = Vec3::ZERO;
                state.accelerations[data.child_a_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_a_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_a_slot] = data.parent_stiffness;
                state.birth_times[data.child_a_slot] = child_birth_time;
                state.split_intervals[data.child_a_slot] = data.child_a_split_interval;
                state.split_nutrient_thresholds[data.child_a_slot] =
                    data.child_a_split_nutrient_threshold;
                state.split_counts[data.child_a_slot] = data.child_a_split_count;
                // Inherit halved reserve; reset Embryocyte accumulation timer
                state.reserves[data.child_a_slot] = child_reserve;
                state.embryocyte_timers[data.child_a_slot] = 0.0;
            }

            if data.child_b_slot < state.capacity {
                let child_b_id = state.next_cell_id;
                state.cell_ids[data.child_b_slot] = child_b_id;
                state.next_cell_id += 1;
                state.set_development_address(data.child_b_slot, data.child_b_development_address);
                state.parent_lineage_hashes[data.child_b_slot] = data.parent_lineage_hash;
                state.positions[data.child_b_slot] = data.child_b_pos;
                state.prev_positions[data.child_b_slot] = data.child_b_pos;
                state.velocities[data.child_b_slot] = data.parent_velocity;

                // Derive mass and radius from nutrients
                state.nutrients[data.child_b_slot] = data.child_b_nutrients;
                let child_b_mass = 1.0 + data.child_b_nutrients / 100.0;
                state.masses[data.child_b_slot] = child_b_mass;
                state.radii[data.child_b_slot] = child_b_mass.clamp(0.5, 2.0);

                state.genome_ids[data.child_b_slot] = data.parent_genome_id;
                state.mode_indices[data.child_b_slot] = data.child_b_mode_idx;
                state.rotations[data.child_b_slot] = data.child_b_orientation;
                state.genome_orientations[data.child_b_slot] = data.child_b_genome_orientation;
                state.angular_velocities[data.child_b_slot] = Vec3::ZERO;
                state.forces[data.child_b_slot] = Vec3::ZERO;
                state.torques[data.child_b_slot] = Vec3::ZERO;
                state.accelerations[data.child_b_slot] = Vec3::ZERO;
                state.prev_accelerations[data.child_b_slot] = Vec3::ZERO;
                state.stiffnesses[data.child_b_slot] = data.parent_stiffness;
                state.birth_times[data.child_b_slot] = child_birth_time;
                state.split_intervals[data.child_b_slot] = data.child_b_split_interval;
                state.split_nutrient_thresholds[data.child_b_slot] =
                    data.child_b_split_nutrient_threshold;
                state.split_counts[data.child_b_slot] = data.child_b_split_count;
                // Inherit halved reserve; reset Embryocyte accumulation timer
                state.reserves[data.child_b_slot] = child_reserve;
                state.embryocyte_timers[data.child_b_slot] = 0.0;
            }

            // Log the division for scaffold descendant tracing
            state.division_log.insert(
                data.parent_lineage_hash,
                (
                    data.child_a_development_address.lineage_hash,
                    data.child_b_development_address.lineage_hash,
                ),
            );

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
                current_time,
                data.parent_split_count,
                data.parent_radius,
            );

            // Create sibling adhesion between children if parent_make_adhesion is enabled.
            // Normal keep_adhesion flags only affect inheritance; after-split keep flags
            // also suppress the sibling bond when the max-splits transition fires.
            let parent_mode = genome.modes.get(data.parent_mode_idx);

            if let Some(mode) = parent_mode {
                let will_reach_max_splits =
                    mode.max_splits >= 0 && (data.parent_split_count + 1) >= mode.max_splits;
                let after_split_sibling_allowed = !will_reach_max_splits
                    || (mode.child_a_after_split_keep_adhesion
                        && mode.child_b_after_split_keep_adhesion);

                if mode.parent_make_adhesion && after_split_sibling_allowed {
                    // Use the full world-space spawn direction (parent_genome_orientation * split_rotation),
                    // then transform into each child's genome-local space - matching division_step exactly.
                    let pitch = mode.parent_split_direction.x.to_radians();
                    let yaw = mode.parent_split_direction.y.to_radians();
                    let split_rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
                    let spawn_direction_vec =
                        (data.parent_genome_orientation * split_rotation) * Vec3::Z;

                    let anchor_direction_a = (data.child_a_genome_orientation.inverse()
                        * (-spawn_direction_vec))
                        .normalize();
                    let anchor_direction_b = (data.child_b_genome_orientation.inverse()
                        * spawn_direction_vec)
                        .normalize();

                    let child_a_mode = genome.modes.get(data.child_a_mode_idx);
                    let child_b_mode = genome.modes.get(data.child_b_mode_idx);

                    // Match GPU: use parent's split direction for sibling zone classification
                    let child_a_split_dir = data.split_direction_local;
                    let child_b_split_dir = data.split_direction_local;

                    // Get split ratios for zone classification
                    let child_a_split_ratio = child_a_mode.map(|m| m.split_ratio).unwrap_or(0.5);
                    let child_b_split_ratio = child_b_mode.map(|m| m.split_ratio).unwrap_or(0.5);

                    // Calculate global mode index for adhesion settings lookup
                    let global_mode_idx = genome_mode_offsets
                        .get(data.parent_genome_id)
                        .copied()
                        .unwrap_or(0)
                        + data.parent_mode_idx;

                    // Disconnect any existing bond on child A whose anchor occupies the same
                    // direction as the new sibling bond (compared in child A's local frame).
                    // Matches division_step and the GPU shader so both scenes behave identically.
                    {
                        let mut old_bond_to_remove: Option<usize> = None;
                        for slot in 0..crate::cell::MAX_ADHESIONS_PER_CELL {
                            let conn_idx = state.adhesion_manager.cell_adhesion_indices
                                [data.child_a_slot][slot];
                            if conn_idx < 0 {
                                continue;
                            }
                            let conn_idx = conn_idx as usize;
                            if conn_idx >= state.adhesion_connections.active_count {
                                continue;
                            }
                            if state.adhesion_connections.is_active[conn_idx] == 0 {
                                continue;
                            }
                            let existing_anchor_local = if state.adhesion_connections.cell_a_index
                                [conn_idx]
                                == data.child_a_slot
                            {
                                state.adhesion_connections.anchor_direction_a[conn_idx]
                            } else {
                                state.adhesion_connections.anchor_direction_b[conn_idx]
                            };
                            if anchor_direction_a.dot(existing_anchor_local)
                                > crate::cell::ANCHOR_OVERLAP_COS
                            {
                                old_bond_to_remove = Some(conn_idx);
                                break;
                            }
                        }
                        if let Some(idx) = old_bond_to_remove {
                            state
                                .adhesion_manager
                                .remove_adhesion(&mut state.adhesion_connections, idx);
                        }
                    }

                    let _ = state.adhesion_manager.add_adhesion_with_directions(
                        &mut state.adhesion_connections,
                        data.child_a_slot,
                        data.child_b_slot,
                        global_mode_idx, // Use global mode index for multi-genome support
                        anchor_direction_a,
                        anchor_direction_b,
                        child_a_split_dir,
                        child_b_split_dir,
                        data.child_a_orientation,
                        data.child_b_orientation,
                        child_a_split_ratio,
                        child_b_split_ratio,
                        current_time,
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
