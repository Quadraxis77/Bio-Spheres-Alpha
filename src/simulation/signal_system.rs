//! Signal system for oculocyte sensing and inter-cell communication.
//!
//! Oculocytes sense targets (cells, food, light, barriers) along their forward direction
//! and add nonnegative production to a conservative diffusion field.
//!
//! Signal semantics:
//! - `None` = null (no signal on this channel)
//! - `Some(value)` = a nonnegative receiver value
//! - Unsaturated concentrations persist; only receiver inputs saturate to `0..1000`
//! - Signals update on the fixed 15 Hz signal clock

use crate::genome::{Genome, SignalResponseMode};
use crate::simulation::canonical_state::CanonicalState;
use glam::Vec3;

/// Number of signal channels (0-15)
pub const SIGNAL_CHANNELS: usize = 16;
pub const SIGNAL_MIN: f32 = 0.0;
pub const SIGNAL_MAX: f32 = 1000.0;
pub const SIGNAL_TICK_HZ: f32 = 15.0;
pub const SIGNAL_TICK_SECONDS: f32 = 1.0 / SIGNAL_TICK_HZ;
pub const MAX_SIGNAL_CATCH_UP_TICKS: usize = 4;
pub const REFERENCE_BASELINE_MAINTENANCE_PER_SECOND: f32 = 1.0;
pub const SIGNAL_BOND_CONSTRUCTION_FRACTION: f32 = 0.05;

#[inline]
pub fn signal_bond_construction_cost(next_division_requirement: f32) -> Option<f32> {
    (next_division_requirement.is_finite() && next_division_requirement >= 0.0)
        .then_some(next_division_requirement * SIGNAL_BOND_CONSTRUCTION_FRACTION)
}

/// Transactionally reserve backbone construction before a bond slot is
/// allocated. `None` means the complete physical bond operation must be
/// dropped; callers must not fall back to a mechanical-only bond.
pub fn reserve_signal_bond_construction(
    available_nutrients: f32,
    next_division_requirement: f32,
) -> Option<f32> {
    if !available_nutrients.is_finite()
        || !next_division_requirement.is_finite()
        || available_nutrients < 0.0
        || next_division_requirement < 0.0
    {
        return None;
    }
    let cost = signal_bond_construction_cost(next_division_requirement)?;
    (available_nutrients >= cost).then_some(available_nutrients - cost)
}
/// Evaluate a signal threshold consistently across preview and GPU paths.
///
/// A zero value represents no active signal. Normal gates require a positive
/// signal at or above the threshold. Inverted gates are active when that normal
/// condition is false: no signal, or a signal below the threshold.
#[inline]
pub fn signal_gate_active(signal_value: f32, threshold: f32, invert: bool) -> bool {
    let at_or_above = signal_value > 0.0 && signal_value >= threshold;
    if invert {
        !at_or_above
    } else {
        at_or_above
    }
}

#[inline]
pub fn listener_response_value(value: f32, mode: SignalResponseMode) -> f32 {
    mode.response_value(value)
}

#[inline]
pub fn listener_active(value: f32, threshold: f32, mode: SignalResponseMode, invert: bool) -> bool {
    signal_gate_active(listener_response_value(value, mode), threshold, invert)
}

/// Oculocyte sense type bitmask bits
pub const SENSE_CELL: u32 = 1 << 0; // bit 0
pub const SENSE_FOOD: u32 = 1 << 1; // bit 1
pub const SENSE_LIGHT: u32 = 1 << 2; // bit 2
pub const SENSE_WALL: u32 = 1 << 3; // bit 3 - world boundary sphere + cave solid voxels + water surface
pub const SENSE_SELF: u32 = 1 << 4; // bit 4
pub const SENSE_MOSSROCK: u32 = 1 << 5; // bit 5

/// Oculocyte cell type index
const OCULOCYTE_TYPE: i32 = 7;
/// Photocyte cell type index
const PHOTOCYTE_TYPE: i32 = 3;
/// Lipocyte cell type index
const LIPOCYTE_TYPE: i32 = 4;
/// Cognocyte cell type index
const COGNOCYTE_TYPE: i32 = 14;
/// Memorocyte cell type index
const MEMOROCYTE_TYPE: i32 = 15;
/// Vasculocyte cell type index

/// Clear all finalized signal channels.
pub fn clear_all_signals(state: &mut CanonicalState) {
    for channel in state.signal_channels.iter_mut() {
        *channel = None;
    }
    state.has_any_signal = false;
}

/// A pending signal emission from an oculocyte or test button.
#[derive(Clone)]
pub struct SignalEmission {
    /// Cell index of the emitter
    pub source_cell: usize,
    /// Channel to send on (0-15)
    pub channel: usize,
    /// Signal value to send
    pub value: f32,
}

/// Run oculocyte sensing for all oculocyte cells.
/// Returns a list of signal emissions that need to be propagated.
pub fn sense_oculocytes(
    state: &CanonicalState,
    genome: &Genome,
    boundary_radius: f32,
) -> Vec<SignalEmission> {
    let mut emissions = Vec::new();

    for cell_idx in 0..state.cell_count {
        let mode_idx = state.mode_indices[cell_idx];
        let mode = match genome.modes.get(mode_idx) {
            Some(m) => m,
            None => continue,
        };

        // Only oculocytes sense
        if mode.cell_type != OCULOCYTE_TYPE {
            continue;
        }

        let sense_mask = mode.oculocyte_sense_type;
        let channel = mode.oculocyte_signal_channel.clamp(0, 7) as usize; // Sensory channels 0-7
        let signal_value = mode.oculocyte_signal_value.clamp(SIGNAL_MIN, SIGNAL_MAX);
        let ray_length = mode.oculocyte_ray_length.clamp(1.0, 100.0);

        // Forward direction from genome orientation
        let forward = state.genome_orientations[cell_idx] * Vec3::Z;
        let pos = state.positions[cell_idx];

        // Bitmask: detect if ANY of the enabled sense types fires.
        // Each bit is checked independently; the cell emits if at least one hits.
        let detected = ((sense_mask & SENSE_SELF) != 0)  // Self always fires
            || ((sense_mask & SENSE_CELL) != 0 && sense_cells_ray(state, cell_idx, pos, forward, ray_length))
            || ((sense_mask & SENSE_WALL) != 0 && sense_barrier_ray(pos, forward, ray_length, boundary_radius))
            // Food and Light require fluid/light systems - not available in preview
            || (sense_mask & SENSE_FOOD) != 0 && false
            || (sense_mask & SENSE_LIGHT) != 0 && false;

        if detected {
            emissions.push(SignalEmission {
                source_cell: cell_idx,
                channel,
                value: signal_value,
            });
        }
    }

    emissions
}

/// Sense other cells along the forward ray.
/// Tests each cell as a sphere against the ray; exits early on first hit.
fn sense_cells_ray(
    state: &CanonicalState,
    self_idx: usize,
    pos: Vec3,
    forward: Vec3,
    ray_length: f32,
) -> bool {
    for other_idx in 0..state.cell_count {
        if other_idx == self_idx {
            continue;
        }

        let other_pos = state.positions[other_idx];
        let radius = state.radii[other_idx];

        // Ray-sphere intersection: ray origin=pos, dir=forward (normalized)
        // Sphere center=other_pos, radius=radius
        let oc = other_pos - pos;
        let tca = oc.dot(forward);
        if tca < 0.0 || tca > ray_length {
            continue;
        }
        let dist_sq = oc.length_squared() - tca * tca;
        if dist_sq <= radius * radius {
            return true;
        }
    }

    false
}

/// Sense barrier/world boundary along the forward ray.
/// Ray-sphere intersection against the world boundary sphere.
fn sense_barrier_ray(pos: Vec3, forward: Vec3, ray_length: f32, boundary_radius: f32) -> bool {
    // For a sphere centered at origin with radius R:
    // |pos + t*forward|^2 = R^2
    // t^2 + 2*(posforward)*t + (|pos|^2 - R^2) = 0
    let b = 2.0 * pos.dot(forward);
    let c = pos.length_squared() - boundary_radius * boundary_radius;

    let discriminant = b * b - 4.0 * c;
    if discriminant < 0.0 {
        return false;
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) * 0.5;
    let t2 = (-b + sqrt_d) * 0.5;

    // We want t > 0 (ahead of us) and t <= ray_length
    let t = if t1 > 0.0 { t1 } else { t2 };
    t > 0.0 && t <= ray_length
}

/// Read a single signal channel value for a specific cell.
/// Returns `None` if the channel has no signal.
#[cfg(test)]
fn read_channel(state: &CanonicalState, cell_idx: usize, channel: usize) -> Option<f32> {
    let idx = cell_idx * SIGNAL_CHANNELS + channel;
    if idx < state.signal_channels.len() {
        state.signal_channels[idx]
    } else {
        None
    }
}

/// Emit signals from Photocyte cells.
///
/// In the preview scene there is no light field, so photocytes emit unconditionally
/// whenever their output channel is enabled. In the GPU scene the actual light check
/// is handled by the photocyte_light shader; this path only applies to the CPU preview.
pub fn process_photocytes(state: &CanonicalState, genome: &Genome) -> Vec<SignalEmission> {
    let mut emissions = Vec::new();

    for cell_idx in 0..state.cell_count {
        let mode_idx = state.mode_indices[cell_idx];
        let mode = match genome.modes.get(mode_idx) {
            Some(m) => m,
            None => continue,
        };

        if mode.cell_type != PHOTOCYTE_TYPE {
            continue;
        }
        if !mode.photocyte_emit_enabled {
            continue;
        }
        let sampled_light = state
            .signal_light_samples
            .get(cell_idx)
            .copied()
            .unwrap_or(0.0);
        let above = sampled_light >= mode.photocyte_emit_threshold;
        if (mode.photocyte_emit_mode == 1) == above {
            continue;
        }

        let ch = mode.photocyte_emit_channel.clamp(0, 15) as usize;

        emissions.push(SignalEmission {
            source_cell: cell_idx,
            channel: ch,
            value: mode.photocyte_emit_value,
        });
    }

    emissions
}

/// Emit signals from Lipocyte cells based on their storage level vs threshold.
///
/// Lipocytes store up to 200 nutrients. The storage fraction (0.0-1.0) is compared
/// against `lipocyte_emit_threshold`. emit_mode 0 = emit when above, 1 = emit when below.
pub fn process_lipocytes(state: &CanonicalState, genome: &Genome) -> Vec<SignalEmission> {
    let mut emissions = Vec::new();

    for cell_idx in 0..state.cell_count {
        let mode_idx = state.mode_indices[cell_idx];
        let mode = match genome.modes.get(mode_idx) {
            Some(m) => m,
            None => continue,
        };

        if mode.cell_type != LIPOCYTE_TYPE {
            continue;
        }
        if !mode.lipocyte_emit_enabled {
            continue;
        }

        let nutrients = state.nutrients.get(cell_idx).copied().unwrap_or(0.0);
        let fraction = (nutrients / 200.0).clamp(0.0, 1.0);
        let threshold = mode.lipocyte_emit_threshold.clamp(0.0, 1.0);
        let above = fraction >= threshold;
        let should_emit = if mode.lipocyte_emit_mode == 1 {
            !above
        } else {
            above
        };

        if !should_emit {
            continue;
        }

        let ch = mode.lipocyte_emit_channel.clamp(0, 15) as usize;

        emissions.push(SignalEmission {
            source_cell: cell_idx,
            channel: ch,
            value: mode.lipocyte_emit_value,
        });
    }

    emissions
}

/// Advance fixed-rate production, diffusion, decay, and explicit processor memory.
pub fn run_signal_system(
    state: &mut CanonicalState,
    genome: &Genome,
    boundary_radius: f32,
    dt: f32,
    current_time: f32,
    manual_emissions: Option<&[SignalEmission]>,
) {
    state.signal_tick_accumulator = (state.signal_tick_accumulator + dt.max(0.0))
        .min(SIGNAL_TICK_SECONDS * MAX_SIGNAL_CATCH_UP_TICKS as f32);
    let mut ticks = 0;
    while state.signal_tick_accumulator + f32::EPSILON >= SIGNAL_TICK_SECONDS
        && ticks < MAX_SIGNAL_CATCH_UP_TICKS
    {
        state.signal_tick_accumulator -= SIGNAL_TICK_SECONDS;
        state.signal_tick_index = state.signal_tick_index.wrapping_add(1);
        run_authoritative_signal_tick(
            state,
            genome,
            boundary_radius,
            current_time,
            manual_emissions.unwrap_or(&[]),
        );
        ticks += 1;
    }
}

fn processor_config(mode: &crate::genome::ModeSettings) -> u64 {
    let mut hash = mode.cell_type as u64;
    for value in [
        mode.cognocyte_operation,
        mode.cognocyte_input_channel_a,
        mode.cognocyte_input_channel_b,
        mode.cognocyte_output_channel,
        mode.memorocyte_input_channel,
        mode.memorocyte_output_channel,
    ] {
        hash = hash.rotate_left(9) ^ value as u64;
    }
    hash ^= (mode.memorocyte_rate.to_bits() as u64) << 17;
    hash ^= (mode.cognocyte_oscillator_rate.to_bits() as u64).rotate_left(7);
    hash ^= (mode.cognocyte_oscillator_phase.to_bits() as u64).rotate_left(19);
    hash ^= (mode.cognocyte_oscillator_strength.to_bits() as u64).rotate_left(31);
    hash ^= (mode.cognocyte_oscillator_polarity as u64).rotate_left(43);
    hash.max(1)
}

pub fn reset_processor_state(state: &mut CanonicalState, cell: usize) {
    if cell < state.capacity {
        state.memo_state[cell] = 0.0;
        state.signal_processor_output[cell] = 0.0;
        state.signal_processor_channel[cell] = 0;
        state.signal_processor_config[cell] = 0;
    }
}

/// New compartments begin empty; separate from mode-change processor reset.
pub fn reset_cell_signal_state(state: &mut CanonicalState, cell: usize) {
    state.signal_concentrations[cell] = [0.0; SIGNAL_CHANNELS];
    state.signal_channels[cell * SIGNAL_CHANNELS..(cell + 1) * SIGNAL_CHANNELS].fill(None);
    reset_processor_state(state, cell);
}

pub(crate) fn deterministic_heat_value(cell_id: u32, channel: usize, tick: u64) -> f32 {
    // Keep this integer sequence byte-for-byte equivalent to the WGSL heat
    // hash. The signal tick intentionally wraps to u32 on both paths.
    let mut hash = cell_id
        ^ (channel as u32).wrapping_mul(0x9e37_79b9)
        ^ (tick as u32).wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb_352d);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846c_a68b);
    hash ^= hash >> 16;
    if hash & 1 == 0 {
        -SIGNAL_MAX
    } else {
        SIGNAL_MAX
    }
}

#[inline]
fn oscillator_polarity(value_01: f32, peak: f32, polarity: i32) -> f32 {
    let magnitude = peak.abs().clamp(0.0, SIGNAL_MAX);
    match polarity {
        1 => -value_01.clamp(0.0, 1.0) * magnitude,
        2 => (value_01.clamp(0.0, 1.0) * 2.0 - 1.0) * magnitude,
        _ => value_01.clamp(0.0, 1.0) * magnitude,
    }
}

fn emission_cost(value: f32) -> f32 {
    REFERENCE_BASELINE_MAINTENANCE_PER_SECOND * 0.25 * value.abs().min(SIGNAL_MAX) / SIGNAL_MAX
        * SIGNAL_TICK_SECONDS
}

fn run_authoritative_signal_tick(
    state: &mut CanonicalState,
    genome: &Genome,
    boundary_radius: f32,
    _current_time: f32,
    manual_emissions: &[SignalEmission],
) {
    use crate::cell::behaviors::cognocyte::{evaluate, OP_NOT, OP_OSCILLATE, OP_WAVE_OSCILLATE};

    let count = state.cell_count;
    let mut requested = vec![[0.0f32; SIGNAL_CHANNELS]; count];
    let mut ordinary_cost = vec![0.0f32; count];
    let mut heat = vec![[0.0f32; SIGNAL_CHANNELS]; count];
    let mut ordinary = sense_oculocytes(state, genome, boundary_radius);
    ordinary.extend(emit_regulation_signals(state, genome));
    ordinary.extend(process_photocytes(state, genome));
    ordinary.extend(process_lipocytes(state, genome));
    ordinary.extend_from_slice(manual_emissions);

    for cell in 0..count {
        let Some(mode) = genome.modes.get(state.mode_indices[cell]) else {
            continue;
        };
        let config = processor_config(mode);
        if state.signal_processor_config[cell] != config {
            reset_processor_state(state, cell);
            state.signal_processor_config[cell] = config;
        }
        if matches!(mode.cell_type, COGNOCYTE_TYPE | MEMOROCYTE_TYPE) {
            let channel = state.signal_processor_channel[cell] as usize;
            let value = state.signal_processor_output[cell].clamp(SIGNAL_MIN, SIGNAL_MAX);
            requested[cell][channel] += value;
            ordinary_cost[cell] += emission_cost(value) * state.signal_diffusion.production_scale;
        }
        if state.cell_thermal_state[cell] == 9 {
            for channel in 0..SIGNAL_CHANNELS {
                heat[cell][channel] = deterministic_heat_value(
                    state.cell_ids[cell],
                    channel,
                    state.signal_tick_index,
                )
                .abs();
            }
        }
    }
    for emission in ordinary {
        if emission.source_cell < count && emission.channel < SIGNAL_CHANNELS {
            let value = emission.value.clamp(SIGNAL_MIN, SIGNAL_MAX);
            requested[emission.source_cell][emission.channel] += value;
            ordinary_cost[emission.source_cell] +=
                emission_cost(value) * state.signal_diffusion.production_scale;
        }
    }

    for cell in 0..count {
        let heat_cost: f32 = heat[cell]
            .iter()
            .map(|&value| emission_cost(value) * state.signal_diffusion.production_scale)
            .sum();
        let available = state.nutrients[cell].max(0.0);
        let paid_heat = available.min(heat_cost);
        state.nutrients[cell] -= paid_heat;
        let total_cost = ordinary_cost[cell];
        let heat_screaming = heat[cell].iter().any(|&value| value != 0.0);
        let funding = if heat_screaming {
            0.0
        } else if total_cost > 0.0 {
            (state.nutrients[cell].max(0.0) / total_cost).clamp(0.0, 1.0)
        } else {
            1.0
        };
        state.nutrients[cell] -= total_cost * funding;
        for channel in 0..SIGNAL_CHANNELS {
            requested[cell][channel] *= funding;
            requested[cell][channel] += heat[cell][channel];
        }
    }

    use crate::cell::adhesion::{BOND_FLAG_BARRIER_BALL, BOND_FLAG_SIGNAL_ACTIVE};
    let mut edges = Vec::new();
    let connections = &mut state.adhesion_connections;
    for edge in 0..connections.active_count {
        let a = connections.cell_a_index[edge];
        let b = connections.cell_b_index[edge];
        let eligible = connections.is_active[edge] != 0
            && connections.bond_flags[edge] & BOND_FLAG_BARRIER_BALL == 0
            && a < count
            && b < count
            && a != b;
        connections.bond_flags[edge] &= !BOND_FLAG_SIGNAL_ACTIVE;
        if eligible {
            connections.bond_flags[edge] |= BOND_FLAG_SIGNAL_ACTIVE;
            edges.push((a, b));
        }
    }
    let next = crate::simulation::signal_diffusion::step(
        &state.signal_concentrations[..count],
        &requested,
        &edges,
        state.signal_diffusion,
        SIGNAL_TICK_SECONDS,
    )
    .expect("validated diffusion settings and cell adjacency");
    state.signal_concentrations[..count].copy_from_slice(&next);
    clear_all_signals(state);
    for cell in 0..count {
        for channel in 0..SIGNAL_CHANNELS {
            let value = next[cell][channel].min(SIGNAL_MAX).round_ties_even();
            if value > 0.0 {
                state.signal_channels[cell * SIGNAL_CHANNELS + channel] = Some(value);
                state.has_any_signal = true;
            }
        }
    }

    // Match GPU processors: cap full-precision input independently of packed receivers.
    let immutable_field: Vec<f32> = next
        .iter()
        .flatten()
        .map(|value| value.min(SIGNAL_MAX))
        .collect();
    let mut next_output = vec![0.0f32; count];
    let mut next_channel = vec![0u8; count];
    let signal_time = state.signal_tick_index as f32 * SIGNAL_TICK_SECONDS;
    for cell in 0..count {
        let Some(mode) = genome.modes.get(state.mode_indices[cell]) else {
            continue;
        };
        if mode.cell_type == COGNOCYTE_TYPE {
            let op = mode.cognocyte_operation;
            let a = immutable_field
                [cell * SIGNAL_CHANNELS + mode.cognocyte_input_channel_a.clamp(0, 15) as usize];
            let b = immutable_field
                [cell * SIGNAL_CHANNELS + mode.cognocyte_input_channel_b.clamp(0, 15) as usize];
            let result = if op == OP_OSCILLATE {
                let phase =
                    mode.cognocyte_oscillator_rate * signal_time + mode.cognocyte_oscillator_phase;
                let sine = (phase * std::f32::consts::TAU).sin();
                let normalized = if mode.cognocyte_oscillator_polarity == 2 {
                    sine * 0.5 + 0.5
                } else {
                    sine.max(0.0)
                };
                oscillator_polarity(
                    normalized,
                    mode.cognocyte_oscillator_strength,
                    mode.cognocyte_oscillator_polarity,
                )
            } else if op == OP_WAVE_OSCILLATE {
                let phase = (mode.cognocyte_oscillator_rate * signal_time
                    + mode.cognocyte_oscillator_phase)
                    .rem_euclid(1.0);
                oscillator_polarity(
                    phase,
                    mode.cognocyte_oscillator_strength,
                    mode.cognocyte_oscillator_polarity,
                )
            } else if matches!(op, OP_NOT | 16..=19) || a != 0.0 && b != 0.0 {
                evaluate(op, a, b)
            } else {
                0.0
            };
            next_output[cell] = if result.is_finite() {
                result.clamp(SIGNAL_MIN, SIGNAL_MAX)
            } else {
                state.signal_invalid_processor_outputs =
                    state.signal_invalid_processor_outputs.saturating_add(1);
                0.0
            };
            next_channel[cell] = mode.cognocyte_output_channel.clamp(0, 15) as u8;
        } else if mode.cell_type == MEMOROCYTE_TYPE {
            let input = immutable_field
                [cell * SIGNAL_CHANNELS + mode.memorocyte_input_channel.clamp(0, 15) as usize];
            let rate = mode.memorocyte_rate.clamp(0.0, 1.0);
            let effective_rate = 1.0 - (1.0 - rate).powf(SIGNAL_TICK_SECONDS);
            state.memo_state[cell] = (state.memo_state[cell]
                + (input - state.memo_state[cell]) * effective_rate)
                .clamp(SIGNAL_MIN, SIGNAL_MAX);
            next_output[cell] = state.memo_state[cell];
            next_channel[cell] = mode.memorocyte_output_channel.clamp(0, 15) as u8;
        }
    }
    state.signal_processor_output[..count].copy_from_slice(&next_output);
    state.signal_processor_channel[..count].copy_from_slice(&next_channel);
}

/// Emit regulation signals for all cells whose mode has regulation_emit_channel >= 8.
/// These are unconditional emissions - any cell type can emit on regulation channels.
pub fn emit_regulation_signals(state: &CanonicalState, genome: &Genome) -> Vec<SignalEmission> {
    let mut emissions = Vec::new();

    for cell_idx in 0..state.cell_count {
        let mode_idx = state.mode_indices[cell_idx];
        let mode = match genome.modes.get(mode_idx) {
            Some(m) => m,
            None => continue,
        };

        // Only emit if regulation channel is enabled (8-15)
        if mode.regulation_emit_channel < 8 || mode.regulation_emit_channel > 15 {
            continue;
        }

        let channel = mode.regulation_emit_channel as usize;
        let value = mode.regulation_emit_value.clamp(SIGNAL_MIN, SIGNAL_MAX);

        if value != 0.0 {
            emissions.push(SignalEmission {
                source_cell: cell_idx,
                channel,
                value,
            });
        }
    }

    emissions
}

#[cfg(test)]
mod signal_gate_tests {
    use super::*;
    use crate::genome::Genome;
    use glam::{Quat, Vec3};

    #[test]
    fn phase4_backbone_construction_is_transactional_and_never_degrades_to_mechanical() {
        assert_eq!(reserve_signal_bond_construction(10.0, 100.0), Some(5.0));
        assert_eq!(reserve_signal_bond_construction(5.0, 100.0), Some(0.0));
        assert_eq!(reserve_signal_bond_construction(4.999, 100.0), None);
        assert_eq!(reserve_signal_bond_construction(f32::NAN, 100.0), None);
    }

    #[test]
    fn phase4_existing_backbone_has_no_continuous_nutrient_cost() {
        let genome = Genome::default();
        let mut state = state_with_cells(2);
        signal_bond(&mut state, 0, 1);
        let before = state.nutrients.clone();

        for tick in 0..30 {
            run_signal_system(
                &mut state,
                &genome,
                200.0,
                SIGNAL_TICK_SECONDS,
                tick as f32 * SIGNAL_TICK_SECONDS,
                None,
            );
        }

        assert_eq!(state.nutrients, before);
    }

    fn state_with_cells(count: usize) -> CanonicalState {
        let mut state = CanonicalState::new(count.max(4));
        for cell in 0..count {
            state
                .add_cell(
                    Vec3::new(cell as f32, 0.0, 0.0),
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Quat::IDENTITY,
                    Vec3::ZERO,
                    100.0,
                    0,
                    0,
                    0.0,
                    1.0,
                    200.0,
                    1.0,
                )
                .unwrap();
        }
        state
    }

    fn signal_bond(state: &mut CanonicalState, a: usize, b: usize) -> usize {
        state
            .adhesion_manager
            .add_ball_joint(&mut state.adhesion_connections, a, b, 0, 0.0, 0)
            .unwrap()
    }

    fn manual(cell: usize, channel: usize, value: f32) -> SignalEmission {
        SignalEmission {
            source_cell: cell,
            channel,
            value,
        }
    }

    #[test]
    fn normal_gate_requires_a_present_signal() {
        assert!(!signal_gate_active(0.0, 0.0, false));
        assert!(!signal_gate_active(0.0, 1.0, false));
        assert!(!signal_gate_active(0.5, 1.0, false));
        assert!(signal_gate_active(1.0, 1.0, false));
    }

    #[test]
    fn inverted_gate_handles_absence_and_below_threshold() {
        assert!(signal_gate_active(0.0, 0.0, true));
        assert!(signal_gate_active(0.0, 1.0, true));
        assert!(signal_gate_active(0.5, 1.0, true));
        assert!(!signal_gate_active(1.0, 1.0, true));
    }

    #[test]
    fn signed_listener_modes_and_inversion_are_exhaustive() {
        use crate::genome::SignalResponseMode::{Magnitude, Negative, Positive};

        for (value, positive, negative, magnitude) in [
            (-500.0, false, true, true),
            (-399.0, false, false, false),
            (0.0, false, false, false),
            (399.0, false, false, false),
            (500.0, true, false, true),
        ] {
            assert_eq!(listener_active(value, 400.0, Positive, false), positive);
            assert_eq!(listener_active(value, 400.0, Negative, false), negative);
            assert_eq!(listener_active(value, 400.0, Magnitude, false), magnitude);
            assert_eq!(listener_active(value, 400.0, Positive, true), !positive);
            assert_eq!(listener_active(value, 400.0, Negative, true), !negative);
            assert_eq!(listener_active(value, 400.0, Magnitude, true), !magnitude);
        }

        assert!(!listener_active(0.0, 0.0, Magnitude, false));
        assert!(listener_active(0.0, 0.0, Magnitude, true));
    }

    #[test]
    fn diffusion_preview_is_delayed_additive_and_survives_topology_changes() {
        let genome = Genome::default();
        let mut state = state_with_cells(3);
        let first = signal_bond(&mut state, 0, 1);
        signal_bond(&mut state, 1, 2);
        state.signal_diffusion.decay_rate = 0.0;
        let source = [manual(0, 0, 300.0)];
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            0.0,
            Some(&source),
        );
        assert!((read_channel(&state, 0, 0).unwrap() - 20.0).abs() < 0.00001);
        assert_eq!(read_channel(&state, 1, 0), None);
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            0.0,
            Some(&source),
        );
        assert!(read_channel(&state, 0, 0).unwrap() > 20.0);
        assert!(read_channel(&state, 1, 0).unwrap() > 0.0);
        assert_eq!(read_channel(&state, 2, 0), None);
        state.adhesion_connections.is_active[first] = 0;
        let retained = state.signal_concentrations[0][0];
        run_signal_system(&mut state, &genome, 200.0, SIGNAL_TICK_SECONDS, 0.0, None);
        assert_eq!(state.signal_concentrations[0][0], retained);
        assert!(state.signal_concentrations[2][0] > 0.0);
    }

    #[test]
    fn diffusion_receiver_saturation_does_not_destroy_quantity_or_emit_on_receipt() {
        let genome = Genome::default();
        let mut state = state_with_cells(2);
        signal_bond(&mut state, 0, 1);
        state.signal_diffusion.decay_rate = 0.0;
        state.signal_concentrations[0][0] = 5000.0;
        run_signal_system(&mut state, &genome, 200.0, SIGNAL_TICK_SECONDS, 0.0, None);
        assert_eq!(read_channel(&state, 0, 0), Some(1000.0));
        assert!(
            (state
                .signal_concentrations
                .iter()
                .map(|v| v[0])
                .sum::<f32>()
                - 5000.0)
                .abs()
                < 0.001
        );
        state.remove_cell(0);
        let carried = state.signal_concentrations[0][0];
        assert!(carried > 0.0 && carried < 1000.0);
    }
}
