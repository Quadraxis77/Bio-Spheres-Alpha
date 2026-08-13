//! Signal system for oculocyte sensing and inter-cell communication.
//!
//! Oculocytes sense targets (cells, food, light, barriers) along their forward direction
//! and inject signed sources into the cached signal backbone.
//!
//! Signal semantics:
//! - `None` = null (no signal on this channel)
//! - `Some(value)` = a finalized signed channel value
//! - Contributions accumulate completely before saturating to `-1000..1000`
//! - Signals update on the fixed 15 Hz signal clock

use crate::genome::{Genome, SignalResponseMode};
use crate::simulation::canonical_state::CanonicalState;
use glam::Vec3;

/// Number of signal channels (0-15)
pub const SIGNAL_CHANNELS: usize = 16;
pub const SIGNAL_MIN: f32 = -1000.0;
pub const SIGNAL_MAX: f32 = 1000.0;
pub const SIGNAL_TICK_HZ: f32 = 15.0;
pub const SIGNAL_TICK_SECONDS: f32 = 1.0 / SIGNAL_TICK_HZ;
pub const MAX_SIGNAL_CATCH_UP_TICKS: usize = 4;
pub const REFERENCE_BASELINE_MAINTENANCE_PER_SECOND: f32 = 1.0;
pub const BACKBONE_CONSTRUCTION_FRACTION: f32 = 0.05;

#[inline]
pub fn backbone_construction_cost(next_division_requirement: f32) -> Option<f32> {
    (next_division_requirement.is_finite() && next_division_requirement >= 0.0)
        .then_some(next_division_requirement * BACKBONE_CONSTRUCTION_FRACTION)
}

/// Transactionally reserve backbone construction before a bond slot is
/// allocated. `None` means the complete physical bond operation must be
/// dropped; callers must not fall back to a mechanical-only bond.
pub fn reserve_backbone_construction(
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
    let cost = backbone_construction_cost(next_division_requirement)?;
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
const VASCULOCYTE_TYPE: i32 = 12;

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
                                                                          // GPU signals use an unsigned 11-bit payload, so authored sensor signals
                                                                          // share the same positive 1..2047 range in both scenes.
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

fn is_signal_transport_vascular(genome: &Genome, mode_idx: usize) -> bool {
    genome
        .modes
        .get(mode_idx)
        .is_some_and(|mode| mode.cell_type == VASCULOCYTE_TYPE && mode.vascular_signal_transport)
}

fn is_signal_exchange_vascular(genome: &Genome, mode_idx: usize) -> bool {
    genome
        .modes
        .get(mode_idx)
        .is_some_and(|mode| mode.cell_type == VASCULOCYTE_TYPE && mode.vascular_signal_exchange)
}

fn can_signal_cross(genome: &Genome, from_mode_idx: usize, to_mode_idx: usize) -> bool {
    let from_vascular = genome
        .modes
        .get(from_mode_idx)
        .is_some_and(|mode| mode.cell_type == VASCULOCYTE_TYPE);
    let to_vascular = genome
        .modes
        .get(to_mode_idx)
        .is_some_and(|mode| mode.cell_type == VASCULOCYTE_TYPE);

    match (from_vascular, to_vascular) {
        (true, true) => {
            is_signal_transport_vascular(genome, from_mode_idx)
                && is_signal_transport_vascular(genome, to_mode_idx)
        }
        (true, false) => is_signal_exchange_vascular(genome, from_mode_idx),
        (false, true) => is_signal_exchange_vascular(genome, to_mode_idx),
        (false, false) => true,
    }
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

/// Run the complete signal system for one frame:
/// 1. Clear all signals
/// 2. Run oculocyte sensing (channels 0-7) + regulation signals (channels 8-15)
///    + photocyte/lipocyte conditional emissions
/// 3. Propagate sensor/regulation signals
/// 4. Cognocytes compute on propagated signals and re-emit
/// 5. Memorocytes update leaky-integrator state and emit
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
        SIGNAL_MIN
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

fn build_cpu_forest(
    state: &mut CanonicalState,
    genome: &Genome,
    sources: Vec<[f32; SIGNAL_CHANNELS]>,
) -> crate::simulation::signal_backbone_bench::SyntheticForest {
    use crate::cell::adhesion::{BOND_FLAG_SIGNAL_ACTIVE, BOND_FLAG_SIGNAL_BACKBONE};
    use crate::simulation::signal_backbone_bench::{
        BondClass, Edge, EdgeClass, NodeRole, SyntheticForest,
    };

    let mut forest = SyntheticForest::new(state.cell_count);
    forest.sources = sources;
    for cell in 0..state.cell_count {
        forest.roles[cell] =
            genome
                .modes
                .get(state.mode_indices[cell])
                .map_or(NodeRole::Disabled, |mode| {
                    if mode.cell_type == OCULOCYTE_TYPE {
                        NodeRole::SourceOnly
                    } else {
                        NodeRole::Relay
                    }
                });
    }
    let connections = &state.adhesion_connections;
    let mut stable_bond_ids = Vec::new();
    let mut physical_edge_indices = Vec::new();
    for edge in 0..connections.active_count {
        if connections.is_active[edge] == 0
            || connections.bond_flags[edge] & BOND_FLAG_SIGNAL_BACKBONE == 0
        {
            continue;
        }
        let a = connections.cell_a_index[edge];
        let b = connections.cell_b_index[edge];
        if a >= state.cell_count || b >= state.cell_count {
            continue;
        }
        if !can_signal_cross(genome, state.mode_indices[a], state.mode_indices[b]) {
            continue;
        }
        let bond_class = match (forest.roles[a], forest.roles[b]) {
            (NodeRole::Relay, NodeRole::Relay) => BondClass::Backbone,
            (NodeRole::SourceOnly, NodeRole::Relay) | (NodeRole::Relay, NodeRole::SourceOnly) => {
                BondClass::SourceAttachment
            }
            _ => BondClass::MechanicalOnly,
        };
        let road = is_signal_transport_vascular(genome, state.mode_indices[a])
            && is_signal_transport_vascular(genome, state.mode_indices[b]);
        forest.edges.push(Edge {
            a: a as u32,
            b: b as u32,
            edge_class: if road {
                EdgeClass::VascularRoad
            } else {
                EdgeClass::Normal
            },
            bond_class,
            active: true,
        });
        stable_bond_ids.push(((connections.slot_generation[edge] as u64) << 32) | edge as u64);
        physical_edge_indices.push(edge);
    }
    match forest.select_active_routes_with_ids(&stable_bond_ids) {
        Ok(routed) => {
            for edge in 0..state.adhesion_connections.active_count {
                if state.adhesion_connections.bond_flags[edge] & BOND_FLAG_SIGNAL_BACKBONE != 0 {
                    state.adhesion_connections.bond_flags[edge] &= !BOND_FLAG_SIGNAL_ACTIVE;
                }
            }
            for (routed_edge, &physical_edge) in
                routed.edges.iter().zip(physical_edge_indices.iter())
            {
                if routed_edge.active {
                    state.adhesion_connections.bond_flags[physical_edge] |= BOND_FLAG_SIGNAL_ACTIVE;
                }
            }
            routed
        }
        Err(error) => {
            log::error!("failed to select CPU signal routes: {error:?}");
            SyntheticForest::new(state.cell_count)
        }
    }
}

fn topology_signature(forest: &crate::simulation::signal_backbone_bench::SyntheticForest) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    mix(forest.roles.len() as u64);
    for role in &forest.roles {
        mix(*role as u64);
    }
    for edge in &forest.edges {
        mix(edge.a as u64);
        mix(edge.b as u64);
        mix(edge.edge_class as u64);
        mix(edge.bond_class as u64);
        mix(edge.active as u64);
    }
    hash.max(1)
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
            ordinary_cost[cell] += emission_cost(value);
        }
        if state.cell_thermal_state[cell] == 9 {
            for channel in 0..SIGNAL_CHANNELS {
                heat[cell][channel] = deterministic_heat_value(
                    state.cell_ids[cell],
                    channel,
                    state.signal_tick_index,
                );
            }
        }
    }
    for emission in ordinary {
        if emission.source_cell < count && emission.channel < SIGNAL_CHANNELS {
            let value = emission.value.clamp(SIGNAL_MIN, SIGNAL_MAX);
            requested[emission.source_cell][emission.channel] += value;
            ordinary_cost[emission.source_cell] += emission_cost(value);
        }
    }

    for cell in 0..count {
        let heat_cost: f32 = heat[cell].iter().map(|&value| emission_cost(value)).sum();
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

    let forest = build_cpu_forest(state, genome, requested);
    let signature = topology_signature(&forest);
    if state.signal_cached_forest.is_none() || state.signal_topology_signature != signature {
        match forest.cache() {
            Ok(cache) => {
                state.signal_cached_forest = Some(cache);
                state.signal_topology_signature = signature;
            }
            Err(error) => {
                log::error!("invalid explicit CPU signal backbone: {error:?}");
                state.signal_cached_forest = None;
                state.signal_topology_signature = 0;
                clear_all_signals(state);
                return;
            }
        }
    }
    let field = match state
        .signal_cached_forest
        .as_ref()
        .expect("cache assigned above")
        .propagate(&forest.sources)
    {
        Ok(field) => field,
        Err(error) => {
            log::error!("invalid explicit CPU signal backbone: {error:?}");
            clear_all_signals(state);
            return;
        }
    };
    clear_all_signals(state);
    for cell in 0..count {
        for channel in 0..SIGNAL_CHANNELS {
            let value = (field[cell][channel] + heat[cell][channel]).clamp(SIGNAL_MIN, SIGNAL_MAX);
            if value != 0.0 {
                state.signal_channels[cell * SIGNAL_CHANNELS + channel] = Some(value);
                state.has_any_signal = true;
            }
        }
    }

    let immutable_field: Vec<f32> = state.signal_channels[..count * SIGNAL_CHANNELS]
        .iter()
        .map(|value| value.unwrap_or(0.0))
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
    use crate::cell::adhesion::BOND_FLAG_SIGNAL_BACKBONE;
    use crate::genome::Genome;
    use glam::{Quat, Vec3};

    #[test]
    fn phase4_backbone_construction_is_transactional_and_never_degrades_to_mechanical() {
        assert_eq!(reserve_backbone_construction(10.0, 100.0), Some(5.0));
        assert_eq!(reserve_backbone_construction(5.0, 100.0), Some(0.0));
        assert_eq!(reserve_backbone_construction(4.999, 100.0), None);
        assert_eq!(reserve_backbone_construction(f32::NAN, 100.0), None);
    }

    #[test]
    fn phase4_existing_backbone_has_no_continuous_nutrient_cost() {
        let genome = Genome::default();
        let mut state = state_with_cells(2);
        backbone(&mut state, 0, 1);
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

    fn backbone(state: &mut CanonicalState, a: usize, b: usize) -> usize {
        state
            .adhesion_manager
            .add_ball_joint(
                &mut state.adhesion_connections,
                a,
                b,
                0,
                0.0,
                BOND_FLAG_SIGNAL_BACKBONE,
            )
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
    fn phase2_fixed_clock_tree_and_economics_contract() {
        let genome = Genome::default();
        let mut state = state_with_cells(3);
        backbone(&mut state, 0, 1);
        backbone(&mut state, 1, 2);
        let source = [manual(0, 0, -1000.0)];

        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS * 0.5,
            0.0,
            Some(&source),
        );
        assert_eq!(
            read_channel(&state, 1, 0),
            None,
            "no render-frame-defined early tick"
        );
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS * 0.5,
            1.0,
            Some(&source),
        );
        assert!((read_channel(&state, 1, 0).unwrap() + 950.0).abs() < 1e-3);
        assert!((read_channel(&state, 2, 0).unwrap() + 902.5).abs() < 1e-3);
        assert_eq!(
            read_channel(&state, 0, 0),
            None,
            "normal source cannot receive itself"
        );

        let retained = state.signal_channels.clone();
        run_signal_system(&mut state, &genome, 200.0, 0.0, 2.0, None);
        assert_eq!(
            state.signal_channels, retained,
            "published field persists between ticks"
        );

        let mut brownout = state_with_cells(2);
        backbone(&mut brownout, 0, 1);
        let full_cost = emission_cost(1000.0);
        brownout.nutrients[0] = full_cost * 0.5;
        run_signal_system(
            &mut brownout,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            0.0,
            Some(&[manual(0, 0, 1000.0)]),
        );
        assert!((read_channel(&brownout, 1, 0).unwrap() - 475.0).abs() < 1e-3);
        assert_eq!(brownout.nutrients[0], 0.0);
    }

    #[test]
    fn phase2_signed_fan_in_vascular_and_explicit_backbone_contract() {
        let mut genome = Genome::default();
        let mut state = state_with_cells(3);
        state
            .adhesion_manager
            .add_ball_joint(&mut state.adhesion_connections, 0, 1, 0, 0.0, 0)
            .unwrap();
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            0.0,
            Some(&[manual(0, 0, 1000.0)]),
        );
        assert_eq!(
            read_channel(&state, 1, 0),
            None,
            "mechanical-only bond is ignored"
        );

        // Repair is a newly created classified bond; the mechanical-only bond
        // above is never promoted.
        let repaired = backbone(&mut state, 0, 1);
        backbone(&mut state, 2, 1);
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            1.0,
            Some(&[manual(0, 0, 1000.0), manual(2, 0, -1000.0)]),
        );
        assert_eq!(
            read_channel(&state, 1, 0),
            None,
            "opposite signs cancel before clamp"
        );

        state.adhesion_connections.is_active[repaired] = 0;
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            1.5,
            Some(&[manual(0, 0, 1000.0)]),
        );
        assert_eq!(
            read_channel(&state, 1, 0),
            None,
            "break masks transport at the next tick"
        );
        backbone(&mut state, 0, 1);

        genome.modes[0].cell_type = VASCULOCYTE_TYPE;
        genome.modes[0].vascular_signal_transport = true;
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            2.0,
            Some(&[manual(0, 1, 1000.0)]),
        );
        assert!((read_channel(&state, 1, 1).unwrap() - 987.5).abs() < 1e-3);
    }

    #[test]
    fn phase2_processors_heat_listeners_and_lifecycle_contract() {
        let mut genome = Genome::default();
        genome.modes[1] = genome.modes[0].clone();
        genome.modes[1].cell_type = COGNOCYTE_TYPE;
        genome.modes[1].cognocyte_operation = crate::cell::behaviors::cognocyte::OP_ADD;
        genome.modes[1].cognocyte_input_channel_a = 0;
        genome.modes[1].cognocyte_input_channel_b = 1;
        genome.modes[1].cognocyte_output_channel = 2;

        let mut state = state_with_cells(3);
        state.mode_indices[1] = 1;
        backbone(&mut state, 0, 1);
        backbone(&mut state, 1, 2);
        let inputs = [manual(0, 0, 100.0), manual(0, 1, 100.0)];
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            0.0,
            Some(&inputs),
        );
        assert_eq!(
            read_channel(&state, 2, 2),
            None,
            "processor result is not visible in tick t"
        );
        assert!((state.signal_processor_output[1] - 190.0).abs() < 1e-3);
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            1.0,
            Some(&inputs),
        );
        assert!((read_channel(&state, 2, 2).unwrap() - 180.5).abs() < 1e-3);

        reset_processor_state(&mut state, 1);
        assert_eq!(state.signal_processor_output[1], 0.0);
        assert_eq!(state.memo_state[1], 0.0);

        state.cell_thermal_state[2] = 9;
        state.nutrients[2] = 10.0;
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            2.0,
            Some(&[manual(2, 15, 500.0)]),
        );
        let expected_heat_cost = SIGNAL_CHANNELS as f32 * emission_cost(1000.0);
        assert!((state.nutrients[2] - (10.0 - expected_heat_cost)).abs() < 1e-5);
        for channel in 0..SIGNAL_CHANNELS {
            assert_eq!(read_channel(&state, 2, channel).unwrap().abs(), 1000.0);
        }

        assert!(listener_active(
            -500.0,
            400.0,
            SignalResponseMode::Negative,
            false
        ));
        assert!(!listener_active(
            -500.0,
            400.0,
            SignalResponseMode::Positive,
            false
        ));
        assert!(listener_active(
            -500.0,
            400.0,
            SignalResponseMode::Magnitude,
            false
        ));

        state.signal_processor_output[2] = 777.0;
        state.signal_processor_config[2] = 99;
        state.remove_cell(1);
        assert_eq!(
            state.signal_processor_output[1], 777.0,
            "swap-remove carries the live cell state"
        );
        let reused = state
            .add_cell(
                Vec3::ZERO,
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
        assert_eq!(
            state.signal_processor_output[reused], 0.0,
            "new slot is zero initialized"
        );
    }

    #[test]
    fn phase2_memorocyte_source_attachment_and_catch_up_contract() {
        let mut genome = Genome::default();
        genome.modes[1] = genome.modes[0].clone();
        genome.modes[1].cell_type = MEMOROCYTE_TYPE;
        genome.modes[1].memorocyte_input_channel = 0;
        genome.modes[1].memorocyte_output_channel = 3;
        genome.modes[1].memorocyte_rate = 0.5;
        genome.modes[2] = genome.modes[0].clone();
        genome.modes[2].cell_type = OCULOCYTE_TYPE;
        genome.modes[2].oculocyte_sense_type = SENSE_SELF;
        genome.modes[2].oculocyte_signal_channel = 4;
        genome.modes[2].oculocyte_signal_value = 100.0;

        let mut state = state_with_cells(4);
        state.mode_indices[1] = 1;
        state.mode_indices[2] = 2;
        state.organism_ids[3] = 99;
        backbone(&mut state, 0, 1);
        backbone(&mut state, 2, 0);
        backbone(&mut state, 2, 3);

        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS,
            0.0,
            Some(&[manual(0, 0, 1000.0), manual(0, 5, 250.0)]),
        );
        let expected_rate = 1.0 - 0.5f32.powf(SIGNAL_TICK_SECONDS);
        assert!((state.memo_state[1] - 950.0 * expected_rate).abs() < 1e-3);
        assert_eq!(
            read_channel(&state, 1, 3),
            None,
            "new memory waits one tick"
        );
        assert!((read_channel(&state, 0, 4).unwrap() - 95.0).abs() < 1e-3);
        assert!((read_channel(&state, 3, 4).unwrap() - 95.0).abs() < 1e-3);
        assert_eq!(
            read_channel(&state, 3, 5),
            None,
            "source-only attachment cannot relay"
        );

        let topology_signature = state.signal_topology_signature;
        genome.modes[2].oculocyte_signal_value = 200.0;
        run_signal_system(&mut state, &genome, 200.0, SIGNAL_TICK_SECONDS, 1.0, None);
        assert_eq!(state.signal_topology_signature, topology_signature);
        assert!((read_channel(&state, 3, 4).unwrap() - 190.0).abs() < 1e-3);

        let before = state.signal_tick_index;
        run_signal_system(
            &mut state,
            &genome,
            200.0,
            SIGNAL_TICK_SECONDS * 20.0,
            2.0,
            None,
        );
        assert_eq!(
            state.signal_tick_index - before,
            MAX_SIGNAL_CATCH_UP_TICKS as u64
        );
        assert!(state.signal_tick_accumulator < SIGNAL_TICK_SECONDS);

        state.signal_light_samples[0] = 0.75;
        genome.modes[0].cell_type = PHOTOCYTE_TYPE;
        genome.modes[0].photocyte_emit_enabled = true;
        genome.modes[0].photocyte_emit_threshold = 0.5;
        genome.modes[0].photocyte_emit_mode = 0;
        assert_eq!(process_photocytes(&state, &genome).len(), 1);

        assert_eq!(oscillator_polarity(0.25, 1000.0, 0), 250.0);
        assert_eq!(oscillator_polarity(0.25, 1000.0, 1), -250.0);
        assert_eq!(oscillator_polarity(0.25, 1000.0, 2), -500.0);
    }
}
