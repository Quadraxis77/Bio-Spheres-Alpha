//! Signal system for oculocyte sensing and inter-cell communication.
//!
//! Oculocytes sense targets (cells, food, light, barriers) along their forward direction
//! and send signals through adhesion connections via BFS propagation.
//!
//! Signal semantics:
//! - `None` = null (no signal on this channel)
//! - `Some(value)` = active signal (including `Some(0.0)` which is a valid signal)
//! - Multiple signals on the same channel are additive
//! - Signals persist only while being actively sent (cleared each frame)

use glam::Vec3;
use crate::genome::Genome;
use crate::simulation::canonical_state::CanonicalState;


/// Number of signal channels (0-15)
pub const SIGNAL_CHANNELS: usize = 16;
/// Signal attenuation factor per hop (signals lose this fraction per hop)
pub const SIGNAL_ATTENUATION_PER_HOP: f32 = 0.5; // 50% signal strength retained per hop

/// Oculocyte sense type bitmask bits
pub const SENSE_CELL: u32    = 1 << 0; // bit 0
pub const SENSE_FOOD: u32    = 1 << 1; // bit 1
pub const SENSE_LIGHT: u32   = 1 << 2; // bit 2
pub const SENSE_BARRIER: u32 = 1 << 3; // bit 3
pub const SENSE_SELF: u32    = 1 << 4; // bit 4
pub const SENSE_MOSSROCK: u32 = 1 << 5; // bit 5

/// Oculocyte cell type index
const OCULOCYTE_TYPE: i32 = 7;

/// Clear all signal channels and flow tracking.
pub fn clear_all_signals(state: &mut CanonicalState) {
    for channel in state.signal_channels.iter_mut() {
        *channel = None;
    }
    state.has_any_signal = false;
    
    // Clear signal flow tracking as well
    state.signal_flow_tracker.clear();
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
    /// Number of BFS hops
    pub hops: usize,
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
        let channel = mode.oculocyte_signal_channel.clamp(0, 7) as usize; // Oculocyte channels 0-7 only
        let signal_value = mode.oculocyte_signal_value.clamp(-100.0, 100.0);
        let hops = mode.oculocyte_signal_hops.clamp(1, 20) as usize;
        let ray_length = mode.oculocyte_ray_length.clamp(1.0, 100.0);

        // Forward direction from genome orientation
        let forward = state.genome_orientations[cell_idx] * Vec3::Z;
        let pos = state.positions[cell_idx];

        // Bitmask: detect if ANY of the enabled sense types fires.
        // Each bit is checked independently; the cell emits if at least one hits.
        let detected =
            ((sense_mask & SENSE_SELF) != 0)  // Self always fires
            || ((sense_mask & SENSE_CELL) != 0 && sense_cells_ray(state, cell_idx, pos, forward, ray_length))
            || ((sense_mask & SENSE_BARRIER) != 0 && sense_barrier_ray(pos, forward, ray_length, boundary_radius))
            // Food and Light require fluid/light systems — not available in preview
            || (sense_mask & SENSE_FOOD) != 0 && false
            || (sense_mask & SENSE_LIGHT) != 0 && false;

        if detected {
            emissions.push(SignalEmission {
                source_cell: cell_idx,
                channel,
                value: signal_value,
                hops,
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
fn sense_barrier_ray(
    pos: Vec3,
    forward: Vec3,
    ray_length: f32,
    boundary_radius: f32,
) -> bool {
    // For a sphere centered at origin with radius R:
    // |pos + t*forward|² = R²
    // t² + 2*(pos·forward)*t + (|pos|² - R²) = 0
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

/// Propagate signal emissions through adhesion connections using hop-level wave propagation.
///
/// Processes one hop-level at a time so that attenuation is applied uniformly at every
/// hop (SIGNAL_ATTENUATION_PER_HOP^N at distance N). Cells reachable via multiple
/// equal-length paths accumulate signal additively from all of those paths — "electricity
/// through wires" behaviour. Cells already reached by a shorter path are not revisited
/// at greater attenuation. Relay cells (oculocytes, regulation emitters) carry the wave
/// forward without accumulating signal themselves.
pub fn propagate_signals(
    state: &mut CanonicalState,
    genome: &Genome,
    emissions: &[SignalEmission],
) {
    if emissions.is_empty() {
        return;
    }

    let cell_count = state.cell_count;

    // Per-emission scratch buffers, reused across emissions to avoid allocation.
    // min_hop_distance[i]: shortest hop distance from the current emitter to cell i
    //   (-1 = not yet reached by this emission's wave).
    // signal_contribution[i]: total signal accumulated from all equal-shortest-length paths.
    let mut min_hop_distance = vec![-1i32; cell_count];
    let mut signal_contribution = vec![0.0f32; cell_count];
    let mut current_frontier: Vec<usize> = Vec::new();
    let mut next_frontier: Vec<usize> = Vec::new();

    for emission in emissions {
        if emission.source_cell >= cell_count {
            continue;
        }

        // Source cell receives its own emission at full (unattenuated) strength.
        add_signal(state, emission.source_cell, emission.channel, emission.value);

        if emission.hops == 0 {
            continue;
        }

        // Reset per-emission scratch state.
        for v in min_hop_distance[..cell_count].iter_mut() { *v = -1; }
        for v in signal_contribution[..cell_count].iter_mut() { *v = 0.0; }
        min_hop_distance[emission.source_cell] = 0;

        current_frontier.clear();
        current_frontier.push(emission.source_cell);

        // Process one hop-level at a time.
        // Hop 1 is full strength (same as the source). Attenuation begins at hop 2:
        //   hop 1 → emission.value
        //   hop 2 → emission.value * SIGNAL_ATTENUATION_PER_HOP
        //   hop N → emission.value * SIGNAL_ATTENUATION_PER_HOP^(N-1)
        let mut signal_at_hop = emission.value;
        for hop in 0..emission.hops {
            let hop_number = (hop + 1) as i32;

            next_frontier.clear();
            for &cell_idx in &current_frontier {
                let neighbors = get_adhesion_neighbors(state, cell_idx);
                for neighbor in neighbors {
                    if neighbor >= cell_count { continue; }
                    if min_hop_distance[neighbor] == -1 {
                        // First path to reach this cell — record distance and seed signal.
                        min_hop_distance[neighbor] = hop_number;
                        signal_contribution[neighbor] = signal_at_hop;
                        next_frontier.push(neighbor);
                        // Record the adhesion edge the signal actually travelled along.
                        state.signal_flow_tracker.add_flow(cell_idx, neighbor);
                    } else if min_hop_distance[neighbor] == hop_number {
                        // Another equal-length path — take the max, not the sum.
                        // Summing would let branching topology amplify signals indefinitely
                        // (a binary tree has 2^(N-1) paths of length N, each contributing
                        // v * 0.5^(N-1), summing to v regardless of distance).
                        // Max gives voltage-like behaviour: signal strength depends only on
                        // hop distance, not on how many wires run in parallel.
                        if signal_at_hop > signal_contribution[neighbor] {
                            signal_contribution[neighbor] = signal_at_hop;
                        }
                        // Record this parallel path's edge too.
                        state.signal_flow_tracker.add_flow(cell_idx, neighbor);
                    }
                    // min_hop_distance < hop_number: shorter path already won —
                    // signal does not flow backward, so don't record this edge.
                }
            }
            std::mem::swap(&mut current_frontier, &mut next_frontier);
            // Attenuate for the *next* hop. This keeps hop 1 at full strength
            // and reduces by SIGNAL_ATTENUATION_PER_HOP for each subsequent hop.
            signal_at_hop *= SIGNAL_ATTENUATION_PER_HOP;
        }

        // Write accumulated contributions to signal channels.
        // Skip the source (already written above) and relay cells (oculocytes /
        // regulation emitters carry the wave but do not receive it).
        for cell_idx in 0..cell_count {
            if cell_idx == emission.source_cell { continue; }
            if min_hop_distance[cell_idx] < 0 { continue; }
            if is_signal_sender(state, genome, cell_idx, emissions) { continue; }
            if signal_contribution[cell_idx] != 0.0 {
                add_signal(state, cell_idx, emission.channel, signal_contribution[cell_idx]);
            }
        }
    }

    state.has_any_signal = true;
}

/// Check if a cell is a transparent relay that carries the signal wave but does
/// not itself receive a signal value written to its channels.
///
/// Only oculocytes qualify: their output is determined by ray-casting and must
/// not be altered by inbound signals from other cells. Every other cell type —
/// including regulation emitters — should both emit *and* receive signals normally.
/// Regulation emitters participating as receivers is exactly how inter-cell signal
/// gating (division, apoptosis, mode-switching) works in practice.
fn is_signal_sender(state: &CanonicalState, genome: &Genome, cell_idx: usize, _emissions: &[SignalEmission]) -> bool {
    let mode_idx = state.mode_indices[cell_idx];
    if let Some(mode) = genome.modes.get(mode_idx) {
        if mode.cell_type == OCULOCYTE_TYPE {
            return true;
        }
    }
    false
}

/// Add a signal value to a cell's channel (additive).
/// If the channel is currently null, sets it to the value.
/// If already has a value, adds to it.
fn add_signal(state: &mut CanonicalState, cell_idx: usize, channel: usize, value: f32) {
    let idx = cell_idx * SIGNAL_CHANNELS + channel;
    if idx >= state.signal_channels.len() {
        return;
    }
    state.signal_channels[idx] = Some(
        state.signal_channels[idx].unwrap_or(0.0) + value,
    );
}

/// Get all adhesion-connected neighbors of a cell.
fn get_adhesion_neighbors(state: &CanonicalState, cell_idx: usize) -> Vec<usize> {
    let connections = state.adhesion_manager.get_connections_for_cell(
        &state.adhesion_connections,
        cell_idx,
    );

    let mut neighbors = Vec::with_capacity(connections.len());
    for conn_idx in connections {
        let cell_a = state.adhesion_connections.cell_a_index[conn_idx];
        let cell_b = state.adhesion_connections.cell_b_index[conn_idx];
        let neighbor = if cell_a == cell_idx { cell_b } else { cell_a };
        neighbors.push(neighbor);
    }
    neighbors
}

/// Track actual signal flow paths for visualization
/// Maps (source_cell, target_cell) -> true if signal flowed from source to target
#[derive(Clone, Debug)]
pub struct SignalFlowTracker {
    flows: std::collections::HashSet<(usize, usize)>,
}

impl SignalFlowTracker {
    pub fn new() -> Self {
        Self {
            flows: std::collections::HashSet::new(),
        }
    }
    
    pub fn add_flow(&mut self, source: usize, target: usize) {
        self.flows.insert((source, target));
    }
    
    pub fn has_flow(&self, source: usize, target: usize) -> bool {
        self.flows.contains(&(source, target)) || self.flows.contains(&(target, source))
    }
    
    pub fn clear(&mut self) {
        self.flows.clear();
    }
}

/// Check if a cell has any active signal on any channel.
pub fn cell_has_any_signal(state: &CanonicalState, cell_idx: usize) -> bool {
    let base = cell_idx * SIGNAL_CHANNELS;
    for ch in 0..SIGNAL_CHANNELS {
        if state.signal_channels[base + ch].is_some() {
            return true;
        }
    }
    false
}

/// Check if an adhesion connection has signal flowing through it.
/// A connection is considered active if both endpoints have signal on the same channel.
/// This correctly handles multi-hop propagation without misattributing relay cells as senders.
pub fn adhesion_has_signal(state: &CanonicalState, cell_a: usize, cell_b: usize) -> bool {
    let base_a = cell_a * SIGNAL_CHANNELS;
    let base_b = cell_b * SIGNAL_CHANNELS;
    for ch in 0..SIGNAL_CHANNELS {
        if state.signal_channels[base_a + ch].is_some() && state.signal_channels[base_b + ch].is_some() {
            return true;
        }
    }
    false
}

/// Run the complete signal system for one frame:
/// 1. Clear all signals
/// 2. Run oculocyte sensing (channels 0-7)
/// 3. Emit regulation signals (channels 8-15)
/// 4. Propagate all signals via BFS
pub fn run_signal_system(
    state: &mut CanonicalState,
    genome: &Genome,
    boundary_radius: f32,
) {
    clear_all_signals(state);
    let mut emissions = sense_oculocytes(state, genome, boundary_radius);
    let regulation_emissions = emit_regulation_signals(state, genome);
    emissions.extend(regulation_emissions);
    propagate_signals(state, genome, &emissions);
}

/// Emit regulation signals for all cells whose mode has regulation_emit_channel >= 8.
/// These are unconditional emissions — any cell type can emit on regulation channels.
pub fn emit_regulation_signals(
    state: &CanonicalState,
    genome: &Genome,
) -> Vec<SignalEmission> {
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
        let value = mode.regulation_emit_value.clamp(0.0, 2047.0);
        let hops = mode.regulation_emit_hops.clamp(1, 20) as usize;

        if value > 0.0 {
            emissions.push(SignalEmission {
                source_cell: cell_idx,
                channel,
                value,
                hops,
            });
        }
    }

    emissions
}

/// Run signal propagation with manually-provided emissions (for test buttons).
/// Does NOT clear signals first — call clear_all_signals() separately if needed.
pub fn propagate_test_signals(
    state: &mut CanonicalState,
    genome: &Genome,
    emissions: Vec<SignalEmission>,
) {
    propagate_signals(state, genome, &emissions);
}
