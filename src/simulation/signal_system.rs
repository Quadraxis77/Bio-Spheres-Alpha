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
use std::collections::VecDeque;

/// Number of signal channels (0-15)
pub const SIGNAL_CHANNELS: usize = 16;
/// Signal attenuation factor per hop (signals lose this fraction per hop)
pub const SIGNAL_ATTENUATION_PER_HOP: f32 = 0.5; // 50% signal strength retained per hop

/// Oculocyte sense types
pub const SENSE_CELL: i32 = 0;
pub const SENSE_FOOD: i32 = 1;
pub const SENSE_LIGHT: i32 = 2;
pub const SENSE_BARRIER: i32 = 3;
pub const SENSE_SELF: i32 = 4;

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

        let sense_type = mode.oculocyte_sense_type;
        let channel = mode.oculocyte_signal_channel.clamp(0, 7) as usize; // Oculocyte channels 0-7 only
        let signal_value = mode.oculocyte_signal_value.clamp(-100.0, 100.0);
        let hops = mode.oculocyte_signal_hops.clamp(1, 20) as usize;
        let ray_length = mode.oculocyte_ray_length.clamp(1.0, 100.0);

        // Forward direction from genome orientation
        let forward = state.genome_orientations[cell_idx] * Vec3::Z;
        let pos = state.positions[cell_idx];

        let detected = match sense_type {
            SENSE_CELL => sense_cells_ray(state, cell_idx, pos, forward, ray_length),
            SENSE_FOOD => false, // Food sensing requires fluid system access — not available in preview
            SENSE_LIGHT => false, // Light sensing requires light field — not available in preview
            SENSE_BARRIER => sense_barrier_ray(pos, forward, ray_length, boundary_radius),
            SENSE_SELF => true, // Self-sense always detects — emits unconditional positional gradient signal
            _ => false,
        };

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

/// Propagate signal emissions through adhesion connections using BFS.
/// Each emission starts at the source cell and floods outward for `hops` steps.
/// Signals are additive — multiple sources on the same channel sum their values.
/// Signal senders (oculocytes) are immune to receiving signals that would alter their output.
pub fn propagate_signals(
    state: &mut CanonicalState,
    genome: &Genome,
    emissions: &[SignalEmission],
) {
    if emissions.is_empty() {
        return;
    }

    // BFS queue: (cell_index, remaining_hops, signal_strength)
    let mut queue: VecDeque<(usize, usize, f32)> = VecDeque::new();
    // Track visited cells per emission to avoid re-visiting in the same BFS
    let mut visited: Vec<bool> = vec![false; state.cell_count];

    for emission in emissions {
        // Check bounds before writing signal to source cell
        if emission.source_cell >= state.cell_count {
            continue;
        }

        let debug_target = 6usize; // Cell we're investigating
        let debug_source = 1usize; // Suspected emission source
        let is_debug_emission = emission.source_cell == debug_source;

        // Write signal to source cell first (full strength)
        add_signal(state, emission.source_cell, emission.channel, emission.value);

        // BFS from source.
        // Cells are written when enqueued, not when dequeued, so the signal strength
        // and visited marking are both set at the correct hop distance.
        // remaining_hops = how many more hops can spread FROM this cell.
        // hops=1 means: spread to direct neighbours (write them at full strength), then stop.
        for v in visited.iter_mut().take(state.cell_count) {
            *v = false;
        }
        if emission.source_cell < state.cell_count {
            visited[emission.source_cell] = true;
        } else {
            continue;
        }
        queue.clear();

        if is_debug_emission {
            println!(
                "[SIGNAL DEBUG] Emission from cell {} | channel={} value={} hops={}",
                emission.source_cell, emission.channel, emission.value, emission.hops
            );
            // Print all adhesion neighbours of the source
            let src_neighbors = get_adhesion_neighbors(state, emission.source_cell);
            println!(
                "[SIGNAL DEBUG]   Direct neighbours of source {}: {:?}",
                emission.source_cell, src_neighbors
            );
            // Print all adhesion neighbours of the target cell
            let tgt_neighbors = get_adhesion_neighbors(state, debug_target);
            println!(
                "[SIGNAL DEBUG]   Direct neighbours of target {}: {:?}",
                debug_target, tgt_neighbors
            );
        }

        // Seed with direct neighbours of source at full strength.
        // Signal senders are marked visited (to block loops) but not written to and not
        // enqueued for further spreading — they are a hard stop in the BFS.
        if emission.hops > 0 {
            let neighbors = get_adhesion_neighbors(state, emission.source_cell);
            for neighbor in neighbors {
                if neighbor < state.cell_count && !visited[neighbor] {
                    visited[neighbor] = true;
                    if is_signal_sender(state, genome, neighbor, emissions) {
                        if is_debug_emission {
                            println!(
                                "[SIGNAL DEBUG]   Hop-1 neighbour {} is a signal sender — hard stop",
                                neighbor
                            );
                        }
                        continue;
                    }
                    if is_debug_emission && neighbor == debug_target {
                        println!(
                            "[SIGNAL DEBUG]   *** Writing signal to target {} at HOP 1 (full strength={}) — source {} IS a direct neighbour!",
                            debug_target, emission.value, emission.source_cell
                        );
                    }
                    add_signal(state, neighbor, emission.channel, emission.value);
                    state.signal_flow_tracker.add_flow(emission.source_cell, neighbor);
                    if emission.hops > 1 {
                        if is_debug_emission {
                            println!(
                                "[SIGNAL DEBUG]   Enqueuing hop-1 neighbour {} for further spread (remaining_hops={})",
                                neighbor, emission.hops - 1
                            );
                        }
                        queue.push_back((neighbor, emission.hops - 1, emission.value));
                    }
                }
            }
        }

        while let Some((cell_idx, remaining_hops, signal_strength)) = queue.pop_front() {
            // Apply attenuation for this hop's outgoing neighbours
            let next_strength = signal_strength * SIGNAL_ATTENUATION_PER_HOP;
            let neighbors = get_adhesion_neighbors(state, cell_idx);
            if is_debug_emission {
                println!(
                    "[SIGNAL DEBUG]   Queue pop: cell={} remaining_hops={} strength={:.4} | neighbours={:?}",
                    cell_idx, remaining_hops, signal_strength, neighbors
                );
            }
            for neighbor in neighbors {
                if neighbor < state.cell_count && !visited[neighbor] {
                    visited[neighbor] = true;
                    if is_signal_sender(state, genome, neighbor, emissions) {
                        if is_debug_emission {
                            println!(
                                "[SIGNAL DEBUG]     Neighbour {} is a signal sender — hard stop",
                                neighbor
                            );
                        }
                        continue;
                    }
                    if is_debug_emission && neighbor == debug_target {
                        println!(
                            "[SIGNAL DEBUG]   *** Writing signal to target {} via cell {} at strength={:.4} remaining_hops={}",
                            debug_target, cell_idx, next_strength, remaining_hops
                        );
                    }
                    add_signal(state, neighbor, emission.channel, next_strength);
                    state.signal_flow_tracker.add_flow(emission.source_cell, neighbor);
                    if remaining_hops > 1 {
                        queue.push_back((neighbor, remaining_hops - 1, next_strength));
                    }
                } else if is_debug_emission && neighbor == debug_target {
                    println!(
                        "[SIGNAL DEBUG]     Target {} already visited or out of bounds (visited={})",
                        debug_target,
                        if neighbor < state.cell_count { visited[neighbor].to_string() } else { "OOB".to_string() }
                    );
                }
            }
        }

        if is_debug_emission {
            let target_signal = state.signal_channels.get(debug_target * SIGNAL_CHANNELS + emission.channel).copied().flatten();
            println!(
                "[SIGNAL DEBUG] After BFS: cell {} channel {} signal = {:?}",
                debug_target, emission.channel, target_signal
            );
        }
    }

    state.has_any_signal = true;
}

/// Check if a cell is a signal sender (oculocyte or test signal source).
/// Signal senders are immune to receiving signals that would alter their own output.
fn is_signal_sender(state: &CanonicalState, genome: &Genome, cell_idx: usize, test_emissions: &[SignalEmission]) -> bool {
    // Check if this cell is an oculocyte
    let mode_idx = state.mode_indices[cell_idx];
    if let Some(mode) = genome.modes.get(mode_idx) {
        if mode.cell_type == OCULOCYTE_TYPE {
            return true;
        }
    }
    
    // Check if this cell is a test signal source
    test_emissions.iter().any(|emission| emission.source_cell == cell_idx)
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
