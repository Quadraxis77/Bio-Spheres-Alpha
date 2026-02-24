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
pub const SIGNAL_ATTENUATION_PER_HOP: f32 = 0.8; // 80% signal strength retained per hop

/// Oculocyte sense types
pub const SENSE_CELL: i32 = 0;
pub const SENSE_FOOD: i32 = 1;
pub const SENSE_LIGHT: i32 = 2;
pub const SENSE_BARRIER: i32 = 3;

/// Oculocyte cell type index
const OCULOCYTE_TYPE: i32 = 7;

/// Clear all signal channels to null for all cells.
/// Called at the start of each frame before sensing.
pub fn clear_all_signals(state: &mut CanonicalState) {
    for i in 0..(state.cell_count * SIGNAL_CHANNELS) {
        state.signal_channels[i] = None;
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
        let channel = mode.oculocyte_signal_channel.clamp(0, 15) as usize;
        let signal_value = mode.oculocyte_signal_value.clamp(-50.0, 50.0);
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
        
        // Write signal to source cell first (full strength)
        add_signal(state, emission.source_cell, emission.channel, emission.value);

        // BFS from source
        // Clear visited for this emission
        for v in visited.iter_mut().take(state.cell_count) {
            *v = false;
        }
        // Check bounds before accessing visited array
        if emission.source_cell < state.cell_count {
            visited[emission.source_cell] = true;
        } else {
            continue; // Skip this emission
        }
        queue.clear();

        // Seed BFS with neighbors of source (first hop gets attenuated signal)
        let attenuated_strength = emission.value * SIGNAL_ATTENUATION_PER_HOP;
        let neighbors = get_adhesion_neighbors(state, emission.source_cell);
        for neighbor in neighbors {
            if neighbor < state.cell_count && !visited[neighbor] {
                // Skip signal senders - they are immune to receiving signals
                if !is_signal_sender(state, genome, neighbor, emissions) {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, emission.hops - 1, attenuated_strength));
                }
            }
        }

        while let Some((cell_idx, remaining_hops, signal_strength)) = queue.pop_front() {
            // Write attenuated signal to this cell (skip if signal sender)
            if !is_signal_sender(state, genome, cell_idx, emissions) {
                add_signal(state, cell_idx, emission.channel, signal_strength);
            }

            // Continue BFS if hops remain
            if remaining_hops > 0 {
                let next_strength = signal_strength * SIGNAL_ATTENUATION_PER_HOP;
                let neighbors = get_adhesion_neighbors(state, cell_idx);
                for neighbor in neighbors {
                    if neighbor < state.cell_count && !visited[neighbor] {
                        // Skip signal senders - they are immune to receiving signals
                        if !is_signal_sender(state, genome, neighbor, emissions) {
                            visited[neighbor] = true;
                            queue.push_back((neighbor, remaining_hops - 1, next_strength));
                        }
                    }
                }
            }
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
/// A bond glows only when both endpoints share signal on the same channel,
/// meaning the signal actually propagated through the bond rather than terminating at one end.
pub fn adhesion_has_signal(state: &CanonicalState, cell_a: usize, cell_b: usize) -> bool {
    let base_a = cell_a * SIGNAL_CHANNELS;
    let base_b = cell_b * SIGNAL_CHANNELS;
    for ch in 0..SIGNAL_CHANNELS {
        if state.signal_channels.get(base_a + ch).copied().flatten().is_some()
            && state.signal_channels.get(base_b + ch).copied().flatten().is_some()
        {
            return true;
        }
    }
    false
}

/// Run the complete signal system for one frame:
/// 1. Clear all signals
/// 2. Run oculocyte sensing
/// 3. Propagate signals via BFS
pub fn run_signal_system(
    state: &mut CanonicalState,
    genome: &Genome,
    boundary_radius: f32,
) {
    clear_all_signals(state);
    let emissions = sense_oculocytes(state, genome, boundary_radius);
    propagate_signals(state, genome, &emissions);
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
