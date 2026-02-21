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

/// Number of signal channels per cell
pub const SIGNAL_CHANNELS: usize = 16;

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
        let sense_range = mode.oculocyte_sense_range.clamp(25.0, 50.0);

        // Forward direction from genome orientation
        let forward = state.genome_orientations[cell_idx] * Vec3::Z;
        let pos = state.positions[cell_idx];

        // FOV curve: quadratic so range drops faster as FOV widens
        // range=25 -> t=0.5 -> cos=0.25 -> half_angle~75°
        // range=50 -> t=1.0 -> cos~0.996 -> half_angle~5°
        let t = (sense_range / 50.0).clamp(0.0, 0.998);
        let cos_half_fov = t * t;

        let detected = match sense_type {
            SENSE_CELL => sense_cells(state, cell_idx, pos, forward, sense_range, cos_half_fov),
            SENSE_FOOD => false, // Food sensing requires fluid system access — not available in preview
            SENSE_LIGHT => false, // Light sensing requires light field — not available in preview
            SENSE_BARRIER => sense_barrier(pos, forward, sense_range, cos_half_fov, boundary_radius),
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

/// Sense other cells in the forward cone.
fn sense_cells(
    state: &CanonicalState,
    self_idx: usize,
    pos: Vec3,
    forward: Vec3,
    range: f32,
    cos_half_fov: f32,
) -> bool {
    let range_sq = range * range;

    for other_idx in 0..state.cell_count {
        if other_idx == self_idx {
            continue;
        }

        let other_pos = state.positions[other_idx];
        let to_other = other_pos - pos;
        let dist_sq = to_other.length_squared();

        if dist_sq > range_sq || dist_sq < 0.001 {
            continue;
        }

        let dir = to_other.normalize();
        if forward.dot(dir) >= cos_half_fov {
            return true;
        }
    }

    false
}

/// Sense barrier/cave walls or world boundary in the forward cone.
fn sense_barrier(
    pos: Vec3,
    forward: Vec3,
    range: f32,
    _cos_half_fov: f32,
    boundary_radius: f32,
) -> bool {
    // Simple ray-sphere intersection with the world boundary
    // Cast a ray from pos along forward direction, check if it hits the boundary sphere
    // within range.
    //
    // For a sphere centered at origin with radius R:
    // |pos + t*forward|² = R²
    // t² + 2*(pos·forward)*t + (|pos|² - R²) = 0
    let a = 1.0; // forward is normalized
    let b = 2.0 * pos.dot(forward);
    let c = pos.length_squared() - boundary_radius * boundary_radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);

    // We want t > 0 (ahead of us) and t <= range
    let t = if t1 > 0.0 { t1 } else { t2 };
    t > 0.0 && t <= range
}

/// Propagate signal emissions through adhesion connections using BFS.
/// Each emission starts at the source cell and floods outward for `hops` steps.
/// Signals are additive — multiple sources on the same channel sum their values.
pub fn propagate_signals(
    state: &mut CanonicalState,
    emissions: &[SignalEmission],
) {
    if emissions.is_empty() {
        return;
    }

    // BFS queue: (cell_index, remaining_hops)
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    // Track visited cells per emission to avoid re-visiting in the same BFS
    let mut visited: Vec<bool> = vec![false; state.cell_count];

    for emission in emissions {
        // Write signal to source cell first
        add_signal(state, emission.source_cell, emission.channel, emission.value);

        // BFS from source
        // Clear visited for this emission
        for v in visited.iter_mut().take(state.cell_count) {
            *v = false;
        }
        visited[emission.source_cell] = true;
        queue.clear();

        // Seed BFS with neighbors of source
        let neighbors = get_adhesion_neighbors(state, emission.source_cell);
        for neighbor in neighbors {
            if neighbor < state.cell_count && !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back((neighbor, emission.hops - 1));
            }
        }

        while let Some((cell_idx, remaining_hops)) = queue.pop_front() {
            // Write signal to this cell
            add_signal(state, cell_idx, emission.channel, emission.value);

            // Continue BFS if hops remain
            if remaining_hops > 0 {
                let neighbors = get_adhesion_neighbors(state, cell_idx);
                for neighbor in neighbors {
                    if neighbor < state.cell_count && !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back((neighbor, remaining_hops - 1));
                    }
                }
            }
        }
    }

    state.has_any_signal = true;
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
    propagate_signals(state, &emissions);
}

/// Run signal propagation with manually-provided emissions (for test buttons).
/// Does NOT clear signals first — call clear_all_signals() separately if needed.
pub fn propagate_test_signals(
    state: &mut CanonicalState,
    emissions: Vec<SignalEmission>,
) {
    propagate_signals(state, &emissions);
}
