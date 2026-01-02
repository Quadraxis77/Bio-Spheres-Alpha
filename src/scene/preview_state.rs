//! Preview simulation state with checkpoint support.
//!
//! Provides state management for the preview scene including
//! checkpoints for fast backward time scrubbing.

use crate::genome::Genome;
use crate::simulation::CanonicalState;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Preview simulation state with checkpoint support.
pub struct PreviewState {
    /// Canonical state for preview simulation
    pub canonical_state: CanonicalState,

    /// Current preview time
    pub current_time: f32,

    /// Checkpoints for fast backward scrubbing (time, state)
    pub checkpoints: Vec<(f32, CanonicalState)>,

    /// Checkpoint interval in seconds
    pub checkpoint_interval: f32,

    /// Hash of genome to detect changes
    pub genome_hash: u64,
}

impl PreviewState {
    pub fn new(capacity: usize) -> Self {
        Self {
            canonical_state: CanonicalState::new(capacity),
            current_time: 0.0,
            checkpoints: Vec::new(),
            checkpoint_interval: 5.0,
            genome_hash: 0,
        }
    }

    /// Compute a comprehensive hash of the genome for change detection
    pub fn compute_genome_hash(genome: &Genome) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash genome fields manually since Genome doesn't implement Hash
        genome.name.hash(&mut hasher);
        genome.initial_mode.hash(&mut hasher);
        
        // Hash quaternion components
        genome.initial_orientation.x.to_bits().hash(&mut hasher);
        genome.initial_orientation.y.to_bits().hash(&mut hasher);
        genome.initial_orientation.z.to_bits().hash(&mut hasher);
        genome.initial_orientation.w.to_bits().hash(&mut hasher);
        
        // Hash each mode
        for mode in &genome.modes {
            mode.name.hash(&mut hasher);
            mode.default_name.hash(&mut hasher);
            mode.color.x.to_bits().hash(&mut hasher);
            mode.color.y.to_bits().hash(&mut hasher);
            mode.color.z.to_bits().hash(&mut hasher);
            mode.opacity.to_bits().hash(&mut hasher);
            mode.emissive.to_bits().hash(&mut hasher);
            mode.cell_type.hash(&mut hasher);
            mode.parent_make_adhesion.hash(&mut hasher);
            mode.split_mass.to_bits().hash(&mut hasher);
            mode.split_interval.to_bits().hash(&mut hasher);
            // Hash split direction (pitch and yaw)
            mode.parent_split_direction.x.to_bits().hash(&mut hasher);
            mode.parent_split_direction.y.to_bits().hash(&mut hasher);
            // Hash other important fields for change detection
            mode.max_splits.hash(&mut hasher);
            mode.max_adhesions.hash(&mut hasher);
            mode.min_adhesions.hash(&mut hasher);
        }
        
        hasher.finish()
    }

    /// Clear checkpoints (called when genome changes)
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Find the best checkpoint to start from for a given target time
    pub fn find_best_checkpoint(&self, target_time: f32) -> Option<(f32, CanonicalState)> {
        self.checkpoints
            .iter()
            .rev()
            .find(|(time, _)| *time <= target_time)
            .cloned()
    }

    /// Add a checkpoint if we've passed a checkpoint interval
    pub fn maybe_add_checkpoint(&mut self, time: f32, state: &CanonicalState) {
        let checkpoint_index = (time / self.checkpoint_interval).floor() as usize;

        if self.checkpoints.len() <= checkpoint_index {
            self.checkpoints.push((time, state.clone()));
        }
    }
}
