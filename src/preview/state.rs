use crate::simulation::CanonicalState;
use crate::genome::Genome;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Preview simulation state
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
        // TODO: Hash genome fields
        genome.hash(&mut hasher);
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
