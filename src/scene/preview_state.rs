//! Preview simulation state with checkpoint support.
//!
//! Provides state management for the preview scene including
//! checkpoints for fast backward time scrubbing.

use crate::genome::Genome;
use crate::simulation::CanonicalState;
use crate::simulation::PhysicsConfig;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Initial state for preview simulation
/// Used to reset simulation when genome changes or seeking to time 0
#[derive(Clone)]
pub struct InitialState {
    /// Initial cell data
    pub initial_cells: Vec<InitialCell>,
    /// Maximum cell capacity
    pub max_cells: usize,
    /// Random seed for deterministic simulation
    pub rng_seed: u64,
}

/// Initial cell data for preview simulation
#[derive(Clone)]
pub struct InitialCell {
    pub position: glam::Vec3,
    pub velocity: glam::Vec3,
    pub rotation: glam::Quat,
    pub genome_orientation: glam::Quat,
    pub angular_velocity: glam::Vec3,
    pub mass: f32,
    pub radius: f32,
    pub genome_id: usize,
    pub mode_index: usize,
    pub split_interval: f32,
    pub split_mass: f32,
    pub stiffness: f32,
}

impl InitialState {
    /// Create initial state from genome
    pub fn from_genome(genome: &Genome, capacity: usize, physics_config: &crate::simulation::physics_config::PhysicsConfig) -> Self {
        let initial_mode_index = genome.initial_mode.max(0) as usize;
        let mode = genome.modes.get(initial_mode_index)
            .or_else(|| genome.modes.first());
        
        let (split_interval, split_mass, membrane_stiffness, cell_type) = if let Some(m) = mode {
            (m.split_interval, m.split_mass, m.membrane_stiffness, m.cell_type)
        } else {
            (5.0, 1.5, physics_config.default_stiffness, 0)
        };

        // Set initial mass based on cell type
        // Phagocytes get extra mass to demonstrate splitting behavior before needing nutrient transport
        let initial_mass = if cell_type == 2 {
            // Phagocyte: In preview scene they auto-gain mass like test cells,
            // but start with extra to show splitting behavior sooner
            (split_mass * 1.2).max(2.0)
        } else {
            1.0
        };

        Self {
            initial_cells: vec![InitialCell {
                position: glam::Vec3::ZERO,
                velocity: glam::Vec3::ZERO,
                rotation: genome.initial_orientation,
                genome_orientation: genome.initial_orientation,
                angular_velocity: glam::Vec3::ZERO,
                mass: initial_mass,
                radius: initial_mass.clamp(0.5, 2.0),
                genome_id: 0,
                mode_index: initial_mode_index,
                split_interval,
                split_mass,
                stiffness: membrane_stiffness, // Use mode-specific membrane stiffness
            }],
            max_cells: capacity,
            rng_seed: 12345,
        }
    }
    
    /// Convert to canonical state
    pub fn to_canonical_state(&self) -> CanonicalState {
        let mut state = CanonicalState::new(self.max_cells);
        
        for cell in &self.initial_cells {
            let _ = state.add_cell(
                cell.position,
                cell.velocity,
                cell.rotation,
                cell.genome_orientation,
                cell.angular_velocity,
                cell.mass,
                cell.radius,
                cell.genome_id,
                cell.mode_index,
                0.0, // birth_time
                cell.split_interval,
                cell.split_mass,
                cell.stiffness,
            );
        }
        
        state
    }
}

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
    
    /// Initial state for resetting simulation
    pub initial_state: InitialState,
    
    /// Target time for seeking (None = no seek requested)
    pub target_time: Option<f32>,
    
    /// Whether currently resimulating
    pub is_resimulating: bool,
}

impl PreviewState {
    pub fn new(capacity: usize, physics_config: &crate::simulation::physics_config::PhysicsConfig) -> Self {
        let genome = Genome::default();
        let initial_state = InitialState::from_genome(&genome, capacity, physics_config);
        
        Self {
            canonical_state: initial_state.to_canonical_state(),
            current_time: 0.0,
            checkpoints: Vec::new(),
            checkpoint_interval: 1.0, // Checkpoint every 1 second for responsive scrubbing
            genome_hash: 0,
            initial_state,
            target_time: None,
            is_resimulating: false,
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
            mode.membrane_stiffness.to_bits().hash(&mut hasher);
            // Hash split direction (pitch and yaw)
            mode.parent_split_direction.x.to_bits().hash(&mut hasher);
            mode.parent_split_direction.y.to_bits().hash(&mut hasher);
            // Hash other important fields for change detection
            mode.max_splits.hash(&mut hasher);
            mode.mode_a_after_splits.hash(&mut hasher);
            mode.mode_b_after_splits.hash(&mut hasher);
            mode.max_adhesions.hash(&mut hasher);
            mode.min_adhesions.hash(&mut hasher);
            mode.enable_parent_angle_snapping.hash(&mut hasher);
            mode.split_ratio.to_bits().hash(&mut hasher);
            mode.nutrient_gain_rate.to_bits().hash(&mut hasher);
            mode.max_cell_size.to_bits().hash(&mut hasher);
            mode.swim_force.to_bits().hash(&mut hasher);
            mode.nutrient_priority.to_bits().hash(&mut hasher);
            mode.prioritize_when_low.hash(&mut hasher);
            // Hash child settings
            mode.child_a.mode_number.hash(&mut hasher);
            mode.child_b.mode_number.hash(&mut hasher);
            mode.child_a.orientation.x.to_bits().hash(&mut hasher);
            mode.child_a.orientation.y.to_bits().hash(&mut hasher);
            mode.child_a.orientation.z.to_bits().hash(&mut hasher);
            mode.child_a.orientation.w.to_bits().hash(&mut hasher);
            mode.child_a.keep_adhesion.hash(&mut hasher);
            mode.child_b.orientation.x.to_bits().hash(&mut hasher);
            mode.child_b.orientation.y.to_bits().hash(&mut hasher);
            mode.child_b.orientation.z.to_bits().hash(&mut hasher);
            mode.child_b.orientation.w.to_bits().hash(&mut hasher);
            mode.child_b.keep_adhesion.hash(&mut hasher);
            
            // Hash adhesion settings
            mode.adhesion_settings.can_break.hash(&mut hasher);
            mode.adhesion_settings.adhesin_length.to_bits().hash(&mut hasher);
            mode.adhesion_settings.adhesin_stretch.to_bits().hash(&mut hasher);
            mode.adhesion_settings.stiffness.to_bits().hash(&mut hasher);
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
    
    /// Request seeking to a specific time
    pub fn seek_to_time(&mut self, target_time: f32) {
        self.target_time = Some(target_time);
    }
    
    /// Update initial state when genome changes
    pub fn update_initial_state(&mut self, genome: &Genome, physics_config: &crate::simulation::physics_config::PhysicsConfig) {
        self.initial_state = InitialState::from_genome(genome, self.canonical_state.capacity, physics_config);
    }
    
    /// Run resimulation to target time
    /// Returns true if resimulation is complete
    pub fn run_resimulation(
        &mut self,
        genome: &Genome,
        config: &PhysicsConfig,
        genome_changed: bool,
    ) -> bool {
        let Some(target_time) = self.target_time else {
            self.is_resimulating = false;
            return true;
        };
        
        self.is_resimulating = true;
        
        // Moving forward: simulate directly from current state (no clone needed)
        if target_time > self.current_time && !genome_changed {
            let start_step = (self.current_time / config.fixed_timestep).ceil() as u32;
            let end_step = (target_time / config.fixed_timestep).ceil() as u32;
            let steps = end_step.saturating_sub(start_step);
            
            let mut last_checkpoint_index = (self.current_time / self.checkpoint_interval).floor() as usize;
            
            // Run physics steps directly on current state
            for step in 0..steps {
                let current_time = (start_step + step) as f32 * config.fixed_timestep;
                
                let _ = crate::simulation::preview_physics::physics_step_with_genome(
                    &mut self.canonical_state,
                    genome,
                    config,
                    current_time,
                );
                
                // Check if we should create a checkpoint
                let current_checkpoint_index = (current_time / self.checkpoint_interval).floor() as usize;
                if current_checkpoint_index > last_checkpoint_index {
                    self.checkpoints.push((current_time, self.canonical_state.clone()));
                    last_checkpoint_index = current_checkpoint_index;
                }
            }
            
            self.current_time = target_time;
            self.target_time = None;
            self.is_resimulating = false;
            return true;
        }
        
        // Moving backward or genome changed: need to resimulate from checkpoint
        let (start_time, mut canonical_state) = if let Some((checkpoint_time, checkpoint_state)) = self.find_best_checkpoint(target_time) {
            (checkpoint_time, checkpoint_state)
        } else {
            // No suitable checkpoint: start from initial state
            (0.0, self.initial_state.to_canonical_state())
        };
        
        let start_step = (start_time / config.fixed_timestep).ceil() as u32;
        let end_step = (target_time / config.fixed_timestep).ceil() as u32;
        let steps = end_step.saturating_sub(start_step);
        
        let mut last_checkpoint_index = (start_time / self.checkpoint_interval).floor() as usize;
        
        // Run physics steps
        for step in 0..steps {
            let current_time = (start_step + step) as f32 * config.fixed_timestep;
            
            let _ = crate::simulation::preview_physics::physics_step_with_genome(
                &mut canonical_state,
                genome,
                config,
                current_time,
            );
            
            // Check if we should create a checkpoint
            let current_checkpoint_index = (current_time / self.checkpoint_interval).floor() as usize;
            if current_checkpoint_index > last_checkpoint_index {
                self.checkpoints.push((current_time, canonical_state.clone()));
                last_checkpoint_index = current_checkpoint_index;
            }
        }
        
        // Apply results
        self.canonical_state = canonical_state;
        self.current_time = target_time;
        self.target_time = None;
        self.is_resimulating = false;
        
        true
    }
}
