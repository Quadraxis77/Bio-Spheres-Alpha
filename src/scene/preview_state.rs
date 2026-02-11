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
///
/// Uses a decoupled display/simulation architecture:
/// - `display_state` + `display_time`: snapshot used for rendering (never blocks UI)
/// - `sim_state`: working copy used during incremental resimulation
/// - Each frame, a time-budgeted batch of physics steps runs, then display is updated
pub struct PreviewState {
    /// Display state snapshot for rendering (decoupled from simulation)
    pub display_state: CanonicalState,
    
    /// Display time corresponding to display_state
    pub display_time: f32,

    /// Simulation working state (advanced incrementally toward target)
    sim_state: CanonicalState,
    
    /// Current simulation time of sim_state
    sim_time: f32,

    /// Current preview time (the last completed target)
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
    
    /// Incremental resimulation tracking: current step index
    sim_current_step: u32,
    
    /// Incremental resimulation tracking: end step index
    sim_end_step: u32,
    
    /// Incremental resimulation tracking: start step index (for time calculation)
    sim_start_step: u32,
    
    /// Last checkpoint index created during current resimulation
    sim_last_checkpoint_index: usize,
    
    /// Whether the current resimulation was triggered by a genome change
    sim_genome_changed: bool,
    
    /// Time budget per frame for incremental resimulation (in seconds)
    pub frame_time_budget: std::time::Duration,
}

impl PreviewState {
    pub fn new(capacity: usize, physics_config: &crate::simulation::physics_config::PhysicsConfig) -> Self {
        let genome = Genome::default();
        let initial_state = InitialState::from_genome(&genome, capacity, physics_config);
        let canonical = initial_state.to_canonical_state();
        
        Self {
            display_state: canonical.clone(),
            display_time: 0.0,
            sim_state: canonical.clone(),
            sim_time: 0.0,
            current_time: 0.0,
            checkpoints: Vec::new(),
            checkpoint_interval: 1.0,
            genome_hash: 0,
            initial_state,
            target_time: None,
            is_resimulating: false,
            sim_current_step: 0,
            sim_end_step: 0,
            sim_start_step: 0,
            sim_last_checkpoint_index: 0,
            sim_genome_changed: false,
            frame_time_budget: std::time::Duration::from_millis(8),
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
    
    /// Get resimulation progress as a fraction (0.0 to 1.0).
    /// Returns 1.0 when not resimulating.
    pub fn resimulation_progress(&self) -> f32 {
        if !self.is_resimulating || self.sim_end_step <= self.sim_start_step {
            return 1.0;
        }
        let total = (self.sim_end_step - self.sim_start_step) as f32;
        let done = (self.sim_current_step - self.sim_start_step) as f32;
        (done / total).clamp(0.0, 1.0)
    }
    
    /// Get the current sim time as a fraction of the target time (for progress bar positioning).
    /// Returns the sim_time directly so the UI can position the bar on the time axis.
    pub fn sim_progress_time(&self) -> f32 {
        if !self.is_resimulating {
            return self.display_time;
        }
        self.sim_time
    }
    
    /// Request seeking to a specific time.
    /// Just sets the target — run_incremental_resimulation handles the rest.
    pub fn seek_to_time(&mut self, target_time: f32) {
        self.target_time = Some(target_time);
    }
    
    /// Force a full restart on the next resimulation (e.g. after genome change).
    pub fn force_restart(&mut self) {
        self.is_resimulating = false;
        self.sim_genome_changed = true;
    }
    
    /// Update initial state when genome changes
    pub fn update_initial_state(&mut self, genome: &Genome, physics_config: &crate::simulation::physics_config::PhysicsConfig) {
        self.initial_state = InitialState::from_genome(genome, self.display_state.capacity, physics_config);
    }
    
    /// Begin or continue incremental resimulation toward the target time.
    ///
    /// Runs physics steps for up to `frame_time_budget` wall-clock time per call,
    /// then returns. The display_state is updated each call so the renderer always
    /// has something fresh to show.
    ///
    /// Returns `true` when the target has been fully reached.
    pub fn run_incremental_resimulation(
        &mut self,
        genome: &Genome,
        config: &PhysicsConfig,
    ) -> bool {
        let Some(target_time) = self.target_time else {
            self.is_resimulating = false;
            return true;
        };
        
        if self.is_resimulating {
            // Already in progress — just update the end step in case target moved forward
            let end_step = (target_time / config.fixed_timestep).ceil() as u32;
            self.sim_end_step = end_step;
        } else {
            // --- Need to start a new resimulation ---
            self.is_resimulating = true;
            
            // Can we continue forward from sim_state?
            let can_continue = target_time >= self.sim_time && !self.sim_genome_changed;
            
            if can_continue {
                // sim_state already holds the state at sim_time — just set up steps
                let start_step = (self.sim_time / config.fixed_timestep).ceil() as u32;
                let end_step = (target_time / config.fixed_timestep).ceil() as u32;
                self.sim_start_step = start_step;
                self.sim_end_step = end_step;
                self.sim_current_step = start_step;
                self.sim_last_checkpoint_index = (self.sim_time / self.checkpoint_interval).floor() as usize;
            } else {
                // Backward seek or genome changed — restart from best checkpoint
                let (start_time, checkpoint_state) = if self.sim_genome_changed {
                    // Genome changed: checkpoints were cleared, start from initial
                    (0.0, self.initial_state.to_canonical_state())
                } else if let Some((cp_time, cp_state)) = self.find_best_checkpoint(target_time) {
                    (cp_time, cp_state)
                } else {
                    (0.0, self.initial_state.to_canonical_state())
                };
                
                self.sim_state = checkpoint_state;
                self.sim_time = start_time;
                self.sim_genome_changed = false;
                
                let start_step = (start_time / config.fixed_timestep).ceil() as u32;
                let end_step = (target_time / config.fixed_timestep).ceil() as u32;
                self.sim_start_step = start_step;
                self.sim_end_step = end_step;
                self.sim_current_step = start_step;
                self.sim_last_checkpoint_index = (start_time / self.checkpoint_interval).floor() as usize;
            }
        }
        
        // --- Run a time-budgeted batch of steps ---
        let deadline = std::time::Instant::now() + self.frame_time_budget;
        let mut steps_done = 0u32;
        
        while self.sim_current_step < self.sim_end_step {
            let step_time = self.sim_current_step as f32 * config.fixed_timestep;
            
            let _ = crate::simulation::preview_physics::physics_step_with_genome(
                &mut self.sim_state,
                genome,
                config,
                step_time,
            );
            
            self.sim_current_step += 1;
            self.sim_time = self.sim_current_step as f32 * config.fixed_timestep;
            steps_done += 1;
            
            // Create checkpoint if we crossed an interval boundary
            let current_checkpoint_index = (step_time / self.checkpoint_interval).floor() as usize;
            if current_checkpoint_index > self.sim_last_checkpoint_index {
                self.checkpoints.push((step_time, self.sim_state.clone()));
                self.sim_last_checkpoint_index = current_checkpoint_index;
            }
            
            // Check time budget every 4 steps to amortize Instant::now() cost
            if steps_done % 4 == 0 && std::time::Instant::now() >= deadline {
                break;
            }
        }
        
        // --- Only update display when resimulation is fully complete ---
        if self.sim_current_step >= self.sim_end_step {
            self.display_state = self.sim_state.clone();
            self.display_time = self.sim_time;
            self.current_time = target_time;
            self.target_time = None;
            self.is_resimulating = false;
            return true;
        }
        
        // Still in progress — keep showing the previous display_state
        false
    }
}
