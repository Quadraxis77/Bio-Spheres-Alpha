//! Preview simulation state with checkpoint support.
//!
//! Provides state management for the preview scene including
//! checkpoints for fast backward time scrubbing.

use crate::genome::Genome;
use crate::simulation::CanonicalState;
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
    pub nutrients: f32,
    pub genome_id: usize,
    pub mode_index: usize,
    pub split_interval: f32,
    pub split_nutrient_threshold: f32,
    pub stiffness: f32,
}

impl InitialState {
    /// Create initial state from genome
    pub fn from_genome(genome: &Genome, capacity: usize, physics_config: &crate::simulation::physics_config::PhysicsConfig) -> Self {
        let initial_mode_index = genome.initial_mode.max(0) as usize;
        let mode = genome.modes.get(initial_mode_index)
            .or_else(|| genome.modes.first());
        
        let (split_interval, split_mass, membrane_stiffness) = if let Some(m) = mode {
            (m.split_interval, m.split_mass, m.membrane_stiffness)
        } else {
            (5.0, 1.5, physics_config.default_stiffness)
        };

        // Convert split_mass to nutrient threshold: (split_mass - 1.0) * 100.0
        let split_nutrient_threshold = (split_mass - 1.0) * 100.0;
        
        // All cells start with full nutrients (100.0)
        let initial_nutrients = 100.0;

        Self {
            initial_cells: vec![InitialCell {
                position: glam::Vec3::ZERO,
                velocity: glam::Vec3::ZERO,
                rotation: genome.initial_orientation,
                genome_orientation: genome.initial_orientation,
                angular_velocity: glam::Vec3::ZERO,
                nutrients: initial_nutrients,
                genome_id: 0,
                mode_index: initial_mode_index,
                split_interval,
                split_nutrient_threshold,
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
                cell.nutrients,
                cell.genome_id,
                cell.mode_index,
                0.0, // birth_time
                cell.split_interval,
                cell.split_nutrient_threshold,
                cell.stiffness,
            );
        }
        
        state
    }
}

/// Preview simulation state with checkpoint support.
///
/// Uses a double-buffer approach:
/// - `work_state` / `work_time`: the in-progress resimulation buffer
/// - `display_state` / `display_time`: what the viewport renders (frozen during resim)
///
/// When resim completes, work is promoted to display. During resim the
/// viewport stays frozen on the last completed result while the yellow
/// progress bar shows `work_time` advancing toward the target.
pub struct PreviewState {
    /// Work buffer — resimulation runs physics steps on this
    pub work_state: CanonicalState,
    /// Current time of the work buffer
    pub work_time: f32,

    /// Display buffer — what the viewport renders (frozen during resim)
    pub display_state: CanonicalState,
    /// Time of the display buffer
    pub display_time: f32,

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
    
    /// Whether currently resimulating (work buffer catching up to target)
    pub is_resimulating: bool,
}

impl PreviewState {
    pub fn new(capacity: usize, physics_config: &crate::simulation::physics_config::PhysicsConfig) -> Self {
        let genome = Genome::new_with_random_colors();
        let initial_state = InitialState::from_genome(&genome, capacity, physics_config);
        let state = initial_state.to_canonical_state();
        
        Self {
            display_state: state.clone(),
            display_time: 0.0,
            work_state: state,
            work_time: 0.0,
            checkpoints: Vec::new(),
            checkpoint_interval: 1.0,
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
            // Hash child split angle orientations
            mode.child_a_after_split_orientation.x.to_bits().hash(&mut hasher);
            mode.child_a_after_split_orientation.y.to_bits().hash(&mut hasher);
            mode.child_a_after_split_orientation.z.to_bits().hash(&mut hasher);
            mode.child_a_after_split_orientation.w.to_bits().hash(&mut hasher);
            mode.child_b_after_split_orientation.x.to_bits().hash(&mut hasher);
            mode.child_b_after_split_orientation.y.to_bits().hash(&mut hasher);
            mode.child_b_after_split_orientation.z.to_bits().hash(&mut hasher);
            mode.child_b_after_split_orientation.w.to_bits().hash(&mut hasher);
            // Hash child split keep adhesion settings
            mode.child_a_after_split_keep_adhesion.hash(&mut hasher);
            mode.child_b_after_split_keep_adhesion.hash(&mut hasher);
            mode.max_adhesions.hash(&mut hasher);
            mode.min_adhesions.hash(&mut hasher);
            mode.enable_parent_angle_snapping.hash(&mut hasher);
            mode.split_ratio.to_bits().hash(&mut hasher);
            mode.nutrient_gain_rate.to_bits().hash(&mut hasher);
            mode.max_cell_size.to_bits().hash(&mut hasher);
            mode.swim_force.to_bits().hash(&mut hasher);
            mode.flagellocyte_use_signal.hash(&mut hasher);
            mode.flagellocyte_signal_channel.hash(&mut hasher);
            mode.flagellocyte_speed_a.to_bits().hash(&mut hasher);
            mode.flagellocyte_speed_b.to_bits().hash(&mut hasher);
            mode.flagellocyte_threshold_c.to_bits().hash(&mut hasher);
            mode.buoyancy_force.to_bits().hash(&mut hasher);
            mode.nutrient_priority.to_bits().hash(&mut hasher);
            mode.prioritize_when_low.hash(&mut hasher);
            mode.glueocyte_cell_adhesion.hash(&mut hasher);
            mode.glueocyte_env_adhesion.hash(&mut hasher);
            mode.glueocyte_cell_adhesion_signal_channel.hash(&mut hasher);
            mode.glueocyte_cell_adhesion_signal_threshold.to_bits().hash(&mut hasher);
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
            
            // Hash oculocyte settings
            mode.oculocyte_sense_type.hash(&mut hasher);
            mode.oculocyte_signal_channel.hash(&mut hasher);
            mode.oculocyte_signal_value.to_bits().hash(&mut hasher);
            mode.oculocyte_signal_hops.hash(&mut hasher);
            mode.oculocyte_ray_length.to_bits().hash(&mut hasher);
            
            // Hash regulation emit settings
            mode.regulation_emit_channel.hash(&mut hasher);
            mode.regulation_emit_value.to_bits().hash(&mut hasher);
            mode.regulation_emit_hops.hash(&mut hasher);
            
            // Hash signal-conditional settings
            mode.division_signal_channel.hash(&mut hasher);
            mode.division_signal_threshold.to_bits().hash(&mut hasher);
            mode.division_signal_invert.hash(&mut hasher);
            mode.apoptosis_signal_channel.hash(&mut hasher);
            mode.apoptosis_signal_threshold.to_bits().hash(&mut hasher);
            mode.apoptosis_signal_invert.hash(&mut hasher);
            mode.signal_child_a_channel.hash(&mut hasher);
            mode.signal_child_a_threshold.to_bits().hash(&mut hasher);
            mode.signal_child_a_mode_above.hash(&mut hasher);
            mode.signal_child_a_mode_below.hash(&mut hasher);
            mode.signal_child_b_channel.hash(&mut hasher);
            mode.signal_child_b_threshold.to_bits().hash(&mut hasher);
            mode.signal_child_b_mode_above.hash(&mut hasher);
            mode.signal_child_b_mode_below.hash(&mut hasher);
            mode.mode_switch_signal_channel.hash(&mut hasher);
            mode.mode_switch_signal_threshold.to_bits().hash(&mut hasher);
            mode.mode_switch_target.hash(&mut hasher);
            mode.mode_switch_invert.hash(&mut hasher);
            
            // Hash adhesion settings
            mode.adhesion_settings.can_break.hash(&mut hasher);
            mode.adhesion_settings.break_force.to_bits().hash(&mut hasher);
            mode.adhesion_settings.rest_length.to_bits().hash(&mut hasher);
            mode.adhesion_settings.linear_spring_stiffness.to_bits().hash(&mut hasher);
            mode.adhesion_settings.linear_spring_damping.to_bits().hash(&mut hasher);
            mode.adhesion_settings.orientation_spring_stiffness.to_bits().hash(&mut hasher);
            mode.adhesion_settings.orientation_spring_damping.to_bits().hash(&mut hasher);
            mode.adhesion_settings.max_angular_deviation.to_bits().hash(&mut hasher);
            mode.adhesion_settings.twist_constraint_stiffness.to_bits().hash(&mut hasher);
            mode.adhesion_settings.twist_constraint_damping.to_bits().hash(&mut hasher);
            mode.adhesion_settings.enable_twist_constraint.hash(&mut hasher);
            
            // Hash cilia settings
            mode.cilia_speed.to_bits().hash(&mut hasher);
            mode.cilia_push_bonded.hash(&mut hasher);
            mode.cilia_use_signal.hash(&mut hasher);
            mode.cilia_signal_channel.hash(&mut hasher);
            mode.cilia_speed_below.to_bits().hash(&mut hasher);
            mode.cilia_speed_above.to_bits().hash(&mut hasher);
            mode.cilia_threshold.to_bits().hash(&mut hasher);
            
            // Hash myocyte settings
            mode.myocyte_contraction.to_bits().hash(&mut hasher);
            mode.myocyte_use_signal.hash(&mut hasher);
            mode.myocyte_signal_channel.hash(&mut hasher);
            mode.myocyte_contraction_above.to_bits().hash(&mut hasher);
            mode.myocyte_contraction_below.to_bits().hash(&mut hasher);
            mode.myocyte_threshold.to_bits().hash(&mut hasher);
            mode.myocyte_pulse_rate.to_bits().hash(&mut hasher);
            mode.myocyte_pulse_phase.hash(&mut hasher);
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
    
    /// Request seeking to a specific time.
    /// Cancels any in-progress resim and starts fresh.
    pub fn seek_to_time(&mut self, target_time: f32) {
        self.target_time = Some(target_time);
        self.is_resimulating = false; // Cancel in-progress resim, force fresh setup
    }
    
    /// Update initial state when genome changes
    pub fn update_initial_state(&mut self, genome: &Genome, physics_config: &crate::simulation::physics_config::PhysicsConfig) {
        self.initial_state = InitialState::from_genome(genome, self.work_state.capacity, physics_config);
    }
    
    /// Run incremental resimulation toward target time.
    ///
    /// Runs a time-budgeted chunk of physics steps per call (~8ms) so the
    /// UI thread never blocks. The viewport stays frozen on `display_state`
    /// while `work_state` catches up. When complete, work is promoted to display.
    ///
    /// Returns true when resim is complete, false when more work remains.
    pub fn step_to(
        &mut self,
        _target_time: f32,
        genome: &Genome,
        config: &crate::simulation::physics_config::PhysicsConfig,
        test_signals: &[crate::simulation::signal_system::SignalEmission],
    ) -> bool {
        let Some(target_time) = self.target_time else {
            self.is_resimulating = false;
            return true;
        };
        
        // First call of a new resim: set up the work buffer starting point
        if !self.is_resimulating {
            self.is_resimulating = true;
            
            // For forward seeks where display is already caught up, continue from display
            if target_time > self.display_time {
                self.work_state = self.display_state.clone();
                self.work_time = self.display_time;
            } else {
                // Backward seek or genome-changed (checkpoints cleared by caller):
                // restore from best checkpoint or initial state
                let (start_time, checkpoint_state) = if let Some((t, s)) = self.find_best_checkpoint(target_time) {
                    (t, s)
                } else {
                    (0.0, self.initial_state.to_canonical_state())
                };
                self.work_state = checkpoint_state;
                self.work_time = start_time;
            }
        }
        
        // Calculate step range from work_time → target_time
        let start_step = (self.work_time / config.fixed_timestep).ceil() as u32;
        let end_step = (target_time / config.fixed_timestep).ceil() as u32;
        let remaining = end_step.saturating_sub(start_step);
        
        if remaining == 0 {
            // Already there — promote to display
            self.work_time = target_time;
            self.display_state = self.work_state.clone();
            self.display_time = target_time;
            self.target_time = None;
            self.is_resimulating = false;
            return true;
        }
        
        // Time-budgeted stepping: run for up to ~8ms per call
        let budget_ms = 8.0_f64;
        let check_interval = 50u32;
        let start_instant = std::time::Instant::now();
        let mut last_checkpoint_index = (self.work_time / self.checkpoint_interval).floor() as usize;
        let mut steps_run = 0u32;
        
        for step in 0..remaining {
            let sim_time = (start_step + step) as f32 * config.fixed_timestep;
            
            let _ = crate::simulation::preview_physics::physics_step_with_genome(
                &mut self.work_state,
                genome,
                config,
                sim_time,
                Some(test_signals),
            );
            
            steps_run += 1;
            
            // Checkpoint at intervals
            let ci = (sim_time / self.checkpoint_interval).floor() as usize;
            if ci > last_checkpoint_index {
                self.checkpoints.push((sim_time, self.work_state.clone()));
                last_checkpoint_index = ci;
            }
            
            // Check time budget periodically
            if steps_run % check_interval == 0 && step + 1 < remaining {
                if start_instant.elapsed().as_secs_f64() * 1000.0 > budget_ms {
                    self.work_time = (start_step + step + 1) as f32 * config.fixed_timestep;
                    return false; // More work next frame
                }
            }
        }
        
        // Completed — promote work to display
        self.work_time = target_time;
        self.display_state = self.work_state.clone();
        self.display_time = target_time;
        self.target_time = None;
        self.is_resimulating = false;
        true
    }

    /// Run incremental resimulation toward target time.
    /// This is a convenience method that calls step_to with the current target time.
    pub fn run_resimulation(&mut self, genome: &Genome, config: &crate::simulation::physics_config::PhysicsConfig, test_signals: &[crate::simulation::signal_system::SignalEmission]) {
        if let Some(target_time) = self.target_time {
            self.step_to(target_time, genome, config, test_signals);
        }
    }
}
