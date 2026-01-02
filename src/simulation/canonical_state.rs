use glam::{Vec3, Quat};
use crate::simulation::cpu_physics::DeterministicSpatialGrid;
use crate::cell::{AdhesionConnections, AdhesionConnectionManager, MAX_ADHESIONS_PER_CELL};
use crate::genome::AdhesionSettings;

/// Event describing a cell division that occurred
#[derive(Debug, Clone)]
pub struct DivisionEvent {
    pub parent_idx: usize,
    pub child_a_idx: usize,
    pub child_b_idx: usize,
}

/// Canonical simulation state using Structure-of-Arrays (SoA) layout
/// Pure data structure with no ECS dependencies
#[derive(Clone)]
pub struct CanonicalState {
    /// Number of active cells
    pub cell_count: usize,
    
    /// Maximum capacity
    pub capacity: usize,
    
    /// Cell unique IDs
    pub cell_ids: Vec<u32>,
    
    // Position and Motion (SoA)
    pub positions: Vec<Vec3>,
    pub prev_positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    
    // Cell Properties (SoA)
    pub masses: Vec<f32>,
    pub radii: Vec<f32>,
    pub genome_ids: Vec<usize>,
    pub mode_indices: Vec<usize>,
    
    // Orientation (SoA)
    pub rotations: Vec<Quat>,
    pub genome_orientations: Vec<Quat>, // Genome-space orientations (separate from physics)
    pub angular_velocities: Vec<Vec3>,
    
    // Physics State (SoA)
    pub forces: Vec<Vec3>,
    pub torques: Vec<Vec3>,
    pub accelerations: Vec<Vec3>,
    pub prev_accelerations: Vec<Vec3>,
    pub stiffnesses: Vec<f32>,
    
    // Division Timers (SoA)
    pub birth_times: Vec<f32>,
    pub split_intervals: Vec<f32>,
    pub split_masses: Vec<f32>,
    pub split_counts: Vec<i32>,
    
    // === Adhesion System ===
    /// Adhesion connections between cells
    pub adhesion_connections: AdhesionConnections,
    /// Adhesion connection manager
    pub adhesion_manager: AdhesionConnectionManager,
    /// Cached adhesion settings from genome (rebuilt when genome changes)
    pub cached_adhesion_settings: Vec<AdhesionSettings>,
    /// Hash of genome modes to detect changes
    pub genome_modes_hash: u64,
    
    /// Spatial partitioning for collision detection
    pub spatial_grid: DeterministicSpatialGrid,
    
    /// Next cell ID to assign
    pub next_cell_id: u32,
    
    // Division processing buffers (pre-allocated to avoid per-frame allocations)
    pub division_events_buffer: Vec<DivisionEvent>,
    pub already_split_buffer: Vec<bool>,
    pub divisions_to_process_buffer: Vec<usize>,
    pub filtered_divisions_buffer: Vec<usize>,
}

impl CanonicalState {
    /// Create a new canonical state with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self::with_grid_density(capacity, 64) // Default 64x64x64 grid
    }
    
    /// Create a new canonical state with specified capacity and grid density
    pub fn with_grid_density(capacity: usize, grid_density: u32) -> Self {
        // Calculate adhesion connection capacity (20 connections per cell)
        let adhesion_capacity = capacity * MAX_ADHESIONS_PER_CELL;
        
        Self {
            cell_count: 0,
            capacity,
            cell_ids: vec![0; capacity],
            positions: vec![Vec3::ZERO; capacity],
            prev_positions: vec![Vec3::ZERO; capacity],
            velocities: vec![Vec3::ZERO; capacity],
            masses: vec![1.0; capacity],
            radii: vec![1.0; capacity],
            genome_ids: vec![0; capacity],
            mode_indices: vec![0; capacity],
            rotations: vec![Quat::IDENTITY; capacity],
            genome_orientations: vec![Quat::IDENTITY; capacity],
            angular_velocities: vec![Vec3::ZERO; capacity],
            forces: vec![Vec3::ZERO; capacity],
            torques: vec![Vec3::ZERO; capacity],
            accelerations: vec![Vec3::ZERO; capacity],
            prev_accelerations: vec![Vec3::ZERO; capacity],
            stiffnesses: vec![10.0; capacity],
            birth_times: vec![0.0; capacity],
            split_intervals: vec![10.0; capacity],
            split_masses: vec![1.5; capacity],
            split_counts: vec![0; capacity],
            adhesion_connections: AdhesionConnections::new(adhesion_capacity),
            adhesion_manager: AdhesionConnectionManager::new(capacity),
            cached_adhesion_settings: Vec::with_capacity(32), // Typical genome has <32 modes
            genome_modes_hash: 0,
            spatial_grid: DeterministicSpatialGrid::with_capacity(grid_density, 200.0, 100.0, capacity),
            next_cell_id: 0,
            division_events_buffer: Vec::with_capacity(256),
            already_split_buffer: vec![false; capacity],
            divisions_to_process_buffer: Vec::with_capacity(256),
            filtered_divisions_buffer: Vec::with_capacity(256),
        }
    }
    
    /// Add a new cell to the canonical state
    pub fn add_cell(
        &mut self,
        position: Vec3,
        velocity: Vec3,
        rotation: Quat,
        genome_orientation: Quat,
        angular_velocity: Vec3,
        mass: f32,
        radius: f32,
        genome_id: usize,
        mode_index: usize,
        birth_time: f32,
        split_interval: f32,
        split_mass: f32,
        stiffness: f32,
    ) -> Option<usize> {
        if self.cell_count >= self.capacity {
            return None;
        }
        
        let index = self.cell_count;
        self.cell_ids[index] = self.next_cell_id;
        self.positions[index] = position;
        self.prev_positions[index] = position;
        self.velocities[index] = velocity;
        self.masses[index] = mass;
        self.radii[index] = radius;
        self.genome_ids[index] = genome_id;
        self.mode_indices[index] = mode_index;
        self.rotations[index] = rotation;
        self.genome_orientations[index] = genome_orientation;
        self.angular_velocities[index] = angular_velocity;
        self.forces[index] = Vec3::ZERO;
        self.torques[index] = Vec3::ZERO;
        self.accelerations[index] = Vec3::ZERO;
        self.prev_accelerations[index] = Vec3::ZERO;
        self.stiffnesses[index] = stiffness;
        self.birth_times[index] = birth_time;
        self.split_intervals[index] = split_interval;
        self.split_masses[index] = split_mass;
        self.split_counts[index] = 0;
        
        self.cell_count += 1;
        self.next_cell_id += 1;
        
        // Initialize adhesion indices for new cell
        self.adhesion_manager.init_cell_adhesion_indices(index);
        
        Some(index)
    }
    
    /// Update cached adhesion settings from genome if needed
    /// Returns true if cache was updated
    pub fn update_adhesion_settings_cache(&mut self, genome: &crate::genome::Genome) -> bool {
        // Simple hash based on mode count and first mode's stiffness
        // This catches most genome changes without expensive full comparison
        let new_hash = (genome.modes.len() as u64) << 32 
            | (genome.modes.first().map(|m| (m.adhesion_settings.linear_spring_stiffness * 1000.0) as u64).unwrap_or(0));
        
        if new_hash != self.genome_modes_hash || self.cached_adhesion_settings.len() != genome.modes.len() {
            self.cached_adhesion_settings.clear();
            for mode in &genome.modes {
                self.cached_adhesion_settings.push(AdhesionSettings {
                    can_break: mode.adhesion_settings.can_break,
                    break_force: mode.adhesion_settings.break_force,
                    rest_length: mode.adhesion_settings.rest_length,
                    linear_spring_stiffness: mode.adhesion_settings.linear_spring_stiffness,
                    linear_spring_damping: mode.adhesion_settings.linear_spring_damping,
                    orientation_spring_stiffness: mode.adhesion_settings.orientation_spring_stiffness,
                    orientation_spring_damping: mode.adhesion_settings.orientation_spring_damping,
                    max_angular_deviation: mode.adhesion_settings.max_angular_deviation,
                    twist_constraint_stiffness: mode.adhesion_settings.twist_constraint_stiffness,
                    twist_constraint_damping: mode.adhesion_settings.twist_constraint_damping,
                    enable_twist_constraint: mode.adhesion_settings.enable_twist_constraint,
                });
            }
            self.genome_modes_hash = new_hash;
            return true;
        }
        false
    }
}
