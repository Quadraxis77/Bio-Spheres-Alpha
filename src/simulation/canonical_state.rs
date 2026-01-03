//! # Canonical Simulation State - Structure-of-Arrays Layout
//! 
//! This module contains the [`CanonicalState`] struct, which is the central data structure
//! for the entire Bio-Spheres simulation. It uses a Structure-of-Arrays (SoA) layout for
//! optimal cache performance when iterating over large numbers of cells.
//! 
//! ## Design Philosophy
//! 
//! ### Structure-of-Arrays (SoA) vs Array-of-Structures (AoS)
//! 
//! Traditional object-oriented design would use Array-of-Structures:
//! ```rust
//! // AoS - Poor cache performance for bulk operations
//! struct Cell {
//!     position: Vec3,
//!     velocity: Vec3,
//!     mass: f32,
//!     // ... 20+ fields
//! }
//! let cells: Vec<Cell> = vec![...];
//! ```
//! 
//! Bio-Spheres uses Structure-of-Arrays for better performance:
//! ```rust
//! // SoA - Excellent cache performance for bulk operations
//! struct CanonicalState {
//!     positions: Vec<Vec3>,    // All positions together
//!     velocities: Vec<Vec3>,   // All velocities together
//!     masses: Vec<f32>,        // All masses together
//!     // ...
//! }
//! ```
//! 
//! ### Performance Benefits
//! 
//! - **Cache Efficiency**: When updating positions, only position data is loaded into cache
//! - **SIMD Friendly**: Contiguous arrays enable vectorized operations
//! - **GPU Compatible**: Matches GPU compute shader memory access patterns
//! - **Parallel Processing**: Different systems can work on different arrays simultaneously
//! 
//! ## Data Organization
//! 
//! The [`CanonicalState`] organizes cell data into logical groups:
//! 
//! ### Position and Motion
//! - `positions` - Current 3D positions
//! - `prev_positions` - Previous frame positions (for Verlet integration)
//! - `velocities` - Current velocities
//! 
//! ### Cell Properties
//! - `masses` - Cell masses (affects division timing and physics)
//! - `radii` - Visual and collision radii
//! - `genome_ids` - Which genome this cell uses
//! - `mode_indices` - Current behavior mode within the genome
//! 
//! ### Orientation System
//! - `rotations` - Physics-based rotations (from angular velocity)
//! - `genome_orientations` - Genome-space orientations (for adhesion zones)
//! - `angular_velocities` - Rotational motion
//! 
//! ### Physics State
//! - `forces` - Accumulated forces (gravity, collisions, adhesion)
//! - `torques` - Rotational forces
//! - `accelerations` - Current frame accelerations
//! - `prev_accelerations` - Previous frame (for integration schemes)
//! 
//! ### Division System
//! - `birth_times` - When each cell was created
//! - `split_intervals` - How often cells divide
//! - `split_masses` - Mass threshold for division
//! - `split_counts` - Number of times each cell has divided
//! 
//! ### Adhesion System
//! - `adhesion_connections` - Cell-cell connection data (SoA layout)
//! - `adhesion_manager` - Manages connection indices per cell
//! - `cached_adhesion_settings` - Genome adhesion parameters (cached for performance)
//! 
//! ## Memory Layout Considerations
//! 
//! ### Capacity vs Count
//! - `capacity` - Maximum number of cells (fixed at creation)
//! - `cell_count` - Current active cells (≤ capacity)
//! - All arrays are pre-allocated to `capacity` to avoid runtime allocations
//! 
//! ### Index Stability
//! - Cell indices remain stable during simulation
//! - Removed cells leave gaps (filled by swapping with last cell)
//! - `cell_ids` provide unique identifiers that persist across index changes
//! 
//! ## Usage Patterns
//! 
//! ### Adding Cells
//! ```rust
//! let idx = state.add_cell(
//!     position, velocity, rotation, genome_orientation,
//!     angular_velocity, mass, radius, genome_id, mode_index,
//!     birth_time, split_interval, split_mass, stiffness
//! )?;
//! ```
//! 
//! ### Bulk Operations (Physics)
//! ```rust
//! // Update all positions - cache-friendly iteration
//! for i in 0..state.cell_count {
//!     state.positions[i] += state.velocities[i] * dt;
//! }
//! ```
//! 
//! ### Parallel Processing
//! ```rust
//! use rayon::prelude::*;
//! 
//! // Parallel force calculation
//! state.forces[0..state.cell_count]
//!     .par_iter_mut()
//!     .enumerate()
//!     .for_each(|(i, force)| {
//!         *force = calculate_forces(i, &state);
//!     });
//! ```
//! 
//! ## Integration with Other Systems
//! 
//! ### Physics Systems
//! - [`cpu_physics`] reads/writes position, velocity, force arrays
//! - [`gpu_physics`] uploads arrays to GPU compute buffers
//! 
//! ### Rendering
//! - [`rendering::cells`] reads position, radius, rotation for instancing
//! - [`rendering::adhesion_lines`] reads adhesion connections for visualization
//! 
//! ### Genome System
//! - Genome changes trigger cache updates via `update_adhesion_settings_cache()`
//! - Mode transitions update `mode_indices` array
//! 
//! ## Performance Optimization
//! 
//! ### Pre-allocation Strategy
//! - All vectors allocated to full capacity at creation
//! - Division buffers pre-allocated to avoid per-frame allocations
//! - Spatial grid pre-allocated with expected cell density
//! 
//! ### Cache-Friendly Access
//! - Systems process one array at a time (not jumping between arrays)
//! - Related data grouped together (position/velocity, force/acceleration)
//! - Hot data (positions, velocities) separated from cold data (birth_times)
//! 
//! ### Memory Alignment
//! - Uses `glam` types (Vec3, Quat) which are SIMD-aligned
//! - Arrays naturally aligned for vectorized operations

use glam::{Vec3, Quat};
use crate::simulation::cpu_physics::DeterministicSpatialGrid;
use crate::cell::{AdhesionConnections, AdhesionConnectionManager, MAX_ADHESIONS_PER_CELL};
use crate::genome::AdhesionSettings;

/// Event describing a cell division that occurred during physics simulation
/// 
/// Division events are collected during the physics step and processed afterwards
/// to maintain stable indices during force calculations.
#[derive(Debug, Clone)]
pub struct DivisionEvent {
    /// Index of the parent cell that divided
    pub parent_idx: usize,
    /// Index of the first child cell created
    pub child_a_idx: usize,
    /// Index of the second child cell created
    pub child_b_idx: usize,
}

/// Central simulation state using Structure-of-Arrays (SoA) layout for optimal performance
/// 
/// This struct contains all cell data organized into separate arrays for each property.
/// This layout provides excellent cache performance when iterating over large numbers
/// of cells, as systems only load the data they need into cache.
/// 
/// ## Memory Layout
/// 
/// All arrays are pre-allocated to `capacity` size to avoid runtime allocations.
/// Only the first `cell_count` elements of each array contain valid data.
/// 
/// ## Thread Safety
/// 
/// This struct is not thread-safe by itself, but different systems can safely
/// read/write different arrays in parallel (e.g., one thread updating positions
/// while another calculates forces).
/// 
/// ## Index Stability
/// 
/// Cell indices remain stable during simulation. When cells are removed, the
/// last cell is swapped into the removed cell's position to maintain contiguous
/// data without gaps.
#[derive(Clone)]
pub struct CanonicalState {
    /// Number of currently active cells (≤ capacity)
    /// 
    /// This is the authoritative count - only indices 0..cell_count contain valid data.
    pub cell_count: usize,
    
    /// Maximum number of cells this state can hold
    /// 
    /// All arrays are pre-allocated to this size. Cannot be changed after creation
    /// to avoid expensive reallocations during simulation.
    pub capacity: usize,
    
    /// Unique identifier for each cell (stable across index changes)
    /// 
    /// When cells are removed and indices are reused, cell_ids provide a way
    /// to track individual cells across their lifetime.
    pub cell_ids: Vec<u32>,
    
    // === Position and Motion (SoA) ===
    // These arrays are accessed together during physics integration
    
    /// Current 3D world positions of all cells
    /// 
    /// Updated each physics step. Used by rendering for instance positions.
    pub positions: Vec<Vec3>,
    
    /// Previous frame positions for Verlet integration
    /// 
    /// Verlet integration uses position history instead of explicit velocities
    /// for better numerical stability in collision-heavy scenarios.
    pub prev_positions: Vec<Vec3>,
    
    /// Current velocities (may be derived from position differences)
    /// 
    /// Used for rendering motion blur and physics calculations that need velocity.
    pub velocities: Vec<Vec3>,
    
    // === Cell Properties (SoA) ===
    // These define the biological and physical characteristics of each cell
    
    /// Cell masses (affects division timing and collision response)
    /// 
    /// Cells accumulate mass over time and divide when reaching `split_masses[i]`.
    /// Higher mass cells are harder to push around in collisions.
    pub masses: Vec<f32>,
    
    /// Visual and collision radii
    /// 
    /// Typically derived from mass via cube root relationship (volume ∝ mass).
    /// Used for rendering billboard size and collision detection.
    pub radii: Vec<f32>,
    
    /// Which genome each cell uses (index into genome array)
    /// 
    /// Multiple genomes can coexist in the same simulation. This determines
    /// which genome's behavior rules apply to each cell.
    pub genome_ids: Vec<usize>,
    
    /// Current behavior mode within the genome
    /// 
    /// Genomes define multiple modes (e.g., "growing", "dividing", "quiescent").
    /// This tracks which mode each cell is currently in.
    pub mode_indices: Vec<usize>,
    
    // === Orientation System (SoA) ===
    // Dual orientation system: physics-based and genome-based
    
    /// Physics-driven rotations from angular velocity integration
    /// 
    /// Updated by physics simulation based on torques and angular velocity.
    /// Used for rendering cell orientation.
    pub rotations: Vec<Quat>,
    
    /// Genome-space orientations for adhesion zone calculations
    /// 
    /// Independent of physics rotation. Defines how the cell's "genome coordinate
    /// system" is oriented, which determines adhesion zone positions.
    pub genome_orientations: Vec<Quat>,
    
    /// Rotational velocities (radians per second)
    /// 
    /// Integrated to update `rotations`. Generated by torques from adhesion
    /// constraints and collision responses.
    pub angular_velocities: Vec<Vec3>,
    
    // === Physics State (SoA) ===
    // Force accumulation and integration data
    
    /// Accumulated forces for current frame (reset each physics step)
    /// 
    /// Sum of gravity, collision forces, adhesion forces, and boundary forces.
    /// Used to calculate accelerations via F = ma.
    pub forces: Vec<Vec3>,
    
    /// Accumulated torques for current frame (reset each physics step)
    /// 
    /// Rotational forces from adhesion constraints and collision responses.
    /// Used to calculate angular accelerations.
    pub torques: Vec<Vec3>,
    
    /// Current frame accelerations (forces / mass)
    /// 
    /// Calculated from forces and used for velocity integration.
    pub accelerations: Vec<Vec3>,
    
    /// Previous frame accelerations for higher-order integration schemes
    /// 
    /// Some integration methods (like Verlet) benefit from acceleration history.
    pub prev_accelerations: Vec<Vec3>,
    
    /// Cell membrane stiffness (resistance to deformation)
    /// 
    /// Higher values make cells more rigid in collisions. Typically set
    /// from genome mode parameters.
    pub stiffnesses: Vec<f32>,
    
    // === Division System (SoA) ===
    // Biological reproduction timing and parameters
    
    /// Simulation time when each cell was created
    /// 
    /// Used to calculate cell age and determine division timing.
    pub birth_times: Vec<f32>,
    
    /// Time interval between divisions (when mass threshold is met)
    /// 
    /// Cells can only divide if both mass ≥ split_mass AND age ≥ split_interval.
    /// Prevents unrealistic rapid division.
    pub split_intervals: Vec<f32>,
    
    /// Mass threshold required for division
    /// 
    /// Cells accumulate mass over time and divide when reaching this threshold
    /// (and meeting the time interval requirement).
    pub split_masses: Vec<f32>,
    
    /// Number of times each cell has divided
    /// 
    /// Used for lineage tracking and potentially limiting division depth.
    pub split_counts: Vec<i32>,
    
    // === Adhesion System ===
    // Cell-cell connection data using SoA layout for performance
    
    /// All adhesion connections in the simulation (SoA layout)
    /// 
    /// Contains arrays of cell pairs, connection strengths, rest lengths, etc.
    /// Organized as Structure-of-Arrays for cache-friendly iteration.
    pub adhesion_connections: AdhesionConnections,
    
    /// Manages which connections belong to each cell
    /// 
    /// Provides fast lookup of all connections for a given cell index.
    /// Essential for force calculation and connection management.
    pub adhesion_manager: AdhesionConnectionManager,
    
    /// Cached adhesion parameters from genome modes
    /// 
    /// Genome mode adhesion settings are cached here to avoid repeated
    /// hash map lookups during physics calculations. Updated when genome changes.
    pub cached_adhesion_settings: Vec<AdhesionSettings>,
    
    /// Hash of genome modes to detect when cache needs updating
    /// 
    /// When this changes, `cached_adhesion_settings` is rebuilt from the genome.
    pub genome_modes_hash: u64,
    
    /// Spatial partitioning grid for efficient collision detection
    /// 
    /// Divides 3D space into grid cells to avoid O(n²) collision checks.
    /// Only cells in the same or adjacent grid cells need collision testing.
    pub spatial_grid: DeterministicSpatialGrid,
    
    /// Flag indicating masses have changed (triggers radius recalculation)
    /// 
    /// Set to true when masses are modified. Physics systems can check this
    /// to know when to recalculate derived properties like radii.
    pub masses_changed: bool,
    
    /// Next unique cell ID to assign
    /// 
    /// Incremented each time a cell is added. Provides stable identifiers
    /// that persist even when cell indices are reused.
    pub next_cell_id: u32,
    
    // === Division Processing Buffers ===
    // Pre-allocated buffers to avoid per-frame allocations during division processing
    
    /// Buffer for collecting division events during physics step
    /// 
    /// Pre-allocated to avoid allocations during simulation. Cleared each frame.
    pub division_events_buffer: Vec<DivisionEvent>,
    
    /// Buffer tracking which cells have already split this frame
    /// 
    /// Prevents cells from dividing multiple times in a single physics step.
    pub already_split_buffer: Vec<bool>,
    
    /// Buffer for cells that want to divide this frame
    /// 
    /// Intermediate buffer used during division processing.
    pub divisions_to_process_buffer: Vec<usize>,
    
    /// Buffer for filtered divisions after conflict resolution
    /// 
    /// Final list of divisions that will actually be processed.
    pub filtered_divisions_buffer: Vec<usize>,
}

impl CanonicalState {
    /// Create a new canonical state with the specified capacity
    /// 
    /// Uses a default spatial grid density of 64³ cells, which works well
    /// for most simulation scenarios.
    /// 
    /// # Arguments
    /// * `capacity` - Maximum number of cells this state can hold
    /// 
    /// # Performance Notes
    /// All arrays are pre-allocated to `capacity` size to avoid runtime
    /// allocations during simulation. Choose capacity based on expected
    /// maximum cell count, not initial count.
    pub fn new(capacity: usize) -> Self {
        Self::with_grid_density(capacity, 64) // Default 64x64x64 grid
    }
    
    /// Create a new canonical state with specified capacity and spatial grid density
    /// 
    /// The spatial grid is used for collision detection optimization. Higher density
    /// grids provide better culling but use more memory. Choose based on simulation
    /// scale and cell density.
    /// 
    /// # Arguments
    /// * `capacity` - Maximum number of cells this state can hold
    /// * `grid_density` - Number of grid cells per dimension (creates density³ total cells)
    /// 
    /// # Grid Density Guidelines
    /// - 32: Small simulations (<1000 cells)
    /// - 64: Medium simulations (1000-10000 cells) - default
    /// - 128: Large simulations (>10000 cells)
    /// 
    /// # Memory Usage
    /// Each grid cell stores a list of cell indices, so memory usage scales
    /// with grid_density³. A 64³ grid uses ~1MB for grid metadata.
    pub fn with_grid_density(capacity: usize, grid_density: u32) -> Self {
        // Calculate adhesion connection capacity
        // Each cell can have up to MAX_ADHESIONS_PER_CELL connections
        let adhesion_capacity = capacity * MAX_ADHESIONS_PER_CELL;
        
        Self {
            cell_count: 0,
            capacity,
            
            // Initialize all arrays to capacity size with default values
            // This avoids allocations during simulation
            cell_ids: vec![0; capacity],
            
            // Position and motion arrays - initialized to zero/identity
            positions: vec![Vec3::ZERO; capacity],
            prev_positions: vec![Vec3::ZERO; capacity],
            velocities: vec![Vec3::ZERO; capacity],
            
            // Cell properties - reasonable defaults
            masses: vec![1.0; capacity],           // 1.0 unit mass
            radii: vec![1.0; capacity],            // 1.0 unit radius
            genome_ids: vec![0; capacity],         // All use genome 0 initially
            mode_indices: vec![0; capacity],       // All start in mode 0
            
            // Orientation - identity rotations, zero angular velocity
            rotations: vec![Quat::IDENTITY; capacity],
            genome_orientations: vec![Quat::IDENTITY; capacity],
            angular_velocities: vec![Vec3::ZERO; capacity],
            
            // Physics state - zero forces/accelerations, default stiffness
            forces: vec![Vec3::ZERO; capacity],
            torques: vec![Vec3::ZERO; capacity],
            accelerations: vec![Vec3::ZERO; capacity],
            prev_accelerations: vec![Vec3::ZERO; capacity],
            stiffnesses: vec![10.0; capacity],     // Moderate stiffness
            
            // Division parameters - reasonable biological defaults
            birth_times: vec![0.0; capacity],      // All born at time 0
            split_intervals: vec![10.0; capacity], // Divide every 10 time units
            split_masses: vec![1.5; capacity],     // Divide at 1.5x base mass
            split_counts: vec![0; capacity],       // No divisions yet
            
            // Adhesion system - pre-allocated for performance
            adhesion_connections: AdhesionConnections::new(adhesion_capacity),
            adhesion_manager: AdhesionConnectionManager::new(capacity),
            cached_adhesion_settings: Vec::with_capacity(32), // Typical genome has <32 modes
            genome_modes_hash: 0,
            
            // Spatial grid for collision detection
            // Parameters: grid_density³ cells, 200.0 world size, 100.0 boundary radius
            spatial_grid: DeterministicSpatialGrid::with_capacity(grid_density, 200.0, 100.0, capacity),
            
            // State tracking
            masses_changed: false,
            next_cell_id: 0,
            
            // Pre-allocated buffers to avoid per-frame allocations
            division_events_buffer: Vec::with_capacity(256),    // Expect <256 divisions/frame
            already_split_buffer: vec![false; capacity],        // One flag per potential cell
            divisions_to_process_buffer: Vec::with_capacity(256),
            filtered_divisions_buffer: Vec::with_capacity(256),
        }
    }
    
    /// Add a new cell to the canonical state
    /// 
    /// This is the primary way to create new cells in the simulation. All cell
    /// properties must be specified at creation time.
    /// 
    /// # Arguments
    /// * `position` - Initial 3D world position
    /// * `velocity` - Initial velocity vector
    /// * `rotation` - Initial physics-based rotation
    /// * `genome_orientation` - Initial genome-space orientation (for adhesion zones)
    /// * `angular_velocity` - Initial rotational velocity
    /// * `mass` - Initial mass (affects division timing and physics)
    /// * `radius` - Visual and collision radius
    /// * `genome_id` - Which genome this cell uses
    /// * `mode_index` - Initial behavior mode within the genome
    /// * `birth_time` - Simulation time when cell was created
    /// * `split_interval` - Minimum time between divisions
    /// * `split_mass` - Mass threshold for division
    /// * `stiffness` - Membrane stiffness for collision response
    /// 
    /// # Returns
    /// * `Some(index)` - Index of the newly created cell
    /// * `None` - If at capacity, no cell was created
    /// 
    /// # Performance Notes
    /// This operation is O(1) and does not require any memory allocation.
    /// The cell is added at index `cell_count` and `cell_count` is incremented.
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
        // Check capacity - cannot add more cells than allocated
        if self.cell_count >= self.capacity {
            return None;
        }
        
        // Add cell at the next available index
        let index = self.cell_count;
        
        // Assign unique cell ID and increment for next cell
        self.cell_ids[index] = self.next_cell_id;
        
        // Initialize all cell properties in their respective arrays
        // Position and motion
        self.positions[index] = position;
        self.prev_positions[index] = position;  // Start with same position for Verlet
        self.velocities[index] = velocity;
        
        // Cell properties
        self.masses[index] = mass;
        self.radii[index] = radius;
        self.genome_ids[index] = genome_id;
        self.mode_indices[index] = mode_index;
        
        // Orientation
        self.rotations[index] = rotation;
        self.genome_orientations[index] = genome_orientation;
        self.angular_velocities[index] = angular_velocity;
        
        // Physics state - start with zero forces/accelerations
        self.forces[index] = Vec3::ZERO;
        self.torques[index] = Vec3::ZERO;
        self.accelerations[index] = Vec3::ZERO;
        self.prev_accelerations[index] = Vec3::ZERO;
        self.stiffnesses[index] = stiffness;
        
        // Division parameters
        self.birth_times[index] = birth_time;
        self.split_intervals[index] = split_interval;
        self.split_masses[index] = split_mass;
        self.split_counts[index] = 0;  // New cell hasn't divided yet
        
        // Update counters
        self.cell_count += 1;
        self.next_cell_id += 1;
        
        // Initialize adhesion system for this cell
        // This sets up the data structures needed to track connections
        self.adhesion_manager.init_cell_adhesion_indices(index);
        
        Some(index)
    }
    
    /// Update cached adhesion settings from genome if needed (single genome version)
    /// 
    /// This is a convenience method for simulations using a single genome.
    /// For multi-genome simulations, use `update_adhesion_settings_cache_multi()`.
    /// 
    /// # Arguments
    /// * `genome` - The genome to cache settings from
    /// 
    /// # Returns
    /// * `true` - Cache was updated (genome changed)
    /// * `false` - Cache was already up-to-date
    /// 
    /// # Performance Notes
    /// This method uses a hash to detect genome changes and only rebuilds the
    /// cache when necessary. The hash includes mode count and key parameters.
    pub fn update_adhesion_settings_cache(&mut self, genome: &crate::genome::Genome) -> bool {
        self.update_adhesion_settings_cache_multi(&[genome.clone()])
    }
    
    /// Update cached adhesion settings from multiple genomes
    /// 
    /// Caches adhesion parameters from all genome modes to avoid repeated lookups
    /// during physics calculations. Mode indices are global across all genomes:
    /// - Genome 0 modes: 0 to (genome[0].modes.len() - 1)
    /// - Genome 1 modes: genome[0].modes.len() to (genome[0].modes.len() + genome[1].modes.len() - 1)
    /// - etc.
    /// 
    /// # Arguments
    /// * `genomes` - Array of all genomes in the simulation
    /// 
    /// # Returns
    /// * `true` - Cache was updated (genomes changed)
    /// * `false` - Cache was already up-to-date
    /// 
    /// # Cache Invalidation
    /// The cache is invalidated when:
    /// - Number of genomes changes
    /// - Total number of modes changes
    /// - Key adhesion parameters change (detected via hash)
    /// 
    /// # Performance Impact
    /// Cache hits avoid hash map lookups during force calculations, providing
    /// significant performance benefits for large simulations.
    pub fn update_adhesion_settings_cache_multi(&mut self, genomes: &[crate::genome::Genome]) -> bool {
        // Calculate total mode count across all genomes
        let total_modes: usize = genomes.iter().map(|g| g.modes.len()).sum();
        
        // Create hash from genome structure and key parameters
        // High bits: genome count, mode count
        // Low bits: XOR of key parameters from first mode of each genome
        let mut new_hash = (genomes.len() as u64) << 48 | (total_modes as u64) << 32;
        for genome in genomes {
            if let Some(m) = genome.modes.first() {
                // Hash key adhesion parameter (scaled to avoid float precision issues)
                new_hash ^= (m.adhesion_settings.linear_spring_stiffness * 1000.0) as u64;
            }
        }
        
        // Check if cache needs updating
        if new_hash != self.genome_modes_hash || self.cached_adhesion_settings.len() != total_modes {
            // Rebuild cache from all genome modes
            self.cached_adhesion_settings.clear();
            for genome in genomes {
                for mode in &genome.modes {
                    // Copy all adhesion settings for fast access during physics
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
            }
            self.genome_modes_hash = new_hash;
            return true;
        }
        false
    }

    /// Update cell membrane stiffness from genome modes (single genome version)
    /// 
    /// This is a convenience method for simulations using a single genome.
    /// Updates the `stiffnesses` array based on each cell's current mode.
    /// 
    /// # Arguments
    /// * `genome` - The genome to read stiffness values from
    /// 
    /// # When to Call
    /// Call this when:
    /// - Genome mode parameters change
    /// - Cells transition between modes
    /// - Loading a new genome into an existing simulation
    pub fn update_membrane_stiffness_from_genome(&mut self, genome: &crate::genome::Genome) {
        self.update_membrane_stiffness_from_genomes(&[genome.clone()])
    }

    /// Update cell membrane stiffness from multiple genomes
    /// 
    /// Updates the `stiffnesses` array for all cells based on their current
    /// genome and mode. This should be called when genome parameters change
    /// to keep existing cells synchronized with their genome definitions.
    /// 
    /// # Arguments
    /// * `genomes` - Array of all genomes in the simulation
    /// 
    /// # Mode Index Mapping
    /// Mode indices are global across all genomes:
    /// - Genome 0 modes: 0 to (genome[0].modes.len() - 1)
    /// - Genome 1 modes: genome[0].modes.len() to (genome[0].modes.len() + genome[1].modes.len() - 1)
    /// - etc.
    /// 
    /// # Performance Notes
    /// This is an O(n) operation over all active cells. It's typically called
    /// infrequently (when genomes change), not every frame.
    pub fn update_membrane_stiffness_from_genomes(&mut self, genomes: &[crate::genome::Genome]) {
        // Update stiffness for all active cells
        for cell_idx in 0..self.cell_count {
            let genome_id = self.genome_ids[cell_idx] as usize;
            let mode_index = self.mode_indices[cell_idx] as usize;
            
            // Look up the genome and mode for this cell
            if let Some(genome) = genomes.get(genome_id) {
                if let Some(mode) = genome.modes.get(mode_index) {
                    // Update stiffness from genome mode
                    self.stiffnesses[cell_idx] = mode.membrane_stiffness;
                }
            }
        }
    }
}
