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
//! - `split_nutrient_thresholds` - Nutrient threshold for division
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
//! - `cell_count` - Current active cells (<= capacity)
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

use crate::cell::{AdhesionConnectionManager, AdhesionConnections, MAX_ADHESIONS_PER_CELL};
use crate::genome::AdhesionSettings;
use crate::simulation::spatial_grid::DeterministicSpatialGrid;
use glam::{Quat, Vec3};

const DEVELOPMENT_ROOT_LINEAGE_HASH: u64 = 0x9E37_79B9_7F4A_7C15;
const DEVELOPMENT_ROOT_MORPHOLOGY_HASH: u64 = 0xC2B2_AE3D_27D4_EB4F;

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

/// Compact developmental identity for a cell.
///
/// The lineage is intentionally bounded: `lineage_hash` is a rolling hash scoped
/// by `organism_id`, not a stored path that grows with generation count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellDevelopmentAddress {
    pub organism_id: u32,
    pub lineage_hash: u64,
    pub morphology_hash: u64,
    pub lineage_depth: u16,
    pub branch_slot: u16,
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
    /// Number of currently active cells (<= capacity)
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

    /// Developmental organism scope for each cell.
    ///
    /// Cells with the same organism ID belong to the same reproductive individual.
    /// Lineage hashes are only meaningful inside this scope.
    pub organism_ids: Vec<u32>,

    /// Bounded rolling lineage hash for each cell, scoped by `organism_ids`.
    pub lineage_hashes: Vec<u64>,

    /// Bounded rolling morphology hash for each cell, derived from mode sequence.
    pub morphology_hashes: Vec<u64>,

    /// Saturating lineage depth for display/debug and selector specificity.
    pub lineage_depths: Vec<u16>,

    /// Last local branch slot used to derive this address.
    pub lineage_branch_slots: Vec<u16>,

    /// Lineage hash of each cell's parent at the moment it was born.
    /// Set once at division and never changed. Root cells have 0 (no parent).
    /// Used by the scaffold resolver to trace descendants through the division tree.
    pub parent_lineage_hashes: Vec<u64>,

    /// Records every division: parent_hash → (child_a_hash, child_b_hash).
    /// Never deleted — enables structural descendant search even after multiple generations
    /// have passed and intermediate ancestor cells no longer exist.
    pub division_log: std::collections::HashMap<u64, (u64, u64)>,

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
    /// Cell nutrients (0-100 normal cells, 0-200 Lipocytes)
    ///
    /// Nutrients are the primary resource for cells. Death occurs when nutrients < 1.0.
    /// Division occurs when nutrients >= split_nutrient_threshold.
    /// Lipocytes (cell_type 4) can store up to 200 nutrients.
    pub nutrients: Vec<f32>,

    /// Cell masses (derived from nutrients: mass = 1.0 + nutrients / 100.0)
    ///
    /// This is a derived value, not independent. Higher mass cells are harder
    /// to push around in collisions.
    pub masses: Vec<f32>,

    /// Visual and collision radii (derived from mass)
    ///
    /// Derived from mass: radius = mass.clamp(0.5, 2.0).
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
    /// Cells can only divide if both mass >= split_mass AND age >= split_interval.
    /// Prevents unrealistic rapid division.
    pub split_intervals: Vec<f32>,

    /// Nutrient threshold required for division
    ///
    /// Cells accumulate nutrients over time and divide when reaching this threshold
    /// (and meeting the time interval requirement).
    /// Value 0-100 for normal cells, can be up to 200 for Lipocytes.
    /// UI shows 1-101, where >100 means "never split".
    /// Derived from split_mass: threshold = (split_mass - 1.0) * 100.0
    pub split_nutrient_thresholds: Vec<f32>,

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
    /// Divides 3D space into grid cells to avoid O(n^2) collision checks.
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

    /// Next developmental organism ID to assign to standalone roots.
    pub next_organism_id: u32,

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

    // === Adhesion Break Cooldown ===
    // Prevents Glueocyte cells from immediately re-forming a bond after it breaks
    // under strain. Without this, broken bonds are instantly re-created because
    // the cells are still overlapping when form_glueocyte_contact_bonds runs.
    /// Recently broken cell pairs: (cell_id_a, cell_id_b, expiry_time)
    /// Glueocyte bond formation skips pairs that appear here until expiry.
    pub adhesion_break_cooldowns: Vec<(u32, u32, f32)>,

    // === Adhesion Expansion Tool ===
    // When active, all adhesion rest_lengths are overridden to the maximum
    // allowed value (5.0) so bonds appear fully stretched.
    // Does NOT affect the genome - purely a visual/editing aid.
    /// Whether the adhesion expansion tool is active.
    /// When true, all adhesion rest_lengths are overridden to 5.0 (the genome max)
    /// so bonds appear fully stretched. Never serialised.
    pub adhesion_expansion_active: bool,

    // === Sister Cell Immunity ===
    // Prevents Glueocyte cells from immediately bonding to their sister cell
    // after division, unless the parent had parent_make_adhesion enabled.
    /// The cell_id of this cell's sister (u32::MAX = no sister immunity)
    pub sister_cell_id: Vec<u32>,

    /// Simulation time at which sister immunity expires (0.0 = already expired)
    pub sister_expiry: Vec<f32>,

    // === Environment Adhesion ===
    // Glueocyte cells can anchor to a fixed world-space position on boundary contact.
    /// World-space anchor position for environment adhesion (only valid when env_anchor_active is true)
    pub env_anchor_pos: Vec<Vec3>,

    /// Whether this cell currently has an active environment anchor
    pub env_anchor_active: Vec<bool>,

    // === Signal System ===
    // Per-cell signal channels for oculocyte sensing and inter-cell communication.
    // 16 channels per cell. None = null (no signal), Some(v) = active signal value.
    /// Signal channels for each cell: signal_channels[cell_index * 16 + channel]
    /// None = null (no signal on this channel), Some(value) = active signal
    pub signal_channels: Vec<Option<f32>>,

    /// Whether any cell has an active signal (optimization flag to skip rendering checks)
    pub has_any_signal: bool,

    /// Per-cell muscle contraction values (0.0 = relaxed, 1.0 = fully contracted).
    /// Written by apply_myocyte_contraction each frame, read by adhesion force computation.
    /// Each cell only controls its own half of the adhesion bond.
    pub muscle_contractions: Vec<f32>,

    /// Per-cell reserve (u32, stored x1000 fixed-point; 0-65,535,000 = 0-65535 whole units).
    ///
    /// Stored in fixed-point (x1000) so sub-unit drain rates (e.g. 1/sec at 64 Hz = 0.015/tick)
    /// accumulate correctly without truncation.
    ///
    /// For Embryocytes (cell_type == 10): the reserve is the ONLY nutrient source.
    /// Incoming adhesion nutrients go here instead of the normal pool.
    /// Once detached, reserve burns at 10 units/sec until the cell dies.
    ///
    /// For non-Embryocyte cells: reserve provides a head-start buffer.
    /// Metabolism burns reserve first before touching normal nutrients.
    /// Death requires both reserve AND normal nutrients to be exhausted.
    ///
    /// Reserve is inherited at birth: child_reserve = parent_reserve >> 1 (integer halving).
    /// The initial Embryocyte placed in a scene starts with a full reserve of 65,535,000 (= 65535 whole units).
    pub reserves: Vec<u32>,

    /// Per-cell accumulation timer for Embryocyte release triggering (seconds).
    ///
    /// For Embryocytes (cell_type == 10) with at least one active adhesion:
    /// incremented by `dt` each physics step. When it reaches
    /// `mode.embryocyte_release_timer` (and `embryocyte_use_timer == true`),
    /// the timer trigger is considered satisfied.
    ///
    /// Reset to 0.0 when the Embryocyte drops all adhesions (releases).
    /// Always 0.0 for non-Embryocyte cells.
    pub embryocyte_timers: Vec<f32>,

    /// Per-cell memory state for Memorocyte cells.
    /// Holds the current integrated value of the leaky integrator.
    /// Updated each signal frame: memo = memo * decay + input * gain.
    /// Always 0.0 for non-Memorocyte cells.
    pub memo_state: Vec<f32>,

    /// Track actual signal flow paths for visualization
    /// Maps (source_cell, target_cell) -> true if signal flowed from source to target
    pub signal_flow_tracker: crate::simulation::signal_system::SignalFlowTracker,
}

impl CanonicalState {
    #[inline]
    fn mix_development_hash(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    #[inline]
    fn development_root_hash(
        seed: u64,
        organism_id: u32,
        genome_id: usize,
        mode_index: usize,
    ) -> u64 {
        Self::mix_development_hash(
            seed ^ ((organism_id as u64) << 32) ^ ((genome_id as u64) << 16) ^ mode_index as u64,
        )
    }

    #[inline]
    fn derive_development_hash(
        parent_hash: u64,
        organism_id: u32,
        parent_mode: usize,
        child_mode: usize,
        branch_slot: u16,
    ) -> u64 {
        Self::mix_development_hash(
            parent_hash
                ^ ((organism_id as u64) << 32)
                ^ ((parent_mode as u64) << 24)
                ^ ((child_mode as u64) << 8)
                ^ branch_slot as u64,
        )
    }

    pub fn development_address(&self, cell_index: usize) -> Option<CellDevelopmentAddress> {
        if cell_index >= self.cell_count {
            return None;
        }
        Some(CellDevelopmentAddress {
            organism_id: self.organism_ids[cell_index],
            lineage_hash: self.lineage_hashes[cell_index],
            morphology_hash: self.morphology_hashes[cell_index],
            lineage_depth: self.lineage_depths[cell_index],
            branch_slot: self.lineage_branch_slots[cell_index],
        })
    }

    pub(crate) fn set_development_address(
        &mut self,
        cell_index: usize,
        address: CellDevelopmentAddress,
    ) {
        if cell_index >= self.capacity {
            return;
        }
        self.organism_ids[cell_index] = address.organism_id;
        self.lineage_hashes[cell_index] = address.lineage_hash;
        self.morphology_hashes[cell_index] = address.morphology_hash;
        self.lineage_depths[cell_index] = address.lineage_depth;
        self.lineage_branch_slots[cell_index] = address.branch_slot;
        if address.organism_id >= self.next_organism_id {
            self.next_organism_id = address.organism_id.saturating_add(1);
        }
    }

    pub fn assign_development_root(
        &mut self,
        cell_index: usize,
        genome_id: usize,
        mode_index: usize,
    ) {
        let organism_id = self.next_organism_id.max(1);
        self.next_organism_id = organism_id.saturating_add(1);
        self.set_development_address(
            cell_index,
            CellDevelopmentAddress {
                organism_id,
                lineage_hash: Self::development_root_hash(
                    DEVELOPMENT_ROOT_LINEAGE_HASH,
                    organism_id,
                    genome_id,
                    mode_index,
                ),
                morphology_hash: Self::development_root_hash(
                    DEVELOPMENT_ROOT_MORPHOLOGY_HASH,
                    organism_id,
                    genome_id,
                    mode_index,
                ),
                lineage_depth: 0,
                branch_slot: 0,
            },
        );
    }

    pub fn derive_division_development_address(
        &self,
        parent_index: usize,
        parent_mode: usize,
        child_mode: usize,
        branch_slot: u16,
    ) -> CellDevelopmentAddress {
        let parent = self
            .development_address(parent_index)
            .unwrap_or(CellDevelopmentAddress {
                organism_id: 0,
                lineage_hash: DEVELOPMENT_ROOT_LINEAGE_HASH,
                morphology_hash: DEVELOPMENT_ROOT_MORPHOLOGY_HASH,
                lineage_depth: 0,
                branch_slot: 0,
            });
        CellDevelopmentAddress {
            organism_id: parent.organism_id,
            lineage_hash: Self::derive_development_hash(
                parent.lineage_hash,
                parent.organism_id,
                parent_mode,
                child_mode,
                branch_slot,
            ),
            morphology_hash: Self::derive_development_hash(
                parent.morphology_hash,
                parent.organism_id,
                parent_mode,
                child_mode,
                branch_slot,
            ),
            lineage_depth: parent.lineage_depth.saturating_add(1),
            branch_slot,
        }
    }

    /// Create a new canonical state with the specified capacity
    ///
    /// Uses a default spatial grid density of 64^3 cells, which works well
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
    /// * `grid_density` - Number of grid cells per dimension (creates density^3 total cells)
    ///
    /// # Grid Density Guidelines
    /// - 32: Small simulations (<1000 cells)
    /// - 64: Medium simulations (1000-10000 cells) - default
    /// - 128: Large simulations (>10000 cells)
    ///
    /// # Memory Usage
    /// Each grid cell stores a list of cell indices, so memory usage scales
    /// with grid_density^3. A 64^3 grid uses ~1MB for grid metadata.
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
            organism_ids: vec![0; capacity],
            lineage_hashes: vec![0; capacity],
            morphology_hashes: vec![0; capacity],
            lineage_depths: vec![0; capacity],
            lineage_branch_slots: vec![0; capacity],
            parent_lineage_hashes: vec![0; capacity],
            division_log: std::collections::HashMap::new(),

            // Position and motion arrays - initialized to zero/identity
            positions: vec![Vec3::ZERO; capacity],
            prev_positions: vec![Vec3::ZERO; capacity],
            velocities: vec![Vec3::ZERO; capacity],

            // Cell properties - reasonable defaults
            nutrients: vec![100.0; capacity], // Start with full nutrients (100.0)
            masses: vec![2.0; capacity],      // Derived: 1.0 + 100.0/100.0 = 2.0
            radii: vec![1.0; capacity],       // Derived from mass
            genome_ids: vec![0; capacity],    // All use genome 0 initially
            mode_indices: vec![0; capacity],  // All start in mode 0

            // Orientation - identity rotations, zero angular velocity
            rotations: vec![Quat::IDENTITY; capacity],
            genome_orientations: vec![Quat::IDENTITY; capacity],
            angular_velocities: vec![Vec3::ZERO; capacity],

            // Physics state - zero forces/accelerations, default stiffness
            forces: vec![Vec3::ZERO; capacity],
            torques: vec![Vec3::ZERO; capacity],
            accelerations: vec![Vec3::ZERO; capacity],
            prev_accelerations: vec![Vec3::ZERO; capacity],
            stiffnesses: vec![10.0; capacity], // Moderate stiffness

            // Division parameters - reasonable biological defaults
            birth_times: vec![0.0; capacity], // All born at time 0
            split_intervals: vec![10.0; capacity], // Divide every 10 time units
            split_nutrient_thresholds: vec![50.0; capacity], // Divide at 50 nutrients (split_mass 1.5 -> 50)
            split_counts: vec![0; capacity],                 // No divisions yet

            // Adhesion system - pre-allocated for performance
            adhesion_connections: AdhesionConnections::new(adhesion_capacity),
            adhesion_manager: AdhesionConnectionManager::new(capacity),
            cached_adhesion_settings: Vec::with_capacity(32), // Typical genome has <32 modes
            genome_modes_hash: 0,

            // Spatial grid for collision detection
            // Parameters: grid_density^3 cells, 400.0 world size, 200.0 boundary radius
            spatial_grid: DeterministicSpatialGrid::with_capacity(
                grid_density,
                400.0,
                200.0,
                capacity,
            ),

            // State tracking
            masses_changed: false,
            next_cell_id: 0,
            next_organism_id: 1,

            // Pre-allocated buffers to avoid per-frame allocations
            division_events_buffer: Vec::with_capacity(256), // Expect <256 divisions/frame
            already_split_buffer: vec![false; capacity],     // One flag per potential cell
            divisions_to_process_buffer: Vec::with_capacity(256),
            filtered_divisions_buffer: Vec::with_capacity(256),

            // Sister immunity - no immunity by default
            sister_cell_id: vec![u32::MAX; capacity],
            sister_expiry: vec![0.0; capacity],

            // Adhesion break cooldown - no cooldowns initially
            adhesion_break_cooldowns: Vec::new(),

            // Adhesion expansion tool - off by default
            adhesion_expansion_active: false,

            // Environment adhesion - no anchors by default
            env_anchor_pos: vec![Vec3::ZERO; capacity],
            env_anchor_active: vec![false; capacity],

            // Signal system - all channels null by default
            signal_channels: vec![None; capacity * 16],
            has_any_signal: false,
            muscle_contractions: vec![0.0; capacity],
            signal_flow_tracker: crate::simulation::signal_system::SignalFlowTracker::new(),

            // Reserve - zero by default; set to 65535 for initial Embryocyte cells
            reserves: vec![0u32; capacity],

            // Embryocyte accumulation timer - 0.0 for all cells initially
            embryocyte_timers: vec![0.0f32; capacity],

            // Memorocyte leaky-integrator state - 0.0 for all cells initially
            memo_state: vec![0.0f32; capacity],
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
    /// * `nutrients` - Initial nutrients (0-100 normal, 0-200 Lipocytes)
    /// * `genome_id` - Which genome this cell uses
    /// * `mode_index` - Initial behavior mode within the genome
    /// * `birth_time` - Simulation time when cell was created
    /// * `split_interval` - Minimum time between divisions
    /// * `split_nutrient_threshold` - Nutrient threshold for division
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
        nutrients: f32,
        genome_id: usize,
        mode_index: usize,
        birth_time: f32,
        split_interval: f32,
        split_nutrient_threshold: f32,
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
        self.assign_development_root(index, genome_id, mode_index);

        // Initialize all cell properties in their respective arrays
        // Position and motion
        self.positions[index] = position;
        self.prev_positions[index] = position; // Start with same position for Verlet
        self.velocities[index] = velocity;

        // Cell properties - mass and radius derived from nutrients
        self.nutrients[index] = nutrients;
        let mass = 1.0 + nutrients / 100.0;
        self.masses[index] = mass;
        self.radii[index] = mass.clamp(0.5, 2.0);
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
        self.split_nutrient_thresholds[index] = split_nutrient_threshold;
        self.split_counts[index] = 0; // New cell hasn't divided yet

        // Update counters
        self.cell_count += 1;
        self.next_cell_id += 1;

        // Initialize adhesion system for this cell
        // This sets up the data structures needed to track connections
        self.adhesion_manager.init_cell_adhesion_indices(index);

        // No sister immunity for externally-added cells
        self.sister_cell_id[index] = u32::MAX;
        self.sister_expiry[index] = 0.0;

        // No environment anchor for new cells
        self.env_anchor_pos[index] = Vec3::ZERO;
        self.env_anchor_active[index] = false;

        // Clear signal channels for new cell
        for ch in 0..16 {
            self.signal_channels[index * 16 + ch] = None;
        }

        // Reserve starts at 0; caller must set it for Embryocytes
        self.reserves[index] = 0;
        self.embryocyte_timers[index] = 0.0;
        self.memo_state[index] = 0.0;

        Some(index)
    }

    /// Add a cell at a specific slot index (for deterministic GPU synchronization)
    ///
    /// This method is used by the GPU scene to add cells at predetermined slots
    /// to maintain consistency between CPU and GPU state. Unlike `add_cell()`,
    /// this method allows specifying the exact index and cell ID.
    ///
    /// # Arguments
    /// * `slot_index` - Exact index where the cell should be placed
    /// * `position` - Initial 3D world position
    /// * `velocity` - Initial velocity vector
    /// * `rotation` - Initial orientation quaternion
    /// * `genome_orientation` - Genome-space orientation for adhesion zones
    /// * `angular_velocity` - Initial rotational velocity
    /// * `nutrients` - Initial nutrients (0-100 normal, 0-200 Lipocytes)
    /// * `genome_id` - Which genome this cell uses
    /// * `mode_index` - Current behavior mode within the genome
    /// * `birth_time` - Simulation time when cell was created
    /// * `split_interval` - Time between divisions
    /// * `split_nutrient_threshold` - Nutrient threshold for division
    /// * `stiffness` - Membrane stiffness for collision response
    /// * `cell_id` - Specific cell ID to assign (for GPU sync)
    ///
    /// # Returns
    /// * `Some(index)` - Index where the cell was placed (should equal slot_index)
    /// * `None` - If slot_index is invalid or already occupied
    ///
    /// # Safety
    /// This method assumes the caller has verified the slot is available.
    /// It will overwrite existing data at the slot index without checking.
    pub fn add_cell_at_slot(
        &mut self,
        slot_index: usize,
        position: Vec3,
        velocity: Vec3,
        rotation: Quat,
        genome_orientation: Quat,
        angular_velocity: Vec3,
        nutrients: f32,
        genome_id: usize,
        mode_index: usize,
        birth_time: f32,
        split_interval: f32,
        split_nutrient_threshold: f32,
        stiffness: f32,
        cell_id: u32,
    ) -> Option<usize> {
        // Validate slot index
        if slot_index >= self.capacity {
            return None;
        }

        // Expand cell_count if necessary to include this slot
        if slot_index >= self.cell_count {
            self.cell_count = slot_index + 1;
        }

        // Assign the specific cell ID (don't auto-increment)
        self.cell_ids[slot_index] = cell_id;
        self.assign_development_root(slot_index, genome_id, mode_index);

        // Update next_cell_id if this ID is higher
        if cell_id >= self.next_cell_id {
            self.next_cell_id = cell_id + 1;
        }

        // Initialize all cell properties at the specified slot
        // Position and motion
        self.positions[slot_index] = position;
        self.prev_positions[slot_index] = position; // Start with same position for Verlet
        self.velocities[slot_index] = velocity;

        // Cell properties - mass and radius derived from nutrients
        self.nutrients[slot_index] = nutrients;
        let mass = 1.0 + nutrients / 100.0;
        self.masses[slot_index] = mass;
        self.radii[slot_index] = mass.clamp(0.5, 2.0);
        self.genome_ids[slot_index] = genome_id;
        self.mode_indices[slot_index] = mode_index;

        // Orientation
        self.rotations[slot_index] = rotation;
        self.genome_orientations[slot_index] = genome_orientation;
        self.angular_velocities[slot_index] = angular_velocity;

        // Physics state - start with zero forces/accelerations
        self.forces[slot_index] = Vec3::ZERO;
        self.torques[slot_index] = Vec3::ZERO;
        self.accelerations[slot_index] = Vec3::ZERO;
        self.prev_accelerations[slot_index] = Vec3::ZERO;
        self.stiffnesses[slot_index] = stiffness;

        // Division parameters
        self.birth_times[slot_index] = birth_time;
        self.split_intervals[slot_index] = split_interval;
        self.split_nutrient_thresholds[slot_index] = split_nutrient_threshold;
        self.split_counts[slot_index] = 0; // New cell hasn't divided yet

        // Initialize adhesion system for this cell
        self.adhesion_manager.init_cell_adhesion_indices(slot_index);

        // No sister immunity for externally-added cells
        self.sister_cell_id[slot_index] = u32::MAX;
        self.sister_expiry[slot_index] = 0.0;

        // No environment anchor for new cells
        self.env_anchor_pos[slot_index] = Vec3::ZERO;
        self.env_anchor_active[slot_index] = false;

        // Clear signal channels for new cell
        for ch in 0..16 {
            self.signal_channels[slot_index * 16 + ch] = None;
        }

        // Reserve starts at 0; caller must set it for Embryocytes
        self.reserves[slot_index] = 0;
        self.embryocyte_timers[slot_index] = 0.0;
        self.memo_state[slot_index] = 0.0;

        Some(slot_index)
    }

    /// Remove a cell at the given index using swap-remove semantics.
    ///
    /// The last cell in the array is moved to fill the gap left by the removed cell.
    /// This maintains contiguous data without gaps while being O(1).
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell to remove
    ///
    /// # Returns
    /// * `true` - Cell was successfully removed
    /// * `false` - Index was out of bounds or no cells to remove
    ///
    /// # Side Effects
    /// - Decrements `cell_count`
    /// - If not removing the last cell, the last cell's data is moved to `cell_index`
    /// - Adhesion connections involving the removed cell are deactivated
    /// - Adhesion indices are updated for the swapped cell
    pub fn remove_cell(&mut self, cell_index: usize) -> bool {
        if cell_index >= self.cell_count || self.cell_count == 0 {
            return false;
        }

        // Remove all adhesion connections for this cell
        self.adhesion_manager
            .remove_all_connections_for_cell(&mut self.adhesion_connections, cell_index);

        let last_index = self.cell_count - 1;

        // If not removing the last cell, swap with the last cell
        if cell_index != last_index {
            // Copy all data from last cell to the removed cell's slot
            self.cell_ids[cell_index] = self.cell_ids[last_index];
            self.organism_ids[cell_index] = self.organism_ids[last_index];
            self.lineage_hashes[cell_index] = self.lineage_hashes[last_index];
            self.morphology_hashes[cell_index] = self.morphology_hashes[last_index];
            self.lineage_depths[cell_index] = self.lineage_depths[last_index];
            self.lineage_branch_slots[cell_index] = self.lineage_branch_slots[last_index];
            self.parent_lineage_hashes[cell_index] = self.parent_lineage_hashes[last_index];
            self.positions[cell_index] = self.positions[last_index];
            self.prev_positions[cell_index] = self.prev_positions[last_index];
            self.velocities[cell_index] = self.velocities[last_index];
            self.nutrients[cell_index] = self.nutrients[last_index];
            self.masses[cell_index] = self.masses[last_index];
            self.radii[cell_index] = self.radii[last_index];
            self.genome_ids[cell_index] = self.genome_ids[last_index];
            self.mode_indices[cell_index] = self.mode_indices[last_index];
            self.rotations[cell_index] = self.rotations[last_index];
            self.genome_orientations[cell_index] = self.genome_orientations[last_index];
            self.angular_velocities[cell_index] = self.angular_velocities[last_index];
            self.forces[cell_index] = self.forces[last_index];
            self.torques[cell_index] = self.torques[last_index];
            self.accelerations[cell_index] = self.accelerations[last_index];
            self.prev_accelerations[cell_index] = self.prev_accelerations[last_index];
            self.stiffnesses[cell_index] = self.stiffnesses[last_index];
            self.birth_times[cell_index] = self.birth_times[last_index];
            self.split_intervals[cell_index] = self.split_intervals[last_index];
            self.split_nutrient_thresholds[cell_index] = self.split_nutrient_thresholds[last_index];
            self.split_counts[cell_index] = self.split_counts[last_index];
            self.sister_cell_id[cell_index] = self.sister_cell_id[last_index];
            self.sister_expiry[cell_index] = self.sister_expiry[last_index];
            self.env_anchor_pos[cell_index] = self.env_anchor_pos[last_index];
            self.env_anchor_active[cell_index] = self.env_anchor_active[last_index];
            self.reserves[cell_index] = self.reserves[last_index];
            self.embryocyte_timers[cell_index] = self.embryocyte_timers[last_index];
            self.memo_state[cell_index] = self.memo_state[last_index];

            // Swap signal channels (16 channels per cell)
            for ch in 0..16 {
                self.signal_channels[cell_index * 16 + ch] =
                    self.signal_channels[last_index * 16 + ch];
            }

            // Update adhesion system for the swapped cell
            self.adhesion_manager.update_cell_index_after_swap(
                &mut self.adhesion_connections,
                last_index,
                cell_index,
            );
        }

        // Decrement cell count
        self.cell_count -= 1;

        true
    }

    /// Remove multiple cells by their indices.
    ///
    /// Cells are removed in reverse order of their indices to maintain correct
    /// swap-remove semantics (removing from highest index first prevents
    /// index invalidation issues).
    ///
    /// # Arguments
    /// * `indices` - Slice of cell indices to remove (will be sorted internally)
    ///
    /// # Returns
    /// Number of cells successfully removed
    pub fn remove_cells(&mut self, indices: &[usize]) -> usize {
        if indices.is_empty() {
            return 0;
        }

        // Sort indices in descending order to remove from end first
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sorted_indices.sort_unstable_by(|a, b| b.cmp(a));

        // Remove duplicates
        sorted_indices.dedup();

        let mut removed_count = 0;
        for &index in &sorted_indices {
            if self.remove_cell(index) {
                removed_count += 1;
            }
        }

        removed_count
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
    pub fn update_adhesion_settings_cache_multi(
        &mut self,
        genomes: &[crate::genome::Genome],
    ) -> bool {
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
        if new_hash != self.genome_modes_hash || self.cached_adhesion_settings.len() != total_modes
        {
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
                        orientation_spring_stiffness: mode
                            .adhesion_settings
                            .orientation_spring_stiffness,
                        orientation_spring_damping: mode
                            .adhesion_settings
                            .orientation_spring_damping,
                        max_angular_deviation: mode.adhesion_settings.max_angular_deviation,
                        twist_constraint_stiffness: mode
                            .adhesion_settings
                            .twist_constraint_stiffness,
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
