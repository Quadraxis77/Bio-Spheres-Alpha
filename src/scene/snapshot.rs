//! GPU Scene Snapshot - save and restore simulation state.
//!
//! ## What is saved
//!
//! A snapshot captures the complete live simulation state so it can be written
//! to disk and restored later.  The data falls into three categories:
//!
//! ### 1. Per-cell GPU data (requires blocking readback)
//! Positions, velocities, rotations, nutrients, genome/mode assignments, birth
//! times, split counts, cell IDs, and death flags are all GPU-only after the
//! first frame.  `save_snapshot` copies each buffer into a staging buffer and
//! maps it synchronously (one `device.poll(Wait)` call per buffer group).
//!
//! ### 2. CPU-side caches (no readback needed)
//! `AdhesionBuffers` maintains `connections_cache` and `cell_indices_cache` as
//! authoritative CPU mirrors of the adhesion GPU buffers.  These are serialised
//! directly.
//!
//! ### 3. Scalar scene settings
//! Physics parameters, time, gravity, damping, etc. are plain Rust fields on
//! `GpuScene` and are trivially serialised.
//!
//! ## What is NOT saved (acceptable loss on restore)
//! - Fluid simulator voxel state (water/lava resets)
//! - Signal flags (transient per-frame, resets to zero)
//! - `env_anchor_buffer` (glueocyte wall anchors reset)
//! - `muscle_contraction_buffer` (resets to zero, one-frame glitch)
//! - `prev_accelerations` (Verlet restarts cleanly)
//! - Spatial grid (rebuilt every frame by GPU)
//! - Organism labels (recomputed by GPU)
//!
//! ## File format
//! Snapshots are serialised as RON (Rusty Object Notation) for human
//! readability and easy debugging.  The extension is `.bss` (Bio-Spheres
//! Snapshot).

use serde::{Deserialize, Serialize};
use crate::simulation::gpu_physics::adhesion::{GpuAdhesionConnection, CellAdhesionIndices};

// --- Per-cell data ------------------------------------------------------------

/// Position and mass for one cell: `[x, y, z, mass]`
pub type CellPositionMass = [f32; 4];

/// Velocity for one cell: `[vx, vy, vz, 0]`
pub type CellVelocity = [f32; 4];

/// Rotation quaternion for one cell: `[x, y, z, w]`
pub type CellRotation = [f32; 4];

/// Genome-space orientation quaternion for one cell: `[x, y, z, w]`
pub type CellGenomeOrientation = [f32; 4];

// --- Main snapshot struct -----------------------------------------------------

/// Complete serialisable snapshot of a GPU scene.
///
/// All arrays are indexed by cell slot (0..`capacity`).  Slots where
/// `death_flags[i] == 1` are dead/free and their other data is ignored on
/// restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSceneSnapshot {
    /// Snapshot format version - bump when the layout changes.
    pub version: u32,

    // -- Capacity / counts ----------------------------------------------------
    /// Buffer capacity the snapshot was taken from.
    pub capacity: u32,
    /// Number of live cells at save time (informational).
    pub live_cell_count: u32,
    /// Total slots used (high-water mark) at save time.
    pub total_cell_slots: u32,

    // -- Per-cell arrays (length == total_cell_slots) -------------------------
    /// `[x, y, z, mass]` per cell slot.
    pub positions_and_mass: Vec<CellPositionMass>,
    /// `[vx, vy, vz, 0]` per cell slot.
    pub velocities: Vec<CellVelocity>,
    /// Physics rotation quaternion `[x, y, z, w]` per cell slot.
    pub rotations: Vec<CellRotation>,
    /// Genome-space orientation quaternion `[x, y, z, w]` per cell slot.
    pub genome_orientations: Vec<CellGenomeOrientation>,
    /// Nutrients in fixed-point i32 (value * 1000) per cell slot.
    pub nutrients: Vec<i32>,
    /// Birth time (seconds) per cell slot.
    pub birth_times: Vec<f32>,
    /// Split interval (seconds) per cell slot.
    pub split_intervals: Vec<f32>,
    /// Split nutrient threshold per cell slot.
    pub split_nutrient_thresholds: Vec<f32>,
    /// Number of times each cell has divided.
    pub split_counts: Vec<u32>,
    /// Genome ID per cell slot.
    pub genome_ids: Vec<u32>,
    /// Absolute mode index (with genome offset) per cell slot.
    pub mode_indices: Vec<u32>,
    /// Unique cell ID per cell slot.
    pub cell_ids: Vec<u32>,
    /// Death flags: `1` = dead/free slot, `0` = alive.
    pub death_flags: Vec<u32>,
    /// Embryocyte reserve (u32, max 65535) per cell slot. Zero for non-Embryocyte cells.
    #[serde(default)]
    pub embryocyte_reserves: Vec<u32>,

    // -- Adhesion state (CPU caches - no readback needed) ---------------------
    /// All adhesion connection slots (including inactive ones).
    /// Length == `capacity * MAX_ADHESIONS_PER_CELL / 2`.
    pub adhesion_connections: Vec<GpuAdhesionConnection>,
    /// Per-cell adhesion index lists.
    /// Length == `capacity`.
    pub cell_adhesion_indices: Vec<CellAdhesionIndices>,
    /// Number of adhesion slots that were allocated at save time.
    pub adhesion_allocated_count: u32,

    // -- Genomes ---------------------------------------------------------------
    /// All genomes present in the scene at save time, serialised as YAML
    /// strings using the existing genome serialization path.
    pub genomes_yaml: Vec<String>,

    // -- Scalar scene settings -------------------------------------------------
    pub current_time: f32,
    pub current_frame: i32,
    pub next_cell_id: u32,
    pub time_scale: f32,
    pub gravity: f32,
    pub gravity_mode: u32,
    pub constraint_iterations: u32,
    pub surface_pressure: f32,
    pub acceleration_damping: f32,
    pub water_viscosity: f32,
    pub solo_metabolism_multiplier: f32,
    pub radiation_level: f32,
    pub subtle_mutations: bool,
    pub lateral_flow_probabilities: [f32; 4],
    pub condensation_probability: f32,
    pub vaporization_probability: f32,
    pub nutrient_density: f32,
    pub nutrient_epoch_duration: f32,
    pub nutrient_epoch_spacing: f32,
    pub nutrient_spawn_end: f32,
    pub nutrient_despawn_start: f32,
    pub world_radius: f32,

    // -- Cave parameters -------------------------------------------------------
    /// Whether a cave system was active at save time.
    pub cave_active: bool,
    pub cave_density: f32,
    pub cave_scale: f32,
    pub cave_octaves: u32,
    pub cave_persistence: f32,
    pub cave_threshold: f32,
    pub cave_smoothness: f32,
    pub cave_seed: u32,
    pub cave_resolution: u32,

    // -- Fluid / water state ---------------------------------------------------
    /// Whether a fluid simulator was active at save time.
    pub fluid_active: bool,
    /// Fluid voxel state: one `u32` per voxel (0=Empty, 1=Water, 2=Lava, 3=Steam).
    /// Length == `TOTAL_VOXELS` (128^3 = 2,097,152) when `fluid_active` is true,
    /// empty otherwise.
    pub fluid_voxels: Vec<u32>,
    /// Nutrient voxel state: one `u32` per voxel (0=empty, 1=has nutrient).
    /// Same length as `fluid_voxels`.
    pub nutrient_voxels: Vec<u32>,
    /// Fluid simulation time at save time.
    pub fluid_time: f32,
    /// Which fluid type was selected for spawning (0=Empty, 1=Water, 2=Lava, 3=Steam).
    pub fluid_type: u32,
    /// Whether continuous spawning was enabled.
    pub fluid_continuous_spawn: bool,
}

impl GpuSceneSnapshot {
    /// Current snapshot format version.
    pub const CURRENT_VERSION: u32 = 2;

    /// File extension used for snapshot files.
    pub const FILE_EXTENSION: &'static str = "sphere";

    /// Returns `true` if this snapshot's version is compatible with the
    /// current code.
    pub fn is_compatible(&self) -> bool {
        self.version == Self::CURRENT_VERSION
    }
}

// --- Error type ---------------------------------------------------------------

/// Errors that can occur during snapshot save or restore.
#[derive(Debug)]
pub enum SnapshotError {
    /// GPU readback failed (buffer mapping error).
    GpuReadback(String),
    /// RON serialisation error.
    Serialise(ron::Error),
    /// RON deserialisation error.
    Deserialise(ron::error::SpannedError),
    /// File I/O error.
    Io(std::io::Error),
    /// Snapshot version is not compatible with this build.
    IncompatibleVersion { found: u32, expected: u32 },
    /// Snapshot capacity exceeds the current scene capacity.
    CapacityMismatch { snapshot: u32, scene: u32 },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuReadback(msg) => write!(f, "GPU readback failed: {msg}"),
            Self::Serialise(e) => write!(f, "Serialisation error: {e}"),
            Self::Deserialise(e) => write!(f, "Deserialisation error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::IncompatibleVersion { found, expected } => {
                write!(f, "Snapshot version {found} is not compatible (expected {expected})")
            }
            Self::CapacityMismatch { snapshot, scene } => {
                write!(
                    f,
                    "Snapshot capacity {snapshot} exceeds scene capacity {scene}; \
                     create a scene with at least {snapshot} cell slots before restoring"
                )
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<ron::Error> for SnapshotError {
    fn from(e: ron::Error) -> Self {
        Self::Serialise(e)
    }
}

impl From<ron::error::SpannedError> for SnapshotError {
    fn from(e: ron::error::SpannedError) -> Self {
        Self::Deserialise(e)
    }
}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
