//! Cell type definitions for Bio-Spheres simulation.
//!
//! This module defines the different types of cells that can exist in the simulation,
//! each with their own behaviors and visual properties. The [`CellType`] enum is the
//! central registry of all available cell types.
//!
//! # Architecture Overview
//!
//! Each cell type in Bio-Spheres has three components:
//! 1. **Enum variant** - Defined in [`CellType`] with a unique index
//! 2. **Behavior module** - Implements [`CellBehavior`](crate::cell::behaviors::CellBehavior) for simulation logic
//! 3. **Appearance shader** - WGSL shader in `shaders/cells/` for rendering
//!
//! # Adding a New Cell Type
//!
//! To add a new cell type (e.g., `Flagellocyte`), follow these steps:
//!
//! ## Step 1: Update the CellType Enum
//!
//! ```ignore
//! #[repr(u32)]
//! pub enum CellType {
//!     Test = 0,
//!     Flagellocyte = 1,  // Add new variant with next index
//! }
//!
//! impl CellType {
//!     pub const COUNT: usize = 2;  // Update count
//!
//!     pub const fn all() -> &'static [CellType] {
//!         &[CellType::Test, CellType::Flagellocyte]  // Add to array
//!     }
//!
//!     pub fn from_index(index: u32) -> Option<Self> {
//!         match index {
//!             0 => Some(CellType::Test),
//!             1 => Some(CellType::Flagellocyte),  // Add match arm
//!             _ => None,
//!         }
//!     }
//!
//!     pub fn shader_path(&self) -> &'static str {
//!         match self {
//!             CellType::Test => "shaders/cells/test_cell.wgsl",
//!             CellType::Flagellocyte => "shaders/cells/flagellocyte.wgsl",  // Add path
//!         }
//!     }
//! }
//! ```
//!
//! ## Step 2: Create Behavior Module
//!
//! Create `src/cell/behaviors/flagellocyte.rs`:
//!
//! ```ignore
//! use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
//! use crate::genome::ModeSettings;
//!
//! pub struct FlagellocyteBehavior;
//!
//! impl CellBehavior for FlagellocyteBehavior {
//!     fn build_instance_data(&self, mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
//!         TypeSpecificInstanceData::flagellocyte(0.0, mode_settings.swim_force)
//!     }
//! }
//! ```
//!
//! ## Step 3: Register Behavior
//!
//! Update `src/cell/behaviors/mod.rs`:
//!
//! ```ignore
//! pub mod flagellocyte;
//!
//! pub fn create_behavior(cell_type: CellType) -> Box<dyn CellBehavior> {
//!     match cell_type {
//!         CellType::Test => Box::new(test_cell::TestCellBehavior),
//!         CellType::Flagellocyte => Box::new(flagellocyte::FlagellocyteBehavior),
//!     }
//! }
//! ```
//!
//! ## Step 4: Create Appearance Shader
//!
//! Create `shaders/cells/flagellocyte.wgsl` with the required vertex and fragment
//! entry points. Use `test_cell.wgsl` as a template.
//!
//! ## Step 5: Update Registry Shader Loading
//!
//! Update `src/cell/type_registry.rs`:
//!
//! ```ignore
//! fn load_shader_source(cell_type: CellType) -> &'static str {
//!     match cell_type {
//!         CellType::Test => include_str!("../../shaders/cells/test_cell.wgsl"),
//!         CellType::Flagellocyte => include_str!("../../shaders/cells/flagellocyte.wgsl"),
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use bytemuck::{Pod, Zeroable};

/// GPU-side behavior flags for cell types.
/// These flags replace hard-coded type checks in shaders with parameterized logic.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCellTypeBehaviorFlags {
    /// If 1, cell ignores split_interval and can divide immediately when mature
    pub ignores_split_interval: u32,
    /// If 1, cell applies swim force (flagellum propulsion)
    pub applies_swim_force: u32,
    /// If 1, cell uses texture atlas UVs (for LOD/instancing)
    pub uses_texture_atlas: u32,
    /// If 1, cell has procedural tail rendering
    pub has_procedural_tail: u32,
    /// If 1, cell gains mass from light intensity
    pub gains_mass_from_light: u32,
    /// If 1, cell is a storage specialist
    pub is_storage_cell: u32,
    /// Padding to 64 bytes for alignment
    pub _padding: [u32; 10],
}

impl Default for GpuCellTypeBehaviorFlags {
    fn default() -> Self {
        Self {
            ignores_split_interval: 0,
            applies_swim_force: 0,
            uses_texture_atlas: 0,
            has_procedural_tail: 0,
            gains_mass_from_light: 0,
            is_storage_cell: 0,
            _padding: [0; 10],
        }
    }
}

/// Visual settings for a cell type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct CellTypeVisuals {
    pub specular_strength: f32,
    pub specular_power: f32,
    pub fresnel_strength: f32,
    /// Scale of the membrane noise pattern (higher = finer detail)
    pub membrane_noise_scale: f32,
    /// Strength of the membrane noise normal perturbation
    pub membrane_noise_strength: f32,
    /// Animation speed of the membrane noise (0 = static)
    pub membrane_noise_speed: f32,
    
    // Flagella parameters (used by Flagellocyte cell type)
    /// Length of the flagellum tail (0.5 - 3.0, default 1.7)
    pub tail_length: f32,
    /// Thickness of the flagellum at the base (0.01 - 0.3, default 0.15)
    pub tail_thickness: f32,
    /// Amplitude of the helical wave motion (0.0 - 0.5, default 0.17)
    pub tail_amplitude: f32,
    /// Frequency of the helical wave (0.5 - 10.0, default 1.0)
    pub tail_frequency: f32,
    /// Taper factor from base to tip (0.0 - 1.0, default 1.0)
    pub tail_taper: f32,
    /// Number of segments used to render the tail (4 - 64, default 10)
    pub tail_segments: f32,

    // Geodesic ridge parameters (used by Photocyte cell type membrane)
    /// Icosphere subdivision level (1-6, default 3)
    pub goldberg_scale: f32,
    /// Gaussian half-width of the ridge line (0.01-0.5, default 0.12)
    pub goldberg_ridge_width: f32,
    /// How far edges meander organically (0.0-0.3, default 0.08)
    pub goldberg_meander: f32,
    /// How much normals deflect at ridges for 3D depth (0.0-0.5, default 0.15)
    pub goldberg_ridge_strength: f32,
    /// Nucleus sphere scale relative to cell radius (0.0-0.95, default 0.6)
    pub nucleus_scale: f32,
}

impl Default for CellTypeVisuals {
    fn default() -> Self {
        Self {
            specular_strength: 0.3,
            specular_power: 32.0,
            fresnel_strength: 0.2,
            membrane_noise_scale: 0.0,        // Disabled by default
            membrane_noise_strength: 0.0,     // Disabled by default
            membrane_noise_speed: 0.0,        // Disabled by default
            // Flagella defaults (from reference implementation)
            tail_length: 1.7,
            tail_thickness: 0.15,
            tail_amplitude: 0.17,
            tail_frequency: 1.0,
            tail_taper: 1.0,
            tail_segments: 10.0,
            // Geodesic ridge defaults
            goldberg_scale: 3.0,
            goldberg_ridge_width: 0.12,
            goldberg_meander: 0.08,
            goldberg_ridge_strength: 0.15,
            nucleus_scale: 0.6,
        }
    }
}

/// Persistent storage for cell type visuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellTypeVisualsStore {
    /// Visual settings indexed by cell type
    pub visuals: Vec<CellTypeVisuals>,
}

impl Default for CellTypeVisualsStore {
    fn default() -> Self {
        Self {
            visuals: CellType::all()
                .iter()
                .map(|_| CellTypeVisuals::default())
                .collect(),
        }
    }
}

impl CellTypeVisualsStore {
    const FILE_PATH: &'static str = "cell_visuals.ron";

    /// Save cell type visuals to disk.
    pub fn save(visuals: &[CellTypeVisuals]) -> Result<(), CellVisualsError> {
        let store = CellTypeVisualsStore {
            visuals: visuals.to_vec(),
        };
        let path = PathBuf::from(Self::FILE_PATH);
        let contents = ron::ser::to_string_pretty(&store, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        log::info!("Saved cell type visuals to {}", Self::FILE_PATH);
        Ok(())
    }

    /// Load cell type visuals from disk, or return defaults if file doesn't exist.
    pub fn load() -> Vec<CellTypeVisuals> {
        let path = PathBuf::from(Self::FILE_PATH);
        
        if path.exists() {
            match Self::load_from_file(&path) {
                Ok(store) => {
                    log::info!("Loaded cell type visuals from {:?}", path);
                    // Ensure we have visuals for all cell types
                    let mut visuals = store.visuals;
                    while visuals.len() < CellType::all().len() {
                        visuals.push(CellTypeVisuals::default());
                    }
                    return visuals;
                }
                Err(e) => {
                    log::warn!("Failed to load cell type visuals: {}. Using defaults.", e);
                }
            }
        }
        
        Self::default().visuals
    }

    fn load_from_file(path: &PathBuf) -> Result<Self, CellVisualsError> {
        let contents = std::fs::read_to_string(path)?;
        let store: Self = ron::from_str(&contents)?;
        Ok(store)
    }
}

/// Error type for cell visuals persistence.
#[derive(Debug, thiserror::Error)]
pub enum CellVisualsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON parse error: {0}")]
    RonParse(#[from] ron::error::SpannedError),
    #[error("RON serialize error: {0}")]
    RonSerialize(#[from] ron::Error),
}

/// Cell type enumeration - each variant corresponds to a unique
/// appearance shader and behavior module.
///
/// `CellType` is the central registry of all cell types in Bio-Spheres. Each variant
/// represents a distinct type of cell with its own:
/// - **Visual appearance** - Defined by a WGSL shader in `shaders/cells/`
/// - **Simulation behavior** - Implemented via [`CellBehavior`](crate::cell::behaviors::CellBehavior)
/// - **Mode settings** - Configured in the genome editor
///
/// # Memory Layout
///
/// The enum uses `#[repr(u32)]` to ensure a stable memory layout for GPU compatibility.
/// Each variant's discriminant is its index in the registry.
///
/// # Extension Guide
///
/// To add a new cell type, you need to modify this enum and create two new files:
///
/// 1. **Add enum variant** with the next available index
/// 2. **Update `COUNT`** to reflect the new total
/// 3. **Add to `all()`** array
/// 4. **Add match arms** in `from_index()`, `name()`, `names()`, and `shader_path()`
/// 5. **Create behavior module** in `src/cell/behaviors/{type_name}.rs`
/// 6. **Create appearance shader** in `shaders/cells/{type_name}.wgsl`
/// 7. **Register behavior** in `src/cell/behaviors/mod.rs`
/// 8. **Update shader loading** in `src/cell/type_registry.rs`
///
/// See the module-level documentation for detailed examples.
///
/// # Example
///
/// ```
/// use biospheres::cell::CellType;
///
/// // Get all cell types
/// for cell_type in CellType::iter() {
///     println!("{}: {}", cell_type.to_index(), cell_type.name());
/// }
///
/// // Convert from index (e.g., from GPU buffer)
/// if let Some(cell_type) = CellType::from_index(0) {
///     assert_eq!(cell_type, CellType::Test);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum CellType {
    Test = 0,
    Flagellocyte = 1,
    Phagocyte = 2,
    Photocyte = 3,
    Lipocyte = 4,
}

impl CellType {
    /// Number of registered cell types. Update when adding new types.
    pub const COUNT: usize = 5;

    /// Maximum number of cell types supported by GPU buffers.
    pub const MAX_TYPES: usize = 30;

    /// Get all available cell types as a slice.
    pub const fn all() -> &'static [CellType] {
        &[
            CellType::Test,
            CellType::Flagellocyte,
            CellType::Phagocyte,
            CellType::Photocyte,
            CellType::Lipocyte,
        ]
    }

    /// Iterator over all cell types for registry initialization.
    pub fn iter() -> impl Iterator<Item = Self> {
        (0..Self::COUNT as u32).filter_map(Self::from_index)
    }

    /// Get the display name for this cell type.
    pub const fn name(&self) -> &'static str {
        match self {
            CellType::Test => "Test",
            CellType::Flagellocyte => "Flagellocyte",
            CellType::Phagocyte => "Phagocyte",
            CellType::Photocyte => "Photocyte",
            CellType::Lipocyte => "Lipocyte",
        }
    }

    /// Get all cell type names as a slice.
    pub const fn names() -> &'static [&'static str] {
        &["Test", "Flagellocyte", "Phagocyte", "Photocyte", "Lipocyte"]
    }

    /// Convert from integer index to cell type.
    pub fn from_index(index: u32) -> Option<Self> {
        match index {
            0 => Some(CellType::Test),
            1 => Some(CellType::Flagellocyte),
            2 => Some(CellType::Phagocyte),
            3 => Some(CellType::Photocyte),
            4 => Some(CellType::Lipocyte),
            _ => None,
        }
    }

    /// Convert to integer index.
    pub const fn to_index(&self) -> u32 {
        *self as u32
    }

    /// Get the path to the appearance shader for this cell type.
    /// All cell types now use the unified shader.
    pub fn shader_path(&self) -> &'static str {
        // All cell types use the unified 3-layer procedural shader
        "shaders/cells/cell_unified.wgsl"
    }

    /// Get GPU behavior flags for this cell type.
    /// These flags control type-specific behavior in shaders.
    pub fn behavior_flags(&self) -> GpuCellTypeBehaviorFlags {
        match self {
            CellType::Test => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                _padding: [0; 10],
            },
            CellType::Flagellocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 1,
                applies_swim_force: 1,
                uses_texture_atlas: 0,
                has_procedural_tail: 1,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                _padding: [0; 10],
            },
            CellType::Phagocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 1,  // Phagocytes use texture atlas like Test cells
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                _padding: [0; 10],
            },
            CellType::Photocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 1,
                is_storage_cell: 0,
                _padding: [0; 10],
            },
            CellType::Lipocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 1,
                _padding: [0; 10],
            },
        }
    }
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Test
    }
}
