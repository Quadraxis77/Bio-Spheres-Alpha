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

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    /// If 1, cell applies upward buoyancy force
    pub applies_buoyancy: u32,
    /// If 1, cell applies cilia contact force
    pub applies_cilia_force: u32,
    /// If 1, cell applies muscle contraction to adhesion rest lengths
    pub applies_muscle_contraction: u32,
    /// Padding to 64 bytes for alignment
    pub _padding: [u32; 7],
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
            applies_buoyancy: 0,
            applies_cilia_force: 0,
            applies_muscle_contraction: 0,
            _padding: [0; 7],
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

    // Cilia ring parameters (used by Ciliocyte cell type)
    /// Number of visible cilia rings along forward axis (2.0-12.0, default 5.0)
    pub cilia_ring_frequency: f32,
    /// Normal perturbation strength for cilia rings (0.0-0.3, default 0.1)
    pub cilia_ring_depth: f32,
    /// Base animation scroll rate for cilia rings (1.0-10.0, default 4.0)
    pub cilia_ring_speed: f32,

    // Generic type params - packed into type_data_0.xyzw for types using the default branch.
    // Interpretation depends on cell type (see build_instances.wgsl for mapping):
    //   Test:       ring_frequency, ring_sharpness, ring_brightness, unused
    //   Phagocyte:  nucleus_radius, nucleus_darkness, nucleus_sharpness, unused
    //   Lipocyte:   droplet_scale, droplet_threshold, boundary_sharpness, brightness
    //   Buoyocyte:  bubble_scale, rotation_speed, wall_brightness, gas_brightness
    //   Devorocyte: spike_height, spike_sharpness(cos), spike_embed, tip_fade
    /// First type-specific parameter (x of type_data_0)
    pub param_a: f32,
    /// Second type-specific parameter (y of type_data_0)
    pub param_b: f32,
    /// Third type-specific parameter (z of type_data_0)
    pub param_c: f32,
    /// Fourth type-specific parameter (w of type_data_0)
    pub param_d: f32,
}

impl Default for CellTypeVisuals {
    fn default() -> Self {
        Self {
            specular_strength: 0.3,
            specular_power: 32.0,
            fresnel_strength: 0.2,
            membrane_noise_scale: 0.0,    // Disabled by default
            membrane_noise_strength: 0.0, // Disabled by default
            membrane_noise_speed: 0.0,    // Disabled by default
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
            // Cilia ring defaults
            cilia_ring_frequency: 5.0,
            cilia_ring_depth: 0.1,
            cilia_ring_speed: 4.0,
            // Generic type params - defaults match old hardcoded shader values per type.
            // Since one default covers all types, we use neutral/zero values here.
            // Per-type defaults are set in CellTypeVisualsStore::default_for_type().
            param_a: 0.0,
            param_b: 0.0,
            param_c: 0.0,
            param_d: 0.0,
        }
    }
}

impl CellTypeVisuals {
    /// Return sensible defaults for a specific cell type, matching the old hardcoded shader values.
    pub fn default_for_type(cell_type: CellType) -> Self {
        let mut v = Self::default();
        match cell_type {
            CellType::Test => {
                // ring_frequency=0 -> plain sphere (matches old no-pattern default)
                v.param_a = 0.0; // ring_frequency (0 = off)
                v.param_b = 0.3; // ring_sharpness
                v.param_c = 0.0; // ring_brightness (0 = off)
                v.param_d = 0.0;
            }
            CellType::Flagellocyte => {
                v.specular_strength = 0.48;
                v.fresnel_strength = 0.2;
                v.tail_length = 2.02;
                v.tail_thickness = 0.226;
                v.tail_amplitude = 0.17;
                v.tail_frequency = 1.0;
                v.tail_taper = 1.0;
                v.tail_segments = 10.0;
            }
            CellType::Phagocyte => {
                // nucleus_radius=0.3, nucleus_darkness=0.4, nucleus_sharpness=0.05
                v.param_a = 0.3;
                v.param_b = 0.4;
                v.param_c = 0.05;
                v.param_d = 0.0;
            }
            CellType::Photocyte => {
                // Geodesic hex membrane ridges
                v.specular_strength = 0.17;
                v.specular_power = 128.0;
                v.fresnel_strength = 0.2;
                v.goldberg_scale = 4.0; // subdivision level
                v.goldberg_ridge_width = 0.01; // very thin ridges
                v.goldberg_meander = 0.0; // clean straight ridges
                v.goldberg_ridge_strength = 0.5;
                v.nucleus_scale = 0.9; // large inner nucleus sphere
            }
            CellType::Lipocyte => {
                // droplet_scale=3.0, droplet_threshold=0.35, boundary_sharpness=0.15, brightness=1.0
                v.param_a = 3.0;
                v.param_b = 0.35;
                v.param_c = 0.15;
                v.param_d = 1.0;
            }
            CellType::Buoyocyte => {
                // bubble_scale=1.0, rotation_speed=1.0, wall_brightness=1.0, gas_brightness=1.0
                v.param_a = 1.0;
                v.param_b = 1.0;
                v.param_c = 1.0;
                v.param_d = 1.0;
            }
            CellType::Glueocyte => {
                // Voronoi slime pattern
                v.goldberg_scale = 3.0; // voro_scale
                v.goldberg_ridge_width = 0.12; // border_width
                v.goldberg_meander = 0.08; // meander
                v.goldberg_ridge_strength = 0.15; // border_dark
                v.membrane_noise_speed = 2.85; // anim_speed
                v.membrane_noise_scale = 0.0;
                v.membrane_noise_strength = 0.0;
                v.specular_strength = 1.0;
                v.specular_power = 96.0;
                v.fresnel_strength = 0.7;
            }
            CellType::Oculocyte => {
                // Eye: pupil_size, iris_freq, iris_texture, pupil_dark
                // goldberg_scale -> pupil_size (type_data_0.x)
                // goldberg_ridge_width -> iris_freq (type_data_0.y)
                // goldberg_meander -> iris_texture (type_data_0.z)
                // goldberg_ridge_strength -> pupil_dark (type_data_0.w)
                v.goldberg_scale = 0.25; // pupil_size
                v.goldberg_ridge_width = 8.0; // iris_freq (striation count)
                v.goldberg_meander = 0.3; // iris_texture blend
                v.goldberg_ridge_strength = 0.85; // pupil_dark
                v.specular_strength = 0.5;
                v.specular_power = 36.0;
                v.fresnel_strength = 0.3;
                v.nucleus_scale = 0.6;
            }
            CellType::Ciliocyte => {
                // Cilia rings
                v.cilia_ring_frequency = 10.0;
                v.cilia_ring_depth = 0.45;
                v.cilia_ring_speed = 4.0;
                v.specular_strength = 0.5;
                v.specular_power = 36.0;
                v.fresnel_strength = 0.3;
            }
            CellType::Myocyte => {
                // Muscle fibers: goldberg_scale=line_freq, goldberg_ridge_width=bulge_strength,
                // goldberg_meander=warp_amt (type_data_0.x/y/z)
                v.goldberg_scale = 8.0; // line_freq (fiber count)
                v.goldberg_ridge_width = 0.55; // bulge_strength
                v.goldberg_meander = 0.12; // warp_amt
                v.goldberg_ridge_strength = 0.0; // unused
                v.specular_strength = 0.45;
                v.specular_power = 48.0;
                v.fresnel_strength = 0.25;
                v.nucleus_scale = 0.6;
            }
            CellType::Embryocyte => {
                // Egg yolk: goldberg_scale=yolk_radius, goldberg_ridge_width=yolk_offset_y,
                // goldberg_meander=yolk_brightness (type_data_0.x/y/z)
                // Note: yolk_offset_y is negative in shader (clamped to -0.35..0.0),
                // stored as positive in UI and negated when packed.
                v.goldberg_scale = 0.5; // yolk_radius
                v.goldberg_ridge_width = 0.15; // yolk_drop (stored positive, negated in shader)
                v.goldberg_meander = 1.0; // yolk_brightness
                v.goldberg_ridge_strength = 0.0; // unused
                v.specular_strength = 0.3;
                v.specular_power = 32.0;
                v.fresnel_strength = 0.2;
                v.nucleus_scale = 0.6;
            }
            CellType::Devorocyte => {
                // spike_height=0.75, spike_sharpness=0.985, spike_embed=0.12, tip_fade=1.0
                v.param_a = 0.75;
                v.param_b = 0.985;
                v.param_c = 0.12;
                v.param_d = 1.0;
                v.specular_strength = 0.55;
                v.specular_power = 52.0;
                v.fresnel_strength = 0.35;
            }
            CellType::Vasculocyte => {
                // Vessel wall cobblestone: goldberg_scale=cell_scale, goldberg_ridge_width=border_width,
                // goldberg_meander=meander, goldberg_ridge_strength=border_depth (type_data_0.x/y/z/w)
                v.goldberg_scale = 8.0; // cell_scale
                v.goldberg_ridge_width = 0.68; // border_width
                v.goldberg_meander = 0.12; // meander
                v.goldberg_ridge_strength = 0.09; // border_depth
                v.specular_strength = 0.3;
                v.specular_power = 32.0;
                v.fresnel_strength = 0.2;
                v.nucleus_scale = 0.6;
            }
            CellType::Gametocyte => {
                // Soft, semi-translucent membrane with a glowing central nucleus.
                // param_a = pulse_speed (pulsing glow animation rate)
                // param_b = nucleus_glow (brightness of inner nucleus)
                // param_c = membrane_translucency (0=opaque, 1=translucent)
                v.param_a = 1.5; // pulse_speed
                v.param_b = 1.2; // nucleus_glow
                v.param_c = 0.6; // membrane_translucency
                v.param_d = 0.0;
                v.specular_strength = 0.6;
                v.specular_power = 24.0;
                v.fresnel_strength = 0.5;
                v.nucleus_scale = 0.55;
                v.membrane_noise_scale = 2.5;
                v.membrane_noise_strength = 0.08;
                v.membrane_noise_speed = 1.5;
            }
            CellType::Cognocyte => {
                // Clean, geometric look: tight Goldberg tessellation to evoke a circuit-like surface.
                v.goldberg_scale = 12.0;
                v.goldberg_ridge_width = 0.55;
                v.goldberg_meander = 0.05;
                v.goldberg_ridge_strength = 0.12;
                v.specular_strength = 0.5;
                v.specular_power = 48.0;
                v.fresnel_strength = 0.3;
                v.nucleus_scale = 0.4;
            }
            CellType::Memorocyte => {
                // Glassy, semi-translucent with concentric ring layering.
                v.specular_strength = 0.6;
                v.specular_power = 32.0;
                v.fresnel_strength = 0.55;
                v.nucleus_scale = 0.5;
                v.membrane_noise_scale = 3.0;
                v.membrane_noise_strength = 0.04;
                v.membrane_noise_speed = 0.3;
            }
        }
        v
    }
}

/// Persistent storage for cell type visuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellTypeVisualsStore {
    /// Visual settings indexed by cell type
    pub visuals: Vec<CellTypeVisuals>,
    // Cell outline width for cel-shaded black outline effect (0.0 = off, 0.15 = thick)
    #[serde(default)]
    pub cell_outline_width: f32,
}

impl Default for CellTypeVisualsStore {
    fn default() -> Self {
        Self {
            visuals: CellType::all()
                .iter()
                .map(|ct| CellTypeVisuals::default_for_type(*ct))
                .collect(),
            cell_outline_width: 0.0,
        }
    }
}

impl CellTypeVisualsStore {
    const FILE_NAME: &'static str = "cell_visuals.ron";

    /// Save cell type visuals to disk.
    pub fn save(
        visuals: &[CellTypeVisuals],
        cell_outline_width: f32,
    ) -> Result<(), CellVisualsError> {
        let store = CellTypeVisualsStore {
            visuals: visuals.to_vec(),
            cell_outline_width,
        };
        let path = crate::app_dirs::config_file(Self::FILE_NAME);
        let contents = ron::ser::to_string_pretty(&store, ron::ser::PrettyConfig::default())?;
        std::fs::write(&path, contents)?;
        log::info!("Saved cell type visuals to {:?}", path);
        Ok(())
    }

    /// Load cell type visuals from disk, or return defaults if file doesn't exist.
    pub fn load() -> (Vec<CellTypeVisuals>, f32) {
        let path = crate::app_dirs::config_file(Self::FILE_NAME);

        if path.exists() {
            match Self::load_from_file(&path) {
                Ok(store) => {
                    log::info!("Loaded cell type visuals from {:?}", path);
                    // Ensure we have visuals for all cell types
                    let mut visuals = store.visuals;
                    let all_types = CellType::all();
                    while visuals.len() < all_types.len() {
                        let ct = all_types[visuals.len()];
                        visuals.push(CellTypeVisuals::default_for_type(ct));
                    }
                    return (visuals, store.cell_outline_width);
                }
                Err(e) => {
                    log::warn!("Failed to load cell type visuals: {}. Using defaults.", e);
                }
            }
        }

        let default = Self::default();
        (default.visuals, default.cell_outline_width)
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
    Buoyocyte = 5,
    Glueocyte = 6,
    Oculocyte = 7,
    Ciliocyte = 8,
    Myocyte = 9,
    Embryocyte = 10,
    Devorocyte = 11,
    Vasculocyte = 12,
    Gametocyte = 13,
    Cognocyte = 14,
    Memorocyte = 15,
}

impl CellType {
    /// Number of registered cell types. Update when adding new types.
    pub const COUNT: usize = 16;

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
            CellType::Buoyocyte,
            CellType::Glueocyte,
            CellType::Oculocyte,
            CellType::Ciliocyte,
            CellType::Myocyte,
            CellType::Embryocyte,
            CellType::Devorocyte,
            CellType::Vasculocyte,
            CellType::Gametocyte,
            CellType::Cognocyte,
            CellType::Memorocyte,
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
            CellType::Buoyocyte => "Buoyocyte",
            CellType::Glueocyte => "Glueocyte",
            CellType::Oculocyte => "Oculocyte",
            CellType::Ciliocyte => "Ciliocyte",
            CellType::Myocyte => "Myocyte",
            CellType::Embryocyte => "Embryocyte",
            CellType::Devorocyte => "Devorocyte",
            CellType::Vasculocyte => "Vasculocyte",
            CellType::Gametocyte => "Gametocyte",
            CellType::Cognocyte => "Cognocyte",
            CellType::Memorocyte => "Memorocyte",
        }
    }

    /// Get all cell type names as a slice.
    pub const fn names() -> &'static [&'static str] {
        &[
            "Test",
            "Flagellocyte",
            "Phagocyte",
            "Photocyte",
            "Lipocyte",
            "Buoyocyte",
            "Glueocyte",
            "Oculocyte",
            "Ciliocyte",
            "Myocyte",
            "Embryocyte",
            "Devorocyte",
            "Vasculocyte",
            "Gametocyte",
            "Cognocyte",
            "Memorocyte",
        ]
    }

    /// Convert from integer index to cell type.
    pub fn from_index(index: u32) -> Option<Self> {
        match index {
            0 => Some(CellType::Test),
            1 => Some(CellType::Flagellocyte),
            2 => Some(CellType::Phagocyte),
            3 => Some(CellType::Photocyte),
            4 => Some(CellType::Lipocyte),
            5 => Some(CellType::Buoyocyte),
            6 => Some(CellType::Glueocyte),
            7 => Some(CellType::Oculocyte),
            8 => Some(CellType::Ciliocyte),
            9 => Some(CellType::Myocyte),
            10 => Some(CellType::Embryocyte),
            11 => Some(CellType::Devorocyte),
            12 => Some(CellType::Vasculocyte),
            13 => Some(CellType::Gametocyte),
            14 => Some(CellType::Cognocyte),
            15 => Some(CellType::Memorocyte),
            _ => None,
        }
    }

    /// Convert to integer index.
    pub const fn to_index(&self) -> u32 {
        *self as u32
    }

    /// A short description of what this cell type does, shown as a tooltip
    /// in the genome editor's Type dropdown.
    pub const fn tooltip(&self) -> &'static str {
        match self {
            CellType::Test => {
                "Gains nutrients automatically at a fixed rate. Useful for quick \
                 prototyping — no food source needed."
            }

            CellType::Flagellocyte => {
                "Propels the organism using a whip-like flagellum tail. Swim force \
                 can be fixed or switched between two speeds via an oculocyte signal."
            }

            CellType::Phagocyte => {
                "Absorbs free-floating nutrient particles from the environment on \
                 contact. The primary food-gathering cell in nutrient-rich worlds."
            }

            CellType::Photocyte => {
                "Converts light into nutrients — self-sustaining near a light source. \
                 Nutrient gain scales with the local light field intensity."
            }

            CellType::Lipocyte => {
                "Stores surplus nutrients as fat reserves, acting as a buffer during \
                 food shortages. Excess mass is donated to hungry neighbours."
            }

            CellType::Buoyocyte => {
                "Generates an upward buoyancy force, counteracting gravity and keeping \
                 the organism neutrally buoyant or floating near the surface."
            }

            CellType::Glueocyte => {
                "Bonds to other cells or cave walls on contact. Useful for anchoring \
                 an organism to a surface or building sticky capture traps."
            }

            CellType::Oculocyte => {
                "Casts a detection ray to sense cells, food particles, light, or \
                 barriers. Broadcasts a numeric signal through the adhesion network \
                 when a target is detected."
            }

            CellType::Ciliocyte => {
                "Uses rows of cilia to push nearby cells and fluid in the \
                 forward direction. Good for conveying particles along a chain or \
                 creating internal circulation."
            }

            CellType::Myocyte => {
                "A muscle cell that rhythmically contracts its adhesion bonds. \
                 Contractions can be timer-driven or signal-gated, producing \
                 peristaltic pumping or coordinated movement."
            }

            CellType::Embryocyte => {
                "Incubates a fully-formed sub-organism and releases it when triggered \
                 by a timer, a nutrient threshold, or an incoming signal. Enables \
                 complex reproductive strategies."
            }

            CellType::Devorocyte => {
                "Aggressively steals nutrients from neighbouring foreign cells within \
                 contact range. A predatory cell type — effective against slow or \
                 undefended organisms."
            }

            CellType::Vasculocyte => {
                "Efficiently transports nutrients through the organism body along \
                 adhesion pathways. Acts as a sealed pipe by default; set Outlet to \
                 release nutrients to non-vascular neighbours."
            }

            CellType::Gametocyte => {
                "A reproductive gamete cell. When two Gametocytes from different \
                 organisms come into contact, their genomes are crossed over and a \
                 new hybrid offspring organism is spawned. Gametocytes never split; \
                 they only detach and merge. Both Gametocytes then die."
            }

            CellType::Cognocyte => {
                "A signal-processing cell. Reads signals from two input channels, \
                 applies an arithmetic, comparison, or boolean operation, and emits \
                 the result on an output channel. Composable into arbitrarily complex \
                 decision-making circuits within an organism."
            }

            CellType::Memorocyte => {
                "A leaky-integrator memory cell. Accumulates incoming signals over \
                 time and slowly forgets them. Decay and gain are independently \
                 configurable. Useful for smoothing noisy sensors, building timers, \
                 and creating hysteresis in decision circuits."
            }
        }
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
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Flagellocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0, // Respect split interval
                applies_swim_force: 1,
                uses_texture_atlas: 0,
                has_procedural_tail: 1,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Phagocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Photocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 1,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Lipocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 1,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Buoyocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 1,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Glueocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Oculocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Ciliocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 1,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Myocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 1,
                _padding: [0; 7],
            },
            CellType::Embryocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 1, // Uses reserve as internal storage
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Devorocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Vasculocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Gametocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 1, // Nutrients go into reserve, same as Embryocyte
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Cognocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
            CellType::Memorocyte => GpuCellTypeBehaviorFlags {
                ignores_split_interval: 0,
                applies_swim_force: 0,
                uses_texture_atlas: 0,
                has_procedural_tail: 0,
                gains_mass_from_light: 0,
                is_storage_cell: 0,
                applies_buoyancy: 0,
                applies_cilia_force: 0,
                applies_muscle_contraction: 0,
                _padding: [0; 7],
            },
        }
    }

    /// Apply type-specific default overrides to a ModeSettings.
    /// Called when the user changes a mode's cell type in the UI.
    pub fn apply_mode_defaults(&self, mode: &mut crate::genome::ModeSettings) {
        match self {
            CellType::Lipocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
            }
            CellType::Buoyocyte => {
                mode.max_cell_size = 2.0;
                mode.buoyancy_force = 0.5;
            }
            CellType::Glueocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
            }
            CellType::Oculocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
            }
            CellType::Ciliocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
            }
            CellType::Myocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
                mode.myocyte_contraction = 0.5;
            }
            CellType::Embryocyte => {
                // Embryocytes live off their reserve, not normal nutrients.
                // Use a very high split_mass so they never split on nutrients alone.
                // Division is gated by reserve > 0 and no adhesion connections.
                mode.split_mass = 2.0;
                mode.max_cell_size = 2.0;
                mode.nutrient_priority = 1.0;
                // Enable timer trigger by default (release after 10 seconds)
                mode.embryocyte_use_timer = true;
                mode.embryocyte_release_timer = 10.0;
            }
            CellType::Devorocyte => {
                mode.nutrient_priority = 1.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
                mode.devorocyte_consume_range = 0.5;
                mode.devorocyte_consume_rate = 30.0;
            }
            CellType::Vasculocyte => {
                // Low storage (stays lean to act as a pipe, not a tank)
                mode.nutrient_priority = 0.5;
                mode.max_cell_size = 1.5;
                mode.split_mass = 2.5;
                // Sealed pipe by default - exchange ports must be explicitly enabled.
                mode.vascular_nutrient_transport = true;
                mode.vascular_outlet = false;
                mode.vascular_signal_transport = false;
                mode.vascular_signal_exchange = false;
                mode.vascular_signal_capacity = 10.0;
            }
            CellType::Gametocyte => {
                // Gametocytes behave like Embryocytes: nutrients go into reserve,
                // release triggers control when they detach and seek a partner.
                // They never split: reproduction happens only through gamete merge.
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 99.0; // Very high - Gametocytes never split on their own
                mode.split_interval = 60.0; // Sentinel: never self-divide when free
                mode.max_splits = 0; // Tooltip-visible rule: no Gametocyte division.
                mode.gametocyte_merge_range = 0.5;
                // Release trigger: detach after accumulating enough reserve
                mode.embryocyte_use_timer = false;
                mode.embryocyte_release_timer = 15.0;
                mode.embryocyte_use_threshold = true;
                mode.embryocyte_threshold_value = 16384; // Half reserve = ready to mate
                mode.embryocyte_use_signal = false;
                mode.embryocyte_signal_channel = 0;
                mode.embryocyte_signal_value = 1.0;
            }
            CellType::Cognocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
                mode.cognocyte_operation = 0; // Add
                mode.cognocyte_input_channel_a = 0;
                mode.cognocyte_input_channel_b = 1;
                mode.cognocyte_output_channel = 8;
                mode.cognocyte_output_hops = 5;
            }
            CellType::Memorocyte => {
                mode.nutrient_priority = 2.0;
                mode.max_cell_size = 2.0;
                mode.split_mass = 3.1;
                mode.memorocyte_rate = 0.1;
                mode.memorocyte_input_channel = 0;
                mode.memorocyte_output_channel = 9;
                mode.memorocyte_output_hops = 5;
            }
            _ => {}
        }
    }
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Test
    }
}
