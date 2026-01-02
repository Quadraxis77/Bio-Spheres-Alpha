//! Cell type definitions for Bio-Spheres simulation.
//!
//! This module defines the different types of cells that can exist in the simulation,
//! each with their own behaviors and properties.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Visual settings for a cell type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
}

impl Default for CellTypeVisuals {
    fn default() -> Self {
        Self {
            specular_strength: 0.3,
            specular_power: 32.0,
            fresnel_strength: 0.2,
            membrane_noise_scale: 8.0,
            membrane_noise_strength: 0.15,
            membrane_noise_speed: 0.0,
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

/// Cell type enumeration.
///
/// Currently only Test cells are implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellType {
    Test = 0,
}

impl CellType {
    /// Get all available cell types as a slice.
    pub const fn all() -> &'static [CellType] {
        &[CellType::Test]
    }

    /// Get the display name for this cell type.
    pub const fn name(&self) -> &'static str {
        match self {
            CellType::Test => "Test",
        }
    }

    /// Get all cell type names as a slice.
    pub const fn names() -> &'static [&'static str] {
        &["Test"]
    }

    /// Convert from integer index to cell type.
    pub fn from_index(index: i32) -> Option<Self> {
        match index {
            0 => Some(CellType::Test),
            _ => None,
        }
    }

    /// Convert to integer index.
    pub const fn to_index(&self) -> i32 {
        *self as i32
    }
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Test
    }
}
