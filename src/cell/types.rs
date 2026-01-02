//! Cell type definitions for Bio-Spheres simulation.
//!
//! This module defines the different types of cells that can exist in the simulation,
//! each with their own behaviors and properties.

use serde::{Deserialize, Serialize};

/// Visual settings for a cell type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CellTypeVisuals {
    pub specular_strength: f32,
    pub specular_power: f32,
    pub fresnel_strength: f32,
}

impl Default for CellTypeVisuals {
    fn default() -> Self {
        Self {
            specular_strength: 0.3,
            specular_power: 32.0,
            fresnel_strength: 0.2,
        }
    }
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
