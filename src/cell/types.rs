//! Cell type definitions for Bio-Spheres simulation.
//!
//! This module defines the different types of cells that can exist in the simulation,
//! each with their own behaviors and properties.

/// Cell type enumeration matching the reference implementation.
///
/// Each cell type has different behaviors:
/// - Photocyte: Absorbs light to gain biomass
/// - Phagocyte: Eats food to gain biomass  
/// - Flagellocyte: Propels itself forward
/// - Devorocyte: Advanced eating behavior
/// - Lipocyte: Fat storage cell
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellType {
    Photocyte = 0,
    Phagocyte = 1,
    Flagellocyte = 2,
    Devorocyte = 3,
    Lipocyte = 4,
}

impl CellType {
    /// Get all available cell types as a slice.
    pub const fn all() -> &'static [CellType] {
        &[
            CellType::Photocyte,
            CellType::Phagocyte,
            CellType::Flagellocyte,
            CellType::Devorocyte,
            CellType::Lipocyte,
        ]
    }

    /// Get the display name for this cell type.
    pub const fn name(&self) -> &'static str {
        match self {
            CellType::Photocyte => "Photocyte",
            CellType::Phagocyte => "Phagocyte",
            CellType::Flagellocyte => "Flagellocyte",
            CellType::Devorocyte => "Devorocyte",
            CellType::Lipocyte => "Lipocyte",
        }
    }

    /// Get all cell type names as a slice.
    pub const fn names() -> &'static [&'static str] {
        &["Photocyte", "Phagocyte", "Flagellocyte", "Devorocyte", "Lipocyte"]
    }

    /// Convert from integer index to cell type.
    pub fn from_index(index: i32) -> Option<Self> {
        match index {
            0 => Some(CellType::Photocyte),
            1 => Some(CellType::Phagocyte),
            2 => Some(CellType::Flagellocyte),
            3 => Some(CellType::Devorocyte),
            4 => Some(CellType::Lipocyte),
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
        CellType::Photocyte
    }
}
