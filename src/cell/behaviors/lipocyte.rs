//! Lipocyte Cell Behavior
//!
//! Lipocyte cells are specialized storage cells that accumulate and transfer
//! nutrients to other cells. They are larger, slower, and have translucent
//! appearance with visible internal structures.
//!
//! Lipocytes have high nutrient_priority and serve as nutrient reservoirs
//! for the colony.

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Lipocyte cells.
///
/// Lipocytes are specialized cells that:
/// - Store large amounts of nutrients (is_storage_cell = 1)
/// - Have larger size and slower movement
/// - Render with translucent appearance showing internal structures
/// - Transfer nutrients to nearby cells
///
/// # Rendering
///
/// Lipocytes use texture atlas rendering with translucent, bubble-like
/// appearance. The shader reveals internal vesicle-like structures.
///
/// # Simulation
///
/// Lipocytes participate in:
/// - Basic physics (collision, gravity, damping)
/// - Enhanced nutrient storage and transfer
/// - Cell division with normal split_interval
/// - No swimming forces (stationary/slow drift)
pub struct LipocyteBehavior;

impl CellBehavior for LipocyteBehavior {
    /// Build instance data for Lipocyte cells.
    ///
    /// Lipocytes use texture atlas rendering, so they return empty
    /// type-specific data. Their translucent appearance comes from the shader.
    ///
    /// # Arguments
    ///
    /// * `_mode_settings` - Mode settings (unused for Lipocytes currently)
    ///
    /// # Returns
    ///
    /// Empty type-specific instance data.
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
