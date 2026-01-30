//! Phagocyte Cell Behavior
//!
//! Phagocyte cells are specialized for engulfing and consuming other cells.
//! They apply forces toward nearby cells and can "eat" cells on contact.
//!
//! Phagocytes have enhanced swimming capability and darker appearance
//! with membrane ripples.

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Phagocyte cells.
///
/// Phagocytes are specialized cells that:
/// - Apply forces toward nearby cells (hunting behavior)
/// - Can consume other cells on contact
/// - Have swimming capability similar to Flagellocytes
/// - Render with darker appearance and membrane ripples
///
/// # Rendering
///
/// Phagocytes use texture atlas rendering with special darker colors.
/// They don't use procedural tails like Flagellocytes.
///
/// # Simulation
///
/// Phagocytes participate in:
/// - Basic physics (collision, gravity, damping)
/// - Swimming forces (applies_swim_force = 1)
/// - Cell division with normal split_interval
/// - Potential future: cell consumption mechanics
pub struct PhagocyteBehavior;

impl CellBehavior for PhagocyteBehavior {
    /// Build instance data for Phagocyte cells.
    ///
    /// Phagocytes use texture atlas rendering, so they return empty
    /// type-specific data. Their unique appearance comes from the shader.
    ///
    /// # Arguments
    ///
    /// * `_mode_settings` - Mode settings (unused for Phagocytes currently)
    ///
    /// # Returns
    ///
    /// Empty type-specific instance data.
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
