//! Photocyte Cell Behavior
//!
//! Photocyte cells are specialized for photosynthesis, gaining mass from light.
//! They have bright green/yellow appearance and glowing visual effects.
//!
//! Photocytes are stationary (no swim force) and gain nutrients based on
//! ambient light intensity in the simulation.

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Photocyte cells.
///
/// Photocytes are specialized cells that:
/// - Gain mass/nutrients from light (gains_mass_from_light = 1)
/// - Do not have swimming capability
/// - Render with bright green/yellow glowing appearance
/// - Follow normal division rules with split_interval
///
/// # Rendering
///
/// Photocytes use texture atlas rendering with bright, luminescent colors.
/// Their shader creates a glowing, photosynthetic appearance.
///
/// # Simulation
///
/// Photocytes participate in:
/// - Basic physics (collision, gravity, damping)
/// - Light-based nutrient gain (future implementation)
/// - Cell division with normal split_interval
/// - No swimming forces (stationary)
pub struct PhotocyteBehavior;

impl CellBehavior for PhotocyteBehavior {
    /// Build instance data for Photocyte cells.
    ///
    /// Photocytes use texture atlas rendering, so they return empty
    /// type-specific data. Their glowing appearance comes from the shader.
    ///
    /// # Arguments
    ///
    /// * `_mode_settings` - Mode settings (unused for Photocytes currently)
    ///
    /// # Returns
    ///
    /// Empty type-specific instance data.
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
