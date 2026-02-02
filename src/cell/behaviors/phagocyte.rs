//! Phagocyte Cell Behavior
//!
//! Phagocyte cells are specialized for consuming nutrients from their environment.
//! They are passive cells that feed on nutrient particles in the fluid system.
//! In the preview scene, this is simulated via automatic nutrient gain.
//!
//! Phagocytes have a darker appearance with membrane ripples.

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Phagocyte cells.
///
/// Phagocytes are specialized cells that:
/// - Are passive nutrient-eating cells (no automatic nutrient generation)
/// - Rely on base metabolism and nutrient transport from connected cells
/// - Can consume other cells on contact (future feature)
/// - Render with darker appearance and membrane ripples
///
/// # Rendering
///
/// Phagocytes use texture atlas rendering with special darker colors.
/// They don't use procedural tails or swim forces.
///
/// # Simulation
///
/// Phagocytes participate in:
/// - Basic physics (collision, gravity, damping)
/// - Base metabolism (consume nutrients to stay alive)
/// - Nutrient transport through adhesion connections
/// - Cell division with normal split_interval
/// - Future: cell consumption mechanics for gaining mass
///
/// # Scene-Specific Behavior
///
/// - **Preview Scene**: Auto-gains mass to simulate eating nutrient particles (GPU feature)
/// - **GPU Scene**: Gains mass by consuming nutrient particles from fluid system (not yet implemented)
///
/// # Metabolism
///
/// Phagocytes consume 0.05 mass/second as base metabolism. In the current implementation,
/// they must rely on nutrient transport from connected producer cells until the nutrient
/// particle system is fully implemented.
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
