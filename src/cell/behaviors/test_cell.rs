//! Test Cell Behavior
//!
//! The Test cell type is the simplest cell type, used for basic simulation
//! testing and as a template for implementing new cell types.
//!
//! Test cells have no special visual effects or simulation behaviors beyond
//! the basic physics (collision, adhesion, etc.).

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Test cells.
///
/// Test cells are the simplest cell type with no type-specific behavior.
/// They serve as:
/// - A baseline for testing the simulation
/// - A template for implementing new cell types
/// - A fallback when no specific behavior is needed
///
/// # Rendering
///
/// Test cells render as simple spheres with mode-based color and emissive.
/// They don't use any of the reserved type_data fields.
///
/// # Simulation
///
/// Test cells participate in:
/// - Basic physics (collision, gravity, damping)
/// - Adhesion connections
/// - Cell division
/// - Nutrient transport
///
/// They do not have any type-specific forces or behaviors.
pub struct TestCellBehavior;

impl CellBehavior for TestCellBehavior {
    /// Build instance data for Test cells.
    ///
    /// Test cells don't use any type-specific instance data, so this
    /// returns empty data (all zeros). The shader will use only the
    /// common instance data fields (position, radius, color, etc.).
    ///
    /// # Arguments
    ///
    /// * `_mode_settings` - Mode settings (unused for Test cells)
    ///
    /// # Returns
    ///
    /// Empty type-specific instance data.
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_behavior_returns_empty_data() {
        let behavior = TestCellBehavior;
        let mode_settings = ModeSettings::default();
        
        let instance_data = behavior.build_instance_data(&mode_settings);
        
        // All type_data fields should be zero for Test cells
        for i in 0..8 {
            assert_eq!(
                instance_data.data[i], 0.0,
                "type_data[{}] should be 0.0 for Test cells",
                i
            );
        }
    }

    #[test]
    fn test_cell_behavior_is_send_sync() {
        // Verify that TestCellBehavior implements Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TestCellBehavior>();
    }
}
