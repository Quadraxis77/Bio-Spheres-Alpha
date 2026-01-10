//! Flagellocyte Cell Behavior
//!
//! Flagellocytes are motile cells with an animated helical tail (flagellum)
//! that provides propulsion. The tail parameters are passed through type_data
//! for shader rendering.
//!
//! # Visual Appearance
//!
//! Flagellocytes render with:
//! - A spherical cell body (same as other cell types)
//! - An animated helical tail attached to the back of the cell
//! - Wave propagation along the tail for swimming animation
//!
//! # Simulation Behavior
//!
//! Flagellocytes participate in:
//! - Basic physics (collision, gravity, damping)
//! - Adhesion connections
//! - Cell division
//! - Nutrient transport
//! - **Swim force** - Forward thrust proportional to swim_force setting
//! - **Nutrient consumption** - Mass loss proportional to swim_force
//!
//! # Instance Data Layout
//!
//! The type_data array is populated with flagella parameters:
//! - `data[0]`: tail_length (0.5 - 3.0)
//! - `data[1]`: tail_thickness (0.01 - 0.3)
//! - `data[2]`: tail_amplitude (0.0 - 0.5)
//! - `data[3]`: tail_frequency (0.5 - 10.0)
//! - `data[4]`: tail_speed (0.0 - 15.0)
//! - `data[5]`: tail_taper (0.0 - 1.0)
//! - `data[6]`: tail_segments (4 - 64)
//! - `data[7]`: reserved

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Flagellocyte cells.
///
/// Flagellocytes are motile cells with an animated helical tail.
/// The tail parameters are passed through type_data for shader rendering.
///
/// # Rendering
///
/// The flagella parameters come from [`CellTypeVisuals`](crate::cell::CellTypeVisuals),
/// not from ModeSettings. The actual values are populated by the instance builder
/// from the visual settings. This behavior returns empty data as a placeholder
/// that will be overwritten by the instance builder with the correct visual parameters.
///
/// # Simulation
///
/// Flagellocytes apply a forward thrust force based on their `swim_force` setting
/// in ModeSettings. The thrust is applied in the cell's forward direction (local +Z axis).
pub struct FlagellocyteBehavior;

impl CellBehavior for FlagellocyteBehavior {
    /// Build instance data for Flagellocyte cells.
    ///
    /// Note: The actual flagella visual parameters come from CellTypeVisuals
    /// and are populated by the instance builder. This method returns empty
    /// data as the visual parameters are not stored in ModeSettings.
    ///
    /// # Arguments
    ///
    /// * `_mode_settings` - Mode settings (swim_force is used by physics, not rendering)
    ///
    /// # Returns
    ///
    /// Empty type-specific instance data. The instance builder will populate
    /// the actual flagella parameters from CellTypeVisuals.
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        // Flagella parameters come from CellTypeVisuals, not ModeSettings
        // The actual values are populated by the instance builder from visuals
        TypeSpecificInstanceData::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flagellocyte_behavior_returns_empty_data() {
        let behavior = FlagellocyteBehavior;
        let mode_settings = ModeSettings::default();
        
        let instance_data = behavior.build_instance_data(&mode_settings);
        
        // Behavior returns empty data - actual values come from CellTypeVisuals
        for i in 0..8 {
            assert_eq!(
                instance_data.data[i], 0.0,
                "type_data[{}] should be 0.0 (populated by instance builder)",
                i
            );
        }
    }

    #[test]
    fn flagellocyte_behavior_is_send_sync() {
        // Verify that FlagellocyteBehavior implements Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FlagellocyteBehavior>();
    }
}
