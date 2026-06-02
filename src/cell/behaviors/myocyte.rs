//! Myocyte Cell Behavior
//!
//! Myocytes are muscle cells that modify the rest length of their adhesion
//! connections based on received signals. When contracted, adhesions shorten,
//! pulling connected cells closer together. When relaxed, adhesions return
//! to their normal rest length.
//!
//! # Simulation Behavior
//!
//! Myocytes participate in:
//! - Basic physics (collision, gravity, damping)
//! - Adhesion connections (with dynamic rest length modification)
//! - Cell division
//! - Nutrient transport
//! - **Muscle contraction** - Modifies adhesion rest lengths based on signal state
//!
//! # Signal-Driven Contraction
//!
//! The myocyte reads a signal channel and applies different contraction amounts
//! depending on whether the signal is above or below a threshold:
//! - Signal >= threshold: applies `contraction_above` (e.g., 0.5 = 50% shorter)
//! - Signal < threshold: applies `contraction_below` (e.g., 0.0 = no contraction)
//!
//! The contraction value (0.0 to 1.0) scales the adhesion rest length:
//! effective_rest_length = rest_length * (1.0 - contraction)

use super::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Myocyte (muscle) cells.
///
/// Myocytes modify adhesion rest lengths based on signal input,
/// acting as contractile elements in multicellular organisms.
pub struct MyocyteBehavior;

impl CellBehavior for MyocyteBehavior {
    /// Build instance data for Myocyte cells.
    ///
    /// Myocytes don't need special rendering data beyond the standard cell appearance.
    ///
    /// # Arguments
    ///
    /// * `_mode_settings` - Mode settings (contraction params used by physics, not rendering)
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
    fn myocyte_behavior_returns_empty_data() {
        let behavior = MyocyteBehavior;
        let mode_settings = ModeSettings::default();

        let instance_data = behavior.build_instance_data(&mode_settings);

        for i in 0..8 {
            assert_eq!(instance_data.data[i], 0.0, "type_data[{}] should be 0.0", i);
        }
    }

    #[test]
    fn myocyte_behavior_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MyocyteBehavior>();
    }
}
