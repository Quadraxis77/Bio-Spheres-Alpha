//! Luminocyte Cell Behavior
//!
//! Luminocytes are signal-reactive light emitters. Their GPU light emission is
//! handled in the light-field pass, while the renderer uses normal instance data.

use super::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

pub struct LuminocyteBehavior;

impl CellBehavior for LuminocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
