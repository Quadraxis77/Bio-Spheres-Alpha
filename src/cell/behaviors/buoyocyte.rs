//! Buoyocyte cell behavior implementation.
//!
//! Buoyocytes are specialized cells that apply upward buoyancy force
//! to float toward the water surface.

use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Buoyocyte cells.
pub struct BuoyocyteBehavior;

impl CellBehavior for BuoyocyteBehavior {
    fn build_instance_data(&self, mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        // Buoyocytes use buoyancy_force parameter for upward force
        TypeSpecificInstanceData::buoyocyte(mode_settings.buoyancy_force)
    }
}
