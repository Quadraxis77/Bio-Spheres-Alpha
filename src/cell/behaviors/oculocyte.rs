//! Oculocyte cell behavior implementation.
//!
//! Oculocytes are sensory cells that detect targets (cells, food, light, barriers)
//! along their forward direction and send signals through adhesion connections.

use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Oculocyte cells.
pub struct OculocyteBehavior;

impl CellBehavior for OculocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
