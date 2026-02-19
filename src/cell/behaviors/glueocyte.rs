//! Glueocyte cell behavior implementation.
//!
//! Glueocytes are specialized adhesion cells that form bonds on contact
//! with any other cell they collide with, using their configured adhesion settings.

use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Glueocyte cells.
pub struct GlueocyteBehavior;

impl CellBehavior for GlueocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
