//! Devorocyte Cell Behavior
//!
//! Devorocytes are predatory cells that steal nutrients from and kill foreign cells
//! they come into contact with. They ignore cells of the same organism ID or genome ID.

use super::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Devorocyte cells.
pub struct DevorocyteBehavior;

impl CellBehavior for DevorocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
