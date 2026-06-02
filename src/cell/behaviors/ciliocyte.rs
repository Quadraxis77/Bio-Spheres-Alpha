//! Ciliocyte cell behavior implementation.
//!
//! Ciliocytes are contact-dependent surface propulsion cells whose surface cilia
//! push against anything touching them - neighboring cells or solid surfaces -
//! generating thrust along the cell's local forward axis (+Z). In open water
//! with nothing to push against, the cell is completely inert.

use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Ciliocyte cells.
pub struct CiliocyteBehavior;

impl CellBehavior for CiliocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        // Actual values (effective_speed, ring_frequency, ring_depth, ring_speed)
        // are populated by InstanceBuilder from visuals + signal state
        TypeSpecificInstanceData::empty()
    }
}
