//! Embryocyte Cell Behavior
//!
//! Embryocytes are nutrient storage vessels that accumulate a reserve while
//! connected to an organism, then get released to divide and distribute that
//! reserve to their descendants across multiple generations.
//!
//! # Unique properties
//! - Carries a `reserve` (u32, max 65535) that is the ONLY nutrient source for Embryocytes
//! - Incoming nutrients from adhesion transport go into reserve, never the normal nutrient pool
//! - Never sends nutrients out; only receives
//! - Normal metabolism is skipped entirely
//! - Once released (all adhesions dropped), reserve burns at 10 units/sec
//! - Division halves reserve; children cannot themselves be Embryocytes

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Embryocyte cells.
///
/// Embryocytes are specialized cells that:
/// - Store reserves (u32, max 65535) as their sole energy source
/// - Release when timer/threshold/signal conditions are met
/// - Distribute reserve to descendants across multiple generations
pub struct EmbryocyteBehavior;

impl CellBehavior for EmbryocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
