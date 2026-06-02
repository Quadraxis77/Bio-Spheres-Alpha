//! Vasculocyte Cell Behavior
//!
//! Vasculocyte cells form nutrient transport networks within organisms.
//! They act as high-throughput conduits, moving nutrients rapidly along
//! vascular chains and delivering them through controllable outlet points.
//!
//! Key properties:
//! - Very high transport rate between adjacent vasculocytes (express highway)
//! - Sealed by default: nutrients do not leak to non-vascular neighbors
//! - Outlet toggle: when enabled, releases nutrients to surrounding cells
//! - Responds to physical compression (e.g. from myocytes) with a transport boost,
//!   enabling myocyte-driven pumping without any explicit signal wiring

use super::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Vasculocyte cells.
pub struct VasculocyteBehavior;

impl CellBehavior for VasculocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
