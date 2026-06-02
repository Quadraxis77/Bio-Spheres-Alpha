//! Gametocyte Cell Behavior
//!
//! Gametocytes are reproductive cells that carry half the organism's genome
//! information. When two Gametocytes from different organisms come into contact,
//! the GPU detects the event and the CPU performs genome crossover to produce
//! a hybrid offspring organism. Both parent Gametocytes die upon merging.

use super::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Gametocyte cells.
///
/// The actual merge logic runs in the GPU compute shader `gametocyte_merge.wgsl`,
/// which writes merge events to a readback buffer. The CPU then reads those
/// events and performs genome crossover + offspring spawning.
pub struct GametocyteBehavior;

impl CellBehavior for GametocyteBehavior {
    fn build_instance_data(&self, mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        // Pack pulse_speed and nucleus_glow into type_data for the visual shader
        let mut data = [0.0f32; 8];
        data[0] = mode_settings.gametocyte_merge_range; // merge detection range
        data[7] = 13.0; // cell_type index for unified shader branching
        TypeSpecificInstanceData { data }
    }
}
