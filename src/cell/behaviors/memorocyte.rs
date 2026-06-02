//! Memorocyte Cell Behavior - leaky integrator memory cell.
//!
//! A Memorocyte maintains a single floating-point memory value that accumulates
//! incoming signals and decays toward zero over time:
//!
//! ```text
//! memo(t+1) = memo(t) * decay + input(t) * gain
//! ```
//!
//! The cell emits its current memory value as a signal every frame, regardless
//! of whether an input arrived. When no input is present the value simply decays.
//!
//! # Parameters
//!
//! - **decay** (0-1): fraction of memory retained per tick. 1.0 = perfect memory,
//!   0.0 = no memory at all. Typical values are 0.90-0.99.
//! - **gain** (0-10): how much of the incoming signal is added to memory per tick.
//!   Values > 1 can drive the memory above the raw signal level.
//! - **input_channel** (0-15): channel to read. If absent the memory just decays.
//! - **output_channel** (0-15): channel the memory value is emitted on.
//! - **output_hops** (1-20): adhesion hops the emitted signal travels.
//!
//! # Uses
//!
//! - **Smoothing**: average out a noisy oculocyte sensor over many frames.
//! - **Timers**: with no input and decay < 1 the cell counts down from a charged
//!   state; with regular input pulses it acts as a frequency integrator.
//! - **Hysteresis**: a Memorocyte downstream of a GreaterThan Cognocyte stays
//!   high for several frames after the trigger disappears, preventing jitter.
//! - **Persistence**: remember that a predator was detected even after it leaves
//!   the oculocyte's field of view.

use super::{CellBehavior, TypeSpecificInstanceData};
use crate::genome::ModeSettings;

/// Behavior implementation for Memorocyte (leaky-integrator) cells.
pub struct MemoryocyteBehavior;

impl CellBehavior for MemoryocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memorocyte_behavior_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryocyteBehavior>();
    }
}
