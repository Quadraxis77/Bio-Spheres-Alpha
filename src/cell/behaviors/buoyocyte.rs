//! Buoyocyte Cell Behavior
//!
//! Buoyocyte cells use Archimedes' principle to generate buoyant force when
//! submerged in water. They displace fluid proportional to their volume,
//! creating an upward force that opposes gravity.
//!
//! # Physics
//!
//! Archimedes' principle: F_buoyancy = ρ_fluid × V_cell × g (upward)
//! - ρ_fluid: density of the surrounding fluid (water ≈ 1.0 in simulation units)
//! - V_cell: volume of the cell (4/3 × π × r³)
//! - g: gravitational acceleration
//!
//! When the buoyant force exceeds the cell's weight (mg), the cell floats upward.
//! Buoyocytes are designed to have lower effective density than water, making
//! them naturally buoyant — like gas vacuoles in real aquatic organisms.

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Behavior implementation for Buoyocyte cells.
///
/// Buoyocytes are specialized cells that:
/// - Apply Archimedes buoyancy force when submerged in water
/// - Float upward through fluid, counteracting gravity
/// - Can serve as flotation aids for organisms via adhesion connections
///
/// # Rendering
///
/// Buoyocytes use texture atlas rendering with a translucent, bubble-like
/// appearance suggesting internal gas vacuoles.
///
/// # Simulation
///
/// Buoyocytes participate in:
/// - Basic physics (collision, gravity, damping)
/// - Archimedes buoyancy (upward force when in water)
/// - Cell division with normal split_interval
/// - No swimming forces (passive flotation only)
pub struct BuoyocyteBehavior;

impl CellBehavior for BuoyocyteBehavior {
    /// Build instance data for Buoyocyte cells.
    ///
    /// Buoyocytes use texture atlas rendering, so they return empty
    /// type-specific data. Their appearance comes from the unified shader.
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}
