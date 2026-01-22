//! Voxel-based fluid simulation system
//!
//! GPU-only fluid simulation on a 128³ grid featuring:
//! - Water, lava, and steam with realistic physics
//! - Phase changes (water ↔ steam)
//! - Incompressible flow (pressure projection)
//! - Triple-buffered execution (zero CPU sync)
//! - One-way coupling to cell physics

pub mod buffers;

pub use buffers::{FluidBuffers, FluidParams, FluidType};

/// Grid resolution (hardcoded)
pub const GRID_RESOLUTION: u32 = 128;

/// Total voxel count
pub const TOTAL_VOXELS: usize = (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
