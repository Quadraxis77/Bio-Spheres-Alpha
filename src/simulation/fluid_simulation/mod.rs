//! Voxel-based fluid simulation system
//!
//! GPU-only fluid simulation on a 128³ grid featuring:
//! - Water falling and spreading with atomic operations
//! - Double-buffered compute shaders
//! - Checkered processing to prevent race conditions

pub mod buffers;
pub mod gpu_simulator;
pub mod solid_mask;

pub use buffers::{FluidBuffers, FluidParams, FluidType};
pub use gpu_simulator::{GpuFluidSimulator, WaterGridParams};
pub use solid_mask::SolidMaskGenerator;

/// Grid resolution (hardcoded)
pub const GRID_RESOLUTION: u32 = 128;

/// Total voxel count
pub const TOTAL_VOXELS: usize = (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
