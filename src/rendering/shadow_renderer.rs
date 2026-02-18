//! Shadow rendering via light field sampling.
//!
//! Surface shadows are implemented by sampling the existing light field
//! (computed by the LightFieldSystem) directly in cave and cell fragment shaders.
//! This module is kept as a placeholder; the actual shadow logic lives in:
//!   - `shaders/cave_system.wgsl` (group 2 shadow field)
//!   - `shaders/cells/cell_unified.wgsl` (group 1 shadow field)
//!   - `src/simulation/gpu_physics/light_field.rs` (ShadowFieldParams, bind group creation)
