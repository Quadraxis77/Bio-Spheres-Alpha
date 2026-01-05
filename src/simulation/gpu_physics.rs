//! GPU Physics Pipeline - Modular Implementation
//! 
//! This module coordinates the GPU physics pipeline using compute shaders.
//! The implementation is split across multiple focused modules.
//! 
//! ## Architecture
//! 
//! ```text
//! src/simulation/gpu_physics/
//! ├── mod.rs                    # Module exports
//! ├── triple_buffer.rs          # Triple-buffered GPU buffers
//! ├── compute_pipelines.rs      # Pipeline creation and bind groups
//! └── gpu_scene_integration.rs  # Pipeline execution
//! ```
//! 
//! ## Key Patterns
//! - **NO atomics** in shaders - uses prefix-sum compaction
//! - **NO CPU readback** during simulation loop
//! - **Triple buffering** for lock-free GPU computation
//! - **64³ spatial grid** for collision acceleration
//! - **256-byte aligned** uniform buffers
//! 
//! ## Physics Pipeline Stages
//! 1. Clear spatial grid
//! 2. Assign cells to grid
//! 3. Insert cells into grid
//! 4. Collision detection
//! 5. Position integration
//! 6. Velocity integration
//!
//! ## Lifecycle Pipeline Stages (for cell division)
//! 7. Death scan - identify dead cells
//! 8. Prefix sum - compact free slots
//! 9. Division scan - identify dividing cells
//! 10. Division execute - create child cells

mod triple_buffer;
mod compute_pipelines;
mod gpu_scene_integration;

pub use triple_buffer::GpuTripleBufferSystem;
pub use compute_pipelines::GpuPhysicsPipelines;
pub use gpu_scene_integration::{execute_gpu_physics_step, execute_lifecycle_pipeline};