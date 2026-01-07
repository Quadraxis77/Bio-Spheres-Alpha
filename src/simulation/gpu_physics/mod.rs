//! GPU Physics Module
//! 
//! Contains the GPU compute physics pipeline for Bio-Spheres simulation.

pub mod adhesion;
pub mod adhesion_buffers;
pub mod adhesion_integration;
pub mod compute_pipelines;
pub mod gpu_scene_integration;
pub mod triple_buffer;

pub use adhesion_buffers::AdhesionBuffers;
pub use compute_pipelines::{CachedBindGroups, GpuPhysicsPipelines};
pub use gpu_scene_integration::{execute_gpu_physics_step, execute_lifecycle_pipeline};
pub use triple_buffer::GpuTripleBufferSystem;