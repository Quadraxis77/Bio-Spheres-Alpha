//! Scene management for Bio-Spheres.
//!
//! This module provides the scene abstraction and management for different
//! simulation modes (Preview and GPU).

pub mod compute_pipelines;
pub mod gpu_buffers;
pub mod gpu_scene;
pub mod manager;
pub mod performance_monitor;
pub mod preview_scene;
pub mod preview_state;
pub mod traits;
pub mod triple_buffer;

pub use gpu_scene::GpuScene;
pub use manager::SceneManager;
pub use preview_scene::PreviewScene;
pub use preview_state::PreviewState;
pub use traits::Scene;
