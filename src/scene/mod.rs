//! Scene management for Bio-Spheres.
//!
//! This module provides the scene abstraction and management for different
//! simulation modes (Preview and GPU).

pub mod gpu_scene;
pub mod main_menu;
pub mod manager;
pub mod preview_scene;
pub mod preview_state;
pub mod snapshot;
pub mod snapshot_io;
pub mod traits;

pub use gpu_scene::GpuScene;
pub use main_menu::MainMenuScene;
pub use manager::SceneManager;
pub use preview_scene::PreviewScene;
pub use preview_state::PreviewState;
pub use snapshot::{GpuSceneSnapshot, SnapshotError};
pub use traits::Scene;
