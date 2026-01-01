//! Scene manager for switching between simulation modes.
//!
//! Handles creation, switching, and lifecycle of Preview and GPU scenes.

use crate::scene::{GpuScene, PreviewScene, Scene};
use crate::ui::SimulationMode;

/// Manages the active scene and handles scene switching.
pub struct SceneManager {
    /// Current simulation mode
    current_mode: SimulationMode,
    /// Preview scene (lazy initialized)
    preview_scene: Option<PreviewScene>,
    /// GPU scene (lazy initialized)
    gpu_scene: Option<GpuScene>,
}

impl SceneManager {
    /// Create a new scene manager with the preview scene active.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        // Start with preview scene
        let preview_scene = Some(PreviewScene::new(device, queue, config));

        Self {
            current_mode: SimulationMode::Preview,
            preview_scene,
            gpu_scene: None,
        }
    }

    /// Get the current simulation mode.
    pub fn current_mode(&self) -> SimulationMode {
        self.current_mode
    }

    /// Switch to a different simulation mode.
    ///
    /// Creates the target scene if it doesn't exist yet.
    pub fn switch_mode(
        &mut self,
        mode: SimulationMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) {
        if mode == self.current_mode {
            return;
        }

        log::info!(
            "Switching from {} to {}",
            self.current_mode.display_name(),
            mode.display_name()
        );

        // Ensure target scene exists
        match mode {
            SimulationMode::Preview => {
                if self.preview_scene.is_none() {
                    self.preview_scene = Some(PreviewScene::new(device, queue, config));
                }
            }
            SimulationMode::Gpu => {
                if self.gpu_scene.is_none() {
                    self.gpu_scene = Some(GpuScene::new(device, queue, config));
                }
            }
        }

        self.current_mode = mode;
    }

    /// Get a reference to the active scene.
    pub fn active_scene(&self) -> &dyn Scene {
        match self.current_mode {
            SimulationMode::Preview => self
                .preview_scene
                .as_ref()
                .expect("Preview scene should exist"),
            SimulationMode::Gpu => self.gpu_scene.as_ref().expect("GPU scene should exist"),
        }
    }

    /// Get a mutable reference to the active scene.
    pub fn active_scene_mut(&mut self) -> &mut dyn Scene {
        match self.current_mode {
            SimulationMode::Preview => self
                .preview_scene
                .as_mut()
                .expect("Preview scene should exist"),
            SimulationMode::Gpu => self.gpu_scene.as_mut().expect("GPU scene should exist"),
        }
    }

    /// Get a reference to the preview scene if it exists.
    pub fn preview_scene(&self) -> Option<&PreviewScene> {
        self.preview_scene.as_ref()
    }

    /// Get a mutable reference to the preview scene if it exists.
    pub fn preview_scene_mut(&mut self) -> Option<&mut PreviewScene> {
        self.preview_scene.as_mut()
    }

    /// Get a reference to the GPU scene if it exists.
    pub fn gpu_scene(&self) -> Option<&GpuScene> {
        self.gpu_scene.as_ref()
    }

    /// Get a mutable reference to the GPU scene if it exists.
    pub fn gpu_scene_mut(&mut self) -> Option<&mut GpuScene> {
        self.gpu_scene.as_mut()
    }

    /// Get a reference to the preview scene for UI access.
    /// Returns None if preview scene doesn't exist or current mode is not Preview.
    pub fn get_preview_scene(&self) -> Option<&PreviewScene> {
        if self.current_mode == SimulationMode::Preview {
            self.preview_scene.as_ref()
        } else {
            None
        }
    }

    /// Get a mutable reference to the preview scene for UI access.
    /// Returns None if preview scene doesn't exist or current mode is not Preview.
    pub fn get_preview_scene_mut(&mut self) -> Option<&mut PreviewScene> {
        if self.current_mode == SimulationMode::Preview {
            self.preview_scene.as_mut()
        } else {
            None
        }
    }

    /// Handle window resize for all existing scenes.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(scene) = &mut self.preview_scene {
            scene.resize(device, width, height);
        }
        if let Some(scene) = &mut self.gpu_scene {
            scene.resize(device, width, height);
        }
    }

    /// Update the active scene.
    pub fn update(&mut self, dt: f32) {
        self.active_scene_mut().update(dt);
    }

    /// Render the active scene.
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) {
        self.active_scene_mut().render(device, queue, view);
    }
}
