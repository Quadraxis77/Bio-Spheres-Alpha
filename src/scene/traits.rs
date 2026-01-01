//! Scene trait definition.
//!
//! Defines the common interface for all scene types (Preview, GPU).

use crate::ui::camera::CameraController;
use glam::Vec3;

/// Common interface for all scene types.
///
/// Each scene manages its own simulation state, renderer, and camera.
/// The App delegates to the active scene for updates and rendering.
pub trait Scene {
    /// Update the scene simulation by the given delta time.
    fn update(&mut self, dt: f32);

    /// Render the scene to the given texture view.
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    );

    /// Handle window resize.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);

    /// Get a reference to the camera controller.
    fn camera(&self) -> &CameraController;

    /// Get a mutable reference to the camera controller.
    fn camera_mut(&mut self) -> &mut CameraController;

    /// Check if the simulation is paused.
    fn is_paused(&self) -> bool;

    /// Set the paused state.
    fn set_paused(&mut self, paused: bool);

    /// Get the current simulation time.
    fn current_time(&self) -> f32;

    /// Get the number of cells in the simulation.
    fn cell_count(&self) -> usize;

    /// Get the camera position for UI display.
    fn camera_position(&self) -> Vec3 {
        self.camera().position()
    }
}
