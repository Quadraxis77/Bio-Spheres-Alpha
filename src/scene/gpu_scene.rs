//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::rendering::cells::CellRenderer;
use crate::scene::Scene;
use crate::simulation::{CanonicalState, PhysicsConfig};
use crate::ui::camera::CameraController;

/// GPU simulation scene for large-scale simulations.
///
/// Uses compute shaders for physics simulation, allowing for
/// much larger cell counts than the CPU preview mode.
pub struct GpuScene {
    /// Canonical state (used for initial setup and readback if needed)
    pub canonical_state: CanonicalState,
    /// Cell renderer for visualization
    pub renderer: CellRenderer,
    /// Physics configuration
    pub config: PhysicsConfig,
    /// Whether simulation is paused
    pub paused: bool,
    /// Camera controller
    pub camera: CameraController,
    /// Current simulation time
    pub current_time: f32,
    // TODO: Add GPU buffers for compute simulation
    // pub position_buffer: wgpu::Buffer,
    // pub velocity_buffer: wgpu::Buffer,
    // pub compute_pipeline: wgpu::ComputePipeline,
}

impl GpuScene {
    /// Create a new GPU scene.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let capacity = 100_000; // GPU can handle many more cells
        let canonical_state = CanonicalState::new(capacity);
        let config = PhysicsConfig::default();

        let renderer = CellRenderer::new(device, queue, surface_config, capacity);

        Self {
            canonical_state,
            renderer,
            config,
            paused: false,
            camera: CameraController::new(),
            current_time: 0.0,
        }
    }

    /// Run GPU compute physics step.
    ///
    /// TODO: Implement actual GPU compute physics.
    fn run_gpu_physics(&mut self, _dt: f32) {
        // Placeholder for GPU compute physics
        // This will dispatch compute shaders to update positions/velocities
    }
}

impl Scene for GpuScene {
    fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        self.run_gpu_physics(dt);
        self.current_time += dt;
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        self.renderer.render(
            device,
            queue,
            view,
            &self.canonical_state,
            self.camera.position(),
            self.camera.rotation,
        );
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
    }

    fn camera(&self) -> &CameraController {
        &self.camera
    }

    fn camera_mut(&mut self) -> &mut CameraController {
        &mut self.camera
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    fn current_time(&self) -> f32 {
        self.current_time
    }

    fn cell_count(&self) -> usize {
        self.canonical_state.cell_count
    }
}
