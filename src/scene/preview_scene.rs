//! Preview scene for genome editing.
//!
//! CPU-based simulation with GPU rendering, optimized for
//! editing and debugging genomes with small cell counts.

use crate::genome::Genome;
use crate::rendering::CellRenderer;
use crate::scene::{PreviewState, Scene};
use crate::simulation::PhysicsConfig;
use crate::ui::camera::CameraController;

/// Preview scene for genome editing.
///
/// Uses CPU physics for small-scale simulations with checkpoint
/// support for time scrubbing.
pub struct PreviewScene {
    /// Preview state with checkpoints
    pub state: PreviewState,
    /// Renderer for visualization
    pub renderer: CellRenderer,
    /// Current genome being edited
    pub genome: Genome,
    /// Physics configuration
    pub config: PhysicsConfig,
    /// Whether simulation is paused
    pub paused: bool,
    /// Camera controller
    pub camera: CameraController,
}

impl PreviewScene {
    /// Create a new preview scene.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let capacity = 256; // Preview capacity limit
        let mut state = PreviewState::new(capacity);
        let genome = Genome::default();
        let config = PhysicsConfig::default();

        state.genome_hash = PreviewState::compute_genome_hash(&genome);

        let renderer = CellRenderer::new(device, queue, surface_config, capacity);

        Self {
            state,
            renderer,
            genome,
            config,
            paused: false,
            camera: CameraController::new(),
        }
    }
}

impl Scene for PreviewScene {
    fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        // TODO: Run CPU physics simulation
        self.state.current_time += dt;
    }

    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) {
        self.renderer.render(
            device,
            queue,
            view,
            &self.state.canonical_state,
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
        self.state.current_time
    }

    fn cell_count(&self) -> usize {
        self.state.canonical_state.cell_count
    }
}
