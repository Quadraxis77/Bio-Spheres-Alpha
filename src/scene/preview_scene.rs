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

        // Add initial test cell at origin
        let initial_mode_index = genome.initial_mode as usize;
        log::info!("Creating initial test cell with mode index: {}", initial_mode_index);
        
        if initial_mode_index < genome.modes.len() {
            let mode = &genome.modes[initial_mode_index];
            log::info!("Initial mode color: {:?}, opacity: {}", mode.color, mode.opacity);
            log::info!("Mode name: {}, cell_type: {}", mode.name, mode.cell_type);
        } else {
            log::error!("Invalid initial mode index: {} (genome has {} modes)", initial_mode_index, genome.modes.len());
        }
        
        let _ = state.canonical_state.add_cell(
            glam::Vec3::ZERO,           // position at origin
            glam::Vec3::ZERO,           // no initial velocity
            genome.initial_orientation, // use genome's initial orientation
            glam::Vec3::ZERO,           // no angular velocity
            1.0,                        // mass
            1.0,                        // radius (1 unit wide)
            0,                          // genome_id (first genome)
            initial_mode_index,         // mode_index (use genome's initial_mode)
            0.0,                        // birth_time
            10.0,                       // split_interval
            1.5,                        // split_mass
            10.0,                       // stiffness
        );

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

    /// Update the genome and refresh the test cell if needed
    pub fn update_genome(&mut self, new_genome: &Genome) {
        let new_hash = PreviewState::compute_genome_hash(new_genome);
        
        // Check if genome has changed
        if new_hash != self.state.genome_hash {
            log::info!("Genome changed, updating test cell");
            
            // Update the genome
            self.genome = new_genome.clone();
            self.state.genome_hash = new_hash;
            
            // Update the test cell's mode if it exists
            if self.state.canonical_state.cell_count > 0 {
                let initial_mode_index = self.genome.initial_mode as usize;
                if initial_mode_index < self.genome.modes.len() {
                    self.state.canonical_state.mode_indices[0] = initial_mode_index;
                    self.state.canonical_state.rotations[0] = self.genome.initial_orientation;
                    log::info!("Updated test cell to mode index: {}", initial_mode_index);
                }
            }
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
            Some(&self.genome),
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
