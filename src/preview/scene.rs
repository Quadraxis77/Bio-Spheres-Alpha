use super::{PreviewState, PreviewRenderer};
use crate::genome::Genome;
use crate::simulation::PhysicsConfig;
use glam::{Vec3, Quat};

pub struct PreviewScene {
    pub state: PreviewState,
    pub renderer: PreviewRenderer,
    pub genome: Genome,
    pub config: PhysicsConfig,
    pub paused: bool,
    pub camera_pos: Vec3,
    pub camera_distance: f32,
}

impl PreviewScene {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        let mut state = PreviewState::new(256); // Preview capacity limit
        let genome = Genome::default();
        let config = PhysicsConfig::default();
        
        // Initialize with single cell at origin
        state.canonical_state.add_cell(
            Vec3::ZERO,           // position
            Vec3::ZERO,           // velocity
            Quat::IDENTITY,       // rotation
            Vec3::ZERO,           // angular_velocity
            1.0,                  // mass
            1.0,                  // radius
            0,                    // genome_id
            0,                    // mode_index
            0.0,                  // birth_time
            5.0,                  // split_interval
            1.0,                  // split_mass
            500.0,                // stiffness
        );
        
        state.genome_hash = PreviewState::compute_genome_hash(&genome);
        
        let renderer = PreviewRenderer::new(device, queue, surface_config);
        
        Self {
            state,
            renderer,
            genome,
            config,
            paused: false,
            camera_pos: Vec3::new(0.0, 0.0, 50.0),
            camera_distance: 50.0,
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }
        
        // TODO: Run physics simulation
        self.state.current_time += dt;
    }
    
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) {
        self.renderer.render(device, queue, view, &self.state.canonical_state, self.camera_pos, self.camera_distance);
    }
    
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
    }
}
