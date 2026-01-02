//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::genome::Genome;
use crate::rendering::CellRenderer;
use crate::scene::Scene;
use crate::simulation::{CanonicalState, PhysicsConfig};
use crate::simulation::cpu_physics;
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
    /// Genome for cell behavior (growth, division)
    pub genome: Genome,
    /// Accumulated time for fixed timestep physics
    time_accumulator: f32,
}

impl GpuScene {
    /// Create a new GPU scene.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let capacity = 10_000; // 10k cell cap for GPU scene
        // Use 64x64x64 grid for spatial partitioning
        let canonical_state = CanonicalState::with_grid_density(capacity, 64);
        let config = PhysicsConfig::default();

        let renderer = CellRenderer::new(device, queue, surface_config, capacity);

        Self {
            canonical_state,
            renderer,
            config,
            paused: false,
            camera: CameraController::new(),
            current_time: 0.0,
            genome: Genome::default(),
            time_accumulator: 0.0,
        }
    }

    /// Reset the simulation to initial state.
    pub fn reset(&mut self) {
        self.canonical_state.cell_count = 0;
        self.canonical_state.next_cell_id = 0;
        self.current_time = 0.0;
        self.time_accumulator = 0.0;
        self.paused = false;
        // Clear adhesion connections
        self.canonical_state.adhesion_connections.active_count = 0;
        self.canonical_state.adhesion_manager.reset();
    }

    /// Run physics step using CPU physics with genome-based features.
    fn run_physics(&mut self) {
        if self.canonical_state.cell_count == 0 {
            return;
        }
        
        // Use CPU physics with genome for nutrient growth and division
        // TODO: Replace with GPU compute shaders for better performance
        let _division_events = cpu_physics::physics_step_with_genome(
            &mut self.canonical_state,
            &self.genome,
            &self.config,
            self.current_time,
        );
    }
    
    /// Insert a cell at the given world position using genome settings.
    /// Also updates the stored genome for physics simulation.
    /// Returns the index of the inserted cell, or None if at capacity.
    pub fn insert_cell_from_genome(
        &mut self,
        world_position: glam::Vec3,
        genome: &Genome,
    ) -> Option<usize> {
        // Update stored genome for physics (growth/division)
        self.genome = genome.clone();
        
        let mode_idx = genome.initial_mode.max(0) as usize;
        let mode = genome.modes.get(mode_idx)?;
        
        // Calculate initial radius from mass (mass = 4/3 * pi * r^3 for unit density)
        let initial_mass = 1.0_f32;
        let radius = (initial_mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        self.canonical_state.add_cell(
            world_position,
            glam::Vec3::ZERO,                    // velocity
            genome.initial_orientation,          // rotation
            genome.initial_orientation,          // genome_orientation
            glam::Vec3::ZERO,                    // angular_velocity
            initial_mass,                        // mass
            radius,                              // radius
            0,                                   // genome_id (single genome for now)
            mode_idx,                            // mode_index
            self.current_time,                   // birth_time
            mode.split_interval,                 // split_interval
            mode.split_mass,                     // split_mass
            500.0,                               // stiffness (match preview scene)
        )
    }
    
    /// Convert screen coordinates to world position on a plane at the camera's focal point.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> glam::Vec3 {
        let width = self.renderer.width as f32;
        let height = self.renderer.height as f32;
        
        // Normalized device coordinates (-1 to 1)
        let ndc_x = (2.0 * screen_x / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / height); // Flip Y
        
        // Camera matrices
        let aspect = width / height;
        let fov = 45.0_f32.to_radians();
        
        // Calculate ray direction in view space
        let tan_half_fov = (fov / 2.0).tan();
        let ray_view = glam::Vec3::new(
            ndc_x * aspect * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        ).normalize();
        
        // Transform ray to world space
        let ray_world = self.camera.rotation * ray_view;
        
        // Place cell at a fixed distance from camera (use camera distance or default)
        let distance = self.camera.distance.max(10.0);
        
        self.camera.position() + ray_world * distance
    }
}


impl Scene for GpuScene {
    fn update(&mut self, dt: f32) {
        if self.paused || self.canonical_state.cell_count == 0 {
            return;
        }

        // Fixed timestep accumulator pattern
        self.time_accumulator += dt;
        let fixed_dt = self.config.fixed_timestep;
        
        // Run physics steps to catch up (max 4 steps per frame to avoid spiral of death)
        let max_steps = 4;
        let mut steps = 0;
        
        while self.time_accumulator >= fixed_dt && steps < max_steps {
            self.run_physics();
            self.current_time += fixed_dt;
            self.time_accumulator -= fixed_dt;
            steps += 1;
        }
        
        // If we hit max steps, discard remaining accumulated time to prevent buildup
        if steps >= max_steps {
            self.time_accumulator = 0.0;
        }
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        // Create command encoder for the entire frame
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Scene Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU Scene Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render cells with genome for proper coloring
            self.renderer.render_in_pass(
                &mut render_pass,
                queue,
                &self.canonical_state,
                Some(&self.genome),
                self.camera.position(),
                self.camera.rotation,
            );
        }

        queue.submit(std::iter::once(encoder.finish()));
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
