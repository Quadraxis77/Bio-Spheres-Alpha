//! Preview scene for genome editing.
//!
//! CPU-based simulation with GPU rendering, optimized for
//! editing and debugging genomes with small cell counts.

use crate::genome::Genome;
use crate::rendering::{CellRenderer, OrientationGizmoRenderer, SplitRingRenderer};
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
    /// Orientation gizmo renderer
    pub gizmo_renderer: OrientationGizmoRenderer,
    /// Split ring renderer
    pub split_ring_renderer: SplitRingRenderer,
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
        let gizmo_renderer = OrientationGizmoRenderer::new(device, queue, surface_config);
        let split_ring_renderer = SplitRingRenderer::new(device, queue, surface_config);

        Self {
            state,
            renderer,
            gizmo_renderer,
            split_ring_renderer,
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
        // Calculate view-projection matrix (same as used by cell renderer)
        let view_matrix = glam::Mat4::look_at_rh(
            self.camera.position(),
            self.camera.position() + self.camera.rotation * glam::Vec3::NEG_Z,
            self.camera.rotation * glam::Vec3::Y,
        );
        let aspect = self.renderer.width as f32 / self.renderer.height as f32;
        let proj_matrix = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        // Create command encoder for the entire frame
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Preview Scene Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Preview Scene Render Pass"),
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

            // Render cells first
            self.renderer.render_in_pass(
                &mut render_pass,
                queue,
                &self.state.canonical_state,
                Some(&self.genome),
                self.camera.position(),
                self.camera.rotation,
            );

            // Render orientation gizmo on the first cell if it exists
            if self.state.canonical_state.cell_count > 0 {
                let cell_position = self.state.canonical_state.positions[0];
                let cell_rotation = self.state.canonical_state.rotations[0];
                let cell_radius = self.state.canonical_state.radii[0];
                let mode_index = self.state.canonical_state.mode_indices[0];

                self.gizmo_renderer.render_in_pass(
                    &mut render_pass,
                    queue,
                    view_proj,
                    self.camera.position(),
                    cell_position,
                    cell_rotation,
                    cell_radius,
                );

                // Render split rings if the mode has split direction settings
                if mode_index < self.genome.modes.len() {
                    let mode = &self.genome.modes[mode_index];
                    
                    // Calculate split direction from pitch and yaw (same as BioSpheres-Q reference)
                    let pitch = mode.parent_split_direction.x.to_radians();
                    let yaw = mode.parent_split_direction.y.to_radians();
                    
                    // Use Euler rotation to match the division code
                    let split_direction_local = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0) * glam::Vec3::Z;
                    
                    // Debug: Log the split direction when it changes
                    log::debug!("Split direction - pitch: {:.1}°, yaw: {:.1}°, direction: {:?}", 
                        mode.parent_split_direction.x, mode.parent_split_direction.y, split_direction_local);
                    
                    self.split_ring_renderer.render_cell_rings(
                        &mut render_pass,
                        queue,
                        view_proj,
                        self.camera.position(),
                        cell_position,
                        cell_rotation,
                        cell_radius,
                        split_direction_local,
                    );
                }
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.gizmo_renderer.resize(device, width, height);
        self.split_ring_renderer.resize(device, width, height);
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

impl PreviewScene {
    /// Update gizmo configuration from UI state
    pub fn update_gizmo_config(&mut self, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        use crate::rendering::orientation_gizmo::GizmoConfig;
        let config = GizmoConfig::from_editor_state(editor_state);
        self.gizmo_renderer.update_config(&config);
    }

    /// Update split ring configuration from UI state
    pub fn update_split_ring_config(&mut self, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        use crate::rendering::split_rings::SplitRingConfig;
        let config = SplitRingConfig::from_editor_state(editor_state);
        self.split_ring_renderer.update_config(&config);
    }
}
