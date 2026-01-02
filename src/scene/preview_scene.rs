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
    /// Last time value from UI (for detecting slider changes)
    last_ui_time_value: f32,
}

impl PreviewScene {
    /// Create a new preview scene.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let capacity = 256; // Preview capacity limit
        let genome = Genome::default();
        let config = PhysicsConfig::default();
        
        let mut state = PreviewState::new(capacity);
        state.genome_hash = PreviewState::compute_genome_hash(&genome);
        state.update_initial_state(&genome);

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
            last_ui_time_value: 0.0,
        }
    }

    /// Update the genome and refresh the test cell if needed
    pub fn update_genome(&mut self, new_genome: &Genome) {
        let new_hash = PreviewState::compute_genome_hash(new_genome);
        
        // Check if genome has changed
        if new_hash != self.state.genome_hash {
            log::info!("Genome changed, clearing checkpoints and triggering resimulation");
            
            // Update the genome
            self.genome = new_genome.clone();
            self.state.genome_hash = new_hash;
            
            // Clear checkpoints since genome changed
            self.state.clear_checkpoints();
            
            // Update initial state with new genome
            self.state.update_initial_state(&self.genome);
            
            // Trigger resimulation from current time with new genome
            self.state.seek_to_time(self.state.current_time);
        }
    }
    
    /// Sync time slider value from UI
    /// Called each frame to check if user moved the slider
    /// Time slider range is 0-60 seconds
    pub fn sync_time_from_ui(&mut self, ui_time_value: f32, _max_duration: f32, _is_dragging: bool) {
        // Slider value is directly in seconds (0-60 range)
        let target_sim_time = ui_time_value;
        
        // Only update if time value actually changed significantly
        if (ui_time_value - self.last_ui_time_value).abs() > 0.01 {
            // Only seek if different from current time (avoid redundant resimulations)
            let needs_update = (self.state.current_time - target_sim_time).abs() > 0.01;
            
            if needs_update {
                self.state.seek_to_time(target_sim_time);
            }
            
            self.last_ui_time_value = ui_time_value;
        }
    }
    
    /// Get current simulation time for syncing back to UI
    pub fn get_time_for_ui(&self) -> f32 {
        self.state.current_time
    }
    
    /// Check if currently resimulating (for UI feedback)
    pub fn is_resimulating(&self) -> bool {
        self.state.is_resimulating
    }
}

impl Scene for PreviewScene {
    fn update(&mut self, _dt: f32) {
        // Preview mode is entirely slider-driven
        // Simulation only runs when seeking to a new time via the slider
        // No automatic time advancement
        
        // Check if there's a pending time seek (from slider or genome change)
        if self.state.target_time.is_some() {
            // Run resimulation to target time
            let genome_changed = false; // Already handled in update_genome
            self.state.run_resimulation(&self.genome, &self.config, genome_changed);
        }
        
        // No automatic time advancement - time is controlled entirely by the slider
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

            // Begin frame for instanced renderers
            self.gizmo_renderer.begin_frame();
            self.split_ring_renderer.begin_frame();

            // Queue gizmos and split rings for ALL cells
            for i in 0..self.state.canonical_state.cell_count {
                let cell_position = self.state.canonical_state.positions[i];
                let cell_rotation = self.state.canonical_state.rotations[i];
                let cell_radius = self.state.canonical_state.radii[i];
                let mode_index = self.state.canonical_state.mode_indices[i];

                // Queue orientation gizmo for this cell
                self.gizmo_renderer.queue_gizmo(cell_position, cell_rotation, cell_radius);

                // Queue split rings if the mode has split direction settings
                if mode_index < self.genome.modes.len() {
                    let mode = &self.genome.modes[mode_index];
                    
                    // Calculate split direction from pitch and yaw (same as BioSpheres-Q reference)
                    let pitch = mode.parent_split_direction.x.to_radians();
                    let yaw = mode.parent_split_direction.y.to_radians();
                    
                    // Use Euler rotation to match the division code
                    let split_direction_local = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0) * glam::Vec3::Z;
                    
                    self.split_ring_renderer.queue_rings(
                        cell_position,
                        cell_rotation,
                        cell_radius,
                        split_direction_local,
                    );
                }
            }

            // Render all queued gizmos and rings in batches
            self.gizmo_renderer.render_queued(&mut render_pass, queue, view_proj, self.camera.position());
            self.split_ring_renderer.render_queued(&mut render_pass, queue, view_proj, self.camera.position());
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
