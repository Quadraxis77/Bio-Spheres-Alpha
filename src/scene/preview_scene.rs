//! Preview scene for genome editing.
//!
//! CPU-based simulation with GPU rendering, optimized for
//! editing and debugging genomes with small cell counts.

use crate::genome::Genome;
use crate::rendering::{AdhesionLineRenderer, CellRenderer, OrientationGizmoRenderer, SplitRingRenderer, TailRenderer, TailInstance};
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
    /// Adhesion line renderer
    pub adhesion_renderer: AdhesionLineRenderer,
    /// Orientation gizmo renderer
    pub gizmo_renderer: OrientationGizmoRenderer,
    /// Split ring renderer
    pub split_ring_renderer: SplitRingRenderer,
    /// Tail renderer for flagellocyte cells
    pub tail_renderer: TailRenderer,
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
    /// Whether to show adhesion lines
    pub show_adhesion_lines: bool,
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
        
        let mut state = PreviewState::new(capacity, &config);
        state.genome_hash = PreviewState::compute_genome_hash(&genome);
        state.update_initial_state(&genome, &config);

        let renderer = CellRenderer::new(device, queue, surface_config, capacity);
        let adhesion_renderer = AdhesionLineRenderer::new(device, queue, surface_config, capacity * 20); // 20 adhesions per cell max
        let gizmo_renderer = OrientationGizmoRenderer::new(device, queue, surface_config);
        let split_ring_renderer = SplitRingRenderer::new(device, queue, surface_config);
        let tail_renderer = TailRenderer::new(device, surface_config.format, capacity);

        Self {
            state,
            renderer,
            adhesion_renderer,
            gizmo_renderer,
            split_ring_renderer,
            tail_renderer,
            genome,
            config,
            paused: false,
            camera: CameraController::new_for_preview_scene(),
            last_ui_time_value: 0.0,
            show_adhesion_lines: true,
        }
    }

    /// Update genome immediately (no debouncing)
    pub fn update_genome(&mut self, new_genome: &Genome) {
        let new_hash = PreviewState::compute_genome_hash(new_genome);
        
        // Only update if actually changed
        if new_hash != self.state.genome_hash {
            log::info!("Applying genome change, triggering resimulation");
            
            self.genome = new_genome.clone();
            self.state.genome_hash = new_hash;
            self.state.clear_checkpoints();
            self.state.update_initial_state(&self.genome, &self.config);
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
            let genome_changed = false; // Already handled in apply_pending_genome
            self.state.run_resimulation(&self.genome, &self.config, genome_changed);
        }
        
        // No automatic time advancement - time is controlled entirely by the slider
    }

    fn render(
        &mut self, 
        device: &wgpu::Device, 
        queue: &wgpu::Queue, 
        view: &wgpu::TextureView, 
        cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>, 
        _world_diameter: f32,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        lod_debug_colors: bool,
        outline_width: f32,
    ) {
        // Calculate view-projection matrix (same as used by cell renderer)
        let view_matrix = glam::Mat4::look_at_rh(
            self.camera.position(),
            self.camera.position() + self.camera.rotation * glam::Vec3::NEG_Z,
            self.camera.rotation * glam::Vec3::Y,
        );
        let aspect = self.renderer.width as f32 / self.renderer.height as f32;
        let proj_matrix = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        // Pass 1: Clear background
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Preview Scene Clear Encoder"),
            });

            {
                let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Preview Scene Clear Pass"),
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
                // Pass ends here, just clearing
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Pass 2: Render cells with depth pre-pass (opaque rendering)
        self.renderer.render_with_depth_prepass(
            device,
            queue,
            view,
            &self.state.canonical_state,
            Some(&self.genome),
            cell_type_visuals,
            self.camera.position(),
            self.camera.rotation,
            self.state.current_time,  // Use simulation time for animation
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
            lod_debug_colors,
            outline_width,
        );

        // Pass 2.5: Render flagellocyte tails
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Preview Scene Tail Encoder"),
            });
            
            // Build tail instances for flagellocyte cells
            let tail_instances = self.build_tail_instances(cell_type_visuals);
            
            if !tail_instances.is_empty() {
                self.tail_renderer.render(
                    device,
                    queue,
                    &mut encoder,
                    view,
                    &self.renderer.depth_view,
                    &tail_instances,
                    self.camera.position(),
                    self.camera.rotation,
                    self.state.current_time,
                    self.renderer.width,
                    self.renderer.height,
                );
            }
            
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Pass 3: Render overlays (adhesion lines, gizmos, split rings)
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Preview Scene Overlay Encoder"),
            });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Preview Scene Overlay Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve cells
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.renderer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve depth from cell rendering
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                // Render adhesion lines if enabled
                if self.show_adhesion_lines {
                    self.adhesion_renderer.render_in_pass(
                        &mut render_pass,
                        queue,
                        &self.state.canonical_state,
                        self.camera.position(),
                        self.camera.rotation,
                    );
                }

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
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.adhesion_renderer.resize(width, height);
        self.gizmo_renderer.resize(device, width, height);
        self.split_ring_renderer.resize(device, width, height);
        self.tail_renderer.resize(width, height);
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
    
    /// Build tail instances for flagellocyte cells
    fn build_tail_instances(&self, cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>) -> Vec<TailInstance> {
        use crate::cell::CellType;
        
        let mut instances = Vec::new();
        let flagellocyte_index = CellType::Flagellocyte as usize;
        
        // Get flagellocyte visuals for tail parameters
        let visuals = cell_type_visuals
            .and_then(|v| v.get(flagellocyte_index))
            .copied()
            .unwrap_or_default();
        
        for i in 0..self.state.canonical_state.cell_count {
            let mode_index = self.state.canonical_state.mode_indices[i];
            
            // Check if this cell is a flagellocyte
            let cell_type_index = if mode_index < self.genome.modes.len() {
                self.genome.modes[mode_index].cell_type as u32
            } else {
                0
            };
            
            if cell_type_index != CellType::Flagellocyte as u32 {
                continue;
            }
            
            let position = self.state.canonical_state.positions[i];
            let rotation = self.state.canonical_state.rotations[i];
            let radius = self.state.canonical_state.radii[i];
            
            // Get color from mode
            let color = if mode_index < self.genome.modes.len() {
                let mode = &self.genome.modes[mode_index];
                [mode.color.x, mode.color.y, mode.color.z, mode.opacity]
            } else {
                [0.8, 0.6, 0.9, 1.0] // Default purple for flagellocyte
            };
            
            // Calculate tail speed from swim_force
            let swim_force = if mode_index < self.genome.modes.len() {
                self.genome.modes[mode_index].swim_force
            } else {
                0.5
            };
            let tail_speed = swim_force * 15.0;
            
            instances.push(TailInstance {
                cell_position: position.to_array(),
                cell_radius: radius,
                rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                color,
                tail_length: visuals.tail_length,
                tail_thickness: visuals.tail_thickness,
                tail_amplitude: visuals.tail_amplitude,
                tail_frequency: visuals.tail_frequency,
                tail_speed,
                tail_taper: visuals.tail_taper,
                time: self.state.current_time,
                _pad: 0.0,
            });
        }
        
        instances
    }
}
