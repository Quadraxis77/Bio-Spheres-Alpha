//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::rendering::{CellRenderer, OrientationGizmoRenderer, SplitRingRenderer};
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
    /// Orientation gizmo renderer
    pub gizmo_renderer: OrientationGizmoRenderer,
    /// Split ring renderer
    pub split_ring_renderer: SplitRingRenderer,
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
        let gizmo_renderer = OrientationGizmoRenderer::new(device, queue, surface_config);
        let split_ring_renderer = SplitRingRenderer::new(device, queue, surface_config);

        Self {
            canonical_state,
            renderer,
            gizmo_renderer,
            split_ring_renderer,
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

            // Render cells first
            self.renderer.render_in_pass(
                &mut render_pass,
                queue,
                &self.canonical_state,
                None,
                self.camera.position(),
                self.camera.rotation,
            );

            // Render orientation gizmo on the first cell if it exists
            if self.canonical_state.cell_count > 0 {
                let cell_position = self.canonical_state.positions[0];
                let cell_rotation = self.canonical_state.rotations[0];
                let cell_radius = self.canonical_state.radii[0];

                self.gizmo_renderer.render_in_pass(
                    &mut render_pass,
                    queue,
                    view_proj,
                    self.camera.position(),
                    cell_position,
                    cell_rotation,
                    cell_radius,
                );

                // For GPU scene, we don't have genome info readily available
                // Split rings would need to be implemented when genome data is accessible
                // TODO: Add split ring rendering when genome integration is complete
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
        self.current_time
    }

    fn cell_count(&self) -> usize {
        self.canonical_state.cell_count
    }
}

impl GpuScene {
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
