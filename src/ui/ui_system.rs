//! UI System for Bio-Spheres using egui-wgpu and egui-winit.
//!
//! This module provides the core UI rendering system that integrates egui
//! with the existing wgpu/winit application.

use egui_wgpu::ScreenDescriptor;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::ui::types::GlobalUiState;

/// The main UI system that manages egui rendering.
///
/// This struct coordinates between egui-winit for input handling and
/// egui-wgpu for GPU rendering.
pub struct UiSystem {
    /// egui context for immediate mode UI
    pub ctx: egui::Context,
    /// egui-winit state for input handling
    pub winit_state: egui_winit::State,
    /// egui-wgpu renderer
    pub renderer: egui_wgpu::Renderer,
    /// Global UI state
    pub state: GlobalUiState,
    /// Viewport rectangle for mouse filtering (set during rendering)
    pub viewport_rect: Option<egui::Rect>,
    /// Last applied UI scale for change detection
    last_scale: f32,
}

impl UiSystem {
    /// Create a new UI system.
    ///
    /// # Arguments
    /// * `device` - The wgpu device for creating GPU resources
    /// * `surface_format` - The texture format of the render surface
    /// * `window` - The winit window for input handling
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &Window,
    ) -> Self {
        // Create egui context
        let ctx = egui::Context::default();

        // Create egui-winit state for input handling
        let viewport_id = egui::ViewportId::ROOT;
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );

        // Create egui-wgpu renderer
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        // Create default UI state
        let state = GlobalUiState::default();

        Self {
            ctx,
            winit_state,
            renderer,
            state,
            viewport_rect: None,
            last_scale: 1.0,
        }
    }


    /// Handle a winit window event.
    ///
    /// Returns `true` if egui consumed the event (i.e., the event should not
    /// be passed to other systems like the camera controller).
    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    /// Check if egui wants pointer (mouse) input.
    ///
    /// Returns `true` if the mouse is over an egui UI element (excluding the viewport)
    /// or if egui is actively using the pointer (e.g., dragging a slider).
    pub fn wants_pointer_input(&self) -> bool {
        // Check if pointer is over viewport - if so, camera should get input
        if let Some(viewport) = self.viewport_rect {
            if let Some(pos) = self.ctx.pointer_hover_pos() {
                if viewport.contains(pos) {
                    return false;
                }
            }
        }
        
        // Otherwise, check if egui wants the pointer
        self.ctx.egui_wants_pointer_input() || self.ctx.is_pointer_over_egui()
    }

    /// Check if egui wants keyboard input.
    ///
    /// Returns `true` if egui has keyboard focus (e.g., a text field is active).
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.egui_wants_keyboard_input()
    }

    /// Begin a new egui frame.
    ///
    /// Call this at the start of each frame before rendering UI.
    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.winit_state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);
        
        // Clear viewport rect at the start of each frame
        self.viewport_rect = None;
    }

    /// End the egui frame and get the output.
    ///
    /// Call this after all UI rendering is complete.
    pub fn end_frame(&mut self) -> egui::FullOutput {
        self.ctx.end_pass()
    }

    /// Render egui output to the screen.
    ///
    /// This method handles texture updates, buffer uploads, and the actual
    /// rendering of egui primitives.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        screen_descriptor: ScreenDescriptor,
        output: egui::FullOutput,
    ) {
        // Handle platform output (clipboard, cursor, etc.)
        // Note: We don't have access to window here, so platform output
        // should be handled separately if needed
        
        // Process texture updates
        for (id, image_delta) in &output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, image_delta);
        }

        // Tessellate shapes into primitives
        let paint_jobs = self.ctx.tessellate(output.shapes, output.pixels_per_point);

        // Update buffers and get any callback command buffers
        let _command_buffers = self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Create render pass and render egui
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear - render on top of 3D scene
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render egui - need to forget lifetime for wgpu render pass
            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        // Free textures that are no longer needed
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    /// Get the egui context for rendering UI.
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    /// Set the viewport rectangle.
    ///
    /// This should be called when rendering the viewport panel to track
    /// where the 3D scene is displayed.
    pub fn set_viewport_rect(&mut self, rect: egui::Rect) {
        self.viewport_rect = Some(rect);
    }

    /// Get the current viewport rectangle.
    pub fn get_viewport_rect(&self) -> Option<egui::Rect> {
        self.viewport_rect
    }
}
