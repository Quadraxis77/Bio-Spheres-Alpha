//! TabViewer implementation for the Bio-Spheres docking system.
//!
//! This module provides the PanelTabViewer which implements egui_dock's
//! TabViewer trait to render panel content and handle panel interactions.

use egui::{Id, Ui, WidgetText};
use egui_dock::{NodeIndex, SurfaceIndex, TabViewer};
use egui_dock::tab_viewer::OnCloseResponse;

use crate::ui::panel::Panel;
use crate::ui::panel_context::PanelContext;
use crate::ui::types::GlobalUiState;

/// TabViewer implementation for Bio-Spheres panels.
///
/// This struct implements the egui_dock TabViewer trait to provide
/// panel rendering and interaction handling for the docking system.
pub struct PanelTabViewer<'a> {
    /// Global UI state for visibility and lock settings
    pub state: &'a mut GlobalUiState,
    /// Panel context with simulation data
    pub context: &'a mut PanelContext<'a>,
    /// Viewport rectangle output (set when Viewport panel renders)
    pub viewport_rect: &'a mut Option<egui::Rect>,
}

impl<'a> PanelTabViewer<'a> {
    /// Create a new PanelTabViewer.
    pub fn new(
        state: &'a mut GlobalUiState,
        context: &'a mut PanelContext<'a>,
        viewport_rect: &'a mut Option<egui::Rect>,
    ) -> Self {
        Self {
            state,
            context,
            viewport_rect,
        }
    }
}

impl<'a> TabViewer for PanelTabViewer<'a> {
    type Tab = Panel;

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
        tab.display_name().into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        match tab {
            Panel::Viewport => render_viewport(ui, self.viewport_rect),
            Panel::LeftPanel => render_placeholder_panel(ui, "Left Panel"),
            Panel::RightPanel => render_placeholder_panel(ui, "Right Panel"),
            Panel::BottomPanel => render_placeholder_panel(ui, "Bottom Panel"),
            Panel::SceneManager => render_scene_manager(ui, self.context, self.state),
            Panel::CellInspector => render_cell_inspector(ui, self.context),
            Panel::GenomeEditor => render_genome_editor(ui, self.context),
            Panel::PerformanceMonitor => render_performance_monitor(ui, self.context),
            Panel::RenderingControls => render_rendering_controls(ui),
            Panel::TimeScrubber => render_time_scrubber(ui, self.context),
            Panel::ThemeEditor => render_theme_editor(ui),
            Panel::CameraSettings => render_camera_settings(ui, self.context),
            Panel::LightingSettings => render_lighting_settings(ui),
            Panel::Modes => render_modes(ui, self.context),
            Panel::NameTypeEditor => render_name_type_editor(ui, self.context),
            Panel::AdhesionSettings => render_adhesion_settings(ui, self.context),
            Panel::ParentSettings => render_parent_settings(ui, self.context),
            Panel::CircleSliders => render_circle_sliders(ui, self.context),
            Panel::QuaternionBall => render_quaternion_ball(ui, self.context),
            Panel::TimeSlider => render_time_slider(ui, self.context),
        }
    }

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        // Use a stable ID based on the panel variant
        Id::new(format!("panel_{:?}", tab))
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> OnCloseResponse {
        // Update visibility state when panel is closed
        let panel_name = format!("{:?}", tab);
        self.state.set_panel_visible(&panel_name, false);
        OnCloseResponse::Close
    }

    fn is_closeable(&self, tab: &Self::Tab) -> bool {
        // Check if close buttons are locked
        if self.state.lock_close_buttons {
            return false;
        }
        // Check if this specific panel is locked
        let panel_name = format!("{:?}", tab);
        if self.state.is_panel_locked(&panel_name) {
            return false;
        }
        // Use the panel's own closeable setting
        tab.is_closeable()
    }

    fn allowed_in_windows(&self, tab: &mut Self::Tab) -> bool {
        tab.allowed_in_windows()
    }

    fn is_viewport(&self, tab: &Self::Tab) -> bool {
        tab.is_viewport()
    }

    fn clear_background(&self, tab: &Self::Tab) -> bool {
        // Viewport should have transparent background to show 3D content
        !tab.is_viewport()
    }

    fn scroll_bars(&self, tab: &Self::Tab) -> [bool; 2] {
        // Viewport doesn't need scroll bars
        if tab.is_viewport() {
            [false, false]
        } else {
            [true, true]
        }
    }

    fn is_draggable(&self, tab: &Self::Tab) -> bool {
        // Check if tabs are locked globally
        if self.state.lock_tabs {
            return false;
        }
        // Check if this specific panel is locked
        let panel_name = format!("{:?}", tab);
        if self.state.is_panel_locked(&panel_name) {
            return false;
        }
        // Viewport is not draggable
        !tab.is_viewport()
    }

    fn context_menu(
        &mut self,
        ui: &mut Ui,
        tab: &mut Self::Tab,
        _surface: SurfaceIndex,
        _node: NodeIndex,
    ) {
        let panel_name = format!("{:?}", tab);
        let is_locked = self.state.is_panel_locked(&panel_name);

        if ui
            .checkbox(&mut is_locked.clone(), "Lock Panel")
            .on_hover_text("Prevent this panel from being moved or closed")
            .changed()
        {
            self.state.set_panel_locked(&panel_name, !is_locked);
        }
    }

    fn hide_tab_button(&self, tab: &Self::Tab) -> bool {
        // Hide tab button if tabs are locked globally
        if self.state.lock_tabs {
            return true;
        }
        
        // Hide tab button if this specific panel is locked
        let panel_name = format!("{:?}", tab);
        if self.state.is_panel_locked(&panel_name) {
            return true;
        }
        
        // Use the panel's own settings (viewport tabs are typically hidden)
        tab.is_viewport()
    }
}

// ============================================================================
// Panel Rendering Functions
// ============================================================================

/// Render a placeholder panel.
///
/// Placeholder panels are empty layout containers that provide structure
/// to the dock system but don't contain actual content.
fn render_placeholder_panel(ui: &mut Ui, name: &str) {
    ui.centered_and_justified(|ui| {
        ui.label(format!("{} (Empty)", name));
    });
}

/// Render the Viewport panel.
///
/// This panel displays the 3D scene and reports its screen rectangle
/// for mouse input filtering.
fn render_viewport(ui: &mut Ui, viewport_rect: &mut Option<egui::Rect>) {
    // Get the available rect for the viewport
    let rect = ui.available_rect_before_wrap();
    *viewport_rect = Some(rect);

    // Allocate the full space - the 3D scene will be rendered behind this
    let response = ui.allocate_rect(rect, egui::Sense::hover());

    // Draw a subtle border to indicate the viewport area
    if ui.is_rect_visible(rect) {
        let visuals = ui.visuals();
        ui.painter().rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color),
            egui::StrokeKind::Inside,
        );
    }

    // Show tooltip with viewport info on hover
    response.on_hover_text(format!(
        "Viewport: {:.0}x{:.0}",
        rect.width(),
        rect.height()
    ));
}

/// Render the SceneManager panel.
fn render_scene_manager(ui: &mut Ui, context: &mut PanelContext, state: &GlobalUiState) {
    ui.heading("Scene Manager");
    ui.separator();

    ui.label("Simulation Mode:");
    ui.horizontal(|ui| {
        let is_preview = context.is_preview_mode();
        let is_gpu = context.is_gpu_mode();

        if ui
            .selectable_label(is_preview, "Preview")
            .on_hover_text("Genome editor with CPU simulation")
            .clicked()
            && !is_preview
        {
            context.request_preview_mode();
        }

        if ui
            .selectable_label(is_gpu, "GPU")
            .on_hover_text("Full GPU simulation")
            .clicked()
            && !is_gpu
        {
            context.request_gpu_mode();
        }
    });

    ui.separator();

    // Show current scene info
    ui.label(format!("Current Mode: {}", state.current_mode.display_name()));
    ui.label(format!("Cell Count: {}", context.cell_count()));
    ui.label(format!("Time: {:.2}s", context.current_time()));

    if context.is_paused() {
        ui.colored_label(egui::Color32::YELLOW, "⏸ Paused");
    } else {
        ui.colored_label(egui::Color32::GREEN, "▶ Running");
    }
}

/// Render the CellInspector panel (placeholder).
fn render_cell_inspector(ui: &mut Ui, context: &PanelContext) {
    ui.heading("Cell Inspector");
    ui.separator();
    ui.label(format!("Total Cells: {}", context.cell_count()));
    ui.label("Select a cell to inspect its properties.");
    // TODO: Implement cell selection and property display
}

/// Render the GenomeEditor panel (placeholder).
fn render_genome_editor(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Genome Editor");
    ui.separator();
    ui.label(format!("Genome: {}", context.genome.name));
    ui.label(format!("Initial Mode: {}", context.genome.initial_mode));
    // TODO: Implement full genome editor
}

/// Render the PerformanceMonitor panel (placeholder).
fn render_performance_monitor(ui: &mut Ui, context: &PanelContext) {
    ui.heading("Performance");
    ui.separator();
    ui.label(format!("Cells: {}", context.cell_count()));
    ui.label(format!("Time: {:.2}s", context.current_time()));
    // TODO: Add FPS, frame time, GPU stats
}

/// Render the RenderingControls panel (placeholder).
fn render_rendering_controls(ui: &mut Ui) {
    ui.heading("Rendering");
    ui.separator();
    ui.label("Fog, bloom, and visual settings.");
    // TODO: Implement rendering controls
}

/// Render the TimeScrubber panel (placeholder).
fn render_time_scrubber(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Time Scrubber");
    ui.separator();

    let time = context.current_time();
    ui.label(format!("Current Time: {:.2}s", time));

    // Simple time slider placeholder
    let mut time_value = context.editor_state.time_value;
    if ui
        .add(egui::Slider::new(&mut time_value, 0.0..=10.0).text("Time"))
        .changed()
    {
        context.editor_state.time_value = time_value;
    }
}

/// Render the ThemeEditor panel (placeholder).
fn render_theme_editor(ui: &mut Ui) {
    ui.heading("Theme Editor");
    ui.separator();
    ui.label("Customize UI appearance.");
    // TODO: Implement theme editor
}

/// Render the CameraSettings panel (placeholder).
fn render_camera_settings(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Camera");
    ui.separator();

    let camera = &mut context.camera;
    ui.label(format!("Mode: {:?}", camera.mode));
    ui.label(format!("Distance: {:.1}", camera.distance));

    ui.add(
        egui::Slider::new(&mut camera.move_speed, 1.0..=50.0)
            .text("Move Speed")
            .logarithmic(true),
    );
    ui.add(
        egui::Slider::new(&mut camera.mouse_sensitivity, 0.001..=0.01)
            .text("Mouse Sensitivity")
            .logarithmic(true),
    );
    ui.checkbox(&mut camera.enable_spring, "Enable Spring");
}

/// Render the LightingSettings panel (placeholder).
fn render_lighting_settings(ui: &mut Ui) {
    ui.heading("Lighting");
    ui.separator();
    ui.label("Scene lighting configuration.");
    // TODO: Implement lighting settings
}

/// Render the Modes panel (placeholder).
fn render_modes(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Modes");
    ui.separator();
    ui.label(format!("Initial Mode: {}", context.genome.initial_mode));
    ui.label("Cell mode configuration and editing.");
    // TODO: Implement modes list widget
}

/// Render the NameTypeEditor panel (placeholder).
fn render_name_type_editor(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Name & Type");
    ui.separator();

    let mut name = context.genome.name.clone();
    if ui.text_edit_singleline(&mut name).changed() {
        context.genome.name = name;
    }
    // TODO: Implement type selection
}

/// Render the AdhesionSettings panel (placeholder).
fn render_adhesion_settings(ui: &mut Ui, _context: &mut PanelContext) {
    ui.heading("Adhesion");
    ui.separator();
    ui.label("Cell adhesion configuration.");
    // TODO: Implement adhesion settings
}

/// Render the ParentSettings panel (placeholder).
fn render_parent_settings(ui: &mut Ui, _context: &mut PanelContext) {
    ui.heading("Parent Settings");
    ui.separator();
    ui.label("Parent cell configuration with rotation controls.");
    // TODO: Implement parent settings with quaternion ball
}

/// Render the CircleSliders panel with pitch and yaw controls for parent split direction.
fn render_circle_sliders(ui: &mut Ui, context: &mut PanelContext) {
    use crate::ui::widgets::circular_slider_float;
    
    ui.checkbox(&mut context.editor_state.enable_snapping, "Enable Snapping (11.25°)");
    ui.add_space(10.0);
    
    // Ensure we have at least one mode
    if context.genome.modes.is_empty() {
        context.genome.modes.push(crate::genome::ModeSettings::default());
    }
    
    // Get the current mode (default to first mode if available)
    if let Some(mode) = context.genome.modes.get_mut(0) {
        // Calculate responsive slider size
        let available_width = ui.available_width();
        let max_radius = ((available_width - 40.0) / 2.0 - 20.0) / 2.0;
        let radius = max_radius.clamp(20.0, 60.0);
        
        // Side by side layout
        ui.horizontal(|ui| {
            ui.add_space(10.0);
            
            ui.vertical(|ui| {
                ui.label("Pitch:");
                circular_slider_float(
                    ui,
                    &mut mode.parent_split_direction.x,
                    -180.0,
                    180.0,
                    radius,
                    context.editor_state.enable_snapping,
                );
            });
            
            ui.vertical(|ui| {
                ui.label("Yaw:");
                circular_slider_float(
                    ui,
                    &mut mode.parent_split_direction.y,
                    -180.0,
                    180.0,
                    radius,
                    context.editor_state.enable_snapping,
                );
            });
        });
    }
}

/// Render the QuaternionBall panel (placeholder).
fn render_quaternion_ball(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Rotation");
    ui.separator();
    ui.checkbox(&mut context.editor_state.qball_snapping, "Enable Snapping");
    // TODO: Implement quaternion ball widget
}

/// Render the TimeSlider panel (placeholder).
fn render_time_slider(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Time Slider");
    ui.separator();

    let max_duration = context.editor_state.max_preview_duration;
    ui.add(
        egui::Slider::new(&mut context.editor_state.time_value, 0.0..=max_duration)
            .text("Preview Time"),
    );

    ui.add(
        egui::Slider::new(&mut context.editor_state.max_preview_duration, 1.0..=60.0)
            .text("Max Duration"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_title() {
        // Verify panel titles are generated correctly
        assert_eq!(Panel::Viewport.display_name(), "Viewport");
        assert_eq!(Panel::SceneManager.display_name(), "Scene Manager");
        assert_eq!(Panel::CellInspector.display_name(), "Cell Inspector");
    }

    #[test]
    fn test_viewport_is_special() {
        assert!(Panel::Viewport.is_viewport());
        assert!(!Panel::Viewport.is_closeable());
        assert!(!Panel::Viewport.allowed_in_windows());
    }

    #[test]
    fn test_regular_panels_are_closeable() {
        assert!(Panel::CellInspector.is_closeable());
        assert!(Panel::SceneManager.is_closeable());
        assert!(Panel::PerformanceMonitor.is_closeable());
    }
}
