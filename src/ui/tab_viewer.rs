//! TabViewer implementation for the Bio-Spheres docking system.
//!
//! This module provides the PanelTabViewer which implements egui_dock's
//! TabViewer trait to render panel content and handle panel interactions.

use egui::{Id, Ui, WidgetText};
use egui_dock::{NodeIndex, SurfaceIndex, TabViewer};
use egui_dock::tab_viewer::OnCloseResponse;

use crate::ui::panel::Panel;
use crate::ui::types::SimulationMode;
use crate::ui::panel_context::PanelContext;
use crate::ui::types::GlobalUiState;
use crate::ui::widgets::{quaternion_ball, modes_buttons, modes_list_items};

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
            Panel::PerformanceMonitor => render_performance_monitor(ui, self.context, self.state),
            Panel::RenderingControls => render_rendering_controls(ui),
            Panel::TimeScrubber => render_time_scrubber(ui, self.context),
            Panel::ThemeEditor => render_theme_editor(ui),
            Panel::CameraSettings => render_camera_settings(ui, self.context),
            Panel::LightingSettings => render_lighting_settings(ui),
            Panel::GizmoSettings => render_gizmo_settings(ui, self.context),
            Panel::CellTypeVisuals => render_cell_type_visuals(ui, self.context),
            Panel::Modes => render_modes(ui, self.context),
            Panel::NameTypeEditor => render_name_type_editor(ui, self.context),
            Panel::AdhesionSettings => render_adhesion_settings(ui, self.context),
            Panel::ParentSettings => render_parent_settings(ui, self.context),
            Panel::CircleSliders => render_circle_sliders(ui, self.context),
            Panel::QuaternionBall => render_quaternion_ball(ui, self.context),
            Panel::TimeSlider => render_time_slider(ui, self.context),
            Panel::ModeGraph => render_mode_graph(ui, self.context),
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
    let _response = ui.allocate_rect(rect, egui::Sense::hover());

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
}

/// Render the SceneManager panel.
fn render_scene_manager(ui: &mut Ui, context: &mut PanelContext, _state: &GlobalUiState) {
    // Single button that switches between scenes
    let (button_text, button_color) = if context.is_preview_mode() {
        ("Live Simulation", egui::Color32::from_rgb(200, 100, 100)) // Red for live simulation
    } else {
        ("Genome Editor", egui::Color32::from_rgb(100, 200, 100)) // Green for genome editor
    };
    
    let available_width = ui.available_width();
    
    // Create a large, full-width button with custom styling
    let button_response = ui.allocate_response(
        egui::Vec2::new(available_width, 40.0), // Full width, 40px height
        egui::Sense::click()
    );
    
    // Draw button background
    let button_rect = button_response.rect;
    let fill_color = if button_response.hovered() {
        // Slightly brighter when hovered
        egui::Color32::from_rgb(
            ((button_color.r() as f32 * 1.1).min(255.0)) as u8,
            ((button_color.g() as f32 * 1.1).min(255.0)) as u8,
            ((button_color.b() as f32 * 1.1).min(255.0)) as u8,
        )
    } else {
        button_color
    };
    
    ui.painter().rect_filled(
        button_rect,
        4.0,
        fill_color
    );
    
    // Draw centered text in bold, large font
    ui.painter().text(
        button_rect.center(),
        egui::Align2::CENTER_CENTER,
        button_text,
        egui::FontId::proportional(18.0), // Large text
        egui::Color32::WHITE, // White text for good contrast
    );
    
    // Handle button click
    if button_response.clicked() {
        if context.is_preview_mode() {
            context.request_gpu_mode();
        } else {
            context.request_preview_mode();
        }
    }
    
    // Show play/pause and reset controls only in GPU mode
    if context.is_gpu_mode() {
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(5.0);
        
        ui.horizontal(|ui| {
            // Play/Pause toggle button with symbol
            let is_paused = context.is_paused();
            let play_pause_symbol = if is_paused { "▶" } else { "⏸" };
            let play_pause_tooltip = if is_paused { "Play" } else { "Pause" };
            
            if ui.add_sized(
                [40.0, 40.0],
                egui::Button::new(egui::RichText::new(play_pause_symbol).size(20.0))
            ).on_hover_text(play_pause_tooltip).clicked() {
                context.request_toggle_pause();
            }
            
            // Reset button with symbol
            if ui.add_sized(
                [40.0, 40.0],
                egui::Button::new(egui::RichText::new("⟲").size(20.0))
            ).on_hover_text("Reset").clicked() {
                context.request_reset();
            }
        });
        
        ui.add_space(5.0);
        
        // Show simulation info
        ui.label(format!("Cells: {}", context.cell_count()));
        ui.label(format!("Time: {:.1}s", context.current_time()));
        if context.is_paused() {
            ui.colored_label(egui::Color32::YELLOW, "⏸ Paused");
        }
    }
}

/// Render the CellInspector panel.
fn render_cell_inspector(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Cell Inspector");
    ui.separator();
    ui.label(format!("Total Cells: {}", context.cell_count()));
    
    // Get the inspected cell index from radial menu state
    let inspected_cell = context.editor_state.radial_menu.inspected_cell;
    
    if let Some(cell_idx) = inspected_cell {
        // Check if we can access the GPU scene
        if let Some(gpu_scene) = context.scene_manager.gpu_scene() {
            let state = &gpu_scene.canonical_state;
            
            // Validate cell index is still valid
            if cell_idx >= state.cell_count {
                ui.colored_label(egui::Color32::RED, "Cell no longer exists");
                if ui.button("Clear Selection").clicked() {
                    context.editor_state.radial_menu.inspected_cell = None;
                }
                return;
            }
            
            ui.add_space(8.0);
            ui.label(format!("Cell #{}", cell_idx));
            ui.separator();
            
            // Position
            let pos = state.positions[cell_idx];
            ui.label(format!("Position: ({:.2}, {:.2}, {:.2})", pos.x, pos.y, pos.z));
            
            // Velocity
            let vel = state.velocities[cell_idx];
            let speed = vel.length();
            ui.label(format!("Velocity: ({:.2}, {:.2}, {:.2})", vel.x, vel.y, vel.z));
            ui.label(format!("Speed: {:.2}", speed));
            
            // Mass and radius
            let mass = state.masses[cell_idx];
            let radius = state.radii[cell_idx];
            ui.label(format!("Mass: {:.2}", mass));
            ui.label(format!("Radius: {:.2}", radius));
            
            // Division info
            let split_mass = state.split_masses[cell_idx];
            let split_interval = state.split_intervals[cell_idx];
            let split_count = state.split_counts[cell_idx];
            ui.add_space(4.0);
            ui.label("Division:");
            ui.label(format!("  Mass threshold: {:.2}", split_mass));
            ui.label(format!("  Progress: {:.0}%", (mass / split_mass * 100.0).min(100.0)));
            ui.label(format!("  Interval: {:.1}s", split_interval));
            ui.label(format!("  Split count: {}", split_count));
            
            // Mode info
            let mode_idx = state.mode_indices[cell_idx];
            let genome_id = state.genome_ids[cell_idx];
            ui.add_space(4.0);
            ui.label(format!("Mode index: {}", mode_idx));
            ui.label(format!("Genome ID: {}", genome_id));
            
            // Get mode name from genome if available
            if genome_id < gpu_scene.genomes.len() {
                let genome = &gpu_scene.genomes[genome_id];
                if mode_idx < genome.modes.len() {
                    let mode_name = &genome.modes[mode_idx].name;
                    ui.label(format!("Mode name: {}", mode_name));
                }
            }
            
            // Birth time and age
            let birth_time = state.birth_times[cell_idx];
            let current_time = gpu_scene.current_time;
            let age = current_time - birth_time;
            ui.add_space(4.0);
            ui.label(format!("Birth time: {:.1}s", birth_time));
            ui.label(format!("Age: {:.1}s", age));
            
            ui.add_space(12.0);
            ui.separator();
            
            // Load Genome button
            if genome_id < gpu_scene.genomes.len() {
                if ui.button("📋 Load Genome in Editor").clicked() {
                    // Clone the genome to load it into the editor
                    let genome_to_load = gpu_scene.genomes[genome_id].clone();
                    *context.genome = genome_to_load;
                    // Switch to preview mode (genome editor)
                    context.request_preview_mode();
                    println!("Loaded genome {} into editor", genome_id);
                }
                ui.label("Opens genome editor with this cell's genome");
            }
            
            ui.add_space(8.0);
            if ui.button("Clear Selection").clicked() {
                context.editor_state.radial_menu.inspected_cell = None;
            }
        } else {
            ui.label("GPU scene not available");
        }
    } else {
        ui.add_space(8.0);
        ui.label("Use the Inspect tool (🔍) to select a cell.");
        ui.label("Hold Alt to open the radial menu.");
    }
}

/// Render the GenomeEditor panel (placeholder).
fn render_genome_editor(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Genome Editor");
    ui.separator();
    ui.label(format!("Genome: {}", context.genome.name));
    ui.label(format!("Initial Mode: {}", context.genome.initial_mode));
    
    ui.separator();
    ui.label("Use the individual panels (Rotation, Parent Settings, etc.) to edit genome properties.");
}

/// Render the PerformanceMonitor panel.
fn render_performance_monitor(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    let perf = context.performance;
    
    egui::ScrollArea::vertical().show(ui, |ui| {
        // FPS and Frame Time section
        ui.heading("Frame Rate");
        ui.separator();
        
        ui.horizontal(|ui| {
            ui.label("FPS:");
            ui.label(format!("{:.1}", perf.fps()));
        });
        
        ui.horizontal(|ui| {
            ui.label("Frame Time:");
            ui.label(format!("{:.2} ms", perf.average_frame_time_ms()));
        });
        
        ui.horizontal(|ui| {
            ui.label("Min/Max:");
            ui.label(format!("{:.2} / {:.2} ms", perf.min_frame_time_ms(), perf.max_frame_time_ms()));
        });
        
        // Frame time graph
        let frame_times: Vec<f32> = perf.frame_time_history().collect();
        if !frame_times.is_empty() {
            let max_time = frame_times.iter().cloned().fold(16.67, f32::max);
            ui.add_space(4.0);
            
            let plot_height = 40.0;
            let (response, painter) = ui.allocate_painter(
                egui::vec2(ui.available_width(), plot_height),
                egui::Sense::hover(),
            );
            let rect = response.rect;
            
            // Background
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));
            
            // 16.67ms line (60 FPS target)
            let target_y = rect.bottom() - (16.67 / max_time) * rect.height();
            painter.line_segment(
                [egui::pos2(rect.left(), target_y), egui::pos2(rect.right(), target_y)],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 100, 50)),
            );
            
            // Frame time bars
            let bar_width = rect.width() / frame_times.len() as f32;
            for (i, &time) in frame_times.iter().enumerate() {
                let x = rect.left() + i as f32 * bar_width;
                let height = (time / max_time) * rect.height();
                let y = rect.bottom() - height;
                
                let color = if time > 16.67 {
                    egui::Color32::from_rgb(200, 80, 80)
                } else {
                    egui::Color32::from_rgb(80, 200, 80)
                };
                
                painter.rect_filled(
                    egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(bar_width.max(1.0), height)),
                    0.0,
                    color,
                );
            }
        }
        
        ui.add_space(8.0);
        
        // Simulation section
        ui.heading("Simulation");
        ui.separator();
        
        ui.horizontal(|ui| {
            ui.label("Cells:");
            ui.label(format!("{}", context.cell_count()));
        });
        
        ui.horizontal(|ui| {
            ui.label("Time:");
            ui.label(format!("{:.2}s", context.current_time()));
        });
        
        ui.horizontal(|ui| {
            ui.label("Mode:");
            ui.label(format!("{:?}", context.current_mode));
        });
        
        // Culling section (GPU mode only)
        if context.current_mode == crate::ui::types::SimulationMode::Gpu {
            ui.add_space(8.0);
            ui.heading("GPU Culling");
            ui.separator();
            
            let (total, visible, frustum_culled, occluded) = perf.culling_stats();
            
            ui.horizontal(|ui| {
                ui.label("Total Cells:");
                ui.label(format!("{}", total));
            });
            
            ui.horizontal(|ui| {
                ui.label("Visible:");
                ui.label(format!("{}", visible));
            });
            
            ui.horizontal(|ui| {
                ui.label("Frustum Culled:");
                ui.label(format!("{}", frustum_culled));
            });
            
            ui.horizontal(|ui| {
                ui.label("Occluded:");
                ui.label(format!("{}", occluded));
            });
            
            if total > 0 {
                let cull_percent = ((total - visible) as f32 / total as f32) * 100.0;
                ui.horizontal(|ui| {
                    ui.label("Cull Rate:");
                    ui.label(format!("{:.1}%", cull_percent));
                });
            }
            
            ui.add_space(4.0);
            
            // Enable/disable toggles
            ui.checkbox(&mut state.frustum_enabled, "Enable Frustum Culling");
            ui.checkbox(&mut state.occlusion_enabled, "Enable Occlusion Culling");
            
            if state.occlusion_enabled {
                ui.add_space(4.0);
                
                ui.label("Depth Bias:");
                ui.add(egui::Slider::new(&mut state.occlusion_bias, -0.1..=0.1)
                    .step_by(0.001)
                    .fixed_decimals(3));
                
                ui.add_space(4.0);
                
                ui.label("Mip Level (-1 = auto):");
                ui.add(egui::Slider::new(&mut state.occlusion_mip_override, -1..=10));
                
                ui.add_space(4.0);
                
                ui.label("Min Screen Size (px):");
                ui.add(egui::Slider::new(&mut state.occlusion_min_screen_size, 0.0..=100.0)
                    .step_by(1.0)
                    .fixed_decimals(0));
                
                ui.add_space(4.0);
                
                ui.label("Min Distance:");
                ui.add(egui::Slider::new(&mut state.occlusion_min_distance, 0.0..=100.0)
                    .step_by(0.5)
                    .fixed_decimals(1));
            }
        }
        
        ui.add_space(8.0);
        
        // CPU section
        ui.heading("CPU");
        ui.separator();
        
        ui.horizontal(|ui| {
            ui.label("Cores:");
            ui.label(format!("{}", perf.cpu_core_count()));
        });
        
        ui.horizontal(|ui| {
            ui.label("Usage:");
            ui.label(format!("{:.1}%", perf.cpu_usage_total()));
        });
        
        // Per-core usage bars
        let core_usage = perf.cpu_usage_per_core();
        if !core_usage.is_empty() {
            ui.add_space(4.0);
            
            let bar_height = 8.0;
            let spacing = 2.0;
            let total_height = core_usage.len() as f32 * (bar_height + spacing);
            
            let (response, painter) = ui.allocate_painter(
                egui::vec2(ui.available_width(), total_height),
                egui::Sense::hover(),
            );
            let rect = response.rect;
            
            for (i, &usage) in core_usage.iter().enumerate() {
                let y = rect.top() + i as f32 * (bar_height + spacing);
                let bar_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.left(), y),
                    egui::vec2(rect.width(), bar_height),
                );
                
                // Background
                painter.rect_filled(bar_rect, 2.0, egui::Color32::from_gray(40));
                
                // Usage bar
                let usage_width = (usage / 100.0) * rect.width();
                let usage_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.left(), y),
                    egui::vec2(usage_width, bar_height),
                );
                
                let color = if usage > 80.0 {
                    egui::Color32::from_rgb(200, 80, 80)
                } else if usage > 50.0 {
                    egui::Color32::from_rgb(200, 180, 80)
                } else {
                    egui::Color32::from_rgb(80, 180, 80)
                };
                
                painter.rect_filled(usage_rect, 2.0, color);
            }
        }
        
        ui.add_space(8.0);
        
        // Memory section
        ui.heading("Memory");
        ui.separator();
        
        let mem_used_mb = perf.memory_used() as f64 / (1024.0 * 1024.0);
        let mem_total_mb = perf.memory_total() as f64 / (1024.0 * 1024.0);
        let mem_used_gb = mem_used_mb / 1024.0;
        let mem_total_gb = mem_total_mb / 1024.0;
        
        ui.horizontal(|ui| {
            ui.label("Used:");
            if mem_used_gb >= 1.0 {
                ui.label(format!("{:.2} GB", mem_used_gb));
            } else {
                ui.label(format!("{:.0} MB", mem_used_mb));
            }
        });
        
        ui.horizontal(|ui| {
            ui.label("Total:");
            ui.label(format!("{:.1} GB", mem_total_gb));
        });
        
        // Memory usage bar
        let usage_percent = perf.memory_usage_percent();
        ui.add_space(4.0);
        
        let bar_height = 12.0;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), bar_height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        
        // Background
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));
        
        // Usage bar
        let usage_width = (usage_percent / 100.0) * rect.width();
        let usage_rect = egui::Rect::from_min_size(
            rect.min,
            egui::vec2(usage_width, bar_height),
        );
        
        let color = if usage_percent > 90.0 {
            egui::Color32::from_rgb(200, 80, 80)
        } else if usage_percent > 70.0 {
            egui::Color32::from_rgb(200, 180, 80)
        } else {
            egui::Color32::from_rgb(80, 140, 200)
        };
        
        painter.rect_filled(usage_rect, 2.0, color);
        
        // Percentage text
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.1}%", usage_percent),
            egui::FontId::default(),
            egui::Color32::WHITE,
        );
    });
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

/// Render the Modes panel with full functionality.
fn render_modes(ui: &mut Ui, context: &mut PanelContext) {
    // The genome should always have 40 modes by default
    if context.genome.modes.is_empty() {
        log::warn!("Genome has no modes, this should not happen with default genome");
        return;
    }
    
    // Get current selected mode index from editor state
    let mut selected_index = context.editor_state.selected_mode_index;
    let mut initial_mode = context.genome.initial_mode as usize;
    
    // Clamp selected index to valid range
    if selected_index >= context.genome.modes.len() {
        selected_index = 0;
        context.editor_state.selected_mode_index = 0;
    }
    
    // Clamp initial mode to valid range
    if initial_mode >= context.genome.modes.len() {
        initial_mode = 0;
        context.genome.initial_mode = 0;
    }
    
    // Control buttons (Copy Into and Reset) - more compact layout
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0; // Reduce button spacing
        let (copy_into_clicked, reset_clicked) = modes_buttons(
            ui,
            context.genome.modes.len(),
            selected_index,
            initial_mode,
        );
        
        if copy_into_clicked {
            let selected_idx = selected_index;
            if selected_idx < context.genome.modes.len() {
                // Enter copy into mode - user will click on target mode directly
                context.editor_state.copy_into_dialog_open = true;
                context.editor_state.copy_into_source = selected_idx;
                log::info!("Entered copy into mode for mode {}", selected_idx);
            }
        }
        
        if reset_clicked {
            // Reset the selected mode to its original default values
            if selected_index < context.genome.modes.len() {
                // Regenerate the original default color for this mode index
                let i = selected_index;
                let hue = (i as f32 * 360.0 / 40.0) % 360.0; // Distribute hues evenly
                let saturation = 0.7 + (i % 3) as f32 * 0.1; // Vary saturation slightly
                let value = 0.8 + (i % 2) as f32 * 0.1; // Vary brightness slightly
                
                // Convert HSV to RGB (same logic as in Default implementation)
                let c = value * saturation;
                let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
                let m = value - c;
                
                let (r_prime, g_prime, b_prime) = if hue < 60.0 {
                    (c, x, 0.0)
                } else if hue < 120.0 {
                    (x, c, 0.0)
                } else if hue < 180.0 {
                    (0.0, c, x)
                } else if hue < 240.0 {
                    (0.0, x, c)
                } else if hue < 300.0 {
                    (x, 0.0, c)
                } else {
                    (c, 0.0, x)
                };
                
                let r = ((r_prime + m) * 255.0) as u8;
                let g = ((g_prime + m) * 255.0) as u8;
                let b = ((b_prime + m) * 255.0) as u8;
                
                // Reset to original default values
                context.genome.modes[selected_index] = crate::genome::ModeSettings {
                    name: format!("M{}", selected_index + 1), // M1, M2, M3, etc.
                    default_name: format!("M{}", selected_index + 1), // Same as name
                    color: glam::Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0), // Convert to Vec3
                    opacity: 1.0,
                    emissive: 0.0,
                    cell_type: 0, // Default to Photocyte
                    parent_make_adhesion: false, // Default to no adhesion
                    split_mass: 1.5,
                    split_interval: 5.0,
                    nutrient_gain_rate: 0.2,
                    max_cell_size: 2.0,
                    split_ratio: 0.5,
                    nutrient_priority: 1.0,
                    prioritize_when_low: true,
                    parent_split_direction: glam::Vec2::ZERO,
                    max_adhesions: 20,
                    min_adhesions: 0,
                    enable_parent_angle_snapping: true,
                    max_splits: -1, // -1 means infinite
                    mode_a_after_splits: -1,
                    mode_b_after_splits: -1,
                    swim_force: 0.5,
                    child_a: crate::genome::ChildSettings {
                        mode_number: selected_index as i32,
                        ..Default::default()
                    },
                    child_b: crate::genome::ChildSettings {
                        mode_number: selected_index as i32,
                        ..Default::default()
                    },
                    adhesion_settings: crate::genome::AdhesionSettings::default(),
                };
                log::info!("Reset mode M{} to original defaults", selected_index + 1);
            }
        }
    });
    
    ui.add_space(4.0); // Reduced spacing instead of separator
    
    // Show copy into mode indicator just above the modes list
    if context.editor_state.copy_into_dialog_open {
        ui.colored_label(egui::Color32::YELLOW, "Select target mode to copy into:");
        if ui.small_button("Cancel").clicked() {
            context.editor_state.copy_into_dialog_open = false;
            log::info!("Cancelled copy into mode");
        }
        ui.add_space(5.0);
    }
    
    // Prepare modes data for the widget (name, color tuples)
    let modes_data: Vec<(String, (u8, u8, u8))> = context.genome.modes
        .iter()
        .map(|mode| {
            let color_vec3 = mode.color;
            let r = (color_vec3.x * 255.0) as u8;
            let g = (color_vec3.y * 255.0) as u8;
            let b = (color_vec3.z * 255.0) as u8;
            (mode.name.clone(), (r, g, b))
        })
        .collect();
    
    // Modes list with compact spacing
    ui.spacing_mut().item_spacing.y = 2.0; // Reduce vertical spacing between mode items
    let available_width = ui.available_width();
    let copy_into_mode = context.editor_state.copy_into_dialog_open;
    
    let (selection_changed, initial_changed, rename_index, color_change) = modes_list_items(
        ui,
        &modes_data,
        &mut selected_index,
        &mut initial_mode,
        available_width,
        copy_into_mode,
        &mut context.editor_state.color_picker_state,
    );
    
    // Handle mode selection change
    if selection_changed {
        // If in copy into mode, this is the target selection
        if copy_into_mode {
            let source_idx = context.editor_state.copy_into_source;
            let target_idx = selected_index;

            if source_idx != target_idx && source_idx < context.genome.modes.len()
                && target_idx < context.genome.modes.len() {
                // Copy all settings from source to target (including color, except name)
                let source_mode = context.genome.modes[source_idx].clone();
                let target_name = context.genome.modes[target_idx].name.clone();
                context.genome.modes[target_idx] = source_mode;
                context.genome.modes[target_idx].name = target_name;
                log::info!("Copied mode {} into mode {}", source_idx, target_idx);
            }

            // Exit copy into mode
            context.editor_state.copy_into_dialog_open = false;
        } else {
            // Normal mode selection
            context.editor_state.selected_mode_index = selected_index;
            
            // Initialize editor state orientations from the selected mode's genome data
            // This ensures the quaternion balls show the correct orientation when switching modes
            if selected_index < context.genome.modes.len() {
                context.editor_state.child_a_orientation = context.genome.modes[selected_index].child_a.orientation;
                context.editor_state.child_b_orientation = context.genome.modes[selected_index].child_b.orientation;
            }
            
            log::info!("Mode selection changed to: {}", selected_index + 1);
        }
    }
    
    // Handle initial mode change
    if initial_changed {
        context.genome.initial_mode = initial_mode as i32;
    }
    
    // Handle rename request
    if let Some(mode_index) = rename_index {
        context.editor_state.renaming_mode = Some(mode_index);
        context.editor_state.rename_buffer = context.genome.modes[mode_index].name.clone();
    }
    
    // Handle color change
    if let Some((mode_index, new_color)) = color_change {
        if mode_index < context.genome.modes.len() {
            let (r, g, b) = new_color;
            context.genome.modes[mode_index].color = glam::Vec3::new(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0
            );
        }
    }
    
    // Handle rename dialog - more compact
    if let Some(rename_index) = context.editor_state.renaming_mode {
        if rename_index < context.genome.modes.len() {
            ui.add_space(4.0); // Reduced spacing
            ui.label("Rename Mode:");
            
            let mut buffer = context.editor_state.rename_buffer.clone();
            let response = ui.text_edit_singleline(&mut buffer);
            context.editor_state.rename_buffer = buffer;
            
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0; // Compact button spacing
                if ui.small_button("OK").clicked() || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    context.genome.modes[rename_index].name = context.editor_state.rename_buffer.clone();
                    context.editor_state.renaming_mode = None;
                    context.editor_state.rename_buffer.clear();
                }
                
                if ui.small_button("Cancel").clicked() || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape))) {
                    context.editor_state.renaming_mode = None;
                    context.editor_state.rename_buffer.clear();
                }
            });
        }
    }
    
    // Show current mode info (for debugging/feedback) - more compact
    if selected_index < context.genome.modes.len() {
        ui.add_space(4.0); // Reduced spacing
        ui.small(format!("Selected: {}", context.genome.modes[selected_index].name));
        ui.small(format!("Initial: {}", context.genome.modes[initial_mode].name));
        ui.small(format!("Total: {}", context.genome.modes.len()));
    }
}

/// Helper function to create a color-coded group container
fn group_container(ui: &mut Ui, title: &str, color: egui::Color32, content: impl FnOnce(&mut Ui)) {
    let frame = egui::Frame::default()
        .fill(egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 30u8))
        .stroke(egui::Stroke::new(1.5, color))
        .corner_radius(egui::CornerRadius::same(4u8))
        .inner_margin(egui::Margin::same(8i8));

    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(egui::RichText::new(title).strong().color(color));
        ui.add_space(4.0);
        content(ui);
    });
    ui.add_space(6.0);
}

/// Render the NameTypeEditor panel.
fn render_name_type_editor(ui: &mut Ui, context: &mut PanelContext) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 2.0;

            // Three buttons at the top
            ui.horizontal(|ui| {
                if ui.button("Save Genome").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Genome", &["genome"])
                        .set_file_name(&format!("{}.genome", context.genome.name))
                        .save_file()
                    {
                        match context.genome.save_to_file(&path) {
                            Ok(()) => log::info!("Genome saved to {:?}", path),
                            Err(e) => log::error!("Failed to save genome: {}", e),
                        }
                    }
                }
                if ui.button("Load Genome").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Genome", &["genome"])
                        .pick_file()
                    {
                        match crate::genome::Genome::load_from_file(&path) {
                            Ok(loaded) => {
                                *context.genome = loaded;
                                log::info!("Genome loaded from {:?}", path);
                            }
                            Err(e) => log::error!("Failed to load genome: {}", e),
                        }
                    }
                }
                if ui.button("Genome Graph").clicked() {
                    context.editor_state.toggle_mode_graph_panel = true;
                }
            });

            ui.add_space(4.0);

            // Genome Name label and field on same line
            ui.horizontal(|ui| {
                ui.label("Genome Name:");
                ui.text_edit_singleline(&mut context.genome.name);
            });

            ui.add_space(4.0);

            // Get current mode
            let selected_idx = context.editor_state.selected_mode_index;
            if selected_idx >= context.genome.modes.len() {
                ui.label("No mode selected");
                return;
            }
            let mode = &mut context.genome.modes[selected_idx];

            // Type dropdown and checkbox on the same line
            ui.horizontal(|ui| {
                ui.label("Type:");
                let cell_types = crate::cell::types::CellType::names();
                egui::ComboBox::from_id_salt("cell_type")
                    .selected_text(cell_types[mode.cell_type as usize])
                    .show_ui(ui, |ui| {
                        for (i, type_name) in cell_types.iter().enumerate() {
                            ui.selectable_value(&mut mode.cell_type, i as i32, *type_name);
                        }
                    });

                ui.checkbox(&mut mode.parent_make_adhesion, "Make Adhesion");
            });
        });
}

/// Render the AdhesionSettings panel (placeholder).
fn render_adhesion_settings(ui: &mut Ui, context: &mut PanelContext) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Force content to fill available width
            ui.set_width(ui.available_width());
            ui.add_space(10.0);

            // Get current mode
            let selected_idx = context.editor_state.selected_mode_index;
            if selected_idx >= context.genome.modes.len() {
                ui.label("No mode selected");
                return;
            }
            let mode = &mut context.genome.modes[selected_idx];

            // Breaking Properties Group (Red)
            group_container(ui, "Breaking Properties", egui::Color32::from_rgb(200, 100, 100), |ui| {
                ui.checkbox(&mut mode.adhesion_settings.can_break, "Adhesion Can Break");

                ui.label("Adhesion Break Force:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.break_force, 0.1..=100.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.break_force).speed(0.1).range(0.1..=100.0));
                });
            });

            // Physical Properties Group (Orange)
            group_container(ui, "Physical Properties", egui::Color32::from_rgb(200, 150, 80), |ui| {
                ui.label("Adhesion Rest Length:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.rest_length, 0.5..=5.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.rest_length).speed(0.01).range(0.5..=5.0));
                });
            });

            // Linear Spring Group (Blue)
            group_container(ui, "Linear Spring", egui::Color32::from_rgb(100, 150, 200), |ui| {
                ui.label("Linear Spring Stiffness:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.linear_spring_stiffness, 0.1..=500.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.linear_spring_stiffness).speed(0.1).range(0.1..=500.0));
                });

                ui.label("Linear Spring Damping:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.linear_spring_damping, 0.0..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.linear_spring_damping).speed(0.01).range(0.0..=10.0));
                });
            });

            // Orientation Spring Group (Green)
            group_container(ui, "Orientation Spring", egui::Color32::from_rgb(100, 180, 120), |ui| {
                ui.label("Orientation Spring Stiffness:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.orientation_spring_stiffness, 0.1..=100.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.orientation_spring_stiffness).speed(0.1).range(0.1..=100.0));
                });

                ui.label("Orientation Spring Damping:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.orientation_spring_damping, 0.0..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.orientation_spring_damping).speed(0.01).range(0.0..=10.0));
                });

                ui.label("Max Angular Deviation:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.max_angular_deviation, 0.0..=180.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.max_angular_deviation).speed(0.1).range(0.0..=180.0));
                });
            });

            // Twist Constraint Group (Purple)
            group_container(ui, "Twist Constraint", egui::Color32::from_rgb(160, 120, 180), |ui| {
                ui.checkbox(&mut mode.adhesion_settings.enable_twist_constraint, "Enable Twist Constraint");

                ui.label("Twist Constraint Stiffness:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.twist_constraint_stiffness, 0.0..=2.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.twist_constraint_stiffness).speed(0.01).range(0.0..=2.0));
                });

                ui.label("Twist Constraint Damping:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.twist_constraint_damping, 0.0..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.twist_constraint_damping).speed(0.01).range(0.0..=10.0));
                });
            });
        });
}

/// Render the ParentSettings panel (placeholder).
fn render_parent_settings(ui: &mut Ui, context: &mut PanelContext) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Force content to fill available width
            ui.set_width(ui.available_width());
            ui.add_space(10.0);

            // Get current mode
            let selected_idx = context.editor_state.selected_mode_index;
            if selected_idx >= context.genome.modes.len() {
                ui.label("No mode selected");
                return;
            }
            let mode = &mut context.genome.modes[selected_idx];

            // Division Settings Group (Yellow)
            group_container(ui, "Division Settings", egui::Color32::from_rgb(200, 180, 80), |ui| {
                ui.label("Split Mass:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.split_mass, 1.0..=3.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.split_mass).speed(0.01).range(1.0..=3.0));
                });

                ui.label("Split Interval:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.split_interval, 1.0..=60.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.split_interval).speed(0.1).range(1.0..=60.0).suffix("s"));
                });
            });

            // Nutrient Settings Group (Green)
            group_container(ui, "Nutrient Settings", egui::Color32::from_rgb(100, 180, 120), |ui| {
                ui.label("Nutrient Priority:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.nutrient_priority, 0.1..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.nutrient_priority).speed(0.01).range(0.1..=10.0));
                });

                ui.checkbox(&mut mode.prioritize_when_low, "Prioritize When Low");
            });

            // Connection Settings Group (Cyan)
            group_container(ui, "Connection Settings", egui::Color32::from_rgb(100, 180, 200), |ui| {
                ui.label("Max Connections:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.max_adhesions, 0..=20).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.max_adhesions).speed(1).range(0..=20));
                });

                ui.label("Min Connections:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.min_adhesions, 0..=20).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.min_adhesions).speed(1).range(0..=20));
                });

                ui.label("Max Splits:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.max_splits, -1..=20).show_value(false));
                    
                    // Custom DragValue that shows infinity symbol for -1
                    let mut drag_value = egui::DragValue::new(&mut mode.max_splits)
                        .speed(0.1)
                        .range(-1.0..=20.0);
                    
                    // Custom formatter to show ∞ for -1
                    drag_value = drag_value.custom_formatter(|n, _| {
                        if n == -1.0 {
                            "∞".to_owned()
                        } else {
                            format!("{}", n as i32)
                        }
                    });
                    
                    // Custom parser to handle ∞ input
                    drag_value = drag_value.custom_parser(|s| {
                        if s == "∞" || s == "inf" || s == "infinity" {
                            Some(-1.0)
                        } else {
                            s.parse::<f64>().ok()
                        }
                    });
                    
                    ui.add(drag_value);
                });
            });

            // Appearance Settings Group (Purple)
            group_container(ui, "Appearance", egui::Color32::from_rgb(160, 120, 200), |ui| {
                ui.label("Opacity:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.opacity, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.opacity).speed(0.01).range(0.0..=1.0));
                });

                ui.label("Emissive Glow:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.emissive, 0.0..=2.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.emissive).speed(0.01).range(0.0..=2.0));
                });
            });
        });
}

/// Render the CircleSliders panel with pitch and yaw controls for parent split direction.
fn render_circle_sliders(ui: &mut Ui, context: &mut PanelContext) {
    use crate::ui::widgets::circular_slider_float;
    
    ui.checkbox(&mut context.editor_state.enable_snapping, "Enable Snapping (11.25°)");
    ui.add_space(4.0); // Reduced spacing
    
    // Ensure we have at least one mode
    if context.genome.modes.is_empty() {
        context.genome.modes.push(crate::genome::ModeSettings::default());
    }
    
    // Get the current mode (default to first mode if available)
    if let Some(mode) = context.genome.modes.get_mut(0) {
        // Calculate responsive slider size - use more of the available width
        let available_width = ui.available_width();
        let max_radius = ((available_width - 20.0) / 2.0 - 10.0) / 2.0; // Reduced margins
        let radius = max_radius.clamp(20.0, 80.0); // Allow larger radius
        
        // Side by side layout with minimal spacing
        ui.horizontal(|ui| {
            ui.add_space(4.0); // Reduced left margin
            
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
            
            ui.add_space(4.0); // Minimal spacing between sliders
            
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

/// Render the QuaternionBall panel with two quaternion balls for Child A and Child B.
fn render_quaternion_ball(ui: &mut Ui, context: &mut PanelContext) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
        ui.checkbox(&mut context.editor_state.qball_snapping, "Enable Snapping (11.25°)");
        ui.add_space(4.0); // Reduced spacing

        // Get selected mode for keep_adhesion checkboxes
        let selected_idx = context.editor_state.selected_mode_index;
        let has_valid_mode = selected_idx < context.genome.modes.len();

        // Calculate responsive ball size - use more of the available width
        let available_width = ui.available_width();
        // Reserve minimal space for padding and two balls side by side
        let max_radius = ((available_width - 20.0) / 2.0 - 8.0) / 2.0; // Reduced margins
        let ball_radius = max_radius.clamp(20.0, 80.0); // Allow larger radius
        let ball_container_width = ball_radius * 2.0 + 8.0; // Minimal container padding

        // Use persistent orientations from editor state
        let mut child_a_orientation = context.editor_state.child_a_orientation;
        let mut child_b_orientation = context.editor_state.child_b_orientation;

        // Display balls horizontally with controls directly below each ball
        ui.horizontal_top(|ui| {
            ui.add_space(4.0); // Reduced left margin

            // Ball 1 (Child A) with controls below
            ui.allocate_ui_with_layout(
                egui::vec2(ball_container_width, 0.0),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.label("Child A");

                    let response = quaternion_ball(
                        ui,
                        &mut child_a_orientation,
                        &mut context.editor_state.child_a_x_axis_lat,
                        &mut context.editor_state.child_a_x_axis_lon,
                        &mut context.editor_state.child_a_y_axis_lat,
                        &mut context.editor_state.child_a_y_axis_lon,
                        &mut context.editor_state.child_a_z_axis_lat,
                        &mut context.editor_state.child_a_z_axis_lon,
                        ball_radius,
                        context.editor_state.qball_snapping,
                        &mut context.editor_state.qball1_locked_axis,
                        &mut context.editor_state.qball1_initial_distance,
                    );

                    if response.changed() {
                        // Store the updated orientation back to editor state
                        context.editor_state.child_a_orientation = child_a_orientation;
                        // CRITICAL: Also update the actual genome mode's child orientation
                        if has_valid_mode {
                            context.genome.modes[selected_idx].child_a.orientation = child_a_orientation;
                        }
                    }

                    ui.add_space(2.0); // Reduced spacing

                    // Keep Adhesion checkbox for Child A - modify genome directly
                    if has_valid_mode {
                        ui.checkbox(&mut context.genome.modes[selected_idx].child_a.keep_adhesion, "Keep Adhesion");
                    } else {
                        ui.add_enabled(false, egui::Checkbox::new(&mut false, "Keep Adhesion"));
                    }

                    // Mode dropdown for Child A
                    ui.label("Mode:");
                    if has_valid_mode {
                        let current_mode = context.genome.modes[selected_idx].child_a.mode_number;
                        let mode_count = context.genome.modes.len();
                        let selected_text = if current_mode >= 0 && (current_mode as usize) < mode_count {
                            context.genome.modes[current_mode as usize].name.clone()
                        } else {
                            "Invalid".to_string()
                        };
                        // Collect mode info to avoid borrow issues
                        let mode_info: Vec<_> = context.genome.modes.iter()
                            .map(|m| (m.name.clone(), m.color))
                            .collect();
                        let mut new_mode: Option<i32> = None;
                        egui::ComboBox::from_id_salt("qball1_mode")
                            .selected_text(selected_text)
                            .width(ball_container_width - 8.0)
                            .show_ui(ui, |ui| {
                                let item_width = ui.available_width();
                                for (i, (name, color)) in mode_info.iter().enumerate() {
                                    let is_selected = current_mode == i as i32;
                                    let bg_color = egui::Color32::from_rgb(
                                        (color.x * 255.0) as u8,
                                        (color.y * 255.0) as u8,
                                        (color.z * 255.0) as u8,
                                    );
                                    // Calculate luminance for text contrast
                                    let luminance = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
                                    let text_color = if luminance > 0.5 {
                                        egui::Color32::BLACK
                                    } else {
                                        egui::Color32::WHITE
                                    };
                                    
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(item_width, 18.0),
                                        egui::Sense::click(),
                                    );
                                    
                                    // Draw background
                                    let bg = if response.hovered() {
                                        bg_color.gamma_multiply(1.2)
                                    } else {
                                        bg_color
                                    };
                                    ui.painter().rect_filled(rect, 2.0, bg);
                                    
                                    // Draw selection indicator
                                    if is_selected {
                                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                    }
                                    
                                    // Draw text centered
                                    ui.painter().text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        name,
                                        egui::FontId::default(),
                                        text_color,
                                    );
                                    
                                    if response.clicked() {
                                        new_mode = Some(i as i32);
                                    }
                                }
                            });
                        if let Some(mode_num) = new_mode {
                            context.genome.modes[selected_idx].child_a.mode_number = mode_num;
                        }
                    } else {
                        egui::ComboBox::from_id_salt("qball1_mode")
                            .selected_text("--")
                            .width(ball_container_width - 8.0)
                            .show_ui(ui, |_ui| {});
                    }
                }
            );

            ui.add_space(4.0); // Minimal spacing between balls

            // Ball 2 (Child B) with controls below
            ui.allocate_ui_with_layout(
                egui::vec2(ball_container_width, 0.0),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.label("Child B");

                    let response = quaternion_ball(
                        ui,
                        &mut child_b_orientation,
                        &mut context.editor_state.child_b_x_axis_lat,
                        &mut context.editor_state.child_b_x_axis_lon,
                        &mut context.editor_state.child_b_y_axis_lat,
                        &mut context.editor_state.child_b_y_axis_lon,
                        &mut context.editor_state.child_b_z_axis_lat,
                        &mut context.editor_state.child_b_z_axis_lon,
                        ball_radius,
                        context.editor_state.qball_snapping,
                        &mut context.editor_state.qball2_locked_axis,
                        &mut context.editor_state.qball2_initial_distance,
                    );

                    if response.changed() {
                        // Store the updated orientation back to editor state
                        context.editor_state.child_b_orientation = child_b_orientation;
                        // CRITICAL: Also update the actual genome mode's child orientation
                        if has_valid_mode {
                            context.genome.modes[selected_idx].child_b.orientation = child_b_orientation;
                        }
                    }

                    ui.add_space(2.0); // Reduced spacing

                    // Keep Adhesion checkbox for Child B - modify genome directly
                    if has_valid_mode {
                        ui.checkbox(&mut context.genome.modes[selected_idx].child_b.keep_adhesion, "Keep Adhesion");
                    } else {
                        ui.add_enabled(false, egui::Checkbox::new(&mut false, "Keep Adhesion"));
                    }

                    // Mode dropdown for Child B
                    ui.label("Mode:");
                    if has_valid_mode {
                        let current_mode = context.genome.modes[selected_idx].child_b.mode_number;
                        let mode_count = context.genome.modes.len();
                        let selected_text = if current_mode >= 0 && (current_mode as usize) < mode_count {
                            context.genome.modes[current_mode as usize].name.clone()
                        } else {
                            "Invalid".to_string()
                        };
                        // Collect mode info to avoid borrow issues
                        let mode_info: Vec<_> = context.genome.modes.iter()
                            .map(|m| (m.name.clone(), m.color))
                            .collect();
                        let mut new_mode: Option<i32> = None;
                        egui::ComboBox::from_id_salt("qball2_mode")
                            .selected_text(selected_text)
                            .width(ball_container_width - 8.0)
                            .show_ui(ui, |ui| {
                                let item_width = ui.available_width();
                                for (i, (name, color)) in mode_info.iter().enumerate() {
                                    let is_selected = current_mode == i as i32;
                                    let bg_color = egui::Color32::from_rgb(
                                        (color.x * 255.0) as u8,
                                        (color.y * 255.0) as u8,
                                        (color.z * 255.0) as u8,
                                    );
                                    // Calculate luminance for text contrast
                                    let luminance = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
                                    let text_color = if luminance > 0.5 {
                                        egui::Color32::BLACK
                                    } else {
                                        egui::Color32::WHITE
                                    };
                                    
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(item_width, 18.0),
                                        egui::Sense::click(),
                                    );
                                    
                                    // Draw background
                                    let bg = if response.hovered() {
                                        bg_color.gamma_multiply(1.2)
                                    } else {
                                        bg_color
                                    };
                                    ui.painter().rect_filled(rect, 2.0, bg);
                                    
                                    // Draw selection indicator
                                    if is_selected {
                                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                    }
                                    
                                    // Draw text centered
                                    ui.painter().text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        name,
                                        egui::FontId::default(),
                                        text_color,
                                    );
                                    
                                    if response.clicked() {
                                        new_mode = Some(i as i32);
                                    }
                                }
                            });
                        if let Some(mode_num) = new_mode {
                            context.genome.modes[selected_idx].child_b.mode_number = mode_num;
                        }
                    } else {
                        egui::ComboBox::from_id_salt("qball2_mode")
                            .selected_text("--")
                            .width(ball_container_width - 8.0)
                            .show_ui(ui, |_ui| {});
                    }
                }
            );
        });
    });
}

/// Render the TimeSlider panel (placeholder).
fn render_time_slider(ui: &mut Ui, context: &mut PanelContext) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Show status text with fixed height to prevent layout shifting
            let is_preview_mode = context.current_mode == SimulationMode::Preview;

            // Always allocate space for status text to prevent shifting
            if !is_preview_mode {
                ui.colored_label(
                    egui::Color32::GRAY,
                    "Time scrubbing only available in Preview mode"
                );
            } else {
                // Reserve space even when no status message to prevent layout shift
                ui.label("");
            }

            ui.horizontal(|ui| {
                ui.label("Time:");

                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;

                // Track dragging state
                // Slider range is 0 to max_preview_duration (in seconds)
                let slider_response = ui.add_enabled(
                    is_preview_mode,
                    egui::Slider::new(&mut context.editor_state.time_value, 0.0..=context.editor_state.max_preview_duration)
                        .show_value(false)
                );
                context.editor_state.time_slider_dragging = slider_response.dragged();

                // Show actual time value in drag value widget
                ui.add_enabled(
                    is_preview_mode,
                    egui::DragValue::new(&mut context.editor_state.time_value)
                        .speed(0.1)
                        .range(0.0..=context.editor_state.max_preview_duration)
                        .suffix("s")
                );
            });
            
            // Show cell count and resimulation status
            if is_preview_mode {
                ui.horizontal(|ui| {
                    ui.label(format!("Cells: {}", context.cell_count()));
                });
            }
        });
}

/// Render the gizmo settings panel.
fn render_gizmo_settings(ui: &mut Ui, context: &mut PanelContext) {
    ui.checkbox(&mut context.editor_state.gizmo_visible, "Show Orientation Lines");
    ui.checkbox(&mut context.editor_state.split_rings_visible, "Show Split Rings");
}

/// Render the Cell Type Visuals panel for appearance settings.
fn render_cell_type_visuals(ui: &mut Ui, context: &mut PanelContext) {
    use crate::cell::types::CellType;
    
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            // Ensure we have visuals for all cell types
            let cell_types = CellType::all();
            while context.editor_state.cell_type_visuals.len() < cell_types.len() {
                context.editor_state.cell_type_visuals.push(crate::cell::types::CellTypeVisuals::default());
            }

            // Cell type selector dropdown
            ui.horizontal(|ui| {
                ui.label("Cell Type:");
                let selected = context.editor_state.selected_cell_type;
                let selected_name = cell_types.get(selected)
                    .map(|t| t.name())
                    .unwrap_or("Unknown");
                egui::ComboBox::from_id_salt("cell_type_visuals_selector")
                    .selected_text(selected_name)
                    .show_ui(ui, |ui| {
                        for (i, cell_type) in cell_types.iter().enumerate() {
                            ui.selectable_value(
                                &mut context.editor_state.selected_cell_type,
                                i,
                                cell_type.name()
                            );
                        }
                    });
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            let selected_idx = context.editor_state.selected_cell_type;
            if selected_idx >= context.editor_state.cell_type_visuals.len() {
                ui.label("Invalid cell type");
                return;
            }

            let visuals = &mut context.editor_state.cell_type_visuals[selected_idx];

            // Lighting section
            ui.label(egui::RichText::new("Lighting").strong());
            ui.add_space(4.0);

            // Specular Strength
            ui.label("Specular Strength:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.specular_strength, 0.0..=1.0).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.specular_strength).speed(0.01).range(0.0..=1.0));
            });

            // Specular Power (shininess)
            ui.label("Specular Sharpness:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.specular_power, 1.0..=128.0).logarithmic(true).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.specular_power).speed(0.5).range(1.0..=128.0));
            });

            // Fresnel (rim lighting)
            ui.label("Rim Lighting:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.fresnel_strength, 0.0..=1.0).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.fresnel_strength).speed(0.01).range(0.0..=1.0));
            });

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(4.0);

            // Membrane texture section
            ui.label(egui::RichText::new("Membrane Texture").strong());
            ui.add_space(4.0);

            // Membrane noise scale
            ui.label("Noise Scale:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.membrane_noise_scale, 0.0..=32.0).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.membrane_noise_scale).speed(0.1).range(0.0..=32.0));
            });

            // Membrane noise strength
            ui.label("Noise Strength:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.membrane_noise_strength, 0.0..=2.0).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.membrane_noise_strength).speed(0.01).range(0.0..=2.0));
            });

            // Membrane noise animation speed
            ui.label("Animation Speed:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.membrane_noise_speed, 0.0..=2.0).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.membrane_noise_speed).speed(0.01).range(0.0..=2.0));
            });

            ui.add_space(12.0);

            // Preset buttons
            ui.separator();
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Presets").strong());
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("Matte").clicked() {
                    visuals.specular_strength = 0.0;
                    visuals.specular_power = 32.0;
                    visuals.fresnel_strength = 0.0;
                    visuals.membrane_noise_scale = 0.0;
                    visuals.membrane_noise_strength = 0.0;
                    visuals.membrane_noise_speed = 0.0;
                }
                if ui.small_button("Glossy").clicked() {
                    visuals.specular_strength = 0.5;
                    visuals.specular_power = 64.0;
                    visuals.fresnel_strength = 0.3;
                    visuals.membrane_noise_scale = 6.0;
                    visuals.membrane_noise_strength = 0.1;
                    visuals.membrane_noise_speed = 0.0;
                }
                if ui.small_button("Shiny").clicked() {
                    visuals.specular_strength = 0.8;
                    visuals.specular_power = 128.0;
                    visuals.fresnel_strength = 0.5;
                    visuals.membrane_noise_scale = 4.0;
                    visuals.membrane_noise_strength = 0.05;
                    visuals.membrane_noise_speed = 0.0;
                }
                if ui.small_button("Organic").clicked() {
                    visuals.specular_strength = 0.3;
                    visuals.specular_power = 32.0;
                    visuals.fresnel_strength = 0.2;
                    visuals.membrane_noise_scale = 8.0;
                    visuals.membrane_noise_strength = 0.15;
                    visuals.membrane_noise_speed = 0.3;
                }
                if ui.small_button("Bumpy").clicked() {
                    visuals.specular_strength = 0.4;
                    visuals.specular_power = 48.0;
                    visuals.fresnel_strength = 0.25;
                    visuals.membrane_noise_scale = 12.0;
                    visuals.membrane_noise_strength = 0.25;
                    visuals.membrane_noise_speed = 0.1;
                }
            });
        });
}

/// Render the Mode Graph panel for visualizing mode connections.
fn render_mode_graph(ui: &mut Ui, context: &mut PanelContext) {
    use crate::genome::node_graph::{ModeGraphViewer, ModeNode};
    use egui_snarl::ui::{SnarlStyle, SnarlWidget};

    // Collect mode data for the viewer
    let mode_names: Vec<String> = context.genome.modes.iter().map(|m| m.name.clone()).collect();
    let mode_colors: Vec<glam::Vec3> = context.genome.modes.iter().map(|m| m.color).collect();
    
    // Collect child mode numbers (we need mutable access)
    let mut child_a_modes: Vec<i32> = context.genome.modes.iter().map(|m| m.child_a.mode_number).collect();
    let mut child_b_modes: Vec<i32> = context.genome.modes.iter().map(|m| m.child_b.mode_number).collect();

    // Helper for contrasting text
    fn contrasting_text(bg: glam::Vec3) -> egui::Color32 {
        let luminance = 0.299 * bg.x + 0.587 * bg.y + 0.114 * bg.z;
        if luminance > 0.5 {
            egui::Color32::BLACK
        } else {
            egui::Color32::WHITE
        }
    }

    // Top toolbar with add mode dropdown
    ui.horizontal(|ui| {
        ui.label("Add:");
        let selected = &mut context.editor_state.mode_graph_state.selected_add_mode;
        
        // Get selected mode color for the combo box display
        let selected_color = mode_colors.get(*selected).copied().unwrap_or(glam::Vec3::ONE);
        let bg_color = egui::Color32::from_rgb(
            (selected_color.x * 255.0) as u8,
            (selected_color.y * 255.0) as u8,
            (selected_color.z * 255.0) as u8,
        );
        let text_color = contrasting_text(selected_color);
        
        egui::ComboBox::from_id_salt("add_mode_combo")
            .selected_text(egui::RichText::new(format!("M{}", *selected + 1)).color(text_color).background_color(bg_color))
            .width(60.0)
            .show_ui(ui, |ui| {
                for i in 0..mode_names.len() {
                    let color = mode_colors.get(i).copied().unwrap_or(glam::Vec3::ONE);
                    let item_bg = egui::Color32::from_rgb(
                        (color.x * 255.0) as u8,
                        (color.y * 255.0) as u8,
                        (color.z * 255.0) as u8,
                    );
                    let item_text = contrasting_text(color);
                    
                    ui.selectable_value(
                        selected, 
                        i, 
                        egui::RichText::new(format!("M{}", i + 1)).color(item_text).background_color(item_bg)
                    );
                }
            });
        
        if ui.button("Add Node").clicked() {
            let idx = *selected;
            let name = mode_names.get(idx).cloned().unwrap_or_default();
            let color = mode_colors.get(idx).copied().unwrap_or(glam::Vec3::ONE);
            // Add node at next position in column
            let pos = context.editor_state.mode_graph_state.next_node_position();
            context.editor_state.mode_graph_state.snarl.insert_node(pos, ModeNode::new(idx, name, color));
        }
        
        ui.separator();
        ui.label("Scroll to zoom, drag to pan");
    });

    ui.separator();

    // Create the viewer with references to mode data
    let mut viewer = ModeGraphViewer {
        mode_names: &mode_names,
        mode_colors: &mode_colors,
        child_a_modes: &mut child_a_modes,
        child_b_modes: &mut child_b_modes,
        mode_settings: &context.genome.modes,
    };

    // Configure style with zoom limits, grid background, and better pin placement
    use egui_snarl::ui::{BackgroundPattern, PinPlacement};
    let style = SnarlStyle {
        collapsible: Some(false),
        min_scale: Some(0.25),
        max_scale: Some(4.0),
        pin_placement: Some(PinPlacement::Edge),
        header_drag_space: Some(egui::vec2(8.0, 0.0)),
        bg_pattern: Some(BackgroundPattern::grid(egui::vec2(50.0, 50.0), 0.0)),
        bg_pattern_stroke: Some(egui::Stroke::new(1.0, egui::Color32::from_gray(60))),
        ..Default::default()
    };

    // Render the snarl graph
    let snarl = &mut context.editor_state.mode_graph_state.snarl;
    
    SnarlWidget::new()
        .style(style)
        .show(snarl, &mut viewer, ui);

    // Write back any changes to child modes
    for (i, mode) in context.genome.modes.iter_mut().enumerate() {
        if i < child_a_modes.len() {
            mode.child_a.mode_number = child_a_modes[i];
        }
        if i < child_b_modes.len() {
            mode.child_b.mode_number = child_b_modes[i];
        }
    }
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
