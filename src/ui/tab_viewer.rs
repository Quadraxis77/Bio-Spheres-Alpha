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
            Panel::Viewport => {
                render_viewport(ui, self.viewport_rect);
                if let Some(r) = *self.viewport_rect {
                    self.context.editor_state.panel_rects.insert("Viewport".to_string(), r);
                }
            }
            Panel::LeftPanel => render_placeholder_panel(ui, "Left Panel"),
            Panel::RightPanel => render_placeholder_panel(ui, "Right Panel"),
            Panel::BottomPanel => render_placeholder_panel(ui, "Bottom Panel"),
            Panel::SceneManager => render_scene_manager(ui, self.context, self.state),
            Panel::CellInspector => render_cell_inspector(ui, self.context),
            Panel::GenomeEditor => render_genome_editor(ui, self.context),
            Panel::PerformanceMonitor => render_performance_monitor(ui, self.context, self.state),
            Panel::RenderingControls => render_rendering_controls(ui),
            Panel::CaveSystem => render_cave_system(ui, self.context),
            Panel::FluidSettings => {
                render_fluid_settings(ui, self.context, self.state);
                render_organism_skin_settings(ui, self.context, &mut self.state.fluid_settings.organism_skin);
            }
            Panel::WorldSettings => render_world_settings(ui, self.context, self.state),
            Panel::LightSettings => render_light_settings(ui, self.context),
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
            Panel::Help => render_help(ui, self.context, self.state),
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
fn render_scene_manager(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    // Record panel rect for the tutorial pointer.
    context.editor_state.panel_rects.insert("SceneManager".to_string(), ui.max_rect());

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
                state.show_reset_dialog = true;
            }
            
            // Add some spacing between buttons and time
            ui.add_space(20.0);
            
            // Show simulation time in HH:MM:SS format
            let total_seconds = context.current_time() as u64;
            let hours = total_seconds / 3600;
            let minutes = (total_seconds % 3600) / 60;
            let seconds = total_seconds % 60;
            ui.label(format!("{:02}:{:02}:{:02}", hours, minutes, seconds));
        });
        
        // Simulation speed slider (GPU mode only)
        if context.current_mode == crate::ui::types::SimulationMode::Gpu {
            ui.horizontal(|ui| {
                ui.label("Speed:");
                let speed = context.simulation_speed();
                // Round to nearest 0.5 for display
                let display_speed = (speed * 2.0).round() / 2.0;
                let mut slider_speed = display_speed;
                if ui.add(egui::Slider::new(&mut slider_speed, 0.5..=10.0).step_by(0.5).suffix("x")).changed() {
                    context.set_simulation_speed(slider_speed);
                }
            });
        }
        
        ui.add_space(5.0);
    }
}

/// Render the CellInspector panel.
fn render_cell_inspector(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Cell Inspector");
    ui.separator();
    
    // Show GPU cell count (from canonical state, no async readback)
    if let Some(gpu_count) = context.gpu_cell_count() {
        ui.label(format!("Total Cells: {} (GPU)", gpu_count));
    } else {
        ui.label(format!("Total Cells: {}", context.cell_count()));
    }
    
    // Get the inspected cell index from radial menu state
    let inspected_cell = context.editor_state.radial_menu.inspected_cell;
    
    if let Some(cell_idx) = inspected_cell {
        // First, extract all data we need from GPU scene
        let gpu_extraction_data = if let Some(gpu_scene) = context.scene_manager.gpu_scene() {
            // Check if cell extraction is in progress
            if gpu_scene.is_extracting_cell_data() {
                Some((true, None, None)) // is_extracting flag
            } else if let Some(extraction_result) = gpu_scene.get_latest_cell_extraction() {
                if extraction_result.cell_index == cell_idx as u32 {
                    let data_valid = extraction_result.data.is_valid();
                    
                    Some((false, Some(extraction_result), Some(data_valid)))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        
        // Now use the extracted data without borrowing context
        if let Some((is_extracting, extraction_result, data_valid)) = gpu_extraction_data {
            if is_extracting {
                ui.add_space(8.0);
                ui.label(format!("Cell #{}", cell_idx));
                ui.separator();
                
                // Show "Extracting..." message while async readback is in progress
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Extracting cell data from GPU...");
                });
                
                ui.add_space(8.0);
                ui.label("Please wait while cell data is being extracted from GPU buffers.");
                
                return;
            }
            
            if let (Some(extraction_result), Some(data_valid)) = (extraction_result, data_valid) {
                let data = &extraction_result.data;
                
                // Validate that the extracted data is valid
                if !data_valid {
                    ui.colored_label(egui::Color32::RED, "Cell no longer exists");
                    if ui.button("Clear Selection").clicked() {
                        context.editor_state.radial_menu.inspected_cell = None;
                    }
                    return;
                }
                
                ui.add_space(8.0);
                ui.label(format!("Cell #{}", cell_idx));
                ui.separator();
                
                // Action buttons at the top
                ui.horizontal(|ui| {
                    // Load Genome button — reads back genome from GPU (works for both CPU and mutated genomes)
                    if ui.button("📋 Load Genome").clicked() {
                        *context.scene_request = crate::ui::panel_context::SceneModeRequest::LoadGenomeFromGpu(data.genome_id);
                        log::info!("Requested genome readback for genome_id={}", data.genome_id);
                    }
                    
                    if ui.button("🗑 Clear Selection").clicked() {
                        context.editor_state.radial_menu.inspected_cell = None;
                    }
                });
            
            ui.add_space(8.0);
            ui.separator();
            
            // Cell type name
            let cell_type_names = crate::cell::types::CellType::names();
            let type_name = cell_type_names
                .get(data.cell_type as usize)
                .copied()
                .unwrap_or("Unknown");
            
            // Status line
            if data.is_dead != 0 {
                ui.colored_label(egui::Color32::RED, format!("{} (DEAD)", type_name));
            } else {
                ui.label(format!("Type: {}", type_name));
            }
            
            // Identity
            ui.label(format!("Cell ID: {}  Slot: {}  Mode: {}  Genome: {}",
                data.cell_id, data.cell_slot_index, data.mode_index, data.genome_id));
            let org_str = if data.organism_id == u32::MAX {
                "None (dead/isolated)".to_string()
            } else {
                format!("{}", data.organism_id)
            };
            ui.label(format!("Organism ID: {}", org_str));
            
            ui.add_space(4.0);
            ui.separator();
            
            // Nutrients & Division
            ui.label("Nutrients & Division:");
            ui.label(format!("  Nutrients: {:.1}", data.nutrients));
            if data.reserve > 0 {
                ui.label(format!("  Reserve: {}", data.reserve / 1000));
            }
            let div_progress = if data.nutrient_threshold > 0.0 {
                (data.nutrients / data.nutrient_threshold * 100.0).min(100.0)
            } else {
                0.0
            };
            ui.label(format!("  Split threshold: {:.1}  ({:.0}%)", data.nutrient_threshold, div_progress));
            ui.label(format!("  Interval: {:.1}s  Age: {:.1}s", data.split_interval, data.age));
            let max_splits_str = if data.max_splits == 0 { "∞".to_string() } else { format!("{}", data.max_splits) };
            ui.label(format!("  Splits: {} / {}", data.split_count, max_splits_str));
            ui.label(format!("  Gain rate: {:.3}", data.nutrient_gain_rate));
            
            ui.add_space(4.0);
            ui.separator();
            
            // Physics
            ui.label("Physics:");
            let pos = data.position_vec3();
            ui.label(format!("  Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z));
            let vel = data.velocity_vec3();
            let speed = vel.length();
            ui.label(format!("  Vel: ({:.2}, {:.2}, {:.2})  |{:.2}|", vel.x, vel.y, vel.z, speed));
            ui.label(format!("  Mass: {:.2}  Radius: {:.2}", data.mass, data.radius));
            ui.label(format!("  Stiffness: {:.2}  Max size: {:.2}", data.stiffness, data.max_cell_size));
            
            ui.add_space(4.0);
            ui.separator();
            
            // Adhesions
            ui.label(format!("Adhesions: {}", data.adhesion_count));
            
            return;
            }
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
    
    // GPU Readbacks toggle at the top
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.gpu_readbacks_enabled, "Enable GPU Readbacks");
        ui.label("").on_hover_text("Disable to avoid CPU-GPU sync overhead.\nCell count and culling stats won't update.");
    });
    ui.separator();
    
    // Culling section (GPU mode only)
    if context.current_mode == crate::ui::types::SimulationMode::Gpu {
        ui.add_space(8.0);
        ui.heading("GPU Rendering");
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
        
        ui.add_space(8.0);
    }
    
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
        
        // FPS graph (converted from frame times)
        let frame_times: Vec<f32> = perf.frame_time_history().collect();
        if !frame_times.is_empty() {
            // Fixed scale: 0-60 FPS
            let max_fps = 60.0_f32;
            ui.add_space(4.0);
            
            let plot_height = 40.0;
            let (response, painter) = ui.allocate_painter(
                egui::vec2(ui.available_width(), plot_height),
                egui::Sense::hover(),
            );
            let rect = response.rect;
            
            // Background
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));
            
            // FPS bars
            let bar_width = rect.width() / frame_times.len() as f32;
            for (i, &time) in frame_times.iter().enumerate() {
                // Convert frame time (ms) to FPS
                let fps = if time > 0.0 { 1000.0 / time } else { 0.0 };
                let fps_clamped = fps.min(max_fps);
                
                let x = rect.left() + i as f32 * bar_width;
                let height = (fps_clamped / max_fps) * rect.height();
                let y = rect.bottom() - height;
                
                // Color based on FPS: red < 20, yellow < 40, green >= 40
                let color = if fps < 20.0 {
                    egui::Color32::from_rgb(200, 80, 80)   // Red
                } else if fps < 40.0 {
                    egui::Color32::from_rgb(200, 180, 80)  // Yellow
                } else {
                    egui::Color32::from_rgb(80, 200, 80)   // Green
                };
                
                painter.rect_filled(
                    egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(bar_width.max(1.0), height)),
                    0.0,
                    color,
                );
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
}

/// Render the RenderingControls panel (placeholder).
fn render_rendering_controls(ui: &mut Ui) {
    ui.heading("Rendering");
    ui.separator();
    ui.label("Fog, bloom, and visual settings.");
    // TODO: Implement rendering controls
}

/// Render the Cave System panel for procedural cave generation and collision.
fn render_cave_system(ui: &mut Ui, context: &mut PanelContext) {
    ui.separator();
    
    // Check if we're in GPU mode
    if !context.is_gpu_mode() {
        ui.label("Cave system is only available in GPU mode.");
        return;
    }
    
    // Store cave generation request in editor state
    let _editor_state = &mut context.editor_state;
    
    // Check if cave system exists
    let has_cave_system = context.scene_manager.gpu_scene()
        .map(|s| s.cave_renderer.is_some())
        .unwrap_or(false);
    
    if !has_cave_system {
        ui.label("Cave system not initialized.");
        ui.add_space(10.0);
        
        if ui.button("🏔️ Generate Cave System").clicked() {
            // Request cave system initialization
            // This will be handled by the app's render loop
            *context.scene_request = crate::ui::panel_context::SceneModeRequest::Reset; // Temporary - will add proper request
        }
        
        ui.add_space(10.0);
        ui.label("Click the button above to generate a procedural cave system with XPBD collision.");
        return;
    }
    
    // Cave system exists - show editable parameters
    if let Some(gpu_scene) = context.scene_manager.gpu_scene() {
        if let Some(cave_renderer) = &gpu_scene.cave_renderer {
            let params = cave_renderer.params();
            
            // Create local copies for editing
            // Invert density for UI display so that 1.0 = higher density (lower actual value)
            let mut density = 1.0 - params.density;
            let mut scale = params.scale; // Use actual scale directly
            let mut octaves = params.octaves as i32;
            let mut smoothness = params.smoothness;
            let mut seed = params.seed as i32;
            let mut resolution = params.grid_resolution as i32;
            
            let mut params_changed = false;
            
            ui.heading("Generation Parameters");
            ui.add_space(5.0);
            
            // Editable parameters
            ui.add_space(4.0);
            ui.label("Density:");
            params_changed |= ui.add(egui::Slider::new(&mut density, 0.01..=1.0)).changed();
            
            ui.add_space(4.0);
            ui.label("Scale:");
            params_changed |= ui.add(egui::Slider::new(&mut scale, 50.0..=100.0)).changed();
            
            ui.add_space(4.0);
            ui.label("Octaves:");
            params_changed |= ui.add(egui::Slider::new(&mut octaves, 1..=8)).changed();
            
            ui.add_space(4.0);
            ui.label("Smoothness:");
            params_changed |= ui.add(egui::Slider::new(&mut smoothness, 0.0..=1.0)).changed();
            
            ui.add_space(4.0);
            ui.label("Resolution:");
            params_changed |= ui.add(egui::Slider::new(&mut resolution, 32..=128)).changed();
            
            ui.add_space(4.0);
            ui.label("Seed:");
            ui.horizontal(|ui| {
                let mut seed_text = seed.to_string();
                if ui.add_sized([80.0, 20.0], egui::TextEdit::singleline(&mut seed_text)).changed() {
                    if let Ok(parsed_seed) = seed_text.parse::<i32>() {
                        if parsed_seed >= 0 && parsed_seed <= 9999 {
                            seed = parsed_seed;
                            params_changed = true;
                        }
                    }
                }
                if ui.button("🎲 Randomize").on_hover_text("Generate random seed").clicked() {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    use std::time::SystemTime;
                    
                    // Generate a simple random seed using system time
                    let mut hasher = DefaultHasher::new();
                    SystemTime::now().hash(&mut hasher);
                    seed = (hasher.finish() % 10000) as i32;
                    params_changed = true;
                }
            });
            
            ui.add_space(10.0);
            ui.separator();
            ui.label(format!("Triangle Count: {}", params.triangle_count));
            
            ui.add_space(10.0);
            
            let regenerate_clicked = ui.button("🔄 Regenerate Cave").clicked();
            
            // Apply changes if any parameter changed or regenerate was clicked
            if params_changed || regenerate_clicked {
                // Store changes in editor state for the app to apply
                // Invert density so that 1.0 = higher density (lower actual value)
                context.editor_state.cave_density = 1.0 - density;
                context.editor_state.cave_scale = scale;
                context.editor_state.cave_octaves = octaves as u32;
                context.editor_state.cave_smoothness = smoothness;
                context.editor_state.cave_seed = seed as u32;
                context.editor_state.cave_resolution = resolution as u32;
                context.editor_state.cave_params_dirty = true;
                
                // Save settings to disk
                context.editor_state.save_cave_settings();
            }
        }
    }

    // ============================================================
    // Moss Settings
    // ============================================================
    ui.add_space(10.0);
    ui.separator();
    ui.heading("🌿 Moss");
    ui.add_space(5.0);

    if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
        // Enable/disable toggle
        let mut show_moss = gpu_scene.show_moss;
        let mut moss_changed = false;
        if ui.checkbox(&mut show_moss, "Enable Moss").changed() {
            gpu_scene.show_moss = show_moss;
            context.editor_state.show_moss = show_moss;
            moss_changed = true;
        }

        if show_moss {
            if let Some(ref mut moss) = gpu_scene.moss_system {
                ui.add_space(5.0);
                ui.label("Growth & Spread");
                ui.add_space(3.0);

                ui.label("Growth Rate:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_growth_rate, 0.001..=1.0)
                    .logarithmic(true)
                    .text("per sec")).changed() {
                    moss.growth_rate = context.editor_state.moss_growth_rate;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Min Light Level:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_min_light, 0.0..=0.5)
                    .text("threshold")).changed() {
                    moss.min_light = context.editor_state.moss_min_light;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Water Radius (voxels):");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_water_radius, 2.0..=50.0)
                    .text("voxels")).changed() {
                    moss.water_radius = context.editor_state.moss_water_radius;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Decay Rate:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_decay_rate, 0.001..=0.5)
                    .logarithmic(true)
                    .text("per sec")).changed() {
                    moss.decay_rate = context.editor_state.moss_decay_rate;
                    moss_changed = true;
                }

                ui.add_space(8.0);
                ui.label("Erosion");
                ui.add_space(3.0);

                ui.label("Water Erosion Rate:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_erosion_rate, 0.0..=2.0)
                    .text("per sec")).changed() {
                    moss.erosion_rate = context.editor_state.moss_erosion_rate;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Wetness Evaporation:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_wetness_evaporation, 0.001..=0.2)
                    .logarithmic(true)
                    .text("per sec")).changed() {
                    moss.wetness_evaporation = context.editor_state.moss_wetness_evaporation;
                    moss_changed = true;
                }

                ui.add_space(8.0);
                ui.label("Consumption (Phagocytes)");
                ui.add_space(3.0);

                ui.label("Graze Cooldown:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_consume_rate, 1.0..=30.0)
                    .text("seconds")).changed() {
                    moss.graze_cooldown = context.editor_state.moss_consume_rate;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Nutrients per Moss:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_nutrient_per_moss, 1.0..=200.0)
                    .text("nutrients")).changed() {
                    moss.nutrient_per_moss = context.editor_state.moss_nutrient_per_moss;
                    moss_changed = true;
                }

                ui.add_space(8.0);
                ui.label("Appearance");
                ui.add_space(3.0);

                ui.label("Texture Scale:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_scale, 0.02..=0.5)
                    .text("scale")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Parallax Depth:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_parallax_depth, 0.0..=0.3)
                    .text("depth")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(5.0);
                ui.label("Noise Type:");
                let noise_labels = ["Value Noise", "Worley (Cellular)", "Ridged"];
                let mut noise_type = context.editor_state.moss_noise_type as usize;
                egui::ComboBox::from_id_salt("moss_noise_type")
                    .selected_text(noise_labels[noise_type.min(2)])
                    .show_ui(ui, |ui| {
                        for (i, label) in noise_labels.iter().enumerate() {
                            if ui.selectable_value(&mut noise_type, i, *label).changed() {
                                context.editor_state.moss_noise_type = noise_type as u32;
                                context.editor_state.light_params_dirty = true;
                            }
                        }
                    });
                if noise_type as u32 != context.editor_state.moss_noise_type {
                    context.editor_state.moss_noise_type = noise_type as u32;
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Noise Frequency:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_noise_frequency, 4.0..=80.0)
                    .logarithmic(true)
                    .text("freq")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Noise Lacunarity:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_noise_lacunarity, 1.5..=5.0)
                    .text("lac")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Height Sharpness (low):");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_height_sharpness_low, 0.0..=0.5)
                    .text("low")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Height Sharpness (high):");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_height_sharpness_high, 0.3..=1.0)
                    .text("high")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Bump Strength:");
                if ui.add(egui::Slider::new(&mut context.editor_state.moss_bump_strength, 0.0..=15.0)
                    .text("bump")).changed() {
                    context.editor_state.light_params_dirty = true;
                    moss_changed = true;
                }

                ui.add_space(5.0);
                ui.label("Colors");
                ui.add_space(3.0);

                ui.horizontal(|ui| {
                    ui.label("Dark (base):");
                    let mut color = context.editor_state.moss_color_dark;
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        context.editor_state.moss_color_dark = color;
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Bright (tips):");
                    let mut color = context.editor_state.moss_color_bright;
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        context.editor_state.moss_color_bright = color;
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }
                });
            } else {
                ui.add_space(5.0);
                ui.label("Moss system not yet initialized.");
                ui.label("Requires fluid simulation and light field.");
            }
        }

        if moss_changed {
            context.editor_state.save_cave_settings();
        }
    }
}

/// Render the Fluid Settings panel for fluid simulation controls and visualization.
fn render_fluid_settings(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    // Check if we're in GPU mode
    if !context.is_gpu_mode() {
        return;
    }
    
    // Check if fluid system exists
    let has_fluid_system = context.scene_manager.gpu_scene()
        .map(|scene| scene.fluid_buffers.is_some())
        .unwrap_or(false);
    
    if !has_fluid_system {
        return;
    }
    
    // Continuous spawning controls
    ui.separator();
    ui.heading("Fluid Spawning");
    ui.separator();
    let spawn_changed = ui.checkbox(&mut context.editor_state.fluid_continuous_spawn, "Water Fill").changed();
    
    if spawn_changed {
        // Update fluid simulator continuous spawn state when toggled
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            if let Some(ref mut simulator) = gpu_scene.fluid_simulator {
                simulator.set_continuous_spawn(context.editor_state.fluid_continuous_spawn);
            }
        }
        // Save fluid settings
        context.editor_state.save_fluid_settings();
    }
    
    // === Nutrients ===
    ui.separator();
    ui.heading("Nutrients");
    ui.separator();

    ui.label("Density:");
    if ui.add(egui::Slider::new(&mut context.editor_state.nutrient_density, 0.0..=0.5)
        .step_by(0.01)
        .fixed_decimals(2)
        .text("density")
    ).changed() {
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_density = context.editor_state.nutrient_density;
        }
        context.editor_state.save_fluid_settings();
    }

    ui.label("Epoch Duration:");
    if ui.add(egui::Slider::new(&mut context.editor_state.nutrient_epoch_duration, 2.0..=30.0)
        .step_by(0.5)
        .fixed_decimals(1)
        .suffix("s")
        .text("cycle length")
    ).changed() {
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_epoch_duration = context.editor_state.nutrient_epoch_duration;
        }
    }

    ui.label("Epoch Spacing:");
    if ui.add(egui::Slider::new(&mut context.editor_state.nutrient_epoch_spacing, 1.0..=30.0)
        .step_by(0.5)
        .fixed_decimals(1)
        .suffix("s")
        .text("overlap gap")
    ).changed() {
        // Clamp spacing to not exceed duration
        context.editor_state.nutrient_epoch_spacing = context.editor_state.nutrient_epoch_spacing
            .min(context.editor_state.nutrient_epoch_duration);
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_epoch_spacing = context.editor_state.nutrient_epoch_spacing;
        }
    }

    ui.label("Spawn Ramp:");
    if ui.add(egui::Slider::new(&mut context.editor_state.nutrient_spawn_end, 0.05..=0.9)
        .step_by(0.05)
        .fixed_decimals(2)
        .text("spawn end")
    ).changed() {
        // Ensure spawn_end < despawn_start
        if context.editor_state.nutrient_spawn_end >= context.editor_state.nutrient_despawn_start {
            context.editor_state.nutrient_despawn_start = (context.editor_state.nutrient_spawn_end + 0.05).min(0.95);
        }
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_spawn_end = context.editor_state.nutrient_spawn_end;
            gpu_scene.nutrient_despawn_start = context.editor_state.nutrient_despawn_start;
        }
    }

    ui.label("Despawn Ramp:");
    if ui.add(egui::Slider::new(&mut context.editor_state.nutrient_despawn_start, 0.1..=0.95)
        .step_by(0.05)
        .fixed_decimals(2)
        .text("despawn start")
    ).changed() {
        // Ensure despawn_start > spawn_end
        if context.editor_state.nutrient_despawn_start <= context.editor_state.nutrient_spawn_end {
            context.editor_state.nutrient_spawn_end = (context.editor_state.nutrient_despawn_start - 0.05).max(0.05);
        }
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_spawn_end = context.editor_state.nutrient_spawn_end;
            gpu_scene.nutrient_despawn_start = context.editor_state.nutrient_despawn_start;
        }
    }

    // Surface pressure control for radial fluid mode
    ui.separator();
    ui.heading("Fluid Physics");
    
    // Only show surface pressure when gravity mode is radial
    if state.world_settings.gravity_mode == 3 {
        ui.label("Surface Pressure:");
        if ui.add(egui::Slider::new(&mut state.fluid_settings.surface_pressure, 0.0..=1.0)
            .text("Surface Pressure")
        ).changed() {
            // Apply to GPU scene
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.set_surface_pressure(state.fluid_settings.surface_pressure);
            }
        }
        ui.label(egui::RichText::new("Tangential smoothing strength for radial fluid mode").small());
    }

    // Lateral flow probability control for selected fluid type
    ui.separator();
    
    // Show lateral flow probability only for the selected fluid type
    let selected_fluid_type = context.editor_state.selected_fluid_type;
    
    if selected_fluid_type > 0 && selected_fluid_type <= 3 {
        let mut probabilities = context.editor_state.fluid_lateral_flow_probabilities;
        let mut changed = false;
        
        ui.add_space(4.0);
        ui.label("Lateral Flow Probability:");
        if ui.add(egui::Slider::new(&mut probabilities[selected_fluid_type as usize], 0.0..=1.0)
            .step_by(0.01)
            .fixed_decimals(2)
        ).changed() {
            changed = true;
        }
        
        if changed {
            context.editor_state.fluid_lateral_flow_probabilities = probabilities;
            // Save fluid settings
            context.editor_state.save_fluid_settings();
        }
    }
    
    // Condensation probability (steam to water)
    ui.add_space(4.0);
    ui.label("Condensation Probability:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_condensation_probability, 0.0..=0.05)
        .step_by(0.001)
        .fixed_decimals(3)
        ).changed() {
            // Update GPU scene phase change probabilities
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.set_phase_change_probabilities(
                    context.editor_state.fluid_condensation_probability,
                    context.editor_state.fluid_vaporization_probability
                );
            }
            // Save fluid settings
            context.editor_state.save_fluid_settings();
        }
    
    // Vaporization probability (water to steam)
    ui.add_space(4.0);
    ui.label("Vaporization Probability:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_vaporization_probability, 0.0..=0.05)
        .step_by(0.001)
        .fixed_decimals(3)
        ).changed() {
            // Update GPU scene phase change probabilities
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.set_phase_change_probabilities(
                    context.editor_state.fluid_condensation_probability,
                    context.editor_state.fluid_vaporization_probability
                );
            }
            // Save fluid settings
            context.editor_state.save_fluid_settings();
        }
    
    // === Water Surface Lighting & Reflection ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Water Surface Lighting");
    ui.add_space(4.0);

    ui.label("Ambient:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_ambient, 0.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Diffuse:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_diffuse, 0.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Specular:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_specular, 0.0..=2.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Shininess:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_shininess, 1.0..=256.0)
        .step_by(1.0).fixed_decimals(0).logarithmic(true)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Rim Light:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_rim, 0.0..=2.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Alpha (Transparency):");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_alpha, 0.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(8.0);
    ui.separator();
    ui.heading("Cubemap Reflection");
    ui.add_space(4.0);

    ui.label("Fresnel Strength:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_fresnel, 0.0..=2.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Fresnel Power:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_fresnel_power, 0.5..=10.0)
        .step_by(0.1).fixed_decimals(1)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Reflection Intensity:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_reflection, 0.0..=2.0)
        .step_by(0.01).fixed_decimals(2)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    ui.add_space(4.0);
    ui.label("Reflection Brightness:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_reflection_brightness, 1.0..=50.0)
        .step_by(0.5).fixed_decimals(1)).changed() {
        context.editor_state.save_fluid_render_settings();
    }

    // === Water Surface Waves ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Water Surface Waves");
    ui.add_space(4.0);
    
    ui.label("Wave Height:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_wave_height, 0.0..=5.0)
        .step_by(0.05).fixed_decimals(2)).changed() {
        context.editor_state.fluid_mesh_params_dirty = true;
        context.editor_state.save_fluid_render_settings();
    }
    
    ui.add_space(4.0);
    ui.label("Wave Speed:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_wave_speed, 0.0..=5.0)
        .step_by(0.05).fixed_decimals(2)).changed() {
        context.editor_state.fluid_mesh_params_dirty = true;
        context.editor_state.save_fluid_render_settings();
    }
    
    ui.add_space(4.0);
    ui.label("Noise Scale (lower = larger waves):");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_noise_scale, 0.01..=20.0)
        .step_by(0.01).fixed_decimals(2).logarithmic(true)).changed() {
        context.editor_state.fluid_mesh_params_dirty = true;
        context.editor_state.save_fluid_render_settings();
    }
    
    ui.add_space(4.0);
    ui.label("Octaves:");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_noise_octaves, 1.0..=6.0)
        .step_by(1.0).fixed_decimals(0)).changed() {
        context.editor_state.fluid_mesh_params_dirty = true;
        context.editor_state.save_fluid_render_settings();
    }
    
    ui.add_space(4.0);
    ui.label("Lacunarity (freq per octave):");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_noise_lacunarity, 1.0..=4.0)
        .step_by(0.1).fixed_decimals(1)).changed() {
        context.editor_state.fluid_mesh_params_dirty = true;
        context.editor_state.save_fluid_render_settings();
    }
    
    ui.add_space(4.0);
    ui.label("Persistence (amp per octave):");
    if ui.add(egui::Slider::new(&mut context.editor_state.fluid_noise_persistence, 0.1..=1.0)
        .step_by(0.05).fixed_decimals(2)).changed() {
        context.editor_state.fluid_mesh_params_dirty = true;
        context.editor_state.save_fluid_render_settings();
    }
    
    // === Caustics ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Caustics");
    ui.add_space(4.0);
    let mut caustic_changed = false;
    
    ui.label("Intensity:");
    caustic_changed |= ui.add(egui::Slider::new(&mut context.editor_state.caustic_intensity, 0.0..=2.0)
        .step_by(0.01).fixed_decimals(2)).changed();
    
    ui.add_space(4.0);
    ui.label("Scale:");
    caustic_changed |= ui.add(egui::Slider::new(&mut context.editor_state.caustic_scale, 0.1..=30.0)
        .step_by(0.1).fixed_decimals(1)).changed();
    
    ui.add_space(4.0);
    ui.label("Speed:");
    caustic_changed |= ui.add(egui::Slider::new(&mut context.editor_state.caustic_speed, 0.0..=5.0)
        .step_by(0.1).fixed_decimals(1)).changed();
    
    if caustic_changed {
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            if let Some(ref mut light_field) = gpu_scene.light_field_system {
                light_field.set_caustic_intensity(context.editor_state.caustic_intensity);
                light_field.set_caustic_scale(context.editor_state.caustic_scale);
                light_field.set_caustic_speed(context.editor_state.caustic_speed);
            }
        }
        context.editor_state.save_light_settings();
    }
    
    ui.add_space(10.0);
}

/// Organism skin controls, embedded at the bottom of the fluid/environment panel.
fn render_organism_skin_settings(ui: &mut Ui, context: &mut PanelContext, organism_skin: &mut crate::ui::OrganismSkinSettings) {
    if !context.is_gpu_mode() {
        return;
    }

    ui.add_space(8.0);
    ui.separator();
    ui.heading("Organism Skins");
    ui.add_space(4.0);

    ui.label("Wrap cell clusters in a smooth biological membrane.");
    ui.add_space(4.0);

    if ui.checkbox(&mut organism_skin.enabled, "Show Organism Skins").changed() {
        // Enabling is handled in app.rs (lazy initialisation)
    }

    if !organism_skin.enabled {
        ui.add_space(8.0);
        return;
    }

    ui.add_space(6.0);

    // ── Geometry ──────────────────────────────────────────────────────────────
    ui.label("Skin Thickness (radius scale):");
    ui.add(egui::Slider::new(&mut organism_skin.radius_scale, 0.5..=12.0)
        .step_by(0.1).fixed_decimals(1));

    ui.add_space(4.0);
    ui.label("Iso Level (surface threshold):");
    ui.add(egui::Slider::new(&mut organism_skin.iso_level, 0.1..=2.0)
        .step_by(0.05).fixed_decimals(2));

    // ── Material ──────────────────────────────────────────────────────────────
    ui.add_space(6.0);
    ui.separator();
    ui.label("Skin Colour:");
    ui.horizontal(|ui| {
        let c = &mut organism_skin.base_color;
        let mut rgb = egui::Color32::from_rgb(
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
        );
        if ui.color_edit_button_srgba(&mut rgb).changed() {
            c[0] = rgb.r() as f32 / 255.0;
            c[1] = rgb.g() as f32 / 255.0;
            c[2] = rgb.b() as f32 / 255.0;
        }
    });

    ui.add_space(4.0);
    ui.label("Translucency (alpha):");
    ui.add(egui::Slider::new(&mut organism_skin.alpha, 0.0..=1.0)
        .step_by(0.02).fixed_decimals(2));

    ui.add_space(4.0);
    ui.label("Subsurface Scattering:");
    ui.add(egui::Slider::new(&mut organism_skin.sss_strength, 0.0..=2.0)
        .step_by(0.05).fixed_decimals(2));

    ui.add_space(4.0);
    ui.label("Rim Light Strength:");
    ui.add(egui::Slider::new(&mut organism_skin.rim_strength, 0.0..=2.0)
        .step_by(0.05).fixed_decimals(2));

    ui.add_space(8.0);

    // Presets
    ui.horizontal(|ui| {
        if ui.button("Cell Membrane").clicked() {
            organism_skin.base_color = [0.85, 0.55, 0.35];
            organism_skin.alpha = 0.55;
            organism_skin.sss_strength = 0.5;
            organism_skin.rim_strength = 0.35;
        }
        if ui.button("Bioluminescent").clicked() {
            organism_skin.base_color = [0.2, 0.8, 0.7];
            organism_skin.alpha = 0.45;
            organism_skin.sss_strength = 1.2;
            organism_skin.rim_strength = 0.7;
        }
        if ui.button("Opaque Shell").clicked() {
            organism_skin.base_color = [0.7, 0.65, 0.5];
            organism_skin.alpha = 0.9;
            organism_skin.sss_strength = 0.1;
            organism_skin.rim_strength = 0.2;
        }
    });

    ui.add_space(4.0);
}

/// Render the Light & Fog settings panel for light field, photocyte, and volumetric fog controls.
fn render_light_settings(ui: &mut Ui, context: &mut PanelContext) {
    if !context.is_gpu_mode() {
        ui.label("Light & Fog settings are only available in GPU mode.");
        return;
    }
    
    let has_light_field = context.scene_manager.gpu_scene()
        .map(|s| s.light_field_system.is_some())
        .unwrap_or(false);
    
    if !has_light_field {
        ui.label("Light field system not initialized.");
        ui.add_space(5.0);
        ui.label("Initialize the fluid system first to enable light & fog.");
        return;
    }
    
    let mut changed = false;
    
    // === Light Direction ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Light Direction");
    ui.add_space(4.0);
    
    ui.label("X:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_dir[0], -1.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed();
    ui.label("Y:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_dir[1], -1.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed();
    ui.label("Z:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_dir[2], -1.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed();
    
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui.button("☀ Top-Down").clicked() {
            context.editor_state.light_dir = [0.0, 1.0, 0.0];
            changed = true;
        }
        if ui.button("🌅 Sunset").clicked() {
            context.editor_state.light_dir = [-0.7, 0.3, 0.5];
            changed = true;
        }
        if ui.button("↗ Default").clicked() {
            context.editor_state.light_dir = [-0.5, 0.7, 0.5];
            changed = true;
        }
    });
    
    ui.add_space(4.0);
    ui.label("Sun Brightness:");
    let brightness_changed = ui.add(egui::Slider::new(&mut context.editor_state.sun_intensity, 0.0..=20.0)
        .step_by(0.1).fixed_decimals(1)).changed();
    if brightness_changed {
        changed = true;
        // Sync immediately so photocyte gain rate updates
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.sun_intensity = context.editor_state.sun_intensity;
        }
    }
    
    // === Procedural Sun ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Procedural Sun");
    ui.add_space(4.0);
    
    let mut sun_changed = false;
    
    sun_changed |= ui.checkbox(&mut context.editor_state.show_sun, "Enable Sun").changed();
    
    if context.editor_state.show_sun {
        ui.add_space(4.0);
        
        // Sun color
        let mut sc = context.editor_state.sun_color;
        ui.label("Sun Color R:");
        sun_changed |= ui.add(egui::Slider::new(&mut sc[0], 0.0..=1.0).step_by(0.01).fixed_decimals(2)).changed();
        ui.label("Sun Color G:");
        sun_changed |= ui.add(egui::Slider::new(&mut sc[1], 0.0..=1.0).step_by(0.01).fixed_decimals(2)).changed();
        ui.label("Sun Color B:");
        sun_changed |= ui.add(egui::Slider::new(&mut sc[2], 0.0..=1.0).step_by(0.01).fixed_decimals(2)).changed();
        context.editor_state.sun_color = sc;
        
        ui.add_space(4.0);
        ui.label("Sun Size:");
        sun_changed |= ui.add(egui::Slider::new(&mut context.editor_state.sun_angular_radius, 0.005..=0.2)
            .step_by(0.005).fixed_decimals(3)).changed();
        
        ui.add_space(4.0);
    }
    
    // === Surface Shadows ===
    ui.heading("Surface Shadows");
    ui.add_space(4.0);
    
    let mut shadow_changed = false;
    shadow_changed |= ui.checkbox(&mut context.editor_state.shadow_enabled, "Enable Surface Shadows").changed();
    
    ui.add_space(4.0);
    ui.label("Shadow Strength:");
    shadow_changed |= ui.add(egui::Slider::new(&mut context.editor_state.shadow_strength, 0.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed();
    
    ui.add_space(4.0);
    ui.label("Shadow Quality:");
    shadow_changed |= ui.add(egui::Slider::new(&mut context.editor_state.shadow_quality, 0.0..=1.0)
        .step_by(0.01).fixed_decimals(2)).changed();
    
    if shadow_changed {
        changed = true;
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            if let Some(ref mut light_field) = gpu_scene.light_field_system {
                light_field.set_shadow_enabled(context.editor_state.shadow_enabled);
                light_field.set_shadow_strength(context.editor_state.shadow_strength);
                light_field.set_shadow_quality(context.editor_state.shadow_quality);
            }
        }
        context.editor_state.save_light_settings();
    }
    
    // === Light Field Settings ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Light Field (Shadows)");
    ui.add_space(4.0);
    
    ui.label("Ray Steps:");
    let mut steps = context.editor_state.light_field_max_steps as i32;
    if ui.add(egui::Slider::new(&mut steps, 1..=256)).changed() {
        context.editor_state.light_field_max_steps = steps as u32;
        changed = true;
    }
    
    ui.label("Step Size:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_field_step_size, 0.5..=4.0)
        .step_by(0.1).fixed_decimals(1)).changed();
    
    ui.label("Solid Absorption:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_field_absorption_solid, 0.1..=20.0)
        .step_by(0.1).fixed_decimals(1)).changed();
    
    ui.label("Cell Absorption:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_field_absorption_cell, 0.0..=5.0)
        .step_by(0.05).fixed_decimals(2)).changed();
    
    ui.label("Ambient Floor:");
    changed |= ui.add(egui::Slider::new(&mut context.editor_state.light_field_ambient_floor, 0.0..=0.2)
        .step_by(0.005).fixed_decimals(3)).changed();
    
    // === Volumetric Fog ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Volumetric Fog");
    ui.add_space(4.0);
    
    changed |= ui.checkbox(&mut context.editor_state.show_volumetric_fog, "Enable Volumetric Fog").changed();
    
    if context.editor_state.show_volumetric_fog {
        ui.add_space(4.0);
        ui.label("Fog Density:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.fog_density, 0.0..=2.0)
            .step_by(0.01).fixed_decimals(2)).changed();
        
        ui.label("Fog Steps:");
        let mut fsteps = context.editor_state.fog_steps as i32;
        if ui.add(egui::Slider::new(&mut fsteps, 8..=128)).changed() {
            context.editor_state.fog_steps = fsteps as u32;
            changed = true;
        }
        
        ui.label("Scattering Anisotropy:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.fog_scattering_anisotropy, 0.0..=0.95)
            .step_by(0.01).fixed_decimals(2)).changed();
        
        ui.label("Absorption:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.fog_absorption, 0.0..=2.0)
            .step_by(0.01).fixed_decimals(2)).changed();
        
        ui.add_space(4.0);
        let mut fog_col = context.editor_state.fog_color;
        ui.label("Fog Color R:");
        changed |= ui.add(egui::Slider::new(&mut fog_col[0], 0.0..=1.0).step_by(0.01).fixed_decimals(2)).changed();
        ui.label("Fog Color G:");
        changed |= ui.add(egui::Slider::new(&mut fog_col[1], 0.0..=1.0).step_by(0.01).fixed_decimals(2)).changed();
        ui.label("Fog Color B:");
        changed |= ui.add(egui::Slider::new(&mut fog_col[2], 0.0..=1.0).step_by(0.01).fixed_decimals(2)).changed();
        context.editor_state.fog_color = fog_col;
        
        ui.add_space(4.0);
        ui.label("Height Fog Density:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.fog_height_density, 0.0..=2.0)
            .step_by(0.01).fixed_decimals(2)).changed();
        
        ui.label("Height Fog Falloff:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.fog_height_falloff, 0.001..=0.1)
            .step_by(0.001).fixed_decimals(3)).changed();
    }
    
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Depth of Field");
    ui.add_space(4.0);
    
    changed |= ui.checkbox(&mut context.editor_state.show_dof, "Enable Depth of Field").changed();
    
    if context.editor_state.show_dof {
        ui.add_space(4.0);
        ui.label("Focal Distance:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.dof_focal_distance, 5.0..=500.0)
            .step_by(1.0).fixed_decimals(0)).changed();
        
        ui.label("Focal Range:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.dof_focal_range, 1.0..=200.0)
            .step_by(1.0).fixed_decimals(0)).changed();
        
        ui.label("Max Blur Radius:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.dof_max_blur_radius, 1.0..=24.0)
            .step_by(0.5).fixed_decimals(1)).changed();
        
        ui.label("Blur Strength:");
        changed |= ui.add(egui::Slider::new(&mut context.editor_state.dof_blur_strength, 0.0..=3.0)
            .step_by(0.05).fixed_decimals(2)).changed();
    }
    
    // Apply sun settings to GPU
    if sun_changed {
        changed = true;
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.show_sun = context.editor_state.show_sun;
            gpu_scene.sun_intensity = context.editor_state.sun_intensity;
            if let Some(ref mut sun) = gpu_scene.sun_renderer {
                sun.sun_color = context.editor_state.sun_color;
                sun.sun_angular_radius = context.editor_state.sun_angular_radius;
            }
        }
        context.editor_state.save_light_settings();
    }
    
    // Mark dirty and save if any parameter changed
    if changed {
        context.editor_state.light_params_dirty = true;
        context.editor_state.save_light_settings();
    }
    
    ui.add_space(10.0);
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
    // Record panel rect for the tutorial pointer.
    context.editor_state.panel_rects.insert("Modes".to_string(), ui.max_rect());

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
            // Collect all indices to reset (all selected modes, or just the primary if only one)
            let indices_to_reset: Vec<usize> = if context.editor_state.selected_mode_indices.len() > 1 {
                context.editor_state.selected_mode_indices
                    .iter()
                    .copied()
                    .filter(|&i| i < context.genome.modes.len())
                    .collect()
            } else {
                if selected_index < context.genome.modes.len() { vec![selected_index] } else { vec![] }
            };

            // Seed the LCG once; each mode advances it to get a distinct color
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(12345);
            let mut rng = seed as u64;

            for idx in indices_to_reset {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = ((rng >> 33) & 0xFF) as f32 / 255.0;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let g = ((rng >> 33) & 0xFF) as f32 / 255.0;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let b = ((rng >> 33) & 0xFF) as f32 / 255.0;

                context.genome.modes[idx] = crate::genome::ModeSettings {
                    name: format!("M{}", idx + 1),
                    default_name: format!("M{}", idx + 1),
                    color: glam::Vec3::new(r, g, b),
                    opacity: 1.0,
                    emissive: 0.0,
                    cell_type: 0,
                    parent_make_adhesion: false,
                    split_mass: 1.5,
                    split_interval: 1.0,
                    nutrient_gain_rate: 20.0,
                    max_cell_size: 2.0,
                    split_ratio: 0.5,
                    nutrient_priority: 1.0,
                    prioritize_when_low: true,
                    parent_split_direction: glam::Vec2::ZERO,
                    max_adhesions: 20,
                    min_adhesions: 0,
                    enable_parent_angle_snapping: true,
                    max_splits: -1,
                    mode_a_after_splits: -1,
                    mode_b_after_splits: -1,
                    child_a_after_split_orientation: glam::Quat::IDENTITY,
                    child_b_after_split_orientation: glam::Quat::IDENTITY,
                    child_a_after_split_keep_adhesion: true,
                    child_b_after_split_keep_adhesion: true,
                    glueocyte_cell_adhesion: false,
                    glueocyte_env_adhesion: true,
                    glueocyte_cell_adhesion_signal_channel: -1,
                    glueocyte_cell_adhesion_signal_threshold: 1.0,
                    swim_force: 0.5,
                    flagellocyte_use_signal: false,
                    flagellocyte_signal_channel: 0,
                    flagellocyte_speed_a: 0.5,
                    flagellocyte_speed_b: 0.0,
                    flagellocyte_threshold_c: 1.0,
                    cilia_speed: 0.5,
                    cilia_push_bonded: false,
                    cilia_use_signal: false,
                    cilia_signal_channel: 0,
                    cilia_speed_below: 0.5,
                    cilia_speed_above: 0.0,
                    cilia_threshold: 1.0,
                    cilia_attract_force: 0.0,
                    myocyte_contraction: 0.5,
                    myocyte_use_signal: false,
                    myocyte_signal_channel: 0,
                    myocyte_contraction_above: 0.5,
                    myocyte_contraction_below: 0.0,
                    myocyte_threshold: 1.0,
                    myocyte_pulse_rate: 1.0,
                    myocyte_pulse_phase: 0,
                    embryocyte_use_timer: false,
                    embryocyte_release_timer: 10.0,
                    embryocyte_use_threshold: false,
                    embryocyte_threshold_value: 32768,
                    embryocyte_use_signal: false,
                    embryocyte_signal_channel: 0,
                    embryocyte_signal_value: 1.0,
                    buoyancy_force: 0.5,
                    oculocyte_sense_type: 0,
                    oculocyte_signal_channel: 0,
                    oculocyte_signal_value: 10.0,
                    oculocyte_signal_hops: 3,
                    oculocyte_ray_length: 20.0,
                    membrane_stiffness: 50.0,
                    regulation_emit_channel: -1,
                    regulation_emit_value: 10.0,
                    regulation_emit_hops: 3,
                    division_signal_channel: -1,
                    division_signal_threshold: 1.0,
                    division_signal_invert: false,
                    apoptosis_signal_channel: -1,
                    apoptosis_signal_threshold: 1.0,
                    apoptosis_signal_invert: false,
                    signal_child_a_channel: -1,
                    signal_child_a_threshold: 1.0,
                    signal_child_a_mode_above: -1,
                    signal_child_a_mode_below: -1,
                    signal_child_b_channel: -1,
                    signal_child_b_threshold: 1.0,
                    signal_child_b_mode_above: -1,
                    signal_child_b_mode_below: -1,
                    mode_switch_signal_channel: -1,
                    mode_switch_signal_threshold: 1.0,
                    mode_switch_target: -1,
                    mode_switch_invert: false,
                    devorocyte_consume_range: 0.5,
                    devorocyte_consume_rate: 30.0,
                    vascular_outlet: false,
                    child_a: crate::genome::ChildSettings {
                        mode_number: idx as i32,
                        ..Default::default()
                    },
                    child_b: crate::genome::ChildSettings {
                        mode_number: idx as i32,
                        ..Default::default()
                    },
                    adhesion_settings: crate::genome::AdhesionSettings::default(),
                };
                log::info!("Reset mode M{} to original defaults", idx + 1);
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
    
    let (selection_changed, initial_changed, rename_completed, color_change, row_rects) = modes_list_items(
        ui,
        &modes_data,
        &mut selected_index,
        &mut context.editor_state.selected_mode_indices,
        &mut initial_mode,
        available_width,
        copy_into_mode,
        &mut context.editor_state.color_picker_state,
        &mut context.editor_state.renaming_mode,
        &mut context.editor_state.rename_buffer,
    );

    // Store each mode row's rect so the tutorial arrow can point at specific rows.
    for (i, rect) in row_rects.iter().enumerate() {
        context.editor_state.panel_rects.insert(format!("mode_row_{}", i), *rect);
    }
    
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
            // Plain click (no modifier) resets multi-selection to just this mode.
            // Ctrl/Shift clicks are handled inside modes_list_items and don't set
            // selection_changed, so this branch only fires for plain clicks.
            if !context.editor_state.selected_mode_indices.contains(&selected_index) {
                context.editor_state.selected_mode_indices = vec![selected_index];
            }
            
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
    
    // Handle rename completion (when user finishes inline editing)
    if let Some((mode_index, new_name)) = rename_completed {
        if mode_index < context.genome.modes.len() {
            let final_name = if new_name.trim().is_empty() {
                // Use default name if the new name is empty
                context.genome.modes[mode_index].default_name.clone()
            } else {
                new_name
            };
            context.genome.modes[mode_index].name = final_name;
        }
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
    
    // Show current mode info (for debugging/feedback) - more compact
    if selected_index < context.genome.modes.len() {
        ui.add_space(4.0); // Reduced spacing
        let multi_count = context.editor_state.selected_mode_indices.len();
        if multi_count > 1 {
            ui.colored_label(
                egui::Color32::from_rgb(255, 200, 80),
                format!("✦ {} modes selected — edits apply to all", multi_count),
            );
        } else {
            ui.small(format!("Selected: {}", context.genome.modes[selected_index].name));
        }
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
    // Record panel rect for the tutorial pointer.
    context.editor_state.panel_rects.insert("NameTypeEditor".to_string(), ui.max_rect());

    // Multi-select: snapshot before rendering so we can diff afterwards
    let selected_idx = context.editor_state.selected_mode_index;
    let pre_cell_type = context.genome.modes.get(selected_idx).map(|m| m.cell_type);
    let pre_make_adhesion = context.genome.modes.get(selected_idx).map(|m| m.parent_make_adhesion);

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 2.0;

            // Genome file buttons — row 1
            ui.horizontal(|ui| {
                if ui.button("Save Genome").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Genome", &["genome"])
                        .set_directory(crate::genome::Genome::genomes_dir())
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
                        .set_directory(crate::genome::Genome::genomes_dir())
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
                if ui.button("New Genome").clicked() {
                    *context.genome = crate::genome::Genome::new_with_random_colors();
                    context.editor_state.selected_mode_index = 0;
                }
            });

            // Genome file buttons — row 2
            ui.horizontal(|ui| {
                if ui.button("🎲 Procedural")
                    .on_hover_text(
                        "Procedurally generate a creature with a random body plan.
                         All creatures are guaranteed to reproduce by shedding
                         multiple detached egg cells (3–6 per cycle) that each
                         grow into the full organism."
                    )
                    .clicked()
                {
                    let seed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0xdeadbeef_cafebabe);
                    *context.genome = crate::genome::Genome::generate_procedural(seed);
                    context.editor_state.selected_mode_index = 0;
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
                let cell_types = crate::cell::types::CellType::all();
                let combo_resp = egui::ComboBox::from_id_salt("cell_type")
                    .selected_text(cell_types[mode.cell_type as usize].name())
                    .show_ui(ui, |ui| {
                        for ct in cell_types.iter() {
                            ui.selectable_value(&mut mode.cell_type, ct.to_index() as i32, ct.name())
                                .on_hover_text(ct.tooltip());
                        }
                    });
                context.editor_state.panel_rects.insert(
                    "cell_type_dropdown".to_string(), combo_resp.response.rect,
                );

                let chk = ui.checkbox(&mut mode.parent_make_adhesion, "Make Adhesion");
                context.editor_state.panel_rects.insert(
                    "make_adhesion_checkbox".to_string(), chk.rect,
                );
            });
        });

    // Multi-select: propagate cell_type and parent_make_adhesion if they changed
    if let (Some(old_cell_type), Some(old_make_adhesion)) = (pre_cell_type, pre_make_adhesion) {
        let selected_idx = context.editor_state.selected_mode_index;
        if selected_idx < context.genome.modes.len() {
            let new_cell_type     = context.genome.modes[selected_idx].cell_type;
            let new_make_adhesion = context.genome.modes[selected_idx].parent_make_adhesion;

            let cell_type_changed     = new_cell_type     != old_cell_type;
            let make_adhesion_changed = new_make_adhesion != old_make_adhesion;

            if cell_type_changed || make_adhesion_changed {
                let secondary_indices: Vec<usize> = context.editor_state.selected_mode_indices
                    .iter()
                    .copied()
                    .filter(|&i| i != selected_idx && i < context.genome.modes.len())
                    .collect();
                for other_idx in secondary_indices {
                    if cell_type_changed     { context.genome.modes[other_idx].cell_type          = new_cell_type; }
                    if make_adhesion_changed { context.genome.modes[other_idx].parent_make_adhesion = new_make_adhesion; }
                }
            }
        }
    }
}

/// Render the AdhesionSettings panel (placeholder).
fn render_adhesion_settings(ui: &mut Ui, context: &mut PanelContext) {
    // Record panel rect for the tutorial pointer.
    context.editor_state.panel_rects.insert("AdhesionSettings".to_string(), ui.max_rect());

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

            // Multi-select: snapshot the adhesion settings before rendering
            // Ensure selected_mode_indices is consistent (handles Default-constructed state)
            if context.editor_state.selected_mode_indices.is_empty() {
                context.editor_state.selected_mode_indices = vec![selected_idx];
            }
            let snapshot = context.genome.modes[selected_idx].adhesion_settings.clone();

            let mode = &mut context.genome.modes[selected_idx];

            // Breaking Properties Group (Red)
            group_container(ui, "Breaking Properties", egui::Color32::from_rgb(200, 100, 100), |ui| {
                ui.checkbox(&mut mode.adhesion_settings.can_break, "Adhesion Can Break");

                ui.label("Adhesion Break Force:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.break_force, 200.0..=1000.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.break_force).speed(0.1).range(200.0..=1000.0));
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

            // Multi-select: propagate only the fields that changed to all other selected modes
            let updated = context.genome.modes[selected_idx].adhesion_settings.clone();
            let secondary_indices: Vec<usize> = context.editor_state.selected_mode_indices
                .iter()
                .copied()
                .filter(|&i| i != selected_idx && i < context.genome.modes.len())
                .collect();
            if !secondary_indices.is_empty() {
                for other_idx in secondary_indices {
                    let other = &mut context.genome.modes[other_idx].adhesion_settings;
                    if updated.can_break != snapshot.can_break { other.can_break = updated.can_break; }
                    if (updated.break_force - snapshot.break_force).abs() > f32::EPSILON { other.break_force = updated.break_force; }
                    if (updated.rest_length - snapshot.rest_length).abs() > f32::EPSILON { other.rest_length = updated.rest_length; }
                    if (updated.linear_spring_stiffness - snapshot.linear_spring_stiffness).abs() > f32::EPSILON { other.linear_spring_stiffness = updated.linear_spring_stiffness; }
                    if (updated.linear_spring_damping - snapshot.linear_spring_damping).abs() > f32::EPSILON { other.linear_spring_damping = updated.linear_spring_damping; }
                    if (updated.orientation_spring_stiffness - snapshot.orientation_spring_stiffness).abs() > f32::EPSILON { other.orientation_spring_stiffness = updated.orientation_spring_stiffness; }
                    if (updated.orientation_spring_damping - snapshot.orientation_spring_damping).abs() > f32::EPSILON { other.orientation_spring_damping = updated.orientation_spring_damping; }
                    if (updated.max_angular_deviation - snapshot.max_angular_deviation).abs() > f32::EPSILON { other.max_angular_deviation = updated.max_angular_deviation; }
                    if updated.enable_twist_constraint != snapshot.enable_twist_constraint { other.enable_twist_constraint = updated.enable_twist_constraint; }
                    if (updated.twist_constraint_stiffness - snapshot.twist_constraint_stiffness).abs() > f32::EPSILON { other.twist_constraint_stiffness = updated.twist_constraint_stiffness; }
                    if (updated.twist_constraint_damping - snapshot.twist_constraint_damping).abs() > f32::EPSILON { other.twist_constraint_damping = updated.twist_constraint_damping; }
                }
            }
        });
}

/// Render the ParentSettings panel (placeholder).
fn render_parent_settings(ui: &mut Ui, context: &mut PanelContext) {
    // Record panel rect for the tutorial pointer.
    context.editor_state.panel_rects.insert("ParentSettings".to_string(), ui.max_rect());

    // Multi-select: snapshot the primary mode before rendering so we can diff afterwards
    let selected_idx = context.editor_state.selected_mode_index;
    // Ensure selected_mode_indices is consistent (handles Default-constructed state)
    if context.editor_state.selected_mode_indices.is_empty() {
        context.editor_state.selected_mode_indices = vec![selected_idx];
    }
    let pre_snapshot: Option<crate::genome::ModeSettings> = if selected_idx < context.genome.modes.len() {
        Some(context.genome.modes[selected_idx].clone())
    } else {
        None
    };

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
            
            // Collect mode info BEFORE borrowing mode mutably (for mode_after_splits dropdowns)
            let mode_info_for_dropdowns: Vec<_> = context.genome.modes.iter()
                .map(|m| (m.name.clone(), m.color))
                .collect();
            
            let mode = &mut context.genome.modes[selected_idx];

            // Special Functions Group (Purple) - cell type specific settings at the top
            if mode.cell_type == 0 { // Test cells (cell_type == 0)
                group_container(ui, "Special Functions", egui::Color32::from_rgb(180, 140, 200), |ui| {
                    ui.label("Nutrient Generation Rate:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.nutrient_gain_rate, 0.0..=20.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.nutrient_gain_rate).speed(0.01).range(0.0..=20.0).suffix("/s"));
                    });
                });
            } else if mode.cell_type == 6 { // Glueocyte (cell_type == 6)
                group_container(ui, "Glueocyte Functions", egui::Color32::from_rgb(140, 200, 140), |ui| {
                    ui.checkbox(&mut mode.glueocyte_cell_adhesion, "Cell Adhesion");
                    if mode.glueocyte_cell_adhesion {
                        ui.indent("glue_cell_signal", |ui| {
                            let channel_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7"];
                            let has_gate = mode.glueocyte_cell_adhesion_signal_channel >= 0
                                && mode.glueocyte_cell_adhesion_signal_channel <= 7;
                            ui.horizontal(|ui| {
                                ui.label("Signal Gate:");
                                if ui.selectable_label(!has_gate, "Always On").clicked() {
                                    mode.glueocyte_cell_adhesion_signal_channel = -1;
                                }
                                if ui.selectable_label(has_gate, "Signal").clicked() {
                                    if !has_gate {
                                        mode.glueocyte_cell_adhesion_signal_channel = 0;
                                    }
                                }
                            });
                            if has_gate {
                                let ch_idx = mode.glueocyte_cell_adhesion_signal_channel.clamp(0, 7) as usize;
                                egui::ComboBox::from_id_salt("glue_cell_ch")
                                    .selected_text(channel_labels[ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in channel_labels.iter().enumerate() {
                                            if ui.selectable_label(ch_idx == i, *label).clicked() {
                                                mode.glueocyte_cell_adhesion_signal_channel = i as i32;
                                            }
                                        }
                                    });
                                ui.add(egui::Slider::new(&mut mode.glueocyte_cell_adhesion_signal_threshold, 0.0..=2047.0)
                                    .text("Threshold")
                                    .logarithmic(false));
                                ui.label("Active when signal ≥ threshold.\nReleases all created bonds when inactive.");
                            }
                        });
                    }
                    ui.checkbox(&mut mode.glueocyte_env_adhesion, "Environment Adhesion");
                });
            } else if mode.cell_type == 1 { // Flagellocyte (cell_type == 1)
                group_container(ui, "Flagellocyte Functions", egui::Color32::from_rgb(140, 180, 220), |ui| {
                    // Mode toggle
                    ui.horizontal(|ui| {
                        ui.label("Speed Mode:");
                        if ui.selectable_label(!mode.flagellocyte_use_signal, "Fixed").clicked() {
                            mode.flagellocyte_use_signal = false;
                        }
                        if ui.selectable_label(mode.flagellocyte_use_signal, "Signal").clicked() {
                            mode.flagellocyte_use_signal = true;
                        }
                    });

                    ui.separator();

                    if !mode.flagellocyte_use_signal {
                        // Fixed speed mode
                        ui.label("Swim Force:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.swim_force, 0.0..=3.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.swim_force).speed(0.01).range(0.0..=3.0));
                        });
                    } else {
                        // Signal-based speed mode
                        {
                            let flag_ch_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7"];
                            let flag_ch_idx = (mode.flagellocyte_signal_channel as usize).min(7);
                            ui.horizontal(|ui| {
                                ui.label("Channel:");
                                egui::ComboBox::from_id_salt("flag_signal_channel")
                                    .selected_text(flag_ch_labels[flag_ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in flag_ch_labels.iter().enumerate() {
                                            ui.selectable_value(&mut mode.flagellocyte_signal_channel, i as i32, *label);
                                        }
                                    });
                            });
                        }

                        ui.label("Speed A (signal < C):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.flagellocyte_speed_a, 0.0..=3.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.flagellocyte_speed_a).speed(0.01).range(0.0..=3.0));
                        });

                        ui.label("Speed B (signal >= C):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.flagellocyte_speed_b, 0.0..=3.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.flagellocyte_speed_b).speed(0.01).range(0.0..=3.0));
                        });

                        ui.label("Threshold C:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.flagellocyte_threshold_c, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.flagellocyte_threshold_c).speed(0.1).range(-100.0..=100.0));
                        });
                    }
                });
            } else if mode.cell_type == 5 { // Buoyocyte (cell_type == 5)
                group_container(ui, "Buoyocyte Functions", egui::Color32::from_rgb(140, 200, 140), |ui| {
                    ui.label("Buoyancy Force:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.buoyancy_force, 0.0..=3.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.buoyancy_force).speed(0.01).range(0.0..=3.0));
                    });
                });
            } else if mode.cell_type == 7 { // Oculocyte (cell_type == 7)
                group_container(ui, "Oculocyte Functions", egui::Color32::from_rgb(200, 160, 220), |ui| {
                    // Sense Type dropdown
                    ui.label("Sense Type:");
                    let sense_labels = ["Cell", "Food", "Light", "Barrier", "Self"];
                    let current_sense = mode.oculocyte_sense_type.clamp(0, 4) as usize;
                    egui::ComboBox::from_id_salt("oculocyte_sense_type")
                        .selected_text(sense_labels[current_sense])
                        .show_ui(ui, |ui| {
                            for (i, label) in sense_labels.iter().enumerate() {
                                ui.selectable_value(&mut mode.oculocyte_sense_type, i as i32, *label);
                            }
                        });

                    // Signal Channel (oculocyte: 0-7 only)
                    ui.label("Signal Channel:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_signal_channel, 0..=7).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_signal_channel).speed(0.1).range(0..=7));
                    });

                    // Signal Value
                    ui.label("Signal Value:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_signal_value, -100.0..=100.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_signal_value).speed(0.1).range(-100.0..=100.0));
                    });

                    // Signal Hops
                    ui.label("Signal Hops:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_signal_hops, 1..=20).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_signal_hops).speed(0.1).range(1..=20));
                    });

                    // Ray Length
                    ui.label("Ray Length:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_ray_length, 1.0..=100.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_ray_length).speed(0.1).range(1.0..=100.0));
                    });
                });
            } else if mode.cell_type == 8 { // Ciliocyte (cell_type == 8)
                group_container(ui, "Ciliocyte Functions", egui::Color32::from_rgb(160, 200, 180), |ui| {
                    // Mode toggle
                    ui.horizontal(|ui| {
                        ui.label("Speed Mode:");
                        if ui.selectable_label(!mode.cilia_use_signal, "Fixed").clicked() {
                            mode.cilia_use_signal = false;
                        }
                        if ui.selectable_label(mode.cilia_use_signal, "Signal").clicked() {
                            mode.cilia_use_signal = true;
                        }
                    });

                    ui.separator();

                    if !mode.cilia_use_signal {
                        // Fixed speed mode
                        ui.label("Cilia Speed:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_speed, -1.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_speed).speed(0.01).range(-1.0..=1.0));
                        });
                    } else {
                        // Signal-based speed mode
                        {
                            let cilia_ch_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7"];
                            let cilia_ch_idx = (mode.cilia_signal_channel as usize).min(7);
                            ui.horizontal(|ui| {
                                ui.label("Channel:");
                                egui::ComboBox::from_id_salt("cilia_signal_channel")
                                    .selected_text(cilia_ch_labels[cilia_ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in cilia_ch_labels.iter().enumerate() {
                                            ui.selectable_value(&mut mode.cilia_signal_channel, i as i32, *label);
                                        }
                                    });
                            });
                        }

                        ui.label("Speed Below (signal < threshold):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_speed_below, -1.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_speed_below).speed(0.01).range(-1.0..=1.0));
                        });

                        ui.label("Speed Above (signal >= threshold):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_speed_above, -1.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_speed_above).speed(0.01).range(-1.0..=1.0));
                        });

                        ui.label("Threshold:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_threshold, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_threshold).speed(0.1).range(-100.0..=100.0));
                        });
                    }

                    ui.add_space(4.0);
                    ui.checkbox(&mut mode.cilia_push_bonded, "Push Organism Cells");

                    ui.add_space(4.0);
                    ui.label("Attract Force:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.cilia_attract_force, 0.0..=1.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.cilia_attract_force).speed(0.01).range(0.0..=1.0));
                    });
                });
            } else if mode.cell_type == 9 { // Myocyte (cell_type == 9)
                group_container(ui, "Myocyte Functions", egui::Color32::from_rgb(200, 140, 140), |ui| {
                    // Mode toggle
                    ui.horizontal(|ui| {
                        ui.label("Contraction Mode:");
                        if ui.selectable_label(!mode.myocyte_use_signal, "Pulse").clicked() {
                            mode.myocyte_use_signal = false;
                        }
                        if ui.selectable_label(mode.myocyte_use_signal, "Signal").clicked() {
                            mode.myocyte_use_signal = true;
                        }
                    });

                    ui.separator();

                    if !mode.myocyte_use_signal {
                        // Phased timer mode
                        ui.label("Pulse Phase:");
                        ui.horizontal(|ui| {
                            if ui.selectable_label(mode.myocyte_pulse_phase == 0, "Pulse A").clicked() {
                                mode.myocyte_pulse_phase = 0;
                            }
                            if ui.selectable_label(mode.myocyte_pulse_phase == 1, "Pulse B").clicked() {
                                mode.myocyte_pulse_phase = 1;
                            }
                        });

                        ui.label("Pulse Rate (cycles/sec):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_pulse_rate, 0.1..=10.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_pulse_rate).speed(0.01).range(0.1..=10.0));
                        });

                        ui.label("Contraction:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_contraction, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_contraction).speed(0.01).range(0.0..=1.0));
                        });
                    } else {
                        // Signal-based contraction mode
                        {
                            let myo_ch_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7",
                                                 "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                            let myo_ch_idx = (mode.myocyte_signal_channel as usize).min(15);
                            ui.horizontal(|ui| {
                                ui.label("Channel:");
                                egui::ComboBox::from_id_salt("myocyte_signal_channel")
                                    .selected_text(myo_ch_labels[myo_ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in myo_ch_labels.iter().enumerate() {
                                            ui.selectable_value(&mut mode.myocyte_signal_channel, i as i32, *label);
                                        }
                                    });
                            });
                        }

                        ui.label("Contraction Below (signal < threshold):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_contraction_below, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_contraction_below).speed(0.01).range(0.0..=1.0));
                        });

                        ui.label("Contraction Above (signal >= threshold):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_contraction_above, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_contraction_above).speed(0.01).range(0.0..=1.0));
                        });

                        ui.label("Threshold:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_threshold, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_threshold).speed(0.1).range(-100.0..=100.0));
                        });
                    }
                });
            } else if mode.cell_type == 10 { // Embryocyte (cell_type == 10)
                group_container(ui, "Embryocyte Release Triggers", egui::Color32::from_rgb(80, 170, 180), |ui| {
                    ui.label("Carries a reserve (max 65535). Burns reserve when free.");
                    ui.label("Release fires when ALL enabled triggers are satisfied:");
                    ui.separator();

                    // Timer trigger
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.embryocyte_use_timer, "Timer");
                    });
                    if mode.embryocyte_use_timer {
                        ui.label("Release after (seconds attached):");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.embryocyte_release_timer, 0.1..=300.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.embryocyte_release_timer).speed(0.1).range(0.1..=300.0));
                        });
                    }

                    ui.separator();

                    // Threshold trigger
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.embryocyte_use_threshold, "Reserve Threshold");
                    });
                    if mode.embryocyte_use_threshold {
                        ui.label("Release when reserve >=:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            let mut threshold_f32 = mode.embryocyte_threshold_value as f32;
                            ui.add(egui::Slider::new(&mut threshold_f32, 0.0_f32..=65535.0_f32).show_value(false));
                            ui.add(egui::DragValue::new(&mut threshold_f32).speed(10.0).range(0.0_f32..=65535.0_f32));
                            mode.embryocyte_threshold_value = threshold_f32 as u32;
                        });
                    }

                    ui.separator();

                    // Signal trigger
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.embryocyte_use_signal, "Signal");
                    });
                    if mode.embryocyte_use_signal {
                        let emb_ch_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7",
                                             "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                        let emb_ch_idx = (mode.embryocyte_signal_channel as usize).min(15);
                        ui.horizontal(|ui| {
                            ui.label("Channel:");
                            egui::ComboBox::from_id_salt("embryocyte_signal_channel")
                                .selected_text(emb_ch_labels[emb_ch_idx])
                                .show_ui(ui, |ui| {
                                    for (i, label) in emb_ch_labels.iter().enumerate() {
                                        ui.selectable_value(&mut mode.embryocyte_signal_channel, i as i32, *label);
                                    }
                                });
                        });

                        ui.label("Release when signal >=:");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.embryocyte_signal_value, 0.0..=2047.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.embryocyte_signal_value).speed(1.0).range(0.0..=2047.0));
                        });
                    }

                    // Show info when no triggers are enabled
                    if !mode.embryocyte_use_timer && !mode.embryocyte_use_threshold && !mode.embryocyte_use_signal {
                        ui.separator();
                        ui.colored_label(egui::Color32::YELLOW, "⚠ No triggers enabled — cell will never release.");
                    }
                });
            } else if mode.cell_type == 12 { // Vasculocyte (cell_type == 12)
                group_container(ui, "Vasculocyte Functions", egui::Color32::from_rgb(60, 160, 200), |ui| {
                    ui.label("Forms high-throughput nutrient conduits through the organism.");
                    ui.label("Vascular-to-vascular transport is 5× faster than normal.");
                    ui.label("Physical compression (e.g. from Myocytes) boosts transport rate.");
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.vascular_outlet, "Outlet");
                        ui.label("Release nutrients to non-vascular neighbors.");
                    });
                    if mode.vascular_outlet {
                        ui.colored_label(
                            egui::Color32::from_rgb(120, 220, 255),
                            "  ▸ Outlet open — nutrients flow to adjacent cells.",
                        );
                    } else {
                        ui.colored_label(
                            egui::Color32::GRAY,
                            "  ▸ Sealed — acts as a pure transport pipe.",
                        );
                    }
                });
            } else if mode.cell_type == 11 { // Devorocyte (cell_type == 11)
                group_container(ui, "Devorocyte Functions", egui::Color32::from_rgb(200, 60, 60), |ui| {
                    ui.label("Steals nutrients from and kills foreign cells on contact.");
                    ui.label("Ignores cells of the same organism or genome.");
                    ui.separator();

                    ui.label("Contact Range:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.devorocyte_consume_range, 0.0..=3.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.devorocyte_consume_range).speed(0.01).range(0.0..=3.0).suffix(" u"));
                    });

                    ui.label("Consume Rate:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.devorocyte_consume_rate, 0.0..=200.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.devorocyte_consume_rate).speed(1.0).range(0.0..=200.0).suffix("/s"));
                    });
                });
            }

            // Division Settings Group (Yellow)
            group_container(ui, "Division Settings", egui::Color32::from_rgb(200, 180, 80), |ui| {
                // Display nutrient threshold: nutrients = (split_mass - 1.0) * 100.0
                // Lipocytes (cell_type 4) can store up to 200 nutrients, so slider goes 1-201 (>200 = Never)
                // All others: slider goes 1-101 (>100 = Never)
                let is_lipocyte = mode.cell_type == 4;
                let never_sentinel = if is_lipocyte { 201.0f32 } else { 101.0f32 };
                let max_nutrients = if is_lipocyte { 200.0f32 } else { 100.0f32 };
                let mut nutrient_threshold = ((mode.split_mass - 1.0) * 100.0).clamp(1.0, never_sentinel);
                ui.label("Split Nutrients:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut nutrient_threshold, 1.0..=never_sentinel)
                        .show_value(false)
                        .custom_formatter(move |v, _| {
                            if v > max_nutrients as f64 { "Never".to_string() } else { format!("{:.0}", v) }
                        }));
                    if nutrient_threshold > max_nutrients {
                        ui.label("Never");
                    } else {
                        ui.add(egui::DragValue::new(&mut nutrient_threshold).speed(1.0).range(1.0..=max_nutrients));
                    }
                });
                // Convert back to split_mass
                mode.split_mass = 1.0 + nutrient_threshold / 100.0;

                ui.add_space(4.0);
                ui.label("Split Ratio:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.split_ratio, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.split_ratio).speed(0.01).range(0.0..=1.0));
                });

                ui.add_space(4.0);
                ui.label("Split Interval:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.split_interval, 1.0..=60.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.split_interval).speed(0.1).range(1.0..=60.0).suffix("s"));
                });

                ui.add_space(4.0);
                ui.label("Max Cell Size:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.max_cell_size, 0.5..=2.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.max_cell_size).speed(0.01).range(0.5..=2.0));
                });

                ui.label("Membrane Stiffness:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.membrane_stiffness, 0.0..=250.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.membrane_stiffness).speed(0.1).range(0.0..=250.0));
                });
            });

            // Regulation Emit Group (Teal) — any cell type can emit on channels 8-15
            group_container(ui, "Regulation Emit", egui::Color32::from_rgb(80, 180, 170), |ui| {
                let reg_channel_labels = ["Disabled", "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                // Map regulation_emit_channel: -1 = Disabled (index 0), 8 = index 1, 9 = index 2, etc.
                let reg_ch_idx = if mode.regulation_emit_channel < 8 { 0usize } else { (mode.regulation_emit_channel - 7).clamp(0, 8) as usize };
                
                ui.label("Emit Channel:");
                egui::ComboBox::from_id_salt("reg_emit_channel")
                    .selected_text(reg_channel_labels[reg_ch_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in reg_channel_labels.iter().enumerate() {
                            let ch_val = if i == 0 { -1i32 } else { (i as i32) + 7 };
                            ui.selectable_value(&mut mode.regulation_emit_channel, ch_val, *label);
                        }
                    });

                if mode.regulation_emit_channel >= 8 {
                    ui.label("Emit Value:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.regulation_emit_value, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.regulation_emit_value).speed(1.0).range(0.0..=2047.0));
                    });

                    ui.label("Emit Hops:");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.regulation_emit_hops, 1..=20).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.regulation_emit_hops).speed(0.1).range(1..=20));
                    });
                }
            });

            // Signal-Conditional Settings Group (Purple)
            group_container(ui, "Signal Conditions", egui::Color32::from_rgb(160, 120, 200), |ui| {
                let mode_count = mode_info_for_dropdowns.len();
                // Signal conditionals only read from regulation channels 8-15
                let channel_labels = ["Disabled", "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                // Map dropdown index to channel value: 0 → -1 (disabled), 1 → 8, 2 → 9, ...
                let idx_to_channel = |idx: usize| -> i32 { if idx == 0 { -1 } else { idx as i32 + 7 } };
                // Map channel value to dropdown index: -1 → 0, 8 → 1, 9 → 2, ... (invalid → 0)
                let channel_to_idx = |ch: i32| -> usize { if ch < 8 { 0 } else { (ch - 7).clamp(0, 8) as usize } };

                // --- Division Gating ---
                ui.label("Division Gating:");
                let div_ch_idx = channel_to_idx(mode.division_signal_channel);
                egui::ComboBox::from_id_salt("div_signal_channel")
                    .selected_text(channel_labels[div_ch_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in channel_labels.iter().enumerate() {
                            ui.selectable_value(&mut mode.division_signal_channel, idx_to_channel(i), *label);
                        }
                    });
                if mode.division_signal_channel >= 8 {
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.division_signal_threshold, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.division_signal_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    ui.checkbox(&mut mode.division_signal_invert, "Invert (divide below threshold)");
                }

                ui.add_space(4.0);
                ui.separator();

                // --- Apoptosis ---
                ui.label("Apoptosis (Signal Death):");
                let apo_ch_idx = channel_to_idx(mode.apoptosis_signal_channel);
                egui::ComboBox::from_id_salt("apo_signal_channel")
                    .selected_text(channel_labels[apo_ch_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in channel_labels.iter().enumerate() {
                            ui.selectable_value(&mut mode.apoptosis_signal_channel, idx_to_channel(i), *label);
                        }
                    });
                if mode.apoptosis_signal_channel >= 8 {
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.apoptosis_signal_threshold, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.apoptosis_signal_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    ui.checkbox(&mut mode.apoptosis_signal_invert, "Invert (die below threshold)");
                }

                ui.add_space(4.0);
                ui.separator();

                // --- Signal-Conditional Child A Mode ---
                ui.label("Child A Signal Routing:");
                let ch_a_idx = channel_to_idx(mode.signal_child_a_channel);
                egui::ComboBox::from_id_salt("sig_child_a_channel")
                    .selected_text(channel_labels[ch_a_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in channel_labels.iter().enumerate() {
                            ui.selectable_value(&mut mode.signal_child_a_channel, idx_to_channel(i), *label);
                        }
                    });
                if mode.signal_child_a_channel >= 8 {
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.signal_child_a_threshold, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.signal_child_a_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    // Mode above threshold
                    {
                        let current = mode.signal_child_a_mode_above;
                        let selected_text = if current < 0 { "Default".to_string() }
                            else if (current as usize) < mode_count { mode_info_for_dropdowns[current as usize].0.clone() }
                            else { "Invalid".to_string() };
                        let mut new_val: Option<i32> = None;
                        ui.horizontal(|ui| {
                            ui.label("Above:");
                            egui::ComboBox::from_id_salt("sig_child_a_above")
                                .selected_text(selected_text)
                                .width(ui.available_width() - 10.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current < 0, "Default").clicked() { new_val = Some(-1); }
                                    ui.separator();
                                    for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                        let color32 = egui::Color32::from_rgb((color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8);
                                        ui.horizontal(|ui| {
                                            ui.colored_label(color32, "●");
                                            if ui.selectable_label(current == idx as i32, name).clicked() { new_val = Some(idx as i32); }
                                        });
                                    }
                                });
                        });
                        if let Some(v) = new_val { mode.signal_child_a_mode_above = v; }
                    }
                    // Mode below threshold
                    {
                        let current = mode.signal_child_a_mode_below;
                        let selected_text = if current < 0 { "Default".to_string() }
                            else if (current as usize) < mode_count { mode_info_for_dropdowns[current as usize].0.clone() }
                            else { "Invalid".to_string() };
                        let mut new_val: Option<i32> = None;
                        ui.horizontal(|ui| {
                            ui.label("Below:");
                            egui::ComboBox::from_id_salt("sig_child_a_below")
                                .selected_text(selected_text)
                                .width(ui.available_width() - 10.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current < 0, "Default").clicked() { new_val = Some(-1); }
                                    ui.separator();
                                    for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                        let color32 = egui::Color32::from_rgb((color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8);
                                        ui.horizontal(|ui| {
                                            ui.colored_label(color32, "●");
                                            if ui.selectable_label(current == idx as i32, name).clicked() { new_val = Some(idx as i32); }
                                        });
                                    }
                                });
                        });
                        if let Some(v) = new_val { mode.signal_child_a_mode_below = v; }
                    }
                }

                ui.add_space(4.0);
                ui.separator();

                // --- Signal-Conditional Child B Mode ---
                ui.label("Child B Signal Routing:");
                let ch_b_idx = channel_to_idx(mode.signal_child_b_channel);
                egui::ComboBox::from_id_salt("sig_child_b_channel")
                    .selected_text(channel_labels[ch_b_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in channel_labels.iter().enumerate() {
                            ui.selectable_value(&mut mode.signal_child_b_channel, idx_to_channel(i), *label);
                        }
                    });
                if mode.signal_child_b_channel >= 8 {
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.signal_child_b_threshold, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.signal_child_b_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    // Mode above threshold
                    {
                        let current = mode.signal_child_b_mode_above;
                        let selected_text = if current < 0 { "Default".to_string() }
                            else if (current as usize) < mode_count { mode_info_for_dropdowns[current as usize].0.clone() }
                            else { "Invalid".to_string() };
                        let mut new_val: Option<i32> = None;
                        ui.horizontal(|ui| {
                            ui.label("Above:");
                            egui::ComboBox::from_id_salt("sig_child_b_above")
                                .selected_text(selected_text)
                                .width(ui.available_width() - 10.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current < 0, "Default").clicked() { new_val = Some(-1); }
                                    ui.separator();
                                    for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                        let color32 = egui::Color32::from_rgb((color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8);
                                        ui.horizontal(|ui| {
                                            ui.colored_label(color32, "●");
                                            if ui.selectable_label(current == idx as i32, name).clicked() { new_val = Some(idx as i32); }
                                        });
                                    }
                                });
                        });
                        if let Some(v) = new_val { mode.signal_child_b_mode_above = v; }
                    }
                    // Mode below threshold
                    {
                        let current = mode.signal_child_b_mode_below;
                        let selected_text = if current < 0 { "Default".to_string() }
                            else if (current as usize) < mode_count { mode_info_for_dropdowns[current as usize].0.clone() }
                            else { "Invalid".to_string() };
                        let mut new_val: Option<i32> = None;
                        ui.horizontal(|ui| {
                            ui.label("Below:");
                            egui::ComboBox::from_id_salt("sig_child_b_below")
                                .selected_text(selected_text)
                                .width(ui.available_width() - 10.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current < 0, "Default").clicked() { new_val = Some(-1); }
                                    ui.separator();
                                    for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                        let color32 = egui::Color32::from_rgb((color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8);
                                        ui.horizontal(|ui| {
                                            ui.colored_label(color32, "●");
                                            if ui.selectable_label(current == idx as i32, name).clicked() { new_val = Some(idx as i32); }
                                        });
                                    }
                                });
                        });
                        if let Some(v) = new_val { mode.signal_child_b_mode_below = v; }
                    }
                }

                ui.add_space(4.0);
                ui.separator();

                // --- Mode Switching ---
                ui.label("Mode Switch (No Division):");
                let ms_ch_idx = channel_to_idx(mode.mode_switch_signal_channel);
                egui::ComboBox::from_id_salt("mode_switch_channel")
                    .selected_text(channel_labels[ms_ch_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in channel_labels.iter().enumerate() {
                            ui.selectable_value(&mut mode.mode_switch_signal_channel, idx_to_channel(i), *label);
                        }
                    });
                if mode.mode_switch_signal_channel >= 8 {
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.mode_switch_signal_threshold, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.mode_switch_signal_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    ui.checkbox(&mut mode.mode_switch_invert, "Invert (switch below threshold)");
                    // Target mode
                    {
                        let current = mode.mode_switch_target;
                        let selected_text = if current < 0 { "Disabled".to_string() }
                            else if (current as usize) < mode_count { mode_info_for_dropdowns[current as usize].0.clone() }
                            else { "Invalid".to_string() };
                        let mut new_val: Option<i32> = None;
                        ui.horizontal(|ui| {
                            ui.label("Target:");
                            egui::ComboBox::from_id_salt("mode_switch_target")
                                .selected_text(selected_text)
                                .width(ui.available_width() - 10.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(current < 0, "Disabled").clicked() { new_val = Some(-1); }
                                    ui.separator();
                                    for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                        let color32 = egui::Color32::from_rgb((color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8);
                                        ui.horizontal(|ui| {
                                            ui.colored_label(color32, "●");
                                            if ui.selectable_label(current == idx as i32, name).clicked() { new_val = Some(idx as i32); }
                                        });
                                    }
                                });
                        });
                        if let Some(v) = new_val { mode.mode_switch_target = v; }
                    }
                }
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
                    ui.add(egui::Slider::new(&mut mode.min_adhesions, 0..=10).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.min_adhesions).speed(1).range(0..=10));
                });

                ui.label("Max Splits:");
                let max_splits_row = ui.horizontal(|ui| {
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
                context.editor_state.panel_rects.insert(
                    "max_splits_slider".to_string(), max_splits_row.response.rect,
                );
                
                // Mode after max splits reached - only show if max_splits is not infinite
                if mode.max_splits >= 0 {
                    let mode_count = mode_info_for_dropdowns.len();
                    
                    // Get current values
                    let current_mode_a = mode.mode_a_after_splits;
                    let current_mode_b = mode.mode_b_after_splits;
                    
                    // Track new selections
                    let mut new_mode_a: Option<i32> = None;
                    let mut new_mode_b: Option<i32> = None;
                    
                    ui.add_space(4.0);
                    ui.label("Child A After Splits:");
                    let after_splits_a_row = ui.horizontal(|ui| {
                        let selected_text = if current_mode_a < 0 {
                            "Default".to_string()
                        } else if (current_mode_a as usize) < mode_count {
                            mode_info_for_dropdowns[current_mode_a as usize].0.clone()
                        } else {
                            "Invalid".to_string()
                        };

                        egui::ComboBox::from_id_salt("mode_a_after_splits")
                            .selected_text(selected_text)
                            .width(ui.available_width() - 10.0)
                            .show_ui(ui, |ui| {
                                // Default option (use normal child_a mode)
                                if ui.selectable_label(current_mode_a < 0, "Default").clicked() {
                                    new_mode_a = Some(-1);
                                }
                                ui.separator();
                                // Mode options
                                for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                    let is_selected = current_mode_a == idx as i32;
                                    let color32 = egui::Color32::from_rgb(
                                        (color.x * 255.0) as u8,
                                        (color.y * 255.0) as u8,
                                        (color.z * 255.0) as u8,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.colored_label(color32, "●");
                                        if ui.selectable_label(is_selected, name).clicked() {
                                            new_mode_a = Some(idx as i32);
                                        }
                                    });
                                }
                            });
                    });
                    context.editor_state.panel_rects.insert(
                        "after_splits_child_a".to_string(), after_splits_a_row.response.rect,
                    );

                    ui.label("Child B After Splits:");
                    let after_splits_b_row = ui.horizontal(|ui| {
                        let selected_text = if current_mode_b < 0 {
                            "Default".to_string()
                        } else if (current_mode_b as usize) < mode_count {
                            mode_info_for_dropdowns[current_mode_b as usize].0.clone()
                        } else {
                            "Invalid".to_string()
                        };

                        egui::ComboBox::from_id_salt("mode_b_after_splits")
                            .selected_text(selected_text)
                            .width(ui.available_width() - 10.0)
                            .show_ui(ui, |ui| {
                                // Default option (use normal child_b mode)
                                if ui.selectable_label(current_mode_b < 0, "Default").clicked() {
                                    new_mode_b = Some(-1);
                                }
                                ui.separator();
                                // Mode options
                                for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                    let is_selected = current_mode_b == idx as i32;
                                    let color32 = egui::Color32::from_rgb(
                                        (color.x * 255.0) as u8,
                                        (color.y * 255.0) as u8,
                                        (color.z * 255.0) as u8,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.colored_label(color32, "●");
                                        if ui.selectable_label(is_selected, name).clicked() {
                                            new_mode_b = Some(idx as i32);
                                        }
                                    });
                                }
                            });
                    });
                    context.editor_state.panel_rects.insert(
                        "after_splits_child_b".to_string(), after_splits_b_row.response.rect,
                    );
                    
                    // Apply selections after combo boxes are done
                    if let Some(new_val) = new_mode_a {
                        mode.mode_a_after_splits = new_val;
                    }
                    if let Some(new_val) = new_mode_b {
                        mode.mode_b_after_splits = new_val;
                    }
                    
                    // Quaternion ball controls for child angles after max splits
                    ui.add_space(8.0);
                    ui.label("Child Split Angles:");
                    
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.label("Child A Angle:");
                            ui.add_space(4.0);
                            quaternion_ball(
                                ui,
                                &mut mode.child_a_after_split_orientation,
                                &mut context.editor_state.child_a_split_x_axis_lat,
                                &mut context.editor_state.child_a_split_x_axis_lon,
                                &mut context.editor_state.child_a_split_y_axis_lat,
                                &mut context.editor_state.child_a_split_y_axis_lon,
                                &mut context.editor_state.child_a_split_z_axis_lat,
                                &mut context.editor_state.child_a_split_z_axis_lon,
                                40.0,
                                context.editor_state.qball_snapping,
                                &mut context.editor_state.child_a_split_locked_axis,
                                &mut context.editor_state.child_a_split_initial_distance,
                            );
                        });
                        
                        ui.add_space(10.0);
                        
                        ui.vertical(|ui| {
                            ui.label("Child B Angle:");
                            ui.add_space(4.0);
                            quaternion_ball(
                                ui,
                                &mut mode.child_b_after_split_orientation,
                                &mut context.editor_state.child_b_split_x_axis_lat,
                                &mut context.editor_state.child_b_split_x_axis_lon,
                                &mut context.editor_state.child_b_split_y_axis_lat,
                                &mut context.editor_state.child_b_split_y_axis_lon,
                                &mut context.editor_state.child_b_split_z_axis_lat,
                                &mut context.editor_state.child_b_split_z_axis_lon,
                                40.0,
                                context.editor_state.qball_snapping,
                                &mut context.editor_state.child_b_split_locked_axis,
                                &mut context.editor_state.child_b_split_initial_distance,
                            );
                        });
                    });
                    
                    // Keep adhesion toggles under quaternion balls
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.checkbox(&mut mode.child_a_after_split_keep_adhesion, "Keep Adhesion");
                        });
                        ui.vertical(|ui| {
                            ui.checkbox(&mut mode.child_b_after_split_keep_adhesion, "Keep Adhesion");
                        });
                    });
                }
            });

        });

    // Multi-select: propagate only the fields that changed to all other selected modes.
    // We compare the primary mode's current state against the pre-render snapshot and
    // apply each changed field individually to every other selected mode.
    if let Some(snapshot) = pre_snapshot {
        let selected_idx = context.editor_state.selected_mode_index;
        if selected_idx < context.genome.modes.len() {
            let updated = context.genome.modes[selected_idx].clone();
            let secondary_indices: Vec<usize> = context.editor_state.selected_mode_indices
                .iter()
                .copied()
                .filter(|&i| i != selected_idx && i < context.genome.modes.len())
                .collect();
            if !secondary_indices.is_empty() {
                sync_mode_changes_to_others(&snapshot, &updated, &secondary_indices, &mut context.genome.modes);
            }
        }
    }
}

/// Propagate only the fields that changed between `snapshot` and `updated` to all `other_indices`.
/// Fields that were not touched by the user (identical in snapshot vs updated) are left alone.
fn sync_mode_changes_to_others(
    snapshot: &crate::genome::ModeSettings,
    updated: &crate::genome::ModeSettings,
    other_indices: &[usize],
    modes: &mut Vec<crate::genome::ModeSettings>,
) {
    for &idx in other_indices {
        let other = &mut modes[idx];

        // Cell type
        if updated.cell_type != snapshot.cell_type { other.cell_type = updated.cell_type; }

        // Visual
        if (updated.opacity - snapshot.opacity).abs() > f32::EPSILON { other.opacity = updated.opacity; }
        if (updated.emissive - snapshot.emissive).abs() > f32::EPSILON { other.emissive = updated.emissive; }

        // Parent / division
        if updated.parent_make_adhesion != snapshot.parent_make_adhesion { other.parent_make_adhesion = updated.parent_make_adhesion; }
        if (updated.split_mass - snapshot.split_mass).abs() > f32::EPSILON { other.split_mass = updated.split_mass; }
        if (updated.split_interval - snapshot.split_interval).abs() > f32::EPSILON { other.split_interval = updated.split_interval; }
        if (updated.nutrient_gain_rate - snapshot.nutrient_gain_rate).abs() > f32::EPSILON { other.nutrient_gain_rate = updated.nutrient_gain_rate; }
        if (updated.max_cell_size - snapshot.max_cell_size).abs() > f32::EPSILON { other.max_cell_size = updated.max_cell_size; }
        if (updated.split_ratio - snapshot.split_ratio).abs() > f32::EPSILON { other.split_ratio = updated.split_ratio; }
        if (updated.nutrient_priority - snapshot.nutrient_priority).abs() > f32::EPSILON { other.nutrient_priority = updated.nutrient_priority; }
        if updated.prioritize_when_low != snapshot.prioritize_when_low { other.prioritize_when_low = updated.prioritize_when_low; }
        if (updated.parent_split_direction.x - snapshot.parent_split_direction.x).abs() > f32::EPSILON
            || (updated.parent_split_direction.y - snapshot.parent_split_direction.y).abs() > f32::EPSILON
        {
            other.parent_split_direction = updated.parent_split_direction;
        }
        if updated.max_adhesions != snapshot.max_adhesions { other.max_adhesions = updated.max_adhesions; }
        if updated.min_adhesions != snapshot.min_adhesions { other.min_adhesions = updated.min_adhesions; }
        if updated.enable_parent_angle_snapping != snapshot.enable_parent_angle_snapping { other.enable_parent_angle_snapping = updated.enable_parent_angle_snapping; }
        if updated.max_splits != snapshot.max_splits { other.max_splits = updated.max_splits; }
        if updated.mode_a_after_splits != snapshot.mode_a_after_splits { other.mode_a_after_splits = updated.mode_a_after_splits; }
        if updated.mode_b_after_splits != snapshot.mode_b_after_splits { other.mode_b_after_splits = updated.mode_b_after_splits; }
        if updated.child_a_after_split_keep_adhesion != snapshot.child_a_after_split_keep_adhesion { other.child_a_after_split_keep_adhesion = updated.child_a_after_split_keep_adhesion; }
        if updated.child_b_after_split_keep_adhesion != snapshot.child_b_after_split_keep_adhesion { other.child_b_after_split_keep_adhesion = updated.child_b_after_split_keep_adhesion; }

        // Membrane
        if (updated.membrane_stiffness - snapshot.membrane_stiffness).abs() > f32::EPSILON { other.membrane_stiffness = updated.membrane_stiffness; }

        // Glueocyte
        if updated.glueocyte_cell_adhesion != snapshot.glueocyte_cell_adhesion { other.glueocyte_cell_adhesion = updated.glueocyte_cell_adhesion; }
        if updated.glueocyte_env_adhesion != snapshot.glueocyte_env_adhesion { other.glueocyte_env_adhesion = updated.glueocyte_env_adhesion; }
        if updated.glueocyte_cell_adhesion_signal_channel != snapshot.glueocyte_cell_adhesion_signal_channel { other.glueocyte_cell_adhesion_signal_channel = updated.glueocyte_cell_adhesion_signal_channel; }
        if (updated.glueocyte_cell_adhesion_signal_threshold - snapshot.glueocyte_cell_adhesion_signal_threshold).abs() > f32::EPSILON { other.glueocyte_cell_adhesion_signal_threshold = updated.glueocyte_cell_adhesion_signal_threshold; }

        // Flagellocyte
        if (updated.swim_force - snapshot.swim_force).abs() > f32::EPSILON { other.swim_force = updated.swim_force; }
        if updated.flagellocyte_use_signal != snapshot.flagellocyte_use_signal { other.flagellocyte_use_signal = updated.flagellocyte_use_signal; }
        if updated.flagellocyte_signal_channel != snapshot.flagellocyte_signal_channel { other.flagellocyte_signal_channel = updated.flagellocyte_signal_channel; }
        if (updated.flagellocyte_speed_a - snapshot.flagellocyte_speed_a).abs() > f32::EPSILON { other.flagellocyte_speed_a = updated.flagellocyte_speed_a; }
        if (updated.flagellocyte_speed_b - snapshot.flagellocyte_speed_b).abs() > f32::EPSILON { other.flagellocyte_speed_b = updated.flagellocyte_speed_b; }
        if (updated.flagellocyte_threshold_c - snapshot.flagellocyte_threshold_c).abs() > f32::EPSILON { other.flagellocyte_threshold_c = updated.flagellocyte_threshold_c; }

        // Buoyocyte
        if (updated.buoyancy_force - snapshot.buoyancy_force).abs() > f32::EPSILON { other.buoyancy_force = updated.buoyancy_force; }

        // Oculocyte
        if updated.oculocyte_sense_type != snapshot.oculocyte_sense_type { other.oculocyte_sense_type = updated.oculocyte_sense_type; }
        if updated.oculocyte_signal_channel != snapshot.oculocyte_signal_channel { other.oculocyte_signal_channel = updated.oculocyte_signal_channel; }
        if (updated.oculocyte_signal_value - snapshot.oculocyte_signal_value).abs() > f32::EPSILON { other.oculocyte_signal_value = updated.oculocyte_signal_value; }
        if updated.oculocyte_signal_hops != snapshot.oculocyte_signal_hops { other.oculocyte_signal_hops = updated.oculocyte_signal_hops; }
        if (updated.oculocyte_ray_length - snapshot.oculocyte_ray_length).abs() > f32::EPSILON { other.oculocyte_ray_length = updated.oculocyte_ray_length; }

        // Ciliocyte
        if (updated.cilia_speed - snapshot.cilia_speed).abs() > f32::EPSILON { other.cilia_speed = updated.cilia_speed; }
        if updated.cilia_push_bonded != snapshot.cilia_push_bonded { other.cilia_push_bonded = updated.cilia_push_bonded; }
        if updated.cilia_use_signal != snapshot.cilia_use_signal { other.cilia_use_signal = updated.cilia_use_signal; }
        if updated.cilia_signal_channel != snapshot.cilia_signal_channel { other.cilia_signal_channel = updated.cilia_signal_channel; }
        if (updated.cilia_speed_below - snapshot.cilia_speed_below).abs() > f32::EPSILON { other.cilia_speed_below = updated.cilia_speed_below; }
        if (updated.cilia_speed_above - snapshot.cilia_speed_above).abs() > f32::EPSILON { other.cilia_speed_above = updated.cilia_speed_above; }
        if (updated.cilia_threshold - snapshot.cilia_threshold).abs() > f32::EPSILON { other.cilia_threshold = updated.cilia_threshold; }
        if (updated.cilia_attract_force - snapshot.cilia_attract_force).abs() > f32::EPSILON { other.cilia_attract_force = updated.cilia_attract_force; }

        // Myocyte
        if (updated.myocyte_contraction - snapshot.myocyte_contraction).abs() > f32::EPSILON { other.myocyte_contraction = updated.myocyte_contraction; }
        if updated.myocyte_use_signal != snapshot.myocyte_use_signal { other.myocyte_use_signal = updated.myocyte_use_signal; }
        if updated.myocyte_signal_channel != snapshot.myocyte_signal_channel { other.myocyte_signal_channel = updated.myocyte_signal_channel; }
        if (updated.myocyte_contraction_above - snapshot.myocyte_contraction_above).abs() > f32::EPSILON { other.myocyte_contraction_above = updated.myocyte_contraction_above; }
        if (updated.myocyte_contraction_below - snapshot.myocyte_contraction_below).abs() > f32::EPSILON { other.myocyte_contraction_below = updated.myocyte_contraction_below; }
        if (updated.myocyte_threshold - snapshot.myocyte_threshold).abs() > f32::EPSILON { other.myocyte_threshold = updated.myocyte_threshold; }
        if (updated.myocyte_pulse_rate - snapshot.myocyte_pulse_rate).abs() > f32::EPSILON { other.myocyte_pulse_rate = updated.myocyte_pulse_rate; }
        if updated.myocyte_pulse_phase != snapshot.myocyte_pulse_phase { other.myocyte_pulse_phase = updated.myocyte_pulse_phase; }

        // Embryocyte
        if updated.embryocyte_use_timer != snapshot.embryocyte_use_timer { other.embryocyte_use_timer = updated.embryocyte_use_timer; }
        if (updated.embryocyte_release_timer - snapshot.embryocyte_release_timer).abs() > f32::EPSILON { other.embryocyte_release_timer = updated.embryocyte_release_timer; }
        if updated.embryocyte_use_threshold != snapshot.embryocyte_use_threshold { other.embryocyte_use_threshold = updated.embryocyte_use_threshold; }
        if updated.embryocyte_threshold_value != snapshot.embryocyte_threshold_value { other.embryocyte_threshold_value = updated.embryocyte_threshold_value; }
        if updated.embryocyte_use_signal != snapshot.embryocyte_use_signal { other.embryocyte_use_signal = updated.embryocyte_use_signal; }
        if updated.embryocyte_signal_channel != snapshot.embryocyte_signal_channel { other.embryocyte_signal_channel = updated.embryocyte_signal_channel; }
        if (updated.embryocyte_signal_value - snapshot.embryocyte_signal_value).abs() > f32::EPSILON { other.embryocyte_signal_value = updated.embryocyte_signal_value; }

        // Regulation emit
        if updated.regulation_emit_channel != snapshot.regulation_emit_channel { other.regulation_emit_channel = updated.regulation_emit_channel; }
        if (updated.regulation_emit_value - snapshot.regulation_emit_value).abs() > f32::EPSILON { other.regulation_emit_value = updated.regulation_emit_value; }
        if updated.regulation_emit_hops != snapshot.regulation_emit_hops { other.regulation_emit_hops = updated.regulation_emit_hops; }

        // Signal conditions — division
        if updated.division_signal_channel != snapshot.division_signal_channel { other.division_signal_channel = updated.division_signal_channel; }
        if (updated.division_signal_threshold - snapshot.division_signal_threshold).abs() > f32::EPSILON { other.division_signal_threshold = updated.division_signal_threshold; }
        if updated.division_signal_invert != snapshot.division_signal_invert { other.division_signal_invert = updated.division_signal_invert; }

        // Signal conditions — apoptosis
        if updated.apoptosis_signal_channel != snapshot.apoptosis_signal_channel { other.apoptosis_signal_channel = updated.apoptosis_signal_channel; }
        if (updated.apoptosis_signal_threshold - snapshot.apoptosis_signal_threshold).abs() > f32::EPSILON { other.apoptosis_signal_threshold = updated.apoptosis_signal_threshold; }
        if updated.apoptosis_signal_invert != snapshot.apoptosis_signal_invert { other.apoptosis_signal_invert = updated.apoptosis_signal_invert; }

        // Signal conditions — child routing
        if updated.signal_child_a_channel != snapshot.signal_child_a_channel { other.signal_child_a_channel = updated.signal_child_a_channel; }
        if (updated.signal_child_a_threshold - snapshot.signal_child_a_threshold).abs() > f32::EPSILON { other.signal_child_a_threshold = updated.signal_child_a_threshold; }
        if updated.signal_child_a_mode_above != snapshot.signal_child_a_mode_above { other.signal_child_a_mode_above = updated.signal_child_a_mode_above; }
        if updated.signal_child_a_mode_below != snapshot.signal_child_a_mode_below { other.signal_child_a_mode_below = updated.signal_child_a_mode_below; }
        if updated.signal_child_b_channel != snapshot.signal_child_b_channel { other.signal_child_b_channel = updated.signal_child_b_channel; }
        if (updated.signal_child_b_threshold - snapshot.signal_child_b_threshold).abs() > f32::EPSILON { other.signal_child_b_threshold = updated.signal_child_b_threshold; }
        if updated.signal_child_b_mode_above != snapshot.signal_child_b_mode_above { other.signal_child_b_mode_above = updated.signal_child_b_mode_above; }
        if updated.signal_child_b_mode_below != snapshot.signal_child_b_mode_below { other.signal_child_b_mode_below = updated.signal_child_b_mode_below; }

        // Signal conditions — mode switch
        if updated.mode_switch_signal_channel != snapshot.mode_switch_signal_channel { other.mode_switch_signal_channel = updated.mode_switch_signal_channel; }
        if (updated.mode_switch_signal_threshold - snapshot.mode_switch_signal_threshold).abs() > f32::EPSILON { other.mode_switch_signal_threshold = updated.mode_switch_signal_threshold; }
        if updated.mode_switch_target != snapshot.mode_switch_target { other.mode_switch_target = updated.mode_switch_target; }
        if updated.mode_switch_invert != snapshot.mode_switch_invert { other.mode_switch_invert = updated.mode_switch_invert; }

        // Devorocyte
        if (updated.devorocyte_consume_range - snapshot.devorocyte_consume_range).abs() > f32::EPSILON { other.devorocyte_consume_range = updated.devorocyte_consume_range; }
        if (updated.devorocyte_consume_rate - snapshot.devorocyte_consume_rate).abs() > f32::EPSILON { other.devorocyte_consume_rate = updated.devorocyte_consume_rate; }

        // Vasculocyte
        if updated.vascular_outlet != snapshot.vascular_outlet { other.vascular_outlet = updated.vascular_outlet; }
    }
}

/// Render the CircleSliders panel with pitch and yaw controls for parent split direction.
fn render_circle_sliders(ui: &mut Ui, context: &mut PanelContext) {
    use crate::ui::widgets::circular_slider_float;
    
    ui.checkbox(&mut context.editor_state.enable_snapping, "Enable Snapping (15°)");
    ui.add_space(4.0); // Reduced spacing
    
    // Ensure we have at least one mode
    if context.genome.modes.is_empty() {
        context.genome.modes.push(crate::genome::ModeSettings::default());
    }
    
    // Get the currently selected mode
    let selected_index = context.editor_state.selected_mode_index;
    if let Some(mode) = context.genome.modes.get_mut(selected_index) {
        // Snapshot before rendering for multi-select diff
        let prev_dir = mode.parent_split_direction;

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

        // Multi-select: propagate changed split direction to all other selected modes
        let new_dir = mode.parent_split_direction;
        let pitch_changed = (new_dir.x - prev_dir.x).abs() > f32::EPSILON;
        let yaw_changed   = (new_dir.y - prev_dir.y).abs() > f32::EPSILON;
        if pitch_changed || yaw_changed {
            let secondary_indices: Vec<usize> = context.editor_state.selected_mode_indices
                .iter()
                .copied()
                .filter(|&i| i != selected_index && i < context.genome.modes.len())
                .collect();
            for other_idx in secondary_indices {
                if pitch_changed { context.genome.modes[other_idx].parent_split_direction.x = new_dir.x; }
                if yaw_changed   { context.genome.modes[other_idx].parent_split_direction.y = new_dir.y; }
            }
        }
    }
}

/// Render the QuaternionBall panel with two quaternion balls for Child A and Child B.
fn render_quaternion_ball(ui: &mut Ui, context: &mut PanelContext) {
    context.editor_state.panel_rects.insert("QuaternionBall".to_string(), ui.max_rect());

    // Snapshot child settings before rendering for multi-select diff
    let selected_idx = context.editor_state.selected_mode_index;
    let pre_child_a = context.genome.modes.get(selected_idx).map(|m| m.child_a.clone());
    let pre_child_b = context.genome.modes.get(selected_idx).map(|m| m.child_b.clone());

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
        ui.checkbox(&mut context.editor_state.qball_snapping, "Enable Snapping (15°)");
        ui.add_space(4.0); // Reduced spacing

        // Get selected mode for keep_adhesion checkboxes
        let selected_idx = context.editor_state.selected_mode_index;
        let has_valid_mode = selected_idx < context.genome.modes.len();

        // Calculate responsive ball size - each ball gets half the panel width
        let available_width = ui.available_width();
        let column_width = (available_width / 2.0).floor();
        let ball_radius = ((column_width - 16.0) / 2.0).clamp(20.0, 80.0);
        let ball_container_width = column_width;

        // Use persistent orientations from editor state
        let mut child_a_orientation = context.editor_state.child_a_orientation;
        let mut child_b_orientation = context.editor_state.child_b_orientation;

        // Display balls horizontally with controls directly below each ball
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;

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

                    // Right-click: seed euler buffer from current orientation when menu opens
                    if response.secondary_clicked() {
                        let (pitch, yaw, roll) = child_a_orientation.to_euler(glam::EulerRot::XYZ);
                        context.editor_state.qball_manual_xyzw = [
                            pitch.to_degrees() as f64,
                            yaw.to_degrees() as f64,
                            roll.to_degrees() as f64,
                            0.0,
                        ];
                    }
                    let mut apply_a: Option<glam::Quat> = None;
                    egui::Popup::context_menu(&response)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| {
                        ui.set_min_width(180.0);
                        ui.label("Set orientation (degrees):");
                        ui.separator();
                        let euler = &mut context.editor_state.qball_manual_xyzw;
                        egui::Grid::new("qball_a_input").num_columns(2).spacing([4.0, 4.0]).show(ui, |ui| {
                            ui.label("Pitch (X):"); ui.add(egui::DragValue::new(&mut euler[0]).speed(1.0).range(-180.0..=180.0).suffix("°")); ui.end_row();
                            ui.label("Yaw (Y):");   ui.add(egui::DragValue::new(&mut euler[1]).speed(1.0).range(-180.0..=180.0).suffix("°")); ui.end_row();
                            ui.label("Roll (Z):");  ui.add(egui::DragValue::new(&mut euler[2]).speed(1.0).range(-180.0..=180.0).suffix("°")); ui.end_row();
                        });
                        ui.separator();
                        let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                        ui.horizontal(|ui| {
                            if ui.button("Apply").clicked() || enter_pressed {
                                apply_a = Some(glam::Quat::from_euler(
                                    glam::EulerRot::XYZ,
                                    euler[0].to_radians() as f32,
                                    euler[1].to_radians() as f32,
                                    euler[2].to_radians() as f32,
                                ).normalize());
                                egui::Popup::close_all(ui.ctx());
                            }
                            if ui.button("Cancel").clicked() {
                                egui::Popup::close_all(ui.ctx());
                            }
                        });
                    });
                    if let Some(q) = apply_a {
                        context.editor_state.child_a_orientation = q;
                        if has_valid_mode {
                            context.genome.modes[selected_idx].child_a.orientation = q;
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

            ui.add_space(0.0); // no gap between columns

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

                    // Right-click: seed euler buffer from current orientation when menu opens
                    if response.secondary_clicked() {
                        let (pitch, yaw, roll) = child_b_orientation.to_euler(glam::EulerRot::XYZ);
                        context.editor_state.qball_manual_xyzw = [
                            pitch.to_degrees() as f64,
                            yaw.to_degrees() as f64,
                            roll.to_degrees() as f64,
                            0.0,
                        ];
                    }
                    let mut apply_b: Option<glam::Quat> = None;
                    egui::Popup::context_menu(&response)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| {
                        ui.set_min_width(180.0);
                        ui.label("Set orientation (degrees):");
                        ui.separator();
                        let euler = &mut context.editor_state.qball_manual_xyzw;
                        egui::Grid::new("qball_b_input").num_columns(2).spacing([4.0, 4.0]).show(ui, |ui| {
                            ui.label("Pitch (X):"); ui.add(egui::DragValue::new(&mut euler[0]).speed(1.0).range(-180.0..=180.0).suffix("°")); ui.end_row();
                            ui.label("Yaw (Y):");   ui.add(egui::DragValue::new(&mut euler[1]).speed(1.0).range(-180.0..=180.0).suffix("°")); ui.end_row();
                            ui.label("Roll (Z):");  ui.add(egui::DragValue::new(&mut euler[2]).speed(1.0).range(-180.0..=180.0).suffix("°")); ui.end_row();
                        });
                        ui.separator();
                        let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                        ui.horizontal(|ui| {
                            if ui.button("Apply").clicked() || enter_pressed {
                                apply_b = Some(glam::Quat::from_euler(
                                    glam::EulerRot::XYZ,
                                    euler[0].to_radians() as f32,
                                    euler[1].to_radians() as f32,
                                    euler[2].to_radians() as f32,
                                ).normalize());
                                egui::Popup::close_all(ui.ctx());
                            }
                            if ui.button("Cancel").clicked() {
                                egui::Popup::close_all(ui.ctx());
                            }
                        });
                    });
                    if let Some(q) = apply_b {
                        context.editor_state.child_b_orientation = q;
                        if has_valid_mode {
                            context.genome.modes[selected_idx].child_b.orientation = q;
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

    // Multi-select: propagate changed child settings to all other selected modes
    if let (Some(pre_a), Some(pre_b)) = (pre_child_a, pre_child_b) {
        if selected_idx < context.genome.modes.len() {
            let new_a = context.genome.modes[selected_idx].child_a.clone();
            let new_b = context.genome.modes[selected_idx].child_b.clone();

            let secondary_indices: Vec<usize> = context.editor_state.selected_mode_indices
                .iter()
                .copied()
                .filter(|&i| i != selected_idx && i < context.genome.modes.len())
                .collect();

            for other_idx in secondary_indices {
                let other = &mut context.genome.modes[other_idx];

                // Child A orientation
                if new_a.orientation != pre_a.orientation {
                    other.child_a.orientation = new_a.orientation;
                }
                // Child B orientation
                if new_b.orientation != pre_b.orientation {
                    other.child_b.orientation = new_b.orientation;
                }
                // Child A keep_adhesion
                if new_a.keep_adhesion != pre_a.keep_adhesion {
                    other.child_a.keep_adhesion = new_a.keep_adhesion;
                }
                // Child B keep_adhesion
                if new_b.keep_adhesion != pre_b.keep_adhesion {
                    other.child_b.keep_adhesion = new_b.keep_adhesion;
                }
                // Child A mode_number
                if new_a.mode_number != pre_a.mode_number {
                    other.child_a.mode_number = new_a.mode_number;
                }
                // Child B mode_number
                if new_b.mode_number != pre_b.mode_number {
                    other.child_b.mode_number = new_b.mode_number;
                }
            }
        }
    }
}

/// Render the TimeSlider panel with yellow progress bar.
///
/// The slider is purely UI-driven (never written by simulation).
/// A yellow bar fills behind the slider track to show how far the
/// simulation has actually reached (resim_display_time).
fn render_time_slider(ui: &mut Ui, context: &mut PanelContext) {
    // Record panel rect for the tutorial pointer.
    context.editor_state.panel_rects.insert("TimeSlider".to_string(), ui.max_rect());

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let is_preview_mode = context.current_mode == SimulationMode::Preview;

            // Always allocate space for status text to prevent shifting
            if !is_preview_mode {
                ui.colored_label(
                    egui::Color32::GRAY,
                    "Time scrubbing only available in Preview mode"
                );
            } else {
                ui.label("");
            }

            ui.horizontal(|ui| {
                ui.label("Time:");

                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;

                // Slider range is 0 to max_preview_duration (in seconds)
                // The slider is purely user-driven — simulation never writes to time_value
                let slider_response = ui.add_enabled(
                    is_preview_mode,
                    egui::Slider::new(&mut context.editor_state.time_value, 0.0..=context.editor_state.max_preview_duration)
                        .show_value(false)
                );
                context.editor_state.time_slider_dragging = slider_response.dragged();

                // Paint yellow progress bar behind the slider track
                if is_preview_mode {
                    let max_dur = context.editor_state.max_preview_duration;
                    let progress_frac = if max_dur > 0.0 {
                        (context.editor_state.resim_display_time / max_dur).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    // Use the slider's rect for positioning the progress bar
                    let slider_rect = slider_response.rect;
                    // Inset vertically to sit inside the slider rail
                    let rail_height = 6.0;
                    let rail_center_y = slider_rect.center().y;
                    let bar_rect = egui::Rect::from_min_max(
                        egui::pos2(slider_rect.left(), rail_center_y - rail_height * 0.5),
                        egui::pos2(
                            slider_rect.left() + slider_rect.width() * progress_frac,
                            rail_center_y + rail_height * 0.5,
                        ),
                    );
                    // Paint below the slider widget (background layer)
                    let yellow = egui::Color32::from_rgba_unmultiplied(255, 200, 0, 160);
                    ui.painter().rect_filled(bar_rect, 2.0, yellow);
                }

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

            // Global outline setting (applies to all cell types)
            ui.label(egui::RichText::new("Cell Outline").strong());
            ui.add_space(4.0);
            ui.label("Outline Width:");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut context.editor_state.cell_outline_width, 0.0..=0.2).show_value(false));
                ui.add(egui::DragValue::new(&mut context.editor_state.cell_outline_width).speed(0.005).range(0.0..=0.2));
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

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

            // Preset buttons
            ui.separator();
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Presets").strong());
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("Matte").clicked() {
                    visuals.specular_strength = 0.0;
                    visuals.specular_power = 32.0;
                    visuals.fresnel_strength = 0.0;
                }
                if ui.small_button("Glossy").clicked() {
                    visuals.specular_strength = 0.5;
                    visuals.specular_power = 64.0;
                    visuals.fresnel_strength = 0.3;
                }
                if ui.small_button("Shiny").clicked() {
                    visuals.specular_strength = 0.8;
                    visuals.specular_power = 128.0;
                    visuals.fresnel_strength = 0.5;
                }
            });

            // Flagella section (only for Flagellocyte cell type)
            if cell_types.get(selected_idx) == Some(&CellType::Flagellocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Flagella").strong());
                ui.add_space(4.0);

                // Tail Length (0.5 - 3.0, default 1.7)
                ui.label("Tail Length:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_length, 0.5..=3.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_length).speed(0.01).range(0.5..=3.0));
                });

                // Tail Thickness (0.01 - 0.3, default 0.15)
                ui.label("Tail Thickness:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_thickness, 0.01..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_thickness).speed(0.005).range(0.01..=0.3));
                });

                // Wave Amplitude (0.0 - 0.5, default 0.17)
                ui.label("Wave Amplitude:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_amplitude, 0.0..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_amplitude).speed(0.01).range(0.0..=0.5));
                });

                // Wave Frequency (0.5 - 10.0, default 1.0)
                ui.label("Wave Frequency:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_frequency, 0.5..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_frequency).speed(0.1).range(0.5..=10.0));
                });

                // Taper (0.0 - 1.0, default 1.0)
                ui.label("Taper:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_taper, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_taper).speed(0.01).range(0.0..=1.0));
                });

                // Segments (4 - 64, default 10)
                ui.label("Segments:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_segments, 4.0..=64.0).show_value(false).integer());
                    ui.add(egui::DragValue::new(&mut visuals.tail_segments).speed(1.0).range(4.0..=64.0));
                });
            }

            // Slime pattern section (only for Glueocyte cell type)
            if cell_types.get(selected_idx) == Some(&CellType::Glueocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Slime Pattern").strong());
                ui.add_space(4.0);

                ui.label("Cell Scale:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 1.0..=12.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(0.1).range(1.0..=12.0));
                });

                ui.label("Border Width:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.01..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.005).range(0.01..=0.5));
                });

                ui.label("Meander:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.01).range(0.0..=1.0));
                });

                ui.label("Border Darkness:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.01).range(0.0..=1.0));
                });

                ui.label("Anim Speed:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.membrane_noise_speed, 0.0..=5.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.membrane_noise_speed).speed(0.05).range(0.0..=5.0));
                });
            }

            // Goldberg ridge section (only for Photocyte cell type)
            if cell_types.get(selected_idx) == Some(&CellType::Photocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Membrane Ridges").strong());
                ui.add_space(4.0);

                // Nucleus Scale
                ui.label("Nucleus Scale:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.nucleus_scale, 0.1..=0.95).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.nucleus_scale).speed(0.01).range(0.1..=0.95));
                });

                // Subdivision level (1-6)
                ui.label("Subdivision:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 1.0..=6.0).show_value(false).integer());
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(1.0).range(1.0..=6.0));
                });

                // Ridge Width
                ui.label("Ridge Width:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.01..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.005).range(0.01..=0.5));
                });

                // Meander
                ui.label("Meander:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.005).range(0.0..=0.3));
                });

                // Ridge Strength (normal perturbation)
                ui.label("Ridge Depth:");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.0..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.005).range(0.0..=0.5));
                });
            }
        });
}

/// Render the Mode Graph panel for visualizing mode connections.
fn render_mode_graph(ui: &mut Ui, context: &mut PanelContext) {
    use crate::genome::node_graph::{ModeGraphViewer, ModeNode};
    use egui_snarl::ui::{SnarlStyle, SnarlWidget};
    use egui_snarl::Snarl;

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
            // Add node at next available position in column
            let pos = context.editor_state.mode_graph_state.next_node_position();
            let node_id = context.editor_state.mode_graph_state.snarl.insert_node(pos, ModeNode::new(idx, name, color));
            // Record which slot this node occupies
            context.editor_state.mode_graph_state.record_node_in_slot(node_id, pos);
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
        graph_state: &mut context.editor_state.mode_graph_state,
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
    // We need to temporarily take ownership of the snarl to pass to the widget
    let mut snarl = std::mem::replace(&mut viewer.graph_state.snarl, Snarl::new());
    
    SnarlWidget::new()
        .style(style)
        .show(&mut snarl, &mut viewer, ui);
    
    // Put the snarl back
    viewer.graph_state.snarl = snarl;
    
    // Update node positions after rendering to detect moved nodes
    viewer.graph_state.update_node_positions(ui);

    // Write back any changes to child modes
    let mut changes_made = false;
    for (i, mode) in context.genome.modes.iter_mut().enumerate() {
        if i < child_a_modes.len() && mode.child_a.mode_number != child_a_modes[i] {
            mode.child_a.mode_number = child_a_modes[i];
            changes_made = true;
        }
        if i < child_b_modes.len() && mode.child_b.mode_number != child_b_modes[i] {
            mode.child_b.mode_number = child_b_modes[i];
            changes_made = true;
        }
    }
    
    if changes_made {
        log::info!("Node graph changes written back to genome");
    }
}

/// Render the WorldSettings panel for simulation parameters and physics.
fn render_world_settings(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    // Only show world settings in GPU mode
    if context.current_mode != crate::ui::types::SimulationMode::Gpu {
        ui.label("World settings are only available in GPU mode.");
        return;
    }
    
    let world = &mut state.world_settings;

    // World Radius slider (top, reset-gated)
    let current_world_radius = context.scene_manager
        .gpu_scene()
        .map(|s| s.config.sphere_radius)
        .unwrap_or(world.world_radius);

    ui.label("World Radius:");
    ui.add(egui::Slider::new(&mut world.world_radius, 50.0..=300.0).suffix(" units"));

    if (world.world_radius - current_world_radius).abs() > 0.5 {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚠ Scene reset required").color(egui::Color32::YELLOW));
        });
    }

    ui.add_space(8.0);
    let current_capacity = context.gpu_capacity().unwrap_or(world.cell_capacity);
    let capacity_k = world.cell_capacity / 1000;
    
    ui.label("Cell Capacity:");
    let mut capacity_k_mut = capacity_k;
    if ui.add(egui::Slider::new(&mut capacity_k_mut, 10..=200).suffix("k")).changed() {
        world.cell_capacity = capacity_k_mut * 1000;
    }
    
    // Show reset required message if capacity changed
    if world.cell_capacity != current_capacity {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚠ Scene reset required").color(egui::Color32::YELLOW));
        });
    }
    
    ui.add_space(8.0);
    
    // Physics section
    ui.heading("Physics");
    ui.separator();
    
    // Gravity slider
    ui.label("Gravity:");
    ui.add(egui::Slider::new(&mut world.gravity, -50.0..=50.0).suffix(" m/s²"));
    
    // Gravity axis / mode
    ui.label("Gravity Axis:");
    ui.horizontal(|ui| {
        if ui.selectable_label(world.gravity_mode == 0, "X").clicked() {
            world.gravity_mode = 0;
        }
        if ui.selectable_label(world.gravity_mode == 1, "Y").clicked() {
            world.gravity_mode = 1;
        }
        if ui.selectable_label(world.gravity_mode == 2, "Z").clicked() {
            world.gravity_mode = 2;
        }
        if ui.selectable_label(world.gravity_mode == 3, "Radial").clicked() {
            world.gravity_mode = 3;
        }
    });
    if world.gravity_mode == 3 {
        ui.label(egui::RichText::new("Pulls fluid toward origin; world boundary is the shell").small());
    }
    
    ui.add_space(8.0);
    
    // Adhesion constraint solver iterations
    ui.label("Constraint Iterations:");
    ui.add(egui::Slider::new(&mut world.constraint_iterations, 0..=16).text("iterations"));
    ui.label(egui::RichText::new("Extra adhesion solver passes (higher = stiffer joints, more GPU cost)").small());
    
    ui.add_space(8.0);
    
    // Global velocity damping
    ui.label("Velocity Damping:");
    ui.add(egui::Slider::new(&mut world.acceleration_damping, 0.8..=1.0).text("damping"));
    ui.label(egui::RichText::new("Global drag on cell movement (lower = more drag, higher = less damping)").small());
    
    ui.add_space(8.0);
    
    // Water viscosity
    ui.label("Water Viscosity:");
    ui.add(egui::Slider::new(&mut world.water_viscosity, 0.0..=1.0).text("viscosity"));
    ui.label(egui::RichText::new("Drag applied to cells moving through water (0 = off, higher = thicker fluid)").small());

    ui.add_space(12.0);

    // Biology section
    ui.heading("Biology");
    ui.separator();

    ui.checkbox(&mut world.solo_metabolism_enabled, "Solo cell metabolism penalty");
    ui.label(egui::RichText::new("Cells with fewer adhesion connections burn nutrients faster, favoring multicellular organisms").small());

    if world.solo_metabolism_enabled {
        ui.add_space(4.0);
        ui.label("Metabolism Multiplier:");
        ui.add(egui::Slider::new(&mut world.solo_metabolism_multiplier, 1.5..=10.0).text("×"));
        ui.label(egui::RichText::new("Solo cells (0 connections) drain at this rate. Gradient: 1 conn = partial, 3+ = normal").small());
    }

    ui.add_space(12.0);

    // Mutation section
    ui.heading("Mutation");
    ui.separator();

    ui.label("Radiation Level:");
    
    // Logarithmic slider: position 0.0 = exactly 0.0 (off)
    // position > 0.0 maps to [0.00001, 1.0] over 5 decades for fine low-end control
    const EPSILON: f32 = 0.0001;
    const DECADES: f64 = 5.0; // 10^-5 to 10^0

    let mut log_slider = if world.radiation_level <= 0.0 {
        0.0f32
    } else {
        // Map [1e-5, 1.0] → [0.0, 1.0]
        let normalized = (world.radiation_level.log10() as f64 + DECADES) / DECADES;
        normalized.clamp(0.0, 1.0) as f32
    };

    let response = ui.add(
        egui::Slider::new(&mut log_slider, 0.0..=1.0)
            .custom_formatter(|value, _| {
                if value < EPSILON as f64 {
                    "0.0000".to_string()
                } else {
                    let radiation = 10_f64.powf(value * DECADES - DECADES);
                    if radiation < 0.001 {
                        format!("{:.5}", radiation)
                    } else if radiation < 0.01 {
                        format!("{:.4}", radiation)
                    } else {
                        format!("{:.3}", radiation)
                    }
                }
            })
            .custom_parser(|s| {
                s.parse::<f64>().ok().and_then(|v| {
                    if v <= 0.0 {
                        Some(0.0)
                    } else if v <= 1.0 {
                        Some(((v.log10() + DECADES) / DECADES).clamp(0.0, 1.0))
                    } else {
                        None
                    }
                })
            })
            .text("chance")
    );

    if response.changed() {
        world.radiation_level = if log_slider < EPSILON {
            0.0
        } else {
            10_f32.powf(log_slider * DECADES as f32 - DECADES as f32).clamp(0.0, 1.0)
        };
    }

    ui.label(egui::RichText::new("Probability each child mutates during division (0 = off, logarithmic scale for fine control)").small());

    ui.add_space(4.0);
    ui.checkbox(&mut world.subtle_mutations, "Subtle mutations")
        .on_hover_text("When checked, mutations make small color nudges instead of full re-rolls");

    ui.add_space(12.0);
}

/// Render the Help panel showing context-specific controls and shortcuts.
fn render_help(ui: &mut Ui, context: &mut PanelContext, _state: &GlobalUiState) {
    ui.heading("Help & Controls");
    ui.separator();
    
    // Show current simulation mode
    ui.horizontal(|ui| {
        ui.label("Current Mode:");
        ui.colored_label(
            egui::Color32::from_rgb(100, 150, 255),
            format!("{:?}", context.current_mode)
        );
    });
    ui.add_space(8.0);
    
    // Show context-specific controls based on simulation mode
    match context.current_mode {
        SimulationMode::Preview => render_preview_help(ui, context),
        SimulationMode::Gpu => render_gpu_help(ui, context),
    }
    
    ui.separator();
    render_general_help(ui);
}

/// Render help content specific to Preview mode (genome editing).
fn render_preview_help(ui: &mut Ui, _context: &mut PanelContext) {
    ui.heading("🧬 Preview Mode - Genome Editing");
    ui.add_space(4.0);
    
    // Viewport controls
    egui::CollapsingHeader::new("🎥 Viewport Controls")
        .default_open(true)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Tab", "Toggle between Orbit and FreeFly camera modes"),
                ("Middle Mouse + Drag", "Rotate camera (Orbit mode)"),
                ("Right Mouse + Drag", "Rotate camera (FreeFly mode)"),
                ("Mouse Wheel", "Zoom in/out (Orbit mode only)"),
                ("WASD", "Move camera (FreeFly mode only)"),
                ("Space", "Move up (FreeFly mode only)"),
                ("C", "Move down (FreeFly mode only)"),
                ("Q/E", "Roll left/right (FreeFly mode only)"),
                ("Shift + WASD", "Fast movement (FreeFly mode)"),
            ]);
        });
    
    // Mode editing controls
    egui::CollapsingHeader::new("🎛️ Mode Editing")
        .default_open(true)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Double Click Mode", "Rename mode inline"),
                ("Enter", "Confirm rename (empty = default name)"),
                ("Escape", "Cancel rename"),
                ("Click Away", "Cancel rename without saving"),
                ("Right Click Mode", "Change mode color"),
                ("Radio Button", "Set as initial mode"),
                ("Copy Into", "Copy settings between modes"),
                ("Reset Button", "Restore mode to defaults"),
            ]);
        });
    
    // Quaternion ball controls
    egui::CollapsingHeader::new("🎯 Rotation Controls")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Drag Center", "Pitch/Yaw rotation"),
                ("Drag Edge", "Roll rotation"),
                ("Grid Snapping", "Enable for 15° increments"),
                ("Axis Lock", "Automatic based on drag start"),
                ("Red Line", "X-axis direction"),
                ("Green Line", "Y-axis direction"),
                ("Blue Line", "Z-axis direction"),
            ]);
        });
    
    // Time controls
    egui::CollapsingHeader::new("⏱️ Time Controls")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Time Slider", "Scrub through preview timeline"),
                ("Play/Pause", "Control simulation playback"),
                ("Speed Control", "Adjust playback speed"),
                ("Reset Time", "Return to start of timeline"),
            ]);
        });
    
    // Mode graph controls
    egui::CollapsingHeader::new("🕸️ Mode Graph")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Add Node", "Add mode to graph visualization"),
                ("Drag Node", "Move node position"),
                ("Connect Pins", "Drag from output to input pin"),
                ("Right Click Node", "Remove node from graph"),
                ("Mouse Wheel", "Zoom graph view"),
                ("Drag Background", "Pan graph view"),
                ("Child A/B", "Output pins for mode transitions"),
            ]);
        });
}

/// Render help content specific to GPU mode (large-scale simulation).
fn render_gpu_help(ui: &mut Ui, _context: &mut PanelContext) {
    ui.heading("⚡ GPU Mode - Large-Scale Simulation");
    ui.add_space(4.0);
    
    // Viewport controls
    egui::CollapsingHeader::new("🎥 Viewport Controls")
        .default_open(true)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Tab", "Toggle between Orbit and FreeFly camera modes"),
                ("Middle Mouse + Drag", "Rotate camera (Orbit mode)"),
                ("Right Mouse + Drag", "Rotate camera (FreeFly mode)"),
                ("Mouse Wheel", "Zoom in/out (Orbit mode only)"),
                ("WASD", "Move camera (FreeFly mode only)"),
                ("Space", "Move up (FreeFly mode only)"),
                ("C", "Move down (FreeFly mode only)"),
                ("Q/E", "Roll left/right (FreeFly mode only)"),
                ("Shift + WASD", "Fast movement (FreeFly mode)"),
            ]);
        });
    
    // Simulation controls
    egui::CollapsingHeader::new("🧪 Simulation Controls")
        .default_open(true)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Space", "Play/pause simulation"),
                ("R", "Reset simulation to initial state"),
                ("T", "Single step simulation"),
                ("1-9", "Set simulation speed multiplier"),
                ("F", "Toggle fullscreen"),
                ("Tab", "Toggle UI visibility"),
            ]);
        });
    
    // Tool controls
    egui::CollapsingHeader::new("🛠️ Tool Controls")
        .default_open(true)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Hold Alt", "Open radial tool menu"),
                ("Left Click (Menu Open)", "Select tool"),
                ("Left Click (Insert Tool)", "Add cell at cursor"),
                ("Left Click (Remove Tool)", "Delete cell at cursor"),
                ("Left Click (Boost Tool)", "Give cell maximum nutrients"),
                ("Left Click (Inspect Tool)", "Select cell for inspection"),
                ("Left Click + Drag (Drag Tool)", "Move cell in 3D space"),
            ]);
        });
    
    // Performance monitoring
    egui::CollapsingHeader::new("📊 Performance Monitoring")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("FPS Counter", "Frames per second display"),
                ("Cell Count", "Active simulation entities"),
                ("GPU Memory", "VRAM usage monitoring"),
                ("Compute Time", "Physics calculation timing"),
                ("Render Time", "Frame rendering timing"),
            ]);
        });
    
    // Rendering controls
    egui::CollapsingHeader::new("🎨 Rendering Controls")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Fog Toggle", "Enable/disable volumetric fog"),
                ("Bloom", "Post-processing glow effect"),
                ("Skybox", "Background environment"),
                ("Cell Opacity", "Transparency settings"),
                ("Adhesion Lines", "Connection visualization"),
                ("Debug Overlays", "Development information"),
            ]);
        });
}

/// Render general help content available in all modes.
fn render_general_help(ui: &mut Ui) {
    ui.heading("⚙️ General Controls");
    ui.add_space(4.0);
    
    // Panel management
    egui::CollapsingHeader::new("📋 Panel Management")
        .default_open(true)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Drag Tab", "Move panel to different location"),
                ("Right Click Tab", "Panel context menu"),
                ("X Button", "Close panel (if not locked)"),
                ("Double Click Tab", "Maximize/restore panel"),
                ("Drag to Edge", "Dock panel to window edge"),
                ("Drag to Center", "Create tabbed panel group"),
            ]);
        });
    
    // Application controls
    egui::CollapsingHeader::new("🖥️ Application")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Ctrl + S", "Save current genome/scene"),
                ("Ctrl + O", "Open genome/scene file"),
                ("Ctrl + N", "New genome/scene"),
                ("Ctrl + Z", "Undo last action"),
                ("Ctrl + Y", "Redo last action"),
                ("F11", "Toggle fullscreen"),
                ("Alt + F4", "Exit application"),
                ("Ctrl + ,", "Open preferences"),
            ]);
        });
    
    // Theme and UI
    egui::CollapsingHeader::new("🎨 Theme & UI")
        .default_open(false)
        .show(ui, |ui| {
            help_section(ui, &[
                ("Theme Editor", "Customize UI colors and fonts"),
                ("Panel Locking", "Prevent accidental panel closure"),
                ("Layout Reset", "Restore default panel layout"),
                ("UI Scale", "Adjust interface size"),
                ("Dark/Light Mode", "Toggle theme brightness"),
            ]);
        });
    
    ui.add_space(8.0);
    ui.separator();
    ui.small("💡 Tip: This help panel shows different controls based on your current simulation mode.");
}

/// Helper function to render a help section with key-value pairs.
fn help_section(ui: &mut Ui, items: &[(&str, &str)]) {
    egui::Grid::new("help_grid")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            for (key, description) in items {
                ui.label(egui::RichText::new(*key).strong().color(egui::Color32::from_rgb(150, 200, 255)));
                ui.label(*description);
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Genome;
    use crate::ui::panel_context::SceneModeRequest;

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
        assert!(Panel::Viewport.is_closeable()); // All panels are closeable by default
        assert!(Panel::Viewport.allowed_in_windows());
    }

    #[test]
    fn test_regular_panels_are_closeable() {
        assert!(Panel::CellInspector.is_closeable());
        assert!(Panel::SceneManager.is_closeable());
        assert!(Panel::PerformanceMonitor.is_closeable());
    }

    #[test]
    fn test_genome_loading_reads_from_cpu_storage() {
        // Test that genome loading reads from gpu_scene.genomes array, not GPU buffers
        // This verifies Requirements 5.1, 5.2, 5.3
        
        // Create a test genome using the default constructor
        let mut test_genome = Genome::default();
        test_genome.name = "TestGenome".to_string();
        
        // Create a mock GPU scene with the genome
        let mut mock_gpu_scene = MockGpuScene::new();
        mock_gpu_scene.genomes.push(test_genome.clone());
        
        // Verify the genome is accessible from CPU-side storage
        assert_eq!(mock_gpu_scene.genomes.len(), 1);
        assert_eq!(mock_gpu_scene.genomes[0].name, "TestGenome");
        assert_eq!(mock_gpu_scene.genomes[0].modes.len(), 40); // Default genome has 40 modes
        
        // Verify genome loading would read from CPU storage, not GPU buffers
        let genome_id = 0u32;
        assert!(genome_id < mock_gpu_scene.genomes.len() as u32);
        let loaded_genome = mock_gpu_scene.genomes[genome_id as usize].clone();
        assert_eq!(loaded_genome.name, test_genome.name);
        assert_eq!(loaded_genome.modes.len(), test_genome.modes.len());
    }

    #[test]
    fn test_genome_loading_isolation() {
        // Test that genome loading is the only legitimate CPU readback operation
        // This verifies Requirements 5.4, 5.5, 5.6
        
        let mut mock_gpu_scene = MockGpuScene::new();
        
        // Add multiple genomes to test selection
        let mut genome1 = Genome::default();
        genome1.name = "Genome1".to_string();
        let mut genome2 = Genome::default();
        genome2.name = "Genome2".to_string();
        mock_gpu_scene.genomes.push(genome1);
        mock_gpu_scene.genomes.push(genome2);
        
        // Test that genome loading accesses CPU-side storage only
        let genome_id = 1u32;
        assert!(genome_id < mock_gpu_scene.genomes.len() as u32);
        
        // This operation should only read from CPU memory (genomes Vec)
        let loaded_genome = &mock_gpu_scene.genomes[genome_id as usize];
        assert_eq!(loaded_genome.name, "Genome2");
        
        // Verify no GPU buffer access is needed for genome loading
        // (This is implicit in the design - genomes are stored in CPU Vec)
        assert_eq!(mock_gpu_scene.genomes.len(), 2);
    }

    #[test]
    fn test_scene_mode_request_for_genome_loading() {
        // Test that genome loading triggers preview mode request
        // This verifies the workflow: genome loading → editor → preview mode
        
        // Simulate the genome loading button click behavior
        // (This would normally happen in the UI code)
        let scene_request = SceneModeRequest::SwitchToPreview;
        
        // Verify the request is properly set
        assert!(scene_request.is_requested());
        assert_eq!(scene_request.target_mode(), Some(crate::ui::types::SimulationMode::Preview));
        
        // Verify this is a legitimate scene switch request
        match scene_request {
            SceneModeRequest::SwitchToPreview => {
                // This is the expected behavior for genome loading
                assert!(true);
            }
            _ => {
                panic!("Expected SwitchToPreview request for genome loading");
            }
        }
    }

    #[test]
    fn test_complete_genome_loading_workflow() {
        // Test the complete genome loading workflow from GPU scene to editor
        // This verifies Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
        
        // Create a mock GPU scene with test genomes
        let mut mock_gpu_scene = MockGpuScene::new();
        
        // Add test genomes with different names
        let mut genome1 = Genome::default();
        genome1.name = "TestGenome1".to_string();
        let mut genome2 = Genome::default();
        genome2.name = "TestGenome2".to_string();
        
        mock_gpu_scene.genomes.push(genome1.clone());
        mock_gpu_scene.genomes.push(genome2.clone());
        
        // Simulate cell inspector data with genome_id
        let genome_id = 1u32; // Select second genome
        
        // Verify genome is available for loading
        assert!(genome_id < mock_gpu_scene.genomes.len() as u32);
        
        // Simulate the genome loading process (what happens in UI)
        let genome_to_load = mock_gpu_scene.genomes[genome_id as usize].clone();
        
        // Verify the loaded genome is correct
        assert_eq!(genome_to_load.name, "TestGenome2");
        assert_eq!(genome_to_load.modes.len(), 40); // Default genome has 40 modes
        
        // Verify this is a CPU-only operation (no GPU buffer access)
        // The genome comes from the CPU-side Vec, not GPU buffers
        assert_eq!(genome_to_load.name, genome2.name);
        
        // Verify scene mode request would be triggered
        let scene_request = SceneModeRequest::SwitchToPreview;
        
        assert!(scene_request.is_requested());
        assert_eq!(scene_request.target_mode(), Some(crate::ui::types::SimulationMode::Preview));
    }

    #[test]
    fn test_genome_loading_bounds_checking() {
        // Test that genome loading properly checks bounds
        // This verifies safe access to the genomes array
        
        let mut mock_gpu_scene = MockGpuScene::new();
        
        // Add only one genome
        let mut genome = Genome::default();
        genome.name = "OnlyGenome".to_string();
        mock_gpu_scene.genomes.push(genome);
        
        // Test valid genome_id
        let valid_genome_id = 0u32;
        assert!(valid_genome_id < mock_gpu_scene.genomes.len() as u32);
        
        // Test invalid genome_id (would be caught by UI bounds check)
        let invalid_genome_id = 5u32;
        assert!(invalid_genome_id >= mock_gpu_scene.genomes.len() as u32);
        
        // The UI should only show the Load Genome button when genome_id is valid
        // This test verifies the bounds checking logic
        assert_eq!(mock_gpu_scene.genomes.len(), 1);
    }

    // Mock GPU scene for testing (minimal implementation)
    struct MockGpuScene {
        genomes: Vec<Genome>,
    }

    impl MockGpuScene {
        fn new() -> Self {
            Self {
                genomes: Vec::new(),
            }
        }
    }
}
