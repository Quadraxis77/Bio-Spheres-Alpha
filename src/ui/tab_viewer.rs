//! TabViewer implementation for the Bio-Spheres docking system.
//!
//! This module provides the PanelTabViewer which implements egui_dock's
//! TabViewer trait to render panel content and handle panel interactions.

use egui::{Id, Ui, WidgetText};
use egui_dock::tab_viewer::OnCloseResponse;
use egui_dock::{NodeIndex, SurfaceIndex, TabViewer};

use crate::field_report::{
    format_simulation_time, render_lineage_snapshot_report, render_specimen_report,
    ArchivedFieldReport, FieldReportHistory, FieldReportSeverity, LineageSnapshotReportSnapshot,
    RenderedFieldReport, SpecimenReportSnapshot, ToneProfile,
};
use crate::ui::panel::Panel;
use crate::ui::panel_context::{CaveAppearanceVisualSettings, PanelContext, SceneModeRequest};
use crate::ui::types::GlobalUiState;
use crate::ui::types::SimulationMode;
use crate::ui::ui_system::{
    average_fps_for_last_samples, draw_headless_fps_graph, fps_color, headless_metric, palette,
    stat_label,
};
use crate::ui::widgets::{modes_buttons, modes_list_items, quaternion_ball};

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
        // Render tab labels in uppercase with extra letter-spacing for the
        // biotech-console aesthetic. The viewport tab keeps its name hidden by
        // the dock anyway, so casing there is irrelevant.
        let raw = tab.display_name();
        let upper: String = raw
            .chars()
            .flat_map(|c| {
                // Insert a hair-space between characters for readability
                [Some(c.to_ascii_uppercase()), Some('\u{2009}')]
                    .into_iter()
                    .flatten()
            })
            .collect();
        // Trim the trailing hair-space
        let upper = upper.trim_end().to_string();
        egui::RichText::new(upper).size(11.0).into()
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
        // When hide_ui is active, only render the viewport - all other panels
        // are skipped entirely so their content doesn't appear on screen.
        if self.context.hide_ui {
            if matches!(tab, Panel::Viewport) {
                let is_gpu_mode =
                    self.context.current_mode == crate::ui::types::SimulationMode::Gpu;
                render_viewport(ui, self.viewport_rect, is_gpu_mode);
                if let Some(r) = *self.viewport_rect {
                    self.context
                        .editor_state
                        .panel_rects
                        .insert("Viewport".to_string(), r);
                }
            }
            return;
        }
        match tab {
            Panel::Viewport => {
                let is_gpu_mode =
                    self.context.current_mode == crate::ui::types::SimulationMode::Gpu;
                render_viewport(ui, self.viewport_rect, is_gpu_mode);
                if let Some(r) = *self.viewport_rect {
                    self.context
                        .editor_state
                        .panel_rects
                        .insert("Viewport".to_string(), r);
                }
            }
            Panel::LeftPanel => render_placeholder_panel(ui, "Left Panel"),
            Panel::RightPanel => render_placeholder_panel(ui, "Right Panel"),
            Panel::BottomPanel => render_placeholder_panel(ui, "Bottom Panel"),
            Panel::SceneManager => render_scene_manager(ui, self.context, self.state),
            Panel::CellInspector => render_cell_inspector(ui, self.context, self.state),
            Panel::LineageViewer => render_lineage_viewer(ui, self.context, self.state),
            Panel::GenomeEditor => render_genome_editor(ui, self.context),
            Panel::PerformanceMonitor => render_performance_monitor(ui, self.context, self.state),
            Panel::RenderingControls => render_rendering_controls(ui),
            Panel::CaveSystem => render_cave_system(ui, self.context, self.state),
            Panel::FluidSettings => {
                render_fluid_settings(ui, self.context, self.state);
                render_organism_skin_settings(
                    ui,
                    self.context,
                    &mut self.state.fluid_settings.organism_skin,
                );
            }
            Panel::WorldSettings => render_world_settings(ui, self.context, self.state),
            Panel::LightSettings => render_light_settings_organized(ui, self.context, self.state),
            Panel::TimeScrubber => render_time_scrubber(ui, self.context),
            Panel::ThemeEditor => render_theme_editor(ui, self.state),
            Panel::CameraSettings => render_camera_settings(ui, self.context, self.state),
            Panel::LightingSettings => render_lighting_settings(ui),
            Panel::GizmoSettings => render_gizmo_settings(ui, self.context),
            Panel::CellTypeVisuals => render_cell_type_visuals(ui, self.context),
            Panel::Modes => render_modes(ui, self.context),
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
        if matches!(tab, Panel::LineageViewer) {
            self.context.set_simulation_speed(1.0);
            return OnCloseResponse::Close;
        }

        // Don't close immediately - show a confirmation dialog first.
        // Store which panel is pending and return Focus to keep it open.
        self.state.pending_close_panel = Some(*tab);
        OnCloseResponse::Focus
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
/// This panel displays the 3D scene and reports its screen rectangle for
/// mouse input filtering. In GPU (live simulation) mode it draws large
/// L-shaped teal corner brackets so the viewport reads as a "scope" framing
/// the live world. The Genome Editor preview keeps a clean, frameless view.
/// Render the Viewport panel.
///
/// Just tracks the panel's screen rectangle so the 3D scene can be rendered
/// behind egui in the same area, and so input filtering and the GPU-mode
/// corner brackets (painted by `UiSystem` directly on the central panel)
/// know where the viewport lives.
fn render_viewport(ui: &mut Ui, viewport_rect: &mut Option<egui::Rect>, _draw_brackets: bool) {
    let rect = ui.available_rect_before_wrap();
    *viewport_rect = Some(rect);

    // Allocate the full space - the 3D scene renders behind this
    let _response = ui.allocate_rect(rect, egui::Sense::hover());
}

/// Render the SceneManager panel.
fn render_scene_manager(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    // Record panel rect for the tutorial pointer.
    context
        .editor_state
        .panel_rects
        .insert("SceneManager".to_string(), ui.max_rect());

    // The mode-switch button lives in the top bar - this panel only hosts
    // playback controls. In Preview mode there's nothing to do here.
    if !context.is_gpu_mode() {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Use ▶ LIVE SIMULATION in the top bar to start the live world.")
                .size(11.0)
                .color(palette().text_secondary),
        );
        return;
    }

    // -- GPU mode: simulation playback controls ------------------------------
    section_header(ui, "SIMULATION CONTROLS");

    ui.horizontal(|ui| {
        let is_paused = context.is_paused();
        let (play_icon, play_tip, play_color) = if is_paused {
            ("▶", "Play", palette().accent_primary)
        } else {
            ("⏸", "Pause", palette().accent_primary)
        };

        if ui
            .add_sized(
                [32.0, 32.0],
                egui::Button::new(egui::RichText::new(play_icon).size(15.0).color(play_color))
                    .fill(palette().bg_widget)
                    .stroke(egui::Stroke::new(1.0, palette().border_normal))
                    .corner_radius(egui::CornerRadius::same(3)),
            )
            .on_hover_text(play_tip)
            .clicked()
        {
            context.request_toggle_pause();
        }

        if ui
            .add_sized(
                [32.0, 32.0],
                egui::Button::new(
                    egui::RichText::new("⟲")
                        .size(15.0)
                        .color(palette().text_secondary),
                )
                .fill(palette().bg_widget)
                .stroke(egui::Stroke::new(1.0, palette().border_normal))
                .corner_radius(egui::CornerRadius::same(3)),
            )
            .on_hover_text("Reset")
            .clicked()
        {
            state.show_reset_dialog = true;
        }

        ui.add_space(8.0);

        // Simulation time
        let total_seconds = context.current_time() as u64;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        ui.label(
            egui::RichText::new(format!("{:02}:{:02}:{:02}", hours, minutes, seconds))
                .size(13.0)
                .color(palette().text_primary)
                .monospace(),
        );
    });

    ui.add_space(6.0);

    // Speed slider
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("SPEED")
                .size(9.0)
                .color(palette().text_dim),
        );
        let speed = context.simulation_speed();
        let display_speed = (speed * 2.0).round() / 2.0;
        let mut slider_speed = display_speed;
        if ui
            .add(
                egui::Slider::new(&mut slider_speed, 0.5..=10.0)
                    .step_by(0.5)
                    .suffix("x")
                    .text_color(palette().text_primary),
            )
            .changed()
        {
            context.set_simulation_speed(slider_speed);
        }
    });

    ui.add_space(4.0);
}

/// Render the CellInspector panel.
fn render_cell_inspector(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    // -- Header row ----------------------------------------------------------
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("CELL INSPECTOR")
                .strong()
                .size(11.5)
                .color(palette().accent_primary),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let count_str = if let Some(n) = context.gpu_cell_count() {
                format!("{} cells", n)
            } else {
                format!("{} cells", context.cell_count())
            };
            ui.label(
                egui::RichText::new(count_str)
                    .size(10.0)
                    .color(palette().text_dim),
            );
        });
    });
    ui.add_space(2.0);
    ui.add(egui::Separator::default().spacing(4.0));

    // Get the inspected cell index from radial menu state
    let inspected_cell = context.editor_state.radial_menu.inspected_cell;

    if let Some(cell_idx) = inspected_cell {
        // Extract data from GPU scene
        let gpu_extraction_data = if let Some(gpu_scene) = context.scene_manager.gpu_scene() {
            let is_extracting = gpu_scene.is_extracting_cell_data();
            if let Some(extraction_result) = gpu_scene.get_latest_cell_extraction() {
                if extraction_result.cell_index == cell_idx as u32 {
                    let data_valid = extraction_result.data.is_valid();
                    Some((is_extracting, Some(extraction_result), Some(data_valid)))
                } else {
                    // Have a result but for a different cell - show spinner until ours arrives
                    Some((true, None, None))
                }
            } else {
                // No result yet at all - show spinner
                Some((true, None, None))
            }
        } else {
            None
        };

        // Now use the extracted data without borrowing context
        if let Some((_is_extracting, extraction_result, data_valid)) = gpu_extraction_data {
            // While a new extraction is in flight we still show the last known data -
            // no spinner needed since updates are continuous and near-instant.

            if let (Some(extraction_result), Some(data_valid)) = (extraction_result, data_valid) {
                let data = extraction_result.data;

                // -- Invalid / dead cell --------------------------------------
                if !data_valid {
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("⚠ Cell no longer exists")
                                .size(12.0)
                                .color(palette().status_err),
                        );
                    });
                    ui.add_space(6.0);
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Clear Selection")
                                    .size(11.0)
                                    .color(palette().text_primary),
                            )
                            .fill(palette().bg_widget)
                            .stroke(egui::Stroke::new(1.0, palette().border_normal)),
                        )
                        .clicked()
                    {
                        context.editor_state.radial_menu.inspected_cell = None;
                    }
                    return;
                }

                // -- Cell type & identity -------------------------------------
                let cell_type_names = crate::cell::types::CellType::names();
                let type_name = cell_type_names
                    .get(data.cell_type as usize)
                    .copied()
                    .unwrap_or("Unknown");

                ui.add_space(6.0);

                // Title row: type name + cell index
                ui.horizontal(|ui| {
                    let title_color = if data.is_dead != 0 {
                        palette().status_err
                    } else {
                        palette().text_primary
                    };
                    let dead_suffix = if data.is_dead != 0 { " — DEAD" } else { "" };
                    ui.label(
                        egui::RichText::new(format!("{}{}", type_name, dead_suffix))
                            .strong()
                            .size(14.0)
                            .color(title_color),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("#{}", cell_idx))
                                .size(11.0)
                                .color(palette().text_dim),
                        );
                    });
                });

                // Identity sub-row
                let org_str = if data.organism_id == u32::MAX {
                    "isolated".to_string()
                } else {
                    format!("org {}", data.organism_id)
                };
                let genome_str = context
                    .scene_manager
                    .gpu_scene()
                    .map(|gpu_scene| {
                        if gpu_scene
                            .reported_genome_matches_absolute_mode(data.genome_id, data.mode_index)
                        {
                            format!("genome {}", data.genome_id)
                        } else {
                            format!("mutated GPU genome (reported {})", data.genome_id)
                        }
                    })
                    .unwrap_or_else(|| format!("genome {}", data.genome_id));
                let mode_str = context
                    .scene_manager
                    .gpu_scene()
                    .map(|gpu_scene| {
                        gpu_scene.inspected_mode_label(data.genome_id, data.mode_index)
                    })
                    .unwrap_or_else(|| format!("M{}", data.mode_index));
                ui.label(
                    egui::RichText::new(format!(
                        "{}  ·  {}  ·  slot {}  ·  {}",
                        mode_str, genome_str, data.cell_slot_index, org_str
                    ))
                    .size(10.0)
                    .color(palette().text_dim),
                );

                ui.add_space(6.0);

                // -- Action buttons -------------------------------------------
                ui.horizontal(|ui| {
                    if ui.add(
                        egui::Button::new(
                            egui::RichText::new("📋 Load Genome").size(11.0).color(palette().text_primary),
                        )
                        .fill(palette().bg_widget)
                        .stroke(egui::Stroke::new(1.0, palette().border_normal)),
                    ).clicked() {
                        *context.scene_request =
                            crate::ui::panel_context::SceneModeRequest::LoadGenomeFromGpuCell {
                                genome_id: data.genome_id,
                                mode_index: data.mode_index,
                            };
                        log::info!(
                            "Requested genome readback for inspected cell: genome_id={} mode_index={}",
                            data.genome_id,
                            data.mode_index
                        );
                    }
                    if ui.add(
                        egui::Button::new(
                            egui::RichText::new("✕ Deselect").size(11.0).color(palette().text_secondary),
                        )
                        .fill(palette().bg_widget)
                        .stroke(egui::Stroke::new(1.0, palette().border_subtle)),
                    ).clicked() {
                        context.editor_state.radial_menu.inspected_cell = None;
                    }
                });

                ui.add_space(8.0);

                let lineage_context = context.scene_manager.gpu_scene().and_then(|scene| {
                    scene
                        .lineage_archive
                        .lineage_for_genome_id(data.genome_id)
                        .and_then(|lineage_id| {
                            scene
                                .lineage_archive
                                .nodes
                                .iter()
                                .find(|node| node.id == lineage_id)
                                .map(|node| (lineage_id, node.display_name.clone()))
                        })
                });
                let specimen_snapshot = SpecimenReportSnapshot {
                    cell_id: data.cell_id,
                    cell_type_name: type_name.to_string(),
                    lineage_id: lineage_context.as_ref().map(|(id, _)| *id),
                    lineage_name: lineage_context.as_ref().map(|(_, name)| name.clone()),
                    organism_id: (data.organism_id != u32::MAX).then_some(data.organism_id),
                    alive: data.is_dead == 0,
                    nutrient_level: data.nutrients,
                    nutrient_gain_rate: data.nutrient_gain_rate,
                    thermal_state: data.cell_thermal_state,
                    adhesion_count: data.adhesion_count,
                    active_signal_channels: (0..16)
                        .filter(|&channel| data.signal_strength(channel) > 0)
                        .count() as u32,
                };
                if let Some(report) = render_specimen_report(
                    &specimen_snapshot,
                    &ToneProfile::naturalist_field_journal(),
                ) {
                    render_field_report_card(ui, &report, Some("SPECIMEN REPORT"));
                }
                if let Some(lineage_id) = specimen_snapshot.lineage_id {
                    if let Some(related) = context
                        .field_reports
                        .history
                        .reports
                        .iter()
                        .rev()
                        .find(|report| report.rendered.involved_lineages.contains(&lineage_id))
                    {
                        ui.label(
                            egui::RichText::new(format!(
                                "Related lineage report: {}",
                                related.rendered.title
                            ))
                            .size(9.5)
                            .color(palette().text_dim),
                        );
                        ui.add_space(4.0);
                    }
                }

                // -- NUTRIENTS section ----------------------------------------
                section_header(ui, "NUTRIENTS");

                // Nutrient bar - same style as preview context menu
                let split_never = data.max_splits == 0 && data.nutrient_threshold <= 0.0;
                let nutrient_max = if split_never || data.nutrient_threshold <= 0.0 {
                    100.0_f32
                } else {
                    data.nutrient_threshold * 2.0
                };
                let nutrient_frac = (data.nutrients / nutrient_max).clamp(0.0, 1.0);
                let bar_width = ui.available_width() - 4.0;
                let bar_height = 8.0;
                let (bar_rect, _) =
                    ui.allocate_exact_size(egui::vec2(bar_width, bar_height), egui::Sense::hover());
                ui.painter()
                    .rect_filled(bar_rect, 3.0, palette().bg_darkest);
                let fill_color = if nutrient_frac > 0.5 {
                    palette().status_ok
                } else if nutrient_frac > 0.2 {
                    palette().status_warn
                } else {
                    palette().status_err
                };
                let fill_rect = egui::Rect::from_min_size(
                    bar_rect.min,
                    egui::vec2(bar_rect.width() * nutrient_frac, bar_height),
                );
                ui.painter().rect_filled(fill_rect, 3.0, fill_color);
                // Split threshold marker
                if !split_never && data.nutrient_threshold > 0.0 {
                    let split_frac = (data.nutrient_threshold / nutrient_max).clamp(0.0, 1.0);
                    let marker_x = bar_rect.min.x + bar_rect.width() * split_frac;
                    ui.painter().line_segment(
                        [
                            egui::pos2(marker_x, bar_rect.min.y),
                            egui::pos2(marker_x, bar_rect.max.y),
                        ],
                        egui::Stroke::new(1.5, palette().text_primary),
                    );
                }
                ui.label(
                    egui::RichText::new(format!("{:.0} / {:.0}", data.nutrients, nutrient_max))
                        .size(10.0)
                        .color(palette().text_dim),
                );

                ui.add_space(4.0);

                egui::Grid::new("inspector_nutrients_grid")
                    .num_columns(2)
                    .spacing([8.0, 2.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Net rate")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        let gain_color = if data.nutrient_gain_rate > 0.01 {
                            palette().status_ok
                        } else if data.nutrient_gain_rate < -0.01 {
                            palette().status_err
                        } else {
                            palette().text_dim
                        };
                        ui.label(
                            egui::RichText::new(format!("{:+.2}/s", data.nutrient_gain_rate))
                                .size(11.0)
                                .color(gain_color),
                        );
                        ui.end_row();

                        if data.reserve > 0 {
                            ui.label(
                                egui::RichText::new("Reserve")
                                    .size(11.0)
                                    .color(palette().text_secondary),
                            );
                            ui.label(
                                egui::RichText::new(format!("{}", data.reserve / 1000))
                                    .size(11.0)
                                    .color(palette().accent_secondary),
                            );
                            ui.end_row();
                        }

                        ui.label(
                            egui::RichText::new("Split at")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        if split_never || data.nutrient_threshold <= 0.0 {
                            ui.label(
                                egui::RichText::new("Never")
                                    .size(11.0)
                                    .color(palette().text_dim),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.nutrient_threshold))
                                    .size(11.0)
                                    .color(palette().text_primary),
                            );
                        }
                        ui.end_row();

                        let max_splits_str = if data.max_splits == 0 {
                            "∞".to_string()
                        } else {
                            format!("{}", data.max_splits)
                        };
                        ui.label(
                            egui::RichText::new("Splits")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{} / {}",
                                data.split_count, max_splits_str
                            ))
                            .size(11.0)
                            .color(palette().text_primary),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Interval")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.1}s", data.split_interval))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Age")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.1}s", data.age))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();
                    });

                ui.add_space(4.0);

                // -- SIGNALS section -----------------------------------------
                section_header(ui, "SIGNALS");

                egui::Grid::new("inspector_signals_grid")
                    .num_columns(4)
                    .spacing([8.0, 2.0])
                    .show(ui, |ui| {
                        for row in 0..8 {
                            for channel in [row, row + 8] {
                                let strength = data.signal_strength(channel);
                                let is_source = data.signal_is_source(channel);
                                let value_color = if is_source {
                                    palette().accent_secondary
                                } else if strength > 0 {
                                    palette().status_warn
                                } else {
                                    palette().text_dim
                                };

                                ui.label(
                                    egui::RichText::new(format!("Ch {channel}"))
                                        .size(10.0)
                                        .color(palette().text_secondary),
                                );
                                let value = if is_source {
                                    format!("{strength}  emit")
                                } else {
                                    strength.to_string()
                                };
                                ui.label(
                                    egui::RichText::new(value)
                                        .size(10.0)
                                        .monospace()
                                        .color(value_color),
                                );
                            }
                            ui.end_row();
                        }
                    });

                ui.label(
                    egui::RichText::new(
                        "emit = direct source · other values are received strength",
                    )
                    .size(9.0)
                    .color(palette().text_dim),
                );

                ui.add_space(4.0);

                // -- PHYSIOLOGY section --------------------------------------
                section_header(ui, "PHYSIOLOGY");

                let thermal_state_label = match data.cell_thermal_state {
                    0 => "Deep frozen",
                    1 => "Frozen",
                    2 => "Chilled",
                    3 => "Cool",
                    4 => "Ideal",
                    5 => "Warm",
                    6 => "Hot safe",
                    7 => "Overheated",
                    8 => "Heat shock",
                    9 => "Critical",
                    _ => "Unknown",
                };
                let thermal_color = match data.cell_thermal_state {
                    0 | 1 | 8 | 9 => palette().status_err,
                    2 | 7 => palette().status_warn,
                    3 | 4 | 5 | 6 => palette().status_ok,
                    _ => palette().text_dim,
                };
                let cell_temp_c = data.cell_cached_temperature / 255.0 * 200.0 - 50.0;

                egui::Grid::new("inspector_physiology_grid")
                    .num_columns(2)
                    .spacing([8.0, 2.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Water reserve")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.3}", data.cell_water))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Body temp")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.1} C", cell_temp_c))
                                .size(11.0)
                                .color(thermal_color),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Thermal comfort")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(thermal_state_label)
                                .size(11.0)
                                .color(thermal_color),
                        );
                        ui.end_row();
                    });

                ui.add_space(4.0);

                // -- PHYSICS section ------------------------------------------
                section_header(ui, "PHYSICS");

                let pos = data.position_vec3();
                let vel = data.velocity_vec3();
                let speed = vel.length();

                egui::Grid::new("inspector_physics_grid")
                    .num_columns(2)
                    .spacing([8.0, 2.0])
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Position")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.1}, {:.1}, {:.1}",
                                pos.x, pos.y, pos.z
                            ))
                            .size(11.0)
                            .color(palette().text_primary)
                            .monospace(),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Velocity")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.2}, {:.2}, {:.2}",
                                vel.x, vel.y, vel.z
                            ))
                            .size(11.0)
                            .color(palette().text_primary)
                            .monospace(),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Speed")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        let speed_color = if speed > 5.0 {
                            palette().status_warn
                        } else {
                            palette().text_primary
                        };
                        ui.label(
                            egui::RichText::new(format!("{:.3}", speed))
                                .size(11.0)
                                .color(speed_color),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Mass")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.2}", data.mass))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Radius")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.2}", data.radius))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Max size")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.2}", data.max_cell_size))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("Stiffness")
                                .size(11.0)
                                .color(palette().text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.2}", data.stiffness))
                                .size(11.0)
                                .color(palette().text_primary),
                        );
                        ui.end_row();
                    });

                ui.add_space(4.0);

                // -- ADHESIONS section ----------------------------------------
                section_header(ui, "ADHESIONS");

                let adhesion_color = if data.adhesion_count == 0 {
                    palette().text_dim
                } else {
                    palette().accent_secondary
                };
                ui.label(
                    egui::RichText::new(format!(
                        "{} active bond{}",
                        data.adhesion_count,
                        if data.adhesion_count == 1 { "" } else { "s" }
                    ))
                    .size(12.0)
                    .color(adhesion_color),
                );

                return;
            } else {
                // First frame - extraction not yet complete, show a brief spinner
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(format!("Reading cell #{}…", cell_idx))
                            .size(11.0)
                            .color(palette().text_secondary),
                    );
                });
                return;
            }
        }
    } else {
        if context.is_gpu_mode() {
            render_field_station_feed(ui, context, state);
            return;
        }
        // -- Empty state ------------------------------------------------------
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new("🔍").size(28.0));
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("No cell selected")
                    .size(13.0)
                    .color(palette().text_secondary),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Hold Alt → select Inspect → click a cell")
                    .size(10.0)
                    .color(palette().text_dim),
            );
        });
    }
}

fn render_field_station_feed(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        section_header(ui, "FIELD STATION REPORT");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let label = if state.field_reports_enabled {
                "Disable Reports"
            } else {
                "Enable Reports"
            };
            if ui
                .button(label)
                .on_hover_text("Stop or resume scheduled ecosystem field-report generation")
                .clicked()
            {
                state.field_reports_enabled = !state.field_reports_enabled;
            }
        });
    });
    let scheduled_status = if state.field_reports_enabled {
        field_report_schedule_status(context)
    } else {
        "Scheduled field reports are disabled.".to_string()
    };
    if let Some(latest) = context.field_reports.history.reports.back() {
        render_field_report_card(ui, &latest.rendered, None);
        let recent: Vec<_> = context
            .field_reports
            .history
            .reports
            .iter()
            .rev()
            .skip(1)
            .take(4)
            .collect();
        if !recent.is_empty() {
            ui.label(
                egui::RichText::new("RECENT OBSERVATIONS")
                    .size(9.5)
                    .color(palette().accent_primary),
            );
            for report in recent {
                render_recent_report_row(ui, report);
            }
        }
    } else {
        egui::Frame::new()
            .fill(palette().bg_widget)
            .stroke(egui::Stroke::new(1.0, palette().border_subtle))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Awaiting a report-grade ecosystem scan.")
                        .size(11.0)
                        .color(palette().text_secondary),
                );
            });
    }
    ui.label(
        egui::RichText::new(scheduled_status)
            .size(9.5)
            .color(palette().text_dim),
    );
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new("Select a cell to switch to specimen reporting.")
            .size(9.5)
            .color(palette().text_dim),
    );
}

fn field_report_schedule_status(context: &PanelContext) -> String {
    context
        .scene_manager
        .gpu_scene()
        .map(|scene| {
            let interval = scene.lineage_capture_interval_seconds.max(1.0);
            let elapsed = scene
                .lineage_archive
                .last_scan_frame
                .map(|_| scene.current_time - scene.lineage_archive.last_scan_time)
                .unwrap_or(scene.current_time);
            let remaining = (interval - elapsed).max(0.0);
            if scene.lineage_archive.last_scan_frame.is_some() {
                format!("Next scheduled report scan in {:.0}s.", remaining.ceil())
            } else {
                format!("First scheduled report scan in {:.0}s.", remaining.ceil())
            }
        })
        .unwrap_or_else(|| "Scheduled reporting begins in GPU biosphere mode.".to_string())
}

fn render_recent_report_row(ui: &mut Ui, report: &ArchivedFieldReport) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            egui::RichText::new("•")
                .size(10.0)
                .color(severity_color(report.rendered.severity)),
        );
        ui.label(
            egui::RichText::new(&report.rendered.title)
                .size(10.0)
                .color(palette().text_primary),
        );
        if let Some(first) = report.rendered.sentences.first() {
            ui.label(
                egui::RichText::new(format!("— {}", first.text))
                    .size(9.5)
                    .color(palette().text_secondary),
            );
        }
    });
}

fn render_field_report_card(ui: &mut Ui, report: &RenderedFieldReport, scope_label: Option<&str>) {
    let accent = severity_color(report.severity);
    egui::Frame::new()
        .fill(palette().bg_panel)
        .stroke(egui::Stroke::new(1.0, accent.linear_multiply(0.75)))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    if let Some(scope) = scope_label {
                        ui.label(
                            egui::RichText::new(scope)
                                .size(8.5)
                                .color(palette().text_dim),
                        );
                    }
                    ui.label(
                        egui::RichText::new(&report.title)
                            .size(14.0)
                            .strong()
                            .color(palette().text_primary),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    ui.label(
                        egui::RichText::new(severity_label(report.severity))
                            .size(8.5)
                            .strong()
                            .color(accent),
                    );
                });
            });
            ui.add_space(5.0);
            ui.label(
                egui::RichText::new(&report.body)
                    .size(10.5)
                    .color(palette().text_secondary),
            );
            if !report.tags.is_empty() {
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    for tag in &report.tags {
                        ui.label(
                            egui::RichText::new(format!("{tag:?}"))
                                .size(8.5)
                                .color(accent),
                        );
                    }
                });
            }
        });
    ui.add_space(6.0);
}

fn severity_label(severity: FieldReportSeverity) -> &'static str {
    match severity {
        FieldReportSeverity::Routine => "INFO",
        FieldReportSeverity::Notable => "NOTABLE",
        FieldReportSeverity::Warning => "WARNING",
        FieldReportSeverity::Critical => "CRITICAL",
    }
}

fn severity_color(severity: FieldReportSeverity) -> egui::Color32 {
    match severity {
        FieldReportSeverity::Routine => palette().status_info,
        FieldReportSeverity::Notable => palette().accent_secondary,
        FieldReportSeverity::Warning => palette().status_warn,
        FieldReportSeverity::Critical => palette().status_err,
    }
}

fn render_headless_section(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    state.gpu_headless_mode = true;

    let p = palette();
    let headless_speed_cap = crate::ui::types::GPU_HEADLESS_MAX_SIM_SPEED;

    let (mut sim_speed, is_paused, sim_time, cell_count, capacity) =
        if let Some(gpu_scene) = context.scene_manager.gpu_scene() {
            (
                gpu_scene.time_scale,
                gpu_scene.paused,
                gpu_scene.current_time,
                gpu_scene.current_cell_count,
                gpu_scene.capacity(),
            )
        } else {
            (1.0, true, 0.0, 0, 0)
        };

    let fps = context.performance.fps();
    let frame_ms = context.performance.average_frame_time_ms();
    let min_ms = context.performance.min_frame_time_ms();
    let max_ms = context.performance.max_frame_time_ms();
    let frame_times: Vec<f32> = context.performance.frame_time_history().collect();
    let avg_fps = average_fps_for_last_samples(&frame_times, 60);
    let long_avg_fps = average_fps_for_last_samples(&frame_times, 120);

    let header_frame = egui::Frame::new()
        .fill(p.bg_darkest)
        .stroke(egui::Stroke::new(1.0, p.border_bright))
        .inner_margin(egui::Margin::symmetric(12, 10));

    header_frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("HEADLESS PERFORMANCE")
                    .size(10.0)
                    .color(p.text_dim),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("RENDERING DISABLED")
                    .size(10.0)
                    .color(p.status_ok),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(
                        egui::RichText::new(if is_paused { "Resume" } else { "Pause" }).size(11.0),
                    )
                    .clicked()
                {
                    *context.scene_request = SceneModeRequest::TogglePause;
                }
            });
        });

        ui.add_space(8.0);
        ui.columns(4, |cols| {
            headless_metric(
                &mut cols[0],
                "FPS",
                &format!("{:.1}", fps),
                fps_color(fps, p),
            );
            headless_metric(
                &mut cols[1],
                "Frame",
                &format!("{:.2} ms", frame_ms),
                p.text_primary,
            );
            headless_metric(
                &mut cols[2],
                "Speed",
                &format!("{:.2}x", sim_speed),
                p.accent_primary,
            );
            headless_metric(
                &mut cols[3],
                "Cells",
                &format!("{} / {}k", cell_count, capacity / 1000),
                p.text_primary,
            );
        });

        ui.add_space(8.0);
        draw_headless_fps_graph(ui, &frame_times, 120.0, state.gpu_headless_target_fps);

        ui.add_space(6.0);
        egui::Grid::new("lineage_headless_stats_grid")
            .num_columns(4)
            .spacing([18.0, 4.0])
            .show(ui, |ui| {
                stat_label(ui, "1s avg", &format!("{:.1} FPS", avg_fps));
                stat_label(ui, "2s avg", &format!("{:.1} FPS", long_avg_fps));
                stat_label(ui, "min/max", &format!("{:.2}/{:.2} ms", min_ms, max_ms));
                stat_label(ui, "sim time", &format!("{:.1}s", sim_time));
                ui.end_row();
            });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Simulation speed")
                    .size(12.0)
                    .color(p.text_secondary),
            );
            sim_speed = sim_speed.clamp(0.1, headless_speed_cap);
            let speed_resp = ui.add(
                egui::Slider::new(&mut sim_speed, 0.1..=headless_speed_cap)
                    .logarithmic(true)
                    .suffix("x"),
            );
            if speed_resp.changed() {
                state.gpu_headless_auto_speed = false;
                *context.scene_request = SceneModeRequest::SetSpeed(sim_speed);
            }
        });

        ui.horizontal(|ui| {
            for speed in [1.0_f32, 2.0, 5.0, 10.0] {
                if ui.button(format!("{:.0}x", speed)).clicked() {
                    state.gpu_headless_auto_speed = false;
                    *context.scene_request = SceneModeRequest::SetSpeed(speed);
                }
            }
            ui.add_space(6.0);
            if ui.button("Adaptive").clicked() {
                state.gpu_headless_auto_speed = true;
                state.gpu_headless_target_fps = 30.0;
            }
            if ui.button("Fast Forward").clicked() {
                state.gpu_headless_auto_speed = true;
                state.gpu_headless_target_fps = 30.0;
                *context.scene_request = SceneModeRequest::SetSpeed(
                    state.gpu_headless_max_speed.min(headless_speed_cap),
                );
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.checkbox(&mut state.gpu_headless_auto_speed, "Auto target FPS");
            ui.add(
                egui::Slider::new(&mut state.gpu_headless_target_fps, 15.0..=60.0).suffix(" FPS"),
            );
            ui.label(
                egui::RichText::new(state.gpu_headless_auto_status.display_name()).color(
                    if state.gpu_headless_auto_speed {
                        p.accent_secondary
                    } else {
                        p.text_dim
                    },
                ),
            );
        });

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Auto bounds")
                    .size(12.0)
                    .color(p.text_secondary),
            );
            ui.add(
                egui::Slider::new(&mut state.gpu_headless_min_speed, 0.1..=headless_speed_cap)
                    .logarithmic(true)
                    .prefix("min ")
                    .suffix("x"),
            );
            ui.add(
                egui::Slider::new(&mut state.gpu_headless_max_speed, 0.1..=headless_speed_cap)
                    .logarithmic(true)
                    .prefix("max ")
                    .suffix("x"),
            );
            state.gpu_headless_min_speed =
                state.gpu_headless_min_speed.clamp(0.1, headless_speed_cap);
            state.gpu_headless_max_speed =
                state.gpu_headless_max_speed.clamp(0.1, headless_speed_cap);
            if state.gpu_headless_min_speed > state.gpu_headless_max_speed {
                std::mem::swap(
                    &mut state.gpu_headless_min_speed,
                    &mut state.gpu_headless_max_speed,
                );
            }
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut state.gpu_readbacks_enabled, "GPU readbacks");
            ui.label(
                egui::RichText::new(if state.gpu_readbacks_enabled {
                    "live counts enabled"
                } else {
                    "reduced CPU/GPU synchronization"
                })
                .color(p.text_dim),
            );
        });
    });
}

fn render_lineage_viewer(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    if !context.is_gpu_mode() {
        section_header(ui, "SPECIES DOSSIER");
        ui.label(
            egui::RichText::new("Lineage records are available in GPU biosphere mode.")
                .size(12.0)
                .color(palette().text_secondary),
        );
        return;
    }

    render_headless_section(ui, context, state);
    ui.add_space(8.0);

    let p = palette();
    let needs_initial_scan = context
        .scene_manager
        .gpu_scene()
        .map(|scene| scene.lineage_archive.last_scan_frame.is_none())
        .unwrap_or(false);
    if needs_initial_scan && matches!(context.scene_request, SceneModeRequest::None) {
        *context.scene_request = SceneModeRequest::ScanLineageForViewer;
    }

    // --- Snapshot interval control (mutable borrow of scene_manager must
    //     finish before the immutable borrow below) --------------------------
    {
        let p = palette();
        egui::Frame::new()
            .fill(p.bg_panel)
            .stroke(egui::Stroke::new(1.0, p.border_subtle))
            .inner_margin(egui::Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("SCAN INTERVAL")
                            .size(10.0)
                            .color(p.accent_primary),
                    );
                    ui.add_space(8.0);
                    if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                        let resp = ui.add(
                            egui::Slider::new(
                                &mut state.field_report_interval_seconds,
                                5.0_f32..=600.0,
                            )
                            .logarithmic(true)
                            .suffix("s"),
                        );
                        if resp.changed() {
                            gpu_scene.lineage_capture_interval_seconds =
                                state.field_report_interval_seconds;
                        }
                        ui.add_space(8.0);
                        let current = state.field_report_interval_seconds;
                        for preset in [15.0_f32, 30.0, 60.0, 120.0, 300.0] {
                            let label = if preset < 60.0 {
                                format!("{:.0}s", preset)
                            } else {
                                format!("{:.0}m", preset / 60.0)
                            };
                            let active = (current - preset).abs() < 0.5;
                            let btn = egui::Button::new(egui::RichText::new(label).size(10.5))
                                .fill(if active {
                                    p.accent_primary.linear_multiply(0.3)
                                } else {
                                    p.bg_widget
                                });
                            if ui.add(btn).clicked() {
                                state.field_report_interval_seconds = preset;
                                gpu_scene.lineage_capture_interval_seconds = preset;
                            }
                        }
                    }
                });
            });
        ui.add_space(6.0);
    }

    let Some(scene) = context.scene_manager.gpu_scene() else {
        ui.label(
            egui::RichText::new("No GPU scene is loaded.")
                .size(12.0)
                .color(palette().text_secondary),
        );
        return;
    };

    let report_history = &context.field_reports.history;
    let archive = &scene.lineage_archive;
    let mut nodes: Vec<_> = archive.nodes.iter().collect();
    nodes.sort_by(|a, b| {
        b.current_cells
            .cmp(&a.current_cells)
            .then_with(|| b.peak_cells.cmp(&a.peak_cells))
            .then_with(|| b.noteworthy_score.total_cmp(&a.noteworthy_score))
            .then_with(|| a.first_frame.cmp(&b.first_frame))
    });

    let selected_id_key = egui::Id::new("lineage_viewer_selected_id");
    let selected_snap_key = egui::Id::new("lineage_viewer_selected_snap_frame");
    let mut selected_lineage_id = ui.ctx().data(|data| data.get_temp::<u64>(selected_id_key));
    if selected_lineage_id
        .map(|id| archive.nodes.iter().any(|node| node.id == id))
        .unwrap_or(false)
        == false
    {
        selected_lineage_id = nodes.first().map(|node| node.id);
        if let Some(id) = selected_lineage_id {
            ui.ctx()
                .data_mut(|data| data.insert_temp(selected_id_key, id));
        }
    }
    let mut selected_snap_frame: Option<i32> =
        ui.ctx().data(|data| data.get_temp(selected_snap_key));
    let selected_node =
        selected_lineage_id.and_then(|id| archive.nodes.iter().find(|node| node.id == id));

    let living = archive
        .nodes
        .iter()
        .filter(|node| node.current_cells > 0)
        .count();
    let extinct = archive
        .nodes
        .iter()
        .filter(|node| node.extinct_frame.is_some())
        .count();
    let hybrids = archive
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.origin,
                crate::scene::lineage::LineageOrigin::Hybrid { .. }
            )
        })
        .count();

    let scan_text = if archive.last_scan_frame.is_some() {
        format!(
            "SCAN LOCKED / {}",
            format_simulation_time(archive.last_scan_time as f64)
        )
    } else {
        "SCAN PENDING ON PANEL OPEN".to_string()
    };

    let header = egui::Frame::new()
        .fill(p.bg_darkest)
        .stroke(egui::Stroke::new(1.0, p.border_bright))
        .inner_margin(egui::Margin::symmetric(12, 10));
    header.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new("ECOSYSTEM LINEAGE ARCHIVE")
                        .size(10.0)
                        .color(p.text_dim),
                );
                ui.label(
                    egui::RichText::new(
                        selected_node
                            .map(|node| node.display_name.as_str())
                            .unwrap_or("NO SPECIES SELECTED"),
                    )
                    .size(18.0)
                    .color(p.text_primary)
                    .strong(),
                );
                ui.label(
                    egui::RichText::new(scan_text)
                        .size(10.0)
                        .color(p.accent_secondary),
                );
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(egui::RichText::new("Refresh Scan").size(11.0))
                    .on_hover_text("Capture one frozen population scan for this viewer")
                    .clicked()
                {
                    *context.scene_request = SceneModeRequest::ScanLineageForViewer;
                }
            });
        });

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            lineage_metric_chip(
                ui,
                "Live Cells",
                archive.last_scan_live_cells.to_string(),
                p.status_ok,
            );
            lineage_metric_chip(
                ui,
                "Tracked",
                archive.last_scan_tracked_cells.to_string(),
                p.status_info,
            );
            if archive.last_scan_untracked_cells > 0 {
                lineage_metric_chip(
                    ui,
                    "Untracked",
                    archive.last_scan_untracked_cells.to_string(),
                    p.status_warn,
                );
            }
            let (minor_variant_observations, minor_variant_cells, _minor_variant_organisms) =
                archive.minor_variant_totals();
            if minor_variant_observations > 0 {
                lineage_metric_chip(
                    ui,
                    "Variants",
                    format!("{}/{}", minor_variant_observations, minor_variant_cells),
                    p.status_warn,
                );
            }
            lineage_metric_chip(ui, "Lines", archive.nodes.len().to_string(), p.status_info);
            lineage_metric_chip(ui, "Living Lines", living.to_string(), p.status_ok);
            lineage_metric_chip(ui, "Extinct", extinct.to_string(), p.status_err);
            lineage_metric_chip(ui, "Hybrids", hybrids.to_string(), p.accent_secondary);
            lineage_metric_chip(
                ui,
                "Events",
                archive.events.len().to_string(),
                p.accent_primary,
            );
            lineage_metric_chip(
                ui,
                "Loadable",
                format!(
                    "{}/{}",
                    archive.loadable_genome_count(),
                    archive.retention.max_bookmarks
                ),
                p.status_warn,
            );
        });
    });

    ui.add_space(8.0);
    if nodes.is_empty() {
        egui::Frame::new()
            .fill(p.bg_widget)
            .stroke(egui::Stroke::new(1.0, p.border_subtle))
            .inner_margin(egui::Margin::same(14))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("No lineage records yet.")
                        .size(12.0)
                        .color(p.text_secondary),
                );
            });
        return;
    }

    let available = ui.available_width();
    let split = available >= 700.0;
    if split {
        ui.columns(2, |columns| {
            columns[0].set_min_width(available * 0.42);
            render_lineage_specimen_panel(
                &mut columns[0],
                archive,
                selected_node,
                selected_snap_frame,
                scene.genomes.len(),
                context.scene_request,
            );
            render_lineage_intel_panel(&mut columns[1], archive, selected_node, report_history);
        });
    } else {
        render_lineage_specimen_panel(
            ui,
            archive,
            selected_node,
            selected_snap_frame,
            scene.genomes.len(),
            context.scene_request,
        );
        ui.add_space(8.0);
        render_lineage_intel_panel(ui, archive, selected_node, report_history);
    }

    ui.add_space(8.0);
    let capture_interval_secs = context
        .scene_manager
        .gpu_scene()
        .map(|s| s.lineage_capture_interval_seconds)
        .unwrap_or(crate::scene::lineage::LINEAGE_CAPTURE_INTERVAL_SECONDS);
    render_lineage_map_panel(
        ui,
        archive,
        selected_id_key,
        selected_snap_key,
        &mut selected_lineage_id,
        &mut selected_snap_frame,
        capture_interval_secs,
    );

    ui.add_space(8.0);
    section_header(ui, "SPECIES INDEX");
    egui::ScrollArea::vertical()
        .id_salt("lineage_viewer_species_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for node in nodes.into_iter().take(80) {
                if render_lineage_branch_row(ui, node, Some(node.id) == selected_lineage_id) {
                    selected_lineage_id = Some(node.id);
                    ui.ctx()
                        .data_mut(|data| data.insert_temp(selected_id_key, node.id));
                }
            }
        });
}

fn lineage_metric_chip(ui: &mut Ui, label: &str, value: String, accent: egui::Color32) {
    egui::Frame::new()
        .fill(palette().bg_widget)
        .stroke(egui::Stroke::new(1.0, accent.linear_multiply(0.65)))
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .size(9.5)
                        .color(palette().text_dim),
                );
                ui.label(egui::RichText::new(value).size(12.0).color(accent).strong());
            });
        });
}

fn render_lineage_specimen_panel(
    ui: &mut Ui,
    archive: &crate::scene::lineage::EcosystemLineageArchive,
    selected_node: Option<&crate::scene::lineage::LineageNode>,
    selected_snap_frame: Option<i32>,
    scene_genome_count: usize,
    scene_request: &mut SceneModeRequest,
) {
    let p = palette();
    egui::Frame::new()
        .fill(p.bg_panel)
        .stroke(egui::Stroke::new(1.0, p.border_bright))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("SPECIMEN")
                        .size(10.0)
                        .color(p.accent_primary),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(
                            selected_node
                                .map(|node| lineage_origin_label(&node.origin))
                                .unwrap_or("UNASSIGNED"),
                        )
                        .size(10.0)
                        .color(p.text_dim),
                    );
                });
            });

            let specimen_height = 210.0;
            let thumb_ar = crate::scene::lineage::LINEAGE_THUMBNAIL_WIDTH as f32
                / crate::scene::lineage::LINEAGE_THUMBNAIL_HEIGHT as f32;
            // Constrain width to the capture aspect ratio so the image fills
            // the rect exactly without any cropping or distortion.
            let specimen_width = (specimen_height * thumb_ar).min(ui.available_width());
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(specimen_width, specimen_height),
                egui::Sense::hover(),
            );
            draw_lineage_specimen(ui, rect, selected_node, selected_snap_frame);

            ui.add_space(8.0);
            if let Some(node) = selected_node {
                ui.label(
                    egui::RichText::new(&node.display_name)
                        .size(18.0)
                        .color(p.text_primary)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "ACC-{:04} / GENOME {:03} / FIRST FRAME {}",
                        node.id,
                        node.genome_id,
                        node.first_frame.max(0)
                    ))
                    .size(10.0)
                    .color(p.text_dim),
                );

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    for tag in lineage_trait_tags(&node.traits).into_iter().take(8) {
                        lineage_badge(ui, tag);
                    }
                });

                let active_snapshot = selected_snap_frame
                    .and_then(|frame| node.snapshot_near_frame(frame))
                    .or_else(|| node.latest_snapshot());
                if let Some(snapshot) = active_snapshot {
                    let telemetry = node
                        .telemetry_history
                        .iter()
                        .min_by_key(|sample| (sample.frame - snapshot.captured_frame).abs());
                    let dominant_cell_type_name = telemetry.and_then(|sample| {
                        crate::cell::types::CellType::names()
                            .get(sample.dominant_cell_type as usize)
                            .map(|name| (*name).to_string())
                    });
                    let snapshot_report = LineageSnapshotReportSnapshot {
                        lineage_id: node.id,
                        lineage_name: node.display_name.clone(),
                        snapshot_frame: snapshot.captured_frame.max(0) as u64,
                        captured_time: snapshot.captured_time,
                        morphology_cells: snapshot.cells.len().min(u32::MAX as usize) as u32,
                        current_cells_at_snapshot: telemetry.map(|sample| sample.cells),
                        peak_cells: node.peak_cells,
                        dominant_cell_type_name,
                        active_modes: telemetry.map(|sample| sample.active_mode_count),
                        territory_radius: snapshot.world_radius,
                    };
                    if let Some(report) = render_lineage_snapshot_report(
                        &snapshot_report,
                        &ToneProfile::naturalist_field_journal(),
                    ) {
                        ui.add_space(8.0);
                        render_field_report_card(ui, &report, Some("SNAPSHOT NOTE"));
                    }
                }

                ui.add_space(8.0);
                let scene_genome_available = (node.genome_id as usize) < scene_genome_count;
                let retained_payload_available =
                    archive.loadable_bookmark_for_lineage(node.id).is_some();
                let genome_available = scene_genome_available || retained_payload_available;
                let load_response = ui.add_enabled(
                    genome_available,
                    egui::Button::new(
                        egui::RichText::new("Load Genome To Preview")
                            .size(11.0)
                            .color(p.text_primary),
                    ),
                );
                if load_response
                    .on_hover_text(if scene_genome_available {
                        "Loads from the biosphere genome table without another GPU readback"
                    } else if retained_payload_available {
                        "Loads from the rolling lineage payload retained in this biosphere save"
                    } else {
                        "This branch's full genome payload has rolled out of the loadable window"
                    })
                    .clicked()
                {
                    *scene_request = if scene_genome_available {
                        SceneModeRequest::LoadGenomeFromSceneGenome(node.genome_id)
                    } else {
                        SceneModeRequest::LoadGenomeFromLineageBookmark(node.id)
                    };
                }
            }
        });
}

fn draw_lineage_specimen(
    ui: &mut Ui,
    rect: egui::Rect,
    selected_node: Option<&crate::scene::lineage::LineageNode>,
    selected_snap_frame: Option<i32>,
) {
    let p = palette();
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, p.bg_darkest);
    draw_corner_brackets(&painter, rect, p.accent_primary);

    for i in 1..5 {
        let y = rect.top() + i as f32 * rect.height() / 5.0;
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(0.5, p.border_subtle.linear_multiply(0.6)),
        );
    }
    for i in 1..5 {
        let x = rect.left() + i as f32 * rect.width() / 5.0;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(0.5, p.border_subtle.linear_multiply(0.5)),
        );
    }

    let center = rect.center();
    let radius = rect.height().min(rect.width()) * 0.25;
    let accent = selected_node
        .map(|node| lineage_color_for_node(node))
        .unwrap_or(p.text_dim);

    let thumbnail_shown = if let Some((node, snapshot)) = selected_node.and_then(|node| {
        let snap = if let Some(frame) = selected_snap_frame {
            node.snapshot_near_frame(frame)
        } else {
            node.latest_snapshot()
        };
        snap.map(|s| (node, s))
    }) {
        if let Some(texture_id) = lineage_thumbnail_texture(
            ui.ctx(),
            node.id,
            snapshot.captured_frame,
            snapshot.scene_thumbnail_png.as_ref(),
        ) {
            let image_rect = rect.shrink(4.0);
            let thumb_w = crate::scene::lineage::LINEAGE_THUMBNAIL_WIDTH as f32;
            let thumb_h = crate::scene::lineage::LINEAGE_THUMBNAIL_HEIGHT as f32;
            // painter is already clipped to `rect`, so the draw_rect may safely
            // overflow — the bleed is hidden behind the rect boundary.
            let draw_rect = thumbnail_cover_rect(image_rect, thumb_w, thumb_h);
            let full_uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
            painter.image(texture_id, draw_rect, full_uv, egui::Color32::WHITE);
            true
        } else {
            let hovered = ui.rect_contains_pointer(rect);
            draw_lineage_adult_snapshot(&painter, rect.shrink(18.0), snapshot, accent, hovered);
            true
        }
    } else {
        false
    };

    if !thumbnail_shown {
        let seed = selected_node.map(|node| node.id as f32).unwrap_or(1.0);
        painter.circle_stroke(
            center,
            radius * 1.35,
            egui::Stroke::new(1.0, p.border_normal),
        );
        painter.circle_stroke(
            center,
            radius * 1.05,
            egui::Stroke::new(1.0, accent.linear_multiply(0.8)),
        );
        painter.circle_filled(center, radius * 0.42, accent.linear_multiply(0.35));
        painter.circle_stroke(center, radius * 0.42, egui::Stroke::new(2.0, accent));

        let lobes = selected_node
            .map(|node| node.traits.mode_count.max(3).min(14))
            .unwrap_or(6);
        for i in 0..lobes {
            let angle = seed * 0.37 + i as f32 * std::f32::consts::TAU / lobes as f32;
            let outer = center + egui::vec2(angle.cos(), angle.sin()) * radius * 0.92;
            let inner = center + egui::vec2(angle.cos(), angle.sin()) * radius * 0.48;
            painter.line_segment([inner, outer], egui::Stroke::new(1.2, accent));
            painter.circle_filled(outer, 4.0, accent.linear_multiply(0.85));
        }
    }

    if let Some(node) = selected_node {
        let status = if node.current_cells > 0 {
            "ACTIVE BIOMASS"
        } else if node.extinct_frame.is_some() {
            "EXTINCT RECORD"
        } else {
            "DORMANT RECORD"
        };
        painter.text(
            rect.left_top() + egui::vec2(12.0, 12.0),
            egui::Align2::LEFT_TOP,
            status,
            egui::FontId::proportional(10.0),
            accent,
        );
        let active_snap = if let Some(frame) = selected_snap_frame {
            node.snapshot_near_frame(frame)
        } else {
            node.latest_snapshot()
        };
        painter.text(
            rect.right_bottom() - egui::vec2(12.0, 12.0),
            egui::Align2::RIGHT_BOTTOM,
            if let Some(snapshot) = active_snap {
                let stage = if snapshot.captured_before_division {
                    "PRE-RELEASE"
                } else {
                    "BEST ADULT"
                };
                format!(
                    "{} / {} CELLS / {:.1}s",
                    stage,
                    snapshot.cells.len(),
                    snapshot.captured_time
                )
            } else {
                format!("CELLS {} / PEAK {}", node.current_cells, node.peak_cells)
            },
            egui::FontId::monospace(10.0),
            p.text_secondary,
        );
    }
}

fn draw_lineage_adult_snapshot(
    painter: &egui::Painter,
    rect: egui::Rect,
    snapshot: &crate::scene::lineage::LineageAdultSnapshot,
    fallback: egui::Color32,
    hovered: bool,
) {
    draw_lineage_adult_snapshot_scaled(painter, rect, snapshot, fallback, 3.0, 14.0, 0.45, hovered);
}

fn draw_lineage_adult_snapshot_scaled(
    painter: &egui::Painter,
    rect: egui::Rect,
    snapshot: &crate::scene::lineage::LineageAdultSnapshot,
    fallback: egui::Color32,
    min_radius: f32,
    max_radius: f32,
    radius_scale: f32,
    hovered: bool,
) {
    if snapshot.cells.is_empty() {
        return;
    }

    // Y-axis rotation angle driven by time when hovered.
    let angle = if hovered {
        painter.ctx().request_repaint();
        painter.ctx().input(|i| i.time as f32) * 0.75
    } else {
        0.0_f32
    };
    let (sin_a, cos_a) = angle.sin_cos();

    // Rotate a 3D point around the Y axis then return (x', y', z').
    let rotate_y = |p: [f32; 3]| -> [f32; 3] {
        [
            p[0] * cos_a + p[2] * sin_a,
            p[1],
            -p[0] * sin_a + p[2] * cos_a,
        ]
    };

    // Compute bounding box of rotated X/Y for stable scale.
    let mut min = egui::pos2(f32::INFINITY, f32::INFINITY);
    let mut max = egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for cell in &snapshot.cells {
        let r = rotate_y(cell.position);
        min.x = min.x.min(r[0]);
        min.y = min.y.min(r[1]);
        max.x = max.x.max(r[0]);
        max.y = max.y.max(r[1]);
    }

    let span = egui::vec2((max.x - min.x).max(1.0), (max.y - min.y).max(1.0));
    let scale = (rect.width() / span.x).min(rect.height() / span.y) * 0.72;
    let center = rect.center();
    let project = |position: [f32; 3]| -> (egui::Pos2, f32) {
        let r = rotate_y(position);
        let x = (r[0] - (min.x + max.x) * 0.5) * scale;
        let y = (r[1] - (min.y + max.y) * 0.5) * scale;
        (center + egui::vec2(x, y), r[2])
    };

    for bond in &snapshot.bonds {
        let a = bond[0] as usize;
        let b = bond[1] as usize;
        if let (Some(cell_a), Some(cell_b)) = (snapshot.cells.get(a), snapshot.cells.get(b)) {
            painter.line_segment(
                [project(cell_a.position).0, project(cell_b.position).0],
                egui::Stroke::new(1.0, palette().border_bright.linear_multiply(0.75)),
            );
        }
    }

    let mut order: Vec<_> = (0..snapshot.cells.len()).collect();
    order.sort_by(|&a, &b| {
        let za = rotate_y(snapshot.cells[a].position)[2];
        let zb = rotate_y(snapshot.cells[b].position)[2];
        za.total_cmp(&zb)
    });
    for i in order {
        let cell = &snapshot.cells[i];
        let (pos, _) = project(cell.position);
        let radius = (cell.radius * scale * radius_scale).clamp(min_radius, max_radius);
        let color = egui::Color32::from_rgb(
            (cell.color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (cell.color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (cell.color[2].clamp(0.0, 1.0) * 255.0) as u8,
        );
        let glow = color.linear_multiply(0.35 + cell.emissive.max(0.0).min(1.5) * 0.25);
        painter.circle_filled(pos + egui::vec2(2.0, 2.0), radius * 1.12, glow);
        painter.circle_filled(pos, radius, color);
        painter.circle_stroke(
            pos,
            radius,
            egui::Stroke::new(1.0, fallback.linear_multiply(0.85)),
        );
    }
}

fn draw_lineage_timeline_snapshot(
    painter: &egui::Painter,
    rect: egui::Rect,
    node: &crate::scene::lineage::LineageNode,
    selected: bool,
) {
    let accent = lineage_color_for_node(node);
    let living = node.current_cells > 0 || node.current_organisms > 0;
    let fill = if living {
        palette().bg_panel
    } else {
        palette().bg_darkest
    };
    painter.rect_filled(rect, 4.0, fill);
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(
            if selected { 1.8 } else { 1.0 },
            if selected {
                palette().text_primary
            } else {
                accent.linear_multiply(if living { 0.75 } else { 0.45 })
            },
        ),
        egui::StrokeKind::Inside,
    );

    if let Some(snapshot) = node.latest_snapshot() {
        let mut image_rect = rect.shrink(3.0);
        image_rect.max.y -= 5.0;
        if let Some(texture_id) = lineage_thumbnail_texture(
            painter.ctx(),
            node.id,
            snapshot.captured_frame,
            snapshot.scene_thumbnail_png.as_ref(),
        ) {
            let thumb_w = crate::scene::lineage::LINEAGE_THUMBNAIL_WIDTH as f32;
            let thumb_h = crate::scene::lineage::LINEAGE_THUMBNAIL_HEIGHT as f32;
            let draw_rect = thumbnail_cover_rect(image_rect, thumb_w, thumb_h);
            let full_uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
            // Clip to image_rect so the draw_rect bleed doesn't overwrite
            // adjacent timeline cards.
            painter.with_clip_rect(image_rect).image(
                texture_id,
                draw_rect,
                full_uv,
                egui::Color32::WHITE,
            );
        } else {
            let hovered = painter.ctx().input(|i| {
                i.pointer
                    .hover_pos()
                    .map(|p| rect.contains(p))
                    .unwrap_or(false)
            });
            draw_lineage_adult_snapshot_scaled(
                painter, image_rect, snapshot, accent, 1.2, 4.8, 0.36, hovered,
            );
        }
    } else {
        let center = rect.center() - egui::vec2(0.0, 2.0);
        let radius = rect.width().min(rect.height()) * 0.18;
        painter.circle_filled(center, radius, accent.linear_multiply(0.45));
        painter.circle_stroke(center, radius * 1.8, egui::Stroke::new(0.8, accent));
    }

    painter.text(
        rect.center_bottom() - egui::vec2(0.0, 2.0),
        egui::Align2::CENTER_BOTTOM,
        format!("L{}", node.id),
        egui::FontId::monospace(7.5),
        palette().text_dim,
    );
}

fn lineage_thumbnail_texture(
    ctx: &egui::Context,
    lineage_id: u64,
    frame: i32,
    png: Option<&Vec<u8>>,
) -> Option<egui::TextureId> {
    let png = png?;
    let texture_key = egui::Id::new(("lineage_scene_thumbnail", lineage_id, frame, png.len()));
    if let Some(handle) = ctx.data(|data| data.get_temp::<egui::TextureHandle>(texture_key)) {
        return Some(handle.id());
    }

    let image = image::load_from_memory(png).ok()?.to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
    let handle = ctx.load_texture(
        format!("lineage_scene_thumbnail_{lineage_id}_{}", png.len()),
        color_image,
        egui::TextureOptions::LINEAR,
    );
    let texture_id = handle.id();
    ctx.data_mut(|data| data.insert_temp(texture_key, handle));
    Some(texture_id)
}

/// Compute the draw rect for object-fit:cover rendering.
///
/// Scales the image uniformly so its shorter axis exactly fills the display
/// rect.  The longer axis overflows beyond the display rect and is clipped by
/// the painter's existing clip rect — no UV cropping required, so the full
/// texture is sampled and the overflow disappears behind the rect boundary.
fn thumbnail_cover_rect(display_rect: egui::Rect, img_w: f32, img_h: f32) -> egui::Rect {
    let img_ar = img_w / img_h;
    let rect_ar = display_rect.width() / display_rect.height();
    if img_ar >= rect_ar {
        // Image wider than rect — scale to fill height, let width bleed left/right.
        let draw_w = display_rect.height() * img_ar;
        egui::Rect::from_center_size(
            display_rect.center(),
            egui::vec2(draw_w, display_rect.height()),
        )
    } else {
        // Image taller than rect — scale to fill width, let height bleed top/bottom.
        let draw_h = display_rect.width() / img_ar;
        egui::Rect::from_center_size(
            display_rect.center(),
            egui::vec2(display_rect.width(), draw_h),
        )
    }
}

fn draw_corner_brackets(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let len = 18.0;
    let stroke = egui::Stroke::new(1.5, color);
    let corners = [
        (rect.left_top(), egui::vec2(len, 0.0), egui::vec2(0.0, len)),
        (
            rect.right_top(),
            egui::vec2(-len, 0.0),
            egui::vec2(0.0, len),
        ),
        (
            rect.left_bottom(),
            egui::vec2(len, 0.0),
            egui::vec2(0.0, -len),
        ),
        (
            rect.right_bottom(),
            egui::vec2(-len, 0.0),
            egui::vec2(0.0, -len),
        ),
    ];
    for (corner, x, y) in corners {
        painter.line_segment([corner, corner + x], stroke);
        painter.line_segment([corner, corner + y], stroke);
    }
}

fn render_lineage_intel_panel(
    ui: &mut Ui,
    archive: &crate::scene::lineage::EcosystemLineageArchive,
    selected_node: Option<&crate::scene::lineage::LineageNode>,
    report_history: &FieldReportHistory,
) {
    let p = palette();
    egui::Frame::new()
        .fill(p.bg_panel)
        .stroke(egui::Stroke::new(1.0, p.border_normal))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("BIO-METRICS")
                    .size(10.0)
                    .color(p.accent_primary),
            );
            ui.add_space(6.0);

            if let Some(node) = selected_node {
                if let Some(report) = report_history
                    .reports
                    .iter()
                    .rev()
                    .find(|report| report.rendered.involved_lineages.contains(&node.id))
                {
                    render_field_report_card(ui, &report.rendered, Some("LINEAGE REPORT"));
                    ui.add_space(4.0);
                }
                egui::Grid::new("lineage_intel_grid")
                    .num_columns(4)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| {
                        lineage_readout(ui, "Cells", node.current_cells.to_string());
                        lineage_readout(ui, "Peak", node.peak_cells.to_string());
                        ui.end_row();
                        let bodies = if archive.last_scan_organism_counts_reliable {
                            node.current_organisms.to_string()
                        } else {
                            "--".to_string()
                        };
                        lineage_readout(ui, "Bodies", bodies);
                        lineage_readout(ui, "Modes", node.traits.mode_count.to_string());
                        ui.end_row();
                        lineage_readout(ui, "Genome", node.genome_id.to_string());
                        lineage_readout(
                            ui,
                            "Generation",
                            archive.generation_for_lineage(node.id).to_string(),
                        );
                        ui.end_row();
                    });

                ui.add_space(8.0);
                if !archive.last_scan_organism_counts_reliable {
                    ui.label(
                        egui::RichText::new(
                            "Body counts unavailable until organism labels have converged.",
                        )
                        .size(10.0)
                        .color(p.status_warn),
                    );
                    ui.add_space(4.0);
                }
                if let crate::scene::lineage::LineageOrigin::Hybrid {
                    parent_a,
                    parent_b,
                    similarity,
                } = &node.origin
                {
                    ui.label(
                        egui::RichText::new(format!(
                            "Parentage: L{} + L{} / {:.0}% compatibility",
                            parent_a,
                            parent_b,
                            similarity * 100.0
                        ))
                        .size(10.0)
                        .color(p.accent_secondary),
                    );
                } else if let Some(extinct_frame) = node.extinct_frame {
                    ui.label(
                        egui::RichText::new(format!(
                            "Archived extinction at frame {}",
                            extinct_frame
                        ))
                        .size(10.0)
                        .color(p.status_err),
                    );
                } else if node.mutation_count > 0 {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} mutation observation(s) recorded",
                            node.mutation_count
                        ))
                        .size(10.0)
                        .color(p.accent_secondary),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("No exceptional lineage flags recorded")
                            .size(10.0)
                            .color(p.text_dim),
                    );
                }

                ui.add_space(10.0);
                lineage_event_feed(ui, archive, node.id);
            } else {
                ui.label(
                    egui::RichText::new("Select a species index entry to open its dossier.")
                        .size(11.0)
                        .color(p.text_secondary),
                );
            }
        });
}

fn lineage_readout(ui: &mut Ui, label: &str, value: String) {
    ui.label(
        egui::RichText::new(label)
            .size(10.0)
            .color(palette().text_dim),
    );
    ui.label(
        egui::RichText::new(value)
            .size(12.0)
            .color(palette().text_primary)
            .strong(),
    );
}

fn lineage_event_feed(
    ui: &mut Ui,
    archive: &crate::scene::lineage::EcosystemLineageArchive,
    lineage_id: u64,
) {
    let p = palette();
    ui.label(
        egui::RichText::new("NOTABLE LOG")
            .size(10.0)
            .color(p.accent_primary),
    );
    let mut events: Vec<_> = archive
        .events
        .iter()
        .filter(|event| event.lineage_id == lineage_id)
        .collect();
    events.sort_by(|a, b| b.frame.cmp(&a.frame));

    if events.is_empty() {
        ui.label(
            egui::RichText::new("No timeline events recorded for this species.")
                .size(10.0)
                .color(p.text_dim),
        );
        return;
    }

    for event in events.into_iter().take(5) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!("F{}", event.frame))
                    .size(9.5)
                    .color(if event.noteworthy {
                        p.accent_secondary
                    } else {
                        p.text_dim
                    }),
            );
            ui.label(
                egui::RichText::new(&event.title)
                    .size(10.0)
                    .color(p.text_primary),
            );
            ui.label(
                egui::RichText::new(&event.detail)
                    .size(10.0)
                    .color(p.text_secondary),
            );
        });
    }
}

fn lineage_badge(ui: &mut Ui, text: &str) {
    egui::Frame::new()
        .fill(palette().bg_darkest)
        .stroke(egui::Stroke::new(
            1.0,
            palette().accent_secondary.linear_multiply(0.55),
        ))
        .inner_margin(egui::Margin::symmetric(6, 3))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text.to_ascii_uppercase())
                    .size(9.0)
                    .color(palette().text_secondary),
            );
        });
}

fn render_lineage_branch_row(
    ui: &mut Ui,
    node: &crate::scene::lineage::LineageNode,
    selected: bool,
) -> bool {
    let p = palette();
    let fill = if selected { p.bg_selected } else { p.bg_widget };
    let stroke = if selected {
        egui::Stroke::new(1.0, p.accent_primary)
    } else {
        egui::Stroke::new(1.0, p.border_subtle)
    };
    let mut clicked = false;
    egui::Frame::new()
        .fill(fill)
        .stroke(stroke)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let color = lineage_color_for_node(node);
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 18.0), egui::Sense::hover());
                ui.painter().circle_filled(dot_rect.center(), 4.0, color);

                clicked |= ui
                    .selectable_label(
                        selected,
                        egui::RichText::new(&node.display_name)
                            .size(12.0)
                            .color(p.text_primary),
                    )
                    .clicked();

                ui.label(
                    egui::RichText::new(format!("L{:04}", node.id))
                        .size(9.5)
                        .color(p.text_dim),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(lineage_origin_label(&node.origin))
                            .size(9.5)
                            .color(color),
                    );
                    ui.label(
                        egui::RichText::new(format!("{} cells", node.current_cells))
                            .size(9.5)
                            .color(p.text_secondary),
                    );
                });
            });
        });
    ui.add_space(4.0);
    clicked
}

fn count_subtree_size(root: u64, children_map: &std::collections::HashMap<u64, Vec<u64>>) -> usize {
    let mut count = 0;
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if let Some(kids) = children_map.get(&id) {
            count += kids.len();
            stack.extend(kids.iter().copied());
        }
    }
    count
}

fn render_lineage_map_panel(
    ui: &mut Ui,
    archive: &crate::scene::lineage::EcosystemLineageArchive,
    selected_id_key: egui::Id,
    selected_snap_key: egui::Id,
    selected_lineage_id: &mut Option<u64>,
    selected_snap_frame: &mut Option<i32>,
    capture_interval_secs: f32,
) {
    egui::Frame::new()
        .fill(palette().bg_panel)
        .stroke(egui::Stroke::new(1.0, palette().border_normal))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            let p = palette();
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("EVOLUTION TIMELINE")
                        .size(10.0)
                        .color(p.accent_primary),
                );
                ui.label(
                    egui::RichText::new("scroll to zoom · drag to pan")
                        .size(9.0)
                        .color(p.text_dim),
                );
            });
            ui.add_space(6.0);
            render_lineage_map(
                ui,
                archive,
                selected_id_key,
                selected_snap_key,
                capture_interval_secs,
                selected_lineage_id,
                selected_snap_frame,
            );
        });
}

fn nice_time_interval(total_seconds: f32) -> f32 {
    const NICE: &[f32] = &[
        5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0,
    ];
    let raw = total_seconds / 8.0; // aim for ~8 ticks
    NICE.iter().copied().find(|&i| i >= raw).unwrap_or(7200.0)
}

fn format_sim_time(seconds: f32) -> String {
    let s = seconds as u32;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{sec:02}s")
    } else {
        format!("{sec}s")
    }
}

fn render_lineage_map(
    ui: &mut Ui,
    archive: &crate::scene::lineage::EcosystemLineageArchive,
    selected_id_key: egui::Id,
    selected_snap_key: egui::Id,
    capture_interval_secs: f32,
    selected_lineage_id: &mut Option<u64>,
    selected_snap_frame: &mut Option<i32>,
) {
    const TIME_STRIP_H: f32 = 20.0;
    const ROW_H: f32 = 64.0;
    const TOP_PAD: f32 = 22.0;
    const MIN_HEIGHT: f32 = 220.0;
    let available_width = ui.available_width().max(240.0);

    let zoom_key = egui::Id::new("lineage_timeline_zoom");
    let pan_key = egui::Id::new("lineage_timeline_pan");
    let collapsed_key = egui::Id::new("lineage_timeline_collapsed");
    let mut zoom: f32 = ui.ctx().data(|d| d.get_temp(zoom_key)).unwrap_or(1.0_f32);
    let mut pan_x: f32 = ui.ctx().data(|d| d.get_temp(pan_key)).unwrap_or(0.0_f32);
    let mut collapsed: std::collections::HashSet<u64> = ui
        .ctx()
        .data(|d| d.get_temp::<std::collections::HashSet<u64>>(collapsed_key))
        .unwrap_or_default();

    if archive.nodes.is_empty() {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(available_width, 120.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, palette().bg_darkest);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, palette().border_subtle),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No branches recorded",
            egui::FontId::proportional(11.0),
            palette().text_dim,
        );
        return;
    }

    let mut visible: Vec<_> = archive.nodes.iter().collect();
    // Keep the highest-value branches, capped at 72.
    visible.sort_by(|a, b| {
        b.noteworthy_score
            .total_cmp(&a.noteworthy_score)
            .then_with(|| b.peak_cells.cmp(&a.peak_cells))
            .then_with(|| a.first_frame.cmp(&b.first_frame))
    });
    visible.truncate(72);

    // --- DFS row assignment: each lineage gets its own dedicated row --------
    // Build a parent→children adjacency map restricted to visible nodes.
    let visible_ids: std::collections::HashSet<u64> = visible.iter().map(|n| n.id).collect();
    let mut children_map: std::collections::HashMap<u64, Vec<u64>> =
        std::collections::HashMap::new();
    let mut has_visible_parent: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for node in &visible {
        for parent_id in [node.parent_a, node.parent_b].into_iter().flatten() {
            if visible_ids.contains(&parent_id) {
                children_map.entry(parent_id).or_default().push(node.id);
                has_visible_parent.insert(node.id);
            }
        }
    }
    // Sort children by first_frame so earlier branches are on top.
    for children in children_map.values_mut() {
        children.sort_by_key(|&id| {
            visible
                .iter()
                .find(|n| n.id == id)
                .map(|n| n.first_frame)
                .unwrap_or(0)
        });
    }

    // Roots: visible nodes with no visible parent.
    let mut roots: Vec<u64> = visible
        .iter()
        .filter(|n| !has_visible_parent.contains(&n.id))
        .map(|n| n.id)
        .collect();
    roots.sort_by_key(|&id| {
        visible
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.first_frame)
            .unwrap_or(0)
    });

    // DFS to build row order. Collapsed nodes: their children are hidden.
    let mut row_order: Vec<u64> = Vec::with_capacity(visible.len());
    let mut stack: Vec<u64> = roots.into_iter().rev().collect();
    while let Some(id) = stack.pop() {
        if row_order.contains(&id) {
            continue;
        }
        row_order.push(id);
        if !collapsed.contains(&id) {
            if let Some(kids) = children_map.get(&id) {
                for &kid in kids.iter().rev() {
                    stack.push(kid);
                }
            }
        }
    }
    // Safety: append any unreached root-level nodes.
    for node in &visible {
        if !row_order.contains(&node.id) && !has_visible_parent.contains(&node.id) {
            row_order.push(node.id);
        }
    }

    let num_rows = row_order.len().max(1);
    let map_height = (TOP_PAD + num_rows as f32 * ROW_H + TIME_STRIP_H + 8.0).max(MIN_HEIGHT);

    let min_frame = visible
        .iter()
        .map(|node| node.first_frame)
        .min()
        .unwrap_or(0);
    let branch_max_frame = visible
        .iter()
        .map(|node| node.first_frame)
        .max()
        .unwrap_or(min_frame)
        .max(min_frame + 1);
    let max_frame = archive
        .last_scan_frame
        .unwrap_or(branch_max_frame)
        .max(branch_max_frame)
        .max(min_frame + 1);

    let estimated_frames_per_second = if archive.last_scan_time > 1.0 {
        max_frame as f32 / archive.last_scan_time.max(1.0)
    } else {
        60.0
    };
    let interval_frames = (capture_interval_secs * estimated_frames_per_second)
        .round()
        .max(1.0) as i32;

    // Fixed viewport: always fills the available width; zoom/pan controls the virtual content.
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(available_width, map_height),
        egui::Sense::click_and_drag(),
    );

    // --- Pan: left-click drag ---
    if resp.dragged_by(egui::PointerButton::Primary) {
        pan_x -= resp.drag_delta().x;
    }

    // --- Zoom: scroll wheel, centred on pointer ---
    // Consume the scroll delta when hovered so it doesn't leak to the parent panel.
    let scroll = if resp.hovered() {
        ui.input_mut(|i| {
            let dy = i.smooth_scroll_delta.y;
            i.smooth_scroll_delta.y = 0.0;
            dy
        })
    } else {
        0.0_f32
    };
    if scroll.abs() > 0.1 {
        let origin_x = rect.left() + 18.0;
        let cursor_x = ui
            .input(|i| i.pointer.hover_pos())
            .map(|p| p.x)
            .unwrap_or(rect.center().x);
        let scale = (1.0 + scroll * 0.008).clamp(0.5, 2.0);
        let new_zoom = (zoom * scale).clamp(0.25, 40.0);
        // Keep the world point under the cursor fixed.
        pan_x = (cursor_x - origin_x + pan_x) * (new_zoom / zoom) - (cursor_x - origin_x);
        zoom = new_zoom;
    }

    // Clamp pan so you can't scroll past the content edges.
    let base_inner_w = (available_width - 36.0).max(1.0);
    let virtual_w = base_inner_w * zoom;
    pan_x = pan_x.clamp(0.0, (virtual_w - base_inner_w).max(0.0));

    // Persist updated values.
    ui.ctx().data_mut(|d| d.insert_temp(zoom_key, zoom));
    ui.ctx().data_mut(|d| d.insert_temp(pan_key, pan_x));

    let inner_left = rect.left() + 18.0 - pan_x;
    let inner_width = virtual_w;

    let painter = ui.painter_at(rect).with_clip_rect(rect);
    painter.rect_filled(rect, 4.0, palette().bg_darkest);
    // Alternating row backgrounds.
    for (row_idx, &node_id) in row_order.iter().enumerate() {
        let row_top = rect.top() + TOP_PAD + row_idx as f32 * ROW_H;
        let row_rect =
            egui::Rect::from_x_y_ranges(rect.left()..=rect.right(), row_top..=(row_top + ROW_H));
        if row_idx % 2 == 1 {
            painter.rect_filled(row_rect, 0.0, palette().bg_widget.linear_multiply(0.18));
        }
        // Collapse toggle + row label on the left margin.
        if let Some(node) = visible.iter().find(|n| n.id == node_id) {
            let center_y = row_top + ROW_H * 0.5;
            let has_children = children_map.contains_key(&node_id);
            let is_collapsed = collapsed.contains(&node_id);

            if has_children {
                // Clickable triangle toggle (12×12 hit area).
                let toggle_center = egui::pos2(rect.left() + 9.0, center_y);
                let toggle_rect =
                    egui::Rect::from_center_size(toggle_center, egui::vec2(14.0, 14.0));
                let toggle_resp = ui.interact(
                    toggle_rect,
                    egui::Id::new(("lineage_collapse_toggle", node_id)),
                    egui::Sense::click(),
                );
                if toggle_resp.clicked() {
                    if is_collapsed {
                        collapsed.remove(&node_id);
                    } else {
                        collapsed.insert(node_id);
                    }
                }
                let tri_color = if toggle_resp.hovered() {
                    palette().text_primary
                } else {
                    palette().text_dim
                };
                // Draw ▶ (collapsed) or ▼ (expanded).
                let s = 4.5_f32;
                if is_collapsed {
                    painter.add(egui::Shape::convex_polygon(
                        vec![
                            toggle_center + egui::vec2(-s * 0.7, -s),
                            toggle_center + egui::vec2(-s * 0.7, s),
                            toggle_center + egui::vec2(s, 0.0),
                        ],
                        tri_color,
                        egui::Stroke::NONE,
                    ));
                } else {
                    painter.add(egui::Shape::convex_polygon(
                        vec![
                            toggle_center + egui::vec2(-s, -s * 0.7),
                            toggle_center + egui::vec2(s, -s * 0.7),
                            toggle_center + egui::vec2(0.0, s),
                        ],
                        tri_color,
                        egui::Stroke::NONE,
                    ));
                }

                // Hidden-count badge when collapsed.
                if is_collapsed {
                    let hidden = count_subtree_size(node_id, &children_map);
                    if hidden > 0 {
                        let badge_text = format!("+{}", hidden);
                        let badge_pos = egui::pos2(rect.left() + 22.0, center_y);
                        painter.text(
                            badge_pos,
                            egui::Align2::LEFT_CENTER,
                            badge_text,
                            egui::FontId::proportional(9.5),
                            palette().accent_secondary,
                        );
                    }
                }
            }

            // ID label (shifted right when there's a toggle).
            let label_x = if has_children {
                rect.left() + 38.0
            } else {
                rect.left() + 6.0
            };
            painter.text(
                egui::pos2(label_x, center_y),
                egui::Align2::LEFT_CENTER,
                format!("L{}", node.id),
                egui::FontId::monospace(7.5),
                palette().text_dim.linear_multiply(0.7),
            );
        }
    }
    // Horizontal row separator lines.
    for i in 0..=num_rows {
        let y = rect.top() + TOP_PAD + i as f32 * ROW_H;
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(0.5, palette().border_subtle.linear_multiply(0.45)),
        );
    }
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, palette().border_subtle),
        egui::StrokeKind::Inside,
    );

    // Linear frame → screen-x projection.
    let project = |frame: i32| -> f32 {
        let t = (frame - min_frame) as f32 / (max_frame - min_frame).max(1) as f32;
        inner_left + t * inner_width
    };

    // Interval tick marks — density adapts to zoom so labels never cluster.
    // Compute how many `interval_frames` steps fit in MIN_LABEL_PX pixels.
    const MIN_LABEL_PX: f32 = 70.0;
    let px_per_interval =
        inner_width * interval_frames as f32 / (max_frame - min_frame).max(1) as f32;
    let step_multiplier = ((MIN_LABEL_PX / px_per_interval).ceil() as i32).max(1);
    let display_interval = interval_frames * step_multiplier;

    let first_interval = ((min_frame + display_interval - 1) / display_interval) * display_interval;
    let mut interval_frame = first_interval;
    while interval_frame <= max_frame {
        let x = project(interval_frame);
        if x >= rect.left() && x <= rect.right() {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(0.75, palette().accent_primary.linear_multiply(0.28)),
            );
            painter.text(
                egui::pos2(x + 3.0, rect.top() + 5.0),
                egui::Align2::LEFT_TOP,
                format_sim_time(interval_frame as f32 / estimated_frames_per_second),
                egui::FontId::monospace(8.5),
                palette().text_dim,
            );
        }
        interval_frame = interval_frame.saturating_add(display_interval);
        if display_interval <= 0 {
            break;
        }
    }

    // Assign each node a position: X from its first_frame, Y from its row index.
    let mut positions: std::collections::HashMap<u64, egui::Pos2> =
        std::collections::HashMap::new();
    for (row_idx, &node_id) in row_order.iter().enumerate() {
        let Some(node) = visible.iter().find(|n| n.id == node_id) else {
            continue;
        };
        let x = project(node.first_frame);
        let y = rect.top() + TOP_PAD + row_idx as f32 * ROW_H + ROW_H * 0.5;
        positions.insert(node_id, egui::pos2(x, y));
    }

    // Thumbnail dimensions.
    let thumb_normal = egui::vec2(38.0, 34.0);
    let thumb_selected = egui::vec2(46.0, 42.0);
    let thumb_end = egui::vec2(30.0, 27.0); // smaller "latest state" thumbnail at bar end

    // Helper: the x-coordinate where a node's lifespan bar ends.
    let bar_end_x = |node: &crate::scene::lineage::LineageNode| -> f32 {
        let end_frame = node
            .extinct_frame
            .unwrap_or_else(|| node.last_seen_frame.max(max_frame));
        project(end_frame)
    };

    // --- Pass 1: horizontal lifespan bars ---
    for node in &visible {
        let Some(&pos) = positions.get(&node.id) else {
            continue;
        };
        let color = lineage_color_for_node(node);
        let alive = node.current_cells > 0 || node.current_organisms > 0;
        let bar_alpha = if alive { 0.65 } else { 0.35 };
        let bar_thickness = if alive { 2.0 } else { 1.2 };
        let end_x = bar_end_x(node);
        // The bar starts from the right edge of the birth thumbnail.
        let start_x = pos.x + thumb_normal.x * 0.5;
        if end_x > start_x {
            painter.line_segment(
                [egui::pos2(start_x, pos.y), egui::pos2(end_x, pos.y)],
                egui::Stroke::new(bar_thickness, color.linear_multiply(bar_alpha)),
            );
        }
    }

    // --- Pass 2: L-shaped branch connectors ---
    // A child branches off from its parent's lifespan bar at the child's
    // first_frame x-coordinate.  Draw a vertical segment from the parent's
    // Y to the child's Y at that x, then a short horizontal lead-in to the
    // child thumbnail.
    for node in &visible {
        let Some(child_pos) = positions.get(&node.id).copied() else {
            continue;
        };
        let color = lineage_color_for_node(node);
        for parent_id in [node.parent_a, node.parent_b].into_iter().flatten() {
            let Some(parent_pos) = positions.get(&parent_id).copied() else {
                continue;
            };
            let branch_x = child_pos.x - thumb_normal.x * 0.5; // left edge of child thumb
                                                               // Vertical from parent bar to child row.
            painter.line_segment(
                [
                    egui::pos2(branch_x, parent_pos.y),
                    egui::pos2(branch_x, child_pos.y),
                ],
                egui::Stroke::new(1.0, color.linear_multiply(0.55)),
            );
            // Short horizontal from branch point to child thumbnail.
            painter.line_segment(
                [
                    egui::pos2(branch_x, child_pos.y),
                    egui::pos2(child_pos.x, child_pos.y),
                ],
                egui::Stroke::new(1.0, color.linear_multiply(0.55)),
            );
        }
    }

    // --- Pass 3: "latest state" thumbnail at bar end (living nodes only) ---
    // Only drawn when the bar is long enough to warrant a separate end marker
    // (i.e., the node has actually been around for a while) and there's a
    // snapshot available.
    for node in &visible {
        let Some(&pos) = positions.get(&node.id) else {
            continue;
        };
        let alive = node.current_cells > 0 || node.current_organisms > 0;
        if !alive || node.latest_snapshot().is_none() {
            continue;
        }
        let end_x = bar_end_x(node);
        // Only show if the bar spans at least two thumbnail-widths of display space.
        if end_x - pos.x < thumb_normal.x * 2.0 {
            continue;
        }
        let end_pos = egui::pos2(end_x, pos.y);
        let end_rect = egui::Rect::from_center_size(end_pos, thumb_end);
        draw_lineage_timeline_snapshot(&painter, end_rect, node, false);
    }

    // --- Pass 5: interval snapshot dots / mini-thumbnails along each node's bar ---
    const DOT_R: f32 = 4.5;
    const MINI_THUMB: egui::Vec2 = egui::vec2(28.0, 25.0);
    let mut snap_interval_frame = first_interval;
    while snap_interval_frame <= max_frame {
        {
            let tick_x = project(snap_interval_frame);
            for node in &visible {
                let Some(&pos) = positions.get(&node.id) else {
                    continue;
                };
                let end_x = bar_end_x(node);
                // Only draw if this interval falls within the node's lifespan bar.
                if tick_x <= pos.x + thumb_normal.x * 0.5 || tick_x > end_x + 1.0 {
                    continue;
                }
                let dot_pos = egui::pos2(tick_x, pos.y);
                let snap = node.snapshot_near_frame(snap_interval_frame);
                let has_snap = snap
                    .map(|s| (s.captured_frame - snap_interval_frame).abs() <= interval_frames / 2)
                    .unwrap_or(false);
                let is_selected = Some(node.id) == *selected_lineage_id
                    && selected_snap_frame
                        .map(|f| (f - snap_interval_frame).abs() <= interval_frames / 2)
                        .unwrap_or(false);

                let interact_rect = if has_snap {
                    egui::Rect::from_center_size(dot_pos, MINI_THUMB)
                } else {
                    egui::Rect::from_center_size(dot_pos, egui::vec2(DOT_R * 2.5, DOT_R * 2.5))
                };
                let resp = ui.interact(
                    interact_rect,
                    egui::Id::new(("lineage_snap_dot", node.id, snap_interval_frame)),
                    egui::Sense::click(),
                );
                if resp.clicked() {
                    *selected_lineage_id = Some(node.id);
                    *selected_snap_frame = Some(snap_interval_frame);
                    ui.ctx()
                        .data_mut(|data| data.insert_temp(selected_id_key, node.id));
                    ui.ctx()
                        .data_mut(|data| data.insert_temp(selected_snap_key, snap_interval_frame));
                }

                let color = lineage_color_for_node(node);
                if has_snap {
                    if let Some(snapshot) = snap {
                        let mini_rect = egui::Rect::from_center_size(dot_pos, MINI_THUMB);
                        let fill = if is_selected {
                            palette().accent_primary.linear_multiply(0.18)
                        } else {
                            palette().bg_darkest
                        };
                        painter.rect_filled(mini_rect, 3.0, fill);
                        if is_selected {
                            // Bright outer glow rect.
                            painter.rect_stroke(
                                mini_rect.expand(2.0),
                                4.0,
                                egui::Stroke::new(
                                    1.5,
                                    palette().accent_primary.linear_multiply(0.55),
                                ),
                                egui::StrokeKind::Outside,
                            );
                        }
                        painter.rect_stroke(
                            mini_rect,
                            3.0,
                            egui::Stroke::new(
                                if is_selected { 2.0 } else { 0.8 },
                                if is_selected {
                                    palette().text_primary
                                } else {
                                    color.linear_multiply(0.55)
                                },
                            ),
                            egui::StrokeKind::Inside,
                        );
                        let image_rect = mini_rect.shrink(2.0);
                        if let Some(texture_id) = lineage_thumbnail_texture(
                            ui.ctx(),
                            node.id,
                            snapshot.captured_frame,
                            snapshot.scene_thumbnail_png.as_ref(),
                        ) {
                            let thumb_w = crate::scene::lineage::LINEAGE_THUMBNAIL_WIDTH as f32;
                            let thumb_h = crate::scene::lineage::LINEAGE_THUMBNAIL_HEIGHT as f32;
                            let draw_rect = thumbnail_cover_rect(image_rect, thumb_w, thumb_h);
                            let full_uv =
                                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
                            painter.with_clip_rect(image_rect).image(
                                texture_id,
                                draw_rect,
                                full_uv,
                                egui::Color32::WHITE,
                            );
                        } else {
                            let hovered = painter.ctx().input(|i| {
                                i.pointer
                                    .hover_pos()
                                    .map(|p| interact_rect.contains(p))
                                    .unwrap_or(false)
                            });
                            draw_lineage_adult_snapshot_scaled(
                                &painter, image_rect, snapshot, color, 0.8, 3.5, 0.3, hovered,
                            );
                        }
                    }
                } else {
                    // Plain dot
                    let dot_r = if is_selected { DOT_R * 1.5 } else { DOT_R };
                    let dot_color = if is_selected {
                        palette().accent_primary
                    } else {
                        color.linear_multiply(0.55)
                    };
                    painter.circle_filled(dot_pos, dot_r, dot_color);
                    if is_selected {
                        painter.circle_stroke(
                            dot_pos,
                            dot_r + 4.0,
                            egui::Stroke::new(1.2, palette().text_primary),
                        );
                    }
                }

                if resp.hovered() {
                    let snap_time = if let Some(s) =
                        snap.filter(|_| has_snap).filter(|s| s.captured_time > 0.0)
                    {
                        format_sim_time(s.captured_time)
                    } else {
                        format_sim_time(snap_interval_frame as f32 / estimated_frames_per_second)
                    };
                    resp.on_hover_text(format!(
                        "{} — {}{}\nClick to view snapshot",
                        node.display_name,
                        snap_time,
                        if has_snap {
                            " (snapshot)"
                        } else {
                            " (no snapshot)"
                        }
                    ));
                }
            }
        }
        snap_interval_frame = snap_interval_frame.saturating_add(interval_frames);
        if interval_frames <= 0 {
            break;
        }
    }

    // --- Pass 4: birth thumbnails and interaction ---
    for node in &visible {
        let Some(pos) = positions.get(&node.id).copied() else {
            continue;
        };
        let selected = Some(node.id) == *selected_lineage_id;
        let thumb_size = if selected {
            thumb_selected
        } else {
            thumb_normal
        };
        let node_rect = egui::Rect::from_center_size(pos, thumb_size);
        let response = ui.interact(
            node_rect,
            egui::Id::new(("lineage_map_node", node.id)),
            egui::Sense::click(),
        );
        if response.clicked() {
            *selected_lineage_id = Some(node.id);
            *selected_snap_frame = None;
            ui.ctx()
                .data_mut(|data| data.insert_temp(selected_id_key, node.id));
            ui.ctx()
                .data_mut(|data| data.insert_temp::<Option<i32>>(selected_snap_key, None));
        }

        draw_lineage_timeline_snapshot(&painter, node_rect, node, selected);
        if node.extinct_frame.is_some() {
            let mark = 6.0;
            let x_center = node_rect.right_top() + egui::vec2(-7.0, 7.0);
            painter.line_segment(
                [
                    x_center + egui::vec2(-mark, -mark),
                    x_center + egui::vec2(mark, mark),
                ],
                egui::Stroke::new(1.4, palette().status_err),
            );
            painter.line_segment(
                [
                    x_center + egui::vec2(-mark, mark),
                    x_center + egui::vec2(mark, -mark),
                ],
                egui::Stroke::new(1.4, palette().status_err),
            );
        }

        if response.hovered() {
            response.on_hover_text(format!(
                "{}\nLineage {} / Genome {}\nFirst frame: {}\nCells: {} / Peak: {}",
                node.display_name,
                node.id,
                node.genome_id,
                node.first_frame,
                node.current_cells,
                node.peak_cells
            ));
        }
    }

    // --- Vertical selection line at the active snap frame -------------------
    if let (Some(_sel_id), Some(sel_frame)) = (*selected_lineage_id, *selected_snap_frame) {
        let sx = project(sel_frame);
        if sx >= rect.left() && sx <= rect.right() {
            let p = palette();
            // Soft backdrop line.
            painter.line_segment(
                [
                    egui::pos2(sx, rect.top()),
                    egui::pos2(sx, rect.bottom() - TIME_STRIP_H),
                ],
                egui::Stroke::new(6.0, p.accent_primary.linear_multiply(0.12)),
            );
            // Sharp foreground line.
            painter.line_segment(
                [
                    egui::pos2(sx, rect.top()),
                    egui::pos2(sx, rect.bottom() - TIME_STRIP_H),
                ],
                egui::Stroke::new(1.5, p.accent_primary.linear_multiply(0.9)),
            );
            // Label at the top.
            let sel_label = format_sim_time(sel_frame as f32 / estimated_frames_per_second);
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(sx + 3.0, rect.top() + TOP_PAD - 14.0),
                    egui::vec2(48.0, 13.0),
                ),
                2.0,
                p.accent_primary.linear_multiply(0.25),
            );
            painter.text(
                egui::pos2(sx + 5.0, rect.top() + TOP_PAD - 13.0),
                egui::Align2::LEFT_TOP,
                sel_label,
                egui::FontId::monospace(9.0),
                p.text_primary,
            );
        }
    }

    // Persist collapse state.
    ui.ctx()
        .data_mut(|d| d.insert_temp(collapsed_key, collapsed));

    // --- Time strip along the bottom ---
    let time_strip_top = rect.bottom() - TIME_STRIP_H;
    painter.line_segment(
        [
            egui::pos2(rect.left(), time_strip_top),
            egui::pos2(rect.right(), time_strip_top),
        ],
        egui::Stroke::new(0.5, palette().border_subtle.linear_multiply(0.6)),
    );

    let total_seconds = archive.last_scan_time.max(1.0);
    let fps = max_frame as f32 / total_seconds;

    // Pick tick interval based on the visible time span (available_width / inner_width
    // of total time), so density stays constant regardless of zoom.
    let visible_seconds = total_seconds * available_width / inner_width.max(1.0);
    let tick_interval_secs = nice_time_interval(visible_seconds);

    // Start from the first whole-interval boundary ≥ time at left viewport edge.
    let left_time = (pan_x / inner_width * total_seconds).max(0.0);
    let t_start = (left_time / tick_interval_secs).floor() * tick_interval_secs;
    let t_end = total_seconds + tick_interval_secs;

    let mut t = t_start;
    while t <= t_end {
        let tick_frame = (t * fps) as i32 + min_frame;
        let x = project(tick_frame);
        if x >= rect.left() && x <= rect.right() {
            painter.line_segment(
                [
                    egui::pos2(x, time_strip_top),
                    egui::pos2(x, time_strip_top + 5.0),
                ],
                egui::Stroke::new(0.75, palette().accent_secondary.linear_multiply(0.6)),
            );
            painter.text(
                egui::pos2(x + 3.0, time_strip_top + 3.0),
                egui::Align2::LEFT_TOP,
                format_sim_time(t),
                egui::FontId::monospace(8.5),
                palette().text_secondary,
            );
        }
        t += tick_interval_secs;
    }
}

fn lineage_color_for_node(node: &crate::scene::lineage::LineageNode) -> egui::Color32 {
    match &node.origin {
        crate::scene::lineage::LineageOrigin::Hybrid { .. } => palette().accent_secondary,
        crate::scene::lineage::LineageOrigin::Mutation { .. } => palette().status_warn,
        _ if node.current_cells > 0 || node.current_organisms > 0 => palette().status_ok,
        _ if node.extinct_frame.is_some() => palette().status_err,
        _ => palette().text_dim,
    }
}

fn lineage_origin_label(origin: &crate::scene::lineage::LineageOrigin) -> &'static str {
    match origin {
        crate::scene::lineage::LineageOrigin::UserInserted => "USER ROOT",
        crate::scene::lineage::LineageOrigin::ProceduralSeed => "SEED",
        crate::scene::lineage::LineageOrigin::Mutation { .. } => "MUTATION",
        crate::scene::lineage::LineageOrigin::Hybrid { .. } => "HYBRID",
        crate::scene::lineage::LineageOrigin::Unknown => "UNKNOWN",
    }
}

fn lineage_trait_tags(traits: &crate::scene::lineage::LineageTraitSummary) -> Vec<&'static str> {
    let mut tags = Vec::new();
    if traits.has_phagocyte {
        tags.push("phagocyte");
    }
    if traits.has_photocyte {
        tags.push("photocyte");
    }
    if traits.has_devorocyte {
        tags.push("devorocyte");
    }
    if traits.has_flagellocyte {
        tags.push("flagellocyte");
    }
    if traits.has_ciliocyte {
        tags.push("ciliocyte");
    }
    if traits.has_gametocyte {
        tags.push("gametocyte");
    }
    if traits.has_cognocyte {
        tags.push("cognocyte");
    }
    if traits.uses_signals {
        tags.push("signals");
    }
    if traits.uses_scaffolds {
        tags.push("scaffold");
    }
    tags
}

/// Render the GenomeEditor panel (placeholder).
fn render_genome_editor(ui: &mut Ui, context: &mut PanelContext) {
    ui.heading("Genome Editor");
    ui.separator();
    ui.label(format!("Genome: {}", context.genome.name));
    ui.label(format!("Initial Mode: {}", context.genome.initial_mode));

    ui.separator();
    ui.label(
        "Use the individual panels (Rotation, Parent Settings, etc.) to edit genome properties.",
    );
}

/// Render the PerformanceMonitor panel.
fn render_performance_monitor(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    let perf = context.performance;

    // GPU Readbacks toggle at the top
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.gpu_readbacks_enabled, "Enable GPU Readbacks")
            .on_hover_text("Allow the CPU to read back cell data from the GPU for the Cell Inspector. Disable to reduce GPU-CPU synchronization overhead if the inspector is not needed");
        ui.checkbox(&mut state.gpu_timing_enabled, "GPU Frame Timing")
            .on_hover_text("Measure per-segment GPU frame time with timestamp queries. Disable to remove timestamp query and readback overhead");
    });
    ui.add_space(4.0);

    // Culling section (GPU mode only)
    if context.current_mode == crate::ui::types::SimulationMode::Gpu {
        section_header(ui, "GPU RENDERING");

        // Culling toggles
        ui.horizontal(|ui| {
            ui.checkbox(&mut state.frustum_enabled, "Frustum")
                .on_hover_text("Discard cells outside the camera frustum");
            ui.checkbox(&mut state.occlusion_enabled, "Occlusion")
                .on_hover_text("Discard cells hidden behind other geometry (requires Hi-Z)");
        });
        ui.add_space(4.0);

        let (total, visible, frustum_culled, occluded) = perf.culling_stats();

        egui::Grid::new("gpu_stats_grid")
            .num_columns(2)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Total Cells")
                        .size(11.0)
                        .color(palette().text_secondary),
                );
                ui.label(
                    egui::RichText::new(format!("{}", total))
                        .size(11.0)
                        .color(palette().text_primary),
                );
                ui.end_row();
                ui.label(
                    egui::RichText::new("Visible")
                        .size(11.0)
                        .color(palette().text_secondary),
                );
                ui.label(
                    egui::RichText::new(format!("{}", visible))
                        .size(11.0)
                        .color(palette().status_ok),
                );
                ui.end_row();
                ui.label(
                    egui::RichText::new("Frustum Culled")
                        .size(11.0)
                        .color(palette().text_secondary),
                );
                ui.label(
                    egui::RichText::new(format!("{}", frustum_culled))
                        .size(11.0)
                        .color(palette().text_dim),
                );
                ui.end_row();
                ui.label(
                    egui::RichText::new("Occluded")
                        .size(11.0)
                        .color(palette().text_secondary),
                );
                ui.label(
                    egui::RichText::new(format!("{}", occluded))
                        .size(11.0)
                        .color(palette().text_dim),
                );
                ui.end_row();
                if total > 0 {
                    let cull_percent = ((total - visible) as f32 / total as f32) * 100.0;
                    ui.label(
                        egui::RichText::new("Cull Rate")
                            .size(11.0)
                            .color(palette().text_secondary),
                    );
                    ui.label(
                        egui::RichText::new(format!("{:.1}%", cull_percent))
                            .size(11.0)
                            .color(palette().accent_secondary),
                    );
                    ui.end_row();
                }
            });

        ui.add_space(6.0);
        section_header(ui, "LOD");
        egui::Grid::new("lod_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Scale")
                        .size(11.0)
                        .color(palette().text_secondary),
                )
                .on_hover_text(
                    "Higher keeps cells at full detail from farther away",
                );
                ui.add(
                    egui::Slider::new(&mut state.lod_scale_factor, 50.0..=2000.0).fixed_decimals(0),
                );
                ui.end_row();
                ui.label(
                    egui::RichText::new("Full Detail")
                        .size(11.0)
                        .color(palette().text_secondary),
                )
                .on_hover_text(
                    "Screen-size threshold for switching from a shadowed basic sphere to full procedural detail",
                );
                ui.add(
                    egui::Slider::new(&mut state.lod_threshold_low, 1.0..=1000.0).fixed_decimals(0),
                );
                ui.end_row();
                ui.label(
                    egui::RichText::new("Debug Colors")
                        .size(11.0)
                        .color(palette().text_secondary),
                )
                .on_hover_text(
                    "Tint cells by LOD level: red=basic sphere, green=full detail",
                );
                ui.checkbox(&mut state.lod_debug_colors, "");
                ui.end_row();
            });

        ui.add_space(6.0);
    }

    // FPS and Frame Time section
    section_header(ui, "FRAME RATE");

    let fps = perf.fps();
    let fps_color = if fps >= 50.0 {
        palette().status_ok
    } else if fps >= 30.0 {
        palette().status_warn
    } else {
        palette().status_err
    };

    egui::Grid::new("fps_grid")
        .num_columns(2)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("FPS")
                    .size(11.0)
                    .color(palette().text_secondary),
            );
            ui.label(
                egui::RichText::new(format!("{:.1}", fps))
                    .size(13.0)
                    .strong()
                    .color(fps_color),
            );
            ui.end_row();
            ui.label(
                egui::RichText::new("Frame Time")
                    .size(11.0)
                    .color(palette().text_secondary),
            );
            ui.label(
                egui::RichText::new(format!("{:.2} ms", perf.average_frame_time_ms()))
                    .size(11.0)
                    .color(palette().text_primary),
            );
            ui.end_row();
            ui.label(
                egui::RichText::new("Min / Max")
                    .size(11.0)
                    .color(palette().text_secondary),
            );
            ui.label(
                egui::RichText::new(format!(
                    "{:.2} / {:.2} ms",
                    perf.min_frame_time_ms(),
                    perf.max_frame_time_ms()
                ))
                .size(11.0)
                .color(palette().text_dim),
            );
            ui.end_row();
        });

    // FPS graph
    let frame_times: Vec<f32> = perf.frame_time_history().collect();
    if !frame_times.is_empty() {
        let max_fps = 60.0_f32;
        ui.add_space(4.0);

        let plot_height = 36.0;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), plot_height),
            egui::Sense::hover(),
        );
        let rect = response.rect;

        painter.rect_filled(rect, 2.0, palette().bg_darkest);
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, palette().border_subtle),
            egui::StrokeKind::Outside,
        );

        let bar_width = rect.width() / frame_times.len() as f32;
        for (i, &time) in frame_times.iter().enumerate() {
            let fps_val = if time > 0.0 { 1000.0 / time } else { 0.0 };
            let fps_clamped = fps_val.min(max_fps);
            let x = rect.left() + i as f32 * bar_width;
            let height = (fps_clamped / max_fps) * rect.height();
            let y = rect.bottom() - height;
            let color = if fps_val < 20.0 {
                palette().status_err
            } else if fps_val < 40.0 {
                palette().status_warn
            } else {
                palette().status_ok
            };
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(bar_width.max(1.0), height)),
                0.0,
                color,
            );
        }
    }

    ui.add_space(6.0);

    // GPU Timing section (per-segment breakdown via timestamp queries)
    let gpu_segments = perf.gpu_segment_times_ms();
    if !gpu_segments.is_empty() {
        section_header(ui, "GPU TIMING");

        const SEGMENT_COLORS: [egui::Color32; crate::scene::gpu_timer::SEGMENT_COUNT] = [
            egui::Color32::from_rgb(80, 160, 220),  // Physics & Compute
            egui::Color32::from_rgb(120, 200, 120), // Instance Build & Culling
            egui::Color32::from_rgb(220, 180, 60),  // Opaque Render
            egui::Color32::from_rgb(220, 120, 200), // Skins & Effects
            egui::Color32::from_rgb(220, 100, 100), // Post-Process
        ];

        let total: f32 = gpu_segments.iter().sum();

        // Stacked horizontal bar showing each segment's share of total GPU time
        let bar_height = 14.0;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), bar_height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 2.0, palette().bg_darkest);
        if total > 0.0 {
            let mut x = rect.left();
            for (i, &t) in gpu_segments.iter().enumerate() {
                let w = (t / total) * rect.width();
                if w > 0.0 {
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x, rect.top()),
                            egui::vec2(w, bar_height),
                        ),
                        0.0,
                        SEGMENT_COLORS[i],
                    );
                    x += w;
                }
            }
        }
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, palette().border_subtle),
            egui::StrokeKind::Outside,
        );

        ui.add_space(3.0);

        egui::Grid::new("gpu_timing_grid")
            .num_columns(3)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                for (i, &t) in gpu_segments.iter().enumerate() {
                    let label = crate::scene::gpu_timer::SEGMENT_LABELS
                        .get(i)
                        .copied()
                        .unwrap_or("Segment");
                    let (swatch_rect, _) =
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(swatch_rect, 1.0, SEGMENT_COLORS[i]);
                    ui.label(
                        egui::RichText::new(label)
                            .size(11.0)
                            .color(palette().text_secondary),
                    );
                    ui.label(
                        egui::RichText::new(format!("{:.2} ms", t))
                            .size(11.0)
                            .color(palette().text_primary),
                    );
                    ui.end_row();
                }
                ui.label("");
                ui.label(
                    egui::RichText::new("Total GPU")
                        .size(11.0)
                        .color(palette().text_secondary),
                );
                ui.label(
                    egui::RichText::new(format!("{:.2} ms", total))
                        .size(11.0)
                        .strong()
                        .color(palette().text_primary),
                );
                ui.end_row();
            });

        ui.add_space(6.0);
    }

    // CPU section
    section_header(ui, "CPU");

    egui::Grid::new("cpu_grid")
        .num_columns(2)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Cores")
                    .size(11.0)
                    .color(palette().text_secondary),
            );
            ui.label(
                egui::RichText::new(format!("{}", perf.cpu_core_count()))
                    .size(11.0)
                    .color(palette().text_primary),
            );
            ui.end_row();
            ui.label(
                egui::RichText::new("Usage")
                    .size(11.0)
                    .color(palette().text_secondary),
            );
            ui.label(
                egui::RichText::new(format!("{:.1}%", perf.cpu_usage_total()))
                    .size(11.0)
                    .color(palette().text_primary),
            );
            ui.end_row();
        });

    // Per-core usage bars
    let core_usage = perf.cpu_usage_per_core();
    if !core_usage.is_empty() {
        ui.add_space(3.0);
        let bar_height = 6.0;
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
            painter.rect_filled(bar_rect, 1.0, palette().bg_darkest);
            let usage_width = (usage / 100.0) * rect.width();
            let usage_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left(), y),
                egui::vec2(usage_width, bar_height),
            );
            let color = if usage > 80.0 {
                palette().status_err
            } else if usage > 50.0 {
                palette().status_warn
            } else {
                palette().accent_primary
            };
            painter.rect_filled(usage_rect, 1.0, color);
        }
    }

    ui.add_space(6.0);

    // Memory section
    section_header(ui, "MEMORY");

    let mem_used_mb = perf.memory_used() as f64 / (1024.0 * 1024.0);
    let mem_used_gb = mem_used_mb / 1024.0;

    egui::Grid::new("mem_grid")
        .num_columns(2)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Process RSS")
                    .size(11.0)
                    .color(palette().text_secondary),
            );
            let used_str = if mem_used_gb >= 1.0 {
                format!("{:.2} GB", mem_used_gb)
            } else {
                format!("{:.0} MB", mem_used_mb)
            };
            ui.label(
                egui::RichText::new(used_str)
                    .size(11.0)
                    .color(palette().text_primary),
            );
            ui.end_row();
        });

    // Memory bar - scaled to a fixed reference (4 GB) so the bar is meaningful
    ui.add_space(3.0);
    let reference_gb = 4.0_f64;
    let usage_percent = ((mem_used_gb / reference_gb) * 100.0).min(100.0) as f32;
    let bar_height = 10.0;
    let (response, painter) = ui.allocate_painter(
        egui::vec2(ui.available_width(), bar_height),
        egui::Sense::hover(),
    );
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, palette().bg_darkest);
    painter.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(1.0, palette().border_subtle),
        egui::StrokeKind::Outside,
    );
    let usage_width = (usage_percent / 100.0) * rect.width();
    let usage_rect = egui::Rect::from_min_size(rect.min, egui::vec2(usage_width, bar_height));
    let mem_color = if usage_percent > 90.0 {
        palette().status_err
    } else if usage_percent > 70.0 {
        palette().status_warn
    } else {
        palette().status_info
    };
    painter.rect_filled(usage_rect, 2.0, mem_color);
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        format!("{:.0} MB", mem_used_mb),
        egui::FontId::proportional(9.0),
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
fn render_cave_system(ui: &mut Ui, context: &mut PanelContext, state: &GlobalUiState) {
    let sw = (ui.available_width() - 60.0).max(60.0);
    ui.style_mut().spacing.slider_width = sw;
    ui.separator();

    // Check if we're in GPU mode
    if !context.is_gpu_mode() {
        ui.label("Cave system is only available in GPU mode.");
        return;
    }

    // Store cave generation request in editor state
    let _editor_state = &mut context.editor_state;

    // Check if cave system exists
    let has_cave_system = context
        .scene_manager
        .gpu_scene()
        .map(|s| s.cave_renderer.is_some())
        .unwrap_or(false);

    if !has_cave_system {
        ui.label("Cave system not initialized.");
        ui.add_space(10.0);

        if ui.button("🏔️ Generate Cave System").clicked() {
            // Request cave system initialization
            // This will be handled by the app's render loop
            *context.scene_request = crate::ui::panel_context::SceneModeRequest::Reset;
            // Temporary - will add proper request
        }

        ui.add_space(10.0);
        ui.label(
            "Click the button above to generate a procedural cave system with XPBD collision.",
        );
        return;
    }

    // Cave system exists - show editable parameters
    if let Some(gpu_scene) = context.scene_manager.gpu_scene() {
        if let Some(cave_renderer) = &gpu_scene.cave_renderer {
            let params = cave_renderer.params();
            let editor = &context.editor_state;

            // Create local copies for editing
            // Invert density for UI display so that 1.0 = higher density (lower actual value)
            let mut density = 1.0 - editor.cave_density;
            let mut scale = editor.cave_scale;
            let mut octaves = editor.cave_octaves as i32;
            let mut smoothness = editor.cave_smoothness;
            let mut seed = editor.cave_seed as i32;
            let mut resolution = editor.cave_resolution as i32;
            let mut isolated_chunk_cull_volume = editor.cave_isolated_chunk_cull_volume;
            let mut mesh_smoothing_iterations = editor.cave_mesh_smoothing_iterations as i32;
            let mut mesh_smoothing_factor = editor.cave_mesh_smoothing_factor;
            let mut mesh_smooth_normals = editor.cave_mesh_smooth_normals;
            let mut geothermal_enabled = editor.geothermal_enabled;
            let mut geothermal_count = editor.geothermal_count as i32;
            let mut geothermal_placement_mode = editor.geothermal_placement_mode;
            let mut geothermal_lower_hemisphere = editor.geothermal_lower_hemisphere;
            let mut geothermal_length = editor.geothermal_length;
            let mut geothermal_width = editor.geothermal_width;
            let mut geothermal_depth = editor.geothermal_depth;
            let mut geothermal_back_margin = editor.geothermal_back_margin;
            let mut geothermal_top_margin = editor.geothermal_top_margin;
            let mut geothermal_heat_output = editor.geothermal_heat_output;
            let mut geothermal_heat_radius = editor.geothermal_heat_radius;
            let mut geothermal_glow_strength = editor.geothermal_glow_strength;
            let mut geothermal_glow_radius = editor.geothermal_glow_radius;
            let mut geothermal_glow_color = editor.geothermal_glow_color;
            let mut cave_appearance = editor.cave_appearance;
            let mut cave_rock_dark_color = editor.cave_rock_dark_color;
            let mut cave_rock_cool_color = editor.cave_rock_cool_color;
            let mut cave_rock_warm_color = editor.cave_rock_warm_color;
            let mut cave_rock_pale_color = editor.cave_rock_pale_color;
            let mut cave_rock_layer_scale = editor.cave_rock_layer_scale;
            let mut cave_rock_warp_strength = editor.cave_rock_warp_strength;
            let mut cave_rock_fine_band_strength = editor.cave_rock_fine_band_strength;
            let mut cave_rock_cool_mottle_strength = editor.cave_rock_cool_mottle_strength;
            let mut cave_rock_grain_strength = editor.cave_rock_grain_strength;
            let mut cave_rock_patch_contrast = editor.cave_rock_patch_contrast;
            let mut cave_rock_seam_darkening = editor.cave_rock_seam_darkening;
            let mut cave_rock_wall_line_strength = editor.cave_rock_wall_line_strength;
            let mut cave_rock_min_color = editor.cave_rock_min_color;
            let mut cave_rock_max_color = editor.cave_rock_max_color;
            let mut cave_rock_ambient_strength = editor.cave_rock_ambient_strength;
            let mut cave_rock_diffuse_strength = editor.cave_rock_diffuse_strength;
            let mut cave_rock_specular_strength = editor.cave_rock_specular_strength;
            let mut cave_rock_specular_power = editor.cave_rock_specular_power;
            let mut cave_rock_texture_scale = editor.cave_rock_texture_scale;
            let mut cave_rock_coarse_frequency = editor.cave_rock_coarse_frequency;
            let mut cave_rock_fine_frequency = editor.cave_rock_fine_frequency;
            let mut cave_rock_seam_frequency = editor.cave_rock_seam_frequency;
            let mut cave_rock_fine_noise_scale = editor.cave_rock_fine_noise_scale;
            let mut cave_rock_fine_noise_strength = editor.cave_rock_fine_noise_strength;
            let mut cave_rock_seam_noise_scale = editor.cave_rock_seam_noise_scale;
            let mut cave_rock_seam_noise_strength = editor.cave_rock_seam_noise_strength;
            let mut cave_rock_coarse_band_low = editor.cave_rock_coarse_band_low;
            let mut cave_rock_coarse_band_high = editor.cave_rock_coarse_band_high;
            let mut cave_rock_fine_band_low = editor.cave_rock_fine_band_low;
            let mut cave_rock_fine_band_high = editor.cave_rock_fine_band_high;
            let mut cave_rock_seam_low = editor.cave_rock_seam_low;
            let mut cave_rock_seam_high = editor.cave_rock_seam_high;
            let mut cave_rock_geometry_conform = editor.cave_rock_geometry_conform;
            let mut cave_rock_parallax_depth = editor.cave_rock_parallax_depth;

            let mut params_changed = false;

            ui.heading("Generation Parameters");
            ui.add_space(5.0);

            // Editable parameters
            ui.add_space(4.0);
            ui.label("Density:")
                .on_hover_text("How much of the world is filled with cave rock. Higher values create denser, more enclosed caves with less open space");
            params_changed |= ui
                .add(egui::Slider::new(&mut density, 0.01..=1.0))
                .changed();

            ui.add_space(4.0);
            ui.label("Scale:")
                .on_hover_text("Size of the cave features. Higher values create larger, more open chambers; lower values create tighter, more intricate passages");
            params_changed |= ui
                .add(egui::Slider::new(&mut scale, 50.0..=100.0))
                .changed();

            if state.show_advanced_options {
                ui.add_space(4.0);
                ui.label("Octaves:")
                    .on_hover_text("Number of noise layers combined to generate the cave shape. More octaves add finer detail and rougher surfaces");
                params_changed |= ui.add(egui::Slider::new(&mut octaves, 1..=8)).changed();

                ui.add_space(4.0);
                ui.label("Smoothness:").on_hover_text(
                    "How smooth the cave walls are. 0 = rough and jagged; 1 = smooth and rounded",
                );
                params_changed |= ui
                    .add(egui::Slider::new(&mut smoothness, 0.0..=1.0))
                    .changed();

                ui.add_space(4.0);
                ui.label("Resolution:")
                    .on_hover_text("Voxel grid resolution for the cave mesh. Higher values produce more detailed geometry but take longer to generate and use more memory");
                params_changed |= ui
                    .add(egui::Slider::new(&mut resolution, 32..=128))
                    .changed();

                ui.add_space(4.0);
                ui.label("Fragment Cull:")
                    .on_hover_text("Minimum isolated fragment volume to keep. Higher values remove larger floating cave chunks; 0 disables fragment cleanup");
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut isolated_chunk_cull_volume,
                        0.0..=1000000.0,
                    ))
                    .changed();

                ui.add_space(4.0);
                ui.label("Mesh Smoothing Iterations:")
                    .on_hover_text("Number of Laplacian smoothing passes applied to the cave mesh after marching cubes. 0 disables smoothing; higher values round off the blocky voxel surface more");
                params_changed |= ui
                    .add(egui::Slider::new(&mut mesh_smoothing_iterations, 0..=8))
                    .changed();

                ui.add_space(4.0);
                ui.label("Mesh Smoothing Factor:")
                    .on_hover_text("How strongly each smoothing pass pulls vertices toward their neighbors' average position. 0 = no effect; 1 = fully snap to the average");
                params_changed |= ui
                    .add(egui::Slider::new(&mut mesh_smoothing_factor, 0.0..=1.0))
                    .changed();

                ui.add_space(4.0);
                params_changed |= ui
                    .checkbox(&mut mesh_smooth_normals, "Smooth Normals")
                    .on_hover_text(
                        "Average normals across shared cave vertices. Softens faceted lighting without moving the cave mesh.",
                    )
                    .changed();
            }

            ui.add_space(4.0);
            ui.label("Seed:");
            ui.horizontal(|ui| {
                let mut seed_text = seed.to_string();
                if ui
                    .add_sized([80.0, 20.0], egui::TextEdit::singleline(&mut seed_text))
                    .changed()
                {
                    if let Ok(parsed_seed) = seed_text.parse::<i32>() {
                        if parsed_seed >= 0 && parsed_seed <= 9999 {
                            seed = parsed_seed;
                            params_changed = true;
                        }
                    }
                }
                if ui
                    .button("🎲 Randomize")
                    .on_hover_text("Generate random seed")
                    .clicked()
                {
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
            ui.heading("Appearance");
            ui.add_space(5.0);

            ui.label("Preset:")
                .on_hover_text("Named cave appearance preset. More appearances can be added here without changing the rest of the panel");
            let previous_cave_appearance = cave_appearance;
            egui::ComboBox::from_id_salt("cave_appearance")
                .selected_text(match cave_appearance {
                    1 => "Lava Tubes",
                    _ => "Layered Shale",
                })
                .show_ui(ui, |ui| {
                    params_changed |= ui
                        .selectable_value(&mut cave_appearance, 0, "Layered Shale")
                        .changed();
                    params_changed |= ui
                        .selectable_value(&mut cave_appearance, 1, "Lava Tubes")
                        .changed();
                });
            if cave_appearance != previous_cave_appearance {
                let previous_visuals = CaveAppearanceVisualSettings {
                    dark_color: cave_rock_dark_color,
                    cool_color: cave_rock_cool_color,
                    warm_color: cave_rock_warm_color,
                    pale_color: cave_rock_pale_color,
                    layer_scale: cave_rock_layer_scale,
                    warp_strength: cave_rock_warp_strength,
                    fine_band_strength: cave_rock_fine_band_strength,
                    cool_mottle_strength: cave_rock_cool_mottle_strength,
                    grain_strength: cave_rock_grain_strength,
                    patch_contrast: cave_rock_patch_contrast,
                    seam_darkening: cave_rock_seam_darkening,
                    wall_line_strength: cave_rock_wall_line_strength,
                    min_color: cave_rock_min_color,
                    max_color: cave_rock_max_color,
                    ambient_strength: cave_rock_ambient_strength,
                    diffuse_strength: cave_rock_diffuse_strength,
                    specular_strength: cave_rock_specular_strength,
                    specular_power: cave_rock_specular_power,
                    texture_scale: cave_rock_texture_scale,
                    coarse_frequency: cave_rock_coarse_frequency,
                    fine_frequency: cave_rock_fine_frequency,
                    seam_frequency: cave_rock_seam_frequency,
                    fine_noise_scale: cave_rock_fine_noise_scale,
                    fine_noise_strength: cave_rock_fine_noise_strength,
                    seam_noise_scale: cave_rock_seam_noise_scale,
                    seam_noise_strength: cave_rock_seam_noise_strength,
                    coarse_band_low: cave_rock_coarse_band_low,
                    coarse_band_high: cave_rock_coarse_band_high,
                    fine_band_low: cave_rock_fine_band_low,
                    fine_band_high: cave_rock_fine_band_high,
                    seam_low: cave_rock_seam_low,
                    seam_high: cave_rock_seam_high,
                    geometry_conform: cave_rock_geometry_conform,
                    parallax_depth: cave_rock_parallax_depth,
                };
                if previous_cave_appearance == 1 {
                    context.editor_state.cave_lava_tube_visuals = previous_visuals;
                } else {
                    context.editor_state.cave_layered_shale_visuals = previous_visuals;
                }

                let selected_visuals = if cave_appearance == 1 {
                    context.editor_state.cave_lava_tube_visuals
                } else {
                    context.editor_state.cave_layered_shale_visuals
                };
                cave_rock_dark_color = selected_visuals.dark_color;
                cave_rock_cool_color = selected_visuals.cool_color;
                cave_rock_warm_color = selected_visuals.warm_color;
                cave_rock_pale_color = selected_visuals.pale_color;
                cave_rock_layer_scale = selected_visuals.layer_scale;
                cave_rock_warp_strength = selected_visuals.warp_strength;
                cave_rock_fine_band_strength = selected_visuals.fine_band_strength;
                cave_rock_cool_mottle_strength = selected_visuals.cool_mottle_strength;
                cave_rock_grain_strength = selected_visuals.grain_strength;
                cave_rock_patch_contrast = selected_visuals.patch_contrast;
                cave_rock_seam_darkening = selected_visuals.seam_darkening;
                cave_rock_wall_line_strength = selected_visuals.wall_line_strength;
                cave_rock_min_color = selected_visuals.min_color;
                cave_rock_max_color = selected_visuals.max_color;
                cave_rock_ambient_strength = selected_visuals.ambient_strength;
                cave_rock_diffuse_strength = selected_visuals.diffuse_strength;
                cave_rock_specular_strength = selected_visuals.specular_strength;
                cave_rock_specular_power = selected_visuals.specular_power;
                cave_rock_texture_scale = selected_visuals.texture_scale;
                cave_rock_coarse_frequency = selected_visuals.coarse_frequency;
                cave_rock_fine_frequency = selected_visuals.fine_frequency;
                cave_rock_seam_frequency = selected_visuals.seam_frequency;
                cave_rock_fine_noise_scale = selected_visuals.fine_noise_scale;
                cave_rock_fine_noise_strength = selected_visuals.fine_noise_strength;
                cave_rock_seam_noise_scale = selected_visuals.seam_noise_scale;
                cave_rock_seam_noise_strength = selected_visuals.seam_noise_strength;
                cave_rock_coarse_band_low = selected_visuals.coarse_band_low;
                cave_rock_coarse_band_high = selected_visuals.coarse_band_high;
                cave_rock_fine_band_low = selected_visuals.fine_band_low;
                cave_rock_fine_band_high = selected_visuals.fine_band_high;
                cave_rock_seam_low = selected_visuals.seam_low;
                cave_rock_seam_high = selected_visuals.seam_high;
                cave_rock_geometry_conform = selected_visuals.geometry_conform;
                cave_rock_parallax_depth = selected_visuals.parallax_depth;
                params_changed = true;
            }

            ui.add_space(4.0);
            let lava_tubes_selected = cave_appearance == 1;
            ui.label(if lava_tubes_selected {
                "Lava Tube Colors"
            } else {
                "Rock Colors"
            });
            ui.horizontal(|ui| {
                ui.label(if lava_tubes_selected {
                    "Basalt:"
                } else {
                    "Dark:"
                });
                params_changed |= ui
                    .color_edit_button_rgb(&mut cave_rock_dark_color)
                    .on_hover_text(if lava_tubes_selected {
                        "Glassy black basalt base color"
                    } else {
                        "Dark base rock tone"
                    })
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label(if lava_tubes_selected {
                    "Glass:"
                } else {
                    "Cool:"
                });
                params_changed |= ui
                    .color_edit_button_rgb(&mut cave_rock_cool_color)
                    .on_hover_text(if lava_tubes_selected {
                        "Blue-black cooled glass tone blended through the lava skin"
                    } else {
                        "Cool mottled slate tone blended through the wall texture"
                    })
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label(if lava_tubes_selected {
                    "Iron:"
                } else {
                    "Warm:"
                });
                params_changed |= ui
                    .color_edit_button_rgb(&mut cave_rock_warm_color)
                    .on_hover_text(if lava_tubes_selected {
                        "Oxidized iron tone for blistered patches"
                    } else {
                        "Warm shale tone used by the broad sediment layers"
                    })
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label(if lava_tubes_selected { "Ash:" } else { "Pale:" });
                params_changed |= ui
                    .color_edit_button_rgb(&mut cave_rock_pale_color)
                    .on_hover_text(if lava_tubes_selected {
                        "Dry ash and mineral dust color for deposits and worn shelves"
                    } else {
                        "Light silt tone used by fine strata highlights"
                    })
                    .changed();
            });

            if state.show_advanced_options {
                ui.add_space(4.0);
                ui.label(if lava_tubes_selected {
                    "Lava Flow"
                } else {
                    "Layering"
                });
                ui.label(if lava_tubes_selected {
                    "Flow Scale:"
                } else {
                    "Layer Scale:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "World-space scale of the directional lava-flow coordinate"
                } else {
                    "Vertical frequency of the sediment layer pattern"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_layer_scale, 0.0..=0.2))
                    .changed();
                ui.label("Warp:").on_hover_text(if lava_tubes_selected {
                    "How much procedural noise bends and sags the lava flow"
                } else {
                    "How much procedural noise bends the sediment bands"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_warp_strength, 0.0..=4.0))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Plate Variation:"
                } else {
                    "Fine Bands:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Strength of ash-gray variation across cooled basalt plates"
                } else {
                    "Strength of the pale fine-layer highlights"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_fine_band_strength,
                        0.0..=1.0,
                    ))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Blistering:"
                } else {
                    "Cool Mottle:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Coverage of oxidized blister patches in the cooled lava skin"
                } else {
                    "Amount of cool slate color variation"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_cool_mottle_strength,
                        0.0..=1.0,
                    ))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Glass Grain:"
                } else {
                    "Grain:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Brightness variation in the glassy basalt surface"
                } else {
                    "Brightness variation from the triplanar rock grain"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_grain_strength, 0.0..=0.25))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Ash Deposits:"
                } else {
                    "Patch Contrast:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Amount of pale ash and mineral dust collected on lower surfaces"
                } else {
                    "Large mottled light and dark patch contrast"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_patch_contrast, 0.0..=0.5))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Seam Darkness:"
                } else {
                    "Seam Darkening:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Darkness of cooled contraction cracks between basalt plates"
                } else {
                    "Darkness of narrow sediment seams"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_seam_darkening, 0.0..=1.0))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Shelf Scuffs:"
                } else {
                    "Wall Lines:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Strength of pale abrasion marks along tube-wall shelves and rub zones"
                } else {
                    "How strongly steep walls keep crisp sediment lines"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_wall_line_strength,
                        0.0..=1.0,
                    ))
                    .changed();

                ui.add_space(4.0);
                ui.label(if lava_tubes_selected {
                    "Tube Pattern"
                } else {
                    "Pattern Layout"
                });
                ui.label("Texture Scale:")
                    .on_hover_text(if lava_tubes_selected {
                        "Scale of the glassy surface noise. Higher values create finer cooled-lava grain"
                    } else {
                        "Scale of the triplanar rock grain. Higher values create finer, denser surface texture"
                    });
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_texture_scale, 0.005..=0.2)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label("Conform:")
                    .on_hover_text(if lava_tubes_selected {
                        "Used by other rock presets; Lava Tubes derives flow from the tube radius and surface direction"
                    } else {
                        "Blends strata from world-horizontal height lines toward lines that follow the local cave surface normal"
                    });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_geometry_conform,
                        0.0..=1.0,
                    ))
                    .changed();
                ui.label("Relief:")
                    .on_hover_text(if lava_tubes_selected {
                        "Stable shading depth for the cooled lava grain and rope ridges"
                    } else {
                        "Stable shading depth for the rock grain and strata. This adds surface relief without sliding the pattern toward the camera"
                    });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_parallax_depth, 0.0..=4.0))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Plate Size:"
                } else {
                    "Major Lines:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Density of polygonal cooling plates. Higher values create smaller basalt plates"
                } else {
                    "Density of broad sediment bands"
                });
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_coarse_frequency, 0.5..=32.0)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Surface Breakup:"
                } else {
                    "Fine Lines:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Secondary breakup used by subtle glass and plate variation"
                } else {
                    "Density of thin pale sediment lines"
                });
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_fine_frequency, 1.0..=96.0)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Crack Density:"
                } else {
                    "Seam Lines:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Density of cooled contraction cracks between basalt plates"
                } else {
                    "Density of dark seam lines"
                });
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_seam_frequency, 1.0..=64.0)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Flow Noise Scale:"
                } else {
                    "Fine Warp Scale:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Spatial scale of lava-flow distortion and glass grain"
                } else {
                    "Spatial scale of distortion on fine lines"
                });
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_fine_noise_scale, 0.005..=0.2)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Flow Warp:"
                } else {
                    "Fine Warp:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "How strongly the cooled lava flow bends and swirls"
                } else {
                    "How strongly fine lines bend and ripple"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_fine_noise_strength,
                        0.0..=12.0,
                    ))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Crack Noise Scale:"
                } else {
                    "Seam Warp Scale:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Spatial scale of distortion on cooled contraction cracks"
                } else {
                    "Spatial scale of distortion on dark seams"
                });
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_seam_noise_scale, 0.005..=0.2)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Crack Warp:"
                } else {
                    "Seam Warp:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "How strongly contraction cracks bend and break"
                } else {
                    "How strongly seam lines bend and break"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_seam_noise_strength,
                        0.0..=12.0,
                    ))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Plate Low:"
                } else {
                    "Major Low:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Reserved for shale bands; Lava Tubes uses crack thickness controls below"
                } else {
                    "Lower smoothstep edge for broad sediment bands"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_coarse_band_low,
                        -1.0..=1.0,
                    ))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Plate High:"
                } else {
                    "Major High:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Reserved for shale bands; Lava Tubes uses crack thickness controls below"
                } else {
                    "Upper smoothstep edge for broad sediment bands"
                });
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_coarse_band_high,
                        -1.0..=1.0,
                    ))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Blister Low:"
                } else {
                    "Fine Low:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Lower smoothstep edge for oxidized blister coverage"
                } else {
                    "Lower smoothstep edge for pale fine-line coverage"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_fine_band_low, -1.0..=1.0))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Blister High:"
                } else {
                    "Fine High:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Upper smoothstep edge for oxidized blister coverage"
                } else {
                    "Upper smoothstep edge for pale fine-line coverage"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_fine_band_high, -1.0..=1.0))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Crack Thin:"
                } else {
                    "Seam Low:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Lower edge for cooled contraction crack thickness"
                } else {
                    "Lower edge for dark seam thickness"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_seam_low, 0.0..=1.0))
                    .changed();
                ui.label(if lava_tubes_selected {
                    "Crack Thick:"
                } else {
                    "Seam High:"
                })
                .on_hover_text(if lava_tubes_selected {
                    "Upper edge for cooled contraction crack thickness"
                } else {
                    "Upper edge for dark seam thickness"
                });
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_seam_high, 0.0..=1.0))
                    .changed();

                ui.add_space(4.0);
                ui.label("Lighting");
                ui.label("Ambient:");
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_ambient_strength,
                        0.0..=0.5,
                    ))
                    .changed();
                ui.label("Diffuse:");
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_diffuse_strength,
                        0.0..=2.0,
                    ))
                    .changed();
                ui.label("Specular:");
                params_changed |= ui
                    .add(egui::Slider::new(
                        &mut cave_rock_specular_strength,
                        0.0..=2.0,
                    ))
                    .changed();
                ui.label("Spec Power:");
                params_changed |= ui
                    .add(
                        egui::Slider::new(&mut cave_rock_specular_power, 1.0..=128.0)
                            .logarithmic(true),
                    )
                    .changed();
                ui.label("Min Tone:");
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_min_color, 0.0..=0.25))
                    .changed();
                ui.label("Max Tone:");
                params_changed |= ui
                    .add(egui::Slider::new(&mut cave_rock_max_color, 0.25..=1.0))
                    .changed();
            }

            ui.add_space(10.0);
            ui.separator();
            ui.heading("Thermal Smoke Stacks");
            ui.add_space(5.0);

            params_changed |= ui
                .checkbox(&mut geothermal_enabled, "Enable Vents")
                .on_hover_text(
                    "Generate procedural thermal stacks on the world sphere boundary that emit inward heat plumes and warm colored light",
                )
                .changed();

            if geothermal_enabled {
                ui.add_space(4.0);
                ui.label("Placement:")
                    .on_hover_text("Choose whether smoke stacks grow from the world sphere boundary or from interior cave wall voxels");
                egui::ComboBox::from_id_salt("geothermal_placement_mode")
                    .selected_text(match geothermal_placement_mode {
                        1 => "Cave Walls",
                        _ => "World Boundary",
                    })
                    .show_ui(ui, |ui| {
                        params_changed |= ui
                            .selectable_value(&mut geothermal_placement_mode, 0, "World Boundary")
                            .changed();
                        params_changed |= ui
                            .selectable_value(&mut geothermal_placement_mode, 1, "Cave Walls")
                            .changed();
                    });

                params_changed |= ui
                    .checkbox(&mut geothermal_lower_hemisphere, "Lower Hemisphere Only")
                    .on_hover_text(
                        "Restrict placement to the lower half according to the active directional gravity axis. Radial gravity ignores this filter",
                    )
                    .changed();

                ui.add_space(4.0);
                ui.label("Frequency:")
                    .on_hover_text("Target number of smoke stacks to place on eligible boundary or cave wall voxels");
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_count, 0..=64))
                    .changed();

                ui.add_space(6.0);
                ui.label("Shape");
                ui.add_space(2.0);

                ui.label("Length:").on_hover_text(
                    "How far each smoke stack footprint runs along the sphere boundary, in voxels",
                );
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_length, 1.0..=32.0))
                    .changed();

                ui.label("Width:")
                    .on_hover_text("Approximate radius of the chimney. The base flares wider and conforms to the sphere boundary");
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_width, 1.0..=10.0))
                    .changed();

                ui.label("Depth:")
                    .on_hover_text("Height of the hollow smoke-stack chimney rising inward from a closed, flared base. Glow is placed deep in the shaft so it peeks out through the opening");
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_depth, 1.0..=20.0))
                    .changed();

                if state.show_advanced_options {
                    ui.label("Back Margin:")
                        .on_hover_text("Reserved for future placement filtering. Boundary smoke stacks use the sphere surface instead of cave wall thickness");
                    params_changed |= ui
                        .add(egui::Slider::new(&mut geothermal_back_margin, 0.0..=20.0))
                        .changed();

                    ui.label("Top Margin:")
                        .on_hover_text("Reserved for future placement filtering. Boundary smoke stacks are independent of gravity direction");
                    params_changed |= ui
                        .add(egui::Slider::new(&mut geothermal_top_margin, 0.0..=12.0))
                        .changed();
                }

                ui.add_space(6.0);
                ui.label("Heat");
                ui.add_space(2.0);

                ui.label("Heat Output:").on_hover_text(
                    "Directional heat injected inward from the vent plume, in degrees Celsius",
                );
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_heat_output, 0.0..=120.0).suffix(" C"))
                    .changed();

                ui.label("Heat Radius:")
                    .on_hover_text("Distance and spread of the baked inward heat plume, in voxels");
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_heat_radius, 1.0..=32.0))
                    .changed();

                ui.add_space(6.0);
                ui.label("Glow");
                ui.add_space(2.0);

                ui.horizontal(|ui| {
                    ui.label("Color:");
                    params_changed |= ui
                        .color_edit_button_rgb(&mut geothermal_glow_color)
                        .on_hover_text("RGB glow emitted from the smoke stack footprint into the local light field")
                        .changed();
                });

                ui.label("Glow Strength:").on_hover_text(
                    "Brightness of the colored glow injected into the local light field",
                );
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_glow_strength, 0.0..=8.0))
                    .changed();

                ui.label("Glow Radius:").on_hover_text(
                    "Radius of the baked colored glow around the lowest crevice point, in voxels",
                );
                params_changed |= ui
                    .add(egui::Slider::new(&mut geothermal_glow_radius, 1.0..=32.0))
                    .changed();
            }

            if state.show_advanced_options {
                ui.add_space(10.0);
                ui.separator();
                ui.label(format!("Triangle Count: {}", params.triangle_count));
            }

            ui.add_space(10.0);

            // Apply changes whenever any parameter changed
            if params_changed {
                // Store changes in editor state for the app to apply
                // Invert density so that 1.0 = higher density (lower actual value)
                context.editor_state.cave_density = 1.0 - density;
                context.editor_state.cave_scale = scale;
                context.editor_state.cave_octaves = octaves as u32;
                context.editor_state.cave_smoothness = smoothness;
                context.editor_state.cave_seed = seed as u32;
                context.editor_state.cave_resolution = resolution as u32;
                context.editor_state.cave_isolated_chunk_cull_volume = isolated_chunk_cull_volume;
                context.editor_state.cave_mesh_smoothing_iterations =
                    mesh_smoothing_iterations.max(0) as u32;
                context.editor_state.cave_mesh_smoothing_factor = mesh_smoothing_factor;
                context.editor_state.cave_mesh_smooth_normals = mesh_smooth_normals;
                context.editor_state.geothermal_enabled = geothermal_enabled;
                context.editor_state.geothermal_count = geothermal_count.max(0) as u32;
                context.editor_state.geothermal_placement_mode = geothermal_placement_mode.min(1);
                context.editor_state.geothermal_lower_hemisphere = geothermal_lower_hemisphere;
                context.editor_state.geothermal_length = geothermal_length;
                context.editor_state.geothermal_width = geothermal_width;
                context.editor_state.geothermal_depth = geothermal_depth;
                context.editor_state.geothermal_back_margin = geothermal_back_margin;
                context.editor_state.geothermal_top_margin = geothermal_top_margin;
                context.editor_state.geothermal_heat_output = geothermal_heat_output;
                context.editor_state.geothermal_heat_radius = geothermal_heat_radius;
                context.editor_state.geothermal_glow_strength = geothermal_glow_strength;
                context.editor_state.geothermal_glow_radius = geothermal_glow_radius;
                context.editor_state.geothermal_glow_color = geothermal_glow_color;
                context.editor_state.cave_appearance = cave_appearance;
                context.editor_state.cave_rock_dark_color = cave_rock_dark_color;
                context.editor_state.cave_rock_cool_color = cave_rock_cool_color;
                context.editor_state.cave_rock_warm_color = cave_rock_warm_color;
                context.editor_state.cave_rock_pale_color = cave_rock_pale_color;
                context.editor_state.cave_rock_layer_scale = cave_rock_layer_scale;
                context.editor_state.cave_rock_warp_strength = cave_rock_warp_strength;
                context.editor_state.cave_rock_fine_band_strength = cave_rock_fine_band_strength;
                context.editor_state.cave_rock_cool_mottle_strength =
                    cave_rock_cool_mottle_strength;
                context.editor_state.cave_rock_grain_strength = cave_rock_grain_strength;
                context.editor_state.cave_rock_patch_contrast = cave_rock_patch_contrast;
                context.editor_state.cave_rock_seam_darkening = cave_rock_seam_darkening;
                context.editor_state.cave_rock_wall_line_strength = cave_rock_wall_line_strength;
                context.editor_state.cave_rock_min_color = cave_rock_min_color;
                context.editor_state.cave_rock_max_color = cave_rock_max_color;
                context.editor_state.cave_rock_ambient_strength = cave_rock_ambient_strength;
                context.editor_state.cave_rock_diffuse_strength = cave_rock_diffuse_strength;
                context.editor_state.cave_rock_specular_strength = cave_rock_specular_strength;
                context.editor_state.cave_rock_specular_power = cave_rock_specular_power;
                context.editor_state.cave_rock_texture_scale = cave_rock_texture_scale;
                context.editor_state.cave_rock_coarse_frequency = cave_rock_coarse_frequency;
                context.editor_state.cave_rock_fine_frequency = cave_rock_fine_frequency;
                context.editor_state.cave_rock_seam_frequency = cave_rock_seam_frequency;
                context.editor_state.cave_rock_fine_noise_scale = cave_rock_fine_noise_scale;
                context.editor_state.cave_rock_fine_noise_strength = cave_rock_fine_noise_strength;
                context.editor_state.cave_rock_seam_noise_scale = cave_rock_seam_noise_scale;
                context.editor_state.cave_rock_seam_noise_strength = cave_rock_seam_noise_strength;
                context.editor_state.cave_rock_coarse_band_low = cave_rock_coarse_band_low;
                context.editor_state.cave_rock_coarse_band_high = cave_rock_coarse_band_high;
                context.editor_state.cave_rock_fine_band_low = cave_rock_fine_band_low;
                context.editor_state.cave_rock_fine_band_high = cave_rock_fine_band_high;
                context.editor_state.cave_rock_seam_low = cave_rock_seam_low;
                context.editor_state.cave_rock_seam_high = cave_rock_seam_high;
                context.editor_state.cave_rock_geometry_conform = cave_rock_geometry_conform;
                context.editor_state.cave_rock_parallax_depth = cave_rock_parallax_depth;
                let active_visuals =
                    CaveAppearanceVisualSettings::from_editor_state(context.editor_state);
                if cave_appearance == 1 {
                    context.editor_state.cave_lava_tube_visuals = active_visuals;
                } else {
                    context.editor_state.cave_layered_shale_visuals = active_visuals;
                }
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
            if show_moss {
                gpu_scene.show_moss = true;
            } else {
                gpu_scene.disable_moss();
            }
            context.editor_state.show_moss = show_moss;
            moss_changed = true;
        }

        if show_moss {
            if let Some(ref mut moss) = gpu_scene.moss_system {
                ui.add_space(5.0);
                ui.label("Growth & Spread");
                ui.add_space(3.0);

                ui.label("Growth Rate:")
                    .on_hover_text("How quickly moss spreads to adjacent voxels per second. Logarithmic scale — small values create slow, patchy growth; large values fill surfaces rapidly");
                if ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.moss_growth_rate, 0.001..=1.0)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    moss.growth_rate = context.editor_state.moss_growth_rate;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Min Light Level:")
                    .on_hover_text("Minimum light intensity required for moss to grow in a voxel. Moss won't grow in completely dark areas below this threshold");
                if ui
                    .add(egui::Slider::new(
                        &mut context.editor_state.moss_min_light,
                        0.0..=0.5,
                    ))
                    .changed()
                {
                    moss.min_light = context.editor_state.moss_min_light;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Water Radius (voxels):")
                    .on_hover_text("Radius around water voxels where moss can grow. Moss requires proximity to water — larger values allow moss to grow further from water sources");
                if ui
                    .add(egui::Slider::new(
                        &mut context.editor_state.moss_water_radius,
                        2.0..=50.0,
                    ))
                    .changed()
                {
                    moss.water_radius = context.editor_state.moss_water_radius;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Decay Rate:")
                    .on_hover_text("How quickly moss dies when conditions are unfavorable (too dark or too dry). Logarithmic scale");
                if ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.moss_decay_rate, 0.001..=0.5)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    moss.decay_rate = context.editor_state.moss_decay_rate;
                    moss_changed = true;
                }

                ui.add_space(8.0);
                ui.label("Erosion");
                ui.add_space(3.0);

                ui.label("Water Erosion Rate:")
                    .on_hover_text("How much flowing water erodes moss. Higher values cause moss to be washed away more quickly by water currents");
                if ui
                    .add(egui::Slider::new(
                        &mut context.editor_state.moss_erosion_rate,
                        0.0..=2.0,
                    ))
                    .changed()
                {
                    moss.erosion_rate = context.editor_state.moss_erosion_rate;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Wetness Evaporation:")
                    .on_hover_text("How quickly surface wetness dries out. Higher values mean moss dries faster and needs more frequent water contact to survive");
                if ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.moss_wetness_evaporation,
                            0.001..=0.2,
                        )
                        .logarithmic(true),
                    )
                    .changed()
                {
                    moss.wetness_evaporation = context.editor_state.moss_wetness_evaporation;
                    moss_changed = true;
                }

                ui.add_space(8.0);
                ui.label("Consumption (Phagocytes)");
                ui.add_space(3.0);

                ui.label("Graze Cooldown:")
                    .on_hover_text("Seconds a phagocyte must wait before it can graze the same moss voxel again. Prevents a single cell from instantly consuming all nearby moss");
                if ui
                    .add(egui::Slider::new(
                        &mut context.editor_state.moss_consume_rate,
                        1.0..=30.0,
                    ))
                    .changed()
                {
                    moss.graze_cooldown = context.editor_state.moss_consume_rate;
                    moss_changed = true;
                }

                ui.add_space(2.0);
                ui.label("Nutrients per Moss:")
                    .on_hover_text("How many nutrients a phagocyte gains from grazing one moss voxel. Higher values make moss a richer food source");
                if ui
                    .add(egui::Slider::new(
                        &mut context.editor_state.moss_nutrient_per_moss,
                        1.0..=200.0,
                    ))
                    .changed()
                {
                    moss.nutrient_per_moss = context.editor_state.moss_nutrient_per_moss;
                    moss_changed = true;
                }

                if state.show_advanced_options {
                    ui.add_space(8.0);
                    ui.label("Appearance");
                    ui.add_space(3.0);

                    ui.label("Texture Scale:")
                        .on_hover_text("UV scale of the moss texture on cave surfaces. Smaller values = larger texture tiles; larger values = finer, more detailed texture");
                    if ui
                        .add(egui::Slider::new(
                            &mut context.editor_state.moss_scale,
                            0.02..=0.5,
                        ))
                        .changed()
                    {
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }

                    ui.add_space(2.0);
                    ui.label("Parallax Depth:")
                        .on_hover_text("Strength of the parallax occlusion mapping effect on moss. Higher values make the moss appear to have more 3D depth and volume");
                    if ui
                        .add(egui::Slider::new(
                            &mut context.editor_state.moss_parallax_depth,
                            0.0..=0.3,
                        ))
                        .changed()
                    {
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
                    if ui
                        .add(
                            egui::Slider::new(
                                &mut context.editor_state.moss_noise_frequency,
                                4.0..=80.0,
                            )
                            .logarithmic(true),
                        )
                        .changed()
                    {
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }

                    ui.add_space(2.0);
                    ui.label("Noise Lacunarity:");
                    if ui
                        .add(egui::Slider::new(
                            &mut context.editor_state.moss_noise_lacunarity,
                            1.5..=5.0,
                        ))
                        .changed()
                    {
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }

                    ui.add_space(2.0);
                    ui.label("Height Sharpness (low):");
                    if ui
                        .add(egui::Slider::new(
                            &mut context.editor_state.moss_height_sharpness_low,
                            0.0..=0.5,
                        ))
                        .changed()
                    {
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }

                    ui.add_space(2.0);
                    ui.label("Height Sharpness (high):");
                    if ui
                        .add(egui::Slider::new(
                            &mut context.editor_state.moss_height_sharpness_high,
                            0.3..=1.0,
                        ))
                        .changed()
                    {
                        context.editor_state.light_params_dirty = true;
                        moss_changed = true;
                    }

                    ui.add_space(2.0);
                    ui.label("Bump Strength:");
                    if ui
                        .add(egui::Slider::new(
                            &mut context.editor_state.moss_bump_strength,
                            0.0..=15.0,
                        ))
                        .changed()
                    {
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
                } // end advanced moss appearance
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

    // -- Boulder Section ------------------------------------------------------
    ui.add_space(10.0);
    ui.separator();
    ui.heading("🪨 Mossrocks");
    ui.add_space(5.0);

    if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
        let mut show_boulders = gpu_scene.show_boulders;
        if ui
            .checkbox(&mut show_boulders, "Enable Mossrocks")
            .changed()
        {
            gpu_scene.show_boulders = show_boulders;
            context.editor_state.show_boulders = show_boulders;
            context.editor_state.save_cave_settings();
        }

        if show_boulders {
            ui.add_space(5.0);
            let mut boulder_changed = false;

            // Target count
            let mut target = gpu_scene.boulder_target_count as i32;
            ui.label("Target Count:")
                .on_hover_text("How many mossrocks the system tries to maintain in the world at any time. New rocks spawn when the count falls below this");
            if ui.add(egui::Slider::new(&mut target, 1..=128)).changed() {
                gpu_scene.boulder_target_count = target as u32;
                if let Some(ref mut bs) = gpu_scene.boulder_system {
                    bs.target_count = target as u32;
                }
                boulder_changed = true;
            }

            ui.add_space(3.0);

            // Spawn frequency
            let mut interval = gpu_scene.boulder_spawn_interval;
            ui.label("Approx. Spawn Interval (seconds):")
                .on_hover_text("Average time between new mossrock spawns. Logarithmic scale — small values spawn rocks rapidly; large values create rare spawns");
            if ui
                .add(egui::Slider::new(&mut interval, 0.5..=60.0).logarithmic(true))
                .changed()
            {
                gpu_scene.boulder_spawn_interval = interval;
                if let Some(ref mut bs) = gpu_scene.boulder_system {
                    bs.spawn_interval = interval;
                }
                boulder_changed = true;
            }

            ui.add_space(5.0);
            ui.label("Radius Range:");
            ui.horizontal(|ui| {
                ui.label("Min");
                let mut rmin = gpu_scene.boulder_radius_min;
                let rmax_cur = gpu_scene.boulder_radius_max;
                if ui
                    .add(
                        egui::DragValue::new(&mut rmin)
                            .range(0.5..=rmax_cur - 0.5)
                            .speed(0.1)
                            .suffix(" u"),
                    )
                    .changed()
                {
                    gpu_scene.boulder_radius_min = rmin;
                    if let Some(ref mut bs) = gpu_scene.boulder_system {
                        bs.radius_min = rmin;
                    }
                    boulder_changed = true;
                }
                ui.label("–  Max");
                let mut rmax = gpu_scene.boulder_radius_max;
                let rmin_cur = gpu_scene.boulder_radius_min;
                if ui
                    .add(
                        egui::DragValue::new(&mut rmax)
                            .range(rmin_cur + 0.5..=30.0)
                            .speed(0.1)
                            .suffix(" u"),
                    )
                    .changed()
                {
                    gpu_scene.boulder_radius_max = rmax;
                    if let Some(ref mut bs) = gpu_scene.boulder_system {
                        bs.radius_max = rmax;
                    }
                    boulder_changed = true;
                }
            });

            if state.show_advanced_options {
                ui.add_space(3.0);
                ui.label("Moss Store Range (nutrients):");
                ui.horizontal(|ui| {
                    ui.label("Min");
                    let mut mmin = gpu_scene.boulder_moss_min;
                    let mmax_cur = gpu_scene.boulder_moss_max;
                    if ui
                        .add(
                            egui::DragValue::new(&mut mmin)
                                .range(100.0..=mmax_cur - 100.0)
                                .speed(100.0),
                        )
                        .changed()
                    {
                        gpu_scene.boulder_moss_min = mmin;
                        if let Some(ref mut bs) = gpu_scene.boulder_system {
                            bs.moss_min = mmin;
                        }
                        boulder_changed = true;
                    }
                    ui.label("–  Max");
                    let mut mmax = gpu_scene.boulder_moss_max;
                    let mmin_cur = gpu_scene.boulder_moss_min;
                    if ui
                        .add(
                            egui::DragValue::new(&mut mmax)
                                .range(mmin_cur + 100.0..=500_000.0)
                                .speed(500.0),
                        )
                        .changed()
                    {
                        gpu_scene.boulder_moss_max = mmax;
                        if let Some(ref mut bs) = gpu_scene.boulder_system {
                            bs.moss_max = mmax;
                        }
                        boulder_changed = true;
                    }
                });

                ui.add_space(5.0);

                // Buoyancy slider
                let mut buoyancy = gpu_scene.boulder_buoyancy;
                ui.label("Buoyancy in Water:");
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new("0 = floats, 1 = sinks at full gravity")
                        .small()
                        .weak(),
                );
                if ui
                    .add(egui::Slider::new(&mut buoyancy, 0.0..=1.0).step_by(0.01))
                    .changed()
                {
                    gpu_scene.boulder_buoyancy = buoyancy;
                    if let Some(ref mut bs) = gpu_scene.boulder_system {
                        bs.buoyancy = buoyancy;
                        bs.buoyancy_dirty = true;
                    }
                    boulder_changed = true;
                }

                ui.add_space(3.0);

                // Size gate
                let mut gate = gpu_scene.boulder_size_gate;
                ui.label("Size Gate (half-saturation):");
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(
                        "Organism size at which consumption reaches 50% of max rate.",
                    )
                    .small()
                    .weak(),
                );
                if ui.add(egui::Slider::new(&mut gate, 1.0..=200.0)).changed() {
                    gpu_scene.boulder_size_gate = gate;
                    boulder_changed = true;
                }
            } // end advanced boulder settings

            if boulder_changed {
                // Mirror to editor_state so save_cave_settings picks them up
                context.editor_state.show_boulders = gpu_scene.show_boulders;
                context.editor_state.boulder_target_count = gpu_scene.boulder_target_count;
                context.editor_state.boulder_initial_moss = gpu_scene.boulder_initial_moss;
                context.editor_state.boulder_radius = gpu_scene.boulder_radius;
                context.editor_state.boulder_size_gate = gpu_scene.boulder_size_gate;
                context.editor_state.boulder_spawn_interval = gpu_scene.boulder_spawn_interval;
                context.editor_state.boulder_buoyancy = gpu_scene.boulder_buoyancy;
                context.editor_state.boulder_radius_min = gpu_scene.boulder_radius_min;
                context.editor_state.boulder_radius_max = gpu_scene.boulder_radius_max;
                context.editor_state.boulder_moss_min = gpu_scene.boulder_moss_min;
                context.editor_state.boulder_moss_max = gpu_scene.boulder_moss_max;
                context.editor_state.save_cave_settings();
            }

            ui.add_space(5.0);

            // Live stats
            if let Some(ref bs) = gpu_scene.boulder_system {
                let live = bs
                    .buffers
                    .cpu_boulders
                    .iter()
                    .filter(|b| b.dead == 0 && b.radius > 0.0)
                    .count();
                ui.label(format!("Live mossrocks: {}", live));
            } else {
                ui.label(
                    egui::RichText::new("Mossrock system not initialized. Requires cave system.")
                        .weak(),
                );
            }
        }
    }
}

/// Render the Fluid Settings panel for fluid simulation controls and visualization.
fn render_fluid_settings(ui: &mut Ui, context: &mut PanelContext, state: &mut GlobalUiState) {
    let sw = (ui.available_width() - 60.0).max(60.0);
    ui.style_mut().spacing.slider_width = sw;
    // Check if we're in GPU mode
    if !context.is_gpu_mode() {
        return;
    }

    // Check if fluid system exists
    let has_fluid_system = context
        .scene_manager
        .gpu_scene()
        .map(|scene| scene.fluid_buffers.is_some())
        .unwrap_or(false);

    if !has_fluid_system {
        return;
    }

    // Continuous spawning controls
    ui.separator();
    ui.heading("Fluid Spawning");
    ui.separator();

    // Toggle button - same as the  rail button in the side bar.
    let water_active = context.editor_state.fluid_continuous_spawn;
    let label = if water_active {
        "🌊  Water Fill: ON"
    } else {
        "🌊  Water Fill: OFF"
    };
    let p = crate::ui::ui_system::palette();
    let btn_fill = if water_active {
        egui::Color32::from_rgba_unmultiplied(
            p.accent_primary.r(),
            p.accent_primary.g(),
            p.accent_primary.b(),
            40,
        )
    } else {
        egui::Color32::TRANSPARENT
    };
    let btn_stroke = if water_active {
        p.accent_primary
    } else {
        p.border_normal
    };
    if ui
        .add(
            egui::Button::new(egui::RichText::new(label).color(if water_active {
                p.accent_primary
            } else {
                p.text_secondary
            }))
            .fill(btn_fill)
            .stroke(egui::Stroke::new(1.0, btn_stroke)),
        )
        .clicked()
    {
        context.editor_state.request_toggle_water = true;
    }
    ui.label(
        egui::RichText::new(
            "Continuously fills the world with water. Same as the 🌊 button in the side bar.",
        )
        .small()
        .color(p.text_dim),
    );
    ui.add_space(4.0);

    // === Nutrients ===
    ui.separator();
    ui.heading("Nutrients");
    ui.separator();

    ui.label("Density:")
        .on_hover_text("Fraction of the world volume that contains nutrient voxels at any given time. Higher values create a richer food environment");
    if ui
        .add(
            egui::Slider::new(&mut context.editor_state.nutrient_density, 0.0..=0.5)
                .step_by(0.01)
                .fixed_decimals(2),
        )
        .changed()
    {
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_density = context.editor_state.nutrient_density;
        }
        context.editor_state.save_fluid_settings();
    }

    ui.label("Epoch Duration:")
        .on_hover_text("Total length of one nutrient spawn cycle in seconds. Nutrients spawn during the first part of the epoch and despawn during the last part");
    if ui
        .add(
            egui::Slider::new(
                &mut context.editor_state.nutrient_epoch_duration,
                2.0..=30.0,
            )
            .step_by(0.5)
            .fixed_decimals(1)
            .suffix("s"),
        )
        .changed()
    {
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_epoch_duration = context.editor_state.nutrient_epoch_duration;
        }
        context.editor_state.save_fluid_settings();
    }

    ui.label("Epoch Spacing:")
        .on_hover_text("Gap between consecutive nutrient epochs in seconds. Longer spacing creates feast-and-famine cycles that drive organism behavior");
    if ui
        .add(
            egui::Slider::new(&mut context.editor_state.nutrient_epoch_spacing, 1.0..=30.0)
                .step_by(0.5)
                .fixed_decimals(1)
                .suffix("s"),
        )
        .changed()
    {
        // Clamp spacing to not exceed duration
        context.editor_state.nutrient_epoch_spacing = context
            .editor_state
            .nutrient_epoch_spacing
            .min(context.editor_state.nutrient_epoch_duration);
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.nutrient_epoch_spacing = context.editor_state.nutrient_epoch_spacing;
        }
        context.editor_state.save_fluid_settings();
    }

    if state.show_advanced_options {
        ui.label("Spawn Ramp:")
            .on_hover_text("Fraction of the epoch duration over which nutrients gradually appear. 0.1 = nutrients appear quickly at the start; 0.9 = nutrients trickle in slowly over most of the epoch");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.nutrient_spawn_end, 0.05..=0.9)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            // Ensure spawn_end < despawn_start
            if context.editor_state.nutrient_spawn_end
                >= context.editor_state.nutrient_despawn_start
            {
                context.editor_state.nutrient_despawn_start =
                    (context.editor_state.nutrient_spawn_end + 0.05).min(0.95);
            }
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.nutrient_spawn_end = context.editor_state.nutrient_spawn_end;
                gpu_scene.nutrient_despawn_start = context.editor_state.nutrient_despawn_start;
            }
            context.editor_state.save_fluid_settings();
        }

        ui.label("Despawn Ramp:")
            .on_hover_text("Fraction of the epoch at which nutrients start disappearing. Must be greater than Spawn Ramp. Lower values give organisms less time to consume nutrients before they vanish");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.nutrient_despawn_start, 0.1..=0.95)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            // Ensure despawn_start > spawn_end
            if context.editor_state.nutrient_despawn_start
                <= context.editor_state.nutrient_spawn_end
            {
                context.editor_state.nutrient_spawn_end =
                    (context.editor_state.nutrient_despawn_start - 0.05).max(0.05);
            }
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.nutrient_spawn_end = context.editor_state.nutrient_spawn_end;
                gpu_scene.nutrient_despawn_start = context.editor_state.nutrient_despawn_start;
            }
            context.editor_state.save_fluid_settings();
        }
    } // end advanced nutrient ramps

    // Surface pressure control for radial fluid mode
    ui.separator();
    ui.heading("Fluid Physics");

    // Only show surface pressure when gravity mode is radial (advanced only)
    if state.show_advanced_options && state.world_settings.gravity_mode == 3 {
        ui.label("Surface Pressure:")
            .on_hover_text("Tangential smoothing strength for radial fluid mode. Higher values push fluid toward the world surface more aggressively, creating a more uniform water layer");
        if ui
            .add(egui::Slider::new(
                &mut state.fluid_settings.surface_pressure,
                0.0..=1.0,
            ))
            .changed()
        {
            // Apply to GPU scene
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.set_surface_pressure(state.fluid_settings.surface_pressure);
            }
        }
        ui.label(
            egui::RichText::new("Tangential smoothing strength for radial fluid mode").small(),
        );
    }

    // Climate: humidity, freeze/melt, and snow tunables - advanced only
    if state.show_advanced_options {
        ui.add_space(4.0);
        ui.label("Humidity Diffusion Rate:")
            .on_hover_text("How quickly atmospheric humidity spreads between voxels each tick");
        if ui
            .add(
                egui::Slider::new(
                    &mut state.fluid_settings.climate.humidity_diffusion_rate,
                    0.0..=1.0,
                )
                .step_by(0.01)
                .fixed_decimals(2),
            )
            .changed()
        {
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.humidity_diffusion_rate =
                    state.fluid_settings.climate.humidity_diffusion_rate;
            }
        }

        ui.label("Freeze Rate:").on_hover_text(
            "How quickly sustained cold accumulates freeze debt, converting water to ice",
        );
        if ui
            .add(
                egui::Slider::new(&mut state.fluid_settings.climate.freeze_rate, 0.0..=5.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.freeze_rate = state.fluid_settings.climate.freeze_rate;
            }
        }

        ui.label("Melt Rate:").on_hover_text(
            "How quickly sustained warmth accumulates melt debt, converting ice to water",
        );
        if ui
            .add(
                egui::Slider::new(&mut state.fluid_settings.climate.melt_rate, 0.0..=5.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.melt_rate = state.fluid_settings.climate.melt_rate;
            }
        }

        ui.label("Snow Melt Rate:").on_hover_text(
            "How quickly sustained warmth accumulates melt debt, converting snow to water",
        );
        if ui
            .add(
                egui::Slider::new(&mut state.fluid_settings.climate.snow_melt_rate, 0.0..=5.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.snow_melt_rate = state.fluid_settings.climate.snow_melt_rate;
            }
        }

        ui.label("Snow Compaction Rate:")
            .on_hover_text("How quickly sustained cold packs snow down into ice (debt units per tick per degree below freezing)");
        if ui
            .add(
                egui::Slider::new(
                    &mut state.fluid_settings.climate.snow_compact_rate,
                    0.0..=5.0,
                )
                .step_by(0.05)
                .fixed_decimals(2),
            )
            .changed()
        {
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                gpu_scene.snow_compact_rate = state.fluid_settings.climate.snow_compact_rate;
            }
        }
    }

    // Lateral flow probability - advanced only
    if state.show_advanced_options {
        let selected_fluid_type = context.editor_state.selected_fluid_type;

        if selected_fluid_type > 0 && selected_fluid_type <= 3 {
            let mut probabilities = context.editor_state.fluid_lateral_flow_probabilities;
            let mut changed = false;

            ui.add_space(4.0);
            ui.label("Lateral Flow Probability:")
                .on_hover_text("Probability that a fluid voxel moves sideways each step. Higher values create more horizontal spreading and mixing");
            if ui
                .add(
                    egui::Slider::new(&mut probabilities[selected_fluid_type as usize], 0.0..=1.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                )
                .changed()
            {
                changed = true;
            }

            if changed {
                context.editor_state.fluid_lateral_flow_probabilities = probabilities;
                context.editor_state.save_fluid_settings();
            }
        }
    } // end advanced fluid physics

    // Condensation and vaporization are now purely a consequence of local
    // temperature and contact with air (see thermal model in fluid_sim.wgsl) -
    // no manual probability controls needed.

    // === Ice Appearance ===
    // Written to fluid_settings.ice (persisted with the UI state); app.rs
    // pushes the values into the ice render params uniform every frame.
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Ice Appearance");
    ui.add_space(4.0);
    {
        let advanced = state.show_advanced_options;
        let ice = &mut state.fluid_settings.ice;

        ui.label("Facet Size:")
            .on_hover_text("Size of each crystal face in world units. Faces are irregular Voronoi regions, independent of mesh tessellation");
        let mut facet_size = 1.0 / ice.facet_scale.max(0.001);
        if ui
            .add(
                egui::Slider::new(&mut facet_size, 1.0..=200.0)
                    .step_by(1.0)
                    .fixed_decimals(0)
                    .logarithmic(true),
            )
            .changed()
        {
            ice.facet_scale = 1.0 / facet_size.max(1.0);
        }

        ui.add_space(4.0);
        ui.label("Facet Depth:")
            .on_hover_text("How far vertices are displaced onto each flat crystal facet plane. 0 = smooth ice (no displacement), higher = deeper, more angular faceting");
        ui.add(
            egui::Slider::new(&mut ice.displacement_strength, 0.0..=2.0)
                .step_by(0.05)
                .fixed_decimals(2),
        );

        ui.add_space(4.0);
        ui.label("Facet Shading:")
            .on_hover_text("How much the flat facet normals affect diffuse shading. Low = facets only show as glints; high = each face shades as its own patch (patchwork look)");
        ui.add(
            egui::Slider::new(&mut ice.facet_diffuse, 0.0..=1.0)
                .step_by(0.01)
                .fixed_decimals(2),
        );

        ui.add_space(4.0);
        ui.label("Glint Intensity:")
            .on_hover_text("Brightness of the per-face specular sparkle");
        ui.add(
            egui::Slider::new(&mut ice.glint_strength, 0.0..=3.0)
                .step_by(0.05)
                .fixed_decimals(2),
        );

        ui.add_space(4.0);
        ui.label("Opacity:")
            .on_hover_text("Base opacity of the ice body. Grazing angles always read more solid. At 1.00, ice is rendered fully opaque (no transparency blending).");
        ui.add(
            egui::Slider::new(&mut ice.alpha, 0.2..=1.0)
                .step_by(0.01)
                .fixed_decimals(2),
        );

        ui.horizontal(|ui| {
            ui.label("Surface Color");
            ui.color_edit_button_rgb(&mut ice.surface_color);
            ui.label("Deep Color");
            ui.color_edit_button_rgb(&mut ice.deep_color);
        });

        if advanced {
            ui.add_space(4.0);
            ui.label("Glint Sharpness:").on_hover_text(
                "Blinn-Phong exponent for the facet glints. Higher = tighter, sharper sparkles",
            );
            ui.add(
                egui::Slider::new(&mut ice.glint_shininess, 4.0..=256.0)
                    .step_by(1.0)
                    .fixed_decimals(0)
                    .logarithmic(true),
            );

            ui.add_space(4.0);
            ui.label("Reflection:")
                .on_hover_text("Fresnel-weighted environment reflection mix at grazing angles");
            ui.add(
                egui::Slider::new(&mut ice.fresnel_reflection, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            );

            ui.add_space(4.0);
            ui.label("Reflection Brightness:")
                .on_hover_text("Brightness multiplier for the reflected environment");
            ui.add(
                egui::Slider::new(&mut ice.reflection_brightness, 0.0..=4.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            );
        }
    }

    // === Water Surface Lighting & Reflection ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Water Surface Lighting");
    ui.add_space(4.0);

    if state.show_advanced_options {
        ui.label("Ambient:")
            .on_hover_text("Minimum light level on the water surface regardless of light direction. Higher values brighten the water in shadowed areas");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_ambient, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Diffuse:")
            .on_hover_text("Strength of the diffuse (Lambertian) lighting on the water surface. Higher values make the water respond more strongly to the light direction");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_diffuse, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Specular:")
            .on_hover_text("Strength of the specular highlight (glint) on the water surface. Higher values create a brighter, more mirror-like reflection of the light source");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_specular, 0.0..=2.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Shininess:")
            .on_hover_text("Sharpness of the specular highlight. Low values = broad, soft glint. High values = tight, sharp glint like polished glass");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_shininess, 1.0..=256.0)
                    .step_by(1.0)
                    .fixed_decimals(0)
                    .logarithmic(true),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Rim Light:")
            .on_hover_text("Backlit glow around the edges of the water surface. Creates a halo effect when the light is behind the water");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_rim, 0.0..=2.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
    } // end advanced water surface lighting

    ui.label("Alpha (Transparency):")
        .on_hover_text("Overall transparency of the water surface. 0 = fully transparent (invisible); 1 = fully opaque");
    if ui
        .add(
            egui::Slider::new(&mut context.editor_state.fluid_alpha, 0.0..=1.0)
                .step_by(0.01)
                .fixed_decimals(2),
        )
        .changed()
    {
        context.editor_state.save_fluid_render_settings();
    }

    if state.show_advanced_options {
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Cubemap Reflection");
        ui.add_space(4.0);

        ui.label("Fresnel Strength:")
            .on_hover_text("How strongly the Fresnel effect controls reflection intensity. Higher values make the water more reflective at grazing angles");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_fresnel, 0.0..=2.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Fresnel Power:")
            .on_hover_text("Sharpness of the Fresnel transition. Higher values make the reflection appear only at very shallow angles; lower values spread it across more of the surface");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_fresnel_power, 0.5..=10.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Reflection Intensity:").on_hover_text(
            "Overall strength of the cubemap/environment reflection on the water surface",
        );
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_reflection, 0.0..=2.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Reflection Brightness:")
            .on_hover_text("Brightness multiplier applied to the reflected environment. Higher values make the reflection appear brighter and more vivid");
        if ui
            .add(
                egui::Slider::new(
                    &mut context.editor_state.fluid_reflection_brightness,
                    1.0..=50.0,
                )
                .step_by(0.5)
                .fixed_decimals(1),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Below-Waterline Opacity:")
            .on_hover_text("Transparency of faces below the water surface (sides and bottom). Set low to see inside the sphere");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_waterline_alpha, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.save_fluid_render_settings();
        }

        // === Water Surface Waves ===
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Water Surface Waves");
        ui.add_space(4.0);

        ui.label("Wave Height:")
            .on_hover_text("Amplitude of the animated wave displacement on the water surface. 0 = flat water; higher values create more dramatic waves");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_wave_height, 0.0..=5.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.fluid_mesh_params_dirty = true;
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Wave Speed:")
            .on_hover_text("How fast the wave pattern animates across the water surface");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_wave_speed, 0.0..=5.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.fluid_mesh_params_dirty = true;
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Noise Scale (lower = larger waves):")
            .on_hover_text("Spatial frequency of the wave noise pattern. Lower values create large, rolling waves; higher values create small, choppy ripples");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_noise_scale, 0.01..=20.0)
                    .step_by(0.01)
                    .fixed_decimals(2)
                    .logarithmic(true),
            )
            .changed()
        {
            context.editor_state.fluid_mesh_params_dirty = true;
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Octaves:")
            .on_hover_text("Number of noise layers combined for the wave pattern. More octaves add finer detail to the waves");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_noise_octaves, 1.0..=6.0)
                    .step_by(1.0)
                    .fixed_decimals(0),
            )
            .changed()
        {
            context.editor_state.fluid_mesh_params_dirty = true;
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Lacunarity (freq per octave):")
            .on_hover_text("How much the frequency increases with each successive noise octave. Higher values create more varied, complex wave patterns");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_noise_lacunarity, 1.0..=4.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed()
        {
            context.editor_state.fluid_mesh_params_dirty = true;
            context.editor_state.save_fluid_render_settings();
        }

        ui.add_space(4.0);
        ui.label("Persistence (amp per octave):")
            .on_hover_text("How much the amplitude decreases with each successive noise octave. Lower values make higher octaves contribute less, creating smoother waves");
        if ui
            .add(
                egui::Slider::new(&mut context.editor_state.fluid_noise_persistence, 0.1..=1.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed()
        {
            context.editor_state.fluid_mesh_params_dirty = true;
            context.editor_state.save_fluid_render_settings();
        }

        // === Caustics ===
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Caustics");
        ui.add_space(4.0);
        let mut caustic_changed = false;

        ui.label("Intensity:").on_hover_text(
            "Brightness of the caustic light patterns projected onto surfaces beneath the water",
        );
        caustic_changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.caustic_intensity, 0.0..=2.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed();

        ui.add_space(4.0);
        ui.label("Scale:")
            .on_hover_text("Size of the caustic pattern. Smaller values create tighter, more detailed caustic ripples; larger values create broader patterns");
        caustic_changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.caustic_scale, 0.1..=30.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed();

        ui.add_space(4.0);
        ui.label("Speed:")
            .on_hover_text("How fast the caustic pattern animates. Higher values create more dynamic, rapidly shifting light patterns");
        caustic_changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.caustic_speed, 0.0..=5.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed();

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
    } // end advanced water visuals (cubemap, waves, caustics)

    ui.add_space(10.0);
}

/// Organism skin controls, embedded at the bottom of the fluid/environment panel.
fn render_organism_skin_settings(
    ui: &mut Ui,
    context: &mut PanelContext,
    organism_skin: &mut crate::ui::OrganismSkinSettings,
) {
    if !context.is_gpu_mode() {
        return;
    }

    ui.add_space(8.0);
    ui.separator();
    ui.heading("Organism Skins");
    ui.add_space(4.0);

    ui.label("Wrap cell clusters in a smooth biological membrane.");
    ui.add_space(4.0);

    if ui.checkbox(&mut organism_skin.enabled, "Show Organism Skins")
        .on_hover_text("Wrap cell clusters in a smooth biological membrane mesh. Requires GPU mode and fluid system initialization")
        .changed() {
        // Enabling is handled in app.rs (lazy initialisation)
    }

    if !organism_skin.enabled {
        ui.add_space(8.0);
        return;
    }

    ui.add_space(6.0);

    // -- Geometry --------------------------------------------------------------
    ui.label("Skin Offset (gap from cell surface, world units):")
        .on_hover_text("How far the skin mesh extends beyond the cell surfaces. Larger values create a more inflated, blobby appearance");
    ui.add(
        egui::Slider::new(&mut organism_skin.radius_scale, 0.0..=5.0)
            .step_by(0.05)
            .fixed_decimals(2),
    );

    ui.add_space(4.0);
    ui.label("Shrink Speed (fraction of gap closed per iteration):")
        .on_hover_text("How aggressively the skin mesh wraps around the cells each iteration. Higher values create a tighter fit but may cause artifacts");
    ui.add(
        egui::Slider::new(&mut organism_skin.shrink_speed, 0.01..=1.0)
            .step_by(0.01)
            .fixed_decimals(2),
    );

    ui.add_space(4.0);
    ui.label("Smooth Factor (Laplacian blend per iteration):")
        .on_hover_text("How much the skin mesh is smoothed each iteration. Higher values create a smoother, more organic surface");
    ui.add(
        egui::Slider::new(&mut organism_skin.smooth_factor, 0.0..=0.8)
            .step_by(0.01)
            .fixed_decimals(2),
    );

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Shrink Iterations:")
            .on_hover_text("Number of shrink-wrap passes. More iterations create a tighter skin but increase GPU cost");
        ui.add(egui::Slider::new(&mut organism_skin.shrink_iters, 1u32..=20u32));
    });

    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label("Smooth Iterations:").on_hover_text(
            "Number of Laplacian smoothing passes. More iterations create a smoother surface",
        );
        ui.add(egui::Slider::new(
            &mut organism_skin.smooth_iters,
            0u32..=8u32,
        ));
    });

    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label("Min Cells for Skin:")
            .on_hover_text("Minimum number of cells an organism must have before a skin is rendered. Prevents skins on isolated single cells");
        ui.add(egui::Slider::new(&mut organism_skin.min_cells, 1u32..=50u32));
    });

    // -- Material --------------------------------------------------------------
    ui.add_space(6.0);
    ui.separator();
    ui.label("Skin Colour:");
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut organism_skin.base_color);
    });

    ui.add_space(4.0);
    ui.label("Opacity:").on_hover_text(
        "Transparency of the organism skin. 0 = fully transparent; 1 = fully opaque",
    );
    ui.add(
        egui::Slider::new(&mut organism_skin.alpha, 0.0..=1.0)
            .step_by(0.01)
            .fixed_decimals(2),
    );

    ui.add_space(4.0);
    ui.label("Subsurface Scattering:")
        .on_hover_text("Simulates light penetrating and scattering beneath the skin surface, creating a soft, translucent biological look");
    ui.add(
        egui::Slider::new(&mut organism_skin.sss_strength, 0.0..=3.0)
            .step_by(0.05)
            .fixed_decimals(2),
    );

    ui.add_space(4.0);
    ui.label("Rim Light:")
        .on_hover_text("Backlit glow around the edges of the organism skin. Creates a halo effect that makes organisms stand out against the background");
    ui.add(
        egui::Slider::new(&mut organism_skin.rim_strength, 0.0..=3.0)
            .step_by(0.05)
            .fixed_decimals(2),
    );

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

fn sun_tint_color32(color: [f32; 3]) -> egui::Color32 {
    egui::Rgba::from_rgb(
        color[0].clamp(0.0, 1.0),
        color[1].clamp(0.0, 1.0),
        color[2].clamp(0.0, 1.0),
    )
    .into()
}

fn render_sun_tint_control(ui: &mut Ui, tint: &mut [f32; 3], sun_intensity: f32) -> bool {
    let mut changed = false;
    let presets = [
        ("White", [1.0, 1.0, 1.0]),
        ("Warm", [1.0, 0.86, 0.62]),
        ("Gold", [1.0, 0.66, 0.28]),
        ("Dusk", [1.0, 0.42, 0.22]),
        ("Cool", [0.72, 0.86, 1.0]),
    ];

    ui.label("Sun Tint")
        .on_hover_text("Hue of the sun. Brightness is controlled by the Brightness slider above.");
    ui.horizontal_wrapped(|ui| {
        for (name, preset) in presets {
            let selected = (*tint)[0].abs_diff_eq(preset[0], 0.01)
                && (*tint)[1].abs_diff_eq(preset[1], 0.01)
                && (*tint)[2].abs_diff_eq(preset[2], 0.01);
            let button = egui::Button::new(name)
                .fill(sun_tint_color32(preset))
                .stroke(if selected {
                    egui::Stroke::new(2.0, ui.visuals().text_color())
                } else {
                    ui.visuals().widgets.inactive.bg_stroke
                });
            if ui.add(button).clicked() {
                *tint = preset;
                changed = true;
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("Custom");
        if ui.color_edit_button_rgb(tint).changed() {
            changed = true;
        }

        let effective = [
            tint[0] * sun_intensity,
            tint[1] * sun_intensity,
            tint[2] * sun_intensity,
        ];
        let preview = [
            1.0 - (-effective[0] * 0.18).exp(),
            1.0 - (-effective[1] * 0.18).exp(),
            1.0 - (-effective[2] * 0.18).exp(),
        ];
        ui.label("Output");
        ui.add(
            egui::Button::new("")
                .fill(sun_tint_color32(preview))
                .min_size(egui::vec2(30.0, 18.0)),
        )
        .on_hover_text(format!(
            "Actual sun color: {:.2}, {:.2}, {:.2}",
            effective[0], effective[1], effective[2]
        ));
    });

    changed
}

fn light_dir_to_pitch_yaw(dir: [f32; 3]) -> (f32, f32) {
    let d = glam::Vec3::from_array(dir).normalize_or_zero();
    if d.length_squared() == 0.0 {
        return (45.0, -45.0);
    }
    let pitch = d.y.clamp(-1.0, 1.0).asin().to_degrees();
    let yaw = d.x.atan2(d.z).to_degrees();
    (pitch, yaw)
}

fn pitch_yaw_to_light_dir(pitch_degrees: f32, yaw_degrees: f32) -> [f32; 3] {
    let pitch = pitch_degrees.to_radians();
    let yaw = yaw_degrees.to_radians();
    let horizontal = pitch.cos();
    [horizontal * yaw.sin(), pitch.sin(), horizontal * yaw.cos()]
}

fn render_sun_direction_control(ui: &mut Ui, light_dir: &mut [f32; 3]) -> bool {
    let (mut pitch, mut yaw) = light_dir_to_pitch_yaw(*light_dir);
    let mut changed = false;

    ui.label("Direction")
        .on_hover_text("Pitch controls sun height. Yaw rotates the sun around the world.");
    ui.label("Pitch");
    changed |= ui
        .add(
            egui::Slider::new(&mut pitch, -89.0..=89.0)
                .text("Pitch")
                .step_by(1.0)
                .fixed_decimals(0),
        )
        .changed();
    ui.label("Yaw");
    changed |= ui
        .add(
            egui::Slider::new(&mut yaw, -180.0..=180.0)
                .text("Yaw")
                .step_by(1.0)
                .fixed_decimals(0),
        )
        .changed();

    ui.horizontal_wrapped(|ui| {
        if ui.button("Top").on_hover_text("Overhead light").clicked() {
            pitch = 89.0;
            yaw = 0.0;
            changed = true;
        }
        if ui.button("Low").on_hover_text("Low angled light").clicked() {
            pitch = 20.0;
            yaw = -55.0;
            changed = true;
        }
        if ui
            .button("Default")
            .on_hover_text("Default angled light")
            .clicked()
        {
            pitch = 44.0;
            yaw = -45.0;
            changed = true;
        }
    });

    if changed {
        *light_dir = pitch_yaw_to_light_dir(pitch, yaw);
    }

    changed
}

trait AbsDiffEq {
    fn abs_diff_eq(self, other: Self, epsilon: Self) -> bool;
}

impl AbsDiffEq for f32 {
    fn abs_diff_eq(self, other: Self, epsilon: Self) -> bool {
        (self - other).abs() <= epsilon
    }
}

/// Push light-panel edits to the scene and persist them. Shared by the
/// early-return (non-advanced) and full panel paths.
fn apply_light_panel_changes(context: &mut PanelContext, sun_changed: bool) {
    if sun_changed {
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.show_sun = context.editor_state.show_sun;
            gpu_scene.sun_intensity = context.editor_state.sun_intensity;
            if let Some(ref mut sun) = gpu_scene.sun_renderer {
                sun.sun_color = context.editor_state.sun_color;
                sun.sun_angular_radius = context.editor_state.sun_angular_radius;
            }
        }
    }
    context.editor_state.light_params_dirty = true;
    context.editor_state.save_light_settings();
}

fn render_light_settings_organized(
    ui: &mut Ui,
    context: &mut PanelContext,
    state: &mut GlobalUiState,
) {
    let sw = (ui.available_width() - 60.0).max(60.0);
    ui.style_mut().spacing.slider_width = sw;

    if !context.is_gpu_mode() {
        ui.label("Lighting settings are only available in GPU mode.");
        return;
    }

    let has_light_field = context
        .scene_manager
        .gpu_scene()
        .map(|s| s.light_field_system.is_some())
        .unwrap_or(false);

    if !has_light_field {
        ui.label("Light field system not initialized.");
        ui.add_space(5.0);
        ui.label("Initialize the fluid system first to enable lighting.");
        return;
    }

    let mut changed = false;
    let mut sun_changed = false;

    egui::CollapsingHeader::new("Sun")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                sun_changed |= ui
                    .checkbox(&mut context.editor_state.show_sun, "Sun Disc")
                    .on_hover_text("Render the visible sun disc in the sky.")
                    .changed();
                let orbit_toggle = ui
                    .checkbox(&mut context.editor_state.sun_rotation_enabled, "Orbit")
                    .on_hover_text("Rotate the sun along the defined orbit path.");
                if orbit_toggle.changed() && context.editor_state.sun_rotation_enabled {
                    context.editor_state.capture_sun_orbit_angle();
                }
                changed |= orbit_toggle.changed();
                if context.editor_state.sun_rotation_enabled {
                    changed |= ui
                        .checkbox(&mut context.editor_state.show_orbit_ring, "Show Ring")
                        .on_hover_text("Keep the orbit path ring visible in the scene.")
                        .changed();
                }
            });

            ui.add_space(4.0);
            ui.label("Brightness (0-5)");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.sun_intensity, 0.0..=5.0)
                        .text("Brightness")
                        .step_by(0.1)
                        .fixed_decimals(1),
                )
                .on_hover_text("Sun brightness controls baseline air temperature: 0=dark, 3=temperate (recommended), 5=extreme heat. Affects photocyte energy and thermal model.")
                .changed();
            if changed {
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    gpu_scene.sun_intensity = context.editor_state.sun_intensity;
                }
            }

            ui.horizontal(|ui| {
                ui.label("Tint");
                if ui
                    .color_edit_button_rgb(&mut context.editor_state.sun_color)
                    .changed()
                {
                    sun_changed = true;
                }
            });

            ui.add_space(4.0);
            if context.editor_state.sun_rotation_enabled {
                let mut orbit_angle = context.editor_state.sun_orbit_angle;
                ui.label("Position");
                let position_response = ui
                    .add(
                        egui::Slider::new(&mut orbit_angle, 0.0..=360.0)
                            .text("Position")
                            .suffix("°")
                            .step_by(0.1)
                            .fixed_decimals(1),
                    )
                    .on_hover_text("Sun position along the visible orbit ring.");
                if position_response.changed()
                    && (position_response.dragged() || position_response.has_focus())
                {
                    context.editor_state.sun_orbit_angle = orbit_angle;
                    context.editor_state.apply_sun_orbit();
                    changed = true;
                }

                let mut speed_degrees = context.editor_state.sun_rotation_speed.to_degrees().max(0.0);
                ui.label("Speed");
                if ui
                    .add(
                        egui::Slider::new(&mut speed_degrees, 0.0..=30.0)
                            .text("Speed")
                            .suffix("°/s")
                            .logarithmic(true)
                            .smallest_positive(0.01)
                            .fixed_decimals(2),
                    )
                    .on_hover_text("How fast the sun moves along the orbit. 0 keeps it parked.")
                    .changed()
                {
                    context.editor_state.sun_rotation_speed = speed_degrees.to_radians();
                    changed = true;
                }

                if state.show_advanced_options {
                    let (mut orbit_pitch, mut orbit_yaw) =
                        light_dir_to_pitch_yaw(context.editor_state.sun_rotation_axis);
                    let mut orbit_changed = false;
                    ui.label("Path Tilt");
                    orbit_changed |= ui
                        .add(
                            egui::Slider::new(&mut orbit_pitch, -90.0..=90.0)
                                .text("Tilt")
                                .suffix("°")
                                .fixed_decimals(0),
                        )
                        .on_hover_text("Tilt of the orbit plane. 0° = equatorial, 90° = polar.")
                        .changed();
                    ui.label("Path Yaw");
                    orbit_changed |= ui
                        .add(
                            egui::Slider::new(&mut orbit_yaw, -180.0..=180.0)
                                .text("Direction")
                                .suffix("°")
                                .fixed_decimals(0),
                        )
                        .on_hover_text("Compass direction the orbit tilts toward.")
                        .changed();
                    if orbit_changed {
                        context.editor_state.sun_rotation_axis =
                            pitch_yaw_to_light_dir(orbit_pitch, orbit_yaw);
                        context.editor_state.apply_sun_orbit();
                        changed = true;
                    }
                }
            } else {
                changed |= render_sun_direction_control(ui, &mut context.editor_state.light_dir);
            }

            if state.show_advanced_options && context.editor_state.show_sun {
                ui.add_space(4.0);
                ui.label("Disc Size");
                sun_changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.sun_angular_radius,
                            0.005..=0.2,
                        )
                        .text("Disc Size")
                        .step_by(0.005)
                        .fixed_decimals(3),
                    )
                    .changed();
            }
        });

    ui.separator();

    egui::CollapsingHeader::new("Day & Night")
        .default_open(true)
        .show(ui, |ui| {
            ui.label("Night Ratio (0-1)");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.sun_night_ratio, 0.0..=1.0)
                        .text("Night Ratio")
                        .step_by(0.05)
                        .fixed_decimals(2),
                )
                .on_hover_text(
                    "Fraction of each cycle spent in full darkness. 0.0 = no night, 0.5 = \
                     half of every cycle is pitch-black, with smooth dawn/dusk transitions. \
                     Independent of the sun's orbit position.",
                )
                .changed();
            ui.label("Cycle Length");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.sun_cycle_period, 10.0..=3600.0)
                        .text("Cycle Length")
                        .suffix(" s")
                        .step_by(1.0)
                        .fixed_decimals(0),
                )
                .on_hover_text("Length of one full day/night (and seasonal) cycle in seconds.")
                .changed();

            // Climate response speed lives here because the day/night cycle
            // is what drives temperature swings. Edits the persisted climate
            // setting (app.rs syncs it into the scene each frame).
            ui.label("Thermal Inertia (0-5)");
            if ui
                .add(
                    egui::Slider::new(
                        &mut state.fluid_settings.climate.thermal_inertia,
                        0.0..=5.0,
                    )
                    .text("Inertia")
                    .step_by(0.1)
                    .fixed_decimals(1),
                )
                .on_hover_text("Controls how fast climate changes: 0=arcade (fast), 3=stable, 4=very stable (recommended), 5=planetary (slow)")
                .changed()
            {
                changed = true;
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    gpu_scene.thermal_inertia = state.fluid_settings.climate.thermal_inertia;
                }
            }

            ui.add_space(6.0);
            changed |= ui
                .checkbox(&mut context.editor_state.sun_cycle_enabled, "Seasonal Cycle")
                .on_hover_text(
                    "Oscillate daytime brightness between Min and Max over the cycle \
                     length, overriding the Brightness slider.",
                )
                .changed();
            if context.editor_state.sun_cycle_enabled {
                ui.label("Season Min");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.sun_cycle_min, 0.0..=20.0)
                            .text("Min")
                            .step_by(0.1)
                            .fixed_decimals(1),
                    )
                    .changed();
                ui.label("Season Max");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.sun_cycle_max, 0.0..=20.0)
                            .text("Max")
                            .step_by(0.1)
                            .fixed_decimals(1),
                    )
                    .changed();
            }
        });

    // Everything below is rendering/engine tuning the player doesn't touch
    // during normal play - gated behind Advanced to keep the panel focused on
    // sun brightness, day/night cycles, timing, and color.
    if !state.show_advanced_options {
        if changed || sun_changed {
            apply_light_panel_changes(context, sun_changed);
        }
        ui.add_space(10.0);
        return;
    }

    ui.separator();

    egui::CollapsingHeader::new("Surfaces & Shadows")
        .default_open(true)
        .show(ui, |ui| {
            let mut shadow_changed = false;
            shadow_changed |= ui
                .checkbox(&mut context.editor_state.shadow_enabled, "Surface Shadows")
                .on_hover_text("Use the light field to shade cave walls, cells, and surfaces.")
                .changed();
            ui.add_enabled_ui(context.editor_state.shadow_enabled, |ui| {
                ui.label("Strength");
                shadow_changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.shadow_strength, 0.0..=1.0)
                            .text("Strength")
                            .step_by(0.01)
                            .fixed_decimals(2),
                    )
                    .changed();
                ui.label("Quality");
                shadow_changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.shadow_quality, 0.0..=1.0)
                            .text("Quality")
                            .step_by(0.01)
                            .fixed_decimals(2),
                    )
                    .on_hover_text("Higher values reduce banding but increase sampling cost.")
                    .changed();
            });

            if shadow_changed {
                changed = true;
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    if let Some(ref mut light_field) = gpu_scene.light_field_system {
                        light_field.set_shadow_enabled(context.editor_state.shadow_enabled);
                        light_field.set_shadow_strength(context.editor_state.shadow_strength);
                        light_field.set_shadow_quality(context.editor_state.shadow_quality);
                    }
                }
            }

            if state.show_advanced_options {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Light Field").strong());
                let mut steps = context.editor_state.light_field_max_steps as i32;
                ui.label("Ray Steps");
                if ui
                    .add(egui::Slider::new(&mut steps, 1..=256).text("Ray Steps"))
                    .on_hover_text("More steps improve shadow accuracy but cost more GPU time.")
                    .changed()
                {
                    context.editor_state.light_field_max_steps = steps as u32;
                    changed = true;
                }
                ui.label("Step Size");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.light_field_step_size,
                            0.5..=4.0,
                        )
                        .text("Step Size")
                        .step_by(0.1)
                        .fixed_decimals(1),
                    )
                    .changed();
                ui.label("Rock Absorption");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.light_field_absorption_solid,
                            0.1..=20.0,
                        )
                        .text("Rock Absorption")
                        .step_by(0.1)
                        .fixed_decimals(1),
                    )
                    .changed();
                ui.label("Cell Absorption");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.light_field_absorption_cell,
                            0.0..=5.0,
                        )
                        .text("Cell Absorption")
                        .step_by(0.05)
                        .fixed_decimals(2),
                    )
                    .changed();
                ui.label("Ambient Floor");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.light_field_ambient_floor,
                            0.0..=0.2,
                        )
                        .text("Ambient Floor")
                        .step_by(0.005)
                        .fixed_decimals(3),
                    )
                    .changed();
            }
        });

    ui.separator();

    egui::CollapsingHeader::new("Volumetric Fog")
        .default_open(true)
        .show(ui, |ui| {
            changed |= ui
                .checkbox(&mut context.editor_state.show_volumetric_fog, "Enabled")
                .on_hover_text("Render volumetric haze, light shafts, and luminocyte halos.")
                .changed();

            ui.add_enabled_ui(context.editor_state.show_volumetric_fog, |ui| {
                ui.label("Density");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.fog_density, 0.0..=2.0)
                            .text("Density")
                            .step_by(0.01)
                            .fixed_decimals(2),
                    )
                    .changed();
                ui.label("Glow");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.light_intensity, 0.0..=20.0)
                            .text("Glow")
                            .step_by(0.1)
                            .fixed_decimals(1),
                    )
                    .on_hover_text("Brightness of fog scattering from sun and luminocytes.")
                    .changed();
                ui.label("Directionality");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.fog_scattering_anisotropy,
                            0.0..=0.95,
                        )
                        .text("Directionality")
                        .step_by(0.01)
                        .fixed_decimals(2),
                    )
                    .on_hover_text(
                        "How strongly fog scatters light toward the viewer (god-ray sharpness).",
                    )
                    .changed();
                ui.label("Absorption");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.fog_absorption, 0.0..=2.0)
                            .text("Absorption")
                            .step_by(0.01)
                            .fixed_decimals(2),
                    )
                    .changed();

                ui.horizontal(|ui| {
                    ui.label("Ambient Color");
                    if ui
                        .color_edit_button_rgb(&mut context.editor_state.fog_color)
                        .changed()
                    {
                        changed = true;
                    }
                });

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Height Fog").strong());
                ui.label("Bottom Density");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.fog_height_density, 0.0..=2.0)
                            .text("Bottom Density")
                            .step_by(0.01)
                            .fixed_decimals(2),
                    )
                    .changed();
                ui.label("Falloff");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.fog_height_falloff,
                            0.001..=0.1,
                        )
                        .text("Falloff")
                        .step_by(0.001)
                        .fixed_decimals(3),
                    )
                    .changed();

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Quality").strong());
                let mut fsteps = context.editor_state.fog_steps as i32;
                ui.label("Steps");
                if ui
                    .add(egui::Slider::new(&mut fsteps, 8..=128).text("Steps"))
                    .changed()
                {
                    context.editor_state.fog_steps = fsteps as u32;
                    changed = true;
                }
                changed |= ui
                    .checkbox(
                        &mut context.editor_state.fog_smooth_light_field,
                        "Smooth Voxels",
                    )
                    .on_hover_text("Trilinear light-field sampling for fog.")
                    .changed();
                ui.label("Blur");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.fog_composite_blur, 0.25..=4.0)
                            .text("Blur")
                            .step_by(0.25)
                            .fixed_decimals(2),
                    )
                    .changed();

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Water Distortion").strong());
                ui.label("Strength");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.fog_water_wave_strength,
                            0.0..=2.0,
                        )
                        .text("Strength")
                        .step_by(0.05)
                        .fixed_decimals(2),
                    )
                    .changed();
                ui.label("Scale");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.fog_water_wave_scale,
                            0.01..=1.0,
                        )
                        .text("Scale")
                        .step_by(0.01)
                        .fixed_decimals(2),
                    )
                    .changed();
            });
        });

    ui.separator();

    egui::CollapsingHeader::new("Luminocyte Glow")
        .default_open(true)
        .show(ui, |ui| {
            changed |= ui
                .checkbox(
                    &mut context.editor_state.luminocyte_bloom_enabled,
                    "Enabled",
                )
                .on_hover_text("Screen-space bloom halo around glowing luminocytes.")
                .changed();

            ui.add_enabled_ui(context.editor_state.luminocyte_bloom_enabled, |ui| {
                ui.label("Halo Radius");
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut context.editor_state.luminocyte_bloom_radius,
                            0.02..=0.8,
                        )
                        .text("Radius")
                        .step_by(0.01)
                        .fixed_decimals(2),
                    )
                    .on_hover_text(
                        "Halo size as a fraction of screen height. Larger = bigger glow.",
                    )
                    .changed();
            });

            if changed {
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    if let Some(ref mut bloom) = gpu_scene.luminocyte_bloom {
                        bloom.bloom_radius = if context.editor_state.luminocyte_bloom_enabled {
                            context.editor_state.luminocyte_bloom_radius
                        } else {
                            0.0
                        };
                    }
                }
            }
        });

    if state.show_advanced_options {
        ui.separator();

        egui::CollapsingHeader::new("Camera Response")
            .default_open(false)
            .show(ui, |ui| {
                changed |= ui
                    .checkbox(&mut context.editor_state.show_dof, "Depth of Field")
                    .on_hover_text("Blur objects outside the focal range.")
                    .changed();

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Contrast & Eye Adaptation").strong());

                ui.label("Contrast").on_hover_text(
                    "Midpoint contrast. Values above 1 separate lights and darks; 1 is neutral.",
                );
                let contrast_changed = ui
                    .add(
                        egui::Slider::new(&mut context.editor_state.pp_contrast, 0.25..=4.0)
                            .step_by(0.05)
                            .fixed_decimals(2),
                    )
                    .changed();
                if contrast_changed {
                    changed = true;
                    if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                        if let Some(pp) = gpu_scene.post_process.as_mut() {
                            pp.contrast = context.editor_state.pp_contrast;
                        }
                    }
                }

                let adapt_enabled_changed = ui
                    .checkbox(&mut context.editor_state.pp_adapt_enabled, "Eye Adaptation")
                    .on_hover_text(
                        "Camera gradually adjusts exposure between bright and dark areas.",
                    )
                    .changed();
                if adapt_enabled_changed {
                    changed = true;
                    if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                        if let Some(pp) = gpu_scene.post_process.as_mut() {
                            pp.adapt_enabled = context.editor_state.pp_adapt_enabled;
                        }
                    }
                }

                if context.editor_state.pp_adapt_enabled {
                    ui.add_space(4.0);
                    ui.label("Adapt Speed");
                    let adapt_speed_changed = ui
                        .add(
                            egui::Slider::new(&mut context.editor_state.pp_adapt_speed, 0.01..=0.4)
                                .text("Adapt Speed")
                                .step_by(0.01)
                                .fixed_decimals(2),
                        )
                        .changed();
                    if adapt_speed_changed {
                        changed = true;
                        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                            if let Some(pp) = gpu_scene.post_process.as_mut() {
                                pp.adapt_speed = context.editor_state.pp_adapt_speed;
                            }
                        }
                    }
                    ui.label("Min Exposure");
                    let adapt_min_changed = ui
                        .add(
                            egui::Slider::new(&mut context.editor_state.pp_adapt_min, 0.05..=2.0)
                                .text("Min Exposure")
                                .step_by(0.05)
                                .fixed_decimals(2),
                        )
                        .changed();
                    if adapt_min_changed {
                        changed = true;
                        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                            if let Some(pp) = gpu_scene.post_process.as_mut() {
                                pp.adapt_min = context.editor_state.pp_adapt_min;
                            }
                        }
                    }
                    ui.label("Max Exposure");
                    let adapt_max_changed = ui
                        .add(
                            egui::Slider::new(&mut context.editor_state.pp_adapt_max, 1.0..=20.0)
                                .text("Max Exposure")
                                .step_by(0.5)
                                .fixed_decimals(1),
                        )
                        .changed();
                    if adapt_max_changed {
                        changed = true;
                        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                            if let Some(pp) = gpu_scene.post_process.as_mut() {
                                pp.adapt_max = context.editor_state.pp_adapt_max;
                            }
                        }
                    }
                }

                if context.editor_state.show_dof {
                    ui.add_space(6.0);
                    ui.label("Focal Distance");
                    changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut context.editor_state.dof_focal_distance,
                                5.0..=500.0,
                            )
                            .text("Focal Distance")
                            .step_by(1.0)
                            .fixed_decimals(0),
                        )
                        .changed();
                    ui.label("Focal Range");
                    changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut context.editor_state.dof_focal_range,
                                1.0..=200.0,
                            )
                            .text("Focal Range")
                            .step_by(1.0)
                            .fixed_decimals(0),
                        )
                        .changed();
                    ui.label("Max Blur");
                    changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut context.editor_state.dof_max_blur_radius,
                                1.0..=24.0,
                            )
                            .text("Max Blur")
                            .step_by(0.5)
                            .fixed_decimals(1),
                        )
                        .changed();
                    ui.label("Blur Strength");
                    changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut context.editor_state.dof_blur_strength,
                                0.0..=3.0,
                            )
                            .text("Blur Strength")
                            .step_by(0.05)
                            .fixed_decimals(2),
                        )
                        .changed();
                }
            });
    }

    // Performance section
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Performance");
    ui.add_space(4.0);

    // Read current values first; closures capture them by copy, write-back happens after.
    let mut lf_interval = state.world_settings.light_field_update_interval;
    let mut phz = state.world_settings.physics_hz;
    let mut max_steps = state.world_settings.max_physics_steps_per_frame;

    ui.label("Light Field Interval:")
        .on_hover_text("Recompute the shadow/light voxel grid every N render frames. 1 = every frame (best quality). /2 /4 /8 = progressively cheaper with minor shadow lag on fast-moving organisms.");
    ui.horizontal(|ui| {
        for &n in &[1u32, 2, 4, 8] {
            let label = if n == 1 {
                "Every".to_owned()
            } else {
                format!("/{n}")
            };
            if ui.selectable_label(lf_interval == n, label).clicked() {
                lf_interval = n;
            }
        }
    });
    ui.label(egui::RichText::new("Every = full quality  /2 /4 /8 = progressively cheaper").small());

    ui.add_space(6.0);

    ui.label("Physics Rate:")
        .on_hover_text("Fixed timestep frequency. Lower Hz = cheaper but coarser integration; affects cell growth timing and spring stiffness. 64 Hz is the default.");
    ui.horizontal(|ui| {
        for &hz in &[32u32, 48, 64] {
            if ui.selectable_label(phz == hz, format!("{hz} Hz")).clicked() {
                phz = hz;
            }
        }
    });
    ui.label(egui::RichText::new("Lower Hz = faster but coarser simulation").small());

    ui.add_space(6.0);

    ui.label("Max Steps / Frame:")
        .on_hover_text("Maximum physics steps per render frame at 1x speed. Lower = better frame times at high simulation speeds; higher = simulation keeps up during fast-forward.");
    ui.add(egui::Slider::new(&mut max_steps, 1..=16));
    ui.label(
        egui::RichText::new(
            "Caps physics budget per frame (lower = smoother, may slow sim at high speed)",
        )
        .small(),
    );

    state.world_settings.light_field_update_interval = lf_interval;
    state.world_settings.physics_hz = phz;
    state.world_settings.max_physics_steps_per_frame = max_steps;

    if changed || sun_changed {
        apply_light_panel_changes(context, sun_changed);
    }

    ui.add_space(10.0);
}

/// Render the Lighting settings panel.
#[allow(dead_code)]
fn render_light_settings(ui: &mut Ui, context: &mut PanelContext, state: &GlobalUiState) {
    let sw = (ui.available_width() - 60.0).max(60.0);
    ui.style_mut().spacing.slider_width = sw;
    if !context.is_gpu_mode() {
        ui.label("Lighting settings are only available in GPU mode.");
        return;
    }

    let has_light_field = context
        .scene_manager
        .gpu_scene()
        .map(|s| s.light_field_system.is_some())
        .unwrap_or(false);

    if !has_light_field {
        ui.label("Light field system not initialized.");
        ui.add_space(5.0);
        ui.label("Initialize the fluid system first to enable lighting.");
        return;
    }

    let mut changed = false;
    let mut sun_changed = false;

    // === Light Direction ===
    ui.add_space(4.0);
    ui.heading("Light Direction");
    ui.add_space(4.0);

    context.editor_state.sun_rotation_enabled = false;
    context.editor_state.sun_rotation_speed = 0.0;
    changed |= render_sun_direction_control(ui, &mut context.editor_state.light_dir);

    if false {
        let rotating = context.editor_state.sun_rotation_enabled;
        if rotating {
            ui.label(
                egui::RichText::new("⟳ Controlled by rotation — disable rotation to set manually")
                    .italics()
                    .weak(),
            );
            ui.add_space(2.0);
        }

        ui.add_enabled_ui(!rotating, |ui| {
        ui.label("X:").on_hover_text(
            "X component of the light direction vector. Negative = light comes from the right",
        );
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.light_dir[0], -1.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed();
        ui.label("Y:").on_hover_text(
            "Y component of the light direction vector. Positive = light comes from above (top-down)",
        );
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.light_dir[1], -1.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed();
        ui.label("Z:")
            .on_hover_text("Z component of the light direction vector");
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.light_dir[2], -1.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed();

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
    });
    }

    // === Sun ===
    ui.add_space(8.0);
    ui.separator();
    ui.heading("Sun");
    ui.add_space(4.0);

    ui.label("Brightness (0-5):")
        .on_hover_text("Overall intensity of the directional sun light (0=dark, 3=recommended, 5=extreme heat). Also affects how much energy photocyte cells generate from light");
    let brightness_changed = ui
        .add(
            egui::Slider::new(&mut context.editor_state.sun_intensity, 0.0..=5.0)
                .step_by(0.1)
                .fixed_decimals(1),
        )
        .changed();
    if brightness_changed {
        changed = true;
        if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
            gpu_scene.sun_intensity = context.editor_state.sun_intensity;
        }
    }

    changed |= ui
        .checkbox(&mut context.editor_state.sun_cycle_enabled, "Cycle Brightness (Seasons)")
        .on_hover_text("Oscillate sun brightness between Min and Max over the configured period. Simulates seasonal light variation.")
        .changed();

    if context.editor_state.sun_cycle_enabled {
        ui.add_space(4.0);
        ui.label("Min Brightness:")
            .on_hover_text("Dimmest point of the season cycle");
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.sun_cycle_min, 0.0..=20.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed();
        ui.label("Max Brightness:")
            .on_hover_text("Brightest point of the season cycle");
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.sun_cycle_max, 0.0..=20.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed();
        ui.label("Period (seconds):")
            .on_hover_text("How long one full season cycle takes");
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.sun_cycle_period, 10.0..=3600.0)
                    .step_by(1.0)
                    .fixed_decimals(0),
            )
            .changed();
        ui.add_space(4.0);
    }

    sun_changed |= ui
        .checkbox(&mut context.editor_state.show_sun, "Show Sun Disc")
        .on_hover_text("Render a visible sun disc in the skybox at the light direction position")
        .changed();

    if context.editor_state.show_sun {
        ui.add_space(4.0);
        let mut sc = context.editor_state.sun_color;
        sun_changed |= render_sun_tint_control(ui, &mut sc, context.editor_state.sun_intensity);
        context.editor_state.sun_color = sc;

        if state.show_advanced_options {
            ui.add_space(4.0);
            ui.label("Sun Size:")
                .on_hover_text("Angular radius of the sun disc in the skybox. Larger values create a bigger, more diffuse sun");
            sun_changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.sun_angular_radius, 0.005..=0.2)
                        .step_by(0.005)
                        .fixed_decimals(3),
                )
                .changed();
        }
    } // end show_sun

    // === Advanced: Shadows, Light Field, Fog, DoF ===
    if state.show_advanced_options {
        // Surface Shadows
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Surface Shadows");
        ui.add_space(4.0);

        let mut shadow_changed = false;
        shadow_changed |= ui
            .checkbox(
                &mut context.editor_state.shadow_enabled,
                "Enable Surface Shadows",
            )
            .on_hover_text(
                "Cast shadows from cave geometry and cells onto surfaces using the light field",
            )
            .changed();

        ui.add_space(4.0);
        ui.label("Shadow Strength:").on_hover_text(
            "How dark the shadows are. 0 = no shadows visible; 1 = fully opaque black shadows",
        );
        shadow_changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.shadow_strength, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed();

        ui.add_space(4.0);
        ui.label("Shadow Quality:")
            .on_hover_text("Sampling quality for shadow rays. Higher values reduce noise and banding but increase GPU cost");
        shadow_changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.shadow_quality, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            )
            .changed();

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
    } // temporary close - rest of advanced below

    if state.show_advanced_options {
        // Light Field
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Light Field");
        ui.add_space(4.0);

        ui.label("Ray Steps:")
            .on_hover_text("Number of steps each light ray takes through the voxel grid. More steps = more accurate shadows and lighting but higher GPU cost");
        let mut steps = context.editor_state.light_field_max_steps as i32;
        if ui.add(egui::Slider::new(&mut steps, 1..=256)).changed() {
            context.editor_state.light_field_max_steps = steps as u32;
            changed = true;
        }

        ui.label("Step Size:")
            .on_hover_text("Distance between ray samples in world units. Smaller steps = more accurate but slower. Larger steps = faster but may miss thin geometry");
        changed |= ui
            .add(
                egui::Slider::new(&mut context.editor_state.light_field_step_size, 0.5..=4.0)
                    .step_by(0.1)
                    .fixed_decimals(1),
            )
            .changed();

        ui.label("Solid Absorption:")
            .on_hover_text("How much light is blocked per unit of solid (cave) material. Higher values create darker shadows behind cave walls");
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut context.editor_state.light_field_absorption_solid,
                    0.1..=20.0,
                )
                .step_by(0.1)
                .fixed_decimals(1),
            )
            .changed();

        ui.label("Cell Absorption:")
            .on_hover_text("How much light is blocked per unit of cell material. Higher values make cells cast darker shadows on each other");
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut context.editor_state.light_field_absorption_cell,
                    0.0..=5.0,
                )
                .step_by(0.05)
                .fixed_decimals(2),
            )
            .changed();

        ui.label("Ambient Floor:")
            .on_hover_text("Minimum light level in fully shadowed areas. Prevents completely black shadows — simulates indirect/ambient light bouncing");
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut context.editor_state.light_field_ambient_floor,
                    0.0..=0.2,
                )
                .step_by(0.005)
                .fixed_decimals(3),
            )
            .changed();

        // Volumetric Fog
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Volumetric Fog");
        ui.add_space(4.0);
        // Fog is toggled via the  rail button. Sliders shown when active.
        if context.editor_state.show_volumetric_fog {
            ui.add_space(4.0);
            ui.label("Fog Density:")
                .on_hover_text("Overall thickness of the fog. Higher values create denser, more opaque fog that obscures distant objects");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_density, 0.0..=2.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                )
                .changed();

            ui.label("Fog Steps:")
                .on_hover_text("Number of ray march steps for volumetric fog. More steps = smoother, more accurate fog but higher GPU cost");
            let mut fsteps = context.editor_state.fog_steps as i32;
            if ui.add(egui::Slider::new(&mut fsteps, 8..=128)).changed() {
                context.editor_state.fog_steps = fsteps as u32;
                changed = true;
            }

            ui.label("Scattering Anisotropy:")
                .on_hover_text("Direction of light scattering. 0 = uniform scatter in all directions. Higher values create a forward-scattering glow around the light source (Mie scattering)");
            changed |= ui
                .add(
                    egui::Slider::new(
                        &mut context.editor_state.fog_scattering_anisotropy,
                        0.0..=0.95,
                    )
                    .step_by(0.01)
                    .fixed_decimals(2),
                )
                .changed();

            ui.label("Absorption:")
                .on_hover_text("How much light the fog absorbs (as opposed to scattering). Higher values make the fog darker and more opaque");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_absorption, 0.0..=2.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                )
                .changed();

            ui.label("Scatter Intensity:")
                .on_hover_text("How brightly the fog glows when lit by a light source. Controls both sun god-rays and luminocyte halos.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.light_intensity, 0.0..=20.0)
                        .step_by(0.1)
                        .fixed_decimals(1),
                )
                .changed();

            ui.add_space(4.0);
            ui.label("Fog Colour:")
                .on_hover_text("Ambient/shadow colour of the fog medium itself.");
            if ui
                .color_edit_button_rgb(&mut context.editor_state.fog_color)
                .changed()
            {
                changed = true;
            }

            ui.add_space(4.0);
            ui.label("Height Fog Density:")
                .on_hover_text("Density of fog that accumulates at the bottom of the world. Creates a ground-hugging mist effect.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_height_density, 0.0..=2.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                )
                .changed();

            ui.label("Height Fog Falloff:").on_hover_text(
                "How quickly the height fog thins with altitude. Smaller = fog extends higher.",
            );
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_height_falloff, 0.001..=0.1)
                        .step_by(0.001)
                        .fixed_decimals(3),
                )
                .changed();

            ui.add_space(4.0);
            ui.add_space(4.0);
            changed |= ui
                .checkbox(&mut context.editor_state.fog_smooth_light_field, "Smooth Voxels")
                .on_hover_text("Trilinear interpolation of the light field — eliminates hard blocky voxel edges on light shafts and luminocyte halos. Costs ~8x more light field reads per step.")
                .changed();

            ui.label("Blur Radius:")
                .on_hover_text("Smoothing kernel applied to the half-res fog before upscaling. Higher values hide grain but soften beam edges. 0.5 = sharp, 1.5 = balanced, 3+ = very soft.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_composite_blur, 0.25..=4.0)
                        .step_by(0.25)
                        .fixed_decimals(2),
                )
                .changed();

            ui.label("Water Wave Strength:")
                .on_hover_text("Distorts light shafts passing through water surfaces, creating caustic-like shimmer.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_water_wave_strength, 0.0..=2.0)
                        .step_by(0.05)
                        .fixed_decimals(2),
                )
                .changed();

            ui.label("Water Wave Scale:")
                .on_hover_text("Spatial frequency of the water wave distortion.");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.fog_water_wave_scale, 0.01..=1.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                )
                .changed();
        }

        // Depth of Field
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Depth of Field");
        ui.add_space(4.0);

        changed |= ui.checkbox(&mut context.editor_state.show_dof, "Enable Depth of Field")
            .on_hover_text("Blur objects that are out of the camera's focal range, simulating a camera lens effect")
            .changed();

        ui.add_space(8.0);
        ui.separator();
        ui.label(egui::RichText::new("Contrast & Eye Adaptation").strong());

        // Contrast slider - always active.
        ui.label("Contrast:").on_hover_text(
            "Midpoint contrast. Values above 1 separate lights and darks; 1.0 = neutral.",
        );
        let contrast_changed = ui
            .add(
                egui::Slider::new(&mut context.editor_state.pp_contrast, 0.25..=4.0)
                    .step_by(0.05)
                    .fixed_decimals(2),
            )
            .changed();
        if contrast_changed {
            changed = true;
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                if let Some(pp) = gpu_scene.post_process.as_mut() {
                    pp.contrast = context.editor_state.pp_contrast;
                }
            }
        }

        // Eye adaptation toggle.
        let adapt_enabled_changed = ui.checkbox(&mut context.editor_state.pp_adapt_enabled, "Eye Adaptation")
            .on_hover_text("Camera gradually adjusts exposure when moving between bright and dark areas, like the human eye.")
            .changed();
        if adapt_enabled_changed {
            changed = true;
            if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                if let Some(pp) = gpu_scene.post_process.as_mut() {
                    pp.adapt_enabled = context.editor_state.pp_adapt_enabled;
                }
            }
        }

        if context.editor_state.pp_adapt_enabled {
            ui.add_space(4.0);
            ui.label("Adaptation Speed:").on_hover_text(
                "How quickly the eye adjusts. 0.01 = very slow (cinematic), 0.3 = fast (arcade).",
            );
            let adapt_speed_changed = ui
                .add(
                    egui::Slider::new(&mut context.editor_state.pp_adapt_speed, 0.01..=0.4)
                        .step_by(0.01)
                        .fixed_decimals(2),
                )
                .changed();
            if adapt_speed_changed {
                changed = true;
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    if let Some(pp) = gpu_scene.post_process.as_mut() {
                        pp.adapt_speed = context.editor_state.pp_adapt_speed;
                    }
                }
            }
            ui.label("Min Exposure:").on_hover_text(
                "Darkest the camera can get (e.g. 0.1 = very dark in full sunlight).",
            );
            let adapt_min_changed = ui
                .add(
                    egui::Slider::new(&mut context.editor_state.pp_adapt_min, 0.05..=2.0)
                        .step_by(0.05)
                        .fixed_decimals(2),
                )
                .changed();
            if adapt_min_changed {
                changed = true;
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    if let Some(pp) = gpu_scene.post_process.as_mut() {
                        pp.adapt_min = context.editor_state.pp_adapt_min;
                    }
                }
            }
            ui.label("Max Exposure:").on_hover_text(
                "Brightest the camera can get (e.g. 6.0 = very bright in a dark cave).",
            );
            let adapt_max_changed = ui
                .add(
                    egui::Slider::new(&mut context.editor_state.pp_adapt_max, 1.0..=20.0)
                        .step_by(0.5)
                        .fixed_decimals(1),
                )
                .changed();
            if adapt_max_changed {
                changed = true;
                if let Some(gpu_scene) = context.scene_manager.gpu_scene_mut() {
                    if let Some(pp) = gpu_scene.post_process.as_mut() {
                        pp.adapt_max = context.editor_state.pp_adapt_max;
                    }
                }
            }
        }

        ui.add_space(8.0);
        ui.separator();

        if context.editor_state.show_dof {
            ui.add_space(4.0);
            ui.label("Focal Distance:")
                .on_hover_text("Distance from the camera at which objects are in perfect focus");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.dof_focal_distance, 5.0..=500.0)
                        .step_by(1.0)
                        .fixed_decimals(0),
                )
                .changed();

            ui.label("Focal Range:")
                .on_hover_text("Depth of the in-focus zone. Objects within this distance from the focal point remain sharp");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.dof_focal_range, 1.0..=200.0)
                        .step_by(1.0)
                        .fixed_decimals(0),
                )
                .changed();

            ui.label("Max Blur Radius:")
                .on_hover_text("Maximum blur radius in pixels for objects far outside the focal range. Higher values create a more dramatic bokeh effect");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.dof_max_blur_radius, 1.0..=24.0)
                        .step_by(0.5)
                        .fixed_decimals(1),
                )
                .changed();

            ui.label("Blur Strength:")
                .on_hover_text("Overall intensity of the depth-of-field blur effect. 0 = no blur; higher values create stronger out-of-focus blurring");
            changed |= ui
                .add(
                    egui::Slider::new(&mut context.editor_state.dof_blur_strength, 0.0..=3.0)
                        .step_by(0.05)
                        .fixed_decimals(2),
                )
                .changed();
        }
    } // end advanced lighting sections

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
        .add(egui::Slider::new(&mut time_value, 0.0..=10.0))
        .changed()
    {
        context.editor_state.time_value = time_value;
    }
}

/// Render the ThemeEditor panel - full custom theme color editor.
fn render_theme_editor(ui: &mut Ui, state: &mut GlobalUiState) {
    use crate::ui::types::{ThemeColor, UiTheme};

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            // -- Theme selector ------------------------------------------------
            ui.label(egui::RichText::new("Active Theme").strong());
            ui.add_space(4.0);
            egui::ComboBox::from_id_salt("theme_editor_selector")
                .selected_text(state.selected_theme.display_name())
                .show_ui(ui, |ui| {
                    for &theme in UiTheme::all() {
                        ui.selectable_value(&mut state.selected_theme, theme, theme.display_name());
                    }
                });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // Only show the editor when Custom is selected
            if state.selected_theme != UiTheme::Custom {
                ui.label(
                    egui::RichText::new("Select \"Custom\" to edit colors.")
                        .color(palette().text_dim),
                );
                ui.add_space(8.0);
                if ui.button("Copy current theme to Custom").clicked() {
                    // Seed the custom palette from the currently active palette
                    let p = palette();
                    state.custom_theme = crate::ui::types::CustomThemePalette {
                        bg_darkest: ThemeColor::from_egui(p.bg_darkest),
                        bg_panel: ThemeColor::from_egui(p.bg_panel),
                        bg_widget: ThemeColor::from_egui(p.bg_widget),
                        bg_hover: ThemeColor::from_egui(p.bg_hover),
                        bg_active: ThemeColor::from_egui(p.bg_active),
                        bg_selected: ThemeColor::from_egui(p.bg_selected),
                        accent_primary: ThemeColor::from_egui(p.accent_primary),
                        accent_secondary: ThemeColor::from_egui(p.accent_secondary),
                        text_primary: ThemeColor::from_egui(p.text_primary),
                        text_secondary: ThemeColor::from_egui(p.text_secondary),
                        text_dim: ThemeColor::from_egui(p.text_dim),
                        border_subtle: ThemeColor::from_egui(p.border_subtle),
                        border_normal: ThemeColor::from_egui(p.border_normal),
                        border_bright: ThemeColor::from_egui(p.border_bright),
                        topbar_bg: ThemeColor::from_egui(p.topbar_bg),
                        topbar_border: ThemeColor::from_egui(p.topbar_border),
                        status_ok: ThemeColor::from_egui(p.status_ok),
                        status_warn: ThemeColor::from_egui(p.status_warn),
                        status_err: ThemeColor::from_egui(p.status_err),
                        status_info: ThemeColor::from_egui(p.status_info),
                        dark_mode: p.bg_darkest.r() < 128,
                    };
                    state.selected_theme = UiTheme::Custom;
                }
                return;
            }

            // -- Dark mode toggle ----------------------------------------------
            ui.label(egui::RichText::new("Mode").strong());
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.radio_value(&mut state.custom_theme.dark_mode, true, "Dark");
                ui.radio_value(&mut state.custom_theme.dark_mode, false, "Light");
            });
            ui.add_space(8.0);

            // Helper macro: one color row with label + color picker
            macro_rules! color_row {
                ($label:expr, $tooltip:expr, $field:expr) => {{
                    let mut c = $field.to_egui();
                    ui.label($label).on_hover_text($tooltip);
                    if ui.color_edit_button_srgba(&mut c).changed() {
                        $field = ThemeColor::from_egui(c);
                    }
                    ui.add_space(2.0);
                }};
            }

            // -- Backgrounds ---------------------------------------------------
            ui.label(egui::RichText::new("Backgrounds").strong());
            ui.add_space(4.0);
            color_row!(
                "Darkest Background",
                "Deepest background — window chrome, gutters",
                state.custom_theme.bg_darkest
            );
            color_row!(
                "Panel Background",
                "Panel and window fill color",
                state.custom_theme.bg_panel
            );
            color_row!(
                "Widget Background",
                "Input fields, buttons, inactive widgets",
                state.custom_theme.bg_widget
            );
            color_row!(
                "Hover Background",
                "Widget background when hovered",
                state.custom_theme.bg_hover
            );
            color_row!(
                "Active Background",
                "Widget background when pressed or active",
                state.custom_theme.bg_active
            );
            color_row!(
                "Selected Background",
                "Selection highlight background",
                state.custom_theme.bg_selected
            );

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // -- Accents -------------------------------------------------------
            ui.label(egui::RichText::new("Accents").strong());
            ui.add_space(4.0);
            color_row!(
                "Primary Accent",
                "Main accent — active tabs, highlights, rail buttons",
                state.custom_theme.accent_primary
            );
            color_row!(
                "Secondary Accent",
                "Secondary accent — hyperlinks, secondary highlights",
                state.custom_theme.accent_secondary
            );

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // -- Text ----------------------------------------------------------
            ui.label(egui::RichText::new("Text").strong());
            ui.add_space(4.0);
            color_row!(
                "Primary Text",
                "Main body text color",
                state.custom_theme.text_primary
            );
            color_row!(
                "Secondary Text",
                "Labels, captions, less important text",
                state.custom_theme.text_secondary
            );
            color_row!(
                "Dim Text",
                "Placeholder text, disabled labels",
                state.custom_theme.text_dim
            );

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // -- Borders -------------------------------------------------------
            ui.label(egui::RichText::new("Borders").strong());
            ui.add_space(4.0);
            color_row!(
                "Subtle Border",
                "Faint dividers, panel edges",
                state.custom_theme.border_subtle
            );
            color_row!(
                "Normal Border",
                "Standard widget borders",
                state.custom_theme.border_normal
            );
            color_row!(
                "Bright Border",
                "Focused/active widget borders, accent outlines",
                state.custom_theme.border_bright
            );

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // -- Top bar -------------------------------------------------------
            ui.label(egui::RichText::new("Top Bar").strong());
            ui.add_space(4.0);
            color_row!(
                "Top Bar Background",
                "Background of the top menu bar",
                state.custom_theme.topbar_bg
            );
            color_row!(
                "Top Bar Border",
                "Bottom border line of the top bar",
                state.custom_theme.topbar_border
            );

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // -- Status colors -------------------------------------------------
            ui.label(egui::RichText::new("Status Colors").strong());
            ui.add_space(4.0);
            color_row!(
                "Status OK",
                "Success indicators, healthy state",
                state.custom_theme.status_ok
            );
            color_row!(
                "Status Warn",
                "Warning indicators",
                state.custom_theme.status_warn
            );
            color_row!(
                "Status Error",
                "Error indicators",
                state.custom_theme.status_err
            );
            color_row!(
                "Status Info",
                "Informational indicators",
                state.custom_theme.status_info
            );

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(4.0);

            // -- Reset ---------------------------------------------------------
            if ui
                .button("Reset to Defaults")
                .on_hover_text("Reset custom theme to Biotech Dark defaults")
                .clicked()
            {
                state.custom_theme = crate::ui::types::CustomThemePalette::default();
            }
        });
}

/// Render the CameraSettings panel (placeholder).
fn render_camera_settings(ui: &mut Ui, context: &mut PanelContext, ui_state: &mut GlobalUiState) {
    let sw = (ui.available_width() - 60.0).max(60.0);
    ui.style_mut().spacing.slider_width = sw;
    ui.heading("Camera");
    ui.separator();

    let camera = &mut *context.camera;
    ui.label(format!("Mode: {:?}", camera.mode));
    ui.label(format!("Distance: {:.1}", camera.distance));

    ui.label("Move Speed:").on_hover_text(
        "Camera movement speed in FreeFly mode. Higher values let you traverse the world faster",
    );
    ui.add(egui::Slider::new(&mut camera.move_speed, 1.0..=50.0).logarithmic(true));
    ui.label("Sprint Multiplier:").on_hover_text(
        "Speed multiplier applied while holding Shift in FreeFly mode. Adjust with Shift+Scroll while flying",
    );
    let sprint_response = ui.add(
        egui::Slider::new(&mut ui_state.camera_sprint_multiplier, 1.0..=20.0)
            .custom_formatter(|value, _| format!("{value:.1}x")),
    );
    let mut sprint_multiplier_changed = sprint_response.changed();
    if sprint_response.double_clicked() {
        ui_state.camera_sprint_multiplier = 6.0;
        sprint_multiplier_changed = true;
    }
    if sprint_multiplier_changed {
        camera.sprint_multiplier = ui_state.camera_sprint_multiplier;
    }
    ui.label("Zoom Speed:").on_hover_text(
        "How fast scrolling zooms the camera in Orbit mode. Adjust with Shift+Scroll while orbiting",
    );
    let zoom_response = ui.add(
        egui::Slider::new(&mut ui_state.camera_scroll_sensitivity, 0.01..=2.0)
            .logarithmic(true)
            .custom_formatter(|value, _| format!("{value:.2}x")),
    );
    let mut scroll_sensitivity_changed = zoom_response.changed();
    if zoom_response.double_clicked() {
        ui_state.camera_scroll_sensitivity = 0.2;
        scroll_sensitivity_changed = true;
    }
    if scroll_sensitivity_changed {
        camera.zoom_speed = ui_state.camera_scroll_sensitivity;
    }
    ui.label("Mouse Sensitivity:")
        .on_hover_text("How much the camera rotates per pixel of mouse movement. Lower values give finer control; higher values feel more responsive");
    ui.add(egui::Slider::new(&mut camera.mouse_sensitivity, 0.001..=0.01).logarithmic(true));
    ui.checkbox(&mut camera.enable_spring, "Enable Spring")
        .on_hover_text("Adds a spring/lag to camera movement for a smoother cinematic feel. Disable for instant, precise camera control");
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
    context
        .editor_state
        .panel_rects
        .insert("Modes".to_string(), ui.max_rect());

    if context.genome.modes.is_empty() {
        log::warn!("Genome has no modes, this should not happen with default genome");
        return;
    }

    let selected_index = context
        .editor_state
        .selected_mode_index
        .min(context.genome.modes.len().saturating_sub(1));
    context.editor_state.selected_mode_index = selected_index;

    // -- Type + Make Adhesion at the very top -----------------------------
    if selected_index < context.genome.modes.len() {
        let cell_types = crate::cell::types::CellType::all();

        // Type dropdown - full width
        let type_before = context.genome.modes[selected_index].cell_type;
        let combo_resp = egui::ComboBox::from_id_salt("modes_panel_cell_type")
            .selected_text(
                egui::RichText::new(
                    cell_types[context.genome.modes[selected_index].cell_type as usize].name(),
                )
                .size(11.5)
                .color(palette().text_primary),
            )
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                let item_h = ui.text_style_height(&egui::TextStyle::Body) + 6.0;
                for ct in cell_types.iter() {
                    let selected =
                        context.genome.modes[selected_index].cell_type == ct.to_index() as i32;
                    let resp = ui
                        .add_sized(
                            [ui.available_width(), item_h],
                            egui::Button::new(ct.name()).selected(selected),
                        )
                        .on_hover_text(ct.tooltip());
                    if resp.clicked() {
                        context.genome.modes[selected_index].cell_type = ct.to_index() as i32;
                    }
                }
            });
        // Propagate cell type change to all selected modes.
        // Don't rely on combo_resp.response.changed() - it reflects the button widget,
        // not the inner selectable_value. Compare before/after instead.
        let type_after = context.genome.modes[selected_index].cell_type;
        if type_after != type_before {
            for &idx in &context.editor_state.selected_mode_indices {
                if idx < context.genome.modes.len() {
                    context.genome.modes[idx].cell_type = type_after;
                }
            }
        }
        context
            .editor_state
            .panel_rects
            .insert("cell_type_dropdown".to_string(), combo_resp.response.rect);

        // Make Adhesion toggle - full width, teal when active
        let on = context.genome.modes[selected_index].parent_make_adhesion;
        let (fill, text_col, stroke_col) = if on {
            let accent = palette().accent_primary;
            (
                accent,
                crate::ui::widgets::color_utils::text_color_for_background(accent),
                accent,
            )
        } else {
            (
                palette().bg_widget,
                palette().text_secondary,
                palette().border_normal,
            )
        };
        let label = if on {
            "Make Adhesion  ON"
        } else {
            "Make Adhesion  OFF"
        };
        let adh_resp = ui.add_sized(
            egui::vec2(ui.available_width(), 22.0),
            egui::Button::new(egui::RichText::new(label).size(11.0).color(text_col))
                .fill(fill)
                .stroke(egui::Stroke::new(1.0, stroke_col))
                .corner_radius(egui::CornerRadius::same(3)),
        );
        if adh_resp.clicked() {
            let new_val = !on;
            for &idx in &context.editor_state.selected_mode_indices {
                if idx < context.genome.modes.len() {
                    context.genome.modes[idx].parent_make_adhesion = new_val;
                }
            }
        }
        context
            .editor_state
            .panel_rects
            .insert("make_adhesion_checkbox".to_string(), adh_resp.rect);

        ui.add_space(4.0);
    }

    // -- Mode list controls ------------------------------------------------
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

    // Control buttons row: Copy Into, Reset, Add (+), Remove (-)
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let (copy_into_clicked, reset_clicked) =
            modes_buttons(ui, context.genome.modes.len(), selected_index, initial_mode);

        // Add mode button
        let at_cap = context.genome.modes.len() >= crate::genome::MAX_MODES;
        let add_btn = ui.add_enabled(
            !at_cap,
            egui::Button::new("+").min_size(egui::Vec2::new(20.0, 0.0)),
        );
        if add_btn
            .on_hover_text(if at_cap {
                format!("At maximum ({} modes)", crate::genome::MAX_MODES)
            } else {
                format!(
                    "Insert mode after selected ({}/{})",
                    context.genome.modes.len(),
                    crate::genome::MAX_MODES
                )
            })
            .clicked()
        {
            let after_idx = context
                .editor_state
                .selected_mode_index
                .min(context.genome.modes.len().saturating_sub(1));
            if let Some(new_idx) = context.genome.insert_mode_after(after_idx) {
                context.editor_state.selected_mode_index = new_idx;
                context.editor_state.selected_mode_indices = vec![new_idx];
                log::info!("Inserted mode after {}", after_idx);
            }
        }

        // Remove selected modes button. The initial mode is preserved by the genome.
        let removable_selected_count = context
            .editor_state
            .selected_mode_indices
            .iter()
            .filter(|&&idx| idx < context.genome.modes.len() && idx != initial_mode)
            .count();
        let can_remove = removable_selected_count > 0;
        let remove_btn = ui.add_enabled(
            can_remove,
            egui::Button::new("−").min_size(egui::Vec2::new(20.0, 0.0)),
        );
        if remove_btn
            .on_hover_text(if can_remove {
                if removable_selected_count == 1 {
                    "Remove selected mode".to_string()
                } else {
                    format!("Remove {} selected modes", removable_selected_count)
                }
            } else {
                "Select a non-initial mode to remove".to_string()
            })
            .clicked()
        {
            let old_selected = context.editor_state.selected_mode_index;
            let old_selection = context.editor_state.selected_mode_indices.clone();
            let removed = context
                .genome
                .remove_modes_except_initial(&context.editor_state.selected_mode_indices);

            if !removed.is_empty() {
                let removed_before = |idx: usize| removed.iter().filter(|&&r| r < idx).count();
                context.editor_state.selected_mode_indices = old_selection
                    .iter()
                    .copied()
                    .filter(|idx| !removed.contains(idx))
                    .map(|idx| idx - removed_before(idx))
                    .filter(|&idx| idx < context.genome.modes.len())
                    .collect();

                context.editor_state.selected_mode_index = if removed.contains(&old_selected) {
                    context.genome.initial_mode.max(0) as usize
                } else {
                    old_selected - removed_before(old_selected)
                }
                .min(context.genome.modes.len().saturating_sub(1));

                if context.editor_state.selected_mode_indices.is_empty() {
                    context.editor_state.selected_mode_indices =
                        vec![context.editor_state.selected_mode_index];
                } else if !context
                    .editor_state
                    .selected_mode_indices
                    .contains(&context.editor_state.selected_mode_index)
                {
                    context
                        .editor_state
                        .selected_mode_indices
                        .push(context.editor_state.selected_mode_index);
                    context.editor_state.selected_mode_indices.sort_unstable();
                    context.editor_state.selected_mode_indices.dedup();
                }

                let remap_surviving_index = |idx: usize| {
                    if removed.contains(&idx) {
                        None
                    } else {
                        let remapped = idx - removed_before(idx);
                        (remapped < context.genome.modes.len()).then_some(remapped)
                    }
                };

                context.editor_state.renaming_mode = context
                    .editor_state
                    .renaming_mode
                    .and_then(remap_surviving_index);
                context.editor_state.color_picker_state = context
                    .editor_state
                    .color_picker_state
                    .take()
                    .and_then(|(idx, hsva)| remap_surviving_index(idx).map(|idx| (idx, hsva)));

                selected_index = context.editor_state.selected_mode_index;
                initial_mode = context.genome.initial_mode.max(0) as usize;
            }
            log::info!(
                "Removed modes {:?}, now {} modes",
                removed,
                context.genome.modes.len()
            );
        }

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
            let indices_to_reset: Vec<usize> =
                if context.editor_state.selected_mode_indices.len() > 1 {
                    context
                        .editor_state
                        .selected_mode_indices
                        .iter()
                        .copied()
                        .filter(|&i| i < context.genome.modes.len())
                        .collect()
                } else {
                    if selected_index < context.genome.modes.len() {
                        vec![selected_index]
                    } else {
                        vec![]
                    }
                };

            // Seed the LCG once; each mode advances it to get a distinct color
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(12345);
            let mut rng = seed as u64;

            for idx in indices_to_reset {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = ((rng >> 33) & 0xFF) as f32 / 255.0;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let g = ((rng >> 33) & 0xFF) as f32 / 255.0;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
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
                    glueocyte_self_adhesion: false,
                    glueocyte_env_adhesion: true,
                    glueocyte_boulder_adhesion: true,
                    glueocyte_cell_adhesion_signal_channel: -1,
                    glueocyte_cell_adhesion_signal_threshold: 1.0,
                    glueocyte_signal_gate_invert: false,
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
                    myocyte_grip_contracted: 0.0,
                    myocyte_grip_extended: 0.0,
                    embryocyte_use_timer: false,
                    embryocyte_release_timer: 10.0,
                    embryocyte_use_threshold: false,
                    embryocyte_threshold_value: 32768,
                    embryocyte_use_signal: false,
                    embryocyte_signal_channel: 0,
                    embryocyte_signal_value: 1.0,
                    buoyancy_force: 0.5,
                    photocyte_emit_enabled: false,
                    photocyte_emit_channel: 0,
                    photocyte_emit_hops: 5,
                    photocyte_emit_threshold: 0.5,
                    photocyte_emit_mode: 0,
                    photocyte_emit_value: 10.0,
                    lipocyte_emit_enabled: false,
                    lipocyte_emit_channel: 0,
                    lipocyte_emit_hops: 5,
                    lipocyte_emit_threshold: 0.8,
                    lipocyte_emit_mode: 1,
                    lipocyte_emit_value: 10.0,
                    oculocyte_sense_type: 1, // bit0 = Cell
                    oculocyte_signal_channel: 0,
                    oculocyte_signal_value: 10.0,
                    oculocyte_signal_hops: 3,
                    oculocyte_ray_length: 20.0,
                    oculocyte_light_target_color: glam::Vec3::new(1.0, 0.95, 0.78),
                    oculocyte_light_color_tolerance: 0.18,
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
                    vascular_nutrient_transport: true,
                    vascular_outlet: false,
                    vascular_signal_transport: false,
                    vascular_signal_exchange: false,
                    vascular_signal_capacity: 10.0,
                    gametocyte_merge_range: 0.5,
                    memorocyte_rate: 0.1,
                    memorocyte_input_channel: 0,
                    memorocyte_output_channel: 9,
                    memorocyte_output_hops: 5,
                    cognocyte_operation: 0,
                    cognocyte_input_channel_a: 0,
                    cognocyte_input_channel_b: 1,
                    cognocyte_output_channel: 8,
                    cognocyte_output_hops: 5,
                    cognocyte_oscillator_rate: 1.0,
                    cognocyte_oscillator_phase: 0.0,
                    cognocyte_oscillator_strength: 1.0,
                    cognocyte_oscillator_step_count: 4,
                    luminocyte_signal_channel: 0,
                    luminocyte_threshold: 1.0,
                    luminocyte_invert: false,
                    siphon_intake_rate: 1.0,
                    siphon_expel_rate: 0.8,
                    siphon_impulse: 0.6,
                    siphon_signal_channel: 0,
                    siphon_signal_threshold: 1.0,
                    siphon_signal_invert: false,
                    siphon_mode: 0,
                    plumocyte_extension: 1.0,
                    plumocyte_drag_mult: 0.7,
                    plumocyte_flow_coupling: 0.5,
                    plumocyte_exposure_mult: 0.25,
                    stemocyte_signal_channel: 8,
                    stemocyte_weak_first: false,
                    stemocyte_outcomes: [-1; 5],
                    stemocyte_thresholds: [20, 40, 60, 80],
                    stemocyte_delay_mode: 0,
                    stemocyte_delay_value: 0.0,
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

    ui.add_space(4.0);

    // Show copy into mode indicator - pinned above the scrollable list
    if context.editor_state.copy_into_dialog_open {
        ui.colored_label(palette().status_warn, "Select target mode to copy into:");
        if ui.small_button("Cancel").clicked() {
            context.editor_state.copy_into_dialog_open = false;
            log::info!("Cancelled copy into mode");
        }
        ui.add_space(5.0);
    }

    // Prepare modes data for the widget (name, color tuples)
    let modes_data: Vec<(String, (u8, u8, u8))> = context
        .genome
        .modes
        .iter()
        .map(|mode| {
            let color_vec3 = mode.color;
            let r = (color_vec3.x * 255.0) as u8;
            let g = (color_vec3.y * 255.0) as u8;
            let b = (color_vec3.z * 255.0) as u8;
            (mode.name.clone(), (r, g, b))
        })
        .collect();

    let available_width = ui.available_width();
    let copy_into_mode = context.editor_state.copy_into_dialog_open;

    // Scrollable mode list - buttons above stay pinned
    let mut selection_changed = false;
    let mut initial_changed = false;
    let mut rename_completed: Option<(usize, String)> = None;
    let mut color_change: Option<(usize, (u8, u8, u8))> = None;
    let mut row_rects: Vec<egui::Rect> = Vec::new();
    let mut reorder: Option<(usize, usize)> = None;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            let (sc, ic, rc, cc, rr, ro) = modes_list_items(
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
            selection_changed = sc;
            initial_changed = ic;
            rename_completed = rc;
            color_change = cc;
            row_rects = rr;
            reorder = ro;

            // Status line inside the scroll area so it stays with the list
            if selected_index < modes_data.len() {
                ui.add_space(4.0);
                let multi_count = context.editor_state.selected_mode_indices.len();
                if multi_count > 1 {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 200, 80),
                        format!("✦ {} modes selected — edits apply to all", multi_count),
                    );
                } else if selected_index < context.genome.modes.len() {
                    ui.small(format!(
                        "Selected: {}",
                        context.genome.modes[selected_index].name
                    ));
                }
                if initial_mode < context.genome.modes.len() {
                    ui.small(format!(
                        "Initial: {}",
                        context.genome.modes[initial_mode].name
                    ));
                }
                ui.small(format!("Total: {}", modes_data.len()));
            }
        });

    // Store each mode row's rect so the tutorial arrow can point at specific rows.
    for (i, rect) in row_rects.iter().enumerate() {
        context
            .editor_state
            .panel_rects
            .insert(format!("mode_row_{}", i), *rect);
    }

    // Handle mode selection change
    if selection_changed {
        // If in copy into mode, this is the target selection
        if copy_into_mode {
            let source_idx = context.editor_state.copy_into_source;
            let target_idx = selected_index;

            if source_idx != target_idx
                && source_idx < context.genome.modes.len()
                && target_idx < context.genome.modes.len()
            {
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
            if !context
                .editor_state
                .selected_mode_indices
                .contains(&selected_index)
            {
                context.editor_state.selected_mode_indices = vec![selected_index];
            }

            // Initialize editor state orientations from the selected mode's genome data
            // This ensures the quaternion balls show the correct orientation when switching modes
            if selected_index < context.genome.modes.len() {
                context.editor_state.child_a_orientation =
                    context.genome.modes[selected_index].child_a.orientation;
                context.editor_state.child_b_orientation =
                    context.genome.modes[selected_index].child_b.orientation;
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
            context.genome.modes[mode_index].color =
                glam::Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
        }
    }

    // Handle drag reorder
    if let Some((from, to)) = reorder {
        let n = context.genome.modes.len();
        // `to` is the raw pre-removal insertion slot (0..=n), so allow to == n
        // for "insert after last item".
        if from < n && to <= n && from != to {
            // Determine which indices to move: the whole multi-selection if the dragged
            // item is part of it, otherwise just the single dragged item.
            let mut moving: Vec<usize> =
                if context.editor_state.selected_mode_indices.contains(&from)
                    && context.editor_state.selected_mode_indices.len() > 1
                {
                    let mut v = context.editor_state.selected_mode_indices.clone();
                    v.sort_unstable();
                    v
                } else {
                    vec![from]
                };
            moving.sort_unstable();

            // Only proceed if the drop target actually changes position
            // (skip if dropping inside the selected block itself)
            let min_sel = *moving.first().unwrap();
            let max_sel = *moving.last().unwrap();
            let drop_inside = to >= min_sel && to <= max_sel + 1;
            if !drop_inside || moving.len() == 1 {
                // Extract the modes to move (back-to-front to keep indices valid)
                let mut extracted: Vec<crate::genome::ModeSettings> =
                    Vec::with_capacity(moving.len());
                for &idx in moving.iter().rev() {
                    extracted.insert(0, context.genome.modes.remove(idx));
                }

                // Compute insertion point: how many selected indices are before `to`
                let before_count = moving.iter().filter(|&&i| i < to).count();
                let insert_at = (to - before_count).min(context.genome.modes.len());

                // Insert all extracted modes at the new position
                for (offset, mode) in extracted.into_iter().enumerate() {
                    context.genome.modes.insert(insert_at + offset, mode);
                }

                // Build a full remap table: old_index -> new_index
                // After removal and re-insertion, compute where each original index ended up.
                let mut remap_table: Vec<usize> = (0..n).collect();
                {
                    // Simulate the same remove+insert on the index table
                    let mut table: Vec<usize> = (0..n).collect();
                    let mut extracted_indices: Vec<usize> = Vec::with_capacity(moving.len());
                    for &idx in moving.iter().rev() {
                        extracted_indices.insert(0, table.remove(idx));
                    }
                    let before_count = moving.iter().filter(|&&i| i < to).count();
                    let insert_at = (to - before_count).min(table.len());
                    for (offset, orig) in extracted_indices.into_iter().enumerate() {
                        table.insert(insert_at + offset, orig);
                    }
                    // table[new_pos] = old_index; invert to get old_index -> new_pos
                    for (new_pos, &old_idx) in table.iter().enumerate() {
                        remap_table[old_idx] = new_pos;
                    }
                }

                let remap = |idx: i32| -> i32 {
                    if idx < 0 || idx as usize >= n {
                        return idx;
                    }
                    remap_table[idx as usize] as i32
                };

                for mode in &mut context.genome.modes {
                    mode.child_a.mode_number = remap(mode.child_a.mode_number);
                    mode.child_b.mode_number = remap(mode.child_b.mode_number);
                    mode.mode_a_after_splits = remap(mode.mode_a_after_splits);
                    mode.mode_b_after_splits = remap(mode.mode_b_after_splits);
                    mode.signal_child_a_mode_above = remap(mode.signal_child_a_mode_above);
                    mode.signal_child_a_mode_below = remap(mode.signal_child_a_mode_below);
                    mode.signal_child_b_mode_above = remap(mode.signal_child_b_mode_above);
                    mode.signal_child_b_mode_below = remap(mode.signal_child_b_mode_below);
                    if mode.mode_switch_target >= 0 {
                        mode.mode_switch_target = remap(mode.mode_switch_target);
                    }
                    for outcome in &mut mode.stemocyte_outcomes {
                        if *outcome >= 0 {
                            *outcome = remap(*outcome);
                        }
                    }
                }

                context.genome.initial_mode = remap(context.genome.initial_mode);
                context.editor_state.selected_mode_index =
                    remap(context.editor_state.selected_mode_index as i32) as usize;
                for idx in &mut context.editor_state.selected_mode_indices {
                    *idx = remap(*idx as i32) as usize;
                }

                log::info!("Reordered {:?} → insert before slot {}", moving, to);
            }
        }
    }
}

/// Draw text inside `rect`, scrolling it horizontally (marquee) when hovered and too long to fit.
/// `marquee_id` must be unique per call site + item index.
/// Returns the `Response` from the allocated rect so callers can detect clicks.
fn draw_marquee_text(
    ui: &mut Ui,
    rect: egui::Rect,
    name: &str,
    text_color: egui::Color32,
    is_hovered: bool,
    marquee_id: egui::Id,
) {
    const START_DELAY_SECS: f32 = 0.6;
    const SCROLL_SPEED: f32 = 40.0;
    const END_PAUSE_SECS: f32 = 0.8;

    let font_id = egui::FontId::default();
    let text_max_width = rect.width() - 6.0;

    let full_galley = ui
        .painter()
        .layout_no_wrap(name.to_owned(), font_id.clone(), text_color);
    let text_overflows = full_galley.rect.width() > text_max_width;

    if text_overflows {
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let (mut offset, mut timer): (f32, f32) = ui
            .ctx()
            .data(|d| d.get_temp(marquee_id).unwrap_or((0.0f32, 0.0f32)));

        if is_hovered {
            timer += dt;
            if timer > START_DELAY_SECS {
                let scroll_time = timer - START_DELAY_SECS;
                let overflow = full_galley.rect.width() - text_max_width;
                let max_offset = overflow + 8.0;
                offset = (scroll_time * SCROLL_SPEED).min(max_offset);
                if offset >= max_offset {
                    let end_pause_elapsed = scroll_time - max_offset / SCROLL_SPEED;
                    if end_pause_elapsed >= END_PAUSE_SECS {
                        timer = START_DELAY_SECS;
                        offset = 0.0;
                    }
                }
            }
            ui.ctx().request_repaint();
        } else {
            offset = 0.0;
            timer = 0.0;
        }

        ui.ctx()
            .data_mut(|d| d.insert_temp(marquee_id, (offset, timer)));

        let clip_rect = rect.shrink2(egui::vec2(3.0, 0.0));
        let painter = ui.painter().with_clip_rect(clip_rect);
        let text_pos = egui::pos2(
            rect.min.x + 3.0 - offset,
            rect.center().y - full_galley.rect.height() / 2.0,
        );
        painter.galley(text_pos, full_galley, text_color);
    } else {
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            name,
            font_id,
            text_color,
        );
    }
}

fn stemocyte_response_name(outcome: i32, modes: &[(String, glam::Vec3)]) -> String {
    match outcome {
        -2 => "Death".to_string(),
        -1 => "Stem".to_string(),
        target => modes
            .get(target as usize)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Stem".to_string()),
    }
}

fn stemocyte_response_color(outcome: i32, modes: &[(String, glam::Vec3)]) -> egui::Color32 {
    match outcome {
        -2 => egui::Color32::from_rgb(170, 55, 70),
        -1 => egui::Color32::from_rgb(132, 88, 170),
        target => modes
            .get(target as usize)
            .map(|(_, color)| {
                egui::Color32::from_rgb(
                    (color.x * 255.0) as u8,
                    (color.y * 255.0) as u8,
                    (color.z * 255.0) as u8,
                )
            })
            .unwrap_or(egui::Color32::from_rgb(132, 88, 170)),
    }
}

fn draw_stemocyte_response_strip(
    ui: &mut Ui,
    weak_first: bool,
    outcomes: &mut [i32; 5],
    thresholds: &mut [u8; 4],
    modes: &[(String, glam::Vec3)],
) {
    // Repair malformed imported values while preserving at least 1% per band.
    for boundary in 0..4 {
        let min = if boundary == 0 {
            1
        } else {
            thresholds[boundary - 1] + 1
        };
        let max = 96 + boundary as u8;
        thresholds[boundary] = thresholds[boundary].clamp(min, max);
    }

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.small("W");

        let width = (ui.available_width() - 18.0).max(120.0);
        let (bar_rect, _) = ui.allocate_exact_size(egui::vec2(width, 25.0), egui::Sense::hover());
        let mut edges = [0.0_f32; 6];
        edges[0] = bar_rect.left();
        for boundary in 0..4 {
            edges[boundary + 1] =
                bar_rect.left() + bar_rect.width() * thresholds[boundary] as f32 / 100.0;
        }
        edges[5] = bar_rect.right();

        for weak_band in 0..5 {
            let outcome_index = if weak_first { weak_band } else { 4 - weak_band };
            let outcome = outcomes[outcome_index];
            let segment_rect = egui::Rect::from_min_max(
                egui::pos2(edges[weak_band], bar_rect.top()),
                egui::pos2(edges[weak_band + 1], bar_rect.bottom()),
            );
            let response = ui.interact(
                segment_rect,
                ui.make_persistent_id(("stemocyte_response_segment", weak_band)),
                egui::Sense::click(),
            );
            let color = stemocyte_response_color(outcome, modes);
            ui.painter()
                .rect_filled(segment_rect.shrink(0.5), 3.0, color);

            let percentage = ((edges[weak_band + 1] - edges[weak_band]) / bar_rect.width() * 100.0)
                .round() as u8;
            if segment_rect.width() >= 27.0 {
                let name = stemocyte_response_name(outcome, modes);
                let label = if segment_rect.width() >= 72.0 {
                    format!("{name} {percentage}%")
                } else {
                    format!("{percentage}%")
                };
                let luminance = color.r() as u16 * 3 + color.g() as u16 * 6 + color.b() as u16;
                let text_color = if luminance > 1450 {
                    egui::Color32::BLACK
                } else {
                    egui::Color32::WHITE
                };
                ui.painter().text(
                    segment_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(10.0),
                    text_color,
                );
            }

            egui::Popup::menu(&response)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .show(|ui| {
                    ui.set_min_width(170.0);
                    ui.horizontal(|ui| {
                        ui.label("Width");
                        let start = if weak_band == 0 {
                            0
                        } else {
                            thresholds[weak_band - 1]
                        };
                        let mut exact_width = percentage;
                        let max_width = if weak_band < 4 {
                            let next_edge = if weak_band == 3 {
                                100
                            } else {
                                thresholds[weak_band + 1]
                            };
                            next_edge.saturating_sub(start + 1)
                        } else {
                            99_u8.saturating_sub(thresholds[2])
                        }
                        .max(1);
                        if ui
                            .add(
                                egui::DragValue::new(&mut exact_width)
                                    .range(1..=max_width)
                                    .suffix("%"),
                            )
                            .changed()
                        {
                            if weak_band < 4 {
                                thresholds[weak_band] = start + exact_width;
                            } else {
                                thresholds[3] = 100 - exact_width;
                            }
                        }
                    });
                    ui.separator();
                    if ui
                        .selectable_label(outcome == -1, "Remain Stemocyte")
                        .clicked()
                    {
                        outcomes[outcome_index] = -1;
                        egui::Popup::close_all(ui.ctx());
                    }
                    if ui
                        .selectable_label(outcome == -2, "Enter Apoptosis")
                        .clicked()
                    {
                        outcomes[outcome_index] = -2;
                        egui::Popup::close_all(ui.ctx());
                    }
                    ui.separator();
                    for (target_index, (name, _)) in modes.iter().enumerate() {
                        if ui
                            .selectable_label(
                                outcome == target_index as i32,
                                format!("Change to {name}"),
                            )
                            .clicked()
                        {
                            outcomes[outcome_index] = target_index as i32;
                            egui::Popup::close_all(ui.ctx());
                        }
                    }
                });
        }

        for boundary in 0..4 {
            let x = edges[boundary + 1];
            let handle_rect = egui::Rect::from_center_size(
                egui::pos2(x, bar_rect.center().y),
                egui::vec2(9.0, bar_rect.height() + 8.0),
            );
            let response = ui.interact(
                handle_rect,
                ui.make_persistent_id(("stemocyte_response_boundary", boundary)),
                egui::Sense::drag(),
            );
            if response.dragged() {
                if let Some(pointer) = response.interact_pointer_pos() {
                    let requested =
                        ((pointer.x - bar_rect.left()) / bar_rect.width() * 100.0).round() as i32;
                    let min = if boundary == 0 {
                        1
                    } else {
                        thresholds[boundary - 1] as i32 + 1
                    };
                    let max = if boundary == 3 {
                        99
                    } else {
                        thresholds[boundary + 1] as i32 - 1
                    };
                    thresholds[boundary] = requested.clamp(min, max) as u8;
                }
            }
            let stroke = if response.hovered() || response.dragged() {
                egui::Stroke::new(2.0, egui::Color32::WHITE)
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(180))
            };
            ui.painter().line_segment(
                [
                    egui::pos2(x, bar_rect.top() + 2.0),
                    egui::pos2(x, bar_rect.bottom() - 2.0),
                ],
                stroke,
            );
        }

        ui.painter().rect_stroke(
            bar_rect,
            3.0,
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(100)),
            egui::StrokeKind::Inside,
        );
        ui.small("S");
    });
}

/// Helper function to create a color-coded group container
fn group_container(ui: &mut Ui, title: &str, color: egui::Color32, content: impl FnOnce(&mut Ui)) {
    // On light themes the passed accent color is too bright/saturated to read
    // as text. Detect light vs dark by checking the panel background luminance
    // and darken the color when the background is light.
    let p = palette();
    let bg = p.bg_panel;
    let luminance = bg.r() as f32 * 0.299 + bg.g() as f32 * 0.587 + bg.b() as f32 * 0.114;
    let is_light_theme = luminance > 140.0;

    // For light themes: darken the color to ~40% brightness for text,
    // keep a subtle tint for the fill/stroke.
    let text_color = if is_light_theme {
        egui::Color32::from_rgb(
            (color.r() as f32 * 0.45) as u8,
            (color.g() as f32 * 0.45) as u8,
            (color.b() as f32 * 0.45) as u8,
        )
    } else {
        color
    };

    let frame = egui::Frame::default()
        .fill(egui::Color32::from_rgba_unmultiplied(
            color.r(),
            color.g(),
            color.b(),
            18u8,
        ))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 80u8),
        ))
        .corner_radius(egui::CornerRadius::same(3u8))
        .inner_margin(egui::Margin {
            left: 8,
            right: 8,
            top: 6,
            bottom: 6,
        });

    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        let header_rect = ui
            .horizontal(|ui| {
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .size(11.5)
                        .color(text_color),
                );
            })
            .response
            .rect;
        let painter = ui.painter();
        let left_x = header_rect.left() - 6.0;
        painter.line_segment(
            [
                egui::pos2(left_x, header_rect.top() + 1.0),
                egui::pos2(left_x, header_rect.bottom() - 1.0),
            ],
            egui::Stroke::new(2.0, text_color),
        );
        ui.add_space(3.0);
        content(ui);
    });
    ui.add_space(4.0);
}

/// Render a section header with a teal left-border accent (for top-level sections).
fn section_header(ui: &mut Ui, title: &str) {
    ui.add_space(4.0);
    let resp = ui
        .horizontal(|ui| {
            ui.label(
                egui::RichText::new(title)
                    .strong()
                    .size(11.5)
                    .color(palette().accent_primary),
            );
        })
        .response;
    // Draw left-border accent
    let painter = ui.painter();
    let left_x = resp.rect.left() - 4.0;
    painter.line_segment(
        [
            egui::pos2(left_x, resp.rect.top() + 1.0),
            egui::pos2(left_x, resp.rect.bottom() - 1.0),
        ],
        egui::Stroke::new(2.5, palette().accent_primary),
    );
    // Thin separator line
    let sep_rect = egui::Rect::from_min_size(
        egui::pos2(resp.rect.left(), resp.rect.bottom() + 1.0),
        egui::vec2(ui.available_width(), 1.0),
    );
    painter.rect_filled(sep_rect, 0.0, palette().border_subtle);
    ui.add_space(3.0);
}

/// Render the AdhesionSettings panel (placeholder).
fn render_adhesion_settings(ui: &mut Ui, context: &mut PanelContext) {
    // Record panel rect for the tutorial pointer.
    context
        .editor_state
        .panel_rects
        .insert("AdhesionSettings".to_string(), ui.max_rect());

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
                ui.checkbox(&mut mode.adhesion_settings.can_break, "Adhesion Can Break")
                    .on_hover_text("When enabled, adhesion bonds will snap if the force between cells exceeds the break force threshold");

                ui.label("Adhesion Break Force:")
                    .on_hover_text("Force magnitude (in simulation units) required to snap this adhesion bond. Higher values make bonds harder to break");
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
                ui.label("Adhesion Rest Length:")
                    .on_hover_text("The natural (equilibrium) distance between two bonded cells. The spring pulls them toward this distance");
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
                ui.label("Linear Spring Stiffness:")
                    .on_hover_text("How strongly the bond resists stretching or compression along its axis. Higher values create stiffer, more rigid connections");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.linear_spring_stiffness, 0.1..=500.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.linear_spring_stiffness).speed(0.1).range(0.1..=500.0));
                });

                ui.label("Linear Spring Damping:")
                    .on_hover_text("Reduces oscillation along the bond axis. Higher values cause the spring to settle faster with less bouncing");
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
                ui.label("Orientation Spring Stiffness:")
                    .on_hover_text("How strongly the bond resists angular bending. Higher values keep bonded cells more rigidly aligned to each other");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.orientation_spring_stiffness, 0.1..=100.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.orientation_spring_stiffness).speed(0.1).range(0.1..=100.0));
                });

                ui.label("Orientation Spring Damping:")
                    .on_hover_text("Reduces rotational oscillation. Higher values prevent cells from wobbling around their bond axis");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.orientation_spring_damping, 0.0..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.orientation_spring_damping).speed(0.01).range(0.0..=10.0));
                });

                ui.label("Max Angular Deviation:")
                    .on_hover_text("Maximum angle (in degrees) the bond can bend before the orientation spring starts pushing back. 0° = perfectly rigid, 180° = fully flexible");
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
                ui.checkbox(&mut mode.adhesion_settings.enable_twist_constraint, "Enable Twist Constraint")
                    .on_hover_text("Prevents bonded cells from spinning freely around the bond axis. Useful for maintaining body orientation in structured organisms");

                ui.label("Twist Constraint Stiffness:")
                    .on_hover_text("How strongly the constraint resists axial rotation. Higher values lock the twist angle more firmly");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.twist_constraint_stiffness, 0.0..=20.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.twist_constraint_stiffness).speed(0.1).range(0.0..=20.0));
                });

                ui.label("Twist Constraint Damping:")
                    .on_hover_text("Reduces rotational oscillation around the bond axis. Higher values prevent spinning after a disturbance");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.adhesion_settings.twist_constraint_damping, 0.0..=50.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.adhesion_settings.twist_constraint_damping).speed(0.1).range(0.0..=50.0));
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
    context
        .editor_state
        .panel_rects
        .insert("ParentSettings".to_string(), ui.max_rect());

    // Multi-select: snapshot the primary mode before rendering so we can diff afterwards
    let selected_idx = context.editor_state.selected_mode_index;
    // Ensure selected_mode_indices is consistent (handles Default-constructed state)
    if context.editor_state.selected_mode_indices.is_empty() {
        context.editor_state.selected_mode_indices = vec![selected_idx];
    }
    let pre_snapshot: Option<crate::genome::ModeSettings> =
        if selected_idx < context.genome.modes.len() {
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
                    ui.label("Nutrient Generation Rate:")
                        .on_hover_text("Nutrients produced per second by this cell. Test cells generate nutrients directly without needing food sources");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.nutrient_gain_rate, 0.0..=20.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.nutrient_gain_rate).speed(0.01).range(0.0..=20.0).suffix("/s"));
                    });
                });
            } else if mode.cell_type == 3 { // Photocyte (cell_type == 3)
                group_container(ui, "Photocyte Functions", egui::Color32::from_rgb(255, 200, 60), |ui| {
                    ui.label("Converts light into nutrients. Can emit a signal when light level crosses a threshold.");
                    ui.label("Use the right-click menu to send a test signal in the preview.");
                    ui.separator();

                    ui.checkbox(&mut mode.photocyte_emit_enabled, "Emit Signal")
                        .on_hover_text("When enabled, emits a signal on the configured channel when the light condition is met");

                    if mode.photocyte_emit_enabled {
                        ui.label("Signal Channel:")
                            .on_hover_text("Channel (0-15) to emit on when the light threshold condition is met");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.photocyte_emit_channel, 0..=15).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.photocyte_emit_channel).speed(0.1).range(0..=15));
                        });

                        ui.label("Signal Value:")
                            .on_hover_text("Value emitted on the channel when the condition is met");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.photocyte_emit_value, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.photocyte_emit_value).speed(0.1).range(-100.0..=100.0));
                        });

                        ui.label("Signal Hops:")
                            .on_hover_text("How many adhesion hops the signal propagates");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.photocyte_emit_hops, 1..=20).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.photocyte_emit_hops).speed(0.1).range(1..=20));
                        });

                        ui.separator();
                        let mode_names = ["Above", "Below"];
                        egui::ComboBox::from_id_salt("photocyte_emit_mode")
                            .selected_text(mode_names[mode.photocyte_emit_mode.clamp(0, 1) as usize])
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut mode.photocyte_emit_mode, 0, "Above");
                                ui.selectable_value(&mut mode.photocyte_emit_mode, 1, "Below");
                            });
                        ui.label("Threshold:")
                            .on_hover_text("Light level (0.0-1.0) that triggers emission");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.photocyte_emit_threshold, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.photocyte_emit_threshold).speed(0.01).range(0.0..=1.0));
                        });
                    }
                });
            } else if mode.cell_type == 16 { // Luminocyte (cell_type == 16)
                group_container(ui, "Luminocyte Functions", egui::Color32::from_rgb(80, 220, 240), |ui| {
                    ui.label("Emits local light into the light field. Signal controls switch between dim and bright emission.");
                    ui.separator();

                    ui.label("Signal Channel:")
                        .on_hover_text("Channel (0-7) read by this luminocyte.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        mode.luminocyte_signal_channel = mode.luminocyte_signal_channel.clamp(0, 7);
                        ui.add(egui::Slider::new(&mut mode.luminocyte_signal_channel, 0..=7).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.luminocyte_signal_channel).speed(0.1).range(0..=7));
                    });

                    ui.label("Threshold:")
                        .on_hover_text("Signal value required to switch state.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.luminocyte_threshold, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.luminocyte_threshold).speed(0.5).range(0.0..=2047.0));
                    });

                    ui.checkbox(&mut mode.luminocyte_invert, "On without signal")
                        .on_hover_text("When enabled, the luminocyte is bright by default and dims when it receives a signal above threshold. When disabled, it is dim by default and brightens on signal.");

                    ui.label("Glow:")
                        .on_hover_text("Maximum emitted light intensity when the signal is above threshold. The visible cell glow uses the same value.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.emissive, 0.0..=8.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.emissive).speed(0.05).range(0.0..=8.0));
                    });
                });
            } else if mode.cell_type == 17 { // Siphonocyte (cell_type == 17)
                group_container(ui, "Siphonocyte Functions", egui::Color32::from_rgb(90, 190, 230), |ui| {
                    ui.label("Intake Rate:")
                        .on_hover_text("Reads the occupied voxel and fills internal reserve. This never removes volume, changes phase, or writes back into voxels.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.siphon_intake_rate, 0.0..=4.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.siphon_intake_rate).speed(0.01).range(0.0..=4.0));
                    });

                    ui.label("Expel Rate:")
                        .on_hover_text("Internal reserve spent per second while expelling. Expulsion does not create water, steam, spray, droplets, or voxel phase.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.siphon_expel_rate, 0.0..=4.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.siphon_expel_rate).speed(0.01).range(0.0..=4.0));
                    });

                    ui.label("Impulse:")
                        .on_hover_text("Directional impulse applied to the cell body when expelling.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.siphon_impulse, 0.0..=3.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.siphon_impulse).speed(0.01).range(0.0..=3.0));
                    });

                    let mode_names = [
                        "Impulse",
                        "Signal Impulse",
                        "Signal Intake",
                        "Signal Expulsion",
                    ];
                    egui::ComboBox::from_id_salt("siphon_mode")
                        .selected_text(mode_names[mode.siphon_mode.clamp(0, 3) as usize])
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut mode.siphon_mode, 0, "Impulse");
                            ui.selectable_value(&mut mode.siphon_mode, 1, "Signal Impulse");
                            ui.selectable_value(&mut mode.siphon_mode, 2, "Signal Intake");
                            ui.selectable_value(&mut mode.siphon_mode, 3, "Signal Expulsion");
                        });

                    if mode.siphon_mode >= 1 {
                        ui.label("Signal Channel:")
                            .on_hover_text("Channel used by the selected signal-gated siphon behavior.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            mode.siphon_signal_channel = mode.siphon_signal_channel.clamp(0, 15);
                            ui.add(egui::Slider::new(&mut mode.siphon_signal_channel, 0..=15).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.siphon_signal_channel).speed(0.1).range(0..=15));
                        });

                        ui.label("Signal Threshold:")
                            .on_hover_text("Signal value required by signal-gated siphon modes.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.siphon_signal_threshold, 0.0..=2047.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.siphon_signal_threshold).speed(1.0).range(0.0..=2047.0));
                        });

                        ui.checkbox(&mut mode.siphon_signal_invert, "Invert")
                            .on_hover_text("When checked, the signal-gated siphon behavior is active below the threshold instead of above it.");
                    }
                });
            } else if mode.cell_type == 18 { // Plumocyte (cell_type == 18)
                group_container(ui, "Plumocyte Functions", egui::Color32::from_rgb(150, 210, 190), |ui| {
                    ui.label("Extension:")
                        .on_hover_text("How extended the passive tendrils are. Higher values increase drag and exposure.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.plumocyte_extension, 0.0..=1.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.plumocyte_extension).speed(0.01).range(0.0..=1.0));
                    });

                    ui.label("Drag:")
                        .on_hover_text("Passive fall-slowing multiplier. It resists downward motion along gravity without braking organism-initiated forward movement.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.plumocyte_drag_mult, 0.0..=3.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.plumocyte_drag_mult).speed(0.01).range(0.0..=3.0));
                    });

                    ui.label("Flow Coupling:")
                        .on_hover_text("Passive coupling strength to surrounding medium. Current v1 strengthens fall-slowing in water without adding thrust.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.plumocyte_flow_coupling, 0.0..=3.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.plumocyte_flow_coupling).speed(0.01).range(0.0..=3.0));
                    });

                    ui.label("Exposure:")
                        .on_hover_text("Weak multiplier for environmental heat exchange and hot/dry water loss while extended.");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.plumocyte_exposure_mult, 0.0..=2.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.plumocyte_exposure_mult).speed(0.01).range(0.0..=2.0));
                    });
                });
            } else if mode.cell_type == 19 { // Stemocyte (cell_type == 19)
                group_container(ui, "Stemocyte Development", egui::Color32::from_rgb(170, 120, 220), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Channel")
                            .on_hover_text("Developmental gradient channel (8-15).");
                        mode.stemocyte_signal_channel = mode.stemocyte_signal_channel.clamp(8, 15);
                        ui.add(
                            egui::DragValue::new(&mut mode.stemocyte_signal_channel)
                                .speed(0.1)
                                .range(8..=15),
                        );
                        ui.separator();
                        ui.small("Drag dividers · click a response");
                    });

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label("Delay").on_hover_text(
                            "None: immediate. Cycles: wait for inherited divisions. Time: inherited developmental seconds. Signal hold: require uninterrupted signal. Threshold: require minimum strength.",
                        );
                        mode.stemocyte_delay_mode = mode.stemocyte_delay_mode.clamp(0, 4);
                        let delay_label = match mode.stemocyte_delay_mode {
                            1 => "Cycles",
                            2 => "Time",
                            3 => "Signal hold",
                            4 => "Threshold",
                            _ => "None",
                        };
                        egui::ComboBox::from_id_salt("stemocyte_response_delay")
                            .selected_text(delay_label)
                            .width(78.0)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut mode.stemocyte_delay_mode, 0, "None");
                                ui.selectable_value(&mut mode.stemocyte_delay_mode, 1, "Cycles");
                                ui.selectable_value(&mut mode.stemocyte_delay_mode, 2, "Time");
                                ui.selectable_value(&mut mode.stemocyte_delay_mode, 3, "Signal hold");
                                ui.selectable_value(&mut mode.stemocyte_delay_mode, 4, "Threshold");
                            });

                        match mode.stemocyte_delay_mode {
                            1 => {
                                mode.stemocyte_delay_value =
                                    mode.stemocyte_delay_value.round().clamp(0.0, 20.0);
                                ui.add(
                                    egui::DragValue::new(&mut mode.stemocyte_delay_value)
                                        .range(0.0..=20.0)
                                        .suffix(" cyc"),
                                );
                            }
                            2 | 3 => {
                                mode.stemocyte_delay_value =
                                    mode.stemocyte_delay_value.clamp(0.0, 120.0);
                                ui.add(
                                    egui::DragValue::new(&mut mode.stemocyte_delay_value)
                                        .speed(0.1)
                                        .range(0.0..=120.0)
                                        .suffix(" s"),
                                );
                            }
                            4 => {
                                mode.stemocyte_delay_value =
                                    mode.stemocyte_delay_value.clamp(0.0, 2047.0);
                                ui.add(
                                    egui::DragValue::new(&mut mode.stemocyte_delay_value)
                                        .speed(1.0)
                                        .range(0.0..=2047.0),
                                );
                            }
                            _ => {
                                ui.small("immediate");
                            }
                        }
                    });

                    draw_stemocyte_response_strip(
                        ui,
                        mode.stemocyte_weak_first,
                        &mut mode.stemocyte_outcomes,
                        &mut mode.stemocyte_thresholds,
                        &mode_info_for_dropdowns,
                    );
                });
            } else if mode.cell_type == 4 { // Lipocyte (cell_type == 4)
                group_container(ui, "Lipocyte Functions", egui::Color32::from_rgb(220, 180, 100), |ui| {
                    ui.label("Stores surplus nutrients as fat reserves. Can emit a signal based on storage level.");
                    ui.label("Use the right-click menu to send a test signal in the preview.");
                    ui.separator();

                    ui.checkbox(&mut mode.lipocyte_emit_enabled, "Emit Signal")
                        .on_hover_text("When enabled, emits a signal on the configured channel when the storage condition is met");

                    if mode.lipocyte_emit_enabled {
                        ui.label("Signal Channel:")
                            .on_hover_text("Channel (0-15) to emit on when the storage threshold condition is met");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.lipocyte_emit_channel, 0..=15).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.lipocyte_emit_channel).speed(0.1).range(0..=15));
                        });

                        ui.label("Signal Value:")
                            .on_hover_text("Value emitted on the channel when the condition is met");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.lipocyte_emit_value, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.lipocyte_emit_value).speed(0.1).range(-100.0..=100.0));
                        });

                        ui.label("Signal Hops:")
                            .on_hover_text("How many adhesion hops the signal propagates");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.lipocyte_emit_hops, 1..=20).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.lipocyte_emit_hops).speed(0.1).range(1..=20));
                        });

                        ui.separator();
                        let mode_names = ["Above", "Below"];
                        egui::ComboBox::from_id_salt("lipocyte_emit_mode")
                            .selected_text(mode_names[mode.lipocyte_emit_mode.clamp(0, 1) as usize])
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut mode.lipocyte_emit_mode, 0, "Above");
                                ui.selectable_value(&mut mode.lipocyte_emit_mode, 1, "Below");
                            });
                        ui.label("Threshold:")
                            .on_hover_text("Storage fraction (0.0-1.0) that triggers emission. 1.0 = completely full");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.lipocyte_emit_threshold, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.lipocyte_emit_threshold).speed(0.01).range(0.0..=1.0));
                        });
                    }
                });
            } else if mode.cell_type == 6 { // Glueocyte (cell_type == 6)
                group_container(ui, "Glueocyte Functions", egui::Color32::from_rgb(140, 200, 140), |ui| {
                    ui.checkbox(&mut mode.glueocyte_cell_adhesion, "Cell Adhesion")
                        .on_hover_text("Form adhesion bonds with other cells on contact");
                    ui.checkbox(&mut mode.glueocyte_self_adhesion, "Bond to Own Organism")
                        .on_hover_text("When enabled, preview glueocytes act as disposable applicators: after touching two own-organism cells they create a black mechanical ball joint between them and are consumed. Two touching glueocytes merge into one larger glueocyte.");
                    ui.checkbox(&mut mode.glueocyte_env_adhesion, "Environment Adhesion")
                        .on_hover_text("Bond to cave walls and the world boundary on contact");
                    ui.checkbox(&mut mode.glueocyte_boulder_adhesion, "Boulder/Mossrock Adhesion")
                        .on_hover_text("Bond to floating mossrock boulders on contact");

                    // Signal gate - shown whenever any adhesion type is enabled.
                    // Gates all adhesion types simultaneously.
                    let any_adhesion = mode.glueocyte_cell_adhesion
                        || mode.glueocyte_env_adhesion
                        || mode.glueocyte_boulder_adhesion;
                    if any_adhesion {
                        ui.separator();
                        let channel_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7"];
                        let has_gate = mode.glueocyte_cell_adhesion_signal_channel >= 0
                            && mode.glueocyte_cell_adhesion_signal_channel <= 7;
                        ui.horizontal(|ui| {
                            ui.label("Signal Gate:")
                                .on_hover_text("Always On: bonds whenever touching a surface. Signal: only bonds when the chosen sensory channel (0–7) is above the threshold. Applies to all adhesion types");
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
                                .logarithmic(false))
                                .on_hover_text("Signal strength threshold");
                            ui.horizontal(|ui| {
                                ui.label("Disconnect when:");
                                if ui.selectable_label(!mode.glueocyte_signal_gate_invert, "No signal").clicked() {
                                    mode.glueocyte_signal_gate_invert = false;
                                }
                                if ui.selectable_label(mode.glueocyte_signal_gate_invert, "Signal").clicked() {
                                    mode.glueocyte_signal_gate_invert = true;
                                }
                            });
                        }
                    }
                });
            } else if mode.cell_type == 1 { // Flagellocyte (cell_type == 1)
                group_container(ui, "Flagellocyte Functions", egui::Color32::from_rgb(140, 180, 220), |ui| {
                    // Mode toggle
                    ui.horizontal(|ui| {
                        ui.label("Speed Mode:")
                            .on_hover_text("Fixed: constant thrust force. Signal: switches between two thrust levels based on a sensory channel reading (0–7)");
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
                        ui.label("Swim Force:")
                            .on_hover_text("Thrust force applied in the cell's forward direction each frame. Higher values = faster swimming. Consumes nutrients proportional to force (5 nutrients/sec at force 1.0)");
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
                                ui.label("Channel:")
                                    .on_hover_text("Sensory channel (0–7) to read. Speed A is used when the channel is below Threshold C; Speed B is used when at or above it");
                                egui::ComboBox::from_id_salt("flag_signal_channel")
                                    .selected_text(flag_ch_labels[flag_ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in flag_ch_labels.iter().enumerate() {
                                            ui.selectable_value(&mut mode.flagellocyte_signal_channel, i as i32, *label);
                                        }
                                    });
                            });
                        }

                        ui.label("Speed A (signal < C):")
                            .on_hover_text("Swim force when the signal channel is below Threshold C. Use 0 to stop swimming when no target is detected");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.flagellocyte_speed_a, 0.0..=3.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.flagellocyte_speed_a).speed(0.01).range(0.0..=3.0));
                        });

                        ui.label("Speed B (signal >= C):")
                            .on_hover_text("Swim force when the signal channel is at or above Threshold C. Use a higher value to accelerate toward a detected target");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.flagellocyte_speed_b, 0.0..=3.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.flagellocyte_speed_b).speed(0.01).range(0.0..=3.0));
                        });

                        ui.label("Threshold C:")
                            .on_hover_text("Signal strength at which the flagellocyte switches from Speed A to Speed B");
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
                    ui.label("Buoyancy Force:")
                        .on_hover_text("Upward force applied to this cell each frame. Positive values push the cell upward (away from gravity). Use to make cells float or rise through water");
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
                    // Sense Type checkboxes (bitmask: multiple types can be active simultaneously)
                    ui.label("Sense Type:")
                        .on_hover_text("What this oculocyte detects with its ray. Multiple types can be active at once. 'Self' fires unconditionally every frame regardless of what is in front");
                    let sense_bits: &[(&str, u32, &str)] = &[
                        ("Cell",      1 << 0, "Detect other cells in the ray's path"),
                        ("Food",      1 << 1, "Detect nutrient voxels in the fluid"),
                        ("Light",     1 << 2, "Detect light intensity along the ray"),
                        ("Wall/Cave", 1 << 3, "Detect cave walls and solid surfaces"),
                        ("Self",      1 << 4, "Always fires — unconditional signal emitter, useful for positional gradients"),
                        ("Mossrock",  1 << 5, "Detect floating mossrock boulders"),
                    ];
                    ui.horizontal_wrapped(|ui| {
                        for (label, bit, tip) in sense_bits {
                            let mut checked = (mode.oculocyte_sense_type & bit) != 0;
                            if ui.checkbox(&mut checked, *label).on_hover_text(*tip).changed() {
                                if checked {
                                    mode.oculocyte_sense_type |= bit;
                                } else {
                                    mode.oculocyte_sense_type &= !bit;
                                }
                            }
                        }
                    });

                    // Signal Channel (sensory: 0-7)
                    ui.label("Signal Channel:")
                        .on_hover_text("Which sensory channel (0–7) to emit on when the ray detects its target. Sensory channels drive behavioural responses (locomotion speed, contraction) — use developmental channels 8–15 to gate division or mode switching");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_signal_channel, 0..=7).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_signal_channel).speed(0.1).range(0..=7));
                    });

                    // Signal Value
                    ui.label("Signal Value:")
                        .on_hover_text("Strength of the signal emitted when the ray detects its target. The signal attenuates by half at each adhesion hop away from this cell");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_signal_value, 1.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_signal_value).speed(1.0).range(1.0..=2047.0));
                    });

                    // Signal Hops
                    ui.label("Signal Hops:")
                        .on_hover_text("How many adhesion bonds the signal travels through. Signal strength halves at each hop beyond the first. 3 hops = 25% strength at the 3rd cell away");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_signal_hops, 1..=20).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_signal_hops).speed(0.1).range(1..=20));
                    });

                    // Ray Length
                    ui.label("Ray Length:")
                        .on_hover_text("Maximum distance the detection ray travels from this cell. Longer rays detect targets further away but may be more expensive");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.oculocyte_ray_length, 1.0..=100.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.oculocyte_ray_length).speed(0.1).range(1.0..=100.0));
                    });

                    if (mode.oculocyte_sense_type & (1 << 2)) != 0 {
                        ui.separator();
                        ui.label("Light Color Filter:")
                            .on_hover_text("Light detection only fires when the ray hits light close to this RGB color. The default is a narrow warm sun band");
                        ui.horizontal(|ui| {
                            ui.label("R");
                            ui.add(egui::Slider::new(&mut mode.oculocyte_light_target_color.x, 0.0..=2.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.oculocyte_light_target_color.x).speed(0.01).range(0.0..=2.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("G");
                            ui.add(egui::Slider::new(&mut mode.oculocyte_light_target_color.y, 0.0..=2.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.oculocyte_light_target_color.y).speed(0.01).range(0.0..=2.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("B");
                            ui.add(egui::Slider::new(&mut mode.oculocyte_light_target_color.z, 0.0..=2.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.oculocyte_light_target_color.z).speed(0.01).range(0.0..=2.0));
                        });
                        ui.label("Tolerance:")
                            .on_hover_text("Maximum RGB distance from the target color. Lower values make the detectable color range narrower");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.oculocyte_light_color_tolerance, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.oculocyte_light_color_tolerance).speed(0.01).range(0.0..=1.0));
                        });
                    }
                });
            } else if mode.cell_type == 8 { // Ciliocyte (cell_type == 8)
                group_container(ui, "Ciliocyte Functions", egui::Color32::from_rgb(160, 200, 180), |ui| {
                    // Mode toggle
                    ui.horizontal(|ui| {
                        ui.label("Speed Mode:")
                            .on_hover_text("Fixed: cilia beat at a constant speed. Signal: speed switches between two values based on a sensory channel reading (0–7)");
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
                        ui.label("Cilia Speed:")
                            .on_hover_text("Beat speed and direction. Positive values push fluid/particles in the cell's forward direction; negative values push backward. 0 = no movement");
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
                                ui.label("Channel:")
                                    .on_hover_text("Sensory channel (0–7) to read. The cilia speed switches between Speed Below and Speed Above based on whether this channel exceeds the threshold");
                                egui::ComboBox::from_id_salt("cilia_signal_channel")
                                    .selected_text(cilia_ch_labels[cilia_ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in cilia_ch_labels.iter().enumerate() {
                                            ui.selectable_value(&mut mode.cilia_signal_channel, i as i32, *label);
                                        }
                                    });
                            });
                        }

                        ui.label("Speed Below (signal < threshold):")
                            .on_hover_text("Cilia speed when the signal channel is below the threshold. Negative = reverse direction");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_speed_below, -1.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_speed_below).speed(0.01).range(-1.0..=1.0));
                        });

                        ui.label("Speed Above (signal >= threshold):")
                            .on_hover_text("Cilia speed when the signal channel is at or above the threshold");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_speed_above, -1.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_speed_above).speed(0.01).range(-1.0..=1.0));
                        });

                        ui.label("Threshold:")
                            .on_hover_text("Signal strength at which the cilia switch from Speed Below to Speed Above");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cilia_threshold, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cilia_threshold).speed(0.1).range(-100.0..=100.0));
                        });
                    }

                    ui.add_space(4.0);
                    ui.checkbox(&mut mode.cilia_push_bonded, "Push Organism Cells")
                        .on_hover_text("When enabled, cilia force is applied to bonded cells in the same organism (useful for internal pumping). When disabled, only pushes unattached particles and fluid");

                    ui.add_space(4.0);
                    ui.label("Attract Force:")
                        .on_hover_text("Pulls nearby unattached cells and particles toward this cell. 0 = no attraction. Useful for filter-feeding structures that funnel food inward");
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
                        ui.label("Contraction Mode:")
                            .on_hover_text("Pulse: contracts on a timer, alternating between Pulse A and Pulse B phases. Signal: contraction strength is controlled by a signal channel (0–15)");
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
                        ui.label("Pulse Phase:")
                            .on_hover_text("Pulse A and Pulse B are offset by half a cycle. Assign opposite ring positions to different phases to create a traveling wave for locomotion");
                        ui.horizontal(|ui| {
                            if ui.selectable_label(mode.myocyte_pulse_phase == 0, "Pulse A").clicked() {
                                mode.myocyte_pulse_phase = 0;
                            }
                            if ui.selectable_label(mode.myocyte_pulse_phase == 1, "Pulse B").clicked() {
                                mode.myocyte_pulse_phase = 1;
                            }
                        });

                        ui.label("Pulse Rate (cycles/sec):")
                            .on_hover_text("How many full contraction-relaxation cycles per second. Higher rates create faster but potentially less powerful strokes. Must match between paired myocytes for coordinated locomotion");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_pulse_rate, 0.1..=10.0).show_value(false).logarithmic(true));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_pulse_rate).speed(0.01).range(0.1..=10.0));
                        });

                        ui.label("Contraction:")
                            .on_hover_text("How much the cell shortens its adhesion bonds during the active phase. 0 = no contraction, 1 = maximum shortening");
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
                                ui.label("Channel:")
                                    .on_hover_text("Signal channel to read. Channels 0–7 are sensory channels (detection-driven behaviour); 8–15 are developmental/regulation channels");
                                egui::ComboBox::from_id_salt("myocyte_signal_channel")
                                    .selected_text(myo_ch_labels[myo_ch_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in myo_ch_labels.iter().enumerate() {
                                            ui.selectable_value(&mut mode.myocyte_signal_channel, i as i32, *label);
                                        }
                                    });
                            });
                        }

                        ui.label("Contraction Below (signal < threshold):")
                            .on_hover_text("How much the cell shortens its adhesion bonds when the signal is below the threshold. 0 = no contraction, 1 = maximum shortening");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_contraction_below, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_contraction_below).speed(0.01).range(0.0..=1.0));
                        });

                        ui.label("Contraction Above (signal >= threshold):")
                            .on_hover_text("How much the cell shortens its adhesion bonds when the signal is at or above the threshold. Pair opposite ring positions with different above/below values to create locomotion");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_contraction_above, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_contraction_above).speed(0.01).range(0.0..=1.0));
                        });

                        ui.label("Threshold:")
                            .on_hover_text("Signal strength at which the myocyte switches between Contraction Below and Contraction Above");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_threshold, -100.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_threshold).speed(0.1).range(-100.0..=100.0));
                        });
                    }

                    // Peristaltic grip — available in both pulse-timer and signal modes
                    ui.separator();
                    let grip_enabled = mode.myocyte_grip_contracted > 0.0 || mode.myocyte_grip_extended > 0.0;
                    let mut grip_on = grip_enabled;
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut grip_on, "Peristaltic Grip")
                            .on_hover_text("Friction against the medium tied to the contraction cycle. Set Grip on Contract > Grip on Extend to push forward; reverse values to go backward. Full strength in water, 5% in air.");
                    });
                    if !grip_on && grip_enabled {
                        mode.myocyte_grip_contracted = 0.0;
                        mode.myocyte_grip_extended = 0.0;
                    }
                    if grip_on {
                        // Ensure at least one value is non-zero when first enabled
                        if !grip_enabled {
                            mode.myocyte_grip_contracted = 10.0;
                        }
                        ui.label("Grip on Contract:")
                            .on_hover_text("Friction drag when fully contracted (~10–20 = strong). Higher than Grip on Extend = pushes forward on the power stroke.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_grip_contracted, 0.0..=30.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_grip_contracted).speed(0.1).range(0.0..=30.0));
                        });
                        ui.label("Grip on Extend:")
                            .on_hover_text("Friction drag when fully extended. Keep near 0 for efficient forward motion; set higher than Grip on Contract to reverse direction.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.myocyte_grip_extended, 0.0..=30.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.myocyte_grip_extended).speed(0.1).range(0.0..=30.0));
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
                        ui.checkbox(&mut mode.embryocyte_use_timer, "Timer")
                            .on_hover_text("Release after the egg has been attached to the parent for at least this many seconds. Ensures minimum incubation time");
                    });
                    if mode.embryocyte_use_timer {
                        ui.label("Release after (seconds attached):")
                            .on_hover_text("Minimum time in seconds the egg must remain attached before it can be released");
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
                        ui.checkbox(&mut mode.embryocyte_use_threshold, "Reserve Threshold")
                            .on_hover_text("Release only when the egg's reserve has been filled to at least this level. Max reserve is 65535. At 62000 (~95%) the egg has ~6.2 seconds of free-floating life");
                    });
                    if mode.embryocyte_use_threshold {
                        ui.label("Release when reserve >=:")
                            .on_hover_text("Reserve level required before release. The egg burns reserve at 10 units/sec when free. 62000 ≈ 6.2 seconds of life after detachment");
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
                        ui.checkbox(&mut mode.embryocyte_use_signal, "Signal")
                            .on_hover_text("Release only when the parent organism is sending a specific signal. Use a maturation signal (ch_mat) to ensure the parent is fully grown before releasing eggs");
                    });
                    if mode.embryocyte_use_signal {
                        let emb_ch_labels = ["Ch 0", "Ch 1", "Ch 2", "Ch 3", "Ch 4", "Ch 5", "Ch 6", "Ch 7",
                                             "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                        let emb_ch_idx = (mode.embryocyte_signal_channel as usize).min(15);
                        ui.horizontal(|ui| {
                            ui.label("Channel:")
                                .on_hover_text("Signal channel to monitor. Channels 0–7 are sensory channels; 8–15 are developmental/regulation channels. Use a regulation channel for maturation/feeding gates");
                            egui::ComboBox::from_id_salt("embryocyte_signal_channel")
                                .selected_text(emb_ch_labels[emb_ch_idx])
                                .show_ui(ui, |ui| {
                                    for (i, label) in emb_ch_labels.iter().enumerate() {
                                        ui.selectable_value(&mut mode.embryocyte_signal_channel, i as i32, *label);
                                    }
                                });
                        });

                        ui.label("Release when signal >=:")
                            .on_hover_text("Minimum signal strength required on the chosen channel before the egg will release");
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
                        ui.colored_label(palette().status_warn, "⚠ No triggers enabled — cell will never release.");
                    }
                });
            } else if mode.cell_type == 13 { // Gametocyte (cell_type == 13)
                group_container(ui, "Gametocyte Functions", egui::Color32::from_rgb(200, 120, 200), |ui| {
                    ui.label("Accumulates reserve while attached. When release triggers fire, it detaches and seeks a compatible partner. On contact, both gametes die and their combined reserve seeds a new Embryocyte with a crossover genome.")
                        .on_hover_text("Gametocytes never split. Split mass, split interval, and max splits do not make this cell type divide.");
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Merge Range:");
                        ui.add(egui::Slider::new(&mut mode.gametocyte_merge_range, 0.0..=2.0)
                            .text("u")
                            .step_by(0.05))
                            .on_hover_text("Extra contact distance beyond cell radii that triggers a merge (0 = must physically touch).");
                    });
                    ui.colored_label(egui::Color32::from_rgb(180, 140, 180),
                        "  ▸ Only merges with genomes of similar cell-type structure.");
                    ui.separator();

                    // Release triggers - identical to Embryocyte
                    ui.label("Release triggers (detach from organism when ALL enabled are satisfied):");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.embryocyte_use_timer, "Timer")
                            .on_hover_text("Release after being attached this many seconds.");
                    });
                    if mode.embryocyte_use_timer {
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            ui.style_mut().spacing.slider_width = (available - 70.0).max(50.0);
                            ui.add(egui::Slider::new(&mut mode.embryocyte_release_timer, 0.1..=300.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.embryocyte_release_timer).speed(0.1).range(0.1..=300.0));
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.embryocyte_use_threshold, "Reserve Threshold")
                            .on_hover_text("Release only once reserve reaches this level. Max is 65535.");
                    });
                    if mode.embryocyte_use_threshold {
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            ui.style_mut().spacing.slider_width = (available - 70.0).max(50.0);
                            let mut threshold_f32 = mode.embryocyte_threshold_value as f32;
                            ui.add(egui::Slider::new(&mut threshold_f32, 0.0_f32..=65535.0_f32).show_value(false));
                            ui.add(egui::DragValue::new(&mut threshold_f32).speed(10.0).range(0.0_f32..=65535.0_f32));
                            mode.embryocyte_threshold_value = threshold_f32 as u32;
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.embryocyte_use_signal, "Signal")
                            .on_hover_text("Release when the parent sends a specific signal.");
                    });
                    if mode.embryocyte_use_signal {
                        let ch_labels = ["Ch 0","Ch 1","Ch 2","Ch 3","Ch 4","Ch 5","Ch 6","Ch 7",
                                         "Ch 8","Ch 9","Ch 10","Ch 11","Ch 12","Ch 13","Ch 14","Ch 15"];
                        let ch_idx = (mode.embryocyte_signal_channel as usize).min(15);
                        ui.horizontal(|ui| {
                            ui.label("Channel:");
                            egui::ComboBox::from_id_salt("gametocyte_signal_channel")
                                .selected_text(ch_labels[ch_idx])
                                .show_ui(ui, |ui| {
                                    for (i, label) in ch_labels.iter().enumerate() {
                                        ui.selectable_value(&mut mode.embryocyte_signal_channel, i as i32, *label);
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            ui.style_mut().spacing.slider_width = (available - 70.0).max(50.0);
                            ui.add(egui::Slider::new(&mut mode.embryocyte_signal_value, 0.0..=2047.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.embryocyte_signal_value).speed(1.0).range(0.0..=2047.0));
                        });
                    }
                    if !mode.embryocyte_use_timer && !mode.embryocyte_use_threshold && !mode.embryocyte_use_signal {
                        ui.separator();
                        ui.colored_label(palette().status_warn, "⚠ No triggers — gamete will never detach.");
                    }
                });
            } else if mode.cell_type == 12 { // Vasculocyte (cell_type == 12)
                group_container(ui, "Vasculocyte Functions", egui::Color32::from_rgb(60, 160, 200), |ui| {
                    ui.label("Forms high-throughput conduits through the organism. Can carry nutrients, signals, or both.");
                    ui.label("Physical compression (e.g. from Myocytes) boosts transport rate.");
                    ui.separator();

                    ui.label("Nutrients:");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.vascular_nutrient_transport, "Transport")
                            .on_hover_text("Carries nutrients efficiently between connected vasculocytes. Disable this to make the mode a nutrient barrier unless exchange is enabled.");
                        ui.checkbox(&mut mode.vascular_outlet, "Exchange Port")
                            .on_hover_text("Bidirectional inlet/outlet: can receive nutrients from adjacent non-vascular tissue and release nutrients back into it.");
                    });

                    ui.label("Signals:");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut mode.vascular_signal_transport, "Transport")
                            .on_hover_text("Carries signals efficiently between connected vasculocytes. A vascular-to-vascular step costs 0.25 travel points instead of 1.0.");
                        ui.checkbox(&mut mode.vascular_signal_exchange, "Exchange Port")
                            .on_hover_text("Bidirectional inlet/outlet: can receive signals from adjacent non-vascular tissue and release signals back into it.");
                    });

                    let nutrient_label = match (mode.vascular_nutrient_transport, mode.vascular_outlet) {
                        (true, true) => "Nutrients: pipe with tissue exchange.",
                        (true, false) => "Nutrients: sealed pipe.",
                        (false, true) => "Nutrients: local exchange port only.",
                        (false, false) => "Nutrients: closed.",
                    };
                    let signal_label = match (mode.vascular_signal_transport, mode.vascular_signal_exchange) {
                        (true, true) => "Signals: road with tissue exchange.",
                        (true, false) => "Signals: sealed road.",
                        (false, true) => "Signals: local exchange port only.",
                        (false, false) => "Signals: closed.",
                    };
                    ui.colored_label(egui::Color32::from_rgb(120, 220, 255), nutrient_label);
                    ui.colored_label(egui::Color32::from_rgb(120, 255, 180), signal_label);
                    // Signal capacity (only relevant when signal transport or exchange is on)
                    if mode.vascular_signal_transport || mode.vascular_signal_exchange {
                        ui.add_space(4.0);
                        ui.separator();
                        ui.label("Signal Capacity:")
                            .on_hover_text("Maximum signal value this node forwards per tick. Caps output regardless of how many upstream paths converge here, preventing junction amplification. Set higher for a main trunk, lower for branches.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.vascular_signal_capacity, 0.1..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.vascular_signal_capacity).speed(0.1).range(0.1..=100.0));
                        });
                    }
                });
            } else if mode.cell_type == 11 { // Devorocyte (cell_type == 11)
                group_container(ui, "Devorocyte Functions", egui::Color32::from_rgb(200, 60, 60), |ui| {
                    ui.label("Steals nutrients from and kills foreign cells on contact.");
                    ui.label("Ignores cells of the same organism or genome.");
                    ui.separator();

                    ui.label("Contact Range:")
                        .on_hover_text("Extra reach beyond the cell's physical radius. A value of 0 means the devorocyte must physically overlap the target. Higher values let it steal nutrients from nearby cells without touching");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.devorocyte_consume_range, 0.0..=3.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.devorocyte_consume_range).speed(0.01).range(0.0..=3.0).suffix(" u"));
                    });

                    ui.label("Consume Rate:")
                        .on_hover_text("Nutrients stolen per second from each target cell in range. The target loses nutrients at this rate and dies when depleted");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.devorocyte_consume_rate, 0.0..=200.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.devorocyte_consume_rate).speed(1.0).range(0.0..=200.0).suffix("/s"));
                    });
                });
            } else if mode.cell_type == 14 { // Cognocyte (cell_type == 14)
                group_container(ui, "Cognocyte Functions", egui::Color32::from_rgb(80, 180, 210), |ui| {
                    ui.label("Reads signals from two input channels, applies an operation, and emits the result. Chain multiple Cognocytes to build logic circuits.");
                    ui.separator();

                    // Operation dropdown
                    let op_names = [
                        "Add", "Subtract", "Multiply", "Divide",
                        "Min", "Max", "Average",
                        "Greater Than", "Less Than", "Equal",
                        "AND", "OR", "NOT", "Select",
                        "Oscillate", "Hops Oscillate",
                    ];
                    let current_op = mode.cognocyte_operation.clamp(0, 15) as usize;
                    ui.label("Operation:")
                        .on_hover_text("Arithmetic: result = A op B.  Comparison: outputs 1.0 (true) or 0.0 (false).  Boolean: treats any value > 0 as true.  NOT uses only Input A.  Select: if A > 0 outputs B, else 0.  Oscillate: generates a half-rectified sine wave from time — no inputs needed.");
                    egui::ComboBox::from_id_salt("cognocyte_op")
                        .selected_text(op_names[current_op])
                        .show_ui(ui, |ui| {
                            for (i, name) in op_names.iter().enumerate() {
                                ui.selectable_value(&mut mode.cognocyte_operation, i as i32, *name);
                            }
                        });

                    ui.separator();

                    let is_oscillate = mode.cognocyte_operation == 14 || mode.cognocyte_operation == 15;

                    if is_oscillate {
                        // Oscillate mode: show rate and phase, hide input channels
                        ui.label("Rate (cycles/sec):")
                            .on_hover_text("How many full oscillation cycles per second. Two Cognocytes at the same rate but phase 0.5 apart give fully complementary signals for left/right gating.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cognocyte_oscillator_rate, 0.1..=10.0).show_value(false).logarithmic(true));
                            ui.add(egui::DragValue::new(&mut mode.cognocyte_oscillator_rate).speed(0.01).range(0.1..=10.0));
                        });
                        ui.label("Phase Offset (0–1):")
                            .on_hover_text("Fraction of a full cycle to offset this oscillator. 0.0 = in phase, 0.5 = opposite phase. Use 0.0 on one side and 0.5 on the other for left/right locomotion gating.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cognocyte_oscillator_phase, 0.0..=1.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cognocyte_oscillator_phase).speed(0.01).range(0.0..=1.0));
                        });
                        if mode.cognocyte_operation == 15 {
                            ui.label("Steps:")
                                .on_hover_text("Number of discrete steps per cycle (also the maximum hop reach). The signal propagates 1 hop on step 1, 2 hops on step 2, and so on. Cells at hop k only receive the signal during steps k–N of each cycle, creating a natural phase gradient along the chain.");
                            ui.horizontal(|ui| {
                                let available = ui.available_width();
                                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                                ui.style_mut().spacing.slider_width = slider_width;
                                ui.add(egui::Slider::new(&mut mode.cognocyte_oscillator_step_count, 1..=20).show_value(false));
                                ui.add(egui::DragValue::new(&mut mode.cognocyte_oscillator_step_count).speed(0.1).range(1..=20));
                            });
                        }
                        ui.label("Signal Strength:")
                            .on_hover_text("Peak value of the output signal. Downstream cells see this value at the top of each cycle and 0 at the trough.");
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cognocyte_oscillator_strength, 0.0..=100.0).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cognocyte_oscillator_strength).speed(0.1).range(0.0..=100.0));
                        });
                    } else {
                        let input_a_label = match mode.cognocyte_operation {
                            13 => "Condition Channel:",
                            10 | 11 | 12 => "Input A - Boolean Channel:",
                            7 | 8 | 9 => "Input A - Test Channel:",
                            _ => "Input A - Channel:",
                        };
                        let input_a_tooltip = match mode.cognocyte_operation {
                            13 => "Signal channel (0-15) used as the selector. If A > 0, the Cognocyte outputs B; otherwise it outputs 0. Missing A means no output",
                            10 | 11 | 12 => "Signal channel (0-15) read as a boolean. Values greater than 0 are true. Missing A means no output",
                            7 | 8 | 9 => "Signal channel (0-15) read as the value being compared. Missing A means no output",
                            _ => "Signal channel (0-15) read as the left operand. Missing A means no output",
                        };

                        // Input Channel A
                        ui.label(input_a_label)
                            .on_hover_text(input_a_tooltip);
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                            ui.style_mut().spacing.slider_width = slider_width;
                            ui.add(egui::Slider::new(&mut mode.cognocyte_input_channel_a, 0..=15).show_value(false));
                            ui.add(egui::DragValue::new(&mut mode.cognocyte_input_channel_a).speed(0.1).range(0..=15));
                        });

                        if mode.cognocyte_operation != 12 {
                            let input_b_label = match mode.cognocyte_operation {
                                13 => "Value Channel:",
                                10 | 11 => "Input B - Boolean Channel:",
                                7 | 8 | 9 => "Input B - Reference Channel:",
                                _ => "Input B - Channel:",
                            };
                            let input_b_tooltip = match mode.cognocyte_operation {
                                13 => "Signal channel (0-15) emitted when the condition channel is true. Missing B means no output",
                                10 | 11 => "Signal channel (0-15) read as a boolean. Values greater than 0 are true. Missing B means no output",
                                7 => "Signal channel (0-15) used as the comparison threshold. Outputs 1.0 when A > B, otherwise 0.0. Missing B means no output",
                                8 => "Signal channel (0-15) used as the comparison threshold. Outputs 1.0 when A < B, otherwise 0.0. Missing B means no output",
                                9 => "Signal channel (0-15) used as the equality reference. Outputs 1.0 when A and B are nearly equal, otherwise 0.0. Missing B means no output",
                                3 => "Signal channel (0-15) read as the divisor. Missing B means no output. If B is zero, Divide emits 0.0",
                                _ => "Signal channel (0-15) read as the right operand. Missing B means no output",
                            };

                            ui.label(input_b_label)
                                .on_hover_text(input_b_tooltip);
                            ui.horizontal(|ui| {
                                let available = ui.available_width();
                                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                                ui.style_mut().spacing.slider_width = slider_width;
                                ui.add(egui::Slider::new(&mut mode.cognocyte_input_channel_b, 0..=15).show_value(false));
                                ui.add(egui::DragValue::new(&mut mode.cognocyte_input_channel_b).speed(0.1).range(0..=15));
                            });
                        } else {
                            ui.label(egui::RichText::new("Input B unused for NOT").small());
                        }
                    }

                    ui.separator();

                    // Output Channel
                    ui.label("Output Channel:")
                        .on_hover_text("Signal channel (0–15) the result is emitted on. Channels 0–7 are sensory channels; 8–15 are developmental/regulation channels that can gate division, apoptosis, and mode switching");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.cognocyte_output_channel, 0..=15).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.cognocyte_output_channel).speed(0.1).range(0..=15));
                    });

                    // Output Hops
                    ui.label("Output Hops:")
                        .on_hover_text("How many adhesion bonds the result signal propagates. Signal halves in strength each hop beyond the first");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.cognocyte_output_hops, 1..=20).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.cognocyte_output_hops).speed(0.1).range(1..=20));
                    });
                });
            } else if mode.cell_type == 15 { // Memorocyte (cell_type == 15)
                group_container(ui, "Memorocyte Functions", egui::Color32::from_rgb(180, 140, 220), |ui| {
                    ui.label("Accumulates incoming signals over time and slowly forgets them. Useful for smoothing sensors, building timers, and adding hysteresis to decision circuits.");
                    ui.separator();

                    // Rate
                    ui.label("Rate:")
                        .on_hover_text("Fraction of the gap between memory and input closed per second. 0 = never tracks (holds state forever), 1 = instant snap (no memory). Output always converges toward input — never amplifies beyond it");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.memorocyte_rate, 0.0..=1.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.memorocyte_rate).speed(0.005).range(0.0..=1.0));
                    });

                    ui.separator();

                    // Input Channel
                    ui.label("Input Channel:")
                        .on_hover_text("Signal channel (0–15) to integrate. If no signal is present on this channel, the memory simply decays");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.memorocyte_input_channel, 0..=15).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.memorocyte_input_channel).speed(0.1).range(0..=15));
                    });

                    // Output Channel
                    ui.label("Output Channel:")
                        .on_hover_text("Signal channel (0–15) the memory value is emitted on every frame. Channels 0–7 reach sensory readers (locomotion, contraction); channels 8–15 can gate division, apoptosis, and mode switching");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.memorocyte_output_channel, 0..=15).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.memorocyte_output_channel).speed(0.1).range(0..=15));
                    });

                    // Output Hops
                    ui.label("Output Hops:")
                        .on_hover_text("How many adhesion bonds the memory signal propagates. Signal halves in strength each hop beyond the first");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.memorocyte_output_hops, 1..=20).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.memorocyte_output_hops).speed(0.1).range(1..=20));
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
                ui.label("Split Nutrients:")
                    .on_hover_text("How many nutrients this cell must accumulate before it can divide. 'Never' means the cell will never divide regardless of nutrients");
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
                ui.label("Split Interval:")
                    .on_hover_text("Minimum time in seconds between consecutive divisions. The cell will not divide again until this cooldown has elapsed after the previous split");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.split_interval, 1.0..=60.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.split_interval).speed(0.1).range(1.0..=60.0).suffix("s"));
                });

                ui.add_space(4.0);
                ui.label("Split Ratio:")
                    .on_hover_text("Controls only how adhesion bonds are distributed between children and the Zone C width. Nutrients and other cell state always split evenly. 0.5 = balanced adhesion inheritance; values toward 0 or 1 bias inheritance toward one child");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.split_ratio, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.split_ratio).speed(0.01).range(0.0..=1.0));
                });

                ui.add_space(4.0);
                ui.label("Max Cell Size:")
                    .on_hover_text("Maximum radius this cell can grow to before it is forced to divide (if nutrients allow). Larger values let the cell grow bigger before splitting");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.max_cell_size, 0.5..=2.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.max_cell_size).speed(0.01).range(0.5..=2.0));
                });

                ui.label("Membrane Stiffness:")
                    .on_hover_text("How strongly this cell resists being compressed by collisions. Higher values make the cell harder and bouncier; lower values make it softer and more deformable");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.membrane_stiffness, 0.0..=250.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.membrane_stiffness).speed(0.1).range(0.0..=250.0));
                });
            });

            // Developmental/Regulation Emit Group (Teal) - any cell type can emit on channels 8-15
            group_container(ui, "Regulation Emit", egui::Color32::from_rgb(80, 180, 170), |ui| {
                let reg_channel_labels = ["Disabled", "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                // Map regulation_emit_channel: -1 = Disabled (index 0), 8 = index 1, 9 = index 2, etc.
                let reg_ch_idx = if mode.regulation_emit_channel < 8 { 0usize } else { (mode.regulation_emit_channel - 7).clamp(0, 8) as usize };

                ui.label("Emit Channel:")
                    .on_hover_text("Developmental/regulation channel (8–15) to broadcast a signal on. Any cell type can emit on these channels. Use them to gate division, mode switching, and apoptosis. Channels 0–7 are sensory channels (oculocyte, photocyte, lipocyte, etc.)");
                egui::ComboBox::from_id_salt("reg_emit_channel")
                    .selected_text(reg_channel_labels[reg_ch_idx])
                    .show_ui(ui, |ui| {
                        for (i, label) in reg_channel_labels.iter().enumerate() {
                            let ch_val = if i == 0 { -1i32 } else { (i as i32) + 7 };
                            ui.selectable_value(&mut mode.regulation_emit_channel, ch_val, *label);
                        }
                    });

                if mode.regulation_emit_channel >= 8 {
                    ui.label("Emit Value:")
                        .on_hover_text("Signal strength broadcast from this cell. Attenuates by half at each adhesion hop. A value of 50 with 20 hops reaches every connected cell at ≥1.0 strength");
                    ui.horizontal(|ui| {
                        let available = ui.available_width();
                        let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                        ui.style_mut().spacing.slider_width = slider_width;
                        ui.add(egui::Slider::new(&mut mode.regulation_emit_value, 0.0..=2047.0).show_value(false));
                        ui.add(egui::DragValue::new(&mut mode.regulation_emit_value).speed(1.0).range(0.0..=2047.0));
                    });

                    ui.label("Emit Hops:")
                        .on_hover_text("How many adhesion bonds the signal travels through. Signal halves at each hop beyond the first. 3 hops = 25% strength at the 3rd cell. Use 15–20 for body-wide flood signals");
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
                // Signal conditionals only read from developmental/regulation channels 8-15
                let channel_labels = ["Disabled", "Ch 8", "Ch 9", "Ch 10", "Ch 11", "Ch 12", "Ch 13", "Ch 14", "Ch 15"];
                // Map dropdown index to channel value: 0 -> -1 (disabled), 1 -> 8, 2 -> 9, ...
                let idx_to_channel = |idx: usize| -> i32 { if idx == 0 { -1 } else { idx as i32 + 7 } };
                // Map channel value to dropdown index: -1 -> 0, 8 -> 1, 9 -> 2, ... (invalid -> 0)
                let channel_to_idx = |ch: i32| -> usize { if ch < 8 { 0 } else { (ch - 7).clamp(0, 8) as usize } };

                // --- Division Gating ---
                ui.label("Division Gating:")
                    .on_hover_text("Gate cell division on a developmental channel (8–15). The cell only divides when the signal condition is met. ⚠ Only channels 8–15 work here — sensory channels 0–7 will silently disable this gate");
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
                        ui.add(egui::Slider::new(&mut mode.division_signal_threshold, 0.0..=2047.0).show_value(false))
                            .on_hover_text("Signal strength threshold. The cell divides when the channel value is above this level (or below it if Invert is checked)");
                        ui.add(egui::DragValue::new(&mut mode.division_signal_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    ui.checkbox(&mut mode.division_signal_invert, "Invert (divide below threshold)")
                        .on_hover_text("Absence gating: when checked, the cell divides when the signal is BELOW the threshold (i.e. when the signal is absent). Use for hunger-driven growth, damage response, or juvenile-only growth that stops when a maturation signal arrives");
                }

                ui.add_space(4.0);
                ui.separator();

                // --- Apoptosis ---
                ui.label("Apoptosis (Signal Death):")
                    .on_hover_text("Trigger programmed cell death based on a developmental channel (8–15). The cell dies when the signal condition is met. ⚠ Only channels 8–15 work here — sensory channels 0–7 silently disable this");
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
                        ui.add(egui::Slider::new(&mut mode.apoptosis_signal_threshold, 0.0..=2047.0).show_value(false))
                            .on_hover_text("Signal strength threshold for triggering death");
                        ui.add(egui::DragValue::new(&mut mode.apoptosis_signal_threshold).speed(1.0).range(0.0..=2047.0));
                    });
                    ui.checkbox(&mut mode.apoptosis_signal_invert, "Invert (die below threshold)")
                        .on_hover_text("Absence gating: when checked, the cell dies when the signal is BELOW the threshold. Use for cells that survive only while a support signal is present (e.g. leaf cells that die when the maturation signal disappears)");
                }

                ui.add_space(4.0);
                ui.separator();

                // --- Signal-Conditional Child A Mode ---
                ui.label("Child A Signal Routing:")
                    .on_hover_text("Override which mode Child A is born as, based on a developmental channel (8–15). When the signal is above the threshold, Child A uses the 'Above' mode; otherwise it uses the 'Below' mode. Disabled = always use the default Child A mode");
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
                ui.label("Child B Signal Routing:")
                    .on_hover_text("Override which mode Child B is born as, based on a developmental channel (8–15). When the signal is above the threshold, Child B uses the 'Above' mode; otherwise it uses the 'Below' mode. Disabled = always use the default Child B mode");
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
                ui.label("Mode Switch (No Division):")
                    .on_hover_text("Switch this cell to a different mode without dividing, triggered by a developmental channel (8–15). The cell changes its behavior in-place. ⚠ Only channels 8–15 work here — sensory channels 0–7 silently disable this");
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
                        ui.add(egui::Slider::new(&mut mode.mode_switch_signal_threshold, 1.0..=2047.0).show_value(false))
                            .on_hover_text("Signal strength threshold for triggering the mode switch. Minimum 1 — a threshold of 0 would fire unconditionally with no signal present");
                        ui.add(egui::DragValue::new(&mut mode.mode_switch_signal_threshold).speed(1.0).range(1.0..=2047.0));
                    });
                    ui.checkbox(&mut mode.mode_switch_invert, "Invert (switch below threshold)")
                        .on_hover_text("Absence gating: when checked, the cell switches mode when the signal is BELOW the threshold. Use for starvation responses or dormancy when a support signal disappears");
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
                ui.label("Nutrient Priority:")
                    .on_hover_text("How aggressively this cell competes for nutrients from its vascular connections. Higher priority cells are fed first. Embryocytes use 4.0, gonads 3.5, structural cells 1.0–1.5, vascular pipes 0.4");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.nutrient_priority, 0.1..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.nutrient_priority).speed(0.01).range(0.1..=10.0));
                });

                ui.checkbox(&mut mode.prioritize_when_low, "Prioritize When Low")
                    .on_hover_text("When enabled, this cell's effective priority increases when its nutrient level is critically low, helping it recover before dying");
            });

            // Connection Settings Group (Cyan)
            group_container(ui, "Connection Settings", egui::Color32::from_rgb(100, 180, 200), |ui| {
                ui.label("Max Connections:")
                    .on_hover_text("Maximum number of adhesion bonds this cell can hold simultaneously. Hard limit is 20 per cell. Ring leaf nodes should budget carefully: 2 ring bonds + 1 per attached sequence");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.max_adhesions, 0..=20).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.max_adhesions).speed(1).range(0..=20));
                });

                ui.label("Min Connections:")
                    .on_hover_text("Minimum adhesion bonds required before this cell will divide or emit signals. Use as a structural completion gate — e.g. set to 4 on a gonad so it only starts shedding eggs once the body is fully connected");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.min_adhesions, 0..=10).show_value(false));
                    ui.add(egui::DragValue::new(&mut mode.min_adhesions).speed(1).range(0..=10));
                });

                ui.label("Max Splits:")
                    .on_hover_text("Maximum number of times this cell can divide. -1 (∞) means unlimited divisions. Once the limit is reached, the cell switches to the 'After Splits' mode if one is set");
                let max_splits_row = ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut mode.max_splits, -1..=20).show_value(false));

                    // Custom DragValue that shows infinity symbol for -1
                    let mut drag_value = egui::DragValue::new(&mut mode.max_splits)
                        .speed(0.1)
                        .range(-1.0..=20.0);

                    // Custom formatter to show  for -1
                    drag_value = drag_value.custom_formatter(|n, _| {
                        if n == -1.0 {
                            "∞".to_owned()
                        } else {
                            format!("{}", n as i32)
                        }
                    });

                    // Custom parser to handle  input
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
                    ui.label("Child A After Splits:")
                        .on_hover_text("Mode that Child A transitions to once this cell has divided the maximum number of times. 'None' keeps the normal Child A mode. A self-referential mode is equivalent to None.");
                    let after_splits_a_row = ui.horizontal(|ui| {
                        let selected_text = if current_mode_a < 0 {
                            "None".to_string()
                        } else if (current_mode_a as usize) < mode_count {
                            let full_name = mode_info_for_dropdowns[current_mode_a as usize].0.clone();
                            let max_chars = ((ui.available_width() - 28.0) / 7.0).max(3.0) as usize;
                            if full_name.len() > max_chars {
                                format!("{}…", &full_name[..full_name.char_indices().nth(max_chars.saturating_sub(1)).map(|(i, _)| i).unwrap_or(full_name.len())])
                            } else {
                                full_name
                            }
                        } else {
                            "Invalid".to_string()
                        };

                        egui::ComboBox::from_id_salt("mode_a_after_splits")
                            .selected_text(selected_text)
                            .width(ui.available_width() - 10.0)
                            .show_ui(ui, |ui| {
                                let item_width = ui.available_width();
                                // Default option
                                let (def_rect, def_response) = ui.allocate_exact_size(
                                    egui::vec2(item_width, 18.0),
                                    egui::Sense::click(),
                                );
                                let def_bg = if current_mode_a < 0 {
                                    egui::Color32::from_rgb(60, 80, 60)
                                } else if def_response.hovered() {
                                    egui::Color32::from_rgb(50, 60, 50)
                                } else {
                                    egui::Color32::from_rgb(35, 45, 35)
                                };
                                ui.painter().rect_filled(def_rect, 2.0, def_bg);
                                if current_mode_a < 0 {
                                    ui.painter().rect_stroke(def_rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                }
                                draw_marquee_text(ui, def_rect, "None", egui::Color32::GRAY, def_response.hovered(), egui::Id::new("after_a_default"));
                                if def_response.clicked() {
                                    new_mode_a = Some(-1);
                                }
                                // Mode options
                                for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                    let is_selected = current_mode_a == idx as i32;
                                    let bg_color = egui::Color32::from_rgb(
                                        (color.x * 255.0) as u8,
                                        (color.y * 255.0) as u8,
                                        (color.z * 255.0) as u8,
                                    );
                                    let luminance = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
                                    let text_color = if luminance > 0.5 { egui::Color32::BLACK } else { egui::Color32::WHITE };
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(item_width, 18.0),
                                        egui::Sense::click(),
                                    );
                                    let bg = if response.hovered() { bg_color.gamma_multiply(1.2) } else { bg_color };
                                    ui.painter().rect_filled(rect, 2.0, bg);
                                    if is_selected {
                                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                    }
                                    draw_marquee_text(ui, rect, name, text_color, response.hovered(), egui::Id::new(("after_a_item", idx)));
                                    if response.clicked() {
                                        new_mode_a = Some(idx as i32);
                                    }
                                }
                            });
                    });
                    context.editor_state.panel_rects.insert(
                        "after_splits_child_a".to_string(), after_splits_a_row.response.rect,
                    );

                    ui.label("Child B After Splits:")
                        .on_hover_text("Mode that Child B transitions to once this cell has divided the maximum number of times. 'None' keeps the normal Child B mode. A self-referential mode is equivalent to None.");
                    let after_splits_b_row = ui.horizontal(|ui| {
                        let selected_text = if current_mode_b < 0 {
                            "None".to_string()
                        } else if (current_mode_b as usize) < mode_count {
                            let full_name = mode_info_for_dropdowns[current_mode_b as usize].0.clone();
                            let max_chars = ((ui.available_width() - 28.0) / 7.0).max(3.0) as usize;
                            if full_name.len() > max_chars {
                                format!("{}…", &full_name[..full_name.char_indices().nth(max_chars.saturating_sub(1)).map(|(i, _)| i).unwrap_or(full_name.len())])
                            } else {
                                full_name
                            }
                        } else {
                            "Invalid".to_string()
                        };

                        egui::ComboBox::from_id_salt("mode_b_after_splits")
                            .selected_text(selected_text)
                            .width(ui.available_width() - 10.0)
                            .show_ui(ui, |ui| {
                                let item_width = ui.available_width();
                                // Default option
                                let (def_rect, def_response) = ui.allocate_exact_size(
                                    egui::vec2(item_width, 18.0),
                                    egui::Sense::click(),
                                );
                                let def_bg = if current_mode_b < 0 {
                                    egui::Color32::from_rgb(60, 80, 60)
                                } else if def_response.hovered() {
                                    egui::Color32::from_rgb(50, 60, 50)
                                } else {
                                    egui::Color32::from_rgb(35, 45, 35)
                                };
                                ui.painter().rect_filled(def_rect, 2.0, def_bg);
                                if current_mode_b < 0 {
                                    ui.painter().rect_stroke(def_rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                }
                                draw_marquee_text(ui, def_rect, "None", egui::Color32::GRAY, def_response.hovered(), egui::Id::new("after_b_default"));
                                if def_response.clicked() {
                                    new_mode_b = Some(-1);
                                }
                                // Mode options
                                for (idx, (name, color)) in mode_info_for_dropdowns.iter().enumerate() {
                                    let is_selected = current_mode_b == idx as i32;
                                    let bg_color = egui::Color32::from_rgb(
                                        (color.x * 255.0) as u8,
                                        (color.y * 255.0) as u8,
                                        (color.z * 255.0) as u8,
                                    );
                                    let luminance = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
                                    let text_color = if luminance > 0.5 { egui::Color32::BLACK } else { egui::Color32::WHITE };
                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(item_width, 18.0),
                                        egui::Sense::click(),
                                    );
                                    let bg = if response.hovered() { bg_color.gamma_multiply(1.2) } else { bg_color };
                                    ui.painter().rect_filled(rect, 2.0, bg);
                                    if is_selected {
                                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                    }
                                    draw_marquee_text(ui, rect, name, text_color, response.hovered(), egui::Id::new(("after_b_item", idx)));
                                    if response.clicked() {
                                        new_mode_b = Some(idx as i32);
                                    }
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
                            ui.checkbox(&mut mode.child_a_after_split_keep_adhesion, "Keep Adhesion")
                                .on_hover_text("When enabled, Child A maintains its adhesion bond with the parent cell after the max splits limit is reached");
                        });
                        ui.vertical(|ui| {
                            ui.checkbox(&mut mode.child_b_after_split_keep_adhesion, "Keep Adhesion")
                                .on_hover_text("When enabled, Child B maintains its adhesion bond with the parent cell after the max splits limit is reached");
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
            let secondary_indices: Vec<usize> = context
                .editor_state
                .selected_mode_indices
                .iter()
                .copied()
                .filter(|&i| i != selected_idx && i < context.genome.modes.len())
                .collect();
            if !secondary_indices.is_empty() {
                sync_mode_changes_to_others(
                    &snapshot,
                    &updated,
                    &secondary_indices,
                    &mut context.genome.modes,
                );
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
        if updated.cell_type != snapshot.cell_type {
            other.cell_type = updated.cell_type;
        }

        // Visual
        if (updated.opacity - snapshot.opacity).abs() > f32::EPSILON {
            other.opacity = updated.opacity;
        }
        if (updated.emissive - snapshot.emissive).abs() > f32::EPSILON {
            other.emissive = updated.emissive;
        }

        // Parent / division
        if updated.parent_make_adhesion != snapshot.parent_make_adhesion {
            other.parent_make_adhesion = updated.parent_make_adhesion;
        }
        if (updated.split_mass - snapshot.split_mass).abs() > f32::EPSILON {
            other.split_mass = updated.split_mass;
        }
        if (updated.split_interval - snapshot.split_interval).abs() > f32::EPSILON {
            other.split_interval = updated.split_interval;
        }
        if (updated.nutrient_gain_rate - snapshot.nutrient_gain_rate).abs() > f32::EPSILON {
            other.nutrient_gain_rate = updated.nutrient_gain_rate;
        }
        if (updated.max_cell_size - snapshot.max_cell_size).abs() > f32::EPSILON {
            other.max_cell_size = updated.max_cell_size;
        }
        if (updated.split_ratio - snapshot.split_ratio).abs() > f32::EPSILON {
            other.split_ratio = updated.split_ratio;
        }
        if (updated.nutrient_priority - snapshot.nutrient_priority).abs() > f32::EPSILON {
            other.nutrient_priority = updated.nutrient_priority;
        }
        if updated.prioritize_when_low != snapshot.prioritize_when_low {
            other.prioritize_when_low = updated.prioritize_when_low;
        }
        if (updated.parent_split_direction.x - snapshot.parent_split_direction.x).abs()
            > f32::EPSILON
            || (updated.parent_split_direction.y - snapshot.parent_split_direction.y).abs()
                > f32::EPSILON
        {
            other.parent_split_direction = updated.parent_split_direction;
        }
        if updated.max_adhesions != snapshot.max_adhesions {
            other.max_adhesions = updated.max_adhesions;
        }
        if updated.min_adhesions != snapshot.min_adhesions {
            other.min_adhesions = updated.min_adhesions;
        }
        if updated.enable_parent_angle_snapping != snapshot.enable_parent_angle_snapping {
            other.enable_parent_angle_snapping = updated.enable_parent_angle_snapping;
        }
        if updated.max_splits != snapshot.max_splits {
            other.max_splits = updated.max_splits;
        }
        if updated.mode_a_after_splits != snapshot.mode_a_after_splits {
            other.mode_a_after_splits = updated.mode_a_after_splits;
        }
        if updated.mode_b_after_splits != snapshot.mode_b_after_splits {
            other.mode_b_after_splits = updated.mode_b_after_splits;
        }
        if updated.child_a_after_split_keep_adhesion != snapshot.child_a_after_split_keep_adhesion {
            other.child_a_after_split_keep_adhesion = updated.child_a_after_split_keep_adhesion;
        }
        if updated.child_b_after_split_keep_adhesion != snapshot.child_b_after_split_keep_adhesion {
            other.child_b_after_split_keep_adhesion = updated.child_b_after_split_keep_adhesion;
        }

        // Membrane
        if (updated.membrane_stiffness - snapshot.membrane_stiffness).abs() > f32::EPSILON {
            other.membrane_stiffness = updated.membrane_stiffness;
        }

        // Glueocyte
        if updated.glueocyte_cell_adhesion != snapshot.glueocyte_cell_adhesion {
            other.glueocyte_cell_adhesion = updated.glueocyte_cell_adhesion;
        }
        if updated.glueocyte_self_adhesion != snapshot.glueocyte_self_adhesion {
            other.glueocyte_self_adhesion = updated.glueocyte_self_adhesion;
        }
        if updated.glueocyte_env_adhesion != snapshot.glueocyte_env_adhesion {
            other.glueocyte_env_adhesion = updated.glueocyte_env_adhesion;
        }
        if updated.glueocyte_boulder_adhesion != snapshot.glueocyte_boulder_adhesion {
            other.glueocyte_boulder_adhesion = updated.glueocyte_boulder_adhesion;
        }
        if updated.glueocyte_cell_adhesion_signal_channel
            != snapshot.glueocyte_cell_adhesion_signal_channel
        {
            other.glueocyte_cell_adhesion_signal_channel =
                updated.glueocyte_cell_adhesion_signal_channel;
        }
        if (updated.glueocyte_cell_adhesion_signal_threshold
            - snapshot.glueocyte_cell_adhesion_signal_threshold)
            .abs()
            > f32::EPSILON
        {
            other.glueocyte_cell_adhesion_signal_threshold =
                updated.glueocyte_cell_adhesion_signal_threshold;
        }
        if updated.glueocyte_signal_gate_invert != snapshot.glueocyte_signal_gate_invert {
            other.glueocyte_signal_gate_invert = updated.glueocyte_signal_gate_invert;
        }

        // Flagellocyte
        if (updated.swim_force - snapshot.swim_force).abs() > f32::EPSILON {
            other.swim_force = updated.swim_force;
        }
        if updated.flagellocyte_use_signal != snapshot.flagellocyte_use_signal {
            other.flagellocyte_use_signal = updated.flagellocyte_use_signal;
        }
        if updated.flagellocyte_signal_channel != snapshot.flagellocyte_signal_channel {
            other.flagellocyte_signal_channel = updated.flagellocyte_signal_channel;
        }
        if (updated.flagellocyte_speed_a - snapshot.flagellocyte_speed_a).abs() > f32::EPSILON {
            other.flagellocyte_speed_a = updated.flagellocyte_speed_a;
        }
        if (updated.flagellocyte_speed_b - snapshot.flagellocyte_speed_b).abs() > f32::EPSILON {
            other.flagellocyte_speed_b = updated.flagellocyte_speed_b;
        }
        if (updated.flagellocyte_threshold_c - snapshot.flagellocyte_threshold_c).abs()
            > f32::EPSILON
        {
            other.flagellocyte_threshold_c = updated.flagellocyte_threshold_c;
        }

        // Buoyocyte
        if (updated.buoyancy_force - snapshot.buoyancy_force).abs() > f32::EPSILON {
            other.buoyancy_force = updated.buoyancy_force;
        }

        // Oculocyte
        if updated.oculocyte_sense_type != snapshot.oculocyte_sense_type {
            other.oculocyte_sense_type = updated.oculocyte_sense_type;
        }
        if updated.oculocyte_signal_channel != snapshot.oculocyte_signal_channel {
            other.oculocyte_signal_channel = updated.oculocyte_signal_channel;
        }
        if (updated.oculocyte_signal_value - snapshot.oculocyte_signal_value).abs() > f32::EPSILON {
            other.oculocyte_signal_value = updated.oculocyte_signal_value;
        }
        if updated.oculocyte_signal_hops != snapshot.oculocyte_signal_hops {
            other.oculocyte_signal_hops = updated.oculocyte_signal_hops;
        }
        if (updated.oculocyte_ray_length - snapshot.oculocyte_ray_length).abs() > f32::EPSILON {
            other.oculocyte_ray_length = updated.oculocyte_ray_length;
        }
        if (updated.oculocyte_light_target_color - snapshot.oculocyte_light_target_color)
            .length_squared()
            > f32::EPSILON
        {
            other.oculocyte_light_target_color = updated.oculocyte_light_target_color;
        }
        if (updated.oculocyte_light_color_tolerance - snapshot.oculocyte_light_color_tolerance)
            .abs()
            > f32::EPSILON
        {
            other.oculocyte_light_color_tolerance = updated.oculocyte_light_color_tolerance;
        }

        // Ciliocyte
        if (updated.cilia_speed - snapshot.cilia_speed).abs() > f32::EPSILON {
            other.cilia_speed = updated.cilia_speed;
        }
        if updated.cilia_push_bonded != snapshot.cilia_push_bonded {
            other.cilia_push_bonded = updated.cilia_push_bonded;
        }
        if updated.cilia_use_signal != snapshot.cilia_use_signal {
            other.cilia_use_signal = updated.cilia_use_signal;
        }
        if updated.cilia_signal_channel != snapshot.cilia_signal_channel {
            other.cilia_signal_channel = updated.cilia_signal_channel;
        }
        if (updated.cilia_speed_below - snapshot.cilia_speed_below).abs() > f32::EPSILON {
            other.cilia_speed_below = updated.cilia_speed_below;
        }
        if (updated.cilia_speed_above - snapshot.cilia_speed_above).abs() > f32::EPSILON {
            other.cilia_speed_above = updated.cilia_speed_above;
        }
        if (updated.cilia_threshold - snapshot.cilia_threshold).abs() > f32::EPSILON {
            other.cilia_threshold = updated.cilia_threshold;
        }
        if (updated.cilia_attract_force - snapshot.cilia_attract_force).abs() > f32::EPSILON {
            other.cilia_attract_force = updated.cilia_attract_force;
        }

        // Myocyte
        if (updated.myocyte_contraction - snapshot.myocyte_contraction).abs() > f32::EPSILON {
            other.myocyte_contraction = updated.myocyte_contraction;
        }
        if updated.myocyte_use_signal != snapshot.myocyte_use_signal {
            other.myocyte_use_signal = updated.myocyte_use_signal;
        }
        if updated.myocyte_signal_channel != snapshot.myocyte_signal_channel {
            other.myocyte_signal_channel = updated.myocyte_signal_channel;
        }
        if (updated.myocyte_contraction_above - snapshot.myocyte_contraction_above).abs()
            > f32::EPSILON
        {
            other.myocyte_contraction_above = updated.myocyte_contraction_above;
        }
        if (updated.myocyte_contraction_below - snapshot.myocyte_contraction_below).abs()
            > f32::EPSILON
        {
            other.myocyte_contraction_below = updated.myocyte_contraction_below;
        }
        if (updated.myocyte_threshold - snapshot.myocyte_threshold).abs() > f32::EPSILON {
            other.myocyte_threshold = updated.myocyte_threshold;
        }
        if (updated.myocyte_pulse_rate - snapshot.myocyte_pulse_rate).abs() > f32::EPSILON {
            other.myocyte_pulse_rate = updated.myocyte_pulse_rate;
        }
        if updated.myocyte_pulse_phase != snapshot.myocyte_pulse_phase {
            other.myocyte_pulse_phase = updated.myocyte_pulse_phase;
        }
        if (updated.myocyte_grip_contracted - snapshot.myocyte_grip_contracted).abs() > f32::EPSILON
        {
            other.myocyte_grip_contracted = updated.myocyte_grip_contracted;
        }
        if (updated.myocyte_grip_extended - snapshot.myocyte_grip_extended).abs() > f32::EPSILON {
            other.myocyte_grip_extended = updated.myocyte_grip_extended;
        }

        // Embryocyte
        if updated.embryocyte_use_timer != snapshot.embryocyte_use_timer {
            other.embryocyte_use_timer = updated.embryocyte_use_timer;
        }
        if (updated.embryocyte_release_timer - snapshot.embryocyte_release_timer).abs()
            > f32::EPSILON
        {
            other.embryocyte_release_timer = updated.embryocyte_release_timer;
        }
        if updated.embryocyte_use_threshold != snapshot.embryocyte_use_threshold {
            other.embryocyte_use_threshold = updated.embryocyte_use_threshold;
        }
        if updated.embryocyte_threshold_value != snapshot.embryocyte_threshold_value {
            other.embryocyte_threshold_value = updated.embryocyte_threshold_value;
        }
        if updated.embryocyte_use_signal != snapshot.embryocyte_use_signal {
            other.embryocyte_use_signal = updated.embryocyte_use_signal;
        }
        if updated.embryocyte_signal_channel != snapshot.embryocyte_signal_channel {
            other.embryocyte_signal_channel = updated.embryocyte_signal_channel;
        }
        if (updated.embryocyte_signal_value - snapshot.embryocyte_signal_value).abs() > f32::EPSILON
        {
            other.embryocyte_signal_value = updated.embryocyte_signal_value;
        }

        // Regulation emit
        if updated.regulation_emit_channel != snapshot.regulation_emit_channel {
            other.regulation_emit_channel = updated.regulation_emit_channel;
        }
        if (updated.regulation_emit_value - snapshot.regulation_emit_value).abs() > f32::EPSILON {
            other.regulation_emit_value = updated.regulation_emit_value;
        }
        if updated.regulation_emit_hops != snapshot.regulation_emit_hops {
            other.regulation_emit_hops = updated.regulation_emit_hops;
        }

        // Signal conditions - division
        if updated.division_signal_channel != snapshot.division_signal_channel {
            other.division_signal_channel = updated.division_signal_channel;
        }
        if (updated.division_signal_threshold - snapshot.division_signal_threshold).abs()
            > f32::EPSILON
        {
            other.division_signal_threshold = updated.division_signal_threshold;
        }
        if updated.division_signal_invert != snapshot.division_signal_invert {
            other.division_signal_invert = updated.division_signal_invert;
        }

        // Signal conditions - apoptosis
        if updated.apoptosis_signal_channel != snapshot.apoptosis_signal_channel {
            other.apoptosis_signal_channel = updated.apoptosis_signal_channel;
        }
        if (updated.apoptosis_signal_threshold - snapshot.apoptosis_signal_threshold).abs()
            > f32::EPSILON
        {
            other.apoptosis_signal_threshold = updated.apoptosis_signal_threshold;
        }
        if updated.apoptosis_signal_invert != snapshot.apoptosis_signal_invert {
            other.apoptosis_signal_invert = updated.apoptosis_signal_invert;
        }

        // Signal conditions - child routing
        if updated.signal_child_a_channel != snapshot.signal_child_a_channel {
            other.signal_child_a_channel = updated.signal_child_a_channel;
        }
        if (updated.signal_child_a_threshold - snapshot.signal_child_a_threshold).abs()
            > f32::EPSILON
        {
            other.signal_child_a_threshold = updated.signal_child_a_threshold;
        }
        if updated.signal_child_a_mode_above != snapshot.signal_child_a_mode_above {
            other.signal_child_a_mode_above = updated.signal_child_a_mode_above;
        }
        if updated.signal_child_a_mode_below != snapshot.signal_child_a_mode_below {
            other.signal_child_a_mode_below = updated.signal_child_a_mode_below;
        }
        if updated.signal_child_b_channel != snapshot.signal_child_b_channel {
            other.signal_child_b_channel = updated.signal_child_b_channel;
        }
        if (updated.signal_child_b_threshold - snapshot.signal_child_b_threshold).abs()
            > f32::EPSILON
        {
            other.signal_child_b_threshold = updated.signal_child_b_threshold;
        }
        if updated.signal_child_b_mode_above != snapshot.signal_child_b_mode_above {
            other.signal_child_b_mode_above = updated.signal_child_b_mode_above;
        }
        if updated.signal_child_b_mode_below != snapshot.signal_child_b_mode_below {
            other.signal_child_b_mode_below = updated.signal_child_b_mode_below;
        }

        // Signal conditions - mode switch
        if updated.mode_switch_signal_channel != snapshot.mode_switch_signal_channel {
            other.mode_switch_signal_channel = updated.mode_switch_signal_channel;
        }
        if (updated.mode_switch_signal_threshold - snapshot.mode_switch_signal_threshold).abs()
            > f32::EPSILON
        {
            other.mode_switch_signal_threshold = updated.mode_switch_signal_threshold;
        }
        if updated.mode_switch_target != snapshot.mode_switch_target {
            other.mode_switch_target = updated.mode_switch_target;
        }
        if updated.stemocyte_signal_channel != snapshot.stemocyte_signal_channel {
            other.stemocyte_signal_channel = updated.stemocyte_signal_channel;
        }
        if updated.stemocyte_weak_first != snapshot.stemocyte_weak_first {
            other.stemocyte_weak_first = updated.stemocyte_weak_first;
        }
        if updated.stemocyte_outcomes != snapshot.stemocyte_outcomes {
            other.stemocyte_outcomes = updated.stemocyte_outcomes;
        }
        if updated.stemocyte_thresholds != snapshot.stemocyte_thresholds {
            other.stemocyte_thresholds = updated.stemocyte_thresholds;
        }
        if updated.stemocyte_delay_mode != snapshot.stemocyte_delay_mode {
            other.stemocyte_delay_mode = updated.stemocyte_delay_mode;
        }
        if (updated.stemocyte_delay_value - snapshot.stemocyte_delay_value).abs() > f32::EPSILON {
            other.stemocyte_delay_value = updated.stemocyte_delay_value;
        }
        if updated.mode_switch_invert != snapshot.mode_switch_invert {
            other.mode_switch_invert = updated.mode_switch_invert;
        }

        // Devorocyte
        if (updated.devorocyte_consume_range - snapshot.devorocyte_consume_range).abs()
            > f32::EPSILON
        {
            other.devorocyte_consume_range = updated.devorocyte_consume_range;
        }
        if (updated.devorocyte_consume_rate - snapshot.devorocyte_consume_rate).abs() > f32::EPSILON
        {
            other.devorocyte_consume_rate = updated.devorocyte_consume_rate;
        }

        // Vasculocyte
        if updated.vascular_nutrient_transport != snapshot.vascular_nutrient_transport {
            other.vascular_nutrient_transport = updated.vascular_nutrient_transport;
        }
        if updated.vascular_outlet != snapshot.vascular_outlet {
            other.vascular_outlet = updated.vascular_outlet;
        }
        if updated.vascular_signal_transport != snapshot.vascular_signal_transport {
            other.vascular_signal_transport = updated.vascular_signal_transport;
        }
        if updated.vascular_signal_exchange != snapshot.vascular_signal_exchange {
            other.vascular_signal_exchange = updated.vascular_signal_exchange;
        }
        if (updated.vascular_signal_capacity - snapshot.vascular_signal_capacity).abs()
            > f32::EPSILON
        {
            other.vascular_signal_capacity = updated.vascular_signal_capacity;
        }

        // Gametocyte
        if (updated.gametocyte_merge_range - snapshot.gametocyte_merge_range).abs() > f32::EPSILON {
            other.gametocyte_merge_range = updated.gametocyte_merge_range;
        }
    }
}

/// Render the CircleSliders panel with pitch and yaw controls for parent split direction.
fn render_circle_sliders(ui: &mut Ui, context: &mut PanelContext) {
    use crate::ui::widgets::circular_slider_float;

    // Record panel rect for the tutorial pointer.
    context
        .editor_state
        .panel_rects
        .insert("CircleSliders".to_string(), ui.max_rect());

    ui.add_space(4.0); // Reduced spacing

    // Ensure we have at least one mode
    if context.genome.modes.is_empty() {
        context
            .genome
            .modes
            .push(crate::genome::ModeSettings::default());
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
        let yaw_changed = (new_dir.y - prev_dir.y).abs() > f32::EPSILON;
        if pitch_changed || yaw_changed {
            let secondary_indices: Vec<usize> = context
                .editor_state
                .selected_mode_indices
                .iter()
                .copied()
                .filter(|&i| i != selected_index && i < context.genome.modes.len())
                .collect();
            for other_idx in secondary_indices {
                if pitch_changed {
                    context.genome.modes[other_idx].parent_split_direction.x = new_dir.x;
                }
                if yaw_changed {
                    context.genome.modes[other_idx].parent_split_direction.y = new_dir.y;
                }
            }
        }
    }
}

/// Render the QuaternionBall panel with two quaternion balls for Child A and Child B.
fn render_quaternion_ball(ui: &mut Ui, context: &mut PanelContext) {
    context
        .editor_state
        .panel_rects
        .insert("QuaternionBall".to_string(), ui.max_rect());

    // Snapshot child settings before rendering for multi-select diff
    let selected_idx = context.editor_state.selected_mode_index;
    let pre_child_a = context
        .genome
        .modes
        .get(selected_idx)
        .map(|m| m.child_a.clone());
    let pre_child_b = context
        .genome
        .modes
        .get(selected_idx)
        .map(|m| m.child_b.clone());

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
        // Reset button - clears both child orientations back to identity
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new(
                egui::RichText::new("↺ Reset Orientations").size(11.0))
                .min_size(egui::vec2(0.0, 20.0)))
                .on_hover_text("Reset both Child A and Child B orientations to identity (no rotation)")
                .clicked()
            {
                let identity = glam::Quat::IDENTITY;
                context.editor_state.child_a_orientation = identity;
                context.editor_state.child_b_orientation = identity;
                context.editor_state.child_a_x_axis_lat = 0.0;
                context.editor_state.child_a_x_axis_lon = 0.0;
                context.editor_state.child_a_y_axis_lat = 0.0;
                context.editor_state.child_a_y_axis_lon = 0.0;
                context.editor_state.child_a_z_axis_lat = 0.0;
                context.editor_state.child_a_z_axis_lon = 0.0;
                context.editor_state.child_b_x_axis_lat = 0.0;
                context.editor_state.child_b_x_axis_lon = 0.0;
                context.editor_state.child_b_y_axis_lat = 0.0;
                context.editor_state.child_b_y_axis_lon = 0.0;
                context.editor_state.child_b_z_axis_lat = 0.0;
                context.editor_state.child_b_z_axis_lon = 0.0;
                context.editor_state.qball1_locked_axis = -1;
                context.editor_state.qball2_locked_axis = -1;
                let idx = context.editor_state.selected_mode_index;
                if idx < context.genome.modes.len() {
                    context.genome.modes[idx].child_a.orientation = identity;
                    context.genome.modes[idx].child_b.orientation = identity;
                }
            }
        });
        ui.add_space(4.0);

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
                        let adh_a = ui.checkbox(&mut context.genome.modes[selected_idx].child_a.keep_adhesion, "Keep Adhesion")
                            .on_hover_text("When enabled, Child A maintains its adhesion bond with the parent cell after division");
                        context.editor_state.panel_rects.insert(
                            "child_a_keep_adhesion".to_string(), adh_a.rect,
                        );
                    } else {
                        ui.add_enabled(false, egui::Checkbox::new(&mut false, "Keep Adhesion"));
                    }

                    // Mode dropdown for Child A
                    ui.label("Mode:");
                    if has_valid_mode {
                        let current_mode = context.genome.modes[selected_idx].child_a.mode_number;
                        let mode_count = context.genome.modes.len();
                        let full_name = if current_mode >= 0 && (current_mode as usize) < mode_count {
                            context.genome.modes[current_mode as usize].name.clone()
                        } else {
                            "Invalid".to_string()
                        };
                        // Truncate the button label so it never expands the dropdown width.
                        // Reserve ~20px for the dropdown arrow; ~7px per character is a safe estimate.
                        let max_chars = ((ball_container_width - 28.0) / 7.0).max(3.0) as usize;
                        let selected_text = if full_name.len() > max_chars {
                            format!("{}…", &full_name[..full_name.char_indices().nth(max_chars.saturating_sub(1)).map(|(i, _)| i).unwrap_or(full_name.len())])
                        } else {
                            full_name.clone()
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

                                    let bg = if response.hovered() {
                                        bg_color.gamma_multiply(1.2)
                                    } else {
                                        bg_color
                                    };
                                    ui.painter().rect_filled(rect, 2.0, bg);

                                    if is_selected {
                                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                    }

                                    let marquee_id = egui::Id::new(("qball1_item", i));
                                    draw_marquee_text(ui, rect, name, text_color, response.hovered(), marquee_id);

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
                        let adh_b = ui.checkbox(&mut context.genome.modes[selected_idx].child_b.keep_adhesion, "Keep Adhesion")
                            .on_hover_text("When enabled, Child B maintains its adhesion bond with the parent cell after division");
                        context.editor_state.panel_rects.insert(
                            "child_b_keep_adhesion".to_string(), adh_b.rect,
                        );
                    } else {
                        ui.add_enabled(false, egui::Checkbox::new(&mut false, "Keep Adhesion"));
                    }

                    // Mode dropdown for Child B
                    ui.label("Mode:");
                    if has_valid_mode {
                        let current_mode = context.genome.modes[selected_idx].child_b.mode_number;
                        let mode_count = context.genome.modes.len();
                        let full_name = if current_mode >= 0 && (current_mode as usize) < mode_count {
                            context.genome.modes[current_mode as usize].name.clone()
                        } else {
                            "Invalid".to_string()
                        };
                        // Truncate the button label so it never expands the dropdown width.
                        let max_chars = ((ball_container_width - 28.0) / 7.0).max(3.0) as usize;
                        let selected_text = if full_name.len() > max_chars {
                            format!("{}…", &full_name[..full_name.char_indices().nth(max_chars.saturating_sub(1)).map(|(i, _)| i).unwrap_or(full_name.len())])
                        } else {
                            full_name.clone()
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

                                    let bg = if response.hovered() {
                                        bg_color.gamma_multiply(1.2)
                                    } else {
                                        bg_color
                                    };
                                    ui.painter().rect_filled(rect, 2.0, bg);

                                    if is_selected {
                                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                                    }

                                    let marquee_id = egui::Id::new(("qball2_item", i));
                                    draw_marquee_text(ui, rect, name, text_color, response.hovered(), marquee_id);

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

            let secondary_indices: Vec<usize> = context
                .editor_state
                .selected_mode_indices
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
    context
        .editor_state
        .panel_rects
        .insert("TimeSlider".to_string(), ui.max_rect());

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let is_preview_mode = context.current_mode == SimulationMode::Preview;

            // Always allocate space for status text to prevent shifting
            if !is_preview_mode {
                ui.colored_label(
                    egui::Color32::GRAY,
                    "Time scrubbing only available in Preview mode",
                );
            } else {
                ui.label("");
            }

            ui.horizontal(|ui| {
                ui.label("Time:");

                let available = ui.available_width();
                let slider_width = if available > 80.0 {
                    available - 70.0
                } else {
                    50.0
                };
                ui.style_mut().spacing.slider_width = slider_width;

                // Slider range is 0 to max_preview_duration (in seconds)
                // The slider is purely user-driven - simulation never writes to time_value
                let slider_response = ui.add_enabled(
                    is_preview_mode,
                    egui::Slider::new(
                        &mut context.editor_state.time_value,
                        0.0..=context.editor_state.max_preview_duration,
                    )
                    .show_value(false),
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
                        .suffix("s"),
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
    section_header(ui, "GIZMO RENDERING");
    ui.checkbox(&mut context.editor_state.gizmo_visible, "Show Orientation Lines")
        .on_hover_text("Display colored axis lines on each cell showing its current orientation. Red = X, Green = Y, Blue = Z");
    ui.checkbox(
        &mut context.editor_state.split_rings_visible,
        "Show Split Rings",
    )
    .on_hover_text("Display animated ring effects around cells that are about to divide");
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
            ui.label("Outline Width:")
                .on_hover_text("Width of the dark outline drawn around each cell. 0 = no outline; higher values create a more pronounced cell border");
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
            ui.label("Specular Strength:")
                .on_hover_text("How much this cell type reflects specular (mirror-like) highlights. 0 = fully matte; 1 = maximum gloss");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.specular_strength, 0.0..=1.0).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.specular_strength).speed(0.01).range(0.0..=1.0));
            });

            // Specular Power (shininess)
            ui.label("Specular Sharpness:")
                .on_hover_text("Tightness of the specular highlight. Low values = broad, soft gloss. High values = tight, sharp glint like polished metal");
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                ui.style_mut().spacing.slider_width = slider_width;
                ui.add(egui::Slider::new(&mut visuals.specular_power, 1.0..=128.0).logarithmic(true).show_value(false));
                ui.add(egui::DragValue::new(&mut visuals.specular_power).speed(0.5).range(1.0..=128.0));
            });

            // Fresnel (rim lighting)
            ui.label("Rim Lighting:")
                .on_hover_text("Backlit glow around the cell edges. Creates a halo effect that makes cells stand out against the background");
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
                ui.label("Tail Length:")
                    .on_hover_text("Length of the flagellum tail relative to the cell radius");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_length, 0.5..=3.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_length).speed(0.01).range(0.5..=3.0));
                });

                // Tail Thickness (0.01 - 0.3, default 0.15)
                ui.label("Tail Thickness:")
                    .on_hover_text("Diameter of the flagellum at its base");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_thickness, 0.01..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_thickness).speed(0.005).range(0.01..=0.3));
                });

                // Wave Amplitude (0.0 - 0.5, default 0.17)
                ui.label("Wave Amplitude:")
                    .on_hover_text("How far the flagellum bends side-to-side during its wave motion");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_amplitude, 0.0..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_amplitude).speed(0.01).range(0.0..=0.5));
                });

                // Wave Frequency (0.5 - 10.0, default 1.0)
                ui.label("Wave Frequency:")
                    .on_hover_text("How many wave cycles appear along the flagellum length. Higher values create more tightly coiled waves");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_frequency, 0.5..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_frequency).speed(0.1).range(0.5..=10.0));
                });

                // Taper (0.0 - 1.0, default 1.0)
                ui.label("Taper:")
                    .on_hover_text("How much the flagellum narrows toward its tip. 0 = uniform thickness; 1 = tapers to a point");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.tail_taper, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.tail_taper).speed(0.01).range(0.0..=1.0));
                });

                // Segments (4 - 64, default 10)
                ui.label("Segments:")
                    .on_hover_text("Number of geometry segments in the flagellum mesh. More segments = smoother curves but higher GPU cost");
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

                ui.label("Cell Scale:")
                    .on_hover_text("Scale of the slime pattern cells on the surface. Higher values create larger, more visible pattern cells");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 1.0..=12.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(0.1).range(1.0..=12.0));
                });

                ui.label("Border Width:")
                    .on_hover_text("Width of the borders between slime pattern cells");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.01..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.005).range(0.01..=0.5));
                });

                ui.label("Meander:")
                    .on_hover_text("How much the pattern borders waver and curve. Higher values create a more organic, irregular look");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.01).range(0.0..=1.0));
                });

                ui.label("Border Darkness:")
                    .on_hover_text("How dark the borders between pattern cells appear. Higher values create more visible, defined borders");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.01).range(0.0..=1.0));
                });

                ui.label("Anim Speed:")
                    .on_hover_text("Speed of the animated slime pattern movement. 0 = static pattern; higher values create faster flowing motion");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.membrane_noise_speed, 0.0..=5.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.membrane_noise_speed).speed(0.05).range(0.0..=5.0));
                });

                ui.label("Pattern Scale:")
                    .on_hover_text("Scale of the underlying noise pattern driving the slime texture. Higher values create finer, more detailed noise");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.membrane_noise_scale, 0.0..=10.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.membrane_noise_scale).speed(0.1).range(0.0..=10.0));
                });

                ui.label("Pattern Strength:")
                    .on_hover_text("How strongly the noise pattern perturbs the surface normals. Higher values create a more bumpy, textured appearance");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.membrane_noise_strength, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.membrane_noise_strength).speed(0.01).range(0.0..=1.0));
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
                ui.label("Nucleus Scale:")
                    .on_hover_text("Size of the inner nucleus sphere relative to the cell. Smaller values create a more prominent nucleus");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.nucleus_scale, 0.1..=0.95).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.nucleus_scale).speed(0.01).range(0.1..=0.95));
                });

                // Subdivision level (1-6)
                ui.label("Subdivision:")
                    .on_hover_text("Number of membrane ridge subdivisions. Higher values create more, finer ridges on the cell surface");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 1.0..=6.0).show_value(false).integer());
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(1.0).range(1.0..=6.0));
                });

                // Ridge Width
                ui.label("Ridge Width:")
                    .on_hover_text("Width of the membrane ridges on the cell surface");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.01..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.005).range(0.01..=0.5));
                });

                // Meander
                ui.label("Meander:")
                    .on_hover_text("How much the ridge lines waver. Higher values create more organic, irregular ridge patterns");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.005).range(0.0..=0.3));
                });

                // Ridge Strength (normal perturbation)
                ui.label("Ridge Depth:")
                    .on_hover_text("How pronounced the ridges appear. Higher values create deeper, more visible ridges with stronger normal map perturbation");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.0..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.005).range(0.0..=0.5));
                });
            }

            // Cilia rings section (only for Ciliocyte cell type)
            if cell_types.get(selected_idx) == Some(&CellType::Ciliocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Cilia Rings").strong());
                ui.add_space(4.0);

                ui.label("Ring Frequency:")
                    .on_hover_text("Number of visible cilia ring bands along the cell's forward axis. Higher values create more, tighter bands");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.cilia_ring_frequency, 2.0..=20.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.cilia_ring_frequency).speed(0.1).range(2.0..=20.0));
                });

                ui.label("Ring Depth:")
                    .on_hover_text("How strongly the cilia rings perturb the surface normals. Higher values create deeper, more pronounced ring grooves");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.cilia_ring_depth, 0.0..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.cilia_ring_depth).speed(0.005).range(0.0..=0.5));
                });

                ui.label("Scroll Speed:")
                    .on_hover_text("Base animation scroll rate of the cilia ring pattern. Higher values create faster-beating cilia motion");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.cilia_ring_speed, 0.0..=20.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.cilia_ring_speed).speed(0.1).range(0.0..=20.0));
                });
            }

            // Test cell: concentric rings pattern
            if cell_types.get(selected_idx) == Some(&CellType::Test) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Concentric Rings").strong());
                ui.add_space(4.0);

                ui.label("Ring Frequency:")
                    .on_hover_text("Number of concentric rings. 0 = plain sphere with no pattern");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.0..=20.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.1).range(0.0..=20.0));
                });

                ui.label("Ring Sharpness:")
                    .on_hover_text("How sharp the ring edges are. Lower values create soft, blended rings; higher values create crisp bands");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.01..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.01).range(0.01..=1.0));
                });

                ui.label("Ring Brightness:")
                    .on_hover_text("Intensity of the ring pattern. 0 = plain sphere; higher values make rings more visible");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.01).range(0.0..=1.0));
                });
            }

            // Phagocyte: nucleus appearance
            if cell_types.get(selected_idx) == Some(&CellType::Phagocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Nucleus").strong());
                ui.add_space(4.0);

                ui.label("Nucleus Radius:")
                    .on_hover_text("Size of the nucleus sphere visible through the cytoplasm");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.1..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.005).range(0.1..=0.5));
                });

                ui.label("Nucleus Darkness:")
                    .on_hover_text("How much darker the nucleus appears compared to the surrounding cytoplasm");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.1..=0.8).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.01).range(0.1..=0.8));
                });

                ui.label("Edge Softness:")
                    .on_hover_text("How soft the nucleus boundary is. Lower values create a sharper, more defined nucleus edge");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.01..=0.15).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.002).range(0.01..=0.15));
                });
            }

            // Lipocyte: fat droplet appearance
            if cell_types.get(selected_idx) == Some(&CellType::Lipocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Fat Droplets").strong());
                ui.add_space(4.0);

                ui.label("Droplet Scale:")
                    .on_hover_text("Size of the fat droplets. Lower values create many small droplets; higher values create fewer, larger blobs");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 1.0..=8.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.05).range(1.0..=8.0));
                });

                ui.label("Droplet Threshold:")
                    .on_hover_text("Noise threshold that determines where droplets form. Lower values create more droplets; higher values create fewer");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.2..=0.6).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.005).range(0.2..=0.6));
                });

                ui.label("Boundary Sharpness:")
                    .on_hover_text("How sharp the dark boundaries between droplets are");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.05..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.005).range(0.05..=0.3));
                });

                ui.label("Brightness:")
                    .on_hover_text("Overall brightness of the droplet pattern");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_d, 0.3..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_d).speed(0.01).range(0.3..=1.0));
                });
            }

            // Buoyocyte: gas bubble appearance
            if cell_types.get(selected_idx) == Some(&CellType::Buoyocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Gas Bubbles").strong());
                ui.add_space(4.0);

                ui.label("Bubble Size:")
                    .on_hover_text("Scale of all gas bubbles inside the cell. Values below 1.0 create smaller bubbles; above 1.0 creates larger ones");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.5..=2.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.01).range(0.5..=2.0));
                });

                ui.label("Rotation Speed:")
                    .on_hover_text("How fast the bubble cluster rotates inside the cell");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.1..=2.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.01).range(0.1..=2.0));
                });

                ui.label("Membrane Brightness:")
                    .on_hover_text("Brightness of the bubble membrane walls");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.3..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.01).range(0.3..=1.0));
                });

                ui.label("Gas Brightness:")
                    .on_hover_text("Brightness of the gas interior inside each bubble");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_d, 0.5..=1.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_d).speed(0.01).range(0.5..=1.5));
                });
            }

            // Devorocyte: spike appearance
            if cell_types.get(selected_idx) == Some(&CellType::Devorocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Spikes").strong());
                ui.add_space(4.0);

                ui.label("Spike Length:")
                    .on_hover_text("How far the spikes extend beyond the cell surface");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.3..=1.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.01).range(0.3..=1.5));
                });

                ui.label("Spike Sharpness:")
                    .on_hover_text("How narrow the spike cones are. Higher values create thinner, needle-like spikes; lower values create broader, stubbier ones");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.95..=0.999).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.001).range(0.95..=0.999));
                });

                ui.label("Embed Depth:")
                    .on_hover_text("How far the spike base is embedded into the cell body. Higher values anchor spikes more deeply");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.05..=0.25).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.005).range(0.05..=0.25));
                });

                ui.label("Tip Fade:")
                    .on_hover_text("How much the spike darkens toward its tip. 0 = uniform color; 1 = fully fades to dark at the tip");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_d, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_d).speed(0.01).range(0.0..=1.0));
                });
            }

            // Myocyte: muscle fiber appearance
            if cell_types.get(selected_idx) == Some(&CellType::Myocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Muscle Fibers").strong());
                ui.add_space(4.0);

                ui.label("Fiber Count:")
                    .on_hover_text("Number of longitudinal muscle fiber bundles visible on the cell surface. Higher values create more, finer fibers");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 4.0..=20.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(0.1).range(4.0..=20.0));
                });

                ui.label("Bulge Strength:")
                    .on_hover_text("How much each fiber bundle bulges outward. Higher values create more pronounced 3D fiber ridges");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.1..=0.9).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.01).range(0.1..=0.9));
                });

                ui.label("Fiber Warp:")
                    .on_hover_text("How much the fiber lines deviate from straight meridians. Higher values create a more organic, twisted fiber appearance");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.005).range(0.0..=0.3));
                });
            }

            // Embryocyte: egg yolk appearance
            if cell_types.get(selected_idx) == Some(&CellType::Embryocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Egg Yolk").strong());
                ui.add_space(4.0);

                ui.label("Yolk Radius:")
                    .on_hover_text("Size of the yolk sphere inside the egg. Larger values fill more of the cell interior");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 0.3..=0.7).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(0.01).range(0.3..=0.7));
                });

                ui.label("Yolk Drop:")
                    .on_hover_text("How far the yolk sinks toward the bottom of the cell. Higher values create a more settled, gravity-affected yolk");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.0..=0.35).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.005).range(0.0..=0.35));
                });

                ui.label("Yolk Brightness:")
                    .on_hover_text("How bright and warm the yolk appears. Higher values create a more vivid amber glow");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.5..=1.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.01).range(0.5..=1.5));
                });
            }

            // Oculocyte: eye appearance
            if cell_types.get(selected_idx) == Some(&CellType::Oculocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Eye").strong());
                ui.add_space(4.0);

                ui.label("Pupil Size:")
                    .on_hover_text("Radius of the dark pupil at the center of the eye. Larger values create a wider, more dilated pupil");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 0.1..=0.45).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(0.005).range(0.1..=0.45));
                });

                ui.label("Iris Detail:")
                    .on_hover_text("Number of radial striations in the iris. Higher values create a more complex, detailed iris pattern");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 2.0..=16.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.1).range(2.0..=16.0));
                });

                ui.label("Iris Texture:")
                    .on_hover_text("Blend between two overlapping striation patterns. Higher values create a more complex, layered iris texture");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=0.6).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.01).range(0.0..=0.6));
                });

                ui.label("Pupil Darkness:")
                    .on_hover_text("How dark the pupil appears. Higher values create a deeper, more absorbing pupil");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.5..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.01).range(0.5..=1.0));
                });
            }

            // Vasculocyte: vessel wall pattern
            if cell_types.get(selected_idx) == Some(&CellType::Vasculocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Vessel Wall").strong());
                ui.add_space(4.0);

                ui.label("Cell Scale:")
                    .on_hover_text("Scale of the endothelial cell pattern on the vessel wall surface");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_scale, 1.0..=12.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_scale).speed(0.1).range(1.0..=12.0));
                });

                ui.label("Border Width:")
                    .on_hover_text("Width of the borders between vessel wall cells");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_width, 0.01..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_width).speed(0.005).range(0.01..=0.5));
                });

                ui.label("Meander:")
                    .on_hover_text("How much the cell borders waver. Higher values create a more organic, irregular vessel wall pattern");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_meander, 0.0..=0.3).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_meander).speed(0.005).range(0.0..=0.3));
                });

                ui.label("Border Depth:")
                    .on_hover_text("How strongly the cell borders perturb the surface normals, creating a 3D cobblestone appearance");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.0..=0.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.005).range(0.0..=0.5));
                });
            }

            // Siphonocyte: rear aperture appearance
            if cell_types.get(selected_idx) == Some(&CellType::Siphonocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Nozzle").strong());
                ui.add_space(4.0);

                ui.label("Crater Radius:")
                    .on_hover_text("Base radius of the rear Siphonocyte volcanic nozzle.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.08..=0.65).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.005).range(0.08..=0.65));
                });

                ui.label("Throat Darkness:")
                    .on_hover_text("Darkening strength inside the nozzle throat.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.0..=1.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.01).range(0.0..=1.0));
                });

                ui.label("Rim Brightness:")
                    .on_hover_text("Highlight strength on the raised crater rim.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.0..=1.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.01).range(0.0..=1.5));
                });

                ui.label("Nozzle Height:")
                    .on_hover_text("How far the rear crater rises off the cell surface.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_d, 0.0..=0.55).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_d).speed(0.005).range(0.0..=0.55));
                });

                ui.label("Embed Depth:")
                    .on_hover_text("How deeply the nozzle base sinks into the cell border so it reads as attached tissue.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.goldberg_ridge_strength, 0.0..=0.28).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.goldberg_ridge_strength).speed(0.002).range(0.0..=0.28));
                });
            }

            // Plumocyte: feathered starfish appearance
            if cell_types.get(selected_idx) == Some(&CellType::Plumocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Feathers").strong());
                ui.add_space(4.0);

                ui.label("Feather Length:")
                    .on_hover_text("Length of each of the 8 fixed radial feathers. The feather equator is oriented to local gravity.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.2..=1.35).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.01).range(0.2..=1.35));
                });

                ui.label("Feather Width:")
                    .on_hover_text("Thickness of each feather stroke and its barbs. Higher values create broader, softer feathers.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.025..=0.22).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.002).range(0.025..=0.22));
                });

                ui.label("Feather Brightness:")
                    .on_hover_text("Brightness of the feathered starfish pattern inside the Plumocyte.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.0..=1.8).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.01).range(0.0..=1.8));
                });

                ui.label("Stroke Speed:")
                    .on_hover_text("Speed of the alternating even/odd feather stroke cycle. Frozen cells halt the animation.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_d, 0.0..=8.0).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_d).speed(0.05).range(0.0..=8.0));
                });
            }

            // Stemocyte: fivefold pluripotent rosette
            if cell_types.get(selected_idx) == Some(&CellType::Stemocyte) {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Pluripotent Rosette").strong());
                ui.label("A central undifferentiated core branches toward five daughter buds, visually echoing the five gradient outcomes.");
                ui.add_space(4.0);

                ui.label("Core Radius:")
                    .on_hover_text("Size of the bright central pluripotent nucleus.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_a, 0.16..=0.38).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_a).speed(0.005).range(0.16..=0.38));
                });

                ui.label("Daughter Bud Size:")
                    .on_hover_text("Size of the five potential daughter-state nuclei around the core.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_b, 0.07..=0.22).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_b).speed(0.004).range(0.07..=0.22));
                });

                ui.label("Branch Brightness:")
                    .on_hover_text("Brightness of the five developmental paths connecting the core to its possible outcomes.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_c, 0.2..=1.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_c).speed(0.01).range(0.2..=1.5));
                });

                ui.label("Developmental Pulse:")
                    .on_hover_text("Speed of the slow pulse and rotation through the five potential states.");
                ui.horizontal(|ui| {
                    let available = ui.available_width();
                    let slider_width = if available > 80.0 { available - 70.0 } else { 50.0 };
                    ui.style_mut().spacing.slider_width = slider_width;
                    ui.add(egui::Slider::new(&mut visuals.param_d, 0.0..=2.5).show_value(false));
                    ui.add(egui::DragValue::new(&mut visuals.param_d).speed(0.02).range(0.0..=2.5));
                });
            }

            // Save / Reset buttons at the bottom
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Save").on_hover_text("Save all cell type visuals to disk").clicked() {
                    context.editor_state.save_cell_type_visuals();
                }
                if ui.button("Reset Type").on_hover_text("Reset the selected cell type to its default visual settings").clicked() {
                    let ct = cell_types[selected_idx];
                    context.editor_state.cell_type_visuals[selected_idx] =
                        crate::cell::types::CellTypeVisuals::default_for_type(ct);
                }
                if ui.button("Reset All").on_hover_text("Reset all cell types to their default visual settings").clicked() {
                    for (i, ct) in cell_types.iter().enumerate() {
                        if i < context.editor_state.cell_type_visuals.len() {
                            context.editor_state.cell_type_visuals[i] =
                                crate::cell::types::CellTypeVisuals::default_for_type(*ct);
                        }
                    }
                    context.editor_state.cell_outline_width = 0.0;
                }
            });
        });
}

/// Render the Mode Graph panel for visualizing mode connections.
fn render_mode_graph(ui: &mut Ui, context: &mut PanelContext) {
    use crate::genome::node_graph::{ModeGraphViewer, ModeNode};
    use egui_snarl::ui::{SnarlStyle, SnarlWidget};
    use egui_snarl::Snarl;

    // Collect mode data for the viewer
    let mode_names: Vec<String> = context
        .genome
        .modes
        .iter()
        .map(|m| m.name.clone())
        .collect();
    let mode_colors: Vec<glam::Vec3> = context.genome.modes.iter().map(|m| m.color).collect();

    // Collect child mode numbers (we need mutable access)
    let mut child_a_modes: Vec<i32> = context
        .genome
        .modes
        .iter()
        .map(|m| m.child_a.mode_number)
        .collect();
    let mut child_b_modes: Vec<i32> = context
        .genome
        .modes
        .iter()
        .map(|m| m.child_b.mode_number)
        .collect();

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
        let selected_color = mode_colors
            .get(*selected)
            .copied()
            .unwrap_or(glam::Vec3::ONE);
        let bg_color = egui::Color32::from_rgb(
            (selected_color.x * 255.0) as u8,
            (selected_color.y * 255.0) as u8,
            (selected_color.z * 255.0) as u8,
        );
        let text_color = contrasting_text(selected_color);

        egui::ComboBox::from_id_salt("add_mode_combo")
            .selected_text(
                egui::RichText::new(format!("M{}", *selected + 1))
                    .color(text_color)
                    .background_color(bg_color),
            )
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
                        egui::RichText::new(format!("M{}", i + 1))
                            .color(item_text)
                            .background_color(item_bg),
                    );
                }
            });

        if ui.button("Add Node").clicked() {
            let idx = *selected;
            let name = mode_names.get(idx).cloned().unwrap_or_default();
            let color = mode_colors.get(idx).copied().unwrap_or(glam::Vec3::ONE);
            // Add node at next available position in column
            let pos = context.editor_state.mode_graph_state.next_node_position();
            let node_id = context
                .editor_state
                .mode_graph_state
                .snarl
                .insert_node(pos, ModeNode::new(idx, name, color));
            // Record which slot this node occupies
            context
                .editor_state
                .mode_graph_state
                .record_node_in_slot(node_id, pos);
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
    // Set slider width to fill the panel, leaving room for value + suffix labels.
    // Use a more conservative margin (80px) to account for suffixes like " units", "k", "x".
    let sw = (ui.available_width() - 80.0).max(60.0);
    ui.style_mut().spacing.slider_width = sw;

    // Only show world settings in GPU mode
    if context.current_mode != crate::ui::types::SimulationMode::Gpu {
        ui.label("World settings are only available in GPU mode.");
        return;
    }

    let world = &mut state.world_settings;

    // World Radius slider (top, reset-gated)
    let current_world_radius = context
        .scene_manager
        .gpu_scene()
        .map(|s| s.config.sphere_radius)
        .unwrap_or(world.world_radius);

    ui.label("World Radius:")
        .on_hover_text("Radius of the spherical world boundary in simulation units. Cells are confined within this sphere. ⚠ Changing this requires a scene reset to take effect");
    ui.add(egui::Slider::new(&mut world.world_radius, 50.0..=500.0).suffix(" units"));

    if (world.world_radius - current_world_radius).abs() > 0.5 {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚠ Scene reset required").color(palette().status_warn));
        });
    }

    ui.add_space(8.0);
    let current_capacity = context.gpu_capacity().unwrap_or(world.cell_capacity);
    let capacity_k = world.cell_capacity / 1000;

    ui.label("Cell Capacity:")
        .on_hover_text("Maximum number of cells the simulation can hold simultaneously. Higher values use more GPU memory. ⚠ Changing this requires a scene reset to take effect");
    let mut capacity_k_mut = capacity_k;
    if ui
        .add(egui::Slider::new(&mut capacity_k_mut, 10..=200).suffix("k"))
        .changed()
    {
        world.cell_capacity = capacity_k_mut * 1000;
    }

    // Show reset required message if capacity changed
    if world.cell_capacity != current_capacity {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚠ Scene reset required").color(palette().status_warn));
        });
    }

    ui.add_space(8.0);

    // Physics section
    ui.heading("Physics");
    ui.separator();

    // Gravity slider
    ui.label("Gravity:")
        .on_hover_text("Gravitational acceleration applied to all cells each frame. Positive = downward (or toward the chosen axis). Negative = upward/repulsive. 0 = weightless");
    ui.add(egui::Slider::new(&mut world.gravity, -50.0..=50.0).suffix(" m/s²"));

    // Gravity axis / mode
    ui.label("Gravity Axis:")
        .on_hover_text("Direction gravity pulls. X/Y/Z = along a world axis. Radial = pulls all cells toward the world center — the boundary sphere becomes the 'floor'");
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
        if ui
            .selectable_label(world.gravity_mode == 3, "Radial")
            .clicked()
        {
            world.gravity_mode = 3;
        }
    });
    if world.gravity_mode == 3 {
        ui.label(
            egui::RichText::new("Pulls fluid toward origin; world boundary is the shell").small(),
        );
    }

    ui.add_space(8.0);

    // Global velocity damping
    ui.label("Velocity Damping:")
        .on_hover_text("Multiplier applied to cell velocity each frame. 1.0 = no drag (cells coast forever). 0.8 = heavy drag (cells slow down quickly). Lower values create a more viscous environment");
    ui.add(egui::Slider::new(
        &mut world.acceleration_damping,
        0.8..=1.0,
    ));
    ui.label(
        egui::RichText::new(
            "Global drag on cell movement (lower = more drag, higher = less damping)",
        )
        .small(),
    );

    if state.show_advanced_options {
        ui.add_space(8.0);

        // Adhesion constraint solver iterations
        ui.label("Constraint Iterations:")
            .on_hover_text("Number of extra adhesion solver passes per physics step. More iterations = stiffer, more accurate joints but higher GPU cost. 0 = single pass (fast, slightly springy). 8–16 = rigid joints");
        ui.add(egui::Slider::new(&mut world.constraint_iterations, 0..=16));
        ui.label(
            egui::RichText::new(
                "Extra adhesion solver passes (higher = stiffer joints, more GPU cost)",
            )
            .small(),
        );

        ui.add_space(8.0);

        // Water viscosity
        ui.label("Water Viscosity:")
            .on_hover_text("Additional drag applied to cells that are inside water voxels. 0 = water has no effect on movement. 1 = maximum resistance — cells move very slowly through water");
        ui.add(egui::Slider::new(&mut world.water_viscosity, 0.0..=1.0));
        ui.label(
            egui::RichText::new(
                "Drag applied to cells moving through water (0 = off, higher = thicker fluid)",
            )
            .small(),
        );
    } // end advanced physics

    ui.add_space(12.0);

    // Biology section
    ui.heading("Biology");
    ui.separator();

    ui.checkbox(&mut world.solo_metabolism_enabled, "Solo cell metabolism penalty")
        .on_hover_text("Isolated cells (with few or no adhesion bonds) burn nutrients faster, making single-cell survival harder and favoring multicellular organisms");
    ui.label(egui::RichText::new("Cells with fewer adhesion connections burn nutrients faster, favoring multicellular organisms").small());

    if world.solo_metabolism_enabled {
        ui.add_space(4.0);
        ui.label("Metabolism Multiplier:")
            .on_hover_text("How much faster solo cells (0 connections) burn nutrients compared to connected cells. Gradient: 1 connection = partial penalty, 3+ connections = normal rate");
        ui.add(egui::Slider::new(
            &mut world.solo_metabolism_multiplier,
            1.5..=10.0,
        ));
        ui.label(egui::RichText::new("Solo cells (0 connections) drain at this rate. Gradient: 1 conn = partial, 3+ = normal").small());
    }

    if state.show_advanced_options {
        ui.add_space(12.0);

        // Mutation section
        ui.heading("Mutation");
        ui.separator();

        ui.label("Radiation Level:")
            .on_hover_text("Probability that each child cell mutates during division. 0 = no mutations. Uses a logarithmic scale for fine control at low values. Higher radiation drives faster evolution but may destabilize organisms");

        // Logarithmic slider: position 0.0 = exactly 0.0 (off)
        // position > 0.0 maps to [0.00001, 1.0] over 5 decades for fine low-end control
        const EPSILON: f32 = 0.0001;
        const DECADES: f64 = 5.0; // 10^-5 to 10^0

        let mut log_slider = if world.radiation_level <= 0.0 {
            0.0f32
        } else {
            // Map [1e-5, 1.0] -> [0.0, 1.0]
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
                }),
        );

        if response.changed() {
            world.radiation_level = if log_slider < EPSILON {
                0.0
            } else {
                10_f32
                    .powf(log_slider * DECADES as f32 - DECADES as f32)
                    .clamp(0.0, 1.0)
            };
        }

        ui.label(egui::RichText::new("Probability each child mutates during division (0 = off, logarithmic scale for fine control)").small());

        ui.add_space(4.0);
        ui.checkbox(&mut world.subtle_mutations, "Subtle mutations")
            .on_hover_text(
                "When checked, mutations make small color nudges instead of full re-rolls",
            );
    } // end advanced biology/mutation

    ui.add_space(12.0);
}

/// Render the Help panel showing context-specific controls and shortcuts.
fn render_help(ui: &mut Ui, _context: &mut PanelContext, _state: &mut GlobalUiState) {
    ui.heading("Help");
    ui.add_space(8.0);
    ui.label("Select a panel to see context-specific controls and shortcuts.");
}
