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
    /// Original style values for scaling
    original_spacing: Option<egui::style::Spacing>,
    original_text_styles: Option<std::collections::BTreeMap<egui::TextStyle, egui::FontId>>,
    /// Timer for auto-save functionality
    save_timer: std::time::Instant,
    /// Whether the UI state has changed since last save
    ui_state_dirty: bool,
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
        let state = GlobalUiState::load();

        Self {
            ctx,
            winit_state,
            renderer,
            state,
            viewport_rect: None,
            last_scale: 1.0,
            original_spacing: None,
            original_text_styles: None,
            save_timer: std::time::Instant::now(),
            ui_state_dirty: false,
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
        
        // NOTE: Don't clear viewport_rect here - it needs to persist for event handling
        // which happens before render(). The viewport rect from the previous frame
        // is still valid for determining if clicks are in the viewport area.
    }

    /// Apply UI scale to the egui context style.
    ///
    /// This method scales spacing, text sizes, and other UI elements based on
    /// the current ui_scale factor.
    fn apply_ui_scale(&mut self) {
        let scale = self.state.ui_scale;
        
        self.ctx.global_style_mut(|style| {
            // Store original values on first run
            if self.original_spacing.is_none() {
                self.original_spacing = Some(style.spacing.clone());
                self.original_text_styles = Some(style.text_styles.clone());
            }
            
            // Apply scale from original values (not multiplicatively)
            if let Some(ref original_spacing) = self.original_spacing {
                style.spacing.item_spacing = original_spacing.item_spacing * scale;
                style.spacing.button_padding = original_spacing.button_padding * scale;
                style.spacing.menu_margin = original_spacing.menu_margin * scale;
                style.spacing.indent = original_spacing.indent * scale;
                style.spacing.interact_size = original_spacing.interact_size * scale;
                style.spacing.slider_width = original_spacing.slider_width * scale;
                style.spacing.combo_width = original_spacing.combo_width * scale;
                style.spacing.text_edit_width = original_spacing.text_edit_width * scale;
                style.spacing.icon_width = original_spacing.icon_width * scale;
                style.spacing.icon_width_inner = original_spacing.icon_width_inner * scale;
                style.spacing.icon_spacing = original_spacing.icon_spacing * scale;
                style.spacing.tooltip_width = original_spacing.tooltip_width * scale;
                style.spacing.menu_width = original_spacing.menu_width * scale;
                style.spacing.combo_height = original_spacing.combo_height * scale;
            }
            
            // Scale text sizes from original values
            if let Some(ref original_text_styles) = self.original_text_styles {
                for (text_style, font_id) in style.text_styles.iter_mut() {
                    if let Some(original_font) = original_text_styles.get(text_style) {
                        font_id.size = original_font.size * scale;
                    }
                }
            }
        });
        
        log::debug!("Applied UI scale: {:.2}x", scale);
    }

    /// Mark UI state as dirty (needs saving).
    pub fn mark_ui_state_dirty(&mut self) {
        self.ui_state_dirty = true;
    }

    /// Auto-save UI state if needed.
    pub fn auto_save_ui_state(&mut self) {
        const AUTO_SAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
        
        if self.ui_state_dirty && self.save_timer.elapsed() >= AUTO_SAVE_INTERVAL {
            if let Err(e) = self.state.save() {
                log::warn!("Failed to auto-save UI state: {}", e);
            } else {
                log::debug!("Auto-saved UI state");
                self.ui_state_dirty = false;
                self.save_timer = std::time::Instant::now();
            }
        }
    }

    /// Save UI state immediately.
    pub fn save_ui_state(&mut self) {
        if let Err(e) = self.state.save() {
            log::warn!("Failed to save UI state: {}", e);
        } else {
            log::info!("Saved UI state");
            self.ui_state_dirty = false;
        }
    }

    /// End the egui frame and get the output.
    ///
    /// Call this after all UI rendering is complete.
    pub fn end_frame(
        &mut self, 
        dock_manager: &mut crate::ui::dock::DockManager,
        genome: &mut crate::genome::Genome,
        editor_state: &mut crate::ui::panel_context::GenomeEditorState,
        scene_manager: &crate::scene::SceneManager,
        camera: &mut crate::ui::camera::CameraController,
        scene_request: &mut crate::ui::panel_context::SceneModeRequest,
        performance: &crate::ui::performance::PerformanceMetrics,
    ) -> egui::FullOutput {
        // Apply UI scale only when it changes
        let scale_changed = (self.last_scale - self.state.ui_scale).abs() > 0.001;
        if scale_changed {
            self.apply_ui_scale();
            self.last_scale = self.state.ui_scale;
        }

        // Auto-save UI state periodically
        self.auto_save_ui_state();

        // Show menu bar at the top using the deprecated but still functional API
        // This is the correct pattern for top-level panels with just a Context
        let mut ui_state_copy = self.state.clone();
        
        #[allow(deprecated)]
        egui::Panel::top("menu_bar").show(&self.ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                use egui::containers::menu::{MenuButton, MenuConfig};
                use egui::PopupCloseBehavior;
                
                let config = MenuConfig::new()
                    .close_behavior(PopupCloseBehavior::CloseOnClickOutside);
                
                MenuButton::new("Windows")
                    .config(config)
                    .ui(ui, |ui| {
                        show_windows_menu(ui, &mut ui_state_copy, dock_manager);
                    });
                
                // Help button next to Windows dropdown
                if ui.button("❓ Help").clicked() {
                    let help_panel = crate::ui::panel::Panel::Help;
                    if is_panel_open(dock_manager.current_tree(), &help_panel) {
                        // If help is already open, close it
                        close_panel(dock_manager.current_tree_mut(), &help_panel);
                    } else {
                        // Open help panel as floating window
                        open_panel(dock_manager.current_tree_mut(), &help_panel);
                    }
                }
            });
        });
        
        // Show dock area in remaining space
        let mut style = egui_dock::Style::from_egui(self.ctx.global_style().as_ref());
        style.separator.extra = 75.0; // Reduce separator minimum constraint

        // Apply lock settings to hide tab bar height if locked
        if ui_state_copy.lock_tab_bar {
            style.tab_bar.height = 0.0;
        }

        // Create panel context for PanelTabViewer
        let current_mode = ui_state_copy.current_mode;
        
        // Show dock area (scoped to release borrows after)
        {
            let mut panel_context = crate::ui::panel_context::PanelContext::new(
                genome,
                editor_state,
                scene_manager,
                camera,
                scene_request,
                current_mode,
                performance,
            );
            
            let mut dock_area = egui_dock::DockArea::new(dock_manager.current_tree_mut())
                .style(style)
                .show_leaf_collapse_buttons(false)
                .show_leaf_close_all_buttons(false)
                .draggable_tabs(true)
                .window_bounds(self.ctx.content_rect());

            // Apply lock settings for tabs and close buttons
            if ui_state_copy.lock_tabs {
                dock_area = dock_area
                    .show_tab_name_on_hover(false)
                    .draggable_tabs(false);
            }

            if ui_state_copy.lock_close_buttons {
                dock_area = dock_area.show_close_buttons(false);
            }

            let mut tab_viewer = crate::ui::tab_viewer::PanelTabViewer::new(
                &mut ui_state_copy,
                &mut panel_context,
                &mut self.viewport_rect,
            );
            dock_area.show(&self.ctx, &mut tab_viewer);
        }
        
        // Handle mode graph panel toggle request
        if editor_state.toggle_mode_graph_panel {
            editor_state.toggle_mode_graph_panel = false;
            let panel = crate::ui::panel::Panel::ModeGraph;
            if let Some(location) = dock_manager.current_tree().find_tab(&panel) {
                // Panel is open - store its location and close it
                editor_state.mode_graph_panel_location = Some(location);
                dock_manager.current_tree_mut().remove_tab(location);
            } else {
                // Panel is closed - restore to original location if available
                if let Some((surface_index, node_index, tab_index)) = editor_state.mode_graph_panel_location {
                    // Try to restore to the original location
                    let dock_state = dock_manager.current_tree_mut();
                    
                    // Check if the surface still exists and has the node
                    let can_restore = dock_state.get_surface(surface_index)
                        .map(|surface| match surface {
                            egui_dock::Surface::Main(tree) | egui_dock::Surface::Window(tree, _) => {
                                node_index.0 < tree.len() && tree[node_index].is_leaf()
                            }
                            egui_dock::Surface::Empty => false,
                        })
                        .unwrap_or(false);
                    
                    if can_restore {
                        // Restore to the original location
                        dock_state[surface_index][node_index].insert_tab(tab_index, panel);
                        editor_state.mode_graph_panel_location = None;
                    } else {
                        // Original location no longer valid, create as floating window
                        let _surface_index = dock_state.add_window(vec![panel]);
                        editor_state.mode_graph_panel_location = None;
                    }
                } else {
                    // No stored location, create as floating window
                    let _surface_index = dock_manager.current_tree_mut().add_window(vec![panel]);
                }
            }
        }
        
        // Render radial menu overlay (GPU mode only)
        // Now editor_state is no longer borrowed by panel_context
        if current_mode == crate::ui::types::SimulationMode::Gpu {
            crate::ui::radial_menu::show_radial_menu(&self.ctx, &mut editor_state.radial_menu);
            crate::ui::radial_menu::show_tool_cursor(&self.ctx, &editor_state.radial_menu);
        }
        
        // Apply any changes back to the original state
        let state_changed = self.state != ui_state_copy;
        self.state = ui_state_copy;
        
        // Mark UI state as dirty if it changed
        if state_changed {
            self.mark_ui_state_dirty();
        }

        // Handle global click to clear text selection
        if self.ctx.input(|i| i.pointer.any_click()) {
            // Clear text selection on any click
            self.ctx.memory_mut(|mem| {
                mem.request_focus(egui::Id::NULL);
            });
            
            // Clear text selection in labels
            let plugin = self.ctx.plugin::<egui::text_selection::LabelSelectionState>();
            plugin.lock().clear_selection();
        }
        
        // Also clear text selection if Escape is pressed
        if self.ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.ctx.memory_mut(|mem| {
                mem.request_focus(egui::Id::NULL);
            });
            
            let plugin = self.ctx.plugin::<egui::text_selection::LabelSelectionState>();
            plugin.lock().clear_selection();
        }

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

/// Show the Windows menu for panel visibility toggles.
fn show_windows_menu(ui: &mut egui::Ui, state: &mut GlobalUiState, dock_manager: &mut crate::ui::dock::DockManager) {
    use crate::ui::panel::Panel;

    // UI Scale radio buttons
    ui.label("UI Scale:");
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            if ui.radio(state.ui_scale == 0.5, "0.5x").clicked() {
                state.ui_scale = 0.5;
            }
            if ui.radio(state.ui_scale == 1.0, "1.0x").clicked() {
                state.ui_scale = 1.0;
            }
            if ui.radio(state.ui_scale == 1.5, "1.5x").clicked() {
                state.ui_scale = 1.5;
            }
            if ui.radio(state.ui_scale == 3.0, "3.0x").clicked() {
                state.ui_scale = 3.0;
            }
        });
        ui.vertical(|ui| {
            if ui.radio(state.ui_scale == 0.75, "0.75x").clicked() {
                state.ui_scale = 0.75;
            }
            if ui.radio(state.ui_scale == 1.25, "1.25x").clicked() {
                state.ui_scale = 1.25;
            }
            if ui.radio(state.ui_scale == 2.0, "2.0x").clicked() {
                state.ui_scale = 2.0;
            }
            if ui.radio(state.ui_scale == 4.0, "4.0x").clicked() {
                state.ui_scale = 4.0;
            }
        });
    });

    ui.separator();

    // List of genome editor panels that can be toggled (only show in Preview mode)
    let genome_editor_panels = [
        Panel::Modes,
        Panel::ModeGraph,
        Panel::NameTypeEditor,
        Panel::AdhesionSettings,
        Panel::ParentSettings,
        Panel::CircleSliders,
        Panel::QuaternionBall,
        Panel::TimeSlider,
        Panel::CellTypeVisuals,
    ];

    // Only show genome editor windows in Preview mode
    if state.current_mode == crate::ui::types::SimulationMode::Preview {
        ui.label("Genome Editor:");
        for panel in &genome_editor_panels {
            let is_open = is_panel_open(dock_manager.current_tree(), panel);
            let panel_name = format!("{:?}", panel);
            let is_locked = state.is_panel_locked(&panel_name);

            ui.horizontal(|ui| {
                // Window toggle button
                if ui.selectable_label(is_open, format!("  {}", panel.display_name())).clicked() {
                    if is_open {
                        close_panel(dock_manager.current_tree_mut(), panel);
                    } else {
                        open_panel(dock_manager.current_tree_mut(), panel);
                    }
                }
                
                // Lock/Unlock button
                let lock_icon = if is_locked { "🔒" } else { "🔓" };
                if ui.small_button(lock_icon).clicked() {
                    state.set_panel_locked(&panel_name, !is_locked);
                }
            });
        }

        ui.separator();
    }

    // Layout Panels
    ui.label("Layout Panels:");
    
    let layout_panels = [
        Panel::LeftPanel,
        Panel::RightPanel,
        Panel::BottomPanel,
        Panel::Viewport,
        Panel::GizmoSettings,
    ];
    
    for panel in &layout_panels {
        let is_open = is_panel_open(dock_manager.current_tree(), panel);
        let panel_name = format!("{:?}", panel);
        let is_locked = state.is_panel_locked(&panel_name);

        ui.horizontal(|ui| {
            // Window toggle button
            if ui.selectable_label(is_open, format!("  {}", panel.display_name())).clicked() {
                if is_open {
                    close_panel(dock_manager.current_tree_mut(), panel);
                } else {
                    open_panel(dock_manager.current_tree_mut(), panel);
                }
            }
            
            // Lock/Unlock button
            let lock_icon = if is_locked { "🔒" } else { "🔓" };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&panel_name, !is_locked);
            }
        });
    }

    ui.separator();

    // Other Windows
    ui.label("Other Windows:");

    // Scene Manager
    let scene_manager_open = is_panel_open(dock_manager.current_tree(), &Panel::SceneManager);
    let scene_manager_name = format!("{:?}", Panel::SceneManager);
    let scene_manager_locked = state.is_panel_locked(&scene_manager_name);
    
    ui.horizontal(|ui| {
        if ui.selectable_label(scene_manager_open, "  Scene Manager").clicked() {
            if scene_manager_open {
                close_panel(dock_manager.current_tree_mut(), &Panel::SceneManager);
            } else {
                open_panel(dock_manager.current_tree_mut(), &Panel::SceneManager);
            }
        }
        
        let lock_icon = if scene_manager_locked { "🔒" } else { "🔓" };
        if ui.small_button(lock_icon).clicked() {
            state.set_panel_locked(&scene_manager_name, !scene_manager_locked);
        }
    });

    // Performance Monitor
    let perf_open = is_panel_open(dock_manager.current_tree(), &Panel::PerformanceMonitor);
    let perf_name = format!("{:?}", Panel::PerformanceMonitor);
    let perf_locked = state.is_panel_locked(&perf_name);
    
    ui.horizontal(|ui| {
        if ui.selectable_label(perf_open, "  Performance").clicked() {
            if perf_open {
                close_panel(dock_manager.current_tree_mut(), &Panel::PerformanceMonitor);
            } else {
                open_panel(dock_manager.current_tree_mut(), &Panel::PerformanceMonitor);
            }
        }
        
        let lock_icon = if perf_locked { "🔒" } else { "🔓" };
        if ui.small_button(lock_icon).clicked() {
            state.set_panel_locked(&perf_name, !perf_locked);
        }
    });

    // Cell Inspector
    let inspector_open = is_panel_open(dock_manager.current_tree(), &Panel::CellInspector);
    let inspector_name = format!("{:?}", Panel::CellInspector);
    let inspector_locked = state.is_panel_locked(&inspector_name);
    
    ui.horizontal(|ui| {
        if ui.selectable_label(inspector_open, "  Cell Inspector").clicked() {
            if inspector_open {
                close_panel(dock_manager.current_tree_mut(), &Panel::CellInspector);
            } else {
                open_panel(dock_manager.current_tree_mut(), &Panel::CellInspector);
            }
        }
        
        let lock_icon = if inspector_locked { "🔒" } else { "🔓" };
        if ui.small_button(lock_icon).clicked() {
            state.set_panel_locked(&inspector_name, !inspector_locked);
        }
    });

    ui.separator();

    // Layout Management
    ui.label("Layout Management:");
    
    // Save current layout as default button
    if ui.button("💾 Save Current as Default").clicked() {
        match dock_manager.save_current_as_default() {
            Ok(()) => {
                log::info!("Successfully saved current layout as default for new players");
            }
            Err(e) => {
                log::error!("Failed to save current layout as default: {}", e);
            }
        }
    }
    
    // Reset to default layout button
    if ui.button("🔄 Reset to Default").clicked() {
        dock_manager.reset_current_to_default();
    }
    
    ui.separator();
    
    // Layout file information
    ui.label("Layout Files:");
    ui.small("Current layout files:");
    
    let current_mode = dock_manager.current_mode();
    let layout_file = format!("dock_state_{}.ron", current_mode.dock_file_suffix());
    let default_file = format!("default_dock_state_{}.ron", current_mode.dock_file_suffix());
    
    ui.horizontal(|ui| {
        ui.small("• Active:");
        ui.small(&layout_file);
    });
    
    ui.horizontal(|ui| {
        ui.small("• Default:");
        if std::path::Path::new(&default_file).exists() {
            ui.small(&default_file);
        } else {
            ui.small("(using hardcoded default)");
        }
    });
}

/// Check if a panel is currently open in the dock tree
fn is_panel_open(tree: &egui_dock::DockState<crate::ui::panel::Panel>, panel: &crate::ui::panel::Panel) -> bool {
    tree.iter_all_tabs().any(|(_, tab)| tab == panel)
}

/// Close a panel in the dock tree
fn close_panel(tree: &mut egui_dock::DockState<crate::ui::panel::Panel>, panel: &crate::ui::panel::Panel) {
    // Simple approach: just remove all instances of this panel
    tree.retain_tabs(|tab| tab != panel);
}

/// Open a panel in the dock tree as a floating window
fn open_panel(tree: &mut egui_dock::DockState<crate::ui::panel::Panel>, panel: &crate::ui::panel::Panel) {
    // Create a new floating window with the panel
    // Ensure we always pass a non-empty vector to prevent crashes
    let _surface_index = tree.add_window(vec![*panel]);
}
