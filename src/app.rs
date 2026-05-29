//! # Application Core - wgpu Setup and Event Loop
//! 
//! This module contains the main [`App`] struct that coordinates the entire Bio-Spheres application.
//! It handles wgpu initialization, window events, scene management, and the render loop.
//! 
//! ## Architecture Overview
//! 
//! The [`App`] struct serves as the central coordinator that:
//! - Manages wgpu resources (device, queue, surface)
//! - Handles window events and input routing
//! - Coordinates between simulation scenes and UI
//! - Orchestrates the render pipeline (3D scene → egui UI → present)
//! 
//! ## Event Flow
//! 
//! ```text
//! Window Event → egui Input Check → Scene Input → Camera Update → Render
//! ```
//! 
//! 1. **Input Routing**: Events are first offered to egui, then to scene/camera if not consumed
//! 2. **Scene Updates**: Physics simulation and camera movement are updated each frame
//! 3. **Rendering**: 3D scene renders first, then egui UI is composited on top
//! 
//! ## Scene Management
//! 
//! The app manages two simulation modes through [`SceneManager`]:
//! - **Preview Mode**: CPU physics for genome editing and small simulations
//! - **GPU Mode**: GPU compute for large-scale simulations with interactive tools
//! 
//! ## Tool System (GPU Mode)
//! 
//! In GPU mode, the app provides interactive tools via a radial menu:
//! - **Insert**: Add cells from the current genome
//! - **Remove**: Delete cells by clicking
//! - **Boost**: Give cells maximum nutrients for immediate division
//! - **Inspect**: Select cells for detailed information
//! - **Drag**: Move cells in 3D space
//! 
//! ## Performance Monitoring
//! 
//! The app tracks:
//! - Frame rate and render times
//! - Culling statistics (frustum and occlusion)
//! - Cell counts and simulation metrics

use crate::scene::{MainMenuScene, SceneManager};
use crate::ui::{DockManager, PerformanceMetrics, UiSystem};
use egui::TextureId;
use egui_wgpu::ScreenDescriptor;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// High-level application phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppPhase {
    /// Showing the main menu (before any simulation is started).
    MainMenu,
    /// Inside the simulation (Preview or GPU mode).
    InGame,
}

/// Button action returned by the main-menu egui pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuAction {
    None,
    Play,
    GenomeEditor,
    Settings,
    Credits,
    Exit,
}

/// Action deferred until after the current frame is presented to the screen.
///
/// Save/load operations involve blocking GPU readbacks or large data uploads.
/// Running them before `output.present()` means the "Saving…" / "Loading…"
/// popup never appears on screen.  By deferring to post-present we guarantee
/// the user sees the overlay for at least one frame.
enum DeferredAction {
    SaveSphere,
    LoadSphere(std::path::PathBuf),
    TakeScreenshot {
        staging: wgpu::Buffer,
        width: u32,
        height: u32,
        padded_bytes_per_row: u32,
        unpadded_bytes_per_row: u32,
        format: wgpu::TextureFormat,
    },
    CaptureGif,
}

pub struct App {
    window: Arc<Window>,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scene_manager: SceneManager,
    dock_manager: DockManager,
    ui: UiSystem,
    last_render_time: std::time::Instant,
    frame_count: u32,
    fps_timer: std::time::Instant,
    /// Persistent genome editor state
    editor_state: crate::ui::panel_context::GenomeEditorState,
    /// Current mouse position for tool interactions
    mouse_position: (f32, f32),
    /// Current keyboard modifier state (Ctrl, Shift, Alt, etc.)
    keyboard_modifiers: winit::event::Modifiers,
    /// Whether a Ctrl+drag selection sweep is currently active in the preview
    ctrl_drag_selecting: bool,
    /// Current working genome (shared between preview and GPU scenes)
    working_genome: crate::genome::Genome,
    /// Performance metrics tracker
    performance: PerformanceMetrics,
    /// Next frame time for 60fps limiting
    next_frame_time: std::time::Instant,
    /// Active test signal emissions (toggleable)
    test_signal_emissions: Vec<crate::simulation::signal_system::SignalEmission>,
    /// Flag to trigger resimulation when test signals change
    test_signals_changed: bool,
    /// Deferred post-present action (save/load sphere — runs after frame is on screen)
    deferred_action: Option<DeferredAction>,
    /// Timestamp of the last left-click for double-click detection
    last_left_click_time: Option<std::time::Instant>,
    /// Screen position of the last left-click for double-click proximity check
    last_left_click_pos: (f32, f32),
    // IMPORTANT: surface must be declared before device so it drops first.
    // Rust drops fields in declaration order; wgpu/Vulkan requires the surface
    // to be destroyed before the device, otherwise the Vulkan validation layer
    // panics with "Trying to destroy a SurfaceAcquireSemaphores that is still
    // in use by a SurfaceTexture".
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    /// Current high-level application phase (main menu vs in-game).
    app_phase: AppPhase,
    /// Main menu scene (two live genome previews + egui overlay).
    main_menu_scene: Option<MainMenuScene>,
}

impl App {
    pub fn new(
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        scene_manager: SceneManager,
        dock_manager: DockManager,
        mut ui: UiSystem,
    ) -> Self {
        // Build the main menu scene before moving `ui` into the struct so we
        // can access `ui.renderer` mutably without fighting the borrow checker.
        let main_menu_scene =
            MainMenuScene::new(&device, &queue, &config, &mut ui.renderer);

        Self {
            window,
            queue,
            config,
            scene_manager,
            dock_manager,
            ui,
            last_render_time: std::time::Instant::now(),
            frame_count: 0,
            fps_timer: std::time::Instant::now(),
            editor_state: crate::ui::panel_context::GenomeEditorState::new(),
            mouse_position: (0.0, 0.0),
            keyboard_modifiers: winit::event::Modifiers::default(),
            ctrl_drag_selecting: false,
            working_genome: crate::genome::Genome::new_with_random_colors(),
            performance: PerformanceMetrics::new(),
            next_frame_time: std::time::Instant::now(),
            test_signal_emissions: Vec::new(),
            test_signals_changed: false,
            deferred_action: None,
            last_left_click_time: None,
            last_left_click_pos: (0.0, 0.0),
            device,
            surface,
            app_phase: AppPhase::MainMenu,
            main_menu_scene: Some(main_menu_scene),
        }
    }
    
    pub fn window(&self) -> &Window {
        &self.window
    }
    
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        // First, let egui handle the event
        let _egui_response = self.ui.handle_event(&self.window, event);
        
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                // Save dock layouts before exit
                self.dock_manager.save_all();
                // Save UI state before exit
                self.ui.save_ui_state();
                // Save cell type visuals before exit
                self.editor_state.save_cell_type_visuals();
                // Save fluid and light settings before exit
                self.editor_state.save_fluid_settings();
                self.editor_state.save_fluid_render_settings();
                self.editor_state.save_light_settings();
                
                // Wait for GPU to finish all work before surface cleanup
                // This prevents SurfaceAcquireSemaphores panic on exit
                log::info!("Waiting for GPU to finish before exit...");
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                
                return false;
            }
            WindowEvent::Resized(physical_size) => {
                // Only configure surface if both dimensions are non-zero
                if physical_size.width > 0 && physical_size.height > 0 {
                    self.config.width = physical_size.width;
                    self.config.height = physical_size.height;
                    self.surface.configure(&self.device, &self.config);
                    self.scene_manager.resize(&self.device, physical_size.width, physical_size.height);
                    if let Some(menu) = &mut self.main_menu_scene {
                        menu.resize(&self.device, &mut self.ui.renderer, physical_size.width, physical_size.height);
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                // Handle cell click in Preview mode to select mode
                if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Preview
                    && *button == MouseButton::Left
                    && *state == ElementState::Pressed
                    && !self.ui.wants_pointer_input()
                {
                    let ctrl_held = self.keyboard_modifiers.state().control_key();
                    // Start a Ctrl+drag sweep if Ctrl is held
                    if ctrl_held {
                        self.ctrl_drag_selecting = true;
                    }

                    if let Some(preview_scene) = self.scene_manager.preview_scene_mut() {
                        let (mx, my) = self.mouse_position;
                        let w = self.config.width as f32;
                        let h = self.config.height as f32;
                        let aspect = w / h;

                        // Build camera ray from screen position
                        let cam_pos = preview_scene.camera.position();
                        let cam_rot = preview_scene.camera.rotation;
                        let fov_y = 45.0_f32.to_radians();
                        let tan_half_fov = (fov_y / 2.0).tan();

                        // NDC [-1, 1]
                        let ndc_x = (mx / w) * 2.0 - 1.0;
                        let ndc_y = 1.0 - (my / h) * 2.0;

                        let ray_dir_cam = glam::Vec3::new(
                            ndc_x * aspect * tan_half_fov,
                            ndc_y * tan_half_fov,
                            -1.0,
                        ).normalize();
                        let ray_dir = cam_rot * ray_dir_cam;

                        // Ray-sphere intersection against all cells
                        let cell_count = preview_scene.state.display_state.cell_count;
                        let mut best_t = f32::MAX;
                        let mut hit_mode: Option<usize> = None;

                        for i in 0..cell_count {
                            let center = preview_scene.state.display_state.positions[i];
                            let radius = preview_scene.state.display_state.radii[i];
                            let oc = cam_pos - center;
                            let b = oc.dot(ray_dir);
                            let c = oc.dot(oc) - radius * radius;
                            let disc = b * b - c;
                            if disc >= 0.0 {
                                let t = -b - disc.sqrt();
                                if t > 0.001 && t < best_t {
                                    best_t = t;
                                    hit_mode = Some(preview_scene.state.display_state.mode_indices[i]);
                                }
                            }
                        }

                        if let Some(mode_idx) = hit_mode {
                            let ctrl_held = self.keyboard_modifiers.state().control_key();
                            let shift_held = self.keyboard_modifiers.state().shift_key();

                            // Ensure selected_mode_indices is consistent before modifying
                            if self.editor_state.selected_mode_indices.is_empty() {
                                self.editor_state.selected_mode_indices = vec![self.editor_state.selected_mode_index];
                            }

                            if ctrl_held {
                                // Ctrl+click: toggle this mode in the multi-selection
                                if self.editor_state.selected_mode_indices.contains(&mode_idx) {
                                    // Don't deselect if it's the only one selected
                                    if self.editor_state.selected_mode_indices.len() > 1 {
                                        self.editor_state.selected_mode_indices.retain(|&i| i != mode_idx);
                                        // If we removed the primary, promote the first remaining
                                        if self.editor_state.selected_mode_index == mode_idx {
                                            let new_primary = self.editor_state.selected_mode_indices[0];
                                            self.editor_state.selected_mode_index = new_primary;
                                            if let Some(mode) = self.working_genome.modes.get(new_primary) {
                                                self.editor_state.child_a_orientation = mode.child_a.orientation;
                                                self.editor_state.child_b_orientation = mode.child_b.orientation;
                                            }
                                        }
                                    }
                                } else {
                                    // Add to selection without changing the primary
                                    self.editor_state.selected_mode_indices.push(mode_idx);
                                }
                                log::info!("Preview Ctrl+click: multi-selection now {:?}", self.editor_state.selected_mode_indices);
                            } else if shift_held {
                                // Shift+click: range select from current primary to clicked mode
                                let anchor = self.editor_state.selected_mode_index;
                                let lo = anchor.min(mode_idx);
                                let hi = anchor.max(mode_idx);
                                for i in lo..=hi {
                                    if !self.editor_state.selected_mode_indices.contains(&i) {
                                        self.editor_state.selected_mode_indices.push(i);
                                    }
                                }
                                log::info!("Preview Shift+click: range selection {:?}", self.editor_state.selected_mode_indices);
                            } else {
                                // Plain click: single selection
                                self.editor_state.selected_mode_index = mode_idx;
                                self.editor_state.selected_mode_indices = vec![mode_idx];
                                // Sync quaternion ball orientations from the selected mode's genome
                                // data — same sync that happens when clicking a mode button directly.
                                if let Some(mode) = self.working_genome.modes.get(mode_idx) {
                                    self.editor_state.child_a_orientation = mode.child_a.orientation;
                                    self.editor_state.child_b_orientation = mode.child_b.orientation;
                                }
                                log::info!("Preview cell click: selected mode {}", mode_idx);
                            }
                        }
                    }
                    self.window.request_redraw();
                }

                // Clear Ctrl+drag sweep on left button release
                if *button == MouseButton::Left && *state == ElementState::Released {
                    self.ctrl_drag_selecting = false;
                }

                // Handle right-click in Preview mode for cell context menu
                if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Preview
                    && *button == MouseButton::Right
                    && *state == ElementState::Pressed
                    && !self.ui.wants_pointer_input()
                {
                    if let Some(preview_scene) = self.scene_manager.preview_scene_mut() {
                        let (mx, my) = self.mouse_position;
                        let w = self.config.width as f32;
                        let h = self.config.height as f32;
                        let aspect = w / h;

                        let cam_pos = preview_scene.camera.position();
                        let cam_rot = preview_scene.camera.rotation;
                        let fov_y = 45.0_f32.to_radians();
                        let tan_half_fov = (fov_y / 2.0).tan();

                        let ndc_x = (mx / w) * 2.0 - 1.0;
                        let ndc_y = 1.0 - (my / h) * 2.0;

                        let ray_dir_cam = glam::Vec3::new(
                            ndc_x * aspect * tan_half_fov,
                            ndc_y * tan_half_fov,
                            -1.0,
                        ).normalize();
                        let ray_dir = cam_rot * ray_dir_cam;

                        let cell_count = preview_scene.state.display_state.cell_count;
                        let mut best_t = f32::MAX;
                        let mut hit_cell: Option<usize> = None;

                        for i in 0..cell_count {
                            let center = preview_scene.state.display_state.positions[i];
                            let radius = preview_scene.state.display_state.radii[i];
                            let oc = cam_pos - center;
                            let b = oc.dot(ray_dir);
                            let c = oc.dot(oc) - radius * radius;
                            let disc = b * b - c;
                            if disc >= 0.0 {
                                let t = -b - disc.sqrt();
                                if t > 0.001 && t < best_t {
                                    best_t = t;
                                    hit_cell = Some(i);
                                }
                            }
                        }

                        if let Some(cell_idx) = hit_cell {
                            preview_scene.context_menu_cell = Some(cell_idx);
                            // Convert physical pixels to egui logical points
                            let scale = self.window.scale_factor() as f32;
                            preview_scene.context_menu_screen_pos = (mx / scale, my / scale);
                            preview_scene.context_menu_open_time = std::time::Instant::now();
                            log::info!("Right-click on cell {} at screen ({}, {})", cell_idx, mx, my);
                        } else {
                            preview_scene.context_menu_cell = None;
                        }
                    }
                    self.window.request_redraw();
                }

                // Handle radial menu click (GPU mode only)
                if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
                    // Read menu state before mutable borrow
                    let menu_visible = self.editor_state.radial_menu.visible;
                    let active_tool = self.editor_state.radial_menu.active_tool;

                    // Double-click detection: when no tool is active and the menu is closed,
                    // a double-click locks the camera to the organism under the cursor.
                    // A subsequent double-click on empty space releases the follow.
                    if *button == MouseButton::Left
                        && *state == ElementState::Pressed
                        && !menu_visible
                        && active_tool == crate::ui::radial_menu::RadialTool::None
                        && !self.ui.wants_pointer_input()
                    {
                        let now = std::time::Instant::now();
                        let (lx, ly) = self.last_left_click_pos;
                        let (mx, my) = self.mouse_position;
                        let dx = mx - lx;
                        let dy = my - ly;
                        let close_enough = dx * dx + dy * dy < 20.0 * 20.0; // within 20px
                        let fast_enough = self.last_left_click_time
                            .map(|t| now.duration_since(t).as_millis() < 400)
                            .unwrap_or(false);

                        if fast_enough && close_enough {
                            // Double-click detected.
                            if self.scene_manager.is_following_organism() {
                                // Second double-click releases the follow.
                                self.scene_manager.clear_organism_follow();
                                log::info!("Organism follow released by double-click");
                            } else {
                                // First double-click: start following the organism under cursor.
                                self.scene_manager.start_organism_follow_query(mx, my);
                                log::info!("Organism follow query started at ({}, {})", mx, my);
                            }
                            // Reset so a third click doesn't immediately re-trigger.
                            self.last_left_click_time = None;
                            self.window.request_redraw();
                        } else {
                            // Record this click as the potential first of a double-click.
                            self.last_left_click_time = Some(now);
                            self.last_left_click_pos = (mx, my);
                        }
                    }
                    
                    if menu_visible && *button == MouseButton::Left && *state == ElementState::Pressed {
                        // Click while menu is open selects the hovered tool
                        self.editor_state.radial_menu.close(true);
                        // Hide cursor if a tool is now active
                        let new_active_tool = self.editor_state.radial_menu.active_tool;
                        let hide_cursor = new_active_tool != crate::ui::radial_menu::RadialTool::None;
                        self.window.set_cursor_visible(!hide_cursor);
                        self.window.request_redraw();
                        return true;
                    }
                    
                    // Handle Insert tool click
                    if !menu_visible 
                        && active_tool == crate::ui::radial_menu::RadialTool::Insert
                        && *button == MouseButton::Left 
                        && *state == ElementState::Pressed
                    {
                        if !self.ui.wants_pointer_input() {
                            let world_pos = self.scene_manager.screen_to_world(
                                self.mouse_position.0,
                                self.mouse_position.1,
                            );
                            // Queue cell insertion to be processed during render phase
                            if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                                gpu_scene.queue_cell_insertion(world_pos, self.working_genome.clone());
                            }
                            self.window.request_redraw();
                            return true;
                        }
                    }
                    
                    // Handle Remove tool click
                    if !menu_visible 
                        && active_tool == crate::ui::radial_menu::RadialTool::Remove
                        && *button == MouseButton::Left 
                        && *state == ElementState::Pressed
                        && !self.ui.wants_pointer_input()
                    {
                        // Initiate GPU spatial query for cell removal via scene manager
                        self.scene_manager.start_remove_tool_query(self.mouse_position.0, self.mouse_position.1);
                        self.window.request_redraw();
                        return true;
                    }
                    
                    // Handle Boost tool click - give cell maximum nutrients (mass = split_mass)
                    if !menu_visible 
                        && active_tool == crate::ui::radial_menu::RadialTool::Boost
                        && *button == MouseButton::Left 
                        && *state == ElementState::Pressed
                        && !self.ui.wants_pointer_input()
                    {
                        // Initiate GPU spatial query for cell boost via scene manager
                        self.scene_manager.start_boost_tool_query(self.mouse_position.0, self.mouse_position.1);
                        self.window.request_redraw();
                        return true;
                    }
                    
                    // Handle Inspect tool click - select cell for inspection
                    if !menu_visible 
                        && active_tool == crate::ui::radial_menu::RadialTool::Inspect
                        && *button == MouseButton::Left 
                        && *state == ElementState::Pressed
                        && !self.ui.wants_pointer_input()
                    {
                        // Initiate GPU spatial query for cell selection via scene manager
                        self.scene_manager.start_cell_selection_query(self.mouse_position.0, self.mouse_position.1);
                        self.window.request_redraw();
                        return true;
                    }
                    
                    // Handle Drag tool - mouse press starts drag
                    if active_tool == crate::ui::radial_menu::RadialTool::Drag
                        && *button == MouseButton::Left 
                        && *state == ElementState::Pressed
                    {
                        if !menu_visible && !self.ui.wants_pointer_input() {
                            // Start GPU spatial query for drag tool via scene manager
                            self.scene_manager.start_drag_selection_query(self.mouse_position.0, self.mouse_position.1);
                        }
                        self.window.request_redraw();
                        return true;
                    }
                    
                    // Handle Drag tool - mouse release ends drag
                    if active_tool == crate::ui::radial_menu::RadialTool::Drag
                        && *button == MouseButton::Left 
                        && *state == ElementState::Released
                        && self.editor_state.radial_menu.dragging_cell.is_some()
                    {
                        log::info!("Stopped dragging cell {:?}", self.editor_state.radial_menu.dragging_cell);
                        self.scene_manager.clear_dragged_cell();
                        self.editor_state.radial_menu.stop_dragging();
                        self.window.request_redraw();
                        return true;
                    }
                }
                
                // Only pass to camera if egui doesn't want the input and not dragging
                if !self.ui.wants_pointer_input() && self.editor_state.radial_menu.dragging_cell.is_none() {
                    self.scene_manager.active_scene_mut().camera_mut().handle_mouse_button(*button, *state);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.keyboard_modifiers = *modifiers;
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Track mouse position for tool interactions
                self.mouse_position = (position.x as f32, position.y as f32);

                // Ctrl+drag: continuously add hovered cells' modes to the selection
                if self.ctrl_drag_selecting
                    && self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Preview
                    && !self.ui.wants_pointer_input()
                {
                    if let Some(preview_scene) = self.scene_manager.preview_scene_mut() {
                        let mx = position.x as f32;
                        let my = position.y as f32;
                        let w = self.config.width as f32;
                        let h = self.config.height as f32;
                        let aspect = w / h;

                        let cam_pos = preview_scene.camera.position();
                        let cam_rot = preview_scene.camera.rotation;
                        let fov_y = 45.0_f32.to_radians();
                        let tan_half_fov = (fov_y / 2.0).tan();

                        let ndc_x = (mx / w) * 2.0 - 1.0;
                        let ndc_y = 1.0 - (my / h) * 2.0;

                        let ray_dir_cam = glam::Vec3::new(
                            ndc_x * aspect * tan_half_fov,
                            ndc_y * tan_half_fov,
                            -1.0,
                        ).normalize();
                        let ray_dir = cam_rot * ray_dir_cam;

                        let cell_count = preview_scene.state.display_state.cell_count;
                        let mut best_t = f32::MAX;
                        let mut hit_mode: Option<usize> = None;

                        for i in 0..cell_count {
                            let center = preview_scene.state.display_state.positions[i];
                            let radius = preview_scene.state.display_state.radii[i];
                            let oc = cam_pos - center;
                            let b = oc.dot(ray_dir);
                            let c = oc.dot(oc) - radius * radius;
                            let disc = b * b - c;
                            if disc >= 0.0 {
                                let t = -b - disc.sqrt();
                                if t > 0.001 && t < best_t {
                                    best_t = t;
                                    hit_mode = Some(preview_scene.state.display_state.mode_indices[i]);
                                }
                            }
                        }

                        if let Some(mode_idx) = hit_mode {
                            if self.editor_state.selected_mode_indices.is_empty() {
                                self.editor_state.selected_mode_indices = vec![self.editor_state.selected_mode_index];
                            }
                            if !self.editor_state.selected_mode_indices.contains(&mode_idx) {
                                self.editor_state.selected_mode_indices.push(mode_idx);
                                self.window.request_redraw();
                            }
                        }
                    }
                }

                // Update radial menu hover state (GPU mode only)
                if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
                    let menu = &mut self.editor_state.radial_menu;
                    if menu.visible {
                        menu.update_hover(egui::Pos2::new(position.x as f32, position.y as f32));
                        self.window.request_redraw();
                    }
                    
                    // Handle Drag tool - update cell position while dragging
                    if let Some(cell_idx) = self.editor_state.radial_menu.dragging_cell {
                        // Move cell to new position at the same distance from camera using GPU operations
                        let new_pos = self.scene_manager.screen_to_world_at_distance(
                            position.x as f32,
                            position.y as f32,
                            self.editor_state.drag_distance,
                        );
                        
                        // Use GPU position update via scene manager
                        self.scene_manager.update_cell_position_gpu(cell_idx as u32, new_pos);
                        self.window.request_redraw();
                    }
                }
                
                // Only pass to camera if egui doesn't want the input and not dragging
                if !self.ui.wants_pointer_input() && self.editor_state.radial_menu.dragging_cell.is_none() {
                    self.scene_manager.active_scene_mut().camera_mut().handle_mouse_move(*position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Only pass to camera if egui doesn't want the input
                if !self.ui.wants_scroll_input() {
                    self.scene_manager.active_scene_mut().camera_mut().handle_scroll(*delta);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Handle radial menu Alt key (GPU mode only)
                if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
                    use winit::keyboard::{KeyCode, PhysicalKey};
                    
                    if let PhysicalKey::Code(KeyCode::AltLeft) | PhysicalKey::Code(KeyCode::AltRight) = event.physical_key {
                        let menu = &mut self.editor_state.radial_menu;
                        
                        if event.state == ElementState::Pressed && !menu.alt_held {
                            // Alt pressed - open menu at current cursor position
                            if let Some(pos) = self.ui.ctx.pointer_hover_pos() {
                                menu.open(pos);
                            } else {
                                // Fallback to center of window
                                let size = self.window.inner_size();
                                menu.open(egui::Pos2::new(size.width as f32 / 2.0, size.height as f32 / 2.0));
                            }
                            // Show cursor while menu is open
                            self.window.set_cursor_visible(true);
                            self.window.request_redraw();
                            return true;
                        } else if event.state == ElementState::Released && menu.alt_held {
                            // Alt released - close menu and select hovered tool
                            menu.close(true);
                            // Hide cursor if a tool is now active
                            let hide_cursor = self.editor_state.radial_menu.active_tool != crate::ui::radial_menu::RadialTool::None;
                            self.window.set_cursor_visible(!hide_cursor);
                            self.window.request_redraw();
                            return true;
                        }
                    }
                    
                    // Clear drag state on Escape key
                    if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                        if event.state == ElementState::Pressed {
                            let menu = &mut self.editor_state.radial_menu;
                            if menu.dragging_cell.is_some() {
                                log::info!("Drag cancelled by Escape key");
                                self.scene_manager.clear_dragged_cell();
                                menu.dragging_cell = None;
                                self.window.request_redraw();
                                return true;
                            }
                        }
                    }
                }
                
                // Test performance spike detection with F12 key
                use winit::keyboard::{KeyCode, PhysicalKey};
                if let PhysicalKey::Code(KeyCode::F12) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        // Trigger a test performance spike log
                        self.performance.log_test_spike(75.5, "F12 key pressed - testing spike detection system");
                        log::info!("Performance spike test triggered via F12 key");
                        return true;
                    }
                }

                // Escape returns to the main menu from any in-game scene
                if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                    if event.state == ElementState::Pressed && self.app_phase == AppPhase::InGame {
                        log::info!("Escape pressed — returning to main menu");
                        // Rebuild the menu scene so the previews are fresh
                        self.main_menu_scene = Some(MainMenuScene::new(
                            &self.device,
                            &self.queue,
                            &self.config,
                            &mut self.ui.renderer,
                        ));
                        self.app_phase = AppPhase::MainMenu;
                        self.window.request_redraw();
                        return true;
                    }
                }
                
                // Only pass to camera if egui doesn't want the input
                if !self.ui.wants_keyboard_input() {
                    self.scene_manager.active_scene_mut().camera_mut().handle_keyboard(event);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            WindowEvent::Focused(focused) => {
                // Clear drag state when window loses focus
                if !focused && self.editor_state.radial_menu.dragging_cell.is_some() {
                    log::info!("Clearing drag state due to window focus loss");
                    self.scene_manager.clear_dragged_cell();
                    self.editor_state.radial_menu.clear_drag_state();
                    self.window.request_redraw();
                }
            }
            _ => {}
        }
        
        // Don't request repaint here - let about_to_wait handle frame timing
        // egui repaints will happen on the next scheduled frame
        
        true
    }
    
    // ─── Main menu ────────────────────────────────────────────────────────────

    /// Full render pass for a single main-menu frame.
    fn render_main_menu_frame(&mut self, dt: f32) {
        // Update genome simulations and orbit cameras.
        if let Some(menu) = &mut self.main_menu_scene {
            menu.update(dt);
        }

        // Render preview scenes into their off-screen textures.
        let cell_type_visuals = &self.editor_state.cell_type_visuals;
        if let Some(menu) = &mut self.main_menu_scene {
            menu.render(&self.device, &self.queue, Some(cell_type_visuals));
        }

        // Acquire swapchain texture.
        let output = loop {
            match self.surface.get_current_texture() {
                Ok(o) => break o,
                Err(wgpu::SurfaceError::Outdated) => {
                    self.surface.configure(&self.device, &self.config);
                }
                Err(e) => {
                    log::error!("Main menu surface error: {:?}", e);
                    return;
                }
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Clear the swapchain to black before egui paints.
        {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("menu_clear"),
            });
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("menu_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.queue.submit(std::iter::once(enc.finish()));
        }

        // egui frame.
        self.ui.begin_frame(&self.window);

        let (left_id, right_id, left_name, right_name, panel_w, panel_h) =
            if let Some(menu) = &self.main_menu_scene {
                (
                    menu.left_tex_id,
                    menu.right_tex_id,
                    menu.left_genome_name.clone(),
                    menu.right_genome_name.clone(),
                    menu.panel_width as f32,
                    menu.panel_height as f32,
                )
            } else {
                return;
            };

        let action = Self::render_main_menu_ui(
            &self.ui.ctx.clone(),
            left_id,
            right_id,
            &left_name,
            &right_name,
            panel_w,
            panel_h,
        );

        let egui_output = self.ui.ctx.end_pass();

        // Submit egui rendering.
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("menu_egui"),
        });
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        self.ui.render(&self.device, &self.queue, &mut enc, &view, screen_desc, egui_output);
        self.queue.submit(std::iter::once(enc.finish()));
        output.present();

        // Handle button actions.
        match action {
            MenuAction::Play => {
                self.app_phase = AppPhase::InGame;
                let target = crate::ui::types::SimulationMode::Gpu;
                self.ui.state.current_mode = target;
                let cave_init = self.scene_manager.switch_mode(
                    target,
                    &self.device,
                    &self.queue,
                    &self.config,
                    self.ui.state.world_diameter,
                    self.ui.state.world_settings.cell_capacity,
                    &self.editor_state,
                );
                if cave_init { self.editor_state.cave_params_dirty = true; }
                self.dock_manager.switch_mode(target);
            }
            MenuAction::GenomeEditor => {
                self.app_phase = AppPhase::InGame;
                let target = crate::ui::types::SimulationMode::Preview;
                self.ui.state.current_mode = target;
                self.scene_manager.switch_mode(
                    target,
                    &self.device,
                    &self.queue,
                    &self.config,
                    self.ui.state.world_diameter,
                    self.ui.state.world_settings.cell_capacity,
                    &self.editor_state,
                );
                self.dock_manager.switch_mode(target);
            }
            MenuAction::Exit => {
                // Save persistent state before exiting.
                self.dock_manager.save_all();
                self.ui.save_ui_state();
                self.editor_state.save_cell_type_visuals();
                std::process::exit(0);
            }
            _ => {}
        }

        // FPS counter.
        self.frame_count += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.frame_count = 0;
            self.fps_timer = std::time::Instant::now();
        }

        self.window.request_redraw();
    }

    /// Draw the main-menu egui overlay and return the button action (if any).
    fn render_main_menu_ui(ctx: &egui::Context, left_id: TextureId, right_id: TextureId, left_name: &str, right_name: &str, _panel_w: f32, _panel_h: f32) -> MenuAction {
        use egui::{Align2, Color32, FontId, FontFamily, Pos2, Rect, Stroke, Vec2};

        // Background: deep navy blue, darker at edges
        let bg          = Color32::from_rgb(7, 10, 22);
        let bg_centre   = Color32::from_rgb(10, 16, 36);
        // Fade colours match the background so panels blend in
        let fade_dark   = Color32::from_rgba_premultiplied(7, 10, 22, 255);
        let fade_clear  = Color32::from_rgba_premultiplied(7, 10, 22, 0);

        // Button palette
        let teal_fill   = Color32::from_rgba_premultiplied(29, 158, 117, 30);
        let teal_fill_h = Color32::from_rgba_premultiplied(29, 158, 117, 58);
        let teal_border = Color32::from_rgb(42, 122, 90);
        let teal_text   = Color32::from_rgb(160, 240, 205);

        let blue_fill   = Color32::from_rgba_premultiplied(55, 138, 221, 20);
        let blue_fill_h = Color32::from_rgba_premultiplied(55, 138, 221, 48);
        let blue_border = Color32::from_rgb(42, 64, 96);
        let blue_text   = Color32::from_rgb(160, 205, 240);

        let muted_fill   = Color32::TRANSPARENT;
        let muted_fill_h = Color32::from_rgba_premultiplied(255, 255, 255, 10);
        let muted_border = Color32::from_rgb(42, 42, 53);
        let muted_text   = Color32::from_rgb(136, 135, 144);

        let mut action = MenuAction::None;

        #[allow(deprecated)]
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(bg))
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let h = rect.height();
                let w = rect.width();
                let cx = rect.center().x;

                // Subtle horizontal gradient: slightly lighter navy in the centre
                let left_half  = Rect::from_min_max(rect.min, Pos2::new(cx, rect.max.y));
                let right_half = Rect::from_min_max(Pos2::new(cx, rect.min.y), rect.max);
                ui.painter().add(egui::Shape::from(Self::gradient_mesh(left_half,  bg, bg_centre)));
                ui.painter().add(egui::Shape::from(Self::gradient_mesh(right_half, bg_centre, bg)));

                // ── genome panel images ───────────────────────────────────────
                let display_panel_w = w / 3.0;
                let left_rect  = Rect::from_min_size(rect.min, Vec2::new(display_panel_w, h));
                let right_rect = Rect::from_min_size(
                    Pos2::new(rect.max.x - display_panel_w, rect.min.y),
                    Vec2::new(display_panel_w, h),
                );
                let full_uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
                ui.painter().image(left_id,  left_rect,  full_uv, Color32::WHITE);
                ui.painter().image(right_id, right_rect, full_uv, Color32::WHITE);

                // ── edge fades ────────────────────────────────────────────────
                let fade_w = 90.0_f32;
                let lr = Rect::from_min_size(
                    Pos2::new(rect.min.x + display_panel_w - fade_w, rect.min.y),
                    Vec2::new(fade_w, h),
                );
                ui.painter().add(egui::Shape::from(Self::gradient_mesh(lr, fade_clear, fade_dark)));
                let rr = Rect::from_min_size(
                    Pos2::new(rect.max.x - display_panel_w, rect.min.y),
                    Vec2::new(fade_w, h),
                );
                ui.painter().add(egui::Shape::from(Self::gradient_mesh(rr, fade_dark, fade_clear)));

                // ── genome name labels ────────────────────────────────────────
                let label_y = rect.max.y - 20.0;
                let label_font = FontId::new(11.0, FontFamily::Proportional);
                let label_color = Color32::from_rgb(47, 110, 84);
                ui.painter().text(
                    Pos2::new(rect.min.x + display_panel_w * 0.5, label_y),
                    Align2::CENTER_CENTER, left_name.to_uppercase(), label_font.clone(), label_color,
                );
                ui.painter().text(
                    Pos2::new(rect.max.x - display_panel_w * 0.5, label_y),
                    Align2::CENTER_CENTER, right_name.to_uppercase(), label_font, label_color,
                );

                // ── centre column ─────────────────────────────────────────────
                let btn_w = 240.0_f32;
                let btn_h = 44.0_f32;
                let gap   = 12.0_f32;

                let block_h = 56.0 + 30.0
                            + btn_h + gap + btn_h + gap
                            + 28.0
                            + btn_h + gap + btn_h + gap
                            + 28.0
                            + btn_h;

                #[allow(unused_assignments)]
                let mut y = rect.center().y - block_h * 0.5;

                // Title
                ui.painter().text(
                    Pos2::new(cx, y + 20.0),
                    Align2::CENTER_CENTER,
                    "BIO-SPHERES",
                    FontId::new(36.0, FontFamily::Proportional),
                    Color32::from_rgb(232, 244, 240),
                );
                y += 48.0;
                ui.painter().text(
                    Pos2::new(cx, y),
                    Align2::CENTER_CENTER,
                    "Evolution simulator",
                    FontId::new(14.0, FontFamily::Proportional),
                    Color32::from_rgb(100, 190, 155),
                );
                y += 38.0;

                macro_rules! btn {
                    ($label:expr, $fill:expr, $fill_h:expr, $border:expr, $text_col:expr) => {{
                        let r = Rect::from_center_size(
                            Pos2::new(cx, y + btn_h * 0.5),
                            Vec2::new(btn_w, btn_h),
                        );
                        let response = ui.put(
                            r,
                            egui::Button::new(
                                egui::RichText::new($label).color($text_col).size(15.0),
                            )
                            .fill($fill)
                            .stroke(Stroke::new(1.0, $border))
                            .corner_radius(32.0_f32),
                        );
                        if response.hovered() {
                            ui.painter().rect_filled(r, 32.0, $fill_h);
                            ui.painter().rect_stroke(r, 32.0, Stroke::new(1.0, $border), egui::StrokeKind::Outside);
                        }
                        y += btn_h + gap;
                        response
                    }};
                }

                if btn!("Play", teal_fill, teal_fill_h, teal_border, teal_text).clicked() {
                    action = MenuAction::Play;
                }
                if btn!("Genome editor", blue_fill, blue_fill_h, blue_border, blue_text).clicked() {
                    action = MenuAction::GenomeEditor;
                }

                y += 4.0;
                ui.painter().line_segment(
                    [Pos2::new(cx - 50.0, y), Pos2::new(cx + 50.0, y)],
                    Stroke::new(1.0, Color32::from_rgb(26, 26, 40)),
                );
                y += 14.0;

                if btn!("Settings", muted_fill, muted_fill_h, muted_border, muted_text).clicked() {
                    action = MenuAction::Settings;
                }
                if btn!("Credits", muted_fill, muted_fill_h, muted_border, muted_text).clicked() {
                    action = MenuAction::Credits;
                }

                y += 4.0;
                ui.painter().line_segment(
                    [Pos2::new(cx - 50.0, y), Pos2::new(cx + 50.0, y)],
                    Stroke::new(1.0, Color32::from_rgb(26, 26, 40)),
                );
                y += 14.0;

                if btn!("Exit", muted_fill, muted_fill_h, muted_border, muted_text).clicked() {
                    action = MenuAction::Exit;
                }
                let _ = y;
            });

        action
    }

    /// Horizontal gradient quad mesh (for edge fades).
    fn gradient_mesh(rect: egui::Rect, left_color: egui::Color32, right_color: egui::Color32) -> egui::Mesh {
        use egui::epaint::Vertex;
        use egui::Pos2;

        let uv = Pos2::new(0.0, 0.0);
        let mut mesh = egui::Mesh::default();
        mesh.vertices.push(Vertex { pos: rect.left_top(),     uv, color: left_color  });
        mesh.vertices.push(Vertex { pos: rect.right_top(),    uv, color: right_color });
        mesh.vertices.push(Vertex { pos: rect.right_bottom(), uv, color: right_color });
        mesh.vertices.push(Vertex { pos: rect.left_bottom(),  uv, color: left_color  });
        mesh.indices = vec![0, 1, 2, 0, 2, 3];
        mesh
    }

    fn render(&mut self) {
        // Don't render if surface has zero dimensions
        if self.config.width == 0 || self.config.height == 0 {
            return;
        }

        let now = std::time::Instant::now();

        // Skip render if we haven't reached the next frame time (60fps limiter)
        // IMPORTANT: This must be BEFORE acquiring surface texture to avoid cleanup issues
        if now < self.next_frame_time {
            return;
        }

        let dt = now.duration_since(self.last_render_time).as_secs_f32().min(0.1);
        self.last_render_time = now;

        // Schedule next frame for 60fps (16.67ms)
        self.next_frame_time = now + std::time::Duration::from_micros(16_667);

        // ── Main menu fast path ───────────────────────────────────────────────
        if self.app_phase == AppPhase::MainMenu {
            self.render_main_menu_frame(dt);
            return;
        }
        
        // Update performance metrics (includes automatic spike detection)
        self.performance.update(dt);
        
        // Update camera gravity direction only for GPU scene (preview scene ignores gravity)
        if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
            self.scene_manager.active_scene_mut().camera_mut().set_gravity_direction(
                self.ui.state.world_settings.gravity,
                self.ui.state.world_settings.gravity_mode,
            );
        }
        self.scene_manager.active_scene_mut().camera_mut().update(dt);

        // Push the camera out of cave walls using the same SDF the cells use.
        // Only applies in GPU scene mode when a cave is active.
        // We correct `camera.center` (the freefly position / orbit pivot) so
        // both modes benefit. The camera is treated as a small sphere so it
        // stays a comfortable distance from the surface.
        if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                if let Some(cave_renderer) = gpu_scene.cave_renderer.as_ref() {
                    let params = cave_renderer.params();
                    if params.collision_enabled != 0 {
                        use crate::rendering::cave_sdf_push_out;
                        const CAMERA_RADIUS: f32 = 3.0;
                        // `camera` is a public field on GpuScene — no trait import needed.
                        gpu_scene.camera.center = cave_sdf_push_out(
                            gpu_scene.camera.center,
                            params,
                            CAMERA_RADIUS,
                        );
                    }
                }
            }
        }

        self.scene_manager.update(dt);
        
        // Poll for async tool operation results (GPU mode only)
        if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
            // Poll for tool operation results and update radial menu state
            self.scene_manager.poll_tool_operation_results(&mut self.editor_state.radial_menu, &mut self.editor_state.drag_distance, &self.queue);

            // Poll organism follow readback and update camera center
            self.scene_manager.poll_organism_follow(&self.device, dt);
            
            // Apply cave parameters from UI if they changed
            if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                if self.editor_state.cave_params_dirty {
                    gpu_scene.apply_cave_params_from_editor(&self.editor_state);
                    // Clear the dirty flag in editor state
                    self.editor_state.cave_params_dirty = false;
                }
                
                // Apply light & fog parameters from UI if they changed
                if self.editor_state.light_params_dirty {
                    gpu_scene.apply_light_params_from_editor(&self.editor_state);
                    self.editor_state.light_params_dirty = false;
                }
                
                // Sync volumetric fog visibility toggle
                gpu_scene.show_volumetric_fog = self.editor_state.show_volumetric_fog;
                
                // Sync fluid voxel visibility toggle
                gpu_scene.show_fluid_voxels = self.editor_state.fluid_show_test_voxels;
                
                // Sync GPU density mesh visibility toggle
                gpu_scene.show_gpu_density_mesh = self.editor_state.fluid_show_mesh;
                
                // Update GPU surface nets params when changed
                if self.editor_state.fluid_mesh_needs_regen || self.editor_state.fluid_mesh_params_dirty {
                    if let Some(ref mut surface_nets) = gpu_scene.gpu_surface_nets {
                        // Update iso level
                        surface_nets.set_iso_level(&self.queue, self.editor_state.fluid_iso_level);
                    }
                    self.editor_state.fluid_mesh_needs_regen = false;
                    self.editor_state.fluid_mesh_params_dirty = false;
                }
                
                // Always update render params every frame (time drives wave animation)
                if let Some(ref surface_nets) = gpu_scene.gpu_surface_nets {
                    let params = crate::rendering::DensityMeshParams {
                        base_color: [0.2, 0.5, 0.9],
                        ambient: self.editor_state.fluid_ambient,
                        diffuse: self.editor_state.fluid_diffuse,
                        specular: self.editor_state.fluid_specular,
                        shininess: self.editor_state.fluid_shininess,
                        fresnel: self.editor_state.fluid_fresnel,
                        fresnel_power: self.editor_state.fluid_fresnel_power,
                        rim: self.editor_state.fluid_rim,
                        reflection: self.editor_state.fluid_reflection,
                        alpha: self.editor_state.fluid_alpha,
                        time: gpu_scene.current_time,
                        wave_height: self.editor_state.fluid_wave_height,
                        wave_speed: self.editor_state.fluid_wave_speed,
                        noise_scale: self.editor_state.fluid_noise_scale,
                        noise_octaves: self.editor_state.fluid_noise_octaves as f32,
                        noise_lacunarity: self.editor_state.fluid_noise_lacunarity,
                        noise_persistence: self.editor_state.fluid_noise_persistence,
                        reflection_brightness: self.editor_state.fluid_reflection_brightness,
                        light_dir: self.editor_state.light_dir,
                        _pad: 0.0,
                    };
                    surface_nets.update_render_params(&self.queue, &params);
                }

                // ── Organism skin sync ─────────────────────────────────────
                let os = &self.ui.state.fluid_settings.organism_skin;
                if os.enabled && gpu_scene.organism_skin_renderer.is_none() {
                    gpu_scene.initialize_organism_skin(&self.device, self.config.format, os);
                }
                gpu_scene.show_organism_skins = os.enabled;

                if let Some(ref mut skin) = gpu_scene.organism_skin_renderer {
                    skin.set_skin_radius_scale(&self.queue, os.radius_scale);
                    skin.set_iso_level(&self.queue, os.iso_level);
                    skin.set_shrink_params(&self.queue, os.shrink_speed, os.smooth_factor, os.shrink_iters, os.smooth_iters, os.min_cells);
                    let mut params = skin.skin_params;
                    params.base_r = os.base_color[0];
                    params.base_g = os.base_color[1];
                    params.base_b = os.base_color[2];
                    params.alpha = os.alpha;
                    params.sss_strength = os.sss_strength;
                    params.rim_strength = os.rim_strength;
                    params.light_dir_x = self.editor_state.light_dir[0];
                    params.light_dir_y = self.editor_state.light_dir[1];
                    params.light_dir_z = self.editor_state.light_dir[2];
                    skin.update_skin_params(&self.queue, params);
                }
            }
        }
        
        // Auto-save dock layouts periodically
        self.dock_manager.auto_save();
        
        let output = loop {
            match self.surface.get_current_texture() {
                Ok(output) => break output,
                Err(wgpu::SurfaceError::Outdated) => {
                    // Surface is outdated, reconfigure it and retry
                    self.surface.configure(&self.device, &self.config);
                    continue;
                }
                Err(e) => {
                    log::error!("Failed to get surface texture: {:?}", e);
                    return;
                }
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Apply occlusion culling settings from UI to GPU scene before rendering
        if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
            gpu_scene.set_occlusion_bias(self.ui.state.occlusion_bias);
            gpu_scene.set_occlusion_mip_override(self.ui.state.occlusion_mip_override);
            gpu_scene.set_occlusion_min_screen_size(self.ui.state.occlusion_min_screen_size);
            gpu_scene.set_occlusion_min_distance(self.ui.state.occlusion_min_distance);
            gpu_scene.set_readbacks_enabled(self.ui.state.gpu_readbacks_enabled);
            gpu_scene.show_adhesion_lines = self.ui.state.show_adhesion_lines;

            // Apply LOD settings from UI
            gpu_scene.set_lod_settings(
                self.ui.state.lod_scale_factor,
                self.ui.state.lod_threshold_low,
                self.ui.state.lod_threshold_medium,
                self.ui.state.lod_threshold_high,
                self.ui.state.lod_debug_colors,
            );

            // Apply gravity from UI
            gpu_scene.gravity = self.ui.state.world_settings.gravity;
            gpu_scene.gravity_mode = self.ui.state.world_settings.gravity_mode;
            gpu_scene.surface_pressure = self.ui.state.fluid_settings.surface_pressure;
            gpu_scene.constraint_iterations = self.ui.state.world_settings.constraint_iterations;
            gpu_scene.acceleration_damping = self.ui.state.world_settings.acceleration_damping;
            gpu_scene.water_viscosity = self.ui.state.world_settings.water_viscosity;
            gpu_scene.solo_metabolism_multiplier = if self.ui.state.world_settings.solo_metabolism_enabled {
                self.ui.state.world_settings.solo_metabolism_multiplier
            } else {
                1.0 // 1.0 means no penalty (feature disabled)
            };
            gpu_scene.radiation_level = self.ui.state.world_settings.radiation_level;
            gpu_scene.subtle_mutations = self.ui.state.world_settings.subtle_mutations;
            // Sync radiation level and mutation mode to mutation system
            if let Some(mutation_system) = &mut gpu_scene.mutation_system {
                mutation_system.set_radiation_level(self.ui.state.world_settings.radiation_level);
                mutation_system.set_subtle_mutations(&self.queue, self.ui.state.world_settings.subtle_mutations);
            }

            // Apply fluid settings from UI
            gpu_scene.lateral_flow_probabilities = self.editor_state.fluid_lateral_flow_probabilities;
            gpu_scene.condensation_probability = self.editor_state.fluid_condensation_probability;
            gpu_scene.vaporization_probability = self.editor_state.fluid_vaporization_probability;
            gpu_scene.nutrient_density = self.editor_state.nutrient_density;
            gpu_scene.nutrient_epoch_duration = self.editor_state.nutrient_epoch_duration;
            gpu_scene.nutrient_epoch_spacing = self.editor_state.nutrient_epoch_spacing;
            gpu_scene.nutrient_spawn_end = self.editor_state.nutrient_spawn_end;
            gpu_scene.nutrient_despawn_start = self.editor_state.nutrient_despawn_start;

            // Apply boulder/mossrock settings from editor_state (loaded from cave_settings.ron)
            gpu_scene.show_boulders = self.editor_state.show_boulders;
            gpu_scene.boulder_target_count = self.editor_state.boulder_target_count;
            gpu_scene.boulder_initial_moss = self.editor_state.boulder_initial_moss;
            gpu_scene.boulder_radius = self.editor_state.boulder_radius;
            gpu_scene.boulder_size_gate = self.editor_state.boulder_size_gate;
            gpu_scene.boulder_spawn_interval = self.editor_state.boulder_spawn_interval;
            gpu_scene.boulder_buoyancy = self.editor_state.boulder_buoyancy;
            gpu_scene.boulder_radius_min = self.editor_state.boulder_radius_min;
            gpu_scene.boulder_radius_max = self.editor_state.boulder_radius_max;
            gpu_scene.boulder_moss_min = self.editor_state.boulder_moss_min;
            gpu_scene.boulder_moss_max = self.editor_state.boulder_moss_max;
            // Propagate to live boulder system if it exists
            if let Some(ref mut bs) = gpu_scene.boulder_system {
                bs.target_count = self.editor_state.boulder_target_count;
                bs.spawn_interval = self.editor_state.boulder_spawn_interval;
                bs.radius_min = self.editor_state.boulder_radius_min;
                bs.radius_max = self.editor_state.boulder_radius_max;
                bs.moss_min = self.editor_state.boulder_moss_min;
                bs.moss_max = self.editor_state.boulder_moss_max;
                if (bs.buoyancy - self.editor_state.boulder_buoyancy).abs() > 1e-6 {
                    bs.buoyancy = self.editor_state.boulder_buoyancy;
                    bs.buoyancy_dirty = true;
                }
            }

            // Set culling mode based on enabled flags
            let culling_mode = match (self.ui.state.frustum_enabled, self.ui.state.occlusion_enabled) {
                (true, true) => crate::rendering::CullingMode::FrustumAndOcclusion,
                (true, false) => crate::rendering::CullingMode::FrustumOnly,
                (false, true) => crate::rendering::CullingMode::OcclusionOnly,
                (false, false) => crate::rendering::CullingMode::Disabled,
            };
            gpu_scene.set_culling_mode(culling_mode);
        }
        
        // Render 3D scene first (pass cell type visuals from editor state)
        let cell_type_visuals = &self.editor_state.cell_type_visuals;
        self.scene_manager.render(
            &self.device, 
            &self.queue, 
            &view, 
            Some(cell_type_visuals), 
            self.ui.state.world_diameter,
            self.ui.state.lod_scale_factor,
            self.ui.state.lod_threshold_low,
            self.ui.state.lod_threshold_medium,
            self.ui.state.lod_threshold_high,
            self.ui.state.lod_debug_colors,
            self.editor_state.cell_outline_width,
        );
        
        // Update culling stats from GPU scene using non-blocking async read
        // Only if GPU readbacks are enabled (can be disabled to avoid CPU-GPU sync overhead)
        if self.ui.state.gpu_readbacks_enabled {
            if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                // Poll for any pending async stats read
                if gpu_scene.instance_builder.poll_culling_stats(&self.device) {
                    let stats = gpu_scene.instance_builder.last_culling_stats();
                    self.performance.set_culling_stats(
                        stats.total_cells,
                        stats.visible_cells,
                        stats.frustum_culled,
                        stats.occluded,
                    );
                }
                
                // Start a new async read periodically (once per second)
                if self.performance.should_refresh_culling_stats() {
                    gpu_scene.instance_builder.start_culling_stats_read();
                }
            }
        }
        
        // Update continuous drag position every frame when dragging
        if let Some(cell_idx) = self.editor_state.radial_menu.dragging_cell {
            // Move cell to current mouse position at the same distance from camera using GPU operations
            let new_pos = self.scene_manager.screen_to_world_at_distance(
                self.mouse_position.0,
                self.mouse_position.1,
                self.editor_state.drag_distance,
            );
            
            // Use GPU position update via scene manager
            self.scene_manager.update_cell_position_gpu(cell_idx as u32, new_pos);
        }
        
        // Begin egui frame
        self.ui.begin_frame(&self.window);
        
        // Get current mode info for UI
        let current_mode = self.scene_manager.current_mode();
        let _cell_count = self.scene_manager.active_scene().cell_count();
        let _sim_time = self.scene_manager.active_scene().current_time();
        let _is_paused = self.scene_manager.active_scene().is_paused();
        
        // Update UI state with current simulation info
        self.ui.state.current_mode = current_mode;
        
        // Use persistent editor state and create scene request
        let mut scene_request = crate::ui::panel_context::SceneModeRequest::None;
        
        // Sync working genome from preview scene if in Preview mode
        // This keeps the genome available for GPU scene cell insertion.
        // Skip if a genome was just loaded this frame (the loaded genome takes priority).
        let egui_output = {
            let mut dummy_camera = crate::ui::camera::CameraController::new();
            
            // Get real data if in Preview mode
            if current_mode == crate::ui::types::SimulationMode::Preview
                && !self.editor_state.genome_just_loaded
            {
                if let Some(preview_scene) = self.scene_manager.get_preview_scene() {
                    // Preserve the name the user has typed — it's display-only and
                    // must not be overwritten by the scene sync every frame.
                    let preserved_name = self.working_genome.name.clone();
                    self.working_genome = preview_scene.genome.clone();
                    self.working_genome.name = preserved_name;
                    
                    // One-way sync: read simulation's actual time for progress bar display only
                    // Never write back to time_value — the slider is purely user-driven
                    self.editor_state.resim_display_time = preview_scene.get_time_for_ui();
                }
            } else if current_mode == crate::ui::types::SimulationMode::Preview {
                // Still sync the display time even when skipping genome sync
                if let Some(preview_scene) = self.scene_manager.get_preview_scene() {
                    self.editor_state.resim_display_time = preview_scene.get_time_for_ui();
                }
            }
            
            // In GPU mode, sync working_genome FROM GPU scene if it has genomes.
            // This ensures the UI always shows the GPU scene's genome, not a stale
            // preview genome. Without this, switching genomes in preview then switching
            // to GPU mode would push the preview genome into the GPU scene via update_genome.
            if current_mode == crate::ui::types::SimulationMode::Gpu {
                if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
                    if !gpu_scene.genomes.is_empty() && self.working_genome.modes.is_empty() {
                        self.working_genome = gpu_scene.genomes[0].clone();
                    }
                }
            }
            
            // Render right-click cell context menu for Preview mode
            if current_mode == crate::ui::types::SimulationMode::Preview {
                if let Some(preview_scene) = self.scene_manager.get_preview_scene_mut() {
                    let ctx = self.ui.ctx().clone();
                    if let Some(cell_idx) = preview_scene.context_menu_cell {
                        let screen_pos = preview_scene.context_menu_screen_pos;
                        let display_state = &preview_scene.state.display_state;
                        let genome = &preview_scene.genome;

                        // Gather cell info before mutable borrow
                        let mode_idx = display_state.mode_indices.get(cell_idx).copied().unwrap_or(0);
                        let mode = genome.modes.get(mode_idx);
                        let cell_type_name = mode
                            .map(|m| crate::cell::CellType::from_index(m.cell_type as u32)
                                .map(|ct| ct.name())
                                .unwrap_or("Unknown"))
                            .unwrap_or("Unknown");
                        let cell_type_idx = mode.map(|m| m.cell_type).unwrap_or(0);
                        let is_oculocyte = cell_type_idx == 7;
                        let _mass = display_state.masses.get(cell_idx).copied().unwrap_or(0.0);
                        let nutrients = display_state.nutrients.get(cell_idx).copied().unwrap_or(0.0);
                        let signal_channel = mode
                            .map(|m| m.oculocyte_signal_channel.clamp(0, 7) as usize)
                            .unwrap_or(0);
                        let signal_value = mode.map(|m| m.oculocyte_signal_value).unwrap_or(10.0);
                        let signal_hops = mode
                            .map(|m| m.oculocyte_signal_hops.clamp(1, 20) as usize)
                            .unwrap_or(3);

                        // Compute metabolism rates (nutrients/sec)
                        // Matches preview_physics.rs logic
                        const BASE_METABOLISM_RATE: f32 = 1.0;
                        const AUTO_GAIN_RATE: f32 = 20.0;
                        const SWIM_CONSUMPTION_RATE: f32 = 1.0; // Must match CONSUMPTION_RATE in preview_physics.rs::consume_swim_nutrients
                        const OCULOCYTE_SENSE_CONSUMPTION_RATE: f32 = 0.08;
                        let is_test_cell = cell_type_idx == 0;
                        let can_auto_gain = is_test_cell
                                         || cell_type_idx == 2  // Phagocyte
                                         || cell_type_idx == 3; // Photocyte
                        let is_flagellocyte = cell_type_idx == 1;
                        // is_oculocyte already defined above
                        // Test: pure 20/sec gain, no drain
                        // Phagocyte/Photocyte: 20/sec gain - 1/sec drain = net 19/sec
                        let gain_rate = if is_test_cell {
                            AUTO_GAIN_RATE
                        } else if can_auto_gain {
                            AUTO_GAIN_RATE - BASE_METABOLISM_RATE
                        } else {
                            0.0
                        };
                        let swim_drain = if is_flagellocyte {
                            let mode_settings = mode;
                            let effective_speed = if mode_settings.map(|m| m.flagellocyte_use_signal).unwrap_or(false) {
                                let channel = mode_settings.map(|m| m.flagellocyte_signal_channel.clamp(0, 7) as usize).unwrap_or(0);
                                let signal_value = display_state.signal_channels.get(cell_idx * 16 + channel).copied().flatten().unwrap_or(0.0);
                                let threshold_c = mode_settings.map(|m| m.flagellocyte_threshold_c).unwrap_or(0.0);
                                if signal_value >= threshold_c {
                                    mode_settings.map(|m| m.flagellocyte_speed_b).unwrap_or(0.0)
                                } else {
                                    mode_settings.map(|m| m.flagellocyte_speed_a).unwrap_or(0.0)
                                }
                            } else {
                                mode_settings.map(|m| m.swim_force).unwrap_or(0.0)
                            };
                            effective_speed * SWIM_CONSUMPTION_RATE
                        } else {
                            0.0
                        };
                        let sense_drain = if is_oculocyte {
                            mode.map(|m| m.oculocyte_ray_length * OCULOCYTE_SENSE_CONSUMPTION_RATE).unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        let base_drain = if can_auto_gain { 0.0 } else { BASE_METABOLISM_RATE };
                        let total_drain = base_drain + swim_drain + sense_drain;
                        let _net_rate = gain_rate - total_drain;

                        // Split info
                        let split_mass = mode.map(|m| m.split_mass).unwrap_or(2.0);
                        let split_never = split_mass > 2.0;
                        let nutrient_priority = mode.map(|m| m.nutrient_priority).unwrap_or(1.0);
                        let prioritize_when_low = mode.map(|m| m.prioritize_when_low).unwrap_or(false);

                        // Read actual flow rates recorded by the physics step.
                        // connection_flow_rates[i] = nutrients/sec, positive = A→B, negative = B→A.
                        // Sum up in/out for this cell from all its active connections.
                        let mut transport_out_rate: f32 = 0.0;
                        let mut transport_in_rate: f32 = 0.0;

                        for (conn_idx, &active) in display_state.adhesion_connections.is_active.iter().enumerate() {
                            if active == 0 { continue; }
                            let cell_a = display_state.adhesion_connections.cell_a_index.get(conn_idx).copied().unwrap_or(0);
                            let cell_b = display_state.adhesion_connections.cell_b_index.get(conn_idx).copied().unwrap_or(0);
                            if cell_a != cell_idx && cell_b != cell_idx { continue; }

                            let flow = display_state.adhesion_connections.connection_flow_rates
                                .get(conn_idx).copied().unwrap_or(0.0);

                            // flow is positive = A→B. Flip sign if we are cell_b.
                            let flow_from_my_perspective = if cell_a == cell_idx { flow } else { -flow };

                            if flow_from_my_perspective > 0.0 {
                                transport_out_rate += flow_from_my_perspective;
                            } else {
                                transport_in_rate += -flow_from_my_perspective;
                            }
                        }

                        let _net_transport_rate = transport_in_rate - transport_out_rate;

                        // Read all 16 signal channels for this cell
                        let cell_signals: [Option<f32>; 16] = std::array::from_fn(|ch| {
                            display_state.signal_channels.get(cell_idx * 16 + ch).copied().flatten()
                        });

                        let mut close_menu = false;
                        let mut send_test_signal = false;

                        let area_resp = egui::Area::new(egui::Id::new("cell_context_menu"))
                            .fixed_pos(egui::Pos2::new(screen_pos.0, screen_pos.1))
                            .interactable(true)
                            .order(egui::Order::Foreground)
                            .show(&ctx, |ui| {
                                egui::Frame::popup(ui.style()).show(ui, |ui| {
                                    ui.set_min_width(220.0);

                                    // --- Header ---
                                    ui.label(egui::RichText::new(format!("{} — M{}", cell_type_name, mode_idx + 1)).strong());
                                    ui.label(egui::RichText::new(format!("Cell #{}", cell_idx)).color(egui::Color32::from_rgb(140, 140, 140)).small());
                                    ui.separator();

                                    let dim = egui::Color32::from_rgb(160, 160, 160);
                                    let red = egui::Color32::from_rgb(220, 80, 80);
                                    let green = egui::Color32::from_rgb(80, 200, 120);
                                    let white = egui::Color32::from_rgb(230, 230, 230);

                                    // --- Nutrients bar ---
                                    // split_nutrient_threshold: nutrients needed to divide
                                    // When split_never, use 100 as display cap (cells max out at 100 normally)
                                    let split_nutrient_threshold = (split_mass - 1.0) * 100.0;
                                    let is_lipocyte = cell_type_idx == 4;
                                    let nutrient_max = if is_lipocyte {
                                        200.0
                                    } else if split_never {
                                        100.0
                                    } else {
                                        split_nutrient_threshold * 2.0
                                    };
                                    let nutrient_frac = (nutrients / nutrient_max).clamp(0.0, 1.0);
                                    let bar_width = 180.0;
                                    let bar_height = 8.0;
                                    let (bar_rect, _) = ui.allocate_exact_size(
                                        egui::vec2(bar_width, bar_height),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(50, 50, 50));
                                    let fill_color = if nutrient_frac > 0.5 { green } else if nutrient_frac > 0.2 { egui::Color32::from_rgb(220, 180, 50) } else { red };
                                    let fill_rect = egui::Rect::from_min_size(bar_rect.min, egui::vec2(bar_rect.width() * nutrient_frac, bar_height));
                                    ui.painter().rect_filled(fill_rect, 2.0, fill_color);
                                    // Split threshold marker
                                    if !split_never {
                                        let split_frac = (split_nutrient_threshold / nutrient_max).clamp(0.0, 1.0);
                                        let marker_x = bar_rect.min.x + bar_rect.width() * split_frac;
                                        ui.painter().line_segment(
                                            [egui::pos2(marker_x, bar_rect.min.y), egui::pos2(marker_x, bar_rect.max.y)],
                                            egui::Stroke::new(1.5, white),
                                        );
                                    }
                                    ui.label(egui::RichText::new(format!("{:.0} / {:.0}", nutrients, nutrient_max)).color(dim).small());

                                    ui.separator();

                                    // --- Metabolism ---
                                    egui::Grid::new("metabolism_grid")
                                        .num_columns(2)
                                        .spacing([8.0, 2.0])
                                        .show(ui, |ui| {
                                            // Gain row (only if cell produces)
                                            if gain_rate > 0.0 {
                                                ui.colored_label(dim, "Gain");
                                                ui.colored_label(green, format!("+{:.1}/s", gain_rate));
                                                ui.end_row();
                                            }

                                            // Base drain
                                            let base_drain = total_drain - swim_drain - sense_drain;
                                            if base_drain > 0.0 {
                                                ui.colored_label(dim, "Upkeep");
                                                ui.colored_label(red, format!("-{:.1}/s", base_drain));
                                                ui.end_row();
                                            }

                                            // Swim drain (flagellocytes)
                                            if swim_drain > 0.0 {
                                                ui.colored_label(dim, "Swimming");
                                                ui.colored_label(red, format!("-{:.1}/s", swim_drain));
                                                ui.end_row();
                                            }

                                            // Sense drain (oculocytes)
                                            if sense_drain > 0.0 {
                                                ui.colored_label(dim, "Sensing");
                                                ui.colored_label(red, format!("-{:.1}/s", sense_drain));
                                                ui.end_row();
                                            }

                                            // Transport (only when connected)
                                            if transport_in_rate > 0.0 || transport_out_rate > 0.0 {
                                                let net_t = transport_in_rate - transport_out_rate;
                                                ui.colored_label(dim, "Transport");
                                                if net_t >= 0.0 {
                                                    ui.colored_label(green, format!("+{:.1}/s", net_t));
                                                } else {
                                                    ui.colored_label(red, format!("{:.1}/s", net_t));
                                                }
                                                ui.end_row();
                                            }

                                            ui.separator();
                                            ui.separator();
                                            ui.end_row();

                                            // Net = everything combined
                                            let total_net = gain_rate - total_drain + transport_in_rate - transport_out_rate;
                                            ui.label(egui::RichText::new("Net").strong());
                                            if total_net >= 0.0 {
                                                ui.colored_label(green, egui::RichText::new(format!("+{:.1}/s", total_net)).strong());
                                            } else {
                                                ui.colored_label(red, egui::RichText::new(format!("{:.1}/s", total_net)).strong());
                                            }
                                            ui.end_row();

                                            // Split threshold
                                            ui.colored_label(dim, "Splits at");
                                            if split_never {
                                                ui.colored_label(dim, "Never");
                                            } else {
                                                ui.colored_label(dim, format!("{:.0} nutrients", split_nutrient_threshold));
                                            }
                                            ui.end_row();

                                            // Priority (only show if non-default)
                                            if nutrient_priority != 1.0 || prioritize_when_low {
                                                ui.colored_label(dim, "Priority");
                                                let low_str = if prioritize_when_low { " (boost low)" } else { "" };
                                                ui.colored_label(dim, format!("{:.1}{}", nutrient_priority, low_str));
                                                ui.end_row();
                                            }
                                        });
                                    ui.separator();

                                    // --- Signal Channels (2 columns: Ch 0-7 left, Ch 8-15 right) ---
                                    ui.label("Signal Channels:");
                                    egui::Grid::new("signal_channels_grid")
                                        .num_columns(4)
                                        .spacing([4.0, 2.0])
                                        .show(ui, |ui| {
                                            let yellow = egui::Color32::from_rgb(255, 220, 50);
                                            let gray = egui::Color32::from_rgb(100, 100, 100);
                                            for row in 0..8usize {
                                                let ch_left = row;
                                                let ch_right = row + 8;

                                                // Left column
                                                ui.label(format!("Ch {:2}:", ch_left));
                                                match cell_signals[ch_left] {
                                                    Some(v) => ui.colored_label(yellow, format!("{:.1}", v)),
                                                    None => ui.colored_label(gray, "—"),
                                                };

                                                // Right column
                                                ui.label(format!("Ch{:2}:", ch_right));
                                                match cell_signals[ch_right] {
                                                    Some(v) => ui.colored_label(yellow, format!("{:.1}", v)),
                                                    None => ui.colored_label(gray, "—"),
                                                };

                                                ui.end_row();
                                            }
                                        });
                                    ui.separator();

                                    if is_oculocyte {
                                        // Check if this cell already has an active test signal
                                        let has_active_signal = self.test_signal_emissions.iter()
                                            .any(|emission| emission.source_cell == cell_idx);
                                        
                                        let button_text = if has_active_signal { 
                                            "Stop Test Signal" 
                                        } else { 
                                            "Send Test Signal" 
                                        };
                                        
                                        if ui.button(button_text).clicked() {
                                            if has_active_signal {
                                                // Remove the signal (toggle off)
                                                self.test_signal_emissions.retain(|emission| emission.source_cell != cell_idx);
                                                log::info!("Stopped test signal from cell {}", cell_idx);
                                                self.test_signals_changed = true;
                                            } else {
                                                // Add the signal (toggle on)
                                                send_test_signal = true;
                                            }
                                            close_menu = true;
                                        }
                                    }

                                    if ui.button("Close").clicked() {
                                        close_menu = true;
                                    }
                                });
                            });

                        // Close on escape
                        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                            close_menu = true;
                        }

                        // Close on click outside the popup, with a 300ms grace period
                        // so the right-click that opened the menu doesn't immediately close it
                        let elapsed = preview_scene.context_menu_open_time.elapsed();
                        if elapsed > std::time::Duration::from_millis(300) {
                            // Check if pointer clicked outside the popup area
                            let popup_rect = area_resp.response.rect;
                            let clicked_outside = ctx.input(|i| {
                                if let Some(pos) = i.pointer.interact_pos() {
                                    i.pointer.any_pressed() && !popup_rect.contains(pos)
                                } else {
                                    false
                                }
                            });
                            if clicked_outside {
                                close_menu = true;
                            }
                        }

                        if send_test_signal {
                            let emission = crate::simulation::signal_system::SignalEmission {
                                source_cell: cell_idx,
                                channel: signal_channel,
                                value: signal_value,
                                hops: signal_hops,
                            };
                            // Add to persistent test signal emissions
                            self.test_signal_emissions.push(emission);
                            log::info!("Started test signal from cell {} on channel {} (value={}, hops={})",
                                cell_idx, signal_channel, signal_value, signal_hops);
                            self.test_signals_changed = true;
                        }

                        if close_menu {
                            preview_scene.context_menu_cell = None;
                        }
                    }
                }
            }

            let output = self.ui.end_frame(
                &mut self.dock_manager,
                &mut self.working_genome,
                &mut self.editor_state,
                &mut self.scene_manager,
                &mut dummy_camera,
                &mut scene_request,
                &self.performance,
            );
            
            // Sync genome changes and time slider back to the scene if in Preview mode
            if current_mode == crate::ui::types::SimulationMode::Preview {
                if let Some(preview_scene) = self.scene_manager.get_preview_scene_mut() {
                    // Sync physics config from UI world settings so constraint_iterations
                    // and other parameters take effect in the preview physics step.
                    preview_scene.config.constraint_iterations = self.ui.state.world_settings.constraint_iterations;

                    preview_scene.update_genome(&self.working_genome);

                    // Clear the just-loaded flag now that the genome has been pushed
                    // into the preview scene. From the next frame the normal sync resumes.
                    self.editor_state.genome_just_loaded = false;
                    
                    // Sync time slider to simulation (when dragging or changed)
                    preview_scene.sync_time_from_ui(
                        self.editor_state.time_value,
                        self.editor_state.max_preview_duration,
                        self.editor_state.time_slider_dragging,
                    );
                    
                    // Bidirectional sync: genome panel selection → preview highlight
                    preview_scene.selected_mode_indices = self.editor_state.selected_mode_indices.clone();
                    
                    // Sync test signals to preview scene (must happen before resimulation trigger)
                    preview_scene.test_signals = self.test_signal_emissions.clone();
                    
                    // Trigger resimulation if test signals changed
                    if self.test_signals_changed {
                        let current_time = preview_scene.state.display_time;
                        preview_scene.state.seek_to_time(current_time);
                        self.test_signals_changed = false;
                    }

                    // Sync adhesion expansion tool. Mirror exactly what update_genome does:
                    // clear checkpoints and seek to display_time so step_to replays from
                    // initial state with the new flag active on every physics step.
                    let now_active = self.ui.state.adhesion_expansion_active;
                    if preview_scene.state.work_state.adhesion_expansion_active != now_active {
                        preview_scene.state.work_state.adhesion_expansion_active = now_active;
                        preview_scene.state.display_state.adhesion_expansion_active = now_active;
                        preview_scene.state.clear_checkpoints();
                        let target = preview_scene.state.display_time;
                        preview_scene.state.seek_to_time(target);
                    }
                }
            }
            
            // In GPU mode, push physics parameter changes (stiffness, damping, rest length,
            // break force, etc.) to the GPU settings buffers so existing cells pick them up
            // immediately. update_genome only rewrites the per-mode settings arrays — it does
            // not touch cell positions, velocities, mode_indices, or adhesion connections, so
            // existing cells are unaffected structurally. New cells spawned after the edit
            output
        };
        
        // Update gizmo configuration for all scenes
        self.scene_manager.update_gizmo_config(&self.editor_state);
        
        // Update split ring configuration for all scenes
        self.scene_manager.update_split_ring_config(&self.editor_state);
        
        // Handle scene mode requests from UI panels
        if scene_request.is_requested() {
            match scene_request {
                crate::ui::panel_context::SceneModeRequest::TogglePause => {
                    let scene = self.scene_manager.active_scene_mut();
                    let current_paused = scene.is_paused();
                    scene.set_paused(!current_paused);
                }
                crate::ui::panel_context::SceneModeRequest::Reset => {
                    // Use capacity from UI slider
                    let capacity = self.ui.state.world_settings.cell_capacity;

                    // Scale cave noise proportionally to world radius (base: scale=100 at radius=200)
                    let new_radius = self.ui.state.world_settings.world_radius;
                    self.editor_state.cave_scale = new_radius / 2.0;

                    // Commit world_diameter from the slider value before reset
                    self.ui.state.world_diameter = new_radius * 2.0;
                    
                    // Recreate GPU scene with appropriate capacity if needed
                    self.scene_manager.recreate_gpu_scene_with_capacity(
                        &self.device,
                        &self.queue,
                        &self.config,
                        self.ui.state.world_diameter,
                        capacity,
                        &self.editor_state,
                    );
                    
                    // Reset the GPU scene
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        gpu_scene.reset(&self.queue);

                        // Reset fluid simulation
                        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Fluid Reset Encoder"),
                        });
                        gpu_scene.reset_fluid(&self.device, &self.queue, &mut encoder);
                        self.queue.submit(std::iter::once(encoder.finish()));

                        // Reapply saved cave settings after reset
                        self.editor_state.cave_params_dirty = true;
                        gpu_scene.apply_cave_params_from_editor(&self.editor_state);
                        gpu_scene.update_cave_params(&self.device, &self.queue);
                        self.editor_state.cave_params_dirty = false;
                    }
                }
                crate::ui::panel_context::SceneModeRequest::ResetCellsOnly => {
                    // Use capacity from UI slider
                    let capacity = self.ui.state.world_settings.cell_capacity;

                    // Scale cave noise proportionally to world radius (base: scale=100 at radius=200)
                    let new_radius = self.ui.state.world_settings.world_radius;
                    self.editor_state.cave_scale = new_radius / 2.0;

                    // Commit world_diameter from the slider value before reset
                    self.ui.state.world_diameter = new_radius * 2.0;
                    
                    // Recreate GPU scene with appropriate capacity if needed
                    self.scene_manager.recreate_gpu_scene_with_capacity(
                        &self.device,
                        &self.queue,
                        &self.config,
                        self.ui.state.world_diameter,
                        capacity,
                        &self.editor_state,
                    );
                    
                    // Reset the GPU scene (cells only, keep fluid)
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        gpu_scene.reset(&self.queue);

                        // Reapply saved cave settings after reset
                        self.editor_state.cave_params_dirty = true;
                        gpu_scene.apply_cave_params_from_editor(&self.editor_state);
                        gpu_scene.update_cave_params(&self.device, &self.queue);
                        self.editor_state.cave_params_dirty = false;
                    }
                }
                crate::ui::panel_context::SceneModeRequest::SetSpeed(speed) => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        gpu_scene.time_scale = speed;
                    }
                }
                crate::ui::panel_context::SceneModeRequest::SetSpeedAndUnpause(speed) => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        gpu_scene.time_scale = speed;
                        gpu_scene.paused = false;
                    }
                }
                crate::ui::panel_context::SceneModeRequest::RegenerateFluidVoxels => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        if gpu_scene.fluid_buffers.is_some() {
                            gpu_scene.generate_test_voxels(&self.queue);
                            log::info!("Regenerated fluid test voxels");
                        }
                    }
                }
                crate::ui::panel_context::SceneModeRequest::RegenerateFluidMesh => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        if gpu_scene.gpu_surface_nets.is_some() {
                            gpu_scene.generate_test_density(&self.queue);
                            log::info!("Generated test density field for GPU surface nets");
                        }
                    }
                }
                crate::ui::panel_context::SceneModeRequest::LoadGenomeFromGpu(genome_id) => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
                        if let Some(genome) = gpu_scene.read_back_genome(&self.device, &self.queue, genome_id) {
                            log::info!("Loaded genome '{}' ({} modes) from GPU", genome.name, genome.modes.len());
                            self.working_genome = genome;
                            
                            // Switch to Preview mode immediately (not deferred) so we can
                            // push the readback genome into the preview scene right away.
                            // If we used request_mode_switch, the next frame's per-frame sync
                            // (working_genome = preview_scene.genome.clone()) would overwrite
                            // our readback genome before update_genome could push it.
                            let preview_mode = crate::ui::types::SimulationMode::Preview;
                            if current_mode != preview_mode {
                                let cave_initialized = self.scene_manager.switch_mode(
                                    preview_mode, &self.device, &self.queue, &self.config,
                                    self.ui.state.world_diameter,
                                    self.ui.state.world_settings.cell_capacity,
                                    &self.editor_state,
                                );
                                if cave_initialized {
                                    self.editor_state.cave_params_dirty = true;
                                }
                                self.dock_manager.switch_mode(preview_mode);
                                // Clear radial menu / drag state from GPU mode
                                if self.editor_state.radial_menu.dragging_cell.is_some() {
                                    self.scene_manager.clear_dragged_cell();
                                    self.editor_state.radial_menu.clear_drag_state();
                                }
                                self.window.set_cursor_visible(true);
                                self.editor_state.radial_menu.active_tool = crate::ui::radial_menu::RadialTool::None;
                                self.editor_state.radial_menu.visible = false;
                                log::info!("Switched to Preview mode for genome inspection");
                            }
                            
                            // Now push the readback genome into the preview scene immediately
                            if let Some(preview_scene) = self.scene_manager.get_preview_scene_mut() {
                                preview_scene.update_genome(&self.working_genome);
                                log::info!("Pushed readback genome into preview scene");
                            }
                        } else {
                            log::error!("Failed to read back genome_id={} from GPU", genome_id);
                        }
                    }
                }
                crate::ui::panel_context::SceneModeRequest::SaveSnapshot => {
                    // Defer the actual work until after output.present() so the
                    // "Saving…" popup is visible on screen before we block.
                    self.deferred_action = Some(DeferredAction::SaveSphere);
                }
                crate::ui::panel_context::SceneModeRequest::LoadSnapshot(path) => {
                    // Defer the actual work until after output.present() so the
                    // "Loading…" popup is visible on screen before we block.
                    self.deferred_action = Some(DeferredAction::LoadSphere(path));
                }
                _ => {
                    if let Some(target_mode) = scene_request.target_mode() {
                        self.ui.state.request_mode_switch(target_mode);
                    }
                }
            }
        }
        
        // Check if mode switch was requested via UI
        if let Some(requested_mode) = self.ui.state.take_mode_request() {
            if requested_mode != current_mode {
                // Sync genome from preview scene before switching to GPU mode
                if requested_mode == crate::ui::types::SimulationMode::Gpu {
                    if let Some(preview_scene) = self.scene_manager.get_preview_scene() {
                        self.working_genome = preview_scene.genome.clone();
                        log::info!("Synced genome to GPU scene: {} modes", self.working_genome.modes.len());
                    }
                }
                
                let cave_initialized = self.scene_manager.switch_mode(requested_mode, &self.device, &self.queue, &self.config, self.ui.state.world_diameter, self.ui.state.world_settings.cell_capacity, &self.editor_state);
                if cave_initialized {
                    // Cave was just initialized, mark params as dirty so they get applied
                    self.editor_state.cave_params_dirty = true;
                }
                self.dock_manager.switch_mode(requested_mode);
                // Reset cursor visibility and radial menu state when switching modes
                
                // Clear any active drag state when switching modes
                if self.editor_state.radial_menu.dragging_cell.is_some() {
                    log::info!("Clearing drag state due to mode switch");
                    self.scene_manager.clear_dragged_cell();
                    self.editor_state.radial_menu.clear_drag_state();
                }
                self.window.set_cursor_visible(true);
                self.editor_state.radial_menu.active_tool = crate::ui::radial_menu::RadialTool::None;
                self.editor_state.radial_menu.visible = false;
                log::info!("Switched to {} mode", requested_mode.display_name());
            }
        }
        
        // Check if mode switch was requested via dock manager (for layout persistence)
        let dock_mode = self.dock_manager.current_mode();
        if dock_mode != current_mode {
            // Sync genome from preview scene before switching to GPU mode
            if dock_mode == crate::ui::types::SimulationMode::Gpu {
                if let Some(preview_scene) = self.scene_manager.get_preview_scene() {
                    self.working_genome = preview_scene.genome.clone();
                }
            }
            let cave_initialized = self.scene_manager.switch_mode(dock_mode, &self.device, &self.queue, &self.config, self.ui.state.world_diameter, self.ui.state.world_settings.cell_capacity, &self.editor_state);
            if cave_initialized {
                // Cave was just initialized, mark params as dirty so they get applied
                self.editor_state.cave_params_dirty = true;
            }
        }
        
        // Create command encoder for egui rendering
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui_encoder"),
        });
        
        // Create screen descriptor
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        
        // Render egui
        self.ui.render(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            screen_descriptor,
            egui_output,
        );

        // If a screenshot was requested, copy the fully-rendered swapchain texture
        // to a staging buffer in the same encoder pass (before submit + present).
        let screenshot_staging = if self.editor_state.request_screenshot {
            self.editor_state.request_screenshot = false;
            let w = self.config.width;
            let h = self.config.height;
            // Bytes per row must be aligned to 256 bytes (wgpu requirement).
            let bytes_per_pixel = 4u32; // RGBA8 or BGRA8
            let unpadded_bytes_per_row = w * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
            let buffer_size = (padded_bytes_per_row * h) as u64;

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Staging Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            encoder.copy_texture_to_buffer(
                output.texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );

            Some((staging, w, h, padded_bytes_per_row, unpadded_bytes_per_row))
        } else {
            None
        };

        // Submit egui commands (includes the screenshot copy if requested)
        self.queue.submit(std::iter::once(encoder.finish()));

        output.present();

        // ── Process screenshot readback ────────────────────────────────────────
        // Runs after present() — the staging buffer is already populated.
        if let Some((staging, w, h, padded_bpr, unpadded_bpr)) = screenshot_staging {
            self.deferred_action = Some(DeferredAction::TakeScreenshot {
                staging,
                width: w,
                height: h,
                padded_bytes_per_row: padded_bpr,
                unpadded_bytes_per_row: unpadded_bpr,
                format: self.config.format,
            });
        }

        // Check for GIF capture request — deferred to after present().
        if self.editor_state.request_gif_capture {
            self.editor_state.request_gif_capture = false;
            if self.deferred_action.is_none() {
                self.deferred_action = Some(DeferredAction::CaptureGif);
            }
        }

        // ── Execute deferred save/load action ─────────────────────────────────
        // This runs AFTER present() so the "Saving…" / "Loading…" popup is
        // already visible on screen before the blocking work begins.
        if let Some(action) = self.deferred_action.take() {
            match action {
                DeferredAction::SaveSphere => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        let was_paused = gpu_scene.paused;
                        gpu_scene.paused = true;

                        match gpu_scene.save_snapshot(&self.device, &self.queue) {
                            Ok(snapshot) => {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Save Sphere")
                                    .add_filter("Bio-Spheres Sphere", &["sphere"])
                                    .set_directory(crate::app_dirs::spheres_dir())
                                    .set_file_name("simulation.sphere")
                                    .save_file()
                                {
                                    match snapshot.save_to_file(&path) {
                                        Ok(()) => log::info!("Sphere saved to {:?}", path),
                                        Err(e) => log::error!("Failed to write sphere: {}", e),
                                    }
                                }
                            }
                            Err(e) => log::error!("Failed to capture sphere: {}", e),
                        }

                        gpu_scene.paused = was_paused;
                    }
                    self.ui.state.show_saving_popup = false;
                    self.ui.state.pending_save_ready = false;
                    // Request a redraw so the popup disappears immediately.
                    self.window.request_redraw();
                }
                DeferredAction::LoadSphere(path) => {
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        match crate::scene::GpuSceneSnapshot::load_from_file(&path) {
                            Ok(snapshot) => {
                                gpu_scene.paused = true;
                                match gpu_scene.restore_from_snapshot(&self.device, &self.queue, &snapshot) {
                                    Ok(()) => log::info!("Sphere loaded from {:?}", path),
                                    Err(e) => log::error!("Failed to restore sphere: {}", e),
                                }
                            }
                            Err(e) => log::error!("Failed to load sphere file: {}", e),
                        }
                    }
                    self.ui.state.show_loading_popup = false;
                    self.window.request_redraw();
                }
                DeferredAction::TakeScreenshot { staging, width, height, padded_bytes_per_row, unpadded_bytes_per_row, format } => {
                    // Map the staging buffer and read back the pixel data.
                    let slice = staging.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
                    match self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }) {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!("Screenshot: device.poll failed: {:?}", e);
                            return;
                        }
                    }
                    if rx.recv().map_err(|_| ()).and_then(|r| r.map_err(|_| ())).is_err() {
                        log::error!("Screenshot: staging buffer map failed");
                        return;
                    }

                    let mapped = slice.get_mapped_range();
                    // Strip row padding and convert BGRA→RGBA if needed.
                    let is_bgra = matches!(
                        format,
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                    );
                    let mut rgba: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
                    for row in 0..height {
                        let row_start = (row * padded_bytes_per_row) as usize;
                        let row_bytes = &mapped[row_start..row_start + unpadded_bytes_per_row as usize];
                        if is_bgra {
                            for chunk in row_bytes.chunks_exact(4) {
                                rgba.push(chunk[2]); // R
                                rgba.push(chunk[1]); // G
                                rgba.push(chunk[0]); // B
                                rgba.push(chunk[3]); // A
                            }
                        } else {
                            rgba.extend_from_slice(row_bytes);
                        }
                    }
                    drop(mapped);
                    staging.unmap();

                    // Build a timestamped filename and save to the screenshots folder.
                    let screenshots_dir = crate::app_dirs::screenshots_dir();
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let filename = format!("screenshot_{}.png", timestamp);
                    let path = screenshots_dir.join(&filename);

                    match image::RgbaImage::from_raw(width, height, rgba) {
                        Some(img) => {
                            match img.save(&path) {
                                Ok(()) => log::info!("Screenshot saved to {:?}", path),
                                Err(e) => log::error!("Screenshot: failed to save PNG: {}", e),
                            }
                        }
                        None => log::error!("Screenshot: failed to construct image from pixel data"),
                    }
                }
                DeferredAction::CaptureGif => {
                    // Start the incremental GIF capture state machine.
                    let (genome, cam_rotation, cam_distance) = if let Some(preview) = self.scene_manager.get_preview_scene() {
                        (
                            preview.genome.clone(),
                            preview.camera.rotation,
                            preview.camera.distance,
                        )
                    } else {
                        (
                            self.working_genome.clone(),
                            glam::Quat::from_axis_angle(glam::Vec3::X, -0.35),
                            40.0,
                        )
                    };
                    let sim_time = self.editor_state.time_value;
                    let cell_type_visuals = self.editor_state.cell_type_visuals.clone();

                    match crate::gif_capture::GifCaptureState::begin(
                        &self.device,
                        &self.queue,
                        self.config.format,
                        &genome,
                        sim_time,
                        cam_rotation,
                        cam_distance,
                        Some(&cell_type_visuals),
                    ) {
                        Ok(state) => {
                            self.editor_state.gif_capture = Some(state);
                        }
                        Err(e) => {
                            log::error!("GIF capture failed to start: {}", e);
                            crate::ui::toast::remove_progress_toasts(&mut self.ui.toasts);
                            self.ui.toasts.push(crate::ui::toast::Toast::error(
                                format!("GIF failed: {}", e)
                            ));
                        }
                    }
                    self.window.request_redraw();
                }
            }
        }

        // ── Drive incremental GIF capture ─────────────────────────────────────
        if let Some(ref mut capture) = self.editor_state.gif_capture {
            let cell_type_visuals = self.editor_state.cell_type_visuals.clone();
            capture.step(&self.device, &self.queue, Some(&cell_type_visuals));

            let done = capture.frames_done();
            let total = capture.frames_total();
            let msg = format!("Capturing GIF… {}/{}", done, total);
            crate::ui::toast::upsert_progress_toast(
                &mut self.ui.toasts,
                &msg,
                capture.progress(),
            );

            if capture.is_done() {
                let result = capture.result.take().unwrap();
                crate::ui::toast::remove_progress_toasts(&mut self.ui.toasts);
                match result {
                    Ok(ref gif_path) => {
                        let gif_stem = gif_path.file_stem()
                            .and_then(|n| n.to_str())
                            .unwrap_or("thumbnail")
                            .to_string();

                        self.ui.toasts.push(crate::ui::toast::Toast::success(
                            format!("✓ GIF saved — {}.gif", gif_stem)
                        ));
                        // Refresh the genome browser so the new thumbnail appears.
                        self.ui.genome_browser.needs_refresh = true;
                        self.ui.genome_browser.force_full_reload = true;
                    }
                    Err(e) => {
                        self.ui.toasts.push(crate::ui::toast::Toast::error(
                            format!("GIF failed: {}", e)
                        ));
                    }
                }
                self.editor_state.gif_capture = None;
            }

            self.window.request_redraw();
        }
        
        // FPS counter
        self.frame_count += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            log::info!("FPS: {}", self.frame_count);
            self.frame_count = 0;
            self.fps_timer = std::time::Instant::now();
        }
    }
    
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
    
    /// Get the next scheduled frame time for 60fps limiting
    pub fn next_frame_time(&self) -> std::time::Instant {
        self.next_frame_time
    }
    

}

struct AppState {
    app: Option<App>,
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        
        // Build window icon from embedded PNG so it is set at creation time.
        // Passing it via with_window_icon() is required for Wayland compositors
        // and ensures the taskbar / launcher shows the correct icon immediately.
        let window_icon = {
            let icon_bytes = include_bytes!("../assets/icon.png");
            image::load_from_memory(icon_bytes).ok().and_then(|img| {
                let img = img.into_rgba8();
                let (w, h) = img.dimensions();
                winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
            })
        };

        let window_attributes = Window::default_attributes()
            .with_title("Bio-Spheres Preview")
            .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
            .with_window_icon(window_icon);

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        window.set_maximized(true);

        // On Windows the taskbar icon comes from the window CLASS icon, not the
        // window instance icon. winit registers its class without an icon, so we
        // patch it after creation using SetClassLongPtrW (GCLP_HICON / GCLP_HICONSM)
        // and also send WM_SETICON for the instance. Both are needed for full coverage
        // across taskbar, alt-tab, and title bar.
        #[cfg(target_os = "windows")]
        {
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::Win32(win32) = handle.as_raw() {
                    unsafe {
                        let hwnd = win32.hwnd.get() as winapi::shared::windef::HWND;
                        let hinstance = winapi::um::libloaderapi::GetModuleHandleW(std::ptr::null());

                        // Load large icon (32x32) from the .exe resource embedded by winres.
                        // MAKEINTRESOURCEW(1) = 1usize cast to LPCWSTR.
                        let resource_id = 1usize as winapi::shared::ntdef::LPCWSTR;
                        let hicon_big = winapi::um::winuser::LoadImageW(
                            hinstance,
                            resource_id,
                            winapi::um::winuser::IMAGE_ICON,
                            32, 32,
                            winapi::um::winuser::LR_DEFAULTCOLOR,
                        ) as winapi::shared::windef::HICON;

                        // Load small icon (16x16) for the title bar / taskbar small slot.
                        let hicon_small = winapi::um::winuser::LoadImageW(
                            hinstance,
                            resource_id,
                            winapi::um::winuser::IMAGE_ICON,
                            16, 16,
                            winapi::um::winuser::LR_DEFAULTCOLOR,
                        ) as winapi::shared::windef::HICON;

                        if !hicon_big.is_null() {
                            // Patch the window CLASS so the taskbar picks it up.
                            winapi::um::winuser::SetClassLongPtrW(
                                hwnd,
                                winapi::um::winuser::GCLP_HICON,
                                hicon_big as winapi::shared::basetsd::LONG_PTR,
                            );
                            // Also set on the window instance.
                            winapi::um::winuser::SendMessageW(
                                hwnd,
                                winapi::um::winuser::WM_SETICON,
                                winapi::um::winuser::ICON_BIG as usize,
                                hicon_big as winapi::shared::minwindef::LPARAM,
                            );
                        }
                        if !hicon_small.is_null() {
                            winapi::um::winuser::SetClassLongPtrW(
                                hwnd,
                                winapi::um::winuser::GCLP_HICONSM,
                                hicon_small as winapi::shared::basetsd::LONG_PTR,
                            );
                            winapi::um::winuser::SendMessageW(
                                hwnd,
                                winapi::um::winuser::WM_SETICON,
                                winapi::um::winuser::ICON_SMALL as usize,
                                hicon_small as winapi::shared::minwindef::LPARAM,
                            );
                        }
                    }
                }
            }
        }
        
        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        
        let surface = instance.create_surface(window.clone()).unwrap();
        
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();

        // Log adapter info to help diagnose GPU-specific issues
        let adapter_info = adapter.get_info();
        log::warn!(
            "GPU adapter: {} ({:?}) driver: {}",
            adapter_info.name, adapter_info.backend, adapter_info.driver
        );

        // Query what the adapter actually supports and clamp our requests to those limits.
        // Hardcoding 512 MB will panic on AMD drivers that report lower limits.
        let adapter_limits = adapter.limits();
        let storage_binding_limit = adapter_limits.max_storage_buffer_binding_size
            .min(512 * 1024 * 1024);
        let buffer_size_limit = adapter_limits.max_buffer_size
            .min(512 * 1024 * 1024);
        log::info!(
            "Adapter limits — max_storage_buffer_binding_size: {} MB, max_buffer_size: {} MB",
            adapter_limits.max_storage_buffer_binding_size / (1024 * 1024),
            adapter_limits.max_buffer_size / (1024 * 1024),
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Bio-Spheres Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // Cell state write bind group uses up to 40 storage buffers on Vulkan/DX12.
                    // Metal (macOS) hard-caps at 31 — requesting 40 panics request_device on Metal.
                    // Use backend to pick the right value; never use adapter.limits() as the
                    // requested value since some drivers report low numbers that would cause
                    // wgpu to validate every bind group against that cap, dropping FPS.
                    max_storage_buffers_per_shader_stage: match adapter_info.backend {
                        wgpu::Backend::Metal => 31,
                        _ => 40,
                    },
                    // Clamp to what the adapter actually supports — requesting more than the
                    // adapter limit causes request_device to fail (panic on .unwrap()).
                    max_storage_buffer_binding_size: storage_binding_limit,
                    max_buffer_size: buffer_size_limit,
                    ..wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            },
        ))
        .expect("Failed to create wgpu device — check log for adapter limits");
        
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer Bgra8UnormSrgb (AMD/Vulkan native) then any sRGB, then driver default.
        // Do NOT search for sRGB generically — on AMD, formats[0] may be Bgra8UnormSrgb
        // while the search finds Rgba8UnormSrgb first, causing a pipeline/pass mismatch
        // because all pipelines compile with the chosen format but the swapchain images
        // come back as the driver's native format.
        let surface_format = if surface_caps.formats.contains(&wgpu::TextureFormat::Bgra8UnormSrgb) {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else if surface_caps.formats.contains(&wgpu::TextureFormat::Rgba8UnormSrgb) {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            surface_caps.formats[0]
        };
        log::warn!("Surface format selected: {:?} (available: {:?})", surface_format, surface_caps.formats);
        
        // Ensure we have non-zero dimensions before configuring
        let width = size.width.max(1);
        let height = size.height.max(1);
        
        // Prefer Opaque alpha mode to prevent the OS compositor from blending the
        // swapchain against the desktop. Using a non-Opaque mode (e.g. PreMultiplied)
        // causes pixels with alpha < 1.0 to show desktop content through the window,
        // which appears as a random semi-transparent ghost overlay of the scene.
        let alpha_mode = if surface_caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            surface_caps.alpha_modes[0]
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width,
            height,
            // Immediate (no vsync) is not guaranteed on all drivers — fall back to Fifo.
            present_mode: if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &config);
        
        // Create scene manager (starts with preview scene)
        let scene_manager = SceneManager::new(&device, &queue, &config);
        
        // Create dock manager for UI layout persistence
        let dock_manager = DockManager::new();
        
        // Create UI system
        let ui = UiSystem::new(&device, surface_format, &window);
        
        self.app = Some(App::new(window, surface, device, queue, config, scene_manager, dock_manager, ui));
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let Some(app) = &mut self.app else { return };
        
        if window_id != app.window().id() {
            return;
        }
        
        if !app.handle_event(&event) {
            event_loop.exit();
        }
    }
    
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(app) = &self.app {
            // Limit to 60fps by waiting until next frame time
            let next_frame = app.next_frame_time();
            let now = std::time::Instant::now();
            
            if now >= next_frame {
                // Time to render - request redraw immediately
                app.request_redraw();
                event_loop.set_control_flow(ControlFlow::Poll);
            } else {
                // Wait until next frame time
                event_loop.set_control_flow(ControlFlow::WaitUntil(next_frame));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
    }
}

pub fn run() {
    // ── Self-replace (update) ─────────────────────────────────────────────────
    // If the user ran a new bio-spheres.exe from a different location than the
    // previous install, this copies it to the canonical path and relaunches.
    // Must run before logging so it can exit cleanly if a relaunch happens.
    crate::updater::run_self_replace();

    // ── Logging setup ────────────────────────────────────────────────────────
    // Write logs to the AppData config directory so they're always findable
    // regardless of where the exe is launched from.
    let log_path = crate::app_dirs::log_file();

    // Open (or create) the log file, truncating it each run so it stays small.
    let log_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&log_path)
        .expect("Failed to open log file");

    // Build env_logger to write to the file.
    // Level: WARN by default; set RUST_LOG=info or RUST_LOG=debug to get more.
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env() // still respect RUST_LOG if set
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .format_timestamp_secs()
        .init();

    // ── Panic hook ───────────────────────────────────────────────────────────
    // Capture panics into the log file and show a message box so the tester
    // knows the crash happened and where to find the log.
    let log_path_for_hook = log_path.clone();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!(
            "PANIC: {}\n\nLog file: {}\n\nPlease send the log file to the developer.",
            info,
            log_path_for_hook.display(),
        );
        log::error!("{}", msg);

        // Flush by dropping — env_logger flushes on drop but we can't drop it here,
        // so write directly to the file as a fallback.
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path_for_hook)
        {
            use std::io::Write;
            let _ = writeln!(f, "\n{}", msg);
        }

        // Show a Windows message box so the tester sees the crash immediately.
        #[cfg(target_os = "windows")]
        {
            use std::ffi::OsStr;
            use std::os::windows::ffi::OsStrExt;
            let title: Vec<u16> = OsStr::new("Bio-Spheres Crashed")
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let body_str = format!(
                "Bio-Spheres has crashed.\n\nPlease send this file to the developer:\n{}\n\nError: {}",
                log_path_for_hook.display(),
                info,
            );
            let body: Vec<u16> = OsStr::new(&body_str)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            unsafe {
                winapi::um::winuser::MessageBoxW(
                    std::ptr::null_mut(),
                    body.as_ptr(),
                    title.as_ptr(),
                    winapi::um::winuser::MB_OK | winapi::um::winuser::MB_ICONERROR,
                );
            }
        }
    }));

    log::warn!("Bio-Spheres starting — log: {}", log_path.display());

    // Migrate config files: write defaults for first-time users, and add any
    // new keys introduced in this version without touching existing user values.
    crate::updater::migrate_config_files();

    let event_loop = EventLoop::new().unwrap();
    let mut state = AppState { app: None };

    event_loop.run_app(&mut state).unwrap();
}


