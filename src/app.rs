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

use crate::scene::SceneManager;
use crate::ui::{DockManager, PerformanceMetrics, UiSystem};
use egui_wgpu::ScreenDescriptor;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

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
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
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
        ui: UiSystem,
    ) -> Self {
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
            working_genome: crate::genome::Genome::default(),
            performance: PerformanceMetrics::new(),
            next_frame_time: std::time::Instant::now(),
            test_signal_emissions: Vec::new(),
            test_signals_changed: false,
            device,
            surface,
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
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                // Handle cell click in Preview mode to select mode
                if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Preview
                    && *button == MouseButton::Left
                    && *state == ElementState::Pressed
                    && !self.ui.wants_pointer_input()
                {
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
                            preview_scene.selected_mode_index = Some(mode_idx);
                            self.editor_state.selected_mode_index = mode_idx;
                            log::info!("Preview cell click: selected mode {}", mode_idx);
                        }
                    }
                    self.window.request_redraw();
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
            WindowEvent::CursorMoved { position, .. } => {
                // Track mouse position for tool interactions
                self.mouse_position = (position.x as f32, position.y as f32);
                
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
                if !self.ui.wants_pointer_input() {
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
        
        let dt = now.duration_since(self.last_render_time).as_secs_f32();
        self.last_render_time = now;
        
        // Schedule next frame for 60fps (16.67ms)
        self.next_frame_time = now + std::time::Duration::from_micros(16_667);
        
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
        self.scene_manager.update(dt);
        
        // Poll for async tool operation results (GPU mode only)
        if self.scene_manager.current_mode() == crate::ui::types::SimulationMode::Gpu {
            // Poll for tool operation results and update radial menu state
            self.scene_manager.poll_tool_operation_results(&mut self.editor_state.radial_menu, &mut self.editor_state.drag_distance, &self.queue);
            
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
                        _pad2: 0.0,
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
            gpu_scene.water_drag_strength = self.ui.state.world_settings.water_drag_strength;
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
        // This keeps the genome available for GPU scene cell insertion
        let egui_output = {
            let mut dummy_camera = crate::ui::camera::CameraController::new();
            
            // Get real data if in Preview mode
            if current_mode == crate::ui::types::SimulationMode::Preview {
                if let Some(preview_scene) = self.scene_manager.get_preview_scene() {
                    // Copy the genome data for editing
                    self.working_genome = preview_scene.genome.clone();
                    
                    // One-way sync: read simulation's actual time for progress bar display only
                    // Never write back to time_value — the slider is purely user-driven
                    self.editor_state.resim_display_time = preview_scene.get_time_for_ui();
                }
            }
            
            // In GPU mode, sync working_genome FROM GPU scene if it has genomes
            // This ensures the UI shows the current genome state
            // Note: We only do this once when switching to GPU mode, not every frame
            // to allow UI edits to persist
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
                    preview_scene.update_genome(&self.working_genome);
                    
                    // Sync time slider to simulation (when dragging or changed)
                    preview_scene.sync_time_from_ui(
                        self.editor_state.time_value,
                        self.editor_state.max_preview_duration,
                        self.editor_state.time_slider_dragging,
                    );
                    
                    // Bidirectional sync: genome panel selection → preview highlight
                    preview_scene.selected_mode_index = Some(self.editor_state.selected_mode_index);
                    
                    // Sync test signals to preview scene (must happen before resimulation trigger)
                    preview_scene.test_signals = self.test_signal_emissions.clone();
                    
                    // Trigger resimulation if test signals changed
                    if self.test_signals_changed {
                        let current_time = preview_scene.state.display_time;
                        preview_scene.state.seek_to_time(current_time);
                        self.test_signals_changed = false;
                    }
                }
            }
            
            // NOTE: In GPU mode, we do NOT call update_genome here because that would
            // modify the existing genome in place, changing the behavior of all existing
            // cells using that genome. Instead, edited genomes are added as NEW genomes
            // during cell insertion (via add_genome in insert_cell_from_genome), preserving
            // existing cells' behavior while allowing new cells to use the edited settings.
            
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
        
        // Submit egui commands
        self.queue.submit(std::iter::once(encoder.finish()));
        
        output.present();
        
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
        
        let window_attributes = Window::default_attributes()
            .with_title("Bio-Spheres Preview")
            .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080));
        
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        window.set_maximized(true);
        
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
        
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Bio-Spheres Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // Cell state write bind group uses 35 storage buffers (0-34) after splitting
                    // genome_mode_data and mode_properties into 5 sub-buffers each
                    max_storage_buffers_per_shader_stage: 40,
                    // adhesion_connections at 200k cells = 200k*10*104 = ~208 MB, exceeds the
                    // default 128 MB limit. Request 512 MB to cover max capacity with headroom.
                    max_storage_buffer_binding_size: 512 * 1024 * 1024,
                    max_buffer_size: 512 * 1024 * 1024,
                    ..wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            },
        ))
        .unwrap();
        
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        // Ensure we have non-zero dimensions before configuring
        let width = size.width.max(1);
        let height = size.height.max(1);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
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
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let mut state = AppState { app: None };
    
    event_loop.run_app(&mut state).unwrap();
}
