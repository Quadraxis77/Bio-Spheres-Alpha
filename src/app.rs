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
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
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
            surface,
            device,
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
                [self.ui.state.world_settings.gravity_x, self.ui.state.world_settings.gravity_y, self.ui.state.world_settings.gravity_z],
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
                
                // Sync fluid voxel visibility toggle
                gpu_scene.show_fluid_voxels = self.editor_state.fluid_show_test_voxels;
                
                // Sync GPU density mesh visibility toggle
                gpu_scene.show_gpu_density_mesh = self.editor_state.fluid_show_mesh;
                
                // Update GPU surface nets params when changed
                if self.editor_state.fluid_mesh_needs_regen || self.editor_state.fluid_mesh_params_dirty {
                    if let Some(ref mut surface_nets) = gpu_scene.gpu_surface_nets {
                        // Update iso level
                        surface_nets.set_iso_level(&self.queue, self.editor_state.fluid_iso_level);
                        
                        // Update render params
                        let params = crate::rendering::DensityMeshParams {
                            base_color: [0.2, 0.5, 0.9], // Default water blue color
                            ambient: self.editor_state.fluid_ambient,
                            diffuse: self.editor_state.fluid_diffuse,
                            specular: self.editor_state.fluid_specular,
                            shininess: self.editor_state.fluid_shininess,
                            fresnel: self.editor_state.fluid_fresnel,
                            fresnel_power: self.editor_state.fluid_fresnel_power,
                            rim: self.editor_state.fluid_rim,
                            reflection: self.editor_state.fluid_reflection,
                            alpha: self.editor_state.fluid_alpha,
                        };
                        surface_nets.update_render_params(&self.queue, &params);
                    }
                    self.editor_state.fluid_mesh_needs_regen = false;
                    self.editor_state.fluid_mesh_params_dirty = false;
                }
            }
        }
        
        // Auto-save dock layouts periodically
        self.dock_manager.auto_save();
        
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Outdated) => {
                // Surface is outdated, reconfigure it
                self.surface.configure(&self.device, &self.config);
                self.surface.get_current_texture().unwrap()
            }
            Err(e) => {
                log::error!("Failed to get surface texture: {:?}", e);
                return;
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
            gpu_scene.gravity_dir = [
                self.ui.state.world_settings.gravity_x,
                self.ui.state.world_settings.gravity_y,
                self.ui.state.world_settings.gravity_z,
            ];

            // Apply lateral flow probabilities from UI
            gpu_scene.lateral_flow_probabilities = self.editor_state.fluid_lateral_flow_probabilities;

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
                    
                    // Sync simulation time to UI slider (when not dragging)
                    if !self.editor_state.time_slider_dragging {
                        let sim_time = preview_scene.get_time_for_ui();
                        // Only update if significantly different (avoid jitter)
                        if (self.editor_state.time_value - sim_time).abs() > 0.1 {
                            self.editor_state.time_value = sim_time.clamp(0.0, self.editor_state.max_preview_duration);
                        }
                    }
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
                }
            }
            
            // Sync genome changes to GPU scene if in GPU mode
            // This ensures cell_type changes are reflected in the mode_cell_types buffer
            if current_mode == crate::ui::types::SimulationMode::Gpu {
                if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                    gpu_scene.update_genome(&self.device, &self.queue, &self.working_genome);
                }
            }
            
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
                    // Cell state write bind group uses 18 storage buffers (rotations, genome_mode_data, max_cell_sizes, mode_properties, etc.)
                    max_storage_buffers_per_shader_stage: 24,
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
