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
        }
    }
    
    pub fn window(&self) -> &Window {
        &self.window
    }
    
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        // First, let egui handle the event
        let egui_response = self.ui.handle_event(&self.window, event);
        
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
                    let menu = &mut self.editor_state.radial_menu;
                    if menu.visible && *button == MouseButton::Left && *state == ElementState::Pressed {
                        // Click while menu is open selects the hovered tool
                        menu.close(true);
                        // Hide cursor if a tool is now active
                        let hide_cursor = self.editor_state.radial_menu.active_tool != crate::ui::radial_menu::RadialTool::None;
                        self.window.set_cursor_visible(!hide_cursor);
                        self.window.request_redraw();
                        return true;
                    }
                    
                    // Handle Insert tool click
                    if !menu.visible 
                        && self.editor_state.radial_menu.active_tool == crate::ui::radial_menu::RadialTool::Insert
                        && *button == MouseButton::Left 
                        && *state == ElementState::Pressed
                        && !self.ui.wants_pointer_input()
                    {
                        if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                            let world_pos = gpu_scene.screen_to_world(
                                self.mouse_position.0,
                                self.mouse_position.1,
                            );
                            if let Some(_idx) = gpu_scene.insert_cell_from_genome(world_pos, &self.working_genome) {
                                log::info!("Inserted cell at {:?}, total: {}", world_pos, gpu_scene.canonical_state.cell_count);
                            }
                        }
                        self.window.request_redraw();
                        return true;
                    }
                }
                
                // Only pass to camera if egui doesn't want the input
                if !self.ui.wants_pointer_input() {
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
                }
                
                // Only pass to camera if egui doesn't want the input
                if !self.ui.wants_pointer_input() {
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
                }
                
                // Only pass to camera if egui doesn't want the input
                if !self.ui.wants_keyboard_input() {
                    self.scene_manager.active_scene_mut().camera_mut().handle_keyboard(event);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            _ => {}
        }
        
        // Request repaint if egui needs it
        if egui_response.repaint {
            self.window.request_redraw();
        }
        
        true
    }
    
    fn render(&mut self) {
        // Don't render if surface has zero dimensions
        if self.config.width == 0 || self.config.height == 0 {
            return;
        }
        
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32();
        self.last_render_time = now;
        
        // Update performance metrics
        self.performance.update(dt);
        
        // Update camera and scene
        self.scene_manager.active_scene_mut().camera_mut().update(dt);
        self.scene_manager.update(dt);
        
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
        
        // Render 3D scene first (pass cell type visuals from editor state)
        let cell_type_visuals = &self.editor_state.cell_type_visuals;
        self.scene_manager.render(&self.device, &self.queue, &view, Some(cell_type_visuals));
        
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
            
            let output = self.ui.end_frame(
                &mut self.dock_manager,
                &mut self.working_genome,
                &mut self.editor_state,
                &self.scene_manager,
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
                    // Reset the GPU scene
                    if let Some(gpu_scene) = self.scene_manager.gpu_scene_mut() {
                        gpu_scene.reset();
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
                
                self.scene_manager.switch_mode(requested_mode, &self.device, &self.queue, &self.config);
                self.dock_manager.switch_mode(requested_mode);
                // Reset cursor visibility and radial menu state when switching modes
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
            self.scene_manager.switch_mode(dock_mode, &self.device, &self.queue, &self.config);
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
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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
        event_loop.set_control_flow(ControlFlow::Poll);
        if let Some(app) = &self.app {
            app.request_redraw();
        }
    }
}

pub fn run() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let mut state = AppState { app: None };
    
    event_loop.run_app(&mut state).unwrap();
}
