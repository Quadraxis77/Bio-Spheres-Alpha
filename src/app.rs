use crate::preview::PreviewScene;
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
    scene: PreviewScene,
    last_render_time: std::time::Instant,
    frame_count: u32,
    fps_timer: std::time::Instant,
}

impl App {
    pub fn new(window: Arc<Window>, surface: wgpu::Surface<'static>, device: wgpu::Device, queue: wgpu::Queue, config: wgpu::SurfaceConfiguration, scene: PreviewScene) -> Self {
        Self {
            window,
            surface,
            device,
            queue,
            config,
            scene,
            last_render_time: std::time::Instant::now(),
            frame_count: 0,
            fps_timer: std::time::Instant::now(),
        }
    }
    
    pub fn window(&self) -> &Window {
        &self.window
    }
    
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                return false;
            }
            WindowEvent::Resized(physical_size) => {
                self.config.width = physical_size.width;
                self.config.height = physical_size.height;
                self.surface.configure(&self.device, &self.config);
                self.scene.resize(&self.device, physical_size.width, physical_size.height);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.scene.camera.handle_mouse_button(*button, *state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.scene.camera.handle_mouse_move(*position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.scene.camera.handle_scroll(*delta);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.scene.camera.handle_keyboard(event);
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            _ => {}
        }
        true
    }
    
    fn render(&mut self) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_render_time).as_secs_f32();
        self.last_render_time = now;
        
        self.scene.camera.update(dt);
        self.scene.update(dt);
        
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.scene.render(&self.device, &self.queue, &view);
        
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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
            },
            None,
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
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &config);
        
        let scene = PreviewScene::new(&device, &queue, &config);
        
        self.app = Some(App::new(window, surface, device, queue, config, scene));
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
