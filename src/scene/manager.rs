//! Scene manager for switching between simulation modes.
//!
//! Handles creation, switching, and lifecycle of Preview and GPU scenes.

use crate::scene::{GpuScene, PreviewScene, Scene};
use crate::ui::SimulationMode;

/// Manages the active scene and handles scene switching.
pub struct SceneManager {
    /// Current simulation mode
    current_mode: SimulationMode,
    /// Preview scene (lazy initialized)
    preview_scene: Option<PreviewScene>,
    /// GPU scene (lazy initialized)
    gpu_scene: Option<GpuScene>,
}

impl SceneManager {
    /// Create a new scene manager with the preview scene active.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        // Start with preview scene
        let preview_scene = Some(PreviewScene::new(device, queue, config));

        Self {
            current_mode: SimulationMode::Preview,
            preview_scene,
            gpu_scene: None,
        }
    }

    /// Get the current simulation mode.
    pub fn current_mode(&self) -> SimulationMode {
        self.current_mode
    }

    /// Switch to a different simulation mode.
    ///
    /// Creates the target scene if it doesn't exist yet.
    pub fn switch_mode(
        &mut self,
        mode: SimulationMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        world_diameter: f32,
        cell_capacity: u32,
    ) -> bool {
        self.switch_mode_with_capacity(mode, device, queue, config, world_diameter, cell_capacity)
    }
    
    /// Switch to a different simulation mode with specified GPU scene capacity.
    pub fn switch_mode_with_capacity(
        &mut self,
        mode: SimulationMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        world_diameter: f32,
        cell_capacity: u32,
    ) -> bool {
        if mode == self.current_mode {
            return false;
        }

        log::info!(
            "Switching from {} to {}",
            self.current_mode.display_name(),
            mode.display_name()
        );

        // Ensure target scene exists
        match mode {
            SimulationMode::Preview => {
                if self.preview_scene.is_none() {
                    self.preview_scene = Some(PreviewScene::new(device, queue, config));
                }
            }
            SimulationMode::Gpu => {
                if self.gpu_scene.is_none() {
                    let mut gpu_scene = GpuScene::with_capacity(device, queue, config, cell_capacity);
                    // Initialize cave system automatically
                    let cave_initialized = gpu_scene.initialize_cave_system(device, config.format, world_diameter);
                    
                    // Initialize fluid system automatically
                    // Create camera bind group layout for voxel rendering
                    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Voxel Camera Layout"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                    
                    let fluid_initialized = gpu_scene.initialize_fluid_system(
                        device,
                        config.format,
                        &camera_bind_group_layout,
                    );
                    
                    if fluid_initialized {
                        log::info!("Fluid system auto-initialized on GPU scene creation");
                        // Generate test voxels
                        gpu_scene.generate_test_voxels(queue);
                    }
                    
                    // Initialize GPU surface nets for density mesh rendering
                    gpu_scene.initialize_gpu_surface_nets(device, config.format);

                    // Initialize fluid simulator with test water sphere
                    gpu_scene.initialize_fluid_simulator(device, queue, config.format);

                    self.gpu_scene = Some(gpu_scene);
                    self.current_mode = mode;
                    return cave_initialized; // Return true if cave was just initialized
                }
            }
        }

        self.current_mode = mode;
        false
    }
    
    /// Recreate the GPU scene with a new capacity.
    /// Used when switching between normal and point cloud mode.
    pub fn recreate_gpu_scene_with_capacity(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        world_diameter: f32,
        capacity: u32,
    ) {
        // Only recreate if capacity actually changed
        if let Some(ref scene) = self.gpu_scene {
            if scene.capacity() == capacity {
                return;
            }
        }
        
        log::info!("Recreating GPU scene with capacity: {}", capacity);
        let mut gpu_scene = GpuScene::with_capacity(device, queue, config, capacity);
        // Initialize cave system automatically
        let _cave_initialized = gpu_scene.initialize_cave_system(device, config.format, world_diameter);
        
        // Initialize fluid system automatically
        // Create camera bind group layout for voxel rendering
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Voxel Camera Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let fluid_initialized = gpu_scene.initialize_fluid_system(
            device,
            config.format,
            &camera_bind_group_layout,
        );
        
        if fluid_initialized {
            log::info!("Fluid system auto-initialized on GPU scene recreation");
            // Generate test voxels
            gpu_scene.generate_test_voxels(queue);
        }
        
        // Initialize GPU surface nets for density mesh rendering
        gpu_scene.initialize_gpu_surface_nets(device, config.format);

        // Initialize fluid simulator with test water sphere
        gpu_scene.initialize_fluid_simulator(device, queue, config.format);

        self.gpu_scene = Some(gpu_scene);
    }

    /// Get a reference to the active scene.
    pub fn active_scene(&self) -> &dyn Scene {
        match self.current_mode {
            SimulationMode::Preview => self
                .preview_scene
                .as_ref()
                .expect("Preview scene should exist"),
            SimulationMode::Gpu => self.gpu_scene.as_ref().expect("GPU scene should exist"),
        }
    }

    /// Get a mutable reference to the active scene.
    pub fn active_scene_mut(&mut self) -> &mut dyn Scene {
        match self.current_mode {
            SimulationMode::Preview => self
                .preview_scene
                .as_mut()
                .expect("Preview scene should exist"),
            SimulationMode::Gpu => self.gpu_scene.as_mut().expect("GPU scene should exist"),
        }
    }

    /// Get a reference to the preview scene if it exists.
    pub fn preview_scene(&self) -> Option<&PreviewScene> {
        self.preview_scene.as_ref()
    }

    /// Get a mutable reference to the preview scene if it exists.
    pub fn preview_scene_mut(&mut self) -> Option<&mut PreviewScene> {
        self.preview_scene.as_mut()
    }

    /// Get a reference to the GPU scene if it exists.
    pub fn gpu_scene(&self) -> Option<&GpuScene> {
        self.gpu_scene.as_ref()
    }

    /// Get a mutable reference to the GPU scene if it exists.
    pub fn gpu_scene_mut(&mut self) -> Option<&mut GpuScene> {
        self.gpu_scene.as_mut()
    }

    /// Get a reference to the preview scene for UI access.
    /// Returns None if preview scene doesn't exist or current mode is not Preview.
    pub fn get_preview_scene(&self) -> Option<&PreviewScene> {
        if self.current_mode == SimulationMode::Preview {
            self.preview_scene.as_ref()
        } else {
            None
        }
    }

    /// Get a mutable reference to the preview scene for UI access.
    /// Returns None if preview scene doesn't exist or current mode is not Preview.
    pub fn get_preview_scene_mut(&mut self) -> Option<&mut PreviewScene> {
        if self.current_mode == SimulationMode::Preview {
            self.preview_scene.as_mut()
        } else {
            None
        }
    }

    /// Handle window resize for all existing scenes.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(scene) = &mut self.preview_scene {
            scene.resize(device, width, height);
        }
        if let Some(scene) = &mut self.gpu_scene {
            scene.resize(device, width, height);
        }
    }

    /// Update the active scene.
    pub fn update(&mut self, dt: f32) {
        self.active_scene_mut().update(dt);
    }

    /// Render the active scene.
    pub fn render(
        &mut self, 
        device: &wgpu::Device, 
        queue: &wgpu::Queue, 
        view: &wgpu::TextureView, 
        cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>, 
        world_diameter: f32,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        lod_debug_colors: bool,
    ) {
        self.active_scene_mut().render(
            device, 
            queue, 
            view, 
            cell_type_visuals, 
            world_diameter,
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
            lod_debug_colors,
        );
    }

    /// Insert a cell from genome using GPU operations (GPU scene only).
    /// 
    /// This method provides access to GPU-specific cell insertion that requires
    /// device, encoder, and queue parameters for direct GPU buffer operations.
    /// For preview scene, this method does nothing and returns None.
    pub fn insert_cell_from_genome_gpu(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        world_position: glam::Vec3,
        genome: &crate::genome::Genome,
    ) -> Option<usize> {
        match self.current_mode {
            crate::ui::SimulationMode::Gpu => {
                if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                    gpu_scene.insert_cell_from_genome(device, encoder, queue, world_position, genome)
                } else {
                    None
                }
            }
            crate::ui::SimulationMode::Preview => {
                // Preview scene doesn't support GPU operations
                None
            }
        }
    }

    /// Extract cell data using GPU operations (GPU scene only).
    /// 
    /// This method provides access to GPU-specific cell data extraction that requires
    /// device, encoder, and queue parameters for GPU compute shader execution.
    /// For preview scene, this method does nothing.
    pub fn extract_cell_data_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cell_index: u32,
    ) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.extract_cell_data(device, queue, encoder, cell_index);
            }
        }
    }

    /// Update cell position using GPU operations (GPU scene only).
    /// 
    /// This method provides access to GPU-specific position updates that operate
    /// directly on GPU buffers without CPU canonical state involvement.
    /// For preview scene, this method does nothing.
    pub fn update_cell_position_gpu(&mut self, cell_index: u32, new_position: glam::Vec3) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.update_cell_position_gpu(cell_index, new_position);
            }
        }
    }

    /// Start GPU spatial query for cell selection (GPU scene only).
    /// 
    /// This method queues a GPU spatial query to find the closest cell to the given screen position.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// For preview scene, this method does nothing.
    pub fn start_cell_selection_query(&mut self, screen_x: f32, screen_y: f32) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.start_cell_selection_query(screen_x, screen_y);
            }
        }
    }

    /// Start GPU spatial query for drag tool (GPU scene only).
    /// 
    /// This method queues a GPU spatial query to find the closest cell for dragging.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// For preview scene, this method does nothing.
    pub fn start_drag_selection_query(&mut self, screen_x: f32, screen_y: f32) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.start_drag_selection_query(screen_x, screen_y);
            }
        }
    }

    /// Start GPU spatial query for remove tool (GPU scene only).
    /// 
    /// This method queues a GPU spatial query to find the closest cell for removal.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// For preview scene, this method does nothing.
    pub fn start_remove_tool_query(&mut self, screen_x: f32, screen_y: f32) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.start_remove_tool_query(screen_x, screen_y);
            }
        }
    }

    /// Start GPU spatial query for boost tool (GPU scene only).
    /// 
    /// This method queues a GPU spatial query to find the closest cell for boosting.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// For preview scene, this method does nothing.
    pub fn start_boost_tool_query(&mut self, screen_x: f32, screen_y: f32) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.start_boost_tool_query(screen_x, screen_y);
            }
        }
    }

    /// Poll for tool operation results (GPU scene only).
    /// 
    /// This method checks for completed spatial query results and updates tool states.
    /// It should be called each frame to process async tool operation completions.
    /// For preview scene, this method does nothing.
    pub fn poll_tool_operation_results(&mut self, radial_menu: &mut crate::ui::radial_menu::RadialMenuState, drag_distance: &mut f32, _queue: &wgpu::Queue) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                // First poll for spatial query results from GPU
                gpu_scene.poll_spatial_query_results();
                
                // Then process the results for each tool
                gpu_scene.poll_inspect_tool_results(radial_menu);
                gpu_scene.poll_drag_tool_results(radial_menu, drag_distance);
                gpu_scene.poll_remove_tool_results();
                gpu_scene.poll_boost_tool_results();
            }
        }
    }

    /// Convert screen coordinates to world position (GPU scene only).
    /// 
    /// This method provides access to GPU scene's screen-to-world conversion for tool operations.
    /// For preview scene, returns a default position.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> glam::Vec3 {
        match self.current_mode {
            crate::ui::SimulationMode::Gpu => {
                if let Some(gpu_scene) = self.gpu_scene.as_ref() {
                    gpu_scene.screen_to_world(screen_x, screen_y)
                } else {
                    glam::Vec3::ZERO
                }
            }
            crate::ui::SimulationMode::Preview => {
                // Preview scene doesn't have screen-to-world conversion for tools
                glam::Vec3::ZERO
            }
        }
    }

    /// Convert screen coordinates to world position at distance (GPU scene only).
    /// 
    /// This method provides access to GPU scene's screen-to-world conversion at a specific distance.
    /// For preview scene, returns a default position.
    pub fn screen_to_world_at_distance(&self, screen_x: f32, screen_y: f32, distance: f32) -> glam::Vec3 {
        match self.current_mode {
            crate::ui::SimulationMode::Gpu => {
                if let Some(gpu_scene) = self.gpu_scene.as_ref() {
                    gpu_scene.screen_to_world_at_distance(screen_x, screen_y, distance)
                } else {
                    glam::Vec3::ZERO
                }
            }
            crate::ui::SimulationMode::Preview => {
                // Preview scene doesn't have screen-to-world conversion for tools
                glam::Vec3::ZERO
            }
        }
    }

    /// Update gizmo configuration for all existing scenes.
    pub fn update_gizmo_config(&mut self, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        // Only preview scene has gizmos
        if let Some(scene) = &mut self.preview_scene {
            scene.update_gizmo_config(editor_state);
        }
    }

    /// Update split ring configuration for all existing scenes.
    pub fn update_split_ring_config(&mut self, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        // Only preview scene has split rings
        if let Some(scene) = &mut self.preview_scene {
            scene.update_split_ring_config(editor_state);
        }
    }
}
