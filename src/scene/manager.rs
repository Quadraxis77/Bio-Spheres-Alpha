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

#[cfg(all(test, feature = "vr"))]
mod vr_tests {
    use super::*;

    #[test]
    fn headset_device_change_retains_gpu_cells_preview_checkpoints_and_cameras() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let new_device = || {
            pollster::block_on(
                adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 64,
                        max_storage_buffer_binding_size: adapter
                            .limits()
                            .max_storage_buffer_binding_size
                            .min(512 * 1024 * 1024),
                        max_buffer_size: adapter.limits().max_buffer_size.min(512 * 1024 * 1024),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
            )
            .unwrap()
        };
        let (old_device, old_queue) = new_device();
        let (device, queue) = new_device();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 64,
            height: 64,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        let mut old = SceneManager::new(&old_device, &old_queue, &config);
        let preview = old.preview_scene.as_mut().unwrap();
        preview.camera.center = glam::vec3(1.0, 2.0, 3.0);
        preview.state.display_time = 3.0;
        preview
            .state
            .checkpoints
            .push((2.0, preview.state.display_state.clone()));
        preview.paused = true;
        let mut gpu =
            GpuScene::with_capacity_and_radius(&old_device, &old_queue, &config, 64, 50.0);
        gpu.camera.center = glam::vec3(4.0, 5.0, 6.0);
        gpu.queue_cell_insertion(glam::Vec3::ZERO, crate::genome::Genome::default());
        let target = old_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VR device migration regression"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: config.usage,
            view_formats: &[],
        });
        gpu.render(
            &old_device,
            &old_queue,
            &target.create_view(&Default::default()),
            None,
            100.0,
            500.0,
            10.0,
            25.0,
            50.0,
            false,
            0.1,
        );
        gpu.set_paused(true);
        let before = gpu.save_snapshot(&old_device, &old_queue).unwrap();
        assert!(before.live_cell_count > 0);
        old.gpu_scene = Some(gpu);
        old.current_mode = SimulationMode::Gpu;
        let new = old
            .recreate_on_device(
                &old_device,
                &old_queue,
                &device,
                &queue,
                &config,
                &Default::default(),
            )
            .unwrap();
        let after = new
            .gpu_scene
            .as_ref()
            .unwrap()
            .save_snapshot(&device, &queue)
            .unwrap();
        assert_eq!(before.live_cell_count, after.live_cell_count);
        assert_eq!(before.positions_and_mass, after.positions_and_mass);
        assert_eq!(before.genomes_yaml, after.genomes_yaml);
        assert_eq!(before.current_time, after.current_time);
        assert!(new.gpu_scene.as_ref().unwrap().is_paused());
        assert_eq!(
            new.gpu_scene.as_ref().unwrap().camera.center,
            glam::vec3(4.0, 5.0, 6.0)
        );
        let preview = new.preview_scene.as_ref().unwrap();
        assert_eq!(preview.state.display_time, 3.0);
        assert_eq!(preview.state.checkpoints[0].0, 2.0);
        assert_eq!(preview.camera.center, glam::vec3(1.0, 2.0, 3.0));
        assert!(preview.paused);
        assert_eq!(new.current_mode, SimulationMode::Gpu);
    }
}

impl SceneManager {
    /// Rebuild graphics once when a headset connects to a desktop-started game.
    /// Wearing/removing an already connected headset never rebuilds the world.
    #[cfg(feature = "vr")]
    pub(crate) fn recreate_on_device(
        &self,
        old_device: &wgpu::Device,
        old_queue: &wgpu::Queue,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        editor: &crate::ui::panel_context::GenomeEditorState,
    ) -> Result<Self, String> {
        let snapshot = self
            .gpu_scene
            .as_ref()
            .map(|scene| scene.save_snapshot(old_device, old_queue))
            .transpose()
            .map_err(|e| e.to_string())?;
        let mut manager = Self::new(device, queue, config);
        manager.preview_scene = self
            .preview_scene
            .as_ref()
            .map(|scene| scene.recreate_on_device(device, queue, config));
        if let (Some(snapshot), Some(old)) = (snapshot, &self.gpu_scene) {
            manager.switch_mode_with_capacity(
                SimulationMode::Gpu,
                device,
                queue,
                config,
                snapshot.world_radius * 2.0,
                snapshot.capacity,
                editor,
            );
            let scene = manager.gpu_scene.as_mut().unwrap();
            scene
                .restore_from_snapshot(device, queue, &snapshot)
                .map_err(|e| e.to_string())?;
            scene.camera = old.camera.clone();
            scene.camera.interaction_ray = None;
            scene.set_paused(old.is_paused());
        }
        manager.current_mode = self.current_mode;
        Ok(manager)
    }
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
        editor_state: &crate::ui::panel_context::GenomeEditorState,
    ) -> bool {
        self.switch_mode_with_capacity(
            mode,
            device,
            queue,
            config,
            world_diameter,
            cell_capacity,
            editor_state,
        )
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
        editor_state: &crate::ui::panel_context::GenomeEditorState,
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
                    let mut gpu_scene = GpuScene::with_capacity_and_radius(
                        device,
                        queue,
                        config,
                        cell_capacity,
                        world_diameter * 0.5,
                    );
                    // Initialize cave system automatically
                    let cave_initialized = gpu_scene.initialize_cave_system(
                        device,
                        queue,
                        config.format,
                        world_diameter,
                    );

                    // Initialize fluid system automatically
                    // Create camera bind group layout for voxel rendering
                    let camera_bind_group_layout =
                        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        queue,
                        config.format,
                        &camera_bind_group_layout,
                    );

                    if fluid_initialized {
                        log::info!("Fluid system auto-initialized on GPU scene creation");
                        // Generate test voxels
                        gpu_scene.generate_test_voxels(queue);

                        // Update solid mask after fluid system is initialized
                        gpu_scene.update_solid_mask(queue);
                    }

                    // Initialize GPU surface nets for density mesh rendering
                    gpu_scene.initialize_gpu_surface_nets(device, config.format);

                    // Set initial render parameters from editor state
                    if let Some(ref surface_nets) = gpu_scene.gpu_surface_nets {
                        surface_nets.set_initial_params(queue, &editor_state);
                    }

                    // Initialize fluid simulator with test water sphere
                    gpu_scene.initialize_fluid_simulator(device, queue, config.format);

                    // Sync lighting and fog from current editor state so reset doesn't revert visuals
                    gpu_scene.apply_light_params_from_editor(&editor_state);

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
        editor_state: &crate::ui::panel_context::GenomeEditorState,
    ) {
        // Only recreate if capacity or world radius actually changed
        if let Some(ref scene) = self.gpu_scene {
            if scene.capacity() == capacity
                && (scene.config.sphere_radius - world_diameter * 0.5).abs() < 0.5
            {
                return;
            }
        }

        let new_world_radius = world_diameter * 0.5;
        let world_radius_unchanged = self
            .gpu_scene
            .as_ref()
            .map(|scene| (scene.config.sphere_radius - new_world_radius).abs() < 0.5)
            .unwrap_or(false);

        log::info!("Recreating GPU scene with capacity: {}", capacity);
        let mut gpu_scene = GpuScene::with_capacity_and_radius(
            device,
            queue,
            config,
            capacity,
            world_diameter * 0.5,
        );
        // Initialize cave system automatically
        let _cave_initialized =
            gpu_scene.initialize_cave_system(device, queue, config.format, world_diameter);

        // Transfer fluid simulator from old scene if it exists - preserves water state
        // across cells-only resets. Only initialize fresh if there was no prior fluid.
        let had_fluid = self
            .gpu_scene
            .as_ref()
            .map(|s| s.fluid_simulator.is_some())
            .unwrap_or(false);
        // Fluid grids, their solid-mask generator, and the surface extraction
        // coordinates are all radius-dependent. Moving them into a differently
        // sized world mixes the old grid with the new renderer and produces
        // malformed normals/surfaces that appear unnaturally glossy. Preserve
        // these resources only for a capacity-only recreation.
        let can_transfer_fluid = had_fluid && world_radius_unchanged;
        if can_transfer_fluid {
            // Move the fluid simulator and all visual renderers from the old scene.
            // Without this, light field / fog / DOF / sun / voxel systems remain None
            // on the new scene, causing them to disappear after a capacity/radius reset.
            if let Some(ref mut old_scene) = self.gpu_scene {
                gpu_scene.fluid_simulator = old_scene.fluid_simulator.take();
                gpu_scene.fluid_buffers = old_scene.fluid_buffers.take();
                gpu_scene.light_field_system = old_scene.light_field_system.take();
                // The light field's grid origin/cell size are derived from world
                // radius. When the world was resized, refresh those cached values
                // so the carried-over field (and the volumetric fog that samples
                // it) stays aligned with the new world bounds.
                if let Some(ref mut light_field) = gpu_scene.light_field_system {
                    let new_world_radius = world_diameter * 0.5;
                    if (light_field.world_radius() - new_world_radius).abs() > 0.1 {
                        light_field.update_world_radius(new_world_radius);
                    }
                }
                gpu_scene.volumetric_fog_renderer = old_scene.volumetric_fog_renderer.take();
                gpu_scene.dof_renderer = old_scene.dof_renderer.take();
                gpu_scene.sun_renderer = old_scene.sun_renderer.take();
                gpu_scene.voxel_renderer = old_scene.voxel_renderer.take();
                gpu_scene.solid_mask_generator = old_scene.solid_mask_generator.take();
                gpu_scene.moss_system = old_scene.moss_system.take();
                gpu_scene.show_moss = old_scene.show_moss;
                gpu_scene.show_sun = old_scene.show_sun;
                gpu_scene.show_dof = old_scene.show_dof;

                // Cell capacity does not affect the water surface renderer. Keep
                // it (including its density history and reflection cubemap) when
                // only capacity changed so Reset Everything cannot abruptly alter
                // the water's appearance. A radius change still rebuilds it below.
                if world_radius_unchanged {
                    gpu_scene.transfer_water_surface_renderer_from(old_scene);
                }

                // Rebuild bind groups that reference fluid buffers in the new scene
                if let Some(ref simulator) = gpu_scene.fluid_simulator {
                    gpu_scene.cached_bind_groups.update_water_buffers(
                        device,
                        &gpu_scene.gpu_physics_pipelines,
                        &gpu_scene.adhesion_buffers,
                        &gpu_scene.gpu_triple_buffers,
                        simulator.water_grid_params_buffer(),
                        simulator.water_bitfield_buffer(),
                        simulator.water_velocity_buffer(),
                        simulator.ice_bitfield_buffer(),
                        simulator.current_state_buffer(),
                        simulator.temperature_field_buffer(),
                        simulator.geothermal_heat_buffer(),
                    );
                    gpu_scene.refresh_division_audio_bind_group(device);
                }
            }
        } else {
            // No prior fluid, or the world radius changed: initialize a fluid
            // grid and all radius-dependent rendering resources at the new size.
            let camera_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                queue,
                config.format,
                &camera_bind_group_layout,
            );
            if fluid_initialized {
                log::info!("Fluid system auto-initialized on GPU scene recreation");
                gpu_scene.generate_test_voxels(queue);
                gpu_scene.update_solid_mask(queue);
            }
            gpu_scene.initialize_fluid_simulator(device, queue, config.format);
        }

        // Preserve user-configurable settings across scene resets (capacity/radius changes).
        // These are independent of fluid state and must always be carried over.
        if let Some(ref old_scene) = self.gpu_scene {
            gpu_scene.show_adhesion_lines = old_scene.show_adhesion_lines;
            gpu_scene.show_world_sphere = old_scene.show_world_sphere;
            gpu_scene.constraint_iterations = old_scene.constraint_iterations;
            gpu_scene.gravity = old_scene.gravity;
            gpu_scene.gravity_mode = old_scene.gravity_mode;
            gpu_scene.surface_pressure = old_scene.surface_pressure;
            gpu_scene.acceleration_damping = old_scene.acceleration_damping;
            gpu_scene.water_viscosity = old_scene.water_viscosity;
            gpu_scene.solo_metabolism_multiplier = old_scene.solo_metabolism_multiplier;
            gpu_scene.radiation_level = old_scene.radiation_level;
            gpu_scene.subtle_mutations = old_scene.subtle_mutations;
            gpu_scene.lod_scale_factor = old_scene.lod_scale_factor;
            gpu_scene.lod_threshold_low = old_scene.lod_threshold_low;
            gpu_scene.lod_threshold_medium = old_scene.lod_threshold_medium;
            gpu_scene.lod_threshold_high = old_scene.lod_threshold_high;
            gpu_scene.lod_debug_colors = old_scene.lod_debug_colors;
            gpu_scene.lateral_flow_probabilities = old_scene.lateral_flow_probabilities;
            gpu_scene.nutrient_density = old_scene.nutrient_density;
            gpu_scene.nutrient_epoch_duration = old_scene.nutrient_epoch_duration;
            gpu_scene.nutrient_epoch_spacing = old_scene.nutrient_epoch_spacing;
            gpu_scene.nutrient_spawn_end = old_scene.nutrient_spawn_end;
            gpu_scene.nutrient_despawn_start = old_scene.nutrient_despawn_start;
        }

        // Initialize GPU surface nets for density mesh rendering (always needed)
        gpu_scene.initialize_gpu_surface_nets(device, config.format);

        // Set initial render parameters from editor state
        if let Some(ref surface_nets) = gpu_scene.gpu_surface_nets {
            surface_nets.set_initial_params(queue, editor_state);
        }

        if gpu_scene.fluid_simulator.is_some() {
            gpu_scene.rewire_transferred_fluid_system(device, queue, config.format);
        }

        // Sync lighting and fog from current editor state so reset doesn't revert visuals
        gpu_scene.apply_light_params_from_editor(editor_state);

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

    /// Drain audio events generated by the active scene.
    pub fn drain_audio_events(&mut self) -> Vec<crate::audio::GameAudioEvent> {
        self.active_scene_mut().drain_audio_events()
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
        outline_width: f32,
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
            outline_width,
        );
    }

    /// Match shared depth and intermediate targets to the current presentation size.
    #[cfg(feature = "vr")]
    pub fn ensure_render_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let current = match self.current_mode {
            SimulationMode::Preview => self
                .preview_scene
                .as_ref()
                .map(|s| (s.renderer.width, s.renderer.height)),
            SimulationMode::Gpu => self
                .gpu_scene
                .as_ref()
                .map(|s| (s.renderer.width, s.renderer.height)),
        };
        if current != Some((width, height)) {
            self.active_scene_mut().resize(device, width, height);
        }
    }

    #[cfg(feature = "vr")]
    pub fn render_stereo(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        eyes: Vec<(wgpu::TextureView, crate::rendering::RenderView)>,
        visuals: &[crate::cell::types::CellTypeVisuals],
        world_diameter: f32,
        lod_scale: f32,
        lod_low: f32,
        lod_medium: f32,
        lod_high: f32,
        lod_debug: bool,
        outline_width: f32,
    ) -> bool {
        if eyes.len() != 2 {
            return false;
        }
        let dimensions = (eyes[0].1.width, eyes[0].1.height);
        self.ensure_render_size(device, dimensions.0, dimensions.1);
        // Hi-Z from another eye or the previous desktop view cannot safely occlude
        // this view. Keep per-eye frustum culling, and remove depth-of-field in VR.
        let previous_gpu = self.gpu_scene.as_mut().map(|scene| {
            if let Some(timer) = &mut scene.gpu_timer {
                timer.set_view_count(2);
            }
            let previous = (
                scene.culling_mode(),
                scene.show_dof,
                scene.headless_no_render,
                scene.post_process.as_ref().map(|pp| pp.water_distortion_enabled),
            );
            scene.set_culling_mode(crate::rendering::CullingMode::FrustumOnly);
            scene.show_dof = false;
            if let Some(pp) = &mut scene.post_process {
                pp.water_distortion_enabled = false;
            }
            scene.headless_no_render = false;
            previous
        });
        for (index, (target, view)) in eyes.into_iter().enumerate() {
            let scene = self.active_scene_mut();
            let previous_view = scene.camera_mut().set_render_view(Some(view));
            if index == 0 {
                scene.render(
                    device,
                    queue,
                    &target,
                    Some(visuals),
                    world_diameter,
                    lod_scale,
                    lod_low,
                    lod_medium,
                    lod_high,
                    lod_debug,
                    outline_width,
                );
            } else {
                scene.render_view(
                    device,
                    queue,
                    &target,
                    Some(visuals),
                    world_diameter,
                    lod_scale,
                    lod_low,
                    lod_medium,
                    lod_high,
                    lod_debug,
                    outline_width,
                );
            }
            scene.camera_mut().set_render_view(previous_view);
        }
        if let (Some(scene), Some(previous)) = (&mut self.gpu_scene, previous_gpu) {
            if let Some(timer) = &mut scene.gpu_timer {
                timer.set_view_count(1);
            }
            scene.set_culling_mode(previous.0);
            scene.show_dof = previous.1;
            scene.headless_no_render = previous.2;
            if let (Some(pp), Some(enabled)) = (&mut scene.post_process, previous.3) {
                pp.water_distortion_enabled = enabled;
            }
        }
        true
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
                    gpu_scene.insert_cell_from_genome(
                        device,
                        encoder,
                        queue,
                        world_position,
                        genome,
                        0,
                        0,
                        None,
                    )
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

    /// Mark a cell as being dragged so physics skips it (GPU scene only).
    pub fn set_dragged_cell(&mut self, cell_index: u32) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.set_dragged_cell(cell_index);
            }
        }
    }

    /// Clear the dragged cell so physics resumes for all cells (GPU scene only).
    pub fn clear_dragged_cell(&mut self) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.clear_dragged_cell();
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

    /// Start a spatial query to find the organism under the cursor for camera following (GPU scene only).
    /// Called on double-click when no tool is active.
    pub fn start_organism_follow_query(&mut self, screen_x: f32, screen_y: f32) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.start_organism_follow_query(screen_x, screen_y);
            }
        }
    }

    /// Stop following any organism and return to free camera (GPU scene only).
    pub fn clear_organism_follow(&mut self) {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            if let Some(gpu_scene) = self.gpu_scene.as_mut() {
                gpu_scene.clear_organism_follow();
            }
        }
    }

    /// Returns true if the GPU scene camera is currently locked to an organism.
    pub fn is_following_organism(&self) -> bool {
        if self.current_mode == crate::ui::SimulationMode::Gpu {
            self.gpu_scene
                .as_ref()
                .map(|s| s.is_following_organism())
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Poll the organism follow readback and update the camera center (GPU scene only).
    /// No-op - follow camera is updated inside GpuScene::render() which has device+queue.
    pub fn poll_organism_follow(&mut self, _device: &wgpu::Device, _dt: f32) {
        // Follow camera update happens inside GpuScene::render() via update_follow_camera().
    }

    /// Poll for tool operation results (GPU scene only).
    ///
    /// This method checks for completed spatial query results and updates tool states.
    /// It should be called each frame to process async tool operation completions.
    /// For preview scene, this method does nothing.
    pub fn poll_tool_operation_results(
        &mut self,
        radial_menu: &mut crate::ui::radial_menu::RadialMenuState,
        drag_distance: &mut f32,
        _queue: &wgpu::Queue,
    ) {
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
    pub fn screen_to_world_at_distance(
        &self,
        screen_x: f32,
        screen_y: f32,
        distance: f32,
    ) -> glam::Vec3 {
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
    pub fn update_gizmo_config(
        &mut self,
        editor_state: &crate::ui::panel_context::GenomeEditorState,
    ) {
        // Only preview scene has gizmos
        if let Some(scene) = &mut self.preview_scene {
            scene.update_gizmo_config(editor_state);
        }
    }

    /// Update split ring configuration for all existing scenes.
    pub fn update_split_ring_config(
        &mut self,
        editor_state: &crate::ui::panel_context::GenomeEditorState,
    ) {
        // Only preview scene has split rings
        if let Some(scene) = &mut self.preview_scene {
            scene.update_split_ring_config(editor_state);
        }
    }
}
