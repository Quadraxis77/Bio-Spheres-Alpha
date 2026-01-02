//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::genome::Genome;
use crate::rendering::{CellRenderer, CullingMode, HizGenerator, InstanceBuilder};
use crate::scene::Scene;
use crate::simulation::{CanonicalState, PhysicsConfig};
use crate::simulation::cpu_physics;
use crate::ui::camera::CameraController;
use glam::Mat4;

/// GPU simulation scene for large-scale simulations.
///
/// Uses compute shaders for physics simulation, allowing for
/// much larger cell counts than the CPU preview mode.
pub struct GpuScene {
    /// Canonical state (used for initial setup and readback if needed)
    pub canonical_state: CanonicalState,
    /// Cell renderer for visualization
    pub renderer: CellRenderer,
    /// GPU instance builder with frustum and occlusion culling
    pub instance_builder: InstanceBuilder,
    /// Hi-Z generator for occlusion culling
    pub hiz_generator: HizGenerator,
    /// Physics configuration
    pub config: PhysicsConfig,
    /// Whether simulation is paused
    pub paused: bool,
    /// Camera controller
    pub camera: CameraController,
    /// Current simulation time
    pub current_time: f32,
    /// Genomes for cell behavior (growth, division) - supports multiple genomes
    pub genomes: Vec<Genome>,
    /// Accumulated time for fixed timestep physics
    time_accumulator: f32,
    /// Whether this is the first frame (no Hi-Z data yet)
    first_frame: bool,
}

impl GpuScene {
    /// Create a new GPU scene.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let capacity = 10_000; // 10k cell cap for GPU scene
        // Use 64x64x64 grid for spatial partitioning
        let canonical_state = CanonicalState::with_grid_density(capacity, 64);
        let config = PhysicsConfig::default();

        let renderer = CellRenderer::new(device, queue, surface_config, capacity);
        
        // Create instance builder - culling mode will be set per-frame in render()
        let instance_builder = InstanceBuilder::new(device, capacity);
        
        // Create Hi-Z generator for occlusion culling
        let mut hiz_generator = HizGenerator::new(device);
        hiz_generator.resize(device, surface_config.width, surface_config.height);

        Self {
            canonical_state,
            renderer,
            instance_builder,
            hiz_generator,
            config,
            paused: false,
            camera: CameraController::new(),
            current_time: 0.0,
            genomes: Vec::new(),
            time_accumulator: 0.0,
            first_frame: true,
        }
    }

    /// Reset the simulation to initial state.
    pub fn reset(&mut self) {
        self.canonical_state.cell_count = 0;
        self.canonical_state.next_cell_id = 0;
        self.current_time = 0.0;
        self.time_accumulator = 0.0;
        self.paused = false;
        self.first_frame = true;
        // Clear adhesion connections
        self.canonical_state.adhesion_connections.active_count = 0;
        self.canonical_state.adhesion_manager.reset();
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
    }
    
    /// Set the culling mode for the instance builder.
    pub fn set_culling_mode(&mut self, mode: CullingMode) {
        self.instance_builder.set_culling_mode(mode);
    }
    
    /// Get the current culling mode.
    pub fn culling_mode(&self) -> CullingMode {
        self.instance_builder.culling_mode()
    }
    
    /// Get culling statistics from the last frame.
    pub fn culling_stats(&self) -> crate::rendering::CullingStats {
        self.instance_builder.culling_stats()
    }
    
    /// Set the occlusion bias for culling.
    /// Negative values = more aggressive culling (cull more cells).
    /// Positive values = more conservative culling (cull fewer cells).
    pub fn set_occlusion_bias(&mut self, bias: f32) {
        self.instance_builder.set_occlusion_bias(bias);
    }
    
    /// Get the current occlusion bias.
    pub fn occlusion_bias(&self) -> f32 {
        self.instance_builder.occlusion_bias()
    }
    
    /// Set the mip level override for occlusion culling.
    pub fn set_occlusion_mip_override(&mut self, mip: i32) {
        self.instance_builder.set_occlusion_mip_override(mip);
    }
    
    /// Set the minimum screen-space size for occlusion culling.
    pub fn set_occlusion_min_screen_size(&mut self, size: f32) {
        self.instance_builder.set_min_screen_size(size);
    }
    
    /// Set the minimum distance for occlusion culling.
    pub fn set_occlusion_min_distance(&mut self, distance: f32) {
        self.instance_builder.set_min_distance(distance);
    }
    
    /// Read culling statistics from GPU (blocking).
    pub fn read_culling_stats(&mut self, device: &wgpu::Device) -> crate::rendering::CullingStats {
        self.instance_builder.read_culling_stats_blocking(device)
    }

    /// Run physics step using CPU physics with genome-based features.
    fn run_physics(&mut self) {
        if self.canonical_state.cell_count == 0 {
            return;
        }
        
        // Use CPU physics with all genomes for division and adhesion
        let _division_events = cpu_physics::physics_step_with_genomes(
            &mut self.canonical_state,
            &self.genomes,
            &self.config,
            self.current_time,
        );
    }
    
    /// Find an existing genome by name, or return None.
    pub fn find_genome_id(&self, name: &str) -> Option<usize> {
        self.genomes.iter().position(|g| g.name == name)
    }
    
    /// Check if two genomes have the same visual properties (modes).
    fn genomes_visually_equal(a: &Genome, b: &Genome) -> bool {
        if a.modes.len() != b.modes.len() {
            return false;
        }
        for (ma, mb) in a.modes.iter().zip(b.modes.iter()) {
            if (ma.color - mb.color).length() > 0.001 
                || (ma.opacity - mb.opacity).abs() > 0.001
                || (ma.emissive - mb.emissive).abs() > 0.001 {
                return false;
            }
        }
        true
    }
    
    /// Add a genome to the scene and return its ID.
    /// If the last genome is visually identical, reuses it.
    /// Otherwise creates a new genome entry to preserve existing cells' visuals.
    pub fn add_genome(&mut self, genome: Genome) -> usize {
        // Check if the last genome is visually identical - if so, reuse it
        if let Some(last) = self.genomes.last() {
            if Self::genomes_visually_equal(last, &genome) {
                return self.genomes.len() - 1;
            }
        }
        
        let id = self.genomes.len();
        self.genomes.push(genome);
        id
    }
    
    /// Insert a cell at the given world position using genome settings.
    /// Adds the genome to the scene if not already present (does not overwrite existing genomes).
    /// Returns the index of the inserted cell, or None if at capacity.
    pub fn insert_cell_from_genome(
        &mut self,
        world_position: glam::Vec3,
        genome: &Genome,
    ) -> Option<usize> {
        // Find or add the genome
        let genome_id = self.add_genome(genome.clone());
        
        let mode_idx = genome.initial_mode.max(0) as usize;
        let mode = &genome.modes[mode_idx];
        
        // Calculate initial radius from mass (mass = 4/3 * pi * r^3 for unit density)
        let initial_mass = 1.0_f32;
        let radius = (initial_mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        self.canonical_state.add_cell(
            world_position,
            glam::Vec3::ZERO,                    // velocity
            genome.initial_orientation,          // rotation
            genome.initial_orientation,          // genome_orientation
            glam::Vec3::ZERO,                    // angular_velocity
            initial_mass,                        // mass
            radius,                              // radius
            genome_id,                           // genome_id
            mode_idx,                            // mode_index (local to this genome)
            self.current_time,                   // birth_time
            mode.split_interval,                 // split_interval
            mode.split_mass,                     // split_mass
            500.0,                               // stiffness (match preview scene)
        )
    }
    
    /// Convert screen coordinates to world position on a plane at the camera's focal point.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> glam::Vec3 {
        let width = self.renderer.width as f32;
        let height = self.renderer.height as f32;
        
        // Normalized device coordinates (-1 to 1)
        let ndc_x = (2.0 * screen_x / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / height); // Flip Y
        
        // Camera matrices
        let aspect = width / height;
        let fov = 45.0_f32.to_radians();
        
        // Calculate ray direction in view space
        let tan_half_fov = (fov / 2.0).tan();
        let ray_view = glam::Vec3::new(
            ndc_x * aspect * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        ).normalize();
        
        // Transform ray to world space
        let ray_world = self.camera.rotation * ray_view;
        
        // Place cell at a fixed distance from camera (use camera distance or default)
        let distance = self.camera.distance.max(10.0);
        
        self.camera.position() + ray_world * distance
    }
}


impl Scene for GpuScene {
    fn update(&mut self, dt: f32) {
        if self.paused || self.canonical_state.cell_count == 0 {
            return;
        }

        // Fixed timestep accumulator pattern
        self.time_accumulator += dt;
        let fixed_dt = self.config.fixed_timestep;
        
        // Run physics steps to catch up (max 4 steps per frame to avoid spiral of death)
        let max_steps = 4;
        let mut steps = 0;
        
        while self.time_accumulator >= fixed_dt && steps < max_steps {
            self.run_physics();
            self.current_time += fixed_dt;
            self.time_accumulator -= fixed_dt;
            steps += 1;
        }
        
        // If we hit max steps, discard remaining accumulated time to prevent buildup
        if steps >= max_steps {
            self.time_accumulator = 0.0;
        }
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>,
    ) {
        // Calculate view-projection matrix for culling
        let view_matrix = Mat4::look_at_rh(
            self.camera.position(),
            self.camera.position() + self.camera.rotation * glam::Vec3::NEG_Z,
            self.camera.rotation * glam::Vec3::Y,
        );
        let aspect = self.renderer.width as f32 / self.renderer.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        // Update instance builder with simulation state FIRST
        // (this may resize buffers and invalidate bind group)
        self.instance_builder.update_from_state(
            device,
            queue,
            &self.canonical_state,
            &self.genomes,
            cell_type_visuals,
        );

        // Set up Hi-Z texture for occlusion culling AFTER update_from_state
        // (so the bind group is created with the correct Hi-Z texture)
        // On first frame, disable culling since we don't have Hi-Z data yet
        if self.first_frame {
            self.instance_builder.set_culling_mode(CullingMode::Disabled);
        } else if let Some(hiz_view) = self.hiz_generator.hiz_view() {
            // Pass Hi-Z texture to instance builder for occlusion culling
            // Note: culling mode is set by app.rs based on UI settings, we just provide the texture
            self.instance_builder.set_hiz_texture(device, hiz_view, self.hiz_generator.mip_count());
        }
        // Don't override culling mode here - it's set by app.rs based on UI settings

        // Create single command encoder for all GPU work to avoid multiple queue.submit() calls
        // (each submit is a sync point that kills performance)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Scene Encoder"),
        });

        // Build instances with GPU culling (compute pass)
        // Calculate total mode count across all genomes
        let total_mode_count: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        self.instance_builder.build_instances_with_encoder(
            &mut encoder,
            queue,
            self.canonical_state.cell_count,
            total_mode_count,
            cell_type_visuals.map(|v| v.len()).unwrap_or(1),
            view_proj,
            self.camera.position(),
            self.renderer.width,
            self.renderer.height,
        );

        // Clear pass
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU Scene Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // Render cells using GPU-culled instance buffer
        let visible_count = self.instance_builder.visible_count();
        self.renderer.render_with_encoder(
            &mut encoder,
            queue,
            view,
            &self.instance_builder,
            visible_count,
            self.camera.position(),
            self.camera.rotation,
            self.current_time,
        );

        // Generate Hi-Z from depth buffer for next frame's occlusion culling
        // Skip if no cells (nothing to cull) or culling is disabled
        if self.canonical_state.cell_count > 0 && self.instance_builder.culling_mode() != CullingMode::Disabled {
            self.hiz_generator.generate(
                device,
                queue,
                &mut encoder,
                &self.renderer.depth_view,
            );
        }

        // Single submit for all GPU work
        queue.submit(std::iter::once(encoder.finish()));

        // Mark that we now have Hi-Z data for next frame
        self.first_frame = false;
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.hiz_generator.resize(device, width, height);
        self.instance_builder.reset_hiz(); // Reset Hi-Z config so bind group is recreated with new texture
        self.first_frame = true; // Need to regenerate Hi-Z
    }

    fn camera(&self) -> &CameraController {
        &self.camera
    }

    fn camera_mut(&mut self) -> &mut CameraController {
        &mut self.camera
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    fn current_time(&self) -> f32 {
        self.current_time
    }

    fn cell_count(&self) -> usize {
        self.canonical_state.cell_count
    }
}
