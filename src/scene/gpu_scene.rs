//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::genome::Genome;
use crate::rendering::{CellRenderer, CullingMode, GpuAdhesionLineRenderer, HizGenerator, InstanceBuilder, WorldSphereRenderer};
use crate::scene::Scene;
use crate::simulation::{CanonicalState, PhysicsConfig};
use crate::simulation::gpu_physics::{execute_gpu_physics_step, execute_lifecycle_pipeline, CachedBindGroups, GpuPhysicsPipelines, GpuTripleBufferSystem, AdhesionBuffers};
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
    /// GPU-based adhesion line renderer (reads directly from GPU buffers)
    pub adhesion_renderer: GpuAdhesionLineRenderer,
    /// World sphere renderer for boundary visualization
    pub world_sphere_renderer: WorldSphereRenderer,
    /// GPU instance builder with frustum and occlusion culling
    pub instance_builder: InstanceBuilder,
    /// Hi-Z generator for occlusion culling
    pub hiz_generator: HizGenerator,
    /// GPU physics pipelines (6 compute shaders)
    pub gpu_physics_pipelines: GpuPhysicsPipelines,
    /// Triple buffer system for GPU physics
    pub gpu_triple_buffers: GpuTripleBufferSystem,
    /// Adhesion buffer system for GPU adhesion physics
    pub adhesion_buffers: AdhesionBuffers,
    /// Cached bind groups (created once, not per-frame)
    pub cached_bind_groups: CachedBindGroups,
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
    /// Cached parent_make_adhesion flags from genome modes for quick lookup during division
    parent_make_adhesion_flags: Vec<bool>,
    /// Accumulated time for fixed timestep physics
    time_accumulator: f32,
    /// Whether this is the first frame (no Hi-Z data yet)
    first_frame: bool,
    /// Whether GPU readbacks are enabled (cell count, etc.)
    readbacks_enabled: bool,
    /// Time scale multiplier (1.0 = normal, 2.0 = 2x speed)
    pub time_scale: f32,
    /// Whether to show adhesion lines
    pub show_adhesion_lines: bool,
    /// Point cloud rendering mode for maximum performance
    pub point_cloud_mode: bool,
    /// Whether to show the world boundary sphere
    pub show_world_sphere: bool,
    /// DEBUG: Track frames since last insertion for debugging
    debug_frames_since_insertion: u32,
    /// DEBUG: Last known cell count for detecting drops
    debug_last_cell_count: usize,
    /// DEBUG: Frame counter for periodic logging
    debug_frame_counter: u32,
}

impl GpuScene {
    /// Create a new GPU scene with the specified capacity.
    pub fn with_capacity(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        capacity: u32,
    ) -> Self {
        // Use 64x64x64 grid for spatial partitioning
        let canonical_state = CanonicalState::with_grid_density(capacity as usize, 64);
        let config = PhysicsConfig::default();

        let renderer = CellRenderer::new(device, queue, surface_config, capacity as usize);
        
        // Create GPU adhesion line renderer
        // For 200K cells with ~5 adhesions average = 500K max adhesions
        let max_adhesions: u32 = 500_000; // Match MAX_ADHESION_CONNECTIONS
        let adhesion_renderer = GpuAdhesionLineRenderer::new(device, surface_config, max_adhesions);
        
        // Create world sphere renderer for boundary visualization
        // Radius matches physics config sphere_radius
        let mut world_sphere_renderer = WorldSphereRenderer::new(
            device,
            surface_config,
            wgpu::TextureFormat::Depth32Float,
        );
        // Sync world sphere radius with physics boundary
        world_sphere_renderer.set_radius(queue, config.sphere_radius);
        
        // Create instance builder - culling mode will be set per-frame in render()
        let instance_builder = InstanceBuilder::new(device, capacity as usize);
        
        // Create Hi-Z generator for occlusion culling
        let mut hiz_generator = HizGenerator::new(device);
        hiz_generator.resize(device, surface_config.width, surface_config.height);

        // Create GPU physics components
        let gpu_physics_pipelines = GpuPhysicsPipelines::new(device);
        let gpu_triple_buffers = GpuTripleBufferSystem::new(device, capacity);
        
        // Create adhesion buffer system
        let max_modes: u32 = 256; // Maximum modes across all genomes
        let mut adhesion_buffers = AdhesionBuffers::new(device, capacity, max_modes);
        
        // Initialize adhesion buffers with default values
        adhesion_buffers.initialize(queue);
        
        // Create cached bind groups (once, not per-frame!)
        let cached_bind_groups = gpu_physics_pipelines.create_cached_bind_groups(device, &gpu_triple_buffers, &adhesion_buffers);

        Self {
            canonical_state,
            renderer,
            adhesion_renderer,
            world_sphere_renderer,
            instance_builder,
            hiz_generator,
            gpu_physics_pipelines,
            gpu_triple_buffers,
            adhesion_buffers,
            cached_bind_groups,
            config,
            paused: false,
            camera: CameraController::new(),
            current_time: 0.0,
            genomes: Vec::new(),
            parent_make_adhesion_flags: Vec::new(),
            time_accumulator: 0.0,
            first_frame: true,
            readbacks_enabled: true,
            time_scale: 1.0,
            show_adhesion_lines: true,
            point_cloud_mode: false,
            show_world_sphere: true,
            debug_frames_since_insertion: u32::MAX,
            debug_last_cell_count: 0,
            debug_frame_counter: 0,
        }
    }
    
    /// Create a new GPU scene with default capacity (100k).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        Self::with_capacity(device, queue, surface_config, 20_000)
    }
    
    /// Get the current cell capacity.
    pub fn capacity(&self) -> u32 {
        self.gpu_triple_buffers.capacity
    }

    /// Reset the simulation to initial state.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.canonical_state.cell_count = 0;
        self.canonical_state.next_cell_id = 0;
        self.current_time = 0.0;
        self.time_accumulator = 0.0;
        self.paused = false;
        self.first_frame = true;
        // Clear adhesion connections
        self.canonical_state.adhesion_connections.active_count = 0;
        self.canonical_state.adhesion_manager.reset();
        
        // Reset adhesion buffers
        self.adhesion_buffers.reset(queue);
        // Clear all genomes since no cells reference them
        self.genomes.clear();
        self.parent_make_adhesion_flags.clear();
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
        // Reset GPU cell count buffer to 0 immediately (don't wait for sync)
        let cell_counts: [u32; 2] = [0, 0];
        queue.write_buffer(&self.gpu_triple_buffers.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        // CRITICAL: Also reset the cached GPU cell count to avoid stale values
        // causing instant explosion when inserting cells after reset
        self.gpu_triple_buffers.reset_gpu_cell_count();
        
        // Reset deterministic cell addition system
        self.gpu_triple_buffers.reset_slot_allocator();
        
        // Mark GPU buffers as needing sync (will be no-op since cell_count is 0)
        self.gpu_triple_buffers.mark_needs_sync();
    }
    
    /// Remove unused genomes from the end of the genomes list.
    /// Called when cells are removed to free up genome slots for reuse.
    pub fn compact_genomes(&mut self) {
        if self.genomes.is_empty() || self.canonical_state.cell_count == 0 {
            self.genomes.clear();
            self.parent_make_adhesion_flags.clear();
            return;
        }
        
        // Find the highest genome_id still in use
        let max_used_genome_id = self.canonical_state.genome_ids[..self.canonical_state.cell_count]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        
        // Truncate genomes to only keep those still referenced
        if max_used_genome_id + 1 < self.genomes.len() {
            self.genomes.truncate(max_used_genome_id + 1);
            
            // Also truncate parent_make_adhesion_flags to match
            let total_modes: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
            self.parent_make_adhesion_flags.truncate(total_modes);
        }
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
    
    /// Set whether GPU readbacks are enabled (cell count, etc.)
    /// Disabling this can improve performance by avoiding CPU-GPU sync overhead.
    pub fn set_readbacks_enabled(&mut self, enabled: bool) {
        self.readbacks_enabled = enabled;
    }
    
    /// Set point cloud rendering mode for maximum performance.
    /// When enabled, cells are rendered as simple colored circles without lighting.
    pub fn set_point_cloud_mode(&mut self, enabled: bool) {
        self.point_cloud_mode = enabled;
    }
    
    /// Set whether to show the world boundary sphere.
    pub fn set_show_world_sphere(&mut self, enabled: bool) {
        self.show_world_sphere = enabled;
    }
    
    /// Get the world sphere renderer for customization.
    pub fn world_sphere_renderer(&self) -> &WorldSphereRenderer {
        &self.world_sphere_renderer
    }
    
    /// Get mutable access to the world sphere renderer for customization.
    pub fn world_sphere_renderer_mut(&mut self) -> &mut WorldSphereRenderer {
        &mut self.world_sphere_renderer
    }
    
    /// Read culling statistics from GPU (blocking).
    pub fn read_culling_stats(&mut self, device: &wgpu::Device) -> crate::rendering::CullingStats {
        self.instance_builder.read_culling_stats_blocking(device)
    }

    /// Run physics step using GPU compute shaders with zero CPU involvement.
    fn run_physics(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, delta_time: f32) {
        // Process pending cell additions first (deterministic order)
        if self.gpu_triple_buffers.pending_cell_addition_count() > 0 {
            let added_cells = self.gpu_triple_buffers.process_pending_cell_additions(
                queue,
                &mut self.canonical_state,
                &self.genomes,
            );
            
            if !added_cells.is_empty() {
                // Reset debug counter to track what happens after additions
                self.debug_frames_since_insertion = 0;
            }
        }
        
        if self.canonical_state.cell_count == 0 && !self.gpu_triple_buffers.needs_sync {
            return;
        }
        
        self.debug_frame_counter += 1;
        
        // DEBUG: Detect cell count drops (sign of the bug)
        let gpu_count = self.gpu_triple_buffers.gpu_cell_count() as usize;
        if self.debug_last_cell_count > 0 && gpu_count > 0 && gpu_count < self.debug_last_cell_count / 2 {
            // Cell count dropped significantly - potential bug
        }
        if gpu_count > 0 {
            self.debug_last_cell_count = gpu_count;
        }
        
        // DEBUG: Log buffer state on frames 1-3 after an insertion to see what physics does
        if self.debug_frames_since_insertion < 5 {
            self.debug_frames_since_insertion += 1;
        }
        
        // DEBUG: Log buffer state BEFORE sync if we have pending insertions
        let has_pending = self.gpu_triple_buffers.has_pending_insertions();
        if has_pending {
            // Reset frame counter to track what happens after this insertion
            self.debug_frames_since_insertion = 0;
        }
        
        // Sync pending cell insertions (only new cells, not all cells)
        self.gpu_triple_buffers.sync_pending_insertions(queue, &self.canonical_state, &self.genomes);
        
        // DEBUG: Log buffer state AFTER sync (if we had pending insertions)
        if has_pending {
            // Buffer state logged after sync
        }
        
        // Full sync only needed for initial setup (first time cells are added)
        if self.gpu_triple_buffers.needs_sync {
            self.gpu_triple_buffers.sync_from_canonical_state(queue, &self.canonical_state, &self.genomes);
        }
        
        // Clear the masses changed flag before physics
        self.canonical_state.masses_changed = false;
        
        // Execute pure GPU physics pipeline (7 compute shader stages)
        // Cell count is read from GPU buffer by shaders
        // Uses cached bind groups (no per-frame allocation!)
        execute_gpu_physics_step(
            device,
            encoder,
            queue,
            &self.gpu_physics_pipelines,
            &mut self.gpu_triple_buffers,
            &self.cached_bind_groups,
            delta_time,
            self.current_time,
        );
        
        // Execute lifecycle pipeline for cell division (4 compute shader stages)
        // This handles death detection, free slot compaction, and cell division
        // Cell count is updated on GPU by division shader
        execute_lifecycle_pipeline(
            device,
            encoder,
            queue,
            &self.gpu_physics_pipelines,
            &mut self.gpu_triple_buffers,
            &self.cached_bind_groups,
            self.current_time,
        );
        
        // Copy GPU physics output to instance builder's buffers
        // Copy full capacity - the build_instances shader will use cell_count_buffer
        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        let vec4_copy_size = (self.gpu_triple_buffers.capacity as usize * 16) as u64; // Vec4<f32> = 16 bytes
        let u32_copy_size = (self.gpu_triple_buffers.capacity as usize * 4) as u64; // u32 = 4 bytes
        
        // Copy positions (Vec4: x, y, z, mass)
        encoder.copy_buffer_to_buffer(
            &self.gpu_triple_buffers.position_and_mass[output_idx],
            0,
            self.instance_builder.positions_buffer(),
            0,
            vec4_copy_size,
        );
        
        // Copy rotations (Vec4: quaternion)
        encoder.copy_buffer_to_buffer(
            &self.gpu_triple_buffers.rotations[output_idx],
            0,
            self.instance_builder.rotations_buffer(),
            0,
            vec4_copy_size,
        );
        
        // Copy mode indices (u32) - critical for correct cell colors after division
        encoder.copy_buffer_to_buffer(
            &self.gpu_triple_buffers.mode_indices,
            0,
            self.instance_builder.mode_indices_buffer(),
            0,
            u32_copy_size,
        );
        
        // Copy cell IDs (u32) - needed for stable animation offsets
        encoder.copy_buffer_to_buffer(
            &self.gpu_triple_buffers.cell_ids,
            0,
            self.instance_builder.cell_ids_buffer(),
            0,
            u32_copy_size,
        );
        
        // Copy genome IDs (u32) - needed for cell type visuals lookup
        encoder.copy_buffer_to_buffer(
            &self.gpu_triple_buffers.genome_ids,
            0,
            self.instance_builder.genome_ids_buffer(),
            0,
            u32_copy_size,
        );
        
        // Start async read of GPU cell count for performance monitoring (if enabled)
        if self.readbacks_enabled {
            self.gpu_triple_buffers.start_async_cell_count_read(encoder);
        }
        
        // Clear dirty flags since we just copied from GPU
        self.instance_builder.clear_positions_dirty();
        self.instance_builder.clear_rotations_dirty();
        self.instance_builder.clear_mode_indices_dirty();
        self.instance_builder.clear_cell_ids_dirty();
        self.instance_builder.clear_genome_ids_dirty();
    }
    
    /// Get the GPU cell count (updated asynchronously, may be 1-2 frames behind).
    /// This is the actual cell count on the GPU, which may differ from canonical_state
    /// when cells are dividing or dying.
    pub fn gpu_cell_count(&self) -> u32 {
        self.gpu_triple_buffers.gpu_cell_count()
    }
    
    /// Poll for async GPU cell count read completion.
    /// Call this after queue.submit() to process the async read.
    pub fn poll_gpu_cell_count(&self, device: &wgpu::Device) {
        self.gpu_triple_buffers.poll_async_cell_count_read(device);
        self.gpu_triple_buffers.try_read_mapped_cell_count();
    }
    
    /// Find an existing genome by name, or return None.
    pub fn find_genome_id(&self, name: &str) -> Option<usize> {
        self.genomes.iter().position(|g| g.name == name)
    }
    
    /// Check if two genomes are fully equal (all properties, not just visual).
    fn genomes_equal(a: &Genome, b: &Genome) -> bool {
        // Compare all relevant properties, not just visual ones
        if a.modes.len() != b.modes.len() || a.initial_mode != b.initial_mode {
            return false;
        }
        for (ma, mb) in a.modes.iter().zip(b.modes.iter()) {
            // Visual properties
            if (ma.color - mb.color).length() > 0.001 
                || (ma.opacity - mb.opacity).abs() > 0.001
                || (ma.emissive - mb.emissive).abs() > 0.001 {
                return false;
            }
            // Division properties
            if ma.parent_make_adhesion != mb.parent_make_adhesion
                || (ma.split_mass - mb.split_mass).abs() > 0.001
                || (ma.split_interval - mb.split_interval).abs() > 0.001
                || (ma.split_ratio - mb.split_ratio).abs() > 0.001
                || ma.max_splits != mb.max_splits {
                return false;
            }
            // Child properties
            if ma.child_a.mode_number != mb.child_a.mode_number
                || ma.child_b.mode_number != mb.child_b.mode_number
                || ma.child_a.keep_adhesion != mb.child_a.keep_adhesion
                || ma.child_b.keep_adhesion != mb.child_b.keep_adhesion {
                return false;
            }
            // Adhesion settings
            if ma.adhesion_settings != mb.adhesion_settings {
                return false;
            }
        }
        true
    }
    
    /// Add a genome to the scene and return its ID.
    /// If the last genome is fully identical, reuses it.
    /// Otherwise creates a new genome entry to preserve existing cells' settings.
    pub fn add_genome(&mut self, genome: Genome) -> usize {
        // Check if the last genome is fully identical - if so, reuse it
        if let Some(last) = self.genomes.last() {
            if Self::genomes_equal(last, &genome) {
                return self.genomes.len() - 1;
            }
        }
        
        let id = self.genomes.len();
        self.genomes.push(genome);
        id
    }
    
    /// Sync adhesion settings from genomes to GPU
    /// Call this after adding genomes to ensure settings are uploaded to GPU
    pub fn sync_adhesion_settings(&mut self, queue: &wgpu::Queue) {
        self.adhesion_buffers.sync_adhesion_settings(queue, &self.genomes);
        
        // Sync parent_make_adhesion_flags to triple buffer system
        self.gpu_triple_buffers.sync_parent_make_adhesion_flags(queue, &self.genomes);
        
        // Sync child keep adhesion flags for zone-based inheritance
        self.gpu_triple_buffers.sync_child_keep_adhesion_flags(queue, &self.genomes);
        
        // Sync mode properties (nutrient_gain_rate, max_cell_size, etc.) for division
        self.gpu_triple_buffers.sync_mode_properties(queue, &self.genomes);
        
        // Cache parent_make_adhesion flags for quick lookup during division
        // The flags are stored sequentially: genome0_mode0, genome0_mode1, ..., genome1_mode0, genome1_mode1, ...
        // This matches the adhesion settings buffer layout and the global mode index calculation
        self.parent_make_adhesion_flags.clear();
        for genome in &self.genomes {
            for mode in &genome.modes {
                self.parent_make_adhesion_flags.push(mode.parent_make_adhesion);
            }
        }
    }
    
    /// Get parent_make_adhesion flag for a specific mode index
    /// Returns false if mode index is out of bounds
    pub fn get_parent_make_adhesion_flag(&self, mode_index: usize) -> bool {
        self.parent_make_adhesion_flags.get(mode_index).copied().unwrap_or(false)
    }
    
    /// Convert local mode index to global mode index for adhesion settings lookup
    /// Returns the global mode index that can be used to access adhesion settings
    pub fn get_global_mode_index(&self, genome_id: usize, local_mode_index: usize) -> usize {
        let mut global_index = 0;
        for (i, genome) in self.genomes.iter().enumerate() {
            if i == genome_id {
                return global_index + local_mode_index;
            }
            global_index += genome.modes.len();
        }
        global_index + local_mode_index // Fallback if genome_id is out of bounds
    }
    
    /// Insert a cell at the given world position using genome settings.
    /// Adds the genome to the scene if not already present (does not overwrite existing genomes).
    /// Returns the index of the inserted cell, or None if at capacity.
    pub fn insert_cell_from_genome(
        &mut self,
        world_position: glam::Vec3,
        genome: &Genome,
    ) -> Option<usize> {
        // CRITICAL: Sync canonical_state.cell_count with GPU cell count before inserting.
        // The GPU may have more cells due to division that the CPU doesn't know about.
        // Without this, we would overwrite existing GPU cells.
        let gpu_count = self.gpu_triple_buffers.gpu_cell_count() as usize;
        if gpu_count > self.canonical_state.cell_count {
            self.canonical_state.cell_count = gpu_count;
        }
        
        // Find or add the genome
        let genome_id = self.add_genome(genome.clone());
        
        let mode_idx = genome.initial_mode.max(0) as usize;
        let mode = &genome.modes[mode_idx];
        
        // Calculate initial radius from mass (mass = 4/3 * pi * r^3 for unit density)
        let initial_mass = 1.0_f32;
        let radius = (initial_mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        // For immediate insertion (UI tools), use the old synchronous method
        let result = self.canonical_state.add_cell(
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
            mode.membrane_stiffness,             // stiffness (use mode-specific membrane stiffness)
        );
        
        // Queue the new cell for GPU sync (only syncs this cell, not all cells)
        if let Some(cell_idx) = result {
            self.gpu_triple_buffers.queue_cell_insertion(cell_idx);
        }
        
        result
    }
    
    /// Queue a cell for deterministic addition (processed during next physics step)
    /// This is used for programmatic cell addition that needs deterministic ordering.
    pub fn queue_cell_from_genome(
        &mut self,
        world_position: glam::Vec3,
        genome: &Genome,
    ) {
        // Find or add the genome
        let genome_id = self.add_genome(genome.clone());
        
        let mode_idx = genome.initial_mode.max(0) as usize;
        
        // Calculate initial radius from mass (mass = 4/3 * pi * r^3 for unit density)
        let initial_mass = 1.0_f32;
        
        // Use deterministic cell addition system for GPU scene
        let request = GpuTripleBufferSystem::create_cell_addition_request(
            world_position,
            glam::Vec3::ZERO,                    // velocity
            initial_mass,                        // mass
            genome.initial_orientation,          // rotation
            genome_id,                           // genome_id
            mode_idx,                            // mode_index (local to this genome)
            self.current_time,                   // birth_time
            &self.genomes,                       // genomes for parameter lookup
        );
        
        // Queue for deterministic processing
        self.gpu_triple_buffers.queue_cell_addition(request);
    }
    
    /// Update a cell's position and sync to GPU buffers.
    /// Used by the drag tool to move cells.
    pub fn set_cell_position(&mut self, queue: &wgpu::Queue, cell_idx: usize, position: glam::Vec3) {
        if cell_idx >= self.canonical_state.cell_count {
            return;
        }
        
        // Update canonical state
        self.canonical_state.positions[cell_idx] = position;
        self.canonical_state.prev_positions[cell_idx] = position;
        self.canonical_state.velocities[cell_idx] = glam::Vec3::ZERO;
        
        // Sync to GPU immediately
        let velocity = glam::Vec3::ZERO;
        let mass = self.canonical_state.masses[cell_idx];
        let rotation = self.canonical_state.rotations[cell_idx];
        self.gpu_triple_buffers.sync_single_cell(queue, cell_idx, position, velocity, mass, rotation);
    }
    
    /// Update a cell's mass and sync to GPU buffers.
    /// Used by the boost tool to increase cell mass.
    pub fn set_cell_mass(&mut self, queue: &wgpu::Queue, cell_idx: usize, mass: f32) {
        if cell_idx >= self.canonical_state.cell_count {
            return;
        }
        
        // Update canonical state
        self.canonical_state.masses[cell_idx] = mass;
        
        // Update radius based on new mass
        let radius = (mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        self.canonical_state.radii[cell_idx] = radius;
        
        // Sync to GPU immediately
        let position = self.canonical_state.positions[cell_idx];
        let velocity = self.canonical_state.velocities[cell_idx];
        let rotation = self.canonical_state.rotations[cell_idx];
        self.gpu_triple_buffers.sync_single_cell(queue, cell_idx, position, velocity, mass, rotation);
        
        // Mark radii dirty so instance builder updates
        self.instance_builder.mark_radii_dirty();
    }
    
    /// Find the cell closest to a world position within a given radius.
    /// Returns the cell index if found.
    pub fn find_cell_at_position(&self, world_pos: glam::Vec3, max_distance: f32) -> Option<usize> {
        let mut closest_idx = None;
        let mut closest_dist_sq = max_distance * max_distance;
        
        for i in 0..self.canonical_state.cell_count {
            let cell_pos = self.canonical_state.positions[i];
            let cell_radius = self.canonical_state.radii[i];
            let dist_sq = (cell_pos - world_pos).length_squared();
            
            // Check if click is within cell radius or within max_distance
            let effective_dist_sq = (dist_sq.sqrt() - cell_radius).max(0.0).powi(2);
            
            if effective_dist_sq < closest_dist_sq {
                closest_dist_sq = effective_dist_sq;
                closest_idx = Some(i);
            }
        }
        
        closest_idx
    }
    
    /// Remove a cell at the given index.
    /// Uses swap-remove for O(1) removal - the last cell is moved to fill the gap.
    /// Also removes all adhesion connections involving this cell.
    pub fn remove_cell(&mut self, cell_idx: usize) -> bool {
        if cell_idx >= self.canonical_state.cell_count {
            return false;
        }
        
        let state = &mut self.canonical_state;
        
        // Remove all adhesion connections for this cell
        state.adhesion_manager.remove_all_connections_for_cell(
            &mut state.adhesion_connections,
            cell_idx,
        );
        
        let last_idx = state.cell_count - 1;
        
        if cell_idx != last_idx {
            // Swap with last cell - copy all properties
            state.cell_ids[cell_idx] = state.cell_ids[last_idx];
            state.positions[cell_idx] = state.positions[last_idx];
            state.prev_positions[cell_idx] = state.prev_positions[last_idx];
            state.velocities[cell_idx] = state.velocities[last_idx];
            state.masses[cell_idx] = state.masses[last_idx];
            state.radii[cell_idx] = state.radii[last_idx];
            state.genome_ids[cell_idx] = state.genome_ids[last_idx];
            state.mode_indices[cell_idx] = state.mode_indices[last_idx];
            state.rotations[cell_idx] = state.rotations[last_idx];
            state.genome_orientations[cell_idx] = state.genome_orientations[last_idx];
            state.angular_velocities[cell_idx] = state.angular_velocities[last_idx];
            state.forces[cell_idx] = state.forces[last_idx];
            state.torques[cell_idx] = state.torques[last_idx];
            state.accelerations[cell_idx] = state.accelerations[last_idx];
            state.prev_accelerations[cell_idx] = state.prev_accelerations[last_idx];
            state.stiffnesses[cell_idx] = state.stiffnesses[last_idx];
            state.birth_times[cell_idx] = state.birth_times[last_idx];
            state.split_intervals[cell_idx] = state.split_intervals[last_idx];
            state.split_masses[cell_idx] = state.split_masses[last_idx];
            state.split_counts[cell_idx] = state.split_counts[last_idx];
            
            // Update adhesion connections that referenced the moved cell
            state.adhesion_manager.update_cell_index_after_swap(
                &mut state.adhesion_connections,
                last_idx,
                cell_idx,
            );
        }
        
        state.cell_count -= 1;
        
        // Compact genomes to free unused slots
        self.compact_genomes();
        
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
        
        // Mark GPU buffers as needing sync
        self.gpu_triple_buffers.mark_needs_sync();
        
        true
    }
    
    /// Cast a ray from screen coordinates and find the closest cell that intersects.
    /// Returns (cell_index, hit_distance) if a cell is hit.
    pub fn raycast_cell(&self, screen_x: f32, screen_y: f32) -> Option<(usize, f32)> {
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
        let ray_origin = self.camera.position();
        let ray_dir = self.camera.rotation * ray_view;
        
        // Find closest cell that intersects the ray
        let mut closest_hit: Option<(usize, f32)> = None;
        
        for i in 0..self.canonical_state.cell_count {
            let cell_pos = self.canonical_state.positions[i];
            let cell_radius = self.canonical_state.radii[i];
            
            // Ray-sphere intersection
            if let Some(t) = ray_sphere_intersect(ray_origin, ray_dir, cell_pos, cell_radius) {
                if t > 0.0 {
                    match closest_hit {
                        None => closest_hit = Some((i, t)),
                        Some((_, closest_t)) if t < closest_t => closest_hit = Some((i, t)),
                        _ => {}
                    }
                }
            }
        }
        
        closest_hit
    }
    
    /// Get the world position on a ray at a given distance from camera.
    pub fn screen_to_world_at_distance(&self, screen_x: f32, screen_y: f32, distance: f32) -> glam::Vec3 {
        let width = self.renderer.width as f32;
        let height = self.renderer.height as f32;
        
        let ndc_x = (2.0 * screen_x / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / height);
        
        let aspect = width / height;
        let fov = 45.0_f32.to_radians();
        let tan_half_fov = (fov / 2.0).tan();
        
        let ray_view = glam::Vec3::new(
            ndc_x * aspect * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        ).normalize();
        
        let ray_world = self.camera.rotation * ray_view;
        self.camera.position() + ray_world * distance
    }
    
    /// Convert screen coordinates to world position at a fixed distance from camera.
    /// Uses a constant distance so cells are always inserted at the same depth.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> glam::Vec3 {
        const INSERT_DISTANCE: f32 = 25.0; // Fixed distance for cell insertion
        self.screen_to_world_at_distance(screen_x, screen_y, INSERT_DISTANCE)
    }
}

/// Ray-sphere intersection test.
/// Returns the distance along the ray to the closest intersection point, or None if no hit.
fn ray_sphere_intersect(ray_origin: glam::Vec3, ray_dir: glam::Vec3, sphere_center: glam::Vec3, sphere_radius: f32) -> Option<f32> {
    let oc = ray_origin - sphere_center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        return None;
    }
    
    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);
    
    // Return the closest positive intersection
    if t1 > 0.0 {
        Some(t1)
    } else if t2 > 0.0 {
        Some(t2)
    } else {
        None
    }
}


impl Scene for GpuScene {
    fn update(&mut self, dt: f32) {
        if self.paused || self.canonical_state.cell_count == 0 {
            return;
        }

        // Fixed timestep accumulator pattern
        // Physics steps will be executed in render() based on accumulated time
        // Apply time_scale to make simulation run faster/slower
        self.time_accumulator += dt * self.time_scale;
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>,
    ) {
        // Sync adhesion settings to GPU when genomes are added or modified
        self.sync_adhesion_settings(queue);
        
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
            self.instance_builder.set_hiz_texture(device, hiz_view, self.hiz_generator.mip_count(), &self.gpu_triple_buffers.cell_count_buffer);
        }
        // Don't override culling mode here - it's set by app.rs based on UI settings

        // Create single command encoder for all GPU work to avoid multiple queue.submit() calls
        // (each submit is a sync point that kills performance)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Scene Encoder"),
        });

        // Execute GPU physics pipeline if not paused and has cells
        // Use fixed timestep accumulator for consistent physics behavior
        if !self.paused && self.canonical_state.cell_count > 0 {
            let fixed_dt = self.config.fixed_timestep;
            // Allow more steps when time_scale > 1 (fast forward)
            let max_steps = (4.0 * self.time_scale).ceil() as i32;
            let mut steps = 0;
            
            while self.time_accumulator >= fixed_dt && steps < max_steps {
                self.run_physics(device, &mut encoder, queue, fixed_dt);
                self.current_time += fixed_dt;
                self.time_accumulator -= fixed_dt;
                steps += 1;
            }
            
            // If we hit max steps, discard remaining accumulated time
            if steps >= max_steps {
                self.time_accumulator = 0.0;
            }
        }

        // Build instances with GPU culling (compute pass)
        // Calculate total mode count across all genomes
        // Use capacity for dispatch - shader reads actual cell_count from GPU buffer
        let total_mode_count: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        self.instance_builder.build_instances_with_encoder(
            device,
            &mut encoder,
            queue,
            self.gpu_triple_buffers.capacity as usize,
            total_mode_count,
            cell_type_visuals.map(|v| v.len()).unwrap_or(1),
            view_proj,
            self.camera.position(),
            self.renderer.width,
            self.renderer.height,
            &self.gpu_triple_buffers.cell_count_buffer,
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
            self.point_cloud_mode,
        );

        // Render world boundary sphere if enabled
        if self.show_world_sphere {
            self.world_sphere_renderer.render(
                &mut encoder,
                queue,
                view,
                &self.renderer.depth_view,
                self.camera.position(),
                self.camera.rotation,
            );
        }

        // Render adhesion lines if enabled
        if self.show_adhesion_lines {
            // Create bind group with current output buffer (physics results)
            let output_idx = self.gpu_triple_buffers.output_buffer_index();
            let adhesion_data_bind_group = self.adhesion_renderer.create_data_bind_group(
                device,
                &self.gpu_triple_buffers.position_and_mass[output_idx],
                &self.adhesion_buffers.adhesion_connections,
                &self.adhesion_buffers.adhesion_counts,
                &self.gpu_triple_buffers.cell_count_buffer,
            );
            
            let mut adhesion_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU Adhesion Lines Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve cell rendering
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Use existing depth for occlusion
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.adhesion_renderer.render_in_pass(
                &mut adhesion_pass,
                queue,
                &adhesion_data_bind_group,
                self.camera.position(),
                self.camera.rotation,
            );
        }

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
        
        // DEBUG: Read back adhesion data only when cell count increases (division happened)
        let current_cell_count = self.gpu_triple_buffers.gpu_cell_count();
        if current_cell_count > self.debug_last_cell_count as u32 && self.debug_last_cell_count > 0 {
            println!("[DIVISION] Cell count increased: {} -> {}", self.debug_last_cell_count, current_cell_count);
            self.adhesion_buffers.debug_sync_readback_adhesion_counts(device, queue);
            self.adhesion_buffers.debug_sync_readback_connections(device, queue, 10);
        }
        self.debug_last_cell_count = current_cell_count as usize;
        
        // Poll for async GPU cell count read completion (if enabled)
        if self.readbacks_enabled {
            self.poll_gpu_cell_count(device);
        }

        // Mark that we now have Hi-Z data for next frame
        self.first_frame = false;
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.adhesion_renderer.resize(width, height);
        self.world_sphere_renderer.resize(width, height);
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
