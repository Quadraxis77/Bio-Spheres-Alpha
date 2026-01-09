//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::genome::Genome;
use crate::rendering::{CellRenderer, CullingMode, GpuAdhesionLineRenderer, HizGenerator, InstanceBuilder, WorldSphereRenderer};
use crate::scene::Scene;
use crate::simulation::{PhysicsConfig};
use crate::simulation::gpu_physics::{execute_gpu_physics_step, execute_lifecycle_pipeline, CachedBindGroups, GpuPhysicsPipelines, GpuTripleBufferSystem, AdhesionBuffers, GpuCellInspector, GpuCellInsertion, GpuToolOperations, AsyncReadbackManager};
use crate::ui::camera::CameraController;
use glam::Mat4;

/// Pending GPU query operation to be executed during render phase
#[derive(Debug, Clone)]
pub enum PendingGpuQuery {
    /// Spatial query for inspect tool
    InspectQuery,
    /// Spatial query for drag tool
    DragQuery,
    /// Spatial query for remove tool
    RemoveQuery,
    /// Spatial query for boost tool
    BoostQuery,
    /// Position update for drag tool
    PositionUpdate { cell_index: u32 },
}

/// Type of tool query being performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolQueryType {
    Inspect,
    Drag,
    Remove,
    Boost,
}

/// GPU simulation scene for large-scale simulations.
///
/// Uses compute shaders for physics simulation, allowing for
/// much larger cell counts than the CPU preview mode.
pub struct GpuScene {
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
    /// GPU cell inspector for real-time cell data extraction
    pub cell_inspector: Option<GpuCellInspector>,
    /// GPU cell insertion system for direct GPU cell creation
    pub cell_insertion: Option<GpuCellInsertion>,
    /// GPU tool operations system for spatial queries and position updates
    pub tool_operations: Option<GpuToolOperations>,
    /// Async readback manager for coordinating all GPU-to-CPU transfers
    pub readback_manager: Option<AsyncReadbackManager>,
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
    /// Pending inspect tool result (temporary until GPU spatial queries are integrated)
    pending_inspect_result: Option<usize>,
    /// Pending drag tool result (temporary until GPU spatial queries are integrated)
    pending_drag_result: Option<(usize, f32)>,
    /// Pending remove tool result (temporary until GPU spatial queries are integrated)
    pending_remove_result: Option<usize>,
    /// Pending boost tool result (temporary until GPU spatial queries are integrated)
    pending_boost_result: Option<usize>,
    /// Pending cell insertion (genome, world position)
    pending_cell_insertion: Option<(glam::Vec3, Genome)>,
    /// Pending tool query position and type for GPU spatial query
    pending_query_position: Option<(f32, f32, ToolQueryType)>,  // (screen_x, screen_y, query_type)
    /// Active query type waiting for GPU result
    active_query_type: Option<ToolQueryType>,
    /// Pending position update for drag tool (cell_index, new_position)
    pending_position_update: Option<(u32, glam::Vec3)>,
    /// Pending cell removal for remove tool (cell_index)
    pending_cell_removal: Option<u32>,
    /// Pending cell boost for boost tool (cell_index)
    pending_cell_boost: Option<u32>,
    /// Pending cell extraction for inspect tool (cell_index)
    pending_cell_extraction: Option<u32>,
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
    /// Current cell count (tracked on GPU, no CPU canonical state)
    pub current_cell_count: u32,
    /// Next cell ID for deterministic cell creation
    next_cell_id: u32,
}

impl GpuScene {
    /// Create a new GPU scene with the specified capacity.
    pub fn with_capacity(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        capacity: u32,
    ) -> Self {
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
        
        // Initialize cell_count_buffer to [0, 0] - CRITICAL: buffer is uninitialized after creation!
        // Without this, the cell insertion shader may read garbage and fail capacity checks.
        let cell_counts: [u32; 2] = [0, 0];
        queue.write_buffer(&gpu_triple_buffers.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        
        // Initialize next_cell_id to 0 - CRITICAL: buffer is uninitialized after creation!
        // Without this, cell IDs may start from garbage values.
        let next_id: [u32; 1] = [0];
        queue.write_buffer(&gpu_triple_buffers.next_cell_id, 0, bytemuck::cast_slice(&next_id));
        
        // Create adhesion buffer system
        let max_modes: u32 = 256; // Maximum modes across all genomes
        let mut adhesion_buffers = AdhesionBuffers::new(device, capacity, max_modes);
        
        // Initialize adhesion buffers with default values
        adhesion_buffers.initialize(queue);
        
        // Create cached bind groups (once, not per-frame!)
        let cached_bind_groups = gpu_physics_pipelines.create_cached_bind_groups(device, &gpu_triple_buffers, &adhesion_buffers);

        // Create GPU cell inspector system (will be initialized later with device)
        let cell_inspector = None; // Will be initialized when needed
        
        // Create GPU cell insertion system (will be initialized later)
        let cell_insertion = None; // Will be initialized when needed
        
        // Create GPU tool operations system (will be initialized later)
        let tool_operations = None; // Will be initialized when needed
        
        // Create async readback manager for coordinating all readback operations
        let readback_manager = None; // Will be initialized when needed

        Self {
            renderer,
            adhesion_renderer,
            world_sphere_renderer,
            instance_builder,
            hiz_generator,
            gpu_physics_pipelines,
            gpu_triple_buffers,
            adhesion_buffers,
            cached_bind_groups,
            cell_inspector,
            cell_insertion,
            tool_operations,
            readback_manager,
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
            pending_inspect_result: None,
            pending_drag_result: None,
            pending_remove_result: None,
            pending_boost_result: None,
            pending_cell_insertion: None,
            pending_query_position: None,
            active_query_type: None,
            pending_position_update: None,
            pending_cell_removal: None,
            pending_cell_boost: None,
            pending_cell_extraction: None,
            show_adhesion_lines: true,
            point_cloud_mode: false,
            show_world_sphere: true,
            debug_frames_since_insertion: u32::MAX,
            debug_last_cell_count: 0,
            debug_frame_counter: 0,
            current_cell_count: 0,
            next_cell_id: 0,
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
        self.current_cell_count = 0;
        self.next_cell_id = 0;
        self.current_time = 0.0;
        self.time_accumulator = 0.0;
        self.paused = false;
        self.first_frame = true;
        
        // Reset adhesion buffers
        self.adhesion_buffers.reset(queue);
        // Clear all genomes since no cells reference them
        self.genomes.clear();
        self.parent_make_adhesion_flags.clear();
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
        // Reset GPU cell count buffer to 0 immediately
        let cell_counts: [u32; 2] = [0, 0];
        queue.write_buffer(&self.gpu_triple_buffers.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        // Reset GPU next_cell_id buffer to 0
        let next_id: [u32; 1] = [0];
        queue.write_buffer(&self.gpu_triple_buffers.next_cell_id, 0, bytemuck::cast_slice(&next_id));
        // Reset deterministic cell addition system
        self.gpu_triple_buffers.reset_slot_allocator();
        
        // Mark GPU buffers as needing sync (will be no-op since cell_count is 0)
        self.gpu_triple_buffers.mark_needs_sync();
    }
    
    /// Remove unused genomes from the end of the genomes list.
    /// Called when cells are removed to free up genome slots for reuse.
    pub fn compact_genomes(&mut self) {
        if self.genomes.is_empty() || self.current_cell_count == 0 {
            self.genomes.clear();
            self.parent_make_adhesion_flags.clear();
            return;
        }
        
        // Note: Without canonical state, we can't determine which genomes are in use
        // This method is kept for API compatibility but doesn't perform compaction
        // Genome compaction will be handled by GPU-based cell management in future tasks
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
    fn run_physics(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, delta_time: f32, world_diameter: f32) {
        if self.current_cell_count == 0 {
            return;
        }
        
        self.debug_frame_counter += 1;
        
        // DEBUG: Track cell count changes
        let gpu_count = self.current_cell_count;
        if self.debug_last_cell_count > 0 && gpu_count > 0 && gpu_count < self.debug_last_cell_count as u32 / 2 {
            // Cell count dropped significantly - potential bug
        }
        if gpu_count > 0 {
            self.debug_last_cell_count = gpu_count as usize;
        }
        
        // DEBUG: Log buffer state on frames 1-3 after an insertion to see what physics does
        if self.debug_frames_since_insertion < 5 {
            self.debug_frames_since_insertion += 1;
        }
        
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
            world_diameter,
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
        self.copy_buffers_to_instance_builder(encoder);
    }
    
    /// Copy data from triple buffers to instance builder
    /// 
    /// This copies positions, rotations, mode_indices, cell_ids, and genome_ids
    /// from the GPU triple buffer system to the instance builder's buffers.
    /// The build_instances shader will use cell_count_buffer to know how many cells to process.
    fn copy_buffers_to_instance_builder(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // For cell insertion, we write to ALL 3 buffer sets, so any buffer index works
        // For physics output, we use output_buffer_index which is where physics wrote
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
        
        // Clear dirty flags since we just copied from GPU
        self.instance_builder.clear_positions_dirty();
        self.instance_builder.clear_rotations_dirty();
        self.instance_builder.clear_mode_indices_dirty();
        self.instance_builder.clear_cell_ids_dirty();
        self.instance_builder.clear_genome_ids_dirty();
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
        
        // Sync child mode indices for division (CRITICAL: determines what mode children get)
        self.gpu_triple_buffers.sync_child_mode_indices(queue, &self.genomes);
        
        // Sync genome mode data (child orientations, split direction) for division
        self.gpu_triple_buffers.sync_genome_mode_data(queue, &self.genomes);
        
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
    
    /// Queue a cell insertion to be processed during the next render frame.
    /// This allows cell insertion to be initiated from input handling without needing device/encoder/queue.
    pub fn queue_cell_insertion(&mut self, world_position: glam::Vec3, genome: Genome) {
        self.pending_cell_insertion = Some((world_position, genome));
    }
    
    /// Process any pending cell insertion during render frame when device/encoder/queue are available.
    /// Returns true if a cell was inserted.
    pub fn process_pending_insertion(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        if let Some((world_position, genome)) = self.pending_cell_insertion.take() {
            self.insert_cell_from_genome(device, encoder, queue, world_position, &genome).is_some()
        } else {
            false
        }
    }
    
    /// Insert a cell at the given world position using genome settings.
    /// Adds the genome to the scene if not already present (does not overwrite existing genomes).
    /// Returns the index of the inserted cell, or None if at capacity.
    pub fn insert_cell_from_genome(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        world_position: glam::Vec3,
        genome: &Genome,
    ) -> Option<usize> {
        // Check capacity
        if self.current_cell_count >= self.gpu_triple_buffers.capacity {
            return None;
        }
        
        // Find or add the genome
        let genome_count_before = self.genomes.len();
        let genome_id = self.add_genome(genome.clone());
        let genome_was_added = self.genomes.len() > genome_count_before;
        let mode_idx = genome.initial_mode.max(0) as usize;
        
        // If a new genome was added, sync settings to GPU
        if genome_was_added {
            self.sync_adhesion_settings(queue);
        }
        
        // Calculate initial radius from mass (mass = 4/3 * pi * r^3 for unit density)
        let initial_mass = 1.0_f32;
        
        // Initialize GPU systems if not already done
        self.initialize_gpu_systems(device, queue);
        
        // Update physics params buffer with cell_capacity before insertion
        // The cell insertion shader reads params.cell_capacity to check limits
        self.update_physics_params_for_insertion(queue);
        
        // Use GPU cell insertion system
        if let Some(ref cell_insertion) = self.cell_insertion {
            cell_insertion.insert_cell_with_id(
                encoder,
                queue,
                world_position,
                glam::Vec3::ZERO,                    // velocity
                initial_mass,                        // mass
                genome.initial_orientation,          // rotation
                genome_id as u32,                    // genome_id
                mode_idx as u32,                     // mode_index (local to this genome)
                self.current_time,                   // birth_time
                self.next_cell_id,                   // cell_id
                &self.genomes,
            );
            
            // Update local tracking
            let cell_index = self.current_cell_count as usize;
            self.current_cell_count += 1;
            self.next_cell_id += 1;
            
            // Reset debug counter to track what happens after additions
            self.debug_frames_since_insertion = 0;
            
            Some(cell_index)
        } else {
            None
        }
    }
    
    /// Update a cell's position using GPU operations
    /// 
    /// This method queues a position update to be executed during the next render phase.
    /// It replaces the canonical state-based position updates.
    pub fn update_cell_position_gpu(&mut self, cell_index: u32, new_position: glam::Vec3) {
        // Note: We don't check bounds here because the GPU cell count may be higher than
        // current_cell_count due to cell division. The shader will validate bounds using
        // the actual GPU cell_count_buffer value.
        
        // Queue the position update to be executed during render
        // The shader will validate cell_index against the GPU cell count
        self.pending_position_update = Some((cell_index, new_position));
    }
    
    /// Update a cell's position using GPU operations.
    /// Used by the drag tool to move cells.
    pub fn set_cell_position(&mut self, cell_idx: usize, new_position: glam::Vec3) {
        // Queue the position update - shader will validate bounds
        self.pending_position_update = Some((cell_idx as u32, new_position));
    }
    
    /// Update a cell's mass using GPU operations.
    /// Used by the boost tool to increase cell mass.
    pub fn set_cell_mass(&mut self, cell_idx: usize) {
        if cell_idx >= self.current_cell_count as usize {
            return;
        }
        
        // TODO: Implement GPU mass update using GpuToolOperations
        // This will use the GPU mass update compute shader when integrated
        // For now, this is a placeholder - actual GPU mass update will be implemented
        // when mass update compute shaders are added to the system
        
        // Mark radii dirty so instance builder updates
        self.instance_builder.mark_radii_dirty();
    }
    
    /// Find the cell closest to a world position within a given radius.
    /// This method is deprecated - use start_cell_selection_query() instead.
    /// Returns None - use the async query system for cell selection.
    pub fn find_cell_at_position(&self) -> Option<usize> {
        // Use the async query system via start_cell_selection_query()
        None
    }
    
    /// Remove a cell at the given index using GPU operations.
    /// Uses GPU-based cell removal without CPU state management.
    /// The cell's mass is set to 0, which causes the lifecycle death scan shader
    /// to mark it as dead and handle the actual removal through the lifecycle pipeline.
    pub fn remove_cell(&mut self, cell_idx: usize) -> bool {
        // Note: We don't check bounds here because the GPU cell count may be higher than
        // current_cell_count due to cell division. The shader will validate bounds using
        // the actual GPU cell_count_buffer value.
        
        // Queue the cell removal to be executed during render
        // The shader will validate cell_idx against the GPU cell count
        self.pending_cell_removal = Some(cell_idx as u32);
        
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
        
        true
    }
    
    /// Cast a ray from screen coordinates and find the closest cell that intersects.
    /// This method is deprecated - use start_drag_selection_query() instead.
    /// Returns None - use the async query system for cell selection.
    pub fn raycast_cell(&mut self) -> Option<(usize, f32)> {
        // Use the async query system via start_drag_selection_query()
        None
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
    
    /// Initialize all GPU systems for the scene
    /// 
    /// This method creates and initializes all GPU systems including:
    /// - GpuCellInspector for real-time cell data extraction
    /// - GpuCellInsertion for direct GPU cell creation  
    /// - GpuToolOperations for spatial queries and position updates
    /// Start a GPU spatial query for cell selection (inspect tool)
    /// 
    /// This method queues a GPU spatial query to find the closest cell to the given screen position.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// The result can be polled later using poll_inspect_tool_results().
    pub fn start_cell_selection_query(&mut self, screen_x: f32, screen_y: f32) {
        // Store screen coordinates for ray computation during render
        self.pending_query_position = Some((screen_x, screen_y, ToolQueryType::Inspect));
    }
    
    /// Start a GPU spatial query for drag tool cell selection
    /// 
    /// This method queues a GPU spatial query to find the closest cell for dragging.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// The result can be polled later using poll_drag_tool_results().
    pub fn start_drag_selection_query(&mut self, screen_x: f32, screen_y: f32) {
        // Store screen coordinates for ray computation during render
        self.pending_query_position = Some((screen_x, screen_y, ToolQueryType::Drag));
    }
    
    /// Start a GPU spatial query for cell removal (remove tool)
    /// 
    /// This method queues a GPU spatial query to find the closest cell for removal.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// The result can be polled later using poll_remove_tool_results().
    pub fn start_remove_tool_query(&mut self, screen_x: f32, screen_y: f32) {
        // Store screen coordinates for ray computation during render
        self.pending_query_position = Some((screen_x, screen_y, ToolQueryType::Remove));
    }
    
    /// Start a GPU spatial query for cell boost (boost tool)
    /// 
    /// This method queues a GPU spatial query to find the closest cell for boosting.
    /// The query will be executed during the next render phase when GPU resources are available.
    /// The result can be polled later using poll_boost_tool_results().
    pub fn start_boost_tool_query(&mut self, screen_x: f32, screen_y: f32) {
        // Store screen coordinates for ray computation during render
        self.pending_query_position = Some((screen_x, screen_y, ToolQueryType::Boost));
    }
    
    /// Poll for remove tool spatial query results
    /// 
    /// This method checks for completed spatial query results and removes the selected cell.
    /// Uses the GPU spatial query system for cell selection.
    pub fn poll_remove_tool_results(&mut self) {
        // Check for pending result from GPU spatial query
        if let Some(cell_idx) = self.pending_remove_result.take() {
            // Remove the cell
            self.remove_cell(cell_idx);
        }
    }
    
    /// Poll for boost tool spatial query results
    /// 
    /// This method checks for completed spatial query results and queues the cell for boosting.
    /// Uses the GPU spatial query system for cell selection.
    pub fn poll_boost_tool_results(&mut self) {
        // Check for pending result from GPU spatial query
        if let Some(cell_idx) = self.pending_boost_result.take() {
            // Queue the cell boost to be executed during render
            self.pending_cell_boost = Some(cell_idx as u32);
        }
    }

    /// Poll for inspect tool spatial query results
    /// 
    /// This method checks for completed spatial query results and updates the radial menu state.
    /// Uses the GPU spatial query system for cell selection.
    /// Also queues cell data extraction to be executed during the next render phase.
    pub fn poll_inspect_tool_results(&mut self, radial_menu: &mut crate::ui::radial_menu::RadialMenuState) {
        // Check for pending result from GPU spatial query
        if let Some(cell_idx) = self.pending_inspect_result.take() {
            radial_menu.inspected_cell = Some(cell_idx);
            // Queue cell data extraction to be executed during render phase
            self.pending_cell_extraction = Some(cell_idx as u32);
            log::info!("Inspecting cell {}", cell_idx);
        }
    }
    
    /// Poll for drag tool spatial query results
    /// 
    /// This method checks for completed spatial query results and starts dragging if a cell was found.
    /// Uses the GPU spatial query system for cell selection.
    pub fn poll_drag_tool_results(&mut self, radial_menu: &mut crate::ui::radial_menu::RadialMenuState, drag_distance: &mut f32) {
        // Check for pending result from GPU spatial query
        if let Some((cell_idx, distance)) = self.pending_drag_result.take() {
            radial_menu.start_dragging(cell_idx);
            *drag_distance = distance;
            log::info!("Started dragging cell {} at distance {}", cell_idx, distance);
        }
    }
    
    /// Execute pending tool queries using GPU spatial query system
    /// 
    /// This method should be called during render when device/encoder/queue are available.
    /// It dispatches the GPU spatial query compute shader and initiates async readback.
    pub fn execute_pending_tool_queries(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) {
        // Check if there's a pending query
        let pending = self.pending_query_position.take();
        if pending.is_none() {
            return;
        }
        
        let (screen_x, screen_y, query_type) = pending.unwrap();
        
        // Initialize GPU systems if needed
        self.initialize_gpu_systems(device, queue);
        
        // Compute ray from screen coordinates
        let (ray_origin, ray_direction) = self.screen_to_ray(screen_x, screen_y);
        
        // Execute GPU spatial query with ray-sphere intersection
        if let Some(ref mut tool_ops) = self.tool_operations {
            let output_idx = self.gpu_triple_buffers.output_buffer_index();
            let physics_bind_group = &self.cached_bind_groups.physics[output_idx];
            
            // Use capacity for dispatch - the shader reads actual cell_count from cell_count_buffer
            // This ensures all cells are checked even if current_cell_count is out of sync with GPU
            let dispatch_count = self.gpu_triple_buffers.capacity;
            tool_ops.find_cell_with_ray(
                encoder,
                physics_bind_group,
                ray_origin,
                ray_direction,
                1000.0, // max_distance - raycast up to 1000 units
                dispatch_count,
            );
            
            // Store the query type for result routing
            self.active_query_type = Some(query_type);
        }
    }
    
    /// Convert screen coordinates to a ray (origin and direction) for raycasting
    /// 
    /// Returns (ray_origin, ray_direction) where ray_origin is the camera position
    /// and ray_direction is the normalized direction from camera through the screen point.
    fn screen_to_ray(&self, screen_x: f32, screen_y: f32) -> (glam::Vec3, glam::Vec3) {
        let width = self.renderer.width as f32;
        let height = self.renderer.height as f32;
        
        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let ndc_x = (2.0 * screen_x / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / height);
        
        // Calculate ray direction in view space
        let aspect = width / height;
        let fov = 45.0_f32.to_radians();
        let tan_half_fov = (fov / 2.0).tan();
        
        let ray_view = glam::Vec3::new(
            ndc_x * aspect * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        ).normalize();
        
        // Transform ray direction to world space
        let ray_direction = self.camera.rotation * ray_view;
        let ray_origin = self.camera.position();
        
        (ray_origin, ray_direction)
    }
    
    /// Poll GPU spatial query results and route to appropriate tool result
    /// 
    /// This method polls the GPU spatial query system for completed results
    /// and routes them to the appropriate pending result field based on query type.
    pub fn poll_spatial_query_results(&mut self) {
        // Check if there's an active query
        let query_type = match self.active_query_type.take() {
            Some(qt) => qt,
            None => return,
        };
        
        // Poll for spatial query completion
        if let Some(ref mut tool_ops) = self.tool_operations {
            if let Some(result) = tool_ops.poll_spatial_query() {
                if result.found != 0 {
                    let cell_idx = result.found_cell_index as usize;
                    let distance = result.distance();
                    
                    // Route result to appropriate pending field
                    match query_type {
                        ToolQueryType::Inspect => {
                            self.pending_inspect_result = Some(cell_idx);
                        }
                        ToolQueryType::Drag => {
                            self.pending_drag_result = Some((cell_idx, distance));
                        }
                        ToolQueryType::Remove => {
                            self.pending_remove_result = Some(cell_idx);
                        }
                        ToolQueryType::Boost => {
                            self.pending_boost_result = Some(cell_idx);
                        }
                    }
                    
                    log::info!("Spatial query found cell {} at distance {}", cell_idx, distance);
                } else {
                    log::info!("Spatial query found no cells");
                }
            } else {
                // Query still in progress, put the query type back
                self.active_query_type = Some(query_type);
            }
        }
    }
    
    /// Execute pending position updates using GPU compute shader
    /// 
    /// This method should be called during render when device/encoder/queue are available.
    /// It dispatches the GPU position update compute shader for drag tool operations.
    /// Returns true if a position update was executed.
    pub fn execute_pending_position_updates(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        // Check if there's a pending position update
        let pending = self.pending_position_update.take();
        if pending.is_none() {
            return false;
        }
        
        let (cell_index, new_position) = pending.unwrap();
        
        // Initialize GPU systems if needed
        self.initialize_gpu_systems(device, queue);
        
        // Update physics params buffer with world_size for boundary clamping
        // This is needed because the position update shader reads params.world_size
        self.update_physics_params_for_position_update(queue);
        
        // Execute GPU position update
        if let Some(ref mut tool_ops) = self.tool_operations {
            tool_ops.update_cell_position(
                encoder,
                cell_index,
                new_position,
            );
            return true;
        }
        
        false
    }
    
    /// Update physics params buffer for position update operations
    fn update_physics_params_for_position_update(&self, queue: &wgpu::Queue) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PhysicsParams {
            delta_time: f32,
            current_time: f32,
            current_frame: i32,
            cell_count: u32,
            world_size: f32,
            boundary_stiffness: f32,
            gravity: f32,
            acceleration_damping: f32,
            grid_resolution: i32,
            grid_cell_size: f32,
            max_cells_per_grid: i32,
            enable_thrust_force: i32,
            cell_capacity: u32,
            _padding2: [f32; 3],
            _padding: [f32; 48],
        }
        
        let world_diameter = self.config.sphere_radius * 2.0;
        let params = PhysicsParams {
            delta_time: 0.0,
            current_time: self.current_time,
            current_frame: 0,
            cell_count: self.current_cell_count,
            world_size: world_diameter,
            boundary_stiffness: 500.0,
            gravity: 0.0,
            acceleration_damping: 0.98,
            grid_resolution: 64,
            grid_cell_size: world_diameter / 64.0,
            max_cells_per_grid: 16,
            enable_thrust_force: 0,
            cell_capacity: self.gpu_triple_buffers.capacity,
            _padding2: [0.0; 3],
            _padding: [0.0; 48],
        };
        
        queue.write_buffer(&self.gpu_triple_buffers.physics_params, 0, bytemuck::bytes_of(&params));
    }
    
    /// Execute pending cell removals using GPU compute shader
    /// 
    /// This method should be called during render when device/encoder/queue are available.
    /// It dispatches the GPU cell removal compute shader to mark cells for death.
    /// Returns true if a cell removal was executed.
    pub fn execute_pending_cell_removals(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        // Check if there's a pending cell removal
        let pending = self.pending_cell_removal.take();
        if pending.is_none() {
            return false;
        }
        
        let cell_index = pending.unwrap();
        
        // Initialize GPU systems if needed
        self.initialize_gpu_systems(device, queue);
        
        // Update physics params buffer (needed for cell count validation in shader)
        self.update_physics_params_for_position_update(queue);
        
        // Execute GPU cell removal
        if let Some(ref mut tool_ops) = self.tool_operations {
            tool_ops.remove_cell(
                encoder,
                cell_index,
            );
            return true;
        }
        
        false
    }
    
    /// Execute pending cell boosts using GPU compute shader
    /// 
    /// This method should be called during render when device/encoder/queue are available.
    /// It dispatches the GPU cell boost compute shader to set cell mass to maximum.
    /// Returns true if a cell boost was executed.
    pub fn execute_pending_cell_boosts(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        // Check if there's a pending cell boost
        let pending = self.pending_cell_boost.take();
        if pending.is_none() {
            return false;
        }
        
        let cell_index = pending.unwrap();
        
        // Initialize GPU systems if needed
        self.initialize_gpu_systems(device, queue);
        
        // Update physics params buffer (needed for cell count validation in shader)
        self.update_physics_params_for_position_update(queue);
        
        // Execute GPU cell boost
        if let Some(ref mut tool_ops) = self.tool_operations {
            tool_ops.boost_cell(
                encoder,
                cell_index,
            );
            return true;
        }
        
        false
    }

    /// Execute pending cell extraction using GPU compute shader
    /// 
    /// This method should be called during render when device/encoder/queue are available.
    /// It dispatches the GPU cell data extraction compute shader for the inspect tool.
    /// Returns true if a cell extraction was executed.
    pub fn execute_pending_cell_extraction(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        // Check if there's a pending cell extraction
        let pending = self.pending_cell_extraction.take();
        if pending.is_none() {
            return false;
        }
        
        let cell_index = pending.unwrap();
        
        // Initialize GPU systems if needed
        self.initialize_gpu_systems(device, queue);
        
        // Execute GPU cell data extraction
        self.extract_cell_data(device, queue, encoder, cell_index);
        
        true
    }

    /// Extract cell data using GPU compute shader with async readback management
    /// 
    /// This method uploads the cell index, dispatches the compute shader,
    /// and initiates async readback. Call poll_cell_extraction() to check for completion.
    pub fn extract_cell_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cell_index: u32,
    ) {
        // Initialize GPU systems if not already done
        self.initialize_gpu_systems(device, queue);
        
        if let Some(ref mut inspector) = self.cell_inspector {
            inspector.extract_cell_data(encoder, queue, cell_index);
        }
    }
    
    /// Poll for cell extraction completion and return extracted data if available
    /// 
    /// Returns Some(data) if extraction is complete, None if still in progress.
    pub fn poll_cell_extraction(&mut self, device: Option<&wgpu::Device>) -> Option<crate::simulation::gpu_physics::InspectedCellData> {
        if let Some(ref mut inspector) = self.cell_inspector {
            inspector.poll_extraction(device)
        } else {
            None
        }
    }
    
    /// Check if cell extraction is currently in progress
    pub fn is_extracting_cell_data(&self) -> bool {
        self.cell_inspector.as_ref().map_or(false, |i| i.is_extracting())
    }
    
    /// Get the most recent cell extraction result
    pub fn get_latest_cell_extraction(&self) -> Option<&crate::simulation::gpu_physics::ReadbackResult> {
        self.cell_inspector.as_ref().and_then(|i| i.get_latest_result())
    }
    
    /// Clear cached cell extraction data
    pub fn clear_cell_extraction_cache(&mut self) {
        if let Some(ref mut inspector) = self.cell_inspector {
            inspector.clear_cache();
        }
    }
    
    /// Initialize all GPU systems for the scene
    /// 
    /// This method creates and initializes all GPU systems including:
    /// - GpuCellInspector for real-time cell data extraction
    /// - GpuCellInsertion for direct GPU cell creation  
    /// - GpuToolOperations for spatial queries and position updates
    /// - AsyncReadbackManager for coordinating GPU-to-CPU transfers
    /// 
    /// Must be called after scene creation to enable GPU-only operations.
    /// Implements requirements 2.1, 3.1, 4.1, 11.1.
    pub fn initialize_gpu_systems(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Initialize async readback manager first (requirement 11.1)
        if self.readback_manager.is_none() {
            let max_concurrent_readbacks = 4; // Limit concurrent operations (requirement 11.6)
            self.readback_manager = Some(AsyncReadbackManager::new(device.clone(), max_concurrent_readbacks));
        }
        
        // Initialize GPU cell inspector (requirement 3.1)
        if self.cell_inspector.is_none() {
            let buffer_index = self.gpu_triple_buffers.output_buffer_index();
            let cell_inspector = GpuCellInspector::new(
                device,
                self.gpu_physics_pipelines.cell_data_extraction.clone(),
                &self.gpu_physics_pipelines.physics_layout,
                &self.gpu_physics_pipelines.cell_extraction_params_layout,
                &self.gpu_physics_pipelines.cell_extraction_state_layout,
                &self.gpu_physics_pipelines.cell_extraction_output_layout,
                &self.gpu_triple_buffers,
                buffer_index,
            );
            self.cell_inspector = Some(cell_inspector);
        }
        
        // Initialize GPU cell insertion system (requirement 2.1)
        if self.cell_insertion.is_none() {
            let cell_insertion = GpuCellInsertion::new(
                device,
                self.gpu_physics_pipelines.cell_insertion.clone(),
                &self.gpu_physics_pipelines.cell_insertion_physics_layout,
                &self.gpu_physics_pipelines.cell_insertion_params_layout,
                &self.gpu_physics_pipelines.cell_insertion_state_layout,
                &self.gpu_triple_buffers,
            );
            self.cell_insertion = Some(cell_insertion);
        }
        
        // Initialize GPU tool operations system (requirement 4.1)
        if self.tool_operations.is_none() {
            use std::sync::Arc;
            let device_arc = Arc::new(device.clone());
            let queue_arc = Arc::new(queue.clone());
            
            let tool_operations = GpuToolOperations::new(
                device_arc,
                queue_arc,
                &self.gpu_physics_pipelines,
                &self.gpu_triple_buffers,
            );
            self.tool_operations = Some(tool_operations);
        }
    }
    
    /// Update physics params buffer with cell_capacity for cell insertion
    /// 
    /// The cell insertion shader reads params.cell_capacity to check if there's room
    /// for new cells. This must be called before cell insertion to ensure the
    /// capacity is set correctly.
    fn update_physics_params_for_insertion(&self, queue: &wgpu::Queue) {
        // Physics params struct matching the WGSL shader
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PhysicsParams {
            delta_time: f32,
            current_time: f32,
            current_frame: i32,
            cell_count: u32,
            world_size: f32,
            boundary_stiffness: f32,
            gravity: f32,
            acceleration_damping: f32,
            grid_resolution: i32,
            grid_cell_size: f32,
            max_cells_per_grid: i32,
            enable_thrust_force: i32,
            cell_capacity: u32,
            _padding2: [f32; 3],
            _padding: [f32; 48],
        }
        
        let world_diameter = self.config.sphere_radius * 2.0;
        let params = PhysicsParams {
            delta_time: 0.0,
            current_time: self.current_time,
            current_frame: 0,
            cell_count: 0, // Not used by insertion shader
            world_size: world_diameter,
            boundary_stiffness: 500.0,
            gravity: 0.0,
            acceleration_damping: 0.98,
            grid_resolution: 64,
            grid_cell_size: world_diameter / 64.0,
            max_cells_per_grid: 16,
            enable_thrust_force: 0,
            cell_capacity: self.gpu_triple_buffers.capacity,
            _padding2: [0.0; 3],
            _padding: [0.0; 48],
        };
        
        queue.write_buffer(&self.gpu_triple_buffers.physics_params, 0, bytemuck::bytes_of(&params));
    }

    /// Convert screen coordinates to world position at a fixed distance from camera.
    /// Uses a constant distance so cells are always inserted at the same depth.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> glam::Vec3 {
        const INSERT_DISTANCE: f32 = 25.0; // Fixed distance for cell insertion
        self.screen_to_world_at_distance(screen_x, screen_y, INSERT_DISTANCE)
    }

    /// Clear all cached readback results
    /// 
    /// Clears cached results from all GPU systems. Call this when scene state
    /// changes significantly to invalidate cached data.
    pub fn clear_readback_caches(&mut self) {
        if let Some(ref mut inspector) = self.cell_inspector {
            inspector.clear_cache();
        }
        
        if let Some(ref mut tool_operations) = self.tool_operations {
            tool_operations.clear_cache();
        }
        
        if let Some(ref mut readback_manager) = self.readback_manager {
            readback_manager.clear_completed();
        }
    }

    /// Poll all async readback operations for completion
    /// 
    /// This method should be called each frame to check for completed async readbacks
    /// from all GPU systems. It coordinates readback operations across:
    /// - Cell inspector data extraction
    /// - Tool operation spatial queries
    /// - Any other GPU-to-CPU transfers
    /// 
    /// Implements requirement 11.3 for non-blocking async readback polling.
    pub fn poll_async_readbacks(&mut self, device: &wgpu::Device) {
        // Poll async readback manager if available
        if let Some(ref mut readback_manager) = self.readback_manager {
            readback_manager.poll_completions();
        }
        
        // Poll cell inspector for extraction completion
        if let Some(ref mut inspector) = self.cell_inspector {
            if let Some(_data) = inspector.poll_extraction(Some(device)) {
                // Cell inspector data extraction completed
                // Data is now cached in the inspector for UI access
            }
        }
        
        // Poll tool operations for spatial query completion
        if let Some(ref mut tool_operations) = self.tool_operations {
            if let Some(_result) = tool_operations.poll_spatial_query() {
                // Spatial query completed
                // Result is now cached in tool operations for tool access
            }
        }
    }
}


impl Scene for GpuScene {
    fn update(&mut self, dt: f32) {
        if self.paused || self.current_cell_count == 0 {
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
        world_diameter: f32,
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

        // TODO: Update instance builder to work without canonical state
        // This will be updated in task 13 when GPU systems are fully integrated
        // For now, we skip the instance builder update since we don't have canonical state
        // The instance builder will be updated to read directly from GPU buffers

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

        // Process any pending cell insertion from input handling
        let cell_inserted = self.process_pending_insertion(device, &mut encoder, queue);

        // Execute any pending tool queries (spatial queries for inspect/drag/remove/boost)
        self.execute_pending_tool_queries(device, &mut encoder, queue);
        
        // Poll for cell extraction completion (inspect tool async readback)
        // This needs to be called each frame to check for completed async readbacks
        if let Some(ref mut inspector) = self.cell_inspector {
            inspector.poll_extraction(Some(device));
        }

        // If a cell was inserted, copy buffers to instance builder before physics
        // The Hi-Z buffer doesn't have valid depth data for the new cell's position yet
        // (Hi-Z is generated at end of frame from the depth buffer)
        if cell_inserted {
            self.instance_builder.set_culling_mode(CullingMode::FrustumOnly);
            self.copy_buffers_to_instance_builder(&mut encoder);
        }

        // Execute GPU physics pipeline if not paused and has cells
        // Use fixed timestep accumulator for consistent physics behavior
        if !self.paused && self.current_cell_count > 0 {
            let fixed_dt = self.config.fixed_timestep;
            // Allow more steps when time_scale > 1 (fast forward)
            let max_steps = (4.0 * self.time_scale).ceil() as i32;
            let mut steps = 0;
            
            while self.time_accumulator >= fixed_dt && steps < max_steps {
                self.run_physics(device, &mut encoder, queue, fixed_dt, world_diameter);
                self.current_time += fixed_dt;
                self.time_accumulator -= fixed_dt;
                steps += 1;
            }
            
            // If we hit max steps, discard remaining accumulated time
            if steps >= max_steps {
                self.time_accumulator = 0.0;
            }
        }
        
        // Execute any pending position updates (drag tool) AFTER physics
        // This ensures the dragged cell's position is set after physics has run,
        // so physics doesn't overwrite the user's drag position
        let position_updated = self.execute_pending_position_updates(device, &mut encoder, queue);
        
        // Execute any pending cell removals (remove tool) AFTER physics
        // This marks cells for death by setting their mass to 0
        // The lifecycle pipeline will handle the actual removal on the next physics step
        let cell_removed = self.execute_pending_cell_removals(device, &mut encoder, queue);
        
        // Execute any pending cell boosts (boost tool) AFTER physics
        // This sets cell mass to maximum to trigger division
        let cell_boosted = self.execute_pending_cell_boosts(device, &mut encoder, queue);
        
        // Execute any pending cell extractions (inspect tool) AFTER physics
        // This extracts detailed cell data for the Cell Inspector panel
        let _cell_extracted = self.execute_pending_cell_extraction(device, &mut encoder, queue);
        
        // Copy buffers to instance builder after position update, cell removal, or cell boost
        if position_updated || cell_removed || cell_boosted {
            self.copy_buffers_to_instance_builder(&mut encoder);
        }

        // Build instances with GPU culling (compute pass)
        // Calculate total mode count across all genomes
        // Use capacity for dispatch - shader reads actual cell_count from GPU buffer
        let total_mode_count: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        
        // Update instance builder with genome information for mode visuals
        // Even though we use GPU buffers for cell data, we still need genome info for colors
        if !self.genomes.is_empty() {
            // Force update of mode visuals directly since we don't need full state update
            self.instance_builder.update_mode_visuals_from_genomes(device, queue, &self.genomes);
        }
        
        // Update cell type visuals for membrane noise and lighting parameters
        if let Some(visuals) = cell_type_visuals {
            self.instance_builder.update_cell_type_visuals_direct(device, queue, visuals);
        }
        
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
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
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
            // Update world sphere radius to match current world diameter
            self.world_sphere_renderer.set_radius(queue, world_diameter * 0.5);
            
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
        if self.current_cell_count > 0 && self.instance_builder.culling_mode() != CullingMode::Disabled {
            self.hiz_generator.generate(
                device,
                queue,
                &mut encoder,
                &self.renderer.depth_view,
            );
        }
        
        // Start async cell count readback (copy to readback buffer)
        // Only start if no readback is pending
        let should_start_readback = !self.gpu_triple_buffers.is_cell_count_read_pending();
        if should_start_readback {
            self.gpu_triple_buffers.start_cell_count_read(&mut encoder);
        }

        // Single submit for all GPU work
        queue.submit(std::iter::once(encoder.finish()));
        
        // Initiate the async map operation after submit (if we started a readback)
        if should_start_readback {
            self.gpu_triple_buffers.initiate_cell_count_map();
        }
        
        // Poll for cell count readback completion and update current_cell_count
        if let Some(count) = self.gpu_triple_buffers.poll_cell_count(device) {
            self.current_cell_count = count;
        }
        
        // Track cell count changes
        self.debug_last_cell_count = self.current_cell_count as usize;
        
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
        self.current_cell_count as usize
    }
}
