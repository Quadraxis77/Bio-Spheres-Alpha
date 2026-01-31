//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::cell::types::CellType;
use crate::genome::Genome;
use crate::rendering::{CaveSystemRenderer, CellRenderer, CullingMode, GpuAdhesionLineRenderer, GpuSurfaceNets, HizGenerator, InstanceBuilder, NutrientParticleRenderer, SteamParticleRenderer, TailRenderer, VoxelRenderer, WaterParticleRenderer, WorldSphereRenderer};
use crate::scene::Scene;
use crate::simulation::{PhysicsConfig};
use crate::simulation::fluid_simulation::{FluidBuffers, GpuFluidSimulator, SolidMaskGenerator};
use crate::simulation::gpu_physics::{execute_gpu_physics_step, execute_lifecycle_pipeline, CachedBindGroups, GpuPhysicsPipelines, GpuTripleBufferSystem, AdhesionBuffers, GpuCellInspector, GpuCellInsertion, GpuToolOperations, AsyncReadbackManager, GenomeBufferManager};
use crate::ui::camera::CameraController;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

/// Camera uniform for voxel rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

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
    /// Genome buffer manager for per-genome GPU resources
    pub genome_buffer_manager: GenomeBufferManager,
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
    /// Whether to show the world boundary sphere
    pub show_world_sphere: bool,
    /// LOD scale factor for distance calculations
    pub lod_scale_factor: f32,
    /// LOD threshold for Low to Medium transition
    pub lod_threshold_low: f32,
    /// LOD threshold for Medium to High transition
    pub lod_threshold_medium: f32,
    /// LOD threshold for High to Ultra transition
    pub lod_threshold_high: f32,
    /// Whether to show debug colors for LOD levels
    pub lod_debug_colors: bool,
    /// Gravity strength for physics simulation
    pub gravity: f32,
    /// Gravity direction flags (X, Y, Z)
    pub gravity_dir: [bool; 3],
    /// Per-fluid-type lateral flow probabilities for fluid simulation (0.0 to 1.0)
    /// Index: 0=Empty (unused), 1=Water, 2=Lava, 3=Steam
    pub lateral_flow_probabilities: [f32; 4],
    /// Phase change probabilities for fluid simulation (0.0 to 1.0)
    pub condensation_probability: f32,  // Steam to Water
    pub vaporization_probability: f32,  // Water to Steam
    /// Nutrient particle density for noise-based spawning (0.0 to 1.0)
    pub nutrient_density: f32,  // Controls threshold for nutrient spawning
    /// Current cell count (tracked on GPU, no CPU canonical state)
    pub current_cell_count: u32,
    /// Next cell ID for deterministic cell creation
    next_cell_id: u32,
    /// Tail renderer for flagellocyte cells
    pub tail_renderer: TailRenderer,
    /// Cave system renderer for procedural cave generation and collision
    pub cave_renderer: Option<CaveSystemRenderer>,
    /// Flag to indicate cave params need GPU update
    pub cave_params_dirty: bool,
    /// Cave-specific physics bind groups (one per buffer index)
    cave_physics_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Track previous world diameter to detect changes for cave regeneration
    previous_world_diameter: f32,
    /// Fluid simulation buffers (128³ voxel grid)
    pub fluid_buffers: Option<FluidBuffers>,
    /// Voxel renderer for fluid visualization
    pub voxel_renderer: Option<VoxelRenderer>,
    /// Whether fluid visualization is enabled
    pub show_fluid_voxels: bool,
    /// Current voxel instance count for rendering
    voxel_instance_count: u32,
        /// GPU-based surface nets renderer for density field visualization
    pub gpu_surface_nets: Option<GpuSurfaceNets>,
    /// Whether to show GPU surface nets density mesh
    pub show_gpu_density_mesh: bool,
    /// GPU fluid simulator for falling/stacking water
    pub fluid_simulator: Option<GpuFluidSimulator>,
    /// Solid mask generator for fluid system
    solid_mask_generator: Option<SolidMaskGenerator>,
    /// Steam particle system renderer
    pub steam_particle_renderer: Option<SteamParticleRenderer>,
    /// Whether to show steam particles
    pub show_steam_particles: bool,
    /// Cached steam particle extract bind group (for compute)
    steam_extract_bind_group: Option<wgpu::BindGroup>,
    /// Cached steam particle render bind group
    steam_render_bind_group: Option<wgpu::BindGroup>,
    /// Water particle system renderer
    pub water_particle_renderer: Option<WaterParticleRenderer>,
    /// Whether to show water particles
    pub show_water_particles: bool,
    /// Water particle prominence factor (0.0-1.0)
    pub water_particle_prominence: f32,
    /// Cached water particle extract bind group (for compute)
    water_extract_bind_group: Option<wgpu::BindGroup>,
    /// Cached water particle render bind group
    water_render_bind_group: Option<wgpu::BindGroup>,
    /// Nutrient particle system renderer
    pub nutrient_particle_renderer: Option<NutrientParticleRenderer>,
    /// Whether to show nutrient particles
    pub show_nutrient_particles: bool,
    /// Nutrient particle spawn probability (0.0-1.0)
    pub nutrient_spawn_probability: f32,
    /// Cached nutrient particle extract bind group (for compute)
    nutrient_extract_bind_group: Option<wgpu::BindGroup>,
    /// Cached nutrient particle render bind group
    nutrient_render_bind_group: Option<wgpu::BindGroup>,
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
        
        // Create genome buffer manager for per-genome GPU resources
        let genome_buffer_manager = GenomeBufferManager::new(crate::simulation::gpu_physics::MAX_GENOMES);
        
        // Create tail renderer for flagellocyte cells
        let tail_renderer = TailRenderer::new(device, surface_config.format, capacity as usize);
        
        // Cave system will be initialized on demand
        let cave_renderer = None;

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
            camera: CameraController::new_for_gpu_scene(),
            current_time: 0.0,
            genomes: Vec::new(),
            genome_buffer_manager,
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
            show_world_sphere: true,
            lod_scale_factor: 500.0,
            lod_threshold_low: 10.0,
            lod_threshold_medium: 25.0,
            lod_threshold_high: 50.0,
            lod_debug_colors: false,
            gravity: 0.0,
            gravity_dir: [false, true, false],
            lateral_flow_probabilities: [1.0, 0.8, 0.6, 0.9],
            condensation_probability: 0.1,
            vaporization_probability: 0.1,
            nutrient_density: 0.2,  // Default nutrient density (20% of 0.5 range)
            current_cell_count: 0,
            next_cell_id: 0,
            tail_renderer,
            cave_renderer,
            cave_params_dirty: false,
            cave_physics_bind_groups: None,
            previous_world_diameter: 400.0, // Default world diameter
            fluid_buffers: None,
            voxel_renderer: None,
            show_fluid_voxels: false,
            voxel_instance_count: 0,
            gpu_surface_nets: None,
            show_gpu_density_mesh: false,
            fluid_simulator: None,
            solid_mask_generator: None,
            steam_particle_renderer: None,
            show_steam_particles: false,
            steam_extract_bind_group: None,
            steam_render_bind_group: None,
            water_particle_renderer: None,
            show_water_particles: false,
            water_particle_prominence: 0.0,
            water_extract_bind_group: None,
            water_render_bind_group: None,
            nutrient_particle_renderer: None,
            show_nutrient_particles: true,
            nutrient_spawn_probability: 0.05,  // 5% spawn probability
            nutrient_extract_bind_group: None,
            nutrient_render_bind_group: None,
        }
    }

    /// Create a new GPU scene with default capacity (100k).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        Self::with_capacity(device, queue, surface_config, 100_000)
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

        // Note: Fluid reset is handled separately via reset_fluid() which requires encoder
    }
    
    /// Remove unused genomes from the scene.
    /// Uses the new reference counting system to safely remove unused genomes.
    /// Returns the number of genomes removed.
    pub fn compact_genomes(&mut self) -> usize {
        if self.genomes.is_empty() || self.current_cell_count == 0 {
            self.genomes.clear();
            self.parent_make_adhesion_flags.clear();
            // Clear all genome buffer groups
            for _ in 0..self.genome_buffer_manager.genome_count() {
                self.genome_buffer_manager.compact();
            }
            return 0;
        }
        
        // Compact the genome buffer manager first
        let removed_count = self.genome_buffer_manager.compact();
        
        // Remove genomes that have no corresponding buffer groups
        let _genome_count_before = self.genomes.len();
        self.genomes.retain(|_genome| true); // Keep all for now - reference counting will handle cleanup
        
        // Rebuild parent_make_adhesion_flags
        self.parent_make_adhesion_flags.clear();
        for genome in &self.genomes {
            for mode in &genome.modes {
                self.parent_make_adhesion_flags.push(mode.parent_make_adhesion);
            }
        }
        
        removed_count
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
    
    /// Set LOD parameters for texture atlas rendering.
    pub fn set_lod_settings(
        &mut self, 
        scale_factor: f32, 
        threshold_low: f32, 
        threshold_medium: f32, 
        threshold_high: f32,
        debug_colors: bool,
    ) {
        self.lod_scale_factor = scale_factor;
        self.lod_threshold_low = threshold_low;
        self.lod_threshold_medium = threshold_medium;
        self.lod_threshold_high = threshold_high;
        self.lod_debug_colors = debug_colors;
    }
    
    /// Set whether GPU readbacks are enabled (cell count, etc.)
    /// Disabling this can improve performance by avoiding CPU-GPU sync overhead.
    pub fn set_readbacks_enabled(&mut self, enabled: bool) {
        self.readbacks_enabled = enabled;
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

        // Execute pure GPU physics pipeline (7 compute shader stages + cave collision if enabled)
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
            self.gravity,
            self.gravity_dir,
            self.cave_renderer.as_ref(),
            self.cave_physics_bind_groups.as_ref(),
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
        
        // CRITICAL: Rebuild spatial grid after lifecycle pipeline
        // This ensures dead cells are not included in collision detection
        crate::simulation::gpu_physics::gpu_scene_integration::rebuild_spatial_grid_after_lifecycle(
            encoder,
            &self.gpu_physics_pipelines,
            &self.gpu_triple_buffers,
            &self.cached_bind_groups,
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
        
        // Copy cell types (u32) - needed for flagella type_data population
        encoder.copy_buffer_to_buffer(
            &self.gpu_triple_buffers.cell_types,
            0,
            self.instance_builder.cell_types_buffer(),
            0,
            u32_copy_size,
        );
        
        // Copy mode properties (12 floats per mode = 48 bytes) - needed for swim_force -> tail_speed calculation
        // Calculate total modes across all genomes
        let total_modes: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        let mode_properties_copy_size = (total_modes * 48) as u64;
        if mode_properties_copy_size > 0 {
            encoder.copy_buffer_to_buffer(
                &self.gpu_triple_buffers.mode_properties,
                0,
                self.instance_builder.mode_properties_buffer(),
                0,
                mode_properties_copy_size,
            );
        }
        
        // Copy mode cell types (1 u32 per mode = 4 bytes) - critical for correct cell type rendering
        // This ensures the InstanceBuilder has the same mode_cell_types as GpuTripleBufferSystem
        let mode_cell_types_copy_size = (total_modes * 4) as u64;
        if mode_cell_types_copy_size > 0 {
            encoder.copy_buffer_to_buffer(
                &self.gpu_triple_buffers.mode_cell_types,
                0,
                self.instance_builder.mode_cell_types_buffer(),
                0,
                mode_cell_types_copy_size,
            );
        }
        
        // Clear dirty flags since we just copied from GPU
        self.instance_builder.clear_positions_dirty();
        self.instance_builder.clear_rotations_dirty();
        self.instance_builder.clear_mode_indices_dirty();
        self.instance_builder.clear_cell_ids_dirty();
        self.instance_builder.clear_genome_ids_dirty();
        self.instance_builder.clear_cell_types_dirty();
    }
    

    
    /// Find an existing genome by name, or return None.
    pub fn find_genome_id(&self, name: &str) -> Option<usize> {
        self.genomes.iter().position(|g| g.name == name)
    }
    
    /// Update the working genome with new settings.
    /// Temporarily disabled to isolate crash issues
    pub fn update_genome(&mut self, _device: &wgpu::Device, genome: &Genome) -> Option<usize> {
        // For now, just update the first genome in place
        if self.genomes.is_empty() {
            self.genomes.push(genome.clone());
            Some(0)
        } else {
            self.genomes[0] = genome.clone();
            Some(0)
        }
    }
    
    /// Add a genome to the scene and return its ID.
    /// Simplified version to isolate crash issues
    pub fn add_genome(&mut self, _device: &wgpu::Device, genome: Genome) -> Option<usize> {
        // For now, just add to CPU storage without GPU buffers
        let id = self.genomes.len();
        self.genomes.push(genome);
        
        log::info!("Added genome {} (total: {})", id, self.genomes.len());
        Some(id)
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
        
        // Sync mode cell types lookup table (for deriving cell_type from mode_index)
        self.gpu_triple_buffers.sync_mode_cell_types(queue, &self.genomes);
        
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
        // Validate bounds
        if genome_id >= self.genomes.len() {
            log::warn!("Invalid genome ID: {} (max: {})", genome_id, self.genomes.len());
            return 0; // Fallback to first mode
        }
        
        let genome = &self.genomes[genome_id];
        if local_mode_index >= genome.modes.len() {
            log::warn!("Invalid local mode index: {} for genome {} (max: {})", 
                local_mode_index, genome_id, genome.modes.len());
            return 0; // Fallback to first mode
        }
        
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
        let genome_id = match self.add_genome(device, genome.clone()) {
            Some(id) => id,
            None => {
                log::error!("Failed to add genome - at maximum capacity");
                return None;
            }
        };
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
            _pad0: f32,
            _pad1: f32,
            _pad2: f32,
        }
        
        let world_diameter = self.config.sphere_radius * 2.0;
        let params = PhysicsParams {
            delta_time: 0.0,
            current_time: self.current_time,
            current_frame: 0,
            cell_count: self.current_cell_count,
            world_size: world_diameter,
            boundary_stiffness: 10.0,
            gravity: 0.0,
            acceleration_damping: 0.95,
            grid_resolution: 64,
            grid_cell_size: world_diameter / 64.0,
            max_cells_per_grid: 16,
            enable_thrust_force: 0,
            cell_capacity: self.gpu_triple_buffers.capacity,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
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
    
    /// Initialize cave system for procedural generation and collision
    pub fn has_cave_renderer(&self) -> bool {
        self.cave_renderer.is_some()
    }
    
    pub fn initialize_cave_system(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat, world_diameter: f32) -> bool {
        if self.cave_renderer.is_none() {
            let width = self.renderer.width;
            let height = self.renderer.height;
            let world_radius = world_diameter * 0.5; // Use actual world diameter instead of config
            
            let cave_renderer = CaveSystemRenderer::new(device, surface_format, width, height, world_radius);
            
            self.cave_renderer = Some(cave_renderer);
            
            // Create cave-specific physics bind groups
            self.create_cave_physics_bind_groups(device);
            
            // Update solid mask after cave system is initialized
            self.update_solid_mask(&queue); // Note: This needs queue parameter
            
            return true; // Cave was just initialized, params need to be applied
        }
        false // Cave was already initialized
    }
    
    /// Create bind groups specifically for cave collision (only bindings 0-3)
    fn create_cave_physics_bind_groups(&mut self, device: &wgpu::Device) {
        if self.cave_renderer.is_none() {
            return;
        }
        
        // Create a bind group layout for cave collision (in-place position/velocity updates)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cave Physics Bind Group Layout"),
            entries: &[
                // @binding(0): PhysicsParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): positions (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): velocities (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): cell_count_buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind groups for each buffer index
        let bind_groups = [
            self.create_cave_physics_bind_group(device, &layout, 0),
            self.create_cave_physics_bind_group(device, &layout, 1),
            self.create_cave_physics_bind_group(device, &layout, 2),
        ];
        
        self.cave_physics_bind_groups = Some(bind_groups);
    }
    
    fn create_cave_physics_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        // Cave collision performs in-place position correction on the buffer that
        // position_update just wrote to. Position_update writes to (buffer_index+1)%3,
        // so cave collision should operate on the same buffer: (buffer_index+1)%3
        let write_buffer = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Cave Physics Bind Group {}", buffer_index)),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gpu_triple_buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_triple_buffers.position_and_mass[write_buffer].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.gpu_triple_buffers.velocity[write_buffer].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.gpu_triple_buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create camera uniform for rendering
    fn create_camera_uniform(&self) -> CameraUniform {
        let camera_pos = self.camera.position();
        let camera_rotation = self.camera.rotation;
        
        // Use NEG_Z for forward direction (consistent with other renderers)
        let view_matrix = Mat4::look_at_rh(
            camera_pos,
            camera_pos + camera_rotation * glam::Vec3::NEG_Z,
            camera_rotation * glam::Vec3::Y,
        );
        let aspect = self.renderer.width as f32 / self.renderer.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;
        
        CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: view_matrix.to_cols_array_2d(),
            proj: proj_matrix.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        }
    }
    
    
    /// Initialize the fluid simulation system
    pub fn initialize_fluid_system(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> bool {
        if self.fluid_buffers.is_some() {
            return false; // Already initialized
        }
        
        // Get world parameters from config
        let world_radius = self.config.sphere_radius;
        let world_center = glam::Vec3::ZERO;
        
        // Create fluid buffers
        let fluid_buffers = FluidBuffers::new(device, world_radius, world_center);
        
        // Validate memory budget
        if let Err(e) = fluid_buffers.validate_memory_budget() {
            log::error!("Fluid system memory budget exceeded: {}", e);
            return false;
        }
        
        log::info!(
            "Fluid system initialized: {:.2} MB memory usage",
            fluid_buffers.memory_usage_mb()
        );
        
        // Create voxel renderer
        let max_voxel_instances = 100_000; // Maximum voxels to render at once
        let voxel_renderer = VoxelRenderer::new(
            device,
            surface_format,
            camera_bind_group_layout,
            max_voxel_instances,
        );
        
        self.fluid_buffers = Some(fluid_buffers);
        self.voxel_renderer = Some(voxel_renderer);
        self.show_fluid_voxels = true;
        
        // Create solid mask generator
        let solid_mask_generator = SolidMaskGenerator::new(
            crate::simulation::fluid_simulation::GRID_RESOLUTION,
            world_center,
            world_radius,
        );
        self.solid_mask_generator = Some(solid_mask_generator);
        
        true
    }
    
    /// Generate test voxels for visualization
    pub fn generate_test_voxels(&mut self, queue: &wgpu::Queue) {
        use crate::rendering::VoxelInstance;
        use crate::simulation::fluid_simulation::{FluidType, GRID_RESOLUTION};
        
        if self.fluid_buffers.is_none() || self.voxel_renderer.is_none() {
            return;
        }
        
        let voxel_renderer = self.voxel_renderer.as_ref().unwrap();
        
        // Calculate grid parameters from world size
        let world_diameter = self.config.sphere_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let world_center = glam::Vec3::ZERO;
        
        // Helper function to convert grid indices to world position
        let grid_to_world = |i: u32, j: u32, k: u32| -> glam::Vec3 {
            // Grid origin is at world_center - (world_diameter / 2)
            let grid_origin = world_center - glam::Vec3::splat(world_diameter / 2.0);
            // Cell center is at grid_origin + (i + 0.5) * cell_size
            grid_origin + glam::Vec3::new(
                (i as f32 + 0.5) * cell_size,
                (j as f32 + 0.5) * cell_size,
                (k as f32 + 0.5) * cell_size,
            )
        };
        
        let mut instances = Vec::new();
        // Voxel size is half-extent. For contiguous voxels:
        // - Grid cells are cell_size apart (center to center)
        // - Full voxel width must equal cell_size
        // - Therefore half-extent = cell_size / 2
        // BUT the size parameter might be full extent, not half-extent. Let's use cell_size.
        let voxel_size = cell_size; // Full extent for rendering
        
        // ===== SOLID (Green) - Cave/Terrain Patterns =====
        // Pattern 1: Single isolated voxel (obstacle)
        let solid_single = (30, 64, 64);
        instances.push(VoxelInstance {
            position: grid_to_world(solid_single.0, solid_single.1, solid_single.2).to_array(),
            voxel_type: FluidType::Solid as u32,
            color: [0.2, 0.8, 0.2, 0.9],
            size: voxel_size,
            _padding: [0.0; 3],
        });
        
        // Pattern 2: Horizontal wall (5x1x3 plane)
        let solid_wall_base = (35, 64, 62);
        for x in 0..5 {
            for z in 0..3 {
                instances.push(VoxelInstance {
                    position: grid_to_world(solid_wall_base.0 + x, solid_wall_base.1, solid_wall_base.2 + z).to_array(),
                    voxel_type: FluidType::Solid as u32,
                    color: [0.2, 0.8, 0.2, 0.9],
                    size: voxel_size,
                    _padding: [0.0; 3],
                });
            }
        }
        
        // Pattern 3: Vertical pillar (1x5x1 column)
        let solid_pillar_base = (42, 60, 64);
        for y in 0..5 {
            instances.push(VoxelInstance {
                position: grid_to_world(solid_pillar_base.0, solid_pillar_base.1 + y, solid_pillar_base.2).to_array(),
                voxel_type: FluidType::Solid as u32,
                color: [0.15, 0.7, 0.15, 0.9],
                size: voxel_size,
                _padding: [0.0; 3],
            });
        }
        
        // ===== WATER (Blue) - Liquid Patterns =====
        // Pattern 1: Single droplet
        let water_droplet = (50, 68, 64);
        instances.push(VoxelInstance {
            position: grid_to_world(water_droplet.0, water_droplet.1, water_droplet.2).to_array(),
            voxel_type: FluidType::Water as u32,
            color: [0.2, 0.5, 1.0, 0.7],
            size: voxel_size,
            _padding: [0.0; 3],
        });
        
        // Pattern 2: Horizontal pool (6x1x4 shallow water)
        let water_pool_base = (52, 60, 62);
        for x in 0..6 {
            for z in 0..4 {
                instances.push(VoxelInstance {
                    position: grid_to_world(water_pool_base.0 + x, water_pool_base.1, water_pool_base.2 + z).to_array(),
                    voxel_type: FluidType::Water as u32,
                    color: [0.2, 0.4, 0.9, 0.7],
                    size: voxel_size,
                    _padding: [0.0; 3],
                });
            }
        }
        
        // Pattern 3: Vertical stream (1x6x1 falling water)
        let water_stream_base = (60, 62, 64);
        for y in 0..6 {
            instances.push(VoxelInstance {
                position: grid_to_world(water_stream_base.0, water_stream_base.1 + y, water_stream_base.2).to_array(),
                voxel_type: FluidType::Water as u32,
                color: [0.3, 0.6, 1.0, 0.6],
                size: voxel_size,
                _padding: [0.0; 3],
            });
        }
        
        // Pattern 4: Small cube (3x3x3 water body)
        let water_cube_base = (63, 62, 62);
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    instances.push(VoxelInstance {
                        position: grid_to_world(water_cube_base.0 + x, water_cube_base.1 + y, water_cube_base.2 + z).to_array(),
                        voxel_type: FluidType::Water as u32,
                        color: [0.2, 0.4, 0.9, 0.7],
                        size: voxel_size,
                        _padding: [0.0; 3],
                    });
                }
            }
        }
        
        // ===== LAVA (Orange/Red) - Viscous Liquid Patterns =====
        // Pattern 1: Single lava blob
        let lava_blob = (72, 64, 64);
        instances.push(VoxelInstance {
            position: grid_to_world(lava_blob.0, lava_blob.1, lava_blob.2).to_array(),
            voxel_type: FluidType::Lava as u32,
            color: [1.0, 0.3, 0.0, 0.95],
            size: voxel_size,
            _padding: [0.0; 3],
        });
        
        // Pattern 2: Lava flow (7x1x2 horizontal flow)
        let lava_flow_base = (74, 60, 63);
        for x in 0..7 {
            for z in 0..2 {
                let color = if x % 2 == 0 {
                    [1.0, 0.4, 0.0, 0.9] // Orange
                } else {
                    [0.9, 0.1, 0.0, 0.9] // Red
                };
                instances.push(VoxelInstance {
                    position: grid_to_world(lava_flow_base.0 + x, lava_flow_base.1, lava_flow_base.2 + z).to_array(),
                    voxel_type: FluidType::Lava as u32,
                    color,
                    size: voxel_size,
                    _padding: [0.0; 3],
                });
            }
        }
        
        // Pattern 3: Lava pillar (1x4x1 rising lava)
        let lava_pillar_base = (83, 62, 64);
        for y in 0..4 {
            instances.push(VoxelInstance {
                position: grid_to_world(lava_pillar_base.0, lava_pillar_base.1 + y, lava_pillar_base.2).to_array(),
                voxel_type: FluidType::Lava as u32,
                color: [0.95, 0.2, 0.0, 0.9],
                size: voxel_size,
                _padding: [0.0; 3],
            });
        }
        
        // Pattern 4: Lava pool (4x2x3 thick lava)
        let lava_pool_base = (85, 62, 62);
        for x in 0..4 {
            for y in 0..2 {
                for z in 0..3 {
                    let color = if (x + y + z) % 2 == 0 {
                        [1.0, 0.4, 0.0, 0.9]
                    } else {
                        [0.9, 0.1, 0.0, 0.9]
                    };
                    instances.push(VoxelInstance {
                        position: grid_to_world(lava_pool_base.0 + x, lava_pool_base.1 + y, lava_pool_base.2 + z).to_array(),
                        voxel_type: FluidType::Lava as u32,
                        color,
                        size: voxel_size,
                        _padding: [0.0; 3],
                    });
                }
            }
        }
        
        // ===== STEAM (Grey/White) - Gas Patterns =====
        // Pattern 1: Single steam particle
        let steam_particle = (95, 68, 64);
        instances.push(VoxelInstance {
            position: grid_to_world(steam_particle.0, steam_particle.1, steam_particle.2).to_array(),
            voxel_type: FluidType::Steam as u32,
            color: [0.9, 0.9, 0.9, 0.4],
            size: voxel_size,
            _padding: [0.0; 3],
        });
        
        // Pattern 2: Steam cloud (5x3x4 dispersed gas)
        let steam_cloud_base = (92, 64, 62);
        for x in 0..5 {
            for y in 0..3 {
                for z in 0..4 {
                    // Sparse pattern - only 50% filled
                    if (x + y + z) % 2 == 0 {
                        let alpha = 0.3 + (y as f32 * 0.1); // More transparent at top
                        instances.push(VoxelInstance {
                            position: grid_to_world(steam_cloud_base.0 + x, steam_cloud_base.1 + y, steam_cloud_base.2 + z).to_array(),
                            voxel_type: FluidType::Steam as u32,
                            color: [0.85, 0.85, 0.85, alpha],
                            size: voxel_size,
                            _padding: [0.0; 3],
                        });
                    }
                }
            }
        }
        
        // Pattern 3: Rising steam column (1x7x1 vertical plume)
        let steam_plume_base = (90, 60, 64);
        for y in 0..7 {
            let alpha = 0.6 - (y as f32 * 0.05); // Fade out as it rises
            instances.push(VoxelInstance {
                position: grid_to_world(steam_plume_base.0, steam_plume_base.1 + y, steam_plume_base.2).to_array(),
                voxel_type: FluidType::Steam as u32,
                color: [0.95, 0.95, 0.95, alpha],
                size: voxel_size,
                _padding: [0.0; 3],
            });
        }
        
        // Pattern 4: Scattered steam particles (random sparse pattern)
        let steam_scatter_base = (98, 62, 62);
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    // Very sparse - only 25% filled
                    if (x * 7 + y * 3 + z * 5) % 4 == 0 {
                        instances.push(VoxelInstance {
                            position: grid_to_world(steam_scatter_base.0 + x, steam_scatter_base.1 + y, steam_scatter_base.2 + z).to_array(),
                            voxel_type: FluidType::Steam as u32,
                            color: [0.8, 0.8, 0.8, 0.35],
                            size: voxel_size,
                            _padding: [0.0; 3],
                        });
                    }
                }
            }
        }
        
        log::info!(
            "Generated {} diverse fluid pattern voxels aligned to 128³ grid (cell_size: {:.4})",
            instances.len(),
            cell_size
        );
        log::info!("Patterns: Solid (wall/pillar), Water (droplet/pool/stream/cube), Lava (blob/flow/pillar/pool), Steam (particle/cloud/plume/scatter)");
        self.voxel_instance_count = instances.len() as u32;
        voxel_renderer.update_instances(queue, &instances);
    }
    
    /// Initialize GPU-based surface nets renderer for density field visualization
    pub fn initialize_gpu_surface_nets(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) {
        if self.gpu_surface_nets.is_some() {
            return;
        }
        
        let gpu_surface_nets = GpuSurfaceNets::new(
            device,
            surface_format,
            wgpu::TextureFormat::Depth32Float,
            self.config.sphere_radius,
            glam::Vec3::ZERO,
            self.renderer.width,
            self.renderer.height,
        );
        
        self.gpu_surface_nets = Some(gpu_surface_nets);
        self.show_gpu_density_mesh = true;
        
        log::info!("GPU surface nets renderer initialized");
    }
    
    /// Generate test density data for GPU surface nets with multiple fluid types
    pub fn generate_test_density(&mut self, queue: &wgpu::Queue) {
        use crate::simulation::fluid_simulation::GRID_RESOLUTION;
        
        if self.gpu_surface_nets.is_none() {
            log::warn!("GPU surface nets not initialized");
            return;
        }
        
        let world_radius = self.config.sphere_radius;
        let world_center = glam::Vec3::ZERO;
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = world_center - glam::Vec3::splat(world_diameter / 2.0);
        let total_voxels = (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
        
        let mut density = vec![0.0f32; total_voxels];
        let mut fluid_types = vec![0u32; total_voxels]; // 0=none, 1=water, 2=lava, 3=steam
        
        // Helper to convert grid coords to world position
        let grid_to_world = |x: u32, y: u32, z: u32| -> glam::Vec3 {
            grid_origin + glam::Vec3::new(
                (x as f32 + 0.5) * cell_size,
                (y as f32 + 0.5) * cell_size,
                (z as f32 + 0.5) * cell_size,
            )
        };
        
        // Metaball contribution function
        let metaball = |pos: glam::Vec3, center: glam::Vec3, radius: f32| -> f32 {
            let dist = (pos - center).length();
            if dist < radius * 2.0 {
                let r = dist / radius;
                (1.0 - r * r).max(0.0)
            } else {
                0.0
            }
        };
        
        // === WATER: Large pool at bottom with droplets ===
        let water_pool_center = glam::Vec3::new(0.0, -world_radius * 0.4, 0.0);
        let water_pool_radius = world_radius * 0.5;
        
        // Water droplets
        let water_droplets = vec![
            (glam::Vec3::new(-world_radius * 0.3, -world_radius * 0.1, world_radius * 0.2), world_radius * 0.12),
            (glam::Vec3::new(-world_radius * 0.25, 0.0, world_radius * 0.15), world_radius * 0.08),
            (glam::Vec3::new(-world_radius * 0.35, world_radius * 0.1, world_radius * 0.1), world_radius * 0.06),
        ];
        
        // === LAVA: Blob cluster on one side ===
        let lava_blobs = vec![
            (glam::Vec3::new(world_radius * 0.5, -world_radius * 0.2, 0.0), world_radius * 0.25),
            (glam::Vec3::new(world_radius * 0.4, -world_radius * 0.1, world_radius * 0.15), world_radius * 0.18),
            (glam::Vec3::new(world_radius * 0.55, 0.0, -world_radius * 0.1), world_radius * 0.15),
            (glam::Vec3::new(world_radius * 0.35, -world_radius * 0.3, -world_radius * 0.1), world_radius * 0.2),
        ];
        
        // === STEAM: Rising wisps at top ===
        let steam_wisps = vec![
            (glam::Vec3::new(-world_radius * 0.1, world_radius * 0.5, -world_radius * 0.2), world_radius * 0.15),
            (glam::Vec3::new(world_radius * 0.05, world_radius * 0.6, -world_radius * 0.1), world_radius * 0.12),
            (glam::Vec3::new(-world_radius * 0.15, world_radius * 0.45, 0.0), world_radius * 0.1),
            (glam::Vec3::new(0.0, world_radius * 0.7, -world_radius * 0.15), world_radius * 0.08),
            (glam::Vec3::new(world_radius * 0.1, world_radius * 0.55, world_radius * 0.1), world_radius * 0.1),
        ];
        
        // Fill density grid with fluid types
        for z in 0..GRID_RESOLUTION {
            for y in 0..GRID_RESOLUTION {
                for x in 0..GRID_RESOLUTION {
                    let idx = (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
                    let pos = grid_to_world(x, y, z);
                    
                    // Track density contribution from each fluid type
                    let mut water_d = 0.0f32;
                    let mut lava_d = 0.0f32;
                    let mut steam_d = 0.0f32;
                    
                    // Water pool (flattened ellipsoid)
                    let water_pos = pos - water_pool_center;
                    let water_dist = (water_pos.x * water_pos.x + water_pos.y * water_pos.y * 4.0 + water_pos.z * water_pos.z).sqrt();
                    if water_dist < water_pool_radius {
                        water_d += (1.0 - water_dist / water_pool_radius).max(0.0);
                    }
                    
                    // Water droplets
                    for (center, radius) in &water_droplets {
                        water_d += metaball(pos, *center, *radius);
                    }
                    
                    // Lava blobs
                    for (center, radius) in &lava_blobs {
                        lava_d += metaball(pos, *center, *radius);
                    }
                    
                    // Steam wisps
                    for (center, radius) in &steam_wisps {
                        steam_d += metaball(pos, *center, *radius);
                    }
                    
                    // Total density is sum of all contributions
                    let total_d = (water_d + lava_d + steam_d).min(1.0);
                    density[idx] = total_d;
                    
                    // Assign fluid type based on dominant contribution
                    if total_d > 0.01 {
                        if water_d >= lava_d && water_d >= steam_d {
                            fluid_types[idx] = 1; // Water
                        } else if lava_d >= water_d && lava_d >= steam_d {
                            fluid_types[idx] = 2; // Lava
                        } else {
                            fluid_types[idx] = 3; // Steam
                        }
                    }
                }
            }
        }
        
        let gpu_surface_nets = self.gpu_surface_nets.as_ref().unwrap();
        gpu_surface_nets.upload_density(queue, &density);
        gpu_surface_nets.upload_fluid_types(queue, &fluid_types);
        
        log::info!("Generated multi-fluid test density: water pool + {} droplets, {} lava blobs, {} steam wisps",
            water_droplets.len(), lava_blobs.len(), steam_wisps.len());
    }
    
    /// Extract mesh from density field using GPU compute shaders
    pub fn extract_gpu_density_mesh(&mut self, _device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref mut gpu_surface_nets) = self.gpu_surface_nets {
            gpu_surface_nets.extract_mesh(encoder);
        }
    }
    
    /// Read back mesh counts from GPU (call after command buffer submission)
    pub fn read_gpu_mesh_counts(&mut self, device: &wgpu::Device) {
        if let Some(ref mut gpu_surface_nets) = self.gpu_surface_nets {
            gpu_surface_nets.read_counts(device);
            log::trace!(
                "GPU mesh: {} vertices, {} triangles",
                gpu_surface_nets.vertex_count,
                gpu_surface_nets.triangle_count()
            );
        }
    }

    /// Initialize the GPU fluid simulator (starts empty)
    pub fn initialize_fluid_simulator(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) {
        // First ensure surface nets is initialized
        if self.gpu_surface_nets.is_none() {
            self.initialize_gpu_surface_nets(device, surface_format);
        }

        // Create GPU fluid simulator
        // Get solid mask buffer from fluid buffers
        let solid_mask_buffer = if let Some(ref fluid_buffers) = self.fluid_buffers {
            fluid_buffers.solid_mask().clone()
        } else {
            // This should not happen if fluid system is properly initialized
            log::error!("Fluid buffers not initialized when creating fluid simulator");
            return;
        };
        
        let simulator = GpuFluidSimulator::new(device, self.config.sphere_radius, glam::Vec3::ZERO, solid_mask_buffer);

        // Update the position update force accum bind group with real water buffers
        self.cached_bind_groups.update_water_buffers(
            device,
            &self.gpu_physics_pipelines,
            &self.adhesion_buffers,
            &self.gpu_triple_buffers,
            simulator.water_grid_params_buffer(),
            simulator.water_bitfield_buffer(),
        );

        // Start with empty fluid - no initial sphere spawn
        // Fluid will only appear when continuous spawning is enabled

        self.fluid_simulator = Some(simulator);
        self.show_gpu_density_mesh = true;

        // Initialize steam particle renderer
        self.initialize_steam_particle_renderer(device, surface_format);

        // Initialize water particle renderer
        self.initialize_water_particle_renderer(device, surface_format);

        // Initialize nutrient particle renderer
        self.initialize_nutrient_particle_renderer(device, queue, surface_format);

        log::info!("GPU fluid simulator initialized with solid mask, water bitfield, and nutrient particles for cell buoyancy");
    }

    /// Step the GPU fluid simulation and update water bitfield for cell buoyancy
    pub fn step_fluid_simulation(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        if let Some(ref simulator) = self.fluid_simulator {
            simulator.step(device, queue, encoder, dt, self.gravity, self.gravity_dir, self.lateral_flow_probabilities, self.condensation_probability, self.vaporization_probability);
            // Update water bitfield for cell physics (compressed 32x for fast lookup)
            simulator.update_water_bitfield(device, encoder);
        }
    }

    /// Clear all fluid
    pub fn clear_fluid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref mut simulator) = self.fluid_simulator {
            simulator.clear(device, queue, encoder);
        }
    }

    /// Reset fluid (clear only, no sphere respawn)
    pub fn reset_fluid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref mut simulator) = self.fluid_simulator {
            simulator.clear(device, queue, encoder);
            // Removed automatic sphere generation - fluid stays empty after reset
        }
    }

    /// Toggle fluid simulation pause state
    pub fn toggle_fluid_pause(&mut self) {
        if let Some(ref mut simulator) = self.fluid_simulator {
            simulator.paused = !simulator.paused;
            log::info!("Fluid simulation {}", if simulator.paused { "paused" } else { "resumed" });
        }
    }

    /// Initialize the steam particle renderer
    pub fn initialize_steam_particle_renderer(&mut self, device: &wgpu::Device, surface_format: wgpu::TextureFormat) {
        if self.steam_particle_renderer.is_some() {
            return;
        }

        let depth_format = wgpu::TextureFormat::Depth32Float;

        // Create camera bind group layout for steam particles
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Steam Particle Camera Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let steam_particle_renderer = SteamParticleRenderer::new(
            device,
            surface_format,
            depth_format,
            &camera_layout,
            self.renderer.width,
            self.renderer.height,
        );

        // Create render bind group
        self.steam_render_bind_group = Some(steam_particle_renderer.create_render_bind_group(device));

        self.steam_particle_renderer = Some(steam_particle_renderer);
        self.show_steam_particles = true;

        log::info!("Steam particle renderer initialized (GPU-based)");
    }

    /// Initialize the water particle renderer
    pub fn initialize_water_particle_renderer(&mut self, device: &wgpu::Device, surface_format: wgpu::TextureFormat) {
        if self.water_particle_renderer.is_some() {
            return;
        }

        let depth_format = wgpu::TextureFormat::Depth32Float;

        // Create camera bind group layout for water particles
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Particle Camera Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mut water_particle_renderer = WaterParticleRenderer::new(
            device,
            surface_format,
            depth_format,
            &camera_layout,
            self.renderer.width,
            self.renderer.height,
        );

        // Set prominence factor
        water_particle_renderer.set_prominence_factor(self.water_particle_prominence);

        // Create render bind group
        let water_particle_renderer = water_particle_renderer; // Make it immutable for the rest
        self.water_render_bind_group = Some(water_particle_renderer.create_render_bind_group(device));

        self.water_particle_renderer = Some(water_particle_renderer);
        self.show_water_particles = true;

        log::info!("Water particle renderer initialized (GPU-based)");
    }

    /// Create or update nutrient particle renderer when fluid simulator is available
    fn ensure_nutrient_particle_renderer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat, depth_format: wgpu::TextureFormat) {
        if self.nutrient_particle_renderer.is_some() {
            return;
        }

        // Create camera layout (same as other renderers)
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Camera Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mut nutrient_particle_renderer = NutrientParticleRenderer::new(
            device,
            queue,
            surface_format,
            depth_format,
            &camera_layout,
            self.renderer.width,
            self.renderer.height,
        );

        // Set spawn probability
        nutrient_particle_renderer.set_spawn_probability(self.nutrient_spawn_probability);

        // Create render bind group
        let nutrient_particle_renderer = nutrient_particle_renderer; // Make it immutable for the rest
        self.nutrient_render_bind_group = Some(nutrient_particle_renderer.create_render_bind_group(device));

        self.nutrient_particle_renderer = Some(nutrient_particle_renderer);
        self.show_nutrient_particles = true;

        log::info!("Nutrient particle renderer initialized (GPU-based)");
    }

    /// Initialize the nutrient particle renderer
    pub fn initialize_nutrient_particle_renderer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) {
        if self.nutrient_particle_renderer.is_some() {
            return;
        }

        let depth_format = wgpu::TextureFormat::Depth32Float;
        self.ensure_nutrient_particle_renderer(device, queue, surface_format, depth_format);
    }

    /// Create or update steam extract bind group when fluid simulator is available
    fn ensure_steam_extract_bind_group(&mut self, device: &wgpu::Device) {
        if self.steam_extract_bind_group.is_some() {
            return;
        }

        if let (Some(ref particle_renderer), Some(ref fluid_sim)) =
            (&self.steam_particle_renderer, &self.fluid_simulator) {
            self.steam_extract_bind_group = Some(
                particle_renderer.create_extract_bind_group(device, fluid_sim.current_state_buffer())
            );
            log::info!("Steam extract bind group created");
        }
    }

    /// Run steam particle extraction compute shader
    pub fn extract_steam_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
    ) {
        // Ensure bind group exists
        self.ensure_steam_extract_bind_group(device);

        if let (Some(ref mut particle_renderer), Some(ref fluid_sim), Some(ref extract_bind_group)) =
            (&mut self.steam_particle_renderer, &self.fluid_simulator, &self.steam_extract_bind_group) {

            let (world_radius, world_center) = fluid_sim.grid_params();
            let grid_resolution = 128u32;
            let world_diameter = world_radius * 2.0;
            let cell_size = world_diameter / grid_resolution as f32;
            let grid_origin = world_center - glam::Vec3::splat(world_diameter / 2.0);

            particle_renderer.extract_steam_particles(
                encoder,
                queue,
                extract_bind_group,
                grid_resolution,
                grid_origin,
                cell_size,
                dt,
            );
        }
    }

    /// Poll for steam particle count after command buffer submission
    pub fn poll_steam_particle_count(&mut self, device: &wgpu::Device) {
        if let Some(ref mut particle_renderer) = self.steam_particle_renderer {
            particle_renderer.poll_particle_count(device);
        }
    }

    /// Create or update nutrient extract bind group when fluid simulator is available
    fn ensure_nutrient_extract_bind_group(&mut self, device: &wgpu::Device) {
        if self.nutrient_extract_bind_group.is_some() {
            return;
        }

        if let (Some(ref particle_renderer), Some(ref fluid_sim)) =
            (&self.nutrient_particle_renderer, &self.fluid_simulator) {
            self.nutrient_extract_bind_group = Some(
                particle_renderer.create_extract_bind_group(device, fluid_sim.current_state_buffer(), fluid_sim.solid_mask_buffer())
            );
            log::info!("Nutrient extract bind group created");
        }
    }

    /// Run nutrient particle extraction compute shader
    pub fn extract_nutrient_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
    ) {
        // Ensure bind group exists
        self.ensure_nutrient_extract_bind_group(device);

        if let (Some(ref mut particle_renderer), Some(ref fluid_sim), Some(ref extract_bind_group)) =
            (&mut self.nutrient_particle_renderer, &self.fluid_simulator, &self.nutrient_extract_bind_group) {

            let (world_radius, world_center) = fluid_sim.grid_params();
            let grid_resolution = 128u32;
            let world_diameter = world_radius * 2.0;
            let cell_size = world_diameter / grid_resolution as f32;
            let grid_origin = world_center - glam::Vec3::splat(world_diameter / 2.0);

            particle_renderer.extract_nutrient_particles(
                encoder,
                queue,
                extract_bind_group,
                grid_resolution,
                grid_origin,
                cell_size,
                world_radius,
                dt,
                self.nutrient_density,
            );
        }
    }

    /// Poll for nutrient particle count after command buffer submission
    pub fn poll_nutrient_particle_count(&mut self, device: &wgpu::Device) {
        if let Some(ref mut particle_renderer) = self.nutrient_particle_renderer {
            particle_renderer.poll_particle_count(device);
        }
    }

    /// Create or update water extract bind group when fluid simulator is available
    fn ensure_water_extract_bind_group(&mut self, device: &wgpu::Device) {
        if self.water_extract_bind_group.is_some() {
            return;
        }

        if let (Some(ref particle_renderer), Some(ref fluid_sim)) =
            (&self.water_particle_renderer, &self.fluid_simulator) {
            self.water_extract_bind_group = Some(
                particle_renderer.create_extract_bind_group(device, fluid_sim.current_state_buffer(), fluid_sim.solid_mask_buffer())
            );
            log::info!("Water extract bind group created");
        }
    }

    /// Run water particle extraction compute shader
    pub fn extract_water_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
    ) {
        // Ensure bind group exists
        self.ensure_water_extract_bind_group(device);

        if let (Some(ref mut particle_renderer), Some(ref fluid_sim), Some(ref extract_bind_group)) =
            (&mut self.water_particle_renderer, &self.fluid_simulator, &self.water_extract_bind_group) {

            let (world_radius, world_center) = fluid_sim.grid_params();
            let grid_resolution = 128u32;
            let world_diameter = world_radius * 2.0;
            let cell_size = world_diameter / grid_resolution as f32;
            let grid_origin = world_center - glam::Vec3::splat(world_diameter / 2.0);

            particle_renderer.extract_water_particles(
                encoder,
                queue,
                extract_bind_group,
                grid_resolution,
                grid_origin,
                cell_size,
                world_radius,
                dt,
            );
        }
    }

    /// Poll for water particle count after command buffer submission
    pub fn poll_water_particle_count(&mut self, device: &wgpu::Device) {
        if let Some(ref mut particle_renderer) = self.water_particle_renderer {
            particle_renderer.poll_particle_count(device);
        }
    }

    /// Set water particle prominence factor (0.0 = barely visible, 1.0 = very prominent)
    pub fn set_water_particle_prominence(&mut self, prominence: f32) {
        self.water_particle_prominence = prominence.clamp(0.0, 1.0);
        if let Some(ref mut renderer) = self.water_particle_renderer {
            renderer.set_prominence_factor(self.water_particle_prominence);
        }
    }
    
    /// Set phase change probabilities
    pub fn set_phase_change_probabilities(&mut self, condensation: f32, vaporization: f32) {
        self.condensation_probability = condensation.clamp(0.0, 1.0);
        self.vaporization_probability = vaporization.clamp(0.0, 1.0);
    }

    /// Apply cave parameters from editor state
    pub fn apply_cave_params_from_editor(&mut self, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        if !editor_state.cave_params_dirty {
            return;
        }
        
        if let Some(ref mut cave_renderer) = self.cave_renderer {
            let mut params = *cave_renderer.params();
            params.density = editor_state.cave_density;
            params.scale = editor_state.cave_scale; // Use user's scale directly
            
            params.octaves = editor_state.cave_octaves;
            params.smoothness = editor_state.cave_smoothness;
            params.seed = editor_state.cave_seed;
            params.grid_resolution = editor_state.cave_resolution;
            
            // Ensure world dimensions match the physics world
            params.world_center = [0.0, 0.0, 0.0];
            params.world_radius = self.config.sphere_radius;
            
            *cave_renderer.params_mut() = params;
            self.cave_params_dirty = true;
        }
    }
    
    /// Update cave parameters and regenerate mesh if needed
    pub fn update_cave_params(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.cave_params_dirty {
            return;
        }
        
        if let Some(ref mut cave_renderer) = self.cave_renderer {
            let params = *cave_renderer.params();
            cave_renderer.update_params(device, queue, params);
            self.cave_params_dirty = false;
            
            // Update solid mask when cave parameters change
            self.update_solid_mask(queue);
        }
    }
    
    /// Update solid mask based on current cave parameters
    pub fn update_solid_mask(&mut self, queue: &wgpu::Queue) {
        if let (Some(ref fluid_buffers), Some(ref solid_mask_generator), Some(ref cave_renderer)) = 
            (&self.fluid_buffers, &self.solid_mask_generator, &self.cave_renderer) {
            
            let cave_params = cave_renderer.params();
            let solid_mask = solid_mask_generator.generate_solid_mask(cave_params);
            
            // Update the solid mask buffer
            fluid_buffers.update_solid_mask(queue, &solid_mask);
            
            // Also update surface nets with the solid mask for greedy water surface generation
            if let Some(ref gpu_surface_nets) = self.gpu_surface_nets {
                gpu_surface_nets.upload_solid_mask(queue, &solid_mask);
            }
            
            log::info!("Updated solid mask with {} solid voxels", 
                solid_mask.iter().sum::<u32>());
        }
    }
    
// ... (rest of the code remains the same)
    /// Check and update cave world radius if world diameter changed
    pub fn check_world_diameter_change(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world_diameter: f32) {
        // Check if world diameter changed significantly (more than 0.1 units)
        if (world_diameter - self.previous_world_diameter).abs() > 0.1 {
            if let Some(ref mut cave_renderer) = self.cave_renderer {
                let new_world_radius = world_diameter * 0.5;
                cave_renderer.update_world_radius(device, queue, new_world_radius);
                self.previous_world_diameter = world_diameter;
            }
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
            _pad0: f32,
            _pad1: f32,
            _pad2: f32,
        }
        
        let world_diameter = self.config.sphere_radius * 2.0;
        let params = PhysicsParams {
            delta_time: 0.0,
            current_time: self.current_time,
            current_frame: 0,
            cell_count: 0, // Not used by insertion shader
            world_size: world_diameter,
            boundary_stiffness: 10.0,
            gravity: 0.0,
            acceleration_damping: 0.95,
            grid_resolution: 64,
            grid_cell_size: world_diameter / 64.0,
            max_cells_per_grid: 16,
            enable_thrust_force: 0,
            cell_capacity: self.gpu_triple_buffers.capacity,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
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
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        _lod_debug_colors: bool,
    ) {
        // Sync adhesion settings to GPU when genomes are added or modified
        self.sync_adhesion_settings(queue);
        
        // Check and update cave world radius if world diameter changed
        self.check_world_diameter_change(device, queue, world_diameter);
        
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

        // Update cave parameters if dirty
        self.update_cave_params(device, queue);

        // Step fluid simulation (GPU compute)
        if self.fluid_simulator.is_some() && !self.paused {
            let dt = 1.0 / 60.0; // Fixed timestep for fluid
            self.step_fluid_simulation(device, queue, &mut encoder, dt);

            // Extract steam particles from fluid state (GPU compute)
            if self.show_steam_particles {
                self.extract_steam_particles(&mut encoder, device, queue, dt);
            }

            // Extract water particles from fluid state (GPU compute)
            if self.show_water_particles {
                self.extract_water_particles(&mut encoder, device, queue, dt);
            }

            // Extract nutrient particles from fluid state (GPU compute)
            if self.show_nutrient_particles {
                self.extract_nutrient_particles(&mut encoder, device, queue, dt);
            }
        }

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
            self.lod_scale_factor,
            self.lod_threshold_low,
            self.lod_threshold_medium,
            self.lod_threshold_high,
            self.lod_debug_colors,
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
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
        );

        // Render flagellocyte tails using GPU instance buffer
        // Render tails for Flagellocytes only
        self.tail_renderer.render_from_gpu_buffer(
            device,
            queue,
            &mut encoder,
            view,
            &self.renderer.depth_view,
            self.instance_builder.get_instance_buffer(),
            self.instance_builder.get_indirect_buffer_for_type(CellType::Flagellocyte),
            self.camera.position(),
            self.camera.rotation,
            self.current_time,
            self.renderer.width,
            self.renderer.height,
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
        
        // Render cave system if initialized
        if let Some(ref mut cave_renderer) = self.cave_renderer {
            cave_renderer.render(
                &mut encoder,
                queue,
                view,
                &self.renderer.depth_view,
                self.camera.position(),
                self.camera.rotation,
            );
        }
        
        // Render fluid voxels if enabled
        if self.show_fluid_voxels {
            if let (Some(ref voxel_renderer), Some(ref _fluid_buffers)) = (&self.voxel_renderer, &self.fluid_buffers) {
                // Create camera bind group for voxel rendering
                // We need to create this each frame since we don't store it
                let camera_uniform = self.create_camera_uniform();
                let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Voxel Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                
                let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Voxel Camera Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
                
                let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Voxel Camera Bind Group"),
                    layout: &camera_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                });
                
                let mut voxel_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Voxel Rendering Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve previous rendering
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.renderer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Use existing depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                // Render voxels with tracked instance count
                voxel_renderer.render(&mut voxel_pass, &camera_bind_group, self.voxel_instance_count);
            }
        }
        
        // Extract and render GPU density mesh if enabled
        if self.show_gpu_density_mesh {
            // First extract density from fluid simulator to surface nets buffers
            if let (Some(ref fluid_sim), Some(ref gpu_surface_nets)) = (&self.fluid_simulator, &self.gpu_surface_nets) {
                fluid_sim.extract_to_surface_nets(
                    device,
                    &mut encoder,
                    gpu_surface_nets.density_buffer(),
                    gpu_surface_nets.fluid_type_buffer(),
                );
            }

            if let Some(ref gpu_surface_nets) = self.gpu_surface_nets {
                // Run compute shaders to extract mesh from density buffer
                gpu_surface_nets.extract_mesh(&mut encoder);

                // Render the extracted mesh
                gpu_surface_nets.render(
                    &mut encoder,
                    queue,
                    view,
                    &self.renderer.depth_view,
                    self.camera.position(),
                    self.camera.rotation,
                );
            }
        }

        // Render steam particles if enabled
        if self.show_steam_particles {
            if let (Some(ref steam_particle_renderer), Some(ref render_bind_group)) =
                (&self.steam_particle_renderer, &self.steam_render_bind_group) {

                // Create camera bind group for steam particles
                let camera_uniform = self.create_camera_uniform();
                let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Steam Particle Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Steam Particle Camera Bind Group"),
                    layout: steam_particle_renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                });

                steam_particle_renderer.render(
                    &mut encoder,
                    view,
                    &self.renderer.depth_view,
                    &camera_bind_group,
                    render_bind_group,
                );
            }
        }

        // Render water particles if enabled
        if self.show_water_particles {
            if let (Some(ref water_particle_renderer), Some(ref render_bind_group)) =
                (&self.water_particle_renderer, &self.water_render_bind_group) {

                // Create camera bind group for water particles
                let camera_uniform = self.create_camera_uniform();
                let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water Particle Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Water Particle Camera Bind Group"),
                    layout: water_particle_renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                });

                water_particle_renderer.render(
                    &mut encoder,
                    view,
                    &self.renderer.depth_view,
                    &camera_bind_group,
                    render_bind_group,
                );
            }
        }

        // Render nutrient particles if enabled
        if self.show_nutrient_particles {
            if let (Some(ref nutrient_particle_renderer), Some(ref render_bind_group)) =
                (&self.nutrient_particle_renderer, &self.nutrient_render_bind_group) {

                // Create camera bind group for nutrient particles
                let camera_uniform = self.create_camera_uniform();
                let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Nutrient Particle Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Nutrient Particle Camera Bind Group"),
                    layout: nutrient_particle_renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }],
                });

                nutrient_particle_renderer.render(
                    &mut encoder,
                    view,
                    &self.renderer.depth_view,
                    &camera_bind_group,
                    render_bind_group,
                );
            }
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

        // Poll for steam particle count (GPU readback)
        if self.show_steam_particles {
            self.poll_steam_particle_count(device);
        }

        // Poll for water particle count (GPU readback)
        if self.show_water_particles {
            self.poll_water_particle_count(device);
        }

        // Poll for nutrient particle count (GPU readback)
        if self.show_nutrient_particles {
            self.poll_nutrient_particle_count(device);
        }

        // Mark that we now have Hi-Z data for next frame
        self.first_frame = false;
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.adhesion_renderer.resize(width, height);
        self.world_sphere_renderer.resize(width, height);
        self.tail_renderer.resize(width, height);
        self.hiz_generator.resize(device, width, height);
        self.instance_builder.reset_hiz(); // Reset Hi-Z config so bind group is recreated with new texture
        self.first_frame = true; // Need to regenerate Hi-Z
        
        // Resize cave renderer if initialized
        if let Some(ref mut cave_renderer) = self.cave_renderer {
            cave_renderer.resize(width, height);
        }
        
        // Resize steam particle renderer if initialized
        if let Some(ref mut steam_particle_renderer) = self.steam_particle_renderer {
            steam_particle_renderer.resize(width, height);
        }
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
