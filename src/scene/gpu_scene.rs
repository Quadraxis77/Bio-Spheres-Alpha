//! GPU simulation scene.
//!
//! This scene runs the full GPU-accelerated simulation using compute shaders.
//! Optimized for large-scale simulations with thousands of cells.

use crate::genome::Genome;
use crate::rendering::{CaveSystemRenderer, CellRenderer, CullingMode, DeathParticleRenderer, DepthOfFieldRenderer, GpuAdhesionLineRenderer, GpuSurfaceNets, InstanceBuilder, NutrientParticleRenderer, OrganismSkinRenderer, SkyboxRenderer, SteamParticleRenderer, SunRenderer, TailRenderer, VolumetricFogRenderer, VoxelRenderer, WaterParticleRenderer, WorldSphereRenderer};
use crate::scene::Scene;
use crate::simulation::{PhysicsConfig};
use crate::simulation::fluid_simulation::{FluidBuffers, GpuFluidSimulator, SolidMaskGenerator};
use crate::simulation::gpu_physics::{execute_gpu_physics_step, execute_gpu_mechanics_step, execute_lifecycle_pipeline, execute_signal_system, CachedBindGroups, GpuPhysicsPipelines, GpuTripleBufferSystem, AdhesionBuffers, GpuCellInspector, GpuCellInsertion, GpuToolOperations, AsyncReadbackManager, GenomeBufferManager, LightFieldSystem, PhagocyteConsumptionSystem, DevorocyteConsumptionSystem, MossSystem, BoulderSystem, GametocyteMergeSystem, GameteMergeEvent};
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
    /// Double-click: find the clicked cell then follow its organism
    Follow,
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
    /// Whether any genome has an oculocyte mode (for signal system gating)
    pub(super) has_oculocytes: bool,
    /// Maximum signal hops across all oculocyte modes in all genomes (for signal propagation dispatch count)
    pub(super) max_signal_hops: u32,
    /// Genome buffer manager for per-genome GPU resources
    pub genome_buffer_manager: GenomeBufferManager,
    /// Cached parent_make_adhesion flags from genome modes for quick lookup during division
    pub(super) parent_make_adhesion_flags: Vec<bool>,
    /// Accumulated time for fixed timestep physics
    time_accumulator: f32,
    /// Current frame counter (for shader time-based logic)
    pub(super) current_frame: i32,
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
    /// Pending cell insertion: (position, genome, initial_reserve_override, initial_nutrients_override).
    /// Both are ×1000 fixed-point; 0 means "use cell_type default".
    pending_cell_insertion: Option<(glam::Vec3, Genome, u32, u32)>,
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
    /// Gravity mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial (toward origin)
    pub gravity_mode: u32,
    /// Number of additional adhesion constraint solver iterations (0 = default single-pass)
    pub constraint_iterations: u32,
    /// Surface pressure: tangential smoothing strength for radial fluid mode (0.0-1.0)
    pub surface_pressure: f32,
    /// Global velocity damping factor (0.0-1.0, higher = less damping, lower = more drag)
    pub acceleration_damping: f32,
    /// Water viscosity: drag applied to cells moving through water (0.0 = off, 1.0 = heavy drag)
    pub water_viscosity: f32,
    /// Metabolism multiplier for solo cells (1.0 = disabled/no penalty, >1.0 = increased drain).
    /// When 1.0, feature is off. When >1.0, cells with zero adhesions burn nutrients faster.
    pub solo_metabolism_multiplier: f32,
    /// Global radiation level controlling mutation probability per division (0.0 = off, 1.0 = always)
    pub radiation_level: f32,
    /// When true, mutations make small color perturbations instead of full re-rolls
    pub subtle_mutations: bool,
    /// Per-fluid-type lateral flow probabilities for fluid simulation (0.0 to 1.0)
    /// Index: 0=Empty (unused), 1=Water, 2=Lava, 3=Steam
    pub lateral_flow_probabilities: [f32; 4],
    /// Phase change probabilities for fluid simulation (0.0 to 1.0)
    pub condensation_probability: f32,  // Steam to Water
    pub vaporization_probability: f32,  // Water to Steam
    /// Nutrient particle density for noise-based spawning (0.0 to 1.0)
    pub nutrient_density: f32,  // Controls threshold for nutrient spawning
    /// Nutrient epoch duration in seconds (how long one nutrient pattern lasts)
    pub nutrient_epoch_duration: f32,
    /// Nutrient epoch spacing in seconds (time between epoch starts; < duration = overlap)
    pub nutrient_epoch_spacing: f32,
    /// Fraction of epoch spent ramping up spawns (0.0–1.0)
    pub nutrient_spawn_end: f32,
    /// Fraction of epoch where despawn ramp begins (0.0–1.0, must be > spawn_end)
    pub nutrient_despawn_start: f32,
    /// Current cell count (tracked on GPU, no CPU canonical state)
    pub current_cell_count: u32,
    /// Total cell slots used (high water mark from GPU readback).
    /// Unlike current_cell_count (live), this includes dead-but-not-yet-recycled slots.
    /// Physics dispatch must cover all slots up to this value so that metabolism
    /// runs on every live cell regardless of its index.
    pub(super) total_cell_slots: u32,
    /// Next cell ID for deterministic cell creation
    pub(super) next_cell_id: u32,
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
    /// Organism skin renderer — wraps cell clusters in a smooth isosurface skin
    pub organism_skin_renderer: Option<OrganismSkinRenderer>,
    /// Whether to show organism skins
    pub show_organism_skins: bool,
    /// Cached density bind groups for organism skin (one per triple-buffer index, kept for API compat)
    organism_skin_density_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Cached count bind group for organism skin (kept for API compat)
    organism_skin_count_bind_group: Option<wgpu::BindGroup>,
    /// Shrink-wrap compute bind groups (one per triple-buffer index — position_and_mass changes)
    organism_skin_compute_bind_groups: Option<[wgpu::BindGroup; 3]>,
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
    /// Death particle system renderer — emits burst particles when cells die
    pub death_particle_renderer: Option<DeathParticleRenderer>,
    /// Whether to show death particles
    pub show_death_particles: bool,
    /// Cached death particle render bind group
    death_render_bind_group: Option<wgpu::BindGroup>,
    /// Cached camera bind group for death particle renderer
    env_camera_bind_group_death: Option<wgpu::BindGroup>,
    /// Phagocyte nutrient consumption system
    phagocyte_consumption: Option<PhagocyteConsumptionSystem>,
    /// Cached phagocyte consumption nutrient bind group
    phagocyte_nutrient_bind_group: Option<wgpu::BindGroup>,
    /// Cached phagocyte consumption physics bind groups (one per triple buffer index)
    phagocyte_physics_bind_groups: Option<[wgpu::BindGroup; 3]>,

    /// Devorocyte consumption system
    devorocyte_consumption: Option<DevorocyteConsumptionSystem>,
    /// Cached devorocyte cell data bind group (shared across frames)
    devorocyte_cell_data_bind_group: Option<wgpu::BindGroup>,
    /// Cached devorocyte spatial bind group (shared across frames)
    devorocyte_spatial_bind_group: Option<wgpu::BindGroup>,
    /// Cached devorocyte physics bind groups (one per triple buffer index)
    devorocyte_physics_bind_groups: Option<[wgpu::BindGroup; 3]>,

    /// Gametocyte merge detection system
    gametocyte_merge_system: Option<GametocyteMergeSystem>,
    /// Cached gametocyte cell data bind group (shared across frames, recreated on organism system change)
    gametocyte_cell_data_bind_group: Option<wgpu::BindGroup>,
    /// Cached gametocyte spatial bind group
    gametocyte_spatial_bind_group: Option<wgpu::BindGroup>,
    /// Cached gametocyte physics bind groups (one per triple buffer index)
    gametocyte_physics_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Pending gamete merge events waiting to be processed (decoded from staging buffer)
    pending_gamete_merges: Vec<GameteMergeEvent>,
    /// Tracks whether a staging readback is in flight for gamete events
    gamete_readback_in_flight: bool,
    /// Moss system for cave wall vegetation (growth, erosion, consumption)
    pub moss_system: Option<MossSystem>,
    /// Whether to show moss on cave walls
    pub show_moss: bool,
    /// Set to true when show_moss transitions false to trigger a one-shot buffer clear
    moss_needs_clear: bool,
    /// Cached moss growth bind group (buffers don't change)
    moss_growth_bind_group: Option<wgpu::BindGroup>,
    /// Cached moss consumption physics bind groups (one per triple buffer index)
    moss_consume_physics_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Cached moss consumption moss bind group (shared across frames)
    moss_consume_moss_bind_group: Option<wgpu::BindGroup>,
    /// Boulder system — falling cave debris with size-gated moss nutrients
    pub boulder_system: Option<BoulderSystem>,
    /// Whether boulders are enabled
    pub show_boulders: bool,
    /// Last physics delta_time (stored so render pass can use it for bubble update)
    last_delta_time: f32,
    /// Target number of boulders to maintain
    pub boulder_target_count: u32,
    /// Boulder initial moss store (nutrients)
    pub boulder_initial_moss: f32,
    /// Boulder radius
    pub boulder_radius: f32,
    /// Boulder size gate half-saturation constant (cells)
    pub boulder_size_gate: f32,
    /// Seconds between boulder spawn attempts
    pub boulder_spawn_interval: f32,
    /// Gravity multiplier when boulder is submerged (0 = float, 1 = full gravity)
    pub boulder_buoyancy: f32,
    /// Minimum boulder radius
    pub boulder_radius_min: f32,
    /// Maximum boulder radius
    pub boulder_radius_max: f32,
    /// Minimum boulder moss store (nutrients)
    pub boulder_moss_min: f32,
    /// Maximum boulder moss store (nutrients)
    pub boulder_moss_max: f32,
    /// Organism label system — GPU-driven connected-component labeling
    pub organism_label_system: Option<crate::simulation::gpu_physics::OrganismLabelSystem>,
    /// Mutation system — GPU-driven genome mutation during cell division
    pub mutation_system: Option<crate::simulation::gpu_physics::MutationSystem>,
    /// Light field system for photocyte nutrients and volumetric fog
    pub light_field_system: Option<LightFieldSystem>,
    /// Cached light field bind group (solid_mask doesn't change per frame)
    cached_light_field_bind_group: Option<wgpu::BindGroup>,
    /// Cached occupancy bind groups (one per triple buffer index, positions change)
    cached_occupancy_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Cached photocyte physics bind groups (one per triple buffer index)
    cached_photocyte_physics_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Cached photocyte system bind group (buffers don't change)
    cached_photocyte_system_bind_group: Option<wgpu::BindGroup>,
    /// Volumetric fog renderer
    pub volumetric_fog_renderer: Option<VolumetricFogRenderer>,
    /// Whether to show volumetric fog
    pub show_volumetric_fog: bool,
    /// Depth of field renderer
    pub dof_renderer: Option<DepthOfFieldRenderer>,
    /// Whether depth of field is enabled
    pub show_dof: bool,
    /// Surface format (needed for DoF resize)
    surface_format: wgpu::TextureFormat,
    /// Procedural sun renderer
    pub sun_renderer: Option<SunRenderer>,
    /// Whether to show the procedural sun
    pub show_sun: bool,
    /// Sun intensity
    pub sun_intensity: f32,
    /// Procedural space skybox renderer
    pub skybox_renderer: Option<SkyboxRenderer>,
    /// Whether to show the procedural skybox
    pub show_skybox: bool,
    /// Index of cell currently being dragged (u32::MAX = none)
    pub dragged_cell_index: u32,
    /// Last value written to cell_count_buffer[2] for drag tracking.
    /// Used to avoid redundant write_buffer calls every frame when not dragging.
    last_written_dragged_index: u32,
    /// Cached shared camera buffer for environment renderers (voxel, particles)
    /// Updated once per frame via queue.write_buffer instead of creating new buffers
    env_camera_buffer: Option<wgpu::Buffer>,
    /// Cached camera bind group for voxel renderer
    env_camera_bind_group_voxel: Option<wgpu::BindGroup>,
    /// Cached camera bind group for steam particle renderer
    env_camera_bind_group_steam: Option<wgpu::BindGroup>,
    /// Cached camera bind group for water particle renderer
    env_camera_bind_group_water: Option<wgpu::BindGroup>,
    /// Cached camera bind group for nutrient particle renderer
    env_camera_bind_group_nutrient: Option<wgpu::BindGroup>,
    /// Cached adhesion data bind groups (one per triple buffer index)
    cached_adhesion_data_bind_groups: Option<[wgpu::BindGroup; 3]>,
    /// Whether the cave shadow bind group has been created and set
    cave_shadow_bind_group_set: bool,
    /// Whether the water surface shadow bind group has been created and set
    water_shadow_bind_group_set: bool,
    /// Whether the cell renderer shadow bind group has been created and set
    cell_shadow_bind_group_set: bool,
    /// Whether the reflection cubemap has been captured
    reflection_cubemap_captured: bool,
    /// Whether genome settings are dirty and need GPU sync
    pub(super) genomes_dirty: bool,
    /// Uniform buffer for signal sense world params (boundary_radius, light_dir, grid params)
    signal_sense_world_params_buffer: wgpu::Buffer,
    /// Dummy nutrient buffer (used when fluid simulator is not yet initialized)
    #[allow(dead_code)]
    signal_sense_dummy_nutrient_buffer: wgpu::Buffer,
    /// Dummy light field buffer (used when light field system is not yet initialized)
    #[allow(dead_code)]
    signal_sense_dummy_light_buffer: wgpu::Buffer,
    /// Dummy solid mask buffer (used when fluid simulator is not yet initialized)
    #[allow(dead_code)]
    signal_sense_dummy_solid_buffer: wgpu::Buffer,
    /// Dummy density field buffer (used when surface nets is not yet initialized)
    #[allow(dead_code)]
    signal_sense_dummy_density_buffer: wgpu::Buffer,

    // ── Organism follow camera ────────────────────────────────────────────────
    /// Root label (min cell index) of the organism the camera is following.
    pub follow_organism_id: Option<u32>,
    /// Raw GPU center-of-mass — snapped to the latest readback value each time one arrives.
    follow_target: glam::Vec3,
    /// Smoothed orbit pivot — lerps toward follow_target every frame.
    follow_center: glam::Vec3,
    /// Persistent staging buffer for position readback (reused every frame).
    follow_pos_staging: Option<wgpu::Buffer>,
    /// Persistent staging buffer for label readback (reused every frame).
    follow_lbl_staging: Option<wgpu::Buffer>,
    /// True when a GPU→staging copy has been submitted and not yet consumed.
    follow_copy_submitted: bool,
    /// True when copy_buffer_to_buffer was encoded this frame and map_async
    /// needs to be called after queue.submit (via tick_follow_camera_post_submit).
    follow_needs_map: bool,
    /// Counts map_async completions (0 = none, 1 = one buffer ready, 2 = both ready).
    /// Written by map_async callbacks; polled each frame.
    follow_map_ready_flag: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// True when the label buffer copy in flight was taken on a reset frame.
    /// On reset frames labels are temporarily set to each cell's own index, so
    /// the CoM scan would find only 1 cell and produce a jump. We skip the
    /// target update for that readback and keep the previous follow_target.
    follow_label_was_reset: bool,
}

impl GpuScene {
    /// Create a new GPU scene with the specified capacity.
    pub fn with_capacity(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        capacity: u32,
    ) -> Self {
        Self::with_capacity_and_radius(device, queue, surface_config, capacity, PhysicsConfig::default().sphere_radius)
    }

    pub fn with_capacity_and_radius(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        capacity: u32,
        world_radius: f32,
    ) -> Self {
        let mut config = PhysicsConfig::default();
        config.sphere_radius = world_radius;

        let renderer = CellRenderer::new(device, queue, surface_config, capacity as usize);
        
        // Create GPU adhesion line renderer
        // Each connection shared by 2 cells: capacity * MAX_ADHESIONS_PER_CELL / 2
        let max_adhesions: u32 = capacity * 20 / 2; // 20 = MAX_ADHESIONS_PER_CELL
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
        
        // Create adhesion buffer system with split sub-buffers (3 × 128 MB) to stay under
        // wgpu's 256 MB/buffer limit. The mutation shader writes adhesion settings for all
        // 8M possible mode slots; splitting into 3 × vec4 sub-buffers matches the pattern
        // used by mode_properties_v0..v4 in triple_buffer.rs.
        let mut adhesion_buffers = AdhesionBuffers::new(device, capacity);
        
        // Initialize adhesion buffers with default values
        adhesion_buffers.initialize(queue);
        
        // Create signal sense world params uniform buffer (48 bytes = 12 floats)
        // Matches SignalSenseWorldParams in signal_sense.wgsl
        let signal_sense_world_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Sense World Params Buffer"),
            size: 48, // 12 x f32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Initialize with the correct boundary_radius immediately so the sphere check
        // in sense_barrier works from the very first frame, before run_physics is called.
        {
            let initial_params: [f32; 12] = [
                world_radius,   // boundary_radius
                -0.5, 0.7, 0.5, // light_dir (default)
                0.0, 1.0,       // grid_resolution (0 = no grid), cell_size
                0.0, 0.0, 0.0,  // grid_origin
                0.0, 0.0, 0.0,  // padding
            ];
            queue.write_buffer(&signal_sense_world_params_buffer, 0, bytemuck::cast_slice(&initial_params));
        }
        // Dummy storage buffers (4 bytes each) - used until real systems are initialized
        let signal_sense_dummy_nutrient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Sense Dummy Nutrient Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let signal_sense_dummy_light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Sense Dummy Light Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let signal_sense_dummy_solid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Sense Dummy Solid Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let signal_sense_dummy_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Sense Dummy Density Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create organism label system (always-on, GPU self-throttled)
        let organism_label_system = Some(
            crate::simulation::gpu_physics::OrganismLabelSystem::new(
                device,
                &gpu_triple_buffers,
                &adhesion_buffers,
            ),
        );

        // Create mutation system for GPU-driven genome mutation
        let mutation_system = Some(
            crate::simulation::gpu_physics::MutationSystem::new(device, queue, capacity),
        );

        // Create cached bind groups (once, not per-frame!)
        // Pass organism label buffer for self-collision filtering
        let cached_bind_groups = gpu_physics_pipelines.create_cached_bind_groups(
            device,
            &gpu_triple_buffers,
            &adhesion_buffers,
            &signal_sense_world_params_buffer,
            &signal_sense_dummy_nutrient_buffer,
            &signal_sense_dummy_light_buffer,
            &signal_sense_dummy_solid_buffer,
            &signal_sense_dummy_density_buffer,
            organism_label_system.as_ref().map(|s| &s.label_buffer),
            organism_label_system.as_ref().map(|s| &s.organism_size_buffer),
            None, // boulder_buffers — created later
        );

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
            has_oculocytes: false,
            max_signal_hops: 0,
            genome_buffer_manager,
            parent_make_adhesion_flags: Vec::new(),
            time_accumulator: 0.0,
            current_frame: 0,
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
            gravity_mode: 1, // default Y axis
            constraint_iterations: 4,
            surface_pressure: 0.5,
            acceleration_damping: 0.98,
            water_viscosity: 0.0,
            solo_metabolism_multiplier: 1.0,
            radiation_level: 0.0,
            subtle_mutations: false,
            lateral_flow_probabilities: [1.0, 0.8, 0.6, 0.9],
            condensation_probability: 0.1,
            vaporization_probability: 0.1,
            nutrient_density: 0.2,
            nutrient_epoch_duration: 10.0,
            nutrient_epoch_spacing: 7.0,
            nutrient_spawn_end: 0.4,
            nutrient_despawn_start: 0.6,
            current_cell_count: 0,
            total_cell_slots: 0,
            next_cell_id: 0,
            tail_renderer,
            cave_renderer,
            cave_params_dirty: false,
            cave_physics_bind_groups: None,
            previous_world_diameter: 400.0, // Default world diameter
            boulder_system: None,
            show_boulders: true,
            last_delta_time: 0.016,
            boulder_target_count: 32,
            boulder_initial_moss: 10_000.0,
            boulder_radius: 4.0,
            boulder_size_gate: 20.0,
            boulder_spawn_interval: 5.0,
            boulder_buoyancy: 0.08,
            boulder_radius_min: 2.0,
            boulder_radius_max: 8.0,
            boulder_moss_min: 2_000.0,
            boulder_moss_max: 20_000.0,
            fluid_buffers: None,
            voxel_renderer: None,
            show_fluid_voxels: false,
            voxel_instance_count: 0,
            gpu_surface_nets: None,
            show_gpu_density_mesh: false,
            organism_skin_renderer: None,
            show_organism_skins: false,
            organism_skin_density_bind_groups: None,
            organism_skin_count_bind_group: None,
            organism_skin_compute_bind_groups: None,
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
            death_particle_renderer: None,
            show_death_particles: true,
            death_render_bind_group: None,
            env_camera_bind_group_death: None,
            phagocyte_consumption: None,
            phagocyte_nutrient_bind_group: None,
            phagocyte_physics_bind_groups: None,
            devorocyte_consumption: None,
            devorocyte_cell_data_bind_group: None,
            devorocyte_spatial_bind_group: None,
            devorocyte_physics_bind_groups: None,
            gametocyte_merge_system: None,
            gametocyte_cell_data_bind_group: None,
            gametocyte_spatial_bind_group: None,
            gametocyte_physics_bind_groups: None,
            pending_gamete_merges: Vec::new(),
            gamete_readback_in_flight: false,
            moss_system: None,
            show_moss: true,
            moss_needs_clear: false,
            moss_growth_bind_group: None,
            moss_consume_physics_bind_groups: None,
            moss_consume_moss_bind_group: None,
            organism_label_system,
            mutation_system,
            light_field_system: None,
            cached_light_field_bind_group: None,
            cached_occupancy_bind_groups: None,
            cached_photocyte_physics_bind_groups: None,
            cached_photocyte_system_bind_group: None,
            volumetric_fog_renderer: None,
            show_volumetric_fog: false,
            dof_renderer: None,
            show_dof: false,
            surface_format: surface_config.format,
            sun_renderer: None,
            show_sun: true,
            sun_intensity: 10.0,
            skybox_renderer: None,
            show_skybox: true,
            dragged_cell_index: u32::MAX,
            // Initialize to a value that differs from both dragged_cell_index (u32::MAX)
            // AND the GPU buffer's zero-initialized value (0), so the first frame always
            // writes u32::MAX to cell_count_buffer[2]. Without this, the GPU buffer starts
            // at 0 and the position_update shader freezes cell index 0 (the first placed cell).
            last_written_dragged_index: 0,
            env_camera_buffer: None,
            env_camera_bind_group_voxel: None,
            env_camera_bind_group_steam: None,
            env_camera_bind_group_water: None,
            env_camera_bind_group_nutrient: None,
            cached_adhesion_data_bind_groups: None,
            cave_shadow_bind_group_set: false,
            water_shadow_bind_group_set: false,
            cell_shadow_bind_group_set: false,
            reflection_cubemap_captured: false,
            genomes_dirty: false,
            signal_sense_world_params_buffer,
            signal_sense_dummy_nutrient_buffer,
            signal_sense_dummy_light_buffer,
            signal_sense_dummy_solid_buffer,
            signal_sense_dummy_density_buffer,
            follow_organism_id: None,
            follow_target: glam::Vec3::ZERO,
            follow_center: glam::Vec3::ZERO,
            follow_pos_staging: None,
            follow_lbl_staging: None,
            follow_copy_submitted: false,
            follow_needs_map: false,
            follow_map_ready_flag: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            follow_label_was_reset: false,
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
        self.total_cell_slots = 0;
        self.next_cell_id = 0;
        self.current_time = 0.0;
        self.time_accumulator = 0.0;
        self.current_frame = 0;
        self.paused = false;
        self.first_frame = true;
        self.dragged_cell_index = u32::MAX;
        // Force a write of u32::MAX to cell_count_buffer[2] on the next frame.
        // The GPU buffer is reset to 0 below, so last_written must differ from
        // dragged_cell_index (u32::MAX) to trigger the change-detection write.
        self.last_written_dragged_index = 0;
        
        // Reset adhesion buffers
        self.adhesion_buffers.reset(queue);
        // Clear all genomes since no cells reference them
        self.genomes.clear();
        self.parent_make_adhesion_flags.clear();
        self.has_oculocytes = false;
        self.max_signal_hops = 0;
        self.instance_builder.mark_all_dirty();
        // Reset GPU cell count buffer to 0 immediately
        let cell_counts: [u32; 2] = [0, 0];
        queue.write_buffer(&self.gpu_triple_buffers.cell_count_buffer, 0, bytemuck::cast_slice(&cell_counts));
        // Reset GPU next_cell_id buffer to 0
        let next_id: [u32; 1] = [0];
        queue.write_buffer(&self.gpu_triple_buffers.next_cell_id, 0, bytemuck::cast_slice(&next_id));
        // Reset deterministic cell addition system
        self.gpu_triple_buffers.reset_slot_allocator();
        // Reset ring buffers to empty state
        self.gpu_triple_buffers.reset_ring_buffers(queue);

        // Clear env anchor buffer so no stale anchors persist across resets
        let capacity = self.gpu_triple_buffers.capacity as usize;
        let zero_anchors: Vec<f32> = vec![0.0; capacity * 4];
        queue.write_buffer(&self.gpu_triple_buffers.env_anchor_buffer, 0, bytemuck::cast_slice(&zero_anchors));

        // Mark GPU buffers as needing sync (will be no-op since cell_count is 0)
        self.gpu_triple_buffers.mark_needs_sync();

        // Reset mutation system so ring state is re-seeded on next genome sync
        if let Some(mutation_system) = &mut self.mutation_system {
            mutation_system.reset(queue);
        }

        // Note: Fluid reset is handled separately via reset_fluid() which requires encoder

        // Clear all boulders — they are environmental objects tied to the session,
        // not to the cell simulation, but should still be removed on any reset.
        if let Some(ref mut bs) = self.boulder_system {
            bs.clear(queue);
        }
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

    /// Read back a genome from GPU buffers and reconstruct it as a CPU-side Genome.
    ///
    /// For user-created genomes (genome_id < self.genomes.len()), returns a clone from CPU storage.
    /// For GPU-mutated genomes, performs synchronous GPU readback of all mode buffers
    /// and reconstructs the Genome from raw buffer data.
    ///
    /// This is a blocking operation (waits for GPU) — only call from UI button clicks.
    pub fn read_back_genome(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        genome_id: u32,
    ) -> Option<crate::genome::Genome> {
        // Fast path: CPU-known genome
        if (genome_id as usize) < self.genomes.len() {
            return Some(self.genomes[genome_id as usize].clone());
        }

        // GPU path: read genome metadata to find mode range
        let mutation_system = self.mutation_system.as_ref()?;
        let meta = self.read_back_buffer_range::<crate::simulation::gpu_physics::GenomeMeta>(
            device, queue,
            mutation_system.genome_meta_buffer(),
            genome_id as u64,
            1,
        );
        let meta = match meta {
            Some(m) => m,
            None => {
                log::warn!("read_back_genome: failed to read genome_meta for genome_id={}", genome_id);
                return None;
            }
        };
        let meta = meta[0];
        let mode_count = meta.mode_count as usize;
        let base_offset = meta.base_mode_offset as usize;
        let initial_mode_local = meta.initial_mode_local as i32;

        if mode_count == 0 || mode_count > 40 {
            log::warn!("read_back_genome: invalid metadata for genome_id={}: mode_count={}, base_offset={}", genome_id, mode_count, base_offset);
            return None;
        }

        log::debug!("read_back_genome: genome_id={} mode_count={} base_offset={}", genome_id, mode_count, base_offset);

        // Read back all mode buffers for this genome's mode range (split into 5 sub-buffers each)
        let mp_v0 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.mode_properties_v0, base_offset as u64, mode_count)?;
        let mp_v1 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.mode_properties_v1, base_offset as u64, mode_count)?;
        let mp_v2 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.mode_properties_v2, base_offset as u64, mode_count)?;
        let mp_v3 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.mode_properties_v3, base_offset as u64, mode_count)?;
        let mp_v4 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.mode_properties_v4, base_offset as u64, mode_count)?;
        let mode_props: Vec<[f32; 20]> = (0..mode_count).map(|i| {
            let mut p = [0f32; 20];
            p[0..4].copy_from_slice(&mp_v0[i]); p[4..8].copy_from_slice(&mp_v1[i]);
            p[8..12].copy_from_slice(&mp_v2[i]); p[12..16].copy_from_slice(&mp_v3[i]);
            p[16..20].copy_from_slice(&mp_v4[i]); p
        }).collect();

        let gmd_v0 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.genome_mode_data_v0, base_offset as u64, mode_count)?;
        let gmd_v1 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.genome_mode_data_v1, base_offset as u64, mode_count)?;
        let gmd_v2 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.genome_mode_data_v2, base_offset as u64, mode_count)?;
        let gmd_v3 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.genome_mode_data_v3, base_offset as u64, mode_count)?;
        let gmd_v4 = self.read_back_buffer_range::<[f32; 4]>(device, queue, &self.gpu_triple_buffers.genome_mode_data_v4, base_offset as u64, mode_count)?;
        let mode_data: Vec<[f32; 20]> = (0..mode_count).map(|i| {
            let mut d = [0f32; 20];
            d[0..4].copy_from_slice(&gmd_v0[i]); d[4..8].copy_from_slice(&gmd_v1[i]);
            d[8..12].copy_from_slice(&gmd_v2[i]); d[12..16].copy_from_slice(&gmd_v3[i]);
            d[16..20].copy_from_slice(&gmd_v4[i]); d
        }).collect();

        let child_indices = self.read_back_buffer_range::<[i32; 2]>(
            device, queue,
            &self.gpu_triple_buffers.child_mode_indices,
            base_offset as u64,
            mode_count,
        )?;

        let cell_types = self.read_back_buffer_range::<u32>(
            device, queue,
            &self.gpu_triple_buffers.mode_cell_types,
            base_offset as u64,
            mode_count,
        )?;

        let parent_make_flags = self.read_back_buffer_range::<u32>(
            device, queue,
            &self.gpu_triple_buffers.parent_make_adhesion_flags,
            base_offset as u64,
            mode_count,
        )?;

        let child_a_keep_flags = self.read_back_buffer_range::<u32>(
            device, queue,
            &self.gpu_triple_buffers.child_a_keep_adhesion_flags,
            base_offset as u64,
            mode_count,
        )?;

        let child_b_keep_flags = self.read_back_buffer_range::<u32>(
            device, queue,
            &self.gpu_triple_buffers.child_b_keep_adhesion_flags,
            base_offset as u64,
            mode_count,
        )?;

        let glueocyte_flags = self.read_back_buffer_range::<u32>(
            device, queue,
            &self.gpu_triple_buffers.glueocyte_env_adhesion_flags,
            base_offset as u64,
            mode_count,
        )?;

        let oculocyte_params = self.read_back_buffer_range::<[u32; 4]>(
            device, queue,
            &self.gpu_triple_buffers.oculocyte_params,
            base_offset as u64,
            mode_count,
        )?;

        // Regulation params: [channel(u32), value_bits(u32), hops(u32), padding(u32)]
        // channel == 0xFFFFFFFF means disabled (-1); otherwise 8-15.
        let regulation_params = self.read_back_buffer_range::<[u32; 4]>(
            device, queue,
            &self.gpu_triple_buffers.regulation_params,
            base_offset as u64,
            mode_count,
        )?;

        // mode_visuals: read colors (1 vec4 per mode) and emissive (1 vec4 per mode) separately
        let colors_raw = self.read_back_buffer_range::<[f32; 4]>(
            device, queue,
            self.instance_builder.mode_colors_buffer(),
            base_offset as u64,
            mode_count,
        )?;
        let emissive_raw = self.read_back_buffer_range::<[f32; 4]>(
            device, queue,
            self.instance_builder.mode_emissive_buffer(),
            base_offset as u64,
            mode_count,
        )?;

        // adhesion_settings: read from split sub-buffers (v0/v1/v2), each 16 bytes per mode.
        // All 8M mode slots are covered so no bounds check needed.
        let adhesion_settings_raw: Vec<crate::simulation::gpu_physics::adhesion::GpuAdhesionSettings> = {
            let v0 = self.read_back_buffer_range::<[f32; 4]>(
                device, queue, &self.adhesion_buffers.adhesion_settings_v0, base_offset as u64, mode_count,
            );
            let v1 = self.read_back_buffer_range::<[f32; 4]>(
                device, queue, &self.adhesion_buffers.adhesion_settings_v1, base_offset as u64, mode_count,
            );
            let v2 = self.read_back_buffer_range::<[f32; 4]>(
                device, queue, &self.adhesion_buffers.adhesion_settings_v2, base_offset as u64, mode_count,
            );
            match (v0, v1, v2) {
                (Some(v0), Some(v1), Some(v2)) => {
                    v0.iter().zip(v1.iter()).zip(v2.iter()).map(|((a, b), c)| {
                        crate::simulation::gpu_physics::adhesion::GpuAdhesionSettings {
                            can_break: a[0] as i32,
                            break_force: a[1],
                            rest_length: a[2],
                            linear_spring_stiffness: a[3],
                            linear_spring_damping: b[0],
                            orientation_spring_stiffness: b[1],
                            orientation_spring_damping: b[2],
                            max_angular_deviation: b[3],
                            twist_constraint_stiffness: c[0],
                            twist_constraint_damping: c[1],
                            enable_twist_constraint: c[2] as i32,
                            _padding: 0,
                        }
                    }).collect()
                }
                _ => vec![crate::simulation::gpu_physics::adhesion::GpuAdhesionSettings::default(); mode_count],
            }
        };

        // Reconstruct Genome from raw data
        let mut modes = Vec::with_capacity(mode_count);
        for i in 0..mode_count {
            let props = &mode_props[i];
            let data = &mode_data[i];
            let ci = &child_indices[i];

            // Convert absolute child mode indices back to genome-local
            let child_a_local = (ci[0] - base_offset as i32).max(0);
            let child_b_local = (ci[1] - base_offset as i32).max(0);

            // Reconstruct orientations from genome_mode_data
            let qa = glam::Quat::from_xyzw(data[0], data[1], data[2], data[3]);
            let qb = glam::Quat::from_xyzw(data[4], data[5], data[6], data[7]);
            let qa_split = glam::Quat::from_xyzw(data[8], data[9], data[10], data[11]);
            let qb_split = glam::Quat::from_xyzw(data[12], data[13], data[14], data[15]);
            let split_quat = glam::Quat::from_xyzw(data[16], data[17], data[18], data[19]);

            // Reconstruct split direction from quaternion (reverse of Euler YXZ encoding)
            let (yaw, pitch, _roll) = split_quat.to_euler(glam::EulerRot::YXZ);

            // Reconstruct color from split mode_colors and mode_emissive buffers
            let color_vec4 = &colors_raw[i];
            let emissive_vec4 = &emissive_raw[i];

            // Reconstruct max_splits: negative GPU value means infinite (-1)
            let gpu_max_splits = props[8];
            let max_splits = if gpu_max_splits < 0.0 { -1 } else { gpu_max_splits as i32 };

            let mode = crate::genome::ModeSettings {
                name: format!("Mode {}", i + 1),
                default_name: format!("Mode {}", i + 1),
                color: glam::Vec3::new(color_vec4[0], color_vec4[1], color_vec4[2]),
                opacity: 1.0,
                emissive: emissive_vec4[0],
                cell_type: cell_types[i] as i32,
                parent_make_adhesion: parent_make_flags[i] != 0,
                split_mass: props[4],
                split_interval: props[3],
                nutrient_gain_rate: props[0],
                max_cell_size: props[1],
                split_ratio: props[9],
                nutrient_priority: props[5],
                prioritize_when_low: props[7] > 0.5,
                parent_split_direction: glam::Vec2::new(pitch.to_degrees(), yaw.to_degrees()),
                max_adhesions: props[16] as i32,
                min_adhesions: props[15] as i32,
                enable_parent_angle_snapping: false,
                max_splits,
                mode_a_after_splits: if props[17] < 0.0 { -1 } else { (props[17] as i32 - base_offset as i32).max(0) },
                mode_b_after_splits: if props[18] < 0.0 { -1 } else { (props[18] as i32 - base_offset as i32).max(0) },
                child_a_after_split_orientation: qa_split,
                child_b_after_split_orientation: qb_split,
                child_a_after_split_keep_adhesion: false,
                child_b_after_split_keep_adhesion: false,
                glueocyte_cell_adhesion: false,
                glueocyte_self_adhesion: false,
                glueocyte_env_adhesion: glueocyte_flags[i] != 0,
                glueocyte_boulder_adhesion: true,
                glueocyte_cell_adhesion_signal_channel: -1,
                glueocyte_cell_adhesion_signal_threshold: 1.0,
                glueocyte_signal_gate_invert: false,
                swim_force: props[6],
                flagellocyte_use_signal: props[14] > 0.5,
                flagellocyte_signal_channel: props[10] as i32,
                flagellocyte_speed_a: props[11],
                flagellocyte_speed_b: props[12],
                flagellocyte_threshold_c: props[13],
                buoyancy_force: 0.0,
                oculocyte_sense_type: oculocyte_params[i][0], // u32 bitmask
                oculocyte_signal_channel: 0,
                oculocyte_signal_value: 0.0,
                oculocyte_signal_hops: oculocyte_params[i][2] as i32,
                oculocyte_ray_length: f32::from_bits(oculocyte_params[i][1]),
                membrane_stiffness: props[2],
                regulation_emit_channel: {
                    let ch = regulation_params[i][0];
                    if ch == 0xFFFFFFFFu32 { -1i32 } else { ch.clamp(8, 15) as i32 }
                },
                regulation_emit_value: f32::from_bits(regulation_params[i][1]),
                regulation_emit_hops: regulation_params[i][2].clamp(1, 20) as i32,
                division_signal_channel: -1,
                division_signal_threshold: 1.0,
                division_signal_invert: false,
                apoptosis_signal_channel: -1,
                apoptosis_signal_threshold: 1.0,
                apoptosis_signal_invert: false,
                signal_child_a_channel: -1,
                signal_child_a_threshold: 1.0,
                signal_child_a_mode_above: -1,
                signal_child_a_mode_below: -1,
                signal_child_b_channel: -1,
                signal_child_b_threshold: 1.0,
                signal_child_b_mode_above: -1,
                signal_child_b_mode_below: -1,
                mode_switch_signal_channel: -1,
                mode_switch_signal_threshold: 1.0,
                mode_switch_target: -1,
                mode_switch_invert: false,
                cilia_speed: 0.5,
                cilia_push_bonded: false,
                cilia_use_signal: false,
                cilia_signal_channel: 0,
                cilia_speed_below: 0.5,
                cilia_speed_above: 0.0,
                cilia_threshold: 1.0,
                cilia_attract_force: 0.0,
                myocyte_contraction: 0.5,
                myocyte_use_signal: false,
                myocyte_signal_channel: 0,
                myocyte_contraction_above: 0.5,
                myocyte_contraction_below: 0.0,
                myocyte_threshold: 1.0,
                myocyte_pulse_rate: 1.0,
                myocyte_pulse_phase: 0,
                embryocyte_use_timer: false,
                embryocyte_release_timer: 10.0,
                embryocyte_use_threshold: false,
                embryocyte_threshold_value: 32768,
                embryocyte_use_signal: false,
                embryocyte_signal_channel: 0,
                embryocyte_signal_value: 1.0,
                devorocyte_consume_range: 0.5,
                devorocyte_consume_rate: 30.0,
                vascular_outlet: false,
                gametocyte_merge_range: 0.5,
                memorocyte_decay: 0.95,
                memorocyte_gain: 1.0,
                memorocyte_input_channel: 0,
                memorocyte_output_channel: 9,
                memorocyte_output_hops: 5,
                cognocyte_operation: 0,
                cognocyte_input_channel_a: 0,
                cognocyte_input_channel_b: 1,
                cognocyte_output_channel: 8,
                cognocyte_output_hops: 5,
                child_a: crate::genome::ChildSettings {
                    mode_number: child_a_local,
                    orientation: qa,
                    keep_adhesion: child_a_keep_flags[i] != 0,
                    enable_angle_snapping: false,
                    x_axis_lat: 0.0, x_axis_lon: 0.0,
                    y_axis_lat: 0.0, y_axis_lon: 0.0,
                    z_axis_lat: 0.0, z_axis_lon: 0.0,
                },
                child_b: crate::genome::ChildSettings {
                    mode_number: child_b_local,
                    orientation: qb,
                    keep_adhesion: child_b_keep_flags[i] != 0,
                    enable_angle_snapping: false,
                    x_axis_lat: 0.0, x_axis_lon: 0.0,
                    y_axis_lat: 0.0, y_axis_lon: 0.0,
                    z_axis_lat: 0.0, z_axis_lon: 0.0,
                },
                adhesion_settings: {
                    let gpu_as = &adhesion_settings_raw[i];
                    crate::genome::AdhesionSettings {
                        can_break: gpu_as.can_break != 0,
                        break_force: gpu_as.break_force,
                        rest_length: gpu_as.rest_length,
                        linear_spring_stiffness: gpu_as.linear_spring_stiffness,
                        linear_spring_damping: gpu_as.linear_spring_damping,
                        orientation_spring_stiffness: gpu_as.orientation_spring_stiffness,
                        orientation_spring_damping: gpu_as.orientation_spring_damping,
                        max_angular_deviation: gpu_as.max_angular_deviation,
                        twist_constraint_stiffness: gpu_as.twist_constraint_stiffness,
                        twist_constraint_damping: gpu_as.twist_constraint_damping,
                        enable_twist_constraint: gpu_as.enable_twist_constraint != 0,
                    }
                },
            };
            modes.push(mode);
        }

        Some(crate::genome::Genome {
            name: format!("Mutated Genome #{}", genome_id),
            initial_mode: initial_mode_local,
            initial_orientation: glam::Quat::IDENTITY,
            modes,
        })
    }

    /// Synchronous GPU buffer readback helper.
    /// Reads `count` elements of type T starting at `start_index` from `source_buffer`.
    /// Blocks until the GPU transfer completes.
    fn read_back_buffer_range<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_buffer: &wgpu::Buffer,
        start_index: u64,
        count: usize,
    ) -> Option<Vec<T>> {
        let elem_size = std::mem::size_of::<T>() as u64;
        let offset = start_index * elem_size;
        let size = count as u64 * elem_size;

        // Bounds check before attempting GPU copy
        if size == 0 {
            return Some(Vec::new());
        }
        if offset + size > source_buffer.size() {
            log::warn!(
                "read_back_buffer_range: out of bounds — offset={} size={} buffer_size={} (T={} start_index={} count={})",
                offset, size, source_buffer.size(),
                std::any::type_name::<T>(), start_index, count
            );
            return None;
        }

        // Create staging buffer for readback
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Readback Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from source to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Genome Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(source_buffer, offset, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        match rx.recv() {
            Ok(Ok(())) => {
                let mapped = slice.get_mapped_range();
                let data: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
                drop(mapped);
                staging.unmap();
                Some(data)
            }
            _ => {
                log::error!("GPU readback failed");
                None
            }
        }
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
    
    /// Set surface pressure for radial fluid mode (tangential smoothing strength).
    pub fn set_surface_pressure(&mut self, pressure: f32) {
        self.surface_pressure = pressure.clamp(0.0, 1.0);
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
        // Note: No CPU-side early-out on current_cell_count here.
        // The GPU shaders check cell_count_buffer[0] (high water mark) internally.
        // Using the async-readback live count would cause premature physics freeze
        // when the readback lags behind the actual GPU state.

        // Execute pure GPU physics pipeline (7 compute shader stages + cave collision if enabled)
        // Repopulate nutrient voxels each physics step so phagocyte intake scales with
        // sim speed. At Nx speed the physics loop runs N steps per frame; phagocytes
        // consume voxel nutrients each step via atomic CAS (1→2), so populate must
        // also run each step to reset consumed voxels back to available.
        if let Some(ref simulator) = self.fluid_simulator {
            simulator.populate_nutrients(
                device, queue, encoder,
                self.nutrient_density, delta_time,
                self.nutrient_epoch_duration,
                self.nutrient_epoch_spacing,
                self.nutrient_spawn_end,
                self.nutrient_despawn_start,
            );
        }
        // Run phagocyte nutrient consumption BEFORE physics so nutrients are available for transport
        self.run_phagocyte_consumption(device, encoder, queue);
        // Run devorocyte consumption — steals nutrients from and kills foreign cells
        // Must run AFTER the spatial grid is built (which happens inside execute_gpu_physics_step)
        // but BEFORE nutrient_transport so stolen nutrients are visible this frame.
        // We run it here (before the main physics step) using the grid from the previous frame,
        // which is a one-frame lag but avoids a second grid build pass.
        self.run_devorocyte_consumption(device, encoder);
        // Run gametocyte merge detection — detects contact between gametes from different organisms
        self.run_gametocyte_merge(device, encoder, queue);
        // Run moss consumption (phagocytes eating moss) alongside nutrient consumption
        self.run_moss_consumption(device, encoder, queue);
        // Run photocyte light consumption each physics step (reads pre-computed light field)
        self.run_photocyte_light_consumption(device, encoder, queue);

        // Update boulder system: spawn new boulders, poll dead readback, update instance buffer.
        // Must run BEFORE execute_gpu_physics_step so active_count() is current and the
        // GPU boulder_count buffer is populated before the boulder physics dispatch.
        if self.show_boulders {
            if let Some(ref mut bs) = self.boulder_system {
                bs.update(device, queue, delta_time);
            }
        }
        
        // Update signal sense world params (boundary_radius, light_dir, fluid grid params)
        {
            let boundary_radius = world_diameter * 0.5;
            let light_dir = self.light_field_system.as_ref()
                .map(|lfs| lfs.light_dir())
                .unwrap_or([-0.5, 0.7, 0.5]);
            let (grid_resolution, cell_size, grid_origin) = self.fluid_simulator.as_ref()
                .map(|fs| {
                    let (wr, wc) = fs.grid_params();
                    let wd = wr * 2.0;
                    let cs = wd / 128.0;
                    let go = wc - glam::Vec3::splat(wd / 2.0);
                    (128u32, cs, [go.x, go.y, go.z])
                })
                .unwrap_or((0u32, 1.0, [0.0, 0.0, 0.0]));
            let params: [f32; 12] = [
                boundary_radius,
                light_dir[0], light_dir[1], light_dir[2],
                f32::from_bits(grid_resolution),
                cell_size,
                grid_origin[0], grid_origin[1], grid_origin[2],
                0.0, 0.0, 0.0, // padding
            ];
            queue.write_buffer(&self.signal_sense_world_params_buffer, 0, bytemuck::cast_slice(&params));
        }

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
            self.current_frame,
            world_diameter,
            self.gravity,
            self.gravity_mode,
            self.acceleration_damping,
            self.cave_renderer.as_ref(),
            self.cave_physics_bind_groups.as_ref(),
            &self.adhesion_buffers,
            self.total_cell_slots,
            self.constraint_iterations,
            self.solo_metabolism_multiplier,
            self.boulder_system.as_ref().map(|bs| bs.active_count()).unwrap_or(0),
            self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_force_accum),
        );
        
        // Increment frame counter for time-based shader logic
        self.current_frame = self.current_frame.wrapping_add(1);
        self.last_delta_time = delta_time;
        
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
            &self.adhesion_buffers,
            self.current_time,
            // Use max(1) so lifecycle runs on the first cell even before the async
            // readback has reported total_cell_slots > 0 (lags 1-3 frames).
            self.total_cell_slots.max(1),
        );

        // Mutation pass: collect candidates from division results, then apply mutations.
        // Runs after division execute so division_flags and division_slot_assignments are set.
        if let Some(mutation_system) = &mut self.mutation_system {
            mutation_system.dispatch(
                device,
                encoder,
                queue,
                self.current_frame as u32,
            );
        }
        
        // NOTE: Spatial grid rebuild after lifecycle is unnecessary because the grid
        // is rebuilt at the START of each physics step (stages 1-3). Nothing between
        // lifecycle and the next physics step reads the spatial grid. Removing this
        // saves 3 compute dispatches (clear 128³ grid + assign + insert) per step.

        // Organism label pass: GPU union-find for connected-component labeling.
        // Dispatches are sized to the high-water mark of used cell slots, not the
        // full buffer capacity, so cost scales with actual cell count.
        if let Some(label_system) = &mut self.organism_label_system {
            label_system.encode_frame(encoder, self.total_cell_slots);
        }

        // NOTE: copy_buffers_to_instance_builder is called once per frame in render(),
        // after all physics steps complete. No need to copy after each step.
    }
    
    /// Run mechanics-only physics step (no nutrient transport, no mass accumulation, no lifecycle).
    /// Used when paused and dragging so connected cells follow via spring forces.
    fn run_mechanics_only(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, delta_time: f32, world_diameter: f32) {
        execute_gpu_mechanics_step(
            device,
            encoder,
            queue,
            &self.gpu_physics_pipelines,
            &mut self.gpu_triple_buffers,
            &self.cached_bind_groups,
            delta_time,
            self.current_time,
            self.current_frame,
            world_diameter,
            self.gravity,
            self.gravity_mode,
            self.acceleration_damping,
            self.cave_renderer.as_ref(),
            self.cave_physics_bind_groups.as_ref(),
            &self.adhesion_buffers,
            self.total_cell_slots,
            self.constraint_iterations,
        );
        
        // NOTE: copy_buffers_to_instance_builder is called once per frame in render(),
        // after mechanics step completes. No need to copy here.
    }
    
    /// Run phagocyte nutrient consumption compute shader
    fn run_phagocyte_consumption(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        // Note: No CPU-side early-out on current_cell_count here.
        // The GPU shader reads cell_count_buffer[0] (high water mark) for bounds checking.
        // Using the async-readback live count would cause premature consumption freeze
        // when the readback lags behind the actual GPU state.
        
        let (consumption_system, fluid_sim) = match (&self.phagocyte_consumption, &self.fluid_simulator) {
            (Some(c), Some(f)) => (c, f),
            _ => return,
        };
        
        // Update params with current grid settings
        let world_diameter = self.config.sphere_radius * 2.0;
        let grid_resolution = 128u32;
        let cell_size = world_diameter / grid_resolution as f32;
        let grid_origin = [-world_diameter / 2.0, -world_diameter / 2.0, -world_diameter / 2.0];
        consumption_system.update_params(queue, grid_resolution, cell_size, grid_origin);
        
        // Cache physics bind groups (one per triple buffer index) instead of creating per-frame
        if self.phagocyte_physics_bind_groups.is_none() {
            self.phagocyte_physics_bind_groups = Some([
                consumption_system.create_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[0],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                consumption_system.create_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[1],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                consumption_system.create_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[2],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
            ]);
        }
        
        // Create nutrient bind group (can be cached as these buffers don't change)
        if self.phagocyte_nutrient_bind_group.is_none() {
            self.phagocyte_nutrient_bind_group = Some(consumption_system.create_nutrient_bind_group(
                device,
                fluid_sim.current_state_buffer(),
                fluid_sim.nutrient_voxels_buffer(),
                &self.gpu_triple_buffers.cell_types,
                &self.gpu_triple_buffers.nutrients_buffer,
                &self.gpu_triple_buffers.split_nutrient_thresholds,
                &self.gpu_triple_buffers.death_flags,
            ));
        }
        
        // Use cached bind groups - select by output buffer index (where physics wrote positions)
        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        if let (Some(ref physics_bgs), Some(ref nutrient_bg)) = (&self.phagocyte_physics_bind_groups, &self.phagocyte_nutrient_bind_group) {
            // Dispatch at full capacity — the shader reads cell_count_buffer[0] internally.
            // Using total_cell_slots (async readback, lags 1-3 frames) would under-dispatch
            // and miss cells at higher slot indices during the lag window.
            consumption_system.run(encoder, &physics_bgs[output_idx], nutrient_bg, self.gpu_triple_buffers.capacity);
        }
    }
    
    /// Run light field computation only (no photocyte consumption).
    /// Must run every frame (even with 0 cells) so cave shadows are computed for volumetric fog.
    /// Photocyte consumption runs separately inside each physics step.
    fn run_light_field(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        let (light_field, fluid_sim) = match (&self.light_field_system, &self.fluid_simulator) {
            (Some(l), Some(f)) => (l, f),
            _ => return,
        };
        
        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        
        // Cache occupancy bind groups (one per triple buffer index)
        if self.cached_occupancy_bind_groups.is_none() {
            self.cached_occupancy_bind_groups = Some([
                light_field.create_occupancy_bind_group(
                    device,
                    &self.gpu_triple_buffers.position_and_mass[0],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                light_field.create_occupancy_bind_group(
                    device,
                    &self.gpu_triple_buffers.position_and_mass[1],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                light_field.create_occupancy_bind_group(
                    device,
                    &self.gpu_triple_buffers.position_and_mass[2],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
            ]);
        }
        
        // Cache light field bind group (solid_mask buffer doesn't change)
        if self.cached_light_field_bind_group.is_none() {
            self.cached_light_field_bind_group = Some(
                light_field.create_light_field_bind_group(device, fluid_sim.solid_mask_buffer())
            );
        }
        
        // Use cached bind groups instead of creating per-frame
        light_field.update_light_field_params(queue, self.current_time);
        light_field.update_occupancy_params(queue);
        
        let total_voxels = 128u32 * 128 * 128;
        let light_workgroups = (total_voxels + 63) / 64;
        
        // Clear occupancy grid using DMA (faster than compute dispatch)
        encoder.clear_buffer(light_field.cell_occupancy_buffer_ref(), 0, None);
        
        // Clear light field buffer so stale values don't persist when the compute
        // dispatch is skipped (e.g. no cells + fog off). Without this, the last
        // frame's shadow pattern burns into the camera until cells reappear.
        encoder.clear_buffer(light_field.light_field_buffer(), 0, None);
        
        let occupancy_bg = &self.cached_occupancy_bind_groups.as_ref().unwrap()[output_idx];
        let light_field_bg = self.cached_light_field_bind_group.as_ref().unwrap();
        
        if self.current_cell_count > 0 {
            let cell_workgroups = (self.gpu_triple_buffers.capacity + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Cell Occupancy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(light_field.build_occupancy_pipeline_ref());
            pass.set_bind_group(0, occupancy_bg, &[]);
            pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }
        
        // Compute light field — skip entirely when there are no cells and volumetric fog
        // is not shown. The ray-march dispatches 32,768 workgroups (128³/64) every frame;
        // with no photocytes the result is a uniform dark field, so the work is wasted.
        // Volumetric fog reads the light field, so we must still run it when fog is on.
        if self.current_cell_count == 0 && !self.show_volumetric_fog {
            return;
        }

        // Compute light field
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Light Field"),
                timestamp_writes: None,
            });
            pass.set_pipeline(light_field.compute_light_pipeline_ref());
            pass.set_bind_group(0, light_field_bg, &[]);
            pass.dispatch_workgroups(light_workgroups, 1, 1);
        }
    }
    
    /// Run photocyte light consumption compute shader.
    /// Called inside each physics step so photocytes gain/lose mass at the same rate as other cells.
    fn run_photocyte_light_consumption(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        // Note: No CPU-side early-out on current_cell_count here.
        // The GPU shader reads cell_count_buffer[0] (high water mark) for bounds checking.
        // Using the async-readback live count would cause premature consumption freeze
        // when the readback lags behind the actual GPU state.
        
        let light_field = match &self.light_field_system {
            Some(l) => l,
            None => return,
        };
        
        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        
        // Cache photocyte physics bind groups (one per triple buffer index)
        if self.cached_photocyte_physics_bind_groups.is_none() {
            self.cached_photocyte_physics_bind_groups = Some([
                light_field.create_photocyte_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[0],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                light_field.create_photocyte_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[1],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                light_field.create_photocyte_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[2],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
            ]);
        }
        
        // Cache photocyte system bind group (buffers don't change)
        if self.cached_photocyte_system_bind_group.is_none() {
            self.cached_photocyte_system_bind_group = Some(
                light_field.create_photocyte_system_bind_group(
                    device,
                    &self.gpu_triple_buffers.cell_types,
                    &self.gpu_triple_buffers.nutrients_buffer,
                    &self.gpu_triple_buffers.split_nutrient_thresholds,
                    &self.gpu_triple_buffers.death_flags,
                )
            );
        }
        
        // Use cached bind groups
        light_field.update_photocyte_params(queue);
        
        let photocyte_physics_bg = &self.cached_photocyte_physics_bind_groups.as_ref().unwrap()[output_idx];
        let photocyte_system_bg = self.cached_photocyte_system_bind_group.as_ref().unwrap();
        
        // Dispatch at full capacity — the shader reads cell_count_buffer[0] (the GPU-side
        // high-water mark) for its own bounds check. Using total_cell_slots (the async
        // readback value, which lags 1-3 frames) would under-dispatch and miss photocytes
        // at higher slot indices during the lag window, causing them to receive no light.
        let cell_workgroups = (self.gpu_triple_buffers.capacity + 255) / 256;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Photocyte Light Consumption (physics step)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(light_field.photocyte_light_pipeline_ref());
        pass.set_bind_group(0, photocyte_physics_bg, &[]);
        pass.set_bind_group(1, photocyte_system_bg, &[]);
        pass.dispatch_workgroups(cell_workgroups, 1, 1);
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
        // Always copy full capacity so that child B slots created by division are
        // included even when the GPU-side high-water mark (cell_count_buffer[0]) has
        // advanced beyond total_cell_slots (which lags 1-3 frames behind the async
        // readback). If we truncated to total_cell_slots, a new child B at a higher
        // slot index would not be copied and the instance builder would render stale
        // data from the previous occupant — causing garbage cells to flicker.
        // The instance builder shader uses cell_count_buffer[0] (always current) for
        // its own bounds check, so bytes beyond the true high-water mark are never read.
        let copy_slots = self.gpu_triple_buffers.capacity as usize;
        let vec4_copy_size = (copy_slots * 16) as u64; // Vec4<f32> = 16 bytes
        let u32_copy_size = (copy_slots * 4) as u64; // u32 = 4 bytes

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
        
        // Copy mode properties v0 only (16 bytes/mode) - needed for swim_force -> tail_speed
        // (swim_force is index 6, in v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low])
        // instance_builder.mode_properties_buffer() now holds only v1 (swim_force sub-buffer)
        let total_modes: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        let mode_properties_v1_copy_size = (total_modes * 16) as u64;
        if mode_properties_v1_copy_size > 0 {
            encoder.copy_buffer_to_buffer(
                &self.gpu_triple_buffers.mode_properties_v1,
                0,
                self.instance_builder.mode_properties_buffer(),
                0,
                mode_properties_v1_copy_size,
            );
        }

        // Copy mode properties v5 (16 bytes/mode) - needed for cilia_speed -> ciliocyte visual animation
        // v5: [cilia_speed, cilia_push_bonded, cilia_use_signal, cilia_signal_channel]
        let mode_properties_v5_copy_size = (total_modes * 16) as u64;
        if mode_properties_v5_copy_size > 0 {
            encoder.copy_buffer_to_buffer(
                &self.gpu_triple_buffers.mode_properties_v5,
                0,
                self.instance_builder.mode_properties_v5_buffer(),
                0,
                mode_properties_v5_copy_size,
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
    /// Performs incremental update to avoid affecting other genomes
    pub fn update_genome(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, genome: &Genome) -> Option<usize> {
        // Find existing genome by name
        let genome_id = match self.genomes.iter().position(|g| g.name == genome.name) {
            Some(id) => id,
            None => {
                // Add as new genome if not found
                return self.add_genome(device, genome.clone()).map(|(id, _)| id);
            }
        };
        
        // Check if genome actually changed by comparing key fields including mode contents
        let existing_genome = &self.genomes[genome_id];
        let genome_unchanged = existing_genome.name == genome.name 
            && existing_genome.modes.len() == genome.modes.len()
            && existing_genome.initial_mode == genome.initial_mode
            && existing_genome.initial_orientation == genome.initial_orientation
            && Self::modes_are_identical(&existing_genome.modes, &genome.modes);
            
        if genome_unchanged {
            return Some(genome_id); // No change needed
        }
        
        // Update the genome in place
        self.genomes[genome_id] = genome.clone();
        self.genomes_dirty = true;
        
        // Perform incremental sync of only this genome's data
        self.incremental_sync_genome(device, queue, genome_id);
        
        Some(genome_id)
    }
    
    /// Add a genome to the scene and return its ID and whether it needs GPU sync.
    /// If an IDENTICAL genome exists (same name AND content), reuses it.
    /// If content differs, adds as a NEW genome to preserve existing cells' behavior.
    /// Returns (genome_id, needs_sync) where needs_sync is true if genome was added.
    pub fn add_genome(&mut self, _device: &wgpu::Device, genome: Genome) -> Option<(usize, bool)> {
        // Check if an identical genome already exists
        for (id, existing) in self.genomes.iter().enumerate() {
            if existing.name == genome.name
                && existing.modes.len() == genome.modes.len()
                && existing.initial_mode == genome.initial_mode
                && existing.initial_orientation == genome.initial_orientation
                && Self::modes_are_identical(&existing.modes, &genome.modes)
            {
                // Genome is truly identical, reuse it
                log::info!("Reusing identical genome {} (total: {})", id, self.genomes.len());
                return Some((id, false)); // needs_sync = false, nothing changed
            }
        }

        // Either no match or content differs - add as new genome
        // This preserves existing cells' behavior when genome is modified
        let id = self.genomes.len();
        self.genomes.push(genome);
        self.genomes_dirty = true;
        
        // Update has_oculocytes flag
        self.update_has_oculocytes();

        log::info!("Added new genome {} (total: {})", id, self.genomes.len());
        Some((id, true)) // needs_sync = true because genome was added
    }
    
    /// Check if two mode lists are identical (for genome deduplication)
    fn modes_are_identical(a: &[crate::genome::ModeSettings], b: &[crate::genome::ModeSettings]) -> bool {
        a == b
    }
    
    /// Update has_oculocytes flag based on current genomes.
    /// Also activates the signal system for regulation emitters (channels 8-15),
    /// which use the same signal_flags buffers and propagation pipeline.
    fn update_has_oculocytes(&mut self) {
        use crate::cell::types::CellType;
        let oculocyte_type = CellType::Oculocyte as u32 as i32;
        self.has_oculocytes = false;
        self.max_signal_hops = 0;
        for g in &self.genomes {
            for m in &g.modes {
                if m.cell_type == oculocyte_type {
                    self.has_oculocytes = true;
                    self.max_signal_hops = self.max_signal_hops.max(m.oculocyte_signal_hops.clamp(1, 20) as u32);
                }
                // Regulation emitters (channels 8-15) also need the signal system.
                // Without this, genomes with no oculocytes but with regulation signals
                // never run signal_clear/sense/propagate, so signal_flags stays zero
                // and division/apoptosis/mode-switch gates never open.
                if m.regulation_emit_channel >= 8 {
                    self.has_oculocytes = true;
                    self.max_signal_hops = self.max_signal_hops.max(m.regulation_emit_hops.clamp(1, 20) as u32);
                }
            }
        }
    }
    
    /// Incremental sync of a single genome's data to GPU buffers
    /// This avoids rebuilding the entire buffer layout which would affect other genomes
    fn incremental_sync_genome(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, genome_id: usize) {
        // Calculate global mode index range for this genome
        let global_start_index = self.get_global_mode_index(genome_id, 0);
        let mode_count = self.genomes[genome_id].modes.len();
        
        // Update adhesion settings for this genome's modes only
        self.incremental_sync_adhesion_settings(queue, genome_id, global_start_index, mode_count);
        
        // Clone genome to avoid borrow conflicts
        let genome = self.genomes[genome_id].clone();
        
        // Update mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_mode_properties(queue, &genome, global_start_index);
        
        // Update cilia mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_cilia_mode_properties(queue, &genome, global_start_index);
        
        // Update myocyte mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_myocyte_mode_properties(queue, &genome, global_start_index);

        // Update embryocyte mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_embryocyte_mode_properties(queue, &genome, global_start_index);
        // Update devorocyte mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_devorocyte_mode_properties(queue, &genome, global_start_index);
        // Update vasculocyte mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_vasculocyte_mode_properties(queue, &genome, global_start_index);
        // Update gametocyte mode properties for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_gametocyte_mode_properties(queue, &genome, global_start_index);

        // Update child mode indices for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_child_mode_indices(device, genome_id, global_start_index, mode_count);
        
        // Update genome mode data for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_genome_mode_data(device, genome_id, global_start_index, mode_count);
        
        // Update mode cell types for this genome's modes only
        self.gpu_triple_buffers.incremental_sync_mode_cell_types(queue, &genome, global_start_index);
        
        // Update parent_make_adhesion flags for this genome's modes only
        self.incremental_sync_parent_make_adhesion_flags(queue, genome_id, global_start_index, mode_count);
        
        // Update child keep adhesion flags for this genome's modes only
        self.incremental_sync_child_keep_adhesion_flags(queue, genome_id, global_start_index, mode_count);

        // Update glueocyte env adhesion flags for this genome's modes only
        let env_flags: Vec<u32> = self.genomes[genome_id].modes.iter()
            .map(|mode| if mode.glueocyte_env_adhesion { 1u32 } else { 0u32 })
            .collect();
        if !env_flags.is_empty() {
            let offset = (global_start_index * 4) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.glueocyte_env_adhesion_flags, offset, bytemuck::cast_slice(&env_flags));
        }

        // Update signal-conditional settings (division gating, apoptosis, mode switching)
        // and regulation/oculocyte emission params for this genome's modes only.
        // These have no incremental variant so we write the full slice for this genome.
        let genome_ref = &self.genomes[genome_id];
        let signal_v0: Vec<[f32; 4]> = genome_ref.modes.iter().map(|mode| [
            mode.division_signal_channel as f32,
            mode.division_signal_threshold,
            if mode.division_signal_invert { 1.0 } else { 0.0 },
            mode.apoptosis_signal_channel as f32,
        ]).collect();
        let signal_v1: Vec<[f32; 4]> = genome_ref.modes.iter().map(|mode| [
            mode.apoptosis_signal_threshold,
            if mode.apoptosis_signal_invert { 1.0 } else { 0.0 },
            mode.signal_child_a_channel as f32,
            mode.signal_child_a_threshold,
        ]).collect();
        // v2 and v3 contain child routing mode indices — must remap local → absolute
        let remap = |local: i32| -> f32 {
            if local < 0 { -1.0 } else { (global_start_index as i32 + local.max(0)) as f32 }
        };
        let signal_v2: Vec<[f32; 4]> = genome_ref.modes.iter().map(|mode| [
            remap(mode.signal_child_a_mode_above),
            remap(mode.signal_child_a_mode_below),
            mode.signal_child_b_channel as f32,
            mode.signal_child_b_threshold,
        ]).collect();
        let signal_v3: Vec<[f32; 4]> = genome_ref.modes.iter().map(|mode| [
            remap(mode.signal_child_b_mode_above),
            remap(mode.signal_child_b_mode_below),
            mode.mode_switch_signal_channel as f32,
            mode.mode_switch_signal_threshold,
        ]).collect();
        let signal_v4: Vec<[f32; 4]> = genome_ref.modes.iter().map(|mode| [
            remap(mode.mode_switch_target),
            if mode.mode_switch_invert { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ]).collect();
        if !signal_v0.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.signal_settings_v0, offset, bytemuck::cast_slice(&signal_v0));
            queue.write_buffer(&self.gpu_triple_buffers.signal_settings_v1, offset, bytemuck::cast_slice(&signal_v1));
            queue.write_buffer(&self.gpu_triple_buffers.signal_settings_v2, offset, bytemuck::cast_slice(&signal_v2));
            queue.write_buffer(&self.gpu_triple_buffers.signal_settings_v3, offset, bytemuck::cast_slice(&signal_v3));
            queue.write_buffer(&self.gpu_triple_buffers.signal_settings_v4, offset, bytemuck::cast_slice(&signal_v4));
        }
        // Regulation params: [channel(u32), value_bits(u32), hops(u32), padding(u32)]
        let reg_params: Vec<[u32; 4]> = genome_ref.modes.iter().map(|mode| {
            let channel = if mode.regulation_emit_channel < 0 {
                0xFFFFFFFFu32
            } else {
                (mode.regulation_emit_channel as u32).clamp(8, 15)
            };
            [
                channel,
                mode.regulation_emit_value.clamp(0.0, 2047.0).to_bits(),
                mode.regulation_emit_hops.clamp(1, 20) as u32,
                0u32,
            ]
        }).collect();
        if !reg_params.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.regulation_params, offset, bytemuck::cast_slice(&reg_params));
        }

        // Oculocyte params: [sense_type(u32), ray_length_bits(u32), hops(u32), channel(u32)]
        let oculo_params: Vec<[u32; 4]> = genome_ref.modes.iter().map(|mode| {
            [
                mode.oculocyte_sense_type, // u32 bitmask, no clamping needed
                mode.oculocyte_ray_length.to_bits(),
                mode.oculocyte_signal_hops.clamp(1, 20) as u32,
                mode.oculocyte_signal_channel.clamp(0, 7) as u32,
            ]
        }).collect();
        if !oculo_params.is_empty() {
            let offset = (global_start_index * 16) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.oculocyte_params, offset, bytemuck::cast_slice(&oculo_params));
        }

        // Oculocyte signal values: one f32 per mode
        let oculo_signal_values: Vec<f32> = genome_ref.modes.iter().map(|mode| {
            mode.oculocyte_signal_value.clamp(0.0, 2047.0)
        }).collect();
        if !oculo_signal_values.is_empty() {
            let offset = (global_start_index * 4) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.oculocyte_signal_values, offset, bytemuck::cast_slice(&oculo_signal_values));
        }

        // Refresh has_oculocytes / max_signal_hops in case oculocyte or regulation settings changed
        self.update_has_oculocytes();
        
        log::info!("Incrementally synced genome {} (modes: {} at global index {})", 
            genome_id, mode_count, global_start_index);
    }
    
    /// Incremental sync of adhesion settings for a single genome
    fn incremental_sync_adhesion_settings(&mut self, queue: &wgpu::Queue, genome_id: usize, global_start_index: usize, _mode_count: usize) {
        let genome = &self.genomes[genome_id];
        
        let mut v0: Vec<[f32; 4]> = Vec::new();
        let mut v1: Vec<[f32; 4]> = Vec::new();
        let mut v2: Vec<[f32; 4]> = Vec::new();
        
        for mode in &genome.modes {
            let s = &mode.adhesion_settings;
            v0.push([if s.can_break { 1.0 } else { 0.0 }, s.break_force, s.rest_length, s.linear_spring_stiffness]);
            v1.push([s.linear_spring_damping, s.orientation_spring_stiffness, s.orientation_spring_damping, s.max_angular_deviation]);
            v2.push([s.twist_constraint_stiffness, s.twist_constraint_damping, if s.enable_twist_constraint { 1.0 } else { 0.0 }, 0.0]);
        }
        
        if !v0.is_empty() {
            let byte_offset = (global_start_index * 16) as u64;
            queue.write_buffer(&self.adhesion_buffers.adhesion_settings_v0, byte_offset, bytemuck::cast_slice(&v0));
            queue.write_buffer(&self.adhesion_buffers.adhesion_settings_v1, byte_offset, bytemuck::cast_slice(&v1));
            queue.write_buffer(&self.adhesion_buffers.adhesion_settings_v2, byte_offset, bytemuck::cast_slice(&v2));
        }
    }
    
    /// Incremental sync of parent_make_adhesion flags for a single genome
    fn incremental_sync_parent_make_adhesion_flags(&mut self, queue: &wgpu::Queue, genome_id: usize, global_start_index: usize, mode_count: usize) {
        let genome = &self.genomes[genome_id];
        
        // Ensure parent_make_adhesion_flags is large enough
        let total_required_size = global_start_index + mode_count;
        if self.parent_make_adhesion_flags.len() < total_required_size {
            self.parent_make_adhesion_flags.resize(total_required_size, false);
        }
        
        // Update flags for this genome's modes only
        let mut flags_data: Vec<u32> = Vec::with_capacity(mode_count);
        for (i, mode) in genome.modes.iter().enumerate() {
            let global_index = global_start_index + i;
            self.parent_make_adhesion_flags[global_index] = mode.parent_make_adhesion;
            flags_data.push(if mode.parent_make_adhesion { 1 } else { 0 });
        }
        
        // Sync to GPU buffer at the correct offset
        if !flags_data.is_empty() {
            let offset = (global_start_index * std::mem::size_of::<u32>()) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.parent_make_adhesion_flags, offset, bytemuck::cast_slice(&flags_data));
        }
    }
    
    /// Incremental sync of child keep adhesion flags for a single genome
    fn incremental_sync_child_keep_adhesion_flags(&self, queue: &wgpu::Queue, genome_id: usize, global_start_index: usize, mode_count: usize) {
        let genome = &self.genomes[genome_id];
        
        let mut child_a_flags: Vec<u32> = Vec::with_capacity(mode_count);
        let mut child_b_flags: Vec<u32> = Vec::with_capacity(mode_count);
        let mut child_a_after_split_flags: Vec<u32> = Vec::with_capacity(mode_count);
        let mut child_b_after_split_flags: Vec<u32> = Vec::with_capacity(mode_count);
        
        for mode in &genome.modes {
            child_a_flags.push(if mode.child_a.keep_adhesion { 1 } else { 0 });
            child_b_flags.push(if mode.child_b.keep_adhesion { 1 } else { 0 });
            child_a_after_split_flags.push(if mode.child_a_after_split_keep_adhesion { 1 } else { 0 });
            child_b_after_split_flags.push(if mode.child_b_after_split_keep_adhesion { 1 } else { 0 });
        }
        
        if !child_a_flags.is_empty() {
            let offset = (global_start_index * std::mem::size_of::<u32>()) as u64;
            queue.write_buffer(&self.gpu_triple_buffers.child_a_keep_adhesion_flags, offset, bytemuck::cast_slice(&child_a_flags));
            queue.write_buffer(&self.gpu_triple_buffers.child_b_keep_adhesion_flags, offset, bytemuck::cast_slice(&child_b_flags));
            queue.write_buffer(&self.gpu_triple_buffers.child_a_after_split_keep_adhesion_flags, offset, bytemuck::cast_slice(&child_a_after_split_flags));
            queue.write_buffer(&self.gpu_triple_buffers.child_b_after_split_keep_adhesion_flags, offset, bytemuck::cast_slice(&child_b_after_split_flags));
        }
    }
    
    /// Sync adhesion settings from genomes to GPU
    /// Call this after adding genomes to ensure settings are uploaded to GPU
    pub fn sync_adhesion_settings(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Grow mode pool if needed before syncing
        let total_modes: u64 = self.genomes.iter().map(|g| g.modes.len() as u64).sum();
        if total_modes > 0 {
            self.gpu_triple_buffers.grow_mode_pool_if_needed(device, total_modes);
            self.adhesion_buffers.grow_adhesion_mode_pool_if_needed(device, total_modes);
        }

        self.adhesion_buffers.sync_adhesion_settings(queue, &self.genomes);
        
        // Sync parent_make_adhesion_flags to triple buffer system
        self.gpu_triple_buffers.sync_parent_make_adhesion_flags(queue, &self.genomes);
        
        // Sync child keep adhesion flags for zone-based inheritance
        self.gpu_triple_buffers.sync_child_keep_adhesion_flags(queue, &self.genomes);
        
        // Sync mode properties (nutrient_gain_rate, max_cell_size, etc.) for division
        self.gpu_triple_buffers.sync_mode_properties(queue, &self.genomes);
        
        // Sync cilia mode properties for cilia_force shader (v5, v6)
        self.gpu_triple_buffers.sync_cilia_mode_properties(queue, &self.genomes);
        
        // Sync myocyte mode properties for muscle_contraction shader (v7, v8)
        self.gpu_triple_buffers.sync_myocyte_mode_properties(queue, &self.genomes);

        // Sync embryocyte mode properties for lifecycle shaders (v9, v10)
        self.gpu_triple_buffers.sync_embryocyte_mode_properties(queue, &self.genomes);
        // Sync devorocyte mode properties (v11)
        self.gpu_triple_buffers.sync_devorocyte_mode_properties(queue, &self.genomes);
        // Sync vasculocyte mode properties (v12)
        self.gpu_triple_buffers.sync_vasculocyte_mode_properties(queue, &self.genomes);
        // Sync gametocyte mode properties (v13)
        self.gpu_triple_buffers.sync_gametocyte_mode_properties(queue, &self.genomes);

        // Sync mode cell types lookup table (for deriving cell_type from mode_index)
        self.gpu_triple_buffers.sync_mode_cell_types(queue, &self.genomes);

        // Sync behavior flags for all cell types (applies_swim_force, etc.)
        self.gpu_triple_buffers.sync_behavior_flags(queue);

        // Sync glueocyte env adhesion flags (one u32 per mode)
        self.gpu_triple_buffers.sync_glueocyte_env_adhesion_flags(queue, &self.genomes);
        self.gpu_triple_buffers.sync_glueocyte_boulder_adhesion_flags(queue, &self.genomes);

        // Sync glueocyte cell adhesion flags (4 u32 per mode: enabled, channel, threshold, padding)
        self.gpu_triple_buffers.sync_glueocyte_cell_adhesion_flags(queue, &self.genomes);

        // Sync oculocyte parameters (sense_type, sense_range, signal_hops, signal_channel per mode)
        self.gpu_triple_buffers.sync_oculocyte_params(queue, &self.genomes);

        // Sync regulation emission parameters (emit_channel, emit_value, emit_hops per mode)
        self.gpu_triple_buffers.sync_regulation_params(queue, &self.genomes);

        // Sync signal-conditional settings (division gating, apoptosis, child routing, mode switching)
        self.gpu_triple_buffers.sync_signal_settings(queue, &self.genomes);

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
        self.pending_cell_insertion = Some((world_position, genome, 0, 0));
    }
    
    /// Process any pending cell insertion during render frame when device/encoder/queue are available.
    /// Returns true if a cell was inserted.
    pub fn process_pending_insertion(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        if let Some((world_position, genome, initial_reserve, initial_nutrients)) = self.pending_cell_insertion.take() {
            self.insert_cell_from_genome(device, encoder, queue, world_position, &genome, initial_reserve, initial_nutrients).is_some()
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
        initial_reserve: u32,
        initial_nutrients: u32,
    ) -> Option<usize> {
        // Note: We don't block insertion based on current_cell_count because:
        // 1. current_cell_count is the "high water mark" - it never decreases when cells die
        // 2. Dead cell slots are recycled via a GPU ring buffer
        // 3. The GPU cell_insertion shader handles slot allocation from the ring buffer
        // 4. The shader will silently fail if truly at capacity (no ring buffer slots and at max)
        
        // Find or add the genome (add_genome now updates existing genomes with same name)
        let (genome_id, needs_sync) = match self.add_genome(device, genome.clone()) {
            Some((id, sync)) => (id, sync),
            None => {
                log::error!("Failed to add genome - at maximum capacity");
                return None;
            }
        };
        let mode_idx = genome.initial_mode.max(0) as usize;
        
        // Sync settings to GPU if genome was added or updated
        if needs_sync {
            self.sync_adhesion_settings(device, queue);
        }
        
        // Calculate initial radius from mass (mass = 4/3 * pi * r^3 for unit density)
        // Set initial mass based on cell type and split mass
        // Phagocytes need enough mass to split once and have offspring survive briefly
        let mode = &genome.modes[mode_idx];
        let initial_mass = if mode.cell_type == 2 {
            // Phagocyte: Start with enough mass to split once
            // Use split_mass * 1.2 to ensure split happens and offspring have buffer
            (mode.split_mass * 1.2).max(2.0)
        } else {
            1.0_f32
        };
        
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
                if initial_reserve != 0 { Some(initial_reserve) } else { None },
                if initial_nutrients != 0 { Some(initial_nutrients) } else { None },
            );
            
            // Update local tracking
            let cell_index = self.current_cell_count as usize;
            self.current_cell_count += 1;
            self.total_cell_slots = self.total_cell_slots.max(self.current_cell_count);
            self.next_cell_id += 1;

            Some(cell_index)
        } else {
            None
        }
    }
    
    /// Mark a cell as being dragged so physics skips it.
    /// The dragged cell's position will be controlled by the user via update_position shader.
    pub fn set_dragged_cell(&mut self, cell_index: u32) {
        self.dragged_cell_index = cell_index;
    }
    
    /// Clear the dragged cell so physics resumes for all cells.
    pub fn clear_dragged_cell(&mut self) {
        self.dragged_cell_index = u32::MAX;
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
        // Check for pending result from GPU spatial query (new cell selected)
        if let Some(cell_idx) = self.pending_inspect_result.take() {
            radial_menu.inspected_cell = Some(cell_idx);
            self.pending_cell_extraction = Some(cell_idx as u32);
            log::info!("Inspecting cell {}", cell_idx);
        }

        // Re-queue extraction every frame while a cell is selected and no extraction
        // is already in flight — this gives live-updating data in the inspector panel.
        if let Some(cell_idx) = radial_menu.inspected_cell {
            let inspector_idle = self.cell_inspector
                .as_ref()
                .map_or(true, |i| !i.is_extracting());
            if inspector_idle && self.pending_cell_extraction.is_none() {
                self.pending_cell_extraction = Some(cell_idx as u32);
            }
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
            // Mark cell as dragged so physics skips it
            self.dragged_cell_index = cell_idx as u32;
            log::info!("Started dragging cell {} at distance {}", cell_idx, distance);
        }
    }

    // ── Organism follow camera ────────────────────────────────────────────────

    /// Start a spatial query to find the cell under the cursor for organism following.
    /// Called on double-click when no tool is active.
    pub fn start_organism_follow_query(&mut self, screen_x: f32, screen_y: f32) {
        self.pending_query_position = Some((screen_x, screen_y, ToolQueryType::Follow));
    }

    /// Stop following any organism and return to free camera.
    pub fn clear_organism_follow(&mut self) {
        self.follow_organism_id = None;
        self.follow_pos_staging = None;
        self.follow_lbl_staging = None;
        self.follow_copy_submitted = false;
        self.follow_needs_map = false;
        self.follow_map_ready_flag.store(0, std::sync::atomic::Ordering::Release);
        self.follow_target = glam::Vec3::ZERO;
        self.switch_to_freefly();
    }

    /// Switch the camera to FreeFly mode, positioning it at its current world position.
    fn switch_to_freefly(&mut self) {
        use crate::ui::camera::CameraMode;
        if self.camera.mode != CameraMode::FreeFly {
            // Place the freefly origin at the camera's current world position so
            // there is no jump when the mode switches.
            self.camera.center = self.camera.position();
            self.camera.distance = 0.0;
            self.camera.target_distance = 0.0;
            self.camera.mode = CameraMode::FreeFly;
        }
    }

    /// Returns true if the camera is currently locked to an organism.
    pub fn is_following_organism(&self) -> bool {
        self.follow_organism_id.is_some()
    }

    /// Called every frame from inside `render()`, while the encoder is still open.
    /// Encodes the GPU→staging copy and polls the previous frame's result.
    /// Does NOT call map_async — that must happen after queue.submit via
    /// tick_follow_camera_post_submit().
    ///
    /// Copies both the position buffer and the label buffer each frame. On readback,
    /// finds the root label of the followed cell and computes the average position of
    /// all cells in the same organism. This means the camera follows the organism's
    /// centre-of-mass rather than a single cell, and survives individual cell death
    /// as long as any cell in the organism is still alive.
    pub fn tick_follow_camera(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
    ) {
        let cell_idx = match self.follow_organism_id {
            Some(id) => id as usize,
            None => return,
        };

        let cell_count = self.total_cell_slots as usize;
        if cell_count == 0 || cell_idx >= cell_count {
            self.camera.center = self.follow_center;
            return;
        }

        // ── Step 1: poll for the previous frame's copy result ─────────────────
        // Both pos and label buffers are mapped; we wait for both (flag >= 2).
        if self.follow_copy_submitted {
            let _ = device.poll(wgpu::PollType::Poll);

            let ready_count = self.follow_map_ready_flag
                .load(std::sync::atomic::Ordering::Acquire);
            if ready_count >= 2 {
                self.follow_map_ready_flag
                    .store(0, std::sync::atomic::Ordering::Release);
                self.follow_copy_submitted = false;

                // If the label buffer was copied on a reset frame, labels are
                // temporarily set to each cell's own index — the CoM scan would
                // find only 1 matching cell and produce a jump. Skip this readback.
                let skip_update = self.follow_label_was_reset;
                self.follow_label_was_reset = false;

                let mut new_target: Option<glam::Vec3> = None;
                let mut organism_alive = false;

                if !skip_update {
                    if let (Some(ref pos_buf), Some(ref lbl_buf)) =
                        (&self.follow_pos_staging, &self.follow_lbl_staging)
                    {
                        let pos_view = pos_buf.slice(..).get_mapped_range();
                        let lbl_view = lbl_buf.slice(..).get_mapped_range();
                        let positions: &[[f32; 4]] = bytemuck::cast_slice(&pos_view);
                        let labels: &[u32] = bytemuck::cast_slice(&lbl_view);

                        // Resolve the organism root label from the clicked cell.
                        let root_label = if let Some(lbl) = labels.get(cell_idx).copied() {
                            if lbl != 0xFFFF_FFFFu32 { lbl } else { cell_idx as u32 }
                        } else {
                            cell_idx as u32
                        };

                        // Compute the centre-of-mass of all live cells with this root label.
                        let mut sum = glam::Vec3::ZERO;
                        let mut count = 0u32;
                        let n = positions.len().min(labels.len());
                        for i in 0..n {
                            if labels[i] == root_label {
                                let mass = positions[i][3];
                                if mass > 0.0 {
                                    sum += glam::Vec3::new(
                                        positions[i][0],
                                        positions[i][1],
                                        positions[i][2],
                                    );
                                    count += 1;
                                    organism_alive = true;
                                }
                            }
                        }

                        if count > 0 {
                            new_target = Some(sum / count as f32);
                            self.follow_organism_id = Some(root_label);
                        }

                        drop(lbl_view);
                        drop(pos_view);
                        lbl_buf.unmap();
                        pos_buf.unmap();
                    }
                } else {
                    // Reset frame — just unmap without reading.
                    organism_alive = true; // assume still alive, check next frame
                    if let Some(ref pos_buf) = self.follow_pos_staging { pos_buf.unmap(); }
                    if let Some(ref lbl_buf) = self.follow_lbl_staging { lbl_buf.unmap(); }
                }

                if let Some(target) = new_target {
                    self.follow_target = target;
                } else if !organism_alive {
                    self.follow_pos_staging = None;
                    self.follow_lbl_staging = None;
                    self.follow_organism_id = None;
                    self.follow_needs_map = false;
                    self.switch_to_freefly();
                    return;
                }
            }
        }

        // ── Step 1.5: lerp follow_center toward follow_target every frame ─────
        // Spring constant 12 → ~95% convergence in 0.25s at 60fps.
        let alpha = 1.0 - (-12.0_f32 * dt).exp();
        self.follow_center = self.follow_center.lerp(self.follow_target, alpha);

        // ── Step 2: encode fresh copies of position + label buffers ──────────
        if !self.follow_copy_submitted && cell_count > 0 {
            let output_idx = self.gpu_triple_buffers.output_buffer_index();
            let pos_src = &self.gpu_triple_buffers.position_and_mass[output_idx];
            let pos_size = cell_count as u64 * 16;
            let lbl_size = cell_count as u64 * 4;

            // Reallocate staging buffers if the cell count grew.
            let pos_needs_alloc = self.follow_pos_staging
                .as_ref().map(|b| b.size() < pos_size).unwrap_or(true);
            if pos_needs_alloc {
                self.follow_pos_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Follow Pos Staging"),
                    size: pos_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            let lbl_needs_alloc = self.follow_lbl_staging
                .as_ref().map(|b| b.size() < lbl_size).unwrap_or(true);
            if lbl_needs_alloc {
                self.follow_lbl_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Follow Label Staging"),
                    size: lbl_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            if let Some(ref pos_dst) = self.follow_pos_staging {
                encoder.copy_buffer_to_buffer(pos_src, 0, pos_dst, 0, pos_size);
            }

            if let (Some(ref lbl_src), Some(ref lbl_dst)) = (
                self.organism_label_system.as_ref().map(|s| &s.label_buffer),
                self.follow_lbl_staging.as_ref(),
            ) {
                encoder.copy_buffer_to_buffer(lbl_src, 0, lbl_dst, 0, lbl_size);
            }

            // Record whether this copy was taken on a label reset frame.
            // On reset frames the label buffer is temporarily set to each cell's own
            // index, so the CoM scan would find only 1 matching cell and jump.
            self.follow_label_was_reset = self.organism_label_system
                .as_ref()
                .map(|s| s.is_reset_frame())
                .unwrap_or(false);

            self.follow_needs_map = true;
        }

        // ── Step 3: set orbit pivot to the smoothed follow center ────────────
        self.camera.center = self.follow_center;
    }

    /// Called after queue.submit() to call map_async on the staging buffers.
    /// map_async must not be called while the buffer is referenced by a pending
    /// command encoder — doing so causes a wgpu validation error.
    pub fn tick_follow_camera_post_submit(&mut self) {
        if !self.follow_needs_map {
            return;
        }
        self.follow_needs_map = false;

        // Both pos and label buffers need to map; the flag counts completions.
        // Processing happens once both reach 2.
        if let Some(ref pos_dst) = self.follow_pos_staging {
            let flag = self.follow_map_ready_flag.clone();
            pos_dst.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    flag.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                }
            });
            self.follow_copy_submitted = true;
        }

        if let Some(ref lbl_dst) = self.follow_lbl_staging {
            let flag = self.follow_map_ready_flag.clone();
            lbl_dst.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    flag.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                }
            });
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
                        ToolQueryType::Follow => {
                            // The clicked cell index becomes the initial follow target.
                            // The actual organism root label is resolved each frame via
                            // the label buffer readback in update_follow_camera().
                            // We store the cell index as the organism ID for now; it will
                            // be replaced by the true root label on the first readback.
                            self.follow_organism_id = Some(cell_idx as u32);
                            self.follow_center = self.camera.center;
                            self.follow_target = self.camera.center;
                            // Ensure we are in Orbit mode so center is the pivot point.
                            if self.camera.mode != crate::ui::camera::CameraMode::Orbit {
                                self.camera.mode = crate::ui::camera::CameraMode::Orbit;
                                self.camera.distance = (self.camera.position() - self.camera.center).length().max(10.0);
                                self.camera.target_distance = self.camera.distance;
                            }
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
            acceleration_damping: 0.98,
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
            // Reset shadow bind group flag so it gets recreated with the new cave renderer
            self.cave_shadow_bind_group_set = false;
            
            // Create cave-specific physics bind groups
            self.create_cave_physics_bind_groups(device);
            
            // Update solid mask after cave system is initialized
            self.update_solid_mask(&queue); // Note: This needs queue parameter

            // Initialize boulder system now that cave params are available
            self.initialize_boulder_system(device, queue, surface_format);
            
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

        // Recreate cilia force spatial bind group with real cave params buffer
        if let Some(ref cave_renderer) = self.cave_renderer {
            let organism_label_buffer = self.organism_label_system.as_ref().map(|s| &s.label_buffer);
            self.cached_bind_groups.cilia_force_spatial = self.gpu_physics_pipelines.create_cilia_force_spatial_bind_group(
                device,
                &self.gpu_triple_buffers,
                organism_label_buffer,
                Some(cave_renderer.cave_params_buffer()),
            );
        }
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
    
    /// Ensure the shared environment camera buffer exists and update it with current camera data.
    /// Returns the camera buffer reference. Creates the buffer on first call, then reuses it.
    fn ensure_env_camera_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let camera_uniform = self.create_camera_uniform();
        
        if let Some(ref buffer) = self.env_camera_buffer {
            // Reuse existing buffer - just update data
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
        } else {
            // First time: create the buffer with COPY_DST so we can update it
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shared Env Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            self.env_camera_buffer = Some(buffer);
        }
    }
    
    /// Ensure cached camera bind group exists for a specific environment renderer.
    /// Creates the bind group on first call using the shared camera buffer.
    fn ensure_env_camera_bind_groups(&mut self, device: &wgpu::Device) {
        let buffer = match &self.env_camera_buffer {
            Some(b) => b,
            None => return,
        };
        
        // Create voxel camera bind group if needed
        if self.env_camera_bind_group_voxel.is_none() {
            if self.voxel_renderer.is_some() {
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Env Voxel Camera Layout"),
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
                self.env_camera_bind_group_voxel = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Env Voxel Camera Bind Group"),
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
            }
        }
        
        // Create steam particle camera bind group if needed
        if self.env_camera_bind_group_steam.is_none() {
            if let Some(ref renderer) = self.steam_particle_renderer {
                self.env_camera_bind_group_steam = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Env Steam Camera Bind Group"),
                    layout: renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
            }
        }
        
        // Create water particle camera bind group if needed
        if self.env_camera_bind_group_water.is_none() {
            if let Some(ref renderer) = self.water_particle_renderer {
                self.env_camera_bind_group_water = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Env Water Camera Bind Group"),
                    layout: renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
            }
        }
        
        // Create nutrient particle camera bind group if needed
        if self.env_camera_bind_group_nutrient.is_none() {
            if let Some(ref renderer) = self.nutrient_particle_renderer {
                self.env_camera_bind_group_nutrient = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Env Nutrient Camera Bind Group"),
                    layout: renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
            }
        }
        
        // Create death particle camera bind group if needed
        if self.env_camera_bind_group_death.is_none() {
            if let Some(ref renderer) = self.death_particle_renderer {
                self.env_camera_bind_group_death = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Env Death Camera Bind Group"),
                    layout: renderer.camera_bind_group_layout(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
            }
        }
    }
    
    /// Ensure cached adhesion data bind groups exist (one per triple buffer index).
    /// These are created once and reused since the underlying buffers don't change.
    fn ensure_cached_adhesion_bind_groups(&mut self, device: &wgpu::Device) {
        if self.cached_adhesion_data_bind_groups.is_some() {
            return;
        }
        
        let bg0 = self.adhesion_renderer.create_data_bind_group(
            device,
            &self.gpu_triple_buffers.position_and_mass[0],
            &self.adhesion_buffers.adhesion_connections,
            &self.adhesion_buffers.adhesion_counts,
            &self.gpu_triple_buffers.cell_count_buffer,
            &self.adhesion_buffers.signal_flags,
        );
        let bg1 = self.adhesion_renderer.create_data_bind_group(
            device,
            &self.gpu_triple_buffers.position_and_mass[1],
            &self.adhesion_buffers.adhesion_connections,
            &self.adhesion_buffers.adhesion_counts,
            &self.gpu_triple_buffers.cell_count_buffer,
            &self.adhesion_buffers.signal_flags,
        );
        let bg2 = self.adhesion_renderer.create_data_bind_group(
            device,
            &self.gpu_triple_buffers.position_and_mass[2],
            &self.adhesion_buffers.adhesion_connections,
            &self.adhesion_buffers.adhesion_counts,
            &self.gpu_triple_buffers.cell_count_buffer,
            &self.adhesion_buffers.signal_flags,
        );
        self.cached_adhesion_data_bind_groups = Some([bg0, bg1, bg2]);
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
        _queue: &wgpu::Queue,
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
        
        // Create light field system for photocyte nutrients and volumetric fog
        let light_field_system = LightFieldSystem::new(device, world_radius);

        // Rebuild signal sense world data bind group with real light field buffer
        {
            let nutrient_buf = self.fluid_simulator.as_ref()
                .map(|fs| fs.nutrient_voxels_buffer())
                .unwrap_or(&self.signal_sense_dummy_nutrient_buffer);
            let solid_buf = self.fluid_simulator.as_ref()
                .map(|fs| fs.solid_mask_buffer())
                .unwrap_or(&self.signal_sense_dummy_solid_buffer);
            let density_buf = self.gpu_surface_nets.as_ref()
                .map(|sn| sn.density_buffer())
                .unwrap_or(&self.signal_sense_dummy_density_buffer);
            self.cached_bind_groups.signal_sense_world_data = self.gpu_physics_pipelines
                .create_signal_sense_world_data_bind_group(
                    device,
                    &self.signal_sense_world_params_buffer,
                    nutrient_buf,
                    light_field_system.light_field_buffer(),
                    solid_buf,
                    density_buf,
                    self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_state),
                    self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_count),
                );
        }

        self.light_field_system = Some(light_field_system);

        // Initialize moss system now that light field is available
        self.initialize_moss_system(device);
        
        // Create volumetric fog renderer (with half-res support)
        let volumetric_fog_renderer = VolumetricFogRenderer::new(device, surface_format, self.renderer.width, self.renderer.height);
        self.volumetric_fog_renderer = Some(volumetric_fog_renderer);
        
        // Create depth of field renderer (with half-res support)
        let dof_renderer = DepthOfFieldRenderer::new(device, surface_format, self.renderer.width, self.renderer.height);
        self.dof_renderer = Some(dof_renderer);
        
        // Create procedural sun renderer
        let mut sun_renderer = SunRenderer::new(device, surface_format);
        sun_renderer.resize(self.renderer.width, self.renderer.height);
        self.sun_renderer = Some(sun_renderer);

        // Create procedural space skybox renderer
        let skybox_renderer = SkyboxRenderer::new(device, surface_format);
        self.skybox_renderer = Some(skybox_renderer);
        
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
            self.light_field_system.as_ref().map(|lf| lf.shadow_bind_group_layout()),
        );
        
        self.gpu_surface_nets = Some(gpu_surface_nets);
        self.show_gpu_density_mesh = true;
        // Reset shadow bind group flag so it gets recreated with the new renderer
        self.water_shadow_bind_group_set = false;

        log::info!("GPU surface nets renderer initialized");
    }

    /// Initialize the organism skin renderer.
    /// Safe to call multiple times — no-op if already initialized.
    pub fn initialize_organism_skin(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        settings: &crate::ui::OrganismSkinSettings,
    ) {
        if self.organism_skin_renderer.is_some() {
            return;
        }

        // Derive capacity from the triple buffer size (each cell = 4×f32 = 16 bytes in position_and_mass)
        let cell_capacity = (self.gpu_triple_buffers.position_and_mass[0].size() / 16) as u32;

        let renderer = OrganismSkinRenderer::new(
            device,
            surface_format,
            wgpu::TextureFormat::Depth32Float,
            self.config.sphere_radius,
            glam::Vec3::ZERO,
            cell_capacity,
            self.renderer.width,
            self.renderer.height,
            settings,
        );

        self.organism_skin_renderer = Some(renderer);
        self.show_organism_skins = true;
        // Invalidate cached bind groups so they are recreated with the new renderer
        self.organism_skin_density_bind_groups = None;
        self.organism_skin_count_bind_group = None;
        self.organism_skin_compute_bind_groups = None;

        log::info!("Organism skin renderer initialized (grid {}³)", settings.grid_resolution);
    }

    /// Ensure organism skin bind groups are created.
    fn ensure_organism_skin_bind_groups(&mut self, device: &wgpu::Device) {
        let renderer = match &self.organism_skin_renderer {
            Some(r) => r,
            None => return,
        };

        // Dummy count bind group (kept for API compat — not used by shrink-wrap)
        if self.organism_skin_count_bind_group.is_none() {
            if let Some(ref label_system) = self.organism_label_system {
                self.organism_skin_count_bind_group = Some(renderer.create_count_bind_group(
                    device,
                    &self.gpu_triple_buffers.cell_count_buffer,
                    &self.gpu_triple_buffers.death_flags,
                    &label_system.label_buffer,
                ));
            }
        }

        // Dummy density bind groups (kept for API compat — not used by shrink-wrap)
        if self.organism_skin_density_bind_groups.is_none() {
            let bgs = std::array::from_fn(|i| {
                renderer.create_density_bind_group(
                    device,
                    &self.gpu_triple_buffers.position_and_mass[i],
                    &self.gpu_triple_buffers.death_flags,
                    &self.gpu_triple_buffers.cell_count_buffer,
                )
            });
            self.organism_skin_density_bind_groups = Some(bgs);
        }

        // Real shrink-wrap compute bind groups (one per triple-buffer index)
        if self.organism_skin_compute_bind_groups.is_none() {
            if let Some(ref label_system) = self.organism_label_system {
                let bgs = std::array::from_fn(|i| {
                    renderer.create_compute_bind_group(
                        device,
                        &self.gpu_triple_buffers.position_and_mass[i],
                        &self.gpu_triple_buffers.death_flags,
                        &self.gpu_triple_buffers.cell_count_buffer,
                        &label_system.label_buffer,
                        &self.gpu_triple_buffers.spatial_grid_counts,
                        &self.gpu_triple_buffers.spatial_grid_cells,
                        &label_system.stable_id_per_cell_buffer,
                    )
                });
                self.organism_skin_compute_bind_groups = Some(bgs);
            }
        }
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
            simulator.water_velocity_buffer(),
        );

        // Also rebuild boulder physics bind group with real water buffers if boulder system exists
        if self.boulder_system.is_some() {
            let boulder_bufs = self.boulder_system.as_ref().map(|bs| &bs.buffers);
            self.cached_bind_groups.boulder_physics_buffers =
                self.gpu_physics_pipelines.create_boulder_physics_buffers_bind_group(
                    device, boulder_bufs,
                    Some(simulator.water_grid_params_buffer()),
                    Some(simulator.water_bitfield_buffer()),
                );
            // Build bubble compute bind group now that both boulder and water are ready
            if let Some(ref mut bs) = self.boulder_system {
                let bg = bs.bubbles.create_compute_bind_group(
                    device,
                    &bs.buffers,
                    simulator.water_grid_params_buffer(),
                    simulator.water_bitfield_buffer(),
                );
                bs.bubble_compute_bg = Some(bg);
            }
        }

        // Start with empty fluid - no initial sphere spawn
        // Fluid will only appear when continuous spawning is enabled

        // Rebuild signal sense world data bind group with real buffers from fluid simulator
        {
            let light_buf = self.light_field_system.as_ref()
                .map(|lfs| lfs.light_field_buffer())
                .unwrap_or(&self.signal_sense_dummy_light_buffer);
            let density_buf = self.gpu_surface_nets.as_ref()
                .map(|sn| sn.density_buffer())
                .unwrap_or(&self.signal_sense_dummy_density_buffer);
            self.cached_bind_groups.signal_sense_world_data = self.gpu_physics_pipelines
                .create_signal_sense_world_data_bind_group(
                    device,
                    &self.signal_sense_world_params_buffer,
                    simulator.nutrient_voxels_buffer(),
                    light_buf,
                    simulator.solid_mask_buffer(),
                    density_buf,
                    self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_state),
                    self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_count),
                );
        }

        self.fluid_simulator = Some(simulator);
        self.show_gpu_density_mesh = true;

        // Initialize steam particle renderer
        self.initialize_steam_particle_renderer(device, surface_format);

        // Initialize water particle renderer
        self.initialize_water_particle_renderer(device, surface_format);

        // Initialize nutrient particle renderer
        self.initialize_nutrient_particle_renderer(device, queue, surface_format);

        // Initialize death particle renderer (independent of fluid — works in all modes)
        self.initialize_death_particle_renderer(device, surface_format);

        // Initialize phagocyte consumption system
        self.initialize_phagocyte_consumption(device);

        // Initialize devorocyte consumption system
        self.initialize_devorocyte_consumption(device);

        // Initialize gametocyte merge system
        self.initialize_gametocyte_merge(device);

        // Initialize moss system for cave wall vegetation
        self.initialize_moss_system(device);

        log::info!("GPU fluid simulator initialized with solid mask, water bitfield, and nutrient particles for cell buoyancy");
    }

    /// Initialize the phagocyte nutrient consumption system
    fn initialize_phagocyte_consumption(&mut self, device: &wgpu::Device) {
        if self.phagocyte_consumption.is_some() {
            return;
        }

        if self.fluid_simulator.is_some() {
            let world_diameter = self.config.sphere_radius * 2.0;
            let grid_resolution = 128u32;
            let cell_size = world_diameter / grid_resolution as f32;
            let grid_origin = [-world_diameter / 2.0, -world_diameter / 2.0, -world_diameter / 2.0];

            let consumption_system = PhagocyteConsumptionSystem::new(
                device,
                grid_resolution,
                cell_size,
                grid_origin,
            );

            self.phagocyte_consumption = Some(consumption_system);
            log::info!("Phagocyte consumption system initialized");
        }
    }

    /// Initialize the devorocyte consumption system
    fn initialize_devorocyte_consumption(&mut self, device: &wgpu::Device) {
        if self.devorocyte_consumption.is_some() {
            return;
        }
        self.devorocyte_consumption = Some(DevorocyteConsumptionSystem::new(device));
        log::info!("Devorocyte consumption system initialized");
    }

    /// Run devorocyte consumption compute shader — steals nutrients from and kills foreign cells.
    fn run_devorocyte_consumption(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let system = match &self.devorocyte_consumption {
            Some(s) => s,
            None => return,
        };

        // Cache physics bind groups (one per triple buffer index)
        if self.devorocyte_physics_bind_groups.is_none() {
            let bufs = &self.gpu_triple_buffers;
            self.devorocyte_physics_bind_groups = Some([
                system.create_physics_bind_group(
                    device,
                    &bufs.physics_params,
                    &bufs.position_and_mass[0],
                    &bufs.velocity[0],
                    &bufs.position_and_mass[0], // positions_out (same buffer — read-only in shader)
                    &bufs.velocity[0],           // velocities_out (same buffer — read-only in shader)
                    &bufs.cell_count_buffer,
                ),
                system.create_physics_bind_group(
                    device,
                    &bufs.physics_params,
                    &bufs.position_and_mass[1],
                    &bufs.velocity[1],
                    &bufs.position_and_mass[1],
                    &bufs.velocity[1],
                    &bufs.cell_count_buffer,
                ),
                system.create_physics_bind_group(
                    device,
                    &bufs.physics_params,
                    &bufs.position_and_mass[2],
                    &bufs.velocity[2],
                    &bufs.position_and_mass[2],
                    &bufs.velocity[2],
                    &bufs.cell_count_buffer,
                ),
            ]);
        }

        // Cache cell data bind group (buffers don't change after init)
        if self.devorocyte_cell_data_bind_group.is_none() {
            let bufs = &self.gpu_triple_buffers;
            if let Some(org_system) = self.organism_label_system.as_ref() {
                self.devorocyte_cell_data_bind_group = Some(system.create_cell_data_bind_group(
                    device,
                    &bufs.cell_types,
                    &bufs.nutrients_buffer,
                    &bufs.split_nutrient_thresholds,
                    &bufs.death_flags,
                    &org_system.label_buffer,
                    &bufs.genome_ids,
                    &bufs.mode_indices,
                    &bufs.mode_properties_v11,
                ));
            }
        }

        // Cache spatial bind group
        if self.devorocyte_spatial_bind_group.is_none() {
            let bufs = &self.gpu_triple_buffers;
            self.devorocyte_spatial_bind_group = Some(system.create_spatial_bind_group(
                device,
                &bufs.spatial_grid_counts,
                &bufs.spatial_grid_cells,
                &bufs.cell_grid_indices,
            ));
        }

        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        if let (Some(ref physics_bgs), Some(ref cell_bg), Some(ref spatial_bg)) = (
            &self.devorocyte_physics_bind_groups,
            &self.devorocyte_cell_data_bind_group,
            &self.devorocyte_spatial_bind_group,
        ) {
            // Dispatch at full capacity — the shader reads cell_count_buffer[0] internally.
            system.run(encoder, &physics_bgs[output_idx], cell_bg, spatial_bg, self.gpu_triple_buffers.capacity);
        }
    }

    fn initialize_gametocyte_merge(&mut self, device: &wgpu::Device) {
        if self.gametocyte_merge_system.is_some() {
            return;
        }
        self.gametocyte_merge_system = Some(GametocyteMergeSystem::new(device));
        log::info!("Gametocyte merge system initialized");
    }

    /// Run gametocyte merge detection: clear events, dispatch shader, schedule readback.
    fn run_gametocyte_merge(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        let system = match &self.gametocyte_merge_system {
            Some(s) => s,
            None => return,
        };

        // Clear event counter at the start of each frame
        system.clear_events(queue);

        // Cache physics bind groups (one per triple buffer index)
        if self.gametocyte_physics_bind_groups.is_none() {
            let bufs = &self.gpu_triple_buffers;
            self.gametocyte_physics_bind_groups = Some([
                system.create_physics_bind_group(device, &bufs.physics_params, &bufs.position_and_mass[0], &bufs.velocity[0], &bufs.position_and_mass[0], &bufs.velocity[0], &bufs.cell_count_buffer),
                system.create_physics_bind_group(device, &bufs.physics_params, &bufs.position_and_mass[1], &bufs.velocity[1], &bufs.position_and_mass[1], &bufs.velocity[1], &bufs.cell_count_buffer),
                system.create_physics_bind_group(device, &bufs.physics_params, &bufs.position_and_mass[2], &bufs.velocity[2], &bufs.position_and_mass[2], &bufs.velocity[2], &bufs.cell_count_buffer),
            ]);
        }

        // Cache cell data bind group (depends on organism_label_system)
        if self.gametocyte_cell_data_bind_group.is_none() {
            let bufs = &self.gpu_triple_buffers;
            if let Some(org_system) = self.organism_label_system.as_ref() {
                self.gametocyte_cell_data_bind_group = Some(system.create_cell_data_bind_group(
                    device,
                    &bufs.cell_types,
                    &bufs.death_flags,
                    &org_system.label_buffer,
                    &bufs.genome_ids,
                    &bufs.mode_indices,
                    &bufs.mode_properties_v13,
                    &bufs.embryocyte_reserve_buffer,
                ));
            }
        }

        // Cache spatial bind group
        if self.gametocyte_spatial_bind_group.is_none() {
            let bufs = &self.gpu_triple_buffers;
            self.gametocyte_spatial_bind_group = Some(system.create_spatial_bind_group(
                device,
                &bufs.spatial_grid_counts,
                &bufs.spatial_grid_cells,
                &bufs.cell_grid_indices,
            ));
        }

        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        if let (Some(ref physics_bgs), Some(ref cell_bg), Some(ref spatial_bg)) = (
            &self.gametocyte_physics_bind_groups,
            &self.gametocyte_cell_data_bind_group,
            &self.gametocyte_spatial_bind_group,
        ) {
            system.run(encoder, &physics_bgs[output_idx], cell_bg, spatial_bg, self.gpu_triple_buffers.capacity as usize);
            // Schedule async readback of events after dispatch
            if !self.gamete_readback_in_flight {
                system.schedule_readback(encoder);
                self.gamete_readback_in_flight = true;
            }
        }
    }

    /// Poll the gamete events staging buffer and process any completed merges.
    /// Uses a non-blocking poll — if the staging buffer isn't mapped yet, waits until next frame.
    fn poll_gamete_merge_events(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue) {
        if !self.gamete_readback_in_flight {
            return;
        }
        if self.gametocyte_merge_system.is_none() {
            return;
        }

        // Try to non-blockingly map the staging buffer and read events into a local Vec.
        // We scope the borrow on `gametocyte_merge_system` tightly so that `self` can be
        // mutated afterward without a live borrow conflict.
        let maybe_events: Option<Vec<GameteMergeEvent>> = {
            let system = self.gametocyte_merge_system.as_ref().unwrap();
            let buffer_slice = system.staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
            let _ = device.poll(wgpu::PollType::Poll);
            if let Ok(Ok(())) = receiver.try_recv() {
                let data = buffer_slice.get_mapped_range();
                let evs = GametocyteMergeSystem::parse_events(&data);
                drop(data);
                system.staging_buffer.unmap();
                Some(evs)
            } else {
                None
            }
        }; // borrow of gametocyte_merge_system ends here

        if let Some(new_events) = maybe_events {
            self.gamete_readback_in_flight = false;
            self.pending_gamete_merges.extend(new_events);
        }

        self.process_gamete_merge_events();
    }

    /// Perform genome crossover for pending gamete merge events and spawn offspring.
    fn process_gamete_merge_events(&mut self) {
        if self.pending_gamete_merges.is_empty() {
            return;
        }
        let events: Vec<GameteMergeEvent> = std::mem::take(&mut self.pending_gamete_merges);
        let current_frame = self.current_frame as u64;

        for (i, event) in events.iter().enumerate() {
            let genome_a_id = event.genome_a_id as usize;
            let genome_b_id = event.genome_b_id as usize;

            if genome_a_id >= self.genomes.len() || genome_b_id >= self.genomes.len() {
                log::warn!("Gamete merge: out-of-range genome ids {} / {}", genome_a_id, genome_b_id);
                continue;
            }

            // --- Similarity gate ---
            // Compute genome similarity: mode-count alignment × cell-type match fraction.
            let similarity = crate::genome::Genome::similarity(
                &self.genomes[genome_a_id],
                &self.genomes[genome_b_id],
            );

            if similarity < crate::genome::GAMETOCYTE_MIN_SIMILARITY {
                log::debug!(
                    "Gamete merge rejected: '{}' × '{}' similarity {:.2} < {:.2}",
                    self.genomes[genome_a_id].name,
                    self.genomes[genome_b_id].name,
                    similarity,
                    crate::genome::GAMETOCYTE_MIN_SIMILARITY,
                );
                continue; // Incompatible — both cells already died, no offspring spawned
            }

            // --- Crossover ---
            let rng_seed = current_frame
                .wrapping_add((event.cell_a_idx as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_add(event.cell_b_idx as u64)
                .wrapping_add(i as u64);

            let offspring_genome = crate::genome::Genome::crossover(
                &self.genomes[genome_a_id],
                &self.genomes[genome_b_id],
                rng_seed,
            );

            // Determine the initial cell type from the crossover genome.
            // The gametes become whatever the offspring genome's initial mode specifies —
            // no forced override. The combined reserve is only meaningful for Embryocyte
            // initial cells (cell_type == 10); for all other types the reserve is discarded
            // and the cell starts with normal full nutrients instead.
            let initial_idx = offspring_genome.initial_mode as usize;
            let initial_cell_type = offspring_genome
                .modes
                .get(initial_idx)
                .map(|m| m.cell_type)
                .unwrap_or(0);

            let is_embryocyte = initial_cell_type == crate::cell::CellType::Embryocyte as i32;

            // Pass combined reserve only when the initial cell is an Embryocyte.
            // For any other cell type: convert the reserve 1:1 into nutrients (same ×1000
            // fixed-point scale), capped at 100000 (= 100.0, a full nutrient pool).
            // The reserve itself is discarded for non-storage cells.
            let initial_reserve = if is_embryocyte { event.combined_reserve } else { 0 };
            let initial_nutrients = if is_embryocyte { 0 } else {
                event.combined_reserve.min(100_000)
            };

            log::info!(
                "Gamete merge: '{}' × '{}' → '{}' cell_type={} (similarity {:.2}, reserve {})",
                self.genomes[genome_a_id].name,
                self.genomes[genome_b_id].name,
                offspring_genome.name,
                initial_cell_type,
                similarity,
                event.combined_reserve / 1000,
            );

            let spawn_pos = glam::Vec3::new(event.spawn_x, event.spawn_y, event.spawn_z);
            if self.pending_cell_insertion.is_none() {
                self.pending_cell_insertion = Some((spawn_pos, offspring_genome, initial_reserve, initial_nutrients));
            }
        }
    }

    /// Initialize the boulder system after cave is ready
    fn initialize_boulder_system(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) {
        if self.boulder_system.is_some() {
            return;
        }
        if let Some(ref cave_renderer) = self.cave_renderer {
            let world_radius = self.config.sphere_radius;
            let width = self.renderer.width;
            let height = self.renderer.height;
            let mut bs = BoulderSystem::new(
                device, queue,
                surface_format,
                wgpu::TextureFormat::Depth32Float,
                width, height,
                cave_renderer.params(),
                world_radius,
                self.gravity_mode,
            );
            // Apply current settings
            bs.set_target_count(self.boulder_target_count);
            bs.set_initial_moss(self.boulder_initial_moss);
            bs.set_radius(self.boulder_radius);
            bs.radius_min = self.boulder_radius_min;
            bs.radius_max = self.boulder_radius_max;
            bs.moss_min   = self.boulder_moss_min;
            bs.moss_max   = self.boulder_moss_max;
            bs.spawn_interval = self.boulder_spawn_interval;
            bs.buoyancy = self.boulder_buoyancy;
            bs.buoyancy_dirty = true; // write to GPU on first update
            self.boulder_system = Some(bs);

            // Build the renderer's boulder storage bind group from the real buffers.
            let boulder_bg = self.boulder_system.as_ref()
                .map(|bs| bs.renderer.create_boulder_bind_group(device, &bs.buffers));
            if let (Some(ref mut bs), Some(bg)) = (self.boulder_system.as_mut(), boulder_bg) {
                bs.renderer.boulder_bind_group = Some(bg);
            }

            // Rebuild the boulder bind groups now that real buffers exist.
            // The cached bind groups were created at init with dummy buffers;
            // they must be replaced so the shaders read from the actual boulder data.
            let boulder_bufs = self.boulder_system.as_ref().map(|bs| &bs.buffers);
            let org_size_buf = self.organism_label_system.as_ref().map(|s| &s.organism_size_buffer);
            let water_params_buf = self.fluid_simulator.as_ref().map(|fs| fs.water_grid_params_buffer());
            let water_bitfield_buf = self.fluid_simulator.as_ref().map(|fs| fs.water_bitfield_buffer());
            self.cached_bind_groups.boulder_physics_buffers =
                self.gpu_physics_pipelines.create_boulder_physics_buffers_bind_group(device, boulder_bufs, water_params_buf, water_bitfield_buf);
            self.cached_bind_groups.boulder_consume_spatial =
                self.gpu_physics_pipelines.create_boulder_consume_spatial_bind_group(device, &self.gpu_triple_buffers);
            self.cached_bind_groups.boulder_consume_cell_data = [
                self.gpu_physics_pipelines.create_boulder_consume_cell_data_bind_group(device, &self.gpu_triple_buffers, org_size_buf, 0),
                self.gpu_physics_pipelines.create_boulder_consume_cell_data_bind_group(device, &self.gpu_triple_buffers, org_size_buf, 1),
                self.gpu_physics_pipelines.create_boulder_consume_cell_data_bind_group(device, &self.gpu_triple_buffers, org_size_buf, 2),
            ];
            self.cached_bind_groups.boulder_consume_buffers =
                self.gpu_physics_pipelines.create_boulder_consume_buffers_bind_group(device, boulder_bufs);

            // Also rebuild collision and env adhesion bind groups with real boulder buffers
            // so cells can collide with boulders and glueocytes can attach to them.
            let boulder_bufs2 = self.boulder_system.as_ref().map(|bs| &bs.buffers);
            self.cached_bind_groups.collision_force_accum = [
                self.gpu_physics_pipelines.create_collision_force_accum_bind_group(device, &self.adhesion_buffers, &self.gpu_triple_buffers, 0, boulder_bufs2),
                self.gpu_physics_pipelines.create_collision_force_accum_bind_group(device, &self.adhesion_buffers, &self.gpu_triple_buffers, 1, boulder_bufs2),
                self.gpu_physics_pipelines.create_collision_force_accum_bind_group(device, &self.adhesion_buffers, &self.gpu_triple_buffers, 2, boulder_bufs2),
            ];
            self.cached_bind_groups.env_adhesion_force_accum = [
                self.gpu_physics_pipelines.create_env_adhesion_force_accum_bind_group(device, &self.adhesion_buffers, &self.gpu_triple_buffers, 0, boulder_bufs2),
                self.gpu_physics_pipelines.create_env_adhesion_force_accum_bind_group(device, &self.adhesion_buffers, &self.gpu_triple_buffers, 1, boulder_bufs2),
                self.gpu_physics_pipelines.create_env_adhesion_force_accum_bind_group(device, &self.adhesion_buffers, &self.gpu_triple_buffers, 2, boulder_bufs2),
            ];

            log::info!("Boulder system initialized");

            // If fluid is already running, build the bubble compute bind group now
            if let Some(ref simulator) = self.fluid_simulator {
                if let Some(ref mut bs) = self.boulder_system {
                    let bg = bs.bubbles.create_compute_bind_group(
                        device,
                        &bs.buffers,
                        simulator.water_grid_params_buffer(),
                        simulator.water_bitfield_buffer(),
                    );
                    bs.bubble_compute_bg = Some(bg);
                }
            }

            // Rebuild signal sense world data bind group with boulder buffers
            {
                let nutrient_buf = self.fluid_simulator.as_ref()
                    .map(|fs| fs.nutrient_voxels_buffer())
                    .unwrap_or(&self.signal_sense_dummy_nutrient_buffer);
                let light_buf = self.light_field_system.as_ref()
                    .map(|lfs| lfs.light_field_buffer())
                    .unwrap_or(&self.signal_sense_dummy_light_buffer);
                let solid_buf = self.fluid_simulator.as_ref()
                    .map(|fs| fs.solid_mask_buffer())
                    .unwrap_or(&self.signal_sense_dummy_solid_buffer);
                let density_buf = self.gpu_surface_nets.as_ref()
                    .map(|sn| sn.density_buffer())
                    .unwrap_or(&self.signal_sense_dummy_density_buffer);
                self.cached_bind_groups.signal_sense_world_data = self.gpu_physics_pipelines
                    .create_signal_sense_world_data_bind_group(
                        device,
                        &self.signal_sense_world_params_buffer,
                        nutrient_buf,
                        light_buf,
                        solid_buf,
                        density_buf,
                        self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_state),
                        self.boulder_system.as_ref().map(|bs| &bs.buffers.boulder_count),
                    );
            }
        }
    }

    /// Initialize the moss system for cave wall vegetation
    fn initialize_moss_system(&mut self, device: &wgpu::Device) {
        if self.moss_system.is_some() {
            return;
        }

        // Moss requires fluid simulator (for fluid_state, water_velocity) and light field
        if self.fluid_simulator.is_some() && self.light_field_system.is_some() {
            let world_diameter = self.config.sphere_radius * 2.0;
            let grid_resolution = 128u32;
            let cell_size = world_diameter / grid_resolution as f32;
            let grid_origin = [-world_diameter / 2.0, -world_diameter / 2.0, -world_diameter / 2.0];

            let moss = MossSystem::new(device, grid_resolution, cell_size, grid_origin);
            self.moss_system = Some(moss);
            // Force cave shadow bind group rebuild to include moss buffer
            self.cave_shadow_bind_group_set = false;
            // Force cell shadow bind group rebuild (light field buffers may have changed)
            self.cell_shadow_bind_group_set = false;
            log::info!("Moss system initialized");
        }
    }

    /// Run moss growth/erosion compute pass
    fn run_moss_growth(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, delta_time: f32) {
        if !self.show_moss {
            return;
        }

        // Create growth bind group if not cached (needs immutable borrows of multiple fields)
        if self.moss_growth_bind_group.is_none() {
            let (moss, fluid_sim, light_field) = match (&self.moss_system, &self.fluid_simulator, &self.light_field_system) {
                (Some(m), Some(f), Some(l)) => (m, f, l),
                _ => return,
            };
            self.moss_growth_bind_group = Some(moss.create_growth_bind_group(
                device,
                fluid_sim.solid_mask_buffer(),
                light_field.light_field_buffer(),
                fluid_sim.current_state_buffer(),
                fluid_sim.water_velocity_buffer(),
            ));
        }

        // Run growth pass — needs &mut self.moss_system for frame throttle counter
        let world_radius = self.config.sphere_radius;
        if let (Some(ref mut moss), Some(ref growth_bg)) = (&mut self.moss_system, &self.moss_growth_bind_group) {
            moss.run_growth(encoder, queue, growth_bg, delta_time, world_radius);
        }
    }

    /// Run moss consumption compute pass (phagocytes eating moss)
    fn run_moss_consumption(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        if !self.show_moss {
            return;
        }

        let moss = match &self.moss_system {
            Some(m) => m,
            None => return,
        };

        // Cache physics bind groups (one per triple buffer index)
        if self.moss_consume_physics_bind_groups.is_none() {
            self.moss_consume_physics_bind_groups = Some([
                moss.create_consume_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[0],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                moss.create_consume_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[1],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
                moss.create_consume_physics_bind_group(
                    device,
                    &self.gpu_triple_buffers.physics_params,
                    &self.gpu_triple_buffers.position_and_mass[2],
                    &self.gpu_triple_buffers.cell_count_buffer,
                ),
            ]);
        }

        // Cache moss bind group
        if self.moss_consume_moss_bind_group.is_none() {
            self.moss_consume_moss_bind_group = Some(moss.create_consume_moss_bind_group(
                device,
                &self.gpu_triple_buffers.cell_types,
                &self.gpu_triple_buffers.nutrients_buffer,
                &self.gpu_triple_buffers.split_nutrient_thresholds,
                &self.gpu_triple_buffers.death_flags,
            ));
        }

        let output_idx = self.gpu_triple_buffers.output_buffer_index();
        if let (Some(ref physics_bgs), Some(ref moss_bg)) = (&self.moss_consume_physics_bind_groups, &self.moss_consume_moss_bind_group) {
            // Dispatch at full capacity — the shader reads cell_count_buffer[0] internally.
            moss.run_consumption(encoder, queue, &physics_bgs[output_idx], moss_bg, self.gpu_triple_buffers.capacity);
        }
    }

    /// Step the GPU fluid simulation and update water bitfield for cell buoyancy
    pub fn step_fluid_simulation(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        if let Some(ref simulator) = self.fluid_simulator {
            simulator.set_gravity_mode(self.gravity_mode);
            simulator.set_surface_pressure(self.surface_pressure);
            simulator.set_water_drag_strength(queue, self.water_viscosity);
            simulator.step(device, queue, encoder, dt, self.gravity, [self.gravity_mode == 0, self.gravity_mode == 1, self.gravity_mode == 2], self.lateral_flow_probabilities, self.condensation_probability, self.vaporization_probability);
            // Update water bitfield for cell physics (compressed 32x for fast lookup)
            simulator.update_water_bitfield(device, encoder);
            
            // NOTE: Nutrient population moved into run_physics() so it executes every
            // physics step. At high sim speeds the physics loop runs N steps per frame;
            // phagocytes consume voxel nutrients each step, so populate must also run
            // each step to reset consumed voxels and keep supply balanced with demand.
        }
    }

    /// Force immediate repopulation of nutrients (e.g., after density change)
    pub fn repopulate_nutrients(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref simulator) = self.fluid_simulator {
            simulator.populate_nutrients(
                device, queue, encoder,
                self.nutrient_density, 0.016,
                self.nutrient_epoch_duration,
                self.nutrient_epoch_spacing,
                self.nutrient_spawn_end,
                self.nutrient_despawn_start,
            );
            log::info!("Nutrients repopulated with density: {}", self.nutrient_density);
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

    /// Disable moss and schedule a one-shot clear of the moss_density buffer.
    /// The clear is issued at the start of the next render() call so the cave
    /// fragment shader immediately stops rendering stale moss.
    pub fn disable_moss(&mut self) {
        self.show_moss = false;
        self.moss_needs_clear = true;
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

    /// Initialize the death particle renderer.
    ///
    /// This can be called at any time — it only needs the device and surface format.
    /// The compute bind group is rebuilt every frame (not cached) because the
    /// position buffer index rotates with the triple-buffer system.
    pub fn initialize_death_particle_renderer(
        &mut self,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) {
        if self.death_particle_renderer.is_some() {
            return;
        }

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Death Particle Camera Bind Group Layout"),
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

        let renderer = DeathParticleRenderer::new(
            device,
            surface_format,
            wgpu::TextureFormat::Depth32Float,
            &camera_layout,
            self.gpu_triple_buffers.capacity,
        );

        self.death_render_bind_group = Some(renderer.create_render_bind_group(device));
        self.death_particle_renderer = Some(renderer);
        self.show_death_particles = true;

        log::info!("Death particle renderer initialized (GPU-based)");
    }

    /// Snapshot death_flags → prev_death_flags BEFORE physics runs this frame.
    /// This must be called at the start of the frame so spawn_new can detect
    /// the alive→dead transition after physics completes.
    pub fn snapshot_death_flags_for_particles(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref renderer) = self.death_particle_renderer {
            renderer.snapshot_death_flags(
                encoder,
                &self.gpu_triple_buffers.death_flags,
                self.gpu_triple_buffers.capacity,
            );
        }
    }

    /// Run the death particle spawn + age compute passes AFTER physics runs.
    /// Rebuilds the compute bind group each frame using the current triple-buffer
    /// position slot (current_index, not output_index).
    pub fn update_death_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
    ) {
        if self.death_particle_renderer.is_none() {
            return;
        }

        // Use current_index() — the last completed physics frame's positions.
        // output_index() is the buffer being written this frame (positions may be
        // partially zeroed by death_scan already).
        let current_idx = self.gpu_triple_buffers.current_index();
        let position_buffer = &self.gpu_triple_buffers.position_and_mass[current_idx];

        // Build a fresh bind group each frame (triple-buffer index rotates)
        let compute_bind_group = {
            let renderer = self.death_particle_renderer.as_ref().unwrap();
            renderer.create_compute_bind_group(
                device,
                &self.gpu_triple_buffers.death_flags,
                position_buffer,
            )
        };

        let capacity = self.gpu_triple_buffers.capacity;
        if let Some(ref mut renderer) = self.death_particle_renderer {
            renderer.update(encoder, queue, &compute_bind_group, capacity, dt);
        }
    }

    /// Poll the death particle counter staging buffer after queue submission.
    pub fn poll_death_particle_count(&mut self, device: &wgpu::Device) {
        if let Some(ref mut renderer) = self.death_particle_renderer {
            renderer.poll_particle_count(device);
        }
    }
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
                particle_renderer.create_extract_bind_group(
                    device, 
                    fluid_sim.current_state_buffer(), 
                    fluid_sim.solid_mask_buffer(),
                    fluid_sim.nutrient_voxels_buffer()
                )
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
                particle_renderer.create_extract_bind_group(device, fluid_sim.current_state_buffer(), fluid_sim.solid_mask_buffer(), fluid_sim.water_velocity_buffer())
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

        // Update gravity mode on the particle renderer
        if let Some(ref mut particle_renderer) = self.water_particle_renderer {
            particle_renderer.set_gravity_mode(self.gravity_mode);
        }

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

    /// Apply light & fog parameters from editor state to GPU systems
    pub fn apply_light_params_from_editor(&mut self, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        // Update light field system parameters
        if let Some(ref mut light_field) = self.light_field_system {
            light_field.set_light_dir(editor_state.light_dir);
            light_field.set_max_steps(editor_state.light_field_max_steps);
            light_field.set_absorption_solid(editor_state.light_field_absorption_solid);
            light_field.set_absorption_cell(editor_state.light_field_absorption_cell);
            light_field.set_ambient_floor(editor_state.light_field_ambient_floor);
            light_field.set_mass_per_second(editor_state.photocyte_mass_per_second * editor_state.sun_intensity);
            light_field.set_min_light_threshold(editor_state.photocyte_min_light_threshold);
            light_field.set_shadow_enabled(editor_state.shadow_enabled);
            light_field.set_shadow_strength(editor_state.shadow_strength);
            light_field.set_shadow_quality(editor_state.shadow_quality);
            light_field.set_caustic_intensity(editor_state.caustic_intensity);
            light_field.set_caustic_scale(editor_state.caustic_scale);
            light_field.set_caustic_speed(editor_state.caustic_speed);
            light_field.set_moss_parallax_depth(editor_state.moss_parallax_depth);
            light_field.set_moss_scale(editor_state.moss_scale);
            light_field.set_moss_noise_type(editor_state.moss_noise_type);
            light_field.set_moss_noise_frequency(editor_state.moss_noise_frequency);
            light_field.set_moss_noise_lacunarity(editor_state.moss_noise_lacunarity);
            light_field.set_moss_height_sharpness_low(editor_state.moss_height_sharpness_low);
            light_field.set_moss_height_sharpness_high(editor_state.moss_height_sharpness_high);
            light_field.set_moss_bump_strength(editor_state.moss_bump_strength);
            light_field.set_moss_color_dark(editor_state.moss_color_dark);
            light_field.set_moss_color_bright(editor_state.moss_color_bright);
            light_field.set_sun_color([
                editor_state.sun_color[0] * editor_state.sun_intensity,
                editor_state.sun_color[1] * editor_state.sun_intensity,
                editor_state.sun_color[2] * editor_state.sun_intensity,
            ]);
        }
        
        // Update volumetric fog renderer parameters
        if let Some(ref mut fog_renderer) = self.volumetric_fog_renderer {
            fog_renderer.enabled = editor_state.show_volumetric_fog;
            fog_renderer.fog_density = editor_state.fog_density;
            fog_renderer.fog_steps = editor_state.fog_steps;
            fog_renderer.light_color = editor_state.sun_color;
            fog_renderer.light_intensity = editor_state.sun_intensity * 0.1; // Scale down fog effect
            fog_renderer.fog_color = editor_state.fog_color;
            fog_renderer.scattering_anisotropy = editor_state.fog_scattering_anisotropy;
            fog_renderer.absorption = editor_state.fog_absorption;
            fog_renderer.height_fog_density = editor_state.fog_height_density;
            fog_renderer.height_fog_falloff = editor_state.fog_height_falloff;
        }
        
        // Combine sun color with intensity for all lighting
        let scaled_sun_color = [
            editor_state.sun_color[0] * editor_state.sun_intensity,
            editor_state.sun_color[1] * editor_state.sun_intensity,
            editor_state.sun_color[2] * editor_state.sun_intensity,
        ];
        
        // Update cell renderer light color and direction
        self.renderer.set_light_color(scaled_sun_color);
        // Cell shader expects light_dir pointing FROM light; editor stores direction TOWARD light
        self.renderer.set_light_dir([
            -editor_state.light_dir[0],
            -editor_state.light_dir[1],
            -editor_state.light_dir[2],
        ]);
        
        // Update tail renderer light color
        self.tail_renderer.set_light_color(scaled_sun_color);
        
        // Update sun renderer parameters
        self.show_sun = editor_state.show_sun;
        self.sun_intensity = editor_state.sun_intensity;
        if let Some(ref mut sun) = self.sun_renderer {
            sun.sun_color = editor_state.sun_color;
            sun.sun_angular_radius = editor_state.sun_angular_radius;
        }
        
        // Update depth of field parameters
        self.show_dof = editor_state.show_dof;
        if let Some(ref mut dof) = self.dof_renderer {
            dof.enabled = editor_state.show_dof;
            dof.focal_distance = editor_state.dof_focal_distance;
            dof.focal_range = editor_state.dof_focal_range;
            dof.max_blur_radius = editor_state.dof_max_blur_radius;
            dof.blur_strength = editor_state.dof_blur_strength;
        }
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
            let label_buf = self.organism_label_system.as_ref()
                .map(|s| &s.label_buffer);
            let cell_inspector = GpuCellInspector::new(
                device,
                self.gpu_physics_pipelines.cell_data_extraction.clone(),
                &self.gpu_physics_pipelines.physics_layout,
                &self.gpu_physics_pipelines.cell_extraction_params_layout,
                &self.gpu_physics_pipelines.cell_extraction_state_layout,
                &self.gpu_physics_pipelines.cell_extraction_output_layout,
                &self.gpu_triple_buffers,
                &self.adhesion_buffers,
                label_buf.expect("OrganismLabelSystem must exist before cell inspector is created"),
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
                &self.adhesion_buffers,
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

        // Initialize mutation system bind groups
        if let Some(mutation_system) = &mut self.mutation_system {
            // Rebuild collect bind groups (reads lifecycle buffers)
            mutation_system.rebuild_collect_bind_groups(
                device,
                queue,
                &self.gpu_triple_buffers.division_flags,
                &self.gpu_triple_buffers.division_slot_assignments,
                &self.gpu_triple_buffers.genome_ids,
                &self.gpu_triple_buffers.cell_count_buffer,
            );

            // Rebuild mutation bind groups (reads/writes genome mode buffers)
            mutation_system.rebuild_bind_groups(
                device,
                &self.gpu_triple_buffers.genome_ids,
                &self.gpu_triple_buffers.mode_indices,
                &self.gpu_triple_buffers.cell_types,
                &self.gpu_triple_buffers.mode_properties_v0,
                &self.gpu_triple_buffers.mode_properties_v1,
                &self.gpu_triple_buffers.mode_properties_v2,
                &self.gpu_triple_buffers.mode_properties_v3,
                &self.gpu_triple_buffers.mode_properties_v4,
                &self.gpu_triple_buffers.genome_mode_data_v0,
                &self.gpu_triple_buffers.genome_mode_data_v1,
                &self.gpu_triple_buffers.genome_mode_data_v2,
                &self.gpu_triple_buffers.genome_mode_data_v3,
                &self.gpu_triple_buffers.genome_mode_data_v4,
                &self.gpu_triple_buffers.child_mode_indices,
                &self.gpu_triple_buffers.mode_cell_types,
                &self.gpu_triple_buffers.parent_make_adhesion_flags,
                &self.gpu_triple_buffers.child_a_keep_adhesion_flags,
                &self.gpu_triple_buffers.child_b_keep_adhesion_flags,
                &self.gpu_triple_buffers.glueocyte_env_adhesion_flags,
                &self.gpu_triple_buffers.oculocyte_params,
                self.instance_builder.mode_colors_buffer(),
                self.instance_builder.mode_emissive_buffer(),
                &self.adhesion_buffers.adhesion_settings_v0,
                &self.adhesion_buffers.adhesion_settings_v1,
                &self.adhesion_buffers.adhesion_settings_v2,
                &self.gpu_triple_buffers.signal_settings_v0,
                &self.gpu_triple_buffers.signal_settings_v1,
                &self.gpu_triple_buffers.signal_settings_v2,
                &self.gpu_triple_buffers.signal_settings_v3,
                &self.gpu_triple_buffers.signal_settings_v4,
                &self.gpu_triple_buffers.regulation_params,
                &self.gpu_triple_buffers.child_a_after_split_keep_adhesion_flags,
                &self.gpu_triple_buffers.child_b_after_split_keep_adhesion_flags,
            );

            // Rebuild GC bind group (for genome recycling)
            mutation_system.rebuild_gc_bind_group(device, queue);

            // Rebuild ref_count sync bind group (for accurate ref_count tracking)
            mutation_system.rebuild_ref_count_sync_bind_group(
                device,
                queue,
                &self.gpu_triple_buffers.genome_ids,
                &self.gpu_triple_buffers.death_flags,
            );

            // Sync genome metadata if genomes are already loaded
            if !self.genomes.is_empty() {
                mutation_system.sync_genome_metadata(queue, &self.genomes);
            }
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
            acceleration_damping: 0.98,
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

    /// Capture the cave system into the water reflection cubemap.
    /// Renders the cave from the world center into 6 cubemap faces.
    /// Call once after the cave renderer is initialized.
    pub fn capture_reflection_cubemap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let surface_nets = match self.gpu_surface_nets {
            Some(ref mut sn) => sn,
            None => return,
        };
        let cave_renderer = match self.cave_renderer {
            Some(ref cave) => cave,
            None => return,
        };

        let face_size = surface_nets.cubemap_face_size();

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cubemap Capture Depth"),
            size: wgpu::Extent3d { width: face_size, height: face_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&Default::default());

        // 6 cubemap face directions: +X, -X, +Y, -Y, +Z, -Z
        let face_directions: [(glam::Vec3, glam::Vec3); 6] = [
            (glam::Vec3::X,     glam::Vec3::NEG_Y),
            (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),
            (glam::Vec3::Y,     glam::Vec3::Z),
            (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z),
            (glam::Vec3::Z,     glam::Vec3::NEG_Y),
            (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),
        ];

        let origin = glam::Vec3::ZERO;
        let proj = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cubemap Capture Encoder"),
        });

        for face in 0u32..6 {
            let (forward, up) = face_directions[face as usize];
            let view_mat = glam::Mat4::look_at_rh(origin, origin + forward, up);
            let view_proj = proj * view_mat;

            let face_view = surface_nets.cubemap_face_view(face);

            // Clear face + depth
            {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Cubemap Face Clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &face_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
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

            // Render cave with correct cubemap projection
            cave_renderer.render_to_cubemap_face(
                &mut encoder,
                queue,
                &face_view,
                &depth_view,
                view_proj,
                origin,
            );
        }

        queue.submit(std::iter::once(encoder.finish()));
        log::info!("Reflection cubemap captured ({}x{} per face, cave)", face_size, face_size);
    }
}


impl Scene for GpuScene {
    fn update(&mut self, dt: f32) {
        if self.paused {
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
        outline_width: f32,
    ) {
        // Sync adhesion settings to GPU only when genomes are added or modified
        if self.genomes_dirty {
            self.sync_adhesion_settings(device, queue);
            // Sync mutation system genome metadata when genomes change
            if let Some(mutation_system) = &mut self.mutation_system {
                mutation_system.sync_genome_metadata(queue, &self.genomes);
            }
            // Update mode visuals (colors) from CPU genomes.
            // This must ONLY happen when CPU genomes change, not every frame,
            // because the mutation shader writes directly to mode_visuals_buffer
            // for GPU-mutated genomes and a per-frame overwrite would clobber those.
            if !self.genomes.is_empty() {
                self.instance_builder.update_mode_visuals_from_genomes(device, queue, &self.genomes);
            }
            self.genomes_dirty = false;
        }
        
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

        // On first frame, disable culling since we don't have depth data yet
        if self.first_frame {
            self.instance_builder.set_culling_mode(CullingMode::Disabled);
        }

        // Prepare organism label system for this frame (writes run_init flag to GPU
        // before the encoder is built, so the shader sees it in the same submit).
        if let Some(ref label_system) = self.organism_label_system {
            label_system.prepare_frame(queue);
        }

        // Create single command encoder for all GPU work to avoid multiple queue.submit() calls
        // (each submit is a sync point that kills performance)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Scene Encoder"),
        });

        // If moss was just disabled, clear the moss_density buffer so the cave shader
        // immediately stops rendering stale moss. The buffer persists between frames and
        // the cave fragment shader reads it unconditionally, so without this clear the
        // old moss stays visible until the buffer is overwritten by new growth.
        if self.moss_needs_clear {
            if let Some(ref moss) = self.moss_system {
                encoder.clear_buffer(moss.moss_density_buffer(), 0, None);
            }
            self.moss_needs_clear = false;
        }

        // Update shared environment camera buffer once per frame (reused by voxel, particle renderers)
        // This replaces 4+ per-frame buffer allocations with a single write_buffer call
        self.ensure_env_camera_buffer(device, queue);
        self.ensure_env_camera_bind_groups(device);

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

        // Poll gamete merge events from the staging buffer and process crossovers
        self.poll_gamete_merge_events(device, queue);

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

        // Compute light field once per frame BEFORE physics loop
        // Needed for: (1) photocyte consumption inside each physics step, (2) volumetric fog rendering
        if !self.paused || self.show_volumetric_fog {
            self.run_light_field(device, &mut encoder, queue);
        }

        // NOTE: Moss growth is dispatched AFTER the cave render pass (near queue.submit)
        // to avoid a read-after-write pipeline stall on moss_density_buffer.
        // The cave fragment shader reads moss_density; if growth runs before the render
        // pass in the same encoder, the GPU serializes them — stalling the render pass
        // until the 32K-workgroup compute finishes. Moving it after rendering means the
        // write lands in the buffer that the NEXT frame's render pass will read, which
        // is a 1-frame lag that is completely invisible for a slowly-changing moss field.

        let is_dragging = self.dragged_cell_index != u32::MAX;

        // Write dragged cell index to cell_count_buffer[2] so position_update shader can skip it.
        // Only write when the value has changed — the buffer retains its last value between frames,
        // so we only need to update it when dragging starts, the target changes, or dragging stops.
        if self.dragged_cell_index != self.last_written_dragged_index {
            queue.write_buffer(
                &self.gpu_triple_buffers.cell_count_buffer,
                8, // byte offset for slot [2]
                bytemuck::cast_slice(&[self.dragged_cell_index]),
            );
            self.last_written_dragged_index = self.dragged_cell_index;
        }

        // Execute drag position update BEFORE physics so adhesion springs see the
        // correct dragged cell position and can pull connected cells toward it.
        // update_position.wgsl writes to ALL THREE triple buffers, and
        // position_update.wgsl preserves the dragged cell's position (copies in→out),
        // so one dispatch before physics is sufficient.
        let position_updated = if is_dragging {
            self.execute_pending_position_updates(device, &mut encoder, queue)
        } else {
            false
        };

        // Snapshot death_flags BEFORE physics runs — needed by spawn_new to detect
        // the alive→dead transition this frame.
        if self.show_death_particles && !self.paused {
            self.snapshot_death_flags_for_particles(&mut encoder);
        }

        // Execute GPU physics pipeline if not paused and has cells
        // Use fixed timestep accumulator for consistent physics behavior
        if !self.paused {
            let fixed_dt = self.config.fixed_timestep;
            // Allow more steps when time_scale > 1 (fast forward)
            let max_steps = (4.0 * self.time_scale).ceil() as i32;
            let mut steps = 0;

            while self.time_accumulator >= fixed_dt && steps < max_steps {
                self.run_physics(device, &mut encoder, queue, fixed_dt, world_diameter);

                // Wrap current_time to prevent f32 precision loss in long runs.
                // f32 loses sub-second precision above ~8M seconds. Wrapping at 65536s
                // keeps full precision. birth_times are also written from current_time
                // so cell_age = current_time - birth_time remains correct after the wrap
                // as long as no cell lives longer than 65536 sim-seconds (they don't).
                self.current_time = (self.current_time + fixed_dt) % 65536.0;
                self.time_accumulator -= fixed_dt;
                steps += 1;
            }

            // If we hit max steps, discard remaining accumulated time
            if steps >= max_steps {
                self.time_accumulator = 0.0;
            }
        } else if is_dragging {
            // Paused but dragging: run mechanics-only physics step so adhesion-connected
            // cells follow via spring forces. Skips nutrient transport, mass accumulation,
            // and lifecycle to preserve cell split timing and nutrient state.
            let fixed_dt = self.config.fixed_timestep;
            self.run_mechanics_only(device, &mut encoder, queue, fixed_dt, world_diameter);
        }

        // Execute non-drag pending position updates (e.g. from other tools)
        let position_updated = position_updated || self.execute_pending_position_updates(device, &mut encoder, queue);

        // Update death particles AFTER physics — spawn_new reads death_flags written
        // by lifecycle_unified.wgsl (death_scan) and positions from the current
        // triple-buffer slot (last completed frame, not the one being written).
        if self.show_death_particles && !self.paused {
            let dt = 1.0 / 60.0;
            self.update_death_particles(&mut encoder, device, queue, dt);
        }

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

        // Execute signal system ONCE PER FRAME (not per physics step)
        // Runs after physics/lifecycle so adhesion state is up-to-date
        // Skipped entirely if no genomes have oculocyte modes
        if !self.paused && self.has_oculocytes {
            execute_signal_system(
                &mut encoder,
                &self.gpu_physics_pipelines,
                &self.gpu_triple_buffers,
                &self.adhesion_buffers,
                &self.cached_bind_groups,
                self.has_oculocytes,
                self.total_cell_slots,
                self.max_signal_hops,
            );
        }

        // Copy buffers to instance builder after physics (always needed for division)
        // Division creates new cells with updated cell_types that must be copied before rendering
        // Also copy after position update, cell removal, or cell boost
        // Skip entirely when there are no cells — the copies are full-capacity DMA transfers
        // (positions, rotations, mode_indices, cell_ids, genome_ids, cell_types, mode_properties)
        // and with 0 cells the instance builder will produce 0 visible instances regardless.
        // IMPORTANT: also copy when cell_inserted is true — total_cell_slots lags 1-3 frames
        // behind the async readback, so the first cell would be invisible without this override.
        let has_cells = self.total_cell_slots > 0 || cell_inserted;

        // Tick follow camera BEFORE copy_buffers_to_instance_builder so the orbit
        // pivot and the rendered cell positions are derived from the same GPU buffer.
        if self.follow_organism_id.is_some() {
            self.tick_follow_camera(device, &mut encoder, 1.0 / 60.0);
        }

        if has_cells && (!self.paused || position_updated || cell_removed || cell_boosted || cell_inserted) {
            self.copy_buffers_to_instance_builder(&mut encoder);
        }

        // Build instances with GPU culling (compute pass)
        // Calculate total mode count across all genomes
        // Use capacity for dispatch - shader reads actual cell_count from GPU buffer
        let total_mode_count: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        
        // Mode visuals are updated in the genomes_dirty block above (not per-frame)
        // so GPU-mutated colors in mode_visuals_buffer are preserved.
        
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
            &self.gpu_triple_buffers.death_flags,
            self.lod_scale_factor,
            self.lod_threshold_low,
            self.lod_threshold_medium,
            self.lod_threshold_high,
            self.lod_debug_colors,
            // Use max(1) when a cell was just inserted so the instance builder
            // doesn't early-out before the async readback has caught up.
            self.total_cell_slots.max(if cell_inserted { 1 } else { 0 }),
        );

        // When DoF is enabled, render the scene to an intermediate texture so the
        // DoF shader can read it. Otherwise render directly to the swapchain.
        let dof_enabled = self.show_dof && self.dof_renderer.is_some();
        // Extract the DoF scene target view pointer before any mutable borrows.
        // SAFETY: The texture view lives as long as self.dof_renderer, which lives for
        // the entire render() call. We just need to avoid holding &self across &mut self.
        let dof_scene_view: Option<*const wgpu::TextureView> = if dof_enabled {
            Some(self.dof_renderer.as_ref().unwrap().scene_target_view() as *const _)
        } else {
            None
        };
        let scene_target: &wgpu::TextureView = if let Some(ptr) = dof_scene_view {
            // SAFETY: dof_renderer is not dropped or moved during render()
            unsafe { &*ptr }
        } else {
            view
        };

        // Clear pass
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU Scene Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_target,
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

        // Render procedural space skybox (before cells so geometry draws over it)
        if self.show_skybox {
            if let Some(ref skybox_renderer) = self.skybox_renderer {
                skybox_renderer.render(
                    &mut encoder,
                    queue,
                    scene_target,
                    device,
                    view_proj,
                    self.camera.position(),
                    self.current_time,
                );
            }
        }

        // Render cells using GPU-culled instance buffer
        let visible_count = self.instance_builder.visible_count();
        // Update shadow bind group for cell renderer — created once and cached.
        // The light field buffers never change after initialization, so there is no
        // need to recreate this bind group every frame.
        if !self.cell_shadow_bind_group_set {
            if let Some(ref light_field) = self.light_field_system {
                let shadow_bg = light_field.create_shadow_bind_group(device);
                self.renderer.set_shadow_bind_group(shadow_bg);
                self.cell_shadow_bind_group_set = true;
            }
        }
        self.renderer.render_with_encoder(
            &mut encoder,
            queue,
            scene_target,
            &self.instance_builder,
            visible_count,
            self.camera.position(),
            self.camera.rotation,
            self.current_time,
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
            outline_width,
        );

        // Render flagellocyte tails using GPU instance buffer
        // With dynamic instance allocation, flagellocytes are scattered throughout the buffer.
        // We use the main indirect buffer (total visible count) - the shader filters by cell_type
        // and outputs degenerate triangles for non-flagellocytes.
        self.tail_renderer.render_from_gpu_buffer(
            device,
            queue,
            &mut encoder,
            scene_target,
            &self.renderer.depth_view,
            self.instance_builder.get_instance_buffer(),
            self.instance_builder.get_indirect_buffer(),  // Use total visible count, not per-type
            self.camera.position(),
            self.camera.rotation,
            self.current_time,
            self.renderer.width,
            self.renderer.height,
            self.instance_builder.capacity(),
        );

        // Update light field time for caustic animation
        if let Some(ref mut light_field) = self.light_field_system {
            light_field.set_time(self.current_time);
        }

        // Render cave system if initialized (before sun so caves occlude it)
        if let Some(ref mut cave_renderer) = self.cave_renderer {
            // Update shadow bind group from light field system (with water bitfield for caustics)
            if let Some(ref light_field) = self.light_field_system {
                light_field.update_shadow_params(queue);
                // Create and set shadow bind group once (buffers never change, params updated via write_buffer)
                if !self.cave_shadow_bind_group_set {
                    let density_buf = self.gpu_surface_nets.as_ref()
                        .map(|sn| sn.smoothed_density_buffer())
                        .unwrap_or(light_field.dummy_water_density());
                    let moss_buf = self.moss_system.as_ref()
                        .map(|ms| ms.moss_density_buffer())
                        .unwrap_or(light_field.dummy_water_density());
                    let shadow_bg = light_field.create_cave_shadow_bind_group(device, density_buf, moss_buf);
                    cave_renderer.set_shadow_bind_group(shadow_bg);
                    self.cave_shadow_bind_group_set = true;
                }
            }
            cave_renderer.render(
                &mut encoder,
                queue,
                scene_target,
                &self.renderer.depth_view,
                self.camera.position(),
                self.camera.rotation,
            );
        }

        // Render boulders after cave (they share the same depth buffer)
        if self.show_boulders {
            if let Some(ref mut bs) = self.boulder_system {
                // Run bubble compute passes before rendering
                bs.update_bubbles(&mut encoder, queue, self.last_delta_time, self.gravity_mode);
                bs.render(
                    &mut encoder,
                    queue,
                    scene_target,
                    &self.renderer.depth_view,
                    self.camera.position(),
                    self.camera.rotation,
                    self.current_time,
                );
            }
        }
        
        // Render procedural sun if enabled (after caves but before world sphere,
        // so caves occlude the sun but the translucent world sphere doesn't)
        if self.show_sun {
            if let Some(ref sun_renderer) = self.sun_renderer {
                // Get light direction from light field system, or use default
                let light_dir = if let Some(ref light_field) = self.light_field_system {
                    light_field.light_dir()
                } else {
                    [0.5, 0.8, 0.3]
                };

                sun_renderer.render(
                    &mut encoder,
                    queue,
                    scene_target,
                    &self.renderer.depth_view,
                    device,
                    view_proj,
                    self.camera.position(),
                    self.current_time,
                    light_dir,
                    self.sun_intensity,
                );
            }
        }
        
        // Render world boundary sphere if enabled (after sun so sun shows through)
        if self.show_world_sphere {
            // Update world sphere radius to match current world diameter
            self.world_sphere_renderer.set_radius(queue, world_diameter * 0.5);
            
            self.world_sphere_renderer.render(
                &mut encoder,
                queue,
                scene_target,
                &self.renderer.depth_view,
                self.camera.position(),
                self.camera.rotation,
            );
        }

        // Render organism skins after all opaque geometry (cave, cells, world sphere)
        // so they are correctly occluded, but before transparent particles.
        // Adhesion lines render BEFORE the skin so the skin's alpha blending dims them.
        if self.show_adhesion_lines {
            self.ensure_cached_adhesion_bind_groups(device);
            let output_idx = self.gpu_triple_buffers.output_buffer_index();

            if let Some(ref cached_bgs) = self.cached_adhesion_data_bind_groups {
                let adhesion_data_bind_group = &cached_bgs[output_idx];

                let mut adhesion_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GPU Adhesion Lines Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: scene_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.renderer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
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
                    adhesion_data_bind_group,
                    self.camera.position(),
                    self.camera.rotation,
                );
            }
        }

        // Run shrink-wrap compute THEN render skin, so the mesh is current this frame.
        // Skin renders after adhesion lines so its alpha blending dims them.
        if self.show_organism_skins {
            self.ensure_organism_skin_bind_groups(device);
            let output_idx = self.gpu_triple_buffers.output_buffer_index();
            let max_cells  = self.current_cell_count;

            // Compute pass: update the shrink-wrap mesh
            if let (Some(ref renderer), Some(ref compute_bgs)) =
                (&self.organism_skin_renderer, &self.organism_skin_compute_bind_groups)
            {
                if max_cells > 0 {
                    renderer.encode_shrinkwrap_frame(
                        &mut encoder,
                        &compute_bgs[output_idx],
                        max_cells,
                    );
                }
            }

            // Render pass: draw the updated mesh over adhesion lines
            if let Some(ref renderer) = self.organism_skin_renderer {
                if self.current_cell_count > 0 {
                    renderer.set_time(queue, self.current_time);
                    renderer.render(
                        &mut encoder,
                        queue,
                        scene_target,
                        &self.renderer.depth_view,
                        self.camera.position(),
                        self.camera.rotation,
                    );
                }
            }
        }
        
        // Render fluid voxels if enabled (uses cached camera bind group)
        if self.show_fluid_voxels {
            if let (Some(ref voxel_renderer), Some(ref _fluid_buffers), Some(ref camera_bind_group)) = 
                (&self.voxel_renderer, &self.fluid_buffers, &self.env_camera_bind_group_voxel) {
                
                let mut voxel_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Voxel Rendering Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: scene_target,
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
                voxel_renderer.render(&mut voxel_pass, camera_bind_group, self.voxel_instance_count);
            }
        }
        
        // Capture reflection cubemap once (cave + sun into 6 faces)
        if !self.reflection_cubemap_captured
            && self.cave_renderer.is_some()
            && self.gpu_surface_nets.is_some()
        {
            self.capture_reflection_cubemap(device, queue);
            self.reflection_cubemap_captured = true;
            // Mark cubemap as generated so the fallback fill doesn't overwrite it
            if let Some(ref mut sn) = self.gpu_surface_nets {
                sn.mark_cubemap_generated();
            }
        }

        // Extract and render GPU density mesh if enabled
        if self.show_gpu_density_mesh {
            // Only re-extract the mesh when the fluid state has actually changed.
            // When the simulation is paused and there are no cells (nothing can disturb
            // the fluid), the density field is static — re-running extract_to_surface_nets,
            // smooth_density, and extract_mesh every frame wastes ~100K workgroups for
            // an identical result. We still render the last extracted mesh every frame.
            let fluid_changed = !self.paused || self.current_cell_count > 0;

            if fluid_changed {
                // First extract density from fluid simulator to surface nets buffers
                if let (Some(ref fluid_sim), Some(ref gpu_surface_nets)) = (&self.fluid_simulator, &self.gpu_surface_nets) {
                    fluid_sim.extract_to_surface_nets(
                        device,
                        &mut encoder,
                        queue,
                        gpu_surface_nets.density_buffer(),
                        gpu_surface_nets.fluid_type_buffer(),
                    );
                }

                if let Some(ref mut gpu_surface_nets) = self.gpu_surface_nets {
                    // Smooth the raw density: spatial blur + temporal blend
                    gpu_surface_nets.smooth_density(&mut encoder);
                    
                    // Run compute shaders to extract mesh from smoothed density buffer
                    gpu_surface_nets.extract_mesh(&mut encoder);
                }
            }

            if let Some(ref mut gpu_surface_nets) = self.gpu_surface_nets {
                if !self.water_shadow_bind_group_set {
                    if let Some(ref light_field) = self.light_field_system {
                        let shadow_bg = light_field.create_shadow_bind_group(device);
                        gpu_surface_nets.set_shadow_bind_group(shadow_bg);
                        self.water_shadow_bind_group_set = true;
                    }
                }

                // Ensure shadow params are up-to-date for this frame
                // (may already be updated by cave renderer, but safe to call again)
                if let Some(ref light_field) = self.light_field_system {
                    light_field.update_shadow_params(queue);
                }

                // Render the extracted mesh
                gpu_surface_nets.render(
                    &mut encoder,
                    queue,
                    scene_target,
                    &self.renderer.depth_view,
                    self.camera.position(),
                    self.camera.rotation,
                );
            }
        }

        // Render steam particles if enabled (uses cached camera bind group)
        if self.show_steam_particles {
            if let (Some(ref steam_particle_renderer), Some(ref render_bind_group), Some(ref camera_bind_group)) =
                (&self.steam_particle_renderer, &self.steam_render_bind_group, &self.env_camera_bind_group_steam) {

                steam_particle_renderer.render(
                    &mut encoder,
                    scene_target,
                    &self.renderer.depth_view,
                    camera_bind_group,
                    render_bind_group,
                );
            }
        }

        // Render water particles if enabled (uses cached camera bind group)
        if self.show_water_particles {
            if let (Some(ref water_particle_renderer), Some(ref render_bind_group), Some(ref camera_bind_group)) =
                (&self.water_particle_renderer, &self.water_render_bind_group, &self.env_camera_bind_group_water) {

                water_particle_renderer.render(
                    &mut encoder,
                    scene_target,
                    &self.renderer.depth_view,
                    camera_bind_group,
                    render_bind_group,
                );
            }
        }

        // Render nutrient particles if enabled (uses cached camera bind group)
        if self.show_nutrient_particles {
            if let (Some(ref nutrient_particle_renderer), Some(ref render_bind_group), Some(ref camera_bind_group)) =
                (&self.nutrient_particle_renderer, &self.nutrient_render_bind_group, &self.env_camera_bind_group_nutrient) {

                nutrient_particle_renderer.render(
                    &mut encoder,
                    scene_target,
                    &self.renderer.depth_view,
                    camera_bind_group,
                    render_bind_group,
                );
            }
        }

        // Render death particles if enabled (uses cached camera bind group)
        if self.show_death_particles {
            if let (Some(ref death_particle_renderer), Some(ref render_bind_group), Some(ref camera_bind_group)) =
                (&self.death_particle_renderer, &self.death_render_bind_group, &self.env_camera_bind_group_death) {

                death_particle_renderer.render(
                    &mut encoder,
                    scene_target,
                    &self.renderer.depth_view,
                    camera_bind_group,
                    render_bind_group,
                );
            }
        }


        // Render volumetric fog if enabled (post-process over scene, half-res + composite)
        if self.show_volumetric_fog {
            if let (Some(ref light_field), true) =
                (&self.light_field_system, self.volumetric_fog_renderer.is_some())
            {
                let lf_buffer = light_field.light_field_buffer();
                let lf_dir = light_field.light_dir();
                let lf_cell_size = light_field.cell_size();
                let lf_origin = light_field.grid_origin();
                let lf_radius = light_field.world_radius();
                let depth_view = &self.renderer.depth_view;
                let cam_pos = self.camera.position();
                let time = self.current_time;
                self.volumetric_fog_renderer.as_mut().unwrap().render(
                    &mut encoder,
                    queue,
                    scene_target,
                    depth_view,
                    device,
                    lf_buffer,
                    view_proj,
                    cam_pos,
                    time,
                    lf_dir,
                    128, // GRID_RESOLUTION
                    lf_cell_size,
                    lf_origin,
                    lf_radius,
                );
            }
        }

        // Render depth of field if enabled (post-process: reads scene_target, writes to swapchain)
        if dof_enabled {
            let depth_view = &self.renderer.depth_view;
            let cam_pos = self.camera.position();
            self.dof_renderer.as_mut().unwrap().render(
                &mut encoder,
                queue,
                view,
                depth_view,
                device,
                view_proj,
                cam_pos,
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

        // Call map_async on follow camera staging buffers NOW — after submit.
        // map_async must not be called while the buffer is referenced by a pending
        // encoder (wgpu validation error: "buffer is still mapped").
        if self.follow_organism_id.is_some() {
            self.tick_follow_camera_post_submit();
        }

        // Moss growth runs in a SEPARATE submit after the main frame.
        // This prevents it from competing with rendering on the GPU timeline —
        // the driver can schedule it in background while the CPU prepares the next frame.
        // The 1-frame lag on moss_density is invisible for a slowly-changing field.
        if !self.paused && self.show_moss {
            let mut moss_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Moss Growth Encoder"),
            });
            let dt = self.config.fixed_timestep;
            self.run_moss_growth(device, &mut moss_encoder, queue, dt);
            queue.submit(std::iter::once(moss_encoder.finish()));
        }

        // Debug: poll label buffer readback.
        if let Some(label_system) = &mut self.organism_label_system {
            label_system.poll_debug_readback(device);
        }

        // Initiate the async map operation after submit (if we started a readback)
        if should_start_readback {
            self.gpu_triple_buffers.initiate_cell_count_map();
        }
        
        // Poll for cell count readback completion and update current_cell_count
        if let Some((total, live)) = self.gpu_triple_buffers.poll_cell_count(device) {
            self.current_cell_count = live;
            self.total_cell_slots = total;
            
            // CRITICAL FIX: Reset high water mark when all cells are dead.
            // cell_count_buffer[0] is a high water mark that never decreases when cells die.
            // This causes all GPU shaders to iterate through slots 0-N even when live count is 0,
            // resulting in massive GPU work for no benefit. Resetting to 0 when live=0 ensures
            // shaders early-exit immediately, eliminating the frame drop after mass starvation.
            if live == 0 && total > 0 {
                log::info!("All cells dead - resetting high water mark from {} to 0", total);
                queue.write_buffer(&self.gpu_triple_buffers.cell_count_buffer, 0, bytemuck::cast_slice(&[0u32, 0u32]));
            }
        }

        // Poll for steam particle count (GPU readback)
        if self.show_steam_particles {
            self.poll_steam_particle_count(device);
        }

        // Poll for organism skin skinned cell count (GPU readback)
        if self.show_organism_skins {
            if let Some(ref mut renderer) = self.organism_skin_renderer {
                renderer.try_read_skinned_count(device);
            }
        }

        // Poll for water particle count (GPU readback)
        if self.show_water_particles {
            self.poll_water_particle_count(device);
        }

        // Poll for nutrient particle count (GPU readback)
        if self.show_nutrient_particles {
            self.poll_nutrient_particle_count(device);
        }

        // Poll for death particle count (GPU readback)
        if self.show_death_particles {
            self.poll_death_particle_count(device);
        }

        // Mark that we now have Hi-Z data for next frame
        self.first_frame = false;
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.adhesion_renderer.resize(width, height);
        self.world_sphere_renderer.resize(width, height);
        self.tail_renderer.resize(width, height);
        self.first_frame = true;
        
        // Resize cave renderer if initialized
        if let Some(ref mut cave_renderer) = self.cave_renderer {
            cave_renderer.resize(width, height);
        }
        
        // Resize steam particle renderer if initialized
        if let Some(ref mut steam_particle_renderer) = self.steam_particle_renderer {
            steam_particle_renderer.resize(width, height);
        }
        
        // Resize sun renderer if initialized
        if let Some(ref mut sun_renderer) = self.sun_renderer {
            sun_renderer.resize(width, height);
        }
        
        // Resize volumetric fog renderer (recreates half-res texture)
        if let Some(ref mut fog_renderer) = self.volumetric_fog_renderer {
            fog_renderer.resize(device, width, height);
        }
        
        // Resize depth of field renderer (recreates intermediate textures)
        if let Some(ref mut dof_renderer) = self.dof_renderer {
            dof_renderer.resize(device, width, height, self.surface_format);
        }
        
        // Resize GPU surface nets (recreates OIT textures)
        if let Some(ref mut gpu_surface_nets) = self.gpu_surface_nets {
            gpu_surface_nets.resize(device, width, height);
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
