//! GPU Compute Pipelines for Physics Simulation
//!
//! Contains the compute pipelines that make up the GPU physics pipeline.
//!
//! ## Bind Group Layouts
//!
//! ### Physics Bind Group (Group 0)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Uniform | physics_params |

use super::GpuTripleBufferSystem;
use wgpu::util::DeviceExt;

/// Cell insertion parameters for GPU cell insertion compute shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellInsertionParams {
    // Cell position and physics properties (16 bytes)
    pub position: [f32; 3],
    pub mass: f32,

    // Cell velocity (16 bytes)
    pub velocity: [f32; 3],
    pub _pad0: f32,

    // Cell rotation quaternion (16 bytes)
    pub rotation: [f32; 4],

    // Cell genome and mode info (16 bytes)
    pub genome_id: u32,
    pub mode_index: u32,
    pub birth_time: f32,
    pub _pad1: f32,

    // Cell division properties (16 bytes)
    pub split_interval: f32,
    pub split_mass: f32,
    pub stiffness: f32,
    pub radius: f32,

    // Cell state properties (16 bytes)
    pub nutrient_gain_rate: f32,
    pub max_cell_size: f32,
    pub max_splits: u32,
    pub cell_id: u32,

    // Cell type + initial reserve + initial nutrients (16 bytes)
    pub cell_type: u32,
    /// Initial embryocyte reserve (x1000 fixed-point). 0 for normal cells.
    /// For Embryocytes created from normal insertion: set to 65535000 (full).
    /// For Embryocytes spawned from gamete merge: set to combined gamete reserve.
    pub initial_reserve: u32,
    /// Initial nutrients (x1000 fixed-point). 0 = use default (100000 = full).
    /// Pass a non-zero value to cap starting nutrients (e.g. gamete merge where
    /// the combined reserve doesn't cover a full nutrient pool).
    pub initial_nutrients: u32,
    pub _pad4: u32,
}

/// Spatial query parameters for GPU spatial query compute shader
/// Size: 32 bytes (must match WGSL struct layout)
/// Uses ray origin and direction for ray-sphere intersection testing
#[repr(C, align(4))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialQueryParams {
    pub ray_origin: [f32; 3],    // 12 bytes - camera position
    pub max_distance: f32,       // 4 bytes - maximum ray distance
    pub ray_direction: [f32; 3], // 12 bytes - normalized ray direction
    pub _pad0: u32,              // 4 bytes - padding
} // Total: 32 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(
    std::mem::size_of::<SpatialQueryParams>() == 32,
    "SpatialQueryParams must be 32 bytes"
);

/// Spatial query result from GPU spatial query compute shader
/// Size: 16 bytes (must match WGSL struct layout)
/// Note: distance_fixed is stored as fixed-point u32 (multiply by 1000) for atomic operations
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialQueryResult {
    pub found_cell_index: u32, // 4 bytes
    pub distance_fixed: u32,   // 4 bytes - fixed-point distance (actual = distance_fixed / 1000.0)
    pub found: u32,            // 4 bytes
    pub _padding: u32,         // 4 bytes
} // Total: 16 bytes

impl SpatialQueryResult {
    /// Get the distance as f32 (converts from fixed-point)
    pub fn distance(&self) -> f32 {
        self.distance_fixed as f32 / 1000.0
    }
}

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(
    std::mem::size_of::<SpatialQueryResult>() == 16,
    "SpatialQueryResult must be 16 bytes"
);

/// Position update parameters for GPU position update compute shader
/// Size: 32 bytes (must match WGSL struct layout with vec3 alignment)
/// WGSL vec3<f32> has 16-byte alignment, so there's padding after cell_index
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PositionUpdateParams {
    pub cell_index: u32,        // 4 bytes at offset 0
    pub _pad0: u32,             // 4 bytes at offset 4 (padding for vec3 alignment)
    pub _pad1: u32,             // 4 bytes at offset 8
    pub _pad2: u32,             // 4 bytes at offset 12
    pub new_position: [f32; 3], // 12 bytes at offset 16
    pub _padding: f32,          // 4 bytes at offset 28
} // Total: 32 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(
    std::mem::size_of::<PositionUpdateParams>() == 32,
    "PositionUpdateParams must be 32 bytes"
);

/// Cell removal parameters for GPU cell removal compute shader
/// Size: 16 bytes (must match WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellRemovalParams {
    pub cell_index: u32, // 4 bytes at offset 0
    pub _pad0: u32,      // 4 bytes at offset 4
    pub _pad1: u32,      // 4 bytes at offset 8
    pub _pad2: u32,      // 4 bytes at offset 12
} // Total: 16 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(
    std::mem::size_of::<CellRemovalParams>() == 16,
    "CellRemovalParams must be 16 bytes"
);

/// Cell boost parameters for GPU cell boost compute shader
/// Size: 16 bytes (must match WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellBoostParams {
    pub cell_index: u32, // 4 bytes at offset 0
    pub _pad0: u32,      // 4 bytes at offset 4
    pub _pad1: u32,      // 4 bytes at offset 8
    pub _pad2: u32,      // 4 bytes at offset 12
} // Total: 16 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(
    std::mem::size_of::<CellBoostParams>() == 16,
    "CellBoostParams must be 16 bytes"
);

/// Cached bind groups for GPU physics pipeline
/// Pre-created for all 3 buffer indices to avoid per-frame allocation
pub struct CachedBindGroups {
    /// Physics bind groups for each buffer index [0, 1, 2]
    pub physics: [wgpu::BindGroup; 3],
    /// Spatial grid bind group (same for all frames)
    pub spatial_grid: wgpu::BindGroup,
    /// Position update spatial grid bind group (read-only, same for all frames)
    pub position_update_spatial_grid: wgpu::BindGroup,
    /// Lifecycle bind group (same for all frames)
    pub lifecycle: wgpu::BindGroup,
    /// Cell state read bind group (same for all frames)
    pub cell_state_read: wgpu::BindGroup,
    /// Cell state write bind groups for each buffer index [0, 1, 2]
    pub cell_state_write: [wgpu::BindGroup; 3],
    /// Mass accumulation bind group (same for all frames)
    pub mass_accum: wgpu::BindGroup,
    /// Rotations bind groups for each buffer index [0, 1, 2] (for adhesion physics)
    pub rotations: [wgpu::BindGroup; 3],
    /// Position update rotations bind groups for each buffer index [0, 1, 2]
    pub position_update_rotations: [wgpu::BindGroup; 3],
    /// Adhesion bind group (same for all frames)
    pub adhesion: wgpu::BindGroup,
    /// Force accumulation bind group (same for all frames)
    pub force_accum: wgpu::BindGroup,
    /// Lifecycle adhesion bind group (same for all frames) - for adhesion_cleanup
    pub lifecycle_adhesion: wgpu::BindGroup,
    /// Division execute adhesion bind group (same for all frames) - for division shader
    pub division_execute_adhesion: wgpu::BindGroup,
    /// Division scan adhesion bind group (read-only, same for all frames)
    pub division_scan_adhesion: wgpu::BindGroup,
    /// Clear forces bind group (same for all frames)
    pub clear_forces: wgpu::BindGroup,
    /// Collision force accum bind groups for each buffer index [0, 1, 2]
    pub collision_force_accum: [wgpu::BindGroup; 3],
    /// Position update force accum bind groups for each buffer index [0, 1, 2]
    pub position_update_force_accum: [wgpu::BindGroup; 3],
    /// Velocity update angular bind groups for each buffer index [0, 1, 2]
    /// Each reads angular state from `buffer_index` and writes the frame output
    /// state to `(buffer_index + 1) % 3`.
    pub velocity_update_angular: [wgpu::BindGroup; 3],
    /// Nutrient system bind group (same for all frames)
    pub nutrient_system: wgpu::BindGroup,
    /// Nutrient transport bind group (same for all frames, includes mode properties)
    pub nutrient_transport: wgpu::BindGroup,
    /// Nutrient apply bind group (same for all frames) - applies accumulated mass deltas
    pub nutrient_apply: wgpu::BindGroup,
    /// Swim force force accumulation bind groups for each buffer index [0, 1, 2]
    pub swim_force_force_accum: [wgpu::BindGroup; 3],
    /// Swim force cell data bind group (same for all frames)
    pub swim_force_cell_data: wgpu::BindGroup,
    /// Glueocyte env adhesion force+anchor bind groups for each buffer index [0, 1, 2]
    pub env_adhesion_force_accum: [wgpu::BindGroup; 3],
    /// Glueocyte env adhesion mode data bind group (same for all frames)
    pub env_adhesion_mode_data: wgpu::BindGroup,
    /// Dummy cave collision bind group - used when glueocyte env adhesion runs without a cave
    /// (e.g. boulder-only adhesion). Contains zeroed CaveParams with collision_enabled=0.
    pub dummy_cave_collision: wgpu::BindGroup,

    // Glueocyte cell-to-cell adhesion bind groups
    /// Adhesion buffers bind group (connections, indices, next_id, free_slots, counts)
    pub cell_adhesion_adhesion: wgpu::BindGroup,
    /// Spatial grid bind group (read-only view for neighbour queries)
    pub cell_adhesion_spatial: wgpu::BindGroup,
    /// Mode/signal data bind group (mode_indices, mode_cell_types, cell_adhesion_flags, signal_flags, rotations, genome_orientations, death_flags)
    pub cell_adhesion_mode: wgpu::BindGroup,

    // Cilia force bind groups
    /// Cilia force force accumulation bind groups for each buffer index [0, 1, 2]
    pub cilia_force_force_accum: [wgpu::BindGroup; 3],
    /// Cilia force cell data bind group (same for all frames)
    pub cilia_force_cell_data: wgpu::BindGroup,
    /// Cilia force spatial bind group (same for all frames)
    pub cilia_force_spatial: wgpu::BindGroup,

    // Muscle contraction bind groups
    /// Muscle contraction group 0: physics params + cell_count_buffer
    pub muscle_contraction_group0: wgpu::BindGroup,
    /// Muscle contraction group 1: mode_indices, mode_cell_types, type_behaviors, signal_flags, v7, v8
    pub muscle_contraction_group1: wgpu::BindGroup,
    /// Muscle contraction group 2: adhesion_connections, adhesion_settings_v0, adhesion_settings_v0_original, cell_adhesion_indices, adhesion_counts
    pub muscle_contraction_group2: wgpu::BindGroup,

    /// Hidden cell physiology buffers: water, heat, cached temperature, thermal state.
    pub physiology: wgpu::BindGroup,
    pub physiology_cell_data: wgpu::BindGroup,
    pub physiology_transport: wgpu::BindGroup,

    // Signal system bind groups
    /// Signal flags bind group (Group 0 for signal_clear and signal_sense)
    pub signal_flags: wgpu::BindGroup,
    /// Signal propagate flags bind group (Group 0 for signal_propagate):
    /// binding 0 = signal_flags (read), binding 1 = cell_count (read), binding 2 = signal_flags_next (read_write)
    pub signal_propagate_flags: wgpu::BindGroup,
    /// Signal sense cell data bind groups for each buffer index [0, 1, 2]
    pub signal_sense_cell_data: [wgpu::BindGroup; 3],
    /// Signal sense world data bind group (Group 2: world params + fluid state)
    pub signal_sense_world_data: wgpu::BindGroup,
    /// Signal propagate adhesion bind group (same for all frames)
    pub signal_propagate_adhesion: wgpu::BindGroup,
    /// Mode switch bind groups (Group 0: cell state, Group 1: signal data, Group 2: per-mode props)
    pub mode_switch_group0: wgpu::BindGroup,
    pub mode_switch_group1: wgpu::BindGroup,
    pub mode_switch_group2: wgpu::BindGroup,

    // Boulder bind groups
    /// Boulder physics group 1: boulder_state, boulder_moss, boulder_moss_dir, boulder_eat_dir, boulder_count
    pub boulder_physics_buffers: wgpu::BindGroup,
    /// Boulder consume group 0: minimal params uniform (delta_time, world_size, grid params)
    pub boulder_consume_params: wgpu::BindGroup,
    /// Boulder consume group 1: spatial grid (read-only)
    pub boulder_consume_spatial: wgpu::BindGroup,
    /// Boulder consume group 2: cell data (positions, types, nutrients, org_size, death_flags, thresholds, cell_count)
    /// One per buffer index - positions rotate each frame.
    pub boulder_consume_cell_data: [wgpu::BindGroup; 3],
    /// Boulder consume group 3: boulder buffers (state, moss, eat_dir, count)
    pub boulder_consume_buffers: wgpu::BindGroup,
    /// Dummy cave params bind group for boulder physics when no cave exists.
    /// Has collision_enabled = 0 so the SDF code is skipped.
    pub boulder_dummy_cave: wgpu::BindGroup,
}

/// GPU physics compute pipelines
pub struct GpuPhysicsPipelines {
    pub spatial_grid_clear: wgpu::ComputePipeline,
    pub spatial_grid_assign: wgpu::ComputePipeline,
    pub spatial_grid_insert: wgpu::ComputePipeline,
    pub spatial_grid_build: wgpu::ComputePipeline,
    pub clear_forces: wgpu::ComputePipeline,
    pub collision_detection: wgpu::ComputePipeline,
    pub position_update: wgpu::ComputePipeline,
    pub velocity_update: wgpu::ComputePipeline,
    pub mass_accum: wgpu::ComputePipeline,

    // Cell insertion pipeline
    pub cell_insertion: wgpu::ComputePipeline,

    // Cell data extraction pipeline
    pub cell_data_extraction: wgpu::ComputePipeline,

    // Spatial query pipeline
    pub spatial_query: wgpu::ComputePipeline,

    // Position update pipeline
    pub position_update_tool: wgpu::ComputePipeline,

    // Cell removal pipeline
    pub cell_removal: wgpu::ComputePipeline,

    // Cell boost pipeline
    pub cell_boost: wgpu::ComputePipeline,

    // Nutrient system pipelines
    pub nutrient_transport: wgpu::ComputePipeline,
    pub nutrient_apply: wgpu::ComputePipeline,

    // Adhesion physics pipeline
    pub adhesion_physics: wgpu::ComputePipeline,

    // Adhesion constraint sub-step pipeline (iterative stiffening, no force accumulators)
    pub adhesion_substep: wgpu::ComputePipeline,

    // Lifecycle pipelines (3-stage with ring buffer for slot allocation)
    // Stage 1: Death scan - detects dead cells and pushes slots to ring buffer
    pub lifecycle_death_scan: wgpu::ComputePipeline,
    // Stage 2: Division scan - allocates slots from ring buffer for dividing cells
    pub lifecycle_division_scan: wgpu::ComputePipeline,
    // Stage 3: Division execute - creates child cells in allocated slots
    pub lifecycle_division_execute: wgpu::ComputePipeline,

    // Adhesion cleanup pipeline (runs after death scan)
    pub adhesion_cleanup: wgpu::ComputePipeline,

    // Swim force pipeline (applies thrust for Flagellocyte cells)
    pub swim_force: wgpu::ComputePipeline,

    // Plumocyte final angular damping pipeline
    pub plumocyte_rotation_damping: wgpu::ComputePipeline,

    // Buoyancy force pipeline (applies upward force for Buoyocyte cells)
    pub buoyancy_force: wgpu::ComputePipeline,

    // Glueocyte environment adhesion pipeline
    pub glueocyte_env_adhesion: wgpu::ComputePipeline,

    // Glueocyte cell-to-cell adhesion pipelines (bond_create + bond_release entry points)
    pub glueocyte_cell_adhesion_create: wgpu::ComputePipeline,
    pub glueocyte_cell_adhesion_release: wgpu::ComputePipeline,

    // Cilia force pipeline (applies contact-dependent surface propulsion for Ciliocyte cells)
    pub cilia_force: wgpu::ComputePipeline,

    // Muscle contraction pipeline (modifies adhesion rest lengths for Myocyte cells)
    pub muscle_contraction: wgpu::ComputePipeline,

    // Slow per-cell water/heat physiology pipeline
    pub physiology_transport: wgpu::ComputePipeline,
    pub physiology_update: wgpu::ComputePipeline,

    // Boulder pipelines
    pub boulder_physics: wgpu::ComputePipeline,
    pub boulder_consume: wgpu::ComputePipeline,

    // Boulder bind group layouts
    pub boulder_physics_buffers_layout: wgpu::BindGroupLayout,
    pub boulder_consume_params_layout: wgpu::BindGroupLayout,
    pub boulder_consume_spatial_layout: wgpu::BindGroupLayout,
    pub boulder_consume_cell_data_layout: wgpu::BindGroupLayout,
    pub boulder_consume_buffers_layout: wgpu::BindGroupLayout,

    /// Uniform buffer for boulder consume params (updated each frame).
    pub boulder_consume_params_buffer: wgpu::Buffer,

    // Bind group layouts
    pub physics_layout: wgpu::BindGroupLayout,
    pub spatial_grid_layout: wgpu::BindGroupLayout,
    pub lifecycle_layout: wgpu::BindGroupLayout,
    pub cell_state_read_layout: wgpu::BindGroupLayout,
    pub cell_state_write_layout: wgpu::BindGroupLayout,
    pub mass_accum_layout: wgpu::BindGroupLayout,
    pub rotations_layout: wgpu::BindGroupLayout,
    pub position_update_rotations_layout: wgpu::BindGroupLayout,
    pub position_update_spatial_grid_layout: wgpu::BindGroupLayout,

    // Cell insertion bind group layouts
    pub cell_insertion_physics_layout: wgpu::BindGroupLayout,
    pub cell_insertion_params_layout: wgpu::BindGroupLayout,
    pub cell_insertion_state_layout: wgpu::BindGroupLayout,

    // Cell data extraction bind group layouts
    pub cell_extraction_params_layout: wgpu::BindGroupLayout,
    pub cell_extraction_state_layout: wgpu::BindGroupLayout,
    pub cell_extraction_output_layout: wgpu::BindGroupLayout,

    // Spatial query bind group layouts
    pub spatial_query_params_layout: wgpu::BindGroupLayout,
    pub spatial_query_result_layout: wgpu::BindGroupLayout,

    // Position update bind group layouts
    pub position_update_params_layout: wgpu::BindGroupLayout,

    // Cell removal bind group layouts
    pub cell_removal_params_layout: wgpu::BindGroupLayout,

    // Cell boost bind group layouts
    pub cell_boost_params_layout: wgpu::BindGroupLayout,

    // Adhesion bind group layouts
    pub adhesion_layout: wgpu::BindGroupLayout,
    pub force_accum_layout: wgpu::BindGroupLayout,
    pub lifecycle_adhesion_layout: wgpu::BindGroupLayout,
    pub division_execute_adhesion_layout: wgpu::BindGroupLayout,
    pub division_scan_adhesion_layout: wgpu::BindGroupLayout,

    // Clear forces bind group layout
    pub clear_forces_layout: wgpu::BindGroupLayout,

    // Collision force accum bind group layout (for collision_detection group 2)
    pub collision_force_accum_layout: wgpu::BindGroupLayout,

    // Position update force accum bind group layout (for position_update group 2)
    pub position_update_force_accum_layout: wgpu::BindGroupLayout,

    // Velocity update angular bind group layout (for velocity_update group 1)
    pub velocity_update_angular_layout: wgpu::BindGroupLayout,

    // Nutrient transport bind group layouts
    pub nutrient_system_layout: wgpu::BindGroupLayout,
    pub nutrient_transport_layout: wgpu::BindGroupLayout,
    pub nutrient_apply_layout: wgpu::BindGroupLayout,

    // Swim force bind group layouts
    pub swim_force_force_accum_layout: wgpu::BindGroupLayout,
    pub swim_force_cell_data_layout: wgpu::BindGroupLayout,

    // Cilia force bind group layouts
    pub cilia_force_cell_data_layout: wgpu::BindGroupLayout,
    pub cilia_force_spatial_layout: wgpu::BindGroupLayout,

    // Muscle contraction bind group layouts
    pub muscle_contraction_group0_layout: wgpu::BindGroupLayout,
    pub muscle_contraction_group1_layout: wgpu::BindGroupLayout,
    pub muscle_contraction_group2_layout: wgpu::BindGroupLayout,
    pub physiology_layout: wgpu::BindGroupLayout,
    pub physiology_cell_data_layout: wgpu::BindGroupLayout,
    pub physiology_transport_layout: wgpu::BindGroupLayout,

    // Glueocyte env adhesion bind group layouts
    pub env_adhesion_force_accum_layout: wgpu::BindGroupLayout,
    pub env_adhesion_mode_data_layout: wgpu::BindGroupLayout,

    // Glueocyte cell adhesion bind group layouts
    pub cell_adhesion_adhesion_layout: wgpu::BindGroupLayout,
    pub cell_adhesion_spatial_layout: wgpu::BindGroupLayout,
    pub cell_adhesion_mode_layout: wgpu::BindGroupLayout,

    // Cave params bind group layout (uniform buffer at binding 0) - matches CaveSystemRenderer's collision_layout
    pub cave_params_layout: wgpu::BindGroupLayout,

    // Signal system pipelines
    pub signal_clear: wgpu::ComputePipeline,
    pub signal_sense: wgpu::ComputePipeline,
    pub signal_propagate: wgpu::ComputePipeline,
    pub signal_propagate_reverse: wgpu::ComputePipeline,
    pub signal_combine_sweeps: wgpu::ComputePipeline,
    pub mode_switch: wgpu::ComputePipeline,

    // Signal system bind group layouts
    /// Group 0 layout for signal_clear and signal_sense: binding 0 = signal_flags (read_write), binding 1 = cell_count (read)
    pub signal_flags_layout: wgpu::BindGroupLayout,
    /// Group 0 layout for signal_propagate: binding 0 = signal_flags (read), binding 1 = cell_count (read), binding 2 = signal_flags_next (read_write)
    pub signal_propagate_flags_layout: wgpu::BindGroupLayout,
    pub signal_sense_cell_data_layout: wgpu::BindGroupLayout,
    pub signal_sense_world_data_layout: wgpu::BindGroupLayout,
    pub signal_propagate_adhesion_layout: wgpu::BindGroupLayout,
    pub mode_switch_layout0: wgpu::BindGroupLayout,
    pub mode_switch_layout1: wgpu::BindGroupLayout,
    pub mode_switch_layout2: wgpu::BindGroupLayout,
}

impl GpuPhysicsPipelines {
    /// Create all compute pipelines
    pub fn new(device: &wgpu::Device) -> Self {
        // Create bind group layouts
        let physics_layout = Self::create_physics_bind_group_layout(device);
        let spatial_grid_layout = Self::create_spatial_grid_bind_group_layout(device);
        let lifecycle_layout = Self::create_lifecycle_bind_group_layout(device);
        let cell_state_read_layout = Self::create_cell_state_bind_group_layout(device, true);
        let cell_state_write_layout = Self::create_cell_state_bind_group_layout(device, false);
        let mass_accum_layout = Self::create_mass_accum_bind_group_layout(device);
        let rotations_layout = Self::create_rotations_bind_group_layout(device);
        let position_update_rotations_layout =
            Self::create_position_update_rotations_bind_group_layout(device);

        // Create spatial grid layout for position update sweep tests
        let position_update_spatial_grid_layout =
            Self::create_position_update_spatial_grid_bind_group_layout(device);

        // Create adhesion bind group layouts
        let adhesion_layout = Self::create_adhesion_bind_group_layout(device);
        let force_accum_layout = Self::create_force_accum_bind_group_layout(device);
        let lifecycle_adhesion_layout = Self::create_lifecycle_adhesion_bind_group_layout(device);
        let division_execute_adhesion_layout =
            Self::create_division_execute_adhesion_bind_group_layout(device);
        let division_scan_adhesion_layout =
            Self::create_division_scan_adhesion_bind_group_layout(device);

        // Create new bind group layouts for force accumulation pipeline
        let clear_forces_layout = Self::create_clear_forces_bind_group_layout(device);
        let collision_force_accum_layout =
            Self::create_collision_force_accum_bind_group_layout(device);
        let position_update_force_accum_layout =
            Self::create_position_update_force_accum_bind_group_layout(device);
        let velocity_update_angular_layout =
            Self::create_velocity_update_angular_bind_group_layout(device);

        // Create nutrient transport bind group layouts
        let nutrient_system_layout = Self::create_nutrient_system_bind_group_layout(device);
        let nutrient_transport_layout = Self::create_nutrient_transport_bind_group_layout(device);
        let nutrient_apply_layout = Self::create_nutrient_apply_bind_group_layout(device);

        // Create swim force bind group layouts
        let swim_force_force_accum_layout =
            Self::create_swim_force_force_accum_bind_group_layout(device);
        let swim_force_cell_data_layout =
            Self::create_swim_force_cell_data_bind_group_layout(device);

        // Create cilia force bind group layouts
        let cilia_force_cell_data_layout =
            Self::create_cilia_force_cell_data_bind_group_layout(device);
        let cilia_force_spatial_layout = Self::create_cilia_force_spatial_bind_group_layout(device);
        let physiology_layout = Self::create_physiology_bind_group_layout(device);
        let physiology_cell_data_layout =
            Self::create_physiology_cell_data_bind_group_layout(device);
        let physiology_transport_layout =
            Self::create_physiology_transport_bind_group_layout(device);

        // Create glueocyte env adhesion bind group layouts
        let env_adhesion_force_accum_layout =
            Self::create_env_adhesion_force_accum_bind_group_layout(device);
        let env_adhesion_mode_data_layout =
            Self::create_env_adhesion_mode_data_bind_group_layout(device);

        // Create glueocyte cell adhesion bind group layouts
        let cell_adhesion_adhesion_layout =
            Self::create_cell_adhesion_adhesion_bind_group_layout(device);
        let cell_adhesion_spatial_layout =
            Self::create_cell_adhesion_spatial_bind_group_layout(device);
        let cell_adhesion_mode_layout = Self::create_cell_adhesion_mode_bind_group_layout(device);

        // Cave params bind group layout: binding 0 = params, binding 1 = optional solid mask.
        // Matches CaveSystemRenderer's collision_layout exactly
        let cave_params_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cave Params Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create cell insertion bind group layouts
        let cell_insertion_physics_layout =
            Self::create_cell_insertion_physics_bind_group_layout(device);
        let cell_insertion_params_layout =
            Self::create_cell_insertion_params_bind_group_layout(device);
        let cell_insertion_state_layout =
            Self::create_cell_insertion_state_bind_group_layout(device);

        // Create cell data extraction bind group layouts
        let cell_extraction_params_layout =
            Self::create_cell_extraction_params_bind_group_layout(device);
        let cell_extraction_state_layout =
            Self::create_cell_extraction_state_bind_group_layout(device);
        let cell_extraction_output_layout =
            Self::create_cell_extraction_output_bind_group_layout(device);

        // Create spatial query bind group layouts
        let spatial_query_params_layout =
            Self::create_spatial_query_params_bind_group_layout(device);
        let spatial_query_result_layout =
            Self::create_spatial_query_result_bind_group_layout(device);

        // Create position update bind group layouts
        let position_update_params_layout =
            Self::create_position_update_params_bind_group_layout(device);

        // Create cell removal bind group layouts
        let cell_removal_params_layout = Self::create_cell_removal_params_bind_group_layout(device);

        // Create cell boost bind group layouts
        let cell_boost_params_layout = Self::create_cell_boost_params_bind_group_layout(device);

        // Create compute pipelines
        let spatial_grid_clear = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_clear.wgsl"),
            "main",
            &[&spatial_grid_layout],
            "Spatial Grid Clear",
        );

        let spatial_grid_assign = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_assign.wgsl"),
            "main",
            &[&physics_layout, &spatial_grid_layout],
            "Spatial Grid Assign",
        );

        let spatial_grid_insert = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_insert.wgsl"),
            "main",
            &[&physics_layout, &spatial_grid_layout],
            "Spatial Grid Insert",
        );

        // Combined spatial grid build (assign + insert + dead cell skip in one dispatch)
        let spatial_grid_build = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_grid_build.wgsl"),
            "main",
            &[&physics_layout, &spatial_grid_layout],
            "Spatial Grid Build",
        );

        // Clear forces pipeline (runs before collision and adhesion)
        let clear_forces = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/clear_forces.wgsl"),
            "main",
            &[&physics_layout, &clear_forces_layout],
            "Clear Forces",
        );

        let collision_detection = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/collision_detection.wgsl"),
            "main",
            &[
                &physics_layout,
                &spatial_grid_layout,
                &collision_force_accum_layout,
            ],
            "Collision Detection",
        );

        let position_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/position_update.wgsl"),
            "main",
            &[
                &physics_layout,
                &position_update_rotations_layout,
                &position_update_force_accum_layout,
                &position_update_spatial_grid_layout,
            ],
            "Position Update",
        );

        let velocity_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/velocity_update.wgsl"),
            "main",
            &[&physics_layout, &velocity_update_angular_layout],
            "Velocity Update",
        );

        let mass_accum = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/mass_accum.wgsl"),
            "main",
            &[&physics_layout, &mass_accum_layout],
            "Mass Accumulation",
        );

        // Create cell insertion pipeline
        let cell_insertion = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/cell_insertion.wgsl"),
            "main",
            &[
                &cell_insertion_physics_layout,
                &cell_insertion_params_layout,
                &cell_insertion_state_layout,
            ],
            "Cell Insertion",
        );

        // Create cell data extraction pipeline
        let cell_data_extraction = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/extract_cell_data.wgsl"),
            "main",
            &[
                &physics_layout,
                &cell_extraction_params_layout,
                &cell_extraction_state_layout,
                &cell_extraction_output_layout,
            ],
            "Cell Data Extraction",
        );

        // Create spatial query pipeline
        let spatial_query = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_query.wgsl"),
            "main",
            &[
                &physics_layout,
                &spatial_query_params_layout,
                &spatial_query_result_layout,
            ],
            "Spatial Query",
        );

        // Create position update pipeline
        // Uses cell_insertion_physics_layout to access all 3 triple-buffered position/velocity sets
        let position_update_tool = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/update_position.wgsl"),
            "main",
            &[
                &cell_insertion_physics_layout,
                &position_update_params_layout,
            ],
            "Position Update Tool",
        );

        // Create cell removal pipeline
        // Uses cell_insertion_physics_layout to access all 3 triple-buffered position/velocity sets
        let cell_removal = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/cell_removal.wgsl"),
            "main",
            &[&cell_insertion_physics_layout, &cell_removal_params_layout],
            "Cell Removal",
        );

        // Create cell boost pipeline
        // Uses cell_insertion_physics_layout to access all 3 triple-buffered position/velocity sets
        let cell_boost = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/cell_boost.wgsl"),
            "main",
            &[&cell_insertion_physics_layout, &cell_boost_params_layout],
            "Cell Boost",
        );

        // Create nutrient transport pipeline (accumulate phase only)
        let nutrient_transport = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/nutrient_transport.wgsl"),
            "main",
            &[
                &physics_layout,
                &nutrient_system_layout,
                &adhesion_layout,
                &nutrient_transport_layout,
            ],
            "Nutrient Transport",
        );

        // Create nutrient apply pipeline (apply accumulated mass deltas - separate dispatch)
        let nutrient_apply = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/nutrient_apply.wgsl"),
            "main",
            &[&physics_layout, &nutrient_apply_layout],
            "Nutrient Apply",
        );

        // Create adhesion physics pipeline (per-cell processing, accumulates to force buffers)
        let adhesion_physics = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/adhesion_physics.wgsl"),
            "main",
            &[
                &physics_layout,
                &adhesion_layout,
                &rotations_layout,
                &force_accum_layout,
            ],
            "Adhesion Physics",
        );

        // Adhesion constraint sub-step pipeline (iterative stiffening)
        // Reuses physics, adhesion, and rotations layouts - no force accum needed
        let adhesion_substep = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/adhesion_substep.wgsl"),
            "main",
            &[&physics_layout, &adhesion_layout, &rotations_layout],
            "Adhesion Substep",
        );

        // Lifecycle pipelines (3-stage with ring buffer for slot allocation)
        // Stage 1: Death scan - detects dead cells, pushes slots to ring buffer
        let lifecycle_death_scan = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_unified.wgsl"),
            "death_scan",
            &[&physics_layout, &lifecycle_layout, &cell_state_read_layout],
            "Lifecycle Death Scan",
        );

        // Stage 2: Division scan - allocates slots from ring buffer for dividing cells
        // Group 3 = division_scan_adhesion_layout for neighbor deferral check
        let lifecycle_division_scan = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_unified.wgsl"),
            "division_scan",
            &[
                &physics_layout,
                &lifecycle_layout,
                &cell_state_read_layout,
                &division_scan_adhesion_layout,
            ],
            "Lifecycle Division Scan",
        );

        let lifecycle_division_execute = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_division_execute_ring.wgsl"),
            "main",
            &[
                &physics_layout,
                &lifecycle_layout,
                &cell_state_write_layout,
                &division_execute_adhesion_layout,
            ],
            "Lifecycle Division Execute",
        );

        // Adhesion cleanup pipeline (runs after death scan, before prefix sum)
        let adhesion_cleanup = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/adhesion_cleanup.wgsl"),
            "main",
            &[
                &physics_layout,
                &lifecycle_layout,
                &lifecycle_adhesion_layout,
            ],
            "Adhesion Cleanup",
        );

        // Swim force pipeline (applies thrust for Flagellocyte cells)
        let swim_force = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/swim_force.wgsl"),
            "main",
            &[
                &physics_layout,
                &swim_force_force_accum_layout,
                &swim_force_cell_data_layout,
            ],
            "Swim Force",
        );

        let plumocyte_rotation_damping = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/swim_force.wgsl"),
            "plumocyte_rotation_damping_main",
            &[
                &physics_layout,
                &swim_force_force_accum_layout,
                &swim_force_cell_data_layout,
            ],
            "Plumocyte Rotation Damping",
        );

        // Buoyancy force pipeline (applies upward force for Buoyocyte cells)
        let buoyancy_force = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/swim_force.wgsl"),
            "buoyancy_main",
            &[
                &physics_layout,
                &swim_force_force_accum_layout,
                &swim_force_cell_data_layout,
            ],
            "Buoyancy Force",
        );

        // Glueocyte environment adhesion pipeline
        // Group 0: physics, Group 1: force_accum+env_anchor, Group 2: mode data, Group 3: cave params
        let glueocyte_env_adhesion = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/glueocyte_env_adhesion.wgsl"),
            "main",
            &[
                &physics_layout,
                &env_adhesion_force_accum_layout,
                &env_adhesion_mode_data_layout,
                &cave_params_layout,
            ],
            "Glueocyte Env Adhesion",
        );

        // Glueocyte cell-to-cell adhesion pipelines
        // Group 0: physics, Group 1: adhesion buffers, Group 2: spatial grid, Group 3: mode/signal data
        let glueocyte_cell_adhesion_create = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/glueocyte_cell_adhesion.wgsl"),
            "bond_create",
            &[
                &physics_layout,
                &cell_adhesion_adhesion_layout,
                &cell_adhesion_spatial_layout,
                &cell_adhesion_mode_layout,
            ],
            "Glueocyte Cell Adhesion Create",
        );
        let glueocyte_cell_adhesion_release = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/glueocyte_cell_adhesion.wgsl"),
            "bond_release",
            &[
                &physics_layout,
                &cell_adhesion_adhesion_layout,
                &cell_adhesion_spatial_layout,
                &cell_adhesion_mode_layout,
            ],
            "Glueocyte Cell Adhesion Release",
        );

        // Cilia force pipeline (contact-dependent surface propulsion for Ciliocyte cells)
        // Group 0: physics, Group 1: force_accum+rotations, Group 2: cell/mode data+v5/v6, Group 3: spatial+organism_labels+cave_params
        let cilia_force = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/cilia_force.wgsl"),
            "main",
            &[
                &physics_layout,
                &swim_force_force_accum_layout,
                &cilia_force_cell_data_layout,
                &cilia_force_spatial_layout,
            ],
            "Cilia Force",
        );

        // Muscle contraction pipeline (modifies adhesion rest lengths for Myocyte cells)
        // Group 0: physics params + cell_count_buffer
        // Group 1: mode_indices, mode_cell_types, type_behaviors, signal_flags, mode_properties_v7, mode_properties_v8
        // Group 2: adhesion_connections, adhesion_settings_v0 (read_write), adhesion_settings_v0_original (read), cell_adhesion_indices, adhesion_counts
        let muscle_contraction_group0_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Muscle Contraction Group 0 Layout"),
                entries: &[
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
                ],
            });
        let muscle_contraction_group1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Muscle Contraction Group 1 Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 6: v11 [consume_range, consume_rate, grip_contracted, grip_extended]
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let muscle_contraction_group2_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Muscle Contraction Group 2 Layout"),
                entries: &[
                    // Binding 0: Per-cell muscle contraction output (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: Per-cell grip output (read-write)
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
                ],
            });
        let muscle_contraction = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/muscle_contraction.wgsl"),
            "main",
            &[
                &muscle_contraction_group0_layout,
                &muscle_contraction_group1_layout,
                &muscle_contraction_group2_layout,
            ],
            "Muscle Contraction",
        );

        let physiology_transport = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/physiology_transport.wgsl"),
            "main",
            &[
                &physics_layout,
                &physiology_layout,
                &physiology_transport_layout,
                &adhesion_layout,
            ],
            "Physiology Transport",
        );

        let physiology_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/physiology_update.wgsl"),
            "main",
            &[
                &physics_layout,
                &physiology_layout,
                &physiology_cell_data_layout,
            ],
            "Physiology Update",
        );

        // Signal system bind group layouts
        let signal_flags_layout = Self::create_signal_flags_bind_group_layout(device);
        let signal_propagate_flags_layout =
            Self::create_signal_propagate_flags_bind_group_layout(device);
        let signal_sense_cell_data_layout =
            Self::create_signal_sense_cell_data_bind_group_layout(device);
        let signal_sense_world_data_layout =
            Self::create_signal_sense_world_data_bind_group_layout(device);
        let signal_propagate_adhesion_layout =
            Self::create_signal_propagate_adhesion_bind_group_layout(device);

        // Signal system compute pipelines
        let signal_clear = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/signal_clear.wgsl"),
            "main",
            &[&signal_flags_layout],
            "Signal Clear",
        );
        let signal_sense = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/signal_sense.wgsl"),
            "main",
            &[
                &signal_flags_layout,
                &signal_sense_cell_data_layout,
                &signal_sense_world_data_layout,
                &cilia_force_spatial_layout,
            ],
            "Signal Sense",
        );
        let signal_propagate_source = include_str!("../../../shaders/signal_propagate.wgsl");
        let signal_propagate = Self::create_compute_pipeline(
            device,
            signal_propagate_source,
            "main",
            &[
                &signal_propagate_flags_layout,
                &signal_propagate_adhesion_layout,
            ],
            "Signal Propagate",
        );
        let reverse_source = signal_propagate_source.replace(
            "const PROPAGATION_DIRECTION: u32 = 0u;",
            "const PROPAGATION_DIRECTION: u32 = 1u;",
        );
        let signal_propagate_reverse = Self::create_compute_pipeline(
            device,
            &reverse_source,
            "main",
            &[
                &signal_propagate_flags_layout,
                &signal_propagate_adhesion_layout,
            ],
            "Signal Propagate Reverse",
        );
        let signal_combine_sweeps = Self::create_compute_pipeline(
            device,
            signal_propagate_source,
            "combine_sweeps",
            &[
                &signal_propagate_flags_layout,
                &signal_propagate_adhesion_layout,
            ],
            "Signal Combine Sweeps",
        );

        // Mode switch bind group layouts and pipeline
        let rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mode_switch_layout0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mode Switch Layout 0"),
                entries: &[
                    rw(0),  // cell_count_buffer
                    ro(1),  // death_flags
                    rw(2),  // mode_indices
                    ro(3),  // mode_cell_types
                    rw(4),  // mode_switch_time
                    uni(5), // physics_params
                    rw(6),  // split_counts - reset to 0 on switch
                    rw(7),  // max_splits - updated to new mode's value
                    rw(8),  // split_intervals - updated to new mode's value
                    rw(9),  // split_nutrient_thresholds - updated to new mode's value
                    rw(10), // nutrient_gain_rates - updated to new mode's value
                    rw(11), // max_cell_sizes - updated to new mode's value
                    rw(12), // stiffnesses - updated to new mode's value
                    rw(13), // cell_types - updated to new mode's value
                    rw(14), // birth_times - reset to current_time so new mode's split_interval starts fresh
                ],
            });
        let mode_switch_layout1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mode Switch Layout 1"),
                entries: &[
                    ro(0), // signal_flags
                    ro(1), // signal_settings_v3
                    ro(2), // signal_settings_v4
                    ro(3), // regulation_params (self-trigger guard)
                ],
            });
        let mode_switch_layout2 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mode Switch Layout 2"),
                entries: &[
                    ro(0), // mode_properties_v0: [gain, max_size, stiffness, split_interval]
                    ro(1), // mode_properties_v1: [split_mass, nutrient_priority, swim_force, ...]
                    ro(2), // mode_properties_v2: [max_splits, split_ratio, ...]
                ],
            });
        let mode_switch = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/mode_switch.wgsl"),
            "main",
            &[
                &mode_switch_layout0,
                &mode_switch_layout1,
                &mode_switch_layout2,
            ],
            "Mode Switch",
        );

        // -- Boulder bind group layouts -----------------------------------------
        let rw_storage = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro_storage = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Group 1 for boulder_physics: state(rw), moss(rw), moss_dir(rw), eat_dir(rw), count(ro), force_accum(rw), water_params(uniform), water_bitfield(ro), buoyancy_params(uniform)
        let boulder_physics_buffers_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Boulder Physics Buffers Layout"),
                entries: &[
                    rw_storage(0),
                    rw_storage(1),
                    rw_storage(2),
                    rw_storage(3),
                    ro_storage(4),
                    rw_storage(5),
                    // Binding 6: water_params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 7: water_bitfield (read-only storage)
                    ro_storage(7),
                    // Binding 8: buoyancy_params uniform [gravity_multiplier, drag_coeff, 0, 0]
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Group 0 for boulder_consume: minimal uniform (delta_time, world_size, grid_cell_size, grid_resolution)
        let boulder_consume_params_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Boulder Consume Params Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Group 1 for boulder_consume: spatial grid counts(ro), offsets(ro), cells(ro)
        let boulder_consume_spatial_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Boulder Consume Spatial Layout"),
                entries: &[ro_storage(0), ro_storage(1), ro_storage(2)],
            });

        // Group 2 for boulder_consume: positions(ro), cell_types(ro), nutrients(rw),
        //   organism_size(ro), death_flags(ro), split_thresholds(ro)
        // cell_count_buffer is NOT included - we use arrayLength() in the shader instead,
        // avoiding a conflict with the physics group which binds it read-write.
        let boulder_consume_cell_data_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Boulder Consume Cell Data Layout"),
                entries: &[
                    ro_storage(0),
                    ro_storage(1),
                    rw_storage(2),
                    ro_storage(3),
                    ro_storage(4),
                    ro_storage(5),
                ],
            });

        // Group 3 for boulder_consume: state(ro), moss(rw), eat_dir(rw), count(ro)
        let boulder_consume_buffers_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Boulder Consume Buffers Layout"),
                entries: &[ro_storage(0), rw_storage(1), rw_storage(2), ro_storage(3)],
            });

        // Boulder consume params buffer: 16 bytes (4 x f32/i32)
        let boulder_consume_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boulder Consume Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Boulder physics pipeline: group 0 = physics_layout, group 1 = boulder_physics_buffers_layout, group 2 = cave_params_layout
        let boulder_physics = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/boulder_physics.wgsl"),
            "main",
            &[
                &physics_layout,
                &boulder_physics_buffers_layout,
                &cave_params_layout,
            ],
            "Boulder Physics",
        );

        // Boulder consume pipeline: group 0 = boulder_consume_params_layout, group 1 = boulder_consume_spatial_layout,
        //   group 2 = boulder_consume_cell_data_layout, group 3 = boulder_consume_buffers_layout
        let boulder_consume = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/boulder_consume.wgsl"),
            "main",
            &[
                &boulder_consume_params_layout,
                &boulder_consume_spatial_layout,
                &boulder_consume_cell_data_layout,
                &boulder_consume_buffers_layout,
            ],
            "Boulder Consume",
        );

        Self {
            spatial_grid_clear,
            spatial_grid_assign,
            spatial_grid_insert,
            spatial_grid_build,
            clear_forces,
            collision_detection,
            position_update,
            velocity_update,
            mass_accum,
            cell_insertion,
            cell_data_extraction,
            spatial_query,
            position_update_tool,
            cell_removal,
            cell_boost,
            nutrient_transport,
            nutrient_apply,
            adhesion_physics,
            adhesion_substep,
            lifecycle_death_scan,
            lifecycle_division_scan,
            lifecycle_division_execute,
            adhesion_cleanup,
            swim_force,
            plumocyte_rotation_damping,
            buoyancy_force,
            glueocyte_env_adhesion,
            glueocyte_cell_adhesion_create,
            glueocyte_cell_adhesion_release,
            cilia_force,
            muscle_contraction,
            physiology_update,
            physiology_transport,
            physics_layout,
            spatial_grid_layout,
            lifecycle_layout,
            cell_state_read_layout,
            cell_state_write_layout,
            mass_accum_layout,
            rotations_layout,
            position_update_rotations_layout,
            position_update_spatial_grid_layout,
            cell_insertion_physics_layout,
            cell_insertion_params_layout,
            cell_insertion_state_layout,
            cell_extraction_params_layout,
            cell_extraction_state_layout,
            cell_extraction_output_layout,
            spatial_query_params_layout,
            spatial_query_result_layout,
            position_update_params_layout,
            cell_removal_params_layout,
            cell_boost_params_layout,
            adhesion_layout,
            force_accum_layout,
            lifecycle_adhesion_layout,
            division_execute_adhesion_layout,
            division_scan_adhesion_layout,
            clear_forces_layout,
            collision_force_accum_layout,
            position_update_force_accum_layout,
            velocity_update_angular_layout,
            nutrient_system_layout,
            nutrient_transport_layout,
            nutrient_apply_layout,
            swim_force_force_accum_layout,
            swim_force_cell_data_layout,
            cilia_force_cell_data_layout,
            cilia_force_spatial_layout,
            muscle_contraction_group0_layout,
            muscle_contraction_group1_layout,
            muscle_contraction_group2_layout,
            physiology_layout,
            physiology_cell_data_layout,
            physiology_transport_layout,
            env_adhesion_force_accum_layout,
            env_adhesion_mode_data_layout,
            cell_adhesion_adhesion_layout,
            cell_adhesion_spatial_layout,
            cell_adhesion_mode_layout,
            cave_params_layout,
            signal_clear,
            signal_sense,
            signal_propagate,
            signal_propagate_reverse,
            signal_combine_sweeps,
            mode_switch,
            signal_flags_layout,
            signal_propagate_flags_layout,
            signal_sense_cell_data_layout,
            signal_sense_world_data_layout,
            signal_propagate_adhesion_layout,
            mode_switch_layout0,
            mode_switch_layout1,
            mode_switch_layout2,
            boulder_physics,
            boulder_consume,
            boulder_physics_buffers_layout,
            boulder_consume_params_layout,
            boulder_consume_spatial_layout,
            boulder_consume_cell_data_layout,
            boulder_consume_buffers_layout,
            boulder_consume_params_buffer,
        }
    }

    /// Create bind groups for the current frame
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        _adhesion_buffers: &super::AdhesionBuffers,
        organism_label_buffer: Option<&wgpu::Buffer>,
        buffer_index: usize,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics Bind Group"),
            layout: &self.physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.velocity[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.position_and_mass[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.velocity[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        });

        let dummy_label_buffer;
        let label_buffer = if let Some(buf) = organism_label_buffer {
            buf
        } else {
            dummy_label_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Organism Label Buffer"),
                size: buffers.capacity as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_label_buffer
        };

        let spatial_grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spatial Grid Bind Group"),
            layout: &self.spatial_grid_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_grid_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.spatial_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: label_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.occupied_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.occupied_grid_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.spatial_grid_overflow_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers
                        .spatial_grid_overflow_grid_indices
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.spatial_grid_overflow_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.death_flags.as_entire_binding(),
                },
            ],
        });

        (physics_bind_group, spatial_grid_bind_group)
    }

    /// Create lifecycle bind group for division pipeline
    /// Uses ring buffer for slot allocation (replaces prefix sum system)
    pub fn create_lifecycle_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lifecycle Bind Group"),
            layout: &self.lifecycle_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.division_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.free_slot_ring.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.division_slot_assignments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.ring_state.as_entire_binding(),
                },
            ],
        })
    }

    /// Create cell state bind group (read-only version for division scan)
    pub fn create_cell_state_read_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell State Read Bind Group"),
            layout: &self.cell_state_read_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.birth_times.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.split_intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.split_ready_frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.behavior_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.mode_properties_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.mode_properties_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.mode_properties_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.mode_properties_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: buffers.mode_properties_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: buffers.signal_settings_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: buffers.signal_settings_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: buffers.embryocyte_reserve_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: buffers.mode_properties_v9.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: buffers.mode_properties_v10.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: buffers.cell_thermal_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: buffers.mode_properties_v11.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: buffers.stemocyte_delay_timers.as_entire_binding(),
                },
            ],
        })
    }

    /// Create cell state bind group (read-write version for division execute)
    pub fn create_cell_state_write_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let output_index = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell State Write Bind Group"),
            layout: &self.cell_state_write_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.birth_times.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.split_intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.split_ready_frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.next_cell_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.max_cell_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.rotations[output_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: buffers.genome_mode_data_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: buffers.genome_mode_data_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: buffers.genome_mode_data_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: buffers.genome_mode_data_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: buffers.genome_mode_data_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: buffers.parent_make_adhesion_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: buffers.child_mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: buffers.mode_properties_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: buffers.mode_properties_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: buffers.mode_properties_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: buffers.mode_properties_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: buffers.mode_properties_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: buffers.child_a_keep_adhesion_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: buffers.child_b_keep_adhesion_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 31,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 32,
                    resource: buffers.genome_orientations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 33,
                    resource: buffers
                        .child_a_after_split_keep_adhesion_flags
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 34,
                    resource: buffers
                        .child_b_after_split_keep_adhesion_flags
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 35,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 36,
                    resource: buffers.signal_settings_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 37,
                    resource: buffers.signal_settings_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 38,
                    resource: buffers.signal_settings_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 39,
                    resource: buffers.embryocyte_reserve_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 41,
                    resource: buffers.development_addresses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 42,
                    resource: buffers.parent_lineage_hashes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 43,
                    resource: buffers.is_initial_mode.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 44,
                    resource: buffers.organism_cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 45,
                    resource: buffers.cell_water.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 46,
                    resource: buffers.cell_heat_energy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 47,
                    resource: buffers.cell_cached_temperature.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 48,
                    resource: buffers.cell_thermal_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 49,
                    resource: buffers.stemocyte_delay_timers.as_entire_binding(),
                },
            ],
        })
    }

    /// Create mass accumulation bind group (nutrient gain rates and split nutrient thresholds per cell)
    fn create_physiology_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physiology Bind Group"),
            layout: &self.physiology_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.cell_water.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.cell_heat_energy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_cached_temperature.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.cell_thermal_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.cell_water_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_heat_energy_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.cell_cached_temperature_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.cell_thermal_state_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.cell_prev_muscle_contraction.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.muscle_contraction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.cell_water_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.cell_heat_delta.as_entire_binding(),
                },
            ],
        })
    }

    fn create_physiology_transport_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physiology Transport Bind Group"),
            layout: &self.physiology_transport_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
            ],
        })
    }

    fn create_physiology_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::adhesion_buffers::AdhesionBuffers,
        water_grid_params_buffer: Option<&wgpu::Buffer>,
        fluid_state_buffer: Option<&wgpu::Buffer>,
        temperature_field_buffer: Option<&wgpu::Buffer>,
        geothermal_heat_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let dummy_water_params;
        let water_grid_params = if let Some(buffer) = water_grid_params_buffer {
            buffer
        } else {
            dummy_water_params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Physiology Water Grid Params"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            &dummy_water_params
        };

        let dummy_fluid_state;
        let fluid_state = if let Some(buffer) = fluid_state_buffer {
            buffer
        } else {
            dummy_fluid_state = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Physiology Fluid State"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_fluid_state
        };

        let dummy_temperature_field;
        let temperature_field = if let Some(buffer) = temperature_field_buffer {
            buffer
        } else {
            dummy_temperature_field = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Physiology Temperature Field"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_temperature_field
        };

        let dummy_geothermal_heat;
        let geothermal_heat = if let Some(buffer) = geothermal_heat_buffer {
            buffer
        } else {
            dummy_geothermal_heat = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Physiology Geothermal Heat"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_geothermal_heat
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physiology Cell Data Bind Group"),
            layout: &self.physiology_cell_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: water_grid_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fluid_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: temperature_field.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: geothermal_heat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.mode_properties_v14.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: adhesion_buffers.adhesion_settings_v0.as_entire_binding(),
                },
            ],
        })
    }

    /// Create mass accumulation bind group (nutrient gain rates and split nutrient thresholds per cell)
    pub fn create_mass_accum_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mass Accumulation Bind Group"),
            layout: &self.mass_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create rotations bind group for adhesion physics shader (per-cell processing)
    /// Includes both input and output buffers for direct velocity updates
    pub fn create_rotations_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let output_index = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rotations Bind Group"),
            layout: &self.rotations_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.angular_velocities[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.rotations[output_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.angular_velocities[output_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.genome_orientations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.muscle_contraction_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create position update rotations bind group (different layout from adhesion physics)
    pub fn create_position_update_rotations_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let output_index = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Position Update Rotations Bind Group"),
            layout: &self.position_update_rotations_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.rotations[output_index].as_entire_binding(),
                },
            ],
        })
    }

    /// Create all cached bind groups for the physics pipeline
    /// Call once at initialization, not per-frame
    pub fn create_cached_bind_groups(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
        signal_sense_world_params_buffer: &wgpu::Buffer,
        signal_sense_nutrient_buffer: &wgpu::Buffer,
        signal_sense_light_field_buffer: &wgpu::Buffer,
        signal_sense_light_color_field_buffer: &wgpu::Buffer,
        signal_sense_solid_mask_buffer: &wgpu::Buffer,
        signal_sense_density_field_buffer: &wgpu::Buffer,
        organism_label_buffer: Option<&wgpu::Buffer>,
        organism_size_buffer: Option<&wgpu::Buffer>,
        boulder_buffers: Option<&super::boulder_buffers::BoulderBuffers>,
    ) -> CachedBindGroups {
        // Create physics bind groups for all 3 buffer indices
        let physics = [
            self.create_physics_bind_group_for_index(device, buffers, 0),
            self.create_physics_bind_group_for_index(device, buffers, 1),
            self.create_physics_bind_group_for_index(device, buffers, 2),
        ];

        // Spatial grid bind group (same for all frames)
        let spatial_grid = self.create_spatial_grid_bind_group_internal(
            device,
            buffers,
            adhesion_buffers,
            organism_label_buffer,
        );

        // Position update spatial grid bind group (read-only, same for all frames)
        let position_update_spatial_grid =
            self.create_position_update_spatial_grid_bind_group_internal(device, buffers);

        // Lifecycle bind group (same for all frames)
        let lifecycle = self.create_lifecycle_bind_group(device, buffers);

        // Cell state read bind group (same for all frames)
        let cell_state_read =
            self.create_cell_state_read_bind_group(device, buffers, adhesion_buffers);

        // Cell state write bind groups for all 3 buffer indices
        let cell_state_write = [
            self.create_cell_state_write_bind_group(device, buffers, adhesion_buffers, 0),
            self.create_cell_state_write_bind_group(device, buffers, adhesion_buffers, 1),
            self.create_cell_state_write_bind_group(device, buffers, adhesion_buffers, 2),
        ];

        // Mass accumulation bind group (same for all frames)
        let mass_accum = self.create_mass_accum_bind_group(device, buffers);

        // Rotations bind groups for all 3 buffer indices (for adhesion physics)
        let rotations = [
            self.create_rotations_bind_group(device, buffers, adhesion_buffers, 0),
            self.create_rotations_bind_group(device, buffers, adhesion_buffers, 1),
            self.create_rotations_bind_group(device, buffers, adhesion_buffers, 2),
        ];

        // Position update rotations bind groups for all 3 buffer indices
        let position_update_rotations = [
            self.create_position_update_rotations_bind_group(device, buffers, 0),
            self.create_position_update_rotations_bind_group(device, buffers, 1),
            self.create_position_update_rotations_bind_group(device, buffers, 2),
        ];

        // Adhesion bind group (same for all frames)
        let adhesion = self.create_adhesion_bind_group(device, adhesion_buffers, buffers);

        // Force accumulation bind group (same for all frames)
        let force_accum = self.create_force_accum_bind_group(device, adhesion_buffers);

        // Lifecycle adhesion bind group (same for all frames) - for adhesion_cleanup
        let lifecycle_adhesion =
            self.create_lifecycle_adhesion_bind_group(device, adhesion_buffers);

        // Division execute adhesion bind group (same for all frames) - for division shader
        let division_execute_adhesion =
            self.create_division_execute_adhesion_bind_group(device, adhesion_buffers);

        // Division scan adhesion bind group (read-only, same for all frames)
        let division_scan_adhesion =
            self.create_division_scan_adhesion_bind_group(device, adhesion_buffers, buffers);

        // Clear forces bind group (same for all frames)
        let clear_forces = self.create_clear_forces_bind_group(device, adhesion_buffers);

        // Collision force accum bind groups (one for each buffer index)
        let collision_force_accum = [
            self.create_collision_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                0,
                boulder_buffers,
            ),
            self.create_collision_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                1,
                boulder_buffers,
            ),
            self.create_collision_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                2,
                boulder_buffers,
            ),
        ];

        // Position update force accum bind group (same for all frames)
        // Water buffers are None - will use default empty buffers
        let position_update_force_accum = [
            self.create_position_update_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                0,
                None,
                None,
                None,
                None,
            ),
            self.create_position_update_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                1,
                None,
                None,
                None,
                None,
            ),
            self.create_position_update_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                2,
                None,
                None,
                None,
                None,
            ),
        ];

        // Velocity update angular bind groups - one per rotation buffer index.
        // The shader reads angular state from the active input buffer and writes
        // the frame's output buffer, matching position_update's triple-buffer flow.
        let velocity_update_angular = [
            self.create_velocity_update_angular_bind_group(device, adhesion_buffers, buffers, 0),
            self.create_velocity_update_angular_bind_group(device, adhesion_buffers, buffers, 1),
            self.create_velocity_update_angular_bind_group(device, adhesion_buffers, buffers, 2),
        ];

        // Nutrient transport bind groups (same for all frames)
        let nutrient_system = self.create_nutrient_system_bind_group(device, buffers);
        let nutrient_transport =
            self.create_nutrient_transport_bind_group(device, buffers, organism_size_buffer);
        let nutrient_apply = self.create_nutrient_apply_bind_group(device, buffers);

        // Swim force bind groups
        let swim_force_force_accum = [
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 0),
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 1),
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 2),
        ];
        let swim_force_cell_data = self.create_swim_force_cell_data_bind_group(
            device,
            buffers,
            adhesion_buffers,
            None,
            None,
        );

        // Cilia force bind groups
        let cilia_force_force_accum = [
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 0),
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 1),
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 2),
        ];
        let cilia_force_cell_data =
            self.create_cilia_force_cell_data_bind_group(device, buffers, adhesion_buffers);
        let cilia_force_spatial = self.create_cilia_force_spatial_bind_group(
            device,
            buffers,
            organism_label_buffer,
            None,
        );

        // Muscle contraction bind groups
        let muscle_contraction_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Muscle Contraction Group 0"),
            layout: &self.muscle_contraction_group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        });
        let muscle_contraction_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Muscle Contraction Group 1"),
            layout: &self.muscle_contraction_group1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.behavior_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.mode_properties_v7.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.mode_properties_v8.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.mode_properties_v11.as_entire_binding(),
                },
            ],
        });
        let muscle_contraction_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Muscle Contraction Group 2"),
            layout: &self.muscle_contraction_group2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.muscle_contraction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.cell_grip_buffer.as_entire_binding(),
                },
            ],
        });
        let physiology = self.create_physiology_bind_group(device, buffers);
        let physiology_transport = self.create_physiology_transport_bind_group(device, buffers);
        let physiology_cell_data = self.create_physiology_cell_data_bind_group(
            device,
            buffers,
            adhesion_buffers,
            None,
            None,
            None,
            None,
        );

        // Glueocyte env adhesion bind groups
        let env_adhesion_force_accum = [
            self.create_env_adhesion_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                0,
                boulder_buffers,
            ),
            self.create_env_adhesion_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                1,
                boulder_buffers,
            ),
            self.create_env_adhesion_force_accum_bind_group(
                device,
                adhesion_buffers,
                buffers,
                2,
                boulder_buffers,
            ),
        ];
        let env_adhesion_mode_data =
            self.create_env_adhesion_mode_data_bind_group(device, buffers, adhesion_buffers);

        // Dummy cave collision bind group - used when glueocyte env adhesion runs without a cave.
        // Contains a zeroed CaveParams buffer with collision_enabled=0 so cave wall checks are skipped.
        let dummy_cave_params_buf = {
            // CaveParams is a large uniform - we only need collision_enabled=0 (offset 32, u32).
            // Allocate the full size (matching the shader struct) filled with zeros.
            // The shader only reads collision_enabled and world_radius; both default to 0 safely.
            let size = 256u64; // CaveParams is padded to 256 bytes in the shader
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Cave Params (Env Adhesion)"),
                size,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            })
        };
        let dummy_cave_solid_mask_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Cave Solid Mask (Env Adhesion)"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_cave_collision_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dummy Cave Collision Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let dummy_cave_collision = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dummy Cave Collision Bind Group"),
            layout: &dummy_cave_collision_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy_cave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_cave_solid_mask_buf.as_entire_binding(),
                },
            ],
        });

        // Glueocyte cell-to-cell adhesion bind groups
        let cell_adhesion_adhesion =
            self.create_cell_adhesion_adhesion_bind_group(device, adhesion_buffers);
        let cell_adhesion_spatial = self.create_cell_adhesion_spatial_bind_group(device, buffers);
        let cell_adhesion_mode = self.create_cell_adhesion_mode_bind_group(
            device,
            buffers,
            adhesion_buffers,
            organism_label_buffer,
        );

        // Signal system bind groups
        let signal_flags = self.create_signal_flags_bind_group(device, adhesion_buffers, buffers);
        let signal_propagate_flags =
            self.create_signal_propagate_flags_bind_group(device, adhesion_buffers, buffers);
        let signal_sense_cell_data = [
            self.create_signal_sense_cell_data_bind_group(device, buffers, 0),
            self.create_signal_sense_cell_data_bind_group(device, buffers, 1),
            self.create_signal_sense_cell_data_bind_group(device, buffers, 2),
        ];
        let signal_propagate_adhesion =
            self.create_signal_propagate_adhesion_bind_group(device, adhesion_buffers, buffers);
        let signal_sense_world_data = self.create_signal_sense_world_data_bind_group(
            device,
            signal_sense_world_params_buffer,
            signal_sense_nutrient_buffer,
            signal_sense_light_field_buffer,
            signal_sense_light_color_field_buffer,
            signal_sense_solid_mask_buffer,
            signal_sense_density_field_buffer,
            boulder_buffers.map(|bb| &bb.boulder_state),
            boulder_buffers.map(|bb| &bb.boulder_count),
        );

        // Mode switch bind groups
        let mode_switch_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mode Switch Group 0"),
            layout: &self.mode_switch_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.mode_switch_time.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.split_intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.max_cell_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.birth_times.as_entire_binding(),
                },
            ],
        });
        let mode_switch_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mode Switch Group 1"),
            layout: &self.mode_switch_layout1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.signal_settings_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.signal_settings_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.regulation_params.as_entire_binding(),
                },
            ],
        });
        let mode_switch_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mode Switch Group 2"),
            layout: &self.mode_switch_layout2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.mode_properties_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.mode_properties_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_properties_v2.as_entire_binding(),
                },
            ],
        });

        CachedBindGroups {
            physics,
            spatial_grid,
            position_update_spatial_grid,
            lifecycle,
            cell_state_read,
            cell_state_write,
            mass_accum,
            rotations,
            position_update_rotations,
            adhesion,
            force_accum,
            lifecycle_adhesion,
            division_execute_adhesion,
            division_scan_adhesion,
            clear_forces,
            collision_force_accum,
            position_update_force_accum,
            velocity_update_angular,
            nutrient_system,
            nutrient_transport,
            nutrient_apply,
            swim_force_force_accum,
            swim_force_cell_data,
            cilia_force_force_accum,
            cilia_force_cell_data,
            cilia_force_spatial,
            muscle_contraction_group0,
            muscle_contraction_group1,
            muscle_contraction_group2,
            physiology,
            physiology_cell_data,
            physiology_transport,
            env_adhesion_force_accum,
            env_adhesion_mode_data,
            dummy_cave_collision,
            signal_flags,
            signal_propagate_flags,
            signal_sense_cell_data,
            signal_sense_world_data,
            signal_propagate_adhesion,
            mode_switch_group0,
            mode_switch_group1,
            mode_switch_group2,
            cell_adhesion_adhesion,
            cell_adhesion_spatial,
            cell_adhesion_mode,
            boulder_physics_buffers: self.create_boulder_physics_buffers_bind_group(
                device,
                boulder_buffers,
                None,
                None,
            ),
            boulder_consume_params: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Boulder Consume Params BG"),
                layout: &self.boulder_consume_params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.boulder_consume_params_buffer.as_entire_binding(),
                }],
            }),
            boulder_consume_spatial: self
                .create_boulder_consume_spatial_bind_group(device, buffers),
            boulder_consume_cell_data: [
                self.create_boulder_consume_cell_data_bind_group(
                    device,
                    buffers,
                    organism_size_buffer,
                    0,
                ),
                self.create_boulder_consume_cell_data_bind_group(
                    device,
                    buffers,
                    organism_size_buffer,
                    1,
                ),
                self.create_boulder_consume_cell_data_bind_group(
                    device,
                    buffers,
                    organism_size_buffer,
                    2,
                ),
            ],
            boulder_consume_buffers: self
                .create_boulder_consume_buffers_bind_group(device, boulder_buffers),
            boulder_dummy_cave: {
                // 256-byte CaveParams with collision_enabled = 0 (all zeros).
                // The shader early-exits when collision_enabled == 0.
                let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Boulder Dummy Cave Params"),
                    contents: &vec![0u8; 256],
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let dummy_solid_mask =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Boulder Dummy Cave Solid Mask"),
                        contents: bytemuck::cast_slice(&[0u32]),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Boulder Dummy Cave BG"),
                    layout: &self.cave_params_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: dummy_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dummy_solid_mask.as_entire_binding(),
                        },
                    ],
                })
            },
        }
    }

    pub fn create_boulder_physics_buffers_bind_group(
        &self,
        device: &wgpu::Device,
        boulder_buffers: Option<&super::boulder_buffers::BoulderBuffers>,
        water_params_buffer: Option<&wgpu::Buffer>,
        water_bitfield_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        // Dummy buffers when boulder system not yet initialized
        let dummy = |size: u64, label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let n = super::boulder_buffers::MAX_BOULDERS as u64;
        let (state_buf, moss_buf, dir_buf, eat_buf, count_buf, force_buf);
        let (state_ref, moss_ref, dir_ref, eat_ref, count_ref, force_ref): (
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
        );
        if let Some(bb) = boulder_buffers {
            state_ref = &bb.boulder_state;
            moss_ref = &bb.boulder_moss;
            dir_ref = &bb.boulder_moss_dir;
            eat_ref = &bb.boulder_eat_dir_accum;
            count_ref = &bb.boulder_count;
            force_ref = &bb.boulder_force_accum;
        } else {
            state_buf = dummy(n * 80, "Dummy Boulder State");
            moss_buf = dummy(n * 4, "Dummy Boulder Moss");
            dir_buf = dummy(n * 16, "Dummy Boulder Moss Dir");
            eat_buf = dummy(n * 12, "Dummy Boulder Eat Dir");
            count_buf = dummy(16, "Dummy Boulder Count");
            force_buf = dummy(n * 12, "Dummy Boulder Force Accum");
            state_ref = &state_buf;
            moss_ref = &moss_buf;
            dir_ref = &dir_buf;
            eat_ref = &eat_buf;
            count_ref = &count_buf;
            force_ref = &force_buf;
        }

        // Water buffers - dummy when fluid system not initialized
        let dummy_water_params;
        let dummy_water_bitfield;
        let water_params_buf: &wgpu::Buffer = if let Some(b) = water_params_buffer {
            b
        } else {
            dummy_water_params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Water Params (Boulder)"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            &dummy_water_params
        };
        let water_bitfield_buf: &wgpu::Buffer = if let Some(b) = water_bitfield_buffer {
            b
        } else {
            dummy_water_bitfield = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Water Bitfield (Boulder)"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_water_bitfield
        };

        // Buoyancy params buffer
        let dummy_buoyancy;
        let buoyancy_buf: &wgpu::Buffer = if let Some(bb) = boulder_buffers {
            &bb.boulder_buoyancy_params
        } else {
            dummy_buoyancy = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Boulder Buoyancy"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            &dummy_buoyancy
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Boulder Physics Buffers BG"),
            layout: &self.boulder_physics_buffers_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: moss_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dir_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: eat_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: force_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: water_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: water_bitfield_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buoyancy_buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_boulder_consume_spatial_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Boulder Consume Spatial BG"),
            layout: &self.boulder_consume_spatial_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.spatial_grid_cells.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_boulder_consume_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        organism_size_buffer: Option<&wgpu::Buffer>,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let dummy_size_buf;
        let size_buf = if let Some(b) = organism_size_buffer {
            b
        } else {
            dummy_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Org Size (Boulder Consume)"),
                size: buffers.capacity as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_size_buf
        };
        // Bind the position buffer for this index as read-only.
        // The dispatch uses the INPUT index (previous frame's output), so this buffer
        // is never simultaneously bound read-write by the physics group.
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Boulder Consume Cell Data BG {}", buffer_index)),
            layout: &self.boulder_consume_cell_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.position_and_mass[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: size_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_boulder_consume_buffers_bind_group(
        &self,
        device: &wgpu::Device,
        boulder_buffers: Option<&super::boulder_buffers::BoulderBuffers>,
    ) -> wgpu::BindGroup {
        let dummy = |size: u64, label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let n = super::boulder_buffers::MAX_BOULDERS as u64;
        let (state_buf, moss_buf, eat_buf, count_buf);
        let (state_ref, moss_ref, eat_ref, count_ref): (
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
            &wgpu::Buffer,
        );
        if let Some(bb) = boulder_buffers {
            state_ref = &bb.boulder_state;
            moss_ref = &bb.boulder_moss;
            eat_ref = &bb.boulder_eat_dir_accum;
            count_ref = &bb.boulder_count;
        } else {
            state_buf = dummy(n * 80, "Dummy Boulder State (Consume)");
            moss_buf = dummy(n * 4, "Dummy Boulder Moss (Consume)");
            eat_buf = dummy(n * 12, "Dummy Boulder Eat Dir (Consume)");
            count_buf = dummy(16, "Dummy Boulder Count (Consume)");
            state_ref = &state_buf;
            moss_ref = &moss_buf;
            eat_ref = &eat_buf;
            count_ref = &count_buf;
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Boulder Consume Buffers BG"),
            layout: &self.boulder_consume_buffers_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: moss_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: eat_ref.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: count_ref.as_entire_binding(),
                },
            ],
        })
    }

    /// Create physics bind group for a specific buffer index
    fn create_physics_bind_group_for_index(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Physics Bind Group {}", buffer_index)),
            layout: &self.physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.velocity[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.position_and_mass[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.velocity[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create spatial grid bind group (internal, for caching)
    fn create_spatial_grid_bind_group_internal(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        _adhesion_buffers: &super::AdhesionBuffers,
        organism_label_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let dummy_label_buffer;
        let label_buffer = if let Some(buf) = organism_label_buffer {
            buf
        } else {
            dummy_label_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Organism Label Buffer"),
                size: buffers.capacity as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_label_buffer
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spatial Grid Bind Group"),
            layout: &self.spatial_grid_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_grid_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.spatial_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: label_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.occupied_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.occupied_grid_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.spatial_grid_overflow_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers
                        .spatial_grid_overflow_grid_indices
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.spatial_grid_overflow_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.death_flags.as_entire_binding(),
                },
            ],
        })
    }

    /// Create read-only spatial grid bind group for position update sweep tests
    fn create_position_update_spatial_grid_bind_group_internal(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Position Update Spatial Grid Bind Group"),
            layout: &self.position_update_spatial_grid_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_grid_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.spatial_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a compute pipeline from WGSL source
    fn create_compute_pipeline(
        device: &wgpu::Device,
        shader_source: &str,
        entry_point: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        label: &str,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts,
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", label)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Create physics bind group layout
    fn create_physics_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physics Bind Group Layout"),
            entries: &[
                // Physics parameters (uniform)
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
                // Position input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Velocity input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Position output (read-write)
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
                // Velocity output (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell count buffer (read-write) - GPU-side cell count tracking
                // [0] = total cells, [1] = live cells
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create spatial grid bind group layout
    fn create_spatial_grid_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Grid Bind Group Layout"),
            entries: &[
                // Grid counts
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid offsets
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
                // Cell grid indices
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
                // Spatial grid cells (sorted cell indices by grid cell)
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
                // Stiffnesses (per-cell membrane stiffness from genome)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Organism labels (per-cell connected-component ID for self-collision filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Occupied grid cell ids, densely appended by spatial_grid_build
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Occupied grid cell count
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Overflow side-list cell ids, appended when a bucket exceeds MAX_CELLS_PER_GRID
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Overflow source bucket ids
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Overflow side-list count
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Death flags - spatial grid build marks cells that exceed bucket capacity
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create read-only spatial grid bind group layout for position update sweep tests
    fn create_position_update_spatial_grid_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Position Update Spatial Grid Bind Group Layout"),
            entries: &[
                // Grid counts (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid offsets (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell grid indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spatial grid cells (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Stiffnesses (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create lifecycle bind group layout for division pipeline
    /// Uses ring buffer for slot allocation (replaces prefix sum system)
    fn create_lifecycle_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lifecycle Bind Group Layout"),
            entries: &[
                // Death flags
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Division flags
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
                // Free slot ring buffer (replaces free_slot_indices)
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
                // Division slot assignments
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
                // Ring state: [head, tail, next_slot_id, reservation_count] (replaces lifecycle_counts)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create cell state bind group layout
    /// read_only: true for division scan (only reads), false for division execute (writes)
    fn create_cell_state_bind_group_layout(
        device: &wgpu::Device,
        read_only: bool,
    ) -> wgpu::BindGroupLayout {
        if read_only {
            // Read-only version for division scan (8 bindings)
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell State Read Bind Group Layout"),
                entries: &[
                    // Birth times
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Split intervals
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Split masses
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Split counts
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Split ready frame (for nutrient transfer delay) - read_write so
                    // division_scan can mark deferred cells to freeze their nutrients
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Max splits
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Cell types (0 = Test, 1 = Flagellocyte) - DEPRECATED, use mode_cell_types
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Mode indices (per-cell mode index)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 9: Cell type behavior flags (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 10: Nutrients buffer (read-write for atomic ops in division scan)
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bindings 11-15: Mode properties v0-v4 (for min_adhesions division gate)
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 16: Signal flags (per-cell, read-only for signal-conditional checks)
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bindings 17-18: Signal settings v0-v1 (per-mode, for division gating + apoptosis)
                    wgpu::BindGroupLayoutEntry {
                        binding: 17,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 18,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 19: Embryocyte reserve buffer (read-write: death_scan burns reserve, sets death when zero)
                    wgpu::BindGroupLayoutEntry {
                        binding: 19,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 20: Mode properties v9 (Embryocyte trigger params: use_timer, release_timer, use_threshold, threshold_value)
                    wgpu::BindGroupLayoutEntry {
                        binding: 20,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 21: Mode properties v10 (Embryocyte signal trigger params: use_signal, signal_channel, signal_value)
                    wgpu::BindGroupLayoutEntry {
                        binding: 21,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 22: Thermal state blocks frozen/heat-shock division.
                    wgpu::BindGroupLayoutEntry {
                        binding: 22,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 23: Shared mode properties v11 (Stemocyte thresholds).
                    wgpu::BindGroupLayoutEntry {
                        binding: 23,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 24,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        } else {
            // Read-write version for division execute (10 bindings)
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell State Write Bind Group Layout"),
                entries: &[
                    // Birth times
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Split intervals
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
                    // Split masses
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
                    // Split counts
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
                    // Split ready frame (for nutrient transfer delay)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Max splits
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Genome IDs
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Mode indices
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Cell IDs
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Next cell ID
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Nutrient gain rates
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Max cell sizes
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Stiffnesses (membrane stiffness per cell)
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Rotations input (read-only, from current buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Rotations output (read-write, to next buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Genome mode data v0-v4 (child orientations, split quat - 5 sub-buffers)
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 17,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 18,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 19,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Parent make adhesion flags
                    wgpu::BindGroupLayoutEntry {
                        binding: 20,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Child mode indices
                    wgpu::BindGroupLayoutEntry {
                        binding: 21,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Mode properties v0-v4 (5 sub-buffers)
                    wgpu::BindGroupLayoutEntry {
                        binding: 22,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 23,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 24,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 25,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 26,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Cell types (DEPRECATED)
                    wgpu::BindGroupLayoutEntry {
                        binding: 27,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Mode cell types
                    wgpu::BindGroupLayoutEntry {
                        binding: 28,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Child A keep adhesion flags
                    wgpu::BindGroupLayoutEntry {
                        binding: 29,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Child B keep adhesion flags
                    wgpu::BindGroupLayoutEntry {
                        binding: 30,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Nutrients buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 31,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Genome orientations
                    wgpu::BindGroupLayoutEntry {
                        binding: 32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Child A after-split keep adhesion flags
                    wgpu::BindGroupLayoutEntry {
                        binding: 33,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Child B after-split keep adhesion flags
                    wgpu::BindGroupLayoutEntry {
                        binding: 34,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 35: Signal flags (per-cell, read-only for signal-conditional child mode routing)
                    wgpu::BindGroupLayoutEntry {
                        binding: 35,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bindings 36-38: Signal settings v1-v3 (per-mode, for child mode routing)
                    wgpu::BindGroupLayoutEntry {
                        binding: 36,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 37,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 38,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 39: Embryocyte reserve buffer (read-write: halved on division, child_reserve = parent_reserve >> 1)
                    wgpu::BindGroupLayoutEntry {
                        binding: 39,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 41: Development address buffer (organism id + lineage hash), updated on division.
                    wgpu::BindGroupLayoutEntry {
                        binding: 41,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 42: Parent lineage hash buffer, written at birth and never changed.
                    wgpu::BindGroupLayoutEntry {
                        binding: 42,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 43: Is-initial-mode flag per mode (read-only, 1 = genome's initial mode).
                    wgpu::BindGroupLayoutEntry {
                        binding: 43,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 44,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bindings 45-48: Physiology inheritance on division.
                    wgpu::BindGroupLayoutEntry {
                        binding: 45,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 46,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 47,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 48,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 49,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        }
    }

    /// Create mass accumulation bind group layout (nutrient gain rates and split nutrient thresholds per cell)
    fn create_physiology_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physiology Bind Group Layout"),
            entries: &[
                ro(0),  // water current
                ro(1),  // heat current
                ro(2),  // cached temperature current
                ro(3),  // thermal state current
                rw(4),  // water next
                rw(5),  // heat next
                rw(6),  // cached temperature next
                rw(7),  // thermal state next
                rw(8),  // previous muscle contraction
                rw(9),  // current muscle contraction
                rw(10), // water transport delta
                rw(11), // heat transport delta
            ],
        })
    }

    fn create_physiology_transport_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physiology Transport Bind Group Layout"),
            entries: &[
                ro(0), // death flags
                ro(1), // mode indices
                ro(2), // mode cell types
            ],
        })
    }

    fn create_physiology_cell_data_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physiology Cell Data Bind Group Layout"),
            entries: &[
                ro(0), // death flags
                ro(1), // mode indices
                ro(2), // mode cell types
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                ro(4),  // fluid voxel state
                ro(5),  // per-voxel temperature field
                ro(6),  // geothermal heat field
                ro(7),  // mode_properties_v14: Siphonocyte params
                ro(9),  // signal flags for signal-gated Siphonocyte intake
                ro(10), // adhesion_settings_v0: rest length for bond heat behavior
            ],
        })
    }

    /// Create mass accumulation bind group layout (nutrient gain rates and split nutrient thresholds per cell)
    fn create_mass_accum_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mass Accumulation Bind Group Layout"),
            entries: &[
                // Nutrient gain rates (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Split nutrient thresholds (read-only) - derived from split_mass
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Death flags (read-only) - to skip dead cells
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mode cell types (read-only) - to check if auto-gain cell
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mode indices (read-only) - to get cell type
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Nutrients buffer (read-write) - fixed-point i32, atomic
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create rotations bind group layout for adhesion physics shader (per-cell processing)
    /// Needs read-write access to rotations and angular velocities for direct updates
    fn create_rotations_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Rotations Bind Group Layout"),
            entries: &[
                // Binding 0: Rotations input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Angular velocities input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Rotations output (read-write)
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
                // Binding 3: Angular velocities output (read-write)
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
                // Binding 4: Genome orientations (read-only) - pure genome-derived orientations for adhesion
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Per-cell muscle contraction values (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create position update rotations bind group layout (different from adhesion physics)
    fn create_position_update_rotations_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Position Update Rotations Bind Group Layout"),
            entries: &[
                // Binding 0: Rotations input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Rotations output (read-write) - required by position update shader
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
            ],
        })
    }

    /// Create adhesion bind group layout (Group 1 in adhesion physics shader)
    fn create_adhesion_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Adhesion Bind Group Layout"),
            entries: &[
                // Binding 0: Adhesion connections (read_write so physics shader can mark bonds inactive)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Adhesion settings V0 (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Adhesion settings V1 (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Adhesion settings V2 (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Adhesion counts (read-only for per-cell processing)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Cell adhesion indices (read-only, 20 indices per cell)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: mode_switch_time (read-only) - used to extend grace period after mode switch
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: cell thermal state (read-only) - cold bonds are more brittle
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create force accumulation bind group layout (Group 3 in adhesion physics shader)
    fn create_force_accum_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Force Accumulation Bind Group Layout"),
            entries: &[
                // Binding 0: Force accumulation X (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Force accumulation Y (atomic i32)
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
                // Binding 2: Force accumulation Z (atomic i32)
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
                // Binding 3: Torque accumulation X (atomic i32)
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
                // Binding 4: Torque accumulation Y (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Torque accumulation Z (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create lifecycle adhesion bind group layout (Group 2 in adhesion_cleanup shader)
    /// Used for adhesion cleanup - 4 bindings: connections, counts, indices, free_slots
    fn create_lifecycle_adhesion_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lifecycle Adhesion Bind Group Layout"),
            entries: &[
                // Binding 0: Adhesion connections (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Adhesion counts (read-write, atomic)
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
                // Binding 2: Cell adhesion indices (read-write)
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
                // Binding 3: Free adhesion slots (read-write)
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
        })
    }

    /// Create division execute adhesion bind group layout (Group 3 in lifecycle division execute shader)
    /// Matches shader: binding 0 = adhesion_connections, binding 1 = cell_adhesion_indices,
    /// binding 2 = next_adhesion_id, binding 3 = free_adhesion_slots, binding 4 = adhesion_counts
    fn create_division_execute_adhesion_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Division Execute Adhesion Bind Group Layout"),
            entries: &[
                // Binding 0: Adhesion connections (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Cell adhesion indices (read-write)
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
                // Binding 2: Next adhesion ID (atomic counter, fallback allocator)
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
                // Binding 3: Free adhesion slots stack (for reuse)
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
                // Binding 4: Adhesion counts [total, live, free_top, padding] (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create division scan adhesion bind group layout (Group 3 in lifecycle division scan shader)
    /// Read-only access to adhesion data for checking neighbor division status
    fn create_division_scan_adhesion_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Division Scan Adhesion Bind Group Layout"),
            entries: &[
                // Binding 0: Adhesion connections (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Cell adhesion indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create adhesion bind group (Group 1 in adhesion physics shader)
    fn create_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &super::GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adhesion Bind Group"),
            layout: &self.adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.adhesion_settings_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.adhesion_settings_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.adhesion_settings_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.adhesion_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: triple_buffers.mode_switch_time.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: triple_buffers.cell_thermal_state.as_entire_binding(),
                },
            ],
        })
    }

    /// Create force accumulation bind group (Group 3 in adhesion physics shader)
    fn create_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Force Accumulation Bind Group"),
            layout: &self.force_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.force_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.force_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.force_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.torque_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.torque_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.torque_accum_z.as_entire_binding(),
                },
            ],
        })
    }

    /// Create lifecycle adhesion bind group (Group 2 in adhesion_cleanup shader)
    /// Used for adhesion cleanup - 4 bindings: connections, counts, indices, free_slots
    fn create_lifecycle_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lifecycle Adhesion Bind Group"),
            layout: &self.lifecycle_adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.adhesion_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.free_adhesion_slots.as_entire_binding(),
                },
            ],
        })
    }

    /// Create division execute adhesion bind group (Group 3 in lifecycle division execute shader)
    /// Matches shader: binding 0 = adhesion_connections, binding 1 = cell_adhesion_indices,
    /// binding 2 = next_adhesion_id, binding 3 = free_adhesion_slots, binding 4 = adhesion_counts
    fn create_division_execute_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Division Execute Adhesion Bind Group"),
            layout: &self.division_execute_adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.next_adhesion_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.free_adhesion_slots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.adhesion_counts.as_entire_binding(),
                },
            ],
        })
    }

    /// Create division scan adhesion bind group (Group 3 in lifecycle division scan shader)
    /// Read-only access to adhesion data for checking neighbor division status
    fn create_division_scan_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        _triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Division Scan Adhesion Bind Group"),
            layout: &self.division_scan_adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
            ],
        })
    }

    /// Create clear forces bind group layout (Group 1 in clear_forces shader)
    fn create_clear_forces_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Clear Forces Bind Group Layout"),
            entries: &[
                // Binding 0: Force accumulation X (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Force accumulation Y (atomic i32)
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
                // Binding 2: Force accumulation Z (atomic i32)
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
                // Binding 3: Torque accumulation X (atomic i32)
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
                // Binding 4: Torque accumulation Y (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Torque accumulation Z (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create collision force accum bind group layout (Group 2 in collision_detection shader)
    fn create_collision_force_accum_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collision Force Accum Bind Group Layout"),
            entries: &[
                // Binding 0: Force accumulation X (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Force accumulation Y (atomic i32)
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
                // Binding 2: Force accumulation Z (atomic i32)
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
                // Binding 3: Torque accumulation X (atomic i32)
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
                // Binding 4: Torque accumulation Y (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Torque accumulation Z (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Rotations (read-only for boundary torque calculation)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Angular velocities (read-only for rolling friction)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Boulder state (read-only - position and radius for cell-boulder collision)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Boulder count (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Boulder force accumulator (read-write atomic - cells push boulders)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: Death flags (read-write - collision culls impossible overcrowding)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 12: Cell adhesion indices (read-only - bonded cells are exempt from density culling)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create position update force accum bind group layout (Group 2 in position_update shader)
    fn create_position_update_force_accum_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Position Update Force Accum Bind Group Layout"),
            entries: &[
                // Binding 0: Force accumulation X (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Force accumulation Y (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Force accumulation Z (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Previous accelerations buffer (read-write)
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
                // Binding 4: Water grid params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Water bitfield (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Water velocity field (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Per-cell grip/friction buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Ice bitfield (read-only storage) - solid obstacle for cells
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Torque accumulation X (read-write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Torque accumulation Y (read-write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: Torque accumulation Z (read-write atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 12: Angular velocities (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create velocity update angular bind group layout (Group 1 in velocity_update shader)
    fn create_velocity_update_angular_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Velocity Update Angular Bind Group Layout"),
            entries: &[
                // Binding 0: Torque accumulation X (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Torque accumulation Y (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Torque accumulation Z (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Angular velocities input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Rotations input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Angular velocities output (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Rotations output (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create clear forces bind group (Group 1 in clear_forces shader)
    fn create_clear_forces_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clear Forces Bind Group"),
            layout: &self.clear_forces_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.force_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.force_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.force_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.torque_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.torque_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.torque_accum_z.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_collision_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
        boulder_buffers: Option<&super::boulder_buffers::BoulderBuffers>,
    ) -> wgpu::BindGroup {
        let n = super::boulder_buffers::MAX_BOULDERS as u64;
        let dummy_state;
        let dummy_count;
        let dummy_force;
        let (state_buf, count_buf, force_accum_buf): (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) =
            if let Some(bb) = boulder_buffers {
                (
                    &bb.boulder_state,
                    &bb.boulder_count,
                    &bb.boulder_force_accum,
                )
            } else {
                dummy_state = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Boulder State (Collision)"),
                    size: n * 80,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                dummy_count = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Boulder Count (Collision)"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                dummy_force = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Boulder Force Accum (Collision)"),
                    size: n * 12,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                (&dummy_state, &dummy_count, &dummy_force)
            };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "Collision Force Accum Bind Group {}",
                buffer_index
            )),
            layout: &self.collision_force_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.force_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.force_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.force_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.torque_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.torque_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.torque_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: triple_buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: adhesion_buffers.angular_velocities[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: force_accum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: triple_buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
            ],
        })
    }

    /// Create position update force accum bind group (Group 2 in position_update shader)
    fn create_position_update_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
        water_grid_params_buffer: Option<&wgpu::Buffer>,
        water_bitfield_buffer: Option<&wgpu::Buffer>,
        water_velocity_buffer: Option<&wgpu::Buffer>,
        ice_bitfield_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        use wgpu::util::DeviceExt;

        // Create default water buffers if not provided
        let (
            water_grid_params_buffer,
            water_bitfield_buffer,
            water_velocity_buffer,
            ice_bitfield_buffer,
        ) = match (
            water_grid_params_buffer,
            water_bitfield_buffer,
            water_velocity_buffer,
            ice_bitfield_buffer,
        ) {
            (Some(params), Some(bitfield), Some(velocity), Some(ice)) => (
                params.clone(),
                bitfield.clone(),
                velocity.clone(),
                ice.clone(),
            ),
            _ => {
                // Default params with grid_resolution=0 which will cause all position lookups to be out of bounds
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct DefaultWaterGridParams {
                    grid_resolution: u32,
                    cell_size: f32,
                    grid_origin_x: f32,
                    grid_origin_y: f32,
                    grid_origin_z: f32,
                    buoyancy_multiplier: f32,
                    water_viscosity: f32,
                    _pad1: f32,
                }

                let default_params = DefaultWaterGridParams {
                    grid_resolution: 0, // Zero resolution = no valid grid cells = no water detection
                    cell_size: 1.0,
                    grid_origin_x: 0.0,
                    grid_origin_y: 0.0,
                    grid_origin_z: 0.0,
                    buoyancy_multiplier: 0.0,
                    water_viscosity: 0.0,
                    _pad1: 0.0,
                };

                let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Default Water Grid Params Buffer"),
                    contents: bytemuck::cast_slice(&[default_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                // Minimal bitfield buffer (just 4 bytes to satisfy buffer requirements)
                let bitfield_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Default Water Bitfield Buffer"),
                        contents: &[0u8; 4],
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                // Minimal velocity buffer (just 4 bytes to satisfy buffer requirements)
                let velocity_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Default Water Velocity Buffer"),
                        contents: &[0u8; 4],
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                // Minimal ice bitfield buffer (zero = no ice anywhere)
                let ice_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Default Ice Bitfield Buffer"),
                    contents: &[0u8; 4],
                    usage: wgpu::BufferUsages::STORAGE,
                });

                (params_buffer, bitfield_buffer, velocity_buffer, ice_buffer)
            }
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Position Update Force Accum Bind Group"),
            layout: &self.position_update_force_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.force_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.force_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.force_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.prev_accelerations.as_entire_binding(),
                },
                // Water buffers (bindings 4, 5, and 6)
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: water_grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: water_bitfield_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: water_velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: triple_buffers.cell_grip_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: ice_bitfield_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: adhesion_buffers.torque_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: adhesion_buffers.torque_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: adhesion_buffers.torque_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: adhesion_buffers.angular_velocities[buffer_index].as_entire_binding(),
                },
            ],
        })
    }

    /// Create velocity update angular bind group (Group 1 in velocity_update shader)
    /// `buffer_index` selects the current input state; output is `(buffer_index + 1) % 3`.
    fn create_velocity_update_angular_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        let output_index = (buffer_index + 1) % 3;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Velocity Update Angular Bind Group"),
            layout: &self.velocity_update_angular_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.torque_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.torque_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.torque_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.angular_velocities[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.angular_velocities[output_index]
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: triple_buffers.rotations[output_index].as_entire_binding(),
                },
            ],
        })
    }

    /// Create nutrient system bind group layout (Group 1 in nutrient transport shader)
    fn create_nutrient_system_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient System Bind Group Layout"),
            entries: &[
                // Binding 0: Nutrient gain rates (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Max cell sizes (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Mode indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Genome IDs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create nutrient transport bind group layout (Group 3 in nutrient transport shader)
    fn create_nutrient_transport_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Transport Bind Group Layout"),
            entries: &[
                // Binding 0: Mass deltas (read-write, for accumulating transfers)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Death flags (read-write, for marking starved cells)
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
                // Bindings 2-6: mode_properties sub-buffers v0-v4
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Split ready frame
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Mode cell types
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Nutrients buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Split nutrient thresholds
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: Embryocyte reserve buffer (read-write: incoming nutrients redirected here for Embryocytes)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 12: Mode properties v9 (Embryocyte trigger params: use_timer, release_timer, use_threshold, threshold_value)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 13: Mode properties v12 (Vasculocyte params)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 14: Organism size buffer (read-only: cell count per organism, indexed by root label)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create mode properties bind group layout (Group 4 in nutrient transport shader)
    /// Create nutrient system bind group (Group 1 in nutrient transport shader)
    fn create_nutrient_system_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient System Bind Group"),
            layout: &self.nutrient_system_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.max_cell_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
            ],
        })
    }

    /// Create nutrient transport bind group (Group 3 in nutrient transport shader)
    fn create_nutrient_transport_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
        organism_size_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        // Use a dummy buffer if organism sizes aren't available yet
        let dummy_size_buffer;
        let size_buf = if let Some(buf) = organism_size_buffer {
            buf
        } else {
            dummy_size_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Organism Size Buffer"),
                size: buffers.capacity as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_size_buffer
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Transport Bind Group"),
            layout: &self.nutrient_transport_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.mass_deltas_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_properties_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.mode_properties_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.mode_properties_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.mode_properties_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.mode_properties_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.split_ready_frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.embryocyte_reserve_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.mode_properties_v9.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.mode_properties_v12.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: size_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Create nutrient apply bind group layout (Group 1 in nutrient_apply shader)
    fn create_nutrient_apply_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Apply Bind Group Layout"),
            entries: &[
                // Binding 0: Mass deltas (read-write, atomic i32) - legacy, kept for layout compatibility
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Death flags (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Nutrients buffer (read-write, atomic i32) - for deriving mass from nutrients
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
            ],
        })
    }

    /// Create nutrient apply bind group (Group 1 in nutrient_apply shader)
    fn create_nutrient_apply_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Apply Bind Group"),
            layout: &self.nutrient_apply_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.mass_deltas_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create cell insertion physics bind group layout (Group 0 in cell_insertion shader)
    /// Contains all 3 triple-buffered position and velocity buffers for writing to all sets
    fn create_cell_insertion_physics_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Insertion Physics Bind Group Layout"),
            entries: &[
                // Binding 0: Physics parameters uniform buffer
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
                // Binding 1: Positions buffer 0 (read-write)
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
                // Binding 2: Positions buffer 1 (read-write)
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
                // Binding 3: Positions buffer 2 (read-write)
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
                // Binding 4: Velocities buffer 0 (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Velocities buffer 1 (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Velocities buffer 2 (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Cell count buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create cell insertion params bind group layout (Group 1 in cell_insertion shader)
    /// Contains insertion params uniform and all 3 rotation buffers
    fn create_cell_insertion_params_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Insertion Params Bind Group Layout"),
            entries: &[
                // Binding 0: Cell insertion parameters uniform buffer
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
                // Binding 1: Rotations buffer 0 (read-write)
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
                // Binding 2: Rotations buffer 1 (read-write)
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
                // Binding 3: Rotations buffer 2 (read-write)
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
                // Binding 4: Free slot ring buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Ring state [head, tail, next_slot_id, reservation_count] (read-write, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Angular velocities buffer 0 (read-write) - zeroed on insertion
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Angular velocities buffer 1 (read-write) - zeroed on insertion
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Angular velocities buffer 2 (read-write) - zeroed on insertion
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Per-cell adhesion indices (read-write) - cleared on insertion
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create cell insertion state bind group layout (Group 2 in cell_insertion shader)
    fn create_cell_insertion_state_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        let rw_storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Insertion State Bind Group Layout"),
            entries: &[
                // Binding 0: Birth times
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Split intervals
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
                // Binding 2: Split masses
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
                // Binding 3: Split counts
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
                // Binding 4: Split ready frame (for nutrient transfer delay)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Max splits
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Genome IDs
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Mode indices
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Cell IDs
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Next cell ID (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Nutrient gain rates
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: Max cell sizes
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 12: Stiffnesses
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 13: Death flags
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 14: Division flags
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 15: Cell types
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 16: Nutrients buffer (atomic i32, fixed-point scale 1000)
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 17: Genome orientations (read-write) - pure genome-derived orientations per cell
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 18: Embryocyte reserve buffer (read-write atomic<u32>)
                // Initialized to 65535000 for Embryocytes, 0 for all others.
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 19: Development address [organism_id, lineage_hash_lo, lineage_hash_hi, depth_branch]
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 20: Parent lineage hash [parent_hash_lo, parent_hash_hi] — written at birth
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 22-29: hidden physiology state, current and next.
                rw_storage(22),
                rw_storage(23),
                rw_storage(24),
                rw_storage(25),
                rw_storage(26),
                rw_storage(27),
                rw_storage(28),
                rw_storage(29),
                rw_storage(30),
            ],
        })
    }

    /// Create cell extraction params bind group layout (Group 1 in extract_cell_data shader)
    fn create_cell_extraction_params_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Extraction Params Bind Group Layout"),
            entries: &[
                // Cell extraction parameters uniform buffer
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
            ],
        })
    }

    /// Create cell extraction state bind group layout (Group 2 in extract_cell_data shader)
    fn create_cell_extraction_state_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Extraction State Bind Group Layout"),
            entries: &[
                // Binding 0: Birth times (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Split intervals (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Split masses (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Split counts (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Split ready frame (read-only, for nutrient transfer delay)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Max splits (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Genome IDs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Mode indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Cell IDs (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Nutrient gain rates (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Max cell sizes (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: Stiffnesses (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 12: Nutrients buffer (read-only, atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 13: Cell types (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 14: Death flags (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 15: Cell adhesion indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 16: Organism label buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 17: Embryocyte reserve buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 18-21: hidden physiology state (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 22: packed signal flags, 16 channels per cell.
                wgpu::BindGroupLayoutEntry {
                    binding: 22,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create cell extraction output bind group layout (Group 3 in extract_cell_data shader)
    fn create_cell_extraction_output_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Extraction Output Bind Group Layout"),
            entries: &[
                // Output buffer for extracted cell data
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create spatial query params bind group layout (Group 1 in spatial_query shader)
    fn create_spatial_query_params_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Query Params Bind Group Layout"),
            entries: &[
                // Spatial query parameters (uniform)
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
            ],
        })
    }

    /// Create spatial query result bind group layout (Group 2 in spatial_query shader)
    fn create_spatial_query_result_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Query Result Bind Group Layout"),
            entries: &[
                // Output buffer for spatial query result
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create position update params bind group layout (Group 1 in update_position shader)
    fn create_position_update_params_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Position Update Params Bind Group Layout"),
            entries: &[
                // Position update parameters (uniform)
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
            ],
        })
    }

    /// Create cell removal params bind group layout (Group 1 in cell_removal shader)
    fn create_cell_removal_params_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Removal Params Bind Group Layout"),
            entries: &[
                // Cell removal parameters (uniform)
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
            ],
        })
    }

    /// Create cell boost params bind group layout (Group 1 in cell_boost shader)
    fn create_cell_boost_params_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Boost Params Bind Group Layout"),
            entries: &[
                // Cell boost parameters (uniform)
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
            ],
        })
    }

    /// Get the cell data extraction compute pipeline
    pub fn get_cell_data_extraction_pipeline(&self) -> Option<wgpu::ComputePipeline> {
        Some(self.cell_data_extraction.clone())
    }

    /// Get the cell data extraction bind group layouts
    pub fn get_cell_data_extraction_layouts(&self) -> CellDataExtractionLayouts {
        CellDataExtractionLayouts {
            physics_layout: self.physics_layout.clone(),
            params_layout: self.cell_extraction_params_layout.clone(),
            state_layout: self.cell_extraction_state_layout.clone(),
            output_layout: self.cell_extraction_output_layout.clone(),
        }
    }

    /// Create swim force force accumulation bind group layout (Group 1 in swim_force shader)
    /// Contains force/torque accumulators plus rotation state buffers.
    fn create_swim_force_force_accum_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Swim Force Force Accum Bind Group Layout"),
            entries: &[
                // Binding 0: Force accumulation X (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Force accumulation Y (atomic i32)
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
                // Binding 2: Force accumulation Z (atomic i32)
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
                // Binding 3: Rotations buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Torque accumulation X (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Torque accumulation Y (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Torque accumulation Z (atomic i32)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Angular velocities buffer (read-write for Plumocyte damping)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create swim force cell data bind group layout (Group 2 in swim_force shader)
    /// Contains mode_indices, cell_types, mode_properties, mode_cell_types, and water lookup buffers.
    fn create_swim_force_cell_data_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Swim Force Cell Data Bind Group Layout"),
            entries: &[
                // Binding 0: Mode indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Cell types (read-only) - DEPRECATED, use mode_cell_types instead
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 2-6: mode_properties sub-buffers v0-v4
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Mode cell types
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Cell type behavior flags
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Signal flags
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: Water grid params
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: Water bitfield
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 12: Siphonocyte mode params
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 13: Plumocyte mode params
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 14: cell water reserve, read-write for Siphonocyte expulsion spend
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 15: cell heat reserve, read-write for steam-assisted heat spend
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 16: thermal state, read-only freeze gate for Plumocytes
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create swim force force accumulation bind group (Group 1 in swim_force shader)
    fn create_swim_force_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "Swim Force Force Accum Bind Group {}",
                buffer_index
            )),
            layout: &self.swim_force_force_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.force_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.force_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.force_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.torque_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.torque_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: adhesion_buffers.torque_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: adhesion_buffers.angular_velocities[buffer_index].as_entire_binding(),
                },
            ],
        })
    }

    /// Create swim force cell data bind group (Group 2 in swim_force shader)
    fn create_swim_force_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
        water_grid_params_buffer: Option<&wgpu::Buffer>,
        water_bitfield_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        use wgpu::util::DeviceExt;

        let (water_params_buf, water_bitfield_buf) =
            match (water_grid_params_buffer, water_bitfield_buffer) {
                (Some(params), Some(bitfield)) => (params.clone(), bitfield.clone()),
                _ => {
                    #[repr(C)]
                    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                    struct DefaultWaterGridParams {
                        grid_resolution: u32,
                        cell_size: f32,
                        grid_origin_x: f32,
                        grid_origin_y: f32,
                        grid_origin_z: f32,
                        buoyancy_multiplier: f32,
                        water_viscosity: f32,
                        _pad1: f32,
                    }

                    let default_params = DefaultWaterGridParams {
                        grid_resolution: 0,
                        cell_size: 1.0,
                        grid_origin_x: 0.0,
                        grid_origin_y: 0.0,
                        grid_origin_z: 0.0,
                        buoyancy_multiplier: 0.0,
                        water_viscosity: 0.0,
                        _pad1: 0.0,
                    };

                    let params_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Default Swim Force Water Grid Params Buffer"),
                            contents: bytemuck::cast_slice(&[default_params]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bitfield_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Default Swim Force Water Bitfield Buffer"),
                            contents: &[0u8; 4],
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                    (params_buffer, bitfield_buffer)
                }
            };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Swim Force Cell Data Bind Group"),
            layout: &self.swim_force_cell_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triple_buffers.mode_properties_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.mode_properties_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.mode_properties_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: triple_buffers.mode_properties_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: triple_buffers.mode_properties_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: triple_buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: triple_buffers.behavior_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: water_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: water_bitfield_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: triple_buffers.mode_properties_v14.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: triple_buffers.mode_properties_v15.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: triple_buffers.cell_water.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: triple_buffers.cell_heat_energy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: triple_buffers.cell_thermal_state.as_entire_binding(),
                },
            ],
        })
    }

    /// Create cilia force cell data bind group layout (Group 2 in cilia_force shader)
    /// Extends swim_force_cell_data_layout with bindings 10, 11 for mode_properties_v5, v6
    fn create_cilia_force_cell_data_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cilia Force Cell Data Bind Group Layout"),
            entries: &[
                // Binding 0: Mode indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Cell types (read-only) - DEPRECATED, use mode_cell_types instead
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 2-6: mode_properties sub-buffers v0-v4
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Mode cell types
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: Cell type behavior flags
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 9: Signal flags
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 10: mode_properties_v5 (cilia params)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 11: mode_properties_v6 (cilia signal params)
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create cilia force spatial bind group layout (Group 3 in cilia_force shader)
    /// Contains spatial_grid_counts, spatial_grid_cells, cell_grid_indices, organism_labels, cave_params
    fn create_cilia_force_spatial_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cilia Force Spatial Bind Group Layout"),
            entries: &[
                // Binding 0: spatial_grid_counts (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: spatial_grid_cells (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: cell_grid_indices (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: organism_labels (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: cave_params (uniform) - merged from former group 4
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create cilia force cell data bind group (Group 2 in cilia_force shader)
    fn create_cilia_force_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cilia Force Cell Data Bind Group"),
            layout: &self.cilia_force_cell_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triple_buffers.mode_properties_v0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.mode_properties_v1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.mode_properties_v2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: triple_buffers.mode_properties_v3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: triple_buffers.mode_properties_v4.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: triple_buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: triple_buffers.behavior_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: triple_buffers.mode_properties_v5.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: triple_buffers.mode_properties_v6.as_entire_binding(),
                },
            ],
        })
    }

    /// Create cilia force spatial bind group (Group 3 in cilia_force shader)
    pub fn create_cilia_force_spatial_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        organism_label_buffer: Option<&wgpu::Buffer>,
        cave_params_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        // Create a dummy buffer if organism labels aren't available
        let dummy_label_buffer;
        let label_buffer = if let Some(buf) = organism_label_buffer {
            buf
        } else {
            dummy_label_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cilia Force Dummy Organism Label Buffer"),
                size: triple_buffers.capacity as u64 * 4, // u32 per cell
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_label_buffer
        };

        // Create a dummy cave params uniform buffer if not available
        let dummy_cave_params_buffer;
        let cave_buf = if let Some(buf) = cave_params_buffer {
            buf
        } else {
            // CaveParams struct size: 68 bytes data + 752 bytes padding = 820 bytes
            // Use 832 (next 16-byte aligned size) for safety
            dummy_cave_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cilia Force Dummy Cave Params Buffer"),
                size: 832,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            &dummy_cave_params_buffer
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cilia Force Spatial Bind Group"),
            layout: &self.cilia_force_spatial_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.spatial_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triple_buffers.cell_grid_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: label_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cave_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Create env adhesion force accum bind group layout (Group 1)
    /// Contains force_accum_x/y/z (atomic i32) + env_anchor buffer (vec4)
    fn create_env_adhesion_force_accum_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Env Adhesion Force Accum Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                // Binding 4: Boulder state (read-only - position and radius for glueocyte attachment)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Boulder count (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create env adhesion mode data bind group layout (Group 2)
    /// Contains mode_indices, mode_cell_types, glueocyte_env_adhesion_flags, glueocyte_boulder_adhesion_flags
    fn create_env_adhesion_mode_data_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Env Adhesion Mode Data Layout"),
            entries: &[
                ro(0), // mode_indices
                ro(1), // mode_cell_types
                ro(2), // glueocyte_env_adhesion_flags
                ro(3), // glueocyte_boulder_adhesion_flags
                ro(4), // glueocyte_cell_adhesion_flags (signal gate: channel, threshold)
                ro(5), // signal_flags (per-cell signal values)
            ],
        })
    }

    pub fn create_env_adhesion_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        _buffer_index: usize,
        boulder_buffers: Option<&super::boulder_buffers::BoulderBuffers>,
    ) -> wgpu::BindGroup {
        let n = super::boulder_buffers::MAX_BOULDERS as u64;
        let dummy_state;
        let dummy_count;
        let (state_buf, count_buf): (&wgpu::Buffer, &wgpu::Buffer) =
            if let Some(bb) = boulder_buffers {
                (&bb.boulder_state, &bb.boulder_count)
            } else {
                dummy_state = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Boulder State (Env Adhesion)"),
                    size: n * 80,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                dummy_count = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Boulder Count (Env Adhesion)"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                (&dummy_state, &dummy_count)
            };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Env Adhesion Force Accum Bind Group"),
            layout: &self.env_adhesion_force_accum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.force_accum_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.force_accum_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.force_accum_z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.env_anchor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: count_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Create env adhesion mode data bind group (Group 2)
    fn create_env_adhesion_mode_data_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Env Adhesion Mode Data Bind Group"),
            layout: &self.env_adhesion_mode_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triple_buffers
                        .glueocyte_env_adhesion_flags
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers
                        .glueocyte_boulder_adhesion_flags
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers
                        .glueocyte_cell_adhesion_flags
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
            ],
        })
    }

    // ---- Glueocyte cell adhesion bind group layouts ----

    /// Group 1 for glueocyte_cell_adhesion: adhesion connections, cell_adhesion_indices (atomic),
    /// next_adhesion_id (atomic), free_adhesion_slots, adhesion_counts (atomic).
    fn create_cell_adhesion_adhesion_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        let rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Adhesion Adhesion Layout"),
            entries: &[rw(0), rw(1), rw(2), rw(3), rw(4)],
        })
    }

    /// Group 2 for glueocyte_cell_adhesion: spatial grid (read-only).
    fn create_cell_adhesion_spatial_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Adhesion Spatial Layout"),
            // bindings 0,1,3 match spatial_grid_counts, offsets, cells (skip 2=cell_grid_indices)
            entries: &[ro(0), ro(1), ro(3)],
        })
    }

    /// Group 3 for glueocyte_cell_adhesion: mode_indices, mode_cell_types,
    /// glueocyte_cell_adhesion_flags, signal_flags, genome_orientations, death_flags.
    fn create_cell_adhesion_mode_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Adhesion Mode Layout"),
            entries: &[ro(0), ro(1), ro(2), ro(3), ro(4), rw(5), ro(6)],
        })
    }

    // ---- Glueocyte cell adhesion bind group creation ----

    fn create_cell_adhesion_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Adhesion Adhesion Bind Group"),
            layout: &self.cell_adhesion_adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.next_adhesion_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.free_adhesion_slots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.adhesion_counts.as_entire_binding(),
                },
            ],
        })
    }

    fn create_cell_adhesion_spatial_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Adhesion Spatial Bind Group"),
            layout: &self.cell_adhesion_spatial_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.spatial_grid_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.spatial_grid_cells.as_entire_binding(),
                },
            ],
        })
    }

    fn create_cell_adhesion_mode_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &super::AdhesionBuffers,
        organism_label_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        // Use a dummy buffer if organism labels aren't available yet.
        let dummy_label_buf;
        let label_buf = if let Some(buf) = organism_label_buffer {
            buf
        } else {
            dummy_label_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cell Adhesion Dummy Organism Label Buffer"),
                size: triple_buffers.capacity as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_label_buf
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Adhesion Mode Bind Group"),
            layout: &self.cell_adhesion_mode_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triple_buffers
                        .glueocyte_cell_adhesion_flags
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.genome_orientations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: triple_buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: label_buf.as_entire_binding(),
                },
            ],
        })
    }

    // ---- Signal system bind group layouts ----

    /// Signal flags bind group layout (Group 0 for signal_clear and signal_sense)
    /// binding 0: signal_flags (read_write), binding 1: cell_count (read)
    fn create_signal_flags_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Flags Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Signal propagate flags bind group layout (Group 0 for signal_propagate only).
    /// Uses a double-buffer design to eliminate the read-write hazard:
    ///   binding 0: signal_flags      (read-only  - source for this hop)
    ///   binding 1: cell_count_buffer (read-only)
    ///   binding 2: signal_flags_next (read_write - destination for this hop)
    ///   binding 3: signal_flags_forward (read-only - completed forward sweep)
    /// After each dispatch the caller copies signal_flags_next -> signal_flags.
    fn create_signal_propagate_flags_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Propagate Flags Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Signal sense cell data bind group layout (Group 1 for signal_sense)
    /// binding 0: positions, binding 1: rotations, binding 2: mode_indices, binding 3: mode_cell_types, binding 4: oculocyte_params, binding 5: regulation_params
    fn create_signal_sense_cell_data_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Sense Cell Data Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: regulation_params (per-mode regulation emission parameters)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: oculocyte_signal_values (per-mode f32 signal value emitted on detection)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: oculocyte_light_filters ([target_rgb, tolerance] per mode)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 8: cell thermal state (critical heat saturates signal channels 0-7)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Signal sense world data bind group layout (Group 2 for signal_sense)
    /// binding 0: uniform SignalSenseWorldParams
    /// binding 1: nutrient_voxels (storage, read) - food detection
    /// binding 2: light_field (storage, read) - light detection
    /// binding 3: solid_mask (storage, read) - cave/barrier detection
    /// binding 4: density_field (storage, read) - water surface detection via surface nets isosurface
    /// binding 7: light_color_field (storage, read) - color-filtered light detection
    fn create_signal_sense_world_data_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Sense World Data Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: boulder_state_sense (read-only) - for sense_type 5 (Boulder)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: boulder_count_sense (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: light_color_field (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create signal sense world data bind group (Group 2)
    /// binding 0: world params uniform, binding 1: nutrient_voxels, binding 2: light_field,
    /// binding 3: solid_mask, binding 4: density_field (water surface isosurface)
    pub fn create_signal_sense_world_data_bind_group(
        &self,
        device: &wgpu::Device,
        world_params_buffer: &wgpu::Buffer,
        nutrient_voxels_buffer: &wgpu::Buffer,
        light_field_buffer: &wgpu::Buffer,
        light_color_field_buffer: &wgpu::Buffer,
        solid_mask_buffer: &wgpu::Buffer,
        density_field_buffer: &wgpu::Buffer,
        boulder_state_buffer: Option<&wgpu::Buffer>,
        boulder_count_buffer: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let n = super::boulder_buffers::MAX_BOULDERS as u64;
        let dummy_state;
        let dummy_count;
        let bstate: &wgpu::Buffer = if let Some(b) = boulder_state_buffer {
            b
        } else {
            dummy_state = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Boulder State (Signal Sense)"),
                size: n * 80,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_state
        };
        let bcount: &wgpu::Buffer = if let Some(b) = boulder_count_buffer {
            b
        } else {
            dummy_count = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Boulder Count (Signal Sense)"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_count
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Sense World Data Bind Group"),
            layout: &self.signal_sense_world_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: world_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nutrient_voxels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: density_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bstate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bcount.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: light_color_field_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Signal propagate adhesion bind group layout (Group 1 for signal_propagate)
    /// binding 0: adhesion_connections (read), binding 1: cell_adhesion_indices (read)
    /// binding 2: mode_indices (read), binding 3: mode_cell_types (read),
    /// binding 4: mode_properties_v12 (read), binding 5: regulation_params (read)
    fn create_signal_propagate_adhesion_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Propagate Adhesion Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    // ---- Signal system bind group creation ----

    /// Create signal flags bind group (Group 0 for signal_clear and signal_sense)
    fn create_signal_flags_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Flags Bind Group"),
            layout: &self.signal_flags_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create signal propagate flags bind group (Group 0 for signal_propagate).
    /// binding 0: signal_flags (read) - source for this hop
    /// binding 1: cell_count_buffer (read)
    /// binding 2: signal_flags_next (read_write) - destination for this hop
    /// binding 3: signal_flags_forward (read) - completed forward sweep
    fn create_signal_propagate_flags_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Propagate Flags Bind Group"),
            layout: &self.signal_propagate_flags_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.signal_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.cell_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.signal_flags_next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.signal_flags_forward.as_entire_binding(),
                },
            ],
        })
    }

    /// Create signal sense cell data bind group (Group 1) for a given buffer index
    /// binding 0: positions, binding 1: rotations, binding 2: mode_indices, binding 3: mode_cell_types, binding 4: oculocyte_params, binding 5: regulation_params, binding 6: oculocyte_signal_values
    fn create_signal_sense_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "Signal Sense Cell Data Bind Group {}",
                buffer_index
            )),
            layout: &self.signal_sense_cell_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: triple_buffers.position_and_mass[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: triple_buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: triple_buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.oculocyte_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: triple_buffers.regulation_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: triple_buffers.oculocyte_signal_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: triple_buffers.oculocyte_light_filters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: triple_buffers.cell_thermal_state.as_entire_binding(),
                },
            ],
        })
    }

    /// Create signal propagate adhesion bind group (Group 1)
    fn create_signal_propagate_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        buffers: &super::GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Propagate Adhesion Bind Group"),
            layout: &self.signal_propagate_adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.mode_properties_v12.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.regulation_params.as_entire_binding(),
                },
            ],
        })
    }
}

impl CachedBindGroups {
    /// Update the position update force accum bind group with real water buffers
    /// This should be called after the fluid simulator is initialized
    pub fn update_water_buffers(
        &mut self,
        device: &wgpu::Device,
        pipelines: &GpuPhysicsPipelines,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        water_grid_params_buffer: &wgpu::Buffer,
        water_bitfield_buffer: &wgpu::Buffer,
        water_velocity_buffer: &wgpu::Buffer,
        ice_bitfield_buffer: &wgpu::Buffer,
        fluid_state_buffer: &wgpu::Buffer,
        temperature_field_buffer: &wgpu::Buffer,
        geothermal_heat_buffer: &wgpu::Buffer,
    ) {
        self.position_update_force_accum = [
            pipelines.create_position_update_force_accum_bind_group(
                device,
                adhesion_buffers,
                triple_buffers,
                0,
                Some(water_grid_params_buffer),
                Some(water_bitfield_buffer),
                Some(water_velocity_buffer),
                Some(ice_bitfield_buffer),
            ),
            pipelines.create_position_update_force_accum_bind_group(
                device,
                adhesion_buffers,
                triple_buffers,
                1,
                Some(water_grid_params_buffer),
                Some(water_bitfield_buffer),
                Some(water_velocity_buffer),
                Some(ice_bitfield_buffer),
            ),
            pipelines.create_position_update_force_accum_bind_group(
                device,
                adhesion_buffers,
                triple_buffers,
                2,
                Some(water_grid_params_buffer),
                Some(water_bitfield_buffer),
                Some(water_velocity_buffer),
                Some(ice_bitfield_buffer),
            ),
        ];
        self.swim_force_cell_data = pipelines.create_swim_force_cell_data_bind_group(
            device,
            triple_buffers,
            adhesion_buffers,
            Some(water_grid_params_buffer),
            Some(water_bitfield_buffer),
        );
        self.physiology_cell_data = pipelines.create_physiology_cell_data_bind_group(
            device,
            triple_buffers,
            adhesion_buffers,
            Some(water_grid_params_buffer),
            Some(fluid_state_buffer),
            Some(temperature_field_buffer),
            Some(geothermal_heat_buffer),
        );
    }
}

/// Cell data extraction bind group layouts
pub struct CellDataExtractionLayouts {
    pub physics_layout: wgpu::BindGroupLayout,
    pub params_layout: wgpu::BindGroupLayout,
    pub state_layout: wgpu::BindGroupLayout,
    pub output_layout: wgpu::BindGroupLayout,
}
