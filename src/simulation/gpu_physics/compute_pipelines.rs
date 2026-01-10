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
//! | 1 | Storage (read) | positions_in |
//! | 2 | Storage (read) | velocities_in |
//! | 3 | Storage (read_write) | positions_out |
//! | 4 | Storage (read_write) | velocities_out |
//! | 5 | Storage (read_write) | cell_count_buffer |
//! 
//! ### Spatial Grid Bind Group (Group 1)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read_write) | spatial_grid_counts |
//! | 1 | Storage (read_write) | spatial_grid_offsets |
//! | 2 | Storage (read_write) | cell_grid_indices |
//! | 3 | Storage (read_write) | spatial_grid_cells |
//!
//! ### Lifecycle Bind Group (Group 1 for lifecycle shaders)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read_write) | death_flags |
//! | 1 | Storage (read_write) | division_flags |
//! | 2 | Storage (read_write) | free_slot_indices |
//! | 3 | Storage (read_write) | division_slot_assignments |
//! | 4 | Storage (read_write) | lifecycle_counts |
//!
//! ### Cell State Bind Group (Group 2 for division shaders)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read/write) | birth_times |
//! | 1 | Storage (read/write) | split_intervals |
//! | 2 | Storage (read/write) | split_masses |
//! | 3 | Storage (read/write) | split_counts |
//! | 4 | Storage (read/write) | max_splits |
//! | 5 | Storage (read/write) | genome_ids |
//! | 6 | Storage (read/write) | mode_indices |
//! | 7 | Storage (read/write) | cell_ids |
//! | 8 | Storage (read_write) | next_cell_id |
//! | 9 | Storage (read_write) | nutrient_gain_rates |
//!
//! ### Mass Accumulation Bind Group (Group 1 for mass_accum shader)
//! | Binding | Type | Buffer |
//! |---------|------|--------|
//! | 0 | Storage (read) | nutrient_gain_rates |

use super::GpuTripleBufferSystem;

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
    
    // Cell type (16 bytes with padding)
    pub cell_type: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
}

/// Spatial query parameters for GPU spatial query compute shader
/// Size: 32 bytes (must match WGSL struct layout)
/// Uses ray origin and direction for ray-sphere intersection testing
#[repr(C, align(4))]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialQueryParams {
    pub ray_origin: [f32; 3],       // 12 bytes - camera position
    pub max_distance: f32,          // 4 bytes - maximum ray distance
    pub ray_direction: [f32; 3],    // 12 bytes - normalized ray direction
    pub _pad0: u32,                 // 4 bytes - padding
}                                   // Total: 32 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(std::mem::size_of::<SpatialQueryParams>() == 32, "SpatialQueryParams must be 32 bytes");

/// Spatial query result from GPU spatial query compute shader
/// Size: 16 bytes (must match WGSL struct layout)
/// Note: distance_fixed is stored as fixed-point u32 (multiply by 1000) for atomic operations
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialQueryResult {
    pub found_cell_index: u32,  // 4 bytes
    pub distance_fixed: u32,    // 4 bytes - fixed-point distance (actual = distance_fixed / 1000.0)
    pub found: u32,             // 4 bytes
    pub _padding: u32,          // 4 bytes
}                               // Total: 16 bytes

impl SpatialQueryResult {
    /// Get the distance as f32 (converts from fixed-point)
    pub fn distance(&self) -> f32 {
        self.distance_fixed as f32 / 1000.0
    }
}

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(std::mem::size_of::<SpatialQueryResult>() == 16, "SpatialQueryResult must be 16 bytes");

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
}                               // Total: 32 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(std::mem::size_of::<PositionUpdateParams>() == 32, "PositionUpdateParams must be 32 bytes");

/// Cell removal parameters for GPU cell removal compute shader
/// Size: 16 bytes (must match WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellRemovalParams {
    pub cell_index: u32,        // 4 bytes at offset 0
    pub _pad0: u32,             // 4 bytes at offset 4
    pub _pad1: u32,             // 4 bytes at offset 8
    pub _pad2: u32,             // 4 bytes at offset 12
}                               // Total: 16 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(std::mem::size_of::<CellRemovalParams>() == 16, "CellRemovalParams must be 16 bytes");

/// Cell boost parameters for GPU cell boost compute shader
/// Size: 16 bytes (must match WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellBoostParams {
    pub cell_index: u32,        // 4 bytes at offset 0
    pub _pad0: u32,             // 4 bytes at offset 4
    pub _pad1: u32,             // 4 bytes at offset 8
    pub _pad2: u32,             // 4 bytes at offset 12
}                               // Total: 16 bytes

// Compile-time assertion to verify struct size matches WGSL
const _: () = assert!(std::mem::size_of::<CellBoostParams>() == 16, "CellBoostParams must be 16 bytes");

/// Cached bind groups for GPU physics pipeline
/// Pre-created for all 3 buffer indices to avoid per-frame allocation
pub struct CachedBindGroups {
    /// Physics bind groups for each buffer index [0, 1, 2]
    pub physics: [wgpu::BindGroup; 3],
    /// Spatial grid bind group (same for all frames)
    pub spatial_grid: wgpu::BindGroup,
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
    /// Lifecycle adhesion bind group (same for all frames)
    pub lifecycle_adhesion: wgpu::BindGroup,
    /// Division scan adhesion bind group (read-only, same for all frames)
    pub division_scan_adhesion: wgpu::BindGroup,
    /// Clear forces bind group (same for all frames)
    pub clear_forces: wgpu::BindGroup,
    /// Collision force accum bind groups for each buffer index [0, 1, 2]
    pub collision_force_accum: [wgpu::BindGroup; 3],
    /// Position update force accum bind group (same for all frames)
    pub position_update_force_accum: wgpu::BindGroup,
    /// Velocity update angular bind group (same for all frames)
    pub velocity_update_angular: wgpu::BindGroup,
    /// Nutrient system bind group (same for all frames)
    pub nutrient_system: wgpu::BindGroup,
    /// Nutrient transport bind group (same for all frames, includes mode properties)
    pub nutrient_transport: wgpu::BindGroup,
    /// Swim force force accumulation bind groups for each buffer index [0, 1, 2]
    pub swim_force_force_accum: [wgpu::BindGroup; 3],
    /// Swim force cell data bind group (same for all frames)
    pub swim_force_cell_data: wgpu::BindGroup,
}

/// GPU physics compute pipelines
pub struct GpuPhysicsPipelines {
    pub spatial_grid_clear: wgpu::ComputePipeline,
    pub spatial_grid_assign: wgpu::ComputePipeline,
    pub spatial_grid_insert: wgpu::ComputePipeline,
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
    
    // Nutrient system pipeline
    pub nutrient_transport: wgpu::ComputePipeline,
    
    // Adhesion physics pipeline
    pub adhesion_physics: wgpu::ComputePipeline,
    
    // Lifecycle pipelines for cell division
    pub lifecycle_death_scan: wgpu::ComputePipeline,
    pub lifecycle_prefix_sum: wgpu::ComputePipeline,
    pub lifecycle_division_scan: wgpu::ComputePipeline,
    pub lifecycle_division_execute: wgpu::ComputePipeline,
    
    // Adhesion cleanup pipeline (runs after death scan)
    pub adhesion_cleanup: wgpu::ComputePipeline,
    
    // Swim force pipeline (applies thrust for Flagellocyte cells)
    pub swim_force: wgpu::ComputePipeline,
    
    // Bind group layouts
    pub physics_layout: wgpu::BindGroupLayout,
    pub spatial_grid_layout: wgpu::BindGroupLayout,
    pub lifecycle_layout: wgpu::BindGroupLayout,
    pub cell_state_read_layout: wgpu::BindGroupLayout,
    pub cell_state_write_layout: wgpu::BindGroupLayout,
    pub mass_accum_layout: wgpu::BindGroupLayout,
    pub rotations_layout: wgpu::BindGroupLayout,
    pub position_update_rotations_layout: wgpu::BindGroupLayout,
    
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
    
    // Swim force bind group layouts
    pub swim_force_force_accum_layout: wgpu::BindGroupLayout,
    pub swim_force_cell_data_layout: wgpu::BindGroupLayout,
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
        let position_update_rotations_layout = Self::create_position_update_rotations_bind_group_layout(device);
        
        // Create adhesion bind group layouts
        let adhesion_layout = Self::create_adhesion_bind_group_layout(device);
        let force_accum_layout = Self::create_force_accum_bind_group_layout(device);
        let lifecycle_adhesion_layout = Self::create_lifecycle_adhesion_bind_group_layout(device);
        let division_scan_adhesion_layout = Self::create_division_scan_adhesion_bind_group_layout(device);
        
        // Create new bind group layouts for force accumulation pipeline
        let clear_forces_layout = Self::create_clear_forces_bind_group_layout(device);
        let collision_force_accum_layout = Self::create_collision_force_accum_bind_group_layout(device);
        let position_update_force_accum_layout = Self::create_position_update_force_accum_bind_group_layout(device);
        let velocity_update_angular_layout = Self::create_velocity_update_angular_bind_group_layout(device);
        
        // Create nutrient transport bind group layouts
        let nutrient_system_layout = Self::create_nutrient_system_bind_group_layout(device);
        let nutrient_transport_layout = Self::create_nutrient_transport_bind_group_layout(device);
        
        // Create swim force bind group layouts
        let swim_force_force_accum_layout = Self::create_swim_force_force_accum_bind_group_layout(device);
        let swim_force_cell_data_layout = Self::create_swim_force_cell_data_bind_group_layout(device);
        
        // Create cell insertion bind group layouts
        let cell_insertion_physics_layout = Self::create_cell_insertion_physics_bind_group_layout(device);
        let cell_insertion_params_layout = Self::create_cell_insertion_params_bind_group_layout(device);
        let cell_insertion_state_layout = Self::create_cell_insertion_state_bind_group_layout(device);
        
        // Create cell data extraction bind group layouts
        let cell_extraction_params_layout = Self::create_cell_extraction_params_bind_group_layout(device);
        let cell_extraction_state_layout = Self::create_cell_extraction_state_bind_group_layout(device);
        let cell_extraction_output_layout = Self::create_cell_extraction_output_bind_group_layout(device);
        
        // Create spatial query bind group layouts
        let spatial_query_params_layout = Self::create_spatial_query_params_bind_group_layout(device);
        let spatial_query_result_layout = Self::create_spatial_query_result_bind_group_layout(device);
        
        // Create position update bind group layouts
        let position_update_params_layout = Self::create_position_update_params_bind_group_layout(device);
        
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
            &[&physics_layout, &spatial_grid_layout, &collision_force_accum_layout],
            "Collision Detection",
        );
        
        let position_update = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/position_update.wgsl"),
            "main",
            &[&physics_layout, &position_update_rotations_layout, &position_update_force_accum_layout],
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
            &[&cell_insertion_physics_layout, &cell_insertion_params_layout, &cell_insertion_state_layout],
            "Cell Insertion",
        );
        
        // Create cell data extraction pipeline
        let cell_data_extraction = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/extract_cell_data.wgsl"),
            "main",
            &[&physics_layout, &cell_extraction_params_layout, &cell_extraction_state_layout, &cell_extraction_output_layout],
            "Cell Data Extraction",
        );
        
        // Create spatial query pipeline
        let spatial_query = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/spatial_query.wgsl"),
            "main",
            &[&physics_layout, &spatial_query_params_layout, &spatial_query_result_layout],
            "Spatial Query",
        );
        
        // Create position update pipeline
        // Uses cell_insertion_physics_layout to access all 3 triple-buffered position/velocity sets
        let position_update_tool = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/update_position.wgsl"),
            "main",
            &[&cell_insertion_physics_layout, &position_update_params_layout],
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
        
        // Create nutrient transport pipeline
        let nutrient_transport = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/nutrient_transport.wgsl"),
            "main",
            &[&physics_layout, &nutrient_system_layout, &adhesion_layout, &nutrient_transport_layout],
            "Nutrient Transport",
        );
        
        // Create adhesion physics pipeline (per-cell processing, accumulates to force buffers)
        let adhesion_physics = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/adhesion_physics.wgsl"),
            "main",
            &[&physics_layout, &adhesion_layout, &rotations_layout, &force_accum_layout],
            "Adhesion Physics",
        );
        
        // Lifecycle pipelines
        let lifecycle_death_scan = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_death_scan.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout],
            "Lifecycle Death Scan",
        );
        
        let lifecycle_prefix_sum = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_prefix_sum.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout],
            "Lifecycle Prefix Sum",
        );
        
        let lifecycle_division_scan = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_division_scan.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout, &cell_state_read_layout, &division_scan_adhesion_layout],
            "Lifecycle Division Scan",
        );
        
        let lifecycle_division_execute = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/lifecycle_division_execute.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout, &cell_state_write_layout, &lifecycle_adhesion_layout],
            "Lifecycle Division Execute",
        );
        
        // Adhesion cleanup pipeline (runs after death scan, before prefix sum)
        let adhesion_cleanup = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/adhesion_cleanup.wgsl"),
            "main",
            &[&physics_layout, &lifecycle_layout, &lifecycle_adhesion_layout],
            "Adhesion Cleanup",
        );
        
        // Swim force pipeline (applies thrust for Flagellocyte cells)
        let swim_force = Self::create_compute_pipeline(
            device,
            include_str!("../../../shaders/swim_force.wgsl"),
            "main",
            &[&physics_layout, &swim_force_force_accum_layout, &swim_force_cell_data_layout],
            "Swim Force",
        );
        
        Self {
            spatial_grid_clear,
            spatial_grid_assign,
            spatial_grid_insert,
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
            adhesion_physics,
            lifecycle_death_scan,
            lifecycle_prefix_sum,
            lifecycle_division_scan,
            lifecycle_division_execute,
            adhesion_cleanup,
            swim_force,
            physics_layout,
            spatial_grid_layout,
            lifecycle_layout,
            cell_state_read_layout,
            cell_state_write_layout,
            mass_accum_layout,
            rotations_layout,
            position_update_rotations_layout,
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
            division_scan_adhesion_layout,
            clear_forces_layout,
            collision_force_accum_layout,
            position_update_force_accum_layout,
            velocity_update_angular_layout,
            nutrient_system_layout,
            nutrient_transport_layout,
            swim_force_force_accum_layout,
            swim_force_cell_data_layout,
        }
    }
    
    /// Create bind groups for the current frame
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
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
            ],
        });
        
        (physics_bind_group, spatial_grid_bind_group)
    }
    
    /// Create lifecycle bind group for division pipeline
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
                    resource: buffers.free_slot_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.division_slot_assignments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.lifecycle_counts.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create cell state bind group (read-only version for division scan)
    pub fn create_cell_state_read_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
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
                    resource: buffers.split_masses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.mode_cell_types.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create cell state bind group (read-write version for division execute)
    pub fn create_cell_state_write_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuTripleBufferSystem,
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
                    resource: buffers.split_masses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.next_cell_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.max_cell_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.rotations[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.rotations[output_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.genome_mode_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: buffers.parent_make_adhesion_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: buffers.child_mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: buffers.mode_properties.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create mass accumulation bind group (nutrient gain rates and max cell sizes per cell)
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
                    resource: buffers.max_cell_sizes.as_entire_binding(),
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
    ) -> CachedBindGroups {
        // Create physics bind groups for all 3 buffer indices
        let physics = [
            self.create_physics_bind_group_for_index(device, buffers, 0),
            self.create_physics_bind_group_for_index(device, buffers, 1),
            self.create_physics_bind_group_for_index(device, buffers, 2),
        ];
        
        // Spatial grid bind group (same for all frames)
        let spatial_grid = self.create_spatial_grid_bind_group_internal(device, buffers);
        
        // Lifecycle bind group (same for all frames)
        let lifecycle = self.create_lifecycle_bind_group(device, buffers);
        
        // Cell state read bind group (same for all frames)
        let cell_state_read = self.create_cell_state_read_bind_group(device, buffers);
        
        // Cell state write bind groups for all 3 buffer indices
        let cell_state_write = [
            self.create_cell_state_write_bind_group(device, buffers, 0),
            self.create_cell_state_write_bind_group(device, buffers, 1),
            self.create_cell_state_write_bind_group(device, buffers, 2),
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
        let adhesion = self.create_adhesion_bind_group(device, adhesion_buffers);
        
        // Force accumulation bind group (same for all frames)
        let force_accum = self.create_force_accum_bind_group(device, adhesion_buffers);
        
        // Lifecycle adhesion bind group (same for all frames)
        let lifecycle_adhesion = self.create_lifecycle_adhesion_bind_group(device, adhesion_buffers, buffers);
        
        // Division scan adhesion bind group (read-only, same for all frames)
        let division_scan_adhesion = self.create_division_scan_adhesion_bind_group(device, adhesion_buffers, buffers);
        
        // Clear forces bind group (same for all frames)
        let clear_forces = self.create_clear_forces_bind_group(device, adhesion_buffers);
        
        // Collision force accum bind groups (one for each buffer index)
        let collision_force_accum = [
            self.create_collision_force_accum_bind_group(device, adhesion_buffers, buffers, 0),
            self.create_collision_force_accum_bind_group(device, adhesion_buffers, buffers, 1),
            self.create_collision_force_accum_bind_group(device, adhesion_buffers, buffers, 2),
        ];
        
        // Position update force accum bind group (same for all frames)
        let position_update_force_accum = self.create_position_update_force_accum_bind_group(device, adhesion_buffers, buffers);
        
        // Velocity update angular bind group (same for all frames)
        let velocity_update_angular = self.create_velocity_update_angular_bind_group(device, adhesion_buffers, buffers);
        
        // Nutrient transport bind groups (same for all frames)
        let nutrient_system = self.create_nutrient_system_bind_group(device, buffers);
        let nutrient_transport = self.create_nutrient_transport_bind_group(device, buffers);
        
        // Swim force bind groups
        let swim_force_force_accum = [
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 0),
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 1),
            self.create_swim_force_force_accum_bind_group(device, adhesion_buffers, buffers, 2),
        ];
        let swim_force_cell_data = self.create_swim_force_cell_data_bind_group(device, buffers);
        
        CachedBindGroups {
            physics,
            spatial_grid,
            lifecycle,
            cell_state_read,
            cell_state_write,
            mass_accum,
            rotations,
            position_update_rotations,
            adhesion,
            force_accum,
            lifecycle_adhesion,
            division_scan_adhesion,
            clear_forces,
            collision_force_accum,
            position_update_force_accum,
            velocity_update_angular,
            nutrient_system,
            nutrient_transport,
            swim_force_force_accum,
            swim_force_cell_data,
        }
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
    ) -> wgpu::BindGroup {
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
            ],
        })
    }
    
    /// Create lifecycle bind group layout for division pipeline
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
                // Free slot indices
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
                // Lifecycle counts
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
    fn create_cell_state_bind_group_layout(device: &wgpu::Device, read_only: bool) -> wgpu::BindGroupLayout {
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
                    // Max splits
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
                    // Cell types (0 = Test, 1 = Flagellocyte) - DEPRECATED, use mode_cell_types
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
                    // Mode indices (per-cell mode index)
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
                    // Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
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
                    // Max splits
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
                    // Genome IDs
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
                    // Mode indices
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
                    // Cell IDs
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
                    // Next cell ID
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
                    // Nutrient gain rates
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
                    // Max cell sizes
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
                    // Stiffnesses (membrane stiffness per cell)
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
                    // Rotations input (read-only, from current buffer)
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
                    // Rotations output (read-write, to next buffer)
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
                    // Genome mode data (child orientations)
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
                    // Parent make adhesion flags
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
                    // Child mode indices (child_a_mode, child_b_mode per mode)
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
                    // Mode properties (nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass per mode)
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
                ],
            })
        }
    }
    
    /// Create mass accumulation bind group layout (nutrient gain rates and max cell sizes per cell)
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
                // Max cell sizes (read-only)
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
            ],
        })
    }
    
    /// Create position update rotations bind group layout (different from adhesion physics)
    fn create_position_update_rotations_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
                // Binding 0: Adhesion connections (read-only for per-cell processing)
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
                // Binding 1: Adhesion settings (read-only)
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
                // Binding 2: Adhesion counts (read-only for per-cell processing)
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
                // Binding 3: Cell adhesion indices (read-only, 20 indices per cell)
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
    
    /// Create lifecycle adhesion bind group layout (Group 3 in lifecycle division execute shader)
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
                // Binding 4: Child A keep adhesion flags (read-only)
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
                // Binding 5: Child B keep adhesion flags (read-only)
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
    
    /// Create division scan adhesion bind group layout (Group 3 in lifecycle division scan shader)
    /// Read-only access to adhesion data for checking neighbor division status
    fn create_division_scan_adhesion_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
                    resource: adhesion_buffers.adhesion_settings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.adhesion_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
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
    
    /// Create lifecycle adhesion bind group (Group 3 in lifecycle division execute shader)
    fn create_lifecycle_adhesion_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.child_a_keep_adhesion_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: triple_buffers.child_b_keep_adhesion_flags.as_entire_binding(),
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
    fn create_collision_force_accum_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
            ],
        })
    }
    
    /// Create position update force accum bind group layout (Group 2 in position_update shader)
    fn create_position_update_force_accum_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
            ],
        })
    }
    
    /// Create velocity update angular bind group layout (Group 1 in velocity_update shader)
    fn create_velocity_update_angular_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
                // Binding 3: Angular velocities buffer (read-write)
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
                // Binding 4: Rotations buffer (read-write)
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
    
    /// Create collision force accum bind group (Group 2 in collision_detection shader)
    fn create_collision_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Collision Force Accum Bind Group {}", buffer_index)),
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
            ],
        })
    }
    
    /// Create position update force accum bind group (Group 2 in position_update shader)
    fn create_position_update_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
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
            ],
        })
    }
    
    /// Create velocity update angular bind group (Group 1 in velocity_update shader)
    fn create_velocity_update_angular_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
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
                    resource: adhesion_buffers.angular_velocities[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: triple_buffers.rotations[0].as_entire_binding(),
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
                // Binding 2: Mode properties (read-only, nutrient priorities and swim forces)
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
    ) -> wgpu::BindGroup {
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
                    resource: buffers.mode_properties.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Create cell insertion physics bind group layout (Group 0 in cell_insertion shader)
    /// Contains all 3 triple-buffered position and velocity buffers for writing to all sets
    fn create_cell_insertion_physics_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_cell_insertion_params_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
            ],
        })
    }
    
    /// Create cell insertion state bind group layout (Group 2 in cell_insertion shader)
    fn create_cell_insertion_state_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
                // Binding 4: Max splits
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
                // Binding 5: Genome IDs
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
                // Binding 6: Mode indices
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
                // Binding 7: Cell IDs
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
                // Binding 8: Next cell ID (atomic)
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
                // Binding 9: Nutrient gain rates
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
                // Binding 10: Max cell sizes
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
                // Binding 11: Stiffnesses
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
                // Binding 12: Death flags
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
                // Binding 13: Division flags
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
                // Binding 14: Cell types
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
            ],
        })
    }
    
    /// Create cell extraction params bind group layout (Group 1 in extract_cell_data shader)
    fn create_cell_extraction_params_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_cell_extraction_state_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
                // Binding 4: Max splits (read-only)
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
                // Binding 5: Genome IDs (read-only)
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
                // Binding 6: Mode indices (read-only)
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
                // Binding 7: Cell IDs (read-only)
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
                // Binding 8: Nutrient gain rates (read-only)
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
                // Binding 9: Max cell sizes (read-only)
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
                // Binding 10: Stiffnesses (read-only)
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
            ],
        })
    }
    
    /// Create cell extraction output bind group layout (Group 3 in extract_cell_data shader)
    fn create_cell_extraction_output_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_spatial_query_params_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_spatial_query_result_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_position_update_params_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    fn create_cell_removal_params_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    /// Contains force accumulators (x, y, z) and rotations buffer
    fn create_swim_force_force_accum_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
            ],
        })
    }
    
    /// Create swim force cell data bind group layout (Group 2 in swim_force shader)
    /// Contains mode_indices, cell_types, mode_properties, and mode_cell_types
    fn create_swim_force_cell_data_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
                // Binding 2: Mode properties (read-only)
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
                // Binding 3: Mode cell types lookup table (read-only)
                // mode_cell_types[mode_index] = cell_type - always up-to-date with genome settings
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
    
    /// Create swim force force accumulation bind group (Group 1 in swim_force shader)
    fn create_swim_force_force_accum_bind_group(
        &self,
        device: &wgpu::Device,
        adhesion_buffers: &super::AdhesionBuffers,
        triple_buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Swim Force Force Accum Bind Group {}", buffer_index)),
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
            ],
        })
    }
    
    /// Create swim force cell data bind group (Group 2 in swim_force shader)
    fn create_swim_force_cell_data_bind_group(
        &self,
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
    ) -> wgpu::BindGroup {
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
                    resource: triple_buffers.mode_properties.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: triple_buffers.mode_cell_types.as_entire_binding(),
                },
            ],
        })
    }
}

/// Cell data extraction bind group layouts
pub struct CellDataExtractionLayouts {
    pub physics_layout: wgpu::BindGroupLayout,
    pub params_layout: wgpu::BindGroupLayout,
    pub state_layout: wgpu::BindGroupLayout,
    pub output_layout: wgpu::BindGroupLayout,
}