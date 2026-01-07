//! GPU Scene Integration
//! 
//! Executes the GPU physics pipeline using compute shaders.
//! 
//! ## Pipeline Stages (all use 256-thread workgroups for optimal GPU occupancy)
//! 1. Clear spatial grid
//! 2. Assign cells to grid
//! 3. Insert cells into grid
//! 4. Collision detection (optimized with pre-computed neighbor indices)
//! 5. Adhesion physics (spring-damper forces between adhered cells)
//! 6. Position integration
//! 7. Velocity integration
//! 8. Mass accumulation - mass increase based on nutrient_gain_rate
//! 
//! ## Lifecycle Pipeline (128-thread workgroups)
//! 1. Death scan - identify dead cells
//! 1.5. Adhesion cleanup - remove adhesions for dead cells, update per-cell indices
//! 2. Prefix sum - compact free slots
//! 3. Division scan - identify dividing cells
//! 4. Division execute - create child cells
//! 
//! ## Performance Optimizations
//! - Unified 256-thread workgroups (8 warps = optimal GPU scheduling)
//! - Pre-computed neighbor grid indices in collision detection
//! - Branchless boundary checks using clamp + select
//! - Cached bind groups (created once, not per-frame)
//! - NO CPU readback during simulation
//! - Triple buffering for lock-free GPU computation
//! - 64³ spatial grid for collision acceleration

use super::{CachedBindGroups, GpuPhysicsPipelines, GpuTripleBufferSystem};

/// Physics parameters for GPU uniform buffer (256-byte aligned)
/// 
/// Must match PhysicsParams struct in all WGSL shaders exactly.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PhysicsParams {
    // Time and frame info (16 bytes)
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    
    // World and physics settings (16 bytes)
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    
    // Grid settings (16 bytes)
    grid_resolution: i32,      // 64 for 64³ grid
    grid_cell_size: f32,       // world_size / 64
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    
    // Capacity for division (16 bytes - cell_capacity + 3 padding floats)
    cell_capacity: u32,        // Maximum cells that can exist
    _padding2: [f32; 3],       // Explicit padding to 16 bytes
    
    // Padding to 256 bytes (192 bytes = 48 floats)
    _padding: [f32; 48],
}

/// Grid resolution: 64³ = 262,144 grid cells
const GRID_RESOLUTION: u32 = 64;

/// Workgroup size for grid operations (optimal for 64³ grid)
const WORKGROUP_SIZE_GRID: u32 = 256;

/// Workgroup size for cell operations (unified for optimal GPU occupancy)
/// 256 threads = 8 warps = optimal for GPU scheduling
const WORKGROUP_SIZE_CELLS: u32 = 256;

/// Workgroup size for lifecycle operations (moderate complexity)
const WORKGROUP_SIZE_LIFECYCLE: u32 = 128;

/// Execute the 8-stage GPU physics pipeline
pub fn execute_gpu_physics_step(
    _device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &mut GpuTripleBufferSystem,
    cached_bind_groups: &CachedBindGroups,
    delta_time: f32,
    current_time: f32,
) {
    // Rotate to next buffer set
    let current_index = triple_buffers.rotate_buffers();
    
    // Update physics params uniform buffer
    // Note: cell_count is now read from cell_count_buffer by shaders
    let world_size = 100.0_f32;
    let params = PhysicsParams {
        delta_time,
        current_time,
        current_frame: 0,
        cell_count: 0, // Placeholder - shaders read from cell_count_buffer
        world_size,
        boundary_stiffness: 500.0,
        gravity: 0.0,
        acceleration_damping: 0.98,
        grid_resolution: GRID_RESOLUTION as i32,
        grid_cell_size: world_size / GRID_RESOLUTION as f32,
        max_cells_per_grid: 16,
        enable_thrust_force: 0,
        cell_capacity: triple_buffers.capacity,
        _padding2: [0.0; 3],
        _padding: [0.0; 48],
    };
    queue.write_buffer(&triple_buffers.physics_params, 0, bytemuck::bytes_of(&params));
    
    // Use cached bind groups (no per-frame allocation!)
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let spatial_grid_bind_group = &cached_bind_groups.spatial_grid;
    let adhesion_rotations_bind_group = &cached_bind_groups.rotations[current_index];
    let position_update_rotations_bind_group = &cached_bind_groups.position_update_rotations[current_index];
    let mass_accum_bind_group = &cached_bind_groups.mass_accum;
    
    // Calculate workgroup counts - dispatch for capacity, shaders check cell_count
    // All cell operations use unified 256-thread workgroups for optimal GPU occupancy
    let total_grid_cells = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
    let grid_workgroups = (total_grid_cells + WORKGROUP_SIZE_GRID - 1) / WORKGROUP_SIZE_GRID;
    let cell_workgroups = (triple_buffers.capacity + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
    
    // Execute the 8 compute shader stages
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Physics Pipeline"),
            timestamp_writes: None,
        });
        
        // Stage 1: Clear spatial grid (256 threads)
        compute_pass.set_pipeline(&pipelines.spatial_grid_clear);
        compute_pass.set_bind_group(0, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(grid_workgroups, 1, 1);
        
        // Stage 2: Assign cells to grid (256 threads)
        compute_pass.set_pipeline(&pipelines.spatial_grid_assign);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 3: Insert cells into grid (256 threads)
        compute_pass.set_pipeline(&pipelines.spatial_grid_insert);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 3.5: Clear force accumulators (256 threads)
        // Must run before collision detection and adhesion physics
        compute_pass.set_pipeline(&pipelines.clear_forces);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.clear_forces, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 4: Collision detection (256 threads - optimized with pre-computed neighbors)
        // Now accumulates forces to force_accum instead of applying directly
        compute_pass.set_pipeline(&pipelines.collision_detection);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.collision_force_accum, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 5: Adhesion physics (64 threads - dispatch per cell)
        // Now processes adhesions per-cell to avoid race conditions
        // Each thread handles one cell and accumulates all its adhesion forces locally
        compute_pass.set_pipeline(&pipelines.adhesion_physics);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
        compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.force_accum, &[]);
        // Dispatch based on cell count (one thread per cell)
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 6: Position integration (256 threads)
        // Now reads accumulated forces and applies them with proper integration
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, position_update_rotations_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.position_update_force_accum, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 7: Angular velocity integration (256 threads)
        // Applies accumulated torques to angular velocities and rotations
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.velocity_update_angular, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 8: Mass accumulation (256 threads)
        compute_pass.set_pipeline(&pipelines.mass_accum);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, mass_accum_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
}

/// Execute the lifecycle pipeline for cell division
/// This is called after the main physics pipeline when division is enabled
pub fn execute_lifecycle_pipeline(
    _device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    _queue: &wgpu::Queue,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &mut GpuTripleBufferSystem,
    cached_bind_groups: &CachedBindGroups,
    _current_time: f32,
) {
    // Get current buffer index (already rotated by physics pipeline)
    let current_index = triple_buffers.current_index();
    
    // Use cached bind groups (no per-frame allocation!)
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let lifecycle_bind_group = &cached_bind_groups.lifecycle;
    let cell_state_read_bind_group = &cached_bind_groups.cell_state_read;
    let cell_state_write_bind_group = &cached_bind_groups.cell_state_write[current_index];
    
    // Calculate workgroup counts - dispatch for capacity, shaders check cell_count
    let cell_workgroups_lifecycle = (triple_buffers.capacity + WORKGROUP_SIZE_LIFECYCLE - 1) / WORKGROUP_SIZE_LIFECYCLE;
    
    // Execute lifecycle pipeline
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline"),
            timestamp_writes: None,
        });
        
        // Stage 1: Death scan - identify dead cells (128 threads)
        compute_pass.set_pipeline(&pipelines.lifecycle_death_scan);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
        
        // Stage 1.5: Adhesion cleanup - remove adhesions for dead cells (128 threads)
        // This must run after death scan so we know which cells are dead
        // and before prefix sum so freed adhesion slots can be reused
        compute_pass.set_pipeline(&pipelines.adhesion_cleanup);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.lifecycle_adhesion, &[]);
        // Dispatch based on adhesion capacity, shader checks actual count
        let adhesion_workgroups = (100_000 + WORKGROUP_SIZE_LIFECYCLE - 1) / WORKGROUP_SIZE_LIFECYCLE;
        compute_pass.dispatch_workgroups(adhesion_workgroups, 1, 1);
        
        // Stage 2: Prefix sum - compact free slots (128 threads)
        compute_pass.set_pipeline(&pipelines.lifecycle_prefix_sum);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
        
        // Stage 3: Division scan - identify cells ready to divide (128 threads)
        compute_pass.set_pipeline(&pipelines.lifecycle_division_scan);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, cell_state_read_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
        
        // Stage 4: Division execute - create child cells (128 threads)
        compute_pass.set_pipeline(&pipelines.lifecycle_division_execute);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, cell_state_write_bind_group, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.lifecycle_adhesion, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
    }
}
