//! GPU Scene Integration
//! 
//! Executes the GPU physics pipeline using compute shaders.
//! 
//! ## Pipeline Stages
//! 1. Clear spatial grid (256 workgroup)
//! 2. Assign cells to grid (64 workgroup)
//! 3. Insert cells into grid (64 workgroup)
//! 4. Collision detection (64 workgroup)
//! 5. Position integration (64 workgroup)
//! 6. Velocity integration (64 workgroup)
//! 7. Mass accumulation (64 workgroup) - mass increase based on nutrient_gain_rate
//! 
//! ## Lifecycle Pipeline (optional, for cell division)
//! 8. Death scan (64 workgroup) - identify dead cells
//! 9. Prefix sum (256 workgroup) - compact free slots
//! 10. Division scan (64 workgroup) - identify dividing cells
//! 11. Division execute (64 workgroup) - create child cells
//! 
//! ## Key Patterns
//! - NO atomics in shaders - uses prefix-sum compaction
//! - NO CPU readback during simulation
//! - Triple buffering for lock-free GPU computation
//! - 64³ spatial grid for collision acceleration

use super::{GpuPhysicsPipelines, GpuTripleBufferSystem};

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
    
    // Capacity for division (4 bytes + 12 bytes padding to align)
    cell_capacity: u32,        // Maximum cells that can exist
    _padding2: [f32; 3],       // Align to 16 bytes
    
    // Padding to 256 bytes (192 bytes)
    _padding: [f32; 48],
}

/// Grid resolution: 64³ = 262,144 grid cells
const GRID_RESOLUTION: u32 = 64;

/// Workgroup size for grid operations (optimal for 64³ grid)
const WORKGROUP_SIZE_GRID: u32 = 256;

/// Workgroup size for cell operations (good balance for cell iteration)
const WORKGROUP_SIZE_CELLS: u32 = 64;

/// Execute the 6-stage GPU physics pipeline
pub fn execute_gpu_physics_step(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &mut GpuTripleBufferSystem,
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
    
    // Create bind groups for this frame (includes cell_count_buffer)
    let (physics_bind_group, spatial_grid_bind_group) = pipelines.create_bind_groups(
        device,
        triple_buffers,
        current_index,
    );
    
    // Calculate workgroup counts - dispatch for capacity, shaders check cell_count
    let total_grid_cells = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
    let grid_workgroups = (total_grid_cells + WORKGROUP_SIZE_GRID - 1) / WORKGROUP_SIZE_GRID;
    let cell_workgroups = (triple_buffers.capacity + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
    
    // Execute the 6 compute shader stages
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Physics Pipeline"),
            timestamp_writes: None,
        });
        
        // Stage 1: Clear spatial grid
        compute_pass.set_pipeline(&pipelines.spatial_grid_clear);
        compute_pass.set_bind_group(0, &spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(grid_workgroups, 1, 1);
        
        // Stage 2: Assign cells to grid
        compute_pass.set_pipeline(&pipelines.spatial_grid_assign);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 3: Insert cells into grid (no-op pass for pipeline consistency)
        compute_pass.set_pipeline(&pipelines.spatial_grid_insert);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 4: Collision detection
        compute_pass.set_pipeline(&pipelines.collision_detection);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 5: Position integration (also propagates rotations)
        let rotations_bind_group = pipelines.create_rotations_bind_group(device, triple_buffers, current_index);
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &rotations_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 6: Velocity integration
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 7: Mass accumulation (mass increase based on nutrient_gain_rate)
        let mass_accum_bind_group = pipelines.create_mass_accum_bind_group(device, triple_buffers);
        compute_pass.set_pipeline(&pipelines.mass_accum);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &mass_accum_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
}

/// Execute the lifecycle pipeline for cell division
/// This is called after the main physics pipeline when division is enabled
pub fn execute_lifecycle_pipeline(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    _queue: &wgpu::Queue,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &mut GpuTripleBufferSystem,
    _current_time: f32,
) {
    // Get current buffer index (already rotated by physics pipeline)
    let current_index = triple_buffers.current_index();
    
    // NOTE: We do NOT update physics_params here - the physics pipeline already set it
    // and we don't want to overwrite delta_time with 0.0
    
    // Create bind groups (includes cell_count_buffer)
    let (physics_bind_group, _) = pipelines.create_bind_groups(
        device,
        triple_buffers,
        current_index,
    );
    let lifecycle_bind_group = pipelines.create_lifecycle_bind_group(device, triple_buffers);
    let cell_state_read_bind_group = pipelines.create_cell_state_read_bind_group(device, triple_buffers);
    let cell_state_write_bind_group = pipelines.create_cell_state_write_bind_group(device, triple_buffers, current_index);
    
    // Calculate workgroup counts - dispatch for capacity, shaders check cell_count
    let cell_workgroups = (triple_buffers.capacity + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
    
    // Execute lifecycle pipeline
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline"),
            timestamp_writes: None,
        });
        
        // Stage 1: Death scan - identify dead cells
        compute_pass.set_pipeline(&pipelines.lifecycle_death_scan);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &lifecycle_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 2: Prefix sum - compact free slots
        compute_pass.set_pipeline(&pipelines.lifecycle_prefix_sum);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &lifecycle_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 3: Division scan - identify cells ready to divide
        compute_pass.set_pipeline(&pipelines.lifecycle_division_scan);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, &cell_state_read_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 4: Division execute - create child cells
        compute_pass.set_pipeline(&pipelines.lifecycle_division_execute);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, &cell_state_write_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
}
