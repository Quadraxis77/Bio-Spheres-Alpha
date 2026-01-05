//! GPU Scene Integration
//! 
//! Executes the 6-stage GPU physics pipeline using compute shaders.
//! 
//! ## Pipeline Stages
//! 1. Clear spatial grid (256 workgroup)
//! 2. Assign cells to grid (64 workgroup)
//! 3. Insert cells into grid (64 workgroup) - placeholder for prefix-sum
//! 4. Collision detection (64 workgroup)
//! 5. Position integration (64 workgroup)
//! 6. Velocity integration (64 workgroup)
//! 
//! ## Key Patterns
//! - NO atomics in shaders - uses spatial grid filtering
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
    
    // Padding to 256 bytes (208 bytes)
    _padding: [f32; 52],
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
    cell_count: u32,
    delta_time: f32,
) {
    if cell_count == 0 {
        return;
    }
    
    // Rotate to next buffer set
    let current_index = triple_buffers.rotate_buffers();
    
    // Update physics params uniform buffer
    let world_size = 100.0_f32;
    let params = PhysicsParams {
        delta_time,
        current_time: 0.0,
        current_frame: 0,
        cell_count,
        world_size,
        boundary_stiffness: 500.0,
        gravity: 0.0,
        acceleration_damping: 0.98,
        grid_resolution: GRID_RESOLUTION as i32,
        grid_cell_size: world_size / GRID_RESOLUTION as f32,
        max_cells_per_grid: 16,
        enable_thrust_force: 0,
        _padding: [0.0; 52],
    };
    queue.write_buffer(&triple_buffers.physics_params, 0, bytemuck::bytes_of(&params));
    
    // Create bind groups for this frame
    let (physics_bind_group, spatial_grid_bind_group) = pipelines.create_bind_groups(
        device,
        triple_buffers,
        current_index,
    );
    
    // Calculate workgroup counts
    let total_grid_cells = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;
    let grid_workgroups = (total_grid_cells + WORKGROUP_SIZE_GRID - 1) / WORKGROUP_SIZE_GRID;
    let cell_workgroups = (cell_count + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
    
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
        
        // Stage 5: Position integration
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 6: Velocity integration
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, &physics_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
}
