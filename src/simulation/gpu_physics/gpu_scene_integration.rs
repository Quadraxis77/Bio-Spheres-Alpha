//! GPU Scene Integration
//! 
//! Executes the GPU physics pipeline using compute shaders.
//! 
//! ## Pipeline Stages (all use 256-thread workgroups for optimal GPU occupancy)
//! 0. DMA zero-fill: spatial grid counts + force/torque accumulators + mass deltas
//! 1. Build spatial grid (combined assign + insert, skips dead cells)
//! 2. Collision detection (optimized with pre-computed neighbor indices)
//! 3. Adhesion physics (spring-damper forces between adhered cells)
//! 3.5. Swim force (thrust for Flagellocyte cells based on rotation and swim_force setting)
//! 4. Position integration
//! 5. Velocity integration
//! 6. Mass accumulation - nutrient growth based on nutrient_gain_rate
//! 7. Nutrient transport - consumption, transport between cells, death detection
//! 
//! ## Lifecycle Pipeline (128-thread workgroups)
//! 1. Death scan - identify dead cells
//! 1.5. Adhesion cleanup - remove adhesions for dead cells, update per-cell indices
//! 2. Division scan - identify dividing cells
//! 3. Division execute - create child cells
//! 
//! ## Performance Optimizations
//! - DMA zero-fill replaces clear_forces compute dispatch (GPU DMA engine vs shader overhead)
//! - Combined spatial grid build (assign + insert in one dispatch, skips dead cells)
//! - Collision shader writes only force/torque atomics (no redundant pos/vel copy)
//! - Unified 256-thread workgroups (8 warps = optimal GPU scheduling)
//! - Pre-computed neighbor grid indices in collision detection
//! - Branchless boundary checks using clamp + select
//! - Cached bind groups (created once, not per-frame)
//! - NO CPU readback during simulation
//! - Triple buffering for lock-free GPU computation
//! - 128³ spatial grid for collision acceleration

use super::{CachedBindGroups, GpuPhysicsPipelines, GpuTripleBufferSystem};
// MAX_ADHESION_CONNECTIONS is now dynamic (adhesion_buffers.max_connections)

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
    
    // Capacity and gravity mode (16 bytes)
    cell_capacity: u32,        // Maximum cells that can exist
    gravity_mode: u32,         // 0=X, 1=Y, 2=Z, 3=radial (toward origin)
    angular_damping: f32,      // fraction of angular velocity retained per second (velocity_update.wgsl _pad1)
    _pad2: f32,
    
    // Padding to 256 bytes (192 bytes = 48 floats)
    _padding: [f32; 47],
}

/// Grid resolution: 128³ = 2,097,152 grid cells
const GRID_RESOLUTION: u32 = 128;

/// Workgroup size for cell operations (unified for optimal GPU occupancy)
/// 256 threads = 8 warps = optimal for GPU scheduling
const WORKGROUP_SIZE_CELLS: u32 = 256;

/// Workgroup size for lifecycle operations (moderate complexity)
const WORKGROUP_SIZE_LIFECYCLE: u32 = 128;

/// Workgroup size for adhesion cleanup (256 for optimal GPU occupancy)
const WORKGROUP_SIZE_ADHESION: u32 = 256;

/// Execute the 9-stage GPU physics pipeline (added nutrient transport)
pub fn execute_gpu_physics_step(
    _device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &mut GpuTripleBufferSystem,
    cached_bind_groups: &CachedBindGroups,
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    world_diameter: f32,
    gravity: f32,
    gravity_mode: u32,
    acceleration_damping: f32,
    cave_renderer: Option<&crate::rendering::CaveSystemRenderer>,
    cave_physics_bind_groups: Option<&[wgpu::BindGroup; 3]>,
    adhesion_buffers: &super::AdhesionBuffers,
    _cell_count_hint: u32,
    constraint_iterations: u32,
) {
    // Rotate to next buffer set
    let current_index = triple_buffers.rotate_buffers();

    // Update physics params uniform buffer
    // Note: cell_count is now read from cell_count_buffer by shaders
    // world_size is the diameter, boundary_radius = world_size * 0.5
    let world_size = world_diameter;
    let params = PhysicsParams {
        delta_time,
        current_time,
        current_frame,
        cell_count: 0, // Placeholder - shaders read from cell_count_buffer
        world_size,
        boundary_stiffness: 500.0,
        gravity,
        acceleration_damping,
        grid_resolution: GRID_RESOLUTION as i32,
        grid_cell_size: world_size / GRID_RESOLUTION as f32,
        max_cells_per_grid: 16,
        enable_thrust_force: 1, // Enable swim force for Flagellocyte cells
        cell_capacity: triple_buffers.capacity,
        gravity_mode,
        angular_damping: 0.94,
        _pad2: 0.0,
        _padding: [0.0; 47],
    };
    queue.write_buffer(&triple_buffers.physics_params, 0, bytemuck::bytes_of(&params));
    
    // DMA zero-fill all buffers that need clearing before the compute pass.
    // This replaces the clear_forces compute dispatch and is significantly faster
    // because the GPU's DMA engine handles bulk zeroing without compute shader overhead.
    // PERFORMANCE: Skip clearing when no cells - saves ~15MB DMA per physics step
    if _cell_count_hint > 0 {
        encoder.clear_buffer(&triple_buffers.mass_deltas_buffer, 0, None);
        encoder.clear_buffer(&triple_buffers.spatial_grid_counts, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_z, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_z, 0, None);
    }
    
    // Use cached bind groups (no per-frame allocation!)
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let spatial_grid_bind_group = &cached_bind_groups.spatial_grid;
    let adhesion_rotations_bind_group = &cached_bind_groups.rotations[current_index];
    let position_update_rotations_bind_group = &cached_bind_groups.position_update_rotations[current_index];
    let mass_accum_bind_group = &cached_bind_groups.mass_accum;
    
    // PERFORMANCE: Dispatch based on actual cell count, not full capacity.
    // At 100K cells, this reduces dispatch from 1024 to ~390 workgroups (2.6x reduction).
    // Use max(live_count, 1) to avoid zero dispatch, and add small buffer for divisions.
    let effective_cell_count = (_cell_count_hint.max(1) + 255) / 256 * 256; // Round up to workgroup boundary
    let cell_workgroups = (effective_cell_count + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
    
    // Execute compute shader stages
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Physics Pipeline"),
            timestamp_writes: None,
        });
        
        // Stage 2+3: Build spatial grid (combined assign + insert + dead cell skip)
        // Single dispatch replaces the previous two-pass approach, and skips dead cells
        // to avoid wasting grid slots and collision checks against dead neighbors.
        compute_pass.set_pipeline(&pipelines.spatial_grid_build);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 4: Collision detection (256 threads - optimized with pre-computed neighbors)
        // Now accumulates forces to force_accum instead of applying directly
        // Also applies boundary torque to rotate cells toward center
        compute_pass.set_pipeline(&pipelines.collision_detection);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.collision_force_accum[current_index], &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 5: Adhesion physics (256 threads per workgroup, per-cell processing)
        // Each thread handles ONE cell and iterates through its adhesion indices
        // Forces are accumulated to force buffers (same as collision detection)
        compute_pass.set_pipeline(&pipelines.adhesion_physics);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
        compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.force_accum, &[]);
        // Dispatch based on cell count (per-cell processing)
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 5.5: Swim force (256 threads) - applies thrust for Flagellocyte cells
        // Accumulates swim force to force buffers based on cell rotation and mode swim_force setting
        compute_pass.set_pipeline(&pipelines.swim_force);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.swim_force_force_accum[current_index], &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 5.6: Glueocyte environment adhesion (when cave is present)
        // Must run BEFORE position_update so forces are accumulated and applied this frame
        if let (Some(cave_renderer), _) = (cave_renderer.as_ref(), cave_physics_bind_groups.as_ref()) {
            compute_pass.set_pipeline(&pipelines.glueocyte_env_adhesion);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.env_adhesion_force_accum[current_index], &[]);
            compute_pass.set_bind_group(2, &cached_bind_groups.env_adhesion_mode_data, &[]);
            compute_pass.set_bind_group(3, cave_renderer.collision_bind_group(), &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }
        
        // Stage 6: Position integration (256 threads)
        // Now reads accumulated forces and applies them with proper integration
        // Also handles buoyancy (cells float in water when fluid system is enabled)
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, position_update_rotations_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.position_update_force_accum, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.position_update_spatial_grid, &[]);
        // Water buffers are now part of bind group 2 (position_update_force_accum)
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 6.5: Cave collision (if enabled) - corrects positions using SDF
        if let (Some(cave_renderer), Some(cave_bind_groups)) = (cave_renderer, cave_physics_bind_groups) {
            if cave_renderer.params().collision_enabled != 0 {
                // SDF-based collision - no spatial grid building needed
                compute_pass.set_pipeline(cave_renderer.collision_pipeline());
                compute_pass.set_bind_group(0, &cave_bind_groups[current_index], &[]);
                compute_pass.set_bind_group(1, cave_renderer.collision_bind_group(), &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }
        
        // Stage 7: Angular velocity integration (256 threads)
        // Applies accumulated torques to angular velocities and rotations
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.velocity_update_angular, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 7.5: Adhesion constraint sub-stepping (N additional iterations)
        // Each iteration re-evaluates adhesion forces against latest positions and
        // applies corrections directly to output buffers. Dramatically increases
        // effective joint stiffness without changing spring constants.
        // Cave collision is re-applied after each substep to prevent adhesion forces
        // from pulling cells through cave walls.
        for _ in 0..constraint_iterations {
            compute_pass.set_pipeline(&pipelines.adhesion_substep);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
            compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            
            // Re-apply cave collision after each substep to enforce cave boundaries
            if let (Some(cave_renderer), Some(cave_bind_groups)) = (cave_renderer, cave_physics_bind_groups) {
                if cave_renderer.params().collision_enabled != 0 {
                    compute_pass.set_pipeline(cave_renderer.collision_pipeline());
                    compute_pass.set_bind_group(0, &cave_bind_groups[current_index], &[]);
                    compute_pass.set_bind_group(1, cave_renderer.collision_bind_group(), &[]);
                    compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
                }
            }
        }
        
        // Stage 8: Mass accumulation (256 threads) - nutrient growth only
        compute_pass.set_pipeline(&pipelines.mass_accum);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, mass_accum_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 9a: Nutrient transport (256 threads) - consumption + transport accumulation only
        // This dispatch accumulates mass deltas via atomics but does NOT apply them.
        compute_pass.set_pipeline(&pipelines.nutrient_transport);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.nutrient_system, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.adhesion, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.nutrient_transport, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
    
    // Stage 9b: Nutrient apply (separate compute pass)
    // CRITICAL: Must be a separate pass so all nutrient_transport workgroups finish
    // their atomic writes to mass_deltas before any thread reads the accumulated result.
    // Without this separation, cell_b's workgroup could read its mass_delta before
    // cell_a's workgroup writes the transport to it, causing asymmetric nutrient flow.
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Nutrient Apply"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipelines.nutrient_apply);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.nutrient_apply, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
}

/// Execute a mechanics-only physics step (no nutrient transport, no mass accumulation).
///
/// Runs stages 1–7 of the physics pipeline (spatial grid, clear forces, collision,
/// adhesion, swim force, position integration, angular velocity integration) but
/// **skips** mass accumulation (stage 8), nutrient transport (stage 9), and the
/// lifecycle pipeline entirely.
///
/// Used when the simulation is paused but a cell is being dragged so that
/// adhesion-connected cells can follow via spring forces without affecting
/// nutrient transport or cell split timing.
pub fn execute_gpu_mechanics_step(
    _device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &mut GpuTripleBufferSystem,
    cached_bind_groups: &CachedBindGroups,
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    world_diameter: f32,
    gravity: f32,
    gravity_mode: u32,
    acceleration_damping: f32,
    cave_renderer: Option<&crate::rendering::CaveSystemRenderer>,
    cave_physics_bind_groups: Option<&[wgpu::BindGroup; 3]>,
    adhesion_buffers: &super::AdhesionBuffers,
    _cell_count_hint: u32,
    constraint_iterations: u32,
) {
    // Rotate to next buffer set
    let current_index = triple_buffers.rotate_buffers();

    let world_size = world_diameter;
    let params = PhysicsParams {
        delta_time,
        current_time,
        current_frame,
        cell_count: 0,
        world_size,
        boundary_stiffness: 500.0,
        gravity,
        acceleration_damping,
        grid_resolution: GRID_RESOLUTION as i32,
        grid_cell_size: world_size / GRID_RESOLUTION as f32,
        max_cells_per_grid: 16,
        enable_thrust_force: 1,
        cell_capacity: triple_buffers.capacity,
        gravity_mode,
        angular_damping: 0.94,
        _pad2: 0.0,
        _padding: [0.0; 47],
    };
    queue.write_buffer(&triple_buffers.physics_params, 0, bytemuck::bytes_of(&params));

    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let spatial_grid_bind_group = &cached_bind_groups.spatial_grid;
    let adhesion_rotations_bind_group = &cached_bind_groups.rotations[current_index];
    let position_update_rotations_bind_group = &cached_bind_groups.position_update_rotations[current_index];

    // PERFORMANCE: Dispatch based on actual cell count, not full capacity
    let effective_cell_count = (_cell_count_hint.max(1) + 255) / 256 * 256;
    let cell_workgroups = (effective_cell_count + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;

    // DMA zero-fill spatial grid + force/torque buffers before compute pass
    // PERFORMANCE: Skip clearing when no cells - saves ~15MB DMA
    if _cell_count_hint > 0 {
        encoder.clear_buffer(&triple_buffers.spatial_grid_counts, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_z, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_z, 0, None);
    }

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU Mechanics-Only Pipeline (Drag)"),
            timestamp_writes: None,
        });

        // Stage 2+3: Build spatial grid (combined assign + insert + dead cell skip)
        compute_pass.set_pipeline(&pipelines.spatial_grid_build);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 4: Collision detection
        compute_pass.set_pipeline(&pipelines.collision_detection);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.collision_force_accum[current_index], &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 5: Adhesion physics
        compute_pass.set_pipeline(&pipelines.adhesion_physics);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
        compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.force_accum, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 5.5: Swim force
        compute_pass.set_pipeline(&pipelines.swim_force);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.swim_force_force_accum[current_index], &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 5.6: Glueocyte environment adhesion (when cave is present)
        // Must run BEFORE position_update so forces are accumulated and applied this frame
        if let (Some(cave_renderer), _) = (cave_renderer.as_ref(), cave_physics_bind_groups.as_ref()) {
            compute_pass.set_pipeline(&pipelines.glueocyte_env_adhesion);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.env_adhesion_force_accum[current_index], &[]);
            compute_pass.set_bind_group(2, &cached_bind_groups.env_adhesion_mode_data, &[]);
            compute_pass.set_bind_group(3, cave_renderer.collision_bind_group(), &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 6: Position integration
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, position_update_rotations_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.position_update_force_accum, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.position_update_spatial_grid, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 6.5: Cave collision (if enabled)
        if let (Some(cave_renderer), Some(cave_bind_groups)) = (cave_renderer, cave_physics_bind_groups) {
            if cave_renderer.params().collision_enabled != 0 {
                compute_pass.set_pipeline(cave_renderer.collision_pipeline());
                compute_pass.set_bind_group(0, &cave_bind_groups[current_index], &[]);
                compute_pass.set_bind_group(1, cave_renderer.collision_bind_group(), &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 7: Angular velocity integration
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.velocity_update_angular, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 7.5: Adhesion constraint sub-stepping (N additional iterations)
        // Cave collision is re-applied after each substep to prevent adhesion forces
        // from pulling cells through cave walls.
        for _ in 0..constraint_iterations {
            compute_pass.set_pipeline(&pipelines.adhesion_substep);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
            compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            
            // Re-apply cave collision after each substep to enforce cave boundaries
            if let (Some(cave_renderer), Some(cave_bind_groups)) = (cave_renderer, cave_physics_bind_groups) {
                if cave_renderer.params().collision_enabled != 0 {
                    compute_pass.set_pipeline(cave_renderer.collision_pipeline());
                    compute_pass.set_bind_group(0, &cave_bind_groups[current_index], &[]);
                    compute_pass.set_bind_group(1, cave_renderer.collision_bind_group(), &[]);
                    compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
                }
            }
        }

        // SKIP Stage 8: Mass accumulation (nutrient growth)
        // SKIP Stage 9: Nutrient transport (consumption, transport, death detection)
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
    adhesion_buffers: &super::AdhesionBuffers,
    _current_time: f32,
) {
    // Get current buffer index (already rotated by physics pipeline)
    let current_index = triple_buffers.current_index();
    
    // Use cached bind groups (no per-frame allocation!)
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let lifecycle_bind_group = &cached_bind_groups.lifecycle;
    let cell_state_read_bind_group = &cached_bind_groups.cell_state_read;
    let cell_state_write_bind_group = &cached_bind_groups.cell_state_write[current_index];
    
    // Always dispatch lifecycle at full capacity — the async readback hint lags 1-3 frames
    // behind the true GPU cell count, so using it here would cause death_scan/division_scan
    // to miss newly-divided cells and halt further splitting. Lifecycle shaders early-exit
    // on dead/empty slots so over-dispatch is cheap.
    let cell_workgroups_lifecycle = (triple_buffers.capacity + WORKGROUP_SIZE_LIFECYCLE - 1) / WORKGROUP_SIZE_LIFECYCLE;
    
    // Execute 3-stage lifecycle pipeline with ring buffer for slot recycling
    // Stage 1: Death scan - detects dead cells and pushes slots to ring buffer
    // Must complete BEFORE division scan so recycled slots are available
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline - Death Scan"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipelines.lifecycle_death_scan);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, cell_state_read_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
        
        drop(compute_pass);
    }
    
    // Stage 1.5: Adhesion cleanup (after death detection completes)
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline - Adhesion Cleanup"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipelines.adhesion_cleanup);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.lifecycle_adhesion, &[]);
        let adhesion_workgroups = (adhesion_buffers.max_connections + WORKGROUP_SIZE_ADHESION - 1) / WORKGROUP_SIZE_ADHESION;
        compute_pass.dispatch_workgroups(adhesion_workgroups, 1, 1);
        
        drop(compute_pass);
    }
    
    // Stage 2: Division scan - allocates slots from ring buffer for dividing cells
    // Runs AFTER death scan so recycled slots are available
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline - Division Scan"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipelines.lifecycle_division_scan);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, cell_state_read_bind_group, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.division_scan_adhesion, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
        
        drop(compute_pass);
    }
    
    // Stage 3: Division execute (uses pre-allocated slots from division scan)
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline - Division Execute Ring"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipelines.lifecycle_division_execute);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, lifecycle_bind_group, &[]);
        compute_pass.set_bind_group(2, cell_state_write_bind_group, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.division_execute_adhesion, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
    }
    
    // Signal system moved to execute_signal_system() - runs once per frame, not per physics step
    // This prevents 4x over-dispatch when running 4 physics steps per frame

    // After division execute, propagate the output buffer to the third (stale) triple buffer.
    // Division writes new child cells only to positions_out (output_idx). The third buffer
    // (stale_idx) still holds old dead-cell data at recycled slot indices. Two physics steps
    // later that stale buffer rotates back into positions_in, causing ghost flickering.
    // Copying output → stale keeps all 3 triple buffers consistent after every division.
    let output_idx = (current_index + 1) % 3;
    let stale_idx  = (current_index + 2) % 3;
    let buf_size   = triple_buffers.capacity as u64 * 16; // Vec4<f32> = 16 bytes per slot
    encoder.copy_buffer_to_buffer(
        &triple_buffers.position_and_mass[output_idx], 0,
        &triple_buffers.position_and_mass[stale_idx],  0,
        buf_size,
    );
    encoder.copy_buffer_to_buffer(
        &triple_buffers.velocity[output_idx], 0,
        &triple_buffers.velocity[stale_idx],  0,
        buf_size,
    );
    encoder.copy_buffer_to_buffer(
        &triple_buffers.rotations[output_idx], 0,
        &triple_buffers.rotations[stale_idx],  0,
        buf_size,
    );

    // Sync next_adhesion_id to adhesion_counts[0] so adhesion line renderer sees GPU-created adhesions
    // The division shader atomically increments next_adhesion_id when creating adhesions,
    // but the adhesion line shader reads from adhesion_counts[0] for the total count.
    encoder.copy_buffer_to_buffer(
        &adhesion_buffers.next_adhesion_id,
        0,
        &adhesion_buffers.adhesion_counts,
        0,  // offset 0 = total_adhesion_count
        4,  // 4 bytes (u32)
    );
}

/// Rebuild spatial grid after lifecycle pipeline
/// NOTE: This is no longer called from the main physics loop since the grid
/// is rebuilt at the start of each physics step. Kept for potential future use.
pub fn rebuild_spatial_grid_after_lifecycle(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &GpuTripleBufferSystem,
    cached_bind_groups: &CachedBindGroups,
    _cell_count_hint: u32,
) {
    let current_index = triple_buffers.current_index();
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let spatial_grid_bind_group = &cached_bind_groups.spatial_grid;
    
    // Always dispatch at full capacity (see execute_gpu_physics_step comment)
    let cell_workgroups = (triple_buffers.capacity + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
    
    // Stage 1: Clear spatial grid counts using DMA zero-fill
    encoder.clear_buffer(&triple_buffers.spatial_grid_counts, 0, None);
    
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Spatial Grid Rebuild After Lifecycle"),
            timestamp_writes: None,
        });
        
        // Stage 2: Assign cells to grid (now only live cells)
        compute_pass.set_pipeline(&pipelines.spatial_grid_assign);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        
        // Stage 3: Insert cells into grid
        compute_pass.set_pipeline(&pipelines.spatial_grid_insert);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
}

/// Execute the signal system (clear → sense → propagate)
/// 
/// This runs ONCE PER FRAME, not per physics step, to avoid 4x over-dispatch.
/// Should be called after lifecycle pipeline so adhesion state is up-to-date.
/// 
/// # Arguments
/// * `has_oculocytes` - If false, skip entire signal system (no genomes have oculocyte modes)
/// * `cell_count_hint` - Live cell count for dispatch scaling
/// * `max_signal_hops` - Maximum signal hops across all oculocyte modes (determines propagation iterations)
pub fn execute_signal_system(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &GpuTripleBufferSystem,
    cached_bind_groups: &CachedBindGroups,
    has_oculocytes: bool,
    cell_count_hint: u32,
    max_signal_hops: u32,
) {
    // Early-out if no oculocytes in any genome - skip entire signal system
    if !has_oculocytes {
        return;
    }
    
    let current_index = triple_buffers.current_index();
    
    // PERFORMANCE: Dispatch based on actual cell count, not full capacity
    let effective_count = (cell_count_hint.max(1) + 63) / 64 * 64; // Round up to workgroup boundary
    let signal_workgroups = (effective_count + 63) / 64;

    // Step 1: Clear signal flags
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Clear"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipelines.signal_clear);
        compute_pass.set_bind_group(0, &cached_bind_groups.signal_flags, &[]);
        compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
    }

    // Step 2: Oculocyte sensing (detect targets, write initial signal values)
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Sense"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipelines.signal_sense);
        compute_pass.set_bind_group(0, &cached_bind_groups.signal_flags, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.signal_sense_cell_data[current_index], &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.signal_sense_world_data, &[]);
        compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
    }

    // Step 3: Pull-based propagation (one hop per dispatch)
    // Use actual max hops from genome data instead of fixed 10.
    // Hop limits are encoded in the signal values themselves, so extra iterations
    // are no-ops, but skipping them saves dispatch overhead.
    let propagation_iterations = max_signal_hops.clamp(1, 20);
    for _ in 0..propagation_iterations {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Propagate"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipelines.signal_propagate);
        compute_pass.set_bind_group(0, &cached_bind_groups.signal_flags, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.signal_propagate_adhesion, &[]);
        compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
    }
}
