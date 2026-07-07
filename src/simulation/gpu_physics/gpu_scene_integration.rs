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
//! - 128^3 spatial grid for collision acceleration

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
    grid_resolution: i32, // 64 for 64^3 grid
    grid_cell_size: f32,  // world_size / 64
    max_cells_per_grid: i32,
    enable_thrust_force: i32,

    // Capacity and gravity mode (16 bytes)
    cell_capacity: u32,              // Maximum cells that can exist
    gravity_mode: u32,               // 0=X, 1=Y, 2=Z, 3=radial (toward origin)
    angular_damping: f32, // fraction of angular velocity retained per second (velocity_update.wgsl _pad1)
    solo_metabolism_multiplier: f32, // Metabolism multiplier for solo cells (1.0 = off, >1.0 = increased drain)

    // Padding to 256 bytes (192 bytes = 48 floats)
    _padding: [f32; 47],
}

/// Grid resolution: 128^3 = 2,097,152 grid cells
const GRID_RESOLUTION: u32 = 128;

/// Workgroup size for cell operations (unified for optimal GPU occupancy)
/// 256 threads = 8 warps = optimal for GPU scheduling
const WORKGROUP_SIZE_CELLS: u32 = 256;

/// Workgroup size for lifecycle operations (moderate complexity)
const WORKGROUP_SIZE_LIFECYCLE: u32 = 128;

/// Workgroup size for adhesion cleanup (256 for optimal GPU occupancy)
const WORKGROUP_SIZE_ADHESION: u32 = 256;

/// Coarse scene capabilities used to skip whole compute passes when the loaded
/// genomes cannot exercise them.
#[derive(Copy, Clone, Debug, Default)]
pub struct PhysicsFeatureFlags {
    pub has_myocytes: bool,
    pub has_flagellocytes: bool,
    pub has_buoyocytes: bool,
    pub has_ciliocytes: bool,
    pub has_siphonocytes: bool,
    pub has_plumocytes: bool,
    pub has_glueocytes: bool,
    pub has_structural_adhesion: bool,
    pub has_mode_switches: bool,
    pub has_auto_nutrient_gain: bool,
    pub has_division: bool,
}

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
    solo_metabolism_multiplier: f32,
    boulder_count: u32,
    boulder_force_accum: Option<&wgpu::Buffer>,
    features: PhysicsFeatureFlags,
) {
    let current_index = triple_buffers.rotate_buffers();
    let has_adhesion_bonds = features.has_structural_adhesion || features.has_glueocytes;

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
        solo_metabolism_multiplier,
        _padding: [0.0; 47],
    };
    queue.write_buffer(
        &triple_buffers.physics_params,
        0,
        bytemuck::bytes_of(&params),
    );

    // DMA zero-fill all buffers that need clearing before the compute pass.
    // This replaces the clear_forces compute dispatch and is significantly faster
    // because the GPU's DMA engine handles bulk zeroing without compute shader overhead.
    // PERFORMANCE: Skip clearing when no cells - saves ~15MB DMA per physics step
    if _cell_count_hint > 0 {
        encoder.clear_buffer(&triple_buffers.mass_deltas_buffer, 0, None);
        encoder.clear_buffer(&triple_buffers.spatial_grid_counts, 0, None);
        encoder.clear_buffer(&triple_buffers.occupied_grid_count, 0, None);
        encoder.clear_buffer(&triple_buffers.spatial_grid_overflow_count, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_z, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_z, 0, None);
        // Clear boulder force accumulator so cell-push forces don't accumulate across frames
        if let Some(bfa) = boulder_force_accum {
            encoder.clear_buffer(bfa, 0, None);
        }
    }

    // Use cached bind groups (no per-frame allocation!)
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let spatial_grid_bind_group = &cached_bind_groups.spatial_grid;
    let adhesion_rotations_bind_group = &cached_bind_groups.rotations[current_index];
    let position_update_rotations_bind_group =
        &cached_bind_groups.position_update_rotations[current_index];
    let mass_accum_bind_group = &cached_bind_groups.mass_accum;

    // IDLE EARLY-OUT: Skip all compute dispatches when there are no cells.
    // With 0 cells every shader would early-exit on the first thread, but the dispatch
    // overhead, pipeline barrier cost, and DMA clears above still consume GPU time.
    // Returning here eliminates ~18 compute dispatches + 4 buffer copies per physics step
    // when the simulation is empty, which is the dominant cost at idle.
    if _cell_count_hint == 0 {
        return;
    }

    // PERFORMANCE: Dispatch based on actual cell count, not full capacity.
    // At 100K cells, this reduces dispatch from 1024 to ~390 workgroups (2.6x reduction).
    // CRITICAL: Must use the HIGH WATER MARK (total slots used), not the live count.
    // Dead cells can be at any index - they're not compacted to the end. Using the
    // live count would skip cells at higher indices, preventing their metabolism from
    // running, so they'd never lose nutrients and never die.
    let effective_cell_count = (_cell_count_hint + 255) / 256 * 256; // Round up to workgroup boundary
    let cell_workgroups = (effective_cell_count + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;

    // Muscle contraction pass: compute per-cell contraction values BEFORE the main physics pass.
    // Running in a separate compute pass ensures a pipeline barrier so all writes to
    // muscle_contraction_buffer are visible to adhesion_physics and adhesion_substep.
    if features.has_myocytes {
        let mut contraction_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Muscle Contraction Pass"),
            timestamp_writes: None,
        });
        contraction_pass.set_pipeline(&pipelines.muscle_contraction);
        contraction_pass.set_bind_group(0, &cached_bind_groups.muscle_contraction_group0, &[]);
        contraction_pass.set_bind_group(1, &cached_bind_groups.muscle_contraction_group1, &[]);
        contraction_pass.set_bind_group(2, &cached_bind_groups.muscle_contraction_group2, &[]);
        contraction_pass.dispatch_workgroups(cell_workgroups, 1, 1);
    }
    // Implicit pipeline barrier: contraction writes are now visible to the physics pass.

    if current_frame.rem_euclid(4) == 0 {
        {
            let mut physiology_transport_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Physiology Transport Pass"),
                    timestamp_writes: None,
                });
            physiology_transport_pass.set_pipeline(&pipelines.physiology_transport);
            physiology_transport_pass.set_bind_group(0, physics_bind_group, &[]);
            physiology_transport_pass.set_bind_group(1, &cached_bind_groups.physiology, &[]);
            physiology_transport_pass.set_bind_group(
                2,
                &cached_bind_groups.physiology_transport,
                &[],
            );
            physiology_transport_pass.set_bind_group(3, &cached_bind_groups.adhesion, &[]);
            physiology_transport_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        {
            let mut physiology_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Physiology Update Pass"),
                timestamp_writes: None,
            });
            physiology_pass.set_pipeline(&pipelines.physiology_update);
            physiology_pass.set_bind_group(0, physics_bind_group, &[]);
            physiology_pass.set_bind_group(1, &cached_bind_groups.physiology, &[]);
            physiology_pass.set_bind_group(2, &cached_bind_groups.physiology_cell_data, &[]);
            physiology_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Only slots below the GPU high-water mark can be read by later shaders.
        // Copying the full allocation here turns sparse or partially-filled worlds
        // into avoidable DMA work on every physiology tick.
        let scalar_bytes = _cell_count_hint.min(triple_buffers.capacity) as u64 * 4;
        encoder.copy_buffer_to_buffer(
            &triple_buffers.cell_water_next,
            0,
            &triple_buffers.cell_water,
            0,
            scalar_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &triple_buffers.cell_heat_energy_next,
            0,
            &triple_buffers.cell_heat_energy,
            0,
            scalar_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &triple_buffers.cell_cached_temperature_next,
            0,
            &triple_buffers.cell_cached_temperature,
            0,
            scalar_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &triple_buffers.cell_thermal_state_next,
            0,
            &triple_buffers.cell_thermal_state,
            0,
            scalar_bytes,
        );
    }

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

        // Stage 4: Collision detection — per-pair parallel dispatch.
        // Pass B (intra-bucket pairs) needs one thread per (bucket, pair).
        // Worst-case: capacity/16 buckets * 120 pairs each.
        // Dispatch the larger of that bound and cell_workgroups so Pass A (per-cell
        // boundary forces) and Pass C (cross-bucket neighbors) are also fully covered.
        {
            let max_buckets = (triple_buffers.capacity + 15) / 16;
            let pair_threads = max_buckets * 120; // 120 = MAX_CELLS_PER_GRID*(MAX_CELLS_PER_GRID-1)/2
            let collision_workgroups = ((pair_threads.max(effective_cell_count)) + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
            compute_pass.set_pipeline(&pipelines.collision_detection);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
            compute_pass.set_bind_group(
                2,
                &cached_bind_groups.collision_force_accum[current_index],
                &[],
            );
            compute_pass.dispatch_workgroups(collision_workgroups, 1, 1);
        }
        if has_adhesion_bonds {
            compute_pass.set_pipeline(&pipelines.adhesion_physics);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
            compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.force_accum, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.5: Swim force (256 threads) - applies thrust for Flagellocyte cells
        // Accumulates swim force to force buffers based on cell rotation and mode swim_force setting
        if features.has_flagellocytes || features.has_siphonocytes || features.has_plumocytes {
            compute_pass.set_pipeline(&pipelines.swim_force);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.swim_force_force_accum[current_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.51: Buoyancy force (256 threads) - applies upward force for Buoyocyte cells
        if features.has_buoyocytes {
            compute_pass.set_pipeline(&pipelines.buoyancy_force);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.swim_force_force_accum[current_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.55: Cilia force (256 threads) - contact-dependent surface propulsion for Ciliocyte cells
        // Pushes against neighbors and walls to generate thrust along forward axis
        if features.has_ciliocytes && cave_renderer.is_some() {
            compute_pass.set_pipeline(&pipelines.cilia_force);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.cilia_force_force_accum[current_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.cilia_force_cell_data, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.cilia_force_spatial, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.6: Glueocyte environment adhesion (cave and/or boulders)
        // Must run BEFORE position_update so forces are accumulated and applied this frame
        {
            let has_cave = cave_renderer.is_some();
            let has_boulders = boulder_count > 0;
            if features.has_glueocytes && (has_cave || has_boulders) {
                let cave_bg = if let Some(ref cave) = cave_renderer {
                    cave.collision_bind_group()
                } else {
                    &cached_bind_groups.dummy_cave_collision
                };
                compute_pass.set_pipeline(&pipelines.glueocyte_env_adhesion);
                compute_pass.set_bind_group(0, physics_bind_group, &[]);
                compute_pass.set_bind_group(
                    1,
                    &cached_bind_groups.env_adhesion_force_accum[current_index],
                    &[],
                );
                compute_pass.set_bind_group(2, &cached_bind_groups.env_adhesion_mode_data, &[]);
                compute_pass.set_bind_group(3, cave_bg, &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 5.7: Glueocyte cell-to-cell adhesion
        // bond_release runs first so a newly-inactive glueocyte releases its bonds before
        // bond_create could re-form them in the same step.
        // bond_create then forms new bonds for active glueocytes that are in contact.
        // Both passes run unconditionally (no cave dependency).
        if features.has_glueocytes {
            compute_pass.set_pipeline(&pipelines.glueocyte_cell_adhesion_release);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.cell_adhesion_adhesion, &[]);
            compute_pass.set_bind_group(2, &cached_bind_groups.cell_adhesion_spatial, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.cell_adhesion_mode, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

            compute_pass.set_pipeline(&pipelines.glueocyte_cell_adhesion_create);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.cell_adhesion_adhesion, &[]);
            compute_pass.set_bind_group(2, &cached_bind_groups.cell_adhesion_spatial, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.cell_adhesion_mode, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }
        // Now reads accumulated forces and applies them with proper integration
        // Also handles buoyancy (cells float in water when fluid system is enabled)
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, position_update_rotations_bind_group, &[]);
        compute_pass.set_bind_group(
            2,
            &cached_bind_groups.position_update_force_accum[current_index],
            &[],
        );
        compute_pass.set_bind_group(3, &cached_bind_groups.position_update_spatial_grid, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 6.5: Boulder physics is now in a separate pass after nutrient apply.
        // (Moved to avoid buffer aliasing with cell_count_buffer.)

        // Stage 7: Angular velocity integration (256 threads)
        // Applies accumulated torques to angular velocities and rotations.
        // Must use the bind group that targets rotations[current_index] so that
        // angular integration runs on the same buffer physics is processing this frame.
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(
            1,
            &cached_bind_groups.velocity_update_angular[current_index],
            &[],
        );
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 7.5: Adhesion constraint sub-stepping (N additional iterations)
        // Each iteration re-evaluates adhesion forces against latest positions and
        // applies corrections directly to output buffers. Dramatically increases
        // effective joint stiffness without changing spring constants.
        if has_adhesion_bonds {
            for _ in 0..constraint_iterations {
                compute_pass.set_pipeline(&pipelines.adhesion_substep);
                compute_pass.set_bind_group(0, physics_bind_group, &[]);
                compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
                compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 7.6: Cave collision (if enabled) - one final environmental SDF
        // enforcement after all position/substep corrections. This replaces the
        // previous per-substep cave pass and avoids repeating the same SDF work
        // `1 + constraint_iterations` times per physics step.
        if let (Some(cave_renderer), Some(cave_bind_groups)) =
            (cave_renderer, cave_physics_bind_groups)
        {
            if cave_renderer.params().collision_enabled != 0 {
                compute_pass.set_pipeline(cave_renderer.collision_pipeline());
                compute_pass.set_bind_group(0, &cave_bind_groups[current_index], &[]);
                compute_pass.set_bind_group(1, cave_renderer.collision_bind_group(), &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 7.7: Final Plumocyte angular resistance after substep/cave corrections.
        if features.has_plumocytes {
            let output_index = (current_index + 1) % 3;
            compute_pass.set_pipeline(&pipelines.plumocyte_rotation_damping);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.swim_force_force_accum[output_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 8: Mass accumulation (256 threads) - nutrient growth only.
        // Only default/Test cells auto-gain nutrients here; specialist
        // monocultures otherwise paid for a full all-cell pass that immediately
        // returned in the shader.
        if features.has_auto_nutrient_gain {
            compute_pass.set_pipeline(&pipelines.mass_accum);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, mass_accum_bind_group, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

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

    // Stage 9c: Boulder consume (separate compute pass).
    // Must be after the main physics pass so cell_count_buffer is no longer bound
    // read-write by the physics group. The pass boundary is a full pipeline barrier.
    // Must also be after nutrient apply so boulder nutrients are visible to transport
    // on the next frame (transport already ran this frame).
    // Boulder physics runs in the same pass immediately after consume.
    if boulder_count > 0 {
        let boulder_workgroups = (boulder_count + 63) / 64;
        // Use the input position buffer (previous frame's output) - it's idle this frame.
        let input_index = (current_index + 2) % 3;

        // Write the minimal boulder consume params (no storage buffers, no conflicts).
        // Layout: [delta_time: f32, world_size: f32, grid_cell_size: f32, grid_resolution: i32]
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct BoulderConsumeParams {
            delta_time: f32,
            world_size: f32,
            grid_cell_size: f32,
            grid_resolution: i32,
        }
        let boulder_params = BoulderConsumeParams {
            delta_time: params.delta_time,
            world_size: params.world_size,
            grid_cell_size: params.grid_cell_size,
            grid_resolution: params.grid_resolution,
        };
        queue.write_buffer(
            &pipelines.boulder_consume_params_buffer,
            0,
            bytemuck::bytes_of(&boulder_params),
        );

        {
            let mut boulder_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Boulder Consume + Physics"),
                timestamp_writes: None,
            });

            // Boulder consume: size-gated moss -> cell nutrients
            boulder_pass.set_pipeline(&pipelines.boulder_consume);
            boulder_pass.set_bind_group(0, &cached_bind_groups.boulder_consume_params, &[]);
            boulder_pass.set_bind_group(1, &cached_bind_groups.boulder_consume_spatial, &[]);
            boulder_pass.set_bind_group(
                2,
                &cached_bind_groups.boulder_consume_cell_data[input_index],
                &[],
            );
            boulder_pass.set_bind_group(3, &cached_bind_groups.boulder_consume_buffers, &[]);
            boulder_pass.dispatch_workgroups(boulder_workgroups, 1, 1);

            // Boulder physics: gravity, cave SDF, integration, death, moss_dir update.
            // Runs in the same pass so eat_dir_accum written by consume is visible here
            // (storage writes are ordered within a single compute pass).
            // Boulder physics always runs - cave SDF collision is skipped if no cave.
            boulder_pass.set_pipeline(&pipelines.boulder_physics);
            boulder_pass.set_bind_group(0, physics_bind_group, &[]);
            boulder_pass.set_bind_group(1, &cached_bind_groups.boulder_physics_buffers, &[]);
            if let Some(cave_renderer) = cave_renderer {
                boulder_pass.set_bind_group(2, cave_renderer.collision_bind_group(), &[]);
            } else {
                // Use a dummy cave bind group - the shader checks collision_enabled == 0
                // which is set in the dummy buffer, so SDF code is skipped.
                boulder_pass.set_bind_group(2, &cached_bind_groups.boulder_dummy_cave, &[]);
            }
            boulder_pass.dispatch_workgroups(boulder_workgroups, 1, 1);
        }
    }
}

/// Execute a mechanics-only physics step (no nutrient transport, no mass accumulation).
///
/// Runs stages 1-7 of the physics pipeline (spatial grid, clear forces, collision,
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
    features: PhysicsFeatureFlags,
) {
    // Rotate to next buffer set
    let current_index = triple_buffers.rotate_buffers();
    let has_adhesion_bonds = features.has_structural_adhesion || features.has_glueocytes;

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
        solo_metabolism_multiplier: 1.0, // Not used in mechanics step
        _padding: [0.0; 47],
    };
    queue.write_buffer(
        &triple_buffers.physics_params,
        0,
        bytemuck::bytes_of(&params),
    );

    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let spatial_grid_bind_group = &cached_bind_groups.spatial_grid;
    let adhesion_rotations_bind_group = &cached_bind_groups.rotations[current_index];
    let position_update_rotations_bind_group =
        &cached_bind_groups.position_update_rotations[current_index];

    // PERFORMANCE: Dispatch based on actual cell count, not full capacity
    let effective_cell_count = (_cell_count_hint.max(1) + 255) / 256 * 256;
    let cell_workgroups = (effective_cell_count + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;

    // DMA zero-fill spatial grid + force/torque buffers before compute pass
    // PERFORMANCE: Skip clearing when no cells - saves ~15MB DMA
    if _cell_count_hint > 0 {
        encoder.clear_buffer(&triple_buffers.spatial_grid_counts, 0, None);
        encoder.clear_buffer(&triple_buffers.occupied_grid_count, 0, None);
        encoder.clear_buffer(&triple_buffers.spatial_grid_overflow_count, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.force_accum_z, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_x, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_y, 0, None);
        encoder.clear_buffer(&adhesion_buffers.torque_accum_z, 0, None);
    }

    // Muscle contraction pass: separate compute pass ensures pipeline barrier
    // so contraction values are visible to adhesion_physics and adhesion_substep.
    if features.has_myocytes {
        let mut contraction_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Muscle Contraction Pass (Drag)"),
            timestamp_writes: None,
        });
        contraction_pass.set_pipeline(&pipelines.muscle_contraction);
        contraction_pass.set_bind_group(0, &cached_bind_groups.muscle_contraction_group0, &[]);
        contraction_pass.set_bind_group(1, &cached_bind_groups.muscle_contraction_group1, &[]);
        contraction_pass.set_bind_group(2, &cached_bind_groups.muscle_contraction_group2, &[]);
        contraction_pass.dispatch_workgroups(cell_workgroups, 1, 1);
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

        // Stage 4: Collision detection — per-pair parallel dispatch.
        {
            let max_buckets = (triple_buffers.capacity + 15) / 16;
            let pair_threads = max_buckets * 120;
            let collision_workgroups = ((pair_threads.max(effective_cell_count)) + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;
            compute_pass.set_pipeline(&pipelines.collision_detection);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, spatial_grid_bind_group, &[]);
            compute_pass.set_bind_group(
                2,
                &cached_bind_groups.collision_force_accum[current_index],
                &[],
            );
            compute_pass.dispatch_workgroups(collision_workgroups, 1, 1);
        }

        // Stage 5: Adhesion physics
        if has_adhesion_bonds {
            compute_pass.set_pipeline(&pipelines.adhesion_physics);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
            compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.force_accum, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.5: Swim force
        if features.has_flagellocytes || features.has_siphonocytes || features.has_plumocytes {
            compute_pass.set_pipeline(&pipelines.swim_force);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.swim_force_force_accum[current_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.51: Buoyancy force
        if features.has_buoyocytes {
            compute_pass.set_pipeline(&pipelines.buoyancy_force);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.swim_force_force_accum[current_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.55: Cilia force (256 threads) - contact-dependent surface propulsion for Ciliocyte cells
        // Pushes against neighbors and walls to generate thrust along forward axis
        if features.has_ciliocytes && cave_renderer.is_some() {
            compute_pass.set_pipeline(&pipelines.cilia_force);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.cilia_force_force_accum[current_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.cilia_force_cell_data, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.cilia_force_spatial, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 5.6: Glueocyte environment adhesion (cave and/or boulders)
        // Must run BEFORE position_update so forces are accumulated and applied this frame
        {
            let has_cave = cave_renderer.is_some();
            let has_boulders = false; // boulder_count not available in mechanics step; cave-only here
            if features.has_glueocytes && (has_cave || has_boulders) {
                let cave_bg = if let Some(ref cave) = cave_renderer {
                    cave.collision_bind_group()
                } else {
                    &cached_bind_groups.dummy_cave_collision
                };
                compute_pass.set_pipeline(&pipelines.glueocyte_env_adhesion);
                compute_pass.set_bind_group(0, physics_bind_group, &[]);
                compute_pass.set_bind_group(
                    1,
                    &cached_bind_groups.env_adhesion_force_accum[current_index],
                    &[],
                );
                compute_pass.set_bind_group(2, &cached_bind_groups.env_adhesion_mode_data, &[]);
                compute_pass.set_bind_group(3, cave_bg, &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 5.7: Glueocyte cell-to-cell adhesion (mechanics step)
        if features.has_glueocytes {
            compute_pass.set_pipeline(&pipelines.glueocyte_cell_adhesion_release);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.cell_adhesion_adhesion, &[]);
            compute_pass.set_bind_group(2, &cached_bind_groups.cell_adhesion_spatial, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.cell_adhesion_mode, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

            compute_pass.set_pipeline(&pipelines.glueocyte_cell_adhesion_create);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.cell_adhesion_adhesion, &[]);
            compute_pass.set_bind_group(2, &cached_bind_groups.cell_adhesion_spatial, &[]);
            compute_pass.set_bind_group(3, &cached_bind_groups.cell_adhesion_mode, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        // Stage 6: Position integration
        compute_pass.set_pipeline(&pipelines.position_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(1, position_update_rotations_bind_group, &[]);
        compute_pass.set_bind_group(
            2,
            &cached_bind_groups.position_update_force_accum[current_index],
            &[],
        );
        compute_pass.set_bind_group(3, &cached_bind_groups.position_update_spatial_grid, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 7: Angular velocity integration
        compute_pass.set_pipeline(&pipelines.velocity_update);
        compute_pass.set_bind_group(0, physics_bind_group, &[]);
        compute_pass.set_bind_group(
            1,
            &cached_bind_groups.velocity_update_angular[current_index],
            &[],
        );
        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

        // Stage 7.5: Adhesion constraint sub-stepping (N additional iterations)
        if has_adhesion_bonds {
            for _ in 0..constraint_iterations {
                compute_pass.set_pipeline(&pipelines.adhesion_substep);
                compute_pass.set_bind_group(0, physics_bind_group, &[]);
                compute_pass.set_bind_group(1, &cached_bind_groups.adhesion, &[]);
                compute_pass.set_bind_group(2, adhesion_rotations_bind_group, &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 7.6: Cave collision (if enabled) - one final environmental SDF
        // enforcement after all drag/mechanics corrections.
        if let (Some(cave_renderer), Some(cave_bind_groups)) =
            (cave_renderer, cave_physics_bind_groups)
        {
            if cave_renderer.params().collision_enabled != 0 {
                compute_pass.set_pipeline(cave_renderer.collision_pipeline());
                compute_pass.set_bind_group(0, &cave_bind_groups[current_index], &[]);
                compute_pass.set_bind_group(1, cave_renderer.collision_bind_group(), &[]);
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
        }

        // Stage 7.7: Final Plumocyte angular resistance after substep/cave corrections.
        if features.has_plumocytes {
            let output_index = (current_index + 1) % 3;
            compute_pass.set_pipeline(&pipelines.plumocyte_rotation_damping);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(
                1,
                &cached_bind_groups.swim_force_force_accum[output_index],
                &[],
            );
            compute_pass.set_bind_group(2, &cached_bind_groups.swim_force_cell_data, &[]);
            compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
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
    total_cell_slots: u32,
    features: PhysicsFeatureFlags,
) -> bool {
    // Get current buffer index (already rotated by physics pipeline)
    let current_index = triple_buffers.current_index();

    // Use cached bind groups (no per-frame allocation!)
    let physics_bind_group = &cached_bind_groups.physics[current_index];
    let lifecycle_bind_group = &cached_bind_groups.lifecycle;
    let cell_state_read_bind_group = &cached_bind_groups.cell_state_read;
    let cell_state_write_bind_group = &cached_bind_groups.cell_state_write[current_index];

    // Dispatch lifecycle based on the high-water mark (total_cell_slots), not full capacity.
    // total_cell_slots is the highest slot index ever used - dead cells can be at any index
    // up to this mark, so we must cover all of them. Using full capacity when only a fraction
    // of slots are occupied wastes GPU time iterating empty slots. The lifecycle shaders
    // early-exit on empty slots, but the dispatch overhead and memory bandwidth for reading
    // death_flags across unused slots is still real.
    // When total_cell_slots == 0 there is nothing to scan - skip all lifecycle work.
    if total_cell_slots == 0 {
        return false;
    }
    let effective_slots = total_cell_slots.min(triple_buffers.capacity);
    let cell_workgroups_lifecycle =
        (effective_slots + WORKGROUP_SIZE_LIFECYCLE - 1) / WORKGROUP_SIZE_LIFECYCLE;

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
        let adhesion_workgroups = (adhesion_buffers.max_connections + WORKGROUP_SIZE_ADHESION - 1)
            / WORKGROUP_SIZE_ADHESION;
        compute_pass.dispatch_workgroups(adhesion_workgroups, 1, 1);

        drop(compute_pass);
    }

    // Stage 1.75: Signal-driven mode switches.
    // Signal propagation runs once before the physics/lifecycle loop. Applying the
    // switch here matches preview ordering: apoptosis first, then mode switch, then division.
    if features.has_mode_switches {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lifecycle Pipeline - Mode Switch"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipelines.mode_switch);
        compute_pass.set_bind_group(0, &cached_bind_groups.mode_switch_group0, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.mode_switch_group1, &[]);
        compute_pass.set_bind_group(2, &cached_bind_groups.mode_switch_group2, &[]);
        compute_pass.dispatch_workgroups(cell_workgroups_lifecycle, 1, 1);
    }

    // Stage 2: Division scan - allocates slots from ring buffer for dividing cells
    // Runs AFTER death scan so recycled slots are available
    if features.has_division {
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
    if features.has_division {
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

    if !features.has_division {
        return false;
    }

    // Signal system moved to execute_signal_system() - runs once per frame, not per physics step
    // This prevents 4x over-dispatch when running 4 physics steps per frame

    // After division execute, propagate the output buffer to the third (stale) triple buffer.
    // Division writes new child cells only to positions_out (output_idx). The third buffer
    // (stale_idx) still holds old dead-cell data at recycled slot indices. Two physics steps
    // later that stale buffer rotates back into positions_in, causing ghost flickering.
    // Copying output -> stale keeps all 3 triple buffers consistent after every division.
    // MUST use full capacity - divided cells land at recycled slot indices scattered
    // throughout the buffer, not at the front. Truncating by total_cell_slots would leave
    // stale data at higher indices, causing explosive teleportation when those slots rotate
    // back into positions_in two steps later.
    let output_idx = (current_index + 1) % 3;
    let stale_idx = (current_index + 2) % 3;
    let buf_size = triple_buffers.capacity as u64 * 16; // Vec4<f32> = 16 bytes per slot
    encoder.copy_buffer_to_buffer(
        &triple_buffers.position_and_mass[output_idx],
        0,
        &triple_buffers.position_and_mass[stale_idx],
        0,
        buf_size,
    );
    encoder.copy_buffer_to_buffer(
        &triple_buffers.velocity[output_idx],
        0,
        &triple_buffers.velocity[stale_idx],
        0,
        buf_size,
    );
    encoder.copy_buffer_to_buffer(
        &triple_buffers.rotations[output_idx],
        0,
        &triple_buffers.rotations[stale_idx],
        0,
        buf_size,
    );

    // Sync next_adhesion_id to adhesion_counts[0] so adhesion line renderer sees GPU-created adhesions
    // The division shader atomically increments next_adhesion_id when creating adhesions,
    // but the adhesion line shader reads from adhesion_counts[0] for the total count.
    encoder.copy_buffer_to_buffer(
        &adhesion_buffers.next_adhesion_id,
        0,
        &adhesion_buffers.adhesion_counts,
        0, // offset 0 = total_adhesion_count
        4, // 4 bytes (u32)
    );

    true
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

    // Dispatch over live cell slots only - shaders check cell_count_buffer[0] internally.
    let effective_slots = _cell_count_hint.min(triple_buffers.capacity).max(1);
    let cell_workgroups = (effective_slots + WORKGROUP_SIZE_CELLS - 1) / WORKGROUP_SIZE_CELLS;

    // Stage 1: Clear spatial grid counts using DMA zero-fill
    encoder.clear_buffer(&triple_buffers.spatial_grid_counts, 0, None);
    encoder.clear_buffer(&triple_buffers.spatial_grid_overflow_count, 0, None);

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

/// Execute the signal system (clear -> sense -> propagate)
///
/// This runs ONCE PER FRAME, not per physics step, to avoid 4x over-dispatch.
/// Should be called before lifecycle so gated actions consume the current frame's signals.
///
/// # Arguments
/// * `has_oculocytes` - If false, skip the signal system (no signal sources or listeners)
/// * `cell_count_hint` - Live cell count for dispatch scaling
/// * `max_signal_hops` - Maximum signal hops across all oculocyte modes (determines propagation iterations)
pub fn execute_signal_system(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &GpuPhysicsPipelines,
    triple_buffers: &GpuTripleBufferSystem,
    adhesion_buffers: &super::AdhesionBuffers,
    cached_bind_groups: &CachedBindGroups,
    has_oculocytes: bool,
    cell_count_hint: u32,
    max_signal_hops: u32,
) {
    // Early-out if no signal sources or listeners exist in any genome.
    if !has_oculocytes {
        return;
    }

    let current_index = triple_buffers.current_index();

    // Dispatch size: round up to workgroup boundary (workgroup_size = 256, matching all other cell passes)
    let signal_workgroups = (cell_count_hint.max(1) + 255) / 256;

    // Copy/clear only the active slot range. At 100K cells this is 6.4 MB per
    // signal buffer instead of touching the full capacity allocation.
    let signal_buffer_size =
        (u64::from(cell_count_hint.max(1)) * 16 * 4).min(adhesion_buffers.signal_flags.size());

    // Step 1: Clear signal flags with DMA. This replaces a shader pass that
    // performed 16 atomic stores per cell every signal frame.
    encoder.clear_buffer(&adhesion_buffers.signal_flags, 0, Some(signal_buffer_size));

    // Step 2: Oculocyte sensing (detect targets, write initial signal values)
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Sense"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipelines.signal_sense);
        compute_pass.set_bind_group(0, &cached_bind_groups.signal_flags, &[]);
        compute_pass.set_bind_group(
            1,
            &cached_bind_groups.signal_sense_cell_data[current_index],
            &[],
        );
        compute_pass.set_bind_group(2, &cached_bind_groups.signal_sense_world_data, &[]);
        compute_pass.set_bind_group(3, &cached_bind_groups.cilia_force_spatial, &[]);
        compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
    }

    // Preserve direct emissions so the reverse sweep can start from the same
    // seed without re-running the expensive sense shader.
    encoder.copy_buffer_to_buffer(
        &adhesion_buffers.signal_flags,
        0,
        &adhesion_buffers.signal_flags_seed,
        0,
        signal_buffer_size,
    );

    // Step 3: Forward double-buffered pull propagation - one hop per dispatch.
    //
    // The propagate shader reads from signal_flags (binding 0, read-only) and writes
    // to signal_flags_next (binding 2, read_write).  After each dispatch we copy
    // signal_flags_next -> signal_flags so the next hop sees the freshly propagated
    // values.  This eliminates the read-write hazard that existed when both reads and
    // writes targeted the same buffer (where thread scheduling order determined whether
    // a relay cell's output was visible to its own neighbors in the same dispatch).
    //
    // The packed budget is measured in quarter-hop cost units, but each dispatch
    // crosses one complete adhesion edge and subtracts that edge's full cost. The
    // number of dispatches is therefore the authored hop count, not budget units.
    //
    let propagation_iterations = max_signal_hops.clamp(1, 20);
    for _ in 0..propagation_iterations {
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Propagate"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipelines.signal_propagate);
            compute_pass.set_bind_group(0, &cached_bind_groups.signal_propagate_flags, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.signal_propagate_adhesion, &[]);
            compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
        }
        // Copy signal_flags_next -> signal_flags so the next hop reads the updated values.
        encoder.copy_buffer_to_buffer(
            &adhesion_buffers.signal_flags_next,
            0,
            &adhesion_buffers.signal_flags,
            0,
            signal_buffer_size,
        );
    }

    // Preserve the forward sweep, then rebuild direct emissions and sweep in the
    // opposite stable-index direction. Combining both gives same-mode emitter
    // chains contributions from both sides without allowing feedback loops.
    encoder.copy_buffer_to_buffer(
        &adhesion_buffers.signal_flags,
        0,
        &adhesion_buffers.signal_flags_forward,
        0,
        signal_buffer_size,
    );

    // Re-seed direct emissions for the reverse sweep from the captured seed.
    encoder.copy_buffer_to_buffer(
        &adhesion_buffers.signal_flags_seed,
        0,
        &adhesion_buffers.signal_flags,
        0,
        signal_buffer_size,
    );

    for _ in 0..propagation_iterations {
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Propagate Reverse"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipelines.signal_propagate_reverse);
            compute_pass.set_bind_group(0, &cached_bind_groups.signal_propagate_flags, &[]);
            compute_pass.set_bind_group(1, &cached_bind_groups.signal_propagate_adhesion, &[]);
            compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &adhesion_buffers.signal_flags_next,
            0,
            &adhesion_buffers.signal_flags,
            0,
            signal_buffer_size,
        );
    }

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Combine Sweeps"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipelines.signal_combine_sweeps);
        compute_pass.set_bind_group(0, &cached_bind_groups.signal_propagate_flags, &[]);
        compute_pass.set_bind_group(1, &cached_bind_groups.signal_propagate_adhesion, &[]);
        compute_pass.dispatch_workgroups(signal_workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &adhesion_buffers.signal_flags_next,
        0,
        &adhesion_buffers.signal_flags,
        0,
        signal_buffer_size,
    );
}
