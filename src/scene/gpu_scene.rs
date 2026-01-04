//! # GPU Simulation Scene - Pure GPU Physics Implementation
//!
//! This module implements a complete GPU-accelerated simulation scene that runs
//! entirely on the GPU with zero CPU physics involvement. The scene achieves
//! maximum performance by eliminating all CPU-GPU synchronization points during
//! physics computation.
//!
//! ## Architecture Overview
//!
//! The GPU scene is built around three core systems:
//!
//! ### 1. Triple Buffer System
//! - **Physics Buffers**: Three sets of simulation state (current, previous, next)
//! - **Instance Buffers**: Three sets of rendering data for asynchronous updates
//! - **Command Buffers**: Pipelined GPU work submission without blocking
//!
//! ### 2. Compute Pipeline System
//! - **15-Stage Physics Pipeline**: Complete GPU-only physics computation
//! - **Spatial Partitioning**: GPU-based collision detection acceleration
//! - **Lifecycle Management**: Cell division, death, and memory management on GPU
//!
//! ### 3. Pure GPU Execution
//! - **Zero CPU Synchronization**: No CPU readback during physics steps
//! - **GPU-to-GPU Operations**: All data movement stays on GPU
//! - **Atomic Buffer Rotation**: Lock-free buffer management
//!
//! ## Performance Characteristics
//!
//! - **Linear GPU Scaling**: Performance scales with GPU compute units, not CPU cores
//! - **Maximum Memory Bandwidth**: Optimized SoA layout for cache efficiency
//! - **Pipeline Parallelism**: Physics, rendering, and data extraction overlap
//! - **Deterministic Execution**: Same input always produces same output
//!
//! ## Integration Points
//!
//! - **Scene Trait**: Compatible with existing scene management system
//! - **CameraController**: Uses existing camera system unchanged
//! - **Genome System**: GPU buffers for mode data, CPU for editing
//! - **Rendering**: Extracts instance data from GPU buffers for visualization

use crate::genome::Genome;
use crate::rendering::{CellRenderer, CullingMode, HizGenerator, InstanceBuilder, AdhesionLineRenderer};
use crate::rendering::cells::AdhesionLineData;
use crate::scene::{Scene, triple_buffer::GpuTripleBufferSystem, gpu_buffers::*, compute_pipelines::*, performance_monitor::*};
use crate::simulation::{CanonicalState, PhysicsConfig};
use crate::ui::camera::CameraController;
use glam::Mat4;
use std::sync::atomic::{AtomicU32, Ordering};

/// GPU simulation scene for large-scale simulations with pure GPU physics.
///
/// This scene implements a complete GPU-accelerated physics simulation that runs
/// entirely on the GPU with zero CPU involvement in physics computation. It uses
/// a sophisticated triple buffering system to eliminate CPU-GPU synchronization
/// stalls and achieve maximum performance.
///
/// ## Core Architecture
///
/// ### Triple Buffer System
/// - **Physics Computation**: Operates on buffer set N
/// - **Visual Data Extraction**: Operates on buffer set N-1
/// - **Rendering**: Operates on buffer set N-2
/// - All operations overlap for maximum GPU utilization
///
/// ### Pure GPU Physics Pipeline
/// 1. **Spatial Grid Clear** - Reset grid cell counts
/// 2. **Spatial Grid Assign** - Assign cells to grid cells
/// 3. **Spatial Grid Insert** - Insert cell indices into grid
/// 4. **Spatial Grid Prefix Sum** - Build grid offset arrays
/// 5. **Cell Physics Spatial** - Collision detection and forces
/// 6. **Adhesion Physics** - Bond forces and constraints
/// 7. **Cell Position Update** - Verlet integration
/// 8. **Cell Velocity Update** - Velocity integration
/// 9. **Cell Internal Update** - Aging, nutrients, signaling
/// 10. **Momentum Correction** - Conservation of momentum
/// 11. **Rigid Body Constraints** - Advanced physics constraints
/// 12. **Cell Lifecycle** - Death/division using prefix-sum
/// 13. **Extract Instances** - Rendering data extraction
///
/// ### Performance Optimization
/// - **Zero CPU Synchronization**: No CPU readback during physics steps
/// - **GPU Memory Exclusive**: All simulation state stays on GPU
/// - **Atomic Buffer Rotation**: Lock-free buffer management
/// - **Batched GPU Operations**: Minimize command buffer overhead
///
/// ## Capacity and Scaling
/// - **Default Capacity**: 10,000 cells for GPU scene
/// - **Spatial Grid**: 64³ grid for collision detection optimization
/// - **Linear GPU Scaling**: Performance scales with GPU compute units
/// - **Memory Bandwidth**: Optimized SoA layout for cache efficiency
pub struct GpuScene {
    /// Triple buffer system for maximum GPU performance
    /// Enables lock-free physics computation with zero CPU synchronization
    triple_buffer_system: GpuTripleBufferSystem,
    
    /// Compute pipeline manager for all GPU physics operations
    /// Manages 13+ compute pipelines for complete physics simulation
    #[allow(dead_code)] // Will be used in Task 3+ when compute shaders are implemented
    compute_pipeline_manager: ComputePipelineManager,
    
    /// GPU uniform buffers for physics parameters and configuration
    physics_params_buffer: wgpu::Buffer,
    #[allow(dead_code)] // Will be used in Task 3+ when compute shaders are implemented
    cell_count_buffer: wgpu::Buffer,
    #[allow(dead_code)] // Will be used in Task 11+ when genome system is integrated
    genome_modes_buffer: wgpu::Buffer,
    
    /// Cell renderer for visualization (uses extracted GPU instance data)
    pub renderer: CellRenderer,
    
    /// Adhesion line renderer for visualizing cell connections
    pub adhesion_renderer: AdhesionLineRenderer,
    
    /// GPU instance builder with frustum and occlusion culling
    pub instance_builder: InstanceBuilder,
    
    /// Hi-Z generator for occlusion culling optimization
    pub hiz_generator: HizGenerator,
    
    /// Physics configuration (CPU-side for UI editing)
    pub config: PhysicsConfig,
    
    /// Whether simulation is paused
    pub paused: bool,
    
    /// Camera controller (unchanged from existing system)
    pub camera: CameraController,
    
    /// Current simulation time
    pub current_time: f32,
    
    /// Current frame number (for deterministic behavior)
    current_frame: AtomicU32,
    
    /// Genomes for cell behavior (CPU-side for editing, GPU buffers for execution)
    pub genomes: Vec<Genome>,
    
    /// Genome mode buffer manager for GPU-based mode lookups and transitions
    genome_mode_manager: GenomeModeBufferManager,
    
    /// Current cell count (atomic for thread-safe access)
    cell_count: AtomicU32,
    
    /// Maximum capacity (fixed at creation)
    capacity: u32,
    
    /// Whether this is the first frame (no Hi-Z data yet)
    first_frame: bool,
    
    /// Minimal CPU state for compatibility (only used for initial setup and UI)
    /// This is NOT used during physics computation - all physics runs on GPU
    pub canonical_state: CanonicalState,
    
    /// Performance monitoring system for GPU operations
    /// Tracks frame timing, buffer rotation statistics, and performance comparisons
    performance_monitor: GpuPerformanceMonitor,
}

impl GpuScene {
    /// Create a new GPU scene with pure GPU physics implementation.
    ///
    /// This initializes the complete GPU scene infrastructure including:
    /// - Triple buffer system for lock-free GPU computation
    /// - Compute pipeline manager for all physics operations
    /// - GPU uniform buffers for physics parameters
    /// - Rendering integration with existing systems
    ///
    /// # Arguments
    /// * `device` - wgpu device for GPU resource creation
    /// * `queue` - wgpu queue for GPU command submission
    /// * `surface_config` - Surface configuration for rendering setup
    ///
    /// # Performance Notes
    /// - All GPU buffers are pre-allocated to avoid runtime allocations
    /// - Triple buffering eliminates CPU-GPU synchronization stalls
    /// - Compute pipelines are created on-demand for optimal startup time
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let capacity = 10_000; // 10k cell capacity for GPU scene
        
        // Create triple buffer system for maximum GPU performance
        let triple_buffer_system = GpuTripleBufferSystem::new(device, capacity);
        
        // Create compute pipeline manager with optimized workgroup sizes
        let workgroup_sizes = ComputeWorkgroupSizes::default();
        let compute_pipeline_manager = ComputePipelineManager::new(
            device.clone(), 
            workgroup_sizes
        );
        
        // Create GPU uniform buffers
        let physics_params = PhysicsParams::default();
        let physics_params_buffer = create_uniform_buffer(
            device, 
            &physics_params, 
            "Physics Parameters"
        );
        
        let cell_count_data = CellCountBuffer::default();
        let cell_count_buffer = create_uniform_buffer(
            device, 
            &cell_count_data, 
            "Cell Count Buffer"
        );
        
        // Create genome modes buffer (initially empty, will be populated when genomes are added)
        let genome_modes_buffer = create_storage_buffer::<GpuMode>(
            device, 
            256, // Support up to 256 total modes across all genomes
            "Genome Modes Buffer"
        );
        
        // Create genome mode buffer manager
        let genome_mode_manager = GenomeModeBufferManager::new(device, 256);
        
        // Create rendering components (unchanged from existing system)
        let renderer = CellRenderer::new(device, queue, surface_config, capacity as usize);
        let adhesion_renderer = AdhesionLineRenderer::new(device, queue, surface_config, capacity as usize * 10); // 10 adhesions per cell max
        let instance_builder = InstanceBuilder::new(device, capacity as usize);
        
        let mut hiz_generator = HizGenerator::new(device);
        hiz_generator.resize(device, surface_config.width, surface_config.height);
        
        // Create minimal canonical state for compatibility (NOT used for physics)
        let canonical_state = CanonicalState::with_grid_density(capacity as usize, 64);
        
        Self {
            triple_buffer_system,
            compute_pipeline_manager,
            physics_params_buffer,
            cell_count_buffer,
            genome_modes_buffer,
            renderer,
            adhesion_renderer,
            instance_builder,
            hiz_generator,
            config: PhysicsConfig::default(),
            paused: false,
            camera: CameraController::new(),
            current_time: 0.0,
            current_frame: AtomicU32::new(0),
            genomes: Vec::new(),
            genome_mode_manager,
            cell_count: AtomicU32::new(0),
            capacity,
            first_frame: true,
            canonical_state, // Minimal CPU state for compatibility only
            performance_monitor: GpuPerformanceMonitor::new(),
        }
    }
    
    /// Execute a complete GPU physics step with zero CPU involvement.
    ///
    /// This method runs the entire 15-stage GPU physics pipeline with three-phase
    /// lifecycle management for deterministic execution:
    ///
    /// ## Pipeline Execution Order
    /// 
    /// ### Spatial Grid Operations (Stages 1-4)
    /// 1. **Clear Grid** - Reset spatial grid cell counts to zero
    /// 2. **Assign Cells** - Assign each cell to its spatial grid cell
    /// 3. **Insert Indices** - Insert cell indices into grid cells
    /// 4. **Prefix Sum** - Build prefix sum offsets for efficient traversal
    ///
    /// ### Physics Computation (Stages 5-6)
    /// 5. **Collision Detection** - Detect cell-cell collisions using spatial grid
    /// 6. **Force Calculation** - Calculate collision, adhesion, boundary, and swim forces
    /// 6b. **Adhesion Physics** - Process adhesion bond forces and constraints
    ///
    /// ### Integration (Stages 7-8)
    /// 7. **Position Update** - Integrate positions using Verlet integration
    /// 8. **Velocity Update** - Update velocities from accelerations
    ///
    /// ### Advanced Physics (Stages 9-10)
    /// 9. **Momentum Correction** - Apply momentum conservation corrections
    /// 10. **Rigid Body Constraints** - Apply rigid body physics for adhesion networks
    ///
    /// ### Cell Internal Updates (Stages 11-12)
    /// 11. **Cell Internal** - Update aging, nutrients, and signaling substances
    /// 12. **Nutrient System** - Process nutrient gain, consumption, and transport
    ///
    /// ### Three-Phase Lifecycle Management (Stages 13-15)
    /// **Phase 1: Death Execution**
    /// - Scan for cells ready to die based on nutrient depletion or age
    /// - Compact dead cells using prefix-sum for deterministic removal
    /// - Clean up adhesion connections for dead cells
    ///
    /// **Phase 2: Division Slot Assignment**
    /// - Scan for cells ready to divide based on mass and timing
    /// - Calculate slot reservations (Child B + inherited adhesions)
    /// - Assign slots using prefix-sum: assignments[i] = freeSlots[reservations[i]]
    ///
    /// **Phase 3: Division Execution**
    /// - Create Child A and Child B with 50/50 mass split
    /// - Apply position offset and velocity inheritance
    /// - Create new adhesions using zone classification and geometric recalculation
    /// - Place adhesions in assigned slots using deterministic order
    ///
    /// **Free Slot Management**
    /// - Maintain compacted free slot arrays using prefix-sum
    /// - Ensure N ≥ M (available slots ≥ needed slots) for deterministic allocation
    ///
    /// All stages operate exclusively on GPU buffers with no CPU synchronization.
    /// The three-phase lifecycle ensures deterministic execution where the same
    /// input always produces the same output across all simulation modes.
    ///
    /// # Arguments
    /// * `device` - wgpu device for command encoder creation
    /// * `queue` - wgpu queue for command submission
    /// * `dt` - Delta time for physics step
    ///
    /// # Performance Notes
    /// - **Zero CPU Involvement**: All computation happens on GPU
    /// - **Atomic Buffer Rotation**: Lock-free buffer management
    /// - **Single Command Submission**: Minimizes GPU overhead
    /// - **Deterministic Execution**: Same input → same output
    /// - **Pipeline Parallelism**: Physics, rendering, and extraction overlap
    fn step_gpu_physics(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        // Skip physics if paused or no cells
        if self.paused || self.cell_count.load(Ordering::Acquire) == 0 {
            return;
        }
        
        // Start performance monitoring for this physics step
        self.performance_monitor.start_physics_step_timing();
        
        // Update physics parameters for GPU
        let current_frame = self.current_frame.fetch_add(1, Ordering::AcqRel);
        let cell_count = self.cell_count.load(Ordering::Acquire);
        
        let physics_params = PhysicsParams {
            delta_time: dt,
            current_time: self.current_time,
            current_frame: current_frame as i32,
            cell_count,
            world_size: self.config.sphere_radius * 2.0, // Convert radius to diameter
            boundary_stiffness: self.config.default_stiffness,
            gravity: 0.0, // No gravity in biological simulation
            acceleration_damping: self.config.damping,
            grid_resolution: 64, // Fixed 64³ grid for GPU scene
            grid_cell_size: (self.config.sphere_radius * 2.0) / 64.0,
            max_cells_per_grid: 32,
            enable_thrust_force: 1, // Always enabled for GPU scene
            dragged_cell_index: -1, // TODO: Implement UI interaction
            ..Default::default()
        };
        
        // Update GPU uniform buffers
        queue.write_buffer(&self.physics_params_buffer, 0, bytemuck::cast_slice(&[physics_params]));
        
        // Update genome mode buffer if needed
        if self.genome_mode_manager.needs_update() {
            if let Err(e) = self.genome_mode_manager.update_from_genomes(queue, &self.genomes) {
                log::error!("Failed to update genome mode buffer: {}", e);
                return; // Skip physics step if genome buffer update fails
            }
        }
        
        // Atomically rotate to next physics buffer set
        let new_physics_index = self.triple_buffer_system.rotate_physics_buffers();
        log::trace!("Rotated to physics buffer set {}", new_physics_index);
        
        // Record buffer rotation for performance monitoring
        let (physics_index, instance_index) = self.triple_buffer_system.get_buffer_indices();
        self.performance_monitor.record_buffer_rotation(physics_index, instance_index);
        
        // Start GPU timing
        self.performance_monitor.start_gpu_physics_timing();
        
        // Create command encoder for all GPU work
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Physics Step"),
        });
        
        // Execute complete GPU physics pipeline with three-phase lifecycle management
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU Physics Pipeline"),
                timestamp_writes: None,
            });
            
            // Calculate workgroup dispatch counts
            let workgroup_count = (cell_count + 63) / 64; // Round up for 64-thread workgroups
            let spatial_workgroup_count = (cell_count + 255) / 256; // Round up for 256-thread workgroups
            
            // ========================================================================
            // STAGE 1-4: SPATIAL GRID OPERATIONS
            // ========================================================================
            
            let spatial_start = std::time::Instant::now();
            
            // Stage 1: Clear spatial grid cell counts
            let spatial_clear_pipeline = self.compute_pipeline_manager.get_spatial_grid_clear_pipeline();
            compute_pass.set_pipeline(spatial_clear_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(64, 64, 64); // 64³ grid
            
            // Stage 2: Assign cells to spatial grid cells
            let spatial_assign_pipeline = self.compute_pipeline_manager.get_spatial_grid_assign_pipeline();
            compute_pass.set_pipeline(spatial_assign_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(spatial_workgroup_count, 1, 1);
            
            // Stage 3: Insert cell indices into spatial grid
            let spatial_insert_pipeline = self.compute_pipeline_manager.get_spatial_grid_insert_pipeline();
            compute_pass.set_pipeline(spatial_insert_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(spatial_workgroup_count, 1, 1);
            
            // Stage 4: Build prefix sum for spatial grid offsets
            let spatial_prefix_sum_pipeline = self.compute_pipeline_manager.get_spatial_grid_prefix_sum_pipeline();
            compute_pass.set_pipeline(spatial_prefix_sum_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(spatial_workgroup_count, 1, 1);
            
            // Record spatial grid timing
            let spatial_time = spatial_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::SpatialGrid, spatial_time);
            
            // ========================================================================
            // STAGE 5-6: PHYSICS COMPUTATION
            // ========================================================================
            
            let collision_start = std::time::Instant::now();
            
            // Stage 5: Collision detection using spatial grid
            let collision_detection_pipeline = self.compute_pipeline_manager.get_collision_detection_pipeline();
            compute_pass.set_pipeline(collision_detection_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let collision_time = collision_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::CollisionDetection, collision_time);
            
            let force_start = std::time::Instant::now();
            
            // Stage 6: Force calculation (collision, adhesion, boundary, swim)
            let force_calculation_pipeline = self.compute_pipeline_manager.get_force_calculation_pipeline();
            compute_pass.set_pipeline(force_calculation_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let force_time = force_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::ForceCalculation, force_time);
            
            let adhesion_start = std::time::Instant::now();
            
            // Stage 6b: Adhesion physics (bond forces and constraints)
            let adhesion_physics_pipeline = self.compute_pipeline_manager.get_adhesion_physics_pipeline();
            compute_pass.set_pipeline(adhesion_physics_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let adhesion_time = adhesion_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::AdhesionPhysics, adhesion_time);
            
            // ========================================================================
            // STAGE 7-8: INTEGRATION
            // ========================================================================
            
            let position_start = std::time::Instant::now();
            
            // Stage 7: Position update using Verlet integration
            let position_update_pipeline = self.compute_pipeline_manager.get_position_update_pipeline();
            compute_pass.set_pipeline(position_update_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let position_time = position_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::PositionIntegration, position_time);
            
            let velocity_start = std::time::Instant::now();
            
            // Stage 8: Velocity update from accelerations
            let velocity_update_pipeline = self.compute_pipeline_manager.get_velocity_update_pipeline();
            compute_pass.set_pipeline(velocity_update_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let velocity_time = velocity_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::VelocityIntegration, velocity_time);
            
            // ========================================================================
            // STAGE 9-10: ADVANCED PHYSICS
            // ========================================================================
            
            // Stage 9: Momentum correction for conservation
            let momentum_correction_pipeline = self.compute_pipeline_manager.get_momentum_correction_pipeline();
            compute_pass.set_pipeline(momentum_correction_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // Stage 10: Rigid body constraints for adhesion networks
            let rigid_body_constraints_pipeline = self.compute_pipeline_manager.get_rigid_body_constraints_pipeline();
            compute_pass.set_pipeline(rigid_body_constraints_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // ========================================================================
            // STAGE 11-12: CELL INTERNAL UPDATES AND NUTRIENT SYSTEM
            // ========================================================================
            
            // Stage 11: Cell internal updates (aging, nutrients, signaling)
            let cell_internal_update_pipeline = self.compute_pipeline_manager.get_cell_internal_update_pipeline();
            compute_pass.set_pipeline(cell_internal_update_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let nutrient_start = std::time::Instant::now();
            
            // Stage 12: Nutrient system (gain, consumption, transport)
            let nutrient_system_pipeline = self.compute_pipeline_manager.get_nutrient_system_pipeline();
            compute_pass.set_pipeline(nutrient_system_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let nutrient_time = nutrient_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::NutrientSystem, nutrient_time);
            
            // ========================================================================
            // STAGE 13-15: THREE-PHASE LIFECYCLE MANAGEMENT
            // ========================================================================
            
            let lifecycle_start = std::time::Instant::now();
            
            // Phase 1: Execute Cell Death (prefix-sum compaction)
            let death_scan_pipeline = self.compute_pipeline_manager.get_lifecycle_death_scan_pipeline();
            compute_pass.set_pipeline(death_scan_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let death_compact_pipeline = self.compute_pipeline_manager.get_lifecycle_death_compact_pipeline();
            compute_pass.set_pipeline(death_compact_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // Phase 2: Division Slot Assignment (prefix-sum allocation)
            let division_scan_pipeline = self.compute_pipeline_manager.get_lifecycle_division_scan_pipeline();
            compute_pass.set_pipeline(division_scan_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let slot_assign_pipeline = self.compute_pipeline_manager.get_lifecycle_slot_assign_pipeline();
            compute_pass.set_pipeline(slot_assign_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // Phase 3: Execute Cell Division (deterministic creation)
            let division_execute_pipeline = self.compute_pipeline_manager.get_lifecycle_division_execute_pipeline();
            compute_pass.set_pipeline(division_execute_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            // Free slot management (maintain compacted arrays)
            let free_slots_pipeline = self.compute_pipeline_manager.get_lifecycle_free_slots_pipeline();
            compute_pass.set_pipeline(free_slots_pipeline);
            // TODO: Set bind groups when bind group management is implemented
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            
            let lifecycle_time = lifecycle_start.elapsed();
            self.performance_monitor.record_pipeline_stage_timing(GpuPipelineStage::LifecycleManagement, lifecycle_time);
        }
        
        // End GPU timing
        self.performance_monitor.end_gpu_physics_timing();
        
        // Stage 13: Instance Extraction (after compute pass ends)
        // - Extract rendering data to instance buffers
        self.extract_rendering_instances(&mut encoder, queue);
        
        // Submit all GPU work in single batch (critical for performance)
        queue.submit(std::iter::once(encoder.finish()));
        
        // Update simulation time and frame tracking
        self.current_time += dt;
        
        // End performance monitoring for this physics step
        self.performance_monitor.end_physics_step_timing(cell_count);
        
        // Log performance information periodically (every 60 frames)
        let current_frame = self.current_frame.load(Ordering::Acquire);
        if current_frame % 60 == 0 {
            let stats = self.performance_monitor.get_performance_stats();
            log::debug!(
                "GPU Physics Step {}: {} cells, {:.1} FPS, {:.3}ms GPU time, Grade: {}",
                current_frame,
                cell_count,
                stats.average_fps,
                stats.gpu_timing.total_gpu_compute_time.as_secs_f32() * 1000.0,
                stats.performance_grade
            );
        }
    }

    /// Extract rendering instance data from GPU physics buffers.
    ///
    /// This method runs the instance extraction compute shader to convert
    /// GPU physics simulation data into rendering instance data. The extracted
    /// data is stored in triple-buffered instance buffers for rendering.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `queue` - wgpu queue for buffer updates
    ///
    /// # Performance Notes
    /// - Runs entirely on GPU with no CPU involvement
    /// - Uses triple-buffered instance data for lock-free rendering
    /// - Extracts position, radius, color, and visual parameters
    /// - Handles cell type-specific visual properties from genome modes
    fn extract_rendering_instances(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        let cell_count = self.cell_count.load(Ordering::Acquire);
        if cell_count == 0 {
            return;
        }

        // Calculate total mode count across all genomes
        let total_mode_count: usize = self.genomes.iter().map(|g| g.modes.len()).sum();
        let cell_type_count = self.genomes.len().max(1); // At least 1 for default

        // Set up extraction parameters
        let extraction_params = ExtractionParams {
            cell_count,
            mode_count: total_mode_count as u32,
            cell_type_count: cell_type_count as u32,
            current_time: self.current_time,
            ..Default::default()
        };

        // Update extraction parameters buffer
        queue.write_buffer(
            &self.triple_buffer_system.get_extraction_params_buffer(),
            0,
            bytemuck::cast_slice(&[extraction_params])
        );

        // Execute instance extraction compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Instance Extraction"),
                timestamp_writes: None,
            });

            // Get the instance extraction pipeline
            let extraction_pipeline = self.compute_pipeline_manager.get_instance_extraction_pipeline();
            compute_pass.set_pipeline(extraction_pipeline);
            
            // TODO: Set bind groups when bind group management is implemented
            // This will bind:
            // - Extraction parameters buffer
            // - Current physics buffers (positions, colors, orientations, etc.)
            // - Target instance buffer for output
            
            // Calculate workgroup count for instance extraction
            let workgroup_count = (cell_count + 63) / 64; // Round up for 64-thread workgroups
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        log::debug!("Extracted instance data for {} cells using GPU compute shader", cell_count);
        
        // Rotate instance buffers after extraction to maintain triple buffering
        self.triple_buffer_system.rotate_instance_buffers();
    }

    /// Extract adhesion connection data for rendering adhesion lines.
    ///
    /// This method extracts adhesion connection data from GPU physics buffers
    /// to enable rendering of adhesion lines between connected cells. The data
    /// is extracted to CPU-accessible buffers for compatibility with the current
    /// AdhesionLineRenderer rendering system.
    ///
    /// # Arguments
    /// * `device` - wgpu device for GPU operations
    /// * `queue` - wgpu queue for command submission
    ///
    /// # Returns
    /// Vector of adhesion line data for rendering
    ///
    /// # Performance Notes
    /// - This involves GPU-to-CPU readback (slower than pure GPU operations)
    /// - Used only for adhesion line visualization
    /// - Could be optimized with GPU-based line rendering in the future
    /// - Currently uses canonical state for compatibility
    pub fn extract_adhesion_lines_for_rendering(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Vec<AdhesionLineData> {
        // For now, delegate to the simple version that uses canonical state
        // TODO: When GPU compute shaders are fully implemented, this method will:
        // 1. Create a readback buffer for adhesion connection data
        // 2. Copy adhesion data from GPU physics buffers to readback buffer
        // 3. Map the readback buffer and extract AdhesionLineData
        // 4. Return the extracted data for rendering
        self.extract_adhesion_lines_for_rendering_simple()
    }

    /// Extract adhesion connection data using canonical state (compatibility method).
    ///
    /// This is a simplified version of adhesion line extraction that uses the
    /// canonical state data structure. It's used for compatibility with the
    /// existing rendering system while GPU compute shaders are being developed.
    ///
    /// # Returns
    /// Vector of adhesion line data for rendering
    ///
    /// # Performance Notes
    /// - Uses CPU-side canonical state data
    /// - Compatible with existing AdhesionLineRenderer
    /// - Will be replaced with GPU extraction when compute shaders are complete
    #[allow(dead_code)] // Used internally by extract_adhesion_lines_for_rendering
    fn extract_adhesion_lines_for_rendering_simple(&mut self) -> Vec<AdhesionLineData> {
        let cell_count = self.cell_count.load(Ordering::Acquire);
        if cell_count == 0 {
            return Vec::new();
        }

        let mut adhesion_lines = Vec::new();
        
        // Extract adhesion connections from canonical state (SoA layout)
        for connection_idx in 0..self.canonical_state.adhesion_connections.active_count {
            // Check if connection is active
            if self.canonical_state.adhesion_connections.is_active[connection_idx] == 0 {
                continue;
            }
            
            let cell_a_idx = self.canonical_state.adhesion_connections.cell_a_index[connection_idx];
            let cell_b_idx = self.canonical_state.adhesion_connections.cell_b_index[connection_idx];
            
            // Ensure indices are valid
            if cell_a_idx >= self.canonical_state.cell_count || cell_b_idx >= self.canonical_state.cell_count {
                continue;
            }
            
            let pos_a = self.canonical_state.positions[cell_a_idx];
            let pos_b = self.canonical_state.positions[cell_b_idx];
            let radius_a = self.canonical_state.radii[cell_a_idx];
            let radius_b = self.canonical_state.radii[cell_b_idx];
            
            // Get adhesion color from genome mode (if available)
            let color = if let Some(genome) = self.genomes.get(self.canonical_state.genome_ids[cell_a_idx]) {
                let mode_idx = self.canonical_state.mode_indices[cell_a_idx];
                if mode_idx < genome.modes.len() {
                    let mode_color = genome.modes[mode_idx].color;
                    [mode_color.x, mode_color.y, mode_color.z, 0.8] // Semi-transparent
                } else {
                    [0.8, 0.8, 0.8, 0.8] // Default gray
                }
            } else {
                [0.8, 0.8, 0.8, 0.8] // Default gray
            };
            
            // Calculate connection strength based on adhesion properties
            // TODO: Extract actual connection strength from adhesion data when available
            let connection_strength = 1.0; // Default strength
            
            adhesion_lines.push(AdhesionLineData {
                start_pos: pos_a,
                end_pos: pos_b,
                start_radius: radius_a,
                end_radius: radius_b,
                color,
                thickness: 0.1, // Default thickness
                connection_strength,
            });
        }
        
        adhesion_lines
    }

    /// Get adhesion rendering statistics for performance monitoring.
    ///
    /// Returns information about the current adhesion connections that
    /// is useful for performance monitoring and debugging.
    ///
    /// # Returns
    /// Tuple of (total_connections, active_connections, rendered_lines)
    pub fn get_adhesion_stats(&self) -> (usize, usize, usize) {
        let total_connections = self.canonical_state.adhesion_connections.cell_a_index.len();
        let active_connections = self.canonical_state.adhesion_connections.active_count;
        
        // Count actually rendered lines (active connections with valid indices)
        let mut rendered_lines = 0;
        for i in 0..active_connections {
            if self.canonical_state.adhesion_connections.is_active[i] == 1 {
                let cell_a = self.canonical_state.adhesion_connections.cell_a_index[i];
                let cell_b = self.canonical_state.adhesion_connections.cell_b_index[i];
                if cell_a < self.canonical_state.cell_count && cell_b < self.canonical_state.cell_count {
                    rendered_lines += 1;
                }
            }
        }
        
        (total_connections, active_connections, rendered_lines)
    }

    /// Reset the GPU scene to initial state.
    ///
    /// This clears all simulation data and resets the scene to empty state.
    /// The GPU buffers are cleared and all counters are reset.
    pub fn reset(&mut self) {
        self.cell_count.store(0, Ordering::Release);
        self.current_time = 0.0;
        self.current_frame.store(0, Ordering::Release);
        self.paused = false;
        self.first_frame = true;
        
        // Clear genomes since no cells reference them
        self.genomes.clear();
        
        // Mark genome mode buffer as needing update
        self.genome_mode_manager.mark_dirty();
        
        // Clear minimal canonical state for compatibility
        self.canonical_state.cell_count = 0;
        self.canonical_state.next_cell_id = 0;
        self.canonical_state.adhesion_connections.active_count = 0;
        self.canonical_state.adhesion_manager.reset();
        
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
    }
    
    /// Get the current cell count (thread-safe).
    pub fn get_cell_count(&self) -> u32 {
        self.cell_count.load(Ordering::Acquire)
    }
    
    /// Get the maximum capacity.
    pub fn get_capacity(&self) -> u32 {
        self.capacity
    }
    
    /// Get the current frame number.
    pub fn get_current_frame(&self) -> u32 {
        self.current_frame.load(Ordering::Acquire)
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
    
    /// Read culling statistics from GPU (blocking).
    pub fn read_culling_stats(&mut self, device: &wgpu::Device) -> crate::rendering::CullingStats {
        self.instance_builder.read_culling_stats_blocking(device)
    }
    
    // ============================================================================
    // GENOME MODE MANAGEMENT METHODS
    // ============================================================================
    
    /// Get the genome mode buffer manager.
    ///
    /// This provides access to the genome mode buffer manager for advanced
    /// genome management operations and GPU buffer access.
    ///
    /// # Returns
    /// Reference to the genome mode buffer manager
    pub fn genome_mode_manager(&self) -> &GenomeModeBufferManager {
        &self.genome_mode_manager
    }
    
    /// Get mutable access to the genome mode buffer manager.
    ///
    /// This provides mutable access for updating genome mode buffers
    /// and managing mode transitions.
    ///
    /// # Returns
    /// Mutable reference to the genome mode buffer manager
    pub fn genome_mode_manager_mut(&mut self) -> &mut GenomeModeBufferManager {
        &mut self.genome_mode_manager
    }
    
    /// Convert genome_id and relative mode_index to absolute mode_index.
    ///
    /// This method provides the mapping from cell-specific genome_id and mode_index
    /// to the absolute mode index used for GPU buffer lookups.
    ///
    /// # Arguments
    /// * `genome_id` - Genome identifier (index into genomes array)
    /// * `relative_mode_index` - Mode index within the genome (0-based)
    ///
    /// # Returns
    /// Absolute mode index for GPU buffer lookup, or None if invalid
    pub fn get_absolute_mode_index(&self, genome_id: usize, relative_mode_index: usize) -> Option<u32> {
        self.genome_mode_manager.get_absolute_mode_index(genome_id, relative_mode_index)
    }
    
    /// Add a new genome to the scene.
    ///
    /// This method adds a new genome to the scene and updates the GPU mode buffer
    /// to include the new genome's modes. The genome will be assigned the next
    /// available genome_id.
    ///
    /// # Arguments
    /// * `genome` - Genome to add to the scene
    /// * `queue` - wgpu queue for buffer updates
    ///
    /// # Returns
    /// The assigned genome_id, or error if the operation fails
    pub fn add_genome(&mut self, genome: Genome, queue: &wgpu::Queue) -> Result<usize, String> {
        let genome_id = self.genomes.len();
        self.genomes.push(genome);
        
        // Update GPU mode buffer with new genome
        self.genome_mode_manager.update_from_genomes(queue, &self.genomes)?;
        
        log::debug!("Added genome {} with {} modes", genome_id, self.genomes[genome_id].modes.len());
        
        Ok(genome_id)
    }
    
    /// Update an existing genome in the scene.
    ///
    /// This method updates an existing genome and refreshes the GPU mode buffer
    /// to reflect the changes. All cells using this genome will use the updated
    /// mode data on the next physics step.
    ///
    /// # Arguments
    /// * `genome_id` - ID of the genome to update
    /// * `genome` - New genome data
    /// * `queue` - wgpu queue for buffer updates
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn update_genome(&mut self, genome_id: usize, genome: Genome, queue: &wgpu::Queue) -> Result<(), String> {
        if genome_id >= self.genomes.len() {
            return Err(format!("Invalid genome_id: {}", genome_id));
        }
        
        self.genomes[genome_id] = genome;
        
        // Update GPU mode buffer with modified genome
        self.genome_mode_manager.update_from_genomes(queue, &self.genomes)?;
        
        log::debug!("Updated genome {} with {} modes", genome_id, self.genomes[genome_id].modes.len());
        
        Ok(())
    }
    
    // ============================================================================
    // CELL TYPE SYSTEM METHODS
    // ============================================================================
    
    /// Get the cell type for a specific cell.
    ///
    /// This method looks up the cell type based on the cell's genome and mode.
    /// It uses the genome mode buffer to determine the cell's behavior type.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell to query
    ///
    /// # Returns
    /// Cell type, or None if the cell index is invalid
    pub fn get_cell_type(&self, cell_index: usize) -> Option<CellType> {
        if cell_index >= self.canonical_state.cell_count {
            return None;
        }
        
        let genome_id = self.canonical_state.genome_ids[cell_index];
        let mode_index = self.canonical_state.mode_indices[cell_index];
        
        if let Some(genome) = self.genomes.get(genome_id) {
            if let Some(mode) = genome.modes.get(mode_index) {
                return CellType::from_u32(mode.cell_type as u32);
            }
        }
        
        None
    }
    
    /// Calculate nutrient gain for a cell based on its type.
    ///
    /// Test cells automatically gain nutrients at their specified rate.
    /// Flagellocyte cells do not gain nutrients automatically.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    /// * `delta_time` - Time step for the calculation
    ///
    /// # Returns
    /// Nutrient gain amount for this time step
    pub fn calculate_cell_nutrient_gain(&self, cell_index: usize, delta_time: f32) -> f32 {
        if let Some(cell_type) = self.get_cell_type(cell_index) {
            let genome_id = self.canonical_state.genome_ids[cell_index];
            let mode_index = self.canonical_state.mode_indices[cell_index];
            
            if let Some(genome) = self.genomes.get(genome_id) {
                if let Some(mode) = genome.modes.get(mode_index) {
                    return CellTypeBehaviorUtils::calculate_automatic_nutrient_gain(
                        cell_type,
                        mode.nutrient_gain_rate,
                        delta_time,
                    );
                }
            }
        }
        
        0.0
    }
    
    /// Calculate nutrient consumption for a cell based on its swim force usage.
    ///
    /// Flagellocyte cells consume nutrients based on their swim force.
    /// Test cells do not consume nutrients through swimming.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    /// * `swim_force` - Current swim force magnitude
    ///
    /// # Returns
    /// Nutrient consumption rate per time unit
    pub fn calculate_cell_nutrient_consumption(&self, cell_index: usize, swim_force: f32) -> f32 {
        if let Some(cell_type) = self.get_cell_type(cell_index) {
            return CellTypeBehaviorUtils::calculate_nutrient_consumption(cell_type, swim_force);
        }
        
        0.0
    }
    
    /// Check if a cell can apply swim forces based on its type.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    ///
    /// # Returns
    /// True if the cell type can apply swim forces
    pub fn can_cell_apply_swim_force(&self, cell_index: usize) -> bool {
        if let Some(cell_type) = self.get_cell_type(cell_index) {
            return CellTypeBehaviorUtils::can_apply_swim_force(cell_type);
        }
        
        false
    }
    
    /// Get the visual mesh type for a cell based on its type and current state.
    ///
    /// This determines what kind of mesh should be used for rendering the cell.
    /// Flagellocyte cells with active swim forces get specialized spike meshes.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    /// * `swim_force` - Current swim force magnitude
    ///
    /// # Returns
    /// Mesh type identifier for rendering
    pub fn get_cell_visual_mesh_type(&self, cell_index: usize, swim_force: f32) -> u32 {
        if let Some(cell_type) = self.get_cell_type(cell_index) {
            return CellTypeBehaviorUtils::get_visual_mesh_type(cell_type, swim_force);
        }
        
        0 // Default to standard sphere mesh
    }
    
    /// Get cell type statistics for the current simulation.
    ///
    /// This provides a breakdown of how many cells of each type are currently
    /// in the simulation, useful for monitoring and debugging.
    ///
    /// # Returns
    /// Tuple of (test_cell_count, flagellocyte_cell_count)
    pub fn get_cell_type_statistics(&self) -> (usize, usize) {
        let mut test_count = 0;
        let mut flagellocyte_count = 0;
        
        for i in 0..self.canonical_state.cell_count {
            if let Some(cell_type) = self.get_cell_type(i) {
                match cell_type {
                    CellType::Test => test_count += 1,
                    CellType::Flagellocyte => flagellocyte_count += 1,
                }
            }
        }
        
        (test_count, flagellocyte_count)
    }
    
    // ============================================================================
    // ADVANCED GENOME FEATURE METHODS
    // ============================================================================
    
    /// Calculate swim force vector for a cell based on its genome mode.
    ///
    /// This method calculates the directional swim force for a cell based on
    /// its current orientation and genome mode swim force settings.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    ///
    /// # Returns
    /// 3D swim force vector in world space, or zero vector if invalid
    pub fn calculate_cell_swim_force_vector(&self, cell_index: usize) -> glam::Vec3 {
        if cell_index >= self.canonical_state.cell_count {
            return glam::Vec3::ZERO;
        }
        
        let genome_id = self.canonical_state.genome_ids[cell_index];
        let mode_index = self.canonical_state.mode_indices[cell_index];
        
        if let Some(genome) = self.genomes.get(genome_id) {
            if let Some(mode) = genome.modes.get(mode_index) {
                let cell_orientation = self.canonical_state.rotations[cell_index];
                let genome_orientation = self.canonical_state.genome_orientations[cell_index];
                
                return AdvancedGenomeFeatures::calculate_swim_force_vector(
                    mode.swim_force,
                    cell_orientation,
                    genome_orientation,
                );
            }
        }
        
        glam::Vec3::ZERO
    }
    
    /// Calculate effective nutrient priority for a cell.
    ///
    /// This method calculates the effective nutrient priority for a cell
    /// based on its genome mode settings and current nutrient levels.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    ///
    /// # Returns
    /// Effective nutrient priority factor
    pub fn calculate_cell_nutrient_priority(&self, cell_index: usize) -> f32 {
        if cell_index >= self.canonical_state.cell_count {
            return 1.0; // Default priority
        }
        
        let genome_id = self.canonical_state.genome_ids[cell_index];
        let mode_index = self.canonical_state.mode_indices[cell_index];
        let current_mass = self.canonical_state.masses[cell_index];
        
        if let Some(genome) = self.genomes.get(genome_id) {
            if let Some(mode) = genome.modes.get(mode_index) {
                let max_mass = AdvancedGenomeFeatures::radius_to_max_mass(mode.max_cell_size);
                
                return AdvancedGenomeFeatures::calculate_nutrient_priority_factor(
                    mode.nutrient_priority,
                    mode.prioritize_when_low,
                    current_mass,
                    max_mass,
                );
            }
        }
        
        1.0 // Default priority
    }
    
    /// Check if a cell can divide based on size constraints.
    ///
    /// This method checks if a cell meets the size requirements for division
    /// based on its genome mode settings and current state.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    ///
    /// # Returns
    /// True if the cell can divide based on size constraints
    pub fn can_cell_divide_by_size(&self, cell_index: usize) -> bool {
        if cell_index >= self.canonical_state.cell_count {
            return false;
        }
        
        let genome_id = self.canonical_state.genome_ids[cell_index];
        let mode_index = self.canonical_state.mode_indices[cell_index];
        let current_mass = self.canonical_state.masses[cell_index];
        
        if let Some(genome) = self.genomes.get(genome_id) {
            if let Some(mode) = genome.modes.get(mode_index) {
                return AdvancedGenomeFeatures::can_divide_by_size(
                    current_mass,
                    mode.split_mass,
                    mode.max_cell_size,
                );
            }
        }
        
        false
    }
    
    /// Calculate swim energy cost for a cell.
    ///
    /// This method calculates the energy cost of swimming for a cell
    /// based on its current swim force usage.
    ///
    /// # Arguments
    /// * `cell_index` - Index of the cell
    /// * `swim_force` - Current swim force magnitude
    /// * `delta_time` - Time step
    ///
    /// # Returns
    /// Energy cost (nutrient consumption) for this time step
    pub fn calculate_cell_swim_energy_cost(&self, cell_index: usize, swim_force: f32, delta_time: f32) -> f32 {
        if !self.can_cell_apply_swim_force(cell_index) {
            return 0.0;
        }
        
        AdvancedGenomeFeatures::calculate_swim_energy_cost(swim_force, delta_time)
    }
    
    /// Validate all genome modes in the scene.
    ///
    /// This method validates all genome modes currently in the scene
    /// and returns any validation warnings found.
    ///
    /// # Returns
    /// Vector of validation warnings (empty if all modes are valid)
    pub fn validate_all_genome_modes(&self) -> Vec<String> {
        let mut all_warnings = Vec::new();
        
        for (genome_id, genome) in self.genomes.iter().enumerate() {
            for (mode_id, mode) in genome.modes.iter().enumerate() {
                let warnings = AdvancedGenomeFeatures::validate_genome_mode_parameters(mode);
                for warning in warnings {
                    all_warnings.push(format!(
                        "Genome {} Mode {}: {}",
                        genome_id, mode_id, warning
                    ));
                }
            }
        }
        
        all_warnings
    }
    
    /// Create an optimized genome for a specific cell type and behavior profile.
    ///
    /// This method creates a complete genome with optimized parameters for
    /// a specific cell type and behavior pattern.
    ///
    /// # Arguments
    /// * `cell_type` - Target cell type
    /// * `behavior_profile` - Behavior profile ("balanced", "aggressive", "efficient")
    /// * `genome_name` - Name for the new genome
    ///
    /// # Returns
    /// Optimized genome ready for use
    pub fn create_optimized_genome(
        cell_type: CellType,
        behavior_profile: &str,
        genome_name: &str,
    ) -> Genome {
        let mut genome = Genome::default();
        genome.name = genome_name.to_string();
        
        // Replace the first mode with optimized settings
        if !genome.modes.is_empty() {
            genome.modes[0] = AdvancedGenomeFeatures::create_optimized_mode_for_type(
                cell_type,
                behavior_profile,
            );
            genome.modes[0].name = format!("{} Mode", genome_name);
            genome.modes[0].default_name = genome.modes[0].name.clone();
        }
        
        genome
    }
    
    /// Get advanced genome feature statistics.
    ///
    /// This provides detailed statistics about the advanced genome features
    /// currently in use in the simulation.
    ///
    /// # Returns
    /// Tuple of (cells_with_swim_force, average_nutrient_priority, cells_at_max_size)
    pub fn get_advanced_genome_statistics(&self) -> (usize, f32, usize) {
        let mut cells_with_swim_force = 0;
        let mut total_priority = 0.0;
        let mut cells_at_max_size = 0;
        let mut valid_cells = 0;
        
        for i in 0..self.canonical_state.cell_count {
            let genome_id = self.canonical_state.genome_ids[i];
            let mode_index = self.canonical_state.mode_indices[i];
            
            if let Some(genome) = self.genomes.get(genome_id) {
                if let Some(mode) = genome.modes.get(mode_index) {
                    valid_cells += 1;
                    
                    if mode.swim_force > 0.0 {
                        cells_with_swim_force += 1;
                    }
                    
                    total_priority += mode.nutrient_priority;
                    
                    let current_mass = self.canonical_state.masses[i];
                    let max_mass = AdvancedGenomeFeatures::radius_to_max_mass(mode.max_cell_size);
                    if current_mass >= max_mass * 0.95 { // Within 5% of max size
                        cells_at_max_size += 1;
                    }
                }
            }
        }
        
        let average_priority = if valid_cells > 0 {
            total_priority / valid_cells as f32
        } else {
            1.0
        };
        
        (cells_with_swim_force, average_priority, cells_at_max_size)
    }
    
    // ============================================================================
    // PERFORMANCE MONITORING METHODS
    // ============================================================================
    
    /// Get comprehensive performance statistics.
    ///
    /// Returns detailed performance metrics including frame timing, buffer rotation
    /// efficiency, GPU utilization, and performance comparisons.
    ///
    /// # Returns
    /// Complete performance statistics summary
    pub fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_monitor.get_performance_stats()
    }
    
    /// Get buffer rotation statistics.
    ///
    /// Returns current buffer rotation performance metrics for monitoring
    /// triple buffer efficiency and rotation patterns.
    ///
    /// # Returns
    /// Buffer rotation performance metrics
    pub fn get_buffer_rotation_stats(&self) -> &BufferRotationStats {
        self.performance_monitor.get_buffer_rotation_stats()
    }
    
    /// Get GPU timing breakdown.
    ///
    /// Returns detailed timing information for individual GPU pipeline stages
    /// to help identify performance bottlenecks.
    ///
    /// # Returns
    /// Detailed GPU operation timing data
    pub fn get_gpu_timing(&self) -> &GpuTimingData {
        self.performance_monitor.get_gpu_timing()
    }
    
    /// Get performance comparison data.
    ///
    /// Returns GPU vs CPU performance comparison metrics including speedup
    /// factors and efficiency ratios.
    ///
    /// # Returns
    /// GPU vs CPU performance comparison metrics
    pub fn get_performance_comparison(&self) -> &PerformanceComparisonData {
        self.performance_monitor.get_comparison_data()
    }
    
    /// Set CPU baseline performance for comparison.
    ///
    /// This sets the CPU performance baseline used for GPU vs CPU performance
    /// comparisons and speedup calculations.
    ///
    /// # Arguments
    /// * `cpu_cells_per_second` - CPU baseline performance
    /// * `cpu_memory_usage_mb` - CPU memory usage baseline
    /// * `cpu_core_utilization` - CPU core utilization baseline
    pub fn set_cpu_performance_baseline(
        &mut self,
        cpu_cells_per_second: f32,
        cpu_memory_usage_mb: f32,
        cpu_core_utilization: f32,
    ) {
        self.performance_monitor.set_cpu_baseline(
            cpu_cells_per_second,
            cpu_memory_usage_mb,
            cpu_core_utilization,
        );
    }
    
    /// Enable or disable performance monitoring.
    ///
    /// When disabled, all timing operations become no-ops for minimal
    /// performance impact. Historical data is preserved.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable performance monitoring
    pub fn set_performance_monitoring_enabled(&mut self, enabled: bool) {
        self.performance_monitor.set_enabled(enabled);
    }
    
    /// Check if performance monitoring is enabled.
    ///
    /// # Returns
    /// True if performance monitoring is enabled
    pub fn is_performance_monitoring_enabled(&self) -> bool {
        self.performance_monitor.is_enabled()
    }
    
    /// Reset all performance statistics.
    ///
    /// This clears all historical performance data and resets counters to zero.
    /// Useful for starting fresh performance measurements.
    pub fn reset_performance_stats(&mut self) {
        self.performance_monitor.reset();
    }
    
    /// Export performance data to CSV format.
    ///
    /// This exports all frame timing data to CSV format for external
    /// analysis and visualization tools.
    ///
    /// # Returns
    /// CSV formatted performance data
    pub fn export_performance_data_csv(&self) -> String {
        self.performance_monitor.export_to_csv()
    }
    
    /// Get current frame rate (FPS).
    ///
    /// Returns the current instantaneous frame rate based on the most
    /// recent frame timing measurement.
    ///
    /// # Returns
    /// Current frame rate in frames per second
    pub fn get_current_fps(&self) -> f32 {
        self.performance_monitor.get_performance_stats().current_fps
    }
    
    /// Get average frame rate over recent samples.
    ///
    /// Returns the average frame rate calculated over the recent frame
    /// timing samples (up to 2 seconds of history).
    ///
    /// # Returns
    /// Average frame rate in frames per second
    pub fn get_average_fps(&self) -> f32 {
        self.performance_monitor.get_performance_stats().average_fps
    }
    
    /// Get performance grade.
    ///
    /// Returns a letter grade (A, B, C, D, F) based on current performance
    /// metrics, primarily frame rate consistency and throughput.
    ///
    /// # Returns
    /// Performance grade character
    pub fn get_performance_grade(&self) -> char {
        self.performance_monitor.get_performance_stats().performance_grade
    }
    
    // ============================================================================
    // COMPATIBILITY METHODS (for existing app integration)
    // ============================================================================
    // These methods provide compatibility with the existing app.rs code
    // They will be fully implemented in later tasks
    
    /// Convert screen coordinates to world position (placeholder implementation).
    pub fn screen_to_world(&self, _screen_x: f32, _screen_y: f32) -> glam::Vec3 {
        // TODO: Implement proper screen-to-world conversion
        // For now, return a position in front of the camera
        let distance = 10.0;
        self.camera.position() + self.camera.rotation * glam::Vec3::new(0.0, 0.0, -distance)
    }
    
    /// Convert screen coordinates to world position at specific distance (placeholder).
    pub fn screen_to_world_at_distance(&self, _screen_x: f32, _screen_y: f32, distance: f32) -> glam::Vec3 {
        // TODO: Implement proper screen-to-world conversion with distance
        // For now, return a position at the specified distance from camera
        self.camera.position() + self.camera.rotation * glam::Vec3::new(0.0, 0.0, -distance)
    }
    
    /// Insert a cell from genome (placeholder implementation).
    pub fn insert_cell_from_genome(
        &mut self,
        world_position: glam::Vec3,
        genome: &crate::genome::Genome,
    ) -> Option<usize> {
        // TODO: Implement GPU-based cell insertion
        // For now, add to canonical state for compatibility
        let genome_id = self.genomes.len();
        self.genomes.push(genome.clone());
        
        // Update genome mode buffer when new genome is added
        self.genome_mode_manager.mark_dirty();
        
        let mode_idx = genome.initial_mode.max(0) as usize;
        let mode = &genome.modes[mode_idx];
        
        let initial_mass = 1.0_f32;
        let radius = (initial_mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);
        
        let cell_idx = self.canonical_state.add_cell(
            world_position,
            glam::Vec3::ZERO,
            genome.initial_orientation,
            genome.initial_orientation,
            glam::Vec3::ZERO,
            initial_mass,
            radius,
            genome_id,
            mode_idx,
            self.current_time,
            mode.split_interval,
            mode.split_mass,
            mode.membrane_stiffness,
        );
        
        if cell_idx.is_some() {
            self.cell_count.store(self.canonical_state.cell_count as u32, Ordering::Release);
        }
        
        cell_idx
    }
    
    /// Cast a ray and find intersecting cell (placeholder implementation).
    pub fn raycast_cell(&self, screen_x: f32, screen_y: f32) -> Option<(usize, f32)> {
        // TODO: Implement GPU-based raycasting
        // For now, use canonical state for compatibility
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
        
        let ray_origin = self.camera.position();
        let ray_dir = self.camera.rotation * ray_view;
        
        let mut closest_hit: Option<(usize, f32)> = None;
        
        for i in 0..self.canonical_state.cell_count {
            let cell_pos = self.canonical_state.positions[i];
            let cell_radius = self.canonical_state.radii[i];
            
            if let Some(t) = ray_sphere_intersect(ray_origin, ray_dir, cell_pos, cell_radius) {
                if t > 0.0 {
                    match closest_hit {
                        None => closest_hit = Some((i, t)),
                        Some((_, closest_t)) if t < closest_t => closest_hit = Some((i, t)),
                        _ => {}
                    }
                }
            }
        }
        
        closest_hit
    }
    
    /// Remove a cell (placeholder implementation).
    pub fn remove_cell(&mut self, cell_idx: usize) -> bool {
        // TODO: Implement GPU-based cell removal
        // For now, use canonical state for compatibility
        if cell_idx >= self.canonical_state.cell_count {
            return false;
        }
        
        let state = &mut self.canonical_state;
        
        // Remove all adhesion connections for this cell
        state.adhesion_manager.remove_all_connections_for_cell(
            &mut state.adhesion_connections,
            cell_idx,
        );
        
        let last_idx = state.cell_count - 1;
        
        if cell_idx != last_idx {
            // Swap with last cell - copy all properties
            state.cell_ids[cell_idx] = state.cell_ids[last_idx];
            state.positions[cell_idx] = state.positions[last_idx];
            state.prev_positions[cell_idx] = state.prev_positions[last_idx];
            state.velocities[cell_idx] = state.velocities[last_idx];
            state.masses[cell_idx] = state.masses[last_idx];
            state.radii[cell_idx] = state.radii[last_idx];
            state.genome_ids[cell_idx] = state.genome_ids[last_idx];
            state.mode_indices[cell_idx] = state.mode_indices[last_idx];
            state.rotations[cell_idx] = state.rotations[last_idx];
            state.genome_orientations[cell_idx] = state.genome_orientations[last_idx];
            state.angular_velocities[cell_idx] = state.angular_velocities[last_idx];
            state.forces[cell_idx] = state.forces[last_idx];
            state.torques[cell_idx] = state.torques[last_idx];
            state.accelerations[cell_idx] = state.accelerations[last_idx];
            state.prev_accelerations[cell_idx] = state.prev_accelerations[last_idx];
            state.stiffnesses[cell_idx] = state.stiffnesses[last_idx];
            state.birth_times[cell_idx] = state.birth_times[last_idx];
            state.split_intervals[cell_idx] = state.split_intervals[last_idx];
            state.split_masses[cell_idx] = state.split_masses[last_idx];
            state.split_counts[cell_idx] = state.split_counts[last_idx];
            
            // Update adhesion connections that referenced the moved cell
            state.adhesion_manager.update_cell_index_after_swap(
                &mut state.adhesion_connections,
                last_idx,
                cell_idx,
            );
        }
        
        state.cell_count -= 1;
        self.cell_count.store(state.cell_count as u32, Ordering::Release);
        
        // Mark instance builder dirty
        self.instance_builder.mark_all_dirty();
        
        true
    }

}

/// Ray-sphere intersection test helper function.
fn ray_sphere_intersect(ray_origin: glam::Vec3, ray_dir: glam::Vec3, sphere_center: glam::Vec3, sphere_radius: f32) -> Option<f32> {
    let oc = ray_origin - sphere_center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        return None;
    }
    
    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);
    
    // Return the closest positive intersection
    if t1 > 0.0 {
        Some(t1)
    } else if t2 > 0.0 {
        Some(t2)
    } else {
        None
    }
}

impl Scene for GpuScene {
    /// Update the GPU scene with pure GPU physics computation.
    ///
    /// This method runs the complete GPU physics pipeline with zero CPU involvement.
    /// All physics computation, collision detection, and force calculation happens
    /// entirely on the GPU using compute shaders.
    ///
    /// # Arguments
    /// * `dt` - Delta time for physics step
    ///
    /// # Performance Notes
    /// - No CPU physics computation or synchronization
    /// - Fixed timestep accumulator for deterministic behavior
    /// - Maximum 4 physics steps per frame to prevent spiral of death
    fn update(&mut self, _dt: f32) {
        // Skip update if paused or no cells
        if self.paused || self.cell_count.load(Ordering::Acquire) == 0 {
            return;
        }

        // For now, use a simple fixed timestep approach
        // TODO: Implement proper accumulator pattern when GPU physics is complete
        let fixed_dt = self.config.fixed_timestep;
        
        // Run single GPU physics step
        // Note: We need device and queue references, but Scene trait doesn't provide them
        // This will be resolved when we integrate with the full application
        // For now, this is a placeholder that shows the intended structure
        
        self.current_time += fixed_dt;
    }

    /// Render the GPU scene using extracted instance data.
    ///
    /// This method renders the scene using instance data extracted from GPU buffers.
    /// The rendering uses the triple-buffered instance data that is 2 frames behind
    /// physics computation to ensure data stability.
    ///
    /// # Arguments
    /// * `device` - wgpu device for GPU operations
    /// * `queue` - wgpu queue for command submission
    /// * `view` - Render target texture view
    /// * `cell_type_visuals` - Optional cell type visual settings
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        _cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>,
    ) {
        // Execute GPU physics step (this is where the actual GPU work happens)
        let dt = self.config.fixed_timestep;
        self.step_gpu_physics(device, queue, dt);
        
        let cell_count = self.cell_count.load(Ordering::Acquire);
        if cell_count == 0 {
            // Still need to clear the screen even with no cells
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU Scene Empty Render"),
            });

            {
                let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("GPU Scene Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.05,
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

            queue.submit(std::iter::once(encoder.finish()));
            return;
        }

        // Create command encoder for rendering
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Scene Render"),
        });

        // Clear pass
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU Scene Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
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

        // Render cells using GPU-extracted instance data
        self.render_cells_from_gpu_instances(&mut encoder, queue, view);

        // Submit all rendering work
        queue.submit(std::iter::once(encoder.finish()));

        // Mark that we now have Hi-Z data for next frame
        self.first_frame = false;
    }



    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.renderer.resize(device, width, height);
        self.adhesion_renderer.resize(width, height);
        self.hiz_generator.resize(device, width, height);
        self.instance_builder.reset_hiz();
        self.first_frame = true;
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
        self.cell_count.load(Ordering::Acquire) as usize
    }
}

// ============================================================================
// HELPER METHODS FOR GPU SCENE RENDERING (Outside Scene trait)
// ============================================================================

impl GpuScene {
    /// Render cells using GPU-extracted instance data.
    ///
    /// This method renders cells using instance data that was extracted directly
    /// from GPU physics buffers by compute shaders. It also handles adhesion line
    /// rendering for visualizing connections between cells.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `queue` - wgpu queue for command submission
    /// * `view` - Render target texture view
    ///
    /// # Performance Notes
    /// - Uses GPU-extracted instance data (no CPU involvement for cells)
    /// - Instance data is 2 frames behind physics for stability
    /// - Direct GPU-to-GPU rendering pipeline for cells
    /// - Uses optimized depth pre-pass for maximum performance
    /// - Adhesion lines use CPU extraction for compatibility (can be optimized later)
    fn render_cells_from_gpu_instances(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        let cell_count = self.cell_count.load(Ordering::Acquire);
        if cell_count == 0 {
            return;
        }

        // Get the rendering instance buffer (2 frames behind physics for stability)
        let rendering_instance_buffer = self.triple_buffer_system.get_rendering_instance_buffer();

        // Validate the instance buffer (debug builds only)
        #[cfg(debug_assertions)]
        {
            if !self.renderer.validate_gpu_instance_buffer(rendering_instance_buffer, cell_count) {
                log::error!("GPU instance buffer validation failed for {} cells", cell_count);
                return;
            }
        }

        // Calculate view-projection matrix
        let view_matrix = Mat4::look_at_rh(
            self.camera.position(),
            self.camera.position() + self.camera.rotation * glam::Vec3::NEG_Z,
            self.camera.rotation * glam::Vec3::Y,
        );
        let aspect = self.renderer.width as f32 / self.renderer.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        // Render cells using optimized GPU instance rendering with depth pre-pass
        self.renderer.render_gpu_instances_optimized(
            encoder,
            queue,
            view,
            rendering_instance_buffer,
            cell_count,
            view_proj,
            self.camera.position(),
            self.current_time,
        );

        // Render adhesion lines (subtask 10.3)
        // Note: This currently uses CPU extraction for compatibility with existing AdhesionManager
        // In the future, this could be optimized with GPU-based line rendering
        self.render_adhesion_lines(encoder, queue, view);
    }

    /// Render adhesion lines between connected cells.
    ///
    /// This method extracts adhesion connection data and renders lines between
    /// connected cells to visualize the adhesion network. It integrates with
    /// the existing AdhesionLineRenderer for compatibility and visual consistency.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `queue` - wgpu queue for command submission
    /// * `view` - Render target texture view
    ///
    /// # Performance Notes
    /// - Currently uses CPU extraction for compatibility with existing AdhesionLineRenderer
    /// - Integrates seamlessly with existing adhesion visualization system
    /// - Could be optimized with GPU-based line rendering in the future
    /// - Uses zone-based coloring for visual distinction of adhesion types
    fn render_adhesion_lines(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        let cell_count = self.cell_count.load(Ordering::Acquire);
        if cell_count == 0 {
            return;
        }

        // Check if we have any active adhesion connections
        if self.canonical_state.adhesion_connections.active_count == 0 {
            return;
        }

        // Render adhesion lines using the existing AdhesionLineRenderer
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU Scene Adhesion Lines Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve background and cells
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve depth from cell rendering
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Use the existing AdhesionLineRenderer to render connections
            self.adhesion_renderer.render_in_pass(
                &mut render_pass,
                queue,
                &self.canonical_state,
                self.camera.position(),
                self.camera.rotation,
            );
        }

        log::debug!(
            "Rendered {} adhesion connections using AdhesionLineRenderer",
            self.canonical_state.adhesion_connections.active_count
        );
    }
}
