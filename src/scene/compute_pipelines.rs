//! # GPU Compute Pipeline Infrastructure
//!
//! This module provides the infrastructure for creating and managing GPU compute pipelines
//! for the Bio-Spheres GPU scene. It includes pipeline creation utilities, bind group
//! management, and compute shader compilation.
//!
//! ## Pipeline Architecture
//!
//! The GPU scene uses a series of compute pipelines that execute in strict order:
//! 1. **Spatial Grid Pipelines** - Build spatial partitioning for collision detection
//! 2. **Physics Pipelines** - Calculate forces and integrate motion
//! 3. **Lifecycle Pipelines** - Handle cell division, death, and lifecycle management
//! 4. **Rendering Pipelines** - Extract visual data for rendering
//!
//! ## Compute Shader Organization
//!
//! Each pipeline corresponds to a specific WGSL compute shader:
//! - `spatial/grid_clear.wgsl` - Clear spatial grid cell counts
//! - `spatial/grid_assign.wgsl` - Assign cells to grid cells
//! - `physics/collision_detection.wgsl` - Detect cell-cell collisions
//! - `physics/force_calculation.wgsl` - Calculate all forces
//! - `lifecycle/cell_division.wgsl` - Handle cell division
//!
//! ## Performance Optimization
//!
//! - **Workgroup Sizes**: Optimized for GPU architecture (typically 64 or 256 threads)
//! - **Memory Access**: Designed for coalesced memory access patterns
//! - **Pipeline Batching**: Multiple dispatches batched into single command buffer
//! - **Resource Binding**: Efficient bind group management with minimal state changes

use wgpu;
use std::collections::HashMap;

/// Compute workgroup sizes optimized for GPU architecture.
///
/// These sizes are chosen based on GPU warp/wavefront sizes and memory
/// access patterns. Different operations may benefit from different sizes.
///
/// ## Size Guidelines
/// - **64 threads**: Good balance for most operations, fits 2 warps on NVIDIA
/// - **256 threads**: Optimal for memory-bound operations with good coalescing
/// - **32 threads**: Minimal size, good for complex operations with divergence
#[derive(Clone, Copy, Debug)]
pub struct ComputeWorkgroupSizes {
    /// Spatial grid operations (memory-intensive)
    pub spatial_grid: u32,
    /// Physics calculations (compute-intensive)
    pub physics: u32,
    /// Adhesion processing (moderate complexity)
    pub adhesion: u32,
    /// Lifecycle events (complex logic with divergence)
    pub lifecycle: u32,
}

impl Default for ComputeWorkgroupSizes {
    fn default() -> Self {
        Self {
            spatial_grid: 256,  // Memory-bound operations benefit from larger workgroups
            physics: 64,        // Balanced for compute and memory access
            adhesion: 64,       // Similar to physics
            lifecycle: 32,      // Complex logic with potential divergence
        }
    }
}

/// Compute pipeline manager for GPU scene operations.
///
/// This struct manages all compute pipelines used by the GPU scene including
/// their creation, bind group layouts, and execution. It provides a centralized
/// way to manage the complex compute shader pipeline.
///
/// ## Pipeline Categories
/// - **Spatial**: Grid-based spatial partitioning for collision detection
/// - **Physics**: Force calculation and motion integration
/// - **Lifecycle**: Cell division, death, and lifecycle management
/// - **Rendering**: Visual data extraction for rendering
pub struct ComputePipelineManager {
    /// All compute pipelines indexed by name
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    
    /// Bind group layouts for different pipeline categories
    bind_group_layouts: HashMap<String, wgpu::BindGroupLayout>,
    
    /// Workgroup sizes for different operation types
    workgroup_sizes: ComputeWorkgroupSizes,
    
    /// Device reference for pipeline creation
    device: wgpu::Device,
}

impl ComputePipelineManager {
    /// Create a new compute pipeline manager.
    ///
    /// This initializes the manager but does not create any pipelines yet.
    /// Pipelines are created on-demand when first requested.
    ///
    /// # Arguments
    /// * `device` - wgpu device for pipeline creation
    /// * `workgroup_sizes` - Workgroup sizes for different operation types
    pub fn new(device: wgpu::Device, workgroup_sizes: ComputeWorkgroupSizes) -> Self {
        Self {
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            workgroup_sizes,
            device,
        }
    }
    
    /// Create or get a compute pipeline by name.
    ///
    /// This method creates compute pipelines on-demand and caches them for
    /// future use. The shader source is loaded from the shaders directory
    /// based on the pipeline name.
    ///
    /// # Arguments
    /// * `name` - Pipeline name (e.g., "spatial_grid_clear")
    /// * `shader_path` - Path to WGSL shader file relative to shaders directory
    /// * `bind_group_layout` - Bind group layout for this pipeline
    ///
    /// # Returns
    /// Reference to the compute pipeline
    ///
    /// # Panics
    /// Panics if shader compilation fails or pipeline creation fails
    pub fn get_or_create_pipeline(
        &mut self,
        name: &str,
        shader_path: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> &wgpu::ComputePipeline {
        if !self.pipelines.contains_key(name) {
            let pipeline = self.create_compute_pipeline(name, shader_path, bind_group_layout);
            self.pipelines.insert(name.to_string(), pipeline);
        }
        
        self.pipelines.get(name).unwrap()
    }
    
    /// Create a compute pipeline from WGSL shader source.
    ///
    /// This method compiles a WGSL compute shader and creates a compute pipeline
    /// with the specified bind group layout.
    ///
    /// # Arguments
    /// * `label` - Debug label for the pipeline
    /// * `shader_path` - Path to WGSL shader file
    /// * `bind_group_layout` - Bind group layout for resource binding
    ///
    /// # Returns
    /// Compiled compute pipeline ready for dispatch
    fn create_compute_pipeline(
        &self,
        label: &str,
        shader_path: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        // Load shader source from file
        // For now, we'll use placeholder shader source since we haven't created the shaders yet
        let shader_source = self.get_placeholder_shader_source(shader_path);
        
        // Create shader module
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", label)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }
    
    /// Load shader source from file or return placeholder for development.
    ///
    /// This method loads WGSL compute shaders from the shaders directory.
    /// If the file doesn't exist, it returns a placeholder shader for development.
    ///
    /// # Arguments
    /// * `shader_path` - Path to the shader file relative to shaders directory
    ///
    /// # Returns
    /// WGSL shader source code as string
    fn get_placeholder_shader_source(&self, shader_path: &str) -> String {
        // Try to load actual shader file first
        let full_path = format!("shaders/{}", shader_path);
        if let Ok(shader_source) = std::fs::read_to_string(&full_path) {
            return shader_source;
        }
        
        // Fall back to placeholder shaders for development
        match shader_path {
            "spatial/grid_clear.wgsl" => {
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read_write> grid_counts: array<u32>;
                @group(0) @binding(2) var<storage, read_write> grid_offsets: array<u32>;
                
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let grid_index = global_id.x;
                    let grid_resolution = u32(physics_params.grid_resolution);
                    let total_grid_cells = grid_resolution * grid_resolution * grid_resolution;
                    
                    if (grid_index >= total_grid_cells) {
                        return;
                    }
                    
                    grid_counts[grid_index] = 0u;
                    grid_offsets[grid_index] = 0u;
                }
                "#.to_string()
            }
            "spatial/grid_assign.wgsl" => {
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> grid_counts: array<atomic<u32>>;
                @group(0) @binding(3) var<storage, read_write> grid_assignments: array<u32>;
                
                fn world_to_grid(world_pos: vec3<f32>, world_size: f32, grid_resolution: i32) -> vec3<i32> {
                    let half_world = world_size * 0.5;
                    let normalized_pos = (world_pos + vec3<f32>(half_world)) / world_size;
                    let grid_pos = normalized_pos * f32(grid_resolution);
                    return vec3<i32>(
                        clamp(i32(grid_pos.x), 0, grid_resolution - 1),
                        clamp(i32(grid_pos.y), 0, grid_resolution - 1),
                        clamp(i32(grid_pos.z), 0, grid_resolution - 1)
                    );
                }
                
                fn grid_coords_to_index(grid_coords: vec3<i32>, grid_resolution: i32) -> u32 {
                    return u32(grid_coords.x + grid_coords.y * grid_resolution + grid_coords.z * grid_resolution * grid_resolution);
                }
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    
                    let position = positions[cell_index].xyz;
                    let grid_coords = world_to_grid(position, physics_params.world_size, physics_params.grid_resolution);
                    let grid_index = grid_coords_to_index(grid_coords, physics_params.grid_resolution);
                    
                    grid_assignments[cell_index] = grid_index;
                    atomicAdd(&grid_counts[grid_index], 1u);
                }
                "#.to_string()
            }
            "spatial/grid_insert.wgsl" => {
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> grid_assignments: array<u32>;
                @group(0) @binding(2) var<storage, read> grid_offsets: array<u32>;
                @group(0) @binding(3) var<storage, read_write> grid_indices: array<u32>;
                @group(0) @binding(4) var<storage, read_write> grid_insertion_counters: array<atomic<u32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    
                    let grid_index = grid_assignments[cell_index];
                    let grid_start_offset = grid_offsets[grid_index];
                    let insertion_offset = atomicAdd(&grid_insertion_counters[grid_index], 1u);
                    let final_index = grid_start_offset + insertion_offset;
                    
                    let max_cells_per_grid = u32(physics_params.max_cells_per_grid);
                    if (insertion_offset < max_cells_per_grid) {
                        grid_indices[final_index] = cell_index;
                    }
                }
                "#.to_string()
            }
            "spatial/grid_prefix_sum.wgsl" => {
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> grid_counts: array<u32>;
                @group(0) @binding(2) var<storage, read_write> grid_offsets: array<u32>;
                
                var<workgroup> shared_data: array<u32, 256>;
                
                fn workgroup_prefix_sum(local_id: u32, value: u32) -> u32 {
                    shared_data[local_id] = value;
                    workgroupBarrier();
                    
                    var stride = 1u;
                    while (stride < 256u) {
                        if (local_id % (stride * 2u) == 0u && local_id + stride < 256u) {
                            shared_data[local_id + stride * 2u - 1u] += shared_data[local_id + stride - 1u];
                        }
                        stride *= 2u;
                        workgroupBarrier();
                    }
                    
                    if (local_id == 0u) {
                        shared_data[255] = 0u;
                    }
                    workgroupBarrier();
                    
                    stride = 128u;
                    while (stride > 0u) {
                        if (local_id % (stride * 2u) == 0u && local_id + stride < 256u) {
                            let temp = shared_data[local_id + stride - 1u];
                            shared_data[local_id + stride - 1u] = shared_data[local_id + stride * 2u - 1u];
                            shared_data[local_id + stride * 2u - 1u] += temp;
                        }
                        stride /= 2u;
                        workgroupBarrier();
                    }
                    
                    return shared_data[local_id];
                }
                
                @compute @workgroup_size(256)
                fn main(
                    @builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>
                ) {
                    let grid_index = global_id.x;
                    let local_index = local_id.x;
                    
                    let grid_resolution = u32(physics_params.grid_resolution);
                    let total_grid_cells = grid_resolution * grid_resolution * grid_resolution;
                    
                    let input_value = select(0u, grid_counts[grid_index], grid_index < total_grid_cells);
                    let local_prefix_sum = workgroup_prefix_sum(local_index, input_value);
                    let workgroup_base_offset = workgroup_id.x * 256u;
                    
                    if (grid_index < total_grid_cells) {
                        grid_offsets[grid_index] = workgroup_base_offset + local_prefix_sum;
                    }
                }
                "#.to_string()
            }
            "physics/cell_physics_spatial.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/cell_physics_spatial.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read_write> acceleration: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: apply gravity only
                    let mass = position_and_mass[cell_index].w;
                    acceleration[cell_index] = vec4<f32>(0.0, -physics_params.gravity, 0.0, 0.0);
                }
                "#.to_string()
            }
            "physics/force_calculation.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/force_calculation.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read_write> acceleration: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: add gravity to existing acceleration
                    let mass = position_and_mass[cell_index].w;
                    let gravity_acceleration = vec3<f32>(0.0, -physics_params.gravity, 0.0);
                    acceleration[cell_index] += vec4<f32>(gravity_acceleration, 0.0);
                }
                "#.to_string()
            }
            "physics/adhesion_physics.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/adhesion_physics.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> acceleration: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: no adhesion forces applied
                }
                "#.to_string()
            }
            "physics/adhesion_formation.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/adhesion_formation.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: no bond formation
                }
                "#.to_string()
            }
            "physics/cell_position_update.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/cell_position_update.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read> velocity: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read> acceleration: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: basic position update
                    let pos_and_mass = position_and_mass[cell_index];
                    let velocity_vec = velocity[cell_index].xyz;
                    let new_position = pos_and_mass.xyz + velocity_vec * physics_params.delta_time;
                    position_and_mass[cell_index] = vec4<f32>(new_position, pos_and_mass.w);
                }
                "#.to_string()
            }
            "physics/cell_velocity_update.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/cell_velocity_update.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read> acceleration: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: basic velocity update
                    let current_velocity = velocity[cell_index].xyz;
                    let acceleration_vec = acceleration[cell_index].xyz;
                    let new_velocity = current_velocity + acceleration_vec * physics_params.delta_time;
                    velocity[cell_index] = vec4<f32>(new_velocity, 0.0);
                }
                "#.to_string()
            }
            "physics/momentum_correction.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/momentum_correction.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: no momentum correction
                }
                "#.to_string()
            }
            "physics/rigid_body_constraints.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/rigid_body_constraints.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> velocity: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: no constraints applied
                }
                "#.to_string()
            }
            "physics/collision_detection.wgsl" => {
                r#"
                @group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read_write> forces: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&positions)) {
                        return;
                    }
                    // Placeholder: no forces applied
                    forces[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                "#.to_string()
            }
            "physics/integration.wgsl" => {
                r#"
                @group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&positions)) {
                        return;
                    }
                    // Placeholder: no integration
                }
                "#.to_string()
            }
            "lifecycle/cell_division.wgsl" => {
                r#"
                @group(0) @binding(0) var<storage, read> masses: array<f32>;
                @group(0) @binding(1) var<storage, read_write> division_flags: array<u32>;
                
                @compute @workgroup_size(32)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&masses)) {
                        return;
                    }
                    // Placeholder: no divisions
                    division_flags[index] = 0u;
                }
                "#.to_string()
            }
            "extract_instances.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/extract_instances.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
                struct ExtractionParams {
                    cell_count: u32,
                    mode_count: u32,
                    cell_type_count: u32,
                    current_time: f32,
                    _padding: vec3<f32>,
                }
                
                struct CellInstance {
                    position: vec3<f32>,
                    radius: f32,
                    color: vec4<f32>,
                    visual_params: vec4<f32>,
                    membrane_params: vec4<f32>,
                    rotation: vec4<f32>,
                }
                
                @group(0) @binding(0) var<uniform> params: ExtractionParams;
                @group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
                @group(0) @binding(9) var<storage, read_write> instances: array<CellInstance>;
                
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    if (idx >= params.cell_count) {
                        return;
                    }
                    // Placeholder: basic instance extraction
                    var instance: CellInstance;
                    instance.position = positions[idx].xyz;
                    instance.radius = 1.0;
                    instance.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                    instance.visual_params = vec4<f32>(0.5, 32.0, 0.3, 0.0);
                    instance.membrane_params = vec4<f32>(8.0, 0.15, 0.0, 0.0);
                    instance.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                    instances[idx] = instance;
                }
                "#.to_string()
            }
            "physics/cell_update_internal.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/cell_update_internal.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read_write> ages: array<f32>;
                @group(0) @binding(4) var<storage, read_write> nitrates: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: update age only
                    ages[cell_index] += physics_params.delta_time;
                }
                "#.to_string()
            }
            "physics/nutrient_system.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/physics/nutrient_system.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read_write> nitrates: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: basic nutrient gain
                    nitrates[cell_index] += 0.1 * physics_params.delta_time;
                }
                "#.to_string()
            }
            "lifecycle/lifecycle_death_scan.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/lifecycle/lifecycle_death_scan.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(3) var<storage, read> nitrates: array<f32>;
                @group(0) @binding(6) var<storage, read_write> death_flags: array<u32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: mark cells with low nutrients for death
                    if (nitrates[cell_index] <= -1.0) {
                        death_flags[cell_index] = 1u;
                    } else {
                        death_flags[cell_index] = 0u;
                    }
                }
                "#.to_string()
            }
            "lifecycle/lifecycle_death_compact.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/lifecycle/lifecycle_death_compact.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(2) var<storage, read> death_flags: array<u32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: compaction logic would go here
                }
                "#.to_string()
            }
            "lifecycle/lifecycle_division_scan.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/lifecycle/lifecycle_division_scan.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(7) var<storage, read_write> division_candidates: array<u32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let cell_index = global_id.x;
                    if (cell_index >= physics_params.cell_count) {
                        return;
                    }
                    // Placeholder: mark cells with sufficient mass for division
                    let mass = position_and_mass[cell_index].w;
                    if (mass >= 2.0) {
                        division_candidates[cell_index] = 1u;
                    } else {
                        division_candidates[cell_index] = 0u;
                    }
                }
                "#.to_string()
            }

            "lifecycle/lifecycle_slot_assign.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/lifecycle/lifecycle_slot_assign.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> division_candidates: array<u32>;
                @group(0) @binding(2) var<storage, read> free_slots: array<u32>;
                @group(0) @binding(3) var<storage, read_write> division_assignments: array<u32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let thread_index = global_id.x;
                    // Placeholder: simple slot assignment
                    if (thread_index < arrayLength(&division_candidates)) {
                        division_assignments[thread_index] = free_slots[thread_index];
                    }
                }
                "#.to_string()
            }
            "lifecycle/lifecycle_division_execute.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/lifecycle/lifecycle_division_execute.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read_write> position_and_mass: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read> division_candidates: array<u32>;
                @group(0) @binding(3) var<storage, read> division_assignments: array<u32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let thread_index = global_id.x;
                    // Placeholder: simple division execution
                    if (thread_index < arrayLength(&division_candidates)) {
                        let parent_index = division_candidates[thread_index];
                        let child_index = division_assignments[thread_index];
                        
                        // Split mass equally
                        let parent_mass = position_and_mass[parent_index].w;
                        position_and_mass[parent_index].w = parent_mass * 0.5;
                        position_and_mass[child_index] = position_and_mass[parent_index];
                    }
                }
                "#.to_string()
            }
            "lifecycle/lifecycle_free_slots.wgsl" => {
                // Load the actual shader file we just created
                if let Ok(shader_source) = std::fs::read_to_string("shaders/lifecycle/lifecycle_free_slots.wgsl") {
                    return shader_source;
                }
                
                // Fallback placeholder if file doesn't exist
                r#"
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
                    dragged_cell_index: i32,
                    _padding1: vec3<f32>,
                    _padding: array<f32, 48>,
                }
                
                @group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
                @group(0) @binding(1) var<storage, read> death_flags: array<u32>;
                @group(0) @binding(2) var<storage, read_write> free_slots: array<u32>;
                @group(0) @binding(3) var<storage, read_write> free_slot_count: u32;
                
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let thread_index = global_id.x;
                    // Placeholder: collect free slots from death flags
                    if (thread_index < physics_params.cell_count && death_flags[thread_index] != 0u) {
                        // Mark slot as free (simplified logic)
                        free_slots[thread_index] = thread_index;
                    }
                }
                "#.to_string()
            }
            _ => {
                // Default placeholder shader
                r#"
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    // Placeholder compute shader
                }
                "#.to_string()
            }
        }
    }
    
    /// Create a bind group layout for physics operations.
    ///
    /// This layout includes all buffers needed for physics computation:
    /// positions, velocities, forces, masses, etc.
    ///
    /// # Returns
    /// Bind group layout for physics pipelines
    pub fn create_physics_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "physics";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Physics Bind Group Layout"),
                entries: &[
                    // Uniform buffer: Physics parameters
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
                    // Storage buffer: Position and mass
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
                    // Storage buffer: Velocity
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
                    // Storage buffer: Forces
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
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for spatial grid clear operations.
    ///
    /// This layout includes buffers needed for clearing spatial grid:
    /// physics parameters, grid counts, and grid offsets.
    ///
    /// # Returns
    /// Bind group layout for spatial grid clear pipeline
    pub fn create_spatial_clear_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_clear";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Clear Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters (for grid settings)
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
                    // Binding 1: Storage buffer - Grid cell counts (read_write)
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
                    // Binding 2: Storage buffer - Grid cell offsets (read_write)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for spatial grid assignment operations.
    ///
    /// This layout includes buffers needed for assigning cells to grid cells:
    /// physics parameters, cell positions, grid counts, and grid assignments.
    ///
    /// # Returns
    /// Bind group layout for spatial grid assignment pipeline
    pub fn create_spatial_assign_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_assign";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Assignment Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters (for grid settings)
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
                    // Binding 1: Storage buffer - Cell positions (read-only)
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
                    // Binding 2: Storage buffer - Grid cell counts (read_write, atomic)
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
                    // Binding 3: Storage buffer - Grid assignments (read_write)
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
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for spatial grid insertion operations.
    ///
    /// This layout includes buffers needed for inserting cell indices into grid:
    /// physics parameters, grid assignments, grid offsets, grid indices, and insertion counters.
    ///
    /// # Returns
    /// Bind group layout for spatial grid insertion pipeline
    pub fn create_spatial_insert_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_insert";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Insertion Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Grid assignments (read-only)
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
                    // Binding 2: Storage buffer - Grid offsets (read-only)
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
                    // Binding 3: Storage buffer - Grid indices (read_write)
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
                    // Binding 4: Storage buffer - Insertion counters (read_write, atomic)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for spatial grid prefix sum operations.
    ///
    /// This layout includes buffers needed for computing prefix sum offsets:
    /// physics parameters, grid counts, and grid offsets.
    ///
    /// # Returns
    /// Bind group layout for spatial grid prefix sum pipeline
    pub fn create_spatial_prefix_sum_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_prefix_sum";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Prefix Sum Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Grid counts (read-only)
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
                    // Binding 2: Storage buffer - Grid offsets (read_write)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for spatial grid operations.
    ///
    /// This is a legacy method that creates the assignment layout for compatibility.
    ///
    /// # Returns
    /// Bind group layout for spatial grid pipelines
    pub fn create_spatial_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        self.create_spatial_assign_bind_group_layout()
    }
    
    /// Get workgroup size for a specific operation type.
    ///
    /// # Arguments
    /// * `operation` - Type of operation ("spatial", "physics", "adhesion", "lifecycle")
    ///
    /// # Returns
    /// Optimal workgroup size for the operation
    pub fn get_workgroup_size(&self, operation: &str) -> u32 {
        match operation {
            "spatial" => self.workgroup_sizes.spatial_grid,
            "physics" => self.workgroup_sizes.physics,
            "adhesion" => self.workgroup_sizes.adhesion,
            "lifecycle" => self.workgroup_sizes.lifecycle,
            _ => 64, // Default workgroup size
        }
    }
    
    /// Create the spatial grid clear compute pipeline.
    ///
    /// This pipeline resets all spatial grid cell counts to zero at the beginning
    /// of each physics step. Uses a workgroup size of 256 for optimal memory
    /// bandwidth utilization when clearing the 64³ grid.
    ///
    /// # Returns
    /// Reference to the spatial grid clear compute pipeline
    pub fn get_spatial_grid_clear_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_spatial_clear_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "spatial_grid_clear",
            "spatial/grid_clear.wgsl",
            &layout,
        )
    }
    
    /// Create the spatial grid assignment compute pipeline.
    ///
    /// This pipeline assigns each cell to its appropriate spatial grid cell based
    /// on the cell's position. Uses a workgroup size of 64 for balanced compute
    /// and memory access when processing cell positions.
    ///
    /// # Returns
    /// Reference to the spatial grid assignment compute pipeline
    pub fn get_spatial_grid_assign_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_spatial_assign_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "spatial_grid_assign",
            "spatial/grid_assign.wgsl",
            &layout,
        )
    }
    
    /// Create the spatial grid insertion compute pipeline.
    ///
    /// This pipeline inserts cell indices into the spatial grid based on their
    /// grid assignments. Uses a workgroup size of 64 for balanced compute and
    /// memory access when processing cell insertions.
    ///
    /// # Returns
    /// Reference to the spatial grid insertion compute pipeline
    pub fn get_spatial_grid_insert_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_spatial_insert_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "spatial_grid_insert",
            "spatial/grid_insert.wgsl",
            &layout,
        )
    }
    
    /// Create the spatial grid prefix sum compute pipeline.
    ///
    /// This pipeline builds prefix sum offsets for the spatial grid to enable
    /// efficient traversal during collision detection. Uses a workgroup size of 256
    /// for optimal shared memory utilization in the prefix sum algorithm.
    ///
    /// # Returns
    /// Reference to the spatial grid prefix sum compute pipeline
    pub fn get_spatial_grid_prefix_sum_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_spatial_prefix_sum_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "spatial_grid_prefix_sum",
            "spatial/grid_prefix_sum.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for collision detection operations.
    ///
    /// This layout includes all buffers needed for collision detection and force calculation:
    /// physics parameters, cell properties, and spatial grid data.
    ///
    /// # Returns
    /// Bind group layout for collision detection pipeline
    pub fn create_collision_detection_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "collision_detection";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Collision Detection Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Velocity (read-only)
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
                    // Binding 3: Storage buffer - Acceleration (read_write)
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
                    // Binding 4: Storage buffer - Orientation (read-only)
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
                    // Binding 5: Storage buffer - Mode indices (read-only)
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
                    // Binding 6: Storage buffer - Genome modes (read-only)
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
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for spatial grid data access.
    ///
    /// This layout provides read-only access to spatial grid data for collision detection.
    ///
    /// # Returns
    /// Bind group layout for spatial grid data access
    pub fn create_spatial_grid_data_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_grid_data";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Data Bind Group Layout"),
                entries: &[
                    // Binding 0: Storage buffer - Grid counts (read-only)
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
                    // Binding 1: Storage buffer - Grid offsets (read-only)
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
                    // Binding 2: Storage buffer - Grid indices (read-only)
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
                    // Binding 3: Storage buffer - Grid assignments (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the collision detection compute pipeline.
    ///
    /// This pipeline implements collision detection and force calculation using
    /// spatial grid acceleration. Uses a workgroup size of 64 for balanced
    /// compute and memory access when processing cell collisions.
    ///
    /// # Returns
    /// Reference to the collision detection compute pipeline
    pub fn get_collision_detection_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_collision_detection_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "collision_detection",
            "physics/cell_physics_spatial.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for force calculation operations.
    ///
    /// This layout includes all buffers needed for additional force calculations:
    /// boundary forces, swim forces, environmental forces, and force accumulation.
    ///
    /// # Returns
    /// Bind group layout for force calculation pipeline
    pub fn create_force_calculation_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "force_calculation";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Force Calculation Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Velocity (read-only)
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
                    // Binding 3: Storage buffer - Acceleration (read_write)
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
                    // Binding 4: Storage buffer - Orientation (read-only)
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
                    // Binding 5: Storage buffer - Mode indices (read-only)
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
                    // Binding 6: Storage buffer - Genome modes (read-only)
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
                    // Binding 7: Storage buffer - Nitrates (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the force calculation compute pipeline.
    ///
    /// This pipeline implements additional force calculations that complement
    /// collision detection including boundary forces, swim forces, and environmental
    /// forces. Uses a workgroup size of 64 for balanced compute and memory access.
    ///
    /// # Returns
    /// Reference to the force calculation compute pipeline
    pub fn get_force_calculation_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_force_calculation_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "force_calculation",
            "physics/force_calculation.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for adhesion physics operations.
    ///
    /// This layout includes all buffers needed for adhesion bond mechanics:
    /// cell properties, adhesion connections, and force/torque accumulation.
    ///
    /// # Returns
    /// Bind group layout for adhesion physics pipeline
    pub fn create_adhesion_physics_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "adhesion_physics";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adhesion Physics Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Velocity (read-only)
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
                    // Binding 3: Storage buffer - Acceleration (read_write)
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
                    // Binding 4: Storage buffer - Orientation (read-only)
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
                    // Binding 5: Storage buffer - Genome orientation (read-only)
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
                    // Binding 6: Storage buffer - Angular velocity (read-only)
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
                    // Binding 7: Storage buffer - Angular acceleration (read_write)
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
                    // Binding 8: Storage buffer - Mode indices (read-only)
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
                    // Binding 9: Storage buffer - Genome modes (read-only)
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
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for adhesion system data access.
    ///
    /// This layout provides access to adhesion connections, indices, and counts
    /// for adhesion physics computation.
    ///
    /// # Returns
    /// Bind group layout for adhesion system data access
    pub fn create_adhesion_system_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "adhesion_system";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adhesion System Bind Group Layout"),
                entries: &[
                    // Binding 0: Storage buffer - Adhesion connections (read_write)
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
                    // Binding 1: Storage buffer - Adhesion indices (read-only)
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
                    // Binding 2: Storage buffer - Adhesion counts (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the adhesion physics compute pipeline.
    ///
    /// This pipeline implements complete adhesion bond mechanics including
    /// spring-damper forces, orientation constraints, twist constraints, and
    /// bond breaking. Uses a workgroup size of 64 for balanced compute and
    /// memory access when processing adhesion connections.
    ///
    /// # Returns
    /// Reference to the adhesion physics compute pipeline
    pub fn get_adhesion_physics_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_adhesion_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "adhesion_physics",
            "physics/adhesion_physics.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for adhesion formation operations.
    ///
    /// This layout includes all buffers needed for adhesion bond formation:
    /// cell properties, adhesion system data, and spatial grid for proximity detection.
    ///
    /// # Returns
    /// Bind group layout for adhesion formation pipeline
    pub fn create_adhesion_formation_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "adhesion_formation";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adhesion Formation Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Orientation (read-only)
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
                    // Binding 3: Storage buffer - Genome orientation (read-only)
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
                    // Binding 4: Storage buffer - Mode indices (read-only)
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
                    // Binding 5: Storage buffer - Genome modes (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the adhesion formation compute pipeline.
    ///
    /// This pipeline implements adhesion bond formation logic including
    /// proximity detection, genome-based formation criteria, zone classification,
    /// and maximum adhesion limits per cell. Uses a workgroup size of 64 for
    /// balanced compute and memory access when processing bond formation.
    ///
    /// # Returns
    /// Reference to the adhesion formation compute pipeline
    pub fn get_adhesion_formation_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_adhesion_formation_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "adhesion_formation",
            "physics/adhesion_formation.wgsl",
            &layout,
        )
    }
    
    /// Create the force calculation compute pipeline.
    
    /// Create a bind group layout for position update operations.
    ///
    /// This layout includes all buffers needed for position integration:
    /// physics parameters, positions, velocities, and accelerations.
    ///
    /// # Returns
    /// Bind group layout for position update pipeline
    pub fn create_position_update_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "position_update";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Position Update Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read_write)
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
                    // Binding 2: Storage buffer - Velocity (read-only)
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
                    // Binding 3: Storage buffer - Acceleration (read-only)
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
                    // Binding 4: Storage buffer - Previous acceleration (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the position update compute pipeline.
    ///
    /// This pipeline implements Verlet integration for updating cell positions
    /// based on velocities and accelerations. Uses a workgroup size of 64 for
    /// balanced compute and memory access when processing position updates.
    ///
    /// # Returns
    /// Reference to the position update compute pipeline
    pub fn get_position_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_position_update_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "position_update",
            "physics/cell_position_update.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for velocity update operations.
    ///
    /// This layout includes all buffers needed for velocity integration:
    /// physics parameters, velocities, accelerations, and angular properties.
    ///
    /// # Returns
    /// Bind group layout for velocity update pipeline
    pub fn create_velocity_update_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "velocity_update";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Velocity Update Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Velocity (read_write)
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
                    // Binding 3: Storage buffer - Acceleration (read-only)
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
                    // Binding 4: Storage buffer - Previous acceleration (read_write)
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
                    // Binding 5: Storage buffer - Angular velocity (read_write)
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
                    // Binding 6: Storage buffer - Angular acceleration (read-only)
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
                    // Binding 7: Storage buffer - Previous angular acceleration (read_write)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the velocity update compute pipeline.
    ///
    /// This pipeline implements velocity integration from accelerations with
    /// damping and velocity limits. Uses a workgroup size of 64 for balanced
    /// compute and memory access when processing velocity updates.
    ///
    /// # Returns
    /// Reference to the velocity update compute pipeline
    pub fn get_velocity_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_velocity_update_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "velocity_update",
            "physics/cell_velocity_update.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for momentum correction operations.
    ///
    /// This layout includes all buffers needed for momentum conservation:
    /// physics parameters, cell properties, and momentum data.
    ///
    /// # Returns
    /// Bind group layout for momentum correction pipeline
    pub fn create_momentum_correction_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "momentum_correction";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Momentum Correction Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Velocity (read_write)
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
                    // Binding 3: Storage buffer - Angular velocity (read_write)
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
                    // Binding 4: Storage buffer - Momentum data (read_write)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the momentum correction compute pipeline.
    ///
    /// This pipeline implements momentum conservation corrections to prevent
    /// drift due to numerical errors. Uses a workgroup size of 64 for balanced
    /// compute and memory access when processing momentum corrections.
    ///
    /// # Returns
    /// Reference to the momentum correction compute pipeline
    pub fn get_momentum_correction_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_momentum_correction_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "momentum_correction",
            "physics/momentum_correction.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for rigid body constraints operations.
    ///
    /// This layout includes all buffers needed for constraint solving:
    /// physics parameters, cell properties, and adhesion system data.
    ///
    /// # Returns
    /// Bind group layout for rigid body constraints pipeline
    pub fn create_rigid_body_constraints_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "rigid_body_constraints";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Rigid Body Constraints Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read_write)
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
                    // Binding 2: Storage buffer - Velocity (read_write)
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
                    // Binding 3: Storage buffer - Orientation (read_write)
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
                    // Binding 4: Storage buffer - Angular velocity (read_write)
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
                    // Binding 5: Storage buffer - Mode indices (read-only)
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
                    // Binding 6: Storage buffer - Genome modes (read-only)
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
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the rigid body constraints compute pipeline.
    ///
    /// This pipeline implements constraint solving for adhesion networks
    /// and other rigid body constraints. Uses a workgroup size of 64 for
    /// balanced compute and memory access when processing constraints.
    ///
    /// # Returns
    /// Reference to the rigid body constraints compute pipeline
    pub fn get_rigid_body_constraints_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_rigid_body_constraints_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "rigid_body_constraints",
            "physics/rigid_body_constraints.wgsl",
            &layout,
        )
    }
    
    /// Calculate the number of workgroups needed for a given number of elements.
    ///
    /// This ensures that all elements are processed by rounding up the division
    /// of element count by workgroup size.
    ///
    /// # Arguments
    /// * `element_count` - Number of elements to process
    /// * `workgroup_size` - Size of each workgroup
    ///
    /// # Returns
    /// Number of workgroups needed
    pub fn calculate_workgroups(element_count: u32, workgroup_size: u32) -> u32 {
        (element_count + workgroup_size - 1) / workgroup_size
    }
    
    /// Dispatch a compute pipeline with the appropriate number of workgroups.
    ///
    /// This is a convenience method that calculates the correct number of
    /// workgroups and dispatches the compute pipeline.
    ///
    /// # Arguments
    /// * `compute_pass` - Active compute pass
    /// * `pipeline` - Compute pipeline to dispatch
    /// * `element_count` - Number of elements to process
    /// * `workgroup_size` - Size of each workgroup
    pub fn dispatch_compute(
        compute_pass: &mut wgpu::ComputePass,
        pipeline: &wgpu::ComputePipeline,
        element_count: u32,
        workgroup_size: u32,
    ) {
        compute_pass.set_pipeline(pipeline);
        let workgroups = Self::calculate_workgroups(element_count, workgroup_size);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
    
    /// Create a bind group layout for cell internal update operations.
    ///
    /// This layout includes all buffers needed for cell internal state updates:
    /// physics parameters, cell properties, internal state, and genome data.
    ///
    /// # Returns
    /// Bind group layout for cell internal update pipeline
    pub fn create_cell_internal_update_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "cell_internal_update";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell Internal Update Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Velocity (read-only)
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
                    // Binding 3: Storage buffer - Ages (read_write)
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
                    // Binding 4: Storage buffer - Nitrates (read_write)
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
                    // Binding 5: Storage buffer - Toxins (read_write)
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
                    // Binding 6: Storage buffer - Signalling substances (read_write)
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
                    // Binding 7: Storage buffer - Mode indices (read-only)
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
                    // Binding 8: Storage buffer - Genome modes (read-only)
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
                    // Binding 9: Storage buffer - Birth times (read-only)
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
                    // Binding 10: Storage buffer - Split intervals (read-only)
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
                    // Binding 11: Storage buffer - Split masses (read-only)
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
                    // Binding 12: Storage buffer - Split ready frame (read_write)
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
                    // Binding 13: Storage buffer - Genome IDs (read-only)
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
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the cell internal update compute pipeline.
    ///
    /// This pipeline handles all internal cell state updates including aging,
    /// nutrient processing, signaling substances, and toxin accumulation.
    /// Uses a workgroup size of 64 for balanced compute and memory access.
    ///
    /// # Returns
    /// Reference to the cell internal update compute pipeline
    pub fn get_cell_internal_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_cell_internal_update_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "cell_internal_update",
            "physics/cell_update_internal.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for nutrient system operations.
    ///
    /// This layout includes all buffers needed for nutrient system computation:
    /// physics parameters, cell properties, nutrient data, and adhesion system.
    ///
    /// # Returns
    /// Bind group layout for nutrient system pipeline
    pub fn create_nutrient_system_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "nutrient_system";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nutrient System Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read_write)
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
                    // Binding 2: Storage buffer - Velocity (read-only)
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
                    // Binding 3: Storage buffer - Nitrates (read_write)
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
                    // Binding 4: Storage buffer - Mode indices (read-only)
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
                    // Binding 5: Storage buffer - Genome modes (read-only)
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
                    // Binding 6: Storage buffer - Genome IDs (read-only)
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
                    // Binding 7: Storage buffer - Split ready frame (read_write)
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
                    // Binding 8: Storage buffer - Death flags (read_write)
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
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for adhesion system data access (for nutrient system).
    ///
    /// This layout provides read-only access to adhesion system data for nutrient flow calculations.
    ///
    /// # Returns
    /// Bind group layout for adhesion system data access
    pub fn create_nutrient_adhesion_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "nutrient_adhesion";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nutrient Adhesion Bind Group Layout"),
                entries: &[
                    // Binding 0: Storage buffer - Adhesion connections (read-only)
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
                    // Binding 1: Storage buffer - Adhesion indices (read-only)
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
                    // Binding 2: Storage buffer - Adhesion counts (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the nutrient system compute pipeline.
    ///
    /// This pipeline implements the complete nutrient system including nutrient gain,
    /// consumption, flow calculations with pressure-based equilibrium, priority-based
    /// distribution, and cell death from nutrient depletion. Uses a workgroup size of 64.
    ///
    /// # Returns
    /// Reference to the nutrient system compute pipeline
    pub fn get_nutrient_system_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_nutrient_system_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "nutrient_system",
            "physics/nutrient_system.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for lifecycle death scan operations.
    ///
    /// This layout includes all buffers needed for death detection:
    /// physics parameters, cell properties, and death flags.
    ///
    /// # Returns
    /// Bind group layout for lifecycle death scan pipeline
    pub fn create_lifecycle_death_scan_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_death_scan";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Death Scan Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Ages (read-only)
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
                    // Binding 3: Storage buffer - Nitrates (read-only)
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
                    // Binding 4: Storage buffer - Toxins (read-only)
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
                    // Binding 5: Storage buffer - Birth times (read-only)
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
                    // Binding 6: Storage buffer - Death flags (read_write)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for lifecycle death compaction operations.
    ///
    /// This layout includes all buffers needed for cell compaction:
    /// physics parameters, cell count, death data, and all cell property arrays.
    ///
    /// # Returns
    /// Bind group layout for lifecycle death compaction pipeline
    pub fn create_lifecycle_death_compact_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_death_compact";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Death Compaction Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Cell count buffer (read_write)
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
                    // Binding 2: Storage buffer - Death flags (read-only)
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
                    // Binding 3: Storage buffer - Death prefix sum (read-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for cell property arrays (for compaction).
    ///
    /// This layout provides read_write access to all cell property arrays
    /// for compaction operations.
    ///
    /// # Returns
    /// Bind group layout for cell property arrays
    pub fn create_cell_properties_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "cell_properties";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell Properties Bind Group Layout"),
                entries: &[
                    // All cell property buffers (read_write for compaction)
                    // Binding 0: Position and mass
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
                    // Binding 1: Velocity
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
                    // Continue with all other cell properties...
                    // (This would be a very long list - simplified for brevity)
                    // In practice, we might need multiple bind groups or a different approach
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the lifecycle death scan compute pipeline.
    ///
    /// This pipeline scans all cells to identify those ready to die and sets
    /// death flags for prefix-sum compaction. Uses a workgroup size of 64.
    ///
    /// # Returns
    /// Reference to the lifecycle death scan compute pipeline
    pub fn get_lifecycle_death_scan_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_death_scan_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_death_scan",
            "lifecycle/lifecycle_death_scan.wgsl",
            &layout,
        )
    }
    
    /// Create the lifecycle death compaction compute pipeline.
    ///
    /// This pipeline compacts cell arrays to remove dead cells using prefix-sum
    /// results and cleans up adhesion connections. Uses a workgroup size of 64.
    ///
    /// # Returns
    /// Reference to the lifecycle death compaction compute pipeline
    pub fn get_lifecycle_death_compact_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_death_compact_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_death_compact",
            "lifecycle/lifecycle_death_compact.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for lifecycle division scan operations.
    ///
    /// This layout includes all buffers needed for division detection and
    /// slot reservation calculation.
    ///
    /// # Returns
    /// Bind group layout for lifecycle division scan pipeline
    pub fn create_lifecycle_division_scan_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_division_scan";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Scan Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Position and mass (read-only)
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
                    // Binding 2: Storage buffer - Split ready frame (read-only)
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
                    // Binding 3: Storage buffer - Split masses (read-only)
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
                    // Binding 4: Storage buffer - Birth times (read-only)
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
                    // Binding 5: Storage buffer - Split intervals (read-only)
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
                    // Binding 6: Storage buffer - Adhesion counts (read-only)
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
                    // Binding 7: Storage buffer - Division candidates (read_write)
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
                    // Binding 8: Storage buffer - Division reservations (read_write)
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
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for lifecycle slot assignment operations.
    ///
    /// This layout includes all buffers needed for deterministic slot allocation
    /// using prefix-sum results and free slot management.
    ///
    /// # Returns
    /// Bind group layout for lifecycle slot assignment pipeline
    pub fn create_lifecycle_slot_assign_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_slot_assign";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Slot Assignment Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Physics parameters
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
                    // Binding 1: Storage buffer - Cell count buffer (read_write)
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
                    // Binding 2: Storage buffer - Division candidates (read-only)
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
                    // Binding 3: Storage buffer - Division reservations (read-only)
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
                    // Binding 4: Storage buffer - Reservation prefix sum (read-only)
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
                    // Binding 5: Storage buffer - Free slots (read-only)
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
                    // Binding 6: Storage buffer - Free slot count (read-only)
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
                    // Binding 7: Storage buffer - Division assignments (read_write)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the lifecycle division scan compute pipeline.
    ///
    /// This pipeline scans all cells to identify those ready to divide and
    /// calculates their slot reservation needs. Uses a workgroup size of 64.
    ///
    /// # Returns
    /// Reference to the lifecycle division scan compute pipeline
    pub fn get_lifecycle_division_scan_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_division_scan_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_division_scan",
            "lifecycle/lifecycle_division_scan.wgsl",
            &layout,
        )
    }
    
    /// Create the lifecycle slot assignment compute pipeline.
    ///
    /// This pipeline implements deterministic slot allocation using prefix-sum
    /// for cell division with the assignment formula. Uses a workgroup size of 64.
    ///
    /// # Returns
    /// Reference to the lifecycle slot assignment compute pipeline
    pub fn get_lifecycle_slot_assign_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_slot_assign_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_slot_assign",
            "lifecycle/lifecycle_slot_assign.wgsl",
            &layout,
        )
    }
    
    /// Create bind group layout for lifecycle division execution pipeline.
    ///
    /// This layout supports the division execution compute shader that creates
    /// Child A and Child B from parent cells with proper mass distribution,
    /// positioning, and adhesion inheritance using zone classification.
    ///
    /// # Bind Groups
    /// - Group 0: Physics parameters and cell count buffer
    /// - Group 1: Cell data buffers (position, velocity, age, nutrients, etc.)
    /// - Group 2: Division management buffers (candidates, assignments, free slots)
    /// - Group 3: Adhesion system buffers (connections, indices, counts)
    /// - Group 4: Genome mode data for division parameters
    ///
    /// # Returns
    /// Bind group layout for lifecycle division execution pipeline
    pub fn create_lifecycle_division_execute_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_division_execute";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Execute Bind Group Layout"),
                entries: &[
                    // Group 0: Physics parameters (uniform)
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
                    // Group 0: Cell count buffer (read-write)
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
                    // Group 1: Cell data buffers (read-write)
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
                    // Group 2: Division management buffers
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Group 3: Adhesion system buffers
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
                    // Group 4: Genome mode data
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
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the lifecycle division execution compute pipeline.
    ///
    /// This pipeline executes cell division by creating Child A and Child B
    /// from parent cells with proper mass distribution, positioning, and
    /// adhesion inheritance using zone classification (A, B, C).
    ///
    /// # Returns
    /// Reference to the lifecycle division execution compute pipeline
    pub fn get_lifecycle_division_execute_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_division_execute_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_division_execute",
            "lifecycle/lifecycle_division_execute.wgsl",
            &layout,
        )
    }
    
    /// Create bind group layout for lifecycle free slot management pipeline.
    ///
    /// This layout supports the free slot management compute shader that maintains
    /// compacted free slot arrays for deterministic cell lifecycle management.
    /// Uses prefix-sum compaction for optimal GPU performance.
    ///
    /// # Bind Groups
    /// - Group 0: Physics parameters and cell count buffer
    /// - Group 1: Lifecycle management buffers (death flags, division data)
    /// - Group 2: Free slot management buffers (slots, counts, temp arrays)
    /// - Group 3: Prefix-sum working buffers for compaction operations
    ///
    /// # Returns
    /// Bind group layout for lifecycle free slot management pipeline
    pub fn create_lifecycle_free_slots_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_free_slots";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Free Slots Bind Group Layout"),
                entries: &[
                    // Group 0: Physics parameters (uniform)
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
                    // Group 0: Cell count buffer (read-write)
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
                    // Group 1: Death flags (read-only)
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
                    // Group 1: Death compacted array (read-only)
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
                    // Group 1: Death count (read-only)
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
                    // Group 1: Division assignments (read-only)
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
                    // Group 1: Division count (read-only)
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
                    // Group 2: Free slots array (read-write)
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
                    // Group 2: Free slot count (read-write)
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
                    // Group 2: Temporary slots array (read-write)
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
                    // Group 2: Slot flags array (read-write)
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
                    // Group 3: Prefix-sum input buffer (read-write)
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
                    // Group 3: Prefix-sum output buffer (read-write)
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
                    // Group 3: Prefix-sum temporary buffer (read-write)
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
                ],
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the lifecycle free slot management compute pipeline.
    ///
    /// This pipeline maintains compacted free slot arrays for deterministic
    /// cell lifecycle management using prefix-sum compaction. It handles
    /// slot recycling from death and division operations.
    ///
    /// # Returns
    /// Reference to the lifecycle free slot management compute pipeline
    pub fn get_lifecycle_free_slots_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_free_slots_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_free_slots",
            "lifecycle/lifecycle_free_slots.wgsl",
            &layout,
        )
    }
    
    /// Create a bind group layout for instance extraction operations.
    ///
    /// This layout includes all buffers needed for extracting rendering data
    /// from GPU physics simulation state: positions, orientations, radii,
    /// mode indices, cell IDs, genome IDs, and visual lookup tables.
    ///
    /// # Returns
    /// Bind group layout for instance extraction pipeline
    pub fn create_instance_extraction_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "instance_extraction";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Instance Extraction Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform buffer - Extraction parameters
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
                    // Binding 1: Storage buffer - Positions (read-only)
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
                    // Binding 2: Storage buffer - Orientations (read-only)
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
                    // Binding 3: Storage buffer - Radii (read-only)
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
                    // Binding 4: Storage buffer - Mode indices (read-only)
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
                    // Binding 5: Storage buffer - Cell IDs (read-only)
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
                    // Binding 6: Storage buffer - Genome IDs (read-only)
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
                    // Binding 7: Storage buffer - Mode visuals lookup table (read-only)
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
                    // Binding 8: Storage buffer - Cell type visuals lookup table (read-only)
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
                    // Binding 9: Storage buffer - Instance output (write-only)
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
            });
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create the instance extraction compute pipeline.
    ///
    /// This pipeline extracts rendering data from GPU physics simulation state
    /// and converts it to instance data for the CellRenderer. It handles
    /// cell type-specific visual properties and mode-based appearance.
    ///
    /// Uses a workgroup size of 256 for optimal memory bandwidth when
    /// processing large numbers of cells for rendering.
    ///
    /// # Returns
    /// Reference to the instance extraction compute pipeline
    pub fn get_instance_extraction_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_instance_extraction_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "instance_extraction",
            "extract_instances.wgsl",
            &layout,
        )
    }
}

/// Utility functions for compute pipeline management.
pub mod utils {
    use super::*;
    
    /// Create a basic compute bind group with common buffer bindings.
    ///
    /// This is a utility function for creating bind groups with a standard
    /// set of buffer bindings for compute shaders.
    ///
    /// # Arguments
    /// * `device` - wgpu device
    /// * `layout` - Bind group layout
    /// * `buffers` - Array of buffers to bind
    /// * `label` - Debug label for the bind group
    ///
    /// # Returns
    /// Created bind group
    pub fn create_compute_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
        label: &str,
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }
    
    /// Validate that a buffer has the expected size for a given element count.
    ///
    /// This is useful for debugging buffer size mismatches that can cause
    /// GPU errors or incorrect behavior.
    ///
    /// # Arguments
    /// * `buffer` - Buffer to validate
    /// * `element_count` - Expected number of elements
    /// * `element_size` - Size of each element in bytes
    /// * `buffer_name` - Name of buffer for error messages
    ///
    /// # Panics
    /// Panics if buffer size doesn't match expected size
    pub fn validate_buffer_size(
        buffer: &wgpu::Buffer,
        element_count: u32,
        element_size: u32,
        buffer_name: &str,
    ) {
        let expected_size = element_count as u64 * element_size as u64;
        let actual_size = buffer.size();
        
        if actual_size < expected_size {
            panic!(
                "Buffer '{}' is too small: expected {} bytes, got {} bytes",
                buffer_name, expected_size, actual_size
            );
        }
    }
}