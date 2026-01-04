//! # GPU Compute Pipeline Manager
//!
//! This module provides a clean, minimal compute pipeline manager for the Bio-Spheres GPU scene.
//! It manages the creation and execution of compute shaders for pure GPU physics simulation.
//!
//! ## Pipeline Architecture
//!
//! The GPU scene uses a series of compute pipelines that execute in strict order:
//! 1. **Spatial Grid Pipelines** - Build spatial partitioning for collision detection
//! 2. **Physics Pipelines** - Calculate forces and integrate motion
//! 3. **Lifecycle Pipelines** - Handle cell division, death, and lifecycle management
//! 4. **Rendering Pipelines** - Extract visual data for rendering
//!
//! ## Design Principles
//!
//! - **Zero CPU Physics**: All computation runs exclusively on GPU
//! - **Minimal Complexity**: Clean, focused implementation without unnecessary abstractions
//! - **Performance First**: Optimized workgroup sizes and memory access patterns
//! - **Clear Separation**: Each pipeline has a single, well-defined responsibility

use wgpu;
use std::collections::HashMap;

/// Compute workgroup sizes optimized for GPU architecture.
///
/// These sizes are chosen based on GPU warp/wavefront sizes and the specific
/// computational characteristics of each operation type.
#[derive(Clone, Copy, Debug)]
pub struct ComputeWorkgroupSizes {
    /// Spatial grid operations (memory-intensive, benefits from larger workgroups)
    pub spatial_grid: u32,
    /// Physics calculations (balanced compute and memory access)
    pub physics: u32,
    /// Adhesion processing (similar to physics)
    pub adhesion: u32,
    /// Lifecycle events (complex logic with potential divergence)
    pub lifecycle: u32,
}

impl Default for ComputeWorkgroupSizes {
    fn default() -> Self {
        Self {
            spatial_grid: 256,  // Optimal for memory-bound grid operations
            physics: 64,        // Balanced for physics computation
            adhesion: 64,       // Similar to physics
            lifecycle: 32,      // Conservative for complex branching logic
        }
    }
}

/// Clean, minimal compute pipeline manager for GPU scene operations.
///
/// This manager creates and caches compute pipelines on-demand, providing
/// a simple interface for GPU physics computation without unnecessary complexity.
pub struct ComputePipelineManager {
    /// Cached compute pipelines indexed by name
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    
    /// Cached bind group layouts indexed by name
    bind_group_layouts: HashMap<String, wgpu::BindGroupLayout>,
    
    /// Workgroup sizes for different operation types
    workgroup_sizes: ComputeWorkgroupSizes,
    
    /// wgpu device for pipeline creation
    device: wgpu::Device,
}

impl ComputePipelineManager {
    /// Create a new compute pipeline manager.
    ///
    /// # Arguments
    /// * `device` - wgpu device for pipeline creation
    /// * `workgroup_sizes` - Custom workgroup sizes
    pub fn new(device: wgpu::Device, workgroup_sizes: ComputeWorkgroupSizes) -> Self {
        Self {
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            workgroup_sizes,
            device,
        }
    }
    
    /// Check if all required compute pipelines are ready for execution.
    ///
    /// # Returns
    /// True if all required pipelines are ready, false otherwise
    pub fn are_pipelines_ready(&self) -> bool {
        // For now, we'll consider pipelines ready if the manager is initialized
        // In a full implementation, this would check for specific required pipelines
        true
    }
    
    /// Get or create a compute pipeline.
    ///
    /// This method creates pipelines on-demand and caches them for future use.
    /// Shader source is loaded from the shaders directory.
    ///
    /// # Arguments
    /// * `name` - Pipeline identifier
    /// * `shader_path` - Path to WGSL shader file relative to shaders directory
    /// * `bind_group_layout` - Bind group layout for resource binding
    ///
    /// # Returns
    /// Reference to the cached compute pipeline
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
    /// # Arguments
    /// * `label` - Debug label for the pipeline
    /// * `shader_path` - Path to WGSL shader file
    /// * `bind_group_layout` - Bind group layout for resource binding
    ///
    /// # Returns
    /// Compiled compute pipeline
    fn create_compute_pipeline(
        &self,
        label: &str,
        shader_path: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        // Load shader source from file
        let full_path = format!("shaders/{}", shader_path);
        let shader_source = std::fs::read_to_string(&full_path)
            .unwrap_or_else(|_| panic!("Failed to load shader: {}", full_path));
        
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
    
    /// Get workgroup size for a specific operation type.
    ///
    /// # Arguments
    /// * `operation` - Operation type ("spatial", "physics", "adhesion", "lifecycle")
    ///
    /// # Returns
    /// Optimal workgroup size for the operation
    pub fn get_workgroup_size(&self, operation: &str) -> u32 {
        match operation {
            "spatial" => self.workgroup_sizes.spatial_grid,
            "physics" => self.workgroup_sizes.physics,
            "adhesion" => self.workgroup_sizes.adhesion,
            "lifecycle" => self.workgroup_sizes.lifecycle,
            _ => 64, // Safe default
        }
    }
    
    /// Create a standard bind group layout for physics operations.
    ///
    /// This layout includes the most common buffers needed for physics computation:
    /// - Physics parameters (uniform)
    /// - Position and mass (storage)
    /// - Velocity (storage)
    /// - Acceleration (storage)
    ///
    /// # Returns
    /// Bind group layout for physics pipelines
    pub fn create_physics_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "physics";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Physics Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Position and mass (storage)
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
                    // Binding 2: Velocity (storage)
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
                    // Binding 3: Acceleration (storage)
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
    
    /// Create a bind group layout for spatial grid operations.
    ///
    /// This layout matches the spatial grid assignment shader requirements:
    /// - Physics parameters (uniform)
    /// - Positions (read-only storage)
    /// - Grid counts (read-write storage with atomics)
    /// - Grid assignments (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for spatial grid pipelines
    pub fn create_spatial_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Positions (read-only storage)
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
                    // Binding 2: Grid counts (read-write storage with atomics)
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
                    // Binding 3: Grid assignments (read-write storage)
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
    
    /// Create a bind group layout for lifecycle operations.
    ///
    /// This layout includes buffers needed for cell lifecycle management:
    /// - Physics parameters (uniform)
    /// - Cell data (storage)
    /// - Death flags (storage)
    /// - Division candidates (storage)
    /// - Free slots (storage)
    ///
    /// # Returns
    /// Bind group layout for lifecycle pipelines
    pub fn create_lifecycle_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Position and mass (storage)
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
                    // Binding 2: Death flags (storage)
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
                    // Binding 3: Division candidates (storage)
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
                    // Binding 4: Free slots (storage)
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
    
    /// Create a bind group layout for collision detection operations.
    ///
    /// This layout matches the cell_physics_spatial.wgsl shader requirements:
    /// Group 0: Physics data buffers
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Velocity (read-only storage)
    /// - Acceleration (read-write storage)
    /// - Orientation (read-only storage)
    /// - Mode indices (read-only storage)
    /// - Genome modes (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for collision detection pipeline
    pub fn create_collision_detection_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "collision_detection";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Collision Detection Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Position and mass (read-only storage)
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
                    // Binding 2: Velocity (read-only storage)
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
                    // Binding 3: Acceleration (read-write storage)
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
                    // Binding 4: Orientation (read-only storage)
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
                    // Binding 5: Mode indices (read-only storage)
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
                    // Binding 6: Genome modes (read-only storage)
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
    
    /// Get the spatial grid clear pipeline.
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
    
    /// Create a bind group layout for spatial grid clear operations.
    ///
    /// This layout matches the spatial grid clear shader requirements:
    /// - Physics parameters (uniform)
    /// - Grid counts (read-write storage)
    /// - Grid offsets (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for spatial grid clear pipeline
    pub fn create_spatial_clear_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_clear";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Clear Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Grid counts (read-write storage)
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
                    // Binding 2: Grid offsets (read-write storage)
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
    
    /// Get the spatial grid assignment pipeline.
    ///
    /// # Returns
    /// Reference to the spatial grid assignment compute pipeline
    pub fn get_spatial_grid_assign_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_spatial_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "spatial_grid_assign",
            "spatial/grid_assign.wgsl",
            &layout,
        )
    }
    
    /// Get the collision detection pipeline.
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
    
    /// Get the force calculation pipeline.
    ///
    /// # Returns
    /// Reference to the force calculation compute pipeline
    pub fn get_force_calculation_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "force_calculation",
            "physics/force_calculation.wgsl",
            &layout,
        )
    }
    
    /// Get the position update pipeline.
    ///
    /// # Returns
    /// Reference to the position update compute pipeline
    pub fn get_position_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "position_update",
            "physics/cell_position_update.wgsl",
            &layout,
        )
    }
    
    /// Get the velocity update pipeline.
    ///
    /// # Returns
    /// Reference to the velocity update compute pipeline
    pub fn get_velocity_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "velocity_update",
            "physics/cell_velocity_update.wgsl",
            &layout,
        )
    }
    
    /// Get the cell lifecycle pipeline.
    ///
    /// # Returns
    /// Reference to the cell lifecycle compute pipeline
    pub fn get_cell_lifecycle_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "cell_lifecycle",
            "lifecycle/lifecycle_death_scan.wgsl",
            &layout,
        )
    }
    
    /// Get the instance extraction pipeline.
    ///
    /// # Returns
    /// Reference to the instance extraction compute pipeline
    pub fn get_instance_extraction_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "instance_extraction",
            "extract_instances.wgsl",
            &layout,
        )
    }
    
    /// Get the spatial grid insertion pipeline.
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
    
    /// Create a bind group layout for spatial grid insertion operations.
    ///
    /// This layout matches the spatial grid insertion shader requirements:
    /// - Physics parameters (uniform)
    /// - Grid assignments (read-only storage)
    /// - Grid offsets (read-only storage)
    /// - Grid indices (read-write storage)
    /// - Grid insertion counters (read-write storage with atomics)
    ///
    /// # Returns
    /// Bind group layout for spatial grid insertion pipeline
    pub fn create_spatial_insert_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_insert";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Insert Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Grid assignments (read-only storage)
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
                    // Binding 2: Grid offsets (read-only storage)
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
                    // Binding 3: Grid indices (read-write storage)
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
                    // Binding 4: Grid insertion counters (read-write storage with atomics)
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
    
    /// Get the spatial grid prefix sum pipeline.
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
    
    /// Create a bind group layout for spatial grid prefix sum operations.
    ///
    /// This layout matches the spatial grid prefix sum shader requirements:
    /// - Physics parameters (uniform)
    /// - Grid counts (read-only storage)
    /// - Grid offsets (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for spatial grid prefix sum pipeline
    pub fn create_spatial_prefix_sum_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_prefix_sum";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Prefix Sum Bind Group Layout"),
                entries: &[
                    // Binding 0: Physics parameters (uniform)
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
                    // Binding 1: Grid counts (read-only storage)
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
                    // Binding 2: Grid offsets (read-write storage)
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
    
    /// Get the adhesion physics pipeline.
    ///
    /// # Returns
    /// Reference to the adhesion physics compute pipeline
    pub fn get_adhesion_physics_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "adhesion_physics",
            "physics/adhesion_physics.wgsl",
            &layout,
        )
    }
    
    /// Get the momentum correction pipeline.
    ///
    /// # Returns
    /// Reference to the momentum correction compute pipeline
    pub fn get_momentum_correction_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "momentum_correction",
            "physics/momentum_correction.wgsl",
            &layout,
        )
    }
    
    /// Get the rigid body constraints pipeline.
    ///
    /// # Returns
    /// Reference to the rigid body constraints compute pipeline
    pub fn get_rigid_body_constraints_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "rigid_body_constraints",
            "physics/rigid_body_constraints.wgsl",
            &layout,
        )
    }
    
    /// Get the cell internal update pipeline.
    ///
    /// # Returns
    /// Reference to the cell internal update compute pipeline
    pub fn get_cell_internal_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "cell_internal_update",
            "physics/cell_update_internal.wgsl",
            &layout,
        )
    }
    
    /// Get the nutrient system pipeline.
    ///
    /// # Returns
    /// Reference to the nutrient system compute pipeline
    pub fn get_nutrient_system_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_physics_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "nutrient_system",
            "physics/nutrient_system.wgsl",
            &layout,
        )
    }
    
    /// Get the lifecycle death scan pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle death scan compute pipeline
    pub fn get_lifecycle_death_scan_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_death_scan",
            "lifecycle/lifecycle_death_scan.wgsl",
            &layout,
        )
    }
    
    /// Get the lifecycle death compact pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle death compact compute pipeline
    pub fn get_lifecycle_death_compact_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_death_compact",
            "lifecycle/lifecycle_death_compact.wgsl",
            &layout,
        )
    }
    
    /// Get the lifecycle division scan pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle division scan compute pipeline
    pub fn get_lifecycle_division_scan_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_division_scan",
            "lifecycle/lifecycle_division_scan.wgsl",
            &layout,
        )
    }
    
    /// Get the lifecycle slot assign pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle slot assign compute pipeline
    pub fn get_lifecycle_slot_assign_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_slot_assign",
            "lifecycle/lifecycle_slot_assign.wgsl",
            &layout,
        )
    }
    
    /// Get the lifecycle division execute pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle division execute compute pipeline
    pub fn get_lifecycle_division_execute_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_division_execute",
            "lifecycle/lifecycle_division_execute.wgsl",
            &layout,
        )
    }
    
    /// Get the lifecycle free slots pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle free slots compute pipeline
    pub fn get_lifecycle_free_slots_pipeline(&mut self) -> &wgpu::ComputePipeline {
        let layout = self.create_lifecycle_bind_group_layout().clone();
        self.get_or_create_pipeline(
            "lifecycle_free_slots",
            "lifecycle/lifecycle_free_slots.wgsl",
            &layout,
        )
    }
}