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
    
    /// Get or create a compute pipeline with multiple bind group layouts.
    ///
    /// This method creates pipelines that use multiple bind groups, such as
    /// the collision detection shader which needs both physics and spatial grid data.
    ///
    /// # Arguments
    /// * `name` - Pipeline identifier
    /// * `shader_path` - Path to WGSL shader file relative to shaders directory
    /// * `bind_group_layouts` - Array of bind group layouts for resource binding
    ///
    /// # Returns
    /// Reference to the cached compute pipeline
    pub fn get_or_create_pipeline_multi_layout(
        &mut self,
        name: &str,
        shader_path: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> &wgpu::ComputePipeline {
        if !self.pipelines.contains_key(name) {
            let pipeline = self.create_compute_pipeline_multi_layout(name, shader_path, bind_group_layouts);
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
    
    /// Create a compute pipeline from WGSL shader source with multiple bind group layouts.
    ///
    /// # Arguments
    /// * `label` - Debug label for the pipeline
    /// * `shader_path` - Path to WGSL shader file
    /// * `bind_group_layouts` - Array of bind group layouts for resource binding
    ///
    /// # Returns
    /// Compiled compute pipeline
    fn create_compute_pipeline_multi_layout(
        &self,
        label: &str,
        shader_path: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
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
        
        // Create pipeline layout with multiple bind group layouts
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts,
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
    
    /// Create a bind group layout for spatial grid data operations.
    ///
    /// This layout matches the spatial grid data requirements for collision detection:
    /// Group 1: Spatial grid buffers
    /// - Spatial grid counts (read-only storage)
    /// - Spatial grid offsets (read-only storage)
    /// - Spatial grid indices (read-only storage)
    /// - Spatial grid assignments (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for spatial grid data
    pub fn create_spatial_grid_data_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "spatial_grid_data";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Spatial Grid Data Bind Group Layout"),
                entries: &[
                    // Binding 0: Spatial grid counts (read-only storage)
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
                    // Binding 1: Spatial grid offsets (read-only storage)
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
                    // Binding 2: Spatial grid indices (read-only storage)
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
                    // Binding 3: Spatial grid assignments (read-only storage)
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
    
    /// Get the spatial grid clear pipeline.
    ///
    /// # Returns
    /// Reference to the spatial grid clear compute pipeline
    pub fn get_spatial_grid_clear_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("spatial_clear") {
            self.create_spatial_clear_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("spatial_grid_clear") {
            let layout = self.bind_group_layouts.get("spatial_clear").unwrap();
            let pipeline = self.create_compute_pipeline("spatial_grid_clear", "spatial/grid_clear.wgsl", layout);
            self.pipelines.insert("spatial_grid_clear".to_string(), pipeline);
        }
        
        self.pipelines.get("spatial_grid_clear").unwrap()
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
    
    /// Create a bind group layout for force calculation operations.
    ///
    /// This layout matches the force_calculation.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Velocity (read-only storage)
    /// - Acceleration (read-write storage)
    /// - Orientation (read-only storage)
    /// - Mode indices (read-only storage)
    /// - Genome modes (read-only storage)
    /// - Nitrates (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for force calculation pipeline
    pub fn create_force_calculation_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "force_calculation";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Force Calculation Bind Group Layout"),
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
                    // Binding 7: Nitrates (read-only storage)
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
    
    /// Get the spatial grid assignment pipeline.
    ///
    /// # Returns
    /// Reference to the spatial grid assignment compute pipeline
    pub fn get_spatial_grid_assign_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("spatial") {
            self.create_spatial_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("spatial_grid_assign") {
            let layout = self.bind_group_layouts.get("spatial").unwrap();
            let pipeline = self.create_compute_pipeline("spatial_grid_assign", "spatial/grid_assign.wgsl", layout);
            self.pipelines.insert("spatial_grid_assign".to_string(), pipeline);
        }
        
        self.pipelines.get("spatial_grid_assign").unwrap()
    }
    
    /// Get the collision detection pipeline.
    ///
    /// # Returns
    /// Reference to the collision detection compute pipeline
    pub fn get_collision_detection_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Collision detection uses TWO bind groups: physics data + spatial grid data
        // Ensure both layouts exist
        if !self.bind_group_layouts.contains_key("collision_detection") {
            self.create_collision_detection_bind_group_layout();
        }
        if !self.bind_group_layouts.contains_key("spatial_grid_data") {
            self.create_spatial_grid_data_bind_group_layout();
        }
        
        // Now we can safely create the pipeline with multiple bind group layouts
        if !self.pipelines.contains_key("collision_detection") {
            let physics_layout = self.bind_group_layouts.get("collision_detection").unwrap();
            let spatial_layout = self.bind_group_layouts.get("spatial_grid_data").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "collision_detection", 
                "physics/cell_physics_spatial.wgsl", 
                &[physics_layout, spatial_layout]
            );
            self.pipelines.insert("collision_detection".to_string(), pipeline);
        }
        
        self.pipelines.get("collision_detection").unwrap()
    }
    
    /// Get the force calculation pipeline.
    ///
    /// # Returns
    /// Reference to the force calculation compute pipeline
    pub fn get_force_calculation_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("force_calculation") {
            self.create_force_calculation_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("force_calculation") {
            let layout = self.bind_group_layouts.get("force_calculation").unwrap();
            let pipeline = self.create_compute_pipeline("force_calculation", "physics/force_calculation.wgsl", layout);
            self.pipelines.insert("force_calculation".to_string(), pipeline);
        }
        
        self.pipelines.get("force_calculation").unwrap()
    }
    
    /// Get the position update pipeline.
    ///
    /// # Returns
    /// Reference to the position update compute pipeline
    pub fn get_position_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("position_update") {
            self.create_position_update_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("position_update") {
            let layout = self.bind_group_layouts.get("position_update").unwrap();
            let pipeline = self.create_compute_pipeline("position_update", "physics/cell_position_update.wgsl", layout);
            self.pipelines.insert("position_update".to_string(), pipeline);
        }
        
        self.pipelines.get("position_update").unwrap()
    }
    
    /// Get the velocity update pipeline.
    ///
    /// # Returns
    /// Reference to the velocity update compute pipeline
    pub fn get_velocity_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("velocity_update") {
            self.create_velocity_update_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("velocity_update") {
            let layout = self.bind_group_layouts.get("velocity_update").unwrap();
            let pipeline = self.create_compute_pipeline("velocity_update", "physics/cell_velocity_update.wgsl", layout);
            self.pipelines.insert("velocity_update".to_string(), pipeline);
        }
        
        self.pipelines.get("velocity_update").unwrap()
    }
    
    /// Get the cell lifecycle pipeline.
    ///
    /// # Returns
    /// Get the instance extraction pipeline.
    ///
    /// # Returns
    /// Reference to the instance extraction compute pipeline
    pub fn get_instance_extraction_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("physics") {
            self.create_physics_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("instance_extraction") {
            let layout = self.bind_group_layouts.get("physics").unwrap();
            let pipeline = self.create_compute_pipeline("instance_extraction", "extract_instances.wgsl", layout);
            self.pipelines.insert("instance_extraction".to_string(), pipeline);
        }
        
        self.pipelines.get("instance_extraction").unwrap()
    }
    
    /// Get the spatial grid insertion pipeline.
    ///
    /// # Returns
    /// Reference to the spatial grid insertion compute pipeline
    pub fn get_spatial_grid_insert_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("spatial_insert") {
            self.create_spatial_insert_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("spatial_grid_insert") {
            let layout = self.bind_group_layouts.get("spatial_insert").unwrap();
            let pipeline = self.create_compute_pipeline("spatial_grid_insert", "spatial/grid_insert.wgsl", layout);
            self.pipelines.insert("spatial_grid_insert".to_string(), pipeline);
        }
        
        self.pipelines.get("spatial_grid_insert").unwrap()
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
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("spatial_prefix_sum") {
            self.create_spatial_prefix_sum_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("spatial_grid_prefix_sum") {
            let layout = self.bind_group_layouts.get("spatial_prefix_sum").unwrap();
            let pipeline = self.create_compute_pipeline("spatial_grid_prefix_sum", "spatial/grid_prefix_sum.wgsl", layout);
            self.pipelines.insert("spatial_grid_prefix_sum".to_string(), pipeline);
        }
        
        self.pipelines.get("spatial_grid_prefix_sum").unwrap()
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
    
    /// Create a bind group layout for adhesion physics operations (Group 0).
    ///
    /// This layout matches the adhesion_physics.wgsl shader requirements for Group 0:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Velocity (read-only storage)
    /// - Acceleration (read-write storage)
    /// - Orientation (read-only storage)
    /// - Genome orientation (read-only storage)
    /// - Angular velocity (read-only storage)
    /// - Angular acceleration (read-write storage)
    /// - Mode indices (read-only storage)
    /// - Genome modes (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for adhesion physics pipeline (Group 0)
    pub fn create_adhesion_physics_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "adhesion_physics";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adhesion Physics Bind Group Layout"),
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
                    // Binding 5: Genome orientation (read-only storage)
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
                    // Binding 6: Angular velocity (read-only storage)
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
                    // Binding 7: Angular acceleration (read-write storage)
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
                    // Binding 8: Mode indices (read-only storage)
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
                    // Binding 9: Genome modes (read-only storage)
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
    
    /// Create a bind group layout for adhesion system operations (Group 1).
    ///
    /// This layout matches the adhesion_physics.wgsl shader requirements for Group 1:
    /// - Adhesion connections (read-write storage)
    /// - Adhesion indices (read-only storage)
    /// - Adhesion counts (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for adhesion system (Group 1)
    pub fn create_adhesion_system_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "adhesion_system";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adhesion System Bind Group Layout"),
                entries: &[
                    // Binding 0: Adhesion connections (read-write storage)
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
                    // Binding 1: Adhesion indices (read-only storage)
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
                    // Binding 2: Adhesion counts (read-only storage)
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
    
    /// Create a bind group layout for nutrient system's read-only adhesion access (Group 1).
    ///
    /// This layout matches the nutrient_system.wgsl shader requirements for Group 1:
    /// - Adhesion connections (read-only storage)
    /// - Adhesion indices (read-only storage)
    /// - Adhesion counts (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for nutrient system adhesion access (Group 1)
    pub fn create_nutrient_adhesion_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "nutrient_adhesion";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nutrient Adhesion Bind Group Layout"),
                entries: &[
                    // Binding 0: Adhesion connections (read-only storage)
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
                    // Binding 1: Adhesion indices (read-only storage)
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
                    // Binding 2: Adhesion counts (read-only storage)
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
    
    /// Create a bind group layout for velocity update operations.
    ///
    /// This layout matches the cell_velocity_update.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Velocity (read-write storage)
    /// - Acceleration (read-only storage)
    /// - Previous acceleration (read-write storage)
    /// - Angular velocity (read-write storage)
    /// - Angular acceleration (read-only storage)
    /// - Previous angular acceleration (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for velocity update pipeline
    pub fn create_velocity_update_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "velocity_update";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Velocity Update Bind Group Layout"),
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
                    // Binding 2: Velocity (read-write storage)
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
                    // Binding 3: Acceleration (read-only storage)
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
                    // Binding 4: Previous acceleration (read-write storage)
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
                    // Binding 5: Angular velocity (read-write storage)
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
                    // Binding 6: Angular acceleration (read-only storage)
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
                    // Binding 7: Previous angular acceleration (read-write storage)
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
    
    /// Create a bind group layout for momentum correction operations.
    ///
    /// This layout matches the momentum_correction.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Velocity (read-write storage)
    /// - Angular velocity (read-write storage)
    /// - Momentum data (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for momentum correction pipeline
    pub fn create_momentum_correction_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "momentum_correction";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Momentum Correction Bind Group Layout"),
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
                    // Binding 2: Velocity (read-write storage)
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
                    // Binding 3: Angular velocity (read-write storage)
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
                    // Binding 4: Momentum data (read-write storage)
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
    
    /// Create a bind group layout for cell internal update operations.
    ///
    /// This layout matches the cell_update_internal.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Velocity (read-only storage)
    /// - Ages (read-write storage)
    /// - Nitrates (read-write storage)
    /// - Toxins (read-write storage)
    /// - Signalling substances (read-write storage)
    /// - Mode indices (read-only storage)
    /// - Genome modes (read-only storage)
    /// - Birth times (read-only storage)
    /// - Split intervals (read-only storage)
    /// - Split masses (read-only storage)
    /// - Split ready frame (read-write storage)
    /// - Genome ids (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for cell internal update pipeline
    pub fn create_cell_internal_update_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "cell_internal_update";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell Internal Update Bind Group Layout"),
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
                    // Binding 3: Ages (read-write storage)
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
                    // Binding 4: Nitrates (read-write storage)
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
                    // Binding 5: Toxins (read-write storage)
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
                    // Binding 6: Signalling substances (read-write storage)
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
                    // Binding 7: Mode indices (read-only storage)
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
                    // Binding 8: Genome modes (read-only storage)
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
                    // Binding 9: Birth times (read-only storage)
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
                    // Binding 10: Split intervals (read-only storage)
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
                    // Binding 11: Split masses (read-only storage)
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
                    // Binding 12: Split ready frame (read-write storage)
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
                    // Binding 13: Genome ids (read-only storage)
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
    
    /// Create a bind group layout for rigid body constraints operations (Group 0).
    ///
    /// This layout matches the rigid_body_constraints.wgsl shader requirements for Group 0:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-write storage)
    /// - Velocity (read-write storage)
    /// - Orientation (read-only storage)
    /// - Angular velocity (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for rigid body constraints pipeline (Group 0)
    pub fn create_rigid_body_constraints_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "rigid_body_constraints";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Rigid Body Constraints Bind Group Layout"),
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
                    // Binding 1: Position and mass (read-write storage)
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
                    // Binding 2: Velocity (read-write storage)
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
                    // Binding 3: Orientation (read-only storage)
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
                    // Binding 4: Angular velocity (read-write storage)
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
    
    /// Create a bind group layout for constraint system operations (Group 1).
    ///
    /// This layout matches the rigid_body_constraints.wgsl shader requirements for Group 1:
    /// - Constraints (read-only storage)
    /// - Constraint count (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for constraint system (Group 1)
    pub fn create_constraint_system_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "constraint_system";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Constraint System Bind Group Layout"),
                entries: &[
                    // Binding 0: Constraints (read-only storage)
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
                    // Binding 1: Constraint count (read-only storage)
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
            
            self.bind_group_layouts.insert(layout_name.to_string(), layout);
        }
        
        self.bind_group_layouts.get(layout_name).unwrap()
    }
    
    /// Create a bind group layout for nutrient system operations (Group 0).
    ///
    /// This layout matches the nutrient_system.wgsl shader requirements for Group 0:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-write storage)
    /// - Velocity (read-only storage)
    /// - Nitrates (read-write storage)
    /// - Mode indices (read-only storage)
    /// - Genome modes (read-only storage)
    /// - Genome ids (read-only storage)
    /// - Split ready frame (read-write storage)
    /// - Death flags (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for nutrient system pipeline (Group 0)
    pub fn create_nutrient_system_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "nutrient_system";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nutrient System Bind Group Layout"),
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
                    // Binding 1: Position and mass (read-write storage)
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
                    // Binding 3: Nitrates (read-write storage)
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
                    // Binding 4: Mode indices (read-only storage)
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
                    // Binding 5: Genome modes (read-only storage)
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
                    // Binding 6: Genome ids (read-only storage)
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
                    // Binding 7: Split ready frame (read-write storage)
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
                    // Binding 8: Death flags (read-write storage)
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
    
    /// Create a bind group layout for position update operations.
    ///
    /// This layout matches the cell_position_update.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-write storage)
    /// - Velocity (read-only storage)
    /// - Acceleration (read-only storage)
    /// - Previous acceleration (read-only storage)
    ///
    /// # Returns
    /// Bind group layout for position update pipeline
    pub fn create_position_update_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "position_update";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Position Update Bind Group Layout"),
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
                    // Binding 1: Position and mass (read-write storage)
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
                    // Binding 3: Acceleration (read-only storage)
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
                    // Binding 4: Previous acceleration (read-only storage)
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
    
    /// Get the adhesion physics pipeline.
    ///
    /// # Returns
    /// Reference to the adhesion physics compute pipeline
    pub fn get_adhesion_physics_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Adhesion physics uses TWO bind groups: physics data + adhesion system
        // Ensure both layouts exist
        if !self.bind_group_layouts.contains_key("adhesion_physics") {
            self.create_adhesion_physics_bind_group_layout();
        }
        if !self.bind_group_layouts.contains_key("adhesion_system") {
            self.create_adhesion_system_bind_group_layout();
        }
        
        // Now we can safely create the pipeline with multiple bind group layouts
        if !self.pipelines.contains_key("adhesion_physics") {
            let physics_layout = self.bind_group_layouts.get("adhesion_physics").unwrap();
            let adhesion_layout = self.bind_group_layouts.get("adhesion_system").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "adhesion_physics", 
                "physics/adhesion_physics.wgsl", 
                &[physics_layout, adhesion_layout]
            );
            self.pipelines.insert("adhesion_physics".to_string(), pipeline);
        }
        
        self.pipelines.get("adhesion_physics").unwrap()
    }
    
    /// Get the momentum correction pipeline.
    ///
    /// # Returns
    /// Reference to the momentum correction compute pipeline
    pub fn get_momentum_correction_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("momentum_correction") {
            self.create_momentum_correction_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("momentum_correction") {
            let layout = self.bind_group_layouts.get("momentum_correction").unwrap();
            let pipeline = self.create_compute_pipeline("momentum_correction", "physics/momentum_correction.wgsl", layout);
            self.pipelines.insert("momentum_correction".to_string(), pipeline);
        }
        
        self.pipelines.get("momentum_correction").unwrap()
    }
    
    /// Get the rigid body constraints pipeline.
    ///
    /// # Returns
    /// Reference to the rigid body constraints compute pipeline
    pub fn get_rigid_body_constraints_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Rigid body constraints uses TWO bind groups: physics data + constraint system
        // Ensure both layouts exist
        if !self.bind_group_layouts.contains_key("rigid_body_constraints") {
            self.create_rigid_body_constraints_bind_group_layout();
        }
        if !self.bind_group_layouts.contains_key("constraint_system") {
            self.create_constraint_system_bind_group_layout();
        }
        
        // Now we can safely create the pipeline with multiple bind group layouts
        if !self.pipelines.contains_key("rigid_body_constraints") {
            let physics_layout = self.bind_group_layouts.get("rigid_body_constraints").unwrap();
            let constraint_layout = self.bind_group_layouts.get("constraint_system").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "rigid_body_constraints", 
                "physics/rigid_body_constraints.wgsl", 
                &[physics_layout, constraint_layout]
            );
            self.pipelines.insert("rigid_body_constraints".to_string(), pipeline);
        }
        
        self.pipelines.get("rigid_body_constraints").unwrap()
    }
    
    /// Get the cell internal update pipeline.
    ///
    /// # Returns
    /// Reference to the cell internal update compute pipeline
    pub fn get_cell_internal_update_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("cell_internal_update") {
            self.create_cell_internal_update_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("cell_internal_update") {
            let layout = self.bind_group_layouts.get("cell_internal_update").unwrap();
            let pipeline = self.create_compute_pipeline("cell_internal_update", "physics/cell_update_internal.wgsl", layout);
            self.pipelines.insert("cell_internal_update".to_string(), pipeline);
        }
        
        self.pipelines.get("cell_internal_update").unwrap()
    }
    
    /// Get the nutrient system pipeline.
    ///
    /// # Returns
    /// Reference to the nutrient system compute pipeline
    pub fn get_nutrient_system_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Nutrient system uses TWO bind groups: nutrient data + read-only adhesion access
        // Ensure both layouts exist
        if !self.bind_group_layouts.contains_key("nutrient_system") {
            self.create_nutrient_system_bind_group_layout();
        }
        if !self.bind_group_layouts.contains_key("nutrient_adhesion") {
            self.create_nutrient_adhesion_bind_group_layout();
        }
        
        // Now we can safely create the pipeline with multiple bind group layouts
        if !self.pipelines.contains_key("nutrient_system") {
            let nutrient_layout = self.bind_group_layouts.get("nutrient_system").unwrap();
            let adhesion_layout = self.bind_group_layouts.get("nutrient_adhesion").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "nutrient_system", 
                "physics/nutrient_system.wgsl", 
                &[nutrient_layout, adhesion_layout]
            );
            self.pipelines.insert("nutrient_system".to_string(), pipeline);
        }
        
        self.pipelines.get("nutrient_system").unwrap()
    }
    
    /// Get the lifecycle death scan pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle death scan compute pipeline
    pub fn get_lifecycle_death_scan_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("lifecycle_death_scan") {
            self.create_lifecycle_death_scan_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("lifecycle_death_scan") {
            let layout = self.bind_group_layouts.get("lifecycle_death_scan").unwrap();
            let pipeline = self.create_compute_pipeline("lifecycle_death_scan", "lifecycle/lifecycle_death_scan.wgsl", layout);
            self.pipelines.insert("lifecycle_death_scan".to_string(), pipeline);
        }
        
        self.pipelines.get("lifecycle_death_scan").unwrap()
    }
    
    /// Create a bind group layout for lifecycle death scan operations.
    ///
    /// This layout matches the lifecycle_death_scan.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Ages (read-only storage)
    /// - Nitrates (read-only storage)
    /// - Toxins (read-only storage)
    /// - Birth times (read-only storage)
    /// - Death flags (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for lifecycle death scan pipeline
    pub fn create_lifecycle_death_scan_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_death_scan";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Death Scan Bind Group Layout"),
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
                    // Binding 2: Ages (read-only storage)
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
                    // Binding 3: Nitrates (read-only storage)
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
                    // Binding 4: Toxins (read-only storage)
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
                    // Binding 5: Birth times (read-only storage)
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
                    // Binding 6: Death flags (read-write storage)
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
    
    /// Get the lifecycle death compact pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle death compact compute pipeline
    pub fn get_lifecycle_death_compact_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure all 3 bind group layouts exist
        if !self.bind_group_layouts.contains_key("lifecycle_death_compact_group0") {
            self.create_lifecycle_death_compact_bind_group_layouts();
        }
        
        // Now we can safely get the layouts and create the pipeline
        if !self.pipelines.contains_key("lifecycle_death_compact") {
            let group0_layout = self.bind_group_layouts.get("lifecycle_death_compact_group0").unwrap();
            let group1_layout = self.bind_group_layouts.get("lifecycle_death_compact_group1").unwrap();
            let group2_layout = self.bind_group_layouts.get("lifecycle_death_compact_group2").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "lifecycle_death_compact", 
                "lifecycle/lifecycle_death_compact.wgsl", 
                &[group0_layout, group1_layout, group2_layout]
            );
            self.pipelines.insert("lifecycle_death_compact".to_string(), pipeline);
        }
        
        self.pipelines.get("lifecycle_death_compact").unwrap()
    }
    
    /// Create bind group layouts for lifecycle death compact operations.
    ///
    /// This shader uses 3 bind groups matching lifecycle_death_compact.wgsl requirements:
    /// - Group 0: Physics parameters and cell data (4 bindings)
    /// - Group 1: Cell property arrays (21 bindings)
    /// - Group 2: Adhesion system (3 bindings)
    ///
    /// # Returns
    /// Creates and stores all 3 bind group layouts
    pub fn create_lifecycle_death_compact_bind_group_layouts(&mut self) {
        // Group 0: Physics parameters and cell data
        if !self.bind_group_layouts.contains_key("lifecycle_death_compact_group0") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Death Compact Group 0 Layout"),
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
                    // Binding 1: Cell count buffer (read-write storage)
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
                    // Binding 2: Death flags (read-only storage)
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
                    // Binding 3: Death prefix sum (read-only storage)
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
            self.bind_group_layouts.insert("lifecycle_death_compact_group0".to_string(), layout);
        }
        
        // Group 1: Cell property arrays (21 bindings, all read-write for compaction)
        if !self.bind_group_layouts.contains_key("lifecycle_death_compact_group1") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Death Compact Group 1 Layout"),
                entries: &[
                    // Binding 0: position_and_mass
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
                    // Binding 1: velocity
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
                    // Binding 2: acceleration
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
                    // Binding 3: prev_acceleration
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
                    // Binding 4: orientation
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
                    // Binding 5: genome_orientation
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
                    // Binding 6: angular_velocity
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
                    // Binding 7: angular_acceleration
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
                    // Binding 8: prev_angular_acceleration
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
                    // Binding 9: signalling_substances
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
                    // Binding 10: mode_indices
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
                    // Binding 11: ages
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
                    // Binding 12: toxins
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
                    // Binding 13: nitrates
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
                    // Binding 14: cell_ids
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
                    // Binding 15: genome_ids
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
                    // Binding 16: birth_times
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
                    // Binding 17: split_intervals
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
                    // Binding 18: split_masses
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
                    // Binding 19: split_counts
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
                    // Binding 20: split_ready_frame
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
                ],
            });
            self.bind_group_layouts.insert("lifecycle_death_compact_group1".to_string(), layout);
        }
        
        // Group 2: Adhesion system (3 bindings)
        if !self.bind_group_layouts.contains_key("lifecycle_death_compact_group2") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Death Compact Group 2 Layout"),
                entries: &[
                    // Binding 0: adhesion_connections (read-write storage)
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
                    // Binding 1: adhesion_indices (read-write storage)
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
                    // Binding 2: adhesion_counts (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_death_compact_group2".to_string(), layout);
        }
    }
    
    /// Get the lifecycle division scan pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle division scan compute pipeline
    pub fn get_lifecycle_division_scan_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("lifecycle_division_scan") {
            self.create_lifecycle_division_scan_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("lifecycle_division_scan") {
            let layout = self.bind_group_layouts.get("lifecycle_division_scan").unwrap();
            let pipeline = self.create_compute_pipeline("lifecycle_division_scan", "lifecycle/lifecycle_division_scan.wgsl", layout);
            self.pipelines.insert("lifecycle_division_scan".to_string(), pipeline);
        }
        
        self.pipelines.get("lifecycle_division_scan").unwrap()
    }
    
    /// Create a bind group layout for lifecycle division scan operations.
    ///
    /// This layout matches the lifecycle_division_scan.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Position and mass (read-only storage)
    /// - Split ready frame (read-only storage)
    /// - Split masses (read-only storage)
    /// - Birth times (read-only storage)
    /// - Split intervals (read-only storage)
    /// - Adhesion counts (read-only storage)
    /// - Division candidates (read-write storage)
    /// - Division reservations (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for lifecycle division scan pipeline
    pub fn create_lifecycle_division_scan_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_division_scan";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Scan Bind Group Layout"),
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
                    // Binding 2: Split ready frame (read-only storage)
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
                    // Binding 3: Split masses (read-only storage)
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
                    // Binding 4: Birth times (read-only storage)
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
                    // Binding 5: Split intervals (read-only storage)
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
                    // Binding 6: Adhesion counts (read-only storage)
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
                    // Binding 7: Division candidates (read-write storage)
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
                    // Binding 8: Division reservations (read-write storage)
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
    
    /// Get the lifecycle slot assign pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle slot assign compute pipeline
    pub fn get_lifecycle_slot_assign_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("lifecycle_slot_assign") {
            self.create_lifecycle_slot_assign_bind_group_layout();
        }
        
        // Now we can safely get the layout and create the pipeline
        if !self.pipelines.contains_key("lifecycle_slot_assign") {
            let layout = self.bind_group_layouts.get("lifecycle_slot_assign").unwrap();
            let pipeline = self.create_compute_pipeline("lifecycle_slot_assign", "lifecycle/lifecycle_slot_assign.wgsl", layout);
            self.pipelines.insert("lifecycle_slot_assign".to_string(), pipeline);
        }
        
        self.pipelines.get("lifecycle_slot_assign").unwrap()
    }
    
    /// Create a bind group layout for lifecycle slot assign operations.
    ///
    /// This layout matches the lifecycle_slot_assign.wgsl shader requirements:
    /// - Physics parameters (uniform)
    /// - Cell count buffer (read-write storage)
    /// - Division candidates (read-only storage)
    /// - Division reservations (read-only storage)
    /// - Reservation prefix sum (read-only storage)
    /// - Free slots (read-only storage)
    /// - Free slot count (read-only storage)
    /// - Division assignments (read-write storage)
    ///
    /// # Returns
    /// Bind group layout for lifecycle slot assign pipeline
    pub fn create_lifecycle_slot_assign_bind_group_layout(&mut self) -> &wgpu::BindGroupLayout {
        let layout_name = "lifecycle_slot_assign";
        
        if !self.bind_group_layouts.contains_key(layout_name) {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Slot Assign Bind Group Layout"),
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
                    // Binding 1: Cell count buffer (read-write storage)
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
                    // Binding 2: Division candidates (read-only storage)
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
                    // Binding 3: Division reservations (read-only storage)
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
                    // Binding 4: Reservation prefix sum (read-only storage)
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
                    // Binding 5: Free slots (read-only storage)
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
                    // Binding 6: Free slot count (read-only storage)
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
                    // Binding 7: Division assignments (read-write storage)
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
    
    /// Get the lifecycle division execute pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle division execute compute pipeline
    pub fn get_lifecycle_division_execute_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure all 5 bind group layouts exist
        if !self.bind_group_layouts.contains_key("lifecycle_division_execute_group0") {
            self.create_lifecycle_division_execute_bind_group_layouts();
        }
        
        // Now we can safely get the layouts and create the pipeline
        if !self.pipelines.contains_key("lifecycle_division_execute") {
            let group0_layout = self.bind_group_layouts.get("lifecycle_division_execute_group0").unwrap();
            let group1_layout = self.bind_group_layouts.get("lifecycle_division_execute_group1").unwrap();
            let group2_layout = self.bind_group_layouts.get("lifecycle_division_execute_group2").unwrap();
            let group3_layout = self.bind_group_layouts.get("lifecycle_division_execute_group3").unwrap();
            let group4_layout = self.bind_group_layouts.get("lifecycle_division_execute_group4").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "lifecycle_division_execute", 
                "lifecycle/lifecycle_division_execute.wgsl", 
                &[group0_layout, group1_layout, group2_layout, group3_layout, group4_layout]
            );
            self.pipelines.insert("lifecycle_division_execute".to_string(), pipeline);
        }
        
        self.pipelines.get("lifecycle_division_execute").unwrap()
    }
    
    /// Create bind group layouts for lifecycle division execute operations.
    ///
    /// This shader uses 5 bind groups matching lifecycle_division_execute.wgsl requirements:
    /// - Group 0: Physics parameters and cell count (2 bindings)
    /// - Group 1: Cell data buffers (7 bindings)
    /// - Group 2: Division management buffers (4 bindings)
    /// - Group 3: Adhesion system buffers (3 bindings)
    /// - Group 4: Genome mode data (2 bindings)
    ///
    /// # Returns
    /// Creates and stores all 5 bind group layouts
    pub fn create_lifecycle_division_execute_bind_group_layouts(&mut self) {
        // Group 0: Physics parameters and cell count
        if !self.bind_group_layouts.contains_key("lifecycle_division_execute_group0") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Execute Group 0 Layout"),
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
                    // Binding 1: Cell count buffer (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_division_execute_group0".to_string(), layout);
        }
        
        // Group 1: Cell data buffers (7 bindings)
        if !self.bind_group_layouts.contains_key("lifecycle_division_execute_group1") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Execute Group 1 Layout"),
                entries: &[
                    // Binding 0: position_and_mass (read-write storage)
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
                    // Binding 1: velocity (read-write storage)
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
                    // Binding 2: acceleration (read-write storage)
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
                    // Binding 3: cell_age (read-write storage)
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
                    // Binding 4: nutrients (read-write storage)
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
                    // Binding 5: signaling_substances (read-write storage)
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
                    // Binding 6: toxins (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_division_execute_group1".to_string(), layout);
        }
        
        // Group 2: Division management buffers (4 bindings)
        if !self.bind_group_layouts.contains_key("lifecycle_division_execute_group2") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Execute Group 2 Layout"),
                entries: &[
                    // Binding 0: division_candidates (read-only storage)
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
                    // Binding 1: division_assignments (read-only storage)
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
                    // Binding 2: division_count (read-only storage)
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
                    // Binding 3: free_slots (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_division_execute_group2".to_string(), layout);
        }
        
        // Group 3: Adhesion system buffers (3 bindings)
        if !self.bind_group_layouts.contains_key("lifecycle_division_execute_group3") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Execute Group 3 Layout"),
                entries: &[
                    // Binding 0: adhesion_connections (read-write storage)
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
                    // Binding 1: adhesion_indices (read-write storage)
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
                    // Binding 2: adhesion_counts (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_division_execute_group3".to_string(), layout);
        }
        
        // Group 4: Genome mode data (2 bindings)
        if !self.bind_group_layouts.contains_key("lifecycle_division_execute_group4") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Division Execute Group 4 Layout"),
                entries: &[
                    // Binding 0: modes (read-only storage)
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
                    // Binding 1: genome_ids (read-only storage for reading, read-write for Child B)
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
            self.bind_group_layouts.insert("lifecycle_division_execute_group4".to_string(), layout);
        }
    }
    
    /// Get the lifecycle free slots pipeline.
    ///
    /// # Returns
    /// Reference to the lifecycle free slots compute pipeline
    pub fn get_lifecycle_free_slots_pipeline(&mut self) -> &wgpu::ComputePipeline {
        // Ensure layout exists
        if !self.bind_group_layouts.contains_key("lifecycle_free_slots_group0") {
            self.create_lifecycle_free_slots_bind_group_layouts();
        }
        
        // Now we can safely get the layouts and create the pipeline
        if !self.pipelines.contains_key("lifecycle_free_slots") {
            let group0_layout = self.bind_group_layouts.get("lifecycle_free_slots_group0").unwrap();
            let group1_layout = self.bind_group_layouts.get("lifecycle_free_slots_group1").unwrap();
            let group2_layout = self.bind_group_layouts.get("lifecycle_free_slots_group2").unwrap();
            let group3_layout = self.bind_group_layouts.get("lifecycle_free_slots_group3").unwrap();
            let pipeline = self.create_compute_pipeline_multi_layout(
                "lifecycle_free_slots", 
                "lifecycle/lifecycle_free_slots.wgsl", 
                &[group0_layout, group1_layout, group2_layout, group3_layout]
            );
            self.pipelines.insert("lifecycle_free_slots".to_string(), pipeline);
        }
        
        self.pipelines.get("lifecycle_free_slots").unwrap()
    }
    
    /// Create bind group layouts for lifecycle free slots operations.
    ///
    /// This shader uses 4 bind groups matching lifecycle_free_slots.wgsl requirements:
    /// Group 0: Physics parameters and cell count
    /// Group 1: Lifecycle management buffers  
    /// Group 2: Free slot management buffers
    /// Group 3: Prefix-sum working buffers
    ///
    /// # Returns
    /// Creates and stores all 4 bind group layouts
    pub fn create_lifecycle_free_slots_bind_group_layouts(&mut self) {
        // Group 0: Physics parameters and cell count
        if !self.bind_group_layouts.contains_key("lifecycle_free_slots_group0") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Free Slots Group 0 Layout"),
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
                    // Binding 1: Cell count buffer (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_free_slots_group0".to_string(), layout);
        }
        
        // Group 1: Lifecycle management buffers
        if !self.bind_group_layouts.contains_key("lifecycle_free_slots_group1") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Free Slots Group 1 Layout"),
                entries: &[
                    // Binding 0: Death flags (read-only storage)
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
                    // Binding 1: Death compacted (read-only storage)
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
                    // Binding 2: Death count (read-only storage)
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
                    // Binding 3: Division assignments (read-only storage)
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
                    // Binding 4: Division count (read-only storage)
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
            self.bind_group_layouts.insert("lifecycle_free_slots_group1".to_string(), layout);
        }
        
        // Group 2: Free slot management buffers
        if !self.bind_group_layouts.contains_key("lifecycle_free_slots_group2") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Free Slots Group 2 Layout"),
                entries: &[
                    // Binding 0: Free slots (read-write storage)
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
                    // Binding 1: Free slot count (read-write storage)
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
                    // Binding 2: Temp slots (read-write storage)
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
                    // Binding 3: Slot flags (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_free_slots_group2".to_string(), layout);
        }
        
        // Group 3: Prefix-sum working buffers
        if !self.bind_group_layouts.contains_key("lifecycle_free_slots_group3") {
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lifecycle Free Slots Group 3 Layout"),
                entries: &[
                    // Binding 0: Prefix sum input (read-write storage)
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
                    // Binding 1: Prefix sum output (read-write storage)
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
                    // Binding 2: Prefix sum temp (read-write storage)
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
            self.bind_group_layouts.insert("lifecycle_free_slots_group3".to_string(), layout);
        }
    }
}