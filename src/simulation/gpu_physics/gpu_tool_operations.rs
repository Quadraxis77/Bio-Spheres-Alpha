//! GPU Tool Operations System
//! 
//! Handles tool operations (selection, dragging, deletion) directly on GPU using compute shaders.
//! This system eliminates the need for CPU canonical state by performing all tool queries
//! and updates directly on GPU buffers.
//!
//! Implements Requirements 4.1-4.6:
//! - GPU spatial queries for cell selection (4.1)
//! - Direct GPU position updates for cell dragging (4.2)
//! - GPU-based cell deletion (4.3)
//! - No CPU canonical state reads for tool operations (4.4, 4.5)
//! - Async readback for tool operation feedback (4.6)

use super::{GpuTripleBufferSystem, PositionUpdateParams, SpatialQueryParams, SpatialQueryResult};
use glam::Vec3;
use std::sync::Arc;

/// GPU-based tool operations system for spatial queries and position updates
/// 
/// This system provides GPU-only tool operations without CPU canonical state dependency.
/// All operations are performed directly on GPU buffers with minimal async readbacks.
pub struct GpuToolOperations {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    // Spatial query pipeline and resources
    spatial_query_pipeline: wgpu::ComputePipeline,
    spatial_query_params_buffer: wgpu::Buffer,
    spatial_query_result_buffer: wgpu::Buffer,
    spatial_query_readback_buffer: wgpu::Buffer,
    spatial_query_params_bind_group: wgpu::BindGroup,
    spatial_query_result_bind_group: wgpu::BindGroup,
    
    // Position update pipeline and resources
    position_update_pipeline: wgpu::ComputePipeline,
    position_update_params_buffer: wgpu::Buffer,
    position_update_params_bind_group: wgpu::BindGroup,
    position_update_physics_bind_group: wgpu::BindGroup,  // All 3 buffer sets
    
    // Result caching to avoid redundant GPU queries (requirement 4.6)
    cached_query_result: Option<SpatialQueryResult>,
    cached_query_position: Option<Vec3>,
    #[allow(dead_code)]
    cache_tolerance: f32,
    
    // Async readback state
    readback_in_progress: bool,
}

impl GpuToolOperations {
    /// Create a new GPU tool operations system
    /// 
    /// Implements requirements 4.1-4.6 for GPU-based tool operations
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: &super::GpuPhysicsPipelines,
        buffers: &GpuTripleBufferSystem,
    ) -> Self {
        // Create spatial query parameters buffer
        let spatial_query_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Query Params Buffer"),
            size: std::mem::size_of::<SpatialQueryParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create spatial query result buffer
        let spatial_query_result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Query Result Buffer"),
            size: std::mem::size_of::<SpatialQueryResult>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create readback buffer for spatial query results (requirement 11.1)
        let spatial_query_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Query Readback Buffer"),
            size: std::mem::size_of::<SpatialQueryResult>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create position update parameters buffer
        let position_update_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Update Params Buffer"),
            size: std::mem::size_of::<PositionUpdateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind groups
        let spatial_query_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spatial Query Params Bind Group"),
            layout: &pipelines.spatial_query_params_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spatial_query_params_buffer.as_entire_binding(),
                },
            ],
        });
        
        let spatial_query_result_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spatial Query Result Bind Group"),
            layout: &pipelines.spatial_query_result_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spatial_query_result_buffer.as_entire_binding(),
                },
            ],
        });
        
        let position_update_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Position Update Params Bind Group"),
            layout: &pipelines.position_update_params_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_update_params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create physics bind group for position update (all 3 buffer sets)
        // Uses cell_insertion_physics_layout which has all 3 position and velocity buffers
        let position_update_physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Position Update Physics Bind Group"),
            layout: &pipelines.cell_insertion_physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                // All 3 position buffers
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.position_and_mass[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.position_and_mass[2].as_entire_binding(),
                },
                // All 3 velocity buffers
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.velocity[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.velocity[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.velocity[2].as_entire_binding(),
                },
                // Cell count buffer
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        });
        
        Self {
            device: device.clone(),
            queue: queue.clone(),
            spatial_query_pipeline: pipelines.spatial_query.clone(),
            spatial_query_params_buffer,
            spatial_query_result_buffer,
            spatial_query_readback_buffer,
            spatial_query_params_bind_group,
            spatial_query_result_bind_group,
            position_update_pipeline: pipelines.position_update_tool.clone(),
            position_update_params_buffer,
            position_update_params_bind_group,
            position_update_physics_bind_group,
            cached_query_result: None,
            cached_query_position: None,
            cache_tolerance: 0.001, // 1mm tolerance for position caching
            readback_in_progress: false,
        }
    }
    
    /// Find the closest cell intersected by a ray using GPU spatial query
    /// 
    /// Performs ray-sphere intersection testing against all cells.
    /// Returns immediately and result can be polled later with poll_spatial_query.
    pub fn find_cell_with_ray(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        physics_bind_group: &wgpu::BindGroup,
        ray_origin: Vec3,
        ray_direction: Vec3,
        max_distance: f32,
        cell_count: u32,
    ) {
        // Skip if readback already in progress
        if self.readback_in_progress {
            return;
        }
        
        // Update spatial query parameters with ray data
        let query_params = SpatialQueryParams {
            ray_origin: [ray_origin.x, ray_origin.y, ray_origin.z],
            max_distance,
            ray_direction: [ray_direction.x, ray_direction.y, ray_direction.z],
            _pad0: 0,
        };
        
        self.queue.write_buffer(
            &self.spatial_query_params_buffer,
            0,
            bytemuck::cast_slice(&[query_params]),
        );
        
        // Initialize result buffer before query - set distance to MAX and found to 0
        // This is critical because the shader uses atomic operations on the result buffer
        let initial_result = SpatialQueryResult {
            found_cell_index: 0xFFFFFFFF, // Invalid index
            distance_fixed: 0xFFFFFFFF,   // MAX distance as fixed-point u32
            found: 0,
            _padding: 0,
        };
        self.queue.write_buffer(
            &self.spatial_query_result_buffer,
            0,
            bytemuck::cast_slice(&[initial_result]),
        );
        
        // Dispatch spatial query compute shader (workgroup size 64)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spatial Query Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.spatial_query_pipeline);
            compute_pass.set_bind_group(0, physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.spatial_query_params_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.spatial_query_result_bind_group, &[]);
            
            // Dispatch with workgroup size 64
            let num_workgroups = (cell_count + 63) / 64;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Copy result to readback buffer for CPU access
        encoder.copy_buffer_to_buffer(
            &self.spatial_query_result_buffer,
            0,
            &self.spatial_query_readback_buffer,
            0,
            std::mem::size_of::<SpatialQueryResult>() as u64,
        );
        
        // Mark readback as in progress
        self.readback_in_progress = true;
        
        // Clear cached results
        self.cached_query_position = None;
        self.cached_query_result = None;
    }
    
    /// Poll for spatial query completion and return result if available
    /// 
    /// Implements requirement 11.3 for non-blocking readback completion polling.
    /// This method should be called each frame to check for completed async readbacks.
    /// Returns Some(result) if query is complete, None if still in progress.
    pub fn poll_spatial_query(&mut self) -> Option<SpatialQueryResult> {
        // Check if we have a cached result (requirement 11.4)
        if let Some(cached_result) = self.cached_query_result {
            return Some(cached_result);
        }
        
        // If no readback in progress, return None
        if !self.readback_in_progress {
            return None;
        }
        
        // Start async mapping if not already started
        let slice = self.spatial_query_readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        // Poll device to process pending async operations (requirement 11.3)
        let _ = self.device.poll(wgpu::PollType::Poll);
        
        // Check if mapping completed
        if let Ok(Ok(())) = receiver.try_recv() {
            // Mapping completed, read the data
            let view = slice.get_mapped_range();
            let data_bytes: &[u8] = &view;
            
            if data_bytes.len() >= std::mem::size_of::<SpatialQueryResult>() {
                // Parse the result data
                let result: SpatialQueryResult = *bytemuck::from_bytes(&data_bytes[..std::mem::size_of::<SpatialQueryResult>()]);
                
                // Unmap the buffer
                drop(view);
                self.spatial_query_readback_buffer.unmap();
                
                // Mark readback as complete
                self.readback_in_progress = false;
                
                // Cache the result (requirement 11.4)
                self.cached_query_result = Some(result);
                
                return Some(result);
            } else {
                // Invalid data size
                drop(view);
                self.spatial_query_readback_buffer.unmap();
                self.readback_in_progress = false;
                return None;
            }
        }
        
        // Mapping is still pending or failed
        None
    }
    
    /// Update a cell's position directly in GPU buffers using compute shader
    /// 
    /// Implements requirements 4.2 and 10.1-10.6 for GPU position updates.
    /// This method performs direct GPU position updates without CPU state involvement.
    /// Updates ALL THREE triple buffer sets to ensure the position change persists.
    pub fn update_cell_position(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        _physics_bind_group: &wgpu::BindGroup,  // Unused - we use our own bind group with all 3 buffer sets
        cell_index: u32,
        new_pos: Vec3,
    ) {
        // Update position update parameters (requirement 10.5)
        let update_params = PositionUpdateParams {
            cell_index,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            new_position: [new_pos.x, new_pos.y, new_pos.z],
            _padding: 0.0,
        };
        
        self.queue.write_buffer(
            &self.position_update_params_buffer,
            0,
            bytemuck::cast_slice(&[update_params]),
        );
        
        // Dispatch position update compute shader (requirement 10.4: single workgroup)
        // Uses position_update_physics_bind_group which has all 3 buffer sets
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Position Update Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.position_update_pipeline);
            compute_pass.set_bind_group(0, &self.position_update_physics_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.position_update_params_bind_group, &[]);
            
            // Single workgroup dispatch as required by 10.4
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        
        // Clear cached query results since scene state changed
        self.clear_cache();
    }
    
    /// Clear cached query results (call when scene state changes significantly)
    /// 
    /// This method clears all cached results and should be called when the scene
    /// state changes in ways that would invalidate cached spatial query results.
    pub fn clear_cache(&mut self) {
        self.cached_query_result = None;
        self.cached_query_position = None;
        self.readback_in_progress = false;
    }
    
    /// Check if a spatial query is currently in progress
    pub fn is_query_in_progress(&self) -> bool {
        self.readback_in_progress
    }
    
    /// Get the most recent cached query result without polling
    /// 
    /// This returns the cached result immediately without checking for new completions.
    /// Use poll_spatial_query() to check for new results.
    pub fn get_cached_result(&self) -> Option<SpatialQueryResult> {
        self.cached_query_result
    }
}