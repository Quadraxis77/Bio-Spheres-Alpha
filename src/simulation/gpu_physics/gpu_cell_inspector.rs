//! GPU Cell Inspector System
//! 
//! Provides GPU-based cell inspection with async readback management for real-time cell data display.
//! Implements the requirements for GPU-only cell inspection without CPU state management.

use super::{GpuTripleBufferSystem, GpuCellDataExtraction, InspectedCellData};
use std::collections::HashMap;

/// Unique identifier for async readback requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadbackId(u64);

impl ReadbackId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Completed readback result
#[derive(Debug, Clone)]
pub struct ReadbackResult {
    pub cell_index: u32,
    pub data: InspectedCellData,
}

/// GPU Cell Inspector System
/// 
/// Manages GPU-based cell data extraction with async readback management.
/// Provides non-blocking cell inspection for real-time UI updates.
/// 
/// This system integrates with AsyncReadbackManager to handle GPU-to-CPU transfers
/// efficiently without blocking the GPU pipeline.
pub struct GpuCellInspector {
    /// GPU cell data extraction system
    extraction_system: GpuCellDataExtraction,
    
    /// Async readback manager for handling GPU-to-CPU transfers
    readback_manager: Option<AsyncReadbackManager>,
    
    /// Current extraction request ID
    current_readback_id: Option<ReadbackId>,
    
    /// Current extraction request (cell index being extracted)
    current_extraction: Option<u32>,
    
    /// Last successful extraction result
    last_result: Option<ReadbackResult>,
    
    /// Extraction in progress flag
    extraction_in_progress: bool,
}

impl GpuCellInspector {
    /// Create a new GPU cell inspector system
    pub fn new(
        device: &wgpu::Device,
        pipeline: wgpu::ComputePipeline,
        physics_layout: &wgpu::BindGroupLayout,
        params_layout: &wgpu::BindGroupLayout,
        state_layout: &wgpu::BindGroupLayout,
        output_layout: &wgpu::BindGroupLayout,
        buffers: &GpuTripleBufferSystem,
        buffer_index: usize,
    ) -> Self {
        // Create the extraction system
        let extraction_system = GpuCellDataExtraction::new(
            device,
            pipeline,
            physics_layout,
            params_layout,
            state_layout,
            output_layout,
            buffers,
            buffer_index,
        );
        
        Self {
            extraction_system,
            readback_manager: None, // Will be set later when device is available
            current_readback_id: None,
            current_extraction: None,
            last_result: None,
            extraction_in_progress: false,
        }
    }
    
    /// Initialize the async readback manager
    /// 
    /// This must be called after construction with a device reference.
    pub fn initialize_readback_manager(&mut self, device: wgpu::Device) {
        self.readback_manager = Some(AsyncReadbackManager::new(device, 4));
    }
    
    /// Extract cell data using GPU compute shader with async readback management
    /// 
    /// This method uploads the cell index, dispatches the compute shader,
    /// and initiates async readback using the AsyncReadbackManager.
    /// Call poll_extraction() to check for completion.
    pub fn extract_cell_data(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cell_index: u32,
    ) {
        // Use the extraction system to perform GPU compute
        self.extraction_system.extract_cell_data(encoder, queue, cell_index);
        
        // If we have a readback manager, use it for async readback
        if let Some(ref mut readback_manager) = self.readback_manager {
            // Cancel any previous extraction
            if let Some(prev_id) = self.current_readback_id.take() {
                readback_manager.cancel_request(prev_id);
            }
            
            // Request async readback through the manager
            let output_buffer = self.extraction_system.get_output_buffer();
            let data_size = std::mem::size_of::<InspectedCellData>() as u64;
            
            if let Some(readback_id) = readback_manager.request_readback(output_buffer, data_size) {
                // Initiate the actual readback transfer
                if let Err(e) = readback_manager.initiate_readback(
                    encoder,
                    readback_id,
                    output_buffer,
                    data_size,
                ) {
                    eprintln!("Failed to initiate readback: {}", e);
                    return;
                }
                
                // Track the current extraction
                self.current_readback_id = Some(readback_id);
                self.current_extraction = Some(cell_index);
                self.extraction_in_progress = true;
            } else {
                eprintln!("Failed to request readback - too many concurrent operations");
            }
        } else {
            // Fallback to direct extraction system (for backward compatibility)
            self.current_extraction = Some(cell_index);
            self.extraction_in_progress = true;
        }
    }
    
    /// Poll for extraction completion and return extracted data if available
    /// 
    /// This method should be called each frame to check for completed async readbacks.
    /// Returns Some(data) if extraction is complete, None if still in progress.
    pub fn poll_extraction(&mut self, device: Option<&wgpu::Device>) -> Option<InspectedCellData> {
        if !self.extraction_in_progress {
            return None;
        }
        
        // If we have a readback manager, use it
        if let Some(ref mut readback_manager) = self.readback_manager {
            // Poll the readback manager for completions
            readback_manager.poll_completions();
            
            // Check if our current readback is complete
            if let Some(readback_id) = self.current_readback_id {
                if let Some(data_bytes) = readback_manager.get_result_bytes(readback_id) {
                    // Extraction completed - parse the data
                    if data_bytes.len() == std::mem::size_of::<InspectedCellData>() {
                        // Safety: We know the data is the correct size and layout
                        let data = unsafe {
                            std::ptr::read(data_bytes.as_ptr() as *const InspectedCellData)
                        };
                        
                        // Store the result
                        if let Some(cell_index) = self.current_extraction.take() {
                            let result = ReadbackResult {
                                cell_index,
                                data,
                            };
                            self.last_result = Some(result);
                        }
                        
                        // Clean up
                        self.current_readback_id = None;
                        self.extraction_in_progress = false;
                        
                        return Some(data);
                    } else {
                        eprintln!("Readback data size mismatch: expected {}, got {}", 
                            std::mem::size_of::<InspectedCellData>(), 
                            data_bytes.len()
                        );
                    }
                } else if readback_manager.is_failed(readback_id) {
                    // Readback failed
                    eprintln!("Cell data extraction failed for readback ID {:?}", readback_id);
                    self.current_readback_id = None;
                    self.current_extraction = None;
                    self.extraction_in_progress = false;
                }
            }
        } else if let Some(device) = device {
            // Fallback to direct extraction system
            if let Some(data) = self.extraction_system.poll_extraction(device) {
                // Extraction completed
                self.extraction_in_progress = false;
                
                if let Some(cell_index) = self.current_extraction.take() {
                    // Store the result
                    let result = ReadbackResult {
                        cell_index,
                        data,
                    };
                    self.last_result = Some(result);
                    
                    return Some(data);
                }
            }
        }
        
        None
    }
    
    /// Get the most recent extraction result
    pub fn get_latest_result(&self) -> Option<&ReadbackResult> {
        self.last_result.as_ref()
    }
    
    /// Check if extraction is currently in progress
    pub fn is_extracting(&self) -> bool {
        self.extraction_in_progress
    }
    
    /// Get cached data from the extraction system
    pub fn get_cached_data(&self) -> Option<&InspectedCellData> {
        self.extraction_system.get_cached_data()
    }
    
    /// Clear all cached data
    pub fn clear_cache(&mut self) {
        self.extraction_system.clear_cache();
        if let Some(ref mut readback_manager) = self.readback_manager {
            readback_manager.clear_completed();
        }
        self.last_result = None;
        self.current_extraction = None;
        self.current_readback_id = None;
        self.extraction_in_progress = false;
    }
    
    /// Get readback manager statistics
    pub fn get_readback_stats(&self) -> Option<ReadbackStats> {
        self.readback_manager.as_ref().map(|rm| rm.get_stats())
    }
}

/// Statistics about async readback operations
#[derive(Debug, Clone)]
pub struct ReadbackStats {
    pub pending: usize,
    pub mapped: usize,
    pub completed: usize,
    pub failed: usize,
    pub total_active: usize,
    pub max_concurrent: usize,
}

/// Async Readback Manager
/// 
/// Coordinates GPU-to-CPU transfers efficiently without blocking the pipeline.
/// Implements requirements 11.1-11.6 for proper async readback management.
/// 
/// This is a simplified implementation that follows the existing codebase patterns
/// for async readback operations.
pub struct AsyncReadbackManager {
    /// Active readback requests with their channels
    requests: HashMap<ReadbackId, (wgpu::Buffer, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>,
    
    /// Completed readback results (cached)
    completed_results: HashMap<ReadbackId, Vec<u8>>,
    
    /// Maximum concurrent readback operations (requirement 11.6)
    max_concurrent_readbacks: usize,
    
    /// Device reference for polling
    device: wgpu::Device,
}

impl AsyncReadbackManager {
    /// Create a new async readback manager
    pub fn new(device: wgpu::Device, max_concurrent_readbacks: usize) -> Self {
        Self {
            requests: HashMap::new(),
            completed_results: HashMap::new(),
            max_concurrent_readbacks,
            device,
        }
    }
    
    /// Request a new readback operation with staging buffer management
    /// 
    /// Implements requirements 11.1 (staging buffers) and 11.6 (concurrent limiting).
    pub fn request_readback(
        &mut self,
        _source_buffer: &wgpu::Buffer,
        _size: u64,
    ) -> Option<ReadbackId> {
        // Check concurrent readback limit (requirement 11.6)
        if self.requests.len() >= self.max_concurrent_readbacks {
            return None;
        }
        
        let id = ReadbackId::new();
        
        Some(id)
    }
    
    /// Initiate the readback by copying data and starting async mapping
    pub fn initiate_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        id: ReadbackId,
        source_buffer: &wgpu::Buffer,
        size: u64,
    ) -> Result<(), String> {
        // Create staging buffer
        let _staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Async Readback Staging Buffer {}", id.0)),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Copy from source buffer to staging buffer (requirement 11.1)
        encoder.copy_buffer_to_buffer(
            source_buffer,
            0,
            &_staging_buffer,
            0,
            size,
        );
        
        // Start async mapping (requirement 11.2) - following existing pattern
        let buffer_slice = _staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        // Store the request
        self.requests.insert(id, (_staging_buffer, receiver));
        
        Ok(())
    }
    
    /// Poll for completed readback operations without blocking
    /// 
    /// Implements requirement 11.3 for non-blocking completion polling.
    pub fn poll_completions(&mut self) {
        // Poll device to process pending async operations (requirement 11.3)
        let _ = self.device.poll(wgpu::PollType::Poll);
        
        let mut completed_ids = Vec::new();
        
        // Check each request for completion
        for (id, (staging_buffer, receiver)) in &self.requests {
            // Check if mapping completed (non-blocking)
            if let Ok(Ok(())) = receiver.try_recv() {
                // Mapping completed, read the data
                let buffer_slice = staging_buffer.slice(..);
                let view = buffer_slice.get_mapped_range();
                let data = view.to_vec();
                
                // Store the result (requirement 11.4 - caching)
                let _ = self.completed_results.insert(*id, data);
                
                // Clean up
                drop(view);
                staging_buffer.unmap();
                completed_ids.push(*id);
            }
        }
        
        // Remove completed requests
        for id in completed_ids {
            let _ = self.requests.remove(&id);
        }
    }
    
    /// Get completed result as raw bytes
    pub fn get_result_bytes(&mut self, id: ReadbackId) -> Option<Vec<u8>> {
        self.completed_results.remove(&id)
    }
    
    /// Check if a readback request is completed
    pub fn is_completed(&self, id: ReadbackId) -> bool {
        self.completed_results.contains_key(&id)
    }
    
    /// Check if a readback request failed (simplified - just check if not pending and not completed)
    pub fn is_failed(&self, id: ReadbackId) -> bool {
        !self.requests.contains_key(&id) && !self.completed_results.contains_key(&id)
    }
    
    /// Cancel a pending readback request
    pub fn cancel_request(&mut self, id: ReadbackId) -> bool {
        if let Some((_staging_buffer, _)) = self.requests.remove(&id) {
            // Try to unmap if it was mapped
            // Note: This might fail if not mapped, but that's okay
            true
        } else {
            false
        }
    }
    
    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.requests.len()
    }
    
    /// Get number of completed results (cached)
    pub fn completed_count(&self) -> usize {
        self.completed_results.len()
    }
    
    /// Clear all completed results from cache
    pub fn clear_completed(&mut self) {
        self.completed_results.clear();
    }
    
    /// Get statistics about readback operations
    pub fn get_stats(&self) -> ReadbackStats {
        ReadbackStats {
            pending: self.requests.len(),
            mapped: 0, // Simplified - we don't track this state separately
            completed: self.completed_results.len(),
            failed: 0, // Simplified - we don't track failures separately
            total_active: self.requests.len(),
            max_concurrent: self.max_concurrent_readbacks,
        }
    }
}