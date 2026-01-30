//! Dynamic buffer management system
//! 
//! This module provides dynamic resizing capabilities for GPU buffers
//! to handle varying genome and cell counts efficiently.

use wgpu;

/// Dynamic buffer that can be resized as needed
pub struct DynamicBuffer {
    /// Current GPU buffer
    buffer: wgpu::Buffer,
    /// Current size in bytes
    current_size: u64,
    /// Usage flags for the buffer
    usage: wgpu::BufferUsages,
    /// Label for debugging
    label: String,
}

impl DynamicBuffer {
    /// Create a new dynamic buffer
    pub fn new(
        device: &wgpu::Device,
        initial_size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: initial_size,
            usage,
            mapped_at_creation: false,
        });
        
        Self {
            buffer,
            current_size: initial_size,
            usage,
            label: label.to_string(),
        }
    }
    
    /// Get the current buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    
    /// Get the current size
    pub fn size(&self) -> u64 {
        self.current_size
    }
    
    /// Ensure the buffer is at least the specified size
    /// Returns true if the buffer was resized
    pub fn ensure_size(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        required_size: u64,
    ) -> bool {
        if required_size <= self.current_size {
            return false; // No resize needed
        }
        
        // Calculate new size (grow by 50% or to required size, whichever is larger)
        let new_size = (self.current_size * 3 / 2).max(required_size);
        
        // Create new buffer
        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        
        // Copy old data to new buffer if there's any data to copy
        if self.current_size > 0 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dynamic Buffer Resize Encoder"),
            });
            
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, self.current_size);
            
            let command_buffer = encoder.finish();
            queue.submit(Some(command_buffer));
        }
        
        // Replace old buffer
        self.buffer = new_buffer;
        self.current_size = new_size;
        
        true
    }
    
    /// Resize the buffer to exactly the specified size
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_size: u64,
    ) -> bool {
        if new_size == self.current_size {
            return false; // No resize needed
        }
        
        // Create new buffer
        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        
        // Copy old data to new buffer (only up to the smaller size)
        let copy_size = self.current_size.min(new_size);
        if copy_size > 0 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dynamic Buffer Resize Encoder"),
            });
            
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, copy_size);
            
            let command_buffer = encoder.finish();
            queue.submit(Some(command_buffer));
        }
        
        // Replace old buffer
        self.buffer = new_buffer;
        self.current_size = new_size;
        
        true
    }
}

/// Manager for dynamic genome buffers
pub struct DynamicGenomeBufferManager {
    /// Dynamic mode properties buffer
    mode_properties: DynamicBuffer,
    
    /// Dynamic mode cell types buffer
    mode_cell_types: DynamicBuffer,
    
    /// Dynamic child mode indices buffer
    child_mode_indices: DynamicBuffer,
    
    /// Dynamic parent make adhesion flags buffer
    parent_make_adhesion_flags: DynamicBuffer,
    
    /// Dynamic child keep adhesion flags buffers
    child_a_keep_adhesion_flags: DynamicBuffer,
    child_b_keep_adhesion_flags: DynamicBuffer,
    
    /// Dynamic genome mode data buffer
    genome_mode_data: DynamicBuffer,
    
    /// Current total modes across all genomes
    total_modes: usize,
    
    /// Maximum supported modes
    max_modes: usize,
}

impl DynamicGenomeBufferManager {
    /// Create a new dynamic genome buffer manager
    pub fn new(device: &wgpu::Device, max_modes: usize) -> Self {
        let initial_size = (max_modes * 48) as u64; // 48 bytes per mode for worst case
        
        Self {
            mode_properties: DynamicBuffer::new(
                device,
                initial_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Mode Properties Buffer",
            ),
            mode_cell_types: DynamicBuffer::new(
                device,
                (max_modes * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Mode Cell Types Buffer",
            ),
            child_mode_indices: DynamicBuffer::new(
                device,
                (max_modes * 8) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Child Mode Indices Buffer",
            ),
            parent_make_adhesion_flags: DynamicBuffer::new(
                device,
                (max_modes * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Parent Make Adhesion Flags Buffer",
            ),
            child_a_keep_adhesion_flags: DynamicBuffer::new(
                device,
                (max_modes * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Child A Keep Adhesion Flags Buffer",
            ),
            child_b_keep_adhesion_flags: DynamicBuffer::new(
                device,
                (max_modes * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Child B Keep Adhesion Flags Buffer",
            ),
            genome_mode_data: DynamicBuffer::new(
                device,
                (max_modes * 48) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                "Dynamic Genome Mode Data Buffer",
            ),
            total_modes: 0,
            max_modes,
        }
    }
    
    /// Update buffers for the given genomes
    /// Returns true if any buffers were resized
    pub fn update_genomes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) -> bool {
        let new_total_modes: usize = genomes.iter().map(|g| g.modes.len()).sum();
        
        if new_total_modes == self.total_modes {
            return false; // No update needed
        }
        
        // Check if we need to resize buffers
        let mut resized = false;
        
        // Calculate required sizes
        let mode_properties_size = (new_total_modes * 48) as u64;
        let mode_cell_types_size = (new_total_modes * 4) as u64;
        let child_mode_indices_size = (new_total_modes * 8) as u64;
        let parent_flags_size = (new_total_modes * 4) as u64;
        let child_flags_size = (new_total_modes * 4) as u64;
        let genome_mode_data_size = (new_total_modes * 48) as u64;
        
        // Resize buffers if needed
        resized |= self.mode_properties.ensure_size(device, queue, mode_properties_size);
        resized |= self.mode_cell_types.ensure_size(device, queue, mode_cell_types_size);
        resized |= self.child_mode_indices.ensure_size(device, queue, child_mode_indices_size);
        resized |= self.parent_make_adhesion_flags.ensure_size(device, queue, parent_flags_size);
        resized |= self.child_a_keep_adhesion_flags.ensure_size(device, queue, child_flags_size);
        resized |= self.child_b_keep_adhesion_flags.ensure_size(device, queue, child_flags_size);
        resized |= self.genome_mode_data.ensure_size(device, queue, genome_mode_data_size);
        
        self.total_modes = new_total_modes;
        resized
    }
    
    /// Get buffer references for binding
    pub fn get_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) {
        (
            self.mode_properties.buffer(),
            self.mode_cell_types.buffer(),
            self.child_mode_indices.buffer(),
            self.parent_make_adhesion_flags.buffer(),
            self.child_a_keep_adhesion_flags.buffer(),
            self.child_b_keep_adhesion_flags.buffer(),
            self.genome_mode_data.buffer(),
        )
    }
    
    /// Get current total modes
    pub fn total_modes(&self) -> usize {
        self.total_modes
    }
    
    /// Get maximum supported modes
    pub fn max_modes(&self) -> usize {
        self.max_modes
    }
    
    /// Check if at capacity
    pub fn at_capacity(&self) -> bool {
        self.total_modes >= self.max_modes
    }
}
