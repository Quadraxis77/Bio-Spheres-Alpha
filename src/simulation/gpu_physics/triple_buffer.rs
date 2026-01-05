//! Triple Buffer System for GPU Physics
//! 
//! Manages three complete sets of simulation buffers to enable lock-free
//! GPU computation with zero CPU synchronization stalls.
//! 
//! ## Buffer Layout
//! 
//! ### Cell Data (Triple Buffered)
//! - `position_and_mass[3]`: Vec4(x, y, z, mass) per cell
//! - `velocity[3]`: Vec4(x, y, z, 0) per cell
//! 
//! ### Spatial Grid (64³ = 262,144 grid cells)
//! - `spatial_grid_counts`: u32 per grid cell (how many cells in each grid cell)
//! - `spatial_grid_offsets`: u32 per grid cell (prefix-sum for O(1) lookup)
//! - `spatial_grid_cells`: u32 per cell (sorted cell indices by grid cell)
//! - `cell_grid_indices`: u32 per cell (which grid cell each cell belongs to)
//! 
//! ### Uniforms
//! - `physics_params`: 256-byte aligned PhysicsParams struct
//! 
//! ## Synchronization
//! - `needs_sync`: Flag set when CPU data changes (cell insertion/removal)
//! - `sync_from_canonical_state()`: Uploads CPU data to all 3 buffer sets
//! - `output_buffer_index()`: Returns buffer with latest physics results

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::simulation::CanonicalState;

/// Triple-buffered GPU simulation state using Structure-of-Arrays layout
pub struct GpuTripleBufferSystem {
    /// Cell positions and masses: Vec4(x, y, z, mass) - triple buffered
    pub position_and_mass: [wgpu::Buffer; 3],
    
    /// Cell velocities: Vec4(x, y, z, padding) - triple buffered  
    pub velocity: [wgpu::Buffer; 3],
    
    /// Spatial grid cell counts (64³ = 262,144 grid cells)
    /// spatial_grid_counts[grid_idx] = number of cells in that grid cell
    pub spatial_grid_counts: wgpu::Buffer,
    
    /// Spatial grid offsets (prefix sum results)
    /// spatial_grid_offsets[grid_idx] = starting index in spatial_grid_cells
    pub spatial_grid_offsets: wgpu::Buffer,
    
    /// Sorted cell indices by grid cell
    /// spatial_grid_cells[offset..offset+count] = cell indices in grid cell
    pub spatial_grid_cells: wgpu::Buffer,
    
    /// Per-cell grid indices
    /// cell_grid_indices[cell_idx] = which grid cell this cell belongs to
    pub cell_grid_indices: wgpu::Buffer,
    
    /// Physics parameters uniform buffer
    pub physics_params: wgpu::Buffer,
    
    /// Current buffer index (atomic for lock-free rotation)
    current_index: AtomicUsize,
    
    /// Cell capacity
    pub capacity: u32,
    
    /// Whether buffers need full sync from canonical state (initial setup only)
    pub needs_sync: bool,
    
    /// Pending cell insertions (cell indices to sync)
    pending_cell_insertions: Vec<usize>,
}

impl GpuTripleBufferSystem {
    /// Create a new triple buffer system
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let buffer_size = capacity as u64 * 16; // Vec4<f32> = 16 bytes
        
        // Create triple-buffered simulation data
        let position_and_mass = [
            Self::create_storage_buffer(device, buffer_size, "Position Mass Buffer 0"),
            Self::create_storage_buffer(device, buffer_size, "Position Mass Buffer 1"), 
            Self::create_storage_buffer(device, buffer_size, "Position Mass Buffer 2"),
        ];
        
        let velocity = [
            Self::create_storage_buffer(device, buffer_size, "Velocity Buffer 0"),
            Self::create_storage_buffer(device, buffer_size, "Velocity Buffer 1"),
            Self::create_storage_buffer(device, buffer_size, "Velocity Buffer 2"),
        ];
        
        // Create spatial grid buffers (64³ = 262,144 grid cells)
        let grid_size = 64 * 64 * 64;
        let spatial_grid_counts = Self::create_storage_buffer(
            device, 
            grid_size * 4, // u32 = 4 bytes
            "Spatial Grid Counts"
        );
        
        let spatial_grid_offsets = Self::create_storage_buffer(
            device,
            grid_size * 4, // u32 = 4 bytes  
            "Spatial Grid Offsets"
        );
        
        let cell_grid_indices = Self::create_storage_buffer(
            device,
            capacity as u64 * 4, // u32 per cell
            "Cell Grid Indices"
        );
        
        // Sorted cell indices by grid cell (16 cells max per grid cell * 64³ grid cells)
        // This enables O(1) neighbor lookup in collision detection
        let spatial_grid_cells = Self::create_storage_buffer(
            device,
            16 * grid_size * 4, // 16 cells per grid cell * 262,144 grid cells * 4 bytes
            "Spatial Grid Cells"
        );
        
        // Physics params uniform buffer (256 bytes aligned)
        let physics_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Physics Params Uniform"),
            size: 256, // Aligned to 256 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            position_and_mass,
            velocity,
            spatial_grid_counts,
            spatial_grid_offsets,
            spatial_grid_cells,
            cell_grid_indices,
            physics_params,
            current_index: AtomicUsize::new(0),
            capacity,
            needs_sync: true,
            pending_cell_insertions: Vec::new(),
        }
    }
    
    /// Create a storage buffer with optimal settings for compute shaders
    fn create_storage_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    
    /// Upload canonical state data to all GPU buffer sets
    /// WARNING: This overwrites ALL GPU positions with CPU data.
    /// Only use for initial setup, not during simulation.
    pub fn sync_from_canonical_state(&mut self, queue: &wgpu::Queue, state: &CanonicalState) {
        if state.cell_count == 0 {
            self.needs_sync = false;
            return;
        }
        
        // Build position_and_mass data: Vec4(x, y, z, mass)
        let mut position_mass_data: Vec<[f32; 4]> = Vec::with_capacity(state.cell_count);
        for i in 0..state.cell_count {
            position_mass_data.push([
                state.positions[i].x,
                state.positions[i].y,
                state.positions[i].z,
                state.masses[i],
            ]);
        }
        
        // Build velocity data: Vec4(x, y, z, 0)
        let mut velocity_data: Vec<[f32; 4]> = Vec::with_capacity(state.cell_count);
        for i in 0..state.cell_count {
            velocity_data.push([
                state.velocities[i].x,
                state.velocities[i].y,
                state.velocities[i].z,
                0.0,
            ]);
        }
        
        // Upload to all three buffer sets
        let position_bytes = bytemuck::cast_slice(&position_mass_data);
        let velocity_bytes = bytemuck::cast_slice(&velocity_data);
        
        for i in 0..3 {
            queue.write_buffer(&self.position_and_mass[i], 0, position_bytes);
            queue.write_buffer(&self.velocity[i], 0, velocity_bytes);
        }
        
        self.needs_sync = false;
    }
    
    /// Sync a single cell to all GPU buffer sets (for cell insertion during simulation)
    pub fn sync_single_cell(&self, queue: &wgpu::Queue, cell_idx: usize, position: glam::Vec3, velocity: glam::Vec3, mass: f32) {
        let position_mass: [f32; 4] = [position.x, position.y, position.z, mass];
        let velocity_data: [f32; 4] = [velocity.x, velocity.y, velocity.z, 0.0];
        
        let offset = (cell_idx * 16) as u64; // Vec4<f32> = 16 bytes
        
        // Upload to all three buffer sets
        for i in 0..3 {
            queue.write_buffer(&self.position_and_mass[i], offset, bytemuck::bytes_of(&position_mass));
            queue.write_buffer(&self.velocity[i], offset, bytemuck::bytes_of(&velocity_data));
        }
    }
    
    /// Mark that buffers need to be synced from canonical state
    pub fn mark_needs_sync(&mut self) {
        self.needs_sync = true;
    }
    
    /// Queue a cell insertion to be synced on next physics step
    pub fn queue_cell_insertion(&mut self, cell_idx: usize) {
        self.pending_cell_insertions.push(cell_idx);
    }
    
    /// Sync pending cell insertions to GPU buffers
    pub fn sync_pending_insertions(&mut self, queue: &wgpu::Queue, state: &crate::simulation::CanonicalState) {
        for &cell_idx in &self.pending_cell_insertions {
            if cell_idx < state.cell_count {
                let position = state.positions[cell_idx];
                let velocity = state.velocities[cell_idx];
                let mass = state.masses[cell_idx];
                self.sync_single_cell(queue, cell_idx, position, velocity, mass);
            }
        }
        self.pending_cell_insertions.clear();
    }
    
    /// Rotate to the next buffer set (lock-free)
    pub fn rotate_buffers(&self) -> usize {
        let current = self.current_index.load(Ordering::Acquire);
        let next = (current + 1) % 3;
        self.current_index.store(next, Ordering::Release);
        next
    }
    
    /// Get the current buffer index
    pub fn current_index(&self) -> usize {
        self.current_index.load(Ordering::Acquire)
    }
    
    /// Get the output buffer index (where physics results are written)
    /// This is (current_index + 1) % 3 since physics reads from current and writes to next
    pub fn output_buffer_index(&self) -> usize {
        let current = self.current_index.load(Ordering::Acquire);
        (current + 1) % 3
    }
}