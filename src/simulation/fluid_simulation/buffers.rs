//! GPU buffer management for fluid simulation
//!
//! Implements triple-buffered fluid state using Structure-of-Arrays layout
//! for optimal GPU memory access patterns.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{self, util::DeviceExt};

use super::{GRID_RESOLUTION, TOTAL_VOXELS};

/// Fluid type enumeration
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidType {
    Empty = 0,
    Water = 1,
    Lava = 2,
    Steam = 3,
    Solid = 4,
    Nutrients = 5,
}

/// Voxel state (type + fill fraction)
/// Packed into 8 bytes for efficient storage
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VoxelState {
    pub fluid_type: u32,
    pub fill_fraction: f32,
}

/// Grid statistics for debugging
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GridStats {
    pub solid_count: u32,
    pub empty_count: u32,
    pub water_count: u32,
    pub lava_count: u32,
    pub steam_count: u32,
    pub nutrients_count: u32,
    pub total_water_mass: f32,
    pub total_lava_mass: f32,
    pub total_steam_mass: f32,
    pub total_nutrients_mass: f32,
    pub _padding: [u32; 6], // Pad to 64 bytes
}

/// Fluid simulation parameters (256-byte aligned for uniform buffer)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FluidParams {
    // Grid configuration
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub world_radius: f32,
    pub _padding0: f32,
    
    pub world_center: [f32; 3],
    pub _padding1: f32,
    
    // Physics constants
    pub gravity: f32,
    pub water_density: f32,
    pub lava_density: f32,
    pub steam_density: f32,
    
    pub gravity_dir: [f32; 3],
    pub _padding2: f32,
    
    // Simulation parameters
    pub dt: f32,
    pub vorticity_epsilon: f32,
    pub sor_omega: f32,
    pub pressure_iterations: u32,
    
    // Surface adhesion (water only)
    pub water_adhesion_strength: f32,
    pub droplet_threshold: f32,
    pub droplet_detach_force: f32,
    pub _padding3: f32,
    
    // Debug visualization flags
    pub show_voxel_grid: u32,
    pub show_solid_only: u32,
    pub color_mode: u32,
    pub show_wireframe: u32,
    
    // Padding to 256 bytes
    pub _padding4: [f32; 36],
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            grid_resolution: GRID_RESOLUTION,
            cell_size: 0.0, // Computed from world diameter
            world_radius: 0.0,
            _padding0: 0.0,
            
            world_center: [0.0, 0.0, 0.0],
            _padding1: 0.0,
            
            gravity: 9.8,
            water_density: 1.0,
            lava_density: 3.0,
            steam_density: 0.1,
            
            gravity_dir: [0.0, -1.0, 0.0],
            _padding2: 0.0,
            
            dt: 1.0 / 30.0, // 30 FPS
            vorticity_epsilon: 0.05,
            sor_omega: 1.9,
            pressure_iterations: 10,
            
            water_adhesion_strength: 0.5,
            droplet_threshold: 1.0,
            droplet_detach_force: 2.0,
            _padding3: 0.0,
            
            show_voxel_grid: 1,
            show_solid_only: 0,
            color_mode: 0,
            show_wireframe: 0,
            
            _padding4: [0.0; 36],
        }
    }
}

/// Triple-buffered GPU fluid simulation state
///
/// Uses Structure-of-Arrays layout for optimal GPU memory access.
/// Triple buffering enables lock-free computation:
/// - Physics operates on buffer set N
/// - Extraction operates on buffer set N-1
/// - Rendering operates on buffer set N-2
pub struct FluidBuffers {
    // Triple-buffered voxel state (type + fill_fraction)
    voxel_state: [wgpu::Buffer; 3],
    
    // Static solid mask (cave + sphere boundary)
    solid_mask: wgpu::Buffer,
    
    // Grid statistics
    grid_stats: wgpu::Buffer,
    
    // Uniform parameters
    params: wgpu::Buffer,
    
    // Current buffer index for physics
    current_index: usize,
    
    // Total memory usage in bytes
    memory_usage: u64,
}

impl FluidBuffers {
    /// Create new fluid buffers with specified world parameters
    pub fn new(device: &wgpu::Device, world_radius: f32, world_center: Vec3) -> Self {
        // Calculate cell size from world diameter
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        
        // Create parameters
        let mut params = FluidParams::default();
        params.world_radius = world_radius;
        params.world_center = world_center.to_array();
        params.cell_size = cell_size;
        
        // Calculate buffer sizes
        let voxel_state_size = (TOTAL_VOXELS * std::mem::size_of::<VoxelState>()) as u64;
        let solid_mask_size = (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64;
        let stats_size = std::mem::size_of::<GridStats>() as u64;
        let params_size = std::mem::size_of::<FluidParams>() as u64;
        
        // Create triple-buffered voxel state
        let voxel_state = [
            Self::create_storage_buffer(device, voxel_state_size, "Voxel State Buffer 0"),
            Self::create_storage_buffer(device, voxel_state_size, "Voxel State Buffer 1"),
            Self::create_storage_buffer(device, voxel_state_size, "Voxel State Buffer 2"),
        ];
        
        // Create static solid mask
        let solid_mask = Self::create_storage_buffer(device, solid_mask_size, "Solid Mask Buffer");
        
        // Create grid statistics buffer
        let grid_stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Stats Buffer"),
            size: stats_size,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create uniform parameters buffer
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Calculate total memory usage
        let memory_usage = (voxel_state_size * 3) + solid_mask_size + stats_size + params_size;
        
        Self {
            voxel_state,
            solid_mask,
            grid_stats,
            params: params_buffer,
            current_index: 0,
            memory_usage,
        }
    }
    
    /// Create a storage buffer with specified size
    fn create_storage_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
    
    /// Get current voxel state buffer for physics
    pub fn current_voxel_state(&self) -> &wgpu::Buffer {
        &self.voxel_state[self.current_index]
    }
    
    /// Get next voxel state buffer for physics output
    pub fn next_voxel_state(&self) -> &wgpu::Buffer {
        let next_index = (self.current_index + 1) % 3;
        &self.voxel_state[next_index]
    }
    
    /// Get rendering voxel state buffer (N-2)
    pub fn rendering_voxel_state(&self) -> &wgpu::Buffer {
        let render_index = (self.current_index + 2) % 3;
        &self.voxel_state[render_index]
    }
    
    /// Get solid mask buffer
    pub fn solid_mask(&self) -> &wgpu::Buffer {
        &self.solid_mask
    }
    
    /// Get grid statistics buffer
    pub fn grid_stats(&self) -> &wgpu::Buffer {
        &self.grid_stats
    }
    
    /// Get parameters buffer
    pub fn params(&self) -> &wgpu::Buffer {
        &self.params
    }
    
    /// Rotate buffers (advance to next frame)
    pub fn rotate_buffers(&mut self) {
        self.current_index = (self.current_index + 1) % 3;
    }
    
    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        self.memory_usage
    }
    
    /// Get total memory usage in megabytes
    pub fn memory_usage_mb(&self) -> f32 {
        self.memory_usage as f32 / (1024.0 * 1024.0)
    }
    
    /// Validate memory budget (< 300 MB)
    pub fn validate_memory_budget(&self) -> Result<(), String> {
        const MAX_MEMORY_MB: f32 = 300.0;
        let usage_mb = self.memory_usage_mb();
        
        if usage_mb > MAX_MEMORY_MB {
            Err(format!(
                "Fluid system memory usage ({:.2} MB) exceeds budget ({:.2} MB)",
                usage_mb, MAX_MEMORY_MB
            ))
        } else {
            Ok(())
        }
    }
    
    /// Update parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &FluidParams) {
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[*params]));
    }
    
    /// Update solid mask data
    pub fn update_solid_mask(&self, queue: &wgpu::Queue, solid_mask: &[u32]) {
        queue.write_buffer(&self.solid_mask, 0, bytemuck::cast_slice(solid_mask));
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grid_constants() {
        assert_eq!(GRID_RESOLUTION, 128);
        assert_eq!(TOTAL_VOXELS, 128 * 128 * 128);
        assert_eq!(TOTAL_VOXELS, 2_097_152);
    }
    
    #[test]
    fn test_voxel_state_size() {
        // Ensure VoxelState is 8 bytes (u32 + f32)
        assert_eq!(std::mem::size_of::<VoxelState>(), 8);
    }
    
    #[test]
    fn test_fluid_params_size() {
        // Ensure FluidParams is 256 bytes (uniform buffer alignment)
        assert_eq!(std::mem::size_of::<FluidParams>(), 256);
    }
    
    #[test]
    fn test_grid_stats_size() {
        // Ensure GridStats is 64 bytes
        assert_eq!(std::mem::size_of::<GridStats>(), 64);
    }
    
    #[test]
    fn test_fluid_params_default() {
        let params = FluidParams::default();
        assert_eq!(params.grid_resolution, 128);
        assert_eq!(params.gravity, 9.8);
        assert_eq!(params.water_density, 1.0);
        assert_eq!(params.lava_density, 3.0);
        assert_eq!(params.steam_density, 0.1);
        assert_eq!(params.gravity_dir, [0.0, -1.0, 0.0]);
    }
    
    #[test]
    fn test_memory_budget_calculation() {
        // Calculate expected memory usage
        let voxel_state_size = TOTAL_VOXELS * std::mem::size_of::<VoxelState>();
        let solid_mask_size = TOTAL_VOXELS * std::mem::size_of::<u32>();
        let stats_size = std::mem::size_of::<GridStats>();
        let params_size = std::mem::size_of::<FluidParams>();
        
        let total = (voxel_state_size * 3) + solid_mask_size + stats_size + params_size;
        let total_mb = total as f32 / (1024.0 * 1024.0);
        
        // Should be well under 300 MB budget
        assert!(total_mb < 300.0, "Memory usage {:.2} MB exceeds budget", total_mb);
        
        // Expected: ~58 MB for triple-buffered state + solid mask
        println!("Fluid system memory usage: {:.2} MB", total_mb);
    }
    
    #[test]
    fn test_cell_size_calculation() {
        let world_radius = 100.0;
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        
        assert_eq!(cell_size, 200.0 / 128.0);
        assert!((cell_size - 1.5625).abs() < 0.0001);
    }
}
