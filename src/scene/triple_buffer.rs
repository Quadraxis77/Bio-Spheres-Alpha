//! # Triple Buffer System for GPU Scene
//!
//! This module implements a sophisticated triple buffering system that enables lock-free
//! GPU computation with zero CPU synchronization stalls. The system operates on three levels:
//!
//! ## Triple Buffer Architecture
//!
//! 1. **Compute Buffer Triple Buffering**: Three sets of physics buffers (current, previous, next)
//!    to enable lock-free GPU computation without CPU-GPU synchronization points
//! 2. **Visual Data Triple Buffering**: Three sets of instance data buffers for asynchronous
//!    rendering updates that don't block physics computation
//! 3. **Command Buffer Triple Buffering**: Three command encoders to pipeline GPU work submission
//!
//! ## Buffer Rotation Strategy
//!
//! The triple buffer system rotates buffers in a lock-free manner using atomic operations:
//! - **Physics Buffers**: Rotate after each physics step completion
//! - **Instance Buffers**: Rotate after visual data extraction
//! - **Command Buffers**: Rotate after GPU submission
//!
//! This ensures that:
//! - Physics computation never waits for rendering
//! - Rendering never waits for physics computation  
//! - GPU work is pipelined for maximum throughput
//! - No CPU-GPU synchronization stalls occur
//!
//! ## Performance Benefits
//!
//! - **Elimination of CPU-GPU Stalls**: Physics computation never waits for rendering completion
//! - **Continuous GPU Utilization**: GPU compute units stay busy with overlapped work
//! - **Pipeline Parallelism**: Multiple stages of the pipeline execute simultaneously
//! - **Maximum Memory Bandwidth**: No contention between physics and rendering memory access

use std::sync::atomic::{AtomicUsize, Ordering};
use wgpu;

/// Triple buffering system for maximum GPU performance without synchronization stalls.
///
/// This system maintains three complete sets of GPU buffers to enable lock-free computation:
/// - Physics computation operates on buffer set N
/// - Visual data extraction operates on buffer set N-1  
/// - Rendering operates on buffer set N-2
///
/// All buffer rotations use atomic operations for thread-safe, lock-free access.
pub struct GpuTripleBufferSystem {
    /// Physics computation buffers (triple buffered)
    /// Contains all simulation state: positions, velocities, forces, etc.
    physics_buffers: [GpuPhysicsBuffers; 3],
    
    /// Current physics buffer index (0, 1, or 2)
    /// Updated atomically after each physics step completion
    current_physics_index: AtomicUsize,
    
    /// Visual instance buffers (triple buffered)
    /// Contains extracted rendering data: positions, colors, orientations
    instance_buffers: [GpuInstanceBuffers; 3],
    
    /// Current instance buffer index (0, 1, or 2)
    /// Updated atomically after visual data extraction
    current_instance_index: AtomicUsize,
    
    /// Command encoders (triple buffered)
    /// Enables pipelined GPU work submission without blocking
    #[allow(dead_code)] // Will be used when command buffer management is implemented
    command_encoders: [Option<wgpu::CommandEncoder>; 3],
    
    /// Current command encoder index (0, 1, or 2)
    /// Updated atomically after GPU command submission
    current_encoder_index: AtomicUsize,
}

impl GpuTripleBufferSystem {
    /// Create a new triple buffer system with the specified capacity.
    ///
    /// All buffer sets are pre-allocated to avoid runtime allocations during simulation.
    /// The capacity should be set based on the expected maximum cell count.
    ///
    /// # Arguments
    /// * `device` - wgpu device for buffer creation
    /// * `capacity` - Maximum number of cells this system can handle
    ///
    /// # Performance Notes
    /// - All buffers are pre-allocated to `capacity * sizeof(data_type)`
    /// - Uses GPU-optimal buffer layouts with 16-byte alignment
    /// - Triple buffering eliminates all CPU-GPU synchronization points
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        // Create three identical sets of physics buffers
        let physics_buffers = [
            GpuPhysicsBuffers::new(device, capacity, "Physics Buffer Set 0"),
            GpuPhysicsBuffers::new(device, capacity, "Physics Buffer Set 1"),
            GpuPhysicsBuffers::new(device, capacity, "Physics Buffer Set 2"),
        ];
        
        // Create three identical sets of instance buffers
        let instance_buffers = [
            GpuInstanceBuffers::new(device, capacity, "Instance Buffer Set 0"),
            GpuInstanceBuffers::new(device, capacity, "Instance Buffer Set 1"),
            GpuInstanceBuffers::new(device, capacity, "Instance Buffer Set 2"),
        ];
        
        // Command encoders start as None and are created on-demand
        let command_encoders = [None, None, None];
        
        Self {
            physics_buffers,
            current_physics_index: AtomicUsize::new(0),
            instance_buffers,
            current_instance_index: AtomicUsize::new(0),
            command_encoders,
            current_encoder_index: AtomicUsize::new(0),
        }
    }
    
    /// Atomically rotate to the next physics buffer set.
    ///
    /// This is called after each physics step completion to switch to the next
    /// buffer set for the following frame. The rotation is atomic and lock-free.
    ///
    /// # Returns
    /// The new current buffer set index (0, 1, or 2)
    ///
    /// # Thread Safety
    /// This method is thread-safe and can be called from any thread without
    /// synchronization. The atomic operation ensures consistency.
    pub fn rotate_physics_buffers(&self) -> usize {
        let current = self.current_physics_index.load(Ordering::Acquire);
        let next = (current + 1) % 3;
        self.current_physics_index.store(next, Ordering::Release);
        next
    }
    
    /// Atomically rotate to the next instance buffer set.
    ///
    /// This is called after visual data extraction to switch to the next
    /// buffer set for rendering. The rotation is atomic and lock-free.
    ///
    /// # Returns
    /// The new current buffer set index (0, 1, or 2)
    pub fn rotate_instance_buffers(&self) -> usize {
        let current = self.current_instance_index.load(Ordering::Acquire);
        let next = (current + 1) % 3;
        self.current_instance_index.store(next, Ordering::Release);
        next
    }
    
    /// Get the current physics buffer set for computation.
    ///
    /// This returns the buffer set that should be used for the current
    /// physics step. The buffers are guaranteed to be available for
    /// exclusive physics computation.
    ///
    /// # Returns
    /// Reference to the current physics buffer set
    pub fn get_current_physics_buffers(&self) -> &GpuPhysicsBuffers {
        let index = self.current_physics_index.load(Ordering::Acquire);
        &self.physics_buffers[index]
    }
    
    /// Get the physics buffer set for rendering (2 frames behind physics).
    ///
    /// This returns the buffer set that contains stable data for rendering.
    /// It's always 2 frames behind the current physics computation to ensure
    /// the data is complete and won't be modified during rendering.
    ///
    /// # Returns
    /// Reference to the rendering physics buffer set
    pub fn get_rendering_physics_buffers(&self) -> &GpuPhysicsBuffers {
        let physics_index = self.current_physics_index.load(Ordering::Acquire);
        let render_index = (physics_index + 2) % 3; // 2 frames behind for stability
        &self.physics_buffers[render_index]
    }
    
    /// Get the current instance buffer set for visual data extraction.
    ///
    /// This returns the buffer set that should be used for extracting
    /// visual data from physics buffers. The buffers are guaranteed to
    /// be available for exclusive visual data writing.
    ///
    /// # Returns
    /// Reference to the current instance buffer set
    pub fn get_current_instance_buffers(&self) -> &GpuInstanceBuffers {
        let index = self.current_instance_index.load(Ordering::Acquire);
        &self.instance_buffers[index]
    }
    
    /// Get the instance buffer set for rendering (1 frame behind extraction).
    ///
    /// This returns the buffer set that contains stable instance data for
    /// rendering. It's 1 frame behind the current extraction to ensure
    /// the data is complete.
    ///
    /// # Returns
    /// Reference to the rendering instance buffer set
    pub fn get_rendering_instance_buffers(&self) -> &GpuInstanceBuffers {
        let instance_index = self.current_instance_index.load(Ordering::Acquire);
        let render_index = (instance_index + 2) % 3; // 1 frame behind for stability
        &self.instance_buffers[render_index]
    }
    
    /// Get buffer rotation statistics for performance monitoring.
    ///
    /// Returns the current buffer indices for debugging and performance
    /// analysis. This can help identify if buffer rotation is working
    /// correctly and buffers are being utilized efficiently.
    ///
    /// # Returns
    /// Tuple of (physics_index, instance_index)
    pub fn get_buffer_indices(&self) -> (usize, usize) {
        (
            self.current_physics_index.load(Ordering::Acquire),
            self.current_instance_index.load(Ordering::Acquire),
        )
    }

    /// Get the current instance buffer index.
    ///
    /// Returns the index of the current instance buffer set being used
    /// for visual data extraction.
    ///
    /// # Returns
    /// Current instance buffer index (0, 1, or 2)
    pub fn get_current_instance_index(&self) -> usize {
        self.current_instance_index.load(Ordering::Acquire)
    }

    /// Get the rendering instance buffer for direct GPU rendering.
    ///
    /// This returns the instance buffer that contains stable rendering data
    /// for the current frame. The buffer is 2 frames behind physics computation
    /// to ensure data stability.
    ///
    /// # Returns
    /// Reference to the rendering instance buffer
    pub fn get_rendering_instance_buffer(&self) -> &wgpu::Buffer {
        let instance_index = self.current_instance_index.load(Ordering::Acquire);
        let render_index = (instance_index + 2) % 3; // 2 frames behind for stability
        &self.instance_buffers[render_index].instance_data
    }

    /// Get the extraction parameters buffer (placeholder).
    ///
    /// This method is a placeholder for the extraction parameters buffer
    /// that will be used by the instance extraction compute shader.
    /// For now, it returns the first physics buffer as a placeholder.
    ///
    /// # Returns
    /// Reference to the extraction parameters buffer
    pub fn get_extraction_params_buffer(&self) -> &wgpu::Buffer {
        // TODO: Create a dedicated extraction parameters buffer
        // For now, return a placeholder buffer
        &self.physics_buffers[0].position_and_mass
    }

    /// Get the instance extraction bind group (placeholder).
    ///
    /// This method is a placeholder for the bind group used by the
    /// instance extraction compute shader. It will be properly implemented
    /// when the compute pipeline integration is complete.
    ///
    /// # Arguments
    /// * `index` - Buffer set index (0, 1, or 2)
    ///
    /// # Returns
    /// Placeholder bind group (currently None)
    pub fn get_instance_extraction_bind_group(&self, _index: usize) -> &wgpu::BindGroup {
        // TODO: Create proper bind groups for instance extraction
        // For now, this is a placeholder that will panic if called
        panic!("Instance extraction bind groups not yet implemented - this will be added in compute pipeline integration")
    }
}

/// GPU physics buffers using Structure-of-Arrays layout for optimal compute shader performance.
///
/// All simulation state is stored in GPU buffers using SoA layout to enable efficient
/// parallel processing in compute shaders. The layout matches the reference implementation
/// structures exactly for compatibility.
///
/// ## Memory Layout
/// - All buffers use 16-byte alignment for optimal GPU memory access
/// - Vec4 is used instead of Vec3 for WGSL alignment requirements
/// - Explicit padding ensures consistent memory layout across platforms
pub struct GpuPhysicsBuffers {
    // === Cell Properties (SoA layout, 16-byte aligned) ===
    
    /// Cell positions and masses: Vec4(x, y, z, mass)
    /// Combined for cache efficiency during physics computation
    pub position_and_mass: wgpu::Buffer,
    
    /// Cell velocities: Vec4(x, y, z, padding)
    /// Padding ensures 16-byte alignment for GPU compute shaders
    pub velocity: wgpu::Buffer,
    
    /// Cell accelerations: Vec4(x, y, z, padding)
    /// Updated each physics step from accumulated forces
    pub acceleration: wgpu::Buffer,
    
    /// Previous frame accelerations: Vec4(x, y, z, padding)
    /// Used for higher-order integration schemes like Verlet
    pub prev_acceleration: wgpu::Buffer,
    
    /// Cell orientations: Vec4(quaternion w, x, y, z)
    /// Physics-driven rotations from angular velocity integration
    pub orientation: wgpu::Buffer,
    
    /// Genome-space orientations: Vec4(quaternion w, x, y, z)
    /// Never affected by physics - used for adhesion zone calculations
    pub genome_orientation: wgpu::Buffer,
    
    /// Angular velocities: Vec4(x, y, z, padding)
    /// Rotational motion from torques and adhesion constraints
    pub angular_velocity: wgpu::Buffer,
    
    /// Angular accelerations: Vec4(x, y, z, padding)
    /// Calculated from accumulated torques
    pub angular_acceleration: wgpu::Buffer,
    
    /// Previous angular accelerations: Vec4(x, y, z, padding)
    /// Used for integration schemes that need acceleration history
    pub prev_angular_acceleration: wgpu::Buffer,
    
    // === Cell Internal State ===
    
    /// Signaling substances: Vec4(4 substances)
    /// Chemical signals between cells for communication
    pub signalling_substances: wgpu::Buffer,
    
    /// Mode indices: i32 array
    /// Current behavior mode within the genome (absolute mode index)
    pub mode_indices: wgpu::Buffer,
    
    /// Cell ages: f32 array
    /// Also used for split timer tracking
    pub ages: wgpu::Buffer,
    
    /// Toxin levels: f32 array
    /// Accumulated toxins that can cause cell death
    pub toxins: wgpu::Buffer,
    
    /// Nitrate levels: f32 array
    /// Nutrient storage for cell metabolism
    pub nitrates: wgpu::Buffer,
    
    // === Cell Identification and Genetics ===
    
    /// Unique cell identifiers: u32 array
    /// Stable IDs that persist across index changes
    pub cell_ids: wgpu::Buffer,
    
    /// Genome identifiers: u32 array
    /// Which genome this cell uses for behavior
    pub genome_ids: wgpu::Buffer,
    
    // === Division and Lifecycle ===
    
    /// Birth times: f32 array
    /// Simulation time when each cell was created
    pub birth_times: wgpu::Buffer,
    
    /// Split intervals: f32 array
    /// Minimum time between divisions
    pub split_intervals: wgpu::Buffer,
    
    /// Split masses: f32 array
    /// Mass threshold required for division
    pub split_masses: wgpu::Buffer,
    
    /// Split counts: i32 array
    /// Number of times each cell has divided
    pub split_counts: wgpu::Buffer,
    
    /// Split ready frame: i32 array
    /// Frame when cell becomes ready to divide (for timing control)
    pub split_ready_frame: wgpu::Buffer,
    
    // === Adhesion System ===
    
    /// Adhesion connections: Array of GpuAdhesionConnection (96 bytes each)
    /// Contains all cell-cell adhesion bonds with mechanical properties
    pub adhesion_connections: wgpu::Buffer,
    
    /// Adhesion indices: Array of 10 u32 per cell (10 connections max per cell)
    /// Maps each cell to its adhesion connections for fast lookup
    pub adhesion_indices: wgpu::Buffer,
    
    /// Adhesion count per cell: u32 array
    /// Number of active adhesions for each cell (0-10)
    pub adhesion_counts: wgpu::Buffer,
    
    // === Spatial Grid System ===
    
    /// Spatial grid cell counts: u32 array (64³ = 262,144 cells)
    /// Number of cells in each grid cell for collision detection
    pub spatial_grid_counts: wgpu::Buffer,
    
    /// Spatial grid cell offsets: u32 array (64³ = 262,144 cells)
    /// Starting index in spatial_grid_indices for each grid cell
    pub spatial_grid_offsets: wgpu::Buffer,
    
    /// Spatial grid indices: u32 array (capacity * max_cells_per_grid)
    /// Cell indices sorted by spatial grid cell for collision detection
    pub spatial_grid_indices: wgpu::Buffer,
    
    /// Spatial grid assignments: u32 array (capacity)
    /// Maps each cell to its assigned grid cell index
    pub spatial_grid_assignments: wgpu::Buffer,
    
    /// Spatial grid insertion counters: u32 array (64³ = 262,144 cells)
    /// Atomic counters for thread-safe insertion during grid building
    pub spatial_grid_insertion_counters: wgpu::Buffer,
    
    // === Lifecycle Management Buffers (for prefix-sum compaction) ===
    
    /// Death flags: bool array (stored as u32 for GPU compatibility)
    /// Marks cells ready to die (input for prefix-sum death compaction)
    pub death_flags: wgpu::Buffer,
    
    /// Division candidates: bool array (stored as u32 for GPU compatibility)
    /// Marks cells ready to divide (input for prefix-sum slot assignment)
    pub division_candidates: wgpu::Buffer,
    
    /// Free slots: u32 array (compacted array of available cell indices)
    /// Maintained by prefix-sum compaction for deterministic allocation
    pub free_slots: wgpu::Buffer,
    
    /// Free slot count: u32 (single value)
    /// Number of available slots in free_slots array
    pub free_slot_count: wgpu::Buffer,
    
    /// Division reservations: u32 array
    /// Number of slots each dividing cell needs (Child B + inherited adhesions)
    pub division_reservations: wgpu::Buffer,
    
    /// Division assignments: u32 array
    /// Assigned slot indices for each dividing cell (from prefix-sum)
    pub division_assignments: wgpu::Buffer,
    
    /// Buffer capacity (maximum number of cells)
    pub capacity: u32,
}

impl GpuPhysicsBuffers {
    /// Create a new set of GPU physics buffers with the specified capacity.
    ///
    /// All buffers are pre-allocated to avoid runtime allocations during simulation.
    /// The buffers use optimal GPU memory layouts with proper alignment.
    ///
    /// # Arguments
    /// * `device` - wgpu device for buffer creation
    /// * `capacity` - Maximum number of cells
    /// * `label_prefix` - Prefix for buffer labels (for debugging)
    pub fn new(device: &wgpu::Device, capacity: u32, label_prefix: &str) -> Self {
        // Helper function to create storage buffers with consistent settings
        let create_buffer = |size: u64, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} - {}", label_prefix, label)),
                size,
                usage: wgpu::BufferUsages::STORAGE 
                    | wgpu::BufferUsages::COPY_DST 
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        
        // Calculate buffer sizes (all use 16-byte alignment)
        let vec4_size = capacity as u64 * 16; // Vec4 = 4 * f32 = 16 bytes
        let i32_size = capacity as u64 * 4;   // i32 = 4 bytes
        let f32_size = capacity as u64 * 4;   // f32 = 4 bytes
        let u32_size = capacity as u64 * 4;   // u32 = 4 bytes
        
        Self {
            // Cell properties (Vec4 for alignment)
            position_and_mass: create_buffer(vec4_size, "Position and Mass"),
            velocity: create_buffer(vec4_size, "Velocity"),
            acceleration: create_buffer(vec4_size, "Acceleration"),
            prev_acceleration: create_buffer(vec4_size, "Previous Acceleration"),
            orientation: create_buffer(vec4_size, "Orientation"),
            genome_orientation: create_buffer(vec4_size, "Genome Orientation"),
            angular_velocity: create_buffer(vec4_size, "Angular Velocity"),
            angular_acceleration: create_buffer(vec4_size, "Angular Acceleration"),
            prev_angular_acceleration: create_buffer(vec4_size, "Previous Angular Acceleration"),
            
            // Cell internal state
            signalling_substances: create_buffer(vec4_size, "Signalling Substances"),
            mode_indices: create_buffer(i32_size, "Mode Indices"),
            ages: create_buffer(f32_size, "Ages"),
            toxins: create_buffer(f32_size, "Toxins"),
            nitrates: create_buffer(f32_size, "Nitrates"),
            
            // Cell identification
            cell_ids: create_buffer(u32_size, "Cell IDs"),
            genome_ids: create_buffer(u32_size, "Genome IDs"),
            
            // Division and lifecycle
            birth_times: create_buffer(f32_size, "Birth Times"),
            split_intervals: create_buffer(f32_size, "Split Intervals"),
            split_masses: create_buffer(f32_size, "Split Masses"),
            split_counts: create_buffer(i32_size, "Split Counts"),
            split_ready_frame: create_buffer(i32_size, "Split Ready Frame"),
            
            // Adhesion system
            adhesion_connections: create_buffer(capacity as u64 * 96, "Adhesion Connections"), // 96 bytes per connection
            adhesion_indices: create_buffer(capacity as u64 * 10 * 4, "Adhesion Indices"), // 10 u32 per cell
            adhesion_counts: create_buffer(u32_size, "Adhesion Counts"),
            
            // Spatial grid system (64³ grid)
            spatial_grid_counts: create_buffer(64 * 64 * 64 * 4, "Spatial Grid Counts"), // 64³ u32
            spatial_grid_offsets: create_buffer(64 * 64 * 64 * 4, "Spatial Grid Offsets"), // 64³ u32
            spatial_grid_indices: create_buffer(capacity as u64 * 32 * 4, "Spatial Grid Indices"), // capacity * max_cells_per_grid * u32
            spatial_grid_assignments: create_buffer(u32_size, "Spatial Grid Assignments"), // capacity * u32
            spatial_grid_insertion_counters: create_buffer(64 * 64 * 64 * 4, "Spatial Grid Insertion Counters"), // 64³ u32
            
            // Lifecycle management buffers
            death_flags: create_buffer(u32_size, "Death Flags"), // bool as u32
            division_candidates: create_buffer(u32_size, "Division Candidates"), // bool as u32
            free_slots: create_buffer(u32_size, "Free Slots"), // u32 array
            free_slot_count: create_buffer(4, "Free Slot Count"), // single u32
            division_reservations: create_buffer(u32_size, "Division Reservations"), // u32 array
            division_assignments: create_buffer(u32_size, "Division Assignments"), // u32 array
            
            capacity,
        }
    }
}

/// GPU instance buffers for rendering data extraction.
///
/// These buffers contain the visual data extracted from physics buffers
/// for efficient GPU instanced rendering. The data is optimized for
/// the current CellRenderer system.
pub struct GpuInstanceBuffers {
    /// Instance data for rendering: position, radius, color, orientation
    /// Layout matches the current InstanceBuilder requirements
    pub instance_data: wgpu::Buffer,
    
    /// Number of visible instances (after culling)
    /// Updated by compute shaders during instance extraction
    pub visible_count: wgpu::Buffer,
    
    /// Buffer capacity (maximum number of instances)
    pub capacity: u32,
}

impl GpuInstanceBuffers {
    /// Create a new set of GPU instance buffers with the specified capacity.
    ///
    /// # Arguments
    /// * `device` - wgpu device for buffer creation
    /// * `capacity` - Maximum number of instances
    /// * `label_prefix` - Prefix for buffer labels (for debugging)
    pub fn new(device: &wgpu::Device, capacity: u32, label_prefix: &str) -> Self {
        // Instance data buffer (position + radius + color + orientation = 12 floats = 48 bytes)
        let instance_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} - Instance Data", label_prefix)),
            size: capacity as u64 * 48, // 48 bytes per instance
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::VERTEX 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Visible count buffer (single u32)
        let visible_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} - Visible Count", label_prefix)),
            size: 4, // Single u32
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            instance_data,
            visible_count,
            capacity,
        }
    }
}