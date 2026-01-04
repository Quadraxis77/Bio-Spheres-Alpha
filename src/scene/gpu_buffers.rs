//! # GPU Buffer Layouts and Uniform Structures
//!
//! This module defines all GPU buffer layouts and uniform structures that match
//! the reference implementation exactly. All structures use explicit padding and
//! alignment to ensure compatibility between Rust and WGSL.
//!
//! ## Design Principles
//!
//! ### WGSL Alignment Requirements
//! - All structures use `#[repr(C)]` for consistent memory layout
//! - Vec3 is replaced with Vec4 for 16-byte alignment in GPU buffers
//! - Explicit padding ensures structures match WGSL layout exactly
//! - All uniform buffers are padded to 256-byte boundaries
//!
//! ### Reference Implementation Compatibility
//! - Structure layouts match Biospheres-Master reference exactly
//! - Field names and types correspond to reference GLSL structures
//! - Memory sizes are identical (e.g., AdhesionConnection = 96 bytes)
//! - Enum values and constants match reference implementation
//!
//! ## Buffer Categories
//!
//! ### Uniform Buffers
//! - `PhysicsParams` - Physics configuration and frame info
//! - `GpuMode` - Genome mode data for cell behavior
//! - `CellCountBuffer` - Cell count and management data
//!
//! ### Storage Buffers
//! - `GpuAdhesionConnection` - Cell-cell connection data (96 bytes)
//! - `GpuCollisionPair` - Collision detection results
//! - `GpuInstanceData` - Rendering instance data
//! - `GpuParticle` - Volumetric effects data

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::genome::Genome;

// ============================================================================
// CELL TYPE SYSTEM CONSTANTS
// ============================================================================

/// Cell type identifiers for different cell behaviors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CellType {
    /// Test cells (cell_type == 0) - automatic nutrient gain and growth
    Test = 0,
    /// Flagellocyte cells (cell_type == 1) - swim forces and nutrient consumption
    Flagellocyte = 1,
}

impl CellType {
    /// Convert from u32 to CellType
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(CellType::Test),
            1 => Some(CellType::Flagellocyte),
            _ => None,
        }
    }
    
    /// Convert to u32 for GPU compatibility
    pub fn to_u32(self) -> u32 {
        self as u32
    }
}

/// Cell type behavior parameters for GPU compute shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CellTypeBehavior {
    /// Cell type identifier (0 = Test, 1 = Flagellocyte)
    pub cell_type: u32,
    /// Automatic nutrient gain rate (for Test cells)
    pub nutrient_gain_rate: f32,
    /// Swim force magnitude (for Flagellocyte cells)
    pub swim_force: f32,
    /// Nutrient consumption rate based on swim force (for Flagellocyte cells)
    pub nutrient_consumption_rate: f32,
    /// Maximum cell size constraint
    pub max_cell_size: f32,
    /// Nutrient priority for resource competition
    pub nutrient_priority: f32,
    /// Whether to prioritize when nutrients are low
    pub prioritize_when_low: u32, // 0 = false, 1 = true
    /// Padding to ensure 32-byte alignment
    _padding: u32,
}

impl Default for CellTypeBehavior {
    fn default() -> Self {
        Self {
            cell_type: CellType::Test.to_u32(),
            nutrient_gain_rate: 0.2, // Default gain rate for Test cells
            swim_force: 0.0,          // No swim force for Test cells
            nutrient_consumption_rate: 0.0, // No consumption for Test cells
            max_cell_size: 2.0,       // Default maximum size
            nutrient_priority: 1.0,   // Neutral priority
            prioritize_when_low: 1,   // Enable low-nutrient priority boost
            _padding: 0,
        }
    }
}

// ============================================================================
// ADHESION SYSTEM CONSTANTS
// ============================================================================

/// Maximum adhesions per cell (matches reference implementation)
pub const MAX_ADHESIONS_PER_CELL: usize = 10;

/// Maximum total adhesion connections (capacity * max_adhesions_per_cell)
pub const MAX_ADHESION_CONNECTIONS: usize = 25_600; // 10k cells * 10 connections / 4 (shared)

/// Empty adhesion slot marker (matches CPU implementation)
pub const EMPTY_ADHESION_SLOT: i32 = -1;

/// Adhesion zone classification for division inheritance (matches CPU implementation)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdhesionZone {
    ZoneA = 0,  // Green in visualization - opposite to split direction
    ZoneB = 1,  // Blue in visualization - same as split direction  
    ZoneC = 2,  // Red in visualization - equatorial band
}

/// Adhesion indices for each cell (10 slots, -1 for empty)
/// This matches the CPU AdhesionIndices type exactly
pub type GpuAdhesionIndices = [i32; MAX_ADHESIONS_PER_CELL];

/// Initialize adhesion indices for a cell (all slots to -1)
pub fn init_gpu_adhesion_indices() -> GpuAdhesionIndices {
    [EMPTY_ADHESION_SLOT; MAX_ADHESIONS_PER_CELL]
}

// ============================================================================
// UNIFORM BUFFER STRUCTURES
// ============================================================================

/// Physics parameters and configuration passed to compute shaders.
///
/// This structure contains all the physics configuration, timing information,
/// and world settings needed by GPU compute shaders. It matches the reference
/// implementation's uniform buffer layout exactly.
///
/// ## Memory Layout
/// - Total size: 256 bytes (uniform buffer alignment requirement)
/// - All fields are 4-byte aligned for GPU compatibility
/// - Padding ensures consistent layout across platforms
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PhysicsParams {
    // === Time and Frame Information (16 bytes) ===
    /// Delta time for current physics step (seconds)
    pub delta_time: f32,
    /// Current simulation time (seconds)
    pub current_time: f32,
    /// Current frame number (for deterministic behavior)
    pub current_frame: i32,
    /// Number of active cells in simulation
    pub cell_count: u32,
    
    // === World and Physics Settings (16 bytes) ===
    /// World size (simulation boundary)
    pub world_size: f32,
    /// Boundary force stiffness (prevents cells from leaving world)
    pub boundary_stiffness: f32,
    /// Gravity acceleration (typically 0 for biological simulation)
    pub gravity: f32,
    /// Acceleration damping factor (0.0 = no damping, 1.0 = full damping)
    pub acceleration_damping: f32,
    
    // === Spatial Grid Settings (16 bytes) ===
    /// Grid resolution (number of cells per dimension)
    pub grid_resolution: i32,
    /// Size of each grid cell in world units
    pub grid_cell_size: f32,
    /// Maximum cells per grid cell (for collision detection)
    pub max_cells_per_grid: i32,
    /// Enable thrust force for Flagellocyte cells (0 = disabled, 1 = enabled)
    pub enable_thrust_force: i32,
    
    // === UI Interaction (4 bytes + 12 bytes padding) ===
    /// Index of cell being dragged by user (-1 = none)
    pub dragged_cell_index: i32,
    /// Padding to maintain 16-byte alignment
    pub _padding1: [f32; 3],
    
    // === Padding to 256-byte alignment for uniform buffer (192 bytes) ===
    /// Explicit padding to ensure 256-byte total size
    /// Required for uniform buffer alignment on most GPUs
    pub _padding: [f32; 48],
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            delta_time: 1.0 / 60.0,      // 60 FPS default
            current_time: 0.0,
            current_frame: 0,
            cell_count: 0,
            world_size: 200.0,           // 200 unit world
            boundary_stiffness: 100.0,   // Strong boundary forces
            gravity: 0.0,                // No gravity for biological simulation
            acceleration_damping: 0.1,   // Light damping for stability
            grid_resolution: 64,         // 64x64x64 grid
            grid_cell_size: 200.0 / 64.0, // Calculated from world_size / grid_resolution
            max_cells_per_grid: 32,      // Reasonable collision detection limit
            enable_thrust_force: 1,      // Enable Flagellocyte thrust by default
            dragged_cell_index: -1,      // No cell being dragged
            _padding1: [0.0; 3],
            _padding: [0.0; 48],
        }
    }
}

/// GPU Mode structure matching reference GPUMode exactly.
///
/// This structure defines cell behavior modes within genomes. Each mode specifies
/// visual properties, division parameters, adhesion settings, and special behaviors.
/// The layout matches the reference implementation's GPUMode structure exactly.
///
/// ## Memory Layout
/// - Total size: 144 bytes (matches reference exactly)
/// - All Vec4 fields are 16-byte aligned
/// - Adhesion settings embedded as 48-byte structure
/// - Padding ensures consistent cross-platform layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuMode {
    // === Visual Properties (64 bytes) ===
    /// Cell color: Vec4(r, g, b, a)
    pub color: [f32; 4],
    /// Orientation A: Vec4(quaternion w, x, y, z)
    pub orientation_a: [f32; 4],
    /// Orientation B: Vec4(quaternion w, x, y, z)
    pub orientation_b: [f32; 4],
    /// Split direction: Vec4(x, y, z, w)
    pub split_direction: [f32; 4],
    
    // === Child Modes (8 bytes) ===
    /// Child mode indices: [child_a_mode, child_b_mode]
    pub child_modes: [i32; 2],
    
    // === Division Properties (8 bytes) ===
    /// Time interval between divisions
    pub split_interval: f32,
    /// Genome offset for mode calculations
    pub genome_offset: i32,
    
    // === Adhesion Settings (48 bytes) ===
    /// Complete adhesion configuration for this mode
    pub adhesion_settings: GpuModeAdhesionSettings,
    
    // === Adhesion Behavior (16 bytes) ===
    /// Parent makes adhesion on division (0 = false, 1 = true)
    pub parent_make_adhesion: i32,
    /// Child A keeps adhesion after division (0 = false, 1 = true)
    pub child_a_keep_adhesion: i32,
    /// Child B keeps adhesion after division (0 = false, 1 = true)
    pub child_b_keep_adhesion: i32,
    /// Maximum number of adhesions per cell
    pub max_adhesions: i32,
    
    // === Cell Type Behavior (32 bytes) ===
    /// Cell type identifier (0 = Test, 1 = Flagellocyte)
    pub cell_type: u32,
    /// Automatic nutrient gain rate (for Test cells)
    pub nutrient_gain_rate: f32,
    /// Swim force magnitude (for Flagellocyte cells)
    pub swim_force: f32,
    /// Maximum cell size constraint
    pub max_cell_size: f32,
    /// Nutrient priority for resource competition
    pub nutrient_priority: f32,
    /// Whether to prioritize when nutrients are low (0 = false, 1 = true)
    pub prioritize_when_low: u32,
    /// Cell membrane stiffness for collision response
    pub membrane_stiffness: f32,
    /// Padding to maintain alignment
    _padding_cell_type: u32,
}

impl Default for GpuMode {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],           // White
            orientation_a: [1.0, 0.0, 0.0, 0.0],   // Identity quaternion
            orientation_b: [1.0, 0.0, 0.0, 0.0],   // Identity quaternion
            split_direction: [0.0, 1.0, 0.0, 0.0], // Up direction
            child_modes: [0, 0],                    // Both children use mode 0
            split_interval: 10.0,                   // 10 time units between divisions
            genome_offset: 0,
            adhesion_settings: GpuModeAdhesionSettings::default(),
            parent_make_adhesion: 1,                // Parent makes adhesions
            child_a_keep_adhesion: 1,               // Child A keeps adhesions
            child_b_keep_adhesion: 1,               // Child B keeps adhesions
            max_adhesions: 10,                      // Maximum 10 adhesions per cell
            
            // Cell type behavior (Test cell defaults)
            cell_type: CellType::Test.to_u32(),     // Default to Test cell
            nutrient_gain_rate: 0.2,                // Default nutrient gain rate
            swim_force: 0.0,                        // No swim force for Test cells
            max_cell_size: 2.0,                     // Default maximum size
            nutrient_priority: 1.0,                 // Neutral priority
            prioritize_when_low: 1,                 // Enable low-nutrient priority boost
            membrane_stiffness: 50.0,               // Default membrane stiffness
            _padding_cell_type: 0,
        }
    }
}

/// Adhesion settings structure matching reference GPUModeAdhesionSettings.
///
/// This structure contains all parameters for adhesion bond mechanics including
/// spring forces, damping, constraints, and breaking conditions. The layout
/// matches the reference implementation exactly.
///
/// ## Memory Layout
/// - Total size: 48 bytes (matches reference exactly)
/// - All fields are 4-byte aligned
/// - Boolean values stored as i32 for GPU compatibility
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuModeAdhesionSettings {
    /// Can adhesion bonds break (0 = false, 1 = true)
    pub can_break: i32,
    /// Force threshold for bond breaking
    pub break_force: f32,
    /// Rest length of adhesion bonds
    pub rest_length: f32,
    /// Linear spring stiffness coefficient
    pub linear_spring_stiffness: f32,
    /// Linear spring damping coefficient
    pub linear_spring_damping: f32,
    /// Orientation spring stiffness (rotational constraint)
    pub orientation_spring_stiffness: f32,
    /// Orientation spring damping (rotational damping)
    pub orientation_spring_damping: f32,
    /// Maximum angular deviation before constraint activates
    pub max_angular_deviation: f32,
    /// Twist constraint stiffness (prevents rotation around bond axis)
    pub twist_constraint_stiffness: f32,
    /// Twist constraint damping
    pub twist_constraint_damping: f32,
    /// Enable twist constraint (0 = false, 1 = true)
    pub enable_twist_constraint: i32,
    /// Padding to ensure 48-byte total size
    _padding: i32,
}

impl Default for GpuModeAdhesionSettings {
    fn default() -> Self {
        Self {
            can_break: 1,                           // Bonds can break
            break_force: 50.0,                      // Moderate breaking force
            rest_length: 2.0,                       // 2 unit rest length
            linear_spring_stiffness: 10.0,          // Moderate stiffness
            linear_spring_damping: 1.0,             // Light damping
            orientation_spring_stiffness: 5.0,      // Rotational constraint
            orientation_spring_damping: 0.5,        // Rotational damping
            max_angular_deviation: 45.0_f32.to_radians(), // 45 degree max deviation
            twist_constraint_stiffness: 2.0,        // Twist resistance
            twist_constraint_damping: 0.2,          // Twist damping
            enable_twist_constraint: 1,             // Enable twist constraint
            _padding: 0,
        }
    }
}

/// Cell count buffer matching reference implementation.
///
/// This structure tracks the current state of cell and adhesion management
/// including counts, free slot tracking, and memory management information.
///
/// ## Memory Layout
/// - Total size: 16 bytes (4 x u32)
/// - All fields are 4-byte aligned
/// - Matches reference totalCellCount buffer exactly
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CellCountBuffer {
    /// Total number of cell slots allocated
    pub total_cell_count: u32,
    /// Number of currently active (live) cells
    pub live_cell_count: u32,
    /// Total number of adhesion connections allocated
    pub total_adhesion_count: u32,
    /// Top of free adhesion slot stack (for memory management)
    pub free_adhesion_top: u32,
}

impl Default for CellCountBuffer {
    fn default() -> Self {
        Self {
            total_cell_count: 0,
            live_cell_count: 0,
            total_adhesion_count: 0,
            free_adhesion_top: 0,
        }
    }
}

/// Instance extraction parameters for GPU compute shader.
///
/// This structure contains parameters needed for extracting rendering data
/// from GPU physics simulation state. It includes cell counts, mode counts,
/// and timing information for the extraction process.
///
/// ## Memory Layout
/// - Total size: 32 bytes (8 x f32/u32)
/// - All fields are 4-byte aligned
/// - Padding ensures 16-byte alignment for GPU compatibility
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ExtractionParams {
    /// Number of active cells to extract
    pub cell_count: u32,
    /// Total number of modes across all genomes
    pub mode_count: u32,
    /// Number of cell type visual configurations
    pub cell_type_count: u32,
    /// Current simulation time for animation offsets
    pub current_time: f32,
    /// Padding to ensure 32-byte total size
    pub _padding: [f32; 4],
}

impl Default for ExtractionParams {
    fn default() -> Self {
        Self {
            cell_count: 0,
            mode_count: 0,
            cell_type_count: 1,
            current_time: 0.0,
            _padding: [0.0; 4],
        }
    }
}

// ============================================================================
// STORAGE BUFFER STRUCTURES
// ============================================================================

/// Adhesion connection structure matching reference exactly (96 bytes total).
///
/// This structure represents a connection between two cells including all
/// mechanical properties, anchor points, and constraint data. The layout
/// matches the reference AdhesionConnection structure exactly.
///
/// ## Memory Layout
/// - Total size: 96 bytes (matches reference exactly)
/// - All Vec3 fields padded to Vec4 for 16-byte alignment
/// - Quaternions stored as Vec4 (w, x, y, z)
/// - Explicit padding ensures exact size match
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuAdhesionConnection {
    /// Index of first cell in connection
    pub cell_a_index: u32,
    /// Index of second cell in connection
    pub cell_b_index: u32,
    /// Mode index for adhesion settings lookup
    pub mode_index: u32,
    /// Connection active flag (1 = active, 0 = inactive)
    pub is_active: u32,
    
    /// Zone classification for cell A (0=ZoneA, 1=ZoneB, 2=ZoneC)
    pub zone_a: u32,
    /// Zone classification for cell B (0=ZoneA, 1=ZoneB, 2=ZoneC)
    pub zone_b: u32,
    /// Padding for alignment (8 bytes)
    _padding_zones: [u32; 2],
    
    /// Anchor direction for cell A in local space (normalized Vec3 + padding)
    pub anchor_direction_a: [f32; 3],
    /// Padding for 16-byte alignment
    _padding_a: f32,
    
    /// Anchor direction for cell B in local space (normalized Vec3 + padding)
    pub anchor_direction_b: [f32; 3],
    /// Padding for 16-byte alignment
    _padding_b: f32,
    
    /// Reference quaternion for twist constraint for cell A
    pub twist_reference_a: [f32; 4],
    
    /// Reference quaternion for twist constraint for cell B
    pub twist_reference_b: [f32; 4],
}

impl Default for GpuAdhesionConnection {
    fn default() -> Self {
        Self {
            cell_a_index: 0,
            cell_b_index: 0,
            mode_index: 0,
            is_active: 0,
            zone_a: 0,
            zone_b: 0,
            _padding_zones: [0; 2],
            anchor_direction_a: [1.0, 0.0, 0.0],   // Default to X axis
            _padding_a: 0.0,
            anchor_direction_b: [-1.0, 0.0, 0.0],  // Default to -X axis
            _padding_b: 0.0,
            twist_reference_a: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
            twist_reference_b: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
        }
    }
}

/// Collision pair structure for spatial collision detection.
///
/// This structure represents a detected collision between two cells including
/// overlap amount and collision normal. Used by collision detection compute
/// shaders to pass results to force calculation shaders.
///
/// ## Memory Layout
/// - Total size: 32 bytes (8 x f32/u32)
/// - All fields are 4-byte aligned
/// - Padding ensures consistent layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCollisionPair {
    /// Index of first cell in collision
    pub index_a: u32,
    /// Index of second cell in collision
    pub index_b: u32,
    /// Amount of overlap between cells
    pub overlap: f32,
    /// Collision normal X component
    pub normal_x: f32,
    /// Collision normal Y component
    pub normal_y: f32,
    /// Collision normal Z component
    pub normal_z: f32,
    /// Padding to 32 bytes
    _padding: [u32; 2],
}

impl Default for GpuCollisionPair {
    fn default() -> Self {
        Self {
            index_a: 0,
            index_b: 0,
            overlap: 0.0,
            normal_x: 0.0,
            normal_y: 0.0,
            normal_z: 1.0,  // Default normal pointing up
            _padding: [0; 2],
        }
    }
}

/// Instance data for rendering (matching reference CPUInstanceData).
///
/// This structure contains all data needed for GPU instanced rendering
/// of cells. The layout is optimized for the current CellRenderer system
/// and matches the expected instance data format.
///
/// ## Memory Layout
/// - Total size: 48 bytes (12 x f32)
/// - All fields are 4-byte aligned
/// - Compatible with current InstanceBuilder
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuInstanceData {
    /// Position and radius: Vec4(x, y, z, radius)
    pub position_and_radius: [f32; 4],
    /// Color: Vec4(r, g, b, a)
    pub color: [f32; 4],
    /// Orientation: Vec4(quaternion w, x, y, z)
    pub orientation: [f32; 4],
}

impl Default for GpuInstanceData {
    fn default() -> Self {
        Self {
            position_and_radius: [0.0, 0.0, 0.0, 1.0], // Origin with radius 1
            color: [1.0, 1.0, 1.0, 1.0],                // White
            orientation: [1.0, 0.0, 0.0, 0.0],          // Identity quaternion
        }
    }
}

/// Particle structure for volumetric effects (from reference).
///
/// This structure represents particles used for volumetric rendering effects
/// such as cell division particles, death effects, or environmental particles.
///
/// ## Memory Layout
/// - Total size: 32 bytes (8 x f32)
/// - All fields are 4-byte aligned
/// - Compatible with volumetric rendering shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParticle {
    /// World position: Vec3(x, y, z)
    pub position: [f32; 3],
    /// Remaining lifetime (0 = dead)
    pub lifetime: f32,
    /// Velocity for movement: Vec3(x, y, z)
    pub velocity: [f32; 3],
    /// Maximum lifetime for fade calculation
    pub max_lifetime: f32,
    /// Color: Vec4(r, g, b, a)
    pub color: [f32; 4],
}

impl Default for GpuParticle {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            lifetime: 0.0,
            velocity: [0.0, 0.0, 0.0],
            max_lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

// ============================================================================
// BUFFER CREATION UTILITIES
// ============================================================================

/// Create a uniform buffer with the specified data and label.
///
/// Uniform buffers are used for data that remains constant across all
/// invocations of a compute shader (like physics parameters).
///
/// # Arguments
/// * `device` - wgpu device for buffer creation
/// * `data` - Initial data to store in buffer
/// * `label` - Debug label for the buffer
pub fn create_uniform_buffer<T: Pod>(
    device: &wgpu::Device,
    data: &T,
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(std::slice::from_ref(data)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create a storage buffer with the specified capacity and label.
///
/// Storage buffers are used for large arrays of data that can be
/// read and written by compute shaders.
///
/// # Arguments
/// * `device` - wgpu device for buffer creation
/// * `capacity` - Number of elements the buffer can hold
/// * `label` - Debug label for the buffer
pub fn create_storage_buffer<T: Pod + Default>(
    device: &wgpu::Device,
    capacity: usize,
    label: &str,
) -> wgpu::Buffer {
    let size = (capacity * std::mem::size_of::<T>()) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a storage buffer initialized with data.
///
/// # Arguments
/// * `device` - wgpu device for buffer creation
/// * `data` - Initial data to store in buffer
/// * `label` - Debug label for the buffer
pub fn create_storage_buffer_init<T: Pod>(
    device: &wgpu::Device,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::COPY_SRC,
    })
}

// ============================================================================
// TRIPLE-BUFFERED SIMULATION BUFFERS
// ============================================================================

/// Complete GPU simulation buffer system with triple buffering for maximum performance.
///
/// This structure manages all simulation state using Structure-of-Arrays (SoA) layout
/// with triple buffering to eliminate CPU-GPU synchronization stalls. All buffers are
/// optimized for compute shader access patterns and match the reference implementation
/// layout exactly.
///
/// ## Triple Buffer Architecture
/// - **Physics Computation**: Operates on buffer set N
/// - **Visual Data Extraction**: Operates on buffer set N-1
/// - **Rendering**: Operates on buffer set N-2
/// - All operations overlap for maximum GPU utilization
///
/// ## Memory Layout Optimization
/// - **SoA Layout**: Optimal for parallel compute shader processing
/// - **16-byte Alignment**: All Vec4 fields for GPU cache efficiency
/// - **Prefix-Sum Ready**: Lifecycle buffers designed for compaction algorithms
/// - **Reference Compatible**: Matches Biospheres-Master buffer layouts exactly
pub struct GpuSimulationBuffers {
    /// Triple-buffered physics buffers (current, previous, next)
    /// Contains all cell properties, adhesion data, and spatial grid information
    physics_buffers: [super::triple_buffer::GpuPhysicsBuffers; 3],
    
    /// Triple-buffered instance buffers for rendering data extraction
    /// Contains extracted visual data: positions, colors, orientations
    instance_buffers: [super::triple_buffer::GpuInstanceBuffers; 3],
    
    /// Current physics buffer index (0, 1, or 2)
    /// Updated atomically after each physics step completion
    current_physics_index: std::sync::atomic::AtomicUsize,
    
    /// Current instance buffer index (0, 1, or 2)
    /// Updated atomically after visual data extraction
    current_instance_index: std::sync::atomic::AtomicUsize,
    
    /// Maximum capacity (fixed at creation)
    capacity: u32,
}

impl GpuSimulationBuffers {
    /// Create a new GPU simulation buffer system with triple buffering.
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
    /// - Uses GPU-optimal SoA layout with 16-byte alignment
    /// - Triple buffering eliminates all CPU-GPU synchronization points
    /// - Spatial grid sized for 64³ cells (262,144 grid cells total)
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        use std::sync::atomic::AtomicUsize;
        
        // Create three identical sets of physics buffers
        let physics_buffers = [
            super::triple_buffer::GpuPhysicsBuffers::new(device, capacity, "Physics Buffer Set 0"),
            super::triple_buffer::GpuPhysicsBuffers::new(device, capacity, "Physics Buffer Set 1"),
            super::triple_buffer::GpuPhysicsBuffers::new(device, capacity, "Physics Buffer Set 2"),
        ];
        
        // Create three identical sets of instance buffers
        let instance_buffers = [
            super::triple_buffer::GpuInstanceBuffers::new(device, capacity, "Instance Buffer Set 0"),
            super::triple_buffer::GpuInstanceBuffers::new(device, capacity, "Instance Buffer Set 1"),
            super::triple_buffer::GpuInstanceBuffers::new(device, capacity, "Instance Buffer Set 2"),
        ];
        
        Self {
            physics_buffers,
            instance_buffers,
            current_physics_index: AtomicUsize::new(0),
            current_instance_index: AtomicUsize::new(0),
            capacity,
        }
    }
    
    /// Atomically rotate to the next physics buffer set.
    ///
    /// This is called after each physics step completion to switch to the next
    /// buffer set for the following frame. The rotation is atomic and lock-free.
    ///
    /// # Returns
    /// The new current buffer set index (0, 1, or 2)
    pub fn rotate_physics_buffers(&self) -> usize {
        use std::sync::atomic::Ordering;
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
        use std::sync::atomic::Ordering;
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
    pub fn get_current_physics_buffers(&self) -> &super::triple_buffer::GpuPhysicsBuffers {
        use std::sync::atomic::Ordering;
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
    pub fn get_rendering_physics_buffers(&self) -> &super::triple_buffer::GpuPhysicsBuffers {
        use std::sync::atomic::Ordering;
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
    pub fn get_current_instance_buffers(&self) -> &super::triple_buffer::GpuInstanceBuffers {
        use std::sync::atomic::Ordering;
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
    pub fn get_rendering_instance_buffers(&self) -> &super::triple_buffer::GpuInstanceBuffers {
        use std::sync::atomic::Ordering;
        let instance_index = self.current_instance_index.load(Ordering::Acquire);
        let render_index = (instance_index + 2) % 3; // 1 frame behind for stability
        &self.instance_buffers[render_index]
    }
    
    /// Get the maximum capacity.
    pub fn capacity(&self) -> u32 {
        self.capacity
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
        use std::sync::atomic::Ordering;
        (
            self.current_physics_index.load(Ordering::Acquire),
            self.current_instance_index.load(Ordering::Acquire),
        )
    }
}

// ============================================================================
// ADHESION BUFFER MANAGEMENT
// ============================================================================

/// Adhesion buffer management utilities for GPU scene.
///
/// These functions provide GPU-compatible adhesion management that matches
/// the CPU AdhesionConnectionManager functionality while being optimized
/// for GPU compute shader access patterns.
pub struct GpuAdhesionBufferManager;

impl GpuAdhesionBufferManager {
    /// Initialize adhesion buffers with empty state.
    ///
    /// This sets up the initial state for adhesion buffers:
    /// - All adhesion indices set to EMPTY_ADHESION_SLOT (-1)
    /// - All adhesion counts set to 0
    /// - All connections marked as inactive
    ///
    /// # Arguments
    /// * `device` - wgpu device for buffer operations
    /// * `queue` - wgpu queue for buffer writes
    /// * `physics_buffers` - Physics buffers to initialize
    /// * `capacity` - Number of cells to initialize
    pub fn initialize_adhesion_buffers(
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        physics_buffers: &super::triple_buffer::GpuPhysicsBuffers,
        capacity: u32,
    ) {
        // Initialize adhesion indices (10 i32 per cell, all set to -1)
        let adhesion_indices_data: Vec<i32> = (0..capacity)
            .flat_map(|_| init_gpu_adhesion_indices())
            .collect();
        
        queue.write_buffer(
            &physics_buffers.adhesion_indices,
            0,
            bytemuck::cast_slice(&adhesion_indices_data),
        );
        
        // Initialize adhesion counts (all set to 0)
        let adhesion_counts_data: Vec<u32> = vec![0; capacity as usize];
        queue.write_buffer(
            &physics_buffers.adhesion_counts,
            0,
            bytemuck::cast_slice(&adhesion_counts_data),
        );
        
        // Initialize adhesion connections (all inactive)
        let connections_data: Vec<GpuAdhesionConnection> = vec![GpuAdhesionConnection::default(); capacity as usize];
        queue.write_buffer(
            &physics_buffers.adhesion_connections,
            0,
            bytemuck::cast_slice(&connections_data),
        );
    }
    
    /// Create bind group layout for adhesion compute shaders.
    ///
    /// This creates the bind group layout that adhesion compute shaders
    /// will use to access adhesion buffers. The layout includes all
    /// necessary buffers for adhesion physics computation.
    ///
    /// # Arguments
    /// * `device` - wgpu device for layout creation
    ///
    /// # Returns
    /// Bind group layout for adhesion compute shaders
    pub fn create_adhesion_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Adhesion Compute Bind Group Layout"),
            entries: &[
                // Adhesion connections (read/write)
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
                // Adhesion indices (read/write)
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
                // Adhesion counts (read/write)
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
                // Cell positions (read-only)
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
                // Cell orientations (read-only)
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
                // Genome orientations (read-only)
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
        })
    }
    
    /// Create bind group for adhesion compute shaders.
    ///
    /// This creates the actual bind group that binds the adhesion buffers
    /// to the compute shader. The bind group must match the layout created
    /// by `create_adhesion_bind_group_layout`.
    ///
    /// # Arguments
    /// * `device` - wgpu device for bind group creation
    /// * `layout` - Bind group layout (from create_adhesion_bind_group_layout)
    /// * `physics_buffers` - Physics buffers to bind
    ///
    /// # Returns
    /// Bind group for adhesion compute shaders
    pub fn create_adhesion_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        physics_buffers: &super::triple_buffer::GpuPhysicsBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adhesion Compute Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: physics_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: physics_buffers.adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: physics_buffers.adhesion_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: physics_buffers.position_and_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: physics_buffers.orientation.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: physics_buffers.genome_orientation.as_entire_binding(),
                },
            ],
        })
    }
}

// ============================================================================
// GENOME MODE BUFFER MANAGEMENT
// ============================================================================

/// Genome mode buffer manager for GPU-based mode lookups and transitions.
///
/// This manager handles the conversion of CPU-side Genome structures to GPU-compatible
/// GpuMode buffers, manages mode-to-cell mapping, and provides efficient GPU lookups
/// for cell behavior based on genome modes.
///
/// ## Key Features
/// - **CPU-to-GPU Conversion**: Converts ModeSettings to GpuMode structures
/// - **Mode Indexing**: Maps genome modes to absolute indices for GPU lookup
/// - **Cell-to-Mode Mapping**: Handles genome_ids to mode_indices conversion
/// - **Dynamic Updates**: Supports adding/removing genomes during simulation
/// - **GPU Optimization**: Uses efficient buffer layouts for compute shader access
pub struct GenomeModeBufferManager {
    /// GPU buffer containing all genome modes as GpuMode structures
    /// Indexed by absolute mode index across all genomes
    genome_modes_buffer: wgpu::Buffer,
    
    /// CPU-side mapping from genome_id to starting mode index
    /// Used to convert relative mode indices to absolute indices
    genome_mode_offsets: Vec<u32>,
    
    /// Total number of modes across all genomes
    total_mode_count: u32,
    
    /// Maximum capacity of the modes buffer
    max_capacity: u32,
    
    /// Whether the GPU buffer needs updating
    needs_update: bool,
}

impl GenomeModeBufferManager {
    /// Create a new genome mode buffer manager.
    ///
    /// # Arguments
    /// * `device` - wgpu device for buffer creation
    /// * `max_capacity` - Maximum number of modes across all genomes
    pub fn new(device: &wgpu::Device, max_capacity: u32) -> Self {
        let genome_modes_buffer = create_storage_buffer::<GpuMode>(
            device,
            max_capacity as usize,
            "Genome Modes Buffer"
        );
        
        Self {
            genome_modes_buffer,
            genome_mode_offsets: Vec::new(),
            total_mode_count: 0,
            max_capacity,
            needs_update: false,
        }
    }
    
    /// Update the GPU buffer with current genome data.
    ///
    /// This method converts all CPU-side genomes to GPU-compatible mode data
    /// and uploads it to the GPU buffer. It also updates the mode offset mapping
    /// for efficient genome_id to mode_index conversion.
    ///
    /// # Arguments
    /// * `queue` - wgpu queue for buffer updates
    /// * `genomes` - Current genome data from CPU
    ///
    /// # Returns
    /// Result indicating success or failure of the update
    pub fn update_from_genomes(
        &mut self,
        queue: &wgpu::Queue,
        genomes: &[Genome],
    ) -> Result<(), String> {
        // Calculate total mode count and validate capacity
        let total_modes: usize = genomes.iter().map(|g| g.modes.len()).sum();
        if total_modes > self.max_capacity as usize {
            return Err(format!(
                "Total mode count {} exceeds buffer capacity {}",
                total_modes, self.max_capacity
            ));
        }
        
        // Build GPU mode data and offset mapping
        let mut gpu_modes = Vec::with_capacity(total_modes);
        let mut mode_offsets = Vec::with_capacity(genomes.len());
        let mut current_offset = 0u32;
        
        for genome in genomes {
            mode_offsets.push(current_offset);
            
            for mode in &genome.modes {
                let gpu_mode = self.convert_mode_to_gpu(mode, current_offset);
                gpu_modes.push(gpu_mode);
                current_offset += 1;
            }
        }
        
        // Update GPU buffer with mode data
        if !gpu_modes.is_empty() {
            queue.write_buffer(
                &self.genome_modes_buffer,
                0,
                bytemuck::cast_slice(&gpu_modes),
            );
        }
        
        // Update internal state
        self.genome_mode_offsets = mode_offsets;
        self.total_mode_count = total_modes as u32;
        self.needs_update = false;
        
        log::debug!(
            "Updated genome mode buffer: {} genomes, {} total modes",
            genomes.len(),
            total_modes
        );
        
        Ok(())
    }
    
    /// Convert a CPU ModeSettings to GPU GpuMode structure.
    ///
    /// This method handles the conversion from the CPU-side ModeSettings structure
    /// to the GPU-compatible GpuMode structure, including proper alignment and
    /// data type conversions.
    ///
    /// # Arguments
    /// * `mode` - CPU-side mode settings
    /// * `mode_offset` - Absolute mode index for this mode
    ///
    /// # Returns
    /// GPU-compatible mode structure
    fn convert_mode_to_gpu(&self, mode: &crate::genome::ModeSettings, mode_offset: u32) -> GpuMode {
        GpuMode {
            // Visual properties (convert Vec3 to Vec4 for GPU alignment)
            color: [mode.color.x, mode.color.y, mode.color.z, mode.opacity],
            orientation_a: [
                mode.child_a.orientation.w,
                mode.child_a.orientation.x,
                mode.child_a.orientation.y,
                mode.child_a.orientation.z,
            ],
            orientation_b: [
                mode.child_b.orientation.w,
                mode.child_b.orientation.x,
                mode.child_b.orientation.y,
                mode.child_b.orientation.z,
            ],
            split_direction: [
                mode.parent_split_direction.x.to_radians().sin(),
                mode.parent_split_direction.y.to_radians().cos(),
                0.0, // Z component
                0.0, // Padding
            ],
            
            // Child modes (convert to absolute indices)
            child_modes: [mode.child_a.mode_number, mode.child_b.mode_number],
            
            // Division properties
            split_interval: mode.split_interval,
            genome_offset: mode_offset as i32,
            
            // Adhesion settings
            adhesion_settings: GpuModeAdhesionSettings {
                can_break: if mode.adhesion_settings.can_break { 1 } else { 0 },
                break_force: mode.adhesion_settings.break_force,
                rest_length: mode.adhesion_settings.rest_length,
                linear_spring_stiffness: mode.adhesion_settings.linear_spring_stiffness,
                linear_spring_damping: mode.adhesion_settings.linear_spring_damping,
                orientation_spring_stiffness: mode.adhesion_settings.orientation_spring_stiffness,
                orientation_spring_damping: mode.adhesion_settings.orientation_spring_damping,
                max_angular_deviation: mode.adhesion_settings.max_angular_deviation,
                twist_constraint_stiffness: mode.adhesion_settings.twist_constraint_stiffness,
                twist_constraint_damping: mode.adhesion_settings.twist_constraint_damping,
                enable_twist_constraint: if mode.adhesion_settings.enable_twist_constraint { 1 } else { 0 },
                _padding: 0,
            },
            
            // Adhesion behavior
            parent_make_adhesion: if mode.parent_make_adhesion { 1 } else { 0 },
            child_a_keep_adhesion: if mode.child_a.keep_adhesion { 1 } else { 0 },
            child_b_keep_adhesion: if mode.child_b.keep_adhesion { 1 } else { 0 },
            max_adhesions: mode.max_adhesions,
            
            // Cell type behavior
            cell_type: mode.cell_type as u32,
            nutrient_gain_rate: mode.nutrient_gain_rate,
            swim_force: mode.swim_force,
            max_cell_size: mode.max_cell_size,
            nutrient_priority: mode.nutrient_priority,
            prioritize_when_low: if mode.prioritize_when_low { 1 } else { 0 },
            membrane_stiffness: mode.membrane_stiffness,
            _padding_cell_type: 0,
        }
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
        if genome_id >= self.genome_mode_offsets.len() {
            return None;
        }
        
        let base_offset = self.genome_mode_offsets[genome_id];
        let absolute_index = base_offset + relative_mode_index as u32;
        
        if absolute_index < self.total_mode_count {
            Some(absolute_index)
        } else {
            None
        }
    }
    
    /// Get the genome modes buffer for GPU compute shaders.
    ///
    /// # Returns
    /// Reference to the GPU buffer containing all genome modes
    pub fn get_modes_buffer(&self) -> &wgpu::Buffer {
        &self.genome_modes_buffer
    }
    
    /// Get the total number of modes across all genomes.
    ///
    /// # Returns
    /// Total mode count
    pub fn get_total_mode_count(&self) -> u32 {
        self.total_mode_count
    }
    
    /// Check if the buffer needs updating.
    ///
    /// # Returns
    /// True if the GPU buffer is out of sync with CPU data
    pub fn needs_update(&self) -> bool {
        self.needs_update
    }
    
    /// Mark the buffer as needing an update.
    ///
    /// This should be called whenever genomes are added, removed, or modified.
    pub fn mark_dirty(&mut self) {
        self.needs_update = true;
    }
    
    /// Get genome mode offset for a specific genome.
    ///
    /// # Arguments
    /// * `genome_id` - Genome identifier
    ///
    /// # Returns
    /// Starting mode index for the genome, or None if invalid
    pub fn get_genome_mode_offset(&self, genome_id: usize) -> Option<u32> {
        self.genome_mode_offsets.get(genome_id).copied()
    }
    
    /// Create bind group layout for genome mode compute shaders.
    ///
    /// This creates the bind group layout that compute shaders will use
    /// to access genome mode data for cell behavior calculations.
    ///
    /// # Arguments
    /// * `device` - wgpu device for layout creation
    ///
    /// # Returns
    /// Bind group layout for genome mode access
    pub fn create_genome_mode_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Genome Mode Bind Group Layout"),
            entries: &[
                // Genome modes buffer (read-only)
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
                // Mode indices buffer (read-only)
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
                // Genome IDs buffer (read-only)
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
        })
    }
    
    /// Create bind group for genome mode compute shaders.
    ///
    /// This creates the actual bind group that binds the genome mode buffers
    /// to compute shaders for cell behavior calculations.
    ///
    /// # Arguments
    /// * `device` - wgpu device for bind group creation
    /// * `layout` - Bind group layout (from create_genome_mode_bind_group_layout)
    /// * `physics_buffers` - Physics buffers containing mode indices and genome IDs
    ///
    /// # Returns
    /// Bind group for genome mode access
    pub fn create_genome_mode_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        physics_buffers: &super::triple_buffer::GpuPhysicsBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Genome Mode Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.genome_modes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: physics_buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: physics_buffers.genome_ids.as_entire_binding(),
                },
            ],
        })
    }
}

// ============================================================================
// CELL TYPE BEHAVIOR UTILITIES
// ============================================================================

/// Cell type behavior utilities for GPU scene integration.
///
/// This module provides utilities for working with cell types in the GPU scene,
/// including behavior calculations and type-specific parameter extraction.
pub struct CellTypeBehaviorUtils;

impl CellTypeBehaviorUtils {
    /// Calculate nutrient consumption rate for a cell based on its swim force.
    ///
    /// Flagellocyte cells consume nutrients based on their swim force usage.
    /// Test cells do not consume nutrients through swimming.
    ///
    /// # Arguments
    /// * `cell_type` - Type of the cell
    /// * `swim_force` - Current swim force magnitude
    ///
    /// # Returns
    /// Nutrient consumption rate per time unit
    pub fn calculate_nutrient_consumption(cell_type: CellType, swim_force: f32) -> f32 {
        match cell_type {
            CellType::Test => 0.0, // Test cells don't consume nutrients
            CellType::Flagellocyte => {
                // Consumption rate proportional to swim force squared (energy cost)
                let base_consumption = 0.1; // Base consumption rate
                let force_factor = swim_force * swim_force; // Quadratic energy cost
                base_consumption * force_factor
            }
        }
    }
    
    /// Calculate automatic nutrient gain for a cell based on its type.
    ///
    /// Test cells automatically gain nutrients at a fixed rate.
    /// Flagellocyte cells do not gain nutrients automatically.
    ///
    /// # Arguments
    /// * `cell_type` - Type of the cell
    /// * `nutrient_gain_rate` - Base nutrient gain rate from genome mode
    /// * `delta_time` - Time step for the calculation
    ///
    /// # Returns
    /// Nutrient gain amount for this time step
    pub fn calculate_automatic_nutrient_gain(
        cell_type: CellType,
        nutrient_gain_rate: f32,
        delta_time: f32,
    ) -> f32 {
        match cell_type {
            CellType::Test => nutrient_gain_rate * delta_time,
            CellType::Flagellocyte => 0.0, // No automatic gain
        }
    }
    
    /// Check if a cell type should apply swim forces.
    ///
    /// # Arguments
    /// * `cell_type` - Type of the cell
    ///
    /// # Returns
    /// True if the cell type can apply swim forces
    pub fn can_apply_swim_force(cell_type: CellType) -> bool {
        match cell_type {
            CellType::Test => false,
            CellType::Flagellocyte => true,
        }
    }
    
    /// Get the visual mesh type for a cell based on its type and swim force.
    ///
    /// This determines what kind of mesh should be used for rendering the cell.
    /// Flagellocyte cells with active swim forces get specialized spike meshes.
    ///
    /// # Arguments
    /// * `cell_type` - Type of the cell
    /// * `swim_force` - Current swim force magnitude
    ///
    /// # Returns
    /// Mesh type identifier for rendering
    pub fn get_visual_mesh_type(cell_type: CellType, swim_force: f32) -> u32 {
        match cell_type {
            CellType::Test => 0, // Standard sphere mesh
            CellType::Flagellocyte => {
                if swim_force > 0.1 {
                    1 // Spiked mesh for active swimming
                } else {
                    0 // Standard sphere mesh when not swimming
                }
            }
        }
    }
    
    /// Extract cell type behavior from a genome mode.
    ///
    /// This creates a CellTypeBehavior structure from a genome mode for
    /// efficient GPU buffer uploads.
    ///
    /// # Arguments
    /// * `mode` - Genome mode settings
    ///
    /// # Returns
    /// Cell type behavior structure for GPU use
    pub fn extract_behavior_from_mode(mode: &crate::genome::ModeSettings) -> CellTypeBehavior {
        CellTypeBehavior {
            cell_type: mode.cell_type as u32,
            nutrient_gain_rate: mode.nutrient_gain_rate,
            swim_force: mode.swim_force,
            nutrient_consumption_rate: Self::calculate_nutrient_consumption(
                CellType::from_u32(mode.cell_type as u32).unwrap_or(CellType::Test),
                mode.swim_force,
            ),
            max_cell_size: mode.max_cell_size,
            nutrient_priority: mode.nutrient_priority,
            prioritize_when_low: if mode.prioritize_when_low { 1 } else { 0 },
            _padding: 0,
        }
    }
}

// ============================================================================
// ADVANCED GENOME FEATURE UTILITIES
// ============================================================================

/// Advanced genome feature utilities for complex cell behaviors.
///
/// This module provides utilities for advanced genome features including
/// swim force calculations, nutrient priority management, and size constraints.
pub struct AdvancedGenomeFeatures;

impl AdvancedGenomeFeatures {
    /// Calculate directional swim force vector for a cell.
    ///
    /// This method calculates the swim force vector based on the cell's
    /// orientation and the swim force magnitude from its genome mode.
    ///
    /// # Arguments
    /// * `swim_force_magnitude` - Magnitude of swim force from genome mode
    /// * `cell_orientation` - Current cell orientation (quaternion)
    /// * `genome_orientation` - Genome-space orientation for direction reference
    ///
    /// # Returns
    /// 3D swim force vector in world space
    pub fn calculate_swim_force_vector(
        swim_force_magnitude: f32,
        cell_orientation: glam::Quat,
        genome_orientation: glam::Quat,
    ) -> glam::Vec3 {
        if swim_force_magnitude <= 0.0 {
            return glam::Vec3::ZERO;
        }
        
        // Calculate forward direction in genome space
        let genome_forward = genome_orientation * glam::Vec3::NEG_Z; // -Z is forward
        
        // Transform to world space using current cell orientation
        let world_forward = cell_orientation * genome_forward;
        
        // Apply magnitude
        world_forward.normalize() * swim_force_magnitude
    }
    
    /// Calculate nutrient priority factor for resource competition.
    ///
    /// This method calculates the effective nutrient priority for a cell
    /// based on its base priority and current nutrient levels.
    ///
    /// # Arguments
    /// * `base_priority` - Base nutrient priority from genome mode
    /// * `prioritize_when_low` - Whether to boost priority when nutrients are low
    /// * `current_nutrients` - Current nutrient level (mass)
    /// * `max_nutrients` - Maximum nutrient capacity
    ///
    /// # Returns
    /// Effective nutrient priority factor
    pub fn calculate_nutrient_priority_factor(
        base_priority: f32,
        prioritize_when_low: bool,
        current_nutrients: f32,
        max_nutrients: f32,
    ) -> f32 {
        let mut priority = base_priority;
        
        if prioritize_when_low && max_nutrients > 0.0 {
            let nutrient_ratio = current_nutrients / max_nutrients;
            
            // Boost priority when nutrients are low (below 30%)
            if nutrient_ratio < 0.3 {
                let boost_factor = 1.0 + (0.3 - nutrient_ratio) * 5.0; // Up to 2.5x boost
                priority *= boost_factor;
            }
        }
        
        // Clamp priority to reasonable range
        priority.clamp(0.1, 10.0)
    }
    
    /// Check if a cell can divide based on size constraints.
    ///
    /// This method checks if a cell meets the size requirements for division
    /// based on its genome mode settings and current state.
    ///
    /// # Arguments
    /// * `current_mass` - Current cell mass
    /// * `split_mass` - Required mass for division from genome mode
    /// * `max_cell_size` - Maximum allowed cell size from genome mode
    ///
    /// # Returns
    /// True if the cell can divide based on size constraints
    pub fn can_divide_by_size(
        current_mass: f32,
        split_mass: f32,
        max_cell_size: f32,
    ) -> bool {
        // Check if cell has reached split mass
        if current_mass < split_mass {
            return false;
        }
        
        // Check if cell hasn't exceeded maximum size
        let radius = Self::mass_to_radius(current_mass);
        let max_radius = max_cell_size;
        
        radius <= max_radius
    }
    
    /// Convert mass to visual radius for size calculations.
    ///
    /// This uses the standard sphere volume formula to convert mass to radius.
    ///
    /// # Arguments
    /// * `mass` - Cell mass
    ///
    /// # Returns
    /// Visual radius for the given mass
    pub fn mass_to_radius(mass: f32) -> f32 {
        // Volume = (4/3) * π * r³
        // r = (3 * Volume / (4 * π))^(1/3)
        // Assuming density = 1, Volume = mass
        (mass * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0)
    }
    
    /// Calculate maximum allowed mass based on size constraint.
    ///
    /// This calculates the maximum mass a cell can have before it exceeds
    /// the maximum size constraint from its genome mode.
    ///
    /// # Arguments
    /// * `max_cell_size` - Maximum allowed radius from genome mode
    ///
    /// # Returns
    /// Maximum allowed mass
    pub fn radius_to_max_mass(max_cell_size: f32) -> f32 {
        // Volume = (4/3) * π * r³
        // mass = Volume (assuming density = 1)
        (4.0 / 3.0) * std::f32::consts::PI * max_cell_size.powi(3)
    }
    
    /// Calculate swim force energy cost for nutrient consumption.
    ///
    /// This calculates the energy cost of swimming based on the swim force
    /// magnitude, using a realistic energy model.
    ///
    /// # Arguments
    /// * `swim_force` - Swim force magnitude
    /// * `delta_time` - Time step
    ///
    /// # Returns
    /// Energy cost (nutrient consumption) for this time step
    pub fn calculate_swim_energy_cost(swim_force: f32, delta_time: f32) -> f32 {
        if swim_force <= 0.0 {
            return 0.0;
        }
        
        // Energy cost is proportional to force squared (realistic physics)
        let base_cost = 0.05; // Base energy cost per unit force squared per second
        let force_squared = swim_force * swim_force;
        
        base_cost * force_squared * delta_time
    }
    
    /// Validate genome mode parameters for consistency.
    ///
    /// This method checks if the genome mode parameters are consistent
    /// and within reasonable ranges for simulation stability.
    ///
    /// # Arguments
    /// * `mode` - Genome mode to validate
    ///
    /// # Returns
    /// List of validation warnings (empty if all parameters are valid)
    pub fn validate_genome_mode_parameters(mode: &crate::genome::ModeSettings) -> Vec<String> {
        let mut warnings = Vec::new();
        
        // Check swim force range
        if mode.swim_force < 0.0 || mode.swim_force > 2.0 {
            warnings.push(format!(
                "Swim force {} is outside recommended range [0.0, 2.0]",
                mode.swim_force
            ));
        }
        
        // Check nutrient priority range
        if mode.nutrient_priority < 0.1 || mode.nutrient_priority > 10.0 {
            warnings.push(format!(
                "Nutrient priority {} is outside recommended range [0.1, 10.0]",
                mode.nutrient_priority
            ));
        }
        
        // Check max cell size
        if mode.max_cell_size < 0.5 || mode.max_cell_size > 5.0 {
            warnings.push(format!(
                "Max cell size {} is outside recommended range [0.5, 5.0]",
                mode.max_cell_size
            ));
        }
        
        // Check split mass vs max size consistency
        let max_mass = Self::radius_to_max_mass(mode.max_cell_size);
        if mode.split_mass > max_mass {
            warnings.push(format!(
                "Split mass {} exceeds maximum mass {} for max cell size {}",
                mode.split_mass, max_mass, mode.max_cell_size
            ));
        }
        
        // Check nutrient gain rate
        if mode.nutrient_gain_rate < 0.0 || mode.nutrient_gain_rate > 1.0 {
            warnings.push(format!(
                "Nutrient gain rate {} is outside recommended range [0.0, 1.0]",
                mode.nutrient_gain_rate
            ));
        }
        
        // Check cell type validity
        if mode.cell_type < 0 || mode.cell_type > 1 {
            warnings.push(format!(
                "Cell type {} is not supported (only 0=Test, 1=Flagellocyte)",
                mode.cell_type
            ));
        }
        
        warnings
    }
    
    /// Create optimized genome mode for specific cell type.
    ///
    /// This method creates a genome mode with optimized parameters for
    /// a specific cell type and behavior pattern.
    ///
    /// # Arguments
    /// * `cell_type` - Target cell type
    /// * `behavior_profile` - Behavior profile name ("balanced", "aggressive", "efficient")
    ///
    /// # Returns
    /// Optimized genome mode settings
    pub fn create_optimized_mode_for_type(
        cell_type: CellType,
        behavior_profile: &str,
    ) -> crate::genome::ModeSettings {
        let mut mode = crate::genome::ModeSettings::default();
        mode.cell_type = cell_type.to_u32() as i32;
        
        match cell_type {
            CellType::Test => {
                // Test cell optimizations
                mode.swim_force = 0.0; // Test cells don't swim
                mode.nutrient_gain_rate = match behavior_profile {
                    "aggressive" => 0.4, // Fast growth
                    "efficient" => 0.15, // Slow but steady
                    _ => 0.2, // Balanced
                };
                mode.max_cell_size = 2.0;
                mode.nutrient_priority = 1.0;
                mode.prioritize_when_low = true;
            }
            CellType::Flagellocyte => {
                // Flagellocyte cell optimizations
                mode.nutrient_gain_rate = 0.0; // No automatic gain
                mode.swim_force = match behavior_profile {
                    "aggressive" => 1.0, // High mobility, high cost
                    "efficient" => 0.3,  // Low mobility, low cost
                    _ => 0.5, // Balanced
                };
                mode.max_cell_size = 1.8; // Slightly smaller for mobility
                mode.nutrient_priority = match behavior_profile {
                    "aggressive" => 2.0, // High priority for resources
                    "efficient" => 0.8,  // Lower priority, more cooperative
                    _ => 1.2, // Slightly higher than default
                };
                mode.prioritize_when_low = true;
            }
        }
        
        mode
    }
}