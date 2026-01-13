//! GPU-based instance buffer builder using compute shaders.
//!
//! Builds cell instance data on the GPU to eliminate CPU-side iteration
//! and reduce CPU→GPU data transfer. Includes frustum and occlusion culling.

use crate::cell::types::CellTypeVisuals;
use crate::genome::Genome;
use crate::simulation::CanonicalState;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Culling mode for the instance builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullingMode {
    /// No culling - all cells are rendered
    #[default]
    Disabled,
    /// Frustum culling only
    FrustumOnly,
    /// Occlusion culling only (requires Hi-Z texture)
    OcclusionOnly,
    /// Frustum + occlusion culling (requires Hi-Z texture)
    FrustumAndOcclusion,
}

impl CullingMode {
    fn as_u32(self) -> u32 {
        match self {
            CullingMode::Disabled => 0,
            CullingMode::FrustumOnly => 1,
            CullingMode::OcclusionOnly => 3,
            CullingMode::FrustumAndOcclusion => 2,
        }
    }
}

/// Culling statistics from the last frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct CullingStats {
    pub total_cells: u32,
    pub visible_cells: u32,
    pub frustum_culled: u32,
    pub occluded: u32,
}

/// GPU instance builder with compute shader pipeline.
pub struct InstanceBuilder {
    // Compute pipeline
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    // Uniform buffer
    params_buffer: wgpu::Buffer,
    
    // Input buffers (simulation data)
    positions_buffer: wgpu::Buffer,
    rotations_buffer: wgpu::Buffer,
    radii_buffer: wgpu::Buffer,
    mode_indices_buffer: wgpu::Buffer,
    cell_ids_buffer: wgpu::Buffer,
    genome_ids_buffer: wgpu::Buffer,
    cell_types_buffer: wgpu::Buffer,
    
    // Lookup table buffers
    mode_visuals_buffer: wgpu::Buffer,
    cell_type_visuals_buffer: wgpu::Buffer,
    mode_properties_buffer: wgpu::Buffer,
    /// Cell type per mode (lookup table: mode_cell_types[mode_index] = cell_type)
    /// Used by shader to derive cell_type from mode_index when genomes change
    mode_cell_types_buffer: wgpu::Buffer,
    
    // Output buffer (instance data for rendering)
    pub instance_buffer: wgpu::Buffer,
    
    // Indirect draw buffer for GPU-driven rendering (total visible count)
    pub indirect_buffer: wgpu::Buffer,
    
    // Per-type indirect draw buffers for multi-pipeline rendering
    // Test cells are in [0, capacity/2), Flagellocytes in [capacity/2, capacity)
    indirect_buffer_test: wgpu::Buffer,
    indirect_buffer_flagellocyte: wgpu::Buffer,
    
    // Counter buffer for atomic operations
    counters_buffer: wgpu::Buffer,
    counters_readback_buffer: wgpu::Buffer,
    
    // Current bind group (recreated when buffers change size)
    bind_group: Option<wgpu::BindGroup>,
    
    // Hi-Z texture for occlusion culling
    hiz_texture: Option<wgpu::Texture>,
    hiz_view: Option<wgpu::TextureView>,
    hiz_mip_count: u32,
    /// Whether external Hi-Z texture has been set up (to avoid recreating bind group every frame)
    hiz_configured: bool,
    
    // Capacity tracking
    cell_capacity: usize,
    mode_capacity: usize,
    cell_type_capacity: usize,
    
    // Dirty tracking
    positions_dirty: bool,
    rotations_dirty: bool,
    radii_dirty: bool,
    mode_indices_dirty: bool,
    cell_ids_dirty: bool,
    genome_ids_dirty: bool,
    cell_types_dirty: bool,
    mode_visuals_dirty: bool,
    cell_type_visuals_dirty: bool,
    
    // Temporary buffers for data conversion (reused to avoid allocations)
    temp_positions: Vec<[f32; 4]>,
    temp_rotations: Vec<[f32; 4]>,
    temp_mode_indices: Vec<u32>,
    temp_genome_ids: Vec<u32>,
    temp_cell_types: Vec<u32>,
    
    // Hash for change detection
    last_cell_count: usize,
    last_genome_count: usize,
    last_mode_hash: u64,
    last_visuals_hash: u64,
    
    // Culling settings
    culling_mode: CullingMode,
    last_stats: CullingStats,
    /// Occlusion bias: negative = more aggressive (cull more), positive = more conservative (cull less)
    occlusion_bias: f32,
    /// Mip level override: -1 = auto, 0+ = force specific mip level
    occlusion_mip_override: i32,
    /// Minimum screen-space size (pixels) to apply occlusion culling
    min_screen_size: f32,
    /// Don't cull objects closer than this distance
    min_distance: f32,
    
    // Last visible count for draw calls
    last_visible_count: u32,
    
    // Async stats readback state
    stats_map_pending: bool,
    stats_receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}


/// Frustum plane representation (normal + distance from origin).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrustumPlane {
    normal_and_dist: [f32; 4], // xyz = normal, w = distance
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BuildParams {
    cell_count: u32,
    mode_count: u32,
    cell_type_count: u32,
    culling_enabled: u32,
    // View-projection matrix (64 bytes)
    view_proj: [[f32; 4]; 4],
    // Camera position - vec3 in WGSL followed by f32 packs together
    camera_pos: [f32; 3],
    near_plane: f32,
    far_plane: f32,
    screen_width: f32,
    screen_height: f32,
    hiz_mip_count: u32,
    // Occlusion culling parameters
    occlusion_bias: f32,
    occlusion_mip_override: i32,
    min_screen_size: f32,
    min_distance: f32,
    // Camera focal length for LOD calculation
    focal_length: f32,
    // LOD parameters for configurable thresholds
    lod_scale_factor: f32,
    lod_threshold_low: f32,
    lod_threshold_medium: f32,
    lod_threshold_high: f32,
    // Debug colors flag for LOD visualization
    lod_debug_colors: u32,  // 0 = disabled, 1 = enabled
    // Padding to maintain 16-byte alignment (reduced by 1 due to debug flag)
    _padding: [f32; 2],
    // Frustum planes array needs 16-byte alignment
    frustum_planes: [FrustumPlane; 6],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ModeVisuals {
    color: [f32; 4],      // xyz = color, w = 1.0 (always opaque)
    emissive_pad: [f32; 4], // x = emissive, yzw = padding
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuCellTypeVisuals {
    specular_strength: f32,
    specular_power: f32,
    fresnel_strength: f32,
    membrane_noise_scale: f32,
    membrane_noise_strength: f32,
    membrane_noise_speed: f32,
    // Flagella parameters (used by Flagellocyte cell type)
    tail_length: f32,
    tail_thickness: f32,
    tail_amplitude: f32,
    tail_frequency: f32,
    // tail_speed removed - now calculated from swim_force in shader
    tail_taper: f32,
    tail_segments: f32,
    _pad: [f32; 4], // Padding to 64 bytes (16 floats)
}

/// Per-cell instance data for GPU rendering.
///
/// Uses a unified layout with reserved fields for type-specific data.
/// All cell types share this structure, with type_data interpreted
/// differently based on cell type.
///
/// # Layout (96 bytes, 16-byte aligned)
///
/// Common fields (64 bytes):
/// - position: [f32; 3] (12 bytes) - World position
/// - radius: f32 (4 bytes) - Cell radius
/// - color: [f32; 4] (16 bytes) - RGBA color from mode
/// - visual_params: [f32; 4] (16 bytes) - specular, power, fresnel, emissive
/// - rotation: [f32; 4] (16 bytes) - Quaternion rotation
///
/// Type-specific reserved fields (32 bytes):
/// - type_data: [f32; 8] - Reserved for type-specific use
///   - type_data[0]: membrane_noise_scale (Test), flagella_angle (Flagellocyte)
///   - type_data[1]: membrane_noise_strength (Test), flagella_speed (Flagellocyte)
///   - type_data[2]: membrane_noise_speed (Test), sensor_direction_x (Neurocyte)
///   - type_data[3]: membrane_anim_offset (Test), sensor_direction_y (Neurocyte)
///   - type_data[4]: sensor_direction_z (Neurocyte)
///   - type_data[5-7]: reserved for future use
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CellInstance {
    /// World position (xyz)
    pub position: [f32; 3],
    /// Cell radius
    pub radius: f32,
    /// RGBA color from mode settings
    pub color: [f32; 4],
    /// Visual parameters: specular_strength, specular_power, fresnel_strength, emissive
    pub visual_params: [f32; 4],
    /// Rotation quaternion (x, y, z, w)
    pub rotation: [f32; 4],
    /// Type-specific data (8 floats, 32 bytes)
    /// Interpretation depends on cell type - see struct documentation
    pub type_data: [f32; 8],
}

// Compile-time assertions to verify struct layout matches shader expectations
const _: () = assert!(
    std::mem::size_of::<CellInstance>() == 96,
    "CellInstance must be exactly 96 bytes to match shader layout"
);
const _: () = assert!(
    std::mem::align_of::<CellInstance>() <= 16,
    "CellInstance must be at most 16-byte aligned for GPU compatibility"
);


impl InstanceBuilder {
    /// Create a new instance builder with the given capacity.
    pub fn new(device: &wgpu::Device, cell_capacity: usize) -> Self {
        // Support up to 100 genomes with 40 modes each = 4000 modes
        let mode_capacity = 4096;
        let cell_type_capacity = 16;
        
        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Instance Builder Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/build_instances.wgsl").into()),
        });
        
        // Create bind group layout with Hi-Z texture support
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instance Builder Bind Group Layout"),
            entries: &[
                // params uniform (binding 0)
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
                // positions (binding 1)
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
                // rotations (binding 2)
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
                // radii (binding 3)
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
                // mode_indices (binding 4)
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
                // cell_ids (binding 5)
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
                // genome_ids (binding 6)
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
                // mode_visuals (binding 7)
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
                // cell_type_visuals (binding 8)
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
                // instances output (binding 9)
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
                // counters (binding 10)
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
                // Hi-Z texture (binding 11) - R32Float is not filterable
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // GPU cell count buffer (binding 12) - [0] = total cells, [1] = live cells
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // cell_types (binding 13) - cell type per cell (0 = Test, 1 = Flagellocyte, etc.)
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
                // mode_properties (binding 14) - per-mode properties including swim_force
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // mode_cell_types (binding 15) - cell type per mode (lookup table)
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
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
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instance Builder Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Instance Builder Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Params"),
            size: std::mem::size_of::<BuildParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let positions_buffer = Self::create_storage_buffer(device, "Positions", cell_capacity * 16);
        let rotations_buffer = Self::create_storage_buffer(device, "Rotations", cell_capacity * 16);
        let radii_buffer = Self::create_storage_buffer(device, "Radii", cell_capacity * 4);
        let mode_indices_buffer = Self::create_storage_buffer(device, "Mode Indices", cell_capacity * 4);
        let cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", cell_capacity * 4);
        let genome_ids_buffer = Self::create_storage_buffer(device, "Genome IDs", cell_capacity * 4);
        let cell_types_buffer = Self::create_storage_buffer(device, "Cell Types", cell_capacity * 4);
        
        let mode_visuals_buffer = Self::create_storage_buffer(device, "Mode Visuals", mode_capacity * std::mem::size_of::<ModeVisuals>());
        let cell_type_visuals_buffer = Self::create_storage_buffer(device, "Cell Type Visuals", cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>());
        // Mode properties: 12 floats per mode (48 bytes) - nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, padding x3
        let mode_properties_buffer = Self::create_storage_buffer(device, "Mode Properties", mode_capacity * 48);
        // Mode cell types: 1 u32 per mode - lookup table for deriving cell_type from mode_index
        let mode_cell_types_buffer = Self::create_storage_buffer(device, "Mode Cell Types", mode_capacity * 4);
        
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (cell_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Indirect draw buffer: vertex_count, instance_count, first_vertex, first_instance
        // Main buffer for total visible count (legacy, kept for compatibility)
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Indirect"),
            size: 16, // 4 u32s
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Per-type indirect draw buffers for multi-pipeline rendering
        // Test cells (type 0): first_instance = 0
        let indirect_buffer_test = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Indirect Test"),
            size: 16, // 4 u32s
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Flagellocyte cells (type 1): first_instance = capacity/2
        let indirect_buffer_flagellocyte = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Indirect Flagellocyte"),
            size: 16, // 4 u32s
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // 6 counters: visible, total, frustum_culled, occluded, test_count, flagellocyte_count
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Counters"),
            size: 24, // 6 u32s
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counters Readback"),
            size: 24, // 6 u32s
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            positions_buffer,
            rotations_buffer,
            radii_buffer,
            mode_indices_buffer,
            cell_ids_buffer,
            genome_ids_buffer,
            cell_types_buffer,
            mode_visuals_buffer,
            cell_type_visuals_buffer,
            mode_properties_buffer,
            mode_cell_types_buffer,
            instance_buffer,
            indirect_buffer,
            indirect_buffer_test,
            indirect_buffer_flagellocyte,
            counters_buffer,
            counters_readback_buffer,
            bind_group: None,
            hiz_texture: None,
            hiz_view: None,
            hiz_mip_count: 0,
            hiz_configured: false,
            cell_capacity,
            mode_capacity,
            cell_type_capacity,
            positions_dirty: true,
            rotations_dirty: true,
            radii_dirty: true,
            mode_indices_dirty: true,
            cell_ids_dirty: true,
            genome_ids_dirty: true,
            cell_types_dirty: true,
            mode_visuals_dirty: true,
            cell_type_visuals_dirty: true,
            last_cell_count: 0,
            last_genome_count: 0,
            last_mode_hash: 0,
            last_visuals_hash: 0,
            culling_mode: CullingMode::FrustumOnly,
            last_stats: CullingStats::default(),
            occlusion_bias: 0.0,
            occlusion_mip_override: -1,
            min_screen_size: 0.0,
            min_distance: 0.0,
            last_visible_count: 0,
            stats_map_pending: false,
            stats_receiver: None,
            temp_positions: Vec::new(),
            temp_rotations: Vec::new(),
            temp_mode_indices: Vec::new(),
            temp_genome_ids: Vec::new(),
            temp_cell_types: Vec::new(),
        }
    }
    
    fn create_storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Instance Builder {}", label)),
            size: size.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Set the culling mode.
    pub fn set_culling_mode(&mut self, mode: CullingMode) {
        self.culling_mode = mode;
    }
    
    /// Get the current culling mode.
    pub fn culling_mode(&self) -> CullingMode {
        self.culling_mode
    }
    
    /// Get culling statistics from the last frame.
    pub fn culling_stats(&self) -> CullingStats {
        self.last_stats
    }
    
    /// Set the occlusion bias.
    /// Negative values = more aggressive culling (cull more cells).
    /// Positive values = more conservative culling (cull fewer cells).
    /// Range: typically -0.1 to 0.1
    pub fn set_occlusion_bias(&mut self, bias: f32) {
        self.occlusion_bias = bias;
    }
    
    /// Get the current occlusion bias.
    pub fn occlusion_bias(&self) -> f32 {
        self.occlusion_bias
    }
    
    /// Set the mip level override for occlusion culling.
    /// -1 = auto (use mip 0), 0+ = force specific mip level
    pub fn set_occlusion_mip_override(&mut self, mip: i32) {
        self.occlusion_mip_override = mip;
    }
    
    /// Get the current mip level override.
    pub fn occlusion_mip_override(&self) -> i32 {
        self.occlusion_mip_override
    }
    
    /// Set the minimum screen-space size (in pixels) for occlusion culling.
    /// Objects smaller than this won't be culled.
    pub fn set_min_screen_size(&mut self, size: f32) {
        self.min_screen_size = size;
    }
    
    /// Get the minimum screen-space size.
    pub fn min_screen_size(&self) -> f32 {
        self.min_screen_size
    }
    
    /// Set the minimum distance for occlusion culling.
    /// Objects closer than this won't be culled.
    pub fn set_min_distance(&mut self, distance: f32) {
        self.min_distance = distance;
    }
    
    /// Get the minimum distance.
    pub fn min_distance(&self) -> f32 {
        self.min_distance
    }
    
    /// Get the number of visible instances from the last build.
    pub fn visible_count(&self) -> u32 {
        self.last_visible_count
    }
    
    /// Mark all buffers as dirty (forces full update).
    pub fn mark_all_dirty(&mut self) {
        self.positions_dirty = true;
        self.rotations_dirty = true;
        self.radii_dirty = true;
        self.mode_indices_dirty = true;
        self.cell_ids_dirty = true;
        self.genome_ids_dirty = true;
        self.cell_types_dirty = true;
        self.mode_visuals_dirty = true;
        self.cell_type_visuals_dirty = true;
    }
    
    /// Mark positions buffer as dirty (call when cell positions change).
    pub fn mark_positions_dirty(&mut self) {
        self.positions_dirty = true;
    }
    
    /// Clear positions dirty flag (call after GPU-to-GPU copy of positions).
    pub fn clear_positions_dirty(&mut self) {
        self.positions_dirty = false;
    }
    
    /// Mark rotations buffer as dirty (call when cell rotations change).
    pub fn mark_rotations_dirty(&mut self) {
        self.rotations_dirty = true;
    }
    
    /// Mark radii buffer as dirty (call when cell masses/radii change).
    pub fn mark_radii_dirty(&mut self) {
        self.radii_dirty = true;
    }
    
    /// Mark mode indices buffer as dirty (call when cell modes change).
    pub fn mark_mode_indices_dirty(&mut self) {
        self.mode_indices_dirty = true;
    }
    
    /// Mark cell types buffer as dirty (call when mode cell_type settings change).
    pub fn mark_cell_types_dirty(&mut self) {
        self.cell_types_dirty = true;
    }
    
    /// Create or resize the Hi-Z texture for occlusion culling.
    pub fn create_hiz_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Calculate mip count
        let max_dim = width.max(height);
        let mip_count = (max_dim as f32).log2().floor() as u32 + 1;
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hi-Z Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                | wgpu::TextureUsages::RENDER_ATTACHMENT 
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.hiz_texture = Some(texture);
        self.hiz_view = Some(view);
        self.hiz_mip_count = mip_count;
        self.bind_group = None; // Force bind group recreation
    }
    
    /// Set an external Hi-Z texture for occlusion culling.
    /// Use this when the Hi-Z texture is generated by HizGenerator.
    /// Only recreates bind group if the Hi-Z texture actually changed.
    pub fn set_hiz_texture(&mut self, device: &wgpu::Device, hiz_view: &wgpu::TextureView, mip_count: u32, cell_count_buffer: &wgpu::Buffer) {
        // Only recreate bind group if mip count changed (indicates texture resize)
        // or if Hi-Z hasn't been configured yet
        if !self.hiz_configured || self.hiz_mip_count != mip_count || self.bind_group.is_none() {
            self.hiz_mip_count = mip_count;
            self.hiz_configured = true;
            self.bind_group = None;
            self.recreate_bind_group_with_hiz(device, hiz_view, cell_count_buffer);
        }
    }
    
    /// Reset Hi-Z configuration (call on resize).
    pub fn reset_hiz(&mut self) {
        self.hiz_configured = false;
        self.bind_group = None;
    }
    
    fn recreate_bind_group_with_hiz(&mut self, device: &wgpu::Device, hiz_view: &wgpu::TextureView, cell_count_buffer: &wgpu::Buffer) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance Builder Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rotations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mode_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cell_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.genome_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.mode_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cell_type_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: cell_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.cell_types_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.mode_properties_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.mode_cell_types_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Update input buffers from simulation state.
    pub fn update_from_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        genomes: &[Genome],
        cell_type_visuals: Option<&[CellTypeVisuals]>,
    ) {
        let cell_count = state.cell_count;
        
        // Check if we need to resize buffers
        if cell_count > self.cell_capacity {
            self.resize_cell_buffers(device, cell_count * 2);
        }
        
        // Detect changes via cell count - mark non-position buffers dirty
        // Positions come from GPU physics, so don't mark them dirty here
        if cell_count != self.last_cell_count {
            // Don't call mark_all_dirty() - positions come from GPU
            self.rotations_dirty = true;
            self.radii_dirty = true;
            self.mode_indices_dirty = true;
            self.cell_ids_dirty = true;
            self.genome_ids_dirty = true;
            self.cell_types_dirty = true;
            self.last_cell_count = cell_count;
        }
        
        if cell_count == 0 {
            return;
        }
        
        // Only update buffers that are marked as dirty
        // NOTE: positions_dirty is only set by mark_all_dirty() which is called
        // on resize or initial setup. During normal operation, positions come
        // from GPU physics via copy_buffer_to_buffer.
        if self.positions_dirty {
            // Update positions (convert Vec3 to vec4 for GPU alignment)
            // Reuse temporary buffer to avoid allocations
            self.temp_positions.clear();
            self.temp_positions.reserve(cell_count);
            for i in 0..cell_count {
                let p = state.positions[i];
                self.temp_positions.push([p.x, p.y, p.z, 0.0]);
            }
            queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&self.temp_positions));
            self.positions_dirty = false;
        }
        
        if self.rotations_dirty {
            // Update rotations
            // Reuse temporary buffer to avoid allocations
            self.temp_rotations.clear();
            self.temp_rotations.reserve(cell_count);
            for i in 0..cell_count {
                let q = state.rotations[i];
                self.temp_rotations.push([q.x, q.y, q.z, q.w]);
            }
            queue.write_buffer(&self.rotations_buffer, 0, bytemuck::cast_slice(&self.temp_rotations));
            self.rotations_dirty = false;
        }
        
        if self.radii_dirty {
            // Update radii
            queue.write_buffer(&self.radii_buffer, 0, bytemuck::cast_slice(&state.radii[..cell_count]));
            self.radii_dirty = false;
        }
        
        if self.mode_indices_dirty {
            // Update mode indices - now includes genome offset
            // Each cell's mode_index is offset by its genome's mode_offset
            let genome_mode_offsets: Vec<usize> = {
                let mut offsets = Vec::with_capacity(genomes.len());
                let mut offset = 0usize;
                for genome in genomes {
                    offsets.push(offset);
                    offset += genome.modes.len();
                }
                offsets
            };
        
            // Reuse temporary buffer to avoid allocations
            self.temp_mode_indices.clear();
            self.temp_mode_indices.reserve(cell_count);
            for i in 0..cell_count {
                let mode_idx = state.mode_indices[i];
                let genome_id = state.genome_ids[i];
                let offset = genome_mode_offsets.get(genome_id).copied().unwrap_or(0);
                self.temp_mode_indices.push((offset + mode_idx) as u32);
            }
            
            queue.write_buffer(&self.mode_indices_buffer, 0, bytemuck::cast_slice(&self.temp_mode_indices));
            self.mode_indices_dirty = false;
        }
        
        if self.cell_ids_dirty {
            // Update cell IDs
            queue.write_buffer(&self.cell_ids_buffer, 0, bytemuck::cast_slice(&state.cell_ids[..cell_count]));
            self.cell_ids_dirty = false;
        }
        
        if self.genome_ids_dirty {
            // Update genome IDs
            // Reuse temporary buffer to avoid allocations
            self.temp_genome_ids.clear();
            self.temp_genome_ids.reserve(cell_count);
            for i in 0..cell_count {
                self.temp_genome_ids.push(state.genome_ids[i] as u32);
            }
            queue.write_buffer(&self.genome_ids_buffer, 0, bytemuck::cast_slice(&self.temp_genome_ids));
            self.genome_ids_dirty = false;
        }
        
        if self.cell_types_dirty {
            // Update cell types from mode settings
            // Each cell's type is determined by its genome's mode settings
            self.temp_cell_types.clear();
            self.temp_cell_types.reserve(cell_count);
            for i in 0..cell_count {
                let genome_id = state.genome_ids[i];
                let mode_idx = state.mode_indices[i];
                let cell_type = if genome_id < genomes.len() {
                    let genome = &genomes[genome_id];
                    if mode_idx < genome.modes.len() {
                        genome.modes[mode_idx].cell_type as u32
                    } else {
                        0 // Default to Test cell type
                    }
                } else {
                    0 // Default to Test cell type
                };
                self.temp_cell_types.push(cell_type);
            }
            queue.write_buffer(&self.cell_types_buffer, 0, bytemuck::cast_slice(&self.temp_cell_types));
            self.cell_types_dirty = false;
        }

        // Update mode visuals from all genomes (combined into single buffer)
        // Force update if genome count changed (new genome added)
        let genome_count = genomes.len();
        let genome_count_changed = genome_count != self.last_genome_count;
        if genome_count_changed {
            self.last_genome_count = genome_count;
        }
        
        if !genomes.is_empty() {
            let mode_hash = Self::hash_all_genome_modes(genomes);
            if mode_hash != self.last_mode_hash || genome_count_changed {
                self.update_mode_visuals_from_genomes(device, queue, genomes);
                self.last_mode_hash = mode_hash;
            }
        }
        
        // Update cell type visuals
        if let Some(visuals) = cell_type_visuals {
            let visuals_hash = Self::hash_cell_type_visuals(visuals);
            if visuals_hash != self.last_visuals_hash {
                self.update_cell_type_visuals(device, queue, visuals);
                self.last_visuals_hash = visuals_hash;
            }
        }
        
        // Note: bind group will be created lazily in build_instances_with_encoder
        // when cell_count_buffer is available
    }
    
    fn hash_all_genome_modes(genomes: &[Genome]) -> u64 {
        let mut hash = genomes.len() as u64;
        for (genome_idx, genome) in genomes.iter().enumerate() {
            // Include genome index to differentiate order
            hash = hash.wrapping_mul(31).wrapping_add(genome_idx as u64);
            hash = hash.wrapping_mul(31).wrapping_add(genome.modes.len() as u64);
            for (mode_idx, mode) in genome.modes.iter().enumerate() {
                // Include mode index for uniqueness
                hash = hash.wrapping_mul(31).wrapping_add(mode_idx as u64);
                hash = hash.wrapping_mul(31).wrapping_add((mode.color.x * 1000.0) as u64);
                hash = hash.wrapping_mul(31).wrapping_add((mode.color.y * 1000.0) as u64);
                hash = hash.wrapping_mul(31).wrapping_add((mode.color.z * 1000.0) as u64);
                // Skip opacity since cells are always opaque
                hash = hash.wrapping_mul(31).wrapping_add((mode.emissive * 1000.0) as u64);
            }
        }
        hash
    }
    
    fn hash_cell_type_visuals(visuals: &[CellTypeVisuals]) -> u64 {
        let mut hash = visuals.len() as u64;
        for v in visuals {
            hash = hash.wrapping_mul(31).wrapping_add((v.specular_strength * 1000.0) as u64);
            // Skip noise parameters since they're disabled
        }
        hash
    }

    pub fn update_mode_visuals_from_genomes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, genomes: &[Genome]) {
        // Calculate total mode count across all genomes
        let total_mode_count: usize = genomes.iter().map(|g| g.modes.len()).sum();
        
        if total_mode_count > self.mode_capacity {
            self.mode_capacity = total_mode_count * 2;
            self.mode_visuals_buffer = Self::create_storage_buffer(
                device,
                "Mode Visuals",
                self.mode_capacity * std::mem::size_of::<ModeVisuals>(),
            );
            // Also resize mode_cell_types_buffer
            self.mode_cell_types_buffer = Self::create_storage_buffer(
                device,
                "Mode Cell Types",
                self.mode_capacity * 4,
            );
        }
        
        // Always invalidate bind group when mode visuals change
        // This ensures the shader sees the updated buffer data
        self.bind_group = None;
        
        // Build combined mode visuals from all genomes
        let mode_visuals: Vec<ModeVisuals> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| {
                    ModeVisuals {
                        color: [mode.color.x, mode.color.y, mode.color.z, 1.0], // Always opaque
                        emissive_pad: [mode.emissive, 0.0, 0.0, 0.0],
                    }
                })
            })
            .collect();
        
        if !mode_visuals.is_empty() {
            queue.write_buffer(&self.mode_visuals_buffer, 0, bytemuck::cast_slice(&mode_visuals));
        }
        
        // Build mode_cell_types lookup table (cell_type per mode)
        // This allows the shader to derive cell_type from mode_index
        let mode_cell_types: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| {
                genome.modes.iter().map(|mode| mode.cell_type as u32)
            })
            .collect();
        
        if !mode_cell_types.is_empty() {
            queue.write_buffer(&self.mode_cell_types_buffer, 0, bytemuck::cast_slice(&mode_cell_types));
        }
    }
    
    fn update_cell_type_visuals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, visuals: &[CellTypeVisuals]) {
        let count = visuals.len();
        
        if count > self.cell_type_capacity {
            self.cell_type_capacity = count * 2;
            self.cell_type_visuals_buffer = Self::create_storage_buffer(
                device,
                "Cell Type Visuals",
                self.cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>(),
            );
            self.bind_group = None;
        }
        
        let gpu_visuals: Vec<GpuCellTypeVisuals> = visuals
            .iter()
            .map(|v| GpuCellTypeVisuals {
                specular_strength: v.specular_strength,
                specular_power: v.specular_power,
                fresnel_strength: v.fresnel_strength,
                membrane_noise_scale: 0.0,        // Noise disabled
                membrane_noise_strength: 0.0,     // Noise disabled
                membrane_noise_speed: 0.0,        // Noise disabled
                // Flagella parameters (used by Flagellocyte cell type)
                tail_length: v.tail_length,
                tail_thickness: v.tail_thickness,
                tail_amplitude: v.tail_amplitude,
                tail_frequency: v.tail_frequency,
                // tail_speed removed - calculated from swim_force in shader
                tail_taper: v.tail_taper,
                tail_segments: v.tail_segments,
                _pad: [0.0; 4],
            })
            .collect();
        
        if !gpu_visuals.is_empty() {
            queue.write_buffer(&self.cell_type_visuals_buffer, 0, bytemuck::cast_slice(&gpu_visuals));
        }
    }
    
    /// Update cell type visuals directly (public version for GPU scene).
    pub fn update_cell_type_visuals_direct(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, visuals: &[CellTypeVisuals]) {
        self.update_cell_type_visuals(device, queue, visuals);
    }

    fn resize_cell_buffers(&mut self, device: &wgpu::Device, new_capacity: usize) {
        self.cell_capacity = new_capacity;
        
        self.positions_buffer = Self::create_storage_buffer(device, "Positions", new_capacity * 16);
        self.rotations_buffer = Self::create_storage_buffer(device, "Rotations", new_capacity * 16);
        self.radii_buffer = Self::create_storage_buffer(device, "Radii", new_capacity * 4);
        self.mode_indices_buffer = Self::create_storage_buffer(device, "Mode Indices", new_capacity * 4);
        self.cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", new_capacity * 4);
        self.genome_ids_buffer = Self::create_storage_buffer(device, "Genome IDs", new_capacity * 4);
        self.cell_types_buffer = Self::create_storage_buffer(device, "Cell Types", new_capacity * 4);
        
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (new_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        self.bind_group = None;
        self.mark_all_dirty();
    }
    
    /// Create a dummy 1x1 Hi-Z texture for when occlusion culling is disabled.
    fn create_dummy_hiz_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Hi-Z Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn recreate_bind_group(&mut self, device: &wgpu::Device, cell_count_buffer: &wgpu::Buffer) {
        // Use existing Hi-Z texture or create a dummy one
        let (_dummy_texture, dummy_view);
        let hiz_view = if let Some(ref view) = self.hiz_view {
            view
        } else {
            (_dummy_texture, dummy_view) = Self::create_dummy_hiz_texture(device);
            &dummy_view
        };
        
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance Builder Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rotations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mode_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cell_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.genome_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.mode_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cell_type_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: cell_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.cell_types_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.mode_properties_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.mode_cell_types_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Extract frustum planes from a view-projection matrix.
    /// Note: The shader now uses a simpler clip-space method, but we still
    /// populate this for potential future use or debugging.
    fn extract_frustum_planes(_view_proj: Mat4) -> [FrustumPlane; 6] {
        // Return zeroed planes - the shader uses clip-space culling instead
        [FrustumPlane { normal_and_dist: [0.0; 4] }; 6]
    }
    
    /// Run the compute shader to build instance data with culling.
    /// Uses an external encoder to allow batching with other GPU work.
    /// cell_capacity: Maximum number of cells (used for dispatch)
    /// cell_count_buffer: GPU buffer containing [total_cells, live_cells]
    pub fn build_instances_with_encoder(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cell_capacity: usize,
        mode_count: usize,
        cell_type_count: usize,
        view_proj: Mat4,
        camera_pos: Vec3,
        screen_width: u32,
        screen_height: u32,
        cell_count_buffer: &wgpu::Buffer,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        lod_debug_colors: bool,
    ) {
        if cell_capacity == 0 {
            self.last_visible_count = 0;
            return;
        }
        
        // Extract frustum planes
        let frustum_planes = Self::extract_frustum_planes(view_proj);
        
        // Update params - cell_count is now read from GPU buffer by shader
        // We pass cell_capacity here for the shader to use as upper bound
        let params = BuildParams {
            cell_count: cell_capacity as u32, // Used as dispatch upper bound, actual count from GPU buffer
            mode_count: mode_count as u32,
            cell_type_count: cell_type_count as u32,
            culling_enabled: self.culling_mode.as_u32(),
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            near_plane: 0.1,
            far_plane: 1000.0,
            screen_width: screen_width as f32,
            screen_height: screen_height as f32,
            hiz_mip_count: self.hiz_mip_count,
            occlusion_bias: self.occlusion_bias,
            occlusion_mip_override: self.occlusion_mip_override,
            min_screen_size: self.min_screen_size,
            min_distance: self.min_distance,
            // Calculate focal length from projection matrix for LOD calculation
            focal_length: {
                // For perspective projection, focal_length = projection[1][1] * screen_height / 2
                // This gives the focal length in pixels
                let proj_y = view_proj.y_axis.y;
                proj_y * screen_height as f32 * 0.5
            },
            // LOD parameters for configurable thresholds
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
            // Debug colors flag for LOD visualization
            lod_debug_colors: if lod_debug_colors { 1 } else { 0 },
            // Padding to maintain 16-byte alignment (reduced by 1 due to debug flag)
            _padding: [0.0, 0.0],
            frustum_planes,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        
        // Clear all 6 counters: visible, total, frustum_culled, occluded, test_count, flagellocyte_count
        let zero_counters: [u32; 6] = [0, 0, 0, 0, 0, 0];
        queue.write_buffer(&self.counters_buffer, 0, bytemuck::cast_slice(&zero_counters));
        
        // Initialize indirect draw buffer (legacy, total visible count)
        // vertex_count=4 (triangle strip quad), instance_count=0, first_vertex=0, first_instance=0
        let indirect_data: [u32; 4] = [4, 0, 0, 0];
        queue.write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&indirect_data));
        
        // Initialize per-type indirect draw buffers
        // Instances are sorted by type: Test cells at [0, test_count), Flagellocytes at [capacity/2, capacity/2 + flagellocyte_count)
        // Test cells: first_instance = 0
        let indirect_test: [u32; 4] = [4, 0, 0, 0];
        queue.write_buffer(&self.indirect_buffer_test, 0, bytemuck::cast_slice(&indirect_test));
        
        // Flagellocyte cells: first_instance = capacity/2 (fixed offset for sorted layout)
        let flagellocyte_first_instance = (cell_capacity / 2) as u32;
        let indirect_flagellocyte: [u32; 4] = [4, 0, 0, flagellocyte_first_instance];
        queue.write_buffer(&self.indirect_buffer_flagellocyte, 0, bytemuck::cast_slice(&indirect_flagellocyte));

        // Ensure bind group exists with cell_count_buffer
        if self.bind_group.is_none() {
            self.recreate_bind_group(device, cell_count_buffer);
        }
        
        let bind_group = self.bind_group.as_ref().unwrap();
        
        // Run compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Instance Builder Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            
            // Dispatch for full capacity - shader reads actual cell_count from GPU buffer
            let workgroup_count = (cell_capacity as u32 + 127) / 128;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // Copy visible count (counters[0]) to legacy indirect buffer's instance_count field (offset 4)
        encoder.copy_buffer_to_buffer(
            &self.counters_buffer,
            0,  // counters[0] = visible count
            &self.indirect_buffer,
            4,  // instance_count field offset
            4,  // 4 bytes (u32)
        );
        
        // Copy Test cell count (counters[4]) to Test indirect buffer's instance_count field
        encoder.copy_buffer_to_buffer(
            &self.counters_buffer,
            16, // counters[4] = test count (4 * 4 bytes offset)
            &self.indirect_buffer_test,
            4,  // instance_count field offset
            4,  // 4 bytes (u32)
        );
        
        // Copy Flagellocyte count (counters[5]) to Flagellocyte indirect buffer's instance_count field
        encoder.copy_buffer_to_buffer(
            &self.counters_buffer,
            20, // counters[5] = flagellocyte count (5 * 4 bytes offset)
            &self.indirect_buffer_flagellocyte,
            4,  // instance_count field offset
            4,  // 4 bytes (u32)
        );
        
        // Copy counters to readback buffer for stats (only if not currently mapped)
        if !self.stats_map_pending {
            encoder.copy_buffer_to_buffer(
                &self.counters_buffer,
                0,
                &self.counters_readback_buffer,
                0,
                24, // 6 u32s
            );
        }
        
        // For stats display, use capacity (actual count is in GPU buffer)
        self.last_visible_count = cell_capacity as u32;
        self.last_stats = CullingStats {
            total_cells: cell_capacity as u32,
            visible_cells: cell_capacity as u32,
            frustum_culled: 0,
            occluded: 0,
        };
    }
    
    /// Run the compute shader to build instance data with culling.
    /// Creates its own encoder and submits - use build_instances_with_encoder for batching.
    #[allow(dead_code)]
    pub fn build_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell_capacity: usize,
        mode_count: usize,
        cell_type_count: usize,
        view_proj: Mat4,
        camera_pos: Vec3,
        screen_width: u32,
        screen_height: u32,
        cell_count_buffer: &wgpu::Buffer,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        lod_debug_colors: bool,
    ) {
        // Ensure bind group exists before creating encoder
        if self.bind_group.is_none() {
            self.recreate_bind_group(device, cell_count_buffer);
        }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Instance Builder Encoder"),
        });
        
        self.build_instances_with_encoder(
            device,
            &mut encoder,
            queue,
            cell_capacity,
            mode_count,
            cell_type_count,
            view_proj,
            camera_pos,
            screen_width,
            screen_height,
            cell_count_buffer,
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
            lod_debug_colors,
        );
        
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Read back culling statistics (call after GPU work is done).
    /// This is a blocking operation - use sparingly.
    #[allow(dead_code)]
    pub fn read_culling_stats_blocking(&mut self, device: &wgpu::Device) -> CullingStats {
        let buffer_slice = self.counters_readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        // Poll the device in a blocking manner
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let counters: &[u32] = bytemuck::cast_slice(&data);
            
            self.last_stats = CullingStats {
                visible_cells: counters[0],
                total_cells: counters[1],
                frustum_culled: counters[2],
                occluded: counters[3],
            };
            self.last_visible_count = counters[0];
            
            drop(data);
            self.counters_readback_buffer.unmap();
        }
        
        self.last_stats
    }
    
    /// Start an async read of culling statistics.
    /// Call poll_culling_stats() to check if the read is complete.
    /// This is non-blocking and won't cause frame spikes.
    pub fn start_culling_stats_read(&mut self) {
        // Don't start a new read if one is already pending
        if self.stats_map_pending {
            return;
        }
        
        let buffer_slice = self.counters_readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        self.stats_map_pending = true;
        self.stats_receiver = Some(rx);
    }
    
    /// Poll for async culling stats read completion.
    /// Returns true if new stats are available, false if still pending or no read started.
    /// This is non-blocking.
    pub fn poll_culling_stats(&mut self, device: &wgpu::Device) -> bool {
        if !self.stats_map_pending {
            return false;
        }
        
        // Do a non-blocking poll to push GPU work forward
        let _ = device.poll(wgpu::PollType::Poll);
        
        // Check if the map operation completed
        if let Some(ref rx) = self.stats_receiver {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    // Map succeeded, read the data
                    let buffer_slice = self.counters_readback_buffer.slice(..);
                    let data = buffer_slice.get_mapped_range();
                    let counters: &[u32] = bytemuck::cast_slice(&data);
                    
                    self.last_stats = CullingStats {
                        visible_cells: counters[0],
                        total_cells: counters[1],
                        frustum_culled: counters[2],
                        occluded: counters[3],
                    };
                    self.last_visible_count = counters[0];
                    
                    drop(data);
                    self.counters_readback_buffer.unmap();
                    
                    self.stats_map_pending = false;
                    self.stats_receiver = None;
                    return true;
                }
                Ok(Err(_)) => {
                    // Map failed, reset state
                    self.stats_map_pending = false;
                    self.stats_receiver = None;
                    return false;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still pending
                    return false;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Channel closed unexpectedly
                    self.stats_map_pending = false;
                    self.stats_receiver = None;
                    return false;
                }
            }
        }
        
        false
    }
    
    /// Get the last known culling stats (may be from a previous frame).
    pub fn last_culling_stats(&self) -> CullingStats {
        self.last_stats
    }
    
    /// Get the instance buffer for rendering.
    pub fn get_instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }
    
    /// Get the current cell capacity.
    pub fn cell_capacity(&self) -> usize {
        self.cell_capacity
    }
    
    /// Get the indirect draw buffer for GPU-driven rendering.
    pub fn get_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_buffer
    }
    
    /// Get the indirect draw buffer for Test cells (type 0).
    /// Test cells are stored at instance indices [0, capacity/2).
    pub fn get_indirect_buffer_test(&self) -> &wgpu::Buffer {
        &self.indirect_buffer_test
    }
    
    /// Get the indirect draw buffer for Flagellocyte cells (type 1).
    /// Flagellocyte cells are stored at instance indices [capacity/2, capacity).
    pub fn get_indirect_buffer_flagellocyte(&self) -> &wgpu::Buffer {
        &self.indirect_buffer_flagellocyte
    }
    
    /// Get current cell capacity.
    pub fn capacity(&self) -> usize {
        self.cell_capacity
    }
    
    /// Get the positions buffer for external GPU-to-GPU copies.
    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buffer
    }
    
    /// Get the rotations buffer for external GPU-to-GPU copies.
    pub fn rotations_buffer(&self) -> &wgpu::Buffer {
        &self.rotations_buffer
    }
    
    /// Get the mode indices buffer for external GPU-to-GPU copies.
    pub fn mode_indices_buffer(&self) -> &wgpu::Buffer {
        &self.mode_indices_buffer
    }
    
    /// Get the cell IDs buffer for external GPU-to-GPU copies.
    pub fn cell_ids_buffer(&self) -> &wgpu::Buffer {
        &self.cell_ids_buffer
    }
    
    /// Get the genome IDs buffer for external GPU-to-GPU copies.
    pub fn genome_ids_buffer(&self) -> &wgpu::Buffer {
        &self.genome_ids_buffer
    }
    
    /// Get the cell types buffer for external GPU-to-GPU copies.
    pub fn cell_types_buffer(&self) -> &wgpu::Buffer {
        &self.cell_types_buffer
    }
    
    /// Get the mode properties buffer for external GPU-to-GPU copies.
    pub fn mode_properties_buffer(&self) -> &wgpu::Buffer {
        &self.mode_properties_buffer
    }
    
    /// Clear mode indices dirty flag (call after GPU-to-GPU copy).
    pub fn clear_mode_indices_dirty(&mut self) {
        self.mode_indices_dirty = false;
    }
    
    /// Clear cell IDs dirty flag (call after GPU-to-GPU copy).
    pub fn clear_cell_ids_dirty(&mut self) {
        self.cell_ids_dirty = false;
    }
    
    /// Clear genome IDs dirty flag (call after GPU-to-GPU copy).
    pub fn clear_genome_ids_dirty(&mut self) {
        self.genome_ids_dirty = false;
    }
    
    /// Clear cell types dirty flag (call after GPU-to-GPU copy).
    pub fn clear_cell_types_dirty(&mut self) {
        self.cell_types_dirty = false;
    }
    
    /// Clear rotations dirty flag (call after GPU-to-GPU copy).
    pub fn clear_rotations_dirty(&mut self) {
        self.rotations_dirty = false;
    }
    
    /// Get the counters buffer for debug readback.
    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters_buffer
    }
    
    /// DEBUG: Blocking readback of counters buffer [visible, total, frustum_culled, occluded].
    pub fn debug_read_counters_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [u32; 4] {
        let read_size = 16u64; // 4 x u32
        
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Counters Staging"),
            size: read_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Counters Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.counters_buffer, 0, &staging, 0, read_size);
        queue.submit(std::iter::once(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let view = slice.get_mapped_range();
            let data: &[u32] = bytemuck::cast_slice(&view);
            let result = [data[0], data[1], data[2], data[3]];
            drop(view);
            staging.unmap();
            result
        } else {
            [0, 0, 0, 0]
        }
    }
}
