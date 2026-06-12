//! GPU-based instance buffer builder using compute shaders.
//!
//! Builds cell instance data on the GPU to eliminate CPU-side iteration
//! and reduce CPU->GPU data transfer. Includes frustum and occlusion culling.

use crate::cell::types::{CellType, CellTypeVisuals};
use crate::genome::Genome;
use crate::simulation::CanonicalState;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

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
    mode_colors_buffer: wgpu::Buffer, // vec4 per mode: RGB color (xyz) + padding (w)
    mode_emissive_buffer: wgpu::Buffer, // vec4 per mode: emissive (x) + padding (yzw)
    cell_type_visuals_buffer: wgpu::Buffer,
    mode_properties_buffer: wgpu::Buffer,
    /// Cell type per mode (lookup table: mode_cell_types[mode_index] = cell_type)
    /// Used by shader to derive cell_type from mode_index when genomes change
    mode_cell_types_buffer: wgpu::Buffer,
    /// Behavior flags per cell type for parameterized shader logic
    behavior_flags_buffer: wgpu::Buffer,
    /// Mode properties v5 (cilia params): [cilia_speed, cilia_push_bonded, cilia_use_signal, cilia_signal_channel] per mode
    mode_properties_v5_buffer: wgpu::Buffer,

    // Output buffer (instance data for rendering)
    pub instance_buffer: wgpu::Buffer,
    pub instance_velocity_buffer: wgpu::Buffer,
    pub siphon_instance_buffer: wgpu::Buffer,
    pub siphon_instance_velocity_buffer: wgpu::Buffer,

    // Indirect draw buffer for GPU-driven rendering (total visible count)
    pub indirect_buffer: wgpu::Buffer,

    // Per-type indirect draw buffers for multi-pipeline rendering (one per type)
    // Dynamic allocation: each type gets consecutive instances based on actual counts
    indirect_buffers: Vec<wgpu::Buffer>,

    // Counter buffer for atomic operations
    // counters[0..4]: visible, total, frustum_culled, occluded
    // counters[4..4+MAX_TYPES]: per-type instance counts
    counters_buffer: wgpu::Buffer,
    counters_readback_buffer: wgpu::Buffer,

    // Current bind group (recreated when buffers change size)
    bind_group: Option<wgpu::BindGroup>,

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

    // Cave solid-mask culling
    cave_cull_params_buffer: wgpu::Buffer,
    dummy_solid_mask_buffer: wgpu::Buffer,
}

/// Frustum plane representation (normal + distance from origin).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrustumPlane {
    normal_and_dist: [f32; 4], // xyz = normal, w = distance
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CaveCullParams {
    pub enabled: u32,
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub origin_z: f32,
    pub _pad0: f32,
    pub _pad1: f32,
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
    lod_debug_colors: u32, // 0 = disabled, 1 = enabled
    // Cell buffer capacity for partition size calculation
    cell_capacity: u32,
    current_time: f32,
    // Frustum planes array needs 16-byte alignment
    frustum_planes: [FrustumPlane; 6],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ModeVisuals {
    color: [f32; 4],        // xyz = color, w = 1.0 (always opaque)
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
    // Goldberg ridge parameters (used by Photocyte membrane)
    goldberg_scale: f32,
    goldberg_ridge_width: f32,
    goldberg_meander: f32,
    goldberg_ridge_strength: f32,
    nucleus_scale: f32,
    _pad: f32,
    // Cilia ring parameters (used by Ciliocyte cell type)
    cilia_ring_frequency: f32,
    cilia_ring_depth: f32,
    cilia_ring_speed: f32,
    _pad2: f32,
    // Generic type params packed into type_data_0 for default-branch types
    param_a: f32,
    param_b: f32,
    param_c: f32,
    param_d: f32,
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
///   - type_data[0]: membrane_noise_scale (Test), flagella_angle (Flagellocyte), effective_speed (Ciliocyte)
///   - type_data[1]: membrane_noise_strength (Test), flagella_speed (Flagellocyte), cilia_ring_frequency (Ciliocyte)
///   - type_data[2]: membrane_noise_speed (Test), sensor_direction_x (Neurocyte), cilia_ring_depth (Ciliocyte)
///   - type_data[3]: membrane_anim_offset (Test), sensor_direction_y (Neurocyte), cilia_ring_speed (Ciliocyte)
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
        // Must match triple_buffer max_modes (8_000_000) so GPU-mutated mode indices never go out of bounds.
        // mode_visuals stores 2 vec4<f32> per mode (color + emissive) = 32 bytes per mode.
        // 8_000_000 * 32 = 256 MB - at wgpu's buffer limit.
        let mode_capacity = 8_000_000;
        let cell_type_capacity = 16;

        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Instance Builder Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/build_instances.wgsl").into(),
            ),
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
                // mode_colors (binding 7) - vec4 per mode: RGB color
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
                // Binding 16: Cell type behavior flags
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 17: Death flags - cells marked for removal
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 18: mode_emissive - vec4 per mode: emissive (x) + padding
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 19: mode_properties_v5 - cilia params per mode (vec4: cilia_speed, push_bonded, use_signal, signal_channel)
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 20: signal_flags - per-cell signal state (cell_idx * 16 + channel)
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 21: mode_properties_v7 - luminocyte signal params per mode [invert, unused, channel, threshold]
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 22: cave_cull_params - grid resolution/size/origin + enabled flag
                wgpu::BindGroupLayoutEntry {
                    binding: 22,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 23: cave_solid_mask - u32 per voxel, non-zero = solid rock
                wgpu::BindGroupLayoutEntry {
                    binding: 23,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 24: cell thermal state, used to freeze visual animation
                wgpu::BindGroupLayoutEntry {
                    binding: 24,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 25: mode_properties_v14 - Siphonocyte params per mode
                wgpu::BindGroupLayoutEntry {
                    binding: 25,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 26: cell water reserve, used to suppress dry Siphonocyte output visuals
                wgpu::BindGroupLayoutEntry {
                    binding: 26,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 27,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 28,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 29,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 30,
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
        let mode_indices_buffer =
            Self::create_storage_buffer(device, "Mode Indices", cell_capacity * 4);
        let cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", cell_capacity * 4);
        let genome_ids_buffer =
            Self::create_storage_buffer(device, "Genome IDs", cell_capacity * 4);
        let cell_types_buffer =
            Self::create_storage_buffer(device, "Cell Types", cell_capacity * 4);

        let mode_colors_buffer =
            Self::create_storage_buffer(device, "Mode Colors", mode_capacity * 16); // vec4 per mode
        let mode_emissive_buffer =
            Self::create_storage_buffer(device, "Mode Emissive", mode_capacity * 16); // vec4 per mode
        let cell_type_visuals_buffer = Self::create_storage_buffer(
            device,
            "Cell Type Visuals",
            cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>(),
        );
        // Mode properties: only mode_properties_v1 is copied here (16 bytes/mode = 1 vec4).
        // swim_force lives in v1 and is the only field the instance builder shader reads.
        let mode_properties_buffer = Self::create_storage_buffer(
            device,
            "Instance Builder Mode Properties",
            mode_capacity * 16,
        );
        // Mode cell types: 1 u32 per mode - lookup table for deriving cell_type from mode_index
        let mode_cell_types_buffer =
            Self::create_storage_buffer(device, "Mode Cell Types", mode_capacity * 4);

        // Cell type behavior flags: 64 bytes per type (GpuCellTypeBehaviorFlags struct)
        // Initialize with behavior flags for all types
        use crate::cell::types::GpuCellTypeBehaviorFlags;
        let flags: Vec<GpuCellTypeBehaviorFlags> =
            CellType::iter().map(|t| t.behavior_flags()).collect();
        let behavior_flags_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Type Behavior Flags"),
            contents: bytemuck::cast_slice(&flags),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Mode properties v5: cilia params per mode (16 bytes per mode = 1 vec4)
        let mode_properties_v5_buffer = Self::create_storage_buffer(
            device,
            "Instance Builder Mode Properties V5",
            mode_capacity * 16,
        );

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (cell_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let instance_velocity_buffer =
            Self::create_storage_buffer(device, "Instance Builder Output Velocities", cell_capacity * 16);
        let siphon_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Siphonocyte Output"),
            size: (cell_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let siphon_instance_velocity_buffer =
            Self::create_storage_buffer(device, "Instance Builder Siphonocyte Output Velocities", cell_capacity * 16);

        // Indirect draw buffer: vertex_count, instance_count, first_vertex, first_instance
        // Main buffer for total visible count (legacy, kept for compatibility)
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Indirect"),
            size: 16, // 4 u32s
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Per-type indirect draw buffers for multi-pipeline rendering (one per type)
        // Each type gets a separate indirect buffer for efficient GPU-driven rendering
        let indirect_buffers: Vec<wgpu::Buffer> = (0..CellType::MAX_TYPES)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Instance Builder Indirect Type {}", i)),
                    size: 16, // 4 u32s: vertex_count, instance_count, first_vertex, first_instance
                    usage: wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // Counters buffer: 4 general counters + MAX_TYPES per-type counters
        // counters[0..4]: visible, total, frustum_culled, occluded
        // counters[4..4+MAX_TYPES]: per-type instance counts
        let counter_count = 4 + CellType::MAX_TYPES;
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Counters"),
            size: (counter_count * 4) as u64, // 4 bytes per u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counters Readback"),
            size: (counter_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cave_cull_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cave Cull Params"),
                contents: bytemuck::bytes_of(&CaveCullParams {
                    enabled: 0,
                    grid_resolution: 128,
                    cell_size: 1.0,
                    origin_x: 0.0,
                    origin_y: 0.0,
                    origin_z: 0.0,
                    _pad0: 0.0,
                    _pad1: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let dummy_solid_mask_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dummy Solid Mask"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE,
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
            mode_colors_buffer,
            mode_emissive_buffer,
            cell_type_visuals_buffer,
            mode_properties_buffer,
            mode_cell_types_buffer,
            behavior_flags_buffer,
            mode_properties_v5_buffer,
            instance_buffer,
            instance_velocity_buffer,
            siphon_instance_buffer,
            siphon_instance_velocity_buffer,
            indirect_buffer,
            indirect_buffers,
            counters_buffer,
            counters_readback_buffer,
            bind_group: None,
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
            cave_cull_params_buffer,
            dummy_solid_mask_buffer,
        }
    }

    /// Enable cave solid-mask culling. Call once after the cave is initialized.
    /// Invalidates the bind group so it gets recreated with the correct solid mask buffer.
    pub fn set_cave_cull_params(&mut self, queue: &wgpu::Queue, params: CaveCullParams) {
        queue.write_buffer(
            &self.cave_cull_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
        self.bind_group = None; // force recreate with new solid mask reference
    }

    pub fn dummy_solid_mask_buffer(&self) -> &wgpu::Buffer {
        &self.dummy_solid_mask_buffer
    }

    fn create_storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Instance Builder {}", label)),
            size: size.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
            queue.write_buffer(
                &self.positions_buffer,
                0,
                bytemuck::cast_slice(&self.temp_positions),
            );
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
            queue.write_buffer(
                &self.rotations_buffer,
                0,
                bytemuck::cast_slice(&self.temp_rotations),
            );
            self.rotations_dirty = false;
        }

        if self.radii_dirty {
            // Update radii
            queue.write_buffer(
                &self.radii_buffer,
                0,
                bytemuck::cast_slice(&state.radii[..cell_count]),
            );
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

            queue.write_buffer(
                &self.mode_indices_buffer,
                0,
                bytemuck::cast_slice(&self.temp_mode_indices),
            );
            self.mode_indices_dirty = false;
        }

        if self.cell_ids_dirty {
            // Update cell IDs
            queue.write_buffer(
                &self.cell_ids_buffer,
                0,
                bytemuck::cast_slice(&state.cell_ids[..cell_count]),
            );
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
            queue.write_buffer(
                &self.genome_ids_buffer,
                0,
                bytemuck::cast_slice(&self.temp_genome_ids),
            );
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
                        2 // Default to Phagocyte cell type
                    }
                } else {
                    2 // Default to Phagocyte cell type
                };
                self.temp_cell_types.push(cell_type);
            }
            queue.write_buffer(
                &self.cell_types_buffer,
                0,
                bytemuck::cast_slice(&self.temp_cell_types),
            );
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
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(genome.modes.len() as u64);
            for (mode_idx, mode) in genome.modes.iter().enumerate() {
                // Include mode index for uniqueness
                hash = hash.wrapping_mul(31).wrapping_add(mode_idx as u64);
                // Include cell_type so type changes trigger buffer update
                hash = hash.wrapping_mul(31).wrapping_add(mode.cell_type as u64);
                hash = hash
                    .wrapping_mul(31)
                    .wrapping_add((mode.color.x * 1000.0) as u64);
                hash = hash
                    .wrapping_mul(31)
                    .wrapping_add((mode.color.y * 1000.0) as u64);
                hash = hash
                    .wrapping_mul(31)
                    .wrapping_add((mode.color.z * 1000.0) as u64);
                // Skip opacity since cells are always opaque
                hash = hash
                    .wrapping_mul(31)
                    .wrapping_add((mode.emissive * 1000.0) as u64);
            }
        }
        hash
    }

    fn hash_cell_type_visuals(visuals: &[CellTypeVisuals]) -> u64 {
        let mut hash = visuals.len() as u64;
        for v in visuals {
            // Hash every field so any change triggers a GPU buffer update
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.specular_strength * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.specular_power * 100.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.fresnel_strength * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.membrane_noise_scale * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.membrane_noise_strength * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.membrane_noise_speed * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.tail_length * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.tail_thickness * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.tail_amplitude * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.tail_frequency * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.tail_taper * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.tail_segments * 10.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.goldberg_scale * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.goldberg_ridge_width * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.goldberg_meander * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.goldberg_ridge_strength * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.nucleus_scale * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.cilia_ring_frequency * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.cilia_ring_depth * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.cilia_ring_speed * 1000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.param_a * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.param_b * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.param_c * 10000.0) as u64);
            hash = hash
                .wrapping_mul(31)
                .wrapping_add((v.param_d * 10000.0) as u64);
        }
        hash
    }

    pub fn update_mode_visuals_from_genomes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        genomes: &[Genome],
    ) {
        // Calculate total mode count across all genomes
        let total_mode_count: usize = genomes.iter().map(|g| g.modes.len()).sum();

        if total_mode_count > self.mode_capacity {
            self.mode_capacity = total_mode_count * 2;
            self.mode_colors_buffer =
                Self::create_storage_buffer(device, "Mode Colors", self.mode_capacity * 16);
            self.mode_emissive_buffer =
                Self::create_storage_buffer(device, "Mode Emissive", self.mode_capacity * 16);
            // Also resize mode_cell_types_buffer
            self.mode_cell_types_buffer =
                Self::create_storage_buffer(device, "Mode Cell Types", self.mode_capacity * 4);
        }

        // Always invalidate bind group when mode visuals change
        self.bind_group = None;

        // Build color and emissive arrays separately
        let mut colors: Vec<[f32; 4]> = Vec::with_capacity(total_mode_count);
        let mut emissives: Vec<[f32; 4]> = Vec::with_capacity(total_mode_count);
        for genome in genomes {
            for mode in &genome.modes {
                colors.push([mode.color.x, mode.color.y, mode.color.z, 1.0]);
                emissives.push([mode.emissive, 0.0, 0.0, 0.0]);
            }
        }

        if !colors.is_empty() {
            queue.write_buffer(&self.mode_colors_buffer, 0, bytemuck::cast_slice(&colors));
            queue.write_buffer(
                &self.mode_emissive_buffer,
                0,
                bytemuck::cast_slice(&emissives),
            );
        }

        // Build mode_cell_types lookup table
        let mode_cell_types: Vec<u32> = genomes
            .iter()
            .flat_map(|genome| genome.modes.iter().map(|mode| mode.cell_type as u32))
            .collect();

        if !mode_cell_types.is_empty() {
            queue.write_buffer(
                &self.mode_cell_types_buffer,
                0,
                bytemuck::cast_slice(&mode_cell_types),
            );
        }
    }

    fn update_cell_type_visuals(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        visuals: &[CellTypeVisuals],
    ) {
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
                membrane_noise_scale: v.membrane_noise_scale,
                membrane_noise_strength: v.membrane_noise_strength,
                membrane_noise_speed: v.membrane_noise_speed,
                // Flagella parameters (used by Flagellocyte cell type)
                tail_length: v.tail_length,
                tail_thickness: v.tail_thickness,
                tail_amplitude: v.tail_amplitude,
                tail_frequency: v.tail_frequency,
                // tail_speed removed - calculated from swim_force in shader
                tail_taper: v.tail_taper,
                tail_segments: v.tail_segments,
                // Goldberg ridge parameters
                goldberg_scale: v.goldberg_scale,
                goldberg_ridge_width: v.goldberg_ridge_width,
                goldberg_meander: v.goldberg_meander,
                goldberg_ridge_strength: v.goldberg_ridge_strength,
                nucleus_scale: v.nucleus_scale,
                _pad: 0.0,
                // Cilia ring parameters (used by Ciliocyte cell type)
                cilia_ring_frequency: v.cilia_ring_frequency,
                cilia_ring_depth: v.cilia_ring_depth,
                cilia_ring_speed: v.cilia_ring_speed,
                _pad2: 0.0,
                param_a: v.param_a,
                param_b: v.param_b,
                param_c: v.param_c,
                param_d: v.param_d,
            })
            .collect();

        if !gpu_visuals.is_empty() {
            queue.write_buffer(
                &self.cell_type_visuals_buffer,
                0,
                bytemuck::cast_slice(&gpu_visuals),
            );
        }
    }

    /// Update cell type visuals directly (public version for GPU scene).
    pub fn update_cell_type_visuals_direct(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        visuals: &[CellTypeVisuals],
    ) {
        self.update_cell_type_visuals(device, queue, visuals);
    }

    fn resize_cell_buffers(&mut self, device: &wgpu::Device, new_capacity: usize) {
        self.cell_capacity = new_capacity;

        self.positions_buffer = Self::create_storage_buffer(device, "Positions", new_capacity * 16);
        self.rotations_buffer = Self::create_storage_buffer(device, "Rotations", new_capacity * 16);
        self.radii_buffer = Self::create_storage_buffer(device, "Radii", new_capacity * 4);
        self.mode_indices_buffer =
            Self::create_storage_buffer(device, "Mode Indices", new_capacity * 4);
        self.cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", new_capacity * 4);
        self.genome_ids_buffer =
            Self::create_storage_buffer(device, "Genome IDs", new_capacity * 4);
        self.cell_types_buffer =
            Self::create_storage_buffer(device, "Cell Types", new_capacity * 4);

        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (new_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.instance_velocity_buffer =
            Self::create_storage_buffer(device, "Instance Builder Output Velocities", new_capacity * 16);
        self.siphon_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Siphonocyte Output"),
            size: (new_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.siphon_instance_velocity_buffer =
            Self::create_storage_buffer(device, "Instance Builder Siphonocyte Output Velocities", new_capacity * 16);

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

    fn recreate_bind_group(
        &mut self,
        device: &wgpu::Device,
        cell_count_buffer: &wgpu::Buffer,
        death_flags_buffer: &wgpu::Buffer,
        signal_flags_buffer: &wgpu::Buffer,
        mode_properties_v7_buffer: &wgpu::Buffer,
        solid_mask_buffer: &wgpu::Buffer,
        cell_thermal_state_buffer: &wgpu::Buffer,
        mode_properties_v14_buffer: &wgpu::Buffer,
        cell_water_buffer: &wgpu::Buffer,
        velocity_buffer: &wgpu::Buffer,
    ) {
        // Create a dummy 1x1 texture for binding 11 (Hi-Z removed)
        let (_dummy_texture, dummy_view) = Self::create_dummy_hiz_texture(device);
        let hiz_view = &dummy_view;

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
                    resource: self.mode_colors_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.behavior_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: death_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: self.mode_emissive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.mode_properties_v5_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: signal_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: mode_properties_v7_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: self.cave_cull_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: cell_thermal_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: mode_properties_v14_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: cell_water_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: self.instance_velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: self.siphon_instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: self.siphon_instance_velocity_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Extract frustum planes from a view-projection matrix (Gribb/Hartmann method).
    /// Each plane is stored as (normal.xyz, distance) with inward-pointing normals.
    /// A point is inside the frustum if dot(normal, point) + distance >= 0 for all planes.
    fn extract_frustum_planes(vp: Mat4) -> [FrustumPlane; 6] {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);

        let raw = [
            r3 + r0, // Left
            r3 - r0, // Right
            r3 + r1, // Bottom
            r3 - r1, // Top
            r2,      // Near  (wgpu depth [0,1])
            r3 - r2, // Far
        ];

        let mut planes = [FrustumPlane {
            normal_and_dist: [0.0; 4],
        }; 6];
        for i in 0..6 {
            let p = raw[i];
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > 1e-8 {
                let inv = 1.0 / len;
                planes[i].normal_and_dist = [p.x * inv, p.y * inv, p.z * inv, p.w * inv];
            }
        }
        planes
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
        death_flags_buffer: &wgpu::Buffer,
        signal_flags_buffer: &wgpu::Buffer,
        mode_properties_v7_buffer: &wgpu::Buffer,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        lod_debug_colors: bool,
        current_time: f32,
        cell_count_hint: u32, // Live cell count for dispatch scaling
        solid_mask_buffer: &wgpu::Buffer,
        cell_thermal_state_buffer: &wgpu::Buffer,
        mode_properties_v14_buffer: &wgpu::Buffer,
        cell_water_buffer: &wgpu::Buffer,
        velocity_buffer: &wgpu::Buffer,
    ) {
        if cell_capacity == 0 {
            self.last_visible_count = 0;
            return;
        }

        // IDLE EARLY-OUT: Skip all GPU work when there are no live cells.
        // cell_count_hint is the high-water mark (total_cell_slots) - when it's 0 no cells
        // have ever been placed (or all were cleared by reset). Zero the indirect buffer so
        // the renderer issues 0 draw calls, then return early.
        if cell_count_hint == 0 {
            self.last_visible_count = 0;
            // Zero the indirect buffer's instance_count field so no cells are drawn.
            // This handles the reset case where the previous frame had live cells.
            let indirect_data: [u32; 4] = [4, 0, 0, 0];
            queue.write_buffer(
                &self.indirect_buffer,
                0,
                bytemuck::cast_slice(&indirect_data),
            );
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
            far_plane: 5000.0,
            screen_width: screen_width as f32,
            screen_height: screen_height as f32,
            hiz_mip_count: 0,
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
            // Cell buffer capacity for partition size calculation
            cell_capacity: cell_capacity as u32,
            current_time,
            frustum_planes,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Clear all counters: 4 general + MAX_TYPES per-type counters
        const COUNTER_COUNT: usize = 4 + CellType::MAX_TYPES;
        let zero_counters = [0u32; COUNTER_COUNT];
        queue.write_buffer(
            &self.counters_buffer,
            0,
            bytemuck::cast_slice(&zero_counters),
        );

        // Initialize indirect draw buffer (legacy, total visible count)
        // vertex_count=4 (triangle strip quad), instance_count=0, first_vertex=0, first_instance=0
        let indirect_data: [u32; 4] = [4, 0, 0, 0];
        queue.write_buffer(
            &self.indirect_buffer,
            0,
            bytemuck::cast_slice(&indirect_data),
        );

        // Initialize per-type indirect draw buffers with fixed partitioning
        // Each type gets a fixed partition: [type_index * partition_size, (type_index+1) * partition_size)
        let _partition_size = cell_capacity / CellType::MAX_TYPES;
        for (_type_index, indirect_buffer) in self.indirect_buffers.iter().enumerate() {
            // first_instance is 0 since we use vertex buffer offset instead
            let indirect_init: [u32; 4] = [4, 0, 0, 0]; // vertex_count, instance_count, first_vertex, first_instance
            queue.write_buffer(indirect_buffer, 0, bytemuck::cast_slice(&indirect_init));
        }

        // Ensure bind group exists with cell_count_buffer and death_flags_buffer
        if self.bind_group.is_none() {
            self.recreate_bind_group(
                device,
                cell_count_buffer,
                death_flags_buffer,
                signal_flags_buffer,
                mode_properties_v7_buffer,
                solid_mask_buffer,
                cell_thermal_state_buffer,
                mode_properties_v14_buffer,
                cell_water_buffer,
                velocity_buffer,
            );
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

            // PERFORMANCE: Dispatch based on actual cell count, not full capacity
            // At 100K cells, this reduces dispatch from 2048 to ~782 workgroups (2.6x reduction)
            let effective_count = (cell_count_hint.max(1) + 127) / 128 * 128; // Round up to workgroup boundary
            let workgroup_count = (effective_count + 127) / 128;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy visible count (counters[0]) to legacy indirect buffer's instance_count field (offset 4)
        encoder.copy_buffer_to_buffer(
            &self.counters_buffer,
            0, // counters[0] = visible count
            &self.indirect_buffer,
            4, // instance_count field offset
            4, // 4 bytes (u32)
        );

        // Copy per-type counts (counters[4..4+MAX_TYPES]) to each type's indirect buffer instance_count field
        for type_index in 0..CellType::MAX_TYPES {
            let counter_offset = (4 + type_index) * 4; // counters[4+type_index] byte offset
            encoder.copy_buffer_to_buffer(
                &self.counters_buffer,
                counter_offset as u64,
                &self.indirect_buffers[type_index],
                4, // instance_count field offset in indirect buffer
                4, // 4 bytes (u32)
            );
        }

        // Copy counters to readback buffer for stats (only if not currently mapped)
        if !self.stats_map_pending {
            let counter_count = 4 + CellType::MAX_TYPES;
            encoder.copy_buffer_to_buffer(
                &self.counters_buffer,
                0,
                &self.counters_readback_buffer,
                0,
                (counter_count * 4) as u64, // All counters
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
        death_flags_buffer: &wgpu::Buffer,
        signal_flags_buffer: &wgpu::Buffer,
        mode_properties_v7_buffer: &wgpu::Buffer,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        lod_debug_colors: bool,
        current_time: f32,
        cell_count_hint: u32,
        solid_mask_buffer: &wgpu::Buffer,
        cell_thermal_state_buffer: &wgpu::Buffer,
        mode_properties_v14_buffer: &wgpu::Buffer,
        cell_water_buffer: &wgpu::Buffer,
        velocity_buffer: &wgpu::Buffer,
    ) {
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
            death_flags_buffer,
            signal_flags_buffer,
            mode_properties_v7_buffer,
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
            lod_debug_colors,
            current_time,
            cell_count_hint,
            solid_mask_buffer,
            cell_thermal_state_buffer,
            mode_properties_v14_buffer,
            cell_water_buffer,
            velocity_buffer,
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

                    // Update stats
                    self.last_stats = CullingStats {
                        total_cells: counters[1],
                        visible_cells: counters[0],
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

    pub fn get_instance_velocity_buffer(&self) -> &wgpu::Buffer {
        &self.instance_velocity_buffer
    }

    pub fn get_siphon_instance_buffer(&self) -> &wgpu::Buffer {
        &self.siphon_instance_buffer
    }

    pub fn get_siphon_instance_velocity_buffer(&self) -> &wgpu::Buffer {
        &self.siphon_instance_velocity_buffer
    }

    pub fn get_counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters_buffer
    }

    /// Get the current cell capacity.
    pub fn cell_capacity(&self) -> usize {
        self.cell_capacity
    }

    /// Get the indirect draw buffer for GPU-driven rendering.
    pub fn get_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_buffer
    }

    /// Get the indirect draw buffer for a specific cell type.
    /// Each type has its own indirect buffer for GPU-driven rendering.
    pub fn get_indirect_buffer_for_type(&self, cell_type: CellType) -> &wgpu::Buffer {
        &self.indirect_buffers[cell_type as usize]
    }

    /// Get all per-type indirect buffers for iterating over types during rendering.
    pub fn get_indirect_buffers(&self) -> &[wgpu::Buffer] {
        &self.indirect_buffers
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

    /// Get the mode colors buffer (vec4 per mode: RGB + padding).
    pub fn mode_colors_buffer(&self) -> &wgpu::Buffer {
        &self.mode_colors_buffer
    }

    /// Get the mode emissive buffer (vec4 per mode: emissive + padding).
    pub fn mode_emissive_buffer(&self) -> &wgpu::Buffer {
        &self.mode_emissive_buffer
    }

    /// Get the mode properties buffer for external GPU-to-GPU copies (holds mode_properties_v1).
    pub fn mode_properties_buffer(&self) -> &wgpu::Buffer {
        &self.mode_properties_buffer
    }

    /// Get the mode properties v5 buffer for external GPU-to-GPU copies (holds cilia params).
    pub fn mode_properties_v5_buffer(&self) -> &wgpu::Buffer {
        &self.mode_properties_v5_buffer
    }

    /// Get the mode cell types buffer for external GPU-to-GPU copies.
    pub fn mode_cell_types_buffer(&self) -> &wgpu::Buffer {
        &self.mode_cell_types_buffer
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
}
