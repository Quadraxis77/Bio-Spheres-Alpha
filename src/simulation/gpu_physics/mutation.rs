//! GPU-side genome mutation system
//!
//! Implements radiation-driven mutation during cell division. When a cell divides,
//! each child independently rolls a mutation chance. On hit, the child receives a
//! new genome that is a clone of the parent's with one parameter perturbed.
//!
//! The system is fully GPU-native: genome cloning, parameter selection, and
//! perturbation all happen in compute shaders with no CPU readback.

use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum number of mutation candidates per frame (2 per division: child A + child B)
const MAX_MUTATION_CANDIDATES: u32 = 8192;

/// Maximum mutation log entries for debug/UI feedback
const MAX_MUTATION_LOG_ENTRIES: u32 = 1024;

/// Maximum genomes the mutation system can allocate
/// (80K genomes × 40 modes each = 3.2M total modes, bounded by wgpu's 256 MB/buffer limit)
const GENOME_RING_CAPACITY: u32 = 80_000;

/// Maximum modes across all genomes (must match triple_buffer.rs: 40 * 200_000 = 8_000_000)
const MAX_TOTAL_MODES: u32 = 8_000_000;

/// Run genome GC every N frames to avoid race conditions with mutations
/// Running every frame causes newly created genomes to be recycled before cells can use them
/// Reduced to 30 frames to reclaim mode buffer space more frequently at high mutation rates
const GC_INTERVAL_FRAMES: u32 = 30;

/// Vulnerability table entry matching the WGSL MutationParamEntry struct.
/// Describes one mutable parameter: which buffer, offset, weight, bounds.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MutationParamEntry {
    pub buffer_id: u32,
    pub element_offset: u32,
    pub weight: f32,
    pub min_delta: f32,
    pub max_delta: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub data_type: u32,
}

/// Data types for mutation parameters
#[allow(dead_code)]
pub mod data_type {
    pub const CONTINUOUS_F32: u32 = 0;
    pub const INTEGER: u32 = 1;
    pub const BOOLEAN: u32 = 2;
    pub const MODE_INDEX_CLAMP: u32 = 3;
}

/// Buffer IDs matching the WGSL switch cases
#[allow(dead_code)]
pub mod buffer_id {
    pub const MODE_PROPERTIES: u32 = 0;
    pub const MODE_CELL_TYPES: u32 = 1;
    pub const CHILD_MODE_INDICES: u32 = 2;
    pub const PARENT_MAKE_ADHESION: u32 = 3;
    pub const CHILD_A_KEEP_ADHESION: u32 = 4;
    pub const CHILD_B_KEEP_ADHESION: u32 = 5;
    pub const GENOME_MODE_DATA: u32 = 6;
    pub const GLUEOCYTE_ENV_ADHESION: u32 = 7;
    pub const OCULOCYTE_PARAMS: u32 = 8;
    pub const MODE_VISUALS: u32 = 9;
}

/// Uniform params matching the WGSL MutationParams struct
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MutationParamsUniform {
    pub radiation_level: f32,
    pub rng_seed: u32,
    pub current_frame: u32,
    pub param_table_size: u32,
    pub total_mode_count: u32,
    pub max_modes_per_genome: u32,
    pub genome_ring_capacity: u32,
    pub _pad0: u32,
}

/// Per-genome metadata matching WGSL: vec4<u32>(mode_count, base_mode_offset, ref_count, flags)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenomeMeta {
    pub mode_count: u32,
    pub base_mode_offset: u32,
    pub ref_count: u32,
    pub flags: u32,
}

/// Uniform params for the collect_candidates shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CollectParams {
    pub cell_capacity: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform params for the genome GC shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GCParams {
    pub genome_capacity: u32,
    pub genome_ring_capacity: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

pub struct MutationSystem {
    // Whether the ring state has been initialized (first sync_genome_metadata call)
    ring_initialized: bool,

    // Compute pipelines
    pipeline: wgpu::ComputePipeline,
    collect_pipeline: wgpu::ComputePipeline,
    gc_pipeline: wgpu::ComputePipeline,
    ref_count_clear_pipeline: wgpu::ComputePipeline,
    ref_count_count_pipeline: wgpu::ComputePipeline,
    mode_offset_reset_pipeline: wgpu::ComputePipeline,

    // Bind group layouts for mutation shader
    params_layout: wgpu::BindGroupLayout,
    candidates_layout: wgpu::BindGroupLayout,
    buffers_layout: wgpu::BindGroupLayout,

    // Bind group layouts for collect_candidates shader
    collect_input_layout: wgpu::BindGroupLayout,
    collect_output_layout: wgpu::BindGroupLayout,

    // Bind group layout for genome GC shader
    gc_layout: wgpu::BindGroupLayout,

    // Bind group layout for ref_count sync shader
    ref_count_sync_layout: wgpu::BindGroupLayout,

    // Bind group layout for mode_offset_reset shader (single buffer)
    mode_offset_reset_layout: wgpu::BindGroupLayout,

    // Uniform buffer for MutationParams
    params_buffer: wgpu::Buffer,

    // Uniform buffer for CollectParams
    collect_params_buffer: wgpu::Buffer,

    // Uniform buffer for GCParams
    gc_params_buffer: wgpu::Buffer,

    // Uniform buffer for ref_count sync params
    ref_count_sync_params_buffer: wgpu::Buffer,

    // Vulnerability table (storage buffer, uploaded from CPU)
    vulnerability_table_buffer: wgpu::Buffer,
    vulnerability_table: Vec<MutationParamEntry>,

    // Genome free slot ring buffer
    genome_ring_state_buffer: wgpu::Buffer,  // [head, tail, next_id, next_mode_offset]
    genome_free_ring_buffer: wgpu::Buffer,

    // Per-genome metadata
    genome_meta_buffer: wgpu::Buffer,

    // Per-genome reference counts (separate atomic buffer)
    genome_ref_counts_buffer: wgpu::Buffer,

    // Mutation candidates (written by collect_candidates, read by mutation shader)
    mutation_candidates_buffer: wgpu::Buffer,
    mutation_candidate_count_buffer: wgpu::Buffer,

    // Mutation event log (for debug/UI)
    mutation_log_buffer: wgpu::Buffer,
    mutation_log_count_buffer: wgpu::Buffer,

    // Cached bind groups for mutation shader (rebuilt when buffers change)
    params_bind_group: Option<wgpu::BindGroup>,
    candidates_bind_group: Option<wgpu::BindGroup>,
    buffers_bind_group: Option<wgpu::BindGroup>,

    // Cached bind groups for collect_candidates shader
    collect_input_bind_group: Option<wgpu::BindGroup>,
    collect_output_bind_group: Option<wgpu::BindGroup>,

    // Cached bind group for genome GC shader
    gc_bind_group: Option<wgpu::BindGroup>,

    // Cached bind group for ref_count sync shader
    ref_count_sync_bind_group: Option<wgpu::BindGroup>,

    // Cached bind group for mode_offset_reset shader
    mode_offset_reset_bind_group: Option<wgpu::BindGroup>,

    // State
    rng_counter: AtomicU32,
    radiation_level: f32,
    cell_capacity: u32,
    gc_frame_counter: AtomicU32,
    // Number of user-owned genomes (indices 0..user_genome_count must stay immortal)
    user_genome_count: u32,
}

impl MutationSystem {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cell_capacity: u32) -> Self {
        // Create shader modules
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mutation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/mutation.wgsl").into()),
        });

        let collect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mutation Collect Candidates Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/mutation_collect_candidates.wgsl").into()),
        });

        let gc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Genome GC Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/genome_gc.wgsl").into()),
        });

        let ref_count_sync_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Genome Ref Count Sync Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/genome_ref_count_sync.wgsl").into()),
        });

        // --- Bind group layouts ---

        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mutation Params Layout"),
            entries: &[
                // binding 0: MutationParams uniform
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
                // binding 1: vulnerability_table (read-only storage)
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
                // binding 2: genome_ring_state (read_write storage)
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
                // binding 3: genome_free_ring (read_write storage)
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
                // binding 4: genome_meta (read_write storage)
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
                // binding 5: genome_ref_counts (read_write storage)
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
            ],
        });

        let candidates_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mutation Candidates Layout"),
            entries: &[
                // binding 0: mutation_candidates (read-only)
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
                // binding 1: mutation_candidate_count (read-only)
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
                // binding 2: genome_ids (read_write)
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
                // binding 3: mode_indices (read_write)
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

        let buffers_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mutation Buffers Layout"),
            entries: &[
                // bindings 0-4: mode_properties sub-buffers (v0-v4)
                Self::storage_rw_entry(0),
                Self::storage_rw_entry(1),
                Self::storage_rw_entry(2),
                Self::storage_rw_entry(3),
                Self::storage_rw_entry(4),
                // bindings 5-9: genome_mode_data sub-buffers (v0-v4)
                Self::storage_rw_entry(5),
                Self::storage_rw_entry(6),
                Self::storage_rw_entry(7),
                Self::storage_rw_entry(8),
                Self::storage_rw_entry(9),
                // binding 10: child_mode_indices
                Self::storage_rw_entry(10),
                // binding 11: mode_cell_types
                Self::storage_rw_entry(11),
                // binding 12: parent_make_adhesion_flags
                Self::storage_rw_entry(12),
                // binding 13: child_a_keep_adhesion_flags
                Self::storage_rw_entry(13),
                // binding 14: child_b_keep_adhesion_flags
                Self::storage_rw_entry(14),
                // binding 15: glueocyte_env_adhesion_flags
                Self::storage_rw_entry(15),
                // binding 16: oculocyte_params
                Self::storage_rw_entry(16),
                // binding 17: mutation_log
                Self::storage_rw_entry(17),
                // binding 18: mutation_log_count
                Self::storage_rw_entry(18),
                // binding 19: mode_colors
                Self::storage_rw_entry(19),
                // binding 20: mode_emissive
                Self::storage_rw_entry(20),
            ],
        });

        // --- Collect candidates bind group layouts ---

        let collect_input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collect Candidates Input Layout"),
            entries: &[
                // binding 0: collect_params (uniform)
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
                // binding 1: division_flags (read-only storage)
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
                // binding 2: division_slot_assignments (read-only storage)
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
                // binding 3: genome_ids (read-only storage)
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
                // binding 4: cell_count_buffer (read-only storage)
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

        let collect_output_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collect Candidates Output Layout"),
            entries: &[
                // binding 0: mutation_candidates (read_write storage)
                Self::storage_rw_entry(0),
                // binding 1: mutation_candidate_count (read_write storage)
                Self::storage_rw_entry(1),
            ],
        });

        let gc_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Genome GC Layout"),
            entries: &[
                // binding 0: genome_ring_state (read_write storage)
                Self::storage_rw_entry(0),
                // binding 1: genome_free_ring (read_write storage)
                Self::storage_rw_entry(1),
                // binding 2: genome_ref_counts (read_write storage)
                Self::storage_rw_entry(2),
                // binding 3: genome_meta (read_write storage)
                Self::storage_rw_entry(3),
                // binding 4: gc_params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // --- Pipelines ---

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mutation Pipeline Layout"),
            bind_group_layouts: &[&params_layout, &candidates_layout, &buffers_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mutation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let collect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mutation Collect Candidates Pipeline Layout"),
            bind_group_layouts: &[&collect_input_layout, &collect_output_layout],
            push_constant_ranges: &[],
        });

        let collect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mutation Collect Candidates Pipeline"),
            layout: Some(&collect_pipeline_layout),
            module: &collect_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gc_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Genome GC Pipeline Layout"),
            bind_group_layouts: &[&gc_layout],
            push_constant_ranges: &[],
        });

        let gc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Genome GC Pipeline"),
            layout: Some(&gc_pipeline_layout),
            module: &gc_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ref_count_sync_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ref Count Sync Layout"),
            entries: &[
                // binding 0: sync params (uniform)
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
                // binding 1: genome_ids (read-only)
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
                // binding 2: death_flags (read-only)
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
                // binding 3: genome_ref_counts (read-write)
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

        let ref_count_sync_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ref Count Sync Pipeline Layout"),
            bind_group_layouts: &[&ref_count_sync_layout],
            push_constant_ranges: &[],
        });

        let ref_count_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ref Count Clear Pipeline"),
            layout: Some(&ref_count_sync_pipeline_layout),
            module: &ref_count_sync_shader,
            entry_point: Some("clear_ref_counts"),
            compilation_options: Default::default(),
            cache: None,
        });

        let ref_count_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ref Count Count Pipeline"),
            layout: Some(&ref_count_sync_pipeline_layout),
            module: &ref_count_sync_shader,
            entry_point: Some("count_genome_usage"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Mode offset reset shader and pipeline
        let mode_offset_reset_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mode Offset Reset Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/mode_offset_reset.wgsl").into()),
        });

        let mode_offset_reset_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mode Offset Reset Layout"),
            entries: &[
                // binding 0: genome_ring_state (read_write storage)
                Self::storage_rw_entry(0),
            ],
        });

        let mode_offset_reset_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mode Offset Reset Pipeline Layout"),
            bind_group_layouts: &[&mode_offset_reset_layout],
            push_constant_ranges: &[],
        });

        let mode_offset_reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mode Offset Reset Pipeline"),
            layout: Some(&mode_offset_reset_pipeline_layout),
            module: &mode_offset_reset_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Buffers ---

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Params Uniform"),
            size: std::mem::size_of::<MutationParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let collect_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collect Candidates Params Uniform"),
            size: std::mem::size_of::<CollectParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gc_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome GC Params Uniform"),
            size: std::mem::size_of::<GCParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ref_count_sync_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ref Count Sync Params Uniform"),
            size: std::mem::size_of::<GCParams>() as u64, // Same size as GCParams (cell_capacity, genome_capacity, pads)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vulnerability_table = Self::build_default_vulnerability_table();
        let table_size = (vulnerability_table.len() * std::mem::size_of::<MutationParamEntry>()) as u64;
        // Allocate for at least 64 entries so set_subtle_mutations can upload larger tables
        // without overrunning the buffer.
        let max_table_size = (64 * std::mem::size_of::<MutationParamEntry>()) as u64;
        let vulnerability_table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Vulnerability Table"),
            size: table_size.max(max_table_size),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vulnerability_table_buffer, 0, bytemuck::cast_slice(&vulnerability_table));

        // Genome ring state: [head, tail, next_id, next_mode_offset, max_active_mode_offset]
        let genome_ring_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Ring State"),
            size: 20, // 5 x u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let genome_free_ring_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Free Ring"),
            size: (GENOME_RING_CAPACITY * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let genome_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Metadata"),
            size: (GENOME_RING_CAPACITY as u64) * std::mem::size_of::<GenomeMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let genome_ref_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Reference Counts"),
            size: (GENOME_RING_CAPACITY * 4) as u64, // u32 per genome
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mutation_candidates_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Candidates"),
            size: (MAX_MUTATION_CANDIDATES * 8) as u64, // vec2<u32> per candidate
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mutation_candidate_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Candidate Count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mutation_log_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Log"),
            size: (MAX_MUTATION_LOG_ENTRIES * 16) as u64, // vec4<u32> per entry
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mutation_log_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Log Count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            collect_pipeline,
            gc_pipeline,
            ref_count_clear_pipeline,
            ref_count_count_pipeline,
            mode_offset_reset_pipeline,
            params_layout,
            candidates_layout,
            buffers_layout,
            collect_input_layout,
            collect_output_layout,
            gc_layout,
            ref_count_sync_layout,
            mode_offset_reset_layout,
            params_buffer,
            collect_params_buffer,
            gc_params_buffer,
            ref_count_sync_params_buffer,
            vulnerability_table_buffer,
            vulnerability_table,
            genome_ring_state_buffer,
            genome_free_ring_buffer,
            genome_meta_buffer,
            genome_ref_counts_buffer,
            mutation_candidates_buffer,
            mutation_candidate_count_buffer,
            mutation_log_buffer,
            mutation_log_count_buffer,
            params_bind_group: None,
            candidates_bind_group: None,
            buffers_bind_group: None,
            collect_input_bind_group: None,
            collect_output_bind_group: None,
            gc_bind_group: None,
            ref_count_sync_bind_group: None,
            mode_offset_reset_bind_group: None,
            rng_counter: AtomicU32::new(0),
            radiation_level: 1.0,  // Default to enabled to avoid timing issues with UI sync
            cell_capacity,
            gc_frame_counter: AtomicU32::new(0),
            ring_initialized: false,
            user_genome_count: 0,
        }
    }

    fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Build the default vulnerability table.
    /// Each entry describes one mutable parameter with its weight, bounds, and type.
    /// Weights are relative — higher weight = more likely to be selected.
    /// This table is intentionally flexible: add/remove/reweight entries freely.
    fn build_default_vulnerability_table() -> Vec<MutationParamEntry> {
        // Dramatic color mutation: re-rolls all 3 RGB components of one random mode.
        // element_offset 0xFF signals the shader to randomize all components.
        vec![
            MutationParamEntry {
                buffer_id: buffer_id::MODE_VISUALS, element_offset: 0xFF,
                weight: 1.0, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
        ]
    }

    /// Reset mutation system state for scene restart.
    /// Clears ring_initialized so the next sync_genome_metadata call re-seeds
    /// next_genome_id and next_mode_offset from scratch.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        self.ring_initialized = false;
        // Zero out the ring state buffer so stale offsets don't persist
        let zeroed: [u32; 5] = [0; 5];
        queue.write_buffer(&self.genome_ring_state_buffer, 0, bytemuck::cast_slice(&zeroed));
    }

    /// Set the global radiation level (0.0 = off, 1.0 = every division mutates)
    pub fn set_radiation_level(&mut self, level: f32) {
        self.radiation_level = level.clamp(0.0, 1.0);
    }

    /// Get the current radiation level
    pub fn radiation_level(&self) -> f32 {
        self.radiation_level
    }

    /// Switch between subtle (small perturbations) and dramatic (full re-roll) color mutations.
    pub fn set_subtle_mutations(&mut self, queue: &wgpu::Queue, subtle: bool) {
        let entries = if subtle {
            vec![
                MutationParamEntry {
                    buffer_id: buffer_id::MODE_VISUALS, element_offset: 0,
                    weight: 1.0, min_delta: 0.03, max_delta: 0.12,
                    min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
                },
                MutationParamEntry {
                    buffer_id: buffer_id::MODE_VISUALS, element_offset: 1,
                    weight: 1.0, min_delta: 0.03, max_delta: 0.12,
                    min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
                },
                MutationParamEntry {
                    buffer_id: buffer_id::MODE_VISUALS, element_offset: 2,
                    weight: 1.0, min_delta: 0.03, max_delta: 0.12,
                    min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
                },
            ]
        } else {
            Self::build_default_vulnerability_table()
        };
        self.update_vulnerability_table(queue, entries);
    }

    /// Update the vulnerability table with new entries.
    /// Call this when the user modifies vulnerability weights in the UI.
    pub fn update_vulnerability_table(&mut self, queue: &wgpu::Queue, entries: Vec<MutationParamEntry>) {
        self.vulnerability_table = entries;
        queue.write_buffer(
            &self.vulnerability_table_buffer,
            0,
            bytemuck::cast_slice(&self.vulnerability_table),
        );
    }

    /// Sync user genome metadata to GPU.
    /// Called when genomes are first loaded or when a genome is added/modified.
    ///
    /// IMPORTANT: Only writes the user genome entries (0..genomes.len()) using
    /// targeted partial writes — never zeros the rest of the buffer, which would
    /// wipe GPU-written metadata for mutated genomes.
    ///
    /// The ring state (next_genome_id, next_mode_offset) is only initialized on
    /// the very first call. Subsequent calls leave it untouched so that mutation
    /// IDs already allocated by the GPU shader are not recycled.
    pub fn sync_genome_metadata(
        &mut self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut offset = 0u32;
        for (i, genome) in genomes.iter().enumerate() {
            if i >= GENOME_RING_CAPACITY as usize {
                break;
            }
            let entry = GenomeMeta {
                mode_count: genome.modes.len() as u32,
                base_mode_offset: offset,
                ref_count: 0,
                flags: 0,
            };
            // Partial write: only touch this genome's entry, leave all others intact
            let byte_offset = (i * std::mem::size_of::<GenomeMeta>()) as u64;
            queue.write_buffer(&self.genome_meta_buffer, byte_offset, bytemuck::bytes_of(&entry));

            // Keep user genomes immortal (ref_count = u32::MAX so GC never recycles them)
            let ref_byte_offset = (i * std::mem::size_of::<u32>()) as u64;
            let immortal: u32 = u32::MAX;
            queue.write_buffer(&self.genome_ref_counts_buffer, ref_byte_offset, bytemuck::bytes_of(&immortal));

            offset += genome.modes.len() as u32;
        }

        // Only reset the ring state on the very first call.
        // After that the GPU mutation shader owns next_genome_id and next_mode_offset —
        // resetting them would cause the shader to re-use IDs already in use by live cells.
        if !self.ring_initialized {
            let initial_ring_state: [u32; 4] = [
                0,                          // head
                0,                          // tail
                genomes.len() as u32,       // next_genome_id (starts after user genomes)
                offset,                     // next_mode_offset (starts after user genome modes)
            ];
            queue.write_buffer(&self.genome_ring_state_buffer, 0, bytemuck::cast_slice(&initial_ring_state));
            self.ring_initialized = true;
        }

        self.user_genome_count = genomes.len() as u32;
    }

    /// Build bind groups. Call after triple buffer or genome buffers change.
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        genome_ids_buffer: &wgpu::Buffer,
        mode_indices_buffer: &wgpu::Buffer,
        mode_properties_v0: &wgpu::Buffer,
        mode_properties_v1: &wgpu::Buffer,
        mode_properties_v2: &wgpu::Buffer,
        mode_properties_v3: &wgpu::Buffer,
        mode_properties_v4: &wgpu::Buffer,
        genome_mode_data_v0: &wgpu::Buffer,
        genome_mode_data_v1: &wgpu::Buffer,
        genome_mode_data_v2: &wgpu::Buffer,
        genome_mode_data_v3: &wgpu::Buffer,
        genome_mode_data_v4: &wgpu::Buffer,
        child_mode_indices_buffer: &wgpu::Buffer,
        mode_cell_types_buffer: &wgpu::Buffer,
        parent_make_adhesion_flags_buffer: &wgpu::Buffer,
        child_a_keep_adhesion_flags_buffer: &wgpu::Buffer,
        child_b_keep_adhesion_flags_buffer: &wgpu::Buffer,
        glueocyte_env_adhesion_flags_buffer: &wgpu::Buffer,
        oculocyte_params_buffer: &wgpu::Buffer,
        mode_colors_buffer: &wgpu::Buffer,
        mode_emissive_buffer: &wgpu::Buffer,
    ) {
        self.params_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mutation Params Bind Group"),
            layout: &self.params_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vulnerability_table_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.genome_ring_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.genome_free_ring_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.genome_meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.genome_ref_counts_buffer.as_entire_binding() },
            ],
        }));

        self.candidates_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mutation Candidates Bind Group"),
            layout: &self.candidates_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.mutation_candidates_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.mutation_candidate_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: genome_ids_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mode_indices_buffer.as_entire_binding() },
            ],
        }));

        self.buffers_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mutation Buffers Bind Group"),
            layout: &self.buffers_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: mode_properties_v0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: mode_properties_v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mode_properties_v2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mode_properties_v3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mode_properties_v4.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: genome_mode_data_v0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: genome_mode_data_v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: genome_mode_data_v2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: genome_mode_data_v3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: genome_mode_data_v4.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: child_mode_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: mode_cell_types_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: parent_make_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: child_a_keep_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: child_b_keep_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: glueocyte_env_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: oculocyte_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: self.mutation_log_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 18, resource: self.mutation_log_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 19, resource: mode_colors_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: mode_emissive_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Build bind groups for the collect_candidates pipeline.
    /// Call after triple buffer changes (needs division_flags, division_slot_assignments, genome_ids, cell_count_buffer).
    pub fn rebuild_collect_bind_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        division_flags_buffer: &wgpu::Buffer,
        division_slot_assignments_buffer: &wgpu::Buffer,
        genome_ids_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) {
        // Upload collect params
        let collect_params = CollectParams {
            cell_capacity: self.cell_capacity,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.collect_params_buffer, 0, bytemuck::bytes_of(&collect_params));

        self.collect_input_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collect Candidates Input Bind Group"),
            layout: &self.collect_input_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.collect_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: division_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: division_slot_assignments_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: genome_ids_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_count_buffer.as_entire_binding() },
            ],
        }));

        self.collect_output_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collect Candidates Output Bind Group"),
            layout: &self.collect_output_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.mutation_candidates_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.mutation_candidate_count_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Build bind group for the genome GC pipeline.
    pub fn rebuild_gc_bind_group(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Upload GC params
        let gc_params = GCParams {
            genome_capacity: GENOME_RING_CAPACITY,
            genome_ring_capacity: GENOME_RING_CAPACITY,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.gc_params_buffer, 0, bytemuck::bytes_of(&gc_params));

        self.gc_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Genome GC Bind Group"),
            layout: &self.gc_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.genome_ring_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.genome_free_ring_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.genome_ref_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.genome_meta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.gc_params_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Build bind group for the ref_count sync pipeline.
    /// Requires genome_ids and death_flags from triple buffer.
    pub fn rebuild_ref_count_sync_bind_group(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        genome_ids_buffer: &wgpu::Buffer,
        death_flags_buffer: &wgpu::Buffer,
    ) {
        // Upload sync params
        let sync_params = GCParams {
            genome_capacity: self.cell_capacity,
            genome_ring_capacity: GENOME_RING_CAPACITY,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.ref_count_sync_params_buffer, 0, bytemuck::bytes_of(&sync_params));

        self.ref_count_sync_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ref Count Sync Bind Group"),
            layout: &self.ref_count_sync_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ref_count_sync_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: genome_ids_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: death_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.genome_ref_counts_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Dispatch the full mutation pipeline: collect_candidates → mutation.
    /// Call this AFTER the division execute pass completes.
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        current_frame: u32,
    ) {
        if self.radiation_level <= 0.0 {
            return; // Mutations disabled
        }

        let (Some(collect_input_bg), Some(collect_output_bg)) =
            (&self.collect_input_bind_group, &self.collect_output_bind_group)
        else {
            return; // Collect bind groups not yet built
        };

        let (Some(params_bg), Some(candidates_bg), Some(buffers_bg)) =
            (&self.params_bind_group, &self.candidates_bind_group, &self.buffers_bind_group)
        else {
            return; // Mutation bind groups not yet built
        };

        // Clear mutation candidate count before collect pass
        queue.write_buffer(&self.mutation_candidate_count_buffer, 0, bytemuck::bytes_of(&0u32));

        // Stage 1: Collect candidates from division results
        // Must be a separate compute pass so all division_execute workgroups have finished
        // writing division_flags and division_slot_assignments before we read them.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mutation Collect Candidates Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.collect_pipeline);
            pass.set_bind_group(0, collect_input_bg, &[]);
            pass.set_bind_group(1, collect_output_bg, &[]);

            // Dispatch at full capacity — collect shader reads cell_count_buffer internally
            let workgroups = (self.cell_capacity + 127) / 128;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Stage 2: Apply mutations to collected candidates
        // Must be a separate compute pass so all collect workgroups have finished
        // writing mutation_candidates before the mutation shader reads them.
        {
            // Advance RNG seed
            let rng_seed = self.rng_counter.fetch_add(1, Ordering::Relaxed);

            // Upload uniform params
            let params = MutationParamsUniform {
                radiation_level: self.radiation_level,
                rng_seed,
                current_frame,
                param_table_size: self.vulnerability_table.len() as u32,
                total_mode_count: MAX_TOTAL_MODES,
                max_modes_per_genome: 40,
                genome_ring_capacity: GENOME_RING_CAPACITY,
                _pad0: 0,
            };
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            // Clear mutation log count
            queue.write_buffer(&self.mutation_log_count_buffer, 0, bytemuck::bytes_of(&0u32));

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mutation Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, params_bg, &[]);
            pass.set_bind_group(1, candidates_bg, &[]);
            pass.set_bind_group(2, buffers_bg, &[]);

            // Dispatch enough workgroups for max candidates
            let workgroups = (MAX_MUTATION_CANDIDATES + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Stage 3: Periodic genome reference count synchronization + GC
        // Run every GC_INTERVAL_FRAMES (120 frames) to avoid race conditions
        // This gives mutations time to stabilize before GC recycles genomes
        let gc_frame = self.gc_frame_counter.fetch_add(1, Ordering::Relaxed);
        let should_run_gc = gc_frame % GC_INTERVAL_FRAMES == 0;
        
        if should_run_gc {
            // Step 0: Clear max_active_mode_offset before GC scan
            // genome_ring_state[4] = 0
            queue.write_buffer(&self.genome_ring_state_buffer, 16, bytemuck::bytes_of(&0u32));
            
            if let Some(sync_bg) = &self.ref_count_sync_bind_group {
                // Step 1: Clear all ref_counts to 0
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Ref Count Clear Pass"),
                        timestamp_writes: None,
                    });

                    pass.set_pipeline(&self.ref_count_clear_pipeline);
                    pass.set_bind_group(0, sync_bg, &[]);

                    let workgroups = (GENOME_RING_CAPACITY + 63) / 64;
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                // Step 2: Count active cells per genome
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Ref Count Count Pass"),
                        timestamp_writes: None,
                    });

                    pass.set_pipeline(&self.ref_count_count_pipeline);
                    pass.set_bind_group(0, sync_bg, &[]);

                    let workgroups = (self.cell_capacity + 63) / 64;
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                // Step 2b: Re-apply immortality for user genomes so GC never recycles them.
                // The clear pass zeroed their ref_counts; the count pass only adds live cells.
                // If there are no live cells yet (e.g. right after reset), user genomes would
                // have ref_count=0 and get recycled, breaking mutation for the new session.
                let immortal: u32 = u32::MAX;
                for i in 0..self.user_genome_count as u64 {
                    queue.write_buffer(
                        &self.genome_ref_counts_buffer,
                        i * std::mem::size_of::<u32>() as u64,
                        bytemuck::bytes_of(&immortal),
                    );
                }
            }

            // Step 3: Genome garbage collection
            // Recycle genomes with ref_count == 0 and track max active mode offset
            if let Some(gc_bg) = &self.gc_bind_group {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Genome GC Pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.gc_pipeline);
                pass.set_bind_group(0, gc_bg, &[]);

                let workgroups = (GENOME_RING_CAPACITY + 63) / 64;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            
            // Step 4: Mode buffer compaction
            // Reset next_mode_offset to max_active_mode_offset (computed by GC shader)
            // This reclaims all trailing unused mode buffer space.
            if self.mode_offset_reset_bind_group.is_none() {
                // Build bind group on first use
                self.mode_offset_reset_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Mode Offset Reset Bind Group"),
                    layout: &self.mode_offset_reset_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.genome_ring_state_buffer.as_entire_binding(),
                        },
                    ],
                }));
            }
            
            if let Some(reset_bg) = &self.mode_offset_reset_bind_group {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Mode Offset Reset Pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.mode_offset_reset_pipeline);
                pass.set_bind_group(0, reset_bg, &[]);

                // Single-thread dispatch (1,1,1)
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
    }

    /// Get the mutation candidates buffer (for the division shader to write to)
    pub fn mutation_candidates_buffer(&self) -> &wgpu::Buffer {
        &self.mutation_candidates_buffer
    }

    /// Get the mutation candidate count buffer (for the division shader to write to)
    pub fn mutation_candidate_count_buffer(&self) -> &wgpu::Buffer {
        &self.mutation_candidate_count_buffer
    }

    /// Debug: Read back genome_ring_state and print values
    pub fn debug_print_genome_ring_state(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Create staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Ring State Staging"),
            size: 20, // 5 x u32
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create encoder and copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Genome Ring State Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.genome_ring_state_buffer, 0, &staging_buffer, 0, 20);
        queue.submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let values: &[u32] = bytemuck::cast_slice(&data);
        println!("[MUTATION DEBUG] genome_ring_state:");
        println!("  [0] head: {}", values[0]);
        println!("  [1] tail: {}", values[1]);
        println!("  [2] next_id: {}", values[2]);
        println!("  [3] next_mode_offset: {} / {} ({:.1}%)", 
            values[3], MAX_TOTAL_MODES, (values[3] as f32 / MAX_TOTAL_MODES as f32) * 100.0);
        println!("  [4] max_active_mode_offset: {} / {} ({:.1}%)", 
            values[4], MAX_TOTAL_MODES, (values[4] as f32 / MAX_TOTAL_MODES as f32) * 100.0);
        
        drop(data);
        staging_buffer.unmap();
    }

    /// Get the mutation log buffer (for readback/UI display)
    pub fn mutation_log_buffer(&self) -> &wgpu::Buffer {
        &self.mutation_log_buffer
    }

    /// Get the mutation log count buffer
    pub fn mutation_log_count_buffer(&self) -> &wgpu::Buffer {
        &self.mutation_log_count_buffer
    }

    /// Get the genome metadata buffer (for other shaders that need genome info)
    pub fn genome_meta_buffer(&self) -> &wgpu::Buffer {
        &self.genome_meta_buffer
    }
}
