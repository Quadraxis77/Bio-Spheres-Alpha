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
/// (shares the 20K genome pool with user-created genomes)
const GENOME_RING_CAPACITY: u32 = 20_000;

/// Maximum modes across all genomes (must match triple_buffer.rs: 40 * 20_000)
const MAX_TOTAL_MODES: u32 = 800_000;

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

pub struct MutationSystem {
    // Compute pipelines
    pipeline: wgpu::ComputePipeline,
    collect_pipeline: wgpu::ComputePipeline,

    // Bind group layouts for mutation shader
    params_layout: wgpu::BindGroupLayout,
    candidates_layout: wgpu::BindGroupLayout,
    buffers_layout: wgpu::BindGroupLayout,

    // Bind group layouts for collect_candidates shader
    collect_input_layout: wgpu::BindGroupLayout,
    collect_output_layout: wgpu::BindGroupLayout,

    // Uniform buffer for MutationParams
    params_buffer: wgpu::Buffer,

    // Uniform buffer for CollectParams
    collect_params_buffer: wgpu::Buffer,

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

    // State
    rng_counter: AtomicU32,
    radiation_level: f32,
    cell_capacity: u32,
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
                // bindings 0-8: mode buffers (all read_write storage)
                Self::storage_rw_entry(0),  // mode_properties
                Self::storage_rw_entry(1),  // genome_mode_data
                Self::storage_rw_entry(2),  // child_mode_indices
                Self::storage_rw_entry(3),  // mode_cell_types
                Self::storage_rw_entry(4),  // parent_make_adhesion_flags
                Self::storage_rw_entry(5),  // child_a_keep_adhesion_flags
                Self::storage_rw_entry(6),  // child_b_keep_adhesion_flags
                Self::storage_rw_entry(7),  // glueocyte_env_adhesion_flags
                Self::storage_rw_entry(8),  // oculocyte_params
                // binding 9: mutation_log (read_write)
                Self::storage_rw_entry(9),
                // binding 10: mutation_log_count (read_write)
                Self::storage_rw_entry(10),
                // binding 11: mode_visuals (read_write)
                Self::storage_rw_entry(11),
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

        let vulnerability_table = Self::build_default_vulnerability_table();
        let table_size = (vulnerability_table.len() * std::mem::size_of::<MutationParamEntry>()) as u64;
        let vulnerability_table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mutation Vulnerability Table"),
            size: table_size.max(32), // minimum size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vulnerability_table_buffer, 0, bytemuck::cast_slice(&vulnerability_table));

        // Genome ring state: [head, tail, next_id, next_mode_offset]
        let genome_ring_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Genome Ring State"),
            size: 16, // 4 x u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            params_layout,
            candidates_layout,
            buffers_layout,
            collect_input_layout,
            collect_output_layout,
            params_buffer,
            collect_params_buffer,
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
            rng_counter: AtomicU32::new(0),
            radiation_level: 0.0,
            cell_capacity,
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
        vec![
            // === mode_properties (buffer_id=0) ===
            // Layout: 20 floats per mode, indexed as element_offset 0-19
            //  0: nutrient_gain_rate
            //  1: max_cell_size
            //  2: membrane_stiffness
            //  3: split_interval
            //  4: split_mass
            //  5: nutrient_priority
            //  6: swim_force
            //  7: prioritize_when_low (bool as f32)
            //  8: max_splits
            //  9: split_ratio
            // 10: flagellocyte_signal_channel
            // 11: flagellocyte_speed_a
            // 12: flagellocyte_speed_b
            // 13: flagellocyte_threshold_c
            // 14: flagellocyte_use_signal (bool as f32)
            // 15: min_adhesions
            // 16: max_adhesions

            // nutrient_gain_rate: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 0,
                weight: 2.0, min_delta: 0.01, max_delta: 0.1,
                min_value: 0.0, max_value: 5.0, data_type: data_type::CONTINUOUS_F32,
            },
            // max_cell_size: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 1,
                weight: 2.0, min_delta: 0.05, max_delta: 0.3,
                min_value: 0.5, max_value: 3.0, data_type: data_type::CONTINUOUS_F32,
            },
            // membrane_stiffness: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 2,
                weight: 1.5, min_delta: 1.0, max_delta: 10.0,
                min_value: 0.0, max_value: 200.0, data_type: data_type::CONTINUOUS_F32,
            },
            // split_interval: medium-high vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 3,
                weight: 3.0, min_delta: 0.5, max_delta: 3.0,
                min_value: 1.0, max_value: 60.0, data_type: data_type::CONTINUOUS_F32,
            },
            // split_mass (as nutrient threshold): medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 4,
                weight: 2.5, min_delta: 0.1, max_delta: 0.5,
                min_value: 1.2, max_value: 5.0, data_type: data_type::CONTINUOUS_F32,
            },
            // nutrient_priority: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 5,
                weight: 2.0, min_delta: 0.1, max_delta: 1.0,
                min_value: 0.1, max_value: 10.0, data_type: data_type::CONTINUOUS_F32,
            },
            // swim_force: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 6,
                weight: 2.0, min_delta: 0.01, max_delta: 0.1,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
            // prioritize_when_low: low vulnerability (boolean)
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 7,
                weight: 0.5, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 1.0, data_type: data_type::BOOLEAN,
            },
            // split_ratio: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 9,
                weight: 2.0, min_delta: 0.02, max_delta: 0.1,
                min_value: 0.1, max_value: 0.9, data_type: data_type::CONTINUOUS_F32,
            },
            // flagellocyte_speed_a: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 11,
                weight: 1.5, min_delta: 0.01, max_delta: 0.1,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
            // flagellocyte_speed_b: medium vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 12,
                weight: 1.5, min_delta: 0.01, max_delta: 0.1,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
            // flagellocyte_threshold_c: low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 13,
                weight: 1.0, min_delta: 0.5, max_delta: 5.0,
                min_value: -50.0, max_value: 50.0, data_type: data_type::CONTINUOUS_F32,
            },
            // min_adhesions: low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 15,
                weight: 0.5, min_delta: 1.0, max_delta: 1.0,
                min_value: 0.0, max_value: 10.0, data_type: data_type::INTEGER,
            },
            // max_adhesions: low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES, element_offset: 16,
                weight: 0.5, min_delta: 1.0, max_delta: 2.0,
                min_value: 0.0, max_value: 20.0, data_type: data_type::INTEGER,
            },

            // === mode_cell_types (buffer_id=1) ===
            // Cell type: very low vulnerability (dramatic change)
            MutationParamEntry {
                buffer_id: buffer_id::MODE_CELL_TYPES, element_offset: 0,
                weight: 0.3, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 9.0, data_type: data_type::INTEGER,
            },

            // === child_mode_indices (buffer_id=2) ===
            // child_a mode pointer: medium-high vulnerability (lineage divergence)
            MutationParamEntry {
                buffer_id: buffer_id::CHILD_MODE_INDICES, element_offset: 0,
                weight: 2.5, min_delta: 1.0, max_delta: 2.0,
                min_value: 0.0, max_value: 39.0, data_type: data_type::MODE_INDEX_CLAMP,
            },
            // child_b mode pointer: medium-high vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::CHILD_MODE_INDICES, element_offset: 1,
                weight: 2.5, min_delta: 1.0, max_delta: 2.0,
                min_value: 0.0, max_value: 39.0, data_type: data_type::MODE_INDEX_CLAMP,
            },

            // === Boolean flag buffers (buffer_id=3,4,5) ===
            // parent_make_adhesion: very low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::PARENT_MAKE_ADHESION, element_offset: 0,
                weight: 0.3, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 1.0, data_type: data_type::BOOLEAN,
            },
            // child_a_keep_adhesion: very low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::CHILD_A_KEEP_ADHESION, element_offset: 0,
                weight: 0.3, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 1.0, data_type: data_type::BOOLEAN,
            },
            // child_b_keep_adhesion: very low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::CHILD_B_KEEP_ADHESION, element_offset: 0,
                weight: 0.3, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 1.0, data_type: data_type::BOOLEAN,
            },

            // === glueocyte_env_adhesion (buffer_id=7) ===
            MutationParamEntry {
                buffer_id: buffer_id::GLUEOCYTE_ENV_ADHESION, element_offset: 0,
                weight: 0.2, min_delta: 0.0, max_delta: 0.0,
                min_value: 0.0, max_value: 1.0, data_type: data_type::BOOLEAN,
            },

            // === oculocyte_params (buffer_id=8) ===
            // sense_type: very low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::OCULOCYTE_PARAMS, element_offset: 0,
                weight: 0.2, min_delta: 1.0, max_delta: 1.0,
                min_value: 0.0, max_value: 3.0, data_type: data_type::INTEGER,
            },
            // signal_hops: low vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::OCULOCYTE_PARAMS, element_offset: 2,
                weight: 0.5, min_delta: 1.0, max_delta: 3.0,
                min_value: 1.0, max_value: 20.0, data_type: data_type::INTEGER,
            },

            // === mode_visuals (buffer_id=9) ===
            // Layout: 2 vec4 per mode — [0] = color (xyz=RGB, w=1.0), [1] = emissive_pad
            // color R: medium-high vulnerability (visible, interesting mutations)
            MutationParamEntry {
                buffer_id: buffer_id::MODE_VISUALS, element_offset: 0,
                weight: 1.5, min_delta: 0.01, max_delta: 0.15,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
            // color G: medium-high vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_VISUALS, element_offset: 1,
                weight: 1.5, min_delta: 0.01, max_delta: 0.15,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
            // color B: medium-high vulnerability
            MutationParamEntry {
                buffer_id: buffer_id::MODE_VISUALS, element_offset: 2,
                weight: 1.5, min_delta: 0.01, max_delta: 0.15,
                min_value: 0.0, max_value: 1.0, data_type: data_type::CONTINUOUS_F32,
            },
        ]
    }

    /// Set the global radiation level (0.0 = off, 1.0 = every division mutates)
    pub fn set_radiation_level(&mut self, level: f32) {
        self.radiation_level = level.clamp(0.0, 1.0);
    }

    /// Get the current radiation level
    pub fn radiation_level(&self) -> f32 {
        self.radiation_level
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

    /// Initialize genome metadata for user-created genomes.
    /// Called when genomes are first loaded or when a new genome is added.
    /// Maps each genome_id to its mode_count and base_mode_offset in the flat buffer.
    pub fn sync_genome_metadata(
        &self,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let mut meta = vec![GenomeMeta { mode_count: 0, base_mode_offset: 0, ref_count: 0, flags: 0 };
                            GENOME_RING_CAPACITY as usize];

        let mut offset = 0u32;
        for (i, genome) in genomes.iter().enumerate() {
            if i >= GENOME_RING_CAPACITY as usize {
                break;
            }
            meta[i] = GenomeMeta {
                mode_count: genome.modes.len() as u32,
                base_mode_offset: offset,
                ref_count: 0, // Will be set by GPU or tracked separately
                flags: 0,
            };
            offset += genome.modes.len() as u32;
        }

        queue.write_buffer(&self.genome_meta_buffer, 0, bytemuck::cast_slice(&meta));

        // Initialize ring state: next_id starts after user genomes, next_mode_offset after their modes
        let initial_ring_state: [u32; 4] = [
            0,                          // head
            0,                          // tail
            genomes.len() as u32,       // next_genome_id (after user genomes)
            offset,                     // next_mode_offset (after user genome modes)
        ];
        queue.write_buffer(&self.genome_ring_state_buffer, 0, bytemuck::cast_slice(&initial_ring_state));
    }

    /// Build bind groups. Call after triple buffer or genome buffers change.
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        genome_ids_buffer: &wgpu::Buffer,
        mode_indices_buffer: &wgpu::Buffer,
        mode_properties_buffer: &wgpu::Buffer,
        genome_mode_data_buffer: &wgpu::Buffer,
        child_mode_indices_buffer: &wgpu::Buffer,
        mode_cell_types_buffer: &wgpu::Buffer,
        parent_make_adhesion_flags_buffer: &wgpu::Buffer,
        child_a_keep_adhesion_flags_buffer: &wgpu::Buffer,
        child_b_keep_adhesion_flags_buffer: &wgpu::Buffer,
        glueocyte_env_adhesion_flags_buffer: &wgpu::Buffer,
        oculocyte_params_buffer: &wgpu::Buffer,
        mode_visuals_buffer: &wgpu::Buffer,
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
                wgpu::BindGroupEntry { binding: 0, resource: mode_properties_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: genome_mode_data_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: child_mode_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mode_cell_types_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: parent_make_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: child_a_keep_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: child_b_keep_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: glueocyte_env_adhesion_flags_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: oculocyte_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.mutation_log_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.mutation_log_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: mode_visuals_buffer.as_entire_binding() },
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

    /// Dispatch the full mutation pipeline: collect_candidates → mutation.
    /// Call this AFTER the division execute pass completes.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        current_frame: u32,
        total_mode_count: u32,
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
                total_mode_count: total_mode_count.min(MAX_TOTAL_MODES),
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
    }

    /// Get the mutation candidates buffer (for the division shader to write to)
    pub fn mutation_candidates_buffer(&self) -> &wgpu::Buffer {
        &self.mutation_candidates_buffer
    }

    /// Get the mutation candidate count buffer (for the division shader to write to)
    pub fn mutation_candidate_count_buffer(&self) -> &wgpu::Buffer {
        &self.mutation_candidate_count_buffer
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
