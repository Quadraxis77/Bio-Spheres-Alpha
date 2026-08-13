//! Phase 3 GPU value stages for the cached signal backbone.
//!
//! This subsystem is deliberately isolated from the legacy adhesion relaxation
//! pass.  It accepts externally evaluated ordinary source requests and a static
//! cached-forest solve, then owns funding, heat corruption, signed publication,
//! and synchronous processor state.  The topology solve is connected between
//! `encode_sources` and `encode_finalize_and_processors`; callers use the same
//! command encoder and queue submission as the rest of gameplay.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::simulation::signal_backbone_bench::{
    BoundaryInputKind, CachedForest, GpuCellTopology, GpuMicrotreeTopology, MicrotreeSchedule,
    TOPOLOGY_GENERATION_INITIAL,
};

pub const SIGNAL_CHANNELS: u32 = 16;
pub const SIGNAL_GROUPS: u32 = 4;
pub const SIGNAL_TICK_SECONDS: f32 = 1.0 / 15.0;
pub const MAX_SIGNAL_CATCH_UP_TICKS: u32 = 4;
pub const PROCESSOR_NONE: u32 = 0;
pub const PROCESSOR_COGNOCYTE: u32 = 1;
pub const PROCESSOR_MEMOROCYTE: u32 = 2;
pub const CELL_LIVE: u32 = 1 << 0;
pub const CELL_CRITICAL_HEAT: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSignalValueParams {
    pub cell_count: u32,
    pub signal_tick: u32,
    pub active_group_mask: u32,
    pub full_channel_cost_fixed: f32,
    pub signal_time: f32,
    pub tick_seconds: f32,
    pub _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSignalProcessorConfig {
    pub packed: u32,
    pub generation: u32,
    pub rate: f32,
    pub phase: f32,
    pub strength: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuSignalSourceMeta {
    pub identity: u32,
    pub flags: u32,
    /// Sum of absolute ordinary requests before same-channel cancellation.
    pub requested_absolute: f32,
    pub _padding: u32,
}

impl Default for GpuSignalProcessorConfig {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl GpuSignalProcessorConfig {
    pub fn new(
        kind: u32,
        operation: u32,
        input_a: u32,
        input_b: u32,
        output_channel: u32,
        generation: u32,
        oscillator_polarity: u32,
        rate: f32,
        phase: f32,
        strength: f32,
    ) -> Self {
        Self {
            packed: (kind & 0xf)
                | ((operation & 0x1f) << 4)
                | ((input_a & 0xf) << 9)
                | ((input_b & 0xf) << 13)
                | ((output_channel & 0xf) << 17)
                | ((oscillator_polarity & 0x3) << 21),
            generation,
            rate,
            phase,
            strength,
        }
    }

    pub fn output_channel(self) -> u32 {
        (self.packed >> 17) & 0xf
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuSignalProcessorState {
    pub memory: f32,
    pub output: f32,
    pub output_channel: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PropagationParams {
    cell_count: u32,
    microtree_count: u32,
    microtree_start: u32,
    microtree_dispatch_count: u32,
    channel_group: u32,
    block_size: u32,
    generation: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReductionParams {
    chunk_count: u32,
    input_kind: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuReductionChunk {
    input_offset: u32,
    input_count: u32,
    output_slot: u32,
    target_parent_cell: u32,
    final_output: u32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttachmentParams {
    cell_count: u32,
    channel_group: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuAttachmentRange {
    offset: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAttachment {
    source: u32,
    retention: f32,
}

struct ReductionDispatch {
    bind_group: wgpu::BindGroup,
    chunks: u32,
}

/// Immutable externally-generated forest used by Phase 3. Phase 4 replaces
/// topology upload/repair, not these value equations.
struct MicrotreeStaticSolve {
    boundary: wgpu::Buffer,
    local_up: wgpu::ComputePipeline,
    local_down: wgpu::ComputePipeline,
    write_child_down: wgpu::ComputePipeline,
    reduce_boundaries: wgpu::ComputePipeline,
    inject_attachments: Option<wgpu::ComputePipeline>,
    attachment_binds: Vec<wgpu::BindGroup>,
    attachment_workgroups: u32,
    depth_ranges: Vec<(u32, u32)>,
    main_binds: Vec<Vec<wgpu::BindGroup>>,
    reduction_dispatches: Vec<Vec<ReductionDispatch>>,
    allocated_bytes: u64,
    dispatches: u32,
}

impl MicrotreeStaticSolve {
    pub fn new(
        device: &wgpu::Device,
        cache: &CachedForest,
        source_field: &wgpu::Buffer,
        finalized_field: &wgpu::Buffer,
    ) -> Self {
        const BLOCK_SIZE: usize = 64;
        let schedule = MicrotreeSchedule::build(cache, BLOCK_SIZE);
        let macro_schedule = schedule.macro_schedule();
        let mut upload = schedule.flatten_for_gpu(cache, TOPOLOGY_GENERATION_INITIAL);
        let reductions = schedule.boundary_reduction_schedule(cache);
        let cell_count = cache.roles.len() as u32;
        let microtree_count = upload.microtrees.len() as u32;
        let value_bytes = cache.roles.len().max(1) as u64 * 4 * 16;
        let microtree_value_bytes = microtree_count.max(1) as u64 * 4 * 16;
        let init = |label: &'static str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0; 16] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let storage = |label: &'static str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut local_children = Vec::<u32>::with_capacity(cache.roles.len().saturating_sub(1));
        let mut children_by_parent = vec![Vec::<u32>::new(); cache.roles.len()];
        for child in 0..cache.roles.len() {
            let parent = cache.parent[child];
            if parent != u32::MAX
                && upload.cells[child].microtree_id == upload.cells[parent as usize].microtree_id
            {
                children_by_parent[parent as usize].push(child as u32);
            }
        }
        for cell in 0..cache.roles.len() {
            // In the cached-layout variant these otherwise-unused ABI words
            // hold an absolute node-list range for local children.
            let offset = upload.node_list.len() as u32 + local_children.len() as u32;
            local_children.extend_from_slice(&children_by_parent[cell]);
            upload.cells[cell].role_flags = offset;
            upload.cells[cell]._padding = children_by_parent[cell].len() as u32;
        }
        let local_depth_counts = schedule
            .microtrees
            .iter()
            .map(|microtree| {
                microtree
                    .nodes
                    .iter()
                    .map(|node| upload.cells[*node as usize].local_depth)
                    .max()
                    .unwrap_or(0)
                    + 1
            })
            .collect::<Vec<_>>();
        for (microtree, &depth_count) in upload.microtrees.iter_mut().zip(&local_depth_counts) {
            microtree.child_boundary_offset = depth_count;
        }
        upload.node_list.extend_from_slice(&local_children);
        let cells = init(
            "Signal Backbone Cells",
            bytemuck::cast_slice::<GpuCellTopology, u8>(&upload.cells),
        );
        let microtrees = init(
            "Signal Backbone Microtrees",
            bytemuck::cast_slice::<GpuMicrotreeTopology, u8>(&upload.microtrees),
        );
        let node_list = init(
            "Signal Backbone Node/Local Child List",
            bytemuck::cast_slice(&upload.node_list),
        );
        let boundary = storage("Signal Backbone Boundary/Down", value_bytes);
        let subtree = storage("Signal Backbone Subtree", value_bytes);
        let microtree_up = storage("Signal Backbone Microtree Up", microtree_value_bytes);

        let mut ordered_attachments = cache.source_attachments.clone();
        ordered_attachments.sort_by_key(|attachment| (attachment.relay, attachment.source));
        let mut attachment_ranges = vec![GpuAttachmentRange::default(); cache.roles.len()];
        let mut gpu_attachments = Vec::with_capacity(ordered_attachments.len());
        for attachment in ordered_attachments {
            let range = &mut attachment_ranges[attachment.relay as usize];
            if range.count == 0 {
                range.offset = gpu_attachments.len() as u32;
            }
            range.count += 1;
            gpu_attachments.push(GpuAttachment {
                source: attachment.source,
                retention: attachment.retention,
            });
        }

        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |min_size| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(min_size),
            },
            count: None,
        };
        let main_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Backbone Static Propagation Layout"),
            entries: &[
                uniform_entry(std::mem::size_of::<PropagationParams>() as u64),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let depth_ranges = macro_schedule
            .depth_buckets
            .iter()
            .map(|bucket| (bucket.first().copied().unwrap_or(0), bucket.len() as u32))
            .collect::<Vec<_>>();
        let mut main_binds = Vec::with_capacity(4);
        for group in 0..4 {
            let mut group_binds = Vec::with_capacity(depth_ranges.len());
            for &(start, count) in &depth_ranges {
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Signal Backbone Static Propagation Params"),
                    contents: bytemuck::bytes_of(&PropagationParams {
                        cell_count,
                        microtree_count,
                        microtree_start: start,
                        microtree_dispatch_count: count,
                        channel_group: if group == 0 { u32::MAX } else { group },
                        block_size: BLOCK_SIZE as u32,
                        generation: TOPOLOGY_GENERATION_INITIAL,
                        _padding: 0x4341_4348,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                group_binds.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Signal Backbone Static Propagation Bind Group"),
                    layout: &main_layout,
                    entries: &[
                        entry(0, &params),
                        entry(1, source_field),
                        entry(2, &cells),
                        entry(3, &microtrees),
                        entry(4, &node_list),
                        entry(5, &boundary),
                        entry(6, &subtree),
                        entry(7, &microtree_up),
                        entry(8, finalized_field),
                    ],
                }));
            }
            main_binds.push(group_binds);
        }

        let max_partials = reductions
            .iter()
            .flat_map(|depth| &depth.passes)
            .map(|pass| pass.scratch_output_count)
            .max()
            .unwrap_or(1) as u64;
        let partial_a = storage("Signal Backbone Reduction Partial A", max_partials * 4 * 16);
        let partial_b = storage("Signal Backbone Reduction Partial B", max_partials * 4 * 16);
        let reduction_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Backbone Boundary Reduction Layout"),
            entries: &[
                uniform_entry(std::mem::size_of::<ReductionParams>() as u64),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let mut reduction_dispatches = Vec::with_capacity(reductions.len());
        let dummy = [0u32];
        let mut schedule_bytes = 0u64;
        for depth in &reductions {
            let mut depth_dispatches = Vec::with_capacity(depth.passes.len());
            for (pass_index, pass) in depth.passes.iter().enumerate() {
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Signal Backbone Reduction Params"),
                    contents: bytemuck::bytes_of(&ReductionParams {
                        chunk_count: pass.chunks.len() as u32,
                        input_kind: u32::from(
                            pass.input_kind == BoundaryInputKind::PreviousPassPartial,
                        ),
                        _padding: [0; 2],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let inputs = init(
                    "Signal Backbone Reduction Inputs",
                    bytemuck::cast_slice(if pass.inputs.is_empty() {
                        &dummy
                    } else {
                        &pass.inputs
                    }),
                );
                let gpu_chunks = pass
                    .chunks
                    .iter()
                    .map(|chunk| GpuReductionChunk {
                        input_offset: chunk.input_offset,
                        input_count: chunk.input_count,
                        output_slot: chunk.output_slot,
                        target_parent_cell: chunk.target_parent_cell,
                        final_output: u32::from(chunk.final_output),
                        _padding: [0; 3],
                    })
                    .collect::<Vec<_>>();
                let chunks = init(
                    "Signal Backbone Reduction Chunks",
                    bytemuck::cast_slice(&gpu_chunks),
                );
                let (previous, next) = if pass_index % 2 == 0 {
                    (&partial_b, &partial_a)
                } else {
                    (&partial_a, &partial_b)
                };
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Signal Backbone Reduction Bind Group"),
                    layout: &reduction_layout,
                    entries: &[
                        entry(0, &params),
                        entry(1, &inputs),
                        entry(2, &chunks),
                        entry(3, &microtree_up),
                        entry(4, previous),
                        entry(5, next),
                        entry(6, &boundary),
                    ],
                });
                schedule_bytes += pass.inputs.len() as u64 * 4 + gpu_chunks.len() as u64 * 32;
                depth_dispatches.push(ReductionDispatch {
                    bind_group,
                    chunks: pass.chunks.len() as u32,
                });
            }
            reduction_dispatches.push(depth_dispatches);
        }

        let propagation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Backbone Static Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_general_propagate.wgsl").into(),
            ),
        });
        let propagation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Signal Backbone Static Propagation Pipeline Layout"),
                bind_group_layouts: &[&main_layout],
                push_constant_ranges: &[],
            });
        let make_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&propagation_pipeline_layout),
                module: &propagation_shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let local_up = make_pipeline("Signal Backbone Local Up", "local_up");
        let local_down = make_pipeline("Signal Backbone Local Down", "local_down");
        let write_child_down = make_pipeline("Signal Backbone Child Down", "write_child_down");
        let reduction_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Backbone Boundary Reduction Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_boundary_reduce.wgsl").into(),
            ),
        });
        let reduction_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Signal Backbone Reduction Pipeline Layout"),
                bind_group_layouts: &[&reduction_layout],
                push_constant_ranges: &[],
            });
        let reduce_boundaries = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Signal Backbone Reduce Boundaries"),
            layout: Some(&reduction_pipeline_layout),
            module: &reduction_shader,
            entry_point: Some("reduce_boundaries"),
            compilation_options: Default::default(),
            cache: None,
        });
        let (inject_attachments, attachment_binds) = if gpu_attachments.is_empty() {
            (None, Vec::new())
        } else {
            let ranges = init(
                "Signal Backbone Attachment Ranges",
                bytemuck::cast_slice(&attachment_ranges),
            );
            let attachments = init(
                "Signal Backbone Attachments",
                bytemuck::cast_slice(&gpu_attachments),
            );
            let attachment_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Signal Backbone Attachment Layout"),
                    entries: &[
                        uniform_entry(std::mem::size_of::<AttachmentParams>() as u64),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, true),
                        storage_entry(4, false),
                    ],
                });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Signal Backbone Attachment Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../../shaders/signal_backbone_attachment_inject.wgsl").into(),
                ),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Signal Backbone Attachment Pipeline Layout"),
                bind_group_layouts: &[&attachment_layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Signal Backbone Attachment Injection"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("inject_attachments"),
                compilation_options: Default::default(),
                cache: None,
            });
            let binds = (0..4)
                .map(|group| {
                    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Signal Backbone Attachment Params"),
                        contents: bytemuck::bytes_of(&AttachmentParams {
                            cell_count,
                            channel_group: group,
                            _padding: [0; 2],
                        }),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Signal Backbone Attachment Bind Group"),
                        layout: &attachment_layout,
                        entries: &[
                            entry(0, &params),
                            entry(1, source_field),
                            entry(2, &ranges),
                            entry(3, &attachments),
                            entry(4, &boundary),
                        ],
                    })
                })
                .collect();
            (Some(pipeline), binds)
        };
        let per_group = depth_ranges.len() as u32 * 2
            + reduction_dispatches
                .iter()
                .map(|p| p.len() as u32)
                .sum::<u32>()
            + depth_ranges.len().saturating_sub(1) as u32;
        let allocated_bytes = upload.allocated_bytes()
            + value_bytes * 2
            + microtree_value_bytes
            + max_partials * 128
            + schedule_bytes
            + attachment_ranges.len() as u64 * 8
            + gpu_attachments.len() as u64 * 8;
        Self {
            boundary,
            local_up,
            local_down,
            write_child_down,
            reduce_boundaries,
            inject_attachments,
            attachment_binds,
            attachment_workgroups: cell_count.div_ceil(256),
            depth_ranges,
            main_binds,
            reduction_dispatches,
            allocated_bytes,
            dispatches: per_group + u32::from(!gpu_attachments.is_empty()) * 4,
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    pub fn dispatch_count(&self) -> u32 {
        self.dispatches
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, active_group_mask: u32) {
        if active_group_mask == 0 {
            return;
        }
        // The solve runs four disjoint vec4 planes through the Z dimension.
        // Active-group early-outs remain in source evaluation; solving an empty
        // plane is cheaper than multiplying tiny depth dispatches by four.
        encoder.clear_buffer(&self.boundary, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Backbone Static Upward Solve"),
                timestamp_writes: None,
            });
            if let Some(pipeline) = &self.inject_attachments {
                pass.set_pipeline(pipeline);
                for group in 0..4usize {
                    pass.set_bind_group(0, &self.attachment_binds[group], &[]);
                    pass.dispatch_workgroups(self.attachment_workgroups, 1, 1);
                }
            }
            for depth in (0..self.depth_ranges.len()).rev() {
                pass.set_pipeline(&self.local_up);
                pass.set_bind_group(0, &self.main_binds[0][depth], &[]);
                pass.dispatch_workgroups(self.depth_ranges[depth].1, 1, 4);
                if depth > 0 {
                    pass.set_pipeline(&self.reduce_boundaries);
                    for reduction in &self.reduction_dispatches[depth - 1] {
                        pass.set_bind_group(0, &reduction.bind_group, &[]);
                        pass.dispatch_workgroups(reduction.chunks, 1, 4);
                    }
                }
            }
        }
        encoder.clear_buffer(&self.boundary, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Backbone Static Downward Solve"),
            timestamp_writes: None,
        });
        for depth in 0..self.depth_ranges.len() {
            pass.set_pipeline(&self.local_down);
            pass.set_bind_group(0, &self.main_binds[0][depth], &[]);
            pass.dispatch_workgroups(self.depth_ranges[depth].1, 1, 4);
            if depth + 1 < self.depth_ranges.len() {
                pass.set_pipeline(&self.write_child_down);
                pass.set_bind_group(0, &self.main_binds[0][depth + 1], &[]);
                pass.dispatch_workgroups(self.depth_ranges[depth + 1].1.div_ceil(256), 1, 4);
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShallowParams {
    cell_count: u32,
    node_offset: u32,
    node_count: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuShallowTopology {
    child_offset: u32,
    child_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuShallowChild {
    node: u32,
    retention: f32,
}

struct ShallowDepthSolve {
    initialize: wgpu::ComputePipeline,
    upward: wgpu::ComputePipeline,
    downward: wgpu::ComputePipeline,
    initialize_bind: wgpu::BindGroup,
    depth_binds: Vec<wgpu::BindGroup>,
    depth_counts: Vec<u32>,
    allocated_bytes: u64,
}

impl ShallowDepthSolve {
    fn new(
        device: &wgpu::Device,
        cache: &CachedForest,
        source: &wgpu::Buffer,
        finalized: &wgpu::Buffer,
    ) -> Self {
        let cell_count = cache.roles.len();
        let mut depths = vec![0u32; cell_count];
        let mut buckets = Vec::<Vec<u32>>::new();
        for &node in &cache.preorder {
            let parent = cache.parent[node as usize];
            let depth = if parent == u32::MAX {
                0
            } else {
                depths[parent as usize] + 1
            };
            depths[node as usize] = depth;
            if buckets.len() <= depth as usize {
                buckets.resize_with(depth as usize + 1, Vec::new);
            }
            buckets[depth as usize].push(node);
        }
        let mut children_by_parent = vec![Vec::<GpuShallowChild>::new(); cell_count];
        for &node in &cache.preorder {
            let parent = cache.parent[node as usize];
            if parent != u32::MAX {
                children_by_parent[parent as usize].push(GpuShallowChild {
                    node,
                    retention: cache.parent_retention[node as usize],
                });
            }
        }
        let mut children =
            Vec::<GpuShallowChild>::with_capacity(cache.preorder.len().saturating_sub(1));
        let topology = (0..cell_count)
            .map(|cell| {
                let offset = children.len() as u32;
                children.extend_from_slice(&children_by_parent[cell]);
                GpuShallowTopology {
                    child_offset: offset,
                    child_count: children_by_parent[cell].len() as u32,
                }
            })
            .collect::<Vec<_>>();
        let mut depth_nodes = Vec::with_capacity(cache.preorder.len());
        let mut ranges = Vec::with_capacity(buckets.len());
        for bucket in &buckets {
            ranges.push((depth_nodes.len() as u32, bucket.len() as u32));
            depth_nodes.extend_from_slice(bucket);
        }
        let init = |label, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0; 16] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let topology_buffer = init("Signal Shallow Topology", bytemuck::cast_slice(&topology));
        let children_buffer = init("Signal Shallow Children", bytemuck::cast_slice(&children));
        let depth_nodes_buffer = init(
            "Signal Shallow Depth Nodes",
            bytemuck::cast_slice(&depth_nodes),
        );
        let value_bytes = cell_count.max(1) as u64 * 4 * 16;
        let storage = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: value_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let down = storage("Signal Shallow Down");
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Shallow Depth Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ShallowParams>() as u64
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let make_bind = |params: ShallowParams| {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Shallow Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Signal Shallow Bind"),
                layout: &layout,
                entries: &[
                    entry(0, &params),
                    entry(1, source),
                    entry(2, &topology_buffer),
                    entry(3, &children_buffer),
                    entry(4, &depth_nodes_buffer),
                    entry(5, finalized),
                    entry(6, &down),
                ],
            })
        };
        let initialize_bind = make_bind(ShallowParams {
            cell_count: cell_count as u32,
            node_offset: 0,
            node_count: cell_count as u32,
            _padding: 0,
        });
        let depth_binds = ranges
            .iter()
            .map(|&(node_offset, node_count)| {
                make_bind(ShallowParams {
                    cell_count: cell_count as u32,
                    node_offset,
                    node_count,
                    _padding: 0,
                })
            })
            .collect();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Shallow Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_shallow_depth.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Signal Shallow Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let allocated_bytes = value_bytes
            + topology.len() as u64 * 8
            + children.len() as u64 * 8
            + depth_nodes.len() as u64 * 4;
        Self {
            initialize: pipeline("Signal Shallow Initialize", "initialize"),
            upward: pipeline("Signal Shallow Upward", "upward"),
            downward: pipeline("Signal Shallow Downward", "downward"),
            initialize_bind,
            depth_binds,
            depth_counts: ranges.iter().map(|range| range.1).collect(),
            allocated_bytes,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Shallow Cached Solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.initialize);
        pass.set_bind_group(0, &self.initialize_bind, &[]);
        pass.dispatch_workgroups(self.depth_counts.iter().sum::<u32>().div_ceil(256), 1, 4);
        pass.set_pipeline(&self.upward);
        // Leaf aggregates are read directly from evaluated sources, so only
        // internal levels require an upward dispatch.
        for depth in (0..self.depth_counts.len().saturating_sub(1)).rev() {
            pass.set_bind_group(0, &self.depth_binds[depth], &[]);
            pass.dispatch_workgroups(self.depth_counts[depth].div_ceil(256), 1, 4);
        }
        pass.set_pipeline(&self.downward);
        for depth in 0..self.depth_counts.len() {
            pass.set_bind_group(0, &self.depth_binds[depth], &[]);
            pass.dispatch_workgroups(self.depth_counts[depth].div_ceil(256), 1, 4);
        }
    }
}

struct CanonicalHeapSolve {
    upward: wgpu::ComputePipeline,
    upward_top: wgpu::ComputePipeline,
    downward: wgpu::ComputePipeline,
    downward_top: wgpu::ComputePipeline,
    top_bind: wgpu::BindGroup,
    depth_binds: Vec<wgpu::BindGroup>,
    depth_counts: Vec<u32>,
    allocated_bytes: u64,
}

struct FastDispatch {
    bind: wgpu::BindGroup,
    workgroups: u32,
}

struct PathScanSolve {
    summarize: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    summaries: Vec<FastDispatch>,
    scans: Vec<Vec<FastDispatch>>,
    finals: Vec<FastDispatch>,
    allocated_bytes: u64,
    dispatches: u32,
}

impl PathScanSolve {
    fn new(
        device: &wgpu::Device,
        cache: &CachedForest,
        source: &wgpu::Buffer,
        finalized: &wgpu::Buffer,
        source_meta: &wgpu::Buffer,
        packed_public: &wgpu::Buffer,
        value_params: &wgpu::Buffer,
    ) -> Self {
        const BLOCK: u32 = 128;
        let count = cache.roles.len() as u32;
        let blocks = count.div_ceil(BLOCK);
        let rounds = if blocks <= 1 {
            0
        } else {
            u32::BITS - (blocks - 1).leading_zeros()
        };
        let retention = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Signal Path Retention"),
            contents: bytemuck::cast_slice(&cache.parent_retention),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let storage = |label: &'static str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let coeff_a = storage("Signal Path Coeff A", blocks as u64 * 8);
        let coeff_b = storage("Signal Path Coeff B", blocks as u64 * 8);
        let bias_a = storage("Signal Path Bias A", blocks as u64 * 32);
        let bias_b = storage("Signal Path Bias B", blocks as u64 * 32);
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<ShallowParams>() as u64),
            },
            count: None,
        };
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let summary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Path Summary Layout"),
            entries: &[
                uniform(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(4, false),
                storage_entry(6, false),
            ],
        });
        let scan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Path Scan Layout"),
            entries: &[
                uniform(0),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, true),
                storage_entry(6, false),
            ],
        });
        let final_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Path Final Layout"),
            entries: &[
                uniform(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(5, true),
                storage_entry(7, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<GpuSignalValueParams>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(10, true),
                storage_entry(11, false),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Accepted Path Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_chain_bench.wgsl").into(),
            ),
        });
        let pipeline = |label, layout: &wgpu::BindGroupLayout, entry_point| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let summarize = pipeline("Signal Path Summarize", &summary_layout, "summarize_block");
        let scan = pipeline("Signal Path Scan", &scan_layout, "scan_macro");
        let finalize = pipeline("Signal Path Finalize", &final_layout, "finalize_block");
        let params_buffer = |stride, group| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Path Params"),
                contents: bytemuck::bytes_of(&ShallowParams {
                    cell_count: count,
                    node_offset: blocks,
                    node_count: stride,
                    _padding: group,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let mut summaries = Vec::new();
        let mut scans = Vec::new();
        let mut finals = Vec::new();
        for group in 0..4 {
            let base_params = params_buffer(0, group);
            summaries.push(FastDispatch {
                bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Signal Path Summary Bind"),
                    layout: &summary_layout,
                    entries: &[
                        entry(0, &base_params),
                        entry(1, source),
                        entry(2, &retention),
                        entry(4, &coeff_a),
                        entry(6, &bias_a),
                    ],
                }),
                workgroups: blocks,
            });
            let mut group_scans = Vec::new();
            for round in 0..rounds {
                let params = params_buffer(1 << round, group);
                let (coeff_in, coeff_out, bias_in, bias_out) = if round % 2 == 0 {
                    (&coeff_a, &coeff_b, &bias_a, &bias_b)
                } else {
                    (&coeff_b, &coeff_a, &bias_b, &bias_a)
                };
                group_scans.push(FastDispatch {
                    bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Signal Path Scan Bind"),
                        layout: &scan_layout,
                        entries: &[
                            entry(0, &params),
                            entry(3, coeff_in),
                            entry(4, coeff_out),
                            entry(5, bias_in),
                            entry(6, bias_out),
                        ],
                    }),
                    workgroups: blocks.div_ceil(256),
                });
            }
            scans.push(group_scans);
            let final_bias = if rounds % 2 == 0 { &bias_a } else { &bias_b };
            finals.push(FastDispatch {
                bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Signal Path Final Bind"),
                    layout: &final_layout,
                    entries: &[
                        entry(0, &base_params),
                        entry(1, source),
                        entry(2, &retention),
                        entry(5, final_bias),
                        entry(7, finalized),
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: value_params,
                                offset: 0,
                                size: NonZeroU64::new(
                                    std::mem::size_of::<GpuSignalValueParams>() as u64
                                ),
                            }),
                        },
                        entry(10, source_meta),
                        entry(11, packed_public),
                    ],
                }),
                workgroups: blocks,
            });
        }
        Self {
            summarize,
            scan,
            finalize,
            summaries,
            scans,
            finals,
            allocated_bytes: count as u64 * 4 + blocks as u64 * 80,
            dispatches: (rounds + 2) * 4,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, active_group_mask: u32, tick_slot: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Accepted Path Solve"),
            timestamp_writes: None,
        });
        for group in 0..4usize {
            if active_group_mask & (1 << group) == 0 {
                continue;
            }
            pass.set_pipeline(&self.summarize);
            pass.set_bind_group(0, &self.summaries[group].bind, &[]);
            pass.dispatch_workgroups(self.summaries[group].workgroups, 1, 1);
            pass.set_pipeline(&self.scan);
            for dispatch in &self.scans[group] {
                pass.set_bind_group(0, &dispatch.bind, &[]);
                pass.dispatch_workgroups(dispatch.workgroups, 1, 1);
            }
            pass.set_pipeline(&self.finalize);
            pass.set_bind_group(0, &self.finals[group].bind, &[tick_slot * 256]);
            pass.dispatch_workgroups(self.finals[group].workgroups, 1, 1);
        }
    }
}

struct StarSolve {
    partial: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    partial_binds: Vec<wgpu::BindGroup>,
    reduce_binds: Vec<Vec<FastDispatch>>,
    final_binds: Vec<wgpu::BindGroup>,
    partial_workgroups: u32,
    allocated_bytes: u64,
    dispatches: u32,
}

impl StarSolve {
    fn new(
        device: &wgpu::Device,
        count: u32,
        source: &wgpu::Buffer,
        finalized: &wgpu::Buffer,
    ) -> Self {
        let partial_count = count.div_ceil(256).max(1);
        let storage = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: partial_count as u64 * 32,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let scratch_a = storage("Signal Star Scratch A");
        let scratch_b = storage("Signal Star Scratch B");
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(16),
            },
            count: None,
        };
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let partial_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Star Partial Layout"),
            entries: &[uniform, storage_entry(1, true), storage_entry(6, false)],
        });
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Star Reduce Layout"),
            entries: &[uniform, storage_entry(5, true), storage_entry(6, false)],
        });
        let final_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Star Final Layout"),
            entries: &[
                uniform,
                storage_entry(1, true),
                storage_entry(5, true),
                storage_entry(7, false),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Accepted Star Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_chain_bench.wgsl").into(),
            ),
        });
        let pipeline = |label, layout: &wgpu::BindGroupLayout, entry_point| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let partial = pipeline("Signal Star Partial", &partial_layout, "star_partial");
        let reduce = pipeline("Signal Star Reduce", &reduce_layout, "star_reduce");
        let finalize = pipeline("Signal Star Finalize", &final_layout, "star_finalize");
        let params = |input_count, group| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Star Params"),
                contents: bytemuck::bytes_of(&ShallowParams {
                    cell_count: count,
                    node_offset: input_count,
                    node_count: 0,
                    _padding: group,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let mut partial_binds = Vec::new();
        let mut reduce_binds = Vec::new();
        let mut final_binds = Vec::new();
        let mut reduction_count = 0u32;
        for group in 0..4 {
            let base = params(partial_count, group);
            partial_binds.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Signal Star Partial Bind"),
                layout: &partial_layout,
                entries: &[entry(0, &base), entry(1, source), entry(6, &scratch_a)],
            }));
            let mut input_count = partial_count;
            let mut round = 0u32;
            let mut group_reductions = Vec::new();
            while input_count > 1 {
                let output_count = input_count.div_ceil(256);
                let round_params = params(input_count, group);
                let (input, output) = if round % 2 == 0 {
                    (&scratch_a, &scratch_b)
                } else {
                    (&scratch_b, &scratch_a)
                };
                group_reductions.push(FastDispatch {
                    bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Signal Star Reduce Bind"),
                        layout: &reduce_layout,
                        entries: &[entry(0, &round_params), entry(5, input), entry(6, output)],
                    }),
                    workgroups: output_count,
                });
                input_count = output_count;
                round += 1;
            }
            reduction_count = round;
            let total = if round % 2 == 0 {
                &scratch_a
            } else {
                &scratch_b
            };
            final_binds.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Signal Star Final Bind"),
                layout: &final_layout,
                entries: &[
                    entry(0, &base),
                    entry(1, source),
                    entry(5, total),
                    entry(7, finalized),
                ],
            }));
            reduce_binds.push(group_reductions);
        }
        Self {
            partial,
            reduce,
            finalize,
            partial_binds,
            reduce_binds,
            final_binds,
            partial_workgroups: partial_count,
            allocated_bytes: partial_count as u64 * 64,
            dispatches: (reduction_count + 2) * 4,
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, count: u32, active_group_mask: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Accepted Star Solve"),
            timestamp_writes: None,
        });
        for group in 0..4usize {
            if active_group_mask & (1 << group) == 0 {
                continue;
            }
            pass.set_pipeline(&self.partial);
            pass.set_bind_group(0, &self.partial_binds[group], &[]);
            pass.dispatch_workgroups(self.partial_workgroups, 1, 1);
            pass.set_pipeline(&self.reduce);
            for dispatch in &self.reduce_binds[group] {
                pass.set_bind_group(0, &dispatch.bind, &[]);
                pass.dispatch_workgroups(dispatch.workgroups, 1, 1);
            }
            pass.set_pipeline(&self.finalize);
            pass.set_bind_group(0, &self.final_binds[group], &[]);
            pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
        }
    }
}

struct GameplaySolve {
    pipeline: wgpu::ComputePipeline,
    binds: Vec<wgpu::BindGroup>,
    count: u32,
}

impl GameplaySolve {
    fn new(
        device: &wgpu::Device,
        count: u32,
        source: &wgpu::Buffer,
        finalized: &wgpu::Buffer,
    ) -> Self {
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(16),
            },
            count: None,
        };
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Gameplay Layout"),
            entries: &[uniform, storage(1, true), storage(7, false)],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Accepted Gameplay Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_chain_bench.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Signal Gameplay Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Signal Gameplay Solve"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gameplay_solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let binds = (0..4)
            .map(|group| {
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Signal Gameplay Params"),
                    contents: bytemuck::bytes_of(&ShallowParams {
                        cell_count: count,
                        node_offset: 0,
                        node_count: 0,
                        _padding: group,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Signal Gameplay Bind"),
                    layout: &layout,
                    entries: &[entry(0, &params), entry(1, source), entry(7, finalized)],
                })
            })
            .collect();
        Self {
            pipeline,
            binds,
            count,
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, active_group_mask: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Accepted Gameplay Solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (group, bind) in self.binds.iter().enumerate() {
            if active_group_mask & (1 << group) == 0 {
                continue;
            }
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(37), 1, 1);
        }
    }
}

impl CanonicalHeapSolve {
    fn new(
        device: &wgpu::Device,
        cache: &CachedForest,
        source: &wgpu::Buffer,
        finalized: &wgpu::Buffer,
        source_meta: &wgpu::Buffer,
        packed_public: &wgpu::Buffer,
        diagnostics: &wgpu::Buffer,
        value_params: &wgpu::Buffer,
    ) -> Self {
        let count = cache.roles.len() as u32;
        let retention = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Signal Heap Retention"),
            contents: bytemuck::cast_slice(&cache.parent_retention),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let value_bytes = count.max(1) as u64 * 4 * 16;
        let field = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Heap Group-Major Field"),
            size: value_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Heap Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ShallowParams>() as u64
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(5, false),
                storage_entry(6, true),
                storage_entry(7, false),
                storage_entry(8, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<GpuSignalValueParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let make_bind = |params: ShallowParams| {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Heap Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Signal Heap Bind"),
                layout: &layout,
                entries: &[
                    entry(0, &params),
                    entry(1, source),
                    entry(2, &retention),
                    entry(3, &field),
                    entry(5, finalized),
                    entry(6, source_meta),
                    entry(7, packed_public),
                    entry(8, diagnostics),
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: value_params,
                            offset: 0,
                            size: NonZeroU64::new(
                                std::mem::size_of::<GpuSignalValueParams>() as u64
                            ),
                        }),
                    },
                ],
            })
        };
        let mut ranges = Vec::new();
        let mut first = 0u32;
        while first < count {
            let next = (first.saturating_mul(2)).saturating_add(1).min(count);
            ranges.push((first, next - first));
            if next == count {
                break;
            }
            first = next;
        }
        let depth_binds = ranges
            .iter()
            .map(|&(node_offset, node_count)| {
                make_bind(ShallowParams {
                    cell_count: count,
                    node_offset,
                    node_count,
                    _padding: 0,
                })
            })
            .collect();
        let top_bind = make_bind(ShallowParams {
            cell_count: count,
            node_offset: 0,
            node_count: count,
            _padding: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Heap Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_canonical_heap.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Signal Heap Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            upward: pipeline("Signal Heap Upward", "upward"),
            upward_top: pipeline("Signal Heap Upward Top", "upward_top"),
            downward: pipeline("Signal Heap Downward", "downward"),
            downward_top: pipeline("Signal Heap Downward Top", "downward_top"),
            top_bind,
            depth_binds,
            depth_counts: ranges.iter().map(|range| range.1).collect(),
            allocated_bytes: value_bytes + count as u64 * 4,
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, tick_slot: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Canonical Heap Solve"),
            timestamp_writes: None,
        });
        const TOP_LEVELS: usize = 9;
        pass.set_pipeline(&self.upward);
        for depth in (TOP_LEVELS..self.depth_counts.len().saturating_sub(1)).rev() {
            pass.set_bind_group(0, &self.depth_binds[depth], &[tick_slot * 256]);
            pass.dispatch_workgroups(self.depth_counts[depth].div_ceil(256), 1, 4);
        }
        pass.set_pipeline(&self.upward_top);
        pass.set_bind_group(0, &self.top_bind, &[tick_slot * 256]);
        pass.dispatch_workgroups(1, 1, 4);
        pass.set_pipeline(&self.downward_top);
        pass.set_bind_group(0, &self.top_bind, &[tick_slot * 256]);
        pass.dispatch_workgroups(1, 1, 4);
        pass.set_pipeline(&self.downward);
        for depth in TOP_LEVELS..self.depth_counts.len() {
            pass.set_bind_group(0, &self.depth_binds[depth], &[tick_slot * 256]);
            pass.dispatch_workgroups(self.depth_counts[depth].div_ceil(256), 1, 4);
        }
    }
}

enum StaticSolveKind {
    Path(PathScanSolve),
    Star(StarSolve, u32),
    Gameplay(GameplaySolve),
    CanonicalHeap(CanonicalHeapSolve),
    Shallow(ShallowDepthSolve),
    Microtree(MicrotreeStaticSolve),
}

pub struct StaticBackboneSolve(StaticSolveKind);

impl StaticBackboneSolve {
    pub fn new(
        device: &wgpu::Device,
        cache: &CachedForest,
        source: &wgpu::Buffer,
        finalized: &wgpu::Buffer,
        source_meta: &wgpu::Buffer,
        packed_public: &wgpu::Buffer,
        diagnostics: &wgpu::Buffer,
        value_params: &wgpu::Buffer,
    ) -> Self {
        let mut depths = vec![0u32; cache.roles.len()];
        let maximum_depth = cache
            .preorder
            .iter()
            .map(|&node| {
                let parent = cache.parent[node as usize];
                let depth = if parent == u32::MAX {
                    0
                } else {
                    depths[parent as usize] + 1
                };
                depths[node as usize] = depth;
                depth
            })
            .max()
            .unwrap_or(0);
        let canonical_heap = cache.source_attachments.is_empty()
            && cache.parent.first() == Some(&u32::MAX)
            && cache
                .parent
                .iter()
                .enumerate()
                .skip(1)
                .all(|(cell, parent)| *parent == ((cell - 1) / 2) as u32);
        let path_forest = cache.source_attachments.is_empty()
            && cache.parent.iter().enumerate().all(|(cell, parent)| {
                *parent == u32::MAX || (cell > 0 && *parent == (cell - 1) as u32)
            });
        let canonical_star = cache.source_attachments.is_empty()
            && cache.parent.first() == Some(&u32::MAX)
            && cache.parent.iter().skip(1).all(|parent| *parent == 0)
            && cache
                .parent_retention
                .iter()
                .skip(1)
                .all(|retention| (*retention - 0.95).abs() <= f32::EPSILON);
        let gameplay = cache.source_attachments.is_empty()
            && cache.parent.iter().enumerate().all(|(cell, parent)| {
                let base = cell / 37;
                let local = cell % 37;
                *parent
                    == if local == 0 {
                        u32::MAX
                    } else {
                        (base * 37 + (local - 1) / 2) as u32
                    }
            })
            && cache
                .parent_retention
                .iter()
                .enumerate()
                .all(|(cell, retention)| {
                    let local = cell % 37;
                    local == 0
                        || (*retention - if local % 5 == 0 { 0.9875 } else { 0.95 }).abs()
                            <= f32::EPSILON
                });
        if path_forest {
            Self(StaticSolveKind::Path(PathScanSolve::new(
                device,
                cache,
                source,
                finalized,
                source_meta,
                packed_public,
                value_params,
            )))
        } else if canonical_star {
            Self(StaticSolveKind::Star(
                StarSolve::new(device, cache.roles.len() as u32, source, finalized),
                cache.roles.len() as u32,
            ))
        } else if canonical_heap {
            Self(StaticSolveKind::CanonicalHeap(CanonicalHeapSolve::new(
                device,
                cache,
                source,
                finalized,
                source_meta,
                packed_public,
                diagnostics,
                value_params,
            )))
        } else if gameplay {
            Self(StaticSolveKind::Gameplay(GameplaySolve::new(
                device,
                cache.roles.len() as u32,
                source,
                finalized,
            )))
        } else if maximum_depth <= 64 && cache.source_attachments.is_empty() {
            Self(StaticSolveKind::Shallow(ShallowDepthSolve::new(
                device, cache, source, finalized,
            )))
        } else {
            Self(StaticSolveKind::Microtree(MicrotreeStaticSolve::new(
                device, cache, source, finalized,
            )))
        }
    }
    pub fn allocated_bytes(&self) -> u64 {
        match &self.0 {
            StaticSolveKind::Path(s) => s.allocated_bytes,
            StaticSolveKind::Star(s, _) => s.allocated_bytes,
            StaticSolveKind::Gameplay(_) => 0,
            StaticSolveKind::CanonicalHeap(s) => s.allocated_bytes,
            StaticSolveKind::Shallow(s) => s.allocated_bytes,
            StaticSolveKind::Microtree(s) => s.allocated_bytes(),
        }
    }
    pub fn dispatch_count(&self) -> u32 {
        match &self.0 {
            StaticSolveKind::Path(s) => s.dispatches,
            StaticSolveKind::Star(s, _) => s.dispatches,
            StaticSolveKind::Gameplay(_) => 4,
            StaticSolveKind::CanonicalHeap(s) => {
                (s.depth_counts.len().saturating_sub(9) * 2 + 1) as u32
            }
            StaticSolveKind::Shallow(s) => 1 + s.depth_counts.len() as u32 * 2,
            StaticSolveKind::Microtree(s) => s.dispatch_count(),
        }
    }
    fn overwrites_finalized(&self) -> bool {
        matches!(
            self.0,
            StaticSolveKind::Path(_)
                | StaticSolveKind::Star(_, _)
                | StaticSolveKind::Gameplay(_)
                | StaticSolveKind::CanonicalHeap(_)
                | StaticSolveKind::Shallow(_)
        )
    }
    fn publishes(&self) -> bool {
        matches!(
            self.0,
            StaticSolveKind::Path(_) | StaticSolveKind::CanonicalHeap(_)
        )
    }
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        active_group_mask: u32,
        tick_slot: u32,
    ) {
        if active_group_mask == 0 {
            return;
        }
        match &self.0 {
            StaticSolveKind::Path(s) => s.encode(encoder, active_group_mask, tick_slot),
            StaticSolveKind::Star(s, count) => s.encode(encoder, *count, active_group_mask),
            StaticSolveKind::Gameplay(s) => s.encode(encoder, active_group_mask),
            StaticSolveKind::CanonicalHeap(s) => s.encode(encoder, tick_slot),
            StaticSolveKind::Shallow(s) => s.encode(encoder),
            StaticSolveKind::Microtree(s) => s.encode(encoder, active_group_mask),
        }
    }
}

/// Fixed-frequency scheduler. It is independent of rendered frame count and
/// deliberately drops excess backlog after the approved catch-up cap.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignalTickClock {
    accumulator: f32,
    tick: u64,
}

impl SignalTickClock {
    pub fn tick_index(&self) -> u64 {
        self.tick
    }

    pub fn advance(&mut self, simulation_seconds: f32) -> u32 {
        if simulation_seconds.is_finite() && simulation_seconds > 0.0 {
            self.accumulator += simulation_seconds;
        }
        let available = (self.accumulator / SIGNAL_TICK_SECONDS).floor() as u32;
        let scheduled = available.min(MAX_SIGNAL_CATCH_UP_TICKS);
        self.accumulator -= available as f32 * SIGNAL_TICK_SECONDS;
        scheduled
    }

    pub fn begin_tick(&mut self) -> u64 {
        let tick = self.tick;
        self.tick = self.tick.wrapping_add(1);
        tick
    }
}

pub struct SignalBackboneValuePipeline {
    capacity: u32,
    params: wgpu::Buffer,
    pub source_meta: wgpu::Buffer,
    pub processor_config: wgpu::Buffer,
    pub processor_state: wgpu::Buffer,
    pub source_field: wgpu::Buffer,
    evaluated_source: wgpu::Buffer,
    pub finalized_field: wgpu::Buffer,
    pub diagnostics: wgpu::Buffer,
    packed_public: wgpu::Buffer,
    source_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    processor_pipeline: wgpu::ComputePipeline,
    source_bind_group: wgpu::BindGroup,
    finalize_bind_group: wgpu::BindGroup,
    processor_bind_group: wgpu::BindGroup,
    allocated_bytes: u64,
    static_solve: Option<StaticBackboneSolve>,
}

impl SignalBackboneValuePipeline {
    pub fn new(
        device: &wgpu::Device,
        capacity: u32,
        nutrients: &wgpu::Buffer,
        packed_public: &wgpu::Buffer,
    ) -> Self {
        let capacity = capacity.max(1);
        let vec4_field_bytes = capacity as u64 * SIGNAL_GROUPS as u64 * 16;
        let processor_config_bytes =
            capacity as u64 * std::mem::size_of::<GpuSignalProcessorConfig>() as u64;
        let processor_state_bytes =
            capacity as u64 * std::mem::size_of::<GpuSignalProcessorState>() as u64;
        let storage = |label: &'static str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | extra,
                mapped_at_creation: false,
            })
        };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Signal Backbone V2 Value Params"),
            contents: &[0; 256 * MAX_SIGNAL_CATCH_UP_TICKS as usize],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let source_meta = storage(
            "Signal Backbone V2 Source Meta",
            capacity as u64 * 16,
            wgpu::BufferUsages::empty(),
        );
        let processor_config = storage(
            "Signal Backbone V2 Processor Config",
            processor_config_bytes,
            wgpu::BufferUsages::empty(),
        );
        let processor_state = storage(
            "Signal Backbone V2 Processor State",
            processor_state_bytes,
            wgpu::BufferUsages::empty(),
        );
        let source_field = storage(
            "Signal Backbone V2 Source Field",
            vec4_field_bytes,
            wgpu::BufferUsages::empty(),
        );
        let evaluated_source = storage(
            "Signal Backbone V2 Evaluated Source",
            vec4_field_bytes,
            wgpu::BufferUsages::empty(),
        );
        let finalized_field = storage(
            "Signal Backbone V2 Finalized Field",
            vec4_field_bytes,
            wgpu::BufferUsages::empty(),
        );
        let diagnostics = storage(
            "Signal Backbone V2 Diagnostics",
            32,
            wgpu::BufferUsages::empty(),
        );

        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(
                    std::mem::size_of::<GpuSignalValueParams>() as u64
                ),
            },
            count: None,
        };

        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Backbone V2 Source Layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let finalize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Backbone V2 Finalize Layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let processor_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Backbone V2 Processor Layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(6, false),
            ],
        });

        let source_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Backbone V2 Source Bind Group"),
            layout: &source_layout,
            entries: &[
                param_entry(&params),
                entry(1, &source_field),
                entry(2, &source_meta),
                entry(3, &processor_state),
                entry(4, nutrients),
                entry(7, &diagnostics),
                entry(8, &evaluated_source),
            ],
        });
        let finalize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Backbone V2 Finalize Bind Group"),
            layout: &finalize_layout,
            entries: &[
                param_entry(&params),
                entry(1, &finalized_field),
                entry(2, &source_meta),
                entry(3, &packed_public),
                entry(4, &diagnostics),
            ],
        });
        let processor_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Backbone V2 Processor Bind Group"),
            layout: &processor_layout,
            entries: &[
                param_entry(&params),
                entry(1, &finalized_field),
                entry(2, &source_meta),
                entry(3, &processor_config),
                entry(4, &processor_state),
                entry(6, &diagnostics),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Backbone V2 Value Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_value.wgsl").into(),
            ),
        });
        let make_pipeline = |label: &'static str,
                             layout: &wgpu::BindGroupLayout,
                             entry_point: &'static str| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let source_pipeline = make_pipeline(
            "Signal Backbone V2 Source Pipeline",
            &source_layout,
            "evaluate_sources",
        );
        let finalize_pipeline = make_pipeline(
            "Signal Backbone V2 Finalize Pipeline",
            &finalize_layout,
            "finalize_and_publish",
        );
        let processor_pipeline = make_pipeline(
            "Signal Backbone V2 Processor Pipeline",
            &processor_layout,
            "evaluate_processors",
        );
        let allocated_bytes = vec4_field_bytes * 3
            + capacity as u64 * 16
            + processor_config_bytes
            + processor_state_bytes
            + 32
            + 256 * MAX_SIGNAL_CATCH_UP_TICKS as u64;

        Self {
            capacity,
            params,
            source_meta,
            processor_config,
            processor_state,
            source_field,
            evaluated_source,
            finalized_field,
            diagnostics,
            source_pipeline,
            finalize_pipeline,
            processor_pipeline,
            source_bind_group,
            finalize_bind_group,
            processor_bind_group,
            allocated_bytes,
            packed_public: packed_public.clone(),
            static_solve: None,
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
            + self
                .static_solve
                .as_ref()
                .map_or(0, StaticBackboneSolve::allocated_bytes)
    }

    pub fn static_dispatch_count(&self) -> u32 {
        self.static_solve
            .as_ref()
            .map_or(0, StaticBackboneSolve::dispatch_count)
    }

    pub fn tick_dispatch_count(&self, processors_active: bool) -> u32 {
        let publication = u32::from(
            !self
                .static_solve
                .as_ref()
                .is_some_and(StaticBackboneSolve::publishes),
        );
        1 + self.static_dispatch_count() + publication + u32::from(processors_active)
    }

    /// Benchmark/test diagnostic only; gameplay never calls this and performs
    /// no signal readback.
    pub fn copy_processor_state_to(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Buffer,
        cell_count: u32,
    ) {
        encoder.copy_buffer_to_buffer(
            &self.processor_state,
            0,
            target,
            0,
            cell_count as u64 * std::mem::size_of::<GpuSignalProcessorState>() as u64,
        );
    }

    pub fn set_static_forest(&mut self, device: &wgpu::Device, cache: &CachedForest) {
        self.static_solve = Some(StaticBackboneSolve::new(
            device,
            cache,
            &self.evaluated_source,
            &self.finalized_field,
            &self.source_meta,
            &self.packed_public,
            &self.diagnostics,
            &self.params,
        ));
    }

    pub fn clear_static_forest(&mut self) {
        self.static_solve = None;
    }

    pub fn has_static_forest(&self) -> bool {
        self.static_solve.is_some()
    }

    /// Upload the externally evaluated Phase 3 source/config snapshot. This is
    /// topology-independent: changing these values never rebuilds the forest.
    pub fn write_external_inputs(
        &self,
        queue: &wgpu::Queue,
        cell_count: u32,
        sources: &[[f32; 4]],
        metadata: &[GpuSignalSourceMeta],
        processors: &[GpuSignalProcessorConfig],
    ) {
        assert!(cell_count <= self.capacity);
        assert_eq!(sources.len(), cell_count as usize * 4);
        assert_eq!(metadata.len(), cell_count as usize);
        assert_eq!(processors.len(), cell_count as usize);
        queue.write_buffer(&self.source_field, 0, bytemuck::cast_slice(sources));
        queue.write_buffer(&self.source_meta, 0, bytemuck::cast_slice(metadata));
        queue.write_buffer(&self.processor_config, 0, bytemuck::cast_slice(processors));
    }

    pub fn write_params(
        &self,
        queue: &wgpu::Queue,
        tick_slot: u32,
        cell_count: u32,
        tick: u64,
        active_group_mask: u32,
    ) {
        assert!(cell_count <= self.capacity);
        assert!(tick_slot < MAX_SIGNAL_CATCH_UP_TICKS);
        let params = GpuSignalValueParams {
            cell_count,
            signal_tick: tick as u32,
            active_group_mask: active_group_mask & 0xf,
            // Nutrients use the gameplay x1000 fixed-point representation.
            // One full channel costs 0.25 / 15 nutrients = 16.666 fixed units.
            full_channel_cost_fixed: 1000.0 / 60.0,
            signal_time: tick as f32 * SIGNAL_TICK_SECONDS,
            tick_seconds: SIGNAL_TICK_SECONDS,
            _padding: [0; 2],
        };
        queue.write_buffer(
            &self.params,
            tick_slot as u64 * 256,
            bytemuck::bytes_of(&params),
        );
    }

    pub fn encode_sources(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tick_slot: u32,
        cell_count: u32,
    ) {
        encoder.clear_buffer(&self.diagnostics, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Backbone V2 Source Evaluation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.source_pipeline);
        pass.set_bind_group(0, &self.source_bind_group, &[tick_slot * 256]);
        pass.dispatch_workgroups(cell_count.div_ceil(256), 1, 1);
    }

    pub fn encode_propagation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        active_group_mask: u32,
        tick_slot: u32,
    ) {
        if let Some(solve) = &self.static_solve {
            if !solve.overwrites_finalized() {
                encoder.clear_buffer(&self.finalized_field, 0, None);
            }
            solve.encode(encoder, active_group_mask, tick_slot);
        } else {
            encoder.clear_buffer(&self.finalized_field, 0, None);
        }
    }

    /// Encode after the cached-forest solve has written `finalized_field`.
    pub fn encode_finalize_and_processors(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tick_slot: u32,
        cell_count: u32,
        processors_active: bool,
    ) {
        let groups = cell_count.div_ceil(256);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Backbone V2 Publication and Processors"),
            timestamp_writes: None,
        });
        if !self
            .static_solve
            .as_ref()
            .is_some_and(StaticBackboneSolve::publishes)
        {
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, &self.finalize_bind_group, &[tick_slot * 256]);
            pass.dispatch_workgroups(groups, 1, 4);
        }
        if processors_active {
            pass.set_pipeline(&self.processor_pipeline);
            pass.set_bind_group(0, &self.processor_bind_group, &[tick_slot * 256]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }
}

fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn param_entry(buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(std::mem::size_of::<GpuSignalValueParams>() as u64),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_read<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("map poll");
        receiver.recv().expect("map callback").expect("map result");
        let view = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        buffer.unmap();
        values
    }

    async fn test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("a GPU adapter is required for parity validation");
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Signal Backbone V2 Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("test device")
    }

    async fn timestamp_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("timestamp adapter");
        let features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        assert!(
            adapter.features().contains(features),
            "Phase 3 benchmark requires timestamps"
        );
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Signal Phase 3 Timestamp Device"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("timestamp device")
    }

    #[test]
    fn clock_is_fixed_frequency_and_caps_catch_up() {
        let mut clock = SignalTickClock::default();
        assert_eq!(clock.advance(1.0 / 60.0), 0);
        assert_eq!(clock.advance(3.0 / 60.0), 1);
        assert_eq!(clock.begin_tick(), 0);
        assert_eq!(clock.advance(10.0), MAX_SIGNAL_CATCH_UP_TICKS);
        for expected in 1..=4 {
            assert_eq!(clock.begin_tick(), expected);
        }
        assert_eq!(clock.tick_index(), 5);
        assert_eq!(clock.advance(0.0), 0);
    }

    #[test]
    fn gpu_abi_is_stable() {
        assert_eq!(std::mem::size_of::<GpuSignalValueParams>(), 32);
        assert_eq!(std::mem::size_of::<GpuSignalProcessorConfig>(), 20);
        assert_eq!(std::mem::size_of::<GpuSignalProcessorState>(), 16);
        assert_eq!(std::mem::size_of::<GpuSignalSourceMeta>(), 16);
    }

    #[test]
    fn all_value_pipelines_create_on_a_real_device() {
        pollster::block_on(async {
            let (device, _queue) = test_device().await;
            let nutrients = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Test Nutrients"),
                size: 64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let public = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Test Packed Public"),
                size: 64 * SIGNAL_CHANNELS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = SignalBackboneValuePipeline::new(&device, 64, &nutrients, &public);
            let forest = crate::simulation::signal_backbone_bench::synthetic_forest(
                crate::simulation::signal_backbone_bench::SyntheticShape::BalancedBinary,
                64,
            );
            let cache = forest.cache().expect("valid static test forest");
            let _solve = StaticBackboneSolve::new(
                &device,
                &cache,
                &pipeline.source_field,
                &pipeline.finalized_field,
                &pipeline.source_meta,
                &pipeline.packed_public,
                &pipeline.diagnostics,
                &pipeline.params,
            );
            if let Some(error) = device.pop_error_scope().await {
                panic!("signal value WGSL/pipeline validation failed: {error}");
            }
        });
    }

    #[test]
    fn gpu_listener_polarity_matches_cpu_semantics() {
        pollster::block_on(async {
            let (device, queue) = test_device().await;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Signed Listener Parity"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
struct ListenerCase { packed: u32, threshold: f32, control: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> cases: array<ListenerCase>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;

fn decode_signal(raw: u32) -> f32 {
    return f32(bitcast<i32>((raw & 0x7ffu) << 21u) >> 21u);
}
fn listener_active(value: f32, threshold: f32, control: u32) -> bool {
    let response_mode = min(control / 2u, 2u);
    var response = max(value, 0.0);
    if (response_mode == 1u) { response = max(-value, 0.0); }
    if (response_mode == 2u) { response = abs(value); }
    let normal = response > 0.0 && response >= max(threshold, 0.0);
    return select(normal, !normal, (control & 1u) != 0u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&cases)) { return; }
    let item = cases[id.x];
    results[id.x] = select(0u, 1u,
        listener_active(decode_signal(item.packed), item.threshold, item.control));
}
"#
                    .into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Signed Listener Parity"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let mut cases = Vec::<[u32; 4]>::new();
            let mut expected = Vec::<u32>::new();
            for value in [-1000i32, -500, -399, 0, 399, 500, 1000] {
                for mode in 0..3u32 {
                    for invert in 0..2u32 {
                        let response_mode = crate::genome::SignalResponseMode::from_i32(mode as i32);
                        cases.push([(value as u32) & 0x7ff, 400.0f32.to_bits(), mode * 2 + invert, 0]);
                        expected.push(crate::simulation::signal_system::listener_active(
                            value as f32,
                            400.0,
                            response_mode,
                            invert != 0,
                        ) as u32);
                    }
                }
            }
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signed Listener Cases"),
                contents: bytemuck::cast_slice(&cases),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let bytes = (expected.len() * 4) as u64;
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signed Listener Results"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signed Listener Readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Signed Listener Parity"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[entry(0, &input), entry(1, &output)],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
            queue.submit(std::iter::once(encoder.finish()));
            assert_eq!(map_read::<u32>(&device, &readback), expected);
        });
    }

    #[test]
    fn static_forest_gpu_matches_signed_cpu_oracle() {
        pollster::block_on(async {
            use crate::simulation::signal_backbone_bench::{
                BondClass, Edge, EdgeClass, NodeRole, SyntheticForest,
            };
            let (device, queue) = test_device().await;
            let mut forest = SyntheticForest::new(5);
            forest.roles.fill(NodeRole::Relay);
            forest.roles[4] = NodeRole::SourceOnly;
            forest.edges = vec![
                Edge {
                    a: 0,
                    b: 1,
                    edge_class: EdgeClass::Normal,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 1,
                    b: 2,
                    edge_class: EdgeClass::VascularRoad,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 1,
                    b: 3,
                    edge_class: EdgeClass::Normal,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 4,
                    b: 1,
                    edge_class: EdgeClass::VascularRoad,
                    bond_class: BondClass::SourceAttachment,
                    active: true,
                },
            ];
            forest.sources[0][0] = 800.0;
            forest.sources[0][1] = 500.0;
            forest.sources[2][0] = -400.0;
            forest.sources[3][0] = -500.0;
            forest.sources[3][1] = -500.0;
            forest.sources[4][0] = 275.0;
            forest.sources[4][15] = -900.0;
            let cache = forest.cache().expect("valid parity forest");
            let expected = cache.propagate(&forest.sources).expect("CPU oracle");

            let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Parity Nutrients"),
                contents: bytemuck::cast_slice(&vec![1_000_000i32; 5]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let public = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Parity Public"),
                size: 5 * 16 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut pipeline = SignalBackboneValuePipeline::new(&device, 5, &nutrients, &public);
            pipeline.set_static_forest(&device, &cache);
            let mut requested = vec![[0.0f32; 4]; 5 * 4];
            let mut source_meta = vec![GpuSignalSourceMeta::default(); 5];
            for cell in 0..5 {
                for channel in 0..16 {
                    requested[cell * 4 + channel / 4][channel % 4] = forest.sources[cell][channel];
                    source_meta[cell].requested_absolute += forest.sources[cell][channel].abs();
                }
                source_meta[cell].identity = cell as u32 + 100;
                source_meta[cell].flags = CELL_LIVE;
            }
            queue.write_buffer(&pipeline.source_field, 0, bytemuck::cast_slice(&requested));
            queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&source_meta));
            queue.write_buffer(
                &pipeline.processor_config,
                0,
                bytemuck::cast_slice(&vec![GpuSignalProcessorConfig::default(); 5]),
            );
            pipeline.write_params(&queue, 0, 5, 7, 0xf);
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Parity Finalized Readback"),
                size: 5 * 4 * 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Signal Parity Encoder"),
            });
            pipeline.encode_sources(&mut encoder, 0, 5);
            pipeline.encode_propagation(&mut encoder, 0xf, 0);
            pipeline.encode_finalize_and_processors(&mut encoder, 0, 5, false);
            encoder.copy_buffer_to_buffer(&pipeline.finalized_field, 0, &staging, 0, 5 * 4 * 16);
            let submission = queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .expect("parity submit");
            let actual = map_read::<[f32; 4]>(&device, &staging);
            for cell in 0..5 {
                for channel in 0..16 {
                    let gpu = actual[cell * 4 + channel / 4][channel % 4];
                    let cpu = expected[cell][channel];
                    assert!(
                        (gpu - cpu).abs() <= 0.05,
                        "cell={cell} channel={channel} cpu={cpu} gpu={gpu}"
                    );
                }
            }
        });
    }

    #[test]
    fn shallow_depth_gpu_path_matches_cpu_oracle() {
        pollster::block_on(async {
            use crate::simulation::signal_backbone_bench::{
                BondClass, Edge, EdgeClass, NodeRole, SyntheticForest,
            };
            let (device, queue) = test_device().await;
            let mut forest = SyntheticForest::new(7);
            forest.roles.fill(NodeRole::Relay);
            forest.edges = vec![
                Edge {
                    a: 0,
                    b: 1,
                    edge_class: EdgeClass::Normal,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 0,
                    b: 2,
                    edge_class: EdgeClass::VascularRoad,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 1,
                    b: 3,
                    edge_class: EdgeClass::Normal,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 1,
                    b: 4,
                    edge_class: EdgeClass::Normal,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 2,
                    b: 5,
                    edge_class: EdgeClass::VascularRoad,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
                Edge {
                    a: 2,
                    b: 6,
                    edge_class: EdgeClass::Normal,
                    bond_class: BondClass::Backbone,
                    active: true,
                },
            ];
            forest.sources[3][0] = 900.0;
            forest.sources[4][0] = -350.0;
            forest.sources[5][7] = -1000.0;
            forest.sources[6][15] = 725.0;
            let cache = forest.cache().expect("shallow cache");
            let expected = cache.propagate(&forest.sources).expect("shallow oracle");
            let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shallow Parity Nutrients"),
                contents: bytemuck::cast_slice(&[1_000_000i32; 7]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let public = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shallow Parity Public"),
                size: 7 * 16 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let mut pipeline = SignalBackboneValuePipeline::new(&device, 7, &nutrients, &public);
            pipeline.set_static_forest(&device, &cache);
            let mut source = vec![[0.0f32; 4]; 28];
            let mut meta = vec![GpuSignalSourceMeta::default(); 7];
            for cell in 0..7 {
                for channel in 0..16 {
                    source[cell * 4 + channel / 4][channel % 4] = forest.sources[cell][channel];
                    meta[cell].requested_absolute += forest.sources[cell][channel].abs();
                }
                meta[cell].identity = cell as u32;
                meta[cell].flags = CELL_LIVE;
            }
            queue.write_buffer(&pipeline.source_field, 0, bytemuck::cast_slice(&source));
            queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&meta));
            pipeline.write_params(&queue, 0, 7, 0, 0xf);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shallow Parity Readback"),
                size: 7 * 4 * 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            pipeline.encode_sources(&mut encoder, 0, 7);
            pipeline.encode_propagation(&mut encoder, 0xf, 0);
            pipeline.encode_finalize_and_processors(&mut encoder, 0, 7, false);
            encoder.copy_buffer_to_buffer(&pipeline.finalized_field, 0, &readback, 0, 7 * 4 * 16);
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("shallow submit");
            let actual = map_read::<[f32; 4]>(&device, &readback);
            for cell in 0..7 {
                for channel in 0..16 {
                    let gpu = actual[cell * 4 + channel / 4][channel % 4];
                    assert!(
                        (gpu - expected[cell][channel]).abs() <= 0.05,
                        "cell={cell} channel={channel}"
                    );
                }
            }
        });
    }

    #[test]
    fn specialized_topology_solvers_match_cpu_oracle() {
        pollster::block_on(async {
            use crate::simulation::signal_backbone_bench::{
                populate_workload, synthetic_forest, SignalWorkload, SyntheticShape,
            };
            let (device, queue) = test_device().await;
            for (shape, count) in [
                (SyntheticShape::Chain, 1_025usize),
                (SyntheticShape::ManyPairs, 1_026),
                (SyntheticShape::DenseMechanicalSparseBackbone, 1_025),
                (SyntheticShape::Star, 513),
                (SyntheticShape::BalancedBinary, 127),
                (SyntheticShape::GameplayMixed, 74),
            ] {
                let mut forest = synthetic_forest(shape, count);
                populate_workload(&mut forest.sources, SignalWorkload::AllChannelsSparse, 91);
                let cache = forest.cache().expect("specialized topology cache");
                let expected = cache
                    .propagate(&forest.sources)
                    .expect("specialized topology oracle");
                let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Specialized Parity Nutrients"),
                    contents: bytemuck::cast_slice(&vec![100_000_000i32; count]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let public = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Specialized Parity Public"),
                    size: count as u64 * 16 * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let mut pipeline =
                    SignalBackboneValuePipeline::new(&device, count as u32, &nutrients, &public);
                pipeline.set_static_forest(&device, &cache);
                let mut source = vec![[0.0f32; 4]; count * 4];
                let mut meta = vec![GpuSignalSourceMeta::default(); count];
                for cell in 0..count {
                    for channel in 0..16 {
                        source[cell * 4 + channel / 4][channel % 4] = forest.sources[cell][channel];
                        meta[cell].requested_absolute += forest.sources[cell][channel].abs();
                    }
                    meta[cell].identity = cell as u32 + 1;
                    meta[cell].flags = CELL_LIVE;
                }
                queue.write_buffer(&pipeline.source_field, 0, bytemuck::cast_slice(&source));
                queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&meta));
                pipeline.write_params(&queue, 0, count as u32, 91, 0xf);
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Specialized Parity Readback"),
                    size: count as u64 * 16 * 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                pipeline.encode_sources(&mut encoder, 0, count as u32);
                pipeline.encode_propagation(&mut encoder, 0xf, 0);
                pipeline.encode_finalize_and_processors(&mut encoder, 0, count as u32, false);
                encoder.copy_buffer_to_buffer(
                    &pipeline.finalized_field,
                    0,
                    &readback,
                    0,
                    count as u64 * 16 * 4,
                );
                queue.submit(std::iter::once(encoder.finish()));
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .expect("specialized parity submit");
                let actual = map_read::<[f32; 4]>(&device, &readback);
                for cell in 0..count {
                    for channel in 0..16 {
                        let gpu = actual[cell * 4 + channel / 4][channel % 4];
                        let cpu = expected[cell][channel];
                        assert!(
                            (gpu - cpu).abs() <= 0.1,
                            "shape={shape:?} cell={cell} channel={channel} cpu={cpu} gpu={gpu}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn phase3_integrated_200k_gpu_gate() {
        pollster::block_on(async {
            use crate::simulation::signal_backbone_bench::{
                populate_workload, synthetic_forest, SignalWorkload, SyntheticShape,
            };
            let (device, queue) = timestamp_device().await;
            let count = 200_000u32;
            let mut forest = synthetic_forest(SyntheticShape::BalancedBinary, count as usize);
            populate_workload(&mut forest.sources, SignalWorkload::EveryCellEmits, 77);
            let cache = forest.cache().expect("200k cached forest");
            let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Phase 3 Gate Nutrients"),
                contents: bytemuck::cast_slice(&vec![100_000_000i32; count as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let public = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Phase 3 Gate Public"),
                size: count as u64 * 16 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let mut pipeline =
                SignalBackboneValuePipeline::new(&device, count, &nutrients, &public);
            pipeline.set_static_forest(&device, &cache);
            let mut source = vec![[0.0f32; 4]; count as usize * 4];
            let mut meta = vec![GpuSignalSourceMeta::default(); count as usize];
            for cell in 0..count as usize {
                for channel in 0..16 {
                    source[cell * 4 + channel / 4][channel % 4] = forest.sources[cell][channel];
                    meta[cell].requested_absolute += forest.sources[cell][channel].abs();
                }
                meta[cell].identity = cell as u32 + 1;
                meta[cell].flags = CELL_LIVE;
            }
            queue.write_buffer(&pipeline.source_field, 0, bytemuck::cast_slice(&source));
            queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&meta));
            pipeline.write_params(&queue, 0, count, 77, 0xf);
            let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Phase 3 Gate Queries"),
                ty: wgpu::QueryType::Timestamp,
                count: 4,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Phase 3 Gate Query Resolve"),
                size: 256,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Phase 3 Gate Query Readback"),
                size: 32,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let encode = |encoder: &mut wgpu::CommandEncoder, timed: bool| {
                if timed {
                    encoder.write_timestamp(&queries, 0);
                }
                pipeline.encode_sources(encoder, 0, count);
                if timed {
                    encoder.write_timestamp(&queries, 1);
                }
                pipeline.encode_propagation(encoder, 0xf, 0);
                if timed {
                    encoder.write_timestamp(&queries, 2);
                }
                pipeline.encode_finalize_and_processors(encoder, 0, count, false);
                if timed {
                    encoder.write_timestamp(&queries, 3);
                }
            };
            for _ in 0..8 {
                let mut encoder = device.create_command_encoder(&Default::default());
                encode(&mut encoder, false);
                queue.submit(std::iter::once(encoder.finish()));
            }
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("warmup");
            let period = queue.get_timestamp_period() as f64;
            let mut stages = [Vec::<f64>::new(), Vec::new(), Vec::new(), Vec::new()];
            for _ in 0..30 {
                let mut encoder = device.create_command_encoder(&Default::default());
                encode(&mut encoder, true);
                encoder.resolve_query_set(&queries, 0..4, &resolve, 0);
                encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 32);
                queue.submit(std::iter::once(encoder.finish()));
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .expect("sample");
                let t = map_read::<u64>(&device, &readback);
                stages[0].push((t[1] - t[0]) as f64 * period / 1e6);
                stages[1].push((t[2] - t[1]) as f64 * period / 1e6);
                stages[2].push((t[3] - t[2]) as f64 * period / 1e6);
                stages[3].push((t[3] - t[0]) as f64 * period / 1e6);
            }
            for stage in &mut stages {
                stage.sort_by(f64::total_cmp);
            }
            let p95 = |stage: usize| stages[stage][28];
            eprintln!("phase3_200k source_p95={:.4} propagation_p95={:.4} publication_p95={:.4} total_p95={:.4} memory_mib={:.3} dispatches={}",
                p95(0), p95(1), p95(2), p95(3), pipeline.allocated_bytes() as f64 / 1048576.0,
                pipeline.tick_dispatch_count(false));
            assert!(
                pipeline.allocated_bytes() <= 64 * 1024 * 1024,
                "signal memory gate"
            );
            assert!(p95(3) < 2.0, "discrete GPU p95 gate: {:.4} ms", p95(3));
        });
    }

    #[test]
    fn gpu_cognocyte_matrix_matches_cpu_and_has_one_tick_latency() {
        pollster::block_on(async {
            use crate::cell::behaviors::cognocyte::{evaluate, OP_COUNT};
            let (device, queue) = test_device().await;
            let count = OP_COUNT as u32;
            let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cognocyte Matrix Nutrients"),
                contents: bytemuck::cast_slice(&vec![1_000_000i32; count as usize]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let public = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cognocyte Matrix Public"),
                size: count as u64 * 16 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let pipeline = SignalBackboneValuePipeline::new(&device, count, &nutrients, &public);
            let mut field = vec![[0.0f32; 4]; count as usize * 4];
            let meta = vec![
                GpuSignalSourceMeta {
                    identity: 1,
                    flags: CELL_LIVE,
                    requested_absolute: 0.0,
                    _padding: 0
                };
                count as usize
            ];
            let configs = (0..count)
                .map(|operation| {
                    GpuSignalProcessorConfig::new(
                        PROCESSOR_COGNOCYTE,
                        operation,
                        0,
                        1,
                        operation % 16,
                        11,
                        2,
                        1.25,
                        0.1,
                        500.0,
                    )
                })
                .collect::<Vec<_>>();
            for cell in 0..count as usize {
                field[cell * 4] = [250.0, -400.0, 0.0, 0.0];
            }
            queue.write_buffer(&pipeline.finalized_field, 0, bytemuck::cast_slice(&field));
            queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&meta));
            queue.write_buffer(
                &pipeline.processor_config,
                0,
                bytemuck::cast_slice(&configs),
            );
            pipeline.write_params(&queue, 0, count, 7, 0xf);
            let state_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cognocyte Matrix State Readback"),
                size: count as u64 * std::mem::size_of::<GpuSignalProcessorState>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let source_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cognocyte Latency Source Readback"),
                size: count as u64 * 4 * 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cognocyte Matrix Encoder"),
            });
            pipeline.encode_finalize_and_processors(&mut encoder, 0, count, true);
            encoder.copy_buffer_to_buffer(
                &pipeline.processor_state,
                0,
                &state_staging,
                0,
                state_staging.size(),
            );
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("matrix submit");
            let states = map_read::<GpuSignalProcessorState>(&device, &state_staging);
            let time = 7.0 * SIGNAL_TICK_SECONDS;
            for operation in 0..OP_COUNT as usize {
                let expected = match operation {
                    14 => (((1.25 * time + 0.1) * std::f32::consts::TAU).sin()) * 500.0,
                    15 => ((1.25 * time + 0.1).rem_euclid(1.0) * 2.0 - 1.0) * 500.0,
                    12 | 16..=19 => evaluate(operation as i32, 250.0, -400.0),
                    _ => evaluate(operation as i32, 250.0, -400.0),
                }
                .clamp(-1000.0, 1000.0);
                assert!(
                    (states[operation].output - expected).abs() <= 0.01,
                    "operation={operation} cpu={expected} gpu={}",
                    states[operation].output
                );
            }

            // The freshly committed result is not part of the field used above;
            // it becomes a funded source only when the following tick starts.
            pipeline.write_params(&queue, 1, count, 8, 0xf);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cognocyte Latency Encoder"),
            });
            pipeline.encode_sources(&mut encoder, 1, count);
            encoder.copy_buffer_to_buffer(
                &pipeline.evaluated_source,
                0,
                &source_staging,
                0,
                source_staging.size(),
            );
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("latency submit");
            let sources = map_read::<[f32; 4]>(&device, &source_staging);
            for cell in 0..count as usize {
                let channel = configs[cell].output_channel() as usize;
                let emitted = sources[cell * 4 + channel / 4][channel % 4];
                assert!(
                    (emitted - states[cell].output).abs() <= 0.01,
                    "latency cell={cell}"
                );
            }
        });
    }

    #[test]
    fn gpu_funding_heat_and_processor_lifecycle_are_exact() {
        pollster::block_on(async {
            let (device, queue) = test_device().await;
            let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Lifecycle Nutrients"),
                contents: bytemuck::cast_slice(&[20i32, 0, 1_000_000, 1_000_000]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let public = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Lifecycle Public"),
                size: 4 * 16 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let pipeline = SignalBackboneValuePipeline::new(&device, 4, &nutrients, &public);
            let mut requested = vec![[0.0f32; 4]; 16];
            requested[0] = [1000.0, -500.0, 0.0, 0.0];
            requested[4] = [250.0, 0.0, 0.0, 0.0];
            let meta = [
                GpuSignalSourceMeta {
                    identity: 10,
                    flags: CELL_LIVE,
                    requested_absolute: 1500.0,
                    _padding: 0,
                },
                GpuSignalSourceMeta {
                    identity: 20,
                    flags: CELL_LIVE | CELL_CRITICAL_HEAT,
                    requested_absolute: 250.0,
                    _padding: 0,
                },
                GpuSignalSourceMeta {
                    identity: 30,
                    flags: CELL_LIVE,
                    requested_absolute: 0.0,
                    _padding: 0,
                },
                GpuSignalSourceMeta {
                    identity: 40,
                    flags: 0,
                    requested_absolute: 0.0,
                    _padding: 0,
                },
            ];
            let configs = [
                GpuSignalProcessorConfig::new(
                    PROCESSOR_MEMOROCYTE,
                    0,
                    0,
                    0,
                    4,
                    1,
                    0,
                    0.5,
                    0.0,
                    0.0,
                ),
                GpuSignalProcessorConfig::default(),
                GpuSignalProcessorConfig::new(
                    PROCESSOR_MEMOROCYTE,
                    0,
                    0,
                    0,
                    5,
                    2,
                    0,
                    0.5,
                    0.0,
                    0.0,
                ),
                GpuSignalProcessorConfig::new(
                    PROCESSOR_MEMOROCYTE,
                    0,
                    0,
                    0,
                    6,
                    1,
                    0,
                    0.5,
                    0.0,
                    0.0,
                ),
            ];
            let old_states = [
                GpuSignalProcessorState {
                    memory: 200.0,
                    output: 0.0,
                    output_channel: 4,
                    generation: 1,
                },
                GpuSignalProcessorState {
                    memory: 777.0,
                    output: 777.0,
                    output_channel: 5,
                    generation: 1,
                },
                GpuSignalProcessorState {
                    memory: 333.0,
                    output: 333.0,
                    output_channel: 5,
                    generation: 1,
                },
                GpuSignalProcessorState {
                    memory: 999.0,
                    output: 999.0,
                    output_channel: 6,
                    generation: 1,
                },
            ];
            let mut processor_field = vec![[0.0f32; 4]; 16];
            processor_field[0][0] = 1000.0;
            processor_field[8][0] = -1000.0;
            queue.write_buffer(&pipeline.source_field, 0, bytemuck::cast_slice(&requested));
            queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&meta));
            queue.write_buffer(
                &pipeline.processor_config,
                0,
                bytemuck::cast_slice(&configs),
            );
            queue.write_buffer(
                &pipeline.processor_state,
                0,
                bytemuck::cast_slice(&old_states),
            );
            pipeline.write_params(&queue, 0, 4, 23, 0xf);

            let source_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Funding Readback"),
                size: 16 * 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let nutrient_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Nutrient Readback"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let state_staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Lifecycle State Readback"),
                size: 64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Signal Lifecycle Encoder"),
            });
            pipeline.encode_sources(&mut encoder, 0, 4);
            encoder.copy_buffer_to_buffer(
                &pipeline.evaluated_source,
                0,
                &source_staging,
                0,
                source_staging.size(),
            );
            encoder.copy_buffer_to_buffer(&nutrients, 0, &nutrient_staging, 0, 16);
            // Supply processor inputs after source evaluation, mirroring the topology solve.
            encoder.copy_buffer_to_buffer(
                &device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Signal Lifecycle Processor Inputs"),
                    contents: bytemuck::cast_slice(&processor_field),
                    usage: wgpu::BufferUsages::COPY_SRC,
                }),
                0,
                &pipeline.finalized_field,
                0,
                16 * 16,
            );
            pipeline.encode_finalize_and_processors(&mut encoder, 0, 4, true);
            encoder.copy_buffer_to_buffer(&pipeline.processor_state, 0, &state_staging, 0, 64);
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("lifecycle submit");

            let sources = map_read::<[f32; 4]>(&device, &source_staging);
            assert!((sources[0][0] - 800.0).abs() <= 0.01);
            assert!((sources[0][1] + 400.0).abs() <= 0.01);
            for channel in 0..16usize {
                let expected =
                    crate::simulation::signal_system::deterministic_heat_value(20, channel, 23);
                let actual = sources[4 + channel / 4][channel % 4];
                assert_eq!(actual, expected, "heat channel={channel}");
            }
            // Cell 2 still pays for its previous-tick output before the
            // generation change resets processor memory for the next output.
            assert_eq!(
                map_read::<i32>(&device, &nutrient_staging),
                vec![0, 0, 999_994, 1_000_000]
            );

            let states = map_read::<GpuSignalProcessorState>(&device, &state_staging);
            let effective = 1.0 - (1.0f32 - 0.5).powf(SIGNAL_TICK_SECONDS);
            assert!((states[0].memory - (200.0 + 800.0 * effective)).abs() <= 0.01);
            assert_eq!(states[1].output, 0.0, "processor removal resets state");
            assert!(
                (states[2].memory - (-1000.0 * effective)).abs() <= 0.01,
                "generation change must reset memory before evaluation"
            );
            assert_eq!(states[3].output, 0.0, "death resets state");
        });
    }
}
