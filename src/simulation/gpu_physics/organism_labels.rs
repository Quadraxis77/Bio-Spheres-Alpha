//! Organism Label System
//!
//! Assigns each live cell a `u32` label equal to the minimum cell index in its
//! connected component (connected via active adhesion bonds).  The system is
//! entirely GPU-driven: a controller shader detects topology changes (cell count
//! or live-bond-count changes) and writes `run_init`/`run_hc` flags into
//! `label_state`; the subsequent dispatches in the same pass read those flags
//! and early-exit when both are 0.
//!
//! ## Why no indirect dispatch
//!
//! Indirect dispatch args are read by the GPU command processor before the prior
//! dispatch's storage writes are guaranteed visible. Writing flags into
//! `label_state` (a storage buffer) and reading them in the same compute pass
//! **is** correctly ordered by the WebGPU memory model, so the flag approach is
//! used instead.
//!
//! ## Reading labels
//!
//! Bind `label_buffer` (read-only, `array<u32>`) in any consumer shader:
//!
//! ```wgsl
//! let root     = label_buffer[cell_i];       // organism identity
//! let local_id = cell_i - root;              // unique within organism (sparse)
//! ```
//!
//! Dead cells have `label = 0xFFFFFFFF`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{AdhesionBuffers, GpuTripleBufferSystem};

// ── GPU-side state (must match WGSL LabelState exactly) ─────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LabelState {
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
    _pad3:    u32,
    run_init: u32,
    run_hc:   u32,
    _pad4:    u32,
    _pad5:    u32,
}

const _: () = assert!(std::mem::size_of::<LabelState>() == 32);

// ── Public system ─────────────────────────────────────────────────────────────

pub struct OrganismLabelSystem {
    /// Per-cell label buffer.  Consumers bind this read-only as `array<u32>`.
    pub label_buffer: wgpu::Buffer,

    /// Per-organism cell count, indexed by root cell index.
    /// `organism_size_buffer[label_buffer[cell_i]]` == number of cells in that organism.
    /// Updated every frame by the clear + count passes.
    pub organism_size_buffer: wgpu::Buffer,

    #[allow(dead_code)]
    label_state_buffer: wgpu::Buffer,

    controller_pipeline:     wgpu::ComputePipeline,
    init_pipeline:           wgpu::ComputePipeline,
    hook_pipeline:           wgpu::ComputePipeline,
    compress_pipeline:       wgpu::ComputePipeline,
    clear_sizes_pipeline:        wgpu::ComputePipeline,
    count_sizes_accumulate_pipeline: wgpu::ComputePipeline,
    count_sizes_broadcast_pipeline:  wgpu::ComputePipeline,

    bind_group: wgpu::BindGroup,

    /// Fixed workgroup count = ceil(cell_capacity / 256).
    cell_workgroups: u32,

    /// Staging buffer for debug readback of label values.
    debug_staging: wgpu::Buffer,
    /// True when a debug copy is pending in the submitted encoder.
    debug_copy_pending: bool,
    /// Debug frame counter (to throttle copies).
    debug_frame: u32,
}

impl OrganismLabelSystem {
    pub fn new(
        device: &wgpu::Device,
        triple_buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &AdhesionBuffers,
    ) -> Self {
        let cell_capacity = triple_buffers.capacity;
        let cell_workgroups = (cell_capacity + 255) / 256;

        // ── Buffers ──────────────────────────────────────────────────────────

        // Initialize label buffer with DEAD_LABEL values to prevent garbage data on first frame
        let initial_labels: Vec<u32> = vec![0xFFFFFFFFu32; cell_capacity as usize];
        let label_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Label Buffer"),
            contents: bytemuck::cast_slice(&initial_labels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Organism size buffer: one u32 per cell slot, indexed by root label.
        // Initialised to zero; cleared and recomputed every frame.
        let organism_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Size Buffer"),
            contents: bytemuck::cast_slice(&vec![0u32; cell_capacity as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let initial_state = LabelState {
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            run_init: 0,
            run_hc:   0,
            _pad4: 0,
            _pad5: 0,
        };
        let label_state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Organism Label State"),
            contents: bytemuck::bytes_of(&initial_state),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Bind group layout & pipelines ────────────────────────────────────

        let layout = Self::create_bind_group_layout(device);

        let shader_src = include_str!("../../../shaders/organism_label.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Organism Label Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Organism Label Pipeline Layout"),
            bind_group_layouts:   &[&layout],
            push_constant_ranges: &[],
        });

        let make = |entry: &str, lbl: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some(lbl),
                layout:              Some(&pipeline_layout),
                module:              &shader,
                entry_point:         Some(entry),
                compilation_options: Default::default(),
                cache:               None,
            })
        };

        let controller_pipeline = make("label_controller", "Label Controller");
        let init_pipeline       = make("init_labels",       "Label Init");
        let hook_pipeline       = make("hook_labels",       "Label Hook");
        let compress_pipeline   = make("compress_labels",   "Label Compress");
        let clear_sizes_pipeline = make("clear_organism_sizes", "Organism Size Clear");
        let count_sizes_accumulate_pipeline = make("count_organism_sizes_accumulate", "Organism Size Accumulate");
        let count_sizes_broadcast_pipeline  = make("count_organism_sizes_broadcast",  "Organism Size Broadcast");

        // ── Bind group ────────────────────────────────────────────────────────

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("Organism Label BG"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: label_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: triple_buffers.cell_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: adhesion_buffers.adhesion_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: label_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: triple_buffers.death_flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: adhesion_buffers.adhesion_connections.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: organism_size_buffer.as_entire_binding() },
            ],
        });

        // 32 u32 values for debug readback
        let debug_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Label Debug Staging"),
            size: 32 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            label_buffer,
            organism_size_buffer,
            label_state_buffer,
            controller_pipeline,
            init_pipeline,
            hook_pipeline,
            compress_pipeline,
            clear_sizes_pipeline,
            count_sizes_accumulate_pipeline,
            count_sizes_broadcast_pipeline,
            bind_group,
            cell_workgroups,
            debug_staging,
            debug_copy_pending: false,
            debug_frame: 0,
        }
    }

    /// Encode the label system into `encoder`.  Call once per render frame.
    ///
    /// All six dispatches share one compute pass.  The controller writes flags
    /// into `label_state` (storage); the subsequent dispatches read those flags.
    /// Storage writes are ordered between dispatches in the same pass per the
    /// WebGPU memory model, so no indirect dispatch or pass split is needed.
    pub fn encode_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("Organism Label Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);

        // Controller: updates state machine, writes run_init / run_hc.
        pass.set_pipeline(&self.controller_pipeline);
        pass.dispatch_workgroups(1, 1, 1);

        // Init: resets label[i] = i when run_init == 1.
        pass.set_pipeline(&self.init_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);

        // Hook + compress pair 1.
        pass.set_pipeline(&self.hook_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);
        pass.set_pipeline(&self.compress_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);

        // Hook + compress pair 2.
        pass.set_pipeline(&self.hook_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);
        pass.set_pipeline(&self.compress_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);

        // Clear organism sizes, then recount from current labels.
        // Runs every frame so sizes stay accurate even when labels don't change.
        pass.set_pipeline(&self.clear_sizes_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);
        // Accumulate: each live cell adds 1 to organism_size_buffer[root_label]
        pass.set_pipeline(&self.count_sizes_accumulate_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);
        // Broadcast: copy root count to every cell's own slot for O(1) consumer lookup
        pass.set_pipeline(&self.count_sizes_broadcast_pipeline);
        pass.dispatch_workgroups(self.cell_workgroups, 1, 1);

        drop(pass);

        // Debug: every 120 frames copy first 32 labels to staging for CPU readback.
        self.debug_frame += 1;
        if self.debug_frame % 120 == 0 {
            let copy_size = (32 * 4).min(self.label_buffer.size());
            encoder.copy_buffer_to_buffer(&self.label_buffer, 0, &self.debug_staging, 0, copy_size);
            self.debug_copy_pending = true;
        }
    }

    /// Call each frame after queue.submit() to read back debug labels.
    pub fn poll_debug_readback(&mut self, device: &wgpu::Device) {
        if !self.debug_copy_pending { return; }
        // Block until GPU finishes so the copy is visible.
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let slice = self.debug_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        if let Ok(Ok(())) = rx.try_recv() {
            let view = slice.get_mapped_range();
            // Debug output removed - organism labeling is working correctly
            drop(view);
            self.debug_staging.unmap();
            self.debug_copy_pending = false;
        }
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Organism Label BGL"),
            entries: &[
                rw(0), // label_state
                ro(1), // cell_count_buffer
                ro(2), // adhesion_counts
                rw(3), // label_buffer (atomic writes)
                ro(4), // death_flags
                ro(5), // cell_adhesion_indices
                ro(6), // adhesion_connections_raw
                rw(7), // organism_size_buffer (atomic writes)
            ],
        })
    }
}
