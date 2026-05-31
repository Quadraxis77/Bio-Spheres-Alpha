//! Organism Label System
//!
//! Assigns each live cell a `u32` label equal to the minimum cell index in its
//! connected component (connected via active adhesion bonds).  The system is
//! entirely GPU-driven.
//!
//! ## Stable IDs
//!
//! Raw labels are volatile — they change when the minimum-index cell dies.
//! `stable_id_per_cell_buffer` maps each cell to a stable sequential ID in
//! [1, 512] that persists across label changes.  The shrinkwrap skin system
//! uses this buffer instead of raw labels.
//!
//! ## Reading stable IDs
//!
//! ```wgsl
//! let stable_id = stable_id_per_cell[cell_i];  // 0 = no skin, 1-512 = organism slot
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{AdhesionBuffers, GpuTripleBufferSystem};

// ── GPU-side state ────────────────────────────────────────────────────────────

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

#[allow(dead_code)]
const MAX_STABLE_ID: u32 = 512;

// ── Public system ─────────────────────────────────────────────────────────────

pub struct OrganismLabelSystem {
    /// Per-cell label buffer (volatile root label).
    pub label_buffer: wgpu::Buffer,

    /// Per-organism cell count, indexed by root cell index.
    pub organism_size_buffer: wgpu::Buffer,

    /// Per-cell stable organism ID in [1, 512].  0 = no skin / too small.
    /// This is what the shrinkwrap system should use — it doesn't change when
    /// the root label changes due to cell death.
    pub stable_id_per_cell_buffer: wgpu::Buffer,

    #[allow(dead_code)]
    label_state_buffer: wgpu::Buffer,

    // Label pipelines
    controller_pipeline:     wgpu::ComputePipeline,
    init_pipeline:           wgpu::ComputePipeline,
    hook_pipeline:           wgpu::ComputePipeline,
    compress_pipeline:       wgpu::ComputePipeline,
    clear_sizes_pipeline:        wgpu::ComputePipeline,
    count_sizes_accumulate_pipeline: wgpu::ComputePipeline,
    count_sizes_broadcast_pipeline:  wgpu::ComputePipeline,
    label_bind_group: wgpu::BindGroup,

    // Stable ID pipelines
    assign_stable_ids_pipeline:    wgpu::ComputePipeline,
    broadcast_stable_ids_pipeline: wgpu::ComputePipeline,
    stable_id_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    stable_id_map_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    stable_id_counter_buffer: wgpu::Buffer,

    /// Fixed workgroup count = ceil(cell_capacity / 256).
    /// Kept for reference but no longer used in encode_frame (which now computes
    /// workgroups dynamically from the live cell slot count).
    #[allow(dead_code)]
    cell_workgroups: u32,

    debug_staging: wgpu::Buffer,
    debug_copy_pending: bool,
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

        // ── Label buffers ─────────────────────────────────────────────────────

        let initial_labels: Vec<u32> = vec![0xFFFFFFFFu32; cell_capacity as usize];
        let label_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Label Buffer"),
            contents: bytemuck::cast_slice(&initial_labels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let organism_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Size Buffer"),
            contents: bytemuck::cast_slice(&vec![0u32; cell_capacity as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let initial_state = LabelState { _pad0:0,_pad1:0,_pad2:0,_pad3:0, run_init:0, run_hc:0, _pad4:0,_pad5:0 };
        let label_state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Organism Label State"),
            contents: bytemuck::bytes_of(&initial_state),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Stable ID buffers ─────────────────────────────────────────────────

        // stable_id_map: one u32 per cell slot, indexed by root label.
        // Stores the stable ID assigned to each organism root.
        let stable_id_map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stable ID Map"),
            contents: bytemuck::cast_slice(&vec![0u32; cell_capacity as usize]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // stable_id_counter: single atomic u32, starts at 0 (IDs assigned as counter+1).
        let stable_id_counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stable ID Counter"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // stable_id_per_cell: output — one u32 per cell, the stable ID for that cell.
        let stable_id_per_cell_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stable ID Per Cell"),
            contents: bytemuck::cast_slice(&vec![0u32; cell_capacity as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // ── Label bind group layout & pipelines ───────────────────────────────

        let label_layout = Self::create_label_bind_group_layout(device);
        let label_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Organism Label Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/organism_label.wgsl").into(),
            ),
        });
        let label_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Organism Label Pipeline Layout"),
            bind_group_layouts:   &[&label_layout],
            push_constant_ranges: &[],
        });

        let make_label = |entry: &str, lbl: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some(lbl),
                layout:              Some(&label_pipeline_layout),
                module:              &label_shader,
                entry_point:         Some(entry),
                compilation_options: Default::default(),
                cache:               None,
            })
        };

        let controller_pipeline = make_label("label_controller", "Label Controller");
        let init_pipeline       = make_label("init_labels",       "Label Init");
        let hook_pipeline       = make_label("hook_labels",       "Label Hook");
        let compress_pipeline   = make_label("compress_labels",   "Label Compress");
        let clear_sizes_pipeline                = make_label("clear_organism_sizes",             "Organism Size Clear");
        let count_sizes_accumulate_pipeline     = make_label("count_organism_sizes_accumulate",  "Organism Size Accumulate");
        let count_sizes_broadcast_pipeline      = make_label("count_organism_sizes_broadcast",   "Organism Size Broadcast");

        let label_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("Organism Label BG"),
            layout: &label_layout,
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

        // ── Stable ID bind group layout & pipelines ───────────────────────────

        let stable_layout = Self::create_stable_id_bind_group_layout(device);
        let stable_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Organism Stable ID Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/organism_stable_id.wgsl").into(),
            ),
        });
        let stable_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Organism Stable ID Pipeline Layout"),
            bind_group_layouts:   &[&stable_layout],
            push_constant_ranges: &[],
        });

        let make_stable = |entry: &str, lbl: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some(lbl),
                layout:              Some(&stable_pipeline_layout),
                module:              &stable_shader,
                entry_point:         Some(entry),
                compilation_options: Default::default(),
                cache:               None,
            })
        };

        let assign_stable_ids_pipeline    = make_stable("assign_stable_ids",    "Stable ID Assign");
        let broadcast_stable_ids_pipeline = make_stable("broadcast_stable_ids", "Stable ID Broadcast");

        let stable_id_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("Stable ID BG"),
            layout: &stable_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: label_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: organism_size_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: stable_id_map_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: stable_id_counter_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: stable_id_per_cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: triple_buffers.cell_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: triple_buffers.death_flags.as_entire_binding() },
            ],
        });

        // Debug staging
        let debug_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Label Debug Staging"),
            size: 32 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            label_buffer,
            organism_size_buffer,
            stable_id_per_cell_buffer,
            label_state_buffer,
            controller_pipeline,
            init_pipeline,
            hook_pipeline,
            compress_pipeline,
            clear_sizes_pipeline,
            count_sizes_accumulate_pipeline,
            count_sizes_broadcast_pipeline,
            label_bind_group,
            assign_stable_ids_pipeline,
            broadcast_stable_ids_pipeline,
            stable_id_bind_group,
            stable_id_map_buffer,
            stable_id_counter_buffer,
            cell_workgroups,
            debug_staging,
            debug_copy_pending: false,
            debug_frame: 0,
        }
    }

    /// Encode the full label + stable ID pipeline.  Call once per render frame.
    ///
    /// `cell_slots_used` is the high-water mark of allocated cell slots (same value
    /// passed as `_cell_count_hint` to the physics pipeline).  Dispatches are sized
    /// to cover only the slots that have ever been used, not the full buffer capacity.
    /// Passing 0 skips all dispatches entirely.
    ///
    /// ## Throttling
    ///
    /// Organism topology changes slowly (cell division takes seconds, death is
    /// infrequent).  Running the full union-find every frame is wasteful.  This
    /// method runs the pipeline every `LABEL_PERIOD` calls and skips the rest.
    ///
    /// Consequences of a stale label:
    /// - **Collision**: a freshly-divided cell pair may collide for up to
    ///   `LABEL_PERIOD` frames.  They are adjacent, so the penetration is tiny
    ///   and the resulting force is negligible.
    /// - **Kleiber discount**: organism size is slightly stale.  The discount
    ///   changes by at most 1 cell per update, which is imperceptible.
    /// - **Shrinkwrap skin**: stable IDs update at the same rate.  The skin
    ///   mesh lags by at most `LABEL_PERIOD` frames — invisible at 60 fps.
    ///
    /// Set `LABEL_PERIOD = 1` to restore every-frame behaviour.
    pub fn encode_frame(&mut self, encoder: &mut wgpu::CommandEncoder, cell_slots_used: u32) {
        // Skip all work when there are no cells.
        if cell_slots_used == 0 {
            return;
        }

        // Throttle: only run the full pipeline every LABEL_PERIOD frames.
        // The label buffer retains its previous values between runs, so consumers
        // always have a valid (possibly 1–LABEL_PERIOD frames stale) result.
        const LABEL_PERIOD: u32 = 60;
        self.debug_frame = self.debug_frame.wrapping_add(1);
        let run_this_frame = (self.debug_frame % LABEL_PERIOD) == 0;
        if !run_this_frame {
            return;
        }

        // Dispatch only over the slots that have been allocated, not the full capacity.
        // Dead cells can sit at any index so we must cover the high-water mark, but we
        // don't need to touch the empty tail of the buffer.
        let active_workgroups = (cell_slots_used + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("Organism Label Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.label_bind_group, &[]);

        // Controller: always sets run_init = run_hc = 1 (recompute every frame).
        pass.set_pipeline(&self.controller_pipeline);
        pass.dispatch_workgroups(1, 1, 1);

        // Init: label[i] = i for live cells, DEAD_LABEL for dead.
        pass.set_pipeline(&self.init_pipeline);
        pass.dispatch_workgroups(active_workgroups, 1, 1);

        // Hook+compress convergence.
        // Each hook pass propagates the minimum label one adhesion hop further.
        // A chain of N cells needs ~N hook passes to converge end-to-end.
        //
        // 8 hook passes handles organisms up to ~8 cells in diameter (ring-based
        // creatures from the procedural generator are compact, not long chains).
        // 6 compress passes flatten the union-find tree after hooking.
        //
        // Cost vs old (20 hook + 10 compress, every frame, full capacity):
        //   Old: 30 dispatches × ceil(capacity/256) workgroups, every frame
        //   New: 14 dispatches × ceil(slots_used/256) workgroups, every 4 frames
        //   → ~85% reduction at 10K cells / 200K capacity
        //
        // If you have organisms with very long linear chains (>8 cells end-to-end),
        // increase HOOK_ITERS. Each extra hook pass costs active_workgroups workgroups.
        const HOOK_ITERS:     u32 = 8;
        const COMPRESS_ITERS: u32 = 6;

        for _ in 0..HOOK_ITERS {
            pass.set_pipeline(&self.hook_pipeline);
            pass.dispatch_workgroups(active_workgroups, 1, 1);
        }
        for _ in 0..COMPRESS_ITERS {
            pass.set_pipeline(&self.compress_pipeline);
            pass.dispatch_workgroups(active_workgroups, 1, 1);
        }

        // Recount organism sizes from converged labels.
        pass.set_pipeline(&self.clear_sizes_pipeline);
        pass.dispatch_workgroups(active_workgroups, 1, 1);
        pass.set_pipeline(&self.count_sizes_accumulate_pipeline);
        pass.dispatch_workgroups(active_workgroups, 1, 1);
        pass.set_pipeline(&self.count_sizes_broadcast_pipeline);
        pass.dispatch_workgroups(active_workgroups, 1, 1);

        drop(pass);

        // Stable ID passes run in a separate compute pass (need label+size results visible).
        {
            let mut stable_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("Organism Stable ID Pass"),
                timestamp_writes: None,
            });
            stable_pass.set_bind_group(0, &self.stable_id_bind_group, &[]);

            stable_pass.set_pipeline(&self.assign_stable_ids_pipeline);
            stable_pass.dispatch_workgroups(active_workgroups, 1, 1);

            stable_pass.set_pipeline(&self.broadcast_stable_ids_pipeline);
            stable_pass.dispatch_workgroups(active_workgroups, 1, 1);
        }

        // Debug readback every 120 label updates (= 480 frames at LABEL_PERIOD=4).
        if self.debug_frame % (120 * LABEL_PERIOD) == 0 {
            let copy_size = (32 * 4).min(self.label_buffer.size());
            encoder.copy_buffer_to_buffer(&self.label_buffer, 0, &self.debug_staging, 0, copy_size);
            self.debug_copy_pending = true;
        }
    }

    pub fn poll_debug_readback(&mut self, device: &wgpu::Device) {
        if !self.debug_copy_pending { return; }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let slice = self.debug_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        if let Ok(Ok(())) = rx.try_recv() {
            let view = slice.get_mapped_range();
            drop(view);
            self.debug_staging.unmap();
            self.debug_copy_pending = false;
        }
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn create_label_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Organism Label BGL"),
            entries: &[
                rw(0), // label_state
                ro(1), // cell_count_buffer
                ro(2), // adhesion_counts
                rw(3), // label_buffer
                ro(4), // death_flags
                ro(5), // cell_adhesion_indices
                ro(6), // adhesion_connections
                rw(7), // organism_size_buffer
            ],
        })
    }

    fn create_stable_id_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Stable ID BGL"),
            entries: &[
                ro(0), // label_buffer
                ro(1), // organism_size_buffer
                rw(2), // stable_id_map
                rw(3), // stable_id_counter
                rw(4), // stable_id_per_cell
                ro(5), // cell_count_buf
                ro(6), // death_flags
            ],
        })
    }
}
