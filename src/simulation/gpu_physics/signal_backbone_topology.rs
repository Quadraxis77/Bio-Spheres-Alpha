//! Bounded, deterministic Phase 4 active/standby topology lifecycle.
//!
//! The GPU consumes stable-identity-sorted jobs. Physical lifecycle shaders
//! invalidate active bonds immediately; this pass commits additions and
//! standby failover independently from signal-value evaluation.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const TOPOLOGY_BOND_VALID: u32 = 1;
pub const TOPOLOGY_BOND_BACKBONE: u32 = 2;
pub const TOPOLOGY_BOND_ACTIVE: u32 = 4;
pub const TOPOLOGY_BOND_PENDING: u32 = 8;
pub const TOPOLOGY_JOB_ADD: u32 = 0;
pub const TOPOLOGY_JOB_REPAIR: u32 = 1;
pub const TOPOLOGY_MAX_ADHESIONS_PER_CELL: usize = 20;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, Eq, PartialEq)]
pub struct GpuBackboneTopologyBond {
    pub stable_lo: u32,
    pub stable_hi: u32,
    pub a: u32,
    pub b: u32,
    pub resistance: u32,
    pub flags: u32,
    pub generation: u32,
    pub _padding: u32,
}

impl GpuBackboneTopologyBond {
    pub fn stable_id(self) -> u64 {
        (u64::from(self.stable_hi) << 32) | u64::from(self.stable_lo)
    }

    pub fn active(self) -> bool {
        self.flags & TOPOLOGY_BOND_ACTIVE != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, Eq, PartialEq)]
pub struct GpuBackboneTopologyJob {
    pub kind: u32,
    pub bond_index: u32,
    pub cut_a: u32,
    pub cut_b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Eq, PartialEq)]
pub struct GpuBackboneTopologyControl {
    pub cursor: u32,
    pub topology_generation: u32,
    pub processed: u32,
    pub invalid_jobs: u32,
    pub phase: u32,
    pub head: u32,
    pub tail: u32,
    pub scan: u32,
    pub selected: u32,
    pub best_lo: u32,
    pub best_hi: u32,
    pub stamp: u32,
}

impl Default for GpuBackboneTopologyControl {
    fn default() -> Self {
        Self {
            cursor: 0,
            topology_generation: 1,
            processed: 0,
            invalid_jobs: 0,
            phase: 0,
            head: 0,
            tail: 0,
            scan: 0,
            selected: u32::MAX,
            best_lo: u32::MAX,
            best_hi: u32::MAX,
            stamp: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuBackboneTopologyParams {
    node_count: u32,
    bond_count: u32,
    job_count: u32,
    job_budget: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuBackboneNodeWork {
    stamp_a: u32,
    stamp_b: u32,
    parent_node: u32,
    parent_bond: u32,
    distance_a_lo: u32,
    distance_a_hi: u32,
    distance_b_lo: u32,
    distance_b_hi: u32,
}

pub struct SignalBackboneTopologyPipeline {
    node_capacity: u32,
    bond_capacity: u32,
    job_capacity: u32,
    params: wgpu::Buffer,
    pub bonds: wgpu::Buffer,
    adjacency: wgpu::Buffer,
    jobs: wgpu::Buffer,
    pub control: wgpu::Buffer,
    _work: wgpu::Buffer,
    _queue: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    allocated_bytes: u64,
}

impl SignalBackboneTopologyPipeline {
    pub fn new(
        device: &wgpu::Device,
        node_capacity: u32,
        bond_capacity: u32,
        job_capacity: u32,
    ) -> Self {
        let node_capacity = node_capacity.max(1);
        let bond_capacity = bond_capacity.max(1);
        let job_capacity = job_capacity.max(1);
        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Backbone Topology Params"),
            size: std::mem::size_of::<GpuBackboneTopologyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bonds = storage(
            "Signal Backbone Topology Bonds",
            u64::from(bond_capacity) * std::mem::size_of::<GpuBackboneTopologyBond>() as u64,
        );
        let adjacency = storage(
            "Signal Backbone Topology Adjacency",
            u64::from(node_capacity * TOPOLOGY_MAX_ADHESIONS_PER_CELL as u32) * 4,
        );
        let jobs = storage(
            "Signal Backbone Topology Jobs",
            u64::from(job_capacity) * std::mem::size_of::<GpuBackboneTopologyJob>() as u64,
        );
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Signal Backbone Topology Control"),
            contents: bytemuck::bytes_of(&GpuBackboneTopologyControl::default()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let work = storage(
            "Signal Backbone Topology Work",
            u64::from(node_capacity) * std::mem::size_of::<GpuBackboneNodeWork>() as u64,
        );
        let queue = storage(
            "Signal Backbone Topology Queue",
            u64::from(node_capacity) * 4,
        );

        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(
                    std::mem::size_of::<GpuBackboneTopologyParams>() as u64
                ),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Signal Backbone Topology Layout"),
            entries: &[
                uniform,
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Signal Backbone Topology Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/signal_backbone_topology.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Signal Backbone Topology Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Signal Backbone Topology Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("process_jobs"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal Backbone Topology Bind Group"),
            layout: &layout,
            entries: &[
                topology_entry(0, &params),
                topology_entry(1, &bonds),
                topology_entry(2, &adjacency),
                topology_entry(3, &jobs),
                topology_entry(4, &control),
                topology_entry(5, &work),
                topology_entry(6, &queue),
            ],
        });
        let allocated_bytes = std::mem::size_of::<GpuBackboneTopologyParams>() as u64
            + u64::from(bond_capacity) * std::mem::size_of::<GpuBackboneTopologyBond>() as u64
            + u64::from(node_capacity * TOPOLOGY_MAX_ADHESIONS_PER_CELL as u32) * 4
            + u64::from(job_capacity) * std::mem::size_of::<GpuBackboneTopologyJob>() as u64
            + std::mem::size_of::<GpuBackboneTopologyControl>() as u64
            + u64::from(node_capacity) * std::mem::size_of::<GpuBackboneNodeWork>() as u64
            + u64::from(node_capacity) * 4;

        Self {
            node_capacity,
            bond_capacity,
            job_capacity,
            params,
            bonds,
            adjacency,
            jobs,
            control,
            _work: work,
            _queue: queue,
            pipeline,
            bind_group,
            allocated_bytes,
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        node_count: u32,
        bonds: &[GpuBackboneTopologyBond],
        adjacency: &[u32],
        jobs: &[GpuBackboneTopologyJob],
    ) {
        assert!(node_count <= self.node_capacity);
        assert!(bonds.len() <= self.bond_capacity as usize);
        assert!(jobs.len() <= self.job_capacity as usize);
        assert_eq!(
            adjacency.len(),
            node_count as usize * TOPOLOGY_MAX_ADHESIONS_PER_CELL
        );
        if !bonds.is_empty() {
            queue.write_buffer(&self.bonds, 0, bytemuck::cast_slice(bonds));
        }
        if !adjacency.is_empty() {
            queue.write_buffer(&self.adjacency, 0, bytemuck::cast_slice(adjacency));
        }
        if !jobs.is_empty() {
            queue.write_buffer(&self.jobs, 0, bytemuck::cast_slice(jobs));
        }
        queue.write_buffer(
            &self.control,
            0,
            bytemuck::bytes_of(&GpuBackboneTopologyControl::default()),
        );
    }

    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        node_count: u32,
        bond_count: u32,
        job_count: u32,
        job_budget: u32,
    ) {
        assert!(node_count <= self.node_capacity);
        assert!(bond_count <= self.bond_capacity);
        assert!(job_count <= self.job_capacity);
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&GpuBackboneTopologyParams {
                node_count,
                bond_count,
                job_count,
                job_budget,
            }),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Backbone Topology Repair"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

fn topology_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = assert!(std::mem::size_of::<GpuBackboneTopologyBond>() == 32);
const _: () = assert!(std::mem::size_of::<GpuBackboneTopologyJob>() == 16);
const _: () = assert!(std::mem::size_of::<GpuBackboneTopologyControl>() == 48);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::signal_backbone_bench::BackboneLifecycleOracle;

    fn adjacency(node_count: usize, bonds: &[GpuBackboneTopologyBond]) -> Vec<u32> {
        let mut result = vec![u32::MAX; node_count * TOPOLOGY_MAX_ADHESIONS_PER_CELL];
        let mut counts = vec![0usize; node_count];
        for (index, bond) in bonds.iter().enumerate() {
            for node in [bond.a as usize, bond.b as usize] {
                let slot = counts[node];
                assert!(slot < TOPOLOGY_MAX_ADHESIONS_PER_CELL);
                result[node * TOPOLOGY_MAX_ADHESIONS_PER_CELL + slot] = index as u32;
                counts[node] += 1;
            }
        }
        result
    }

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

    async fn device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Phase 4 GPU adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Phase 4 Topology Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("Phase 4 test device")
    }

    #[test]
    fn phase4_gpu_addition_exchange_and_repair_match_cpu_oracle() {
        pollster::block_on(async {
            let (device, queue) = device().await;
            let definitions = [
                (10u64, 0u32, 1u32, 51_293u32),
                (20, 1, 2, 51_293),
                (30, 2, 3, 51_293),
                (40, 0, 3, 12_579),
                (50, 1, 3, 51_293),
            ];
            let mut oracle = BackboneLifecycleOracle::new(4);
            let mut bonds = Vec::new();
            for &(stable, a, b, resistance) in &definitions {
                oracle
                    .insert_backbone(stable, a, b, resistance)
                    .expect("oracle insert");
                bonds.push(GpuBackboneTopologyBond {
                    stable_lo: stable as u32,
                    stable_hi: (stable >> 32) as u32,
                    a,
                    b,
                    resistance,
                    flags: TOPOLOGY_BOND_VALID | TOPOLOGY_BOND_BACKBONE | TOPOLOGY_BOND_PENDING,
                    generation: 1,
                    _padding: 0,
                });
            }
            let jobs = (0..bonds.len())
                .map(|index| GpuBackboneTopologyJob {
                    kind: TOPOLOGY_JOB_ADD,
                    bond_index: index as u32,
                    cut_a: u32::MAX,
                    cut_b: u32::MAX,
                })
                .collect::<Vec<_>>();
            let topology = SignalBackboneTopologyPipeline::new(
                &device,
                4,
                bonds.len() as u32,
                jobs.len() as u32,
            );
            topology.upload(&queue, 4, &bonds, &adjacency(4, &bonds), &jobs);

            // Two bounded commits prove cursor persistence and job budgeting.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Phase 4 Addition Encoder"),
            });
            topology.encode(
                &queue,
                &mut encoder,
                4,
                bonds.len() as u32,
                jobs.len() as u32,
                2,
            );
            topology.encode(
                &queue,
                &mut encoder,
                4,
                bonds.len() as u32,
                jobs.len() as u32,
                3,
            );
            let bond_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Phase 4 Bond Readback"),
                size: bonds.len() as u64 * std::mem::size_of::<GpuBackboneTopologyBond>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let control_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Phase 4 Control Readback"),
                size: std::mem::size_of::<GpuBackboneTopologyControl>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                &topology.bonds,
                0,
                &bond_readback,
                0,
                bond_readback.size(),
            );
            encoder.copy_buffer_to_buffer(
                &topology.control,
                0,
                &control_readback,
                0,
                std::mem::size_of::<GpuBackboneTopologyControl>() as u64,
            );
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("Phase 4 additions");

            assert_eq!(oracle.commit(2), 2);
            assert_eq!(oracle.commit(3), 3);
            let gpu_bonds = map_read::<GpuBackboneTopologyBond>(&device, &bond_readback);
            for (index, gpu) in gpu_bonds.iter().enumerate() {
                assert_eq!(gpu.active(), oracle.bonds[index].active, "bond {index}");
                assert_eq!(gpu.flags & TOPOLOGY_BOND_PENDING, 0, "bond {index}");
            }
            let control = map_read::<GpuBackboneTopologyControl>(&device, &control_readback)[0];
            assert_eq!(control.cursor, jobs.len() as u32);
            assert_eq!(control.processed, jobs.len() as u32);
            assert_eq!(control.invalid_jobs, 0);

            // Break the selected shortcut immediately, then commit one repair.
            let broken = 3usize;
            assert!(oracle.bonds[broken].active);
            oracle.invalidate(definitions[broken].0);
            let mut repair_bonds = gpu_bonds;
            repair_bonds[broken].flags &= !(TOPOLOGY_BOND_VALID | TOPOLOGY_BOND_ACTIVE);
            let repair = [GpuBackboneTopologyJob {
                kind: TOPOLOGY_JOB_REPAIR,
                bond_index: broken as u32,
                cut_a: definitions[broken].1,
                cut_b: definitions[broken].2,
            }];
            let repair_pipeline =
                SignalBackboneTopologyPipeline::new(&device, 4, repair_bonds.len() as u32, 1);
            repair_pipeline.upload(
                &queue,
                4,
                &repair_bonds,
                &adjacency(4, &repair_bonds),
                &repair,
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Phase 4 Repair Encoder"),
            });
            for _ in 0..32 {
                repair_pipeline.encode(&queue, &mut encoder, 4, repair_bonds.len() as u32, 1, 1);
            }
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Phase 4 Repair Readback"),
                size: repair_bonds.len() as u64
                    * std::mem::size_of::<GpuBackboneTopologyBond>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&repair_pipeline.bonds, 0, &readback, 0, readback.size());
            queue.submit(std::iter::once(encoder.finish()));
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("Phase 4 repair");
            assert_eq!(oracle.commit(1), 1);
            for (index, gpu) in map_read::<GpuBackboneTopologyBond>(&device, &readback)
                .iter()
                .enumerate()
            {
                assert_eq!(
                    gpu.active(),
                    oracle.bonds[index].active,
                    "repair bond {index}"
                );
            }
        });
    }
}
