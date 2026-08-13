use std::sync::mpsc;

use bio_spheres::simulation::gpu_physics::signal_backbone_topology::{
    GpuBackboneTopologyBond, GpuBackboneTopologyControl, GpuBackboneTopologyJob,
    SignalBackboneTopologyPipeline, TOPOLOGY_BOND_ACTIVE, TOPOLOGY_BOND_BACKBONE,
    TOPOLOGY_BOND_PENDING, TOPOLOGY_BOND_VALID, TOPOLOGY_JOB_ADD, TOPOLOGY_JOB_REPAIR,
    TOPOLOGY_MAX_ADHESIONS_PER_CELL,
};
use bytemuck::Pod;

const NORMAL: u32 = 51_293;
const VASCULAR: u32 = 12_579;

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}

fn adjacency(nodes: usize, bonds: &[GpuBackboneTopologyBond]) -> Vec<u32> {
    let mut result = vec![u32::MAX; nodes * TOPOLOGY_MAX_ADHESIONS_PER_CELL];
    let mut counts = vec![0usize; nodes];
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

fn bond(index: usize, a: u32, b: u32, resistance: u32, flags: u32) -> GpuBackboneTopologyBond {
    GpuBackboneTopologyBond {
        stable_lo: index as u32,
        stable_hi: 1,
        a,
        b,
        resistance,
        flags,
        generation: 1,
        _padding: 0,
    }
}

fn map_read<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
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

async fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    nodes: u32,
    case: &str,
    samples: usize,
) {
    let operation_budget = 1024u32;
    let mut bonds = Vec::with_capacity(nodes as usize + 8);
    let jobs;
    if case == "leaf" {
        for node in 1..nodes - 1 {
            bonds.push(bond(
                bonds.len(),
                node - 1,
                node,
                NORMAL,
                TOPOLOGY_BOND_VALID | TOPOLOGY_BOND_BACKBONE | TOPOLOGY_BOND_ACTIVE,
            ));
        }
        let pending = bonds.len();
        bonds.push(bond(
            pending,
            nodes - 2,
            nodes - 1,
            NORMAL,
            TOPOLOGY_BOND_VALID | TOPOLOGY_BOND_BACKBONE | TOPOLOGY_BOND_PENDING,
        ));
        jobs = vec![GpuBackboneTopologyJob {
            kind: TOPOLOGY_JOB_ADD,
            bond_index: pending as u32,
            cut_a: u32::MAX,
            cut_b: u32::MAX,
        }];
    } else {
        let cut = nodes / 2 - 1;
        for node in 1..nodes {
            let mut flags = TOPOLOGY_BOND_VALID | TOPOLOGY_BOND_BACKBONE | TOPOLOGY_BOND_ACTIVE;
            if node - 1 == cut {
                flags = TOPOLOGY_BOND_BACKBONE;
            }
            bonds.push(bond(bonds.len(), node - 1, node, NORMAL, flags));
        }
        for offset in 1..=4u32 {
            bonds.push(bond(
                bonds.len(),
                cut.saturating_sub(offset),
                (cut + 1 + offset).min(nodes - 1),
                if offset == 4 { VASCULAR } else { NORMAL },
                TOPOLOGY_BOND_VALID | TOPOLOGY_BOND_BACKBONE,
            ));
        }
        jobs = vec![GpuBackboneTopologyJob {
            kind: TOPOLOGY_JOB_REPAIR,
            bond_index: cut,
            cut_a: cut,
            cut_b: cut + 1,
        }];
    }
    let topology =
        SignalBackboneTopologyPipeline::new(device, nodes, bonds.len() as u32, jobs.len() as u32);
    topology.upload(
        queue,
        nodes,
        &bonds,
        &adjacency(nodes as usize, &bonds),
        &jobs,
    );
    let query = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Phase 4 Topology Benchmark Query"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Phase 4 Topology Benchmark Resolve"),
        size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Phase 4 Topology Benchmark Readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let original_bonds = bytemuck::cast_slice(&bonds);
    let control = GpuBackboneTopologyControl::default();
    let period = queue.get_timestamp_period() as f64 / 1_000_000.0;
    let mut timings = Vec::with_capacity(samples);
    for sample in 0..samples + 3 {
        queue.write_buffer(&topology.bonds, 0, original_bonds);
        queue.write_buffer(&topology.control, 0, bytemuck::bytes_of(&control));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Phase 4 Topology Benchmark Encoder"),
        });
        encoder.write_timestamp(&query, 0);
        topology.encode(
            queue,
            &mut encoder,
            nodes,
            bonds.len() as u32,
            1,
            operation_budget,
        );
        encoder.write_timestamp(&query, 1);
        encoder.resolve_query_set(&query, 0..2, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("benchmark wait");
        let ticks = map_read::<u64>(device, &readback);
        if sample >= 3 {
            timings.push(ticks[1].saturating_sub(ticks[0]) as f64 * period);
        }
    }
    let min = timings.iter().copied().fold(f64::INFINITY, f64::min);
    let median = percentile(&mut timings.clone(), 0.5);
    let p95 = percentile(&mut timings, 0.95);
    let completion_frames = if case == "leaf" {
        1
    } else {
        (nodes * 2 + bonds.len() as u32 + operation_budget - 1) / operation_budget + 4
    };
    queue.write_buffer(&topology.bonds, 0, original_bonds);
    queue.write_buffer(&topology.control, 0, bytemuck::bytes_of(&control));
    let mut completion_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Phase 4 Completion Verification Encoder"),
    });
    for _ in 0..completion_frames {
        topology.encode(
            queue,
            &mut completion_encoder,
            nodes,
            bonds.len() as u32,
            1,
            operation_budget,
        );
    }
    let control_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Phase 4 Completion Control Readback"),
        size: std::mem::size_of::<GpuBackboneTopologyControl>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    completion_encoder.copy_buffer_to_buffer(
        &topology.control,
        0,
        &control_readback,
        0,
        control_readback.size(),
    );
    let submission = queue.submit(std::iter::once(completion_encoder.finish()));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("completion wait");
    let completed = map_read::<GpuBackboneTopologyControl>(device, &control_readback)[0];
    assert_eq!(completed.cursor, 1, "bounded repair must eventually commit");
    assert_eq!(completed.processed, 1);
    assert_eq!(completed.invalid_jobs, 0);
    println!(
        "cells={nodes} case={case} work_budget={operation_budget} min_ms={min:.6} median_ms={median:.6} p95_ms={p95:.6} allocated_mib={:.3} dispatches_per_frame=1 completion_frames={completion_frames} eventual_commit=pass",
        topology.allocated_bytes() as f64 / (1024.0 * 1024.0)
    );
}

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Phase 4 benchmark GPU");
        let info = adapter.get_info();
        let features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        assert!(adapter.features().contains(features));
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Phase 4 Topology Benchmark Device"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("Phase 4 benchmark device");
        println!(
            "adapter={} backend={:?} device_type={:?} driver={} driver_info={}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        for nodes in [20_000, 100_000, 200_000] {
            run_case(&device, &queue, nodes, "leaf", 20).await;
            run_case(&device, &queue, nodes, "central_repair", 20).await;
        }
    });
}
