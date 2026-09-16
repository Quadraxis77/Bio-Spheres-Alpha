//! Exact persistence of GPU-born genomes. Runs only for explicit world save/load;
//! simulation and reproduction never wait for these readbacks.
use super::{
    gpu_fusion::genome_mode_buffers, AdhesionBuffers, GpuTripleBufferSystem, MutationSystem,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedGpuGenome {
    pub id: u32,
    pub metadata: [u32; 4],
    pub initial_orientation: [f32; 4],
    pub modes: BTreeMap<String, Vec<u32>>,
}

fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
) -> Result<Vec<u32>, String> {
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    Ok(words)
}
fn staging(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU Genome Snapshot"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    b: &GpuTripleBufferSystem,
    a: &AdhesionBuffers,
    m: &MutationSystem,
    colors: &wgpu::Buffer,
    emissive: &wgpu::Buffer,
    ids: impl IntoIterator<Item = u32>,
    authored_count: usize,
) -> Result<Vec<SavedGpuGenome>, String> {
    let ids: BTreeSet<_> = ids
        .into_iter()
        .filter(|id| *id as usize >= authored_count)
        .collect();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    if ids
        .iter()
        .any(|id| *id >= super::mutation::GENOME_RING_CAPACITY)
    {
        return Err("Invalid GPU genome ID in snapshot".into());
    }
    let header = staging(device, ids.len() as u64 * 32);
    let mut encoder = device.create_command_encoder(&Default::default());
    for (i, id) in ids.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            m.genome_meta_buffer(),
            *id as u64 * 16,
            &header,
            i as u64 * 32,
            16,
        );
        encoder.copy_buffer_to_buffer(
            &m.genome_initial_orientations,
            *id as u64 * 16,
            &header,
            i as u64 * 32 + 16,
            16,
        );
    }
    let header = read_words(device, queue, encoder, &header)?;
    let mut genomes = Vec::new();
    for (i, id) in ids.into_iter().enumerate() {
        let meta: [u32; 4] = header[i * 8..i * 8 + 4].try_into().unwrap();
        if meta[0] == 0
            || meta[0] as usize > crate::genome::MAX_MODES
            || meta[1] as u64 + meta[0] as u64 > b.mode_pool_capacity
        {
            return Err(format!("Invalid GPU genome metadata for {id}"));
        }
        genomes.push(SavedGpuGenome {
            id,
            metadata: meta,
            initial_orientation: std::array::from_fn(|c| f32::from_bits(header[i * 8 + 4 + c])),
            modes: BTreeMap::new(),
        });
    }
    let fields = genome_mode_buffers(b, a, colors, emissive);
    let total_words: u64 = genomes.iter().map(|g| g.metadata[0] as u64).sum::<u64>()
        * fields
            .iter()
            .map(|(_, _, width)| *width as u64)
            .sum::<u64>();
    let data = staging(device, total_words * 4);
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut offset = 0;
    for g in &genomes {
        for (_, buffer, width) in &fields {
            let bytes = g.metadata[0] as u64 * *width as u64 * 4;
            encoder.copy_buffer_to_buffer(
                buffer,
                g.metadata[1] as u64 * *width as u64 * 4,
                &data,
                offset,
                bytes,
            );
            offset += bytes;
        }
    }
    let words = read_words(device, queue, encoder, &data)?;
    let mut offset = 0;
    for g in &mut genomes {
        for (name, _, width) in &fields {
            let count = g.metadata[0] as usize * *width as usize;
            g.modes
                .insert((*name).into(), words[offset..offset + count].to_vec());
            offset += count;
        }
    }
    Ok(genomes)
}

pub fn restore(
    queue: &wgpu::Queue,
    b: &GpuTripleBufferSystem,
    a: &AdhesionBuffers,
    m: &mut MutationSystem,
    colors: &wgpu::Buffer,
    emissive: &wgpu::Buffer,
    genomes: &[SavedGpuGenome],
    authored_count: usize,
    authored_modes: u32,
) -> Result<(), String> {
    let fields = genome_mode_buffers(b, a, colors, emissive);
    let mut used_ids = BTreeSet::new();
    let mut ranges = Vec::new();
    let mut next_id = super::mutation::AUTHORED_GENOME_RESERVE.max(authored_count as u32);
    let mut next_mode = super::mutation::AUTHORED_MODE_RESERVE.max(authored_modes);
    // Validate the complete payload before any writes, including overlap checks.
    for g in genomes {
        let count = g.metadata[0];
        let start = g.metadata[1];
        let end = start.checked_add(count).ok_or("GPU mode range overflow")?;
        if g.id < authored_count as u32
            || g.id >= super::mutation::GENOME_RING_CAPACITY
            || !used_ids.insert(g.id)
            || count == 0
            || count as usize > crate::genome::MAX_MODES
            || start < authored_modes
            || end as u64 > b.mode_pool_capacity
            || g.metadata[2] >= count
        {
            return Err("Invalid GPU genome snapshot allocation".into());
        }
        for (name, _, width) in &fields {
            if g.modes.get(*name).map(Vec::len) != Some(count as usize * *width as usize) {
                return Err(format!("Invalid GPU genome field {name}"));
            }
        }
        ranges.push(start..end);
        next_id = next_id.max(g.id + 1);
        next_mode = next_mode.max(end);
    }
    ranges.sort_by_key(|range| range.start);
    if ranges.windows(2).any(|pair| pair[0].end > pair[1].start) {
        return Err("Overlapping GPU genome ranges".into());
    }
    for g in genomes {
        queue.write_buffer(
            m.genome_meta_buffer(),
            g.id as u64 * 16,
            bytemuck::cast_slice(&g.metadata),
        );
        queue.write_buffer(
            &m.genome_initial_orientations,
            g.id as u64 * 16,
            bytemuck::cast_slice(&g.initial_orientation),
        );
        queue.write_buffer(
            &m.genome_ref_counts_buffer,
            g.id as u64 * 4,
            bytemuck::bytes_of(&1u32),
        );
        for (name, buffer, width) in &fields {
            queue.write_buffer(
                buffer,
                g.metadata[1] as u64 * *width as u64 * 4,
                bytemuck::cast_slice(&g.modes[*name]),
            );
        }
    }
    queue.write_buffer(
        &m.genome_ring_state_buffer,
        0,
        bytemuck::cast_slice(&[0u32, 0, next_id, next_mode, next_mode]),
    );
    Ok(())
}

pub fn restore_cell_properties(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    b: &GpuTripleBufferSystem,
    slots: u32,
) {
    if slots == 0 {
        return;
    }
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Restore GPU Genome Cell Properties"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../shaders/restore_cell_properties.wgsl").into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Restore GPU Genome Cell Properties"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffers = [
        &b.cell_count_buffer,
        &b.mode_indices,
        &b.mode_cell_types,
        &b.mode_properties_v0,
        &b.mode_properties_v2,
        &b.cell_types,
        &b.max_splits,
        &b.nutrient_gain_rates,
        &b.max_cell_sizes,
        &b.stiffnesses,
    ];
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Restore Cell Properties"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bindings, &[]);
    pass.dispatch_workgroups(slots.div_ceil(64), 1, 1);
    drop(pass);
    queue.submit([encoder.finish()]);
}
