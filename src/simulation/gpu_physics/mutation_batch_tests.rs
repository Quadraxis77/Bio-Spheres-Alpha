use super::{MutationParamsUniform, MutationSystem, MUTATION_TELEMETRY_COUNTERS};
use wgpu::util::DeviceExt;

#[test]
fn type_mutation_probability_survives_repeated_dense_birth_batches() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    const CAPACITY: u32 = 20_000;
    const MODE_CAPACITY: u32 = 131_072;
    let mut system = MutationSystem::new(&device, &queue, CAPACITY);
    let storage = |size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Type mutation fixture"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let flags = storage(u64::from(CAPACITY) * 4);
    let assignments = storage(u64::from(CAPACITY) * 4);
    let genomes = storage(u64::from(CAPACITY) * 4);
    let modes = storage(u64::from(CAPACITY) * 4);
    let types = storage(u64::from(CAPACITY) * 4);
    let count = storage(4);
    let split_counts = storage(u64::from(CAPACITY) * 4);
    let runtime: Vec<_> = (0..6).map(|_| storage(u64::from(CAPACITY) * 4)).collect();
    queue.write_buffer(&count, 0, bytemuck::bytes_of(&CAPACITY));
    queue.write_buffer(
        &assignments,
        0,
        bytemuck::cast_slice(&(0..CAPACITY).map(|i| CAPACITY / 2 + i).collect::<Vec<_>>()),
    );
    let buffers: Vec<_> = (0..41)
        .map(|_| storage(u64::from(MODE_CAPACITY) * 16))
        .collect();
    queue.write_buffer(&buffers[11], 0, bytemuck::bytes_of(&2u32));
    let mut genome = crate::genome::Genome::default();
    genome.modes.truncate(1);
    genome.modes[0].cell_type = 2;
    system.sync_genome_metadata(&queue, &[genome]);
    // Isolate cell-type mutations, exercising several destination types rather
    // than relying on brightness or on survival after a mutation was applied.
    system.vulnerability_table = vec![super::MutationParamEntry {
        buffer_id: super::buffer_id::MODE_CELL_TYPES,
        element_offset: 0,
        weight: 1.0,
        min_delta: 0.0,
        max_delta: 0.0,
        min_value: 0.0,
        max_value: 19.0,
        data_type: super::data_type::INTEGER,
    }];
    queue.write_buffer(
        &system.vulnerability_table_buffer,
        0,
        bytemuck::cast_slice(&system.vulnerability_table),
    );
    system.rebuild_bind_groups(
        &device,
        &genomes,
        &modes,
        &types,
        &runtime[0],
        &runtime[1],
        &runtime[2],
        &runtime[3],
        &runtime[4],
        &runtime[5],
        &split_counts,
        &buffers[0],
        &buffers[1],
        &buffers[2],
        &buffers[3],
        &buffers[4],
        &buffers[5],
        &buffers[6],
        &buffers[7],
        &buffers[8],
        &buffers[9],
        &buffers[10],
        &buffers[11],
        &buffers[12],
        &buffers[13],
        &buffers[14],
        &buffers[15],
        &buffers[16],
        &buffers[19],
        &buffers[20],
        &buffers[21],
        &buffers[22],
        &buffers[23],
        &buffers[24],
        &buffers[25],
        &buffers[26],
        &buffers[27],
        &buffers[28],
        &buffers[29],
        &buffers[30],
        &buffers[31],
        &buffers[32],
        &buffers[33],
        &buffers[34],
        &buffers[35],
        &buffers[36],
        &buffers[37],
        &buffers[38],
        &buffers[39],
        &buffers[40],
    );
    system.rebuild_collect_bind_groups(&device, &queue, &flags, &assignments, &genomes, &count);
    let births = [(2_000u32, 16u32), (10_000, 3), (10_000, 7), (10_000, 16)];
    let types_size = u64::from(CAPACITY) * 4;
    let telemetry_size = u64::from(MUTATION_TELEMETRY_COUNTERS) * 4;
    let stride = types_size + telemetry_size + 20;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Applied mutation snapshots"),
        size: stride * births.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (step, &(divisions, target)) in births.iter().enumerate() {
        let mut divided = vec![0u32; CAPACITY as usize];
        divided[..divisions as usize].fill(1);
        for (destination, values) in [
            (&flags, divided),
            (&genomes, vec![0; CAPACITY as usize]),
            (&modes, vec![0; CAPACITY as usize]),
            (&types, vec![2; CAPACITY as usize]),
        ] {
            crate::simulation::gpu_upload::encode_buffer_write(
                &device,
                &mut encoder,
                destination,
                bytemuck::cast_slice(&values),
            );
        }
        system.set_radiation_level(1.0);
        system.set_cell_type_gene_pool_mask(1 << target);
        system.dispatch(
            &device,
            &mut encoder,
            &queue,
            1_000 + step as u32,
            MODE_CAPACITY,
        );
        let offset = step as u64 * stride;
        encoder.copy_buffer_to_buffer(&types, 0, &readback, offset, types_size);
        encoder.copy_buffer_to_buffer(
            &system.mutation_log_count_buffer,
            0,
            &readback,
            offset + types_size,
            telemetry_size,
        );
        encoder.copy_buffer_to_buffer(
            &system.genome_ring_state_buffer,
            0,
            &readback,
            offset + types_size + telemetry_size,
            20,
        );
    }
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let data = readback.slice(..).get_mapped_range();
    for (step, &(divisions, target)) in births.iter().enumerate() {
        let snapshot = &data[step * stride as usize..][..stride as usize];
        let types: &[u32] = bytemuck::cast_slice(&snapshot[..types_size as usize]);
        let telemetry: &[u32] = bytemuck::cast_slice(
            &snapshot[types_size as usize..(types_size + telemetry_size) as usize],
        );
        let allocator: &[u32] =
            bytemuck::cast_slice(&snapshot[(types_size + telemetry_size) as usize..]);
        assert!(
            allocator[3] < MODE_CAPACITY,
            "fixture must retain free mode space"
        );
        assert_eq!(
            &telemetry[1..5],
            &[0, 0, 0, 0],
            "batch {step}, mode offset {}: overflow, allocation failures, and no-ops must not silently lower the rate", allocator[3]
        );
        let mutated = types.iter().filter(|&&t| t == target).count();
        let rate = mutated as f32 / (divisions * 2) as f32;
        // Existing core-mode protection applies a 0.25 exposure multiplier.
        assert!(
            (0.21..0.29).contains(&rate),
            "batch {step}, target {target}: applied {mutated} type mutations, rate {rate}"
        );
        assert_eq!(telemetry[0] as usize, mutated);
        for (cell, &cell_type) in types.iter().enumerate() {
            let newborn = (cell as u32) < divisions
                || (CAPACITY / 2..CAPACITY / 2 + divisions).contains(&(cell as u32));
            assert!(cell_type == 2 || (newborn && cell_type == target));
        }
    }
}

#[test]
fn dense_divisions_keep_all_newborns_and_each_batched_step_keeps_its_own_state() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    const CAPACITY: u32 = 20_000;
    let mut system = MutationSystem::new(&device, &queue, CAPACITY);
    assert_eq!(
        system.mutation_candidates_buffer.size(),
        u64::from(CAPACITY) * 8
    );
    let storage = |values: &[u32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mutation batch fixture"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let flags = storage(&vec![0; CAPACITY as usize]);
    let assignments = storage(&(0..CAPACITY).map(|i| CAPACITY / 2 + i).collect::<Vec<_>>());
    let genomes = storage(&vec![42; CAPACITY as usize]);
    let count = storage(&[CAPACITY]);
    system.rebuild_collect_bind_groups(&device, &queue, &flags, &assignments, &genomes, &count);
    let params_size = std::mem::size_of::<MutationParamsUniform>() as u64;
    // Readback is only needed by this regression.
    system.params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mutation parameter snapshots"),
        size: params_size,
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let divisions = [250u32, 5_000, 1_500, 7_000, 17, 8_192, 4_097, 10_000];
    let candidates_size = system.mutation_candidates_buffer.size();
    let telemetry_size = u64::from(MUTATION_TELEMETRY_COUNTERS) * 4;
    let stride = 4 + params_size + telemetry_size + candidates_size;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mutation batch readback"),
        size: stride * divisions.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (step, &divisions) in divisions.iter().enumerate() {
        let mut flags_data = vec![0u32; CAPACITY as usize];
        flags_data[..divisions as usize].fill(1);
        crate::simulation::gpu_upload::encode_buffer_write(
            &device,
            &mut encoder,
            &flags,
            bytemuck::cast_slice(&flags_data),
        );
        system.collect_candidates(
            &mut encoder,
            system.collect_input_bind_group.as_ref().unwrap(),
            system.collect_output_bind_group.as_ref().unwrap(),
        );
        system.radiation_level = 0.125 * (step + 1) as f32;
        system.cell_type_gene_pool_mask = 1 << (step + 1);
        system.subtle_mutations = step % 2 == 0;
        // A per-step telemetry reset must occur after earlier encoded work.
        crate::simulation::gpu_upload::encode_buffer_write(
            &device,
            &mut encoder,
            &system.mutation_log_count_buffer,
            bytemuck::cast_slice(&[99u32; MUTATION_TELEMETRY_COUNTERS as usize]),
        );
        system.encode_dispatch_params(
            &device,
            &mut encoder,
            100 + step as u32,
            65_536 + step as u32,
        );
        let offset = stride * step as u64;
        encoder.copy_buffer_to_buffer(
            &system.mutation_candidate_count_buffer,
            0,
            &readback,
            offset,
            4,
        );
        encoder.copy_buffer_to_buffer(&system.params_buffer, 0, &readback, offset + 4, params_size);
        encoder.copy_buffer_to_buffer(
            &system.mutation_log_count_buffer,
            0,
            &readback,
            offset + 4 + params_size,
            telemetry_size,
        );
        encoder.copy_buffer_to_buffer(
            &system.mutation_candidates_buffer,
            0,
            &readback,
            offset + 4 + params_size + telemetry_size,
            candidates_size,
        );
    }
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = readback.slice(..).get_mapped_range();
    for (step, &divisions) in divisions.iter().enumerate() {
        let snapshot = &bytes[step * stride as usize..][..stride as usize];
        assert_eq!(
            u32::from_ne_bytes(snapshot[..4].try_into().unwrap()),
            2 * divisions,
            "candidate count must reset between steps {step}"
        );
        let params: MutationParamsUniform =
            bytemuck::pod_read_unaligned(&snapshot[4..4 + params_size as usize]);
        assert_eq!(params.rng_seed, step as u32);
        assert_eq!(params.current_frame, 100 + step as u32);
        assert_eq!(params.radiation_level, 0.125 * (step + 1) as f32);
        assert_eq!(params.cell_type_gene_pool_mask, 1 << (step + 1));
        assert_eq!(params.subtle_color_mutation, u32::from(step % 2 == 0));
        let telemetry_start = 4 + params_size as usize;
        assert!(
            snapshot[telemetry_start..telemetry_start + telemetry_size as usize]
                .iter()
                .all(|&byte| byte == 0)
        );
        let candidates: &[[u32; 2]] =
            bytemuck::cast_slice(&snapshot[telemetry_start + telemetry_size as usize..]);
        let mut seen = vec![false; CAPACITY as usize];
        for &[cell, genome] in &candidates[..(2 * divisions) as usize] {
            assert_eq!(genome, 42);
            assert!(
                !seen[cell as usize],
                "a newborn must not be collected twice"
            );
            seen[cell as usize] = true;
        }
        for cell in 0..CAPACITY {
            let expected =
                cell < divisions || (CAPACITY / 2..CAPACITY / 2 + divisions).contains(&cell);
            assert_eq!(
                seen[cell as usize], expected,
                "missing newborn {cell} at step {step}"
            );
        }
    }
}
