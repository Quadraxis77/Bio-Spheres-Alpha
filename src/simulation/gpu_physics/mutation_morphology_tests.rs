use super::{buffer_id, data_type, MutationParamEntry, MutationSystem};
use crate::genome::{Genome, ModeSettings};
use glam::{Quat, Vec3};

fn read<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<T> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mutated morphology readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, offset, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

// Feed actual GPU-mutated geometry and bond settings into the real division
// executor. This checks physical connectivity, not just matching quaternions.
fn assert_structure(genome: &Genome, pattern: u32) {
    use crate::cell::division::division_step;
    use crate::scene::preview_state::InitialState;
    use crate::simulation::physics_config::PhysicsConfig;

    let config = PhysicsConfig::default();
    let mut state = InitialState::from_genome(genome, 64, &config).to_canonical_state();
    let generations = if pattern == 0 { 2 } else { 3 };
    for generation in 0..generations {
        for cell in 0..state.cell_count {
            state.nutrients[cell] = 100.0;
            state.reserves[cell] = 0;
        }
        division_step(&mut state, genome, 10.0 * (generation + 1) as f32, 64, 42);
        assert_eq!(state.cell_count, 1 << (generation + 1));
    }
    let expected_cells = 1 << generations;
    let mut edges = std::collections::HashSet::new();
    for bond in 0..state.adhesion_connections.active_count {
        if state.adhesion_connections.is_active[bond] != 0 {
            let a = state.adhesion_connections.cell_a_index[bond];
            let b = state.adhesion_connections.cell_b_index[bond];
            edges.insert((a.min(b), a.max(b)));
        }
    }
    if pattern == 2 {
        let max_degree = (0..expected_cells)
            .map(|cell| {
                edges
                    .iter()
                    .filter(|&&(a, b)| a == cell || b == cell)
                    .count()
            })
            .max()
            .unwrap();
        assert!(
            max_degree >= 3,
            "branching must produce a junction, not just a bent chain"
        );
        let mut connected = std::collections::HashSet::from([0]);
        for _ in 0..expected_cells {
            for &(a, b) in &edges {
                if connected.contains(&a) {
                    connected.insert(b);
                }
                if connected.contains(&b) {
                    connected.insert(a);
                }
            }
        }
        assert_eq!(connected.len(), expected_cells);
        return;
    }
    let expected_degree = generations;
    assert_eq!(edges.len(), expected_cells * expected_degree / 2);
    for cell in 0..expected_cells {
        let degree = edges
            .iter()
            .filter(|&&(a, b)| a as usize == cell || b as usize == cell)
            .count();
        assert_eq!(degree, expected_degree, "pattern {pattern}, cell {cell}");
    }
    // Split axes span a plane for a square and all three dimensions for a cube.
    let q = genome.modes[0].child_a.orientation;
    let a = Vec3::Z;
    let b = q * a;
    assert!(a.dot(b).abs() < 1e-5);
    let c = q * b;
    assert!((a.cross(b).dot(c).abs() - if pattern == 0 { 0.0 } else { 1.0 }).abs() < 1e-5);
}

const CELLS: u32 = 128;
const MODES: u32 = super::AUTHORED_MODE_RESERVE + CELLS * 64;

struct MutationFixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    system: MutationSystem,
    genomes: wgpu::Buffer,
    modes: wgpu::Buffer,
    _types: wgpu::Buffer,
    buffers: Vec<wgpu::Buffer>,
    runtime: Vec<wgpu::Buffer>,
    split_counts: wgpu::Buffer,
}

impl MutationFixture {
    fn mutate(&mut self, entry: MutationParamEntry, frame: u32) {
        self.system
            .update_vulnerability_table(&self.queue, vec![entry]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.system
            .encode_dispatch_params(&self.device, &mut encoder, frame, MODES);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.system.pipeline);
            pass.set_bind_group(0, self.system.params_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, self.system.candidates_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, self.system.buffers_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(CELLS.div_ceil(64), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
    }
}

fn mutation_fixture(source_mode_count: usize) -> MutationFixture {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();

    let mut system = MutationSystem::new(&device, &queue, CELLS);
    system.set_cell_type_gene_pool_mask(1 << 2);
    let storage = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morphology mutation fixture"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let genomes = storage(u64::from(CELLS) * 4);
    let modes = storage(u64::from(CELLS) * 4);
    let types = storage(u64::from(CELLS) * 4);
    let split_counts = storage(u64::from(CELLS) * 4);
    let runtime: Vec<_> = (0..6).map(|_| storage(u64::from(CELLS) * 4)).collect();
    let buffers: Vec<_> = (0..41).map(|_| storage(u64::from(MODES) * 16)).collect();
    let mut parent = Genome::default();
    parent.modes = vec![ModeSettings::default(); source_mode_count];
    parent.modes[0].cell_type = 2;
    parent.initial_mode = 0;
    system.sync_genome_metadata(&queue, &[parent]);
    queue.write_buffer(
        &buffers[2],
        0,
        bytemuck::cast_slice(&[3.0f32, 0.65, -1.0, 0.0]),
    );
    queue.write_buffer(
        &buffers[4],
        0,
        bytemuck::cast_slice(&[2.0f32, -1.0, -1.0, 0.0]),
    );
    queue.write_buffer(&buffers[11], 0, bytemuck::bytes_of(&2u32));
    // Begin with straight growth, no bonds, and asymmetric inheritance.
    for buffer in &buffers[5..10] {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.0]));
    }
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
    let candidates: Vec<[u32; 2]> = (0..CELLS).map(|cell| [cell, 0]).collect();
    queue.write_buffer(
        &system.mutation_candidates_buffer,
        0,
        bytemuck::cast_slice(&candidates),
    );
    queue.write_buffer(
        &system.mutation_candidate_count_buffer,
        0,
        bytemuck::bytes_of(&CELLS),
    );
    system.set_radiation_level(1.0);

    MutationFixture {
        device,
        queue,
        system,
        genomes,
        modes,
        _types: types,
        buffers,
        runtime,
        split_counts,
    }
}

#[test]
fn gpu_body_plan_mutations_develop_connected_squares_cubes_and_branches() {
    let MutationFixture {
        device,
        queue,
        mut system,
        genomes,
        modes,
        buffers,
        ..
    } = mutation_fixture(1);
    for pattern in 0..3 {
        system.update_vulnerability_table(
            &queue,
            vec![MutationParamEntry {
                buffer_id: buffer_id::GENOME_MODE_DATA,
                element_offset: pattern,
                weight: 1.0,
                min_delta: 0.0,
                max_delta: 0.0,
                min_value: 0.0,
                max_value: 0.0,
                data_type: data_type::BODY_PLAN,
            }],
        );
        queue.write_buffer(
            &genomes,
            0,
            bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
        );
        queue.write_buffer(&modes, 0, bytemuck::cast_slice(&vec![0u32; CELLS as usize]));
        let mut encoder = device.create_command_encoder(&Default::default());
        system.encode_dispatch_params(&device, &mut encoder, pattern, MODES);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&system.pipeline);
            pass.set_bind_group(0, system.params_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, system.candidates_bind_group.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, system.buffers_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(CELLS.div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let ids: Vec<u32> = read(&device, &queue, &genomes, 0, u64::from(CELLS) * 4);
        let mutated: Vec<_> = ids.into_iter().filter(|&id| id != 0).collect();
        assert!(
            mutated.len() >= 8,
            "fixture needs several independently mutated genomes"
        );
        for &id in mutated.iter().take(8) {
            let meta: Vec<u32> = read(
                &device,
                &queue,
                &system.genome_meta_buffer,
                u64::from(id) * 16,
                16,
            );
            assert_eq!(
                meta[0], 2,
                "single-mode mutation must seed its intentional development partner"
            );
            let base = meta[1];
            let vecs = |index: usize| -> Vec<[f32; 4]> {
                read(
                    &device,
                    &queue,
                    &buffers[index],
                    u64::from(base) * 16,
                    u64::from(meta[0]) * 16,
                )
            };
            let p2 = vecs(2);
            let p4 = vecs(4);
            let rotations: Vec<_> = (5..10).map(vecs).collect();
            let routes: Vec<[i32; 2]> = read(
                &device,
                &queue,
                &buffers[10],
                u64::from(base) * 8,
                u64::from(meta[0]) * 8,
            );
            let cell_types: Vec<u32> = read(
                &device,
                &queue,
                &buffers[11],
                u64::from(base) * 4,
                u64::from(meta[0]) * 4,
            );
            let mut offspring = Genome::default();
            offspring.initial_mode = 0;
            offspring.modes = vec![ModeSettings::default(); meta[0] as usize];
            for m in 0..meta[0] as usize {
                let mode = &mut offspring.modes[m];
                assert_eq!(cell_types[m], 2);
                mode.cell_type = cell_types[m] as i32;
                mode.max_splits = p2[m][0] as i32;
                mode.split_ratio = p2[m][1];
                mode.max_adhesions = p4[m][0] as i32;
                assert_eq!(mode.split_ratio, 0.5);
                assert_eq!(mode.max_splits, if m == 0 { 3 } else { 2 });
                assert!(mode.max_adhesions >= 12);
                assert_eq!(rotations[4][m], [0.0, 0.0, 0.0, 1.0]);
                mode.child_a.orientation = Quat::from_array(rotations[0][m]);
                mode.child_b.orientation = Quat::from_array(rotations[1][m]);
                mode.child_a_after_split_orientation = Quat::from_array(rotations[2][m]);
                mode.child_b_after_split_orientation = Quat::from_array(rotations[3][m]);
                mode.child_a.mode_number = routes[m][0] - base as i32;
                mode.child_b.mode_number = routes[m][1] - base as i32;
                assert!((0..meta[0] as i32).contains(&mode.child_a.mode_number));
                assert!((0..meta[0] as i32).contains(&mode.child_b.mode_number));
                if m == 0 {
                    assert_eq!(mode.child_a.mode_number, 1);
                } else {
                    assert_eq!(mode.child_a.mode_number, 0);
                }
                for index in [12, 13, 14, 30, 31] {
                    let flags: Vec<u32> = read(
                        &device,
                        &queue,
                        &buffers[index],
                        u64::from(base) * 4,
                        u64::from(meta[0]) * 4,
                    );
                    assert_eq!(flags[m], 1);
                }
                mode.parent_make_adhesion = true;
                mode.child_a.keep_adhesion = true;
                mode.child_b.keep_adhesion = true;
                mode.child_a_after_split_keep_adhesion = true;
                mode.child_b_after_split_keep_adhesion = true;
                assert!(mode.child_a.orientation.is_normalized());
                if pattern < 2 {
                    assert_eq!(mode.child_a.orientation, mode.child_b.orientation);
                } else {
                    assert_ne!(mode.child_a.orientation, mode.child_b.orientation);
                    assert_eq!(mode.child_b.orientation, Quat::IDENTITY);
                }
            }
            assert_structure(&offspring, pattern);
        }
        // Mutation works on a clone; authored source settings must survive.
        let source: Vec<f32> = read(&device, &queue, &buffers[2], 0, 16);
        assert_eq!(source[1], 0.65);
    }
    // The user's subtle-mutation setting still excludes structural jumps.
    system.set_subtle_mutations(&queue, true);
    assert!(!system
        .vulnerability_table
        .iter()
        .any(|entry| entry.data_type == data_type::BODY_PLAN));
    system.set_subtle_mutations(&queue, false);
    assert_eq!(
        system
            .vulnerability_table
            .iter()
            .filter(|entry| entry.data_type == data_type::BODY_PLAN)
            .count(),
        3
    );
}

#[test]
fn repeated_radiation_keeps_development_cycles_and_updates_newborn_settings() {
    let mut fixture = mutation_fixture(3);
    let p0 = [
        [0.1f32, 1.2, 250.0, 60.0],
        [0.2, 1.3, 300.0, 12.0],
        [0.3, 1.4, 350.0, 9.0],
    ];
    let p2 = [
        [0.0f32, 0.65, -1.0, 0.0],
        [1.0, 0.65, -1.0, 0.0],
        [-1.0, 0.65, -1.0, 0.0],
    ];
    fixture
        .queue
        .write_buffer(&fixture.buffers[0], 0, bytemuck::cast_slice(&p0));
    fixture
        .queue
        .write_buffer(&fixture.buffers[2], 0, bytemuck::cast_slice(&p2));
    fixture.queue.write_buffer(
        &fixture.buffers[11],
        0,
        bytemuck::cast_slice(&[2u32, 2, 19]),
    );
    fixture.queue.write_buffer(
        &fixture.buffers[10],
        0,
        bytemuck::cast_slice(&[[0i32; 2], [1; 2], [2; 2]]),
    );
    // A released initial cell must be able to germinate despite inherited gates.
    fixture.queue.write_buffer(
        &fixture.buffers[3],
        0,
        bytemuck::cast_slice(&[[0.0f32, 0.0, 0.0, 3.0]; 3]),
    );
    fixture.queue.write_buffer(
        &fixture.buffers[24],
        0,
        bytemuck::cast_slice(&[[8.0f32, 200.0, 0.0, -1.0]; 3]),
    );
    let local_modes: Vec<u32> = (0..CELLS).map(|cell| cell % 3).collect();
    fixture
        .queue
        .write_buffer(&fixture.modes, 0, bytemuck::cast_slice(&local_modes));
    let mut previous_ids = vec![0u32; CELLS as usize];
    for step in 0..12 {
        fixture.mutate(
            MutationParamEntry {
                buffer_id: buffer_id::MODE_VISUALS,
                element_offset: 0,
                data_type: data_type::CONTINUOUS_F32,
                weight: 1.0,
                min_delta: 0.01,
                max_delta: 0.01,
                min_value: 0.0,
                max_value: 1.0,
            },
            step,
        );
        let ids: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.genomes,
            0,
            u64::from(CELLS) * 4,
        );
        let active_modes: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.modes,
            0,
            u64::from(CELLS) * 4,
        );
        let caches: Vec<Vec<u32>> = fixture
            .runtime
            .iter()
            .map(|buffer| {
                read(
                    &fixture.device,
                    &fixture.queue,
                    buffer,
                    0,
                    u64::from(CELLS) * 4,
                )
            })
            .collect();
        let changed: Vec<_> = ids
            .iter()
            .enumerate()
            .filter(|&(cell, id)| *id != previous_ids[cell])
            .collect();
        assert!(!changed.is_empty(), "must exercise actual GPU radiation");
        for &(cell, &id) in changed.iter().take(6) {
            let meta: Vec<u32> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.system.genome_meta_buffer,
                u64::from(id) * 16,
                16,
            );
            assert_eq!(meta[0], 3);
            let base = meta[1];
            let props = |index: usize| -> Vec<[f32; 4]> {
                read(
                    &fixture.device,
                    &fixture.queue,
                    &fixture.buffers[index],
                    u64::from(base) * 16,
                    48,
                )
            };
            let routes: Vec<[i32; 2]> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.buffers[10],
                u64::from(base) * 8,
                24,
            );
            // The initial mode's established partner remains the return path.
            assert_eq!(routes[0][0], base as i32 + 1);
            assert!(routes[1].contains(&(base as i32)));
            for children in &routes {
                assert!(children
                    .iter()
                    .all(|&child| (base as i32..(base + 3) as i32).contains(&child)));
            }
            let actual0 = props(0);
            let actual1 = props(1);
            let actual2 = props(2);
            assert_eq!(actual0[0][3], 8.0);
            assert_eq!(actual0[1][3], 12.0);
            assert_eq!(actual0[2][3], 4.0);
            assert_eq!(
                actual2.iter().map(|p| p[0]).collect::<Vec<_>>(),
                vec![2.0, 2.0, 5.0]
            );
            assert_eq!(props(3)[0][3], 0.0);
            assert_eq!(props(24)[0][0], -1.0);
            assert_eq!(props(24)[1][0], 8.0, "downstream gates remain available");
            for index in [12, 13, 14, 30, 31] {
                let flags: Vec<u32> = read(
                    &fixture.device,
                    &fixture.queue,
                    &fixture.buffers[index],
                    u64::from(base) * 4,
                    4,
                );
                assert_eq!(flags[0], 1, "initial germination must retain adhesion");
            }
            // A Stemocyte retains self-renewing division with in-genome fates.
            assert_eq!(routes[2], [base as i32 + 2; 2]);
            let v9 = props(32)[2];
            let v10 = props(33)[2];
            for target in [v9[3], v10[0], v10[1], v10[2]] {
                assert!((base as f32..(base + 3) as f32).contains(&target));
            }
            let local = (active_modes[cell] - base) as usize;
            assert_eq!(f32::from_bits(caches[0][cell]), actual0[local][3]);
            assert_eq!(
                f32::from_bits(caches[1][cell]),
                (actual1[local][0] - 1.0) * 100.0
            );
            assert_eq!(caches[2][cell], actual2[local][0] as u32);
            for (cache, component) in [(3, 0), (4, 1), (5, 2)] {
                assert_eq!(
                    f32::from_bits(caches[cache][cell]),
                    actual0[local][component]
                );
            }
        }
        previous_ids = ids;
        let candidates: Vec<[u32; 2]> = previous_ids
            .iter()
            .enumerate()
            .map(|(cell, &id)| [cell as u32, id])
            .collect();
        fixture.queue.write_buffer(
            &fixture.system.mutation_candidates_buffer,
            0,
            bytemuck::cast_slice(&candidates),
        );
    }
    // Seeding and germination modify the clone, never the authored source.
    let source: Vec<[f32; 4]> = read(&fixture.device, &fixture.queue, &fixture.buffers[0], 0, 48);
    assert_eq!(source, p0);
}

#[test]
fn singleton_append_reserves_space_for_both_cycle_partner_and_new_mode() {
    let mut fixture = mutation_fixture(1);
    fixture.mutate(
        MutationParamEntry {
            buffer_id: buffer_id::GENOME_STRUCTURE,
            element_offset: 0,
            data_type: data_type::MODE_APPEND,
            weight: 1.0,
            min_delta: 0.0,
            max_delta: 0.0,
            min_value: 0.0,
            max_value: 0.0,
        },
        0,
    );
    let ids: Vec<u32> = read(
        &fixture.device,
        &fixture.queue,
        &fixture.genomes,
        0,
        u64::from(CELLS) * 4,
    );
    let mut ranges = Vec::new();
    for id in ids.into_iter().filter(|&id| id != 0) {
        let meta: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.system.genome_meta_buffer,
            u64::from(id) * 16,
            16,
        );
        assert_eq!(
            meta[0], 3,
            "the seeded partner and explicit appended mode need separate slots"
        );
        ranges.push(meta[1]..meta[1] + meta[0]);
        let routes: Vec<[i32; 2]> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.buffers[10],
            u64::from(meta[1]) * 8,
            24,
        );
        assert_eq!(routes[0][0], meta[1] as i32 + 1);
        assert!(routes[1].contains(&(meta[1] as i32)));
        assert!(routes
            .iter()
            .flatten()
            .all(|&child| ranges.last().unwrap().contains(&(child as u32))));
    }
    assert!(ranges.len() > 8);
    ranges.sort_by_key(|range| range.start);
    for pair in ranges.windows(2) {
        assert!(
            pair[0].end <= pair[1].start,
            "mutation mode allocations must not overlap"
        );
    }
}

#[test]
fn split_interval_mutations_respect_one_second_floor_in_both_modes() {
    let mut fixture = mutation_fixture(1);
    let mut frame = 0;
    for subtle in [false, true] {
        fixture.system.set_subtle_mutations(&fixture.queue, subtle);
        let entry = *fixture
            .system
            .vulnerability_table
            .iter()
            .find(|entry| {
                entry.buffer_id == buffer_id::MODE_PROPERTIES && entry.element_offset == 3
            })
            .expect("split interval must remain mutable");
        for initial in [1.0f32, 1.1, 59.9, 60.0] {
            fixture.queue.write_buffer(
                &fixture.genomes,
                0,
                bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
            );
            fixture.queue.write_buffer(
                &fixture.modes,
                0,
                bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
            );
            fixture.queue.write_buffer(
                &fixture.buffers[0],
                0,
                bytemuck::cast_slice(&[0.1f32, 1.2, 250.0, initial]),
            );
            fixture.mutate(entry, frame);
            frame += 1;

            let ids: Vec<u32> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.genomes,
                0,
                u64::from(CELLS) * 4,
            );
            let modes: Vec<u32> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.modes,
                0,
                u64::from(CELLS) * 4,
            );
            let runtime: Vec<f32> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.runtime[0],
                0,
                u64::from(CELLS) * 4,
            );
            let properties: Vec<[f32; 4]> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.buffers[0],
                0,
                u64::from(MODES) * 16,
            );
            let mut changed = 0;
            let mut increased = false;
            let mut decreased = false;
            let mut stayed_at_floor = false;
            for (cell, id) in ids.iter().enumerate().filter(|&(_, &id)| id != 0) {
                changed += 1;
                let interval = properties[modes[cell] as usize][3];
                assert!((1.0..=60.0).contains(&interval),
                    "subtle={subtle}, initial={initial}, genome={id}: split interval mutated to {interval}");
                assert_eq!(
                    runtime[cell], interval,
                    "newborn must use the bounded interval immediately"
                );
                increased |= interval > initial;
                decreased |= interval < initial;
                stayed_at_floor |= interval == 1.0;
            }
            assert!(changed > 0, "must exercise actual GPU mutations");
            if initial == 1.0 {
                assert!(
                    increased && stayed_at_floor,
                    "at one second, mutations can increase or stop at the floor"
                );
            }
            if initial == 60.0 {
                assert!(
                    decreased,
                    "longer intervals must still be able to evolve downward within the valid range"
                );
            }
        }
    }
}

#[test]
fn division_parameter_mutations_refresh_the_active_cells_cache_immediately() {
    let mut fixture = mutation_fixture(1);
    for (step, offset, value, cache) in [(0u32, 3, 17.0f32, 0), (1, 4, 1.75, 1), (2, 8, 0.0, 2)] {
        fixture.queue.write_buffer(
            &fixture.genomes,
            0,
            bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
        );
        fixture.queue.write_buffer(
            &fixture.modes,
            0,
            bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
        );
        fixture.mutate(
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES,
                element_offset: offset,
                data_type: data_type::CONTINUOUS_F32,
                weight: 1.0,
                min_delta: 0.0,
                max_delta: 0.0,
                min_value: value,
                max_value: value,
            },
            step,
        );
        let ids: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.genomes,
            0,
            u64::from(CELLS) * 4,
        );
        let actual: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.runtime[cache],
            0,
            u64::from(CELLS) * 4,
        );
        assert!(ids.iter().any(|&id| id != 0));
        let active_modes: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.modes,
            0,
            u64::from(CELLS) * 4,
        );
        for (cell, &id) in ids.iter().enumerate().filter(|&(_, &id)| id != 0) {
            let mode = active_modes[cell];
            let properties: Vec<[f32; 4]> = read(
                &fixture.device,
                &fixture.queue,
                &fixture.buffers[if offset == 8 {
                    2
                } else if offset == 4 {
                    1
                } else {
                    0
                }],
                u64::from(mode) * 16,
                16,
            );
            let expected = match offset {
                4 => ((properties[0][0] - 1.0) * 100.0).to_bits(),
                8 => properties[0][0] as u32,
                _ => properties[0][3].to_bits(),
            };
            assert_eq!(
                actual[cell], expected,
                "genome {id} has stale runtime parameters"
            );
        }
    }
}

#[test]
fn trimming_preserves_live_modes_and_developmental_references() {
    let mut fixture = mutation_fixture(3);
    let local_modes: Vec<u32> = (0..CELLS).map(|cell| cell % 3).collect();
    for step in 0..4 {
        fixture.queue.write_buffer(
            &fixture.genomes,
            0,
            bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
        );
        fixture
            .queue
            .write_buffer(&fixture.modes, 0, bytemuck::cast_slice(&local_modes));
        // Each of these references independently makes the tail indispensable.
        fixture.queue.write_buffer(
            &fixture.buffers[4],
            16,
            bytemuck::cast_slice(&[2.0f32, if step == 1 { 2.0 } else { -1.0 }, -1.0, 0.0]),
        );
        fixture.queue.write_buffer(
            &fixture.buffers[28],
            16,
            bytemuck::cast_slice(&[if step == 2 { 2.0f32 } else { -1.0 }, 0.0, 0.0, 0.0]),
        );
        if step == 3 {
            fixture
                .queue
                .write_buffer(&fixture.buffers[11], 4, bytemuck::bytes_of(&19u32));
            fixture.queue.write_buffer(
                &fixture.buffers[26],
                16,
                bytemuck::cast_slice(&[2.0f32, -1.0, -1.0, 1.0]),
            );
        }
        fixture.mutate(
            MutationParamEntry {
                buffer_id: buffer_id::GENOME_STRUCTURE,
                element_offset: 0,
                data_type: data_type::MODE_TRIM,
                weight: 1.0,
                min_delta: 0.0,
                max_delta: 0.0,
                min_value: 0.0,
                max_value: 0.0,
            },
            step,
        );
        let ids: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.genomes,
            0,
            u64::from(CELLS) * 4,
        );
        if step > 0 {
            assert!(
                ids.iter().all(|&id| id == 0),
                "a referenced tail mode must not be trimmed"
            );
            continue;
        }
        assert!(
            ids.iter().any(|&id| id != 0),
            "unreferenced inactive tail modes should remain trimmable"
        );
        for (cell, &id) in ids.iter().enumerate() {
            if local_modes[cell] == 2 {
                assert_eq!(id, 0, "the newborn must not lose its active mode");
            }
            if id != 0 {
                let meta: Vec<u32> = read(
                    &fixture.device,
                    &fixture.queue,
                    &fixture.system.genome_meta_buffer,
                    u64::from(id) * 16,
                    16,
                );
                assert_eq!(meta[0], 2);
            }
        }
    }
}

#[test]
fn authored_mode_cycles_still_reset_split_counts() {
    use crate::cell::division::division_step;
    use crate::scene::preview_state::InitialState;
    use crate::simulation::physics_config::PhysicsConfig;

    for cycle in [false, true] {
        let mut genome = Genome::default();
        genome.initial_mode = 0;
        genome.modes = vec![ModeSettings::default(); 2];
        for (index, mode) in genome.modes.iter_mut().enumerate() {
            mode.cell_type = 2;
            mode.max_splits = 1;
            mode.split_interval = 1.0;
            let next = if cycle { 1 - index } else { index } as i32;
            mode.child_a.mode_number = next;
            mode.child_b.mode_number = next;
        }
        let mut state =
            InitialState::from_genome(&genome, 32, &PhysicsConfig::default()).to_canonical_state();
        for generation in 1..=4 {
            for cell in 0..state.cell_count {
                state.nutrients[cell] = 100.0;
            }
            division_step(&mut state, &genome, generation as f32 * 2.0, 32, 42);
            assert_eq!(state.cell_count, if cycle { 1 << generation } else { 2 });
            for cell in 0..state.cell_count {
                assert_eq!(state.split_counts[cell], if cycle { 0 } else { 1 });
            }
        }
    }
}

#[test]
fn old_default_lineages_can_mutate_into_connected_multicellular_growth() {
    use crate::cell::division::division_step;
    use crate::scene::preview_state::InitialState;
    use crate::simulation::physics_config::PhysicsConfig;

    let source = Genome::default();
    let mut fixture = mutation_fixture(source.modes.len());
    fixture
        .system
        .sync_genome_metadata(&fixture.queue, &[source.clone()]);
    // Upload the actual default's division program, rather than the deliberately
    // adhesive, low-count source used by the isolated geometry tests.
    let properties: Vec<Vec<[f32; 4]>> = (0..5)
        .map(|index| {
            source
                .modes
                .iter()
                .map(|mode| match index {
                    0 => [
                        mode.nutrient_gain_rate,
                        mode.max_cell_size,
                        mode.membrane_stiffness,
                        mode.split_interval,
                    ],
                    1 => [
                        mode.split_mass,
                        mode.nutrient_priority,
                        mode.swim_force,
                        1.0,
                    ],
                    2 => [mode.max_splits as f32, mode.split_ratio, -1.0, 0.0],
                    3 => [0.0, 0.0, 0.0, mode.min_adhesions as f32],
                    _ => [
                        mode.max_adhesions as f32,
                        mode.mode_a_after_splits as f32,
                        mode.mode_b_after_splits as f32,
                        0.0,
                    ],
                })
                .collect()
        })
        .collect();
    for (index, values) in properties.iter().enumerate() {
        fixture
            .queue
            .write_buffer(&fixture.buffers[index], 0, bytemuck::cast_slice(values));
    }
    for (index, flag) in [(12, 0), (13, 1), (14, 1), (30, 1), (31, 1)] {
        fixture.queue.write_buffer(
            &fixture.buffers[index],
            0,
            bytemuck::cast_slice(&vec![flag as u32; source.modes.len()]),
        );
    }
    let routes: Vec<_> = source
        .modes
        .iter()
        .map(|mode| [mode.child_a.mode_number, mode.child_b.mode_number])
        .collect();
    fixture
        .queue
        .write_buffer(&fixture.buffers[10], 0, bytemuck::cast_slice(&routes));
    fixture.queue.write_buffer(
        &fixture.buffers[11],
        0,
        bytemuck::cast_slice(&vec![2u32; source.modes.len()]),
    );
    for buffer in &fixture.buffers[5..10] {
        fixture.queue.write_buffer(
            buffer,
            0,
            bytemuck::cast_slice(&vec![[0.0f32, 0.0, 0.0, 1.0]; source.modes.len()]),
        );
    }
    let old_counts: Vec<u32> = (0..CELLS)
        .map(|cell| [0, 1, 4, 5, 7, 32][cell as usize % 6])
        .collect();
    fixture
        .queue
        .write_buffer(&fixture.split_counts, 0, bytemuck::cast_slice(&old_counts));
    fixture.queue.write_buffer(
        &fixture.runtime[2],
        0,
        bytemuck::cast_slice(&vec![u32::MAX; CELLS as usize]),
    );
    fixture.mutate(
        MutationParamEntry {
            buffer_id: buffer_id::MODE_VISUALS,
            element_offset: 0,
            data_type: data_type::CONTINUOUS_F32,
            weight: 1.0,
            min_delta: 0.01,
            max_delta: 0.01,
            min_value: 0.0,
            max_value: 1.0,
        },
        0,
    );
    let ids: Vec<u32> = read(
        &fixture.device,
        &fixture.queue,
        &fixture.genomes,
        0,
        u64::from(CELLS) * 4,
    );
    let counts: Vec<u32> = read(
        &fixture.device,
        &fixture.queue,
        &fixture.split_counts,
        0,
        u64::from(CELLS) * 4,
    );
    let caps: Vec<u32> = read(
        &fixture.device,
        &fixture.queue,
        &fixture.runtime[2],
        0,
        u64::from(CELLS) * 4,
    );
    let mut developed = 0;
    for (cell, &id) in ids.iter().enumerate() {
        if id == 0 {
            assert_eq!(
                counts[cell], old_counts[cell],
                "unmutated cells keep their history"
            );
            continue;
        }
        assert!(counts[cell] < caps[cell],
            "default descendant with {} inherited splits is sterile after mutation: count={}, cap={}",
            old_counts[cell], counts[cell], caps[cell]);
        if old_counts[cell] < caps[cell] {
            assert_eq!(
                counts[cell], old_counts[cell],
                "a usable split budget must not reset"
            );
        }
        if old_counts[cell] < 5 || developed >= 4 {
            continue;
        }
        let meta: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.system.genome_meta_buffer,
            u64::from(id) * 16,
            16,
        );
        let base = meta[1];
        let props = |index: usize| -> Vec<[f32; 4]> {
            read(
                &fixture.device,
                &fixture.queue,
                &fixture.buffers[index],
                u64::from(base) * 16,
                u64::from(meta[0]) * 16,
            )
        };
        let p0 = props(0);
        let p1 = props(1);
        let p2 = props(2);
        let p3 = props(3);
        let p4 = props(4);
        let routes: Vec<[i32; 2]> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.buffers[10],
            u64::from(base) * 8,
            u64::from(meta[0]) * 8,
        );
        let flags: Vec<Vec<u32>> = [12, 13, 14, 30, 31]
            .iter()
            .map(|&index| {
                read(
                    &fixture.device,
                    &fixture.queue,
                    &fixture.buffers[index],
                    u64::from(base) * 4,
                    u64::from(meta[0]) * 4,
                )
            })
            .collect();
        let mut mutant = source.clone();
        for (index, mode) in mutant.modes.iter_mut().enumerate() {
            mode.split_interval = p0[index][3];
            mode.split_mass = p1[index][0];
            mode.max_splits = p2[index][0] as i32;
            mode.min_adhesions = p3[index][3] as i32;
            mode.max_adhesions = p4[index][0] as i32;
            mode.mode_a_after_splits = p4[index][1] as i32;
            mode.mode_b_after_splits = p4[index][2] as i32;
            mode.child_a.mode_number = routes[index][0] - base as i32;
            mode.child_b.mode_number = routes[index][1] - base as i32;
            mode.parent_make_adhesion = flags[0][index] != 0;
            mode.child_a.keep_adhesion = flags[1][index] != 0;
            mode.child_b.keep_adhesion = flags[2][index] != 0;
            mode.child_a_after_split_keep_adhesion = flags[3][index] != 0;
            mode.child_b_after_split_keep_adhesion = flags[4][index] != 0;
        }
        let mut state =
            InitialState::from_genome(&mutant, 128, &PhysicsConfig::default()).to_canonical_state();
        state.split_counts[0] = counts[cell] as i32;
        for generation in 1..=8 {
            for index in 0..state.cell_count {
                state.nutrients[index] = 200.0;
            }
            division_step(&mut state, &mutant, generation as f32 * 10.0, 128, 42);
        }
        assert!(
            state.cell_count >= 8,
            "mutants must actually develop beyond a pair"
        );
        let mut components: Vec<_> = (0..state.cell_count).collect();
        for _ in 0..state.cell_count {
            for bond in 0..state.adhesion_connections.active_count {
                if state.adhesion_connections.is_active[bond] == 0 {
                    continue;
                }
                let a = state.adhesion_connections.cell_a_index[bond] as usize;
                let b = state.adhesion_connections.cell_b_index[bond] as usize;
                let component = components[a].min(components[b]);
                components[a] = component;
                components[b] = component;
            }
        }
        let largest = (0..state.cell_count)
            .map(|label| components.iter().filter(|&&c| c == label).count())
            .max()
            .unwrap();
        assert!(
            largest >= 4,
            "default mutant only formed a {largest}-cell body despite {} cells",
            state.cell_count
        );
        developed += 1;
    }
    assert_eq!(
        developed, 4,
        "must test several independently mutated older cells"
    );
}

#[test]
fn mutation_rebases_only_newly_exhausted_split_budgets() {
    let mut fixture = mutation_fixture(1);
    for (frame, (previous_cap, count, new_cap, expected)) in [
        (u32::MAX, 32u32, 5u32, 0u32),
        (10, 7, 5, 0),
        (5, 3, 5, 3),
        (5, 5, 5, 5),
        (2, 1, 2, 1),
        (2, 2, 2, 2),
        (5, 5, 2, 5),
    ]
    .into_iter()
    .enumerate()
    {
        for buffer in [&fixture.genomes, &fixture.modes] {
            fixture.queue.write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(&vec![0u32; CELLS as usize]),
            );
        }
        fixture.queue.write_buffer(
            &fixture.runtime[2],
            0,
            bytemuck::cast_slice(&vec![previous_cap; CELLS as usize]),
        );
        fixture.queue.write_buffer(
            &fixture.split_counts,
            0,
            bytemuck::cast_slice(&vec![count; CELLS as usize]),
        );
        fixture.mutate(
            MutationParamEntry {
                buffer_id: buffer_id::MODE_PROPERTIES,
                element_offset: 8,
                data_type: data_type::CONTINUOUS_F32,
                weight: 1.0,
                min_delta: 0.0,
                max_delta: 0.0,
                min_value: new_cap as f32,
                max_value: new_cap as f32,
            },
            frame as u32,
        );
        let ids: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.genomes,
            0,
            u64::from(CELLS) * 4,
        );
        let counts: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.split_counts,
            0,
            u64::from(CELLS) * 4,
        );
        let caps: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.runtime[2],
            0,
            u64::from(CELLS) * 4,
        );
        assert!(ids.iter().any(|&id| id != 0));
        for (cell, &id) in ids.iter().enumerate() {
            assert_eq!(counts[cell], if id == 0 { count } else { expected });
            assert_eq!(caps[cell], if id == 0 { previous_cap } else { new_cap });
        }
    }
}

#[test]
fn established_cycle_partners_can_still_release_unbonded_offspring() {
    let mut fixture = mutation_fixture(2);
    fixture.queue.write_buffer(
        &fixture.buffers[10],
        0,
        bytemuck::cast_slice(&[[1i32, 0], [0, 1]]),
    );
    fixture.mutate(
        MutationParamEntry {
            buffer_id: buffer_id::MODE_VISUALS,
            element_offset: 0,
            data_type: data_type::CONTINUOUS_F32,
            weight: 1.0,
            min_delta: 0.01,
            max_delta: 0.01,
            min_value: 0.0,
            max_value: 1.0,
        },
        0,
    );
    let ids: Vec<u32> = read(
        &fixture.device,
        &fixture.queue,
        &fixture.genomes,
        0,
        u64::from(CELLS) * 4,
    );
    assert!(ids.iter().any(|&id| id != 0));
    for id in ids.into_iter().filter(|&id| id != 0) {
        let meta: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.system.genome_meta_buffer,
            u64::from(id) * 16,
            16,
        );
        let flags: Vec<u32> = read(
            &fixture.device,
            &fixture.queue,
            &fixture.buffers[12],
            u64::from(meta[1] + 1) * 4,
            4,
        );
        assert_eq!(
            flags[0], 0,
            "established downstream release behavior must remain evolvable"
        );
    }
}
