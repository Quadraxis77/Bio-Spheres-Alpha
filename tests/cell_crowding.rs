use wgpu::util::DeviceExt;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl Gpu {
    fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 32,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .await
                .unwrap();
            Self { device, queue }
        })
    }
    fn buffer<T: bytemuck::Pod>(&self, data: &[T], uniform: bool) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: (if uniform {
                    wgpu::BufferUsages::UNIFORM
                } else {
                    wgpu::BufferUsages::STORAGE
                }) | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }
    fn pipeline(&self, source: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    }
    fn bind(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group: u32,
        buffers: &[(u32, &wgpu::Buffer)],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(group),
            entries: &buffers
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        })
    }
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        groups: &[wgpu::BindGroup],
        workgroups: u32,
    ) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
    }
    fn read<T: bytemuck::Pod>(&self, source: &wgpu::Buffer) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
    }
}

fn physics_params(count: usize) -> [f32; 16] {
    [
        0.1,
        1.0,
        0.0,
        f32::from_bits(count as u32),
        64.0,
        0.0,
        0.0,
        1.0,
        f32::from_bits(8),
        8.0,
        f32::from_bits(16),
        0.0,
        f32::from_bits(count as u32),
        0.0,
        0.94,
        0.0,
    ]
}

#[test]
fn only_extreme_interpenetration_is_culled_including_across_bucket_edges() {
    let gpu = Gpu::new();
    let pipeline = gpu.pipeline(include_str!("../shaders/overcrowding_cull.wgsl"), "main");
    let touching: Vec<[f32; 4]> = (0..64)
        .map(|i| {
            [
                1.0 + (i % 4) as f32 * 1.01,
                1.0 + ((i / 4) % 4) as f32 * 1.01,
                1.0 + (i / 16) as f32 * 1.01,
                0.5,
            ]
        })
        .collect();
    let boundary: Vec<[f32; 4]> = (0..128)
        .map(|i| {
            [
                if i & 1 == 0 { -0.02 } else { 0.02 },
                if i & 2 == 0 { -0.02 } else { 0.02 },
                if i & 4 == 0 { -0.02 } else { 0.02 },
                1.0,
            ]
        })
        .collect();
    // A small overlapping colony and distant cells sharing a large bucket
    // must not be mistaken for a huge local clump via raw occupancy estimates.
    let mut mixed = vec![[1.0, 1.0, 1.0, 0.5]; 12];
    mixed.extend(
        touching
            .iter()
            .map(|p| [p[0] + 2.0, p[1] + 2.0, p[2] + 2.0, p[3]]),
    );
    for (name, positions, should_cull) in [
        ("touching colony", touching, false),
        ("small bonded core", vec![[2.0, 2.0, 2.0, 1.0]; 12], false),
        ("mixed bucket", mixed, false),
        ("collapsed core", vec![[2.0, 2.0, 2.0, 1.0]; 128], true),
        ("bucket boundary", boundary, true),
    ] {
        let count = positions.len();
        let mut counts = vec![0u32; 8 * 8 * 8];
        let mut cells = vec![u32::MAX; counts.len() * 16];
        let mut indices = Vec::new();
        for (index, pos) in positions.iter().enumerate() {
            let coord = |axis: usize| ((pos[axis] + 32.0) / 8.0) as usize;
            let grid = coord(0) + coord(1) * 8 + coord(2) * 64;
            if counts[grid] < 16 {
                cells[grid * 16 + counts[grid] as usize] = index as u32;
            }
            counts[grid] += 1;
            indices.push(grid as u32);
        }
        let params = gpu.buffer(&physics_params(count), true);
        let positions = gpu.buffer(&positions, false);
        let cell_count = gpu.buffer(&[count as u32, count as u32, u32::MAX], false);
        let deaths = gpu.buffer(&vec![0u32; count], false);
        let counts = gpu.buffer(&counts, false);
        let indices = gpu.buffer(&indices, false);
        let cells = gpu.buffer(&cells, false);
        let groups = [
            gpu.bind(
                &pipeline,
                0,
                &[(0, &params), (1, &positions), (5, &cell_count)],
            ),
            gpu.bind(&pipeline, 1, &[(0, &deaths)]),
            gpu.bind(&pipeline, 2, &[(0, &counts), (2, &indices), (3, &cells)]),
        ];
        gpu.dispatch(&pipeline, &groups, (count as u32).div_ceil(256));
        let flags = gpu.read::<u32>(&deaths);
        let survivors = flags.iter().filter(|flag| **flag == 0).count();
        if should_cull {
            assert!(
                (8..=24).contains(&survivors),
                "{name}: {survivors} survivors"
            );
        } else {
            assert_eq!(survivors, count, "{name} must survive");
        }
        assert_eq!(
            gpu.read::<u32>(&cell_count)[1],
            count as u32,
            "lifecycle owns live-count decrements"
        );
        assert!(
            gpu.read::<[f32; 4]>(&positions).iter().all(|p| p[3] >= 0.5),
            "retain mass until lifecycle reclaims the slot"
        );
    }
}

#[test]
fn same_organism_cores_separate_but_normal_bonds_do_not_repulse() {
    let gpu = Gpu::new();
    let positions: [[f32; 4]; 10] = [
        [0.0, 0.0, 0.0, 1.0],
        [1.5, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.5, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 1.0],
    ];
    let mut velocities = [[0.0f32; 4]; 10];
    velocities[8][0] = -1000.0;
    velocities[9][0] = 1000.0;
    let positions = gpu.buffer(&positions, false);
    let velocities = gpu.buffer(&velocities, false);
    let labels = gpu.buffer(&[1u32, 1, 1, 1, 1, 1, 2, 3, 1, 1], false);
    let stiffness = gpu.buffer(&[100.0f32; 10], false);
    let angular = gpu.buffer(&[[0.0f32; 4]; 10], false);
    for function in ["resolve_cell_pair", "resolve_cell_pair_dense"] {
        let shader = format!("{}\n@compute @workgroup_size(1) fn test_pair(@builtin(global_invocation_id) id: vec3<u32>) {{ {function}(id.x * 2u, id.x * 2u + 1u); }}", include_str!("../shaders/collision_detection.wgsl"));
        let pipeline = gpu.pipeline(&shader, "test_pair");
        let forces: Vec<_> = (0..6).map(|_| gpu.buffer(&[0i32; 10], false)).collect();
        let mut accum: Vec<_> = (0..3).map(|i| (i as u32, &forces[i])).collect();
        if function == "resolve_cell_pair" {
            accum.extend((3..6).map(|i| (i as u32, &forces[i])));
            accum.push((7, &angular));
        }
        let groups = [
            gpu.bind(&pipeline, 0, &[(1, &positions), (2, &velocities)]),
            gpu.bind(&pipeline, 1, &[(4, &stiffness), (5, &labels)]),
            gpu.bind(&pipeline, 2, &accum),
        ];
        gpu.dispatch(&pipeline, &groups, 5);
        let force: Vec<Vec<i32>> = forces.iter().take(3).map(|b| gpu.read(b)).collect();
        for axis in &force {
            assert_eq!(
                (axis[0], axis[1]),
                (0, 0),
                "normal bonded overlap is allowed"
            );
            assert_eq!(
                (axis[8], axis[9]),
                (0, 0),
                "separating cells must not be pulled together by damping"
            );
            assert_eq!(
                axis[4], -axis[5],
                "coincident forces conserve pair momentum"
            );
            assert_ne!(
                axis[4], 0,
                "coincident cells need separation in all three axes"
            );
        }
        assert!(
            force[0][2] < 0 && force[0][3] > 0,
            "collapsed organism receives outward force"
        );
        assert!(
            force[0][6] < 0 && force[0][7] > 0,
            "unrelated cells retain normal surface collisions"
        );
    }
}

#[test]
fn ice_entombment_releases_on_thaw_and_surface_contacts_can_slide_or_escape() {
    let gpu = Gpu::new();
    let pipeline = gpu.pipeline(include_str!("../shaders/position_update.wgsl"), "main");
    let positions = gpu.buffer(&[[0.2f32, 0.2, 0.2, 1.0], [-0.05, 0.2, 0.2, 1.0]], false);
    let velocities = gpu.buffer(&[[-1.0f32, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]], false);
    let positions_out = gpu.buffer(&[[0.0f32; 4]; 2], false);
    let velocities_out = gpu.buffer(&[[0.0f32; 4]; 2], false);
    let params = gpu.buffer(&physics_params(2), true);
    let count = gpu.buffer(&[2u32, 2, u32::MAX], false);
    let rotations = gpu.buffer(&[[0.0f32, 0.0, 0.0, 1.0]; 2], false);
    let rotations_out = gpu.buffer(&[[0.0f32; 4]; 2], false);
    let forces: Vec<_> = (0..3).map(|_| gpu.buffer(&[0i32; 2], false)).collect();
    let previous = gpu.buffer(&[[0.0f32; 4]; 2], false);
    let water_params = gpu.buffer(
        &[f32::from_bits(128), 1.0, -64.0, -64.0, -64.0, 1.0, 0.0, 0.0],
        true,
    );
    let water = gpu.buffer(&vec![0u32; 4 * 128 * 128], false);
    let ice = gpu.buffer(&vec![u32::MAX; 4 * 128 * 128], false);
    let grip = gpu.buffer(&[0.0f32; 2], false);
    let torques: Vec<_> = (0..3).map(|_| gpu.buffer(&[0i32; 2], false)).collect();
    let angular = gpu.buffer(&[[0.0f32; 4]; 2], false);
    let groups = [
        gpu.bind(
            &pipeline,
            0,
            &[
                (0, &params),
                (1, &positions),
                (2, &velocities),
                (3, &positions_out),
                (4, &velocities_out),
                (5, &count),
            ],
        ),
        gpu.bind(&pipeline, 1, &[(0, &rotations), (1, &rotations_out)]),
        gpu.bind(
            &pipeline,
            2,
            &[
                (0, &forces[0]),
                (1, &forces[1]),
                (2, &forces[2]),
                (3, &previous),
                (4, &water_params),
                (5, &water),
                (7, &grip),
                (8, &ice),
                (9, &torques[0]),
                (10, &torques[1]),
                (11, &torques[2]),
                (12, &angular),
            ],
        ),
    ];
    gpu.dispatch(&pipeline, &groups, 1);
    assert_eq!(
        gpu.read::<[f32; 4]>(&positions_out),
        gpu.read::<[f32; 4]>(&positions),
        "fully entombed cells stay fixed"
    );
    assert_eq!(gpu.read::<[f32; 4]>(&velocities_out), vec![[0.0; 4]; 2]);
    assert_eq!(gpu.read::<[f32; 4]>(&previous), vec![[0.0; 4]; 2]);

    gpu.queue
        .write_buffer(&ice, 0, bytemuck::cast_slice(&vec![0u32; 4 * 128 * 128]));
    gpu.queue
        .write_buffer(&velocities, 0, bytemuck::cast_slice(&[[0.0f32; 4]; 2]));
    gpu.queue
        .write_buffer(&forces[0], 0, bytemuck::cast_slice(&[10000i32; 2]));
    gpu.dispatch(&pipeline, &groups, 1);
    assert!(
        gpu.read::<[f32; 4]>(&positions_out)[0][0] > 0.2,
        "collision forces resume immediately after ice disappears"
    );

    // Ice fills the x >= 0 half-space. One exposed cell moves out of the skin;
    // another approaches from open space with tangential velocity.
    let mut bits = vec![0u32; 4 * 128 * 128];
    for z in 0..128 {
        for y in 0..128 {
            for x in 2..4 {
                bits[x + y * 4 + z * 4 * 128] = u32::MAX;
            }
        }
    }
    gpu.queue.write_buffer(&ice, 0, bytemuck::cast_slice(&bits));
    gpu.queue.write_buffer(
        &velocities,
        0,
        bytemuck::cast_slice(&[[-1.0f32, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]),
    );
    gpu.queue
        .write_buffer(&forces[0], 0, bytemuck::cast_slice(&[0i32; 2]));
    gpu.queue
        .write_buffer(&previous, 0, bytemuck::cast_slice(&[[0.0f32; 4]; 2]));
    gpu.dispatch(&pipeline, &groups, 1);
    let final_pos = gpu.read::<[f32; 4]>(&positions_out);
    assert!(
        final_pos[0][0] < 0.2 && final_pos[0][2] > 0.2,
        "partly exposed cell must escape: {final_pos:?}"
    );
    assert!(
        final_pos[1][0] < 0.0 && final_pos[1][2] > 0.2,
        "ice contact must preserve sliding: {final_pos:?}"
    );
}
