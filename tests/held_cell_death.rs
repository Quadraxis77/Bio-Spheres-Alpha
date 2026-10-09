use wgpu::util::DeviceExt;

// Exercise the production drag shader with stale triple-buffer masses, then
// recycle the same slot while the controller is still holding the old target.
#[test]
fn held_cell_death_cannot_resurrect_or_drag_a_recycled_slot() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Held cell death regression"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/update_position.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |bytes: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mut params = [0u32; 16];
        params[4] = 1000.0f32.to_bits();
        params[12] = 1;
        let params = buffer(bytemuck::cast_slice(&params), wgpu::BufferUsages::UNIFORM);
        let positions: Vec<_> = (0..3)
            .map(|i| {
                buffer(
                    bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, if i == 2 { 0.0 } else { 1.0 }]),
                    storage,
                )
            })
            .collect();
        let velocities: Vec<_> = (0..3)
            .map(|_| buffer(bytemuck::cast_slice(&[1.0f32; 4]), storage))
            .collect();
        let counts = buffer(bytemuck::cast_slice(&[1u32, 0, 0, 0]), storage);
        let deaths = buffer(bytemuck::cast_slice(&[1u32]), storage);
        let update = buffer(
            bytemuck::cast_slice(&[
                0u32,
                1,
                0,
                0,
                20.0f32.to_bits(),
                30.0f32.to_bits(),
                40.0f32.to_bits(),
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        let bind = |index, buffers: Vec<&wgpu::Buffer>| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(index),
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let main = bind(
            0,
            vec![
                &params,
                &positions[0],
                &positions[1],
                &positions[2],
                &velocities[0],
                &velocities[1],
                &velocities[2],
                &counts,
            ],
        );
        let update_bind = bind(1, vec![&update]);
        let death_bind = bind(2, vec![&deaths]);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 112,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatch = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &main, &[]);
                pass.set_bind_group(1, &update_bind, &[]);
                pass.set_bind_group(2, &death_bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            for i in 0..3 {
                encoder.copy_buffer_to_buffer(&positions[i], 0, &readback, i as u64 * 16, 16);
                encoder.copy_buffer_to_buffer(&velocities[i], 0, &readback, 48 + i as u64 * 16, 16);
            }
            encoder.copy_buffer_to_buffer(&counts, 0, &readback, 96, 16);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let words =
                bytemuck::cast_slice::<u8, u32>(&readback.slice(..).get_mapped_range()).to_vec();
            readback.unmap();
            words
        };
        let dead = dispatch();
        assert_eq!(
            f32::from_bits(dead[11]),
            0.0,
            "newest dead position must stay dead"
        );
        assert_eq!(dead[26], u32::MAX, "death releases the GPU hold");
        for i in 0..2 {
            assert_eq!(
                f32::from_bits(dead[i * 4]),
                1.0,
                "older buffers must not be dragged"
            );
        }

        // The lifecycle can now reuse this index. A held old trigger must not
        // move this newborn, even after its death flag has been cleared.
        queue.write_buffer(&deaths, 0, bytemuck::cast_slice(&[0u32]));
        queue.write_buffer(&counts, 0, bytemuck::cast_slice(&[1u32, 1, u32::MAX, 0]));
        for position in &positions {
            queue.write_buffer(position, 0, bytemuck::cast_slice(&[5.0f32, 6.0, 7.0, 0.6]));
        }
        let recycled = dispatch();
        for i in 0..3 {
            assert_eq!(
                f32::from_bits(recycled[i * 4]),
                5.0,
                "old hold must not move newborn"
            );
            assert_eq!(f32::from_bits(recycled[i * 4 + 3]), 0.6);
        }

        // A fresh explicit hold of the live cell still moves all three buffers.
        queue.write_buffer(&counts, 0, bytemuck::cast_slice(&[1u32, 1, 0, 0]));
        let live = dispatch();
        for i in 0..3 {
            let actual: Vec<_> = live[i * 4..i * 4 + 4]
                .iter()
                .map(|w| f32::from_bits(*w))
                .collect();
            assert_eq!(actual, vec![20.0, 30.0, 40.0, 0.6]);
            assert_eq!(&live[12 + i * 4..16 + i * 4], &[0u32; 4]);
        }
    });
}
