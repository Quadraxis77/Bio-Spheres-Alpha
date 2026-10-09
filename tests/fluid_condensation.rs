use bio_spheres::simulation::fluid_simulation::gpu_simulator::GpuFluidParams;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

fn storage_buffer(device: &wgpu::Device, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fluid condensation test buffer"),
        contents,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

#[test]
fn rain_falls_through_steam_and_cool_steam_condenses_at_the_world_shell() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("a GPU adapter is required for the fluid shader regression");
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 64
                        .min(adapter_limits.max_storage_buffers_per_shader_stage),
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_buffer_size: adapter_limits.max_buffer_size,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .expect("create a device for the fluid shader regression");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid condensation regression"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fluid/fluid_sim.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid condensation regression layout"),
            entries: &(0..11)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: matches!(binding, 2 | 5 | 9 | 10),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid condensation regression pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid condensation regression pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fluid_swap"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut params = GpuFluidParams::zeroed();
        params.grid_resolution = 4;
        params.world_radius = 4.0;
        params.cell_size = 1.0;
        params.grid_origin_x = -2.0;
        params.grid_origin_y = -2.0;
        params.grid_origin_z = -2.0;
        params.gravity_mode = 1;
        params.gravity_magnitude = 50.0;
        params.lateral_flow_probability_steam = 0.0;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid condensation regression parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let steam_index = 3 + 3 * 4 + 3 * 16;
        let pool_index = 1 + 1 * 4 + 1 * 16;
        let rain_index = 1 + 3 * 4 + 1 * 16;
        let blocking_steam_index = 1 + 2 * 4 + 1 * 16;
        let mut voxels = vec![0u32; 64];
        voxels[rain_index] = (u16::MAX as u32) << 16 | 1;
        voxels[blocking_steam_index] = (u16::MAX as u32) << 16 | 3;
        let mut temperatures = vec![0u32; 64];
        temperatures[rain_index] = ((10.0 + 50.0) * 256.0) as u32;
        temperatures[blocking_steam_index] = ((10.0 + 50.0) * 256.0) as u32;
        let mut solids = vec![0u32; 64];
        for z in 0..4 {
            for x in 0..4 {
                solids[x + z * 16] = 1;
            }
        }
        solids[steam_index - 4] = 1;
        let buffers = vec![
            uniform,
            storage_buffer(&device, bytemuck::cast_slice(&voxels)),
            storage_buffer(&device, bytemuck::cast_slice(&solids)),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0u32; 64])),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0u32; 8])),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64])),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0u32; 64])),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64])),
            storage_buffer(&device, bytemuck::cast_slice(&temperatures)),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64])),
            storage_buffer(&device, bytemuck::cast_slice(&vec![0u32; 256])),
        ];
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid condensation regression bindings"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(index, buffer)| wgpu::BindGroupEntry {
                    binding: index as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });

        // Rain must displace the steam below it, reach the floor, and remain
        // liquid at a temperature below the evaporation threshold.
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut parameter_copies = Vec::with_capacity(128);
        for tick in 0..128 {
            params.time = tick as f32;
            parameter_copies.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fluid condensation tick parameters"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::COPY_SRC,
                }),
            );
            encoder.copy_buffer_to_buffer(
                &parameter_copies[tick as usize],
                0,
                &buffers[0],
                0,
                std::mem::size_of::<GpuFluidParams>() as u64,
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid condensation tick"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
        }
        queue.submit([encoder.finish()]);

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid condensation regression readback"),
            size: (voxels.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buffers[1], 0, &staging, 0, staging.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let final_voxels: &[u32] = bytemuck::cast_slice(&mapped);

        assert_eq!(
            final_voxels[pool_index],
            (u16::MAX as u32) << 16 | 1,
            "rain should displace steam and settle as liquid above the solid floor"
        );
        drop(mapped);
        staging.unmap();

        // Retain the existing outer-shell condensation regression as a second
        // run in the same GPU fixture, using radial gravity so the resulting
        // liquid does not immediately fall out of the test voxel.
        let mut voxels = vec![0u32; 64];
        voxels[steam_index] = (u16::MAX as u32) << 16 | 3;
        let mut temperatures = vec![0u32; 64];
        temperatures[steam_index] = ((10.0 + 50.0) * 256.0) as u32;
        queue.write_buffer(&buffers[1], 0, bytemuck::cast_slice(&voxels));
        queue.write_buffer(&buffers[8], 0, bytemuck::cast_slice(&temperatures));
        params.gravity_mode = 3;
        queue.write_buffer(&buffers[0], 0, bytemuck::bytes_of(&params));
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut parameter_copies = Vec::with_capacity(128);
        for tick in 0..128 {
            params.time = (tick + 128) as f32;
            parameter_copies.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fluid shell-condensation tick parameters"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::COPY_SRC,
                }),
            );
            encoder.copy_buffer_to_buffer(
                &parameter_copies[tick as usize],
                0,
                &buffers[0],
                0,
                std::mem::size_of::<GpuFluidParams>() as u64,
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid shell-condensation tick"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buffers[1], 0, &staging, 0, staging.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let final_voxels: &[u32] = bytemuck::cast_slice(&mapped);
        assert_eq!(
            final_voxels.iter().filter(|state| **state & 7 == 1).count(),
            1,
            "cool steam should condense above the blocking solid"
        );
    });
}
