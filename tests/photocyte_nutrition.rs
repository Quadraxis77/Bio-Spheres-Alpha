use bio_spheres::simulation::gpu_physics::light_field::{
    photocyte_solar_mass_rate, PhotocyteParams, DEFAULT_PHOTOCYTE_MASS_RATE,
};
use wgpu::util::DeviceExt;

fn storage(device: &wgpu::Device, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

#[test]
fn photocyte_intake_is_moderate_across_brightness_and_tick_rates() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Photocyte nutrition regression"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/photocyte_light.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = |bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let physics = uniform(&[0u8; 64]);
        let position = storage(&device, bytemuck::cast_slice(&[[6.0f32, 6.0, 6.0, 1.0]]));
        let count = storage(&device, bytemuck::cast_slice(&[1u32]));
        let params = uniform(&[0u8; 48]);
        let light = storage(&device, bytemuck::cast_slice(&[0.0f32; 64]));
        let cell_type = storage(&device, bytemuck::cast_slice(&[3u32]));
        let nutrients = storage(&device, bytemuck::cast_slice(&[20_000i32]));
        let threshold = storage(&device, bytemuck::cast_slice(&[50.0f32]));
        let dead = storage(&device, bytemuck::cast_slice(&[0u32]));
        let mode = storage(&device, bytemuck::cast_slice(&[0u32]));
        let control = storage(&device, &[0u8; 16]);
        let emissive = storage(&device, &[0u8; 16]);
        let signals = storage(&device, &[0u8; 64]);
        let color = storage(&device, &[0u8; 16]);
        let glow = storage(&device, &[0u8; 16]);
        let light_colors = storage(&device, &[0u8; 64 * 16]);
        let emission = storage(&device, &[0u8; 64 * 16]);
        let bind = |group: u32, buffers: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(group),
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
        let physics_group = bind(0, &[&physics, &position, &count]);
        let photo_group = bind(
            1,
            &[
                &params,
                &light,
                &cell_type,
                &nutrients,
                &threshold,
                &dead,
                &mode,
                &control,
                &emissive,
                &signals,
                &color,
                &glow,
                &light_colors,
                &emission,
            ],
        );
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Run the entire production shader for one simulated second. Inputs
        // are separate solar, geothermal and paid luminocyte radiance.
        let run = |[brightness, rate, sun, vent, lamp]: [f32; 5],
                   hz: u32,
                   initial: f32,
                   is_dead: bool| {
            let mut physics_words = [0u32; 16];
            physics_words[0] = (1.0 / hz as f32).to_bits();
            physics_words[3] = 1;
            physics_words[12] = 1;
            queue.write_buffer(&physics, 0, bytemuck::cast_slice(&physics_words));
            let photo_params = PhotocyteParams {
                grid_resolution: 4,
                cell_size: 4.0,
                grid_origin_x: 0.0,
                grid_origin_y: 0.0,
                grid_origin_z: 0.0,
                mass_per_second_full_light: photocyte_solar_mass_rate(rate, brightness),
                geothermal_mass_per_second_full_light: rate.max(0.0),
                min_light_threshold: 0.05,
                ambient_floor: 0.0,
                light_dir_x: 0.0,
                light_dir_y: 1.0,
                light_dir_z: 0.0,
            };
            queue.write_buffer(&params, 0, bytemuck::bytes_of(&photo_params));
            queue.write_buffer(&light, 0, bytemuck::cast_slice(&[sun + vent + lamp; 64]));
            queue.write_buffer(
                &light_colors,
                0,
                bytemuck::cast_slice(&[[0.0, 0.0, 0.0, vent + lamp]; 64]),
            );
            queue.write_buffer(
                &emission,
                0,
                bytemuck::cast_slice(&[[0u32, 0, 0, (lamp * 1024.0) as u32]; 64]),
            );
            queue.write_buffer(
                &nutrients,
                0,
                bytemuck::cast_slice(&[(initial * 1000.0) as i32]),
            );
            queue.write_buffer(&dead, 0, bytemuck::cast_slice(&[u32::from(is_dead)]));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &physics_group, &[]);
                pass.set_bind_group(1, &photo_group, &[]);
                for _ in 0..hz {
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            encoder.copy_buffer_to_buffer(&nutrients, 0, &staging, 0, 4);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let value = bytemuck::cast_slice::<u8, i32>(&staging.slice(..).get_mapped_range())[0];
            staging.unmap();
            value as f32 / 1000.0 - initial
        };
        let check = |actual: f32, expected: f32| {
            // Each fixed-point tick can truncate less than 0.001 nutrient.
            assert!(
                (actual - expected).abs() < 0.15,
                "expected {expected}, got {actual}"
            );
        };
        let rate = DEFAULT_PHOTOCYTE_MASS_RATE;
        for hz in [30, 60, 144] {
            for (brightness, expected) in [
                (-1.0, 0.0),
                (0.0, 0.0),
                (1.0, 5.0 / 9.0),
                (2.0, 20.0 / 9.0),
                (3.0, 5.0),
                (4.0, 80.0 / 9.0),
                (5.0, 125.0 / 9.0),
                (6.0, 125.0 / 9.0),
            ] {
                check(
                    run([brightness, rate, 1.0, 0.0, 0.0], hz, 20.0, false),
                    expected,
                );
            }
            // Each row: [brightness, configured rate, solar, vent, lamp],
            // starting nutrients, dead flag, expected gain after one second.
            for (input, initial, is_dead, expected) in [
                ([3.0, rate, 0.5, 0.0, 0.0], 20.0, false, 2.5),
                ([3.0, rate, 0.0, 0.0, 0.0], 20.0, false, -10.0),
                ([0.0, rate, 0.0, 3.75, 0.0], 20.0, false, 7.5),
                ([3.0, rate, 0.0, 3.75, 0.0], 20.0, false, 7.5),
                ([5.0, rate, 0.0, 3.75, 0.0], 20.0, false, 7.5),
                ([3.0, rate, 0.0, 0.0, 1.0], 20.0, false, 5.0),
                ([3.0, rate, 1.0, 3.75, 1.0], 20.0, false, 17.5),
                ([3.0, 0.012, 1.0, 0.0, 0.0], 20.0, false, 1.2),
                ([3.0, 0.0, 1.0, 0.0, 0.0], 20.0, false, 0.0),
                ([3.0, -0.1, 1.0, 0.0, 0.0], 20.0, false, 0.0),
                ([5.0, rate, 1.0, 0.0, 0.0], 99.0, false, 1.0),
                ([3.0, rate, 1.0, 0.0, 0.0], 20.0, true, 0.0),
            ] {
                check(run(input, hz, initial, is_dead), expected);
            }
        }
    });
}

#[test]
fn photocyte_defaults_match_the_calibrated_rate() {
    #[derive(serde::Deserialize)]
    struct Settings {
        photocyte_mass_per_second: f32,
    }
    for source in [
        include_str!("../light_settings.ron"),
        include_str!("../assets/defaults/main-scene/light_settings.ron"),
    ] {
        let settings: Settings = ron::from_str(source).unwrap();
        assert_eq!(
            settings.photocyte_mass_per_second,
            DEFAULT_PHOTOCYTE_MASS_RATE
        );
    }
    assert_eq!(
        bio_spheres::ui::GenomeEditorState::factory_defaults().photocyte_mass_per_second,
        DEFAULT_PHOTOCYTE_MASS_RATE,
    );
}
