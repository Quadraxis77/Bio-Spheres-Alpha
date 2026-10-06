use bio_spheres::genome::Genome;
use bio_spheres::simulation::gpu_physics::gametocyte_merge::{
    GametocyteMergeSystem,
};
use wgpu::util::DeviceExt;

#[test]
fn gpu_fusion_checks_compatibility_and_claims_each_parent_once() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let system = GametocyteMergeSystem::new(&device);
        let buffer = |data: &[u32], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        for (compatible, same_parent) in [(false, false), (true, true), (true, false)] {
            let mut params = [0u32; 16];
            params[4] = 100f32.to_bits();
            params[8] = 1;
            params[10] = 64;
            params[12] = 3;
            let params = buffer(&params, wgpu::BufferUsages::UNIFORM);
            let positions = buffer(
                &[
                    0,
                    0,
                    0,
                    1f32.to_bits(),
                    0,
                    0,
                    0,
                    1f32.to_bits(),
                    0,
                    0,
                    0,
                    1f32.to_bits(),
                ],
                storage,
            );
            let count = buffer(&[3], storage);
            let types = buffer(&[13, 13, 13], storage);
            let flags = buffer(&[0, 0, 0], storage);
            let parents = buffer(
                &[
                    1,
                    0,
                    0,
                    0,
                    if same_parent { 1 } else { 2 },
                    0,
                    0,
                    0,
                    if same_parent { 1 } else { 3 },
                    0,
                    0,
                    0,
                ],
                storage,
            );
            let ids = buffer(&[0, 1, 2], storage);
            let modes = buffer(&[0, 0, 0], storage);
            let props = buffer(&[0; 4], storage);
            let reserves = buffer(&[1000; 3], storage);
            let mut genomes = vec![Genome::default(); 3];
            if !compatible {
                for (i, g) in genomes.iter_mut().enumerate() {
                    for m in &mut g.modes {
                        m.cell_type = i as i32;
                    }
                }
            }
            let mut metadata = Vec::new();
            let mut mode_types = Vec::new();
            for g in &genomes {
                metadata.extend([g.modes.len() as u32, mode_types.len() as u32, 0, u32::MAX]);
                mode_types.extend(g.modes.iter().map(|m| m.cell_type as u32));
            }
            let metadata = buffer(&metadata, storage);
            let mode_types = buffer(&mode_types, storage);
            let counts = buffer(&[3], storage);
            let mut cells = [0; 64];
            cells[1] = 1;
            cells[2] = 2;
            let cells = buffer(&cells, storage);
            let physics = system.create_physics_bind_group(
                &device, &params, &positions, &positions, &positions, &positions, &count,
            );
            let data = system.create_cell_data_bind_group(
                &device, &types, &flags, &parents, &ids, &modes, &props, &reserves, &metadata, &mode_types,
            );
            let spatial = system.create_spatial_bind_group(&device, &counts, &cells, &modes);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 48,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            system.clear_events(&mut encoder);
            system.run(&mut encoder, &physics, &data, &spatial, 3);
            encoder.copy_buffer_to_buffer(&system.merge_events_buffer, 0, &readback, 0, 36);
            encoder.copy_buffer_to_buffer(&flags, 0, &readback, 36, 12);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = readback.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            if compatible && !same_parent {
                assert_eq!(words[0], 1, "three gametes must produce only one pair");
                assert_eq!(words[8], 2000, "fusion must preserve both reserves");
                assert_eq!(words[9..].iter().filter(|&&v| v == 2).count(), 2);
                assert!(
                    !words[9..].contains(&1),
                    "fusion must not set the death-particle state"
                );
            } else {
                assert_eq!(words[0], 0, "ineligible pairs must not fuse");
                assert_eq!(
                    &words[9..],
                    &[0, 0, 0],
                    "ineligible parents must remain alive"
                );
            }
        }
    });
}

#[test]
fn photocyte_surface_exposure_preserves_sun_shadow_and_vent_light() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let source = include_str!("../shaders/photocyte_light.wgsl");
        let params_start = source.find("struct PhotocyteParams {").unwrap();
        let params_end = params_start + source[params_start..].find('}').unwrap() + 1;
        let helpers_start = source.find("fn light_index(").unwrap();
        let helpers_end = source.find("fn signal_value(").unwrap();
        let shader = format!("{}\nstruct Emission {{ r: atomic<u32>, g: atomic<u32>, b: atomic<u32>, strength: atomic<u32>, }}\nconst LUMINOCYTE_FIELD_FIXED_POINT_SCALE: f32 = 1024.0;\n@group(0) @binding(0) var<uniform> photocyte_params: PhotocyteParams;\n@group(0) @binding(1) var<storage, read> light_field: array<f32>;\n@group(0) @binding(2) var<storage, read> light_color_field: array<vec4<f32>>;\n@group(0) @binding(3) var<storage, read_write> luminocyte_emission: array<Emission>;\n@group(0) @binding(4) var<storage, read_write> results: array<f32>;\n{}\n@compute @workgroup_size(1) fn main() {{ results[0] = sample_photocyte_light(vec3<f32>(2.0, 2.0, 2.0), 0.5).x; results[1] = sample_photocyte_light(vec3<f32>(2.0, 4.0, 2.0), 0.5).x; let local = sample_photocyte_light(vec3<f32>(2.0, 6.0, 2.0), 0.5); results[2] = local.x; results[3] = local.y; }}", &source[params_start..params_end], &source[helpers_start..helpers_end]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = [
            8u32,
            1f32.to_bits(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1f32.to_bits(),
            0,
            0,
        ];
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut field = vec![0f32; 512];
        // Center is self-shadowed; the sun-facing sample is lit.
        field[3 + 2 * 8 + 2 * 64] = 1.0;
        // An entirely dark neighborhood stays dark; vent exposure remains local.
        field[2 + 6 * 8 + 2 * 64] = 3.75;
        let lights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&field),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut colors = vec![[0f32; 4]; 512];
        colors[2 + 6 * 8 + 2 * 64][3] = 3.75;
        let light_colors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_colors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &readback, 0, 16);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let data = readback.slice(..).get_mapped_range();
        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(&data),
            &[1.0, 0.0, 0.0, 3.75]
        );
    });
}
