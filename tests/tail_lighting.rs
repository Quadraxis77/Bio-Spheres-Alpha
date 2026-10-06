use bio_spheres::rendering::TailRenderer;
use bio_spheres::simulation::gpu_physics::light_field::ShadowFieldParams;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

#[test]
fn tails_follow_scene_light_direction_color_and_shadow() {
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
        // Validate the real preview/GPU vertex pipelines and shared shadow layout.
        let _renderer = TailRenderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, 1);
        // Exercise the production fragment function directly with a deliberately
        // non-unit interpolated normal, independent of tail mesh coverage.
        let source = include_str!("../shaders/cells/tail_gpu.wgsl")
            .replace("@fragment\r\n", "")
            .replace("@fragment\n", "")
            .replace("-> @location(0) vec4<f32>", "-> vec4<f32>");
        let source = format!(
            "{source}\n{}",
            r#"
@group(2) @binding(0) var<storage, read_write> result: vec4<f32>;
@compute @workgroup_size(1)
fn test_light() {
    var input: VertexOutput;
    input.world_normal = vec3<f32>(0.0, 0.5, 0.0);
    input.world_position = vec3<f32>(0.0);
    input.color = vec4<f32>(1.0);
    input.cell_radius = 0.1;
    result = fs_main(input);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tail lighting regression"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("test_light"),
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
        let light = uniform(bytemuck::cast_slice(&[0f32, -1., 0., 0., 1., 1., 1., 0.]));
        let mut params = ShadowFieldParams::zeroed();
        params.grid_resolution = 4;
        params.cell_size = 1.;
        params.grid_origin_x = -2.;
        params.grid_origin_y = -2.;
        params.grid_origin_z = -2.;
        params.shadow_strength = 1.;
        params.shadow_enabled = 1;
        let shadow_params = uniform(bytemuck::bytes_of(&params));
        let texture = || {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let shadow = texture();
        let color = texture();
        let upload = |texture: &wgpu::Texture, value: [u8; 4]| {
            queue.write_texture(
                texture.as_image_copy(),
                &value,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            )
        };
        let shadow_view = shadow.create_view(&Default::default());
        let color_view = color.create_view(&Default::default());
        let sampler = device.create_sampler(&Default::default());
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = [
            vec![wgpu::BindGroupEntry {
                binding: 1,
                resource: light.as_entire_binding(),
            }],
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        ];
        let groups: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, entries)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(i as u32),
                    entries,
                })
            })
            .collect();
        for (visibility, direction, enabled, expected) in [
            (255, -1f32, 1, [1., 0., 0., 1.]), // local red light, normalized normal
            (0, -1., 1, [0., 0., 0., 1.]),   // no artificial ambient in shadow
            (255, 1., 1, [0., 0., 0., 1.]),  // moving sun reverses lit side
            (0, -1., 0, [1., 1., 1., 1.]),   // disabled field uses uniform light
        ] {
            upload(&shadow, [visibility, 0, 0, 255]);
            upload(&color, [255, 0, 0, 255]);
            queue.write_buffer(&light, 4, bytemuck::bytes_of(&direction));
            params.shadow_enabled = enabled;
            queue.write_buffer(&shadow_params, 0, bytemuck::bytes_of(&params));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                for (i, group) in groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, group, &[]);
                }
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 16);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range();
            let actual: &[f32] = bytemuck::cast_slice(&mapped);
            for (a, e) in actual.iter().zip(expected) {
                assert!((a - e).abs() < 0.001, "{actual:?} != {expected:?}");
            }
            drop(mapped);
            staging.unmap();
        }
    });
}
