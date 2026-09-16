use bio_spheres::rendering::cave_system::CaveParams;
use wgpu::util::DeviceExt;

#[test]
fn gpu_cave_contacts_use_sphere_radius() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(), ..Default::default()
        }).await.unwrap();
        let source = format!("{}\n{}", include_str!("../shaders/cave_collision.wgsl"), r#"
@group(0) @binding(0) var<storage, read> samples: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> contacts: array<vec4<f32>>;
@compute @workgroup_size(1)
fn test_contact(@builtin(global_invocation_id) id: vec3<u32>) {
    let sphere = samples[id.x];
    contacts[id.x] = cave_sphere_contact(sphere.xyz, sphere.w);
}
"#);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cave sphere regression"), source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &module, entry_point: Some("test_contact"),
            compilation_options: Default::default(), cache: None,
        });
        let mut params = CaveParams::default();
        params.world_radius = 5.0;
        params.grid_resolution = 32;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
        });
        // Plane, oblique wall, and a thin opaque vent-like slab.
        for (normal, slope, slab) in [
            (glam::Vec3::X, 0.1, false),
            (glam::Vec3::new(1.0, 1.0, 1.0).normalize(), 0.02, false),
            (glam::Vec3::X, 0.1, true),
        ] {
            let mut density = Vec::new();
            for z in 0..=32 { for y in 0..=32 { for x in 0..=32 {
                let p = glam::Vec3::new(x as f32, y as f32, z as f32) * 0.5 - glam::Vec3::splat(8.0);
                let d = p.dot(normal);
                density.push(params.threshold + slope * if slab { d.min(1.0 - d) } else { d });
            }}}
            let mut inputs = Vec::new();
            for radius in [0.5f32, 1.0, 2.0] {
                for gap in [0.25f32, -0.15] {
                    let p = -normal * (radius + gap);
                    inputs.push([p.x, p.y, p.z, radius]);
                }
            }
            let buffer = |bytes: &[u8], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytes, usage,
            });
            let input = buffer(bytemuck::cast_slice(&inputs), wgpu::BufferUsages::STORAGE);
            let grid = buffer(bytemuck::cast_slice(&density), wgpu::BufferUsages::STORAGE);
            let output = buffer(&vec![0u8; inputs.len()*16], wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: output.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = |index, buffers: [&wgpu::Buffer; 2]| device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &pipeline.get_bind_group_layout(index), entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buffers[1].as_entire_binding() },
                ],
            });
            let cells = bind(0, [&input, &output]);
            let cave = bind(1, [&uniform, &grid]);
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &cells, &[]);
                pass.set_bind_group(1, &cave, &[]);
                pass.dispatch_workgroups(inputs.len() as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = readback.slice(..).get_mapped_range();
            let contacts: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            for (i, contact) in contacts.iter().enumerate() {
                if i % 2 == 0 {
                    assert_eq!(contact[3], 0.0, "separated sphere must not be pushed away");
                } else {
                    assert!((contact[3] - 0.15).abs() < 0.015, "surface overlap: {contact:?}, input {:?}", inputs[i]);
                    assert!(glam::Vec3::new(contact[0], contact[1], contact[2]).dot(-normal) > 0.99);
                }
            }
        }
    });
}
