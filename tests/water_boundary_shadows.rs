use bio_spheres::simulation::gpu_physics::light_field::ShadowFieldParams;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

#[test]
fn water_and_ice_have_no_axis_shadows_outside_the_spherical_light_field() {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        const RES: u32 = 64;
        const PANEL: u32 = 64;
        const SAMPLES: u32 = 16 + 6 * PANEL * PANEL;
        let mut params = ShadowFieldParams::zeroed();
        params.grid_resolution = RES;
        params.cell_size = 1.0;
        params.grid_origin_x = -32.0;
        params.grid_origin_y = -32.0;
        params.grid_origin_z = -32.0;
        params.shadow_enabled = 1;
        params.shadow_strength = 1.0;
        params.shadow_quality = 0.8;
        let sun = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
        params.light_dir_x = sun.x;
        params.light_dir_y = sun.y;
        params.light_dir_z = sun.z;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dark world with sunlit exterior"),
            size: wgpu::Extent3d {
                width: RES,
                height: RES,
                depth_or_array_layers: RES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Match the real field's convention: outside the sphere is open sky;
        // inside may be dark. At each cube-face center the last texel is inside
        // the sphere, so clamping an exterior sample creates six dark disks.
        let mut texels = Vec::<u16>::new();
        for z in 0..RES {
            for y in 0..RES {
                for x in 0..RES {
                    let r2: f32 = [x, y, z]
                        .into_iter()
                        .map(|v| (v as f32 + 0.5 - 32.0).powi(2))
                        .sum();
                    let light = if r2 >= 1024.0 { 0x3c00 } else { 0 }; // f16: 1 or 0
                    texels.extend([light, 0, 0, 0x3c00]);
                }
            }
        }
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(RES * 8),
                rows_per_image: Some(RES),
            },
            texture.size(),
        );
        for (name, production) in [
            ("water", include_str!("../shaders/fluid/density_mesh.wgsl")),
            ("ice", include_str!("../shaders/fluid/ice_mesh.wgsl")),
        ] {
            let source = format!(
                "{production}\n{}",
                r#"
@group(2) @binding(2) var<storage, read_write> shadow_results: array<f32>;
fn probe_axis(face: u32) -> vec3<f32> {
    let axis = face / 2u;
    let sign = select(1.0, -1.0, (face & 1u) != 0u);
    return vec3<f32>(select(0.0, sign, axis == 0u),
        select(0.0, sign, axis == 1u), select(0.0, sign, axis == 2u));
}
@compute @workgroup_size(64)
fn probe_boundary_shadows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= 24592u { return; }
    var position = vec3<f32>(0.0);
    if i < 6u {
        position = probe_axis(i) * 33.0;
    } else if i < 14u {
        let corner = i - 6u;
        position = normalize(vec3<f32>(
            select(-1.0, 1.0, (corner & 1u) != 0u),
            select(-1.0, 1.0, (corner & 2u) != 0u),
            select(-1.0, 1.0, (corner & 4u) != 0u))) * 33.0;
    } else if i == 15u {
        // A valid near-boundary sample must still see actual occlusion.
        shadow_results[i] = sample_light_field(vec3<f32>(31.75, 0.0, 0.0));
        return;
    } else if i >= 16u {
        let pixel = i - 16u;
        let axis = probe_axis(pixel / 4096u);
        let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
        let tangent = normalize(cross(axis, helper));
        let bitangent = cross(axis, tangent);
        let uv = (vec2<f32>(f32(pixel % 64u), f32((pixel / 64u) % 64u)) + 0.5) / 64.0 - 0.5;
        position = normalize(axis + (uv.x * tangent + uv.y * bitangent) * 0.8) * 33.0;
    }
    shadow_results[i] = sample_sun_shadow_soft(position);
}
"#
            );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("probe_boundary_shadows"),
                compilation_options: Default::default(),
                cache: None,
            });
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: SAMPLES as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[],
            });
            let shadows = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let results = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(2),
                entries: &[wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                }],
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: output.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &empty, &[]);
                pass.set_bind_group(1, &shadows, &[]);
                pass.set_bind_group(2, &results, &[]);
                pass.dispatch_workgroups(SAMPLES.div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range();
            let values: &[f32] = bytemuck::cast_slice(&mapped);
            if let Ok(folder) = std::env::var("BIOSPHERES_SHADOW_OUTPUT") {
                std::fs::create_dir_all(&folder).unwrap();
                let mut preview = image::RgbImage::new(PANEL * 6, PANEL);
                for face in 0..6 {
                    for y in 0..PANEL {
                        for x in 0..PANEL {
                            let value =
                                values[(16 + face * PANEL * PANEL + y * PANEL + x) as usize];
                            let shade = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                            preview.put_pixel(face * PANEL + x, y, image::Rgb([shade; 3]));
                        }
                    }
                }
                preview
                    .save(std::path::Path::new(&folder).join(format!("{name}.png")))
                    .unwrap();
            }
            eprintln!(
                "{name}: axis {:?}, diagonals {:?}, interior {:?}",
                &values[..6],
                &values[6..14],
                &values[14..16]
            );
            assert!(
                values[..14].iter().all(|v| *v > 0.99),
                "{name}: exterior must be unshadowed at every orientation"
            );
            assert!(
                values[14..16].iter().all(|v| *v < 0.01),
                "{name}: real interior occlusion must remain"
            );
        }
    });
}
