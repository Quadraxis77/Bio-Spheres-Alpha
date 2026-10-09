use super::*;

#[test]
fn photocytes_and_embryocytes_follow_sun_without_shadows() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: 64,
        height: 64,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    let mut renderer = CellRenderer::new(&device, &queue, &config, 1);
    renderer.light_dir = [0.0, 0.0, -1.0];
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&Default::default());
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64 * 64 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut render = |cell_type: CellType, lod: f32, light: [f32; 3], emissive: f32| {
        renderer.set_light_color(light);
        renderer.update_lighting(&queue, 0.0);
        renderer.update_camera(
            &queue,
            Vec3::new(0.0, 0.0, 4.0),
            Quat::IDENTITY,
            0.0,
            lod,
            1.0,
            1.0,
            1.0,
            60.0,
        );
        let cell = CellInstance {
            position: [0.0; 3],
            radius: 1.0,
            color: [0.4, 0.8, 0.2, 1.0],
            visual_params: [0.3, 20.0, 0.2, emissive],
            rotation: [0.0, 0.0, 0.0, 1.0],
            type_data: [5.0, 0.1, 0.1, 0.1, 0.6, 1.0, 0.0, cell_type as u32 as f32],
        };
        queue.write_buffer(&renderer.instance_buffer, 0, bytemuck::bytes_of(&cell));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(renderer.type_registry.get_pipeline(cell_type));
            pass.set_bind_group(0, &renderer.bind_group, &[]);
            pass.set_bind_group(1, renderer.shadow_bind_group.as_ref().unwrap(), &[]);
            pass.set_vertex_buffer(0, renderer.instance_buffer.slice(..));
            pass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        output
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = output.slice(..).get_mapped_range().to_vec();
        output.unmap();
        bytes
    };
    for cell_type in [CellType::Photocyte, CellType::Embryocyte] {
        for lod in [0.0, 2000.0] {
            let lit = render(cell_type, lod, [1.0; 3], 0.0);
            assert!(
                lit.chunks_exact(4).filter(|p| p[1] > 20).count() > 100,
                "cell must actually render"
            );
            let dark = render(cell_type, lod, [0.0; 3], 0.0);
            assert!(
                dark.chunks_exact(4).all(|p| p[..3] == [0, 0, 0]),
                "{cell_type:?} LOD {lod} glows with zero light"
            );
            let red = render(cell_type, lod, [1.0, 0.0, 0.0], 0.0);
            assert!(
                red.chunks_exact(4).all(|p| p[1] == 0 && p[2] == 0),
                "cell must follow sun colour"
            );
            let emissive = render(cell_type, lod, [0.0; 3], 1.0);
            assert!(
                emissive.chunks_exact(4).any(|p| p[1] > 20),
                "intentional emission must survive darkness"
            );
        }
    }
}
