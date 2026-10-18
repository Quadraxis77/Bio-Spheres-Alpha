use super::*;
use crate::rendering::cursor_spotlight::CursorSpotlight;

#[test]
fn flashlight_fog_is_visible_without_global_fog_and_stops_at_depth() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut fog = VolumetricFogRenderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, 64, 64);
    fog.enabled = false;
    let texture = |format, dimension, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: if dimension == wgpu::TextureDimension::D3 {
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                }
            } else {
                wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                }
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = texture(
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureDimension::D2,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let color_view = color.create_view(&Default::default());
    let depth = texture(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureDimension::D2,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let depth_view = depth.create_view(&Default::default());
    let field = texture(
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureDimension::D3,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let field_view = field.create_view(&Default::default());
    let water = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let sampler = device.create_sampler(&Default::default());
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64 * 64 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let camera_pos = glam::Vec3::new(0.0, 0.0, 5.0);
    let projection = crate::rendering::CameraProjection::from(60.0_f32).matrix(1.0, 0.1, 5000.0);
    let view_proj = projection * glam::Mat4::from_translation(-camera_pos);
    let mut render = |cursor, depth_value: f32| {
        fog.cursor_spotlight =
            CursorSpotlight::from_cursor(cursor, 64, 64, view_proj, camera_pos, 200.0);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(depth_value),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        fog.render(
            &mut encoder,
            &queue,
            &color_view,
            &depth_view,
            &device,
            &field_view,
            &field_view,
            &sampler,
            &water,
            view_proj,
            camera_pos,
            glam::Quat::IDENTITY,
            60.0,
            0.0,
            [0.0, -1.0, 0.0],
            1,
            400.0,
            [-200.0; 3],
            200.0,
        );
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
        let (send, receive) = std::sync::mpsc::channel();
        output
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(20)),
            })
            .unwrap();
        receive.recv().unwrap().unwrap();
        let bytes = output.slice(..).get_mapped_range().to_vec();
        output.unmap();
        bytes
    };
    let unlit = render(None, 1.0);
    let lit = render(Some((32.0, 32.0)), 1.0);
    let blocked = render(Some((32.0, 32.0)), 0.0);
    let aimed = render(Some((58.0, 32.0)), 1.0);
    queue.write_texture(
        field.as_image_copy(),
        &[0, 0, 0, 0, 0, 0x3c, 0, 0x3c],
        Default::default(),
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let unshadowed = render(Some((32.0, 32.0)), 1.0);
    let center = (32 * 64 + 32) * 4;
    assert!(
        lit[center] > unlit[center] + 3,
        "beam must scatter with global fog disabled"
    );
    assert!(
        lit[center] < unlit[center] + 20,
        "beam haze must stay subtle"
    );
    let footprint = lit
        .chunks_exact(4)
        .zip(unlit.chunks_exact(4))
        .filter(|(lit, unlit)| lit[0] > unlit[0] + 2)
        .count();
    assert!(
        footprint > 64 * 64 / 6 && footprint < 64 * 64 * 3 / 4,
        "the broad, soft beam must cover a useful area while retaining dark surroundings"
    );
    let shoulder = (32 * 64 + 47) * 4;
    assert!(
        lit[shoulder] > unlit[shoulder] && lit[shoulder] < lit[center],
        "the wide shoulder must fade softly away from the central beam"
    );
    assert!(
        aimed[(32 * 64 + 58) * 4] > unlit[center] + 3,
        "the beam must point through the hovered pixel"
    );
    assert_eq!(
        aimed[center], unlit[center],
        "moving the cursor must steer the beam away from screen center"
    );
    assert_eq!(
        lit[..4],
        unlit[..4],
        "outside the cone must keep its brightness"
    );
    assert_eq!(
        blocked, unlit,
        "foreground geometry must stop the volumetric beam"
    );
    assert_eq!(
        unshadowed, lit,
        "the flashlight beam must ignore occlusion metadata"
    );
}
