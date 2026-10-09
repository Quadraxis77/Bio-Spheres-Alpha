use super::*;
#[test]
fn tracked_models_render_stereo_with_transparent_background() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let width = 640;
    let height = 480;
    let stride = width * 4;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = Controllers::new(&device, &queue, format, width, height);
    let hand = |x| openxr::Posef {
        position: openxr::Vector3f {
            x,
            y: -0.16,
            z: -0.45,
        },
        ..openxr::Posef::IDENTITY
    };
    let input = VrInput {
        grips: [Some(hand(-0.16)), Some(hand(0.16))],
        ..Default::default()
    };
    let mut images = Vec::new();
    for eye in 0..2 {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Controller visual regression"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let projection =
            Mat4::perspective_rh(
                70.0_f32.to_radians(),
                width as f32 / height as f32,
                0.005,
                250.0,
            ) * Mat4::from_translation(Vec3::new(if eye == 0 { 0.032 } else { -0.032 }, 0.0, 0.0));
        renderer.draw(
            &device,
            &queue,
            &texture.create_view(&Default::default()),
            eye,
            projection,
            &input,
            Some((Vec3::new(0.16, -0.16, -0.5), Vec3::new(0.0, 0.0, -1.6))),
        );
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (stride * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap();
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        receive.recv().unwrap().unwrap();
        let pixels = buffer.slice(..).get_mapped_range().to_vec();
        assert_eq!(pixels[3], 0, "background must remain transparent");
        let colored = pixels.chunks_exact(4).filter(|p| p[3] > 0).count();
        assert!(
            colored > 1000 && colored < 15000,
            "controllers must be visibly drawn without filling the eye: {colored}"
        );
        let gray = pixels
            .chunks_exact(4)
            .filter(|p| {
                p[3] > 0 && p[0].abs_diff(p[1]) < 12 && p[1].abs_diff(p[2]) < 12 && p[0] > 35
            })
            .count();
        assert!(
            gray > 1000,
            "Original neutral controller materials must be visible: {gray}"
        );
        images.push(pixels);
        buffer.unmap();
    }
    assert_ne!(
        images[0], images[1],
        "controller stereo disparity must follow the individual eyes"
    );
    std::fs::create_dir_all("target/vr-visual-checks").unwrap();
    image::save_buffer(
        "target/vr-visual-checks/tracked-controllers.png",
        &images[0],
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
