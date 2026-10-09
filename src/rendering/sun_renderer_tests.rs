use super::*;
use glam::{Mat4, Quat, Vec3};

#[test]
fn sun_is_a_stereo_world_sphere_with_roll_stable_detail_and_real_occlusion() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    const SIZE: u32 = 384;
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sun test output"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sun test depth"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sun pixel readback"),
        size: (SIZE * SIZE * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let projection = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 5000.0);
    let mut sun = SunRenderer::new(&device, wgpu::TextureFormat::Rgba8Unorm);
    sun.resize(SIZE, SIZE);
    sun.orbit_world_radius = 20.0;
    sun.sun_angular_radius = 0.18;
    sun.sun_color = [1.0, 0.65, 0.25];
    let mut render = |eye: Vec3, rotation: Quat, projection: Mat4, depth_value: f32, tint: [f32; 3], intensity: f32| {
        sun.sun_color = tint;
        let vp = projection * Mat4::from_rotation_translation(rotation, eye).inverse();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear test scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
        sun.render(
            &mut encoder,
            &queue,
            &color_view,
            &depth_view,
            &device,
            vp,
            eye,
            12.0,
            [0.0, 0.0, -1.0],
            intensity,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        output
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let pixels = output.slice(..).get_mapped_range().to_vec();
        output.unmap();
        pixels
    };
    let base = render(Vec3::ZERO, Quat::IDENTITY, projection, 1.0, [1.0, 0.65, 0.25], 0.4);
    let rolled = render(
        Vec3::ZERO,
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        projection,
        1.0, [1.0, 0.65, 0.25], 0.4,
    );
    let pixel = |image: &[u8], x: usize, y: usize| -> [u8; 4] {
        image[(y * SIZE as usize + x) * 4..][..4]
            .try_into()
            .unwrap()
    };
    let mut error = 0u64;
    let mut samples = 0u64;
    let mut minimum = 255;
    let mut maximum = 0;
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let p = pixel(&base, x, y);
            let rotated = pixel(&rolled, SIZE as usize - 1 - y, x);
            // Check both textured surface and the coronal shell, skipping the AA fringe.
            if p[3] > 40 && (p[3] == 255 || p[3] < 200) {
                for c in 0..4 {
                    error += p[c].abs_diff(rotated[c]) as u64;
                    samples += 1;
                }
                if p[3] == 255 {
                    minimum = minimum.min(p[0]);
                    maximum = maximum.max(p[0]);
                }
            }
        }
    }
    assert!(samples > 10000, "sun and corona must actually render");
    assert!(
        (error as f64 / samples as f64) < 1.5,
        "surface/corona must follow world rays under head roll: mean error {}",
        error as f64 / samples as f64
    );
    assert!(
        maximum - minimum > 25,
        "limb darkening and surface detail must give the globe visible shape"
    );

    let centroid = |pixels: &[u8]| {
        let mut sum = 0.0;
        let mut count = 0.0;
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                if pixel(pixels, x, y)[3] > 250 {
                    sum += x as f32 + 0.5;
                    count += 1.0;
                }
            }
        }
        assert!(count > 1000.0);
        sum / count
    };
    let left = render(Vec3::new(-1.2, 0.0, 0.0), Quat::IDENTITY, projection, 1.0, [1.0, 0.65, 0.25], 0.4);
    let right = render(Vec3::new(1.2, 0.0, 0.0), Quat::IDENTITY, projection, 1.0, [1.0, 0.65, 0.25], 0.4);
    assert!(
        centroid(&left) - centroid(&right) > 4.0,
        "finite sun must have binocular disparity"
    );
    let mut asymmetric = projection;
    asymmetric.z_axis.x = 0.17;
    let eye = Vec3::new(3.0, 0.0, 0.0);
    let offset = render(eye, Quat::IDENTITY, asymmetric, 1.0, [1.0, 0.65, 0.25], 0.4);
    let center =
        asymmetric * Mat4::from_translation(-eye) * Vec3::new(0.0, 0.0, -160.0).extend(1.0);
    let expected = (center.x / center.w * 0.5 + 0.5) * SIZE as f32;
    assert!(
        (centroid(&offset) - expected).abs() < 1.0,
        "OpenXR asymmetric projections must locate the same sphere correctly"
    );

    let depth_at = |z: f32| {
        let p = projection * Vec3::new(0.0, 0.0, z).extend(1.0);
        p.z / p.w
    };
    let occluded = render(Vec3::ZERO, Quat::IDENTITY, projection, depth_at(-40.0), [1.0, 0.65, 0.25], 0.4);
    assert!(
        occluded.iter().all(|v| *v == 0),
        "foreground geometry hides the globe and its corona"
    );
    let behind = render(Vec3::ZERO, Quat::IDENTITY, projection, depth_at(-300.0), [1.0, 0.65, 0.25], 0.4);
    assert_eq!(base, behind, "geometry beyond the sun must not eclipse it");
    // Change uniforms on the same renderer to exercise cached uploads.
    for (tint, intensity) in [([1.0, 0.65, 0.25], 0.0), ([0.0; 3], 3.0)] {
        let dark = render(Vec3::ZERO, Quat::IDENTITY, projection, 1.0, tint, intensity);
        assert!(dark.chunks_exact(4).all(|p| p[..3] == [0, 0, 0]), "black sun must emit no surface or corona light");
    }
    let blue = render(Vec3::ZERO, Quat::IDENTITY, projection, 1.0, [0.0, 0.0, 1.0], 3.0);
    assert!(blue.chunks_exact(4).all(|p| p[0] == 0 && p[1] == 0));
    assert!(blue.chunks_exact(4).any(|p| p[2] > 100));
    let dim = render(Vec3::ZERO, Quat::IDENTITY, projection, 1.0, [1.0, 0.65, 0.25], 0.04);
    assert!(pixel(&dim, 192, 192)[0] < pixel(&base, 192, 192)[0]);
    assert_eq!(render(Vec3::ZERO, Quat::IDENTITY, projection, 1.0, [1.0, 0.65, 0.25], 0.4), base);
    std::fs::create_dir_all("target/vr-visual-checks").unwrap();
    for (name, pixels) in [
        ("sun-world-sphere", base),
        ("sun-head-roll", rolled),
        ("sun-left-eye", left),
        ("sun-right-eye", right),
    ] {
        image::save_buffer(
            format!("target/vr-visual-checks/{name}.png"),
            &pixels,
            SIZE,
            SIZE,
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
}
