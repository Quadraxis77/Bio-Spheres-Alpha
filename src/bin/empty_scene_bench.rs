//! Offscreen empty-world profiling; no game window or headset session is opened.

use bio_spheres::{
    scene::{Scene, SceneManager},
    ui::{panel_context::GenomeEditorState, SimulationMode},
};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let number = |name: &str, default: u32| {
        args.windows(2)
            .find(|a| a[0] == name)
            .and_then(|a| a[1].parse().ok())
            .unwrap_or(default)
    };
    let width = number("--width", 1920);
    let height = number("--height", 1080);
    let capacity = number("--capacity", 200_000);
    let frames = number("--frames", 120);
    let stereo = args.iter().any(|arg| arg == "--stereo");
    let headless = args.iter().any(|arg| arg == "--headless");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 64,
                max_storage_buffer_binding_size: adapter
                    .limits()
                    .max_storage_buffer_binding_size
                    .min(512 * 1024 * 1024),
                max_buffer_size: adapter.limits().max_buffer_size.min(512 * 1024 * 1024),
                ..Default::default()
            },
            ..Default::default()
        }),
    )
    .unwrap();
    println!(
        "GPU={} size={width}x{height} capacity={capacity} stereo={stereo} headless={headless}",
        adapter.get_info().name
    );
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    let mut manager = SceneManager::new(&device, &queue, &config);
    manager.switch_mode(
        SimulationMode::Gpu,
        &device,
        &queue,
        &config,
        400.0,
        capacity,
        &GenomeEditorState::default(),
    );
    let scene = manager.gpu_scene_mut().unwrap();
    scene.set_gpu_timing_enabled(true);
    scene.headless_no_render = headless;
    if let Some(simulator) = &scene.fluid_simulator {
        simulator.set_static_water_world(args.iter().any(|arg| arg == "--static-water"));
    }
    if let Some(timer) = &mut scene.gpu_timer {
        timer.set_view_count(if stereo && !headless { 2 } else { 1 });
    }
    scene.show_dof = !stereo;
    if args.iter().any(|arg| arg == "--no-mesh") {
        scene.show_gpu_density_mesh = false;
    }
    if args.iter().any(|arg| arg == "--paused") {
        scene.set_paused(true);
    }
    let position = scene.camera.position();
    let rotation = scene.camera.view_rotation();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Empty-world profile"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 2,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: config.usage,
        view_formats: &[],
    });
    let views: Vec<_> = (0..2)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    let mut samples = Vec::new();
    let mut totals = [0.0_f64; bio_spheres::scene::gpu_timer::SEGMENT_COUNT];
    for frame in 0..frames + 30 {
        let start = std::time::Instant::now();
        scene.update(1.0 / 120.0);
        if stereo {
            scene
                .camera
                .set_render_view(Some(bio_spheres::rendering::RenderView {
                    position: position + rotation * glam::vec3(-0.64, 0.0, 0.0),
                    rotation,
                    projection: bio_spheres::rendering::CameraProjection::from_fov(
                        -0.8, 0.8, -0.75, 0.75, 0.1, 5000.0,
                    ),
                    width,
                    height,
                }));
        }
        scene.render(
            &device, &queue, &views[0], None, 400.0, 500.0, 10.0, 25.0, 50.0, false, 0.1,
        );
        if stereo && !headless {
            scene
                .camera
                .set_render_view(Some(bio_spheres::rendering::RenderView {
                    position: position + rotation * glam::vec3(0.64, 0.0, 0.0),
                    rotation,
                    projection: bio_spheres::rendering::CameraProjection::from_fov(
                        -0.8, 0.8, -0.75, 0.75, 0.1, 5000.0,
                    ),
                    width,
                    height,
                }));
            scene.render_view(
                &device, &queue, &views[1], None, 400.0, 500.0, 10.0, 25.0, 50.0, false, 0.1,
            );
        }
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        if frame >= 30 {
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            if let Some(timer) = &scene.gpu_timer {
                for (total, sample) in totals.iter_mut().zip(timer.segment_times_ms()) {
                    *total += f64::from(sample);
                }
            }
        }
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    println!(
        "Frame avg={:.2}ms median={:.2}ms p95={:.2}ms",
        samples.iter().sum::<f64>() / frames as f64,
        samples[samples.len() / 2],
        samples[samples.len() * 95 / 100]
    );
    for (label, total) in bio_spheres::scene::gpu_timer::SEGMENT_LABELS
        .iter()
        .zip(totals)
    {
        println!("{label}={:.3}ms", total / frames as f64);
    }
}
