use super::*;

fn feed_once(scene: &mut GpuScene, device: &wgpu::Device, queue: &wgpu::Queue) {
    // Put one hungry phagocyte in a water voxel containing fresh food. This
    // isolates ingestion from population, food spawning and metabolic timing.
    let cells = &scene.gpu_triple_buffers;
    queue.write_buffer(
        &cells.cell_count_buffer,
        0,
        bytemuck::cast_slice(&[1u32, 1]),
    );
    queue.write_buffer(&cells.cell_types, 0, bytemuck::bytes_of(&2u32));
    queue.write_buffer(&cells.death_flags, 0, bytemuck::bytes_of(&0u32));
    queue.write_buffer(&cells.nutrients_buffer, 0, bytemuck::bytes_of(&10_000i32));
    queue.write_buffer(
        &cells.split_nutrient_thresholds,
        0,
        bytemuck::bytes_of(&50.0f32),
    );
    for positions in &cells.position_and_mass {
        queue.write_buffer(positions, 0, bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.1]));
    }
    let fluid = scene.fluid_simulator.as_ref().unwrap();
    let offset = (64 + 64 * 128 + 64 * 128 * 128) * 4;
    queue.write_buffer(
        fluid.current_state_buffer(),
        offset,
        bytemuck::bytes_of(&1u32),
    );
    queue.write_buffer(
        fluid.nutrient_voxels_buffer(),
        offset,
        bytemuck::bytes_of(&1u32),
    );

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Post-reset phagocyte intake"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    scene.run_phagocyte_consumption(device, &mut encoder, queue);
    encoder.copy_buffer_to_buffer(
        &scene.gpu_triple_buffers.nutrients_buffer,
        0,
        &staging,
        0,
        4,
    );
    encoder.copy_buffer_to_buffer(
        scene
            .fluid_simulator
            .as_ref()
            .unwrap()
            .nutrient_voxels_buffer(),
        offset,
        &staging,
        4,
        4,
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let values = staging.slice(..).get_mapped_range();
    let values: &[u32] = bytemuck::cast_slice(&values);
    assert_eq!(
        values[0], 30_000,
        "a phagocyte must gain 20 nutrients from food after reset"
    );
    assert_eq!(
        values[1], 2,
        "feeding must consume the active fluid world's nutrient"
    );
}

#[test]
fn phagocytes_keep_feeding_after_scene_resets_and_fluid_transfers() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
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
    let mut scene = GpuScene::with_capacity_and_radius(&device, &queue, &config, 64, 50.0);
    // Build the initial water/light resources directly, without generating a
    // cave or running weather: only ownership and consumption matter here.
    let light = LightFieldSystem::new(&device, 50.0, 64);
    let solid = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Empty feeding-test terrain"),
        size: 128 * 128 * 128 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    scene.fluid_simulator = Some(GpuFluidSimulator::new(
        &device,
        50.0,
        glam::Vec3::ZERO,
        solid,
        light.solar_transmittance_buffer(),
        &light.luminocyte_emission.buffer,
    ));
    scene.light_field_system = Some(light);
    scene.initialize_phagocyte_consumption(&device);
    feed_once(&mut scene, &device, &queue);

    scene.reset(&queue);
    feed_once(&mut scene, &device, &queue);

    // Match the capacity-only reset path: move the existing fluid/light into
    // a new scene, then reconnect it to newly allocated cell buffers.
    for capacity in [128, 256] {
        let mut replacement =
            GpuScene::with_capacity_and_radius(&device, &queue, &config, capacity, 50.0);
        replacement.fluid_simulator = scene.fluid_simulator.take();
        replacement.light_field_system = scene.light_field_system.take();
        replacement.rewire_transferred_fluid_system(&device, &queue, config.format);
        scene = replacement;
        scene.reset(&queue);
        feed_once(&mut scene, &device, &queue);

        // A full reset also clears the fluid. Ingestion must resume once water
        // and food return, using the same active buffers.
        let mut encoder = device.create_command_encoder(&Default::default());
        scene.reset_fluid(&device, &queue, &mut encoder);
        queue.submit([encoder.finish()]);
        feed_once(&mut scene, &device, &queue);
    }
}
