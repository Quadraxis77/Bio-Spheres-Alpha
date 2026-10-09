use bio_spheres::rendering::VolumetricFogRenderer;
use bio_spheres::simulation::fluid_simulation::gpu_simulator::{GpuFluidParams, GpuFluidSimulator};
use bio_spheres::simulation::gpu_physics::light_field::{LightFieldParams, LightFieldSystem};
use bio_spheres::simulation::gpu_physics::luminocyte_emission::{
    emission_shader_source, LuminocyteEmission,
};
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

fn shader_f32_const(source: &str, name: &str) -> f32 {
    let prefix = format!("const {name}: f32 = ");
    source
        .lines()
        .find_map(|line| line.trim().strip_prefix(&prefix))
        .and_then(|value| value.trim_end_matches(';').parse().ok())
        .unwrap_or_else(|| panic!("missing f32 constant {name}"))
}

#[test]
fn luminocyte_food_conversion_is_subcritical_per_receiver() {
    let shader = include_str!("../shaders/photocyte_light.wgsl");
    let cost = shader_f32_const(shader, "LUMINOCYTE_NUTRIENT_COST_PER_LIGHT_SECOND");
    let recovery = shader_f32_const(shader, "LUMINOCYTE_PHOTOCYTE_NUTRIENTS_PER_LIGHT_SECOND");

    assert!(recovery < cost);
    assert_eq!(recovery / cost, 5.0 / 6.0);
    for distance_voxels in [0.0f32, 1.0, 3.0, 5.0] {
        let falloff = (1.0 - distance_voxels / 6.0).max(0.0).powi(2);
        assert!(recovery * falloff < cost);
    }

    // The production shaders round nutrient payment up and emitted strength
    // down. Check the low-brightness boundary where opposite rounding could
    // otherwise produce a small amount of free light.
    let dt = 1.0 / 60.0;
    for brightness in [0.0011f32, 0.00147, 0.01, 0.15, 1.0, 3.0, 4.0] {
        let paid_units = (brightness * cost * dt * 1000.0).ceil() as u32;
        let emitted = (brightness.min(4.0) * 1024.0).floor() / 1024.0;
        let recovered_units = (emitted * recovery * dt * 1000.0).floor() as u32;
        assert!(recovered_units < paid_units);
    }
}

#[test]
fn raw_solar_field_keeps_occlusion_separate_from_visual_ambient_floor() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();

        let params = LightFieldParams {
            grid_resolution: 4,
            cell_size: 1.0,
            grid_origin_x: -2.0,
            grid_origin_y: -2.0,
            grid_origin_z: -2.0,
            world_radius: 2.0,
            light_dir_x: 0.0,
            light_dir_y: 1.0,
            light_dir_z: 0.0,
            max_steps: 4,
            step_size: 1.0,
            absorption_solid: 100.0,
            absorption_cell: 0.0,
            ambient_floor: 0.25,
            scattering_coefficient: 0.0,
            time: 0.0,
            sun_color_r: 1.0,
            sun_color_g: 1.0,
            sun_color_b: 1.0,
            water_light_attenuation: 1.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let index = |x: usize, y: usize, z: usize| x + y * 4 + z * 16;
        let mut solids = vec![0u32; 64];
        solids[index(1, 2, 1)] = 1;
        solids[index(1, 3, 1)] = 1;
        let solid = buffer(&device, bytemuck::cast_slice(&solids));
        let occupancy = buffer(&device, bytemuck::cast_slice(&vec![0u32; 64]));
        let light = buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64]));
        let colors = buffer(&device, bytemuck::cast_slice(&vec![[0.0f32; 4]; 64]));
        let water = buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64]));
        let humidity = buffer(&device, bytemuck::cast_slice(&vec![0u32; 64]));
        let ice = buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64]));
        let glow = buffer(&device, bytemuck::cast_slice(&vec![[0.0f32; 4]; 64]));
        let solar = buffer(&device, bytemuck::cast_slice(&vec![0.0f32; 64]));

        let layout_entries: Vec<_> = (0..10)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 1 | 2 | 5 | 6 | 7 | 8),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &layout_entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/light_field_compute.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_light_field"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = [
            params_buffer,
            solid,
            occupancy,
            light.clone(),
            colors,
            water,
            humidity,
            ice,
            glow,
            solar.clone(),
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);

        let shadowed_idx = index(1, 1, 1) as u64 * 4;
        let lit_idx = index(1, 1, 0) as u64 * 4;
        let shadowed_sun = f32::from_bits(read(&device, &queue, &solar, shadowed_idx, 4)[0]);
        let lit_sun = f32::from_bits(read(&device, &queue, &solar, lit_idx, 4)[0]);
        let shadowed_visual = f32::from_bits(read(&device, &queue, &light, shadowed_idx, 4)[0]);
        assert_eq!(shadowed_sun, 0.0, "occluding rock blocks thermal sunlight");
        assert!(lit_sun > 0.99, "clear sky preserves solar transmittance");
        let rock_surface = index(1, 3, 1) as u64 * 4;
        let rock_sun = f32::from_bits(read(&device, &queue, &solar, rock_surface, 4)[0]);
        assert!(
            rock_sun > 0.99,
            "exposed rock must receive thermal sunlight"
        );
        let exterior_sun = f32::from_bits(read(&device, &queue, &solar, 0, 4)[0]);
        assert_eq!(
            exterior_sun, 0.0,
            "exterior containment is not a solar heat source"
        );
        assert_eq!(
            shadowed_visual, 0.25,
            "the ambient visual floor must not leak into thermal sunlight"
        );

        // Optical cloud/ice scattering must not lock visibly sunlit pools
        // below freezing. Real solid occlusion must still block both fields.
        queue.write_buffer(&buffers[7], 0, bytemuck::cast_slice(&vec![1.0f32; 64]));
        queue.write_buffer(
            &buffers[6],
            0,
            bytemuck::cast_slice(&vec![120u32 * 256; 64]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let thermal = f32::from_bits(read(&device, &queue, &solar, lit_idx, 4)[0]);
        let optical = f32::from_bits(read(&device, &queue, &light, lit_idx, 4)[0]);
        assert!(thermal > 0.85 && thermal > optical + 0.1,
            "ice/clouds may scatter visible light without blocking pool heating: thermal={thermal}, optical={optical}");
        assert_eq!(
            f32::from_bits(read(&device, &queue, &solar, shadowed_idx, 4)[0]),
            0.0
        );
    });
}

fn buffer(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, offset, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

#[test]
fn luminocytes_add_occlude_and_clear_without_heating_water() {
    run_emission_test(false);
}

#[test]
fn hardware_ray_queries_occlude_and_rebuild_when_supported() {
    run_emission_test(true);
}

#[test]
fn validate_both_emission_shader_variants() {
    for hardware in [false, true] {
        let source = emission_shader_source(hardware);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    for (name, source) in [
        (
            "luminocyte resolve",
            include_str!("../shaders/luminocyte_resolve.wgsl"),
        ),
        (
            "photocyte consumption",
            include_str!("../shaders/photocyte_light.wgsl"),
        ),
        (
            "volumetric haze",
            include_str!("../shaders/volumetric_fog.wgsl"),
        ),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn shared_radiative_field_pipelines_create() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let limits = adapter.limits();
        let (device, _) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 64
                        .min(limits.max_storage_buffers_per_shader_stage),
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    max_buffer_size: limits.max_buffer_size,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();
        let _light = LightFieldSystem::new(&device, 200.0, 16);
        let _fog = VolumetricFogRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, 64, 64);
    });
}

fn run_emission_test(hardware: bool) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        let supported = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        eprintln!(
            "Adapter: {:?}; ray queries supported: {supported}; hardware test: {hardware}",
            adapter.get_info()
        );
        if hardware && !supported {
            eprintln!("SKIP hardware execution: adapter has no ray-query feature; shader validation still runs");
            return;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                required_features: if hardware {
                    wgpu::Features::EXPERIMENTAL_RAY_QUERY
                } else {
                    wgpu::Features::empty()
                },
                // SAFETY: testing wgpu's validated experimental ray-query API.
                experimental_features: if hardware {
                    unsafe { wgpu::ExperimentalFeatures::enabled() }
                } else {
                    Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();
        // Compile the production receiving-surface shaders as well.
        for source in [
            include_str!("../shaders/cells/cell_unified.wgsl"),
            include_str!("../shaders/cave_system.wgsl"),
        ] {
            let _ = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Local light receivers"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        }
        let emission = LuminocyteEmission::new(&device, [0.; 3], 1.);
        let positions = buffer(
            &device,
            bytemuck::cast_slice(&[[64.5f32, 64.5, 64.5, 1.]; 2]),
        );
        let glow = buffer(
            &device,
            bytemuck::cast_slice(&[[1f32, 0., 0., 1.], [0., 0., 1., 1.]]),
        );
        let count = buffer(&device, bytemuck::cast_slice(&[1u32]));
        let n = 128usize.pow(3);
        let mut walls = vec![0u32; n];
        let index = |x: usize, y: usize, z: usize| x + y * 128 + z * 128 * 128;
        for z in 0..128 {
            for y in 0..128 {
                walls[index(66, y, z)] = 1;
            }
        }
        emission.set_solid_mask(&walls);
        let solid = buffer(&device, bytemuck::cast_slice(&walls));
        let mut occupancy_values = vec![0u32; n];
        let occupancy = buffer(&device, bytemuck::cast_slice(&occupancy_values));
        let colors = buffer(&device, bytemuck::cast_slice(&vec![[0f32; 4]; n]));
        let intensity = buffer(&device, bytemuck::cast_slice(&vec![0f32; n]));
        let dispatch = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&colors, 0, None);
            encoder.clear_buffer(&intensity, 0, None);
            emission.scatter(
                &device,
                &mut encoder,
                &positions,
                &glow,
                &count,
                &solid,
                &occupancy,
                2,
            );
            emission.resolve(&device, &mut encoder, &colors, &intensity);
            queue.submit([encoder.finish()]);
        };
        dispatch();
        assert_eq!(emission.hardware_ray_tracing_active(), hardware);
        let sample = |x| {
            read(
                &device,
                &queue,
                &emission.buffer,
                index(x, 64, 64) as u64 * 16,
                16,
            )
        };
        let one = sample(64)[3];
        assert_eq!(one, 1024);
        assert!(sample(66)[3] > 0, "receiving cave wall must be lit");
        assert_eq!(
            sample(67)[3],
            0,
            "opaque wall must occlude light and radiant heat"
        );
        assert_eq!(sample(71)[3], 0, "finite light radius");
        assert!(sample(62)[3] > 0, "unblocked cell must receive light");
        occupancy_values[index(63, 64, 64)] = 1;
        queue.write_buffer(&occupancy, 0, bytemuck::cast_slice(&occupancy_values));
        dispatch();
        assert_eq!(sample(62)[3], 0, "intervening cells must cast shadows");
        occupancy_values[index(63, 64, 64)] = 0;
        queue.write_buffer(&occupancy, 0, bytemuck::cast_slice(&occupancy_values));
        dispatch();
        let resolved = read(&device, &queue, &colors, index(64, 64, 64) as u64 * 16, 16);
        assert!(f32::from_bits(resolved[0]) > 0. && f32::from_bits(resolved[3]) > 0.);
        let resolved_intensity = read(&device, &queue, &intensity, index(64, 64, 64) as u64 * 4, 4);
        assert_eq!(f32::from_bits(resolved_intensity[0]), 1.0);
        let shadowed_intensity = read(&device, &queue, &intensity, index(67, 64, 64) as u64 * 4, 4);
        assert_eq!(f32::from_bits(shadowed_intensity[0]), 0.0);
        queue.write_buffer(&count, 0, bytemuck::cast_slice(&[2u32]));
        dispatch();
        assert_eq!(sample(64)[3], one * 2, "emitters must add");
        let blended_intensity = read(&device, &queue, &intensity, index(64, 64, 64) as u64 * 4, 4);
        assert_eq!(f32::from_bits(blended_intensity[0]), 2.0);
        let blended_color = read(&device, &queue, &colors, index(64, 64, 64) as u64 * 16, 16);
        assert_eq!(
            [
                f32::from_bits(blended_color[0]),
                f32::from_bits(blended_color[1]),
                f32::from_bits(blended_color[2]),
            ],
            [0.5, 0.0, 0.5]
        );
        queue.write_buffer(&glow, 0, bytemuck::cast_slice(&[[0f32; 4]; 2]));
        dispatch();
        assert_eq!(
            sample(64)[3],
            0,
            "off/dead emitters must not leave heat behind"
        );

        assert_eq!(emission.hardware_ray_tracing_supported(), hardware);
        queue.write_buffer(&glow, 0, bytemuck::cast_slice(&[[1f32, 0., 0., 1.]; 2]));
        // The UI mode switch must select the actual dispatch, not just its label.
        emission.set_hardware_ray_tracing_enabled(false);
        dispatch();
        assert!(!emission.hardware_ray_tracing_active());
        assert_eq!(sample(67)[3], 0, "voxel mode must still occlude walls");
        assert_eq!(sample(64)[3], one * 2, "switching must preserve emission");

        // Edit geometry while hardware mode is off; re-enabling must rebuild it.
        // Remove all cave geometry and confirm the previous TLAS is discarded.
        walls.fill(0);
        queue.write_buffer(&solid, 0, bytemuck::cast_slice(&walls));
        emission.set_solid_mask(&walls);
        assert!(!emission.hardware_ray_tracing_active());
        dispatch();
        assert!(!emission.hardware_ray_tracing_active());
        assert!(sample(67)[3] > 0, "voxel mode must follow cave edits");
        emission.set_hardware_ray_tracing_enabled(true);
        queue.write_buffer(&glow, 0, bytemuck::cast_slice(&[[1f32, 0., 0., 1.]; 2]));
        dispatch();
        assert_eq!(emission.hardware_ray_tracing_active(), hardware);
        assert!(
            sample(67)[3] > 0,
            "removed cave must not leave stale occluders"
        );

        // Validate the production fluid binding layout with the shared emission buffer.
        let sunlight = buffer(&device, bytemuck::cast_slice(&vec![0f32; n]));
        let _fluid = GpuFluidSimulator::new(
            &device,
            64.,
            glam::Vec3::ZERO,
            solid,
            &sunlight,
            &emission.buffer,
        );

        // Run the real thermal shader on a small cold volume. Solar heating
        // must follow the raw local sun field; geothermal and luminocyte
        // signals remain non-thermal.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Luminocyte climate regression"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fluid/fluid_sim.wgsl").into(),
            ),
        });
        let layout_entries: Vec<_> = (0..12)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 2 | 5 | 9 | 10 | 11),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &layout_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let thermal = pipeline("update_temperature");
        let apply = pipeline("apply_temperature");
        let mut params = GpuFluidParams::zeroed();
        params.grid_resolution = 4;
        params.cell_size = 1.;
        params.world_radius = 100.;
        params.thermal_inertia = 4.;
        let mut run_case =
            |medium: u32, sunlight: f32, other_sources: bool, ticks: u32, brightness: f32| {
                params.sun_brightness = brightness;
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let mut buffers = vec![uniform.clone()];
                for binding in 1..12 {
                    let data: Vec<u8> = match binding {
                        1 => bytemuck::cast_slice(&vec![0xffff0000u32 | medium; 64]).to_vec(),
                        5 => bytemuck::cast_slice(&vec![sunlight; 64]).to_vec(),
                        8 => bytemuck::cast_slice(&vec![32u32 * 256; 64]).to_vec(), // -18 C dark baseline
                        9 if other_sources => bytemuck::cast_slice(&vec![100.0f32; 64]).to_vec(),
                        10 if other_sources => bytemuck::cast_slice(
                            &(0..64).flat_map(|_| [0u32, 0, 0, 2048]).collect::<Vec<_>>(),
                        )
                        .to_vec(),
                        _ => vec![0; 64 * 4],
                    };
                    buffers.push(buffer(&device, &data));
                }
                let entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect();
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &entries,
                });
                let mut remaining = ticks;
                while remaining > 0 {
                    let batch = remaining.min(64);
                    let mut encoder = device.create_command_encoder(&Default::default());
                    for _ in 0..batch {
                        encoder.copy_buffer_to_buffer(&buffers[8], 0, &buffers[11], 0, 64 * 4);
                        encoder.clear_buffer(&buffers[8], 0, None);
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&thermal);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                        drop(pass);
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&apply);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    queue.submit([encoder.finish()]);
                    remaining -= batch;
                }
                read(&device, &queue, &buffers[8], 21 * 4, 4)[0]
            };

        for medium in [0, 1, 3] {
            let dark_with_other_sources = run_case(medium, 0.0, true, 1, 3.0);
            let dark_without_other_sources = run_case(medium, 0.0, false, 1, 3.0);
            assert_eq!(
                dark_with_other_sources, dark_without_other_sources,
                "fluid type {medium} must not gain heat from geothermal or luminocyte sources"
            );
            assert_eq!(
                run_case(medium, 0.0, true, 1, 0.0),
                32 * 256,
                "turning the sun off must also remove redistributed atmospheric heating"
            );
        }
        for medium in [0, 1, 3] {
            let temperate = run_case(medium, 1.0, false, 4000, 3.0);
            let temp_c = -50.0 + temperate as f32 / 256.0;
            let target = if medium == 1 { 49.2 } else { 30.0 };
            assert!(
                (temp_c - target).abs() < 2.0,
                "clear sunlight at brightness 3: medium={medium}, got {temp_c} C vs {target} C"
            );

            let bright = run_case(medium, 1.0, false, 2000, 5.0);
            let temp_c = -50.0 + bright as f32 / 256.0;
            assert!(
                temp_c > 115.0,
                "maximum sun brightness must reach boiling; medium={medium}, got {temp_c} C"
            );
        }
    });
}
