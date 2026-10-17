use bio_spheres::simulation::fluid_simulation::gpu_simulator::GpuFluidParams;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

const RES: u32 = 8;
const COUNT: usize = (RES * RES * RES) as usize;

fn encode(celsius: f32) -> u32 {
    ((celsius + 50.0) * 256.0).round().max(1.0) as u32
}

fn index(x: u32, y: u32, z: u32) -> usize {
    (x + y * RES + z * RES * RES) as usize
}

fn evaporation_tick(draw: std::ops::Range<u32>) -> u32 {
    // Select a deterministic draw for the source at (4, 4, 4), so a rare
    // phase event can be exercised without waiting minutes for random weather.
    (0..10_000_000u32)
        .find(|tick| {
            let time_hash = ((*tick as f32 / 60.0) * 1000.0) as u32;
            let mut hash = (4u32 * 73856093 ^ 4u32 * 19349663 ^ 4u32 * 83492791) ^ time_hash;
            hash ^= hash >> 16;
            hash = hash.wrapping_mul(0x7FEB352D);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x846CA68B);
            hash ^= hash >> 16;
            draw.contains(&(hash & 0x00ffffff))
        })
        .unwrap()
}

type Climate = ClimateGrid<RES>;

struct ClimateGrid<const N: u32> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<wgpu::Buffer>,
    bindings: wgpu::BindGroup,
    thermal: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    sliced: wgpu::ComputePipeline,
    weather: wgpu::ComputePipeline,
    movement: wgpu::ComputePipeline,
    static_phase: wgpu::ComputePipeline,
    fog: wgpu::ComputePipeline,
    evaporation: wgpu::ComputePipeline,
    params: GpuFluidParams,
    tick: u32,
}

impl<const N: u32> ClimateGrid<N> {
    const COUNT: usize = (N * N * N) as usize;
    fn new() -> Self {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&Default::default());
            let adapter = instance
                .request_adapter(&Default::default())
                .await
                .expect("GPU required for climate regression");
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Climate regression bindings"),
                entries: &(0..12)
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
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Production climate shader"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{}\n{}",
                        include_str!("../shaders/fluid/fluid_sim.wgsl"),
                        r#"
@compute @workgroup_size(4, 4, 4)
fn sample_evaporation(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= vec3<u32>(params.grid_resolution)) { return; }
    let idx = grid_index(gid.x, gid.y, gid.z);
    atomicStore(&water_velocity[idx], bitcast<u32>(evaporation_probability(idx, field_temp_c(idx))));
}
"#,
                    )
                    .into(),
                ),
            });
            let pipeline = |entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let thermal = pipeline("update_temperature");
            let apply = pipeline("apply_temperature");
            let sliced = pipeline("update_temperature_slice");
            let weather = pipeline("condense_humidity");
            let movement = pipeline("fluid_swap");
            let static_phase = pipeline("fluid_static_water_phase");
            let fog = pipeline("diffuse_humidity");
            let evaporation = pipeline("sample_evaporation");
            let mut params = GpuFluidParams::zeroed();
            params.grid_resolution = N;
            params.cell_size = 1.0;
            params.world_radius = 100.0;
            params.grid_origin_x = -4.0;
            params.grid_origin_y = -4.0;
            params.grid_origin_z = -4.0;
            params.gravity_mode = 3;
            params.gravity_magnitude = 9.8;
            params.thermal_inertia = 4.0;
            params.sun_brightness = 3.0;
            let defaults = bio_spheres::ui::types::ClimateSettings::default();
            params.freeze_threshold = defaults.freeze_threshold;
            params.melt_threshold = defaults.melt_threshold;
            params.freeze_rate = 1.0;
            params.melt_rate = 1.5;
            params.snow_melt_rate = 3.0;
            params.snow_compact_rate = 1.0;
            let mut buffers = vec![
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
            ];
            for binding in 1..12 {
                buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: if binding == 10 {
                        Self::COUNT * 16
                    } else {
                        Self::COUNT * 4
                    } as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            Self {
                device,
                queue,
                buffers,
                bindings,
                thermal,
                apply,
                sliced,
                weather,
                movement,
                static_phase,
                fog,
                evaporation,
                params,
                tick: 0,
            }
        })
    }

    fn write(&self, binding: usize, data: &[u32]) {
        self.queue
            .write_buffer(&self.buffers[binding], 0, bytemuck::cast_slice(data));
    }

    fn reset(&self, phase: u32, celsius: f32, sunlight: f32) {
        for binding in [2, 3, 4, 6, 7, 9] {
            self.write(binding, &vec![0; Self::COUNT]);
        }
        self.write(1, &vec![0xffff0000 | phase; Self::COUNT]);
        self.write(5, &vec![sunlight.to_bits(); Self::COUNT]);
        self.write(8, &vec![encode(celsius); Self::COUNT]);
        self.write(10, &vec![0; Self::COUNT * 4]);
    }

    fn run(&mut self, entry: &str, ticks: u32) {
        let pipeline = match entry {
            "thermal" => &self.thermal,
            "sliced" => &self.sliced,
            "weather" => &self.weather,
            "movement" => &self.movement,
            "static" => &self.static_phase,
            "fog" => &self.fog,
            "evaporation" => &self.evaporation,
            _ => unreachable!(),
        };
        for start in (0..ticks).step_by(64) {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for _ in start..(start + 64).min(ticks) {
                self.params.time = self.tick as f32 / 60.0;
                self.params.climate_phase = self.tick % 4;
                self.tick += if matches!(entry, "thermal" | "weather" | "static" | "fog") {
                    4
                } else {
                    1
                };
                let params = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::bytes_of(&self.params),
                        usage: wgpu::BufferUsages::COPY_SRC,
                    });
                encoder.copy_buffer_to_buffer(&params, 0, &self.buffers[0], 0, params.size());
                encoder.copy_buffer_to_buffer(
                    &self.buffers[8],
                    0,
                    &self.buffers[11],
                    0,
                    (Self::COUNT * 4) as u64,
                );
                if matches!(entry, "thermal" | "sliced") {
                    encoder.clear_buffer(&self.buffers[8], 0, None);
                }
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bindings, &[]);
                pass.dispatch_workgroups(
                    N.div_ceil(4),
                    N.div_ceil(4),
                    if entry == "sliced" {
                        N.div_ceil(16)
                    } else {
                        N.div_ceil(4)
                    },
                );
                drop(pass);
                if matches!(entry, "thermal" | "sliced") {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.apply);
                    pass.set_bind_group(0, &self.bindings, &[]);
                    pass.dispatch_workgroups(N.div_ceil(4), N.div_ceil(4), N.div_ceil(4));
                }
            }
            self.queue.submit([encoder.finish()]);
        }
    }

    fn read(&self, binding: usize) -> Vec<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (Self::COUNT * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers[binding], 0, &staging, 0, staging.size());
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
    }

    fn temperatures(&self) -> Vec<f32> {
        self.read(8)
            .iter()
            .map(|raw| *raw as f32 / 256.0 - 50.0)
            .collect()
    }

    fn mean_temperature(&self) -> f32 {
        self.temperatures().iter().sum::<f32>() / Self::COUNT as f32
    }

    fn phase_counts(&self) -> [usize; 5] {
        let mut counts = [0; 5];
        for state in self.read(1) {
            counts[(state & 7) as usize] += 1;
        }
        counts
    }

    /// Feed live phase data to the production particle extractor so a vapor
    /// count alone cannot pass while the renderer hides temperate evaporation.
    fn visible_vapor(&self) -> usize {
        use bio_spheres::rendering::steam_particles::{ExtractParams, SteamParticle};
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Temperate vapor visibility"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/steam_extract.wgsl").into(),
                ),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&ExtractParams {
                    grid_resolution: N,
                    cell_size: 1.0,
                    max_particles: Self::COUNT as u32,
                    time: self.params.time,
                    grid_origin: [-4.0; 3],
                    sun_brightness: 1.0,
                    gravity_mode: 1,
                    _padding: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let size = (Self::COUNT * std::mem::size_of::<SteamParticle>()) as u64;
        let storage = |size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let particles = storage(size);
        let counter = storage(4);
        // Unsmoothed water density is enough for this above-surface visibility
        // check; the extractor independently excludes steam enclosed by water.
        let density: Vec<f32> = self
            .read(1)
            .into_iter()
            .map(|s| if s & 7 == 1 { 1.0 } else { 0.0 })
            .collect();
        let water = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&density),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                &self.buffers[1],
                &particles,
                &counter,
                &params,
                &self.buffers[5],
                &water,
                &self.buffers[10],
            ]
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size + 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(N.div_ceil(4), N.div_ceil(4), N.div_ceil(4));
        }
        encoder.copy_buffer_to_buffer(&counter, 0, &staging, 0, 4);
        encoder.copy_buffer_to_buffer(&particles, 0, &staging, 4, size);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = staging.slice(..).get_mapped_range();
        let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        bytes[4..]
            .chunks_exact(std::mem::size_of::<SteamParticle>())
            .take(count)
            .map(bytemuck::pod_read_unaligned::<SteamParticle>)
            // Quiet vapor should still be lit and drawn at 2-4% opacity.
            // Exclude opaque snow and geothermal soot from this vapor check.
            .filter(|p| {
                p.animation[1] == 1.0
                    && (0.02..=0.04).contains(&p.color[3])
                    && p.color[0] > 0.5
            })
            .count()
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        sliced: bool,
    ) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(
            N.div_ceil(4),
            N.div_ceil(4),
            if sliced {
                N.div_ceil(16)
            } else {
                N.div_ceil(4)
            },
        );
    }

    // Run the actual dynamic sequence: 60 Hz, one temperature slice/tick,
    // fog and cloud precipitation every four ticks, four movement substeps.
    fn dynamic_seconds(&mut self, seconds: u32) {
        self.fluid_seconds(seconds, true);
    }

    // Holding the thermal field fixed lets cold-source tests isolate vapor
    // production/lifetime from the separate sun-driven climate equilibrium.
    fn fluid_seconds(&mut self, seconds: u32, update_temperature: bool) {
        for _ in 0..seconds {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for _ in 0..60 {
                self.params.time = self.tick as f32 / 60.0;
                self.params.climate_phase = self.tick % 4;
                self.tick += 1;
                for sub_step in 0..4 {
                    self.params.sub_step = sub_step;
                    let params =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: None,
                                contents: bytemuck::bytes_of(&self.params),
                                usage: wgpu::BufferUsages::COPY_SRC,
                            });
                    encoder.copy_buffer_to_buffer(&params, 0, &self.buffers[0], 0, params.size());
                    if sub_step == 0 {
                        if update_temperature {
                            if self.params.climate_phase == 0 {
                                encoder.clear_buffer(&self.buffers[4], 0, None);
                            }
                            encoder.copy_buffer_to_buffer(
                                &self.buffers[8],
                                0,
                                &self.buffers[11],
                                0,
                                (Self::COUNT * 4) as u64,
                            );
                            encoder.clear_buffer(&self.buffers[8], 0, None);
                            self.dispatch(&mut encoder, &self.sliced, true);
                            self.dispatch(&mut encoder, &self.apply, false);
                        }
                        if self.params.climate_phase == 2 {
                            self.dispatch(&mut encoder, &self.fog, false);
                        }
                        if self.params.climate_phase == 3 {
                            self.dispatch(&mut encoder, &self.weather, false);
                        }
                    }
                    self.dispatch(&mut encoder, &self.movement, false);
                }
            }
            self.queue.submit([encoder.finish()]);
            let stats = self.read(4);
            self.params.air_temperature_reference = if stats[3] > 0 {
                encode(stats[2] as f32 / stats[3] as f32 - 50.0)
            } else {
                0
            };
        }
        self.params.sub_step = 0;
    }
}

#[test]
fn vents_warm_air_and_water_in_darkness_and_heat_spreads_to_neighbors() {
    let mut climate = Climate::new();
    climate.params.sun_brightness = 0.0;
    let source = index(4, 4, 4);
    let neighbor = index(4, 4, 3);
    for entry in ["thermal", "sliced"] {
        for medium in [0, 1] {
            climate.reset(medium, 18.0, 0.0);
            climate.tick = 0;
            let ticks_per_second = if entry == "thermal" { 15 } else { 60 };
            climate.run(entry, 15 * ticks_per_second);
            let unheated = climate.temperatures();

            climate.reset(medium, 18.0, 0.0);
            climate.tick = 0;
            let mut heat = vec![0.0f32; COUNT];
            heat[source] = 34.0;
            climate.write(
                9,
                &heat.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            );
            climate.run(entry, 15 * ticks_per_second);
            let warmed = climate.temperatures();
            eprintln!(
                "vent, {entry}, medium {medium}: source +{:.2} C, neighbor +{:.2} C",
                warmed[source] - unheated[source],
                warmed[neighbor] - unheated[neighbor]
            );
            assert!(
                warmed[source] > unheated[source] + 1.0,
                "the vent must supply real heat to medium {medium}, {entry}"
            );
            assert!(
                warmed[neighbor] > unheated[neighbor] + 0.25,
                "vent heat must conduct into neighboring medium {medium}, {entry}"
            );
            assert!(
                warmed[source] > warmed[neighbor],
                "vent heating must remain local"
            );
            assert!(
                warmed.iter().all(|t| *t < 100.0),
                "the default vent must warm gradually"
            );
        }
    }
}

#[test]
fn shaded_water_slowly_tracks_seventy_f_air_without_freezing() {
    let mut climate = Climate::new();
    let air_c = (70.0 - 32.0) / 1.8;
    for entry in ["thermal", "sliced"] {
        for phase in [1, 2] {
            climate.reset(phase, -5.0, 0.0);
            climate.params.air_temperature_reference = encode(air_c);
            climate.run(entry, if entry == "thermal" { 4 * 15 } else { 4 * 60 });
            let early = climate.mean_temperature();
            assert!(
                early > -5.0 && early < 0.0,
                "{entry}, phase {phase}: water must warm gradually, got {early} C"
            );
            climate.run(
                entry,
                if entry == "thermal" {
                    360 * 15
                } else {
                    360 * 60
                },
            );
            let settled = climate.mean_temperature();
            assert!(
                (air_c - 4.0..air_c - 2.0).contains(&settled),
                "{entry}, phase {phase}: shaded water must settle slightly below 70 F air, got {settled} C"
            );
            climate.run(if phase == 1 { "static" } else { "movement" }, 600);
            assert_eq!(climate.phase_counts()[1], COUNT);
        }
    }
}

#[test]
fn shaded_pool_follows_measured_air_and_freezes_only_when_air_turns_cold() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    climate.reset(0, 21.0, 0.0);
    let mut states = vec![0; COUNT];
    let mut solids = vec![0; COUNT];
    for z in 0..RES {
        for y in 0..=2 {
            for x in 0..RES {
                let i = index(x, y, z);
                if y == 0 {
                    solids[i] = 1;
                } else {
                    states[i] = 0xffff0001;
                }
            }
        }
    }
    climate.write(1, &states);
    climate.write(2, &solids);
    climate.dynamic_seconds(180);
    let phases = climate.read(1);
    let temps = climate.temperatures();
    let mean = |phase| {
        let selected: Vec<_> = (0..COUNT)
            .filter(|i| solids[*i] == 0 && phases[*i] & 7 == phase)
            .map(|i| temps[i])
            .collect();
        selected.iter().sum::<f32>() / selected.len() as f32
    };
    let air = mean(0);
    let water = mean(1);
    eprintln!("shaded pool: air {air:.1} C, water {water:.1} C");
    assert!((18.0..26.0).contains(&air));
    assert!(water > air - 6.0 && water < air);
    assert_eq!(climate.phase_counts()[2], 0);
    climate.params.sun_brightness = 0.0;
    climate.dynamic_seconds(240);
    assert!(
        climate.phase_counts()[2] > (RES * RES) as usize / 2,
        "sustained freezing air must still freeze a pool"
    );
}

#[test]
fn ice_needs_sustained_warmth_in_dynamic_and_static_water() {
    let mut climate = Climate::new();
    for entry in ["movement", "static"] {
        climate.tick = 0;
        climate.reset(2, 6.0, 0.0);
        let cadence = if entry == "movement" { 4 } else { 1 };
        climate.run(entry, 95 * cadence);
        assert_eq!(
            climate.phase_counts()[2],
            COUNT,
            "{entry}: a short warm spell must retain the ice"
        );
        climate.run(entry, 80 * cadence);
        assert_eq!(
            climate.phase_counts()[1],
            COUNT,
            "{entry}: sustained warmth must still melt the ice"
        );
        assert!(climate.read(7).iter().all(|debt| *debt == 0));
    }
}

#[test]
fn static_water_forgets_brief_cold_spells_before_freezing() {
    let mut climate = Climate::new();
    climate.reset(1, -6.0, 0.0);
    climate.run("static", 180);
    let cold_debt = f32::from_bits(climate.read(7)[0]);
    assert!(cold_debt > 0.0);
    climate.write(8, &vec![encode(0.0); COUNT]);
    climate.run("static", 32);
    assert!(f32::from_bits(climate.read(7)[0]) < cold_debt / 10.0);
    climate.write(8, &vec![encode(-6.0); COUNT]);
    climate.run("static", 350);
    assert_eq!(
        climate.phase_counts()[1],
        COUNT,
        "brief cold followed by a thaw must not cause premature freezing"
    );
    climate.run("static", 200);
    assert_eq!(climate.phase_counts()[2], COUNT);
    assert!(climate.temperatures().iter().all(|temp| *temp == 0.0));
}

#[test]
fn neighboring_ice_spreads_gradually_even_in_deep_cold() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    climate.reset(0, -18.0, 0.0);
    let water = index(4, 4, 4);
    let seed = index(3, 4, 4);
    let mut states = vec![0; COUNT];
    states[water] = 0xffff0001;
    states[seed] = 0xffff0002;
    let mut solids = vec![1; COUNT];
    for voxel in [water, seed, index(4, 5, 4)] {
        solids[voxel] = 0;
    }
    climate.write(1, &states);
    climate.write(2, &solids);
    climate.run("movement", 20);
    assert_eq!(
        climate.read(1)[water] & 7,
        1,
        "contact with an ice crystal must not immediately freeze surface water"
    );
    climate.run("movement", 40);
    let states = climate.read(1);
    assert_eq!(states[water], 0xffff0002);
    assert_eq!(states[seed], 0xffff0002);
    assert_eq!(climate.temperatures()[water], 0.0);
}

#[test]
fn legacy_melt_threshold_cannot_keep_warm_ice_frozen() {
    let mut climate = Climate::new();
    climate.params.melt_threshold = 75; // the user's saved pre-calibration setting
    for entry in ["movement", "static"] {
        climate.reset(2, 6.0, 0.0);
        climate.run(entry, if entry == "static" { 150 } else { 600 });
        assert!(
            climate.phase_counts()[2] < COUNT / 10,
            "{entry}: ice at 43 F must melt even with a legacy threshold"
        );
    }
}

/// Use the real ray marcher through changing water/ice density, rather than
/// assigning sunlight=1 to every submerged cell as the old calibration did.
struct PoolSunlight {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: bio_spheres::simulation::gpu_physics::light_field::LightFieldParams,
    params_buffer: wgpu::Buffer,
    water: wgpu::Buffer,
    ice: wgpu::Buffer,
}

impl PoolSunlight {
    fn new<const N: u32>(climate: &ClimateGrid<N>) -> Self {
        use bio_spheres::simulation::gpu_physics::light_field::LightFieldParams;
        let count = (N * N * N) as usize;
        let device = &climate.device;
        let params = LightFieldParams {
            grid_resolution: N,
            cell_size: 1.0,
            grid_origin_x: -(N as f32) * 0.5,
            grid_origin_y: -(N as f32) * 0.5,
            grid_origin_z: -(N as f32) * 0.5,
            world_radius: N as f32 * 0.5,
            light_dir_x: 0.0,
            light_dir_y: 1.0,
            light_dir_z: 0.0,
            max_steps: N * 2,
            step_size: 1.0,
            absorption_solid: 20.0,
            absorption_cell: 5.0,
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let storage = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let occupancy = storage((count * 4) as u64);
        let visual = storage((count * 4) as u64);
        let color = storage((count * 16) as u64);
        let water = storage((count * 4) as u64);
        let ice = storage((count * 4) as u64);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pool sunlight regression"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/light_field_compute.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("compute_light_field"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                &params_buffer,
                &climate.buffers[2],
                &occupancy,
                &visual,
                &color,
                &water,
                &climate.buffers[6],
                &ice,
                &climate.buffers[10],
                &climate.buffers[5],
            ]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        Self {
            pipeline,
            group,
            params,
            params_buffer,
            water,
            ice,
        }
    }

    fn update<const N: u32>(&self, climate: &ClimateGrid<N>) {
        let states = climate.read(1);
        for (buffer, phase) in [(&self.water, 1), (&self.ice, 2)] {
            let density: Vec<f32> = states
                .iter()
                .map(|s| if s & 7 == phase { 1.0 } else { 0.0 })
                .collect();
            climate
                .queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&density));
        }
        climate
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        let mut encoder = climate.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups((N * N * N).div_ceil(64), 1, 1);
        }
        climate.queue.submit([encoder.finish()]);
    }
}

#[test]
fn brightness_three_thaws_a_sunlit_icy_pool_and_keeps_its_average_at_60_to_90_f() {
    const N: u32 = 16;
    let count = (N * N * N) as usize;
    let mut climate = ClimateGrid::<N>::new();
    climate.params.gravity_mode = 1;
    climate.params.world_radius = N as f32 * 0.5;
    climate.params.grid_origin_x = -8.0;
    climate.params.grid_origin_y = -8.0;
    climate.params.grid_origin_z = -8.0;
    climate.params.thermal_inertia = 3.6; // user's current settings
    climate.params.melt_threshold = 75; // legacy save must work without a reset
    climate.params.humidity_diffusion_rate = 0.27;
    climate.reset(0, 18.0, 0.0);
    let mut solids = vec![0u32; count];
    let mut states = vec![0u32; count];
    let mut temperatures = vec![encode(18.0); count];
    let mut volume = 0;
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let i = (x + y * N + z * N * N) as usize;
                let radius_sq = [x, y, z]
                    .into_iter()
                    .map(|v| (v as f32 + 0.5 - 8.0).powi(2))
                    .sum::<f32>();
                if radius_sq > 64.0 || y <= 2 {
                    solids[i] = 1;
                } else if y <= 7 {
                    states[i] = 0xffff0002;
                    temperatures[i] = encode(-8.0);
                    volume += 1;
                }
            }
        }
    }
    climate.write(1, &states);
    climate.write(2, &solids);
    climate.write(8, &temperatures);
    let mut sun = PoolSunlight::new(&climate);
    // Sweep 120 degrees of the sunlit arc at the requested 1 degree/second.
    // Persistent overhead vapor stresses the actual cloud-attenuation path.
    for second in 0..120 {
        let angle = (second as f32 - 60.0).to_radians();
        sun.params.light_dir_x = angle.sin();
        sun.params.light_dir_y = angle.cos();
        let fog: Vec<u32> = (0..count)
            .map(|i| {
                if (i as u32 / N) % N >= 9 {
                    120 * 256
                } else {
                    0
                }
            })
            .collect();
        climate.write(6, &fog);
        sun.update(&climate);
        climate.dynamic_seconds(1);
        if second == 89 {
            let phases = climate.phase_counts();
            eprintln!("sunlit pool after 90 seconds: {phases:?}");
            assert!(
                phases[2] < volume / 20,
                "sustained warm air and sunlight must thaw the pool within 90 seconds"
            );
        }
    }
    let phases = climate.phase_counts();
    let final_states = climate.read(1);
    let final_temps = climate.temperatures();
    let water: Vec<f32> = final_states
        .iter()
        .zip(final_temps)
        .filter_map(|(s, t)| matches!(s & 7, 1 | 2 | 4).then_some(t))
        .collect();
    let mean_c = water.iter().sum::<f32>() / water.len() as f32;
    eprintln!(
        "ray-marched icy pool: {phases:?}, water-phase mean {:.1} F",
        mean_c * 1.8 + 32.0
    );
    assert_eq!(phases[1..].iter().sum::<usize>(), volume);
    assert!(phases[2] < volume / 20, "sunlit ice must melt: {phases:?}");
    assert!(
        (15.555..=32.223).contains(&mean_c),
        "brightness 3 water average must be 60-90 F, got {} F",
        mean_c * 1.8 + 32.0
    );
}

#[test]
fn whole_world_air_and_water_stay_temperate_over_a_complete_sun_orbit() {
    orbit_climate::<16>(2, 5);
}

#[test]
fn deep_water_does_not_accumulate_excess_heat_over_a_complete_sun_orbit() {
    orbit_climate::<32>(3, 15);
}

fn orbit_climate<const N: u32>(floor: u32, water_top: u32) {
    let count = (N * N * N) as usize;
    let mut climate = ClimateGrid::<N>::new();
    let radius = N as f32 / 2.0;
    climate.params.gravity_mode = 1;
    climate.params.world_radius = radius;
    climate.params.grid_origin_x = -radius;
    climate.params.grid_origin_y = -radius;
    climate.params.grid_origin_z = -radius;
    climate.params.thermal_inertia = 3.6;
    climate.params.humidity_diffusion_rate = 0.27;
    climate.reset(0, 0.0, 0.0);
    let mut solids = vec![0; count];
    let mut states = vec![0; count];
    let mut volume = 0;
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let i = (x + y * N + z * N * N) as usize;
                let radius_sq = [x, y, z]
                    .into_iter()
                    .map(|v| (v as f32 + 0.5 - radius).powi(2))
                    .sum::<f32>();
                if radius_sq >= radius * radius
                    || y <= floor
                    || (x == N / 2 - 1
                        && (water_top + 1..=water_top + 5).contains(&y)
                        && (N / 4..3 * N / 4).contains(&z))
                {
                    solids[i] = 1;
                } else if y <= water_top {
                    states[i] = 0xffff0001;
                    volume += 1;
                }
            }
        }
    }
    climate.write(1, &states);
    climate.write(2, &solids);
    let mut sun = PoolSunlight::new(&climate);
    let mut air_samples = Vec::new();
    let mut water_samples = Vec::new();
    let mut most_frozen = 0;
    // First revolution establishes the climate; measure the entire second
    // revolution, including low sun, terrain shadows, and the dark half.
    for second in 0..720 {
        let angle = (second as f32).to_radians();
        sun.params.light_dir_x = angle.sin();
        sun.params.light_dir_y = angle.cos();
        sun.update(&climate);
        climate.dynamic_seconds(1);
        if second == 29 {
            let states = climate.read(1);
            let temps = climate.temperatures();
            let air: Vec<_> = (0..count)
                .filter(|i| solids[*i] == 0 && states[*i] & 7 == 0)
                .map(|i| temps[i])
                .collect();
            let air_mean = air.iter().sum::<f32>() / air.len() as f32;
            assert!(
                air_mean > 15.555,
                "cold-start air must warm within 30 s: {air_mean} C"
            );
        }
        if second >= 360 && second % 15 == 14 {
            let states = climate.read(1);
            let temps = climate.temperatures();
            let mean = |air: bool| {
                let selected: Vec<f32> = (0..count)
                    .filter_map(|i| {
                        let phase = states[i] & 7;
                        (solids[i] == 0
                            && if air {
                                phase == 0
                            } else {
                                matches!(phase, 1 | 2 | 4)
                            })
                        .then_some(temps[i])
                    })
                    .collect();
                assert!(!selected.is_empty());
                selected.iter().sum::<f32>() / selected.len() as f32
            };
            air_samples.push(mean(true));
            water_samples.push(mean(false));
            most_frozen = most_frozen.max(states.iter().filter(|s| **s & 7 == 2).count());
        }
    }
    let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
    let range = |values: &[f32]| {
        (
            values.iter().copied().fold(f32::INFINITY, f32::min),
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    };
    eprintln!("whole orbit, {N}^3: air mean {:.1} C, range {:?}; water mean {:.1} C, range {:?}; peak ice {most_frozen}/{volume}", mean(&air_samples), range(&air_samples), mean(&water_samples), range(&water_samples));
    assert!(
        (15.555..=32.223).contains(&mean(&air_samples)),
        "bulk atmosphere must be temperate at brightness 3"
    );
    assert!(
        (15.555..=32.223).contains(&mean(&water_samples)),
        "water's full-orbit average must be 60-90 F"
    );
    assert!(
        range(&air_samples).0 > 15.555,
        "nightfall must not freeze the entire atmosphere"
    );
    if N == 16 {
        assert!(
            most_frozen == 0,
            "shadows must not freeze water while the atmosphere stays temperate"
        );
    }
    assert_eq!(climate.phase_counts()[1..].iter().sum::<usize>(), volume);
}

#[test]
fn thermal_masses_track_a_six_minute_sun_orbit() {
    let mut climate = Climate::new();
    // Each full thermal sweep represents 4/60 s. The measured time to
    // absorb 63% of a step checks physical timing, not just eventual targets.
    for (name, medium, seconds, solid) in [
        ("air", 0, 4, false),
        ("water", 1, 72, false),
        ("ice", 2, 93, false),
        ("steam", 3, 4, false),
        ("snow", 4, 7, false),
        ("rock", 0, 33, true),
    ] {
        climate.reset(medium, -18.0, 1.0);
        if solid {
            climate.write(2, &vec![1; COUNT]);
        }
        climate.run("thermal", seconds * 15);
        let target = if matches!(medium, 1 | 2) { 23.0 } else { 30.0 };
        let fraction = (climate.mean_temperature() + 18.0) / (target + 18.0);
        assert!(
            (0.60..0.66).contains(&fraction),
            "{name}: {fraction:.3} response after {seconds}s"
        );
    }

    for brightness in [2.8, 3.0, 3.2] {
        climate.params.sun_brightness = brightness;
        climate.reset(1, 18.0, 0.0);
        climate.run("thermal", 3 * 15);
        let brief_shadow = climate.temperatures();
        eprintln!(
            "3 seconds of shade: mean {:.1} C, minimum {:.1} C",
            climate.mean_temperature(),
            brief_shadow.iter().copied().fold(f32::INFINITY, f32::min)
        );
        assert!(
            brief_shadow.iter().all(|t| *t > 0.0),
            "a brief shadow must not freeze warm water"
        );
        climate.run("thermal", 177 * 15);
        let night = climate.mean_temperature();
        assert!(
            (10.0..30.0).contains(&night),
            "shaded water must stay temperate while the atmosphere is warm: {night}"
        );
        climate.write(5, &vec![1.0f32.to_bits(); COUNT]);
        climate.run("thermal", 180 * 15);
        let day = climate.mean_temperature();
        assert!(
            (15.0..35.0).contains(&day),
            "sun {brightness}: daylight must keep water temperate without boiling: {day}"
        );
        eprintln!("1 degree/s, brightness {brightness}: exposed water {night:.1}..{day:.1} C before latent heat");
    }
    climate.params.sun_brightness = 5.0;
    climate.reset(1, 18.0, 0.4); // hot climate with cloud shading
    climate.run("thermal", 120 * 15);
    assert!(
        (45.0..70.0).contains(&climate.mean_temperature()),
        "brightness 5 should warm shaded water without forcing it to boil: {} C",
        climate.mean_temperature()
    );
}

#[test]
fn upper_sun_brightness_warms_air_gradually_without_hitting_the_temperature_cap() {
    let mut climate = Climate::new();
    for sunlight in [0.0, 0.4, 1.0] {
        let mut previous = None;
        for step in 0..=8 {
            let brightness = 3.0 + step as f32 * 0.25;
            climate.params.sun_brightness = brightness;
            climate.reset(0, 30.0, sunlight);
            climate.run("thermal", 120 * 15);
            let air = climate.mean_temperature();
            if let Some(last) = previous {
                assert!(air > last && air - last < 8.1,
                    "brightness {brightness}, sunlight {sunlight}: abrupt air change {last} -> {air} C");
            }
            if step == 0 {
                assert!(
                    (23.5..30.5).contains(&air),
                    "brightness 3 must remain temperate: {air}"
                );
            } else if step == 8 {
                assert!(
                    (67.0..80.5).contains(&air),
                    "brightness 5 air should be hot without reaching the cap: {air}"
                );
                if sunlight == 0.4 {
                    assert!(
                        (70.0..80.0).contains(&air),
                        "cloud-shaded air must meet the requested 70-80 C range: {air}"
                    );
                }
            }
            eprintln!("sun {brightness}, sunlight {sunlight}: air {air:.1} C");
            previous = Some(air);
        }
    }
}

#[test]
fn brightness_three_clouds_produce_substantial_rain_within_five_seconds() {
    let mut climate = Climate::new();
    climate.params.sun_brightness = 3.0;
    climate.reset(3, 27.0, 1.0);
    for _ in 0..5 * 15 {
        // Both climate passes run on the same 15 Hz tick. Use the actual
        // steam fog source, rather than an artificial humidity budget.
        let tick = climate.tick;
        climate.run("fog", 1);
        climate.tick = tick;
        climate.run("weather", 1);
    }
    let counts = climate.phase_counts();
    eprintln!("brightness 3 cloud after 5s: {counts:?}");
    assert!(
        counts[1] >= COUNT / 2,
        "temperate clouds should release at least half their vapor as real rain within five seconds: {counts:?}"
    );
    assert_eq!(counts[1..].iter().sum::<usize>(), COUNT);
    assert_eq!(counts[2] + counts[4], 0);
}

#[test]
fn clouds_can_rain_and_snow_at_attainable_fog_density() {
    let mut climate = Climate::new();
    for (temperature, expected_phase, sweeps) in [(18.0, 1, 900), (-5.0, 4, 120)] {
        climate.reset(3, temperature, 0.0);
        climate.write(6, &vec![120 * 256; COUNT]);
        climate.run("weather", sweeps);
        let counts = climate.phase_counts();
        assert!(
            counts[expected_phase] > COUNT / 5,
            "cloud at {temperature} C failed to precipitate: {counts:?}"
        );
        assert_eq!(
            counts[1] + counts[3] + counts[4],
            COUNT,
            "precipitation must conserve water"
        );
        assert_eq!(counts[if expected_phase == 1 { 4 } else { 1 }], 0);
        if temperature < 0.0 {
            assert!(
                climate.temperatures().iter().all(|t| *t <= 0.0),
                "deposition must not create warm snow"
            );
        }
    }
    climate.reset(3, -18.0, 0.0); // no fog budget or surfaces required in deep cold
    climate.run("movement", 1);
    assert!(
        climate.phase_counts()[3] > COUNT * 9 / 10,
        "deeply cold vapor must linger rather than all turning into snow immediately"
    );
    climate.run("movement", 180);
    let counts = climate.phase_counts();
    assert!(
        counts[2] + counts[4] > COUNT / 4,
        "cold vapor must still deposit: {counts:?}"
    );
    assert!(
        counts[3] > 0,
        "some cold vapor must remain airborne: {counts:?}"
    );
    assert_eq!(counts[1..].iter().sum::<usize>(), COUNT);
}

#[test]
fn liquid_evaporation_has_a_strong_cold_baseline_and_increases_with_temperature() {
    let mut climate = Climate::new();
    let temperatures = [-49.0, -18.0, 0.0, 5.0, 20.0, 40.0, 70.0, 99.0];
    climate.reset(0, 0.0, 1.0);
    climate.write(
        8,
        &(0..COUNT)
            .map(|idx| encode(temperatures[idx % temperatures.len()]))
            .collect::<Vec<_>>(),
    );
    for humidity in [0, 255 * 256] {
        climate.write(6, &vec![humidity; COUNT]);
        climate.run("evaporation", 1);
        let rates: Vec<_> = climate.read(3)[..temperatures.len()]
            .iter()
            .map(|raw| f32::from_bits(*raw))
            .collect();
        assert!(
            rates.iter().all(|rate| *rate >= 0.0014),
            "cold/humid source too faint: {rates:?}"
        );
        assert!(
            rates.windows(2).all(|pair| pair[1] >= pair[0]),
            "temperature must never reduce evaporation: {rates:?}"
        );
        assert!(
            rates[6] > rates[2] * 2.0,
            "hot water must emit substantially more than cold water: {rates:?}"
        );
    }
}

#[test]
fn cold_liquid_keeps_visible_vapor_in_the_air_across_successive_seconds() {
    let mut climate = ClimateGrid::<16>::new();
    climate.params.gravity_mode = 1;
    // Keep liquid available at each tested temperature. Freezing has its own
    // regressions; a fully frozen source no longer represents liquid water.
    climate.params.freeze_rate = 0.0;
    climate.params.humidity_diffusion_rate = 0.15;
    let idx = |x, y, z| (x + y * 16 + z * 256) as usize;
    let mut solids = vec![0; 16 * 16 * 16];
    let mut pool = vec![0; 16 * 16 * 16];
    for z in 0..16 {
        for x in 0..16 {
            solids[idx(x, 0, z)] = 1;
            pool[idx(x, 1, z)] = 0xffff0001;
            pool[idx(x, 2, z)] = 0xffff0001;
        }
    }
    let volume = 16 * 16 * 2;
    for temperature in [-49.0, -18.0, 0.0, 20.0] {
        climate.reset(0, temperature, 1.0);
        climate.tick = 0;
        climate.write(1, &pool);
        climate.write(2, &solids);
        for second in 1..=8 {
            climate.fluid_seconds(1, false);
            let counts = climate.phase_counts();
            assert_eq!(counts[1..].iter().sum::<usize>(), volume);
            assert!(
                counts[1] >= volume / 2,
                "the pool's bottom layer must survive at brightness 3: {counts:?}"
            );
            if second >= 2 {
                assert!(
                    counts[3] > 0,
                    "{temperature} C, second {second}: {counts:?}"
                );
                assert!(
                    climate.visible_vapor() > 0,
                    "{temperature} C, second {second}: visible wisps required"
                );
            }
        }
    }
}

#[test]
fn only_liquid_water_supplies_vapor_at_every_temperature() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    climate.params.sun_brightness = 4.0;
    let center = index(4, 4, 4);
    let tick = evaporation_tick(0..8);
    for (phase, temperature, evaporates) in [
        (1, 5.0, true),
        (1, 0.0, true),
        (1, -18.0, true),
        (1, -49.0, true),
        (2, -5.0, false),
        (4, -5.0, false),
        (2, -18.0, false),
    ] {
        climate.reset(0, temperature, 0.0);
        climate.tick = tick;
        let mut states = vec![0; COUNT];
        states[center] = 0xffff0000 | phase;
        climate.write(1, &states);
        let mut solids = vec![0; COUNT];
        solids[index(4, 3, 4)] = 1;
        climate.write(2, &solids);
        climate.run("movement", 1);
        let counts = climate.phase_counts();
        assert_eq!(
            counts[3],
            usize::from(evaporates),
            "phase {phase}, {temperature} C: {counts:?}"
        );
        assert_eq!(counts[1..].iter().sum::<usize>(), 1);
    }

    // Saturated air slows evaporation but cannot switch liquid vapor off.
    climate.reset(0, 0.0, 0.0);
    climate.tick = tick;
    let mut states = vec![0; COUNT];
    states[center] = 0xffff0001;
    climate.write(1, &states);
    let mut solids = vec![0; COUNT];
    solids[index(4, 3, 4)] = 1;
    climate.write(2, &solids);
    climate.write(6, &vec![255 * 256; COUNT]);
    climate.run("movement", 1);
    assert_eq!(climate.phase_counts()[3], 1);

    // Exposed falling droplets are liquid too: support is needed for
    // freezing, but must not gate their evaporation.
    climate.reset(0, 0.0, 0.0);
    climate.tick = tick;
    climate.write(1, &states);
    climate.run("movement", 1);
    assert_eq!(climate.phase_counts()[3], 1);
}

#[test]
fn warm_water_emits_vapor_on_draws_that_cold_water_does_not() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    climate.params.sun_brightness = 4.0;
    let center = index(4, 4, 4);
    let tick = evaporation_tick(65_000..67_000);
    for (temperature, evaporates) in [(0.0, false), (5.0, false), (40.0, true), (70.0, true)] {
        climate.reset(0, temperature, 0.0);
        climate.tick = tick;
        let mut states = vec![0; COUNT];
        states[center] = 0xffff0001;
        let mut solids = vec![0; COUNT];
        solids[index(4, 3, 4)] = 1;
        climate.write(1, &states);
        climate.write(2, &solids);
        climate.run("movement", 1);
        assert_eq!(climate.phase_counts()[3], usize::from(evaporates));
        assert_eq!(climate.phase_counts()[1..].iter().sum::<usize>(), 1);
    }
}

#[test]
fn pools_keep_their_last_liquid_layer_until_brightness_four() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    let source = index(4, 4, 4);
    let mut states = vec![0; COUNT];
    states[source] = 0xffff0001;
    // Enclose one last-layer liquid cell so movement cannot obscure the
    // brightness gate. Probe both passive evaporation and boiling.
    let mut solids = vec![1; COUNT];
    for y in 4..7 {
        solids[index(4, y, 4)] = 0;
    }
    for brightness in [0.0, 3.0, 3.999, 4.0, 5.0] {
        climate.params.sun_brightness = brightness;
        for temperature in [0.0, 25.0, 110.0] {
            climate.reset(0, temperature, 0.0);
            climate.tick = evaporation_tick(0..8);
            climate.write(1, &states);
            climate.write(2, &solids);
            climate.run("movement", 1);
            let counts = climate.phase_counts();
            assert_eq!(counts[1..].iter().sum::<usize>(), 1);
            assert_eq!(
                counts[3],
                usize::from(brightness >= 4.0),
                "brightness {brightness}, {temperature} C: {counts:?}"
            );
            if brightness < 4.0 {
                climate.run("movement", 600);
                assert_eq!(
                    climate.read(1)[source] & 7,
                    1,
                    "the reserve must survive repeated evaporation and boiling attempts"
                );
            }
        }
    }
}

#[test]
fn pool_reserve_follows_axis_and_radial_gravity() {
    let mut climate = Climate::new();
    let source = index(4, 4, 4);
    for (gravity, axis) in [(0, 0), (1, 1), (2, 2), (3, 2)] {
        climate.params.gravity_mode = gravity;
        // The radial source is at (0.5, 0.5, 0.5); the simulation's
        // deterministic gravity tie-break chooses its Z direction.
        let mut solids = vec![1; COUNT];
        for coordinate in 4..RES {
            let mut pos = [4, 4, 4];
            pos[axis] = coordinate;
            solids[index(pos[0], pos[1], pos[2])] = 0;
        }
        for brightness in [3.999, 4.0] {
            climate.params.sun_brightness = brightness;
            climate.reset(0, 110.0, 0.0);
            climate.tick = evaporation_tick(0..8);
            let mut states = vec![0; COUNT];
            states[source] = 0xffff0001;
            climate.write(1, &states);
            climate.write(2, &solids);
            climate.run("movement", 128);
            let counts = climate.phase_counts();
            assert_eq!(counts[1..].iter().sum::<usize>(), 1);
            assert_eq!(
                counts[1],
                usize::from(brightness < 4.0),
                "gravity {gravity}, brightness {brightness}: {counts:?}"
            );
        }
    }
}

#[test]
fn freezing_point_liquid_supplies_vapor_that_condenses_into_rain() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    climate.params.freeze_rate = 0.0;
    climate.reset(0, 0.0, 1.0);
    climate.tick = evaporation_tick(0..8);
    let source = index(4, 4, 4);
    let mut states = vec![0; COUNT];
    states[source] = 0xffff0001;
    states[index(4, 3, 4)] = 0xffff0001;
    let mut solids = vec![1; COUNT];
    for y in 3..RES {
        solids[index(4, y, 4)] = 0;
    }
    climate.write(1, &states);
    climate.write(2, &solids);
    climate.run("movement", 1);
    // Newly spawned vapor can already rise during the in-place motion pass.
    let vapor = climate
        .read(1)
        .iter()
        .position(|state| *state & 7 == 3)
        .unwrap();
    assert!(climate.visible_vapor() > 0);
    // Use the real vapor fog source, without seeding an artificial humidity
    // budget. Cold liquid must feed the complete conserved rain cycle.
    climate.run("fog", 1);
    assert!(climate.read(6)[vapor] > 0);
    climate.run("weather", 600);
    assert_eq!(climate.read(1)[vapor], 0xffff0001);
    assert_eq!(climate.phase_counts()[1..].iter().sum::<usize>(), 2);
    for _ in 0..128 {
        climate.run("movement", 1);
        let states = climate.read(1);
        if states[source] & 7 == 1 && states[index(4, 3, 4)] & 7 == 1 {
            break;
        }
    }
    assert_eq!(
        climate.read(1)[source] & 7,
        1,
        "rain must return to the pool surface"
    );
    assert_eq!(
        climate.read(1)[index(4, 3, 4)] & 7,
        1,
        "the liquid reserve must survive"
    );
    climate.tick = evaporation_tick(0..8);
    climate.run("movement", 1);
    let counts = climate.phase_counts();
    assert_eq!(
        counts[3], 1,
        "rain must recharge the next real evaporation cycle"
    );
    assert_eq!(counts[1..].iter().sum::<usize>(), 2);
}

#[test]
fn temperate_pool_keeps_raining_and_brightness_five_warms_it_without_boiling() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    climate.params.sun_brightness = 5.0;
    climate.params.humidity_diffusion_rate = 0.15;
    climate.reset(0, 18.0, 0.4);
    let mut solids = vec![0; COUNT];
    let mut states = vec![0; COUNT];
    for z in 0..RES {
        for x in 0..RES {
            solids[index(x, 0, z)] = 1;
            states[index(x, 1, z)] = 0xffff0001;
            states[index(x, 2, z)] = 0xffff0001;
        }
    }
    climate.write(1, &states);
    climate.write(2, &solids);
    let water_volume = (RES * RES * 2) as usize;
    climate.params.sun_brightness = 3.0;
    climate.write(5, &vec![1.0f32.to_bits(); COUNT]);
    climate.dynamic_seconds(10);
    let temperate = climate.phase_counts();
    eprintln!("brightness 3 after 10s sunlight: {temperate:?}");
    assert!(
        temperate[3] > water_volume / 10,
        "temperate evaporation must supply significant vapor within ten seconds"
    );
    assert!(
        temperate[1] >= water_volume / 2,
        "persistent temperate vapor must preserve the pool's bottom layer"
    );
    assert!(
        climate.visible_vapor() > 0,
        "evaporation must produce visible above-surface wisps"
    );
    climate.write(5, &vec![0; COUNT]);
    climate.dynamic_seconds(60);
    let night = climate.phase_counts();
    eprintln!("brightness 3 after 60s shade: {night:?}");
    assert_eq!(night[1..].iter().sum::<usize>(), water_volume);
    assert!(
        night[2] == 0 && night[1] >= water_volume / 2,
        "a sustained shadow must preserve the pool's bottom layer when the air is warm"
    );
    climate.write(5, &vec![1.0f32.to_bits(); COUNT]);
    climate.dynamic_seconds(45);
    let day = climate.phase_counts();
    eprintln!("brightness 3 after 45s sunlight: {day:?}");
    assert_eq!(day[1..].iter().sum::<usize>(), water_volume);
    assert!(
        day[1] >= water_volume / 2,
        "daylight must melt ice and preserve the pool's bottom layer alongside significant vapor: {day:?}"
    );
    climate.params.sun_brightness = 5.0;
    climate.write(5, &vec![0.4f32.to_bits(); COUNT]);
    climate.dynamic_seconds(180);
    let counts = climate.phase_counts();
    eprintln!("brightness 5, 40% sunlight, after 180s: {counts:?}");
    assert_eq!(
        counts[1..].iter().sum::<usize>(),
        water_volume,
        "phase cycling must conserve total water"
    );
    assert!(counts[3] > water_volume / 2 && counts[1] < water_volume / 2,
        "brightness 5 must permit depletion past the protected pool reserve through real evaporation: {counts:?}");
    let temperatures = climate.temperatures();
    let states = climate.read(1);
    let air: Vec<_> = states
        .iter()
        .zip(&temperatures)
        .filter_map(|(state, temperature)| matches!(state & 7, 0 | 3).then_some(*temperature))
        .collect();
    let air_mean = air.iter().sum::<f32>() / air.len() as f32;
    eprintln!("brightness 5 pool atmosphere: {air_mean:.1} C");
    assert!(
        (70.0..80.0).contains(&air_mean),
        "the pool's atmosphere must stay near the requested hot climate: {air_mean}"
    );
    assert!(
        temperatures.iter().all(|t| *t < 90.0),
        "brightness 5 must not force boiling"
    );
    assert_eq!(counts[2] + counts[4], 0, "no ice or snow in this hot world");
}

#[test]
fn mixed_hot_and_cold_media_stay_bounded_across_workgroups_and_slices() {
    let mut climate = Climate::new();
    for entry in ["thermal", "sliced"] {
        for inertia in [0.0, 4.0, 5.0] {
            climate.reset(0, -10.0, 0.0);
            climate.params.thermal_inertia = inertia;
            climate.params.sun_brightness = 0.0;
            climate.write(
                1,
                &(0..COUNT)
                    .map(|i| 0xffff0000 | (i % 5) as u32)
                    .collect::<Vec<_>>(),
            );
            climate.write(
                2,
                &(0..COUNT)
                    .map(|i| u32::from(i % 7 == 0))
                    .collect::<Vec<_>>(),
            );
            climate.write(
                8,
                &(0..COUNT)
                    .map(|i| encode(if i % 3 == 0 { 40.0 } else { -10.0 }))
                    .collect::<Vec<_>>(),
            );
            for ticks in [1, 63] {
                climate.run(entry, ticks);
                let temps = climate.temperatures();
                let min = temps.iter().copied().fold(f32::INFINITY, f32::min);
                let max = temps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    min >= -21.01 && max <= 40.01,
                    "{entry}, inertia {inertia}, after {ticks} passes: heat exchange invented extremes {min}..{max} C"
                );
            }
        }
    }
}

#[test]
fn conduction_matches_pair_distances_and_conserves_heat_in_buried_rock() {
    let mut climate = Climate::new();
    climate.reset(0, -10.0, 0.0);
    climate.params.thermal_inertia = 0.0;
    climate.write(2, &vec![1; COUNT]);
    let center = index(4, 4, 4);
    let mut temperatures = vec![encode(-10.0); COUNT];
    temperatures[center] = encode(40.0);
    climate.write(8, &temperatures);
    let mut expected = temperatures.clone();
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let distance_sq = dx * dx + dy * dy + dz * dz;
                if distance_sq == 0 {
                    continue;
                }
                // Rock capacity 10; pair conductance is 0.286 per sweep at
                // minimum inertia, below the stability cap for this mass.
                let delta = (50.0 * (0.286 / 10.0) / distance_sq as f32 * 256.0).round() as u32;
                expected[index((4 + dx) as u32, (4 + dy) as u32, (4 + dz) as u32)] += delta;
                expected[center] -= delta;
            }
        }
    }
    climate.run("thermal", 1);
    for (i, (actual, expected)) in climate.read(8).iter().zip(&expected).enumerate() {
        assert_eq!(
            actual, expected,
            "voxel {i}: each neighbor must receive only its distance-weighted heat transfer"
        );
    }
    climate.run("thermal", 63);
    assert_eq!(
        climate.read(8).iter().map(|v| *v as u64).sum::<u64>(),
        temperatures.iter().map(|v| *v as u64).sum::<u64>(),
        "conduction in closed rock must conserve total heat"
    );
}

#[test]
fn sun_response_is_monotonic_for_air_water_and_rock() {
    let mut climate = Climate::new();
    for inertia in [0.0, 4.0] {
        climate.params.thermal_inertia = inertia;
        for medium in [0, 1, 2, 3, 4] {
            let mut previous = f32::NEG_INFINITY;
            for brightness in [0.0, 3.0, 5.0] {
                climate.reset(medium, -10.0, 1.0);
                climate.params.sun_brightness = brightness;
                // Allow over six ice response times at default inertia so
                // this measures equilibrium rather than the slower warmup.
                climate.run("thermal", 9000);
                let temps = climate.temperatures();
                let mean = temps.iter().sum::<f32>() / COUNT as f32;
                let target = match brightness {
                    0.0 => {
                        if matches!(medium, 1 | 2) {
                            -21.0
                        } else {
                            -18.0
                        }
                    }
                    3.0 => {
                        if matches!(medium, 1 | 2) {
                            23.0
                        } else {
                            30.0
                        }
                    }
                    5.0 => {
                        if matches!(medium, 1 | 2) {
                            68.83333
                        } else {
                            80.0
                        }
                    }
                    _ => unreachable!(),
                };
                assert!(
                    (mean - target).abs() < 1.0,
                    "medium {medium}, inertia {inertia}, sun {brightness}: {mean} C vs {target} C"
                );
                assert!(
                    mean > previous,
                    "increasing sunlight must increase temperature"
                );
                assert!(
                    temps.iter().all(|t| *t <= target + 1.0),
                    "medium {medium}, inertia {inertia}, sun {brightness}: max {:?} vs {target}",
                    temps.iter().copied().reduce(f32::max)
                );
                previous = mean;
            }
        }
    }

    // An isolated exposed boulder must absorb sunlight even before its air
    // neighbors are warmed. Its greater heat capacity delays its response.
    climate.reset(0, -18.0, 0.0);
    climate.params.thermal_inertia = 4.0;
    climate.params.sun_brightness = 3.0;
    let center = index(4, 4, 4);
    let mut solids = vec![0; COUNT];
    solids[center] = 1;
    climate.write(2, &solids);
    let mut sunlight = vec![0; COUNT];
    sunlight[center] = 1.0f32.to_bits();
    climate.write(5, &sunlight);
    climate.run("thermal", 1);
    assert!(
        climate.temperatures()[center] > -18.0,
        "sunlit rock must absorb heat"
    );
    climate.run("thermal", 128);
    let temperatures = climate.temperatures();
    assert!(
        temperatures[index(4, 4, 3)] > -18.0,
        "rock must warm neighboring shaded air"
    );
    climate.params.sun_brightness = 0.0;
    climate.run("thermal", 128);
    assert!(
        climate.temperatures()[center] < temperatures[center],
        "rock must cool after the sun goes out"
    );
}

#[test]
fn warm_snow_melts_in_flight_and_on_rock_and_hot_steam_cannot_rain() {
    let mut climate = Climate::new();
    climate.params.gravity_mode = 1;
    for on_rock in [false, true] {
        climate.reset(0, 40.0, 0.0);
        let mut states = vec![0; COUNT];
        states[index(4, 6, 4)] = 0xffff0004;
        climate.write(1, &states);
        if on_rock {
            let mut solids = vec![0; COUNT];
            solids[index(4, 5, 4)] = 1;
            climate.write(2, &solids);
        }
        climate.run("movement", 1);
        let states = climate.read(1);
        assert_eq!(
            states.iter().filter(|s| **s & 7 == 1).count(),
            1,
            "warm snow should become water, rock={on_rock}"
        );
        assert_eq!(
            states.iter().filter(|s| matches!(**s & 7, 2 | 4)).count(),
            0
        );
    }
    // Warm enclosed snow must still melt when the motion fast path skips it.
    climate.reset(2, 40.0, 0.0);
    let mut states = vec![0xffff0002; COUNT];
    states[index(4, 4, 4)] = 0xffff0004;
    climate.write(1, &states);
    climate.run("movement", 1);
    assert_eq!(climate.read(1)[index(4, 4, 4)] & 7, 1);

    climate.reset(3, 120.0, 0.0);
    climate.write(6, &vec![255 * 256; COUNT]);
    climate.run("weather", 256);
    assert!(
        climate.read(1).iter().all(|state| state & 7 == 3),
        "steam above boiling must not condense into precipitation"
    );
}

#[test]
fn static_water_preserves_rock_heat_and_evaporation_preserves_displaced_air_heat() {
    let mut climate = Climate::new();
    climate.reset(1, 18.0, 0.0);
    let center = index(4, 4, 4);
    let mut solids = vec![0; COUNT];
    solids[center] = 1;
    climate.write(2, &solids);
    let mut temperatures = vec![encode(18.0); COUNT];
    temperatures[center] = encode(40.0);
    climate.write(8, &temperatures);
    climate.run("static", 64);
    assert_eq!(
        climate.read(8),
        temperatures,
        "static water must not reset rock temperatures each climate sweep"
    );

    climate.reset(0, 30.0, 0.0);
    climate.params.gravity_mode = 1;
    climate.params.sun_brightness = 4.0;
    let mut states = vec![0; COUNT];
    states[center] = 0xffff0001;
    climate.write(1, &states);
    let mut solids = vec![0; COUNT];
    solids[index(4, 3, 4)] = 1;
    climate.write(2, &solids);
    let mut temperatures = vec![encode(30.0); COUNT];
    temperatures[center] = encode(110.0);
    climate.write(8, &temperatures);
    climate.run("movement", 128);
    assert_eq!(
        climate
            .read(1)
            .iter()
            .filter(|state| **state & 7 == 3)
            .count(),
        1,
        "the boiling water must evaporate"
    );
    assert!(
        climate
            .temperatures()
            .iter()
            .all(|temperature| *temperature >= 30.0),
        "evaporation must not invent a cold starting-temperature air pocket"
    );
}

#[test]
fn exterior_cube_is_excluded_from_heat_exchange_and_temperature_statistics() {
    let mut climate = Climate::new();
    climate.reset(0, 30.0, 1.0);
    climate.params.world_radius = 3.0;
    let mut temperatures = vec![encode(30.0); COUNT];
    let mut inside = 0;
    for z in 0..RES {
        for y in 0..RES {
            for x in 0..RES {
                let p = [x, y, z].map(|v| v as f32 - 3.5);
                if p.iter().map(|v| v * v).sum::<f32>() < 9.0 {
                    inside += 1;
                } else {
                    temperatures[index(x, y, z)] = encode(150.0);
                }
            }
        }
    }
    climate.write(8, &temperatures);
    climate.run("thermal", 1);
    let stats = climate.read(4);
    assert_eq!(stats[3], inside);
    assert_eq!(stats[2], inside * 80);
    assert_eq!(
        climate.read(8),
        temperatures,
        "exterior containment must not heat the world"
    );
}
