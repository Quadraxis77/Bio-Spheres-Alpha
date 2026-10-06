//! Batched live-graph diffusion. The legacy tree solver remains benchmark-only.
use super::adhesion::MAX_ADHESIONS_PER_CELL;
use super::{AdhesionBuffers, GpuTripleBufferSystem};
use crate::simulation::signal_diffusion::DiffusionSettings;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DiffusionParams {
    pub count: u32,
    pub tick: u32,
    pub degree: u32,
    pub resolution: u32,
    pub dt: f32,
    pub conductance: f32,
    pub retention: f32,
    pub production_scale: f32,
    pub time: f32,
    pub radius: f32,
    pub grid_cell: f32,
    pub padding: f32,
    pub grid_origin: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct DiffusionCellState {
    pub identity: u32,
    pub mode: u32,
    pub config_hash: u32,
    pub live: u32,
    pub memory: f32,
    pub output: f32,
    pub channel: u32,
    pub padding: u32,
}
/// Immutable authored signal settings. signal_settings_v4.z stores the table
/// reference; GPU mutation's existing mode-copy path preserves that reference.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct DiffusionMode {
    photo: [f32; 4],
    lipo: [f32; 4],
    values: [f32; 4],
    processor: [u32; 4],
    oscillator: [f32; 4],
    light_filter: [f32; 4],
}
impl From<&crate::genome::ModeSettings> for DiffusionMode {
    fn from(m: &crate::genome::ModeSettings) -> Self {
        let memo = m.cell_type == 15;
        Self {
            photo: [
                m.photocyte_emit_enabled as u32 as f32,
                m.photocyte_emit_channel.clamp(0, 15) as f32,
                m.photocyte_emit_threshold,
                m.photocyte_emit_mode as f32,
            ],
            lipo: [
                m.lipocyte_emit_enabled as u32 as f32,
                m.lipocyte_emit_channel.clamp(0, 15) as f32,
                m.lipocyte_emit_threshold,
                m.lipocyte_emit_mode as f32,
            ],
            values: [
                m.photocyte_emit_value.max(0.0),
                m.lipocyte_emit_value.max(0.0),
                m.oculocyte_signal_value.max(0.0),
                m.cognocyte_oscillator_polarity as f32,
            ],
            processor: [
                m.cognocyte_operation as u32,
                if memo {
                    m.memorocyte_input_channel
                } else {
                    m.cognocyte_input_channel_a
                }
                .clamp(0, 15) as u32,
                m.cognocyte_input_channel_b.clamp(0, 15) as u32,
                if memo {
                    m.memorocyte_output_channel
                } else {
                    m.cognocyte_output_channel
                }
                .clamp(0, 15) as u32,
            ],
            oscillator: [
                m.cognocyte_oscillator_rate,
                m.cognocyte_oscillator_phase,
                m.cognocyte_oscillator_strength.max(0.0),
                m.memorocyte_rate,
            ],
            light_filter: [
                m.oculocyte_light_target_color.x,
                m.oculocyte_light_target_color.y,
                m.oculocyte_light_target_color.z,
                m.oculocyte_light_color_tolerance,
            ],
        }
    }
}
pub fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
fn pipeline(device: &wgpu::Device, source: &str, entry: &str) -> wgpu::ComputePipeline {
    let source = format!(
        "{}\n{}",
        include_str!("../../../shaders/signal_diffusion_common.wgsl"),
        source
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
fn bind(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Diffusion Bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

pub struct SignalDiffusionPipeline {
    pub settings: DiffusionSettings,
    pub concentrations: [wgpu::Buffer; 2],
    pub production: wgpu::Buffer,
    pub states: wgpu::Buffer,
    pub modes: wgpu::Buffer,
    pub dummy: wgpu::Buffer,
    params: [wgpu::Buffer; 4],
    transport: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    dispatch: wgpu::Buffer,
    sources: wgpu::ComputePipeline,
    receivers: wgpu::ComputePipeline,
    spatial: wgpu::Buffer,
    activity: wgpu::Buffer,
    current: usize,
    mode_data: Vec<DiffusionMode>,
}
impl SignalDiffusionPipeline {
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        Self {
            settings: Default::default(),
            activity: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Signal Field Activity"),
                // Standalone transport starts enabled; live ticks reduce this on GPU.
                contents: bytemuck::cast_slice(&[1u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            concentrations: std::array::from_fn(|_| {
                storage(device, "Signal Concentration", capacity as u64 * 64)
            }),
            production: storage(device, "Signal Production", capacity as u64 * 64),
            states: storage(device, "Signal Cell State", capacity as u64 * 32),
            modes: storage(device, "Signal Authored Modes", 96),
            dummy: storage(device, "Absent Signal Sensor Field", 16),
            params: std::array::from_fn(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Signal Tick Params"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }),
            spatial: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Sensor Grid"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            prepare: pipeline(
                device,
                include_str!("../../../shaders/signal_diffusion_dispatch.wgsl"),
                "prepare_dispatch",
            ),
            dispatch: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Signal Dispatch Sizes"),
                size: 24,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            transport: pipeline(
                device,
                include_str!("../../../shaders/signal_diffusion.wgsl"),
                "transport",
            ),
            sources: pipeline(
                device,
                include_str!("../../../shaders/signal_diffusion_sources.wgsl"),
                "sources",
            ),
            receivers: pipeline(
                device,
                include_str!("../../../shaders/signal_diffusion_receivers.wgsl"),
                "receivers",
            ),
            current: 0,
            mode_data: Vec::new(),
        }
    }
    pub fn sync_modes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        genomes: &[crate::genome::Genome],
    ) {
        let data: Vec<_> = std::iter::once(DiffusionMode::default())
            .chain(
                genomes
                    .iter()
                    .flat_map(|g| g.modes.iter().map(DiffusionMode::from)),
            )
            .collect();
        if bytemuck::cast_slice::<_, u8>(&data) == bytemuck::cast_slice::<_, u8>(&self.mode_data) {
            return;
        }
        if self.modes.size() < data.len() as u64 * 96 {
            self.modes = storage(device, "Signal Authored Modes", data.len() as u64 * 96);
        }
        queue.write_buffer(&self.modes, 0, bytemuck::cast_slice(&data));
        self.mode_data = data;
    }
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        // Invalidate lifecycle state; the next source stage clears both old and
        // newly live compartments before any neighbor can read them.
        queue.write_buffer(&self.states, 0, &vec![0; self.states.size() as usize]);
        self.current = 0;
    }
    pub fn current_field(&self) -> &wgpu::Buffer {
        &self.concentrations[self.current]
    }
    pub fn write_params(&self, queue: &wgpu::Queue, slot: usize, params: DiffusionParams) {
        DiffusionSettings {
            conductance: params.conductance,
            decay_rate: 0.0,
            production_scale: params.production_scale,
        }
        .validate(params.dt, MAX_ADHESIONS_PER_CELL)
        .expect("invalid signal diffusion settings");
        assert!(params.retention.is_finite() && (0.0..=1.0).contains(&params.retention));
        assert_eq!(params.degree, MAX_ADHESIONS_PER_CELL as u32);
        assert!(params.count as u64 * 64 <= self.concentrations[0].size());
        queue.write_buffer(&self.params[slot], 0, bytemuck::bytes_of(&params));
    }
    pub fn encode_transport(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        count: u32,
        edges: &wgpu::Buffer,
        adjacency: &wgpu::Buffer,
    ) {
        self.encode_transport_impl(device, encoder, slot, count, edges, adjacency, false);
    }
    fn encode_transport_impl(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        count: u32,
        edges: &wgpu::Buffer,
        adjacency: &wgpu::Buffer,
        indirect: bool,
    ) {
        let binds = bind(
            device,
            &self.transport,
            &[
                &self.params[slot],
                &self.concentrations[self.current],
                &self.concentrations[1 - self.current],
                &self.production,
                edges,
                adjacency,
                &self.states,
                &self.activity,
            ],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Transport + Decay + Production"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transport);
            pass.set_bind_group(0, &binds, &[]);
            if indirect {
                pass.dispatch_workgroups_indirect(&self.dispatch, 12);
            } else {
                pass.dispatch_workgroups(count.div_ceil(128), 1, 1);
            }
        }
        self.current = 1 - self.current;
    }
    #[allow(clippy::too_many_arguments)]
    pub fn encode_tick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        params: DiffusionParams,
        b: &GpuTripleBufferSystem,
        a: &AdhesionBuffers,
        light: Option<&super::LightFieldSystem>,
        fluid: Option<&crate::simulation::fluid_simulation::GpuFluidSimulator>,
        moss: Option<&super::moss::MossSystem>,
        grid: [f32; 4],
    ) {
        self.write_params(queue, slot, params);
        queue.write_buffer(&self.spatial, 0, bytemuck::cast_slice(&grid));
        let prepare_bind = bind(
            device,
            &self.prepare,
            &[&self.params[slot], &b.cell_count_buffer, &self.dispatch],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Dispatch Setup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare);
            pass.set_bind_group(0, &prepare_bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.clear_buffer(&self.activity, 0, None);
        let source_bind = bind(
            device,
            &self.sources,
            &[
                &self.params[slot],
                &self.concentrations[self.current],
                &self.production,
                &self.states,
                &b.death_flags,
                &b.cell_ids,
                &b.mode_indices,
                &b.signal_settings_v4,
                &self.modes,
                &b.regulation_params,
                &b.oculocyte_params,
                &b.cell_types,
                &b.nutrients_buffer,
                &b.cell_thermal_state,
                &b.position_and_mass[b.current_index()],
                &b.genome_orientations,
                light.map_or(&self.dummy, |l| l.light_field_buffer()),
                light.map_or(&self.dummy, |l| l.light_color_field_buffer()),
                fluid.map_or(&self.dummy, |f| f.nutrient_voxels_buffer()),
                fluid.map_or(&self.dummy, |f| f.solid_mask_buffer()),
                moss.map_or(&self.dummy, |m| m.moss_density_buffer()),
                &b.cell_count_buffer,
                &b.spatial_grid_counts,
                &b.spatial_grid_cells,
                &self.spatial,
                &self.activity,
            ],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Live Sources"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sources);
            pass.set_bind_group(0, &source_bind, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch, 0);
        }
        self.encode_transport_impl(
            device,
            encoder,
            slot,
            params.count,
            &a.adhesion_connections,
            &a.cell_adhesion_indices,
            true,
        );
        let receiver_bind = bind(
            device,
            &self.receivers,
            &[
                &self.params[slot],
                self.current_field(),
                &self.states,
                &a.signal_flags,
                &b.signal_settings_v4,
                &self.modes,
                &b.cell_types,
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Receivers + Memory"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.receivers);
        pass.set_bind_group(0, &receiver_bind, &[]);
        pass.dispatch_workgroups_indirect(&self.dispatch, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::super::adhesion::GpuAdhesionConnection;
    use super::*;
    fn device() -> (wgpu::Device, wgpu::Queue) {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .expect("GPU diffusion validation requires an adapter");
            eprintln!("Diffusion GPU: {:?}", adapter.get_info());
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 31,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .await
                .unwrap()
        })
    }
    fn upload<T: Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Diffusion Fixture"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }
    fn graph(
        device: &wgpu::Device,
        n: usize,
        edges: &[(usize, usize)],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let mut adjacency = vec![-1i32; n * 20];
        let mut degree = vec![0; n];
        let mut records = vec![];
        for (index, &(a, b)) in edges.iter().enumerate() {
            let mut edge = GpuAdhesionConnection::inactive();
            edge.cell_a_index = a as u32;
            edge.cell_b_index = b as u32;
            edge.is_active = 1;
            records.push(edge);
            for cell in [a, b] {
                assert!(degree[cell] < 20);
                adjacency[cell * 20 + degree[cell]] = index as i32;
                degree[cell] += 1;
            }
        }
        if records.is_empty() {
            records.push(GpuAdhesionConnection::inactive());
        }
        (upload(device, &records), upload(device, &adjacency))
    }
    fn params(n: usize, s: DiffusionSettings) -> DiffusionParams {
        let dt = 1.0 / 15.0;
        DiffusionParams {
            count: n as u32,
            tick: 0,
            degree: 20,
            resolution: 128,
            dt,
            conductance: s.conductance,
            retention: (-s.decay_rate * dt).exp(),
            production_scale: s.production_scale,
            time: dt,
            radius: 200.0,
            grid_cell: 400.0 / 128.0,
            padding: 0.0,
            grid_origin: [-200.0, -200.0, -200.0, 0.0],
        }
    }
    fn read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        n: usize,
    ) -> Vec<[f32; 16]> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n as u64 * 64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, n as u64 * 64);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let result = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        staging.unmap();
        result
    }
    #[test]
    fn gpu_diffusion_matches_reference_and_dynamic_graph() {
        let (device, queue) = device();
        let n = 21;
        let mut pipeline = SignalDiffusionPipeline::new(&device, n as u32);
        pipeline.settings.decay_rate = 0.0;
        let mut field = vec![[0.0; 16]; n];
        field[0][0] = 1000.0;
        field[3][8] = 400.0;
        let mut sources = vec![[0.0; 16]; n];
        sources[1][2] = 100.0;
        sources[2][2] = 100.0;
        queue.write_buffer(pipeline.current_field(), 0, bytemuck::cast_slice(&field));
        queue.write_buffer(&pipeline.production, 0, bytemuck::cast_slice(&sources));
        queue.write_buffer(
            &pipeline.states,
            0,
            bytemuck::cast_slice(&vec![
                DiffusionCellState {
                    live: 1,
                    ..Default::default()
                };
                n
            ]),
        );
        pipeline.write_params(&queue, 0, params(n, pipeline.settings));
        let mut edges: Vec<_> = (1..21).map(|i| (0, i)).collect();
        edges.extend([(1, 2), (2, 3), (3, 1)]);
        for phase in 0..3 {
            if phase == 1 {
                edges.retain(|&(a, b)| a != 0 && b != 0);
            }
            if phase == 2 {
                edges.push((0, 20));
            }
            let (edge_buffer, adjacency) = graph(&device, n, &edges);
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..30 {
                pipeline.encode_transport(
                    &device,
                    &mut encoder,
                    0,
                    n as u32,
                    &edge_buffer,
                    &adjacency,
                );
                field = crate::simulation::signal_diffusion::step(
                    &field,
                    &sources,
                    &edges,
                    pipeline.settings,
                    1.0 / 15.0,
                )
                .unwrap();
            }
            queue.submit([encoder.finish()]);
            let actual = read(&device, &queue, pipeline.current_field(), n);
            for (a, b) in actual.iter().flatten().zip(field.iter().flatten()) {
                assert!(*a >= 0.0);
                assert!((a - b).abs() < 0.01, "GPU {a} CPU {b}");
            }
            assert!((actual.iter().map(|v| v[0]).sum::<f32>() - 1000.0).abs() < 0.02);
        }
    }
    #[test]
    fn idle_field_wakes_for_new_sources_and_preserves_residuals() {
        let (device, queue) = device();
        let b = GpuTripleBufferSystem::new(&device, 4);
        let mut a = AdhesionBuffers::new(&device, 4);
        a.initialize(&queue);
        let mut p = SignalDiffusionPipeline::new(&device, 4);
        let mut genome = crate::genome::Genome::default();
        genome.modes[0].regulation_emit_channel = -1;
        p.sync_modes(&device, &queue, &[genome.clone()]);
        b.sync_signal_settings(&queue, &[genome.clone()]);
        b.sync_regulation_params(&queue, &[genome.clone()]);
        queue.write_buffer(&b.cell_count_buffer, 0, bytemuck::cast_slice(&[2u32, 2, 0, 0]));
        queue.write_buffer(&b.cell_ids, 0, bytemuck::cast_slice(&[1u32, 2]));
        queue.write_buffer(&b.cell_types, 0, bytemuck::cast_slice(&[1u32; 4]));
        queue.write_buffer(&b.nutrients_buffer, 0, bytemuck::cast_slice(&[100000i32; 4]));
        let mut previous = 0.0;
        for tick in 0..5 {
            if tick == 2 || tick == 3 {
                genome.modes[0].regulation_emit_channel = if tick == 2 { 8 } else { -1 };
                genome.modes[0].regulation_emit_value = 300.0;
                b.sync_regulation_params(&queue, &[genome.clone()]);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            p.encode_tick(&device, &queue, &mut encoder, 0, params(2, p.settings),
                &b, &a, None, None, None, [200.0, 6.25, 64.0, 16.0]);
            queue.submit([encoder.finish()]);
            let field = read(&device, &queue, p.current_field(), 2);
            if tick < 2 {
                assert!(field.iter().flatten().all(|v| *v == 0.0));
            } else if tick == 2 {
                assert!(field[0][8] > 0.0, "new sources must wake transport immediately");
            } else {
                let expected = previous * params(2, p.settings).retention;
                assert!((field[0][8] - expected).abs() < 0.0001, "silent sources must retain decaying signals");
            }
            previous = field[0][8];
        }
    }

    #[test]
    fn gpu_gameplay_sources_self_reception_and_slot_reuse() {
        let (device, queue) = device();
        let b = GpuTripleBufferSystem::new(&device, 4);
        let mut a = AdhesionBuffers::new(&device, 4);
        a.initialize(&queue);
        let mut p = SignalDiffusionPipeline::new(&device, 4);
        let mut genome = crate::genome::Genome::default();
        genome.modes[0].regulation_emit_channel = 8;
        genome.modes[0].regulation_emit_value = 300.0;
        p.sync_modes(&device, &queue, &[genome.clone()]);
        b.sync_signal_settings(&queue, &[genome.clone()]);
        b.sync_regulation_params(&queue, &[genome]);
        queue.write_buffer(
            &b.cell_count_buffer,
            0,
            bytemuck::cast_slice(&[2u32, 2, 0, 0]),
        );
        queue.write_buffer(&b.cell_ids, 0, bytemuck::cast_slice(&[1u32, 2]));
        queue.write_buffer(
            &b.nutrients_buffer,
            0,
            bytemuck::cast_slice(&[100000i32; 4]),
        );
        queue.write_buffer(&b.cell_types, 0, bytemuck::cast_slice(&[1u32; 4]));
        for tick in 0..2 {
            let mut encoder = device.create_command_encoder(&Default::default());
            p.encode_tick(
                &device,
                &queue,
                &mut encoder,
                0,
                params(2, p.settings),
                &b,
                &a,
                None,
                None,
                None,
                [200.0, 6.25, 64.0, 16.0],
            );
            queue.submit([encoder.finish()]);
            let field = read(&device, &queue, p.current_field(), 2);
            assert!(field[0][8] >= 20.0);
            if tick == 1 {
                assert!(field[0][8] > 39.0);
            }
        }
        // Reusing a live slot with a new identity must clear its previous field.
        queue.write_buffer(&b.cell_ids, 0, bytemuck::cast_slice(&[3u32, 2]));
        let mut encoder = device.create_command_encoder(&Default::default());
        p.encode_tick(
            &device,
            &queue,
            &mut encoder,
            0,
            params(2, p.settings),
            &b,
            &a,
            None,
            None,
            None,
            [200.0, 6.25, 64.0, 16.0],
        );
        queue.submit([encoder.finish()]);
        let field = read(&device, &queue, p.current_field(), 2);
        assert!((field[0][8] - 20.0).abs() < 0.0001);
        assert!(field[1][8] > 58.0);
    }
    #[test]
    fn gpu_live_sources_and_processors_match_preview() {
        let (device, queue) = device();
        let n = 6;
        let b = GpuTripleBufferSystem::new(&device, 8);
        let mut a = AdhesionBuffers::new(&device, 8);
        a.initialize(&queue);
        let mut p = SignalDiffusionPipeline::new(&device, 8);
        let mut genome = crate::genome::Genome::default();
        for i in 0..n {
            genome.modes[i] = genome.modes[0].clone();
        }
        let types = [1, 7, 3, 4, 14, 15];
        for i in 0..n {
            genome.modes[i].cell_type = types[i];
            genome.modes[i].regulation_emit_channel = -1;
        }
        genome.modes[0].regulation_emit_channel = 8;
        genome.modes[0].regulation_emit_value = 900.0;
        genome.modes[1].oculocyte_sense_type = 16;
        genome.modes[1].oculocyte_signal_channel = 0;
        genome.modes[1].oculocyte_signal_value = 600.0;
        genome.modes[2].photocyte_emit_enabled = true;
        genome.modes[2].photocyte_emit_mode = 1;
        genome.modes[2].photocyte_emit_threshold = 0.5;
        genome.modes[2].photocyte_emit_channel = 1;
        genome.modes[2].photocyte_emit_value = 300.0;
        genome.modes[3].lipocyte_emit_enabled = true;
        genome.modes[3].lipocyte_emit_mode = 0;
        genome.modes[3].lipocyte_emit_threshold = 0.1;
        genome.modes[3].lipocyte_emit_channel = 2;
        genome.modes[3].lipocyte_emit_value = 300.0;
        genome.modes[4].cognocyte_input_channel_a = 8;
        genome.modes[4].cognocyte_input_channel_b = 8;
        genome.modes[4].cognocyte_output_channel = 9;
        genome.modes[4].cognocyte_operation = 0;
        genome.modes[5].memorocyte_input_channel = 8;
        genome.modes[5].memorocyte_output_channel = 10;
        genome.modes[5].memorocyte_rate = 0.5;
        p.sync_modes(&device, &queue, &[genome.clone()]);
        b.sync_signal_settings(&queue, &[genome.clone()]);
        b.sync_regulation_params(&queue, &[genome.clone()]);
        b.sync_oculocyte_params(&queue, &[genome.clone()]);
        queue.write_buffer(
            &b.cell_count_buffer,
            0,
            bytemuck::cast_slice(&[n as u32, n as u32, 0, 0]),
        );
        queue.write_buffer(
            &b.cell_ids,
            0,
            bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()),
        );
        queue.write_buffer(
            &b.mode_indices,
            0,
            bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()),
        );
        queue.write_buffer(
            &b.cell_types,
            0,
            bytemuck::cast_slice(&types.map(|t| t as u32)),
        );
        queue.write_buffer(
            &b.nutrients_buffer,
            0,
            bytemuck::cast_slice(&[100000i32; 8]),
        );
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 4)];
        let (edge_buffer, adjacency) = graph(&device, 8, &edges);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &edge_buffer,
            0,
            &a.adhesion_connections,
            0,
            edges.len() as u64 * 104,
        );
        encoder.copy_buffer_to_buffer(&adjacency, 0, &a.cell_adhesion_indices, 0, 8 * 20 * 4);
        queue.submit([encoder.finish()]);
        let mut cpu = crate::simulation::CanonicalState::new(8);
        for i in 0..n {
            cpu.add_cell(
                glam::Vec3::ZERO,
                glam::Vec3::ZERO,
                glam::Quat::IDENTITY,
                glam::Quat::IDENTITY,
                glam::Vec3::ZERO,
                100.0,
                0,
                i,
                0.0,
                1.0,
                200.0,
                1.0,
            )
            .unwrap();
        }
        for (x, y) in edges {
            cpu.adhesion_manager
                .add_ball_joint(&mut cpu.adhesion_connections, x, y, 0, 0.0, 0)
                .unwrap();
        }
        for tick in 0..30 {
            let mut params = params(n, p.settings);
            params.tick = tick;
            params.time = (tick + 1) as f32 * params.dt;
            let mut encoder = device.create_command_encoder(&Default::default());
            p.encode_tick(
                &device,
                &queue,
                &mut encoder,
                0,
                params,
                &b,
                &a,
                None,
                None,
                None,
                [200.0, 6.25, 64.0, 16.0],
            );
            queue.submit([encoder.finish()]);
            crate::simulation::signal_system::run_signal_system(
                &mut cpu,
                &genome,
                200.0,
                params.dt,
                params.time,
                None,
            );
            let actual = read(&device, &queue, p.current_field(), n);
            for (a, b) in actual
                .iter()
                .flatten()
                .zip(cpu.signal_concentrations[..n].iter().flatten())
            {
                assert!((a - b).abs() < 0.03, "tick {tick} GPU {a} preview {b}");
            }
        }
        assert!(cpu.signal_concentrations[4][9] > 0.0);
        assert!(cpu.signal_concentrations[5][10] > 0.0);
    }

    #[test]
    #[ignore = "population gameplay benchmark"]
    fn benchmark_gameplay_diffusion_sparse_and_population() {
        let (device, queue) = device();
        let n = std::env::var("DIFFUSION_BENCH_CELLS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000usize);
        let b = GpuTripleBufferSystem::new(&device, n as u32);
        let mut a = AdhesionBuffers::new(&device, n as u32);
        a.initialize(&queue);
        let edges: Vec<_> = (0..n - 1)
            .filter(|i| i % 100 != 99)
            .map(|i| (i, i + 1))
            .collect();
        let (edge_buffer, adjacency) = graph(&device, n, &edges);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &edge_buffer,
            0,
            &a.adhesion_connections,
            0,
            edges.len() as u64 * 104,
        );
        encoder.copy_buffer_to_buffer(
            &adjacency,
            0,
            &a.cell_adhesion_indices,
            0,
            n as u64 * 20 * 4,
        );
        queue.submit([encoder.finish()]);
        let mut genome = crate::genome::Genome::default();
        genome.modes[0].regulation_emit_channel = -1;
        genome.modes[1] = genome.modes[0].clone();
        genome.modes[1].regulation_emit_channel = 8;
        genome.modes[1].regulation_emit_value = 100.0;
        b.sync_signal_settings(&queue, &[genome.clone()]);
        b.sync_regulation_params(&queue, &[genome.clone()]);
        queue.write_buffer(
            &b.cell_count_buffer,
            0,
            bytemuck::cast_slice(&[n as u32, n as u32, 0, 0]),
        );
        queue.write_buffer(&b.cell_types, 0, bytemuck::cast_slice(&vec![1u32; n]));
        queue.write_buffer(
            &b.cell_ids,
            0,
            bytemuck::cast_slice(&(0..n as u32).collect::<Vec<_>>()),
        );
        queue.write_buffer(
            &b.nutrients_buffer,
            0,
            bytemuck::cast_slice(&vec![100000i32; n]),
        );
        for stride in [1000usize, 1] {
            let modes: Vec<_> = (0..n).map(|i| u32::from(i % stride == 0)).collect();
            queue.write_buffer(&b.mode_indices, 0, bytemuck::cast_slice(&modes));
            let mut p = SignalDiffusionPipeline::new(&device, n as u32);
            p.sync_modes(&device, &queue, &[genome.clone()]);
            for batch in 0..4 {
                let start = std::time::Instant::now();
                let mut encoder = device.create_command_encoder(&Default::default());
                for _ in 0..30 {
                    p.encode_tick(
                        &device,
                        &queue,
                        &mut encoder,
                        0,
                        params(n, p.settings),
                        &b,
                        &a,
                        None,
                        None,
                        None,
                        [200.0, 6.25, 64.0, 16.0],
                    );
                }
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                if batch > 0 {
                    eprintln!("gameplay diffusion cells={n} source_stride={stride} complete_tick_ms={:.3}",start.elapsed().as_secs_f64()*1000.0/30.0);
                }
            }
        }
    }

    /// Run explicitly; includes bind creation/encoding, queue execution and wait.
    #[test]
    #[ignore = "population benchmark"]
    fn benchmark_diffusion_sparse_and_population() {
        let (device, queue) = device();
        let n = std::env::var("DIFFUSION_BENCH_CELLS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000usize);
        let edges: Vec<_> = (0..n - 1)
            .filter(|i| i % 100 != 99)
            .map(|i| (i, i + 1))
            .collect();
        let (edges, adjacency) = graph(&device, n, &edges);
        for stride in [1000usize, 1] {
            let mut p = SignalDiffusionPipeline::new(&device, n as u32);
            let mut sources = vec![[0.0f32; 16]; n];
            for cell in (0..n).step_by(stride) {
                sources[cell] = [100.0; 16];
            }
            queue.write_buffer(&p.production, 0, bytemuck::cast_slice(&sources));
            queue.write_buffer(
                &p.states,
                0,
                bytemuck::cast_slice(&vec![
                    DiffusionCellState {
                        live: 1,
                        ..Default::default()
                    };
                    n
                ]),
            );
            p.write_params(&queue, 0, params(n, p.settings));
            for batch in 0..4 {
                let start = std::time::Instant::now();
                let mut encoder = device.create_command_encoder(&Default::default());
                for _ in 0..30 {
                    p.encode_transport(&device, &mut encoder, 0, n as u32, &edges, &adjacency);
                }
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                if batch > 0 {
                    eprintln!("diffusion cells={n} source_stride={stride} transport_ms_per_tick={:.3} field_bytes={}",start.elapsed().as_secs_f64()*1000.0/30.0,n*128);
                }
            }
        }
    }
}
