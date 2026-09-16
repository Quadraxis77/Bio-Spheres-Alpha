//! GPU-owned particle draw counts. No staging buffer or CPU count dependency.
use wgpu::util::DeviceExt;

pub struct ParticleDraw {
    pub args: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
}
impl ParticleDraw {
    pub fn new(
        device: &wgpu::Device,
        counter: &wgpu::Buffer,
        vertices: u32,
        capacity: u32,
    ) -> Self {
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle indirect draw arguments"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let limits = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle draw limits"),
            contents: bytemuck::cast_slice(&[vertices, capacity, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle draw arguments"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/particle_draw_args.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle draw arguments"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle draw arguments"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: limits.as_entire_binding(),
                },
            ],
        });
        Self {
            args,
            pipeline,
            bindings,
        }
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indirect_particle_counts_are_current_and_bounded() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&Default::default());
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let counter = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let cases = [0u32, 1, 17, 500_000, u32::MAX, 0];
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (cases.len() * 16) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            for vertices in [3, 6] {
                let draw = ParticleDraw::new(&device, &counter, vertices, 64);
                let mut encoder = device.create_command_encoder(&Default::default());
                for (i, count) in cases.iter().enumerate() {
                    crate::simulation::gpu_upload::encode_buffer_write(
                        &device,
                        &mut encoder,
                        &counter,
                        bytemuck::bytes_of(count),
                    );
                    draw.encode(&mut encoder);
                    encoder.copy_buffer_to_buffer(&draw.args, 0, &readback, i as u64 * 16, 16);
                }
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).unwrap();
                });
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .unwrap();
                rx.recv().unwrap().unwrap();
                let mapped = readback.slice(..).get_mapped_range();
                let words: &[u32] = bytemuck::cast_slice(&mapped);
                for (i, count) in cases.iter().enumerate() {
                    assert_eq!(
                        &words[i * 4..i * 4 + 4],
                        &[vertices, (*count).min(64), 0, 0]
                    );
                }
                drop(mapped);
                readback.unmap();
            }
            // Validate the actual five renderer pipelines with their new draw resources.
            let camera = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let color = wgpu::TextureFormat::Rgba8Unorm;
            let depth = wgpu::TextureFormat::Depth32Float;
            use crate::rendering::{
                death_particles::DeathParticleRenderer,
                nutrient_particles::NutrientParticleRenderer,
                rain_splash_particles::RainSplashParticleRenderer,
                steam_particles::SteamParticleRenderer, water_particles::WaterParticleRenderer,
            };
            drop(SteamParticleRenderer::new(
                &device, color, depth, &camera, 32, 32,
            ));
            drop(WaterParticleRenderer::new(
                &device, color, depth, &camera, 32, 32,
            ));
            drop(NutrientParticleRenderer::new(
                &device, &queue, color, depth, &camera, 32, 32,
            ));
            drop(RainSplashParticleRenderer::new(
                &device, color, depth, &camera,
            ));
            drop(DeathParticleRenderer::new(
                &device, color, depth, &camera, 16,
            ));
        });
    }
}
