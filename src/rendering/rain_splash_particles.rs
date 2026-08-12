//! Rain splash particle system renderer (GPU-based)
//!
//! Where falling rain (already classified by the same "falling" check the
//! rain-audio system uses) is about to land on an occupied voxel, a ring
//! particle spawns at the impact point and expands/fades over its own short
//! lifetime - a ripple where a raindrop hits the water surface.
//!
//! # Pipeline per frame
//!
//! 1. `spawn_new` compute pass - scans the fluid grid for falling-water
//!    impacts and appends ring particles to a ring buffer (same monotonic
//!    atomic-counter-with-wraparound pattern as `death_particles.rs`).
//! 2. `age_particles` compute pass - advances ages, zeroes expired particles.
//! 3. `render` - draws all `min(counter, MAX_PARTICLES)` instances as flat,
//!    surface-oriented (not camera-facing) quads.

use std::sync::mpsc::Receiver;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Rain splash particle instance data.
/// Must match `SplashParticle` in rain_splash_extract.wgsl and the vertex
/// attribute layout in rain_splash_particles.wgsl. Total: 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RainSplashParticle {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub animation: [f32; 4],   // x=age, y=max_lifetime, z/w unused
    pub orientation: [f32; 4], // xyz=surface normal at impact, w unused
}

/// Parameters for the rain splash extract compute shaders.
/// Note: WGSL requires 16-byte alignment for the vec3<f32> grid_origin field,
/// so the layout mirrors `WaterAudioSummaryParams`/`ExtractParams` elsewhere.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SplashExtractParams {
    pub grid_resolution: u32,
    pub gravity_mode: u32,
    pub max_particles: u32,
    /// Only every `sample_stride`-th voxel along each axis is scanned for
    /// impacts - same reasoning and math as `WATER_AUDIO_SAMPLE_STRIDE` in
    /// gpu_simulator.rs: `spawn_new` early-exits unless its voxel is
    /// actively falling water, so its *actual* cost (not just thread count)
    /// scales with how much water is currently falling, spiking hardest in
    /// heavy rain - exactly the case that most needs this pass to be cheap.
    pub sample_stride: u32,
    pub grid_origin: [f32; 3],
    pub cell_size: f32,
    pub time: f32,
    pub delta_time: f32,
    /// Live water mesh alpha (`GpuScene::water_alpha`) - splashes match the
    /// actual rendered water's opacity instead of a guessed constant.
    pub water_alpha: f32,
    pub _pad2: u32,
}

/// Atomic counter for the particle ring buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleCounter {
    pub count: u32,
}

/// GPU-based rain splash particle system renderer.
pub struct RainSplashParticleRenderer {
    render_pipeline: wgpu::RenderPipeline,

    spawn_pipeline: wgpu::ComputePipeline,
    age_pipeline: wgpu::ComputePipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout,

    particle_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    counter_staging_buffer: wgpu::Buffer,
    counter_readback_receiver: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    params_buffer: wgpu::Buffer,

    camera_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,

    max_particles: u32,
    time: f32,
    particle_count: u32,
}

impl RainSplashParticleRenderer {
    /// Maximum simultaneous splash rings. Measured with the GPU frame timer
    /// (performance monitor's "Particles & Fog" segment): during sustained
    /// heavy rain over open water, spawn rate comfortably keeps the ring
    /// buffer full at the old cap of 1536 - i.e. up to 1536 alpha-blended,
    /// overlapping quads over the (typically small, on-screen) water surface
    /// at once. That's real fill-rate cost, not just an unfiltered-candidate
    /// worry - cut 4x, with the spawn throttle in rain_splash_extract.wgsl
    /// tightened to match so the buffer doesn't just refill just as fast and
    /// start thrashing (each slot getting overwritten before its ring
    /// finishes fading, which reads as popping).
    pub const MAX_PARTICLES: u32 = 384;

    /// See `SplashExtractParams::sample_stride`. Kept in step with
    /// `WATER_AUDIO_SAMPLE_STRIDE` in gpu_simulator.rs, since both passes
    /// scan the same grid for the same "falling water" condition.
    const SAMPLE_STRIDE: u32 = 2;

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let max_particles = Self::MAX_PARTICLES;

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rain Splash Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/rain_splash_particles.wgsl").into(),
            ),
        });

        let extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rain Splash Extract Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/rain_splash_extract.wgsl").into(),
            ),
        });

        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rain Splash Particle Buffer"),
            size: (std::mem::size_of::<RainSplashParticle>() * max_particles as usize) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rain Splash Particle Counter"),
            size: std::mem::size_of::<ParticleCounter>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rain Splash Counter Staging"),
            size: std::mem::size_of::<ParticleCounter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rain Splash Extract Params"),
            size: std::mem::size_of::<SplashExtractParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Rain Splash Extract Bind Group Layout"),
                entries: &[
                    // 0: params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: fluid_state (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 2: water_velocity (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: particles (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: counter (read_write, atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Rain Splash Extract Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let spawn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Rain Splash Spawn Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &extract_shader,
            entry_point: Some("spawn_new"),
            compilation_options: Default::default(),
            cache: None,
        });

        let age_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Rain Splash Age Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &extract_shader,
            entry_point: Some("age_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Rain Splash Render Bind Group Layout"),
                entries: &[],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Rain Splash Particle Pipeline Layout"),
                bind_group_layouts: &[camera_layout, &render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RainSplashParticle>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 4,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Rain Splash Particle Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Rain Splash Camera Bind Group Layout (stored)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        Self {
            render_pipeline,
            spawn_pipeline,
            age_pipeline,
            compute_bind_group_layout,
            particle_buffer,
            counter_buffer,
            counter_staging_buffer,
            counter_readback_receiver: None,
            params_buffer,
            camera_bind_group_layout,
            render_bind_group_layout,
            max_particles,
            time: 0.0,
            particle_count: 0,
        }
    }

    /// Build a compute bind group for this frame - cheap enough to rebuild
    /// every frame (same approach as the other fluid-derived particle systems),
    /// since the fluid state/velocity buffer identities don't change.
    pub fn create_compute_bind_group(
        &self,
        device: &wgpu::Device,
        fluid_state_buffer: &wgpu::Buffer,
        water_velocity_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rain Splash Extract Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fluid_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: water_velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.counter_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn create_render_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rain Splash Render Bind Group"),
            layout: &self.render_bind_group_layout,
            entries: &[],
        })
    }

    /// Spawn + age passes. Call once per fluid step, after the fluid sim has
    /// updated `water_velocity` for this step (so "falling" classification is
    /// current) - `compute_bind_group` must reference this step's fluid_state/
    /// water_velocity buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        compute_bind_group: &wgpu::BindGroup,
        grid_resolution: u32,
        grid_origin: Vec3,
        cell_size: f32,
        gravity_mode: u32,
        water_alpha: f32,
        dt: f32,
    ) {
        self.time += dt;

        let params = SplashExtractParams {
            grid_resolution,
            gravity_mode,
            max_particles: self.max_particles,
            sample_stride: Self::SAMPLE_STRIDE,
            grid_origin: grid_origin.to_array(),
            cell_size,
            time: self.time,
            delta_time: dt,
            water_alpha,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rain Splash Spawn Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spawn_pipeline);
            pass.set_bind_group(0, compute_bind_group, &[]);
            let workgroup_size = 4u32;
            let sampled_resolution =
                (grid_resolution + Self::SAMPLE_STRIDE - 1) / Self::SAMPLE_STRIDE;
            let workgroups = (sampled_resolution + workgroup_size - 1) / workgroup_size;
            pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Rain Splash Age Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.age_pipeline);
            pass.set_bind_group(0, compute_bind_group, &[]);
            let workgroups = (self.max_particles + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        if self.counter_readback_receiver.is_none() {
            encoder.copy_buffer_to_buffer(
                &self.counter_buffer,
                0,
                &self.counter_staging_buffer,
                0,
                std::mem::size_of::<ParticleCounter>() as u64,
            );
        }
    }

    /// Poll for particle count (call after command buffer submission).
    pub fn poll_particle_count(&mut self, device: &wgpu::Device) {
        if self.counter_readback_receiver.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            self.counter_staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).ok();
                });
            self.counter_readback_receiver = Some(rx);
        }

        let _ = device.poll(wgpu::PollType::Poll);
        let Some(rx) = self.counter_readback_receiver.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(())) => {
                {
                    let data = self.counter_staging_buffer.slice(..).get_mapped_range();
                    let count: &[u32] = bytemuck::cast_slice(&data);
                    self.particle_count = count[0].min(self.max_particles);
                }
                self.counter_staging_buffer.unmap();
                self.counter_readback_receiver = None;
            }
            Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.counter_readback_receiver = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }

    /// Render splash rings.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_bind_group: &wgpu::BindGroup,
        render_bind_group: &wgpu::BindGroup,
    ) {
        if self.particle_count == 0 {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Rain Splash Particle Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
        render_pass.draw(0..6, 0..self.particle_count);
    }

    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bind_group_layout
    }

    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
}
