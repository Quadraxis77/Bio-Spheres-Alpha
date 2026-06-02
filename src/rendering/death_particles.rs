//! Death particle system renderer (GPU-based)
//!
//! When a cell dies, a burst of orange-red particles is emitted from its position.
//! Particles expand outward, drift along random directions, and fade over ~1 second.
//!
//! # Pipeline per frame
//!
//! 1. **snapshot** - `copy_buffer_to_buffer(death_flags -> prev_death_flags)` at the
//!    *start* of the frame, before physics runs, so `prev_death_flags` reflects last
//!    frame's state and `spawn_new` can detect the alive->dead transition.
//! 2. **spawn_new** compute pass - runs after physics; detects newly-dead cells and
//!    appends particles to the ring buffer.
//! 3. **age_particles** compute pass - advances ages, zeroes expired particles.
//! 4. **render** - draws all `min(counter, MAX_PARTICLES)` instances as billboards.
//!
//! # Design notes
//!
//! - The compute bind group is **rebuilt every frame** because the position buffer
//!   index rotates with the triple-buffer system.
//! - The particle counter is a monotonically-increasing atomic that wraps at
//!   `MAX_PARTICLES`, giving a ring-buffer effect with no CPU readback needed.
//! - `particle_count` for the draw call is derived from the counter value written
//!   by the GPU last frame (one-frame latency via a staging buffer readback that
//!   completes asynchronously).

use bytemuck::{Pod, Zeroable};

/// Death particle instance data.
/// Must match the `DeathParticle` struct in `death_extract.wgsl` and the vertex
/// attribute layout in `death_particles.wgsl`. Total: 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DeathParticle {
    pub position: [f32; 3],  // World position at spawn
    pub size: f32,           // Base size (cell radius at death)
    pub color: [f32; 4],     // RGBA color
    pub animation: [f32; 4], // x=age, y=max_lifetime, z=vel_dir_x, w=vel_dir_y
    pub velocity: [f32; 4],  // x=vel_dir_x, y=vel_dir_y, z=vel_dir_z, w=unused
}

/// Parameters for the death extract compute shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DeathExtractParams {
    pub cell_capacity: u32,
    pub max_particles: u32,
    pub delta_time: f32,
    pub time: f32,
}

/// Atomic counter for the particle ring buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleCounter {
    pub count: u32,
}

/// GPU-based death particle system renderer.
pub struct DeathParticleRenderer {
    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,

    // Compute pipelines
    spawn_pipeline: wgpu::ComputePipeline,
    age_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    pub particle_buffer: wgpu::Buffer,
    pub counter_buffer: wgpu::Buffer,
    counter_staging_buffer: wgpu::Buffer,
    /// Snapshot of death_flags from the previous frame.
    pub prev_death_flags_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

    // Bind group layouts
    camera_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,

    // State
    pub max_particles: u32,
    time: f32,
    /// How many particles to draw this frame (updated from counter staging readback).
    particle_count: u32,
    #[allow(dead_code)]
    cell_capacity: u32,
}

impl DeathParticleRenderer {
    /// Maximum simultaneous death particles.
    pub const MAX_PARTICLES: u32 = 200_000;

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        cell_capacity: u32,
    ) -> Self {
        let max_particles = Self::MAX_PARTICLES;

        // -- Shaders --------------------------------------------------------------
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Death Particle Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/death_particles.wgsl").into(),
            ),
        });

        let extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Death Extract Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/death_extract.wgsl").into(),
            ),
        });

        // -- Buffers ---------------------------------------------------------------
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Death Particle Buffer"),
            size: (std::mem::size_of::<DeathParticle>() * max_particles as usize) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Death Particle Counter"),
            size: std::mem::size_of::<ParticleCounter>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Death Counter Staging"),
            size: std::mem::size_of::<ParticleCounter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prev_death_flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Death Particle Prev Death Flags"),
            size: cell_capacity as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Death Extract Params"),
            size: std::mem::size_of::<DeathExtractParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -- Compute bind group layout ---------------------------------------------
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Death Extract Bind Group Layout"),
                entries: &[
                    // 0: death_flags (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: prev_death_flags (read)
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
                    // 2: position_and_mass (read) - current triple-buffer slot
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
                    // 5: params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // -- Compute pipelines -----------------------------------------------------
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Death Extract Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let spawn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Death Spawn Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &extract_shader,
            entry_point: Some("spawn_new"),
            compilation_options: Default::default(),
            cache: None,
        });

        let age_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Death Age Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &extract_shader,
            entry_point: Some("age_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        // -- Render bind group layout (empty - camera is group 0) -----------------
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Death Render Bind Group Layout"),
                entries: &[],
            });

        // -- Render pipeline -------------------------------------------------------
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Death Particle Pipeline Layout"),
                bind_group_layouts: &[camera_layout, &render_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Vertex buffer layout - instanced, one entry per DeathParticle (64 bytes)
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DeathParticle>() as u64,
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
            label: Some("Death Particle Pipeline"),
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
                            // Standard alpha blending - tissue is translucent, not glowing
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

        // -- Camera bind group layout (stored for bind group creation) -------------
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Death Camera Bind Group Layout (stored)"),
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
            prev_death_flags_buffer,
            params_buffer,
            camera_bind_group_layout,
            render_bind_group_layout,
            max_particles,
            time: 0.0,
            particle_count: 0,
            cell_capacity,
        }
    }

    /// Build a compute bind group for this frame.
    ///
    /// `position_and_mass_buffer` must be the **current** triple-buffer slot
    /// (i.e. `position_and_mass[current_index()]`), which holds the last
    /// completed physics frame's positions - not the one being written this frame.
    pub fn create_compute_bind_group(
        &self,
        device: &wgpu::Device,
        death_flags_buffer: &wgpu::Buffer,
        position_and_mass_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Death Extract Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: death_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.prev_death_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: position_and_mass_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the (empty) render bind group.
    pub fn create_render_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Death Render Bind Group"),
            layout: &self.render_bind_group_layout,
            entries: &[],
        })
    }

    /// **Step 1 - snapshot** (call BEFORE physics runs this frame).
    ///
    /// Copies `death_flags` -> `prev_death_flags` so that `spawn_new` can detect
    /// the alive->dead transition after physics completes.
    pub fn snapshot_death_flags(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        death_flags_buffer: &wgpu::Buffer,
        cell_capacity: u32,
    ) {
        let copy_size = (cell_capacity as u64 * std::mem::size_of::<u32>() as u64)
            .min(death_flags_buffer.size())
            .min(self.prev_death_flags_buffer.size());
        encoder.copy_buffer_to_buffer(
            death_flags_buffer,
            0,
            &self.prev_death_flags_buffer,
            0,
            copy_size,
        );
    }

    /// **Steps 2 & 3 - spawn + age** (call AFTER physics runs this frame).
    ///
    /// `compute_bind_group` must have been built this frame with
    /// `create_compute_bind_group` using the current triple-buffer position slot.
    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        compute_bind_group: &wgpu::BindGroup,
        cell_capacity: u32,
        dt: f32,
    ) {
        self.time += dt;

        let params = DeathExtractParams {
            cell_capacity,
            max_particles: self.max_particles,
            delta_time: dt,
            time: self.time,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Pass 1: spawn particles for newly-dead cells
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Death Spawn Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spawn_pipeline);
            pass.set_bind_group(0, compute_bind_group, &[]);
            let workgroups = (cell_capacity + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: age all particles (dispatch over full ring buffer)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Death Age Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.age_pipeline);
            pass.set_bind_group(0, compute_bind_group, &[]);
            let workgroups = (self.max_particles + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy counter to staging for async readback (one-frame latency is fine)
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.counter_staging_buffer,
            0,
            std::mem::size_of::<ParticleCounter>() as u64,
        );
    }

    /// Poll the staging buffer for the particle count written last frame.
    /// Call this after `queue.submit()`.
    pub fn poll_particle_count(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let buffer_slice = self.counter_staging_buffer.slice(..);
        let mapped = Arc::new(AtomicBool::new(false));
        let mapped_clone = mapped.clone();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                mapped_clone.store(true, Ordering::SeqCst);
            }
        });

        while !mapped.load(Ordering::SeqCst) {
            let _ = device.poll(wgpu::PollType::Poll);
        }

        {
            let data = buffer_slice.get_mapped_range();
            let count: &[u32] = bytemuck::cast_slice(&data);
            // Counter is monotonically increasing; clamp to ring buffer size
            self.particle_count = count[0].min(self.max_particles);
        }

        self.counter_staging_buffer.unmap();
    }

    /// Render death particles.
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
            label: Some("Death Particle Pass"),
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

    /// Camera bind group layout (for creating the camera bind group in gpu_scene).
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bind_group_layout
    }

    /// Current particle count (updated from staging readback).
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
}
