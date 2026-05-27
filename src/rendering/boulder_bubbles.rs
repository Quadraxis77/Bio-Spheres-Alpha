//! Boulder bubble particle system.
//!
//! Emits small rising bubbles from boulders that are submerged in water and moving.
//! Uses a GPU ring buffer: spawn_bubbles writes new particles, age_bubbles advances
//! and expires them, and the render pass draws them as billboarded circles.
//!
//! The compute and render shaders share the same file (boulder_bubbles.wgsl) but
//! use different bind group layouts — the compute pass uses group 0 with boulder
//! and water data, while the render pass uses group 0 with just the camera uniform.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::simulation::gpu_physics::boulder_buffers::BoulderBuffers;

/// Maximum simultaneous bubble particles.
pub const MAX_BUBBLES: u32 = 4096;

/// Per-particle data. Must match `BubbleParticle` in boulder_bubbles.wgsl (48 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BubbleParticle {
    pub position: [f32; 3],
    pub size:     f32,
    pub velocity: [f32; 3],
    pub age:      f32,
    pub max_age:  f32,
    pub _pad:     [f32; 3],
}

const _: () = assert!(std::mem::size_of::<BubbleParticle>() == 48);

/// Uniform params for the compute passes. Must be exactly 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BubbleParams {
    delta_time:     f32,
    current_time:   f32,
    current_frame:  u32,
    max_particles:  u32,
    boulder_count:  u32,
    min_speed:      f32,
    emit_rate:      f32,
    burst_duration: f32,
    gravity_mode:   u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
    _pad3:          [f32; 4],
}

const _: () = assert!(std::mem::size_of::<BubbleParams>() == 64);

/// Camera uniform for the render pass.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj:  [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad:       f32,
}

pub struct BoulderBubbleSystem {
    // Compute pipelines
    spawn_pipeline: wgpu::ComputePipeline,
    age_pipeline:   wgpu::ComputePipeline,
    compute_bgl:    wgpu::BindGroupLayout,

    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    camera_buffer:   wgpu::Buffer,
    #[allow(dead_code)]
    render_bgl:      wgpu::BindGroupLayout,
    render_bg:       wgpu::BindGroup,

    // Shared buffers
    particle_buffer:  wgpu::Buffer,
    counter_buffer:   wgpu::Buffer,
    params_buffer:    wgpu::Buffer,
    prev_in_water:    wgpu::Buffer,
    entry_timer:      wgpu::Buffer,

    current_time:   f32,
    current_frame:  u32,
    #[allow(dead_code)]
    particle_count: u32,

    pub width:  u32,
    pub height: u32,
}

impl BoulderBubbleSystem {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // ── Buffers ───────────────────────────────────────────────────────────
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bubble Particle Buffer"),
            size: MAX_BUBBLES as u64 * std::mem::size_of::<BubbleParticle>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bubble Counter"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bubble Params"),
            size: std::mem::size_of::<BubbleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bubble Camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Boulder Bubbles Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/boulder_bubbles.wgsl").into(),
            ),
        });

        let prev_in_water = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bubble Prev In Water"),
            contents: bytemuck::cast_slice(&vec![0u32; crate::simulation::gpu_physics::boulder_buffers::MAX_BOULDERS as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let entry_timer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bubble Entry Timer"),
            contents: bytemuck::cast_slice(&vec![0.0f32; crate::simulation::gpu_physics::boulder_buffers::MAX_BOULDERS as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uni = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bubble Compute BGL"),
            // 0: bubble_params(uniform), 1: boulder_state(ro), 2: particles(rw),
            // 3: counter(rw), 4: water_params(uniform), 5: water_bitfield(ro),
            // 6: prev_in_water(rw), 7: entry_timer(rw)
            entries: &[uni(0), ro(1), rw(2), rw(3), uni(4), ro(5), rw(6), rw(7)],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bubble Compute Layout"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });

        let spawn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bubble Spawn Pipeline"),
            layout: Some(&compute_layout),
            module: &shader,
            entry_point: Some("spawn_bubbles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let age_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bubble Age Pipeline"),
            layout: Some(&compute_layout),
            module: &shader,
            entry_point: Some("age_bubbles"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Render bind group layout (camera only) ────────────────────────────
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bubble Render BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bubble Render BG"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // ── Render pipeline ───────────────────────────────────────────────────
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bubble Render Layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BubbleParticle>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 28, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 32, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 36, shader_location: 5 },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bubble Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Write default params so the buffer is valid before first frame
        let default_params = BubbleParams {
            delta_time:     0.016,
            current_time:   0.0,
            current_frame:  0,
            max_particles:  MAX_BUBBLES,
            boulder_count:  0,
            min_speed:      0.5,
            emit_rate:      12.0,
            burst_duration: 0.6,
            gravity_mode:   1,
            _pad0: 0, _pad1: 0, _pad2: 0, _pad3: [0.0; 4],
        };
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&default_params));

        Self {
            spawn_pipeline,
            age_pipeline,
            compute_bgl,
            render_pipeline,
            camera_buffer,
            render_bgl,
            render_bg,
            particle_buffer,
            counter_buffer,
            params_buffer,
            prev_in_water,
            entry_timer,
            current_time: 0.0,
            current_frame: 0,
            particle_count: MAX_BUBBLES, // draw all slots; dead ones collapse in vertex shader
            width,
            height,
        }
    }

    /// Build the compute bind group for this frame.
    /// Call after boulder system and fluid system are both initialized.
    pub fn create_compute_bind_group(
        &self,
        device: &wgpu::Device,
        boulder_buffers: &BoulderBuffers,
        water_params_buffer: &wgpu::Buffer,
        water_bitfield_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bubble Compute BG"),
            layout: &self.compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: boulder_buffers.boulder_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.counter_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: water_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: water_bitfield_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.prev_in_water.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.entry_timer.as_entire_binding() },
            ],
        })
    }

    /// Run spawn + age compute passes. Call after boulder physics runs.
    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        compute_bg: &wgpu::BindGroup,
        boulder_count: u32,
        dt: f32,
        gravity_mode: u32,
    ) {
        self.current_time  += dt;
        self.current_frame += 1;

        let params = BubbleParams {
            delta_time:     dt,
            current_time:   self.current_time,
            current_frame:  self.current_frame,
            max_particles:  MAX_BUBBLES,
            boulder_count,
            min_speed:      0.5,
            emit_rate:      12.0,
            burst_duration: 0.6,
            gravity_mode,
            _pad0: 0, _pad1: 0, _pad2: 0, _pad3: [0.0; 4],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Spawn pass: one thread per boulder
        if boulder_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bubble Spawn"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spawn_pipeline);
            pass.set_bind_group(0, compute_bg, &[]);
            pass.dispatch_workgroups((boulder_count + 63) / 64, 1, 1);
        }

        // Age pass: one thread per particle slot
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bubble Age"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.age_pipeline);
            pass.set_bind_group(0, compute_bg, &[]);
            pass.dispatch_workgroups((MAX_BUBBLES + 255) / 256, 1, 1);
        }
    }

    /// Render bubbles. Call after cells and boulders are rendered.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: glam::Vec3,
        camera_rotation: glam::Quat,
    ) {
        // Update camera uniform
        let view = glam::Mat4::from_rotation_translation(camera_rotation, camera_pos).inverse();
        let aspect = self.width as f32 / self.height as f32;
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
        let cam = CameraUniform {
            view_proj:  (proj * view).to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad:       0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bubble Render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bg, &[]);
        pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
        // Draw all MAX_BUBBLES instances — dead ones collapse to degenerate triangles
        pass.draw(0..6, 0..MAX_BUBBLES);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
