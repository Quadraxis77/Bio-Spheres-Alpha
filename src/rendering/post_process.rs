//! Post-process renderer: eye adaptation (auto exposure) + contrast.
//!
//! Owns an intermediate scene texture that all scene passes write into.
//! At end-of-frame:
//!   1. `cs_adapt`  (compute) — samples 16 pixels, exponentially smooths exposure.
//!   2. `fs_tonemap` (render) — applies exposure + contrast, writes to swapchain.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PostProcessParams {
    contrast: f32,
    adapt_speed: f32,
    adapt_min: f32,
    adapt_max: f32,
    adapt_enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct PostProcessRenderer {
    // All scene passes render into this texture.
    pub scene_texture: wgpu::Texture,
    pub scene_view: wgpu::TextureView,
    width: u32,
    height: u32,
    surface_format: wgpu::TextureFormat,

    // Persistent single-f32 exposure (updated by cs_adapt, read by fs_tonemap).
    exposure_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,

    // Shared layout and bind group for both passes.
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,

    adapt_pipeline: wgpu::ComputePipeline,
    tonemap_pipeline: wgpu::RenderPipeline,

    // Public settings.
    pub contrast: f32,
    pub adapt_enabled: bool,
    pub adapt_speed: f32,
    pub adapt_min: f32,
    pub adapt_max: f32,
}

impl PostProcessRenderer {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Process Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post_process.wgsl").into(),
            ),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Process BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PP Params"),
            size: std::mem::size_of::<PostProcessParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let exposure_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PP Exposure"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let init: f32 = 1.0;
            exposure_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&init));
        }
        exposure_buf.unmap();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PP Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (scene_texture, scene_view) =
            Self::create_scene_texture(device, width, height, surface_format);

        let bind_group = Self::make_bind_group(
            device,
            &layout,
            &params_buf,
            &exposure_buf,
            &scene_view,
            &sampler,
        );

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PP Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let adapt_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PP Adapt"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_adapt"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PP Tonemap"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_tonemap"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_tonemap"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            scene_texture,
            scene_view,
            width,
            height,
            surface_format,
            exposure_buf,
            params_buf,
            layout,
            bind_group,
            sampler,
            adapt_pipeline,
            tonemap_pipeline,
            contrast: 1.0,
            adapt_enabled: false,
            adapt_speed: 0.05,
            adapt_min: 0.1,
            adapt_max: 8.0,
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        exposure: &wgpu::Buffer,
        scene: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PP Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exposure.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(scene),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    fn create_scene_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PP Scene Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let (tex, view) = Self::create_scene_texture(device, width, height, self.surface_format);
        self.scene_texture = tex;
        self.scene_view = view;
        self.bind_group = Self::make_bind_group(
            device,
            &self.layout,
            &self.params_buf,
            &self.exposure_buf,
            &self.scene_view,
            &self.sampler,
        );
    }

    /// Run adapt compute + tonemap render.  Call after all scene passes complete.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        swapchain: &wgpu::TextureView,
    ) {
        let p = PostProcessParams {
            contrast: self.contrast,
            adapt_speed: self.adapt_speed,
            adapt_min: self.adapt_min,
            adapt_max: self.adapt_max,
            adapt_enabled: self.adapt_enabled as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));

        if self.adapt_enabled {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PP Adapt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adapt_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PP Tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: swapchain,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
