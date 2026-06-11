//! Depth of Field Renderer
//!
//! Full-screen post-process effect that blurs objects outside a focal plane.
//! Uses a Poisson-disc gather blur weighted by circle of confusion (CoC)
//! derived from the depth buffer.
//!
//! Renders at half resolution for performance, then composites back to
//! full resolution with bilinear upscaling (same pattern as volumetric fog).

use bytemuck::{Pod, Zeroable};

/// Camera uniforms for DoF (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DofCameraUniforms {
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
}

/// DoF parameters (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DofParams {
    pub focal_distance: f32,
    pub focal_range: f32,
    pub max_blur_radius: f32,
    pub blur_strength: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub near_clip: f32,
    pub far_clip: f32,
}

/// Depth of Field renderer
///
/// Renders DoF blur at half resolution for ~4x speedup, then composites
/// back to full resolution with bilinear upscaling.
pub struct DepthOfFieldRenderer {
    // DoF blur pipeline (renders to half-res offscreen texture)
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    camera_layout: wgpu::BindGroupLayout,
    dof_data_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    dof_params_buffer: wgpu::Buffer,
    depth_sampler: wgpu::Sampler,
    scene_sampler: wgpu::Sampler,

    // Scene color copy texture (full-res, copied from swapchain before DoF)
    scene_copy_texture: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,
    scene_copy_width: u32,
    scene_copy_height: u32,

    // Half-resolution offscreen target
    dof_texture: wgpu::Texture,
    dof_texture_view: wgpu::TextureView,
    dof_width: u32,
    dof_height: u32,

    // Composite pipeline (upscales half-res DoF to full-res scene)
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,
    dof_sampler: wgpu::Sampler,

    // Cached bind groups
    cached_camera_bind_group: wgpu::BindGroup,
    cached_dof_data_bind_group: Option<wgpu::BindGroup>,

    // Configurable parameters
    pub focal_distance: f32,
    pub focal_range: f32,
    pub max_blur_radius: f32,
    pub blur_strength: f32,
    pub enabled: bool,
}

impl DepthOfFieldRenderer {
    const HALF_RES_DIVISOR: u32 = 2;
    const DOF_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // Camera bind group layout (group 0)
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF Camera Layout"),
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

        // DoF data bind group layout (group 1)
        let dof_data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF Data Layout"),
            entries: &[
                // Binding 0: DofParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: scene color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: scene color sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 3: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create DoF blur shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/dof_blur.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DoF Blur Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &dof_data_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline - renders to half-res Rgba16Float offscreen texture
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Blur Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: Self::DOF_FORMAT,
                    blend: None,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // === Composite pipeline ===
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DoF Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dof_composite.wgsl").into(),
            ),
        });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DoF Composite Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("DoF Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Composite writes directly - DoF replaces the entire scene color
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DoF Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None, // Direct overwrite - DoF is the final color
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create uniform buffers
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DoF Camera Buffer"),
            size: std::mem::size_of::<DofCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dof_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DoF Params Buffer"),
            size: std::mem::size_of::<DofParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Samplers
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let scene_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Scene Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let dof_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DoF Upscale Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create scene copy texture (full-res, COPY_DST so we can copy swapchain into it)
        let (scene_copy_texture, scene_copy_view) =
            Self::create_scene_copy_texture(device, width, height, surface_format);

        // Create half-res offscreen texture
        let dof_width = (width / Self::HALF_RES_DIVISOR).max(1);
        let dof_height = (height / Self::HALF_RES_DIVISOR).max(1);
        let (dof_texture, dof_texture_view) =
            Self::create_dof_texture(device, dof_width, dof_height);

        // Cached camera bind group
        let cached_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DoF Camera Bind Group (cached)"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Composite bind group
        let composite_bind_group = Self::create_composite_bind_group(
            device,
            &composite_bind_group_layout,
            &dof_texture_view,
            &dof_sampler,
        );

        Self {
            pipeline,
            camera_layout,
            dof_data_layout,
            camera_buffer,
            dof_params_buffer,
            depth_sampler,
            scene_sampler,
            scene_copy_texture,
            scene_copy_view,
            scene_copy_width: width,
            scene_copy_height: height,
            dof_texture,
            dof_texture_view,
            dof_width,
            dof_height,
            composite_pipeline,
            composite_bind_group_layout,
            composite_bind_group,
            dof_sampler,
            cached_camera_bind_group,
            cached_dof_data_bind_group: None,
            focal_distance: 50.0,
            focal_range: 30.0,
            max_blur_radius: 8.0,
            blur_strength: 1.0,
            enabled: false,
        }
    }

    fn create_scene_copy_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DoF Scene Copy Texture"),
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_dof_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Half-Res DoF Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DOF_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_composite_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        dof_view: &wgpu::TextureView,
        dof_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DoF Composite Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(dof_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(dof_sampler),
                },
            ],
        })
    }

    /// Resize textures (call on window resize)
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) {
        // Recreate scene copy texture at full resolution
        if width != self.scene_copy_width || height != self.scene_copy_height {
            let (texture, view) =
                Self::create_scene_copy_texture(device, width, height, surface_format);
            self.scene_copy_texture = texture;
            self.scene_copy_view = view;
            self.scene_copy_width = width;
            self.scene_copy_height = height;
        }

        // Recreate half-res DoF texture
        let dof_width = (width / Self::HALF_RES_DIVISOR).max(1);
        let dof_height = (height / Self::HALF_RES_DIVISOR).max(1);
        if dof_width != self.dof_width || dof_height != self.dof_height {
            self.dof_width = dof_width;
            self.dof_height = dof_height;
            let (texture, view) = Self::create_dof_texture(device, dof_width, dof_height);
            self.dof_texture = texture;
            self.dof_texture_view = view;
            self.composite_bind_group = Self::create_composite_bind_group(
                device,
                &self.composite_bind_group_layout,
                &self.dof_texture_view,
                &self.dof_sampler,
            );
        }

        // Invalidate cached bind group (depth_view / scene_copy_view changed)
        self.cached_dof_data_bind_group = None;
    }

    /// Ensure the DoF data bind group is created and cached
    fn ensure_dof_data_bind_group(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
    ) {
        if self.cached_dof_data_bind_group.is_none() {
            self.cached_dof_data_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DoF Data Bind Group (cached)"),
                    layout: &self.dof_data_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.dof_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.scene_copy_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.scene_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                        },
                    ],
                }));
        }
    }

    /// Render DoF effect: blur at half-res -> composite back to swapchain
    ///
    /// The scene must have already been rendered to `self.scene_copy_view` (the
    /// intermediate texture). This method reads from that texture, applies the
    /// depth-based blur, and composites the result onto `color_view` (the swapchain).
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        device: &wgpu::Device,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
    ) {
        // Update camera uniforms
        let inv_view_proj = view_proj.inverse();
        let camera_uniform = DofCameraUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad0: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update DoF parameters - use half-res dimensions for the blur shader
        let dof_params = DofParams {
            focal_distance: self.focal_distance,
            focal_range: self.focal_range,
            max_blur_radius: self.max_blur_radius,
            blur_strength: self.blur_strength,
            screen_width: self.dof_width as f32,
            screen_height: self.dof_height as f32,
            near_clip: 0.1,
            far_clip: 5000.0,
        };
        queue.write_buffer(&self.dof_params_buffer, 0, bytemuck::bytes_of(&dof_params));

        // Ensure cached bind group exists
        self.ensure_dof_data_bind_group(device, depth_view);

        // Pass 1: Blur at half resolution -> offscreen texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF Blur Pass (half-res)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.dof_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.cached_camera_bind_group, &[]);
            pass.set_bind_group(1, self.cached_dof_data_bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Composite half-res DoF onto full-res swapchain
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Get the scene copy texture view for rendering the scene into when DoF is enabled.
    pub fn scene_target_view(&self) -> &wgpu::TextureView {
        &self.scene_copy_view
    }
}
