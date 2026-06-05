//! Volumetric Fog Renderer
//!
//! Full-screen post-process effect that ray marches through the light field
//! to produce volumetric fog, god rays, and atmospheric scattering.
//! Composites over the scene using alpha blending.

use bytemuck::{Pod, Zeroable};

/// Camera uniforms for volumetric fog (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FogCameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub time: f32,
}

/// Fog parameters (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FogParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub world_radius: f32,
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    pub fog_density: f32,
    pub fog_steps: u32,
    pub light_color_r: f32,
    pub light_color_g: f32,
    pub light_color_b: f32,
    pub light_intensity: f32,
    pub fog_color_r: f32,
    pub fog_color_g: f32,
    pub fog_color_b: f32,
    pub scattering_anisotropy: f32,
    pub absorption: f32,
    pub height_fog_density: f32,
    pub height_fog_falloff: f32,
    pub ray_start: f32,
    pub ray_end: f32,
    /// Amplitude of wave distortion applied to light field samples (0 = off)
    pub water_wave_strength: f32,
    /// Spatial frequency of the wave distortion
    pub water_wave_scale: f32,
    /// 1 = trilinear interpolation of light field (smooth voxels), 0 = nearest
    pub smooth_light_field: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Volumetric fog renderer
///
/// Renders fog at half resolution for ~4x speedup, then composites
/// back to full resolution with bilinear upscaling.
pub struct VolumetricFogRenderer {
    // Fog ray-march pipeline (renders to half-res offscreen texture)
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)] // Kept alive - referenced by cached_camera_bind_group
    camera_layout: wgpu::BindGroupLayout,
    fog_data_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    fog_params_buffer: wgpu::Buffer,
    depth_sampler: wgpu::Sampler,

    // Half-resolution offscreen target
    fog_texture: wgpu::Texture,
    fog_texture_view: wgpu::TextureView,
    fog_width: u32,
    fog_height: u32,

    // Composite pipeline (upscales half-res fog to full-res scene)
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,
    composite_params_buffer: wgpu::Buffer,
    fog_sampler: wgpu::Sampler,

    // Cached bind groups (invalidated on resize)
    cached_camera_bind_group: wgpu::BindGroup,
    cached_fog_data_bind_group: Option<wgpu::BindGroup>,

    // Configurable parameters
    pub fog_density: f32,
    pub fog_steps: u32,
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub fog_color: [f32; 3],
    pub scattering_anisotropy: f32,
    pub absorption: f32,
    pub height_fog_density: f32,
    pub height_fog_falloff: f32,
    pub ray_start: f32,
    pub ray_end: f32,
    pub enabled: bool,
    /// Amplitude of wave distortion on light field samples (0 = off, ~0.5 = visible)
    pub water_wave_strength: f32,
    /// Spatial frequency of wave distortion
    pub water_wave_scale: f32,
    /// Trilinear interpolation of light field (true = smooth voxels, false = hard edges)
    pub smooth_light_field: bool,
    /// Composite blur kernel radius in texels (0.5 = tight, 2.0 = very soft)
    pub composite_blur_radius: f32,
}

impl VolumetricFogRenderer {
    /// Half-resolution divisor (2 = half-res = 4x fewer pixels)
    const HALF_RES_DIVISOR: u32 = 2;
    /// Format for the offscreen fog texture (needs alpha for blending)
    const FOG_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // Camera bind group layout (group 0)
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fog Camera Layout"),
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

        // Fog data bind group layout (group 1) - no solid_mask, uses light_field sentinel instead
        let fog_data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fog Data Layout"),
            entries: &[
                // Binding 0: FogParams uniform
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
                // Binding 1: light_field storage (also encodes solid as 0.0)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Binding 4: light_color_field storage
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: water density field (attenuates light beams through water)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volumetric Fog Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/volumetric_fog.wgsl").into(),
            ),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volumetric Fog Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &fog_data_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline - renders to half-res Rgba16Float offscreen texture
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volumetric Fog Pipeline"),
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
                    format: Self::FOG_FORMAT,
                    blend: None, // No blending - write raw fog to offscreen
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

        // === Composite pipeline (upscales half-res fog to full-res scene) ===
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fog Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/volumetric_fog_composite.wgsl").into(),
            ),
        });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fog Composite Layout"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fog Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fog Composite Pipeline"),
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
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create camera buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Camera Buffer"),
            size: std::mem::size_of::<FogCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create fog params buffer
        let fog_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Params Buffer"),
            size: std::mem::size_of::<FogParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth sampler (non-filtering for depth texture)
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fog Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Bilinear sampler for upscaling the half-res fog texture
        let fog_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fog Upscale Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create half-res offscreen texture
        let fog_width = (width / Self::HALF_RES_DIVISOR).max(1);
        let fog_height = (height / Self::HALF_RES_DIVISOR).max(1);
        let (fog_texture, fog_texture_view) =
            Self::create_fog_texture(device, fog_width, fog_height);

        // Cached camera bind group (buffer never changes)
        let cached_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fog Camera Bind Group (cached)"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Composite params buffer (blur_radius + pad)
        let composite_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Composite Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Composite bind group (references the fog texture)
        let composite_bind_group = Self::create_composite_bind_group(
            device,
            &composite_bind_group_layout,
            &fog_texture_view,
            &fog_sampler,
            &composite_params_buffer,
        );

        Self {
            pipeline,
            camera_layout,
            fog_data_layout,
            camera_buffer,
            fog_params_buffer,
            depth_sampler,
            fog_texture,
            fog_texture_view,
            fog_width,
            fog_height,
            composite_pipeline,
            composite_bind_group_layout,
            composite_bind_group,
            composite_params_buffer,
            fog_sampler,
            cached_camera_bind_group,
            cached_fog_data_bind_group: None,
            fog_density: 0.5,
            fog_steps: 32,
            light_color: [1.0, 0.95, 0.85],
            light_intensity: 2.0,
            fog_color: [0.4, 0.5, 0.6],
            scattering_anisotropy: 0.6,
            absorption: 0.3,
            height_fog_density: 0.0,
            height_fog_falloff: 0.01,
            ray_start: 1.0,
            ray_end: 1500.0,
            enabled: false,
            water_wave_strength: 0.4,
            water_wave_scale: 0.15,
            smooth_light_field: true,
            composite_blur_radius: 1.5,
        }
    }

    fn create_fog_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Half-Res Fog Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FOG_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_composite_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        fog_view: &wgpu::TextureView,
        fog_sampler: &wgpu::Sampler,
        params_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fog Composite Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(fog_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(fog_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Resize the half-res fog texture (call on window resize)
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let fog_width = (width / Self::HALF_RES_DIVISOR).max(1);
        let fog_height = (height / Self::HALF_RES_DIVISOR).max(1);
        if fog_width == self.fog_width && fog_height == self.fog_height {
            return;
        }
        self.fog_width = fog_width;
        self.fog_height = fog_height;
        let (texture, view) = Self::create_fog_texture(device, fog_width, fog_height);
        self.fog_texture = texture;
        self.fog_texture_view = view;
        // Recreate composite bind group (references new texture view)
        self.composite_bind_group = Self::create_composite_bind_group(
            device,
            &self.composite_bind_group_layout,
            &self.fog_texture_view,
            &self.fog_sampler,
            &self.composite_params_buffer,
        );
        // Invalidate cached fog data bind group (depth_view may have changed)
        self.cached_fog_data_bind_group = None;
    }

    /// Ensure the fog data bind group is created and cached
    fn ensure_fog_data_bind_group(
        &mut self,
        device: &wgpu::Device,
        light_field_buffer: &wgpu::Buffer,
        light_color_field_buffer: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
        water_density_buffer: &wgpu::Buffer,
    ) {
        if self.cached_fog_data_bind_group.is_none() {
            self.cached_fog_data_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Fog Data Bind Group (cached)"),
                    layout: &self.fog_data_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.fog_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: light_field_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: light_color_field_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: water_density_buffer.as_entire_binding(),
                        },
                    ],
                }));
        }
    }

    /// Update camera and fog parameters, then render at half resolution + composite
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        device: &wgpu::Device,
        light_field_buffer: &wgpu::Buffer,
        light_color_field_buffer: &wgpu::Buffer,
        water_density_buffer: &wgpu::Buffer,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        time: f32,
        light_dir: [f32; 3],
        grid_resolution: u32,
        cell_size: f32,
        grid_origin: [f32; 3],
        world_radius: f32,
    ) {
        // Update camera uniforms
        let inv_view_proj = view_proj.inverse();
        let camera_uniform = FogCameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update fog parameters
        let fog_params = FogParams {
            grid_resolution,
            cell_size,
            grid_origin_x: grid_origin[0],
            grid_origin_y: grid_origin[1],
            grid_origin_z: grid_origin[2],
            world_radius,
            light_dir_x: light_dir[0],
            light_dir_y: light_dir[1],
            light_dir_z: light_dir[2],
            fog_density: self.fog_density,
            fog_steps: self.fog_steps,
            light_color_r: self.light_color[0],
            light_color_g: self.light_color[1],
            light_color_b: self.light_color[2],
            light_intensity: self.light_intensity,
            fog_color_r: self.fog_color[0],
            fog_color_g: self.fog_color[1],
            fog_color_b: self.fog_color[2],
            scattering_anisotropy: self.scattering_anisotropy,
            absorption: self.absorption,
            height_fog_density: self.height_fog_density,
            height_fog_falloff: self.height_fog_falloff,
            ray_start: self.ray_start,
            ray_end: self.ray_end,
            water_wave_strength: self.water_wave_strength,
            water_wave_scale: self.water_wave_scale,
            smooth_light_field: self.smooth_light_field as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.fog_params_buffer, 0, bytemuck::bytes_of(&fog_params));

        // Ensure cached fog data bind group exists
        self.ensure_fog_data_bind_group(
            device,
            light_field_buffer,
            light_color_field_buffer,
            depth_view,
            water_density_buffer,
        );

        // Pass 1: Ray march fog at half resolution -> offscreen texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Volumetric Fog Pass (half-res)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.fog_texture_view,
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
            pass.set_bind_group(1, self.cached_fog_data_bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        // Update composite params (blur radius).
        let composite_params: [f32; 4] = [self.composite_blur_radius, 0.0, 0.0, 0.0];
        queue.write_buffer(
            &self.composite_params_buffer,
            0,
            bytemuck::cast_slice(&composite_params),
        );

        // Pass 2: Composite half-res fog onto full-res scene with bilinear upscaling
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fog Composite Pass"),
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
}
