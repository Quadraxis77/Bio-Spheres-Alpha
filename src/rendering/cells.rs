//! Cell rendering with wgpu.
//!
//! Provides GPU-instanced rendering of cells as billboarded spheres
//! with Weighted Blended Order-Independent Transparency (WBOIT).

use crate::cell::types::CellTypeVisuals;
use crate::genome::Genome;
use crate::simulation::CanonicalState;
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// Renderer for cells using GPU instancing with OIT.
///
/// Renders cells as camera-facing billboards with sphere-like shading
/// using weighted blended order-independent transparency.
pub struct CellRenderer {
    // Opaque pass pipeline (for fully opaque cells)
    opaque_pipeline: wgpu::RenderPipeline,
    // Depth-only pipeline for pre-pass
    #[allow(dead_code)]
    depth_only_pipeline: wgpu::RenderPipeline,
    // OIT accumulation pass pipeline
    oit_pipeline: wgpu::RenderPipeline,
    // OIT composite pass pipeline
    composite_pipeline: wgpu::RenderPipeline,
    // Shared resources
    #[allow(dead_code)]
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    quad_vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    transparent_instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    // Depth texture
    #[allow(dead_code)]
    depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    // OIT textures
    #[allow(dead_code)]
    accum_texture: wgpu::Texture,
    accum_view: wgpu::TextureView,
    #[allow(dead_code)]
    revealage_texture: wgpu::Texture,
    revealage_view: wgpu::TextureView,
    // Composite bind group
    composite_bind_group: wgpu::BindGroup,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    // Surface format for recreation
    #[allow(dead_code)]
    surface_format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
    light_direction: [f32; 3],
    _padding1: f32,
    light_color: [f32; 3],
    _padding2: f32,
    ambient_color: [f32; 3],
    _padding3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CellInstance {
    position: [f32; 3],
    radius: f32,
    color: [f32; 4],
    visual_params: [f32; 4],
}


// Quad vertices for billboard rendering
const QUAD_VERTICES: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

impl CellRenderer {
    /// Create a new cell renderer with OIT support.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        capacity: usize,
    ) -> Self {
        let width = config.width;
        let height = config.height;
        let surface_format = config.format;

        // Create camera uniform buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create lighting uniform buffer
        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Lighting Buffer"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create camera bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cell Camera Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        // Create camera bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_buffer.as_entire_binding(),
                },
            ],
        });

        // Create quad vertex buffer
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create instance buffer for opaque cells
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Instance Buffer"),
            size: (capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create instance buffer for transparent cells
        let transparent_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transparent Cell Instance Buffer"),
            size: (capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);

        // Create OIT textures
        let (accum_texture, accum_view) = Self::create_accum_texture(device, width, height);
        let (revealage_texture, revealage_view) =
            Self::create_revealage_texture(device, width, height);

        // Create composite sampler
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("OIT Composite Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create composite bind group layout
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("OIT Composite Bind Group Layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create composite bind group
        let composite_bind_group = Self::create_composite_bind_group(
            device,
            &composite_bind_group_layout,
            &accum_view,
            &revealage_view,
            &composite_sampler,
        );

        // Vertex buffer layouts
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CellInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // position
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32, // radius
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4, // color
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4, // visual_params
                },
            ],
        };

        // Create pipeline layout for opaque and OIT passes
        let cell_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Cell Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Load opaque shader
        let opaque_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Billboard Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cell_billboard.wgsl").into()),
        });

        // Create opaque pipeline
        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Opaque Pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &opaque_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone(), instance_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &opaque_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Load depth-only shader
        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Billboard Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cell_billboard_depth.wgsl").into()),
        });

        // Create depth-only pipeline for pre-pass (no color output)
        let depth_only_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Depth-Only Pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone(), instance_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_main"),
                targets: &[], // No color targets
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Load OIT shader
        let oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Billboard OIT Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cell_billboard_oit.wgsl").into()),
        });

        // Create OIT accumulation pipeline
        // Outputs to two render targets: accum (RGBA16Float) and revealage (R8)
        let oit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell OIT Pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &oit_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone(), instance_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &oit_shader,
                entry_point: Some("fs_main"),
                targets: &[
                    // Accumulation texture - additive blending
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Revealage texture - multiplicative blending (1 - alpha)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Read-only - don't write to depth buffer
                depth_compare: wgpu::CompareFunction::Less, // Discard fragments behind opaque geometry
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Load composite shader
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("OIT Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/oit_composite.wgsl").into()),
        });

        // Create composite pipeline layout
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("OIT Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create composite pipeline
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("OIT Composite Pipeline"),
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
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            opaque_pipeline,
            depth_only_pipeline,
            oit_pipeline,
            composite_pipeline,
            camera_bind_group_layout,
            camera_bind_group,
            camera_buffer,
            lighting_buffer,
            quad_vertex_buffer,
            instance_buffer,
            transparent_instance_buffer,
            instance_capacity: capacity,
            depth_texture,
            depth_view,
            accum_texture,
            accum_view,
            revealage_texture,
            revealage_view,
            composite_bind_group,
            composite_bind_group_layout,
            composite_sampler,
            surface_format,
            width,
            height,
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cell Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_accum_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("OIT Accumulation Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_revealage_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("OIT Revealage Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_composite_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        accum_view: &wgpu::TextureView,
        revealage_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OIT Composite Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(revealage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Resize the renderer textures.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;

        // Recreate depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;

        // Recreate OIT textures
        let (accum_texture, accum_view) = Self::create_accum_texture(device, width, height);
        let (revealage_texture, revealage_view) =
            Self::create_revealage_texture(device, width, height);
        self.accum_texture = accum_texture;
        self.accum_view = accum_view;
        self.revealage_texture = revealage_texture;
        self.revealage_view = revealage_view;

        // Recreate composite bind group with new texture views
        self.composite_bind_group = Self::create_composite_bind_group(
            device,
            &self.composite_bind_group_layout,
            &self.accum_view,
            &self.revealage_view,
            &self.composite_sampler,
        );
    }

    /// Render cells within an existing render pass.
    /// Uses OIT for proper transparency handling.
    pub fn render_in_pass<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
        camera_pos: Vec3,
        camera_rotation: Quat,
    ) {
        if state.cell_count == 0 {
            return;
        }

        // Update camera uniform
        let view_matrix = Mat4::look_at_rh(
            camera_pos,
            camera_pos + camera_rotation * Vec3::NEG_Z,
            camera_rotation * Vec3::Y,
        );
        let aspect = self.width as f32 / self.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update lighting uniform
        let light_dir = Vec3::new(0.5, 1.0, 0.3).normalize();
        let lighting_uniform = LightingUniform {
            light_direction: light_dir.to_array(),
            _padding1: 0.0,
            light_color: [0.8, 0.8, 0.8],
            _padding2: 0.0,
            ambient_color: [0.3, 0.3, 0.35],
            _padding3: 0.0,
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting_uniform));

        // Build instance data - combine opaque and transparent for simple in-pass rendering
        let (opaque_instances, transparent_instances) = self.build_instances(state, genome, cell_type_visuals);
        
        // Combine all instances for simple rendering (OIT not available in render_in_pass)
        let mut all_instances = opaque_instances;
        all_instances.extend(transparent_instances);
        
        if all_instances.is_empty() {
            return;
        }

        // Update instance buffer
        if all_instances.len() <= self.instance_capacity {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&all_instances));
        }

        // Render using opaque pipeline (OIT requires separate passes which we handle in render_oit())
        // For render_in_pass, we use the simpler opaque pipeline with alpha blending
        render_pass.set_pipeline(&self.opaque_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw(0..6, 0..all_instances.len() as u32);
    }

    /// Full render with OIT support (creates its own render passes).
    /// Call this instead of render_in_pass for proper transparency.
    #[allow(dead_code)]
    pub fn render_oit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
        camera_pos: Vec3,
        camera_rotation: Quat,
    ) {
        if state.cell_count == 0 {
            return;
        }

        // Update camera uniform
        let view_matrix = Mat4::look_at_rh(
            camera_pos,
            camera_pos + camera_rotation * Vec3::NEG_Z,
            camera_rotation * Vec3::Y,
        );
        let aspect = self.width as f32 / self.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update lighting uniform
        let light_dir = Vec3::new(0.5, 1.0, 0.3).normalize();
        let lighting_uniform = LightingUniform {
            light_direction: light_dir.to_array(),
            _padding1: 0.0,
            light_color: [0.8, 0.8, 0.8],
            _padding2: 0.0,
            ambient_color: [0.3, 0.3, 0.35],
            _padding3: 0.0,
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting_uniform));

        // Build instance data - separate opaque and transparent
        let (opaque_instances, transparent_instances) = self.build_instances(state, genome, cell_type_visuals);
        
        if opaque_instances.is_empty() && transparent_instances.is_empty() {
            return;
        }

        // Update instance buffers
        if !opaque_instances.is_empty() && opaque_instances.len() <= self.instance_capacity {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&opaque_instances));
        }
        if !transparent_instances.is_empty() && transparent_instances.len() <= self.instance_capacity {
            queue.write_buffer(&self.transparent_instance_buffer, 0, bytemuck::cast_slice(&transparent_instances));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cell OIT Render Encoder"),
        });

        // Pass 1: Render opaque cells with depth writing
        if !opaque_instances.is_empty() {
            let mut opaque_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cell Opaque Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve background
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            opaque_pass.set_pipeline(&self.opaque_pipeline);
            opaque_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            opaque_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            opaque_pass.draw(0..6, 0..opaque_instances.len() as u32);
        }

        // Pass 2: OIT for transparent cells only
        if !transparent_instances.is_empty() {
            // OIT accumulation pass
            {
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("OIT Accumulation Pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.accum_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.revealage_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Keep depth from opaque pass
                            store: wgpu::StoreOp::Store, // Don't need to store since we don't write
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                oit_pass.set_pipeline(&self.oit_pipeline);
                oit_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                oit_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
                oit_pass.set_vertex_buffer(1, self.transparent_instance_buffer.slice(..));
                oit_pass.draw(0..6, 0..transparent_instances.len() as u32);
            }

            // OIT composite pass
            {
                let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("OIT Composite Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
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

                composite_pass.set_pipeline(&self.composite_pipeline);
                composite_pass.set_bind_group(0, &self.composite_bind_group, &[]);
                composite_pass.draw(0..3, 0..1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Build instance data from simulation state, separated into opaque and transparent.
    fn build_instances(
        &self,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
    ) -> (Vec<CellInstance>, Vec<CellInstance>) {
        let mut opaque_instances = Vec::with_capacity(state.cell_count);
        let mut transparent_instances = Vec::with_capacity(state.cell_count);

        for i in 0..state.cell_count {
            let position = state.positions[i];
            let radius = state.radii[i];
            let mode_index = state.mode_indices[i];
            // Use genome_id as cell type for now (cell type 0 = default)
            let cell_type = state.genome_ids[i];

            // Get color and opacity from genome mode
            let (color, opacity, emissive) = if let Some(genome) = genome {
                if mode_index < genome.modes.len() {
                    let mode = &genome.modes[mode_index];
                    (mode.color.to_array(), mode.opacity, mode.emissive)
                } else {
                    ([0.5, 0.5, 0.5], 1.0, 0.0)
                }
            } else {
                ([0.5, 0.5, 0.5], 1.0, 0.0)
            };

            // Get visual params from cell type visuals
            let (specular_strength, specular_power, fresnel_strength) =
                if let Some(visuals) = cell_type_visuals {
                    if cell_type < visuals.len() {
                        let v = &visuals[cell_type];
                        (v.specular_strength, v.specular_power, v.fresnel_strength)
                    } else {
                        (0.5, 32.0, 0.3)
                    }
                } else {
                    (0.5, 32.0, 0.3)
                };

            let instance = CellInstance {
                position: position.to_array(),
                radius,
                color: [color[0], color[1], color[2], opacity],
                visual_params: [specular_strength, specular_power, fresnel_strength, emissive],
            };

            // Separate opaque (alpha >= 0.99) from transparent
            if opacity >= 0.99 {
                opaque_instances.push(instance);
            } else {
                transparent_instances.push(instance);
            }
        }

        (opaque_instances, transparent_instances)
    }
}