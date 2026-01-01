use crate::simulation::CanonicalState;
use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};

pub struct PreviewRenderer {
    render_pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl PreviewRenderer {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, config: &wgpu::SurfaceConfiguration) -> Self {
        // Create camera buffer
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        };
        
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create bind group layout
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
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
        
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cell.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
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
        
        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(device, config.width, config.height);
        
        Self {
            render_pipeline,
            camera_bind_group,
            camera_buffer,
            depth_texture,
            depth_view,
        }
    }
    
    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
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
    
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }
    
    pub fn render(&self, _device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView, _state: &CanonicalState) {
        // Update camera
        let view_matrix = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 50.0),
            Vec3::ZERO,
            Vec3::Y,
        );
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;
        
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
        };
        
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
        
        // Create command encoder
        let mut encoder = queue.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            
            // TODO: Draw cells
            // For now, just clear the screen
        }
        
        queue.submit(std::iter::once(encoder.finish()));
    }
}
