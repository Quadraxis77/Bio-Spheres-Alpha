//! FOV cone visualization for oculocyte cells.
//!
//! Renders a wireframe cone showing the oculocyte's field of view and sensing range.
//! The cone apex is at the cell position, pointing along the cell's forward direction.

use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// Number of segments around the cone base circle
const CONE_SEGMENTS: usize = 24;

/// Max cones per frame (one per cell of the selected mode)
const MAX_CONES: usize = 64;

/// Vertices per cone: (CONE_SEGMENTS/3) apex lines * 2 + CONE_SEGMENTS base circle * 2
const VERTS_PER_CONE: usize = CONE_SEGMENTS / 3 * 2 + CONE_SEGMENTS * 2;

const MAX_VERTICES: usize = MAX_CONES * VERTS_PER_CONE;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

/// Renderer for oculocyte FOV cone visualization.
pub struct FovConeRenderer {
    render_pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    /// Pending vertices accumulated this frame
    pending_vertices: Vec<LineVertex>,
}

impl FovConeRenderer {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FOV Cone Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FOV Cone Bind Group Layout"),
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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FOV Cone Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FOV Cone Vertex Buffer"),
            size: (MAX_VERTICES * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FOV Cone Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fov_cone.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FOV Cone Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FOV Cone Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
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

        Self {
            render_pipeline,
            camera_bind_group,
            camera_buffer,
            vertex_buffer,
            pending_vertices: Vec::with_capacity(VERTS_PER_CONE * 4),
        }
    }

    /// Clear pending cones for this frame. Call at the start of the overlay pass.
    pub fn begin_frame(&mut self) {
        self.pending_vertices.clear();
    }

    /// Queue a cone for a single oculocyte cell.
    pub fn queue_cone(
        &mut self,
        cell_position: Vec3,
        cell_rotation: Quat,
        sense_range: f32,
        cos_half_fov: f32,
    ) {
        if self.pending_vertices.len() + VERTS_PER_CONE > MAX_VERTICES {
            return;
        }

        let forward = cell_rotation * Vec3::Z;

        let half_angle = cos_half_fov.acos();
        let base_radius = sense_range * half_angle.sin();
        let cone_length = sense_range * half_angle.cos();
        let base_center = cell_position + forward * cone_length;

        let right = if forward.y.abs() > 0.99 {
            forward.cross(Vec3::X).normalize()
        } else {
            forward.cross(Vec3::Y).normalize()
        };
        let up = right.cross(forward).normalize();

        let cone_color = [0.3, 0.9, 1.0, 0.7];
        let base_color = [0.3, 0.9, 1.0, 0.4];

        for i in 0..CONE_SEGMENTS {
            let angle1 = std::f32::consts::TAU * (i as f32) / (CONE_SEGMENTS as f32);
            let angle2 = std::f32::consts::TAU * ((i + 1) as f32) / (CONE_SEGMENTS as f32);

            let p1 = base_center + right * angle1.cos() * base_radius + up * angle1.sin() * base_radius;
            let p2 = base_center + right * angle2.cos() * base_radius + up * angle2.sin() * base_radius;

            // Apex-to-base line (every 3rd segment)
            if i % 3 == 0 {
                self.pending_vertices.push(LineVertex { position: cell_position.to_array(), color: cone_color });
                self.pending_vertices.push(LineVertex { position: p1.to_array(), color: cone_color });
            }

            // Base circle segment
            self.pending_vertices.push(LineVertex { position: p1.to_array(), color: base_color });
            self.pending_vertices.push(LineVertex { position: p2.to_array(), color: base_color });
        }
    }

    /// Upload and render all queued cones.
    pub fn render_queued(
        &self,
        render_pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        camera_position: Vec3,
    ) {
        if self.pending_vertices.is_empty() {
            return;
        }

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_position.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.pending_vertices));

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.pending_vertices.len() as u32, 0..1);
    }

    pub fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // No resize-dependent resources
    }
}
