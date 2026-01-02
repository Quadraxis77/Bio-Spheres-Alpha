//! Adhesion line rendering with wgpu.
//!
//! Renders adhesion connections between cells as colored line segments.
//! Each connection is rendered as two segments (cell A → midpoint, midpoint → cell B)
//! with colors based on zone classification.

use crate::simulation::CanonicalState;
use crate::cell::{AdhesionZone, get_zone_color};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Renderer for adhesion connection lines.
pub struct AdhesionLineRenderer {
    render_pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
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
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl AdhesionLineRenderer {
    /// Create a new adhesion line renderer.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        max_connections: usize,
    ) -> Self {
        // Create camera buffer
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 50.0],
            _padding: 0.0,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adhesion Line Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adhesion Line Camera Bind Group Layout"),
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
            label: Some("Adhesion Line Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Each connection needs 4 vertices (2 line segments × 2 vertices each)
        let vertex_capacity = max_connections * 4;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adhesion Line Vertex Buffer"),
            size: (vertex_capacity * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adhesion Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/adhesion_line.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adhesion Line Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Adhesion Line Pipeline"),
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
                depth_write_enabled: false, // Lines don't write depth
                depth_compare: wgpu::CompareFunction::Always, // Always pass - show through cells
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
            vertex_capacity,
            width: config.width,
            height: config.height,
        }
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }


    /// Render adhesion lines within an existing render pass
    pub fn render_in_pass(
        &self,
        render_pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
    ) {
        // Update camera
        let forward = camera_rotation * Vec3::NEG_Z;
        let look_target = camera_pos + forward;
        let up = camera_rotation * Vec3::Y;

        let view_matrix = Mat4::look_at_rh(camera_pos, look_target, up);

        let aspect = self.width as f32 / self.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };

        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );

        // Build vertex data for all active connections
        let connections = &state.adhesion_connections;
        let mut vertices = Vec::with_capacity(connections.active_count * 4);

        for i in 0..connections.active_count {
            if connections.is_active[i] == 0 {
                continue;
            }

            let cell_a_idx = connections.cell_a_index[i];
            let cell_b_idx = connections.cell_b_index[i];

            // Validate indices
            if cell_a_idx >= state.cell_count || cell_b_idx >= state.cell_count {
                continue;
            }

            let pos_a = state.positions[cell_a_idx];
            let pos_b = state.positions[cell_b_idx];
            let midpoint = (pos_a + pos_b) * 0.5;

            // Get zone colors
            let zone_a = match connections.zone_a[i] {
                0 => AdhesionZone::ZoneA,
                1 => AdhesionZone::ZoneB,
                _ => AdhesionZone::ZoneC,
            };
            let zone_b = match connections.zone_b[i] {
                0 => AdhesionZone::ZoneA,
                1 => AdhesionZone::ZoneB,
                _ => AdhesionZone::ZoneC,
            };

            let color_a = get_zone_color(zone_a);
            let color_b = get_zone_color(zone_b);

            // Segment 1: Cell A → Midpoint (Zone A color)
            vertices.push(LineVertex {
                position: pos_a.to_array(),
                color: color_a,
            });
            vertices.push(LineVertex {
                position: midpoint.to_array(),
                color: color_a,
            });

            // Segment 2: Midpoint → Cell B (Zone B color)
            vertices.push(LineVertex {
                position: midpoint.to_array(),
                color: color_b,
            });
            vertices.push(LineVertex {
                position: pos_b.to_array(),
                color: color_b,
            });
        }

        // Update vertex buffer and render
        if !vertices.is_empty() && vertices.len() <= self.vertex_capacity {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..vertices.len() as u32, 0..1);
        }
    }
}
