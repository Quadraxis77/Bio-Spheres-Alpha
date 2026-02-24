//! Adhesion line rendering with wgpu.
//!
//! Renders adhesion connections between cells as outlined quads.
//! Each connection half is a camera-facing quad with zone colors in the center
//! and signal-state colors (black/yellow) as an outline.

use crate::cell::adhesion_zones::{AdhesionZone, get_zone_color};
use crate::simulation::CanonicalState;
use crate::simulation::signal_system;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Half-width of the billboard quad in world units
const LINE_HALF_WIDTH: f32 = 0.04;

/// Renderer for adhesion connection lines (outlined quads).
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
    zone_color: [f32; 4],
    signal_color: [f32; 4],
    edge_factor: f32,
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

        // Each connection needs 12 vertices (2 half-segments × 2 triangles × 3 vertices)
        let vertex_capacity = max_connections * 12;
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
                // Zone color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Signal color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Edge factor
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        // Create render pipeline - TriangleList for billboard quads
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
                topology: wgpu::PrimitiveTopology::TriangleList,
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

    /// Push 6 vertices (2 triangles) forming a camera-facing quad for one half-segment.
    fn push_quad(
        vertices: &mut Vec<LineVertex>,
        start: Vec3,
        end: Vec3,
        perp: Vec3,
        zone_color: [f32; 4],
        signal_color: [f32; 4],
    ) {
        let hw = LINE_HALF_WIDTH;
        // Quad corners: left edge = +perp, right edge = -perp
        let v0 = start + perp * hw; // start, left  (edge=+1)
        let v1 = start - perp * hw; // start, right (edge=-1)
        let v2 = end + perp * hw;   // end, left    (edge=+1)
        let v3 = end - perp * hw;   // end, right   (edge=-1)

        // Triangle 1: v0, v1, v2
        vertices.push(LineVertex { position: v0.to_array(), zone_color, signal_color, edge_factor: 1.0 });
        vertices.push(LineVertex { position: v1.to_array(), zone_color, signal_color, edge_factor: -1.0 });
        vertices.push(LineVertex { position: v2.to_array(), zone_color, signal_color, edge_factor: 1.0 });

        // Triangle 2: v1, v3, v2
        vertices.push(LineVertex { position: v1.to_array(), zone_color, signal_color, edge_factor: -1.0 });
        vertices.push(LineVertex { position: v3.to_array(), zone_color, signal_color, edge_factor: -1.0 });
        vertices.push(LineVertex { position: v2.to_array(), zone_color, signal_color, edge_factor: 1.0 });
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
        let mut vertices = Vec::with_capacity(connections.active_count * 12);

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

            // Zone colors from zone classification
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
            let zone_color_a = get_zone_color(zone_a);
            let zone_color_b = get_zone_color(zone_b);

            // Signal outline color: yellow if signal actually flowed along this connection,
            // black otherwise. Use the flow tracker rather than checking whether both
            // endpoints happen to have signal (which would falsely light up connections
            // between two 1-hop neighbours that never relayed signal to each other).
            let has_signal = state.signal_flow_tracker.has_flow(cell_a_idx, cell_b_idx);
            let signal_color = if has_signal {
                [1.0, 0.9, 0.0, 0.9] // Yellow
            } else {
                [0.0, 0.0, 0.0, 0.6] // Black
            };

            // Compute billboard perpendicular direction
            let line_dir = (pos_b - pos_a).normalize_or_zero();
            let view_dir = (camera_pos - midpoint).normalize_or_zero();
            let mut perp = line_dir.cross(view_dir).normalize_or_zero();
            // Fallback if line is pointing at camera
            if perp.length_squared() < 0.001 {
                perp = Vec3::Y;
            }

            // Half-segment 1: Cell A → Midpoint (zone A color)
            Self::push_quad(&mut vertices, pos_a, midpoint, perp, zone_color_a, signal_color);

            // Half-segment 2: Midpoint → Cell B (zone B color)
            Self::push_quad(&mut vertices, midpoint, pos_b, perp, zone_color_b, signal_color);
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
