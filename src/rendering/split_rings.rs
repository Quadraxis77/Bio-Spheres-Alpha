//! Split ring rendering with wgpu.
//!
//! Renders colored rings around cells to show their split plane direction.
//! The rings are positioned perpendicular to the split direction and scale with cell radius.
//! Uses instancing to render rings for multiple cells in one draw call.

use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// Maximum number of rings that can be rendered in a single frame (per color)
const MAX_RINGS: usize = 256;

/// Renderer for split plane rings.
///
/// Renders two colored rings (blue and green) positioned on either side of the split plane
/// to visualize where a cell will divide.
pub struct SplitRingRenderer {
    render_pipeline_blue: wgpu::RenderPipeline,  // Culls back faces
    render_pipeline_green: wgpu::RenderPipeline, // Culls front faces
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instance_buffer_blue: wgpu::Buffer,
    instance_buffer_green: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// Split ring configuration
    pub config: SplitRingConfig,
    /// Pending blue ring instances for this frame
    pending_blue_instances: Vec<RingInstance>,
    /// Pending green ring instances for this frame
    pending_green_instances: Vec<RingInstance>,
}

/// Configuration for split ring appearance and behavior
#[derive(Debug, Clone)]
pub struct SplitRingConfig {
    /// Whether the rings are visible
    pub visible: bool,
    /// Inner radius multiplier (relative to cell radius)
    pub inner_radius_factor: f32,
    /// Outer radius multiplier (relative to cell radius)
    pub outer_radius_factor: f32,
    /// Offset distance from cell center (relative to cell radius)
    pub offset_factor: f32,
    /// Number of segments in the ring
    pub segments: u32,
}

impl SplitRingConfig {
    /// Create split ring config from UI editor state
    pub fn from_editor_state(editor_state: &crate::ui::panel_context::GenomeEditorState) -> Self {
        Self {
            visible: editor_state.split_rings_visible,
            inner_radius_factor: 1.2,
            outer_radius_factor: 1.4,
            offset_factor: 0.001,
            segments: 32,
        }
    }
}

impl Default for SplitRingConfig {
    fn default() -> Self {
        Self {
            visible: true,
            inner_radius_factor: 1.2,
            outer_radius_factor: 1.4,
            offset_factor: 0.001, // Very small offset like in reference
            segments: 32,
        }
    }
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
struct RingInstance {
    /// Transform matrix for the ring (position + rotation + scale)
    transform: [[f32; 4]; 4],
    /// Ring color
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RingVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl SplitRingRenderer {
    /// Create a new split ring renderer.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        // Create camera buffer
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Split Ring Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create instance buffers for blue and green rings
        let instance_buffer_blue = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Split Ring Blue Instance Buffer"),
            size: (MAX_RINGS * std::mem::size_of::<RingInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buffer_green = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Split Ring Green Instance Buffer"),
            size: (MAX_RINGS * std::mem::size_of::<RingInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout (camera only)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Split Ring Bind Group Layout"),
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
            ],
        });

        // Create bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Split Ring Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        // Create ring geometry
        let segments = 32u32;
        let (vertices, indices) = Self::create_ring_geometry(segments);
        let index_count = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Split Ring Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Split Ring Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Split Ring Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/split_rings.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Split Ring Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Define vertex buffer layout
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RingVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        // Define instance buffer layout
        let instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RingInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Transform matrix (4 vec4s)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        // Create render pipelines with opposite face culling
        let render_pipeline_blue = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Split Ring Blue Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout.clone(), instance_buffer_layout.clone()],
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
                cull_mode: Some(wgpu::Face::Back), // Blue ring culls back faces
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

        let render_pipeline_green = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Split Ring Green Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout, instance_buffer_layout],
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
                cull_mode: Some(wgpu::Face::Front), // Green ring culls front faces
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

        Self {
            render_pipeline_blue,
            render_pipeline_green,
            camera_buffer,
            camera_bind_group,
            instance_buffer_blue,
            instance_buffer_green,
            vertex_buffer,
            index_buffer,
            index_count,
            config: SplitRingConfig::default(),
            pending_blue_instances: Vec::with_capacity(MAX_RINGS),
            pending_green_instances: Vec::with_capacity(MAX_RINGS),
        }
    }

    /// Create ring geometry (flat ring mesh)
    fn create_ring_geometry(segments: u32) -> (Vec<RingVertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Create ring vertices in local space (XY plane, centered at origin)
        let inner_radius = 1.2;
        let outer_radius = 1.4;

        for i in 0..segments {
            let angle1 = (2.0 * std::f32::consts::PI * i as f32) / segments as f32;
            let angle2 = (2.0 * std::f32::consts::PI * ((i + 1) % segments) as f32) / segments as f32;

            let cos1 = angle1.cos();
            let sin1 = angle1.sin();
            let cos2 = angle2.cos();
            let sin2 = angle2.sin();

            let inner1 = Vec3::new(cos1 * inner_radius, sin1 * inner_radius, 0.0);
            let outer1 = Vec3::new(cos1 * outer_radius, sin1 * outer_radius, 0.0);
            let inner2 = Vec3::new(cos2 * inner_radius, sin2 * inner_radius, 0.0);
            let outer2 = Vec3::new(cos2 * outer_radius, sin2 * outer_radius, 0.0);

            let normal = Vec3::Z;

            let base_idx = vertices.len() as u16;

            vertices.push(RingVertex {
                position: inner1.to_array(),
                normal: normal.to_array(),
            });
            vertices.push(RingVertex {
                position: outer1.to_array(),
                normal: normal.to_array(),
            });
            vertices.push(RingVertex {
                position: inner2.to_array(),
                normal: normal.to_array(),
            });
            vertices.push(RingVertex {
                position: outer2.to_array(),
                normal: normal.to_array(),
            });

            indices.push(base_idx);
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);

            indices.push(base_idx + 1);
            indices.push(base_idx + 3);
            indices.push(base_idx + 2);
        }

        (vertices, indices)
    }

    /// Update split ring configuration
    pub fn update_config(&mut self, config: &SplitRingConfig) {
        self.config = config.clone();
    }

    /// Clear pending instances (call at start of frame)
    pub fn begin_frame(&mut self) {
        self.pending_blue_instances.clear();
        self.pending_green_instances.clear();
    }

    /// Queue split rings for a cell
    pub fn queue_rings(
        &mut self,
        cell_position: Vec3,
        cell_rotation: Quat,
        cell_radius: f32,
        split_direction: Vec3,
    ) {
        if !self.config.visible {
            return;
        }
        if self.pending_blue_instances.len() >= MAX_RINGS {
            return;
        }

        // Create rotation to align ring perpendicular to split direction
        let ring_rotation = if split_direction.dot(Vec3::Z).abs() < 0.99 {
            Quat::from_rotation_arc(Vec3::Z, split_direction)
        } else {
            if split_direction.z > 0.0 {
                Quat::IDENTITY
            } else {
                Quat::from_rotation_x(std::f32::consts::PI)
            }
        };

        let world_ring_rotation = cell_rotation * ring_rotation;

        // Blue ring (positive offset)
        let offset_distance = cell_radius * self.config.offset_factor;
        let blue_position = cell_position + (cell_rotation * split_direction) * offset_distance;
        let blue_transform = Mat4::from_translation(blue_position)
            * Mat4::from_quat(world_ring_rotation)
            * Mat4::from_scale(Vec3::splat(cell_radius));

        self.pending_blue_instances.push(RingInstance {
            transform: blue_transform.to_cols_array_2d(),
            color: [0.0, 0.0, 1.0, 1.0],
        });

        // Green ring (negative offset)
        let green_position = cell_position - (cell_rotation * split_direction) * offset_distance;
        let green_transform = Mat4::from_translation(green_position)
            * Mat4::from_quat(world_ring_rotation)
            * Mat4::from_scale(Vec3::splat(cell_radius));

        self.pending_green_instances.push(RingInstance {
            transform: green_transform.to_cols_array_2d(),
            color: [0.0, 1.0, 0.0, 1.0],
        });
    }

    /// Render all queued rings
    pub fn render_queued(
        &self,
        render_pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        view_proj_matrix: Mat4,
        camera_position: Vec3,
    ) {
        if self.pending_blue_instances.is_empty() && self.pending_green_instances.is_empty() {
            return;
        }

        // Update camera uniform
        let camera_uniform = CameraUniform {
            view_proj: view_proj_matrix.to_cols_array_2d(),
            camera_pos: camera_position.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Set shared state
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // Render blue rings
        if !self.pending_blue_instances.is_empty() {
            queue.write_buffer(&self.instance_buffer_blue, 0, bytemuck::cast_slice(&self.pending_blue_instances));
            render_pass.set_pipeline(&self.render_pipeline_blue);
            render_pass.set_vertex_buffer(1, self.instance_buffer_blue.slice(..));
            render_pass.draw_indexed(0..self.index_count, 0, 0..self.pending_blue_instances.len() as u32);
        }

        // Render green rings
        if !self.pending_green_instances.is_empty() {
            queue.write_buffer(&self.instance_buffer_green, 0, bytemuck::cast_slice(&self.pending_green_instances));
            render_pass.set_pipeline(&self.render_pipeline_green);
            render_pass.set_vertex_buffer(1, self.instance_buffer_green.slice(..));
            render_pass.draw_indexed(0..self.index_count, 0, 0..self.pending_green_instances.len() as u32);
        }
    }

    /// Legacy single-cell render (for compatibility)
    pub fn render_cell_rings(
        &mut self,
        render_pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        view_proj_matrix: Mat4,
        camera_position: Vec3,
        cell_position: Vec3,
        cell_rotation: Quat,
        cell_radius: f32,
        split_direction: Vec3,
    ) {
        if !self.config.visible {
            return;
        }

        self.pending_blue_instances.clear();
        self.pending_green_instances.clear();
        self.queue_rings(cell_position, cell_rotation, cell_radius, split_direction);
        self.render_queued(render_pass, queue, view_proj_matrix, camera_position);
    }

    /// Handle window resize
    pub fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // No resize-dependent resources
    }
}
