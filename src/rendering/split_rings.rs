//! Split ring rendering with wgpu.
//!
//! Renders colored rings around cells to show their split plane direction.
//! The rings are positioned perpendicular to the split direction and scale with cell radius.

use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// Renderer for split plane rings.
///
/// Renders two colored rings (blue and green) positioned on either side of the split plane
/// to visualize where a cell will divide.
pub struct SplitRingRenderer {
    render_pipeline_blue: wgpu::RenderPipeline,  // Culls back faces
    render_pipeline_green: wgpu::RenderPipeline, // Culls front faces
    camera_buffer: wgpu::Buffer,
    ring_buffer_blue: wgpu::Buffer,
    ring_buffer_green: wgpu::Buffer,
    bind_group_blue: wgpu::BindGroup,
    bind_group_green: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    /// Split ring configuration
    pub config: SplitRingConfig,
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
struct RingUniform {
    /// Transform matrix for the ring (position + rotation + scale)
    transform: [[f32; 4]; 4],
    /// Ring parameters (inner_radius, outer_radius, offset, _padding)
    params: [f32; 4],
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

        // Create ring uniform buffers (separate for blue and green rings)
        let ring_uniform_blue = RingUniform {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
            params: [1.2, 1.4, 0.001, 0.0], // inner_radius, outer_radius, offset, padding
            color: [0.0, 0.0, 1.0, 1.0], // Blue
        };

        let ring_buffer_blue = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Split Ring Blue Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ring_uniform_blue]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ring_uniform_green = RingUniform {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
            params: [1.2, 1.4, 0.001, 0.0], // inner_radius, outer_radius, offset, padding
            color: [0.0, 1.0, 0.0, 1.0], // Green
        };

        let ring_buffer_green = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Split Ring Green Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ring_uniform_green]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Split Ring Bind Group Layout"),
            entries: &[
                // Camera uniform
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
                // Ring uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        // Create bind groups for both rings
        let bind_group_blue = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Split Ring Blue Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ring_buffer_blue.as_entire_binding(),
                },
            ],
        });

        let bind_group_green = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Split Ring Green Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ring_buffer_green.as_entire_binding(),
                },
            ],
        });

        // Create ring geometry
        let (vertices, indices) = Self::create_ring_geometry(32);

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

        // Create render pipelines with opposite face culling
        let render_pipeline_blue = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Split Ring Blue Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout.clone()],
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
            ring_buffer_blue,
            ring_buffer_green,
            bind_group_blue,
            bind_group_green,
            vertex_buffer,
            index_buffer,
            config: SplitRingConfig::default(),
        }
    }

    /// Create ring geometry (flat ring mesh)
    fn create_ring_geometry(segments: u32) -> (Vec<RingVertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Create ring vertices in local space (XY plane, centered at origin)
        // Inner radius = 1.2, outer radius = 1.4 (will be scaled by cell radius in transform)
        let inner_radius = 1.2;
        let outer_radius = 1.4;

        for i in 0..segments {
            let angle1 = (2.0 * std::f32::consts::PI * i as f32) / segments as f32;
            let angle2 = (2.0 * std::f32::consts::PI * ((i + 1) % segments) as f32) / segments as f32;

            let cos1 = angle1.cos();
            let sin1 = angle1.sin();
            let cos2 = angle2.cos();
            let sin2 = angle2.sin();

            // Inner and outer vertices for this segment
            let inner1 = Vec3::new(cos1 * inner_radius, sin1 * inner_radius, 0.0);
            let outer1 = Vec3::new(cos1 * outer_radius, sin1 * outer_radius, 0.0);
            let inner2 = Vec3::new(cos2 * inner_radius, sin2 * inner_radius, 0.0);
            let outer2 = Vec3::new(cos2 * outer_radius, sin2 * outer_radius, 0.0);

            let normal = Vec3::Z; // Ring faces up in local space

            let base_idx = vertices.len() as u16;

            // Add vertices (inner1, outer1, inner2, outer2)
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

            // Triangle 1: inner1, outer1, inner2
            indices.push(base_idx);
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);

            // Triangle 2: outer1, outer2, inner2
            indices.push(base_idx + 1);
            indices.push(base_idx + 3);
            indices.push(base_idx + 2);
        }

        (vertices, indices)
    }

    /// Update camera and ring uniforms
    fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_proj_matrix: Mat4,
        camera_position: Vec3,
        cell_position: Vec3,
        cell_rotation: Quat,
        cell_radius: f32,
        split_direction: Vec3,
        ring_color: [f32; 4],
        offset_sign: f32, // +1.0 or -1.0 for the two rings
        is_blue_ring: bool,
    ) {
        // Update camera uniform (shared between both rings)
        let camera_uniform = CameraUniform {
            view_proj: view_proj_matrix.to_cols_array_2d(),
            camera_pos: camera_position.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Calculate ring transform
        // The ring is positioned perpendicular to the split direction
        let offset_distance = cell_radius * self.config.offset_factor * offset_sign;
        let ring_position = cell_position + split_direction * offset_distance;

        // Create rotation to align ring perpendicular to split direction
        // The ring is created in XY plane (normal = Z), we want it perpendicular to split_direction
        // So we need to rotate Z to be perpendicular to split_direction
        // This means we want the ring's normal to be the split_direction
        let ring_rotation = if split_direction.dot(Vec3::Z).abs() < 0.99 {
            Quat::from_rotation_arc(Vec3::Z, split_direction)
        } else {
            // If split direction is very close to Z, use a different approach
            if split_direction.z > 0.0 {
                Quat::IDENTITY
            } else {
                Quat::from_rotation_x(std::f32::consts::PI)
            }
        };

        // Apply cell's rotation to the split direction
        let world_ring_rotation = cell_rotation * ring_rotation;

        let transform = Mat4::from_translation(ring_position) 
            * Mat4::from_quat(world_ring_rotation)
            * Mat4::from_scale(Vec3::splat(cell_radius));

        let ring_uniform = RingUniform {
            transform: transform.to_cols_array_2d(),
            params: [
                self.config.inner_radius_factor,
                self.config.outer_radius_factor,
                self.config.offset_factor,
                0.0,
            ],
            color: ring_color,
        };

        // Update the appropriate ring buffer
        let buffer = if is_blue_ring {
            &self.ring_buffer_blue
        } else {
            &self.ring_buffer_green
        };
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[ring_uniform]));
    }

    /// Update split ring configuration
    pub fn update_config(&mut self, config: &SplitRingConfig) {
        self.config = config.clone();
    }

    /// Render split rings for a cell within an existing render pass
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

        let index_count = (self.config.segments * 6) as u32; // 2 triangles per segment

        // Set vertex/index buffers once
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // Render blue ring (positive offset) with back face culling
        self.update_uniforms(
            queue,
            view_proj_matrix,
            camera_position,
            cell_position,
            cell_rotation,
            cell_radius,
            split_direction,
            [0.0, 0.0, 1.0, 1.0], // Blue
            1.0,
            true, // is_blue_ring
        );

        render_pass.set_pipeline(&self.render_pipeline_blue);
        render_pass.set_bind_group(0, &self.bind_group_blue, &[]);
        render_pass.draw_indexed(0..index_count, 0, 0..1);

        // Render green ring (negative offset) with front face culling
        self.update_uniforms(
            queue,
            view_proj_matrix,
            camera_position,
            cell_position,
            cell_rotation,
            cell_radius,
            split_direction,
            [0.0, 1.0, 0.0, 1.0], // Green
            -1.0,
            false, // is_blue_ring
        );

        render_pass.set_pipeline(&self.render_pipeline_green);
        render_pass.set_bind_group(0, &self.bind_group_green, &[]);
        render_pass.draw_indexed(0..index_count, 0, 0..1);
    }

    /// Handle window resize
    pub fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // No resize-dependent resources
    }
}