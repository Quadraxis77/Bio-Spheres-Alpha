//! Orientation gizmo rendering with wgpu.
//!
//! Provides 3D orientation gizmos showing X, Y, Z axes with interactive visual feedback.

use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// Renderer for 3D orientation gizmos.
///
/// Renders colored axis lines (X=blue, Y=green, Z=red) with optional labels
/// and interactive highlighting for spatial orientation feedback.
pub struct OrientationGizmoRenderer {
    render_pipeline: wgpu::RenderPipeline,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    gizmo_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    /// Gizmo configuration
    pub config: GizmoConfig,
}

/// Configuration for orientation gizmo appearance and behavior
#[derive(Debug, Clone)]
pub struct GizmoConfig {
    /// Whether the gizmo is visible
    pub visible: bool,
    /// Size of the gizmo in world units
    pub size: f32,
    /// Opacity of the gizmo (0.0 to 1.0)
    pub opacity: f32,
    /// Length of each axis line
    pub axis_length: f32,
    /// Thickness of axis lines (for future use)
    pub line_thickness: f32,
}

impl GizmoConfig {
    /// Create gizmo config from UI editor state
    pub fn from_editor_state(editor_state: &crate::ui::panel_context::GenomeEditorState) -> Self {
        Self {
            visible: editor_state.gizmo_visible,
            size: 0.8, // Fixed size in world units
            opacity: 1.0, // Fixed opacity
            axis_length: 1.0,
            line_thickness: 3.0,
        }
    }
}

impl Default for GizmoConfig {
    fn default() -> Self {
        Self {
            visible: true,
            size: 1.0, // 100% of cell radius
            opacity: 0.8,
            axis_length: 1.0,
            line_thickness: 3.0,
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
struct GizmoUniform {
    /// Transform matrix for the gizmo (position + rotation)
    transform: [[f32; 4]; 4],
    /// Gizmo parameters (size, opacity, _padding, _padding)
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GizmoVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl OrientationGizmoRenderer {
    /// Create a new orientation gizmo renderer.
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
            label: Some("Gizmo Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create gizmo uniform buffer
        let gizmo_uniform = GizmoUniform {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
            params: [0.8, 0.8, 0.0, 0.0], // size, opacity, padding, padding
        };

        let gizmo_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gizmo Uniform Buffer"),
            contents: bytemuck::cast_slice(&[gizmo_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gizmo Bind Group Layout"),
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
                // Gizmo uniform
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

        // Create bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gizmo Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gizmo_buffer.as_entire_binding(),
                },
            ],
        });

        // Create gizmo geometry (3 axis lines)
        let (vertices, indices) = Self::create_gizmo_geometry();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gizmo Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gizmo Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gizmo Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/orientation_gizmo.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gizmo Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Define vertex buffer layout
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GizmoVertex>() as wgpu::BufferAddress,
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
            label: Some("Gizmo Pipeline"),
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
                depth_write_enabled: true, // Write depth so gizmo can occlude itself
                depth_compare: wgpu::CompareFunction::Less, // Standard depth testing
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

        // No longer need depth texture since we render in shared render pass

        Self {
            render_pipeline,
            camera_bind_group,
            camera_buffer,
            gizmo_buffer,
            vertex_buffer,
            index_buffer,
            config: GizmoConfig::default(),
        }
    }

    /// Create gizmo geometry (3 colored axis lines protruding from sphere surface)
    fn create_gizmo_geometry() -> (Vec<GizmoVertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Lines start from sphere surface (radius 1.0) and extend outward
        // The shader will scale these by the actual cell radius
        
        // X axis (blue) - from sphere surface outward
        vertices.push(GizmoVertex {
            position: [1.0, 0.0, 0.0], // Start at sphere surface
            color: [0.0, 0.0, 1.0, 1.0], // Blue
        });
        vertices.push(GizmoVertex {
            position: [2.0, 0.0, 0.0], // Extend to 2x radius
            color: [0.0, 0.0, 1.0, 1.0], // Blue
        });

        // Y axis (green) - from sphere surface outward
        vertices.push(GizmoVertex {
            position: [0.0, 1.0, 0.0], // Start at sphere surface
            color: [0.0, 1.0, 0.0, 1.0], // Green
        });
        vertices.push(GizmoVertex {
            position: [0.0, 2.0, 0.0], // Extend to 2x radius
            color: [0.0, 1.0, 0.0, 1.0], // Green
        });

        // Z axis (red) - from sphere surface outward
        vertices.push(GizmoVertex {
            position: [0.0, 0.0, 1.0], // Start at sphere surface
            color: [1.0, 0.0, 0.0, 1.0], // Red
        });
        vertices.push(GizmoVertex {
            position: [0.0, 0.0, 2.0], // Extend to 2x radius
            color: [1.0, 0.0, 0.0, 1.0], // Red
        });

        // Line indices (3 lines, 2 vertices each)
        indices.extend_from_slice(&[0, 1, 2, 3, 4, 5]);

        (vertices, indices)
    }

    /// Update camera and gizmo uniforms
    fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_proj_matrix: Mat4,
        camera_position: Vec3,
        cell_position: Vec3,
        cell_rotation: Quat,
        cell_radius: f32,
    ) {
        // Update camera uniform with actual view-projection matrix and camera position
        let camera_uniform = CameraUniform {
            view_proj: view_proj_matrix.to_cols_array_2d(),
            camera_pos: camera_position.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Scale gizmo based on cell radius
        // The geometry extends to 2.0, so this will make the gizmo extend 2x the cell radius
        let gizmo_scale = cell_radius;
        
        // Create transform matrix: translate to cell position, rotate by cell orientation, scale by cell radius
        let transform = Mat4::from_translation(cell_position) 
            * Mat4::from_quat(cell_rotation) 
            * Mat4::from_scale(Vec3::splat(gizmo_scale));

        let gizmo_uniform = GizmoUniform {
            transform: transform.to_cols_array_2d(),
            params: [gizmo_scale, self.config.opacity, 0.0, 0.0],
        };
        queue.write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&[gizmo_uniform]));
    }

    /// Update gizmo configuration
    pub fn update_config(&mut self, config: &GizmoConfig) {
        self.config = config.clone();
    }

    /// Render the orientation gizmo within an existing render pass
    pub fn render_in_pass(
        &mut self,
        render_pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        view_proj_matrix: Mat4,
        camera_position: Vec3,
        cell_position: Vec3,
        cell_rotation: Quat,
        cell_radius: f32,
    ) {
        if !self.config.visible {
            return;
        }

        // Update uniforms
        self.update_uniforms(queue, view_proj_matrix, camera_position, cell_position, cell_rotation, cell_radius);

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..1); // 6 indices for 3 lines
    }

    /// Handle window resize
    pub fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // No longer need to manage depth texture since we use shared depth buffer
    }
}