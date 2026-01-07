//! GPU-based adhesion line rendering.
//!
//! Renders adhesion connections directly from GPU buffers without CPU readback.
//! Uses instanced rendering where each instance is one adhesion connection,
//! and each instance draws 4 vertices (2 line segments).

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// GPU-based renderer for adhesion connection lines.
/// Reads directly from GPU adhesion buffers - no CPU readback needed.
pub struct GpuAdhesionLineRenderer {
    render_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    camera_bind_group_layout: wgpu::BindGroupLayout,
    data_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    max_adhesions: u32,
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

impl GpuAdhesionLineRenderer {
    /// Create a new GPU adhesion line renderer.
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        max_adhesions: u32,
    ) -> Self {
        // Create camera buffer
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 50.0],
            _padding: 0.0,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Adhesion Line Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Camera bind group layout (group 0)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GPU Adhesion Line Camera Layout"),
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
            label: Some("GPU Adhesion Line Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Data bind group layout (group 1) - reads from GPU buffers
        let data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GPU Adhesion Line Data Layout"),
                entries: &[
                    // Binding 0: Cell positions
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: Adhesion connections
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: Adhesion counts
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: Cell count buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Adhesion Line Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/gpu_adhesion_line.wgsl").into(),
            ),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU Adhesion Line Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &data_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline - no vertex buffer needed, all data from storage buffers
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GPU Adhesion Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers - data comes from storage buffers
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
            camera_bind_group_layout,
            data_bind_group_layout,
            camera_bind_group,
            camera_buffer,
            max_adhesions,
            width: config.width,
            height: config.height,
        }
    }

    /// Create a data bind group for the given buffers.
    /// Call this each frame with the current output position buffer.
    pub fn create_data_bind_group(
        &self,
        device: &wgpu::Device,
        positions_buffer: &wgpu::Buffer,
        adhesion_connections_buffer: &wgpu::Buffer,
        adhesion_counts_buffer: &wgpu::Buffer,
        cell_count_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Adhesion Line Data Bind Group"),
            layout: &self.data_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_connections_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Render adhesion lines using GPU buffers.
    /// Must be called within an existing render pass.
    pub fn render_in_pass(
        &self,
        render_pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        data_bind_group: &wgpu::BindGroup,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
    ) {
        // Update camera uniform
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

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Set up render pass
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);

        // Draw: 4 vertices per instance (2 line segments), max_adhesions instances
        // The shader will skip inactive adhesions by outputting degenerate vertices
        render_pass.draw(0..4, 0..self.max_adhesions);
    }
}
