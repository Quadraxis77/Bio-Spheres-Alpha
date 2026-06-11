//! Boulder Renderer
//!
//! Renders boulders by reading directly from the GPU physics storage buffers.
//! No CPU->GPU instance upload needed - the vertex shader indexes into
//! `boulder_state` and `boulder_moss_dir` by `instance_index`.
//! Dead boulders are culled in the vertex shader by outputting degenerate geometry.
//!
//! Draw call: `draw(0..240, 0..MAX_BOULDERS)` - always draws all slots,
//! dead ones collapse to a zero-area triangle and are discarded by the rasterizer.

use bytemuck::{Pod, Zeroable};

use crate::simulation::gpu_physics::boulder_buffers::{BoulderBuffers, MAX_BOULDERS};

/// Number of vertices per boulder instance (20 faces x 4 sub-tris x 3 corners).
pub const VERTICES_PER_BOULDER: u32 = 240;

/// Camera uniform matching the cell renderer layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    lod_scale_factor: f32,
    lod_threshold_low: f32,
    lod_threshold_medium: f32,
    lod_threshold_high: f32,
}

/// Lighting uniform matching the cell renderer layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LightingUniform {
    light_dir: [f32; 3],
    ambient: f32,
    light_color: [f32; 3],
    outline_width: f32,
}

pub struct BoulderRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    /// Bind group for boulder storage buffers (group 1).
    /// Rebuilt when boulder system is initialized.
    pub boulder_bind_group: Option<wgpu::BindGroup>,
    boulder_bind_group_layout: wgpu::BindGroupLayout,
    pub width: u32,
    pub height: u32,
}

impl BoulderRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // -- Group 0: camera + lighting ----------------------------------------
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Boulder Camera BGL"),
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

        // -- Group 1: boulder storage buffers ---------------------------------
        let ro = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let boulder_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Boulder Storage BGL"),
                // state(ro), count(ro)
                entries: &[ro(0), ro(1)],
            });

        // -- Buffers -----------------------------------------------------------
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boulder Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boulder Lighting Buffer"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_lighting = LightingUniform {
            light_dir: [-0.5, -0.7, -0.5],
            ambient: 0.15,
            light_color: [1.0, 0.98, 0.95],
            outline_width: 0.0,
        };
        queue.write_buffer(&lighting_buffer, 0, bytemuck::bytes_of(&default_lighting));

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Boulder Camera BG"),
            layout: &camera_bgl,
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

        // -- Shader ------------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Boulder Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/boulder.wgsl").into()),
        });

        // -- Pipeline layout ---------------------------------------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Boulder Pipeline Layout"),
            bind_group_layouts: &[&camera_bgl, &boulder_bind_group_layout],
            push_constant_ranges: &[],
        });

        // -- Render pipeline - no vertex buffers, geometry is procedural -------
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Boulder Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // no vertex buffer - reads from storage buffers
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_buffer,
            lighting_buffer,
            camera_bind_group,
            boulder_bind_group: None,
            boulder_bind_group_layout,
            width,
            height,
        }
    }

    /// Build the boulder storage bind group from the real GPU buffers.
    /// Call once after BoulderBuffers is created.
    pub fn create_boulder_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &BoulderBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Boulder Storage BG"),
            layout: &self.boulder_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.boulder_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.boulder_count.as_entire_binding(),
                },
            ],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn set_light_dir(&self, queue: &wgpu::Queue, dir: [f32; 3]) {
        let lighting = LightingUniform {
            light_dir: dir,
            ambient: 0.15,
            light_color: [1.0, 0.98, 0.95],
            outline_width: 0.0,
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
    }

    /// Render all boulders. Draws MAX_BOULDERS instances; dead ones are culled
    /// in the vertex shader by outputting degenerate geometry.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: glam::Vec3,
        camera_rotation: glam::Quat,
        current_time: f32,
        horizontal_fov_degrees: f32,
        active_count: u32,
    ) {
        // Don't render if no boulders or no bind group yet
        let boulder_bg = match &self.boulder_bind_group {
            Some(bg) => bg,
            None => return,
        };
        if active_count == 0 {
            return;
        }

        // Update camera uniform
        let view = glam::Mat4::from_rotation_translation(camera_rotation, camera_pos).inverse();
        let aspect = self.width as f32 / self.height as f32;
        let proj = glam::Mat4::perspective_rh(
            crate::ui::camera::CameraController::vertical_fov_radians_for_horizontal(
                horizontal_fov_degrees,
                aspect,
            ),
            aspect,
            0.1,
            5000.0,
        );
        let view_proj = proj * view;

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time: current_time,
            lod_scale_factor: 1.0,
            lod_threshold_low: 0.0,
            lod_threshold_medium: 0.0,
            lod_threshold_high: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Boulder Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_bind_group(1, boulder_bg, &[]);
        // Draw MAX_BOULDERS instances - dead ones output degenerate triangles
        pass.draw(0..VERTICES_PER_BOULDER, 0..MAX_BOULDERS);
    }
}
