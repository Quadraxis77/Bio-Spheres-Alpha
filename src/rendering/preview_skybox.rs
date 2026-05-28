// Preview scene skybox renderer.
//
// Renders a dark teal/navy gradient background with a perspective grid floor,
// matching the Bio-Spheres Lab genome editor aesthetic from the reference image.
//
// Uses the same fullscreen-triangle + rotation-only inv_view_rot_proj pattern
// as the GPU scene skybox to avoid shaking.

use bytemuck::{Pod, Zeroable};

/// Camera uniforms for the preview skybox shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PreviewSkyboxCamera {
    /// Inverse of (proj × rotation-only view) — for direction reconstruction.
    pub inv_view_rot_proj: [[f32; 4]; 4],
    /// Full view-proj — used for grid floor projection.
    pub view_proj: [[f32; 4]; 4],
    /// Camera world position — needed for floor ray intersection.
    pub camera_pos: [f32; 3],
    pub time: f32,
}

/// Preview scene skybox renderer.
pub struct PreviewSkyboxRenderer {
    pipeline:      wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
}

impl PreviewSkyboxRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Preview Skybox Camera Layout"),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Preview Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/preview_skybox.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Preview Skybox Pipeline Layout"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Preview Skybox Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None, // opaque background
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None, // no depth — renders at far plane via vertex shader
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Preview Skybox Camera Buffer"),
            size: std::mem::size_of::<PreviewSkyboxCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_layout,
            camera_buffer,
        }
    }

    /// Render the preview skybox.
    ///
    /// Call after the clear pass (which clears depth) and before cell rendering.
    /// `view_proj` is the full camera view-projection matrix.
    pub fn render(
        &self,
        encoder:    &mut wgpu::CommandEncoder,
        queue:      &wgpu::Queue,
        color_view: &wgpu::TextureView,
        device:     &wgpu::Device,
        view_proj:  glam::Mat4,
        camera_pos: glam::Vec3,
        time:       f32,
    ) {
        // Rotation-only view-proj: cancel the translation so the sky doesn't shake.
        // view_proj = P * V,  V = R * T(-cam)
        // view_proj * T(+cam) = P * R  (translation cancels)
        let rot_view_proj     = view_proj * glam::Mat4::from_translation(camera_pos);
        let inv_view_rot_proj = rot_view_proj.inverse();

        let uniform = PreviewSkyboxCamera {
            inv_view_rot_proj: inv_view_rot_proj.to_cols_array_2d(),
            view_proj:         view_proj.to_cols_array_2d(),
            camera_pos:        camera_pos.to_array(),
            time,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preview Skybox Camera BG"),
            layout: &self.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Preview Skybox Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // preserve cleared depth from clear pass
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &camera_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
