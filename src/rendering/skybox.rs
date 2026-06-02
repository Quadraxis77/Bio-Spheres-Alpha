// Skybox rendering
//
// Procedural space skybox - renders a fullscreen starfield as the background.
// Uses the same fullscreen-triangle pattern as SunRenderer:
//   - No vertex buffer; 3 vertices generated in the vertex shader
//   - Depth = 1.0 (far plane) so all scene geometry draws over it
//   - LoadOp::Load to preserve the cleared color/depth from the clear pass
//
// The fragment shader reconstructs the world-space view direction from the
// inverse view-projection matrix and samples a procedural star field.

use bytemuck::{Pod, Zeroable};

/// Camera uniforms shared with the skybox shader (must match shader struct).
///
/// Uses a rotation-only (translation-stripped) inverse view-projection so the
/// reconstructed ray direction is purely orientation-based. This avoids the
/// catastrophic cancellation that causes shaking when `world_pos - camera_pos`
/// is computed with large world-space coordinates.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SkyboxCameraUniforms {
    /// Inverse of (proj * rotation-only view) - no translation component.
    /// Multiply by NDC to get a world-space direction directly.
    pub inv_view_rot_proj: [[f32; 4]; 4],
    pub time:              f32,
    pub _pad0:             f32,
    pub _pad1:             f32,
    pub _pad2:             f32,
}

/// Skybox appearance parameters (must match shader struct).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SkyboxParams {
    /// Controls how many stars appear (0.0-1.0; default 0.8)
    pub star_density:        f32,
    /// Overall star brightness multiplier (default 1.0)
    pub star_brightness:     f32,
    /// How fast stars twinkle in Hz (default 1.5)
    pub twinkle_speed:       f32,
    /// Twinkle amplitude - 0 = no twinkle, 1 = full twinkle (default 0.25)
    pub twinkle_amount:      f32,
    /// Nebula color cloud intensity (default 1.0)
    pub nebula_intensity:    f32,
    /// Milky Way band brightness (default 1.0)
    pub milky_way_intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

impl Default for SkyboxParams {
    fn default() -> Self {
        Self {
            star_density:        0.8,
            star_brightness:     1.0,
            twinkle_speed:       1.5,
            twinkle_amount:      0.25,
            nebula_intensity:    1.0,
            milky_way_intensity: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Procedural space skybox renderer.
///
/// Renders a fullscreen starfield behind all scene geometry.
/// Call [`SkyboxRenderer::render`] immediately after the clear pass and
/// before any opaque geometry.
pub struct SkyboxRenderer {
    pipeline:       wgpu::RenderPipeline,
    camera_layout:  wgpu::BindGroupLayout,
    params_layout:  wgpu::BindGroupLayout,
    camera_buffer:  wgpu::Buffer,
    params_buffer:  wgpu::Buffer,

    /// Publicly configurable appearance parameters.
    pub params: SkyboxParams,
}

impl SkyboxRenderer {
    /// Create a new skybox renderer.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        // -- Bind group layouts ------------------------------------------------

        // Group 0: camera uniforms
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox Camera Layout"),
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

        // Group 1: skybox appearance params
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skybox Params Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // -- Shader ------------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/skybox.wgsl").into()),
        });

        // -- Pipeline ----------------------------------------------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &params_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // fullscreen triangle - no vertex buffer
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Opaque write - the skybox IS the background
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // fullscreen triangle - no culling
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // No depth attachment - the skybox writes at depth 1.0 via the
            // vertex shader; we don't need a depth test or write here.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // -- GPU buffers -------------------------------------------------------
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skybox Camera Buffer"),
            size: std::mem::size_of::<SkyboxCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skybox Params Buffer"),
            size: std::mem::size_of::<SkyboxParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_layout,
            params_layout,
            camera_buffer,
            params_buffer,
            params: SkyboxParams::default(),
        }
    }

    /// Render the skybox as a fullscreen background pass.
    ///
    /// Call this **after** the clear pass and **before** any opaque geometry.
    ///
    /// `view_proj` is the full camera view-projection matrix. The renderer
    /// strips the translation internally so the skybox never shakes as the
    /// camera moves.
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
        // Build a rotation-only view-projection by cancelling the camera
        // translation. Multiplying view_proj by T(+camera_pos) on the right
        // is equivalent to rebuilding the view matrix without its translation
        // column, then multiplying by the projection matrix.
        //
        // view_proj = P * V
        // V = R * T(-cam)   (rotation then translate-to-origin)
        // V * T(+cam) = R   (translation cancels)
        // rot_view_proj = P * R = view_proj * T(+cam)
        let rot_view_proj = view_proj * glam::Mat4::from_translation(camera_pos);
        let inv_view_rot_proj = rot_view_proj.inverse();

        let camera_uniform = SkyboxCameraUniforms {
            inv_view_rot_proj: inv_view_rot_proj.to_cols_array_2d(),
            time,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Upload appearance params
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // Build bind groups (cheap - no textures, just two small uniforms)
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Camera Bind Group"),
            layout: &self.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_buffer.as_entire_binding(),
            }],
        });

        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Params Bind Group"),
            layout: &self.params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.params_buffer.as_entire_binding(),
            }],
        });

        // Render pass - color only, no depth attachment
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skybox Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load preserves the cleared depth from the clear pass.
                        // The skybox writes opaque color over the black clear.
                        load: wgpu::LoadOp::Load,
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
            pass.set_bind_group(1, &params_bg, &[]);
            pass.draw(0..3, 0..1); // fullscreen triangle
        }
    }
}
