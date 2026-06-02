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
    /// Inverse of (proj x rotation-only view) - for direction reconstruction.
    pub inv_view_rot_proj: [[f32; 4]; 4],
    /// Full view-proj - used for grid floor projection.
    pub view_proj: [[f32; 4]; 4],
    /// Camera world position - needed for floor ray intersection.
    pub camera_pos: [f32; 3],
    pub time: f32,
}

/// Theme-driven color params for the preview skybox (must match shader struct).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PreviewSkyboxThemeParams {
    pub zenith_color: [f32; 3],
    pub _pad0: f32,
    pub horizon_color: [f32; 3],
    pub _pad1: f32,
    pub glow_color: [f32; 3],
    pub _pad2: f32,
    pub grid_color: [f32; 3],
    pub grid_opacity: f32,
    pub floor_color: [f32; 3],
    pub _pad3: f32,
}

impl Default for PreviewSkyboxThemeParams {
    fn default() -> Self {
        Self {
            zenith_color: [0.012, 0.020, 0.055],
            _pad0: 0.0,
            horizon_color: [0.018, 0.065, 0.085],
            _pad1: 0.0,
            glow_color: [0.0, 0.04, 0.06],
            _pad2: 0.0,
            grid_color: [0.05, 0.55, 0.65],
            grid_opacity: 0.35,
            floor_color: [0.008, 0.012, 0.020],
            _pad3: 0.0,
        }
    }
}

impl PreviewSkyboxThemeParams {
    /// Return hand-tuned skybox colors for each theme.
    ///
    /// Each theme gets a sky gradient and grid that complement its UI palette:
    /// - Dark themes: near-black sky, accent-tinted horizon, bright accent grid
    /// - Light themes: pale sky matching the theme hue, subtle grid
    /// - High-contrast: pure black sky, full-brightness white/yellow grid
    pub fn from_palette(p: &crate::ui::ui_system::ActivePalette) -> Self {
        use crate::ui::types::UiTheme;

        // Helper: rgb bytes -> [f32; 3]
        let rgb = |r: u8, g: u8, b: u8| -> [f32; 3] {
            [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
        };

        // Floor is always near-black so grid lines are always visible.
        // The exact hue matches the theme's darkest background.
        let floor = |r: u8, g: u8, b: u8| -> [f32; 3] {
            // Clamp to a very dark range so no theme produces a bright floor.
            let scale = 0.06_f32;
            [
                r as f32 / 255.0 * scale,
                g as f32 / 255.0 * scale,
                b as f32 / 255.0 * scale,
            ]
        };

        match p.theme {
            UiTheme::BiotechDark => Self {
                zenith_color: rgb(3, 5, 14),
                _pad0: 0.0,
                horizon_color: rgb(4, 16, 22),
                _pad1: 0.0,
                glow_color: rgb(0, 10, 15),
                _pad2: 0.0,
                grid_color: rgb(20, 200, 160),
                grid_opacity: 0.40,
                floor_color: floor(0, 220, 175),
                _pad3: 0.0,
            },
            UiTheme::Arctic => Self {
                zenith_color: rgb(8, 14, 28),
                _pad0: 0.0,
                horizon_color: rgb(12, 22, 48),
                _pad1: 0.0,
                glow_color: rgb(5, 12, 30),
                _pad2: 0.0,
                grid_color: rgb(60, 140, 255),
                grid_opacity: 0.45,
                floor_color: floor(10, 90, 200),
                _pad3: 0.0,
            },
            UiTheme::Parchment => Self {
                zenith_color: rgb(10, 6, 2),
                _pad0: 0.0,
                horizon_color: rgb(18, 10, 4),
                _pad1: 0.0,
                glow_color: rgb(12, 5, 2),
                _pad2: 0.0,
                grid_color: rgb(200, 90, 30),
                grid_opacity: 0.45,
                floor_color: floor(165, 55, 15),
                _pad3: 0.0,
            },
            UiTheme::Blossom => Self {
                zenith_color: rgb(10, 2, 6),
                _pad0: 0.0,
                horizon_color: rgb(18, 4, 10),
                _pad1: 0.0,
                glow_color: rgb(12, 2, 6),
                _pad2: 0.0,
                grid_color: rgb(220, 40, 120),
                grid_opacity: 0.45,
                floor_color: floor(185, 10, 105),
                _pad3: 0.0,
            },
            UiTheme::Crimson => Self {
                zenith_color: rgb(6, 1, 2),
                _pad0: 0.0,
                horizon_color: rgb(14, 4, 6),
                _pad1: 0.0,
                glow_color: rgb(10, 4, 0),
                _pad2: 0.0,
                grid_color: rgb(225, 178, 35),
                grid_opacity: 0.50,
                floor_color: floor(225, 178, 35),
                _pad3: 0.0,
            },
            UiTheme::NeonSynthwave => Self {
                zenith_color: rgb(3, 1, 8),
                _pad0: 0.0,
                horizon_color: rgb(8, 2, 14),
                _pad1: 0.0,
                glow_color: rgb(10, 0, 10),
                _pad2: 0.0,
                grid_color: rgb(255, 20, 175),
                grid_opacity: 0.50,
                floor_color: floor(255, 20, 175),
                _pad3: 0.0,
            },
            UiTheme::NeonToxic => Self {
                zenith_color: rgb(0, 3, 0),
                _pad0: 0.0,
                horizon_color: rgb(1, 8, 1),
                _pad1: 0.0,
                glow_color: rgb(0, 6, 0),
                _pad2: 0.0,
                grid_color: rgb(40, 255, 40),
                grid_opacity: 0.55,
                floor_color: floor(40, 255, 40),
                _pad3: 0.0,
            },
            UiTheme::NeonUltraviolet => Self {
                zenith_color: rgb(2, 0, 6),
                _pad0: 0.0,
                horizon_color: rgb(5, 1, 12),
                _pad1: 0.0,
                glow_color: rgb(4, 0, 8),
                _pad2: 0.0,
                grid_color: rgb(175, 25, 255),
                grid_opacity: 0.55,
                floor_color: floor(175, 25, 255),
                _pad3: 0.0,
            },
            UiTheme::HighContrast => Self {
                zenith_color: rgb(0, 0, 0),
                _pad0: 0.0,
                horizon_color: rgb(0, 0, 0),
                _pad1: 0.0,
                glow_color: rgb(0, 0, 0),
                _pad2: 0.0,
                grid_color: rgb(255, 230, 0),
                grid_opacity: 0.80,
                floor_color: [0.0, 0.0, 0.0],
                _pad3: 0.0,
            },
            // -- CUSTOM -------------------------------------------------------
            // Derive from the active palette's bg and accent colors.
            UiTheme::Custom => {
                let bg = p.bg_darkest;
                let acc = p.accent_primary;
                // Sky: very dark tint of bg_darkest
                let zenith = [
                    bg.r() as f32 / 255.0 * 0.5,
                    bg.g() as f32 / 255.0 * 0.5,
                    bg.b() as f32 / 255.0 * 0.5,
                ];
                let horizon = [
                    bg.r() as f32 / 255.0 * 0.5 + acc.r() as f32 / 255.0 * 0.06,
                    bg.g() as f32 / 255.0 * 0.5 + acc.g() as f32 / 255.0 * 0.06,
                    bg.b() as f32 / 255.0 * 0.5 + acc.b() as f32 / 255.0 * 0.06,
                ];
                let glow = [
                    acc.r() as f32 / 255.0 * 0.04,
                    acc.g() as f32 / 255.0 * 0.04,
                    acc.b() as f32 / 255.0 * 0.04,
                ];
                let grid = [
                    acc.r() as f32 / 255.0 * 0.8,
                    acc.g() as f32 / 255.0 * 0.8,
                    acc.b() as f32 / 255.0 * 0.8,
                ];
                let floor = [
                    acc.r() as f32 / 255.0 * 0.04,
                    acc.g() as f32 / 255.0 * 0.04,
                    acc.b() as f32 / 255.0 * 0.04,
                ];
                Self {
                    zenith_color: zenith,
                    _pad0: 0.0,
                    horizon_color: horizon,
                    _pad1: 0.0,
                    glow_color: glow,
                    _pad2: 0.0,
                    grid_color: grid,
                    grid_opacity: 0.45,
                    floor_color: floor,
                    _pad3: 0.0,
                }
            }
        }
    }
}

/// Preview scene skybox renderer.
pub struct PreviewSkyboxRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
    theme_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    theme_buffer: wgpu::Buffer,
    /// Current theme params - update each frame via `set_theme_params`.
    pub theme_params: PreviewSkyboxThemeParams,
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

        let theme_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Preview Skybox Theme Layout"),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Preview Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/preview_skybox.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Preview Skybox Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &theme_layout],
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
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
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

        let theme_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Preview Skybox Theme Buffer"),
            size: std::mem::size_of::<PreviewSkyboxThemeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_layout,
            theme_layout,
            camera_buffer,
            theme_buffer,
            theme_params: PreviewSkyboxThemeParams::default(),
        }
    }

    /// Render the preview skybox.
    ///
    /// Call after the clear pass (which clears depth) and before cell rendering.
    /// `view_proj` is the full camera view-projection matrix.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        device: &wgpu::Device,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        time: f32,
    ) {
        // Rotation-only view-proj: cancel the translation so the sky doesn't shake.
        // view_proj = P * V,  V = R * T(-cam)
        // view_proj * T(+cam) = P * R  (translation cancels)
        let rot_view_proj = view_proj * glam::Mat4::from_translation(camera_pos);
        let inv_view_rot_proj = rot_view_proj.inverse();

        let uniform = PreviewSkyboxCamera {
            inv_view_rot_proj: inv_view_rot_proj.to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(
            &self.theme_buffer,
            0,
            bytemuck::bytes_of(&self.theme_params),
        );

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preview Skybox Camera BG"),
            layout: &self.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_buffer.as_entire_binding(),
            }],
        });

        let theme_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preview Skybox Theme BG"),
            layout: &self.theme_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.theme_buffer.as_entire_binding(),
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
            pass.set_bind_group(1, &theme_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
