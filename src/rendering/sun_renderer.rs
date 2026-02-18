//! Procedural Sun Renderer
//!
//! Full-screen post-process that renders a procedural sun at infinite distance
//! along the light direction vector. Features:
//!   - Animated sun disk with sunspots and solar flares
//!   - Corona glow with animated tendrils
//!   - Volumetric sun rays
//!   - Lens flare effects (ghosts, halo, starburst)
//!   - Eclipse support: depth buffer occlusion fades all effects

use bytemuck::{Pod, Zeroable};

/// Camera uniforms for sun renderer (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SunCameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub time: f32,
}

/// Sun parameters (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SunParams {
    // Light direction (normalized, pointing toward sun)
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    // Sun angular radius in radians (visual size)
    pub sun_angular_radius: f32,
    // Sun color (RGB)
    pub sun_color_r: f32,
    pub sun_color_g: f32,
    pub sun_color_b: f32,
    // Sun intensity multiplier
    pub sun_intensity: f32,
    // Corona parameters
    pub corona_radius: f32,
    pub corona_intensity: f32,
    pub corona_falloff: f32,
    // Lens flare parameters
    pub flare_intensity: f32,
    pub flare_ghost_count: f32,
    pub flare_ghost_dispersal: f32,
    pub flare_halo_radius: f32,
    // Sun ray parameters
    pub ray_intensity: f32,
    pub ray_count: f32,
    pub ray_falloff: f32,
    // Eclipse occlusion (0.0 = fully eclipsed, 1.0 = fully visible)
    pub eclipse_factor: f32,
    // Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
    // Solar flare parameters
    pub flare_speed: f32,
    pub sunspot_scale: f32,
    // Additional flare settings
    pub starburst_intensity: f32,
    pub starburst_points: f32,
    pub starburst_falloff: f32,
    pub streak_intensity: f32,
    pub streak_width: f32,
    pub ghost_size: f32,
    pub chromatic_aberration: f32,
    pub prominence_intensity: f32,
    pub glow_intensity: f32,
    pub prominence_extent: f32,
}

impl Default for SunParams {
    fn default() -> Self {
        Self {
            light_dir_x: 0.5,
            light_dir_y: 0.8,
            light_dir_z: 0.3,
            sun_angular_radius: 0.04,
            sun_color_r: 1.0,
            sun_color_g: 0.9,
            sun_color_b: 0.7,
            sun_intensity: 2.0,
            corona_radius: 4.0,
            corona_intensity: 0.8,
            corona_falloff: 2.0,
            flare_intensity: 0.5,
            flare_ghost_count: 5.0,
            flare_ghost_dispersal: 1.5,
            flare_halo_radius: 0.25,
            ray_intensity: 0.3,
            ray_count: 12.0,
            ray_falloff: 4.0,
            eclipse_factor: 1.0,
            screen_width: 1280.0,
            screen_height: 720.0,
            flare_speed: 1.0,
            sunspot_scale: 5.0,
            starburst_intensity: 0.4,
            starburst_points: 6.0,
            starburst_falloff: 8.0,
            streak_intensity: 0.3,
            streak_width: 80.0,
            ghost_size: 0.02,
            chromatic_aberration: 0.15,
            prominence_intensity: 1.0,
            glow_intensity: 0.15,
            prominence_extent: 3.0,
        }
    }
}

/// Procedural sun renderer
pub struct SunRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
    sun_data_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    sun_params_buffer: wgpu::Buffer,
    depth_sampler: wgpu::Sampler,

    // Configurable parameters (exposed for UI)
    pub sun_angular_radius: f32,
    pub sun_color: [f32; 3],
    pub enabled: bool,

    // Screen dimensions
    width: u32,
    height: u32,
}

impl SunRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // Camera bind group layout (group 0)
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sun Camera Layout"),
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

        // Sun data bind group layout (group 1)
        let sun_data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sun Data Layout"),
            entries: &[
                // Binding 0: SunParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Procedural Sun Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sun.wgsl").into(),
            ),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sun Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &sun_data_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline (full-screen triangle, additive blending)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sun Pipeline"),
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
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create camera buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sun Camera Buffer"),
            size: std::mem::size_of::<SunCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create sun params buffer
        let sun_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sun Params Buffer"),
            size: std::mem::size_of::<SunParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth sampler (non-filtering for depth texture)
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sun Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            camera_layout,
            sun_data_layout,
            camera_buffer,
            sun_params_buffer,
            depth_sampler,
            sun_angular_radius: 0.025,
            sun_color: [1.0, 1.0, 0.85],
            enabled: true,
            width: 1280,
            height: 720,
        }
    }

    /// Resize the renderer (update screen dimensions)
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Render the sun as a full-screen post-process pass
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        device: &wgpu::Device,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        time: f32,
        light_dir: [f32; 3],
        sun_intensity: f32,
    ) {
        // Update camera uniforms
        let inv_view_proj = view_proj.inverse();
        let camera_uniform = SunCameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Procedural sun settings
    let corona_radius = 2.0;
    let corona_intensity = 3.0;
    let corona_falloff = 5.0;
    let flare_intensity = 0.0;
    let flare_ghost_count = 1.0;
    let flare_ghost_dispersal = 3.0;
    let flare_halo_radius = 0.05;
    let ray_intensity = 0.0;
    let ray_count = 8.0;
    let ray_falloff = 6.0;
    let flare_speed = 1.0;
    let sunspot_scale = 20.0;
    let starburst_intensity = 1.05;
    let starburst_points = 6.0;
    let starburst_falloff = 8.0;
    let streak_intensity = 1.0;
    let streak_width = 175.0;
    let ghost_size = 0.005;
    let chromatic_aberration = 0.29;
    let prominence_intensity = 0.0;
    let glow_intensity = 0.0;
    let prominence_extent = 0.0;

    // Update sun parameters
    let params = SunParams {
        light_dir_x: light_dir[0],
        light_dir_y: light_dir[1],
        light_dir_z: light_dir[2],
        sun_angular_radius: self.sun_angular_radius,
        sun_color_r: self.sun_color[0],
        sun_color_g: self.sun_color[1],
        sun_color_b: self.sun_color[2],
        sun_intensity,
        corona_radius,
        corona_intensity,
        corona_falloff,
        flare_intensity,
        flare_ghost_count,
        flare_ghost_dispersal,
        flare_halo_radius,
        ray_intensity,
        ray_count,
        ray_falloff,
        eclipse_factor: 1.0, // Shader computes eclipse from depth
        screen_width: self.width as f32,
        screen_height: self.height as f32,
        flare_speed,
        sunspot_scale,
        starburst_intensity,
        starburst_points,
        starburst_falloff,
        streak_intensity,
        streak_width,
        ghost_size,
        chromatic_aberration,
        prominence_intensity,
        glow_intensity,
        prominence_extent,
    };
        queue.write_buffer(&self.sun_params_buffer, 0, bytemuck::bytes_of(&params));

        // Create camera bind group
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sun Camera Bind Group"),
            layout: &self.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_buffer.as_entire_binding(),
            }],
        });

        // Create sun data bind group
        let sun_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sun Data Bind Group"),
            layout: &self.sun_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sun_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        });

        // Render full-screen sun pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sun Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve existing scene
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
            pass.set_bind_group(1, &sun_data_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
