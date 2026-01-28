//! Steam ray marching renderer
//!
//! Renders steam (fluid type 3) as a volumetric effect using ray marching.
//! This provides a more realistic, cloudy appearance compared to surface mesh rendering.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

/// Camera uniform for steam rendering (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Steam rendering parameters (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SteamParams {
    pub grid_resolution: u32,
    pub world_radius: f32,
    pub cell_size: f32,
    pub _pad0: u32,

    pub grid_origin: [f32; 3],
    pub max_steps: u32,

    // Steam appearance
    pub steam_color: [f32; 3],
    pub density_multiplier: f32,

    // Light direction
    pub light_dir: [f32; 3],
    pub light_intensity: f32,

    // Scattering params
    pub absorption: f32,
    pub scattering: f32,
    pub phase_g: f32,
    pub _pad1: f32,
}

impl Default for SteamParams {
    fn default() -> Self {
        Self {
            grid_resolution: 128,
            world_radius: 200.0,
            cell_size: 3.125, // 400 / 128
            _pad0: 0,
            grid_origin: [-200.0, -200.0, -200.0],
            max_steps: 48,  // Reduced for performance
            steam_color: [0.95, 0.95, 1.0],
            density_multiplier: 3.0,
            light_dir: [0.5, 1.0, 0.3],
            light_intensity: 1.2,
            absorption: 0.05,
            scattering: 1.0,
            phase_g: 0.2, // Slight forward scattering
            _pad1: 0.0,
        }
    }
}

/// Steam ray marching renderer
pub struct SteamRenderer {
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    params: SteamParams,
    width: u32,
    height: u32,
}

impl SteamRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        world_radius: f32,
        world_center: Vec3,
        width: u32,
        height: u32,
    ) -> Self {
        // Calculate grid parameters
        let grid_resolution = 128u32;
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / grid_resolution as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);

        let mut params = SteamParams::default();
        params.grid_resolution = grid_resolution;
        params.world_radius = world_radius;
        params.cell_size = cell_size;
        params.grid_origin = grid_origin.to_array();

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Steam Ray Marching Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/steam_raymarching.wgsl").into(),
            ),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Steam Renderer Bind Group Layout"),
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
                // Steam params
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
                // Fluid state buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Steam Renderer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Steam Ray Marching Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // Fullscreen triangle, no vertex buffer
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // Volumetric, don't write depth
                depth_compare: wgpu::CompareFunction::Always, // Always render (we handle occlusion in shader)
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

        // Create camera buffer (will be written to in render)
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Steam Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Steam Params Buffer"),
            size: std::mem::size_of::<SteamParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            render_pipeline,
            bind_group_layout,
            camera_buffer,
            params_buffer,
            params,
            width,
            height,
        }
    }

    /// Update steam parameters
    pub fn set_params(&mut self, queue: &wgpu::Queue, params: SteamParams) {
        self.params = params;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Get current params for UI editing
    pub fn params(&self) -> &SteamParams {
        &self.params
    }

    /// Get mutable params for UI editing
    pub fn params_mut(&mut self) -> &mut SteamParams {
        &mut self.params
    }

    /// Upload params to GPU
    pub fn upload_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Resize for new screen dimensions
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Create bind group with fluid state buffer
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        fluid_state_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Steam Renderer Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fluid_state_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Render steam volumetrics
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        bind_group: &wgpu::BindGroup,
        camera_pos: Vec3,
        camera_rotation: Quat,
    ) {
        // Update camera uniform
        let view = Mat4::look_at_rh(
            camera_pos,
            camera_pos + camera_rotation * Vec3::NEG_Z,
            camera_rotation * Vec3::Y,
        );
        let aspect = self.width as f32 / self.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Begin render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Steam Ray Marching Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Preserve existing content
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

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Fullscreen triangle
    }
}
