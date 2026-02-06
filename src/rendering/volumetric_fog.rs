//! Volumetric Fog Renderer
//!
//! Full-screen post-process effect that ray marches through the light field
//! to produce volumetric fog, god rays, and atmospheric scattering.
//! Composites over the scene using alpha blending.

use bytemuck::{Pod, Zeroable};

/// Camera uniforms for volumetric fog (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FogCameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub time: f32,
}

/// Fog parameters (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FogParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub world_radius: f32,
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    pub fog_density: f32,
    pub fog_steps: u32,
    pub light_color_r: f32,
    pub light_color_g: f32,
    pub light_color_b: f32,
    pub light_intensity: f32,
    pub fog_color_r: f32,
    pub fog_color_g: f32,
    pub fog_color_b: f32,
    pub scattering_anisotropy: f32,
    pub absorption: f32,
    pub height_fog_density: f32,
    pub height_fog_falloff: f32,
    pub ray_start: f32,
    pub ray_end: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Volumetric fog renderer
pub struct VolumetricFogRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
    fog_data_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    fog_params_buffer: wgpu::Buffer,
    depth_sampler: wgpu::Sampler,

    // Configurable parameters
    pub fog_density: f32,
    pub fog_steps: u32,
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub fog_color: [f32; 3],
    pub scattering_anisotropy: f32,
    pub absorption: f32,
    pub height_fog_density: f32,
    pub height_fog_falloff: f32,
    pub ray_start: f32,
    pub ray_end: f32,
    pub enabled: bool,
}

impl VolumetricFogRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // Camera bind group layout (group 0)
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fog Camera Layout"),
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

        // Fog data bind group layout (group 1)
        let fog_data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fog Data Layout"),
            entries: &[
                // Binding 0: FogParams uniform
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
                // Binding 1: light_field storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: solid_mask storage
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
                // Binding 3: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volumetric Fog Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/volumetric_fog.wgsl").into(),
            ),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volumetric Fog Pipeline Layout"),
            bind_group_layouts: &[&camera_layout, &fog_data_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline (full-screen triangle, alpha blending)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volumetric Fog Pipeline"),
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
            label: Some("Fog Camera Buffer"),
            size: std::mem::size_of::<FogCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create fog params buffer
        let fog_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fog Params Buffer"),
            size: std::mem::size_of::<FogParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create depth sampler (non-filtering for depth texture)
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fog Depth Sampler"),
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
            fog_data_layout,
            camera_buffer,
            fog_params_buffer,
            depth_sampler,
            fog_density: 0.5,
            fog_steps: 48,
            light_color: [1.0, 0.95, 0.85],
            light_intensity: 2.0,
            fog_color: [0.4, 0.5, 0.6],
            scattering_anisotropy: 0.6,
            absorption: 0.3,
            height_fog_density: 0.0,
            height_fog_falloff: 0.01,
            ray_start: 1.0,
            ray_end: 500.0,
            enabled: false,
        }
    }

    /// Create fog data bind group
    pub fn create_fog_data_bind_group(
        &self,
        device: &wgpu::Device,
        light_field_buffer: &wgpu::Buffer,
        solid_mask_buffer: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fog Data Bind Group"),
            layout: &self.fog_data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.fog_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_field_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        })
    }

    /// Update camera and fog parameters, then render
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        device: &wgpu::Device,
        light_field_buffer: &wgpu::Buffer,
        solid_mask_buffer: &wgpu::Buffer,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        time: f32,
        light_dir: [f32; 3],
        grid_resolution: u32,
        cell_size: f32,
        grid_origin: [f32; 3],
        world_radius: f32,
    ) {
        // Update camera uniforms
        let inv_view_proj = view_proj.inverse();
        let camera_uniform = FogCameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update fog parameters
        let fog_params = FogParams {
            grid_resolution,
            cell_size,
            grid_origin_x: grid_origin[0],
            grid_origin_y: grid_origin[1],
            grid_origin_z: grid_origin[2],
            world_radius,
            light_dir_x: light_dir[0],
            light_dir_y: light_dir[1],
            light_dir_z: light_dir[2],
            fog_density: self.fog_density,
            fog_steps: self.fog_steps,
            light_color_r: self.light_color[0],
            light_color_g: self.light_color[1],
            light_color_b: self.light_color[2],
            light_intensity: self.light_intensity,
            fog_color_r: self.fog_color[0],
            fog_color_g: self.fog_color[1],
            fog_color_b: self.fog_color[2],
            scattering_anisotropy: self.scattering_anisotropy,
            absorption: self.absorption,
            height_fog_density: self.height_fog_density,
            height_fog_falloff: self.height_fog_falloff,
            ray_start: self.ray_start,
            ray_end: self.ray_end,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.fog_params_buffer, 0, bytemuck::bytes_of(&fog_params));

        // Create camera bind group
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fog Camera Bind Group"),
            layout: &self.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_buffer.as_entire_binding(),
            }],
        });

        // Create fog data bind group
        let fog_data_bg =
            self.create_fog_data_bind_group(device, light_field_buffer, solid_mask_buffer, depth_view);

        // Render full-screen fog pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Volumetric Fog Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
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
            pass.set_bind_group(1, &fog_data_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
