//! World Sphere Renderer
//!
//! Renders a transparent boundary sphere with proper PBR lighting.
//! The sphere is rendered from the inside using front-face culling, creating
//! a containment boundary visual for the simulation.
//! Uses the same lighting as the cell renderer for visual consistency.

use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// Parameters for the world sphere appearance
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldSphereParams {
    /// Base color (RGBA) - sRGB color space, default: dark blue-gray with 35% opacity
    pub base_color: [f32; 4],
    /// Emissive color (RGB, linear space) + unused
    pub emissive: [f32; 4],
    /// Sphere radius
    pub radius: f32,
    /// Metallic factor (0.0 for non-metallic)
    pub metallic: f32,
    /// Perceptual roughness (0.2 for slightly glossy)
    pub perceptual_roughness: f32,
    /// Reflectance factor (0.95 for high reflectance)
    pub reflectance: f32,
    /// Padding for 16-byte alignment
    pub _padding: [f32; 4],
}

impl Default for WorldSphereParams {
    fn default() -> Self {
        Self {
            // Match reference: Color::srgba(0.2, 0.25, 0.35, 0.35)
            base_color: [0.2, 0.25, 0.35, 0.35],
            // Match reference: LinearRgba::rgb(0.05, 0.08, 0.12)
            emissive: [0.05, 0.08, 0.12, 0.0],
            radius: 100.0,
            metallic: 0.0,
            perceptual_roughness: 0.2,
            reflectance: 0.95,
            _padding: [0.0; 4],
        }
    }
}

/// Camera uniform structure matching the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Lighting uniform - matches the cell renderer's lighting
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
    /// Direction of primary light source (normalized)
    light_direction: [f32; 3],
    _padding1: f32,
    /// Color of primary light source
    light_color: [f32; 3],
    _padding2: f32,
    /// Ambient light color
    ambient_color: [f32; 3],
    _padding3: f32,
}

/// World sphere renderer for the GPU scene
pub struct WorldSphereRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    params: WorldSphereParams,
    /// Screen dimensions for aspect ratio
    pub width: u32,
    pub height: u32,
}

impl WorldSphereRenderer {
    /// Create a new world sphere renderer
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Sphere Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/world_sphere.wgsl").into()),
        });

        // Create camera bind group layout (camera + lighting)
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("World Sphere Camera Layout"),
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

        // Create params bind group layout
        let params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("World Sphere Params Layout"),
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("World Sphere Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &params_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline with front-face culling (render inside of sphere)
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("World Sphere Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24, // 6 floats * 4 bytes
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
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
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
                // Front-face culling to render inside of sphere
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                // Read depth but don't write (so cells render in front)
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 1, // Push sphere back slightly in depth
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Generate icosphere mesh (subdivision level 5 for smooth appearance)
        let (vertices, indices) = generate_icosphere(5);
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Sphere Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create camera uniform buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("World Sphere Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create lighting uniform buffer
        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("World Sphere Lighting Buffer"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Sphere Camera Bind Group"),
            layout: &camera_bind_group_layout,
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

        // Create params uniform buffer
        let params = WorldSphereParams::default();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Sphere Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Sphere Params Bind Group"),
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        Self {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            camera_buffer,
            lighting_buffer,
            camera_bind_group,
            params_buffer,
            params_bind_group,
            params,
            width: surface_config.width,
            height: surface_config.height,
        }
    }

    /// Update the world sphere parameters
    pub fn set_params(&mut self, queue: &wgpu::Queue, params: WorldSphereParams) {
        self.params = params;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Set the sphere radius
    pub fn set_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.params.radius = radius;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Set the base color (RGBA)
    pub fn set_color(&mut self, queue: &wgpu::Queue, color: Vec4) {
        self.params.base_color = color.to_array();
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Set the opacity (alpha channel)
    pub fn set_opacity(&mut self, queue: &wgpu::Queue, opacity: f32) {
        self.params.base_color[3] = opacity;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Set the emissive color (linear RGB)
    pub fn set_emissive(&mut self, queue: &wgpu::Queue, emissive: Vec3) {
        self.params.emissive = [emissive.x, emissive.y, emissive.z, 0.0];
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Set the perceptual roughness (0.0 = mirror, 1.0 = rough)
    pub fn set_roughness(&mut self, queue: &wgpu::Queue, roughness: f32) {
        self.params.perceptual_roughness = roughness;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Resize the renderer
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Render the world sphere
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
    ) {
        // Calculate view and projection matrices
        let view_matrix = Mat4::look_at_rh(
            camera_pos,
            camera_pos + camera_rotation * Vec3::NEG_Z,
            camera_rotation * Vec3::Y,
        );
        let aspect = self.width as f32 / self.height as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj_matrix * view_matrix;

        // Update camera uniform
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // Update lighting uniform - match the cell renderer's lighting
        let light_dir = Vec3::new(0.5, 1.0, 0.3).normalize();
        let lighting_uniform = LightingUniform {
            light_direction: light_dir.to_array(),
            _padding1: 0.0,
            light_color: [0.8, 0.8, 0.8],
            _padding2: 0.0,
            ambient_color: [0.15, 0.15, 0.2],
            _padding3: 0.0,
        };
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::cast_slice(&[lighting_uniform]));

        // Begin render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("World Sphere Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
                    load: wgpu::LoadOp::Load, // Use existing depth
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.params_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Vertex with position and normal
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Generate an icosphere mesh with the given subdivision level
/// Returns (vertices, indices) where vertices contain position and normal
fn generate_icosphere(subdivisions: u32) -> (Vec<Vertex>, Vec<u32>) {
    // Golden ratio for icosahedron
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let inv_len = 1.0 / (1.0 + phi * phi).sqrt();
    
    // Initial icosahedron vertices (normalized to unit sphere)
    let mut vertices: Vec<Vec3> = vec![
        Vec3::new(-1.0, phi, 0.0) * inv_len,
        Vec3::new(1.0, phi, 0.0) * inv_len,
        Vec3::new(-1.0, -phi, 0.0) * inv_len,
        Vec3::new(1.0, -phi, 0.0) * inv_len,
        Vec3::new(0.0, -1.0, phi) * inv_len,
        Vec3::new(0.0, 1.0, phi) * inv_len,
        Vec3::new(0.0, -1.0, -phi) * inv_len,
        Vec3::new(0.0, 1.0, -phi) * inv_len,
        Vec3::new(phi, 0.0, -1.0) * inv_len,
        Vec3::new(phi, 0.0, 1.0) * inv_len,
        Vec3::new(-phi, 0.0, -1.0) * inv_len,
        Vec3::new(-phi, 0.0, 1.0) * inv_len,
    ];
    
    // Initial icosahedron faces (20 triangles)
    let mut indices: Vec<[u32; 3]> = vec![
        // 5 faces around point 0
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        // 5 adjacent faces
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        // 5 faces around point 3
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        // 5 adjacent faces
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    
    // Subdivide
    use std::collections::HashMap;
    
    for _ in 0..subdivisions {
        let mut new_indices = Vec::with_capacity(indices.len() * 4);
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
        
        for tri in &indices {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];
            
            // Get or create midpoints
            let a = get_midpoint(&mut vertices, &mut midpoint_cache, v0, v1);
            let b = get_midpoint(&mut vertices, &mut midpoint_cache, v1, v2);
            let c = get_midpoint(&mut vertices, &mut midpoint_cache, v2, v0);
            
            // Create 4 new triangles
            new_indices.push([v0, a, c]);
            new_indices.push([v1, b, a]);
            new_indices.push([v2, c, b]);
            new_indices.push([a, b, c]);
        }
        
        indices = new_indices;
    }
    
    // Convert to vertex format with normals (normal = position for unit sphere)
    let output_vertices: Vec<Vertex> = vertices
        .iter()
        .map(|v| Vertex {
            position: v.to_array(),
            normal: v.normalize().to_array(),
        })
        .collect();
    
    // Flatten indices
    let output_indices: Vec<u32> = indices.iter().flat_map(|tri| tri.iter().copied()).collect();
    
    (output_vertices, output_indices)
}

/// Get or create a midpoint vertex between two vertices
fn get_midpoint(
    vertices: &mut Vec<Vec3>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
    v0: u32,
    v1: u32,
) -> u32 {
    // Ensure consistent key ordering
    let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
    
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    
    // Create new midpoint vertex
    let p0 = vertices[v0 as usize];
    let p1 = vertices[v1 as usize];
    let midpoint = ((p0 + p1) / 2.0).normalize(); // Project onto unit sphere
    
    let idx = vertices.len() as u32;
    vertices.push(midpoint);
    cache.insert(key, idx);
    
    idx
}
