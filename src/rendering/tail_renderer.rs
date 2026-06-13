//! Flagellocyte Tail Renderer - Instanced 3D Helical Tube Geometry
//!
//! Renders flagellocyte tails as instanced 3D helical tubes that rotate
//! with the cell's orientation quaternion. Uses the same instance buffer
//! as cell rendering - tail parameters are stored in type_data for flagellocytes.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// Vertex data for tail mesh
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TailVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub t: f32, // Parameter along helix (0-1)
    pub _pad: f32,
}

/// Per-instance data for tail rendering (used by PreviewScene)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TailInstance {
    pub cell_position: [f32; 3],
    pub cell_radius: f32,
    pub rotation: [f32; 4], // Quaternion
    pub color: [f32; 4],
    pub tail_length: f32,
    pub tail_thickness: f32,
    pub tail_amplitude: f32,
    pub tail_frequency: f32,
    pub tail_speed: f32,
    pub tail_taper: f32,
    pub time: f32,
    pub _pad: f32,
}

/// Vertex data for Plumocyte feather ribbon strokes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PlumageVertex {
    /// x = feather index, y = line kind (0 arm, 1 barb), z = t along stroke, w = ribbon side
    pub data0: [f32; 4],
    /// x = barb root t, y = barb side, zw = unused
    pub data1: [f32; 4],
}

/// Per-instance data for Plumocyte feather rendering (used by PreviewScene).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PlumageInstance {
    pub cell_position: [f32; 3],
    pub cell_radius: f32,
    pub rotation: [f32; 4],
    pub color: [f32; 4],
    pub feather_length: f32,
    pub feather_width: f32,
    pub feather_brightness: f32,
    pub stroke_speed: f32,
    pub frozen: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SiphonApertureVertex {
    /// x/y = unit circle direction, z = radial band 0..1, w = unused
    pub data: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SiphonApertureInstance {
    pub cell_position: [f32; 3],
    pub cell_radius: f32,
    pub rotation: [f32; 4],
    pub color: [f32; 4],
    pub aperture_radius: f32,
    pub aperture_darkness: f32,
    pub rim_brightness: f32,
    pub nozzle_height: f32,
    pub activity: f32,
    pub embed_depth: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SiphonJetParticle {
    position: [f32; 3],
    size: f32,
    velocity: [f32; 3],
    age: f32,
    max_age: f32,
    style: f32,
    seed: f32,
    _pad: f32,
}

const _: () = assert!(std::mem::size_of::<SiphonJetParticle>() == 48);

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SiphonJetParams {
    delta_time: f32,
    current_time: f32,
    current_frame: u32,
    max_particles: u32,
    cell_capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: [f32; 4],
    _pad4: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SiphonJetParams>() == 64);

/// Camera uniform for tail shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TailCameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    partition_offset: u32,
    gravity_mode: u32,
    gravity: f32,
    _padding: f32,
}

/// Lighting uniform for tail shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TailLightingUniform {
    light_dir: [f32; 3],
    ambient: f32,
    light_color: [f32; 3],
    _padding: f32,
}

/// LOD level info for tail mesh
#[derive(Clone, Copy)]
struct TailLodInfo {
    #[allow(dead_code)]
    vertex_offset: u32,
    index_offset: u32,
    index_count: u32,
}

pub struct TailRenderer {
    // Pipeline for CPU-built instances (PreviewScene)
    pipeline: wgpu::RenderPipeline,
    // Pipeline for GPU instance buffer (GpuScene) - reads from CellInstance buffer
    gpu_pipeline: wgpu::RenderPipeline,
    plumage_pipeline: Option<wgpu::RenderPipeline>,
    plumage_gpu_pipeline: Option<wgpu::RenderPipeline>,
    siphon_pipeline: Option<wgpu::RenderPipeline>,
    siphon_gpu_pipeline: Option<wgpu::RenderPipeline>,
    siphon_jet_gpu_pipeline: Option<wgpu::RenderPipeline>,
    siphon_jet_spawn_pipeline: Option<wgpu::ComputePipeline>,
    siphon_jet_age_pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    gpu_bind_group_layout: wgpu::BindGroupLayout,
    plumage_gpu_bind_group_layout: Option<wgpu::BindGroupLayout>,
    siphon_gpu_bind_group_layout: Option<wgpu::BindGroupLayout>,
    siphon_jet_compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    plumage_vertex_buffer: wgpu::Buffer,
    plumage_index_buffer: wgpu::Buffer,
    siphon_vertex_buffer: wgpu::Buffer,
    siphon_index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    plumage_instance_buffer: wgpu::Buffer,
    siphon_instance_buffer: wgpu::Buffer,
    siphon_jet_particle_buffer: wgpu::Buffer,
    siphon_jet_counter_buffer: wgpu::Buffer,
    siphon_jet_params_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    light_color: [f32; 3],
    bind_group: wgpu::BindGroup,
    // Indexed indirect buffer for GPU rendering (persistent, updated via copy)
    indexed_indirect_buffer: wgpu::Buffer,
    plumage_indexed_indirect_buffer: wgpu::Buffer,
    siphon_indexed_indirect_buffer: wgpu::Buffer,
    // LOD info for each level
    lod_info: [TailLodInfo; 4],
    // Default LOD index count (for backward compatibility)
    #[allow(dead_code)]
    index_count: u32,
    plumage_index_count: u32,
    siphon_index_count: u32,
    instance_capacity: usize,
    plumage_instance_capacity: usize,
    siphon_instance_capacity: usize,
    siphon_jet_frame: u32,
    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
}

impl TailRenderer {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    const MAX_SIPHON_JET_PARTICLES: u32 = 32_768;
    const SIPHON_JET_EMISSION_LANES: u32 = 4;

    // LOD mesh parameters: (segments_along_length, radial_segments)
    // LOD 0 (far): 4 segments, 3 radial = minimal detail
    // LOD 1 (medium): 8 segments, 4 radial = balanced
    // LOD 2 (close): 16 segments, 6 radial = high detail
    // LOD 3 (very close): 24 segments, 8 radial = maximum detail
    const LOD_PARAMS: [(u32, u32); 4] = [
        (4, 3),  // LOD 0: 4*3*2 = 24 triangles
        (8, 4),  // LOD 1: 8*4*2 = 64 triangles
        (16, 6), // LOD 2: 16*6*2 = 192 triangles
        (24, 8), // LOD 3: 24*8*2 = 384 triangles
    ];

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        capacity: usize,
    ) -> Self {
        // Generate all LOD meshes and combine into single buffers
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();
        let mut lod_info = [TailLodInfo {
            vertex_offset: 0,
            index_offset: 0,
            index_count: 0,
        }; 4];

        for (lod, &(segments, radial_segments)) in Self::LOD_PARAMS.iter().enumerate() {
            let vertex_offset = all_vertices.len() as u32;
            let index_offset = all_indices.len() as u32;

            let (vertices, indices) = Self::generate_helix_mesh(segments, radial_segments);

            // Offset indices to account for vertex offset
            let offset_indices: Vec<u32> = indices.iter().map(|i| i + vertex_offset).collect();

            lod_info[lod] = TailLodInfo {
                vertex_offset,
                index_offset,
                index_count: indices.len() as u32,
            };

            all_vertices.extend(vertices);
            all_indices.extend(offset_indices);
        }

        // Use LOD 2 as default for backward compatibility
        let index_count = lod_info[2].index_count;
        let (plumage_vertices, plumage_indices) = Self::generate_plumage_mesh();
        let plumage_index_count = plumage_indices.len() as u32;
        let (siphon_vertices, siphon_indices) = Self::generate_siphon_aperture_mesh();
        let siphon_index_count = siphon_indices.len() as u32;
        // Create vertex buffer with all LOD meshes
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Vertex Buffer"),
            size: (all_vertices.len() * std::mem::size_of::<TailVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&all_vertices));
        vertex_buffer.unmap();

        // Create index buffer with all LOD meshes
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Index Buffer"),
            size: (all_indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&all_indices));
        index_buffer.unmap();

        let plumage_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Plumocyte Feather Vertex Buffer"),
            size: (plumage_vertices.len() * std::mem::size_of::<PlumageVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        plumage_vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&plumage_vertices));
        plumage_vertex_buffer.unmap();

        let plumage_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Plumocyte Feather Index Buffer"),
            size: (plumage_indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        plumage_index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&plumage_indices));
        plumage_index_buffer.unmap();

        let siphon_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Siphonocyte Aperture Vertex Buffer"),
            size: (siphon_vertices.len() * std::mem::size_of::<SiphonApertureVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        siphon_vertex_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&siphon_vertices));
        siphon_vertex_buffer.unmap();

        let siphon_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Siphonocyte Aperture Index Buffer"),
            size: (siphon_indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        siphon_index_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&siphon_indices));
        siphon_index_buffer.unmap();

        // Create instance buffer for CPU-built instances
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Instance Buffer"),
            size: (capacity * std::mem::size_of::<TailInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plumage_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Plumocyte Feather Instance Buffer"),
            size: (capacity * std::mem::size_of::<PlumageInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let siphon_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Siphonocyte Aperture Instance Buffer"),
            size: (capacity * std::mem::size_of::<SiphonApertureInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let siphon_jet_particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Siphonocyte Jet Particle Buffer"),
            size: Self::MAX_SIPHON_JET_PARTICLES as u64
                * std::mem::size_of::<SiphonJetParticle>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let siphon_jet_counter_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Siphonocyte Jet Ring Counter"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let siphon_jet_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Siphonocyte Jet Params"),
            size: std::mem::size_of::<SiphonJetParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffers
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Camera Buffer"),
            size: std::mem::size_of::<TailCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Lighting Buffer"),
            size: std::mem::size_of::<TailLightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout for CPU instances
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tail Bind Group Layout"),
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

        // Create bind group layout for GPU instances (reads from cell instance buffer)
        let gpu_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Tail GPU Bind Group Layout"),
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
                    // Lighting uniform
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
                    // Cell instance buffer (storage, read-only)
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
                ],
            });

        // Create bind group for CPU instances
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tail Bind Group"),
            layout: &bind_group_layout,
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

        // Create shader for CPU instances
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tail Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cells/tail.wgsl").into()),
        });

        // Create shader for GPU instances
        let gpu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tail GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cells/tail_gpu.wgsl").into(),
            ),
        });

        // Create pipeline layout for CPU instances
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tail Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipeline layout for GPU instances
        let gpu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tail GPU Pipeline Layout"),
            bind_group_layouts: &[&gpu_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline for CPU instances
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tail Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Vertex buffer
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<TailVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                    // Instance buffer
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<TailInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // cell_position + cell_radius
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
                            },
                            // rotation
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            // color
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 6,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            // tail params (length, thickness, amplitude, frequency)
                            wgpu::VertexAttribute {
                                offset: 48,
                                shader_location: 7,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                            // tail params 2 (speed, taper, time, pad)
                            wgpu::VertexAttribute {
                                offset: 64,
                                shader_location: 8,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                        ],
                    },
                ],
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
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create render pipeline for GPU instances (vertex-only buffer, reads instances from storage)
        let gpu_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tail GPU Render Pipeline"),
            layout: Some(&gpu_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gpu_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Vertex buffer only - instances read from storage buffer
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<TailVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gpu_shader,
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
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create indexed indirect buffer for GPU rendering
        // Format: [index_count, instance_count, first_index, base_vertex, first_instance]
        let indexed_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Indexed Indirect Buffer"),
            size: 20, // 5 u32s
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plumage_indexed_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Plumocyte Feather Indexed Indirect Buffer"),
            size: 20,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let siphon_indexed_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Siphonocyte Aperture Indexed Indirect Buffer"),
            size: 20,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            gpu_pipeline,
            plumage_pipeline: None,
            plumage_gpu_pipeline: None,
            siphon_pipeline: None,
            siphon_gpu_pipeline: None,
            siphon_jet_gpu_pipeline: None,
            siphon_jet_spawn_pipeline: None,
            siphon_jet_age_pipeline: None,
            bind_group_layout,
            gpu_bind_group_layout,
            plumage_gpu_bind_group_layout: None,
            siphon_gpu_bind_group_layout: None,
            siphon_jet_compute_bind_group_layout: None,
            vertex_buffer,
            index_buffer,
            plumage_vertex_buffer,
            plumage_index_buffer,
            siphon_vertex_buffer,
            siphon_index_buffer,
            instance_buffer,
            plumage_instance_buffer,
            siphon_instance_buffer,
            siphon_jet_particle_buffer,
            siphon_jet_counter_buffer,
            siphon_jet_params_buffer,
            camera_buffer,
            lighting_buffer,
            light_color: [1.0, 0.98, 0.95], // Default warm white
            bind_group,
            indexed_indirect_buffer,
            plumage_indexed_indirect_buffer,
            siphon_indexed_indirect_buffer,
            lod_info,
            index_count,
            plumage_index_count,
            siphon_index_count,
            instance_capacity: capacity,
            plumage_instance_capacity: capacity,
            siphon_instance_capacity: capacity,
            siphon_jet_frame: 0,
            surface_format,
            width: 800,
            height: 600,
        }
    }

    /// Generate a unit helix tube mesh (will be transformed per-instance)
    fn generate_helix_mesh(segments: u32, radial_segments: u32) -> (Vec<TailVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Generate vertices along helix
        for i in 0..=segments {
            let t = i as f32 / segments as f32;

            // Unit helix: extends along +Z, with radius 1 at base
            // Actual position/scale applied in shader based on instance params
            let z = t; // 0 to 1 along length

            // Generate ring of vertices around tube at this point
            for j in 0..radial_segments {
                let angle = (j as f32 / radial_segments as f32) * std::f32::consts::TAU;
                let (sin_a, cos_a) = angle.sin_cos();

                // Local position on unit circle (will be scaled by thickness in shader)
                let local_x = cos_a;
                let local_y = sin_a;

                vertices.push(TailVertex {
                    position: [local_x, local_y, z],
                    normal: [cos_a, sin_a, 0.0],
                    t,
                    _pad: 0.0,
                });
            }
        }

        // Generate indices for tube triangles
        for i in 0..segments {
            for j in 0..radial_segments {
                let current = i * radial_segments + j;
                let next = i * radial_segments + (j + 1) % radial_segments;
                let current_next_ring = (i + 1) * radial_segments + j;
                let next_next_ring = (i + 1) * radial_segments + (j + 1) % radial_segments;

                // Two triangles per quad - CCW winding for outward-facing normals
                indices.push(current);
                indices.push(next);
                indices.push(current_next_ring);

                indices.push(next);
                indices.push(next_next_ring);
                indices.push(current_next_ring);
            }
        }

        (vertices, indices)
    }

    fn push_ribbon_line(
        vertices: &mut Vec<PlumageVertex>,
        indices: &mut Vec<u32>,
        feather_index: u32,
        line_kind: f32,
        root_t: f32,
        barb_side: f32,
        segments: u32,
    ) {
        let base = vertices.len() as u32;
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            for &side in &[-1.0f32, 1.0] {
                vertices.push(PlumageVertex {
                    data0: [feather_index as f32, line_kind, t, side],
                    data1: [root_t, barb_side, 0.0, 0.0],
                });
            }
        }

        for i in 0..segments {
            let a = base + i * 2;
            let b = a + 1;
            let c = a + 2;
            let d = a + 3;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    fn generate_plumage_mesh() -> (Vec<PlumageVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for feather in 0..8 {
            Self::push_ribbon_line(&mut vertices, &mut indices, feather, 0.0, 0.0, 0.0, 10);

            for barb in 0..5 {
                let root_t = 0.28 + barb as f32 * 0.12;
                Self::push_ribbon_line(&mut vertices, &mut indices, feather, 1.0, root_t, -1.0, 3);
                Self::push_ribbon_line(&mut vertices, &mut indices, feather, 1.0, root_t, 1.0, 3);
            }
        }
        (vertices, indices)
    }

    fn generate_siphon_aperture_mesh() -> (Vec<SiphonApertureVertex>, Vec<u32>) {
        let segments = 32u32;
        let bands = 5u32;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for band in 0..=bands {
            let radial = band as f32 / bands as f32;
            for segment in 0..segments {
                let angle = segment as f32 / segments as f32 * std::f32::consts::TAU;
                vertices.push(SiphonApertureVertex {
                    data: [angle.cos(), angle.sin(), radial, 0.0],
                });
            }
        }

        for band in 0..bands {
            let row = band * segments;
            let next_row = (band + 1) * segments;
            for segment in 0..segments {
                let next = (segment + 1) % segments;
                let a = row + segment;
                let b = row + next;
                let c = next_row + segment;
                let d = next_row + next;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        (vertices, indices)
    }

    /// Update camera uniform
    fn update_camera(
        &self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        partition_offset: u32,
        horizontal_fov_degrees: f32,
        gravity: f32,
        gravity_mode: u32,
    ) {
        let view = Mat4::from_rotation_translation(camera_rotation, camera_pos).inverse();
        let aspect = self.width as f32 / self.height as f32;
        let proj = Mat4::perspective_rh(
            crate::ui::camera::CameraController::vertical_fov_radians_for_horizontal(
                horizontal_fov_degrees,
                aspect,
            ),
            aspect,
            0.1,
            5000.0,
        );
        let view_proj = proj * view;

        let uniform = TailCameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
            partition_offset,
            gravity_mode,
            gravity,
            _padding: 0.0,
        };

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Set the light color.
    pub fn set_light_color(&mut self, color: [f32; 3]) {
        self.light_color = color;
    }

    /// Update lighting uniform
    fn update_lighting(&self, queue: &wgpu::Queue) {
        let uniform = TailLightingUniform {
            light_dir: [-0.5, -0.7, -0.5],
            ambient: 0.15,
            light_color: self.light_color,
            _padding: 0.0,
        };

        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Ensure instance buffer has sufficient capacity
    fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required > self.instance_capacity {
            let new_capacity = required.max(self.instance_capacity * 2);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tail Instance Buffer"),
                size: (new_capacity * std::mem::size_of::<TailInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_capacity;
        }
    }

    fn ensure_plumage_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required > self.plumage_instance_capacity {
            let new_capacity = required.max(self.plumage_instance_capacity * 2);
            self.plumage_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Plumocyte Feather Instance Buffer"),
                size: (new_capacity * std::mem::size_of::<PlumageInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.plumage_instance_capacity = new_capacity;
        }
    }

    fn ensure_plumage_pipelines(&mut self, device: &wgpu::Device) {
        if self.plumage_pipeline.is_some() && self.plumage_gpu_pipeline.is_some() {
            return;
        }

        let plumage_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plumocyte Feather Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cells/plumocyte_feathers.wgsl").into(),
            ),
        });
        let plumage_gpu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plumocyte Feather GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cells/plumocyte_feathers_gpu.wgsl").into(),
            ),
        });

        let plumage_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Plumocyte Feather Pipeline Layout"),
                bind_group_layouts: &[&self.bind_group_layout],
                push_constant_ranges: &[],
            });
        let plumage_gpu_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Plumocyte Feather GPU Bind Group Layout"),
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
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });
        let plumage_gpu_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Plumocyte Feather GPU Pipeline Layout"),
                bind_group_layouts: &[&plumage_gpu_bind_group_layout],
                push_constant_ranges: &[],
            });

        let plumage_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PlumageVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        let plumage_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PlumageInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        self.plumage_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Plumocyte Feather Render Pipeline"),
                layout: Some(&plumage_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &plumage_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[plumage_vertex_layout.clone(), plumage_instance_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &plumage_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));

        self.plumage_gpu_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Plumocyte Feather GPU Render Pipeline"),
                layout: Some(&plumage_gpu_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &plumage_gpu_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[plumage_vertex_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &plumage_gpu_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));
        self.plumage_gpu_bind_group_layout = Some(plumage_gpu_bind_group_layout);
    }

    fn ensure_siphon_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required > self.siphon_instance_capacity {
            let new_capacity = required.max(self.siphon_instance_capacity * 2);
            self.siphon_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Siphonocyte Aperture Instance Buffer"),
                size: (new_capacity * std::mem::size_of::<SiphonApertureInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.siphon_instance_capacity = new_capacity;
        }
    }

    fn ensure_siphon_pipelines(&mut self, device: &wgpu::Device) {
        if self.siphon_pipeline.is_some() && self.siphon_gpu_pipeline.is_some() {
            if self.siphon_jet_gpu_pipeline.is_some()
                && self.siphon_jet_spawn_pipeline.is_some()
                && self.siphon_jet_age_pipeline.is_some()
            {
                return;
            }
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Siphonocyte Aperture Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cells/siphonocyte_aperture.wgsl").into(),
            ),
        });
        let gpu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Siphonocyte Aperture GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cells/siphonocyte_aperture_gpu.wgsl").into(),
            ),
        });
        let jet_gpu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Siphonocyte Jet GPU Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/cells/siphonocyte_jet_gpu.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Siphonocyte Aperture Pipeline Layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            push_constant_ranges: &[],
        });
        let gpu_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Siphonocyte Aperture GPU Bind Group Layout"),
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
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uni = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let jet_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Siphonocyte Jet Compute Bind Group Layout"),
                entries: &[uni(0), ro(1), ro(2), rw(3), rw(4), ro(5)],
            });
        let gpu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Siphonocyte Aperture GPU Pipeline Layout"),
            bind_group_layouts: &[&gpu_bind_group_layout],
            push_constant_ranges: &[],
        });
        let jet_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Siphonocyte Jet Compute Pipeline Layout"),
                bind_group_layouts: &[&jet_compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SiphonApertureVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x4,
            }],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SiphonApertureInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 68,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        };
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: Self::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        self.siphon_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Siphonocyte Aperture Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout.clone(), instance_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive,
                depth_stencil: depth_stencil.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));

        self.siphon_gpu_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Siphonocyte Aperture GPU Render Pipeline"),
                layout: Some(&gpu_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &gpu_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &gpu_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive,
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));
        self.siphon_jet_spawn_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Siphonocyte Jet Spawn Pipeline"),
                layout: Some(&jet_compute_pipeline_layout),
                module: &jet_gpu_shader,
                entry_point: Some("spawn_jets"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        self.siphon_jet_age_pipeline = Some(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Siphonocyte Jet Age Pipeline"),
                layout: Some(&jet_compute_pipeline_layout),
                module: &jet_gpu_shader,
                entry_point: Some("age_jets"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let siphon_jet_particle_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SiphonJetParticle>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 44,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };
        self.siphon_jet_gpu_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Siphonocyte Jet GPU Render Pipeline"),
                layout: Some(&gpu_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &jet_gpu_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[siphon_jet_particle_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &jet_gpu_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive,
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            },
        ));
        self.siphon_gpu_bind_group_layout = Some(gpu_bind_group_layout);
        self.siphon_jet_compute_bind_group_layout = Some(jet_compute_bind_group_layout);
    }

    /// Resize the renderer
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Render tails for flagellocyte cells (CPU-built instances, for PreviewScene)
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        instances: &[TailInstance],
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
    ) {
        if instances.is_empty() {
            return;
        }

        self.width = width;
        self.height = height;

        // Ensure capacity
        self.ensure_capacity(device, instances.len());

        // Upload instance data
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));

        // Update uniforms (partition_offset = 0 for PreviewScene which uses contiguous data)
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            1.0,
            1,
        );
        self.update_lighting(queue);

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tail Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // Select LOD based on average distance to instances
            let avg_distance = if instances.is_empty() {
                100.0
            } else {
                let total_dist: f32 = instances
                    .iter()
                    .map(|inst| {
                        let pos = Vec3::from_array(inst.cell_position);
                        (pos - camera_pos).length()
                    })
                    .sum();
                total_dist / instances.len() as f32
            };

            // LOD thresholds (matching cell LOD system)
            let lod_level = if avg_distance > 50.0 {
                0 // Far - minimal detail
            } else if avg_distance > 25.0 {
                1 // Medium distance
            } else if avg_distance > 10.0 {
                2 // Close
            } else {
                3 // Very close - maximum detail
            };

            let lod = &self.lod_info[lod_level];
            render_pass.draw_indexed(
                lod.index_offset..(lod.index_offset + lod.index_count),
                0,
                0..instances.len() as u32,
            );
        }
    }

    pub fn render_plumage(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        instances: &[PlumageInstance],
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
    ) {
        if instances.is_empty() {
            return;
        }

        self.width = width;
        self.height = height;
        self.ensure_plumage_pipelines(device);
        self.ensure_plumage_capacity(device, instances.len());
        queue.write_buffer(
            &self.plumage_instance_buffer,
            0,
            bytemuck::cast_slice(instances),
        );
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            1.0,
            1,
        );
        self.update_lighting(queue);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Plumocyte Feather Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        render_pass.set_pipeline(self.plumage_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.plumage_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.plumage_instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.plumage_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.plumage_index_count, 0, 0..instances.len() as u32);
    }

    pub fn render_siphon_apertures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        instances: &[SiphonApertureInstance],
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
    ) {
        if instances.is_empty() {
            return;
        }
        self.width = width;
        self.height = height;
        self.ensure_siphon_pipelines(device);
        self.ensure_siphon_capacity(device, instances.len());
        queue.write_buffer(
            &self.siphon_instance_buffer,
            0,
            bytemuck::cast_slice(instances),
        );
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            1.0,
            1,
        );
        self.update_lighting(queue);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Siphonocyte Aperture Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        render_pass.set_pipeline(self.siphon_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.siphon_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.siphon_instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.siphon_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.siphon_index_count, 0, 0..instances.len() as u32);
    }

    /// Render tails using GPU instance buffer (for GpuScene)
    /// Reads cell instances directly from the instance builder's partitioned buffer.
    /// Uses first_instance to offset into the Flagellocyte partition.
    /// The shader also filters by cell_type as a safety check.
    pub fn render_from_gpu_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        cell_instance_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
        cell_capacity: usize,
    ) {
        // Update dimensions for correct aspect ratio
        self.width = width;
        self.height = height;

        // With dynamic instance allocation, Flagellocytes are mixed throughout the buffer
        // The shader will filter by cell_type (stored in instance.type_data_1.w)
        // No partition offset needed - shader iterates through all cells
        let _cell_capacity = cell_capacity; // Keep for potential future use

        // Update uniforms (no partition offset with dynamic allocation)
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            1.0,
            1,
        );
        self.update_lighting(queue);

        // Set up indexed indirect buffer using LOD 2 (medium-high detail)
        // For GPU rendering, we use a fixed LOD since all instances share the same mesh
        // LOD 2 provides good quality at reasonable cost for most viewing distances
        let lod = &self.lod_info[2];

        // Source format (draw_indirect): [vertex_count, instance_count, first_vertex, first_instance]
        // Target format (draw_indexed_indirect): [index_count, instance_count, first_index, base_vertex, first_instance]
        // first_instance is 0 - shader iterates all instances and filters by cell_type
        let indexed_indirect_data: [u32; 5] = [lod.index_count, 0, lod.index_offset, 0, 0];
        queue.write_buffer(
            &self.indexed_indirect_buffer,
            0,
            bytemuck::cast_slice(&indexed_indirect_data),
        );

        // Copy instance_count from indirect buffer (offset 4) to indexed indirect buffer (offset 4)
        encoder.copy_buffer_to_buffer(
            indirect_buffer,
            4, // instance_count in source
            &self.indexed_indirect_buffer,
            4, // instance_count in dest
            4, // 4 bytes
        );

        // Note: first_instance is now set to the partition offset above, no need to copy from indirect buffer

        // Create bind group with cell instance buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tail GPU Bind Group"),
            layout: &self.gpu_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_instance_buffer.as_entire_binding(),
                },
            ],
        });

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tail GPU Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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

            render_pass.set_pipeline(&self.gpu_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed_indirect(&self.indexed_indirect_buffer, 0);
        }
    }

    pub fn render_plumage_from_gpu_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        cell_instance_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
        gravity: f32,
        gravity_mode: u32,
    ) {
        self.width = width;
        self.height = height;
        self.ensure_plumage_pipelines(device);
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            gravity,
            gravity_mode,
        );
        self.update_lighting(queue);

        let indexed_indirect_data: [u32; 5] = [self.plumage_index_count, 0, 0, 0, 0];
        queue.write_buffer(
            &self.plumage_indexed_indirect_buffer,
            0,
            bytemuck::cast_slice(&indexed_indirect_data),
        );
        encoder.copy_buffer_to_buffer(
            indirect_buffer,
            4,
            &self.plumage_indexed_indirect_buffer,
            4,
            4,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Plumocyte Feather GPU Bind Group"),
            layout: self.plumage_gpu_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_instance_buffer.as_entire_binding(),
                },
            ],
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Plumocyte Feather GPU Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        render_pass.set_pipeline(self.plumage_gpu_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.plumage_vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.plumage_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed_indirect(&self.plumage_indexed_indirect_buffer, 0);
    }

    pub fn render_siphon_apertures_from_gpu_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        cell_instance_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
    ) {
        self.width = width;
        self.height = height;
        self.ensure_siphon_pipelines(device);
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            1.0,
            1,
        );
        self.update_lighting(queue);

        let indexed_indirect_data: [u32; 5] = [self.siphon_index_count, 0, 0, 0, 0];
        queue.write_buffer(
            &self.siphon_indexed_indirect_buffer,
            0,
            bytemuck::cast_slice(&indexed_indirect_data),
        );
        encoder.copy_buffer_to_buffer(
            indirect_buffer,
            4,
            &self.siphon_indexed_indirect_buffer,
            4,
            4,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Siphonocyte Aperture GPU Bind Group"),
            layout: self.siphon_gpu_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_instance_buffer.as_entire_binding(),
                },
            ],
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Siphonocyte Aperture GPU Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        render_pass.set_pipeline(self.siphon_gpu_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.siphon_vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.siphon_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed_indirect(&self.siphon_indexed_indirect_buffer, 0);
    }

    pub fn render_siphon_jets_from_gpu_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        cell_instance_buffer: &wgpu::Buffer,
        velocity_buffer: &wgpu::Buffer,
        counters_buffer: &wgpu::Buffer,
        _indirect_buffer: &wgpu::Buffer,
        cell_capacity: usize,
        camera_pos: Vec3,
        camera_rotation: Quat,
        time: f32,
        delta_time: f32,
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
    ) {
        self.width = width;
        self.height = height;
        self.ensure_siphon_pipelines(device);
        self.update_camera(
            queue,
            camera_pos,
            camera_rotation,
            time,
            0,
            horizontal_fov_degrees,
            1.0,
            1,
        );
        self.update_lighting(queue);
        self.siphon_jet_frame = self.siphon_jet_frame.wrapping_add(1);

        let params = SiphonJetParams {
            delta_time: delta_time.clamp(0.0, 0.1),
            current_time: time,
            current_frame: self.siphon_jet_frame,
            max_particles: Self::MAX_SIPHON_JET_PARTICLES,
            cell_capacity: cell_capacity as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: [0.0; 4],
            _pad4: [0.0; 4],
        };
        queue.write_buffer(
            &self.siphon_jet_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Siphonocyte Jet Compute Bind Group"),
            layout: self.siphon_jet_compute_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.siphon_jet_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.siphon_jet_particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.siphon_jet_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Siphonocyte Jet Spawn"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.siphon_jet_spawn_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &compute_bind_group, &[]);
            let spawn_invocations = cell_capacity as u32 * Self::SIPHON_JET_EMISSION_LANES;
            pass.dispatch_workgroups((spawn_invocations + 63) / 64, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Siphonocyte Jet Age"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.siphon_jet_age_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &compute_bind_group, &[]);
            pass.dispatch_workgroups((Self::MAX_SIPHON_JET_PARTICLES + 255) / 256, 1, 1);
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Siphonocyte Jet GPU Bind Group"),
            layout: self.siphon_gpu_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: velocity_buffer.as_entire_binding(),
                },
            ],
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Siphonocyte Jet GPU Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        render_pass.set_pipeline(self.siphon_jet_gpu_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.siphon_jet_particle_buffer.slice(..));
        render_pass.draw(0..6, 0..Self::MAX_SIPHON_JET_PARTICLES);
    }
}
