//! Flagellocyte Tail Renderer - Instanced 3D Helical Tube Geometry
//!
//! Renders flagellocyte tails as instanced 3D helical tubes that rotate
//! with the cell's orientation quaternion. Uses the same instance buffer
//! as cell rendering - tail parameters are stored in type_data for flagellocytes.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

/// Vertex data for tail mesh
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TailVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub t: f32,  // Parameter along helix (0-1)
    pub _pad: f32,
}

/// Per-instance data for tail rendering (used by PreviewScene)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TailInstance {
    pub cell_position: [f32; 3],
    pub cell_radius: f32,
    pub rotation: [f32; 4],  // Quaternion
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

/// Camera uniform for tail shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TailCameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    partition_offset: u32,
    _padding: [u32; 3],
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
    gpu_bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    // Indexed indirect buffer for GPU rendering (persistent, updated via copy)
    indexed_indirect_buffer: wgpu::Buffer,
    // LOD info for each level
    lod_info: [TailLodInfo; 4],
    // Default LOD index count (for backward compatibility)
    #[allow(dead_code)]
    index_count: u32,
    instance_capacity: usize,
    width: u32,
    height: u32,
}

impl TailRenderer {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    
    // LOD mesh parameters: (segments_along_length, radial_segments)
    // LOD 0 (far): 4 segments, 3 radial = minimal detail
    // LOD 1 (medium): 8 segments, 4 radial = balanced
    // LOD 2 (close): 16 segments, 6 radial = high detail
    // LOD 3 (very close): 24 segments, 8 radial = maximum detail
    const LOD_PARAMS: [(u32, u32); 4] = [
        (4, 3),   // LOD 0: 4*3*2 = 24 triangles
        (8, 4),   // LOD 1: 8*4*2 = 64 triangles
        (16, 6),  // LOD 2: 16*6*2 = 192 triangles
        (24, 8),  // LOD 3: 24*8*2 = 384 triangles
    ];
    
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        capacity: usize,
    ) -> Self {
        // Generate all LOD meshes and combine into single buffers
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();
        let mut lod_info = [TailLodInfo { vertex_offset: 0, index_offset: 0, index_count: 0 }; 4];
        
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
        
        // Create vertex buffer with all LOD meshes
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Vertex Buffer"),
            size: (all_vertices.len() * std::mem::size_of::<TailVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&all_vertices));
        vertex_buffer.unmap();
        
        // Create index buffer with all LOD meshes
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Index Buffer"),
            size: (all_indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&all_indices));
        index_buffer.unmap();
        
        // Create instance buffer for CPU-built instances
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tail Instance Buffer"),
            size: (capacity * std::mem::size_of::<TailInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
        
        // Create bind group layout for GPU instances (reads from cell instance buffer)
        let gpu_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cells/tail_gpu.wgsl").into()),
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
        
        Self {
            pipeline,
            gpu_pipeline,
            gpu_bind_group_layout,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            camera_buffer,
            lighting_buffer,
            bind_group,
            indexed_indirect_buffer,
            lod_info,
            index_count,
            instance_capacity: capacity,
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
            let z = t;  // 0 to 1 along length
            
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
    
    /// Update camera uniform
    fn update_camera(&self, queue: &wgpu::Queue, camera_pos: Vec3, camera_rotation: Quat, time: f32, partition_offset: u32) {
        let view = Mat4::from_rotation_translation(camera_rotation, camera_pos).inverse();
        let aspect = self.width as f32 / self.height as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
        let view_proj = proj * view;

        let uniform = TailCameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
            partition_offset,
            _padding: [0; 3],
        };

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }
    
    /// Update lighting uniform
    fn update_lighting(&self, queue: &wgpu::Queue) {
        let uniform = TailLightingUniform {
            light_dir: [-0.5, -0.7, -0.5],
            ambient: 0.15,
            light_color: [1.0, 0.98, 0.95],
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
        self.update_camera(queue, camera_pos, camera_rotation, time, 0);
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
                let total_dist: f32 = instances.iter()
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
                0..instances.len() as u32
            );
        }
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
        width: u32,
        height: u32,
        cell_capacity: usize,
    ) {
        // Update dimensions for correct aspect ratio
        self.width = width;
        self.height = height;
        
        // Calculate partition offset for Flagellocytes (cell_type = 1)
        // The instance buffer is partitioned by cell type to support multi-pipeline rendering
        // Partition size = cell_capacity / MAX_TYPES (30)
        use crate::cell::types::CellType;
        let partition_size = cell_capacity / CellType::MAX_TYPES;
        let flagellocyte_offset = (CellType::Flagellocyte as usize) * partition_size;

        log::info!("Tail Renderer: cell_capacity={}, partition_size={}, flagellocyte_offset={}",
                   cell_capacity, partition_size, flagellocyte_offset);

        // Update uniforms (pass partition_offset to camera uniform)
        self.update_camera(queue, camera_pos, camera_rotation, time, flagellocyte_offset as u32);
        self.update_lighting(queue);

        // Set up indexed indirect buffer using LOD 2 (medium-high detail)
        // For GPU rendering, we use a fixed LOD since all instances share the same mesh
        // LOD 2 provides good quality at reasonable cost for most viewing distances
        let lod = &self.lod_info[2];

        // Source format (draw_indirect): [vertex_count, instance_count, first_vertex, first_instance]
        // Target format (draw_indexed_indirect): [index_count, instance_count, first_index, base_vertex, first_instance]
        // first_instance is set to 0 since we handle partition offset in the shader via uniform
        let indexed_indirect_data: [u32; 5] = [lod.index_count, 0, lod.index_offset, 0, 0];
        queue.write_buffer(&self.indexed_indirect_buffer, 0, bytemuck::cast_slice(&indexed_indirect_data));

        log::info!("Tail Renderer: index_count={}, first_index={}, partition_offset passed to shader={}",
                   lod.index_count, lod.index_offset, flagellocyte_offset);

        // Copy instance_count from indirect buffer (offset 4) to indexed indirect buffer (offset 4)
        encoder.copy_buffer_to_buffer(
            indirect_buffer,
            4,  // instance_count in source
            &self.indexed_indirect_buffer,
            4,  // instance_count in dest
            4,  // 4 bytes
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
}
