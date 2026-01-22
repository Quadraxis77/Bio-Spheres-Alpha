//! Voxel renderer for fluid simulation visualization.
//!
//! Renders voxels using GPU instancing with color-coded types:
//! - Purple: Test fluid voxels
//! - Green: Solid cave voxels
//! - Voxels outside world sphere are not rendered

use bytemuck::{Pod, Zeroable};
use wgpu::{self, util::DeviceExt};

/// Instance data for a single voxel
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VoxelInstance {
    /// World position of voxel center
    pub position: [f32; 3],
    /// Voxel type (0=empty, 1=water, 2=lava, 3=steam, 4=solid)
    pub voxel_type: u32,
    /// Color override (RGBA)
    pub color: [f32; 4],
    /// Voxel size (half-extent)
    pub size: f32,
    /// Padding to 48 bytes
    pub _padding: [f32; 3],
}

/// Voxel renderer for instanced cube rendering
pub struct VoxelRenderer {
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Vertex buffer (cube geometry)
    vertex_buffer: wgpu::Buffer,
    /// Index buffer (cube indices)
    index_buffer: wgpu::Buffer,
    /// Number of indices
    index_count: u32,
    /// Instance buffer
    instance_buffer: wgpu::Buffer,
    /// Maximum instance capacity
    max_instances: u32,
}

impl VoxelRenderer {
    /// Create a new voxel renderer
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        max_instances: u32,
    ) -> Self {
        // Create cube geometry
        let (vertices, indices) = Self::create_cube_geometry();
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        let index_count = indices.len() as u32;
        
        // Create instance buffer
        let instance_buffer_size = (max_instances as u64) * std::mem::size_of::<VoxelInstance>() as u64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Instance Buffer"),
            size: instance_buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fluid/voxel_instance.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Vertex buffer (position)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    },
                    // Instance buffer
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<VoxelInstance>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            1 => Float32x3,  // position
                            2 => Uint32,     // voxel_type
                            3 => Float32x4,  // color
                            4 => Float32,    // size
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
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
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
        
        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            instance_buffer,
            max_instances,
        }
    }
    
    /// Create cube geometry (vertices and indices)
    fn create_cube_geometry() -> (Vec<[f32; 3]>, Vec<u16>) {
        // Cube vertices (unit cube centered at origin)
        let vertices = vec![
            // Front face
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
            // Back face
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
        ];
        
        // Cube indices (12 triangles, 36 indices)
        let indices = vec![
            // Front
            0, 1, 2,  2, 3, 0,
            // Back
            5, 4, 7,  7, 6, 5,
            // Left
            4, 0, 3,  3, 7, 4,
            // Right
            1, 5, 6,  6, 2, 1,
            // Top
            3, 2, 6,  6, 7, 3,
            // Bottom
            4, 5, 1,  1, 0, 4,
        ];
        
        (vertices, indices)
    }
    
    /// Update instance buffer with new voxel data
    pub fn update_instances(&self, queue: &wgpu::Queue, instances: &[VoxelInstance]) {
        if instances.len() as u32 > self.max_instances {
            log::warn!(
                "Too many voxel instances: {} > {}. Truncating.",
                instances.len(),
                self.max_instances
            );
        }
        
        let count = instances.len().min(self.max_instances as usize);
        if count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances[..count]),
            );
        }
    }
    
    /// Render voxels
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        instance_count: u32,
    ) {
        if instance_count == 0 {
            return;
        }
        
        let count = instance_count.min(self.max_instances);
        
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.index_count, 0, 0..count);
    }
}
