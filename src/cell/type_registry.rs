//! Cell Type Registry
//!
//! Central registry for cell type renderers. Each cell type has its own
//! appearance shader pipeline, while sharing common infrastructure for
//! depth passes and instance data.
//!
//! # Architecture
//!
//! The registry manages render pipelines for all registered cell types:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         CellTypeRegistry                                │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    Shared Bind Group Layout                      │   │
//! │  │  • Binding 0: Camera uniform (view_proj, camera_pos)            │   │
//! │  │  • Binding 1: Lighting uniform (light_dir, ambient, etc.)       │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
//! │  │   Test      │  │Flagellocyte │  │  Lipocyte   │  ...                │
//! │  │  Pipeline   │  │  Pipeline   │  │  Pipeline   │                     │
//! │  │             │  │             │  │             │                     │
//! │  │ test_cell   │  │flagellocyte │  │  lipocyte   │                     │
//! │  │   .wgsl     │  │   .wgsl     │  │   .wgsl     │                     │
//! │  └─────────────┘  └─────────────┘  └─────────────┘                     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! The registry is created once at application startup and used by the
//! [`CellRenderer`](crate::rendering::CellRenderer) to render cells:
//!
//! ```ignore
//! // Create registry at startup
//! let registry = CellTypeRegistry::new(&device, surface_format, depth_format);
//!
//! // During rendering, get pipeline for each cell type
//! for (cell_type, cell_range) in grouped_cells {
//!     let pipeline = registry.get_pipeline(cell_type);
//!     render_pass.set_pipeline(pipeline);
//!     render_pass.draw(0..4, cell_range);
//! }
//! ```
//!
//! # Adding a New Cell Type
//!
//! When adding a new cell type, the registry automatically picks it up if:
//!
//! 1. The [`CellType`] enum has the new variant
//! 2. The shader file exists at the path returned by [`CellType::shader_path()`]
//! 3. The `load_shader_source()` method has a match arm for the new type
//!
//! ## Required Changes
//!
//! Update `load_shader_source()` in this file:
//!
//! ```ignore
//! fn load_shader_source(cell_type: CellType) -> &'static str {
//!     match cell_type {
//!         CellType::Test => include_str!("../../shaders/cells/test_cell.wgsl"),
//!         CellType::Flagellocyte => include_str!("../../shaders/cells/flagellocyte.wgsl"),
//!         // Add new cell types here
//!     }
//! }
//! ```
//!
//! # Instance Buffer Layout
//!
//! All cell type pipelines share the same instance buffer layout (96 bytes):
//!
//! | Offset | Size | Attribute | Description |
//! |--------|------|-----------|-------------|
//! | 0      | 12   | position  | World position (vec3) |
//! | 12     | 4    | radius    | Cell radius (f32) |
//! | 16     | 16   | color     | RGBA color (vec4) |
//! | 32     | 16   | visual_params | Specular, power, fresnel, emissive (vec4) |
//! | 48     | 16   | rotation  | Orientation quaternion (vec4) |
//! | 64     | 16   | type_data_0 | Type-specific data [0..4] (vec4) |
//! | 80     | 16   | type_data_1 | Type-specific data [4..8] (vec4) |
//!
//! # Shader Requirements
//!
//! Each cell type shader must define:
//!
//! - `vs_main`: Vertex shader entry point
//! - `fs_main`: Fragment shader entry point
//!
//! And accept the instance buffer layout as vertex attributes.

use crate::cell::types::CellType;

/// Central registry for cell type renderers.
///
/// Manages render pipelines for each cell type. All pipelines share the same
/// bind group layouts for camera and instance data, but use type-specific
/// appearance shaders.
///
/// # Initialization
///
/// The registry is created once at application startup. It iterates over all
/// [`CellType`] variants and creates a render pipeline for each:
///
/// ```ignore
/// let registry = CellTypeRegistry::new(&device, surface_format, depth_format);
/// ```
///
/// # Pipeline Lookup
///
/// During rendering, use [`get_pipeline()`](Self::get_pipeline) to retrieve
/// the appropriate pipeline for each cell type:
///
/// ```ignore
/// let pipeline = registry.get_pipeline(CellType::Test);
/// render_pass.set_pipeline(pipeline);
/// ```
///
/// # Shared Resources
///
/// All pipelines share:
/// - **Bind group layout**: Camera and lighting uniforms
/// - **Instance buffer layout**: 96-byte per-instance data
/// - **Depth/stencil state**: Consistent depth testing
///
/// # Thread Safety
///
/// The registry is immutable after creation and can be safely shared
/// across threads (it implements `Send + Sync` implicitly through its fields).
pub struct CellTypeRegistry {
    /// Render pipelines indexed by CellType.
    /// Each pipeline uses the type-specific appearance shader.
    pipelines: Vec<wgpu::RenderPipeline>,
    
    /// Bind group layout shared by all cell type pipelines.
    /// Contains camera uniforms and lighting data.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CellTypeRegistry {
    /// Create a new cell type registry with pipelines for all registered cell types.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating GPU resources
    /// * `surface_format` - The surface texture format for color output
    /// * `depth_format` - The depth texture format
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        // Create shared bind group layout for camera and lighting
        let bind_group_layout = Self::create_bind_group_layout(device);
        
        // Create pipelines for all registered cell types
        let mut pipelines = Vec::with_capacity(CellType::COUNT);
        
        for cell_type in CellType::iter() {
            let pipeline = Self::create_pipeline(
                device,
                surface_format,
                depth_format,
                &bind_group_layout,
                cell_type,
            );
            pipelines.push(pipeline);
        }
        
        Self {
            pipelines,
            bind_group_layout,
        }
    }
    
    /// Get the render pipeline for a specific cell type.
    ///
    /// # Arguments
    ///
    /// * `cell_type` - The cell type to get the pipeline for
    ///
    /// # Returns
    ///
    /// Reference to the render pipeline for the specified cell type.
    pub fn get_pipeline(&self, cell_type: CellType) -> &wgpu::RenderPipeline {
        &self.pipelines[cell_type as usize]
    }
    
    /// Create the shared bind group layout for all cell type pipelines.
    ///
    /// Layout:
    /// - Binding 0: Camera uniform (view_proj matrix, camera position)
    /// - Binding 1: Lighting uniform (light direction, ambient, etc.)
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Type Registry Bind Group Layout"),
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
            ],
        })
    }
    
    /// Create a render pipeline for a specific cell type.
    fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        bind_group_layout: &wgpu::BindGroupLayout,
        cell_type: CellType,
    ) -> wgpu::RenderPipeline {
        // Load the shader for this cell type
        let shader_source = Self::load_shader_source(cell_type);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{:?} Cell Shader", cell_type)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{:?} Cell Pipeline Layout", cell_type)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{:?} Cell Pipeline", cell_type)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Self::instance_buffer_layout()],
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
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Billboards face camera, no culling needed
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                // Use Less comparison - the shader outputs proper sphere surface depth
                // which enables correct mutual overlap where spheres intersect
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }
    
    /// Load shader source for a cell type.
    ///
    /// This uses include_str! at compile time for built-in shaders.
    /// Each cell type has its own shader: cell_0.wgsl, cell_1.wgsl, etc.
    fn load_shader_source(cell_type: CellType) -> &'static str {
        match cell_type {
            CellType::Test => include_str!("../../shaders/cells/cell_0.wgsl"),
            CellType::Flagellocyte => include_str!("../../shaders/cells/cell_1.wgsl"),
            CellType::Phagocyte => include_str!("../../shaders/cells/cell_2.wgsl"),
            CellType::Photocyte => include_str!("../../shaders/cells/cell_3.wgsl"),
            CellType::Lipocyte => include_str!("../../shaders/cells/cell_4.wgsl"),
        }
    }
    
    /// Get the vertex buffer layout for cell instances.
    ///
    /// This layout matches the CellInstance struct:
    /// - position: vec3<f32> (12 bytes)
    /// - radius: f32 (4 bytes)
    /// - color: vec4<f32> (16 bytes)
    /// - visual_params: vec4<f32> (16 bytes)
    /// - rotation: vec4<f32> (16 bytes)
    /// - type_data: [f32; 8] (32 bytes)
    /// Total: 96 bytes (16-byte aligned)
    fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 96, // 96 bytes per instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Position (vec3)
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Radius (f32)
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                // Color (vec4)
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Visual params (vec4: specular, power, fresnel, emissive)
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Rotation (vec4: quaternion)
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Type data 0-3 (vec4)
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Type data 4-7 (vec4)
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
