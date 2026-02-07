//! # Cell Rendering - GPU Instanced Billboards
//!
//! This module implements high-performance cell rendering using GPU instancing.
//! Cells are rendered as camera-facing billboards with realistic sphere-like shading.
//!
//! ## Architecture
//!
//! The cell renderer supports multiple cell types with distinct shaders:
//! - CellTypeRegistry manages per-type render pipelines
//! - Shared depth pass across all cell types for early-Z optimization
//! - Type-specific color passes with grouped draw calls
//!
//! See `.kiro/specs/cell-rendering-pipeline/` for full design documentation.

use crate::cell::types::{CellType, CellTypeVisuals};
use crate::cell::type_registry::CellTypeRegistry;
use crate::cell::behaviors::{create_behavior, CellBehavior};
use crate::genome::{Genome, ModeSettings};
use crate::rendering::instance_builder::{CellInstance, InstanceBuilder};
use crate::simulation::CanonicalState;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::ops::Range;

/// Camera uniform data for shaders.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    /// Current time for animated shaders (e.g., flagellocyte tail animation)
    time: f32,
    /// LOD scale factor for distance calculations (higher = more aggressive LOD)
    lod_scale_factor: f32,
    /// LOD threshold for Low (32x32) to Medium (64x64) transition
    lod_threshold_low: f32,
    /// LOD threshold for Medium (64x64) to High (128x128) transition
    lod_threshold_medium: f32,
    /// LOD threshold for High (128x128) to Ultra (256x256) transition
    lod_threshold_high: f32,
}

/// Lighting uniform data for shaders.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LightingUniform {
    light_dir: [f32; 3],
    ambient: f32,
    light_color: [f32; 3],
    _padding: f32,
}

/// Cell group for batched rendering by type.
#[derive(Debug, Clone)]
pub struct CellGroup {
    pub cell_type: CellType,
    pub range: Range<u32>,
}

/// High-performance cell renderer using GPU instancing.
///
/// Supports multiple cell types with distinct appearance shaders.
/// Uses a shared depth pass for early-Z optimization and type-specific
/// color passes for visual variety.
pub struct CellRenderer {
    /// Current render target width
    pub width: u32,
    
    /// Current render target height
    pub height: u32,
    
    /// Depth texture view for render pass attachment
    pub depth_view: wgpu::TextureView,
    
    /// Depth texture for depth testing
    depth_texture: wgpu::Texture,
    
    /// Cell type registry with per-type pipelines
    type_registry: CellTypeRegistry,
    
    /// Shared depth-only pipeline for all cell types.
    /// Currently unused - depth pre-pass is skipped for ray-marched billboards.
    /// Kept for future optimization with proper depth shader.
    #[allow(dead_code)]
    depth_pipeline: wgpu::RenderPipeline,
    
    /// Camera uniform buffer
    camera_buffer: wgpu::Buffer,
    
    /// Lighting uniform buffer
    lighting_buffer: wgpu::Buffer,
    
    /// Bind group for camera and lighting uniforms
    bind_group: wgpu::BindGroup,
    
    /// Instance buffer for cell data
    instance_buffer: wgpu::Buffer,
    
    /// Current instance buffer capacity
    instance_capacity: usize,
    
    /// Behavior handlers indexed by CellType
    behaviors: Vec<Box<dyn CellBehavior>>,
}

impl CellRenderer {
    /// Depth texture format used by the renderer.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    
    /// Create a new cell renderer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for creating GPU resources
    /// * `queue` - The wgpu queue for buffer uploads
    /// * `config` - Surface configuration for format and dimensions
    /// * `capacity` - Initial instance buffer capacity
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        capacity: usize,
    ) -> Self {
        let width = config.width;
        let height = config.height;
        
        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);
        
        // Create cell type registry with per-type pipelines
        let type_registry = CellTypeRegistry::new(device, config.format, Self::DEPTH_FORMAT);
        
        // Create shared depth pipeline
        let depth_pipeline = Self::create_depth_pipeline(device, &type_registry.bind_group_layout);
        
        // Create camera uniform buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Renderer Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create lighting uniform buffer
        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Renderer Lighting Buffer"),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Bake hex pattern texture via compute shader
        let (hex_texture_view, hex_sampler) = Self::bake_hex_texture(device, queue);
        
        // Create bind group with hex bake texture
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Renderer Bind Group"),
            layout: &type_registry.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&hex_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&hex_sampler),
                },
            ],
        });
        
        // Create instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Renderer Instance Buffer"),
            size: (capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create behavior handlers for all cell types
        let mut behaviors: Vec<Box<dyn CellBehavior>> = Vec::with_capacity(CellType::COUNT);
        for cell_type in CellType::iter() {
            behaviors.push(create_behavior(cell_type));
        }
        
        Self {
            width,
            height,
            depth_view,
            depth_texture,
            type_registry,
            depth_pipeline,
            camera_buffer,
            lighting_buffer,
            bind_group,
            instance_buffer,
            instance_capacity: capacity,
            behaviors,
        }
    }
    
    /// Bake the Goldberg hex triplet pattern into an equirectangular texture via compute shader.
    /// Returns (texture_view, sampler). Run once at init — the texture is static.
    fn bake_hex_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::TextureView, wgpu::Sampler) {
        let tex_width = 512u32;
        let tex_height = 256u32;

        // Create the output texture (RG32Float for edge_dist + is_hex)
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hex Bake Texture"),
            size: wgpu::Extent3d {
                width: tex_width,
                height: tex_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler with linear filtering and repeat wrapping (for longitude wrap)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Hex Bake Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create compute pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hex Bake Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/hex_bake.wgsl").into()),
        });

        let bake_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hex Bake Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hex Bake Pipeline Layout"),
            bind_group_layouts: &[&bake_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hex Bake Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bake_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hex Bake Bind Group"),
            layout: &bake_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Hex Bake Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hex Bake Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compute_pipeline);
            pass.set_bind_group(0, &bake_bind_group, &[]);
            // Workgroup size is 8x8, dispatch enough to cover tex_width x tex_height
            pass.dispatch_workgroups(
                (tex_width + 7) / 8,
                (tex_height + 7) / 8,
                1,
            );
        }
        queue.submit(std::iter::once(encoder.finish()));

        (texture_view, sampler)
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cell Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
    
    /// Create the shared depth-only pipeline.
    fn create_depth_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        // Load depth shader (uses unified cell shader)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cells/cell_unified.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cell Depth Pipeline Layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create depth-only render pipeline
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Depth Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Self::instance_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: None, // Depth-only, no fragment shader
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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
        })
    }
    
    /// Get the vertex buffer layout for cell instances.
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

    /// Resize the renderer textures.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;

        // Recreate depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    
    /// Ensure instance buffer has sufficient capacity.
    fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required > self.instance_capacity {
            let new_capacity = required.max(self.instance_capacity * 2);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cell Renderer Instance Buffer"),
                size: (new_capacity * std::mem::size_of::<CellInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_capacity;
        }
    }
    
    /// Update camera uniform buffer.
    fn update_camera(
        &self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        camera_rotation: Quat,
        current_time: f32,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
    ) {
        // Calculate view matrix from camera position and rotation
        let view = Mat4::from_rotation_translation(camera_rotation, camera_pos).inverse();
        
        // Calculate projection matrix (perspective)
        let aspect = self.width as f32 / self.height as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 1000.0);
        
        let view_proj = proj * view;
        
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time: current_time,
            lod_scale_factor,
            lod_threshold_low,
            lod_threshold_medium,
            lod_threshold_high,
        };
        
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
    }
    
    /// Update lighting uniform buffer.
    fn update_lighting(&self, queue: &wgpu::Queue) {
        let lighting_uniform = LightingUniform {
            light_dir: [-0.5, -0.7, -0.5], // Normalized in shader
            ambient: 0.15,
            light_color: [1.0, 0.98, 0.95],
            _padding: 0.0,
        };
        
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting_uniform));
    }
    
    /// Group cells by type for batched rendering.
    ///
    /// Returns a list of cell groups, each containing a cell type and the
    /// range of instances in the buffer for that type. Empty groups are excluded.
    pub fn group_cells_by_type(
        &self,
        state: &CanonicalState,
        genome: Option<&Genome>,
    ) -> Vec<CellGroup> {
        if state.cell_count == 0 {
            return Vec::new();
        }
        
        // Count cells per type
        let mut type_counts = [0u32; CellType::COUNT];
        
        for i in 0..state.cell_count {
            let mode_index = state.mode_indices[i];
            let cell_type_index = if let Some(g) = genome {
                if let Some(mode) = g.modes.get(mode_index) {
                    mode.cell_type as usize
                } else {
                    0
                }
            } else {
                0
            };
            
            let cell_type_index = cell_type_index.min(CellType::COUNT - 1);
            type_counts[cell_type_index] += 1;
        }
        
        // Build groups with ranges
        let mut groups = Vec::new();
        let mut offset = 0u32;
        
        for (type_index, &count) in type_counts.iter().enumerate() {
            if count > 0 {
                if let Some(cell_type) = CellType::from_index(type_index as u32) {
                    groups.push(CellGroup {
                        cell_type,
                        range: offset..(offset + count),
                    });
                    offset += count;
                }
            }
        }
        
        groups
    }
    
    /// Build instance data for all cells.
    ///
    /// Reads cell_type from mode settings and populates type_data from
    /// the appropriate behavior module. Cells are sorted by type for
    /// efficient batched rendering.
    pub fn build_instances(
        &self,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
        _lod_scale_factor: f32,
        _lod_threshold_low: f32,
        _lod_threshold_medium: f32,
        _lod_threshold_high: f32,
        _lod_debug_colors: bool,
    ) -> Vec<CellInstance> {
        if state.cell_count == 0 {
            return Vec::new();
        }
        
        // First pass: collect cells with their types for sorting
        let mut cells_with_types: Vec<(usize, usize)> = Vec::with_capacity(state.cell_count);
        
        for i in 0..state.cell_count {
            let mode_index = state.mode_indices[i];
            let cell_type_index = if let Some(g) = genome {
                if let Some(mode) = g.modes.get(mode_index) {
                    mode.cell_type as usize
                } else {
                    0
                }
            } else {
                0
            };
            
            let cell_type_index = cell_type_index.min(CellType::COUNT - 1);
            cells_with_types.push((i, cell_type_index));
        }
        
        // Sort by cell type for batched rendering
        cells_with_types.sort_by_key(|&(_, type_idx)| type_idx);
        
        // Second pass: build instance data
        let mut instances = Vec::with_capacity(state.cell_count);
        
        // Simulate camera position for LOD calculation (center of world)
        let _camera_pos = Vec3::new(0.0, 0.0, 0.0);
        
        for (cell_index, cell_type_index) in cells_with_types {
            let position = state.positions[cell_index];
            let rotation = state.rotations[cell_index];
            let radius = state.radii[cell_index];
            let mode_index = state.mode_indices[cell_index];
            
            // Get mode settings
            let mode_settings = if let Some(g) = genome {
                g.modes.get(mode_index).cloned().unwrap_or_default()
            } else {
                ModeSettings::default()
            };
            
            // Get cell type visuals
            let visuals = cell_type_visuals
                .and_then(|v| v.get(cell_type_index))
                .copied()
                .unwrap_or_default();
            
            // Get type-specific instance data from behavior module
            // All cell types use their behavior modules for consistent handling
            // IMPORTANT: data[7] must contain cell_type for the hybrid shader to branch correctly
            let mut data = if cell_type_index < self.behaviors.len() {
                self.behaviors[cell_type_index].build_instance_data(&mode_settings)
            } else {
                crate::cell::behaviors::TypeSpecificInstanceData::empty()
            };
            
            // Ensure cell_type is stored in data[7] for hybrid shader branching
            data.data[4] = visuals.nucleus_scale;
            data.data[7] = cell_type_index as f32;
            
            // Pack Goldberg ridge params into type_data[0..3] for non-tail cells
            // (Flagellocyte tail params are already packed by its behavior module)
            let cell_type = CellType::all().get(cell_type_index).copied().unwrap_or(CellType::Test);
            if cell_type != CellType::Flagellocyte {
                data.data[0] = visuals.goldberg_scale;
                data.data[1] = visuals.goldberg_ridge_width;
                data.data[2] = visuals.goldberg_meander;
                data.data[3] = visuals.goldberg_ridge_strength;
            }
            
            let type_data = data;
            
            // Build instance
            let instance = CellInstance {
                position: position.to_array(),
                radius,
                color: [
                    mode_settings.color.x,
                    mode_settings.color.y,
                    mode_settings.color.z,
                    mode_settings.opacity,
                ],
                visual_params: [
                    visuals.specular_strength,
                    visuals.specular_power,
                    visuals.fresnel_strength,
                    mode_settings.emissive,
                ],
                rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                type_data: type_data.data,
            };
            
            instances.push(instance);
        }
        
        instances
    }
    
    /// Render cells with depth pre-pass for PreviewScene.
    ///
    /// This method handles the complete render pipeline:
    /// 1. Build instance data from canonical state
    /// 2. Upload instances to GPU
    /// 3. Render depth pre-pass (all cells, shared shader)
    /// 4. Render color passes (grouped by cell type)
    pub fn render_with_depth_prepass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
        camera_pos: Vec3,
        camera_rotation: Quat,
        current_time: f32,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
        _lod_debug_colors: bool,
    ) {
        if state.cell_count == 0 {
            return;
        }
        
        // Build instance data
        let instances = self.build_instances(
            state, 
            genome, 
            cell_type_visuals, 
            lod_scale_factor, 
            lod_threshold_low, 
            lod_threshold_medium, 
            lod_threshold_high,
            _lod_debug_colors,
        );
        let instance_count = instances.len() as u32;
        
        if instance_count == 0 {
            return;
        }
        
        // Ensure buffer capacity
        self.ensure_capacity(device, instances.len());
        
        // Upload instance data
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        
        // Update uniforms
        self.update_camera(queue, camera_pos, camera_rotation, current_time, lod_scale_factor, lod_threshold_low, lod_threshold_medium, lod_threshold_high);
        self.update_lighting(queue);
        
        // Get cell groups for batched rendering
        let groups = self.group_cells_by_type(state, genome);
        
        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cell Renderer Encoder"),
        });
        
        // Note: Depth pre-pass is skipped for ray-marched billboards because the
        // fragment shader discards pixels outside the sphere. A depth-only pass
        // would write depth for the full quad, causing the color pass to fail
        // depth tests for valid sphere pixels.
        //
        // For proper depth pre-pass with ray-marched spheres, we would need a
        // depth shader that also performs ray-sphere intersection and writes
        // the correct sphere depth. For now, we render directly with depth write.
        
        // Single pass: Color with depth write (grouped by cell type)
        {
            let mut color_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cell Color Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve background
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear depth at start
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            color_pass.set_bind_group(0, &self.bind_group, &[]);
            color_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            
            // Render each cell type group with its specific pipeline
            for group in &groups {
                if group.range.is_empty() {
                    continue;
                }
                
                let pipeline = self.type_registry.get_pipeline(group.cell_type);
                color_pass.set_pipeline(pipeline);
                color_pass.draw(0..4, group.range.clone());
            }
        }
        
        queue.submit(std::iter::once(encoder.finish()));
    }
    
    /// Render cells using pre-built instance buffer from InstanceBuilder (for GpuScene).
    ///
    /// This method is used when instance data is built on the GPU via compute shaders.
    /// It skips the CPU-side instance building and uses the GPU-generated buffer directly.
    /// Uses indirect drawing to read the actual visible count from the GPU buffer.
    ///
    /// Instances are dynamically partitioned by cell type with each type getting
    /// consecutive instances based on actual counts. Each cell type is rendered
    /// with its appropriate pipeline for correct visual appearance.
    pub fn render_with_encoder(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        instance_builder: &InstanceBuilder,
        _visible_count: u32, // Deprecated: use indirect buffer instead
        camera_pos: Vec3,
        camera_rotation: Quat,
        current_time: f32,
        lod_scale_factor: f32,
        lod_threshold_low: f32,
        lod_threshold_medium: f32,
        lod_threshold_high: f32,
    ) {
        // Update uniforms with LOD settings from UI
        self.update_camera(queue, camera_pos, camera_rotation, current_time, lod_scale_factor, lod_threshold_low, lod_threshold_medium, lod_threshold_high);
        self.update_lighting(queue);
        
        // Note: Depth pre-pass is skipped for ray-marched billboards because the
        // fragment shader discards pixels outside the sphere. See render_with_depth_prepass
        // for detailed explanation.
        
        // Render each cell type with its appropriate pipeline
        // Instances are sorted by type: Test cells at [0, test_count), Flagellocytes at [capacity/2, capacity/2 + flagellocyte_count)
        {
            let mut color_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cell Color Pass"),
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
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear depth at start
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            color_pass.set_bind_group(0, &self.bind_group, &[]);

            // Render all cells with a single draw call
            // Instance buffer is dynamically allocated (not partitioned by type)
            // Cell type is stored in instance data (type_data_1.w) for shader to use
            let pipeline = self.type_registry.get_pipeline(CellType::Test);
            color_pass.set_pipeline(pipeline);
            color_pass.set_vertex_buffer(0, instance_builder.instance_buffer.slice(..));
            
            // Use main indirect buffer which has total visible count
            color_pass.draw_indirect(&instance_builder.indirect_buffer, 0);
        }
    }
}
