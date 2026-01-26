//! GPU-based Surface Nets for density field rendering
//!
//! Extracts isosurfaces entirely on GPU using compute shaders.
//! Input: density buffer (f32 per voxel)
//! Output: triangle mesh via indirect draw

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Grid resolution (matching fluid simulation)
pub const GRID_RESOLUTION: u32 = 128;
pub const TOTAL_VOXELS: usize = (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION) as usize;

/// GPU vertex format (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub fluid_type: f32,  // 0=empty, 1=water, 2=lava, 3=steam
    pub normal: [f32; 3],
    pub _pad1: f32,
}

/// Surface nets parameters (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SurfaceNetsGpuParams {
    pub grid_resolution: u32,
    pub iso_level: f32,
    pub cell_size: f32,
    pub max_vertices: u32,
    
    pub grid_origin: [f32; 3],
    pub max_indices: u32,
}

/// Counter struct for reading back (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Counters {
    pub vertex_count: u32,
    pub index_count: u32,
}

/// GPU Surface Nets renderer
pub struct GpuSurfaceNets {
    // Compute pipelines
    reset_pipeline: wgpu::ComputePipeline,
    vertex_pipeline: wgpu::ComputePipeline,
    index_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    
    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    
    // Buffers
    density_buffer: wgpu::Buffer,
    fluid_type_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    vertex_map_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    counter_staging_buffer: wgpu::Buffer,
    indirect_draw_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    render_params_buffer: wgpu::Buffer,
    
    // Bind groups
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    
    // Configuration
    max_vertices: u32,
    max_indices: u32,
    world_radius: f32,
    world_center: Vec3,
    iso_level: f32,
    
    // Cached counts (updated after GPU readback)
    pub vertex_count: u32,
    pub index_count: u32,
    
    // Screen dimensions
    pub width: u32,
    pub height: u32,
}

/// Camera uniform for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Render params for lighting control
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DensityMeshParams {
    pub base_color: [f32; 3],
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
    pub fresnel: f32,
    pub fresnel_power: f32,
    pub rim: f32,
    pub reflection: f32,
    pub alpha: f32,
}

impl Default for DensityMeshParams {
    fn default() -> Self {
        Self {
            base_color: [0.2, 0.5, 0.9],
            ambient: 0.15,
            diffuse: 0.6,
            specular: 0.8,
            shininess: 64.0,
            fresnel: 0.5,
            fresnel_power: 3.0,
            rim: 0.3,
            reflection: 0.3,
            alpha: 0.85,
        }
    }
}

impl GpuSurfaceNets {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        world_radius: f32,
        world_center: Vec3,
        width: u32,
        height: u32,
    ) -> Self {
        let max_vertices = 500_000u32;
        let max_indices = 1_500_000u32;
        
        // Calculate grid parameters
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);
        
        // Create compute shader
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Surface Nets Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/surface_nets_gpu.wgsl").into()
            ),
        });
        
        // Create render shader
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Surface Nets Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/density_mesh.wgsl").into()
            ),
        });
        
        // Create buffers
        let density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let fluid_type_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Type Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Nets Vertex Buffer"),
            size: (max_vertices as usize * std::mem::size_of::<GpuVertex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Nets Index Buffer"),
            size: (max_indices as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        
        let vertex_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Map Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Buffer"),
            size: std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Staging Buffer"),
            size: std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Indirect draw buffer: index_count, instance_count, first_index, base_vertex, first_instance
        let indirect_draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Draw Buffer"),
            size: 20, // 5 * u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        
        let params = SurfaceNetsGpuParams {
            grid_resolution: GRID_RESOLUTION,
            iso_level: 0.5,
            cell_size,
            max_vertices,
            grid_origin: grid_origin.to_array(),
            max_indices,
        };
        
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Surface Nets Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let camera_uniform = CameraUniform {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };
        
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Surface Nets Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let render_params = DensityMeshParams::default();
        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Density Mesh Render Params Buffer"),
            contents: bytemuck::cast_slice(&[render_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Compute bind group layout
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surface Nets Compute Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface Nets Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: density_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fluid_type_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vertex_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: index_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: vertex_map_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: counter_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: indirect_draw_buffer.as_entire_binding() },
            ],
        });
        
        // Compute pipeline layout
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Surface Nets Compute Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipelines
        let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reset Counters Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("reset_counters"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let vertex_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generate Vertices Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("generate_vertices"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let index_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generate Indices Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("generate_indices"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Finalize Indirect Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("finalize_indirect"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Render bind group layout
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surface Nets Render Layout"),
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
        
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface Nets Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Render pipeline layout
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Surface Nets Render Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface Nets Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3, // position
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32, // fluid_type
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3, // normal
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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
                cull_mode: None, // Render both sides
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // Disable depth writing for transparent surfaces
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
            reset_pipeline,
            vertex_pipeline,
            index_pipeline,
            finalize_pipeline,
            render_pipeline,
            density_buffer,
            fluid_type_buffer,
            vertex_buffer,
            index_buffer,
            vertex_map_buffer,
            counter_buffer,
            counter_staging_buffer,
            indirect_draw_buffer,
            params_buffer,
            camera_buffer,
            render_params_buffer,
            compute_bind_group,
            render_bind_group,
            max_vertices,
            max_indices,
            world_radius,
            world_center,
            iso_level: 0.5,
            vertex_count: 0,
            index_count: 0,
            width,
            height,
        }
    }
    
    /// Upload density data to GPU
    pub fn upload_density(&self, queue: &wgpu::Queue, density: &[f32]) {
        assert_eq!(density.len(), TOTAL_VOXELS);
        queue.write_buffer(&self.density_buffer, 0, bytemuck::cast_slice(density));
    }
    
    /// Upload fluid type data to GPU
    pub fn upload_fluid_types(&self, queue: &wgpu::Queue, fluid_types: &[u32]) {
        assert_eq!(fluid_types.len(), TOTAL_VOXELS);
        queue.write_buffer(&self.fluid_type_buffer, 0, bytemuck::cast_slice(fluid_types));
    }

    /// Get density buffer for GPU writes
    pub fn density_buffer(&self) -> &wgpu::Buffer {
        &self.density_buffer
    }

    /// Get fluid type buffer for GPU writes
    pub fn fluid_type_buffer(&self) -> &wgpu::Buffer {
        &self.fluid_type_buffer
    }

    /// Update iso level
    pub fn set_iso_level(&mut self, queue: &wgpu::Queue, iso_level: f32) {
        self.iso_level = iso_level;
        self.update_params(queue);
    }
    
    /// Set smoothing level for voxel aliasing reduction
    /// Lower values = smoother, more organic surfaces (0.2-0.4)
    /// Higher values = sharper, more detailed surfaces (0.6-0.8)
    pub fn set_smoothing_level(&mut self, queue: &wgpu::Queue, smoothing: f32) {
        // Map smoothing to iso level inversely for intuitive control
        // High smoothing = low iso (larger, smoother surfaces)
        // Low smoothing = high iso (smaller, sharper surfaces)
        let iso_level = 0.5 + (0.5 - smoothing) * 0.6; // Maps 0.0->0.8, 1.0->0.2
        self.set_iso_level(queue, iso_level.clamp(0.1, 0.9));
    }
    
    /// Enable ultra-smooth mode for maximum voxel aliasing reduction
    pub fn enable_ultra_smooth(&mut self, queue: &wgpu::Queue) {
        self.set_iso_level(queue, 0.15); // Very low iso for maximum smoothing
    }
    
    /// Enable sharp mode for detailed surfaces (more voxel definition)
    pub fn enable_sharp(&mut self, queue: &wgpu::Queue) {
        self.set_iso_level(queue, 0.75); // High iso for sharp details
    }
    
    /// Update render params (lighting, colors, etc.)
    pub fn update_render_params(&self, queue: &wgpu::Queue, params: &DensityMeshParams) {
        queue.write_buffer(&self.render_params_buffer, 0, bytemuck::cast_slice(&[*params]));
    }
    
    /// Set initial render params from editor state (called once during initialization)
    pub fn set_initial_params(&self, queue: &wgpu::Queue, editor_state: &crate::ui::panel_context::GenomeEditorState) {
        let params = DensityMeshParams {
            base_color: [0.2, 0.5, 0.9], // Default water blue color
            ambient: editor_state.fluid_ambient,
            diffuse: editor_state.fluid_diffuse,
            specular: editor_state.fluid_specular,
            shininess: editor_state.fluid_shininess,
            fresnel: editor_state.fluid_fresnel,
            fresnel_power: editor_state.fluid_fresnel_power,
            rim: editor_state.fluid_rim,
            reflection: editor_state.fluid_reflection,
            alpha: editor_state.fluid_alpha,
        };
        self.update_render_params(queue, &params);
    }
    
    /// Update params buffer
    fn update_params(&self, queue: &wgpu::Queue) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        
        let params = SurfaceNetsGpuParams {
            grid_resolution: GRID_RESOLUTION,
            iso_level: self.iso_level,
            cell_size,
            max_vertices: self.max_vertices,
            grid_origin: grid_origin.to_array(),
            max_indices: self.max_indices,
        };
        
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }
    
    /// Run surface nets extraction on GPU
    pub fn extract_mesh(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_count = (GRID_RESOLUTION + 3) / 4;
        
        // Pass 0: Reset counters
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reset Counters Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reset_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        
        // Pass 1: Generate vertices
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generate Vertices Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vertex_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }
        
        // Pass 2: Generate indices
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generate Indices Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.index_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }
        
        // Pass 3: Finalize indirect draw buffer
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Finalize Indirect Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        
        // Copy counters for readback (optional, for debug/stats)
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.counter_staging_buffer,
            0,
            std::mem::size_of::<Counters>() as u64,
        );
    }
    
    /// Read back mesh counts from GPU (call after extract_mesh completes)
    pub fn read_counts(&mut self, device: &wgpu::Device) {
        let slice = self.counter_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        {
            let data = slice.get_mapped_range();
            let counters: &Counters = bytemuck::from_bytes(&data);
            self.vertex_count = counters.vertex_count.min(self.max_vertices);
            self.index_count = counters.index_count.min(self.max_indices);
        }
        self.counter_staging_buffer.unmap();
    }
    
    /// Render the extracted mesh
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
    ) {
        // Update camera
        let view = glam::Mat4::look_at_rh(
            camera_pos,
            camera_pos + camera_rotation * Vec3::NEG_Z,
            camera_rotation * Vec3::Y,
        );
        let aspect = self.width as f32 / self.height as f32;
        let proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
        let view_proj = proj * view;
        
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
        
        // Render using indirect draw (count comes from GPU buffer)
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Surface Nets Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed_indirect(&self.indirect_draw_buffer, 0);
    }
    
    /// Resize for new screen dimensions
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
    
    /// Get triangle count
    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
}

/// Generate test density data - simple sphere
pub fn generate_test_density_sphere(center: Vec3, radius: f32, world_radius: f32, world_center: Vec3) -> Vec<f32> {
    let mut density = vec![0.0f32; TOTAL_VOXELS];
    
    let world_diameter = world_radius * 2.0;
    let cell_size = world_diameter / GRID_RESOLUTION as f32;
    let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);
    
    for z in 0..GRID_RESOLUTION {
        for y in 0..GRID_RESOLUTION {
            for x in 0..GRID_RESOLUTION {
                let world_pos = grid_origin + Vec3::new(
                    (x as f32 + 0.5) * cell_size,
                    (y as f32 + 0.5) * cell_size,
                    (z as f32 + 0.5) * cell_size,
                );
                
                let dist = (world_pos - center).length();
                let d = 1.0 - (dist / radius).clamp(0.0, 2.0);
                
                let idx = (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
                density[idx] = d.max(0.0);
            }
        }
    }
    
    density
}

/// Generate test density data - metaballs
pub fn generate_test_density_metaballs(
    balls: &[(Vec3, f32)],  // (center, radius)
    world_radius: f32,
    world_center: Vec3,
) -> Vec<f32> {
    let mut density = vec![0.0f32; TOTAL_VOXELS];
    
    let world_diameter = world_radius * 2.0;
    let cell_size = world_diameter / GRID_RESOLUTION as f32;
    let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);
    
    for z in 0..GRID_RESOLUTION {
        for y in 0..GRID_RESOLUTION {
            for x in 0..GRID_RESOLUTION {
                let world_pos = grid_origin + Vec3::new(
                    (x as f32 + 0.5) * cell_size,
                    (y as f32 + 0.5) * cell_size,
                    (z as f32 + 0.5) * cell_size,
                );
                
                let mut value = 0.0f32;
                for (center, radius) in balls {
                    let dist_sq = (world_pos - *center).length_squared();
                    if dist_sq > 0.001 {
                        value += (radius * radius) / dist_sq;
                    }
                }
                
                let idx = (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
                density[idx] = value;
            }
        }
    }
    
    density
}

/// Generate test density data - noise-based fluid blob
pub fn generate_test_density_noise(
    center: Vec3,
    base_radius: f32,
    world_radius: f32,
    world_center: Vec3,
    seed: u32,
) -> Vec<f32> {
    let mut density = vec![0.0f32; TOTAL_VOXELS];
    
    let world_diameter = world_radius * 2.0;
    let cell_size = world_diameter / GRID_RESOLUTION as f32;
    let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);
    
    // Simple hash function
    let hash = |x: i32, y: i32, z: i32| -> f32 {
        let n = (x.wrapping_mul(374761393) as u32)
            .wrapping_add(y.wrapping_mul(668265263) as u32)
            .wrapping_add(z.wrapping_mul(1274126177) as u32)
            .wrapping_add(seed);
        let n = n ^ (n >> 13);
        let n = n.wrapping_mul(1103515245);
        ((n & 0x7fffffff) as f32) / (0x7fffffff as f32)
    };
    
    for z in 0..GRID_RESOLUTION {
        for y in 0..GRID_RESOLUTION {
            for x in 0..GRID_RESOLUTION {
                let world_pos = grid_origin + Vec3::new(
                    (x as f32 + 0.5) * cell_size,
                    (y as f32 + 0.5) * cell_size,
                    (z as f32 + 0.5) * cell_size,
                );
                
                let offset = world_pos - center;
                let dist = offset.length();
                
                // Base sphere
                let base = 1.0 - (dist / base_radius).clamp(0.0, 1.5);
                
                // Add noise displacement
                let noise_scale = 0.15;
                let noise = hash(x as i32, y as i32, z as i32) * 2.0 - 1.0;
                
                let idx = (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
                density[idx] = (base + noise * noise_scale * base).max(0.0);
            }
        }
    }
    
    density
}
