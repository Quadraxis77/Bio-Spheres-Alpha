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

/// Padded resolution for surface nets processing (+1 cell padding on each side)
const PADDED_RESOLUTION: u32 = GRID_RESOLUTION + 2; // 130
const PADDED_VOXELS: usize = (PADDED_RESOLUTION * PADDED_RESOLUTION * PADDED_RESOLUTION) as usize;

/// GPU vertex format (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub fluid_type: f32, // 0=empty, 1=water, 2=lava, 3=steam
    pub normal: [f32; 3],
    pub _pad1: f32,
}

/// Surface nets parameters (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SurfaceNetsGpuParams {
    pub grid_resolution: u32, // Padded processing grid (130)
    pub iso_level: f32,
    pub cell_size: f32,
    pub max_vertices: u32,

    pub grid_origin: [f32; 3],
    pub max_indices: u32,

    pub density_resolution: u32, // Actual density data size (128)
    pub use_fast_early_out: u32, // 1 = check only 8 corners (organism skin), 0 = wider check (water)
    pub _pad_b: u32,
    pub _pad_c: u32,
}

/// Counter struct for reading back (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Counters {
    pub vertex_count: u32,
    pub index_count: u32,
}

/// Ice appearance uniform (must match IceRenderParams in ice_mesh.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct IceRenderParams {
    pub surface_color: [f32; 3],
    pub facet_scale: f32,
    pub deep_color: [f32; 3],
    pub displacement_strength: f32,
    pub facet_diffuse: f32,
    pub glint_shininess: f32,
    pub glint_strength: f32,
    pub alpha_base: f32,
    pub reflection_brightness: f32,
    pub fresnel_reflection: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    /// Tail padding: the ice pipeline shares its pipeline layout with the
    /// water pipeline, and wgpu validates the bound buffer against the
    /// LARGEST shader requirement across pipelines using that layout (the
    /// water shader's 112-byte RenderParams). The ice shader only reads the
    /// first 64 bytes; this padding satisfies the shared-layout minimum.
    pub _pad_tail: [f32; 12],
}

impl Default for IceRenderParams {
    fn default() -> Self {
        Self {
            surface_color: [0.80, 0.90, 1.00],
            facet_scale: 0.02,
            deep_color: [0.25, 0.45, 0.75],
            displacement_strength: 0.9,
            facet_diffuse: 0.2,
            glint_shininess: 64.0,
            glint_strength: 1.0,
            alpha_base: 0.82,
            reflection_brightness: 1.0,
            fresnel_reflection: 0.35,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad_tail: [0.0; 12],
        }
    }
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

    // Ice mesh: separate full-volume surface from water (rigid, faceted,
    // semitransparent, alpha-blended forward pass instead of WBOIT).
    // Reuses the same surface-nets compute pipelines with its own buffers.
    ice_render_pipeline: wgpu::RenderPipeline,
    ice_density_buffer: wgpu::Buffer,
    ice_vertex_buffer: wgpu::Buffer,
    ice_index_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    ice_vertex_map_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    ice_counter_buffer: wgpu::Buffer,
    ice_indirect_draw_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    ice_params_buffer: wgpu::Buffer,
    ice_compute_bind_group: wgpu::BindGroup,
    /// Ice appearance uniform (UI-driven) + its camera/params bind group.
    ice_render_params_buffer: wgpu::Buffer,
    ice_render_bind_group: wgpu::BindGroup,

    // Buffers
    density_buffer: wgpu::Buffer,
    fluid_type_buffer: wgpu::Buffer,
    solid_mask_buffer: wgpu::Buffer,
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

    // Smoothing
    smoothed_density_buffer: wgpu::Buffer,
    smooth_temp_buffer: wgpu::Buffer,
    smooth_params_buffer: wgpu::Buffer,
    smooth_pipeline: wgpu::ComputePipeline,
    smooth_bind_group: wgpu::BindGroup,
    smooth_blend_factor: f32,

    // Bind groups
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,

    // Shadow field (optional, set from gpu_scene after light field init)
    shadow_bind_group: Option<wgpu::BindGroup>,

    // Environment cubemap for reflections (fields kept alive to prevent GPU resource deallocation)
    #[allow(dead_code)]
    cubemap_texture: wgpu::Texture,
    #[allow(dead_code)]
    cubemap_view: wgpu::TextureView,
    #[allow(dead_code)]
    cubemap_sampler: wgpu::Sampler,
    cubemap_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    cubemap_bind_group_layout: wgpu::BindGroupLayout,
    cubemap_generated: bool,

    // WBOIT (Weighted Blended Order-Independent Transparency)
    oit_accum_texture: wgpu::Texture,
    oit_accum_view: wgpu::TextureView,
    oit_revealage_texture: wgpu::Texture,
    oit_revealage_view: wgpu::TextureView,
    oit_composite_pipeline: wgpu::RenderPipeline,
    oit_composite_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    oit_composite_bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    oit_sampler: wgpu::Sampler,

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

/// Smooth density params (must match shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SmoothDensityParams {
    grid_resolution: u32,
    blend_factor: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Camera uniform for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Cubemap face resolution
const CUBEMAP_FACE_SIZE: u32 = 2048;

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
    pub time: f32,
    pub wave_height: f32,
    pub wave_speed: f32,
    pub noise_scale: f32,
    pub noise_octaves: f32,
    pub noise_lacunarity: f32,
    pub noise_persistence: f32,
    pub reflection_brightness: f32,
    pub light_dir: [f32; 3],
    pub waterline_alpha: f32,
    pub gravity: [f32; 3],
    pub gravity_mode: u32,
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
            time: 0.0,
            wave_height: 0.8,
            wave_speed: 1.0,
            noise_scale: 0.5,
            noise_octaves: 3.0,
            noise_lacunarity: 2.0,
            noise_persistence: 0.5,
            reflection_brightness: 10.0,
            light_dir: [0.5, 1.0, 0.3],
            waterline_alpha: 0.05,
            gravity: [0.0, 1.0, 0.0],
            gravity_mode: 1,
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
        shadow_bind_group_layout: Option<&wgpu::BindGroupLayout>,
    ) -> Self {
        let max_vertices = 1_000_000u32;
        let max_indices = 3_000_000u32;

        // Calculate grid parameters
        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);

        // Create compute shader
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Surface Nets Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/surface_nets_gpu.wgsl").into(),
            ),
        });

        // Create render shader
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Surface Nets Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/density_mesh.wgsl").into(),
            ),
        });

        // Create buffers
        let density_buf_size = (TOTAL_VOXELS * std::mem::size_of::<f32>()) as u64;
        let density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Buffer"),
            size: density_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fluid_type_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Type Buffer"),
            size: (TOTAL_VOXELS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solid_mask_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Solid Mask Buffer"),
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
            size: (PADDED_VOXELS * std::mem::size_of::<u32>()) as u64,
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

        // Shift grid origin back by 1 cell so the 128^3 density data sits
        // at padded cells [1..128], with 1 cell of empty padding on each side
        let padded_origin = grid_origin - Vec3::splat(cell_size);

        let params = SurfaceNetsGpuParams {
            grid_resolution: PADDED_RESOLUTION,
            iso_level: 0.5,
            cell_size,
            max_vertices,
            grid_origin: padded_origin.to_array(),
            max_indices,
            density_resolution: GRID_RESOLUTION,
            use_fast_early_out: 0,
            _pad_b: 0,
            _pad_c: 0,
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
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
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
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fluid_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: vertex_map_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: indirect_draw_buffer.as_entire_binding(),
                },
            ],
        });

        // Compute pipeline layout
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        // === Environment cubemap bind group layout (group 2) ===
        let cubemap_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cubemap Reflection Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let render_pipeline_layout = if let Some(shadow_layout) = shadow_bind_group_layout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surface Nets Render Layout"),
                bind_group_layouts: &[
                    &render_bind_group_layout,
                    shadow_layout,
                    &cubemap_bind_group_layout,
                ],
                push_constant_ranges: &[],
            })
        } else {
            // Create a dummy shadow bind group layout matching the shader's group(1) expectations
            let dummy_shadow_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Dummy Shadow Field Layout"),
                    entries: &[
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
                    ],
                });
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surface Nets Render Layout"),
                bind_group_layouts: &[
                    &render_bind_group_layout,
                    &dummy_shadow_layout,
                    &cubemap_bind_group_layout,
                ],
                push_constant_ranges: &[],
            })
        };

        // Create render pipeline - WBOIT accumulation pass (two render targets)
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface Nets WBOIT Pipeline"),
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
                targets: &[
                    // Target 0: Accumulation (Rgba16Float) - additive blend: src*1 + dst*1
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 1: Revealage (R8Unorm) - multiplicative: dst * (1 - src_alpha)
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
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

        // === Ice mesh: separate surface-nets extraction + forward render ===
        // Ice volumes are typically far smaller than the water surface, so the
        // mesh buffers get half the water caps.
        let ice_max_vertices = max_vertices / 2;
        let ice_max_indices = max_indices / 2;

        let ice_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ice Density Buffer"),
            size: density_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ice_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ice Surface Nets Vertex Buffer"),
            size: (ice_max_vertices as usize * std::mem::size_of::<GpuVertex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let ice_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ice Surface Nets Index Buffer"),
            size: (ice_max_indices as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        let ice_vertex_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ice Vertex Map Buffer"),
            size: (PADDED_VOXELS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let ice_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ice Counter Buffer"),
            size: std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let ice_indirect_draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ice Indirect Draw Buffer"),
            size: 20, // 5 * u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // Same grid params as water, with the ice mesh's smaller caps. Must
        // use the wide early-out (not the 8-corner fast path): ice corners
        // go through the same 3x3x3 height filter as water, which bleeds
        // density one cell beyond a cell's own 8 corners. With the fast
        // 8-corner check, cells just outside an ice block (especially at
        // concave edges/corners) had all-zero raw corners and were skipped
        // even though their filtered corners crossed the isosurface,
        // leaving tiny holes in the mesh.
        let ice_params = SurfaceNetsGpuParams {
            max_vertices: ice_max_vertices,
            max_indices: ice_max_indices,
            use_fast_early_out: 0,
            ..params
        };
        let ice_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ice Surface Nets Params Buffer"),
            contents: bytemuck::cast_slice(&[ice_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Reuses the water compute pipelines - only the buffers differ.
        let ice_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ice Surface Nets Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ice_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ice_density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fluid_type_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ice_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ice_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: ice_vertex_map_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: ice_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: ice_indirect_draw_buffer.as_entire_binding(),
                },
            ],
        });

        // Ice appearance uniform + its own group(0) bind group (same layout
        // as the water render bind group: camera + params uniforms).
        let ice_render_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ice Render Params Buffer"),
                contents: bytemuck::cast_slice(&[IceRenderParams::default()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let ice_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ice Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ice_render_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Ice render pipeline: classic alpha blend straight into the scene
        // with depth writes - a rigid translucent solid, not an OIT fluid.
        let ice_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ice Mesh Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/ice_mesh.wgsl").into(),
            ),
        });

        let ice_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ice Mesh Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ice_shader,
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
                module: &ice_shader,
                entry_point: Some("fs_main"),
                // Same WBOIT targets as the water pipeline: ice draws into
                // the shared OIT pass, so water and ice composite
                // order-independently with no depth contention (a depth-write
                // forward pass z-fought the water surface at the ice boundary
                // and hid water behind ice completely).
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Cull back faces so a slab contributes one layer to the
                // weighted blend instead of front + back doubling its alpha.
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // OIT: no depth writes for transparents
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

        // === Density smoothing infrastructure ===
        let smooth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Smooth Density Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/smooth_density.wgsl").into(),
            ),
        });

        let smoothed_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Smoothed Density Buffer"),
            size: density_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let smooth_temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Smooth Temp Buffer"),
            size: density_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let smooth_params = SmoothDensityParams {
            grid_resolution: GRID_RESOLUTION,
            blend_factor: 0.15,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let smooth_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Smooth Density Params Buffer"),
            contents: bytemuck::cast_slice(&[smooth_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let smooth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Smooth Density Layout"),
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
                ],
            });

        let smooth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Smooth Density Bind Group"),
            layout: &smooth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: smooth_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: smoothed_density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: smooth_temp_buffer.as_entire_binding(),
                },
            ],
        });

        let smooth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Smooth Density Pipeline Layout"),
                bind_group_layouts: &[&smooth_bind_group_layout],
                push_constant_ranges: &[],
            });

        let smooth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Smooth Density Pipeline"),
            layout: Some(&smooth_pipeline_layout),
            module: &smooth_shader,
            entry_point: Some("smooth_density"),
            compilation_options: Default::default(),
            cache: None,
        });

        // === Environment cubemap creation ===
        let cubemap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment Cubemap"),
            size: wgpu::Extent3d {
                width: CUBEMAP_FACE_SIZE,
                height: CUBEMAP_FACE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let cubemap_view = cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Environment Cubemap View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let cubemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment Cubemap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let cubemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cubemap Reflection Bind Group"),
            layout: &cubemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cubemap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cubemap_sampler),
                },
            ],
        });

        // === CPU-generated cubemap uploaded via write_texture (no compute shader needed) ===

        // === WBOIT textures and composite pipeline ===
        let (oit_accum_texture, oit_accum_view) =
            Self::create_oit_accum_texture(device, width, height);
        let (oit_revealage_texture, oit_revealage_view) =
            Self::create_oit_revealage_texture(device, width, height);

        let oit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("OIT Composite Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let oit_composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("OIT Composite Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let oit_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OIT Composite Bind Group"),
            layout: &oit_composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&oit_accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&oit_revealage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&oit_sampler),
                },
            ],
        });

        let oit_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("OIT Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/oit_composite.wgsl").into(),
            ),
        });

        let oit_composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("OIT Composite Pipeline Layout"),
                bind_group_layouts: &[&oit_composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let oit_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("OIT Composite Pipeline"),
                layout: Some(&oit_composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &oit_composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &oit_composite_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState {
                            // Premultiplied alpha: output.rgb already multiplied by output.a
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
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
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Self {
            reset_pipeline,
            vertex_pipeline,
            index_pipeline,
            finalize_pipeline,
            render_pipeline,
            ice_render_pipeline,
            ice_density_buffer,
            ice_vertex_buffer,
            ice_index_buffer,
            ice_vertex_map_buffer,
            ice_counter_buffer,
            ice_indirect_draw_buffer,
            ice_params_buffer,
            ice_compute_bind_group,
            ice_render_params_buffer,
            ice_render_bind_group,
            density_buffer,
            fluid_type_buffer,
            solid_mask_buffer,
            vertex_buffer,
            index_buffer,
            vertex_map_buffer,
            counter_buffer,
            counter_staging_buffer,
            indirect_draw_buffer,
            params_buffer,
            camera_buffer,
            render_params_buffer,
            smoothed_density_buffer,
            smooth_temp_buffer,
            smooth_params_buffer,
            smooth_pipeline,
            smooth_bind_group,
            smooth_blend_factor: 0.15,
            compute_bind_group,
            render_bind_group,
            shadow_bind_group: None,
            cubemap_texture,
            cubemap_view,
            cubemap_sampler,
            cubemap_bind_group,
            cubemap_bind_group_layout,
            cubemap_generated: false,
            oit_accum_texture,
            oit_accum_view,
            oit_revealage_texture,
            oit_revealage_view,
            oit_composite_pipeline,
            oit_composite_bind_group,
            oit_composite_bind_group_layout,
            oit_sampler,
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
        queue.write_buffer(
            &self.fluid_type_buffer,
            0,
            bytemuck::cast_slice(fluid_types),
        );
    }

    /// Upload solid mask data to GPU
    pub fn upload_solid_mask(&self, queue: &wgpu::Queue, solid_mask: &[u32]) {
        assert_eq!(solid_mask.len(), TOTAL_VOXELS);
        queue.write_buffer(&self.solid_mask_buffer, 0, bytemuck::cast_slice(solid_mask));
    }

    /// Get density buffer for GPU writes
    /// Get the ice density buffer (filled by the fluid extract pass)
    pub fn ice_density_buffer(&self) -> &wgpu::Buffer {
        &self.ice_density_buffer
    }

    /// Update the ice appearance uniform (driven by the Fluid Settings UI)
    pub fn update_ice_render_params(&self, queue: &wgpu::Queue, params: &IceRenderParams) {
        queue.write_buffer(
            &self.ice_render_params_buffer,
            0,
            bytemuck::bytes_of(params),
        );
    }

    pub fn density_buffer(&self) -> &wgpu::Buffer {
        &self.density_buffer
    }

    /// Get smoothed density buffer (for cave shader caustics etc.)
    pub fn smoothed_density_buffer(&self) -> &wgpu::Buffer {
        &self.smoothed_density_buffer
    }

    /// Get fluid type buffer for GPU writes
    pub fn fluid_type_buffer(&self) -> &wgpu::Buffer {
        &self.fluid_type_buffer
    }

    /// Set the temporal blend factor for density smoothing
    /// Lower = more stable/slow (0.05), higher = more responsive (0.5)
    pub fn set_smooth_blend_factor(&mut self, queue: &wgpu::Queue, factor: f32) {
        self.smooth_blend_factor = factor;
        let params = SmoothDensityParams {
            grid_resolution: GRID_RESOLUTION,
            blend_factor: factor,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(
            &self.smooth_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    /// Run density smoothing: spatial blur + temporal blend
    /// Call after extract_to_surface_nets and before extract_mesh.
    /// Smoothed result is copied back to density_buffer so surface nets reads it.
    pub fn smooth_density(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_count = (GRID_RESOLUTION + 3) / 4;
        let buf_size = (TOTAL_VOXELS * std::mem::size_of::<f32>()) as u64;

        // Compute: read raw density + prev smoothed -> write to temp
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Smooth Density Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.smooth_pipeline);
            pass.set_bind_group(0, &self.smooth_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }

        // Copy temp -> smoothed (update temporal state for next frame)
        encoder.copy_buffer_to_buffer(
            &self.smooth_temp_buffer,
            0,
            &self.smoothed_density_buffer,
            0,
            buf_size,
        );

        // Copy smoothed -> density_buffer (so surface nets extraction reads smoothed data)
        encoder.copy_buffer_to_buffer(
            &self.smoothed_density_buffer,
            0,
            &self.density_buffer,
            0,
            buf_size,
        );
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
        queue.write_buffer(
            &self.render_params_buffer,
            0,
            bytemuck::cast_slice(&[*params]),
        );
    }

    /// Set the shadow bind group for light field shadow sampling.
    /// Called from gpu_scene after the light field system is initialized.
    pub fn set_shadow_bind_group(&mut self, bind_group: wgpu::BindGroup) {
        self.shadow_bind_group = Some(bind_group);
    }

    /// Generate the static environment cubemap by uploading CPU-generated sky data.
    /// Called once on first render. Uses queue.write_texture for maximum compatibility.
    pub fn generate_cubemap(&mut self, queue: &wgpu::Queue) {
        if self.cubemap_generated {
            return;
        }

        // Procedural sky cubemap - each face has sky gradient + sun + atmospheric scattering
        let size = CUBEMAP_FACE_SIZE as usize;
        let light_dir = glam::Vec3::new(0.5, 0.8, 0.3).normalize();

        for face in 0u32..6 {
            let mut pixels: Vec<u8> = Vec::with_capacity(size * size * 4);

            for y in 0..size {
                for x in 0..size {
                    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

                    let dir = match face {
                        0 => glam::Vec3::new(1.0, -v, -u),
                        1 => glam::Vec3::new(-1.0, -v, u),
                        2 => glam::Vec3::new(u, 1.0, v),
                        3 => glam::Vec3::new(u, -1.0, -v),
                        4 => glam::Vec3::new(u, -v, 1.0),
                        _ => glam::Vec3::new(-u, -v, -1.0),
                    }
                    .normalize();

                    let up = dir.y;
                    let deep = glam::Vec3::new(0.01, 0.02, 0.06);
                    let horizon = glam::Vec3::new(0.25, 0.40, 0.60);
                    let zenith = glam::Vec3::new(0.10, 0.25, 0.55);

                    let mut color = if up < 0.0 {
                        let t = (-up).clamp(0.0, 1.0);
                        horizon * 0.3 * (1.0 - t * t) + deep * (t * t)
                    } else {
                        let t = up.clamp(0.0, 1.0);
                        horizon * (1.0 - t.sqrt()) + zenith * t.sqrt()
                    };

                    // Atmospheric scattering near sun
                    let sun_dot = dir.dot(light_dir).max(0.0);
                    color += glam::Vec3::new(1.0, 0.85, 0.6) * sun_dot.powf(8.0) * 0.4;

                    // Horizon glow
                    let glow = (-up.abs() * 6.0).exp() * 0.2;
                    let glow_tint = glam::Vec3::new(0.5, 0.6, 0.8)
                        .lerp(glam::Vec3::new(1.0, 0.8, 0.5), sun_dot * sun_dot);
                    color += glow_tint * glow;

                    // Sun disc
                    let sun_radius: f32 = 0.05;
                    let disc = ((sun_dot - (sun_radius * 1.5).cos())
                        / ((sun_radius * 0.5).cos() - (sun_radius * 1.5).cos()))
                    .clamp(0.0, 1.0);
                    color += glam::Vec3::new(1.0, 0.95, 0.8) * disc * 3.0;

                    // Corona
                    let corona = (-(1.0 - sun_dot).max(0.0) * 20.0).exp() * 0.6;
                    color += glam::Vec3::new(1.0, 0.9, 0.7) * corona;

                    // Reinhard tone mapping
                    color = color / (color + glam::Vec3::ONE);

                    // BGRA order for Bgra8UnormSrgb, with sRGB gamma
                    let r = (color.x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                    let g = (color.y.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                    let b = (color.z.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                    pixels.push(b);
                    pixels.push(g);
                    pixels.push(r);
                    pixels.push(255u8);
                }
            }

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.cubemap_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((size as u32) * 4),
                    rows_per_image: Some(size as u32),
                },
                wgpu::Extent3d {
                    width: size as u32,
                    height: size as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.cubemap_generated = true;
    }

    /// Create a texture view for a single cubemap face (for rendering into it).
    pub fn cubemap_face_view(&self, face: u32) -> wgpu::TextureView {
        self.cubemap_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Cubemap Face {} View", face)),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: face,
                array_layer_count: Some(1),
                ..Default::default()
            })
    }

    /// Get the cubemap face resolution.
    pub fn cubemap_face_size(&self) -> u32 {
        CUBEMAP_FACE_SIZE
    }

    /// Mark the cubemap as needing recapture (e.g. after scene changes).
    pub fn invalidate_cubemap(&mut self) {
        self.cubemap_generated = false;
    }

    /// Mark the cubemap as already generated (prevents fallback fill from overwriting).
    pub fn mark_cubemap_generated(&mut self) {
        self.cubemap_generated = true;
    }

    /// Set initial render params from editor state (called once during initialization)
    pub fn set_initial_params(
        &self,
        queue: &wgpu::Queue,
        editor_state: &crate::ui::panel_context::GenomeEditorState,
    ) {
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
            time: 0.0,
            wave_height: editor_state.fluid_wave_height,
            wave_speed: editor_state.fluid_wave_speed,
            noise_scale: editor_state.fluid_noise_scale,
            noise_octaves: editor_state.fluid_noise_octaves as f32,
            noise_lacunarity: editor_state.fluid_noise_lacunarity,
            noise_persistence: editor_state.fluid_noise_persistence,
            reflection_brightness: 10.0,
            light_dir: editor_state.light_dir,
            waterline_alpha: editor_state.fluid_waterline_alpha,
            gravity: [0.0, 1.0, 0.0],
            gravity_mode: 1,
        };
        self.update_render_params(queue, &params);
    }

    /// Update params buffer
    fn update_params(&self, queue: &wgpu::Queue) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        // Shift origin back by 1 cell for padding
        let padded_origin = grid_origin - Vec3::splat(cell_size);

        let params = SurfaceNetsGpuParams {
            grid_resolution: PADDED_RESOLUTION,
            iso_level: self.iso_level,
            cell_size,
            max_vertices: self.max_vertices,
            grid_origin: padded_origin.to_array(),
            max_indices: self.max_indices,
            density_resolution: GRID_RESOLUTION,
            use_fast_early_out: 0,
            _pad_b: 0,
            _pad_c: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Run surface nets extraction on GPU
    pub fn extract_mesh(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_count = (PADDED_RESOLUTION + 3) / 4;

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

    /// Extract the ice mesh from the ice density field. Reuses the water
    /// surface-nets compute pipelines with the ice bind group; no smoothing
    /// pass (the extractor's internal height filter plus flat shading gives
    /// the large faceted look).
    pub fn extract_ice_mesh(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_count = (PADDED_RESOLUTION + 3) / 4;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ice Reset Counters Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reset_pipeline);
            pass.set_bind_group(0, &self.ice_compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ice Generate Vertices Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vertex_pipeline);
            pass.set_bind_group(0, &self.ice_compute_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ice Generate Indices Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.index_pipeline);
            pass.set_bind_group(0, &self.ice_compute_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, workgroup_count, workgroup_count);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ice Finalize Indirect Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, &self.ice_compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // No counter readback: the indirect draw needs no CPU involvement.
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

    /// Render the extracted mesh using WBOIT (Weighted Blended Order-Independent Transparency).
    /// Pass 1: Render water into accumulation + revealage textures.
    /// Pass 2: Composite the OIT result over the scene.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
    ) {
        // Generate cubemap on first render
        self.generate_cubemap(queue);

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
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );

        // === Pass 1: WBOIT accumulation into offscreen targets ===
        {
            let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Surface Nets WBOIT Pass"),
                color_attachments: &[
                    // Target 0: Accumulation - clear to (0,0,0,0)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.oit_accum_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    // Target 1: Revealage - clear to 1.0 (fully transparent = no fragments yet)
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.oit_revealage_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                // Use existing depth buffer (read-only) to occlude water behind opaque geometry
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

            oit_pass.set_pipeline(&self.render_pipeline);
            oit_pass.set_bind_group(0, &self.render_bind_group, &[]);
            if let Some(ref shadow_bg) = self.shadow_bind_group {
                oit_pass.set_bind_group(1, shadow_bg, &[]);
            } else {
                return; // Shadow bind group not yet ready - skip draw to avoid validation error
            }
            oit_pass.set_bind_group(2, &self.cubemap_bind_group, &[]);
            oit_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            oit_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            oit_pass.draw_indexed_indirect(&self.indirect_draw_buffer, 0);

            // Ice mesh: same OIT pass, its own faceted pipeline and its own
            // appearance params at group(0) - weighted blending composites
            // ice and water order-independently.
            oit_pass.set_pipeline(&self.ice_render_pipeline);
            oit_pass.set_bind_group(0, &self.ice_render_bind_group, &[]);
            oit_pass.set_vertex_buffer(0, self.ice_vertex_buffer.slice(..));
            oit_pass.set_index_buffer(self.ice_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            oit_pass.draw_indexed_indirect(&self.ice_indirect_draw_buffer, 0);
        }

        // === Pass 2: Composite OIT result over the scene ===
        {
            let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("OIT Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
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

            composite_pass.set_pipeline(&self.oit_composite_pipeline);
            composite_pass.set_bind_group(0, &self.oit_composite_bind_group, &[]);
            composite_pass.draw(0..3, 0..1); // Fullscreen triangle
        }
    }

    /// Resize for new screen dimensions
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Recreate OIT textures at new resolution
        let (accum_tex, accum_view) = Self::create_oit_accum_texture(device, width, height);
        let (reveal_tex, reveal_view) = Self::create_oit_revealage_texture(device, width, height);
        self.oit_accum_texture = accum_tex;
        self.oit_accum_view = accum_view;
        self.oit_revealage_texture = reveal_tex;
        self.oit_revealage_view = reveal_view;

        // Recreate composite bind group with new texture views
        self.oit_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OIT Composite Bind Group"),
            layout: &self.oit_composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.oit_accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.oit_revealage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.oit_sampler),
                },
            ],
        });
    }

    fn create_oit_accum_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("OIT Accumulation Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_oit_revealage_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("OIT Revealage Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    /// Get triangle count
    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
}

/// Generate test density data - simple sphere
pub fn generate_test_density_sphere(
    center: Vec3,
    radius: f32,
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
                let world_pos = grid_origin
                    + Vec3::new(
                        (x as f32 + 0.5) * cell_size,
                        (y as f32 + 0.5) * cell_size,
                        (z as f32 + 0.5) * cell_size,
                    );

                let dist = (world_pos - center).length();
                let d = 1.0 - (dist / radius).clamp(0.0, 2.0);

                let idx =
                    (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
                density[idx] = d.max(0.0);
            }
        }
    }

    density
}

/// Generate test density data - metaballs
pub fn generate_test_density_metaballs(
    balls: &[(Vec3, f32)], // (center, radius)
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
                let world_pos = grid_origin
                    + Vec3::new(
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

                let idx =
                    (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
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
                let world_pos = grid_origin
                    + Vec3::new(
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

                let idx =
                    (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize;
                density[idx] = (base + noise * noise_scale * base).max(0.0);
            }
        }
    }

    density
}
