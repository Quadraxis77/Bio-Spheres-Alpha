//! Organism Skin Renderer
//!
//! Renders a smooth isosurface "skin" around connected cell clusters.
//!
//! ## Pipeline
//!
//! ```text
//! Cell positions (GPU triple buffer)
//!        │
//!        ▼
//! [clear_density]   ── zeros atomic i32 accumulation buffer (64³)
//!        │
//!        ▼
//! [generate_density] ── per-cell: metaball contribution → atomicAdd into buffer
//!        │
//!        ▼
//! [normalize_density] ── i32 fixed-point → f32 density [0..N]
//!        │
//!        ▼
//! [smooth_density] ── 3×3×3 spatial blur + temporal EMA (stabilises surface)
//!        │
//!        ▼
//! [surface_nets (4 passes)] ── extracts isosurface mesh
//!        │
//!        ▼
//! [render pass] ── organic skin shading
//! ```

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

// Organism density grid resolution
pub const ORGANISM_GRID_RES: u32 = 256;  // Doubled from 128
const ORGANISM_PADDED_RES: u32 = ORGANISM_GRID_RES + 2; // 258

// ─────────────────────────────────────────────────────────────────────────────
// GPU data types (must match shader structs)
// ─────────────────────────────────────────────────────────────────────────────

/// Organism density compute params (must match OrganismDensityParams in WGSL).
/// Layout: vec3<f32> (12) + f32 (4) + 4×u32/f32 (16) = 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OrganismDensityParams {
    pub grid_origin: [f32; 3],
    pub cell_size: f32,
    pub grid_resolution: u32,
    pub skin_radius_scale: f32,
    pub max_cells: u32,
    pub _pad: u32,
}

/// Temporal smoothing params (must match SmoothParams in smooth_density.wgsl).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SmoothDensityParams {
    grid_resolution: u32,
    blend_factor: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Surface nets params (must match SurfaceNetsParams in surface_nets_gpu.wgsl).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SurfaceNetsParams {
    grid_resolution: u32,
    iso_level: f32,
    cell_size: f32,
    max_vertices: u32,
    grid_origin: [f32; 3],
    max_indices: u32,
    density_resolution: u32,
    use_fast_early_out: u32,
    _pad_b: u32,
    _pad_c: u32,
}

/// Camera uniform (must match CameraUniform in organism_skin.wgsl).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Organism skin material params (must match SkinParams in organism_skin.wgsl).
/// 64 bytes = 4 × vec4.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OrganismSkinParams {
    // vec4 0 — base colour + ambient
    pub base_r: f32, pub base_g: f32, pub base_b: f32, pub ambient: f32,
    // vec4 1 — diffuse/specular
    pub diffuse: f32, pub specular: f32, pub shininess: f32, pub fresnel: f32,
    // vec4 2 — fresnel power, alpha, time, sss strength
    pub fresnel_power: f32, pub alpha: f32, pub time: f32, pub sss_strength: f32,
    // vec4 3 — sss colour + rim
    pub sss_r: f32, pub sss_g: f32, pub sss_b: f32, pub rim_strength: f32,
    // vec4 4 — light direction (world space, pointing toward light) + padding
    pub light_dir_x: f32, pub light_dir_y: f32, pub light_dir_z: f32, pub _pad: f32,
}

impl Default for OrganismSkinParams {
    fn default() -> Self {
        Self {
            // Warm pinkish-amber membrane colour
            base_r: 0.85, base_g: 0.55, base_b: 0.35,
            ambient: 0.12,
            diffuse: 0.6, specular: 0.5, shininess: 48.0, fresnel: 0.08,
            fresnel_power: 3.0,
            alpha: 0.55,
            time: 0.0,
            sss_strength: 0.5,
            // Warm orange SSS bleed (light transmission)
            sss_r: 1.0, sss_g: 0.4, sss_b: 0.1,
            rim_strength: 0.35,
            // Default light direction (toward light, matches scene default)
            light_dir_x: 0.4, light_dir_y: 0.8, light_dir_z: 0.4,
            _pad: 0.0,
        }
    }
}

/// Vertex counter struct (must match surface_nets_gpu.wgsl Counter)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Counters {
    vertex_count: u32,
    index_count: u32,
}

/// Vertex format produced by surface nets (must match Vertex in surface_nets_gpu.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    fluid_type: f32, // unused for skin, kept for buffer format compatibility
    normal: [f32; 3],
    _pad1: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// OrganismSkinRenderer
// ─────────────────────────────────────────────────────────────────────────────

/// GPU renderer that wraps cells in a smooth organic skin via surface nets.
pub struct OrganismSkinRenderer {
    // ── Density generation ──────────────────────────────────────────────────
    clear_pipeline: wgpu::ComputePipeline,
    generate_pipeline: wgpu::ComputePipeline,
    normalize_pipeline: wgpu::ComputePipeline,
    /// Layout for the density bind group (bindings: params, pos_mass, death_flags, cell_count, accum, density)
    pub density_bind_group_layout: wgpu::BindGroupLayout,

    // Own density buffers
    density_accum_buffer: wgpu::Buffer,   // atomic i32 per voxel
    organism_density_buffer: wgpu::Buffer, // f32 per voxel → fed into surface nets

    // Density params (world geometry + skin_radius_scale)
    density_params_buffer: wgpu::Buffer,

    // ── Surface nets ────────────────────────────────────────────────────────
    sn_reset_pipeline: wgpu::ComputePipeline,
    sn_vertex_pipeline: wgpu::ComputePipeline,
    sn_index_pipeline: wgpu::ComputePipeline,
    sn_finalize_pipeline: wgpu::ComputePipeline,
    sn_compute_bind_group: wgpu::BindGroup,

    // Dummy buffers for the fluid_types / solid_mask bindings in surface_nets_gpu.wgsl
    #[allow(dead_code)]
    dummy_fluid_types: wgpu::Buffer,
    #[allow(dead_code)]
    dummy_solid_mask: wgpu::Buffer,

    sn_params_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    vertex_map_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    counter_staging_buffer: wgpu::Buffer,
    indirect_draw_buffer: wgpu::Buffer,

    // ── Temporal smoothing ───────────────────────────────────────────────────
    smooth_pipeline: wgpu::ComputePipeline,
    smooth_bind_group: wgpu::BindGroup,
    smoothed_density_buffer: wgpu::Buffer, // persistent EMA state
    smooth_temp_buffer: wgpu::Buffer,      // current-frame output before copy-back
    #[allow(dead_code)]
    smooth_params_buffer: wgpu::Buffer,
    pub smooth_blend_factor: f32,

    // ── Render ───────────────────────────────────────────────────────────────
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    skin_params_buffer: wgpu::Buffer,

    // ── Config ────────────────────────────────────────────────────────────────
    world_radius: f32,
    world_center: Vec3,
    max_vertices: u32,
    max_indices: u32,
    iso_level: f32,

    /// Scale factor applied to each cell's radius to determine skin influence thickness.
    pub skin_radius_scale: f32,

    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub skin_params: OrganismSkinParams,
    
    /// Total number of voxels in the density grid
    pub total_voxels: u32,
    
    /// Grid resolution (for shader calculations)
    pub grid_resolution: u32,
}

impl OrganismSkinRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        world_radius: f32,
        world_center: Vec3,
        capacity: u32, // max cell count for dispatch sizing
        width: u32,
        height: u32,
        settings: &crate::ui::OrganismSkinSettings,
    ) -> Self {
        let max_vertices: u32 = 2_000_000;  // Increased for higher resolution
        let max_indices: u32 = 6_000_000;   // Increased for higher resolution
        let iso_level: f32 = settings.iso_level;
        let skin_radius_scale: f32 = settings.radius_scale;
        let grid_resolution: u32 = settings.grid_resolution;

        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / grid_resolution as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);
        
        // Calculate voxel counts based on dynamic resolution
        let total_voxels = (grid_resolution * grid_resolution * grid_resolution) as usize;
        let padded_resolution = grid_resolution + 2; // Only for surface nets processing
        let padded_voxels = (padded_resolution * padded_resolution * padded_resolution) as usize;

        // ── Density compute shader ──────────────────────────────────────────
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Density Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_density.wgsl").into()
            ),
        });

        // ── Surface nets compute shader (shared with fluid surface nets) ────
        let sn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Surface Nets Compute"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/surface_nets_gpu.wgsl").into()
            ),
        });

        // ── Organism skin render shader ─────────────────────────────────────
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Skin Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_skin.wgsl").into()
            ),
        });

        // ── Density buffers ─────────────────────────────────────────────────
        // Use original resolution like water surface nets (not padded)
        let density_accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Density Accum (i32)"),
            size: (total_voxels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let organism_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Density (f32)"),
            size: (total_voxels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let density_params = OrganismDensityParams {
            grid_origin: grid_origin.to_array(),
            cell_size,
            grid_resolution,
            skin_radius_scale,
            max_cells: capacity,
            _pad: 0,
        };
        let density_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Density Params"),
            contents: bytemuck::bytes_of(&density_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Density bind group layout ────────────────────────────────────────
        //  binding 0: uniform  OrganismDensityParams
        //  binding 1: storage  position_and_mass  (read)  ← external, per-frame
        //  binding 2: storage  death_flags        (read)  ← external
        //  binding 3: storage  cell_count         (read)  ← external
        //  binding 4: storage  density_accum      (read_write, atomic)
        //  binding 5: storage  organism_density   (read_write)
        let density_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Organism Density Layout"),
                entries: &[
                    // 0: params uniform
                    bgl_entry(0, wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false, min_binding_size: None,
                        }),
                    // 1: position_and_mass (external)
                    bgl_entry(1, wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }),
                    // 2: death_flags (external)
                    bgl_entry(2, wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }),
                    // 3: cell_count (external)
                    bgl_entry(3, wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }),
                    // 4: density_accum (atomic i32, internal)
                    bgl_entry(4, wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false, min_binding_size: None,
                        }),
                    // 5: organism_density (f32 output, internal)
                    bgl_entry(5, wgpu::ShaderStages::COMPUTE,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false, min_binding_size: None,
                        }),
                ],
            }
        );

        // ── Density compute pipelines ────────────────────────────────────────
        let density_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism Density Pipeline Layout"),
            bind_group_layouts: &[&density_bind_group_layout],
            push_constant_ranges: &[],
        });

        let clear_pipeline = make_compute_pipeline(device, &density_pipeline_layout,
            &density_shader, "clear_density", "Organism Clear Density");
        let generate_pipeline = make_compute_pipeline(device, &density_pipeline_layout,
            &density_shader, "generate_density", "Organism Generate Density");
        let normalize_pipeline = make_compute_pipeline(device, &density_pipeline_layout,
            &density_shader, "normalize_density", "Organism Normalize Density");

        // ── Surface nets buffers ─────────────────────────────────────────────
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Vertex Buffer"),
            size: (max_vertices as usize * std::mem::size_of::<GpuVertex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Index Buffer"),
            size: (max_indices as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
            mapped_at_creation: false,
        });
        let vertex_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Vertex Map"),
            size: (padded_voxels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Counter"),
            size: std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Counter Staging"),
            size: std::mem::size_of::<Counters>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect_draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Indirect Draw"),
            size: 20, // 5 × u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // Create dummy buffers for bindings 2 (fluid_types) and 3 (solid_mask)
        // These are required by the surface nets shader but not used for organism rendering
        let dummy_size = (total_voxels * 4) as u64;
        let dummy_fluid_types = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism SN Dummy Fluid Types"),
            size: dummy_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dummy_solid_mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism SN Dummy Solid Mask"),
            size: dummy_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // SN params — use the padded resolution with density_resolution = grid_resolution
        let padded_origin = grid_origin - Vec3::splat(cell_size); // shift for padding
        let sn_params = SurfaceNetsParams {
            grid_resolution: padded_resolution,
            iso_level,
            cell_size,
            max_vertices,
            grid_origin: padded_origin.to_array(),
            max_indices,
            density_resolution: grid_resolution,
            use_fast_early_out: 1, _pad_b: 0, _pad_c: 0,
        };
        let sn_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism SN Params"),
            contents: bytemuck::bytes_of(&sn_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Surface nets compute bind group layout ────────────────────────────
        // Matches surface_nets_gpu.wgsl group(0) bindings 0-8
        let sn_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Organism SN Compute Layout"),
            entries: &[
                bgl_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(1, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(2, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(3, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(4, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(5, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(6, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(7, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(8, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
            ],
        });

        let sn_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism SN Compute Bind Group"),
            layout: &sn_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sn_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: organism_density_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dummy_fluid_types.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dummy_solid_mask.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: vertex_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: index_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: vertex_map_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: counter_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: indirect_draw_buffer.as_entire_binding() },
            ],
        });

        let sn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism SN Compute Pipeline Layout"),
            bind_group_layouts: &[&sn_layout],
            push_constant_ranges: &[],
        });

        let sn_reset_pipeline  = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "reset_counters",   "Organism SN Reset");
        let sn_vertex_pipeline = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "generate_vertices","Organism SN Vertices");
        let sn_index_pipeline  = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "generate_indices", "Organism SN Indices");
        let sn_finalize_pipeline = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "finalize_indirect","Organism SN Finalize");

        // ── Render pipeline ───────────────────────────────────────────────────
        let camera_uniform = CameraUniform {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3], _padding: 0.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Skin Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let skin_params = OrganismSkinParams::default();
        let skin_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Skin Params Buffer"),
            contents: bytemuck::bytes_of(&skin_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Organism Skin Render Layout"),
            entries: &[
                bgl_entry(0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(1,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }),
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism Skin Render Bind Group"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: skin_params_buffer.as_entire_binding() },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism Skin Render Pipeline Layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Organism Skin Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,  shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3, // position
                        },
                        wgpu::VertexAttribute {
                            offset: 12, shader_location: 1,
                            format: wgpu::VertexFormat::Float32, // fluid_type (unused)
                        },
                        wgpu::VertexAttribute {
                            offset: 16, shader_location: 2,
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false, // Transparent — no depth writes
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Temporal smoothing ───────────────────────────────────────────────
        let smooth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Smooth Density"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fluid/smooth_density.wgsl").into()
            ),
        });

        let density_buf_size = (total_voxels * 4) as u64; // Use original resolution like water surface nets

        let smoothed_density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Smoothed Density"),
            size: density_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let smooth_temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Smooth Temp"),
            size: density_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let smooth_blend_factor = 0.15_f32; // Match water surface nets stability
        let smooth_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Smooth Params"),
            contents: bytemuck::bytes_of(&SmoothDensityParams {
                grid_resolution, // Use original resolution like water surface nets
                blend_factor: smooth_blend_factor,
                _pad0: 0.0,
                _pad1: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let smooth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Organism Smooth Layout"),
            entries: &[
                bgl_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(1, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(2, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(3, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
            ],
        });
        let smooth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism Smooth Bind Group"),
            layout: &smooth_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: smooth_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: organism_density_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: smoothed_density_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: smooth_temp_buffer.as_entire_binding() },
            ],
        });
        let smooth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism Smooth Pipeline Layout"),
            bind_group_layouts: &[&smooth_layout],
            push_constant_ranges: &[],
        });
        let smooth_pipeline = make_compute_pipeline(
            device, &smooth_pipeline_layout, &smooth_shader, "smooth_density", "Organism Smooth Density");

        Self {
            clear_pipeline,
            generate_pipeline,
            normalize_pipeline,
            density_bind_group_layout,
            density_accum_buffer,
            organism_density_buffer,
            density_params_buffer,
            sn_reset_pipeline,
            sn_vertex_pipeline,
            sn_index_pipeline,
            sn_finalize_pipeline,
            sn_compute_bind_group,
            dummy_fluid_types,
            dummy_solid_mask,
            sn_params_buffer,
            vertex_buffer,
            index_buffer,
            vertex_map_buffer,
            counter_buffer,
            counter_staging_buffer,
            indirect_draw_buffer,
            smooth_pipeline,
            smooth_bind_group,
            smoothed_density_buffer,
            smooth_temp_buffer,
            smooth_params_buffer,
            smooth_blend_factor,
            render_pipeline,
            render_bind_group,
            camera_buffer,
            skin_params_buffer,
            world_radius,
            world_center,
            max_vertices,
            max_indices,
            iso_level,
            skin_radius_scale,
            width,
            height,
            vertex_count: 0,
            index_count: 0,
            skin_params,
            total_voxels: total_voxels as u32, // Use original resolution like water surface nets
            grid_resolution,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Create a per-frame density bind group that binds the (triple-buffered) position buffer
    /// along with the shared death_flags and cell_count buffers.
    ///
    /// Call this once per unique combination of buffers (e.g. once per triple-buffer index) and
    /// cache the results.
    pub fn create_density_bind_group(
        &self,
        device: &wgpu::Device,
        position_and_mass: &wgpu::Buffer,
        death_flags: &wgpu::Buffer,
        cell_count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism Density Bind Group"),
            layout: &self.density_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.density_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: position_and_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: death_flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.density_accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.organism_density_buffer.as_entire_binding() },
            ],
        })
    }

    /// GPU pass: generate organism density from cell positions.
    /// Must be called before `extract_mesh`.
    pub fn generate_density(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        density_bind_group: &wgpu::BindGroup,
        max_cells: u32,
    ) {
        // Early exit if no cells to process
        if max_cells == 0 {
            return;
        }

        // Pass 1: clear atomic accumulators (clear original resolution)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism Clear Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            // Dispatch for original buffer total (not padded)
            let workgroups_x = ((self.total_voxels + 255) / 256).min(65535);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        // Pass 2: per-cell metaball contributions
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism Generate Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            pass.dispatch_workgroups((max_cells + 63) / 64, 1, 1);
        }

        // Pass 3: i32 → f32 normalisation (use original resolution)
        {
            let wg = (self.grid_resolution + 3) / 4;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism Normalize Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.normalize_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
    }

    /// GPU pass: spatial blur + temporal EMA on the density field.
    /// Call after `generate_density` and before `extract_mesh`.
    /// Copies the smoothed result back into `organism_density_buffer` so surface nets
    /// reads the stabilised field instead of the raw per-frame data.
    pub fn smooth_density(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg = (self.grid_resolution + 3) / 4;
        // Use original resolution buffer size like water surface nets
        let buf_size = (self.total_voxels * 4) as u64;

        // Clear smoothed density buffer at start to prevent garbage data on first frame
        encoder.clear_buffer(&self.smoothed_density_buffer, 0, None);

        // smooth pass: raw density + prev EMA → smooth_temp
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism Smooth Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.smooth_pipeline);
            pass.set_bind_group(0, &self.smooth_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
        
        // smooth_temp → smoothed_density  (persist EMA state for next frame)
        encoder.copy_buffer_to_buffer(&self.smooth_temp_buffer, 0, &self.smoothed_density_buffer, 0, buf_size);
        // smoothed_density → organism_density  (surface nets reads smoothed data)
        encoder.copy_buffer_to_buffer(&self.smoothed_density_buffer, 0, &self.organism_density_buffer, 0, buf_size);
    }

    /// GPU pass: extract isosurface mesh from organism density.
    /// Must be called after `generate_density`.
    pub fn extract_mesh(&self, encoder: &mut wgpu::CommandEncoder) {
        let padded_res = self.grid_resolution + 2;
        let wg = (padded_res + 3) / 4;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism SN Reset"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_reset_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism SN Vertices"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_vertex_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism SN Indices"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_index_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Organism SN Finalize"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_finalize_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.counter_buffer, 0,
            &self.counter_staging_buffer, 0,
            std::mem::size_of::<Counters>() as u64,
        );
    }

    /// Render the extracted organism skin mesh.
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
        let cam = CameraUniform {
            view_proj: (proj * view).to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam));

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Organism Skin Render Pass"),
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

    /// Update the skin material parameters on the GPU.
    pub fn update_skin_params(&mut self, queue: &wgpu::Queue, params: OrganismSkinParams) {
        self.skin_params = params;
        queue.write_buffer(&self.skin_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Update skin_radius_scale and upload new density params.
    pub fn set_skin_radius_scale(&mut self, queue: &wgpu::Queue, scale: f32) {
        self.skin_radius_scale = scale;
        self.upload_density_params(queue);
    }

    /// Update the iso level and upload new surface nets params.
    pub fn set_iso_level(&mut self, queue: &wgpu::Queue, level: f32) {
        self.iso_level = level;
        self.upload_sn_params(queue);
    }

    /// Update the time uniform in skin params (call every frame for animation).
    pub fn set_time(&self, queue: &wgpu::Queue, time: f32) {
        // Write only the `time` field (offset 40 bytes into the struct: 10 f32s × 4)
        queue.write_buffer(&self.skin_params_buffer, 40, bytemuck::bytes_of(&time));
    }

    /// Resize for new screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Optionally read back mesh counts after `extract_mesh` (for UI stats).
    pub fn try_read_counts(&mut self, device: &wgpu::Device) {
        let slice = self.counter_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        {
            let data = slice.get_mapped_range();
            let c: &Counters = bytemuck::from_bytes(&data);
            self.vertex_count = c.vertex_count.min(self.max_vertices);
            self.index_count  = c.index_count.min(self.max_indices);
        }
        self.counter_staging_buffer.unmap();
    }

    /// Triangle count in the last extracted mesh.
    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn upload_density_params(&self, queue: &wgpu::Queue) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / ORGANISM_GRID_RES as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        let p = OrganismDensityParams {
            grid_origin: grid_origin.to_array(),
            cell_size,
            grid_resolution: ORGANISM_GRID_RES,
            skin_radius_scale: self.skin_radius_scale,
            max_cells: 0, // not used for upload; max_cells set at creation
            _pad: 0,
        };
        queue.write_buffer(&self.density_params_buffer, 0, bytemuck::bytes_of(&p));
    }

    fn upload_sn_params(&self, queue: &wgpu::Queue) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / ORGANISM_GRID_RES as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        let padded_origin = grid_origin - Vec3::splat(cell_size);
        let p = SurfaceNetsParams {
            grid_resolution: ORGANISM_PADDED_RES,
            iso_level: self.iso_level,
            cell_size,
            max_vertices: self.max_vertices,
            grid_origin: padded_origin.to_array(),
            max_indices: self.max_indices,
            density_resolution: ORGANISM_GRID_RES,
            use_fast_early_out: 1, _pad_b: 0, _pad_c: 0,
        };
        queue.write_buffer(&self.sn_params_buffer, 0, bytemuck::bytes_of(&p));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers to reduce boilerplate
// ─────────────────────────────────────────────────────────────────────────────

fn bgl_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None }
}

fn make_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
