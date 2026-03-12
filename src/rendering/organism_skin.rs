//! Organism Skin Renderer — Per-Organism Overlapping Skins
//!
//! Each organism gets its own independent, non-merged isosurface topology.
//! Uses K=2 density slots per voxel so two organisms can overlap; a third is dropped.
//!
//! ## Pipeline
//!
//! ```text
//! Cell positions + organism labels (GPU triple buffer)
//!   → [clear_histogram + count_organisms + assign_skin_ids]  (organism_skin_count.wgsl)
//!   → [clear_density + generate_density + normalize_density]  (organism_skin_density.wgsl)
//!   → [reset + generate_vertices + generate_indices + finalize] (organism_skin_surface_nets.wgsl)
//!   → [render pass with per-organism coloring]                 (organism_skin_render.wgsl)
//! ```

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Organism density grid resolution
pub const ORGANISM_GRID_RES: u32 = 256;

/// Histogram size for organism ID hashing (max 65536 unique organisms)
const HISTOGRAM_SIZE: u32 = 65536;

/// Minimum cells for an organism to get a skin
const MIN_CELLS_FOR_SKIN: u32 = 4;

// ─────────────────────────────────────────────────────────────────────────────
// GPU data types (must match shader structs)
// ─────────────────────────────────────────────────────────────────────────────

/// Matches OrganismDensityParams in organism_skin_density.wgsl (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OrganismDensityParams {
    pub grid_origin: [f32; 3],
    pub cell_size: f32,
    pub grid_resolution: u32,
    pub skin_radius_scale: f32,
    pub max_cells: u32,
    pub min_cells_for_skin: u32,
}

/// Matches SkinCountParams in organism_skin_count.wgsl (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SkinCountParams {
    min_cells: u32,
    histogram_size: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Matches SurfaceNetsParams in organism_skin_surface_nets.wgsl (48 bytes)
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
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
}

/// Camera uniform (matches CameraUniform in organism_skin_render.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Organism skin material params (matches SkinParams in organism_skin_render.wgsl, 80 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OrganismSkinParams {
    pub base_r: f32, pub base_g: f32, pub base_b: f32, pub ambient: f32,
    pub diffuse: f32, pub specular: f32, pub shininess: f32, pub fresnel: f32,
    pub fresnel_power: f32, pub alpha: f32, pub time: f32, pub sss_strength: f32,
    pub sss_r: f32, pub sss_g: f32, pub sss_b: f32, pub rim_strength: f32,
    pub light_dir_x: f32, pub light_dir_y: f32, pub light_dir_z: f32, pub _pad: f32,
}

impl Default for OrganismSkinParams {
    fn default() -> Self {
        Self {
            base_r: 0.85, base_g: 0.55, base_b: 0.35,
            ambient: 0.12,
            diffuse: 0.6, specular: 0.5, shininess: 48.0, fresnel: 0.08,
            fresnel_power: 3.0, alpha: 0.55, time: 0.0, sss_strength: 0.5,
            sss_r: 1.0, sss_g: 0.4, sss_b: 0.1, rim_strength: 0.35,
            light_dir_x: 0.4, light_dir_y: 0.8, light_dir_z: 0.4, _pad: 0.0,
        }
    }
}

/// Vertex counter (matches Counter in organism_skin_surface_nets.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Counters {
    vertex_count: u32,
    index_count: u32,
}

/// Vertex format (matches Vertex in organism_skin_surface_nets.wgsl)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    organism_id: f32,
    normal: [f32; 3],
    _pad1: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// OrganismSkinRenderer
// ─────────────────────────────────────────────────────────────────────────────

pub struct OrganismSkinRenderer {
    // ── Organism counting (organism_skin_count.wgsl) ────────────────────────
    count_clear_pipeline: wgpu::ComputePipeline,
    count_organisms_pipeline: wgpu::ComputePipeline,
    assign_skin_ids_pipeline: wgpu::ComputePipeline,
    pub count_bind_group_layout: wgpu::BindGroupLayout,
    count_params_buffer: wgpu::Buffer,
    histogram_buffer: wgpu::Buffer,
    cell_skin_id_buffer: wgpu::Buffer,

    // ── K=2 density generation (organism_skin_density.wgsl) ─────────────────
    density_clear_pipeline: wgpu::ComputePipeline,
    density_generate_pipeline: wgpu::ComputePipeline,
    density_normalize_pipeline: wgpu::ComputePipeline,
    pub density_bind_group_layout: wgpu::BindGroupLayout,
    density_params_buffer: wgpu::Buffer,
    // K=2 slot buffers — organism IDs (atomic u32)
    slot_org_0: wgpu::Buffer,
    slot_org_1: wgpu::Buffer,
    // K=2 slot buffers — density accumulators (atomic i32)
    slot_density_0: wgpu::Buffer,
    slot_density_1: wgpu::Buffer,
    // K=2 normalized output (f32 density + u32 org_id)
    density_out_0: wgpu::Buffer,
    density_out_1: wgpu::Buffer,
    org_id_out_0: wgpu::Buffer,
    org_id_out_1: wgpu::Buffer,

    // ── Organism-aware surface nets (organism_skin_surface_nets.wgsl) ────────
    sn_reset_pipeline: wgpu::ComputePipeline,
    sn_vertex_pipeline: wgpu::ComputePipeline,
    sn_index_pipeline: wgpu::ComputePipeline,
    sn_finalize_pipeline: wgpu::ComputePipeline,
    sn_compute_bind_group: wgpu::BindGroup,
    sn_params_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    vertex_map_0: wgpu::Buffer,
    #[allow(dead_code)]
    vertex_map_1: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    counter_staging_buffer: wgpu::Buffer,
    indirect_draw_buffer: wgpu::Buffer,

    // ── Render (organism_skin_render.wgsl) ───────────────────────────────────
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
    pub skin_radius_scale: f32,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub skin_params: OrganismSkinParams,
    pub total_voxels: u32,
    pub grid_resolution: u32,
}

impl OrganismSkinRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        world_radius: f32,
        world_center: Vec3,
        capacity: u32,
        width: u32,
        height: u32,
        settings: &crate::ui::OrganismSkinSettings,
    ) -> Self {
        let max_vertices: u32 = 4_000_000; // Doubled: 2 organisms can each produce vertices
        let max_indices: u32 = 12_000_000;
        let iso_level: f32 = settings.iso_level;
        let skin_radius_scale: f32 = settings.radius_scale;
        let grid_resolution: u32 = settings.grid_resolution;

        let world_diameter = world_radius * 2.0;
        let cell_size = world_diameter / grid_resolution as f32;
        let grid_origin = world_center - Vec3::splat(world_diameter / 2.0);

        let total_voxels = (grid_resolution * grid_resolution * grid_resolution) as usize;
        let padded_resolution = grid_resolution + 2;
        let padded_voxels = (padded_resolution * padded_resolution * padded_resolution) as usize;

        // ── Shader modules ──────────────────────────────────────────────────
        let count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Skin Count Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_skin_count.wgsl").into(),
            ),
        });
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Skin Density Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_skin_density.wgsl").into(),
            ),
        });
        let sn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Skin Surface Nets Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_skin_surface_nets.wgsl").into(),
            ),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism Skin Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_skin_render.wgsl").into(),
            ),
        });

        // ── Organism counting buffers ───────────────────────────────────────
        let count_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skin Count Params"),
            contents: bytemuck::bytes_of(&SkinCountParams {
                min_cells: MIN_CELLS_FOR_SKIN,
                histogram_size: HISTOGRAM_SIZE,
                _pad0: 0, _pad1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Histogram"),
            size: (HISTOGRAM_SIZE as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cell_skin_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Skin ID"),
            size: capacity as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Count bind group layout: params, cell_count, death_flags, label_buffer, histogram, cell_skin_id
        let count_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Skin Count Layout"),
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
                ],
            },
        );

        let count_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skin Count Pipeline Layout"),
            bind_group_layouts: &[&count_bind_group_layout],
            push_constant_ranges: &[],
        });
        let count_clear_pipeline = make_compute_pipeline(device, &count_pipeline_layout,
            &count_shader, "clear_histogram", "Skin Clear Histogram");
        let count_organisms_pipeline = make_compute_pipeline(device, &count_pipeline_layout,
            &count_shader, "count_organisms", "Skin Count Organisms");
        let assign_skin_ids_pipeline = make_compute_pipeline(device, &count_pipeline_layout,
            &count_shader, "assign_skin_ids", "Skin Assign IDs");

        // ── K=2 density buffers ─────────────────────────────────────────────
        let voxel_buf_size = (total_voxels * 4) as u64;

        let slot_org_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Slot Org 0"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let slot_org_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Slot Org 1"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let slot_density_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Slot Density 0"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let slot_density_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Slot Density 1"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let density_out_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Density Out 0"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let density_out_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Density Out 1"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let org_id_out_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Org ID Out 0"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let org_id_out_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin Org ID Out 1"), size: voxel_buf_size,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });

        let density_params = OrganismDensityParams {
            grid_origin: grid_origin.to_array(),
            cell_size,
            grid_resolution,
            skin_radius_scale,
            max_cells: capacity,
            min_cells_for_skin: MIN_CELLS_FOR_SKIN,
        };
        let density_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Skin Density Params"),
            contents: bytemuck::bytes_of(&density_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Density bind group layout — 13 bindings matching organism_skin_density.wgsl
        // 0: params (uniform), 1: position_and_mass (read), 2: death_flags (read),
        // 3: cell_count (read), 4-5: slot_org_0/1 (rw), 6-7: slot_density_0/1 (rw),
        // 8-9: density_out_0/1 (rw), 10-11: org_id_out_0/1 (rw), 12: cell_skin_id (read)
        let density_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Organism Skin Density Layout"),
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
                    // bindings 4-11: K=2 slot buffers (all read_write for atomics)
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
                    bgl_entry(9, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None }),
                    bgl_entry(10, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None }),
                    bgl_entry(11, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None }),
                    // binding 12: cell_skin_id (read-only)
                    bgl_entry(12, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None }),
                ],
            },
        );

        let density_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism Skin Density Pipeline Layout"),
            bind_group_layouts: &[&density_bind_group_layout],
            push_constant_ranges: &[],
        });
        let density_clear_pipeline = make_compute_pipeline(device, &density_pipeline_layout,
            &density_shader, "clear_density", "Skin Clear Density");
        let density_generate_pipeline = make_compute_pipeline(device, &density_pipeline_layout,
            &density_shader, "generate_density", "Skin Generate Density");
        let density_normalize_pipeline = make_compute_pipeline(device, &density_pipeline_layout,
            &density_shader, "normalize_density", "Skin Normalize Density");

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
        // Two vertex maps — one per organism slot at each voxel cell
        let vertex_map_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Vertex Map 0"),
            size: (padded_voxels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let vertex_map_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Skin Vertex Map 1"),
            size: (padded_voxels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
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

        let padded_origin = grid_origin - Vec3::splat(cell_size);
        let sn_params = SurfaceNetsParams {
            grid_resolution: padded_resolution,
            iso_level,
            cell_size,
            max_vertices,
            grid_origin: padded_origin.to_array(),
            max_indices,
            density_resolution: grid_resolution,
            _pad_a: 0, _pad_b: 0, _pad_c: 0,
        };
        let sn_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Organism Skin SN Params"),
            contents: bytemuck::bytes_of(&sn_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Surface nets bind group layout — 11 bindings matching organism_skin_surface_nets.wgsl
        // 0: params, 1-2: density_0/1 (read), 3-4: org_id_0/1 (read),
        // 5: vertices (rw), 6: indices (rw), 7-8: vertex_map_0/1 (rw),
        // 9: counters (rw), 10: indirect_draw (rw)
        let sn_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Organism Skin SN Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                bgl_entry(9, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(10, wgpu::ShaderStages::COMPUTE, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }),
            ],
        });

        let sn_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism Skin SN Bind Group"),
            layout: &sn_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sn_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: density_out_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: density_out_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: org_id_out_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: org_id_out_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: vertex_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: index_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: vertex_map_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: vertex_map_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: counter_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: indirect_draw_buffer.as_entire_binding() },
            ],
        });

        let sn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Organism Skin SN Pipeline Layout"),
            bind_group_layouts: &[&sn_layout],
            push_constant_ranges: &[],
        });
        let sn_reset_pipeline = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "reset_counters", "Skin SN Reset");
        let sn_vertex_pipeline = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "generate_vertices", "Skin SN Vertices");
        let sn_index_pipeline = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "generate_indices", "Skin SN Indices");
        let sn_finalize_pipeline = make_compute_pipeline(device, &sn_pipeline_layout,
            &sn_shader, "finalize_indirect", "Skin SN Finalize");

        // ── Render pipeline ─────────────────────────────────────────────────
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
                bgl_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None }),
                bgl_entry(1, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                            offset: 0, shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3, // position
                        },
                        wgpu::VertexAttribute {
                            offset: 12, shader_location: 1,
                            format: wgpu::VertexFormat::Float32, // organism_id
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            count_clear_pipeline,
            count_organisms_pipeline,
            assign_skin_ids_pipeline,
            count_bind_group_layout,
            count_params_buffer,
            histogram_buffer,
            cell_skin_id_buffer,
            density_clear_pipeline,
            density_generate_pipeline,
            density_normalize_pipeline,
            density_bind_group_layout,
            density_params_buffer,
            slot_org_0, slot_org_1,
            slot_density_0, slot_density_1,
            density_out_0, density_out_1,
            org_id_out_0, org_id_out_1,
            sn_reset_pipeline, sn_vertex_pipeline, sn_index_pipeline, sn_finalize_pipeline,
            sn_compute_bind_group,
            sn_params_buffer,
            vertex_buffer, index_buffer,
            vertex_map_0, vertex_map_1,
            counter_buffer, counter_staging_buffer, indirect_draw_buffer,
            render_pipeline, render_bind_group,
            camera_buffer, skin_params_buffer,
            world_radius, world_center,
            max_vertices, max_indices, iso_level,
            skin_radius_scale,
            width, height,
            vertex_count: 0, index_count: 0,
            skin_params,
            total_voxels: total_voxels as u32,
            grid_resolution,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Create a per-frame count bind group that binds the organism label buffer.
    /// Call once per unique combination of buffers and cache the results.
    pub fn create_count_bind_group(
        &self,
        device: &wgpu::Device,
        cell_count: &wgpu::Buffer,
        death_flags: &wgpu::Buffer,
        label_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism Skin Count Bind Group"),
            layout: &self.count_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.count_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: death_flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: label_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.cell_skin_id_buffer.as_entire_binding() },
            ],
        })
    }

    /// Create a per-frame density bind group that binds the triple-buffered position buffer.
    pub fn create_density_bind_group(
        &self,
        device: &wgpu::Device,
        position_and_mass: &wgpu::Buffer,
        death_flags: &wgpu::Buffer,
        cell_count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Organism Skin Density Bind Group"),
            layout: &self.density_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.density_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: position_and_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: death_flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.slot_org_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.slot_org_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.slot_density_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.slot_density_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.density_out_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.density_out_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.org_id_out_0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.org_id_out_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.cell_skin_id_buffer.as_entire_binding() },
            ],
        })
    }

    /// GPU pass: count organisms and assign skin IDs.
    /// Must be called before `generate_density`.
    pub fn count_organisms(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        count_bind_group: &wgpu::BindGroup,
        max_cells: u32,
    ) {
        if max_cells == 0 { return; }

        // Pass 0: clear histogram
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin Clear Histogram"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_clear_pipeline);
            pass.set_bind_group(0, count_bind_group, &[]);
            pass.dispatch_workgroups((HISTOGRAM_SIZE + 255) / 256, 1, 1);
        }
        // Pass 1: count cells per organism
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin Count Organisms"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_organisms_pipeline);
            pass.set_bind_group(0, count_bind_group, &[]);
            pass.dispatch_workgroups((max_cells + 255) / 256, 1, 1);
        }
        // Pass 2: assign skin IDs
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin Assign IDs"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.assign_skin_ids_pipeline);
            pass.set_bind_group(0, count_bind_group, &[]);
            pass.dispatch_workgroups((max_cells + 255) / 256, 1, 1);
        }
    }

    /// GPU pass: generate K=2 organism density from cell positions.
    /// Must be called after `count_organisms` and before `extract_mesh`.
    pub fn generate_density(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        density_bind_group: &wgpu::BindGroup,
        max_cells: u32,
    ) {
        if max_cells == 0 { return; }

        // Pass 1: clear all K=2 slot data
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin Clear Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_clear_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            let workgroups_x = ((self.total_voxels + 255) / 256).min(65535);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
        // Pass 2: per-cell metaball splatting with CAS slot claiming
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin Generate Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_generate_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            pass.dispatch_workgroups((max_cells + 63) / 64, 1, 1);
        }
        // Pass 3: normalize fixed-point → f32 and copy org IDs
        {
            let wg = (self.grid_resolution + 3) / 4;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin Normalize Density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_normalize_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
    }

    /// GPU pass: extract organism-aware isosurface mesh.
    /// Must be called after `generate_density`.
    pub fn extract_mesh(&self, encoder: &mut wgpu::CommandEncoder) {
        let padded_res = self.grid_resolution + 2;
        let wg = (padded_res + 3) / 4;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin SN Reset"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_reset_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin SN Vertices"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_vertex_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin SN Indices"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.sn_index_pipeline);
            pass.set_bind_group(0, &self.sn_compute_bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Skin SN Finalize"), timestamp_writes: None,
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

    pub fn update_skin_params(&mut self, queue: &wgpu::Queue, params: OrganismSkinParams) {
        self.skin_params = params;
        queue.write_buffer(&self.skin_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn set_skin_radius_scale(&mut self, queue: &wgpu::Queue, scale: f32) {
        self.skin_radius_scale = scale;
        self.upload_density_params(queue);
    }

    pub fn set_iso_level(&mut self, queue: &wgpu::Queue, level: f32) {
        self.iso_level = level;
        self.upload_sn_params(queue);
    }

    pub fn set_time(&self, queue: &wgpu::Queue, time: f32) {
        queue.write_buffer(&self.skin_params_buffer, 40, bytemuck::bytes_of(&time));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn try_read_counts(&mut self, device: &wgpu::Device) {
        let slice = self.counter_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        {
            let data = slice.get_mapped_range();
            let c: &Counters = bytemuck::from_bytes(&data);
            self.vertex_count = c.vertex_count.min(self.max_vertices);
            self.index_count = c.index_count.min(self.max_indices);
        }
        self.counter_staging_buffer.unmap();
    }

    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn upload_density_params(&self, queue: &wgpu::Queue) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / self.grid_resolution as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        let p = OrganismDensityParams {
            grid_origin: grid_origin.to_array(),
            cell_size,
            grid_resolution: self.grid_resolution,
            skin_radius_scale: self.skin_radius_scale,
            max_cells: 0,
            min_cells_for_skin: MIN_CELLS_FOR_SKIN,
        };
        queue.write_buffer(&self.density_params_buffer, 0, bytemuck::bytes_of(&p));
    }

    fn upload_sn_params(&self, queue: &wgpu::Queue) {
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / self.grid_resolution as f32;
        let grid_origin = self.world_center - Vec3::splat(world_diameter / 2.0);
        let padded_origin = grid_origin - Vec3::splat(cell_size);
        let p = SurfaceNetsParams {
            grid_resolution: self.grid_resolution + 2,
            iso_level: self.iso_level,
            cell_size,
            max_vertices: self.max_vertices,
            grid_origin: padded_origin.to_array(),
            max_indices: self.max_indices,
            density_resolution: self.grid_resolution,
            _pad_a: 0, _pad_b: 0, _pad_c: 0,
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
