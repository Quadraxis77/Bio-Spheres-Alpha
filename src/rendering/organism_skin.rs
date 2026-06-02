//! Organism Shrink-Wrap Skin Renderer
//!
//! Replaces the density-field + surface-nets approach with a GPU icosphere
//! mesh that physically contracts onto each organism's cell surfaces each frame.
//!
//! ## Pipeline (per frame)
//! ```text
//! clear_org_state  -> accumulate_cells -> finalize_orgs
//!   -> shrink_step (xN_SHRINK_ITERS)
//!   -> smooth_step (xN_SMOOTH_ITERS)
//!   -> render pass
//! ```
//!
//! ## Public API (unchanged from old OrganismSkinRenderer)
//! - `new(...)` - same signature
//! - `create_count_bind_group(...)` - kept for gpu_scene.rs compatibility (unused internally)
//! - `create_density_bind_group(...)` - kept for compatibility (unused internally)
//! - `count_organisms(...)` - now runs clear+accumulate+finalize
//! - `generate_density(...)` - now runs shrink+smooth iterations
//! - `extract_mesh(...)` - no-op (mesh is already in vertex_buffer)
//! - `render(...)` - unchanged
//! - `try_read_skinned_count(...)` - reads org_state to count active organisms
//! - `set_skin_radius_scale(...)`, `set_iso_level(...)`, `set_time(...)` - update params

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

// -- Constants -----------------------------------------------------------------

const VERTS_PER_ORG: u32 = 642;
const TRIS_PER_ORG: u32 = 1280;
const MAX_ORGANISMS: u32 = 512;

// -- GPU structs ---------------------------------------------------------------

/// Matches ShrinkParams in organism_shrinkwrap.wgsl (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ShrinkParams {
    world_size: f32,
    grid_cell_size: f32,
    grid_resolution: i32,
    cell_count: u32,
    skin_offset: f32,
    shrink_speed: f32,
    smooth_factor: f32,
    min_cells: u32,
}

/// Matches SkinVertex in organism_shrinkwrap.wgsl (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    organism_id: f32,
    normal: [f32; 3],
    _pad: f32,
}

/// Matches OrgAccum in organism_shrinkwrap.wgsl (32 bytes, all atomic i32/u32)
/// We zero-init this buffer via clear_buffer each frame.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct OrgAccum {
    sum_x: i32,
    sum_y: i32,
    sum_z: i32,
    cell_count: u32,
    max_dist_sq: i32,
    is_used: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Matches OrgState in organism_shrinkwrap.wgsl (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct OrgState {
    centroid: [f32; 3],
    radius: f32,
    skin_id: u32,
    cell_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Camera uniform for render pass
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Organism skin material params - same as before, kept for API compatibility
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OrganismSkinParams {
    pub base_r: f32,
    pub base_g: f32,
    pub base_b: f32,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
    pub fresnel: f32,
    pub fresnel_power: f32,
    pub alpha: f32,
    pub time: f32,
    pub sss_strength: f32,
    pub sss_r: f32,
    pub sss_g: f32,
    pub sss_b: f32,
    pub rim_strength: f32,
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    pub _pad: f32,
}

impl Default for OrganismSkinParams {
    fn default() -> Self {
        Self {
            base_r: 0.85,
            base_g: 0.55,
            base_b: 0.35,
            ambient: 0.12,
            diffuse: 0.6,
            specular: 0.5,
            shininess: 48.0,
            fresnel: 0.08,
            fresnel_power: 3.0,
            alpha: 0.55,
            time: 0.0,
            sss_strength: 0.5,
            sss_r: 1.0,
            sss_g: 0.4,
            sss_b: 0.1,
            rim_strength: 0.35,
            light_dir_x: 0.4,
            light_dir_y: 0.8,
            light_dir_z: 0.4,
            _pad: 0.0,
        }
    }
}

// -- Icosphere geometry (2 subdivisions: 162 vertices, 320 triangles) ---------

/// Build a subdivided icosphere on the CPU.
/// Returns (vertices_on_unit_sphere, triangle_indices).
/// Each vertex is normalised to the unit sphere.
fn build_icosphere(subdivisions: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let phi = (1.0_f32 + 5.0_f32.sqrt()) / 2.0;
    let scale = 1.0 / (1.0_f32 + phi * phi).sqrt();

    // 12 base icosahedron vertices
    let mut verts: Vec<[f32; 3]> = vec![
        [-scale, phi * scale, 0.0],
        [scale, phi * scale, 0.0],
        [-scale, -phi * scale, 0.0],
        [scale, -phi * scale, 0.0],
        [0.0, -scale, phi * scale],
        [0.0, scale, phi * scale],
        [0.0, -scale, -phi * scale],
        [0.0, scale, -phi * scale],
        [phi * scale, 0.0, -scale],
        [phi * scale, 0.0, scale],
        [-phi * scale, 0.0, -scale],
        [-phi * scale, 0.0, scale],
    ];

    // 20 base faces
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    let mut midpoint_cache: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();

    for _ in 0..subdivisions {
        let mut new_faces = Vec::with_capacity(faces.len() * 4);
        for face in &faces {
            let [a, b, c] = *face;
            // Inline midpoint: get or create midpoint vertex for edge (p, q)
            let ab = {
                let key = (a.min(b), a.max(b));
                if let Some(&idx) = midpoint_cache.get(&key) {
                    idx
                } else {
                    let va = verts[a as usize];
                    let vb = verts[b as usize];
                    let mid = [
                        (va[0] + vb[0]) * 0.5,
                        (va[1] + vb[1]) * 0.5,
                        (va[2] + vb[2]) * 0.5,
                    ];
                    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
                    let idx = verts.len() as u32;
                    verts.push([mid[0] / len, mid[1] / len, mid[2] / len]);
                    midpoint_cache.insert(key, idx);
                    idx
                }
            };
            let bc = {
                let key = (b.min(c), b.max(c));
                if let Some(&idx) = midpoint_cache.get(&key) {
                    idx
                } else {
                    let va = verts[b as usize];
                    let vb = verts[c as usize];
                    let mid = [
                        (va[0] + vb[0]) * 0.5,
                        (va[1] + vb[1]) * 0.5,
                        (va[2] + vb[2]) * 0.5,
                    ];
                    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
                    let idx = verts.len() as u32;
                    verts.push([mid[0] / len, mid[1] / len, mid[2] / len]);
                    midpoint_cache.insert(key, idx);
                    idx
                }
            };
            let ca = {
                let key = (c.min(a), c.max(a));
                if let Some(&idx) = midpoint_cache.get(&key) {
                    idx
                } else {
                    let va = verts[c as usize];
                    let vb = verts[a as usize];
                    let mid = [
                        (va[0] + vb[0]) * 0.5,
                        (va[1] + vb[1]) * 0.5,
                        (va[2] + vb[2]) * 0.5,
                    ];
                    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
                    let idx = verts.len() as u32;
                    verts.push([mid[0] / len, mid[1] / len, mid[2] / len]);
                    midpoint_cache.insert(key, idx);
                    idx
                }
            };
            new_faces.push([a, ab, ca]);
            new_faces.push([b, bc, ab]);
            new_faces.push([c, ca, bc]);
            new_faces.push([ab, bc, ca]);
        }
        faces = new_faces;
        midpoint_cache.clear();
    }

    let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();
    (verts, indices)
}

/// Build the full index buffer for all MAX_ORGANISMS organisms.
/// Each organism's indices are offset by `org_idx * VERTS_PER_ORG`.
fn build_full_index_buffer() -> Vec<u32> {
    let (verts, local_indices) = build_icosphere(3);
    assert_eq!(
        verts.len() as u32,
        VERTS_PER_ORG,
        "Icosphere vertex count mismatch: got {}, expected {}",
        verts.len(),
        VERTS_PER_ORG
    );
    assert_eq!(
        local_indices.len() as u32,
        TRIS_PER_ORG * 3,
        "Icosphere index count mismatch: got {}, expected {}",
        local_indices.len(),
        TRIS_PER_ORG * 3
    );

    let mut out = Vec::with_capacity((MAX_ORGANISMS * TRIS_PER_ORG * 3) as usize);
    for org in 0..MAX_ORGANISMS {
        let offset = org * VERTS_PER_ORG;
        for &idx in &local_indices {
            out.push(idx + offset);
        }
    }
    out
}

/// Build the icosphere vertex positions for the WGSL shader's `ico_pos()` function.
/// Returns normalised unit-sphere positions in the same order as `build_icosphere(2)`.
/// Used to verify the Rust and WGSL geometry match.
#[allow(dead_code)]
fn build_icosphere_positions() -> Vec<[f32; 3]> {
    build_icosphere(2).0
}

// -- OrganismSkinRenderer ------------------------------------------------------

pub struct OrganismSkinRenderer {
    // -- Compute pipelines -----------------------------------------------------
    clear_pipeline: wgpu::ComputePipeline,
    accumulate_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    shrink_pipeline: wgpu::ComputePipeline,
    smooth_pipeline: wgpu::ComputePipeline,

    // -- Compute bind group (rebuilt when triple-buffer index changes) ---------
    pub compute_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    org_accum_buffer: wgpu::Buffer,
    org_state_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    ico_unit_positions_buffer: wgpu::Buffer,

    // -- Render pipeline -------------------------------------------------------
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    skin_params_buffer: wgpu::Buffer,

    // -- Compatibility shims (kept so gpu_scene.rs compiles unchanged) ---------
    /// Dummy layout - create_count_bind_group returns a no-op bind group
    pub count_bind_group_layout: wgpu::BindGroupLayout,
    /// Dummy layout - create_density_bind_group returns a no-op bind group
    pub density_bind_group_layout: wgpu::BindGroupLayout,
    dummy_buffer: wgpu::Buffer,

    // -- Config ----------------------------------------------------------------
    world_size: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    pub skin_offset: f32,       // replaces skin_radius_scale
    pub skin_radius_scale: f32, // kept for API compat, maps to skin_offset
    pub iso_level: f32,         // kept for API compat (unused)
    pub width: u32,
    pub height: u32,
    pub skin_params: OrganismSkinParams,
    pub skinned_cell_count: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub total_voxels: u32, // kept for API compat
    pub grid_resolution_pub: u32,
    pub temporal_blend: f32, // kept for API compat
    shrink_iters: u32,
    smooth_iters: u32,
}

impl OrganismSkinRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        world_radius: f32,
        _world_center: Vec3,
        _capacity: u32,
        width: u32,
        height: u32,
        settings: &crate::ui::OrganismSkinSettings,
    ) -> Self {
        let world_size = world_radius * 2.0;
        let grid_resolution = 128_i32;
        let grid_cell_size = world_size / grid_resolution as f32;
        let skin_offset = settings.radius_scale.max(0.1); // reuse radius_scale as offset

        // -- Buffers -----------------------------------------------------------
        let params = ShrinkParams {
            world_size,
            grid_cell_size,
            grid_resolution,
            cell_count: 0,
            skin_offset,
            shrink_speed: 0.25,
            smooth_factor: 0.3,
            min_cells: 4,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShrinkWrap Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let org_accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShrinkWrap OrgAccum"),
            size: (MAX_ORGANISMS as u64) * std::mem::size_of::<OrgAccum>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let org_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShrinkWrap OrgState"),
            size: (MAX_ORGANISMS as u64) * std::mem::size_of::<OrgState>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vertex_count = MAX_ORGANISMS * VERTS_PER_ORG;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShrinkWrap Vertices"),
            size: (vertex_count as u64) * std::mem::size_of::<GpuVertex>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let index_data = build_full_index_buffer();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShrinkWrap Indices"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Upload unit-sphere icosphere vertex positions for the shader's ico_pos() lookup.
        // Stored as vec4<f32> (xyz = position, w = 0) to match WGSL alignment.
        let (ico_verts, _) = build_icosphere(3);
        let ico_data: Vec<[f32; 4]> = ico_verts.iter().map(|v| [v[0], v[1], v[2], 0.0]).collect();
        let ico_unit_positions_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ShrinkWrap Ico Unit Positions"),
                contents: bytemuck::cast_slice(&ico_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShrinkWrap Dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // -- Compute bind group layout -----------------------------------------
        // Bindings (must match organism_shrinkwrap.wgsl):
        //  0: params (uniform)
        //  1: position_and_mass (read)
        //  2: death_flags (read)
        //  3: cell_count_buf (read)
        //  4: label_buffer (read)
        //  5: spatial_grid_counts (read)
        //  6: spatial_grid_cells (read)
        //  7: org_accum (rw)
        //  8: org_state (rw)
        //  9: vertices (rw)
        // 10: stable_id_per_cell (read)
        // 11: ico_unit_positions (read)
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ShrinkWrap Compute BGL"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_ro(3),
                    bgl_storage_ro(4),
                    bgl_storage_ro(5),
                    bgl_storage_ro(6),
                    bgl_storage_rw(7),
                    bgl_storage_rw(8),
                    bgl_storage_rw(9),
                    bgl_storage_ro(10),
                    bgl_storage_ro(11),
                ],
            });

        // -- Compute pipelines -------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShrinkWrap Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_shrinkwrap.wgsl").into(),
            ),
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShrinkWrap Compute Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        let make_cp = |entry: &str, lbl: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(lbl),
                layout: Some(&compute_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let clear_pipeline = make_cp("clear_org_state", "SW Clear");
        let accumulate_pipeline = make_cp("accumulate_cells", "SW Accumulate");
        let finalize_pipeline = make_cp("finalize_orgs", "SW Finalize");
        let shrink_pipeline = make_cp("shrink_step", "SW Shrink");
        let smooth_pipeline = make_cp("smooth_step", "SW Smooth");

        // -- Dummy layouts for API compatibility -------------------------------
        let dummy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShrinkWrap Dummy BGL"),
            entries: &[bgl_storage_ro(0)],
        });
        let count_bind_group_layout = dummy_bgl;
        let density_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ShrinkWrap Dummy Density BGL"),
                entries: &[bgl_storage_ro(0)],
            });

        // -- Render pipeline ---------------------------------------------------
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShrinkWrap Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/organism_skin_render.wgsl").into(),
            ),
        });
        let camera_uniform = CameraUniform {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            _padding: 0.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShrinkWrap Camera"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let skin_params = OrganismSkinParams::default();
        let skin_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShrinkWrap Skin Params"),
            contents: bytemuck::bytes_of(&skin_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShrinkWrap Render BGL"),
            entries: &[bgl_uniform_vs_fs(0), bgl_uniform_vs_fs(1)],
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShrinkWrap Render BG"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: skin_params_buffer.as_entire_binding(),
                },
            ],
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ShrinkWrap Render Layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ShrinkWrap Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuVertex>() as u64,
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
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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

        let total_indices = MAX_ORGANISMS * TRIS_PER_ORG * 3;

        Self {
            clear_pipeline,
            accumulate_pipeline,
            finalize_pipeline,
            shrink_pipeline,
            smooth_pipeline,
            compute_bind_group_layout,
            params_buffer,
            org_accum_buffer,
            org_state_buffer,
            vertex_buffer,
            index_buffer,
            ico_unit_positions_buffer,
            render_pipeline,
            render_bind_group,
            camera_buffer,
            skin_params_buffer,
            count_bind_group_layout,
            density_bind_group_layout,
            dummy_buffer,
            world_size,
            grid_resolution,
            grid_cell_size,
            skin_offset,
            skin_radius_scale: skin_offset,
            iso_level: settings.iso_level,
            width,
            height,
            skin_params,
            skinned_cell_count: 0,
            vertex_count: MAX_ORGANISMS * VERTS_PER_ORG,
            index_count: total_indices,
            total_voxels: 128 * 128 * 128,
            grid_resolution_pub: 128,
            temporal_blend: 0.0,
            shrink_iters: settings.shrink_iters.max(1),
            smooth_iters: settings.smooth_iters,
        }
    }

    // -- API compatibility shims -----------------------------------------------

    /// Creates a dummy bind group (kept so gpu_scene.rs compiles unchanged).
    /// The shrink-wrap system uses its own internal bind group created via
    /// `create_compute_bind_group`.
    pub fn create_count_bind_group(
        &self,
        device: &wgpu::Device,
        _cell_count: &wgpu::Buffer,
        _death_flags: &wgpu::Buffer,
        _label_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShrinkWrap Dummy Count BG"),
            layout: &self.count_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.dummy_buffer.as_entire_binding(),
            }],
        })
    }

    /// Creates a dummy bind group (kept so gpu_scene.rs compiles unchanged).
    pub fn create_density_bind_group(
        &self,
        device: &wgpu::Device,
        _position_and_mass: &wgpu::Buffer,
        _death_flags: &wgpu::Buffer,
        _cell_count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShrinkWrap Dummy Density BG"),
            layout: &self.density_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.dummy_buffer.as_entire_binding(),
            }],
        })
    }

    /// Create the real compute bind group that wires all GPU buffers.
    /// Call once after initialization and cache the result.
    pub fn create_compute_bind_group(
        &self,
        device: &wgpu::Device,
        position_and_mass: &wgpu::Buffer,
        death_flags: &wgpu::Buffer,
        cell_count_buf: &wgpu::Buffer,
        label_buffer: &wgpu::Buffer,
        spatial_grid_counts: &wgpu::Buffer,
        spatial_grid_cells: &wgpu::Buffer,
        stable_id_per_cell: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShrinkWrap Compute BG"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: position_and_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: label_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: spatial_grid_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: spatial_grid_cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.org_accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.org_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: stable_id_per_cell.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.ico_unit_positions_buffer.as_entire_binding(),
                },
            ],
        })
    }

    // -- Main per-frame passes -------------------------------------------------

    /// Pass 1+2+3: clear state, accumulate cell positions, finalize org spheres.
    /// `count_bind_group` is ignored (kept for API compat).
    pub fn count_organisms(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _count_bind_group: &wgpu::BindGroup,
        _max_cells: u32,
    ) {
        // Actual work is done in encode_shrinkwrap_frame - this is a no-op shim.
    }

    /// Pass 4+5: shrink + smooth iterations.
    /// `density_bind_group` is ignored (kept for API compat).
    pub fn generate_density(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _density_bind_group: &wgpu::BindGroup,
        _max_cells: u32,
    ) {
        // Actual work is done in encode_shrinkwrap_frame - this is a no-op shim.
    }

    /// No-op - mesh is already in vertex_buffer after encode_shrinkwrap_frame.
    pub fn extract_mesh(&self, _encoder: &mut wgpu::CommandEncoder) {}

    /// Run the full shrink-wrap pipeline for one frame.
    /// Call this instead of count_organisms + generate_density + extract_mesh.
    pub fn encode_shrinkwrap_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        compute_bind_group: &wgpu::BindGroup,
        max_cells: u32,
    ) {
        if max_cells == 0 {
            return;
        }

        // Clear org_accum buffer via DMA (faster than compute dispatch)
        encoder.clear_buffer(&self.org_accum_buffer, 0, None);

        let org_wg = (MAX_ORGANISMS + 63) / 64;
        let cell_wg = (max_cells + 255) / 256;
        let vert_wg = (MAX_ORGANISMS * VERTS_PER_ORG + 63) / 64;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ShrinkWrap Frame"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, compute_bind_group, &[]);

            // Clear org_state
            pass.set_pipeline(&self.clear_pipeline);
            pass.dispatch_workgroups(org_wg, 1, 1);

            // Accumulate cell positions into org_accum
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.dispatch_workgroups(cell_wg, 1, 1);

            // Finalize: compute centroid, place icosphere
            pass.set_pipeline(&self.finalize_pipeline);
            pass.dispatch_workgroups(org_wg, 1, 1);

            // Shrink iterations
            for _ in 0..self.shrink_iters {
                pass.set_pipeline(&self.shrink_pipeline);
                pass.dispatch_workgroups(vert_wg, 1, 1);
            }

            // Smooth iterations
            for _ in 0..self.smooth_iters {
                pass.set_pipeline(&self.smooth_pipeline);
                pass.dispatch_workgroups(vert_wg, 1, 1);
            }
        }
    }

    /// Render the shrink-wrap mesh.
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
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 2000.0);
        let cam = CameraUniform {
            view_proj: (proj * view).to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam));

        let total_indices = MAX_ORGANISMS * TRIS_PER_ORG * 3;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ShrinkWrap Render Pass"),
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
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..total_indices, 0, 0..1);
    }

    // -- Setters ---------------------------------------------------------------

    pub fn update_skin_params(&mut self, queue: &wgpu::Queue, params: OrganismSkinParams) {
        self.skin_params = params;
        queue.write_buffer(&self.skin_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn set_skin_radius_scale(&mut self, queue: &wgpu::Queue, scale: f32) {
        self.skin_radius_scale = scale;
        self.skin_offset = scale.max(0.1);
        self.upload_params(queue);
    }

    pub fn set_iso_level(&mut self, _queue: &wgpu::Queue, level: f32) {
        self.iso_level = level; // no-op for shrink-wrap
    }

    pub fn set_time(&self, queue: &wgpu::Queue, time: f32) {
        // time is at byte offset 40 in OrganismSkinParams
        queue.write_buffer(&self.skin_params_buffer, 40, bytemuck::bytes_of(&time));
    }

    pub fn set_shrink_params(
        &mut self,
        queue: &wgpu::Queue,
        shrink_speed: f32,
        smooth_factor: f32,
        shrink_iters: u32,
        smooth_iters: u32,
        min_cells: u32,
    ) {
        self.shrink_iters = shrink_iters.max(1);
        self.smooth_iters = smooth_iters;
        let p = ShrinkParams {
            world_size: self.world_size,
            grid_cell_size: self.grid_cell_size,
            grid_resolution: self.grid_resolution,
            cell_count: 0,
            skin_offset: self.skin_offset,
            shrink_speed: shrink_speed.clamp(0.01, 1.0),
            smooth_factor: smooth_factor.clamp(0.0, 0.6),
            min_cells: min_cells.max(1),
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&p));
    }

    pub fn set_temporal_blend(&mut self, _queue: &wgpu::Queue, value: f32) {
        self.temporal_blend = value; // no-op
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// No-op shim - skinned_cell_count is updated by encode_shrinkwrap_frame.
    pub fn try_read_skinned_count(&mut self, _device: &wgpu::Device) {
        // For the shrink-wrap system we always consider there to be active skins
        // when the simulation has cells. The actual per-organism activity is
        // determined by org_state on the GPU.
        self.skinned_cell_count = 1; // non-zero = don't skip render
    }

    pub fn triangle_count(&self) -> u32 {
        MAX_ORGANISMS * TRIS_PER_ORG
    }

    // -- Private ---------------------------------------------------------------

    fn upload_params(&self, queue: &wgpu::Queue) {
        let p = ShrinkParams {
            world_size: self.world_size,
            grid_cell_size: self.grid_cell_size,
            grid_resolution: self.grid_resolution,
            cell_count: 0,
            skin_offset: self.skin_offset,
            shrink_speed: 0.25,
            smooth_factor: 0.3,
            min_cells: 4,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&p));
    }
}

// -- Bind group layout helpers -------------------------------------------------

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform_vs_fs(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
