//! GPU-based instance buffer builder using compute shaders.
//!
//! Builds cell instance data on the GPU to eliminate CPU-side iteration
//! and reduce CPU→GPU data transfer. Includes frustum and occlusion culling.

use crate::cell::types::CellTypeVisuals;
use crate::genome::Genome;
use crate::simulation::CanonicalState;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Culling mode for the instance builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullingMode {
    /// No culling - all cells are rendered
    #[default]
    Disabled,
    /// Frustum culling only
    FrustumOnly,
    /// Occlusion culling only (requires Hi-Z texture)
    OcclusionOnly,
    /// Frustum + occlusion culling (requires Hi-Z texture)
    FrustumAndOcclusion,
}

impl CullingMode {
    fn as_u32(self) -> u32 {
        match self {
            CullingMode::Disabled => 0,
            CullingMode::FrustumOnly => 1,
            CullingMode::OcclusionOnly => 3,
            CullingMode::FrustumAndOcclusion => 2,
        }
    }
}

/// Culling statistics from the last frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct CullingStats {
    pub total_cells: u32,
    pub visible_cells: u32,
    pub frustum_culled: u32,
    pub occluded: u32,
}

/// GPU instance builder with compute shader pipeline.
pub struct InstanceBuilder {
    // Compute pipeline
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    // Uniform buffer
    params_buffer: wgpu::Buffer,
    
    // Input buffers (simulation data)
    positions_buffer: wgpu::Buffer,
    rotations_buffer: wgpu::Buffer,
    radii_buffer: wgpu::Buffer,
    mode_indices_buffer: wgpu::Buffer,
    cell_ids_buffer: wgpu::Buffer,
    genome_ids_buffer: wgpu::Buffer,
    
    // Lookup table buffers
    mode_visuals_buffer: wgpu::Buffer,
    cell_type_visuals_buffer: wgpu::Buffer,
    
    // Output buffer (instance data for rendering)
    pub instance_buffer: wgpu::Buffer,
    
    // Indirect draw buffer for GPU-driven rendering
    pub indirect_buffer: wgpu::Buffer,
    
    // Counter buffer for atomic operations
    counters_buffer: wgpu::Buffer,
    counters_readback_buffer: wgpu::Buffer,
    
    // Current bind group (recreated when buffers change size)
    bind_group: Option<wgpu::BindGroup>,
    
    // Hi-Z texture for occlusion culling
    hiz_texture: Option<wgpu::Texture>,
    hiz_view: Option<wgpu::TextureView>,
    hiz_mip_count: u32,
    /// Whether external Hi-Z texture has been set up (to avoid recreating bind group every frame)
    hiz_configured: bool,
    
    // Capacity tracking
    cell_capacity: usize,
    mode_capacity: usize,
    cell_type_capacity: usize,
    
    // Dirty tracking
    positions_dirty: bool,
    rotations_dirty: bool,
    radii_dirty: bool,
    mode_indices_dirty: bool,
    cell_ids_dirty: bool,
    genome_ids_dirty: bool,
    mode_visuals_dirty: bool,
    cell_type_visuals_dirty: bool,
    
    // Hash for change detection
    last_cell_count: usize,
    last_mode_hash: u64,
    last_visuals_hash: u64,
    
    // Culling settings
    culling_mode: CullingMode,
    last_stats: CullingStats,
    /// Occlusion bias: negative = more aggressive (cull more), positive = more conservative (cull less)
    occlusion_bias: f32,
    /// Mip level override: -1 = auto, 0+ = force specific mip level
    occlusion_mip_override: i32,
    /// Minimum screen-space size (pixels) to apply occlusion culling
    min_screen_size: f32,
    /// Don't cull objects closer than this distance
    min_distance: f32,
    
    // Last visible count for draw calls
    last_visible_count: u32,
}


/// Frustum plane representation (normal + distance from origin).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrustumPlane {
    normal_and_dist: [f32; 4], // xyz = normal, w = distance
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BuildParams {
    cell_count: u32,
    mode_count: u32,
    cell_type_count: u32,
    culling_enabled: u32,
    // View-projection matrix (64 bytes)
    view_proj: [[f32; 4]; 4],
    // Camera position - vec3 in WGSL followed by f32 packs together
    camera_pos: [f32; 3],
    near_plane: f32,
    far_plane: f32,
    screen_width: f32,
    screen_height: f32,
    hiz_mip_count: u32,
    // Occlusion culling parameters
    occlusion_bias: f32,
    occlusion_mip_override: i32,
    min_screen_size: f32,
    min_distance: f32,
    // Frustum planes array needs 16-byte alignment
    // Current offset is 128, which is 16-byte aligned, so no padding needed
    frustum_planes: [FrustumPlane; 6],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ModeVisuals {
    color: [f32; 3],
    opacity: f32,
    emissive: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuCellTypeVisuals {
    specular_strength: f32,
    specular_power: f32,
    fresnel_strength: f32,
    membrane_noise_scale: f32,
    membrane_noise_strength: f32,
    membrane_noise_speed: f32,
    _pad: [f32; 2],
}

// Instance struct must match shader and CellRenderer
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CellInstance {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    pub visual_params: [f32; 4],
    pub membrane_params: [f32; 4],
    pub rotation: [f32; 4],
}


impl InstanceBuilder {
    /// Create a new instance builder with the given capacity.
    pub fn new(device: &wgpu::Device, cell_capacity: usize) -> Self {
        let mode_capacity = 64;
        let cell_type_capacity = 16;
        
        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Instance Builder Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/build_instances.wgsl").into()),
        });
        
        // Create bind group layout with Hi-Z texture support
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instance Builder Bind Group Layout"),
            entries: &[
                // params uniform (binding 0)
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
                // positions (binding 1)
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
                // rotations (binding 2)
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
                // radii (binding 3)
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
                // mode_indices (binding 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // cell_ids (binding 5)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // genome_ids (binding 6)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // mode_visuals (binding 7)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // cell_type_visuals (binding 8)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // instances output (binding 9)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // counters (binding 10)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Hi-Z texture (binding 11) - R32Float is not filterable
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instance Builder Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Instance Builder Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Params"),
            size: std::mem::size_of::<BuildParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let positions_buffer = Self::create_storage_buffer(device, "Positions", cell_capacity * 16);
        let rotations_buffer = Self::create_storage_buffer(device, "Rotations", cell_capacity * 16);
        let radii_buffer = Self::create_storage_buffer(device, "Radii", cell_capacity * 4);
        let mode_indices_buffer = Self::create_storage_buffer(device, "Mode Indices", cell_capacity * 4);
        let cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", cell_capacity * 4);
        let genome_ids_buffer = Self::create_storage_buffer(device, "Genome IDs", cell_capacity * 4);
        
        let mode_visuals_buffer = Self::create_storage_buffer(device, "Mode Visuals", mode_capacity * std::mem::size_of::<ModeVisuals>());
        let cell_type_visuals_buffer = Self::create_storage_buffer(device, "Cell Type Visuals", cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>());
        
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (cell_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Indirect draw buffer: vertex_count, instance_count, first_vertex, first_instance
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Indirect"),
            size: 16, // 4 u32s
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 4 counters: visible, total, frustum_culled, occluded
        let counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Counters"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counters_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counters Readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            positions_buffer,
            rotations_buffer,
            radii_buffer,
            mode_indices_buffer,
            cell_ids_buffer,
            genome_ids_buffer,
            mode_visuals_buffer,
            cell_type_visuals_buffer,
            instance_buffer,
            indirect_buffer,
            counters_buffer,
            counters_readback_buffer,
            bind_group: None,
            hiz_texture: None,
            hiz_view: None,
            hiz_mip_count: 0,
            hiz_configured: false,
            cell_capacity,
            mode_capacity,
            cell_type_capacity,
            positions_dirty: true,
            rotations_dirty: true,
            radii_dirty: true,
            mode_indices_dirty: true,
            cell_ids_dirty: true,
            genome_ids_dirty: true,
            mode_visuals_dirty: true,
            cell_type_visuals_dirty: true,
            last_cell_count: 0,
            last_mode_hash: 0,
            last_visuals_hash: 0,
            culling_mode: CullingMode::FrustumOnly,
            last_stats: CullingStats::default(),
            occlusion_bias: 0.0,
            occlusion_mip_override: -1,
            min_screen_size: 0.0,
            min_distance: 0.0,
            last_visible_count: 0,
        }
    }
    
    fn create_storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Instance Builder {}", label)),
            size: size.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Set the culling mode.
    pub fn set_culling_mode(&mut self, mode: CullingMode) {
        self.culling_mode = mode;
    }
    
    /// Get the current culling mode.
    pub fn culling_mode(&self) -> CullingMode {
        self.culling_mode
    }
    
    /// Get culling statistics from the last frame.
    pub fn culling_stats(&self) -> CullingStats {
        self.last_stats
    }
    
    /// Set the occlusion bias.
    /// Negative values = more aggressive culling (cull more cells).
    /// Positive values = more conservative culling (cull fewer cells).
    /// Range: typically -0.1 to 0.1
    pub fn set_occlusion_bias(&mut self, bias: f32) {
        self.occlusion_bias = bias;
    }
    
    /// Get the current occlusion bias.
    pub fn occlusion_bias(&self) -> f32 {
        self.occlusion_bias
    }
    
    /// Set the mip level override for occlusion culling.
    /// -1 = auto (use mip 0), 0+ = force specific mip level
    pub fn set_occlusion_mip_override(&mut self, mip: i32) {
        self.occlusion_mip_override = mip;
    }
    
    /// Get the current mip level override.
    pub fn occlusion_mip_override(&self) -> i32 {
        self.occlusion_mip_override
    }
    
    /// Set the minimum screen-space size (in pixels) for occlusion culling.
    /// Objects smaller than this won't be culled.
    pub fn set_min_screen_size(&mut self, size: f32) {
        self.min_screen_size = size;
    }
    
    /// Get the minimum screen-space size.
    pub fn min_screen_size(&self) -> f32 {
        self.min_screen_size
    }
    
    /// Set the minimum distance for occlusion culling.
    /// Objects closer than this won't be culled.
    pub fn set_min_distance(&mut self, distance: f32) {
        self.min_distance = distance;
    }
    
    /// Get the minimum distance.
    pub fn min_distance(&self) -> f32 {
        self.min_distance
    }
    
    /// Get the number of visible instances from the last build.
    pub fn visible_count(&self) -> u32 {
        self.last_visible_count
    }
    
    /// Mark all buffers as dirty (forces full update).
    pub fn mark_all_dirty(&mut self) {
        self.positions_dirty = true;
        self.rotations_dirty = true;
        self.radii_dirty = true;
        self.mode_indices_dirty = true;
        self.cell_ids_dirty = true;
        self.genome_ids_dirty = true;
        self.mode_visuals_dirty = true;
        self.cell_type_visuals_dirty = true;
    }
    
    /// Create or resize the Hi-Z texture for occlusion culling.
    pub fn create_hiz_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Calculate mip count
        let max_dim = width.max(height);
        let mip_count = (max_dim as f32).log2().floor() as u32 + 1;
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hi-Z Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                | wgpu::TextureUsages::RENDER_ATTACHMENT 
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.hiz_texture = Some(texture);
        self.hiz_view = Some(view);
        self.hiz_mip_count = mip_count;
        self.bind_group = None; // Force bind group recreation
    }
    
    /// Set an external Hi-Z texture for occlusion culling.
    /// Use this when the Hi-Z texture is generated by HizGenerator.
    /// Only recreates bind group if the Hi-Z texture actually changed.
    pub fn set_hiz_texture(&mut self, device: &wgpu::Device, hiz_view: &wgpu::TextureView, mip_count: u32) {
        // Only recreate bind group if mip count changed (indicates texture resize)
        // or if Hi-Z hasn't been configured yet
        if !self.hiz_configured || self.hiz_mip_count != mip_count || self.bind_group.is_none() {
            self.hiz_mip_count = mip_count;
            self.hiz_configured = true;
            self.bind_group = None;
            self.recreate_bind_group_with_hiz(device, hiz_view);
        }
    }
    
    /// Reset Hi-Z configuration (call on resize).
    pub fn reset_hiz(&mut self) {
        self.hiz_configured = false;
        self.bind_group = None;
    }
    
    fn recreate_bind_group_with_hiz(&mut self, device: &wgpu::Device, hiz_view: &wgpu::TextureView) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance Builder Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rotations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mode_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cell_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.genome_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.mode_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cell_type_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
            ],
        }));
    }

    /// Update input buffers from simulation state.
    pub fn update_from_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &CanonicalState,
        genome: Option<&Genome>,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
    ) {
        let cell_count = state.cell_count;
        
        // Check if we need to resize buffers
        if cell_count > self.cell_capacity {
            self.resize_cell_buffers(device, cell_count * 2);
        }
        
        // Detect changes via cell count
        if cell_count != self.last_cell_count {
            self.mark_all_dirty();
            self.last_cell_count = cell_count;
        }
        
        if cell_count == 0 {
            return;
        }
        
        // Update positions (convert Vec3 to vec4 for GPU alignment)
        let positions_vec4: Vec<[f32; 4]> = state.positions[..cell_count]
            .iter()
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();
        queue.write_buffer(&self.positions_buffer, 0, bytemuck::cast_slice(&positions_vec4));
        
        // Update rotations
        let rotations: Vec<[f32; 4]> = state.rotations[..cell_count]
            .iter()
            .map(|q| [q.x, q.y, q.z, q.w])
            .collect();
        queue.write_buffer(&self.rotations_buffer, 0, bytemuck::cast_slice(&rotations));
        
        // Update radii
        queue.write_buffer(&self.radii_buffer, 0, bytemuck::cast_slice(&state.radii[..cell_count]));
        
        // Update mode indices
        let mode_indices: Vec<u32> = state.mode_indices[..cell_count]
            .iter()
            .map(|&i| i as u32)
            .collect();
        queue.write_buffer(&self.mode_indices_buffer, 0, bytemuck::cast_slice(&mode_indices));
        
        // Update cell IDs
        queue.write_buffer(&self.cell_ids_buffer, 0, bytemuck::cast_slice(&state.cell_ids[..cell_count]));
        
        // Update genome IDs
        let genome_ids: Vec<u32> = state.genome_ids[..cell_count]
            .iter()
            .map(|&i| i as u32)
            .collect();
        queue.write_buffer(&self.genome_ids_buffer, 0, bytemuck::cast_slice(&genome_ids));

        // Update mode visuals from genome
        if let Some(genome) = genome {
            let mode_hash = Self::hash_genome_modes(genome);
            if mode_hash != self.last_mode_hash {
                self.update_mode_visuals(device, queue, genome);
                self.last_mode_hash = mode_hash;
            }
        }
        
        // Update cell type visuals
        if let Some(visuals) = cell_type_visuals {
            let visuals_hash = Self::hash_cell_type_visuals(visuals);
            if visuals_hash != self.last_visuals_hash {
                self.update_cell_type_visuals(device, queue, visuals);
                self.last_visuals_hash = visuals_hash;
            }
        }
        
        // Recreate bind group if needed
        if self.bind_group.is_none() {
            self.recreate_bind_group(device);
        }
    }
    
    fn hash_genome_modes(genome: &Genome) -> u64 {
        let mut hash = genome.modes.len() as u64;
        for mode in &genome.modes {
            hash = hash.wrapping_mul(31).wrapping_add((mode.color.x * 1000.0) as u64);
            hash = hash.wrapping_mul(31).wrapping_add((mode.opacity * 1000.0) as u64);
        }
        hash
    }
    
    fn hash_cell_type_visuals(visuals: &[CellTypeVisuals]) -> u64 {
        let mut hash = visuals.len() as u64;
        for v in visuals {
            hash = hash.wrapping_mul(31).wrapping_add((v.specular_strength * 1000.0) as u64);
            hash = hash.wrapping_mul(31).wrapping_add((v.membrane_noise_scale * 1000.0) as u64);
        }
        hash
    }

    fn update_mode_visuals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, genome: &Genome) {
        let mode_count = genome.modes.len();
        
        if mode_count > self.mode_capacity {
            self.mode_capacity = mode_count * 2;
            self.mode_visuals_buffer = Self::create_storage_buffer(
                device,
                "Mode Visuals",
                self.mode_capacity * std::mem::size_of::<ModeVisuals>(),
            );
            self.bind_group = None;
        }
        
        let mode_visuals: Vec<ModeVisuals> = genome.modes
            .iter()
            .map(|mode| ModeVisuals {
                color: mode.color.to_array(),
                opacity: mode.opacity,
                emissive: mode.emissive,
                _pad: [0.0; 3],
            })
            .collect();
        
        if !mode_visuals.is_empty() {
            queue.write_buffer(&self.mode_visuals_buffer, 0, bytemuck::cast_slice(&mode_visuals));
        }
    }
    
    fn update_cell_type_visuals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, visuals: &[CellTypeVisuals]) {
        let count = visuals.len();
        
        if count > self.cell_type_capacity {
            self.cell_type_capacity = count * 2;
            self.cell_type_visuals_buffer = Self::create_storage_buffer(
                device,
                "Cell Type Visuals",
                self.cell_type_capacity * std::mem::size_of::<GpuCellTypeVisuals>(),
            );
            self.bind_group = None;
        }
        
        let gpu_visuals: Vec<GpuCellTypeVisuals> = visuals
            .iter()
            .map(|v| GpuCellTypeVisuals {
                specular_strength: v.specular_strength,
                specular_power: v.specular_power,
                fresnel_strength: v.fresnel_strength,
                membrane_noise_scale: v.membrane_noise_scale,
                membrane_noise_strength: v.membrane_noise_strength,
                membrane_noise_speed: v.membrane_noise_speed,
                _pad: [0.0; 2],
            })
            .collect();
        
        if !gpu_visuals.is_empty() {
            queue.write_buffer(&self.cell_type_visuals_buffer, 0, bytemuck::cast_slice(&gpu_visuals));
        }
    }

    fn resize_cell_buffers(&mut self, device: &wgpu::Device, new_capacity: usize) {
        self.cell_capacity = new_capacity;
        
        self.positions_buffer = Self::create_storage_buffer(device, "Positions", new_capacity * 16);
        self.rotations_buffer = Self::create_storage_buffer(device, "Rotations", new_capacity * 16);
        self.radii_buffer = Self::create_storage_buffer(device, "Radii", new_capacity * 4);
        self.mode_indices_buffer = Self::create_storage_buffer(device, "Mode Indices", new_capacity * 4);
        self.cell_ids_buffer = Self::create_storage_buffer(device, "Cell IDs", new_capacity * 4);
        self.genome_ids_buffer = Self::create_storage_buffer(device, "Genome IDs", new_capacity * 4);
        
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Builder Output"),
            size: (new_capacity * std::mem::size_of::<CellInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.bind_group = None;
        self.mark_all_dirty();
    }
    
    /// Create a dummy 1x1 Hi-Z texture for when occlusion culling is disabled.
    fn create_dummy_hiz_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Hi-Z Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn recreate_bind_group(&mut self, device: &wgpu::Device) {
        // Use existing Hi-Z texture or create a dummy one
        let (_dummy_texture, dummy_view);
        let hiz_view = if let Some(ref view) = self.hiz_view {
            view
        } else {
            (_dummy_texture, dummy_view) = Self::create_dummy_hiz_texture(device);
            &dummy_view
        };
        
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance Builder Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rotations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mode_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.cell_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.genome_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.mode_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cell_type_visuals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.counters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
            ],
        }));
    }

    /// Extract frustum planes from a view-projection matrix.
    /// Note: The shader now uses a simpler clip-space method, but we still
    /// populate this for potential future use or debugging.
    fn extract_frustum_planes(_view_proj: Mat4) -> [FrustumPlane; 6] {
        // Return zeroed planes - the shader uses clip-space culling instead
        [FrustumPlane { normal_and_dist: [0.0; 4] }; 6]
    }
    
    /// Run the compute shader to build instance data with culling.
    /// Uses an external encoder to allow batching with other GPU work.
    pub fn build_instances_with_encoder(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cell_count: usize,
        mode_count: usize,
        cell_type_count: usize,
        view_proj: Mat4,
        camera_pos: Vec3,
        screen_width: u32,
        screen_height: u32,
    ) {
        if cell_count == 0 {
            self.last_visible_count = 0;
            return;
        }
        
        // Extract frustum planes
        let frustum_planes = Self::extract_frustum_planes(view_proj);
        
        // Update params
        let params = BuildParams {
            cell_count: cell_count as u32,
            mode_count: mode_count as u32,
            cell_type_count: cell_type_count as u32,
            culling_enabled: self.culling_mode.as_u32(),
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            near_plane: 0.1,
            far_plane: 1000.0,
            screen_width: screen_width as f32,
            screen_height: screen_height as f32,
            hiz_mip_count: self.hiz_mip_count,
            occlusion_bias: self.occlusion_bias,
            occlusion_mip_override: self.occlusion_mip_override,
            min_screen_size: self.min_screen_size,
            min_distance: self.min_distance,
            frustum_planes,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        
        // Clear counters
        let zero_counters: [u32; 4] = [0, 0, 0, 0];
        queue.write_buffer(&self.counters_buffer, 0, bytemuck::cast_slice(&zero_counters));
        
        // Initialize indirect draw buffer: vertex_count=6 (quad), instance_count=0 (will be set by shader), first_vertex=0, first_instance=0
        let indirect_data: [u32; 4] = [6, 0, 0, 0];
        queue.write_buffer(&self.indirect_buffer, 0, bytemuck::cast_slice(&indirect_data));

        // Ensure bind group exists
        if self.bind_group.is_none() {
            // Can't recreate bind group without device - caller must ensure it exists
            return;
        }
        
        let bind_group = self.bind_group.as_ref().unwrap();
        
        // Run compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Instance Builder Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            
            // Dispatch with workgroup size of 64
            let workgroup_count = (cell_count as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // Copy visible count (counters[0]) to indirect buffer's instance_count field (offset 4)
        encoder.copy_buffer_to_buffer(
            &self.counters_buffer,
            0,  // counters[0] = visible count
            &self.indirect_buffer,
            4,  // instance_count field offset
            4,  // 4 bytes (u32)
        );
        
        // Copy counters to readback buffer for stats
        encoder.copy_buffer_to_buffer(
            &self.counters_buffer,
            0,
            &self.counters_readback_buffer,
            0,
            16,
        );
        
        // For stats display, assume all cells visible (actual count is in GPU buffer)
        self.last_visible_count = cell_count as u32;
        self.last_stats = CullingStats {
            total_cells: cell_count as u32,
            visible_cells: cell_count as u32,
            frustum_culled: 0,
            occluded: 0,
        };
    }
    
    /// Run the compute shader to build instance data with culling.
    /// Creates its own encoder and submits - use build_instances_with_encoder for batching.
    #[allow(dead_code)]
    pub fn build_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell_count: usize,
        mode_count: usize,
        cell_type_count: usize,
        view_proj: Mat4,
        camera_pos: Vec3,
        screen_width: u32,
        screen_height: u32,
    ) {
        // Ensure bind group exists before creating encoder
        if self.bind_group.is_none() {
            self.recreate_bind_group(device);
        }
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Instance Builder Encoder"),
        });
        
        self.build_instances_with_encoder(
            &mut encoder,
            queue,
            cell_count,
            mode_count,
            cell_type_count,
            view_proj,
            camera_pos,
            screen_width,
            screen_height,
        );
        
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Read back culling statistics (call after GPU work is done).
    /// This is a blocking operation - use sparingly.
    #[allow(dead_code)]
    pub fn read_culling_stats_blocking(&mut self, device: &wgpu::Device) -> CullingStats {
        let buffer_slice = self.counters_readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        // Poll the device in a blocking manner
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        
        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let counters: &[u32] = bytemuck::cast_slice(&data);
            
            self.last_stats = CullingStats {
                visible_cells: counters[0],
                total_cells: counters[1],
                frustum_culled: counters[2],
                occluded: counters[3],
            };
            self.last_visible_count = counters[0];
            
            drop(data);
            self.counters_readback_buffer.unmap();
        }
        
        self.last_stats
    }
    
    /// Get the instance buffer for rendering.
    pub fn get_instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }
    
    /// Get the indirect draw buffer for GPU-driven rendering.
    pub fn get_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_buffer
    }
    
    /// Get current cell capacity.
    pub fn capacity(&self) -> usize {
        self.cell_capacity
    }
}
