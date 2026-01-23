//! Surface Nets algorithm for fluid mesh generation
//!
//! Extracts smooth isosurfaces from the fluid voxel grid.
//! Surface nets produces smoother meshes than marching cubes with simpler topology.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::simulation::fluid_simulation::{GRID_RESOLUTION, TOTAL_VOXELS};

/// Vertex data for the fluid mesh
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FluidVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub fluid_type: u32,
    pub _padding: u32,
}

/// Parameters for surface nets extraction
#[derive(Clone, Debug)]
pub struct SurfaceNetsParams {
    /// Isosurface threshold (fill fraction at which surface exists)
    pub iso_level: f32,
    /// World center position
    pub world_center: Vec3,
    /// World radius
    pub world_radius: f32,
    /// Minimum fill fraction to consider as fluid
    pub min_fill: f32,
}

impl Default for SurfaceNetsParams {
    fn default() -> Self {
        Self {
            iso_level: 0.5,
            world_center: Vec3::ZERO,
            world_radius: 100.0,
            min_fill: 0.01,
        }
    }
}

/// Surface nets mesh extractor
pub struct SurfaceNets {
    params: SurfaceNetsParams,
    /// Cell size computed from world parameters
    cell_size: f32,
    /// Grid origin (world position of voxel 0,0,0)
    grid_origin: Vec3,
}

impl SurfaceNets {
    /// Create a new surface nets extractor
    pub fn new(params: SurfaceNetsParams) -> Self {
        let world_diameter = params.world_radius * 2.0;
        let cell_size = world_diameter / GRID_RESOLUTION as f32;
        let grid_origin = params.world_center - Vec3::splat(world_diameter / 2.0);
        
        Self {
            params,
            cell_size,
            grid_origin,
        }
    }
    
    /// Update parameters (e.g., when world size changes)
    pub fn update_params(&mut self, params: SurfaceNetsParams) {
        let world_diameter = params.world_radius * 2.0;
        self.cell_size = world_diameter / GRID_RESOLUTION as f32;
        self.grid_origin = params.world_center - Vec3::splat(world_diameter / 2.0);
        self.params = params;
    }
    
    /// Convert grid indices to linear index
    #[inline]
    fn grid_index(x: u32, y: u32, z: u32) -> usize {
        (x + y * GRID_RESOLUTION + z * GRID_RESOLUTION * GRID_RESOLUTION) as usize
    }
    
    /// Convert grid indices to world position (cell center)
    #[inline]
    fn grid_to_world(&self, x: u32, y: u32, z: u32) -> Vec3 {
        self.grid_origin + Vec3::new(
            (x as f32 + 0.5) * self.cell_size,
            (y as f32 + 0.5) * self.cell_size,
            (z as f32 + 0.5) * self.cell_size,
        )
    }
    
    /// Extract mesh from voxel data
    /// 
    /// `voxel_data` should contain TOTAL_VOXELS entries of (fluid_type, fill_fraction)
    /// Returns (vertices, indices) for the mesh
    pub fn extract_mesh(
        &self,
        voxel_types: &[u32],
        fill_fractions: &[f32],
        fluid_type_filter: Option<u32>,
    ) -> (Vec<FluidVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        // Map from voxel index to vertex index
        let mut vertex_map: HashMap<usize, u32> = HashMap::new();
        
        let iso = self.params.iso_level;
        let res = GRID_RESOLUTION;
        
        // Phase 1: Generate vertices for each cell that contains the isosurface
        for z in 0..res - 1 {
            for y in 0..res - 1 {
                for x in 0..res - 1 {
                    // Get fill values at the 8 corners of this cell
                    let corners = self.get_cell_corners(x, y, z, voxel_types, fill_fractions, fluid_type_filter);
                    
                    // Check if this cell contains the isosurface
                    let (has_surface, vertex_pos) = self.compute_surface_vertex(&corners, x, y, z, iso);
                    
                    if has_surface {
                        let idx = Self::grid_index(x, y, z);
                        let vertex_idx = vertices.len() as u32;
                        vertex_map.insert(idx, vertex_idx);
                        
                        // Compute normal from gradient
                        let normal = self.compute_normal(x, y, z, fill_fractions, fluid_type_filter, voxel_types);
                        
                        // Get the dominant fluid type for this cell
                        let fluid_type = self.get_dominant_fluid_type(x, y, z, voxel_types, fill_fractions);
                        
                        vertices.push(FluidVertex {
                            position: vertex_pos.to_array(),
                            normal: normal.to_array(),
                            fluid_type,
                            _padding: 0,
                        });
                    }
                }
            }
        }
        
        // Phase 2: Generate quads connecting adjacent cells
        for z in 0..res - 1 {
            for y in 0..res - 1 {
                for x in 0..res - 1 {
                    let idx = Self::grid_index(x, y, z);
                    
                    // Skip if this cell doesn't have a vertex
                    if !vertex_map.contains_key(&idx) {
                        continue;
                    }
                    
                    // Get fill values at corners
                    let corners = self.get_cell_corners(x, y, z, voxel_types, fill_fractions, fluid_type_filter);
                    
                    // Check each of the 3 edges from corner 0 (at x,y,z)
                    // The winding depends on whether corner 0 is inside or outside
                    // Invert the flip to fix winding order (normals should face outward)
                    let corner0_inside = corners[0] >= iso;
                    let flip = !corner0_inside; // Inverted for correct outward-facing normals
                    
                    // Edge along X axis (0-1)
                    if (corners[0] < iso) != (corners[1] < iso) {
                        self.try_create_quad_x(x, y, z, flip, &vertex_map, &mut indices);
                    }
                    
                    // Edge along Y axis (0-2)
                    if (corners[0] < iso) != (corners[2] < iso) {
                        self.try_create_quad_y(x, y, z, flip, &vertex_map, &mut indices);
                    }
                    
                    // Edge along Z axis (0-4)
                    if (corners[0] < iso) != (corners[4] < iso) {
                        self.try_create_quad_z(x, y, z, flip, &vertex_map, &mut indices);
                    }
                }
            }
        }
        
        (vertices, indices)
    }
    
    /// Get fill fractions at the 8 corners of a cell
    fn get_cell_corners(
        &self,
        x: u32,
        y: u32,
        z: u32,
        voxel_types: &[u32],
        fill_fractions: &[f32],
        fluid_type_filter: Option<u32>,
    ) -> [f32; 8] {
        let mut corners = [0.0f32; 8];
        
        // Corner ordering:
        // 0: (x,   y,   z  )
        // 1: (x+1, y,   z  )
        // 2: (x,   y+1, z  )
        // 3: (x+1, y+1, z  )
        // 4: (x,   y,   z+1)
        // 5: (x+1, y,   z+1)
        // 6: (x,   y+1, z+1)
        // 7: (x+1, y+1, z+1)
        
        let offsets: [(u32, u32, u32); 8] = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ];
        
        for (i, (dx, dy, dz)) in offsets.iter().enumerate() {
            let idx = Self::grid_index(x + dx, y + dy, z + dz);
            if idx < TOTAL_VOXELS {
                let vtype = voxel_types[idx];
                let fill = fill_fractions[idx];
                
                // Filter by fluid type if specified
                if let Some(filter) = fluid_type_filter {
                    if vtype == filter && fill > self.params.min_fill {
                        corners[i] = fill;
                    }
                } else {
                    // Include all non-empty, non-solid fluid types
                    // 0 = Empty, 4 = Solid
                    if vtype != 0 && vtype != 4 && fill > self.params.min_fill {
                        corners[i] = fill;
                    }
                }
            }
        }
        
        corners
    }
    
    /// Compute vertex position for a cell containing the isosurface
    fn compute_surface_vertex(
        &self,
        corners: &[f32; 8],
        x: u32,
        y: u32,
        z: u32,
        iso: f32,
    ) -> (bool, Vec3) {
        // Determine which corners are inside/outside the isosurface
        let mut inside_count = 0;
        let mut outside_count = 0;
        
        for &c in corners.iter() {
            if c >= iso {
                inside_count += 1;
            } else {
                outside_count += 1;
            }
        }
        
        // If all corners are on the same side, no surface in this cell
        if inside_count == 0 || outside_count == 0 {
            return (false, Vec3::ZERO);
        }
        
        // Find edge crossings and average their positions
        let mut sum = Vec3::ZERO;
        let mut count = 0;
        
        // Edge table: pairs of corner indices for the 12 edges
        const EDGES: [(usize, usize, Vec3); 12] = [
            // X edges
            (0, 1, Vec3::new(1.0, 0.0, 0.0)),
            (2, 3, Vec3::new(1.0, 0.0, 0.0)),
            (4, 5, Vec3::new(1.0, 0.0, 0.0)),
            (6, 7, Vec3::new(1.0, 0.0, 0.0)),
            // Y edges
            (0, 2, Vec3::new(0.0, 1.0, 0.0)),
            (1, 3, Vec3::new(0.0, 1.0, 0.0)),
            (4, 6, Vec3::new(0.0, 1.0, 0.0)),
            (5, 7, Vec3::new(0.0, 1.0, 0.0)),
            // Z edges
            (0, 4, Vec3::new(0.0, 0.0, 1.0)),
            (1, 5, Vec3::new(0.0, 0.0, 1.0)),
            (2, 6, Vec3::new(0.0, 0.0, 1.0)),
            (3, 7, Vec3::new(0.0, 0.0, 1.0)),
        ];
        
        // Corner positions relative to cell origin
        const CORNER_OFFSETS: [Vec3; 8] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        
        for (i0, i1, _dir) in EDGES.iter() {
            let v0 = corners[*i0];
            let v1 = corners[*i1];
            
            // Check if this edge crosses the isosurface
            if (v0 >= iso) != (v1 >= iso) {
                // Interpolate position along edge
                let t = if (v1 - v0).abs() > 1e-6 {
                    (iso - v0) / (v1 - v0)
                } else {
                    0.5
                };
                let t = t.clamp(0.0, 1.0);
                
                let p0 = CORNER_OFFSETS[*i0];
                let p1 = CORNER_OFFSETS[*i1];
                let edge_pos = p0 + (p1 - p0) * t;
                
                sum += edge_pos;
                count += 1;
            }
        }
        
        if count == 0 {
            return (false, Vec3::ZERO);
        }
        
        // Average position in cell-local coordinates [0,1]
        let local_pos = sum / count as f32;
        
        // Convert to world coordinates
        let cell_origin = self.grid_origin + Vec3::new(x as f32, y as f32, z as f32) * self.cell_size;
        let world_pos = cell_origin + local_pos * self.cell_size;
        
        (true, world_pos)
    }
    
    /// Compute normal at a grid position using central differences
    fn compute_normal(
        &self,
        x: u32,
        y: u32,
        z: u32,
        fill_fractions: &[f32],
        fluid_type_filter: Option<u32>,
        voxel_types: &[u32],
    ) -> Vec3 {
        let sample = |sx: i32, sy: i32, sz: i32| -> f32 {
            let nx = (x as i32 + sx).clamp(0, GRID_RESOLUTION as i32 - 1) as u32;
            let ny = (y as i32 + sy).clamp(0, GRID_RESOLUTION as i32 - 1) as u32;
            let nz = (z as i32 + sz).clamp(0, GRID_RESOLUTION as i32 - 1) as u32;
            let idx = Self::grid_index(nx, ny, nz);
            
            if idx >= TOTAL_VOXELS {
                return 0.0;
            }
            
            let vtype = voxel_types[idx];
            let fill = fill_fractions[idx];
            
            if let Some(filter) = fluid_type_filter {
                if vtype == filter {
                    return fill;
                }
            } else if vtype != 0 && vtype != 4 {
                return fill;
            }
            0.0
        };
        
        // Central differences for gradient
        let gx = sample(1, 0, 0) - sample(-1, 0, 0);
        let gy = sample(0, 1, 0) - sample(0, -1, 0);
        let gz = sample(0, 0, 1) - sample(0, 0, -1);
        
        // Normal points from high density to low density (outward from fluid)
        let normal = Vec3::new(-gx, -gy, -gz);
        
        if normal.length_squared() > 1e-10 {
            normal.normalize()
        } else {
            Vec3::Y // Default up if gradient is zero
        }
    }
    
    /// Get the dominant fluid type for a cell (the type with highest fill)
    fn get_dominant_fluid_type(
        &self,
        x: u32,
        y: u32,
        z: u32,
        voxel_types: &[u32],
        fill_fractions: &[f32],
    ) -> u32 {
        // Check all 8 corners and return the fluid type with the highest fill
        let offsets: [(u32, u32, u32); 8] = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ];
        
        let mut best_type = 1u32; // Default to water
        let mut best_fill = 0.0f32;
        
        for (dx, dy, dz) in offsets.iter() {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            
            if nx < GRID_RESOLUTION && ny < GRID_RESOLUTION && nz < GRID_RESOLUTION {
                let idx = Self::grid_index(nx, ny, nz);
                if idx < TOTAL_VOXELS {
                    let vtype = voxel_types[idx];
                    let fill = fill_fractions[idx];
                    
                    // Only consider fluid types (1=water, 2=lava, 3=steam), not empty(0) or solid(4)
                    if vtype >= 1 && vtype <= 3 && fill > best_fill {
                        best_fill = fill;
                        best_type = vtype;
                    }
                }
            }
        }
        
        best_type
    }
    
    /// Try to create a quad for an X-axis edge crossing
    fn try_create_quad_x(
        &self,
        x: u32,
        y: u32,
        z: u32,
        flip: bool,
        vertex_map: &HashMap<usize, u32>,
        indices: &mut Vec<u32>,
    ) {
        if y == 0 || z == 0 {
            return;
        }
        
        // Get the 4 cells that share this edge
        let idx0 = Self::grid_index(x, y, z);
        let idx1 = Self::grid_index(x, y - 1, z);
        let idx2 = Self::grid_index(x, y - 1, z - 1);
        let idx3 = Self::grid_index(x, y, z - 1);
        
        if let (Some(&v0), Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&idx0),
            vertex_map.get(&idx1),
            vertex_map.get(&idx2),
            vertex_map.get(&idx3),
        ) {
            // Two triangles for the quad - flip winding based on inside/outside
            if flip {
                indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
            } else {
                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
            }
        }
    }
    
    /// Try to create a quad for a Y-axis edge crossing
    fn try_create_quad_y(
        &self,
        x: u32,
        y: u32,
        z: u32,
        flip: bool,
        vertex_map: &HashMap<usize, u32>,
        indices: &mut Vec<u32>,
    ) {
        if x == 0 || z == 0 {
            return;
        }
        
        let idx0 = Self::grid_index(x, y, z);
        let idx1 = Self::grid_index(x, y, z - 1);
        let idx2 = Self::grid_index(x - 1, y, z - 1);
        let idx3 = Self::grid_index(x - 1, y, z);
        
        if let (Some(&v0), Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&idx0),
            vertex_map.get(&idx1),
            vertex_map.get(&idx2),
            vertex_map.get(&idx3),
        ) {
            // Flip winding based on inside/outside
            if flip {
                indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
            } else {
                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
            }
        }
    }
    
    /// Try to create a quad for a Z-axis edge crossing
    fn try_create_quad_z(
        &self,
        x: u32,
        y: u32,
        z: u32,
        flip: bool,
        vertex_map: &HashMap<usize, u32>,
        indices: &mut Vec<u32>,
    ) {
        if x == 0 || y == 0 {
            return;
        }
        
        let idx0 = Self::grid_index(x, y, z);
        let idx1 = Self::grid_index(x - 1, y, z);
        let idx2 = Self::grid_index(x - 1, y - 1, z);
        let idx3 = Self::grid_index(x, y - 1, z);
        
        if let (Some(&v0), Some(&v1), Some(&v2), Some(&v3)) = (
            vertex_map.get(&idx0),
            vertex_map.get(&idx1),
            vertex_map.get(&idx2),
            vertex_map.get(&idx3),
        ) {
            // Flip winding based on inside/outside
            if flip {
                indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
            } else {
                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
            }
        }
    }
}

/// Fluid mesh renderer using surface nets
pub struct FluidMeshRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    surface_nets: SurfaceNets,
    vertex_count: u32,
    index_count: u32,
    max_vertices: u32,
    max_indices: u32,
    /// Screen dimensions
    pub width: u32,
    pub height: u32,
}

/// Camera uniform for fluid mesh shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

/// Fluid rendering parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FluidMeshParams {
    /// Base color for water (RGBA)
    pub water_color: [f32; 4],
    /// Base color for lava (RGBA)
    pub lava_color: [f32; 4],
    /// Base color for steam (RGBA)
    pub steam_color: [f32; 4],
    
    // Lighting parameters (16 bytes = vec4)
    /// Ambient light strength
    pub ambient: f32,
    /// Diffuse light strength
    pub diffuse: f32,
    /// Specular intensity
    pub specular_intensity: f32,
    /// Shininess (specular power)
    pub shininess: f32,
    
    // More lighting (16 bytes = vec4)
    /// Fresnel effect strength
    pub fresnel_strength: f32,
    /// Fresnel power (higher = sharper edge effect)
    pub fresnel_power: f32,
    /// Reflection strength (environment reflection simulation)
    pub reflection: f32,
    /// Overall alpha/opacity
    pub alpha: f32,
    
    // Emissive and misc (16 bytes = vec4)
    /// Emissive strength (for lava glow)
    pub emissive: f32,
    /// Rim light strength
    pub rim_strength: f32,
    /// Fluid type (unused now, per-vertex)
    pub fluid_type: u32,
    /// SSR enabled flag (as f32 for alignment)
    pub ssr_enabled: f32,
    
    // SSR parameters (16 bytes = vec4)
    /// SSR ray march steps
    pub ssr_max_steps: f32,
    /// SSR ray step size
    pub ssr_step_size: f32,
    /// SSR max distance
    pub ssr_max_distance: f32,
    /// SSR thickness threshold
    pub ssr_thickness: f32,
    
    // SSR quality (16 bytes = vec4)
    /// SSR intensity/strength
    pub ssr_intensity: f32,
    /// SSR fade distance start
    pub ssr_fade_start: f32,
    /// SSR fade distance end
    pub ssr_fade_end: f32,
    /// SSR roughness blur
    pub ssr_roughness: f32,
}

impl Default for FluidMeshParams {
    fn default() -> Self {
        Self {
            water_color: [0.2, 0.5, 0.9, 0.8],
            lava_color: [1.0, 0.3, 0.0, 0.95],
            steam_color: [0.9, 0.9, 0.9, 0.4],
            ambient: 0.15,
            diffuse: 0.6,
            specular_intensity: 0.8,
            shininess: 64.0,
            fresnel_strength: 0.5,
            fresnel_power: 3.0,
            reflection: 0.3,
            alpha: 0.8,
            emissive: 0.4,
            rim_strength: 0.5,
            fluid_type: 1,
            ssr_enabled: 1.0,
            ssr_max_steps: 32.0,
            ssr_step_size: 0.1,
            ssr_max_distance: 50.0,
            ssr_thickness: 0.5,
            ssr_intensity: 0.8,
            ssr_fade_start: 0.6,
            ssr_fade_end: 1.0,
            ssr_roughness: 0.1,
        }
    }
}

impl FluidMeshRenderer {
    /// Create a new fluid mesh renderer
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        world_radius: f32,
        world_center: Vec3,
        width: u32,
        height: u32,
    ) -> Self {
        // Create surface nets extractor
        let params = SurfaceNetsParams {
            iso_level: 0.5,
            world_center,
            world_radius,
            min_fill: 0.01,
        };
        let surface_nets = SurfaceNets::new(params);
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fluid/fluid_mesh.wgsl").into()),
        });
        
        // Camera bind group layout
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Mesh Camera Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Params bind group layout
        let params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Mesh Params Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Mesh Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &params_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fluid Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<FluidVertex>() as u64,
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
                            format: wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
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
                format: depth_format,
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
        
        // Allocate buffers with reasonable capacity
        let max_vertices = 500_000u32;
        let max_indices = 1_500_000u32;
        
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Mesh Vertex Buffer"),
            size: (max_vertices as usize * std::mem::size_of::<FluidVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Mesh Index Buffer"),
            size: (max_indices as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create camera buffer and bind group
        let camera_uniform = CameraUniform {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };
        
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Mesh Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Mesh Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        
        // Create params buffer and bind group
        let params = FluidMeshParams::default();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Mesh Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Mesh Params Bind Group"),
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });
        
        Self {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            camera_bind_group,
            params_buffer,
            params_bind_group,
            surface_nets,
            vertex_count: 0,
            index_count: 0,
            max_vertices,
            max_indices,
            width,
            height,
        }
    }
    
    /// Update mesh from voxel data
    pub fn update_mesh(
        &mut self,
        queue: &wgpu::Queue,
        voxel_types: &[u32],
        fill_fractions: &[f32],
        fluid_type_filter: Option<u32>,
    ) {
        let (vertices, indices) = self.surface_nets.extract_mesh(voxel_types, fill_fractions, fluid_type_filter);
        
        self.vertex_count = (vertices.len() as u32).min(self.max_vertices);
        self.index_count = (indices.len() as u32).min(self.max_indices);
        
        if self.vertex_count > 0 {
            queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&vertices[..self.vertex_count as usize]),
            );
        }
        
        if self.index_count > 0 {
            queue.write_buffer(
                &self.index_buffer,
                0,
                bytemuck::cast_slice(&indices[..self.index_count as usize]),
            );
        }
        
        log::trace!(
            "Fluid mesh updated: {} vertices, {} indices",
            self.vertex_count,
            self.index_count
        );
    }
    
    /// Update rendering parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &FluidMeshParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[*params]));
    }
    
    /// Update surface nets parameters
    pub fn update_surface_nets_params(&mut self, params: SurfaceNetsParams) {
        self.surface_nets.update_params(params);
    }
    
    /// Render the fluid mesh
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
    ) {
        if self.index_count == 0 {
            return;
        }
        
        // Update camera uniform
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
        
        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Fluid Mesh Render Pass"),
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
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.params_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
    
    /// Get vertex count
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    
    /// Get index count (triangle count * 3)
    pub fn index_count(&self) -> u32 {
        self.index_count
    }
    
    /// Get triangle count
    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }
    
    /// Resize for new screen dimensions
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
