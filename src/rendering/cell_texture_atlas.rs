//! Cell texture atlas system for LOD-based rendering.
//!
//! Provides pre-generated cell textures at multiple LOD levels (32x32, 64x64, 128x128, 256x256)
//! to replace expensive procedural rendering with fast texture sampling.

/// LOD levels for cell textures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    Low = 0,    // 32x32
    Medium = 1, // 64x64
    High = 2,   // 128x128
    Ultra = 3,  // 256x256
}

impl LodLevel {
    /// Get texture size for this LOD level
    pub fn texture_size(self) -> u32 {
        match self {
            LodLevel::Low => 32,
            LodLevel::Medium => 64,
            LodLevel::High => 128,
            LodLevel::Ultra => 256,
        }
    }
    
    /// Select LOD level based on screen radius with configurable thresholds
    pub fn from_screen_radius_with_thresholds(
        screen_radius: f32, 
        threshold_low: f32, 
        threshold_medium: f32, 
        threshold_high: f32
    ) -> Self {
        if screen_radius < threshold_low {
            LodLevel::Low
        } else if screen_radius < threshold_medium {
            LodLevel::Medium
        } else if screen_radius < threshold_high {
            LodLevel::High
        } else {
            LodLevel::Ultra
        }
    }
    
    /// Select LOD level based on screen radius with wider thresholds (legacy)
    pub fn from_screen_radius(screen_radius: f32) -> Self {
        Self::from_screen_radius_with_thresholds(screen_radius, 10.0, 25.0, 50.0)
    }
}

/// Cell texture atlas containing all LOD levels for all cell types
pub struct CellTextureAtlas {
    /// Atlas texture containing all cell textures
    pub texture: wgpu::Texture,
    /// Texture view for shader binding
    pub view: wgpu::TextureView,
    /// Sampler for texture filtering
    pub sampler: wgpu::Sampler,
    /// Atlas dimensions
    pub width: u32,
    pub height: u32,
}

impl CellTextureAtlas {
    /// Create a new cell texture atlas
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Atlas layout: 4 LODs horizontally, 2 cell types vertically
        // Total size: 1024x512 (4 * 256 x 2 * 256)
        let atlas_width = 1024;
        let atlas_height = 512;
        
        // Create atlas texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cell Texture Atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cell Texture Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,   // Smooth edges
            min_filter: wgpu::FilterMode::Linear,   // Smooth edges
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        let atlas = Self {
            texture,
            view,
            sampler,
            width: atlas_width,
            height: atlas_height,
        };
        
        // Generate all cell textures
        atlas.generate_textures(device, queue);
        
        atlas
    }
    
    /// Generate all cell textures and upload to atlas
    fn generate_textures(&self, _device: &wgpu::Device, queue: &wgpu::Queue) {
        // Generate textures for each cell type and LOD level
        for cell_type in 0..2 {
            for lod_level in 0..4 {
                let lod = match lod_level {
                    0 => LodLevel::Low,
                    1 => LodLevel::Medium,
                    2 => LodLevel::High,
                    3 => LodLevel::Ultra,
                    _ => unreachable!(),
                };
                
                let texture_data = match cell_type {
                    0 => self.generate_test_cell_texture(lod),
                    1 => self.generate_flagellocyte_cell_texture(lod),
                    _ => unreachable!(),
                };
                
                // Calculate position in atlas - center texture within 256x256 slot
                let slot_size = 256u32;
                let texture_size = lod.texture_size();
                let x_offset = lod_level * slot_size;
                let y_offset = cell_type * slot_size;
                
                // Center the texture within the slot
                let x_padding = (slot_size - texture_size) / 2;
                let y_padding = (slot_size - texture_size) / 2;
                
                // Upload texture data to atlas
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: x_offset + x_padding,
                            y: y_offset + y_padding,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &texture_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(texture_size * 4), // RGBA = 4 bytes per pixel
                        rows_per_image: Some(texture_size),
                    },
                    wgpu::Extent3d {
                        width: texture_size,
                        height: texture_size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
    
    /// Generate Test cell texture (flat circle with smooth anti-aliased edge)
    fn generate_test_cell_texture(&self, lod: LodLevel) -> Vec<u8> {
        let size = lod.texture_size();
        let mut data = Vec::with_capacity((size * size * 4) as usize);
        
        let center = size as f32 * 0.5;
        let radius = center - 1.0; // Leave 1px for anti-aliasing
        
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center + 0.5;
                let dy = y as f32 - center + 0.5;
                let dist = (dx * dx + dy * dy).sqrt();
                
                // Smooth falloff at edge (1px transition)
                let alpha = (1.0 - (dist - radius).max(0.0)).clamp(0.0, 1.0);
                let alpha_byte = (alpha * 255.0) as u8;
                
                data.extend_from_slice(&[255, 255, 255, alpha_byte]);
            }
        }
        
        data
    }
    
    /// Generate Flagellocyte cell texture (same as Test cell for body)
    fn generate_flagellocyte_cell_texture(&self, lod: LodLevel) -> Vec<u8> {
        // For now, Flagellocyte body uses same texture as Test cell
        // The tail will be rendered as 3D geometry
        self.generate_test_cell_texture(lod)
    }
    
    /// Get UV coordinates for a specific cell type and LOD level
    pub fn get_uv_coords(&self, cell_type: u32, lod: LodLevel) -> (f32, f32, f32, f32) {
        let lod_index = lod as u32;
        let slot_size = 256.0; // Each slot is 256x256 pixels
        let actual_size = lod.texture_size() as f32;
        
        // Calculate slot position
        let x_offset = lod_index as f32 * slot_size;
        let y_offset = cell_type as f32 * slot_size;
        
        // Center the actual texture within the slot
        let x_padding = (slot_size - actual_size) * 0.5;
        let y_padding = (slot_size - actual_size) * 0.5;
        
        let uv_min_x = (x_offset + x_padding) / self.width as f32;
        let uv_min_y = (y_offset + y_padding) / self.height as f32;
        let uv_max_x = (x_offset + x_padding + actual_size) / self.width as f32;
        let uv_max_y = (y_offset + y_padding + actual_size) / self.height as f32;
        
        (uv_min_x, uv_min_y, uv_max_x, uv_max_y)
    }
}