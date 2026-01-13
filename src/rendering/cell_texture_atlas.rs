//! Cell texture atlas system for LOD-based sphere rendering.
//!
//! Loads 2048x1024 equirectangular textures and downsamples them to LOD levels
//! (256, 128, 64, 32) for efficient billboard rendering with sphere UV mapping.

use std::path::Path;

/// LOD levels for cell textures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    Low = 0,    // 32x16
    Medium = 1, // 64x32
    High = 2,   // 128x64
    Ultra = 3,  // 256x128
}

impl LodLevel {
    /// Get texture width for this LOD level (2:1 aspect ratio for equirectangular)
    pub fn texture_width(self) -> u32 {
        match self {
            LodLevel::Low => 32,
            LodLevel::Medium => 64,
            LodLevel::High => 128,
            LodLevel::Ultra => 256,
        }
    }
    
    /// Get texture height for this LOD level
    pub fn texture_height(self) -> u32 {
        self.texture_width() / 2
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
    /// Source texture dimensions
    const SOURCE_WIDTH: u32 = 2048;
    const SOURCE_HEIGHT: u32 = 1024;
    const NUM_CELL_TYPES: u32 = 2;
    const NUM_LODS: u32 = 4;
    
    /// Atlas slot size (largest LOD)
    const SLOT_WIDTH: u32 = 256;
    const SLOT_HEIGHT: u32 = 128;
    
    /// Create a new cell texture atlas
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Atlas layout: 4 LODs horizontally (256px each), 2 cell types vertically (128px each)
        // Total size: 1024x256
        let atlas_width = Self::SLOT_WIDTH * Self::NUM_LODS;
        let atlas_height = Self::SLOT_HEIGHT * Self::NUM_CELL_TYPES;
        
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
        
        // Repeat U for longitude wrap, clamp V for poles
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cell Texture Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
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
        
        atlas.load_textures(queue);
        
        atlas
    }
    
    /// Load all cell textures into the atlas
    fn load_textures(&self, queue: &wgpu::Queue) {
        for cell_type in 0..Self::NUM_CELL_TYPES {
            // Load source texture (2048x1024)
            let source_data = self.load_source_texture(cell_type);
            
            // Generate each LOD level
            for lod_idx in 0..Self::NUM_LODS {
                let lod = match lod_idx {
                    0 => LodLevel::Low,
                    1 => LodLevel::Medium,
                    2 => LodLevel::High,
                    3 => LodLevel::Ultra,
                    _ => unreachable!(),
                };
                
                // Width is always full slot for proper U wrapping, height varies by LOD
                let lod_width = Self::SLOT_WIDTH;
                let lod_height = lod.texture_height();
                
                // Downsample source to LOD size
                let lod_data = Self::downsample(&source_data, Self::SOURCE_WIDTH, Self::SOURCE_HEIGHT, lod_width, lod_height);
                
                // Calculate position in atlas - full width, centered vertically
                let x_offset = lod_idx * Self::SLOT_WIDTH;
                let y_offset = cell_type * Self::SLOT_HEIGHT;
                let y_padding = (Self::SLOT_HEIGHT - lod_height) / 2;
                
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: x_offset,
                            y: y_offset + y_padding,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &lod_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(lod_width * 4),
                        rows_per_image: Some(lod_height),
                    },
                    wgpu::Extent3d {
                        width: lod_width,
                        height: lod_height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
    
    /// Load source texture for a cell type (2048x1024)
    /// Naming scheme: assets/textures/cell_{cell_type}.png
    fn load_source_texture(&self, cell_type: u32) -> Vec<u8> {
        let filename = format!("assets/textures/cell_{}.png", cell_type);
        
        if let Some(data) = Self::load_texture_from_file(&filename) {
            log::info!("Loaded cell type {} texture from {}", cell_type, filename);
            return data;
        }
        
        log::warn!("Could not load {}, using procedural texture", filename);
        self.generate_procedural_equirect_texture()
    }
    
    /// Load texture from PNG file
    fn load_texture_from_file(path: &str) -> Option<Vec<u8>> {
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return None;
        }
        
        let img = match image::open(path_obj) {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to load texture {}: {}", path, e);
                return None;
            }
        };
        
        let (width, height) = (img.width(), img.height());
        
        if width != CellTextureAtlas::SOURCE_WIDTH || height != CellTextureAtlas::SOURCE_HEIGHT {
            log::warn!(
                "Texture {} has size {}x{}, expected {}x{} - will resize",
                path, width, height,
                CellTextureAtlas::SOURCE_WIDTH, CellTextureAtlas::SOURCE_HEIGHT
            );
            // Resize to expected dimensions
            let resized = img.resize_exact(
                CellTextureAtlas::SOURCE_WIDTH,
                CellTextureAtlas::SOURCE_HEIGHT,
                image::imageops::FilterType::Lanczos3
            );
            return Some(resized.to_rgba8().into_raw());
        }
        
        Some(img.to_rgba8().into_raw())
    }
    
    /// Downsample RGBA image using box filter (simple averaging)
    fn downsample(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
        let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
        
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        
        for dst_y in 0..dst_h {
            for dst_x in 0..dst_w {
                // Calculate source region to sample
                let src_x0 = (dst_x as f32 * scale_x) as u32;
                let src_y0 = (dst_y as f32 * scale_y) as u32;
                let src_x1 = ((dst_x + 1) as f32 * scale_x).ceil() as u32;
                let src_y1 = ((dst_y + 1) as f32 * scale_y).ceil() as u32;
                
                // Average all pixels in the source region
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut a_sum = 0u32;
                let mut count = 0u32;
                
                for sy in src_y0..src_y1.min(src_h) {
                    for sx in src_x0..src_x1.min(src_w) {
                        let idx = ((sy * src_w + sx) * 4) as usize;
                        r_sum += src[idx] as u32;
                        g_sum += src[idx + 1] as u32;
                        b_sum += src[idx + 2] as u32;
                        a_sum += src[idx + 3] as u32;
                        count += 1;
                    }
                }
                
                let dst_idx = ((dst_y * dst_w + dst_x) * 4) as usize;
                if count > 0 {
                    dst[dst_idx] = (r_sum / count) as u8;
                    dst[dst_idx + 1] = (g_sum / count) as u8;
                    dst[dst_idx + 2] = (b_sum / count) as u8;
                    dst[dst_idx + 3] = (a_sum / count) as u8;
                }
            }
        }
        
        dst
    }
    
    /// Generate procedural equirectangular texture (grid pattern for debugging)
    fn generate_procedural_equirect_texture(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity((Self::SOURCE_WIDTH * Self::SOURCE_HEIGHT * 4) as usize);
        
        for y in 0..Self::SOURCE_HEIGHT {
            for x in 0..Self::SOURCE_WIDTH {
                let u = x as f32 / Self::SOURCE_WIDTH as f32;
                let v = y as f32 / Self::SOURCE_HEIGHT as f32;
                
                // Longitude lines every 30 degrees (12 lines)
                let lon_line = ((u * 12.0).fract() < 0.02) as u8 * 80;
                // Latitude lines every 30 degrees (6 lines)
                let lat_line = ((v * 6.0).fract() < 0.02) as u8 * 80;
                
                let intensity = 180u8.saturating_add(lon_line).saturating_add(lat_line);
                
                data.extend_from_slice(&[intensity, intensity, intensity, 255]);
            }
        }
        
        data
    }
}
