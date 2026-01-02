//! Hi-Z (Hierarchical Depth Buffer) generator for occlusion culling.
//!
//! Generates a mipmap pyramid from the depth buffer where each level
//! contains the maximum depth of the corresponding 2x2 block from the
//! previous level. This allows efficient occlusion testing in the
//! instance builder compute shader.

use bytemuck::{Pod, Zeroable};

/// Hi-Z texture generator using compute shaders.
pub struct HizGenerator {
    // Pipeline for copying depth to Hi-Z mip 0
    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    
    // Pipeline for downsampling Hi-Z mips
    downsample_pipeline: wgpu::ComputePipeline,
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    
    params_buffer: wgpu::Buffer,
    
    // Hi-Z texture with mip chain
    hiz_texture: Option<wgpu::Texture>,
    hiz_view: Option<wgpu::TextureView>,
    mip_views: Vec<wgpu::TextureView>,
    
    // Cached bind groups (recreated on resize)
    copy_bind_group: Option<wgpu::BindGroup>,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    
    width: u32,
    height: u32,
    mip_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HizParams {
    src_width: u32,
    src_height: u32,
    _pad: [u32; 2],
}

impl HizGenerator {
    /// Create a new Hi-Z generator.
    pub fn new(device: &wgpu::Device) -> Self {
        // Load separate shaders for copy and downsample
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hi-Z Copy Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/hiz_copy_depth.wgsl").into(),
            ),
        });

        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hi-Z Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/hiz_downsample.wgsl").into(),
            ),
        });

        // Bind group layout for copying depth texture to Hi-Z mip 0
        let copy_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hi-Z Copy Bind Group Layout"),
            entries: &[
                // Params uniform
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
                // Source depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Destination storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Bind group layout for downsampling Hi-Z mips
        let downsample_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hi-Z Downsample Bind Group Layout"),
            entries: &[
                // Params uniform
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
                // Source float texture (Hi-Z mip)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Destination storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hi-Z Copy Pipeline Layout"),
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });

        let downsample_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hi-Z Downsample Pipeline Layout"),
            bind_group_layouts: &[&downsample_bind_group_layout],
            push_constant_ranges: &[],
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hi-Z Copy Pipeline"),
            layout: Some(&copy_pipeline_layout),
            module: &copy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hi-Z Downsample Pipeline"),
            layout: Some(&downsample_pipeline_layout),
            module: &downsample_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hi-Z Params Buffer"),
            size: std::mem::size_of::<HizParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            copy_pipeline,
            copy_bind_group_layout,
            downsample_pipeline,
            downsample_bind_group_layout,
            params_buffer,
            hiz_texture: None,
            hiz_view: None,
            mip_views: Vec::new(),
            copy_bind_group: None,
            downsample_bind_groups: Vec::new(),
            width: 0,
            height: 0,
            mip_count: 0,
        }
    }

    /// Resize the Hi-Z texture to match the depth buffer dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;
        self.mip_count = ((width.max(height) as f32).log2().floor() as u32 + 1).max(1);

        // Clear cached bind groups (will be recreated on first generate)
        self.copy_bind_group = None;
        self.downsample_bind_groups.clear();

        // Create Hi-Z texture with full mip chain
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hi-Z Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: self.mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create view for the full texture (for sampling in culling shader)
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Hi-Z Full View"),
            ..Default::default()
        });

        // Create views for each mip level (for writing during generation)
        let mut mip_views = Vec::with_capacity(self.mip_count as usize);
        for mip in 0..self.mip_count {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Hi-Z Mip {} View", mip)),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            mip_views.push(view);
        }

        self.hiz_texture = Some(texture);
        self.hiz_view = Some(full_view);
        self.mip_views = mip_views;
    }

    /// Generate the Hi-Z mip chain from a depth texture.
    /// The depth texture should be Depth32Float format.
    pub fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
    ) {
        if self.hiz_texture.is_none() || self.mip_count < 1 {
            return;
        }

        // Create bind groups on first use (or after resize)
        if self.copy_bind_group.is_none() {
            self.create_bind_groups(device, depth_view);
        }

        let copy_bind_group = self.copy_bind_group.as_ref().unwrap();

        // Pass 0: Copy depth buffer to Hi-Z mip 0
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hi-Z Copy Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.copy_pipeline);
            compute_pass.set_bind_group(0, copy_bind_group, &[]);

            let workgroups_x = (self.width + 7) / 8;
            let workgroups_y = (self.height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Generate remaining mip levels by downsampling
        let mut src_width = self.width;
        let mut src_height = self.height;

        for mip in 0..(self.mip_count - 1) {
            let dst_width = (src_width / 2).max(1);
            let dst_height = (src_height / 2).max(1);

            // Update params for this mip level
            let params = HizParams {
                src_width,
                src_height,
                _pad: [0; 2],
            };
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

            let bind_group = &self.downsample_bind_groups[mip as usize];

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Hi-Z Downsample {} Pass", mip + 1)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.downsample_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            let workgroups_x = (dst_width + 7) / 8;
            let workgroups_y = (dst_height + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

            src_width = dst_width;
            src_height = dst_height;
        }
    }

    /// Create and cache bind groups for Hi-Z generation.
    fn create_bind_groups(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        // Create copy bind group
        self.copy_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hi-Z Copy Bind Group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.mip_views[0]),
                },
            ],
        }));

        // Create downsample bind groups for each mip transition
        self.downsample_bind_groups.clear();
        for mip in 0..(self.mip_count - 1) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Hi-Z Downsample {} Bind Group", mip + 1)),
                layout: &self.downsample_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.mip_views[mip as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.mip_views[(mip + 1) as usize]),
                    },
                ],
            });
            self.downsample_bind_groups.push(bind_group);
        }
    }

    /// Get the Hi-Z texture view for use in culling.
    pub fn hiz_view(&self) -> Option<&wgpu::TextureView> {
        self.hiz_view.as_ref()
    }

    /// Get the Hi-Z texture for binding.
    pub fn hiz_texture(&self) -> Option<&wgpu::Texture> {
        self.hiz_texture.as_ref()
    }

    /// Get the number of mip levels.
    pub fn mip_count(&self) -> u32 {
        self.mip_count
    }
}
