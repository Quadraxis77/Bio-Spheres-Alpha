//! Nutrient particle system renderer (GPU-based)
//!
//! Renders nutrients (fluid type 5) as small camera-facing triangles that spawn
//! in water voxels with a probability based on non-isolation detection.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Nutrient particle instance data (must match shader struct)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NutrientParticle {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub animation: [f32; 4],  // Time offset, rotation speed, etc.
}

/// Parameters for nutrient extraction compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NutrientExtractParams {
    pub grid_resolution: u32,
    pub cell_size: f32,
    pub max_particles: u32,
    pub time: f32,
    pub grid_origin: [f32; 3],
    pub world_radius: f32,
    pub spawn_probability: f32,  // Legacy parameter (kept for compatibility)
    pub nutrient_density: f32,   // Density parameter (0.0 = sparse, 1.0 = dense)
    pub _padding: [f32; 2],      // Pad to 48 bytes for WGSL struct alignment
}

/// Particle counter (for atomic counting in compute shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct NutrientParticleCounter {
    pub count: u32,
}

/// GPU-based nutrient particle system renderer
pub struct NutrientParticleRenderer {
    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,

    // Compute pipeline for extracting nutrient positions
    extract_pipeline: wgpu::ComputePipeline,
    extract_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    particle_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    counter_staging_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

    // Bind group layouts
    camera_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,

    // State
    max_particles: u32,
    time: f32,
    particle_count: u32,
    spawn_probability: f32,
    width: u32,
    height: u32,
}

impl NutrientParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Self {
        let max_particles = 800_000u32;  // Support up to 800k nutrient particles

        // Create render shader
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nutrient Particle Render Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/nutrient_particles.wgsl").into(),
            ),
        });

        // Create compute shader for extraction
        let extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nutrient Extract Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/nutrient_extract.wgsl").into(),
            ),
        });

        // Create particle buffer
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nutrient Particle Buffer"),
            size: (std::mem::size_of::<NutrientParticle>() * max_particles as usize) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create counter buffer (for atomic counting)
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nutrient Particle Counter"),
            size: std::mem::size_of::<NutrientParticleCounter>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffer to read back particle count
        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nutrient Counter Staging"),
            size: std::mem::size_of::<NutrientParticleCounter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nutrient Extract Params"),
            size: std::mem::size_of::<NutrientExtractParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create extract compute bind group layout
        let extract_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Extract Bind Group Layout"),
            entries: &[
                // Fluid state buffer (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle buffer (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counter buffer (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Solid mask buffer (read)
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
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Nutrient voxels buffer (read) - stores which voxels have nutrients
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
            ],
        });

        // Create compute pipeline
        let extract_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nutrient Extract Pipeline Layout"),
            bind_group_layouts: &[&extract_bind_group_layout],
            push_constant_ranges: &[],
        });

        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nutrient Extract Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &extract_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create render bind group layout
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Render Bind Group Layout"),
            entries: &[],
        });

        // Create render pipeline layout
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nutrient Particle Pipeline Layout"),
            bind_group_layouts: &[camera_layout, &render_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout for instanced particles
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<NutrientParticle>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as u64,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 8]>() as u64,
                    shader_location: 3,
                },
            ],
        };

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Nutrient Particle Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
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
                cull_mode: None,  // Camera-facing triangles need no culling
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
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

        // Clone the camera layout for later use
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Camera Bind Group Layout (stored)"),
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
            ],
        });

        Self {
            render_pipeline,
            extract_pipeline,
            extract_bind_group_layout,
            particle_buffer,
            counter_buffer,
            counter_staging_buffer,
            params_buffer,
            camera_bind_group_layout,
            render_bind_group_layout,
            max_particles,
            time: 0.0,
            particle_count: 0,
            spawn_probability: 0.1,  // 10% chance to spawn in eligible water voxels
            width,
            height,
        }
    }

    /// Create bind group for compute extraction
    pub fn create_extract_bind_group(
        &self,
        device: &wgpu::Device,
        fluid_state_buffer: &wgpu::Buffer,
        solid_mask_buffer: &wgpu::Buffer,
        nutrient_voxels_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Extract Bind Group"),
            layout: &self.extract_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fluid_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: solid_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: nutrient_voxels_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create bind group for rendering
    pub fn create_render_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Render Bind Group"),
            layout: &self.render_bind_group_layout,
            entries: &[],
        })
    }

    /// Set spawn probability for nutrient particles (0.0 to 1.0)
    pub fn set_spawn_probability(&mut self, probability: f32) {
        self.spawn_probability = probability.clamp(0.0, 1.0);
    }

    /// Get current spawn probability
    pub fn spawn_probability(&self) -> f32 {
        self.spawn_probability
    }

    /// Run compute shader to extract nutrient particles from water voxels
    pub fn extract_nutrient_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        extract_bind_group: &wgpu::BindGroup,
        grid_resolution: u32,
        grid_origin: Vec3,
        cell_size: f32,
        world_radius: f32,
        dt: f32,
        nutrient_density: f32,
    ) {
        self.time += dt;

        // Reset counter to 0
        queue.write_buffer(&self.counter_buffer, 0, bytemuck::cast_slice(&[0u32]));

        // Update params
        let params = NutrientExtractParams {
            grid_resolution,
            cell_size,
            max_particles: self.max_particles,
            time: self.time,
            grid_origin: grid_origin.to_array(),
            world_radius,
            spawn_probability: self.spawn_probability,
            nutrient_density,
            _padding: [0.0; 2],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Run compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nutrient Extract Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.extract_pipeline);
            compute_pass.set_bind_group(0, extract_bind_group, &[]);

            // Dispatch workgroups to cover entire grid (4x4x4 = 64 threads per workgroup)
            let workgroup_size = 4u32;
            let workgroups = (grid_resolution + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        // Copy counter to staging buffer for readback
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.counter_staging_buffer,
            0,
            std::mem::size_of::<NutrientParticleCounter>() as u64,
        );
    }

    /// Poll for particle count (call after command buffer submission)
    pub fn poll_particle_count(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let buffer_slice = self.counter_staging_buffer.slice(..);
        let mapped = Arc::new(AtomicBool::new(false));
        let mapped_clone = mapped.clone();

        // Map the buffer with callback
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                mapped_clone.store(true, Ordering::SeqCst);
            }
        });

        // Poll until mapped
        while !mapped.load(Ordering::SeqCst) {
            let _ = device.poll(wgpu::PollType::Poll);
        }

        // Read the count
        {
            let data = buffer_slice.get_mapped_range();
            let count: &[u32] = bytemuck::cast_slice(&data);
            self.particle_count = count[0].min(self.max_particles);
        }

        // Unmap
        self.counter_staging_buffer.unmap();
    }

    /// Set particle count directly (for async readback)
    pub fn set_particle_count(&mut self, count: u32) {
        self.particle_count = count.min(self.max_particles);
    }

    /// Resize for new screen dimensions
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Render nutrient particles
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_bind_group: &wgpu::BindGroup,
        render_bind_group: &wgpu::BindGroup,
    ) {
        if self.particle_count == 0 {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Nutrient Particle Pass"),
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
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
        render_pass.draw(0..3, 0..self.particle_count);  // 3 vertices for triangle
    }

    /// Get particle count
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Get max particles
    pub fn max_particles(&self) -> u32 {
        self.max_particles
    }

    /// Get the particle buffer
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particle_buffer
    }

    /// Get counter buffer
    pub fn counter_buffer(&self) -> &wgpu::Buffer {
        &self.counter_buffer
    }

    /// Get camera bind group layout
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bind_group_layout
    }
}
