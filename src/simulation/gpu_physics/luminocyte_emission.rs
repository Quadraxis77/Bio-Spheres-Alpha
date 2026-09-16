//! GPU-only colored emission and heat source shared by rendering and climate.
use super::luminocyte_ray_tracing::CaveRayTracing;
use std::cell::{Cell, RefCell};
use wgpu::util::DeviceExt;

/// Shared emission kernel with either hardware or portable occlusion.
pub fn emission_shader_source(hardware: bool) -> String {
    include_str!("../../../shaders/luminocyte_emission.wgsl").replace(
        "// OCCLUSION_IMPLEMENTATION",
        if hardware {
            include_str!("../../../shaders/luminocyte_ray_query.wgsl")
        } else {
            include_str!("../../../shaders/luminocyte_voxel_occlusion.wgsl")
        },
    )
}

pub struct LuminocyteEmission {
    pub buffer: wgpu::Buffer,
    params: wgpu::Buffer,
    scatter: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    ray_tracing_enabled: Cell<bool>,
    ray_tracing: Option<(wgpu::ComputePipeline, RefCell<CaveRayTracing>)>,
}
impl LuminocyteEmission {
    pub fn new(device: &wgpu::Device, origin: [f32; 3], cell_size: f32) -> Self {
        let pipeline = |source: &str, entry: &'static str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(source.to_owned().into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let ray_tracing = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .then(|| {
                (
                    pipeline(&emission_shader_source(true), "scatter"),
                    RefCell::new(CaveRayTracing::new()),
                )
            });
        Self {
            ray_tracing_enabled: Cell::new(true),
            ray_tracing,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Luminocyte radiance and heat"),
                size: 128 * 128 * 128 * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Luminocyte grid"),
                contents: bytemuck::cast_slice(&[origin[0], origin[1], origin[2], cell_size]),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            scatter: pipeline(&emission_shader_source(false), "scatter"),
            resolve: pipeline(
                include_str!("../../../shaders/luminocyte_resolve.wgsl"),
                "resolve",
            ),
        }
    }
    pub fn set_solid_mask(&self, mask: &[u32]) {
        if let Some((_, rt)) = &self.ray_tracing {
            rt.borrow_mut().set_solid_mask(mask);
        }
    }
    pub fn hardware_ray_tracing_supported(&self) -> bool {
        self.ray_tracing.is_some()
    }
    pub fn set_hardware_ray_tracing_enabled(&self, enabled: bool) {
        self.ray_tracing_enabled.set(enabled);
    }
    pub fn hardware_ray_tracing_active(&self) -> bool {
        self.ray_tracing_enabled.get()
            && self
                .ray_tracing
                .as_ref()
                .is_some_and(|(_, rt)| rt.borrow().tlas.is_some())
    }
    pub fn scatter(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        positions: &wgpu::Buffer,
        glow: &wgpu::Buffer,
        count: &wgpu::Buffer,
        solid: &wgpu::Buffer,
        slots: u32,
    ) {
        encoder.clear_buffer(&self.buffer, 0, None);
        if let Some((pipeline, rt)) = self
            .ray_tracing
            .as_ref()
            .filter(|_| self.ray_tracing_enabled.get())
        {
            let mut rt = rt.borrow_mut();
            rt.prepare(device, encoder);
            if let Some(tlas) = &rt.tlas {
                let buffers = [&self.params, positions, glow, count, solid, &self.buffer];
                let mut entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect();
                entries.push(wgpu::BindGroupEntry {
                    binding: 6,
                    resource: tlas.as_binding(),
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Luminocyte hardware ray queries"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Luminocyte hardware ray queries"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(slots.div_ceil(64), 1, 1);
                return;
            }
        }
        self.dispatch(
            device,
            encoder,
            &self.scatter,
            &[&self.params, positions, glow, count, solid, &self.buffer],
            slots.div_ceil(64),
        );
    }
    pub fn resolve(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        colors: &wgpu::Buffer,
    ) {
        self.dispatch(
            device,
            encoder,
            &self.resolve,
            &[&self.buffer, colors],
            (128u32 * 128 * 128).div_ceil(64),
        );
    }
    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        groups: u32,
    ) {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Luminocyte emission"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Luminocyte emission"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}
