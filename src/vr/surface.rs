//! Reprojects the two UI images onto the same adjustable screen surface.
use super::{
    input::{position, rotation},
    panel::PanelSettings,
};
use glam::{Mat4, Vec2};
use openxr as xr;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    clip: [f32; 4],
    uv: [f32; 2],
}
pub(super) struct Surface {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}
impl Surface {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Stereo screen surface"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var image: texture_2d<f32>;
                @group(0) @binding(1) var filtering: sampler;
                struct V { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> }
                @vertex fn vs(@location(0) clip: vec4<f32>, @location(1) uv: vec2<f32>) -> V {
                    var v: V; v.clip = clip; v.uv = uv; return v;
                }
                @fragment fn fs(v: V) -> @location(0) vec4<f32> {
                    return textureSample(image, filtering, v.uv);
                }
            "#
                .into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Curved stereo menu and preview"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { pipeline, sampler }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        source: &wgpu::TextureView,
        panel: xr::Posef,
        settings: PanelSettings,
        projection: Mat4,
        clear: bool,
    ) {
        let matrix = projection * Mat4::from_rotation_translation(rotation(panel), position(panel));
        let segments = if settings.curvature < 0.001 { 1 } else { 64 };
        let mut vertices = Vec::with_capacity((segments + 1) * 2);
        let mut indices = Vec::<u32>::with_capacity(segments * 6);
        for i in 0..=segments {
            for v in [0.0, 1.0] {
                let uv = Vec2::new(i as f32 / segments as f32, v);
                vertices.push(Vertex {
                    clip: (matrix * settings.point(uv).extend(1.0)).to_array(),
                    uv: uv.to_array(),
                });
            }
            if i > 0 {
                let b = i as u32 * 2;
                indices.extend([b - 2, b - 1, b, b, b - 1, b + 1]);
            }
        }
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Screen surface vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Screen surface indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Stereo screen image"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Stereo screen projection"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: if clear {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    #[test]
    fn curved_surface_samples_each_eye_image_at_the_picked_coordinates() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let surface = Surface::new(&device, format);
        let settings = PanelSettings {
            curvature: 75.0,
            distance: 1.6,
            aspect: 2.0,
        };
        let mut anchor = super::super::panel::PanelAnchor::default();
        anchor.settings = settings;
        anchor.capture(xr::Posef::IDENTITY);
        let panel = anchor.pose().unwrap();
        let size = 256;
        for eye in 0..2 {
            let blue = if eye == 0 { 40 } else { 160 };
            let pixels: Vec<u8> = (0..128)
                .flat_map(|y| {
                    (0..256).flat_map(move |x| {
                        [
                            (20.0 + x as f32 / 255.0 * 200.0) as u8,
                            (20.0 + y as f32 / 127.0 * 200.0) as u8,
                            blue,
                            255,
                        ]
                    })
                })
                .collect();
            let source = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Independent stereo screen test image"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 128,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                source.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256 * 4),
                    rows_per_image: Some(128),
                },
                source.size(),
            );
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Curved stereo screen test target"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let origin = Vec3::new(if eye == 0 { -0.032 } else { 0.032 }, 0.0, 0.0);
            let projection = Mat4::perspective_rh(70.0f32.to_radians(), 1.0, 0.005, 250.0)
                * Mat4::from_translation(-origin);
            surface.draw(
                &device,
                &queue,
                &target.create_view(&Default::default()),
                &source.create_view(&Default::default()),
                panel,
                settings,
                projection,
                true,
            );
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (size * size * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 4),
                        rows_per_image: Some(size),
                    },
                },
                target.size(),
            );
            queue.submit([encoder.finish()]);
            let (send, receive) = std::sync::mpsc::channel();
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                send.send(r).unwrap();
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            receive.recv().unwrap().unwrap();
            let rendered = buffer.slice(..).get_mapped_range();
            for uv in [Vec2::splat(0.5), Vec2::new(0.2, 0.2), Vec2::new(0.8, 0.8)] {
                let point = position(panel) + rotation(panel) * settings.point(uv);
                let ndc = projection.project_point3(point);
                let x = ((ndc.x * 0.5 + 0.5) * size as f32) as usize;
                let y = ((0.5 - ndc.y * 0.5) * size as f32) as usize;
                let color = &rendered[(y * size as usize + x) * 4..][..4];
                assert!(color[0].abs_diff((20.0 + uv.x * 200.0) as u8) < 8);
                assert!(color[1].abs_diff((20.0 + uv.y * 200.0) as u8) < 8);
                assert!(color[2].abs_diff(blue) <= 1, "eye images must not be mixed");
                let hit = super::super::panel::ray_hit_with_settings(
                    panel,
                    settings,
                    origin,
                    (point - origin).normalize(),
                    256,
                    128,
                )
                .unwrap()
                .0;
                assert!((hit - uv * Vec2::new(256.0, 128.0)).length() < 0.1);
            }
            assert_eq!(
                &rendered[..4],
                &[0, 0, 0, 255],
                "screen corners do not fill the headset"
            );
        }
    }
}
