//! Published Quest 3 Touch Plus models in OpenXR grip space; meshes stay cached on the GPU.
use super::input::{position, rotation, VrInput};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
#[path = "controller_asset.rs"]
mod asset;
struct Hand {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}
#[cfg(test)]
#[path = "controllers_tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 4],
}

pub(super) struct Controllers {
    triangles: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    depth: [wgpu::TextureView; 2],
    hands: [Hand; 2],
}
impl Controllers {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tracked controllers"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                struct V { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }
                @vertex fn vs(@location(0) p: vec4<f32>, @location(1) c: vec4<f32>) -> V {
                    var v: V; v.position = p; v.color = c; return v;
                }
                @fragment fn fs(v: V) -> @location(0) vec4<f32> { return v.color; }
            "#
                .into(),
            ),
        });
        let pipeline = |topology, depth_write, compare| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Tracked controller geometry"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: depth_write,
                    depth_compare: compare,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let model_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Original Touch Plus texture and geometry"),
            source: wgpu::ShaderSource::Wgsl(r#"
                struct Camera { projection: mat4x4<f32>, model: mat4x4<f32> }
                @group(0) @binding(0) var<uniform> camera: Camera;
                @group(0) @binding(1) var albedo: texture_2d<f32>;
                @group(0) @binding(2) var filtering: sampler;
                struct V { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) uv: vec2<f32> }
                @vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) uv: vec2<f32>) -> V {
                    var v: V; v.position = camera.projection * camera.model * vec4<f32>(p, 1.0);
                    v.normal = normalize((camera.model * vec4<f32>(n, 0.0)).xyz); v.uv = uv; return v;
                }
                @fragment fn fs(v: V) -> @location(0) vec4<f32> {
                    let shade = 0.45 + 0.55 * max(dot(normalize(v.normal), normalize(vec3<f32>(0.3, 0.8, 0.6))), 0.0);
                    return vec4<f32>(textureSample(albedo, filtering, v.uv).rgb * shade, 1.0);
                }
            "#.into()),
        });
        let triangles = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Published Touch Plus meshes"), layout: None,
            vertex: wgpu::VertexState { module: &model_shader, entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout { array_stride: 32, step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2] }], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &model_shader, entry_point: Some("fs"), targets: &[Some(format.into())], compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview: None, cache: None,
        });
        let hands = [asset::LEFT, asset::RIGHT].map(|bytes| {
            let model = asset::load(bytes);
            let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cached Touch Plus vertices"),
                contents: bytemuck::cast_slice(&model.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cached Touch Plus indices"),
                contents: bytemuck::cast_slice(&model.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Touch Plus pose"),
                size: 128,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let image = image::load_from_memory(&model.png)
                .expect("Embedded Touch Plus PNG")
                .to_rgba8();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Original Touch Plus labels and materials"),
                size: wgpu::Extent3d {
                    width: image.width(),
                    height: image.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &image,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(image.width() * 4),
                    rows_per_image: Some(image.height()),
                },
                texture.size(),
            );
            let view = texture.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Touch Plus material"),
                layout: &triangles.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            Hand {
                vertices,
                indices,
                count: model.indices.len() as u32,
                uniform,
                bind_group,
            }
        });
        Self {
            triangles,
            hands,
            lines: pipeline(
                wgpu::PrimitiveTopology::LineList,
                false,
                wgpu::CompareFunction::Always,
            ),
            depth: std::array::from_fn(|_| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("Controller overlay depth"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            }),
        }
    }
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        eye: usize,
        projection: Mat4,
        input: &VrInput,
        ray: Option<(Vec3, Vec3)>,
    ) {
        for (index, hand) in self.hands.iter().enumerate() {
            // The asset coordinate frame is grip space, never the aim ray's frame.
            if let Some(pose) = input.grips[index] {
                let model = Mat4::from_rotation_translation(rotation(pose), position(pose));
                queue.write_buffer(
                    &hand.uniform,
                    0,
                    bytemuck::cast_slice(&[projection.to_cols_array(), model.to_cols_array()]),
                );
            }
        }
        let mut beam = Vec::new();
        if let Some((origin, end)) = ray {
            let color = Vec3::new(0.0, 0.9, 0.75);
            beam.extend([
                vertex(projection, origin, color),
                vertex(projection, end, color),
            ]);
            let point = projection * end.extend(1.0);
            if point.w > 0.0 {
                for axis in [
                    glam::Vec4::new(point.w * 0.003, 0.0, 0.0, 0.0),
                    glam::Vec4::new(0.0, point.w * 0.003, 0.0, 0.0),
                ] {
                    beam.extend([
                        Vertex {
                            position: (point - axis).to_array(),
                            color: color.extend(1.0).to_array(),
                        },
                        Vertex {
                            position: (point + axis).to_array(),
                            color: color.extend(1.0).to_array(),
                        },
                    ]);
                }
            }
        }
        let beam_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Controller beam vertices"),
            contents: bytemuck::cast_slice(&beam),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tracked controllers and pointer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth[eye],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.triangles);
            for (index, hand) in self.hands.iter().enumerate() {
                if input.grips[index].is_none() {
                    continue;
                }
                pass.set_bind_group(0, &hand.bind_group, &[]);
                pass.set_vertex_buffer(0, hand.vertices.slice(..));
                pass.set_index_buffer(hand.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..hand.count, 0, 0..1);
            }
            if !beam.is_empty() {
                pass.set_pipeline(&self.lines);
                pass.set_vertex_buffer(0, beam_buffer.slice(..));
                pass.draw(0..beam.len() as u32, 0..1);
            }
        }
        queue.submit([encoder.finish()]);
    }
}
fn vertex(transform: Mat4, point: Vec3, color: Vec3) -> Vertex {
    Vertex {
        position: (transform * point.extend(1.0)).to_array(),
        color: color.extend(1.0).to_array(),
    }
}
