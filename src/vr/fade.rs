//! Fade only the world projection; tracked hands and wrist UI stay visible.
use wgpu::util::DeviceExt;
pub(super) struct Fade { pipeline: wgpu::RenderPipeline, uniform: wgpu::Buffer, bind_group: wgpu::BindGroup }
impl Fade {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VR turn fade"), source: wgpu::ShaderSource::Wgsl(r#"
                @group(0) @binding(0) var<uniform> opacity: vec4<f32>;
                @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
                    let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
                    return vec4<f32>(p[index], 0.0, 1.0);
                }
                @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, opacity.x); }
            "#.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VR black fade overlay"), layout: None,
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[Some(wgpu::ColorTargetState {
                format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL,
            })], compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("VR fade opacity"), contents: bytemuck::cast_slice(&[0.0_f32; 4]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("VR fade"), layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }] });
        Self { pipeline, uniform, bind_group }
    }
    pub fn draw(&self, device: &wgpu::Device, queue: &wgpu::Queue, targets: &[wgpu::TextureView], opacity: f32) {
        if opacity <= 0.0 { return; }
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[opacity.clamp(0.0, 1.0), 0.0_f32, 0.0, 0.0]));
        let mut encoder = device.create_command_encoder(&Default::default());
        for target in targets {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fade headset world"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })], ..Default::default()
            });
            pass.set_pipeline(&self.pipeline); pass.set_bind_group(0, &self.bind_group, &[]); pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fade_covers_both_eye_targets_and_reaches_completely_black() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let fade = Fade::new(&device, format);
        let textures: Vec<_> = (0..2).map(|_| device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fade stereo regression"), size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        })).collect();
        let views: Vec<_> = textures.iter().map(|t| t.create_view(&Default::default())).collect();
        for (opacity, expected) in [(0.0, 255_i16), (0.5, 128), (1.0, 0)] {
            let mut encoder = device.create_command_encoder(&Default::default());
            for view in &views {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    })], ..Default::default()
                });
            }
            queue.submit([encoder.finish()]);
            fade.draw(&device, &queue, &views, opacity);
            for texture in &textures {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_texture_to_buffer(texture.as_image_copy(), wgpu::TexelCopyBufferInfo {
                    buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) },
                }, texture.size());
                queue.submit([encoder.finish()]);
                let (send, receive) = std::sync::mpsc::channel();
                buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| { send.send(r).unwrap(); });
                device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(10)) }).unwrap();
                receive.recv().unwrap().unwrap();
                let pixels = buffer.slice(..).get_mapped_range();
                for channel in &pixels[..3] { assert!((i16::from(*channel) - expected).abs() <= 1); }
                assert_eq!(pixels[3], 255, "The world stays opaque while fading");
                drop(pixels); buffer.unmap();
            }
        }
    }
}
