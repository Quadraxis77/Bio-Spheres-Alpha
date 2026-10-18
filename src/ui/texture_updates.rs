//! Keep UI textures current even when no panel is being drawn.

pub(super) fn upload(
    renderer: &mut egui_wgpu::Renderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output: &mut egui::FullOutput,
) {
    // Consume updates before cloning the output for multiple views. Each atlas
    // growth must reach the GPU even during hidden immersive UI frames.
    for (id, delta) in std::mem::take(&mut output.textures_delta.set) {
        renderer.update_texture(device, queue, id, &delta);
    }
    // Free textures only after every view has rendered this output.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_atlas_growth_is_uploaded_before_the_panel_returns_with_a_partial_update() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let mut renderer =
            egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, Default::default());
        let id = egui::TextureId::Managed(0);
        let options = egui::TextureOptions::LINEAR;
        let mut output = egui::FullOutput::default();
        output.textures_delta.set.push((
            id,
            egui::epaint::ImageDelta::full(
                egui::ColorImage::filled([128, 128], egui::Color32::WHITE),
                options,
            ),
        ));
        upload(&mut renderer, &device, &queue, &mut output);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        // The immersive panel is hidden. Egui grows its font atlas, but no UI
        // draw occurs; dropping this full update caused the reported crash.
        output.textures_delta.set.push((
            id,
            egui::epaint::ImageDelta::full(
                egui::ColorImage::filled([128, 256], egui::Color32::WHITE),
                options,
            ),
        ));
        upload(&mut renderer, &device, &queue, &mut output);
        assert_eq!(
            renderer
                .texture(&id)
                .unwrap()
                .texture
                .as_ref()
                .unwrap()
                .height(),
            256
        );

        // Exact failing Y range from the game log: 84..187, formerly uploaded
        // into the stale 128-row texture when the main panel became visible.
        output.textures_delta.set.push((
            id,
            egui::epaint::ImageDelta::partial(
                [0, 84],
                egui::ColorImage::filled([64, 103], egui::Color32::GREEN),
                options,
            ),
        ));
        output.textures_delta.free.push(id);
        upload(&mut renderer, &device, &queue, &mut output);
        assert!(output.textures_delta.set.is_empty());
        assert_eq!(output.textures_delta.free, [id]);
        // Reusing the prepared output for desktop and both eyes cannot replay
        // an old full atlas or prematurely delete a texture.
        for mut view_output in [output.clone(), output.clone(), output] {
            upload(&mut renderer, &device, &queue, &mut view_output);
        }
        assert!(renderer.texture(&id).is_some());
        queue.submit([]);
        assert!(pollster::block_on(device.pop_error_scope()).is_none());
    }
}
