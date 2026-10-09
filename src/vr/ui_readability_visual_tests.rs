use super::*;

#[test]
#[allow(deprecated)]
fn ordinary_panel_renders_with_heavier_text_and_lines() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let mut screenshots = Vec::new();
    for presenting in [false, true] {
        let ctx = egui::Context::default();
        // Match the application's Unicode fallback for arrows and symbols.
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "NotoSans".into(),
            egui::FontData::from_static(include_bytes!("../../assets/fonts/NotoSans-Regular.ttf"))
                .into(),
        );
        fonts.font_data.insert(
            "SegoeSymbol".into(),
            egui::FontData::from_static(include_bytes!("../../assets/fonts/seguisym.ttf")).into(),
        );
        fonts.font_data.insert(
            "SegoeEmoji".into(),
            egui::FontData::from_static(include_bytes!("../../assets/fonts/seguiemj.ttf")).into(),
        );
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts
                .families
                .get_mut(&family)
                .unwrap()
                .extend(["NotoSans", "SegoeSymbol", "SegoeEmoji"].map(str::to_owned));
        }
        ctx.set_fonts(fonts);
        let palette = ActivePalette::default();
        set_text_weight(&ctx, presenting);
        ctx.global_style_mut(|style| {
            style.visuals.panel_fill = palette.bg_panel;
            style.visuals.override_text_color = Some(palette.text_primary);
            style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, palette.border_normal);
            style.visuals.widgets.noninteractive.bg_stroke =
                Stroke::new(0.5, palette.border_subtle);
            for font in style.text_styles.values_mut() {
                font.size = 12.0;
            }
        });
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640.0, 360.0),
            )),
            ..Default::default()
        };
        input
            .viewports
            .get_mut(&input.viewport_id)
            .unwrap()
            .native_pixels_per_point = Some(2.0);
        ctx.begin_pass(input);
        egui::CentralPanel::default().show(&ctx, |ui| {
            ui.heading(egui::RichText::new("Bio-Spheres · Simulation settings").size(28.0));
            ui.label("Environment, display, and organism controls");
            ui.separator();
            ui.horizontal(|ui| {
                let _ = ui.button("Start simulation");
                let _ = ui.button("Load world");
                let _ = ui.button("Settings");
            });
            ui.add_space(8.0);
            let mut brightness = 1.2;
            ui.add(egui::Slider::new(&mut brightness, 0.0..=5.0).text("Sun brightness"));
            ui.checkbox(&mut true, "Show water and weather");
            ui.add_enabled(false, egui::Button::new("Unavailable action"));
            let painter = ui.painter();
            for (i, (size, text)) in [
                (10.0, "Small: 0.0125 · [World] → [Weather] · 1234567890"),
                (11.0, "Air 23.5 °C     Water 18.2 °C     Rock 25.8 °C"),
                (
                    12.0,
                    "Readouts, labels, tooltips, tabs, and editor controls",
                ),
                (14.0, "Open counters: a e s 8 B R @ % &"),
            ]
            .iter()
            .enumerate()
            {
                painter.text(
                    egui::pos2(12.0, 195.0 + i as f32 * 24.0),
                    egui::Align2::LEFT_TOP,
                    text,
                    egui::FontId::proportional(*size),
                    palette.text_primary,
                );
            }
            painter.text(
                egui::pos2(12.0, 292.0),
                egui::Align2::LEFT_TOP,
                "MONOSPACE: 0.0125   FPS 120   Cells 12,345",
                egui::FontId::monospace(11.0),
                palette.text_secondary,
            );
            for (i, width) in [0.5, 0.75, 1.0, 1.5, 2.0].into_iter().enumerate() {
                let y = 207.0 + i as f32 * 24.0;
                painter.line_segment(
                    [egui::pos2(450.0, y), egui::pos2(610.0, y)],
                    Stroke::new(width, palette.border_normal),
                );
            }
        });
        let mut output = ctx.end_pass();
        if presenting {
            strengthen_shapes(&mut output.shapes, palette);
        }
        let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        let (width, height) = (1280, 720);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal UI readability check"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let mut renderer = egui_wgpu::Renderer::new(&device, format, Default::default());
        for (id, delta) in output.textures_delta.set {
            renderer.update_texture(&device, &queue, id, &delta);
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 2.0,
        };
        let commands = renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        let stride = width * 4; // 1280 pixels is already aligned to 256 bytes.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (stride * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(commands.into_iter().chain([encoder.finish()]));
        let (send, receive) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap();
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        receive.recv().unwrap().unwrap();
        let rgba = staging.slice(..).get_mapped_range().to_vec();
        std::fs::create_dir_all("target/vr-visual-checks").unwrap();
        let name = if presenting {
            "normal-ui-vr"
        } else {
            "normal-ui-desktop"
        };
        image::save_buffer(
            format!("target/vr-visual-checks/{name}.png"),
            &rgba,
            width,
            height,
            image::ColorType::Rgba8,
        )
        .unwrap();
        screenshots.push(rgba);
    }
    // In the small-label region, more bright foreground pixels must survive.
    let ink = |rgba: &[u8]| {
        (390..620)
            .flat_map(|y| (24..850).map(move |x| (y * 1280 + x) * 4))
            .filter(|offset| rgba[*offset] > 170)
            .count()
    };
    assert!(ink(&screenshots[1]) > ink(&screenshots[0]) * 5 / 4);
}
