use super::App;

#[test]
fn initialization_waits_until_loading_dialog_has_visible_text() {
    for size in [egui::vec2(640.0, 360.0), egui::vec2(1280.0, 720.0)] {
        let ctx = egui::Context::default();
        for pass in 0..3 {
            ctx.begin_pass(egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                ..Default::default()
            });
            let ready = App::show_main_scene_loading_dialog(&ctx);
            let output = ctx.end_pass();
            let heading = output.shapes.iter().find_map(|shape| match &shape.shape {
                egui::Shape::Text(text) if text.galley.job.text == "Loading main scene…" => {
                    Some(text.visual_bounding_rect())
                }
                _ => None,
            });
            if pass == 0 {
                assert!(
                    !ready,
                    "egui's invisible sizing pass must never start initialization"
                );
            } else {
                assert!(ready, "a visible loading frame must allow initialization");
                let heading = heading.expect("ready means the loading title is actually painted");
                assert!(egui::Rect::from_min_size(egui::Pos2::ZERO, size).contains_rect(heading));
            }
        }
    }
}
