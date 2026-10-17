use super::*;

#[test]
fn settings_rows_fit_narrow_panels_and_keep_value_fields_aligned() {
    for width in [140.0, 180.0, 280.0, 420.0] {
        for font_size in [13.0, 22.0] {
            let ctx = egui::Context::default();
            ctx.global_style_mut(|style| {
                style.text_styles.insert(
                    egui::TextStyle::Monospace,
                    egui::FontId::monospace(font_size),
                );
            });
            let mut brightness = 3.12567_f32;
            let mut period = 3600.0_f32;
            let mut probability = 0.00001_f32;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                ui.set_width(width);
                let right = ui.max_rect().right();
                let a = setting(
                    ui,
                    "Brightness",
                    Slider::new(&mut brightness, 0.0..=5.0).fixed_decimals(1),
                );
                let b = setting(
                    ui,
                    "Cycle duration",
                    Slider::new(&mut period, 10.0..=3600.0)
                        .suffix(" s")
                        .fixed_decimals(0),
                );
                let c = setting(
                    ui,
                    "Mutation probability",
                    Slider::new(&mut probability, 0.0..=1.0)
                        .logarithmic(true)
                        .max_decimals(5),
                );
                for row in [&a, &b, &c] {
                    assert!(
                        row.rect.right() <= right + 0.1,
                        "row {:?} overflows {width}px panel at font size {font_size}",
                        row.rect
                    );
                    assert!(
                        (row.rect.right() - right).abs() < 0.1,
                        "number fields must align at the right edge"
                    );
                    assert!(!row.changed(), "displaying settings must not edit them");
                }
            });
            assert_eq!(brightness, 3.12567);
            assert_eq!(probability, 0.00001);
        }
    }
}

#[test]
fn long_numeric_edits_stay_inside_the_authoring_row() {
    let ctx = egui::Context::default();
    let mut value = 1.0;
    let mut field_id = egui::Id::NULL;
    for frame in 0..3 {
        if frame == 1 {
            ctx.memory_mut(|memory| memory.request_focus(field_id));
            ctx.data_mut(|data| {
                data.insert_temp(field_id, "123456789012345678901234567890".to_owned())
            });
        }
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.set_width(180.0);
            let right = ui.max_rect().right();
            let row = paired_row(ui, |ui| {
                ui.add(Slider::new(&mut value, 0.0..=1000.0).show_value(false));
                let field = ui.add(egui::DragValue::new(&mut value).range(0.0..=1000.0));
                field_id = field.id;
                if frame > 0 {
                    assert!(field.has_focus());
                }
            });
            assert!(
                row.response.rect.right() <= right + 0.1,
                "numeric editing must not expand its panel"
            );
        });
    }
}

#[test]
fn responsive_sliders_still_accept_keyboard_input() {
    let ctx = egui::Context::default();
    let mut value = 3.0;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        ui.set_width(180.0);
        setting(
            ui,
            "Brightness",
            Slider::new(&mut value, 0.0..=5.0).step_by(0.1),
        )
        .request_focus();
    });
    let _ = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowRight,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ui| {
            ui.set_width(180.0);
            assert!(setting(
                ui,
                "Brightness",
                Slider::new(&mut value, 0.0..=5.0).step_by(0.1)
            )
            .changed());
        },
    );
    assert!(value > 3.0);
}
