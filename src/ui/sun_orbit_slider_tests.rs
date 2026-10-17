use super::sun_orbit_position_slider;

#[test]
fn orbit_position_display_preserves_animation_with_and_without_focus() {
    for focused in [false, true] {
        let ctx = egui::Context::default();
        for frame in 0..120 {
            let expected = 159.23485 + frame as f32 / 60.0;
            let mut angle = expected;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                let response = sun_orbit_position_slider(ui, &mut angle);
                if focused {
                    response.request_focus();
                }
                assert!(
                    !response.changed(),
                    "displaying the orbit angle must not count as an edit (focus={focused})"
                );
            });
            assert_eq!(angle, expected, "display must preserve the animated angle");
        }
    }
}

#[test]
fn orbit_position_still_accepts_keyboard_edits() {
    let ctx = egui::Context::default();
    let mut angle = 159.23485;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        sun_orbit_position_slider(ui, &mut angle).request_focus();
    });
    let before = angle;
    let input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = ctx.run_ui(input, |ui| {
        assert!(sun_orbit_position_slider(ui, &mut angle).changed());
    });
    assert!(
        angle > before,
        "keyboard edits must still advance the slider"
    );
}
