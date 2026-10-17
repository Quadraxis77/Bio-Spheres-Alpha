use super::{App, MenuAction};

#[test]
fn main_menu_report_button_opens_the_same_reporting_flow_as_f8() {
    let ctx = egui::Context::default();
    let mut settings_open = false;
    let mut fullscreen = false;
    let mut music = 0.18;
    let mut sfx = 0.45;
    let mut limit = 120;
    let mut frame = |events| {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 720.0),
            )),
            events,
            ..Default::default()
        };
        ctx.begin_pass(input);
        let result = App::render_main_menu_ui(
            &ctx,
            egui::TextureId::User(0),
            egui::TextureId::User(1),
            "Left",
            "Right",
            320.0,
            720.0,
            true,
            &mut settings_open,
            &mut fullscreen,
            &mut music,
            &mut sfx,
            &mut limit,
        );
        let _ = ctx.end_pass();
        result.action
    };
    assert_eq!(frame(vec![]), MenuAction::None);
    let pos = egui::pos2(870.0, 26.0);
    assert_eq!(
        frame(vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE
            },
        ]),
        MenuAction::None
    );
    assert_eq!(
        frame(vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE
        },]),
        MenuAction::BugReport
    );
}
