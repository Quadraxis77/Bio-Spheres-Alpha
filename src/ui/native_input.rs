//! Merge headset pointing with real desktop input without inventing window events.
#[derive(Default)]
pub(super) struct NativeInput {
    pub active: bool,
    pub pointer: Option<egui::Pos2>,
    pub pressed: bool,
    pub scroll_y: f32,
    controller_owns_pointer: bool,
    controller_down: bool,
    controller_cancelled: bool,
    previous_pressed: bool,
    last_controller_position: Option<egui::Pos2>,
    press_position: Option<egui::Pos2>,
    dragging: bool,
    desktop_position: Option<egui::Pos2>,
    desktop_buttons: [bool; 5],
    desktop_until: f64,
}

impl NativeInput {
    pub fn controller_cancelled(&self) -> bool {
        self.controller_cancelled
    }
    pub fn controller_pointing(&self) -> bool {
        (self.active && self.controller_owns_pointer) || self.controller_cancelled
    }
    pub fn merge(&mut self, raw: &mut egui::RawInput) -> Option<egui::Pos2> {
        self.controller_cancelled = false;
        let time = raw.time.unwrap_or(0.0);
        let mut events = Vec::with_capacity(raw.events.len() + 4);
        for event in std::mem::take(&mut raw.events) {
            // Keep the focus-loss event so egui releases held keyboard keys,
            // while the effective viewport focus below comes from OpenXR.
            if self.active && matches!(event, egui::Event::WindowFocused(_)) {
                if matches!(event, egui::Event::WindowFocused(false)) {
                    self.desktop_buttons.fill(false);
                }
            }
            let desktop_activity = matches!(
                event,
                egui::Event::PointerMoved(_)
                    | egui::Event::PointerButton { .. }
                    | egui::Event::MouseWheel { .. }
            );
            if desktop_activity {
                // Finish any controller drag before handing ownership to a physical mouse.
                if self.controller_down {
                    self.cancel_press(&mut events);
                }
                self.controller_owns_pointer = false;
                self.desktop_until = time + 1.0;
            }
            match &event {
                egui::Event::PointerMoved(pos) => self.desktop_position = Some(*pos),
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    ..
                } => {
                    self.desktop_position = Some(*pos);
                    self.desktop_buttons[*button as usize] = *pressed;
                }
                egui::Event::PointerGone => {
                    self.desktop_position = None;
                    if self.controller_owns_pointer {
                        continue;
                    }
                }
                egui::Event::WindowFocused(false) => self.desktop_buttons.fill(false),
                _ => {}
            }
            events.push(event);
        }
        let mouse_down = self.desktop_buttons.iter().any(|down| *down);
        let trigger_edge = self.pressed && !self.previous_pressed;
        if self.active && self.pointer.is_some() && !mouse_down {
            if trigger_edge || time >= self.desktop_until {
                self.controller_owns_pointer = true;
            }
        }
        if self.active && self.controller_owns_pointer && self.pointer.is_some() {
            let mut pos = self.pointer.unwrap();
            if trigger_edge {
                self.press_position = Some(pos);
                self.dragging = false;
            }
            // Pulling a trigger produces small aim motion. Keep clicks stable;
            // deliberate movement still turns a held trigger into a slider drag.
            if let Some(anchor) = self.press_position {
                let slop = raw
                    .screen_rect
                    .map_or(8.0, |rect| (rect.width() * 0.015 / 1.8).max(8.0));
                if !self.dragging && pos.distance(anchor) <= slop {
                    pos = anchor;
                } else {
                    self.dragging = true;
                }
            }
            events.push(egui::Event::PointerMoved(pos));
            self.last_controller_position = Some(pos);
            if self.pressed != self.controller_down {
                // A held trigger cannot start a second drag after mouse takeover.
                if !self.pressed || trigger_edge {
                    events.push(button(pos, self.pressed));
                    self.controller_down = self.pressed;
                    if !self.pressed {
                        self.press_position = None;
                    }
                }
            }
            if self.scroll_y.abs() > f32::EPSILON {
                events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, self.scroll_y),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::NONE,
                });
            }
        } else if self.controller_owns_pointer || self.controller_down {
            self.cancel_press(&mut events);
            self.controller_owns_pointer = false;
            if let Some(pos) = self.desktop_position {
                events.push(egui::Event::PointerMoved(pos));
            } else {
                events.push(egui::Event::PointerGone);
            }
        }
        self.previous_pressed = self.pressed;
        if self.active {
            raw.focused = true;
            raw.viewports.entry(raw.viewport_id).or_default().focused = Some(true);
        }
        raw.events = events;
        if self.controller_owns_pointer {
            self.last_controller_position
        } else {
            self.desktop_position
        }
    }

    fn cancel_press(&mut self, events: &mut Vec<egui::Event>) {
        if self.controller_down {
            self.controller_cancelled = true;
            // Release outside every widget: lost tracking must cancel, not click.
            let outside = egui::pos2(-1_000_000.0, -1_000_000.0);
            events.push(egui::Event::PointerMoved(outside));
            events.push(button(outside, false));
            self.controller_down = false;
            self.press_position = None;
        }
    }
}

fn button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        ctx: &egui::Context,
        input: &mut NativeInput,
        time: f64,
        events: Vec<egui::Event>,
    ) -> (bool, egui::Response) {
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 300.0),
            )),
            time: Some(time),
            focused: false,
            events,
            ..Default::default()
        };
        input.merge(&mut raw);
        ctx.begin_pass(raw);
        #[allow(deprecated)]
        let response = egui::CentralPanel::default()
            .show(ctx, |ui| {
                ui.put(
                    egui::Rect::from_min_size(egui::pos2(40.0, 40.0), egui::vec2(120.0, 40.0)),
                    egui::Button::new("Select"),
                )
            })
            .inner;
        let clicked = response.clicked();
        let _ = ctx.end_pass();
        (clicked, response)
    }

    #[test]
    fn controller_click_works_with_unfocused_desktop() {
        let ctx = egui::Context::default();
        let mut input = NativeInput {
            active: true,
            pointer: Some(egui::pos2(80.0, 60.0)),
            ..Default::default()
        };
        frame(&ctx, &mut input, 0.0, vec![]);
        input.pressed = true;
        assert!(
            !frame(
                &ctx,
                &mut input,
                0.1,
                vec![egui::Event::WindowFocused(false)]
            )
            .0
        );
        input.pressed = false;
        assert!(frame(&ctx, &mut input, 0.2, vec![]).0);
    }

    #[test]
    fn controller_scroll_is_sent_to_egui_while_it_owns_the_pointer() {
        let mut input = NativeInput {
            active: true,
            pointer: Some(egui::pos2(80.0, 60.0)),
            scroll_y: -24.0,
            ..Default::default()
        };
        let mut raw = egui::RawInput {
            time: Some(1.0),
            ..Default::default()
        };
        input.merge(&mut raw);
        assert!(raw.events.iter().any(|event| matches!(
            event,
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta,
                ..
            } if *delta == egui::vec2(0.0, -24.0)
        )));
    }

    #[test]
    fn mouse_click_survives_missing_or_competing_controller_pointer() {
        for pointer in [None, Some(egui::pos2(300.0, 250.0))] {
            let ctx = egui::Context::default();
            let mut input = NativeInput {
                active: true,
                pointer,
                ..Default::default()
            };
            frame(&ctx, &mut input, 0.0, vec![]);
            let pos = egui::pos2(80.0, 60.0);
            frame(
                &ctx,
                &mut input,
                0.1,
                vec![egui::Event::PointerMoved(pos), button(pos, true)],
            );
            assert!(frame(&ctx, &mut input, 0.2, vec![button(pos, false)]).0);
        }
    }

    #[test]
    fn tracking_loss_cancels_drag_and_restores_desktop_pointer() {
        let ctx = egui::Context::default();
        let pos = egui::pos2(80.0, 60.0);
        let mut input = NativeInput {
            active: true,
            pointer: Some(pos),
            ..Default::default()
        };
        frame(
            &ctx,
            &mut input,
            0.0,
            vec![egui::Event::PointerMoved(egui::pos2(300.0, 250.0))],
        );
        input.pressed = true;
        frame(&ctx, &mut input, 0.1, vec![]);
        input.pointer = None;
        input.pressed = false;
        assert!(!frame(&ctx, &mut input, 0.2, vec![]).0);
        assert!(!ctx.input(|i| i.pointer.primary_down()));
        assert_eq!(ctx.pointer_hover_pos(), Some(egui::pos2(300.0, 250.0)));
    }

    #[test]
    fn keyboard_activation_survives_xr_focus() {
        let ctx = egui::Context::default();
        let mut input = NativeInput {
            active: true,
            ..Default::default()
        };
        let (_, response) = frame(&ctx, &mut input, 0.0, vec![]);
        response.request_focus();
        let key = |pressed| egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let down = frame(
            &ctx,
            &mut input,
            0.1,
            vec![egui::Event::WindowFocused(false), key(true)],
        )
        .0;
        let up = frame(&ctx, &mut input, 0.2, vec![key(false)]).0;
        assert!(down || up);
    }

    #[test]
    fn desktop_focus_loss_releases_keys_without_disabling_xr_ui() {
        let ctx = egui::Context::default();
        let mut input = NativeInput {
            active: true,
            ..Default::default()
        };
        frame(
            &ctx,
            &mut input,
            0.0,
            vec![egui::Event::Key {
                key: egui::Key::W,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(ctx.input(|i| i.key_down(egui::Key::W)));
        frame(
            &ctx,
            &mut input,
            0.1,
            vec![egui::Event::WindowFocused(false)],
        );
        assert!(!ctx.input(|i| i.key_down(egui::Key::W)));
        assert!(ctx.input(|i| i.focused));
    }

    #[test]
    fn trigger_jitter_keeps_clicks_stable_but_deliberate_motion_drags() {
        let ctx = egui::Context::default();
        let pos = egui::pos2(80.0, 60.0);
        let mut input = NativeInput {
            active: true,
            pointer: Some(pos),
            ..Default::default()
        };
        frame(&ctx, &mut input, 0.0, vec![]);
        input.pressed = true;
        frame(&ctx, &mut input, 0.1, vec![]);
        input.pointer = Some(pos + egui::vec2(7.0, 0.0));
        frame(&ctx, &mut input, 0.2, vec![]);
        assert_eq!(ctx.pointer_hover_pos(), Some(pos));
        input.pressed = false;
        assert!(frame(&ctx, &mut input, 0.3, vec![]).0);

        input.pointer = Some(pos);
        input.pressed = true;
        frame(&ctx, &mut input, 0.4, vec![]);
        input.pointer = Some(pos + egui::vec2(30.0, 0.0));
        frame(&ctx, &mut input, 0.5, vec![]);
        assert_eq!(ctx.pointer_hover_pos(), input.pointer);
        input.pressed = false;
        assert!(!frame(&ctx, &mut input, 0.6, vec![]).0);
    }
    fn slider_frame(
        ctx: &egui::Context,
        input: &mut NativeInput,
        time: f64,
        value: &mut f64,
        logarithmic: bool,
        disabled: bool,
    ) -> egui::Response {
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 300.0),
            )),
            time: Some(time),
            ..Default::default()
        };
        input.merge(&mut raw);
        egui::ControllerSlider::set_input(ctx, input.active, input.controller_pointing());
        egui::ControllerSlider::set_cancelled(ctx, input.controller_cancelled());
        ctx.begin_pass(raw);
        #[allow(deprecated)]
        let response = egui::CentralPanel::default()
            .show(ctx, |ui| {
                ui.add_enabled_ui(!disabled, |ui| {
                    ui.put(
                        egui::Rect::from_min_size(egui::pos2(40.0, 40.0), egui::vec2(300.0, 30.0)),
                        if logarithmic {
                            egui::Slider::new(value, 1.0..=1000.0)
                                .logarithmic(true)
                                .text("Logarithmic")
                        } else {
                            egui::Slider::new(value, 0.0..=100.0)
                                .step_by(2.0)
                                .text("Integer steps")
                        },
                    )
                })
                .inner
            })
            .inner;
        let _ = ctx.end_pass();
        response
    }
    #[test]
    fn controller_slider_selects_without_jumping_and_ring_edits_respect_log_and_steps() {
        for logarithmic in [false, true] {
            let ctx = egui::Context::default();
            egui::ControllerSlider::set_circular(&ctx, true);
            let mut input = NativeInput {
                active: true,
                pointer: Some(egui::pos2(80.0, 50.0)),
                ..Default::default()
            };
            let mut value = if logarithmic { 10.0 } else { 20.0 };
            slider_frame(&ctx, &mut input, 0.0, &mut value, logarithmic, false);
            let before = value;
            input.pressed = true;
            slider_frame(&ctx, &mut input, 0.1, &mut value, logarithmic, false);
            let selected =
                egui::ControllerSlider::active(&ctx).expect("controller press selects slider");
            assert_eq!(
                value, before,
                "selection must not jump to where the ray hit the linear rail"
            );
            input.pointer = None;
            input.pressed = false;
            let t = if logarithmic { 2.0 / 3.0 } else { 0.53 };
            egui::ControllerSlider::request(&ctx, selected.id, t);
            let response = slider_frame(&ctx, &mut input, 0.2, &mut value, logarithmic, false);
            assert!(
                response.changed(),
                "existing setting callbacks must observe the ring edit"
            );
            let expected = if logarithmic { 100.0 } else { 54.0 };
            assert!((value - expected).abs() < 0.002);
            assert!(
                (egui::ControllerSlider::active(&ctx).unwrap().normalized
                    - if logarithmic { 2.0 / 3.0 } else { 0.54 })
                .abs()
                    < 0.001
            );
            egui::ControllerSlider::request(&ctx, selected.id, 0.9);
            slider_frame(&ctx, &mut input, 0.3, &mut value, logarithmic, true);
            assert!(
                egui::ControllerSlider::active(&ctx).is_none(),
                "disabled setting loses the dial"
            );
            assert!((value - expected).abs() < 0.002);
        }
    }
    #[test]
    fn closed_wheel_slider_drags_normally_off_rail_and_accepts_stick_edits() {
        let ctx = egui::Context::default();
        let mut input = NativeInput {
            active: true,
            pointer: Some(egui::pos2(80.0, 50.0)),
            ..Default::default()
        };
        let mut value = 20.0;
        slider_frame(&ctx, &mut input, 0.0, &mut value, false, false);
        input.pressed = true;
        slider_frame(&ctx, &mut input, 0.1, &mut value, false, false);
        let selected = egui::ControllerSlider::active(&ctx).unwrap();
        input.pointer = Some(egui::pos2(170.0, 110.0));
        let r = slider_frame(&ctx, &mut input, 0.2, &mut value, false, false);
        assert!(
            r.changed() && value > 60.0,
            "held drag follows the cursor beyond the rail"
        );
        input.pressed = false;
        slider_frame(&ctx, &mut input, 0.3, &mut value, false, false);
        egui::ControllerSlider::request(&ctx, selected.id, 0.3);
        assert!(slider_frame(&ctx, &mut input, 0.4, &mut value, false, false).changed());
        assert_eq!(value, 30.0);
        let stops = egui::ControllerSlider::active(&ctx).unwrap().stops;
        assert!(stops
            .iter()
            .any(|(t, label)| (*t - 0.4).abs() < 1e-5 && label == "40"));
    }
    #[test]
    fn normal_slider_tracking_loss_never_moves_value_to_the_cancel_position() {
        let ctx = egui::Context::default();
        let mut input = NativeInput {
            active: true,
            pointer: Some(egui::pos2(80.0, 50.0)),
            ..Default::default()
        };
        let mut value = 20.0;
        slider_frame(&ctx, &mut input, 0.0, &mut value, false, false);
        input.pressed = true;
        slider_frame(&ctx, &mut input, 0.1, &mut value, false, false);
        input.pointer = Some(egui::pos2(130.0, 50.0));
        slider_frame(&ctx, &mut input, 0.2, &mut value, false, false);
        let before = value;
        input.pointer = None;
        slider_frame(&ctx, &mut input, 0.3, &mut value, false, false);
        assert_eq!(value, before);
    }
    #[test]
    fn slider_detents_follow_the_widgets_logarithmic_scale() {
        let ctx = egui::Context::default();
        egui::ControllerSlider::set_circular(&ctx, true);
        let mut input = NativeInput {
            active: true,
            pointer: Some(egui::pos2(80.0, 50.0)),
            ..Default::default()
        };
        let mut value = 10.0;
        slider_frame(&ctx, &mut input, 0.0, &mut value, true, false);
        input.pressed = true;
        slider_frame(&ctx, &mut input, 0.1, &mut value, true, false);
        let stops = egui::ControllerSlider::active(&ctx).unwrap().stops;
        assert!(stops
            .iter()
            .any(|(t, label)| (*t - 2.0 / 3.0).abs() < 1e-5 && label == "100"));
    }
    #[test]
    fn leaving_vr_discards_any_queued_slider_edit() {
        let ctx = egui::Context::default();
        egui::ControllerSlider::set_circular(&ctx, true);
        let mut input = NativeInput {
            active: true,
            pointer: Some(egui::pos2(80.0, 50.0)),
            ..Default::default()
        };
        let mut value = 20.0;
        slider_frame(&ctx, &mut input, 0.0, &mut value, false, false);
        input.pressed = true;
        slider_frame(&ctx, &mut input, 0.1, &mut value, false, false);
        let selected = egui::ControllerSlider::active(&ctx).unwrap();
        egui::ControllerSlider::request(&ctx, selected.id, 0.9);
        input.active = false;
        input.pressed = false;
        input.pointer = None;
        slider_frame(&ctx, &mut input, 0.2, &mut value, false, false);
        assert_eq!(value, 20.0);
        assert!(egui::ControllerSlider::active(&ctx).is_none());
    }
}
