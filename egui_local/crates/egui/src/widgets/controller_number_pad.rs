//! A numeric keypad overlay for controller-driven UI.

use crate::{Align2, Area, Button, Context, Id, Order, Vec2};

#[derive(Clone)]
struct Selection {
    id: Id,
    prefix: String,
    suffix: String,
    value: String,
    allow_decimal: bool,
    replace_on_next_entry: bool,
    last_seen: u64,
}

#[derive(Clone, Default)]
struct State {
    enabled: bool,
    selected: Option<Selection>,
}

#[derive(Clone, Copy)]
enum Key {
    Digit(char),
    Decimal,
    Sign,
    Backspace,
    Clear,
    Done,
}

/// Displays a number pad for the focused `DragValue` when enabled by the host.
pub struct ControllerNumberPad;

impl ControllerNumberPad {
    fn key() -> Id {
        Id::new("controller_number_pad")
    }

    /// Enable the overlay for a VR UI frame, or hide it outside VR.
    pub fn set_enabled(ctx: &Context, enabled: bool) {
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            state.enabled = enabled;
            if !enabled {
                state.selected = None;
            }
        });
    }

    /// Track the numeric edit buffer belonging to the focused `DragValue`.
    pub fn register(
        ctx: &Context,
        id: Id,
        prefix: &str,
        suffix: &str,
        value: &str,
        allow_decimal: bool,
    ) {
        let pass = ctx.cumulative_pass_nr();
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if !state.enabled {
                return;
            }

            let replace_on_next_entry = match state.selected.as_ref() {
                Some(selection) if selection.id == id && selection.last_seen + 1 >= pass => {
                    selection.replace_on_next_entry
                }
                _ => true,
            };

            state.selected = Some(Selection {
                id,
                prefix: prefix.to_owned(),
                suffix: suffix.to_owned(),
                value: value.to_owned(),
                allow_decimal,
                replace_on_next_entry,
                last_seen: pass,
            });
        });
    }

    /// Preserve the text caret's normal editing behavior after physical typing.
    pub fn mark_text_edited(ctx: &Context, id: Id) {
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if let Some(selection) = state
                .selected
                .as_mut()
                .filter(|selection| selection.id == id)
            {
                selection.replace_on_next_entry = false;
            }
        });
    }

    /// Whether a numeric input is currently being edited.
    pub fn is_active(ctx: &Context) -> bool {
        Self::selection(ctx).is_some()
    }

    /// Render the keypad on a controller-owned surface while applying edits to
    /// the context that owns the focused numeric input.
    pub fn show_on_controller(
        controller_ctx: &Context,
        value_ctx: &Context,
        pointer: Option<crate::Pos2>,
    ) {
        let Some(selection) = Self::selection(value_ctx) else {
            return;
        };

        let mut clicked_key = None;
        Area::new(Id::new("controller_number_pad_surface"))
            .order(Order::Foreground)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(controller_ctx, |ui| {
                ui.set_width(300.0);
                ui.group(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.label("Editing numeric input");
                        ui.add(
                            crate::Label::new(
                                crate::RichText::new(format!(
                                    "{}{}{}",
                                    selection.prefix, selection.value, selection.suffix
                                ))
                                .monospace(),
                            )
                            .wrap_mode(crate::TextWrapMode::Extend),
                        );
                    });
                    ui.add_space(5.0);

                    for row in [["7", "8", "9"], ["4", "5", "6"], ["1", "2", "3"]] {
                        ui.horizontal(|ui| {
                            for digit in row {
                                if ui.add_sized([88.0, 44.0], Button::new(digit)).clicked() {
                                    clicked_key = digit.chars().next().map(Key::Digit);
                                }
                            }
                        });
                    }

                    ui.horizontal(|ui| {
                        if ui.add_sized([88.0, 44.0], Button::new("+/-")).clicked() {
                            clicked_key = Some(Key::Sign);
                        }
                        if ui.add_sized([88.0, 44.0], Button::new("0")).clicked() {
                            clicked_key = Some(Key::Digit('0'));
                        }
                        if ui
                            .add_enabled_ui(selection.allow_decimal, |ui| {
                                ui.add_sized([88.0, 44.0], Button::new("."))
                            })
                            .inner
                            .clicked()
                        {
                            clicked_key = Some(Key::Decimal);
                        }
                    });

                    ui.horizontal(|ui| {
                        if ui.add_sized([88.0, 44.0], Button::new("Del")).clicked() {
                            clicked_key = Some(Key::Backspace);
                        }
                        if ui.add_sized([88.0, 44.0], Button::new("Clear")).clicked() {
                            clicked_key = Some(Key::Clear);
                        }
                        if ui.add_sized([88.0, 44.0], Button::new("Done")).clicked() {
                            clicked_key = Some(Key::Done);
                        }
                    });
                });
            });

        if let Some(key) = clicked_key {
            Self::apply_key(value_ctx, &selection, key);
        }

        if let Some(pointer) = pointer {
            let painter = controller_ctx.layer_painter(crate::LayerId::new(
                Order::Tooltip,
                Id::new("controller_number_pad_pointer"),
            ));
            painter.circle_filled(pointer, 8.0, crate::Color32::from_rgb(0, 200, 160));
            painter.circle_stroke(
                pointer,
                10.0,
                crate::Stroke::new(2.0, crate::Color32::WHITE),
            );
        }
    }

    fn selection(ctx: &Context) -> Option<Selection> {
        let pass = ctx.cumulative_pass_nr();
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if !state.enabled {
                return None;
            }
            if state
                .selected
                .as_ref()
                .is_some_and(|selection| selection.last_seen + 1 < pass)
            {
                state.selected = None;
            }
            state.selected.clone()
        })
    }

    fn apply_key(ctx: &Context, selection: &Selection, key: Key) {
        if matches!(key, Key::Done) {
            ctx.memory_mut(|memory| memory.surrender_focus(selection.id));
            ctx.data_mut(|data| {
                let state = data.get_temp_mut_or_default::<State>(Self::key());
                if state
                    .selected
                    .as_ref()
                    .is_some_and(|current| current.id == selection.id)
                {
                    state.selected = None;
                }
            });
            return;
        }

        let mut value = ctx
            .data_mut(|data| data.get_temp::<String>(selection.id))
            .unwrap_or_else(|| selection.value.clone());
        let mut replace_on_next_entry = selection.replace_on_next_entry;

        match key {
            Key::Digit(digit) => {
                if replace_on_next_entry {
                    value.clear();
                }
                value.push(digit);
                replace_on_next_entry = false;
            }
            Key::Decimal if selection.allow_decimal => {
                if replace_on_next_entry || value.is_empty() {
                    value = "0.".to_owned();
                } else if value == "-" || value == "−" {
                    value.push_str("0.");
                } else if !value.contains('.') {
                    value.push('.');
                }
                replace_on_next_entry = false;
            }
            Key::Sign => {
                if replace_on_next_entry || value.is_empty() {
                    value = "-".to_owned();
                } else if value.starts_with('-') || value.starts_with('−') {
                    value.remove(0);
                } else {
                    value.insert(0, '-');
                }
                replace_on_next_entry = false;
            }
            Key::Backspace => {
                if replace_on_next_entry {
                    value = "0".to_owned();
                } else {
                    value.pop();
                }
                replace_on_next_entry = false;
            }
            Key::Clear => {
                value = "0".to_owned();
                replace_on_next_entry = true;
            }
            Key::Decimal | Key::Done => return,
        }

        ctx.data_mut(|data| data.insert_temp(selection.id, value.clone()));
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if let Some(current) = state
                .selected
                .as_mut()
                .filter(|current| current.id == selection.id)
            {
                current.value = value;
                current.replace_on_next_entry = replace_on_next_entry;
            }
        });
        ctx.memory_mut(|memory| memory.request_focus(selection.id));
    }
}

#[cfg(test)]
mod tests {
    use super::{ControllerNumberPad, Key, Selection, State};
    use crate::{Context, Id};

    fn selection(value: &str, allow_decimal: bool) -> Selection {
        Selection {
            id: Id::new("test_numeric_field"),
            prefix: String::new(),
            suffix: String::new(),
            value: value.to_owned(),
            allow_decimal,
            replace_on_next_entry: true,
            last_seen: 1,
        }
    }

    fn apply(ctx: &Context, selection: &Selection, key: Key) -> String {
        ControllerNumberPad::apply_key(ctx, selection, key);
        ctx.data_mut(|data| data.get_temp::<String>(selection.id).unwrap())
    }

    #[test]
    fn keypad_edits_numeric_buffer_and_supports_clear_and_backspace() {
        let ctx = Context::default();
        ControllerNumberPad::set_enabled(&ctx, true);
        let field = selection("123", true);

        assert_eq!(apply(&ctx, &field, Key::Digit('4')), "4");

        let field = selection("4", true);
        let mut field = field;
        field.replace_on_next_entry = false;
        assert_eq!(apply(&ctx, &field, Key::Digit('5')), "45");

        let field = selection("45", true);
        let mut field = field;
        field.replace_on_next_entry = false;
        assert_eq!(apply(&ctx, &field, Key::Decimal), "45.");

        let field = selection("45.", true);
        let mut field = field;
        field.replace_on_next_entry = false;
        assert_eq!(apply(&ctx, &field, Key::Backspace), "45");
        assert_eq!(apply(&ctx, &field, Key::Clear), "0");

        let ctx = Context::default();
        ControllerNumberPad::set_enabled(&ctx, true);
        let field = selection("-", true);
        let mut field = field;
        field.replace_on_next_entry = false;
        assert_eq!(apply(&ctx, &field, Key::Decimal), "-0.");
        assert_eq!(apply(&ctx, &field, Key::Sign), "0.");
    }

    #[test]
    fn decimal_key_is_ignored_for_integral_fields() {
        let ctx = Context::default();
        ControllerNumberPad::set_enabled(&ctx, true);
        let field = selection("12", false);

        ControllerNumberPad::apply_key(&ctx, &field, Key::Decimal);
        assert_eq!(ctx.data_mut(|data| data.get_temp::<String>(field.id)), None);
    }

    #[test]
    fn disabling_keypad_clears_selection_and_prevents_registration() {
        let ctx = Context::default();
        ControllerNumberPad::set_enabled(&ctx, false);
        ControllerNumberPad::register(&ctx, Id::new("field"), "", "", "3", true);

        let state = ctx.data_mut(|data| data.get_temp::<State>(ControllerNumberPad::key()));
        assert!(state.is_some_and(|state| state.selected.is_none()));
    }
}
