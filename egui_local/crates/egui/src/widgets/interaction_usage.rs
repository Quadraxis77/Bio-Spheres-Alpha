//! Opt-in interaction observations. The application owns aggregation and storage.
use crate::{Context, Id, Response};

#[derive(Clone, Debug)]
pub struct UsageEvent {
    pub id: Id,
    pub path: String,
    pub label: String,
    pub kind: &'static str,
    pub changed: bool,
    pub held: bool,
}

#[derive(Clone, Default)]
struct State {
    enabled: bool,
    path: Vec<String>,
    events: Vec<UsageEvent>,
    last_label: Option<(u64, Vec<String>, crate::Rect, String)>,
}

/// Optional observations of intentional input, never setting values.
pub struct InteractionUsage;
impl InteractionUsage {
    fn key() -> Id {
        Id::new("interaction_usage")
    }

    pub fn enable(ctx: &Context) {
        ctx.data_mut(|d| d.get_temp_mut_or_default::<State>(Self::key()).enabled = true);
    }

    pub fn scoped<R>(ctx: &Context, label: &str, body: impl FnOnce() -> R) -> R {
        let enabled = ctx.data_mut(|d| {
            let state = d.get_temp_mut_or_default::<State>(Self::key());
            if state.enabled {
                state.path.push(label.to_owned());
            }
            state.enabled
        });
        let result = body();
        if enabled {
            ctx.data_mut(|d| {
                d.get_temp_mut_or_default::<State>(Self::key()).path.pop();
            });
        }
        result
    }

    pub fn menu(ctx: &Context, id: Id, label: &str) {
        Self::record(ctx, id, label, "Menu", true, false);
    }

    pub(crate) fn label(ctx: &Context, rect: crate::Rect, text: &str) {
        // The normal UI often puts the caption beside/above a slider instead
        // of in Slider::text. Remember only a nearby short caption, never a
        // changing numeric readout or text from a previous frame/panel.
        if text.len() > 80
            || !text.chars().any(char::is_alphabetic)
            || text
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit() || c == '-' || c == '+')
        {
            return;
        }
        let pass = ctx.cumulative_pass_nr();
        ctx.data_mut(|d| {
            let state = d.get_temp_mut_or_default::<State>(Self::key());
            if state.enabled {
                state.last_label = Some((
                    pass,
                    state.path.clone(),
                    rect,
                    text.trim_end_matches(':').to_owned(),
                ));
            }
        });
    }

    pub(crate) fn slider_label(ctx: &Context, rect: crate::Rect, explicit: &str) -> String {
        if !explicit.is_empty() {
            return explicit.to_owned();
        }
        let pass = ctx.cumulative_pass_nr();
        ctx.data(|d| {
            let state = d.get_temp::<State>(Self::key())?;
            let (seen, path, caption, text) = state.last_label?;
            let beside = (caption.center().y - rect.center().y).abs() <= 12.0
                && (0.0..=80.0).contains(&(rect.left() - caption.right()));
            let above = (0.0..=24.0).contains(&(rect.top() - caption.bottom()))
                && (caption.left() - rect.left()).abs() < 24.0;
            (seen == pass && path == state.path && (beside || above)).then_some(text)
        })
        .unwrap_or_default()
    }

    pub fn slider(response: &Response, id: Id, label: &str) {
        let held = response.dragged() || response.is_pointer_button_down_on();
        let pressed = response.hovered() && response.ctx.input(|i| i.pointer.primary_pressed());
        Self::record(
            &response.ctx,
            id,
            label,
            "Slider",
            response.changed() || pressed,
            held,
        );
    }

    pub fn record(
        ctx: &Context,
        id: Id,
        label: &str,
        kind: &'static str,
        changed: bool,
        held: bool,
    ) {
        if !changed && !held {
            return;
        }
        ctx.data_mut(|d| {
            let state = d.get_temp_mut_or_default::<State>(Self::key());
            if !state.enabled {
                return;
            }
            state.events.push(UsageEvent {
                id,
                path: state.path.join(" / "),
                label: if label.is_empty() {
                    format!("Slider {:016x}", id.value())
                } else {
                    label.to_owned()
                },
                kind,
                changed,
                held,
            });
        });
    }

    pub fn drain(ctx: &Context) -> Vec<UsageEvent> {
        ctx.data_mut(|d| {
            std::mem::take(&mut d.get_temp_mut_or_default::<State>(Self::key()).events)
        })
    }
}
