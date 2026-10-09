//! Optional external slider input, shared by standard and application-specific sliders.
use crate::{Context, Id, PointerButton, Response, Ui};

#[derive(Clone, Debug)]
pub struct ControllerSliderSelection {
    pub id: Id,
    pub label: String,
    pub value: String,
    pub normalized: f64,
    pub minimum: String,
    pub maximum: String,
    pub last_seen: u64,
    pub stops: Vec<(f64, String)>,
}
#[derive(Clone, Default)]
struct State {
    enabled: bool,
    pointing: bool,
    circular: bool,
    cancelled: bool,
    pointer_press_blocked: bool,
    selected: Option<ControllerSliderSelection>,
    pending: Option<(Id, f64)>,
}
pub struct ControllerSlider;
impl ControllerSlider {
    fn key() -> Id {
        Id::new("controller_slider_input")
    }
    pub fn set_input(ctx: &Context, enabled: bool, pointing: bool) {
        let primary_down = ctx.input(|input| input.pointer.button_down(PointerButton::Primary));
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            state.enabled = enabled;
            state.pointing = pointing;
            if !primary_down {
                state.pointer_press_blocked = false;
            }
            if !enabled {
                state.selected = None;
                state.pending = None;
            }
        });
    }
    pub fn set_circular(ctx: &Context, circular: bool) {
        ctx.data_mut(|data| data.get_temp_mut_or_default::<State>(Self::key()).circular = circular);
    }
    pub fn set_cancelled(ctx: &Context, cancelled: bool) {
        ctx.data_mut(|data| {
            data.get_temp_mut_or_default::<State>(Self::key()).cancelled = cancelled
        });
    }
    pub fn pointer_editing_blocked(ctx: &Context) -> bool {
        ctx.data(|data| {
            data.get_temp::<State>(Self::key())
                .is_some_and(|s| s.cancelled || s.pointer_press_blocked)
        })
    }
    /// Readable detents in value space; callers map them through their own slider scale.
    pub fn significant_values(min: f64, max: f64, logarithmic: bool) -> Vec<f64> {
        let (lo, hi) = (min.min(max), min.max(max));
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return vec![];
        }
        let mut values = vec![lo, hi];
        if logarithmic && lo > 0.0 {
            let first = lo.log10().floor() as i32;
            let last = hi.log10().ceil() as i32;
            let stride = ((last - first + 1) as usize).div_ceil(4).max(1);
            for power in (first..=last).step_by(stride) {
                for factor in [1.0, 2.0, 5.0] {
                    let value = factor * 10.0_f64.powi(power);
                    if value > lo && value < hi {
                        values.push(value);
                    }
                }
            }
        } else {
            let rough = (hi - lo) / 6.0;
            let power = 10.0_f64.powf(rough.log10().floor());
            let factor = [1.0, 2.0, 2.5, 5.0, 10.0]
                .into_iter()
                .find(|f| *f * power >= rough)
                .unwrap_or(10.0);
            let step = factor * power;
            for i in 0..=8 {
                let value = (lo / step).ceil() * step + i as f64 * step;
                if value > lo && value < hi {
                    values.push(value);
                }
            }
        }
        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() <= (hi - lo) * 1e-8);
        values
    }
    pub fn stop_label(value: f64) -> String {
        if value != 0.0 && (value.abs() < 0.001 || value.abs() >= 1e6) {
            return format!("{value:.1e}");
        }
        format!("{value:.3}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
    pub fn pointing(ctx: &Context) -> bool {
        ctx.data(|data| {
            data.get_temp::<State>(Self::key())
                .is_some_and(|s| s.pointing)
        })
    }
    pub fn active(ctx: &Context) -> Option<ControllerSliderSelection> {
        let pass = ctx.cumulative_pass_nr();
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if state
                .selected
                .as_ref()
                .is_some_and(|s| s.last_seen + 1 < pass)
            {
                state.selected = None;
                state.pending = None;
            }
            state.selected.clone()
        })
    }
    pub fn close(ctx: &Context) {
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            state.selected = None;
            state.pending = None;
        });
    }
    pub fn request(ctx: &Context, id: Id, normalized: f64) {
        if !normalized.is_finite() {
            return;
        }
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if state.enabled && state.selected.as_ref().is_some_and(|s| s.id == id) {
                state.pending = Some((id, normalized.clamp(0.0, 1.0)));
            }
        });
    }

    pub fn refresh(ctx: &Context, id: Id, normalized: f64, value: String) {
        ctx.data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if let Some(selected) = state.selected.as_mut().filter(|s| s.id == id) {
                selected.normalized = normalized;
                selected.value = value;
            }
        });
    }
    pub fn selected(ctx: &Context, id: Id) -> bool {
        ctx.data(|data| {
            data.get_temp::<State>(Self::key())
                .is_some_and(|s| s.selected.as_ref().is_some_and(|s| s.id == id))
        })
    }

    /// Call before the widget's own pointer editing. A controller press selects
    /// the widget without jumping its value; queued normalized edits use its
    /// existing range/log/rounding logic and must mark the response changed.
    pub fn interact(
        ui: &Ui,
        response: &Response,
        label: &str,
        value: String,
        normalized: f64,
        minimum: String,
        maximum: String,
        stops: Vec<(f64, String)>,
    ) -> Option<f64> {
        let pass = ui.ctx().cumulative_pass_nr();
        let press =
            response.hovered() && ui.input(|i| i.pointer.button_pressed(PointerButton::Primary));
        let enabled = ui.is_enabled();
        ui.ctx().data_mut(|data| {
            let state = data.get_temp_mut_or_default::<State>(Self::key());
            if !state.enabled {
                return None;
            }
            if enabled && (state.pointing || !state.circular) && press {
                state.pointer_press_blocked = !state
                    .selected
                    .as_ref()
                    .is_some_and(|selected| selected.id == response.id);
                state.selected = Some(ControllerSliderSelection {
                    id: response.id,
                    label: if label.is_empty() {
                        "Selected slider".to_owned()
                    } else {
                        label.to_owned()
                    },
                    value: value.clone(),
                    normalized,
                    minimum: minimum.clone(),
                    maximum: maximum.clone(),
                    last_seen: pass,
                    stops: stops.clone(),
                });
                state.pending = None;
            }
            if !state.selected.as_ref().is_some_and(|s| s.id == response.id) {
                return None;
            }
            if !enabled {
                state.selected = None;
                state.pending = None;
                return None;
            }
            if let Some(selected) = &mut state.selected {
                selected.last_seen = pass;
                selected.normalized = normalized;
                selected.value = value;
                selected.minimum = minimum;
                selected.maximum = maximum;
                selected.stops = stops;
            }
            if state
                .pending
                .as_ref()
                .is_some_and(|(id, _)| *id == response.id)
            {
                state.pending.take().map(|(_, value)| value)
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_edit_block_is_released_after_focus_press() {
        let ctx = Context::default();
        ControllerSlider::set_input(&ctx, true, true);
        ctx.data_mut(|data| {
            data.get_temp_mut_or_default::<State>(ControllerSlider::key())
                .pointer_press_blocked = true;
        });

        assert!(ControllerSlider::pointer_editing_blocked(&ctx));

        ControllerSlider::set_input(&ctx, true, true);
        assert!(!ControllerSlider::pointer_editing_blocked(&ctx));
    }
}
