//! A finite 300-degree arc with a bottom gap; dragging never wraps min to max.
use super::controls::Choice;
use glam::Vec2;
pub const RADIUS: f32 = 322.0;
pub const MIN_RADIUS: f32 = 310.0;
pub const MAX_RADIUS: f32 = 347.0;
pub const START: f32 = std::f32::consts::PI * 0.75;
pub const SWEEP: f32 = std::f32::consts::PI * 5.0 / 3.0;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Choice(Choice),
    Ui(egui::Id),
}
pub struct Dial {
    pub target: Target,
    pub label: String,
    pub value: String,
    pub normalized: f32,
    pub minimum: String,
    pub maximum: String,
    pub color: egui::Color32,
    pub pending: Option<f32>,
    pub stops: Vec<(f32, String)>,
    drag_angle: Option<f32>,
    pointer_angle: Option<f32>,
    stick_value: Option<f32>,
    armed: bool,
}
impl Dial {
    pub fn new(
        target: Target,
        label: String,
        normalized: f32,
        value: String,
        minimum: String,
        maximum: String,
        color: egui::Color32,
    ) -> Self {
        Self {
            target,
            label,
            value,
            normalized,
            minimum,
            maximum,
            color,
            pending: None,
            stops: Vec::new(),
            drag_angle: None,
            pointer_angle: None,
            stick_value: None,
            armed: false,
        }
    }
    fn commit(&mut self, raw: f32) {
        let raw = raw.clamp(0.0, 1.0);
        self.normalized = self
            .stops
            .iter()
            .min_by(|a, b| (a.0 - raw).abs().total_cmp(&(b.0 - raw).abs()))
            .filter(|s| (s.0 - raw).abs() <= 0.008)
            .map_or(raw, |s| s.0);
        self.pending = Some(self.normalized);
    }
    pub fn grabbing(&self) -> bool {
        self.drag_angle.is_some()
    }
    pub fn update(&mut self, pointer: Option<Vec2>, pressed: bool, edge: bool) {
        if !pressed {
            self.armed = true;
            self.drag_angle = None;
            self.pointer_angle = None;
            return;
        }
        let offset = pointer.map(|p| p - Vec2::splat(360.0));
        if let Some(unwrapped) = self.drag_angle {
            // Once acquired, the ray follows the infinite wheel plane. Moving
            // off the ring never releases the grab. Missing tracking or crossing
            // the center pauses motion; returning establishes a fresh baseline.
            if let Some(offset) = offset.filter(|p| p.length() >= 8.0) {
                let angle = offset.y.atan2(offset.x);
                if let Some(previous) = self.pointer_angle {
                    let delta = (angle - previous + std::f32::consts::PI)
                        .rem_euclid(std::f32::consts::TAU)
                        - std::f32::consts::PI;
                    let unwrapped = unwrapped + delta;
                    self.drag_angle = Some(unwrapped);
                    self.commit((unwrapped - START) / SWEEP);
                    if delta.abs() > 1e-5 {
                        self.stick_value = None;
                    }
                }
                self.pointer_angle = Some(angle);
            } else {
                self.pointer_angle = None;
            }
            return;
        }
        let Some(offset) = offset else {
            return;
        };
        if !(MIN_RADIUS..=MAX_RADIUS).contains(&offset.length()) {
            return;
        }
        if edge && self.armed {
            let angle = offset.y.atan2(offset.x);
            let mut progress = (angle - START).rem_euclid(std::f32::consts::TAU);
            if progress > std::f32::consts::TAU - 1e-5 {
                progress = 0.0;
            }
            if progress <= SWEEP + 1e-5 {
                progress = progress.min(SWEEP);
                self.drag_angle = Some(START + progress);
                self.pointer_angle = Some(angle);
                self.commit(progress / SWEEP);
                self.stick_value = None;
            }
            self.armed = false;
        }
    }
    pub fn adjust_stick(&mut self, stick: Vec2, dt: f32) {
        // Either right-stick direction works: right/up increases, left/down
        // decreases. The dominant axis prevents diagonals doubling the speed.
        let axis = if stick.x.abs() >= stick.y.abs() {
            stick.x
        } else {
            stick.y
        };
        if !axis.is_finite() || axis.abs() <= 0.18 {
            self.stick_value = None;
            return;
        }
        let amount = ((axis.abs() - 0.18) / 0.82).clamp(0.0, 1.0);
        let value = self.stick_value.unwrap_or(self.normalized)
            + axis.signum() * amount * amount * 0.35 * dt.clamp(0.0, 0.1);
        let raw = value.clamp(0.0, 1.0);
        self.commit(raw);
        // Preserve sub-step edits while the stick is held, even when the
        // underlying integer slider rounds its displayed value each frame.
        self.stick_value = Some(raw);
        if self.grabbing() {
            self.drag_angle = Some(START + raw * SWEEP);
        }
    }
    pub fn draw(&self, painter: &egui::Painter, palette: crate::ui::ui_system::ActivePalette) {
        let color = match self.target {
            Target::Choice(choice) => choice.category_with_palette(palette).1,
            Target::Ui(_) => palette.accent_primary,
        };

        let center = egui::pos2(360.0, 360.0);
        // Outer labels sit over the world rather than the console body. Give
        // them themed backing so dark text in a light theme stays readable.
        let caption = |position: egui::Pos2, text: &str, size: f32| {
            let galley = painter.layout_no_wrap(
                text.to_owned(),
                egui::FontId::proportional(size),
                palette.text_primary,
            );
            let rect =
                egui::Rect::from_center_size(position, galley.size()).expand2(egui::vec2(5.0, 3.0));
            painter.rect_filled(rect, 5.0, super::theme::alpha(palette.bg_panel, 235));
            painter.galley(position - galley.size() * 0.5, galley, palette.text_primary);
        };

        let points = |end: f32| {
            (0..=120)
                .map(|i| {
                    let a = START + SWEEP * end * i as f32 / 120.0;
                    center + egui::vec2(a.cos(), a.sin()) * RADIUS
                })
                .collect::<Vec<_>>()
        };
        painter.add(egui::Shape::line(
            points(1.0),
            egui::Stroke::new(22.0, super::theme::alpha(palette.bg_widget, 225)),
        ));
        painter.add(egui::Shape::line(
            points(self.normalized),
            egui::Stroke::new(12.0, color),
        ));
        for i in 0..=20 {
            let a = START + SWEEP * i as f32 / 20.0;
            let dir = egui::vec2(a.cos(), a.sin());
            painter.line_segment(
                [
                    center + dir * (RADIUS - 12.0),
                    center + dir * (RADIUS - 17.0),
                ],
                egui::Stroke::new(1.0, super::theme::alpha(palette.text_secondary, 180)),
            );
        }
        let a = START + SWEEP * self.normalized;
        let handle = center + egui::vec2(a.cos(), a.sin()) * RADIUS;
        painter.circle_filled(handle, 12.0, palette.text_primary);
        painter.circle_filled(handle, 7.0, color);
        let endpoints = vec![(0.0, self.minimum.clone()), (1.0, self.maximum.clone())];
        let stops = if self.stops.is_empty() {
            &endpoints
        } else {
            &self.stops
        };
        for (t, text) in stops {
            let a = START + SWEEP * t;
            let dir = egui::vec2(a.cos(), a.sin());
            painter.line_segment(
                [
                    center + dir * (RADIUS - 20.0),
                    center + dir * (RADIUS + 10.0),
                ],
                egui::Stroke::new(2.0, palette.text_primary),
            );
            let a = START + SWEEP * t;
            caption(
                center + egui::vec2(a.cos(), a.sin()) * (RADIUS - 35.0),
                text,
                14.0,
            );
        }
        caption(
            center + egui::vec2(0.0, 300.0),
            "Right stick / hold trigger + drag",
            16.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pixel(t: f32) -> Vec2 {
        let a = START + SWEEP * t;
        Vec2::splat(360.0) + Vec2::new(a.cos(), a.sin()) * RADIUS
    }
    fn dial() -> Dial {
        Dial::new(
            Target::Choice(Choice::Gravity),
            "Gravity".into(),
            0.5,
            "0".into(),
            "-100".into(),
            "100".into(),
            egui::Color32::WHITE,
        )
    }
    #[test]
    fn pointer_and_stick_snap_to_labeled_detents_without_getting_stuck() {
        let mut d = dial();
        d.stops = vec![(0.0, "-100".into()), (0.5, "0".into()), (1.0, "100".into())];
        d.update(None, false, false);
        d.update(Some(pixel(0.504)), true, true);
        assert_eq!(d.pending, Some(0.5));
        d.update(Some(pixel(0.52)), true, false);
        assert!((d.normalized - 0.52).abs() < 0.001);
        d.update(None, false, false);
        d.normalized = 0.5;
        for _ in 0..20 {
            d.adjust_stick(Vec2::new(0.5, 0.0), 1.0 / 120.0);
        }
        assert!(
            d.normalized > 0.508,
            "raw stick accumulation escapes the snap band"
        );
    }
    #[test]
    fn selecting_a_slider_cannot_reuse_the_held_selection_trigger() {
        let mut d = dial();
        d.update(Some(pixel(0.8)), true, true);
        assert!(d.pending.is_none());
        d.update(None, false, false);
        d.update(Some(pixel(0.8)), true, true);
        assert!((d.pending.unwrap() - 0.8).abs() < 0.001);
    }
    #[test]
    fn full_range_and_wrapping_angles_are_continuous_and_never_wrap_values() {
        for t in [0.0, 0.1, 0.4, 0.7, 1.0] {
            let mut d = dial();
            d.update(None, false, false);
            d.update(Some(pixel(t)), true, true);
            assert!((d.pending.unwrap() - t).abs() < 0.001);
        }
        let mut d = dial();
        d.update(None, false, false);
        d.update(Some(pixel(0.95)), true, true);
        d.update(Some(pixel(1.04)), true, false);
        assert_eq!(
            d.normalized, 1.0,
            "dragging past maximum clamps in the bottom gap"
        );
        d.update(Some(pixel(1.08)), true, false);
        assert_eq!(d.normalized, 1.0);
        d.update(Some(pixel(0.98)), true, false);
        assert!((d.normalized - 0.98).abs() < 0.001);
    }
    #[test]
    fn grab_tracks_cursor_beyond_the_ring_and_releases_only_with_trigger() {
        let mut d = dial();
        d.update(None, false, false);
        d.update(Some(pixel(0.2)), true, true);
        d.pending = None;
        let outside = Vec2::splat(360.0) + (pixel(0.4) - Vec2::splat(360.0)) * 4.0;
        d.update(Some(outside), true, false);
        assert!(d.grabbing());
        assert!((d.normalized - 0.4).abs() < 0.001);
        let inside = Vec2::splat(360.0) + (pixel(0.5) - Vec2::splat(360.0)) * 0.4;
        d.update(Some(inside), true, false);
        assert!((d.normalized - 0.5).abs() < 0.001);
        d.pending = None;
        d.update(None, true, false);
        assert!(d.grabbing(), "missing ray freezes a held grab");
        d.update(Some(pixel(0.8)), true, false);
        assert!(
            d.pending.is_none(),
            "restored tracking rebases without a jump"
        );
        d.update(Some(pixel(0.9)), true, false);
        assert!((d.normalized - 0.6).abs() < 0.001);
        d.update(None, false, false);
        assert!(!d.grabbing());
        d.pending = None;
        d.update(Some(pixel(0.1)), false, false);
        assert!(
            d.pending.is_none(),
            "released cursor cannot change the setting"
        );
    }
    #[test]
    fn stick_is_frame_rate_independent_and_accumulates_integer_substeps() {
        for hz in [60, 120] {
            let mut d = dial();
            for _ in 0..hz {
                d.adjust_stick(Vec2::X, 1.0 / hz as f32);
                // Simulate a setting that rounds to coarse 10% steps.
                d.normalized = (d.normalized * 10.0).round() / 10.0;
            }
            assert!((d.stick_value.unwrap() - 0.85).abs() < 0.001);
            d.adjust_stick(Vec2::ZERO, 0.1);
            let before = d.normalized;
            d.adjust_stick(Vec2::NEG_Y, 0.1);
            assert!(d.normalized < before);
            d.normalized = 0.999;
            d.stick_value = None;
            d.adjust_stick(Vec2::ONE, 0.1);
            assert_eq!(d.normalized, 1.0);
        }
    }
}
