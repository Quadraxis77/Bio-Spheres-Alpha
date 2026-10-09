//! A bounded live-population history drawn in an annulus around the wrist menu.
use std::collections::VecDeque;

const WINDOW: f64 = 60.0;
const INTERVAL: f64 = 0.5;
const MAX_SAMPLES: usize = 121;
const INNER: f32 = 281.0;
const OUTER: f32 = 307.0;
const START: f32 = std::f32::consts::PI * 5.0 / 6.0;
const SWEEP: f32 = std::f32::consts::PI * 4.0 / 3.0;

#[derive(Default)]
pub struct Population {
    pub count: u32,
    samples: VecDeque<(f64, u32)>,
    now: f64,
}
impl Population {
    pub fn clear(&mut self) {
        self.count = 0;
        self.samples.clear();
        self.now = 0.0;
    }
    pub fn observe(&mut self, time: f64, count: u32) {
        if !time.is_finite() || time < 0.0 {
            return;
        }
        if time < self.now {
            self.clear();
        }
        self.now = time;
        self.count = count;
        // Keep the current endpoint live, even when insertion/removal happens
        // while paused. Older points are sampled at simulation-time intervals.
        if self
            .samples
            .back()
            .is_some_and(|last| time - last.0 < INTERVAL)
        {
            self.samples.back_mut().unwrap().1 = count;
        } else {
            self.samples.push_back((time, count));
        }
        while self.samples.len() > MAX_SAMPLES
            || self.samples.front().is_some_and(|p| p.0 < time - WINDOW)
        {
            self.samples.pop_front();
        }
    }
    pub fn count_text(&self) -> String {
        format_count(self.count)
    }
    pub fn draw(
        &self,
        painter: &egui::Painter,
        show_legend: bool,
        palette: crate::ui::ui_system::ActivePalette,
    ) {
        let center = egui::pos2(360.0, 360.0);
        let tint = palette.status_ok;
        let point = |t: f32, radius: f32| {
            let angle = START + SWEEP * t;
            center + egui::vec2(angle.cos(), angle.sin()) * radius
        };
        let arc = |radius: f32| {
            (0..=120)
                .map(|i| point(i as f32 / 120.0, radius))
                .collect::<Vec<_>>()
        };
        let mut background = egui::Mesh::default();
        for i in 0..=120 {
            let t = i as f32 / 120.0;
            for radius in [INNER, OUTER] {
                background
                    .colored_vertex(point(t, radius), super::theme::alpha(palette.bg_panel, 180));
            }
            if i > 0 {
                let b = i * 2;
                background.add_triangle(b - 2, b - 1, b);
                background.add_triangle(b, b - 1, b + 1);
            }
        }
        painter.add(egui::Shape::mesh(background));
        for radius in [INNER, (INNER + OUTER) * 0.5, OUTER] {
            painter.add(egui::Shape::line(
                arc(radius),
                egui::Stroke::new(1.0, super::theme::alpha(tint, 65)),
            ));
        }
        let peak = self.samples.iter().map(|p| p.1).max().unwrap_or(0);
        let scale = peak.max(1) as f32;
        let mut trace = Vec::with_capacity(self.samples.len());
        for &(time, count) in &self.samples {
            let t = (1.0 - (self.now - time) / WINDOW).clamp(0.0, 1.0) as f32;
            trace.push(point(t, INNER + (OUTER - INNER) * count as f32 / scale));
        }
        if let Some(last) = trace.last().copied() {
            if trace.len() > 1 {
                painter.add(egui::Shape::line(trace, egui::Stroke::new(2.5, tint)));
            }
            painter.circle_filled(last, 3.0, tint);
        }
        if show_legend {
            let galley = painter.layout_no_wrap(
                format!("60 sim seconds  /  peak {}", format_count(peak)),
                egui::FontId::proportional(12.0),
                palette.text_primary,
            );
            let position = center + egui::vec2(0.0, 290.0);
            let rect =
                egui::Rect::from_center_size(position, galley.size()).expand2(egui::vec2(5.0, 3.0));
            painter.rect_filled(rect, 5.0, super::theme::alpha(palette.bg_panel, 235));
            painter.galley(position - galley.size() * 0.5, galley, palette.text_primary);
        }
    }
}
fn format_count(count: u32) -> String {
    let digits = count.to_string();
    let mut result = String::new();
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn population_tracks_paused_changes_and_stays_bounded_at_high_refresh_rates() {
        let mut history = Population::default();
        for frame in 0..120 * 180 {
            history.observe(frame as f64 / 120.0, frame as u32);
        }
        assert!(history.samples.len() <= MAX_SAMPLES);
        assert!(history.samples.front().unwrap().0 >= history.now - WINDOW);
        let length = history.samples.len();
        history.observe(history.now, 12345);
        assert_eq!(history.samples.len(), length);
        assert_eq!(history.samples.back().unwrap().1, 12345);
        assert_eq!(history.count_text(), "12,345");
        history.observe(0.0, 0);
        assert_eq!(
            history.samples.len(),
            1,
            "reset cannot connect unrelated runs"
        );
        assert_eq!(history.count, 0);
    }
}
