//! Local, grouped control usage for planning the wrist wheel.
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BURST_SECONDS: f64 = 2.0;
type Shared = Arc<Mutex<Tracker>>;
fn key() -> egui::Id {
    egui::Id::new("biospheres_control_usage")
}
fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    scene: String,
    surface: String,
    menu: String,
    label: String,
    kind: String,
    uses: u64,
    first_used: u64,
    last_used: u64,
    #[serde(default)]
    daily: BTreeMap<u64, u64>,
}

#[derive(Default, Serialize, Deserialize)]
struct History {
    #[serde(default = "version")]
    version: u32,
    entries: BTreeMap<String, Entry>,
}
fn version() -> u32 {
    1
}

#[derive(Default)]
struct Burst {
    last_event: f64,
    last_sample: f64,
    held: bool,
}

struct Tracker {
    history: History,
    bursts: BTreeMap<String, Burst>,
    last_control: Option<String>,
    started: Instant,
    saved: Instant,
    dirty: bool,
    path: PathBuf,
}

impl Tracker {
    fn load(path: PathBuf) -> Self {
        let history = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(history) => history,
                Err(error) => {
                    log::warn!("Cannot read control usage history: {error}");
                    // Preserve malformed history for recovery before the next save.
                    if let Err(error) = std::fs::copy(&path, path.with_extension("invalid.json")) {
                        log::warn!("Cannot preserve control usage history: {error}");
                    }
                    History {
                        version: 1,
                        ..Default::default()
                    }
                }
            },
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    log::warn!("Cannot load control usage: {error}");
                }
                History {
                    version: 1,
                    ..Default::default()
                }
            }
        };
        Self {
            history,
            bursts: BTreeMap::new(),
            last_control: None,
            started: Instant::now(),
            saved: Instant::now(),
            dirty: false,
            path,
        }
    }

    fn observe(
        &mut self,
        scene: &str,
        surface: &str,
        event: &egui::UsageEvent,
        now: f64,
        wall_time: u64,
    ) {
        // Stable egui ID distinguishes repeated labels (e.g. X/Y/Z sliders).
        // Scene, menu and surface keep wheel, VR screen and desktop separate.
        let identity = format!(
            "{scene}|{surface}|{}|{}|{}|{:016x}",
            event.path,
            event.kind,
            event.label,
            event.id.value()
        );
        let previous = self.bursts.get(&identity);
        let continuous_drag =
            event.held && previous.is_some_and(|b| b.held && now - b.last_sample < 0.25);
        let same_control = self.last_control.as_ref() == Some(&identity);
        let recent = previous.is_some_and(|b| now - b.last_event <= BURST_SECONDS);
        let new_burst = event.changed && (!same_control || !(recent || continuous_drag));
        let burst = self.bursts.entry(identity.clone()).or_default();
        burst.last_sample = now;
        burst.held = event.held;
        if event.changed || continuous_drag {
            burst.last_event = now;
        }
        if event.changed {
            self.last_control = Some(identity.clone());
        }
        if !new_burst {
            return;
        }
        let entry = self
            .history
            .entries
            .entry(identity)
            .or_insert_with(|| Entry {
                scene: scene.to_owned(),
                surface: surface.to_owned(),
                menu: event.path.clone(),
                label: event.label.clone(),
                kind: event.kind.to_owned(),
                uses: 0,
                first_used: wall_time,
                last_used: wall_time,
                daily: BTreeMap::new(),
            });
        entry.uses += 1;
        entry.last_used = wall_time;
        *entry.daily.entry(wall_time / 86400).or_default() += 1;
        self.dirty = true;
    }

    fn save(&mut self) {
        if !self.dirty {
            return;
        }
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let data = serde_json::to_vec_pretty(&self.history)?;
            let temp = self.path.with_extension("tmp");
            std::fs::write(&temp, data)?;
            std::fs::rename(temp, &self.path)?;
            Ok(())
        })();
        match result {
            Ok(()) => self.dirty = false,
            Err(error) => log::warn!("Cannot save control usage: {error}"),
        }
        self.saved = Instant::now();
    }
}
impl Drop for Tracker {
    fn drop(&mut self) {
        self.save();
    }
}

fn tracker(ctx: &egui::Context) -> Option<Shared> {
    ctx.data(|d| d.get_temp::<Shared>(key()))
}

pub fn install(ctx: &egui::Context) {
    let tracker = Arc::new(Mutex::new(Tracker::load(crate::app_dirs::config_file(
        "control_usage.json",
    ))));
    ctx.data_mut(|d| d.insert_temp(key(), tracker));
    egui::InteractionUsage::enable(ctx);
}

pub fn flush(ctx: &egui::Context) {
    if let Some(shared) = tracker(ctx) {
        shared.lock().unwrap().save();
    }
}

pub fn collect(ctx: &egui::Context, scene: &str, vr_screen: bool) {
    let events = egui::InteractionUsage::drain(ctx);
    if let Some(shared) = tracker(ctx) {
        let mut tracker = shared.lock().unwrap();
        let now = tracker.started.elapsed().as_secs_f64();
        let wall_time = unix_seconds();
        for event in events {
            tracker.observe(
                scene,
                if vr_screen { "VR screen" } else { "Desktop" },
                &event,
                now,
                wall_time,
            );
        }
        if tracker.saved.elapsed() >= Duration::from_secs(10) {
            tracker.save();
        }
    }
}

pub fn wrist(
    ctx: &egui::Context,
    scene: &str,
    page: &str,
    id: &str,
    label: &str,
    kind: &'static str,
    changed: bool,
    held: bool,
) {
    if let Some(shared) = tracker(ctx) {
        let mut tracker = shared.lock().unwrap();
        let now = tracker.started.elapsed().as_secs_f64();
        tracker.observe(
            scene,
            "Wrist wheel",
            &egui::UsageEvent {
                id: egui::Id::new(id),
                path: page.to_owned(),
                label: label.to_owned(),
                kind,
                changed,
                held,
            },
            now,
            unix_seconds(),
        );
        if tracker.saved.elapsed() >= Duration::from_secs(10) {
            tracker.save();
        }
    }
}

pub fn panel_input(ctx: &egui::Context, id: egui::Id, name: &str) {
    let changed = ctx.data_mut(|d| {
        let key = egui::Id::new("usage_last_panel");
        let previous = d.get_temp::<egui::Id>(key);
        d.insert_temp(key, id);
        previous != Some(id)
    });
    if changed && name != "Viewport" {
        egui::InteractionUsage::menu(ctx, id, name);
    }
}

pub fn show(ui: &mut egui::Ui) {
    ui.heading("Control usage");
    ui.label("One use per interaction burst. Rapid repeats within 2 seconds are grouped; a held drag stays one use. Switching controls starts a new use.");
    let Some(shared) = tracker(ui.ctx()) else {
        return;
    };
    // Clone the small aggregate table before painting; never hold the lock across UI callbacks.
    let (mut entries, path) = {
        let t = shared.lock().unwrap();
        (
            t.history.entries.values().cloned().collect::<Vec<_>>(),
            t.path.clone(),
        )
    };
    let today = unix_seconds() / 86400;
    let recent = |e: &Entry| {
        e.daily
            .range(today.saturating_sub(6)..)
            .map(|(_, count)| count)
            .sum::<u64>()
    };
    entries.sort_by(|a, b| {
        recent(b)
            .cmp(&recent(a))
            .then(b.uses.cmp(&a.uses))
            .then(a.label.cmp(&b.label))
    });
    ui.label("Ranked by the last 7 days. Frequent VR screen settings are candidates for wheel shortcuts.");
    if entries.is_empty() {
        ui.label("Use menus and sliders to start collecting a history.");
    }
    if ui.button("Copy usage report").clicked() {
        let mut report = "Control usage — grouped interactions\nLast 7 days\tAll time\tKind\tControl\tMenu\tSurface\tScene\n".to_owned();
        for entry in &entries {
            report.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                recent(entry),
                entry.uses,
                entry.kind,
                entry.label,
                entry.menu,
                entry.surface,
                entry.scene
            ));
        }
        ui.ctx().copy_text(report);
    }
    egui::ScrollArea::both().max_height(500.0).show(ui, |ui| {
        egui::Grid::new("control_usage_rankings")
            .striped(true)
            .show(ui, |ui| {
                for label in ["7 days", "Total", "Control", "Menu", "Surface", "Scene"] {
                    ui.strong(label);
                }
                ui.end_row();
                for entry in entries {
                    ui.label(recent(&entry).to_string());
                    ui.label(entry.uses.to_string());
                    ui.label(format!("{} · {}", entry.kind, entry.label));
                    ui.label(&entry.menu);
                    ui.label(&entry.surface);
                    ui.label(&entry.scene);
                    ui.end_row();
                }
            });
    });
    ui.small(format!("Local history: {}", path.display()));
}

#[cfg(test)]
mod tests {
    use super::*;
    fn event(label: &str, changed: bool, held: bool) -> egui::UsageEvent {
        egui::UsageEvent {
            id: egui::Id::new(label),
            path: "Lighting".into(),
            label: label.into(),
            kind: "Slider",
            changed,
            held,
        }
    }
    fn tracker_for_test(name: &str) -> Tracker {
        Tracker::load(std::env::temp_dir().join(format!(
            "biospheres_usage_{}_{name}.json",
            std::process::id()
        )))
    }
    #[test]
    fn rapid_changes_and_paused_held_drags_count_as_one_burst() {
        let mut tracker = tracker_for_test("bursts");
        tracker.history.entries.clear();
        for frame in 0..1000 {
            tracker.observe(
                "Simulation",
                "VR screen",
                &event("Brightness", frame < 10 || frame > 990, true),
                frame as f64 / 90.0,
                1000,
            );
        }
        assert_eq!(tracker.history.entries.values().next().unwrap().uses, 1);
        tracker.observe(
            "Simulation",
            "VR screen",
            &event("Brightness", true, false),
            12.0,
            1000,
        );
        assert_eq!(tracker.history.entries.values().next().unwrap().uses, 1);
        tracker.observe(
            "Simulation",
            "VR screen",
            &event("Brightness", true, false),
            15.0,
            1000,
        );
        assert_eq!(tracker.history.entries.values().next().unwrap().uses, 2);
        tracker.dirty = false;
    }
    #[test]
    fn switching_controls_counts_separately_and_idle_rendering_does_not_count() {
        let mut tracker = tracker_for_test("switching");
        tracker.history.entries.clear();
        for (label, time) in [("Brightness", 1.0), ("Inertia", 1.1), ("Brightness", 1.2)] {
            tracker.observe(
                "Simulation",
                "VR screen",
                &event(label, true, false),
                time,
                1000,
            );
        }
        tracker.observe(
            "Simulation",
            "Desktop",
            &event("Brightness", false, false),
            4.0,
            1000,
        );
        assert_eq!(tracker.history.entries.len(), 2);
        assert_eq!(
            tracker
                .history
                .entries
                .values()
                .map(|e| e.uses)
                .sum::<u64>(),
            3
        );
        tracker.dirty = false;
    }
    #[test]
    fn saved_counts_survive_restart_without_merging_the_next_session() {
        let mut tracker = tracker_for_test("persistence");
        tracker.history.entries.clear();
        let mut menu = event("Lighting", true, false);
        menu.kind = "Menu";
        for time in [1.0, 1.2, 1.5] {
            tracker.observe("Simulation", "Wrist wheel", &menu, time, 1000);
        }
        tracker.save();
        let path = tracker.path.clone();
        drop(tracker);
        let mut tracker = Tracker::load(path.clone());
        tracker.observe("Simulation", "Wrist wheel", &menu, 0.0, 1100);
        assert_eq!(tracker.history.entries.values().next().unwrap().uses, 2);
        tracker.save();
        assert_eq!(
            Tracker::load(path.clone())
                .history
                .entries
                .values()
                .next()
                .unwrap()
                .uses,
            2
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[allow(deprecated)]
    fn real_slider_drag_is_named_and_grouped_without_counting_idle_frames() {
        let ctx = egui::Context::default();
        egui::InteractionUsage::enable(&ctx);
        let mut tracker = tracker_for_test("widget");
        tracker.history.entries.clear();
        let mut value = 0.5;
        let mut slider = egui::Rect::NOTHING;
        let mut draw = |time: f64, events: Vec<egui::Event>| {
            ctx.begin_pass(egui::RawInput {
                time: Some(time),
                events,
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(500.0, 400.0),
                )),
                ..Default::default()
            });
            egui::CentralPanel::default().show(&ctx, |ui| {
                egui::InteractionUsage::scoped(&ctx, "Lighting", || {
                    egui::CollapsingHeader::new("Sun")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label("Brightness:");
                            slider = ui.add(egui::Slider::new(&mut value, 0.0..=1.0)).rect;
                        });
                });
            });
            let _ = ctx.end_pass();
            for event in egui::InteractionUsage::drain(&ctx) {
                tracker.observe("Simulation", "Desktop", &event, time, 1000);
            }
            slider
        };
        let rect = draw(0.0, vec![]);
        let point = egui::pos2(rect.left() + 30.0, rect.center().y);
        draw(
            0.1,
            vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ],
        );
        // Pause on the rail for 4 seconds while still holding, then resume.
        for frame in 0..400 {
            let point = point
                + egui::vec2(
                    if frame > 380 {
                        (frame - 380) as f32
                    } else {
                        0.0
                    },
                    0.0,
                );
            draw(
                0.11 + frame as f64 / 90.0,
                vec![egui::Event::PointerMoved(point)],
            );
        }
        draw(
            5.0,
            vec![egui::Event::PointerButton {
                pos: point,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: Default::default(),
            }],
        );
        for frame in 0..10 {
            draw(6.0 + frame as f64, vec![]);
        }
        let entries = tracker.history.entries.values().collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].label, "Brightness");
        assert_eq!(entries[0].menu, "Lighting / Sun");
        assert_eq!(entries[0].uses, 1);
        tracker.dirty = false;
    }
}
