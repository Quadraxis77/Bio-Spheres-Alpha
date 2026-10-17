//! Bounded warning/error locations without potentially identifying message text.
use std::{
    collections::VecDeque,
    sync::{Mutex, OnceLock},
    time::Instant,
};

#[derive(Clone, serde::Serialize)]
struct Event {
    level: String,
    module: String,
    source_file: String,
    line: Option<u32>,
    occurrences: u64,
    last_elapsed_seconds: f64,
}

#[derive(Default)]
struct Events(VecDeque<Event>);

impl Events {
    fn record(&mut self, event: Event) {
        if let Some(index) = self.0.iter().position(|old| {
            old.level == event.level
                && old.module == event.module
                && old.source_file == event.source_file
                && old.line == event.line
        }) {
            let old = self.0.remove(index).unwrap();
            self.0.push_back(Event {
                occurrences: old.occurrences.saturating_add(1),
                ..event
            });
        } else {
            if self.0.len() == 128 {
                self.0.pop_front();
            }
            self.0.push_back(event);
        }
    }
}

static EVENTS: OnceLock<Mutex<Events>> = OnceLock::new();
static STARTED: OnceLock<Instant> = OnceLock::new();

pub(super) fn record(record: &log::Record<'_>) {
    if record.level() > log::Level::Warn {
        return;
    }
    let elapsed = STARTED.get_or_init(Instant::now).elapsed().as_secs_f64();
    let file = record
        .file()
        .and_then(|file| file.rsplit(['/', '\\']).next())
        .unwrap_or("unknown");
    if let Ok(mut events) = EVENTS.get_or_init(|| Mutex::new(Events::default())).lock() {
        events.record(Event {
            level: record.level().to_string(),
            module: record.module_path().unwrap_or("external").into(),
            source_file: file.into(),
            line: record.line(),
            occurrences: 1,
            last_elapsed_seconds: elapsed,
        });
    }
}

pub(super) fn snapshot() -> serde_json::Value {
    let events = EVENTS
        .get_or_init(|| Mutex::new(Events::default()))
        .lock()
        .map(|e| e.0.clone())
        .unwrap_or_default();
    serde_json::json!({"warning_and_error_sites": events, "clock": "seconds since the first logged warning/error", "message_text": "omitted for privacy; warning sites include performance logging"})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_sites_are_counted_without_evicting_distinct_failures() {
        let mut events = Events::default();
        let event = |line| Event {
            level: "ERROR".into(),
            module: "renderer".into(),
            source_file: "render.rs".into(),
            line: Some(line),
            occurrences: 1,
            last_elapsed_seconds: 0.0,
        };
        events.record(event(1));
        for _ in 0..300 {
            events.record(event(2));
        }
        assert_eq!(events.0.len(), 2);
        assert_eq!(events.0[1].occurrences, 300);
        for line in 3..200 {
            events.record(event(line));
        }
        assert_eq!(events.0.len(), 128);
    }
}
