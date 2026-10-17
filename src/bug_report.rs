//! In-game reports with automatic HTTPS delivery and a durable retry queue.

#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

mod delivery;
mod events;
mod privacy;
use delivery::{DeliveryQueue, DeliveryUpdate, ReportRequest};

pub(crate) fn record_diagnostic_event(record: &log::Record<'_>) {
    events::record(record);
}
pub(crate) fn diagnostic_events() -> serde_json::Value {
    events::snapshot()
}
use privacy::Redactor;

pub const REPORT_EMAIL: &str = "Quadraxis77@gmail.com";
const LOG_LIMIT: u64 = 256 * 1024;
const PREVIOUS_LOG_LIMIT: u64 = 128 * 1024;
const CONTEXT_LIMIT: usize = 384 * 1024;
const SETTINGS_LIMIT: u64 = 32 * 1024;
const SETTINGS: &[&str] = &[
    "cave_settings.ron",
    "fluid_settings.ron",
    "fluid_render_settings.ron",
    "light_settings.ron",
    "sun_settings.ron",
    "cell_visuals.ron",
];
const MESSAGE_LIMIT: usize = 4_000;

#[derive(Default)]
pub struct BugReporter {
    pub open: bool,
    queue: Option<DeliveryQueue>,
    description: String,
    current_id: Option<String>,
    sending: bool,
    submitted: bool,
    status: String,
    incident_context: Option<serde_json::Value>,
}

impl BugReporter {
    /// Run once at startup so previously queued reports resume without player input.
    pub fn resume_pending(&mut self) {
        if self.queue.is_none() {
            self.queue = DeliveryQueue::start(crate::app_dirs::config_dir());
        }
    }

    pub fn open_form(&mut self, context: serde_json::Value) {
        self.open = true;
        if self.submitted && !self.sending {
            self.description.clear();
            self.status.clear();
            self.current_id = None;
            self.submitted = false;
            self.incident_context = None;
        }
        if self.incident_context.is_none() {
            self.incident_context = Some(context);
        }
    }

    pub fn submit(&mut self, context: serde_json::Value) {
        if self.sending || self.submitted || self.description.trim().is_empty() {
            return;
        }
        self.resume_pending();
        let Some(queue) = &self.queue else {
            self.status = "Could not start the report sender. Please try again.".into();
            return;
        };
        let id = new_report_id();
        let request = ReportRequest {
            id: id.clone(),
            message: self.description.trim().into(),
            context: serde_json::json!({"diagnostics_schema": 2, "when_opened": self.incident_context, "when_sent": context}),
        };
        if queue.submit(request).is_err() {
            self.queue = None;
            self.status = "Could not start the report sender. Please try again.".into();
            return;
        }
        self.current_id = Some(id);
        self.sending = true;
        self.submitted = true;
        self.status = "Collecting and sending your report…".into();
    }

    pub fn poll(&mut self) {
        let Some(queue) = &self.queue else {
            return;
        };
        loop {
            let update = match queue.updates.try_recv() {
                Ok(update) => update,
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.queue = None;
                    self.sending = false;
                    self.submitted = false;
                    self.status =
                        "The report sender stopped. Your message is still here; please try again."
                            .into();
                    return;
                }
            };
            if self.current_id.as_deref() != Some(update.id()) {
                continue;
            }
            match update {
                DeliveryUpdate::Sent(_) => {
                    self.sending = false;
                    self.status = "Report sent. Thank you!".into();
                }
                DeliveryUpdate::Queued(_, reason) => {
                    self.sending = false;
                    self.status = reason;
                }
                DeliveryUpdate::Failed(_, reason) => {
                    self.sending = false;
                    self.submitted = false;
                    self.status = reason;
                }
            }
        }
    }

    /// Returns true only for the Send button, so the app captures diagnostics then.
    pub fn show(&mut self, ctx: &egui::Context) -> bool {
        let mut send = false;
        egui::Window::new("Bug report")
            .open(&mut self.open).collapsible(false).resizable(false).default_width(420.0)
            .show(ctx, |ui| {
                ui.label("What went wrong?");
                ui.add_enabled(!self.submitted, egui::TextEdit::multiline(&mut self.description)
                    .hint_text("Describe the bug and what you were doing when it happened.")
                    .desired_rows(6).desired_width(f32::INFINITY).char_limit(MESSAGE_LIMIT));
                ui.label("Sends your message, filtered diagnostics, hardware details, settings, and recent performance.");
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Privacy notice").strong());
                ui.label("Known personal identifiers are removed locally before sending. Hardware details may still make you identifiable. Please leave personal information out of your message.");
                ui.add_space(6.0);
                send = ui.add_enabled(!self.submitted && !self.description.trim().is_empty(), egui::Button::new("Send report")).clicked();
                if self.sending { ui.spinner(); }
                if !self.status.is_empty() { ui.label(&self.status); }
            });
        send
    }
}

fn read_tail(path: &Path, limit: u64) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let offset = length.saturating_sub(limit);
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    file.take(limit).read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes);
    Ok(if offset > 0 {
        format!("[Earlier content omitted]\n{text}")
    } else {
        text.into_owned()
    })
}

pub(super) fn new_report_id() -> String {
    // Per-report random values: no machine identifier, PID or wall-clock timestamp.
    format!(
        "bio-spheres-report-{}-{}",
        fastrand::u64(..),
        fastrand::u64(..)
    )
}

fn bounded(mut text: String, limit: usize) -> String {
    if text.len() > limit {
        let marker = "\n[Section truncated]\n";
        let mut end = limit.saturating_sub(marker.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(marker);
    }
    text
}

fn diagnostic_log(path: &Path, limit: u64) -> String {
    let log = match read_tail(path, limit) {
        Ok(log) => log,
        Err(error) => return format!("[Unavailable: {:?}]", error.kind()),
    };
    // Arbitrary log lines may contain player-authored names or loaded file paths.
    // Keep only the engine's structured, numeric performance records.
    let lines = diagnostic_log_text(&log);
    format!("[Only allowlisted performance records; other log lines omitted]\n{lines}")
}

/// Keep one bounded previous-session log before the application's logger truncates it.
pub(crate) fn preserve_previous_log(config_dir: &Path) {
    if let Ok(log) = read_tail(&config_dir.join("bio_spheres.log"), PREVIOUS_LOG_LIMIT) {
        let safe = Redactor::local().text(&diagnostic_log_text(&log));
        let _ = std::fs::write(config_dir.join("bio_spheres.previous.log"), safe);
    }
}

fn diagnostic_log_text(log: &str) -> String {
    log.lines()
        .filter_map(|line| {
            ["[perf-menu]", "[perf-cpu]", "[perf]", "[diagnostics-build]"]
                .iter()
                .find_map(|tag| line.find(tag).map(|i| &line[i..]))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn settings_snapshot(path: &Path) -> String {
    let mut bytes = Vec::new();
    let result =
        File::open(path).and_then(|file| file.take(SETTINGS_LIMIT + 1).read_to_end(&mut bytes));
    if result.is_err() {
        return "[Not available]".into();
    }
    if bytes.len() > SETTINGS_LIMIT as usize {
        return "[File exceeds collection limit; live settings are captured above]".into();
    }
    let Ok(mut value) = ron::de::from_bytes::<ron::Value>(&bytes) else {
        return "[Cannot parse numeric settings; raw text omitted]".into();
    };
    fn remove_text(value: &mut ron::Value) {
        match value {
            ron::Value::String(_) | ron::Value::Char(_) => {
                *value = ron::Value::String("<text omitted>".into())
            }
            ron::Value::Map(map) => {
                for (_, value) in map.iter_mut() {
                    remove_text(value);
                }
            }
            ron::Value::Seq(values) => {
                for value in values {
                    remove_text(value);
                }
            }
            ron::Value::Option(Some(value)) => remove_text(value),
            _ => {}
        }
    }
    remove_text(&mut value);
    ron::ser::to_string_pretty(&value, ron::ser::PrettyConfig::default())
        .unwrap_or_else(|_| "[Unavailable]".into())
}

pub(crate) fn build_identity() -> serde_json::Value {
    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "source_fingerprint": env!("BIO_SPHERES_SOURCE_ID"),
        "profile": env!("BIO_SPHERES_BUILD_PROFILE"),
        "built_at_unix_seconds": env!("BIO_SPHERES_BUILT_AT"),
        "vr_feature": cfg!(feature = "vr"),
    })
}

pub(crate) fn record_panic_location(location: Option<&std::panic::Location<'_>>) {
    // Panic payloads and backtraces may embed player data and local paths.
    let data = serde_json::json!({"build": build_identity(), "location": location.map(|site| {
        serde_json::json!({"file": Path::new(site.file()).file_name().map(|s| s.to_string_lossy()), "line": site.line(), "column": site.column()})
    }), "panic_message": "omitted for privacy"});
    if let Ok(bytes) = serde_json::to_vec(&data) {
        let _ = std::fs::write(crate::app_dirs::config_file("last_crash.json"), bytes);
    }
}

pub fn create_report(config_dir: &Path, context: &serde_json::Value) -> std::io::Result<PathBuf> {
    let redactor = Redactor::local();
    let context = redactor.text(&serde_json::to_string_pretty(
        &serde_json::json!({"build": build_identity(), "snapshot": context}),
    )?);
    let mut system = sysinfo::System::new();
    system.refresh_cpu_list(sysinfo::CpuRefreshKind::new());
    system.refresh_memory();
    let mut text = format!(
        "BIO-SPHERES BUG REPORT\nVersion: {}\nBuild: {}\nPlatform: {} / {}\nOS: {}\nCPU: {}\nLogical CPUs: {}\nRAM: {} MiB total / {} MiB available\n\nCURRENT GAME STATE\n{}\n",
        env!("CARGO_PKG_VERSION"), if cfg!(debug_assertions) { "debug" } else { "release" },
        std::env::consts::OS, std::env::consts::ARCH,
        sysinfo::System::long_os_version().unwrap_or_else(|| "unknown".into()),
        system.cpus().first().map_or("unknown", |cpu| cpu.brand()), system.cpus().len(),
        system.total_memory() / (1024 * 1024), system.available_memory() / (1024 * 1024),
        bounded(context, CONTEXT_LIMIT),
    );
    text.push_str("\nLAST RECORDED PANIC (may be an earlier build; message omitted for privacy)\n");
    text.push_str(
        &read_tail(&config_dir.join("last_crash.json"), 16 * 1024)
            .unwrap_or_else(|_| "[Not recorded]".into()),
    );
    text.push_str("\n\nFILTERED PERFORMANCE LOG\n");
    text.push_str(&diagnostic_log(
        &config_dir.join("bio_spheres.log"),
        LOG_LIMIT,
    ));
    text.push_str("\n\nPREVIOUS SESSION PERFORMANCE (may be an earlier build)\n");
    text.push_str(&diagnostic_log(
        &config_dir.join("bio_spheres.previous.log"),
        PREVIOUS_LOG_LIMIT,
    ));
    for name in SETTINGS {
        text.push_str(&format!("\n\nSETTINGS: {name}\n"));
        text.push_str(&settings_snapshot(&config_dir.join(name)));
    }
    text = redactor.text(&text);
    text = bounded(text, 1024 * 1024);
    let reports = config_dir.join("bug_reports");
    std::fs::create_dir_all(&reports)?;
    let path = reports.join(format!("{}.txt", new_report_id()));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    output.write_all(text.as_bytes())?;
    output.flush()?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_the_dialog_does_not_submit_and_an_empty_message_cannot_send() {
        let mut reporter = BugReporter::default();
        reporter.open_form(serde_json::json!({}));
        assert!(reporter.open);
        reporter.description = " \n ".into();
        reporter.submit(serde_json::json!({}));
        assert!(reporter.queue.is_none());
        assert!(!reporter.submitted);
    }

    #[test]
    fn send_includes_the_message_once_and_only_a_receipt_shows_success() {
        let (queue, requests, updates) = DeliveryQueue::test_channels();
        let mut reporter = BugReporter {
            queue: Some(queue),
            description: "  The menu freezes.  ".into(),
            ..Default::default()
        };
        reporter.submit(serde_json::json!({"phase": "MainMenu"}));
        let request = requests.try_recv().unwrap();
        assert_eq!(request.message, "The menu freezes.");
        assert_eq!(request.context["when_sent"]["phase"], "MainMenu");
        reporter.submit(serde_json::json!({}));
        assert!(
            requests.try_recv().is_err(),
            "double clicking must not submit twice"
        );
        updates
            .send(DeliveryUpdate::Sent("unrelated".into()))
            .unwrap();
        reporter.poll();
        assert!(reporter.sending);
        updates.send(DeliveryUpdate::Sent(request.id)).unwrap();
        reporter.poll();
        assert!(!reporter.sending);
        assert_eq!(reporter.status, "Report sent. Thank you!");
    }

    #[test]
    fn opening_preserves_the_incident_while_the_player_types_and_reopens_the_form() {
        let (queue, requests, _) = DeliveryQueue::test_channels();
        let mut reporter = BugReporter {
            queue: Some(queue),
            ..Default::default()
        };
        reporter.open_form(serde_json::json!({"fps": 3}));
        reporter.description = "The menu was slow".into();
        reporter.open = false;
        reporter.open_form(serde_json::json!({"fps": 60}));
        reporter.submit(serde_json::json!({"fps": 120}));
        let request = requests.try_recv().unwrap();
        assert_eq!(request.context["when_opened"]["fps"], 3);
        assert_eq!(request.context["when_sent"]["fps"], 120);
    }

    #[test]
    fn raw_log_text_is_excluded_and_previous_session_performance_survives_restart() {
        let root = PathBuf::from("target/bug-report-tests").join(new_report_id());
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("bio_spheres.log"),
            "Loaded genome PrivatePlayerName\n[perf-menu] fps=3\nPassword=PRIVATE_SECRET\n[perf-cpu] Update=120ms\n").unwrap();
        preserve_previous_log(&root);
        std::fs::write(root.join("bio_spheres.log"), "[perf-menu] fps=60\n").unwrap();
        let report = create_report(
            &root,
            &serde_json::json!({"example_path": "C:\\Users\\PrivatePlayerName\\game"}),
        )
        .unwrap();
        let text = std::fs::read_to_string(report).unwrap();
        assert!(text.contains("fps=3"));
        assert!(text.contains("fps=60"));
        assert!(!text.contains("PrivatePlayerName"));
        assert!(!text.contains("PRIVATE_SECRET"));
        assert!(text.contains("source_fingerprint"));
    }

    #[test]
    fn report_collects_only_bounded_allowed_files_and_redacts_home() {
        let root = PathBuf::from("target/bug-report-tests").join(format!(
            "{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("bio_spheres.log"),
            "x".repeat(LOG_LIMIT as usize + 200) + "\n[perf-menu] slow frame\n",
        )
        .unwrap();
        std::fs::write(
            root.join("sun_settings.ron"),
            "(sun_intensity: 2.8, note: \"PRIVATE CUSTOM NAME\")",
        )
        .unwrap();
        std::fs::write(root.join("credentials.txt"), "MUST_NOT_COLLECT").unwrap();
        let report = create_report(
            &root,
            &serde_json::json!({"phase": "MainMenu", "fps": 12.5}),
        )
        .unwrap();
        let text = std::fs::read_to_string(report).unwrap();
        assert!(text.contains("other log lines omitted"));
        assert!(text.contains("[perf-menu] slow frame"));
        assert!(text.contains("sun_intensity"));
        assert!(text.contains("2.8"));
        assert!(!text.contains("PRIVATE CUSTOM NAME"));
        assert!(text.contains("MainMenu"));
        assert!(!text.contains("MUST_NOT_COLLECT"));
        assert!(text.len() < LOG_LIMIT as usize + 10_000);
        assert_eq!(
            privacy::redact_home(
                "C:/Users/Alice/game".into(),
                Some(Path::new("C:/Users/Alice"))
            ),
            "<private>"
        );
    }
}
