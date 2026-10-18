use super::{create_report, new_report_id, privacy::Redactor};
use serde::{Deserialize, Serialize};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender},
    time::{Duration, Instant},
};

const MAX_QUEUE_FILE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_DIAGNOSTICS_BYTES: usize = 1024 * 1024;

pub(super) struct ReportRequest {
    pub id: String,
    pub message: String,
    pub context: serde_json::Value,
}

pub(super) enum DeliveryUpdate {
    Sent(String),
    Queued(String, String),
    Failed(String, String),
}

impl DeliveryUpdate {
    pub fn id(&self) -> &str {
        match self {
            Self::Sent(id) | Self::Queued(id, _) | Self::Failed(id, _) => id,
        }
    }
}

pub(super) struct DeliveryQueue {
    requests: Sender<ReportRequest>,
    pub updates: Receiver<DeliveryUpdate>,
}

#[derive(Serialize, Deserialize)]
struct Payload {
    schema_version: u32,
    #[serde(default)]
    privacy_revision: u32,
    report_id: String,
    version: String,
    message: String,
    diagnostics: String,
}

struct Pending {
    path: PathBuf,
    payload: Payload,
    due: Instant,
    backoff_seconds: u64,
}

impl DeliveryQueue {
    #[cfg(test)]
    pub fn test_channels() -> (Self, Receiver<ReportRequest>, Sender<DeliveryUpdate>) {
        let (requests, receiver) = mpsc::channel();
        let (sender, updates) = mpsc::channel();
        (Self { requests, updates }, receiver, sender)
    }

    pub fn start(config_dir: PathBuf) -> Option<Self> {
        let (requests, receiver) = mpsc::channel();
        let (sender, updates) = mpsc::channel();
        let endpoint = configured_endpoint();
        std::thread::Builder::new()
            .name("Bug report delivery".into())
            .spawn(move || run(&config_dir, endpoint, receiver, sender))
            .map_err(|error| log::warn!("Could not start bug report delivery: {error}"))
            .ok()?;
        Some(Self { requests, updates })
    }

    pub fn submit(&self, request: ReportRequest) -> Result<(), mpsc::SendError<ReportRequest>> {
        self.requests.send(request)
    }
}

fn configured_endpoint() -> Option<String> {
    let value = std::env::var("BIO_SPHERES_BUG_REPORT_URL")
        .ok()
        .or_else(|| option_env!("BIO_SPHERES_BUG_REPORT_URL").map(str::to_owned))?;
    let value = value.trim();
    let uri = value.parse::<ureq::http::Uri>().ok()?;
    if uri.scheme_str() != Some("https")
        || uri.host().is_none()
        || uri.authority()?.as_str().contains('@')
    {
        return None;
    }
    Some(value.into())
}

fn run(
    config_dir: &Path,
    endpoint: Option<String>,
    receiver: Receiver<ReportRequest>,
    updates: Sender<DeliveryUpdate>,
) {
    let mut pending = load_pending(&config_dir.join("bug_reports"));
    loop {
        match receiver.recv_timeout(Duration::from_secs(1)) {
            Ok(request) => {
                let id = request.id.clone();
                match save_pending(config_dir, request) {
                    Ok(report) => pending.push(report),
                    Err(error) => {
                        log::warn!("Could not save bug report: {error}");
                        let _ = updates.send(DeliveryUpdate::Failed(
                            id,
                            "Could not save the report. Please try again.".into(),
                        ));
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        let Some(index) = pending
            .iter()
            .position(|report| report.due <= Instant::now())
        else {
            continue;
        };
        let mut report = pending.remove(index);
        let result = endpoint
            .as_deref()
            .ok_or_else(|| {
                "Automatic sending isn't configured in this build. Your report is saved.".to_owned()
            })
            .and_then(|url| upload(url, &report.payload));
        match result {
            Ok(()) => {
                // The receiver deduplicates retries after a lost acknowledgement.
                if let Err(error) = std::fs::remove_file(&report.path) {
                    log::warn!("Could not remove delivered bug report marker: {error}");
                }
                let _ = updates.send(DeliveryUpdate::Sent(report.payload.report_id));
            }
            Err(error) => {
                let message = if endpoint.is_none() {
                    error.clone()
                } else {
                    "Report saved. We'll send it automatically when the connection is available."
                        .into()
                };
                let _ = updates.send(DeliveryUpdate::Queued(
                    report.payload.report_id.clone(),
                    message,
                ));
                log::warn!("Bug report queued for automatic retry: {error}");
                let delay = if endpoint.is_none() {
                    3600
                } else {
                    report.backoff_seconds
                };
                report.due = Instant::now() + Duration::from_secs(delay);
                report.backoff_seconds = (report.backoff_seconds * 2).min(300);
                pending.push(report);
            }
        }
    }
}

fn save_pending(config_dir: &Path, request: ReportRequest) -> std::io::Result<Pending> {
    log::logger().flush();
    let report_path = create_report(config_dir, &request.context)?;
    let mut diagnostics = std::fs::read_to_string(&report_path)?;
    if diagnostics.len() > MAX_DIAGNOSTICS_BYTES {
        let mut end = MAX_DIAGNOSTICS_BYTES;
        while !diagnostics.is_char_boundary(end) {
            end -= 1;
        }
        diagnostics.truncate(end);
        diagnostics.push_str("\n[Report truncated]\n");
    }
    let payload = Payload {
        schema_version: 1,
        privacy_revision: 1,
        report_id: request.id,
        version: env!("BIO_SPHERES_VERSION").into(),
        message: Redactor::local().text(&request.message),
        diagnostics,
    };
    let path = report_path.with_extension("pending.json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    serde_json::to_writer(&mut file, &payload)?;
    file.flush()?;
    file.sync_all()?;
    Ok(Pending {
        path,
        payload,
        due: Instant::now(),
        backoff_seconds: 5,
    })
}

fn load_pending(folder: &Path) -> Vec<Pending> {
    let Ok(files) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    files
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(".pending.json")
        })
        .filter_map(|entry| {
            let file = std::fs::File::open(entry.path()).ok()?;
            if file.metadata().ok()?.len() > MAX_QUEUE_FILE_BYTES {
                return None;
            }
            let mut payload: Payload = serde_json::from_reader(file.take(MAX_QUEUE_FILE_BYTES)).ok()?;
            if payload.schema_version != 1 || payload.message.is_empty() {
                return None;
            }
            // Upgrade old queued reports locally before they can be sent.
            let redactor = Redactor::local();
            payload.message = redactor.text(&payload.message);
            payload.diagnostics = redactor.text(&payload.diagnostics);
            if payload.privacy_revision == 0 {
                payload.report_id = new_report_id();
                payload.privacy_revision = 1;
                // Older versions saved arbitrary logs and player-authored names.
                // Those cannot be reliably scrubbed after the session has ended.
                payload.diagnostics = "BIO-SPHERES BUG REPORT\nLegacy diagnostics omitted by the privacy upgrade. The sanitized player message is retained.\n".into();
            }
            // Preserve the original queue entry if an upgrade write is interrupted.
            let temporary = entry.path().with_extension("redacted.tmp");
            let mut saved = std::fs::File::create(&temporary).ok()?;
            serde_json::to_writer(&mut saved, &payload).ok()?;
            saved.sync_all().ok()?;
            drop(saved);
            std::fs::rename(temporary, entry.path()).ok()?;
            Some(Pending {
                path: entry.path(),
                payload,
                due: Instant::now(),
                backoff_seconds: 5,
            })
        })
        .collect()
}

fn upload(endpoint: &str, payload: &Payload) -> Result<(), String> {
    let redactor = Redactor::local();
    let safe = Payload {
        schema_version: payload.schema_version,
        privacy_revision: 1,
        report_id: payload.report_id.clone(),
        version: payload.version.clone(),
        message: redactor.text(&payload.message),
        diagnostics: redactor.text(&payload.diagnostics),
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        .https_only(!cfg!(test))
        .build()
        .into();
    let mut response = agent
        .post(endpoint)
        .send_json(&safe)
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(16 * 1024)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    validate_receipt(&bytes, &payload.report_id)
}

fn validate_receipt(bytes: &[u8], report_id: &str) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| "Receiver did not acknowledge the report".to_owned())?;
    if value.get("ok").and_then(|v| v.as_bool()) == Some(true)
        && value.get("report_id").and_then(|v| v.as_str()) == Some(report_id)
    {
        Ok(())
    } else {
        Err(value
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Receiver did not acknowledge the report")
            .to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "sends a synthetic report to the configured live receiver"]
    fn live_receiver_accepts_a_synthetic_report() {
        let id = new_report_id();
        let payload = Payload {
            schema_version: 1,
            privacy_revision: 1,
            report_id: id.clone(),
            version: env!("BIO_SPHERES_VERSION").into(),
            message: "Automatic bug report delivery test. This verifies the message and diagnostics attachment; no player data is included.".into(),
            diagnostics: "BIO-SPHERES BUG REPORT\nSynthetic delivery test. No player logs or settings were collected.\n".into(),
        };
        upload(
            &configured_endpoint().expect("receiver URL must be configured"),
            &payload,
        )
        .expect("live receiver must acknowledge the test report");
        println!("Receiver accepted synthetic report: {id}");
    }

    #[test]
    fn only_a_matching_receiver_acknowledgement_counts_as_sent() {
        assert!(validate_receipt(br#"{"ok":true,"report_id":"report-1"}"#, "report-1").is_ok());
        assert!(validate_receipt(br#"{"ok":true,"report_id":"wrong"}"#, "report-1").is_err());
        assert!(validate_receipt(br#"{"ok":false,"error":"quota reached"}"#, "report-1").is_err());
        assert!(validate_receipt(b"<html>Sign in</html>", "report-1").is_err());
        assert!(validate_receipt(br#"{"ok":true}"#, "report-1").is_err());
    }

    #[test]
    fn queued_reports_keep_the_message_and_diagnostics_across_restart() {
        let root = PathBuf::from("target/bug-report-tests").join(format!(
            "queue-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("bio_spheres.log"), "[perf-menu] slow").unwrap();
        let report = save_pending(
            &root,
            ReportRequest {
                id: "report-1".into(),
                message: "Menu is lagging".into(),
                context: serde_json::json!({"fps": 3}),
            },
        )
        .unwrap();
        let recovered = load_pending(&root.join("bug_reports"));
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].payload.message, "Menu is lagging");
        assert!(recovered[0]
            .payload
            .diagnostics
            .contains("[perf-menu] slow"));
        assert_eq!(recovered[0].payload.report_id, "report-1");
        assert_eq!(recovered[0].path, report.path);
    }

    #[test]
    fn messages_are_redacted_before_persistence_and_legacy_diagnostics_are_never_replayed() {
        let root = PathBuf::from("target/bug-report-tests").join(new_report_id());
        std::fs::create_dir_all(&root).unwrap();
        let report = save_pending(
            &root,
            ReportRequest {
                id: new_report_id(),
                message: "Reach me at alice@example.com or https://example.org/secret".into(),
                context: serde_json::json!({}),
            },
        )
        .unwrap();
        let saved = std::fs::read_to_string(&report.path).unwrap();
        assert!(!saved.contains("alice@example.com"));
        assert!(!saved.contains("example.org"));
        let old_path = root.join("bug_reports/old.pending.json");
        std::fs::write(&old_path, serde_json::to_vec(&serde_json::json!({
            "schema_version": 1, "report_id": "bio-spheres-report-123-456", "version": "old",
            "message": "It froze; alice@example.com", "diagnostics": "BIO-SPHERES BUG REPORT\nPrivatePlayerName old log"
        })).unwrap()).unwrap();
        let recovered = load_pending(&root.join("bug_reports"));
        let legacy = recovered.iter().find(|r| r.path == old_path).unwrap();
        assert_ne!(legacy.payload.report_id, "bio-spheres-report-123-456");
        assert!(!legacy.payload.diagnostics.contains("PrivatePlayerName"));
        assert!(!legacy.payload.message.contains("alice@example.com"));
        assert_eq!(legacy.payload.privacy_revision, 1);
    }

    #[test]
    fn upload_sends_message_and_diagnostics_together_and_checks_receipt() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut byte = [0u8; 1];
                socket.read_exact(&mut byte).unwrap();
                bytes.push(byte[0]);
                if bytes.ends_with(b"\r\n\r\n") {
                    break bytes.len();
                }
                assert!(bytes.len() < 8192);
            };
            let headers = String::from_utf8_lossy(&bytes);
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap())
                })
                .unwrap();
            bytes.resize(header_end + length, 0);
            socket.read_exact(&mut bytes[header_end..]).unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&bytes[header_end..]).unwrap();
            let message = payload["message"].as_str().unwrap();
            assert!(message.starts_with("Menu is lagging"));
            assert!(!message.contains("alice@example.com"));
            let diagnostics = payload["diagnostics"].as_str().unwrap();
            assert!(diagnostics.starts_with("log contents"));
            assert!(!diagnostics.contains("Alice"));
            assert!(!diagnostics.contains("192.168.1.1"));
            let body = r#"{"ok":true,"report_id":"report-1"}"#;
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        let payload = Payload {
            schema_version: 1,
            privacy_revision: 1,
            report_id: "report-1".into(),
            version: "test".into(),
            message: "Menu is lagging alice@example.com".into(),
            diagnostics: "log contents C:\\Users\\Alice\\game 192.168.1.1".into(),
        };
        upload(&format!("http://{address}/report"), &payload).unwrap();
        server.join().unwrap();
    }
}
