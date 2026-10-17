//! Local redaction runs before reports are saved and again before upload.
//! This removes direct identifiers; arbitrary prose cannot be guaranteed anonymous.
use regex::{Regex, RegexBuilder};
use std::sync::OnceLock;

pub(super) struct Redactor {
    local_identifiers: Vec<Regex>,
}

impl Redactor {
    pub fn local() -> Self {
        let mut identifiers = Vec::new();
        if let Some(home) = dirs::home_dir() {
            let home = home.to_string_lossy();
            identifiers.extend([
                home.to_string(),
                home.replace('\\', "/"),
                home.replace('\\', "\\\\"),
            ]);
        }
        // Used only to remove these values locally, never added to report metadata.
        for key in ["USERNAME", "USER", "COMPUTERNAME", "HOSTNAME"] {
            if let Ok(value) = std::env::var(key) {
                if value.chars().count() >= 2 {
                    identifiers.push(value);
                }
            }
        }
        if let Some(host) = sysinfo::System::host_name() {
            identifiers.push(host);
        }
        Self::with_identifiers(identifiers)
    }

    fn with_identifiers(identifiers: Vec<String>) -> Self {
        Self {
            local_identifiers: identifiers
                .into_iter()
                .filter(|s| !s.is_empty())
                .map(|s| {
                    RegexBuilder::new(&regex::escape(&s))
                        .case_insensitive(true)
                        .build()
                        .unwrap()
                })
                .collect(),
        }
    }

    pub fn text(&self, text: &str) -> String {
        static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
        let mut result = text.to_owned();
        let patterns = PATTERNS.get_or_init(|| [
            // Credentials, URLs (including query tokens), email, local/UNC paths,
            // MAC addresses, UUIDs and telephone numbers.
                r#"(?i)\b(?:bearer\s+[^\s\"',;]+|(?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|secret|username|hostname|user[_-]?id|account[_-]?id|device[_-]?id|serial(?:[_-]?number)?)[\"']?\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s\"',;]+))"#,
            r#"(?i)\b(?:https?|ftp|file)://[^\s\"<>]+"#,
            r#"(?i)[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"#,
                r#"(?i)\b[a-z]:[\\/][^\r\n\"<>]*"#,
                r#"\\\\[^\r\n\"<>]+"#,
                r#"/(?:home|Users|mnt|media|Volumes|private|tmp|var|opt|usr)/[^\r\n\"<>]*"#,
            r#"(?i)\b[0-9a-f]{2}(?:[:-][0-9a-f]{2}){5}\b"#,
            r#"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b"#,
            r#"\+\d[\d ()-]{7,}\d|\b\d{3}[-. ]\d{3}[-. ]\d{4}\b"#,
        ].into_iter().map(|p| Regex::new(p).unwrap()).collect());
        for pattern in patterns {
            result = pattern.replace_all(&result, "<private>").into_owned();
        }
        static IPS: OnceLock<Regex> = OnceLock::new();
        let ips = IPS.get_or_init(|| {
            Regex::new(r"(?i)\b(?:\d{1,3}\.){3}\d{1,3}\b|(?:[0-9a-f]{0,4}:){2,}[0-9a-f:.]+")
                .unwrap()
        });
        result = ips
            .replace_all(&result, |caps: &regex::Captures<'_>| {
                if caps[0].parse::<std::net::IpAddr>().is_ok() {
                    "<private>".to_owned()
                } else {
                    caps[0].to_owned()
                }
            })
            .into_owned();
        for pattern in &self.local_identifiers {
            result = pattern.replace_all(&result, "<private>").into_owned();
        }
        result
    }
}

#[cfg(test)]
pub(super) fn redact_home(text: String, home: Option<&std::path::Path>) -> String {
    let Some(home) = home else {
        return text;
    };
    let home = home.to_string_lossy();
    let redactor = Redactor::with_identifiers(vec![
        home.to_string(),
        home.replace('\\', "/"),
        home.replace('\\', "\\\\"),
    ]);
    redactor.text(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_removed_in_plain_text_and_json_escaped_paths() {
        let redactor = Redactor::with_identifiers(vec!["Alice".into(), "ALICE-PC".into()]);
        let input = r#"Alice ALICE-PC alice@example.com C:\Users\Alice\save.json C:\\Users\\Alice\\save.json /home/alice/game \\server\share 192.168.1.20 2001:db8::1 aa:bb:cc:dd:ee:ff https://example.org/?token=secret api_key=secret +1 (555) 123-4567"#;
        let output = redactor.text(input);
        for private in [
            "Alice",
            "example.com",
            "save.json",
            "server",
            "192.168",
            "2001:db8",
            "aa:bb",
            "secret",
            "555",
        ] {
            assert!(!output.contains(private), "{private} survived: {output}");
        }
        assert_eq!(
            redactor.text("RTX 4090, driver 32.0.15.6094, 95% CPU, frame 16.67ms"),
            "RTX 4090, driver 32.0.15.6094, 95% CPU, frame 16.67ms"
        );
    }

    #[test]
    fn quoted_secrets_identifiers_and_paths_with_spaces_are_removed_in_full() {
        let redactor = Redactor::with_identifiers(vec![]);
        let input = "password=\"two private words\"\nusername=SomePlayer\nserial_number=ABC123XYZ\n\"C:\\Users\\Alice Smith\\Private File.txt\"\n/home/alice smith/private file\n";
        let output = redactor.text(input);
        for private in [
            "two private words",
            "SomePlayer",
            "ABC123XYZ",
            "Alice",
            "Smith",
            "Private File",
            "alice smith",
            "private file",
        ] {
            assert!(!output.contains(private), "{private} survived: {output}");
        }
    }
}
