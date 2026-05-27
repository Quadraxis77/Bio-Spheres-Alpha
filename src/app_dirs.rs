//! Platform-aware application directory helpers.
//!
//! All file I/O in Bio-Spheres goes through these helpers so the app can be
//! run from any location without needing a hand-crafted folder next to the exe.
//!
//! Layout:
//!   Config / settings  →  %APPDATA%\Bio-Spheres\          (Windows)
//!                          ~/.config/Bio-Spheres/          (Linux/macOS)
//!   Genome files       →  Documents\Bio-Spheres\genomes\  (Windows)
//!                          ~/Documents/Bio-Spheres/genomes/ (Linux/macOS)
//!   Log file           →  %APPDATA%\Bio-Spheres\bio_spheres.log

use std::path::PathBuf;

const APP_NAME: &str = "Bio-Spheres";

// ── Config directory ─────────────────────────────────────────────────────────

/// Returns `%APPDATA%\Bio-Spheres` (Windows) or `~/.config/Bio-Spheres`
/// (Linux/macOS), creating it if it doesn't exist.
pub fn config_dir() -> PathBuf {
    let base = dirs::config_dir()
        .unwrap_or_else(|| {
            log::warn!("Could not determine config directory; falling back to current directory");
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });
    let dir = base.join(APP_NAME);
    ensure_dir(&dir);
    dir
}

/// Returns the full path for a named config file inside the config directory.
///
/// Example: `config_file("cave_settings.ron")` →
///   `%APPDATA%\Bio-Spheres\cave_settings.ron`
pub fn config_file(filename: &str) -> PathBuf {
    config_dir().join(filename)
}

// ── Genomes directory ────────────────────────────────────────────────────────

/// Returns `Documents\Bio-Spheres\genomes` (Windows) or
/// `~/Documents/Bio-Spheres/genomes` (Linux/macOS), creating it if needed.
pub fn genomes_dir() -> PathBuf {
    let base = dirs::document_dir()
        .unwrap_or_else(|| {
            log::warn!("Could not determine Documents directory; falling back to config dir");
            config_dir()
        });
    let dir = base.join(APP_NAME).join("genomes");
    ensure_dir(&dir);
    dir
}

// ── Log file ─────────────────────────────────────────────────────────────────

/// Returns the path for the application log file inside the config directory.
pub fn log_file() -> PathBuf {
    config_file("bio_spheres.log")
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn ensure_dir(dir: &PathBuf) {
    if !dir.exists() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("Could not create directory {:?}: {}", dir, e);
        }
    }
}
