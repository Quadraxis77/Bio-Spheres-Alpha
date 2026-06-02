//! Platform-aware application directory helpers.
//!
//! All file I/O in Bio-Spheres goes through these helpers so the app can be
//! run from any location without needing a hand-crafted folder next to the exe.
//!
//! Layout (all under Documents\Bio-Spheres\ or ~/Documents/Bio-Spheres/):
//!   Genome files   ->  Bio-Spheres\genomes\
//!   Sphere files   ->  Bio-Spheres\spheres\
//!   Screenshots    ->  Bio-Spheres\screenshots\
//!   Videos         ->  Bio-Spheres\videos\
//!
//! Config / settings  ->  %APPDATA%\Bio-Spheres\          (Windows)
//!                        ~/.config/Bio-Spheres/          (Linux/macOS)
//!   Log file           ->  %APPDATA%\Bio-Spheres\bio_spheres.log

use std::path::PathBuf;

const APP_NAME: &str = "Bio-Spheres";

// -- Config directory ---------------------------------------------------------

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
pub fn config_file(filename: &str) -> PathBuf {
    config_dir().join(filename)
}

// -- Bio-Spheres documents root -----------------------------------------------

/// Returns `Documents\Bio-Spheres`, creating it if needed.
pub fn biospheres_dir() -> PathBuf {
    let base = dirs::document_dir()
        .unwrap_or_else(|| {
            log::warn!("Could not determine Documents directory; falling back to config dir");
            config_dir()
        });
    let dir = base.join(APP_NAME);
    ensure_dir(&dir);
    dir
}

// -- Sub-directories ----------------------------------------------------------

/// Returns `Documents\Bio-Spheres\genomes`, creating it if needed.
pub fn genomes_dir() -> PathBuf {
    let dir = biospheres_dir().join("genomes");
    ensure_dir(&dir);
    dir
}

/// Returns `Documents\Bio-Spheres\spheres`, creating it if needed.
pub fn spheres_dir() -> PathBuf {
    let dir = biospheres_dir().join("spheres");
    ensure_dir(&dir);
    dir
}

/// Returns `Documents\Bio-Spheres\screenshots`, creating it if needed.
pub fn screenshots_dir() -> PathBuf {
    let dir = biospheres_dir().join("screenshots");
    ensure_dir(&dir);
    dir
}

/// Returns `Documents\Bio-Spheres\videos`, creating it if needed.
pub fn videos_dir() -> PathBuf {
    let dir = biospheres_dir().join("videos");
    ensure_dir(&dir);
    dir
}

// -- Log file -----------------------------------------------------------------

/// Returns the path for the application log file inside the config directory.
pub fn log_file() -> PathBuf {
    config_file("bio_spheres.log")
}

// -- Utilities ----------------------------------------------------------------

/// Strip characters that are invalid in filenames.
/// Used when constructing genome save paths from user-entered names.
pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

// -- Internal helpers ---------------------------------------------------------

fn ensure_dir(dir: &PathBuf) {
    if !dir.exists() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("Could not create directory {:?}: {}", dir, e);
        }
    }
}
