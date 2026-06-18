//! Self-update and config migration logic.
//!
//! ## Self-replace (exe update)
//!
//! When the user downloads a new `bio-spheres.exe` and runs it from the same
//! directory as the old one, Windows prevents overwriting the running process.
//! The fix is the rename-then-replace trick:
//!
//!   1. Rename the OLD exe to `bio-spheres.exe.old`  (works on a running file)
//!   2. The NEW exe is already running - it IS the new version
//!   3. Copy the new exe to the canonical path (where the old one was)
//!   4. Delete the `.old` file
//!   5. Relaunch from the canonical path and exit
//!
//! "Canonical path" = the path where the exe was on the previous run, stored
//! in `%APPDATA%\Bio-Spheres\install_path.txt`.  On first run this file
//! doesn't exist, so we write the current exe path as the canonical path and
//! skip the update dance.
//!
//! ## Config migration
//!
//! On every startup, each RON config file in AppData is compared against the
//! compiled-in defaults.  Any key present in the default but absent from the
//! user's file is added with the default value.  Existing user values are
//! never touched.  The file is only rewritten if at least one key was added.

use std::path::{Path, PathBuf};

// -- Public entry points -------------------------------------------------------

/// Call this at the very start of `run()`, before logging is set up.
///
/// If a self-replace is needed it will relaunch the exe and exit, so this
/// function never returns in that case.  Otherwise it returns normally and
/// startup continues.
pub fn run_self_replace() {
    // Only meaningful on Windows - on Linux/macOS the OS allows overwriting
    // a running exe directly, so no dance is needed.
    #[cfg(target_os = "windows")]
    {
        if let Err(e) = try_self_replace() {
            // Non-fatal: log and continue.  The user might be running from a
            // read-only location or a network share.
            eprintln!("Bio-Spheres: self-replace skipped: {}", e);
        }
    }

    // Clean up any leftover .old file from a previous update.
    cleanup_old_exe();
}

/// Call this after logging is set up, before the event loop starts.
///
/// Merges new default keys into existing user config files without touching
/// values the user has already set.
/// Writes bundled example genomes to the user's genomes directory if they are
/// not already present.  Only runs if the file is missing — never overwrites
/// user edits.
pub fn seed_default_genomes() {
    const BUNDLED: &[(&str, &[u8])] = &[
        (
            "Triangular Prism.genome",
            include_bytes!("../genomes/Triangular Prism.genome"),
        ),
        (
            "Octo-Tube.genome",
            include_bytes!("../genomes/Octo-Tube.genome"),
        ),
        (
            "Octopus.genome",
            include_bytes!("../genomes/Octopus.genome"),
        ),
    ];

    let dir = crate::app_dirs::genomes_dir();
    for (filename, bytes) in BUNDLED {
        let path = dir.join(filename);
        if !path.exists() {
            if let Err(e) = std::fs::write(&path, bytes) {
                log::warn!("Failed to seed genome {}: {}", filename, e);
            }
        }
    }
}

pub fn migrate_config_files() {
    const EMBEDDED_DEFAULTS: &[(&str, &str)] = &[
        ("cave_settings.ron", include_str!("../cave_settings.ron")),
        ("cell_visuals.ron", include_str!("../cell_visuals.ron")),
        ("fluid_settings.ron", include_str!("../fluid_settings.ron")),
        ("light_settings.ron", include_str!("../light_settings.ron")),
        ("sun_settings.ron", include_str!("../sun_settings.ron")),
        (
            "fluid_render_settings.ron",
            include_str!("../fluid_render_settings.ron"),
        ),
    ];

    for (filename, default_content) in EMBEDDED_DEFAULTS {
        let path = crate::app_dirs::config_file(filename);

        if !path.exists() {
            // First run - just write the default wholesale.
            if let Err(e) = std::fs::write(&path, default_content) {
                log::warn!("Failed to write default {}: {}", filename, e);
            }
            continue;
        }

        let legacy_fluid_schema = if *filename == "fluid_settings.ron" {
            std::fs::read_to_string(&path)
                .map(|text| !text.contains("nutrient_epoch_duration"))
                .unwrap_or(false)
        } else {
            false
        };

        // File exists - merge missing keys.
        match merge_ron_keys(&path, default_content) {
            Ok(true) => log::info!("Config migration: added new keys to {}", filename),
            Ok(false) => {} // nothing to do
            Err(e) => log::warn!("Config migration failed for {}: {}", filename, e),
        }

        if legacy_fluid_schema {
            match migrate_legacy_fluid_nutrient_density(&path) {
                Ok(true) => log::info!("Config migration: updated legacy nutrient density default"),
                Ok(false) => {}
                Err(e) => log::warn!("Config migration failed for fluid nutrient density: {}", e),
            }
        }

        if *filename == "light_settings.ron" {
            match migrate_legacy_photocyte_production(&path) {
                Ok(true) => {
                    log::info!("Config migration: increased legacy photocyte production rate")
                }
                Ok(false) => {}
                Err(e) => log::warn!(
                    "Config migration failed for photocyte production rate: {}",
                    e
                ),
            }
        }
    }
}

// -- Self-replace --------------------------------------------------------------

#[cfg(target_os = "windows")]
fn try_self_replace() -> Result<(), String> {
    let current_exe =
        std::env::current_exe().map_err(|e| format!("cannot get current exe: {}", e))?;

    let canonical_path = read_canonical_path();

    match canonical_path {
        None => {
            // First run - record this path as canonical and continue normally.
            write_canonical_path(&current_exe);
            Ok(())
        }
        Some(canonical) if canonical == current_exe => {
            // Running from the canonical path - nothing to do.
            Ok(())
        }
        Some(canonical) => {
            // Running from a DIFFERENT path than last time.
            // This means the user dropped a new exe somewhere and ran it.
            // Perform the rename-then-replace dance.
            do_self_replace(&current_exe, &canonical)
        }
    }
}

/// Performs the actual rename-then-replace and relaunches.
/// Never returns on success.
#[cfg(target_os = "windows")]
fn do_self_replace(new_exe: &Path, canonical: &Path) -> Result<(), String> {
    let backup = canonical.with_extension("exe.old");

    // Step 1: rename the old exe out of the way (works even if it's running).
    if canonical.exists() {
        // Remove stale backup first.
        let _ = std::fs::remove_file(&backup);

        std::fs::rename(canonical, &backup)
            .map_err(|e| format!("cannot rename {:?} to {:?}: {}", canonical, backup, e))?;
    }

    // Step 2: copy the new exe to the canonical path.
    std::fs::copy(new_exe, canonical).map_err(|e| {
        // Rollback: try to restore the old exe.
        let _ = std::fs::rename(&backup, canonical);
        format!("cannot copy new exe to {:?}: {}", canonical, e)
    })?;

    // Step 3: update the stored canonical path (it hasn't changed, but
    // write it anyway to refresh the file's mtime).
    write_canonical_path(canonical);

    // Step 4: relaunch from the canonical path.
    std::process::Command::new(canonical)
        .spawn()
        .map_err(|e| format!("cannot relaunch {:?}: {}", canonical, e))?;

    // Step 5: exit this (the new-but-misplaced) process.
    std::process::exit(0);
}

/// Read the stored canonical exe path from AppData.
fn read_canonical_path() -> Option<PathBuf> {
    let path = crate::app_dirs::config_file("install_path.txt");
    let text = std::fs::read_to_string(&path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

/// Write the canonical exe path to AppData.
fn write_canonical_path(exe_path: &Path) {
    let path = crate::app_dirs::config_file("install_path.txt");
    if let Err(e) = std::fs::write(&path, exe_path.to_string_lossy().as_bytes()) {
        log::warn!("Could not write install_path.txt: {}", e);
    }
}

/// Delete `bio-spheres.exe.old` next to the canonical exe if it exists.
/// Called on every startup to clean up after a successful update.
fn cleanup_old_exe() {
    if let Some(canonical) = read_canonical_path() {
        let backup = canonical.with_extension("exe.old");
        if backup.exists() {
            match std::fs::remove_file(&backup) {
                Ok(()) => log::info!("Cleaned up old exe: {:?}", backup),
                Err(e) => log::warn!("Could not remove old exe {:?}: {}", backup, e),
            }
        }
    }
}

// -- RON config migration ------------------------------------------------------

/// Merge keys from `default_content` into the RON file at `path`.
///
/// Returns `Ok(true)` if the file was updated, `Ok(false)` if it was already
/// up to date, or `Err` if parsing or writing failed.
fn merge_ron_keys(path: &Path, default_content: &str) -> Result<bool, String> {
    let existing_text = std::fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;

    let existing_val: ron::Value = ron::from_str(&existing_text)
        .map_err(|e| format!("parse error in existing file: {}", e))?;

    let default_val: ron::Value =
        ron::from_str(default_content).map_err(|e| format!("parse error in default: {}", e))?;

    // Both must be maps (RON structs parse as Map).
    let mut existing_map = match existing_val {
        ron::Value::Map(m) => m,
        _ => return Err("existing file is not a RON struct/map".into()),
    };
    let default_map = match default_val {
        ron::Value::Map(m) => m,
        _ => return Err("default is not a RON struct/map".into()),
    };

    // Find keys in default that are missing from the existing file.
    let mut added = 0usize;
    for (key, value) in default_map.iter() {
        let already_present = existing_map.iter().any(|(k, _)| k == key);
        if !already_present {
            existing_map.insert(key.clone(), value.clone());
            added += 1;
            log::info!(
                "Config migration: added key {:?} to {}",
                key,
                path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }

    if added == 0 {
        return Ok(false);
    }

    // Serialise the merged map back to RON.
    // We wrap it in the same `(...)` struct syntax the app uses.
    let merged = ron::Value::Map(existing_map);
    let new_text = ron::ser::to_string_pretty(&merged, ron::ser::PrettyConfig::default())
        .map_err(|e| format!("serialise error: {}", e))?;

    std::fs::write(path, new_text).map_err(|e| format!("write error: {}", e))?;

    Ok(true)
}

fn migrate_legacy_fluid_nutrient_density(path: &Path) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;

    if !text.contains("nutrient_density: 0.2") {
        return Ok(false);
    }

    let updated = text.replacen("nutrient_density: 0.2", "nutrient_density: 0.35", 1);
    std::fs::write(path, updated).map_err(|e| format!("write error: {}", e))?;
    Ok(true)
}

fn migrate_legacy_photocyte_production(path: &Path) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;

    let legacy_value = if text.contains("photocyte_mass_per_second: 0.012") {
        "photocyte_mass_per_second: 0.012"
    } else if text.contains("photocyte_mass_per_second: 2.0") {
        "photocyte_mass_per_second: 2.0"
    } else {
        return Ok(false);
    };

    let updated = text.replacen(
        legacy_value,
        "photocyte_mass_per_second: 0.2",
        1,
    );
    std::fs::write(path, updated).map_err(|e| format!("write error: {}", e))?;
    Ok(true)
}
