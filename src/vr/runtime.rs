//! Select a live SteamVR connection before the Windows default (often VDXR).
//! No registry edits. Explicit XR_RUNTIME_JSON overrides always win.
use super::VrResult;

pub(super) fn connect<T>(mut attempt: impl FnMut() -> VrResult<T>) -> VrResult<T> {
    #[cfg(target_os = "windows")]
    {
        // OpenXR's loader chooses one runtime at a time. Serialize discovery,
        // instance creation, and the HMD check. Failed attempts drop their
        // instance before trying the next runtime.
        static LOADER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = LOADER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if std::env::var_os("XR_RUNTIME_JSON").is_some() {
            return attempt();
        }
        let steam = running_steamvr_manifest();
        return try_candidates(steam.as_deref(), |manifest| {
            let _override = RuntimeOverride::new(manifest);
            attempt()
        });
    }
    #[cfg(not(target_os = "windows"))]
    attempt()
}

#[cfg(any(target_os = "windows", test))]
fn try_candidates<T>(
    steam: Option<&std::path::Path>,
    mut attempt: impl FnMut(Option<&std::path::Path>) -> VrResult<T>,
) -> VrResult<T> {
    if let Some(path) = steam {
        match attempt(Some(path)) {
            Ok(value) => {
                log::info!("VR connected through SteamVR: {}", path.display());
                return Ok(value);
            }
            Err(error) => {
                log::debug!(
                    "SteamVR headset unavailable; checking default OpenXR runtime: {error}"
                );
            }
        }
    }
    attempt(None)
}

#[cfg(target_os = "windows")]
struct RuntimeOverride(Option<std::ffi::OsString>);

#[cfg(target_os = "windows")]
impl RuntimeOverride {
    fn new(manifest: Option<&std::path::Path>) -> Self {
        let original = std::env::var_os("XR_RUNTIME_JSON");
        if let Some(path) = manifest {
            // Windows environment mutation is thread-safe; this code is never
            // compiled on POSIX. The loader lock protects our XR selection.
            std::env::set_var("XR_RUNTIME_JSON", path);
        }
        Self(original)
    }
}

#[cfg(target_os = "windows")]
impl Drop for RuntimeOverride {
    fn drop(&mut self) {
        match &self.0 {
            Some(value) => std::env::set_var("XR_RUNTIME_JSON", value),
            None => std::env::remove_var("XR_RUNTIME_JSON"),
        }
    }
}

#[cfg(target_os = "windows")]
fn running_steamvr_manifest() -> Option<std::path::PathBuf> {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::new().with_exe(UpdateKind::OnlyIfNotSet),
    );
    let servers: Vec<_> = system
        .processes()
        .values()
        .filter(|process| process.name().eq_ignore_ascii_case("vrserver.exe"))
        .collect();
    if servers.is_empty() {
        return None; // Never start SteamVR merely because it is installed.
    }
    // vrserver lives in <SteamVR>/bin/win64, including custom Steam libraries.
    for server in servers {
        if let Some(exe) = server.exe() {
            if let Some(root) = exe
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
            {
                let manifest = root.join("steamxr_win64.json");
                if manifest.is_file() {
                    return Some(manifest);
                }
            }
        }
    }
    // Some installations deny executable-path access; OpenVR records the root.
    let paths = std::path::PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
        .join("openvr")
        .join("openvrpaths.vrpath");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(paths).ok()?).ok()?;
    json.get("runtime")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str())
        .map(|root| std::path::Path::new(root).join("steamxr_win64.json"))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn steam_link_uses_running_steamvr_before_default_vdxr() {
        let steam = std::path::Path::new("SteamVR/steamxr_win64.json");
        let mut seen = Vec::new();
        let result = try_candidates(Some(steam), |path| {
            seen.push(path.map(|p| p.to_path_buf()));
            Ok("Steam Link")
        });
        assert_eq!(result.unwrap(), "Steam Link");
        assert_eq!(seen, vec![Some(steam.to_path_buf())]);
    }
    #[test]
    fn unavailable_steamvr_falls_back_to_virtual_desktop() {
        let mut seen = Vec::new();
        let result = try_candidates(Some(std::path::Path::new("steam.json")), |path| {
            seen.push(path.is_some());
            if path.is_some() {
                Err("No headset".into())
            } else {
                Ok("VDXR")
            }
        });
        assert_eq!(result.unwrap(), "VDXR");
        assert_eq!(seen, vec![true, false]);
    }
    #[test]
    fn no_running_steamvr_keeps_default_runtime() {
        let result = try_candidates(None, |path| {
            assert!(path.is_none());
            Ok("default")
        });
        assert_eq!(result.unwrap(), "default");
    }
}
