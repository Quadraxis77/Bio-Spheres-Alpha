//! Prevents system sleep while Bio-Spheres needs to keep a simulation alive.

#[cfg(target_os = "linux")]
mod linux {
    use zbus::{blocking::Connection, proxy};

    #[proxy(
        interface = "org.freedesktop.login1.Manager",
        default_service = "org.freedesktop.login1",
        default_path = "/org/freedesktop/login1"
    )]
    trait Manager {
        fn inhibit(
            &self,
            what: &str,
            who: &str,
            why: &str,
            mode: &str,
        ) -> zbus::Result<zbus::zvariant::OwnedFd>;
    }

    /// Low-level lid-switch lock. Many logind installations intentionally
    /// ignore ordinary `sleep` inhibitors for lid-close events, so this lock is
    /// required to keep a long-running simulation alive with the lid closed.
    pub struct LidSwitchInhibitor {
        _fd: zbus::zvariant::OwnedFd,
    }

    impl LidSwitchInhibitor {
        pub fn acquire() -> zbus::Result<Self> {
            let connection = Connection::system()?;
            let manager = ManagerProxyBlocking::new(&connection)?;
            let fd = manager.inhibit(
                "handle-lid-switch",
                "Bio-Spheres",
                "A Bio-Spheres simulation is running",
                "block",
            )?;
            Ok(Self { _fd: fd })
        }
    }
}

/// Owns the platform sleep-inhibition assertion.
///
/// `keep_active::KeepActive` releases its assertion when dropped, so clearing
/// the guard on focus loss also covers normal shutdown and unwinding.
pub struct SleepInhibitor {
    guard: Option<keep_active::KeepActive>,
    #[cfg(target_os = "linux")]
    lid_switch_guard: Option<linux::LidSwitchInhibitor>,
    active: bool,
}

impl SleepInhibitor {
    pub fn new(active: bool) -> Self {
        let mut inhibitor = Self {
            guard: None,
            #[cfg(target_os = "linux")]
            lid_switch_guard: None,
            active: false,
        };
        inhibitor.set_active(active);
        inhibitor
    }

    /// Acquires or releases all platform sleep assertions for the current app state.
    pub fn set_active(&mut self, active: bool) {
        if self.active == active {
            return;
        }
        self.active = active;

        if !active {
            #[cfg(target_os = "linux")]
            let released_lid_switch = self.lid_switch_guard.take().is_some();
            let released_system = self.guard.take().is_some();
            if released_system || {
                #[cfg(target_os = "linux")]
                {
                    released_lid_switch
                }
                #[cfg(not(target_os = "linux"))]
                {
                    false
                }
            } {
                log::info!("Released system sleep inhibitors");
            }
            return;
        }

        match keep_active::Builder::default()
            .display(true)
            .idle(true)
            .sleep(true)
            .reason("Bio-Spheres is active")
            .app_name("Bio-Spheres")
            .app_reverse_domain("com.biospheres.app")
            .create()
        {
            Ok(guard) => {
                self.guard = Some(guard);
                log::info!("Acquired display, idle, and system sleep inhibitors");
            }
            Err(error) => {
                log::warn!("Could not acquire full sleep inhibitor: {error}");

                // Some minimal Linux desktops have no ScreenSaver service. Keep
                // system sleep inhibited even when display inhibition is absent.
                match keep_active::Builder::default()
                    .idle(true)
                    .sleep(true)
                    .reason("Bio-Spheres is active")
                    .app_name("Bio-Spheres")
                    .app_reverse_domain("com.biospheres.app")
                    .create()
                {
                    Ok(guard) => self.guard = Some(guard),
                    Err(error) => {
                        log::warn!("Could not acquire fallback sleep inhibitor: {error}")
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        match linux::LidSwitchInhibitor::acquire() {
            Ok(guard) => {
                self.lid_switch_guard = Some(guard);
                log::info!("Acquired lid-switch sleep inhibitor");
            }
            Err(error) => {
                log::warn!("Could not acquire lid-switch sleep inhibitor: {error}");
            }
        }
    }
}
