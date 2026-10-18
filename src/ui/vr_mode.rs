//! The player's presentation choice survives headset presence changes.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Preference {
    #[default]
    Automatic,
    Desktop,
    Vr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VrModeControl {
    preference: Preference,
    pub available: bool,
    pub active: bool,
}

impl Default for VrModeControl {
    fn default() -> Self {
        Self {
            preference: Preference::Automatic,
            available: cfg!(feature = "vr"),
            active: false,
        }
    }
}

impl VrModeControl {
    pub fn enabled(&self) -> bool {
        self.available && self.preference != Preference::Desktop
    }

    pub fn selected(&self) -> bool {
        match self.preference {
            Preference::Automatic => self.active,
            Preference::Desktop => false,
            Preference::Vr => true,
        }
    }

    pub fn toggle(&mut self) {
        if self.available {
            self.preference = if self.selected() {
                Preference::Desktop
            } else {
                Preference::Vr
            };
        }
    }

    pub fn label(&self) -> &'static str {
        if self.selected() {
            "Desktop (F10)"
        } else {
            "Enter VR (F10)"
        }
    }

    pub fn tooltip(&self) -> &'static str {
        if !self.available {
            "VR is unavailable in this build or disabled for this launch"
        } else if self.selected() && !self.active {
            "Waiting for headset. Click or press F10 to return to desktop."
        } else if self.selected() {
            "Switch to desktop (F10)"
        } else {
            "Switch to VR (F10)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_desktop_choice_survives_a_connected_or_reworn_headset() {
        let mut control = VrModeControl {
            available: true,
            active: true,
            ..Default::default()
        };
        control.toggle();
        for active in [true, false, true] {
            control.active = active;
            assert!(!control.enabled());
            assert!(!control.selected());
        }
        control.toggle();
        assert!(control.enabled());
        assert!(control.selected());
    }

    #[test]
    fn entering_vr_without_a_headset_can_be_cancelled_before_attachment() {
        let mut control = VrModeControl {
            available: true,
            ..Default::default()
        };
        assert!(!control.selected());
        control.toggle();
        assert!(control.enabled());
        assert!(control.selected());
        control.toggle();
        assert!(!control.enabled());
        control.active = true; // a late presence event cannot undo the choice
        assert!(!control.selected());
    }
}
