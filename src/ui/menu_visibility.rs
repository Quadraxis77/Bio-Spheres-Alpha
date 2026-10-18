//! Visibility decisions approved in the player's menu checklist.
//! Developer controls remain functional and retain their values; F11 reveals them.

use super::{panel::Panel, types::GlobalUiState};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub enum Visibility {
    Show,
    Advanced,
    Developer,
}

#[path = "menu_visibility_choices.rs"]
mod choices;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MenuAccess {
    pub advanced: bool,
    pub developer: bool,
}

impl MenuAccess {
    pub fn new(state: &GlobalUiState) -> Self {
        Self {
            advanced: state.show_advanced_options,
            developer: state.show_developer_options,
        }
    }

    pub fn allows(self, item: usize) -> bool {
        match choices::CHOICES[item.checked_sub(1).expect("menu IDs start at one")] {
            Visibility::Show => true,
            Visibility::Advanced => self.advanced || self.developer,
            Visibility::Developer => self.developer,
        }
    }

    pub fn allows_panel(self, panel: Panel) -> bool {
        let item = match panel {
            Panel::Viewport => 31,
            Panel::SceneManager => 32,
            Panel::CellInspector => 33,
            Panel::LineageViewer => 34,
            Panel::GenomeEditor => 35,
            Panel::PerformanceMonitor => 36,
            Panel::RenderingControls => 37,
            Panel::TimeScrubber => 38,
            Panel::ThemeEditor => 39,
            Panel::CameraSettings => 40,
            Panel::LightSettings | Panel::LightingSettings => 41,
            Panel::GizmoSettings => 42,
            Panel::CellTypeVisuals => 43,
            Panel::CaveSystem => 44,
            Panel::FluidSettings => 45,
            Panel::WorldSettings => 46,
            Panel::Modes => 47,
            Panel::AdhesionSettings => 48,
            Panel::ParentSettings => 49,
            Panel::CircleSliders => 50,
            Panel::QuaternionBall => 51,
            Panel::TimeSlider => 52,
            Panel::ModeGraph => 53,
            Panel::Help => 54,
            Panel::LeftPanel => 55,
            Panel::RightPanel => 56,
            Panel::BottomPanel => 57,
        };
        self.allows(item)
    }
}

impl GlobalUiState {
    pub fn menu_access(&self) -> MenuAccess {
        MenuAccess::new(self)
    }

    /// Developer access is deliberately transient and never serialized.
    pub fn toggle_developer_settings(&mut self) -> bool {
        self.show_developer_options = !self.show_developer_options;
        self.show_developer_options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct ReviewedChoice {
        id: String,
        visibility: Visibility,
    }

    #[test]
    fn reviewed_checklist_is_applied_exactly_at_all_three_access_levels() {
        let reviewed: Vec<ReviewedChoice> =
            serde_json::from_str(include_str!("../../assets/settings/menu-visibility.json"))
                .unwrap();
        assert_eq!(reviewed.len(), 402);
        for (index, choice) in reviewed.iter().enumerate() {
            assert_eq!(choice.id, format!("menu-{:04}", index + 1));
            for access in [
                MenuAccess::default(),
                MenuAccess {
                    advanced: true,
                    developer: false,
                },
                MenuAccess {
                    advanced: false,
                    developer: true,
                },
            ] {
                assert_eq!(
                    access.allows(index + 1),
                    match choice.visibility {
                        Visibility::Show => true,
                        Visibility::Advanced => access.advanced || access.developer,
                        Visibility::Developer => access.developer,
                    },
                    "{} at {access:?}",
                    choice.id
                );
            }
        }
    }

    #[test]
    fn advanced_never_unlocks_developer_controls_and_developer_access_does_not_persist() {
        let mut state = GlobalUiState::default();
        state.show_advanced_options = true;
        assert!(!state.menu_access().allows(61)); // Cave octaves.
        let before = ron::to_string(&state).unwrap();
        assert!(state.toggle_developer_settings());
        assert!(state.menu_access().allows(61));
        assert_eq!(ron::to_string(&state).unwrap(), before);
        let loaded: GlobalUiState = ron::from_str(&before).unwrap();
        assert!(!loaded.show_developer_options);
        assert!(!state.toggle_developer_settings());
    }

    #[test]
    fn cell_appearance_and_gravity_stay_visible_while_parent_panel_rules_take_precedence() {
        let ordinary = MenuAccess::default();
        assert!(ordinary.allows_panel(Panel::CellTypeVisuals));
        for item in 281..=357 {
            assert!(ordinary.allows(item));
        }
        for item in [263, 267, 270, 272, 273, 274] {
            assert!(ordinary.allows(item));
        }
        assert!(!ordinary.allows_panel(Panel::ThemeEditor));
        assert!(!ordinary.allows_panel(Panel::ModeGraph));
        assert!(ordinary.allows(358)); // Theme control shown once its parent is accessible.
    }
}
