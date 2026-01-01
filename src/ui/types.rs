//! UI type definitions for the Bio-Spheres application.
//!
//! This module contains core types used throughout the UI system,
//! including simulation modes and per-scene configuration.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Available simulation modes/scenes.
///
/// - Preview: Single-threaded CPU simulation + GPU rendering (genome editor)
/// - Gpu: Fully GPU-based simulation without CPU readbacks (high performance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SimulationMode {
    /// Genome editor preview - single-threaded CPU simulation with GPU rendering.
    /// Optimized for editing and debugging genomes with a single cell or small colony.
    #[default]
    Preview,
    /// Full GPU simulation - entirely on GPU without CPU readbacks.
    /// Optimized for large-scale simulations with thousands of cells.
    Gpu,
}

impl SimulationMode {
    /// Get the filename suffix for this mode's dock state file.
    pub fn dock_file_suffix(&self) -> &'static str {
        match self {
            SimulationMode::Preview => "preview",
            SimulationMode::Gpu => "gpu",
        }
    }

    /// Get display name for UI.
    pub fn display_name(&self) -> &'static str {
        match self {
            SimulationMode::Preview => "Genome Editor",
            SimulationMode::Gpu => "GPU Simulation",
        }
    }

    /// Get all available simulation modes.
    pub fn all() -> &'static [SimulationMode] {
        &[SimulationMode::Preview, SimulationMode::Gpu]
    }
}

impl std::fmt::Display for SimulationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}


/// Per-scene UI configuration.
///
/// Each simulation mode can have its own independent window configuration,
/// visibility settings, and locked windows.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneUiConfig {
    /// Which panels are visible in this scene.
    pub visibility: WindowVisibilitySettings,
    /// Which panels are locked (can't be moved/closed).
    pub locked_windows: HashSet<String>,
    /// Lock settings for this scene.
    pub lock_settings: LockSettings,
}

impl Default for SceneUiConfig {
    fn default() -> Self {
        Self {
            visibility: WindowVisibilitySettings::default(),
            locked_windows: HashSet::new(),
            lock_settings: LockSettings::default(),
        }
    }
}

impl SceneUiConfig {
    /// Create default config for a specific simulation mode.
    pub fn default_for_mode(mode: SimulationMode) -> Self {
        match mode {
            SimulationMode::Preview => Self {
                visibility: WindowVisibilitySettings {
                    show_cell_inspector: false,
                    show_genome_editor: true,
                    show_scene_manager: true,
                    show_performance_monitor: false,
                    show_rendering_controls: false,
                    show_time_scrubber: true,
                    show_theme_editor: false,
                    show_camera_settings: false,
                    show_lighting_settings: false,
                },
                locked_windows: HashSet::new(),
                lock_settings: LockSettings::default(),
            },
            SimulationMode::Gpu => Self {
                visibility: WindowVisibilitySettings {
                    show_cell_inspector: true,
                    show_genome_editor: false,
                    show_scene_manager: true,
                    show_performance_monitor: true,
                    show_rendering_controls: true,
                    show_time_scrubber: false,
                    show_theme_editor: false,
                    show_camera_settings: true,
                    show_lighting_settings: false,
                },
                locked_windows: HashSet::new(),
                lock_settings: LockSettings::default(),
            },
        }
    }
}

/// Window visibility settings for each panel type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowVisibilitySettings {
    pub show_cell_inspector: bool,
    pub show_genome_editor: bool,
    pub show_scene_manager: bool,
    pub show_performance_monitor: bool,
    pub show_rendering_controls: bool,
    pub show_time_scrubber: bool,
    pub show_theme_editor: bool,
    pub show_camera_settings: bool,
    pub show_lighting_settings: bool,
}

impl Default for WindowVisibilitySettings {
    fn default() -> Self {
        Self {
            show_cell_inspector: true,
            show_genome_editor: true,
            show_scene_manager: true,
            show_performance_monitor: true,
            show_rendering_controls: true,
            show_time_scrubber: true,
            show_theme_editor: false,
            show_camera_settings: false,
            show_lighting_settings: false,
        }
    }
}

/// Lock settings for UI elements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockSettings {
    /// Whether the tab bar is locked (can't be dragged).
    pub lock_tab_bar: bool,
    /// Whether tabs are locked (can't be reordered).
    pub lock_tabs: bool,
    /// Whether close buttons are locked (can't close panels).
    pub lock_close_buttons: bool,
}

impl Default for LockSettings {
    fn default() -> Self {
        Self {
            lock_tab_bar: false,
            lock_tabs: false,
            lock_close_buttons: false,
        }
    }
}

/// Global UI state shared across all UI components.
///
/// This struct tracks the current simulation mode, UI scale, lock settings,
/// and per-scene configurations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalUiState {
    /// Current simulation mode/scene.
    pub current_mode: SimulationMode,

    /// Global UI scale factor.
    pub ui_scale: f32,

    /// Whether windows are globally locked.
    pub windows_locked: bool,

    /// Whether the tab bar is locked.
    pub lock_tab_bar: bool,

    /// Whether tabs are locked.
    pub lock_tabs: bool,

    /// Whether close buttons are locked.
    pub lock_close_buttons: bool,

    /// Per-scene configurations.
    pub scene_configs: HashMap<SimulationMode, SceneUiConfig>,
}

impl Default for GlobalUiState {
    fn default() -> Self {
        let mut scene_configs = HashMap::new();
        for mode in SimulationMode::all() {
            scene_configs.insert(*mode, SceneUiConfig::default_for_mode(*mode));
        }

        Self {
            current_mode: SimulationMode::Preview,
            ui_scale: 1.0,
            windows_locked: false,
            lock_tab_bar: false,
            lock_tabs: false,
            lock_close_buttons: false,
            scene_configs,
        }
    }
}

impl GlobalUiState {
    /// Get the current scene's config.
    pub fn current_config(&self) -> &SceneUiConfig {
        self.scene_configs
            .get(&self.current_mode)
            .expect("Scene config should exist for current mode")
    }

    /// Get mutable reference to current scene's config.
    pub fn current_config_mut(&mut self) -> &mut SceneUiConfig {
        self.scene_configs
            .get_mut(&self.current_mode)
            .expect("Scene config should exist for current mode")
    }

    /// Check if a panel is visible in the current scene by name.
    pub fn is_panel_visible(&self, panel_name: &str) -> bool {
        let vis = &self.current_config().visibility;
        match panel_name {
            "CellInspector" => vis.show_cell_inspector,
            "GenomeEditor" => vis.show_genome_editor,
            "SceneManager" => vis.show_scene_manager,
            "PerformanceMonitor" => vis.show_performance_monitor,
            "RenderingControls" => vis.show_rendering_controls,
            "TimeScrubber" => vis.show_time_scrubber,
            "ThemeEditor" => vis.show_theme_editor,
            "CameraSettings" => vis.show_camera_settings,
            "LightingSettings" => vis.show_lighting_settings,
            _ => true, // Unknown panels are visible by default
        }
    }

    /// Set panel visibility in the current scene by name.
    pub fn set_panel_visible(&mut self, panel_name: &str, visible: bool) {
        let vis = &mut self.current_config_mut().visibility;
        match panel_name {
            "CellInspector" => vis.show_cell_inspector = visible,
            "GenomeEditor" => vis.show_genome_editor = visible,
            "SceneManager" => vis.show_scene_manager = visible,
            "PerformanceMonitor" => vis.show_performance_monitor = visible,
            "RenderingControls" => vis.show_rendering_controls = visible,
            "TimeScrubber" => vis.show_time_scrubber = visible,
            "ThemeEditor" => vis.show_theme_editor = visible,
            "CameraSettings" => vis.show_camera_settings = visible,
            "LightingSettings" => vis.show_lighting_settings = visible,
            _ => {} // Unknown panels are ignored
        }
    }

    /// Check if a panel is locked in the current scene.
    pub fn is_panel_locked(&self, panel_name: &str) -> bool {
        self.current_config().locked_windows.contains(panel_name)
    }

    /// Set panel lock state in the current scene.
    pub fn set_panel_locked(&mut self, panel_name: &str, locked: bool) {
        let config = self.current_config_mut();
        if locked {
            config.locked_windows.insert(panel_name.to_string());
        } else {
            config.locked_windows.remove(panel_name);
        }
    }
}
