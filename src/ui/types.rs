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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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

    /// Occlusion culling bias (negative = more aggressive, positive = more conservative)
    #[serde(default = "default_occlusion_bias")]
    pub occlusion_bias: f32,

    /// Occlusion mip level override (-1 = auto, 0+ = force specific mip)
    #[serde(default = "default_mip_override")]
    pub occlusion_mip_override: i32,

    /// Minimum screen-space size (pixels) for occlusion culling
    #[serde(default)]
    pub occlusion_min_screen_size: f32,

    /// Minimum distance for occlusion culling
    #[serde(default)]
    pub occlusion_min_distance: f32,

    /// Whether occlusion culling is enabled
    #[serde(default = "default_true")]
    pub occlusion_enabled: bool,

    /// Whether frustum culling is enabled
    #[serde(default = "default_true")]
    pub frustum_enabled: bool,

    /// Whether GPU readbacks are enabled (cell count, culling stats)
    /// Disabling this can improve performance by avoiding CPU-GPU sync
    #[serde(default = "default_true")]
    pub gpu_readbacks_enabled: bool,



    /// Whether to show adhesion lines between cells
    #[serde(default = "default_true")]
    pub show_adhesion_lines: bool,

    /// Target cell capacity (applied on scene reset)
    #[serde(default = "default_cell_capacity")]
    pub cell_capacity: u32,

    /// World diameter in simulation units (applied on scene reset)
    #[serde(default = "default_world_diameter")]
    pub world_diameter: f32,

    /// LOD scale factor for distance calculations (higher = more aggressive LOD)
    #[serde(default = "default_lod_scale_factor")]
    pub lod_scale_factor: f32,

    /// LOD threshold for Low (32x32) to Medium (64x64) transition
    #[serde(default = "default_lod_threshold_low")]
    pub lod_threshold_low: f32,

    /// LOD threshold for Medium (64x64) to High (128x128) transition
    #[serde(default = "default_lod_threshold_medium")]
    pub lod_threshold_medium: f32,

    /// LOD threshold for High (128x128) to Ultra (256x256) transition
    #[serde(default = "default_lod_threshold_high")]
    pub lod_threshold_high: f32,

    /// Whether to show debug colors for LOD levels
    #[serde(default)]
    pub lod_debug_colors: bool,

    /// Whether the low FPS warning dialog is shown
    #[serde(skip)]
    pub show_low_fps_dialog: bool,

    /// Whether the reset confirmation dialog is shown
    #[serde(skip)]
    pub show_reset_dialog: bool,

    /// Requested mode change (processed by main app loop)
    #[serde(skip)]
    pub mode_request: Option<SimulationMode>,
}

fn default_mip_override() -> i32 {
    -1
}

fn default_occlusion_bias() -> f32 {
    0.005 // Small positive bias to prevent self-occlusion from temporal jitter
}

fn default_cell_capacity() -> u32 {
    20_000
}

fn default_world_diameter() -> f32 {
    200.0
}

fn default_lod_scale_factor() -> f32 {
    500.0 // Default scale factor for screen radius calculation
}

fn default_lod_threshold_low() -> f32 {
    10.0 // Low to Medium transition
}

fn default_lod_threshold_medium() -> f32 {
    25.0 // Medium to High transition
}

fn default_lod_threshold_high() -> f32 {
    50.0 // High to Ultra transition
}

fn default_true() -> bool {
    true
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
            occlusion_bias: 0.0,
            occlusion_mip_override: -1,
            occlusion_min_screen_size: 0.0,
            occlusion_min_distance: 0.0,
            occlusion_enabled: true,
            frustum_enabled: true,
            gpu_readbacks_enabled: true,
            show_adhesion_lines: true,
            cell_capacity: 20_000,
            world_diameter: 200.0,
            lod_scale_factor: 500.0,
            lod_threshold_low: 10.0,
            lod_threshold_medium: 25.0,
            lod_threshold_high: 50.0,
            lod_debug_colors: false,
            show_low_fps_dialog: false,
            show_reset_dialog: false,
            mode_request: None,
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

    /// Request a mode switch.
    pub fn request_mode_switch(&mut self, mode: SimulationMode) {
        if mode != self.current_mode {
            self.mode_request = Some(mode);
        }
    }

    /// Reset to embedded default UI state.
    pub fn reset_to_embedded_default(&mut self) {
        *self = Self::load_embedded_default();
        log::info!("Reset UI state to embedded default");
    }

    /// Take the pending mode request, if any.
    pub fn take_mode_request(&mut self) -> Option<SimulationMode> {
        self.mode_request.take()
    }

    /// Save UI state to disk.
    pub fn save(&self) -> Result<(), UiStateSaveError> {
        let path = std::path::PathBuf::from("ui_state.ron");
        let contents = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Load UI state from disk, or create default if file doesn't exist.
    pub fn load() -> Self {
        let path = std::path::PathBuf::from("ui_state.ron");
        
        if path.exists() {
            match Self::load_from_file(&path) {
                Ok(state) => {
                    log::info!("Loaded UI state from {:?}", path);
                    return state;
                }
                Err(e) => {
                    log::warn!("Failed to load UI state: {}. Using embedded default.", e);
                }
            }
        } else {
            log::info!("No saved UI state found, using embedded default");
        }
        
        // Try to load from embedded default
        Self::load_embedded_default()
    }

    /// Load embedded default UI state.
    fn load_embedded_default() -> Self {
        let embedded_content = include_str!("../../default_ui_state.ron");
        
        log::info!("Attempting to load embedded default UI state");
        log::debug!("Embedded UI state content length: {} characters", embedded_content.len());
        
        match ron::from_str::<GlobalUiState>(embedded_content) {
            Ok(state) => {
                log::info!("Successfully loaded embedded default UI state");
                state
            }
            Err(e) => {
                log::error!(
                    "Failed to parse embedded default UI state: {}. Using hardcoded default.",
                    e
                );
                log::debug!("Embedded content preview: {}", &embedded_content[..embedded_content.len().min(200)]);
                Self::default()
            }
        }
    }

    /// Load UI state from a specific file.
    fn load_from_file(path: &std::path::PathBuf) -> Result<Self, UiStateLoadError> {
        let contents = std::fs::read_to_string(path)?;
        let mut state: Self = ron::from_str(&contents)?;
        // Clear mode request on load (it's transient)
        state.mode_request = None;
        Ok(state)
    }
}

/// Error type for UI state loading.
#[derive(Debug, thiserror::Error)]
pub enum UiStateLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON parse error: {0}")]
    Ron(#[from] ron::error::SpannedError),
}

/// Error type for UI state saving.
#[derive(Debug, thiserror::Error)]
pub enum UiStateSaveError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON serialize error: {0}")]
    Ron(#[from] ron::Error),
}