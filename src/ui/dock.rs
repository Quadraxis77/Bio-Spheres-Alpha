//! Dock manager for the Bio-Spheres docking system.
//!
//! This module provides the DockManager which handles per-scene dock layouts
//! using egui_dock. Each simulation mode has its own independent dock state
//! that persists across sessions.

use crate::ui::panel::Panel;
use crate::ui::types::SimulationMode;
use egui_dock::{DockState, NodeIndex};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Auto-save interval for dock layouts (30 seconds).
const AUTO_SAVE_INTERVAL: Duration = Duration::from_secs(30);

/// Manages dock layouts for all simulation modes.
///
/// The DockManager maintains separate dock states for each simulation mode,
/// allowing users to have different panel arrangements for different workflows.
pub struct DockManager {
    /// Per-scene dock states keyed by SimulationMode.
    scene_trees: HashMap<SimulationMode, DockState<Panel>>,
    /// Current simulation mode.
    current_mode: SimulationMode,
    /// Timer for auto-save functionality.
    save_timer: Instant,
    /// Whether the layout has changed since last save.
    dirty: bool,
}

impl DockManager {
    /// Create a new DockManager, loading saved layouts or creating defaults.
    pub fn new() -> Self {
        let mut scene_trees = HashMap::new();

        // Initialize dock states for all modes
        for mode in SimulationMode::all() {
            let dock_state = Self::load_or_create_default(*mode);
            scene_trees.insert(*mode, dock_state);
        }

        Self {
            scene_trees,
            current_mode: SimulationMode::Preview,
            save_timer: Instant::now(),
            dirty: false,
        }
    }

    /// Get the current scene's dock state.
    pub fn current_tree(&self) -> &DockState<Panel> {
        self.scene_trees
            .get(&self.current_mode)
            .expect("Dock state should exist for current mode")
    }

    /// Get mutable reference to current scene's dock state.
    pub fn current_tree_mut(&mut self) -> &mut DockState<Panel> {
        self.dirty = true;
        self.scene_trees
            .get_mut(&self.current_mode)
            .expect("Dock state should exist for current mode")
    }

    /// Get the current simulation mode.
    pub fn current_mode(&self) -> SimulationMode {
        self.current_mode
    }

    /// Switch to a different simulation mode's layout.
    ///
    /// This saves the current layout before switching and loads
    /// the target mode's layout.
    pub fn switch_mode(&mut self, mode: SimulationMode) {
        if mode == self.current_mode {
            return;
        }

        // Save current layout before switching
        self.save_current();

        // Switch to new mode
        self.current_mode = mode;

        // Ensure the new mode has a dock state
        if !self.scene_trees.contains_key(&mode) {
            let dock_state = Self::load_or_create_default(mode);
            self.scene_trees.insert(mode, dock_state);
        }

        log::info!("Switched to {} mode", mode.display_name());
    }

    /// Mark the layout as dirty (needs saving).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Check if auto-save is needed and save if so.
    pub fn auto_save(&mut self) {
        if self.dirty && self.save_timer.elapsed() >= AUTO_SAVE_INTERVAL {
            self.save_current();
            self.save_timer = Instant::now();
        }
    }

    /// Reset current scene to embedded default layout.
    pub fn reset_current_to_embedded_default(&mut self) {
        log::info!("Resetting {} layout to embedded default", self.current_mode.display_name());
        let default_state = Self::load_default_layout(self.current_mode);
        self.scene_trees.insert(self.current_mode, default_state);
        self.dirty = true;
        log::info!(
            "Successfully reset {} layout to embedded default",
            self.current_mode.display_name()
        );
    }

    /// Reset all scenes to embedded default layouts.
    pub fn reset_all_to_embedded_default(&mut self) {
        for mode in SimulationMode::all() {
            let default_state = Self::load_default_layout(*mode);
            self.scene_trees.insert(*mode, default_state);
        }
        self.dirty = true;
        log::info!("Reset all layouts to embedded defaults");
    }

    /// Reset current scene to default layout.
    pub fn reset_current_to_default(&mut self) {
        log::info!("Resetting {} layout to default", self.current_mode.display_name());
        let default_state = Self::load_default_layout(self.current_mode);
        self.scene_trees.insert(self.current_mode, default_state);
        self.dirty = true;
        log::info!(
            "Successfully reset {} layout to saved default",
            self.current_mode.display_name()
        );
    }

    /// Reset all scenes to default layouts.
    pub fn reset_all_to_default(&mut self) {
        for mode in SimulationMode::all() {
            let default_state = Self::load_default_layout(*mode);
            self.scene_trees.insert(*mode, default_state);
        }
        self.dirty = true;
        log::info!("Reset all layouts to saved defaults");
    }

    /// Get list of available simulation modes.
    pub fn available_modes() -> &'static [SimulationMode] {
        SimulationMode::all()
    }

    /// Get the dock file path for a specific mode.
    fn dock_file_path(mode: SimulationMode) -> PathBuf {
        PathBuf::from(format!("dock_state_{}.ron", mode.dock_file_suffix()))
    }

    /// Save current layout as the new default for new players.
    ///
    /// This copies the current dock layout files to serve as the new defaults
    /// that will be used when the layout files don't exist (first-time startup).
    pub fn save_current_as_default(&self) -> Result<(), DockSaveError> {
        // Save current layouts to backup files that will be used as defaults
        for (mode, state) in &self.scene_trees {
            let default_path = Self::default_layout_file_path(*mode);
            Self::save_to_file(&default_path, state)?;
            log::info!("Saved current {} layout as new default to {:?}", mode.display_name(), default_path);
        }
        Ok(())
    }

    /// Get the path for the default layout file for a mode.
    fn default_layout_file_path(mode: SimulationMode) -> PathBuf {
        PathBuf::from(format!("default_dock_state_{}.ron", mode.dock_file_suffix()))
    }

    /// Load default layout from embedded resources or create hardcoded default if not available.
    fn load_default_layout(mode: SimulationMode) -> DockState<Panel> {
        // Try to load from embedded resources first
        let embedded_content = match mode {
            SimulationMode::Preview => include_str!("../../default_dock_state_preview.ron"),
            SimulationMode::Gpu => include_str!("../../default_dock_state_gpu.ron"),
        };
        
        log::info!("Attempting to load embedded default layout for {}", mode.display_name());
        log::debug!("Embedded content length: {} characters", embedded_content.len());
        
        match ron::from_str::<DockState<Panel>>(embedded_content) {
            Ok(state) => {
                log::info!("Successfully loaded embedded default layout for {}", mode.display_name());
                return state;
            }
            Err(e) => {
                log::error!(
                    "Failed to parse embedded default layout for {}: {}. Using hardcoded default.",
                    mode.display_name(),
                    e
                );
                log::debug!("Embedded content preview: {}", &embedded_content[..embedded_content.len().min(200)]);
            }
        }
        
        // Fall back to hardcoded default
        log::info!("Using hardcoded default layout for {}", mode.display_name());
        Self::create_hardcoded_default_layout(mode)
    }

    /// Load dock state from disk or create default for a mode.
    fn load_or_create_default(mode: SimulationMode) -> DockState<Panel> {
        let path = Self::dock_file_path(mode);

        if path.exists() {
            match Self::load_from_file(&path) {
                Ok(state) => {
                    log::info!("Loaded dock layout for {} from {:?}", mode.display_name(), path);
                    return state;
                }
                Err(e) => {
                    log::warn!(
                        "Failed to load dock layout for {}: {}. Using embedded default.",
                        mode.display_name(),
                        e
                    );
                }
            }
        } else {
            log::info!("No saved dock layout found for {}, using embedded default", mode.display_name());
        }

        // Use embedded default layout (this will be used for first-time users)
        Self::load_default_layout(mode)
    }

    /// Load dock state from a RON file.
    fn load_from_file(path: &PathBuf) -> Result<DockState<Panel>, DockLoadError> {
        let contents = std::fs::read_to_string(path)?;
        let state: DockState<Panel> = ron::from_str(&contents)?;
        Ok(state)
    }

    /// Save dock state to a RON file.
    fn save_to_file(path: &PathBuf, state: &DockState<Panel>) -> Result<(), DockSaveError> {
        let contents = ron::ser::to_string_pretty(state, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Save current scene's layout to disk.
    pub fn save_current(&mut self) {
        if !self.dirty {
            return;
        }

        let path = Self::dock_file_path(self.current_mode);
        if let Some(state) = self.scene_trees.get(&self.current_mode) {
            match Self::save_to_file(&path, state) {
                Ok(()) => {
                    log::debug!(
                        "Saved dock layout for {} to {:?}",
                        self.current_mode.display_name(),
                        path
                    );
                    self.dirty = false;
                }
                Err(e) => {
                    log::warn!(
                        "Failed to save dock layout for {}: {}",
                        self.current_mode.display_name(),
                        e
                    );
                }
            }
        }
    }

    /// Save all scene layouts to disk.
    pub fn save_all(&self) {
        for (mode, state) in &self.scene_trees {
            let path = Self::dock_file_path(*mode);
            match Self::save_to_file(&path, state) {
                Ok(()) => {
                    log::debug!("Saved dock layout for {} to {:?}", mode.display_name(), path);
                }
                Err(e) => {
                    log::warn!("Failed to save dock layout for {}: {}", mode.display_name(), e);
                }
            }
        }
    }

    /// Load layout from disk for a specific mode.
    pub fn load_for_mode(&mut self, mode: SimulationMode) {
        let dock_state = Self::load_or_create_default(mode);
        self.scene_trees.insert(mode, dock_state);
        log::info!("Reloaded dock layout for {}", mode.display_name());
    }

    /// Create hardcoded default layout for a specific mode.
    fn create_hardcoded_default_layout(mode: SimulationMode) -> DockState<Panel> {
        match mode {
            SimulationMode::Preview => create_default_preview_layout(),
            SimulationMode::Gpu => create_default_gpu_layout(),
        }
    }
}

impl Default for DockManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for dock layout loading.
#[derive(Debug, thiserror::Error)]
pub enum DockLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON parse error: {0}")]
    Ron(#[from] ron::error::SpannedError),
}

/// Error type for dock layout saving.
#[derive(Debug, thiserror::Error)]
pub enum DockSaveError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON serialize error: {0}")]
    Ron(#[from] ron::Error),
}


/// Create the default dock layout for Preview mode (Genome Editor).
///
/// Layout:
/// ```text
/// +--------+--------------------------------+--------+
/// |        |                                |        |
/// | Modes  |                                | Adhes. |
/// |        |                                |        |
/// +--------+          Viewport              +--------+
/// |        |                                |        |
/// | Name/  |                                | Parent |
/// | Type   |                                |        |
/// +--------+--------------------------------+--------+
/// |              Time Slider / Scrubber            |
/// +------------------------------------------------+
/// ```
pub fn create_default_preview_layout() -> DockState<Panel> {
    // Start with viewport in the center
    let mut dock_state = DockState::new(vec![Panel::Viewport]);
    let tree = dock_state.main_surface_mut();

    // Split left panel (20% width)
    let [_center, left] = tree.split_left(NodeIndex::root(), 0.20, vec![Panel::Modes]);

    // Split the left panel vertically for Name/Type below Modes
    tree.split_below(left, 0.5, vec![Panel::NameTypeEditor]);

    // Split right panel (20% width) from the center
    let [center, right] = tree.split_right(NodeIndex::root(), 0.80, vec![Panel::AdhesionSettings]);

    // Split the right panel vertically for Parent below Adhesion
    tree.split_below(right, 0.5, vec![Panel::ParentSettings]);

    // Split bottom panel (15% height) for time controls
    tree.split_below(center, 0.85, vec![Panel::TimeSlider, Panel::TimeScrubber]);

    dock_state
}

/// Create the default dock layout for GPU mode (Full Simulation).
///
/// Layout:
/// ```text
/// +--------+--------------------------------+--------+
/// |        |                                |        |
/// | Cell   |                                | Render |
/// | Insp.  |                                |        |
/// +--------+          Viewport              +--------+
/// |        |                                |        |
/// | Scene  |                                | Camera |
/// | Mgr    |                                |        |
/// +--------+--------------------------------+--------+
/// |              Performance Monitor               |
/// +------------------------------------------------+
/// ```
pub fn create_default_gpu_layout() -> DockState<Panel> {
    // Start with viewport in the center
    let mut dock_state = DockState::new(vec![Panel::Viewport]);
    let tree = dock_state.main_surface_mut();

    // Split left panel (20% width)
    let [_center, left] = tree.split_left(NodeIndex::root(), 0.20, vec![Panel::CellInspector]);

    // Split the left panel vertically for Scene Manager below Cell Inspector
    tree.split_below(left, 0.5, vec![Panel::SceneManager]);

    // Split right panel (20% width) from the center
    let [center, right] = tree.split_right(NodeIndex::root(), 0.80, vec![Panel::RenderingControls]);

    // Split the right panel vertically for Camera below Rendering
    tree.split_below(right, 0.5, vec![Panel::CameraSettings]);

    // Split bottom panel (15% height) for performance monitor
    tree.split_below(center, 0.85, vec![Panel::PerformanceMonitor]);

    dock_state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dock_manager_creation() {
        let manager = DockManager::new();
        assert_eq!(manager.current_mode(), SimulationMode::Preview);
    }

    #[test]
    fn test_mode_switching() {
        let mut manager = DockManager::new();
        assert_eq!(manager.current_mode(), SimulationMode::Preview);

        manager.switch_mode(SimulationMode::Gpu);
        assert_eq!(manager.current_mode(), SimulationMode::Gpu);

        manager.switch_mode(SimulationMode::Preview);
        assert_eq!(manager.current_mode(), SimulationMode::Preview);
    }

    #[test]
    fn test_default_preview_layout_has_viewport() {
        let layout = create_default_preview_layout();
        let has_viewport = layout
            .iter_all_tabs()
            .any(|(_, tab)| *tab == Panel::Viewport);
        assert!(has_viewport, "Preview layout should contain Viewport");
    }

    #[test]
    fn test_default_gpu_layout_has_viewport() {
        let layout = create_default_gpu_layout();
        let has_viewport = layout
            .iter_all_tabs()
            .any(|(_, tab)| *tab == Panel::Viewport);
        assert!(has_viewport, "GPU layout should contain Viewport");
    }

    #[test]
    fn test_default_preview_layout_has_genome_panels() {
        let layout = create_default_preview_layout();
        let tabs: Vec<_> = layout.iter_all_tabs().map(|(_, tab)| *tab).collect();

        assert!(tabs.contains(&Panel::Modes), "Should have Modes panel");
        assert!(
            tabs.contains(&Panel::NameTypeEditor),
            "Should have NameTypeEditor panel"
        );
        assert!(
            tabs.contains(&Panel::TimeSlider),
            "Should have TimeSlider panel"
        );
    }

    #[test]
    fn test_default_gpu_layout_has_monitoring_panels() {
        let layout = create_default_gpu_layout();
        let tabs: Vec<_> = layout.iter_all_tabs().map(|(_, tab)| *tab).collect();

        assert!(
            tabs.contains(&Panel::PerformanceMonitor),
            "Should have PerformanceMonitor panel"
        );
        assert!(
            tabs.contains(&Panel::CellInspector),
            "Should have CellInspector panel"
        );
    }

    #[test]
    fn test_reset_to_default() {
        let mut manager = DockManager::new();

        // Modify the layout
        manager.current_tree_mut().push_to_focused_leaf(Panel::ThemeEditor);

        // Reset to default
        manager.reset_current_to_default();

        // Verify it's back to default (no ThemeEditor in default preview)
        let tabs: Vec<_> = manager
            .current_tree()
            .iter_all_tabs()
            .map(|(_, tab)| *tab)
            .collect();

        // ThemeEditor is not in the default preview layout
        assert!(
            !tabs.contains(&Panel::ThemeEditor),
            "ThemeEditor should not be in default preview layout"
        );
    }
}
