//! Panel context for passing data to panel renderers.
//!
//! This module provides the PanelContext struct which contains all the
//! data and state needed by panels to render their UI and interact with
//! the simulation.

use crate::genome::Genome;
use crate::scene::SceneManager;
use crate::ui::camera::CameraController;
use crate::ui::performance::PerformanceMetrics;
use crate::ui::types::SimulationMode;
use std::path::PathBuf;

/// Request for scene mode changes.
///
/// Panels can request a scene mode change by setting this value.
/// The main application loop will process the request and switch scenes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SceneModeRequest {
    /// No change requested
    #[default]
    None,
    /// Request to switch to Preview mode
    SwitchToPreview,
    /// Request to switch to GPU mode
    SwitchToGpu,
    /// Request to toggle pause state
    TogglePause,
    /// Request to reset the simulation
    Reset,
    /// Request to toggle fast forward mode (2x speed)
    ToggleFastForward,
    /// Request to set simulation speed
    SetSpeed(f32),
    /// Request to set simulation speed and unpause
    SetSpeedAndUnpause(f32),
}

impl SceneModeRequest {
    /// Get the target mode if a switch is requested.
    pub fn target_mode(&self) -> Option<SimulationMode> {
        match self {
            SceneModeRequest::SwitchToPreview => Some(SimulationMode::Preview),
            SceneModeRequest::SwitchToGpu => Some(SimulationMode::Gpu),
            _ => None,
        }
    }

    /// Check if a mode switch is requested.
    pub fn is_requested(&self) -> bool {
        !matches!(self, SceneModeRequest::None)
    }

    /// Clear the request.
    pub fn clear(&mut self) {
        *self = SceneModeRequest::None;
    }
}

/// State for the genome editor UI.
///
/// Contains transient UI state for genome editing panels like
/// rename dialogs, color pickers, and widget configurations.
#[derive(Debug, Clone, Default)]
pub struct GenomeEditorState {
    // Mode panel state
    /// Index of mode currently being renamed (None if not renaming)
    pub renaming_mode: Option<usize>,
    /// Buffer for rename text input
    pub rename_buffer: String,
    /// Currently selected mode index
    pub selected_mode_index: usize,
    /// Whether the "copy into" dialog is open
    pub copy_into_dialog_open: bool,
    /// Source mode index for copy operation
    pub copy_into_source: usize,
    /// Color picker state: (mode index, current color)
    pub color_picker_state: Option<(usize, egui::ecolor::Hsva)>,

    // Quaternion ball state
    /// Whether quaternion ball snapping is enabled
    pub qball_snapping: bool,
    /// Locked axis for first quaternion ball (-1 = none, 0 = X, 1 = Y, 2 = Z)
    pub qball1_locked_axis: i32,
    /// Initial distance for first quaternion ball
    pub qball1_initial_distance: f32,
    /// Locked axis for second quaternion ball
    pub qball2_locked_axis: i32,
    /// Initial distance for second quaternion ball
    pub qball2_initial_distance: f32,
    /// Persistent orientation for Child A quaternion ball
    pub child_a_orientation: glam::Quat,
    /// Persistent orientation for Child B quaternion ball
    pub child_b_orientation: glam::Quat,
    
    // Axis tracking for Child A quaternion ball (UI feedback only)
    pub child_a_x_axis_lat: f32,
    pub child_a_x_axis_lon: f32,
    pub child_a_y_axis_lat: f32,
    pub child_a_y_axis_lon: f32,
    pub child_a_z_axis_lat: f32,
    pub child_a_z_axis_lon: f32,
    
    // Axis tracking for Child B quaternion ball (UI feedback only)
    pub child_b_x_axis_lat: f32,
    pub child_b_x_axis_lon: f32,
    pub child_b_y_axis_lat: f32,
    pub child_b_y_axis_lon: f32,
    pub child_b_z_axis_lat: f32,
    pub child_b_z_axis_lon: f32,

    // Keep Adhesion state
    /// Whether Child A should keep adhesion
    pub child_a_keep_adhesion: bool,
    /// Whether Child B should keep adhesion
    pub child_b_keep_adhesion: bool,

    // Circular slider state
    /// Whether angle snapping is enabled for circular sliders
    pub enable_snapping: bool,

    // Time slider state
    /// Current time value for preview (0-60 seconds)
    pub time_value: f32,
    /// Maximum preview duration (60 seconds)
    pub max_preview_duration: f32,
    /// Whether the time slider is being dragged
    pub time_slider_dragging: bool,
    
    // Cave system parameters
    pub cave_density: f32,
    pub cave_scale: f32,
    pub cave_octaves: u32,
    pub cave_persistence: f32,
    pub cave_threshold: f32,
    pub cave_smoothness: f32,
    pub cave_seed: u32,
    pub cave_resolution: u32,
    pub cave_collision_enabled: bool,
    pub cave_collision_stiffness: f32,
    pub cave_collision_damping: f32,
    pub cave_substeps: u32,
    pub cave_params_dirty: bool,
    
    // Orientation gizmo state
    /// Whether the orientation gizmo is visible
    pub gizmo_visible: bool,
    
    // Split ring state
    /// Whether the split rings are visible
    pub split_rings_visible: bool,
    
    // Radial menu state (GPU scene only)
    /// Radial menu state for tool selection
    pub radial_menu: crate::ui::radial_menu::RadialMenuState,
    
    /// Distance from camera to dragged cell (for maintaining depth during drag)
    pub drag_distance: f32,
    
    // Cell type visuals state
    /// Visual settings per cell type (indexed by CellType)
    pub cell_type_visuals: Vec<crate::cell::types::CellTypeVisuals>,
    /// Currently selected cell type index for the visuals panel
    pub selected_cell_type: usize,
    /// Mode graph state for node visualization
    pub mode_graph_state: crate::genome::node_graph::ModeGraphState,
    /// Request to toggle the mode graph panel
    pub toggle_mode_graph_panel: bool,
    /// Stored location of mode graph panel when hidden (surface, node, tab indices)
    pub mode_graph_panel_location: Option<(egui_dock::SurfaceIndex, egui_dock::NodeIndex, egui_dock::TabIndex)>,
}

impl GenomeEditorState {
    /// Create a new genome editor state with default values.
    pub fn new() -> Self {
        let (cave_density, cave_scale, cave_octaves, cave_persistence, cave_threshold, 
             cave_smoothness, cave_seed, cave_resolution, cave_collision_enabled, cave_collision_stiffness, 
             cave_collision_damping, cave_substeps) = Self::load_cave_settings();
        
        Self {
            renaming_mode: None,
            rename_buffer: String::new(),
            selected_mode_index: 0,
            copy_into_dialog_open: false,
            copy_into_source: 0,
            color_picker_state: None,
            qball_snapping: true,
            qball1_locked_axis: -1,
            qball1_initial_distance: 1.0,
            qball2_locked_axis: -1,
            qball2_initial_distance: 1.0,
            child_a_orientation: glam::Quat::IDENTITY,
            child_b_orientation: glam::Quat::IDENTITY,
            child_a_x_axis_lat: 0.0,
            child_a_x_axis_lon: 0.0,
            child_a_y_axis_lat: 0.0,
            child_a_y_axis_lon: 0.0,
            child_a_z_axis_lat: 0.0,
            child_a_z_axis_lon: 0.0,
            child_b_x_axis_lat: 0.0,
            child_b_x_axis_lon: 0.0,
            child_b_y_axis_lat: 0.0,
            child_b_y_axis_lon: 0.0,
            child_b_z_axis_lat: 0.0,
            child_b_z_axis_lon: 0.0,
            child_a_keep_adhesion: false,
            child_b_keep_adhesion: false,
            enable_snapping: true,
            time_value: 0.0,
            max_preview_duration: 60.0,
            time_slider_dragging: false,
            cave_density,
            cave_scale,
            cave_octaves,
            cave_persistence,
            cave_threshold,
            cave_smoothness,
            cave_seed,
            cave_resolution,
            cave_collision_enabled,
            cave_collision_stiffness,
            cave_collision_damping,
            cave_substeps,
            cave_params_dirty: false,
            gizmo_visible: true,
            split_rings_visible: true,
            radial_menu: crate::ui::radial_menu::RadialMenuState::new(),
            drag_distance: 0.0,
            cell_type_visuals: crate::cell::types::CellTypeVisualsStore::load(),
            selected_cell_type: 0,
            mode_graph_state: crate::genome::node_graph::ModeGraphState::new(),
            toggle_mode_graph_panel: false,
            mode_graph_panel_location: None,
        }
    }

    /// Save cell type visuals to disk.
    pub fn save_cell_type_visuals(&self) {
        if let Err(e) = crate::cell::types::CellTypeVisualsStore::save(&self.cell_type_visuals) {
            log::error!("Failed to save cell type visuals: {}", e);
        }
    }
    
    /// Save cave settings to disk.
    pub fn save_cave_settings(&self) {
        if let Err(e) = Self::save_cave_settings_to_file(
            self.cave_density,
            self.cave_scale,
            self.cave_octaves,
            self.cave_persistence,
            self.cave_threshold,
            self.cave_smoothness,
            self.cave_seed,
            self.cave_resolution,
            self.cave_collision_enabled,
            self.cave_collision_stiffness,
            self.cave_collision_damping,
            self.cave_substeps,
        ) {
            log::error!("Failed to save cave settings: {}", e);
        }
    }
    
    fn save_cave_settings_to_file(
        density: f32,
        scale: f32,
        octaves: u32,
        persistence: f32,
        threshold: f32,
        smoothness: f32,
        seed: u32,
        resolution: u32,
        collision_enabled: bool,
        collision_stiffness: f32,
        collision_damping: f32,
        substeps: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct CaveSettings {
            density: f32,
            scale: f32,
            octaves: u32,
            persistence: f32,
            threshold: f32,
            smoothness: f32,
            seed: u32,
            resolution: u32,
            collision_enabled: bool,
            collision_stiffness: f32,
            collision_damping: f32,
            substeps: u32,
        }
        
        let settings = CaveSettings {
            density,
            scale,
            octaves,
            persistence,
            threshold,
            smoothness,
            seed,
            resolution,
            collision_enabled,
            collision_stiffness,
            collision_damping,
            substeps,
        };
        
        let path = PathBuf::from("cave_settings.ron");
        let contents = ron::ser::to_string_pretty(&settings, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Load cave settings from disk, or return defaults if file doesn't exist.
    pub fn load_cave_settings() -> (f32, f32, u32, f32, f32, f32, u32, u32, bool, f32, f32, u32) {
        #[derive(serde::Deserialize)]
        struct CaveSettings {
            density: f32,
            scale: f32,
            octaves: u32,
            persistence: f32,
            threshold: f32,
            smoothness: f32,
            seed: u32,
            #[serde(default = "default_resolution")]
            resolution: u32,
            collision_enabled: bool,
            collision_stiffness: f32,
            collision_damping: f32,
            substeps: u32,
        }
        
        fn default_resolution() -> u32 { 64 }
        
        let path = PathBuf::from("cave_settings.ron");
        
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                match ron::from_str::<CaveSettings>(&contents) {
                    Ok(settings) => {
                        return (
                                settings.density,
                                settings.scale,
                                settings.octaves,
                                settings.persistence,
                                settings.threshold,
                                settings.smoothness,
                                settings.seed,
                                settings.resolution,
                                settings.collision_enabled,
                                settings.collision_stiffness,
                                settings.collision_damping,
                                settings.substeps,
                            );
                        }
                        Err(e) => {
                            log::warn!("Failed to parse cave settings: {}. Using defaults.", e);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read cave settings: {}. Using defaults.", e);
                }
            }
        }
        
        // Return defaults
        (1.0, 10.0, 4u32, 0.5, 0.5, 0.1, 12345u32, 64u32, true, 1000.0, 0.5, 4u32)
    }
}

/// Context passed to panel renderers.
///
/// This struct provides panels with access to all the data they need
/// to render their UI and interact with the simulation. It uses references
/// to avoid copying large data structures.
pub struct PanelContext<'a> {
    /// Current genome being edited (mutable for genome editor panels)
    pub genome: &'a mut Genome,
    /// Genome editor UI state
    pub editor_state: &'a mut GenomeEditorState,
    /// Scene manager for accessing simulation state
    pub scene_manager: &'a SceneManager,
    /// Camera controller (mutable for camera settings panel)
    pub camera: &'a mut CameraController,
    /// Request for scene mode changes
    pub scene_request: &'a mut SceneModeRequest,
    /// Current simulation mode
    pub current_mode: SimulationMode,
    /// Performance metrics
    pub performance: &'a PerformanceMetrics,
}

impl<'a> PanelContext<'a> {
    /// Create a new panel context.
    pub fn new(
        genome: &'a mut Genome,
        editor_state: &'a mut GenomeEditorState,
        scene_manager: &'a SceneManager,
        camera: &'a mut CameraController,
        scene_request: &'a mut SceneModeRequest,
        current_mode: SimulationMode,
        performance: &'a PerformanceMetrics,
    ) -> Self {
        Self {
            genome,
            editor_state,
            scene_manager,
            camera,
            scene_request,
            current_mode,
            performance,
        }
    }

    /// Check if we're in preview mode.
    pub fn is_preview_mode(&self) -> bool {
        self.current_mode == SimulationMode::Preview
    }

    /// Check if we're in GPU mode.
    pub fn is_gpu_mode(&self) -> bool {
        self.current_mode == SimulationMode::Gpu
    }

    /// Request a switch to preview mode.
    pub fn request_preview_mode(&mut self) {
        *self.scene_request = SceneModeRequest::SwitchToPreview;
    }

    /// Request a switch to GPU mode.
    pub fn request_gpu_mode(&mut self) {
        *self.scene_request = SceneModeRequest::SwitchToGpu;
    }

    /// Request to toggle pause state.
    pub fn request_toggle_pause(&mut self) {
        *self.scene_request = SceneModeRequest::TogglePause;
    }

    /// Request to reset the simulation.
    pub fn request_reset(&mut self) {
        *self.scene_request = SceneModeRequest::Reset;
    }

    /// Request to toggle fast forward mode.
    pub fn request_toggle_fast_forward(&mut self) {
        *self.scene_request = SceneModeRequest::ToggleFastForward;
    }

    /// Check if fast forward mode is enabled.
    pub fn is_fast_forward(&self) -> bool {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.time_scale > 1.5
        } else {
            false
        }
    }
    
    /// Get the current simulation speed (time_scale).
    pub fn simulation_speed(&self) -> f32 {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.time_scale
        } else {
            1.0
        }
    }
    
    /// Set the simulation speed (time_scale).
    pub fn set_simulation_speed(&mut self, speed: f32) {
        *self.scene_request = SceneModeRequest::SetSpeed(speed);
    }

    /// Get the current cell count from the active scene.
    pub fn cell_count(&self) -> usize {
        self.scene_manager.active_scene().cell_count()
    }
    
    /// Get the GPU cell count buffer (GPU-only, no CPU readback).
    /// Returns None if not in GPU mode.
    /// Note: This returns the canonical state cell count as the GPU scene
    /// now uses GPU cell count buffer exclusively without async readback.
    pub fn gpu_cell_count(&self) -> Option<u32> {
        self.scene_manager.gpu_scene().map(|s| s.current_cell_count)
    }
    
    /// Get the GPU scene capacity.
    /// Returns None if not in GPU mode.
    pub fn gpu_capacity(&self) -> Option<u32> {
        self.scene_manager.gpu_scene().map(|s| s.capacity())
    }

    /// Get the current simulation time from the active scene.
    pub fn current_time(&self) -> f32 {
        self.scene_manager.active_scene().current_time()
    }

    /// Check if the simulation is paused.
    pub fn is_paused(&self) -> bool {
        self.scene_manager.active_scene().is_paused()
    }
    
    /// Check if GPU cell extraction is currently in progress
    pub fn is_gpu_cell_extraction_in_progress(&self) -> bool {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.is_extracting_cell_data()
        } else {
            false
        }
    }
    
    /// Get the latest GPU cell extraction result
    pub fn get_latest_gpu_cell_extraction(&self) -> Option<&crate::simulation::gpu_physics::ReadbackResult> {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.get_latest_cell_extraction()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_mode_request_default() {
        let request = SceneModeRequest::default();
        assert_eq!(request, SceneModeRequest::None);
        assert!(!request.is_requested());
        assert!(request.target_mode().is_none());
    }

    #[test]
    fn test_scene_mode_request_preview() {
        let request = SceneModeRequest::SwitchToPreview;
        assert!(request.is_requested());
        assert_eq!(request.target_mode(), Some(SimulationMode::Preview));
    }

    #[test]
    fn test_scene_mode_request_gpu() {
        let request = SceneModeRequest::SwitchToGpu;
        assert!(request.is_requested());
        assert_eq!(request.target_mode(), Some(SimulationMode::Gpu));
    }

    #[test]
    fn test_scene_mode_request_clear() {
        let mut request = SceneModeRequest::SwitchToGpu;
        request.clear();
        assert_eq!(request, SceneModeRequest::None);
    }

    #[test]
    fn test_genome_editor_state_default() {
        let state = GenomeEditorState::new();
        assert!(state.renaming_mode.is_none());
        assert!(state.rename_buffer.is_empty());
        assert!(!state.copy_into_dialog_open);
        assert!(state.qball_snapping);
        assert!(state.enable_snapping);
        assert_eq!(state.qball1_locked_axis, -1);
        assert_eq!(state.time_value, 0.0);
    }
}
