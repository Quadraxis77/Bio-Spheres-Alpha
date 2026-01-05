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

/// Request for scene mode changes.
///
/// Panels can request a scene mode change by setting this value.
/// The main application loop will process the request and switch scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
            max_preview_duration: 60.0, // 60 second preview range
            time_slider_dragging: false,
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

    /// Get the current cell count from the active scene.
    pub fn cell_count(&self) -> usize {
        self.scene_manager.active_scene().cell_count()
    }
    
    /// Get the GPU cell count (async, may be 1-2 frames behind).
    /// Returns None if not in GPU mode.
    pub fn gpu_cell_count(&self) -> Option<u32> {
        self.scene_manager.gpu_scene().map(|s| s.gpu_cell_count())
    }

    /// Get the current simulation time from the active scene.
    pub fn current_time(&self) -> f32 {
        self.scene_manager.active_scene().current_time()
    }

    /// Check if the simulation is paused.
    pub fn is_paused(&self) -> bool {
        self.scene_manager.active_scene().is_paused()
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
