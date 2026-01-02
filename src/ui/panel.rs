//! Panel definitions for the Bio-Spheres docking system.
//!
//! This module defines all available panel types and their properties,
//! including which panels are available in each simulation mode.

use crate::ui::types::SimulationMode;
use serde::{Deserialize, Serialize};
use std::fmt;

/// All available panel types in the Bio-Spheres UI.
///
/// Each panel represents a dockable window that can be moved, tabbed, or floated.
/// Some panels are only available in certain simulation modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Panel {
    // Structural panels
    /// The main 3D viewport where the simulation is displayed.
    Viewport,
    /// Left placeholder panel for dock layout.
    LeftPanel,
    /// Right placeholder panel for dock layout.
    RightPanel,
    /// Bottom placeholder panel for dock layout.
    BottomPanel,

    // Dynamic windows
    /// Inspector for viewing selected cell properties.
    CellInspector,
    /// Main genome editor panel.
    GenomeEditor,
    /// Scene/mode manager for switching simulation modes.
    SceneManager,
    /// Performance monitor showing FPS and timing information.
    PerformanceMonitor,
    /// Rendering controls for visual settings (fog, bloom, etc.).
    RenderingControls,
    /// Time scrubber for simulation time control.
    TimeScrubber,
    /// Theme editor for customizing UI appearance.
    ThemeEditor,
    /// Camera settings panel.
    CameraSettings,
    /// Lighting settings panel.
    LightingSettings,
    /// Orientation gizmo settings panel.
    GizmoSettings,
    /// Cell type visuals panel for appearance settings.
    CellTypeVisuals,

    // Genome editor sub-panels
    /// Cell modes list panel.
    Modes,
    /// Name and type editor for cells.
    NameTypeEditor,
    /// Adhesion settings panel.
    AdhesionSettings,
    /// Parent settings panel with rotation controls.
    ParentSettings,
    /// Circular sliders panel for angle-based inputs.
    CircleSliders,
    /// Quaternion ball panel for 3D rotation input.
    QuaternionBall,
    /// Time slider panel for preview duration control.
    TimeSlider,
    /// Mode graph panel for visualizing mode connections.
    ModeGraph,
}

impl Panel {
    /// Check if this panel is available in the given simulation mode.
    ///
    /// Some panels are only relevant for certain modes:
    /// - Preview mode: Genome editing panels (Modes, NameTypeEditor, etc.)
    /// - GPU mode: Performance monitoring, large-scale simulation controls
    pub fn available_in_mode(&self, mode: SimulationMode) -> bool {
        match self {
            // Always available
            Panel::Viewport => true,
            Panel::LeftPanel => true,
            Panel::RightPanel => true,
            Panel::BottomPanel => true,
            Panel::SceneManager => true,
            Panel::ThemeEditor => true,
            Panel::CameraSettings => true,
            Panel::LightingSettings => true,
            Panel::GizmoSettings => true,
            Panel::RenderingControls => true,

            // Preview mode only (genome editing)
            Panel::GenomeEditor => mode == SimulationMode::Preview,
            Panel::Modes => mode == SimulationMode::Preview,
            Panel::NameTypeEditor => mode == SimulationMode::Preview,
            Panel::AdhesionSettings => mode == SimulationMode::Preview,
            Panel::ParentSettings => mode == SimulationMode::Preview,
            Panel::CircleSliders => mode == SimulationMode::Preview,
            Panel::QuaternionBall => mode == SimulationMode::Preview,
            Panel::TimeSlider => mode == SimulationMode::Preview,
            Panel::ModeGraph => mode == SimulationMode::Preview,
            Panel::TimeScrubber => mode == SimulationMode::Preview,
            Panel::CellTypeVisuals => mode == SimulationMode::Preview,

            // GPU mode primarily (but can be shown in preview)
            Panel::CellInspector => true,
            Panel::PerformanceMonitor => true,
        }
    }

    /// Get the default panels for a specific simulation mode.
    ///
    /// Returns a list of panels that should be shown by default when
    /// entering the given mode.
    pub fn defaults_for_mode(mode: SimulationMode) -> Vec<Panel> {
        match mode {
            SimulationMode::Preview => vec![
                Panel::Viewport,
                Panel::SceneManager,
                Panel::Modes,
                Panel::NameTypeEditor,
                Panel::AdhesionSettings,
                Panel::ParentSettings,
                Panel::TimeSlider,
                Panel::TimeScrubber,
                Panel::GizmoSettings,
            ],
            SimulationMode::Gpu => vec![
                Panel::Viewport,
                Panel::SceneManager,
                Panel::CellInspector,
                Panel::PerformanceMonitor,
                Panel::RenderingControls,
                Panel::CameraSettings,
                Panel::GizmoSettings,
            ],
        }
    }

    /// Get all panel variants.
    pub fn all() -> &'static [Panel] {
        &[
            Panel::Viewport,
            Panel::LeftPanel,
            Panel::RightPanel,
            Panel::BottomPanel,
            Panel::CellInspector,
            Panel::GenomeEditor,
            Panel::SceneManager,
            Panel::PerformanceMonitor,
            Panel::RenderingControls,
            Panel::TimeScrubber,
            Panel::ThemeEditor,
            Panel::CameraSettings,
            Panel::LightingSettings,
            Panel::GizmoSettings,
            Panel::CellTypeVisuals,
            Panel::Modes,
            Panel::NameTypeEditor,
            Panel::AdhesionSettings,
            Panel::ParentSettings,
            Panel::CircleSliders,
            Panel::QuaternionBall,
            Panel::TimeSlider,
            Panel::ModeGraph,
        ]
    }

    /// Get the display name for this panel.
    pub fn display_name(&self) -> &'static str {
        match self {
            Panel::Viewport => "Viewport",
            Panel::LeftPanel => "Left Panel",
            Panel::RightPanel => "Right Panel",
            Panel::BottomPanel => "Bottom Panel",
            Panel::CellInspector => "Cell Inspector",
            Panel::GenomeEditor => "Genome Editor",
            Panel::SceneManager => "Scene Manager",
            Panel::PerformanceMonitor => "Performance",
            Panel::RenderingControls => "Rendering",
            Panel::TimeScrubber => "Time Scrubber",
            Panel::ThemeEditor => "Theme Editor",
            Panel::CameraSettings => "Camera",
            Panel::LightingSettings => "Lighting",
            Panel::GizmoSettings => "Gizmo Rendering",
            Panel::CellTypeVisuals => "Cell Visuals",
            Panel::Modes => "Modes",
            Panel::NameTypeEditor => "Name & Type",
            Panel::AdhesionSettings => "Adhesion Settings",
            Panel::ParentSettings => "Parent Settings",
            Panel::CircleSliders => "Circle Sliders",
            Panel::QuaternionBall => "Child Rotation",
            Panel::TimeSlider => "Time Slider",
            Panel::ModeGraph => "Mode Graph",
        }
    }

    /// Check if this panel is a viewport (special rendering treatment).
    pub fn is_viewport(&self) -> bool {
        matches!(self, Panel::Viewport)
    }

    /// Check if this panel is a placeholder layout panel.
    pub fn is_placeholder(&self) -> bool {
        matches!(self, Panel::LeftPanel | Panel::RightPanel | Panel::BottomPanel | Panel::Viewport)
    }

    /// Check if this panel should be closeable by the user.
    ///
    /// All panels are closeable by default - locking is handled at the UI level.
    pub fn is_closeable(&self) -> bool {
        true // All panels are closeable unless locked by the UI system
    }

    /// Check if this panel can be floated into a separate window.
    pub fn allowed_in_windows(&self) -> bool {
        true // All panels can be floated
    }
}

impl fmt::Display for Panel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_display() {
        assert_eq!(Panel::Viewport.to_string(), "Viewport");
        assert_eq!(Panel::CellInspector.to_string(), "Cell Inspector");
        assert_eq!(Panel::PerformanceMonitor.to_string(), "Performance");
    }

    #[test]
    fn test_viewport_always_available() {
        assert!(Panel::Viewport.available_in_mode(SimulationMode::Preview));
        assert!(Panel::Viewport.available_in_mode(SimulationMode::Gpu));
    }

    #[test]
    fn test_genome_panels_preview_only() {
        assert!(Panel::Modes.available_in_mode(SimulationMode::Preview));
        assert!(!Panel::Modes.available_in_mode(SimulationMode::Gpu));

        assert!(Panel::NameTypeEditor.available_in_mode(SimulationMode::Preview));
        assert!(!Panel::NameTypeEditor.available_in_mode(SimulationMode::Gpu));
    }

    #[test]
    fn test_defaults_contain_viewport() {
        let preview_defaults = Panel::defaults_for_mode(SimulationMode::Preview);
        assert!(preview_defaults.contains(&Panel::Viewport));

        let gpu_defaults = Panel::defaults_for_mode(SimulationMode::Gpu);
        assert!(gpu_defaults.contains(&Panel::Viewport));
    }

    #[test]
    fn test_all_panels_are_closeable() {
        // All panels should be closeable by default - locking is handled at UI level
        assert!(Panel::Viewport.is_closeable());
        assert!(Panel::CellInspector.is_closeable());
        assert!(Panel::LeftPanel.is_closeable());
        assert!(Panel::RightPanel.is_closeable());
        assert!(Panel::BottomPanel.is_closeable());
    }
}
