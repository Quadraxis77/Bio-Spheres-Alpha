pub mod camera;
pub mod dock;
pub mod panel;
pub mod panel_context;
pub mod performance;
pub mod radial_menu;
pub mod settings;
pub mod tab_viewer;
pub mod types;
pub mod ui_system;
pub mod widgets;

// Re-export commonly used types
pub use dock::DockManager;
pub use panel::Panel;
pub use panel_context::{GenomeEditorState, PanelContext, SceneModeRequest};
pub use performance::PerformanceMetrics;
pub use radial_menu::{RadialMenuState, RadialTool};
pub use tab_viewer::PanelTabViewer;
pub use types::{GlobalUiState, SimulationMode};
pub use ui_system::UiSystem;
