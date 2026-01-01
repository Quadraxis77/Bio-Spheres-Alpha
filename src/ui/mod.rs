pub mod camera;
pub mod dock;
pub mod panel;
pub mod settings;
pub mod types;
pub mod ui_system;
pub mod widgets;

// Re-export commonly used types
pub use dock::DockManager;
pub use panel::Panel;
pub use types::{GlobalUiState, SimulationMode};
pub use ui_system::UiSystem;
