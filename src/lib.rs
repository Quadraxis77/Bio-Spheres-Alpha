//! # Bio-Spheres: GPU-Accelerated Biological Cell Simulation
//! 
//! Bio-Spheres is a real-time biological cell simulation that combines GPU-accelerated physics
//! with interactive genome editing. The simulation models cell division, adhesion, and movement
//! in 3D space with realistic biological behaviors.
//! 
//! ## Architecture Overview
//! 
//! The codebase is organized into four main subsystems:
//! 
//! ### 1. Simulation Engine ([`simulation`])
//! 
//! The core physics and state management:
//! - [`simulation::CanonicalState`] - Structure-of-Arrays (SoA) layout for all cell data
//! - [`simulation::preview_physics`] - CPU-based physics for preview scene with time scrubbing
//! - [`simulation::gpu_physics`] - GPU compute shader physics for large-scale simulations
//! - [`simulation::PhysicsConfig`] - Configurable physics parameters
//! 
//! **Key Design**: Uses SoA layout for cache-friendly iteration over large cell populations.
//! 
//! ### 2. Rendering Pipeline ([`rendering`])
//! 
//! GPU-accelerated 3D visualization using wgpu:
//! - [`rendering::cells`] - Instanced cell rendering with transparency
//! - [`rendering::adhesion_lines`] - Visualization of cell-cell connections
//! - [`rendering::skybox`] - Environment background
//! - [`rendering::volumetric_fog`] - Atmospheric effects
//! 
//! **Key Design**: Uses GPU instancing to render thousands of cells efficiently.
//! 
//! ### 3. Genome System ([`genome`])
//! 
//! Node-based genome representation and editing:
//! - [`genome::Genome`] - Mode-based cell behavior definition
//! - [`genome::node_graph`] - Visual genome editor using egui-snarl
//! 
//! **Key Design**: Genomes define cell behavior through modes and transitions.
//! 
//! ### 4. User Interface ([`ui`])
//! 
//! egui-based interface with docking panels:
//! - [`ui::camera`] - 6DOF camera controller (Space Engineers style)
//! - [`ui::dock`] - Dockable panel system
//! - [`ui::settings`] - Physics and rendering configuration
//! 
//! ## Application Entry Points
//! 
//! - [`app::App`] - Main application struct with wgpu setup and event loop
//! - [`scene::PreviewScene`] - CPU physics mode for genome editing
//! - [`scene::GpuScene`] - GPU compute mode for large simulations
//! - [`scene::Manager`] - Coordinates switching between simulation modes
//! 
//! ## Key Data Structures
//! 
//! ### Central State
//! - [`simulation::CanonicalState`] - All cell data in Structure-of-Arrays layout
//! - [`cell::adhesion::AdhesionData`] - Cell-cell connection data
//! - [`genome::Genome`] - Cell behavior definition
//! 
//! ### Rendering
//! - [`rendering::cells::CellRenderer`] - GPU instanced cell rendering
//! - [`ui::camera::CameraController`] - 6DOF camera with spring smoothing
//! 
//! ## Simulation Modes
//! 
//! ### Preview Mode (CPU Physics)
//! - **Use Case**: Genome editing, small simulations (<1000 cells)
//! - **Physics**: CPU-based with deterministic spatial grid
//! - **Features**: Time scrubbing, checkpoints, real-time genome editing
//! 
//! ### GPU Mode (GPU Compute)
//! - **Use Case**: Large-scale simulations (10k+ cells)
//! - **Physics**: GPU compute shaders with parallel processing
//! - **Features**: Frustum culling, Hi-Z occlusion culling, massive scale
//! 
//! ## Performance Characteristics
//! 
//! - **Structure-of-Arrays**: Cache-friendly iteration over cell properties
//! - **GPU Instancing**: Single draw call for all cells
//! - **Spatial Partitioning**: O(n) collision detection with grid-based culling
//! - **Double Buffering**: GPU compute without CPU-GPU synchronization stalls
//! 
//! ## Getting Started
//! 
//! 1. **Understanding the Data Flow**:
//!    ```
//!    Input Events → Camera Update → Physics Step → State Update → Render
//!    ```
//! 
//! 2. **Key Files to Read First**:
//!    - `src/app.rs` - Application setup and event loop
//!    - `src/simulation/canonical_state.rs` - Central data structure
//!    - `src/scene/preview_scene.rs` - CPU simulation mode
//!    - `src/rendering/cells.rs` - GPU rendering pipeline
//! 
//! 3. **Adding New Features**:
//!    - **New Cell Property**: Add to `CanonicalState`, update physics, expose in UI
//!    - **New Render Effect**: Create shader, add renderer, integrate into render pass
//!    - **New Physics Force**: Implement in CPU physics, mirror in GPU compute
//! 
//! ## Dependencies
//! 
//! - **Graphics**: `wgpu` (GPU abstraction), `winit` (windowing)
//! - **Math**: `glam` (SIMD math types), `bytemuck` (safe transmutation)
//! - **UI**: `egui` (immediate mode GUI), `egui-snarl` (node graphs)
//! - **Concurrency**: `rayon` (parallel iteration), `crossbeam-channel` (messaging)
//! - **Serialization**: `serde` + `ron` (human-readable config files)

pub mod app;
pub mod cell;
pub mod genome;
pub mod input;
pub mod rendering;
pub mod scene;
pub mod simulation;
pub mod ui;