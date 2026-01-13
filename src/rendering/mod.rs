//! # Rendering Pipeline - GPU-Accelerated 3D Visualization
//! 
//! This module implements the complete rendering pipeline for Bio-Spheres using wgpu.
//! The system is designed for high performance with large numbers of cells (10k+) using
//! GPU instancing, culling, and modern rendering techniques.
//! 
//! ## Rendering Architecture
//! 
//! The rendering system follows a modular design where each visual element has its own
//! renderer with dedicated pipelines and resources:
//! 
//! ```
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Cell Renderer │    │ Adhesion Lines  │    │    Skybox       │
//! │   (instanced)   │    │   (line strips) │    │  (cube map)     │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          ▼                       ▼                       ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Render Pass Coordinator                     │
//! │  1. Clear depth/color  4. Adhesion lines  7. Debug overlays    │
//! │  2. Skybox (optional)  5. Split rings     8. UI (egui)         │
//! │  3. Cells (instanced)  6. Gizmos                               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//! 
//! ## Key Design Principles
//! 
//! ### GPU Instancing
//! All cells are rendered in a single draw call using GPU instancing:
//! - **Shared Geometry**: One quad (6 vertices) used for all cells
//! - **Instance Data**: Per-cell position, radius, color uploaded to GPU
//! - **Performance**: Scales to 100k+ cells with minimal CPU overhead
//! 
//! ### Culling System
//! Multiple levels of culling reduce GPU workload:
//! - **Frustum Culling**: Remove cells outside camera view
//! - **Occlusion Culling**: Remove cells hidden behind others (Hi-Z)
//! - **Distance Culling**: Remove cells too far to be visible
//! 
//! ### Transparency Handling
//! Uses Weighted Blended Order-Independent Transparency (WBOIT):
//! - **Problem**: Traditional alpha blending requires depth sorting
//! - **Solution**: WBOIT approximates correct blending without sorting
//! - **Benefits**: Handles overlapping transparent cells efficiently
//! 
//! ## Render Pipeline Organization
//! 
//! ### Bind Group Layout
//! All pipelines share a common bind group structure:
//! - **Group 0**: Camera data (view/projection matrices, position)
//! - **Group 1+**: Pipeline-specific data (textures, parameters)
//! 
//! ### Render Pass Order
//! 1. **Clear**: Depth buffer to 1.0, color to background
//! 2. **Skybox**: Rendered at infinite depth (depth = 1.0)
//! 3. **Cells**: Instanced rendering with transparency
//! 4. **Adhesion Lines**: Cell-cell connections
//! 5. **Split Rings**: Division timing visualization
//! 6. **Gizmos**: Orientation and debug overlays
//! 7. **UI**: egui immediate mode interface
//! 
//! ## Performance Optimization
//! 
//! ### Instance Building
//! - **CPU Culling**: Remove invisible cells before GPU upload
//! - **Batch Updates**: Upload all instance data in single buffer write
//! - **Memory Pool**: Reuse instance buffers to avoid allocations
//! 
//! ### State Management
//! - **Pipeline Caching**: Pipelines created once, reused every frame
//! - **Bind Group Sharing**: Camera bind group shared across pipelines
//! - **Resource Pooling**: Textures and buffers reused when possible
//! 
//! ### GPU Memory Layout
//! - **Vertex Buffers**: Interleaved vertex data for cache efficiency
//! - **Instance Buffers**: Structure-of-Arrays for GPU-friendly access
//! - **Uniform Buffers**: 256-byte aligned for optimal GPU performance
//! 
//! ## Shader Architecture
//! 
//! ### Vertex Shaders
//! - Transform vertices from model to clip space
//! - Apply per-instance transformations (position, scale, rotation)
//! - Calculate lighting parameters and pass to fragment shader
//! 
//! ### Fragment Shaders
//! - Implement physically-based shading for realistic appearance
//! - Handle transparency using WBOIT accumulation
//! - Apply procedural textures and noise for organic look
//! 
//! ## Module Organization
//! 
//! - [`cells`] - Main cell rendering with instancing and transparency
//! - [`adhesion_lines`] - Visualization of cell-cell connections
//! - [`skybox`] - Environment background rendering
//! - [`volumetric_fog`] - Atmospheric effects
//! - [`instance_builder`] - CPU-side culling and instance data preparation
//! - [`hiz_generator`] - Hierarchical Z-buffer for occlusion culling
//! - [`debug`] - Debug visualization tools
//! 
//! ## Usage Example
//! 
//! ```rust
//! // Create renderers (typically done once at startup)
//! let cell_renderer = CellRenderer::new(&device, surface_format, &camera_layout);
//! let line_renderer = AdhesionLineRenderer::new(&device, surface_format, &camera_layout);
//! 
//! // Render frame
//! let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
//! 
//! // Render cells with instancing
//! cell_renderer.render(&mut render_pass, &camera_bind_group, &instance_buffer, instance_count);
//! 
//! // Render adhesion lines
//! line_renderer.render(&mut render_pass, &camera_bind_group, &line_data);
//! ```
//! 
//! ## Adding New Visual Effects
//! 
//! 1. **Create Renderer**: Implement new renderer struct with pipeline and resources
//! 2. **Add Shader**: Create vertex/fragment shaders in `shaders/` directory
//! 3. **Integrate**: Add render calls to appropriate scene render method
//! 4. **Optimize**: Consider culling, instancing, and GPU memory usage

pub mod adhesion_lines;
pub mod boundary_crossing;
pub mod cell_texture_atlas;
pub mod cells;
pub mod debug;
pub mod gpu_adhesion_lines;
pub mod hiz_generator;
pub mod instance_builder;
pub mod orientation_gizmo;
pub mod skybox;
pub mod split_rings;
pub mod tail_renderer;
pub mod volumetric_fog;
pub mod world_sphere;

pub use adhesion_lines::AdhesionLineRenderer;
pub use cell_texture_atlas::CellTextureAtlas;
pub use cells::CellRenderer;
pub use gpu_adhesion_lines::GpuAdhesionLineRenderer;
pub use hiz_generator::HizGenerator;
pub use instance_builder::{CellInstance, CullingMode, CullingStats, InstanceBuilder};
pub use orientation_gizmo::OrientationGizmoRenderer;
pub use split_rings::SplitRingRenderer;
pub use tail_renderer::{TailRenderer, TailInstance};
pub use world_sphere::{WorldSphereRenderer, WorldSphereParams};
