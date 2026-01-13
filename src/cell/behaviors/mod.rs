//! Cell Behavior Module
//!
//! This module defines the [`CellBehavior`] trait and type-specific behavior implementations.
//! Each cell type has a corresponding behavior module that implements simulation logic
//! and provides type-specific instance data for rendering.
//!
//! # Architecture
//!
//! The behavior system follows a trait-based design pattern:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     CellBehavior Trait                      │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │  build_instance_data(&self, mode: &ModeSettings)    │   │
//! │  │  -> TypeSpecificInstanceData                        │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          ▼                   ▼                   ▼
//!   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//!   │ TestCell    │    │Flagellocyte │    │  Lipocyte   │
//!   │ Behavior    │    │  Behavior   │    │  Behavior   │
//!   └─────────────┘    └─────────────┘    └─────────────┘
//! ```
//!
//! # Adding a New Cell Type Behavior
//!
//! ## Step 1: Create Behavior File
//!
//! Create `src/cell/behaviors/{cell_type}.rs`:
//!
//! ```ignore
//! use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
//! use crate::genome::ModeSettings;
//!
//! /// Behavior implementation for {CellType} cells.
//! pub struct {CellType}Behavior;
//!
//! impl CellBehavior for {CellType}Behavior {
//!     fn build_instance_data(&self, mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
//!         // Build type-specific data for rendering
//!         TypeSpecificInstanceData::empty()
//!     }
//! }
//! ```
//!
//! ## Step 2: Register in mod.rs
//!
//! Add to this file:
//!
//! ```ignore
//! pub mod {cell_type};
//!
//! pub fn create_behavior(cell_type: CellType) -> Box<dyn CellBehavior> {
//!     match cell_type {
//!         // ... existing types ...
//!         CellType::{CellType} => Box::new({cell_type}::{CellType}Behavior),
//!     }
//! }
//! ```
//!
//! ## Step 3: Update TypeSpecificInstanceData (if needed)
//!
//! If your cell type needs custom rendering data, add a constructor:
//!
//! ```ignore
//! impl TypeSpecificInstanceData {
//!     pub fn {cell_type}(param1: f32, param2: f32) -> Self {
//!         let mut data = [0.0f32; 8];
//!         data[0] = param1;
//!         data[1] = param2;
//!         Self { data }
//!     }
//! }
//! ```
//!
//! # Thread Safety
//!
//! All behavior implementations must be `Send + Sync` to support parallel physics
//! processing. The trait bound enforces this at compile time.

pub mod flagellocyte;
pub mod test_cell;

use crate::genome::ModeSettings;

/// Type-specific instance data returned by behavior modules.
///
/// Each cell type can define how it uses the 8 reserved floats in the
/// instance data. This struct provides a type-safe way to build that data.
///
/// # Layout
///
/// The 8 floats are reserved for type-specific use. Each cell type defines
/// its own layout:
///
/// ## Flagellocyte Layout
/// - `data[0]`: tail_length - Length of the flagellum (0.5 - 3.0)
/// - `data[1]`: tail_thickness - Thickness at the base (0.01 - 0.3)
/// - `data[2]`: tail_amplitude - Wave amplitude (0.0 - 0.5)
/// - `data[3]`: tail_frequency - Wave frequency (0.5 - 10.0)
/// - `data[4]`: tail_speed - Wave propagation speed (0.0 - 15.0)
/// - `data[5]`: tail_taper - Taper from base to tip (0.0 - 1.0)
/// - `data[6]`: tail_segments - Number of render segments (4 - 64)
/// - `data[7]`: cell_type - Cell type for unified shader branching
///
/// ## Neurocyte Layout
/// - `data[0-1]`: reserved
/// - `data[2]`: sensor_direction_x
/// - `data[3]`: sensor_direction_y
/// - `data[4]`: sensor_direction_z
/// - `data[5-7]`: reserved
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeSpecificInstanceData {
    /// The 8 reserved floats for type-specific rendering data.
    pub data: [f32; 8],
}

impl TypeSpecificInstanceData {
    /// Create empty type-specific data with all zeros.
    ///
    /// Used by cell types that don't need type-specific rendering data,
    /// such as the Test cell type.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create type-specific data for a Flagellocyte cell.
    ///
    /// Packs all flagella visual parameters into the type_data array for
    /// shader rendering. The shader uses these values to render the animated
    /// helical tail.
    ///
    /// # Layout
    ///
    /// - `data[0]`: tail_length - Length of the flagellum (0.5 - 3.0)
    /// - `data[1]`: tail_thickness - Thickness at the base (0.01 - 0.3)
    /// - `data[2]`: tail_amplitude - Wave amplitude (0.0 - 0.5)
    /// - `data[3]`: tail_frequency - Wave frequency (0.5 - 10.0)
    /// - `data[4]`: tail_speed - Wave propagation speed (0.0 - 15.0)
    /// - `data[5]`: tail_taper - Taper from base to tip (0.0 - 1.0)
    /// - `data[6]`: debug_colors_enabled - Debug colors flag (0.0 or 1.0)
    /// - `data[7]`: cell_type - Cell type for unified shader branching
    ///
    /// # Arguments
    ///
    /// * `tail_length` - Length of the flagellum tail
    /// * `tail_thickness` - Thickness of the flagellum at the base
    /// * `tail_amplitude` - Amplitude of the helical wave motion
    /// * `tail_frequency` - Frequency of the helical wave
    /// * `tail_speed` - Speed of wave propagation along the tail
    /// * `tail_taper` - Taper factor from base to tip (1.0 = full taper)
    /// * `debug_colors_enabled` - Debug colors flag (0.0 or 1.0)
    /// * `cell_type` - Cell type index for unified shader branching
    ///
    /// # Example
    ///
    /// ```ignore
    /// let type_data = TypeSpecificInstanceData::flagellocyte(
    ///     1.7,   // tail_length
    ///     0.15,  // tail_thickness
    ///     0.17,  // tail_amplitude
    ///     1.0,   // tail_frequency
    ///     9.5,   // tail_speed
    ///     1.0,   // tail_taper
    ///     1.0,   // debug_colors_enabled
    ///     1.0,   // cell_type (Flagellocyte = 1)
    /// );
    /// ```
    pub fn flagellocyte(
        tail_length: f32,
        tail_thickness: f32,
        tail_amplitude: f32,
        tail_frequency: f32,
        tail_speed: f32,
        tail_taper: f32,
        debug_colors_enabled: f32, // Debug colors flag
        cell_type: f32,
    ) -> Self {
        Self {
            data: [
                tail_length,           // data[0]
                tail_thickness,        // data[1]
                tail_amplitude,        // data[2]
                tail_frequency,        // data[3]
                tail_speed,            // data[4]
                tail_taper,            // data[5]
                debug_colors_enabled,  // data[6] - debug colors flag
                cell_type,             // data[7] - cell_type for unified shader
            ],
        }
    }

    /// Create type-specific data for a Neurocyte cell.
    ///
    /// # Arguments
    ///
    /// * `sensor_direction` - The direction the sensor is pointing (normalized)
    #[allow(dead_code)]
    pub fn neurocyte(sensor_direction: glam::Vec3) -> Self {
        let mut data = [0.0f32; 8];
        data[2] = sensor_direction.x;
        data[3] = sensor_direction.y;
        data[4] = sensor_direction.z;
        Self { data }
    }
}

/// Trait for cell type-specific simulation behavior.
///
/// Each cell type implements this trait to define type-specific rendering data.
/// The behavior is queried during instance data building to populate the
/// type-specific fields in the GPU instance buffer.
///
/// # Design Philosophy
///
/// The `CellBehavior` trait separates cell-type-specific logic from the generic
/// rendering pipeline. This allows:
///
/// - **Easy extension**: Add new cell types without modifying the renderer
/// - **Type safety**: Each cell type defines its own data layout
/// - **Testability**: Behaviors can be unit tested independently
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow parallel physics processing.
/// This is enforced by the trait bounds.
///
/// # Instance Data Flow
///
/// ```text
/// ModeSettings ──► CellBehavior::build_instance_data()
///                          │
///                          ▼
///              TypeSpecificInstanceData
///                          │
///                          ▼
///              CellInstance.type_data[0..8]
///                          │
///                          ▼
///              GPU Instance Buffer
///                          │
///                          ▼
///              Appearance Shader (type_data_0, type_data_1)
/// ```
///
/// # Example Implementation
///
/// ```ignore
/// use crate::cell::behaviors::{CellBehavior, TypeSpecificInstanceData};
/// use crate::genome::ModeSettings;
///
/// /// Behavior for Flagellocyte cells with animated flagella.
/// pub struct FlagellocyteBehavior;
///
/// impl CellBehavior for FlagellocyteBehavior {
///     fn build_instance_data(&self, mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
///         // Extract swim_force from mode settings to control flagella animation
///         let flagella_speed = mode_settings.swim_force;
///         let flagella_angle = 0.0; // Would be updated by physics simulation
///         
///         TypeSpecificInstanceData::flagellocyte(flagella_angle, flagella_speed)
///     }
/// }
/// ```
///
/// # Shader Integration
///
/// The `type_data` array is passed to the appearance shader as two vec4 attributes:
///
/// ```wgsl
/// struct VertexInput {
///     // ... common fields ...
///     @location(5) type_data_0: vec4<f32>,  // data[0..4]
///     @location(6) type_data_1: vec4<f32>,  // data[4..8]
/// }
/// ```
///
/// Access individual values in the shader:
///
/// ```wgsl
/// let flagella_angle = input.type_data_0.x;  // data[0]
/// let flagella_speed = input.type_data_0.y;  // data[1]
/// ```
pub trait CellBehavior: Send + Sync {
    /// Build type-specific instance data fields for rendering.
    ///
    /// This method is called when building instance data for GPU rendering.
    /// It should return the type-specific data that the cell's appearance
    /// shader needs for rendering.
    ///
    /// # Arguments
    ///
    /// * `mode_settings` - The mode settings for this cell, containing
    ///   type-specific configuration values
    ///
    /// # Returns
    ///
    /// Type-specific instance data to be included in the GPU instance buffer.
    fn build_instance_data(&self, mode_settings: &ModeSettings) -> TypeSpecificInstanceData;
}

/// Create a behavior instance for the given cell type.
///
/// This function is used by the `CellTypeRegistry` to create behavior
/// instances for each registered cell type.
///
/// # Arguments
///
/// * `cell_type` - The cell type to create a behavior for
///
/// # Returns
///
/// A boxed behavior instance implementing `CellBehavior`.
pub fn create_behavior(cell_type: crate::cell::CellType) -> Box<dyn CellBehavior> {
    match cell_type {
        crate::cell::CellType::Test => Box::new(test_cell::TestCellBehavior),
        crate::cell::CellType::Flagellocyte => Box::new(flagellocyte::FlagellocyteBehavior),
    }
}
