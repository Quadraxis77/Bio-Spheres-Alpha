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

pub mod test_cell;

use crate::genome::ModeSettings;

/// Type-specific instance data returned by behavior modules.
///
/// Each cell type can define how it uses the 8 reserved floats in the
/// instance data. This struct provides a type-safe way to build that data.
///
/// # Layout
///
/// The 8 floats are reserved for type-specific use:
/// - `data[0]`: flagella_angle (Flagellocyte)
/// - `data[1]`: flagella_speed (Flagellocyte)
/// - `data[2]`: sensor_direction_x (Neurocyte)
/// - `data[3]`: sensor_direction_y (Neurocyte)
/// - `data[4]`: sensor_direction_z (Neurocyte)
/// - `data[5-7]`: reserved for future use
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
    /// # Arguments
    ///
    /// * `angle` - Current flagella rotation angle in radians
    /// * `speed` - Current flagella rotation speed
    #[allow(dead_code)]
    pub fn flagellocyte(angle: f32, speed: f32) -> Self {
        let mut data = [0.0f32; 8];
        data[0] = angle;
        data[1] = speed;
        Self { data }
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
        // Future cell types will be added here:
        // crate::cell::CellType::Flagellocyte => Box::new(flagellocyte::FlagellocyteBehavior),
        // crate::cell::CellType::Lipocyte => Box::new(lipocyte::LipocyteBehavior),
    }
}
