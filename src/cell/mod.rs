pub mod adhesion;
pub mod adhesion_forces;
pub mod adhesion_manager;
pub mod adhesion_zones;
pub mod behaviors;
pub mod division;
pub mod type_registry;
pub mod types;

// Re-export adhesion types
pub use adhesion::{
    init_adhesion_indices, AdhesionConnections, AdhesionIndices, ANCHOR_OVERLAP_COS,
    MAX_ADHESIONS_PER_CELL, MAX_ADHESION_CONNECTIONS,
};
pub use adhesion_forces::{
    compute_adhesion_forces, compute_adhesion_forces_batched, compute_adhesion_forces_parallel,
    compute_adhesion_substep,
};
pub use adhesion_manager::AdhesionConnectionManager;
pub use adhesion_zones::{
    classify_bond_direction, compute_equatorial_degrees, compute_ratio_shift, get_zone_color,
    AdhesionZone, EQUATORIAL_THRESHOLD_DEGREES, EQUATORIAL_THRESHOLD_DEGREES_MAX,
    EQUATORIAL_THRESHOLD_DEGREES_MIN,
};

// Re-export cell type registry
pub use type_registry::CellTypeRegistry;
pub use types::CellType;

// Re-export behavior types
pub use behaviors::{create_behavior, CellBehavior, TypeSpecificInstanceData};
