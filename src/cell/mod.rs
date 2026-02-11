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
    AdhesionConnections, 
    AdhesionIndices, 
    MAX_ADHESIONS_PER_CELL, 
    MAX_ADHESION_CONNECTIONS,
    init_adhesion_indices,
};
pub use adhesion_forces::{
    solve_adhesion_pbd,
    solve_adhesion_pbd_parallel,
};
pub use adhesion_manager::AdhesionConnectionManager;
pub use adhesion_zones::{
    AdhesionZone, 
    classify_bond_direction, 
    get_zone_color, 
    EQUATORIAL_THRESHOLD_DEGREES,
};

// Re-export cell type registry
pub use type_registry::CellTypeRegistry;
pub use types::CellType;

// Re-export behavior types
pub use behaviors::{CellBehavior, TypeSpecificInstanceData, create_behavior};
