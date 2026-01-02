pub mod adhesion;
pub mod adhesion_forces;
pub mod adhesion_manager;
pub mod adhesion_zones;
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
    compute_adhesion_forces, 
    compute_adhesion_forces_parallel, 
    compute_adhesion_forces_batched,
};
pub use adhesion_manager::AdhesionConnectionManager;
pub use adhesion_zones::{
    AdhesionZone, 
    classify_bond_direction, 
    get_zone_color, 
    EQUATORIAL_THRESHOLD_DEGREES,
};
