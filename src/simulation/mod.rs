//! Simulation module - GPU physics pipeline
//! 
//! Contains the core simulation state and GPU compute physics.

pub mod adhesion_inheritance;
pub mod canonical_state;
pub mod fluid_simulation;
pub mod gpu_physics;
pub mod physics_config;
pub mod preview_physics;
pub mod spatial_grid;

pub use adhesion_inheritance::inherit_adhesions_on_division;
pub use canonical_state::CanonicalState;
pub use physics_config::PhysicsConfig;
pub use spatial_grid::DeterministicSpatialGrid;
pub use gpu_physics::*;