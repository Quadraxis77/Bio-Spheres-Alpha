pub mod adhesion_inheritance;
pub mod canonical_state;
pub mod cell_allocation;
pub mod clock;
pub mod cpu_physics;
pub mod double_buffer;
pub mod gpu_physics;
pub mod initial_state;
pub mod nutrient_system;
pub mod physics_config;

pub use adhesion_inheritance::inherit_adhesions_on_division;
pub use canonical_state::CanonicalState;
pub use physics_config::PhysicsConfig;
