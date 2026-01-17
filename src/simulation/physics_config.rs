use serde::{Deserialize, Serialize};

/// Physics configuration for deterministic simulation
/// 
/// This configuration is shared by both CPU and GPU physics implementations.
/// All values are deterministic and produce identical results across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Fixed timestep for physics integration (64 Hz ≈ 15.6ms)
    pub fixed_timestep: f32,
    
    /// Spherical boundary radius for active simulation
    pub sphere_radius: f32,
    
    /// Default cell stiffness for collision response
    pub default_stiffness: f32,
    
    /// Collision damping coefficient (velocity-based resistance)
    pub damping: f32,
    
    /// Velocity damping coefficient (applied as pow(velocity_damping, dt * 100.0))
    pub velocity_damping: f32,
    
    /// Tangential friction coefficient for rolling contact
    pub friction_coefficient: f32,
    
    /// Angular velocity damping coefficient
    pub angular_damping: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0 / 64.0, // 64 Hz
            sphere_radius: 200.0,  // Updated for 400-unit world diameter
            default_stiffness: 0.0,
            damping: 0.0,
            velocity_damping: 0.98,
            friction_coefficient: 0.3,
            angular_damping: 0.95,
        }
    }
}
