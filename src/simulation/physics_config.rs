use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub fixed_timestep: f32,
    pub gravity: f32,
    pub damping: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            fixed_timestep: 0.016,
            gravity: 0.0,
            damping: 0.99,
        }
    }
}
