pub mod node_graph;

use std::hash::{Hash, Hasher};
use glam::Vec2;

#[derive(Debug, Clone, Hash)]
pub struct Genome {
    pub name: String,
    pub initial_mode: i32,
    pub modes: Vec<ModeSettings>,
}

#[derive(Debug, Clone)]
pub struct ModeSettings {
    pub name: String,
    pub parent_split_direction: Vec2, // pitch (x), yaw (y) in degrees
    // TODO: Add more mode fields as needed
}

impl Hash for ModeSettings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.parent_split_direction.x.to_bits().hash(state);
        self.parent_split_direction.y.to_bits().hash(state);
    }
}

impl Default for ModeSettings {
    fn default() -> Self {
        Self {
            name: "Default Mode".to_string(),
            parent_split_direction: Vec2::ZERO,
        }
    }
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            initial_mode: 0,
            modes: vec![ModeSettings::default()],
        }
    }
}
