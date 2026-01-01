pub mod node_graph;

pub use node_graph::*;

use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Hash)]
pub struct Genome {
    pub name: String,
    pub initial_mode: i32,
    // TODO: Add more genome fields
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            initial_mode: 0,
        }
    }
}
