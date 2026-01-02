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
    pub color: (u8, u8, u8), // Mode color as RGB tuple
    pub parent_split_direction: Vec2, // pitch (x), yaw (y) in degrees
    // TODO: Add more mode fields as needed
}

impl Hash for ModeSettings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.color.hash(state); // Hash RGB tuple
        self.parent_split_direction.x.to_bits().hash(state);
        self.parent_split_direction.y.to_bits().hash(state);
    }
}

impl Default for ModeSettings {
    fn default() -> Self {
        Self {
            name: "M1".to_string(), // Default to M1 naming convention
            color: (100, 150, 200), // Default blue color
            parent_split_direction: Vec2::ZERO,
        }
    }
}

impl Default for Genome {
    fn default() -> Self {
        // Create 40 default modes like in the reference implementation
        let mut modes = Vec::new();
        
        for i in 0..40 {
            // Generate colors with variation across the spectrum
            let hue = (i as f32 * 360.0 / 40.0) % 360.0; // Distribute hues evenly
            let saturation = 0.7 + (i % 3) as f32 * 0.1; // Vary saturation slightly
            let value = 0.8 + (i % 2) as f32 * 0.1; // Vary brightness slightly
            
            // Convert HSV to RGB
            let c = value * saturation;
            let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
            let m = value - c;
            
            let (r_prime, g_prime, b_prime) = if hue < 60.0 {
                (c, x, 0.0)
            } else if hue < 120.0 {
                (x, c, 0.0)
            } else if hue < 180.0 {
                (0.0, c, x)
            } else if hue < 240.0 {
                (0.0, x, c)
            } else if hue < 300.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };
            
            let r = ((r_prime + m) * 255.0) as u8;
            let g = ((g_prime + m) * 255.0) as u8;
            let b = ((b_prime + m) * 255.0) as u8;
            
            modes.push(ModeSettings {
                name: format!("M{}", i + 1), // M1, M2, M3, etc.
                color: (r, g, b),
                parent_split_direction: Vec2::ZERO,
            });
        }
        
        Self {
            name: "Default".to_string(),
            initial_mode: 0, // First mode (Mode 1)
            modes,
        }
    }
}
