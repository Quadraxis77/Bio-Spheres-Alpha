pub mod node_graph;
pub mod serialization;

use glam::{Vec2, Vec3, Quat};

pub use serialization::{GenomeDeserializeError, GenomeSerializeError};

#[derive(Debug, Clone)]
pub struct Genome {
    pub name: String,
    pub initial_mode: i32,
    pub initial_orientation: Quat,
    pub modes: Vec<ModeSettings>,
}

/// Child settings for mode transitions
#[derive(Debug, Clone)]
pub struct ChildSettings {
    pub mode_number: i32,
    pub orientation: Quat,
    pub keep_adhesion: bool,
    pub enable_angle_snapping: bool,
    // Lat/lon tracking for quaternion ball widget (UI feedback only)
    pub x_axis_lat: f32,
    pub x_axis_lon: f32,
    pub y_axis_lat: f32,
    pub y_axis_lon: f32,
    pub z_axis_lat: f32,
    pub z_axis_lon: f32,
}

impl Default for ChildSettings {
    fn default() -> Self {
        Self {
            mode_number: 0,
            orientation: Quat::IDENTITY,
            keep_adhesion: true,
            enable_angle_snapping: true,
            x_axis_lat: 0.0,
            x_axis_lon: 0.0,
            y_axis_lat: 0.0,
            y_axis_lon: 0.0,
            z_axis_lat: 0.0,
            z_axis_lon: 0.0,
        }
    }
}

/// Adhesion configuration for cell connections
#[derive(Debug, Clone, PartialEq)]
pub struct AdhesionSettings {
    pub can_break: bool,
    pub break_force: f32,
    pub rest_length: f32,
    pub linear_spring_stiffness: f32,
    pub linear_spring_damping: f32,
    pub orientation_spring_stiffness: f32,
    pub orientation_spring_damping: f32,
    pub max_angular_deviation: f32,
    pub twist_constraint_stiffness: f32,
    pub twist_constraint_damping: f32,
    pub enable_twist_constraint: bool,
}

impl Default for AdhesionSettings {
    fn default() -> Self {
        Self {
            can_break: true,
            break_force: 10.0,
            rest_length: 1.0,
            linear_spring_stiffness: 150.0,
            linear_spring_damping: 5.0,
            orientation_spring_stiffness: 50.0,
            orientation_spring_damping: 5.0,
            max_angular_deviation: 0.0,
            twist_constraint_stiffness: 2.0,
            twist_constraint_damping: 0.5,
            enable_twist_constraint: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModeSettings {
    pub name: String,
    pub default_name: String, // Original/default name to revert to when user clears the name
    pub color: Vec3, // Mode color as Vec3 (0.0-1.0 range)
    pub opacity: f32, // Cell transparency (0.0 = fully transparent, 1.0 = fully opaque)
    pub emissive: f32, // Emissive glow intensity (0.0 = no glow, 1.0+ = bright glow)

    // Cell type
    pub cell_type: i32,

    // Parent settings
    pub parent_make_adhesion: bool,
    pub split_mass: f32,
    pub split_interval: f32,
    pub nutrient_gain_rate: f32, // Mass gained per second (for Test cells)
    pub max_cell_size: f32, // Maximum visual size (1.0 to 2.0 units)
    pub split_ratio: f32, // Ratio of parent mass going to Child A (0.0 to 1.0, default 0.5 for 50/50 split)
    pub nutrient_priority: f32, // Priority for nutrient transport (0.1 to 10.0, default 1.0)
    pub prioritize_when_low: bool, // When enabled, priority increases when nutrients are low to prevent death
    pub parent_split_direction: Vec2, // pitch, yaw in degrees
    pub max_adhesions: i32,
    pub min_adhesions: i32, // Minimum number of connections required before cell can split
    pub enable_parent_angle_snapping: bool,
    pub max_splits: i32, // Maximum number of times a cell can split (1-20, or -1 for infinite). Split count resets to 0 when switching modes
    pub mode_a_after_splits: i32, // Mode that Child A transitions to when max_splits is reached (-1 = use normal child_a mode)
    pub mode_b_after_splits: i32, // Mode that Child B transitions to when max_splits is reached (-1 = use normal child_b mode)
    
    // Flagellocyte settings
    pub swim_force: f32, // Forward thrust force (0.0 to 1.0, for Flagellocyte cells)
    
    // Membrane settings
    pub membrane_stiffness: f32, // Cell membrane stiffness for collision response (0.0 = no repulsion, higher = more rigid)

    // Child settings
    pub child_a: ChildSettings,
    pub child_b: ChildSettings,

    // Adhesion settings
    pub adhesion_settings: AdhesionSettings,
}

impl ModeSettings {
    /// Get split interval (potentially randomized from range)
    pub fn get_split_interval(&self, _cell_id: u32, _tick: u64, _rng_seed: u64) -> f32 {
        // For now, return the fixed value. In the future, this could be randomized
        self.split_interval
    }
    
    /// Get split mass threshold (potentially randomized from range)
    pub fn get_split_mass(&self, _cell_id: u32, _tick: u64, _rng_seed: u64) -> f32 {
        // For now, return the fixed value. In the future, this could be randomized
        self.split_mass
    }
}

impl Default for ModeSettings {
    fn default() -> Self {
        Self {
            name: "Untitled Mode".to_string(),
            default_name: "Untitled Mode".to_string(),
            color: Vec3::new(1.0, 1.0, 1.0),
            opacity: 1.0, // Default: fully opaque
            emissive: 0.0, // Default: no glow
            cell_type: 0,
            parent_make_adhesion: false,
            split_mass: 1.5,
            split_interval: 5.0,
            nutrient_gain_rate: 0.2, // Default: gain 0.2 mass per second
            max_cell_size: 2.0, // Default: max size of 2.0 units
            split_ratio: 0.5, // Default: 50/50 split
            nutrient_priority: 1.0, // Default: neutral priority
            prioritize_when_low: true, // Default: protect cells from death
            parent_split_direction: Vec2::ZERO,
            max_adhesions: 20,
            min_adhesions: 0, // No minimum by default
            enable_parent_angle_snapping: true,
            max_splits: -1, // Infinite by default
            mode_a_after_splits: -1, // Use normal child_a mode by default
            mode_b_after_splits: -1, // Use normal child_b mode by default
            swim_force: 0.5, // Default swim force for flagellocytes
            membrane_stiffness: 50.0, // Default: moderate membrane stiffness
            child_a: ChildSettings::default(),
            child_b: ChildSettings::default(),
            adhesion_settings: AdhesionSettings::default(),
        }
    }
}

impl Default for Genome {
    fn default() -> Self {
        let mut genome = Self {
            name: "Untitled Genome".to_string(),
            initial_mode: 0, // Default to first mode
            initial_orientation: Quat::IDENTITY,
            modes: Vec::new(),
        };
        
        // Create all 40 modes
        for i in 0..40 {
            let mode_name = format!("M {}", i + 1);  // Start mode numbering from 1
            let mut mode = ModeSettings {
                name: mode_name.clone(),
                default_name: mode_name,
                ..Default::default()
            };

            // Generate a color based on the mode number using HSV
            let hue = (i as f32 / 40.0) * 360.0;
            let (r, g, b) = hue_to_rgb(hue);
            mode.color = Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
            
            // Set child modes to split back to themselves
            mode.child_a.mode_number = i as i32;
            mode.child_b.mode_number = i as i32;
            
            genome.modes.push(mode);
        }
        
        genome
    }
}

// Helper function to convert HSV hue to RGB
fn hue_to_rgb(hue: f32) -> (u8, u8, u8) {
    let h = hue / 60.0;
    let c = 1.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    
    let (r, g, b) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    // Scale to 100-255 range for better visibility
    let scale = |v: f32| ((v * 155.0) + 100.0) as u8;
    (scale(r), scale(g), scale(b))
}
