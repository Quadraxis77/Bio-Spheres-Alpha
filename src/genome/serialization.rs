//! Genome serialization for save/load functionality.
//!
//! Saves genomes as human-readable YAML (.genome files), only including modes that differ from defaults.

use super::{AdhesionSettings, ChildSettings, Genome, ModeSettings};
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GenomeSerializeError {
    #[error("Failed to serialize genome: {0}")]
    Serialize(#[from] serde_yaml::Error),
    #[error("Failed to write genome file: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum GenomeDeserializeError {
    #[error("Failed to read genome file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse genome: {0}")]
    Parse(#[from] serde_yaml::Error),
    #[error("Invalid mode index: {0}")]
    InvalidModeIndex(usize),
}

/// Serializable genome format - only stores modified data
#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableGenome {
    pub name: String,
    #[serde(skip_serializing_if = "is_default_initial_mode")]
    #[serde(default)]
    pub initial_mode: i32,
    #[serde(skip_serializing_if = "is_identity_quat")]
    #[serde(default = "default_quat")]
    pub initial_orientation: [f32; 4],
    /// Only modes that differ from defaults are stored
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub modified_modes: Vec<SerializableMode>,
}

fn is_default_initial_mode(mode: &i32) -> bool {
    *mode == 0
}

fn is_identity_quat(q: &[f32; 4]) -> bool {
    (q[0] - 0.0).abs() < 0.0001
        && (q[1] - 0.0).abs() < 0.0001
        && (q[2] - 0.0).abs() < 0.0001
        && (q[3] - 1.0).abs() < 0.0001
}

fn default_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// A mode with its index and only the fields that differ from default
#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableMode {
    pub index: usize,
    #[serde(flatten)]
    pub settings: SerializableModeSettings,
}


/// Serializable mode settings - all fields optional, only non-default values serialized
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SerializableModeSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emissive: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_type: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_make_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mass: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_interval: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nutrient_gain_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cell_size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nutrient_priority: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prioritize_when_low: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_split_direction: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_adhesions: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_adhesions: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_parent_angle_snapping: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_splits: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_a_after_splits: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_b_after_splits: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swim_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_a: Option<SerializableChildSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_b: Option<SerializableChildSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adhesion_settings: Option<SerializableAdhesionSettings>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SerializableChildSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_number: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_angle_snapping: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SerializableAdhesionSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub can_break: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adhesin_length: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adhesin_stretch: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stiffness: Option<f32>,
}


// ============================================================================
// Conversion: Genome -> SerializableGenome (for saving)
// ============================================================================

impl Genome {
    /// Save genome to a YAML file (.genome).
    pub fn save_to_file(&self, path: &Path) -> Result<(), GenomeSerializeError> {
        let serializable = self.to_serializable();
        let yaml = serde_yaml::to_string(&serializable)?;
        std::fs::write(path, yaml)?;
        log::info!("Saved genome to {:?}", path);
        Ok(())
    }

    /// Load genome from a YAML file (.genome).
    pub fn load_from_file(path: &Path) -> Result<Self, GenomeDeserializeError> {
        let yaml = std::fs::read_to_string(path)?;
        let serializable: SerializableGenome = serde_yaml::from_str(&yaml)?;
        let genome = Self::from_serializable(serializable)?;
        log::info!("Loaded genome from {:?}", path);
        Ok(genome)
    }

    /// Convert to serializable format, only including modified modes.
    fn to_serializable(&self) -> SerializableGenome {
        let default_genome = Genome::default();
        let mut modified_modes = Vec::new();

        for (i, mode) in self.modes.iter().enumerate() {
            let default_mode = &default_genome.modes[i];
            if let Some(serializable) = mode_to_serializable(mode, default_mode, i) {
                modified_modes.push(serializable);
            }
        }

        SerializableGenome {
            name: self.name.clone(),
            initial_mode: self.initial_mode,
            initial_orientation: quat_to_array(self.initial_orientation),
            modified_modes,
        }
    }

    /// Create genome from serializable format, applying modifications to defaults.
    fn from_serializable(ser: SerializableGenome) -> Result<Self, GenomeDeserializeError> {
        let mut genome = Genome::default();
        genome.name = ser.name;
        genome.initial_mode = ser.initial_mode;
        genome.initial_orientation = array_to_quat(ser.initial_orientation);

        // Apply modified modes
        for modified in ser.modified_modes {
            if modified.index >= genome.modes.len() {
                return Err(GenomeDeserializeError::InvalidModeIndex(modified.index));
            }
            apply_mode_settings(&mut genome.modes[modified.index], &modified.settings);
        }

        Ok(genome)
    }
}

/// Convert a mode to serializable format if it differs from default.
fn mode_to_serializable(
    mode: &ModeSettings,
    default: &ModeSettings,
    index: usize,
) -> Option<SerializableMode> {
    let settings = SerializableModeSettings {
        name: diff_option(&mode.name, &default.name),
        color: diff_vec3(&mode.color, &default.color),
        opacity: diff_f32(mode.opacity, default.opacity),
        emissive: diff_f32(mode.emissive, default.emissive),
        cell_type: diff_i32(mode.cell_type, default.cell_type),
        parent_make_adhesion: diff_bool(mode.parent_make_adhesion, default.parent_make_adhesion),
        split_mass: diff_f32(mode.split_mass, default.split_mass),
        split_interval: diff_f32(mode.split_interval, default.split_interval),
        nutrient_gain_rate: diff_f32(mode.nutrient_gain_rate, default.nutrient_gain_rate),
        max_cell_size: diff_f32(mode.max_cell_size, default.max_cell_size),
        split_ratio: diff_f32(mode.split_ratio, default.split_ratio),
        nutrient_priority: diff_f32(mode.nutrient_priority, default.nutrient_priority),
        prioritize_when_low: diff_bool(mode.prioritize_when_low, default.prioritize_when_low),
        parent_split_direction: diff_vec2(&mode.parent_split_direction, &default.parent_split_direction),
        max_adhesions: diff_i32(mode.max_adhesions, default.max_adhesions),
        min_adhesions: diff_i32(mode.min_adhesions, default.min_adhesions),
        enable_parent_angle_snapping: diff_bool(mode.enable_parent_angle_snapping, default.enable_parent_angle_snapping),
        max_splits: diff_i32(mode.max_splits, default.max_splits),
        mode_a_after_splits: diff_i32(mode.mode_a_after_splits, default.mode_a_after_splits),
        mode_b_after_splits: diff_i32(mode.mode_b_after_splits, default.mode_b_after_splits),
        swim_force: diff_f32(mode.swim_force, default.swim_force),
        child_a: child_to_serializable(&mode.child_a, &default.child_a),
        child_b: child_to_serializable(&mode.child_b, &default.child_b),
        adhesion_settings: adhesion_to_serializable(&mode.adhesion_settings, &default.adhesion_settings),
    };

    // Only include if any field is set
    if settings.has_any_field() {
        Some(SerializableMode { index, settings })
    } else {
        None
    }
}


impl SerializableModeSettings {
    fn has_any_field(&self) -> bool {
        self.name.is_some()
            || self.color.is_some()
            || self.opacity.is_some()
            || self.emissive.is_some()
            || self.cell_type.is_some()
            || self.parent_make_adhesion.is_some()
            || self.split_mass.is_some()
            || self.split_interval.is_some()
            || self.nutrient_gain_rate.is_some()
            || self.max_cell_size.is_some()
            || self.split_ratio.is_some()
            || self.nutrient_priority.is_some()
            || self.prioritize_when_low.is_some()
            || self.parent_split_direction.is_some()
            || self.max_adhesions.is_some()
            || self.min_adhesions.is_some()
            || self.enable_parent_angle_snapping.is_some()
            || self.max_splits.is_some()
            || self.mode_a_after_splits.is_some()
            || self.mode_b_after_splits.is_some()
            || self.swim_force.is_some()
            || self.child_a.is_some()
            || self.child_b.is_some()
            || self.adhesion_settings.is_some()
    }
}

fn child_to_serializable(
    child: &ChildSettings,
    default: &ChildSettings,
) -> Option<SerializableChildSettings> {
    let ser = SerializableChildSettings {
        mode_number: diff_i32(child.mode_number, default.mode_number),
        orientation: diff_quat(&child.orientation, &default.orientation),
        keep_adhesion: diff_bool(child.keep_adhesion, default.keep_adhesion),
        enable_angle_snapping: diff_bool(child.enable_angle_snapping, default.enable_angle_snapping),
    };

    if ser.mode_number.is_some()
        || ser.orientation.is_some()
        || ser.keep_adhesion.is_some()
        || ser.enable_angle_snapping.is_some()
    {
        Some(ser)
    } else {
        None
    }
}

fn adhesion_to_serializable(
    adhesion: &AdhesionSettings,
    default: &AdhesionSettings,
) -> Option<SerializableAdhesionSettings> {
    let ser = SerializableAdhesionSettings {
        can_break: diff_bool(adhesion.can_break, default.can_break),
        adhesin_length: diff_f32(adhesion.adhesin_length, default.adhesin_length),
        adhesin_stretch: diff_f32(adhesion.adhesin_stretch, default.adhesin_stretch),
        stiffness: diff_f32(adhesion.stiffness, default.stiffness),
    };

    if ser.can_break.is_some()
        || ser.adhesin_length.is_some()
        || ser.adhesin_stretch.is_some()
        || ser.stiffness.is_some()
    {
        Some(ser)
    } else {
        None
    }
}


// ============================================================================
// Apply serialized settings back to mode (for loading)
// ============================================================================

fn apply_mode_settings(mode: &mut ModeSettings, ser: &SerializableModeSettings) {
    if let Some(ref name) = ser.name {
        mode.name = name.clone();
    }
    if let Some(color) = ser.color {
        mode.color = Vec3::from_array(color);
    }
    if let Some(opacity) = ser.opacity {
        mode.opacity = opacity;
    }
    if let Some(emissive) = ser.emissive {
        mode.emissive = emissive;
    }
    if let Some(cell_type) = ser.cell_type {
        mode.cell_type = cell_type;
    }
    if let Some(parent_make_adhesion) = ser.parent_make_adhesion {
        mode.parent_make_adhesion = parent_make_adhesion;
    }
    if let Some(split_mass) = ser.split_mass {
        mode.split_mass = split_mass;
    }
    if let Some(split_interval) = ser.split_interval {
        mode.split_interval = split_interval;
    }
    if let Some(nutrient_gain_rate) = ser.nutrient_gain_rate {
        mode.nutrient_gain_rate = nutrient_gain_rate;
    }
    if let Some(max_cell_size) = ser.max_cell_size {
        mode.max_cell_size = max_cell_size;
    }
    if let Some(split_ratio) = ser.split_ratio {
        mode.split_ratio = split_ratio;
    }
    if let Some(nutrient_priority) = ser.nutrient_priority {
        mode.nutrient_priority = nutrient_priority;
    }
    if let Some(prioritize_when_low) = ser.prioritize_when_low {
        mode.prioritize_when_low = prioritize_when_low;
    }
    if let Some(dir) = ser.parent_split_direction {
        mode.parent_split_direction = Vec2::from_array(dir);
    }
    if let Some(max_adhesions) = ser.max_adhesions {
        mode.max_adhesions = max_adhesions;
    }
    if let Some(min_adhesions) = ser.min_adhesions {
        mode.min_adhesions = min_adhesions;
    }
    if let Some(enable) = ser.enable_parent_angle_snapping {
        mode.enable_parent_angle_snapping = enable;
    }
    if let Some(max_splits) = ser.max_splits {
        mode.max_splits = max_splits;
    }
    if let Some(mode_a) = ser.mode_a_after_splits {
        mode.mode_a_after_splits = mode_a;
    }
    if let Some(mode_b) = ser.mode_b_after_splits {
        mode.mode_b_after_splits = mode_b;
    }
    if let Some(swim_force) = ser.swim_force {
        mode.swim_force = swim_force;
    }
    if let Some(ref child_a) = ser.child_a {
        apply_child_settings(&mut mode.child_a, child_a);
    }
    if let Some(ref child_b) = ser.child_b {
        apply_child_settings(&mut mode.child_b, child_b);
    }
    if let Some(ref adhesion) = ser.adhesion_settings {
        apply_adhesion_settings(&mut mode.adhesion_settings, adhesion);
    }
}

fn apply_child_settings(child: &mut ChildSettings, ser: &SerializableChildSettings) {
    if let Some(mode_number) = ser.mode_number {
        child.mode_number = mode_number;
    }
    if let Some(orientation) = ser.orientation {
        child.orientation = array_to_quat(orientation);
    }
    if let Some(keep_adhesion) = ser.keep_adhesion {
        child.keep_adhesion = keep_adhesion;
    }
    if let Some(enable) = ser.enable_angle_snapping {
        child.enable_angle_snapping = enable;
    }
}

fn apply_adhesion_settings(adhesion: &mut AdhesionSettings, ser: &SerializableAdhesionSettings) {
    if let Some(can_break) = ser.can_break {
        adhesion.can_break = can_break;
    }
    if let Some(adhesin_length) = ser.adhesin_length {
        adhesion.adhesin_length = adhesin_length;
    }
    if let Some(adhesin_stretch) = ser.adhesin_stretch {
        adhesion.adhesin_stretch = adhesin_stretch;
    }
    if let Some(stiffness) = ser.stiffness {
        adhesion.stiffness = stiffness;
    }
}


// ============================================================================
// Helper functions for diffing values
// ============================================================================

fn diff_option(value: &String, default: &String) -> Option<String> {
    if value != default {
        Some(value.clone())
    } else {
        None
    }
}

fn diff_f32(value: f32, default: f32) -> Option<f32> {
    if (value - default).abs() > 0.0001 {
        Some(value)
    } else {
        None
    }
}

fn diff_i32(value: i32, default: i32) -> Option<i32> {
    if value != default {
        Some(value)
    } else {
        None
    }
}

fn diff_bool(value: bool, default: bool) -> Option<bool> {
    if value != default {
        Some(value)
    } else {
        None
    }
}

fn diff_vec2(value: &Vec2, default: &Vec2) -> Option<[f32; 2]> {
    if (*value - *default).length() > 0.0001 {
        Some(value.to_array())
    } else {
        None
    }
}

fn diff_vec3(value: &Vec3, default: &Vec3) -> Option<[f32; 3]> {
    if (*value - *default).length() > 0.0001 {
        Some(value.to_array())
    } else {
        None
    }
}

fn diff_quat(value: &Quat, default: &Quat) -> Option<[f32; 4]> {
    // Compare quaternions - they're equal if dot product is ~1 or ~-1
    let dot = value.dot(*default).abs();
    if dot < 0.9999 {
        Some(quat_to_array(*value))
    } else {
        None
    }
}

fn quat_to_array(q: Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

fn array_to_quat(arr: [f32; 4]) -> Quat {
    Quat::from_xyzw(arr[0], arr[1], arr[2], arr[3])
}
