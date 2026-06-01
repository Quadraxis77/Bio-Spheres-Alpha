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
    /// Total number of modes in this genome.
    /// Stored so loading restores the exact mode count rather than defaulting to 1.
    /// Old files without this field default to 80 for backward compatibility.
    #[serde(skip_serializing_if = "is_default_mode_count")]
    #[serde(default = "default_mode_count")]
    pub mode_count: usize,
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

/// Old files without mode_count default to 80 for backward compatibility.
fn default_mode_count() -> usize { 80 }

/// Skip serializing mode_count when it equals 10 (the new default).
fn is_default_mode_count(n: &usize) -> bool { *n == 10 }

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
    pub child_a_after_split_orientation: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_b_after_split_orientation: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_a_after_split_keep_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_b_after_split_keep_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_cell_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_self_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_env_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_boulder_adhesion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_cell_adhesion_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_cell_adhesion_signal_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glueocyte_signal_gate_invert: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swim_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flagellocyte_use_signal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flagellocyte_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flagellocyte_speed_a: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flagellocyte_speed_b: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flagellocyte_threshold_c: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buoyancy_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photocyte_emit_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photocyte_emit_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photocyte_emit_hops: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photocyte_emit_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photocyte_emit_mode: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photocyte_emit_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipocyte_emit_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipocyte_emit_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipocyte_emit_hops: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipocyte_emit_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipocyte_emit_mode: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipocyte_emit_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oculocyte_sense_type: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oculocyte_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oculocyte_signal_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oculocyte_signal_hops: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oculocyte_ray_length: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regulation_emit_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regulation_emit_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regulation_emit_hops: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub division_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub division_signal_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub division_signal_invert: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apoptosis_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apoptosis_signal_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apoptosis_signal_invert: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_a_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_a_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_a_mode_above: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_a_mode_below: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_b_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_b_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_b_mode_above: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_child_b_mode_below: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_switch_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_switch_signal_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_switch_target: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode_switch_invert: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_push_bonded: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_use_signal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_speed_below: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_speed_above: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cilia_attract_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_contraction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_use_signal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_contraction_above: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_contraction_below: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_pulse_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub myocyte_pulse_phase: Option<i32>,
    // Embryocyte settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_use_timer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_release_timer: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_use_threshold: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_threshold_value: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_use_signal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_signal_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryocyte_signal_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devorocyte_consume_range: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devorocyte_consume_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vascular_outlet: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vascular_signal_transport: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vascular_signal_capacity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gametocyte_merge_range: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memorocyte_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memorocyte_input_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memorocyte_output_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memorocyte_output_hops: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognocyte_operation: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognocyte_input_channel_a: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognocyte_input_channel_b: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognocyte_output_channel: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognocyte_output_hops: Option<i32>,
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
    pub break_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rest_length: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linear_spring_stiffness: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linear_spring_damping: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation_spring_stiffness: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation_spring_damping: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_angular_deviation: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub twist_constraint_stiffness: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub twist_constraint_damping: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_twist_constraint: Option<bool>,
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

    /// Serialise this genome to a YAML string (for embedding in snapshots).
    pub fn to_yaml_string(&self) -> Result<String, GenomeSerializeError> {
        let serializable = self.to_serializable();
        let yaml = serde_yaml::to_string(&serializable)?;
        Ok(yaml)
    }

    /// Deserialise a genome from a YAML string (for restoring from snapshots).
    pub fn from_yaml_string(yaml: &str) -> Result<Self, GenomeDeserializeError> {
        let serializable: SerializableGenome = serde_yaml::from_str(yaml)?;
        Self::from_serializable(serializable)
    }

    /// Convert to serializable format, only including modified modes.
    fn to_serializable(&self) -> SerializableGenome {
        let mut modified_modes = Vec::new();

        for (i, mode) in self.modes.iter().enumerate() {
            // Build a per-mode default that matches the load baseline exactly:
            // child_a and child_b default to self-referencing (mode_number = i).
            // This ensures child mode numbers are written correctly even when they
            // point to mode 0 (which would otherwise match the global default and
            // be silently omitted, then revert to self-referencing on load).
            let mut default_mode = ModeSettings::default();
            default_mode.child_a.mode_number = i as i32;
            default_mode.child_b.mode_number = i as i32;

            if let Some(serializable) = mode_to_serializable(mode, &default_mode, i) {
                modified_modes.push(serializable);
            }
        }

        SerializableGenome {
            name: self.name.clone(),
            initial_mode: self.initial_mode,
            initial_orientation: quat_to_array(self.initial_orientation),
            mode_count: self.modes.len(),
            modified_modes,
        }
    }

    /// Create genome from serializable format, applying modifications to defaults.
    fn from_serializable(ser: SerializableGenome) -> Result<Self, GenomeDeserializeError> {
        use super::MAX_MODES;

        // Determine how many modes to create. Old files without mode_count default to 80.
        // Clamp to MAX_MODES so corrupt/future files can't allocate unbounded memory.
        let target_count = ser.mode_count.min(MAX_MODES).max(1);

        // Build a genome with the right number of modes.
        // The baseline for each mode is ModeSettings::default() (cell_type = 0, Test cell)
        // because that is the same baseline used by mode_to_serializable when writing the file.
        // Using any other baseline (e.g. cell_type = 2 from new_with_mode_count) would cause
        // modes whose cell_type matches the write baseline to load with the wrong type.
        let mut genome = super::Genome {
            name: String::new(),
            initial_mode: 0,
            initial_orientation: glam::Quat::IDENTITY,
            modes: (0..target_count).map(|i| {
                let mode_name = format!("M {}", i + 1);
                let mut mode = super::ModeSettings {
                    name: mode_name.clone(),
                    default_name: mode_name,
                    ..super::ModeSettings::default()
                };
                mode.child_a.mode_number = i as i32;
                mode.child_b.mode_number = i as i32;
                mode
            }).collect(),
        };
        genome.name = ser.name;
        genome.initial_mode = ser.initial_mode;
        genome.initial_orientation = array_to_quat(ser.initial_orientation);

        // Apply modified modes. If a mode index exceeds the current vec length
        // (e.g. a file saved with 80 modes but mode_count was missing), extend.
        for modified in ser.modified_modes {
            // Extend the vec if needed (handles old files referencing high indices)
            while modified.index >= genome.modes.len() {
                if genome.modes.len() >= MAX_MODES {
                    return Err(GenomeDeserializeError::InvalidModeIndex(modified.index));
                }
                let idx = genome.modes.len();
                let mode_name = format!("M {}", idx + 1);
                let mut mode = super::ModeSettings {
                    name: mode_name.clone(),
                    default_name: mode_name,
                    ..super::ModeSettings::default()
                };
                mode.child_a.mode_number = idx as i32;
                mode.child_b.mode_number = idx as i32;
                genome.modes.push(mode);
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
        child_a_after_split_orientation: diff_quat(&mode.child_a_after_split_orientation, &default.child_a_after_split_orientation),
        child_b_after_split_orientation: diff_quat(&mode.child_b_after_split_orientation, &default.child_b_after_split_orientation),
        child_a_after_split_keep_adhesion: diff_bool(mode.child_a_after_split_keep_adhesion, default.child_a_after_split_keep_adhesion),
        child_b_after_split_keep_adhesion: diff_bool(mode.child_b_after_split_keep_adhesion, default.child_b_after_split_keep_adhesion),
        glueocyte_cell_adhesion: diff_bool(mode.glueocyte_cell_adhesion, default.glueocyte_cell_adhesion),
        glueocyte_self_adhesion: diff_bool(mode.glueocyte_self_adhesion, default.glueocyte_self_adhesion),
        glueocyte_env_adhesion: diff_bool(mode.glueocyte_env_adhesion, default.glueocyte_env_adhesion),
        glueocyte_boulder_adhesion: diff_bool(mode.glueocyte_boulder_adhesion, default.glueocyte_boulder_adhesion),
        glueocyte_cell_adhesion_signal_channel: diff_i32(mode.glueocyte_cell_adhesion_signal_channel, default.glueocyte_cell_adhesion_signal_channel),
        glueocyte_cell_adhesion_signal_threshold: diff_f32(mode.glueocyte_cell_adhesion_signal_threshold, default.glueocyte_cell_adhesion_signal_threshold),
        glueocyte_signal_gate_invert: diff_bool(mode.glueocyte_signal_gate_invert, default.glueocyte_signal_gate_invert),
        swim_force: diff_f32(mode.swim_force, default.swim_force),
        flagellocyte_use_signal: diff_bool(mode.flagellocyte_use_signal, default.flagellocyte_use_signal),
        flagellocyte_signal_channel: diff_i32(mode.flagellocyte_signal_channel, default.flagellocyte_signal_channel),
        flagellocyte_speed_a: diff_f32(mode.flagellocyte_speed_a, default.flagellocyte_speed_a),
        flagellocyte_speed_b: diff_f32(mode.flagellocyte_speed_b, default.flagellocyte_speed_b),
        flagellocyte_threshold_c: diff_f32(mode.flagellocyte_threshold_c, default.flagellocyte_threshold_c),
        buoyancy_force: diff_f32(mode.buoyancy_force, default.buoyancy_force),
        photocyte_emit_enabled: diff_bool(mode.photocyte_emit_enabled, default.photocyte_emit_enabled),
        photocyte_emit_channel: diff_i32(mode.photocyte_emit_channel, default.photocyte_emit_channel),
        photocyte_emit_hops: diff_i32(mode.photocyte_emit_hops, default.photocyte_emit_hops),
        photocyte_emit_threshold: diff_f32(mode.photocyte_emit_threshold, default.photocyte_emit_threshold),
        photocyte_emit_mode: diff_i32(mode.photocyte_emit_mode, default.photocyte_emit_mode),
        photocyte_emit_value: diff_f32(mode.photocyte_emit_value, default.photocyte_emit_value),
        lipocyte_emit_enabled: diff_bool(mode.lipocyte_emit_enabled, default.lipocyte_emit_enabled),
        lipocyte_emit_channel: diff_i32(mode.lipocyte_emit_channel, default.lipocyte_emit_channel),
        lipocyte_emit_hops: diff_i32(mode.lipocyte_emit_hops, default.lipocyte_emit_hops),
        lipocyte_emit_threshold: diff_f32(mode.lipocyte_emit_threshold, default.lipocyte_emit_threshold),
        lipocyte_emit_mode: diff_i32(mode.lipocyte_emit_mode, default.lipocyte_emit_mode),
        lipocyte_emit_value: diff_f32(mode.lipocyte_emit_value, default.lipocyte_emit_value),
        oculocyte_sense_type: diff_u32(mode.oculocyte_sense_type, default.oculocyte_sense_type),
        oculocyte_signal_channel: diff_i32(mode.oculocyte_signal_channel, default.oculocyte_signal_channel),
        oculocyte_signal_value: diff_f32(mode.oculocyte_signal_value, default.oculocyte_signal_value),
        oculocyte_signal_hops: diff_i32(mode.oculocyte_signal_hops, default.oculocyte_signal_hops),
        oculocyte_ray_length: diff_f32(mode.oculocyte_ray_length, default.oculocyte_ray_length),
        regulation_emit_channel: diff_i32(mode.regulation_emit_channel, default.regulation_emit_channel),
        regulation_emit_value: diff_f32(mode.regulation_emit_value, default.regulation_emit_value),
        regulation_emit_hops: diff_i32(mode.regulation_emit_hops, default.regulation_emit_hops),
        division_signal_channel: diff_i32(mode.division_signal_channel, default.division_signal_channel),
        division_signal_threshold: diff_f32(mode.division_signal_threshold, default.division_signal_threshold),
        division_signal_invert: diff_bool(mode.division_signal_invert, default.division_signal_invert),
        apoptosis_signal_channel: diff_i32(mode.apoptosis_signal_channel, default.apoptosis_signal_channel),
        apoptosis_signal_threshold: diff_f32(mode.apoptosis_signal_threshold, default.apoptosis_signal_threshold),
        apoptosis_signal_invert: diff_bool(mode.apoptosis_signal_invert, default.apoptosis_signal_invert),
        signal_child_a_channel: diff_i32(mode.signal_child_a_channel, default.signal_child_a_channel),
        signal_child_a_threshold: diff_f32(mode.signal_child_a_threshold, default.signal_child_a_threshold),
        signal_child_a_mode_above: diff_i32(mode.signal_child_a_mode_above, default.signal_child_a_mode_above),
        signal_child_a_mode_below: diff_i32(mode.signal_child_a_mode_below, default.signal_child_a_mode_below),
        signal_child_b_channel: diff_i32(mode.signal_child_b_channel, default.signal_child_b_channel),
        signal_child_b_threshold: diff_f32(mode.signal_child_b_threshold, default.signal_child_b_threshold),
        signal_child_b_mode_above: diff_i32(mode.signal_child_b_mode_above, default.signal_child_b_mode_above),
        signal_child_b_mode_below: diff_i32(mode.signal_child_b_mode_below, default.signal_child_b_mode_below),
        mode_switch_signal_channel: diff_i32(mode.mode_switch_signal_channel, default.mode_switch_signal_channel),
        mode_switch_signal_threshold: diff_f32(mode.mode_switch_signal_threshold, default.mode_switch_signal_threshold),
        mode_switch_target: diff_i32(mode.mode_switch_target, default.mode_switch_target),
        mode_switch_invert: diff_bool(mode.mode_switch_invert, default.mode_switch_invert),
        cilia_speed: diff_f32(mode.cilia_speed, default.cilia_speed),
        cilia_push_bonded: diff_bool(mode.cilia_push_bonded, default.cilia_push_bonded),
        cilia_use_signal: diff_bool(mode.cilia_use_signal, default.cilia_use_signal),
        cilia_signal_channel: diff_i32(mode.cilia_signal_channel, default.cilia_signal_channel),
        cilia_speed_below: diff_f32(mode.cilia_speed_below, default.cilia_speed_below),
        cilia_speed_above: diff_f32(mode.cilia_speed_above, default.cilia_speed_above),
        cilia_threshold: diff_f32(mode.cilia_threshold, default.cilia_threshold),
        cilia_attract_force: diff_f32(mode.cilia_attract_force, default.cilia_attract_force),
        myocyte_contraction: diff_f32(mode.myocyte_contraction, default.myocyte_contraction),
        myocyte_use_signal: diff_bool(mode.myocyte_use_signal, default.myocyte_use_signal),
        myocyte_signal_channel: diff_i32(mode.myocyte_signal_channel, default.myocyte_signal_channel),
        myocyte_contraction_above: diff_f32(mode.myocyte_contraction_above, default.myocyte_contraction_above),
        myocyte_contraction_below: diff_f32(mode.myocyte_contraction_below, default.myocyte_contraction_below),
        myocyte_threshold: diff_f32(mode.myocyte_threshold, default.myocyte_threshold),
        myocyte_pulse_rate: diff_f32(mode.myocyte_pulse_rate, default.myocyte_pulse_rate),
        myocyte_pulse_phase: diff_i32(mode.myocyte_pulse_phase, default.myocyte_pulse_phase),
        embryocyte_use_timer: diff_bool(mode.embryocyte_use_timer, default.embryocyte_use_timer),
        embryocyte_release_timer: diff_f32(mode.embryocyte_release_timer, default.embryocyte_release_timer),
        embryocyte_use_threshold: diff_bool(mode.embryocyte_use_threshold, default.embryocyte_use_threshold),
        embryocyte_threshold_value: if mode.embryocyte_threshold_value != default.embryocyte_threshold_value { Some(mode.embryocyte_threshold_value) } else { None },
        embryocyte_use_signal: diff_bool(mode.embryocyte_use_signal, default.embryocyte_use_signal),
        embryocyte_signal_channel: diff_i32(mode.embryocyte_signal_channel, default.embryocyte_signal_channel),
        embryocyte_signal_value: diff_f32(mode.embryocyte_signal_value, default.embryocyte_signal_value),
        devorocyte_consume_range: diff_f32(mode.devorocyte_consume_range, default.devorocyte_consume_range),
        devorocyte_consume_rate: diff_f32(mode.devorocyte_consume_rate, default.devorocyte_consume_rate),
        vascular_outlet: diff_bool(mode.vascular_outlet, default.vascular_outlet),
        vascular_signal_transport: diff_bool(mode.vascular_signal_transport, default.vascular_signal_transport),
        vascular_signal_capacity: diff_f32(mode.vascular_signal_capacity, default.vascular_signal_capacity),
        gametocyte_merge_range: diff_f32(mode.gametocyte_merge_range, default.gametocyte_merge_range),
        memorocyte_rate: diff_f32(mode.memorocyte_rate, default.memorocyte_rate),
        memorocyte_input_channel: diff_i32(mode.memorocyte_input_channel, default.memorocyte_input_channel),
        memorocyte_output_channel: diff_i32(mode.memorocyte_output_channel, default.memorocyte_output_channel),
        memorocyte_output_hops: diff_i32(mode.memorocyte_output_hops, default.memorocyte_output_hops),
        cognocyte_operation: diff_i32(mode.cognocyte_operation, default.cognocyte_operation),
        cognocyte_input_channel_a: diff_i32(mode.cognocyte_input_channel_a, default.cognocyte_input_channel_a),
        cognocyte_input_channel_b: diff_i32(mode.cognocyte_input_channel_b, default.cognocyte_input_channel_b),
        cognocyte_output_channel: diff_i32(mode.cognocyte_output_channel, default.cognocyte_output_channel),
        cognocyte_output_hops: diff_i32(mode.cognocyte_output_hops, default.cognocyte_output_hops),
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
            || self.child_a_after_split_orientation.is_some()
            || self.child_b_after_split_orientation.is_some()
            || self.child_a_after_split_keep_adhesion.is_some()
            || self.child_b_after_split_keep_adhesion.is_some()
            || self.glueocyte_cell_adhesion.is_some()
            || self.glueocyte_self_adhesion.is_some()
            || self.glueocyte_env_adhesion.is_some()
            || self.glueocyte_boulder_adhesion.is_some()
            || self.glueocyte_cell_adhesion_signal_channel.is_some()
            || self.glueocyte_cell_adhesion_signal_threshold.is_some()
            || self.glueocyte_signal_gate_invert.is_some()
            || self.swim_force.is_some()
            || self.flagellocyte_use_signal.is_some()
            || self.flagellocyte_signal_channel.is_some()
            || self.flagellocyte_speed_a.is_some()
            || self.flagellocyte_speed_b.is_some()
            || self.flagellocyte_threshold_c.is_some()
            || self.buoyancy_force.is_some()
            || self.photocyte_emit_enabled.is_some()
            || self.photocyte_emit_channel.is_some()
            || self.photocyte_emit_hops.is_some()
            || self.photocyte_emit_threshold.is_some()
            || self.photocyte_emit_mode.is_some()
            || self.photocyte_emit_value.is_some()
            || self.lipocyte_emit_enabled.is_some()
            || self.lipocyte_emit_channel.is_some()
            || self.lipocyte_emit_hops.is_some()
            || self.lipocyte_emit_threshold.is_some()
            || self.lipocyte_emit_mode.is_some()
            || self.lipocyte_emit_value.is_some()
            || self.oculocyte_sense_type.is_some()
            || self.oculocyte_signal_channel.is_some()
            || self.oculocyte_signal_value.is_some()
            || self.oculocyte_signal_hops.is_some()
            || self.oculocyte_ray_length.is_some()
            || self.regulation_emit_channel.is_some()
            || self.regulation_emit_value.is_some()
            || self.regulation_emit_hops.is_some()
            || self.division_signal_channel.is_some()
            || self.division_signal_threshold.is_some()
            || self.division_signal_invert.is_some()
            || self.apoptosis_signal_channel.is_some()
            || self.apoptosis_signal_threshold.is_some()
            || self.apoptosis_signal_invert.is_some()
            || self.signal_child_a_channel.is_some()
            || self.signal_child_a_threshold.is_some()
            || self.signal_child_a_mode_above.is_some()
            || self.signal_child_a_mode_below.is_some()
            || self.signal_child_b_channel.is_some()
            || self.signal_child_b_threshold.is_some()
            || self.signal_child_b_mode_above.is_some()
            || self.signal_child_b_mode_below.is_some()
            || self.mode_switch_signal_channel.is_some()
            || self.mode_switch_signal_threshold.is_some()
            || self.mode_switch_target.is_some()
            || self.mode_switch_invert.is_some()
            || self.cilia_speed.is_some()
            || self.cilia_push_bonded.is_some()
            || self.cilia_use_signal.is_some()
            || self.cilia_signal_channel.is_some()
            || self.cilia_speed_below.is_some()
            || self.cilia_speed_above.is_some()
            || self.cilia_threshold.is_some()
            || self.cilia_attract_force.is_some()
            || self.myocyte_contraction.is_some()
            || self.myocyte_use_signal.is_some()
            || self.myocyte_signal_channel.is_some()
            || self.myocyte_contraction_above.is_some()
            || self.myocyte_contraction_below.is_some()
            || self.myocyte_threshold.is_some()
            || self.myocyte_pulse_rate.is_some()
            || self.myocyte_pulse_phase.is_some()
            || self.embryocyte_use_timer.is_some()
            || self.embryocyte_release_timer.is_some()
            || self.embryocyte_use_threshold.is_some()
            || self.embryocyte_threshold_value.is_some()
            || self.embryocyte_use_signal.is_some()
            || self.embryocyte_signal_channel.is_some()
            || self.embryocyte_signal_value.is_some()
            || self.devorocyte_consume_range.is_some()
            || self.devorocyte_consume_rate.is_some()
            || self.vascular_outlet.is_some()
            || self.vascular_signal_transport.is_some()
            || self.vascular_signal_capacity.is_some()
            || self.gametocyte_merge_range.is_some()
            || self.memorocyte_rate.is_some()
            || self.memorocyte_input_channel.is_some()
            || self.memorocyte_output_channel.is_some()
            || self.memorocyte_output_hops.is_some()
            || self.cognocyte_operation.is_some()
            || self.cognocyte_input_channel_a.is_some()
            || self.cognocyte_input_channel_b.is_some()
            || self.cognocyte_output_channel.is_some()
            || self.cognocyte_output_hops.is_some()
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
        break_force: diff_f32(adhesion.break_force, default.break_force),
        rest_length: diff_f32(adhesion.rest_length, default.rest_length),
        linear_spring_stiffness: diff_f32(adhesion.linear_spring_stiffness, default.linear_spring_stiffness),
        linear_spring_damping: diff_f32(adhesion.linear_spring_damping, default.linear_spring_damping),
        orientation_spring_stiffness: diff_f32(adhesion.orientation_spring_stiffness, default.orientation_spring_stiffness),
        orientation_spring_damping: diff_f32(adhesion.orientation_spring_damping, default.orientation_spring_damping),
        max_angular_deviation: diff_f32(adhesion.max_angular_deviation, default.max_angular_deviation),
        twist_constraint_stiffness: diff_f32(adhesion.twist_constraint_stiffness, default.twist_constraint_stiffness),
        twist_constraint_damping: diff_f32(adhesion.twist_constraint_damping, default.twist_constraint_damping),
        enable_twist_constraint: diff_bool(adhesion.enable_twist_constraint, default.enable_twist_constraint),
    };

    if ser.can_break.is_some()
        || ser.break_force.is_some()
        || ser.rest_length.is_some()
        || ser.linear_spring_stiffness.is_some()
        || ser.linear_spring_damping.is_some()
        || ser.orientation_spring_stiffness.is_some()
        || ser.orientation_spring_damping.is_some()
        || ser.max_angular_deviation.is_some()
        || ser.twist_constraint_stiffness.is_some()
        || ser.twist_constraint_damping.is_some()
        || ser.enable_twist_constraint.is_some()
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
    if let Some(orientation) = ser.child_a_after_split_orientation {
        mode.child_a_after_split_orientation = Quat::from_array(orientation);
    }
    if let Some(orientation) = ser.child_b_after_split_orientation {
        mode.child_b_after_split_orientation = Quat::from_array(orientation);
    }
    if let Some(keep_adhesion) = ser.child_a_after_split_keep_adhesion {
        mode.child_a_after_split_keep_adhesion = keep_adhesion;
    }
    if let Some(keep_adhesion) = ser.child_b_after_split_keep_adhesion {
        mode.child_b_after_split_keep_adhesion = keep_adhesion;
    }
    if let Some(v) = ser.glueocyte_cell_adhesion { mode.glueocyte_cell_adhesion = v; }
    if let Some(v) = ser.glueocyte_self_adhesion  { mode.glueocyte_self_adhesion  = v; }
    if let Some(v) = ser.glueocyte_env_adhesion  { mode.glueocyte_env_adhesion  = v; }
    if let Some(v) = ser.glueocyte_boulder_adhesion { mode.glueocyte_boulder_adhesion = v; }
    if let Some(v) = ser.glueocyte_cell_adhesion_signal_channel {
        mode.glueocyte_cell_adhesion_signal_channel = v;
    }
    if let Some(v) = ser.glueocyte_cell_adhesion_signal_threshold {
        mode.glueocyte_cell_adhesion_signal_threshold = v;
    }
    if let Some(v) = ser.glueocyte_signal_gate_invert { mode.glueocyte_signal_gate_invert = v; }
    if let Some(swim_force) = ser.swim_force {
        mode.swim_force = swim_force;
    }
    if let Some(use_signal) = ser.flagellocyte_use_signal {
        mode.flagellocyte_use_signal = use_signal;
    }
    if let Some(channel) = ser.flagellocyte_signal_channel {
        mode.flagellocyte_signal_channel = channel;
    }
    if let Some(speed_a) = ser.flagellocyte_speed_a {
        mode.flagellocyte_speed_a = speed_a;
    }
    if let Some(speed_b) = ser.flagellocyte_speed_b {
        mode.flagellocyte_speed_b = speed_b;
    }
    if let Some(threshold_c) = ser.flagellocyte_threshold_c {
        mode.flagellocyte_threshold_c = threshold_c;
    }
    if let Some(buoyancy_force) = ser.buoyancy_force {
        mode.buoyancy_force = buoyancy_force;
    }
    if let Some(sense_type) = ser.oculocyte_sense_type {
        mode.oculocyte_sense_type = sense_type;
    }
    if let Some(channel) = ser.oculocyte_signal_channel {
        mode.oculocyte_signal_channel = channel;
    }
    if let Some(value) = ser.oculocyte_signal_value {
        mode.oculocyte_signal_value = value;
    }
    if let Some(hops) = ser.oculocyte_signal_hops {
        mode.oculocyte_signal_hops = hops;
    }
    if let Some(len) = ser.oculocyte_ray_length {
        mode.oculocyte_ray_length = len;
    }
    if let Some(v) = ser.regulation_emit_channel {
        mode.regulation_emit_channel = v;
    }
    if let Some(v) = ser.regulation_emit_value {
        mode.regulation_emit_value = v;
    }
    if let Some(v) = ser.regulation_emit_hops {
        mode.regulation_emit_hops = v;
    }
    if let Some(v) = ser.division_signal_channel {
        mode.division_signal_channel = v;
    }
    if let Some(v) = ser.division_signal_threshold {
        mode.division_signal_threshold = v;
    }
    if let Some(v) = ser.division_signal_invert {
        mode.division_signal_invert = v;
    }
    if let Some(v) = ser.apoptosis_signal_channel {
        mode.apoptosis_signal_channel = v;
    }
    if let Some(v) = ser.apoptosis_signal_threshold {
        mode.apoptosis_signal_threshold = v;
    }
    if let Some(v) = ser.apoptosis_signal_invert {
        mode.apoptosis_signal_invert = v;
    }
    if let Some(v) = ser.signal_child_a_channel {
        mode.signal_child_a_channel = v;
    }
    if let Some(v) = ser.signal_child_a_threshold {
        mode.signal_child_a_threshold = v;
    }
    if let Some(v) = ser.signal_child_a_mode_above {
        mode.signal_child_a_mode_above = v;
    }
    if let Some(v) = ser.signal_child_a_mode_below {
        mode.signal_child_a_mode_below = v;
    }
    if let Some(v) = ser.signal_child_b_channel {
        mode.signal_child_b_channel = v;
    }
    if let Some(v) = ser.signal_child_b_threshold {
        mode.signal_child_b_threshold = v;
    }
    if let Some(v) = ser.signal_child_b_mode_above {
        mode.signal_child_b_mode_above = v;
    }
    if let Some(v) = ser.signal_child_b_mode_below {
        mode.signal_child_b_mode_below = v;
    }
    if let Some(v) = ser.mode_switch_signal_channel {
        mode.mode_switch_signal_channel = v;
    }
    if let Some(v) = ser.mode_switch_signal_threshold {
        mode.mode_switch_signal_threshold = v;
    }
    if let Some(v) = ser.mode_switch_target {
        mode.mode_switch_target = v;
    }
    if let Some(v) = ser.mode_switch_invert {
        mode.mode_switch_invert = v;
    }
    if let Some(v) = ser.cilia_speed {
        mode.cilia_speed = v;
    }
    if let Some(v) = ser.cilia_push_bonded {
        mode.cilia_push_bonded = v;
    }
    if let Some(v) = ser.cilia_use_signal {
        mode.cilia_use_signal = v;
    }
    if let Some(v) = ser.cilia_signal_channel {
        mode.cilia_signal_channel = v;
    }
    if let Some(v) = ser.cilia_speed_below {
        mode.cilia_speed_below = v;
    }
    if let Some(v) = ser.cilia_speed_above {
        mode.cilia_speed_above = v;
    }
    if let Some(v) = ser.cilia_threshold {
        mode.cilia_threshold = v;
    }
    if let Some(v) = ser.cilia_attract_force {
        mode.cilia_attract_force = v;
    }
    if let Some(v) = ser.myocyte_contraction {
        mode.myocyte_contraction = v;
    }
    if let Some(v) = ser.myocyte_use_signal {
        mode.myocyte_use_signal = v;
    }
    if let Some(v) = ser.myocyte_signal_channel {
        mode.myocyte_signal_channel = v;
    }
    if let Some(v) = ser.myocyte_contraction_above {
        mode.myocyte_contraction_above = v;
    }
    if let Some(v) = ser.myocyte_contraction_below {
        mode.myocyte_contraction_below = v;
    }
    if let Some(v) = ser.myocyte_threshold {
        mode.myocyte_threshold = v;
    }
    if let Some(v) = ser.myocyte_pulse_rate {
        mode.myocyte_pulse_rate = v;
    }
    if let Some(v) = ser.myocyte_pulse_phase {
        mode.myocyte_pulse_phase = v;
    }
    if let Some(v) = ser.embryocyte_use_timer {
        mode.embryocyte_use_timer = v;
    }
    if let Some(v) = ser.embryocyte_release_timer {
        mode.embryocyte_release_timer = v;
    }
    if let Some(v) = ser.embryocyte_use_threshold {
        mode.embryocyte_use_threshold = v;
    }
    if let Some(v) = ser.embryocyte_threshold_value {
        mode.embryocyte_threshold_value = v;
    }
    if let Some(v) = ser.embryocyte_use_signal {
        mode.embryocyte_use_signal = v;
    }
    if let Some(v) = ser.embryocyte_signal_channel {
        mode.embryocyte_signal_channel = v;
    }
    if let Some(v) = ser.embryocyte_signal_value {
        mode.embryocyte_signal_value = v;
    }
    if let Some(v) = ser.devorocyte_consume_range {
        mode.devorocyte_consume_range = v;
    }
    if let Some(v) = ser.devorocyte_consume_rate {
        mode.devorocyte_consume_rate = v;
    }
    if let Some(v) = ser.vascular_outlet {
        mode.vascular_outlet = v;
    }
    if let Some(v) = ser.gametocyte_merge_range {
        mode.gametocyte_merge_range = v;
    }
    if let Some(v) = ser.photocyte_emit_enabled    { mode.photocyte_emit_enabled = v; }
    if let Some(v) = ser.photocyte_emit_channel   { mode.photocyte_emit_channel = v; }
    if let Some(v) = ser.photocyte_emit_hops       { mode.photocyte_emit_hops = v; }
    if let Some(v) = ser.photocyte_emit_threshold  { mode.photocyte_emit_threshold = v; }
    if let Some(v) = ser.photocyte_emit_mode       { mode.photocyte_emit_mode = v; }
    if let Some(v) = ser.photocyte_emit_value      { mode.photocyte_emit_value = v; }
    if let Some(v) = ser.lipocyte_emit_enabled      { mode.lipocyte_emit_enabled = v; }
    if let Some(v) = ser.lipocyte_emit_channel     { mode.lipocyte_emit_channel = v; }
    if let Some(v) = ser.lipocyte_emit_hops        { mode.lipocyte_emit_hops = v; }
    if let Some(v) = ser.lipocyte_emit_threshold   { mode.lipocyte_emit_threshold = v; }
    if let Some(v) = ser.lipocyte_emit_mode        { mode.lipocyte_emit_mode = v; }
    if let Some(v) = ser.lipocyte_emit_value       { mode.lipocyte_emit_value = v; }
    if let Some(v) = ser.vascular_signal_transport { mode.vascular_signal_transport = v; }
    if let Some(v) = ser.vascular_signal_capacity { mode.vascular_signal_capacity = v; }
    if let Some(v) = ser.memorocyte_rate           { mode.memorocyte_rate = v; }
    if let Some(v) = ser.memorocyte_input_channel  { mode.memorocyte_input_channel = v; }
    if let Some(v) = ser.memorocyte_output_channel { mode.memorocyte_output_channel = v; }
    if let Some(v) = ser.memorocyte_output_hops    { mode.memorocyte_output_hops = v; }
    if let Some(v) = ser.cognocyte_operation {
        mode.cognocyte_operation = v;
    }
    if let Some(v) = ser.cognocyte_input_channel_a {
        mode.cognocyte_input_channel_a = v;
    }
    if let Some(v) = ser.cognocyte_input_channel_b {
        mode.cognocyte_input_channel_b = v;
    }
    if let Some(v) = ser.cognocyte_output_channel {
        mode.cognocyte_output_channel = v;
    }
    if let Some(v) = ser.cognocyte_output_hops {
        mode.cognocyte_output_hops = v;
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
    if let Some(break_force) = ser.break_force {
        adhesion.break_force = break_force;
    }
    if let Some(rest_length) = ser.rest_length {
        adhesion.rest_length = rest_length;
    }
    if let Some(stiffness) = ser.linear_spring_stiffness {
        adhesion.linear_spring_stiffness = stiffness;
    }
    if let Some(damping) = ser.linear_spring_damping {
        adhesion.linear_spring_damping = damping;
    }
    if let Some(stiffness) = ser.orientation_spring_stiffness {
        adhesion.orientation_spring_stiffness = stiffness;
    }
    if let Some(damping) = ser.orientation_spring_damping {
        adhesion.orientation_spring_damping = damping;
    }
    if let Some(max_angular) = ser.max_angular_deviation {
        adhesion.max_angular_deviation = max_angular;
    }
    if let Some(stiffness) = ser.twist_constraint_stiffness {
        adhesion.twist_constraint_stiffness = stiffness;
    }
    if let Some(damping) = ser.twist_constraint_damping {
        adhesion.twist_constraint_damping = damping;
    }
    if let Some(enable) = ser.enable_twist_constraint {
        adhesion.enable_twist_constraint = enable;
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

fn diff_u32(value: u32, default: u32) -> Option<u32> {
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
