pub mod node_graph;
pub mod procedural_name;
pub mod serialization;

use glam::{Quat, Vec2, Vec3};

pub use serialization::{GenomeDeserializeError, GenomeSerializeError};

/// Minimum genome similarity [0.0, 1.0] required for two Gametocytes to merge.
/// Computed as: mode-count alignment x cell-type match fraction across the shared mode prefix.
/// At 0.5, organisms must share at least half their cell-type sequence (weighted by size alignment).
pub const GAMETOCYTE_MIN_SIMILARITY: f32 = 0.5;

/// Maximum number of modes a genome can have.
/// Raised from 80 to 128 to give more room for complex creatures.
pub const MAX_MODES: usize = 128;

#[derive(Debug, Clone)]
pub struct Genome {
    pub name: String,
    pub initial_mode: i32,
    pub initial_orientation: Quat,
    pub modes: Vec<ModeSettings>,
}

/// Child settings for mode transitions
#[derive(Debug, Clone, PartialEq)]
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
            break_force: 500.0,
            rest_length: 1.0,
            linear_spring_stiffness: 150.0,
            linear_spring_damping: 5.0,
            orientation_spring_stiffness: 50.0,
            orientation_spring_damping: 0.5,
            max_angular_deviation: 0.0,
            twist_constraint_stiffness: 2.0,
            twist_constraint_damping: 20.0,
            enable_twist_constraint: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModeSettings {
    pub name: String,
    pub default_name: String, // Original/default name to revert to when user clears the name
    pub color: Vec3,          // Mode color as Vec3 (0.0-1.0 range)
    pub opacity: f32,         // Cell transparency (0.0 = fully transparent, 1.0 = fully opaque)
    pub emissive: f32,        // Emissive glow intensity (0.0 = no glow, 1.0+ = bright glow)

    // Cell type
    pub cell_type: i32,

    // Parent settings
    pub parent_make_adhesion: bool,
    pub split_mass: f32,
    pub split_interval: f32,
    pub nutrient_gain_rate: f32, // Mass gained per second (for Test cells)
    pub max_cell_size: f32,      // Maximum visual size (1.0 to 2.0 units)
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
    pub child_a_after_split_orientation: Quat, // Orientation for Child A when max_splits is reached
    pub child_b_after_split_orientation: Quat, // Orientation for Child B when max_splits is reached
    pub child_a_after_split_keep_adhesion: bool, // Whether Child A keeps adhesion when max_splits is reached
    pub child_b_after_split_keep_adhesion: bool, // Whether Child B keeps adhesion when max_splits is reached

    // Glueocyte settings
    pub glueocyte_cell_adhesion: bool, // Whether this Glueocyte bonds to other cells on contact
    /// Whether this Glueocyte bonds to cells of its own organism.
    /// When false (default), only bonds to foreign cells.
    pub glueocyte_self_adhesion: bool,
    pub glueocyte_env_adhesion: bool, // Whether this Glueocyte bonds to the environment on contact
    pub glueocyte_boulder_adhesion: bool, // Whether this Glueocyte bonds to boulders/mossrocks
    /// Signal channel that controls cell-adhesion activation (-1 = always active, 0-15 = any sensory or regulation channel).
    /// When set, the glueocyte only forms new bonds when the signal is above the threshold,
    /// and releases all glueocyte-created bonds when the signal drops below the threshold.
    pub glueocyte_cell_adhesion_signal_channel: i32,
    pub glueocyte_cell_adhesion_signal_threshold: f32,
    /// When true, the gate logic is inverted: active (bonds) when signal < threshold,
    /// releases when signal >= threshold. "Disconnect when signal" instead of "disconnect when no signal".
    pub glueocyte_signal_gate_invert: bool,

    // Flagellocyte settings
    pub swim_force: f32, // Forward thrust force (0.0 to 1.0, for Flagellocyte cells)
    pub flagellocyte_use_signal: bool, // If true, use signal-based speed; if false, use fixed swim_force
    pub flagellocyte_signal_channel: i32, // Which signal channel to read (0-7, sensory channels)
    pub flagellocyte_speed_a: f32,     // Swim speed when signal < threshold_c
    pub flagellocyte_speed_b: f32,     // Swim speed when signal >= threshold_c
    pub flagellocyte_threshold_c: f32, // Signal threshold for speed switching

    // Buoyocyte settings
    pub buoyancy_force: f32, // Upward buoyancy force (0.0 to 1.0, for Buoyocyte cells)

    // Photocyte signal emission
    pub photocyte_emit_enabled: bool, // Whether signal emission is active
    pub photocyte_emit_channel: i32,  // 0-15 = channel to emit on
    pub photocyte_emit_hops: i32,     // How many adhesion hops (1-20)
    pub photocyte_emit_threshold: f32, // Light level threshold (0.0-1.0)
    pub photocyte_emit_mode: i32,     // 0 = above threshold, 1 = below threshold
    pub photocyte_emit_value: f32,    // Signal value to emit (-100.0 to 100.0)

    // Lipocyte signal emission
    pub lipocyte_emit_enabled: bool, // Whether signal emission is active
    pub lipocyte_emit_channel: i32,  // 0-15 = channel to emit on
    pub lipocyte_emit_hops: i32,     // How many adhesion hops (1-20)
    pub lipocyte_emit_threshold: f32, // Storage fraction threshold (0.0-1.0)
    pub lipocyte_emit_mode: i32,     // 0 = above threshold, 1 = below threshold
    pub lipocyte_emit_value: f32,    // Signal value to emit (-100.0 to 100.0)

    // Oculocyte settings
    pub oculocyte_sense_type: u32, // Bitmask: bit0=Cell, bit1=Food, bit2=Light, bit3=Barrier, bit4=Self, bit5=Mossrock
    pub oculocyte_signal_channel: i32, // Which channel to send on (0-7, sensory channel range)
    pub oculocyte_signal_value: f32, // Signal value to send when target detected (-50.0 to 50.0)
    pub oculocyte_signal_hops: i32, // How many adhesion hops the signal propagates (1-20)
    pub oculocyte_ray_length: f32, // How far ahead the oculocyte ray reaches (1.0 to 100.0)

    // Ciliocyte settings
    pub cilia_speed: f32, // Cilia force magnitude (-1.0 to +1.0, for Ciliocyte cells)
    pub cilia_push_bonded: bool, // Whether to push same-organism cells
    pub cilia_use_signal: bool, // If true, use signal-based speed; if false, use fixed cilia_speed
    pub cilia_signal_channel: i32, // Which signal channel to read (0-7, sensory channels)
    pub cilia_speed_below: f32, // Cilia speed when signal < threshold
    pub cilia_speed_above: f32, // Cilia speed when signal >= threshold
    pub cilia_threshold: f32, // Signal threshold for speed switching
    pub cilia_attract_force: f32, // Gentle attraction force pulling nearby non-organism cells toward the ciliocyte (0.0 = off, 1.0 = max); helps convey cells along a ciliocyte chain

    // Membrane settings
    pub membrane_stiffness: f32, // Cell membrane stiffness for collision response (0.0 = no repulsion, higher = more rigid)

    // Myocyte settings
    pub myocyte_contraction: f32, // Contraction amount during active pulse phase (0.0 = no contraction, 1.0 = full contraction)
    pub myocyte_use_signal: bool, // If true, contraction is signal-driven; if false, uses phased timer
    pub myocyte_signal_channel: i32, // Which signal channel to read (0-15)
    pub myocyte_contraction_above: f32, // Contraction amount when signal >= threshold (0.0 to 1.0)
    pub myocyte_contraction_below: f32, // Contraction amount when signal < threshold (0.0 to 1.0)
    pub myocyte_threshold: f32,   // Signal threshold for contraction switching
    pub myocyte_pulse_rate: f32,  // Pulse oscillation rate in cycles per second (0.1 to 10.0)
    pub myocyte_pulse_phase: i32, // Which pulse phase to contract on (0 = Pulse A, 1 = Pulse B)

    // Embryocyte settings
    // Release triggers (AND logic): all enabled triggers must be satisfied simultaneously
    pub embryocyte_use_timer: bool,      // Enable timer trigger
    pub embryocyte_release_timer: f32,   // Seconds since cell creation before timer is satisfied
    pub embryocyte_use_threshold: bool,  // Enable reserve threshold trigger
    pub embryocyte_threshold_value: u32, // Reserve must be >= this value (0-65535)
    pub embryocyte_use_signal: bool,     // Enable signal trigger
    pub embryocyte_signal_channel: i32,  // Signal channel to monitor (0-15)
    pub embryocyte_signal_value: f32,    // Minimum signal value required for trigger

    // Developmental/regulation signal emission: any cell mode can emit on channels 8-15
    pub regulation_emit_channel: i32, // Channel to emit on (-1 = disabled, 8-15 = regulation/developmental channel)
    pub regulation_emit_value: f32,   // Signal value to emit (0.0 to 2047.0)
    pub regulation_emit_hops: i32,    // How many adhesion hops the signal propagates (1-20)

    // Signal-conditional behavior settings
    // Division gating: cell only divides if signal condition is met
    pub division_signal_channel: i32, // Signal channel to check (-1 = disabled, 0-15 = channel)
    pub division_signal_threshold: f32, // Signal value threshold for division
    pub division_signal_invert: bool, // If true, divide when signal BELOW threshold instead of above

    // Apoptosis: signal-triggered cell death
    pub apoptosis_signal_channel: i32, // Signal channel to check (-1 = disabled, 0-15 = channel)
    pub apoptosis_signal_threshold: f32, // Signal value threshold for death
    pub apoptosis_signal_invert: bool, // If true, die when signal BELOW threshold instead of above

    // Signal-conditional child mode routing: override child mode based on signal state at division
    pub signal_child_a_channel: i32, // Signal channel to check (-1 = disabled)
    pub signal_child_a_threshold: f32, // Threshold for mode override
    pub signal_child_a_mode_above: i32, // Mode index when signal >= threshold (-1 = use default)
    pub signal_child_a_mode_below: i32, // Mode index when signal < threshold (-1 = use default)
    pub signal_child_b_channel: i32, // Signal channel to check (-1 = disabled)
    pub signal_child_b_threshold: f32, // Threshold for mode override
    pub signal_child_b_mode_above: i32, // Mode index when signal >= threshold (-1 = use default)
    pub signal_child_b_mode_below: i32, // Mode index when signal < threshold (-1 = use default)

    // Mode switching without division: signal-triggered mode transition
    pub mode_switch_signal_channel: i32, // Signal channel to check (-1 = disabled)
    pub mode_switch_signal_threshold: f32, // Threshold for mode switch
    pub mode_switch_target: i32,         // Target mode index (-1 = disabled)
    pub mode_switch_invert: bool,        // If true, switch when signal BELOW threshold

    // Devorocyte settings
    pub devorocyte_consume_range: f32, // Extra contact range beyond cell radii (0.0 to 2.0)
    pub devorocyte_consume_rate: f32, // Nutrients stolen per second from each victim (1.0 to 100.0)

    // Vasculocyte settings
    pub vascular_nutrient_transport: bool, // When true, this mode participates in vascular nutrient pipes
    pub vascular_outlet: bool, // Nutrient exchange port: exchanges with non-vascular neighbors in both directions
    pub vascular_signal_transport: bool, // When true, this mode participates in vascular signal pipes
    pub vascular_signal_exchange: bool, // Signal exchange port: exchanges with non-vascular neighbors in both directions
    pub vascular_signal_capacity: f32, // Node-level throughput cap prevents fan-in amplification

    // Gametocyte settings
    pub gametocyte_merge_range: f32, // Extra contact range for merge detection beyond cell radii (0.0 to 2.0)

    // Memorocyte settings
    /// Fraction of the gap between current memory and input closed per second (0.0-1.0).
    /// 0.0 = never tracks input, 1.0 = instant snap. Steady-state converges to input - never amplifies.
    pub memorocyte_rate: f32,
    pub memorocyte_input_channel: i32,
    /// Channel to emit the current memory value on (0-15).
    pub memorocyte_output_channel: i32,
    /// How many adhesion hops the emitted memory propagates (1-20).
    pub memorocyte_output_hops: i32,

    // Cognocyte settings
    /// Which arithmetic/logic operation to perform on the two input signals.
    /// 0=Add, 1=Subtract, 2=Multiply, 3=Divide, 4=Min, 5=Max, 6=Average,
    /// 7=GreaterThan, 8=LessThan, 9=Equal, 10=AND, 11=OR, 12=NOT, 13=Select
    pub cognocyte_operation: i32,
    /// First input signal channel (0-15). Emits nothing if channel has no signal.
    pub cognocyte_input_channel_a: i32,
    /// Second input signal channel (0-15). Emits nothing if channel has no signal.
    /// Unused for NOT (which only uses A).
    pub cognocyte_input_channel_b: i32,
    /// Channel to emit the result on (0-15).
    pub cognocyte_output_channel: i32,
    /// How many adhesion hops the result signal propagates (1-20).
    pub cognocyte_output_hops: i32,

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
            opacity: 1.0,  // Default: fully opaque
            emissive: 0.0, // Default: no glow
            cell_type: 0,
            parent_make_adhesion: false,
            split_mass: 1.5,
            split_interval: 1.0,
            nutrient_gain_rate: 10.0,
            max_cell_size: 2.0,        // Default: max size of 2.0 units
            split_ratio: 0.5,          // Default: 50/50 split
            nutrient_priority: 1.0,    // Default: neutral priority
            prioritize_when_low: true, // Default: protect cells from death
            parent_split_direction: Vec2::ZERO,
            max_adhesions: 20,
            min_adhesions: 0, // No minimum by default
            enable_parent_angle_snapping: true,
            max_splits: -1,                                  // Infinite by default
            mode_a_after_splits: -1,                         // Use normal child_a mode by default
            mode_b_after_splits: -1,                         // Use normal child_b mode by default
            child_a_after_split_orientation: Quat::IDENTITY, // Default orientation for Child A after max splits
            child_b_after_split_orientation: Quat::IDENTITY, // Default orientation for Child B after max splits
            child_a_after_split_keep_adhesion: true, // Default: keep adhesion for Child A after max splits
            child_b_after_split_keep_adhesion: true, // Default: keep adhesion for Child B after max splits
            glueocyte_cell_adhesion: false,          // Default: cell adhesion disabled
            glueocyte_self_adhesion: false,          // Default: don't bond to own organism
            glueocyte_env_adhesion: true,            // Default: environment adhesion enabled
            glueocyte_boulder_adhesion: true,        // Default: boulder adhesion enabled
            glueocyte_cell_adhesion_signal_channel: -1, // Default: always active (no signal gate)
            glueocyte_cell_adhesion_signal_threshold: 1.0,
            glueocyte_signal_gate_invert: false, // Default: active when signal >= threshold
            swim_force: 0.5,                     // Default swim force for flagellocytes
            flagellocyte_use_signal: false,      // Default: fixed speed mode
            flagellocyte_signal_channel: 0,      // Default: channel 0
            flagellocyte_speed_a: 0.5,           // Default: same as swim_force
            flagellocyte_speed_b: 0.0,           // Default: stop when signal received
            flagellocyte_threshold_c: 1.0,       // Default: threshold of 1.0
            cilia_speed: 0.5,                    // Default cilia speed for ciliocytes
            cilia_push_bonded: false,            // Default: don't push same-organism cells
            cilia_use_signal: false,             // Default: fixed speed mode
            cilia_signal_channel: 0,             // Default: channel 0
            cilia_speed_below: 0.5,              // Default: same as cilia_speed
            cilia_speed_above: 0.0,              // Default: stop when signal received
            cilia_threshold: 1.0,                // Default: threshold of 1.0
            cilia_attract_force: 0.0,            // Default: no attraction (off)
            buoyancy_force: 0.5,                 // Default buoyancy force for buoyocytes
            oculocyte_sense_type: 1,             // Default: sense cells (bit 0)
            oculocyte_signal_channel: 0,         // Default: channel 0
            oculocyte_signal_value: 10.0,        // Default: +10 signal
            oculocyte_signal_hops: 3,            // Default: 3 hops
            oculocyte_ray_length: 20.0,          // Default: 20 units ray length
            membrane_stiffness: 250.0,           // Default: moderate membrane stiffness
            myocyte_contraction: 0.5,            // Default: 50% contraction
            myocyte_use_signal: false, // Default: pulse timer mode (works without signals)
            myocyte_signal_channel: 0, // Default: channel 0
            myocyte_contraction_above: 0.5, // Default: contract when signal received
            myocyte_contraction_below: 0.0, // Default: relaxed when no signal
            myocyte_threshold: 1.0,    // Default: threshold of 1.0
            myocyte_pulse_rate: 1.0,   // Default: 1 cycle per second
            myocyte_pulse_phase: 0,    // Default: pulse A
            embryocyte_use_timer: false, // Default: no timer trigger
            embryocyte_release_timer: 10.0, // Default: 10 seconds
            embryocyte_use_threshold: false, // Default: no threshold trigger
            embryocyte_threshold_value: 32768, // Default: half full (32768)
            embryocyte_use_signal: false, // Default: no signal trigger
            embryocyte_signal_channel: 0, // Default: channel 0
            embryocyte_signal_value: 1.0, // Default: threshold of 1.0
            regulation_emit_channel: -1, // Disabled by default
            regulation_emit_value: 10.0, // Default: 10 signal value
            regulation_emit_hops: 3,   // Default: 3 hops
            division_signal_channel: -1, // Disabled by default
            division_signal_threshold: 1.0,
            division_signal_invert: false,
            apoptosis_signal_channel: -1, // Disabled by default
            apoptosis_signal_threshold: 1.0,
            apoptosis_signal_invert: false,
            signal_child_a_channel: -1, // Disabled by default
            signal_child_a_threshold: 1.0,
            signal_child_a_mode_above: -1,
            signal_child_a_mode_below: -1,
            signal_child_b_channel: -1, // Disabled by default
            signal_child_b_threshold: 1.0,
            signal_child_b_mode_above: -1,
            signal_child_b_mode_below: -1,
            mode_switch_signal_channel: -1, // Disabled by default
            mode_switch_signal_threshold: 1.0,
            mode_switch_target: -1,
            mode_switch_invert: false,
            devorocyte_consume_range: 0.5,
            devorocyte_consume_rate: 30.0,
            vascular_nutrient_transport: true,
            vascular_outlet: false,
            photocyte_emit_enabled: false,
            photocyte_emit_channel: 0,
            photocyte_emit_hops: 5,
            photocyte_emit_threshold: 0.5,
            photocyte_emit_mode: 0,
            photocyte_emit_value: 10.0,
            lipocyte_emit_enabled: false,
            lipocyte_emit_channel: 0,
            lipocyte_emit_hops: 5,
            lipocyte_emit_threshold: 0.8,
            lipocyte_emit_mode: 1,
            lipocyte_emit_value: 10.0,
            vascular_signal_transport: false,
            vascular_signal_exchange: false,
            vascular_signal_capacity: 10.0,
            gametocyte_merge_range: 0.5,
            memorocyte_rate: 0.1,
            memorocyte_input_channel: 0,
            memorocyte_output_channel: 9,
            memorocyte_output_hops: 5,
            cognocyte_operation: 0, // Add
            cognocyte_input_channel_a: 0,
            cognocyte_input_channel_b: 1,
            cognocyte_output_channel: 8,
            cognocyte_output_hops: 5,
            child_a: ChildSettings::default(),
            child_b: ChildSettings::default(),
            adhesion_settings: AdhesionSettings::default(),
        }
    }
}

impl Default for Genome {
    /// Creates a genome with 10 default modes.
    ///
    /// All modes are Phagocyte. This is the serialization
    /// baseline - `from_serializable` starts from this and applies diffs.
    ///
    /// Old `.genome` files still load correctly because `from_serializable`
    /// extends or trims the vec to match the saved `mode_count`.
    fn default() -> Self {
        let mut genome = Self {
            name: "Untitled Genome".to_string(),
            initial_mode: 0,
            initial_orientation: Quat::IDENTITY,
            modes: Vec::with_capacity(10),
        };

        for i in 0..10 {
            let mode_name = format!("M {}", i + 1);
            let hue = (i as f32 / 10.0) * 360.0;
            let (r, g, b) = hue_to_rgb(hue);
            let mut mode = ModeSettings {
                name: mode_name.clone(),
                default_name: mode_name,
                cell_type: 2, // All modes = Phagocyte
                color: Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0),
                ..Default::default()
            };
            mode.child_a.mode_number = i as i32;
            mode.child_b.mode_number = i as i32;
            genome.modes.push(mode);
        }

        genome
    }
}

impl Genome {
    /// Returns the path to the genomes folder in the user's Documents directory,
    /// creating it if it doesn't exist.
    /// This is the canonical location for saving and loading `.genome` files.
    pub fn genomes_dir() -> std::path::PathBuf {
        crate::app_dirs::genomes_dir()
    }

    /// Collect all `.genome` files from the genomes folder, sorted for determinism.
    /// Returns an empty vec if the folder doesn't exist or has no genome files.
    pub fn list_genomes_dir() -> Vec<std::path::PathBuf> {
        let dir = Self::genomes_dir();
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("genome"))
            .collect();
        paths.sort(); // deterministic order so index selection is stable
        paths
    }

    /// Load a genome from the genomes folder by index (wraps around).
    /// `seed` is mixed with the index so two calls with different seeds pick different files.
    /// Returns `None` if the folder has no `.genome` files.
    pub fn load_from_genomes_dir_at(seed: u64) -> Option<Self> {
        let paths = Self::list_genomes_dir();
        if paths.is_empty() {
            return None;
        }
        // Use a hash mix that avoids the modulo-bias of a single LCG step.
        // splitmix64: well-distributed across all output bits.
        let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        let idx = (x as usize) % paths.len();
        let path = &paths[idx];
        log::info!(
            "Main menu selecting genome index {}/{}: {:?}",
            idx,
            paths.len(),
            path
        );
        match Self::load_from_file(path) {
            Ok(genome) => {
                log::info!("Main menu loaded genome from {:?}", path);
                Some(genome)
            }
            Err(e) => {
                log::warn!("Failed to load genome {:?}: {}", path, e);
                None
            }
        }
    }

    /// Create a new genome with a single mode and a random color.
    /// Use this for user-facing "new genome" creation.
    /// Do NOT use this as a serialization baseline - `Default::default()` must stay deterministic.
    pub fn new_with_random_colors() -> Self {
        let mut genome = Self::default();

        // Randomize the single starting mode's color
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(12345);
        let mut rng = seed as u64;

        for mode in &mut genome.modes {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((rng >> 33) & 0xFF) as f32 / 255.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let g = ((rng >> 33) & 0xFF) as f32 / 255.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((rng >> 33) & 0xFF) as f32 / 255.0;
            mode.color = Vec3::new(r, g, b);
        }

        genome
    }

    /// Create a genome pre-populated with `n` modes (clamped to `MAX_MODES`).
    /// Used by `generate_procedural` which needs a fixed slot pool to assign roles into.
    /// Each mode gets a deterministic hue spread and self-referencing children.
    pub fn new_with_mode_count(n: usize) -> Self {
        let count = n.min(MAX_MODES).max(1);
        let mut genome = Self {
            name: "Untitled Genome".to_string(),
            initial_mode: 0,
            initial_orientation: Quat::IDENTITY,
            modes: Vec::with_capacity(count),
        };

        // Randomize colors using LCG seeded from system time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(12345);
        let mut rng = seed as u64;

        for i in 0..count {
            let mode_name = format!("M {}", i + 1);
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((rng >> 33) & 0xFF) as f32 / 255.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let g = ((rng >> 33) & 0xFF) as f32 / 255.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((rng >> 33) & 0xFF) as f32 / 255.0;

            let mut mode = ModeSettings {
                name: mode_name.clone(),
                default_name: mode_name,
                color: Vec3::new(r, g, b),
                cell_type: 2, // Phagocyte default
                ..Default::default()
            };
            mode.child_a.mode_number = i as i32;
            mode.child_b.mode_number = i as i32;
            genome.modes.push(mode);
        }

        genome
    }

    fn next_mode_label(&self) -> String {
        let next_number = self
            .modes
            .iter()
            .filter_map(|mode| {
                let trimmed = mode.name.trim();
                let number = trimmed
                    .strip_prefix('M')
                    .or_else(|| trimmed.strip_prefix('m'))?
                    .trim();
                number.parse::<usize>().ok()
            })
            .max()
            .unwrap_or(self.modes.len())
            + 1;
        format!("M {}", next_number)
    }

    fn random_mode_color(seed_index: usize) -> Vec3 {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(12345) as u64;
        let mut rng = seed.wrapping_add((seed_index as u64).wrapping_mul(0x9e3779b97f4a7c15));
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((rng >> 33) & 0xFF) as f32 / 255.0;
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let g = ((rng >> 33) & 0xFF) as f32 / 255.0;
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let b = ((rng >> 33) & 0xFF) as f32 / 255.0;

        Vec3::new(r, g, b)
    }

    fn make_inserted_mode(&self, idx: usize) -> ModeSettings {
        let mode_name = self.next_mode_label();
        let mut mode = ModeSettings {
            name: mode_name.clone(),
            default_name: mode_name,
            color: Self::random_mode_color(idx),
            cell_type: 2, // Phagocyte - all modes after the first default to Phagocyte
            ..Default::default()
        };
        mode.child_a.mode_number = idx as i32;
        mode.child_b.mode_number = idx as i32;
        mode
    }

    fn shift_mode_references_after_insert(&mut self, insert_idx: usize) {
        let remap = |idx: i32| {
            if idx >= insert_idx as i32 {
                idx + 1
            } else {
                idx
            }
        };

        self.initial_mode = remap(self.initial_mode);
        for mode in &mut self.modes {
            mode.child_a.mode_number = remap(mode.child_a.mode_number);
            mode.child_b.mode_number = remap(mode.child_b.mode_number);
            mode.mode_a_after_splits = remap(mode.mode_a_after_splits);
            mode.mode_b_after_splits = remap(mode.mode_b_after_splits);
            mode.signal_child_a_mode_above = remap(mode.signal_child_a_mode_above);
            mode.signal_child_a_mode_below = remap(mode.signal_child_a_mode_below);
            mode.signal_child_b_mode_above = remap(mode.signal_child_b_mode_above);
            mode.signal_child_b_mode_below = remap(mode.signal_child_b_mode_below);
            if mode.mode_switch_target >= 0 {
                mode.mode_switch_target = remap(mode.mode_switch_target);
            }
        }
    }

    /// Add a new mode to this genome, returning its index.
    /// Returns `None` if the genome is already at `MAX_MODES`.
    /// The new mode gets a random color and self-referencing children.
    pub fn add_mode(&mut self) -> Option<usize> {
        let after_index = self.modes.len().saturating_sub(1);
        self.insert_mode_after(after_index)
    }

    /// Insert a new mode immediately after `after_index`, returning its new index.
    /// Existing mode names are preserved, so labels stay stable across insertion.
    pub fn insert_mode_after(&mut self, after_index: usize) -> Option<usize> {
        if self.modes.len() >= MAX_MODES {
            return None;
        }

        let insert_idx = (after_index + 1).min(self.modes.len());
        self.shift_mode_references_after_insert(insert_idx);
        let mode = self.make_inserted_mode(insert_idx);
        self.modes.insert(insert_idx, mode);
        Some(insert_idx)
    }

    /// Remove selected modes while preserving the initial mode.
    /// Surviving references shift with their target. References to removed child
    /// modes become self-references; other removed targets revert to disabled defaults.
    /// Returns removed indices in old-index space.
    pub fn remove_modes_except_initial(&mut self, selected_indices: &[usize]) -> Vec<usize> {
        if self.modes.len() <= 1 {
            return Vec::new();
        }

        let initial_idx = self
            .initial_mode
            .max(0)
            .min(self.modes.len().saturating_sub(1) as i32) as usize;
        let mut removed: Vec<usize> = selected_indices
            .iter()
            .copied()
            .filter(|&idx| idx < self.modes.len() && idx != initial_idx)
            .collect();
        removed.sort_unstable();
        removed.dedup();

        if removed.is_empty() {
            return Vec::new();
        }

        let old_len = self.modes.len();
        let mut remap_table = vec![0usize; old_len];
        let mut next_idx = 0usize;
        for (old_idx, slot) in remap_table.iter_mut().enumerate() {
            if removed.binary_search(&old_idx).is_ok() {
                continue;
            }
            *slot = next_idx;
            next_idx += 1;
        }

        let selected_for_reset = removed.clone();
        let defaults = ModeSettings::default();
        for (old_idx, mode) in self.modes.iter_mut().enumerate() {
            if selected_for_reset.binary_search(&old_idx).is_ok() {
                continue;
            }
            let new_self_idx = remap_table[old_idx] as i32;
            let child_remap = |idx: i32| -> Option<i32> {
                if idx < 0 || idx as usize >= old_len {
                    return Some(idx);
                }
                if selected_for_reset.binary_search(&(idx as usize)).is_ok() {
                    None
                } else {
                    Some(remap_table[idx as usize] as i32)
                }
            };

            mode.child_a.mode_number = child_remap(mode.child_a.mode_number).unwrap_or(new_self_idx);
            mode.child_b.mode_number = child_remap(mode.child_b.mode_number).unwrap_or(new_self_idx);

            if let Some(target) = child_remap(mode.mode_a_after_splits) {
                mode.mode_a_after_splits = target;
            } else {
                mode.mode_a_after_splits = defaults.mode_a_after_splits;
                mode.child_a_after_split_orientation = defaults.child_a_after_split_orientation;
                mode.child_a_after_split_keep_adhesion = defaults.child_a_after_split_keep_adhesion;
            }
            if let Some(target) = child_remap(mode.mode_b_after_splits) {
                mode.mode_b_after_splits = target;
            } else {
                mode.mode_b_after_splits = defaults.mode_b_after_splits;
                mode.child_b_after_split_orientation = defaults.child_b_after_split_orientation;
                mode.child_b_after_split_keep_adhesion = defaults.child_b_after_split_keep_adhesion;
            }

            let signal_child_a_above = child_remap(mode.signal_child_a_mode_above);
            let signal_child_a_below = child_remap(mode.signal_child_a_mode_below);
            if let (Some(above), Some(below)) = (signal_child_a_above, signal_child_a_below) {
                mode.signal_child_a_mode_above = above;
                mode.signal_child_a_mode_below = below;
            } else {
                mode.signal_child_a_channel = defaults.signal_child_a_channel;
                mode.signal_child_a_threshold = defaults.signal_child_a_threshold;
                mode.signal_child_a_mode_above = defaults.signal_child_a_mode_above;
                mode.signal_child_a_mode_below = defaults.signal_child_a_mode_below;
            }

            let signal_child_b_above = child_remap(mode.signal_child_b_mode_above);
            let signal_child_b_below = child_remap(mode.signal_child_b_mode_below);
            if let (Some(above), Some(below)) = (signal_child_b_above, signal_child_b_below) {
                mode.signal_child_b_mode_above = above;
                mode.signal_child_b_mode_below = below;
            } else {
                mode.signal_child_b_channel = defaults.signal_child_b_channel;
                mode.signal_child_b_threshold = defaults.signal_child_b_threshold;
                mode.signal_child_b_mode_above = defaults.signal_child_b_mode_above;
                mode.signal_child_b_mode_below = defaults.signal_child_b_mode_below;
            }

            if let Some(target) = child_remap(mode.mode_switch_target) {
                mode.mode_switch_target = target;
            } else {
                mode.mode_switch_signal_channel = defaults.mode_switch_signal_channel;
                mode.mode_switch_signal_threshold = defaults.mode_switch_signal_threshold;
                mode.mode_switch_target = defaults.mode_switch_target;
                mode.mode_switch_invert = defaults.mode_switch_invert;
            }
        }

        for &idx in removed.iter().rev() {
            self.modes.remove(idx);
        }
        self.initial_mode = remap_table[initial_idx] as i32;

        removed
    }

    /// Remove the last mode from this genome.
    /// Returns `false` if the genome only has 1 mode or the last mode is initial.
    pub fn remove_last_mode(&mut self) -> bool {
        if self.modes.len() <= 1 {
            return false;
        }
        let removed_idx = self.modes.len() - 1;
        !self.remove_modes_except_initial(&[removed_idx]).is_empty()
    }
    pub fn generate_procedural(seed: u64) -> Self {
        // -- Seeded RNG (splitmix64 + LCG) -----------------------------------
        struct Rng(u64);
        #[allow(dead_code)]
        impl Rng {
            fn new(s: u64) -> Self {
                Self(s ^ 0xcafebabe_deadbeef)
            }
            fn step(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = self.0 ^ (self.0 >> 30);
                let x = x.wrapping_mul(0xbf58476d1ce4e5b9);
                let x = x ^ (x >> 27);
                let x = x.wrapping_mul(0x94d049bb133111eb);
                x ^ (x >> 31)
            }
            fn u32(&mut self, max: u32) -> u32 {
                if max == 0 {
                    return 0;
                }
                (self.step() >> 33) as u32 % max
            }
            fn f32(&mut self, lo: f32, hi: f32) -> f32 {
                let t = ((self.step() >> 33) as f32) / (0xFFFF_FFFFu64 as f32);
                lo + t * (hi - lo)
            }
            fn bool(&mut self, probability: f32) -> bool {
                self.f32(0.0, 1.0) < probability
            }
            fn pick<'a, T>(&mut self, s: &'a [T]) -> &'a T {
                &s[self.u32(s.len() as u32) as usize]
            }
            fn i32_range(&mut self, lo: i32, hi: i32) -> i32 {
                lo + self.u32((hi - lo) as u32) as i32
            }
            fn quat_axis_angle(&mut self, ax: f32, ay: f32, az: f32, deg: f32) -> Quat {
                let half = deg.to_radians() * 0.5;
                let s = half.sin();
                Quat::from_xyzw(ax * s, ay * s, az * s, half.cos())
            }
        }

        let mut rng = Rng::new(seed);
        // Pre-allocate exactly as many modes as the generator will use (8-12 roles).
        // We don't know num_roles yet, so allocate the maximum (13) and trim unused
        // slots at the end. This keeps the genome lean vs the old 80-mode allocation.
        let mut genome = Self::new_with_mode_count(13);

        // -- Name -------------------------------------------------------------
        let label = (rng.step() & 0xFFFF) as u16;
        genome.name = format!("Creature {:04X}", label);

        //
        // STEP 1: ASSIGN ROLES TO RANDOM MODE SLOTS
        //
        // Roles are functional identities. Mode indices are just storage slots.
        // The generator picks which slot each role lives in, then wires them
        // together with signal channels. Nothing is locked to a fixed index.
        //
        let num_roles: usize = rng.i32_range(8, 13) as usize;
        // Trim the genome to exactly num_roles modes (we pre-allocated 13 above)
        genome.modes.truncate(num_roles);
        let mut slots: Vec<usize> = (0..num_roles).collect();
        for i in (1..slots.len()).rev() {
            let j = rng.u32(i as u32 + 1) as usize;
            slots.swap(i, j);
        }

        let r_zygote: usize = slots[0];
        let r_stem: usize = slots[1];
        let r_struct_a: usize = slots[2];
        let r_struct_b: usize = slots[3];
        let r_feeder: usize = slots[4];
        let r_loco: usize = slots[5];
        let r_specialist: usize = slots[6];
        let r_gonad: usize = slots[7];

        let r_sensor: Option<usize> = if num_roles > 8 { Some(slots[8]) } else { None };
        let r_adult_struct: Option<usize> = if num_roles > 9 { Some(slots[9]) } else { None };
        let r_extra_spec: Option<usize> = if num_roles > 10 {
            Some(slots[10])
        } else {
            None
        };
        let r_anchor: Option<usize> = if num_roles > 11 {
            Some(slots[11])
        } else {
            None
        };

        genome.initial_mode = r_stem as i32;

        //
        // STEP 2: INDEPENDENT TRAIT SELECTION
        //

        // Locomotion: 1=flagella  2=cilia  3=myocyte  4=buoyancy  5=none
        let loco_cell_type: i32 = *rng.pick(&[1i32, 1, 2, 3, 4, 5]);

        // Feeding: 2=phagocyte  3=photocyte  11=devorocyte
        let feed_cell_type: i32 = *rng.pick(&[2i32, 2, 3, 3, 11]);

        // Specialist terminal function
        let spec_cell_type: i32 = *rng.pick(&[2i32, 4, 6, 8, 9, 11]);

        // Body stiffness 0=soft .. 1=rigid
        let stiffness: f32 = rng.f32(0.0, 1.0);
        let adh_rest = rng.f32(2.0, 4.5) * (1.0 - stiffness * 0.4);
        let adh_lin = 80.0 + stiffness * 240.0;
        let adh_ang = 15.0 + stiffness * 85.0;
        let membrane = 120.0 + stiffness * 280.0;
        let flex = stiffness < 0.35;

        // -- GEOMETRY ---------------------------------------------------------
        // Body plans built via flat ring + axial extrusion:
        //
        //   STEM (1D chain)
        //     Splits forward in its own frame. Child A becomes perpendicular
        //     to spine (rotated 90 deg around local X) to seed a ring. Child B
        //     continues the spine identity.
        //
        //   STRUCT_A (2D ring around spine)
        //     Each cell is perpendicular to spine. Splits forward (outward
        //     from spine in its own frame). Child A is the next ring cell,
        //     rotated by ring_angle around its local Y axis - which after the
        //     90 deg X rotation from stem corresponds to the spine axis in world
        //     space, so child A ends up rotated around the spine. Child B
        //     redirects back to spine direction (rotated -90 deg around X) and
        //     becomes the extrusion seed.
        //
        //   STRUCT_B (1D extrusion along spine)
        //     Now oriented along spine direction. Splits forward (along spine
        //     axis) extruding a column. Child A continues the column, child B
        //     terminates as specialist (controlled via mode_b_after_splits).
        //
        // All splits use parent_split_direction = ZERO (split along local
        // forward). All geometry comes from child orientations relative to
        // each cell's local frame. This is the same pattern used by the
        // Triangular Prism and Octo-Tube player genomes.
        //
        // The free parameters:
        //   ring_angle    - angle between adjacent ring cells (discrete: 60/72/90/120/180)
        //                   180 = bilateral pair, 120 = triangle, 90 = square,
        //                   72 = pentagon, 60 = hexagon
        //   ring_segments - how many cells form the ring (derived from ring_angle)
        //   extrude_segs  - how many cells extrude along the spine per ring position
        //   pattern_solid - if true, each ring position extrudes (full prism);
        //                   if false, only some extrude (sparse skeleton)

        // Discrete ring angle for clean polygons
        let ring_angle: f32 = *rng.pick(&[60.0f32, 72.0, 90.0, 120.0, 180.0]);
        // Number of cells in the ring is 360/ring_angle
        let ring_segments = (360.0 / ring_angle).round() as i32;
        // How many cells extrude per ring position (small = sheet, larger = prism)
        let extrude_segs: i32 = rng.i32_range(1, 4);
        // Whether every ring cell extrudes, or only one in the middle (less dense)
        let pattern_solid = rng.bool(0.7);

        // Body axis orientation in world space (just a starting rotation).
        // After this, all geometry is local-frame relative.
        let body_pitch = rng.f32(-30.0, 30.0);
        let body_yaw = rng.f32(0.0, 360.0);

        // Stem child A: 90 deg around local X - child becomes perpendicular to spine.
        let q_perpendicular = rng.quat_axis_angle(1.0, 0.0, 0.0, 90.0);
        // Struct_a child A: ring_angle around local Y - rotates next ring cell
        // around the spine axis (which after the 90 deg X rotation is the local Y).
        let q_ring = rng.quat_axis_angle(0.0, 1.0, 0.0, ring_angle);
        let q_ring_mirror = rng.quat_axis_angle(0.0, 1.0, 0.0, -ring_angle);
        // Struct_a child B: -90 deg around local X - redirects forward back along
        // the spine direction so struct_b can extrude axially.
        let q_extrude_seed = rng.quat_axis_angle(1.0, 0.0, 0.0, -90.0);

        // All split directions are ZERO - split along local forward.
        // Body orientation goes on the stem only as a starting rotation.
        let stem_split_dir = Vec2::new(body_pitch, body_yaw);
        let struct_a_split_dir = Vec2::ZERO;
        let struct_b_split_dir = Vec2::ZERO;

        // Convenience aliases for the wiring section
        let stem_child_a_q = q_perpendicular;
        let _stem_child_a_mirror_q = q_perpendicular; // ring is symmetric, no mirror needed
        let struct_a_child_a_q = q_ring;
        let struct_a_child_a_mirror_q = q_ring_mirror;
        // Struct_a child B is the extrusion seed (redirects to spine direction)
        let struct_a_child_b_q = q_extrude_seed;
        // Struct_b children: identity - extrude straight along spine
        let struct_b_child_a_q = Quat::IDENTITY;
        let struct_b_child_b_q = Quat::IDENTITY;

        // Stem mirror: alternates ring start direction on consecutive spine
        // segments so adjacent rings are offset (looks more organic).
        let _stem_mirror = rng.bool(0.5);
        let struct_a_mirror = !pattern_solid; // sparse pattern uses alternating sides

        // Override branch_segs to match ring_segments so the ring closes cleanly
        let branch_segs: i32 = ring_segments - 1; // each split adds one cell, so N-1 splits = N cells
        let branch_ext: i32 = extrude_segs;
        let spine_segs: i32 = rng.i32_range(3, 9);

        // Cell sizes - strict gradient: stem > spine > branch > spec
        let stem_size: f32 = rng.f32(1.7, 2.0);
        let spine_size: f32 = stem_size * rng.f32(0.75, 0.90);
        let branch_size: f32 = spine_size * rng.f32(0.70, 0.88);
        let spec_size: f32 = branch_size * rng.f32(0.55, 0.75);

        // Timing - axial tissue grows faster than lateral, lateral faster than terminal
        let stem_mass: f32 = rng.f32(1.2, 1.6);
        let stem_ivl: f32 = rng.f32(0.3, 0.7);
        let spine_mass: f32 = rng.f32(1.3, 1.7);
        let spine_ivl: f32 = rng.f32(0.4, 0.9);
        let branch_mass: f32 = rng.f32(1.4, 1.9);
        let branch_ivl: f32 = rng.f32(0.5, 1.1);
        let spec_mass: f32 = rng.f32(1.5, 2.1);
        let spec_ivl: f32 = rng.f32(0.8, 1.6);

        // Reproduction
        let num_eggs: i32 = rng.i32_range(2, 6);

        //
        // STEP 3: SIGNAL CHANNEL ASSIGNMENTS
        //
        // Channels 8-15 are developmental/regulation (any cell can emit/receive).
        // Channels 0-7 are sensory (detection signals: oculocyte, photocyte, lipocyte, etc.).
        //
        // Shuffled so different creatures use different channels for the same
        // roles - prevents cross-talk when multiple creatures share a world.
        //
        let mut ch_pool: Vec<i32> = (8..=15).collect();
        for i in (1..ch_pool.len()).rev() {
            let j = rng.u32(i as u32 + 1) as usize;
            ch_pool.swap(i, j);
        }
        let ch_anterior: i32 = ch_pool[0]; // emitted by stem, marks head end
        let ch_lateral: i32 = ch_pool[1]; // emitted by spine, marks axis distance
        let ch_feeder: i32 = ch_pool[2]; // emitted by feeder, gates stem division
        let ch_maturity: i32 = ch_pool[3]; // emitted by gonad, triggers grow->adult
        let _ch_repro: i32 = ch_pool[4]; // reserved for future egg-shedding gate

        // Sensory channel (0-7): detection cells emit here, locomotion cells read it
        let ch_sense: i32 = rng.i32_range(0, 8);

        //
        // STEP 4: HELPER CLOSURES
        //

        // Apply shared adhesion parameters
        fn apply_adhesion(m: &mut ModeSettings, rest: f32, lin: f32, ang: f32, flex: bool) {
            m.adhesion_settings.rest_length = rest;
            m.adhesion_settings.linear_spring_stiffness = lin;
            m.adhesion_settings.orientation_spring_stiffness = ang;
            m.adhesion_settings.linear_spring_damping = lin * 0.03;
            m.adhesion_settings.orientation_spring_damping = ang * 0.01;
            m.adhesion_settings.break_force = 1000.0;
            m.adhesion_settings.enable_twist_constraint = !flex;
        }

        // Apply cell-type-specific behaviour parameters
        fn apply_type(m: &mut ModeSettings, cell_type: i32, ch_sense: i32, rng: &mut Rng) {
            m.cell_type = cell_type;
            match cell_type {
                1 => {
                    // Flagellocyte: signal-gated speed
                    m.swim_force = rng.f32(0.8, 2.5);
                    m.flagellocyte_use_signal = true;
                    m.flagellocyte_signal_channel = ch_sense;
                    m.flagellocyte_speed_a = rng.f32(0.2, 0.8); // slow: no signal
                    m.flagellocyte_speed_b = rng.f32(1.0, 2.5); // fast: signal detected
                    m.flagellocyte_threshold_c = 1.0;
                }
                2 => {} // Phagocyte - no extra params
                3 => {} // Photocyte - no extra params
                4 => {
                    m.nutrient_priority = rng.f32(1.5, 3.0);
                } // Lipocyte
                5 => {
                    m.buoyancy_force = rng.f32(0.3, 0.8);
                } // Buoyocyte
                6 => {
                    // Glueocyte
                    m.glueocyte_env_adhesion = true;
                    m.glueocyte_cell_adhesion = false;
                }
                8 => {
                    // Ciliocyte: signal-gated speed
                    m.cilia_use_signal = true;
                    m.cilia_signal_channel = ch_sense;
                    m.cilia_speed_below = rng.f32(0.2, 0.6);
                    m.cilia_speed_above = rng.f32(0.7, 1.0);
                    m.cilia_threshold = 1.0;
                    m.cilia_push_bonded = false;
                    m.cilia_attract_force = rng.f32(0.0, 0.3);
                }
                9 => {
                    // Myocyte: signal-gated contraction
                    m.myocyte_use_signal = true;
                    m.myocyte_signal_channel = ch_sense;
                    m.myocyte_contraction_above = rng.f32(0.4, 0.8);
                    m.myocyte_contraction_below = rng.f32(0.0, 0.2);
                    m.myocyte_threshold = 1.0;
                    m.myocyte_pulse_rate = rng.f32(0.5, 2.0);
                    m.myocyte_pulse_phase = rng.u32(2) as i32;
                }
                10 => {
                    // Embryocyte
                    m.embryocyte_use_timer = true;
                    m.embryocyte_release_timer = rng.f32(3.0, 8.0);
                }
                11 => {
                    // Devorocyte
                    m.devorocyte_consume_range = rng.f32(1.5, 3.0);
                    m.devorocyte_consume_rate = rng.f32(30.0, 80.0);
                }
                12 => {
                    // Vasculocyte
                    m.vascular_nutrient_transport = true;
                    m.vascular_outlet = true;
                    m.nutrient_priority = 0.5;
                }
                _ => {}
            }
        }

        //
        // STEP 5: WIRE THE MODES
        //
        // Each mode is configured by its role. The mode index (r_stem, r_feeder,
        // etc.) is whatever the shuffle assigned - the wiring uses those indices
        // directly, so the body plan is fully determined by the signal graph,
        // not by slot positions.
        //

        // -- ZYGOTE (repurposed as Struct-C) ----------------------------------
        // The gonad now sheds stems directly, so the zygote slot is free.
        // Repurposed as a third structural tier: a smaller branch that struct_b
        // can optionally spawn as its child A, adding one more level of branching
        // depth. Configured as a self-renewing phagocyte with the same feeder
        // gate and maturity switch as struct_b. If never reached, it's harmless.
        {
            let m = &mut genome.modes[r_zygote];
            m.name = "Struct-C".to_string();
            m.cell_type = 2; // Phagocyte structural
            m.max_cell_size = branch_size * 0.8; // smaller than struct_b
            m.nutrient_priority = 0.9;
            m.split_mass = branch_mass * 1.1;
            m.split_interval = branch_ivl * 1.2;
            m.parent_make_adhesion = true;
            m.parent_split_direction = Vec2::ZERO;
            m.max_splits = branch_ext.max(1);
            m.enable_parent_angle_snapping = false;
            m.split_ratio = 0.5;
            m.membrane_stiffness = membrane * 0.75;
            m.max_adhesions = rng.i32_range(3, 7);
            // Same feeder gate as struct_b
            m.division_signal_channel = ch_feeder;
            m.division_signal_threshold = 0.5;
            // On maturity: becomes specialist
            m.mode_switch_signal_channel = ch_maturity;
            m.mode_switch_signal_threshold = 1.0;
            m.mode_switch_target = r_specialist as i32;
            apply_adhesion(m, adh_rest * 1.1, adh_lin * 0.6, adh_ang * 0.55, flex);
            // Self-renews until max_splits, then becomes specialist
            m.child_a.mode_number = r_zygote as i32;
            m.child_a.keep_adhesion = true;
            m.child_a.orientation = struct_b_child_a_q;
            m.child_b.mode_number = r_zygote as i32;
            m.child_b.keep_adhesion = true;
            m.child_b.orientation = struct_b_child_b_q;
            m.mode_a_after_splits = r_specialist as i32;
            m.mode_b_after_splits = r_specialist as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // -- STEM --------------------------------------------------------------
        // Self-renewing growth engine. Emits ch_anterior (head gradient).
        // First split always produces a feeder (child A = feeder when ch_feeder
        // is absent). Once the feeder is alive and emitting ch_feeder, subsequent
        // splits route child A to struct_a instead - bootstrapping is solved by
        // signal-conditional child routing.
        // Child B always = stem (self-renewal, IDENTITY orientation = straight spine).
        // After spine_segs splits -> both children become gonad.
        // On ch_maturity -> switches to feeder role in the adult body.
        {
            let m = &mut genome.modes[r_stem];
            m.name = "Stem".to_string();
            m.cell_type = 2; // Phagocyte
            m.max_cell_size = stem_size;
            m.nutrient_priority = 2.5;
            m.split_mass = stem_mass;
            m.split_interval = stem_ivl;
            m.parent_make_adhesion = true;
            m.parent_split_direction = stem_split_dir;
            m.max_splits = spine_segs;
            m.enable_parent_angle_snapping = false;
            m.split_ratio = 0.5;
            m.membrane_stiffness = membrane * 1.1;
            // Emit anterior gradient so all cells know their head-to-tail position
            m.regulation_emit_channel = ch_anterior;
            m.regulation_emit_value = 50.0;
            m.regulation_emit_hops = 20;
            // On maturity: stem becomes a feeder in the adult body
            m.mode_switch_signal_channel = ch_maturity;
            m.mode_switch_signal_threshold = 1.0;
            m.mode_switch_target = r_feeder as i32;
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Default child A = feeder (no feeder signal yet = bootstrap split)
            m.child_a.mode_number = r_feeder as i32;
            m.child_a.keep_adhesion = true;
            m.child_a.orientation = stem_child_a_q;
            // Once feeder is alive (ch_feeder present), child A -> struct_a instead
            m.signal_child_a_channel = ch_feeder;
            m.signal_child_a_threshold = 0.5;
            m.signal_child_a_mode_above = r_struct_a as i32; // fed -> grow structure
            m.signal_child_a_mode_below = r_feeder as i32; // not fed -> make feeder
                                                           // Child B = stem continues along spine axis (IDENTITY = straight)
            m.child_b.mode_number = r_stem as i32;
            m.child_b.keep_adhesion = true;
            m.child_b.orientation = Quat::IDENTITY;
            // After spine_segs: both become gonad
            m.mode_a_after_splits = r_gonad as i32;
            m.mode_b_after_splits = r_gonad as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // -- STRUCT_A (flat ring around spine - Tier 2) ------------------------
        // Each cell is perpendicular to the spine. Builds a flat ring/star/disc
        // by having child A continue around the spine (rotated by ring_angle).
        // Child B is the extrusion seed - redirects forward back along the spine
        // direction and becomes struct_b for axial extrusion.
        // Division gated on ch_feeder. Emits ch_lateral so extrusion cells
        // know their distance from the axis.
        // After branch_segs (= ring_segments - 1) splits, the ring closes and
        // both children transition to struct_b for the extrusion phase.
        // On ch_maturity -> switches to locomotion role.
        {
            let m = &mut genome.modes[r_struct_a];
            m.name = "Struct-A".to_string();
            m.cell_type = 2; // Phagocyte structural
            m.max_cell_size = spine_size;
            m.nutrient_priority = 1.5;
            m.split_mass = spine_mass;
            m.split_interval = spine_ivl;
            m.parent_make_adhesion = true;
            m.parent_split_direction = struct_a_split_dir; // ZERO = local forward
            m.max_splits = branch_segs;
            m.enable_parent_angle_snapping = false;
            m.split_ratio = 0.5;
            m.membrane_stiffness = membrane;
            m.division_signal_channel = ch_feeder;
            m.division_signal_threshold = 0.5;
            m.division_signal_invert = false;
            m.regulation_emit_channel = ch_lateral;
            m.regulation_emit_value = 30.0;
            m.regulation_emit_hops = 10;
            m.mode_switch_signal_channel = ch_maturity;
            m.mode_switch_signal_threshold = 1.0;
            m.mode_switch_target = r_loco as i32;
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Child A = struct_a, rotated by ring_angle around spine - builds the ring
            m.child_a.mode_number = r_struct_a as i32;
            m.child_a.keep_adhesion = true;
            m.child_a.orientation = if struct_a_mirror {
                struct_a_child_a_mirror_q // bilateral: alternating sides
            } else {
                struct_a_child_a_q // radial: same rotation each time
            };
            // Child B = struct_b, redirected back to spine direction for extrusion
            m.child_b.mode_number = r_struct_b as i32;
            m.child_b.keep_adhesion = true;
            m.child_b.orientation = struct_a_child_b_q; // q_extrude_seed
                                                        // After ring closes: both children become struct_b for extrusion
            m.mode_a_after_splits = r_struct_b as i32;
            m.mode_b_after_splits = r_struct_b as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // -- STRUCT_B (secondary structural / bilateral mirror) ----------------
        // Branches off struct_a. Self-extends branch_ext times.
        // Child orientations: struct_a_mirror controls whether children fan
        // symmetrically (bilateral) or continue in the same direction (radial).
        // Division gated on ch_feeder.
        //
        // FIX 5 - Positional specialist routing:
        // Tips near the head (high ch_anterior) -> r_specialist (primary function)
        // Tips near the tail (low ch_anterior)  -> r_extra_spec if present, else r_specialist
        // This gives the creature a head/tail functional distinction without
        // any hardcoded positions - purely signal-derived.
        // -- STRUCT_B (axial extrusion - Tier 3) -------------------------------
        // Already oriented along the spine direction (via struct_a_child_b_q).
        // Splits forward in its own frame, sweeping each ring cell into a column
        // parallel to the spine. Both children = struct_b with IDENTITY orientation
        // so the column extrudes straight. branch_ext (= extrude_segs) controls
        // how many cells deep the extrusion goes.
        // Division gated on ch_feeder.
        //
        // Positional routing at max_splits: head tips -> r_specialist,
        // tail tips -> r_extra_spec (if present). Gives head/tail differentiation.
        {
            let m = &mut genome.modes[r_struct_b];
            m.name = "Struct-B".to_string();
            m.cell_type = 2; // Phagocyte structural
            m.max_cell_size = branch_size;
            m.nutrient_priority = 1.0;
            m.split_mass = branch_mass;
            m.split_interval = branch_ivl;
            m.parent_make_adhesion = true;
            m.parent_split_direction = struct_b_split_dir; // ZERO = local forward (= spine direction)
            m.max_splits = branch_ext;
            m.enable_parent_angle_snapping = false;
            m.split_ratio = 0.5;
            m.membrane_stiffness = membrane * 0.85;
            m.max_adhesions = rng.i32_range(4, 10);
            m.division_signal_channel = ch_feeder;
            m.division_signal_threshold = 0.5;
            m.division_signal_invert = false;
            m.mode_switch_signal_channel = ch_maturity;
            m.mode_switch_signal_threshold = 1.0;
            m.mode_switch_target = r_specialist as i32;
            // Positional routing: head tips and tail tips get different specialists
            m.signal_child_a_channel = ch_anterior;
            m.signal_child_a_threshold = 15.0;
            m.signal_child_a_mode_above = r_specialist as i32;
            m.signal_child_a_mode_below = r_extra_spec.unwrap_or(r_specialist) as i32;
            apply_adhesion(m, adh_rest * 1.05, adh_lin * 0.75, adh_ang * 0.65, flex);
            // Both children continue extruding straight along the spine direction
            m.child_a.mode_number = r_struct_b as i32;
            m.child_a.keep_adhesion = true;
            m.child_a.orientation = struct_b_child_a_q; // IDENTITY
            m.child_b.mode_number = r_struct_b as i32;
            m.child_b.keep_adhesion = true;
            m.child_b.orientation = struct_b_child_b_q; // IDENTITY
                                                        // After branch_ext: route via signal_child_a above
            m.mode_a_after_splits = r_specialist as i32;
            m.mode_b_after_splits = r_specialist as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // -- FEEDER ------------------------------------------------------------
        // Nutrient producer. Emits ch_feeder so struct_a/struct_b know food is
        // available and can divide. Splits a small cluster along the spine axis,
        // then self-renews. The stem transitions into this role on ch_maturity
        // so the adult body has more feeders than the juvenile.
        // Devorocyte excluded - it can't bootstrap nutrients from nothing.
        {
            let m = &mut genome.modes[r_feeder];
            m.name = "Feeder".to_string();
            m.max_cell_size = branch_size;
            m.nutrient_priority = 1.8;
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = true;
            m.parent_split_direction = stem_split_dir;
            m.max_splits = rng.i32_range(1, 3);
            m.enable_parent_angle_snapping = false;
            m.split_ratio = 0.5;
            m.membrane_stiffness = membrane * 0.9;
            // Emit feeder abundance signal continuously
            m.regulation_emit_channel = ch_feeder;
            m.regulation_emit_value = 20.0;
            m.regulation_emit_hops = 15;
            apply_adhesion(m, adh_rest, adh_lin * 0.8, adh_ang * 0.7, flex);
            // Exclude devorocyte - it can't produce nutrients independently
            let safe_feed_type = if feed_cell_type == 11 {
                2
            } else {
                feed_cell_type
            };
            apply_type(m, safe_feed_type, ch_sense, &mut rng);
            m.child_a.mode_number = r_feeder as i32;
            m.child_a.keep_adhesion = true;
            m.child_a.orientation = Quat::IDENTITY;
            m.child_b.mode_number = r_feeder as i32;
            m.child_b.keep_adhesion = true;
            m.child_b.orientation = Quat::IDENTITY;
            m.mode_a_after_splits = r_feeder as i32;
            m.mode_b_after_splits = r_feeder as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // -- LOCOMOTION --------------------------------------------------------
        // Active movement cell. Reads ch_sense so it speeds up when the sensor
        // detects food or a target. Structural cells transition into this role
        // when ch_maturity floods the body - the whole body activates movement.
        // Terminal: does not divide. No parent_make_adhesion needed - this role
        // is reached via mode_switch, not division, so bonds already exist.
        // Myocyte phase assigned from ch_anterior: high signal = near head = phase A.
        {
            let m = &mut genome.modes[r_loco];
            m.name = "Loco".to_string();
            m.max_cell_size = spec_size * 1.2;
            m.nutrient_priority = rng.f32(1.0, 2.0);
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = false;
            m.parent_split_direction = Vec2::ZERO;
            m.max_splits = 0; // terminal
            m.enable_parent_angle_snapping = false;
            m.membrane_stiffness = membrane * rng.f32(0.8, 1.1);
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            let effective_loco = if loco_cell_type == 5 {
                5
            } else {
                loco_cell_type
            };
            apply_type(m, effective_loco, ch_sense, &mut rng);
            // Myocyte: assign phase A to head-end cells via ch_anterior signal routing.
            // Cells receiving high anterior signal (near head) use phase A;
            // cells with low signal (near tail) use phase B - creates peristaltic wave.
            if effective_loco == 9 {
                m.myocyte_use_signal = true;
                m.myocyte_signal_channel = ch_anterior;
                m.myocyte_contraction_above = rng.f32(0.4, 0.8); // phase A: contract
                m.myocyte_contraction_below = rng.f32(0.0, 0.2); // phase B: relax
                m.myocyte_threshold = 15.0; // above = near head
            }
            m.child_a.mode_number = r_loco as i32;
            m.child_b.mode_number = r_loco as i32;
            m.mode_a_after_splits = r_loco as i32;
            m.mode_b_after_splits = r_loco as i32;
        }

        // -- SPECIALIST --------------------------------------------------------
        // Terminal body function. Reached via mode_switch on ch_maturity or
        // via max_splits exhaustion on struct_b. No division needed.
        {
            let m = &mut genome.modes[r_specialist];
            m.name = "Specialist".to_string();
            m.max_cell_size = spec_size;
            m.nutrient_priority = rng.f32(0.8, 1.6);
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = false;
            m.parent_split_direction = Vec2::ZERO;
            m.max_splits = 0; // terminal
            m.enable_parent_angle_snapping = false;
            m.membrane_stiffness = membrane * rng.f32(0.7, 1.1);
            apply_adhesion(m, adh_rest, adh_lin * 0.9, adh_ang * 0.9, flex);
            apply_type(m, spec_cell_type, ch_sense, &mut rng);
            m.child_a.mode_number = r_specialist as i32;
            m.child_b.mode_number = r_specialist as i32;
            m.mode_a_after_splits = r_specialist as i32;
            m.mode_b_after_splits = r_specialist as i32;
        }

        // -- GONAD -------------------------------------------------------------
        // Reproductive organ. The stem transitions here after spine_segs splits.
        // Emits ch_maturity - floods the whole body, triggering all grow->adult
        // mode switches simultaneously. Sheds num_eggs free stems directly
        // (no zygote wrapper needed), then reverts to stem to rebuild.
        //
        // FIX 4 - Gonad gated on min_adhesions:
        // The gonad only starts dividing (shedding eggs) when it has enough
        // adhesion connections, meaning the body is structurally complete.
        // This prevents ch_maturity from firing mid-construction.
        {
            let m = &mut genome.modes[r_gonad];
            m.name = "Gonad".to_string();
            m.cell_type = 2; // Phagocyte - auto-gains mass
            m.max_cell_size = stem_size;
            m.nutrient_priority = 2.0;
            m.nutrient_gain_rate = 20.0;
            m.split_mass = stem_mass;
            m.split_interval = stem_ivl;
            m.parent_make_adhesion = false; // eggs detach freely
            m.parent_split_direction = stem_split_dir;
            m.max_splits = num_eggs;
            m.enable_parent_angle_snapping = false;
            m.split_ratio = 0.5;
            m.membrane_stiffness = membrane;
            // Emit maturity signal: floods body, triggers all grow->adult switches
            m.regulation_emit_channel = ch_maturity;
            m.regulation_emit_value = 50.0;
            m.regulation_emit_hops = 20;
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Child A = free stem (detaches, starts new organism directly)
            m.child_a.mode_number = r_stem as i32;
            m.child_a.keep_adhesion = false;
            m.child_a.orientation = Quat::IDENTITY;
            // Child B = gonad continues shedding
            m.child_b.mode_number = r_gonad as i32;
            m.child_b.keep_adhesion = true;
            m.child_b.orientation = Quat::IDENTITY;
            // After num_eggs: gonad reverts to stem -> organism rebuilds
            m.mode_a_after_splits = r_stem as i32;
            m.mode_b_after_splits = r_stem as i32;
            m.child_a_after_split_keep_adhesion = false;
            m.child_b_after_split_keep_adhesion = true;
        }

        // -- SENSOR (optional) -------------------------------------------------
        // Oculocyte. Senses food or cells along its forward axis and fires
        // ch_sense into the adhesion network. Locomotion cells read ch_sense
        // and speed up. Only present if the creature has locomotion.
        if let Some(idx) = r_sensor {
            let m = &mut genome.modes[idx];
            m.name = "Sensor".to_string();
            m.cell_type = 7; // Oculocyte
            m.max_cell_size = spec_size;
            m.nutrient_priority = 2.0;
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = false;
            m.max_splits = 0; // terminal
            m.enable_parent_angle_snapping = false;
            m.membrane_stiffness = membrane;
            // Sense food for phagocytes/photocytes, cells for devorocytes
            m.oculocyte_sense_type = if feed_cell_type == 11 { 1 << 1 } else { 1 << 0 }; // Food=bit1, Cell=bit0
            m.oculocyte_signal_channel = ch_sense;
            m.oculocyte_signal_value = rng.f32(5.0, 20.0);
            m.oculocyte_signal_hops = rng.i32_range(3, 12);
            m.oculocyte_ray_length = rng.f32(10.0, 40.0);
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            m.child_a.mode_number = idx as i32;
            m.child_b.mode_number = idx as i32;
            m.mode_a_after_splits = idx as i32;
            m.mode_b_after_splits = idx as i32;
        }

        // -- ADULT STRUCT (optional) -------------------------------------------
        // An alternative adult form for structural cells. When present, struct_a
        // switches to this instead of r_loco on ch_maturity - giving the creature
        // a distinct adult structural identity (e.g. vasculocyte transport network).
        if let Some(idx) = r_adult_struct {
            let m = &mut genome.modes[idx];
            m.name = "Adult-Struct".to_string();
            m.cell_type = 12; // Vasculocyte: nutrient transport
            m.max_cell_size = spine_size;
            m.nutrient_priority = 0.5;
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = false;
            m.max_splits = 0; // terminal
            m.enable_parent_angle_snapping = false;
            m.membrane_stiffness = membrane;
            m.vascular_nutrient_transport = true;
            m.vascular_outlet = rng.bool(0.4); // some are outlets
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            m.child_a.mode_number = idx as i32;
            m.child_b.mode_number = idx as i32;
            m.mode_a_after_splits = idx as i32;
            m.mode_b_after_splits = idx as i32;
            // Redirect struct_a's maturity switch to this adult form
            genome.modes[r_struct_a].mode_switch_target = idx as i32;
        }

        // -- EXTRA SPECIALIST (optional) ---------------------------------------
        // A second specialist type. Struct_b switches to this on ch_maturity
        // instead of r_specialist, giving branch tips a different function
        // from spine tips. Creates functional differentiation along the axis.
        if let Some(idx) = r_extra_spec {
            let extra_type: i32 = *rng.pick(&[1i32, 3, 5, 8, 9, 11]);
            let m = &mut genome.modes[idx];
            m.name = "Extra-Spec".to_string();
            m.max_cell_size = spec_size;
            m.nutrient_priority = rng.f32(0.6, 1.4);
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = false;
            m.max_splits = 0; // terminal
            m.enable_parent_angle_snapping = false;
            m.membrane_stiffness = membrane * 0.85;
            apply_adhesion(m, adh_rest, adh_lin * 0.9, adh_ang * 0.9, flex);
            apply_type(m, extra_type, ch_sense, &mut rng);
            m.child_a.mode_number = idx as i32;
            m.child_b.mode_number = idx as i32;
            m.mode_a_after_splits = idx as i32;
            m.mode_b_after_splits = idx as i32;
            // Redirect struct_b's maturity switch to this extra specialist
            genome.modes[r_struct_b].mode_switch_target = idx as i32;
        }

        // -- ANCHOR (optional) -------------------------------------------------
        // Glueocyte. Present in sessile creatures (no locomotion). Struct_b
        // switches to this on ch_maturity, anchoring branch tips to the
        // environment. Bonding is gated on ch_maturity so the creature finishes
        // growing before locking itself in place.
        if let Some(idx) = r_anchor {
            let m = &mut genome.modes[idx];
            m.name = "Anchor".to_string();
            m.cell_type = 6; // Glueocyte
            m.max_cell_size = spec_size;
            m.nutrient_priority = 1.5;
            m.split_mass = spec_mass;
            m.split_interval = spec_ivl;
            m.parent_make_adhesion = false;
            m.parent_split_direction = Vec2::ZERO;
            m.max_splits = 0; // terminal
            m.enable_parent_angle_snapping = false;
            m.membrane_stiffness = membrane * 1.2;
            m.glueocyte_env_adhesion = true;
            m.glueocyte_cell_adhesion = false;
            // Only bond after maturity signal arrives - prevents premature anchoring
            m.glueocyte_cell_adhesion_signal_channel = ch_maturity;
            m.glueocyte_cell_adhesion_signal_threshold = 1.0;
            apply_adhesion(m, adh_rest, adh_lin * 1.2, adh_ang * 1.2, false); // always rigid
            m.child_a.mode_number = idx as i32;
            m.child_b.mode_number = idx as i32;
            m.mode_a_after_splits = idx as i32;
            m.mode_b_after_splits = idx as i32;
            // Redirect struct_b's maturity switch to anchor
            genome.modes[r_struct_b].mode_switch_target = idx as i32;
        }

        genome
    }

    /// Compute a similarity score [0.0, 1.0] between two genomes based on their mode sequences.
    ///
    /// The score is the product of two factors:
    /// 1. **Mode-count alignment**: `min(len_a, len_b) / max(len_a, len_b)` - penalises very
    ///    different genome sizes (a genome with 4 modes vs one with 16 scores only 0.25 here).
    /// 2. **Cell-type match fraction**: for the overlapping prefix of modes (up to `min(len_a, len_b)`),
    ///    the proportion of positions where both genomes have the same cell_type.
    ///
    /// A genome compared with itself always returns 1.0.
    /// Returns 0.0 if either genome has no modes.
    pub fn similarity(a: &Genome, b: &Genome) -> f32 {
        let len_a = a.modes.len();
        let len_b = b.modes.len();
        if len_a == 0 || len_b == 0 {
            return 0.0;
        }

        let max_len = len_a.max(len_b) as f32;
        let min_len = len_a.min(len_b) as f32;
        let count_alignment = min_len / max_len; // 1.0 when equal, drops toward 0 as sizes diverge

        let compare_len = len_a.min(len_b);
        let matching = (0..compare_len)
            .filter(|&i| a.modes[i].cell_type == b.modes[i].cell_type)
            .count();
        let type_match = matching as f32 / compare_len as f32;

        count_alignment * type_match
    }

    /// Create a hybrid offspring genome by crossing over two parent genomes.
    ///
    /// Each mode in the offspring is independently drawn from either parent_a or parent_b
    /// with equal probability (per-mode uniform crossover). The offspring genome's mode
    /// count matches parent_a's. Child mode references are remapped to stay within the
    /// offspring's valid range. The initial_mode is taken from parent_a.
    ///
    /// This mirrors biological sexual reproduction: the offspring inherits a random mix
    /// of traits from both parents, enabling exploration of the combined trait space.
    ///
    /// # Arguments
    /// * `parent_a` - First parent genome (determines mode count and initial_mode)
    /// * `parent_b` - Second parent genome
    /// * `rng_seed` - Seed for crossover RNG (e.g. frame number XOR'd with cell IDs)
    pub fn crossover(parent_a: &Genome, parent_b: &Genome, rng_seed: u64) -> Genome {
        let a_count = parent_a.modes.len();
        let b_count = parent_b.modes.len();
        // Use the longer parent's length so dormant tail modes from either parent
        // are preserved in the offspring rather than silently dropped.
        let mode_count = a_count.max(b_count).max(1);

        // Simple LCG for per-mode coin flips
        let mut rng = rng_seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut next_bit = || -> bool {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 63) == 1
        };

        let mut modes = Vec::with_capacity(mode_count);
        for i in 0..mode_count {
            let have_a = i < a_count;
            let have_b = i < b_count;

            // Pick mode source based on what's available:
            // - Both present: coin flip
            // - Only one present: take it directly (preserves tail experiments)
            let pick_b = match (have_a, have_b) {
                (true, true) => next_bit(),
                (false, true) => true,
                (true, false) => false,
                (false, false) => unreachable!(),
            };

            let mut mode = if pick_b {
                parent_b.modes[i].clone()
            } else {
                parent_a.modes[i].clone()
            };

            // Remap child mode indices to stay within [0, mode_count)
            let clamp_mode = |idx: i32| -> i32 {
                if idx < 0 {
                    idx
                } else {
                    idx.min(mode_count as i32 - 1)
                }
            };
            mode.child_a.mode_number = clamp_mode(mode.child_a.mode_number);
            mode.child_b.mode_number = clamp_mode(mode.child_b.mode_number);
            mode.mode_a_after_splits = clamp_mode(mode.mode_a_after_splits);
            mode.mode_b_after_splits = clamp_mode(mode.mode_b_after_splits);
            mode.mode_switch_target = clamp_mode(mode.mode_switch_target);

            // Blend color from both parents at shared positions; tail modes keep their own color
            if have_a && have_b {
                mode.color = (parent_a.modes[i].color + parent_b.modes[i].color) * 0.5;
            }

            modes.push(mode);
        }

        let name = format!("{} × {}", parent_a.name, parent_b.name);
        let initial_mode = parent_a.initial_mode.min(mode_count as i32 - 1).max(0);

        Genome {
            name,
            initial_mode,
            initial_orientation: parent_a.initial_orientation,
            modes,
        }
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
