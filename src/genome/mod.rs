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
            break_force: 500.0,
            rest_length: 1.0,
            linear_spring_stiffness: 150.0,
            linear_spring_damping: 5.0,
            orientation_spring_stiffness: 50.0,
            orientation_spring_damping: 0.5,
            max_angular_deviation: 0.0,
            twist_constraint_stiffness: 2.0,
            twist_constraint_damping: 0.05,
            enable_twist_constraint: true,
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
    pub child_a_after_split_orientation: Quat, // Orientation for Child A when max_splits is reached
    pub child_b_after_split_orientation: Quat, // Orientation for Child B when max_splits is reached
    pub child_a_after_split_keep_adhesion: bool, // Whether Child A keeps adhesion when max_splits is reached
    pub child_b_after_split_keep_adhesion: bool, // Whether Child B keeps adhesion when max_splits is reached
    
    // Glueocyte settings
    pub glueocyte_cell_adhesion: bool, // Whether this Glueocyte bonds to other cells on contact
    pub glueocyte_env_adhesion: bool,  // Whether this Glueocyte bonds to the environment on contact
    /// Signal channel that controls cell-adhesion activation (-1 = always active, 0-7 = oculocyte channel).
    /// When set, the glueocyte only forms new bonds when the signal is above the threshold,
    /// and releases all glueocyte-created bonds when the signal drops below the threshold.
    pub glueocyte_cell_adhesion_signal_channel: i32,
    pub glueocyte_cell_adhesion_signal_threshold: f32,

    // Flagellocyte settings
    pub swim_force: f32, // Forward thrust force (0.0 to 1.0, for Flagellocyte cells)
    pub flagellocyte_use_signal: bool, // If true, use signal-based speed; if false, use fixed swim_force
    pub flagellocyte_signal_channel: i32, // Which signal channel to read (0-7, oculocyte channels only)
    pub flagellocyte_speed_a: f32, // Swim speed when signal < threshold_c
    pub flagellocyte_speed_b: f32, // Swim speed when signal >= threshold_c
    pub flagellocyte_threshold_c: f32, // Signal threshold for speed switching
    
    // Buoyocyte settings
    pub buoyancy_force: f32, // Upward buoyancy force (0.0 to 1.0, for Buoyocyte cells)
    
    // Oculocyte settings
    pub oculocyte_sense_type: i32, // 0=Cell, 1=Food, 2=Light, 3=Barrier, 4=Self
    pub oculocyte_signal_channel: i32, // Which channel to send on (0-7, oculocyte-only range)
    pub oculocyte_signal_value: f32, // Signal value to send when target detected (-50.0 to 50.0)
    pub oculocyte_signal_hops: i32, // How many adhesion hops the signal propagates (1-20)
    pub oculocyte_ray_length: f32, // How far ahead the oculocyte ray reaches (1.0 to 100.0)
    
    // Ciliocyte settings
    pub cilia_speed: f32, // Cilia force magnitude (-1.0 to +1.0, for Ciliocyte cells)
    pub cilia_push_bonded: bool, // Whether to push same-organism cells
    pub cilia_use_signal: bool, // If true, use signal-based speed; if false, use fixed cilia_speed
    pub cilia_signal_channel: i32, // Which signal channel to read (0-7, oculocyte channels only)
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
    pub myocyte_threshold: f32, // Signal threshold for contraction switching
    pub myocyte_pulse_rate: f32, // Pulse oscillation rate in cycles per second (0.1 to 10.0)
    pub myocyte_pulse_phase: i32, // Which pulse phase to contract on (0 = Pulse A, 1 = Pulse B)

    // Embryocyte settings
    // Release triggers (AND logic): all enabled triggers must be satisfied simultaneously
    pub embryocyte_use_timer: bool,       // Enable timer trigger
    pub embryocyte_release_timer: f32,    // Seconds since cell creation before timer is satisfied
    pub embryocyte_use_threshold: bool,   // Enable reserve threshold trigger
    pub embryocyte_threshold_value: u32,  // Reserve must be >= this value (0-65535)
    pub embryocyte_use_signal: bool,      // Enable signal trigger
    pub embryocyte_signal_channel: i32,   // Signal channel to monitor (0-15)
    pub embryocyte_signal_value: f32,     // Minimum signal value required for trigger

    // Regulation signal emission: any cell mode can emit a signal on channels 8-15
    pub regulation_emit_channel: i32, // Channel to emit on (-1 = disabled, 8-15 = regulation channel)
    pub regulation_emit_value: f32, // Signal value to emit (0.0 to 2047.0)
    pub regulation_emit_hops: i32, // How many adhesion hops the signal propagates (1-20)

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
    pub mode_switch_target: i32, // Target mode index (-1 = disabled)
    pub mode_switch_invert: bool, // If true, switch when signal BELOW threshold

    // Devorocyte settings
    pub devorocyte_consume_range: f32, // Extra contact range beyond cell radii (0.0 to 2.0)
    pub devorocyte_consume_rate: f32,  // Nutrients stolen per second from each victim (1.0 to 100.0)

    // Vasculocyte settings
    pub vascular_outlet: bool, // When true, releases nutrients to non-vascular neighbors; when false, sealed (pipe only)

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
            cell_type: 2,
            parent_make_adhesion: false,
            split_mass: 1.5,
            split_interval: 1.0,
            nutrient_gain_rate: 10.0,
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
            child_a_after_split_orientation: Quat::IDENTITY, // Default orientation for Child A after max splits
            child_b_after_split_orientation: Quat::IDENTITY, // Default orientation for Child B after max splits
            child_a_after_split_keep_adhesion: true, // Default: keep adhesion for Child A after max splits
            child_b_after_split_keep_adhesion: true, // Default: keep adhesion for Child B after max splits
            glueocyte_cell_adhesion: false,  // Default: cell adhesion disabled
            glueocyte_env_adhesion: true,      // Default: environment adhesion enabled
            glueocyte_cell_adhesion_signal_channel: -1, // Default: always active (no signal gate)
            glueocyte_cell_adhesion_signal_threshold: 1.0,
            swim_force: 0.5, // Default swim force for flagellocytes
            flagellocyte_use_signal: false, // Default: fixed speed mode
            flagellocyte_signal_channel: 0, // Default: channel 0
            flagellocyte_speed_a: 0.5, // Default: same as swim_force
            flagellocyte_speed_b: 0.0, // Default: stop when signal received
            flagellocyte_threshold_c: 1.0, // Default: threshold of 1.0
            cilia_speed: 0.5, // Default cilia speed for ciliocytes
            cilia_push_bonded: false, // Default: don't push same-organism cells
            cilia_use_signal: false, // Default: fixed speed mode
            cilia_signal_channel: 0, // Default: channel 0
            cilia_speed_below: 0.5, // Default: same as cilia_speed
            cilia_speed_above: 0.0, // Default: stop when signal received
            cilia_threshold: 1.0, // Default: threshold of 1.0
            cilia_attract_force: 0.0, // Default: no attraction (off)
            buoyancy_force: 0.5, // Default buoyancy force for buoyocytes
            oculocyte_sense_type: 0, // Default: sense cells
            oculocyte_signal_channel: 0, // Default: channel 0
            oculocyte_signal_value: 10.0, // Default: +10 signal
            oculocyte_signal_hops: 3, // Default: 3 hops
            oculocyte_ray_length: 20.0, // Default: 20 units ray length
            membrane_stiffness: 250.0, // Default: moderate membrane stiffness
            myocyte_contraction: 0.5, // Default: 50% contraction
            myocyte_use_signal: false, // Default: pulse timer mode (works without signals)
            myocyte_signal_channel: 0, // Default: channel 0
            myocyte_contraction_above: 0.5, // Default: contract when signal received
            myocyte_contraction_below: 0.0, // Default: relaxed when no signal
            myocyte_threshold: 1.0, // Default: threshold of 1.0
            myocyte_pulse_rate: 1.0, // Default: 1 cycle per second
            myocyte_pulse_phase: 0, // Default: pulse A
            embryocyte_use_timer: false,       // Default: no timer trigger
            embryocyte_release_timer: 10.0,    // Default: 10 seconds
            embryocyte_use_threshold: false,   // Default: no threshold trigger
            embryocyte_threshold_value: 32768, // Default: half full (32768)
            embryocyte_use_signal: false,      // Default: no signal trigger
            embryocyte_signal_channel: 0,      // Default: channel 0
            embryocyte_signal_value: 1.0,      // Default: threshold of 1.0
            regulation_emit_channel: -1, // Disabled by default
            regulation_emit_value: 10.0, // Default: 10 signal value
            regulation_emit_hops: 3, // Default: 3 hops
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
            vascular_outlet: false,
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
        
        // Create all 80 modes with evenly distributed hues (stable defaults for serialization)
        for i in 0..80 {
            let mode_name = format!("M {}", i + 1);  // Start mode numbering from 1
            let mut mode = ModeSettings {
                name: mode_name.clone(),
                default_name: mode_name,
                ..Default::default()
            };

            // Deterministic color spread — must stay stable so serialization diffs work correctly.
            // Random colors are assigned by the UI when the user creates a new genome.
            let hue = (i as f32 / 80.0) * 360.0;
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

impl Genome {
    /// Create a new genome with random colors for each mode.
    /// Use this for user-facing "new genome" creation.
    /// Do NOT use this as a serialization baseline — `Default::default()` must stay deterministic.
    pub fn new_with_random_colors() -> Self {
        let mut genome = Self::default();

        // Randomize colors using LCG seeded from system time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(12345);
        let mut rng = seed as u64;

        for mode in &mut genome.modes {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((rng >> 33) & 0xFF) as f32 / 255.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let g = ((rng >> 33) & 0xFF) as f32 / 255.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = ((rng >> 33) & 0xFF) as f32 / 255.0;
            mode.color = Vec3::new(r, g, b);
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
