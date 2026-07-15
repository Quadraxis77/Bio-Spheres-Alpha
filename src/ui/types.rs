//! UI type definitions for the Bio-Spheres application.
//!
//! This module contains core types used throughout the UI system,
//! including simulation modes and per-scene configuration.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::ui::tutorial::TutorialState;

/// Available UI themes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum UiTheme {
    /// Dark navy + teal accents - the original Bio-Spheres look.
    #[default]
    BiotechDark,
    /// Clean white panels with deep blue accents - clinical / lab notebook.
    Arctic,
    /// Warm cream paper with ink-brown text and rust accents - parchment.
    Parchment,
    /// Soft lavender panels with rose-gold accents - pastel studio.
    Blossom,
    /// Rich dark burgundy panels with gold accents - deep wine.
    Crimson,
    /// Hot pink + electric cyan on deep purple-black - synthwave.
    NeonSynthwave,
    /// Acid green + electric yellow on pure black - toxic matrix.
    NeonToxic,
    /// Electric purple + hot magenta on near-black - ultraviolet.
    NeonUltraviolet,
    /// Near-black with bright white text - maximum contrast.
    HighContrast,
    /// Fully user-defined colors, edited in the Theme Editor panel.
    Custom,
}

impl UiTheme {
    pub fn all() -> &'static [UiTheme] {
        &[
            UiTheme::BiotechDark,
            UiTheme::Arctic,
            UiTheme::Parchment,
            UiTheme::Blossom,
            UiTheme::Crimson,
            UiTheme::NeonSynthwave,
            UiTheme::NeonToxic,
            UiTheme::NeonUltraviolet,
            UiTheme::HighContrast,
            UiTheme::Custom,
        ]
    }

    pub fn display_name(self) -> &'static str {
        match self {
            UiTheme::BiotechDark => "Biotech Dark",
            UiTheme::Arctic => "Arctic",
            UiTheme::Parchment => "Parchment",
            UiTheme::Blossom => "Blossom",
            UiTheme::Crimson => "Crimson",
            UiTheme::NeonSynthwave => "Neon Synthwave",
            UiTheme::NeonToxic => "Neon Toxic",
            UiTheme::NeonUltraviolet => "Neon Ultraviolet",
            UiTheme::HighContrast => "High Contrast",
            UiTheme::Custom => "Custom",
        }
    }

    pub fn accent_color(self) -> egui::Color32 {
        match self {
            UiTheme::BiotechDark => egui::Color32::from_rgb(0, 200, 160),
            UiTheme::Arctic => egui::Color32::from_rgb(30, 120, 220),
            UiTheme::Parchment => egui::Color32::from_rgb(180, 80, 30),
            UiTheme::Blossom => egui::Color32::from_rgb(210, 90, 150),
            UiTheme::Crimson => egui::Color32::from_rgb(210, 165, 40),
            UiTheme::NeonSynthwave => egui::Color32::from_rgb(255, 30, 180),
            UiTheme::NeonToxic => egui::Color32::from_rgb(50, 255, 50),
            UiTheme::NeonUltraviolet => egui::Color32::from_rgb(180, 30, 255),
            UiTheme::HighContrast => egui::Color32::from_rgb(255, 255, 255),
            UiTheme::Custom => egui::Color32::from_rgb(200, 200, 200),
        }
    }
}

/// Serializable color stored as [r, g, b] bytes for RON compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThemeColor(pub [u8; 3]);

impl ThemeColor {
    pub fn to_egui(self) -> egui::Color32 {
        egui::Color32::from_rgb(self.0[0], self.0[1], self.0[2])
    }
    pub fn from_egui(c: egui::Color32) -> Self {
        Self([c.r(), c.g(), c.b()])
    }
}

/// All 20 user-editable palette colors for the Custom theme.
/// Matches the 20 fields extracted from `apply_theme` (excludes derived rail_icon fields).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomThemePalette {
    pub bg_darkest: ThemeColor,
    pub bg_panel: ThemeColor,
    pub bg_widget: ThemeColor,
    pub bg_hover: ThemeColor,
    pub bg_active: ThemeColor,
    pub bg_selected: ThemeColor,
    pub accent_primary: ThemeColor,
    pub accent_secondary: ThemeColor,
    pub text_primary: ThemeColor,
    pub text_secondary: ThemeColor,
    pub text_dim: ThemeColor,
    pub border_subtle: ThemeColor,
    pub border_normal: ThemeColor,
    pub border_bright: ThemeColor,
    pub topbar_bg: ThemeColor,
    pub topbar_border: ThemeColor,
    pub status_ok: ThemeColor,
    pub status_warn: ThemeColor,
    pub status_err: ThemeColor,
    pub status_info: ThemeColor,
    /// Whether to use dark_mode (true) or light_mode (false) for egui internals.
    pub dark_mode: bool,
}

impl Default for CustomThemePalette {
    fn default() -> Self {
        // Starts as a copy of BiotechDark so the user has a sensible starting point.
        Self {
            bg_darkest: ThemeColor([6, 9, 18]),
            bg_panel: ThemeColor([12, 17, 32]),
            bg_widget: ThemeColor([22, 30, 52]),
            bg_hover: ThemeColor([32, 44, 72]),
            bg_active: ThemeColor([42, 58, 95]),
            bg_selected: ThemeColor([18, 55, 85]),
            accent_primary: ThemeColor([0, 220, 175]),
            accent_secondary: ThemeColor([60, 195, 240]),
            text_primary: ThemeColor([225, 235, 255]),
            text_secondary: ThemeColor([155, 175, 210]),
            text_dim: ThemeColor([80, 100, 145]),
            border_subtle: ThemeColor([28, 42, 72]),
            border_normal: ThemeColor([50, 72, 115]),
            border_bright: ThemeColor([0, 180, 140]),
            topbar_bg: ThemeColor([4, 6, 14]),
            topbar_border: ThemeColor([0, 160, 125]),
            status_ok: ThemeColor([60, 210, 100]),
            status_warn: ThemeColor([220, 185, 50]),
            status_err: ThemeColor([225, 70, 70]),
            status_info: ThemeColor([60, 150, 230]),
            dark_mode: true,
        }
    }
}

/// Available simulation modes/scenes.
///
/// - Preview: Single-threaded CPU simulation + GPU rendering (genome editor)
/// - Gpu: Fully GPU-based simulation without CPU readbacks (high performance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SimulationMode {
    /// Genome editor preview - single-threaded CPU simulation with GPU rendering.
    /// Optimized for editing and debugging genomes with a single cell or small colony.
    #[default]
    Preview,
    /// Full GPU simulation - entirely on GPU without CPU readbacks.
    /// Optimized for large-scale simulations with thousands of cells.
    Gpu,
}

impl SimulationMode {
    /// Get the filename suffix for this mode's dock state file.
    pub fn dock_file_suffix(&self) -> &'static str {
        match self {
            SimulationMode::Preview => "preview",
            SimulationMode::Gpu => "gpu",
        }
    }

    /// Get display name for UI.
    pub fn display_name(&self) -> &'static str {
        match self {
            SimulationMode::Preview => "Genome Editor",
            SimulationMode::Gpu => "GPU Simulation",
        }
    }

    /// Get all available simulation modes.
    pub fn all() -> &'static [SimulationMode] {
        &[SimulationMode::Preview, SimulationMode::Gpu]
    }
}

impl std::fmt::Display for SimulationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Per-scene UI configuration.
///
/// Each simulation mode can have its own independent window configuration,
/// visibility settings, and locked windows.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SceneUiConfig {
    /// Which panels are visible in this scene.
    pub visibility: WindowVisibilitySettings,
    /// Which panels are locked (can't be moved/closed).
    pub locked_windows: HashSet<String>,
    /// Lock settings for this scene.
    pub lock_settings: LockSettings,
}

impl Default for SceneUiConfig {
    fn default() -> Self {
        Self {
            visibility: WindowVisibilitySettings::default(),
            locked_windows: HashSet::new(),
            lock_settings: LockSettings::default(),
        }
    }
}

impl SceneUiConfig {
    /// Create default config for a specific simulation mode.
    pub fn default_for_mode(mode: SimulationMode) -> Self {
        match mode {
            SimulationMode::Preview => Self {
                visibility: WindowVisibilitySettings {
                    show_cell_inspector: false,
                    show_genome_editor: true,
                    show_scene_manager: true,
                    show_performance_monitor: false,
                    show_rendering_controls: false,
                    show_time_scrubber: true,
                    show_theme_editor: false,
                    show_camera_settings: false,
                    show_lighting_settings: false,
                },
                locked_windows: HashSet::new(),
                lock_settings: LockSettings::default(),
            },
            SimulationMode::Gpu => Self {
                visibility: WindowVisibilitySettings {
                    show_cell_inspector: true,
                    show_genome_editor: false,
                    show_scene_manager: true,
                    show_performance_monitor: true,
                    show_rendering_controls: true,
                    show_time_scrubber: false,
                    show_theme_editor: false,
                    show_camera_settings: true,
                    show_lighting_settings: false,
                },
                locked_windows: HashSet::new(),
                lock_settings: LockSettings::default(),
            },
        }
    }
}

/// Window visibility settings for each panel type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WindowVisibilitySettings {
    pub show_cell_inspector: bool,
    pub show_genome_editor: bool,
    pub show_scene_manager: bool,
    pub show_performance_monitor: bool,
    pub show_rendering_controls: bool,
    pub show_time_scrubber: bool,
    pub show_theme_editor: bool,
    pub show_camera_settings: bool,
    pub show_lighting_settings: bool,
}

impl Default for WindowVisibilitySettings {
    fn default() -> Self {
        Self {
            show_cell_inspector: true,
            show_genome_editor: true,
            show_scene_manager: true,
            show_performance_monitor: true,
            show_rendering_controls: true,
            show_time_scrubber: true,
            show_theme_editor: false,
            show_camera_settings: false,
            show_lighting_settings: false,
        }
    }
}

/// Lock settings for UI elements.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LockSettings {
    /// Whether the tab bar is locked (can't be dragged).
    pub lock_tab_bar: bool,
    /// Whether tabs are locked (can't be reordered).
    pub lock_tabs: bool,
    /// Whether close buttons are locked (can't close panels).
    pub lock_close_buttons: bool,
}

impl Default for LockSettings {
    fn default() -> Self {
        Self {
            lock_tab_bar: false,
            lock_tabs: false,
            lock_close_buttons: false,
        }
    }
}

/// World-specific simulation and physics settings.
///
/// These settings control the simulation world parameters including
/// physics and world configuration options.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WorldSettings {
    /// Target cell capacity (applied on scene reset)
    #[serde(default = "default_cell_capacity")]
    pub cell_capacity: u32,

    /// Gravity strength (negative = downward)
    #[serde(default)]
    pub gravity: f32,

    /// Gravity mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial (toward origin)
    #[serde(default = "default_gravity_mode")]
    pub gravity_mode: u32,

    /// Number of additional adhesion constraint solver iterations (0 = single-pass, higher = stiffer)
    #[serde(default = "default_constraint_iterations")]
    pub constraint_iterations: u32,

    /// Global velocity damping factor (0.0-1.0, higher = less damping, lower = more drag)
    #[serde(default = "default_acceleration_damping")]
    pub acceleration_damping: f32,

    /// Water viscosity: drag applied to cells moving through water (0.0 = off, 1.0 = heavy drag)
    #[serde(default = "default_water_drag_strength", alias = "water_drag_strength")]
    pub water_viscosity: f32,

    /// Global radiation level controlling mutation probability per division (0.0 = off, 1.0 = every division mutates)
    #[serde(default)]
    pub radiation_level: f32,

    /// When true, mutations make small color perturbations instead of full re-rolls
    #[serde(default)]
    pub subtle_mutations: bool,

    /// Bitmask of biological cell types available to GPU cell-type specialization.
    /// Bit N corresponds to CellType index N. Test (0) is intentionally excluded.
    #[serde(default = "default_mutation_gene_pool_mask")]
    pub mutation_gene_pool_mask: u32,

    /// World sphere radius in simulation units (applied on scene reset).
    /// Affects cave scale, fog bounds, fluid grid cell size, and physics boundary.
    #[serde(default = "default_world_radius")]
    pub world_radius: f32,

    /// When enabled, cells with zero adhesion connections burn nutrients faster.
    /// The multiplier scales metabolism for solo cells (e.g. 3.0 = 3x drain).
    /// Cells with 1-2 connections get partial penalty; 3+ connections = no penalty.
    #[serde(default)]
    pub solo_metabolism_enabled: bool,

    /// Metabolism multiplier for cells with zero adhesion connections.
    /// Only active when solo_metabolism_enabled is true.
    /// Range: 1.0 (no penalty) to 10.0 (extreme penalty).
    #[serde(default = "default_solo_metabolism_multiplier")]
    pub solo_metabolism_multiplier: f32,

    /// How often the light field is recomputed, in render frames.
    /// 1 = every frame (best quality), 2 = every other frame, 4 = every 4th frame.
    /// Higher values reduce GPU cost but cause light/shadow to lag cell movement.
    #[serde(default = "default_light_field_update_interval")]
    pub light_field_update_interval: u32,

    /// Physics simulation rate in Hz (32, 48, or 64).
    /// Lower Hz = cheaper per real-time second but coarser timestep; changes simulation timing.
    #[serde(default = "default_physics_hz")]
    pub physics_hz: u32,

    /// Maximum physics steps allowed per render frame at 1x speed.
    /// Caps how much catch-up work can happen in a single frame; auto = 4.
    #[serde(default = "default_max_physics_steps")]
    pub max_physics_steps_per_frame: u32,
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self {
            cell_capacity: 20_000,
            gravity: 0.0,
            gravity_mode: 1, // default Y axis
            constraint_iterations: 4,
            acceleration_damping: 0.98,
            water_viscosity: 0.0,
            radiation_level: 0.0,
            subtle_mutations: false,
            mutation_gene_pool_mask: default_mutation_gene_pool_mask(),
            world_radius: 200.0,
            solo_metabolism_enabled: false,
            solo_metabolism_multiplier: 3.0,
            light_field_update_interval: 1,
            physics_hz: 64,
            max_physics_steps_per_frame: 4,
        }
    }
}

/// Fluid simulation and rendering settings.
/// These settings control fluid-specific parameters that should persist
/// independently of world physics settings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FluidSettings {
    /// Surface pressure: tangential smoothing strength for radial fluid mode (0.0-1.0)
    #[serde(default = "default_surface_pressure")]
    pub surface_pressure: f32,
    /// Organism skin rendering settings
    #[serde(default)]
    pub organism_skin: OrganismSkinSettings,
    /// Climate/weather tunables (humidity, freeze/melt, snow)
    #[serde(default)]
    pub climate: ClimateSettings,
    /// Ice mesh appearance (facets, glints, colors, opacity)
    #[serde(default)]
    pub ice: IceAppearanceSettings,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            surface_pressure: 0.5,
            organism_skin: OrganismSkinSettings::default(),
            climate: ClimateSettings::default(),
            ice: IceAppearanceSettings::default(),
        }
    }
}

/// Ice mesh appearance tunables - mirrors `IceRenderParams` in the renderer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IceAppearanceSettings {
    /// Facets per world unit (lower = larger crystal faces)
    #[serde(default = "default_ice_facet_scale")]
    pub facet_scale: f32,
    /// How far each crystal facet plane tilts off the smooth surface, and
    /// how far vertices are displaced onto it (0 = no displacement, smooth)
    #[serde(default = "default_ice_displacement_strength")]
    pub displacement_strength: f32,
    /// Fraction of flat facet normal in diffuse shading (high = patchwork)
    #[serde(default = "default_ice_facet_diffuse")]
    pub facet_diffuse: f32,
    /// Blinn-Phong exponent for per-face glints
    #[serde(default = "default_ice_glint_shininess")]
    pub glint_shininess: f32,
    /// Glint intensity multiplier
    #[serde(default = "default_ice_glint_strength")]
    pub glint_strength: f32,
    /// Base opacity (fresnel adds on top)
    #[serde(default = "default_ice_alpha")]
    pub alpha: f32,
    /// Environment reflection brightness
    #[serde(default = "default_ice_reflection_brightness")]
    pub reflection_brightness: f32,
    /// Fresnel-weighted reflection mix at grazing angles
    #[serde(default = "default_ice_fresnel_reflection")]
    pub fresnel_reflection: f32,
    /// Pale color of faces toward the viewer
    #[serde(default = "default_ice_surface_color")]
    pub surface_color: [f32; 3],
    /// Deep interior color at glancing angles
    #[serde(default = "default_ice_deep_color")]
    pub deep_color: [f32; 3],
}

impl Default for IceAppearanceSettings {
    fn default() -> Self {
        Self {
            facet_scale: 0.05263158,
            displacement_strength: 0.15,
            facet_diffuse: 0.4,
            glint_shininess: 64.0,
            glint_strength: 3.0,
            alpha: 0.9,
            reflection_brightness: 0.8,
            fresnel_reflection: 0.38,
            surface_color: [0.6862745, 0.6862745, 0.6862745],
            deep_color: [0.4235294, 0.9254902, 1.0],
        }
    }
}

fn default_ice_facet_scale() -> f32 {
    0.05263158
}
fn default_ice_displacement_strength() -> f32 {
    0.15
}
fn default_ice_facet_diffuse() -> f32 {
    0.4
}
fn default_ice_glint_shininess() -> f32 {
    64.0
}
fn default_ice_glint_strength() -> f32 {
    3.0
}
fn default_ice_alpha() -> f32 {
    0.9
}
fn default_ice_reflection_brightness() -> f32 {
    0.8
}
fn default_ice_fresnel_reflection() -> f32 {
    0.38
}
fn default_ice_surface_color() -> [f32; 3] {
    [0.6862745, 0.6862745, 0.6862745]
}
fn default_ice_deep_color() -> [f32; 3] {
    [0.4235294, 0.9254902, 1.0]
}

/// Climate/weather simulation tunables - humidity diffusion and
/// debt-based freeze/melt/snow-compaction rates.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClimateSettings {
    /// Atmospheric humidity diffusion rate per tick (0.0-1.0)
    #[serde(default = "default_humidity_diffusion_rate")]
    pub humidity_diffusion_rate: f32,

    /// Global thermal inertia (0-5): heat flow speed plus water/ice phase resistance.
    /// 0 = arcade (fast), 4 = very stable (recommended), 5 = planetary (slow)
    #[serde(default = "default_thermal_inertia")]
    pub thermal_inertia: f32,

    /// Freeze debt accumulation rate for water -> ice
    #[serde(default = "default_freeze_rate")]
    pub freeze_rate: f32,

    /// Melt debt accumulation rate for ice -> water
    #[serde(default = "default_melt_rate")]
    pub melt_rate: f32,

    /// Melt debt accumulation rate for snow -> water
    #[serde(default = "default_snow_melt_rate")]
    pub snow_melt_rate: f32,

    /// Compaction debt accumulation rate for snow -> ice (sustained cold packs snow into ice)
    #[serde(default = "default_snow_compact_rate")]
    pub snow_compact_rate: f32,

    /// Water freezing threshold (internal 0-255 scale). Default: 65 (0°C/32°F)
    #[serde(default = "default_freeze_threshold")]
    pub freeze_threshold: u32,

    /// Ice melting threshold (internal 0-255 scale). Default: 75 (5°C/41°F)
    #[serde(default = "default_melt_threshold")]
    pub melt_threshold: u32,

    /// Snow formation threshold (internal 0-255 scale). Default: 60 (-2°C/28°F)
    #[serde(default = "default_snow_threshold")]
    pub snow_threshold: u32,

    /// Evaporation begins threshold (internal 0-255 scale). Default: 120 (28°C/82°F)
    #[serde(default = "default_evaporation_threshold")]
    pub evaporation_threshold: u32,

    /// Optimal cell temperature (internal 0-255 scale). Default: 105 (20°C/68°F)
    #[serde(default = "default_optimal_cell_temp")]
    pub optimal_cell_temp: u32,
}

impl Default for ClimateSettings {
    fn default() -> Self {
        Self {
            humidity_diffusion_rate: default_humidity_diffusion_rate(),
            thermal_inertia: default_thermal_inertia(),
            freeze_rate: default_freeze_rate(),
            melt_rate: default_melt_rate(),
            snow_melt_rate: default_snow_melt_rate(),
            snow_compact_rate: default_snow_compact_rate(),
            freeze_threshold: default_freeze_threshold(),
            melt_threshold: default_melt_threshold(),
            snow_threshold: default_snow_threshold(),
            evaporation_threshold: default_evaporation_threshold(),
            optimal_cell_temp: default_optimal_cell_temp(),
        }
    }
}

fn default_humidity_diffusion_rate() -> f32 {
    0.15
}

fn default_thermal_inertia() -> f32 {
    4.0
}

fn default_freeze_rate() -> f32 {
    1.0
}

fn default_melt_rate() -> f32 {
    1.5
}

fn default_snow_melt_rate() -> f32 {
    3.0
}

fn default_snow_compact_rate() -> f32 {
    1.0
}

fn default_freeze_threshold() -> u32 {
    65 // 0°C / 32°F
}

fn default_melt_threshold() -> u32 {
    75 // 5°C / 41°F
}

fn default_snow_threshold() -> u32 {
    60 // -2°C / 28°F
}

fn default_evaporation_threshold() -> u32 {
    120 // 28°C / 82°F
}

fn default_optimal_cell_temp() -> u32 {
    105 // 20°C / 68°F
}

/// Organism skin rendering settings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrganismSkinSettings {
    /// Whether organism skin rendering is enabled
    #[serde(default = "default_organism_skin_enabled")]
    pub enabled: bool,

    /// Grid resolution (kept for serialization compat, unused by shrink-wrap)
    #[serde(default = "default_organism_grid_resolution")]
    pub grid_resolution: u32,

    /// Skin offset - gap between the mesh surface and cell surfaces (world units)
    #[serde(default = "default_skin_radius_scale")]
    pub radius_scale: f32,

    /// Iso level (kept for serialization compat, unused by shrink-wrap)
    #[serde(default = "default_iso_level")]
    pub iso_level: f32,

    /// Shrink speed - fraction of remaining gap closed per iteration (0.05-0.5)
    #[serde(default = "default_shrink_speed")]
    pub shrink_speed: f32,

    /// Smooth factor - Laplacian blend weight per smooth iteration (0.0-0.6)
    #[serde(default = "default_smooth_factor")]
    pub smooth_factor: f32,

    /// Number of shrink iterations per frame (1-8)
    #[serde(default = "default_shrink_iters")]
    pub shrink_iters: u32,

    /// Number of smooth iterations per frame (0-4)
    #[serde(default = "default_smooth_iters")]
    pub smooth_iters: u32,

    /// Minimum cells for an organism to get a skin (1-20)
    #[serde(default = "default_min_cells_for_skin")]
    pub min_cells: u32,

    /// Base color RGB values
    #[serde(default = "default_skin_base_color")]
    pub base_color: [f32; 3],

    /// Material properties
    #[serde(default = "default_skin_ambient")]
    pub ambient: f32,

    #[serde(default = "default_skin_diffuse")]
    pub diffuse: f32,

    #[serde(default = "default_skin_specular")]
    pub specular: f32,

    #[serde(default = "default_skin_shininess")]
    pub shininess: f32,

    #[serde(default = "default_skin_alpha")]
    pub alpha: f32,

    #[serde(default = "default_skin_sss_strength")]
    pub sss_strength: f32,

    #[serde(default = "default_skin_rim_strength")]
    pub rim_strength: f32,
}

impl Default for OrganismSkinSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_resolution: 128,
            radius_scale: 1.2,
            iso_level: 0.5,
            shrink_speed: 0.25,
            smooth_factor: 0.3,
            shrink_iters: 8,
            smooth_iters: 3,
            min_cells: 4,
            base_color: [0.85, 0.55, 0.35],
            ambient: 0.12,
            diffuse: 0.6,
            specular: 0.5,
            shininess: 48.0,
            alpha: 0.55,
            sss_strength: 0.5,
            rim_strength: 0.35,
        }
    }
}

fn default_mip_override() -> i32 {
    -1
}

fn default_occlusion_bias() -> f32 {
    0.005 // Small positive bias to prevent self-occlusion from temporal jitter
}

fn default_constraint_iterations() -> u32 {
    4
}

fn default_cell_capacity() -> u32 {
    20_000
}

pub fn default_mutation_gene_pool_mask() -> u32 {
    crate::cell::types::CellType::all()
        .iter()
        .copied()
        .filter(|cell_type| *cell_type != crate::cell::types::CellType::Test)
        .fold(0u32, |mask, cell_type| {
            mask | (1u32 << cell_type.to_index())
        })
}

fn default_world_diameter() -> f32 {
    395.0
}

fn default_world_radius() -> f32 {
    200.0
}

fn default_lod_scale_factor() -> f32 {
    500.0 // Default scale factor for screen radius calculation
}

fn default_lod_threshold_low() -> f32 {
    10.0 // Low to Medium transition
}

fn default_lod_threshold_medium() -> f32 {
    25.0 // Medium to High transition
}

fn default_lod_threshold_high() -> f32 {
    50.0 // High to Ultra transition
}

fn default_horizontal_fov_degrees() -> f32 {
    crate::ui::camera::DEFAULT_HORIZONTAL_FOV_DEGREES
}

fn default_camera_sprint_multiplier() -> f32 {
    6.0
}

fn default_camera_scroll_sensitivity() -> f32 {
    0.2
}

fn default_field_report_interval_seconds() -> f32 {
    crate::scene::lineage::LINEAGE_CAPTURE_INTERVAL_SECONDS
}

fn default_organism_skin_enabled() -> bool {
    false
}

fn default_true() -> bool {
    true
}

fn default_headless_target_fps() -> f32 {
    30.0
}

fn default_headless_min_speed() -> f32 {
    0.1
}

pub const GPU_HEADLESS_MAX_SIM_SPEED: f32 = 10.0;

fn default_headless_max_speed() -> f32 {
    GPU_HEADLESS_MAX_SIM_SPEED
}

/// Status shown by the GPU headless auto-speed controller.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum HeadlessAutoStatus {
    #[default]
    Off,
    Increasing,
    Holding,
    Reducing,
    AtMinimum,
    AtMaximum,
}

impl HeadlessAutoStatus {
    pub fn display_name(&self) -> &'static str {
        match self {
            HeadlessAutoStatus::Off => "Manual",
            HeadlessAutoStatus::Increasing => "Increasing speed",
            HeadlessAutoStatus::Holding => "Holding steady",
            HeadlessAutoStatus::Reducing => "Reducing speed",
            HeadlessAutoStatus::AtMinimum => "At minimum speed",
            HeadlessAutoStatus::AtMaximum => "At maximum speed",
        }
    }
}

fn default_gravity_mode() -> u32 {
    1 // Y axis
}

fn default_surface_pressure() -> f32 {
    0.7
}

fn default_water_drag_strength() -> f32 {
    0.5
}

fn default_acceleration_damping() -> f32 {
    0.98
}

fn default_solo_metabolism_multiplier() -> f32 {
    3.0
}

fn default_light_field_update_interval() -> u32 {
    1
}

fn default_physics_hz() -> u32 {
    64
}

fn default_max_physics_steps() -> u32 {
    4
}

fn default_organism_grid_resolution() -> u32 {
    128
}

fn default_skin_radius_scale() -> f32 {
    1.2
}

fn default_shrink_speed() -> f32 {
    0.25
}
fn default_smooth_factor() -> f32 {
    0.3
}
fn default_shrink_iters() -> u32 {
    8
}
fn default_smooth_iters() -> u32 {
    3
}
fn default_min_cells_for_skin() -> u32 {
    4
}

fn default_skin_sss_strength() -> f32 {
    0.5
}

fn default_skin_rim_strength() -> f32 {
    0.35
}

fn default_iso_level() -> f32 {
    0.5
}

fn default_skin_base_color() -> [f32; 3] {
    [0.75, 0.45, 0.25] // Warm pinkish-amber
}

fn default_skin_ambient() -> f32 {
    0.15
}

fn default_skin_diffuse() -> f32 {
    0.7
}

fn default_skin_specular() -> f32 {
    0.6
}

fn default_skin_shininess() -> f32 {
    50.0
}

fn default_skin_alpha() -> f32 {
    0.9
}

/// Global UI state shared across all UI components.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GlobalUiState {
    /// Current simulation mode/scene.
    pub current_mode: SimulationMode,

    /// Global UI scale factor.
    pub ui_scale: f32,

    /// Whether windows are globally locked.
    pub windows_locked: bool,

    /// Whether the tab bar is locked.
    pub lock_tab_bar: bool,

    /// Whether tabs are locked.
    pub lock_tabs: bool,

    /// Whether close buttons are locked.
    pub lock_close_buttons: bool,

    /// Per-scene configurations.
    pub scene_configs: HashMap<SimulationMode, SceneUiConfig>,

    /// Occlusion culling bias (negative = more aggressive, positive = more conservative)
    #[serde(default = "default_occlusion_bias")]
    pub occlusion_bias: f32,

    /// Occlusion mip level override (-1 = auto, 0+ = force specific mip)
    #[serde(default = "default_mip_override")]
    pub occlusion_mip_override: i32,

    /// Minimum screen-space size (pixels) for occlusion culling
    #[serde(default)]
    pub occlusion_min_screen_size: f32,

    /// Minimum distance for occlusion culling
    #[serde(default)]
    pub occlusion_min_distance: f32,

    /// Whether occlusion culling is enabled
    #[serde(default = "default_true")]
    pub occlusion_enabled: bool,

    /// Whether frustum culling is enabled
    #[serde(default = "default_true")]
    pub frustum_enabled: bool,

    /// Whether GPU readbacks are enabled (cell count, culling stats)
    /// Disabling this can improve performance by avoiding CPU-GPU sync
    #[serde(default = "default_true")]
    pub gpu_readbacks_enabled: bool,

    /// Whether GPU frame timing timestamp queries are enabled.
    /// Disabling this removes per-frame timestamp writes and timing readbacks.
    #[serde(default = "default_true")]
    pub gpu_timing_enabled: bool,

    /// Whether scheduled ecosystem field reports are generated.
    #[serde(default = "default_true")]
    pub field_reports_enabled: bool,

    /// Horizontal camera field of view shared by preview and GPU scenes.
    #[serde(default = "default_horizontal_fov_degrees")]
    pub horizontal_fov_degrees: f32,

    /// FreeFly run speed multiplier shared by preview and GPU scenes.
    #[serde(default = "default_camera_sprint_multiplier")]
    pub camera_sprint_multiplier: f32,

    /// Mouse-wheel scroll sensitivity shared by preview and GPU scenes.
    #[serde(default = "default_camera_scroll_sensitivity")]
    pub camera_scroll_sensitivity: f32,

    /// Interval between scheduled ecosystem field-report scans.
    #[serde(default = "default_field_report_interval_seconds")]
    pub field_report_interval_seconds: f32,

    /// GPU mode no-render performance cockpit. Simulation continues, visual render passes stop.
    #[serde(skip)]
    pub gpu_headless_mode: bool,

    /// Automatically adjust GPU simulation speed to stay near the target FPS.
    #[serde(default)]
    pub gpu_headless_auto_speed: bool,

    /// Target FPS for the GPU headless auto-speed controller.
    #[serde(default = "default_headless_target_fps")]
    pub gpu_headless_target_fps: f32,

    /// Lower clamp for automatic GPU simulation speed.
    #[serde(default = "default_headless_min_speed")]
    pub gpu_headless_min_speed: f32,

    /// Upper clamp for automatic GPU simulation speed.
    #[serde(default = "default_headless_max_speed")]
    pub gpu_headless_max_speed: f32,

    /// Current auto-speed status. Session-only; derived from recent performance.
    #[serde(skip)]
    pub gpu_headless_auto_status: HeadlessAutoStatus,

    /// Whether to show adhesion lines between cells
    #[serde(default = "default_true")]
    pub show_adhesion_lines: bool,

    /// World diameter in simulation units (applied on scene reset)
    #[serde(default = "default_world_diameter")]
    pub world_diameter: f32,

    /// LOD scale factor for distance calculations (higher = full detail farther away)
    #[serde(default = "default_lod_scale_factor")]
    pub lod_scale_factor: f32,

    /// Screen-size threshold from LOD 0 basic sphere to LOD 1 full detail
    #[serde(default = "default_lod_threshold_low")]
    pub lod_threshold_low: f32,

    /// Legacy serialized setting retained for compatibility
    #[serde(default = "default_lod_threshold_medium")]
    pub lod_threshold_medium: f32,

    /// Legacy serialized setting retained for compatibility
    #[serde(default = "default_lod_threshold_high")]
    pub lod_threshold_high: f32,

    /// Whether to show debug colors for LOD levels
    #[serde(default)]
    pub lod_debug_colors: bool,

    /// World-specific simulation and physics settings
    #[serde(default)]
    pub world_settings: WorldSettings,

    /// Fluid simulation and rendering settings
    #[serde(default)]
    pub fluid_settings: FluidSettings,

    /// Active UI theme.
    #[serde(default)]
    pub selected_theme: UiTheme,

    /// Custom theme palette - used when `selected_theme == UiTheme::Custom`.
    #[serde(default)]
    pub custom_theme: CustomThemePalette,

    /// Tutorial system state (step index, active flag, ever-shown flag).
    #[serde(default)]
    pub tutorial: TutorialState,

    /// Whether the low FPS warning dialog is shown
    #[serde(skip)]
    pub show_low_fps_dialog: bool,

    /// Whether the low FPS dialog is suppressed for this session
    #[serde(skip)]
    pub suppress_low_fps_dialog: bool,

    /// Whether the reset confirmation dialog is shown
    #[serde(skip)]
    pub show_reset_dialog: bool,

    /// Whether the adhesion expansion tool is active.
    /// When true, all adhesion rest_lengths are temporarily set to the genome
    /// maximum (5.0) so bonds appear fully stretched - useful for inspecting
    /// creature structure. Does not affect the genome; purely a visual editing aid.
    #[serde(skip)]
    pub adhesion_expansion_active: bool,

    /// Whether the "Saving..." progress popup is shown
    #[serde(skip)]
    pub show_saving_popup: bool,

    /// Whether the "Loading..." progress popup is shown
    #[serde(skip)]
    pub show_loading_popup: bool,

    /// Path chosen by the load file dialog, waiting for the popup to render
    /// before the actual restore work begins.
    #[serde(skip)]
    pub pending_load_path: Option<std::path::PathBuf>,

    /// Whether the save work is ready to start (popup has rendered at least once).
    #[serde(skip)]
    pub pending_save_ready: bool,

    /// Requested mode change (processed by main app loop)
    #[serde(skip)]
    pub mode_request: Option<SimulationMode>,

    /// Whether the Advanced Options overlay is active in GPU mode.
    /// When true, fine-tuning sliders that are not needed for day-to-day use
    /// are revealed inside each panel.
    #[serde(skip)]
    pub show_advanced_options: bool,

    /// Whether the UI chrome (top bar, side rail, status bar, panels) is hidden.
    /// When true only the raw viewport is visible. Toggle with the rail button or Tab key.
    #[serde(skip)]
    pub hide_ui: bool,

    /// Panel pending close confirmation. When Some, a confirmation dialog is shown
    /// asking the user to confirm before the panel is removed from the layout.
    #[serde(skip)]
    pub pending_close_panel: Option<crate::ui::panel::Panel>,
}

impl Default for GlobalUiState {
    fn default() -> Self {
        let mut scene_configs = HashMap::new();
        for mode in SimulationMode::all() {
            scene_configs.insert(*mode, SceneUiConfig::default_for_mode(*mode));
        }

        Self {
            current_mode: SimulationMode::Preview,
            ui_scale: 1.0,
            windows_locked: false,
            lock_tab_bar: false,
            lock_tabs: false,
            lock_close_buttons: false,
            scene_configs,
            occlusion_bias: 0.0,
            occlusion_mip_override: -1,
            occlusion_min_screen_size: 0.0,
            occlusion_min_distance: 0.0,
            occlusion_enabled: true,
            frustum_enabled: true,
            gpu_readbacks_enabled: true,
            gpu_timing_enabled: true,
            field_reports_enabled: true,
            horizontal_fov_degrees: default_horizontal_fov_degrees(),
            camera_sprint_multiplier: default_camera_sprint_multiplier(),
            camera_scroll_sensitivity: default_camera_scroll_sensitivity(),
            field_report_interval_seconds: default_field_report_interval_seconds(),
            gpu_headless_mode: false,
            gpu_headless_auto_speed: false,
            gpu_headless_target_fps: default_headless_target_fps(),
            gpu_headless_min_speed: default_headless_min_speed(),
            gpu_headless_max_speed: default_headless_max_speed(),
            gpu_headless_auto_status: HeadlessAutoStatus::Off,
            show_adhesion_lines: true,
            world_diameter: 395.0,
            lod_scale_factor: 500.0,
            lod_threshold_low: 10.0,
            lod_threshold_medium: 25.0,
            lod_threshold_high: 50.0,
            lod_debug_colors: false,
            world_settings: WorldSettings::default(),
            fluid_settings: FluidSettings::default(),
            selected_theme: UiTheme::default(),
            custom_theme: CustomThemePalette::default(),
            tutorial: TutorialState::default(),
            show_low_fps_dialog: false,
            suppress_low_fps_dialog: false,
            show_reset_dialog: false,
            adhesion_expansion_active: false,
            show_saving_popup: false,
            show_loading_popup: false,
            pending_load_path: None,
            pending_save_ready: false,
            mode_request: None,
            show_advanced_options: false,
            hide_ui: false,
            pending_close_panel: None,
        }
    }
}

impl GlobalUiState {
    /// Get the current scene's config.
    pub fn current_config(&self) -> &SceneUiConfig {
        self.scene_configs
            .get(&self.current_mode)
            .expect("Scene config should exist for current mode")
    }

    /// Get mutable reference to current scene's config.
    pub fn current_config_mut(&mut self) -> &mut SceneUiConfig {
        self.scene_configs
            .get_mut(&self.current_mode)
            .expect("Scene config should exist for current mode")
    }

    /// Check if a panel is visible in the current scene by name.
    pub fn is_panel_visible(&self, panel_name: &str) -> bool {
        let vis = &self.current_config().visibility;
        match panel_name {
            "CellInspector" => vis.show_cell_inspector,
            "GenomeEditor" => vis.show_genome_editor,
            "SceneManager" => vis.show_scene_manager,
            "PerformanceMonitor" => vis.show_performance_monitor,
            "RenderingControls" => vis.show_rendering_controls,
            "TimeScrubber" => vis.show_time_scrubber,
            "ThemeEditor" => vis.show_theme_editor,
            "CameraSettings" => vis.show_camera_settings,
            "LightingSettings" => vis.show_lighting_settings,
            _ => true, // Unknown panels are visible by default
        }
    }

    /// Set panel visibility in the current scene by name.
    pub fn set_panel_visible(&mut self, panel_name: &str, visible: bool) {
        let vis = &mut self.current_config_mut().visibility;
        match panel_name {
            "CellInspector" => vis.show_cell_inspector = visible,
            "GenomeEditor" => vis.show_genome_editor = visible,
            "SceneManager" => vis.show_scene_manager = visible,
            "PerformanceMonitor" => vis.show_performance_monitor = visible,
            "RenderingControls" => vis.show_rendering_controls = visible,
            "TimeScrubber" => vis.show_time_scrubber = visible,
            "ThemeEditor" => vis.show_theme_editor = visible,
            "CameraSettings" => vis.show_camera_settings = visible,
            "LightingSettings" => vis.show_lighting_settings = visible,
            _ => {} // Unknown panels are ignored
        }
    }

    /// Check if a panel is locked in the current scene.
    pub fn is_panel_locked(&self, panel_name: &str) -> bool {
        self.current_config().locked_windows.contains(panel_name)
    }

    /// Set panel lock state in the current scene.
    pub fn set_panel_locked(&mut self, panel_name: &str, locked: bool) {
        let config = self.current_config_mut();
        if locked {
            config.locked_windows.insert(panel_name.to_string());
        } else {
            config.locked_windows.remove(panel_name);
        }
    }

    /// Request a mode switch.
    pub fn request_mode_switch(&mut self, mode: SimulationMode) {
        if mode != self.current_mode {
            self.mode_request = Some(mode);
        }
    }

    /// Reset to embedded default UI state.
    pub fn reset_to_embedded_default(&mut self) {
        *self = Self::load_embedded_default();
        log::info!("Reset UI state to embedded default");
    }

    /// Take the pending mode request, if any.
    pub fn take_mode_request(&mut self) -> Option<SimulationMode> {
        self.mode_request.take()
    }

    /// Save UI state to disk.
    pub fn save(&self) -> Result<(), UiStateSaveError> {
        let path = crate::app_dirs::config_file("ui_state.ron");
        let contents = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Load UI state from disk, or create default if file doesn't exist.
    pub fn load() -> Self {
        let path = crate::app_dirs::config_file("ui_state.ron");

        if path.exists() {
            match Self::load_from_file(&path) {
                Ok(state) => {
                    log::info!("Loaded UI state from {:?}", path);
                    return state;
                }
                Err(e) => {
                    log::warn!("Failed to load UI state: {}. Using embedded default.", e);
                }
            }
        } else {
            log::info!("No saved UI state found, using embedded default");
        }

        // Try to load from embedded default
        Self::load_embedded_default()
    }

    /// Load embedded default UI state.
    fn load_embedded_default() -> Self {
        let embedded_content = include_str!("../../default_ui_state.ron");

        log::info!("Attempting to load embedded default UI state");
        log::debug!(
            "Embedded UI state content length: {} characters",
            embedded_content.len()
        );

        match ron::from_str::<GlobalUiState>(embedded_content) {
            Ok(state) => {
                log::info!("Successfully loaded embedded default UI state");
                state
            }
            Err(e) => {
                log::error!(
                    "Failed to parse embedded default UI state: {}. Using hardcoded default.",
                    e
                );
                log::debug!(
                    "Embedded content preview: {}",
                    &embedded_content[..embedded_content.len().min(200)]
                );
                Self::default()
            }
        }
    }

    /// Load UI state from a specific file.
    fn load_from_file(path: &std::path::PathBuf) -> Result<Self, UiStateLoadError> {
        let contents = std::fs::read_to_string(path)?;
        let mut state: Self = ron::from_str(&contents)?;
        // Clear mode request on load (it's transient)
        state.mode_request = None;
        Ok(state)
    }
}

/// Error type for UI state loading.
#[derive(Debug, thiserror::Error)]
pub enum UiStateLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON parse error: {0}")]
    Ron(#[from] ron::error::SpannedError),
}

/// Error type for UI state saving.
#[derive(Debug, thiserror::Error)]
pub enum UiStateSaveError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("RON serialize error: {0}")]
    Ron(#[from] ron::Error),
}
