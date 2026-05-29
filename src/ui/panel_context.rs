//! Panel context for passing data to panel renderers.
//!
//! This module provides the PanelContext struct which contains all the
//! data and state needed by panels to render their UI and interact with
//! the simulation.

use crate::genome::Genome;
use crate::scene::SceneManager;
use crate::ui::camera::CameraController;
use crate::ui::performance::PerformanceMetrics;
use crate::ui::types::SimulationMode;

/// Request for scene mode changes.
///
/// Panels can request a scene mode change by setting this value.
/// The main application loop will process the request and switch scenes.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SceneModeRequest {
    /// No change requested
    #[default]
    None,
    /// Request to switch to Preview mode
    SwitchToPreview,
    /// Request to switch to GPU mode
    SwitchToGpu,
    /// Request to toggle pause state
    TogglePause,
    /// Request to reset the simulation (cells + fluid)
    Reset,
    /// Request to reset cells only (keep fluid/water)
    ResetCellsOnly,
    /// Request to toggle fast forward mode (2x speed)
    ToggleFastForward,
    /// Request to set simulation speed
    SetSpeed(f32),
    /// Request to set simulation speed and unpause
    SetSpeedAndUnpause(f32),
    /// Request to regenerate fluid test voxels
    RegenerateFluidVoxels,
    /// Request to regenerate fluid mesh using surface nets
    RegenerateFluidMesh,
    /// Request to read back a mutated genome from GPU and load it into the editor
    LoadGenomeFromGpu(u32),
    /// Request to save a GPU scene snapshot (path chosen by file dialog)
    SaveSnapshot,
    /// Request to restore a GPU scene snapshot from the given path
    LoadSnapshot(std::path::PathBuf),
}

impl SceneModeRequest {
    /// Get the target mode if a switch is requested.
    pub fn target_mode(&self) -> Option<SimulationMode> {
        match self {
            SceneModeRequest::SwitchToPreview => Some(SimulationMode::Preview),
            SceneModeRequest::SwitchToGpu => Some(SimulationMode::Gpu),
            _ => None,
        }
    }

    /// Check if a mode switch is requested.
    pub fn is_requested(&self) -> bool {
        !matches!(self, SceneModeRequest::None)
    }

    /// Clear the request.
    pub fn clear(&mut self) {
        *self = SceneModeRequest::None;
    }
}

/// State for the genome editor UI.
///
/// Contains transient UI state for genome editing panels like
/// rename dialogs, color pickers, and widget configurations.
#[derive(Debug, Clone, Default)]
pub struct GenomeEditorState {
    // Mode panel state
    /// Index of mode currently being renamed (None if not renaming)
    pub renaming_mode: Option<usize>,
    /// Buffer for rename text input
    pub rename_buffer: String,
    /// Currently selected mode index (primary selection — the mode whose settings are shown)
    pub selected_mode_index: usize,
    /// All currently selected mode indices (for multi-select editing).
    /// Always contains `selected_mode_index` as one of its entries.
    pub selected_mode_indices: Vec<usize>,
    /// Whether the "copy into" dialog is open
    pub copy_into_dialog_open: bool,
    /// Source mode index for copy operation
    pub copy_into_source: usize,
    /// Color picker state: (mode index, current color)
    pub color_picker_state: Option<(usize, egui::ecolor::Hsva)>,

    // Quaternion ball state
    /// Whether quaternion ball snapping is enabled
    pub qball_snapping: bool,
    /// Locked axis for first quaternion ball (-1 = none, 0 = X, 1 = Y, 2 = Z)
    pub qball1_locked_axis: i32,
    /// Initial distance for first quaternion ball
    pub qball1_initial_distance: f32,
    /// Locked axis for second quaternion ball
    pub qball2_locked_axis: i32,
    /// Initial distance for second quaternion ball
    pub qball2_initial_distance: f32,
    /// Persistent orientation for Child A quaternion ball
    pub child_a_orientation: glam::Quat,
    /// Persistent orientation for Child B quaternion ball
    pub child_b_orientation: glam::Quat,
    
    // Axis tracking for Child A quaternion ball (UI feedback only)
    pub child_a_x_axis_lat: f32,
    pub child_a_x_axis_lon: f32,
    pub child_a_y_axis_lat: f32,
    pub child_a_y_axis_lon: f32,
    pub child_a_z_axis_lat: f32,
    pub child_a_z_axis_lon: f32,
    
    // Axis tracking for Child B quaternion ball (UI feedback only)
    pub child_b_x_axis_lat: f32,
    pub child_b_x_axis_lon: f32,
    pub child_b_y_axis_lat: f32,
    pub child_b_y_axis_lon: f32,
    pub child_b_z_axis_lat: f32,
    pub child_b_z_axis_lon: f32,
    
    // Quaternion ball state for child split angles
    /// Locked axis for Child A split angle quaternion ball
    pub child_a_split_locked_axis: i32,
    /// Initial distance for Child A split angle quaternion ball
    pub child_a_split_initial_distance: f32,
    /// Locked axis for Child B split angle quaternion ball
    pub child_b_split_locked_axis: i32,
    /// Initial distance for Child B split angle quaternion ball
    pub child_b_split_initial_distance: f32,
    
    // Axis tracking for Child A split angle quaternion ball (UI feedback only)
    pub child_a_split_x_axis_lat: f32,
    pub child_a_split_x_axis_lon: f32,
    pub child_a_split_y_axis_lat: f32,
    pub child_a_split_y_axis_lon: f32,
    pub child_a_split_z_axis_lat: f32,
    pub child_a_split_z_axis_lon: f32,
    
    // Axis tracking for Child B split angle quaternion ball (UI feedback only)
    pub child_b_split_x_axis_lat: f32,
    pub child_b_split_x_axis_lon: f32,
    pub child_b_split_y_axis_lat: f32,
    pub child_b_split_y_axis_lon: f32,
    pub child_b_split_z_axis_lat: f32,
    pub child_b_split_z_axis_lon: f32,

    // Quaternion ball manual-input context menu state
    /// Which ball has the context menu open: 0 = none, 1 = Child A, 2 = Child B
    pub qball_context_menu_open: u8,
    /// Manual input buffer: [x, y, z, w]
    pub qball_manual_xyzw: [f64; 4],

    // Keep Adhesion state
    /// Whether Child A should keep adhesion
    pub child_a_keep_adhesion: bool,
    /// Whether Child B should keep adhesion
    pub child_b_keep_adhesion: bool,

    // Circular slider state
    /// Whether angle snapping is enabled for circular sliders
    pub enable_snapping: bool,

    // Time slider state
    /// Current time value for preview (0-60 seconds) — purely UI-driven, never written by simulation
    pub time_value: f32,
    /// Maximum preview duration (60 seconds)
    pub max_preview_duration: f32,
    /// Whether the time slider is being dragged
    pub time_slider_dragging: bool,
    /// Actual simulation time reached (read-only from sim, used for progress bar display)
    pub resim_display_time: f32,
    
    // Cave system parameters
    pub cave_density: f32,
    pub cave_scale: f32,
    pub cave_octaves: u32,
    pub cave_persistence: f32,
    pub cave_threshold: f32,
    pub cave_smoothness: f32,
    pub cave_seed: u32,
    pub cave_resolution: u32,
    pub cave_params_dirty: bool,
    
    // Fluid simulation parameters
    pub fluid_gravity: f32,
    pub fluid_gravity_x: bool,
    pub fluid_gravity_y: bool,
    pub fluid_gravity_z: bool,
    pub fluid_vorticity_epsilon: f32,
    pub fluid_pressure_iterations: u32,
    /// Per-fluid-type lateral flow probabilities for fluid simulation (0.0 to 1.0)
    /// Index: 0=Empty (unused), 1=Water, 2=Lava, 3=Steam
    pub fluid_lateral_flow_probabilities: [f32; 4],
    /// Condensation probability for steam to water conversion (0.0 to 1.0)
    pub fluid_condensation_probability: f32,
    /// Vaporization probability for water to steam conversion (0.0 to 1.0)
    pub fluid_vaporization_probability: f32,
    /// Nutrient particle density for noise-based spawning (0.0 to 1.0)
    pub nutrient_density: f32,
    /// Nutrient epoch duration in seconds
    pub nutrient_epoch_duration: f32,
    /// Nutrient epoch spacing in seconds (< duration = overlap)
    pub nutrient_epoch_spacing: f32,
    /// Fraction of epoch for spawn ramp (0.0–1.0)
    pub nutrient_spawn_end: f32,
    /// Fraction of epoch where despawn starts (0.0–1.0)
    pub nutrient_despawn_start: f32,
    
    // Fluid visualization
    pub fluid_show_voxel_grid: bool,
    pub fluid_show_solid_only: bool,
    pub fluid_show_wireframe: bool,
    pub fluid_color_mode: u32,
    
    // Fluid statistics (read-only, updated from GPU)
    pub fluid_solid_count: u32,
    pub fluid_empty_count: u32,
    pub fluid_water_count: u32,
    pub fluid_lava_count: u32,
    pub fluid_steam_count: u32,
    pub fluid_memory_usage_mb: f32,
    pub fluid_water_mass: f32,
    pub fluid_lava_mass: f32,
    pub fluid_steam_mass: f32,
    
    // Fluid initialization
    pub fluid_water_percent: f32,
    pub fluid_lava_percent: f32,
    pub fluid_steam_percent: f32,
    
    // Fluid type selection
    pub selected_fluid_type: u32, // 0=Empty, 1=Water, 2=Lava, 3=Steam
    
    // Fluid visualization toggle
    pub fluid_show_test_voxels: bool,
    /// Whether to render fluid as smooth mesh (surface nets)
    pub fluid_show_mesh: bool,
    /// Whether to enable continuous fluid spawning
    pub fluid_continuous_spawn: bool,
    
    // Fluid mesh settings
    /// Iso level for surface extraction (0.0-1.0)
    pub fluid_iso_level: f32,
    // Fluid lighting settings
    /// Ambient light strength
    pub fluid_ambient: f32,
    /// Diffuse light strength
    pub fluid_diffuse: f32,
    /// Specular intensity
    pub fluid_specular: f32,
    /// Shininess (specular power)
    pub fluid_shininess: f32,
    /// Fresnel effect strength
    pub fluid_fresnel: f32,
    /// Fresnel power
    pub fluid_fresnel_power: f32,
    /// Reflection strength
    pub fluid_reflection: f32,
    /// Reflection brightness multiplier
    pub fluid_reflection_brightness: f32,
    /// Overall alpha/transparency
    pub fluid_alpha: f32,
    /// Rim light strength
    pub fluid_rim: f32,
    /// Wave height (displacement amplitude along normal)
    pub fluid_wave_height: f32,
    /// Wave animation speed
    pub fluid_wave_speed: f32,
    /// Perlin noise spatial scale (lower = larger waves spanning more voxels)
    pub fluid_noise_scale: f32,
    /// Number of fractal noise octaves
    pub fluid_noise_octaves: f32,
    /// Frequency multiplier per octave
    pub fluid_noise_lacunarity: f32,
    /// Amplitude multiplier per octave
    pub fluid_noise_persistence: f32,
    
    /// Flag to indicate mesh params need update
    pub fluid_mesh_params_dirty: bool,
    /// Flag to indicate mesh needs regeneration (iso level changed)
    pub fluid_mesh_needs_regen: bool,
    
    // Light field & volumetric fog settings
    /// Light direction (x, y, z) - will be normalized
    pub light_dir: [f32; 3],
    /// Whether volumetric fog is enabled
    pub show_volumetric_fog: bool,
    /// Fog density (0.0 = no fog, 1.0 = dense)
    pub fog_density: f32,
    /// Number of fog ray march steps
    pub fog_steps: u32,
    /// Light color RGB
    pub light_color: [f32; 3],
    /// Light intensity multiplier
    pub light_intensity: f32,
    /// Fog/shadow color RGB
    pub fog_color: [f32; 3],
    /// Scattering anisotropy (Henyey-Greenstein g, 0 = isotropic, ~0.7 = forward scatter)
    pub fog_scattering_anisotropy: f32,
    /// Fog absorption coefficient
    pub fog_absorption: f32,
    /// Height fog density
    pub fog_height_density: f32,
    /// Height fog falloff
    pub fog_height_falloff: f32,
    /// Light field max ray march steps
    pub light_field_max_steps: u32,
    /// Light field step size multiplier
    pub light_field_step_size: f32,
    /// Absorption per solid voxel
    pub light_field_absorption_solid: f32,
    /// Absorption per cell-occupied voxel
    pub light_field_absorption_cell: f32,
    /// Ambient light floor
    pub light_field_ambient_floor: f32,
    /// Photocyte mass gain rate at full light
    pub photocyte_mass_per_second: f32,
    /// Photocyte minimum light threshold
    pub photocyte_min_light_threshold: f32,
    /// Flag to indicate light params need GPU update
    pub light_params_dirty: bool,
    
    // Depth of field settings
    /// Whether depth of field is enabled
    pub show_dof: bool,
    /// Focal distance from camera (world units)
    pub dof_focal_distance: f32,
    /// Range around focal distance that stays sharp (world units)
    pub dof_focal_range: f32,
    /// Maximum blur radius in pixels
    pub dof_max_blur_radius: f32,
    /// Blur intensity multiplier
    pub dof_blur_strength: f32,
    
    // Sun renderer settings
    /// Whether the procedural sun is visible
    pub show_sun: bool,
    /// Sun color (RGB)
    pub sun_color: [f32; 3],
    /// Sun angular radius (visual size)
    pub sun_angular_radius: f32,
    /// Sun intensity
    pub sun_intensity: f32,
    
    // Shadow settings
    /// Whether surface shadows are enabled
    pub shadow_enabled: bool,
    /// Shadow strength (0.0 = no shadows, 1.0 = full shadows)
    pub shadow_strength: f32,
    /// Shadow quality (0.0 = low, 1.0 = high - affects sample offset distance)
    pub shadow_quality: f32,
    // Caustic settings
    /// Caustic intensity (0.0 = off, 1.0 = full)
    pub caustic_intensity: f32,
    /// Caustic pattern scale
    pub caustic_scale: f32,
    /// Caustic animation speed
    pub caustic_speed: f32,

    // Organism skin settings
    /// Whether organism skins are enabled
    pub show_organism_skins: bool,

    // Moss system settings
    /// Whether moss is enabled on cave walls
    pub show_moss: bool,
    /// Moss growth rate (units per second at full light)
    pub moss_growth_rate: f32,
    /// Moss erosion rate from water flow (units per second)
    pub moss_erosion_rate: f32,
    /// Moss decay rate when conditions not met (units per second)
    pub moss_decay_rate: f32,
    /// Minimum light level for moss growth
    pub moss_min_light: f32,
    /// Nutrients gained per unit of moss consumed by phagocytes
    pub moss_nutrient_per_moss: f32,
    /// Moss consumption rate by phagocytes (units per second)
    pub moss_consume_rate: f32,
    /// Wetness evaporation rate (how fast moisture memory fades)
    pub moss_wetness_evaporation: f32,
    /// Parallax depth scale for moss visual effect
    pub moss_parallax_depth: f32,
    /// Moss texture scale (higher = finer detail)
    pub moss_scale: f32,
    /// Moss noise type (0=value, 1=worley, 2=ridged)
    pub moss_noise_type: u32,
    /// Moss noise primary frequency
    pub moss_noise_frequency: f32,
    /// Moss noise lacunarity (frequency multiplier between octaves)
    pub moss_noise_lacunarity: f32,
    /// Moss height sharpness lower bound (smoothstep)
    pub moss_height_sharpness_low: f32,
    /// Moss height sharpness upper bound (smoothstep)
    pub moss_height_sharpness_high: f32,
    /// Moss bump/normal map strength
    pub moss_bump_strength: f32,
    /// Moss dark (base/shadow) color RGB
    pub moss_color_dark: [f32; 3],
    /// Moss bright (tip/highlight) color RGB
    pub moss_color_bright: [f32; 3],
    /// Water search radius in voxels for moss growth
    pub moss_water_radius: f32,
    /// Flag to sync moss params to GPU
    pub moss_params_dirty: bool,

    // ── Boulder / Mossrock settings ───────────────────────────────────────────
    /// Whether mossrocks are enabled
    pub show_boulders: bool,
    /// Target number of boulders to maintain
    pub boulder_target_count: u32,
    /// Initial moss store per boulder (nutrients)
    pub boulder_initial_moss: f32,
    /// Boulder radius (legacy single value)
    pub boulder_radius: f32,
    /// Boulder size gate half-saturation constant (cells)
    pub boulder_size_gate: f32,
    /// Seconds between boulder spawn attempts
    pub boulder_spawn_interval: f32,
    /// Gravity multiplier when boulder is submerged (0=float, 1=full gravity)
    pub boulder_buoyancy: f32,
    /// Minimum boulder radius
    pub boulder_radius_min: f32,
    /// Maximum boulder radius
    pub boulder_radius_max: f32,
    /// Minimum boulder moss store (nutrients)
    pub boulder_moss_min: f32,
    /// Maximum boulder moss store (nutrients)
    pub boulder_moss_max: f32,
    /// Skin radius scale (multiplier on cell radius for skin thickness)
    pub skin_radius_scale: f32,
    /// Iso level for organism skin surface extraction
    pub skin_iso_level: f32,
    /// Skin base colour (RGB)
    pub skin_base_color: [f32; 3],
    /// Skin alpha (translucency)
    pub skin_alpha: f32,
    /// Skin subsurface scattering strength
    pub skin_sss_strength: f32,
    /// Skin rim light strength
    pub skin_rim_strength: f32,
    
    // Orientation gizmo state
    /// Whether the orientation gizmo is visible
    pub gizmo_visible: bool,
    
    // Split ring state
    /// Whether the split rings are visible
    pub split_rings_visible: bool,
    
    // Radial menu state (GPU scene only)
    /// Radial menu state for tool selection
    pub radial_menu: crate::ui::radial_menu::RadialMenuState,
    
    /// Distance from camera to dragged cell (for maintaining depth during drag)
    pub drag_distance: f32,
    
    // Cell type visuals state
    /// Cell outline width for cel-shaded black outline effect (0.0 = off, 0.3 = thick)
    pub cell_outline_width: f32,
    /// Visual settings per cell type (indexed by CellType)
    pub cell_type_visuals: Vec<crate::cell::types::CellTypeVisuals>,
    /// Currently selected cell type index for the visuals panel
    pub selected_cell_type: usize,
    /// Mode graph state for node visualization
    pub mode_graph_state: crate::genome::node_graph::ModeGraphState,
    /// Request to toggle the mode graph panel
    pub toggle_mode_graph_panel: bool,
    /// Stored location of mode graph panel when hidden (surface, node, tab indices)
    pub mode_graph_panel_location: Option<(egui_dock::SurfaceIndex, egui_dock::NodeIndex, egui_dock::TabIndex)>,

    /// Request to generate a procedural genome with this seed (set by rail button, consumed in end_frame).
    pub procedural_genome_seed: Option<u64>,

    /// Request to toggle water fill (fluid_continuous_spawn) from the rail button.
    /// Processed in end_frame after the dock renders so scene_manager is accessible.
    pub request_toggle_water: bool,

    /// Request to toggle volumetric fog (show_volumetric_fog) from the rail button.
    /// Processed in end_frame after the dock renders so scene_manager is accessible.
    pub request_toggle_fog: bool,

    /// Request to take a screenshot of the current viewport.
    /// Processed in app.rs after the frame is presented.
    pub request_screenshot: bool,

    /// Request to capture a GIF thumbnail of the current genome.
    /// Processed in app.rs after the frame is presented.
    pub request_gif_capture: bool,
    /// The path the genome was just saved to — used to name the GIF correctly.
    pub gif_capture_save_path: Option<std::path::PathBuf>,

    /// Request to open the genome browser in save mode.
    pub open_genome_browser_save: bool,

    /// Request to open the genome browser in load mode.
    pub open_genome_browser_load: bool,

    /// Set to true for one frame after a genome is loaded from the browser.
    /// Prevents the pre-frame preview-scene sync from overwriting the loaded genome.
    pub genome_just_loaded: bool,

    /// After a successful genome save, automatically trigger GIF capture.
    pub pending_gif_after_save: bool,

    /// When set, show an overwrite-confirmation dialog before saving.
    /// Contains the path that would be overwritten.
    pub overwrite_confirm_path: Option<std::path::PathBuf>,

    /// Multi-frame GIF capture state. None = idle.
    pub gif_capture: Option<crate::gif_capture::GifCaptureState>,

    /// Whether to show the "New Genome" confirmation dialog.
    pub confirm_new_genome: bool,
    /// Whether the name field in the top bar should flash red (empty name on save attempt).
    pub name_field_error: bool,
    /// Timer for the name field error flash.
    pub name_field_error_timer: f32,
    /// Whether the "name your genome" dialog is open (triggered when saving with empty name).
    pub show_name_dialog: bool,
    /// Text buffer for the name dialog's custom name field.
    pub name_dialog_buffer: String,
    /// Variation seed for the procedural name — incremented each time the user clicks Regenerate.
    pub name_dialog_seed: u64,
    /// Names already in use on disk — populated when the dialog opens so suggestions avoid clashes.
    pub name_dialog_used_names: Vec<String>,
    /// Whether the name dialog text field has been focused this session (prevents re-stealing focus).
    pub name_dialog_focused: bool,

    /// Screen-space rects for each named panel, updated every frame by the
    /// panel render functions. Used by the tutorial overlay to position the
    /// schematic pointer line.
    ///
    /// Keys match [`TutorialTarget::panel_key()`] values:
    /// `"Modes"`, `"NameTypeEditor"`, `"ParentSettings"`,
    /// `"AdhesionSettings"`, `"TimeSlider"`, `"SceneManager"`.
    pub panel_rects: std::collections::HashMap<String, egui::Rect>,
}

impl GenomeEditorState {
    /// Create a new genome editor state with default values.
    pub fn new() -> Self {
        let (cave_density, cave_scale, cave_octaves, cave_persistence, cave_threshold, 
             cave_smoothness, cave_seed, cave_resolution,
             show_moss, moss_growth_rate, moss_erosion_rate, moss_decay_rate,
             moss_min_light, moss_nutrient_per_moss, moss_consume_rate,
             moss_wetness_evaporation, moss_parallax_depth, moss_scale, moss_water_radius,
             moss_noise_type, moss_noise_frequency, moss_noise_lacunarity,
             moss_height_sharpness_low, moss_height_sharpness_high, moss_bump_strength,
             moss_color_dark, moss_color_bright,
             show_boulders, boulder_target_count, boulder_initial_moss, boulder_radius,
             boulder_size_gate, boulder_spawn_interval, boulder_buoyancy,
             boulder_radius_min, boulder_radius_max, boulder_moss_min, boulder_moss_max,
             ) = Self::load_cave_settings();
        
        let (fluid_gravity, fluid_gravity_x, fluid_gravity_y, fluid_gravity_z, 
             fluid_vorticity_epsilon, fluid_pressure_iterations, fluid_lateral_flow_probabilities,
             _fluid_continuous_spawn, selected_fluid_type, fluid_condensation_probability, fluid_vaporization_probability, nutrient_density) = Self::load_fluid_settings();
        
        let (light_dir, show_volumetric_fog, fog_density, fog_steps, light_color, light_intensity,
             fog_color, fog_scattering_anisotropy, fog_absorption, fog_height_density, fog_height_falloff,
             light_field_max_steps, light_field_step_size, light_field_absorption_solid,
             light_field_absorption_cell, light_field_ambient_floor, show_sun, sun_color, sun_angular_radius, sun_intensity,
             shadow_enabled, shadow_strength, shadow_quality,
             caustic_intensity, caustic_scale, caustic_speed,
             photocyte_mass_per_second, photocyte_min_light_threshold) = Self::load_light_settings();
        
        let (fluid_iso_level, fluid_ambient, fluid_diffuse, fluid_specular, fluid_shininess,
             fluid_fresnel, fluid_fresnel_power, fluid_reflection, fluid_alpha, fluid_rim,
             fluid_wave_height, fluid_wave_speed, fluid_noise_scale, fluid_noise_octaves,
             fluid_noise_lacunarity, fluid_noise_persistence) = Self::load_fluid_render_settings();
        
        let (cell_type_visuals, cell_outline_width) = crate::cell::types::CellTypeVisualsStore::load();
        
        let state = Self {
            renaming_mode: None,
            rename_buffer: String::new(),
            selected_mode_index: 0,
            selected_mode_indices: vec![0],
            copy_into_dialog_open: false,
            copy_into_source: 0,
            color_picker_state: None,
            qball_snapping: true,
            qball1_locked_axis: -1,
            qball1_initial_distance: 1.0,
            qball2_locked_axis: -1,
            qball2_initial_distance: 1.0,
            child_a_orientation: glam::Quat::IDENTITY,
            child_b_orientation: glam::Quat::IDENTITY,
            child_a_x_axis_lat: 0.0,
            child_a_x_axis_lon: 0.0,
            child_a_y_axis_lat: 0.0,
            child_a_y_axis_lon: 0.0,
            child_a_z_axis_lat: 0.0,
            child_a_z_axis_lon: 0.0,
            child_b_x_axis_lat: 0.0,
            child_b_x_axis_lon: 0.0,
            child_b_y_axis_lat: 0.0,
            child_b_y_axis_lon: 0.0,
            child_b_z_axis_lat: 0.0,
            child_b_z_axis_lon: 0.0,
            
            // Quaternion ball state for child split angles
            child_a_split_locked_axis: -1,
            child_a_split_initial_distance: 1.0,
            child_b_split_locked_axis: -1,
            child_b_split_initial_distance: 1.0,
            
            // Axis tracking for Child A split angle quaternion ball (UI feedback only)
            child_a_split_x_axis_lat: 0.0,
            child_a_split_x_axis_lon: 0.0,
            child_a_split_y_axis_lat: 0.0,
            child_a_split_y_axis_lon: 0.0,
            child_a_split_z_axis_lat: 0.0,
            child_a_split_z_axis_lon: 0.0,
            
            // Axis tracking for Child B split angle quaternion ball (UI feedback only)
            child_b_split_x_axis_lat: 0.0,
            child_b_split_x_axis_lon: 0.0,
            child_b_split_y_axis_lat: 0.0,
            child_b_split_y_axis_lon: 0.0,
            child_b_split_z_axis_lat: 0.0,
            child_b_split_z_axis_lon: 0.0,
            qball_context_menu_open: 0,
            qball_manual_xyzw: [0.0, 0.0, 0.0, 1.0],
            child_a_keep_adhesion: false,
            child_b_keep_adhesion: false,
            enable_snapping: true,
            time_value: 0.0,
            max_preview_duration: 60.0,
            time_slider_dragging: false,
            resim_display_time: 0.0,
            cave_density,
            cave_scale,
            cave_octaves,
            cave_persistence,
            cave_threshold,
            cave_smoothness,
            cave_seed,
            cave_resolution,
            cave_params_dirty: false,
            fluid_gravity,
            fluid_gravity_x,
            fluid_gravity_y,
            fluid_gravity_z,
            fluid_vorticity_epsilon,
            fluid_pressure_iterations,
            fluid_lateral_flow_probabilities,
            fluid_condensation_probability,
            fluid_vaporization_probability,
            nutrient_density,
            nutrient_epoch_duration: 10.0,
            nutrient_epoch_spacing: 7.0,
            nutrient_spawn_end: 0.4,
            nutrient_despawn_start: 0.6,
            fluid_show_voxel_grid: true,
            fluid_show_solid_only: false,
            fluid_show_wireframe: false,
            fluid_color_mode: 0,
            fluid_solid_count: 0,
            fluid_empty_count: 0,
            fluid_water_count: 0,
            fluid_lava_count: 0,
            fluid_steam_count: 0,
            fluid_memory_usage_mb: 0.0,
            fluid_water_mass: 0.0,
            fluid_lava_mass: 0.0,
            fluid_steam_mass: 0.0,
            fluid_water_percent: 25.0,
            fluid_lava_percent: 25.0,
            fluid_steam_percent: 25.0,
            selected_fluid_type,
            fluid_show_test_voxels: false,
            fluid_show_mesh: true,
            fluid_continuous_spawn: false,  // Always start disabled by default
            fluid_iso_level,
            fluid_ambient,
            fluid_diffuse,
            fluid_specular,
            fluid_shininess,
            fluid_fresnel,
            fluid_fresnel_power,
            fluid_reflection,
            fluid_reflection_brightness: 10.0,
            fluid_alpha,
            fluid_rim,
            fluid_wave_height,
            fluid_wave_speed,
            fluid_noise_scale,
            fluid_noise_octaves,
            fluid_noise_lacunarity,
            fluid_noise_persistence,
            fluid_mesh_params_dirty: true,  // Trigger GPU upload on first frame
            fluid_mesh_needs_regen: false,
            light_dir,
            show_volumetric_fog,
            fog_density,
            fog_steps,
            light_color,
            light_intensity,
            fog_color,
            fog_scattering_anisotropy,
            fog_absorption,
            fog_height_density,
            fog_height_falloff,
            light_field_max_steps,
            light_field_step_size,
            light_field_absorption_solid,
            light_field_absorption_cell,
            light_field_ambient_floor,
            photocyte_mass_per_second,
            photocyte_min_light_threshold,
            light_params_dirty: true,
            show_dof: false,
            dof_focal_distance: 50.0,
            dof_focal_range: 30.0,
            dof_max_blur_radius: 8.0,
            dof_blur_strength: 1.0,
            show_sun,
            sun_color,
            sun_angular_radius,
            sun_intensity,
            shadow_enabled,
            shadow_strength,
            shadow_quality,
            caustic_intensity,
            caustic_scale,
            caustic_speed,
            show_organism_skins: false,
            show_moss,
            moss_growth_rate,
            moss_erosion_rate,
            moss_decay_rate,
            moss_min_light,
            moss_nutrient_per_moss,
            moss_consume_rate,
            moss_wetness_evaporation,
            moss_parallax_depth,
            moss_scale,
            moss_noise_type,
            moss_noise_frequency,
            moss_noise_lacunarity,
            moss_height_sharpness_low,
            moss_height_sharpness_high,
            moss_bump_strength,
            moss_color_dark,
            moss_color_bright,
            moss_water_radius,
            moss_params_dirty: false,
            show_boulders,
            boulder_target_count,
            boulder_initial_moss,
            boulder_radius,
            boulder_size_gate,
            boulder_spawn_interval,
            boulder_buoyancy,
            boulder_radius_min,
            boulder_radius_max,
            boulder_moss_min,
            boulder_moss_max,
            skin_radius_scale: 1.5,
            skin_iso_level: 0.5,
            skin_base_color: [0.85, 0.55, 0.35],
            skin_alpha: 0.55,
            skin_sss_strength: 0.5,
            skin_rim_strength: 0.35,
            gizmo_visible: true,
            split_rings_visible: true,
            radial_menu: crate::ui::radial_menu::RadialMenuState::new(),
            drag_distance: 0.0,
            cell_outline_width,
            cell_type_visuals,
            selected_cell_type: 0,
            mode_graph_state: crate::genome::node_graph::ModeGraphState::new(),
            toggle_mode_graph_panel: false,
            mode_graph_panel_location: None,
            procedural_genome_seed: None,
            request_toggle_water: false,
            request_toggle_fog: false,
            request_screenshot: false,
            request_gif_capture: false,
            gif_capture_save_path: None,
            open_genome_browser_save: false,
            open_genome_browser_load: false,
            genome_just_loaded: false,
            pending_gif_after_save: false,
            overwrite_confirm_path: None,
            gif_capture: None,
            name_field_error: false,
            name_field_error_timer: 0.0,
            confirm_new_genome: false,
            show_name_dialog: false,
            name_dialog_buffer: String::new(),
            name_dialog_seed: 0,
            name_dialog_used_names: Vec::new(),
            name_dialog_focused: false,
            panel_rects: std::collections::HashMap::new(),
        };
        state
    }

    /// Save cell type visuals to disk.
    pub fn save_cell_type_visuals(&self) {
        if let Err(e) = crate::cell::types::CellTypeVisualsStore::save(&self.cell_type_visuals, self.cell_outline_width) {
            log::error!("Failed to save cell type visuals: {}", e);
        }
    }
    
    /// Save cave settings to disk.
    pub fn save_cave_settings(&self) {
        if let Err(e) = Self::save_cave_settings_to_file(
            self.cave_density,
            self.cave_scale,
            self.cave_octaves,
            self.cave_persistence,
            self.cave_threshold,
            self.cave_smoothness,
            self.cave_seed,
            self.cave_resolution,
            self.show_moss,
            self.moss_growth_rate,
            self.moss_erosion_rate,
            self.moss_decay_rate,
            self.moss_min_light,
            self.moss_nutrient_per_moss,
            self.moss_consume_rate,
            self.moss_wetness_evaporation,
            self.moss_parallax_depth,
            self.moss_scale,
            self.moss_water_radius,
            self.moss_noise_type,
            self.moss_noise_frequency,
            self.moss_noise_lacunarity,
            self.moss_height_sharpness_low,
            self.moss_height_sharpness_high,
            self.moss_bump_strength,
            self.moss_color_dark,
            self.moss_color_bright,
            self.show_boulders,
            self.boulder_target_count,
            self.boulder_initial_moss,
            self.boulder_radius,
            self.boulder_size_gate,
            self.boulder_spawn_interval,
            self.boulder_buoyancy,
            self.boulder_radius_min,
            self.boulder_radius_max,
            self.boulder_moss_min,
            self.boulder_moss_max,
        ) {
            log::error!("Failed to save cave settings: {}", e);
        }
    }
    
    /// Save fluid settings to disk.
    pub fn save_fluid_settings(&self) {
        if let Err(e) = Self::save_fluid_settings_to_file(
            self.fluid_gravity,
            self.fluid_gravity_x,
            self.fluid_gravity_y,
            self.fluid_gravity_z,
            self.fluid_vorticity_epsilon,
            self.fluid_pressure_iterations,
            self.fluid_lateral_flow_probabilities,
            self.fluid_continuous_spawn,
            self.selected_fluid_type,
            self.fluid_condensation_probability,
            self.fluid_vaporization_probability,
            self.nutrient_density,
        ) {
            log::error!("Failed to save fluid settings: {}", e);
        }
    }
    
    fn save_cave_settings_to_file(
        density: f32,
        scale: f32,
        octaves: u32,
        persistence: f32,
        threshold: f32,
        smoothness: f32,
        seed: u32,
        resolution: u32,
        show_moss: bool,
        moss_growth_rate: f32,
        moss_erosion_rate: f32,
        moss_decay_rate: f32,
        moss_min_light: f32,
        moss_nutrient_per_moss: f32,
        moss_graze_cooldown: f32,
        moss_wetness_evaporation: f32,
        moss_parallax_depth: f32,
        moss_scale: f32,
        moss_water_radius: f32,
        moss_noise_type: u32,
        moss_noise_frequency: f32,
        moss_noise_lacunarity: f32,
        moss_height_sharpness_low: f32,
        moss_height_sharpness_high: f32,
        moss_bump_strength: f32,
        moss_color_dark: [f32; 3],
        moss_color_bright: [f32; 3],
        show_boulders: bool,
        boulder_target_count: u32,
        boulder_initial_moss: f32,
        boulder_radius: f32,
        boulder_size_gate: f32,
        boulder_spawn_interval: f32,
        boulder_buoyancy: f32,
        boulder_radius_min: f32,
        boulder_radius_max: f32,
        boulder_moss_min: f32,
        boulder_moss_max: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct CaveSettings {
            density: f32,
            scale: f32,
            octaves: u32,
            persistence: f32,
            threshold: f32,
            smoothness: f32,
            seed: u32,
            resolution: u32,
            show_moss: bool,
            moss_growth_rate: f32,
            moss_erosion_rate: f32,
            moss_decay_rate: f32,
            moss_min_light: f32,
            moss_nutrient_per_moss: f32,
            moss_graze_cooldown: f32,
            moss_wetness_evaporation: f32,
            moss_parallax_depth: f32,
            moss_scale: f32,
            moss_water_radius: f32,
            moss_noise_type: u32,
            moss_noise_frequency: f32,
            moss_noise_lacunarity: f32,
            moss_height_sharpness_low: f32,
            moss_height_sharpness_high: f32,
            moss_bump_strength: f32,
            moss_color_dark: [f32; 3],
            moss_color_bright: [f32; 3],
            show_boulders: bool,
            boulder_target_count: u32,
            boulder_initial_moss: f32,
            boulder_radius: f32,
            boulder_size_gate: f32,
            boulder_spawn_interval: f32,
            boulder_buoyancy: f32,
            boulder_radius_min: f32,
            boulder_radius_max: f32,
            boulder_moss_min: f32,
            boulder_moss_max: f32,
        }
        
        let settings = CaveSettings {
            density,
            scale,
            octaves,
            persistence,
            threshold,
            smoothness,
            seed,
            resolution,
            show_moss,
            moss_growth_rate,
            moss_erosion_rate,
            moss_decay_rate,
            moss_min_light,
            moss_nutrient_per_moss,
            moss_graze_cooldown,
            moss_wetness_evaporation,
            moss_parallax_depth,
            moss_scale,
            moss_water_radius,
            moss_noise_type,
            moss_noise_frequency,
            moss_noise_lacunarity,
            moss_height_sharpness_low,
            moss_height_sharpness_high,
            moss_bump_strength,
            moss_color_dark,
            moss_color_bright,
            show_boulders,
            boulder_target_count,
            boulder_initial_moss,
            boulder_radius,
            boulder_size_gate,
            boulder_spawn_interval,
            boulder_buoyancy,
            boulder_radius_min,
            boulder_radius_max,
            boulder_moss_min,
            boulder_moss_max,
        };
        
        let path = crate::app_dirs::config_file("cave_settings.ron");
        let contents = ron::ser::to_string_pretty(&settings, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    fn save_fluid_settings_to_file(
        gravity: f32,
        gravity_x: bool,
        gravity_y: bool,
        gravity_z: bool,
        vorticity_epsilon: f32,
        pressure_iterations: u32,
        lateral_flow_probabilities: [f32; 4],
        continuous_spawn: bool,
        selected_fluid_type: u32,
        condensation_probability: f32,
        vaporization_probability: f32,
        nutrient_density: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct FluidSettings {
            gravity: f32,
            gravity_x: bool,
            gravity_y: bool,
            gravity_z: bool,
            vorticity_epsilon: f32,
            pressure_iterations: u32,
            lateral_flow_probabilities: [f32; 4],
            continuous_spawn: bool,
            selected_fluid_type: u32,
            condensation_probability: f32,
            vaporization_probability: f32,
            nutrient_density: f32,
        }
        
        let settings = FluidSettings {
            gravity,
            gravity_x,
            gravity_y,
            gravity_z,
            vorticity_epsilon,
            pressure_iterations,
            lateral_flow_probabilities,
            continuous_spawn,
            selected_fluid_type,
            condensation_probability,
            vaporization_probability,
            nutrient_density,
        };
        
        let path = crate::app_dirs::config_file("fluid_settings.ron");
        let contents = ron::ser::to_string_pretty(&settings, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Load cave settings from disk, or return defaults if file doesn't exist.
    #[allow(clippy::type_complexity)]
    pub fn load_cave_settings() -> (f32, f32, u32, f32, f32, f32, u32, u32,
                                     bool, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
                                     u32, f32, f32, f32, f32, f32, [f32; 3], [f32; 3],
                                     bool, u32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        #[derive(serde::Deserialize)]
        struct CaveSettings {
            density: f32,
            scale: f32,
            octaves: u32,
            persistence: f32,
            threshold: f32,
            smoothness: f32,
            seed: u32,
            #[serde(default = "default_resolution")]
            resolution: u32,
            #[serde(default = "default_show_moss")]
            show_moss: bool,
            #[serde(default = "default_moss_growth_rate")]
            moss_growth_rate: f32,
            #[serde(default = "default_moss_erosion_rate")]
            moss_erosion_rate: f32,
            #[serde(default = "default_moss_decay_rate")]
            moss_decay_rate: f32,
            #[serde(default = "default_moss_min_light")]
            moss_min_light: f32,
            #[serde(default = "default_moss_nutrient_per_moss")]
            moss_nutrient_per_moss: f32,
            #[serde(default = "default_moss_graze_cooldown")]
            moss_graze_cooldown: f32,
            #[serde(default = "default_moss_wetness_evaporation")]
            moss_wetness_evaporation: f32,
            #[serde(default = "default_moss_parallax_depth")]
            moss_parallax_depth: f32,
            #[serde(default = "default_moss_scale")]
            moss_scale: f32,
            #[serde(default = "default_moss_water_radius")]
            moss_water_radius: f32,
            #[serde(default = "default_moss_noise_type")]
            moss_noise_type: u32,
            #[serde(default = "default_moss_noise_frequency")]
            moss_noise_frequency: f32,
            #[serde(default = "default_moss_noise_lacunarity")]
            moss_noise_lacunarity: f32,
            #[serde(default = "default_moss_height_sharpness_low")]
            moss_height_sharpness_low: f32,
            #[serde(default = "default_moss_height_sharpness_high")]
            moss_height_sharpness_high: f32,
            #[serde(default = "default_moss_bump_strength")]
            moss_bump_strength: f32,
            #[serde(default = "default_moss_color_dark")]
            moss_color_dark: [f32; 3],
            #[serde(default = "default_moss_color_bright")]
            moss_color_bright: [f32; 3],
            #[serde(default = "default_show_boulders")]
            show_boulders: bool,
            #[serde(default = "default_boulder_target_count")]
            boulder_target_count: u32,
            #[serde(default = "default_boulder_initial_moss")]
            boulder_initial_moss: f32,
            #[serde(default = "default_boulder_radius")]
            boulder_radius: f32,
            #[serde(default = "default_boulder_size_gate")]
            boulder_size_gate: f32,
            #[serde(default = "default_boulder_spawn_interval")]
            boulder_spawn_interval: f32,
            #[serde(default = "default_boulder_buoyancy")]
            boulder_buoyancy: f32,
            #[serde(default = "default_boulder_radius_min")]
            boulder_radius_min: f32,
            #[serde(default = "default_boulder_radius_max")]
            boulder_radius_max: f32,
            #[serde(default = "default_boulder_moss_min")]
            boulder_moss_min: f32,
            #[serde(default = "default_boulder_moss_max")]
            boulder_moss_max: f32,
        }
        
        fn default_resolution() -> u32 { 128 }
        fn default_show_moss() -> bool { true }
        fn default_moss_growth_rate() -> f32 { 0.15 }
        fn default_moss_erosion_rate() -> f32 { 0.3 }
        fn default_moss_decay_rate() -> f32 { 0.05 }
        fn default_moss_min_light() -> f32 { 0.05 }
        fn default_moss_nutrient_per_moss() -> f32 { 50.0 }
        fn default_moss_graze_cooldown() -> f32 { 5.0 }
        fn default_moss_wetness_evaporation() -> f32 { 0.02 }
        fn default_moss_parallax_depth() -> f32 { 0.08 }
        fn default_moss_scale() -> f32 { 0.15 }
        fn default_moss_water_radius() -> f32 { 20.0 }
        fn default_moss_noise_type() -> u32 { 0 }
        fn default_moss_noise_frequency() -> f32 { 18.0 }
        fn default_moss_noise_lacunarity() -> f32 { 2.5 }
        fn default_moss_height_sharpness_low() -> f32 { 0.25 }
        fn default_moss_height_sharpness_high() -> f32 { 0.7 }
        fn default_moss_bump_strength() -> f32 { 5.0 }
        fn default_moss_color_dark() -> [f32; 3] { [0.06, 0.12, 0.04] }
        fn default_moss_color_bright() -> [f32; 3] { [0.20, 0.38, 0.10] }
        fn default_show_boulders() -> bool { true }
        fn default_boulder_target_count() -> u32 { 32 }
        fn default_boulder_initial_moss() -> f32 { 10_000.0 }
        fn default_boulder_radius() -> f32 { 4.0 }
        fn default_boulder_size_gate() -> f32 { 20.0 }
        fn default_boulder_spawn_interval() -> f32 { 5.0 }
        fn default_boulder_buoyancy() -> f32 { 0.08 }
        fn default_boulder_radius_min() -> f32 { 2.0 }
        fn default_boulder_radius_max() -> f32 { 8.0 }
        fn default_boulder_moss_min() -> f32 { 2_000.0 }
        fn default_boulder_moss_max() -> f32 { 20_000.0 }
        
        let path = crate::app_dirs::config_file("cave_settings.ron");
        
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                match ron::from_str::<CaveSettings>(&contents) {
                    Ok(settings) => {
                        return (
                                settings.density,
                                settings.scale,
                                settings.octaves,
                                settings.persistence,
                                settings.threshold,
                                settings.smoothness,
                                settings.seed,
                                settings.resolution,
                                settings.show_moss,
                                settings.moss_growth_rate,
                                settings.moss_erosion_rate,
                                settings.moss_decay_rate,
                                settings.moss_min_light,
                                settings.moss_nutrient_per_moss,
                                settings.moss_graze_cooldown,
                                settings.moss_wetness_evaporation,
                                settings.moss_parallax_depth,
                                settings.moss_scale,
                                settings.moss_water_radius,
                                settings.moss_noise_type,
                                settings.moss_noise_frequency,
                                settings.moss_noise_lacunarity,
                                settings.moss_height_sharpness_low,
                                settings.moss_height_sharpness_high,
                                settings.moss_bump_strength,
                                settings.moss_color_dark,
                                settings.moss_color_bright,
                                settings.show_boulders,
                                settings.boulder_target_count,
                                settings.boulder_initial_moss,
                                settings.boulder_radius,
                                settings.boulder_size_gate,
                                settings.boulder_spawn_interval,
                                settings.boulder_buoyancy,
                                settings.boulder_radius_min,
                                settings.boulder_radius_max,
                                settings.boulder_moss_min,
                                settings.boulder_moss_max,
                            );
                        }
                        Err(e) => {
                            log::warn!("Failed to parse cave settings: {}. Using defaults.", e);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read cave settings: {}. Using defaults.", e);
                }
            }
        }
        
        // Return defaults
        (0.5, 100.0, 2u32, 0.5, 1.0, 0.0, 12345u32, 128u32,
         true, 0.15, 0.3, 0.05, 0.05, 50.0, 5.0, 0.02, 0.08, 0.15, 20.0,
         0, 18.0, 2.5, 0.25, 0.7, 5.0, [0.06, 0.12, 0.04], [0.20, 0.38, 0.10],
         true, 32u32, 10_000.0f32, 4.0f32, 20.0f32, 5.0f32, 0.08f32, 2.0f32, 8.0f32, 2_000.0f32, 20_000.0f32)
    }
    
    /// Save light settings to disk.
    pub fn save_light_settings(&self) {
        if let Err(e) = Self::save_light_settings_to_file(
            self.light_dir,
            self.show_volumetric_fog,
            self.fog_density,
            self.fog_steps,
            self.light_color,
            self.light_intensity,
            self.fog_color,
            self.fog_scattering_anisotropy,
            self.fog_absorption,
            self.fog_height_density,
            self.fog_height_falloff,
            self.light_field_max_steps,
            self.light_field_step_size,
            self.light_field_absorption_solid,
            self.light_field_absorption_cell,
            self.light_field_ambient_floor,
            // Sun settings
            self.show_sun,
            self.sun_color,
            self.sun_angular_radius,
            self.sun_intensity,
            // Shadow settings
            self.shadow_enabled,
            self.shadow_strength,
            self.shadow_quality,
            // Caustic settings
            self.caustic_intensity,
            self.caustic_scale,
            self.caustic_speed,
            // Photocyte settings
            self.photocyte_mass_per_second,
            self.photocyte_min_light_threshold,
        ) {
            log::warn!("Failed to save light settings: {}", e);
        }
    }
    
    fn save_light_settings_to_file(
        light_dir: [f32; 3],
        show_volumetric_fog: bool,
        fog_density: f32,
        fog_steps: u32,
        light_color: [f32; 3],
        light_intensity: f32,
        fog_color: [f32; 3],
        fog_scattering_anisotropy: f32,
        fog_absorption: f32,
        fog_height_density: f32,
        fog_height_falloff: f32,
        light_field_max_steps: u32,
        light_field_step_size: f32,
        light_field_absorption_solid: f32,
        light_field_absorption_cell: f32,
        light_field_ambient_floor: f32,
        // Sun settings
        show_sun: bool,
        sun_color: [f32; 3],
        sun_angular_radius: f32,
        sun_intensity: f32,
        // Shadow settings
        shadow_enabled: bool,
        shadow_strength: f32,
        shadow_quality: f32,
        // Caustic settings
        caustic_intensity: f32,
        caustic_scale: f32,
        caustic_speed: f32,
        // Photocyte settings
        photocyte_mass_per_second: f32,
        photocyte_min_light_threshold: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize)]
        struct LightSettings {
            light_dir: [f32; 3],
            show_volumetric_fog: bool,
            fog_density: f32,
            fog_steps: u32,
            light_color: [f32; 3],
            light_intensity: f32,
            fog_color: [f32; 3],
            fog_scattering_anisotropy: f32,
            fog_absorption: f32,
            fog_height_density: f32,
            fog_height_falloff: f32,
            light_field_max_steps: u32,
            light_field_step_size: f32,
            light_field_absorption_solid: f32,
            light_field_absorption_cell: f32,
            light_field_ambient_floor: f32,
            // Sun settings
            show_sun: bool,
            sun_color: [f32; 3],
            sun_angular_radius: f32,
            sun_intensity: f32,
            // Shadow settings
            shadow_enabled: bool,
            shadow_strength: f32,
            shadow_quality: f32,
            caustic_intensity: f32,
            caustic_scale: f32,
            caustic_speed: f32,
            // Photocyte settings
            photocyte_mass_per_second: f32,
            photocyte_min_light_threshold: f32,
        }
        
        let settings = LightSettings {
            light_dir,
            show_volumetric_fog,
            fog_density,
            fog_steps,
            light_color,
            light_intensity,
            fog_color,
            fog_scattering_anisotropy,
            fog_absorption,
            fog_height_density,
            fog_height_falloff,
            light_field_max_steps,
            light_field_step_size,
            light_field_absorption_solid,
            light_field_absorption_cell,
            light_field_ambient_floor,
            // Sun settings
            show_sun,
            sun_color,
            sun_angular_radius,
            sun_intensity,
            // Shadow settings
            shadow_enabled,
            shadow_strength,
            shadow_quality,
            // Caustic settings
            caustic_intensity,
            caustic_scale,
            caustic_speed,
            // Photocyte settings
            photocyte_mass_per_second,
            photocyte_min_light_threshold,
        };
        
        let path = crate::app_dirs::config_file("light_settings.ron");
        let contents = ron::ser::to_string_pretty(&settings, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Load light settings from disk, or return defaults if file doesn't exist.
    pub fn load_light_settings() -> ([f32; 3], bool, f32, u32, [f32; 3], f32, [f32; 3], f32, f32, f32, f32, u32, f32, f32, f32, f32, bool, [f32; 3], f32, f32, bool, f32, f32, f32, f32, f32, f32, f32) {
        #[derive(serde::Deserialize)]
        struct LightSettings {
            light_dir: [f32; 3],
            show_volumetric_fog: bool,
            fog_density: f32,
            fog_steps: u32,
            light_color: [f32; 3],
            light_intensity: f32,
            fog_color: [f32; 3],
            fog_scattering_anisotropy: f32,
            fog_absorption: f32,
            fog_height_density: f32,
            fog_height_falloff: f32,
            light_field_max_steps: u32,
            light_field_step_size: f32,
            light_field_absorption_solid: f32,
            light_field_absorption_cell: f32,
            light_field_ambient_floor: f32,
            // Sun settings - add #[serde(default)] for backward compatibility
            #[serde(default)]
            show_sun: bool,
            #[serde(default)]
            sun_color: [f32; 3],
            #[serde(default)]
            sun_angular_radius: f32,
            #[serde(default)]
            sun_intensity: f32,
            // Shadow settings
            #[serde(default = "default_shadow_enabled")]
            shadow_enabled: bool,
            #[serde(default = "default_shadow_strength")]
            shadow_strength: f32,
            #[serde(default = "default_shadow_quality")]
            shadow_quality: f32,
            #[serde(default = "default_caustic_intensity")]
            caustic_intensity: f32,
            #[serde(default = "default_caustic_scale")]
            caustic_scale: f32,
            #[serde(default = "default_caustic_speed")]
            caustic_speed: f32,
            #[serde(default = "default_photocyte_mass")]
            photocyte_mass_per_second: f32,
            #[serde(default = "default_photocyte_threshold")]
            photocyte_min_light_threshold: f32,
        }
        fn default_shadow_enabled() -> bool { true }
        fn default_shadow_strength() -> f32 { 0.7 }
        fn default_shadow_quality() -> f32 { 0.8 }
        fn default_caustic_intensity() -> f32 { 0.5 }
        fn default_caustic_scale() -> f32 { 8.0 }
        fn default_caustic_speed() -> f32 { 1.0 }
        fn default_photocyte_mass() -> f32 { 0.012 }
        fn default_photocyte_threshold() -> f32 { 0.05 }
        
        let path = crate::app_dirs::config_file("light_settings.ron");
        
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                    match ron::from_str::<LightSettings>(&contents) {
                        Ok(s) => {
                            return (
                                s.light_dir,
                                s.show_volumetric_fog,
                                s.fog_density,
                                s.fog_steps,
                                s.light_color,
                                s.light_intensity,
                                s.fog_color,
                                s.fog_scattering_anisotropy,
                                s.fog_absorption,
                                s.fog_height_density,
                                s.fog_height_falloff,
                                s.light_field_max_steps,
                                s.light_field_step_size,
                                s.light_field_absorption_solid,
                                s.light_field_absorption_cell,
                                s.light_field_ambient_floor,
                                // Sun settings
                                s.show_sun,
                                s.sun_color,
                                s.sun_angular_radius,
                                s.sun_intensity,
                                // Shadow settings
                                s.shadow_enabled,
                                s.shadow_strength,
                                s.shadow_quality,
                                // Caustic settings
                                s.caustic_intensity,
                                s.caustic_scale,
                                s.caustic_speed,
                                // Photocyte settings
                                s.photocyte_mass_per_second,
                                s.photocyte_min_light_threshold,
                            );
                        }
                        Err(e) => {
                            log::warn!("Failed to parse light settings: {}. Using defaults.", e);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read light settings: {}. Using defaults.", e);
                }
            }
        }
        
        // Return defaults
        (
            [0.0, 1.0, 0.0],    // light_dir
            true,               // show_volumetric_fog
            0.15,               // fog_density
            20,                 // fog_steps
            [1.0, 0.95, 0.95], // light_color
            2.6,                // light_intensity
            [0.4, 0.5, 0.6],   // fog_color
            0.5,                // fog_scattering_anisotropy
            0.13,               // fog_absorption
            1.35,               // fog_height_density
            0.002,              // fog_height_falloff
            90,                 // light_field_max_steps
            1.5,                // light_field_step_size
            20.0,               // light_field_absorption_solid
            5.0,                // light_field_absorption_cell
            0.02,               // light_field_ambient_floor
            // Sun settings
            true,               // show_sun
            [0.6, 0.4, 0.2],   // sun_color
            0.05,               // sun_angular_radius
            15.0,               // sun_intensity
            // Shadow settings
            true,               // shadow_enabled
            0.7,                // shadow_strength
            0.8,                // shadow_quality
            // Caustic settings
            0.5,                // caustic_intensity
            8.0,                // caustic_scale
            1.0,                // caustic_speed
            // Photocyte settings
            0.012,              // photocyte_mass_per_second (scaled by sun_intensity at sync)
            0.05,               // photocyte_min_light_threshold
        )
    }
    
    /// Load fluid settings from disk, or return defaults if file doesn't exist.
    pub fn load_fluid_settings() -> (f32, bool, bool, bool, f32, u32, [f32; 4], bool, u32, f32, f32, f32) {
        #[derive(serde::Deserialize)]
        struct FluidSettings {
            gravity: f32,
            gravity_x: bool,
            gravity_y: bool,
            gravity_z: bool,
            vorticity_epsilon: f32,
            pressure_iterations: u32,
            lateral_flow_probabilities: [f32; 4],
            continuous_spawn: bool,
            selected_fluid_type: u32,
            condensation_probability: f32,
            vaporization_probability: f32,
            nutrient_density: f32,
        }
        
        let path = crate::app_dirs::config_file("fluid_settings.ron");
        
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                    match ron::from_str::<FluidSettings>(&contents) {
                        Ok(settings) => {
                            return (
                                settings.gravity,
                                settings.gravity_x,
                                settings.gravity_y,
                                settings.gravity_z,
                                settings.vorticity_epsilon,
                                settings.pressure_iterations,
                                settings.lateral_flow_probabilities,
                                settings.continuous_spawn,
                                settings.selected_fluid_type,
                                settings.condensation_probability,
                                settings.vaporization_probability,
                                settings.nutrient_density,
                            );
                        }
                        Err(e) => {
                            log::warn!("Failed to parse fluid settings: {}. Using defaults.", e);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read fluid settings: {}. Using defaults.", e);
                }
            }
        }
        
        // Return defaults
        (9.8, false, true, false, 0.05, 10, [1.0, 0.8, 0.6, 0.9], false, 1, 0.1, 0.1, 0.2)
    }
    
    /// Save sun settings to disk.
    pub fn save_sun_settings(&self) {
        #[derive(serde::Serialize)]
        struct SunSettings {
            show_sun: bool,
            sun_color: [f32; 3],
            sun_angular_radius: f32,
            sun_intensity: f32,
        }
        
        let settings = SunSettings {
            show_sun: self.show_sun,
            sun_color: self.sun_color,
            sun_angular_radius: self.sun_angular_radius,
            sun_intensity: self.sun_intensity,
        };
        
        let path = crate::app_dirs::config_file("sun_settings.ron");
        match ron::ser::to_string_pretty(&settings, ron::ser::PrettyConfig::default()) {
            Ok(contents) => {
                if let Err(e) = std::fs::write(path, contents) {
                    log::warn!("Failed to save sun settings: {}", e);
                }
            }
            Err(e) => {
                log::warn!("Failed to serialize sun settings: {}", e);
            }
        }
    }
    
    /// Save fluid render settings to disk.
    pub fn save_fluid_render_settings(&self) {
        #[derive(serde::Serialize)]
        struct FluidRenderSettings {
            iso_level: f32,
            ambient: f32,
            diffuse: f32,
            specular: f32,
            shininess: f32,
            fresnel: f32,
            fresnel_power: f32,
            reflection: f32,
            alpha: f32,
            rim: f32,
            wave_height: f32,
            wave_speed: f32,
            noise_scale: f32,
            noise_octaves: f32,
            noise_lacunarity: f32,
            noise_persistence: f32,
        }
        
        let settings = FluidRenderSettings {
            iso_level: self.fluid_iso_level,
            ambient: self.fluid_ambient,
            diffuse: self.fluid_diffuse,
            specular: self.fluid_specular,
            shininess: self.fluid_shininess,
            fresnel: self.fluid_fresnel,
            fresnel_power: self.fluid_fresnel_power,
            reflection: self.fluid_reflection,
            alpha: self.fluid_alpha,
            rim: self.fluid_rim,
            wave_height: self.fluid_wave_height,
            wave_speed: self.fluid_wave_speed,
            noise_scale: self.fluid_noise_scale,
            noise_octaves: self.fluid_noise_octaves,
            noise_lacunarity: self.fluid_noise_lacunarity,
            noise_persistence: self.fluid_noise_persistence,
        };
        
        let path = crate::app_dirs::config_file("fluid_render_settings.ron");
        match ron::ser::to_string_pretty(&settings, ron::ser::PrettyConfig::default()) {
            Ok(contents) => {
                if let Err(e) = std::fs::write(path, contents) {
                    log::warn!("Failed to save fluid render settings: {}", e);
                }
            }
            Err(e) => {
                log::warn!("Failed to serialize fluid render settings: {}", e);
            }
        }
    }
    
    /// Load fluid render settings from disk, or return defaults if file doesn't exist.
    pub fn load_fluid_render_settings() -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        #[derive(serde::Deserialize)]
        #[serde(default)]
        struct FluidRenderSettings {
            iso_level: f32,
            ambient: f32,
            diffuse: f32,
            specular: f32,
            shininess: f32,
            fresnel: f32,
            fresnel_power: f32,
            reflection: f32,
            alpha: f32,
            rim: f32,
            wave_height: f32,
            wave_speed: f32,
            noise_scale: f32,
            noise_octaves: f32,
            noise_lacunarity: f32,
            noise_persistence: f32,
        }
        impl Default for FluidRenderSettings {
            fn default() -> Self {
                Self {
                    iso_level: 0.05,
                    ambient: 0.15,
                    diffuse: 0.6,
                    specular: 0.8,
                    shininess: 64.0,
                    fresnel: 0.5,
                    fresnel_power: 3.0,
                    reflection: 0.3,
                    alpha: 0.25,
                    rim: 0.5,
                    wave_height: 0.8,
                    wave_speed: 1.0,
                    noise_scale: 0.5,
                    noise_octaves: 3.0,
                    noise_lacunarity: 2.0,
                    noise_persistence: 0.5,
                }
            }
        }
        
        let path = crate::app_dirs::config_file("fluid_render_settings.ron");
        
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                    match ron::from_str::<FluidRenderSettings>(&contents) {
                        Ok(s) => {
                            return (
                                s.iso_level, s.ambient, s.diffuse, s.specular, s.shininess,
                                s.fresnel, s.fresnel_power, s.reflection, s.alpha, s.rim,
                                s.wave_height, s.wave_speed, s.noise_scale, s.noise_octaves,
                                s.noise_lacunarity, s.noise_persistence,
                            );
                        }
                        Err(e) => {
                            log::warn!("Failed to parse fluid render settings: {}. Using defaults.", e);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read fluid render settings: {}. Using defaults.", e);
                }
            }
        }
        
        let d = FluidRenderSettings::default();
        (d.iso_level, d.ambient, d.diffuse, d.specular, d.shininess,
         d.fresnel, d.fresnel_power, d.reflection, d.alpha, d.rim,
         d.wave_height, d.wave_speed, d.noise_scale, d.noise_octaves,
         d.noise_lacunarity, d.noise_persistence)
    }
    
    /// Load sun settings from disk. Returns defaults if file doesn't exist.
    pub fn load_sun_settings(&mut self) {
        #[derive(serde::Deserialize)]
        #[serde(default)]
        struct SunSettings {
            show_sun: bool,
            sun_color: [f32; 3],
            sun_angular_radius: f32,
            sun_intensity: f32,
        }
        impl Default for SunSettings {
            fn default() -> Self {
                Self {
                    show_sun: true,
                    sun_color: [1.0, 1.0, 0.85],
                    sun_angular_radius: 0.025,
                    sun_intensity: 10.0,
                }
            }
        }
        
        let path = crate::app_dirs::config_file("sun_settings.ron");
        if !path.exists() {
            return; // Use defaults already set in new()
        }
        
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                match ron::from_str::<SunSettings>(&contents) {
                    Ok(s) => {
                        self.show_sun = s.show_sun;
                        self.sun_color = s.sun_color;
                        self.sun_angular_radius = s.sun_angular_radius;
                        self.sun_intensity = s.sun_intensity;
                    }
                    Err(e) => {
                        log::warn!("Failed to parse sun settings: {}. Using defaults.", e);
                    }
                }
            }
            Err(e) => {
                log::warn!("Failed to read sun settings: {}. Using defaults.", e);
            }
        }
    }
}

/// Context passed to panel renderers.
///
/// This struct provides panels with access to all the data they need
/// to render their UI and interact with the simulation. It uses references
/// to avoid copying large data structures.
pub struct PanelContext<'a> {
    /// Current genome being edited (mutable for genome editor panels)
    pub genome: &'a mut Genome,
    /// Genome editor UI state
    pub editor_state: &'a mut GenomeEditorState,
    /// Scene manager for accessing simulation state
    pub scene_manager: &'a mut SceneManager,
    /// Camera controller (mutable for camera settings panel)
    pub camera: &'a mut CameraController,
    /// Request for scene mode changes
    pub scene_request: &'a mut SceneModeRequest,
    /// Current simulation mode
    pub current_mode: SimulationMode,
    /// Performance metrics
    pub performance: &'a PerformanceMetrics,
    /// When true, panel content should not render (hide UI mode).
    pub hide_ui: bool,
}

impl<'a> PanelContext<'a> {
    /// Create a new panel context.
    pub fn new(
        genome: &'a mut Genome,
        editor_state: &'a mut GenomeEditorState,
        scene_manager: &'a mut SceneManager,
        camera: &'a mut CameraController,
        scene_request: &'a mut SceneModeRequest,
        current_mode: SimulationMode,
        performance: &'a PerformanceMetrics,
        hide_ui: bool,
    ) -> Self {
        Self {
            genome,
            editor_state,
            scene_manager,
            camera,
            scene_request,
            current_mode,
            performance,
            hide_ui,
        }
    }

    /// Check if we're in preview mode.
    pub fn is_preview_mode(&self) -> bool {
        self.current_mode == SimulationMode::Preview
    }

    /// Check if we're in GPU mode.
    pub fn is_gpu_mode(&self) -> bool {
        self.current_mode == SimulationMode::Gpu
    }

    /// Request a switch to preview mode.
    pub fn request_preview_mode(&mut self) {
        *self.scene_request = SceneModeRequest::SwitchToPreview;
    }

    /// Request a switch to GPU mode.
    pub fn request_gpu_mode(&mut self) {
        *self.scene_request = SceneModeRequest::SwitchToGpu;
    }

    /// Request to toggle pause state.
    pub fn request_toggle_pause(&mut self) {
        *self.scene_request = SceneModeRequest::TogglePause;
    }

    /// Request to reset the simulation.
    pub fn request_reset(&mut self) {
        *self.scene_request = SceneModeRequest::Reset;
    }

    /// Request to toggle fast forward mode.
    pub fn request_toggle_fast_forward(&mut self) {
        *self.scene_request = SceneModeRequest::ToggleFastForward;
    }

    /// Check if fast forward mode is enabled.
    pub fn is_fast_forward(&self) -> bool {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.time_scale > 1.5
        } else {
            false
        }
    }
    
    /// Get the current simulation speed (time_scale).
    pub fn simulation_speed(&self) -> f32 {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.time_scale
        } else {
            1.0
        }
    }
    
    /// Set the simulation speed (time_scale).
    pub fn set_simulation_speed(&mut self, speed: f32) {
        *self.scene_request = SceneModeRequest::SetSpeed(speed);
    }

    /// Get the current cell count from the active scene.
    pub fn cell_count(&self) -> usize {
        self.scene_manager.active_scene().cell_count()
    }
    
    /// Get the GPU cell count buffer (GPU-only, no CPU readback).
    /// Returns None if not in GPU mode.
    /// Note: This returns the canonical state cell count as the GPU scene
    /// now uses GPU cell count buffer exclusively without async readback.
    pub fn gpu_cell_count(&self) -> Option<u32> {
        self.scene_manager.gpu_scene().map(|s| s.current_cell_count)
    }
    
    /// Get the GPU scene capacity.
    /// Returns None if not in GPU mode.
    pub fn gpu_capacity(&self) -> Option<u32> {
        self.scene_manager.gpu_scene().map(|s| s.capacity())
    }

    /// Get the current simulation time from the active scene.
    pub fn current_time(&self) -> f32 {
        self.scene_manager.active_scene().current_time()
    }

    /// Check if the simulation is paused.
    pub fn is_paused(&self) -> bool {
        self.scene_manager.active_scene().is_paused()
    }
    
    /// Check if GPU cell extraction is currently in progress
    pub fn is_gpu_cell_extraction_in_progress(&self) -> bool {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.is_extracting_cell_data()
        } else {
            false
        }
    }
    
    /// Get the latest GPU cell extraction result
    pub fn get_latest_gpu_cell_extraction(&self) -> Option<&crate::simulation::gpu_physics::ReadbackResult> {
        if let Some(gpu_scene) = self.scene_manager.gpu_scene() {
            gpu_scene.get_latest_cell_extraction()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_mode_request_default() {
        let request = SceneModeRequest::default();
        assert_eq!(request, SceneModeRequest::None);
        assert!(!request.is_requested());
        assert!(request.target_mode().is_none());
    }

    #[test]
    fn test_scene_mode_request_preview() {
        let request = SceneModeRequest::SwitchToPreview;
        assert!(request.is_requested());
        assert_eq!(request.target_mode(), Some(SimulationMode::Preview));
    }

    #[test]
    fn test_scene_mode_request_gpu() {
        let request = SceneModeRequest::SwitchToGpu;
        assert!(request.is_requested());
        assert_eq!(request.target_mode(), Some(SimulationMode::Gpu));
    }

    #[test]
    fn test_scene_mode_request_clear() {
        let mut request = SceneModeRequest::SwitchToGpu;
        request.clear();
        assert_eq!(request, SceneModeRequest::None);
    }

    #[test]
    fn test_genome_editor_state_default() {
        let state = GenomeEditorState::new();
        assert!(state.renaming_mode.is_none());
        assert!(state.rename_buffer.is_empty());
        assert!(!state.copy_into_dialog_open);
        assert!(state.qball_snapping);
        assert!(state.enable_snapping);
        assert_eq!(state.qball1_locked_axis, -1);
        assert_eq!(state.time_value, 0.0);
    }
}
