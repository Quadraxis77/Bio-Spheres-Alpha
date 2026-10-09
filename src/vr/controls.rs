//! Controller interaction in tracking space. Desktop mouse gestures never navigate VR.
use super::dial::{Dial, Target};
use super::input::{position, rotation, VrInput};
use super::locomotion::Gravity;
use crate::ui::{camera::CameraController, radial_menu::RadialTool};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use openxr as xr;

pub const WHEEL_PIXELS: u32 = 720;
pub const WHEEL_METERS: f32 = 0.46;

pub fn clip_planes(units_per_meter: f32) -> (f32, f32) {
    // Keep a 5 mm physical near plane and enough range when the world is
    // shrunk. Fixed world-space clipping made two-hand scaling lose the scene.
    (
        (0.005 * units_per_meter).max(0.005),
        (250.0 * units_per_meter).max(5000.0),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Tool(RadialTool),
    CellDetails,
    CellOverview,
    CellBiology,
    CellPhysics,
    CellSignals,
    LoadInspectedGenome,
    Scenes,
    World,
    WaterSettings,
    PhysicsSettings,
    LightingSettings,
    SimulationControls,
    Water,
    Floor,
    Gravity,
    StaticWater,
    SunPosition,
    Brightness,
    Nutrients,
    Pause,
    SimSpeed,
    ResetScene,
    ResetCellsOnly,
    ResetEverything,
    Orbit,
    FullUi,
    Simulation,
    GenomeEditor,
    MainMenu,
    Tools,
    Navigation,
    Help,
    Back,
    Close,
    Fly,
    Sensitivity,
    ScreenSettings,
    ScreenCurvature,
    ScreenDistance,
    ScreenAspect,
    MoveSpeed,
    TurnThreshold,
    Recenter,
    Slow,
    Normal,
    Fast,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelContext {
    MainMenu,
    Preview,
    Gpu,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    CellDetails,
    Home,
    Scenes,
    World,
    Water,
    Physics,
    Lighting,
    Simulation,
    Tools,
    Navigation,
    Help,
    Tune,
    Screen,
    Reset,
}
const HOME: [Choice; 9] = [
    Choice::Pause,
    Choice::SimSpeed,
    Choice::ResetScene,
    Choice::Scenes,
    Choice::World,
    Choice::Navigation,
    Choice::FullUi,
    Choice::MainMenu,
    Choice::CellDetails,
];
const NAVIGATION: [Choice; 6] = [
    Choice::Fly,
    Choice::Orbit,
    Choice::Sensitivity,
    Choice::ScreenSettings,
    Choice::Recenter,
    Choice::Help,
];
const TOOLS: [RadialTool; 6] = [
    RadialTool::None,
    RadialTool::Insert,
    RadialTool::Inspect,
    RadialTool::Drag,
    RadialTool::Boost,
    RadialTool::Remove,
];
impl Choice {
    pub fn label(self) -> &'static str {
        match self {
            Self::CellDetails => "Cell details",
            Self::CellOverview => "Identity",
            Self::CellBiology => "Biology",
            Self::CellPhysics => "Physics",
            Self::CellSignals => "Signals",
            Self::LoadInspectedGenome => "Load into Preview",
            Self::Scenes => "Scenes",
            Self::World => "World",
            Self::WaterSettings => "Water",
            Self::PhysicsSettings => "Physics",
            Self::LightingSettings => "Lighting",
            Self::SimulationControls => "Biology",
            Self::Tool(RadialTool::None) => "Navigate",
            Self::Tool(tool) => tool.display_name(),
            Self::Water => "Add water",
            Self::Floor => "Floor",
            Self::Gravity => "Gravity",
            Self::StaticWater => "Static water",
            Self::SunPosition => "Sun position",
            Self::Brightness => "Brightness",
            Self::Nutrients => "Nutrients",
            Self::Pause => "Pause / Run",
            Self::SimSpeed => "Speed",
            Self::ResetScene => "Reset scene",
            Self::ResetCellsOnly => "Cells only",
            Self::ResetEverything => "Cells + water",
            Self::Orbit => "Grab scene",
            Self::FullUi => "All settings",
            Self::Simulation => "Simulation",
            Self::GenomeEditor => "Genome editor",
            Self::MainMenu => "Main menu",
            Self::Tools => "Tools",
            Self::Navigation => "View / screen",
            Self::Help => "Controls",
            Self::Back => "Back",
            Self::Close => "Close",
            Self::Fly => "Ground travel",
            Self::Sensitivity => "Tune speed",
            Self::ScreenSettings => "VR screen",
            Self::ScreenCurvature => "Curvature",
            Self::ScreenDistance => "Distance",
            Self::ScreenAspect => "Aspect ratio",
            Self::MoveSpeed => "Travel speed",
            Self::TurnThreshold => "Turn sensitivity",
            Self::Recenter => "Reset view",
            Self::Slow => "Slow",
            Self::Normal => "Normal",
            Self::Fast => "Fast",
        }
    }

    fn category(self) -> (&'static str, egui::Color32) {
        self.category_with_palette(crate::ui::ui_system::palette())
    }
    pub(super) fn category_with_palette(
        self,
        p: crate::ui::ui_system::ActivePalette,
    ) -> (&'static str, egui::Color32) {
        match self {
            Self::Scenes | Self::Simulation | Self::GenomeEditor | Self::MainMenu => {
                ("SCENES", p.accent_secondary)
            }
            Self::Tool(RadialTool::Remove) => ("TOOLS", p.status_err),
            Self::ResetScene | Self::ResetCellsOnly | Self::ResetEverything => {
                ("RESET", p.status_err)
            }
            Self::CellDetails
            | Self::CellOverview
            | Self::CellBiology
            | Self::CellPhysics
            | Self::CellSignals
            | Self::LoadInspectedGenome => ("INSPECTOR", p.status_ok),
            Self::Tool(_) | Self::Tools => ("TOOLS", p.accent_primary),
            Self::Navigation
            | Self::Fly
            | Self::Orbit
            | Self::Sensitivity
            | Self::MoveSpeed
            | Self::TurnThreshold
            | Self::Recenter
            | Self::Slow
            | Self::Normal
            | Self::Fast => ("NAVIGATION", p.status_info),
            Self::World | Self::WaterSettings | Self::Water | Self::StaticWater => {
                ("WATER", p.accent_secondary)
            }
            Self::PhysicsSettings | Self::Floor | Self::Gravity => ("PHYSICS", p.status_warn),
            Self::LightingSettings | Self::SunPosition | Self::Brightness => {
                ("LIGHTING", p.status_warn)
            }
            Self::SimulationControls | Self::Nutrients | Self::Pause | Self::SimSpeed => {
                ("SIMULATION", p.status_ok)
            }
            Self::ScreenSettings
            | Self::ScreenCurvature
            | Self::ScreenDistance
            | Self::ScreenAspect => ("VR SCREEN", p.accent_secondary),
            _ => ("SYSTEM", p.text_secondary),
        }
    }
    fn status_index(self) -> Option<usize> {
        [
            Self::Water,
            Self::Floor,
            Self::Gravity,
            Self::StaticWater,
            Self::SunPosition,
            Self::Brightness,
            Self::Nutrients,
            Self::Pause,
            Self::Navigation,
            Self::FullUi,
        ]
        .iter()
        .position(|choice| *choice == self)
    }

    pub fn opens_submenu(self) -> bool {
        matches!(
            self,
            Self::CellDetails
                | Self::Scenes
                | Self::Tools
                | Self::World
                | Self::SimulationControls
                | Self::Navigation
                | Self::WaterSettings
                | Self::PhysicsSettings
                | Self::LightingSettings
                | Self::Sensitivity
                | Self::ScreenSettings
                | Self::Help
                | Self::ResetScene
        )
    }
    pub fn adjustable(self) -> bool {
        matches!(
            self,
            Self::Gravity
                | Self::SimSpeed
                | Self::SunPosition
                | Self::Brightness
                | Self::Nutrients
                | Self::MoveSpeed
                | Self::TurnThreshold
                | Self::ScreenCurvature
                | Self::ScreenDistance
                | Self::ScreenAspect
        )
    }
}

pub struct Controls {
    pub context: WheelContext,
    page: Page,
    pub menu_pose: Option<xr::Posef>,
    gravity: Gravity,
    speed: f32,
    turn_sensitivity: f32,
    pub sensitivity_adjustment: bool,
    initial_rig: Option<(Vec3, Quat, f32)>,
    wheel_introduced: bool,
    pub full_ui: bool,
    pub wheel_open: bool,
    pub number_pad_active: bool,
    pub orbit: bool,
    pub wheel_pose: Option<xr::Posef>,
    pub pointer: Option<Vec2>,
    pub wheel_pointer_pressed: bool,
    pub wheel_pointer_released: bool,
    pub hovered: Option<Choice>,
    pub adjustment: Option<Choice>,
    pub value_label: String,
    pub dial: Option<Dial>,
    pub population: super::population::Population,
    pub avg_air_temp_c: Option<f32>,
    pub avg_water_temp_c: Option<f32>,
    pub temp_display_fahrenheit: bool,
    temp_unit_hovered: bool,
    pub cell_info: super::inspector::View,
    pub cell_death_notice: f32,
    cell_tab: Choice,
    pub panel_settings: super::PanelSettings,
    entry_radius: Option<f32>,
    pub status: [String; 10],
    pub simulation_speed: f32,
    pub trigger_consumed: bool,
    pub wheel_button_hovered: bool,
    menus: [bool; 2],
    trigger: bool,
    pressed_button: Option<(Page, Choice)>,
    press_flash: f32,
    motion: Vec2,
    turning: Option<Turn>,
    turn_armed: bool,
    pub scene_fade: f32,
    navigation_blocked: bool,
    pivot: Vec3,
    grab: Option<Grab>,
}
#[derive(Clone, Copy)]
struct Turn {
    elapsed: f32,
    angle: f32,
    applied: bool,
}
const FADE_OUT: f32 = 0.09;
const BLACK_HOLD: f32 = 0.03;
const FADE_IN: f32 = 0.12;
#[derive(Clone, Copy)]
struct Grab {
    hands: u8,
    position: Vec3,
    frame: Quat,
    separation: f32,
    rig_position: Vec3,
    rig_rotation: Quat,
    scale: f32,
    translation_units: f32,
    pivot_tracking: Vec3,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            context: WheelContext::Gpu,
            page: Page::Home,
            menu_pose: None,
            gravity: Gravity::default(),
            speed: 1.5,
            turn_sensitivity: 1.0,
            sensitivity_adjustment: false,
            initial_rig: None,
            wheel_introduced: false,
            full_ui: false,
            wheel_open: false,
            number_pad_active: false,
            orbit: false,
            wheel_pose: None,
            pointer: None,
            wheel_pointer_pressed: false,
            wheel_pointer_released: false,
            hovered: None,
            adjustment: None,
            value_label: String::new(),
            dial: None,
            population: Default::default(),
            avg_air_temp_c: None,
            avg_water_temp_c: None,
            temp_display_fahrenheit: true,
            temp_unit_hovered: false,
            cell_info: Default::default(),
            cell_death_notice: 0.0,
            cell_tab: Choice::CellOverview,
            panel_settings: Default::default(),
            entry_radius: None,
            status: Default::default(),
            simulation_speed: 1.0,
            trigger_consumed: false,
            wheel_button_hovered: false,
            menus: [false; 2],
            trigger: false,
            pressed_button: None,
            press_flash: 0.0,
            motion: Vec2::ZERO,
            turning: None,
            turn_armed: true,
            scene_fade: 0.0,
            navigation_blocked: false,
            pivot: Vec3::ZERO,
            grab: None,
        }
    }
}
impl Controls {
    /// A successful cell pick opens the inspector on the existing left-hand
    /// surface. Refresh its readings without reopening it every frame.
    pub fn sync_cell_inspection(&mut self, inspected: &mut crate::ui::inspection::Inspection) -> bool {
        self.cell_info.data = inspected.data;
        self.cell_info.dead = inspected.dead;
        self.cell_info.loadable = inspected.genome.is_some();
        self.cell_info.genome_name = inspected
            .genome
            .as_ref()
            .map(|g| g.name.clone())
            .unwrap_or_default();
        self.cell_info.modes = inspected.genome.as_ref().map_or(0, |g| g.modes.len());
        if !inspected.take_open_request() {
            return false;
        }
        self.page = Page::CellDetails;
        self.cell_tab = Choice::CellOverview;
        self.wheel_open = true;
        self.wheel_introduced = true;
        self.full_ui = false;
        self.dial = None;
        self.adjustment = None;
        self.sensitivity_adjustment = false;
        self.hovered = None;
        self.pressed_button = None;
        self.press_flash = 0.0;
        self.trigger_consumed = true;
        true
    }

    /// The wrist surface takes the ray on hover as well as during a drag.
    /// Use the same owner for background input and reticle visibility.
    pub fn wrist_owns_pointer(&self) -> bool {
        self.wheel_button_hovered
            || ((self.wheel_open || self.number_pad_active)
                && (self.pointer.is_some() || self.dial.as_ref().is_some_and(Dial::grabbing)))
    }

    pub fn usage_page(&self) -> String {
        format!("{:?}", self.page)
    }

    pub fn usage_scene(&self) -> &'static str {
        match self.context {
            WheelContext::MainMenu => "Main menu",
            WheelContext::Preview => "Genome Editor",
            WheelContext::Gpu => "GPU Simulation",
        }
    }

    pub fn set_number_pad_active(&mut self, active: bool) {
        if active {
            self.wheel_open = true;
            self.adjustment = None;
        }
        self.number_pad_active = active;
    }

    pub fn set_entry_radius(&mut self, radius: f32) {
        self.entry_radius = Some(radius);
    }
    pub fn sync_ui_slider(&mut self, slider: &egui::ControllerSliderSelection) {
        let target = super::dial::Target::Ui(slider.id);
        if self.dial.as_ref().is_some_and(|dial| dial.target == target) {
            let dial = self.dial.as_mut().unwrap();
            if dial.pending.is_none() {
                dial.normalized = slider.normalized as f32;
            }
            dial.value = slider.value.clone();
        } else {
            self.dial = Some(super::dial::Dial::new(
                target,
                slider.label.clone(),
                slider.normalized as f32,
                slider.value.clone(),
                slider.minimum.clone(),
                slider.maximum.clone(),
                egui::Color32::from_rgb(90, 171, 255),
            ));
            self.adjustment = None;
            self.sensitivity_adjustment = false;
        }
        if let Some(dial) = &mut self.dial {
            dial.stops = slider
                .stops
                .iter()
                .map(|(v, l)| (*v as f32, l.clone()))
                .collect();
        }
    }
    pub fn open_choice_slider(&mut self, choice: Choice, value: f32, min: f32, max: f32) {
        let log = matches!(
            choice,
            Choice::MoveSpeed | Choice::TurnThreshold | Choice::SimSpeed
        );
        let normalized = if log {
            (value / min).ln() / (max / min).ln()
        } else {
            (value - min) / (max - min)
        };
        let label = if choice == Choice::SimSpeed {
            "Simulation speed"
        } else {
            choice.label()
        }
        .to_owned();
        let target = super::dial::Target::Choice(choice);
        let text = match choice {
            Choice::SimSpeed => format!("{value:.2}x"),
            Choice::ScreenCurvature => format!("{value:.0}°"),
            Choice::ScreenDistance => format!("{value:.2} m"),
            Choice::ScreenAspect => format!("{value:.2}:1"),
            _ => format!("{value:.2}"),
        };
        if self.dial.as_ref().is_some_and(|dial| dial.target == target) {
            let dial = self.dial.as_mut().unwrap();
            if dial.pending.is_none() {
                dial.normalized = normalized.clamp(0.0, 1.0);
            }
            dial.value = text;
        } else {
            self.dial = Some(super::dial::Dial::new(
                target,
                label,
                normalized.clamp(0.0, 1.0),
                text,
                if choice == Choice::SimSpeed {
                    format!("{min}x")
                } else {
                    format!("{min}")
                },
                if choice == Choice::SimSpeed {
                    format!("{max}x")
                } else {
                    format!("{max}")
                },
                choice.category().1,
            ));
        }
        let values: Vec<f32> = match choice {
            Choice::Gravity => vec![-100.0, -50.0, 0.0, 50.0, 100.0],
            Choice::Brightness => vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Choice::SunPosition => vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
            Choice::Nutrients => vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            Choice::SimSpeed => vec![0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, max],
            Choice::MoveSpeed => vec![0.075, 0.15, 0.5, 1.5, 5.0, 15.0, 50.0, 150.0],
            Choice::TurnThreshold => vec![0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
            Choice::ScreenCurvature => vec![0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 110.0],
            Choice::ScreenDistance => vec![0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
            Choice::ScreenAspect => vec![1.0, 4.0 / 3.0, 16.0 / 9.0, 2.0, 21.0 / 9.0, 3.0],
            _ => egui::ControllerSlider::significant_values(min as f64, max as f64, log)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        };
        let mut stops = values
            .into_iter()
            .filter(|v| *v >= min && *v <= max)
            .map(|v| {
                let t = if log {
                    (v / min).ln() / (max / min).ln()
                } else {
                    (v - min) / (max - min)
                };
                let number = egui::ControllerSlider::stop_label(v as f64);
                let label = match choice {
                    Choice::SimSpeed => format!("{number}x"),
                    Choice::ScreenCurvature | Choice::SunPosition => format!("{number}°"),
                    Choice::ScreenDistance => format!("{number} m"),
                    Choice::ScreenAspect => {
                        if (v - 4.0 / 3.0).abs() < 0.001 {
                            "4:3".into()
                        } else if (v - 16.0 / 9.0).abs() < 0.001 {
                            "16:9".into()
                        } else if (v - 21.0 / 9.0).abs() < 0.001 {
                            "21:9".into()
                        } else {
                            format!("{number}:1")
                        }
                    }
                    _ => number,
                };
                (t, label)
            })
            .collect::<Vec<_>>();
        stops.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-5);
        self.dial.as_mut().unwrap().stops = stops;
        self.wheel_open = true;
    }
    pub fn navigation_slider_value(&self, choice: Choice) -> f32 {
        if choice == Choice::MoveSpeed {
            self.speed
        } else {
            self.turn_sensitivity
        }
    }
    pub fn adjust_navigation_slider(&mut self, choice: Choice, normalized: f32) {
        if choice == Choice::MoveSpeed {
            self.speed = 0.075 * (150.0_f32 / 0.075).powf(normalized);
        } else {
            self.turn_sensitivity = 0.1 * 40.0_f32.powf(normalized);
        }
    }

    pub fn set_gravity(&mut self, strength: f32, mode: u32) {
        self.gravity = Gravity::new(strength, mode);
    }
    pub fn set_context(&mut self, context: WheelContext) {
        if self.context != context {
            let held = (self.menus, self.trigger, self.trigger_consumed);
            self.suspend();
            self.context = context;
            self.page = Page::Home;
            (self.menus, self.trigger, self.trigger_consumed) = held;
            self.full_ui = false;
            self.initial_rig = None;
        }
    }
    fn back(&mut self) {
        self.adjustment = None;
        if let Some(dial) = self.dial.take() {
            if matches!(dial.target, Target::Ui(_)) {
                self.wheel_open = false;
            }
            return;
        }
        if self.sensitivity_adjustment {
            self.sensitivity_adjustment = false;
            self.navigation_blocked = true;
            return;
        }
        if self.full_ui {
            self.full_ui = false;
            self.wheel_open = true;
            self.page = Page::Home;
        } else if self.page != Page::Home {
            self.page = match self.page {
                Page::Water | Page::Physics | Page::Lighting => Page::World,
                Page::Tune | Page::Help | Page::Screen => Page::Navigation,
                Page::Simulation => Page::World,
                Page::Reset => Page::Home,
                _ => Page::Home,
            };
        } else {
            self.wheel_open = false;
        }
    }
    fn enabled(&self, choice: Choice) -> bool {
        if choice == Choice::LoadInspectedGenome {
            return self.cell_info.loadable;
        }
        if choice == Choice::CellDetails {
            return self.cell_info.data.is_some() || self.context == WheelContext::Gpu;
        }
        self.context == WheelContext::Gpu
            || !matches!(
                choice,
                Choice::Tools
                    | Choice::World
                    | Choice::WaterSettings
                    | Choice::PhysicsSettings
                    | Choice::LightingSettings
                    | Choice::SimulationControls
                    | Choice::Tool(_)
                    | Choice::Water
                    | Choice::Floor
                    | Choice::Gravity
                    | Choice::StaticWater
                    | Choice::SunPosition
                    | Choice::Brightness
                    | Choice::Nutrients
                    | Choice::Pause
                    | Choice::SimSpeed
                    | Choice::ResetScene
                    | Choice::ResetCellsOnly
                    | Choice::ResetEverything
                    | Choice::Orbit
                    | Choice::Fly
                    | Choice::Sensitivity
                    | Choice::MoveSpeed
                    | Choice::TurnThreshold
                    | Choice::Recenter
                    | Choice::Slow
                    | Choice::Normal
                    | Choice::Fast
            )
    }
    fn start_angle(&self, count: usize) -> f32 {
        if self.page == Page::Home {
            -std::f32::consts::FRAC_PI_2 - 1.5 * std::f32::consts::TAU / count as f32
        } else {
            -std::f32::consts::FRAC_PI_2
        }
    }
    fn heading(
        &self,
        palette: crate::ui::ui_system::ActivePalette,
    ) -> (&'static str, egui::Color32) {
        let (text, choice) = match self.page {
            Page::CellDetails => ("CELL INSPECTOR", Choice::CellDetails),
            Page::Home => ("BIO-SPHERES", Choice::Tools),
            Page::Scenes => ("SCENES", Choice::Scenes),
            Page::Tools => ("TOOLS", Choice::Tools),
            Page::World => ("WORLD", Choice::World),
            Page::Water => ("WORLD / WATER", Choice::WaterSettings),
            Page::Physics => ("WORLD / PHYSICS", Choice::PhysicsSettings),
            Page::Lighting => ("WORLD / LIGHTING", Choice::LightingSettings),
            Page::Simulation => ("WORLD / BIOLOGY", Choice::SimulationControls),
            Page::Navigation => ("VIEW / SCREEN", Choice::Navigation),
            Page::Tune => ("NAVIGATION / SPEED", Choice::Navigation),
            Page::Screen => ("VR SCREEN", Choice::ScreenSettings),
            Page::Reset => ("RESET SCENE", Choice::ResetScene),
            Page::Help => ("CONTROLS", Choice::Help),
        };
        (text, choice.category_with_palette(palette).1)
    }
    fn footer_rect() -> egui::Rect {
        egui::Rect::from_center_size(egui::pos2(360.0, 455.0), egui::vec2(124.0, 26.0))
    }
    fn close_rect() -> egui::Rect {
        egui::Rect::from_center_size(egui::pos2(360.0, 500.0), egui::vec2(124.0, 26.0))
    }
    fn temperature_unit_rect() -> egui::Rect {
        egui::Rect::from_center_size(egui::pos2(360.0, 421.0), egui::vec2(72.0, 22.0))
    }
    fn rings(&self) -> Vec<(Vec<Choice>, f32, f32)> {
        if self.page == Page::Help {
            return vec![(vec![Choice::Back, Choice::Close], 275.0, 350.0)];
        }
        let choices = match self.page {
            Page::CellDetails => vec![],
            Page::Home => HOME.to_vec(),
            Page::Scenes => vec![Choice::Simulation, Choice::GenomeEditor, Choice::MainMenu],
            Page::Tools => TOOLS.map(Choice::Tool).to_vec(),
            Page::World => vec![
                Choice::WaterSettings,
                Choice::PhysicsSettings,
                Choice::LightingSettings,
                Choice::SimulationControls,
            ],
            Page::Water => vec![Choice::Water, Choice::StaticWater],
            Page::Physics => vec![Choice::Floor, Choice::Gravity],
            Page::Lighting => vec![Choice::SunPosition, Choice::Brightness],
            Page::Simulation => vec![Choice::Nutrients],
            Page::Navigation => NAVIGATION.to_vec(),
            Page::Screen => vec![
                Choice::ScreenCurvature,
                Choice::ScreenDistance,
                Choice::ScreenAspect,
            ],
            Page::Tune => vec![
                Choice::MoveSpeed,
                Choice::TurnThreshold,
                Choice::Slow,
                Choice::Normal,
                Choice::Fast,
            ],
            Page::Reset => vec![
                Choice::ResetCellsOnly,
                Choice::ResetEverything,
                Choice::Back,
            ],
            Page::Help => unreachable!(),
        };
        if self.page == Page::Home && self.context == WheelContext::Gpu {
            // All scene tools are directly selectable without opening a submenu.
            vec![
                (choices, 100.0, 205.0),
                (TOOLS.map(Choice::Tool).to_vec(), 215.0, 275.0),
            ]
        } else {
            vec![(choices, 100.0, 275.0)]
        }
    }
    fn choice_at(&self, pixel: Vec2) -> Option<Choice> {
        let offset = pixel - Vec2::splat(360.0);
        if self.context == WheelContext::Gpu
            && self.page != Page::Help
            && self.page != Page::Reset
            && Self::temperature_unit_rect().contains(egui::pos2(pixel.x, pixel.y))
        {
            return None;
        }
        if self.dial.is_some() {
            let p = egui::pos2(pixel.x, pixel.y);
            return if Self::footer_rect().contains(p) {
                Some(Choice::Back)
            } else if Self::close_rect().contains(p) {
                Some(Choice::Close)
            } else {
                None
            };
        }
        if self.page == Page::CellDetails {
            return super::inspector::hit(pixel).filter(|c| self.enabled(*c));
        }
        if self.page != Page::Help && Self::footer_rect().contains(egui::pos2(pixel.x, pixel.y)) {
            return Some(if self.dial.is_some() {
                Choice::Back
            } else if self.page == Page::Home {
                Choice::Close
            } else {
                Choice::Back
            });
        }
        for (choices, min, max) in self.rings() {
            if (min..=max).contains(&offset.length()) {
                let angle = (offset.y.atan2(offset.x) - self.start_angle(choices.len()))
                    .rem_euclid(std::f32::consts::TAU);
                let index = (angle / std::f32::consts::TAU * choices.len() as f32) as usize;
                let choice = choices[index.min(choices.len() - 1)];
                return self.enabled(choice).then_some(choice);
            }
        }
        None
    }
    pub fn suspend(&mut self) {
        self.grab = None;
        self.motion = Vec2::ZERO;
        self.turning = None;
        self.scene_fade = 0.0;
        self.navigation_blocked = true;
        self.wheel_pose = None;
        self.menu_pose = None;
        self.pointer = None;
        self.hovered = None;
        self.temp_unit_hovered = false;
        self.wheel_pointer_pressed = false;
        self.wheel_pointer_released = false;
        self.menus = [false; 2];
        self.trigger = false;
        self.pressed_button = None;
        self.press_flash = 0.0;
        self.wheel_open = false;
        self.adjustment = None;
        self.dial = None;
        self.trigger_consumed = false;
        self.wheel_button_hovered = false;
        self.sensitivity_adjustment = false;
    }
    /// Returns a single trigger-edge selection. Holding a trigger never repeats toggles.
    pub fn update(
        &mut self,
        input: &VrInput,
        camera: &mut CameraController,
        scale: &mut f32,
        dt: f32,
    ) -> Option<Choice> {
        self.cell_death_notice = (self.cell_death_notice - dt).max(0.0);
        let pressed = input.triggers[1] && !self.trigger;
        let released = !input.triggers[1] && self.trigger;
        if !input.triggers[1] {
            self.press_flash = (self.press_flash - dt / 0.18).max(0.0);
        }
        self.menu_pose = input.grips[0]
            .or(input.aims[0])
            .zip(input.head_position)
            .map(|(hand, head)| {
                let center = position(hand) + Vec3::Y * 0.14;
                let up = if (head - center).normalize_or_zero().dot(Vec3::Y).abs() > 0.95 {
                    Vec3::Z
                } else {
                    Vec3::Y
                };
                pose(
                    center,
                    Quat::from_mat4(&Mat4::look_at_rh(head, center, up).inverse()),
                )
            });
        let projected_pointer = self.menu_pose.zip(input.aims[1]).and_then(|(panel, aim)| {
            wheel_projection(panel, position(aim), rotation(aim) * Vec3::NEG_Z)
        });
        let menu_pointer = projected_pointer
            .filter(|p| p.min_element() >= 0.0 && p.max_element() <= WHEEL_PIXELS as f32);
        self.wheel_button_hovered = !self.wheel_open
            && !self.sensitivity_adjustment
            && menu_pointer.is_some_and(|pixel| (pixel - Vec2::splat(360.0)).length() <= 60.0);
        let left_button = input.menus[0] || input.wheel_button || input.stick_clicks[0];
        let opening_click = !self.wheel_open && pressed && self.wheel_button_hovered;
        let toggle_wheel =
            !self.number_pad_active && ((left_button && !self.menus[0]) || opening_click);
        if toggle_wheel {
            let selected_ui = !self.wheel_open
                && self
                    .dial
                    .as_ref()
                    .is_some_and(|d| matches!(d.target, Target::Ui(_)));
            self.wheel_open = !self.wheel_open;
            self.adjustment = None;
            if !selected_ui {
                self.dial = None;
            }
            if self.wheel_open {
                if !selected_ui {
                    self.full_ui = false;
                    self.page = Page::Home;
                }
                self.sensitivity_adjustment = false;
            }
        }
        // Show real choices on first entry; the user should not have to discover
        // an undocumented controller button before seeing what they can do.
        let introducing_wheel = !self.wheel_introduced && self.menu_pose.is_some();
        if introducing_wheel {
            self.wheel_introduced = true;
            if !toggle_wheel && !self.full_ui && !self.number_pad_active {
                self.wheel_open = true;
                self.page = Page::Home;
            }
        }
        if !self.number_pad_active && input.menus[1] && !self.menus[1] {
            self.back();
        }
        self.menus = [left_button, input.menus[1]];
        self.wheel_pose = (self.wheel_open || self.number_pad_active)
            .then_some(self.menu_pose)
            .flatten();
        let captured = self.dial.as_ref().is_some_and(Dial::grabbing);
        let keypad_pointer_captured = self.number_pad_active && (input.triggers[1] || self.trigger);
        self.pointer = self.wheel_pose.and(if captured || keypad_pointer_captured {
            projected_pointer
        } else {
            menu_pointer
        });
        self.temp_unit_hovered = self.wheel_open
            && self.context == WheelContext::Gpu
            && self.dial.is_none()
            && !self.number_pad_active
            && self.page != Page::Help
            && self.page != Page::Reset
            && self
                .pointer
                .is_some_and(|p| Self::temperature_unit_rect().contains(egui::pos2(p.x, p.y)));
        if let Some(dial) = &mut self.dial {
            dial.update(self.pointer, input.triggers[1], pressed);
            if input.ui_pointer && matches!(dial.target, Target::Ui(_)) {
                dial.adjust_stick(Vec2::new(input.turn, 0.0), dt);
            } else if !input.ui_pointer {
                dial.adjust_stick(Vec2::new(input.turn, input.lift), dt);
            }
        }
        let captured = self.dial.as_ref().is_some_and(Dial::grabbing);
        self.hovered = if captured || self.number_pad_active {
            None
        } else {
            self.pointer.and_then(|p| self.choice_at(p))
        };
        self.wheel_pointer_pressed = self.number_pad_active && pressed && self.pointer.is_some();
        self.wheel_pointer_released = self.number_pad_active && released;
        if !input.triggers[1] {
            self.trigger_consumed = false;
        }
        if (self.wheel_open
            && input.triggers[1]
            && self
                .pointer
                .is_some_and(|p| (p - Vec2::splat(360.0)).length() <= 352.0))
            || opening_click
            || captured
            || (self.number_pad_active && input.triggers[1] && self.pointer.is_some())
        {
            self.trigger_consumed = true;
        }
        self.trigger = input.triggers[1];
        let choice = if !self.number_pad_active && pressed && !toggle_wheel && !introducing_wheel {
            self.hovered
        } else {
            None
        };
        if self.temp_unit_hovered && pressed && !toggle_wheel && !introducing_wheel {
            self.temp_display_fahrenheit = !self.temp_display_fahrenheit;
        }
        if let Some(choice) = choice {
            self.pressed_button = Some((self.page, choice));
            self.press_flash = 1.0;
            if choice != Choice::Back {
                self.dial = None;
            }
            self.adjustment = choice.adjustable().then_some(choice);
            match choice {
                Choice::CellDetails => self.page = Page::CellDetails,
                Choice::CellOverview
                | Choice::CellBiology
                | Choice::CellPhysics
                | Choice::CellSignals => self.cell_tab = choice,
                Choice::LoadInspectedGenome => self.wheel_open = false,
                Choice::Scenes => self.page = Page::Scenes,
                Choice::World => self.page = Page::World,
                Choice::WaterSettings => self.page = Page::Water,
                Choice::PhysicsSettings => self.page = Page::Physics,
                Choice::LightingSettings => self.page = Page::Lighting,
                Choice::SimulationControls => self.page = Page::Simulation,
                Choice::Tools => self.page = Page::Tools,
                Choice::Navigation => self.page = Page::Navigation,
                Choice::Help => self.page = Page::Help,
                Choice::ResetScene => self.page = Page::Reset,
                Choice::ResetCellsOnly | Choice::ResetEverything => {
                    self.page = Page::Home;
                    self.wheel_open = false;
                }
                Choice::Back => self.back(),
                Choice::Close => self.wheel_open = false,
                Choice::Fly => {
                    self.orbit = false;
                    self.wheel_open = false;
                }
                Choice::Orbit => {
                    self.orbit = true;
                    self.wheel_open = false;
                }
                Choice::Sensitivity => {
                    self.page = Page::Tune;
                }
                Choice::ScreenSettings => self.page = Page::Screen,
                Choice::Slow => self.speed = 0.35,
                Choice::Normal => self.speed = 1.5,
                Choice::Fast => self.speed = 5.0,
                Choice::Recenter => {
                    if let Some((p, q, s)) = self.initial_rig {
                        camera.set_vr_rig(p, q);
                        *scale = s;
                    }
                    self.wheel_open = false;
                }
                Choice::FullUi => {
                    self.full_ui = true;
                    self.wheel_open = false;
                }
                Choice::Tool(_) | Choice::Simulation | Choice::GenomeEditor | Choice::MainMenu => {
                    self.wheel_open = false
                }
                _ => {}
            }
        }
        if self.context == WheelContext::Gpu && self.initial_rig.is_none() {
            if let (Some(radius), Some(head)) = (self.entry_radius, input.head_position) {
                let world_head = Vec3::new(
                    0.0,
                    -radius / 3.0 + (radius * 0.02).max(*scale * 0.5),
                    radius * 1.15,
                );
                camera.set_vr_rig(world_head - head * *scale, Quat::IDENTITY);
            }
            if self.entry_radius.is_none() || input.head_position.is_some() {
                self.initial_rig = Some((camera.position(), camera.view_rotation(), *scale));
            }
        }
        if self.context == WheelContext::Gpu
            && !(self.orbit && input.squeeze[1] && input.grips[1].is_some())
        {
            let (position, rotation) = self.gravity.align(
                camera.position(),
                camera.view_rotation(),
                input.head_position.unwrap_or(Vec3::ZERO),
                *scale,
            );
            camera.set_vr_rig(position, rotation);
        }
        if self.context == WheelContext::Preview
            && (choice.is_none() || self.wheel_open || self.full_ui)
        {
            self.update_preview(input, camera, dt);
            return choice;
        }
        if self.context != WheelContext::Gpu
            || (choice.is_some() && !self.wheel_open && !self.full_ui)
        {
            self.grab = None;
            self.motion = Vec2::ZERO;
            self.turning = None;
            self.scene_fade = 0.0;
            self.navigation_blocked = true;
            return choice;
        }
        let menu_open = self.wheel_open || self.full_ui;
        if self.dial.is_some() || self.adjustment.is_some() {
            // Only a focused slider owns horizontal right-stick input. Keep
            // left-stick travel available, and require neutral after editing
            // so releasing slider focus cannot turn on an already-held stick.
            self.grab = None;
            self.turning = None;
            self.scene_fade = 0.0;
            self.turn_armed = false;
            self.move_along_gravity(input, camera, *scale, dt);
            return choice;
        }
        if !menu_open && self.orbit && input.squeeze[1] && input.grips[1].is_some() {
            self.sensitivity_adjustment = false;
        }
        if !menu_open && self.sensitivity_adjustment {
            self.trigger_consumed |= input.triggers[1];
            let stick = deadzone(input.movement);
            let dt = dt.clamp(0.0, 0.1);
            // Continuous multiplicative tuning makes both fine and very fast
            // travel reachable with the same stick. Never moves the camera.
            self.speed = (self.speed * 2.0_f32.powf(stick.y * 2.0 * dt)).clamp(0.075, 150.0);
            self.turn_sensitivity =
                (self.turn_sensitivity * 2.0_f32.powf(stick.x * dt)).clamp(0.1, 4.0);
            self.grab = None;
            return choice;
        }
        // Vertical right-stick input remains available for menu scrolling.
        let travel_input = if menu_open {
            0.0
        } else {
            deadzone(Vec2::new(0.0, input.lift)).y
        };
        let dt = dt.clamp(0.0, 0.1);
        self.speed = (self.speed * 2.0_f32.powf(travel_input * 2.0 * dt)).clamp(0.075, 150.0);
        let deflection = Vec2::new(input.turn, input.lift);
        if deflection.x.abs() < 0.25 {
            self.turn_armed = true;
        }
        if self.turning.is_some() {
            self.advance_turn(input, camera, *scale, dt);
            return choice;
        }
        if !self.navigation_blocked
            && self.turn_armed
            && (menu_open || !(self.orbit && input.squeeze[1] && input.grips[1].is_some()))
            && deflection.x.abs() > (0.65 / self.turn_sensitivity).clamp(0.35, 0.9)
        {
            self.turning = Some(Turn {
                elapsed: 0.0,
                angle: -deflection.x.signum() * std::f32::consts::FRAC_PI_4,
                applied: false,
            });
            self.turn_armed = false;
            self.motion = Vec2::ZERO;
            self.grab = None;
            return choice;
        }
        if menu_open {
            // Menus allow travel and snap turns, but controller grips must
            // not pan, rotate, or scale the world while using their controls.
            self.grab = None;
            self.move_along_gravity(input, camera, *scale, dt);
            return choice;
        }
        let hands = if self.orbit && input.squeeze[1] {
            (0..2).fold(0, |mask, i| {
                mask | if input.squeeze[i] && input.grips[i].is_some() {
                    1 << i
                } else {
                    0
                }
            })
        } else {
            0
        };
        if hands != 0 {
            self.motion = Vec2::ZERO;
            self.turning = None;
            self.scene_fade = 0.0;
            let hand_frame = if hands == 3 {
                two_hand_frame(input.grips[0].unwrap(), input.grips[1].unwrap())
            } else {
                Some((
                    position(input.grips[if hands == 1 { 0 } else { 1 }].unwrap()),
                    Quat::IDENTITY,
                    1.0,
                ))
            };
            let Some((point, frame, separation)) = hand_frame else {
                self.grab = None;
                return choice;
            };
            if self.grab.is_none_or(|grab| grab.hands != hands) {
                // Rebase whenever another hand joins/leaves: never jump at a grip transition.
                let rig_rotation = camera.view_rotation();
                let rig_position = camera.position();
                let head = rig_position
                    + rig_rotation * input.head_position.unwrap_or(Vec3::ZERO) * *scale;
                self.grab = Some(Grab {
                    hands,
                    position: point,
                    frame,
                    separation,
                    rig_position,
                    rig_rotation,
                    scale: *scale,
                    translation_units: (head - self.pivot).length().clamp(*scale, *scale * 50.0),
                    pivot_tracking: rig_rotation.inverse() * (self.pivot - rig_position) / *scale,
                });
                return choice;
            }
            let grab = self.grab.unwrap();
            if hands == 3 {
                *scale = (grab.scale * grab.separation / separation).clamp(1.0, 2000.0);
                let rig_rotation = (grab.rig_rotation * grab.frame * frame.inverse()).normalize();
                let target = grab.pivot_tracking
                    + (point - grab.position) * (grab.translation_units / grab.scale);
                camera.set_vr_rig(self.pivot - rig_rotation * target * *scale, rig_rotation);
            } else {
                // A relaxed one-hand grab only translates. Wrist rotation must
                // not accidentally tip the world while reaching for a control.
                camera.set_vr_rig(
                    grab.rig_position
                        - grab.rig_rotation * (point - grab.position) * grab.translation_units,
                    grab.rig_rotation,
                );
            }
            return choice;
        }
        self.grab = None;

        self.move_along_gravity(input, camera, *scale, dt);
        choice
    }

    fn update_preview(&mut self, input: &VrInput, camera: &mut CameraController, dt: f32) {
        self.grab = None;
        self.turning = None;
        self.scene_fade = 0.0;
        let movement = deadzone(input.movement);
        let desired_turn = deadzone(Vec2::new(input.turn, input.lift));
        if self.dial.is_some() {
            self.turn_armed = false;
        } else if desired_turn == Vec2::ZERO {
            self.turn_armed = true;
        }
        let turn = if self.dial.is_some() || !self.turn_armed {
            Vec2::ZERO
        } else {
            desired_turn
        };
        if self.navigation_blocked {
            if movement == Vec2::ZERO && turn == Vec2::ZERO {
                self.navigation_blocked = false;
            }
            return;
        }
        let dt = dt.clamp(0.0, 0.1);
        // Orbit around the ground-plane up axis. Horizontal stick input must
        // preserve elevation rather than yaw around a pitched camera's local Y.
        if turn != Vec2::ZERO {
            let radians = 1.25 * self.turn_sensitivity * dt;
            let up = camera.up_direction.try_normalize().unwrap_or(Vec3::Y);
            let view = camera.view_rotation();
            let forward = view * Vec3::NEG_Z;
            let heading = (forward - up * forward.dot(up))
                .try_normalize()
                .unwrap_or_else(|| up.cross(view * Vec3::X).normalize_or_zero());
            let heading = Quat::from_axis_angle(up, -turn.x * radians) * heading;
            let limit = 85.0f32.to_radians();
            let pitch =
                (forward.dot(up).clamp(-1.0, 1.0).asin() + turn.y * radians).clamp(-limit, limit);
            let forward = (heading * pitch.cos() + up * pitch.sin()).normalize();
            let right = heading.cross(up).normalize();
            let rotation = Quat::from_mat3(&Mat3::from_cols(
                right,
                right.cross(forward).normalize(),
                -forward,
            ))
            .normalize();
            camera.rotation = rotation;
            camera.target_rotation = rotation;
            camera.look_offset = Quat::IDENTITY;
        }
        let ease = 1.0 - (-dt / 0.08).exp();
        self.motion = if movement == Vec2::ZERO {
            Vec2::ZERO
        } else {
            self.motion.lerp(movement, ease)
        };
        if self.motion != Vec2::ZERO {
            let up = camera.up_direction.try_normalize().unwrap_or(Vec3::Y);
            let view_forward = camera.rotation * Vec3::NEG_Z;
            let forward = (view_forward - up * view_forward.dot(up))
                .try_normalize()
                .unwrap_or_else(|| up.cross(camera.rotation * Vec3::X).normalize_or_zero());
            let right = forward.cross(up).normalize_or_zero();
            let vertical = if input.squeeze[0] { up } else { forward };
            // Pan relative to the preview's zoom level, so a tiny cell remains
            // easy to inspect. Move the pivot without changing orbit distance.
            let speed = camera.distance.max(0.1) * 0.4 * (self.speed / 1.5);
            camera.center += (right * self.motion.x + vertical * self.motion.y) * speed * dt;
        }
    }
    fn move_along_gravity(
        &mut self,
        input: &VrInput,
        camera: &mut CameraController,
        scale: f32,
        dt: f32,
    ) {
        let desired_motion = deadzone(input.movement);
        let desired_orbit = deadzone(Vec2::new(input.turn, 0.0));
        if self.navigation_blocked {
            if desired_motion == Vec2::ZERO
                && (self.wheel_open || self.full_ui || desired_orbit == Vec2::ZERO)
            {
                self.navigation_blocked = false;
            }
            return;
        }
        // Ease into analog motion; letting go stops immediately, without drift.
        let dt = dt.clamp(0.0, 0.1);
        let ease = 1.0 - (-dt / 0.08).exp();
        self.motion = if desired_motion == Vec2::ZERO {
            Vec2::ZERO
        } else {
            self.motion.lerp(desired_motion, ease)
        };
        if self.motion != Vec2::ZERO {
            let (rig, rotation) = self.gravity.travel(
                camera.position(),
                camera.view_rotation(),
                input.head_position.unwrap_or(Vec3::ZERO),
                input.head_rotation.unwrap_or(Quat::IDENTITY),
                self.motion,
                scale * self.speed * dt,
                scale,
                input.squeeze[0],
            );
            camera.set_vr_rig(rig, rotation);
        }
    }
    fn advance_turn(
        &mut self,
        input: &VrInput,
        camera: &mut CameraController,
        scale: f32,
        dt: f32,
    ) {
        let Some(mut turn) = self.turning.take() else {
            return;
        };
        // First present a completely black world. Apply the 45-degree change
        // only on a subsequent frame, while it is still fully black.
        if self.scene_fade >= 1.0 && !turn.applied {
            let head = input.head_position.unwrap_or(Vec3::ZERO);
            let rig = camera.position();
            let rotation = camera.view_rotation();
            let head_world = rig + rotation * head * scale;
            let world_turn =
                Quat::from_axis_angle(self.gravity.up(head_world, rotation * Vec3::Y), turn.angle);
            let rotation = (world_turn * rotation).normalize();
            camera.set_vr_rig(head_world - rotation * head * scale, rotation);
            turn.applied = true;
        }
        turn.elapsed += dt.clamp(0.0, 0.1);
        let ease = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            v * v * (3.0 - 2.0 * v)
        };
        self.scene_fade = if turn.elapsed < FADE_OUT {
            ease(turn.elapsed / FADE_OUT)
        } else if !turn.applied || turn.elapsed < FADE_OUT + BLACK_HOLD {
            1.0
        } else {
            1.0 - ease((turn.elapsed - FADE_OUT - BLACK_HOLD) / FADE_IN)
        };
        if turn.elapsed < FADE_OUT + BLACK_HOLD + FADE_IN || !turn.applied {
            self.turning = Some(turn);
        }
    }
    pub fn draw(&self, ctx: &egui::Context, active_tool: RadialTool) {
        self.draw_with_palette(ctx, active_tool, crate::ui::ui_system::palette());
    }
    fn draw_with_palette(
        &self,
        ctx: &egui::Context,
        active_tool: RadialTool,
        palette: crate::ui::ui_system::ActivePalette,
    ) {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("wrist_wheel"),
        ));
        let center = egui::pos2(360.0, 360.0);
        let accent = palette.accent_primary;
        if !self.wheel_open {
            if self.cell_death_notice > 0.0 {
                painter.text(
                    center + egui::vec2(0.0, 110.0),
                    egui::Align2::CENTER_CENTER,
                    "Inspected cell died\nGenome retained",
                    egui::FontId::proportional(20.0),
                    palette.status_err,
                );
            }
            if self.sensitivity_adjustment {
                painter.circle_filled(center, 150.0, super::theme::alpha(palette.bg_panel, 235));
                painter.circle_stroke(center, 150.0, egui::Stroke::new(2.0, accent));
                painter.text(center, egui::Align2::CENTER_CENTER,
                    format!("Sensitivity\n\nMove: {:.2}x\nTurn: {:.2}x\n\nLeft stick: adjust\nX / B: done", self.speed / 1.5, self.turn_sensitivity),
                    egui::FontId::proportional(24.0), accent);
                return;
            }
            painter.circle_filled(
                center,
                60.0,
                if self.wheel_button_hovered {
                    super::theme::alpha(palette.bg_hover, 235)
                } else {
                    super::theme::alpha(palette.bg_widget, 225)
                },
            );
            painter.circle_stroke(center, 60.0, egui::Stroke::new(2.0, accent));
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "Menu\nX",
                egui::FontId::proportional(24.0),
                palette.text_primary,
            );
            return;
        }
        super::theme::chassis(&painter, center, palette);
        if let Some(dial) = &self.dial {
            dial.draw(&painter, palette);
            painter.text(
                center - egui::vec2(0.0, 75.0),
                egui::Align2::CENTER_CENTER,
                "ADJUSTING",
                egui::FontId::proportional(13.0),
                accent,
            );
            painter.text(
                center - egui::vec2(0.0, 38.0),
                egui::Align2::CENTER_CENTER,
                &dial.label,
                egui::FontId::proportional(25.0),
                palette.text_primary,
            );
            painter.text(
                center + egui::vec2(0.0, 6.0),
                egui::Align2::CENTER_CENTER,
                &dial.value,
                egui::FontId::proportional(34.0),
                accent,
            );
            for (rect, choice, label) in [
                (Self::footer_rect(), Choice::Back, "Back / B"),
                (Self::close_rect(), Choice::Close, "Close / X"),
            ] {
                super::theme::footer(
                    &painter,
                    rect,
                    label,
                    self.hovered == Some(choice),
                    self.pressed_button == Some((self.page, choice)) && self.press_flash > 0.0,
                    palette,
                );
            }
            if let Some(pointer) = self.pointer {
                painter.circle_stroke(
                    egui::pos2(pointer.x, pointer.y),
                    8.0,
                    egui::Stroke::new(3.0, palette.text_primary),
                );
            }
            return;
        }
        if self.page == Page::CellDetails {
            painter.circle_filled(center, 270.0, super::theme::alpha(palette.bg_panel, 232));
            super::inspector::draw(
                &painter,
                &self.cell_info,
                self.cell_tab,
                self.hovered,
                palette,
            );
            if let Some(pointer) = self.pointer {
                painter.circle_stroke(
                    egui::pos2(pointer.x, pointer.y),
                    8.0,
                    egui::Stroke::new(3.0, palette.text_primary),
                );
            }
            return;
        }
        if self.cell_death_notice > 0.0 {
            let r = egui::Rect::from_center_size(egui::pos2(360.0, 658.0), egui::vec2(380.0, 38.0));
            painter.rect_filled(r, 8.0, super::theme::alpha(palette.bg_panel, 240));
            painter.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                "Cell died — selected genome retained",
                egui::FontId::proportional(18.0),
                palette.status_err,
            );
        }
        if self.page != Page::Help {
            let (heading, color) = self.heading(palette);
            painter.text(
                center + egui::vec2(0.0, -62.0),
                egui::Align2::CENTER_CENTER,
                heading,
                egui::FontId::proportional(11.0),
                color,
            );
        }
        if self.page == Page::Home {
            let start = self.start_angle(HOME.len());
            let span = std::f32::consts::TAU * 3.0 / HOME.len() as f32;
            let points = (0..=60)
                .map(|i| {
                    let a = start + span * i as f32 / 60.0;
                    let radius = if self.context == WheelContext::Gpu {
                        207.0
                    } else {
                        277.0
                    };
                    center + egui::vec2(a.cos(), a.sin()) * radius
                })
                .collect();
            let color = Choice::Pause.category_with_palette(palette).1;
            painter.add(egui::Shape::line(points, egui::Stroke::new(4.0, color)));
        }
        for (choices, inner, outer) in self.rings() {
            let count = choices.len();
            for (index, choice) in choices.into_iter().enumerate() {
                let angle =
                    index as f32 * std::f32::consts::TAU / count as f32 + self.start_angle(count);
                let span = std::f32::consts::TAU / count as f32;
                let selected = self.hovered == Some(choice)
                    || self.adjustment == Some(choice)
                    || choice == Choice::Tool(active_tool)
                    || (choice == Choice::Fly && !self.orbit)
                    || (choice == Choice::Orbit && self.orbit)
                    || (choice == Choice::Slow && self.speed == 0.35)
                    || (choice == Choice::Normal && self.speed == 1.5)
                    || (choice == Choice::Fast && self.speed == 5.0);
                let enabled = self.enabled(choice);
                let (_, tint) = choice.category_with_palette(palette);
                let press = if self.pressed_button == Some((self.page, choice)) {
                    self.press_flash
                } else {
                    0.0
                };
                let group_edges = if self.page == Page::Home && inner > 200.0 {
                    (false, false) // One tools group with shallow internal grooves.
                } else if self.page == Page::Home {
                    (index == 0 || index >= 3, index >= 2)
                } else if self.page == Page::World {
                    (true, true)
                } else {
                    (false, false)
                };
                let pos = super::theme::sector(
                    &painter,
                    center,
                    inner,
                    outer,
                    angle,
                    span,
                    tint,
                    enabled,
                    selected,
                    self.hovered == Some(choice),
                    press,
                    group_edges,
                    palette,
                );
                let mut label = if choice == Choice::Pause {
                    if self.status[7] == "RUNNING" {
                        "Pause".into()
                    } else {
                        "Play".into()
                    }
                } else {
                    choice.label().replace(' ', "\n")
                };
                if self.page == Page::Screen {
                    label.push_str(&match choice {
                        Choice::ScreenCurvature => {
                            format!("\n{:.0}°", self.panel_settings.curvature)
                        }
                        Choice::ScreenDistance => {
                            format!("\n{:.2} m", self.panel_settings.distance)
                        }
                        Choice::ScreenAspect => format!("\n{:.2}:1", self.panel_settings.aspect),
                        _ => String::new(),
                    });
                }
                if choice == Choice::SimSpeed {
                    label.push_str(&format!("\n{:.2}x", self.simulation_speed));
                }
                if choice.opens_submenu() {
                    label.push_str("\n›");
                } else if self.context == WheelContext::Gpu && choice != Choice::FullUi {
                    if let Some(index) = choice.status_index() {
                        label.push('\n');
                        label.push_str(&self.status[index]);
                    }
                }
                if self.page == Page::Reset {
                    label = match choice {
                        Choice::ResetCellsOnly => "Cells only\nKeep water".into(),
                        Choice::ResetEverything => "Cells + water\nClear scene".into(),
                        _ => label,
                    };
                }
                let text_color = if enabled {
                    palette.text_primary
                } else {
                    palette.text_dim
                };
                let mut job = egui::text::LayoutJob::simple(
                    label,
                    egui::FontId::proportional(if inner > 200.0 { 15.0 } else { 17.0 }),
                    text_color,
                    f32::INFINITY,
                );
                job.halign = egui::Align::Center;
                let galley = painter.layout_job(job);
                painter.galley(
                    pos - egui::vec2(0.0, galley.size().y * 0.5),
                    galley,
                    text_color,
                );
            }
        }
        if self.page == Page::Home {
            painter.text(
                center
                    + egui::vec2(
                        0.0,
                        if self.context == WheelContext::Gpu {
                            -190.0
                        } else {
                            -252.0
                        },
                    ),
                egui::Align2::CENTER_CENTER,
                "SIMULATION",
                egui::FontId::proportional(10.5),
                Choice::Pause.category_with_palette(palette).1,
            );
        }
        if self.context == WheelContext::Gpu && self.page == Page::Home {
            painter.text(
                center + egui::vec2(0.0, 210.0),
                egui::Align2::CENTER_CENTER,
                "TOOLS",
                egui::FontId::proportional(9.0),
                palette.accent_primary,
            );
        }
        if self.context == WheelContext::Gpu && self.page != Page::Help {
            self.population.draw(&painter, self.dial.is_none(), palette);
        }
        if let Some(dial) = &self.dial {
            dial.draw(&painter, palette);
        }
        let label = if let Some(dial) = &self.dial {
            format!("{}\n{}", dial.label, dial.value)
        } else if self.page == Page::Help {
            "X / Y: menu    B: back\nRight ray + trigger: select\nRight stick click: screenshot\nRight stick up/down: scroll / travel speed\n\nLeft stick: ground travel\nHold left grip: rise / fall\nRight stick left/right: 45 deg turn\nRadial: walk around the planet\n\nHome: play / reset / sim speed\nOuter ring: scene tools\nWorld: water / physics / lighting\nNavigation: travel / turn sliders\nWheel slider: right stick changes value\nHold trigger: grab dial\nGrab scene: right grip to pan\nBoth grips: rotate / scale"
                .to_owned()
        } else if self.page == Page::Reset {
            "Choose a scope\nCells or cells + water".to_owned()
        } else if self.context == WheelContext::Gpu {
            String::new()
        } else if self.adjustment.is_some() {
            format!("{}\nSelect slider", self.value_label)
        } else {
            format!(
                "{}\n{}\nB: back",
                if self.page == Page::Home {
                    "Close"
                } else {
                    "Back"
                },
                match self.context {
                    WheelContext::MainMenu => "Main menu",
                    WheelContext::Preview => "Editor",
                    WheelContext::Gpu => "Simulation",
                }
            )
        };
        painter.text(
            if self.dial.is_some() {
                center - egui::vec2(0.0, 21.0)
            } else {
                center
            },
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(if self.page == Page::Help { 22.0 } else { 17.0 }),
            accent,
        );
        if self.context == WheelContext::Gpu && self.page != Page::Help && self.page != Page::Reset
        {
            let badge_y = if self.dial.is_some() { 29.0 } else { -8.0 };
            painter.text(
                center + egui::vec2(0.0, badge_y - 19.0),
                egui::Align2::CENTER_CENTER,
                "POPULATION",
                egui::FontId::proportional(10.0),
                palette.status_ok,
            );
            painter.text(
                center + egui::vec2(0.0, badge_y + 3.0),
                egui::Align2::CENTER_CENTER,
                self.population.count_text(),
                egui::FontId::proportional(if self.dial.is_some() { 21.0 } else { 32.0 }),
                palette.text_primary,
            );
            let format_temp = |temperature_c: Option<f32>| {
                temperature_c.map_or_else(
                    || "--.-".to_owned(),
                    |temperature_c| {
                        let (temperature, unit) = if self.temp_display_fahrenheit {
                            (temperature_c * 9.0 / 5.0 + 32.0, "°F")
                        } else {
                            (temperature_c, "°C")
                        };
                        format!("{temperature:.1}{unit}")
                    },
                )
            };
            let temperature_y = badge_y + 35.0;
            for (index, (label, temperature)) in [
                ("AIR", self.avg_air_temp_c),
                ("WATER", self.avg_water_temp_c),
            ]
            .into_iter()
            .enumerate()
            {
                painter.text(
                    center + egui::vec2(0.0, temperature_y + index as f32 * 16.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{label}  {}", format_temp(temperature)),
                    egui::FontId::proportional(11.0),
                    palette.text_secondary,
                );
            }
            super::theme::footer(
                &painter,
                Self::temperature_unit_rect(),
                "°F / °C",
                self.temp_unit_hovered,
                false,
                palette,
            );
        }
        if self.page != Page::Help {
            let footer_choice = if self.page == Page::Home && self.dial.is_none() {
                Choice::Close
            } else {
                Choice::Back
            };
            let label = if self.dial.is_some() {
                "Done / B"
            } else if self.page == Page::Home {
                "Close / B"
            } else {
                "Back / B"
            };
            super::theme::footer(
                &painter,
                Self::footer_rect(),
                label,
                self.hovered == Some(footer_choice),
                self.pressed_button == Some((self.page, footer_choice)) && self.press_flash > 0.0,
                palette,
            );
        }
        if let Some(pointer) = self.pointer {
            painter.circle_stroke(
                egui::pos2(pointer.x, pointer.y),
                8.0,
                egui::Stroke::new(3.0, palette.text_primary),
            );
        }
    }
}
fn deadzone(value: Vec2) -> Vec2 {
    let length = value.length();
    if length <= 0.2 {
        Vec2::ZERO
    } else {
        value / length * ((length - 0.2) / 0.8).min(1.0)
    }
}
fn two_hand_frame(left: xr::Posef, right: xr::Posef) -> Option<(Vec3, Quat, f32)> {
    let a = position(left);
    let b = position(right);
    let separation = (b - a).length();
    if separation < 0.08 {
        return None;
    }
    let x = (b - a) / separation;
    let preferred_up = (rotation(left) * Vec3::Y + rotation(right) * Vec3::Y).normalize_or_zero();
    let up = if preferred_up.cross(x).length_squared() < 0.01 {
        if x.dot(Vec3::Y).abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::Z
        }
    } else {
        preferred_up
    };
    let z = x.cross(up).normalize();
    let y = z.cross(x).normalize();
    Some((
        (a + b) * 0.5,
        Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize(),
        separation,
    ))
}
/// Remote placement enters the world before spawning, rather than inserting
/// at the desktop tool's fixed distance from a controller outside the sphere.
pub fn placement_distance(origin: Vec3, direction: Vec3, radius: f32, reach: f32) -> Option<f32> {
    let b = origin.dot(direction);
    let discriminant = b * b - origin.length_squared() + radius * radius;
    if discriminant < 0.0 {
        return None;
    }
    let near = -b - discriminant.sqrt();
    let far = -b + discriminant.sqrt();
    if far <= 0.0 {
        return None;
    }
    Some(if near > 0.0 {
        near + ((far - near) * 0.25).min(reach * 0.15)
    } else {
        reach.min(far * 0.5)
    })
}
pub(super) fn pose(p: Vec3, q: Quat) -> xr::Posef {
    xr::Posef {
        position: xr::Vector3f {
            x: p.x,
            y: p.y,
            z: p.z,
        },
        orientation: xr::Quaternionf {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        },
    }
}
pub fn wheel_hit(panel: xr::Posef, origin: Vec3, direction: Vec3) -> Option<Vec2> {
    wheel_projection(panel, origin, direction)
        .filter(|pixel| pixel.min_element() >= 0.0 && pixel.max_element() <= WHEEL_PIXELS as f32)
}
fn wheel_projection(panel: xr::Posef, origin: Vec3, direction: Vec3) -> Option<Vec2> {
    let inverse = rotation(panel).inverse();
    let origin = inverse * (origin - position(panel));
    let direction = inverse * direction;
    if direction.z >= -0.001 || origin.z <= 0.0 {
        return None;
    }
    let hit = origin + direction * (-origin.z / direction.z);
    let uv = Vec2::new(hit.x / WHEEL_METERS + 0.5, 0.5 - hit.y / WHEEL_METERS);
    let pixel = uv * WHEEL_PIXELS as f32;
    pixel.is_finite().then_some(pixel)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn aim_at_wheel(input: &mut VrInput, panel: xr::Posef, pixel: Vec2) {
        let local = Vec3::new(
            (pixel.x / 720.0 - 0.5) * WHEEL_METERS,
            (0.5 - pixel.y / 720.0) * WHEEL_METERS,
            0.3,
        );
        input.aims[1] = Some(pose(
            position(panel) + rotation(panel) * local,
            rotation(panel),
        ));
    }
    fn sector_pixel(index: usize, count: usize, radius: f32) -> Vec2 {
        let angle = (index as f32 + 0.5) * std::f32::consts::TAU / count as f32
            - std::f32::consts::FRAC_PI_2;
        Vec2::splat(360.0) + Vec2::new(angle.cos(), angle.sin()) * radius
    }
    fn home_pixel(choice: Choice, radius: f32) -> Vec2 {
        let index = HOME.iter().position(|c| *c == choice).unwrap();
        let angle = -std::f32::consts::FRAC_PI_2
            + (index as f32 - 1.0) * std::f32::consts::TAU / HOME.len() as f32;
        Vec2::splat(360.0) + Vec2::new(angle.cos(), angle.sin()) * radius
    }
    #[test]
    fn reset_scope_requires_a_fresh_selection_and_back_keeps_parent_group() {
        let mut controls = Controls {
            wheel_open: true,
            wheel_introduced: true,
            ..Default::default()
        };
        let mut camera = CameraController::new();
        let mut input = VrInput {
            head_position: Some(Vec3::ZERO),
            grips: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let home_pixel = |choice: Choice| {
            let i = HOME.iter().position(|c| *c == choice).unwrap();
            let a = -std::f32::consts::FRAC_PI_2
                + (i as f32 - 1.0) * std::f32::consts::TAU / HOME.len() as f32;
            Vec2::splat(360.0) + Vec2::new(a.cos(), a.sin()) * 187.0
        };
        aim_at_wheel(
            &mut input,
            controls.wheel_pose.unwrap(),
            home_pixel(Choice::ResetScene),
        );
        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            Some(Choice::ResetScene)
        );
        assert_eq!(controls.page, Page::Reset);
        aim_at_wheel(
            &mut input,
            controls.wheel_pose.unwrap(),
            sector_pixel(1, 3, 187.0),
        );
        assert!(controls
            .update(&input, &mut camera, &mut 20.0, 0.01)
            .is_none());
        input.triggers[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            Some(Choice::ResetEverything)
        );
        assert!(!controls.wheel_open && controls.trigger_consumed);

        controls.wheel_open = true;
        controls.page = Page::Lighting;
        controls.back();
        assert_eq!(controls.page, Page::World);
        controls.back();
        assert_eq!(controls.page, Page::Home);
        assert!(HOME.contains(&Choice::Pause) && HOME.contains(&Choice::SimSpeed));
        controls.open_choice_slider(Choice::SimSpeed, 1.0, 0.1, 10.0);
        assert!(
            (controls.dial.as_ref().unwrap().normalized - 0.5).abs() < 0.001,
            "1x must be centered in the logarithmic speed slider"
        );
    }
    #[test]
    fn temperature_unit_toggle_is_visible_and_clickable_in_the_wrist_menu() {
        let mut controls = Controls {
            wheel_open: true,
            wheel_introduced: true,
            ..Default::default()
        };
        assert!(controls.temp_display_fahrenheit);
        let mut camera = CameraController::new();
        let mut input = VrInput {
            head_position: Some(Vec3::ZERO),
            grips: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let panel = controls.wheel_pose.unwrap();
        let unit_pixel = Controls::temperature_unit_rect().center();
        aim_at_wheel(&mut input, panel, Vec2::new(unit_pixel.x, unit_pixel.y));

        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            None,
            "the unit button must not select a wheel item"
        );
        assert!(controls.temp_unit_hovered);
        assert!(!controls.temp_display_fahrenheit);

        input.triggers[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        input.triggers[1] = true;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(controls.temp_display_fahrenheit);
    }
    #[test]
    fn active_slider_captures_off_panel_drag_and_stick_without_moving_camera() {
        let mut controls = Controls {
            wheel_open: true,
            wheel_introduced: true,
            ..Default::default()
        };
        controls.open_choice_slider(Choice::Gravity, 0.0, -100.0, 100.0);
        let mut camera = CameraController::new();
        let mut scale = 20.0;
        let mut input = VrInput {
            head_position: Some(Vec3::new(0.0, 1.5, 0.0)),
            grips: [Some(pose(Vec3::new(-0.2, 1.0, -0.5), Quat::IDENTITY)), None],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut scale, 0.01);
        let panel = controls.wheel_pose.unwrap();
        let point = |t: f32, radius: f32| {
            let angle = super::super::dial::START + super::super::dial::SWEEP * t;
            Vec2::splat(360.0) + Vec2::new(angle.cos(), angle.sin()) * radius
        };
        aim_at_wheel(&mut input, panel, point(0.3, 322.0));
        input.triggers[1] = true;
        assert!(controls
            .update(&input, &mut camera, &mut scale, 0.01)
            .is_none());
        assert!(controls.dial.as_ref().unwrap().grabbing());
        aim_at_wheel(&mut input, panel, point(0.5, 1000.0));
        controls.update(&input, &mut camera, &mut scale, 0.01);
        assert!((controls.dial.as_ref().unwrap().normalized - 0.5).abs() < 0.001);
        assert!(
            controls.trigger_consumed,
            "off-panel grab must not click the flat UI"
        );
        input.triggers[1] = false;
        input.turn = 1.0;
        let before = (camera.position(), camera.view_rotation());
        controls.update(&input, &mut camera, &mut scale, 0.1);
        assert!(!controls.dial.as_ref().unwrap().grabbing());
        assert!(controls.dial.as_ref().unwrap().normalized > 0.5);

        assert_eq!((camera.position(), camera.view_rotation()), before);
        let value = controls.dial.as_ref().unwrap().normalized;
        input.movement = Vec2::Y;
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut scale, 0.01);
        }
        assert!(
            (camera.position() - before.0).length() > 5.0,
            "left-stick travel works while the right stick adjusts a slider"
        );
        assert_eq!(camera.view_rotation(), before.1);
        assert!(controls.dial.as_ref().unwrap().normalized > value);
        assert!(controls.turning.is_none() && controls.grab.is_none());
    }

    #[test]
    fn preview_sticks_orbit_pan_and_lift_while_the_wheel_is_open() {
        let mut controls = Controls::default();
        controls.set_context(WheelContext::Preview);
        controls.wheel_introduced = true;
        controls.wheel_open = true;
        let mut camera = CameraController::new_for_preview_scene();
        let mut input = VrInput {
            head_position: Some(Vec3::ZERO),
            grips: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let distance = camera.distance;
        let before = camera.rotation;
        input.turn = 1.0;
        input.lift = 0.5;
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!(camera.rotation.dot(before).abs() < 0.99);
        assert!((camera.rotation.length() - 1.0).abs() < 1e-5);
        assert_eq!(camera.rotation, camera.target_rotation);
        assert_eq!(camera.center, Vec3::ZERO);
        assert_eq!(camera.distance, distance);
        assert_eq!(camera.mode, crate::ui::camera::CameraMode::Orbit);
        assert!(controls.turning.is_none() && controls.scene_fade == 0.0);
        input.turn = 0.0;
        input.lift = 0.0;
        input.movement = Vec2::Y;
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!(camera.center.length() > 4.0);
        assert!(
            camera.center.y.abs() < 0.001,
            "ordinary pan stays on the ground plane"
        );
        let before = camera.center;
        input.squeeze[0] = true;
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!(camera.center.y > before.y + 5.0);
        assert!((camera.center - before).truncate().x.abs() < 0.001);
        assert!((camera.center.z - before.z).abs() < 0.001);
        assert_eq!(camera.distance, distance);
    }
    #[test]
    fn preview_slider_keeps_right_stick_priority_and_left_stick_pan() {
        let mut controls = Controls::default();
        controls.set_context(WheelContext::Preview);
        controls.wheel_introduced = true;
        let mut camera = CameraController::new_for_preview_scene();
        let mut input = VrInput::default();
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        controls.open_choice_slider(Choice::ScreenCurvature, 0.0, 0.0, 110.0);
        let rotation = camera.rotation;
        input.turn = 1.0;
        input.lift = 1.0;
        input.movement = Vec2::X;
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert_eq!(
            camera.rotation, rotation,
            "adjusting a slider cannot rotate the preview"
        );
        assert!(camera.center.x > 4.0);
        assert!(controls.dial.as_ref().unwrap().normalized > 0.0);
        assert!(controls.grab.is_none() && controls.turning.is_none());
        controls.back();
        controls.update(&input, &mut camera, &mut 20.0, 0.1);
        assert_eq!(
            camera.rotation, rotation,
            "held slider input must not leak into scene rotation"
        );
        input.turn = 0.0;
        input.lift = 0.0;
        controls.update(&input, &mut camera, &mut 20.0, 0.1);
        input.turn = 1.0;
        controls.update(&input, &mut camera, &mut 20.0, 0.1);
        assert!(
            camera.rotation.dot(rotation).abs() < 0.999,
            "centering the stick restores preview rotation"
        );
    }

    #[test]
    fn preview_yaw_preserves_elevation_and_never_rolls_against_the_ground_plane() {
        for up in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_Y] {
            let mut controls = Controls::default();
            controls.set_context(WheelContext::Preview);
            controls.wheel_introduced = true;
            let mut camera = CameraController::new_for_preview_scene();
            camera.up_direction = up;
            let heading = up
                .cross(if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X })
                .normalize();
            let forward = heading * 0.8 - up * 0.6;
            let right = heading.cross(up).normalize();
            let rotation = Quat::from_mat3(&Mat3::from_cols(right, right.cross(forward), -forward))
                .normalize();
            camera.rotation = rotation;
            camera.target_rotation = rotation;
            // A previous mouse free-look roll must not leak into VR orbit.
            camera.look_offset = Quat::from_rotation_z(0.35);
            controls.update(&VrInput::default(), &mut camera, &mut 20.0, 0.01);
            let mut input = VrInput {
                turn: 1.0,
                ..Default::default()
            };
            for _ in 0..1200 {
                controls.update(&input, &mut camera, &mut 20.0, 1.0 / 120.0);
                let view = camera.view_rotation();
                assert!(
                    ((view * Vec3::NEG_Z).dot(up) + 0.6).abs() < 0.001,
                    "horizontal orbit must retain the preview's pitch"
                );
                assert!(
                    (view * Vec3::X).dot(up).abs() < 0.001,
                    "preview must remain level with the ground plane"
                );
                assert!(view.is_finite() && (view.length() - 1.0).abs() < 1e-5);
            }
            input.lift = 1.0;
            for _ in 0..600 {
                controls.update(&input, &mut camera, &mut 20.0, 1.0 / 120.0);
                assert!((camera.view_rotation() * Vec3::X).dot(up).abs() < 0.001);
            }
            assert!(
                (camera.view_rotation() * Vec3::NEG_Z).dot(up) < 0.997,
                "pitch stays below the pole and cannot flip the horizon"
            );
            assert_eq!(camera.mode, crate::ui::camera::CameraMode::Orbit);
            assert_eq!(camera.distance, 50.0);
            assert_eq!(camera.center, Vec3::ZERO);
        }
    }
    #[test]
    fn preview_rotation_and_pan_are_independent_of_headset_refresh_rate() {
        let run = |fps: usize| {
            let mut controls = Controls::default();
            controls.set_context(WheelContext::Preview);
            controls.wheel_introduced = true;
            let mut camera = CameraController::new_for_preview_scene();
            controls.update(&VrInput::default(), &mut camera, &mut 20.0, 0.01);
            let input = VrInput {
                turn: 0.8,
                movement: Vec2::X,
                ..Default::default()
            };
            for _ in 0..fps {
                controls.update(&input, &mut camera, &mut 20.0, 1.0 / fps as f32);
            }
            (camera.center, camera.rotation)
        };
        let a = run(60);
        let b = run(120);
        assert!(a.1.dot(b.1).abs() > 0.9999);
        assert!((a.0 - b.0).length() < 0.2);
    }
    #[test]
    fn screen_controls_are_selectable_in_all_scenes_and_use_the_shared_dial() {
        for context in [
            WheelContext::MainMenu,
            WheelContext::Preview,
            WheelContext::Gpu,
        ] {
            let mut controls = Controls::default();
            controls.set_context(context);
            controls.wheel_open = true;
            controls.wheel_introduced = true;
            let mut camera = CameraController::new();
            let mut input = VrInput {
                head_position: Some(Vec3::ZERO),
                grips: [
                    Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                    None,
                ],
                ..Default::default()
            };
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            aim_at_wheel(
                &mut input,
                controls.wheel_pose.unwrap(),
                home_pixel(Choice::Navigation, 170.0),
            );
            input.triggers[1] = true;
            assert_eq!(
                controls.update(&input, &mut camera, &mut 20.0, 0.01),
                Some(Choice::Navigation)
            );
            input.triggers[1] = false;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            aim_at_wheel(
                &mut input,
                controls.wheel_pose.unwrap(),
                sector_pixel(3, NAVIGATION.len(), 170.0),
            );
            input.triggers[1] = true;
            assert_eq!(
                controls.update(&input, &mut camera, &mut 20.0, 0.01),
                Some(Choice::ScreenSettings)
            );
            assert_eq!(controls.page, Page::Screen);
            for choice in [
                Choice::ScreenCurvature,
                Choice::ScreenDistance,
                Choice::ScreenAspect,
            ] {
                assert!(controls.enabled(choice) && choice.adjustable());
            }
            input.triggers[1] = false;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            aim_at_wheel(
                &mut input,
                controls.wheel_pose.unwrap(),
                sector_pixel(0, 3, 170.0),
            );
            input.triggers[1] = true;
            assert_eq!(
                controls.update(&input, &mut camera, &mut 20.0, 0.01),
                Some(Choice::ScreenCurvature)
            );
            controls.open_choice_slider(Choice::ScreenCurvature, 0.0, 0.0, 110.0);
            assert_eq!(
                controls.dial.as_ref().unwrap().target,
                Target::Choice(Choice::ScreenCurvature)
            );
            input.triggers[1] = false;
            input.turn = 1.0;
            controls.update(&input, &mut camera, &mut 20.0, 0.1);
            assert!(controls
                .dial
                .as_ref()
                .unwrap()
                .pending
                .is_some_and(|value| value > 0.0));
            controls.back();
            assert!(controls.dial.is_none() && controls.page == Page::Screen);
            controls.back();
            assert_eq!(controls.page, Page::Navigation);
            assert!(
                HOME.contains(&Choice::MainMenu),
                "main menu has a direct home button"
            );
        }
    }
    #[test]
    fn first_vr_entry_exposes_selectable_scenes_in_every_context() {
        for context in [
            WheelContext::MainMenu,
            WheelContext::Preview,
            WheelContext::Gpu,
        ] {
            let mut controls = Controls::default();
            controls.set_context(context);
            let mut camera = CameraController::new();
            let mut input = VrInput {
                grips: [
                    Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                    None,
                ],
                head_position: Some(Vec3::ZERO),
                ..Default::default()
            };
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert!(controls.wheel_open && controls.wheel_pose.is_some());
            let radius = if context == WheelContext::Gpu {
                170.0
            } else {
                225.0
            };
            for choice in HOME.iter().copied() {
                assert_eq!(
                    controls.choice_at(home_pixel(choice, radius)),
                    controls.enabled(choice).then_some(choice)
                );
            }
            aim_at_wheel(
                &mut input,
                controls.wheel_pose.unwrap(),
                home_pixel(Choice::Scenes, radius),
            );
            input.triggers[1] = true;
            assert_eq!(
                controls.update(&input, &mut camera, &mut 20.0, 0.01),
                Some(Choice::Scenes)
            );
            assert_eq!(controls.page, Page::Scenes);
            input.triggers[1] = false;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            aim_at_wheel(
                &mut input,
                controls.wheel_pose.unwrap(),
                sector_pixel(1, 3, radius),
            );
            input.triggers[1] = true;
            assert_eq!(
                controls.update(&input, &mut camera, &mut 20.0, 0.01),
                Some(Choice::GenomeEditor)
            );
            assert!(!controls.wheel_open && controls.trigger_consumed);
            controls.set_context(WheelContext::Preview);
            assert_eq!(controls.update(&input, &mut camera, &mut 20.0, 0.01), None);
            assert!(controls.trigger_consumed && !controls.wheel_open);
        }
    }
    #[test]
    fn missing_grip_pose_still_exposes_the_wheel_on_the_tracked_left_aim() {
        let mut controls = Controls::default();
        let input = VrInput {
            aims: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            head_position: Some(Vec3::ZERO),
            ..Default::default()
        };
        controls.update(&input, &mut CameraController::new(), &mut 20.0, 0.01);
        assert!(controls.wheel_open && controls.wheel_pose.is_some());
    }
    #[test]
    fn open_wheel_does_not_consume_clicks_aimed_at_the_normal_panel() {
        let mut controls = Controls::default();
        controls.set_context(WheelContext::MainMenu);
        let mut camera = CameraController::new();
        let mut input = VrInput {
            grips: [
                Some(pose(Vec3::new(-0.3, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            head_position: Some(Vec3::ZERO),
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(controls.wheel_open);
        input.aims[1] = Some(pose(Vec3::new(0.5, 0.0, -0.3), Quat::IDENTITY));
        input.triggers[1] = true;
        assert_eq!(controls.update(&input, &mut camera, &mut 20.0, 0.01), None);
        assert!(controls.pointer.is_none() && !controls.trigger_consumed);
        assert!(!controls.wrist_owns_pointer());
    }

    #[test]
    fn wrist_hover_owns_the_pointer_before_pressing_in_menu_and_preview() {
        for context in [WheelContext::MainMenu, WheelContext::Preview] {
            let mut controls = Controls::default();
            controls.set_context(context);
            let mut camera = CameraController::new();
            let mut input = VrInput {
                head_position: Some(Vec3::ZERO),
                grips: [
                    Some(pose(Vec3::new(-0.3, -0.2, -0.5), Quat::IDENTITY)),
                    None,
                ],
                pointer: Some(Vec2::new(500.0, 300.0)), // screen behind the wrist
                ..Default::default()
            };
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            let panel = controls.wheel_pose.unwrap();
            aim_at_wheel(&mut input, panel, Vec2::splat(360.0));
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert!(!input.triggers[1]);
            assert!(
                controls.wrist_owns_pointer(),
                "hovering the wrist must hide the background reticle before any click"
            );

            // Moving the ray outside the wrist quad restores screen input.
            aim_at_wheel(&mut input, panel, Vec2::new(900.0, 360.0));
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert!(!controls.wrist_owns_pointer());

            // The collapsed menu button also stops the ray at the wrist.
            controls.wheel_open = false;
            aim_at_wheel(&mut input, panel, Vec2::splat(360.0));
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert!(controls.wheel_button_hovered && controls.wrist_owns_pointer());
        }
    }
    #[test]
    fn back_returns_to_home_then_closes_and_settings_returns_to_home() {
        let mut controls = Controls {
            page: Page::Navigation,
            wheel_open: true,
            ..Default::default()
        };
        let mut input = VrInput {
            menus: [false, true],
            ..Default::default()
        };
        let mut camera = CameraController::new();
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert_eq!(controls.page, Page::Home);
        assert!(controls.wheel_open);
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(controls.wheel_open, "Held back must not close both levels");
        input.menus[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        input.menus[1] = true;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(!controls.wheel_open);
        controls.full_ui = true;
        controls.back();
        assert!(!controls.full_ui && controls.wheel_open && controls.page == Page::Home);
    }
    #[test]
    fn flight_sensitivity_is_manual_and_independent_of_distance_and_frame_rate() {
        let run = |distance: f32, speed: f32, hz: u32| {
            let mut controls = Controls {
                speed,
                ..Default::default()
            };
            let mut camera = CameraController::new();
            camera.set_vr_rig(Vec3::new(0.0, 0.0, distance), Quat::IDENTITY);
            let input = VrInput {
                movement: Vec2::X,
                ..Default::default()
            };
            for _ in 0..hz {
                controls.update(&input, &mut camera, &mut 20.0, 1.0 / hz as f32);
            }
            camera.position().x
        };
        let far = run(2000.0, 1.5, 120);
        let near = run(20.0, 1.5, 120);
        assert!(near > 20.0 && near < 35.0);
        assert!(
            (near - far).abs() < 0.001,
            "Distance must not choose the user's speed"
        );
        assert!(run(2000.0, 0.35, 120) < far * 0.3);
        assert!(run(2000.0, 5.0, 120) > far * 3.0);
        assert!(
            (run(2000.0, 1.5, 60) - far).abs() < 10.0,
            "Speed must not depend on headset frame rate"
        );
    }
    #[test]
    fn wheel_tuning_uses_the_same_dial_without_moving_the_camera() {
        let mut controls = Controls::default();
        let mut camera = CameraController::new();
        camera.set_vr_rig(Vec3::new(0.0, 0.0, 600.0), Quat::IDENTITY);
        let mut input = VrInput {
            head_position: Some(Vec3::ZERO),
            grips: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        controls.page = Page::Navigation;
        aim_at_wheel(
            &mut input,
            controls.wheel_pose.unwrap(),
            sector_pixel(2, NAVIGATION.len(), 150.0),
        );
        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            Some(Choice::Sensitivity)
        );
        assert_eq!(controls.page, Page::Tune);
        assert!(controls.wheel_open);
        let before = (camera.position(), camera.view_rotation());
        controls.open_choice_slider(Choice::MoveSpeed, 1.5, 0.075, 150.0);
        input.triggers[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let t = 0.8;
        let a = super::super::dial::START + super::super::dial::SWEEP * t;
        aim_at_wheel(
            &mut input,
            controls.wheel_pose.unwrap(),
            Vec2::splat(360.0) + Vec2::new(a.cos(), a.sin()) * super::super::dial::RADIUS,
        );
        input.triggers[1] = true;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let normalized = controls.dial.as_mut().unwrap().pending.take().unwrap();
        assert!((normalized - t).abs() < 0.001);
        controls.adjust_navigation_slider(Choice::MoveSpeed, normalized);
        assert!(controls.speed > 20.0 && controls.speed < 50.0);
        assert_eq!((camera.position(), camera.view_rotation()), before);
        input.triggers[1] = false;
        input.menus[1] = true;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(controls.dial.is_none() && controls.wheel_open);
        assert_eq!((camera.position(), camera.view_rotation()), before);
    }
    #[test]
    fn vr_entry_is_outside_and_near_the_y_ground_layer_and_does_not_repeat() {
        let mut controls = Controls::default();
        controls.set_entry_radius(500.0);
        let head = Vec3::new(0.2, 1.25, -0.1);
        let input = VrInput {
            head_position: Some(head),
            ..Default::default()
        };
        let mut camera = CameraController::new();
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let world_head = camera.position() + camera.view_rotation() * head * 20.0;
        assert!(world_head.length() > 500.0);
        assert!((world_head.y - (-500.0 / 3.0 + 10.0)).abs() < 0.001);
        assert!((camera.view_rotation() * Vec3::Y - Vec3::Y).length() < 0.001);
        assert!((camera.view_rotation() * Vec3::NEG_Z - Vec3::NEG_Z).length() < 0.001);
        camera.set_vr_rig(camera.position() + Vec3::X * 10.0, camera.view_rotation());
        let moved = camera.position();
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert_eq!(camera.position(), moved);
    }
    #[test]
    fn shrinking_the_world_does_not_clip_it_out_of_the_headset() {
        for scale in [1.0, 20.0, 200.0, 2000.0] {
            let (near, far) = clip_planes(scale);
            let projection =
                crate::rendering::CameraProjection::from_fov(-0.8, 0.8, -0.8, 0.8, near, far)
                    .matrix(1.0, near, far);
            let depth = projection
                .project_point3(Vec3::new(0.0, 0.0, -30.0 * scale))
                .z;
            assert!(
                depth > 0.0 && depth < 1.0,
                "scene at 30 physical meters must remain visible at scale {scale}"
            );
        }
    }
    #[test]
    fn remote_insertion_and_laser_target_stay_inside_the_world() {
        let origin = Vec3::new(0.0, 0.0, 600.0);
        let distance = placement_distance(origin, Vec3::NEG_Z, 500.0, 20.0).unwrap();
        assert!(distance > 100.0);
        assert!((origin + Vec3::NEG_Z * distance).length() < 500.0);
        assert!(placement_distance(origin, Vec3::Z, 500.0, 20.0).is_none());
        assert!(placement_distance(origin, Vec3::X, 500.0, 20.0).is_none());
        assert_eq!(
            placement_distance(Vec3::ZERO, Vec3::Z, 500.0, 20.0),
            Some(20.0)
        );
    }
    #[test]
    fn holding_selection_cannot_toggle_twice_or_activate_a_world_tool() {
        let mut controls = Controls::default();
        let mut camera = CameraController::new();
        let hand = pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY);
        let mut input = VrInput {
            grips: [Some(hand), None],
            head_position: Some(Vec3::ZERO),
            menus: [true, false],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        controls.page = Page::Tools;
        let wheel = controls.wheel_pose.unwrap();
        let angle = 1.5 * std::f32::consts::TAU / 6.0 - std::f32::consts::FRAC_PI_2;
        let pixel = Vec2::splat(360.0) + Vec2::new(angle.cos(), angle.sin()) * 150.0;
        let point = position(wheel)
            + rotation(wheel)
                * Vec3::new(
                    (pixel.x / 720.0 - 0.5) * WHEEL_METERS,
                    (0.5 - pixel.y / 720.0) * WHEEL_METERS,
                    0.0,
                );
        let origin = point + rotation(wheel) * Vec3::Z * 0.3;
        input.aims[1] = Some(pose(origin, rotation(wheel)));
        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            Some(Choice::Tool(RadialTool::Insert))
        );
        assert!(!controls.wheel_open && controls.trigger_consumed);
        assert_eq!(controls.update(&input, &mut camera, &mut 20.0, 0.01), None);
        assert!(!controls.wheel_open && controls.trigger_consumed);
        input.triggers[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(!controls.trigger_consumed);
    }

    #[test]
    fn focused_sliders_allow_travel_and_lift_without_grabs_or_unintended_turns() {
        for full_ui in [false, true] {
            let mut controls = Controls {
                full_ui,
                wheel_open: !full_ui,
                wheel_introduced: true,
                orbit: true,
                ..Default::default()
            };
            controls.open_choice_slider(Choice::Gravity, 0.0, -100.0, 100.0);
            controls.wheel_open = !full_ui;
            let mut camera = CameraController::new();
            camera.set_vr_rig(Vec3::new(0.0, 10.0, 600.0), Quat::IDENTITY);
            let mut input = VrInput {
                movement: Vec2::Y,
                turn: 1.0,
                lift: 1.0,
                head_position: Some(Vec3::ZERO),
                grips: [
                    Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                    Some(pose(Vec3::new(0.2, -0.2, -0.5), Quat::IDENTITY)),
                ],
                squeeze: [false, true],
                ..Default::default()
            };
            let before = camera.position();
            for _ in 0..30 {
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
            }
            assert!(camera.position().z < before.z - 5.0);
            assert!((camera.position().y - before.y).abs() < 0.001);
            assert_eq!(camera.view_rotation(), Quat::IDENTITY);
            assert!(controls.grab.is_none() && controls.turning.is_none());
            assert_eq!(controls.scene_fade, 0.0);

            input.squeeze[0] = true;
            let before = camera.position();
            for _ in 0..30 {
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
            }
            assert!(camera.position().y > before.y + 5.0);
            assert!((camera.position().z - before.z).abs() < 0.001);
            assert!(controls.grab.is_none(), "menu grips cannot grab the world");

            // Leaving a menu while holding the slider stick must not trigger
            // a turn. Left-stick travel continues without needing to re-center.
            controls.back();
            controls.full_ui = false;
            controls.wheel_open = false;
            input.squeeze = [false, false];
            let before = camera.position();
            for _ in 0..30 {
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
            }
            assert!(camera.position().z < before.z - 5.0);
            assert!(controls.turning.is_none());
            input.turn = 0.0;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            input.turn = 1.0;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert!(
                controls.turning.is_some(),
                "neutral stick re-arms normal turning"
            );
        }
    }
    #[test]
    fn menus_allow_left_and_right_snap_turns_without_a_focused_slider() {
        for full_ui in [false, true] {
            for direction in [-1.0, 1.0] {
                let mut controls = Controls {
                    full_ui,
                    wheel_open: !full_ui,
                    wheel_introduced: true,
                    orbit: true,
                    ..Default::default()
                };
                let mut camera = CameraController::new();
                camera.set_vr_rig(Vec3::new(0.0, 10.0, 600.0), Quat::IDENTITY);
                let head = Vec3::new(0.2, 1.5, -0.1);
                let mut input = VrInput {
                    turn: direction,
                    lift: 1.0,
                    ui_pointer: true,
                    head_position: Some(head),
                    grips: [
                        Some(pose(Vec3::new(-0.2, 1.0, -0.5), Quat::IDENTITY)),
                        Some(pose(Vec3::new(0.2, 1.0, -0.5), Quat::IDENTITY)),
                    ],
                    squeeze: [true, true],
                    ..Default::default()
                };
                let seat = camera.position() + head * 20.0;
                let speed = controls.speed;
                for _ in 0..100 {
                    controls.update(&input, &mut camera, &mut 20.0, 0.01);
                }
                let expected = Quat::from_rotation_y(-direction * std::f32::consts::FRAC_PI_4);
                assert!(
                    camera.view_rotation().angle_between(expected) < 0.001,
                    "an open menu must allow exactly one turn per stick deflection"
                );
                assert!(
                    (camera.position() + camera.view_rotation() * head * 20.0 - seat).length()
                        < 0.001
                );
                assert!(controls.grab.is_none() && controls.turning.is_none());
                assert_eq!(controls.scene_fade, 0.0);
                assert_eq!(
                    controls.speed, speed,
                    "scrolling a menu must not change travel speed"
                );
                assert!(controls.wheel_open || controls.full_ui);

                controls.open_choice_slider(Choice::Gravity, 0.0, -100.0, 100.0);
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
                controls.back();
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
                assert!(
                    controls.turning.is_none(),
                    "held slider input must not leak into turning"
                );
                input.turn = 0.0;
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
                input.turn = -direction;
                for _ in 0..50 {
                    controls.update(&input, &mut camera, &mut 20.0, 0.01);
                }
                assert!(
                    camera.view_rotation().angle_between(Quat::IDENTITY) < 0.001,
                    "turning resumes in the still-open menu after centering the stick"
                );
            }
        }
    }
    #[test]
    fn focused_slider_exposes_only_back_close_and_labeled_stops() {
        let mut c = Controls {
            wheel_open: true,
            ..Default::default()
        };
        c.open_choice_slider(Choice::ScreenAspect, 16.0 / 9.0, 1.0, 3.0);
        assert!(c
            .dial
            .as_ref()
            .unwrap()
            .stops
            .iter()
            .any(|(_, l)| l == "16:9"));
        assert_eq!(c.choice_at(Vec2::new(360.0, 180.0)), None);
        let back = Controls::footer_rect().center();
        let close = Controls::close_rect().center();
        assert_eq!(c.choice_at(Vec2::new(back.x, back.y)), Some(Choice::Back));
        assert_eq!(
            c.choice_at(Vec2::new(close.x, close.y)),
            Some(Choice::Close)
        );
        c.back();
        assert!(c.wheel_open && c.dial.is_none());
    }
    #[test]
    fn flat_ui_slider_allows_horizontal_edits_but_keeps_vertical_scroll_separate() {
        let mut c = Controls {
            context: WheelContext::Preview,
            wheel_introduced: true,
            ..Default::default()
        };
        let slider = egui::ControllerSliderSelection {
            id: egui::Id::new("test"),
            label: "Setting".into(),
            value: "50".into(),
            normalized: 0.5,
            minimum: "0".into(),
            maximum: "100".into(),
            last_seen: 0,
            stops: vec![(0.5, "50".into())],
        };
        c.sync_ui_slider(&slider);
        assert!(!c.wheel_open);
        let mut camera = CameraController::new();
        let q = camera.view_rotation();
        c.update(
            &VrInput {
                turn: 1.0,
                lift: 0.0,
                ui_pointer: true,
                ..Default::default()
            },
            &mut camera,
            &mut 20.0,
            0.1,
        );
        let horizontal_value = c.dial.as_ref().unwrap().normalized;
        assert!(horizontal_value > 0.5);
        assert!(c.dial.as_ref().unwrap().pending.is_some());
        c.dial.as_mut().unwrap().pending = None;
        c.update(
            &VrInput {
                lift: 1.0,
                ui_pointer: true,
                ..Default::default()
            },
            &mut camera,
            &mut 20.0,
            0.1,
        );
        assert!(c.dial.as_ref().unwrap().pending.is_none());
        assert!((c.dial.as_ref().unwrap().normalized - horizontal_value).abs() < 0.001);
        assert_eq!(camera.view_rotation(), q);
        c.full_ui = true;
        c.update(
            &VrInput {
                menus: [true, false],
                ..Default::default()
            },
            &mut camera,
            &mut 20.0,
            0.1,
        );
        assert!(
            c.wheel_open && c.full_ui && matches!(c.dial.as_ref().unwrap().target, Target::Ui(_)),
            "X expands the selected slider without hiding its source widget"
        );
        c.back();
        assert!(!c.wheel_open && c.full_ui && c.dial.is_none());
    }

    #[test]
    fn right_stick_vertical_adjusts_shared_main_scene_travel_speed() {
        let mut controls = Controls {
            wheel_introduced: true,
            ..Default::default()
        };
        let mut camera = CameraController::new();
        controls.update(
            &VrInput {
                lift: 1.0,
                ..Default::default()
            },
            &mut camera,
            &mut 20.0,
            0.1,
        );
        let expected_speed = 1.5 * 2.0_f32.powf(0.2);
        assert!((controls.speed - expected_speed).abs() < 0.001);

        let travel_distance = |climbing| {
            let mut controls = Controls {
                speed: controls.speed,
                wheel_introduced: true,
                ..Default::default()
            };
            let mut camera = CameraController::new();
            camera.set_vr_rig(Vec3::new(0.0, 0.0, 600.0), Quat::IDENTITY);
            let before = camera.position();
            let input = VrInput {
                movement: Vec2::Y,
                head_position: Some(Vec3::ZERO),
                squeeze: [climbing, false],
                ..Default::default()
            };
            for _ in 0..10 {
                controls.update(&input, &mut camera, &mut 20.0, 0.02);
            }
            (camera.position() - before).length()
        };
        let forward = travel_distance(false);
        let vertical = travel_distance(true);
        assert!(forward > 0.1 && vertical > 0.1);
        assert!(
            (forward - vertical).abs() < 0.01,
            "the same adjusted speed applies to forward and left-grip vertical travel"
        );
    }

    #[test]
    fn successful_inspection_opens_on_left_hand_once_per_selection() {
        let mut controls = Controls::default();
        let mut inspection = crate::ui::inspection::Inspection::default();
        let data = crate::simulation::gpu_physics::InspectedCellData {
            is_valid: 1,
            cell_id: 101,
            ..Default::default()
        };
        inspection.select(Some(3));
        assert!(!controls.sync_cell_inspection(&mut inspection));
        assert!(!controls.wheel_open, "wait for a successful cell readback");
        inspection.observe(data);
        assert!(controls.sync_cell_inspection(&mut inspection));
        assert_eq!(controls.page, Page::CellDetails);
        assert_eq!(controls.cell_tab, Choice::CellOverview);
        assert!(controls.wheel_open && !controls.full_ui);
        assert_eq!(controls.cell_info.data.unwrap().cell_id, 101);

        // The trigger that selected the cell must not click the appearing panel.
        controls.trigger = true;
        let mut camera = CameraController::new();
        let mut input = VrInput {
            head_position: Some(Vec3::ZERO),
            grips: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            triggers: [false, true],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let first_pose = controls.wheel_pose.unwrap();
        input.grips[0] = Some(pose(Vec3::new(-0.3, -0.2, -0.5), Quat::IDENTITY));
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(
            (position(controls.wheel_pose.unwrap()) - position(first_pose)
                - Vec3::new(-0.1, 0.0, 0.0))
                .length()
                < 0.001
        );
        aim_at_wheel(
            &mut input,
            controls.wheel_pose.unwrap(),
            Vec2::new(428.0, 600.0),
        );
        assert_eq!(controls.update(&input, &mut camera, &mut 20.0, 0.01), None);
        assert!(controls.wheel_open);
        input.triggers[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            Some(Choice::Close)
        );
        assert!(!controls.wheel_open);
        inspection.observe(data);
        assert!(!controls.sync_cell_inspection(&mut inspection));
        assert!(!controls.wheel_open, "live readings must respect Close");
        inspection.select(Some(3));
        inspection.observe(data);
        assert!(controls.sync_cell_inspection(&mut inspection));
        assert!(
            controls.wheel_open,
            "reselecting the same cell reopens inspection"
        );
    }

    #[test]
    fn inspector_hit_regions_allow_tabs_and_preview_but_never_load_an_empty_genome() {
        let mut c = Controls {
            wheel_open: true,
            page: Page::CellDetails,
            ..Default::default()
        };
        assert_eq!(
            c.choice_at(Vec2::new(306.0, 185.0)),
            Some(Choice::CellBiology)
        );
        assert_eq!(c.choice_at(Vec2::new(360.0, 550.0)), None);
        c.cell_info.loadable = true;
        assert_eq!(
            c.choice_at(Vec2::new(360.0, 550.0)),
            Some(Choice::LoadInspectedGenome)
        );
        c.back();
        assert_eq!(c.page, Page::Home);
    }
    #[test]
    fn wheel_renders_with_transparent_corners_and_readable_buttons() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let mut controls = Controls {
            wheel_open: true,
            ..Default::default()
        };
        for i in 0..=120 {
            let count = (4000.0 + (i as f32 * 0.12).sin() * 2000.0 + i as f32 * 40.0) as u32;
            controls.population.observe(i as f64 * 0.5, count);
        }
        controls.pressed_button = Some((Page::Home, Choice::Pause));
        controls.press_flash = 1.0;
        controls.hovered = Some(Choice::Pause);
        controls.dial = Some(Dial::new(
            Target::Choice(Choice::Brightness),
            "Brightness".into(),
            0.6,
            "3.00".into(),
            "0".into(),
            "5".into(),
            Choice::Brightness.category().1,
        ));
        controls.open_choice_slider(Choice::Brightness, 3.0, 0.0, 5.0);
        controls.status = [
            "ON",
            "ON",
            "30.0",
            "OFF",
            "45 deg",
            "3.0",
            "0.35",
            "RUNNING",
            "SCENE",
            "Open panel",
        ]
        .map(str::to_owned);

        let size = WHEEL_PIXELS;
        let dark = crate::ui::ui_system::ActivePalette::default();
        // An Arctic-style light palette and a user-created pink/cyan palette
        // exercise resolved colors directly, without changing global UI state.
        let light = crate::ui::ui_system::ActivePalette {
            bg_darkest: egui::Color32::from_rgb(235, 242, 255),
            bg_panel: egui::Color32::from_rgb(230, 238, 252),
            bg_widget: egui::Color32::from_rgb(218, 228, 245),
            bg_hover: egui::Color32::from_rgb(190, 210, 240),
            bg_active: egui::Color32::from_rgb(175, 200, 235),
            bg_selected: egui::Color32::from_rgb(180, 215, 235),
            text_primary: egui::Color32::from_rgb(8, 15, 35),
            text_secondary: egui::Color32::from_rgb(45, 65, 100),
            text_dim: egui::Color32::from_rgb(75, 90, 120),
            accent_primary: egui::Color32::from_rgb(10, 90, 200),
            accent_secondary: egui::Color32::from_rgb(0, 155, 185),
            border_subtle: egui::Color32::from_rgb(185, 205, 230),
            border_normal: egui::Color32::from_rgb(140, 170, 205),
            border_bright: egui::Color32::from_rgb(10, 90, 200),
            theme: crate::ui::types::UiTheme::Arctic,
            ..dark
        };
        let custom = crate::ui::ui_system::ActivePalette {
            bg_darkest: egui::Color32::from_rgb(10, 1, 18),
            bg_panel: egui::Color32::from_rgb(22, 5, 36),
            bg_widget: egui::Color32::from_rgb(42, 12, 63),
            bg_hover: egui::Color32::from_rgb(70, 20, 92),
            bg_active: egui::Color32::from_rgb(85, 28, 105),
            bg_selected: egui::Color32::from_rgb(65, 15, 85),
            accent_primary: egui::Color32::from_rgb(255, 60, 205),
            accent_secondary: egui::Color32::from_rgb(25, 225, 245),
            border_bright: egui::Color32::from_rgb(255, 60, 205),
            text_primary: egui::Color32::from_rgb(245, 222, 255),
            theme: crate::ui::types::UiTheme::Custom,
            ..dark
        };
        for (name, palette) in [
            ("left-hand-wheel", dark),
            ("left-hand-wheel-light", light),
            ("left-hand-wheel-custom", custom),
            ("home-tools-wheel", dark),
            ("world-wheel-light", light),
            ("inspected-cell", dark),
            ("inspected-cell-dead", light),
        ] {
            if name == "home-tools-wheel" {
                controls.dial = None;
            }
            if name == "world-wheel-light" {
                controls.page = Page::World;
                controls.dial = None;
                controls.hovered = None;
            }
            if name.starts_with("inspected-cell") {
                controls.page = Page::CellDetails;
                controls.cell_info.data = Some(crate::simulation::gpu_physics::InspectedCellData {
                    is_valid: 1,
                    cell_id: 12345,
                    genome_id: 7,
                    age: 42.5,
                    mode_index: 12,
                    cell_type: 0,
                    nutrients: 98.2,
                    nutrient_gain_rate: 3.4,
                    split_count: 2,
                    max_splits: 8,
                    cell_cached_temperature: 95.0,
                    cell_thermal_state: 4,
                    ..Default::default()
                });
                controls.cell_info.genome_name = "Pelagic signal foundation".into();
                controls.cell_info.modes = 12;
                controls.cell_info.loadable = true;
                controls.cell_info.dead = name == "inspected-cell-dead";
            }
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(size as f32, size as f32),
                )),
                ..Default::default()
            });
            controls.draw_with_palette(&ctx, RadialTool::Inspect, palette);
            let output = ctx.end_pass();
            let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Wrist wheel visual regression"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let mut renderer = egui_wgpu::Renderer::new(&device, format, Default::default());
            for (id, delta) in output.textures_delta.set {
                renderer.update_texture(&device, &queue, id, &delta);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size, size],
                pixels_per_point: 1.0,
            };
            let commands = renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &screen);
            {
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
            }
            let stride = (size * 4 + 255) / 256 * 256;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (stride * size) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride),
                        rows_per_image: Some(size),
                    },
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(commands.into_iter().chain([encoder.finish()]));
            let (send, receive) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                send.send(r).unwrap();
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            receive.recv().unwrap().unwrap();
            let pixels = staging.slice(..).get_mapped_range();
            assert_eq!(pixels[3], 0, "corner must not hide immersive world");
            let rgba: Vec<u8> = pixels
                .chunks(stride as usize)
                .flat_map(|row| row[..size as usize * 4].iter().copied())
                .collect();
            assert!(
                rgba.chunks_exact(4)
                    .filter(|pixel| pixel[3] > 200
                        && if palette.text_primary.r() < 100 {
                            pixel[0] < 100 && pixel[1] < 100 && pixel[2] < 100
                        } else {
                            pixel[0] > 200 && pixel[1] > 200 && pixel[2] > 200
                        })
                    .count()
                    > 1000,
                "wheel labels must render"
            );

            // The clear hub surface must change with the resolved palette,
            // even when switching themes while the same wheel remains open.
            let hub = &rgba[(389 * size as usize + 340) * 4..][..4];
            let brightness = hub[0] as u32 + hub[1] as u32 + hub[2] as u32;
            if palette.bg_panel.r() > 200 {
                assert!(brightness > 450, "light theme must produce a light console");
            } else {
                assert!(brightness < 300, "dark theme must produce a dark console");
            }
            assert!(hub[3] < 255, "console surfaces retain transparency");
            std::fs::create_dir_all("target/vr-visual-checks").unwrap();

            image::save_buffer(
                format!("target/vr-visual-checks/{name}.png"),
                &rgba,
                size,
                size,
                image::ColorType::Rgba8,
            )
            .unwrap();
        }
    }
    #[test]
    fn wheel_requires_a_front_hit_and_visible_button() {
        let wheel = pose(Vec3::new(0.0, 0.0, -1.0), Quat::IDENTITY);
        assert_eq!(
            wheel_hit(wheel, Vec3::ZERO, Vec3::NEG_Z),
            Some(Vec2::splat(360.0))
        );
        assert!(wheel_hit(wheel, Vec3::new(0.0, 0.0, -2.0), Vec3::Z).is_none());
        let controls = Controls {
            page: Page::Tools,
            ..Default::default()
        };
        assert_eq!(controls.choice_at(Vec2::splat(360.0)), None);
        let back = Controls::footer_rect().center();
        assert_eq!(
            controls.choice_at(Vec2::new(back.x, back.y)),
            Some(Choice::Back)
        );
        assert_eq!(
            controls.choice_at(Vec2::new(435.0, 360.0 - 150.0 * 0.8660254)),
            Some(Choice::Tool(RadialTool::None))
        );
        assert_eq!(controls.choice_at(Vec2::new(360.0, 5.0)), None);
    }
    #[test]
    fn one_hand_moves_the_scene_without_wrist_rotation_and_releases_on_tracking_loss() {
        let mut controls = Controls {
            orbit: true,
            ..Default::default()
        };
        let mut camera = CameraController::new();
        let initial = pose(Vec3::new(0.2, 0.0, -0.5), Quat::IDENTITY);
        let mut input = VrInput {
            grips: [None, Some(initial)],
            squeeze: [false, true],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let rig = camera.position();
        let orientation = camera.view_rotation();
        let gain = controls.grab.unwrap().translation_units;
        input.grips[1] = Some(pose(Vec3::new(0.4, 0.1, -0.5), Quat::from_rotation_y(0.4)));
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let current = input.grips[1].unwrap();
        assert!(
            (camera.position()
                - (rig - orientation * (position(current) - position(initial)) * gain))
                .length()
                < 0.001
        );
        assert_eq!(
            camera.view_rotation(),
            orientation,
            "a one-hand wrist turn must not tip the scene"
        );
        input.grips[1] = None;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(controls.grab.is_none());
    }
    #[test]
    fn turning_in_grab_mode_still_turns_the_seated_view_around_gravity() {
        let mut controls = Controls {
            orbit: true,
            ..Default::default()
        };
        let mut camera = CameraController::new();
        camera.set_vr_rig(Vec3::new(0.0, 0.0, 600.0), Quat::IDENTITY);
        let input = VrInput {
            turn: 1.0,
            head_position: Some(Vec3::new(0.3, 0.2, -0.1)),
            ..Default::default()
        };
        let seat = camera.position() + camera.view_rotation() * input.head_position.unwrap() * 20.0;
        for _ in 0..50 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!(
            (camera.view_rotation().angle_between(Quat::IDENTITY) - std::f32::consts::FRAC_PI_4)
                .abs()
                < 0.001
        );
        assert!(
            (camera.position() + camera.view_rotation() * input.head_position.unwrap() * 20.0
                - seat)
                .length()
                < 0.001
        );
        let stopped = camera.view_rotation();
        for _ in 0..100 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert_eq!(camera.view_rotation(), stopped);
    }
    #[test]
    fn two_hands_rotate_and_scale_around_the_scene_without_transition_jumps() {
        let mut controls = Controls {
            orbit: true,
            ..Default::default()
        };
        let mut camera = CameraController::new();
        let mut scale = 20.0;
        let mid = Vec3::new(0.0, 0.0, -0.5);
        let left = pose(mid - Vec3::X * 0.2, Quat::IDENTITY);
        let right = pose(mid + Vec3::X * 0.2, Quat::IDENTITY);
        let mut input = VrInput {
            grips: [Some(left), Some(right)],
            squeeze: [true, false],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut scale, 0.01);
        let initial_rig = camera.position();
        let initial_rotation = camera.view_rotation();
        input.squeeze[1] = true;
        controls.update(&input, &mut camera, &mut scale, 0.01);
        assert_eq!(
            camera.position(),
            initial_rig,
            "second hand must join without a jump"
        );
        let pivot = camera.view_rotation().inverse() * (Vec3::ZERO - camera.position()) / scale;
        let turn = Quat::from_rotation_y(0.4);
        input.grips = [
            Some(pose(mid - turn * Vec3::X * 0.4, turn)),
            Some(pose(mid + turn * Vec3::X * 0.4, turn)),
        ];
        controls.update(&input, &mut camera, &mut scale, 0.01);
        assert!(
            (scale - 10.0).abs() < 0.001,
            "spreading the hands doubles the world's perceived size"
        );
        assert!(
            (camera.view_rotation().inverse() * (Vec3::ZERO - camera.position()) / scale - pivot)
                .length()
                < 0.001
        );
        assert!((camera.view_rotation().angle_between(initial_rotation) - 0.4).abs() < 0.001);
        let released = camera.position();
        input.squeeze[1] = false;
        controls.update(&input, &mut camera, &mut scale, 0.01);
        assert_eq!(
            camera.position(),
            released,
            "releasing one hand must not jump"
        );
        input.grips = [None, None];
        controls.update(&input, &mut camera, &mut scale, 0.01);
        assert_eq!(camera.position(), released);
        assert!(controls.grab.is_none());
    }

    #[test]
    fn turns_are_45_degrees_after_black_and_holding_the_stick_does_not_repeat() {
        for (x, y) in [(1.0, 0.0), (-1.0, 0.0)] {
            let mut controls = Controls::default();
            let mut camera = CameraController::new();
            let head = Vec3::new(0.2, 0.15, -0.1);
            camera.set_vr_rig(Vec3::new(0.0, 0.0, 600.0), Quat::IDENTITY);
            let seat = camera.position() + camera.view_rotation() * head * 20.0;
            let mut input = VrInput {
                head_position: Some(head),
                turn: x,
                lift: y,
                ..Default::default()
            };
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            for _ in 0..20 {
                if controls.scene_fade == 1.0 {
                    break;
                }
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
            }
            assert_eq!(controls.scene_fade, 1.0);
            assert_eq!(
                camera.view_rotation(),
                Quat::IDENTITY,
                "Never rotate before a black frame"
            );
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert_eq!(controls.scene_fade, 1.0);
            assert!(
                (camera.view_rotation().angle_between(Quat::IDENTITY)
                    - std::f32::consts::FRAC_PI_4)
                    .abs()
                    < 0.001
            );
            for _ in 0..200 {
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
            }
            assert_eq!(controls.scene_fade, 0.0);
            assert!(
                (camera.view_rotation().angle_between(Quat::IDENTITY)
                    - std::f32::consts::FRAC_PI_4)
                    .abs()
                    < 0.001,
                "A held stick must not keep rotating"
            );
            assert!(
                (camera.position() + camera.view_rotation() * head * 20.0 - seat).length() < 0.001
            );
            input.turn = 0.0;
            input.lift = 0.0;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            input.turn = x;
            input.lift = y;
            for _ in 0..40 {
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
            }
            assert!(
                (camera.view_rotation().angle_between(Quat::IDENTITY)
                    - std::f32::consts::FRAC_PI_2)
                    .abs()
                    < 0.001
            );
        }
    }
    #[test]
    fn wheel_front_faces_the_actual_head_and_right_ray_can_select_a_sector() {
        let mut controls = Controls::default();
        let mut camera = CameraController::new();
        let head = Vec3::new(0.0, 1.2, 0.0);
        let mut input = VrInput {
            head_position: Some(head),
            grips: [
                Some(pose(Vec3::new(-0.25, 0.9, -0.45), Quat::IDENTITY)),
                None,
            ],
            ..Default::default()
        };
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let panel = controls.wheel_pose.unwrap();
        let normal = rotation(panel) * Vec3::Z;
        assert!(
            normal.dot(head - position(panel)) > 0.0,
            "OpenXR quad front (+Z) must face the player"
        );
        assert_eq!(
            wheel_hit(panel, head, (position(panel) - head).normalize())
                .unwrap()
                .round(),
            Vec2::splat(360.0)
        );
        let angle =
            controls.start_angle(TOOLS.len()) + 2.5 * std::f32::consts::TAU / TOOLS.len() as f32;
        let pixel = Vec2::splat(360.0) + Vec2::new(angle.cos(), angle.sin()) * 245.0;
        let target = position(panel)
            + rotation(panel)
                * Vec3::new(
                    (pixel.x / 720.0 - 0.5) * WHEEL_METERS,
                    (0.5 - pixel.y / 720.0) * WHEEL_METERS,
                    0.0,
                );
        let right = head + Vec3::new(0.18, -0.15, -0.1);
        let aim = Quat::from_rotation_arc(Vec3::NEG_Z, (target - right).normalize());
        input.aims[1] = Some(pose(right, aim));
        input.triggers[1] = true;
        assert_eq!(
            controls.update(&input, &mut camera, &mut 20.0, 0.01),
            Some(Choice::Tool(RadialTool::Inspect))
        );
    }
    #[test]
    fn left_grip_is_a_held_altitude_modifier_and_never_enters_tuning() {
        let mut controls = Controls::default();
        let mut camera = CameraController::new();
        camera.set_vr_rig(
            Vec3::new(0.0, 0.0, 600.0),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        );
        let mut input = VrInput {
            movement: Vec2::Y,
            squeeze: [true, false],
            head_rotation: Some(Quat::from_rotation_x(-1.0)),
            ..Default::default()
        };
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!(!controls.sensitivity_adjustment);
        assert!(camera.position().y > 5.0 && camera.position().x.abs() < 0.001);
        assert!((camera.view_rotation() * Vec3::Y - Vec3::Y).length() < 0.001);
        let before = camera.position();
        input.squeeze[0] = false;
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!((camera.position().y - before.y).abs() < 0.001);
        assert!((camera.position() - before).length() > 5.0);
        input.squeeze = [false, true];
        let before = camera.position();
        for _ in 0..30 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert!(
            (camera.position().y - before.y).abs() < 0.001,
            "Right grip does not climb"
        );
    }
    #[test]
    fn right_stick_vertical_does_not_pitch_or_roll_the_gravity_floor() {
        let mut controls = Controls::default();
        let mut camera = CameraController::new();
        camera.set_vr_rig(Vec3::new(0.0, 0.0, 600.0), Quat::IDENTITY);
        let input = VrInput {
            lift: 1.0,
            ..Default::default()
        };
        for _ in 0..100 {
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
        }
        assert_eq!(camera.view_rotation(), Quat::IDENTITY);
        assert!(controls.turning.is_none() && controls.scene_fade == 0.0);
    }
    #[test]
    fn faded_yaw_uses_gravity_in_all_modes_even_when_the_head_is_tilted() {
        for mode in 0..4 {
            for sign in [1.0, -1.0] {
                let mut controls = Controls::default();
                controls.set_gravity(sign, mode);
                let mut camera = CameraController::new();
                camera.set_vr_rig(Vec3::new(50.0, 60.0, 70.0), Quat::IDENTITY);
                let head = Vec3::new(0.2, 1.0, -0.15);
                let mut input = VrInput {
                    head_position: Some(head),
                    head_rotation: Some(Quat::from_rotation_z(0.7) * Quat::from_rotation_x(1.0)),
                    ..Default::default()
                };
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
                let seat = camera.position() + camera.view_rotation() * head * 20.0;
                let up = controls.gravity.up(seat, Vec3::Y);
                let before = camera.view_rotation();
                input.turn = 1.0;
                for _ in 0..50 {
                    controls.update(&input, &mut camera, &mut 20.0, 0.01);
                }
                let expected = Quat::from_axis_angle(up, -std::f32::consts::FRAC_PI_4) * before;
                assert!(camera.view_rotation().angle_between(expected) < 0.001);
                assert!((camera.view_rotation() * Vec3::Y - up).length() < 0.001);
                assert!(
                    (camera.position() + camera.view_rotation() * head * 20.0 - seat).length()
                        < 0.002
                );
            }
        }
    }
    #[test]
    fn left_controller_buttons_toggle_once_and_visible_button_opens_wheel() {
        for binding in 0..3 {
            let mut controls = Controls::default();
            let mut camera = CameraController::new();
            let mut input = VrInput {
                head_position: Some(Vec3::ZERO),
                grips: [
                    Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                    None,
                ],
                ..Default::default()
            };
            match binding {
                0 => input.wheel_button = true,
                1 => input.stick_clicks[0] = true,
                _ => input.menus[0] = true,
            }
            for _ in 0..5 {
                controls.update(&input, &mut camera, &mut 20.0, 0.01);
                assert!(controls.wheel_open);
            }
            input.wheel_button = false;
            input.stick_clicks[0] = false;
            input.menus[0] = false;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            input.wheel_button = true;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            assert!(!controls.wheel_open);
            input.wheel_button = false;
            controls.update(&input, &mut camera, &mut 20.0, 0.01);
            let menu = controls.menu_pose.unwrap();
            input.aims[1] = Some(pose(
                position(menu) + rotation(menu) * Vec3::Z * 0.3,
                rotation(menu),
            ));
            input.triggers[1] = true;
            assert_eq!(controls.update(&input, &mut camera, &mut 20.0, 0.01), None);
            assert!(controls.wheel_open && controls.trigger_consumed);
        }
    }

    #[test]
    fn numeric_keypad_uses_wrist_pointer_without_selecting_wheel_choices() {
        let mut controls = Controls::default();
        controls.set_number_pad_active(true);
        let mut camera = CameraController::new();
        let mut input = VrInput {
            head_position: Some(Vec3::ZERO),
            grips: [
                Some(pose(Vec3::new(-0.2, -0.2, -0.5), Quat::IDENTITY)),
                None,
            ],
            ..Default::default()
        };

        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        let panel = controls.menu_pose.unwrap();
        aim_at_wheel(&mut input, panel, Vec2::splat(360.0));
        input.triggers[1] = true;

        assert_eq!(controls.update(&input, &mut camera, &mut 20.0, 0.01), None);
        assert!(controls.wheel_open && controls.number_pad_active);
        assert!(controls
            .pointer
            .is_some_and(|pointer| (pointer - Vec2::splat(360.0)).length() < 0.001));
        assert!(controls.wheel_pointer_pressed && controls.trigger_consumed);
        assert!(controls.hovered.is_none());

        input.triggers[1] = false;
        controls.update(&input, &mut camera, &mut 20.0, 0.01);
        assert!(controls.wheel_pointer_released);
    }
}
