//! In-game tutorial system for Bio-Spheres.
//!
//! Walks the player step-by-step through building a 2-cell-type organism
//! (Phagocyte head + Flagellocyte tail) in the Genome Editor.
//!
//! # Gate system
//! Every step except the introductory and summary steps has a [`StepGate`]
//! condition that must be satisfied before "Next" is enabled.  The gate is
//! checked each frame against the live genome / editor state, so the player
//! can't skip ahead without actually performing the action.
//!
//! # Visual overlay
//! A centred dialogue box is rendered with a teal accent, a progress bar,
//! a gate-status row (lock icon + hint when blocked; green tick when clear),
//! and a schematic pointer line (orthogonal, circuit-board style) that
//! highlights the panel being discussed.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Gate system
// ─────────────────────────────────────────────────────────────────────────────

/// A condition that must be true before the player can advance to the next step.
pub enum StepGate {
    /// Always satisfied — the player can proceed immediately.
    None,
    /// `editor_state.selected_mode_index` must equal `idx`.
    ModeSelected(usize),
    /// `genome.modes[mode_idx].cell_type` must equal `expected`.
    CellTypeSet { mode_idx: usize, expected: i32 },
    /// `genome.modes[mode_idx].parent_make_adhesion` must be `true`.
    AdhesionEnabled(usize),
    /// `genome.modes[mode_idx].child_b.mode_number` must equal `target`.
    ChildBMode { mode_idx: usize, target: i32 },
    /// Both child mode numbers of `mode_idx` must match `a_target` and `b_target`.
    ChildRouting { mode_idx: usize, a_target: i32, b_target: i32 },
    /// `genome.modes[mode_idx].max_splits` must equal `expected`.
    MaxSplitsSet { mode_idx: usize, expected: i32 },
}

impl StepGate {
    /// Evaluate the gate against the current genome / editor state.
    pub fn is_satisfied(
        &self,
        genome: &crate::genome::Genome,
        selected_mode: usize,
    ) -> bool {
        match self {
            StepGate::None => true,

            StepGate::ModeSelected(idx) => selected_mode == *idx,

            StepGate::CellTypeSet { mode_idx, expected } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.cell_type == *expected)
                .unwrap_or(false),

            StepGate::AdhesionEnabled(mode_idx) => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.parent_make_adhesion)
                .unwrap_or(false),

            StepGate::ChildBMode { mode_idx, target } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.child_b.mode_number == *target)
                .unwrap_or(false),

            StepGate::ChildRouting { mode_idx, a_target, b_target } => genome
                .modes
                .get(*mode_idx)
                .map(|m| {
                    m.child_a.mode_number == *a_target
                        && m.child_b.mode_number == *b_target
                })
                .unwrap_or(false),

            StepGate::MaxSplitsSet { mode_idx, expected } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.max_splits == *expected)
                .unwrap_or(false),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step data
// ─────────────────────────────────────────────────────────────────────────────

/// Which panel or element a tutorial step wants to point at.
///
/// Panel-level variants highlight the whole panel; element-level variants
/// (`ModeRow`, `CellTypeDropdown`, etc.) highlight and point at the specific
/// widget captured in `panel_rects` each frame.
pub enum TutorialTarget {
    // Panel-level
    ModesPanel,
    NameTypePanel,
    ParentSettingsPanel,
    AdhesionSettingsPanel,
    ChildRotationPanel,
    CircleSlidersPanel,
    TimeSliderPanel,
    SceneManagerPanel,
    None,
    // Element-level — point at a specific captured widget rect
    /// The Nth mode row button in the Modes list (0-indexed).
    ModeRow(usize),
    /// The cell-type ComboBox in the Name & Type panel.
    CellTypeDropdown,
    /// The "Make Adhesion" checkbox in the Name & Type panel.
    MakeAdhesionCheckbox,
    /// The Max Splits slider row in the Parent Settings panel.
    MaxSplitsSlider,
}

impl TutorialTarget {
    /// The key used in `panel_rects` to look up this target's screen rect.
    /// Returns `None` for `TutorialTarget::None`.
    pub fn panel_key(&self) -> Option<String> {
        match self {
            TutorialTarget::ModesPanel            => Some("Modes".to_string()),
            TutorialTarget::NameTypePanel         => Some("Modes".to_string()), // merged into Modes panel
            TutorialTarget::ParentSettingsPanel   => Some("ParentSettings".to_string()),
            TutorialTarget::AdhesionSettingsPanel => Some("AdhesionSettings".to_string()),
            TutorialTarget::ChildRotationPanel    => Some("QuaternionBall".to_string()),
            TutorialTarget::CircleSlidersPanel    => Some("CircleSliders".to_string()),
            TutorialTarget::TimeSliderPanel       => Some("TimeSlider".to_string()),
            TutorialTarget::SceneManagerPanel     => Some("SceneManager".to_string()),
            TutorialTarget::None                  => None,
            // Element-level keys — matched by `render_modes` and `render_parent_settings`
            TutorialTarget::ModeRow(idx)          => Some(format!("mode_row_{}", idx)),
            TutorialTarget::CellTypeDropdown      => Some("cell_type_dropdown".to_string()),
            TutorialTarget::MakeAdhesionCheckbox  => Some("make_adhesion_checkbox".to_string()),
            TutorialTarget::MaxSplitsSlider       => Some("max_splits_slider".to_string()),
        }
    }

    /// Returns `true` for element-level targets (specific widgets captured in
    /// `panel_rects`).  For these the arrow tip lands at `target_pos` inside
    /// the rect.  For panel-level targets the tip is clamped to the border so
    /// the arrow points *at* the panel rather than into its interior.
    pub fn is_element_level(&self) -> bool {
        matches!(
            self,
            TutorialTarget::ModeRow(_)
                | TutorialTarget::CellTypeDropdown
                | TutorialTarget::MakeAdhesionCheckbox
                | TutorialTarget::MaxSplitsSlider
        )
    }
}

/// Data for one tutorial step.
pub struct TutorialStepData {
    pub title: &'static str,
    pub body: &'static str,
    /// Short instruction shown in the status row when the gate is **not** satisfied.
    /// Use `""` for ungated steps (no status row is shown).
    pub gate_hint: &'static str,
    pub gate: StepGate,
    pub target: TutorialTarget,
    /// Normalised [x, y] position within the target panel where the arrow tip lands.
    /// [0.0, 0.0] = top-left, [1.0, 1.0] = bottom-right, [0.5, 0.5] = centre.
    pub target_pos: [f32; 2],
}

// ─────────────────────────────────────────────────────────────────────────────
// Tutorial content  —  "Head and Tail" (2-cell swimmer with repeating lifecycle)
//
// Creature design:
//   M1 = Phagocyte (head) — splits once to produce the tail, then on its
//                           next split sheds a free egg (Child A → M1,
//                           keep_adhesion = false) while keeping the tail
//                           (Child B → M2, keep_adhesion = true).
//                           max_splits = 1.  After-splits: A → M1, B → M2.
//   M2 = Flagellocyte (tail) — terminal swimmer.  max_splits = 0.
//
// Lifecycle:
//   Single M1 → splits once → M1 head + M2 tail (bonded, swimming pair)
//   M1 head hits max_splits → sheds a free M1 egg (detaches) + keeps M2 tail
//   Free M1 egg restarts the cycle independently
//
// This teaches: cell types, adhesion, max_splits, after-splits routing, and
// the keep-adhesion flag — all with a creature that actually reproduces.
// ─────────────────────────────────────────────────────────────────────────────

pub const TUTORIAL_STEPS: &[TutorialStepData] = &[
    // ── 0 ── Welcome ─────────────────────────────────────────────────────────
    TutorialStepData {
        title: "Welcome to Bio-Spheres!",
        body:  "In this tutorial you'll build a real living creature from scratch — \
                one that swims, eats, and reproduces on its own.\n\n\
                Your creature will have just two cell types:\n\n\
                • M1 — a Phagocyte head that eats food and eventually sheds an egg\n\
                • M2 — a Flagellocyte tail that propels the whole body\n\n\
                The head divides once to grow the tail, then later sheds a free \
                copy of itself that drifts off and starts the whole cycle again. \
                No dead ends — this creature keeps going.\n\n\
                Follow the steps one at a time. You can't move on until you've \
                done the action described. Tip: drag the Time Slider at the \
                bottom of the screen at any point to preview your creature.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::None,
        target_pos: [0.5, 0.5],
    },

    // ── 1 ── Select M1 ───────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 1 — Open Mode 1",
        body:  "Look at the Modes panel on the left. It lists the cell blueprints \
                in your creature's genome — each one is a recipe for a different \
                type of cell. A fresh genome starts with 10 modes; you can add \
                or remove them with the + and − buttons at the top.\n\n\
                M1 is selected by default, so all the other panels are already \
                showing M1's settings. Confirm it's highlighted, then continue.",
        gate_hint:  "Click M1 in the Modes panel to continue.",
        gate:       StepGate::ModeSelected(0),
        target:     TutorialTarget::ModeRow(0),
        target_pos: [0.5, 0.5],
    },

    // ── 2 ── Modes panel overview ────────────────────────────────────────────
    TutorialStepData {
        title: "Step 2 — The Modes Panel",
        body:  "A few things worth knowing about the Modes panel before we start.\n\n\
                The small dot to the left of each mode is the Initial Mode marker. \
                The mode with the dot lit is the cell your creature starts as. \
                It's on M1 right now — that's correct.\n\n\
                The buttons at the top:\n\
                • Copy Into — copies settings from the selected mode into the \
                  next mode you click. Handy for making similar cell types.\n\
                • ⟲ Reset — wipes the selected mode back to defaults.\n\
                • + / − — add or remove modes.\n\n\
                You can click the coloured square on a mode row to change its \
                colour, double-click the name to rename it, and hold Ctrl or \
                Shift to select multiple modes at once.",
        gate_hint:  "",
        gate:       StepGate::None,
        target:     TutorialTarget::ModesPanel,
        target_pos: [0.5, 0.04],
    },

    // ── 3 ── M1 cell type ────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 3 — Make M1 a Phagocyte",
        body:  "Every mode has a Type — this decides what the cell actually does \
                when it's alive in the simulation.\n\n\
                Look at the Name & Type panel. You'll see the Type dropdown for \
                the currently selected mode. Open it and choose Phagocyte.\n\n\
                A Phagocyte absorbs free-floating nutrient particles from the \
                environment on contact. It's the main food-gathering cell type — \
                our head cell will eat to fuel the whole creature.",
        gate_hint:  "Set M1's Type to Phagocyte to continue.",
        gate:       StepGate::CellTypeSet { mode_idx: 0, expected: 2 },
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    // ── 4 ── M1 adhesion ─────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 4 — Stick the Head to the Tail",
        body:  "By default, when a cell divides its two children float away from \
                each other. We want a body — cells that stay connected.\n\n\
                Tick the Make Adhesion checkbox in the Name & Type panel.\n\n\
                With adhesion on, each new child bonds to its sibling \
                The head and tail will stay glued together as one organism \
                instead of drifting apart.\n\n\
                Try dragging the Time Slider after ticking this to see the \
                difference.",
        gate_hint:  "Tick Make Adhesion on M1 to continue.",
        gate:       StepGate::AdhesionEnabled(0),
        target:     TutorialTarget::MakeAdhesionCheckbox,
        target_pos: [0.5, 0.5],
    },

    // ── 5 ── Child Rotation panel explained ──────────────────────────────────
    // ── 5 ── Circle Sliders panel explained ──────────────────────────────────
    TutorialStepData {
        title: "Step 5 — The Circle Sliders Panel",
        body:  "The Circle Sliders panel has two circular dials: Pitch and Yaw. \
                They set a rotation offset that is applied to the parent's \
                current orientation each time it divides.\n\n\
                Each division, the split axis is the parent's accumulated \
                orientation rotated by this pitch/yaw offset. Because the \
                offset compounds with each generation, even a small angle \
                produces a consistent incremental turn at every split — \
                useful for things like curved chains or angled branches.\n\n\
                At zero (the default) the split axis stays aligned with \
                whatever direction the parent is already facing. For this \
                tutorial you don't need to change these.",
        gate_hint:  "",
        gate:       StepGate::None,
        target:     TutorialTarget::CircleSlidersPanel,
        target_pos: [0.5, 0.5],
    },

    // ── 6 ── Child Rotation panel explained ──────────────────────────────────
    TutorialStepData {
        title: "Step 6 — The Child Rotation Panel",
        body:  "Below the Circle Sliders is the Child Rotation panel — two 3D \
                balls labelled Child A and Child B.\n\n\
                Where the Circle Sliders control the parent's split axis, these \
                balls control each child's own orientation after it is born — \
                which way it is facing when it appears. Drag a ball to rotate \
                that child's starting orientation.\n\n\
                Below each ball you'll find:\n\
                • A Mode dropdown — which mode (cell type) that child becomes.\n\
                • A Keep Adhesion checkbox — whether the child stays bonded to \
                  the parent after the split.\n\n\
                You'll use the Mode dropdown and Keep Adhesion checkbox in the \
                next steps.",
        gate_hint:  "",
        gate:       StepGate::None,
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.5, 0.5],
    },

    // ── 7 ── M1 max splits = 1 ───────────────────────────────────────────────
    TutorialStepData {
        title: "Step 7 — The Head Divides Once",
        body:  "Open the Parent Settings panel and set Max Splits to 1.\n\n\
                This means M1 is allowed to divide exactly once during its \
                normal growth phase. That one split produces the tail (M2). \
                After that, the head is \"used up\" — but instead of stopping \
                forever, it will do something special on its next split. \
                You'll set that up in the next steps.",
        gate_hint:  "Set Max Splits to 1 on M1 to continue.",
        gate:       StepGate::MaxSplitsSet { mode_idx: 0, expected: 1 },
        target:     TutorialTarget::MaxSplitsSlider,
        target_pos: [0.5, 0.5],
    },

    // ── 7b ── M1 normal child routing ────────────────────────────────────────
    TutorialStepData {
        title: "Step 8 — Grow the Tail",
        body:  "Look at the Child Rotation panel. Under the Child B ball you'll \
                see a Mode dropdown — this decides what cell type Child B becomes \
                when M1 divides.\n\n\
                Change Child B's Mode to M2.\n\n\
                Right now both children default to M1, so the head just clones \
                itself. By routing Child B → M2, every split produces a \
                Flagellocyte tail instead of another head.\n\n\
                Drag the Time Slider after making this change — you should see \
                a two-cell creature: one Phagocyte head, one Flagellocyte tail.",
        gate_hint:  "Set Child B's Mode to M2 in the Child Rotation panel to continue.",
        gate:       StepGate::ChildBMode { mode_idx: 0, target: 1 },
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.75, 0.7],
    },

    // ── 8 ── M1 egg shedding via keep_adhesion ────────────────────────────────
    TutorialStepData {
        title: "Step 9 — Shed an Egg",
        body:  "The head divides once and stops — but we want it to reproduce. \
                Here's the trick: once M1 hits its split limit, it uses the same \
                Child A and Child B modes again. Child A becomes a new M1 head, \
                Child B stays as the M2 tail.\n\n\
                The only difference is that the new M1 head should float away \
                freely instead of staying bonded. Find the Keep Adhesion \
                checkbox under the Child A ball in the Child Rotation panel \
                and untick it.\n\n\
                Now every time the head reaches its limit it sheds a free M1 \
                egg that drifts off and starts the cycle again — while the \
                original tail stays attached.",
        gate_hint:  "",
        gate:       StepGate::None,
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.25, 0.7],
    },

    // ── 9 ── Select M2 ───────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 10 — Open Mode 2",
        body:  "M1 is fully set up. Now let's define the tail.\n\n\
                Click M2 in the Modes panel.",
        gate_hint:  "Click M2 in the Modes panel to continue.",
        gate:       StepGate::ModeSelected(1),
        target:     TutorialTarget::ModeRow(1),
        target_pos: [0.5, 0.5],
    },

    // ── 10 ── M2 cell type ────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 11 — Give M2 a Flagella",
        body:  "Open the Type dropdown and choose Flagellocyte.\n\n\
                A Flagellocyte has a whip-like flagellum tail. It beats the \
                flagellum to push itself — and anything bonded to it — through \
                the fluid. This is what will actually move your creature around.",
        gate_hint:  "Set M2's Type to Flagellocyte to continue.",
        gate:       StepGate::CellTypeSet { mode_idx: 1, expected: 1 },
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    // ── 11 ── M2 adhesion ────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 12 — Attach the Tail to the Body",
        body:  "Tick Make Adhesion for M2.\n\n\
                A Flagellocyte without adhesion just swims off on its own. With \
                adhesion on, its thrust is applied to the whole bonded cluster — \
                head and tail move together as one creature.",
        gate_hint:  "Tick Make Adhesion on M2 to continue.",
        gate:       StepGate::AdhesionEnabled(1),
        target:     TutorialTarget::MakeAdhesionCheckbox,
        target_pos: [0.5, 0.5],
    },

    // ── 12 ── M2 max splits = 0 ──────────────────────────────────────────────
    TutorialStepData {
        title: "Step 13 — The Tail Never Divides",
        body:  "Set Max Splits to 0 for M2.\n\n\
                The tail's only job is to swim — it should never divide and \
                produce more cells. Zero means it is permanently locked as a \
                terminal cell. It will just keep beating its flagellum for the \
                life of the creature.",
        gate_hint:  "Set Max Splits to 0 on M2 to continue.",
        gate:       StepGate::MaxSplitsSet { mode_idx: 1, expected: 0 },
        target:     TutorialTarget::MaxSplitsSlider,
        target_pos: [0.5, 0.5],
    },

    // ── 13 ── Preview ────────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 14 — Watch It Come to Life",
        body:  "Drag the Time Slider to fast-forward through time.\n\n\
                You'll see a single Phagocyte appear, then split into a head-tail \
                pair. The Flagellocyte tail starts beating and the pair swims \
                around eating food.\n\n\
                After a while the head hits its split limit and sheds a new free \
                Phagocyte egg. That egg drifts off, grows its own tail, and the \
                cycle repeats — your creature reproduces.",
        gate_hint:  "",
        gate:       StepGate::None,
        target:     TutorialTarget::TimeSliderPanel,
        target_pos: [0.5, 0.5],
    },

    // ── 13 ── Complete ───────────────────────────────────────────────────────
    TutorialStepData {
        title: "You Did It!",
        body:  "You've built a creature with a repeating lifecycle — \
                it grows, swims, eats, and reproduces, all from just two modes.\n\n\
                Some ideas for what to try next:\n\
                • Add a second Flagellocyte (M3, Max Splits = 0) and route M1's \
                  normal Child B → M3 — now the creature grows two tails\n\
                • Try a Vasculocyte with Outlet enabled — it forms high-speed \
                  nutrient pipes through the body, 5× faster than normal transport\n\
                • Add a Glueocyte to bond to other organisms or cave walls\n\
                • Add a Devorocyte to steal nutrients from foreign cells on contact\n\
                • Use the Signal system (Division Signal Channel in Parent Settings) \
                  to gate cell division on a signal — cells only divide when they \
                  receive the right chemical cue\n\
                • Use the 🎲 Procedural button to generate a random creature \
                  and study how it's built\n\
                • When you're happy, click Live Simulation and to release your \
                  creature into the full GPU world just hold ALT and select + from\
                  the wheel",
        gate_hint:  "",
        gate:       StepGate::None,
        target:     TutorialTarget::None,
        target_pos: [0.5, 0.5],
    },
];

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent tutorial playback state, stored inside [`GlobalUiState`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TutorialState {
    /// Whether the tutorial dialogue is currently visible.
    pub active: bool,
    /// Index into [`TUTORIAL_STEPS`] for the current step.
    pub current_step: usize,
    /// `true` once the tutorial has been started at least once — suppresses
    /// the automatic first-launch opening on subsequent sessions.
    pub ever_shown: bool,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self { active: false, current_step: 0, ever_shown: false }
    }
}

impl TutorialState {
    pub fn total_steps() -> usize { TUTORIAL_STEPS.len() }

    /// Data for the current step (clamped so we never go out of bounds).
    pub fn current(&self) -> &TutorialStepData {
        &TUTORIAL_STEPS[self.current_step.min(TUTORIAL_STEPS.len() - 1)]
    }

    /// Advance one step.  Returns `true` when the last step is passed.
    pub fn next(&mut self) -> bool {
        if self.current_step + 1 >= TUTORIAL_STEPS.len() {
            self.active = false;
            return true;
        }
        self.current_step += 1;
        false
    }

    pub fn prev(&mut self) {
        if self.current_step > 0 { self.current_step -= 1; }
    }

    pub fn start(&mut self) {
        self.active = true;
        self.current_step = 0;
        self.ever_shown = true;
    }

    pub fn close(&mut self) { self.active = false; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Visual constants
// ─────────────────────────────────────────────────────────────────────────────

const TEAL:          egui::Color32 = egui::Color32::from_rgb(0, 220, 175);
const TEAL_DIM:      egui::Color32 = egui::Color32::from_rgb(0, 100, 80);
const PANEL_GLOW:    egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 0, 0, 0);
const GATE_LOCKED:   egui::Color32 = egui::Color32::from_rgb(220, 160, 40);
const GATE_OPEN:     egui::Color32 = egui::Color32::from_rgb(80, 220, 120);
const LINE_W:  f32 = 1.5;
const DOT_R:   f32 = 3.5;
const ARROW_LEN:  f32 = 11.0;
const ARROW_HALF: f32 = 5.0;

// ─────────────────────────────────────────────────────────────────────────────
// Public render entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Render the tutorial overlay.
///
/// Call once per frame in `UiSystem::end_frame`, **after** the dock area
/// renders (so panel rects are populated) and **before** `ctx.end_pass()`.
pub fn render_tutorial(
    ctx:            &egui::Context,
    state:          &mut TutorialState,
    panel_rects:    &HashMap<String, egui::Rect>,
    genome:         &crate::genome::Genome,
    selected_mode:  usize,
) {
    if !state.active { return; }

    let step_index  = state.current_step;
    let total       = TutorialState::total_steps();
    // Extract all values we need from the step before any mutable borrows of state.
    // `target_key` is Option<String> because element-level keys are dynamic (e.g. "mode_row_0").
    let (gate_ok, has_hint, target_key, target_pos, target_is_element, step_title, step_body, step_gate_hint) = {
        let step = state.current();
        (
            step.gate.is_satisfied(genome, selected_mode),
            !step.gate_hint.is_empty(),
            step.target.panel_key(),   // Option<String>
            step.target_pos,
            step.target.is_element_level(),
            step.title,
            step.body,
            step.gate_hint,
        )
    };

    // ── Panel highlight ──────────────────────────────────────────────────────
    if let Some(ref key) = target_key {
        if let Some(&panel_rect) = panel_rects.get(key.as_str()) {
            let bg = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("tut_panel_bg"),
            ));
            let t = ctx.input(|i| i.time) as f32;
            let pulse = (t * 2.2).sin() * 0.5 + 0.5;
            let expand = 3.0 + pulse * 3.0;
            bg.rect(
                panel_rect.expand(expand),
                4.0,
                PANEL_GLOW,
                egui::Stroke::new(
                    1.5 + pulse * 0.5,
                    egui::Color32::from_rgba_premultiplied(0, 220, 175, 75),
                ),
                egui::StrokeKind::Outside,
            );
        }
    }

    // ── Dialogue window ──────────────────────────────────────────────────────
    let mut dialogue_rect: Option<egui::Rect> = None;
    let mut next_clicked  = false;
    let mut prev_clicked  = false;
    let mut close_clicked = false;

    let frame = egui::Frame::window(&ctx.global_style())
        .inner_margin(egui::Margin::same(16))
        .stroke(egui::Stroke::new(1.5, TEAL))
        .shadow(egui::Shadow {
            offset: [0, 4],
            blur: 18,
            spread: 2,
            color: egui::Color32::from_black_alpha(130),
        });

    // Place the dialogue at the top-left of the viewport, falling back to the
    // screen origin if the viewport rect hasn't been captured yet.
    let viewport_origin = panel_rects
        .get("Viewport")
        .map(|r| r.min + egui::vec2(10.0, 10.0))
        .unwrap_or(egui::pos2(10.0, 10.0));

    let win_response = egui::Window::new("▶  Tutorial")
        .id(egui::Id::new("tutorial_dialog"))
        .collapsible(false)
        .resizable(false)
        .default_pos(viewport_origin)
        .fixed_size([380.0, 0.0])
        .frame(frame)
        .show(ctx, |ui| {
            // dialogue_rect is captured from the outer response below
            let _ = ui.max_rect(); // keep borrow happy

            // Title
            ui.label(
                egui::RichText::new(step_title)
                    .strong()
                    .size(15.0)
                    .color(TEAL),
            );
            ui.add_space(5.0);
            ui.separator();
            ui.add_space(5.0);

            // Body
            ui.label(egui::RichText::new(step_body).size(13.0));
            ui.add_space(10.0);

            // Gate status row (only shown for gated steps)
            if has_hint {
                ui.separator();
                ui.add_space(4.0);
                if gate_ok {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("✔  Done — click Next to continue.")
                                .size(12.0)
                                .color(GATE_OPEN),
                        );
                    });
                } else {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("⟳  {}", step_gate_hint))
                                .size(12.0)
                                .color(GATE_LOCKED),
                        );
                    });
                }
                ui.add_space(4.0);
            }

            // Progress bar
            let progress = (step_index + 1) as f32 / total as f32;
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_height(3.0)
                    .fill(TEAL),
            );
            ui.add_space(8.0);

            // Button row
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("{} / {}", step_index + 1, total))
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Close
                    if ui.button(egui::RichText::new("✕ Close").size(12.0)).clicked() {
                        close_clicked = true;
                    }
                    ui.add_space(4.0);

                    // Next / Finish  — disabled until gate is satisfied
                    let is_last   = step_index + 1 >= total;
                    let next_text = if is_last { "  Finish ✓  " } else { "  Next ›  " };
                    let next_btn  = egui::Button::new(
                        egui::RichText::new(next_text).size(12.0).strong(),
                    )
                    .fill(if gate_ok {
                        egui::Color32::from_rgb(0, 70, 55)
                    } else {
                        egui::Color32::from_rgb(30, 45, 40)
                    });

                    if ui.add_enabled(gate_ok, next_btn).clicked() {
                        next_clicked = true;
                    }
                    ui.add_space(4.0);

                    // Back (hidden on first step)
                    if step_index > 0
                        && ui.button(egui::RichText::new("‹ Back").size(12.0)).clicked()
                    {
                        prev_clicked = true;
                    }
                });
            });
        });

    // Capture the true outer window rect (includes title bar + frame).
    if let Some(ref r) = win_response {
        dialogue_rect = Some(r.response.rect);
    }

    // ── Apply navigation ─────────────────────────────────────────────────────
    if close_clicked { state.close(); return; }
    if next_clicked  { state.next(); }
    else if prev_clicked { state.prev(); }

    // ── Schematic pointer line ───────────────────────────────────────────────
    if let (Some(d_rect), Some(ref key)) = (dialogue_rect, target_key) {
        if let Some(&p_rect) = panel_rects.get(key.as_str()) {
            let tip = if target_is_element {
                // Element-level target: land at the exact widget position inside the rect.
                let tp = target_pos;
                egui::pos2(
                    p_rect.left() + p_rect.width()  * tp[0],
                    p_rect.top()  + p_rect.height() * tp[1],
                )
            } else {
                // Panel-level target: land on the border edge of the panel that
                // faces the dialogue, so the arrow points *at* the panel rather
                // than into its interior.
                nearest_edge_midpoint(p_rect, d_rect.center())
            };
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("tut_pointer"),
            ));
            draw_schematic_pointer(&painter, d_rect, tip, ctx, gate_ok);
        }
    }

    ctx.request_repaint();
}

// ─────────────────────────────────────────────────────────────────────────────
// Schematic pointer drawing
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a high-tech schematic pointer from the nearest edge-midpoint of
/// `d_rect` (dialogue) to the exact `tip` position within the target panel.
///
/// Colour shifts: amber when the gate is locked, teal when open/satisfied.
fn draw_schematic_pointer(
    painter:  &egui::Painter,
    d_rect:   egui::Rect,
    tip:      egui::Pos2,
    ctx:      &egui::Context,
    gate_ok:  bool,
) {
    let line_color = if gate_ok { TEAL } else { GATE_LOCKED };
    let dim_color  = if gate_ok { TEAL_DIM } else { egui::Color32::from_rgb(100, 70, 0) };

    let start = nearest_edge_midpoint(d_rect, tip);
    let end   = tip;

    if (end - start).length() < 10.0 { return; }

    // Orthogonal routing: horizontal-first if Δx ≥ Δy, otherwise vertical-first.
    let dx = (end.x - start.x).abs();
    let dy = (end.y - start.y).abs();
    let corner = if dx >= dy {
        egui::Pos2::new(end.x, start.y)
    } else {
        egui::Pos2::new(start.x, end.y)
    };

    let stroke     = egui::Stroke::new(LINE_W, line_color);
    let dim_stroke = egui::Stroke::new(LINE_W * 0.5, dim_color);

    // Segment 1: start → corner
    painter.line_segment([start, corner], stroke);

    // Segment 2: corner → just before arrowhead
    let dir2 = (end - corner).normalized();
    let pre  = end - dir2 * ARROW_LEN;
    if (pre - corner).length() > 1.0 {
        painter.line_segment([corner, pre], stroke);
    }

    // Tick marks along segment 2
    let seg2_len = (pre - corner).length();
    if seg2_len > 40.0 {
        let count = ((seg2_len / 20.0) as usize).max(1).min(14);
        let perp  = egui::Vec2::new(-dir2.y, dir2.x);
        for i in 1..=count {
            let t = i as f32 / (count + 1) as f32;
            let c = corner + (pre - corner) * t;
            painter.line_segment([c - perp * 4.5, c + perp * 4.5], dim_stroke);
        }
    }

    // Arrowhead
    draw_arrowhead(painter, end, dir2, line_color);

    // Corner dot
    painter.circle_filled(corner, DOT_R, line_color);

    // Hollow origin circle
    painter.circle_stroke(start, 4.0, egui::Stroke::new(1.5, line_color));

    // Animated scan dot — colour changes with gate state
    let t      = ctx.input(|i| i.time) as f32;
    let period = 1.8_f32;
    let phase  = (t % period) / period;

    let total_path = (corner - start).length() + (end - corner).length();
    let traveled   = phase * total_path;
    let seg1_len   = (corner - start).length();

    let scan_pos = if traveled <= seg1_len || seg1_len < 1.0 {
        let dir1 = if seg1_len > 0.001 { (corner - start).normalized() } else { dir2 };
        start + dir1 * traveled.min(seg1_len)
    } else {
        let rem = traveled - seg1_len;
        corner + dir2 * rem.min((end - corner).length())
    };

    let alpha = ((t * 4.0).sin() * 0.5 + 0.5) * 200.0 + 55.0;
    let (sr, sg, sb) = if gate_ok { (0, 220, 175) } else { (220, 160, 40) };
    let scan_col  = egui::Color32::from_rgba_premultiplied(sr, sg, sb, alpha as u8);
    let scan_ring = egui::Color32::from_rgba_premultiplied(sr, sg, sb, 70);
    painter.circle_filled(scan_pos, 3.0, scan_col);
    painter.circle_stroke(scan_pos, 4.5, egui::Stroke::new(1.0, scan_ring));
}

/// Return the midpoint of whichever edge of `rect` faces `toward`.
///
/// Compared to a ray-exit approach, this always produces a clean anchor at
/// the centre of one of the four sides — the arrow never slides around
/// diagonally as the two rects move relative to each other.
fn nearest_edge_midpoint(rect: egui::Rect, toward: egui::Pos2) -> egui::Pos2 {
    let c  = rect.center();
    let dx = toward.x - c.x;
    let dy = toward.y - c.y;

    if dx.abs() >= dy.abs() {
        // Left or right edge
        egui::pos2(if dx >= 0.0 { rect.right() } else { rect.left() }, c.y)
    } else {
        // Top or bottom edge
        egui::pos2(c.x, if dy >= 0.0 { rect.bottom() } else { rect.top() })
    }
}

fn draw_arrowhead(
    painter: &egui::Painter,
    tip: egui::Pos2,
    dir: egui::Vec2,
    color: egui::Color32,
) {
    let perp  = egui::Vec2::new(-dir.y, dir.x);
    let base  = tip - dir * ARROW_LEN;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, base + perp * ARROW_HALF, base - perp * ARROW_HALF],
        color,
        egui::Stroke::NONE,
    ));
}
