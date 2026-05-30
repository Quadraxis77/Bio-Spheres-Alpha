//! In-game tutorial system for Bio-Spheres.
//!
//! Teaches the core mechanics through guided experimentation rather than
//! prescribing a single organism.  Each chapter introduces one mechanic,
//! asks the player to change something, and has them observe the result
//! in the Time Slider before moving on.
//!
//! # Gate system
//! Action steps have a [`StepGate`] that must be satisfied before "Next"
//! is enabled.  Gates are checked each frame against the live genome so
//! the player can't skip ahead without actually doing the thing.
//!
//! # Visual overlay
//! A floating dialogue with a teal accent, progress bar, gate-status row,
//! and an orthogonal "circuit-board" pointer line to the relevant panel.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Gate system
// ─────────────────────────────────────────────────────────────────────────────

/// A condition that must be true before the player can advance.
pub enum StepGate {
    /// Always satisfied.
    None,
    /// The selected mode index must equal `idx`.
    ModeSelected(usize),
    /// `genome.modes[mode_idx].cell_type` must equal `expected`.
    CellTypeSet { mode_idx: usize, expected: i32 },
    /// At least one mode has `cell_type != 0` (player changed a type).
    AnyCellTypeChanged,
    /// `genome.modes[mode_idx].parent_make_adhesion` must be `true`.
    AdhesionEnabled(usize),
    /// At least one mode has `parent_make_adhesion == true`.
    AnyAdhesionEnabled,
    /// At least one mode has `parent_make_adhesion == false` (player turned one off).
    AnyAdhesionDisabled,
    /// `genome.modes[mode_idx].child_b.mode_number` must equal `target`.
    ChildBMode { mode_idx: usize, target: i32 },
    /// At least one mode routes Child B to a different mode than Child A.
    AnyChildBDifferentMode,
    /// `genome.modes[mode_idx].max_splits` must equal `expected`.
    MaxSplitsSet { mode_idx: usize, expected: i32 },
    /// At least one mode has `max_splits` set to a finite value (>= 0).
    AnyFiniteMaxSplits,
    /// `genome.modes[mode_idx].child_a.keep_adhesion` must be `false`.
    ChildAKeepAdhesionOff(usize),
    /// At least one mode has `split_interval` different from the default (1.0).
    AnySplitIntervalChanged,
    /// At least one mode has `split_mass` different from the default (1.5).
    AnySplitMassChanged,
    /// Both child mode numbers of `mode_idx` must match `a_target` and `b_target`.
    ChildRouting { mode_idx: usize, a_target: i32, b_target: i32 },
}

impl StepGate {
    pub fn is_satisfied(
        &self,
        genome: &crate::genome::Genome,
        selected_mode: usize,
    ) -> bool {
        match self {
            StepGate::None => true,

            StepGate::ModeSelected(idx) => selected_mode == *idx,

            StepGate::CellTypeSet { mode_idx, expected } => genome
                .modes.get(*mode_idx)
                .map(|m| m.cell_type == *expected)
                .unwrap_or(false),

            StepGate::AnyCellTypeChanged =>
                genome.modes.iter().any(|m| m.cell_type != 0),

            StepGate::AdhesionEnabled(mode_idx) => genome
                .modes.get(*mode_idx)
                .map(|m| m.parent_make_adhesion)
                .unwrap_or(false),

            StepGate::AnyAdhesionEnabled =>
                genome.modes.iter().any(|m| m.parent_make_adhesion),

            StepGate::AnyAdhesionDisabled =>
                genome.modes.iter().any(|m| !m.parent_make_adhesion),

            StepGate::ChildBMode { mode_idx, target } => genome
                .modes.get(*mode_idx)
                .map(|m| m.child_b.mode_number == *target)
                .unwrap_or(false),

            StepGate::AnyChildBDifferentMode =>
                genome.modes.iter().any(|m| {
                    m.child_b.mode_number != m.child_a.mode_number
                }),

            StepGate::MaxSplitsSet { mode_idx, expected } => genome
                .modes.get(*mode_idx)
                .map(|m| m.max_splits == *expected)
                .unwrap_or(false),

            StepGate::AnyFiniteMaxSplits =>
                genome.modes.iter().any(|m| m.max_splits >= 0),

            StepGate::ChildAKeepAdhesionOff(mode_idx) => genome
                .modes.get(*mode_idx)
                .map(|m| !m.child_a.keep_adhesion)
                .unwrap_or(false),

            StepGate::AnySplitIntervalChanged =>
                genome.modes.iter().any(|m| (m.split_interval - 1.0).abs() > 0.05),

            StepGate::AnySplitMassChanged =>
                genome.modes.iter().any(|m| (m.split_mass - 1.5).abs() > 0.05),

            StepGate::ChildRouting { mode_idx, a_target, b_target } => genome
                .modes.get(*mode_idx)
                .map(|m| {
                    m.child_a.mode_number == *a_target
                        && m.child_b.mode_number == *b_target
                })
                .unwrap_or(false),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step data
// ─────────────────────────────────────────────────────────────────────────────

/// Which panel or element a tutorial step points at.
pub enum TutorialTarget {
    ModesPanel,
    NameTypePanel,
    ParentSettingsPanel,
    AdhesionSettingsPanel,
    ChildRotationPanel,
    CircleSlidersPanel,
    TimeSliderPanel,
    SceneManagerPanel,
    None,
    ModeRow(usize),
    CellTypeDropdown,
    MakeAdhesionCheckbox,
    MaxSplitsSlider,
    AfterSplitsChildA,
    AfterSplitsChildB,
    ChildAKeepAdhesion,
    ChildBKeepAdhesion,
}

impl TutorialTarget {
    pub fn panel_key(&self) -> Option<String> {
        match self {
            TutorialTarget::ModesPanel            => Some("Modes".to_string()),
            TutorialTarget::NameTypePanel         => Some("Modes".to_string()),
            TutorialTarget::ParentSettingsPanel   => Some("ParentSettings".to_string()),
            TutorialTarget::AdhesionSettingsPanel => Some("AdhesionSettings".to_string()),
            TutorialTarget::ChildRotationPanel    => Some("QuaternionBall".to_string()),
            TutorialTarget::CircleSlidersPanel    => Some("CircleSliders".to_string()),
            TutorialTarget::TimeSliderPanel       => Some("TimeSlider".to_string()),
            TutorialTarget::SceneManagerPanel     => Some("SceneManager".to_string()),
            TutorialTarget::None                  => None,
            TutorialTarget::ModeRow(idx)          => Some(format!("mode_row_{}", idx)),
            TutorialTarget::CellTypeDropdown      => Some("cell_type_dropdown".to_string()),
            TutorialTarget::MakeAdhesionCheckbox  => Some("make_adhesion_checkbox".to_string()),
            TutorialTarget::MaxSplitsSlider       => Some("max_splits_slider".to_string()),
            TutorialTarget::AfterSplitsChildA     => Some("after_splits_child_a".to_string()),
            TutorialTarget::AfterSplitsChildB     => Some("after_splits_child_b".to_string()),
            TutorialTarget::ChildAKeepAdhesion    => Some("child_a_keep_adhesion".to_string()),
            TutorialTarget::ChildBKeepAdhesion    => Some("child_b_keep_adhesion".to_string()),
        }
    }

    pub fn is_element_level(&self) -> bool {
        matches!(
            self,
            TutorialTarget::ModeRow(_)
                | TutorialTarget::CellTypeDropdown
                | TutorialTarget::MakeAdhesionCheckbox
                | TutorialTarget::MaxSplitsSlider
                | TutorialTarget::AfterSplitsChildA
                | TutorialTarget::AfterSplitsChildB
                | TutorialTarget::ChildAKeepAdhesion
                | TutorialTarget::ChildBKeepAdhesion
        )
    }
}

pub struct TutorialStepData {
    pub title: &'static str,
    pub body: &'static str,
    pub gate_hint: &'static str,
    pub gate: StepGate,
    pub target: TutorialTarget,
    pub target_pos: [f32; 2],
}

// ─────────────────────────────────────────────────────────────────────────────
// Tutorial steps
//
// Structure: 8 chapters, each teaching one mechanic through observation.
// The player is never told to build a specific organism — they experiment
// with settings and watch what changes.  Gates confirm they touched the
// mechanic, not that they set it to a prescribed value.
//
// Chapter 1 — Orientation:    what you're looking at, how to observe
// Chapter 2 — Division:       split nutrients and interval, watching timing
// Chapter 3 — Cell types:     changing type, observing behavior
// Chapter 4 — Adhesion:       bonding on/off, bodies vs solo cells
// Chapter 5 — Body shape:     child routing and orientation
// Chapter 6 — Lifecycle:      max splits, finite growth, after-splits
// Chapter 7 — Signals:        brief intro to the signal system
// Chapter 8 — Go live:        releasing into the full world
// ─────────────────────────────────────────────────────────────────────────────

pub const TUTORIAL_STEPS: &[TutorialStepData] = &[

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 1 — ORIENTATION
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 1 — The Genome Editor",
        body: "This is the Genome Editor. It runs a single organism from your \
               genome in a sandboxed simulation.\n\n\
               A genome is a list of modes. Each mode is a blueprint that \
               defines what a cell does and how it divides.\n\n\
               The Modes panel lists all modes in the genome. Click a mode \
               (M1, M2, …) to select it — the surrounding panels update to \
               show that mode's settings.\n\n\
               The Time Slider at the bottom scrubs through time. Drag it \
               right to watch your organism grow, left to rewind. Changes \
               to the genome take effect immediately.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ModesPanel,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Try It — Scrub Through Time",
        body: "Drag the Time Slider to the right.\n\n\
               Cells appear and divide. The default genome has 10 Phagocyte \
               modes — each one divides and routes its children back to \
               itself by default.\n\n\
               The genome can start from any mode and any cell type — there \
               is no required starting cell.\n\n\
               Drag the slider back to zero. The simulation rewinds — the \
               same genome always produces the same result.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::TimeSliderPanel,
        target_pos: [0.5, 0.5],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 2 — DIVISION TIMING
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 2 — Division Timing",
        body: "The Parent Settings panel controls when a cell divides.\n\n\
               The two most important settings:\n\n\
               Split Nutrients — how many nutrients the cell must accumulate \
               before it can divide. Higher = slower growth, larger cells.\n\n\
               Split Interval — minimum time in seconds between divisions. \
               A cell that has enough mass still waits this long before \
               splitting again.\n\n\
               Together these two sliders control the pace of your organism. \
               A fast-dividing creature with low split nutrients grows a large \
               colony quickly but each cell is small. A slow-dividing creature \
               with high split nutrients grows fewer, larger cells.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.2],
    },

    TutorialStepData {
        title: "Experiment — Change the Division Speed",
        body: "In the Parent Settings panel, try dragging Split Interval \
               to a high value (5–10 seconds), then scrub the Time Slider. \
               Notice how much slower the colony grows — each cell waits \
               longer before dividing.\n\n\
               Now try the minimum (1 second). The colony grows as fast as \
               nutrients allow.\n\n\
               Try the same with Split Nutrients — higher values mean cells need \
               more food before they can split, so growth slows down even if \
               the interval is short.\n\n\
               Change Split Interval or Split Nutrients on any mode to continue.",
        gate_hint: "Change Split Interval or Split Nutrients on any mode to continue.",
        gate:       StepGate::AnySplitIntervalChanged,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 3 — CELL TYPES
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 3 — Cell Types",
        body: "The Type dropdown sets what a cell does while it's alive.\n\n\
               Nutrient sources:\n\
               Phagocyte — absorbs free-floating nutrient particles on contact.\n\
               Photocyte — converts light into nutrients near a light source.\n\
               Devorocyte — steals nutrients from neighbouring foreign cells.\n\n\
               Locomotion:\n\
               Flagellocyte — beats a flagellum to propel itself and anything bonded to it.\n\
               Ciliocyte — uses rows of cilia to push nearby cells and fluid in the forward direction.\n\
               Myocyte — rhythmically contracts its adhesion bonds, producing \
               peristaltic pumping or coordinated movement.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Experiment — Give a Cell a Job",
        body: "Select any mode in the Modes panel and change its Type to \
               something other than Test.\n\n\
               You can mix types across modes — a Phagocyte mode and a \
               Flagellocyte mode together already give you a creature that \
               feeds and swims. The genome doesn't care which mode you \
               change — experiment freely.\n\n\
               Change at least one mode's Type to continue.",
        gate_hint: "Change at least one mode's Type to something other than Test.",
        gate:       StepGate::AnyCellTypeChanged,
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 4 — ADHESION
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 4 — Adhesion",
        body: "Adhesion is what turns a collection of dividing cells into a \
               body. Without it, every division produces two cells that \
               immediately drift apart — you get a cloud of individuals, \
               not an organism.\n\n\
               The Make Adhesion checkbox controls whether a cell creates a \
               spring bond between its two children when it divides. It is \
               off by default — siblings scatter freely unless you enable it.\n\n\
               The Adhesion Settings panel controls the physics of that \
               spring: stiffness, damping, break force, and twist constraints. \
               The defaults work well for most creatures.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::AdhesionSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    TutorialStepData {
        title: "Experiment — Bonds On and Off",
        body: "Select any mode and tick Make Adhesion on. Scrub the Time Slider.\n\n\
               The two children produced by each division are now bonded \
               together — they form a connected body instead of drifting apart.\n\n\
               Untick it again and the siblings scatter on every division.\n\n\
               Adhesion is per-mode. You can have some modes bonded and others \
               free. A mode with adhesion off produces free cells that drift \
               away and start their own independent lifecycle — this is how \
               reproduction works.\n\n\
               Enable Make Adhesion on any mode to continue.",
        gate_hint: "Enable Make Adhesion on any mode to continue.",
        gate:       StepGate::AnyAdhesionEnabled,
        target:     TutorialTarget::MakeAdhesionCheckbox,
        target_pos: [0.5, 0.5],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 5 — BODY SHAPE
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 5 — Body Shape",
        body: "So far every cell divides into two copies of itself. To build \
               a body with different cell types in different positions, you \
               need to route children to different modes.\n\n\
               The Child Rotation panel has two 3D balls: Child A and Child B. \
               Below each ball:\n\n\
               Mode — which blueprint this child uses after the split. \
               Change Child B's Mode to M2 and every division produces one \
               M1 cell and one M2 cell instead of two M1 cells.\n\n\
               Keep Adhesion — whether this child stays bonded to the parent. \
               Untick it and that child floats away freely.\n\n\
               The balls themselves control orientation — drag a ball to set \
               which direction that child faces when it's born. A 180° rotation \
               makes cells grow in a line; 90° makes them branch sideways.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Experiment — Route Children to Different Modes",
        body: "Select M1 and look at the Child Rotation panel.\n\n\
               Change Child B's Mode dropdown to M2 (or any other mode). \
               Scrub the Time Slider — every M1 division now produces one \
               M1 and one M2. If M2 has a different cell type, you'll see \
               two distinct cell types in the body.\n\n\
               Now try dragging the Child B ball to a different orientation. \
               The M2 cells will be born facing a different direction — \
               if M2 is a Flagellocyte, the thrust direction changes.\n\n\
               Try the Circle Sliders panel (the pitch/yaw dials). These \
               rotate the split axis itself — a small yaw angle makes the \
               organism curve as it grows, producing spirals or arcs.\n\n\
               Route Child B to a different mode than Child A to continue.",
        gate_hint: "Set Child B's Mode to a different mode than Child A on any mode.",
        gate:       StepGate::AnyChildBDifferentMode,
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.75, 0.75],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 6 — LIFECYCLE
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 6 — Lifecycle and Finite Growth",
        body: "By default, cells divide forever. Real organisms don't — they \
               grow to a fixed size and stop, or they reproduce by shedding \
               offspring.\n\n\
               Max Splits in the Parent Settings panel limits how many times \
               a cell can divide. Set it to 2 and the cell divides twice, \
               then stops permanently. Set it to 0 and the cell never divides \
               at all — it's a terminal cell.\n\n\
               After Splits (further down in Parent Settings) lets you define \
               a second set of child routes that activate once the limit is \
               reached. The cell gets one final division using these alternate \
               routes. This is how you build reproduction: the normal splits \
               grow the body, and the after-splits division sheds a free egg \
               (Keep Adhesion off) that starts a new lifecycle.\n\n\
               Setting an after-splits child to a self-referential mode \
               is the same as leaving it at None.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    TutorialStepData {
        title: "Experiment — Limit a Cell's Growth",
        body: "Select any mode and set Max Splits to a small number — try 2 \
               or 3. Scrub the Time Slider.\n\n\
               The cells of that type now stop dividing after hitting the \
               limit. If it's a structural cell, the body stops growing at \
               a fixed size. If it's a terminal cell (sensor, swimmer), \
               set Max Splits to 0 so it never divides at all.\n\n\
               Now scroll down in Parent Settings to the After Splits section. \
               Try setting the after-splits Child A to a different mode with \
               Keep Adhesion off — that mode will be shed as a free egg when \
               the limit is reached.\n\n\
               Scrub the Time Slider and watch the egg detach and drift away. \
               If the egg's mode has adhesion on and the right cell type, it \
               will grow its own body independently.\n\n\
               Set Max Splits to any finite value (≥ 0) on any mode to continue.",
        gate_hint: "Set Max Splits to a finite value on any mode to continue.",
        gate:       StepGate::AnyFiniteMaxSplits,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 7 — SIGNALS
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 7 — The Signal System",
        body: "Signals are chemical messages that travel through the adhesion \
               network from cell to cell. They let one part of the body \
               influence another part's behaviour without any direct wiring.\n\n\
               There are 16 signal channels (0–15). Channels 0–7 are \
               oculocyte channels — only Oculocyte cells can emit on them, \
               but any cell can read them. Channels 8–15 are regulation \
               channels — any cell type can emit and read.\n\n\
               An Oculocyte fires a ray and emits a signal on its channel \
               when it detects something (food, other cells, light, barriers). \
               That signal propagates through the adhesion bonds to nearby \
               cells. A Flagellocyte reading that channel can speed up or \
               slow down based on the signal value.\n\n\
               Division and mode-switching can also be gated on signals: \
               a cell only divides when it receives the right chemical cue. \
               This is how complex developmental programs work.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.7],
    },

    TutorialStepData {
        title: "Experiment — Add a Sensor",
        body: "Try adding an Oculocyte to your genome.\n\n\
               Select any mode and set its Type to Oculocyte. In the \
               Oculocyte Settings section that appears, set Sense Type to \
               Food and Signal Channel to 0. Set Signal Value to 10 and \
               Signal Hops to 5.\n\n\
               Now select a Flagellocyte mode (or set another mode to \
               Flagellocyte). In its settings, enable Use Signal, set \
               Signal Channel to 0, Speed A to 0.1 (slow), and Speed B \
               to 1.0 (fast). Speed B activates when the signal is above \
               the threshold.\n\n\
               Scrub the Time Slider. When the Oculocyte detects food, it \
               emits a signal that travels through the adhesion bonds to the \
               Flagellocyte, which speeds up. The creature accelerates toward \
               food automatically.\n\n\
               This step is optional — click Next to skip if you want to \
               explore signals on your own later.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    // ════════════════════════════════════════════════════════════════════════
    // CHAPTER 8 — GO LIVE
    // ════════════════════════════════════════════════════════════════════════

    TutorialStepData {
        title: "Chapter 8 — Release Into the World",
        body: "The Preview editor is a sandbox — one organism, no competition, \
               no fluid dynamics. The Live Simulation is the real world.\n\n\
               When you're ready, click Live Simulation in the top bar or the \
               Scene Manager panel. The full GPU world loads: up to 200,000 \
               cells, fluid simulation, cave systems, light fields, and \
               other organisms competing for the same nutrients.\n\n\
               In the live world, hold Alt to open the radial tool menu. \
               Select the + Insert tool, then click anywhere in the viewport \
               to place your creature. It will start growing immediately.\n\n\
               You can switch back to Preview at any time to edit the genome. \
               Changes take effect the next time you insert a new organism — \
               existing cells in the world keep their old genome.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::SceneManagerPanel,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "What to Explore Next",
        body: "You've seen all the core mechanics. Here are some directions \
               to explore on your own:\n\n\
               Vasculocyte networks — add Vasculocyte modes with Outlet \
               enabled to build nutrient highways through large bodies. \
               Without them, nutrients can't reach distant cells.\n\n\
               Signal-gated division — in Parent Settings, set Division \
               Signal Channel to a regulation channel (8–15) and a threshold. \
               The cell only divides when it receives that signal. Use this \
               to synchronise growth across the body.\n\n\
               Glueocyte — bonds to other organisms or cave walls on contact. \
               Combine with a Devorocyte (steals nutrients from bonded cells) \
               for a predatory organism.\n\n\
               Myocyte — a contractile cell that shortens its adhesion bonds \
               on a timer or signal. Pairs of Myocytes on opposite sides of \
               a body create bending waves — peristaltic locomotion.\n\n\
               🎲 Procedural button — generates a random creature. Study its \
               genome to see how complex body plans are structured.",
        gate_hint: "",
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
    pub active: bool,
    pub current_step: usize,
    pub ever_shown: bool,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self { active: false, current_step: 0, ever_shown: false }
    }
}

impl TutorialState {
    pub fn total_steps() -> usize { TUTORIAL_STEPS.len() }

    pub fn current(&self) -> &TutorialStepData {
        &TUTORIAL_STEPS[self.current_step.min(TUTORIAL_STEPS.len() - 1)]
    }

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

const TEAL:        egui::Color32 = egui::Color32::from_rgb(0, 220, 175);
const TEAL_DIM:    egui::Color32 = egui::Color32::from_rgb(0, 100, 80);
const PANEL_GLOW:  egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 0, 0, 0);
const GATE_LOCKED: egui::Color32 = egui::Color32::from_rgb(220, 160, 40);
const GATE_OPEN:   egui::Color32 = egui::Color32::from_rgb(80, 220, 120);
const LINE_W:      f32 = 1.5;
const DOT_R:       f32 = 3.5;
const ARROW_LEN:   f32 = 11.0;
const ARROW_HALF:  f32 = 5.0;

// ─────────────────────────────────────────────────────────────────────────────
// Public render entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Render the tutorial overlay.  Call once per frame after the dock area
/// renders (so panel rects are populated) and before `ctx.end_pass()`.
pub fn render_tutorial(
    ctx:           &egui::Context,
    state:         &mut TutorialState,
    panel_rects:   &HashMap<String, egui::Rect>,
    genome:        &crate::genome::Genome,
    selected_mode: usize,
) {
    if !state.active { return; }

    let step_index = state.current_step;
    let total      = TutorialState::total_steps();

    let (gate_ok, has_hint, target_key, target_pos, target_is_element,
         step_title, step_body, step_gate_hint) = {
        let step = state.current();
        (
            step.gate.is_satisfied(genome, selected_mode),
            !step.gate_hint.is_empty(),
            step.target.panel_key(),
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
            ui.label(
                egui::RichText::new(step_title)
                    .strong()
                    .size(15.0)
                    .color(TEAL),
            );
            ui.add_space(5.0);
            ui.separator();
            ui.add_space(5.0);

            ui.label(egui::RichText::new(step_body).size(13.0));
            ui.add_space(10.0);

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

            let progress = (step_index + 1) as f32 / total as f32;
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_height(3.0)
                    .fill(TEAL),
            );
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("{} / {}", step_index + 1, total))
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(egui::RichText::new("✕ Close").size(12.0)).clicked() {
                        close_clicked = true;
                    }
                    ui.add_space(4.0);

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

                    if step_index > 0
                        && ui.button(egui::RichText::new("‹ Back").size(12.0)).clicked()
                    {
                        prev_clicked = true;
                    }
                });
            });
        });

    if let Some(ref r) = win_response {
        dialogue_rect = Some(r.response.rect);
    }

    if close_clicked { state.close(); return; }
    if next_clicked  { state.next(); }
    else if prev_clicked { state.prev(); }

    // ── Schematic pointer line ───────────────────────────────────────────────
    if let (Some(d_rect), Some(ref key)) = (dialogue_rect, target_key) {
        if let Some(&p_rect) = panel_rects.get(key.as_str()) {
            let tip = if target_is_element {
                let tp = target_pos;
                egui::pos2(
                    p_rect.left() + p_rect.width()  * tp[0],
                    p_rect.top()  + p_rect.height() * tp[1],
                )
            } else {
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
// Drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn draw_schematic_pointer(
    painter: &egui::Painter,
    d_rect:  egui::Rect,
    tip:     egui::Pos2,
    ctx:     &egui::Context,
    gate_ok: bool,
) {
    let line_color = if gate_ok { TEAL } else { GATE_LOCKED };
    let dim_color  = if gate_ok { TEAL_DIM } else { egui::Color32::from_rgb(100, 70, 0) };

    let start = nearest_edge_midpoint(d_rect, tip);
    let end   = tip;

    if (end - start).length() < 10.0 { return; }

    let dx = (end.x - start.x).abs();
    let dy = (end.y - start.y).abs();
    let corner = if dx >= dy {
        egui::Pos2::new(end.x, start.y)
    } else {
        egui::Pos2::new(start.x, end.y)
    };

    let stroke     = egui::Stroke::new(LINE_W, line_color);
    let dim_stroke = egui::Stroke::new(LINE_W * 0.5, dim_color);

    painter.line_segment([start, corner], stroke);

    let dir2 = (end - corner).normalized();
    let pre  = end - dir2 * ARROW_LEN;
    if (pre - corner).length() > 1.0 {
        painter.line_segment([corner, pre], stroke);
    }

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

    draw_arrowhead(painter, end, dir2, line_color);
    painter.circle_filled(corner, DOT_R, line_color);
    painter.circle_stroke(start, 4.0, egui::Stroke::new(1.5, line_color));

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

fn nearest_edge_midpoint(rect: egui::Rect, toward: egui::Pos2) -> egui::Pos2 {
    let c  = rect.center();
    let dx = toward.x - c.x;
    let dy = toward.y - c.y;

    if dx.abs() >= dy.abs() {
        egui::pos2(if dx >= 0.0 { rect.right() } else { rect.left() }, c.y)
    } else {
        egui::pos2(c.x, if dy >= 0.0 { rect.bottom() } else { rect.top() })
    }
}

fn draw_arrowhead(
    painter: &egui::Painter,
    tip:     egui::Pos2,
    dir:     egui::Vec2,
    color:   egui::Color32,
) {
    let perp = egui::Vec2::new(-dir.y, dir.x);
    let base = tip - dir * ARROW_LEN;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, base + perp * ARROW_HALF, base - perp * ARROW_HALF],
        color,
        egui::Stroke::NONE,
    ));
}
