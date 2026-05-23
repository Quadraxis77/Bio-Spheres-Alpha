//! In-game tutorial system for Bio-Spheres.
//!
//! Walks the player step-by-step through building a 3-cell-type organism
//! (Photocyte → Flagellocyte → Phagocyte) in the Genome Editor.
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
    /// `genome.modes[mode_idx].mode_a_after_splits` and `mode_b_after_splits`
    /// must equal `a_target` and `b_target` respectively.
    AfterSplitsRouting { mode_idx: usize, a_target: i32, b_target: i32 },
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

            StepGate::AfterSplitsRouting { mode_idx, a_target, b_target } => genome
                .modes
                .get(*mode_idx)
                .map(|m| {
                    m.mode_a_after_splits == *a_target
                        && m.mode_b_after_splits == *b_target
                })
                .unwrap_or(false),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step data
// ─────────────────────────────────────────────────────────────────────────────

/// Which panel a tutorial step wants to point at.
pub enum TutorialTarget {
    ModesPanel,
    NameTypePanel,
    ParentSettingsPanel,
    AdhesionSettingsPanel,
    ChildRotationPanel,
    TimeSliderPanel,
    SceneManagerPanel,
    None,
}

impl TutorialTarget {
    /// The key used in `panel_rects` to look up this panel's screen rect.
    pub fn panel_key(&self) -> Option<&'static str> {
        match self {
            TutorialTarget::ModesPanel          => Some("Modes"),
            TutorialTarget::NameTypePanel       => Some("NameTypeEditor"),
            TutorialTarget::ParentSettingsPanel => Some("ParentSettings"),
            TutorialTarget::AdhesionSettingsPanel => Some("AdhesionSettings"),
            TutorialTarget::ChildRotationPanel  => Some("QuaternionBall"),
            TutorialTarget::TimeSliderPanel     => Some("TimeSlider"),
            TutorialTarget::SceneManagerPanel   => Some("SceneManager"),
            TutorialTarget::None                => None,
        }
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Tutorial content  —  "Embryo, Light, Swim" (3-cell creature with lifecycle)
//
// Creature design:
//   M1 = Embryocyte  — stem cell, splits 3 times producing more Embryocytes,
//                      then differentiates: Child A → M2, Child B → M3.
//                      Normal children: both M1.  max_splits = 3.
//   M2 = Photocyte   — terminal energy cell.  max_splits = 0 (never divides).
//   M3 = Flagellocyte — terminal swimmer.      max_splits = 0 (never divides).
//
// Lifecycle:
//   1 Embryocyte → splits 3× → ~8 embryocytes (adhered cluster)
//   Each embryocyte's final split → Photocyte (A) + Flagellocyte (B)
//   End state: ~8 Photocytes + ~8 Flagellocytes, connected body, no further growth.
//
// This teaches the Max Splits / After-Splits routing system — the key tool
// for giving creatures a defined body plan rather than endless growth.
// ─────────────────────────────────────────────────────────────────────────────

pub const TUTORIAL_STEPS: &[TutorialStepData] = &[
    // ── 0 ── Welcome ─────────────────────────────────────────────────────────
    TutorialStepData {
        title: "Welcome to Bio-Spheres!",
        body:  "In this tutorial you'll build a real living creature from scratch.\n\n\
                Your creature will start as a single cell, grow into a small body, \
                and then stop growing — just like a real animal does. It will have \
                three different cell types:\n\n\
                • M1 — a stem cell that builds the body\n\
                • M2 — an energy cell that soaks up light\n\
                • M3 — a swimming cell that moves the creature around\n\n\
                Follow the steps one at a time. Each step explains what you're \
                doing and why. You can't move on until you've done the action — \
                so take your time and read everything!",
        gate_hint: "",
        gate:   StepGate::None,
        target: TutorialTarget::None,
    },

    // ── 1 ── Select M1 ───────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 1 — Open Mode 1",
        body:  "Look at the Modes panel on the left. It shows all 80 cell \
                blueprints in your creature's genome — think of each one as a \
                recipe for a different type of cell.\n\n\
                M1 is already selected by default, so all the other panels on \
                screen are already showing M1's settings. Go ahead and confirm \
                it's highlighted, then click Next to continue.",
        gate_hint: "Click M1 in the Modes panel to continue.",
        gate:   StepGate::ModeSelected(0),
        target: TutorialTarget::ModesPanel,
    },

    // ── 2 ── Modes panel controls ────────────────────────────────────────────
    TutorialStepData {
        title: "Step 2 — The Modes Panel Controls",
        body:  "Before we start editing, take a look at the Modes panel. \
                There are a few controls worth knowing about.\n\n\
                The small dot (radio button) to the left of each mode is the \
                Initial Mode marker. Whichever mode has this dot lit up is the \
                cell your creature starts as — it's the very first cell when the \
                simulation begins. Right now it's set to M1, which is exactly \
                what we want for our stem cell.\n\n\
                At the top of the panel you'll also see two small buttons:\n\
                • Copy Into — copies all the settings from the currently \
                  selected mode into whichever mode you click next. Useful \
                  when you want two similar cell types.\n\
                • ⟲ Reset — wipes the selected mode back to a completely \
                  blank default. Handy if you make a mistake and want to \
                  start that mode over.\n\n\
                You don't need to use any of these right now — just keep them \
                in mind as you build.",
        gate_hint: "",
        gate:   StepGate::None,
        target: TutorialTarget::ModesPanel,
    },

    // ── 3 ── M1 cell type ────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 3 — Make M1 a Stem Cell",
        body:  "Every mode has a Type — this decides what the cell actually does \
                when it's alive in the simulation.\n\n\
                In the Name & Type panel, open the Type dropdown and choose \
                Embryocyte.\n\n\
                An Embryocyte is a pure stem cell. It doesn't eat, swim, or do \
                anything fancy — it just divides over and over to build up the \
                body. Once it's done its job it will turn into the useful cells \
                that do the real work. We'll set that up in a moment.",
        gate_hint: "Set M1's Type to Embryocyte to continue.",
        gate:   StepGate::CellTypeSet { mode_idx: 0, expected: 10 },
        target: TutorialTarget::NameTypePanel,
    },

    // ── 3 ── M1 adhesion ─────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 4 — Stick the Cells Together",
        body:  "By default, when a cell divides its two children just float away \
                from each other. That's fine for single-celled creatures, but we \
                want a body — cells that stay connected.\n\n\
                Tick the Make Adhesion checkbox in the Name & Type panel.\n\n\
                This tells the cell to glue itself to the cell it was born from. \
                With adhesion on, every division adds another cell to the growing \
                cluster instead of releasing it into the world.",
        gate_hint: "Tick Make Adhesion on M1 to continue.",
        gate:   StepGate::AdhesionEnabled(0),
        target: TutorialTarget::NameTypePanel,
    },

    // ── 4 ── Two rotation panels explained ────────────────────────────���─────
    TutorialStepData {
        title: "Step 5 — Two Rotation Panels, Two Different Things",
        body:  "You'll notice there are two separate rotation panels in the \
                genome editor. They look similar but control completely different \
                things, so it's worth knowing the difference before you need them.\n\n\
                The Circle Sliders panel controls the parent split direction — \
                the angle that the parent cell itself is facing when it divides. \
                Think of it as deciding which way the cell is \"pointing\" at the \
                moment it splits in two. Adjusting pitch and yaw here lets you \
                control the overall orientation of the division.\n\n\
                The Child Rotation panel (the two 3D balls) controls the \
                orientation of each daughter cell after the split. Child A and \
                Child B each get their own ball — drag them to set the direction \
                each newborn cell will be facing when it appears. This panel also \
                has dropdowns to set which Mode each child will be.\n\n\
                In short: Circle Sliders = how the parent is aimed when it \
                splits. Child Rotation = where each child ends up pointing and \
                what type it becomes.",
        gate_hint: "",
        gate:   StepGate::None,
        target: TutorialTarget::ChildRotationPanel,
    },

    // ── 5 ── M1 max splits ───────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 6 — Limit How Many Times M1 Divides",
        body:  "Without a limit, stem cells divide forever and the creature just \
                keeps growing with no end. Real creatures don't do that — they \
                grow to a set size and stop.\n\n\
                Open the Parent Settings panel and drag the Max Splits slider to 3.\n\n\
                This means each Embryocyte is only allowed to divide 3 times. \
                After the third split it reaches its limit and something special \
                happens — you'll set that up in the next step.",
        gate_hint: "Set Max Splits to 3 in the Parent Settings panel to continue.",
        gate:   StepGate::MaxSplitsSet { mode_idx: 0, expected: 3 },
        target: TutorialTarget::ParentSettingsPanel,
    },

    // ── 5 ── M1 after-splits routing ─────────────────────────────────────────
    TutorialStepData {
        title: "Step 7 — Tell M1 What to Become",
        body:  "Now that Max Splits is set, two new dropdowns have appeared at \
                the bottom of the Parent Settings panel: the After-Splits children.\n\n\
                These decide what the cell turns into on its very last division. \
                Set Child A → M2 and Child B → M3.\n\n\
                So the full story for M1 is: divide 3 times, building up the \
                cluster — then on the final split, produce one energy cell (M2) \
                and one swimming cell (M3). The embryo has done its job and hands \
                off to the mature cells.",
        gate_hint: "Set After-Splits Child A → M2 and Child B → M3 to continue.",
        gate:   StepGate::AfterSplitsRouting { mode_idx: 0, a_target: 1, b_target: 2 },
        target: TutorialTarget::ParentSettingsPanel,
    },

    // ── 6 ── Select M2 ───────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 8 — Open Mode 2",
        body:  "Great work — M1 is fully set up! Now let's define the mature \
                cells that M1 will turn into.\n\n\
                Click M2 in the Modes panel. The editing panels will switch over \
                to M2's settings, which are all blank defaults right now.",
        gate_hint: "Click M2 in the Modes panel to continue.",
        gate:   StepGate::ModeSelected(1),
        target: TutorialTarget::ModesPanel,
    },

    // ── 7 ── M2 cell type ────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 9 — Make M2 an Energy Cell",
        body:  "Open the Type dropdown in the Name & Type panel and choose \
                Photocyte.\n\n\
                A Photocyte soaks up light from the environment and turns it into \
                nutrients that feed the whole organism. Think of it like a leaf — \
                it just sits there quietly doing photosynthesis, keeping the \
                creature alive even when there's no food nearby.\n\n\
                This will be one of the main workers in your mature creature.",
        gate_hint: "Set M2's Type to Photocyte to continue.",
        gate:   StepGate::CellTypeSet { mode_idx: 1, expected: 3 },
        target: TutorialTarget::NameTypePanel,
    },

    // ── 8 ── M2 adhesion ─────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 10 — Keep the Energy Cell in the Body",
        body:  "Just like with M1, we need to turn on adhesion for M2.\n\n\
                Tick the Make Adhesion checkbox for M2.\n\n\
                When an Embryocyte finishes its last split and produces a \
                Photocyte, that Photocyte needs to stay attached to the rest of \
                the body. If it floats away, its energy goes with it and the \
                other cells stop benefiting from it.",
        gate_hint: "Tick Make Adhesion on M2 to continue.",
        gate:   StepGate::AdhesionEnabled(1),
        target: TutorialTarget::NameTypePanel,
    },

    // ── 9 ── M2 max splits = 0 ───────────────────────────────────────────────
    TutorialStepData {
        title: "Step 11 — Make M2 a Dead-End Cell",
        body:  "The Photocyte's only job is to harvest light — it should never \
                divide and add more cells to the body.\n\n\
                In the Parent Settings panel, set Max Splits to 0.\n\n\
                Zero means this cell is never allowed to divide at all. It will \
                just sit quietly in the body for the rest of the simulation, \
                doing its job. This is how you create a stable, mature cell type \
                — give it useful behaviour and stop it from growing.",
        gate_hint: "Set Max Splits to 0 on M2 in the Parent Settings panel to continue.",
        gate:   StepGate::MaxSplitsSet { mode_idx: 1, expected: 0 },
        target: TutorialTarget::ParentSettingsPanel,
    },

    // ── 10 ── Select M3 ──────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 12 — Open Mode 3",
        body:  "One more cell type to go — the swimmer that will actually move \
                your creature through the world.\n\n\
                Click M3 in the Modes panel.",
        gate_hint: "Click M3 in the Modes panel to continue.",
        gate:   StepGate::ModeSelected(2),
        target: TutorialTarget::ModesPanel,
    },

    // ── 11 ── M3 cell type ───────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 13 — Give M3 a Flagella",
        body:  "Open the Type dropdown and choose Flagellocyte.\n\n\
                A Flagellocyte has a long whip-like tail called a flagellum. It \
                beats the flagellum to push the cell — and anything attached to \
                it — through the fluid. The more Flagellocytes your creature has, \
                the faster and more powerfully it can move.\n\n\
                Because every Embryocyte produces one Flagellocyte when it \
                finishes, your creature will end up with a good spread of \
                swimmers across the body.",
        gate_hint: "Set M3's Type to Flagellocyte to continue.",
        gate:   StepGate::CellTypeSet { mode_idx: 2, expected: 1 },
        target: TutorialTarget::NameTypePanel,
    },

    // ── 12 ── M3 adhesion ────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 14 — Attach the Swimmer to the Body",
        body:  "Turn on Make Adhesion for M3.\n\n\
                A Flagellocyte that isn't attached to anything will just swim off \
                on its own — it has no idea there's a creature it's supposed to \
                be part of. With adhesion on, its push is applied directly to the \
                cluster and the whole organism moves.",
        gate_hint: "Tick Make Adhesion on M3 to continue.",
        gate:   StepGate::AdhesionEnabled(2),
        target: TutorialTarget::NameTypePanel,
    },

    // ── 13 ── M3 max splits = 0 ──────────────────────────────────────────────
    TutorialStepData {
        title: "Step 15 — Make M3 a Dead-End Cell Too",
        body:  "Set Max Splits to 0 for M3 in the Parent Settings panel.\n\n\
                Same idea as the Photocyte — the Flagellocyte's job is to swim, \
                not to divide. Setting Max Splits to 0 locks it in place as a \
                permanent part of the mature body.\n\n\
                With this final step, your creature now has a complete lifecycle:\n\
                → M1 builds the body by dividing 3 times\n\
                → On its last split, M1 hands off to M2 and M3\n\
                → M2 and M3 never divide — they just work\n\
                The creature will grow, mature, and then hold its shape forever.",
        gate_hint: "Set Max Splits to 0 on M3 in the Parent Settings panel to continue.",
        gate:   StepGate::MaxSplitsSet { mode_idx: 2, expected: 0 },
        target: TutorialTarget::ParentSettingsPanel,
    },

    // ── 14 ── Preview ────────────────────────────────────────────────────────
    TutorialStepData {
        title: "Step 16 — Watch It Come to Life",
        body:  "Drag the Time Slider at the bottom of the screen to fast-forward \
                through time.\n\n\
                At first you'll see a small ball of Embryocytes dividing and \
                sticking together. As time passes they hit their split limit and \
                you'll start to see Photocytes and Flagellocytes appear. Eventually \
                the Embryocytes are all gone — replaced entirely by the mature \
                cell types — and the creature settles into its final shape.\n\n\
                The Flagellocytes will begin beating their tails and pushing the \
                whole cluster through the fluid.",
        gate_hint: "",
        gate:   StepGate::None,
        target: TutorialTarget::TimeSliderPanel,
    },

    // ── 15 ── Complete ───────────────────────────────────────────────────────
    TutorialStepData {
        title: "You Did It!",
        body:  "You've just built a living creature with a real lifecycle — \
                it grows, matures, and stops, all from just three cell types in \
                the genome.\n\n\
                Some ideas for what to try next:\n\
                • Raise M1's Max Splits to 4 or 5 to grow a larger body\n\
                • Add M4 as a Phagocyte (set Max Splits to 0) and route some of \
                  M1's after-splits children to it — now it eats food too\n\
                • Try tweaking the Adhesion Settings to make the body bouncier \
                  or more rigid\n\
                • When you're happy, click Live Simulation to release your \
                  creature into the full GPU world and watch it survive on its own",
        gate_hint: "",
        gate:   StepGate::None,
        target: TutorialTarget::None,
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
    let step        = state.current();
    let gate_ok     = step.gate.is_satisfied(genome, selected_mode);
    let has_hint    = !step.gate_hint.is_empty();
    let target_key  = step.target.panel_key();

    // ── Panel highlight ──────────────────────────────────────────────────────
    if let Some(key) = target_key {
        if let Some(&panel_rect) = panel_rects.get(key) {
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

    egui::Window::new("▶  Tutorial")
        .id(egui::Id::new("tutorial_dialog"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([400.0, 0.0])
        .frame(frame)
        .show(ctx, |ui| {
            dialogue_rect = Some(ui.max_rect().expand(16.0));

            // Title
            ui.label(
                egui::RichText::new(step.title)
                    .strong()
                    .size(15.0)
                    .color(TEAL),
            );
            ui.add_space(5.0);
            ui.separator();
            ui.add_space(5.0);

            // Body
            ui.label(egui::RichText::new(step.body).size(13.0));
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
                            egui::RichText::new(format!("⟳  {}", step.gate_hint))
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

    // ── Apply navigation ─────────────────────────────────────────────────────
    if close_clicked { state.close(); return; }
    if next_clicked  { state.next(); }
    else if prev_clicked { state.prev(); }

    // ── Schematic pointer line ───────────────────────────────────────────────
    if let (Some(d_rect), Some(key)) = (dialogue_rect, target_key) {
        if let Some(&p_rect) = panel_rects.get(key) {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("tut_pointer"),
            ));
            draw_schematic_pointer(&painter, d_rect, p_rect, ctx, gate_ok);
        }
    }

    ctx.request_repaint();
}

// ─────────────────────────────────────────────────────────────────────────────
// Schematic pointer drawing
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a high-tech schematic pointer from `d_rect` (dialogue) to `p_rect` (panel).
///
/// Colour shifts: amber when the gate is locked, teal when open/satisfied.
fn draw_schematic_pointer(
    painter:  &egui::Painter,
    d_rect:   egui::Rect,
    p_rect:   egui::Rect,
    ctx:      &egui::Context,
    gate_ok:  bool,
) {
    let line_color = if gate_ok { TEAL } else { GATE_LOCKED };
    let dim_color  = if gate_ok { TEAL_DIM } else { egui::Color32::from_rgb(100, 70, 0) };

    let start = closest_edge_point(d_rect, p_rect.center());
    let end   = closest_edge_point(p_rect, d_rect.center());

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

/// Return the edge point of `rect` that a ray from `rect`'s centre toward
/// `toward` would exit through.
fn closest_edge_point(rect: egui::Rect, toward: egui::Pos2) -> egui::Pos2 {
    let c  = rect.center();
    let dx = toward.x - c.x;
    let dy = toward.y - c.y;
    let hw = rect.width()  * 0.5;
    let hh = rect.height() * 0.5;

    if hw < 1.0 || hh < 1.0 || (dx == 0.0 && dy == 0.0) {
        return c;
    }

    let tx = if dx != 0.0 { hw / dx.abs() } else { f32::INFINITY };
    let ty = if dy != 0.0 { hh / dy.abs() } else { f32::INFINITY };

    if tx <= ty {
        let exit_x = if dx >= 0.0 { rect.right() } else { rect.left() };
        egui::Pos2::new(exit_x, (c.y + dy * tx).clamp(rect.top(), rect.bottom()))
    } else {
        let exit_y = if dy >= 0.0 { rect.bottom() } else { rect.top() };
        egui::Pos2::new((c.x + dx * ty).clamp(rect.left(), rect.right()), exit_y)
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
