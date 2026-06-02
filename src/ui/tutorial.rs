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

// -----------------------------------------------------------------------------
// Gate system
// -----------------------------------------------------------------------------

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
    ChildRouting {
        mode_idx: usize,
        a_target: i32,
        b_target: i32,
    },
}

impl StepGate {
    pub fn is_satisfied(&self, genome: &crate::genome::Genome, selected_mode: usize) -> bool {
        match self {
            StepGate::None => true,

            StepGate::ModeSelected(idx) => selected_mode == *idx,

            StepGate::CellTypeSet { mode_idx, expected } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.cell_type == *expected)
                .unwrap_or(false),

            StepGate::AnyCellTypeChanged => genome.modes.iter().any(|m| m.cell_type != 0),

            StepGate::AdhesionEnabled(mode_idx) => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.parent_make_adhesion)
                .unwrap_or(false),

            StepGate::AnyAdhesionEnabled => genome.modes.iter().any(|m| m.parent_make_adhesion),

            StepGate::AnyAdhesionDisabled => genome.modes.iter().any(|m| !m.parent_make_adhesion),

            StepGate::ChildBMode { mode_idx, target } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.child_b.mode_number == *target)
                .unwrap_or(false),

            StepGate::AnyChildBDifferentMode => genome
                .modes
                .iter()
                .any(|m| m.child_b.mode_number != m.child_a.mode_number),

            StepGate::MaxSplitsSet { mode_idx, expected } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.max_splits == *expected)
                .unwrap_or(false),

            StepGate::AnyFiniteMaxSplits => genome.modes.iter().any(|m| m.max_splits >= 0),

            StepGate::ChildAKeepAdhesionOff(mode_idx) => genome
                .modes
                .get(*mode_idx)
                .map(|m| !m.child_a.keep_adhesion)
                .unwrap_or(false),

            StepGate::AnySplitIntervalChanged => genome
                .modes
                .iter()
                .any(|m| (m.split_interval - 1.0).abs() > 0.05),

            StepGate::AnySplitMassChanged => genome
                .modes
                .iter()
                .any(|m| (m.split_mass - 1.5).abs() > 0.05),

            StepGate::ChildRouting {
                mode_idx,
                a_target,
                b_target,
            } => genome
                .modes
                .get(*mode_idx)
                .map(|m| m.child_a.mode_number == *a_target && m.child_b.mode_number == *b_target)
                .unwrap_or(false),
        }
    }
}

// -----------------------------------------------------------------------------
// Step data
// -----------------------------------------------------------------------------

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
            TutorialTarget::ModesPanel => Some("Modes".to_string()),
            TutorialTarget::NameTypePanel => Some("Modes".to_string()),
            TutorialTarget::ParentSettingsPanel => Some("ParentSettings".to_string()),
            TutorialTarget::AdhesionSettingsPanel => Some("AdhesionSettings".to_string()),
            TutorialTarget::ChildRotationPanel => Some("QuaternionBall".to_string()),
            TutorialTarget::CircleSlidersPanel => Some("CircleSliders".to_string()),
            TutorialTarget::TimeSliderPanel => Some("TimeSlider".to_string()),
            TutorialTarget::SceneManagerPanel => Some("SceneManager".to_string()),
            TutorialTarget::None => None,
            TutorialTarget::ModeRow(idx) => Some(format!("mode_row_{}", idx)),
            TutorialTarget::CellTypeDropdown => Some("cell_type_dropdown".to_string()),
            TutorialTarget::MakeAdhesionCheckbox => Some("make_adhesion_checkbox".to_string()),
            TutorialTarget::MaxSplitsSlider => Some("max_splits_slider".to_string()),
            TutorialTarget::AfterSplitsChildA => Some("after_splits_child_a".to_string()),
            TutorialTarget::AfterSplitsChildB => Some("after_splits_child_b".to_string()),
            TutorialTarget::ChildAKeepAdhesion => Some("child_a_keep_adhesion".to_string()),
            TutorialTarget::ChildBKeepAdhesion => Some("child_b_keep_adhesion".to_string()),
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

// -----------------------------------------------------------------------------
// Tutorial steps
//
// Structure: 8 chapters, each teaching one mechanic through observation.
// The player is never told to build a specific organism - they experiment
// with settings and watch what changes.  Gates confirm they touched the
// mechanic, not that they set it to a prescribed value.
//
// Chapter 1 - Orientation:    what you're looking at, how to observe
// Chapter 2 - Division:       split nutrients and interval, watching timing
// Chapter 3 - Cell types:     changing type, observing behavior
// Chapter 4 - Adhesion:       bonding on/off, bodies vs solo cells
// Chapter 5 - Body shape:     child routing and orientation
// Chapter 6 - Lifecycle:      max splits, finite growth, after-splits
// Chapter 7 - Signals:        brief intro to the signal system
// Chapter 8 - Go live:        releasing into the full world
// -----------------------------------------------------------------------------

pub const TUTORIAL_STEPS: &[TutorialStepData] = &[

    // 
    // CHAPTER 1 - ORIENTATION
    // 

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
        title: "The Initial Mode",
        body: "Each row in the Modes list has a small radio button on the left. \
               This sets the Initial Mode — the mode the first cell is born into \
               when the organism is spawned.\n\n\
               Everything grows from that one seed cell. If M1 is initial, the \
               organism starts as an M1 cell and divides according to M1's \
               settings. Switch the radio to M2 and the seed is M2 instead — a \
               completely different growth trajectory from the same genome.\n\n\
               For most genomes you'll leave this on M1. It matters when you \
               want a specific mode to act as the root — for example, a \
               stem-cell mode that grows the body before handing off to \
               specialised types.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ModeRow(0),
        target_pos: [0.0, 0.5],
    },

    TutorialStepData {
        title: "The Time Slider",
        body: "Drag the Time Slider to the right and watch the organism grow. \
               Drag it back to zero — everything rewinds. The same genome \
               always produces the same result.\n\n\
               Try scrubbing back and forth a few times. Notice that changes \
               you make to the genome take effect immediately when you \
               re-scrub. The slider is your main tool for seeing what a \
               change actually does.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::TimeSliderPanel,
        target_pos: [0.5, 0.5],
    },

    // 
    // CHAPTER 2 - DIVISION TIMING
    // 

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
        title: "Division Timing — Try It",
        body: "Pick any mode and move Split Interval or Split Nutrients, \
               then re-scrub. Higher interval means that cell type waits \
               longer between divisions. Higher split nutrients means it \
               needs more food first. Either one slows growth for that mode, \
               but the overall organism speed also depends on how many modes \
               are active, what cell types they are, and how much food is \
               available — so the effect you see will vary based on what \
               you've already set up.\n\n\
               Try pushing the sliders to their extremes and re-scrubbing \
               each time. The goal is to feel how these two numbers relate \
               to colony density, not to hit any specific target.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    // 
    // CHAPTER 3 - CELL TYPES
    // 

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
        title: "Cell Types — Try It",
        body: "Select any mode and change its Type. Re-scrub and observe \
               what's different. The effect depends on what else is in \
               the genome — a Flagellocyte in a body with adhesion will \
               push the whole structure, but without adhesion it just \
               propels itself away. A Phagocyte needs to be near food \
               to show anything useful.\n\n\
               Types interact with each other. What any one type does in \
               isolation may look very different once other modes, adhesion, \
               and routing are involved.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    // 
    // CHAPTER 4 - ADHESION
    // 

    TutorialStepData {
        title: "Chapter 4 — Adhesion",
        body: "Adhesion is what turns a collection of dividing cells into a \
               body. Without it, every division produces two cells that \
               immediately drift apart — you get a cloud of individuals, \
               not an organism.\n\n\
               The Make Adhesion checkbox on each mode controls whether that \
               mode creates a spring bond between its two children when it \
               divides. When it's off, siblings scatter freely. When it's on, \
               they stay connected.\n\n\
               The Adhesion Settings panel controls the physics of that \
               spring: stiffness, damping, break force, and twist constraints. \
               The defaults work well for most creatures.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::AdhesionSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    TutorialStepData {
        title: "Adhesion — Try It",
        body: "Toggle Make Adhesion on a mode and re-scrub each time. \
               Turning it on will bond that mode's children together on \
               each division — if the rest of the genome also has adhesion \
               on, this contributes to a connected body. Turning it off \
               means that mode's children scatter freely on every split, \
               regardless of what other modes do.\n\n\
               The overall structure you see is the combined result of \
               every mode's adhesion setting. A single mode with adhesion \
               off in an otherwise bonded genome will shed loose cells \
               from that point in the lineage.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::MakeAdhesionCheckbox,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Adhesion Zones — Blue Side and Green Side",
        body: "One thing to understand before zones make sense: when a cell \
               divides, the parent ceases to exist and is replaced by two \
               clone children. Child A is a clone that takes the parent's \
               place — it spawns at the parent's position and inherits the \
               parent's bonds on its side. Child B is cloned and pushed out \
               in the split direction.\n\n\
               If you route Child A back to M1 and Child B to M2, each \
               division replaces the M1 cell in place with a new M1 clone \
               and pushes a fresh M2 outward. The M1 position in the body \
               stays stable while M2 cells accumulate around it.\n\n\
               When a cell divides, the split is visualised as a ring around \
               the cell. The ring is half blue and half green:\n\n\
               Blue side = Child A (the in-place clone).\n\
               Green side = Child B (the outward clone).\n\n\
               Every existing bond the parent had is classified by which side \
               of that ring it sits on:\n\n\
               Zone B (blue side) — inherited by Child A.\n\
               Zone A (green side) — inherited by Child B.\n\
               Zone C (equator) — inherited by both.\n\n\
               Child A naturally inherits the bonds on the blue side — the \
               ones pointing away from where Child B is being pushed.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Keep Adhesion and What It Actually Means",
        body: "Each child has a Keep Adhesion checkbox. It controls whether \
               that child participates in zone inheritance at all.\n\n\
               Keep Adhesion ON — the child inherits the bonds in its zone \
               (blue side for Child A, green side for Child B).\n\n\
               Keep Adhesion OFF — the child is born completely free of all \
               inherited bonds, regardless of what bonds the parent had.\n\n\
               Crucially: Keep Adhesion only governs inherited bonds. It does \
               nothing to the sibling bond created by Make Adhesion — that bond \
               is always formed between the two new children when Make Adhesion \
               is on, independently of Keep Adhesion.\n\n\
               A common pattern: a stem cell (M1) has Make Adhesion ON and \
               routes Child B to a terminal cell (M2). M1 keeps dividing, \
               growing a chain. M2 has Make Adhesion OFF. When M2 divides, \
               its single bond to the chain sits entirely on one side of its \
               split ring, so one child inherits it and the other is born \
               loose. To keep M2 cells from detaching their offspring, either \
               enable Make Adhesion on M2 (to bond siblings together) or set \
               Max Splits to 0 so M2 never divides.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ChildAKeepAdhesion,
        target_pos: [0.5, 0.5],
    },

    // 
    // CHAPTER 5 - BODY SHAPE
    // 

    TutorialStepData {
        title: "Chapter 5 — Body Shape",
        body: "Unless you've already changed it, each mode routes both \
               children back to itself — every division produces two \
               identical cells. To build a body with different cell types \
               in different positions, route children to different modes.\n\n\
               The Child Rotation panel has two 3D balls: Child A and Child B. \
               Below each ball:\n\n\
               Mode — which blueprint this child uses after the split. \
               Change Child B's Mode to M2 and every division produces one \
               M1 cell and one M2 cell instead of two M1 cells.\n\n\
               Remember: Child A is a clone that replaces the parent in place — \
               it spawns where the parent was and inherits its bonds on that \
               side. Child B is cloned and pushed outward. So routing Child A \
               back to M1 means a new M1 clone takes the parent's spot each \
               division. Routing Child B to M2 means a fresh M2 is pushed \
               outward each time.\n\n\
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
        title: "Body Shape — Try It",
        body: "Pick a mode and change Child B's Mode dropdown to something \
               different, then re-scrub. Each division of that mode will \
               now produce two different cell types. What you see depends \
               on the types involved, whether adhesion is on, and how far \
               into the simulation you scrub.\n\n\
               Try dragging the Child B orientation ball and re-scrubbing. \
               The born direction of Child B shifts — this affects physical \
               position in the body, not just facing angle. The same change \
               looks very different depending on whether other modes in the \
               lineage also have curved or offset splits.\n\n\
               The Circle Sliders panel rotates the split axis. A small \
               offset curves growth; a large one branches it. The cumulative \
               effect builds up over many generations.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ChildRotationPanel,
        target_pos: [0.75, 0.75],
    },

    // 
    // CHAPTER 6 - LIFECYCLE
    // 

    TutorialStepData {
        title: "Chapter 6 — Lifecycle and Finite Growth",
        body: "Unless Max Splits has been set, cells divide indefinitely. \
               Limiting splits lets you define a fixed growth budget — the \
               body reaches a size and stops.\n\n\
               Max Splits in the Parent Settings panel limits how many times \
               a cell can divide. Set it to 2 and the cell divides twice, \
               then stops permanently. Set it to 0 and the cell never divides \
               at all — it's a terminal cell.\n\n\
               After Splits (further down in Parent Settings) lets you define \
               a second set of child routes that activate once the limit is \
               reached. The cell gets one final division using these alternate \
               routes. This is one way to build reproduction: the normal splits \
               grow the body, and the after-splits division sheds a free egg \
               (Keep Adhesion off) that starts a new lifecycle.\n\n\
               But After Splits is not the only way to release offspring. \
               You have several options depending on your design:\n\n\
               Disable Make Adhesion on a mode — every division from that \
               mode scatters both children freely, so the mode continuously \
               releases offspring throughout its lifetime rather than only at \
               the end.\n\n\
               Disable Keep Adhesion on Child B — Child B will not inherit \
               any of the parent's existing bonds. However, this only works \
               as a detach if Make Adhesion is also off. If Make Adhesion is \
               on, a fresh sibling bond is always created between Child A and \
               Child B at the moment of division, regardless of Keep Adhesion \
               — so Child B ends up tethered to Child A anyway. Keep Adhesion \
               controls inherited bonds only; Make Adhesion controls the new \
               sibling bond. To truly shed Child B free you need Make Adhesion \
               off, or Keep Adhesion off with Make Adhesion also off.\n\n\
               After Splits is best when you want the body to grow first and \
               reproduce only once at the end. Make/Keep Adhesion off is best \
               when you want continuous or early shedding.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    TutorialStepData {
        title: "Lifecycle — Try It",
        body: "Set Max Splits on any mode and re-scrub. That mode stops \
               dividing once the count is reached. The exact effect on \
               the organism depends on which modes are still dividing — \
               limiting one mode while others continue can stop one part \
               of the body growing while the rest carries on.\n\n\
               Scroll to After Splits and change a child route there. \
               That route only fires on the final division of that mode. \
               Whether the child detaches depends on its mode's adhesion \
               settings and whether Keep Adhesion is on — the after-splits \
               route follows the same inheritance rules as any other split.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.3],
    },

    // 
    // CHAPTER 7 - SIGNALS
    // 

    TutorialStepData {
        title: "Chapter 7 — Signals (Optional)",
        body: "The signal system is the most complex part of Bio-Spheres. \
               You do not need it to build a functional organism — many \
               strong creatures use no signals at all. Skip this chapter \
               if you want to go live first and come back to it later.\n\n\
               Signals are messages that travel through adhesion bonds from \
               cell to cell. They let one part of the body influence another \
               part's behaviour without any direct wiring — a sensor at the \
               front can change the speed of a swimmer at the back.\n\n\
               There are 16 channels (0–15). Channels 0–7 are sensory \
               channels — emitted and read by sensory cell types such as \
               oculocytes, photocytes, lipocytes, cognocytes, and \
               memorocytes. Locomotion cells (flagellocyte, ciliocyte, \
               myocyte) also read from these. \
               Channels 8–15 are developmental/regulation channels — any \
               cell type can emit and read them, and they can gate \
               division, apoptosis, and mode switching. A signal is just \
               a number; what it does depends on what is listening.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.7],
    },

    TutorialStepData {
        title: "Two Ways to Emit",
        body: "There are two kinds of emitter, and they work very differently.\n\n\
               Sensory (event-driven) — sensory cell types such as oculocytes, \
               photocytes, and lipocytes emit on channels 0–7 only when they \
               detect a target or cross an internal threshold. The signal is \
               absent between events, so receivers can distinguish \
               'condition met' from 'no condition'. Many cell types share \
               this range — an oculocyte spots food, a photocyte reports \
               light level, a lipocyte reports fat reserves, all on the \
               same set of channels.\n\n\
               Regulation Emit (continuous) — any mode can turn on a constant \
               broadcast on any developmental channel (8–15). Every frame that \
               cell is alive it writes its configured value onto the channel. \
               There is no on/off edge — the signal is simply always present \
               while the cell exists. Use this for positional identity, \
               maturity markers, or body-wide pressure signals.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "How Signals Travel — Summation",
        body: "Each propagation step, a cell looks at every bonded neighbor \
               and SUMS the attenuated contributions it can receive on that \
               channel. Hops is the maximum across all contributors.\n\n\
               This means signal strength reflects how many sources are \
               nearby, not just whether any single source is strong enough. \
               Two cells each emitting value 10 combine to produce value ~20 \
               at a shared neighbor.\n\n\
               Signal attenuates 50% per bond-hop beyond the emitter, so \
               distant sources contribute less than close ones. The combined \
               value is clamped at 2047 (the 11-bit maximum).\n\n\
               Practical implication: set a threshold higher than any single \
               emitter's value and the receiver only fires when a cluster of \
               emitters surrounds it — quorum sensing.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::AdhesionSettingsPanel,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Seeing Signals Propagate",
        body: "When signals are active, the adhesion bonds between cells \
               light up as the signal passes through them. This is the \
               easiest way to confirm a signal is actually reaching its \
               destination.\n\n\
               If your organism is tightly packed and the bonds are hard \
               to see, use the Adhesion Expansion button in the viewport \
               toolbar. It stretches all bonds to their maximum length so \
               the network is visible and spread out. The signal lighting \
               still works in this mode.\n\n\
               If a bond is not lighting up, the signal is not reaching \
               that cell — either the hops ran out, the channel numbers \
               don't match, or there is no adhesion path between the \
               emitter and receiver.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::AdhesionSettingsPanel,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Receiving and Reacting",
        body: "Any cell that reads a signal channel compares the incoming \
               value against a threshold and changes its behaviour \
               accordingly. A Flagellocyte can switch between two speeds \
               depending on whether the signal is above or below the \
               threshold. Division and mode-switching can be gated the \
               same way — a cell only divides when it receives the cue.\n\n\
               The receiver and emitter must share the same channel number \
               and be connected by an unbroken chain of adhesion bonds \
               within the hops limit. If any of those three conditions \
               isn't met, the receiver sees nothing regardless of what \
               the emitter does.\n\n\
               Invert threshold — check 'Invert' to flip the logic: the \
               cell reacts when the signal DROPS below the threshold. Use \
               this for absence gating — a cell that goes dormant when its \
               support signal disappears.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::CellTypeDropdown,
        target_pos: [0.5, 0.5],
    },

    TutorialStepData {
        title: "Mode Switch (No Division)",
        body: "Under Signal Conditions, 'Mode Switch (No Division)' changes a \
               cell's mode in-place without dividing. Pick a trigger channel \
               (8–15), a threshold, and a target mode. When the signal on \
               that channel exceeds the threshold the cell immediately \
               becomes the target mode — new cell type, new split settings, \
               everything.\n\n\
               Important channel rule: do not use the same channel number \
               for both a mode's Regulation Emit and its Mode Switch trigger. \
               The cell would see its own emission and fire the switch every \
               frame. The engine blocks self-triggers from the same frame, \
               but the safest design is to use separate channels — emit on \
               Ch 8, switch on Ch 9, for example.\n\n\
               A common pattern: stem cells emit a 'maturity' signal on \
               Ch 8 after enough divisions; neighbouring cells watch Ch 8 \
               and switch to their terminal mode when they receive it.",
        gate_hint: "",
        gate:       StepGate::None,
        target:     TutorialTarget::ParentSettingsPanel,
        target_pos: [0.5, 0.5],
    },

    // 
    // CHAPTER 8 - GO LIVE
    // 

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
               Quorum sensing — because signals from multiple emitters SUM \
               at each receiver, you can set a threshold higher than any \
               single emitter's value. The gate only opens when enough \
               nearby cells are broadcasting — useful for triggering \
               differentiation only once a tissue reaches a critical mass.\n\n\
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

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

/// Persistent tutorial playback state, stored inside [`GlobalUiState`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TutorialState {
    pub active: bool,
    pub current_step: usize,
    pub ever_shown: bool,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self {
            active: false,
            current_step: 0,
            ever_shown: false,
        }
    }
}

impl TutorialState {
    pub fn total_steps() -> usize {
        TUTORIAL_STEPS.len()
    }

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
        if self.current_step > 0 {
            self.current_step -= 1;
        }
    }

    pub fn start(&mut self) {
        self.active = true;
        self.current_step = 0;
        self.ever_shown = true;
    }

    pub fn close(&mut self) {
        self.active = false;
    }
}

// -----------------------------------------------------------------------------
// Visual constants
// -----------------------------------------------------------------------------

const TEAL: egui::Color32 = egui::Color32::from_rgb(0, 220, 175);
const TEAL_DIM: egui::Color32 = egui::Color32::from_rgb(0, 100, 80);
const PANEL_GLOW: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 0, 0, 0);
const GATE_LOCKED: egui::Color32 = egui::Color32::from_rgb(220, 160, 40);
const GATE_OPEN: egui::Color32 = egui::Color32::from_rgb(80, 220, 120);
const LINE_W: f32 = 1.5;
const DOT_R: f32 = 3.5;
const ARROW_LEN: f32 = 11.0;
const ARROW_HALF: f32 = 5.0;

// -----------------------------------------------------------------------------
// Public render entry point
// -----------------------------------------------------------------------------

/// Render the tutorial overlay.  Call once per frame after the dock area
/// renders (so panel rects are populated) and before `ctx.end_pass()`.
pub fn render_tutorial(
    ctx: &egui::Context,
    state: &mut TutorialState,
    panel_rects: &HashMap<String, egui::Rect>,
    genome: &crate::genome::Genome,
    selected_mode: usize,
) {
    if !state.active {
        return;
    }

    let step_index = state.current_step;
    let total = TutorialState::total_steps();

    let (
        gate_ok,
        has_hint,
        target_key,
        target_pos,
        target_is_element,
        step_title,
        step_body,
        step_gate_hint,
    ) = {
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

    // -- Panel highlight ------------------------------------------------------
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

    // -- Dialogue window ------------------------------------------------------
    let mut dialogue_rect: Option<egui::Rect> = None;
    let mut next_clicked = false;
    let mut prev_clicked = false;
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
                    if ui
                        .button(egui::RichText::new("✕ Close").size(12.0))
                        .clicked()
                    {
                        close_clicked = true;
                    }
                    ui.add_space(4.0);

                    let is_last = step_index + 1 >= total;
                    let next_text = if is_last {
                        "  Finish ✓  "
                    } else {
                        "  Next ›  "
                    };
                    let next_btn = egui::Button::new(
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
                        && ui
                            .button(egui::RichText::new("‹ Back").size(12.0))
                            .clicked()
                    {
                        prev_clicked = true;
                    }
                });
            });
        });

    if let Some(ref r) = win_response {
        dialogue_rect = Some(r.response.rect);
    }

    if close_clicked {
        state.close();
        return;
    }
    if next_clicked {
        state.next();
    } else if prev_clicked {
        state.prev();
    }

    // -- Schematic pointer line -----------------------------------------------
    if let (Some(d_rect), Some(ref key)) = (dialogue_rect, target_key) {
        if let Some(&p_rect) = panel_rects.get(key.as_str()) {
            let tip = if target_is_element {
                let tp = target_pos;
                egui::pos2(
                    p_rect.left() + p_rect.width() * tp[0],
                    p_rect.top() + p_rect.height() * tp[1],
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

// -----------------------------------------------------------------------------
// Drawing helpers
// -----------------------------------------------------------------------------

fn draw_schematic_pointer(
    painter: &egui::Painter,
    d_rect: egui::Rect,
    tip: egui::Pos2,
    ctx: &egui::Context,
    gate_ok: bool,
) {
    let line_color = if gate_ok { TEAL } else { GATE_LOCKED };
    let dim_color = if gate_ok {
        TEAL_DIM
    } else {
        egui::Color32::from_rgb(100, 70, 0)
    };

    let start = nearest_edge_midpoint(d_rect, tip);
    let end = tip;

    if (end - start).length() < 10.0 {
        return;
    }

    let dx = (end.x - start.x).abs();
    let dy = (end.y - start.y).abs();
    let corner = if dx >= dy {
        egui::Pos2::new(end.x, start.y)
    } else {
        egui::Pos2::new(start.x, end.y)
    };

    let stroke = egui::Stroke::new(LINE_W, line_color);
    let dim_stroke = egui::Stroke::new(LINE_W * 0.5, dim_color);

    painter.line_segment([start, corner], stroke);

    let dir2 = (end - corner).normalized();
    let pre = end - dir2 * ARROW_LEN;
    if (pre - corner).length() > 1.0 {
        painter.line_segment([corner, pre], stroke);
    }

    let seg2_len = (pre - corner).length();
    if seg2_len > 40.0 {
        let count = ((seg2_len / 20.0) as usize).max(1).min(14);
        let perp = egui::Vec2::new(-dir2.y, dir2.x);
        for i in 1..=count {
            let t = i as f32 / (count + 1) as f32;
            let c = corner + (pre - corner) * t;
            painter.line_segment([c - perp * 4.5, c + perp * 4.5], dim_stroke);
        }
    }

    draw_arrowhead(painter, end, dir2, line_color);
    painter.circle_filled(corner, DOT_R, line_color);
    painter.circle_stroke(start, 4.0, egui::Stroke::new(1.5, line_color));

    let t = ctx.input(|i| i.time) as f32;
    let period = 1.8_f32;
    let phase = (t % period) / period;

    let total_path = (corner - start).length() + (end - corner).length();
    let traveled = phase * total_path;
    let seg1_len = (corner - start).length();

    let scan_pos = if traveled <= seg1_len || seg1_len < 1.0 {
        let dir1 = if seg1_len > 0.001 {
            (corner - start).normalized()
        } else {
            dir2
        };
        start + dir1 * traveled.min(seg1_len)
    } else {
        let rem = traveled - seg1_len;
        corner + dir2 * rem.min((end - corner).length())
    };

    let alpha = ((t * 4.0).sin() * 0.5 + 0.5) * 200.0 + 55.0;
    let (sr, sg, sb) = if gate_ok {
        (0, 220, 175)
    } else {
        (220, 160, 40)
    };
    let scan_col = egui::Color32::from_rgba_premultiplied(sr, sg, sb, alpha as u8);
    let scan_ring = egui::Color32::from_rgba_premultiplied(sr, sg, sb, 70);
    painter.circle_filled(scan_pos, 3.0, scan_col);
    painter.circle_stroke(scan_pos, 4.5, egui::Stroke::new(1.0, scan_ring));
}

fn nearest_edge_midpoint(rect: egui::Rect, toward: egui::Pos2) -> egui::Pos2 {
    let c = rect.center();
    let dx = toward.x - c.x;
    let dy = toward.y - c.y;

    if dx.abs() >= dy.abs() {
        egui::pos2(if dx >= 0.0 { rect.right() } else { rect.left() }, c.y)
    } else {
        egui::pos2(c.x, if dy >= 0.0 { rect.bottom() } else { rect.top() })
    }
}

fn draw_arrowhead(painter: &egui::Painter, tip: egui::Pos2, dir: egui::Vec2, color: egui::Color32) {
    let perp = egui::Vec2::new(-dir.y, dir.x);
    let base = tip - dir * ARROW_LEN;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, base + perp * ARROW_HALF, base - perp * ARROW_HALF],
        color,
        egui::Stroke::NONE,
    ));
}
