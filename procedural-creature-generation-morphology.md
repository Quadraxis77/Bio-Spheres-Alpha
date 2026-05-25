# Sequence Morphology — Supplement to Procedural Creature Generation Spec

This document extends the main spec with geometric parameters for each sequence.
It is grounded in the mechanics of the actual genome system — specifically the
relationship between `parent_split_direction`, child `orientation` quaternions,
and Zone C bond inheritance. Spine pattern reference genomes will be documented
here as they are created.

---

## Morphogenesis Mechanics

Shape is not configured directly. It emerges from three interacting parameters
per cell division:

### 1. `parent_split_direction` — the growth axis

`parent_split_direction: [pitch, yaw]` sets the axis along which the parent
splits. The two children appear on opposite sides of this axis.

```
[90, 0]    →  splits upward (stem growth)
[-90, 0]   →  splits downward
[0, 0]     →  splits outward along genome Z (default)
[0, 90]    →  splits laterally around Y axis
[±45, 90]  →  diagonal splits (fan / branching patterns)
```

**Which axis produces which Zone:** Zone C is the narrow band perpendicular
to the split axis. Bonds that happen to be perpendicular to the split survive
to both children. Everything else goes to one child only.

Critical implication: the split axis determines what existing bonds survive.
To extend a structure, split along the growth axis — existing lateral bonds are
then perpendicular and land in Zone C, so connectivity propagates into new layers.

### 2. Child `orientation` quaternion — fanning and steering

The `orientation` field on `child_a` / `child_b` specifies the child's initial
orientation relative to the parent. This controls:

- **Fan angle:** rotating each child ±θ from the parent's facing direction
  spreads them laterally — used for fans, branching patterns, and spread structures.
- **Arm direction:** a large orientation offset steers a child arm away from
  the parent's axis.
- **Parallel / linear:** no orientation offset means children face the same
  direction as the parent — produces a straight stalk.

**Common quaternions and their effect:**

| Quaternion [x, y, z, w] | Approx rotation | Effect |
|--------------------------|-----------------|--------|
| `[0, ±0.383, 0, 0.924]` | ±45° around Y | Fans children ±45° |
| `[0, ±0.707, 0, 0.707]` | ±90° around Y | Fans children ±90° |
| `[0, ±0.500, 0, 0.866]` | ±60° around Y | Fans children ±60° |
| `[0.707, 0, 0, 0.707]`  | 90° around X  | Rotates child 90° around X |
| `[0.5, 0.5, 0.5, -0.5]` | 120° (cubic)  | Cubic lattice step |

### 3. Zone C bond inheritance — connecting the topology

Zone C is a band of bond angles that get inherited by both children when the
parent divides. The classification formula (from `adhesion_zones.rs`):

```
Zone C  ←→  |dot(bond_dir, split_dir) - ratio_shift| ≤ sin(equatorial_degrees)

where:
  ratio_shift        = 2 * split_ratio - 1
  equatorial_degrees = 3° + 19° × (|split_ratio − 0.5| / 0.2)   [clamped to 22°]
```

At **`split_ratio: 0.5`** (the default):
- `ratio_shift = 0.0` — Zone C is centered exactly at perpendicular (dot = 0)
- `equatorial_degrees = 3°`, threshold = sin(3°) ≈ 0.052
- Bonds within ±3° of perpendicular land in Zone C; everything else goes to one child
- Use this whenever bonds that are geometrically perpendicular to the split axis
  need to survive to both children

At **`split_ratio: 0.65`**:
- `ratio_shift = 0.3` — Zone C shifts to bonds at ~72° from the split axis (not 90°)
- `equatorial_degrees ≈ 17.25°`, threshold ≈ 0.297
- A perpendicular bond (dot = 0): `|0 − 0.3| = 0.3 > 0.297` → barely **Zone A**
- Perpendicular bonds do not benefit from 0.65 — the ratio_shift moves Zone C away
  from perpendicular

**What `split_ratio` actually controls:** which hemisphere of the parent's
existing adhesion network each child inherits. Non-0.5 values shift the Zone A/B
boundary asymmetrically, giving child_b more of the parent's bond network when
ratio_shift > 0. Use this when child_b must stay embedded in the parent
neighborhood while child_a migrates away.

**Jitter note:** `pseudo_random_rotation` (0.001–0.1 rad) is applied to spawn
*position* only, not to genome orientation or zone classification. Zone C
classification uses the genome-frame split direction, which is fully deterministic
and jitter-immune.

**The linear stalk pattern:**

```
split_direction: consistent along stalk axis
child_b: SelfRef (no orientation offset → continues same direction)
child_a: optional offset to branch a functional cell laterally
Zone C bonds: bond to previous stalk cell (perpendicular to growth axis)
Result: a chain where each cell bonds to its predecessor → flexible stalk
```

**The cyclic lattice pattern (Flat Square, Cubic Lattice genomes):**

```
parent_split_direction: [0, 0]  (default — along genome Z)
child_a orientation: SAME delta as child_b (not mirrored)
split_ratio: 0.5 (default)
parent_make_adhesion: true

2D flat grid — 90° around Y:
  child quaternion ≈ [0, 0.707, 0, −0.707]
  split direction cycles: Z → X → −Z → −X → Z (period 4)
  bonds from each generation are perpendicular to the next generation's split axis
  → all land in Zone C → full 4-connected XZ grid

3D cubic lattice — 120° around (1,1,1):
  child quaternion ≈ [0.5, 0.5, 0.5, −0.5]
  split direction cycles: Z → X → Y → Z (period 3)
  → full 6-connected cubic grid
```

The critical insight: giving **both children the same rotation delta** causes the
split direction to precess through perpendicular axes. Each generation's adhesion
bonds are by construction perpendicular to the next generation's splits, so they
always land in Zone C at split_ratio=0.5 without any special engineering.
This is a distinct structural primitive from fan branching (which uses mirrored
angles to spread children apart, not identical deltas to tile a plane).

---

## Multi-Mode Structural Primitives

### Terminology

**Backbone node** — any mode that is part of the structural skeleton the
generator uses to establish spatial positions. Every backbone node is a unique
mode index.

**Internal backbone node** — a backbone node whose two children are both also
backbone nodes. It is a branching point inside the structural skeleton.

**Backbone leaf** — a backbone node whose children are not more backbone nodes.
Its children are the start of functional sequences (vasculocyte → feature → stasis).
In a linear spine, this is the tip cell or the last branching node. In a lattice,
it is a boundary cell.

**Terminal backbone leaf** — a backbone leaf with no structural sequence attached
distally. Only terminal backbone leaves are valid gonad attachment points.

**Functional sequence** — any sequence that attaches at a backbone leaf and grows
the organism's working biology (sensors, muscles, feeders, gonads, vasculature,
photosynthesis). Functional sequence internals do not need to be unique-per-node;
SelfRef and shared modes are valid inside a functional sequence.

**Stasis mode** — any mode with `max_splits: 0`. The mandatory terminus of every
lineage except gonad cycles.

### The Problem with Monolithic Modes

The single-mode demonstration genomes (Flat Square, Octo-Tube) grow an entire
structure from one shared mode where every cell is identical. That is useful for
testing mechanics but useless for a generator: there are no independently
addressable positions and nowhere to branch off different functional sequences at
specific structural points.

SelfRef has the same problem at the backbone level. A stalk that self-extends
through one shared mode collapses all mid-nodes into a single mode index — the
generator cannot wire a sensor to node 3 and a myocyte to node 5 independently,
because nodes 3 and 5 are the same mode.

### The Binary Tree Model

Every structural primitive is built as a **full binary tree of unique mode indices**.
Each split produces exactly two **fresh, unused** child modes — never a SelfRef
back to a shared mode, never a mode index reused at a different tree position.
The tree has a configurable depth D, giving `2^(D+1) − 1` total backbone modes
and `2^D` leaf nodes. Every node in the tree is independently addressable.

```
Notation:
  [A], [B], [C] … = unique mode indices, assigned by the Slot Allocator
  [ATTACH]        = leaf node: wired to first mode of a functional sequence
  child_a / child_b point to unique mode indices, never SelfRef
```

**Depth D = organism resolution.** This is set per organism before slot
allocation. It controls how finely the generator can differentiate positions:

| D | Backbone modes | Leaf nodes | Slots remaining (of 80) |
|---|---------------|------------|-------------------------|
| 2 | 7  | 4  | 73 |
| 3 | 15 | 8  | 65 |
| 4 | 31 | 16 | 49 |
| 5 | 63 | 32 | 17 |

Depth 3–4 is the practical range for creatures that also need functional sequences.
Depth 5 consumes almost the entire budget.

### Linear Chain Spine

*Spine patterns will be documented here from reference genomes once created.*

### Linear Chain Stalk

A stalk backbone is a linear chain of N unique vasculocyte modes. Each node
branches one functional sequence laterally while the axial chain continues.
The split axis at every node is lateral (perpendicular to the stalk axis) so
the axial chain bond is Zone C and the backbone nutrient path is unbroken.

```
[Chain-0]  vasculocyte, unique
  child_a: [Chain-1]          ← continues stalk axially (backbone path)
  child_b: [Branch-0]         ← functional sequence at this position
  split_direction: lateral    ← perpendicular to axial direction
  max_splits: 1

[Chain-1]  vasculocyte, unique
  child_a: [Chain-2]
  child_b: [Branch-1]
  split_direction: lateral
  max_splits: 1

         … (N nodes = N branching positions) …

[Chain-Tip]  vasculocyte, unique
  child_a: [Stasis-Vasculocyte]   ← backbone terminates here
  child_b: [Branch-N]             ← functional sequence at tip
  max_splits: 1
```

Stalk length = N chain nodes. Every position is independently addressable because
every chain node is a unique mode. The nutrient highway runs root-to-tip
uninterrupted through the axial child_a chain.

Note: SelfRef is valid **inside functional sequences** (a feeder that self-extends,
a root that fans). The backbone chain must be unique-per-node to maintain
independent addressability. The distinction: backbone nodes are generator-addressed;
functional sequence internals are genome-autonomous.

### Binary Tree Flat Lattice

The cyclic-delta orientation still applies at each split, but each split produces
unique child modes rather than pointing back to a shared mode.

```
[Lattice-Root]
  orientation delta: [0, 0.707, 0, −0.707]  (90° around Y, on both children)
  child_a: [Lattice-A]    (unique)
  child_b: [Lattice-B]    (unique)

[Lattice-A]                       [Lattice-B]
  same orientation delta            same orientation delta
  child_a: [Lattice-AA]             child_a: [Lattice-BA]
  child_b: [Lattice-AB]             child_b: [Lattice-BB]

         … (depth = grid resolution) …

[Lattice-Leaf-*]  (all unique)
  → [ATTACH]   upward frond, photocyte, etc.
```

The split direction precesses through 90° increments exactly as in the monolithic
version, producing the same flat XZ grid. The difference is that every grid cell
is a unique mode, so the generator can place different functional cells at
different grid positions (center vs. edge vs. corner).

### Stasis Termination (Hard Constraint)

**Every lineage in every mode tree must reach a stasis mode.**

A stasis mode is any mode where `max_splits: 0`. A cell in stasis mode does not
divide further. It maintains its cell type, participates in adhesion bonds, and
responds to signals — it simply never splits again.

This constraint applies without exception to:
- Every backbone binary tree branch (all leaves must reach stasis through their
  downstream sequences)
- Every vasculocyte node lineage
- Every terminal feature mode lineage
- Every internal functional sequence branch — including both child_a and child_b
  of every internal node, and both targets of every mode_a_after_splits /
  mode_b_after_splits transition

A genome with any lineage that never reaches stasis will grow indefinitely,
consuming all available nutrients and preventing other cells from receiving
resources. The Slot Allocator must verify stasis reachability for all branches
before emitting a genome.

**Stasis modes can be shared.** A single stasis mode definition can serve as
the terminus for many different lineages across the genome. The stasis mode's
cell_type determines the final differentiated state of the cell. Common patterns:

```
one stasis mode per functional cell type in the organism:
  [Stasis-Photocyte]     max_splits: 0, cell_type: Photocyte
  [Stasis-Myocyte]       max_splits: 0, cell_type: Myocyte
  [Stasis-Vasculocyte]   max_splits: 0, cell_type: Vasculocyte
  [Stasis-Flagellocyte]  max_splits: 0, cell_type: Flagellocyte
  …

all lineages of the same terminal cell type converge to the same stasis mode
```

Sharing stasis modes saves slots and is the norm. The number of stasis modes
needed equals the number of distinct terminal cell types in the organism, not the
number of lineages.

**Stasis depth is a growth parameter.** How many splits occur before stasis
determines local cell count and feature size. The backbone tree positions the
feature; the intervening modes between backbone leaf and stasis determine how many
cells fill that position. Deeper stasis = more cells = larger feature.

**The complete lineage template:**

```
[Backbone leaf]          — unique, addresses the structural position
      ↓
[Vasculocyte node]       — shared per region; forms supply network
      ↓
[Feature growth mode(s)] — shared per feature type; 1–N modes of functional
                           cell growth (can branch internally)
      ↓
[Stasis mode]            — shared per cell type; max_splits: 0
                           mandatory terminus for every branch
```

Every fork within feature growth modes must also reach stasis — if a feature
mode has child_a branching to a secondary cell type and child_b continuing the
primary lineage, both child_a and child_b must eventually reach a stasis mode.

**Exception: egg-shedding cycles.** Gonad lineages are explicitly exempt from
the stasis requirement. They form a deliberate loop in the mode graph:

```
[Gonad-Stem]  ──divide──→  [Egg]
     ↑                        │
     │                   hold (embryocyte_use_timer / min_adhesions gate)
     │                        │
     └──regrow────────── release (embryocyte shed, leaves parent body plan)
```

This cycle never reaches permanent stasis. The gonad stem regrows after each
release and divides again, producing eggs continuously as long as the organism
is fed.

**Gonad placement constraint — terminal leaves only.**
The gonad sequence must be assigned exclusively to terminal leaf positions in the
backbone tree — positions that have no distal structure beyond them. A gonad placed
at an internal backbone node has adhesion bonds pointing both toward the body center
and toward distal cells. When the gonad divides to shed an embryocyte, Zone A/B/C
inheritance distributes those bonds across the two children. The embryocyte departs
with some of them. Any bond the embryocyte carried to a distal cell is severed on
departure — disconnecting that region of the body from the rest of the organism.

A gonad at a terminal leaf has bonds only pointing inward (toward the body). The
departing embryocyte may inherit a Zone C copy of the inward bond, but the stem
retains the original. No structure exists beyond the gonad to be disconnected.

The Slot Allocator must enforce this: gonad sequences are only assignable to
backbone leaf nodes, never to internal backbone nodes or to any position that
has further structural sequences attached distally.

**The shed embryocyte carries the full parent genome and develops identically.**
It is not a simplified or partial organism — it starts from `initial_mode` and
follows the same growth sequence as its parent: backbone expansion, functional
sequences, vascular network, stasis terminations, and its own gonad cycle. The genome is validated once per species. Both parent and every offspring
satisfy that validation by identity; no separate embryocyte validation is required.

From the parent body plan's perspective, the embryocyte is permanently gone at
the moment of shedding. Its subsequent development is outside the parent's mode
graph and outside the parent's validation scope.

The Slot Allocator treats any mode cycle that satisfies all three of the following
as a valid exempt pattern rather than an error:

1. **Gated hold phase:** the egg mode uses `embryocyte_use_timer` with a
   nonzero `embryocyte_use_threshold`, or the stem uses a `min_adhesions` gate
   on division, enforcing a minimum hold duration before release.
2. **Clean exit:** the shed cell departs the parent body plan permanently and
   does not re-enter the parent's mode graph or adhesion network.
3. **No structural output:** the cycle produces only shedable embryocytes — it
   does not generate new permanent structural cells or expand the parent body plan.

Any mode cycle that does not satisfy all three conditions is treated as runaway
division and flagged as invalid by the Slot Allocator.

### Terminal Growth Collapse

Once the binary tree has established a cell's structural address, the cells that
grow out of that position to fill the feature (photocytes tiling a surface,
myocytes filling a muscle band, flagellocytes spanning a propulsion fan) do not
need individual addresses — they need to fill space. Collapsing terminal growth
to a single shared mode is correct and necessary for mode budget reasons.

The binary tree provides position. The terminal shared mode provides growth.
These are distinct responsibilities; conflating them wastes slots.

### Vascular Network Distribution

The backbone itself is the primary nutrient highway. Every backbone node must be
a vasculocyte. When a functional sequence branches off a backbone node, the split
that causes that branch must not disrupt the backbone's continuous vasculocyte path.

**The Zone C continuity rule:** at every branching point, the backbone's direction
of travel must be perpendicular to the split axis used for branching. This makes
the backbone's chain bonds Zone C — they survive to the backbone-continuation
child and the path remains unbroken. The branching child also gets a copy of those
bonds (Zone C gives both children the bond), which connects the functional sequence
to the backbone highway for nutrient delivery.

```
Backbone chain bond direction: ──→  (e.g., along the spine axis)
Branch split axis:              ↑   (perpendicular to chain direction)

Result:
  chain bond ⊥ split axis  →  chain bond is Zone C
  → backbone-continuation child: retains chain bond  ✓  path unbroken
  → functional-branch child:     gets copy of chain bond  ✓  connected to highway
```

If the branch split axis is NOT perpendicular to the backbone direction, the
chain bond lands in Zone A or B — only one child gets it — and the backbone
vasculocyte highway is severed at that node.

**Body plan topologies and their backbone structure:**

The generator selects a body plan topology before allocating modes. The topology
determines the shape of the backbone and how the continuous nutrient highway runs
through it. Topology options will be expanded once spine reference genomes are
documented.

*Linear spine:*
The main body is a chain of vasculocyte modes running along one axis. Each node
branches one functional sequence laterally while continuing the chain axially.
The split axis at each branch is lateral (perpendicular to the spine axis), making
the axial chain bonds Zone C. The highway runs unbroken from one end to the other.

```
[Spine-0] ─axial─→ [Spine-1] ─axial─→ [Spine-2] ─axial─→ [Spine-Tip]
    │                   │                   │
  lateral             lateral             lateral
    ↓                   ↓                   ↓
[Branch-0]          [Branch-1]          [Branch-2]
```

*Flat lattice (mat / pad):*
The main body is a 2D grid of vasculocyte cells formed via the cyclic-delta split
pattern. The grid bonds are Zone C relative to upward branching splits, so
functional sequences growing upward from lattice cells do not disrupt the grid
connectivity. The grid bond network is the nutrient highway.

*Branching tree (bush / colonial):*
The main body is an open branching tree of vasculocyte nodes. The nutrient highway
follows the tree from root to each branch tip. Functional sequences attach at leaf
positions of the vasculocyte tree.

**The universal constraint across all topologies:** at every point where a
functional sequence branches off the backbone, the branch split axis must be
perpendicular to the backbone's direction of travel. This keeps the backbone's
chain bonds Zone C, preserving the nutrient highway through the branch point.

**Consequence for the Slot Allocator:** the Slot Allocator first resolves the
body plan topology, then builds the backbone vasculocyte chain for that topology,
then assigns functional sequences to branch positions. Every backbone node is a
vasculocyte. It verifies that at every branching point the branch split direction
is perpendicular to the backbone direction of travel.

### Reusing Feature Sequences Across Leaves

The backbone tree is all-unique, but functional sequences attached at leaves
**can be shared**. Multiple backbone leaf modes can wire to the **same starting
mode** of a feature sequence. The feature sequence is instantiated once per
leaf cell that reaches it at runtime, but only one set of mode definitions is
allocated in the slot table.

```
Backbone leaf modes (all unique — binary tree guarantee):
  [Leaf-0] child_a → [Sensor-Root]   ┐
  [Leaf-2] child_a → [Sensor-Root]   ├── three leaves, one sequence allocation
  [Leaf-5] child_a → [Sensor-Root]   ┘

  [Leaf-1] child_a → [Arm-Root]      ┐
  [Leaf-3] child_a → [Arm-Root]      ├── two leaves, one sequence allocation
  [Leaf-4] child_a → [Arm-Root]      ┘

  [Leaf-6] child_a → [Gonad-Root]    ── one leaf, one sequence allocation
  [Leaf-7] child_a → [Anchor-Root]   ── one leaf, one sequence allocation

Total backbone modes (depth 3):  15
Total functional sequence modes:  Sensor(4) + Arm(5) + Gonad(6) + Anchor(4) = 19
Total:  34 modes  (vs. 15 + 8 × avg(5) = 55 modes if each leaf got its own copy)
```

This is the primary mechanism for keeping mode budgets tractable on creatures
with repeated features. An 8-armed symmetric creature uses one arm sequence and
8 unique leaf modes that all point to its root. An asymmetric creature where each
arm is different would need 8 separate arm sequence allocations — each unique leaf
wires to a different sequence root. The Slot Allocator tracks which sequences have
already been allocated and reuses their starting mode index when the same sequence
type appears at multiple leaves.

The rule: **backbone nodes are always unique; feature sequence nodes are unique
per sequence type, shared across leaves that use the same type.**

### Resolution as an Organism Parameter

The tree depth D is resolved before slot allocation and stored in the organism
descriptor. The Slot Allocator generates the full binary tree of mode indices from
D, then presents the leaf list to the generator for functional sequence assignment.

```rust
struct OrganismDescriptor {
    /// The body plan topology — determines the shape of the backbone
    /// and how the continuous vasculocyte highway runs through it.
    body_plan: BodyPlanTopology,

    /// Number of functional sequence attachment positions on the backbone.
    /// For a spine: number of nodes in the chain.
    /// For a lattice: number of surface cells.
    /// For a tree: number of leaf nodes.
    branch_count: u32,

    /// Functional sequences assigned to each branch position.
    /// Length must equal branch_count.
    branch_sequences: Vec<SequenceRef>,
}

enum BodyPlanTopology {
    Spine,    // Linear chain; bilateral body layout; simplest highway
    Lattice,  // 2D grid; flat organisms; cyclic-delta formation
    Tree,     // Open branching tree; colonial / bush organisms
    // Additional topologies to be added from spine reference genomes
}
```

The generator selects `body_plan` and `branch_count` based on:
- Ecological niche
- Available mode budget (80 slots minus backbone vasculocyte chain minus functional sequences)
- Morphological complexity target

---

## Morphology Data Structures

```rust
/// Describes how a sequence grows into 3D space.
/// All fields are expressed in terms the generator can directly translate
/// into parent_split_direction and child orientation values.
struct SequenceMorphology {
    /// Named shape produced by this sequence when fully grown.
    shape: MorphShape,

    /// Ordered growth phases. Nodes within each phase share the same
    /// split_direction strategy and Zone C target pattern.
    phases: Vec<GrowthPhase>,

    /// Expected extents of the grown structure, in cell diameters.
    /// Derived from max_splits ranges in the sequence nodes.
    extent: MorphExtent,
}

enum MorphShape {
    Stalk,      // Linear elongated structure. Aspect ratio > 4:1.
    Fan,        // Flat spreading sheet. Width > depth.
    Cup,        // Concave radial bowl. Rim wider than base.
    Tube,       // Hollow extruded cylinder (ring + extrusion phases).
    Bush,       // Multi-branched 3D mass. Width ≈ height.
    Spike,      // Single tapered linear extension. High aspect + taper.
    Pad,        // Flat adhesive base. Very short stalk, wide footprint.
    Cluster,    // Dense rounded mass. Aspect ratio ≈ 1:1.
    Coil,       // Helical / spiral extension.
    Mat,        // Horizontal spreading sheet. Very low aspect ratio.
    Frond,      // Vertical flat sheet (buoyancy-driven or upward fan).
}

struct GrowthPhase {
    /// Name of this phase (e.g., "ring_formation", "extrusion", "arm_growth").
    name: &'static str,

    /// Number of distinct modes in this phase.
    /// A phase with mode_count > 1 has one mode per structural depth,
    /// enabling the generator to attach functional sequences at specific positions.
    mode_count: u32,

    /// parent_split_direction for all nodes in this phase.
    /// [pitch_deg, yaw_deg]. 0° pitch = horizontal outward. 90° = up. -90° = down.
    split_direction: [f32; 2],

    /// How children are oriented relative to parent at each split in this phase.
    child_rotation: ChildRotationPattern,

    /// split_ratio to use for all nodes in this phase.
    /// 0.5 (default): Zone C centered exactly at perpendicular bonds, ±3° window.
    ///   Use this when bonds that are geometrically perpendicular to the split axis
    ///   need to survive to both children.
    /// Other values: ratio_shift = 2*split_ratio−1 moves the Zone C band away from
    ///   perpendicular, shifting which hemisphere of bonds child_b inherits.
    ///   Does NOT widen Zone C for perpendicular bonds; at 0.65 a perpendicular bond
    ///   sits barely outside Zone C.
    split_ratio: f32,

    /// Which existing adhesion bonds must survive (land in Zone C) for the
    /// sequence topology to form correctly.
    zone_c_target: &'static str,

    /// Attachment points in this phase — mode-relative positions where the
    /// generator can wire in functional sequences.
    /// "root"  = child_a or child_b of the first mode in this phase
    /// "leaf"  = child_a or child_b of the last mode (terminal structural node)
    /// "node"  = child_a of a self-extending mid mode (fires at every split)
    /// "tip"   = mode_a_after_splits / mode_b_after_splits target
    attachment_points: &'static [&'static str],
}

enum ChildRotationPattern {
    /// Both children face same direction as parent. Produces linear stalk.
    Parallel,

    /// Children fanned symmetrically at ±θ from parent facing direction.
    /// Used for radial fan growth and spreading structures.
    MirroredFan { half_angle_deg: f32 },

    /// child_a offset by θ_a, child_b offset by θ_b (different values).
    /// Used for asymmetric steering and diagonal arms.
    Asymmetric { child_a_deg: f32, child_b_deg: f32 },

    /// One child parallel (SelfRef self-extend), one child offset for a branch.
    SelfExtendWithBranch { branch_angle_deg: f32 },

    /// No specific rotation — child inherits parent orientation unchanged.
    Inherit,

    /// Both children receive the SAME rotation delta (not mirrored).
    /// The delta rotates the split direction by a fixed angle around a fixed axis,
    /// so split direction cycles through perpendicular axes across generations.
    /// Previous-generation bonds are always perpendicular to the next split →
    /// automatically Zone C → full lattice connectivity propagates.
    ///
    /// CyclicLattice2D: 90° around Y → splits cycle Z → X → −Z → −X → Z
    ///   Result: flat XZ grid. child quaternion ≈ [0, 0.707, 0, −0.707]
    ///
    /// CyclicLattice3D: 120° around (1,1,1) diagonal → splits cycle Z → X → Y → Z
    ///   Result: 3D cubic grid. child quaternion ≈ [0.5, 0.5, 0.5, −0.5]
    ///
    /// Key: no parent_split_direction needed (leave at default [0,0]).
    ///      split_ratio: 0.5 (default). parent_make_adhesion: true required.
    CyclicLattice { child_quaternion: [f32; 4], cycle_length: u32 },
}

struct MorphExtent {
    /// Length along primary growth axis, in cells. [min, max].
    length: [u32; 2],

    /// Width perpendicular to growth axis, in cells. [min, max].
    width: [u32; 2],

    /// Cross-section taper from base to tip.
    /// -1.0 = widens strongly toward tip. 0.0 = uniform. +1.0 = tapers to point.
    taper: f32,

    /// Approximate cell-diameter radius at the backbone attachment point.
    /// Used by body-plan assembler to avoid sequence overlap.
    attachment_footprint: f32,
}
```

---

## Sequence Morphology Blocks

One morphology block per sequence in the library. Each block is included
inline with the corresponding sequence definition in the main spec.

---

### Spine Sequences

*To be documented from reference genomes.*

---

### Arm Sequence (5 modes)

```
morphology:
  shape: Stalk

  phases:
  - name: arm_growth
    split_direction: [0, 0]           # outward from backbone attachment (radial)
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef continues outward
                                      # child_a branches ±0° (forward) → tip
    split_ratio: 0.5                  # stalk bonds; Zone C catches axial chain bond
    zone_c_target: "bond to previous stalk cell"

  - name: tip_differentiation         # at Arm-Mid terminal split
    split_direction: [0, 0]
    child_rotation: Asymmetric        # child_a toward ch_ap high (anterior)
                                      # child_b toward ch_ap low (posterior)
    split_ratio: 0.5
    zone_c_target: "bond to Arm-Mid parent"

  extent:
    length: [3, 8]      # cells along arm axis
    width: [1, 3]       # 1 at root, widens to 2-3 at forked tip
    taper: 0.1          # nearly cylindrical
    attachment_footprint: 1.0
```

**Notes:** The arm stalk uses split_ratio 0.5 — it is building a simple chain
where Zone C only needs to catch the parent-child bond (which is always
perpendicular). Tip differentiation by ch_ap is a signal effect, not morphological.

---

### Feeder Sequence (5 modes)

```
morphology:
  shape: Bush

  phases:
  - name: branching
    split_direction: [0, 0]           # initially outward, then varies per branch
    child_rotation: MirroredFan       # half_angle: 60–90°
                                      # wide fan angle produces bush shape
    split_ratio: 0.5
    zone_c_target: "bond to parent branch cell"

  - name: tip_storage                 # Feeder-Tip (Lipocyte), no further splits
    split_direction: N/A
    child_rotation: N/A
    split_ratio: 0.5
    zone_c_target: N/A

  extent:
    length: [2, 5]       # depth from backbone to feeder tips
    width: [3, 7]        # lateral spread (num branches × spacing)
    taper: 0.5           # tapers from bushy base to individual tip cells
    attachment_footprint: 1.5
```

---

### Sensor Sequence (5 modes)

```
morphology:
  shape: Stalk

  phases:
  - name: stalk_growth
    split_direction: [15, 0]          # slightly upward-forward outward from backbone
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → straight stalk
                                      # child_a N/A (stalk is unbranched)
    split_ratio: 0.5
    zone_c_target: "bond to previous stalk cell"

  - name: eye_placement               # Sensor-Eye terminal node
    split_direction: N/A              # max_splits: 0, no further division
    child_rotation: N/A
    split_ratio: 0.5
    zone_c_target: N/A

  extent:
    length: [3, 6]
    width: [1, 1]        # single-cell stalk
    taper: 0.1
    attachment_footprint: 1.0
```

---

### Locomotion Sequence (6 modes)

```
morphology:
  shape: Fan             # for Flagellocyte / Ciliocyte niches
                         # (Myocyte locomotion: see Arm Sequence with myocyte tips)

  phases:
  - name: inlet_extension
    split_direction: [0, 0]           # outward from ring
    child_rotation: SelfExtendWithBranch
    split_ratio: 0.5
    zone_c_target: "bond to backbone attachment"

  - name: loco_fan                    # Loco-Root → Loco-Mid branches
    split_direction: [0, 0]           # continues outward
    child_rotation: MirroredFan       # half_angle: 30–60°
                                      # fans propulsion cells laterally
    split_ratio: 0.5
    zone_c_target: "bond to Loco-Root"

  extent:
    length: [2, 5]
    width: [2, 5]        # fan spread
    taper: 0.4
    attachment_footprint: 1.5
```

---

### Gonad Sequence (6 modes)

```
morphology:
  shape: Cluster

  phases:
  - name: egg_production              # Gonad-Stem sheds Embryocytes radially
    split_direction: [0, 0]           # outward / radial from gonad center
    child_rotation: MirroredFan       # half_angle: 45–90° → eggs spread outward
    split_ratio: 0.65                 # keeps eggs bonded to Gonad-Stem (Zone C)
    zone_c_target: "bond to Gonad-Stem (keep_adhesion: true)"

  - name: rest_and_rebuild            # Gonad-Rest → Gonad-Rebuild cycle
    split_direction: [0, 0]
    child_rotation: Asymmetric
    split_ratio: 0.5
    zone_c_target: "bond to backbone attachment"

  extent:
    length: [2, 4]       # radial distance from backbone to egg perimeter
    width: [3, 6]        # egg count defines cluster width
    taper: -0.2          # bulges outward (eggs at perimeter)
    attachment_footprint: 2.0
```

**Notes:** Creature 73C2's Gonad (mode 7) uses `split_ratio: 0.65` on the stem
and `parent_make_adhesion: true` to keep eggs bonded during feeding. The cluster
is roughly spherical, with eggs radiating in all directions from the stem center.

---

### Vascular Sequence (4 modes)

```
morphology:
  shape: Stalk           # sealed pipe, no functional branching

  phases:
  - name: pipe_extension
    split_direction: [0, 0]           # along the body axis (varies by placement)
    child_rotation: Parallel          # straight pipe, no fan
    split_ratio: 0.5
    zone_c_target: "bond to previous pipe cell"

  extent:
    length: [5, 13]      # long pipe
    width: [1, 1]
    taper: 0.0           # uniform cylinder
    attachment_footprint: 1.0
```

---

### Anchor Sequence (5 modes)

```
morphology:
  shape: Pad

  phases:
  - name: stalk_descent
    split_direction: [-75, 0]         # steeply downward from backbone attachment
    child_rotation: SelfExtendWithBranch
                                      # child_b continues downward (SelfRef)
                                      # child_a: Glueocyte pad cell laterally
    split_ratio: 0.5
    zone_c_target: "bond to previous stalk cell"

  - name: pad_spread                  # Anchor-Pad Glueocyte spread
    split_direction: [0, 0]           # horizontal spread at substrate level
    child_rotation: MirroredFan       # half_angle: 45–90° → widens at base
    split_ratio: 0.5
    zone_c_target: "substrate adhesion bond (env adhesion)"

  extent:
    length: [3, 6]       # stalk depth
    width: [1, 3]        # 1 at top, 2-3 at pad
    taper: -0.3          # widens toward substrate
    attachment_footprint: 1.0
```

---

### Tendril / Grasper Sequence (6 modes)

```
morphology:
  shape: Stalk           # long flexible arm with lateral myocyte pairs

  phases:
  - name: tendril_growth
    split_direction: [20, 0]          # slightly upward-forward
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → long flexible shaft
                                      # child_a: lateral myocyte at each node
    split_ratio: 0.5
    zone_c_target: "bond to previous shaft cell"

  - name: grip_placement              # Tendril-Tip Glueocyte terminal
    split_direction: N/A
    child_rotation: N/A
    split_ratio: 0.5
    zone_c_target: N/A

  extent:
    length: [5, 10]      # long reach
    width: [1, 1]        # shaft is single-cell wide; myocytes are lateral "wings"
    taper: 0.2
    attachment_footprint: 1.0

  notes: >
    The paired Myocyte-A / Myocyte-B cells attach as child_a branches at each
    shaft node with ±90° orientation offsets from the shaft axis. They are not
    counted in the width_range — they project perpendicular to the stalk and
    constitute the bending mechanism, not the structural extent.
    See: Triangular Prism genome — it uses a similar pattern for placing
    flagellocyte pairs at ±135°/±45° off the main arm direction.
```

---

### Ciliary Conveyor Sequence (5 modes)

```
morphology:
  shape: Fan             # flat lateral sheet of ciliocytes

  phases:
  - name: fan_spread
    split_direction: [0, 90]          # lateral (perpendicular to arm axis)
    child_rotation: MirroredFan       # half_angle: 30–60° → spreads sheet laterally
    split_ratio: 0.5
    zone_c_target: "bond to backbone attachment / previous fan cell"

  extent:
    length: [2, 4]       # depth of sheet (cells outward from backbone)
    width: [4, 8]        # lateral spread
    taper: 0.3
    attachment_footprint: 2.0

  notes: >
    The split_direction [0, 90] fans cells in the lateral plane (same direction
    used for ring formation, but here the ring is already closed — the fan
    spreads sideways rather than closing a loop). Zone C at this split catches
    the ring leaf bond, keeping all fan cells attached to the ring highway.
```

---

### Pulsatile Jet Sequence (5 modes)

```
morphology:
  shape: Cup             # muscular bell / chamber, open at trailing end

  phases:
  - name: chamber_ring              # Jet-Chamber grows a mini-ring of myocytes
    split_direction: [0, 90]        # same as ring formation
    child_rotation: MirroredFan     # half_angle: 45° → 4-cell myocyte ring
    split_ratio: 0.5                # Zone C centered at perpendicular (ring bonds)
    zone_c_target: "jet chamber ring bonds"

  - name: chamber_extension         # extends chamber depth axially
    split_direction: [-20, 0]       # slightly trailing
    child_rotation: Parallel
    split_ratio: 0.5
    zone_c_target: "chamber ring bonds (inherited as tube)"

  extent:
    length: [2, 5]       # chamber depth
    width: [3, 7]        # chamber ring diameter
    taper: -0.6          # bulges outward (bell shape)
    attachment_footprint: 2.5

  notes: >
    The jet chamber is structurally a mini-tube: a small closed loop formed
    via binary doubling in [0, 90] then extruded in [-20, 0].
    The myocytes contract synchronously (same phase) to squeeze the chamber
    and produce the jet pulse.
```

---

### Chemoreceptor Spine Sequence (4–5 modes)

```
morphology:
  shape: Stalk           # rigid spine with dual-sensor fork at tip

  phases:
  - name: spine_stalk
    split_direction: [20, 0]          # forward-upward
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → straight spine
                                      # child_a: Spine-Node at tip for fork
    split_ratio: 0.5
    zone_c_target: "bond to previous spine cell"

  - name: sensor_fork               # Spine-Node → Eye-Primary + Eye-Secondary
    split_direction: [20, 0]
    child_rotation: Asymmetric        # child_a: primary sense type, 0° offset
                                      # child_b: secondary sense type, ±20° offset
    split_ratio: 0.5
    zone_c_target: "bond to Spine-Shaft"

  extent:
    length: [4, 8]
    width: [1, 2]        # 1 along shaft, 2 at forked tip
    taper: 0.1
    attachment_footprint: 1.0
```

---

### Defensive Spine Sequence (5 modes)

```
morphology:
  shape: Spike

  phases:
  - name: spine_extension
    split_direction: [0, varies]      # outward, yaw randomized per position
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → spike continues
                                      # child_a: Spine-Tip (devorocyte/glueocyte)
    split_ratio: 0.5
    zone_c_target: "bond to previous spine cell"

  extent:
    length: [2, 5]
    width: [1, 1]
    taper: 0.7           # sharp point
    attachment_footprint: 0.8

  notes: >
    The yaw angle of the split_direction determines which direction the spike
    points. Randomizing yaw across ring positions produces an all-around
    defensive array. The Spine-Base myocyte retracts the spike by contracting
    the adhesion bonds when ch_stress is absent — the geometry collapses
    inward rather than extending outward.
```

---

### Nutrient Trap Sequence (5 modes)

```
morphology:
  shape: Cup

  phases:
  - name: cup_wall
    split_direction: [10, 0]          # slightly forward-facing
    child_rotation: MirroredFan       # half_angle: 45–60° → forms cup rim
    split_ratio: 0.5
    zone_c_target: "bond to Trap-Wall / previous rim cell"

  - name: rim_ciliation              # Trap-Rim ciliocyte ring
    split_direction: [10, 0]
    child_rotation: MirroredFan       # half_angle: 30° → continues rim circle
    split_ratio: 0.5
    zone_c_target: "bond to rim neighbors"

  extent:
    length: [2, 4]       # cup depth (floor to rim)
    width: [4, 8]        # rim diameter
    taper: -0.8          # strongly widens toward mouth
    attachment_footprint: 2.5
```

---

### Bioluminescent Lure Sequence (5 modes)

```
morphology:
  shape: Stalk           # flexible stalk with glowing bulb at tip

  phases:
  - name: stalk_growth
    split_direction: [45, 0]          # upward-forward at 45°
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → stalk grows upward
                                      # child_a: Lure-Light photocyte at tip
    split_ratio: 0.5
    zone_c_target: "bond to previous stalk cell"

  extent:
    length: [2, 5]
    width: [1, 2]        # 1 stalk + 1 photocyte bulb
    taper: -0.3          # slightly wider at tip (the light bulb)
    attachment_footprint: 1.5

  notes: >
    The stalk is myocyte-driven (Lure-Stalk is a Vasculocyte with myocyte
    behavior), contracting rhythmically to make the light bob. The bioluminescent
    point is child_a at each stalk node; child_b (SelfRef) continues the stalk.
    The Lure-Base devorocyte sits at the base, fed directly by the backbone vasculocyte outlet.
```

---

### Symbiont Dock Sequence (3 modes)

```
morphology:
  shape: Pad

  phases:
  - name: dock_arm
    split_direction: [0, varies]      # outward, yaw varies (placed on any ring pos)
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → short arm extends
                                      # child_a: Dock-Pad at arm tip
    split_ratio: 0.5
    zone_c_target: "bond to backbone attachment"

  extent:
    length: [2, 4]
    width: [1, 1]
    taper: 0.0
    attachment_footprint: 1.0
```

---

### Regeneration Sequence (5 modes)

```
morphology:
  shape: Cluster         # dormant niche embedded in body

  phases:
  - name: scaffold_growth            # Regen-Scaffold positions dormant cell
    split_direction: [0, 0]          # inward / toward body center
    child_rotation: SelfExtendWithBranch
                                      # child_a: Regen-Dormant (positioned inward)
                                      # child_b SelfRef: continues scaffold
    split_ratio: 0.5
    zone_c_target: "bond to backbone attachment / body attachment"

  - name: regrowth                   # activated after damage (ch_stress absent)
    split_direction: [0, varies]     # regrowth directed by ch_ap toward lost tissue
    child_rotation: MirroredFan
    split_ratio: 0.5
    zone_c_target: "bond to Regen-Active parent"

  extent:
    length: [1, 3]
    width: [2, 4]
    taper: 0.3
    attachment_footprint: 1.5
```

---

## Plant Sequence Morphology

Plant sequences use the same structural primitives but interpret
`parent_split_direction` differently:

- **Upward growth:** `[90, 0]` or `[88, 0]` — stems growing toward light
- **Downward growth:** `[-80, 0]` — roots growing toward substrate
- **Lateral spread:** `[0, 90]` or `[0, varies]` — leaves and ground cover

---

### Root Anchor Sequence (5 modes)

```
morphology:
  shape: Bush            # inverted — bush grows downward

  phases:
  - name: root_descent
    split_direction: [-80, 0]         # steeply downward
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → continues downward
                                      # child_a: Root-Hair branches laterally
    split_ratio: 0.5
    zone_c_target: "bond to parent root cell"

  - name: root_branching             # lateral branching for coverage
    split_direction: [-60, varies]    # downward with yaw spread
    child_rotation: MirroredFan       # half_angle: 40–70° → root fans out
    split_ratio: 0.5
    zone_c_target: "bond to root branch parent"

  extent:
    length: [4, 8]       # depth below backbone
    width: [4, 8]        # lateral spread of root mat
    taper: 0.6           # tapers to fine root tips
    attachment_footprint: 2.0
```

---

### Stem Sequence (5 modes)

```
morphology:
  shape: Stalk           # vertical column

  phases:
  - name: vertical_growth
    split_direction: [88, 0]          # nearly vertical (slight variance for lean)
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → continues upward
                                      # child_a: Stem-Node branches laterally for leaves
    split_ratio: 0.5
    zone_c_target: "bond to previous stem cell"

  - name: node_branching             # Stem-Node → Stem-Photo + Stem-Lift
    split_direction: [0, 90]          # lateral
    child_rotation: Asymmetric        # child_a: Photocyte outward
                                      # child_b: Buoyocyte upward
    split_ratio: 0.5
    zone_c_target: "bond to Stem-Core"

  extent:
    length: [5, 11]      # cells in vertical column
    width: [1, 2]
    taper: 0.1
    attachment_footprint: 1.5
```

---

### Leaf Sequence (6 modes)

```
morphology:
  shape: Frond           # vertical flat sheet, fans laterally

  phases:
  - name: petiole
    split_direction: [0, 90]          # lateral from stem attachment
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → petiole extends
                                      # child_a: Leaf-Blade fan at attachment
    split_ratio: 0.5
    zone_c_target: "bond to Stem-Node"

  - name: blade_fan
    split_direction: [0, 90]          # continues lateral
    child_rotation: MirroredFan       # half_angle: 20–40° → flat fan
    split_ratio: 0.5
    zone_c_target: "bond to previous blade cell"

  extent:
    length: [3, 7]       # cells from stem to tip
    width: [5, 10]       # lateral spread
    taper: 0.4
    attachment_footprint: 1.5

  notes: >
    The petiole grows laterally ([0, 90]) and the blade fans further outward
    from the petiole. The Leaf-Vein (Vasculocyte) runs through the petiole;
    Leaf-Blade photocytes fan off the vein. The result is a flat photocyte sheet
    oriented perpendicular to the incoming light (maximizing surface area).
```

---

### Branch Sequence (5 modes)

```
morphology:
  shape: Stalk           # angled stalk with leaf attachment nodes

  phases:
  - name: branch_ascent
    split_direction: [40, varies]     # upward at 40°, yaw varies by position
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → branch continues
                                      # child_a: Branch-Leaf-Node every N splits
    split_ratio: 0.5
    zone_c_target: "bond to stem attachment"

  extent:
    length: [3, 8]
    width: [1, 2]        # 1 along branch, leaf nodes project laterally
    taper: 0.2
    attachment_footprint: 1.5

  notes: >
    Branch-Bud emits ch_ap in addition to being gated by ch_ap (inverted).
    This creates an apical dominance gradient: the Branch-Tip emits ch_ap
    at high strength, suppressing sub-branching on nearby nodes (close to tip =
    strong signal = inverted gate = suppressed). Lower nodes (weak ch_ap) grow
    freely. Vary split_direction yaw by ±20° per branch for natural spread.
```

---

### Flower / Spore Sequence (6 modes)

```
morphology:
  shape: Cup             # open radially upward, like a flower

  phases:
  - name: flower_inlet
    split_direction: [80, 0]          # upward (flower faces sky)
    child_rotation: SelfExtendWithBranch
    split_ratio: 0.5
    zone_c_target: "bond to stem/branch tip"

  - name: petal_ring                 # Flower-Base → Flower-Petal radial spread
    split_direction: [0, 90]          # lateral fan
    child_rotation: MirroredFan       # half_angle: 30–60° → petals open outward
    split_ratio: 0.5
    zone_c_target: "bond to Flower-Base"

  - name: petal_extension            # Flower-Petal self-fans outward
    split_direction: [0, 90]
    child_rotation: MirroredFan       # half_angle: 15–30° → finer petal spread
    split_ratio: 0.5
    zone_c_target: "bond to previous petal cell"

  extent:
    length: [2, 5]       # height from base to petal tips
    width: [3, 8]        # petal ring diameter
    taper: -0.5          # opens wide at top
    attachment_footprint: 2.0
```

---

### Tendril Climb Sequence (5 modes)

```
morphology:
  shape: Stalk           # flexible exploratory arm

  phases:
  - name: tendril_reach
    split_direction: [40, varies]     # upward-outward, exploratory yaw
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → continues probing
                                      # child_a: Tendril-Grip glueocyte at tip
    split_ratio: 0.5
    zone_c_target: "bond to vessel/stem attachment"

  extent:
    length: [5, 10]
    width: [1, 1]
    taper: 0.1
    attachment_footprint: 1.0

  notes: >
    The high axis_variance (exploratory yaw) is intentional — tendrils probe
    in different directions from different stem positions. Once Tendril-Grip
    bonds to a surface, Tendril-Base (Myocyte) contracts on ch_water, pulling
    the plant toward the anchor. The yaw is not truly random but is determined
    by the stem position's orientation inherited from the backbone.
```

---

### Storage Root Sequence (3 modes)

```
morphology:
  shape: Cluster         # bulbous underground storage organ

  phases:
  - name: bulb_growth
    split_direction: [-70, 0]         # downward
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → continues deeper
                                      # child_a: Lipocyte storage cell laterally
    split_ratio: 0.5
    zone_c_target: "bond to previous storage cell"

  extent:
    length: [2, 7]
    width: [2, 4]        # Lipocytes swell → effective width larger than cell count
    taper: -0.7          # bulges at base, narrow neck at attachment
    attachment_footprint: 2.0
```

---

### Moss / Ground Cover Sequence (5 modes)

```
morphology:
  shape: Mat

  phases:
  - name: lateral_spread             # flat XZ sheet via cyclic lattice pattern
    split_direction: [0, 0]          # default — no explicit split_direction field
    child_rotation: CyclicLattice    # BOTH children same delta: [0, 0.707, 0, −0.707]
                                     # splits cycle Z → X → −Z → −X → flat XZ plane
    split_ratio: 0.5
    zone_c_target: "previous-generation bonds (perpendicular by construction)"

  - name: frond_fan                  # Moss-Frond fans photocytes upward off mat
    split_direction: [90, 0]         # upward from mat surface
    child_rotation: MirroredFan      # half_angle: 20–40° → short upward fans
    split_ratio: 0.5
    zone_c_target: "bond to Moss-Rhizoid (mat attachment bond)"

  extent:
    length: [1, 3]       # very shallow (mat lies flat)
    width: [6, 15]       # wide lateral spread
    taper: 0.2
    attachment_footprint: 3.0

  notes: >
    The flat mat is NOT produced by MirroredFan or [0, 360] splits — that would
    form concentric rings or a radial star, not a flat sheet. The cyclic lattice
    pattern (same 90°-Y delta on both children, no parent_split_direction) causes
    the split axis to precess through four perpendicular directions in the XZ plane,
    tiling a flat grid. See: Flat Square Test Lattice.genome.
```

---

### Algae / Aquatic Plant Sequence (5 modes)

```
morphology:
  shape: Frond

  phases:
  - name: buoyant_rise
    split_direction: [75, varies]     # upward, slight yaw variation per instance
    child_rotation: SelfExtendWithBranch
                                      # child_b SelfRef → continues rising
                                      # child_a: Algae-Frond fans off axis
    split_ratio: 0.5
    zone_c_target: "bond to previous float cell"

  - name: frond_fan
    split_direction: [30, 90]         # lateral with upward tilt
    child_rotation: MirroredFan       # half_angle: 20–45°
    split_ratio: 0.5
    zone_c_target: "bond to Algae-Float"

  extent:
    length: [3, 8]       # vertical rise
    width: [3, 7]        # frond spread
    taper: 0.3
    attachment_footprint: 2.0

  notes: >
    Algae-Holdfast only anchors under mechanical stress (ch_stress gate). In
    calm water the plant free-floats, drifting upward via Algae-Float buoyancy.
    Under current, ch_stress fires and Holdfast bonds to substrate — the split
    direction for Holdfast growth is downward [-60, 0] to reach the substrate.
```

---

## Zone C Quick Reference — Common Topology Patterns

| Structure goal | split_direction | child_rotation | Zone C catches |
|----------------|-----------------|----------------|----------------|
| Vertical stalk | [88, 0] | SelfExtendWithBranch | Stalk chain bonds |
| Downward root | [-80, 0] | SelfExtendWithBranch + fan | Root chain bonds |
| Lateral fan/sheet | [0, 90] | MirroredFan ±30–60° | Backbone attachment bond |
| Radial flower cup | [0, 90] | MirroredFan ±30–60° | Base center bond |
| Diagonal arm | [±45, 90] | Asymmetric | Attachment bond |
| Flat 2D grid (Mat) | default [0,0] | CyclicLattice 90°-Y (same on both children) | Prior-gen bonds (perpendicular by precession) |
| 3D cubic grid | default [0,0] | CyclicLattice 120°-(1,1,1) (same on both children) | Prior-gen bonds (all 3 axes) |
| Spine branch-off | lateral to spine | backbone continuation axial | Spine chain bonds |

**The universal rule:** to keep a bond in Zone C, design the split axis to be
perpendicular to that bond. The exact condition is:

```
|dot(bond_dir, split_dir) - ratio_shift| ≤ sin(equatorial_degrees)
ratio_shift        = 2 * split_ratio - 1
equatorial_degrees = 3° + 19° × (|split_ratio − 0.5| / 0.2)  [clamped to 22°]
```

At `split_ratio: 0.5` (default): `ratio_shift = 0`, `equatorial_degrees = 3°`.
Zone C is centered at dot = 0 (exactly perpendicular). Use this whenever the bonds
to preserve are geometrically perpendicular to the split axis by construction.

At `split_ratio: 0.65`: `ratio_shift = 0.3`, `equatorial_degrees ≈ 17.25°`. Zone C
is centered at dot ≈ 0.3 (bonds at ~72° from split axis). A perpendicular bond
gives `|0 − 0.3| = 0.3` vs threshold ≈ 0.297 → barely Zone A. **Do not use 0.65
to preserve perpendicular bonds** — it moves the Zone C window away from perpendicular.

Jitter is applied to spawn *position* only. Zone C classification uses the
genome-frame split direction, which is deterministic and jitter-immune regardless
of split_ratio.
