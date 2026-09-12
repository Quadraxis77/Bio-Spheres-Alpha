# Organism Design Procedurals

Living notes for Bio-Spheres genome construction. These notes are empirical and
probabilistic: main-simulation reports are strong evidence, but not universal law.
Each report should influence future design decisions in context instead of being
treated as a hard commandment. Every new test report should be folded into the
pattern library as:

1. Observation: what happened in preview or main simulation.
2. Mechanism: why the engine likely produced that behavior.
3. Design heuristic: what this suggests for future genomes, including when it
   might not apply.
4. Confidence and context: how strongly to weight the lesson.
5. Genome action: the concrete setting change, if any.
6. Verification: test run, hash, or main-sim result.

The aim is not to accumulate universal commandments. The aim is to build a set
of weighted, reusable design patterns that can produce organisms which are
varied and dynamic while still looking authored. User reports are especially
valuable evidence, but they remain observations from a particular organism,
environment, engine version, and test duration. When a report conflicts with an
older lesson, preserve the context of both and narrow the heuristic instead of
silently declaring either one gospel.

## Current LUCA Lessons

### Bootstrap population

Observation: starting directly as `Founder Scout` forfeits the embryocyte reserve
boost. A single phagocyte founder starts with normal nutrients only, so early
population viability depends heavily on immediate contact with external food.

Mechanism: GPU insertion gives Embryocytes and Gametocytes a full reserve by
default. Non-storage cell types start with normal nutrients but no reserve. Free
Embryocytes burn reserve and hatch after their `split_interval`; their children
receive half the reserve and half the normal nutrients.

Design heuristic: for harsh or uncertain starts, an initial Embryocyte can be a
strong founder packet. This is not mandatory for every organism; direct founder
starts can still be valid for deliberately lean, opportunistic, or ecosystem-
dependent organisms. If an Embryocyte is used, its children should be
non-embryocyte starter cells so the reserve becomes a bootstrap subsidy without
creating embryo-reserve doubling chains.

Confidence and context: high for first-organism LUCA starts where the goal is a
viable seed population. Lower for later ecosystem organisms that are allowed to
depend on local food webs, hosts, mats, carrion, anchors, or reproductive luck.

Historical genome action: the earlier `LUCA Pelagic Signal Foundation` used a
dedicated startup Embryocyte. The current `LUCA Triskelion Ray` preserves the
same principle in mode 8 `Triskelion Embryo`, which hatches into detached mode 0
founders rather than another Embryocyte generation.

### Sustainable reproduction

Observation: early versions plateaued after apparent success because population
growth relied on initial embryocyte surplus rather than feeding into future
embryos. A later LUCA test showed a related pattern: after the initial nutrient
burst, eggs stopped being shed, suggesting the adult brood path was not filling
pods reliably.

Mechanism: reserve inherited from a startup Embryocyte can sustain several
generations, but if later embryocytes are not fed by active feeder tissue, the
population reaches a fixed plateau.

Design heuristic: separate launch reserve from reproductive reserve. Initial
Embryocytes may bootstrap, but attached brood Embryocytes usually need to be fed
by living phagocyte or photosynthetic tissue and should release only after a
threshold when the goal is sustainable population growth.

Confidence and context: high when population growth plateaus after early success.
Less relevant for intentionally semelparous organisms, scavenger blooms, or
organisms whose ecological role is to create a short pulse rather than persist.

Historical genome action: the earlier forked LUCA used a phase-locked gland and
separate fed pod. The current Triskelion uses mode 3 `Brood Core` to bud mode 8
`Triskelion Embryo`; the embryo still requires reserve before release, but its
foundation profile does not add a signal gate that could prevent reproduction
entirely.

### Brood nutrient flow

Observation: LUCA organisms shed eggs during the startup reserve burst, then egg
shedding stopped once the population was living on environmental feeding.

Mechanism: an attached Embryocyte is a pure nutrient sink. Its `nutrient_priority`
scales the receiver rate cap, while the sender's `prioritize_when_low` protects a
floor of about 10 nutrients. If a self-renewing gland can keep budding pods while
older pods are still attached, one small donor can create several unfinished
Embryocyte sinks. Low-threshold developmental bridge cells also have small direct
food buffers, so they are poor adult nutrient reservoirs even when they are good
shape builders.

Design heuristic: sustainable brood organs usually work better as a single-egg
queue than as uncontrolled budding tissue. Give the feeder gland enough priority
and storage to collect surplus, keep structural cells at lower base priority so
surplus drifts toward the gland, and prevent a second attached egg until the
first one releases. Egg priority should be high enough to fill, but not so high
that every brief surplus pulse is instantly scattered across multiple pods.

Confidence and context: high for small-bodied organisms with one brood gland.
Conditional for mat-formers, broadcast spawners, or organisms with multiple
independent feeder glands.

Genome action: LUCA brood rebalance changed the body priorities downward, raised
mode 8 `Phase-Locked Brood Gland` to `nutrient_priority: 3.2`, raised its split buffer
to `split_mass: 1.24`, and reduced mode 10 `Fed Embryo Pod` priority to `2.5`.
The unbraced fork gave the mature gland three structural adhesions, so
`max_adhesions: 4` initially provided one egg slot. The later 3D eye-to-gland
cross-brace raised the mature degree to four and the cap to five.

### Preview versus main simulation

Observation: shapes can appear viable in preview and fail in the main simulation.
The brood gland failure was one example: preview growth made an underfed gland
look functional. A later egg-shedding failure exposed a more specific mismatch:
adult eggs worked in preview but disappeared in the main sim after startup
reserve was exhausted.

Mechanism: preview can grant generous automatic gains to some cell types, while
the main GPU simulation requires phagocytes to physically occupy nutrient-bearing
water voxels. A phagocyte also cannot consume if it is below the low-nutrient
cutoff. Main sim also briefly marks newly split children as split-deferred to
avoid asymmetric nutrient transfer. Adult Embryocyte pods born from a normal
phagocyte parent can have zero reserve; before the fix, GPU death scan could kill
that empty newborn pod during the defer window, before nutrient transport was
allowed to fill it. Startup-reserve pods survived because they inherited reserve,
which made the failure appear only after the initial burst.

Design heuristic: do not accept preview viability as proof of ecological
viability. For main-sim organisms, compute nutrient inheritance through the full
birth-to-first-reproduction path and make sure each critical organ is born with
enough survival budget to reach food. Also check whether GPU ordering creates a
short unfed newborn interval for reproductive pods.

Confidence and context: very high. Preview remains useful for geometry,
connectivity, and gross timing, but ecological claims need main-sim validation.

Genome action: the brood path was shortened and mode 8's threshold lowered so it
can become useful after modest feeding rather than ideal feeding.

Engine action: empty newborn Embryocytes now get a short feed grace before death
scan treats zero reserve as fatal. This preserves the normal rule that unfed eggs
die, but lets attached adult eggs survive long enough for nutrient transport to
start after GPU split deferral clears.

### High-count performance regressions

Observation: a LUCA population around 50k cells could feel heavier than older
100k-cell complex-organism runs, which contradicted a simple "cell count alone"
explanation.

Mechanism: the collision broadphase had been changed to dispatch a per-pair
same-bucket path using roughly `active_slots * 60` threads per physics step. At
50k active slots, that means about 3 million collision lanes before the rest of
the physics, adhesion, nutrient, light, and lifecycle work. The path also assumed
pair-bearing buckets were packed at the front of `occupied_grid_cells`, but that
list is atomic append order, so sparse worlds paid a huge multiplier and could
still miss some same-bucket pairs.

Design heuristic: when an organism appears to be "killing performance," first
separate organism traits from engine dispatch shape. A small multicell organism
can expose regressions by creating many active slots, many divisions, or a
particular feature gate, but the decisive question is whether any pass scales
with capacity, high-water slots, pair lanes, or full grids instead of the actual
work present.

Engine action: collision dispatch was restored to active-slot scaling, and
same-bucket collision work now runs once per occupied bucket instead of launching
up to 120 pair lanes per bucket.

### Population-scale cost of pattern scaffolds

Observation: in a nutrient-rich LUCA run, the logged `Physics & Lifecycle` GPU
segment rose from 19.04 ms at 20,586 cells to 61.94 ms at 40,427 cells and
119.87 ms at 50,025 cells. Rendering remained about 10 ms in total, while the
physics scheduler reached its four-step catch-up cap.

Mechanism: the current GPU mode-pattern scaffold resolver launches both endpoint
passes and, for every live source matching either endpoint mode, scans every live
cell to choose a target. Its work is approximately
`(matching endpoint sources) * (all live cells)`, not simply proportional to cell
count. The Triskelion expresses Solar Ray and Brood Core endpoints throughout a
rapidly reproducing population, so a large fraction of 50k cells become sources
for a 50k-candidate scan. `max_formation_range` is not used to prune that target
search in the present GPU shader. The fixed 64 Hz scheduler then executes as many
as four normal physics/lifecycle steps on each slow rendered frame, adding a
secondary catch-up multiplier.

Design heuristic: treat mode-pattern scaffolds as population-scale features, not
only per-organism geometry. Even with spatial indexing, cost is proportional to
the number of matching sources, the number of buckets intersecting formation
range, and local bucket occupancy. Prefer short formation ranges and minimal
rules for bloom organisms, and validate the population-level source fraction
explicitly. Small preview scenes cannot expose this cost.

Confidence and context: very high for this logged Triskelion run and the current
GPU resolver. Collision density and four catch-up steps contribute additional
cost, but neither explains the resolver's explicit superlinear scan structure.

Engine action: the GPU resolver now uses the existing 128-cubed physics spatial
grid. Every active scaffold path enumerates only buckets intersecting the rule's
`max_formation_range`, applies an exact spherical distance check, and then applies
genome, selector, ancestry, and current-component filters. Dispatch width now
uses the active cell-slot high-water mark instead of configured capacity. For the
Triskelion's 5.5-unit range in a 400-unit world, the bound is 125 buckets times 16
fixed occupants, or 2,000 candidate slots per matching source rather than 50,000
live cells. Cells temporarily in an overflow bucket or newly born after the grid
build defer scaffold formation to a later frame; they do not trigger a global
fallback scan. CPU preview now applies the same formation-range boundary so an
out-of-range rule cannot appear viable only in preview. A later main-scene run
showed the whole frame near 27 ms at 49,593 cells; the roughly 50 ms point moved
to about 105,000 cells, where `Physics & Lifecycle` accounted for roughly 35--45
ms across three to four fixed-timestep steps. This confirms the local search
removed the original 50k-cell cliff while leaving per-frame source count as the
next scaling term. The resolver now divides sources into eight deterministic
render-frame phases. Both endpoint passes use the same phase, persistent bonds
remain active between visits, and every eligible source is reconsidered within
eight frames. This bounds new-bond latency to a small fraction of a second while
reducing average scaffold search work by approximately eightfold.

Verification after source sharding: the next main-scene run measured 21.28 ms
for the whole frame and 5.97 ms for `Physics & Lifecycle` at 50,182 cells and two
fixed steps. At 99,864 cells the values were 43.48 ms and 30.75 ms across three
steps; at 124,578 cells they were 62.50 ms and 41.58 ms across four steps; and at
145,058 cells they were 83.33 ms and 67.48 ms across four steps. The original
50k-cell scaffold cliff is therefore resolved. The remaining knee begins around
75k--100k cells, where slower frames request three and then four catch-up steps.
Once the four-step cap is sustained, frame delay and simulation catch-up form a
feedback loop, while the cost of each physics/lifecycle step also continues to
rise with population. Treat these as two separate optimization targets: reduce
the remaining per-step population scaling, then prevent catch-up policy from
turning overload into persistently multiplied work. Confidence is high for the
timing curve and bottleneck classification, but this aggregate timer does not
yet identify the dominant subpass inside physics/lifecycle.

Engine follow-up: the performance timer now separates repeated
`Physics/Lifecycle Steps` from once-per-render `Physics Frame Maintenance`, which
contains component labeling, scaffold matching, particle maintenance, tools, and
render-buffer synchronization. Cell-indexed nutrient, force, and torque clears
stop at the persistent active-slot high-water mark. Spatial-grid reset now walks
the dense list of buckets occupied by the preceding step rather than clearing all
2,097,152 bucket counters. Adhesion cleanup launches over the maximum connection
range reachable by used cell slots rather than the configured world capacity,
while retaining the shader's exact GPU-side adhesion-count guard. Cosmetic
division-audio candidate collection runs only on the final fixed step encoded for
a rendered frame instead of scanning every cell on every catch-up step. Fixed
timestep frequency, the four adhesion constraint iterations, lifecycle cadence,
and the four-step catch-up limit remain unchanged, so a follow-up timing change
measures removed redundant work rather than reduced simulation fidelity.

Verification after the bandwidth and dispatch pass: in the 125k--149k population
bucket, average whole-frame time fell from 78.75 ms to 61.98 ms (21%), average
GPU time from 80.51 ms to 67.60 ms (16%), and the comparable combined physics
work from 63.70 ms to 51.83 ms (19%). The new breakdown attributes an average
46.46 ms to repeated `Physics/Lifecycle Steps` and only 5.37 ms to `Physics Frame
Maintenance` at an average 135,758 cells. The next optimization target is
therefore inside each fixed step--collision, adhesion solving, nutrient work, or
lifecycle scans--rather than scaffold/label/render synchronization. GPU timing
readback intentionally lags several rendered frames, whereas the logged cell
count and step count are current; do not over-interpret a single rapidly growing
row as an exact same-frame ratio. Population-bucket averages and sustained
plateaus are the reliable evidence. No grid, adhesion, validation, capacity, or
lifecycle errors accompanied this run.

Low-risk fixed-step audit: nutrient epoch population was a full 128-cubed voxel
pass inside every catch-up step even though fluid time advances only once per
rendered frame. Consumed voxels remain marked for the rest of their epoch, so
the second through fourth evaluations in one rendered frame had identical time,
inputs, and results. The pass now runs once before the frame's fixed-step batch;
phagocyte consumption still runs on every simulation step. Specialized
phagocyte, moss, photocyte, luminocyte-sensing, occupancy, glow, and death-effect
passes now use the observed persistent cell-slot high-water mark plus 8,192
newborn slots rather than configured capacity, retaining a conservative margin
for asynchronous count readback. Render-buffer copies use twice the observed
high-water mark, which is the maximum one division scan can produce, capped by
capacity. Collision endpoints from the immediately preceding spatial build no
longer repeat per-pair death, mass, and bounds reads; source liveness is hoisted
once per cell. Dead slots are rejected before grid-coordinate computation, and
cells without a live non-scaffold transport bond stop after metabolism rather
than performing two additional twenty-slot nutrient scans. These changes do not
alter timestep, solver iteration count, lifecycle cadence, nutrient epoch state,
or active-bond transport behavior. A main-scene timing run remains required to
measure the gain and validate that the 8,192-slot asynchronous guard is ample for
the fastest observed reproduction burst.

Genome implication: the Triskelion's single Ray-to-Core pattern brace remains a
population-level cost, but it is now locally bounded and its formation range has
real performance meaning. A follow-up 50k main-scene run should compare the same
logged segment against the 119.87 ms baseline.

### Brood gland phase locking

Observation: a self-renewing brood gland with nonzero split yaw rotated every
cycle until it detached from the body.

Mechanism: child orientation is compounded as `parent * split_rotation *
child_orientation`. With `parent_split_direction: [0, 75]` and identity
`child_a.orientation`, every retained `m5 -> m5` gland inherited another +75
degrees of phase.

Design heuristic: any infinite self-renewing organ that is intended to stay
attached should usually cancel its own split rotation on the retained child. Use a
counter-orientation for the retained stem/gland child, while allowing the
disposable bud child to point outward. Deliberate rotating organs are still valid
if the rotation is part of their function and their bonds are designed to tolerate
it.

Confidence and context: high for attached glands, roots, stalks, and regenerative
nodes. Conditional for free-floating chains, spiral dispersers, or rotating
developmental structures.

Genome action: mode 8 retained child uses the inverse 75-degree yaw quaternion:
`[0.0, -0.6087614, 0.0, 0.7933533]`.

General procedure:

1. Identify any mode that can repeat indefinitely or many times while remaining
   part of the same body: glands, stalk tips, roots, feeding loops, reproductive
   nodes, regenerative anchors, and permanent motors.
2. Compute the net retained-child orientation:
   `net = split_rotation * retained_child_orientation`.
3. If the organ is not intended to rotate, keep `net` near identity. The usual
   retained-child setting is:
   `retained_child_orientation = inverse(split_rotation)`.
4. If the retained child should keep a deliberate fixed offset `target`, use:
   `retained_child_orientation = inverse(split_rotation) * target`.
5. Check the inherited body bond. The retained child must be the child that
   keeps the body-facing adhesion, either by zone inheritance or explicit
   keep-adhesion settings.
6. Check for positional walking separately from rotational drift. Even a
   phase-locked child can migrate away if the retained child is always spawned on
   the outward side of the split axis. When the organ should stay rooted, make
   the inward/body-facing daughter the retained self-renewing child and the
   outward daughter the disposable bud.

Failure signature: the organ looks correct for the first few cycles, then each
repeat changes the attachment angle until the body bond stretches, crosses an
unfriendly inheritance zone, or tears under physics. This is accidental phase
drift, not a nutrient failure.

Useful exception: intentional spirals, sweepers, screws, dispersal whips, and
rotating developmental probes may want accumulated phase. In those cases, design
the bond network around the rotation instead of canceling it.

### Locomotion threshold

Observation: Flagellocytes with swim force below 2 are ineffective for organisms
larger than roughly two or three cells.

Mechanism: thrust has to overcome drag, adhesion coupling, body inertia, and
nutrient-search requirements. A decorative or weak motor may move a tiny body in
preview but cannot reliably pull a multicell organism through resource patches.

Design heuristic: for mobile multicell organisms, treat `swim_force >= 2.0` as a
practical starting point. Use higher force when the body has more cells,
asymmetrical drag, or needs to actively search for nutrients. Lower values can still
be useful for tiny bodies, steering trim, weak drift bias, or non-locomotor
signal/shape roles.

Confidence and context: high for active nutrient-searching bodies above two or
three cells; conditional for micro-organisms, passive drifters, or ballast-based
movement strategies.

Genome action: LUCA's mode 3 tail uses a fallback speed of `2.6` and a
light-signaled speed of `3.25`.

### Explicit signal-backbone development

Observation: after the signed signal overhaul, an adhesion network can be
mechanically intact while its signal-dependent organs remain functionally
disconnected. Sources only propagate across bonds that were classified as
backbone bonds when those bonds were created.

Mechanism: `adhesion_settings.creates_backbone` is a property of the operation
that creates a bond, not a live property that retrofits an existing adhesion.
The creating parent also pays a one-time construction cost before its nutrients
are divided. An unaffordable intended backbone bond is not created as a physical
bond at all.

Design heuristic: decide which developmental edges are the organism's nervous or
regulatory skeleton while drawing the body plan. Mark the parent modes that
create those edges, and budget enough pre-division nutrients for the bond as well
as the daughters. Do not assume that nearby cells, inherited mechanical
connectivity, or a later mode switch will repair omitted wiring.

Confidence and context: very high for the current signed cached-backbone system.
The exact construction price and attenuation may change, but creation-time
classification is an authoritative semantic unless the engine design changes.

Genome action: every developmental split in `LUCA Pelagic Signal Foundation`
that joins adult tissue creates a backbone bond. Mode 7's reserve signal on
channel 8 can therefore gate mode 8 reproduction, and mode 2's light signal on
channel 0 can modulate mode 3 locomotion.

Verification: the genome loads under the signed schema without deprecated hop
fields, and the preview publishes the reserve signal at the brood gland and the
light signal at connected motor tissue.

### Topology-aware organ queues

Observation: the new forked LUCA reached a mature brood gland with three active
structural adhesions but never budded an egg while `max_adhesions` was two. The
same gland began reproducing after its cap was changed to four.

Mechanism: the division adhesion gate requires the current active count to be
strictly less than `max_adhesions`. Adhesion inheritance and sibling bonds can
give a mature organ more body-facing bonds than its apparent single connection
in the developmental sketch. A cap below the mature structural degree disables
the organ; a cap more than one above it permits multiple simultaneous egg sinks.

Design heuristic: derive queue caps from observed mature topology rather than a
fixed magic number. For a one-egg queue, start with:

`max_adhesions = maximum mature structural adhesions + 1 egg adhesion`

Count inherited, duplicated, scaffold, and dynamic bonds in the actual preview
or main simulation. Recompute this budget whenever the body topology changes.

Confidence and context: high for division-gated buds in the current engine.
Conditional for organs that use a different release mechanism or intentionally
support several simultaneous propagules.

Genome action: mode 8 uses `max_adhesions: 5`: four observed structural bonds in
the cross-braced adult plus one attached embryo slot.

Verification: before the first topology-aware change, the preview stopped at two
ten-cell adults with no offspring. A cap of four restored reproduction in the
unbraced fork. After adding the eye-to-gland 3D cross-brace, a cap of five kept
the same one-egg queue. The 45-second nutrient-rich preview produced all core
tissues, embryos and later-generation founders, reaching the 256-cell preview
capacity.

### Intentional branching and evolutionary affordances

Observation: angled divisions alone can make a linear developmental sequence
look curved without creating a genuinely branched body plan. A LUCA intended as
an evolutionary foundation benefits from multiple functional arms whose traits
can vary semi-independently.

Mechanism: morphology is determined by both split orientation and developmental
graph topology. A sequence in which only one child continues development remains
a decorated spine. A fork in which both daughters lead to different downstream
modules creates real topological and functional branching.

Design heuristic: give foundational organisms at least one early developmental
fork when a branched body is part of the design intent. Put coherent modules on
the arms rather than distributing cell types randomly. Preserve a small number
of dormant or weakly expressed modes only when they offer plausible evolutionary
routes; dormant modes do not count as present-day body complexity.

Useful modular contrasts include feeder versus motor, solar versus sensory, and
reserve versus brood. Their parameters can mutate independently while the fork
continues to express an intentional organism. Symmetry is optional: controlled
asymmetry often gives locomotion a front, a rear, and meaningful steering
consequences.

Confidence and context: high as a design heuristic, not an ecological law.
Linear worms, filaments, stalks, and chains can be excellent intentional forms.
The warning is against accidental linearity when the stated goal is a branched
foundation.

Genome action: `LUCA Pelagic Signal Foundation` builds a posterior signal-driven
tail and a bilateral fork. One arm resolves into a Solar Sail and Nutrient Eye;
the other resolves into a Reserve Lobe and Phase-Locked Brood Gland. Mode 9 is a
dormant vascular option for mutation rather than a claimed adult organ.

### Silhouette grammar before functional variety

Observation: the first branched LUCA read as a discordant, unsymmetrical cluster
even though its modes had distinct functions and its graph was not a chain.

Mechanism: topological complexity does not automatically create visual order.
Mixed branch lengths, unrelated axes, many terminal roles, cross-braces, and a
broad palette can erase the repetition by which an observer recognizes a body
plan. A technically branched organism can therefore look procedurally random.

Design heuristic: choose a silhouette grammar before assigning most cell roles.
For a symmetric design, name the symmetry group, construct one repeated module,
and generate its siblings through a deliberate transform. Keep repeated modules
identical in role, scale, color family, and radial distance until the primary
shape reads clearly. Introduce asymmetry afterward, and only where it provides a
legible axis, behavior, or evolutionary affordance. Functional diversity should
be grouped into organs rather than scattered cell-by-cell.

Symmetry is not a universal requirement. Deliberately asymmetric swimmers,
bottom dwellers, parasites, and damaged or colonial organisms can be excellent
designs. The stronger rule is that departures from repetition should have an
explainable morphological purpose.

Confidence and context: high for foundational organisms that must communicate an
intentional body plan at a glance. Conditional for organisms whose ecological
story specifically calls for irregularity.

Genome action: `LUCA Triskelion Ray` uses C3 rotational symmetry. One meristem
repeats a single green Solar Ray three times at 120-degree phase offsets around a
shallow cone. A red Brood Core occupies the center and one blue motor establishes
the axial direction. Dormant evolutionary modes do not appear in the founding
adult and therefore do not muddy its silhouette.

Verification: the schema regression fixes the repeated meristem at three splits,
one repeated ray fate, and one retained 120-degree transform. Preview judgment
should still inspect front, side, and oblique views because a correct transform
can be visually obscured by physics or adhesion inheritance.

### Reproductive maturity as a deadline budget

Observation: the first LUCA failed to reproduce at all, while the intended
contract required a complete growth cycle and reproductive capability in under
30 seconds.

Mechanism: reproduction time is the sum of several serial and conditional
delays: embryo hatch, developmental splits, nutrient accumulation, organ
maturation, brood division, egg filling, release, and descendant hatch. A single
unmet signal gate or an adhesion cap equal to the organ's mature degree can turn
a slow pathway into a permanently disabled one.

Design heuristic: treat time to first viable descendant as an explicit design
budget. Add the authored split intervals along the critical path, then reserve
margin for nutrient transfer, scheduler cadence, and physical settling. For a
30-second external deadline, target the deterministic path well below 20 seconds
rather than exactly 30. Validate a descendant developmental mode, not merely the
appearance of an egg. Foundation genomes may use signals to modulate movement or
fecundity, but should avoid making one newly introduced signal the sole gate on
baseline reproduction unless that dependency is the experiment being tested.

Confidence and context: very high when fast maturity is part of the organism's
acceptance contract. The actual target is ecological: long-lived specialists can
legitimately mature slowly, while bloom organisms may need much shorter budgets.

Genome action: the Triskelion uses a 1.8-second initial hatch, 1.5-second axis
split, three 1.6-second meristem cycles, a 3.0-second brood cycle, and a short egg
release timer. The Brood Core has ample adhesion headroom and baseline
reproduction is not signal-gated. Light signal instead modulates the axial motor,
so the overhauled signal system remains evolvable without becoming a single point
of reproductive failure.

Verification: the deterministic 28-second CPU lifecycle regression requires a
mature Brood Core, axial motor, at least three Solar Rays, live authored scaffold
bonds, and at least two later-generation mode 0 founders.

### Staged three-dimensional scaffolding

Observation: a development graph can be genuinely branched yet still collapse
into a visually flat organism when all authored split directions use yaw alone.
The simulation supplies full quaternion inheritance, pitch and yaw split axes,
and persistent scaffold rules, so planar authoring leaves useful morphology
untapped.

Mechanism: every developmental split axis is evaluated in the parent's inherited
genome frame. Combining nonzero pitch and yaw at different generations creates
non-coplanar axes; later orientations compound unless deliberately canceled.
Scaffold rules can then join already-developed endpoints into diagonal braces,
turning a tree-shaped developmental graph into a mechanically closed 3D frame.
Those braces also change mature adhesion degree and may affect division gates,
nutrient paths, and signal topology depending on the creating endpoint.

Design heuristic: design a 3D organism in morphological stages:

1. Establish a primary axis that gives the organism a front, rear, dorsal, or
   ventral bias.
2. Create an early fork across a different axis.
3. Give downstream arms non-coplanar pitch/yaw directions rather than mirroring
   every split in one plane.
4. Let terminal modules occupy distinct volumes around the core.
5. Add only the cross-braces needed to preserve the intended volume or load
   path, then recompute adhesion budgets and check for constraint conflict.

Use at least two demonstrably non-coplanar developmental axes when the design is
meant to exploit 3D. Preview the organism from several camera angles; a shape
that only reads well from one view may still be an accidental plane. Closed
frames, tetrahedral clusters, helices, cages, radial crowns, offset fins, and
layered shells are useful vocabularies, but they are not mandatory templates.

Confidence and context: high as an authoring principle for volumetric creatures.
Conditional for organisms intentionally adapted to surfaces, films, cave walls,
interfaces, or flat light-collecting mats.

Genome action: the Triskelion first establishes an axial motor, then pitches its
three repeated Solar Rays 35 degrees off the axis while advancing them in
120-degree yaw steps. Developmental adhesion inheritance closes the three rays
into a triangular crown. A pattern scaffold then adds three Ray-to-Brood-Core
spokes, converting the crown into a shallow tetrahedral frame. The authored
brace intentionally targets pairs that do not already have developmental bonds;
the scaffold resolver will not duplicate an existing connection.

Verification: the schema test requires the threefold repeated meristem and one
Ray-to-Core scaffold rule. The 28-second lifecycle test requires at least one
live bond carrying that scaffold rule while the body matures and reproduces.

### Detached hatch siblings and scaffold scope

Observation: in the GPU scene, the two free organisms hatched from one LUCA egg
later formed a scaffold adhesion that pulled them back together. The same genome
kept them separate in CPU preview.

Mechanism: GPU division assigned both hatchlings the Embryocyte parent's
developmental organism scope because the Embryocyte was itself the genome's
initial mode. The GPU scaffold resolver correctly restricts matches by that
scope, but the shared stale identity made tissues in the two detached descendants
look like members of one organism. CPU preview masked the defect by recomputing
organism scopes from connected developmental adhesion components after division.

Design heuristic: when one propagule hatches into multiple detached founders,
each founder must receive a fresh developmental scope before any lineage- or
mode-based scaffold resolution runs. Physical separation is not sufficient;
identity data used by scaffold, lineage, skin, or organism-level systems must
split at the same lifecycle boundary. Treat preview/main differences involving
impossible cross-organism bonds as identity-scope evidence before weakening an
otherwise intentional scaffold.

Confidence and context: very high for free Embryocyte hatches in the current GPU
pipeline. Conditional for divisions whose daughters remain connected directly
or through inherited body bonds.

Engine action: GPU lifecycle division now marks both children of an Embryocyte
hatch with no sibling bond as new developmental organisms, including when the
Embryocyte is the genome's initial mode. A deliberately bonded hatch and ordinary
initial-mode self-renewal retain their existing scope.

Verification: the LUCA regression suite checks that the GPU lifecycle shader
contains the Embryocyte-specific fresh-scope rule. Main-scene confirmation should
verify that the two hatchlings develop independent scaffold frames and never
acquire a cross-organism brace.

### Scaffold scope requires ancestry and current connectivity

Observation: unrelated organisms were pulled together by scaffold adhesion again
after the Embryocyte-specific fresh-identity fix. The specific hatch boundary was
correct, so that patch did not explain every way a lineage could become physically
independent.

Mechanism: a development address answers where a cell came from; it does not by
itself answer whether the cell is still part of the same organism. GPU scaffold
matching used the persistent ancestry ID alone. CPU preview first recomputed
organism components through live normal adhesions, excluding scaffold bonds, and
therefore refused matches across a detached component. Any later fragmentation
could retain shared ancestry on the GPU and be incorrectly reconnected even when
the original hatch identities were distinct.

Design heuristic: scaffold endpoints should satisfy two independent scopes:

1. Compatible developmental ancestry, for deterministic lineage addressing.
2. Membership in the same current component of normal developmental adhesions,
   for physical organism identity.

Never let the scaffold bonds being tested define the component test themselves;
that makes an erroneous cross-link self-validating. Delayed scaffold formation is
preferable to a false cross-organism bond while connectivity labels converge.

Confidence and context: very high for the present GPU/CPU discrepancy. The
principle applies to any persistent ancestry system whose organisms can hatch,
bud, fragment, shed organs, or break adhesions.

Engine action: the GPU scaffold resolver now intersects development-address scope
with the component labels produced from normal adhesions. Scaffold resolution is
encoded after the component-label pass and runs once per rendered frame instead
of once per physics substep. Structural barrier-ball bonds remain excluded from
component labeling.

Verification: the LUCA regression checks that the GPU scaffold shader binds the
normal-adhesion component labels, includes them in endpoint scope matching, and
that the scene schedules scaffold matching after component labeling. A fresh
main-scene run remains the decisive behavioral check.

## First-Pass Design Procedure

1. Define the ecological contract: where food comes from, where light comes from,
   whether the organism drifts, anchors, swims, grazes, or cycles niches.
2. Define the launch path separately from the adult reproduction path.
3. Compute the nutrient budget from initial mode to first feeder tissue.
4. Compute the nutrient budget from first feeder tissue to first released child.
   Check whether one reproductive node can accidentally create multiple attached
   nutrient sinks before the first one releases.
5. Check every repeating organ for phase drift. Infinite self-renewal usually
   needs a retained-child orientation that cancels unwanted accumulated rotation;
   deliberate rotating organs need bond geometry designed to tolerate the drift.
6. Check adhesion inheritance. `split_ratio` controls bond-zone inheritance, not
   nutrient allocation.
7. Verify that motors are scaled to body size. Weak Flagellocytes are noise on
   multicell bodies.
8. Draw the developmental graph, not only the split angles. Confirm that any
   intended branch has two daughters with meaningful downstream fates.
9. Choose a silhouette grammar: symmetry group or intentional asymmetry,
   repeated module, restrained role palette, and the purpose of each exception.
10. Stage the body in 3D: primary axis, non-coplanar fork axes, terminal volumes,
   then minimal cross-bracing. Inspect it from multiple views.
11. Mark every intended signal-carrying bond at its creation point and include
   backbone construction in the division nutrient budget.
12. Derive reproductive queue limits from the mature organ's observed structural
   degree, then add only the number of simultaneous propagules intended.
13. Budget the complete path to a viable descendant and test it against the
   maturity deadline with margin.
14. Treat light as a supplement unless the environment guarantees it. A robust
   LUCA should not require regular light to reproduce.
15. After every test report, update this document before or alongside genome
   changes, framing the lesson as weighted evidence rather than a universal rule.

## LUCA Triskelion Ray Current Intent

Initial mode: mode 8 `Triskelion Embryo`.

Startup life cycle: the placed Embryocyte releases two detached mode 0 `Axis
Founder` cells. Each receives a separate developmental organism identity and
builds independently.

Adult life cycle: mode 0 divides into a continuing mode 1 `Triradial Meristem`
and mode 4 `Axial Signal Motor`. The meristem emits three identical mode 2 `Solar
Ray` cells while rotating its retained developmental frame by 120 degrees after
each split. It then resolves into the central mode 3 `Brood Core`, which buds
detached mode 8 embryos while retaining itself. Released embryos repeat the
founding lifecycle.

3D body plan: the motor defines the longitudinal axis. Three Solar Rays occupy a
35-degree shallow cone with C3 rotational symmetry. Their inherited developmental
bonds form a triangular crown, while authored Ray-to-Core braces pull the Brood
Core into the volume and produce a shallow tetrahedral scaffold. This is a
radial body plan with an axial exception, not a linear chain or an arbitrary
cluster.

Signal intent: every Solar Ray publishes channel 0 in light. The axial motor has
a viable fallback swim force and increases its drive when that light signal is
present. Baseline reproduction does not depend on the signal, leaving mutations
free to evolve stronger coupling without making the founding organism sterile.

Evolution intent: modes 5 through 7 provide dormant sensory, reserve, and
vascular destinations. They do not participate in the founding silhouette, but
mutations can recruit them into a ray, core, or axial lineage. The repeated ray
module gives evolution three comparable surfaces on which changes can preserve,
break, or elaborate symmetry in a legible way.

Timing intent: the full deterministic critical path is budgeted well below 30
seconds. The regression test observes mature rays, core, motor, scaffold bonds,
and second-generation founders by 28 seconds in the CPU preview. Main-simulation
testing remains authoritative for sustained nutrient economics and the final
visual read under GPU physics.
