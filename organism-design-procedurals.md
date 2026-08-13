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

Genome action: `LUCA Pelagic Foundation` now uses `initial_mode: 7`, where mode
7 is `Fed Embryo Pod`. A placed organism hatches into mode 0 founders after the
embryocyte hatch interval, while later reproduction still uses the fed brood
pathway.

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

Genome action: mode 5 `Brood Grazer Gland` buds mode 7 `Fed Embryo Pod`; mode 7
uses a reserve threshold so reproductive embryos should release after real
nutrient accumulation.

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
mode 5 `Brood Grazer Gland` to `nutrient_priority: 3.2`, raised its split buffer
to `split_mass: 1.24`, set `max_adhesions: 2` so one body bond plus one egg blocks
additional egg budding, and reduced mode 7 `Fed Embryo Pod` priority to `2.5`.

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

Genome action: the brood path was shortened and mode 5's threshold lowered so it
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

Genome action: mode 5 retained child uses the inverse 75-degree yaw quaternion:
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

Genome action: LUCA's mode 2 tail was raised to `swim_force: 2.5`.

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
8. Treat light as a supplement unless the environment guarantees it. A robust
   LUCA should not require regular light to reproduce.
9. After every test report, update this document before or alongside genome
   changes, framing the lesson as weighted evidence rather than a universal rule.

## LUCA Pelagic Foundation Current Intent

Initial mode: mode 7 `Fed Embryo Pod`.

Startup life cycle: placed Embryocyte hatches into two mode 0 `Founder Scout`
cells, carrying inherited reserve into the founder population.

Adult life cycle: feeder body grows from mode 0 through the grazing spine,
develops the crown and phase-locked brood gland, feeds an attached mode 7
Embryocyte, then releases a fed embryo only after reserve threshold is reached.

Design intent: use embryocyte reserve for initial viability, but require real
feeding for sustained reproduction.
