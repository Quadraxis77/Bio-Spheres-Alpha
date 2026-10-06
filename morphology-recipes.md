# Morphology Recipes — Living Notebook

## Purpose

Learn how to construct intentional organism body shapes in Bio-Spheres from
user-authored genomes, one example at a time. Each lesson connects concrete
genome settings to the developmental steps that produce a physical structure.
Over time, these recipes should support building organisms on demand.

This file is the persistent record of those lessons. Update it as the user teaches
new patterns, explains settings, reports results, or corrects an interpretation.

**Genome settings → developmental sequence → physical structure**

## Primary design objective — compact, reusable, mutation-resilient genomes

**User instruction, 2026-09-12:** The ultimate design goal is a compact, highly
efficient genome with as few modes as possible and as many reusable modes as
possible, while allowing hardening and adaptability in the face of mutation.

Prefer the smallest reusable developmental program that achieves the intended
organism behavior. Mode count is an optimization objective alongside function,
robustness, and adaptability, rather than a reason to sacrifice them blindly.

- Begin with repeated/self-referencing modes, inherited rotations, split counters,
  and reusable mode sequences. Add distinct modes when they provide a needed
  behavior or a justified robustness benefit.
- Reuse modes across compatible structural positions and functions. Individually
  addressable corners or branches are optional tools, not the default architecture.
- Evaluate genome efficiency separately from simulation cost: fewer modes do not
  automatically imply fewer cells, bonds, divisions, or cheaper physics.
- Treat hardening as preserving essential structure and function under relevant
  mutations. Record which parameters and transitions are sensitive, and test
  perturbations before claiming robustness. The specific mutation operators and
  acceptable failure rates remain to be established.
- Treat adaptability as allowing viable variation, not merely preserving an exact
  outline. Judge changes in a moving 3D body by survival and intended functions.
- Examine shared-mode tradeoffs: one mutation can affect many positions at once.
  Reuse can coordinate useful changes, but can also spread a failure. Separate
  modes or add redundant controls only when the demonstrated benefit justifies
  the extra complexity.

For each recipe, record the minimum demonstrated mode count, what can be reused,
which extra modes are optional, and known mutation sensitivities. Compare compact
variants before expanding the developmental graph.

### Reusable anatomical sequences

**User clarification, 2026-09-12:** Reusable mode sequences represent anatomical
parts such as an eye, a foot, or a leg. The body plan can invoke the same sequence
at multiple locations instead of duplicating its modes for every instance.

The body plan establishes where a part begins, its entry mode, orientation, and
attachment context. The shared sequence then develops that part using the same
mode definitions at each occurrence. For example, several limb locations can
enter one leg sequence, and multiple legs can lead into one shared foot sequence.
These are architectural examples, not yet demonstrated anatomy recipes.

Reuse therefore applies to whole developmental subgraphs, not only repeated
splits within a single mode. Instances have separate cell state while sharing
genome instructions. Local nutrient supply, signals, connections, and inherited
orientation can affect whether otherwise identical sequences develop alike.

For each reusable anatomical recipe, document:

- Entry mode and required starting state, including relevant counter/timer resets.
- Local orientation convention and how the body plan places and aims each instance.
- Expected attachment and bond inheritance at the connection to the body.
- Internal sequence, termination, and any downstream part sequences it invokes.
- Required nutrients/signals and possible interference between repeated instances.
- Which mutations affect every instance because they share the same mode definitions.

Prefer referencing an existing compatible part sequence over copying its modes.
Create a distinct variant only when a required difference cannot be supplied by
the entry context or when isolation provides a justified functional or robustness
benefit. Rotating a shared part is not automatically equivalent to mirroring its
handedness; verify that separately for left/right anatomy.

This instruction supersedes any older project-note prescription that every
structural backbone position must have a unique mode. The explicit-corner square
variant below is an optional specialization strategy; the one-mode square is the
preferred baseline when it meets the organism's functional requirements.

### Terminal-node addressability through gradients and Stemocytes

**User lesson, 2026-09-12:** Addressability is a key limitation of shared modes:
we need to assign an anatomical sequence to a specific place in the body without
giving every structural position a unique mode. Gradients and Stemocytes are one
way to implement this.

A shared Stemocyte mode can read different local gradient strengths at different
body locations. Its response bands select different outcomes, including in-place
transition to the entry mode of a reusable anatomical sequence. Thus cells sharing
the same genome instructions can begin different anatomy according to their
local signal context.

```
Body construction → positioned signal sources and connected routes
                  → local gradient strength at shared Stemocytes
                  → response band selects an anatomical entry mode
                  → shared anatomical sequence develops at selected sites
```

This separates three responsibilities: structural modes build the body,
gradients and Stemocyte responses distinguish attachment regions, and anatomical
sequences build the parts. Dedicated structural modes are not required for every
site that a gradient can reliably distinguish.

Implementation basis: the current Stemocyte reads one channel (8–15), applies its
selected signed response, and selects among five strength bands separated by four
thresholds. Each band can remain Stemocyte, enter apoptosis, or switch to a target
mode. Developmental delay options control when differentiation becomes eligible.
Remaining a Stemocyte or awaiting a delay does not by itself prevent ordinary
division; that requires the appropriate division settings.

**Design implications to verify with authored examples:**

- A gradient supplies a contextual address, not a unique cell identifier. Sites
  with the same interpreted signal and developmental state cannot be distinguished
  by that Stemocyte's band selection alone. Symmetric sites may intentionally
  start the same anatomical sequence.
- Signal strength follows active routes and attenuation through the connected
  body; it is not simply Euclidean distance from the source. Folding alone does
  not establish a new signal connection.
- Source placement, route topology, strength, thresholds, and evaluation timing
  jointly determine which regions receive each anatomical assignment.
- Wait for an appropriate developmental state before committing anatomy when the
  gradient changes during construction; identify and test that state explicitly.
- For mutation hardening, measure separation from band boundaries and sensitivity
  to changes in source strength, routes, and thresholds. Do not assume an address
  stays stable because it worked in one unmutated body.

No specific source placement, band values, or anatomical routing genome has been
taught or tested yet. Those belong in subsequent authored lessons.

## LUCA requirement — rudimentary, evolvable internal signaling

**User instruction, 2026-09-13:** LUCA should contain rudimentary internal
signaling with a sensor and something that reacts to its output. The system must
be general and flexible enough for mutation to meaningfully evolve it over time.

Start with a small, functioning sensor–response circuit:

```
Sensed condition → signal source → connected signal route → responding tissue
```

Both ends must have a demonstrable role: the sensor changes its output with the
sensed condition, and the receiver changes behavior in response. Merely assigning
signal fields or adding a constant emitter does not demonstrate sensing. Extra
processing modes should remain minimal. The user's subsequent logic-center
requirement below extends the preferred design beyond a direct sensor–response link.

Prefer reusable sensing and responding sequences with compatible channels and
clear local orientation conventions. Use the body's actual signal routes and
attenuation when choosing receiver thresholds. Shared channels can intentionally
coordinate multiple responders; document coupling so unintended interference is
recognizable.

**Design implications to verify:**

- Choose an initial response with observable functional consequences, such as a
  change in locomotion or contraction. No specific sensor/actuator pairing has
  been selected by this lesson.
- Preserve meaningful variation in sensor configuration, emitted strength,
  receiver response, thresholds, placement, and routing. Determine which of these
  are reachable through the actual mutation operators before claiming evolvability.
- Avoid configurations whose useful behavior exists only at one exact parameter
  value. Seek ranges of viable behavior with scope for altered sensitivity,
  response strength, or coordination.
- Balance robustness and adaptability: the circuit should tolerate relevant small
  changes while allowing some mutations to change behavior usefully. Do not assume
  arbitrary mutations improve function or preserve channel compatibility.
- Favor the fewest shared modes needed for a real circuit; add signal processing
  or dedicated variants when they supply a demonstrated capability.

**Verification:** Show sensor output and receiver behavior under contrasting sensed
conditions. Then perturb settings using representative engine mutations and record
whether behavior remains functional, changes meaningfully, or fails. Immediate
behavioral variation demonstrates a possible substrate for evolution; population
tests are needed to establish adaptive improvement over generations.

### Small logic center for signal management and transformation

**User instruction, 2026-09-13:** The ideal LUCA includes a small, rudimentary
logic center that manages and transforms signals, with capacity for dynamic
evolution.

The preferred circuit is now:

```
Sensor input → small logic center → responding tissue
```

Keep the center compact and functional from the beginning. It should perform an
observable signal transformation or decision rather than merely exist as named
tissue. Cognocytes provide candidate arithmetic, comparison, Boolean, selection,
and waveform operations. Memorocytes can supply temporal integration if useful;
memory is a possible extension, not a requirement established by this lesson.

Choose the initial operation to match the available inputs. Binary Cognocyte
operations require appropriate nonzero inputs in the inspected processor, whereas
unary operations can work with a single sensor channel. Verify actual outputs,
channel compatibility, attenuation, and response thresholds rather than assuming
that a nominally connected circuit computes the intended result.

Preserve room for evolution in how signals are combined, transformed, routed,
and acted upon. Favor viable ranges of settings and reusable processing sequences
over a large circuit tailored to one exact behavior. Confirm which variations
the engine's mutation operators can produce. Additional processing stages or
memory should earn their mode cost through function or demonstrated adaptability.

**Verification:** Observe input, intermediate output, and resulting behavior under
changing conditions; then test representative mutations for functional variation
and failure. The specific layout, mode count, operation, and channels remain open.
Dynamic evolvability is a design objective, not yet a measured property.

## Preferred LUCA reproductive lifecycle

**User instruction, 2026-09-13:** The ideal LUCA starts as an Embryocyte and
develops into a mature adult whose final reproductive stage uses gametes. The
user's rationale is that this helps mutations spread more effectively through
the population.

Use this lifecycle as the default target for LUCA designs:

```
Founder Embryocyte → developing organism → mature adult
                  → gamete production and release
                  → compatible gamete fusion → offspring Embryocyte
                  → next generation
```

The founder Embryocyte supplies the initial developmental reserve. The mature
adult must acquire enough resources to support its own function and produce
viable Gametocytes; the initial reserve alone is not evidence of sustainable
reproduction. Adult reproduction should culminate in gametes rather than direct
asexual Embryocyte production as the default LUCA strategy.

The previously inspected Gametocyte lifecycle uses compatible gametes from
different organisms to produce an Embryocyte with a crossover genome. This is the
mechanistic basis for combining genetic variants from different lineages. The
claimed improvement in mutation spread is a design rationale to evaluate in
population tests, not a measured result or a guarantee that every mutation spreads.

When testing a LUCA design, follow the complete reproductive cycle: maturation,
adult-funded gamete production, release, encounter and compatibility, fusion,
offspring development, and reproduction by that offspring generation. Evaluate
mutation transmission and viable recombination alongside the robustness of shared
anatomical sequences. Specific gamete modes, release settings, and maturation
controls remain to be established through authored examples.

### Attached gamete provisioning before release

**User instruction, 2026-09-13:** A gamete should remain attached to the parent
and accumulate enough nutrients to support development of a new sexually mature
organism before it releases. Treat this attached provisioning period as an early
developmental stage of the gamete.

The design target is therefore not merely a gamete that survives detachment or
an offspring that can hatch. Budget provision for the path through offspring
development to sexual maturity. The reserve target must account for free-gamete
waiting/encounter costs, fusion and embryonic development, body construction, and
survival until maturity. Determine actual requirements experimentally rather than
choosing a release threshold solely as a percentage of storage capacity.

Use the Gametocyte's reserve-threshold release condition to keep it attached
while filling. Timer or signal conditions can additionally require incubation or
adult readiness; all enabled release conditions must pass. The parent must retain
the attachment and supply nutrients throughout provisioning. Premature bond loss
can bypass the intended release schedule and must be considered in testing.

This provisioning stage does not make a free Gametocyte develop directly into an
adult: under the recorded engine lifecycle it must still fuse with a compatible
gamete to produce an offspring Embryocyte. Fusion combines parental reserves;
record each gamete's contribution explicitly. Do not silently weaken the user's
per-gamete provisioning target by assuming a well-provisioned partner will make
up a deficit.

**Verification target:** After release and compatible fusion, follow the offspring
until it reaches sexual maturity and can enter its own gamete-production phase.
Record parental investment, reserves at release and fusion, environmental feeding
during development, and developmental losses. Reliance on external food should
be explicit rather than confused with development funded by the gamete reserve.

### Gamete provisioning must preserve parental condition

**User instruction, 2026-09-13:** Pay special attention to nutrient transfer into
gametes. The reproductive system must not consume excessive nutrients and starve
or emaciate the parent organism.

Meet the offspring provisioning target without sacrificing the adult's maintenance,
body condition, feeding, or other essential functions. A large release reserve
is not sufficient evidence of good reproduction if accumulating it depletes the
parent. Evaluate both the rate of nutrient diversion and its total cost over
repeated reproductive cycles.

**User clarification, 2026-09-13 — preferred solution:** Manage nutrient priority
and arrange Vasculocytes in the overall body plan to route nutrients appropriately.
Use this as the primary approach to balancing parental maintenance with gamete
provisioning. Additional production gates, queues, or recovery timers are optional
fallbacks if demonstrated necessary, not the default solution.

Priorities determine the relative demand and retention of tissues on eligible
transfer routes; Vasculocyte placement, transport, and exchange ports determine
the available supply paths. Design these together so feeding tissue supports the
adult's essential functions while supplying the reproductive site and attached
gamete without progressively draining the body. The aim is balanced ongoing
distribution, not simply maximizing gamete priority or nutrient throughput.

Specific priority values and vascular arrangements remain to be taught and tested.
Validate their combined behavior across repeated provisioning cycles and changing
food conditions before introducing additional control mechanisms.

**Design implications to establish through authored tests:**

- Budget provisioning against sustainable adult nutrient income and maintenance,
  rather than treating all stored adult nutrients as available reproductive surplus.
- Coordinate gamete production, nutrient transfer, and release. A release threshold
  determines when a gamete can detach; it does not by itself limit its nutrient
  demand while attached or prevent additional gametes from being produced.
- Control the number of simultaneously provisioning gametes. Multiple sinks can
  collectively drain the parent even when each is individually viable. A single
  provisioning slot is a candidate baseline, not yet a mandated implementation.
- Evaluate nutrient priorities, low-nutrient protection, vascular exchange, and
  production gates together. Do not assume high gamete priority is desirable or
  that any one setting guarantees protection in every transfer path.
- Allow reduced provisioning or delayed next-gamete production when parental
  condition or nutrient income is inadequate. A new cycle after release should
  depend on readiness, not require immediate replacement regardless of cost.
- Preserve the requirement that a released gamete is adequately provisioned;
  parental protection should not be achieved by routinely releasing underfunded
  gametes. Reconcile the two requirements through investment rate and timing.

**Verification:** Track adult tissue nutrients and size/body condition, feeding
and functional performance, gamete reserve accumulation, and reproductive cadence
over many cycles under abundant, limited, and fluctuating food. Look for progressive
parental depletion as well as acute starvation. Establish acceptable parental
reserves and recovery criteria experimentally; no transfer rates, priorities, or
threshold values are confirmed by this lesson alone.

### Repeated gamete production must preserve the reproductive site

**User instruction, 2026-09-13:** Pay special attention to the reproductive cell
that repeatedly spawns a gamete after each release. Gamete formation must not
change that cell's position relative to the organism, and repeated divisions
must not accumulate changes in its relative angle and deform the body.

Treat the continuing reproductive child as a persistent functional site even
though each division replaces its parent with two children. Design for two
separate invariants:

- **Positional stability:** the continuing reproductive cell retains its intended
  location relative to neighboring body tissue across production/release cycles.
- **Angular stability:** its orientation relative to that tissue does not acquire
  an incremental turn on each division. There must be no cumulative twist or bend
  of the reproductive site or surrounding organism.

These are body-relative requirements, not a demand that the organism remain
stationary or fixed in world orientation. Distinguish reversible deformation
during provisioning from persistent displacement or angular drift after release.

**Implementation implications, not yet a tested reproductive recipe:**

- Self-reference preserves a mode identity, not necessarily a spatial position or
  orientation. Trace which child continues the reproductive role, which produces
  the gamete, and which body bonds each inherits.
- For the continuing child, the inherited rotation is `S × C` (parent split
  rotation followed by effective child offset). With an unchanged continuing mode,
  choosing `C = inverse(S)` cancels the programmed turn per division. Check normal
  and after-split offsets and any signal-selected routing. This removes one source
  of angular drift; it does not guarantee mechanical stability under forces.
- Division places both children at offsets from the parent's position. Angular
  cancellation therefore does not solve positional displacement. Split direction,
  inherited body anchors, gamete attachment, and subsequent mechanical relaxation
  must be examined together. Do not claim exact position preservation merely
  because the continuing child keeps its bonds.
- Keep the provisioning gamete attached until its release conditions are met, and
  preserve the reproductive site's structural connections when it releases. Verify
  that the next production cycle does not progressively alter those connections.

**Verification:** Compare the continuing cell's location and orientation relative
to the same neighboring tissue before division, during attached provisioning, and
after release/relaxation. Repeat over many cycles and inspect cumulative drift,
bond topology, and deformation of surrounding tissue. One successful egg release
does not demonstrate a stable reusable reproductive sequence. Numerical tolerances
and the concrete anchoring arrangement remain to be established with examples.

## Guiding principle — design in three dimensions

**User instruction, 2026-09-12:** All creature designs must be considered in three
dimensions. Nothing we create should be expected to stay flat for long.

Treat squares, sheets, and other initially planar patterns as developmental
starting arrangements inside a 3D simulation, not as permanently flat bodies.
Growth, forces, collisions, and bond mechanics can bend, twist, fold, or otherwise
rearrange them. A named shape describes an intended construction or observed
stage; its name is not a constraint imposed by the engine.

For every recipe and future organism design:

- Trace split axes and child orientations in 3D, including out-of-plane growth.
- Distinguish connectivity from instantaneous geometry: a four-edge square loop
  can retain its connections while losing its square outline or planarity.
- Consider how deformation changes feeding, sensing, propulsion, attachment, and
  the relative positions of functional tissue.
- Evaluate the developing and moving body over time, not only a single flat view.
- If a function requires a particular spatial relationship, identify the mechanical
  support it needs and verify that relationship under motion. Do not assume it
  persists simply because the genome initially constructs it.

The 2×2 square remains a useful basic construction lesson. Its initial planarity
is not a promise of the organism's lasting shape, and loss of planarity alone is
not failure unless a specific function depends on it.

## How we use this notebook

1. The user identifies an authored genome and explains its intended shape or lesson.
2. Read the genome and resolve relevant omitted settings against the engine's
   defaults. Record the file and revision or content hash when available.
3. Trace the divisions, child orientations, mode transitions, and bond behavior
   that appear to construct the shape.
4. Explain the recipe back to the user. Distinguish user observations, mechanisms
   checked in code, and hypotheses that still need testing.
5. Incorporate corrections and record the reusable construction rule, its limits,
   and any demonstrated variations.

Do not treat a plausible explanation as a tested recipe. A successful example
demonstrates behavior under its recorded conditions; resizing or combining it with
another pattern may require another test. Preserve earlier observations and their
context when a lesson changes.

Keep authored reference genomes intact unless the user asks to change them. Link
experimental variants separately so the original lesson remains reproducible.

## Recipe index

The first construction hypothesis is user-confirmed; its authored reference genome
has not yet been identified. Alternative encodings remain predictions until checked.

| ID | Pattern | Reference genome | Evidence / status |
| --- | --- | --- | --- |
| MR-001 | Adhesion-linked 2×2 square | Awaiting user's reference | Original hypothesis user-confirmed; alternatives code-derived |

## Foundation checks and proposed testing strategy

Code inspection on 2026-09-12; these are working explanations, not user-taught or
simulation-tested recipes. Sources: [preview division](src/cell/division.rs),
[GPU lifecycle](shaders/lifecycle_unified.wgsl),
[GPU division execution](shaders/lifecycle_division_execute_ring.wgsl), and
[adhesion inheritance](src/simulation/adhesion_inheritance.rs).

- `parent_make_adhesion` requests a new bond between the two children.
- Ordinary child `keep_adhesion` flags permit inheritance of existing parent bonds,
  according to their split-relative zones. They do not disable the new sibling bond.
- On the split that reaches `max_splits`, after-split keep flags replace ordinary
  keep flags. Both must be true to permit the new sibling bond on that split.
- Bond creation and duplication also depend on available resources and capacity;
  keep/make flags express permission rather than guaranteeing every bond survives.
- Ordinary division requires sufficient age, nutrients plus reserve, remaining
  splits, an allowed adhesion count, and satisfaction of any enabled division
  signal gate. Failing any gate prevents division while that condition persists.
- Nutrient threshold is `(split_mass - 1) * 100`. Thresholds above 100 explicitly
  disable ordinary division, except Lipocytes allow up to 200. Nutrient supply and
  transport affect when a reachable threshold is met.
- `max_splits: 0` prevents division in that mode; `-1` permits unlimited splits.
  Mode transitions can reset the lineage split counter, so a finite value alone
  does not bound the entire organism's growth.
- Embryocytes hatch only when free, with their timer satisfied and splits remaining;
  they bypass ordinary nutrient, adhesion-count, and division-signal gates. Attached
  release conditions are a separate control. Gametocytes never divide.
- **Unresolved discrepancy:** preview explicitly rejects `split_interval > 59`
  for ordinary cells. The inspected GPU ordinary-cell path only compares age with
  the interval. GPU Embryocytes do enforce the >59 stop. Verify before relying on
  the interval sentinel across both environments.
- Engine scheduling and cell capacity can delay an otherwise eligible split.

### Make and keep adhesion combinations

For ordinary splits before the maximum-splits transition:

| Make adhesion | Keep A / Keep B | Expected effect, subject to bond resources and capacity |
| --- | --- | --- |
| On | Both off | Children can bond to each other but do not inherit the parent's old bonds. |
| Off | Both on | Children can inherit eligible old bonds without creating a sibling bond. |
| On | Both on | A sibling bond and inheritance of eligible old bonds can occur together. |
| Off | Both off | Neither a sibling bond nor inherited parent bonds are requested. |
| Either | Only one on | Only that child can inherit eligible old bonds; ordinary sibling bonding still follows make adhesion. |

Keep is permission to inherit bonds assigned by the adhesion zones, not a request
to transfer all of the parent's bonds to a particular child. Test the final-split
overrides separately from this table.

### Timing and stopping distinctions

- Split interval is a minimum age requirement; reaching it does not override other
  gates. A cell may split later because nutrients, connections, or signals are not
  ready.
- For ordinary cells, `split_mass > 2.0` is an explicit never-split setting;
  Lipocytes instead use `split_mass > 3.0`. This differs from a reachable threshold
  that the cell has not yet accumulated enough nutrients to meet.
- `min_adhesions` blocks division below the required active bond count.
  `max_adhesions` blocks at or above the limit. In the inspected ordinary division
  checks, the maximum gate applies even when make adhesion is off.
- Division signals use the selected response mode (positive, negative, or magnitude),
  threshold, and inversion. They gate eligibility rather than replacing the timer
  or nutrient requirement.
- Distinguish a permanent stop in the current mode from a conditional pause. For
  each conditional gate, test both failure to divide while blocked and resumed
  division after the condition becomes permissive.
- Changing child modes changes the settings used for subsequent divisions. Trace
  those transitions and split-counter resets when predicting the growth sequence.

Proposed order, awaiting discussion with the user:

1. Use a small, fixed Test-cell reference with sufficient nutrients and stable
   bonds. Observe one division at a time and record child modes and actual bonds.
2. Test make adhesion on/off for a parent without existing bonds.
3. Give a parent an existing bond, then vary keep A and keep B independently while
   holding geometry fixed. Repeat with make on/off to separate the two effects.
4. Test final-split keep overrides separately. A `max_splits: 1` parent immediately
   uses those overrides, so it is unsuitable for isolating ordinary keep flags.
5. Isolate each timing or blocking gate with every other gate satisfied. For
   conditional blocks, also demonstrate that division resumes when the gate opens.
6. Compare preview and main simulation for the same reference before promoting a
   finding to a reusable recipe. Then combine controls to study division order.

## Parent Settings panel reference

Inspected 2026-09-12. Scope: every editable control in `render_parent_settings`,
including sections shown only for particular cell types. This is a code-based
working reference, not a record of simulation experiments. UI ranges below are
editor ranges; imported genomes and mutation code can have different limits.

### What editing this panel changes

The panel edits the selected entry in `genome.modes` (`ModeSettings`). It changes
instructions used by cells in that mode, not just a single parent cell. Children
use their assigned modes for subsequent behavior. A hidden type-specific section
does not mean its stored fields were deleted.

The editor attempts to copy changed fields to other selected modes through
`sync_mode_changes_to_others`. Inspection found no copy entries for final-split
orientations, Photocyte/Lipocyte emit settings, Luminocyte signal settings,
Siphonocyte, Plumocyte, Cognocyte, or Memorocyte settings. These panel edits should
not be assumed to propagate to secondary selections. Saving writes YAML `.genome` data through the serializer,
which stores differences from defaults. An omitted field therefore requires
resolving defaults; it does not necessarily mean zero, false, or disabled.

Primary sources:

- [Panel controls and multi-selection](src/ui/tab_viewer.rs)
  (`render_parent_settings`, `signal_response_mode_control`,
  `draw_stemocyte_response_strip`, `sync_mode_changes_to_others`).
- [Genome fields and defaults](src/genome/mod.rs),
  [serialization](src/genome/serialization.rs).
- [Preview division](src/cell/division.rs),
  [GPU lifecycle checks](shaders/lifecycle_unified.wgsl),
  [GPU division execution](shaders/lifecycle_division_execute_ring.wgsl).
- [Signal processing](src/simulation/signal_system.rs),
  [Cognocyte operations](src/cell/behaviors/cognocyte.rs).
- [Nutrient transport](shaders/nutrient_transport.wgsl),
  [preview physics](src/simulation/preview_physics.rs),
  [movement and passive forces](shaders/swim_force.wgsl),
  [in-place GPU mode switching](shaders/mode_switch.wgsl).

### Shared signal response control

Each **Response** dropdown edits a separate slot of `signal_response_modes` for
that listener: Division, Apoptosis, Child A, Child B, Mode Switch, Glueocyte,
Flagellocyte, Ciliocyte, Myocyte, Embryocyte (also Gametocyte), Luminocyte,
Siphonocyte, or Stemocyte. It is not a global setting for every listener.

| Choice | Stored value | Interpretation of raw signal `s` |
| --- | --- | --- |
| Positive | 0 | `max(s, 0)` |
| Negative | 1 | `max(-s, 0)` |
| Magnitude | 2 | `abs(s)` |

Shared threshold listeners activate when the interpreted response is **positive
and at least the threshold**. Inversion negates that result. Thus zero signal
with zero threshold is not normally active; inversion can make it active.
“Above/Below” labels refer to the interpreted response, not necessarily the raw
signed value. Stemocyte band selection uses the interpreted strength directly.

### Division Settings — all cell types

| UI setting | Genome field / range | Working understanding and design effect |
| --- | --- | --- |
| Split Nutrients | `split_mass`; UI 1–100 nutrients, or 1–200 for Lipocytes, plus Never | UI stores `1 + nutrients/100`. Ordinary cells require nutrients plus reserve to meet the threshold. Raising it usually delays division and increases the resource requirement; it does not prescribe a shape directly. UI Never stores a threshold of 101 or 201, respectively. Embryocyte hatching and Gametocyte reproduction bypass ordinary nutrient division rules. |
| Split Interval | `split_interval`, 1–60 seconds | Minimum cell age before ordinary division; both children receive a new birth time at division. All other gates still apply. Preview rejects values above 59; the inspected GPU ordinary-cell path does not enforce that sentinel. GPU Embryocytes do. |
| Split Ratio | `split_ratio`, 0–1 | Alters which existing bonds are inherited by A, B, or both, including the equatorial Zone C width. It does not allocate unequal shares of nutrients. This is a connectivity/geometry control, not a child mass ratio. At 0.5 the inheritance boundary is balanced; changing it shifts the boundary and broadens Zone C. |
| Max Cell Size | `max_cell_size`, 0.5–2 | Caps radius in the inspected preview growth paths (`new_mass.min(max_cell_size).clamp(0.5, 2)`). Changes cell dimensions and packing. The tooltip's claim that this forces division is not supported by the inspected division gates. Exact main-simulation size parity remains to be tested. |
| Membrane Stiffness | `membrane_stiffness`, 0–250 | Controls resistance to collision penetration. GPU collision response uses the average stiffness of a pair in its normal force. Higher values resist overlap more strongly. This is distinct from adhesion stiffness; zero does not remove all other forces or constraints. |

Ordinary division is jointly constrained by time, nutrients, split count,
connections, and enabled signals. The GPU also blocks division in frozen or
heat-shock thermal states; capacity and scheduling can defer eligible divisions.
These environmental constraints are not additional controls in this panel.

### Regulation Emit — all cell types

| UI setting | Genome field / range | Working understanding and design effect |
| --- | --- | --- |
| Emit Channel | `regulation_emit_channel`, Disabled (-1) or 8–15 | Enables a constitutive developmental signal source in this mode. Can create spatial gradients that other modes use to change growth or fate. |
| Emit Value | `regulation_emit_value`, -1000–1000 | Signed source strength. Strength, route attenuation, and receiver thresholds determine where a response occurs. Ordinary routes retain 95% per edge; Vasculocyte-to-Vasculocyte roads retain 98.75%. |
| Network Reach | No editable genome field here | Informational text only. There is no hop-limit slider; reach follows connected active signal routes and attenuation. |

### Signal Conditions — all cell types

The channel dropdowns in this group offer Disabled (-1) and channels 8–15.
Threshold controls range from 0 to 1000. Each listener has its own Response
setting as described above.

| UI setting | Genome fields | Working understanding and design effect |
| --- | --- | --- |
| Division Gating: channel, Response, threshold, Invert | `division_signal_channel`, `signal_response_modes[DIVISION]`, `division_signal_threshold`, `division_signal_invert` | Adds an eligibility condition for ordinary division. Invert permits growth when the normal signal condition is false. It does not override nutrient, timer, or connection gates. |
| Apoptosis: channel, Response, threshold, Invert | `apoptosis_signal_channel`, `signal_response_modes[APOPTOSIS]`, `apoptosis_signal_threshold`, `apoptosis_signal_invert` | Causes programmed death when active. Can remove temporary tissue and open or disconnect structures. This kills the cell rather than merely preventing its next split. |
| Child A Signal Routing: channel, Response, threshold | `signal_child_a_channel`, `signal_response_modes[CHILD_A]`, `signal_child_a_threshold` | Reads the parent's signal when dividing to select A's birth mode. Does not itself trigger division. |
| Child A: Above / Below | `signal_child_a_mode_above`, `signal_child_a_mode_below` | Each dropdown selects a mode index or Default (-1). An explicit target overrides the previously selected child mode, including after-splits routing. Default leaves that prior selection intact. |
| Child B Signal Routing: channel, Response, threshold | `signal_child_b_channel`, `signal_response_modes[CHILD_B]`, `signal_child_b_threshold` | Same decision for B, independently of A. |
| Child B: Above / Below | `signal_child_b_mode_above`, `signal_child_b_mode_below` | Same target/default semantics for B. Different A/B choices allow asymmetric differentiation. |
| Mode Switch (No Division): channel, Response, threshold, Invert | `mode_switch_signal_channel`, `signal_response_modes[MODE_SWITCH]`, `mode_switch_signal_threshold`, `mode_switch_invert` | Changes a living cell's mode in place when the condition is met, without creating a sibling. This can replace growth instructions, stop growth, or activate a specialist. |
| Mode Switch: Target | `mode_switch_target`, None (-1) or mode index | Selects the replacement mode. None supplies no target. The inspected GPU switch resets split count and birth time and loads the target mode's cached settings, so its division timer starts fresh. Do not model this as a division. |

Child mode selection in the inspected GPU executor proceeds: normal child mode →
explicit after-splits target if the limit is reached → explicit signal-selected
target → validity/type safeguards. Signal routing does not cancel the final-split
orientation or keep-adhesion overrides.

### Nutrient Settings — all cell types

| UI setting | Genome field / range | Working understanding and design effect |
| --- | --- | --- |
| Nutrient Priority | `nutrient_priority`, 0.1–10 | Biases transport through eligible connections toward this cell. Ordinary GPU pressure includes `nutrients / priority`, so higher priority favors retaining/receiving more nutrients; this is not a literal global feeding queue. Embryocyte receiver priority also scales its filling rate. Can change which parts grow first without changing split intervals. |
| Prioritize When Low | `prioritize_when_low`, boolean | GPU transport multiplies priority by 10 below 10 nutrients, and also protects a low-nutrient sender floor in applicable transfers. Helps survival but can restrict nutrient supply to attached growing tissue. Does not generate nutrients. |

### Connection Settings — all cell types

| UI setting | Genome field / range | Working understanding and design effect |
| --- | --- | --- |
| Max Connections | `max_adhesions`, 0–20 | Positive values gate ordinary division when active count is at or above the limit. GPU zero selects the hardware cap (20); it does not mean “no bonds.” Preview zero disables this explicit user-limit gate. Do not assume all bond-creation paths enforce this as a universal degree cap. |
| Min Connections | `min_adhesions`, 0–10 | Blocks ordinary division below the required active bond count. Zero imposes no minimum. Can hold growth until a structure has the required connectivity. The tooltip also claims it blocks signal emission; no such gate was found in the inspected current signal emitters, so do not rely on that claim. |
| Max Splits | `max_splits`, -1–20 | -1 is unlimited; 0 prevents division; positive values limit continued division in a lineage. Same-mode children generally inherit count +1; mode changes reset it. This is neither total organism cell count nor a global generation budget. |
| Child A After Splits | `mode_a_after_splits`, None (-1) or mode index | Chooses A's mode on the split that reaches the limit. None retains its normal child mode. Explicit target differences can also reset the split counter; see executor logic. |
| Child B After Splits | `mode_b_after_splits`, None (-1) or mode index | Same for B, allowing the last split to create two different terminal or continuing roles. |
| Child A Angle | `child_a_after_split_orientation`, quaternion | Rotation of A relative to the parent's genome orientation on the final allowed split. Changes the orientation of its later growth and directional behaviors. This does not change the parent's current split axis. |
| Child B Angle | `child_b_after_split_orientation`, quaternion | Independent final-split orientation for B. |
| Child A Keep Adhesion | `child_a_after_split_keep_adhesion`, boolean | Final-split permission to inherit eligible old parent bonds. If false, also suppresses the new A–B sibling bond. |
| Child B Keep Adhesion | `child_b_after_split_keep_adhesion`, boolean | Same for B. Both final-split keep flags must be true to permit the sibling bond requested by make adhesion. |

The after-splits controls appear when `max_splits >= 0`, including zero. Zero
nevertheless allows no split on which those birth overrides could run. A finite
limit with routes into continuing modes can produce indefinite development.

The quaternion balls use editor snapping/axis-drag state, which is not an
additional mode gene. Ordinary `parent_make_adhesion`, `child_a.keep_adhesion`,
`child_b.keep_adhesion`, `parent_split_direction`, and ordinary child orientations
are related controls located elsewhere, not editable controls in this function.

### Cell-type-specific controls

The following sections are conditional on the selected mode's `cell_type`.
For sections whose full simulation path has not been experimentally checked,
effects below describe the current UI/field contract; they are not a claim of
preview/main-simulation parity. No authored organism was run for this reference.

#### Test cell — Special Functions

| Setting | Field / range | Effect |
| --- | --- | --- |
| Nutrient Generation Rate | `nutrient_gain_rate`, 0–20 per second | Automatically produces nutrients without external food. Useful for isolating split geometry from feeding conditions. Raising it can make the nutrient gate pass sooner. |

#### Photocyte and Lipocyte

These use the same five-control pattern, but measure different environmental or
internal quantities.

| Setting | Photocyte field | Lipocyte field | Effect |
| --- | --- | --- | --- |
| Emit Signal | `photocyte_emit_enabled` | `lipocyte_emit_enabled` | Enables the conditional source. Turning it off does not disable photosynthesis or storage. |
| Signal Channel | `photocyte_emit_channel` | `lipocyte_emit_channel` | Destination channel 0–15. |
| Signal Value | `photocyte_emit_value` | `lipocyte_emit_value` | Signed emitted strength, -1000–1000. |
| Above / Below | `photocyte_emit_mode` | `lipocyte_emit_mode` | 0 emits at/above threshold; 1 emits below threshold. |
| Threshold | `photocyte_emit_threshold` | `lipocyte_emit_threshold` | 0–1; sampled light for Photocytes, stored nutrient fraction for Lipocytes. Turns light exposure or nutritional state into a signal that can influence development. |

#### Luminocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Signal Channel | `luminocyte_signal_channel`, 0–7 | Input controlling bright versus dim state. |
| Response | `signal_response_modes[LUMINOCYTE]` | Which signal polarity contributes to activation. |
| Threshold | `luminocyte_threshold`, 0–1000 | Activation strength. |
| On without signal | `luminocyte_invert`, boolean | Reverses activation: bright without the normal qualifying signal, dim with it. |
| Glow | `emissive`, 0–8 | Maximum emitted light/visible glow. Can affect light-responsive tissue indirectly; does not directly alter division geometry. |

#### Siphonocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Intake Rate | `siphon_intake_rate`, 0–4 | Rate of filling internal reserve from the occupied environment; UI contract says it does not remove voxel volume or change its phase. |
| Expel Rate | `siphon_expel_rate`, 0–4 | Internal reserve spending rate during expulsion. Does not create water or steam voxels. |
| Impulse | `siphon_impulse`, 0–3 | Directional body thrust strength; the inspected GPU stroke frequency also depends on this value. |
| Mode | `siphon_mode`, 0–3 | 0 Impulse: automatic expulsion strokes. 1 Signal Impulse: strokes require signal. 2 Signal Intake: intake-focused mode; GPU thrust path does not expel. 3 Signal Expulsion: expels while signal-active without the automatic stroke gate. |
| Signal Channel | `siphon_signal_channel`, 0–15 | Input for modes 1–3. |
| Response | `signal_response_modes[SIPHONOCYTE]` | Accepted signal polarity. |
| Signal Threshold | `siphon_signal_threshold`, 0–1000 | Activation strength. |
| Invert | `siphon_signal_invert`, boolean | Reverses signal gating. |

Siphon force also depends on water/heat state; lack of water does not necessarily
mean exactly zero thrust because the GPU has a dry-efficiency fallback. Its
forces can move a bonded body, but these settings do not specify body topology.

#### Plumocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Drag | `plumocyte_drag_mult`, 0–3 | Passive resistance to falling along gravity, rather than an active forward motor. Can change settling behavior and loading on attached tissue. |
| Rotation | `plumocyte_rotation_resistance`, 0–3 | Damps angular velocity to resist tumbling. Does not prescribe an absolute target orientation. |

#### Stemocyte Development

| Setting | Field / range | Effect |
| --- | --- | --- |
| Channel | `stemocyte_signal_channel`, 8–15 | Developmental gradient input. |
| Response | `signal_response_modes[STEMOCYTE]` | Converts signed input to the strength used for delay and band selection. |
| Delay | `stemocyte_delay_mode`, 0–4 | None (0): immediate evaluation. Cycles (1): require split count. Time (2): accumulate developmental time. Signal hold (3): accumulate uninterrupted positive interpreted signal, reset hold time when absent. Threshold (4): require a minimum interpreted signal strength. |
| Delay value | `stemocyte_delay_value` | Cycles: integer 0–20; Time/Signal hold: 0–120 seconds; Threshold: 0–1000. This delays differentiation, not necessarily division: ordinary division can continue while the delay is unmet. |
| Response-strip dividers / band width | `stemocyte_thresholds`, four percentage boundaries | Divide normalized strength 0–100% into five bands. Drag dividers or edit a band's width; UI repairs ordering and preserves at least 1% per band. Moves the spatial boundaries between fates. |
| Each band's response | `stemocyte_outcomes`, five entries | Remain Stemocyte (-1), Enter Apoptosis (-2), or a target mode index. A target changes mode in place. “Remain” proceeds to ordinary division checks. |

`stemocyte_weak_first` determines how stored outcome order maps to strength bands;
it is read by this panel but has no editable toggle here. Do not mistake the
visual left/right order for the serialized index order. Developmental timers are
inherited at division in the inspected GPU executor; mode transitions can reset
state, and the Cycles option uses split counts with their reset rules.

#### Glueocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Cell Adhesion | `glueocyte_cell_adhesion`, boolean | Enables contact-based bonding to cells. Separate from creating sibling bonds during division. |
| Bond to Own Organism | `glueocyte_self_adhesion`, boolean | Current UI describes a special preview applicator: touching two own-organism cells creates a mechanical ball joint between them and consumes the Glueocyte; touching Glueocytes can merge. Do not assume this is merely permission for ordinary self-bonds or that GPU behavior is identical without testing. |
| Environment Adhesion | `glueocyte_env_adhesion`, boolean | Permits contact attachment to cave/world surfaces. |
| Boulder/Mossrock Adhesion | `glueocyte_boulder_adhesion`, boolean | Permits contact attachment to floating boulders. |
| Signal Gate | `glueocyte_cell_adhesion_signal_channel`, Always On (-1) or UI channels 0–7 | Gates attachment. Field comments allow broader channels, but this panel offers sensory channels only. Gate controls appear when an adhesion option is enabled. |
| Response | `signal_response_modes[GLUEOCYTE]` | Accepted polarity for the attachment gate. |
| Threshold | `glueocyte_cell_adhesion_signal_threshold`, 0–1000 | Required interpreted strength. |
| Disconnect when: No signal / Signal | `glueocyte_signal_gate_invert`, false / true | Chooses whether attachment is active with the signal or with its absence; controls release of relevant Glueocyte attachments. Exact ownership/release behavior of applicator-created joints needs its own test. |

#### Flagellocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Speed Mode: Fixed / Signal | `flagellocyte_use_signal`, false / true | Chooses constant thrust or two signal-selected thrust levels. |
| Swim Force | `swim_force`, 0–3 | Forward thrust in fixed mode. Orientation determines force direction; bonding transmits its load to the organism. |
| Channel | `flagellocyte_signal_channel`, 0–7 | Input in Signal mode. |
| Response | `signal_response_modes[FLAGELLOCYTE]` | Accepted polarity. |
| Speed A | `flagellocyte_speed_a`, 0–3 | Thrust when the signal condition is inactive. |
| Speed B | `flagellocyte_speed_b`, 0–3 | Thrust when active. |
| Threshold C | `flagellocyte_threshold_c`, -100–100 | Switch threshold. Shared response logic clamps threshold to at least zero; negative UI values should not be treated as an independent negative-signal selector. |

#### Buoyocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Buoyancy Force | `buoyancy_force`, 0–3 | Upward force opposing gravity. Can lift or load bonded structures; does not make a new connection. |

#### Oculocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Sense Type checkboxes | `oculocyte_sense_type`, bitmask | Cell=1, Food=2, Light=4, Wall/Cave=8, Self=16, Mossrock=32; combine enabled bits. Self is an unconditional source, not a ray test for the organism's own tissue. |
| Signal Channel | `oculocyte_signal_channel`, 0–7 | Sensory output channel. |
| Signal Value | `oculocyte_signal_value`, -1000–1000 | Signed strength emitted upon detection (or continuously for Self). |
| Ray Length | `oculocyte_ray_length`, 1–100 | Detection reach; ray orientation depends on the cell's orientation. |
| Light Color Filter: R, G, B | `oculocyte_light_target_color.x/y/z`, each 0–2 | Target RGB for Light sensing; shown when Light is selected. |
| Color Tolerance | `oculocyte_light_color_tolerance`, 0–1 | Allowed color distance from that target. Larger values admit more colors. |

#### Ciliocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Speed Mode: Fixed / Signal | `cilia_use_signal`, false / true | Chooses fixed motion or signal-selected speeds. |
| Cilia Speed | `cilia_speed`, -1–1 | Fixed signed directional pushing speed; negative reverses direction, zero stops directional pushing. |
| Channel | `cilia_signal_channel`, 0–7 | Signal-mode input. |
| Response | `signal_response_modes[CILIOCYTE]` | Accepted polarity. |
| Speed Below / Speed Above | `cilia_speed_below`, `cilia_speed_above`, each -1–1 | Speeds for inactive/active signal condition. |
| Threshold | `cilia_threshold`, -100–100 | Signal switch threshold; see shared response rules. |
| Push Organism Cells | `cilia_push_bonded`, boolean | UI contract enables pushing bonded organism cells, permitting internal mechanical effects. |
| Attract Force | `cilia_attract_force`, 0–1 | Attraction toward the cell for nearby eligible unattached cells/particles. Can funnel material toward feeding tissue. |

#### Myocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Contraction Mode: Pulse / Signal | `myocyte_use_signal`, false / true | Selects periodic or signal-dependent contraction. |
| Pulse Phase: A / B | `myocyte_pulse_phase`, 0 / 1 | Half-cycle phase offset between groups. Does not alone guarantee a traveling wave. |
| Pulse Rate | `myocyte_pulse_rate`, 0.1–10 cycles/second | Frequency in Pulse mode. Matching rates maintain relative phase. |
| Contraction | `myocyte_contraction`, 0–1 | Pulse-mode shortening strength; zero means no active shortening. |
| Channel | `myocyte_signal_channel`, 0–15 | Signal-mode input. |
| Response | `signal_response_modes[MYOCYTE]` | Accepted polarity. |
| Contraction Below / Above | `myocyte_contraction_below`, `myocyte_contraction_above`, each 0–1 | Strengths for inactive/active signal condition. |
| Threshold | `myocyte_threshold`, -100–100 | Switch threshold; see shared response rules. |
| Grip on Contract | `myocyte_grip_contracted`, 0–30 | Medium drag/grip at full contraction. |
| Grip on Extend | `myocyte_grip_extended`, 0–30 | Medium drag/grip at full extension. Differences between grip states can bias locomotion, but direction and effectiveness depend on geometry and environment. |

Myocyte contraction acts through the mechanical system; a contraction schedule
alone does not establish a useful body. The grip subsection is collapsible, and
its open/closed state is editor UI state, not another genome setting.

#### Embryocyte Release Triggers

| Setting | Field / range | Effect |
| --- | --- | --- |
| Timer | `embryocyte_use_timer`, boolean | Includes the release-time condition. |
| Release after | `embryocyte_release_timer`, 0.1–300 seconds | Required age for release while attached. GPU checks time since birth; it is not a separately accumulated attachment-duration clock. |
| Reserve Threshold | `embryocyte_use_threshold`, boolean | Includes reserve readiness in release conditions. |
| Release when reserve >= | `embryocyte_threshold_value`, integer 0–65535 | Required reserve in whole units. GPU compares fixed-point reserve with this value multiplied by 1000. |
| Signal | `embryocyte_use_signal`, boolean | Includes signal readiness in release conditions. |
| Channel | `embryocyte_signal_channel`, 0–15 | Signal read by this cell, not necessarily emitted directly by its parent. |
| Response | `signal_response_modes[EMBRYOCYTE]` | Accepted polarity. |
| Release when signal >= | `embryocyte_signal_value`, 0–1000 | Required interpreted positive strength. No release-signal inversion toggle exists here. |

At least one release trigger must be enabled, and **all enabled triggers must
pass**. Release drops attachments; it is separate from hatching. Once free, an
Embryocyte burns reserve and uses the split interval/count hatching rules. The GPU
release executor resets birth time at release; verify preview parity when timing
experiments depend on that reset. The tooltip saying “62000 ≈ 6.2 seconds” conflicts
with whole-unit thresholds and a 10-units/second burn: do not use that example as
a calibrated lifetime prediction.

#### Gametocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Merge Range | `gametocyte_merge_range`, 0–2 | Extra merge distance beyond cell radii for compatible gametes. Zero requires contact. |
| Timer, release time, Reserve Threshold, reserve value, Signal, channel, Response, signal value | Same eight `embryocyte_*` fields and `signal_response_modes[EMBRYOCYTE]` listed above | Same AND-combined attachment-release controls and ranges. Gametocytes never divide; split settings do not turn them into ordinary dividing cells. Their UI describes compatible gametes from different organisms merging into an Embryocyte with a crossover genome. |

#### Vasculocyte

| Setting | Field | Effect |
| --- | --- | --- |
| Nutrients: Transport | `vascular_nutrient_transport`, boolean | Enables high-throughput nutrient conduction along the vascular network. |
| Nutrients: Exchange Port | `vascular_outlet`, boolean | Enables bidirectional nutrient exchange with adjacent nonvascular tissue. |

Both on = pipe with tissue exchange; Transport alone = sealed pipe; Exchange
alone = local exchange port; both off = nutrient-closed. Signal-road behavior is
automatic between bonded Vasculocytes and is not toggled by these nutrient
checkboxes. These settings can change developmental nutrient supply without
changing the original division geometry.

#### Devorocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Contact Range | `devorocyte_consume_range`, 0–3 | Extra reach beyond contact for consuming eligible foreign cells. |
| Consume Rate | `devorocyte_consume_rate`, 0–200 nutrients/second | Rate of nutrient theft from targets in range. Depleting a target can kill it. UI states same-organism and same-genome cells are excluded. Can affect feeding and neighboring tissue, not directly set the organism's developmental shape. |

#### Cognocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Operation | `cognocyte_operation`, 0–19 | Selects the computation listed below. |
| Input A / Input B | `cognocyte_input_channel_a`, `cognocyte_input_channel_b`, each 0–15 | Source channels. Unary operations hide/ignore B; oscillators hide/ignore both. Labels vary with operation. |
| Output Channel | `cognocyte_output_channel`, 0–15 | Sends the computed signal to behavior or developmental listeners. |
| Rate | `cognocyte_oscillator_rate`, 0.1–10 cycles/second | Oscillator frequency. |
| Phase Offset | `cognocyte_oscillator_phase`, 0–1 cycle | Relative timing; 0.5 gives a half-cycle shift. |
| Signal Strength | `cognocyte_oscillator_strength`, 0–1000 | Peak oscillator magnitude. |
| Polarity | `cognocyte_oscillator_polarity`, 0 Positive / 1 Negative / 2 Bipolar | Chooses positive, negative, or alternating-sign waveform. |
| Envelope / Boolean true text | No editable field | Wave Oscillate is a strength ramp, not a configurable hop budget. Boolean true is +1000. |

Actual evaluator operations (before final output clamping to -1000–1000):

| Code / operation | Result |
| --- | --- |
| 0 Add; 1 Subtract | `A+B`; `A-B` |
| 2 Multiply; 3 Divide | `A*B/1000`; `A*1000/B` (0 when `abs(B) <= 0.1`) |
| 4 Min; 5 Max; 6 Average | `min(A,B)`; `max(A,B)`; `(A+B)/2` |
| 7 Greater Than; 8 Less Than; 9 Equal | +1000 if `A>B`, `A<B`, or `abs(A-B)<=0.1`, respectively; otherwise 0 |
| 10 AND; 11 OR; 12 NOT | Positive values are true; output +1000 or 0; NOT reads only A |
| 13 Select | B if A is positive, otherwise 0 |
| 14 Oscillate | Half-rectified sine for positive/negative polarity, full signed sine for bipolar |
| 15 Wave Oscillate | Sawtooth envelope; polarity maps its sign/range |
| 16 ABS; 17 Negate; 18 Positive; 19 Negative | `abs(A)`; `-A`; `max(A,0)`; `min(A,0)` |

The current processor suppresses binary operations when either input is zero;
therefore OR with a zero input can differ from the standalone Boolean evaluator.
Unary operations and oscillators follow their separate paths. Computation uses
signal ticks and a prior immutable field, so circuit depth can introduce delay.
Tooltips claiming Boolean true is 1 or multiplication is plain `A*B` are stale.

#### Memorocyte

| Setting | Field / range | Effect |
| --- | --- | --- |
| Rate | `memorocyte_rate`, 0–1 | Fraction of the remaining input-memory gap closed over one second. Actual signal-tick interpolation is `1-(1-rate)^tick_seconds`. Zero retains memory; one tracks immediately; intermediate values smooth changes. |
| Input Channel | `memorocyte_input_channel`, 0–15 | Signed signal to track. Missing/zero input makes memory approach zero when rate is positive. |
| Output Channel | `memorocyte_output_channel`, 0–15 | Emits the stored value, allowing delayed or smoothed behavior and developmental responses. |

### Interpretation cautions and next verification

This inventory covers the panel's editable controls, including controls hidden
behind enable switches, finite split counts, signal modes, or collapsible groups.
`cell_type` chooses the conditional section but is not changed by this panel.
Cell types with no dedicated branch, such as Phagocytes, still get the common
sections. Informational labels are identified above to avoid inventing genes.

Before promoting settings into tested recipes, resolve these specific issues:

1. Split Interval >59: compare preview and ordinary GPU division after 60 seconds.
2. Max Cell Size: verify radius changes separately from division eligibility.
3. Min Connections: test division and signal emission independently; do not assume
   the tooltip's signal gate exists.
4. After-split keep: test inherited bonds and sibling bonds separately; the parent
   becomes two children, so “maintains a bond with the parent” is misleading.
5. Glueocyte own-organism applicators: compare preview and GPU joint creation,
   consumption, and signal-controlled release.
6. Embryocyte timing/units: verify birth-time reset on release and actual reserve
   consumption before predicting incubation or free lifetime.
7. Cognocyte operations: test zero-input suppression and signal polarity using
   actual processor outputs, not tooltip arithmetic.
8. Multi-selection: the missing copy entries listed above require individual mode
   edits or a separate engine fix before relying on those edits across selections.

## Parent split angle versus child angles — construction utility

Code-inspected 2026-09-12, following the user's square lesson. Sources:
[preview division](src/cell/division.rs) and
[GPU division execution](shaders/lifecycle_division_execute_ring.wgsl).
The square solution is user-confirmed; the general design implications below are
derived from the implementation, not newly simulation-tested recipes.

### The two controls act at different stages

Let `P` be the parent's orientation, `S` its configured split rotation, and
`C_A` / `C_B` the effective child rotation offsets. `S` is constructed from
`parent_split_direction: [pitch, yaw]` using Euler order YXZ (yaw, pitch, zero roll).

Ignoring spawn jitter:

```
Current split axis         = P × S × local_Z
Child A position           = parent_position + offset × split_axis
Child B position           = parent_position - offset × split_axis
Child A orientation        = P × S × C_A
Child B orientation        = P × S × C_B
Next split axis for child  = child_orientation × that_child_mode's_S × local_Z
```

These products compose rotations, not additions of Euler angles. Rotations around
different axes generally do not commute. Physics and genome orientations are
tracked separately; the child composition has the same form for each.

| Effect | Parent split angle (`S`) | Child angle (`C_A` or `C_B`) |
| --- | --- | --- |
| Positions at this division | Changes the shared axis along which A and B are born, on opposite sides of the parent. | Does not directly change either child's spawn position or the current split axis. |
| Existing bond inheritance at this division | Changes the split-relative zones and therefore which eligible old bonds can go to A, B, or both. | Does not directly change the parent's zone classification or keep flags. |
| New child orientation | Rotates the common frame inherited by both children. | Adds an independent rotation within that split frame for the selected child. |
| Bond anchors and twist references | Contributes to both children's orientation frames and the inheritance geometry. | Changes how inherited/sibling anchors are represented in that child's local frame and contributes to its twist reference. These are structural orientation controls, not merely visual rotation. |
| Later growth | Compounds when the same mode repeats, even with identity child offsets. | Steers the selected child's later split axis, combined with the split setting in its next mode. |
| Directional cell functions | Reorients both descendants through the shared split frame. | Independently aims descendants' local directional behaviors, such as thrust or sensing. |

The cell-angle controls do not directly create a V-shaped separation at birth:
the two children are still placed along one common parent split axis. Divergent
child angles can produce branching through later growth and mechanical evolution.

### When to choose each control

- **Choose parent pitch/yaw to place the current division.** This is the direct
  control for dividing along or across an existing structure, and for changing
  the relation between its old bonds and the division plane.
- **Choose parent pitch/yaw for a shared repeated turn.** A self-referencing mode
  can rotate the developmental frame each generation with identity child offsets.
  The user's 90° pitch-or-yaw, Max Splits 2 square demonstrates this utility.
- **Choose child angles to aim the branches independently while preserving the
  current split placement and inheritance classification.** A and B may receive
  equal offsets for repeated tiling, different offsets for asymmetric roles, or
  opposed offsets for divergent future growth. None guarantees a particular
  mature branch shape without tracing later divisions and bonds.
- **Use both to separate placement from facing.** For a desired final orientation
  `D` relative to the parent, choose `C = inverse(S) × D`. For example, choosing
  `C_A = C_B = inverse(S)` leaves both children facing like the parent while still
  placing them along the turned split axis. This cancels the inherited turn, not
  the current placement or its bond-inheritance consequences.
- **Use child roll to turn around a facing axis.** A quaternion can change lateral
  axes and twist reference without changing local forward direction. This matters
  for later off-axis growth. The parent UI exposes pitch/yaw only, whereas child
  orientation is a full 3D rotation.

### What can and cannot be substituted

Moving a shared rotation from both child offsets into the parent split setting
can preserve child orientation while changing the current birth positions and
which old bonds are inherited. Therefore the two encodings are not universally
interchangeable inside an already connected organism.

For a founder with no old bonds, the two square encodings can produce the same
square topology in differently oriented planes/axes. That equivalence does not
prove they are interchangeable when attaching the square to an existing stalk.
Compare the first split axis, old-bond zones, and adult orientations as well as
the final silhouette.

At the maximum-splits transition, `C_A` and `C_B` come from the **after-split**
orientation fields. The normal parent split rotation `S` still applies; there is
no separate final-split parent angle in these settings. An identity after-split
child offset does not cancel a nonzero parent split rotation.

**Implementation caveat:** The inspected preview spawn path applies `S` to the
parent's genome orientation, whereas the GPU spawn path applies it to the parent's
physics orientation. Both accumulate child genome orientation as `P_genome × S × C`.
The geometric comparison above is clearest when those frames agree; strong
physical rotation can expose a preview/GPU placement difference worth testing.

**Useful comparison experiment:** Hold everything else fixed and compare parent
90°/children identity, parent identity/both children 90°, and parent 90°/both
children inverse-90°. Observe one split for placement, two for future growth,
then repeat with an existing parent bond to expose inheritance differences.

## MR-001 — Adhesion-linked 2×2 square

**Status:** Original construction hypothesis confirmed by the user on 2026-09-12.
No simulation was run by the assistant for this lesson.

**Reference genome:** Awaiting identification of the user's authored example.
[Flat Square Test Lattice](genomes/Flat%20Square%20Test%20Lattice.genome) is an
existing related reference, not the user's submitted lesson. It uses identical
quarter-turn child rotations and a larger split count.

**Intended structure:** Four cells in a plane with four perimeter bonds, two
neighbors per cell, no diagonal bond.

### Confirmed construction principle

1. The founder splits into two bonded children with Make Adhesion enabled.
2. Both children receive the same quarter-turn orientation, making their next
   split axes perpendicular to the first bond.
3. Both divide with Make Adhesion enabled, both effective Keep Adhesion flags
   enabled, and Split Ratio 0.5. Zone C inheritance and sibling bonding construct
   the square perimeter.
4. Four resulting cells stop dividing. The original proposal used founder,
   second-division, and terminal modes, with terminal Max Splits 0.

Finite-limit divisions use after-split orientations and keep flags. Keep these
consistent with the intended sequence. Provide sufficient nutrients and connection
capacity for the intermediate structure, and avoid bond breakage in the basic test.

### Alternative genome encodings

The parent-rotation variant below is user-confirmed. The other alternatives are
code-derived alternatives to the confirmed principle. They target the
same physical connectivity; final cell orientations, mode identities, and later
behavior need not be identical. None has been simulation-tested by the assistant.

| Alternative | Concrete construction | How growth ends |
| --- | --- | --- |
| One repeating mode with parent rotation — user-confirmed | Both children reference the same mode. Set parent rotation pitch **or** yaw to 90° (`parent_split_direction: [90, 0]` or `[0, 90]`), with identity child rotation offsets. Make on, effective keeps on, Split Ratio 0.5. | Max Splits 2, after-splits mode targets None, both final-split keeps on. The four same-mode cells exhaust their split counts. |
| One repeating mode | Both children reference the same mode. Make on, both ordinary keeps on, Split Ratio 0.5, identical 90° child rotations. Set Max Splits 2 and leave both after-splits mode targets None. Keep both final-split keeps on. | Founder count 0 produces two count-1 children, which produce four count-2 cells in the same mode. Count exhaustion blocks further division; no terminal mode needed. Final-split angles may be chosen for the desired adult orientation. |
| Two modes: repeating growth plus terminal | Use the same quarter-turn growth mode with Max Splits 2. Route both children after the second split to a terminal mode. | Terminal Split Nutrients Never (`split_mass: 2.01` for Test cells) can stop growth even with terminal Max Splits -1. Alternatively use terminal Max Splits 0. |
| Change the second mode's split direction | Founder splits along local Z with identity child rotations. Both children enter a second mode whose Parent Split Direction is `[0, 90]`, perpendicular to that first axis, with identity child offsets. Both splits make adhesions and keep eligible bonds at Split Ratio 0.5. Route the final children to a terminal mode. | Terminal Max Splits 0 or Split Nutrients Never. The quarter-turn is encoded in the second mode's split direction instead of in the founder's child rotations. |
| Explicit individual modes | Founder routes A and B to separate intermediate modes with the same required geometry. Those modes route to four distinct terminal modes. | Each terminal mode independently uses Max Splits 0 or Split Nutrients Never. More modes, but every corner is separately addressable for later specialization. |

The executor composes child orientation as parent orientation × split rotation ×
child offset. Thus changing the split-direction control can also rotate the born
children even when their offset is identity. Match adult orientations explicitly
when comparing more than the square's shape and bonds.

**User correction, 2026-09-12:** A second mode is not required to turn the split
axis. With parent pitch or yaw at 90°, the split rotation is also inherited into
the children's orientation. Reusing the same mode applies the quarter-turn again
relative to that new frame, giving a perpendicular second-generation split axis.
Max Splits 2 then ends growth. This is a valid single-mode alternative to putting
the quarter-turn into both child rotation offsets. Set one of pitch/yaw to 90°;
this lesson does not claim that setting both to 90° is equivalent.

### A tempting stopping rule that is not equivalent

Do not assume a single repeating mode with Max Connections 2 will finish this
square. After one of the first pair divides, the still-undivided partner can have
two inherited connections. Its `count >= max_adhesions` gate would then block the
remaining division, leaving three cells. The final degree is not necessarily the
maximum degree needed for a cell that still has to divide.

Signal gates, unavailable nutrients, and connection gates can stop terminal modes,
but need additional conditions to remain stopped. The interval sentinel still has
the preview/GPU discrepancy. These are not unconditional replacements without
specifying their context.

### Reusable lesson

Separate **geometry**, **connectivity**, **mode addressing**, and **termination**.
The same square can be encoded using repeated or explicit modes, with the
perpendicular axis specified through child rotation or a later split direction,
and with different mechanisms that stop division after the structure is formed.

**Next verification:** Read the user's reference genome, identify its chosen
encoding, then compare alternatives for cell count, perimeter bonds, planarity,
and persistent cessation of division.

## Lesson template

Copy this section for each new pattern. Leave unknown details explicitly unknown;
not every lesson needs every field filled immediately.

### MR-001 — [Pattern name]

**Status:** Awaiting explanation / interpreted / user-confirmed / simulation-tested

**Reference genome:** [Relative file link]

**Reference version:** [Revision or content hash, if available]

**Intended shape:** [The user's description of the physical structure]

**Lesson from the user:** [What this example is intended to teach, preserving any
important terminology or distinctions]

#### Construction settings

| Mode / lineage | Setting | Value | Role in forming the shape | Evidence |
| --- | --- | --- | --- | --- |
| [Mode] | [Field, including relevant defaults] | [Value] | [Effect] | [User explanation / code / hypothesis] |

Identify which settings are essential, which may be adjustable, and which are
merely incidental to this example. Mark those distinctions as uncertain until
supported by evidence.

#### Developmental sequence

1. **Starting cell:** [Initial mode, orientation, and relevant conditions]
2. **Divisions:** [How successive splits place and orient children]
3. **Connectivity:** [How bonds form, transfer, persist, or break]
4. **Result and termination:** [How the structure reaches its shape and whether
   growth stops or continues]

Use explicit local versus world coordinate descriptions where orientation matters.

#### Observations and explanation

- **User-reported result:** [What the user observed]
- **Directly verified result:** [What was inspected or tested, if anything]
- **Mechanism:** [Explanation and supporting code references where checked]
- **Open hypotheses:** [Unverified causal claims or unresolved details]
- **Conditions:** [Preview or main simulation, engine revision, environment,
  duration, and other relevant settings when known]

#### Reusable recipe

**Construction rule:** [Plain-language instructions for reproducing the pattern]

**Demonstrated variations:** [Changes shown to preserve or intentionally alter it]

**Limits and failure cases:** [Known constraints and unsuccessful variations]

**Composition:** [Where other structures could attach; distinguish tested
combinations from proposed ones]

**Next check:** [Smallest useful test or clarification, if one is needed]

#### Corrections and history

- [Date: initial lesson or correction, its evidence, and what interpretation changed]

## Related project notes

- [Organism design procedurals](organism-design-procedurals.md): broader organism
  design observations, including viability and simulation behavior.
- [Procedural creature generation morphology](procedural-creature-generation-morphology.md):
  existing geometric mechanics and structural generation notes. Check relevant
  claims against the current engine and authored examples before adopting them.
- [Genomes](genomes/): authored reference files.

## Notebook history

- 2026-09-12: Created the notebook and lesson format. Awaiting the user's first
  authored example; no morphology recipes have been inferred or confirmed yet.
- 2026-09-12: Recorded the initial adhesion and split-control discussion, including
  ordinary versus final-split keep behavior, proposed basic tests, and the unresolved
  preview/GPU split-interval discrepancy. These remain code-inspected foundations,
  not simulation-confirmed recipes.

- 2026-09-12: Added the full Parent Settings panel reference, including conditional
  cell-type controls, field mappings, UI ranges, source links, and unresolved
  implementation/tooltips differences. No simulation tests performed for this inventory.
