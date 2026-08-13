# Cached Signal Backbone System

Status: Approved semantic design; implementation benchmark required  
Scope: CPU preview and GPU simulation signal semantics, propagation, processing, topology, performance, migration, and verification  
Supersedes: The incomplete bounded per-emitter route-cache design in the current working tree  

## 1. Executive Summary

Bio-Spheres should replace iterative adhesion-graph signal relaxation with a cached, visible signal backbone. Each signal-connected region is represented by a deterministic forest. Live sources are evaluated at a fixed signal frequency, their values are propagated additively along the forest with distance attenuation, and all local signal processors update synchronously for the following signal tick.

The intended result is:

- Signal strength remains a meaningful indicator of distance and network structure.
- Runtime does not scale with the number of emitters or the density of mechanical cross-links.
- Dynamic sensors work without rebuilding topology.
- Cognocyte and Memorocyte circuits have deterministic feedback and arbitrary depth.
- GPU work is bounded primarily by live cell count and active channel groups.
- Topology work is incremental, budgeted, and never requires CPU readback.

The final implementation should use blocked or hierarchical tree processing rather than a single root-relative prefix calculation. Microtrees prevent floating-point underflow on deep organisms, improve locality, and give topology maintenance a useful unit of incremental work.

This design deliberately removes or redefines features that prevent scalable aggregation:

- Per-source hard hop limits are removed.
- Vascular signal capacity is removed.
- `Hops Oscillate` becomes a strength-ramp wave oscillator.
- Only selected backbone bonds carry signals; incidental mechanical cross-links do not.
- Zero is the sole representation of silence.
- Signals are signed in the range `-1000..1000`.
- Signal production and one-time backbone construction consume nutrients.

## Approved Owner Decisions

The following visible semantics were approved before implementation:

| Area | Approved decision |
|---|---|
| Signal topology | Only bonds explicitly classified as backbone bonds at creation carry signals. Mechanical-only bonds never carry signals and can never be promoted. |
| Backbone creation | Any cell-to-cell bond operation may have a heritable `creates backbone bond` property. Every affordable qualifying bond becomes an immutable backbone bond, including cycle-forming redundant bonds. If construction is unaffordable, the physical bond is not created. Environment and boulder bonds are never eligible. |
| Active routing | Backbone cycles are reduced to one cached active propagation forest. Active edges are yellow and redundant standby edges are black, including while the active route is silent. Routing is automatic and never genome-authored. |
| Route selection | Route resistance is the sum of fixed-point `-ln(retention)` edge costs. A new cycle path is selected only when its total resistance is strictly lower than the active route it bypasses; exact ties preserve the established route. |
| Damage and repair | Breaking an active backbone stops transmission immediately. The lowest-resistance valid standby reconnection activates at the next signal tick; the network remains split only when no redundant backbone route reconnects it. Mechanical-only bonds are never searched or promoted. |
| Inter-organism links | Eligible bonds may join different organisms. All 16 channels cross bidirectionally and the two trees become one shared network. |
| Aggregation | Same-channel contributions add, cancel by sign, and saturate symmetrically after complete accumulation. |
| Range | Signals use `-1000..1000`; zero is silence and neutral. |
| Self reception | A cell never receives its own normal emission. Heat-stroke dysregulation is an explicit local exception. |
| Reach | Attenuation applies on the first edge. Hard hop limits are removed. |
| Vascular behavior | Vascular capacity is removed. Normal retention is `0.95`; vascular-road retention is `0.9875`. |
| Clock | Signals update at a fixed 15 Hz with capped catch-up. |
| Processors | Cognocytes and Memorocytes have one signal-tick latency and reset state/output on mode changes and division. |
| Boolean logic | Only positive values are true. Zero and negative values are false. Boolean true emits `+1000`. |
| Arithmetic scale | Cognocyte Multiply uses `A * B / 1000`; Divide uses `A * 1000 / B`. Signals behave as fixed-point values in `-1..1` for multiplicative operations. |
| Signed operations | Cognocytes include `ABS`, `NEGATE`, `POSITIVE`, and `NEGATIVE`. Oscillators support positive, negative, and bipolar output. |
| Listener polarity | Every listener selects positive, negative, or magnitude response, with optional inversion. |
| Heat stroke | Every tick, every channel independently emits deterministic full-magnitude `-1000` or `+1000`, both locally and through the backbone, without nutrient brownout. |
| Preview parity | Preview and GPU gameplay expose the same sources, processors, and semantics. |
| Legacy genomes | No automatic unsigned-to-signed scaling is performed. Legacy genomes require explicit manual signal review. |
| Debug visualization | Adhesion routing uses yellow for the selected active backbone and black for redundant standby backbone bonds. The separate channel inspector uses red for positive values, blue for negative values, and brightness for magnitude. |
| Topology timing | Broken edges stop transmitting immediately. New backbone edges activate on the next signal tick. |
| Economics | Construction costs 5% of the creator's next-division requirement exactly once when the physical backbone bond forms. There is no continuous per-bond maintenance cost. Full-strength one-channel emission costs 25% of reference baseline metabolism per second. |
| Creator attribution | The creator pays the one-time construction cost. Routing, standby state, inheritance, organism boundaries, and later mode changes add no continuing per-bond charge. Ordinary emission uses proportional analog brownout and is paid only by its sender. |
| Creator selection | The initiating cell creates dynamic contact bonds; the parent creates and pays for developmental sibling bonds and newly duplicated inherited bonds before its nutrients are divided; authored scaffold endpoint A creates scaffold bonds. Future symmetric operations choose one initiator before affordability using stable cell identity as the simultaneous-initiation tie-breaker. An unaffordable chosen creator is not replaced by the other endpoint. |

Performance-driven implementation choices that do not alter these visible semantics are decided by the benchmark gates in this document. Any newly discovered semantic fork MUST be flagged for owner input rather than silently chosen during implementation.

## 2. Normative Language

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** describe implementation requirements.

- A **signal tick** is one complete source, propagation, consumption, and processor-update cycle.
- A **backbone edge** is an adhesion bond immutably classified as signal-capable at creation.
- An **active backbone edge** is a backbone edge selected into the cached propagation forest.
- A **standby backbone edge** is a valid redundant backbone edge excluded from the active forest to prevent cyclic multipath amplification.
- A **mechanical edge** is any active adhesion bond, whether or not it is a backbone edge.
- A **signal tree** is one connected tree of active backbone edges.
- A **signal forest** is the cached collection of all active signal trees in the scene.
- A **backbone graph** is the complete active-plus-standby graph of valid backbone bonds.
- A **microtree** is a bounded connected portion of a signal tree used for GPU evaluation and incremental maintenance.
- A **processor output** is a Cognocyte or Memorocyte emission stored from the preceding signal tick.
- A **pathological source** is a source, such as heat-stroke screaming, that may bypass ordinary funding rules.

## 3. Goals

### 3.1 Functional goals

- Preserve distance-dependent signal attenuation.
- Support all 16 channels.
- Support any number of simultaneously active emitters without per-emitter route storage.
- Preserve additive contributions and final saturation.
- Support sensory, regulation, conditional physiological, oscillator, Cognocyte, and Memorocyte emissions.
- Give every actuator and lifecycle gate one consistent finalized local field.
- Make signal paths visible and understandable.
- Make CPU preview and GPU simulation share the same semantics.

### 3.2 Performance goals

- Normal signal cost MUST be independent of mechanical adhesion density.
- Normal signal cost MUST be independent of emitter count.
- Normal signal cost MUST NOT be proportional to authored reach or organism diameter.
- The hot path SHOULD approach linear work in live cells and active channel groups.
- The implementation MUST support the current maximum configured capacity of 200,000 cells.
- Signal processing MUST remain in the existing command encoder and queue submission.
- Signal processing MUST NOT require CPU readback.
- Expensive topology reconstruction MUST be budgetable across frames.

### 3.3 Determinism goals

- Results MUST NOT depend on GPU thread scheduling or cell iteration order.
- All processors MUST read one immutable field for a tick and commit outputs simultaneously.
- Topology changes MUST take effect only at a signal-tick boundary.
- Preview and GPU results SHOULD agree within documented floating-point and quantization tolerances.

## 4. Non-Goals

- Reproducing the current iterative shader's cross-emitter coupling.
- Summing multiple physical paths from one emitter to one receiver.
- Making every adhesion bond a signal-carrying edge.
- Preserving source-specific hard hop limits.
- Preserving vascular throughput capacity.
- Solving arbitrary combinational Cognocyte chains within one signal tick.
- Zero-latency response to topology changes.
- Bit-identical CPU and GPU floating-point results.

## 5. Problems in the Abandoned Route-Table Design

The existing partial route-cache work MUST NOT be completed as designed.

### 5.1 Live-state contamination

It seeds topology routes from emitters active during rebuilding. An Oculocyte that is inactive during a rebuild has no route when it later activates.

### 5.2 Bounded silent loss

Eight slots are shared across all emitters and all 16 channels. Valid contributors are silently discarded when a cell is reachable from more than eight emitter/channel pairs.

### 5.3 Unsafe invalidation identity

Dirty flags use stable organism IDs that are eventually consistent, absent for small components, limited to 512, recycled, and changed by splits. They are not safe cache identities.

### 5.4 Missing invalidation causes

Emitter type, channel, strength, and hop changes can alter the table without changing vascular topology. Current invalidation covers only a subset of those changes.

### 5.5 Unrepresented nonlinear behavior

Vascular capacity cannot be represented by one scalar attenuation weight. Selecting routes by attenuation also does not select the largest live contributions when emitter strengths vary.

### 5.6 Unbounded exact alternative

An exact sparse emitter-to-receiver matrix scales with reachable source/destination pairs. At 100 routes per cell and eight bytes per route, 100,000 cells require approximately 80 MB for the completed matrix before rebuild scratch space.

## 6. Authoritative Signal Semantics

### 6.1 Domain and silence

- A signal value is a signed scalar.
- Final public values are clamped to `[-1000, 1000]`.
- Zero means silence.
- There is no separate present-zero state.
- Intermediate accumulation MUST use `f32` or a wider non-saturating representation.
- Contributions MUST be clamped only when producing a finalized local field or packed public value.
- Positive and negative contributions attenuate toward zero.
- Opposite polarities cancel before final saturation.

### 6.2 Additive transport

For cell `v`, channel `c`, and all other sources `s` in the same signal tree:

```text
field[v,c] = clamp(sum(source[s,c] * path_weight(s,v), s != v), -1000, 1000)
```

`path_weight(s,v)` is the product of the attenuation of each active backbone edge on the unique cached-forest path between `s` and `v`.

All contributions are positive. Multiple physical paths never amplify one source because only the selected active forest propagates; standby backbone edges do not transmit until selected during a topology commit.

### 6.2.1 Fan-in, cancellation, and collective signaling

- Same-channel sources add exactly before final saturation.
- Source count does not normalize the result.
- Same-polarity weak emitters may collectively cross a threshold that no emitter could cross alone.
- Opposite polarities cancel naturally; a zero result is indistinguishable from silence to listeners.
- Each source pays its own emission cost before cancellation.
- Dense unconditional emitters may saturate a region and erase visible distance differences; this is an intentional consequence of additive semantics, not a numerical error.
- The inspector and benchmark diagnostics SHOULD report saturation so genome authors can distinguish collective saturation from transport failure.
- Changing aggregation to maximum, mean, or soft saturation is a gameplay redesign and requires owner review.

### 6.2.2 No normal self-reception

- A cell never receives its own normal emission.
- Its source enters adjacent valid backbone edges with first-edge attenuation.
- The cell may still receive the same channel from every other source.
- This rule applies uniformly to sensors, regulation emissions, Cognocytes, Memorocytes, Photocytes, Lipocytes, oscillators, and manual sources.
- Existing feature-specific self-trigger workarounds become unnecessary and MUST be removed after parity tests prove the global rule.
- Direct requested and funded source values remain inspectable even though they are excluded from the cell's received field.
- Heat-stroke screaming is an explicit exception: it corrupts the emitting cell locally and also enters the backbone.

### 6.3 Edge attenuation

Recommended initial constants:

| Edge class | Retained strength |
|---|---:|
| Normal traversable edge | `0.95` |
| Vascular road edge | `0.9875` |
| Blocked edge | `0.0` / absent from backbone |

Attenuation MUST apply on the first edge as well as subsequent edges. The previous lossless-first-hop exception is removed because it complicates aggregation and makes the response less uniform.

The constants MUST be centralized and shared by CPU and GPU code.

### 6.4 Numerical cutoff

- Contributions below a documented epsilon MAY be discarded.
- The initial epsilon SHOULD correspond to less than half of the smallest representable final signal unit.
- A positive minimum such as `max(1, contribution)` MUST NOT be applied.
- Deep organisms MUST retain locally meaningful signals even when root-relative products would underflow.

### 6.5 Saturation

- Source values are individually clamped to `[-1000, 1000]` before funding and injection.
- Internal sums MAY exceed either bound.
- Public values saturate symmetrically at `-1000` and `1000` after complete signed accumulation and cancellation.
- Saturation MUST NOT occur at intermediate tree nodes because it would make results depend on tree orientation and grouping.

### 6.6 Hard reach

Per-source hop limits are removed. Practical reach emerges from:

- Source strength.
- Edge attenuation.
- Receiver thresholds.
- Vascular road selection.
- Numerical cutoff.

Legacy hop settings require migration as described in Section 21.

## 7. Backbone Semantics

### 7.1 Mechanical versus signal connectivity

- Every backbone edge MUST be a currently valid mechanical adhesion bond.
- A mechanical bond need not be a backbone edge.
- Signal classification is immutable for the lifetime of a bond.
- Active/standby routing state is mutable cached topology and MUST NOT be confused with immutable backbone classification.
- A mechanical-only bond MUST never be promoted, even after a topology split.
- Breaking a non-backbone bond MUST NOT require signal topology work.
- Creating or breaking a mechanical-only cross-link MUST NOT reorganize signal routing.
- Backbone membership MUST be inspectable and visualized.

### 7.2 Backbone creation eligibility

- Each cell-to-cell bond-forming operation MAY carry a heritable `creates backbone bond` property.
- This property applies independently to developmental and dynamic contact-adhesion operations.
- Environment, boundary, boulder, and other non-cell bonds MUST never be eligible.
- Classification occurs exactly once when the bond is created.
- Every affordable eligible cell-to-cell bond becomes a backbone bond.
- A backbone bond connecting two active trees becomes active at the next signal tick.
- A backbone bond whose endpoints are already active-signal-connected becomes a standby candidate and is evaluated by Section 7.3; it MUST NOT become mechanical-only merely because it forms a cycle.
- If the designated creator cannot afford construction, the cell-to-cell bond operation is dropped and no physical bond is created.
- Simultaneous candidates MUST be resolved in deterministic bond/cell order, never GPU race order.
- Mutation may add or remove eligibility for bonds created in the future but MUST NOT alter existing bond classifications.

### 7.3 Cross-links

- Glueocyte and other dynamic bonds are mechanical-only unless their creating operation is explicitly eligible and the new bond passes the creation rules above.
- Mechanical-only cross-links MUST NOT be searched, promoted, or used as replacements.
- Eligible affordable cross-links are backbone bonds from creation and pay the same one-time construction cost whether they initially become active or standby.
- The active edges MUST remain a forest; standby edges MUST NOT propagate and therefore never add duplicate physical-path contributions.
- Edge resistance is derived from attenuation as `-ln(retention)`. Route resistance is the sum of edge resistance. Implementations MUST use approved fixed-point constants for CPU/GPU deterministic comparison rather than runtime floating-point logarithms.
- When a newly formed backbone edge closes an active cycle, compare its resistance with the total resistance of the existing active path between its endpoints. It becomes active only when it is strictly lower.
- Activating a cycle shortcut MUST demote one bypassed active edge to standby. Choose the highest-resistance bypassed edge; among equal edges choose the one nearest the route's resistance midpoint; use stable bond identity only as the final exact tie-breaker.
- An exact-resistance tie MUST preserve the established active route and leave the new edge standby.
- Breaking or invalidating an active edge masks it from transport immediately.
- At the next signal tick, repair selects the standby reconnection with minimum replacement-route resistance: in-component distance to one candidate endpoint, candidate-edge resistance, and in-component distance from its other endpoint. Stable bond identity breaks exact ties.
- A valid selected standby edge becomes active; all other redundant edges remain standby. The graph remains split only if no standby backbone edge crosses the cut.
- Breaking a standby edge MUST NOT alter the active forest.
- Topology repair MUST be bounded and separately timed. Until a replacement generation commits, the invalid edge remains masked and its components remain temporarily split; stale transmission is forbidden.
- The routing choice is automatic. Genomes do not author preferred, normal, or backup routing priority.

### 7.4 Inter-organism backbones

- An eligible cell-to-cell bond may connect different organisms.
- All 16 channels cross such an edge bidirectionally.
- The two signal trees become one signal tree regardless of organism identity.
- Signals may therefore enable symbiosis, interference, parasitism, cancellation, and heat-stroke contamination across organisms.
- Organism identity MUST NOT be used to block propagation or choose the construction payer.
- The creating cell pays the one-time construction cost under Section 10.8.
- Creator selection is operation-specific and deterministic:
  - The initiating cell owns a dynamic contact bond, including a Glueocyte bond.
  - The parent creates developmental sibling bonds and newly duplicated inherited bonds; their construction costs are reserved from the parent before nutrient division.
  - Reassigning an already-existing physical bond from a parent to one child is not a new creation and is not charged again.
  - Authored scaffold endpoint A owns a scaffold bond.
  - A future symmetric operation selects one initiator before affordability is tested, using stable cell identity only to break simultaneous-initiation ties.
- If the selected creator cannot afford construction, the other endpoint MUST NOT be retried as an alternate payer.

### 7.5 Vascular traversal

The existing symmetric transport/exchange intent remains:

- Nonvascular to nonvascular: traversable.
- Vascular to vascular: traversable only when both endpoints enable signal transport.
- Vascular to nonvascular: traversable only when the vascular endpoint enables signal exchange.
- Nonvascular to vascular: the same exchange rule, bidirectionally.
- A vascular-to-vascular road uses reduced attenuation.
- Vascular signal capacity is removed.
- The lower vascular-road resistance makes vascular routes automatic signal conductors. Approximately four vascular edges have less total resistance than one normal edge; selection always uses the exact approved fixed-point constants.

Changing transport or exchange may invalidate backbone edges and MUST schedule topology repair.

### 7.6 Oculocytes

Oculocytes are source-only attachments:

- They sense and emit.
- They do not receive transported signals.
- They do not relay signals between neighboring regions.
- Their emissions enter every adjacent traversable backbone region with first-edge attenuation.
- Their own requested and funded source value remains inspectable but is absent from their received field.

An Oculocyte with neighbors in two separate trees may emit into both but MUST NOT bridge them.

- Only eligible backbone-classified bonds incident to the Oculocyte carry its emission.
- Such bonds are represented as source-attachment edges outside the relay forest. They may attach one Oculocyte to multiple trees without unioning those trees or creating a relay cycle.

### 7.7 Dead and invalid cells

- Dead cells MUST cease emitting immediately.
- Dead cells MUST be masked from transport immediately, even if topology repair is deferred.
- A backbone edge with a dead endpoint MUST be treated as invalid before the next propagation.
- Recycled cell slots MUST initialize all signal and processor state to zero before becoming visible.

## 8. Fixed Signal Clock

- Signal evaluation SHOULD run at a fixed simulation frequency, initially 15 Hz.
- It MUST NOT be defined as exactly every fourth rendered frame.
- At 60 FPS this normally corresponds to one tick every four frames.
- At other frame rates the accumulator schedules the same simulation frequency.
- Catch-up ticks MUST be capped to prevent a signal spiral of death.
- Memorocyte integration and oscillator phases MUST use signal time, not render-frame count.
- Consumers continue reading the latest completed field between ticks.

The fixed clock preserves current approximate refresh frequency while making behavior hardware-independent.

## 9. Signal Tick Pipeline

Each tick MUST observe a stable topology generation and proceed in this order:

1. **Commit topology**: swap in any completed topology repair generation.
2. **Clear live source scratch** for active channel groups.
3. **Evaluate, fund, and inject environmental, unconditional, and pathological network sources**, including heat-stroke screaming.
4. **Inject stored processor outputs** from the previous signal tick.
5. **Inject source-only attachments**, including Oculocytes, into adjacent trees.
6. **Propagate through the cached backbone**.
7. **Finalize and clamp the local field**.
8. **Apply pathological local corruption**, copying the already-generated heat-stroke scream into its source cell despite the no-self rule.
9. **Publish packed values** for existing consumers and inspectors.
10. **Evaluate Cognocytes** from the immutable finalized field.
11. **Update Memorocytes** from the same immutable finalized field.
12. **Commit all next processor outputs simultaneously**.
13. **Record visualization summaries and diagnostics**.

Actuators and lifecycle systems consume the published field after this pipeline. Topology-changing behavior triggered by that field affects a later tick.

## 10. Source Inventory and Rules

### 10.1 Oculocyte

Supported detection modes:

- Cell.
- Food.
- Light and color.
- World boundary, cave solid, and water surface.
- Self/always-on.
- Boulder.

Rules:

- Channels remain restricted to 0–7 unless the game design is changed separately.
- The source emits only when any enabled sense condition fires.
- Ray and field sensing stays dynamic and never invalidates topology.
- Legacy hop count is ignored after migration.

### 10.2 Regulation emission

- Any cell mode may emit unconditionally on channels 8–15.
- Channel or value edits update source parameters but do not rebuild topology.
- A cell may combine its regulation emission with a type-specific source only if they use different channels or if additive same-channel behavior is explicitly intended.

### 10.3 Photocyte emission

- GPU and preview MUST implement the same conditional emission.
- Emission depends on sampled light compared with the configured threshold and above/below mode.
- Source value and channel are live parameters.
- Light changes do not rebuild topology.

### 10.4 Lipocyte emission

- GPU and preview MUST implement the same conditional emission.
- Emission depends on storage fraction compared with the configured threshold and above/below mode.
- Nutrient changes do not rebuild topology.

### 10.5 Oscillator emission

- Oscillators support positive, negative, and bipolar polarity modes.
- Positive and negative modes emit unipolar time-varying strength with the selected sign.
- Bipolar mode sweeps continuously between negative and positive configured strength through zero.
- Oscillator evaluation uses fixed signal time.
- Phase and rate edits do not rebuild topology.

### 10.6 Manual/test emission

- Test sources participate as normal live sources.
- They MUST identify a source cell, channel, and value.
- They do not bypass attenuation unless an explicitly local test mode is selected.

### 10.7 Critical heat

Critical heat represents uncontrolled signal dysregulation, not a warning:

- Every signal tick, every one of the 16 channels independently selects exactly `-1000` or `+1000`.
- Values are deterministic from cell identity, channel, and signal tick so replay and CPU/GPU parity remain possible.
- The scream corrupts the overheated cell locally despite the normal no-self rule.
- The same 16 values enter every adjacent valid backbone edge and attenuate normally.
- Heat screaming may contaminate other organisms through inter-organism backbones.
- It does not change topology.
- It is not reduced by nutrient brownout and continues at full magnitude until the cell dies or leaves the critical thermal state.
- It consumes any available nutrients under the ordinary emission rate, but exhaustion does not reduce its output.

### 10.8 Signal economics

#### 10.8.1 Emission cost

Ordinary source generation consumes nutrients according to absolute magnitude and active time:

```text
requested_cost = reference_baseline_maintenance_per_second
               * 0.25
               * abs(requested_value) / 1000
               * signal_tick_seconds
```

- Each emitted channel pays independently.
- Positive and negative values cost equally.
- Opposing sources pay before cancellation.
- Passive transport and relay do not consume additional nutrients in this version.
- Routing and failover do not add a second signal charge; the originating sender alone pays emission cost.
- Low-strength and low-duty-cycle sources are proportionally inexpensive.
- A full-strength channel active continuously costs 25% of reference baseline maintenance.
- Heat-stroke screaming is charged but exempt from affordability scaling.

#### 10.8.2 Analog brownout

All ordinary emissions requested by one cell in one tick are funded proportionally as a group:

```text
total_requested_cost = sum(requested_channel_cost)
funding_fraction = clamp(available_nutrients / total_requested_cost, 0, 1)
emitted_value[channel] = requested_value[channel] * funding_fraction
paid_cost = min(available_nutrients, total_requested_cost)
```

- Polarity is preserved.
- Cost is based on the magnitude actually funded.
- A half-funded `-1000` source emits `-500`.
- Source funding occurs before backbone propagation.
- All ordinary channels from one cell brown out by the same fraction, avoiding channel-order priority.
- Heat-stroke screaming is evaluated first, consumes whatever nutrients are available, and remains full-strength; ordinary sources from the same critical cell therefore receive no remaining funding that tick.

#### 10.8.3 Backbone construction cost

- Creating a backbone bond costs 5% of the creating cell's next-division nutrient requirement.
- The creator pays once at bond creation.
- Construction is transactional. If the designated creator cannot pay the complete cost, the physical bond is not created and no nutrients are charged.
- Developmental division SHOULD reserve the construction amount before other nonessential division spending so genetically intended wiring is deterministic.
- The parent pays before nutrient division for each new physical bond produced by division, including a sibling bond and each equatorial inherited-bond duplicate. A transferred existing bond is not charged again.
- Construction payment is not refunded when the bond breaks.
- Active and standby backbone bonds pay the identical construction cost.

#### 10.8.4 No continuous per-bond maintenance

- A backbone bond has no continuous nutrient or metabolic maintenance cost after construction.
- Active and standby routing states have identical zero maintenance cost.
- Rerouting, failover, organism boundaries, transfer of an existing inherited bond, and mode changes add no bond charge. A newly allocated inherited duplicate is a new physical bond and pays the one-time construction cost.
- Creator identity MAY remain recorded for deterministic diagnostics and construction attribution, but creates no ongoing obligation and never requires transfer.
- Ordinary physical invalidation still follows the normal death and adhesion lifecycle.

## 11. Cognocyte Semantics

### 11.1 Scheduling

- Cognocytes read the finalized field at tick `t`.
- They write a stored output used as a source at tick `t + 1`.
- Every Cognocyte commits simultaneously.
- Arbitrary chains work with one tick of latency per processor.
- Cycles become deterministic discrete-time feedback systems.

### 11.2 Input rules

- Zero means silence.
- Binary operations receiving a silent required input output zero.
- `NOT` intentionally treats zero as false and may output true.
- Only positive values are Boolean true; zero and negative values are false.
- The global no-self rule already excludes the Cognocyte's own preceding output.
- It continues to receive same-channel contributions from every other source.

### 11.3 Output rules

- Results are clamped to `[-1000, 1000]` before storage.
- Negative arithmetic results remain negative.
- Divide by zero produces zero.
- Equal comparisons and effectively-zero Divide checks use an absolute
  tolerance of `0.1` in the `-1000..1000` signal scale. This is twice the
  accepted `0.05` CPU/GPU propagation error and 0.01% of full scale.
- NaN and infinity produce zero and increment a diagnostic counter.
- Multiply and Divide treat the signed range as fixed-point `-1..1`, with `1000` representing `1.0` and `-1000` representing `-1.0`.

### 11.4 Operations

| Operation | Result before clamp |
|---|---|
| Add | `A + B` |
| Subtract | `A - B` |
| Multiply | `A * B / 1000` |
| Divide | `A * 1000 / B`, or zero when `B` is effectively zero |
| Minimum | `min(A, B)` |
| Maximum | `max(A, B)` |
| Average | `(A + B) / 2` |
| Greater Than | `+1000` when `A > B`, otherwise zero |
| Less Than | `+1000` when `A < B`, otherwise zero |
| Equal | `+1000` when equal under documented tolerance, otherwise zero |
| AND | `+1000` when both inputs are positive |
| OR | `+1000` when either input is positive |
| NOT | `+1000` when `A` is zero |
| Select | `B` when `A` is positive, otherwise zero |
| ABS | `abs(A)` |
| NEGATE | `-A` |
| POSITIVE | `max(A, 0)` |
| NEGATIVE | `min(A, 0)` |
| Oscillate | configured positive, negative, or bipolar oscillator |
| Wave Oscillate | strength ramp described below |

### 11.5 Boolean strength

- Comparison and Boolean true MUST NOT be hard-coded to `1.0`.
- Boolean true is the centralized fixed strength `+1000`.
- Boolean false is zero.
- The UI SHOULD make the relationship between true strength, attenuation, and receiver threshold visible.

### 11.6 Wave oscillator migration

`Hops Oscillate` becomes `Wave Oscillate`:

- It emits a periodic ramp or shaped envelope from zero to configured peak strength.
- Nearby thresholds are crossed earlier than distant thresholds as strength rises.
- Resetting the ramp collapses the active frontier.
- It does not change topology or carry a dynamic reach value.
- It supports positive, negative, and bipolar polarity modes.

The initial envelope SHOULD be a sawtooth ramp for closest behavioral correspondence. Other envelopes may be added later.

## 12. Memorocyte Semantics

For input `x`, state `m`, configured rate `r`, and fixed signal interval `dt`:

```text
effective_rate = 1 - (1 - clamp(r, 0, 1)) ^ dt
m_next = m + (x - m) * effective_rate
```

Rules:

- Memorocytes read finalized tick `t` and emit `m_next` at tick `t + 1`.
- Input silence is zero and memory decays toward zero.
- Output is clamped to `[-1000, 1000]` and preserves polarity.
- The global no-self rule excludes the Memorocyte's own preceding output.
- Entering a Memorocyte mode initializes state and output to zero.
- Leaving a Memorocyte mode clears state and output immediately.
- New children initialize state and output to zero.
- Death clears contribution immediately.
- Memory inheritance during division is not supported in this version.

## 13. Processor and Mode Lifecycle Edge Cases

### 13.1 Mode switch

- Source-only parameter changes update live source evaluation without topology repair.
- Cognocyte operation/channel changes clear its stored output before using the new configuration.
- Memorocyte input/output/rate changes SHOULD clear memory unless an explicit preserve-state policy is later added.
- Converting into or out of Oculocyte may change relay topology and MUST schedule repair.
- Vascular transport/exchange changes MUST schedule repair.

### 13.2 Division

- The child has no source, field, or processor state until initialized.
- A developmental parent/child bond SHOULD attach the child to the parent's backbone incrementally.
- The child reads the most recently published field until the next signal tick; its own slot remains zero until then.
- Signal-based newborn apoptosis grace MUST cover at least one completed signal tick plus topology attachment latency.

### 13.3 Death and slot reuse

- Death masks source and transport in the same command ordering epoch.
- Processor state for a dead slot MUST never leak into a later occupant.
- Reused slots MUST be zeroed before the cell count or live flag exposes them.

### 13.4 Same-channel processor loops

- Own previous output is removed locally.
- Feedback through another cell remains valid and arrives on later ticks.
- Saturation and one-tick delay bound scheduling ambiguity but do not guarantee dynamical stability; unstable circuits are valid emergent behavior.

## 14. Consumer Inventory

All consumers read the same finalized packed field.

### 14.1 Movement and physical behavior

- Flagellocyte speed selection.
- Ciliocyte speed selection.
- Myocyte contraction and grip behavior.
- Siphonocyte impulse, intake, and expulsion modes.
- Luminocyte brightness and energy use.

### 14.2 Adhesion behavior

- Glueocyte cell adhesion gate.
- Glueocyte environment adhesion gate.
- Glueocyte boulder adhesion gate.

Signal-driven adhesion forms a delayed feedback loop:

```text
signal -> glue behavior -> bond changes -> topology repair -> later signal
```

The implementation MUST NOT attempt to rebuild and re-evaluate signals recursively within the same tick.

### 14.3 Lifecycle and development

- Division gate.
- Apoptosis/survival gate.
- Child A and child B mode routing.
- Signal-triggered mode switching.
- Embryocyte/Gametocyte release.
- Stemocyte five-band fate selection.
- Stemocyte signal-hold and threshold delays.

### 14.4 Diagnostics and presentation

- Cell inspector.
- Cell emissive state.
- Adhesion/backbone line visualization.
- Field reports.
- Preview test controls.

### 14.5 Signed listener response

Every signal-driven listener MUST store one polarity response mode and a nonnegative threshold magnitude:

| Mode | Normal condition |
|---|---|
| Positive | `value > 0 && value >= threshold` |
| Negative | `value < 0 && -value >= threshold` |
| Magnitude | `value != 0 && abs(value) >= threshold` |

- Optional invert negates the complete normal condition.
- A zero signal never satisfies a non-inverted listener, including when threshold is zero.
- Magnitude mode intentionally discards polarity.
- Existing listeners migrate to Positive mode only after manual genome review under Section 21.
- Listener mode is a heritable/mutable genome parameter and therefore part of the mutation surface.

## 15. Packed Public Representation

Existing consumers read the lower 11 bits of a `u32`. Retain one packed `u32` per cell per channel, but reinterpret the payload as an 11-bit signed two's-complement integer.

Recommended transitional layout:

| Bits | Meaning |
|---|---|
| 0–10 | finalized signed integer value `-1000..1000` in 11-bit two's complement |
| 11 | pathological local-corruption flag |
| 12 | saturation diagnostic flag |
| 13–15 | reserved diagnostics/visualization |
| 16–31 | reserved; hop budget removed |

All consumers MUST use shared encode/decode helpers. Masking with `0x7FF` and converting directly to positive `f32` is invalid. The decoder MUST sign-extend bit 10 before conversion. No new algorithm may depend on the old hop or source bits.

The propagation workspace uses unpacked `f32` values. Packing occurs only at publication. Requested and funded direct-source values are retained in separate diagnostic/source scratch where needed; they are not represented by a received-field source flag.

## 16. GPU Topology Representation

### 16.1 Per-cell metadata

Target metadata includes:

- Parent cell or invalid root sentinel.
- Parent backbone bond ID.
- Microtree ID.
- Local index within microtree.
- Cell role flags: relay, source-only, dead/disabled.
- Parent-edge attenuation class.
- Topology generation.

Avoid duplicating data derivable cheaply from another field.

### 16.2 Per-microtree metadata

- Node range or node-list offset and count.
- Parent microtree and attachment cells.
- Child-boundary range.
- Local traversal order.
- Active channel-group mask.
- Dirty/rebuilding state.
- Topology generation.

Recommended initial microtree size: 64–128 cells. The benchmark prototype decides the final value.

### 16.3 Macro forest

Microtrees form a much smaller forest. Macro evaluation MAY use:

- Pointer jumping.
- Depth buckets.
- A second blocking level.

The selection MUST be benchmarked on a 200,000-cell chain and many-small-tree cases.

### 16.4 Double-buffered topology metadata

- Active topology remains immutable during a signal tick.
- Repairs write secondary metadata.
- Completed repairs swap at a tick boundary.
- Immediate dead/edge-invalid masks override both generations.

## 17. GPU Value Representation and Workspace

### 17.1 Layout

- Cell-major groups of four channels are preferred initially: four `vec4<f32>` values per cell.
- This matches per-cell processor evaluation and permits skipping unused channel groups.
- Scan and microtree kernels MUST use coalesced node ordering within cached blocks.

### 17.2 Active channel groups

Compute a global four-bit mask for channel groups:

- 0: channels 0–3.
- 1: channels 4–7.
- 2: channels 8–11.
- 3: channels 12–15.

The mask is the union of channels used by sources, processors, actuators, lifecycle gates, and diagnostics requiring live values.

Unused groups MUST be skipped. Per-microtree masks MAY be added only if profiling justifies the bookkeeping.

### 17.3 Buffer reuse

The implementation SHOULD target:

- Final packed field.
- Stored processor output field.
- Two reusable `f32` scratch fields.
- Topology metadata.
- Small macro-tree and diagnostics buffers.

Scratch storage MUST be reused between source, upward, downward, and processor phases when lifetimes do not overlap.

Target total signal workspace:

| Capacity | Target |
|---:|---:|
| 100,000 cells | 28–32 MB |
| 200,000 cells | 56–64 MB |

The final design MUST report actual allocated bytes in debug/performance UI.

## 18. Blocked Propagation Algorithm

The exact kernel strategy is benchmark-gated, but it MUST implement the following mathematical message rule on the selected tree.

For directed edge `u -> v`:

```text
message(u -> v) = attenuation(u,v) *
                  (local_source[u] + sum(message(w -> u), w != v))
```

The finalized unsaturated received field at `v` excludes its own source:

```text
sum(message(u -> v), all neighbors u)
```

Heat-stroke local corruption is added separately after this solve. This rule is orientation-independent and exact for additive attenuation on a tree while enforcing the no-self constraint.

### 18.1 Microtree local solve

- Each microtree stores a bounded traversal schedule.
- Child-boundary messages are inputs to its upward solve.
- The upward solve produces one parent-boundary message.
- The downward solve receives its parent-boundary message and produces child-boundary messages plus all local cell fields.
- Local shared memory SHOULD be used where adapter limits permit.
- The implementation MUST support the minimum guaranteed WebGPU/wgpu workgroup storage limit or select a smaller block size dynamically.

### 18.2 Macro solve

- The macro forest contains approximately `N / block_size` nodes.
- It carries 16-channel boundary messages only for active groups.
- Pointer jumping over the macro forest is acceptable because the macro node count is small.
- A single 200,000-cell chain MUST remain numerically stable and complete in bounded dispatches.

### 18.3 Why global root products are forbidden

A single root-relative product such as `0.95^depth` underflows on deep trees, even though nearby deep cells should influence each other normally. Attenuation MUST be rebased at microtree boundaries or represented by another locally stable formulation.

## 19. Topology Maintenance

### 19.1 Dirty seeds, not stable organism IDs

Topology events MUST identify affected endpoint cells directly. Do not use stable organism IDs as cache ownership.

Dirty seed events include:

- Backbone bond creation.
- Backbone bond break.
- Death of a backbone cell.
- Relay-to-Oculocyte or Oculocyte-to-relay conversion.
- Vascular transport/exchange change affecting an incident edge.
- Explicit user topology edits.

Non-backbone bond churn does not dirty signal topology.

- A backbone break writes an immediate invalid-edge mask in the same ordered GPU workload that deactivates the physical bond; it cannot transmit in the next propagation even if structural repair metadata is pending.
- A newly created backbone is staged and becomes active only when topology commits at the next 15 Hz signal-tick boundary.

### 19.2 Common incremental operations

The following SHOULD avoid regional reconstruction:

- Attach a newly divided child as a leaf.
- Remove a dead non-branching leaf.
- Change source values or channels.
- Change processor operations or channels.
- Add or remove a non-backbone cross-link.
- Join two different signal trees with one newly created eligible backbone bond.

### 19.3 Fragmentation and compaction

Incremental insertion can fragment microtrees. Track:

- Live nodes per block.
- Free slots.
- Cross-block child count.
- Rebuild age.

Schedule compaction when a block crosses documented fragmentation thresholds. Compaction MUST be budgeted and double-buffered.

### 19.4 Repair budget

- Topology repair MUST have a configurable GPU work budget per rendered frame.
- Large repairs MUST continue over multiple frames.
- The last valid topology remains active while repair proceeds.
- Invalid/dead edges remain masked immediately.
- The performance monitor MUST report repair time separately from normal signal evaluation.

### 19.5 No promotion or replacement search

- A broken active backbone edge masks immediately and splits its active tree until a
  bounded topology commit selects a valid standby backbone edge across the cut.
- Existing mechanical-only bonds are irrelevant to repair and MUST NOT be searched.
- No topology job may promote or reinterpret an existing bond: failover candidates
  are limited to bonds classified as backbone bonds when they were created.
- If no standby backbone crosses the cut, reconnection requires a newly formed
  eligible and affordable backbone bond.
- These restrictions are both gameplay constraints and performance guarantees.

## 20. Performance Model and Budgets

### 20.1 Current iterative cost

Current worst-case neighbor/channel examinations:

```text
2 directions * 20 hops * N cells * 16 channels * 20 adhesions
```

At 100,000 cells this is approximately 1.28 billion examinations per refresh. Five effective hops per direction still produce approximately 320 million.

### 20.2 Target cost

Normal propagation SHOULD approximate:

```text
O(N * active_channel_groups) + O((N / block_size) * channels * macro_solve_rounds)
```

It MUST NOT contain a hot loop over all adhesions.

### 20.3 Provisional budgets at 200,000 live cells

| Metric | Target |
|---|---:|
| Discrete-GPU signal tick | under 2 ms |
| Conservative integrated-GPU signal tick | under 4 ms |
| Average 15 Hz frame cost | approximately 0.5–1.0 ms or less |
| Signal workspace | 64 MB or less |
| CPU readbacks | zero |
| Additional queue submissions | zero |
| Unbounded topology work in one frame | zero |

These are acceptance targets, not promises. Target hardware testing may revise them before integration, but revisions MUST be explicit.

### 20.4 Early-outs

- Skip all signal work when no source, processor, listener, inverted gate, or diagnostic requires it.
- Skip unused channel groups.
- Reuse the previous field when no dynamic source, processor output, or pathological corruption can change.
- Dynamic sources include moving Oculocyte rays, oscillators, Memorocytes, heat, light, nutrients, and manual inputs.
- Avoid fine-grained microtree early-outs until profiling proves they exceed bookkeeping cost.

## 21. Data and UI Migration

### 21.1 Removed settings

The following settings become obsolete:

- `oculocyte_signal_hops`.
- `regulation_emit_hops`.
- `photocyte_emit_hops`.
- `lipocyte_emit_hops`.
- `cognocyte_output_hops`.
- `memorocyte_output_hops`.
- `vascular_signal_capacity`.

### 21.2 Save compatibility

Owner override (2026-08-13): Bio-Spheres will rebuild genomes for this release. The compatibility-period requirements below are superseded for the current implementation: deprecated signal fields are rejected with a named error, and no unsigned or hop-based genome is migrated automatically.

- Existing serialized fields SHOULD remain readable for at least one compatibility period.
- Loading legacy saves ignores deprecated hop and capacity values after logging a migration notice.
- Unsigned source values and thresholds MUST NOT be automatically scaled, clamped, or assigned listener polarity modes in saved genome data.
- A legacy genome is marked `signal review required` and MUST NOT enter ordinary simulation until the user explicitly reviews and saves its signal settings under the new signed schema.
- The editor SHOULD present old values alongside the new valid range and identify every field requiring a decision.
- Runtime safety clamps remain mandatory if invalid legacy data reaches a shader, but such clamping is not considered migration and MUST generate a diagnostic.
- Saving in the new format SHOULD omit deprecated fields once versioned serialization supports it.
- Genome mutation code MUST stop mutating deprecated fields.
- UI controls and tooltips MUST be removed or replaced.

### 21.3 Replacement controls

- Display the fixed Boolean true strength of `+1000`.
- Rename `Hops Oscillate` to `Wave Oscillate`.
- Replace step-count/reach controls with envelope, polarity, or peak-strength controls.
- Add listener polarity controls: Positive, Negative, and Magnitude.
- Add unary Cognocyte operations: ABS, NEGATE, POSITIVE, and NEGATIVE.
- Show effective attenuation examples near source strength and receiver threshold controls.
- Add a backbone visualization toggle.

### 21.4 Inspector

The inspector SHOULD display:

- Final local value per channel.
- Direct local source contribution.
- Pathological local-corruption state.
- Stored Cognocyte/Memorocyte output.
- Signal tree ID and microtree ID.
- Parent backbone cell/bond.
- Topology generation and repair status.

## 22. Visualization

- Backbone bonds MUST be visually distinguishable from mechanical-only bonds in debug mode.
- Signal flow MUST NOT be inferred from obsolete hop-bit differences.
- An edge is active for visualization when its directed message exceeds a configurable visual epsilon.
- Positive diagnostic flow is red; negative diagnostic flow is blue; brightness represents absolute magnitude.
- Zero and exact cancellation are dark.
- Direction MAY be shown using animation or intensity gradient.
- Multiple active channels require a deterministic display policy: selected channel, strongest channel, or blended diagnostic view.
- Oculocyte source injection edges should be visible even though the Oculocyte is not a relay node.
- Repairing or stale topology SHOULD have a debug overlay.
- This visualization is diagnostic only. Organisms use Luminocytes when visible state communication is part of gameplay.

## 23. CPU Preview

- The preview MUST implement the same source, processor, timing, and attenuation semantics.
- It may use a direct linear two-pass tree traversal because preview organisms are smaller and CPU dependencies are straightforward.
- It MUST still use one-tick processor delay.
- It MUST treat Oculocytes, heat screaming, signed saturation, no-self reception, economics, and inter-organism links identically.
- Preview tests serve as the reference oracle for small deterministic trees.

The preview must no longer contain features absent from GPU gameplay without an explicit preview-only label.

## 24. Verification Strategy

### 24.1 Mathematical unit tests

Test exact expected results for:

- Single source on a line.
- Multiple sources on a line.
- Positive/negative cancellation before saturation.
- Branching tree fan-out and fan-in.
- Normal and vascular attenuation.
- Saturation after complete accumulation.
- Orientation independence under different chosen roots.
- Deep locally active paths without underflow.
- Oculocyte injection without relay.
- No normal self-reception.
- Inter-organism full-channel joining.
- Split trees and dead-edge masking.

### 24.2 Processor tests

- Every Cognocyte operation.
- Normalized signed Multiply and Divide, including polarity and saturation.
- Negative, overflow, divide-by-zero, NaN, and infinity handling.
- Boolean strength attenuation.
- NOT of silence.
- Positive-only Boolean truth.
- ABS, NEGATE, POSITIVE, and NEGATIVE.
- Normal own-output exclusion.
- Two-cell feedback loop.
- Multi-stage Cognocyte chain with one-tick-per-stage latency.
- Memorocyte frame-rate independence.
- Memorocyte decay after input disappears.
- Mode entry/exit state reset.
- Division and slot-reuse state reset.
- Wave oscillator threshold-front progression.
- Positive, negative, and bipolar oscillators.

### 24.2.1 Economic tests

- Full-strength one-channel cost matches 25% reference baseline per second.
- Cost is symmetric for positive and negative values.
- Duty cycle and magnitude scale cost linearly.
- Multiple channels brown out proportionally rather than by channel order.
- Cancellation does not refund either source.
- Heat screaming remains full-strength without nutrients.
- Backbone construction consumes 5% of next-division requirement.
- Unaffordable eligible cell-to-cell bond operations create no physical bond and charge no nutrients.
- Backbone bonds incur no continuous per-bond maintenance cost across any organism boundary.

### 24.3 Consumer tests

Verify finalized fields drive:

- Flagellocyte.
- Ciliocyte.
- Myocyte.
- Siphonocyte.
- Luminocyte.
- Glueocyte cell/environment/boulder adhesion.
- Division.
- Apoptosis and newborn grace.
- Child mode routing.
- Mode switching and self-trigger behavior.
- Embryocyte/Gametocyte release.
- Stemocyte bands and delays.

### 24.4 CPU/GPU parity

For deterministic synthetic trees:

- Compare all 16 final channels.
- Compare processor outputs across multiple ticks.
- Compare topology split and repair behavior.
- Define a small absolute tolerance before tests are written.
- Packed integer outputs SHOULD normally match exactly after final quantization.

### 24.5 Property tests

- Before saturation, the unsaturated field equals the signed linear sum of all non-self source/path contributions.
- Adding a positive source cannot reduce the unsaturated field; adding a negative source cannot increase it.
- Equal opposite contributions cancel to zero independent of source order.
- Re-rooting the same selected tree does not change results.
- Adding a mechanical non-backbone edge does not change results.
- A broken backbone edge cannot transmit after immediate masking.
- A newly created backbone does not transmit before the next signal-tick commit.
- An existing mechanical-only edge can never become a backbone.
- Results do not depend on cell IDs except where IDs select deterministic topology ties.

## 25. Performance Benchmark Matrix

The benchmark prototype MUST cover 20k, 100k, and 200k cells.

### 25.1 Tree shapes

| Shape | Purpose |
|---|---|
| One long chain | Maximum depth and numerical stability |
| One star | Extreme degree and accumulation |
| Balanced tree | Large normal topology |
| Many two-cell trees | Segmentation and dispatch overhead |
| Gameplay-like mixed organisms | Representative workload |
| Dense mechanical graph with sparse backbone | Independence from adhesion density |

### 25.2 Signal workloads

- No active signals but active inverted listeners.
- One channel and one source.
- One active `vec4` channel group.
- All 16 channels.
- Every cell emits.
- Every cell is a Cognocyte.
- Every cell is a Memorocyte.
- Saturated fan-in.
- Signed cancellation fan-in.
- Continuous oscillators.
- Sixteen-channel heat screaming.

### 25.3 Topology workloads

- Continuous leaf division.
- Non-backbone glue bond churn.
- Continuous backbone leaf loss.
- Repeated central backbone breaks.
- Vascular transport mode churn.
- Microtree fragmentation and compaction.
- Full scene rebuild.

### 25.4 Metrics

- GPU milliseconds for source evaluation.
- GPU milliseconds for normal propagation.
- GPU milliseconds for processors.
- GPU milliseconds for topology repair.
- Bytes allocated.
- Bytes read/written per tick where tools permit estimation.
- Dispatch count.
- Dirty microtrees and repair backlog.
- Signal tick age/staleness.
- p50, p95, and worst observed timing.

## 26. Implementation Phases and Gates

### Phase 0: Preserve and measure the baseline

- Record current GPU timing at 20k, 100k, and 200k where possible.
- Save representative genomes and synthetic stress scenes.
- Record current signal semantics tests before deleting code.
- Separate unrelated working-tree changes from signal work before implementation.

Gate: Reproducible baseline measurements and scenes exist.

### Phase 1: Standalone propagation benchmark

- Implement synthetic cached forests and microtrees.
- Implement all-16-channel blocked propagation without live game integration.
- Compare block sizes and macro solve strategies.
- Validate numerical results against a CPU oracle.

Gate: Meets timing, memory, depth, and correctness budgets at 200k.

### Phase 2: Authoritative CPU semantics

- Refactor preview signal code to the new tree equations and fixed clock.
- Implement synchronous Cognocyte/Memorocyte outputs.
- Implement source and lifecycle rules.
- Add complete semantic tests.

Gate: Reviewed CPU reference behavior and passing tests.

### Phase 3: GPU value pipeline behind a feature flag

- Integrate source evaluation, processor state, blocked solve, and publication.
- Continue using a static or externally generated backbone initially.
- Preserve the old implementation for A/B timing and visual comparison.

Gate: CPU/GPU parity and performance targets pass.

### Phase 4: Incremental backbone lifecycle

- Implement developmental leaf attachment.
- Implement immediate invalid masks.
- Implement immutable active/standby backbone classification for affordable eligible cycle bonds.
- Implement deterministic fixed-point route resistance, strictly-better cycle exchange, split-on-break behavior, and standby failover.
- Implement microtree repair, compaction, generations, and budgeting.
- Keep normal propagation and topology repair separately timed and diagnosed.

Gate: Topology stress tests pass without frame spikes, stale transmission through invalid edges, cyclic amplification, nondeterministic route selection, or CPU/GPU active-forest disagreement.

Implementation acceptance note (2026-08-13): Phase 4 uses a persistent, resumable GPU repair cursor with a fixed operation budget. Developmental leaf attachment takes the fast path; a regional repair that exceeds the budget continues across rendered frames while the broken edge stays immediately masked. Detailed hardware, parity, memory, dispatch, and timing results are recorded in `benchmarks/signal_backbone/results/2026-08-13-phase4-topology-acceptance.md`.

### Phase 5: Consumer and visualization migration

- Migrate every reader of signal flags.
- Replace hop-based adhesion-line visualization.
- Expand inspector and performance diagnostics.
- Validate all cell behavior integrations.

Gate: Consumer inventory is fully checked off.

Implementation acceptance note (2026-08-13): Phase 5 migrated the complete consumer inventory to signed decoding and independent Positive/Negative/Magnitude response modes without adding gameplay buffers, dispatches, submissions, or readbacks. Yellow/black route visualization and signed red/blue inspector diagnostics are active. The release suite passed 194/194 and a real-GPU listener oracle passed 42/42 cases. Detailed inventory and verification are recorded in `benchmarks/signal_backbone/results/2026-08-13-phase5-consumer-acceptance.md`.

### Phase 6: Save/UI migration

- Deprecate hop and capacity fields.
- Replace oscillator controls.
- Add Boolean strength and backbone UI.
- Validate legacy save loading.

Gate: Migration tests and user-facing review pass.

Implementation acceptance note (2026-08-13): Phase 6 removed the deprecated hop, vascular-capacity, and oscillator-step fields from the genome, save, mutation, preview-hash, GPU-upload, and editor surfaces. Wave Oscillate is a signed sawtooth strength envelope, all twenty Cognocyte operations and thirteen listener polarity selectors are exposed, and the UI documents fixed Boolean strength, attenuation, backbone route colors, and construction economics. Per the owner's explicit clean-rebuild decision, legacy signal fields are rejected rather than migrated. The release suite passed 195/195 before the final rejection guard; the focused serialization suite then passed 9/9, and all affected WGSL passed `naga` validation. Detailed results are recorded in `benchmarks/signal_backbone/results/2026-08-13-phase6-save-ui-acceptance.md`.

### Phase 7: Remove old systems

- Remove iterative forward/reverse propagation.
- Remove obsolete signal buffers and pipelines.
- Remove Claude's route-table buffers, shader, and dirty-ID wiring.
- Remove deprecated runtime fields after the compatibility policy allows.

Gate: Feature flag no longer needed; benchmark and regression suite pass in release builds.

Implementation acceptance note (2026-08-13): Phase 7 deleted the iterative CPU BFS/travel-budget path, transitional hop field and flow tracker, old clear/sense/forward/reverse/combine GPU system, obsolete workspace buffers, and the rejected route-cache/dirty-ID surface. The cached backbone is now the only signal implementation in preview and gameplay; no live feature switch remains. The serialized release suite passed 196/196, all affected WGSL validated, the complete 198-case value matrix passed with a worst 200k p95 of 1.9517 ms and 59.510 MiB allocation, and bounded topology repair passed all sizes with a worst 200k central-repair p95 of 1.059840 ms. Detailed final evidence is recorded in `benchmarks/signal_backbone/results/2026-08-13-phase7-final-acceptance.md`.

## 27. Instrumentation Requirements

Add separate GPU timestamp segments or subsegments for:

- Signal source evaluation.
- Backbone propagation.
- Signal processors.
- Topology repair.

Expose debug counters for:

- Live signal trees.
- Live microtrees.
- Active channel-group mask.
- Dirty and rebuilding microtrees.
- Repair backlog age.
- Backbone edges and mechanical-only bonds.
- Saturated cells per channel group.
- Invalid processor outputs.
- Signal memory allocation.

Performance regressions MUST be detectable without external profiling tools.

## 28. Failure Handling and Fallbacks

- If topology metadata is invalid, affected cells receive local sources and overlays only; do not fall back to unbounded graph propagation.
- If repair backlog exceeds a threshold, report it and degrade topology freshness, not frame time.
- If adapter limits cannot support the selected block size, choose a smaller supported size.
- If signal workspace allocation fails, disable signal processing with an explicit user-visible error rather than running a partially bound pipeline.
- Debug builds SHOULD validate parent cycles, attachment consistency, generation ownership, and node uniqueness.
- Release shaders MUST bounds-check externally generated indices where practical.

## 29. Review Decisions

Owner review resolved the following decisions:

- [x] Signals travel only along explicitly created backbone bonds.
- [x] Mechanical-only bonds never carry signals and can never be promoted.
- [x] Any eligible cell-to-cell bond operation may create a backbone at birth, including across organisms.
- [x] Affordable cycle-forming eligible bonds remain immutable standby backbones and may replace an active route only when strictly lower resistance.
- [x] Breaking an active backbone masks immediately; bounded next-tick repair selects the lowest-resistance valid standby reconnection when one exists.
- [x] Inter-organism backbone links carry all 16 channels bidirectionally.
- [x] Signed same-channel sources add, cancel, and saturate symmetrically at `-1000..1000`.
- [x] A cell does not receive its own normal emission.
- [x] Signal attenuation applies on the first edge.
- [x] Hard hop limits and vascular signal capacity are removed.
- [x] Normal retention is `0.95`; vascular-road retention is `0.9875`.
- [x] Signals update on a fixed 15 Hz clock.
- [x] Cognocyte and Memorocyte outputs have one signal-tick latency.
- [x] Zero is silence and false; negative is also false for Boolean logic.
- [x] Boolean true is fixed at `+1000`.
- [x] Multiply and Divide use normalized fixed-point arithmetic with scale 1000.
- [x] Cognocytes include ABS, NEGATE, POSITIVE, and NEGATIVE.
- [x] Oscillators support positive, negative, and bipolar modes.
- [x] Every listener supports Positive, Negative, and Magnitude response.
- [x] Processor state/output resets on mode changes and division.
- [x] Critical heat creates full-strength deterministic signed chaos on all 16 channels locally and through the backbone, without brownout.
- [x] Oculocytes are source-only and cannot relay.
- [x] CPU preview and GPU gameplay use the same signal feature set.
- [x] Backbone construction and direct emission consume approved nutrient costs; bonds have no continuous per-bond maintenance cost.
- [x] Ordinary underfunded emitters brown out proportionally.
- [x] Legacy unsigned genomes receive no automatic value migration and require manual review.
- [x] Debug signal polarity is red/blue; gameplay-visible signaling remains a Luminocyte role.
- [x] Breaks mask immediately; additions activate at the next signal tick.
- [x] The blocked benchmark must pass before game integration begins.

## 30. Definition of Done

The overhaul is complete only when:

- All accepted review decisions are implemented and documented.
- CPU and GPU parity tests pass.
- Every listed source, processor, consumer, and lifecycle edge case is covered.
- The 200,000-cell benchmark matrix meets approved timing and memory budgets.
- Signal cost does not grow with non-backbone adhesion density or emitter count.
- Deep trees remain numerically stable.
- Topology churn cannot create unbounded frame work.
- Invalid/dead edges never transmit after masking.
- Legacy signal fields are handled safely according to the owner-approved policy (explicit rejection for the current clean rebuild; no automatic unsigned migration).
- Old propagation, hop-control, route-table, and dirty-stable-ID code is removed.
- The performance monitor exposes normal signal and topology repair cost separately.
- Backbone behavior is visible and understandable in the UI.
