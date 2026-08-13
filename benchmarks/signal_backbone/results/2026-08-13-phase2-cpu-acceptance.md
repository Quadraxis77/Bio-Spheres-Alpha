# Phase 2 CPU Acceptance - 2026-08-13

Status: PASS. CPU preview now uses the approved fixed-clock, signed cached-tree
semantics. This phase does not integrate the Phase 1 WGSL value pipeline or
implement Phase 4 bond-classification/repair policy.

## Implemented behavior

- Fixed 15 Hz signal clock stored in canonical state, with four-tick capped
  catch-up and retention of the latest published field between ticks.
- Explicit immutable-backbone input: only bonds carrying
  `BOND_FLAG_SIGNAL_BACKBONE` enter the CPU forest. Mechanical-only bonds are
  ignored and are never searched as replacement paths.
- Source-independent cached topology. Oculocyte attachment identity and
  retention are cached; live source values are evaluated every tick.
- Signed 16-channel propagation, first-edge attenuation, complete additive
  accumulation/cancellation, final-only saturation, and no normal self receipt.
- Normal retention 0.95 and vascular-road retention 0.9875. Hop limits and
  vascular capacity are absent from the authoritative tick.
- Oculocyte source-only attachments cannot receive or relay. Organism identity
  is absent from topology and transport decisions.
- Ordinary per-cell source funding uses proportional analog brownout. Costs use
  the reference preview metabolism of 1 nutrient/second. Critical heat emits
  deterministic independent +/-1000 values on all channels locally and through
  the backbone without brownout; its ordinary sources receive no funding.
- Manual, regulation, Oculocyte, Photocyte, Lipocyte, stored processor, and
  oscillator sources enter the same source/funding pipeline. Photocyte emission
  consumes an explicit per-cell preview light sample.
- Cognocytes and Memorocytes read one immutable finalized field and commit
  outputs simultaneously for the next tick. Configuration changes, mode
  switches, division, death, and slot reuse clear or preserve state as approved.
- Cognocyte fixed-point Multiply/Divide, +1000 Boolean truth, 0.1 decision
  epsilon, ABS, NEGATE, POSITIVE, NEGATIVE, signed oscillator polarity, and
  Wave Oscillate sawtooth semantics.
- Positive, Negative, and Magnitude listener response functions are centralized.
  Consumer field migration remains Phase 5.
- Signed oscillator polarity is serialized in the clean-slate genome schema.

## Condensed verification

Fast iteration used six consolidated signal-system tests rather than a separate
test binary for every behavior. Those fixtures exercise complete ticks and
jointly cover clocking, funding, signed propagation, cancellation, vascular
retention, explicit mechanical/backbone distinction, break/new-bond behavior,
processor latency, Memorocyte decay, heat, listeners, Oculocyte attachments,
inter-organism crossing, conditional Photocytes, oscillator polarity, mode
reset, death, division initialization, and slot reuse.

Additional focused results:

- Phase 2 end-to-end signal fixtures: 6/6 pass.
- Cognocyte operation tests: 7/7 pass.
- Serialization tests: 6/6 pass.
- Phase 1 mathematical/cache tests after source-independent cache change: 24/24 pass.
- Consolidated repository gate: 172/172 library tests and 2/2 integration tests pass.

## Phase boundary and remaining risks

- Existing preview bonds are not automatically reclassified. Phase 4 owns
  eligibility, immutable creation classification, construction/maintenance
  economics, immediate invalid masks, generation commits, and bounded repair.
- `signal_light_samples` is the authoritative CPU input for conditional
  Photocyte emission. Scene-specific light sampling must populate it when the
  preview lighting model is connected; changing the sample never rebuilds topology.
- Existing consumers still expose their legacy positive-only authored fields.
  Phase 5 maps those fields to Positive, Negative, and Magnitude response modes.
- The live GPU path remains unchanged. Phase 3 must integrate source evaluation,
  processor state, propagation, publication, CPU/GPU parity, and split timings
  behind its feature flag.

Phase 2 is complete. Phase 3 is the next implementation phase.
