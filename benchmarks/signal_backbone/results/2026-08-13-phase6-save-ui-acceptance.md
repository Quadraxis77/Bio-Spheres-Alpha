# Phase 6 Save and UI Acceptance — 2026-08-13

## Result

Phase 6 save/UI migration gate: **PASS**.

The project owner explicitly chose a clean rebuild instead of legacy unsigned-genome compatibility. New saves use only the signed backbone schema. A genome containing any removed hop, capacity, or oscillator-step field is rejected with a named `UnsupportedLegacySignalField` error; it is never silently migrated or admitted with default signal semantics.

## Removed schema and mutation surface

- Removed `oculocyte_signal_hops`, `regulation_emit_hops`, `photocyte_emit_hops`, `lipocyte_emit_hops`, `cognocyte_output_hops`, `memorocyte_output_hops`, `vascular_signal_capacity`, and `cognocyte_oscillator_step_count` from `ModeSettings` and serialized mode settings.
- Removed defaults, procedural assignments, preview hashing, multi-mode editor merging, CPU source reads, GPU upload reads, and mutation entries for those fields.
- Existing fixed-width GPU parameter words keep the removed components reserved at zero, so Phase 6 adds no buffer, bind group, dispatch, submission, or readback.
- Signed authored sources are constrained to `-1000..1000`; listener thresholds are constrained to `0..1000` after Positive/Negative/Magnitude response transformation.

## User-facing behavior

- `Hops Oscillate` is now `Wave Oscillate`, a fixed-time sawtooth strength envelope with Positive, Negative, and Bipolar polarity. It never changes routing reach.
- The editor exposes all 20 approved Cognocyte operations, including ABS, NEGATE, POSITIVE, and NEGATIVE.
- The editor exposes independent Positive, Negative, and Magnitude selectors for all 13 listeners.
- The UI displays fixed Boolean true strength `+1000`, first-edge attenuation examples (`950` normal, `987.5` vascular road), and signed source ranges.
- Backbone creation UI explains the one-time 5% construction cost, zero continuous bond maintenance, and yellow-active/black-standby route colors.
- Tutorial text no longer describes hop budgets, lossless first hops, unsigned `2047`, or generic mechanical adhesion as the signal network.

## Verification

Hardware for real-GPU tests: NVIDIA GeForce GTX 1080, Vulkan, driver 582.66 (the adapter recorded by the Phase 3–5 acceptance runs on this workstation).

- Release library suite after the schema/UI implementation: **195 passed, 0 failed, 0 ignored**; test execution 2.19 s after a 2 m 22 s release compile.
- Focused serialization suite after adding explicit legacy-field rejection: **9 passed, 0 failed**; includes signed round trips, deprecated-field omission, and rejection rather than migration.
- Real-GPU pipeline creation, signed CPU/GPU propagation parity, 20-operation Cognocyte matrix, listener polarity parity, processor lifecycle/economics, topology repair parity, and integrated 200k GPU gate all passed within the release library suite.
- `naga --bulk-validate` passed for `mutation.wgsl`, `mode_switch.wgsl`, `lifecycle_unified.wgsl`, and `signal_backbone_value.wgsl`.
- Deprecated-surface static audit found no removed genome, serialization, mutation, upload, or UI identifiers and no remaining `2047` signal scale in `src` or signal shaders.
- `git diff --check`: **PASS**.

## GPU scheduling and memory impact

- New gameplay buffers: **0**
- New gameplay allocations: **0 bytes**
- New gameplay dispatches: **0**
- New queue submissions: **0**
- New CPU readbacks: **0**

Normal signal evaluation and bounded topology repair remain independently timestamped. Phase 6 does not change the accepted Phase 3/4 dispatch schedule or topology budget.

## Remaining work

Phase 7 removes the now-unreachable iterative preview propagation functions, their transitional `SignalEmission::hops` field, obsolete signal buffers/pipelines, and any remaining abandoned route-table or dirty-stable-ID infrastructure. The user-facing genome/save/mutation surface no longer depends on any of them.
