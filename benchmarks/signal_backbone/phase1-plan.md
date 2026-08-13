# Phase 1 Standalone Implementation Plan

Status: complete. The acceptance evidence is recorded in
`results/2026-08-13-phase1-acceptance.md`.

This phase remains isolated from preview, live GPU gameplay, adhesion lifecycle,
saves, and the rejected route-cache implementation.

## Accepted mathematical contract

For each directed tree edge `u -> v`:

```text
message(u -> v) = attenuation(u,v) *
                  (local_source[u] + sum(message(w -> u), w != v))
```

Final reception excludes the cell's own ordinary source, accumulates signed
contributions completely, and clamps only at publication. The general
microtree CPU schedule already matches the independent two-pass oracle for all
six shapes at block sizes 64, 96, and 128.

## Flattened topology metadata contract

The standalone GPU prototype will upload immutable arrays generated from the
cached forest:

- Per cell: parent cell, parent-edge retention, microtree ID, local traversal
  index, role flags, and topology generation.
- Per microtree: node-list offset/count, parent microtree, attachment node,
  external parent cell, child-boundary offset/count, and topology generation.
- Traversal: one flattened parent-before-child node list; local reverse order is
  obtained from each microtree range.
- Macro forest: parent IDs plus deterministic depth-bucket offsets/IDs.
- Values: cell-major four `vec4<f32>` groups, with inactive groups skipped.

No metadata depends on organism identity, emitter count, mechanical-only edge
density, or the rejected eight-slot route representation.

## GPU implementation sequence

1. Add a topology upload/inspection mode to the standalone binary and verify
   byte counts and invariants against the CPU schedule.
2. Implement one bounded local-up kernel over the flattened microtree node
   lists, producing one parent-boundary message per microtree.
3. Implement topology-selected macro evaluation:
   - pointer jumping for macro forests whose maximum child count is at most one;
   - deterministic depth buckets for branching macro forests.
4. Implement one bounded local-down/finalize kernel consuming the parent
   boundary and publishing every local received field.
5. Reuse the same value/scratch buffers across phases; do not request device
   limits above the project's normal WebGPU-compatible limits.
6. Timestamp the complete standalone value tick. Synthetic source preparation
   remains outside the tick, processor rows include processor traffic, and
   topology upload/validation remains separate because Phase 1 topology is
   externally generated and static. The live source/processor timing split is
   a Phase 3 integration metric, and incremental repair timing begins in Phase
   4.
7. Compare all 16 channels with the independent CPU oracle and compare stored
   processor outputs for the processor rows.
8. Run 20k/100k/200k across the full matrix, then compare 64/96/128 blocks and
   retain raw results and a summarized report.

## Boundedness contract

- Exactly two local propagation passes per active channel group.
- Pointer-jumping macro passes are `ceil(log2(maximum macro path nodes))`.
- Branching macro passes are bounded by deterministic macro depth buckets.
- No whole-graph retry loop, convergence loop, CPU readback, or extra queue
  submission is permitted in the candidate tick.
- A result that exceeds the dispatch, memory, numerical, or 2 ms 200k p95 gate
  is diagnosed and iterated in this harness; it is not integrated live.

## Gate evidence required before Phase 2

- Actual wgpu pipeline creation passes on the measured adapter.
- CPU/GPU maximum absolute transport error is at most `0.05`; processor
  comparison uses the documented `0.1` absolute decision tolerance.
- 200k discrete-GPU p95 is at most 2 ms for every topology and required value
  workload.
- Total standalone workspace is at most 64 MiB at 200k.
- Deep-chain local influence remains numerically stable.
- Timings, allocation bytes, estimated traffic, dispatch counts, adapter,
  driver, backend, toolchain, and unmet risks are recorded.

Only after this gate passes may Phase 2 change authoritative CPU preview
semantics. Live GPU integration remains Phase 3.

The gate passed on 2026-08-13. This does not authorize live GPU integration;
Phase 2 remains the next implementation phase.
