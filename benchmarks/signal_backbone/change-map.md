# Signal Backbone Change Map

Audit date: 2026-08-12. This map is intentionally conservative because the
working tree contains unrelated user work and an abandoned implementation.

## Rejected bounded route-cache work

Do not complete or use these changes as the Cached Signal Backbone design:

- `shaders/signal_route_discover.wgsl`: abandoned eight-route-slot discovery.
- Route-slot buffers, bind groups, and dispatch wiring in
  `src/simulation/gpu_physics/adhesion_buffers.rs`,
  `src/simulation/gpu_physics/compute_pipelines.rs`, and
  `src/simulation/gpu_physics/gpu_scene_integration.rs`.
- Stable-organism-ID dirty propagation in adhesion, mutation, division,
  lifecycle, and mode-switch shaders and their Rust-side buffer wiring.
- `mode_pool_topology_dirty` invalidation associated only with that route cache.

These files may also contain useful or unrelated edits. Removal must therefore
be done hunk-by-hunk only after the replacement path is independently accepted.

## Useful pre-existing signal work

- `src/scene/gpu_timer.rs` and timestamp placement in
  `src/scene/gpu_scene.rs`: isolates legacy signal GPU time from other physics.
- Iterative-hop convergence controls in `shaders/signal_propagate.wgsl` and
  corresponding pipeline code: useful as a measurable legacy baseline, not as
  the new propagation algorithm.
- Periodic signal refresh correction in `src/scene/gpu_scene.rs`: prevents
  continuous slot growth from silently defeating throttling.

These remain baseline/reference candidates until their individual ownership is
confirmed during live integration.

## Unrelated user work to preserve

- Collision/spatial-grid and fluid broadphase changes.
- Embryocyte newborn-grace behavior and integration tests.
- Kira PCM/audio changes.
- Organism procedural-design documents and other UI/rendering work.

## Isolated Phase 0/1 files

- `src/simulation/signal_backbone_bench.rs`
- `src/bin/signal_backbone_bench.rs`
- `shaders/signal_backbone_chain_bench.wgsl`
- `benchmarks/signal_backbone/**`
- The `signal-backbone-bench` Cargo feature and binary declaration.

These are not connected to preview or live GPU gameplay.

## Production files expected in later phases

No production changes are authorized by Phase 0/1 results. Later phases will
require carefully scoped changes to preview signal evaluation, GPU buffers and
pipelines, adhesion classification/lifecycle, mutation/division initialization,
serialization, UI controls, consumers, diagnostics, and save handling.

