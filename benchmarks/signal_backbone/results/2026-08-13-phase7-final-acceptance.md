# Phase 7 Final Overhaul Acceptance — 2026-08-13

## Result

Phase 7 and the complete Cached Signal Backbone overhaul: **PASS**.

Hardware: NVIDIA GeForce GTX 1080, Vulkan, NVIDIA driver 582.66. Rust profile: `release`.

## Removed legacy systems

- Deleted the iterative CPU per-emitter BFS/travel-budget propagator, its lossless-first-hop behavior, per-emitter scratch allocation, strongest-path overwrite, and source-self injection.
- Removed `SignalEmission::hops`, dead Cognocyte/Memorocyte duplicate evaluators, the unused preview test propagator, `SignalFlowTracker`, and its canonical-state allocation.
- Deleted the old `signal_clear.wgsl`, `signal_sense.wgsl`, and `signal_propagate.wgsl` modules.
- Confirmed the old forward, reverse, and combine pipelines; sense bind groups and dummy buffers; and `signal_flags_next`, `signal_flags_forward`, and `signal_flags_seed` are absent.
- Confirmed Claude's `signal_route_discover.wgsl`, eight-slot route table/cache, mode-pool topology dirty flag, and dirty-stable-ID wiring are absent.
- Confirmed there is no conditional live-system feature path. The `signal-backbone-bench` Cargo feature only isolates diagnostic binaries from normal product builds.
- Retained one public packed `signal_flags` field because it is the authoritative signed gameplay output consumed by listeners, rendering, inspection, and lifecycle shaders; it is not an obsolete propagation workspace.

## Correctness and validation

- Static forbidden-symbol audit: **PASS**.
- `git diff --check`: **PASS**.
- Naga validation: **PASS** for all cached-backbone modules and every affected producer, consumer, lifecycle, mutation, visualization, and inspector shader.
- Release library suite, serialized to avoid multiple test devices contaminating GPU timing: **196 passed, 0 failed, 0 ignored**.
- Real-device pipeline creation, CPU/GPU propagation parity, every Cognocyte operation, Memorocyte lifecycle, listener polarity, heat, economics, processor resets, topology repair, and slot reuse passed.
- One parallel-suite timing sample measured 2.3951 ms because multiple GPU tests ran concurrently. The unchanged strict 2.0 ms gate passed in isolation at 1.8115 ms and passed again in the serialized full suite. No threshold was relaxed.

## Final 198-case value matrix

Configuration: 5 warm-up ticks, 20 timed samples, standard nearest-rank p95, integrated production pipeline, one queue submission per gameplay tick, zero production CPU readbacks.

- Cases: **198/198 passed**
- Sizes: 20,000, 100,000, and 200,000 cells
- Topologies: chain, star, balanced binary, many pairs, gameplay mixed, and dense mechanical/sparse backbone
- Workloads: silent/inverted, one source, vec4, all-channel sparse, every-cell emission, Cognocytes, Memorocytes, saturation, signed cancellation, oscillators, and heat screaming
- Maximum parity error: **0.021729** (gate 0.05)
- Maximum mismatches: **0**
- Maximum dispatch count: **54**

| Cells | Cases | Maximum p50 | Maximum p95 | Maximum sample | Maximum memory |
|---:|---:|---:|---:|---:|---:|
| 20,000 | 66 | 0.2294 ms | 0.2314 ms | 0.2335 ms | 5.952 MiB |
| 100,000 | 66 | 0.8387 ms | 0.9892 ms | 1.2503 ms | 29.756 MiB |
| 200,000 | 66 | 1.6927 ms | 1.9517 ms | 2.1955 ms | 59.510 MiB |

## Final bounded-topology matrix

Each row uses one topology dispatch per rendered frame and a 1,024 node/bond operation budget.

| Cells | Workload | p95 | Allocation | Completion frames | Result |
|---:|---|---:|---:|---:|---|
| 20,000 | Developmental leaf attachment | 0.012288 ms | 2.823 MiB | 1 | pass |
| 20,000 | Central active-edge repair | 1.053696 ms | 2.823 MiB | 63 | pass |
| 100,000 | Developmental leaf attachment | 0.012288 ms | 14.114 MiB | 1 | pass |
| 100,000 | Central active-edge repair | 1.329152 ms | 14.115 MiB | 297 | pass |
| 200,000 | Developmental leaf attachment | 0.011264 ms | 28.229 MiB | 1 | pass |
| 200,000 | Central active-edge repair | 1.059840 ms | 28.229 MiB | 590 | pass |

Every repair reached a committed generation with zero invalid jobs. The invalidated edge remains masked while a long repair resumes across frames.

## Definition of done

The approved signed semantics, cached propagation, redundant lowest-resistance routing, bounded repair, consumers, saves/UI, economics, visualization, CPU/GPU parity, performance instrumentation, and removal requirements are satisfied. Legacy unsigned/hop genomes are explicitly rejected under the owner's clean-rebuild decision rather than migrated.
