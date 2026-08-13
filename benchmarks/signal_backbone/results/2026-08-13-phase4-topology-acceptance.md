# Phase 4 topology acceptance — 2026-08-13

## Hardware and validation environment

- Adapter: NVIDIA GeForce GTX 1080
- Backend: Vulkan
- Device type: discrete GPU
- Driver: NVIDIA 582.66
- Build: Rust `release`, `signal-backbone-bench` enabled
- GPU validation: actual wgpu device, shader module, bind-group, compute-pipeline, dispatch, and result readback in tests/benchmarks
- Topology work budget: 1,024 node/bond operations per rendered-frame dispatch
- Queue submissions in gameplay design: no additional submission; topology and signal passes share the frame encoder
- Production readbacks: none; benchmark/test readbacks are validation-only

## Correctness results

- Full release library suite: **190 passed, 0 failed**.
- CPU/GPU active-forest parity covers pending additions, strict shortcut exchange, exact ties, immediate invalidation, standby failover, and persistent bounded repair.
- Deterministic churn stress repeats 64 active-edge invalidations over a 512-node graph and verifies identical results and an acyclic active forest.
- Parent-funded division construction, transactional failure, creator attribution, and zero continuous bond maintenance are covered.
- All modified Phase 4 WGSL passes Naga validation. Actual topology and signal pipelines create and dispatch successfully on the adapter.

## GPU topology timing

All numbers are milliseconds. Each row uses one topology dispatch per rendered frame.

| Cells | Workload | min | median | p95 | Allocation | Completion frames | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 20,000 | Developmental leaf attachment | 0.010240 | 0.010240 | 0.011264 | 2.823 MiB | 1 | pass |
| 20,000 | Central active-edge repair | 1.038336 | 1.045504 | 1.220608 | 2.823 MiB | 63 | pass |
| 100,000 | Developmental leaf attachment | 0.011264 | 0.012288 | 0.012288 | 14.114 MiB | 1 | pass |
| 100,000 | Central active-edge repair | 1.143808 | 1.149952 | 1.469440 | 14.115 MiB | 297 | pass |
| 200,000 | Developmental leaf attachment | 0.010240 | 0.011264 | 0.011264 | 28.229 MiB | 1 | pass |
| 200,000 | Central active-edge repair | 1.048576 | 1.054720 | 1.274880 | 28.229 MiB | 590 | pass |

Every completion run reached a committed generation with no invalid jobs. The deliberately adversarial 200k chain repair takes multiple rendered frames, while its invalid edge remains masked throughout. This trades repair freshness for bounded frame time as required by Section 28.

## Signal-value regression gate

The integrated 200k Phase 3 gate passed in the final full suite. A prior loaded-suite sample measured 2.1228 ms p95 and failed the 2.0 ms gate; the required isolated rerun measured:

- Source evaluation p95: 0.3697 ms
- Propagation p95: 1.4756 ms
- Publication p95: 0.0041 ms
- Total p95: **1.8452 ms**
- Signal allocation: 59.510 MiB
- Dispatch count: 20

## Gate decision

**Phase 4 gate satisfied.** No stale transmission through invalid edges, cyclic amplification, CPU/GPU route disagreement, nondeterministic selection, or unbounded central-repair frame spike was observed. Repair backlog age remains an intentional diagnostic: a maximally deep 200k chain can take 590 rendered frames at the conservative 1,024-operation budget.
