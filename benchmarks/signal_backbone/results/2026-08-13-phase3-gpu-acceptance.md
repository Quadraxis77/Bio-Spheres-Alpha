# Phase 3 GPU acceptance — 2026-08-13

Phase 3 passed its integrated release acceptance matrix.

## Environment

- GPU: NVIDIA GeForce GTX 1080 (discrete)
- Backend: Vulkan
- Driver: NVIDIA 582.66
- Rust: 1.93.0 (254b59607 2026-01-19)
- Profile: `release`
- Warm-up ticks per case: 5
- Timed samples per case: 20
- Percentile: standard nearest-rank p95
- Queue submissions per gameplay signal tick: 1
- CPU readbacks per gameplay signal tick: 0

## Matrix result

- Cases: 198/198 passed
- Sizes: 20,000, 100,000, and 200,000 cells
- Topologies: chain, star, balanced binary, many pairs, gameplay mixed, and dense mechanical/sparse backbone
- Workloads: all 11 approved matrix workloads, including Cognocytes, Memorocytes, signed cancellation, saturation, oscillators, and heat screaming
- Correctness failures: 0
- Timing failures: 0
- Memory failures: 0
- Maximum parity error: 0.021729 (tolerance 0.05)
- Maximum parity mismatches: 0
- Maximum signal allocation: 59.510 MiB (64 MiB gate)
- Maximum dispatch count: 54

| Cells | Cases | Maximum p50 | Maximum p95 | Maximum sample | Maximum memory |
|---:|---:|---:|---:|---:|---:|
| 20,000 | 66 | 0.2294 ms | 0.2314 ms | 0.2345 ms | 5.952 MiB |
| 100,000 | 66 | 0.8387 ms | 1.0230 ms | 1.1254 ms | 29.756 MiB |
| 200,000 | 66 | 1.6988 ms | 1.9988 ms | 2.0910 ms | 59.510 MiB |

The worst p95 case was 200,000-cell balanced-binary Cognocytes at 1.9988 ms. The worst single sample is reported for diagnosis only; the approved gate is p95.

## Correctness and pipeline validation

- The condensed signal library suite passed 40/40 tests before the final release matrix.
- The focused integrated GPU suite passed CPU-oracle parity for canonical heaps, arbitrary shallow trees, path forests, stars, many-pair forests, and gameplay microtrees.
- Processor tests cover every Cognocyte operation, one-tick latency, Memorocyte lifecycle, generation/division/mode/death resets, nutrient funding, and deterministic heat.
- WGSL was validated through actual wgpu pipeline creation and all release matrix cases executed the pipelines on the GPU.

Raw per-case outputs are retained locally in `raw-phase3-final/`. That directory is deliberately ignored so reproducible evidence does not add hundreds of working-tree entries.

## Phase boundary

Phase 3 provides the fixed-rate integrated value path and an externally installed immutable cached forest. Phase 4 remains responsible for live adhesion classification, bounded topology construction/repair, creator ownership and maintenance economics, additions-at-next-tick activation, and immediate split handling.
