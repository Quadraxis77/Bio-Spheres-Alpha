# Phase 0 Baseline — 2026-08-12

Status: reproducible isolated baseline established; live-scene timing remains a
recorded limitation and is not substituted with standalone timing.

## Repository state

- Dirty working tree audited before production edits.
- Rejected route-cache work, useful pre-existing optimizations, unrelated user
  changes, and isolated Phase 0/1 files are separated in `change-map.md`.
- No files were reset, discarded, committed, or overwritten.

## Hardware and toolchain

- NVIDIA GeForce GTX 1080 (`0x10de:0x1b80`), Vulkan, driver 582.66.
- Rust 1.93.0 (`254b59607`, 2026-01-19), x86_64-pc-windows-msvc.
- wgpu 27.0.1; repository release profile with LTO.
- Maximum workgroup storage: 49,152 bytes.

## Preserved behavioral baseline

- Initial full library run: 142/143 tests passed.
- Current full library run after isolated scaffolding: 163/164 tests passed.
- The sole failure in both runs is the unrelated pre-existing
  `fragment_culling_never_opens_world_containment_shell` test.
- The isolated signed tree oracle currently passes 21/21 focused tests.

## Reproducible scenes

- `matrix.ron` fixes cell counts, shapes, workloads, block sizes, and gates.
- `fixtures.ron` fixes clean-slate logical signal-genome roles and seed 77.
- All synthetic topology and source generation is deterministic in
  `src/simulation/signal_backbone_bench.rs`.
- `run_matrix.ps1` builds the release/LTO executable once and runs either a
  smoke or full matrix, optionally retaining one raw output file per case.

Runner validation used one warmup and three samples per case. All 54 smoke
combinations completed in 48.9 seconds: 54/54 created their actual WGSL
pipelines, 54/54 had zero parity mismatches, and 54/54 reported passing memory,
correctness, and discrete-GPU p95 gates. The 91,518 bytes of raw output are
retained under `results/raw-smoke-validation`. These reduced-sample timings are
runner validation only, not acceptance measurements.

Example:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\signal_backbone\run_matrix.ps1 -Preset smoke -OutputDirectory .\benchmarks\signal_backbone\results\raw-smoke
```

## Measured algorithm baseline

The standalone chain report preserves the rejected unblocked/global and serial
blocked candidates alongside the accepted parallel blocked candidate. These
numbers are algorithm baselines, not live-game timings.

| Candidate | 200k p95 | Workspace | Dispatches |
|---|---:|---:|---:|
| Four full-chain global scans | 6.8343 ms | 40.436 MiB | 80 |
| Blocked 128, serial local solve | 3.6678 ms | 25.296 MiB | 52 |
| Blocked 128, parallel local scan | 0.9058 ms | 25.296 MiB | 52 |

## Live baseline limitation

The dirty tree already contains useful timestamp boundaries that isolate
legacy “Signal Processing,” but no automated live-scene runner can load a
specific snapshot, set exact active-cell counts, warm up, sample, and exit.
Inventing standalone numbers as live measurements would make the baseline
misleading. Live A/B timing must therefore preserve the legacy implementation
through Phase 3 and use the same in-game scene and timestamp facility once the
feature-flagged value path exists.
