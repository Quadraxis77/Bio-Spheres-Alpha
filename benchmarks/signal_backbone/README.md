# Signal Backbone Phase 0/1 Harness

This directory is isolated from live simulation. `matrix.ron` is the approved
benchmark matrix in machine-readable form; benchmark result files include its
schema version so runs remain reproducible.

Run the standalone release harness with:

```text
cargo run --release --features signal-backbone-bench --bin signal-backbone-bench -- --cells 200000
```

The harness must create the real wgpu pipeline, report adapter/backend/driver
and limits, use GPU timestamps, compare packed results with the CPU oracle, and
report workspace bytes and dispatch count. A successful process exit is not by
itself an acceptance-gate pass.

Phase 1 is accepted on the measured GTX 1080/Vulkan system. The authoritative
summary is `results/2026-08-13-phase1-acceptance.md`; its 198 raw run logs are
under `results/raw-phase1-general-final/`.

Each raw row prints `phase1_gate=INCOMPLETE_MATRIX` because a single process
cannot certify the other 197 cases. The aggregate acceptance report performs
that matrix-level decision.

Phase 2 CPU semantics are accepted in
`results/2026-08-13-phase2-cpu-acceptance.md`.

Run the deterministic matrix with:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\signal_backbone\run_matrix.ps1 -Preset smoke
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\benchmarks\signal_backbone\run_matrix.ps1 -Preset full -OutputDirectory .\benchmarks\signal_backbone\results\raw-full
```

`fixtures.ron` records the clean-slate logical signal-genome fixtures. They are
benchmark inputs, not legacy `.genome` migrations or live-game save files.
`change-map.md` is the authoritative dirty-tree ownership map for this work.
