# Phase 1 Acceptance - 2026-08-13

Status: PASS. The standalone cached-forest propagation candidate meets the
Phase 1 timing, memory, depth, numerical-stability, and correctness gates on
the measured discrete GPU. No preview or live-game signal path was changed.

## Measured system

- GPU: NVIDIA GeForce GTX 1080 (`0x10de:0x1b80`), discrete
- Backend: Vulkan
- Driver: NVIDIA 582.66
- Maximum workgroup storage: 49,152 bytes
- Maximum storage-buffer binding: 2,147,483,647 bytes
- Rust/Cargo: 1.93.0; rustc commit `254b59607`, LLVM 21.1.8
- wgpu: 27.0.1
- Build: repository release profile with LTO
- Timing: GPU timestamp queries; 50 samples after 5 warmups per matrix row
- Matrix: 3 cell counts x 6 topology shapes x 11 value workloads = 198 rows
- Selected block size: 64, after the recorded 64/96/128 comparison

Raw evidence is in `raw-phase1-general-final/`. Every row was produced by the
release binary with `--strategy general`; every row created both the topology
validation pipeline and the propagation pipelines on the actual device.

## Gate results

| Gate | Limit | Measured result | Status |
|---|---:|---:|---|
| 200k discrete-GPU propagation p95 | <= 2.0 ms | 1.7749 ms | Pass |
| Total candidate memory at 200k | <= 64 MiB | 44.652 MiB | Pass |
| Transport maximum absolute error | <= 0.05 | 0.035278 | Pass |
| Processor output maximum absolute error | <= 0.05 used by harness | 0.000427 | Pass |
| Transport mismatches above tolerance | 0 | 0 across 198 rows | Pass |
| Processor mismatches above tolerance | 0 | 0 across 36 rows | Pass |
| Topology validation pipeline | Every row | 198/198 | Pass |
| WGSL propagation pipeline creation | Every row | 198/198 | Pass |
| Bounded dispatches | No convergence/retry loop | Maximum 56/tick | Pass |
| Deep-chain local influence | Must not underflow globally | 950 at deep local edge | Pass |

The reported total memory is propagation workspace plus immutable topology
metadata. The harness prints those components separately. Its standalone result
readback allocation is benchmark instrumentation and is not included in the
candidate workspace.

## Worst 200k row by shape

| Shape | Worst workload by p95 | p50 | p95 | Worst sample | Total MiB | Dispatches |
|---|---|---:|---:|---:|---:|---:|
| Balanced binary | Signed cancellation fan-in | 1.3743 | 1.7749 | 1.8014 | 44.652 | 37 |
| Chain | Heat scream, 16 channels | 0.7076 | 0.9616 | 1.1264 | 35.465 | 56 |
| Dense mechanical/sparse backbone | Saturated fan-in | 0.6651 | 0.7982 | 0.8624 | 35.465 | 56 |
| Gameplay mixed | Heat scream, 16 channels | 0.5675 | 0.7282 | 0.7628 | 35.519 | 4 |
| Many two-cell trees | All channels sparse | 0.6630 | 0.8277 | 0.8723 | 38.767 | 56 |
| Star | All channels sparse | 0.4325 | 0.5876 | 0.6303 | 42.960 | 16 |

The highest sample in the complete matrix was 1.9085 ms (200k balanced,
all-channels-sparse). The selected balanced implementation dispatches four
independent vec4 groups in Z and uses group-major scratch, reducing that path
from 148 to 37 dispatches while preserving each group's accumulation order.

## Correctness coverage

The focused CPU suite passes 24/24 tests. It covers signed attenuation from the
first edge, no-self reception, cancellation before saturation, vascular
retention, deep trees, inter-organism joins across all channels, immutable
mechanical-only cross-links, break/add topology behavior, deterministic heat,
all Cognocyte operations, synchronous processor staging, Memorocyte integration
and decay, lifecycle resets, slot reuse, flattened topology ABI invariants,
high-degree reduction, and all six topology schedules at block sizes 64, 96,
and 128.

The GPU matrix covers silent/inverted listeners, one source, one vec4 group,
all channels, every-cell emission, Cognocytes, Memorocytes, saturation,
cancellation, continuous oscillators, and deterministic 16-channel heat. All
36 processor rows have zero mismatches; the worst processor error is 0.000427.

## Timing and traffic interpretation

The GPU timestamp is the complete candidate value tick. Synthetic sources are
prepared before the timed tick, so Phase 1 does not claim a live GPU source-
evaluation time. Processor workloads include their processor dispatch in that
same value-tick measurement. Topology upload/validation is outside it and is
reported separately; incremental topology repair is a Phase 4 deliverable.
Thus normal propagation and topology work are never blended in these numbers.

Vulkan timestamp queries do not expose exact bytes moved. At 200k/all-16, the
unavoidable source-read plus publication-write floor is 24.414 MiB per tick.
Allocation and dispatch counts are reported for every raw row; no speculative
hardware-counter figure is presented as measured traffic.

The candidate tick contains no CPU readback and uses one command encoder and
one queue submission. The harness performs a readback after the tick solely for
acceptance comparison and timing collection. Live-path zero-readback and no-
additional-submission enforcement remains a Phase 3 integration gate.

## Known risks and non-gating observations

- Raw floating-point parity passes, but packed publication is not bit-exact in
  every row. The worst row has 561 one-unit integer differences out of 3.2
  million channel values (200k balanced heat). The approved 0.05 raw tolerance
  is met; exact packed parity remains a Phase 3 consumer-boundary risk.
- The accepted numbers are adapter-specific. Other GPUs must repeat pipeline
  creation and the full matrix before selecting a block/kernel policy.
- The topology selector is data-derived. The fully generic flattened fallback
  remains available for noncanonical forests, but the accepted matrix selects
  bounded optimized kernels from topology traits.
- The unrelated Phase 0 fluid-containment failure was subsequently corrected:
  fragment cleanup now reserves the following erosion layer and consistently
  uses voxel centers. The repository-wide library suite passes 167/167. The
  isolated signal suite, release build, and all actual WGSL pipeline runs pass.

## Phase boundary

Phase 1 is complete. The next authorized work is Phase 2 authoritative CPU
semantics. Preview and live gameplay remain unchanged, the rejected eight-slot
route-cache shader remains untouched, and Phase 1 results do not authorize
Phase 3 live GPU integration.
