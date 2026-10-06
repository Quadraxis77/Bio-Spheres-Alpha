# Additive signal diffusion

Status: implemented in GPU gameplay and CPU preview, September 15, 2026.
Supersedes selected-tree transport in [the historical backbone design](signal-backbone-system-design.md).

## Confirmed integration gaps in the previous implementation

The desktop scene advanced signal ticks only when the cached pipeline had an
installed static forest. No application caller installed that forest. The
pipeline's source/config upload method also had no callers. CPU preview did run
a complete cached-tree solve, replacing its field every tick.

The scene now owns `SignalDiffusionPipeline`. GPU-generated indirect dispatch
sizes use the live GPU cell-slot count, independent of CPU count readback and
forest installation. The historical solver remains available for its existing
benchmarks and tests, but neither gameplay nor preview uses it.

## Accepted decisions and initial settings

| Area | Implementation |
|---|---|
| Channels | Nonnegative floating-point concentrations; all 16 channels diffuse. |
| Volumes | Equal unit transport volumes, independent of physical cell radius. |
| Self-reception | Included, including accumulated earlier local production. |
| Clock | Existing 15 Hz simulation clock, at most four catch-up ticks. |
| Conductance | Uniform 0.5 per second on every eligible connection. |
| Production | Authored amplitudes specify quantity/second; default scale 1.0. |
| Degradation | First-order rate 0.25 per second; half-life about 2.77 seconds. |
| Receivers | Input cap of 1000; transport storage is never saturated. |
| Memory | Explicit memorocyte leaky integrator; processor output becomes production on the next tick. |

`DiffusionSettings` exposes conductance, decay rate, and production scale in both
`GpuScene::signal_diffusion.settings` and `CanonicalState::signal_diffusion`.
These starting values are tunable, not a claim of final gameplay balance.

Negative legacy source requests and processor results produce zero quantity;
negative transport does not subtract from neighboring cells. Receiver inversion
provides inhibition. Legacy negative-polarity listeners still have their existing
meaning and therefore will not detect a nonnegative field. Genomes depending on
negative channels need explicit receiver edits. Source editors now allow 0–1000.

## Equations and timing

For concentration C, production rate P, timestep dt, and symmetric conductance g:

1. Evaluate and fund explicit source requests. Only configured sources and
   processors produce signal. Ordinary receipt is not a production request.
2. Gather transport from immutable current concentrations:

   `T_i = C_i + dt * sum_j(g * (C_j - C_i))`

3. Degrade transported signal: `D_i = exp(-decay_rate * dt) * T_i`.
4. Add production: `next_i = D_i + dt * production_scale * P_i`.
5. Publish receiver values and evaluate processor memory/output for the next tick.

Transport, degradation, and production share one invocation with separate
terms. New production appears locally at the end of its tick and first crosses
an edge on the following tick. Propagation has no hop cutoff and does not solve
the organism to equilibrium.

The GPU evaluates the transport expression in convex form:

`T_i = (1 - dt*g*degree_i)*C_i + dt*g*sum_j(C_j)`.

Validation requires an export weight at most 0.9. At the supported maximum of
20 connections, the default export weight is 2/3. Invalid settings are rejected;
there is no clamp of negative transport results. Every edge uses the same
conductance at both endpoints, without independent endpoint normalization.

Without production or degradation on a fixed live graph, total quantity is
conserved up to floating-point error. With uniform degradation and production:

`Q_next = exp(-decay_rate * dt) * Q + dt * production_scale * sum_i(P_i)`.

## Implementation

- Two concentration buffers hold 16 f32 channels per cell: 128 bytes per cell
  together. Production adds 64 bytes and lifecycle/processor state adds 32 bytes
  per cell. The existing packed receiver buffer is reused.
- Each invocation scans the cell's adhesion list once, gathering all channels.
  It writes only its own next concentration. No neighbor atomics, source-route
  histories, source-specific fields, or forest repair are used.
- All active ordinary connections participate, including cycles, parallel bonds,
  redundant links, and eligible inter-organism links. Barrier/environment joints
  remain excluded. Old selected-route bits do not control transport.
- Adhesion bookkeeping must list each physical bond once at each endpoint.
  Parallel physical bonds are separate connections and are accounted for twice,
  symmetrically. Arbitrary malformed/asymmetric adjacency is not repaired by the
  diffusion kernel.
- Link additions and breaks take effect on the next signal tick. They do not
  reset concentrations. Each bond core has two halves colored by the attached
  zones: A green, B blue, C red. Eligible bonds, including redundant links, get a
  yellow outline when both endpoints have a nonzero published receiver value on
  the same channel; otherwise the outline is black.
  This visualizes signal presence, rather than signed net transfer.
- Live GPU production reads regulation settings, oculocyte sensing, photocyte
  light thresholds, lipocyte storage thresholds, and prior processor output.
  Heat dysregulation explicitly produces on all channels, once per tick.
- Authored source/processor settings have an inherited mode-table reference in
  `signal_settings_v4.z`; the existing GPU mutation mode-copy path carries it.
  Regulation and oculocyte sensing/channel settings use the live mutable buffers.
- Cell identities are used only for slot lifecycle invalidation, never for
  transport or source separation. New/reused slots start empty. Division creates
  empty child compartments and resets processor memory in both paths; parent
  signal is discarded, not duplicated. Cell death also removes its quantity.
  Conservation tests therefore apply to fixed live compartments, not division.
- GPU public receiver values retain the existing integer packing, rounded and
  capped at 1000. Processors consume full-precision capped input. Publication
  never writes back into transport storage. CPU preview uses the same receiver
  rounding while retaining full-precision processor inputs.
- GPU sensor rays sample environmental voxel fields and the existing spatial
  grid. Preview retains its available cell/wall/self sensing and supplied light
  samples; it has no fluid/food field. These environmental differences predate
  diffusion. Sensor-grid overflow and sub-voxel ray sampling remain limitations.

## Validation

The CPU transport reference uses balanced edge-scatter transfers; the GPU uses
per-cell gather. Tests cover conservation, a maximum-degree hub, cycles, parallel
bonds, isolated-source settled gradients, overlapping-source superposition,
shutdown decay, finite propagation time, reconnecting retained fields, receiver
saturation without transport loss, and rejection of unstable settings.

GPU readback tests compare against the CPU reference and exercise live source
bindings, self-reception, repeated additive production, and slot reuse. A live
GPU/preview fixture covers regulation, oculocytes, photocytes, lipocytes,
cognocytes, and memorocytes on a cyclic graph.

Validation commands:

```sh
cargo test --lib -- --test-threads=1
cargo test --lib benchmark_diffusion_sparse_and_population -- --ignored --nocapture
cargo test --lib benchmark_gameplay_diffusion_sparse_and_population -- --ignored --nocapture
```

The full library suite passed: 203 tests, with the two population benchmarks run
separately. `cargo check --all-targets` also passed. The separate
`luca_pelagic_signal_foundation` integration suite passed its two GPU tests, but
two genome tests could not run successfully because the checkout lacks
`genomes/luca_pelagic_signal_foundation.genome`. These are headless GPU tests,
not a visual desktop gameplay session.

## Benchmark results

AMD Radeon 610M, RADV Vulkan, Mesa 25.2.8; development build. Each scenario has
100,000 cells in chains of 100 cells, with three measured batches of 30 ticks
after warm-up. Sparse activity means one producer per 1,000 cells. No work is
skipped merely because a source is silent; residual fields continue evolving.

| Measurement | Sparse producers | Every cell producing |
|---|---:|---:|
| Transport + degradation + production, 16-channel source fields | 5.45–5.66 ms/tick | 5.80–5.86 ms/tick |
| Complete tick with live one-channel regulation sources | 7.42–7.64 ms/tick | 7.48–7.63 ms/tick |

These are wall-clock submission-to-completion measurements including CPU command
encoding and binding creation, not isolated GPU timestamp measurements. Complete
ticks include indirect dispatch setup, source evaluation, transport, and receiver
publication/processing. They exclude physics, rendering, and expensive sensor-ray
activity. Production density barely changes the complete-tick cost on this fixture;
topology, population, and sensor workloads still matter.

Gathering all channels during one adjacency scan reduced measured transport cost
from roughly 20–24 ms/tick in the initial four-scan kernel to the results above.
`DIFFUSION_BENCH_CELLS` changes the benchmark population size.
