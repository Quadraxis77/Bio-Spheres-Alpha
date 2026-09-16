# GPU readback audit and changes

Updated 2026-09-16. This records source-level changes and GPU regression checks, not a live-scene FPS benchmark.

## Recurring paths

| Path | Current behavior | Details | Implementation |
|---|---|---|---|
| Steam, water, nutrients, rain splashes, death particles | Removed | GPU computes clamped draw arguments after extraction/spawning; draw_indirect consumes them in the same submission. | [ParticleDraw](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/rendering/particle_draw.rs:4) |
| Organism-label debug sample | Removed | Deleted its staging buffer, periodic copy, mapping and polling. | src/simulation/gpu_physics/organism_labels.rs |
| Follow camera | Reduced to 32 bytes/sample | Parallel GPU reduction replaces full positions/labels transfers and CPU scan. Pending-map protection, reset-frame skipping and camera smoothing remain. | [OrganismFollow](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/simulation/gpu_physics/organism_follow.rs:12) |
| Culling statistics | Optional, at most 1 Hz | Both the GPU-to-staging copy and map are conditional. Previously the copy ran each frame even between one-second polls. In-flight maps drain after disabling. | [Culling telemetry](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/rendering/instance_builder.rs:2036) |
| Climate statistics | Optional, at most 4 Hz | Disabled for dynamic climate when telemetry is off. Static-water phase-change notification remains because it invalidates the cached water/ice mesh. EMA smoothing now uses elapsed wall time. | [Climate cadence](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/simulation/fluid_simulation/gpu_simulator.rs:1734) |
| Listener water | At most 30 Hz, disabled headless | Retains the 4-byte result needed by CPU audio and underwater post-processing. | [Listener sampling](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/simulation/fluid_simulation/gpu_simulator.rs:1854) |
| Water/rain audio | Skipped when effects are muted or headless | Active audio keeps the existing approximately 15 Hz compact bucket summaries. | [Environment audio](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/simulation/fluid_simulation/gpu_simulator.rs:2210) |
| Division audio | Skipped when effects are muted or headless | Active audio retains bounded candidate readbacks; collection occurs only on the final eligible physics step of a rendered frame. | [Division audio](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/scene/gpu_scene.rs:7860) |
| Cell counts | Only when scene count may have changed | An 8-byte asynchronous operational read remains while physics/insertion/removal run. No repeated samples in an unchanged paused scene. CPU currently uses these counts to bound work and maintain scene/UI state. | [Count scheduling](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/scene/gpu_scene.rs:8343) |
| GPU timestamp timings | Already optional | Retained under the separate GPU Frame Timing switch; required to present actual GPU timings. | [GPU timer](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/scene/gpu_timer.rs:149) |
| Spatial queries and cell inspection | Explicit/active interaction | Small asynchronous results remain for CPU UI and tool interaction. | [GPU tools](/home/quadraxis77/Desktop/Bio-Spheres-Alpha/src/simulation/gpu_physics/gpu_tool_operations.rs:426) |

The UI control is now named **GPU Telemetry** and gates optional culling/climate statistics. It does not imply that CPU camera, audio, cell-count bookkeeping, explicit inspection, or GPU timing can operate without their required results. Muting effects separately gates audio sampling.

At 20,000 occupied slots, follow sampling previously transferred 400,000 bytes (16-byte position/mass plus 4-byte label per slot). It now transfers 32 bytes regardless of population. Reduction work stays on the GPU.

## Explicit operations and dormant helpers

- World/lineage saves and explicit lineage scans: snapshot_io.rs and genome_snapshot.rs use blocking readbacks; fluid snapshot_voxels reads the fluid/nutrient grids. Automatic lineage refresh already uses cached counts and does not perform a full scan.
- Genome inspection/export: read_back_genome uses blocking property-buffer reads.
- Screenshot/GIF capture: pixel readbacks remain necessary for writing CPU image files. Bundled egui capture/test utilities also contain pixel readbacks.
- Legacy blocking debug helpers remain in triple_buffer.rs, adhesion_buffers.rs and mutation.rs. No runtime callers were found in the audit.
- Legacy surface mesh count and blocking culling-stat helpers have wrappers but no active callers found.
- Boulder dead-flag readback remains dormant: its request method has no callers and encodes no GPU copy. The frame-loop poll returns immediately.
- Organism-skin try_read_skinned_count is a no-op shim, not a GPU readback; its recurring scene call was removed.
- GPU tests and benchmarks intentionally read back results. Fusion has no production readback path.

## Validation

- `cargo test --lib`: 214 passed, 2 ignored.
- Particle regression checks zero, normal, overflow and reset counts in one submission, for both triangle and quad arguments, and constructs all five production renderers.
- Follow regression compares GPU centroids with a CPU oracle across multiple workgroups, dead slots, population growth, all triple-buffer indices and empty worlds. It checks the staging buffer is exactly 32 bytes.
- Culling regression checks disabled sampling, rate limits, pending-map protection, and draining after disabling.

The companion `gpu-readback-search.txt` is a current lexical index, including production code, explicit actions, tests and bundled dependencies. Matches are not operation counts.
