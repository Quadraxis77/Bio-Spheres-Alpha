# Luminocyte light and heat

Active luminocytes now scatter their nutrient/signal-controlled glow into a shared
128³ local emission grid. Overlapping sources add; switching off a cell removes
its source on the next field rebuild, while existing warmth remains in the fluid
simulation. The glow buffer clear covers all 16 bytes per cell.

The initial radius is six grid voxels, with squared radial falloff. World-space
radius is `6 * (world diameter / 128)`. Cave voxel ray marches block transmission
through rock but allow the receiving wall/vent surface to be lit. RGB and source
strength use atomic fixed-point sums (1024 units per light unit). A source is
capped at brightness 4, with RGB channels capped at 4, keeping sums within u32
for the supported 200k-cell capacity.

The renderer merges this field into its existing local-color texture. Cells get
soft local fill independently of sun direction, and cave/vent surfaces use the
sampled emitter color. Sunlight/metabolic scalar values stay separate, so the
existing luminocyte-to-photocyte sensing pass does not award energy twice.

The climate shader reads source strength separately from sunlight. The starting
heat input is `1.5 * source strength * climate rate / thermal mass`, limited to
8°C per climate update. Water warms more slowly than air, carries temperature
with flow, and conducts heat to neighbors. Ice uses the existing temperature and
phase-debt model, so sustained collective emission melts it over time. A cluster
of eight full-brightness cells provides approximately two light units halfway
through the radius. These are initial gameplay values, not a physical energy
calibration. Local cooling, thermal inertia, and nutrient supply affect melt time.

The field is rebuilt once per rendered frame, with no CPU readback. Climate uses
the preceding field, like the existing sunlight input. Cost scales with active
emitters and the fixed local radius; the field adds 32 MiB of GPU storage. Large
population performance and in-game appearance still need playtesting.

On adapters exposing wgpu 27's `EXPERIMENTAL_RAY_QUERY`, luminocyte occlusion
uses hardware ray queries against a cave BLAS/TLAS. Device setup automatically
requests the feature and acceleration-structure limits. If the platform default
backend lacks queries, setup checks for a surface-compatible Vulkan RT adapter,
preferring discrete GPUs. Unsupported devices use voxel ray marching; device
creation retries without ray queries if the optional feature request fails.

The cave solid mask is converted into opaque triangle boxes by merging X-axis
runs. Geometry stays in grid coordinates and is rebuilt only after cave changes,
including culled fragments. No GPU-to-CPU geometry readback is needed. Ray ranges
exclude the source and receiving voxels so surfaces receive illumination without
self-shadowing. Queries terminate on the first opaque hit. Empty caves use a
degenerate triangle with no occlusion. Geometry beyond the device primitive or
buffer limit (or the two-million-triangle budget) uses the voxel fallback.

The experimental wgpu API is explicitly opted into; it is not a full path tracer
or a reflection pipeline. See the pinned [wgpu ray-query example](https://github.com/gfx-rs/wgpu/blob/v27.0.1/examples/features/src/ray_scene/shader.wgsl).
Hardware execution was tested on NVIDIA GeForce RTX 5070 Ti Laptop GPU
(driver 595.91.07, Vulkan), and AMD Radeon 610M (RADV/Mesa 25.2.8, Vulkan).
The regression test requests HighPerformance, matching the app, to select the
dedicated NVIDIA GPU on hybrid systems.

Validation: `cargo test --test luminocyte_emission --test tail_lighting` exercises
GPU accumulation, finite reach, wall occlusion, source removal, receiving shader
validation, fluid binding compatibility, water warming, and delayed ice melting
with an unheated control. It also validates both shader variants and exercises
hardware ray queries and cave rebuilds when the adapter supports them (otherwise
the hardware execution test reports its skip). `cargo test --lib luminocyte`
checks mesh merging, gaps, empty caves, and geometry budgets.

The **Lighting → Luminocyte Lighting → Hardware Ray Tracing** toggle switches
between hardware ray queries and voxel ray marching immediately. It is available
without Advanced options and saved with lighting settings. On unsupported active
devices it is unchecked and disabled, with a hover explanation; a saved hardware
preference is retained for supported machines. Old settings default to hardware
when available. Cave edits made while the toggle is off are rebuilt when it is
turned back on. Both modes retain illumination and water/ice heating.
