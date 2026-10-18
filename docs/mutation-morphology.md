# Structural mutations

The construction reference is [Morphogenesis Mechanics](../procedural-creature-generation-morphology.md#morphogenesis-mechanics),
with the compact square recipe in [MR-001](../morphology-recipes.md#mr-001--adhesion-linked-22-square).
Developmental graph loops do not by themselves produce physical lattices.
Lattices require coordinated geometry and bond inheritance.

The normal GPU mutation table includes three coordinated body-plan operators:

| Operator | Geometry | Intended structural change |
| --- | --- | --- |
| Planar lattice | Identical quarter turns on both children, around X or Y | Repeated perpendicular splits can build squares and sheets. |
| Cubic lattice | Identical 120-degree body-diagonal turns on both children | Splits cycle through three perpendicular axes, allowing cubic structures. |
| Side branch | One quarter-turn child around X or Y, one straight child | Perpendicular inheritance duplicates bonds into junctions and branching meshes. |

Each operator changes a selected mode and its immediate child modes. It keeps
their routing and cell types, so existing shared anatomical sequences remain
available. Single-mode genomes remain single-mode. Other modes retain their
existing settings.

These mutations set Split Ratio to 0.5, enable Make Adhesion and both Keep
Adhesion flags, and coordinate the separate final-split rotations and keep flags.
They reset the local parent split rotation so it does not cancel the new child
turns. Connection capacity is raised to at least 12 to accommodate intermediate
divisions. Geometry mutations preserve Max Splits, division intervals, nutrient
thresholds, and signal gates. A terminal mode stays terminal; a lattice can grow
only as far as its existing division program permits.

The three operators share the existing orientation/body-plan family with the
five independent quaternion snaps. At default weights, together they account
for approximately 9.6% of mutation selections, before mutation no-ops and
allocation limits. Radiation and the existing protection of earlier modes still
control how often a birth gets a mutation. The Subtle mutations setting excludes these
coordinated structural jumps.

These are opportunities for structural variation, not guaranteed adult shapes.
Nutrients, division timing, signal gates, cell types, attachment mechanics,
existing mode transitions, and further mutations still affect development.
Longer developmental programs can combine changed and unchanged geometry.

`mutation_morphology_tests.rs` exercises the actual GPU mutation path, including
single-mode genomes, and develops the resulting geometry using the division
executor. It checks square/cube connectivity, connected branch junctions, unit
rotations, bounded split counts, in-genome routes, final-split adhesion retention, source preservation,
and exclusion from subtle mutations. The fixture supplies nutrients and a
Phagocyte-only gene pool to isolate construction from ecology and specialization.
Population-level persistence and adaptive benefit remain unmeasured.

## Mutation fidelity and explosive growth

Radiation used to run a whole-genome growth rewrite before and after each
selected mutation. It forced a root/partner cycle, raised terminal split limits,
shortened root division intervals, cleared root signal gates and final-split
routes, and replaced Stemocyte programs. Mode transitions reset split counters,
so those injected cycles defeated finite split limits. Even color mutations
could therefore restart growth in otherwise terminal tissue.
Resetting split counters on a real mode switch is intended behavior and remains
unchanged. Authored cycles and cycles produced by an explicit graph mutation
remain valid; the fault was adding cycles during unrelated mutations.

That automatic rewrite has been removed. A mutation applies its selected
operator, preserving other developmental instructions. Cell-type initialization
also preserves division timing, nutrient thresholds, and split limits.
The changed newborn immediately refreshes its cached division and metabolic
settings from its mutated mode, retaining its birth time and split count.
Trimming refuses to remove the newborn's active mode or a referenced mode.
Stemocyte differentiation targets are remapped into the cloned genome while
their negative remain/apoptosis outcomes and delay settings are preserved.

The regression tests repeatedly mutate the same lineages and check terminal,
one-split, and unlimited modes, root gates, final-split routes, Stemocyte routing,
and immediate runtime parameter updates. Existing genomes containing growth
cycles introduced by the old behavior are not automatically reconstructed;
retest from an original genome to evaluate the corrected mutation behavior.
