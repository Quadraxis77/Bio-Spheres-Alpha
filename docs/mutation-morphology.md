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

Each operator selects from every inherited mode, including modes not currently
expressed. It coordinates that selected mode's connected program through normal
and final-split division routes. This keeps a multi-mode cycle's split geometry
consistent without changing its routing or cell types. Other disconnected
programs remain independently mutable. The mutation pipeline also maintains
the intentional initial-mode development cycle
described below. A single-mode parent receives a partner before the selected
operator runs, so coordinated geometry can include that partner.

These mutations set Split Ratio to 0.5, enable Make Adhesion and both Keep
Adhesion flags, and coordinate the separate final-split rotations and keep flags.
They reset the local parent split rotation so it does not cancel the new child
turns. Connection capacity is raised to at least 12 to accommodate intermediate
divisions. Geometry operators work alongside the standard evolved-genome
rules: split limits remain bounded per mode, and the initial development cycle
can renew growth by changing modes. Division timing, nutrient thresholds, and
signal gates are also subject to the germination rules below.

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
rotations, bounded split counts, in-genome routes, final-split adhesion retention,
source preservation, and exclusion from subtle mutations. The fixture supplies nutrients and a
Phagocyte-only gene pool to isolate construction from ecology and specialization.
Population-level persistence and adaptive benefit remain unmeasured.

## Intentional development cycles

Mutated genomes maintain a route from their initial mode through a partner and
back to the initial mode. This is an intentional part of reproduction and
regrowth. A one-mode genome gains a second mode when the mode limit permits it.
An existing non-initial child is preferred as the entry. Any existing normal
division path back to the initial mode satisfies the cycle requirement,
including long loops. Maintenance preserves intermediate modes instead of
forcing a two-mode shortcut. When a return must be added, it uses a reachable
self-reference when possible so a newly extended chain remains connected. When
a cycle is newly constructed, both ends create and retain sibling adhesions so
growth can form a connected body. Established cycle partners retain their
downstream release choices. An explicit append mutation reserves its own
additional slot after any seeded partner. It splices the new mode into either child route of a randomly selected
inherited mode, retaining the route's previous destination through the new mode.
Storage order does not determine the attachment point. Selection includes both
expressed and unexpressed modes; subsequent graph mutations can connect either
program to the rest of development.

Mode transitions reset split counters. Max Splits therefore limits divisions
within a mode; it is not a lifetime limit for a lineage following a cycle.
Evolved modes use split limits from 2 through 5 so newborns can continue their
development. Player-authored source genomes are cloned before these rules run.

The initial mode must be able to germinate as a free egg. It keeps adhesive
children, uses a 1-8 second division interval and a split mass of 1.2-2.0, and
has no minimum-adhesion requirement. Initial-mode signal gates, signal routing,
and final-split route overrides are cleared so they do not bypass germination.
Downstream modes retain those tools. Stemocytes receive self-renewing division
routes and developmental-channel differentiation targets within the clone.
All mutated modes retain the one-second division-interval floor.

These rules run before and after the selected mutation to maintain the cycle
when routing, the initial mode, or the mode count changes. The changed newborn
immediately refreshes its cached division and metabolic settings from its final
mode, retaining its birth time and usable split progress. If a mutation lowers
a previously viable split budget below the inherited count, that count restarts
at zero. This lets old default lineages enter their newly bounded development
cycle instead of immediately becoming sterile. Unmutated cells and already
exhausted finite programs retain their counts. Trimming refuses to remove the
newborn's active mode or a referenced mode.

Regression tests check cycle seeding, return paths across repeated mutations,
Stemocyte renewal, source preservation, append allocation without overlapping
mode ranges, immediate runtime settings, and the division-interval floor in
normal and subtle mutation modes. Geometry tests develop the actual mutated
cycles through the division executor. The default-lineage regression starts
from the actual ten-mode default, includes descendants with 32 inherited splits,
and checks connected multicellular growth with nutrients supplied.

## Mode recruitment and graph preservation

Unexpressed modes are part of the evolving genome. They remain eligible for
geometry, cell-type, physiology, and graph mutations. Chain extension can bring
such a mode into a currently expressed program, and cycle maintenance preserves
that new route while maintaining a return path. Geometry changes in unexpressed
programs can therefore become expressed after later routing mutations.

The GPU regressions cover selection of currently unexpressed modes, recruitment
by chain extension, preservation of loops up to the supported 128-mode limit,
and connected square/cube/branch development through multi-mode programs.
Mutation family budgets, radiation protection, and the subtle-mutation setting
are unchanged. These checks do not measure long-run ecological selection.
