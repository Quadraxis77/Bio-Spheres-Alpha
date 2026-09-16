// Gametocyte Merge Detection Shader
//
// Detects contact between Gametocyte cells from different organisms.
// When two Gametocytes (cell type 13) from different organisms come within
// merge_range of each other, a merge event is written to the events buffer
// and both cells are marked as fused.
//
// Each merge event (8 u32s = 32 bytes):
//   [0] cell_a_idx
//   [1] cell_b_idx
//   [2] genome_a_id
//   [3] genome_b_id
//   [4] position_x as bits (f32 encoded as u32)
//   [5] position_y as bits
//   [6] position_z as bits
//   [7] combined reserve (fixed-point x1000)
//
// The events buffer has a leading u32 atomic counter (events_count) followed
// by MAX_GAMETE_MERGE_EVENTS events. GPU allocation/crossover/birth consume these
// in the same command stream. Parent locks are rolled back if allocation fails.

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Group 0: Standard physics bind group (read-only positions + cell_count)
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> velocities_out: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read> cell_count_buffer: array<u32>;

// Group 1: Cell data needed for Gametocyte logic
@group(1) @binding(0)
var<storage, read> cell_types: array<u32>;

@group(1) @binding(1)
var<storage, read_write> death_flags: array<atomic<u32>>;

// Persistent developmental identity: [organism_id, lineage_hash_lo,
// lineage_hash_hi, depth_branch]. Unlike a connected-component label,
// organism_id survives detachment, so gametes produced by the same organism
// cannot self-fertilize after they become separate free cells.
@group(1) @binding(2)
var<storage, read> development_addresses: array<vec4<u32>>;

@group(1) @binding(3)
var<storage, read> genome_ids: array<u32>;

@group(1) @binding(4)
var<storage, read> mode_indices: array<u32>;

// mode_properties_v13: [merge_range, 0, 0, 0]
@group(1) @binding(5)
var<storage, read> mode_properties_v13: array<vec4<f32>>;

// Merge events buffer:
//   [0]          = atomic event counter (u32)
//   [1..1+N*8]   = N events, each 8 u32s
@group(1) @binding(6)
var<storage, read_write> merge_events: array<atomic<u32>>;

// Embryocyte/Gametocyte reserve buffer (x1000 fixed-point, read-only here)
@group(1) @binding(7)
var<storage, read> embryocyte_reserves: array<u32>;

@group(1) @binding(8)
var<storage, read> genome_meta: array<vec4<u32>>;
@group(1) @binding(9)
var<storage, read> mode_cell_types: array<u32>;

fn genomes_compatible(a: u32, b: u32) -> bool {
    if (a >= arrayLength(&genome_meta) || b >= arrayLength(&genome_meta)) { return false; }
    let ma = genome_meta[a];
    let mb = genome_meta[b];
    let overlap = min(ma.x, mb.x);
    if (overlap == 0u) { return false; }
    if (a == b) { return true; }
    let alignment = f32(overlap) / f32(max(ma.x, mb.x));
    if (alignment < 0.5) { return false; }
    var matches = 0u;
    for (var i = 0u; i < overlap; i++) {
        if (mode_cell_types[ma.y + i] == mode_cell_types[mb.y + i]) { matches += 1u; }
    }
    return alignment * f32(matches) / f32(overlap) >= 0.5;
}

// Group 2: Spatial grid
@group(2) @binding(0)
var<storage, read> spatial_grid_counts: array<u32>;

@group(2) @binding(1)
var<storage, read> spatial_grid_cells: array<u32>;

@group(2) @binding(2)
var<storage, read> cell_grid_indices: array<u32>;

// ---- Constants ----
const GAMETOCYTE_TYPE: u32 = 13u;
const MAX_GAMETE_MERGE_EVENTS: u32 = 64u;
const MAX_CELLS_PER_GRID: u32 = 16u;
const EVENTS_HEADER_U32S: u32 = 1u;  // event_count atomic at index 0
const EVENT_STRIDE: u32 = 8u;         // u32s per event

fn grid_coords_to_index(x: i32, y: i32, z: i32, res: i32) -> u32 {
    let cx = clamp(x, 0, res - 1);
    let cy = clamp(y, 0, res - 1);
    let cz = clamp(z, 0, res - 1);
    return u32(cx + cy * res + cz * res * res);
}

fn world_to_grid(pos: vec3<f32>, world_size: f32, res: i32) -> vec3<i32> {
    let half = world_size * 0.5;
    let cell_size = world_size / f32(res);
    let gx = i32((pos.x + half) / cell_size);
    let gy = i32((pos.y + half) / cell_size);
    let gz = i32((pos.z + half) / cell_size);
    return vec3<i32>(gx, gy, gz);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    let live_count = cell_count_buffer[0];
    if cell_idx >= live_count {
        return;
    }

    // Only process Gametocyte cells
    if cell_types[cell_idx] != GAMETOCYTE_TYPE {
        return;
    }

    // Skip dead cells
    if atomicLoad(&death_flags[cell_idx]) != 0u {
        return;
    }

    let pos_a = positions_in[cell_idx].xyz;
    let mode_idx = mode_indices[cell_idx];
    let merge_range = mode_properties_v13[mode_idx].x;
    let reproductive_parent_a = development_addresses[cell_idx].x;
    let genome_a = genome_ids[cell_idx];
    if (reproductive_parent_a == 0u) { return; }
    let radius_a = clamp(positions_in[cell_idx].w, 0.5, 2.0);

    // Query spatial grid neighbours
    let res = params.grid_resolution;
    let gc = world_to_grid(pos_a, params.world_size, res);

    // Clip the neighborhood once. Clamping each lookup repeats edge buckets
    // up to 27 times, multiplying candidate reads and compatibility work.
    let grid_lo = max(gc - vec3<i32>(1), vec3<i32>(0));
    let grid_hi = min(gc + vec3<i32>(1), vec3<i32>(res - 1));
    for (var z = grid_lo.z; z <= grid_hi.z; z++) {
        for (var y = grid_lo.y; y <= grid_hi.y; y++) {
            for (var x = grid_lo.x; x <= grid_hi.x; x++) {
                let grid_idx = u32(x + y * res + z * res * res);
                let count = min(spatial_grid_counts[grid_idx], MAX_CELLS_PER_GRID);
                let base = grid_idx * MAX_CELLS_PER_GRID;

                for (var k: u32 = 0u; k < count; k++) {
                    let other_idx = spatial_grid_cells[base + k];

                    // Reject duplicate pairs and stale entries before storage reads.
                    if other_idx <= cell_idx || other_idx >= live_count {
                        continue;
                    }

                    // Only consider Gametocyte cells
                    if cell_types[other_idx] != GAMETOCYTE_TYPE {
                        continue;
                    }

                    // Skip dead
                    if atomicLoad(&death_flags[other_idx]) != 0u {
                        continue;
                    }

                    // The developmental organism ID remains fixed when a gamete
                    // detaches. Reject same-parent fusion before either gamete is
                    // consumed. A zero ID is invalid, so it cannot participate.
                    let reproductive_parent_b = development_addresses[other_idx].x;
                    if reproductive_parent_a == 0u
                        || reproductive_parent_b == 0u
                        || reproductive_parent_a == reproductive_parent_b {
                        continue;
                    }

                    // Check contact distance (mass stored in .w, radius = clamp(mass, 0.5, 2.0))
                    let pos_b = positions_in[other_idx].xyz;
                    let delta = pos_a - pos_b;
                    let mass_b = positions_in[other_idx].w;
                    let radius_b = clamp(mass_b, 0.5, 2.0);
                    let contact_dist = radius_a + radius_b + merge_range;
                    if dot(delta, delta) > contact_dist * contact_dist {
                        continue;
                    }

                    if (!genomes_compatible(genome_a, genome_ids[other_idx])) {
                        continue;
                    }

                    // Atomically claim both parents; competing invocations must
                    // never consume a gamete in more than one fusion.
                    if (!atomicCompareExchangeWeak(&death_flags[cell_idx], 0u, 2u).exchanged) {
                        return;
                    }
                    if (!atomicCompareExchangeWeak(&death_flags[other_idx], 0u, 2u).exchanged) {
                        atomicStore(&death_flags[cell_idx], 0u);
                        continue;
                    }
                    let slot = atomicAdd(&merge_events[0], 1u);
                    if slot >= MAX_GAMETE_MERGE_EVENTS {
                        atomicStore(&death_flags[cell_idx], 0u);
                        atomicStore(&death_flags[other_idx], 0u);
                        return;
                    }

                    // Write event at slot offset (header = 1 u32, stride = 8)
                    let event_base = EVENTS_HEADER_U32S + slot * EVENT_STRIDE;
                    atomicStore(&merge_events[event_base + 0u], cell_idx);
                    atomicStore(&merge_events[event_base + 1u], other_idx);
                    atomicStore(&merge_events[event_base + 2u], genome_a);
                    atomicStore(&merge_events[event_base + 3u], genome_ids[other_idx]);
                    // Encode spawn position as float bits
                    let mid = (pos_a + pos_b) * 0.5;
                    atomicStore(&merge_events[event_base + 4u], bitcast<u32>(mid.x));
                    atomicStore(&merge_events[event_base + 5u], bitcast<u32>(mid.y));
                    atomicStore(&merge_events[event_base + 6u], bitcast<u32>(mid.z));
                    // Combine reserves from both gametes (capped at max reserve x1000)
                    let reserve_a = embryocyte_reserves[cell_idx];
                    let reserve_b = embryocyte_reserves[other_idx];
                    let combined = min(reserve_a + reserve_b, 65535000u);
                    atomicStore(&merge_events[event_base + 7u], combined);

                    // Both parents are fused (2), not dead (1). Stop after one pair.
                    return;
                }
            }
        }
    }
}
