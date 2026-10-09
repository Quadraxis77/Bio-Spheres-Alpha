// Remove pathological overlap cores, not ordinary dense/bonded colonies.
// Use the completed spatial-grid snapshot; only write our own death flag.
// The following lifecycle death scan owns counts, effects, and slot recycling.
struct PhysicsParams {
    delta_time: f32, current_time: f32, current_frame: i32, cell_count: u32,
    world_size: f32, boundary_stiffness: f32, gravity: f32, acceleration_damping: f32,
    grid_resolution: i32, grid_cell_size: f32, max_cells_per_grid: i32, enable_thrust_force: i32,
    cell_capacity: u32, _pad0: f32, _pad1: f32, _pad2: f32,
}
@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> cell_count_buffer: array<u32>;
@group(1) @binding(0) var<storage, read_write> death_flags: array<u32>;
// Reuse the read-only position-update spatial layout, avoiding aliases of
// death_flags in the general-purpose spatial bind group.
@group(2) @binding(0) var<storage, read> spatial_grid_counts: array<u32>;
@group(2) @binding(2) var<storage, read> cell_grid_indices: array<u32>;
@group(2) @binding(3) var<storage, read> spatial_grid_cells: array<u32>;

const MAX_CELLS_PER_GRID: u32 = 16u;
const EXTREME_CORE_NEIGHBORS: u32 = 16u;
const PRESERVE_CORE_CELLS: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.x;
    if (cell >= cell_count_buffer[0]) { return; }
    if (death_flags[cell] != 0u || positions_in[cell].w < 0.5 || cell == cell_count_buffer[2]) { return; }
    let res = params.grid_resolution;
    let grid = i32(cell_grid_indices[cell]);
    let coords = vec3<i32>(grid % res, (grid / res) % res, grid / (res * res));
    let pos = positions_in[cell].xyz;
    let radius = clamp(positions_in[cell].w, 0.5, 2.0);
    var overlaps = 0u;
    var earlier_cells = 0u;
    // Count actual deep contacts in a bounded neighborhood. Never extrapolate
    // from raw bucket occupancy: a large bucket can contain a healthy colony.
    for (var n = 0u; n < 27u; n++) {
        let offset = vec3<i32>(i32(n % 3u) - 1, i32((n / 3u) % 3u) - 1, i32(n / 9u) - 1);
        let p = coords + offset;
        if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(res))) { continue; }
        let bucket = u32(p.x + p.y * res + p.z * res * res);
        let count = min(spatial_grid_counts[bucket], MAX_CELLS_PER_GRID);
        for (var slot = 0u; slot < count; slot++) {
            let other = spatial_grid_cells[bucket * MAX_CELLS_PER_GRID + slot];
            if (other == cell || other >= cell_count_buffer[0]) { continue; }
            let neighbor = positions_in[other];
            if (neighbor.w < 0.5) { continue; }
            let core_distance = 0.5 * (radius + clamp(neighbor.w, 0.5, 2.0));
            let delta = pos - neighbor.xyz;
            if (dot(delta, delta) < core_distance * core_distance) {
                overlaps++;
                if (other < cell) { earlier_cells++; }
                // A stable ordering preserves a core instead of every thread
                // deciding to kill its cell in the same dense snapshot.
                if (overlaps >= EXTREME_CORE_NEIGHBORS && earlier_cells >= PRESERVE_CORE_CELLS) {
                    death_flags[cell] = 1u;
                    return;
                }
            }
        }
    }
}
