// Boulder Consume Shader
//
// Per-boulder: queries the spatial grid for nearby phagocyte cells and applies
// size-gated moss consumption. Larger organisms consume faster (Michaelis-Menten).
// Accumulated eat direction is written to boulder_eat_dir for moss_dir update.
//
// NOTE — architectural debt:
// This shader queries the spatial grid that was built by the cell pipeline, which
// is correct. However, because boulders are not in the cell slot system, cells
// cannot consume moss from boulders they are standing on — only phagocytes within
// the boulder's radius are detected. If boulders had been placed in cell slots,
// the existing moss_consume.wgsl could have handled this with a cell type branch,
// and the spatial grid query would have been unnecessary (cells already know their
// neighbors via adhesion indices).
//
// Dispatched per-boulder (workgroup 64), in a separate compute pass after the
// main physics pipeline to avoid buffer aliasing with cell_count_buffer.
//
// Bind groups:
//   Group 0: boulder consume params (minimal uniform — delta_time, world_size,
//            grid params only; avoids binding cell_count_buffer which is
//            STORAGE_READ_WRITE in the main physics group)
//   Group 1: spatial grid (read-only — grid already built by cell pipeline)
//   Group 2: cell data (positions, cell_types, nutrients, organism_size,
//            death_flags, split_thresholds)
//   Group 3: boulder buffers (state, moss, eat_dir_accum, count)

// Boulder consume only needs a small subset of physics params.
// Using a dedicated uniform avoids binding cell_count_buffer (which is
// STORAGE_READ_WRITE in the main physics group) in this pass.
struct BoulderConsumeParams {
    delta_time:      f32,
    world_size:      f32,
    grid_cell_size:  f32,
    grid_resolution: i32,
}

// ── Group 0: Boulder consume params (minimal uniform, no storage buffers) ────
@group(0) @binding(0) var<uniform> params: BoulderConsumeParams;

struct GpuBoulder {
    position:         vec3<f32>,
    radius:           f32,
    velocity:         vec3<f32>,
    dead:             u32,
    seed:             u32,
    _pad:             array<u32, 3>,
    angular_velocity: vec4<f32>,
    orientation:      vec4<f32>,
}

// ── Group 1: Spatial grid (read-only) ────────────────────────────────────────
@group(1) @binding(0) var<storage, read> spatial_grid_counts:  array<u32>;
@group(1) @binding(1) var<storage, read> spatial_grid_offsets: array<u32>;
@group(1) @binding(2) var<storage, read> spatial_grid_cells:   array<u32>;

// ── Group 2: Cell data ────────────────────────────────────────────────────────
@group(2) @binding(0) var<storage, read>       cell_positions:           array<vec4<f32>>;
@group(2) @binding(1) var<storage, read>       cell_types:               array<u32>;
@group(2) @binding(2) var<storage, read_write> nutrients_buffer:         array<atomic<i32>>;
@group(2) @binding(3) var<storage, read>       organism_size_buffer:     array<u32>;
@group(2) @binding(4) var<storage, read>       death_flags:              array<u32>;
@group(2) @binding(5) var<storage, read>       split_nutrient_thresholds: array<f32>;

// ── Group 3: Boulder buffers ──────────────────────────────────────────────────
@group(3) @binding(0) var<storage, read>       boulder_state:    array<GpuBoulder>;
@group(3) @binding(1) var<storage, read_write> boulder_moss:     array<atomic<i32>>;
@group(3) @binding(2) var<storage, read_write> boulder_eat_dir:  array<atomic<i32>>; // 3 per boulder
@group(3) @binding(3) var<storage, read>       boulder_count:    array<u32>;

// ── Constants ─────────────────────────────────────────────────────────────────
const PHAGOCYTE_TYPE: u32 = 2u;
const FIXED_POINT_SCALE: f32 = 1000.0;

// Michaelis-Menten size gate:
//   rate = MAX_CONSUME_RATE * org_size / (org_size + SIZE_GATE)
// Solo cell (size 1): 1/21 ≈ 5% of max rate
// 10-cell organism:   10/30 ≈ 33%
// 100-cell organism:  100/120 ≈ 83%
const MAX_CONSUME_RATE: f32 = 400.0;  // nutrients/sec at infinite size
const SIZE_GATE: f32 = 20.0;          // half-saturation constant (cells)

fn float_to_fixed(v: f32) -> i32 { return i32(v * FIXED_POINT_SCALE); }
fn fixed_to_float(v: i32) -> f32 { return f32(v) / FIXED_POINT_SCALE; }

fn world_to_grid(pos: vec3<f32>) -> vec3<i32> {
    let half = params.world_size * 0.5;
    let cell_size = params.grid_cell_size;
    let res = params.grid_resolution;
    let gx = i32((pos.x + half) / cell_size);
    let gy = i32((pos.y + half) / cell_size);
    let gz = i32((pos.z + half) / cell_size);
    return clamp(vec3<i32>(gx, gy, gz), vec3<i32>(0), vec3<i32>(res - 1));
}

fn grid_index(gx: i32, gy: i32, gz: i32) -> u32 {
    let res = params.grid_resolution;
    return u32(gx + gy * res + gz * res * res);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let boulder_idx = gid.x;
    let boulder_total = boulder_count[0];
    if (boulder_idx >= boulder_total) { return; }

    let b = boulder_state[boulder_idx];
    if (b.dead != 0u) { return; }

    // Check if boulder has any moss left
    let current_moss = fixed_to_float(atomicLoad(&boulder_moss[boulder_idx]));
    if (current_moss <= 0.0) { return; }

    let cell_count = arrayLength(&cell_positions);
    let dt = params.delta_time;

    // Query spatial grid: 5×5×5 neighborhood around boulder center
    let gc = world_to_grid(b.position);
    let res = params.grid_resolution;

    for (var dx = -2i; dx <= 2i; dx++) {
        for (var dy = -2i; dy <= 2i; dy++) {
            for (var dz = -2i; dz <= 2i; dz++) {
                let nx = gc.x + dx;
                let ny = gc.y + dy;
                let nz = gc.z + dz;
                if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res) {
                    continue;
                }
                let gi = grid_index(nx, ny, nz);
                let count = spatial_grid_counts[gi];
                let offset = spatial_grid_offsets[gi];

                for (var k = 0u; k < count; k++) {
                    let cell_idx = spatial_grid_cells[offset + k];
                    if (cell_idx >= cell_count) { continue; }
                    if (death_flags[cell_idx] != 0u) { continue; }
                    if (cell_types[cell_idx] != PHAGOCYTE_TYPE) { continue; }

                    let cell_pos = cell_positions[cell_idx].xyz;
                    let diff = cell_pos - b.position;
                    let dist = length(diff);

                    // Must be in close surface contact — cell centre within 2.5 units of the boulder surface.
                    // This means the cell must be nearly touching the boulder, not just inside it.
                    let surface_dist = abs(dist - b.radius);
                    if (surface_dist > 2.5) { continue; }

                    // Nutrient cap check on receiving cell
                    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
                    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;
                    if (current_nutrients >= max_nutrients) { continue; }

                    // Size-gated consumption rate (Michaelis-Menten)
                    let org_size = f32(max(organism_size_buffer[cell_idx], 1u));
                    let rate = MAX_CONSUME_RATE * org_size / (org_size + SIZE_GATE);
                    let desired_gain = rate * dt;

                    // Clamp by cell headroom only — boulder atomic handles over-consumption.
                    // The physics pass clamps boulder_moss to 0 on death check.
                    let headroom = max(max_nutrients - current_nutrients, 0.0);
                    let actual_gain = min(desired_gain, headroom);

                    if (actual_gain <= 0.0) { continue; }

                    // Atomic subtract from boulder, add to cell
                    atomicAdd(&boulder_moss[boulder_idx], -float_to_fixed(actual_gain));
                    atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(actual_gain));

                    // Accumulate eat direction (from boulder toward eating cell)
                    if (dist > 0.001) {
                        let eat_dir = diff / dist;
                        let weight = actual_gain; // weight by amount eaten
                        atomicAdd(&boulder_eat_dir[boulder_idx * 3u + 0u], float_to_fixed(eat_dir.x * weight));
                        atomicAdd(&boulder_eat_dir[boulder_idx * 3u + 1u], float_to_fixed(eat_dir.y * weight));
                        atomicAdd(&boulder_eat_dir[boulder_idx * 3u + 2u], float_to_fixed(eat_dir.z * weight));
                    }
                }
            }
        }
    }
}
