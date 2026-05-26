// Devorocyte Nutrient Consumption Shader
//
// Devorocytes steal nutrients from and kill foreign cells they touch.
// They skip cells of the same organism ID or same genome ID.
//
// Each Devorocyte thread:
//   1. Queries the spatial grid for nearby cells within consume_range
//   2. For each victim (different organism AND different genome):
//      a. Atomically subtracts nutrients from the victim
//      b. Atomically adds those nutrients to itself (capped at max)
//      c. If victim nutrients drop to 0, marks it dead via death_flags
//
// Runs as a separate compute pass before nutrient_transport so the stolen
// nutrients are visible to the transport shader in the same frame.

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

struct DevorocyteParams {
    // Per-mode consume_range and consume_rate packed as vec4 per mode.
    // [consume_range, consume_rate, 0, 0]
    // consume_range: extra contact distance beyond sum of radii (world units)
    // consume_rate:  nutrients stolen per second per victim
    dummy: u32, // placeholder — actual params come from mode_properties buffers
}

// Group 0: Standard physics bind group
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

// Group 1: Cell data needed for Devorocyte logic
@group(1) @binding(0)
var<storage, read> cell_types: array<u32>;

@group(1) @binding(1)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

@group(1) @binding(2)
var<storage, read> split_nutrient_thresholds: array<f32>;

@group(1) @binding(3)
var<storage, read_write> death_flags: array<u32>;

@group(1) @binding(4)
var<storage, read> organism_labels: array<u32>;

@group(1) @binding(5)
var<storage, read> genome_ids: array<u32>;

@group(1) @binding(6)
var<storage, read> mode_indices: array<u32>;

// mode_properties_v0: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
// mode_properties_v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
// We pack devorocyte params into a dedicated sub-buffer (v7):
// v7: [consume_range, consume_rate, 0, 0]
@group(1) @binding(7)
var<storage, read> mode_properties_v7: array<vec4<f32>>;

// Group 2: Spatial grid
@group(2) @binding(0)
var<storage, read> spatial_grid_counts: array<u32>;

@group(2) @binding(1)
var<storage, read> spatial_grid_cells: array<u32>;

@group(2) @binding(2)
var<storage, read> cell_grid_indices: array<u32>;

// ---- Constants ----

const DEVOROCYTE_TYPE: u32 = 11u;
const FIXED_POINT_SCALE: f32 = 1000.0;
const MAX_CELLS_PER_GRID: u32 = 16u;
const DEATH_NUTRIENT_THRESHOLD: f32 = 1.0;

fn fixed_to_float(v: i32) -> f32 {
    return f32(v) / FIXED_POINT_SCALE;
}

fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

fn grid_coords_to_index(x: i32, y: i32, z: i32, res: i32) -> u32 {
    return u32(x + y * res + z * res * res);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Only Devorocytes run this shader
    if (cell_types[cell_idx] != DEVOROCYTE_TYPE) {
        return;
    }

    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }

    let pos = positions_in[cell_idx].xyz;
    let mass = positions_in[cell_idx].w;
    if (mass < 0.5) {
        return;
    }

    let my_radius = calculate_radius_from_mass(mass);
    let my_org = organism_labels[cell_idx];
    let my_genome = genome_ids[cell_idx];
    let my_mode = mode_indices[cell_idx];

    // Read per-mode devorocyte params
    let v7 = mode_properties_v7[my_mode];
    let consume_range = v7.x; // extra contact distance beyond sum of radii
    let consume_rate  = v7.y; // nutrients/sec stolen per victim

    // Current nutrients and cap.
    // Cap at 200 before doubling so the "never split" sentinel (threshold > 100)
    // doesn't inflate the cap to an absurd value.
    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;

    // Don't steal if already full
    if (current_nutrients >= max_nutrients) {
        return;
    }

    // Amount to steal per victim this frame (capped so we don't overshoot max)
    let steal_per_victim = consume_rate * params.delta_time;
    if (steal_per_victim <= 0.0) {
        return;
    }

    // Query spatial grid — 27-cell neighbourhood
    let my_grid_idx = cell_grid_indices[cell_idx];
    let res = params.grid_resolution;
    let gz_base = i32(my_grid_idx) / (res * res);
    let gy_base = (i32(my_grid_idx) - gz_base * res * res) / res;
    let gx_base = i32(my_grid_idx) - gz_base * res * res - gy_base * res;

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nx = gx_base + dx;
                let ny = gy_base + dy;
                let nz = gz_base + dz;

                if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res) {
                    continue;
                }

                let grid_idx = grid_coords_to_index(nx, ny, nz, res);
                let count = min(spatial_grid_counts[grid_idx], MAX_CELLS_PER_GRID);
                let base = grid_idx * MAX_CELLS_PER_GRID;

                for (var i = 0u; i < count; i++) {
                    let other_idx = spatial_grid_cells[base + i];

                    if (other_idx == cell_idx) {
                        continue;
                    }

                    // Skip dead victims
                    if (death_flags[other_idx] == 1u) {
                        continue;
                    }

                    let other_mass = positions_in[other_idx].w;
                    if (other_mass < 0.5) {
                        continue;
                    }

                    // Skip same organism or same genome — Devorocytes only attack foreigners
                    let other_org    = organism_labels[other_idx];
                    let other_genome = genome_ids[other_idx];

                    let same_org    = (my_org    != 0xFFFFFFFFu && other_org    != 0xFFFFFFFFu && my_org    == other_org);
                    let same_genome = (my_genome != 0xFFFFFFFFu && other_genome != 0xFFFFFFFFu && my_genome == other_genome);

                    if (same_org || same_genome) {
                        continue;
                    }

                    // Distance check: contact = sum of radii + consume_range
                    let other_radius = calculate_radius_from_mass(other_mass);
                    let delta = pos - positions_in[other_idx].xyz;
                    let dist  = length(delta);
                    let contact_dist = my_radius + other_radius + consume_range;

                    if (dist >= contact_dist) {
                        continue;
                    }

                    // --- Steal nutrients ---
                    // Read victim's current nutrients to determine how much we can take
                    let victim_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[other_idx]));

                    // Can only steal what the victim has above the death threshold
                    let available = max(victim_nutrients - DEATH_NUTRIENT_THRESHOLD, 0.0);
                    if (available <= 0.0) {
                        continue;
                    }

                    // Clamp steal to: available, steal_per_victim, and our own headroom
                    let my_headroom = max(max_nutrients - fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx])), 0.0);
                    let actual_steal = min(min(steal_per_victim, available), my_headroom);

                    if (actual_steal <= 0.0) {
                        continue;
                    }

                    // Atomically subtract from victim and add to self
                    atomicAdd(&nutrients_buffer[other_idx], -float_to_fixed(actual_steal));
                    atomicAdd(&nutrients_buffer[cell_idx],   float_to_fixed(actual_steal));

                    // If victim is now below death threshold, mark it dead.
                    // We use a conservative check: if the value we read was already
                    // at or below threshold after our subtraction, flag it.
                    // The lifecycle shader will confirm and clean up next frame.
                    let victim_after = victim_nutrients - actual_steal;
                    if (victim_after < DEATH_NUTRIENT_THRESHOLD) {
                        death_flags[other_idx] = 1u;
                    }
                }
            }
        }
    }
}
