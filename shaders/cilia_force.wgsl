// Cilia Force Shader - Applies contact-dependent surface propulsion for Ciliocyte cells
//
// Ciliocytes push against anything touching them - neighboring cells or solid surfaces -
// generating thrust along the cell's local forward axis (+Z). In open water with nothing
// to push against, the cell is completely inert.
//
// Force model:
// - Wall/boundary contact: self-propulsion along forward axis (forward x speed x mass x CILIA_WALL_MULTIPLIER)
// - Neighbor attraction: gentle radial pull on non-organism cells within attract_range (v6.w)
// - Neighbor contact: one-sided conveyor force - pushes cargo cells in +forward direction
//   (force_mag = speed x mass x CILIA_PUSH_MULTIPLIER; no reaction on self - wall thrust
//   already handles locomotion, and a reaction force would stall the conveyor against cargo)
//
// Must run after swim_force and before glueocyte_env_adhesion.
// Workgroup size: 256 threads for optimal GPU occupancy

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

struct CaveParams {
    world_center: vec3<f32>,
    world_radius: f32,
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
    grid_resolution: u32,
    triangle_count: u32,
    collision_enabled: u32,
    collision_stiffness: f32,
    collision_damping: f32,
    substeps: u32,
    _padding: f32,
    _padding2: vec4<f32>,
    _padding3: vec4<f32>,
    _padding4: vec4<f32>,
    _padding5: vec4<f32>,
    _padding6: vec4<f32>,
    _padding7: vec4<f32>,
    _padding8: vec4<f32>,
    _padding9: vec4<f32>,
    _padding10: vec4<f32>,
    _padding11: vec4<f32>,
    _padding12: vec4<f32>,
    _padding13: vec4<f32>,
    _padding14: vec4<f32>,
    _padding15: vec4<f32>,
    _padding16: vec4<f32>,
    _padding17: vec4<f32>,
    _padding18: vec4<f32>,
    _padding19: vec4<f32>,
    _padding20: vec4<f32>,
    _padding21: vec4<f32>,
    _padding22: vec4<f32>,
    _padding23: vec4<f32>,
    _padding24: vec4<f32>,
    _padding25: vec4<f32>,
    _padding26: vec4<f32>,
    _padding27: vec4<f32>,
    _padding28: vec4<f32>,
    _padding29: vec4<f32>,
    _padding30: vec4<f32>,
    _padding31: vec4<f32>,
    _padding32: vec4<f32>,
    _padding33: vec4<f32>,
    _padding34: vec4<f32>,
    _padding35: vec4<f32>,
    _padding36: vec4<f32>,
    _padding37: vec4<f32>,
    _padding38: vec4<f32>,
    _padding39: vec4<f32>,
    _padding40: vec4<f32>,
    _padding41: vec4<f32>,
    _padding42: vec4<f32>,
    _padding43: vec4<f32>,
    _padding44: vec4<f32>,
    _padding45: vec4<f32>,
    _padding46: vec4<f32>,
    _padding47: vec4<f32>,
}

// Cell type behavior flags for parameterized shader logic
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    applies_buoyancy: u32,
    applies_cilia_force: u32,
    _padding: array<u32, 8>,
}

// Group 0: Standard physics bind group
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Group 1: Force accumulation + rotations
@group(1) @binding(0)
var<storage, read_write> force_accum_x: array<atomic<i32>>;

@group(1) @binding(1)
var<storage, read_write> force_accum_y: array<atomic<i32>>;

@group(1) @binding(2)
var<storage, read_write> force_accum_z: array<atomic<i32>>;

@group(1) @binding(3)
var<storage, read> rotations: array<vec4<f32>>;

// Group 2: Cell/mode data + signals + v5/v6
@group(2) @binding(0)
var<storage, read> mode_indices: array<u32>;

@group(2) @binding(1)
var<storage, read> cell_types: array<u32>;  // DEPRECATED - use mode_cell_types instead

// mode_properties split into 5 vec4 sub-buffers (bindings 2-6)
@group(2) @binding(2)
var<storage, read> mode_properties_v0: array<vec4<f32>>;
@group(2) @binding(3)
var<storage, read> mode_properties_v1: array<vec4<f32>>;
@group(2) @binding(4)
var<storage, read> mode_properties_v2: array<vec4<f32>>;
@group(2) @binding(5)
var<storage, read> mode_properties_v3: array<vec4<f32>>;
@group(2) @binding(6)
var<storage, read> mode_properties_v4: array<vec4<f32>>;

// Mode cell types lookup table
@group(2) @binding(7)
var<storage, read> mode_cell_types: array<u32>;

// Cell type behavior flags
@group(2) @binding(8)
var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// Signal flags buffer (read-only)
@group(2) @binding(9)
var<storage, read> signal_flags: array<atomic<u32>>;

// Cilia mode properties
@group(2) @binding(10)
var<storage, read> mode_properties_v5: array<vec4<f32>>;

@group(2) @binding(11)
var<storage, read> mode_properties_v6: array<vec4<f32>>;

// Group 3: Spatial grid + organism labels
@group(3) @binding(0)
var<storage, read> spatial_grid_counts: array<u32>;

@group(3) @binding(1)
var<storage, read> spatial_grid_cells: array<u32>;

@group(3) @binding(2)
var<storage, read> cell_grid_indices: array<u32>;

@group(3) @binding(3)
var<storage, read> organism_labels: array<u32>;

// Cave params for SDF sampling (merged into group 3 to stay within max_bind_groups=4)
@group(3) @binding(4)
var<uniform> cave_params: CaveParams;

// ---- Constants ----

const CILIA_WALL_MULTIPLIER: f32 = 800.0;
const CILIA_PUSH_MULTIPLIER: f32 = 200.0;
const CILIA_ATTRACT_MULTIPLIER: f32 = 80.0; // Attraction force scale - intentionally small relative to push
const CONTACT_THRESHOLD: f32 = 3.0;
const CONTACT_THRESHOLD_NEIGHBOR: f32 = 0.5;
const MAX_CELLS_PER_GRID: u32 = 16u;
const FIXED_POINT_SCALE: f32 = 1000.0;
const CILIA_MAX_CRAWL_SPEED: f32 = 8.0; // Terminal crawl speed - high traction, low top speed
const CILIA_ATTRACT_MAX_RANGE: f32 = 10.0; // Maximum attraction radius at force=1.0 (world units)

// ---- Utility Functions ----

fn float_to_fixed(v: f32) -> i32 {
    return i32(v * FIXED_POINT_SCALE);
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Rotate a vector by a quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn grid_coords_to_index(x: i32, y: i32, z: i32, grid_resolution: i32) -> u32 {
    return u32(x + y * grid_resolution + z * grid_resolution * grid_resolution);
}

fn grid_index_to_coords(grid_idx: u32, grid_resolution: i32) -> vec3<i32> {
    let res = grid_resolution;
    let z = i32(grid_idx) / (res * res);
    let y = (i32(grid_idx) - z * res * res) / res;
    let x = i32(grid_idx) - z * res * res - y * res;
    return vec3<i32>(x, y, z);
}

// ---- Cave SDF Functions (from glueocyte_env_adhesion.wgsl) ----

fn hash1(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    var h = seed;
    h = h * 374761393u + u32(x);
    h = h * 668265263u + u32(y);
    h = h * 1274126177u + u32(z);
    h ^= h >> 13u;
    h = h * 1274126177u;
    h ^= h >> 16u;
    return f32(h) / f32(0xFFFFFFFFu);
}

fn smoothstep_custom(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let ix = i32(floor(pos.x));
    let iy = i32(floor(pos.y));
    let iz = i32(floor(pos.z));
    let fx = pos.x - floor(pos.x);
    let fy = pos.y - floor(pos.y);
    let fz = pos.z - floor(pos.z);
    let ux = smoothstep_custom(fx);
    let uy = smoothstep_custom(fy);
    let uz = smoothstep_custom(fz);
    let c000 = hash1(ix,     iy,     iz,     seed);
    let c100 = hash1(ix + 1, iy,     iz,     seed);
    let c010 = hash1(ix,     iy + 1, iz,     seed);
    let c110 = hash1(ix + 1, iy + 1, iz,     seed);
    let c001 = hash1(ix,     iy,     iz + 1, seed);
    let c101 = hash1(ix + 1, iy,     iz + 1, seed);
    let c011 = hash1(ix,     iy + 1, iz + 1, seed);
    let c111 = hash1(ix + 1, iy + 1, iz + 1, seed);
    let x00 = mix(c000, c100, ux);
    let x10 = mix(c010, c110, ux);
    let x01 = mix(c001, c101, ux);
    let x11 = mix(c011, c111, ux);
    let y0 = mix(x00, x10, uy);
    let y1 = mix(x01, x11, uy);
    return mix(y0, y1, uz);
}

fn fbm(pos: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    for (var i = 0u; i < cave_params.octaves; i = i + 1u) {
        let sample_pos = pos * frequency / cave_params.scale;
        let octave_seed = cave_params.seed + i * 1337u;
        value = value + amplitude * value_noise_3d(sample_pos, octave_seed);
        max_value = max_value + amplitude;
        amplitude = amplitude * cave_params.persistence;
        frequency = frequency * 2.0;
    }
    return value / max_value;
}

fn warp_domain(pos: vec3<f32>) -> vec3<f32> {
    let warp_scale = cave_params.scale * 0.5;
    let warp_strength = cave_params.smoothness * cave_params.scale;
    let warp_seed = cave_params.seed + 9999u;
    let wx = value_noise_3d(pos / warp_scale, warp_seed) - 0.5;
    let wy = value_noise_3d(pos / warp_scale + vec3<f32>(31.7, 47.3, 13.1), warp_seed) - 0.5;
    let wz = value_noise_3d(pos / warp_scale + vec3<f32>(73.9, 19.4, 67.2), warp_seed) - 0.5;
    return vec3<f32>(pos.x + wx * warp_strength, pos.y + wy * warp_strength, pos.z + wz * warp_strength);
}

fn sample_cave_density(pos: vec3<f32>) -> f32 {
    let dist_from_center = length(pos - cave_params.world_center);
    let sphere_sdf = dist_from_center - cave_params.world_radius;
    if (sphere_sdf > 0.0) { return 1.0; }
    let warped_pos = warp_domain(pos);
    let noise = fbm(warped_pos);
    let cave_threshold = clamp(cave_params.density, 0.0, 1.0);
    if (noise > cave_threshold) {
        let wall_factor = (noise - cave_threshold) / max(1.0 - cave_threshold, 0.001);
        return cave_params.threshold + wall_factor * 0.5;
    } else {
        return cave_params.threshold - 0.5;
    }
}

// Returns true if pos is touching a solid surface (cave wall or boundary sphere)
fn is_touching_surface(pos: vec3<f32>, contact_threshold: f32) -> bool {
    let boundary_radius = cave_params.world_radius;
    let dist_from_center = length(pos - cave_params.world_center);

    // Boundary sphere contact
    if (dist_from_center >= boundary_radius - contact_threshold) {
        return true;
    }

    // Cave wall contact (only when caves are enabled)
    if (cave_params.collision_enabled != 0u) {
        let offsets = array<vec3<f32>, 6>(
            vec3<f32>( contact_threshold, 0.0, 0.0),
            vec3<f32>(-contact_threshold, 0.0, 0.0),
            vec3<f32>(0.0,  contact_threshold, 0.0),
            vec3<f32>(0.0, -contact_threshold, 0.0),
            vec3<f32>(0.0, 0.0,  contact_threshold),
            vec3<f32>(0.0, 0.0, -contact_threshold),
        );
        for (var i = 0u; i < 6u; i++) {
            let sample_pos = pos + offsets[i];
            let density = sample_cave_density(sample_pos);
            if (density > cave_params.threshold) {
                return true;
            }
        }
    }

    return false;
}

// ---- Main Compute Shader ----

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Get mode index first, then derive cell type from mode
    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types) || mode_idx >= arrayLength(&mode_properties_v5)) {
        return;
    }

    // Derive cell type from mode (always up-to-date with genome settings)
    let cell_type = mode_cell_types[mode_idx];

    // Check if this cell type applies cilia force using behavior flags
    let behavior = type_behaviors[cell_type];
    if (behavior.applies_cilia_force == 0u) {
        return;
    }

    let pos = positions_in[cell_idx].xyz;
    let mass = positions_in[cell_idx].w;

    // Skip dead cells
    if (mass < 0.5) {
        return;
    }

    // Read cilia params from mode_properties_v5, v6
    // v5: [cilia_speed, cilia_push_bonded, cilia_use_signal, cilia_signal_channel]
    // v6: [cilia_speed_below, cilia_speed_above, cilia_threshold, cilia_attract_force]
    let v5 = mode_properties_v5[mode_idx];
    let v6 = mode_properties_v6[mode_idx];

    // Resolve effective_speed (fixed or signal-reactive)
    var effective_speed: f32;
    if (v5.z >= 0.5) { // cilia_use_signal
        // Signal-based mode: read from the specific channel
        let sig_channel = clamp(u32(v5.w), 0u, 7u);
        let raw_signal = atomicLoad(&signal_flags[cell_idx * 16u + sig_channel]);
        let signal_value = f32(raw_signal & 2047u); // Extract value component
        if (signal_value >= v6.z) { // cilia_threshold
            effective_speed = v6.y; // cilia_speed_above
        } else {
            effective_speed = v6.x; // cilia_speed_below
        }
    } else {
        // Fixed speed mode
        effective_speed = v5.x; // cilia_speed
    }

    // Skip if no effective speed (cell is inert)
    if (abs(effective_speed) < 0.001) {
        return;
    }

    // Get cell rotation and calculate forward direction
    // Forward is +Z in local space (matching preview physics)
    let rotation = rotations[cell_idx];
    let forward = quat_rotate(rotation, vec3<f32>(0.0, 0.0, 1.0));

    let radius = calculate_radius_from_mass(mass);

    // === WALL/BOUNDARY CONTACT ===
    var wall_contact = false;

    // Check if touching any surface (boundary sphere or cave wall)
    if (is_touching_surface(pos, CONTACT_THRESHOLD)) {
        wall_contact = true;
    }

    // Apply wall self-propulsion (once, if touching any wall)
    // High traction at low speeds, force tapers to zero at max crawl speed
    if (wall_contact) {
        let vel = velocities_in[cell_idx].xyz;
        let speed_along_forward = dot(vel, forward);
        // Traction factor: 1.0 at rest, 0.0 at max crawl speed (in thrust direction)
        let target_speed = effective_speed * CILIA_MAX_CRAWL_SPEED;
        let traction = clamp(1.0 - speed_along_forward / target_speed, 0.0, 1.0);
        let wall_thrust = forward * effective_speed * mass * CILIA_WALL_MULTIPLIER * traction;
        atomicAdd(&force_accum_x[cell_idx], float_to_fixed(wall_thrust.x));
        atomicAdd(&force_accum_y[cell_idx], float_to_fixed(wall_thrust.y));
        atomicAdd(&force_accum_z[cell_idx], float_to_fixed(wall_thrust.z));
    }

    // === NEIGHBOR CONTACT + PUSH ===
    // Query spatial grid for neighbors (27-cell neighborhood)
    let my_org = organism_labels[cell_idx];
    let push_bonded = v5.y >= 0.5; // cilia_push_bonded
    let my_grid_idx = cell_grid_indices[cell_idx];

    // Skip overflow cells - same reason as collision_detection.wgsl.
    // An overflow ciliocyte querying the grid would push neighbours one-sidedly.
    if (my_grid_idx == 0xFFFFFFFFu) {
        return;
    }

    let my_grid_coords = grid_index_to_coords(my_grid_idx, params.grid_resolution);

    // Pre-compute all 27 neighbor grid indices and counts
    var neighbor_indices: array<u32, 27>;
    var neighbor_counts: array<u32, 27>;
    var n_idx = 0u;

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nx = clamp(my_grid_coords.x + dx, 0, params.grid_resolution - 1);
                let ny = clamp(my_grid_coords.y + dy, 0, params.grid_resolution - 1);
                let nz = clamp(my_grid_coords.z + dz, 0, params.grid_resolution - 1);

                // Check if this is actually a valid neighbor (not clamped)
                let is_valid = (my_grid_coords.x + dx >= 0) && (my_grid_coords.x + dx < params.grid_resolution) &&
                               (my_grid_coords.y + dy >= 0) && (my_grid_coords.y + dy < params.grid_resolution) &&
                               (my_grid_coords.z + dz >= 0) && (my_grid_coords.z + dz < params.grid_resolution);

                let neighbor_grid_idx = grid_coords_to_index(nx, ny, nz, params.grid_resolution);
                neighbor_indices[n_idx] = neighbor_grid_idx;
                neighbor_counts[n_idx] = select(0u, min(spatial_grid_counts[neighbor_grid_idx], MAX_CELLS_PER_GRID), is_valid);
                n_idx++;
            }
        }
    }

    // Iterate neighbors and apply push forces
    for (var n = 0u; n < 27u; n++) {
        let cell_count_in_grid = neighbor_counts[n];
        if (cell_count_in_grid == 0u) {
            continue;
        }

        let grid_base_offset = neighbor_indices[n] * MAX_CELLS_PER_GRID;

        for (var i = 0u; i < cell_count_in_grid; i++) {
            let other_idx = spatial_grid_cells[grid_base_offset + i];

            if (other_idx == cell_idx) {
                continue;
            }

            let other_pos = positions_in[other_idx].xyz;
            let other_mass = positions_in[other_idx].w;

            // Skip dead neighbors
            if (other_mass < 0.5) {
                continue;
            }

            let other_radius = calculate_radius_from_mass(other_mass);
            let delta = pos - other_pos;
            let dist = length(delta);
            let contact_dist = radius + other_radius + CONTACT_THRESHOLD_NEIGHBOR;

            if (dist < contact_dist) {
                // Neighbor is in contact - skip if same organism (ciliocytes never
                // push cells belonging to their own organism regardless of settings)
                let other_org = organism_labels[other_idx];
                if (my_org != 0xFFFFFFFFu && other_org != 0xFFFFFFFFu && my_org == other_org) {
                    continue;
                }

                // Conveyor force: push the cargo cell along the cilia beat direction (+forward).
                // The ciliocyte's own locomotion is already driven entirely by wall-contact
                // thrust above, so we apply this as a one-sided force on the cargo cell only.
                // Applying an equal-opposite reaction on self would fight the wall thrust and
                // stall the conveyor every time it touches cargo.

                // Convey direction follows the sign of effective_speed
                var convey_dir: vec3<f32>;
                if (effective_speed >= 0.0) {
                    convey_dir = forward;
                } else {
                    convey_dir = -forward;
                }

                let force_mag = abs(effective_speed) * mass * CILIA_PUSH_MULTIPLIER;

                // Apply convey force to cargo cell only
                let convey_on_neighbor = convey_dir * force_mag;
                atomicAdd(&force_accum_x[other_idx], float_to_fixed(convey_on_neighbor.x));
                atomicAdd(&force_accum_y[other_idx], float_to_fixed(convey_on_neighbor.y));
                atomicAdd(&force_accum_z[other_idx], float_to_fixed(convey_on_neighbor.z));
            }
        }
    }

    // === NEIGHBOR ATTRACTION ===
    // Pull nearby non-organism cells gently toward this ciliocyte.
    // This is a one-sided force (applied only to the attracted cell, not self) so it
    // doesn't interfere with wall-crawling.  The intent is a "conveyor channel" effect:
    // cells drifting near a chain of ciliocytes are drawn in and kept close enough
    // to be handed along by successive pushes, rather than escaping sideways.
    //
    // Range scales with attract_force (2-10 units) so low values give a tight,
    // short-range pull while high values cast a wider net.
    // Force falls off linearly from full strength at contact_dist to zero at attract_range.
    let attract_force = v6.w; // cilia_attract_force (0.0 = off, 1.0 = max)
    if (attract_force > 0.001) {
        // Range: 2 units at force=0, up to CILIA_ATTRACT_MAX_RANGE at force=1
        let attract_range = 2.0 + attract_force * CILIA_ATTRACT_MAX_RANGE;

        for (var n = 0u; n < 27u; n++) {
            let cell_count_in_grid = neighbor_counts[n];
            if (cell_count_in_grid == 0u) {
                continue;
            }

            let grid_base_offset = neighbor_indices[n] * MAX_CELLS_PER_GRID;

            for (var i = 0u; i < cell_count_in_grid; i++) {
                let other_idx = spatial_grid_cells[grid_base_offset + i];

                if (other_idx == cell_idx) {
                    continue;
                }

                // Never attract same-organism cells
                let other_org = organism_labels[other_idx];
                if (my_org != 0xFFFFFFFFu && other_org != 0xFFFFFFFFu && my_org == other_org) {
                    continue;
                }

                let other_pos = positions_in[other_idx].xyz;
                let other_mass = positions_in[other_idx].w;

                if (other_mass < 0.5) {
                    continue;
                }

                let other_radius = calculate_radius_from_mass(other_mass);
                let delta = pos - other_pos; // points from other -> self
                let dist = length(delta);
                let contact_dist = radius + other_radius + CONTACT_THRESHOLD_NEIGHBOR;

                // Only attract cells that are outside contact range but within attract_range.
                // Cells already in contact are handled by the push section above.
                if (dist <= contact_dist || dist > attract_range) {
                    continue;
                }

                // Linear falloff: full force at contact edge, zero at attract_range
                let t = 1.0 - (dist - contact_dist) / (attract_range - contact_dist);

                // Direction toward this ciliocyte (delta points away from it, so negate)
                let attract_dir = -normalize(delta);

                // Scale by attract_force, other cell mass and the falloff
                // Using other_mass keeps the acceleration (force/mass) constant regardless of cell size
                let attract_mag = attract_force * other_mass * CILIA_ATTRACT_MULTIPLIER * t;
                let attract_vec = attract_dir * attract_mag;

                atomicAdd(&force_accum_x[other_idx], float_to_fixed(attract_vec.x));
                atomicAdd(&force_accum_y[other_idx], float_to_fixed(attract_vec.y));
                atomicAdd(&force_accum_z[other_idx], float_to_fixed(attract_vec.z));
            }
        }
    }
}
