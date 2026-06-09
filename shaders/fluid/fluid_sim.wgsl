// GPU Fluid Simulation - Pair-based swapping
// 6 directional passes (X, Y, Z), each with 2 checkered phases
// Simple rule: swap neighbors unless it's air-above-water (anti-gravity)

struct FluidParams {
    grid_resolution: u32,
    world_radius: f32,
    cell_size: f32,
    direction: u32,  // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z

    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    time: f32,  // Time for wave animations

    // Gravity parameters
    gravity_magnitude: f32,
    gravity_dir_x: f32,
    gravity_dir_y: f32,
    gravity_dir_z: f32,

    // Per-fluid-type lateral flow probabilities (0.0 to 1.0)
    // Index: 0=Empty (unused), 1=Water, 2=Lava, 3=Steam
    lateral_flow_probability_empty: f32,
    lateral_flow_probability_water: f32,
    lateral_flow_probability_lava: f32,
    lateral_flow_probability_steam: f32,
    
    // Fluid type for spawning (0=Empty, 1=Water, 2=Lava, 3=Steam)
    spawn_fluid_type: u32,

    sub_step: u32,

    // Gravity mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial (toward origin)
    // Radial: world sphere boundary is the effective shell. gravity_magnitude controls strength.
    gravity_mode: u32,
    surface_pressure: f32,  // Tangential smoothing strength for radial mode (0.0-1.0)
    // Overall sun brightness driving the thermal model's baseline air temperature
    // (0 = dark, 3 = comfortable max, >3 = extreme heat).
    sun_brightness: f32,
    _pad_rg2: u32,
}

struct ExtractParams {
    grid_resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read_write> voxels: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> water_velocity: array<atomic<u32>>;
// Rolling-average accumulator: [water_temp_sum, water_count, air_temp_sum, air_count].
// Temperatures are stored as round(celsius) + 50 (range 0..200) to keep sums
// safely within u32 range without needing 64-bit atomics.
@group(0) @binding(4) var<storage, read_write> temp_stats: array<atomic<u32>>;

// Encode a displacement vector (dx, dy, dz each in {-1, 0, +1}) into a packed u32.
// Encoding: 2 bits per axis. 0b00=0, 0b01=+1, 0b10=-1.
// Result is 0 only when all components are zero (no movement sentinel).
fn encode_velocity(dx: i32, dy: i32, dz: i32) -> u32 {
    var result = 0u;
    if dx == 1 { result |= 1u; }
    else if dx == -1 { result |= 2u; }
    if dy == 1 { result |= (1u << 2u); }
    else if dy == -1 { result |= (2u << 2u); }
    if dz == 1 { result |= (1u << 4u); }
    else if dz == -1 { result |= (2u << 4u); }
    return result;
}

// Write water velocity at destination voxel after a successful move.
// Only writes for water (type 1) to avoid steam noise.
fn write_water_velocity(dst_idx: u32, fluid_type: u32, dx: i32, dy: i32, dz: i32) {
    if fluid_type != 1u { return; }
    let packed = encode_velocity(dx, dy, dz);
    if packed != 0u {
        atomicStore(&water_velocity[dst_idx], packed);
    }
}

// Get lateral flow probability for a specific fluid type
fn get_lateral_flow_probability(fluid_type: u32) -> f32 {
    // Return probability based on fluid type
    switch fluid_type {
        case 0u: { return params.lateral_flow_probability_empty; }   // Empty (unused)
        case 1u: { return params.lateral_flow_probability_water; }   // Water
        case 2u: { return params.lateral_flow_probability_lava; }    // Lava
        case 3u: { return params.lateral_flow_probability_steam; }   // Steam
        default: { return params.lateral_flow_probability_water; }  // Default to water
    }
}

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

// Voxel state word layout (32 bits):
//   bits 0-2:   fluid_type (0-7; only 0-3 currently used: empty/water/ice/steam)
//   bits 3-10:  quantized temperature (8 bits, see TEMP_MIN_C/TEMP_MAX_C)
//   bits 11-15: unused
//   bits 16-31: fill_fraction (16 bits, 0-65535 -> 0.0-1.0)
//
// Packing temperature into the same word that gets atomically swapped means
// it travels with the fluid through every existing move/swap site for free —
// no swap call site needs to change.
const FLUID_TYPE_MASK: u32 = 0x7u;
const TEMP_MASK: u32 = 0xFFu;
const TEMP_SHIFT: u32 = 3u;
const TEMP_MIN_C: f32 = -50.0;
const TEMP_MAX_C: f32 = 150.0;

fn get_fluid_type(state: u32) -> u32 {
    return state & FLUID_TYPE_MASK;
}

// Unpack the 8-bit quantized temperature field to degrees Celsius.
fn get_temperature_c(state: u32) -> f32 {
    let q = (state >> TEMP_SHIFT) & TEMP_MASK;
    return TEMP_MIN_C + (f32(q) / 255.0) * (TEMP_MAX_C - TEMP_MIN_C);
}

// Quantize a Celsius value into the 8-bit temperature field (bits already shifted into place).
fn pack_temperature_c(temp_c: f32) -> u32 {
    let t = saturate((temp_c - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C));
    let q = u32(t * 255.0 + 0.5);
    return (q & TEMP_MASK) << TEMP_SHIFT;
}

// Replace just the temperature bits of a state word, preserving fluid_type and fill_fraction.
fn with_temperature_c(state: u32, temp_c: f32) -> u32 {
    return (state & ~(TEMP_MASK << TEMP_SHIFT)) | pack_temperature_c(temp_c);
}

// ---- Thermal model constants ----
// Comfortable sun brightness spans 0 (dark) to 3 (bright); above 3 is extreme heat.
const COMFORTABLE_BRIGHTNESS_MAX: f32 = 3.0;
const DARK_BASELINE_C: f32 = -10.0;        // sun_brightness = 0 reference; well below freezing
const EVAPORATION_FLOOR_C: f32 = 15.6;     // 60°F - dark-voxel floor; no passive evaporation below this
const BOILING_POINT_C: f32 = 100.0;        // steam stays vapor at/above this; condensation possible at any cooler ambient
const SNOW_FALL_PROBABILITY: f32 = 0.04;   // snow drifts down far slower than water/rain
const EVAPORATION_BRISK_C: f32 = 29.4;     // ~85°F - brightness 3 reference; "decent rainfall" territory
const EXTREME_HEAT_SLOPE_C: f32 = 25.0;    // °C added per unit of brightness above the comfortable ceiling
const EVAPORATION_CURVE_POWER: f32 = 2.5;  // shapes the floor->brisk evaporation ramp (gentle at the low end)
const FREEZE_POINT_C: f32 = 0.0;
const FREEZE_HYSTERESIS_C: f32 = 2.0;      // water freezes below (FREEZE_POINT - this); ice melts above (FREEZE_POINT + this) - prevents flicker at the boundary
const THERMAL_INERTIA_RATE: f32 = 0.0015;  // per-tick lerp factor toward ambient - water has real thermal mass and should drift far slower than air (which has none and tracks ambient instantly)
const NATURAL_VAPORIZATION_RATE: f32 = 0.15;  // base per-tick chance scalar once warm enough to evaporate
const NATURAL_CONDENSATION_RATE: f32 = 0.15;  // base per-tick chance scalar once cool enough to condense

const DARKNESS_PENALTY_MAX_C: f32 = 25.0; // unlit voxels run at most this much colder than baseline - large enough that prolonged local darkness can push water below freezing even when overall brightness is moderate

// Baseline air temperature, derived from overall sun brightness (0 = dark,
// 3 = comfortable max, >3 = extreme heat). This is the "reference" temperature
// for fully-lit conditions; individual voxels then sit at-or-below this based
// on their own local light level (see ambient_temp_c).
// TODO: drive `sun_brightness` from the actual sun/light-field intensity once
// that's available in this pass — exposed as a pure function for now so the
// mapping is correct and ready to wire up.
fn baseline_air_temp_c(sun_brightness: f32) -> f32 {
    if sun_brightness <= COMFORTABLE_BRIGHTNESS_MAX {
        return mix(DARK_BASELINE_C, EVAPORATION_BRISK_C, saturate(sun_brightness / COMFORTABLE_BRIGHTNESS_MAX));
    }
    let excess = sun_brightness - COMFORTABLE_BRIGHTNESS_MAX;
    return EVAPORATION_BRISK_C + excess * EXTREME_HEAT_SLOPE_C;
}

// Temporary stand-in for the light-field brightness sample at a world position:
// approximates "upper/sunlit regions are brighter, lower/shadowed regions are
// darker" using height within the world sphere. Replace with an actual
// light-field lookup (see organism_labels-style buffer binding pattern).
fn approx_local_brightness(world_pos: vec3<f32>) -> f32 {
    let height_norm = clamp(world_pos.y / params.world_radius, -1.0, 1.0);
    return saturate(height_norm * 0.5 + 0.5) * params.sun_brightness;
}

// Per-voxel ambient air temperature: starts from the brightness-derived
// baseline, then voxels that receive less local light run cooler — clamped
// to at most DARKNESS_PENALTY_MAX_C below baseline (a fully dark voxel is
// never colder than "baseline minus 10", not an independently-floored value).
fn ambient_temp_c(world_pos: vec3<f32>) -> f32 {
    let baseline_c = baseline_air_temp_c(params.sun_brightness);

    let local_brightness = approx_local_brightness(world_pos);
    let brightness_scale = max(params.sun_brightness, 0.001);
    let darkness = saturate(1.0 - local_brightness / brightness_scale);
    return baseline_c - darkness * DARKNESS_PENALTY_MAX_C;
}

// Check if a grid position is solid based on the solid mask
fn is_solid(x: u32, y: u32, z: u32) -> bool {
    let idx = grid_index(x, y, z);
    return solid_mask[idx] == 1u;
}

// Check if a voxel is encapsulated (surrounded on all 6 sides by solids or water)
// If true, this voxel can be skipped during processing as it cannot move
fn is_encapsulated(x: u32, y: u32, z: u32) -> bool {
    // Never skip fluid inside solid - it needs the push-out path
    if is_solid(x, y, z) {
        return false;
    }
    
    let res = params.grid_resolution;
    
    // Check all 6 neighbors
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1)   // -Z
    );
    
    // If any neighbor is empty (0), this voxel is not encapsulated
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + offsets[i].x;
        let ny = i32(y) + offsets[i].y;
        let nz = i32(z) + offsets[i].z;
        
        // Bounds check - if at boundary, consider not encapsulated
        if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
            return false;
        }
        
        let neighbor_idx = grid_index(u32(nx), u32(ny), u32(nz));
        let neighbor_state = atomicLoad(&voxels[neighbor_idx]);
        let neighbor_type = get_fluid_type(neighbor_state);
        
        // If any neighbor is empty (0), steam (3), or water (1), this voxel can potentially move
        // (water can swap with steam, so they don't encapsulate each other)
        if neighbor_type == 0u || neighbor_type == 3u || neighbor_type == 1u {
            return false;
        }
    }
    
    // All 6 neighbors are solids (2+), so this voxel is truly encapsulated
    return true;
}

fn grid_to_world(x: u32, y: u32, z: u32) -> vec3<f32> {
    return vec3<f32>(
        params.grid_origin_x + (f32(x) + 0.5) * params.cell_size,
        params.grid_origin_y + (f32(y) + 0.5) * params.cell_size,
        params.grid_origin_z + (f32(z) + 0.5) * params.cell_size
    );
}

fn is_in_bounds(pos: vec3<f32>) -> bool {
    // Inset by one voxel so fluid is pushed slightly inside the solid mask,
    // preventing air-gap bubbles around curved cave surfaces.
    let inset = params.world_radius - params.cell_size;
    return dot(pos, pos) < inset * inset;
}

// Get the effective gravity direction for a voxel.
// gravity_mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial (toward origin)
fn get_effective_gravity(gid: vec3<u32>) -> vec3<f32> {
    let mag = params.gravity_magnitude;
    
    if params.gravity_mode == 0u {
        // X axis gravity
        return vec3<f32>(-mag, 0.0, 0.0);
    } else if params.gravity_mode == 1u {
        // Y axis gravity
        return vec3<f32>(0.0, -mag, 0.0);
    } else if params.gravity_mode == 2u {
        // Z axis gravity
        return vec3<f32>(0.0, 0.0, -mag);
    }
    
    // Radial: positive mag = pull toward origin (shell), negative = push away (explode)
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let r = length(world_pos);
    if r < 0.001 {
        return vec3<f32>(0.0, -mag, 0.0);
    }
    // radial_dir points outward; negate to get inward pull, sign(mag) flips for outward push
    let radial_dir = world_pos / r;
    return -radial_dir * mag;
}

// Convert a gravity direction vector to a discrete direction index (0-5).
// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
// Uses position-based noise to probabilistically break cubic symmetry:
// when two axes have similar magnitude, randomly alternate between them
// so the boundary between octants is fuzzy rather than a sharp plane.
fn gravity_dir_to_index_noisy(grav_dir: vec3<f32>, gid: vec3<u32>) -> u32 {
    let ax = abs(grav_dir.x);
    let ay = abs(grav_dir.y);
    let az = abs(grav_dir.z);
    let total = ax + ay + az + 0.0001; // avoid div-by-zero
    
    // Position + time hash for noise (0..255)
    let h = (gid.x * 73u + gid.y * 157u + gid.z * 239u + u32(params.time * 60.0)) & 255u;
    let noise = f32(h) / 255.0; // 0..1
    
    // Weighted random selection: probability of picking each axis  its component magnitude
    let px = ax / total;
    let py = ay / total;
    // pz = az / total = 1 - px - py
    
    if noise < px {
        return select(1u, 0u, grav_dir.x > 0.0);
    } else if noise < px + py {
        return select(3u, 2u, grav_dir.y > 0.0);
    } else if az > 0.0 {
        return select(5u, 4u, grav_dir.z > 0.0);
    }
    return 3u; // Default to -Y
}

// Deterministic version (no noise) for support checks and fast-drop
fn gravity_dir_to_index(grav_dir: vec3<f32>) -> u32 {
    let ax = abs(grav_dir.x);
    let ay = abs(grav_dir.y);
    let az = abs(grav_dir.z);
    
    if ax > ay && ax > az {
        return select(1u, 0u, grav_dir.x > 0.0);
    } else if ay > az {
        return select(3u, 2u, grav_dir.y > 0.0);
    } else if az > 0.0 {
        return select(5u, 4u, grav_dir.z > 0.0);
    }
    return 3u; // Default to -Y
}

// Direction offsets: +X, -X, +Y, -Y, +Z, -Z
fn get_offset(dir: u32) -> vec3<i32> {
    switch dir {
        case 0u: { return vec3<i32>(1, 0, 0); }   // +X
        case 1u: { return vec3<i32>(-1, 0, 0); }  // -X
        case 2u: { return vec3<i32>(0, 1, 0); }   // +Y (up)
        case 3u: { return vec3<i32>(0, -1, 0); }  // -Y (down/gravity)
        case 4u: { return vec3<i32>(0, 0, 1); }   // +Z
        default: { return vec3<i32>(0, 0, -1); }  // -Z
    }
}

// Get the coordinate used for checkering based on direction
fn get_checker_coord(pos: vec3<u32>, dir: u32) -> u32 {
    switch dir {
        case 0u, 1u: { return pos.x; }  // X direction: checker on X
        case 2u, 3u: { return pos.y; }  // Y direction: checker on Y
        default: { return pos.z; }       // Z direction: checker on Z
    }
}

// Hash-based randomization for fair direction competition
fn hash_position(pos: vec3<u32>) -> u32 {
    return (pos.x * 73856093u ^ pos.y * 19349663u ^ pos.z * 83492791u) & 0xFFFFFFFFu;
}

// Enhanced dispersion for steam - makes steam spread out more like a gas
fn get_steam_dispersion_bias(gid: vec3<u32>, direction: u32) -> f32 {
    // Steam naturally wants to disperse and fill available space
    let pos_hash = hash_position(gid);
    let time_hash = u32(params.time * 1000.0);
    
    // Create different dispersion patterns based on direction
    let dispersion_factor = sin(f32(pos_hash) * 0.001 + params.time * 2.0) * 0.3 + 0.7;
    
    // Steam has higher dispersion in horizontal directions (spreads out)
    if direction == 0u || direction == 1u || direction == 4u || direction == 5u {
        return dispersion_factor * 1.5; // Boost horizontal spreading
    } else {
        return dispersion_factor * 1.2; // Increase vertical movement bias for rising
    }
}

// True if any of the 6 face-neighbors of this voxel is empty (air). Used to
// gate evaporation - water needs an open surface to vaporize from, the same
// way real water evaporates from its exposed surface rather than its interior.
fn in_contact_with_air(gid: vec3<u32>) -> bool {
    let res = params.grid_resolution;
    let neighbor_offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),
        vec3<i32>(-1, 0, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 1),
        vec3<i32>(0, 0, -1)
    );
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(gid.x) + neighbor_offsets[i].x;
        let ny = i32(gid.y) + neighbor_offsets[i].y;
        let nz = i32(gid.z) + neighbor_offsets[i].z;
        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            let n_state = atomicLoad(&voxels[grid_index(u32(nx), u32(ny), u32(nz))]);
            if get_fluid_type(n_state) == 0u {
                return true;
            }
        }
    }
    return false;
}

// Condensation mechanic - steam can condense back to water when contacting solids or boundaries
fn should_condense_steam(gid: vec3<u32>) -> bool {
    let res = params.grid_resolution;

    // Steam needs a surface to condense onto: near the world boundary or
    // touching a solid (cave wall, rock, etc). Floating mid-air steam stays
    // vapor regardless of temperature.
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let distance_from_center = length(world_pos);
    let boundary_threshold = params.world_radius - params.cell_size;
    let near_boundary = distance_from_center > boundary_threshold;

    let solid_neighbors = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),
        vec3<i32>(-1, 0, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 1),
        vec3<i32>(0, 0, -1)
    );

    var adjacent_solids = 0u;
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(gid.x) + solid_neighbors[i].x;
        let ny = i32(gid.y) + solid_neighbors[i].y;
        let nz = i32(gid.z) + solid_neighbors[i].z;

        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            if is_solid(u32(nx), u32(ny), u32(nz)) {
                adjacent_solids++;
            }
        }
    }

    if !near_boundary && adjacent_solids == 0u {
        return false;
    }

    // Purely thermal: steam can condense at any ambient temperature below
    // boiling - all the way down through freezing (where it forms snow
    // instead of liquid water; see fluid_swap). It is NOT gated by the
    // much-higher evaporation floor - that threshold only governs whether
    // liquid water passively evaporates, not whether vapor condenses back.
    let ambient_c = ambient_temp_c(world_pos);
    if ambient_c >= BOILING_POINT_C {
        return false;
    }

    // Use hash-based randomization for natural variation
    let pos_hash = hash_position(gid);
    let time_hash = u32(params.time * 1000.0);
    let combined_hash = pos_hash ^ time_hash;

    return (combined_hash & 255u) < u32(NATURAL_CONDENSATION_RATE * 255.0);
}


// Check if water voxel is supported from below (in the gravity direction)
// Only water resting on solid or other water should spread laterally.
// Water clinging to walls or ceilings is NOT supported and should fall.
fn water_is_supported(gid: vec3<u32>) -> bool {
    let res = params.grid_resolution;
    let grav_dir = get_effective_gravity(gid);
    
    // Find the primary gravity direction
    let gravity_dir_index = gravity_dir_to_index(grav_dir);
    
    let down = get_offset(gravity_dir_index);
    let nx = i32(gid.x) + down.x;
    let ny = i32(gid.y) + down.y;
    let nz = i32(gid.z) + down.z;
    
    // Out of bounds below = not supported
    if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
        return false;
    }
    
    // Solid below = supported (water rests on this surface and can spread laterally).
    // Lateral spreading + gravity naturally produces wall-sliding behaviour:
    // each lateral step may expose an empty cell below, letting gravity act.
    if is_solid(u32(nx), u32(ny), u32(nz)) {
        return true;
    }
    
    // Water, lava, or steam below = supported (resting on fluid)
    let neighbor_idx = u32(nx) + u32(ny) * res + u32(nz) * res * res;
    let neighbor_state = atomicLoad(&voxels[neighbor_idx]);
    let neighbor_type = get_fluid_type(neighbor_state);
    
    // Steam (3) also supports water - water sits on top of steam bubbles
    return neighbor_type >= 1u && neighbor_type <= 3u;
}

// Check if a voxel is at or very close to the sphere boundary
fn is_at_sphere_boundary(gid: vec3<u32>) -> bool {
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let distance_from_center = length(world_pos);
    // Match the is_in_bounds inset so boundary sliding kicks in at the fluid edge
    let boundary_threshold = params.world_radius - params.cell_size;
    return distance_from_center >= boundary_threshold;
}

// Get randomized horizontal direction order based on position and time, relative to gravity
fn get_horizontal_direction_order(gid: vec3<u32>, time: f32, gravity_dir_index: u32) -> array<u32, 4> {
    let pos_hash = hash_position(gid);
    let time_hash = u32(time * 1000.0) & 0xFFFFFFFFu;
    let combined_hash = pos_hash ^ time_hash;
    
    // Use hash to determine starting offset in direction array
    let start_offset = combined_hash & 3u;
    
    // Define horizontal directions based on gravity axis
    var all_directions: array<u32, 4>;
    
    if gravity_dir_index == 2u || gravity_dir_index == 3u {
        // Y gravity - use X and Z directions
        all_directions = array<u32, 4>(0u, 1u, 4u, 5u); // +X, -X, +Z, -Z
    } else if gravity_dir_index == 0u || gravity_dir_index == 1u {
        // X gravity - use Y and Z directions
        all_directions = array<u32, 4>(2u, 3u, 4u, 5u); // +Y, -Y, +Z, -Z
    } else {
        // Z gravity - use X and Y directions
        all_directions = array<u32, 4>(0u, 1u, 2u, 3u); // +X, -X, +Y, -Y
    }
    
    // Create rotated order based on hash
    var order: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        order[i] = all_directions[(start_offset + i) & 3u];
    }
    
    return order;
}

// Compute tangential surface-tension force for radial gravity mode.
// Iterates all 26 neighbors, decomposes each neighbor direction into
// radial + tangential components, and accumulates a net tangential force
// toward the less-occupied side. This avoids the round()-quantization
// bias of sampling along fixed tangent vectors.
fn get_surface_force(gid: vec3<u32>) -> vec3<f32> {
    if params.gravity_mode != 3u {
        return vec3<f32>(0.0);
    }

    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let r = length(world_pos);
    if r < 0.001 { return vec3<f32>(0.0); }
    let radial = world_pos / r;

    let res = i32(params.grid_resolution);
    var tangential_sum = vec3<f32>(0.0);

    // Accumulate tangential component of each occupied neighbor's direction
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let nx = i32(gid.x) + dx;
                let ny = i32(gid.y) + dy;
                let nz = i32(gid.z) + dz;
                if nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res { continue; }

                let n_type = get_fluid_type(atomicLoad(&voxels[grid_index(u32(nx), u32(ny), u32(nz))]));
                if n_type == 0u { continue; } // only occupied neighbors pull

                // Direction to neighbor (normalized)
                let dir = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let dir_n = dir / length(dir);

                // Tangential component = dir - (dirradial)*radial
                let tangential = dir_n - dot(dir_n, radial) * radial;
                tangential_sum += tangential;
            }
        }
    }

    // Net tangential force pulls toward the denser side;
    // surface pressure slides voxel away from dense neighbors -> negate.
    // Scale by gravity magnitude so surface smoothing is proportional to gravity strength.
    let grav_mag = length(get_effective_gravity(gid));
    return -tangential_sum * params.surface_pressure * grav_mag * 0.1;
}

// Radial movement: score all 26 neighbors, pick the best empty one, atomic swap.
// This is the 3D equivalent of the JS reference demo's "evaluate 8 neighbors" logic.
fn radial_move(gid: vec3<u32>) {
    let res = params.grid_resolution;
    let idx = grid_index(gid.x, gid.y, gid.z);
    let state = atomicLoad(&voxels[idx]);
    let fluid_type = get_fluid_type(state);

    // Only move water (1) and steam (3)
    if fluid_type == 0u || fluid_type == 2u { return; }

    // Gravity force (used for probability gate)
    let grav_force = get_effective_gravity(gid);
    let grav_mag = length(grav_force);
    if grav_mag < 0.001 { return; }

    // Combined force for direction scoring: gravity + tangential surface tension
    let surf_force = get_surface_force(gid);
    var total_force = grav_force + surf_force;

    // Reverse force for steam (rises against gravity)
    if fluid_type == 3u {
        total_force = -total_force;
    }

    let force_mag = length(total_force);
    if force_mag < 0.001 { return; }
    let force_dir = total_force / force_mag;

    // Probability gate: surface voxels with significant tangential force always attempt
    // to move so the surface can continuously level itself. Interior voxels use the
    // gravity-magnitude gate to avoid wasted work.
    // Use non-linear hash to avoid planar banding artifacts.
    let surf_mag = length(surf_force);
    let is_surface_voxel = surf_mag > 0.02;
    if !is_surface_voxel {
        let gravity_probability = min(1.0, grav_mag * grav_mag * 0.0004);
        let prob_hash = (hash_position(gid) ^ u32(params.time * 1000.0)) & 255u;
        if prob_hash > u32(gravity_probability * 255.0) { return; }
    }

    // Score all 26 neighbors, pick the best empty target.
    // Lower threshold for surface voxels so tangential moves toward empty space are accepted.
    var best_score = select(0.05, 0.01, is_surface_voxel);
    var best_dx = 0;
    var best_dy = 0;
    var best_dz = 0;
    var found = false;

    // Noise seed for tie-breaking - use non-linear hash to avoid planar banding
    let noise_seed = hash_position(gid) ^ u32(params.time * 1000.0);

    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                if dx == 0 && dy == 0 && dz == 0 { continue; }

                let nx = i32(gid.x) + dx;
                let ny = i32(gid.y) + dy;
                let nz = i32(gid.z) + dz;

                // Bounds
                if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) { continue; }

                // Solid
                if is_solid(u32(nx), u32(ny), u32(nz)) { continue; }

                // Must be empty
                let n_idx = grid_index(u32(nx), u32(ny), u32(nz));
                let n_state = atomicLoad(&voxels[n_idx]);
                if get_fluid_type(n_state) != 0u { continue; }

                // Alignment score: dot(neighbor_direction, force_direction)
                let dir = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let dir_len = length(dir);
                var score = dot(dir, force_dir) / dir_len;

                // Tie-breaking noise (0.04)
                let nh = (noise_seed ^ (u32(nx) * 31u + u32(ny) * 97u + u32(nz) * 61u)) & 255u;
                score += (f32(nh) / 255.0 - 0.5) * 0.08;

                if score > best_score {
                    best_score = score;
                    best_dx = dx;
                    best_dy = dy;
                    best_dz = dz;
                    found = true;
                }
            }
        }
    }

    if !found { return; }

    let target_idx = grid_index(u32(i32(gid.x) + best_dx),
                                u32(i32(gid.y) + best_dy),
                                u32(i32(gid.z) + best_dz));

    // Atomic CAS swap: claim source, then claim target
    let claim_src = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
    if !claim_src.exchanged { return; }

    let target_state = atomicLoad(&voxels[target_idx]);
    if get_fluid_type(target_state) != 0u {
        // Target no longer empty - restore source
        atomicExchange(&voxels[idx], state);
        return;
    }

    let claim_dst = atomicCompareExchangeWeak(&voxels[target_idx], target_state, 0xFFFFFFFFu);
    if !claim_dst.exchanged {
        atomicExchange(&voxels[idx], state);
        return;
    }

    // Both claimed - swap
    atomicStore(&voxels[idx], target_state);
    atomicStore(&voxels[target_idx], state);

    // Record water velocity at destination
    write_water_velocity(target_idx, fluid_type, best_dx, best_dy, best_dz);
}

// Per-tick thermal update: water and ice drift toward ambient temperature with
// inertia (Newton's-law-of-cooling style exponential approach). Runs as its own
// pass, separate from fluid_swap, so the read-modify-write here never contends
// with the swap CAS loops — if a voxel moves away mid-tick, the CAS below simply
// loses the race harmlessly and the new occupant picks up the drift next tick.
//
// Temperature is packed into the same word that gets atomically swapped
// (see with_temperature_c), so once set here it travels with the fluid through
// every existing move/swap site for free.
@compute @workgroup_size(4, 4, 4)
fn update_temperature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let state = atomicLoad(&voxels[idx]);
    let fluid_type = get_fluid_type(state);
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let ambient_c = ambient_temp_c(world_pos);

    // Only water and ice carry tracked thermal mass; empty space, steam and
    // solids derive/ignore temperature rather than storing it.
    if fluid_type != 1u && fluid_type != 2u {
        // Empty/air voxel: contribute the ambient temperature to the air
        // rolling-average accumulator.
        if fluid_type == 0u {
            let encoded = u32(round(ambient_c) + 50.0);
            atomicAdd(&temp_stats[2], encoded);
            atomicAdd(&temp_stats[3], 1u);
        }
        return;
    }

    let current_c = get_temperature_c(state);
    let new_c = mix(current_c, ambient_c, THERMAL_INERTIA_RATE);

    let new_state = with_temperature_c(state, new_c);
    if new_state != state {
        atomicCompareExchangeWeak(&voxels[idx], state, new_state);
    }

    let encoded_water = u32(round(new_c) + 50.0);
    atomicAdd(&temp_stats[0], encoded_water);
    atomicAdd(&temp_stats[1], 1u);
}

// Main simulation pass - GPU handles all movement with proper physics order
@compute @workgroup_size(4, 4, 4)
fn fluid_swap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    // Optimization: Skip processing encapsulated voxels
    // These voxels are surrounded on all 6 sides by solids or water and cannot move
    if is_encapsulated(gid.x, gid.y, gid.z) {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let state = atomicLoad(&voxels[idx]);
    let fluid_type = get_fluid_type(state);
    
    // Steam condensation check - convert steam back to water when contacting solids.
    // Condensed water inherits the ambient temperature of the spot it forms at
    // (steam itself doesn't track temperature) rather than an arbitrary default.
    if fluid_type == 3u && should_condense_steam(gid) {
        let world_pos = grid_to_world(gid.x, gid.y, gid.z);
        let condensed_c = ambient_temp_c(world_pos);
        // Below freezing, vapor condenses directly into snow rather than
        // liquid water - condensation itself never stops in cold weather,
        // only evaporation does.
        var condensed_type = 1u;
        if condensed_c < FREEZE_POINT_C {
            condensed_type = 4u;
        }
        let new_state = (65535u << 16u) | pack_temperature_c(condensed_c) | condensed_type;
        let result = atomicCompareExchangeWeak(&voxels[idx], state, new_state);
        if result.exchanged {
            return; // Successfully condensed, no further processing needed
        }
    }

    // --- Snow: drifts down slowly, and turns directly to ice on contact
    // with liquid water or any solid surface (it doesn't melt to water first). ---
    if fluid_type == 4u {
        let res_snow = params.grid_resolution;
        let neighbor_offsets_snow = array<vec3<i32>, 6>(
            vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0),
            vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0),
            vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1)
        );
        var touches_water_or_solid = false;
        for (var i = 0u; i < 6u; i++) {
            let nx = i32(gid.x) + neighbor_offsets_snow[i].x;
            let ny = i32(gid.y) + neighbor_offsets_snow[i].y;
            let nz = i32(gid.z) + neighbor_offsets_snow[i].z;
            if nx < 0 || nx >= i32(res_snow) || ny < 0 || ny >= i32(res_snow) || nz < 0 || nz >= i32(res_snow) {
                continue;
            }
            if is_solid(u32(nx), u32(ny), u32(nz)) {
                touches_water_or_solid = true;
                break;
            }
            let n_state_snow = atomicLoad(&voxels[grid_index(u32(nx), u32(ny), u32(nz))]);
            if get_fluid_type(n_state_snow) == 1u {
                touches_water_or_solid = true;
                break;
            }
        }

        if touches_water_or_solid {
            let new_state = (state & ~FLUID_TYPE_MASK) | 2u;
            let result = atomicCompareExchangeWeak(&voxels[idx], state, new_state);
            if result.exchanged { return; }
        } else {
            // Drift slowly downward in the gravity direction into empty space.
            let pos_hash_snow = hash_position(gid);
            let time_hash_snow = u32(params.time * 1000.0);
            if ((pos_hash_snow ^ time_hash_snow) & 255u) < u32(SNOW_FALL_PROBABILITY * 255.0) {
                let grav_dir_snow = get_effective_gravity(gid);
                let gravity_dir_index_snow = gravity_dir_to_index(grav_dir_snow);
                let down_snow = get_offset(gravity_dir_index_snow);
                let dx = i32(gid.x) + down_snow.x;
                let dy = i32(gid.y) + down_snow.y;
                let dz = i32(gid.z) + down_snow.z;
                if dx >= 0 && dx < i32(res_snow) && dy >= 0 && dy < i32(res_snow) && dz >= 0 && dz < i32(res_snow) {
                    if !is_solid(u32(dx), u32(dy), u32(dz)) {
                        let down_idx = grid_index(u32(dx), u32(dy), u32(dz));
                        let down_state = atomicLoad(&voxels[down_idx]);
                        let down_type = get_fluid_type(down_state);
                        // Snow falls freely through both empty space and steam -
                        // it's denser than vapor and shouldn't be held up by it.
                        if down_type == 0u || down_type == 3u {
                            let claim = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
                            if claim.exchanged {
                                let result_down = atomicCompareExchangeWeak(&voxels[down_idx], down_state, state);
                                if result_down.exchanged {
                                    atomicStore(&voxels[idx], down_state);
                                } else {
                                    atomicStore(&voxels[idx], state);
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // Thermally-driven phase changes for water and ice.
    if fluid_type == 1u || fluid_type == 2u {
        let temp_c = get_temperature_c(state);

        // --- Vaporization: passive evaporation ramping toward a rolling boil. ---
        // Purely thermal now - no rock proximity and no manual control. Below
        // the floor (60°F/15.6°C) nothing evaporates; the rate ramps with a
        // curve that stays nearly imperceptible just above the floor and
        // climbs toward saturation as the water approaches the brisk/boiling
        // reference. Also requires the voxel to actually be in contact with
        // air (steam needs somewhere to go) - water buried deep in a pool
        // doesn't spontaneously vaporize.
        if fluid_type == 1u && in_contact_with_air(gid) {
            if temp_c > EVAPORATION_FLOOR_C {
                let warmth = saturate((temp_c - EVAPORATION_FLOOR_C) / (EVAPORATION_BRISK_C - EVAPORATION_FLOOR_C));
                let rate = NATURAL_VAPORIZATION_RATE * pow(warmth, EVAPORATION_CURVE_POWER);
                let pos_hash = hash_position(gid);
                let time_hash = u32(params.time * 1000.0);
                let combined_hash = pos_hash ^ time_hash;
                if (combined_hash & 255u) < u32(rate * 255.0) {
                    // Steam carries no tracked temperature - clear the field.
                    let new_state = (65535u << 16u) | 3u;
                    let result = atomicCompareExchangeWeak(&voxels[idx], state, new_state);
                    if result.exchanged {
                        return;
                    }
                }
            }
        }

        // --- Freezing / melting with hysteresis. ---
        // In-place 1:1 phase flip (water <-> ice occupy the same fluid slot,
        // no displacement). Temperature is preserved across the flip so the
        // voxel keeps drifting thermally afterward without a discontinuity.
        // Freeze and melt use offset thresholds (FREEZE_POINT ± hysteresis)
        // rather than a single shared point - this stops a voxel hovering
        // right at 0°C from flickering between water and ice every tick.
        // Ice always claims the FULL voxel volume (fill = 1.0) when it forms,
        // replacing the water's volume outright rather than inheriting a
        // possibly-partial fill fraction (ice expands to fill its space).
        if fluid_type == 1u && temp_c < FREEZE_POINT_C - FREEZE_HYSTERESIS_C {
            let new_state = (65535u << 16u) | (state & (TEMP_MASK << TEMP_SHIFT)) | 2u;
            let result = atomicCompareExchangeWeak(&voxels[idx], state, new_state);
            if result.exchanged {
                return;
            }
        } else if fluid_type == 2u && temp_c > FREEZE_POINT_C + FREEZE_HYSTERESIS_C {
            let new_state = (state & ~FLUID_TYPE_MASK) | 1u;
            let result = atomicCompareExchangeWeak(&voxels[idx], state, new_state);
            if result.exchanged {
                return;
            }
        }
    }

    // Steam teleportation - find nearest water above and swap with it (directional mode only)
    // Steam rises by swapping with the lowest water voxel directly above it.
    // The scan passes through empty cells and other steam, but stops at solids.
    if fluid_type == 3u && params.gravity_mode != 3u {
        let grav_dir_for_steam = get_effective_gravity(gid);
        let gravity_dir_index_steam = gravity_dir_to_index(grav_dir_for_steam);
        // "Up" for steam is opposite to gravity
        let up_offset = get_offset(gravity_dir_index_steam ^ 1u);

        // Scan upward (against gravity) to find the nearest water voxel
        // Cap at 16 steps - enough for realistic steam behavior, avoids 64-step worst case
        for (var step = 1; step <= 16; step++) {
            let sx = i32(gid.x) + up_offset.x * step;
            let sy = i32(gid.y) + up_offset.y * step;
            let sz = i32(gid.z) + up_offset.z * step;

            if sx < 0 || sx >= i32(params.grid_resolution) ||
               sy < 0 || sy >= i32(params.grid_resolution) ||
               sz < 0 || sz >= i32(params.grid_resolution) {
                break;
            }

            if is_solid(u32(sx), u32(sy), u32(sz)) {
                break; // Solid blocks the path
            }

            let scan_idx = grid_index(u32(sx), u32(sy), u32(sz));
            let scan_state = atomicLoad(&voxels[scan_idx]);
            let scan_fluid_type = get_fluid_type(scan_state);

            if scan_fluid_type == 1u {
                // Found water - swap with it (nearest water above, not topmost)
                let water_idx_tele = scan_idx;
                let water_state_tele = scan_state;

                // Two-phase CAS swap: claim water first, then steam
                let water_result_tele = atomicCompareExchangeWeak(&voxels[water_idx_tele], water_state_tele, 0xFFFFFFFFu);
                if water_result_tele.exchanged {
                    let steam_result_tele = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
                    if steam_result_tele.exchanged {
                        atomicStore(&voxels[water_idx_tele], state);       // steam goes up
                        atomicStore(&voxels[idx], water_state_tele);       // water comes down
                        return;
                    } else {
                        atomicStore(&voxels[water_idx_tele], water_state_tele);
                    }
                }
                break; // Whether swap succeeded or not, stop scanning
            }
            // Empty (0) or steam (3): keep scanning upward through them
        }
        // Swap is handled inline above; fall through to normal movement if no swap occurred
    }

    // Steam lateral escape: when the upward path is blocked (solid or out of bounds),
    // try all 4 lateral directions with high probability so steam spreads around obstacles.
    if fluid_type == 3u && params.gravity_mode != 3u {
        let grav_dir_lat = get_effective_gravity(gid);
        let gravity_dir_index_lat = gravity_dir_to_index(grav_dir_lat);
        let up_offset_lat = get_offset(gravity_dir_index_lat ^ 1u);

        // Check if the cell directly above is blocked (solid, out of bounds, or water that
        // the teleportation already handled - meaning we're still here because it failed)
        let above_x = i32(gid.x) + up_offset_lat.x;
        let above_y = i32(gid.y) + up_offset_lat.y;
        let above_z = i32(gid.z) + up_offset_lat.z;
        let res_i = i32(params.grid_resolution);

        // Check if the cell directly above is blocked (solid or out of bounds).
        // We check bounds first, then solid, then fluid type.
        let above_oob =
            above_x < 0 || above_x >= res_i ||
            above_y < 0 || above_y >= res_i ||
            above_z < 0 || above_z >= res_i;

        var above_blocked = above_oob;
        if !above_oob {
            let ux = u32(above_x);
            let uy = u32(above_y);
            let uz = u32(above_z);
            if is_solid(ux, uy, uz) {
                above_blocked = true;
            } else {
                let above_type = get_fluid_type(atomicLoad(&voxels[grid_index(ux, uy, uz)]));
                // Non-empty above means blocked: water is handled by teleport (we're still
                // here because it failed or CAS raced), steam above = already occupied
                above_blocked = above_type != 0u;
            }
        }

        if above_blocked {
            // Try all 4 lateral directions in randomised order
            let lateral_order = get_horizontal_direction_order(gid, params.time, gravity_dir_index_lat);
            for (var li = 0u; li < 4u; li++) {
                let lat_dir = lateral_order[li];
                let lat_off = get_offset(lat_dir);
                let lx = i32(gid.x) + lat_off.x;
                let ly = i32(gid.y) + lat_off.y;
                let lz = i32(gid.z) + lat_off.z;

                if lx < 0 || lx >= res_i || ly < 0 || ly >= res_i || lz < 0 || lz >= res_i {
                    continue;
                }
                if is_solid(u32(lx), u32(ly), u32(lz)) {
                    continue;
                }

                let lat_idx = grid_index(u32(lx), u32(ly), u32(lz));
                let lat_state = atomicLoad(&voxels[lat_idx]);
                let lat_type = get_fluid_type(lat_state);

                // Only move into empty cells laterally (steam-water lateral handled by process_direction)
                if lat_type != 0u {
                    continue;
                }

                // High probability lateral escape - steam behaves like a gas
                let time_hash = u32(params.time * 1000.0) + lat_dir * 12345u;
                let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
                let combined_hash = time_hash ^ pos_hash;
                let lateral_prob = params.lateral_flow_probability_steam;
                if (combined_hash & 255u) > u32(lateral_prob * 255.0) {
                    continue;
                }

                // CAS swap
                let claim_src = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
                if !claim_src.exchanged { break; }

                let claim_dst = atomicCompareExchangeWeak(&voxels[lat_idx], lat_state, 0xFFFFFFFFu);
                if !claim_dst.exchanged {
                    atomicStore(&voxels[idx], state);
                    continue;
                }

                atomicStore(&voxels[idx], lat_state);   // empty goes to steam's old spot
                atomicStore(&voxels[lat_idx], state);   // steam moves laterally
                return;
            }
        }
    }

    // Cache support check once - reused by fast-drop and all process_direction calls below.
    // water_is_supported() does a neighbor atomic load; calling it once here avoids
    // repeating it up to 8 more times (twice per process_direction x 4 horizontal calls).
    let this_voxel_supported = params.gravity_mode == 3u || fluid_type != 1u || water_is_supported(gid);

    // Water fast-drop: unsupported water falls instantly to nearest support
    // This bypasses checker/probability gates in process_direction
    // Skip in radial mode - radial_move() handles all movement there.
    if fluid_type == 1u && params.gravity_mode != 3u && !this_voxel_supported {
        let grav_dir = get_effective_gravity(gid);
        let gravity_dir_index = gravity_dir_to_index(grav_dir);
        
        let down = get_offset(gravity_dir_index);
        
        // Scan downward to find the lowest empty cell above a surface
        // Cap at 16 steps - matches steam teleport cap, avoids 64-step worst case
        var target_y = -1;
        for (var step = 1; step <= 16; step++) {
            let sx = i32(gid.x) + down.x * step;
            let sy = i32(gid.y) + down.y * step;
            let sz = i32(gid.z) + down.z * step;
            
            if sx < 0 || sx >= i32(res) || sy < 0 || sy >= i32(res) || sz < 0 || sz >= i32(res) {
                break;
            }
            
            if is_solid(u32(sx), u32(sy), u32(sz)) {
                break; // Hit solid, target is one step above
            }
            
            let scan_idx = grid_index(u32(sx), u32(sy), u32(sz));
            let scan_state = atomicLoad(&voxels[scan_idx]);
            let scan_type = get_fluid_type(scan_state);
            
            if scan_type == 3u {
                // Steam blocks water from falling through - water sits on top of steam
                break;
            } else if scan_type == 0u {
                // Empty cell - this is the target (don't skip through empty space)
                if !is_solid(u32(sx), u32(sy), u32(sz)) {
                    target_y = step;
                    break; // Found empty space, stop scanning
                }
            } else if scan_type == 1u || scan_type == 2u {
                break; // Hit water/lava surface, target is one step above
            }
        }
        
        // If we found a position lower than current, teleport there
        if target_y > 1 {
            let tx = i32(gid.x) + down.x * target_y;
            let ty = i32(gid.y) + down.y * target_y;
            let tz = i32(gid.z) + down.z * target_y;
            let target_idx = grid_index(u32(tx), u32(ty), u32(tz));
            
            // Claim source first to prevent duplication
            let claim_result = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
            if claim_result.exchanged {
                let target_state = atomicLoad(&voxels[target_idx]);
                if get_fluid_type(target_state) == 0u {
                    let result = atomicCompareExchangeWeak(&voxels[target_idx], target_state, state);
                    if result.exchanged {
                        // Success: clear the claimed source
                        atomicStore(&voxels[idx], 0u);
                        // Record water velocity (falling in gravity direction)
                        write_water_velocity(target_idx, fluid_type, down.x, down.y, down.z);
                        return;
                    }
                }
                // Target was taken, restore source
                atomicStore(&voxels[idx], state);
            }
        }
        // If target_y == 1, just one cell below - normal process_direction handles it
    }

    // Special case: Push any fluid out of solid voxels
    // Search up to 16 voxels along each axis to find a non-solid empty cell
    if fluid_type >= 1u && is_solid(gid.x, gid.y, gid.z) {
        let directions = array<vec3<i32>, 6>(
            vec3<i32>(1, 0, 0),   // +X
            vec3<i32>(-1, 0, 0),  // -X
            vec3<i32>(0, 1, 0),   // +Y
            vec3<i32>(0, -1, 0),  // -Y
            vec3<i32>(0, 0, 1),   // +Z
            vec3<i32>(0, 0, -1)   // -Z
        );
        
        // Randomize starting direction to avoid bias
        let dir_hash = (hash_position(gid) ^ u32(params.time * 1000.0)) % 6u;
        
        var escaped = false;
        for (var d = 0u; d < 6u; d++) {
            let dir_idx = (dir_hash + d) % 6u;
            let dir = directions[dir_idx];
            
            // Walk along this axis up to 16 voxels
            for (var step = 1; step <= 16; step++) {
                let nx = i32(gid.x) + dir.x * step;
                let ny = i32(gid.y) + dir.y * step;
                let nz = i32(gid.z) + dir.z * step;
                
                if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
                    break; // Hit grid boundary, try next direction
                }
                
                // Skip cells that are still solid
                if is_solid(u32(nx), u32(ny), u32(nz)) {
                    continue;
                }
                
                let target_idx = grid_index(u32(nx), u32(ny), u32(nz));
                let target_state = atomicLoad(&voxels[target_idx]);
                let target_type = get_fluid_type(target_state);
                
                if target_type == 0u {
                    // Found empty non-solid cell - teleport fluid there
                    let result = atomicCompareExchangeWeak(&voxels[target_idx], target_state, state);
                    if result.exchanged {
                        atomicStore(&voxels[idx], 0u);
                        escaped = true;
                    }
                    break;
                }
                
                // Hit another fluid - can't place here, stop this direction
                break;
            }
            
            if escaped { break; }
        }
        
        // If no escape route found, leave the fluid in place.
        // It will try again next frame. This preserves total fluid volume.
        // (Previously this destroyed the fluid, causing volume loss.)
        
        return;
    }

    // Branch: radial mode uses 26-neighbor scoring (like reference demo),
    // directional mode uses the 6-direction sequential processing.
    if params.gravity_mode == 3u {
        radial_move(gid);
        return;
    }

    // Phase 1: Process gravity (primary movement direction)
    let grav_dir = get_effective_gravity(gid);
    let gravity_dir_index = gravity_dir_to_index_noisy(grav_dir, gid);
    
    // Process gravity direction first (highest priority)
    process_direction(gid, gravity_dir_index, this_voxel_supported);
    
    // Process opposite direction for upward prevention
    let opposite_dir = gravity_dir_index ^ 1u; // Flip last bit to get opposite
    process_direction(gid, opposite_dir, this_voxel_supported);
    
    // Phase 2: Accelerated horizontal spreading for faster settling
    let horizontal_order = get_horizontal_direction_order(gid, params.time, gravity_dir_index);
    
    // Process all 4 horizontal directions with increased probability
    for (var i = 0u; i < 4u; i++) {
        process_direction(gid, horizontal_order[i], this_voxel_supported);
    }
}

fn process_direction(gid: vec3<u32>, direction: u32, a_supported: bool) {
    let res = params.grid_resolution;
    let offset = get_offset(direction);
    let nx = i32(gid.x) + offset.x;
    let ny = i32(gid.y) + offset.y;
    let nz = i32(gid.z) + offset.z;

    // Bounds check
    if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
        return;
    }

    let idx_a = grid_index(gid.x, gid.y, gid.z);
    let idx_b = grid_index(u32(nx), u32(ny), u32(nz));

    // Check solid mask - prevent movement from or into solid voxels
    if is_solid(gid.x, gid.y, gid.z) || is_solid(u32(nx), u32(ny), u32(nz)) {
        return;
    }

    // Use simple checkerboard pattern based on grid position to avoid duplicate work
    // This is more predictable than hash-based checkering (temporarily disabled for debugging)
    // let checker_coord = get_checker_coord(gid, direction);
    // if ((checker_coord + params.sub_step) & 1u) == 0u {
    //     return;
    // }

    let state_a = atomicLoad(&voxels[idx_a]);
    let state_b = atomicLoad(&voxels[idx_b]);

    let type_a = get_fluid_type(state_a);
    let type_b = get_fluid_type(state_b);

    // Only consider swaps involving water (1), steam (3), and empty (0)
    let a_is_water = type_a == 1u;
    let b_is_water = type_b == 1u;
    let a_is_steam = type_a == 3u;
    let b_is_steam = type_b == 3u;
    let a_is_empty = type_a == 0u;
    let b_is_empty = type_b == 0u;

    // Skip if both same or neither fluid/empty
    // Now also allow steam-water exchanges
    if !((a_is_water && b_is_empty) || (a_is_empty && b_is_water) ||
          (a_is_steam && b_is_empty) || (a_is_empty && b_is_steam) ||
          (a_is_water && b_is_steam) || (a_is_steam && b_is_water)) {
        return;
    }

    // Check if this direction aligns with gravity (per-voxel for radial mode)
    let grav_dir = get_effective_gravity(gid);
    
    // Simple dot product to check if direction aligns with gravity
    let dir_vec = get_offset(direction);
    var alignment = dot(grav_dir, vec3<f32>(f32(dir_vec.x), f32(dir_vec.y), f32(dir_vec.z)));
    
    // Reverse alignment for steam (steam flows against gravity)
    if a_is_steam || b_is_steam {
        alignment = -alignment;
    }
    
    // If alignment is positive, this direction flows with gravity
    // If alignment is negative, this direction flows against gravity
    // If alignment is zero, this direction is perpendicular to gravity
    
    // For gravity-aligned directions: use magnitude for probability
    if abs(alignment) > 0.1 {
        let gravity_strength = length(grav_dir);
        // Fall rate proportional to gravity magnitude
        // Linear scaling: gravity_magnitude of 50.0 = 100% fall rate
        let gravity_probability = min(1.0, gravity_strength / 50.0);
        
        // Use hash-based probability for gravity direction
        let time_hash = u32(params.time * 1000.0) + direction * 12345u;
        let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
        let combined_hash = time_hash ^ pos_hash;
        
        // Skip movement based on gravity strength (lower gravity = less movement)
        // EXCEPTION: Unsupported water gets guaranteed fall chance to prevent hanging in air
        // (a_supported is pre-computed by the caller to avoid redundant neighbor lookups)
        
        // Always respect gravity probability, even for unsupported water
        // Unsupported water just gets priority in the movement logic, not guaranteed movement
        if (combined_hash & 255u) > u32(gravity_probability * 255.0) {
            return;
        }
        
        // Allow steam-water exchanges regardless of gravity direction
        if (a_is_water && b_is_steam) || (a_is_steam && b_is_water) {
            // Steam-water exchanges can happen in any direction
            // No gravity restrictions for fluid-to-fluid exchanges
        } else {
            // If alignment is positive: fluid flows with gravity (current to neighbor)
            if alignment > 0.0 {
                // Movement with gravity: fluid flows from current cell to neighbor
                if !((a_is_water && b_is_empty) || (a_is_steam && b_is_empty)) {
                    return;
                }
            }
            // If alignment is negative: fluid flows against gravity (neighbor to current)
            else {
                // Movement against gravity: fluid flows from neighbor to current cell
                if !((a_is_empty && b_is_water) || (a_is_empty && b_is_steam)) {
                    return;
                }
            }
        }
    }
    
    // For non-gravity directions: Use configurable probability for lateral spreading
    if abs(alignment) <= 0.5 {
        // Water can only move laterally if it's supported (touching other water or solids)
        // Exception: radial mode - water on the shell surface must flow freely to round out
        if params.gravity_mode != 3u && (a_is_water || b_is_water) {
            // a_supported is pre-computed by the caller; only compute b's support on demand.
            var b_supported = true;
            if b_is_water {
                b_supported = water_is_supported(vec3<u32>(u32(nx), u32(ny), u32(nz)));
            }
            
            if !a_supported || !b_supported {
                return; // Water not supported, no lateral movement
            }
        }
        
        // Use the configurable lateral flow probability
        let time_hash = u32(params.time * 1000.0) + direction * 12345u;
        let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
        let combined_hash = time_hash ^ pos_hash;
        
        // Apply steam dispersion bias for more gas-like behavior
        var fluid_probability = get_lateral_flow_probability(type_a) * get_lateral_flow_probability(type_b);
        
        
        // Preferential steam-water swapping when fluids are stacked vertically
        if (a_is_steam && b_is_water) || (a_is_water && b_is_steam) {
            // Check if this is a vertical direction (up/down)
            let is_vertical = (direction == 2u || direction == 3u); // +Y or -Y
            
            if is_vertical {
                // Boost probability for steam-water exchanges in vertical directions
                // This helps steam rise through water
                fluid_probability = fluid_probability * 2.0; // Double the probability
            }
        }
        
        if a_is_steam || b_is_steam {
            fluid_probability = fluid_probability * get_steam_dispersion_bias(gid, direction);
        }
        
        let probability_threshold = fluid_probability * 255.0;
        if (combined_hash & 255u) > u32(probability_threshold) {
            return;
        }
    }

    // World sphere boundary is handled by the solid mask (same as cave walls),
    // so no separate is_in_bounds check is needed here.

    // Atomic swap using compare-and-exchange
    // First try to claim cell A
    let result_a = atomicCompareExchangeWeak(&voxels[idx_a], state_a, 0xFFFFFFFFu);
    if !result_a.exchanged {
        return; // Someone else modified it
    }
    
    // Try to claim cell B
    let result_b = atomicCompareExchangeWeak(&voxels[idx_b], state_b, 0xFFFFFFFFu);
    if !result_b.exchanged {
        // Restore cell A and abort
        atomicExchange(&voxels[idx_a], state_a);
        return;
    }
    
    // Both cells claimed, perform the swap
    atomicStore(&voxels[idx_a], state_b);
    atomicStore(&voxels[idx_b], state_a);

    // Record water velocity at the destination of the water
    let dir = get_offset(direction);
    if type_a == 1u {
        // Water moved from A to B
        write_water_velocity(idx_b, 1u, dir.x, dir.y, dir.z);
    } else if type_b == 1u {
        // Water moved from B to A (opposite direction)
        write_water_velocity(idx_a, 1u, -dir.x, -dir.y, -dir.z);
    }
}

// Initialize a sphere of water
@compute @workgroup_size(4, 4, 4)
fn fluid_init_sphere(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);

    // Sphere center at (0, world_radius * 0.5, 0), radius = world_radius * 0.45
    let sphere_center = vec3<f32>(0.0, params.world_radius * 0.5, 0.0);
    let sphere_radius = params.world_radius * 0.45;

    let dist = length(world_pos - sphere_center);

    if dist < sphere_radius && is_in_bounds(world_pos) && !is_solid(gid.x, gid.y, gid.z) {
        // Water: type=1, fill=1.0, seeded at the local ambient temperature
        // -> packed as (65535 << 16) | temperature_bits | 1
        let init_temp = pack_temperature_c(ambient_temp_c(world_pos));
        atomicStore(&voxels[idx], (65535u << 16u) | init_temp | 1u);
    } else {
        atomicStore(&voxels[idx], 0u);
    }
}

// Continuous water spawn function
// Spawn zone probes downward from the intended Y to avoid embedding in solids (e.g. cave ceilings)
@compute @workgroup_size(4, 4, 4)
fn fluid_spawn_continuous(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let world_pos = grid_to_world(gid.x, gid.y, gid.z);

    // Spawn zone: horizontal plane at top of world, centered
    let spawn_center = vec3<f32>(0.0, params.world_radius * 0.8, 0.0);
    let spawn_radius = params.world_radius * 0.15;
    let spawn_thickness = params.cell_size * 2.0; // 2-voxel thick spawn plane

    let horizontal_dist = length(world_pos.xz - spawn_center.xz);
    let vertical_dist = abs(world_pos.y - spawn_center.y);

    // Check if position is within the horizontal spawn zone and vertical band
    if horizontal_dist >= spawn_radius || vertical_dist >= spawn_thickness || !is_in_bounds(world_pos) {
        return;
    }

    // Probe downward from this voxel's Y through the entire volume to find
    // the first non-solid, in-bounds position below cave ceilings or other solids
    var spawn_y = gid.y;
    while spawn_y > 0u && is_solid(gid.x, spawn_y, gid.z) {
        spawn_y -= 1u;
    }

    // If the bottom voxel is still solid, no spawnable space exists in this column
    if is_solid(gid.x, spawn_y, gid.z) {
        return;
    }

    // Verify the probed position is still in bounds
    let probed_world_pos = grid_to_world(gid.x, spawn_y, gid.z);
    if !is_in_bounds(probed_world_pos) {
        return;
    }

    let idx = grid_index(gid.x, spawn_y, gid.z);

    // Use time-based animation for pulsing effect
    let pulse = sin(params.time * 2.0) * 0.5 + 0.5; // Oscillates between 0 and 1
    let spawn_probability = 0.3 + pulse * 0.4; // 30% to 70% chance
    
    // Hash-based randomization for natural flow
    let pos_hash = hash_position(gid);
    let time_hash = u32(params.time * 1000.0);
    let combined_hash = pos_hash ^ time_hash;
    
    // Spawn fluid based on probability
    if (combined_hash & 255u) < u32(spawn_probability * 255.0) {
        // Only spawn if current cell is empty
        let current_state = atomicLoad(&voxels[idx]);
        if get_fluid_type(current_state) == 0u {
            // Spawn selected fluid type with full fill, seeded at the local
            // ambient temperature (only meaningful for water/ice; harmless
            // padding for other types since they don't read the field).
            let init_temp = pack_temperature_c(ambient_temp_c(probed_world_pos));
            atomicStore(&voxels[idx], (65535u << 16u) | init_temp | params.spawn_fluid_type);
        }
    }
}

// Clear all fluid voxels
@compute @workgroup_size(4, 4, 4)
fn fluid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    atomicStore(&voxels[idx], 0u);
}
