// GPU Fluid Simulation - Pair-based swapping
// 6 directional passes (±X, ±Y, ±Z), each with 2 checkered phases
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
    
    // Phase change probabilities (0.0 to 1.0)
    condensation_probability: f32,  // Steam to Water
    vaporization_probability: f32,  // Water to Steam
    
    // Fluid type for spawning (0=Empty, 1=Water, 2=Lava, 3=Steam)
    spawn_fluid_type: u32,

    sub_step: u32,

    // Gravity mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial (toward origin)
    // Radial: world sphere boundary is the effective shell. gravity_magnitude controls strength.
    gravity_mode: u32,
    _pad_rg0: u32,
    _pad_rg1: u32,
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

fn get_fluid_type(state: u32) -> u32 {
    return state & 0xFFFFu;
}

// Check if a grid position is solid based on the solid mask
fn is_solid(x: u32, y: u32, z: u32) -> bool {
    let idx = grid_index(x, y, z);
    return solid_mask[idx] == 1u;
}

// Check if a voxel is encapsulated (surrounded on all 6 sides by solids or water)
// If true, this voxel can be skipped during processing as it cannot move
fn is_encapsulated(x: u32, y: u32, z: u32) -> bool {
    // Never skip fluid inside solid — it needs the push-out path
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
    let threshold = params.world_radius * 0.98;
    return dot(pos, pos) < threshold * threshold;
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
    
    // Weighted random selection: probability of picking each axis ∝ its component magnitude
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

// Condensation mechanic - steam can condense back to water when contacting solids or boundaries
fn should_condense_steam(gid: vec3<u32>) -> bool {
    let res = params.grid_resolution;
    
    // Check if steam voxel is near the spherical world boundary (edge of simulation)
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let distance_from_center = length(world_pos);
    let boundary_threshold = params.world_radius * 0.95; // Near boundary threshold
    let near_boundary = distance_from_center > boundary_threshold;
    
    // Check if steam voxel is adjacent to solid surfaces
    let solid_neighbors = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1)   // -Z
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
    
    // Condense if near boundary or adjacent to solids
    if !near_boundary && adjacent_solids == 0u {
        return false;
    }
    
    // Use condensation probability from uniform parameters
    let condensation_probability = params.condensation_probability;
    
    // Use hash-based randomization for natural variation
    let pos_hash = hash_position(gid);
    let time_hash = u32(params.time * 1000.0);
    let combined_hash = pos_hash ^ time_hash;
    
    return (combined_hash & 255u) < u32(condensation_probability * 255.0);
}

// Water-to-steam conversion - water converts to steam when touching hot red cave rocks in bottom quarter
fn should_convert_to_steam(gid: vec3<u32>) -> bool {
    let res = params.grid_resolution;
    
    // Check if water voxel is adjacent to solid surfaces that are in the red layer
    let solid_neighbors = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1)   // -Z
    );
    
    var touching_red_rocks = false;
    var adjacent_solids = 0u;
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(gid.x) + solid_neighbors[i].x;
        let ny = i32(gid.y) + solid_neighbors[i].y;
        let nz = i32(gid.z) + solid_neighbors[i].z;
        
        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            if is_solid(u32(nx), u32(ny), u32(nz)) {
                adjacent_solids++;
                
                // Check if this solid is in the red layer AND bottom quarter
                let solid_world_pos = grid_to_world(u32(nx), u32(ny), u32(nz));
                let solid_distance_from_center = length(solid_world_pos);
                let red_layer_start = params.world_radius * 0.9;
                
                // Check if in bottom quarter (simplified - just check Y position)
                let is_in_outer_layer = solid_distance_from_center >= red_layer_start && solid_distance_from_center <= params.world_radius;
                
                // Simple bottom quarter check: Y must be negative enough (bottom 25%)
                let y_normalized = solid_world_pos.y / params.world_radius;
                let is_bottom_quarter = y_normalized < -0.5; // Bottom quarter
                
                if is_in_outer_layer && is_bottom_quarter {
                    touching_red_rocks = true;
                }
            }
        }
    }
    
    // Convert only if touching red cave rocks in bottom quarter
    if !touching_red_rocks {
        return false;
    }
    
    // Conversion probability based on how many red rocks are touching
    // Use vaporization probability from uniform parameters
    let conversion_probability = params.vaporization_probability;
    
    // Use hash-based randomization for natural variation
    let pos_hash = hash_position(gid);
    let time_hash = u32(params.time * 1000.0);
    let combined_hash = pos_hash ^ time_hash;
    
    return (combined_hash & 255u) < u32(conversion_probability * 255.0);
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
    
    // Water or lava below = supported (resting on fluid)
    let neighbor_idx = u32(nx) + u32(ny) * res + u32(nz) * res * res;
    let neighbor_state = atomicLoad(&voxels[neighbor_idx]);
    let neighbor_type = get_fluid_type(neighbor_state);
    
    return neighbor_type >= 1u && neighbor_type <= 2u;
}

// Check if a voxel is at or very close to the sphere boundary
fn is_at_sphere_boundary(gid: vec3<u32>) -> bool {
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    let distance_from_center = length(world_pos);
    let boundary_threshold = params.world_radius * 0.95; // 95% of world radius
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

                // Tangential component = dir - (dir·radial)*radial
                let tangential = dir_n - dot(dir_n, radial) * radial;
                tangential_sum += tangential;
            }
        }
    }

    // Net tangential force pulls toward the denser side;
    // surface pressure slides voxel away from dense neighbors → negate
    return -tangential_sum * 0.15;
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

    // Probability gate uses only gravity magnitude so surface force always has effect
    let gravity_probability = min(1.0, grav_mag * grav_mag * 0.0004);
    let prob_hash = (gid.x * 7u + gid.y * 13u + gid.z * 17u + u32(params.time * 1000.0)) & 255u;
    if prob_hash > u32(gravity_probability * 255.0) { return; }

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

    // Score all 26 neighbors, pick the best empty target
    var best_score = 0.05; // minimum threshold to bother moving
    var best_dx = 0;
    var best_dy = 0;
    var best_dz = 0;
    var found = false;

    // Noise seed for tie-breaking
    let noise_seed = gid.x * 73u + gid.y * 157u + gid.z * 239u + u32(params.time * 1000.0);

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

                // World bounds
                if !is_in_bounds(grid_to_world(u32(nx), u32(ny), u32(nz))) { continue; }

                // Alignment score: dot(neighbor_direction, force_direction)
                let dir = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let dir_len = length(dir);
                var score = dot(dir, force_dir) / dir_len;

                // Tie-breaking noise (±0.04)
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
        // Target no longer empty — restore source
        atomicExchange(&voxels[idx], state);
        return;
    }

    let claim_dst = atomicCompareExchangeWeak(&voxels[target_idx], target_state, 0xFFFFFFFFu);
    if !claim_dst.exchanged {
        atomicExchange(&voxels[idx], state);
        return;
    }

    // Both claimed — swap
    atomicStore(&voxels[idx], target_state);
    atomicStore(&voxels[target_idx], state);
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
    
    // Steam condensation check - convert steam back to water when contacting solids
    if fluid_type == 3u && should_condense_steam(gid) {
        // Try to condense steam back to water
        let result = atomicCompareExchangeWeak(&voxels[idx], state, (65535u << 16u) | 1u);
        if result.exchanged {
            return; // Successfully condensed, no further processing needed
        }
    }

    // Water-to-steam conversion check - convert water to steam when touching hot surfaces near boundary
    if fluid_type == 1u && should_convert_to_steam(gid) {
        // Try to convert water to steam
        let result = atomicCompareExchangeWeak(&voxels[idx], state, (65535u << 16u) | 3u);
        if result.exchanged {
            return; // Successfully converted to steam, no further processing needed
        }
    }

    // Steam teleportation - find topmost water and swap with it (directional mode only)
    if fluid_type == 3u && params.gravity_mode != 3u {
        // Scan upward to find the topmost water voxel above this steam
        var found_water_above = false;
        var water_top_y = gid.y;
        
        for (var scan_y = gid.y + 1u; scan_y < params.grid_resolution; scan_y++) {
            let scan_idx = grid_index(gid.x, scan_y, gid.z);
            let scan_state = atomicLoad(&voxels[scan_idx]);
            let scan_fluid_type = get_fluid_type(scan_state);
            
            if scan_fluid_type == 1u {
                // Found water - update the top position
                found_water_above = true;
                water_top_y = scan_y;
            } else if scan_fluid_type != 3u {
                // Hit something other than steam (empty or solid), stop scanning
                break;
            }
        }
        
        // If we found water above, swap with the topmost water
        if found_water_above {
            let water_idx = grid_index(gid.x, water_top_y, gid.z);
            let water_state = atomicLoad(&voxels[water_idx]);
            
            // Use two-phase CAS swap to exchange positions
            // First claim the water position
            let water_result = atomicCompareExchangeWeak(&voxels[water_idx], water_state, 0xFFFFFFFFu);
            if water_result.exchanged {
                // Water position claimed, now claim steam position
                let steam_result = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
                if steam_result.exchanged {
                    // Both positions claimed, perform the swap
                    atomicStore(&voxels[water_idx], state);        // Move steam to water position
                    atomicStore(&voxels[idx], water_state);        // Move water to steam position
                    
                    // Check if the swapped water is now unsupported (clinging to cave surface)
                    let water_gid = vec3<u32>(gid.x, gid.y, gid.z); // Water is now at steam's old position
                    if !water_is_supported(water_gid) {
                        // Water is unsupported - trigger fast-drop immediately
                        let grav_dir = get_effective_gravity(water_gid);
                        let gravity_dir_index = gravity_dir_to_index(grav_dir);
                        
                        let down = get_offset(gravity_dir_index);
                        
                        // Scan downward to find the lowest empty cell above a surface
                        var target_y = -1;
                        for (var step = 1; step <= 64; step++) {
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
                                // Steam - skip through it, keep scanning
                                continue;
                            } else if scan_type == 0u {
                                // Empty cell - this is the target (don't skip through empty space)
                                let world_pos = grid_to_world(u32(sx), u32(sy), u32(sz));
                                if is_in_bounds(world_pos) {
                                    target_y = step;
                                    break; // Found empty space, stop scanning
                                }
                            } else if scan_type == 1u || scan_type == 2u {
                                break; // Hit water/lava surface, target is one step above
                            }
                        }
                        
                        // If we found a position lower than current, teleport the water there
                        if target_y > 1 {
                            let tx = i32(gid.x) + down.x * target_y;
                            let ty = i32(gid.y) + down.y * target_y;
                            let tz = i32(gid.z) + down.z * target_y;
                            let target_idx = grid_index(u32(tx), u32(ty), u32(tz));
                            
                            let target_state = atomicLoad(&voxels[target_idx]);
                            if get_fluid_type(target_state) == 0u {
                                let result = atomicCompareExchangeWeak(&voxels[target_idx], target_state, water_state);
                                if result.exchanged {
                                    // We own idx (placed water_state there on line above),
                                    // clear it directly — no CAS needed since we control it
                                    atomicStore(&voxels[idx], 0u);
                                }
                            }
                        }
                    }
                    
                    return; // Steam-water swap complete
                } else {
                    // Steam position was modified, restore water and abort
                    atomicStore(&voxels[water_idx], water_state);
                }
            }
            // CAS failed — positions were modified, fall through to normal movement
        }
    }

    // Water fast-drop: unsupported water falls instantly to nearest support
    // This bypasses checker/probability gates in process_direction
    // Skip in radial mode — radial_move() handles all movement there.
    if fluid_type == 1u && params.gravity_mode != 3u && !water_is_supported(gid) {
        let grav_dir = get_effective_gravity(gid);
        let gravity_dir_index = gravity_dir_to_index(grav_dir);
        
        let down = get_offset(gravity_dir_index);
        
        // Scan downward to find the lowest empty cell above a surface
        var target_y = -1;
        for (var step = 1; step <= 64; step++) {
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
                // Steam - skip through it, keep scanning
                continue;
            } else if scan_type == 0u {
                // Empty cell - this is the target (don't skip through empty space)
                let world_pos = grid_to_world(u32(sx), u32(sy), u32(sz));
                if is_in_bounds(world_pos) {
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
                    // Found empty non-solid cell — teleport fluid there
                    let result = atomicCompareExchangeWeak(&voxels[target_idx], target_state, state);
                    if result.exchanged {
                        atomicStore(&voxels[idx], 0u);
                        escaped = true;
                    }
                    break;
                }
                
                // Hit another fluid — can't place here, stop this direction
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
    process_direction(gid, gravity_dir_index);
    
    // Process opposite direction for upward prevention
    let opposite_dir = gravity_dir_index ^ 1u; // Flip last bit to get opposite
    process_direction(gid, opposite_dir);
    
    // Phase 2: Accelerated horizontal spreading for faster settling
    let horizontal_order = get_horizontal_direction_order(gid, params.time, gravity_dir_index);
    
    // Process all 4 horizontal directions with increased probability
    for (var i = 0u; i < 4u; i++) {
        process_direction(gid, horizontal_order[i]);
    }
}

fn process_direction(gid: vec3<u32>, direction: u32) {
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
        // Use quadratic scaling for more gradual low-gravity behavior
        // gravity_magnitude of 1.0 = 0.5% fall rate, 9.8 = 4.8% fall rate, 50.0 = 100% fall rate
        let gravity_probability = min(1.0, gravity_strength * gravity_strength * 0.0004);
        
        // Use hash-based probability for gravity direction
        let time_hash = u32(params.time * 1000.0) + direction * 12345u;
        let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
        let combined_hash = time_hash ^ pos_hash;
        
        // Skip movement based on gravity strength (lower gravity = less movement)
        // EXCEPTION: Unsupported water gets guaranteed fall chance to prevent hanging in air
        var is_unsupported_water = false;
        if (a_is_water && !water_is_supported(gid)) {
            is_unsupported_water = true;
        }
        if (b_is_water && !water_is_supported(vec3<u32>(u32(nx), u32(ny), u32(nz)))) {
            is_unsupported_water = true;
        }
        
        // Only skip if not unsupported water
        if !is_unsupported_water && (combined_hash & 255u) > u32(gravity_probability * 255.0) {
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
        // Exception: water at sphere boundary can slide off like cave walls
        // Exception: radial mode — water on the shell surface must flow freely to round out
        if params.gravity_mode != 3u && (a_is_water || b_is_water) {
            var a_supported = true;
            var b_supported = true;
            
            if a_is_water {
                a_supported = water_is_supported(gid);
                // If water is at sphere boundary and not supported, allow it to slide off
                if !a_supported && is_at_sphere_boundary(gid) {
                    a_supported = true; // Allow lateral movement away from boundary
                }
            }
            
            if b_is_water {
                b_supported = water_is_supported(vec3<u32>(u32(nx), u32(ny), u32(nz)));
                // If water is at sphere boundary and not supported, allow it to slide off
                if !b_supported && is_at_sphere_boundary(vec3<u32>(u32(nx), u32(ny), u32(nz))) {
                    b_supported = true; // Allow lateral movement away from boundary
                }
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

    // Check world boundaries for both cells
    let world_a = grid_to_world(gid.x, gid.y, gid.z);
    let world_b = grid_to_world(u32(nx), u32(ny), u32(nz));
    if !is_in_bounds(world_a) || !is_in_bounds(world_b) {
        return;
    }

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
        // Water: type=1, fill=1.0 -> packed as (65535 << 16) | 1
        atomicStore(&voxels[idx], (65535u << 16u) | 1u);
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
            // Spawn selected fluid type with full fill
            atomicStore(&voxels[idx], (65535u << 16u) | params.spawn_fluid_type);
        }

        let idx = gid.x + gid.y * res + gid.z * res * res;
        let state = atomicLoad(&voxels[idx]);

        let fluid_type = state & 0xFFFFu;
        let fill = f32((state >> 16u) & 0xFFFFu) / 65535.0;

        if (fluid_type == 1u || fluid_type == 2u) && fill > 0.0 {
            // This was trying to write to density_out, but that's not available in this shader
            // We'll just spawn the fluid and return
        }

        // Spawn selected fluid type with full fill
        atomicStore(&voxels[idx], (65535u << 16u) | params.spawn_fluid_type);
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
