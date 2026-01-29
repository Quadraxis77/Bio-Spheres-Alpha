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
    
    // Fluid type for spawning (0=Empty, 1=Water, 2=Lava, 3=Steam)
    spawn_fluid_type: u32,

    _pad0: u32,
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
        
        // If any neighbor is empty (0), this voxel can potentially move
        if neighbor_type == 0u {
            return false;
        }
    }
    
    // All 6 neighbors are either water (1) or solids (2+), so this voxel is encapsulated
    return true;
}

fn count_surrounding_water(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    var count = 0u;
    
    // Check 6 neighbors (not up/down, just horizontal)
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1),  // -Z
        vec3<i32>(0, 1, 0),   // +Y (same level)
        vec3<i32>(0, -1, 0)   // -Y (below)
    );
    
    for (var i = 0; i < 6; i++) {
        let offset = offsets[i];
        let nx = i32(x) + offset.x;
        let ny = i32(y) + offset.y;
        let nz = i32(z) + offset.z;
        
        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            let neighbor_idx = grid_index(u32(nx), u32(ny), u32(nz));
            let neighbor_state = atomicLoad(&voxels[neighbor_idx]);
            if get_fluid_type(neighbor_state) == 1u {
                count++;
            }
        }
    }
    
    return count;
}

fn grid_to_world(x: u32, y: u32, z: u32) -> vec3<f32> {
    return vec3<f32>(
        params.grid_origin_x + (f32(x) + 0.5) * params.cell_size,
        params.grid_origin_y + (f32(y) + 0.5) * params.cell_size,
        params.grid_origin_z + (f32(z) + 0.5) * params.cell_size
    );
}

fn is_in_bounds(pos: vec3<f32>) -> bool {
    return length(pos) < params.world_radius * 0.98; // Reduced dead zone near boundaries
}

// Perlin noise implementation
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(t: f32, a: f32, b: f32) -> f32 {
    return a + t * (b - a);
}

fn grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    var u: f32;
    var v: f32;
    
    if h < 8u {
        u = x;
    } else {
        u = y;
    }
    
    if h < 4u {
        v = y;
    } else if h == 12u || h == 14u {
        v = x;
    } else {
        v = z;
    }
    
    var result: f32;
    if (h & 1u) == 0u {
        result = u;
    } else {
        result = -u;
    }
    
    if (h & 2u) == 0u {
        result = result + v;
    } else {
        result = result - v;
    }
    
    return result;
}

fn perlin_noise(pos: vec3<f32>) -> f32 {
    // Convert to integer lattice points
    let X = u32(floor(pos.x)) & 255u;
    let Y = u32(floor(pos.y)) & 255u;
    let Z = u32(floor(pos.z)) & 255u;
    
    // Relative position within lattice cell
    let x = pos.x - floor(pos.x);
    let y = pos.y - floor(pos.y);
    let z = pos.z - floor(pos.z);
    
    // Fade curves
    let u = fade(x);
    let v = fade(y);
    let w = fade(z);
    
    // Hash coordinates for the 8 corners of the cube
    let A = u32(X) + 1u;
    let B = u32(Y) + 1u;
    let C = u32(Z) + 1u;
    
    let h000 = (u32(X) + u32(Y) + u32(Z)) * 73856093u ^ 19349663u;
    let h001 = (u32(X) + u32(Y) + u32(C)) * 73856093u ^ 83492791u;
    let h010 = (u32(X) + u32(B) + u32(Z)) * 73856093u ^ 12673291u;
    let h011 = (u32(X) + u32(B) + u32(C)) * 73856093u ^ 52328767u;
    let h100 = (u32(A) + u32(Y) + u32(Z)) * 73856093u ^ 29537387u;
    let h101 = (u32(A) + u32(Y) + u32(C)) * 73856093u ^ 97284321u;
    let h110 = (u32(A) + u32(B) + u32(Z)) * 73856093u ^ 41283947u;
    let h111 = (u32(A) + u32(B) + u32(C)) * 73856093u ^ 62348719u;
    
    // Interpolate
    let n000 = grad(h000, x, y, z);
    let n001 = grad(h001, x, y, z - 1.0);
    let n010 = grad(h010, x, y - 1.0, z);
    let n011 = grad(h011, x, y - 1.0, z - 1.0);
    let n100 = grad(h100, x - 1.0, y, z);
    let n101 = grad(h101, x - 1.0, y, z - 1.0);
    let n110 = grad(h110, x - 1.0, y - 1.0, z);
    let n111 = grad(h111, x - 1.0, y - 1.0, z - 1.0);
    
    let n00 = lerp(u, n000, n100);
    let n01 = lerp(u, n001, n101);
    let n10 = lerp(u, n010, n110);
    let n11 = lerp(u, n011, n111);
    
    let n0 = lerp(v, n00, n10);
    let n1 = lerp(v, n01, n11);
    
    return lerp(w, n0, n1);
}

// Get wave-based flow direction for smooth water surface movement
fn get_wave_flow_direction(gid: vec3<u32>, time: f32) -> f32 {
    // Use world position for wave patterns
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);
    
    // Create traveling waves in XZ plane
    let wave_speed = 0.5;
    let wave_frequency = 0.1;
    
    // Primary wave in X direction
    let wave_x = sin(world_pos.x * wave_frequency + time * wave_speed);
    
    // Secondary wave in Z direction (90 degrees out of phase)
    let wave_z = cos(world_pos.z * wave_frequency + time * wave_speed * 0.7);
    
    // Combine waves to create circular/elliptical flow patterns
    let combined_wave = (wave_x + wave_z) * 0.5;
    
    // Normalize to [0, 1] for direction selection
    return (combined_wave + 1.0) * 0.5;
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
    let direction_factor = f32(direction + 1u) * 0.3;
    let dispersion_factor = sin(f32(pos_hash) * 0.001 + params.time * 2.0) * 0.3 + 0.7;
    
    // Steam has higher dispersion in horizontal directions (spreads out)
    if direction == 0u || direction == 1u || direction == 4u || direction == 5u {
        return dispersion_factor * 1.5; // Boost horizontal spreading
    } else {
        return dispersion_factor * 0.8; // Reduce vertical movement bias
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
    
    // Higher condensation probability near boundaries (simulating cooling)
    var condensation_probability = 0.004 + f32(adjacent_solids) * 0.002; // 0.4% to 0.6% for solids
    if near_boundary {
        condensation_probability = condensation_probability; // Same 0.4% to 0.6% for boundaries
    }
    
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
    let solid_factor = min(f32(adjacent_solids) / 6.0, 1.0); // 0 to 1, more solids = higher chance
    let conversion_probability = 0.05 + solid_factor * 0.1; // Increased to 5% to 15% for testing
    
    // Use hash-based randomization for natural variation
    let pos_hash = hash_position(gid);
    let time_hash = u32(params.time * 1000.0);
    let combined_hash = pos_hash ^ time_hash;
    
    return (combined_hash & 255u) < u32(conversion_probability * 255.0);
}

// Check if water voxel is in contact with other water or solid surfaces
fn water_is_supported(gid: vec3<u32>) -> bool {
    let res = params.grid_resolution;
    
    // Check all 6 neighbors for water or solid contact
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1)   // -Z
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(gid.x) + offsets[i].x;
        let ny = i32(gid.y) + offsets[i].y;
        let nz = i32(gid.z) + offsets[i].z;
        
        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            let neighbor_idx = u32(nx) + u32(ny) * res + u32(nz) * res * res;
            let neighbor_state = atomicLoad(&voxels[neighbor_idx]);
            let neighbor_type = get_fluid_type(neighbor_state);
            
            // If neighbor is water or solid, water is supported
            if neighbor_type == 1u || neighbor_type >= 2u {
                return true;
            }
        }
    }
    
    return false;
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

    // Special case: Push water out of solid voxels
    // If water is trapped in a solid voxel, force it to move to the nearest non-solid voxel
    if fluid_type == 1u && is_solid(gid.x, gid.y, gid.z) {
        // Try all 6 directions to find an escape route
        let directions = array<vec3<i32>, 6>(
            vec3<i32>(1, 0, 0),   // +X
            vec3<i32>(-1, 0, 0),  // -X
            vec3<i32>(0, 1, 0),   // +Y
            vec3<i32>(0, -1, 0),  // -Y
            vec3<i32>(0, 0, 1),   // +Z
            vec3<i32>(0, 0, -1)   // -Z
        );
        
        // Check each direction for an escape
        for (var i = 0u; i < 6u; i++) {
            let nx = i32(gid.x) + directions[i].x;
            let ny = i32(gid.y) + directions[i].y;
            let nz = i32(gid.z) + directions[i].z;
            
            // Bounds check
            if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
                let neighbor_idx = grid_index(u32(nx), u32(ny), u32(nz));
                let neighbor_state = atomicLoad(&voxels[neighbor_idx]);
                let neighbor_type = get_fluid_type(neighbor_state);
                
                // If neighbor is empty and not solid, move water there
                if neighbor_type == 0u && !is_solid(u32(nx), u32(ny), u32(nz)) {
                    // Try to claim both voxels atomically
                    let result_current = atomicCompareExchangeWeak(&voxels[idx], state, 0xFFFFFFFFu);
                    if result_current.exchanged {
                        let result_neighbor = atomicCompareExchangeWeak(&voxels[neighbor_idx], neighbor_state, state);
                        if result_neighbor.exchanged {
                            // Successfully moved water
                            atomicStore(&voxels[idx], neighbor_state);
                            return;
                        } else {
                            // Failed to claim neighbor, restore current
                            atomicStore(&voxels[idx], state);
                        }
                    }
                }
            }
        }
        
        // If no escape route found, stay in place (don't process further)
        return;
    }

    // Phase 1: Process gravity (primary movement direction)
    // Use gravity direction from parameters instead of hardcoded -Y
    let grav_dir = vec3<f32>(params.gravity_dir_x, params.gravity_dir_y, params.gravity_dir_z);
    
    // Find the primary gravity axis and direction
    var gravity_dir_index = 3u; // Default to -Y
    if abs(grav_dir.x) > abs(grav_dir.y) && abs(grav_dir.x) > abs(grav_dir.z) {
        gravity_dir_index = select(1u, 0u, grav_dir.x > 0.0); // -X or +X
    } else if abs(grav_dir.y) > abs(grav_dir.z) {
        gravity_dir_index = select(3u, 2u, grav_dir.y > 0.0); // -Y or +Y
    } else if abs(grav_dir.z) > 0.0 {
        gravity_dir_index = select(5u, 4u, grav_dir.z > 0.0); // -Z or +Z
    }
    
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

// Check if vertical swap is possible from this position
fn can_swap_vertical(gid: vec3<u32>, direction: u32) -> bool {
    let res = params.grid_resolution;
    let offset = get_offset(direction);
    let nx = i32(gid.x) + offset.x;
    let ny = i32(gid.y) + offset.y;
    let nz = i32(gid.z) + offset.z;

    // Bounds check
    if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
        return false;
    }

    // Check solid mask - prevent movement from or into solid voxels
    if is_solid(gid.x, gid.y, gid.z) || is_solid(u32(nx), u32(ny), u32(nz)) {
        return false;
    }

    let idx_a = grid_index(gid.x, gid.y, gid.z);
    let idx_b = grid_index(u32(nx), u32(ny), u32(nz));

    let state_a = atomicLoad(&voxels[idx_a]);
    let state_b = atomicLoad(&voxels[idx_b]);

    let type_a = get_fluid_type(state_a);
    let type_b = get_fluid_type(state_b);

    // Check if this is a valid water-empty vertical pair
    if direction == 3u {
        // -Y: water above, empty below
        return type_a == 1u && type_b == 0u;
    } else if direction == 2u {
        // +Y: empty below, water above
        return type_a == 0u && type_b == 1u;
    }
    
    return false;
}

// Fast horizontal processing for accelerated settling
fn process_direction_fast(gid: vec3<u32>, direction: u32) {
    let res = params.grid_resolution;
    let offset = get_offset(direction);
    let nx = i32(gid.x) + offset.x;
    let ny = i32(gid.y) + offset.y;
    let nz = i32(gid.z) + offset.z;

    // Bounds check
    if nx < 0 || nx >= i32(res) || ny < 0 || ny >= i32(res) || nz < 0 || nz >= i32(res) {
        return;
    }

    // Check solid mask - prevent movement from or into solid voxels
    if is_solid(gid.x, gid.y, gid.z) || is_solid(u32(nx), u32(ny), u32(nz)) {
        return;
    }

    let idx_a = grid_index(gid.x, gid.y, gid.z);
    let idx_b = grid_index(u32(nx), u32(ny), u32(nz));

    // Use checkerboard pattern to avoid duplicate work
    let checker_coord = get_checker_coord(gid, direction);
    if (checker_coord & 1u) == 0u {
        return;
    }

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

    // For horizontal directions: Use configurable probability for lateral movement
    if direction == 0u || direction == 1u || direction == 4u || direction == 5u {
        // Water can only move laterally if it's supported (touching other water or solids)
        if a_is_water || b_is_water {
            if a_is_water && !water_is_supported(gid) {
                return; // Water not supported, no lateral movement
            }
            if b_is_water && !water_is_supported(vec3<u32>(u32(nx), u32(ny), u32(nz))) {
                return; // Water not supported, no lateral movement
            }
        }
        
        // Use the configurable lateral flow probability
        let time_hash = u32(params.time * 1000.0) + direction * 12345u;
        let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
        let combined_hash = time_hash ^ pos_hash;
        
        // Apply steam dispersion bias for more gas-like behavior
        var fluid_probability = get_lateral_flow_probability(type_a) * get_lateral_flow_probability(type_b);
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
    let result_a = atomicCompareExchangeWeak(&voxels[idx_a], state_a, 0xFFFFFFFFu);
    if !result_a.exchanged {
        return;
    }
    
    let result_b = atomicCompareExchangeWeak(&voxels[idx_b], state_b, 0xFFFFFFFFu);
    if !result_b.exchanged {
        atomicExchange(&voxels[idx_a], state_a);
        return;
    }
    
    // Both cells claimed, perform the swap
    atomicStore(&voxels[idx_a], state_b);
    atomicStore(&voxels[idx_b], state_a);
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
    // This is more predictable than hash-based checkering
    let checker_coord = get_checker_coord(gid, direction);
    if (checker_coord & 1u) == 0u {
        return;
    }

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

    // Check if this direction aligns with gravity
    let grav_dir = vec3<f32>(params.gravity_dir_x, params.gravity_dir_y, params.gravity_dir_z);
    
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
        // Higher gravity magnitude = higher probability of movement
        // Halved fall rates for both water and steam
        let gravity_probability = min(1.0, (gravity_strength / 10.0) * 0.5);
        
        // Use hash-based probability for gravity direction
        let time_hash = u32(params.time * 1000.0) + direction * 12345u;
        let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
        let combined_hash = time_hash ^ pos_hash;
        
        // Skip movement based on gravity strength (lower gravity = less movement)
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
        if a_is_water || b_is_water {
            if a_is_water && !water_is_supported(gid) {
                return; // Water not supported, no lateral movement
            }
            if b_is_water && !water_is_supported(vec3<u32>(u32(nx), u32(ny), u32(nz))) {
                return; // Water not supported, no lateral movement
            }
        }
        
        // Use the configurable lateral flow probability
        let time_hash = u32(params.time * 1000.0) + direction * 12345u;
        let pos_hash = gid.x * 7u + gid.y * 13u + gid.z * 17u;
        let combined_hash = time_hash ^ pos_hash;
        
        // Apply steam dispersion bias for more gas-like behavior
        var fluid_probability = get_lateral_flow_probability(type_a) * get_lateral_flow_probability(type_b);
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
@compute @workgroup_size(4, 4, 4)
fn fluid_spawn_continuous(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    let world_pos = grid_to_world(gid.x, gid.y, gid.z);

    // Spawn zone: horizontal plane at top of world, centered
    let spawn_center = vec3<f32>(0.0, params.world_radius * 0.8, 0.0);
    let spawn_radius = params.world_radius * 0.15;
    let spawn_thickness = params.cell_size * 2.0; // 2-voxel thick spawn plane

    let horizontal_dist = length(world_pos.xz - spawn_center.xz);
    let vertical_dist = abs(world_pos.y - spawn_center.y);

    // Check if position is within spawn zone (circular area)
    if horizontal_dist < spawn_radius && vertical_dist < spawn_thickness && is_in_bounds(world_pos) && !is_solid(gid.x, gid.y, gid.z) {
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
        }
    }
}

// Clear all fluid
@compute @workgroup_size(4, 4, 4)
fn fluid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    atomicStore(&voxels[idx], 0u);
}

// === Density extraction for Surface Nets ===

struct ExtractParams {
    grid_resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> extract_params: ExtractParams;
@group(0) @binding(1) var<storage, read> fluid_state: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> density_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> fluid_type_out: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn extract_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = extract_params.grid_resolution;

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = gid.x + gid.y * res + gid.z * res * res;
    let state = atomicLoad(&fluid_state[idx]);

    let fluid_type = state & 0xFFFFu;
    let fill = f32((state >> 16u) & 0xFFFFu) / 65535.0;

    if (fluid_type == 1u || fluid_type == 2u) && fill > 0.0 {
        density_out[idx] = fill;
    } else {
        density_out[idx] = 0.0;
    }

    fluid_type_out[idx] = fluid_type;
}
