// Compute shader to extract water voxel positions from fluid state
// and write them to the particle instance buffer

struct WaterParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,
}

struct ExtractParams {
    grid_resolution: u32,
    cell_size: f32,
    max_particles: u32,
    time: f32,
    grid_origin: vec3<f32>,
    world_radius: f32,  // World boundary radius
    prominence_factor: f32,  // How prominent to make water particles (0.0-1.0)
    _padding: f32,
}

// Atomic counter for particle count
struct ParticleCounter {
    count: atomic<u32>,
}

// Fluid state is just an array of u32 fluid types (0=empty, 1=water, 2=lava, 3=steam, 4=solid)
@group(0) @binding(0) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(1) var<storage, read_write> particles: array<WaterParticle>;
@group(0) @binding(2) var<storage, read_write> counter: ParticleCounter;
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(4) var<uniform> params: ExtractParams;

// Check if a voxel is solid
fn is_solid(x: u32, y: u32, z: u32) -> bool {
    let res = params.grid_resolution;
    let idx = x + y * res + z * res * res;
    return solid_mask[idx] == 1u;
}

// Hash function for generating pseudo-random values from cell ID
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

// Generate a random float in [0, 1) from a seed
fn random_float(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967296.0;
}

// Check if a water voxel should be rendered as a particle
// Only render isolated water particles (not touching other water)
// Also exclude particles touching solids or world boundary
fn should_render_as_particle(x: u32, y: u32, z: u32) -> bool {
    let res = params.grid_resolution;
    
    // Check if water voxel is near the spherical world boundary (edge of simulation)
    let world_pos = grid_to_world(x, y, z);
    let distance_from_center = length(world_pos);
    let boundary_threshold = params.world_radius * 0.95; // Near boundary threshold
    let near_boundary = distance_from_center > boundary_threshold;
    
    // Count water neighbors and check for solids contact
    var water_neighbors = 0u;
    var touches_solid = false;
    
    // Check all 6 neighbors
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 0, 1),   // +Z
        vec3<i32>(0, 0, -1)   // -Z
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + offsets[i].x;
        let ny = i32(y) + offsets[i].y;
        let nz = i32(z) + offsets[i].z;
        
        if nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res) {
            let neighbor_idx = u32(nx) + u32(ny) * res + u32(nz) * res * res;
            let neighbor_state = fluid_state[neighbor_idx];
            let neighbor_type = neighbor_state & 0xFFFFu;
            
            if neighbor_type == 1u {  // Water
                water_neighbors++;
            } else if is_solid(u32(nx), u32(ny), u32(nz)) {  // Solid
                touches_solid = true;
            }
        }
    }
    
    // Only render if completely isolated (no water neighbors) 
    // AND not touching solids 
    // AND not near world boundary
    return water_neighbors == 0u && !touches_solid && !near_boundary;
}

// Convert grid coordinates to world position with random offset within voxel bounds
fn grid_to_world(x: u32, y: u32, z: u32) -> vec3<f32> {
    return vec3<f32>(
        params.grid_origin.x + (f32(x) + 0.5) * params.cell_size,
        params.grid_origin.y + (f32(y) + 0.5) * params.cell_size,
        params.grid_origin.z + (f32(z) + 0.5) * params.cell_size,
    );
}

// Convert 3D grid coords to 1D index
fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = params.grid_resolution;

    // Check bounds
    if global_id.x >= res || global_id.y >= res || global_id.z >= res {
        return;
    }

    let idx = grid_index(global_id.x, global_id.y, global_id.z);
    let state = fluid_state[idx];

    // Fluid type is stored in lower 16 bits
    let fluid_type = state & 0xFFFFu;

    // Check if this is a water voxel (fluid_type == 1)
    if fluid_type != 1u {
        return;
    }
    
    // Check if this water voxel should be rendered as a particle
    if !should_render_as_particle(global_id.x, global_id.y, global_id.z) {
        return;
    }

    // Atomically get a particle slot
    let particle_idx = atomicAdd(&counter.count, 1u);

    // Check if we have room
    if particle_idx >= params.max_particles {
        return;
    }

    // Calculate world position
    let world_pos = grid_to_world(global_id.x, global_id.y, global_id.z);

    // Create particle
    var particle: WaterParticle;
    particle.position = world_pos;
    
    // Water particles size based on prominence factor
    let base_size = params.cell_size * 0.4;  // Base size for smallest particles
    let max_size = params.cell_size * 1.2;   // Maximum size for prominent particles
    particle.size = mix(base_size, max_size, params.prominence_factor);

    // Water color: blue with consistent transparency
    particle.color = vec4<f32>(0.2, 0.5, 0.8, 0.3);

    // Animation data (time offset based on position for variation)
    particle.animation = vec4<f32>(
        params.time + f32(global_id.x) * 0.05,
        params.prominence_factor,
        f32(global_id.y) * 0.05,
        f32(global_id.z) * 0.05
    );

    particles[particle_idx] = particle;
}
