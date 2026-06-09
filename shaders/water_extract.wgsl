// Compute shader to extract water voxel positions from fluid state
// and write them to the particle instance buffer

struct WaterParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,  // x=time_offset, y=velocity_x, z=velocity_y, w=velocity_z
}

struct ExtractParams {
    grid_resolution: u32,
    cell_size: f32,
    max_particles: u32,
    time: f32,
    grid_origin: vec3<f32>,
    world_radius: f32,  // World boundary radius
    prominence_factor: f32,  // How prominent to make water particles (0.0-1.0)
    gravity_mode: u32,  // 0=X, 1=Y, 2=Z, 3=radial
    _pad0: f32,
    _pad1: f32,
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
@group(0) @binding(5) var<storage, read> water_velocity: array<u32>;

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
            let neighbor_type = neighbor_state & 0x7u;
            
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
    // Create a unique seed for this voxel position
    let seed = hash_u32(x + y * 1009u + z * 1009u * 1009u);
    
    // Generate random offsets within [0, 1) range for each dimension
    let random_x = random_float(seed);
    let random_y = random_float(seed + 1u);
    let random_z = random_float(seed + 2u);
    
    return params.grid_origin + vec3<f32>(
        (f32(x) + random_x) * params.cell_size,
        (f32(y) + random_y) * params.cell_size,
        (f32(z) + random_z) * params.cell_size
    );
}

// Convert 3D grid coords to 1D index
fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

// Decode a packed velocity u32 into a direction vector.
// Encoding: 2 bits per axis. 0b00=0, 0b01=+1, 0b10=-1.
fn decode_velocity_component(v: u32) -> f32 {
    if v == 1u { return 1.0; }
    if v == 2u { return -1.0; }
    return 0.0;
}

fn decode_water_velocity(packed: u32) -> vec3<f32> {
    if packed == 0u { return vec3<f32>(0.0); }
    let dx = decode_velocity_component(packed & 3u);
    let dy = decode_velocity_component((packed >> 2u) & 3u);
    let dz = decode_velocity_component((packed >> 4u) & 3u);
    return vec3<f32>(dx, dy, dz);
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
    let fluid_type = state & 0x7u;

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
    
    // Water particle size - small droplets, clearly visible but not oversized
    let base_size = params.cell_size * 0.4;
    let max_size = params.cell_size * 1.2;
    particle.size = mix(base_size, max_size, params.prominence_factor);

    // Water color: bright blue, opaque enough to read as solid drops
    particle.color = vec4<f32>(0.35, 0.6, 0.95, 0.75);

    // Sample water velocity at this voxel
    let vel = decode_water_velocity(water_velocity[idx]);

    // Animation data: x=time offset for shimmer, yzw=velocity direction for elongation
    particle.animation = vec4<f32>(
        params.time + f32(global_id.x) * 0.05,
        vel.x,
        vel.y,
        vel.z
    );

    particles[particle_idx] = particle;
}
