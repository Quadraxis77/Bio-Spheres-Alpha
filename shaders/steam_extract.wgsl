// Compute shader to extract steam voxel positions from fluid state
// and write them to the particle instance buffer

struct SteamParticle {
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
    _padding: f32,
}

// Atomic counter for particle count
struct ParticleCounter {
    count: atomic<u32>,
}

// Fluid state is just an array of u32 fluid types (0=empty, 1=water, 2=lava, 3=steam, 4=solid)
@group(0) @binding(0) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(1) var<storage, read_write> particles: array<SteamParticle>;
@group(0) @binding(2) var<storage, read_write> counter: ParticleCounter;
@group(0) @binding(3) var<uniform> params: ExtractParams;

// Convert grid coordinates to world position
fn grid_to_world(x: u32, y: u32, z: u32) -> vec3<f32> {
    return params.grid_origin + vec3<f32>(
        (f32(x) + 0.5) * params.cell_size,
        (f32(y) + 0.5) * params.cell_size,
        (f32(z) + 0.5) * params.cell_size
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

    // Check if this is a steam voxel (fluid_type == 3)
    if fluid_type != 3u {
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
    var particle: SteamParticle;
    particle.position = world_pos;
    particle.size = params.cell_size * 1.5;  // Slightly larger than voxel

    // Steam color: white/grey, semi-transparent
    particle.color = vec4<f32>(0.9, 0.9, 0.95, 0.6);

    // Animation data (time offset based on position for variation)
    particle.animation = vec4<f32>(
        params.time + f32(global_id.x) * 0.1,
        1.0,
        f32(global_id.y) * 0.1,
        f32(global_id.z) * 0.1
    );

    particles[particle_idx] = particle;
}
