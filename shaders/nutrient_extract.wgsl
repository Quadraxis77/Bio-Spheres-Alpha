// Nutrient particle extraction compute shader
// Spawns nutrient particles in water voxels that are not isolated

struct NutrientExtractParams {
    grid_resolution: u32,
    cell_size: f32,
    max_particles: u32,
    time: f32,
    grid_origin: vec3<f32>,
    world_radius: f32,
    spawn_probability: f32,  // Legacy parameter (kept for compatibility)
    nutrient_density: f32,   // New density parameter (0.0 = sparse, 1.0 = dense)
    _padding: vec2<f32>,
}

struct NutrientParticle {
    position: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    animation: vec4<f32>,
}

// Atomic counter for particle count
struct NutrientParticleCounter {
    count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(1) var<storage, read_write> particles: array<NutrientParticle>;
@group(0) @binding(2) var<storage, read_write> counter: NutrientParticleCounter;
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(4) var<uniform> params: NutrientExtractParams;

// Atomic counter helper
fn atomic_increment() -> u32 {
    return atomicAdd(&counter.count, 1u);
}

// Smooth interpolation function (like cave system)
fn smoothstep(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// 3D value noise - interpolates between random values at lattice points
fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    // Integer and fractional parts
    let ix = floor(pos.x);
    let iy = floor(pos.y);
    let iz = floor(pos.z);
    let fx = fract(pos.x);
    let fy = fract(pos.y);
    let fz = fract(pos.z);
    
    // Hash function for each corner of the cube
    let h000 = fract(sin(dot(vec3<f32>(ix, iy, iz) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h001 = fract(sin(dot(vec3<f32>(ix, iy, iz + 1.0) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h010 = fract(sin(dot(vec3<f32>(ix, iy + 1.0, iz) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h011 = fract(sin(dot(vec3<f32>(ix, iy + 1.0, iz + 1.0) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h100 = fract(sin(dot(vec3<f32>(ix + 1.0, iy, iz) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h101 = fract(sin(dot(vec3<f32>(ix + 1.0, iy, iz + 1.0) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h110 = fract(sin(dot(vec3<f32>(ix + 1.0, iy + 1.0, iz) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    let h111 = fract(sin(dot(vec3<f32>(ix + 1.0, iy + 1.0, iz + 1.0) + vec3<f32>(f32(seed)), vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    
    // Interpolate along X
    let x00 = mix(h000, h100, smoothstep(fx));
    let x01 = mix(h001, h101, smoothstep(fx));
    let x10 = mix(h010, h110, smoothstep(fx));
    let x11 = mix(h011, h111, smoothstep(fx));
    
    // Interpolate along Y
    let y0 = mix(x00, x10, smoothstep(fy));
    let y1 = mix(x01, x11, smoothstep(fy));
    
    // Interpolate along Z
    let result = mix(y0, y1, smoothstep(fz));
    
    return result;  // All hash values are in [0,1], so result stays in [0,1]
}

// Fractal Brownian Motion - combines multiple octaves of value noise
fn fbm(pos: vec3<f32>, scale: f32, octaves: u32, persistence: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    
    for (var i = 0u; i < octaves; i = i + 1u) {
        let sample_pos = pos * frequency / scale;
        value += amplitude * value_noise_3d(sample_pos, 1337u + i * 999u);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

// Check if a voxel should spawn particles based on Perlin-like noise
fn should_spawn_particle(voxel_index: u32, world_pos: vec3<f32>, density: f32) -> bool {
    // Use FBM noise like cave system
    let noise = fbm(world_pos, 20.0, 3u, 0.5);  // Low scale, few octaves for large patches
    
    // Use density parameter to control threshold
    // Higher density = more particles (direct relationship)
    let threshold = 1.0 - density;  // density 0.0 = threshold 1.0 (no particles), density 1.0 = threshold 0.0 (max particles)
    
    // Fixed: Use > instead of < for correct behavior
    return noise > threshold;
}

// Check if a voxel is water (fluid type 1)
fn is_water_voxel(voxel_index: u32) -> bool {
    let state = fluid_state[voxel_index];
    let fluid_type = state & 0xFFFFu;  // Fluid type is stored in lower 16 bits
    return fluid_type == 1u;  // Water fluid type
}

// Check if a water voxel is isolated (no neighboring water)
fn is_water_isolated(voxel_index: u32) -> bool {
    let grid_res = params.grid_resolution;
    
    // Calculate 3D coordinates
    let x = voxel_index % grid_res;
    let y = (voxel_index / grid_res) % grid_res;
    let z = voxel_index / (grid_res * grid_res);
    
    var water_neighbors = 0u;
    
    // Check all 6 neighbors
    for (var i = 0i; i < 6i; i++) {
        var nx = i32(x);
        var ny = i32(y);
        var nz = i32(z);
        
        // Apply neighbor offset
        switch (i) {
            case 0: { nx -= 1; }  // -X
            case 1: { nx += 1; }  // +X
            case 2: { ny -= 1; }  // -Y
            case 3: { ny += 1; }  // +Y
            case 4: { nz -= 1; }  // -Z
            case 5: { nz += 1; }  // +Z
            default: {}
        }
        
        // Check bounds
        if (nx >= 0 && nx < i32(grid_res) && 
            ny >= 0 && ny < i32(grid_res) && 
            nz >= 0 && nz < i32(grid_res)) {
            
            let neighbor_index = u32(nx) + u32(ny) * grid_res + u32(nz) * grid_res * grid_res;
            if (is_water_voxel(neighbor_index)) {
                water_neighbors += 1u;
            }
        }
    }
    
    return water_neighbors == 0u;  // Isolated if no water neighbors
}

// Convert voxel index to world position
fn voxel_to_world(voxel_index: u32) -> vec3<f32> {
    let grid_res = params.grid_resolution;
    let cell_size = params.cell_size;
    
    // Calculate 3D coordinates
    let x = f32(voxel_index % grid_res);
    let y = f32((voxel_index / grid_res) % grid_res);
    let z = f32(voxel_index / (grid_res * grid_res));
    
    // Convert to world coordinates with center offset
    let world_pos = vec3<f32>(
        x * cell_size + params.grid_origin.x,
        y * cell_size + params.grid_origin.y,
        z * cell_size + params.grid_origin.z
    );
    
    return world_pos + vec3<f32>(cell_size * 0.5);  // Center of voxel
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_index = global_id.x + global_id.y * params.grid_resolution + global_id.z * params.grid_resolution * params.grid_resolution;
    
    // Check bounds
    if (voxel_index >= arrayLength(&fluid_state)) {
        return;
    }
    
    // Check if this is a water voxel and not isolated
    if (!is_water_voxel(voxel_index) || is_water_isolated(voxel_index)) {
        return;
    }
    
    // Create nutrient particle position for noise calculation
    let world_pos = voxel_to_world(voxel_index);
    
    // Use noise pattern for uneven distribution instead of uniform probability
    if (!should_spawn_particle(voxel_index, world_pos, params.nutrient_density)) {
        return;
    }
    
    // Try to add a particle
    let particle_count = atomic_increment();
    if (particle_count >= params.max_particles) {
        return;  // Particle buffer full
    }
    
    // Add some randomness to position within voxel
    let offset = vec3<f32>(
        fract(sin(f32(voxel_index) * 7.3)) * 0.8 - 0.4,
        fract(sin(f32(voxel_index) * 13.7)) * 0.8 - 0.4,
        fract(sin(f32(voxel_index) * 19.1)) * 0.8 - 0.4
    ) * params.cell_size;
    
    particles[particle_count] = NutrientParticle(
        world_pos + offset,
        0.1 + fract(sin(f32(voxel_index) * 7.3)) * 0.05,  // Small size with variation (0.1-0.15)
        vec4<f32>(0.6, 0.4, 0.2, 1.0),  // Brown opaque color
        vec4<f32>(
            params.time + fract(sin(f32(voxel_index) * 12.9898)) * 6.28,  // Animation time offset
            2.0 + fract(sin(f32(voxel_index) * 13.7)),  // Rotation speed
            f32(voxel_index % 4) * 1.570796,  // 4 distinct rotations (0°, 90°, 180°, 270°) for debugging
            0.0
        )
    );
}
