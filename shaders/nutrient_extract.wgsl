// Nutrient particle extraction compute shader
// Spawns visual particles for voxels that have nutrients in the nutrient_voxels buffer

struct NutrientExtractParams {
    grid_resolution: u32,
    cell_size: f32,
    max_particles: u32,
    time: f32,
    grid_origin: vec3<f32>,
    world_radius: f32,
    spawn_probability: f32,  // Legacy parameter (kept for compatibility)
    nutrient_density: f32,   // Density parameter for population shader
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
@group(0) @binding(5) var<storage, read> nutrient_voxels: array<u32>;  // 0 = empty, 1 = has nutrient, 2 = consumed

// Atomic counter helper - returns the slot index, or max_particles if full.
// Checks capacity BEFORE incrementing so the counter never overflows past max_particles.
fn try_allocate_particle() -> u32 {
    let current = atomicLoad(&counter.count);
    if (current >= params.max_particles) {
        return params.max_particles;  // Full - signal caller to skip
    }
    return atomicAdd(&counter.count, 1u);
}

// Check if a voxel is water (fluid type 1)
fn is_water_voxel(voxel_index: u32) -> bool {
    let state = fluid_state[voxel_index];
    let fluid_type = state & 0x7u;  // Fluid type is stored in lower 16 bits
    return fluid_type == 1u;  // Water fluid type
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

// Check if voxel has a nutrient (reads from nutrient buffer)
fn has_nutrient(voxel_index: u32) -> bool {
    return nutrient_voxels[voxel_index] == 1u;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_index = global_id.x + global_id.y * params.grid_resolution + global_id.z * params.grid_resolution * params.grid_resolution;
    
    // Check bounds
    if (voxel_index >= arrayLength(&fluid_state)) {
        return;
    }
    
    // Check if this voxel has a nutrient (from nutrient buffer, populated by noise)
    if (!has_nutrient(voxel_index)) {
        return;
    }
    
    // Also verify it's still a water voxel (nutrient may have been orphaned)
    if (!is_water_voxel(voxel_index)) {
        return;
    }
    
    // Get world position for particle rendering
    let world_pos = voxel_to_world(voxel_index);
    
    // Try to add a particle - check capacity before incrementing so the counter
    // never wraps past max_particles (which would cause particle_count to read as
    // 0 after a u32 overflow, making all particles disappear permanently).
    let particle_count = try_allocate_particle();
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
            f32(voxel_index % 4) * 1.570796,  // 4 distinct rotations (0 deg, 90 deg, 180 deg, 270 deg) for debugging
            0.0
        )
    );
}
