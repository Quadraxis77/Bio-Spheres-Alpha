// Nutrient particle extraction compute shader (Lifecycle-aware)
//
// Reads lifecycle-encoded nutrient_voxels buffer and spawns visual particles
// with crossfade alpha based on lifecycle phase (fade-in / sustain / fade-out).
//
// nutrient_voxels encoding (u32):
//   0              = empty
//   bits 0-15      = spawn time (tenths of a second, wrapping u16)
//   bits 16-31     = lifetime duration (tenths of a second)

struct NutrientExtractParams {
    grid_resolution: u32,
    cell_size: f32,
    max_particles: u32,
    time: f32,
    grid_origin: vec3<f32>,
    world_radius: f32,
    spawn_probability: f32,  // Legacy parameter (kept for compatibility)
    nutrient_density: f32,   // Density parameter for population shader
    fade_duration: f32,      // Crossfade duration in seconds (e.g. 1.5)
    _padding: f32,
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
@group(0) @binding(5) var<storage, read> nutrient_voxels: array<u32>;

// Atomic counter helper
fn atomic_increment() -> u32 {
    return atomicAdd(&counter.count, 1u);
}

// Check if a voxel is water (fluid type 1)
fn is_water_voxel(voxel_index: u32) -> bool {
    let state = fluid_state[voxel_index];
    let fluid_type = state & 0xFFFFu;
    return fluid_type == 1u;
}

// Convert voxel index to world position
fn voxel_to_world(voxel_index: u32) -> vec3<f32> {
    let grid_res = params.grid_resolution;
    let cell_size = params.cell_size;
    
    let x = f32(voxel_index % grid_res);
    let y = f32((voxel_index / grid_res) % grid_res);
    let z = f32(voxel_index / (grid_res * grid_res));
    
    let world_pos = vec3<f32>(
        x * cell_size + params.grid_origin.x,
        y * cell_size + params.grid_origin.y,
        z * cell_size + params.grid_origin.z
    );
    
    return world_pos + vec3<f32>(cell_size * 0.5);
}

// Unpack spawn_time (tenths) from packed nutrient value
fn unpack_spawn_time(packed: u32) -> u32 {
    return packed & 0xFFFFu;
}

// Unpack lifetime (tenths) from packed nutrient value
fn unpack_lifetime(packed: u32) -> u32 {
    return (packed >> 16u) & 0xFFFFu;
}

// Get current time in tenths of a second (wrapping u16)
fn current_time_tenths() -> u32 {
    return u32(params.time * 10.0) & 0xFFFFu;
}

// Calculate age of a nutrient in tenths of a second (handles u16 wrapping)
fn nutrient_age_tenths(spawn_time: u32) -> u32 {
    let now = current_time_tenths();
    return (now - spawn_time) & 0xFFFFu;
}

// Compute crossfade alpha for a nutrient based on its lifecycle phase
// Timeline: [spawn] -> fade-in -> sustain -> fade-out -> [dead]
//   fade-in:  age 0 to fade_duration
//   sustain:  age fade_duration to lifetime
//   fade-out: age lifetime to lifetime + fade_duration
//   dead:     age > lifetime + fade_duration
fn compute_crossfade_alpha(packed: u32) -> f32 {
    if (packed == 0u) {
        return 0.0;
    }
    
    let spawn_time = unpack_spawn_time(packed);
    let lifetime_tenths = unpack_lifetime(packed);
    let age_tenths = nutrient_age_tenths(spawn_time);
    let fade_tenths = u32(params.fade_duration * 10.0);
    
    // Fully dead (past fade-out grace period)
    if (age_tenths >= lifetime_tenths + fade_tenths) {
        return 0.0;
    }
    
    let age_seconds = f32(age_tenths) * 0.1;
    let lifetime_seconds = f32(lifetime_tenths) * 0.1;
    let fade = params.fade_duration;
    
    // Fade-in phase: age 0 to fade_duration
    if (age_seconds < fade) {
        return age_seconds / fade;
    }
    
    // Fade-out phase: age lifetime to lifetime + fade_duration
    if (age_seconds > lifetime_seconds) {
        let fade_out_progress = age_seconds - lifetime_seconds;
        return max(1.0 - fade_out_progress / fade, 0.0);
    }
    
    // Sustain phase (full opacity)
    return 1.0;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_index = global_id.x + global_id.y * params.grid_resolution + global_id.z * params.grid_resolution * params.grid_resolution;
    
    // Check bounds
    if (voxel_index >= arrayLength(&fluid_state)) {
        return;
    }
    
    // Read packed nutrient lifecycle data
    let packed = nutrient_voxels[voxel_index];
    if (packed == 0u) {
        return;
    }
    
    // Compute crossfade alpha - skip dead nutrients
    let alpha = compute_crossfade_alpha(packed);
    if (alpha <= 0.0) {
        return;
    }
    
    // Also verify it's still a water voxel (nutrient may have been orphaned)
    if (!is_water_voxel(voxel_index)) {
        return;
    }
    
    // Get world position for particle rendering
    let world_pos = voxel_to_world(voxel_index);
    
    // Try to add a particle
    let particle_count = atomic_increment();
    if (particle_count >= params.max_particles) {
        return;
    }
    
    // Add some randomness to position within voxel
    let offset = vec3<f32>(
        fract(sin(f32(voxel_index) * 7.3)) * 0.8 - 0.4,
        fract(sin(f32(voxel_index) * 13.7)) * 0.8 - 0.4,
        fract(sin(f32(voxel_index) * 19.1)) * 0.8 - 0.4
    ) * params.cell_size;
    
    // Scale size by alpha for smooth crossfade (particles grow in and shrink out)
    let base_size = 0.15 + fract(sin(f32(voxel_index) * 7.3)) * 0.08;
    let scaled_size = base_size * alpha;
    
    particles[particle_count] = NutrientParticle(
        world_pos + offset,
        scaled_size,
        vec4<f32>(0.6, 0.4, 0.2, alpha),  // Brown color with crossfade alpha
        vec4<f32>(
            params.time + fract(sin(f32(voxel_index) * 12.9898)) * 6.28,
            2.0 + fract(sin(f32(voxel_index) * 13.7)),
            f32(voxel_index % 4) * 1.570796,
            alpha  // Pass alpha to animation.w for vertex shader use
        )
    );
}
