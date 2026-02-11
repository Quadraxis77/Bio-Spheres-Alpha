// Nutrient Population Shader (Lifecycle-based)
//
// Manages per-voxel nutrient lifecycles with staggered crossfade.
// Each nutrient has a randomized spawn time and lifetime so they don't all
// appear/disappear at once. The crossfade ensures consistent overall density.
//
// nutrient_voxels encoding (u32):
//   0              = empty (no nutrient)
//   bits 0-15      = spawn time (in tenths of a second, wrapping at 65536 = ~109 min)
//   bits 16-31     = lifetime duration (in tenths of a second, randomized per-voxel)
//
// A nutrient is "alive" when: (current_time_tenths - spawn_time) < lifetime
// Fade-in during first fade_duration, fade-out during last fade_duration.

struct NutrientPopulateParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    world_radius: f32,
    nutrient_density: f32,   // 0.0 = sparse, 1.0 = dense
    time: f32,               // Current time in seconds
    base_lifetime: f32,      // Base lifetime in seconds (e.g. 8.0)
    lifetime_variance: f32,  // Random variance range in seconds (e.g. 4.0)
    fade_duration: f32,      // Crossfade duration in seconds (e.g. 1.5)
    spawn_rate: f32,         // Probability of spawning per eligible empty voxel per frame (e.g. 0.02)
}

@group(0) @binding(0)
var<uniform> params: NutrientPopulateParams;

@group(0) @binding(1)
var<storage, read> fluid_state: array<u32>;

@group(0) @binding(2)
var<storage, read_write> nutrient_voxels: array<atomic<u32>>;

// Smooth interpolation function
fn smoothstep_custom(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// 3D value noise - interpolates between random values at lattice points
fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
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
    let x00 = mix(h000, h100, smoothstep_custom(fx));
    let x01 = mix(h001, h101, smoothstep_custom(fx));
    let x10 = mix(h010, h110, smoothstep_custom(fx));
    let x11 = mix(h011, h111, smoothstep_custom(fx));
    
    // Interpolate along Y
    let y0 = mix(x00, x10, smoothstep_custom(fy));
    let y1 = mix(x01, x11, smoothstep_custom(fy));
    
    // Interpolate along Z
    return mix(y0, y1, smoothstep_custom(fz));
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

// Convert voxel index to world position
fn voxel_to_world(voxel_index: u32) -> vec3<f32> {
    let grid_res = params.grid_resolution;
    let cell_size = params.cell_size;
    
    let x = f32(voxel_index % grid_res);
    let y = f32((voxel_index / grid_res) % grid_res);
    let z = f32(voxel_index / (grid_res * grid_res));
    
    return vec3<f32>(
        x * cell_size + params.grid_origin_x + cell_size * 0.5,
        y * cell_size + params.grid_origin_y + cell_size * 0.5,
        z * cell_size + params.grid_origin_z + cell_size * 0.5
    );
}

// Check if voxel contains water
fn is_water_voxel(voxel_index: u32) -> bool {
    let state = fluid_state[voxel_index];
    let fluid_type = state & 0xFFFFu;
    return fluid_type == 1u;
}

// Check if a water voxel is isolated (no neighboring water) - skip isolated voxels
fn is_water_isolated(x: u32, y: u32, z: u32) -> bool {
    let grid_res = params.grid_resolution;
    var water_neighbors = 0u;
    
    // Check all 6 neighbors
    for (var i = 0i; i < 6i; i++) {
        var nx = i32(x);
        var ny = i32(y);
        var nz = i32(z);
        
        switch (i) {
            case 0: { nx -= 1; }
            case 1: { nx += 1; }
            case 2: { ny -= 1; }
            case 3: { ny += 1; }
            case 4: { nz -= 1; }
            case 5: { nz += 1; }
            default: {}
        }
        
        if (nx >= 0 && nx < i32(grid_res) && 
            ny >= 0 && ny < i32(grid_res) && 
            nz >= 0 && nz < i32(grid_res)) {
            
            let neighbor_index = u32(nx) + u32(ny) * grid_res + u32(nz) * grid_res * grid_res;
            if (is_water_voxel(neighbor_index)) {
                water_neighbors += 1u;
            }
        }
    }
    
    return water_neighbors == 0u;
}

// Fast hash for per-voxel randomization (returns 0.0-1.0)
fn voxel_hash(voxel_index: u32, seed: u32) -> f32 {
    let n = voxel_index * 1103515245u + seed * 12345u + 2531011u;
    return f32((n >> 16u) & 0x7FFFu) / 32767.0;
}

// Pack spawn_time (tenths) and lifetime (tenths) into a u32
// spawn_time in bits 0-15, lifetime in bits 16-31
fn pack_nutrient(spawn_time_tenths: u32, lifetime_tenths: u32) -> u32 {
    return (spawn_time_tenths & 0xFFFFu) | ((lifetime_tenths & 0xFFFFu) << 16u);
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

// Check if a nutrient is still alive or in its fade-out grace period
// Keeps the nutrient in the buffer for an extra fade_duration so the
// extract shader can render the fade-out before it's cleared.
fn is_nutrient_alive(packed: u32) -> bool {
    if (packed == 0u) {
        return false;
    }
    let spawn_time = unpack_spawn_time(packed);
    let lifetime = unpack_lifetime(packed);
    let fade_tenths = u32(params.fade_duration * 10.0);
    let age = nutrient_age_tenths(spawn_time);
    return age < (lifetime + fade_tenths);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = params.grid_resolution;
    
    if (global_id.x >= res || global_id.y >= res || global_id.z >= res) {
        return;
    }
    
    let voxel_index = global_id.x + global_id.y * res + global_id.z * res * res;
    
    // Read current nutrient state
    let current = atomicLoad(&nutrient_voxels[voxel_index]);
    
    // If not a water voxel, clear any nutrient
    if (!is_water_voxel(voxel_index)) {
        if (current != 0u) {
            atomicStore(&nutrient_voxels[voxel_index], 0u);
        }
        return;
    }
    
    // Skip isolated water voxels
    if (is_water_isolated(global_id.x, global_id.y, global_id.z)) {
        if (current != 0u) {
            atomicStore(&nutrient_voxels[voxel_index], 0u);
        }
        return;
    }
    
    // --- Batch placement model ---
    //
    // Each cycle (base_lifetime seconds), a single noise pattern determines which
    // voxels get nutrients. The pattern is fixed for the entire cycle — it does NOT
    // drift or change until the next cycle begins with a fresh noise seed.
    // Per-voxel lifetime randomization staggers the fade-out within a batch.
    
    let world_pos = voxel_to_world(voxel_index);
    
    // Determine which cycle we're in. Each cycle gets a unique noise seed.
    let cycle_index = u32(params.time / params.base_lifetime);
    let cycle_seed = cycle_index * 5381u + 31u;
    
    // Evaluate noise pattern for THIS cycle (stable for entire cycle duration)
    // The seed offsets the sampling position so each cycle has a different pattern
    let seed_offset = vec3<f32>(
        f32(cycle_seed & 0xFFu) * 0.37,
        f32((cycle_seed >> 8u) & 0xFFu) * 0.53,
        f32((cycle_seed >> 16u) & 0xFFu) * 0.71
    );
    let noise = fbm(world_pos + seed_offset, 20.0, 3u, 0.5);
    
    // Density threshold from user setting
    let threshold = 1.0 - params.nutrient_density;
    let in_eligible_zone = noise > threshold;
    
    // --- Lifecycle management ---
    
    if (current != 0u) {
        if (is_nutrient_alive(current) && in_eligible_zone) {
            // Still alive and still in this cycle's pattern, keep it
            return;
        } else {
            // Expired or pattern changed — clear it
            atomicStore(&nutrient_voxels[voxel_index], 0u);
            // Fall through: if we're in a new cycle and eligible, spawn below
            if (!in_eligible_zone) {
                return;
            }
        }
    }
    
    // --- Spawn logic ---
    
    if (!in_eligible_zone) {
        return;
    }
    
    // Per-voxel lifetime randomization (deterministic per voxel+cycle)
    // This staggers the fade-out so not all particles disappear at the exact same frame
    let lifetime_rand = voxel_hash(voxel_index, cycle_seed + 7777u);
    let lifetime_seconds = params.base_lifetime + (lifetime_rand * 2.0 - 1.0) * params.lifetime_variance;
    let lifetime_tenths = u32(max(lifetime_seconds, 2.0) * 10.0);
    
    // Per-voxel spawn delay: small random offset (0 to fade_duration) so they don't
    // ALL pop in on the exact same frame — creates a brief staggered fade-in
    let spawn_delay_rand = voxel_hash(voxel_index, cycle_seed + 3333u);
    let spawn_delay_tenths = u32(spawn_delay_rand * params.fade_duration * 10.0);
    
    // The batch's nominal spawn time is the start of this cycle
    let cycle_start_tenths = u32(f32(cycle_index) * params.base_lifetime * 10.0) & 0xFFFFu;
    let spawn_time = (cycle_start_tenths + spawn_delay_tenths) & 0xFFFFu;
    
    // Only actually spawn if we've reached this voxel's delayed spawn time
    let now = current_time_tenths();
    let time_since_cycle_start = (now - cycle_start_tenths) & 0xFFFFu;
    if (time_since_cycle_start < spawn_delay_tenths) {
        return;  // Not yet — stagger the fade-in
    }
    
    let packed = pack_nutrient(spawn_time, lifetime_tenths);
    
    if (packed == 0u) {
        atomicStore(&nutrient_voxels[voxel_index], 1u);
    } else {
        atomicStore(&nutrient_voxels[voxel_index], packed);
    }
}
