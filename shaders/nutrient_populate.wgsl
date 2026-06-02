// Nutrient Population Shader - Epoch-based with gradual spawn
//
// Instead of drifting a noise field continuously, nutrients appear in discrete
// "epochs".  Each epoch has a fixed noise seed that determines WHERE nutrients
// can appear.  Within an epoch, nutrients spawn gradually: a per-voxel random
// spawn-time is compared against the epoch progress (0->1) so early voxels
// appear first and the pattern fills in over the epoch duration.
//
// When phagocytes consume a voxel (state 2), it stays consumed for the rest
// of the epoch.  At the epoch boundary the seed changes, giving a completely
// fresh spatial pattern.
//
// Runs every physics step.

struct NutrientPopulateParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    world_radius: f32,
    nutrient_density: f32,
    time: f32,
    delta_time: f32,
    epoch_duration: f32,
    epoch_spacing: f32,
    spawn_end: f32,
    despawn_start: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> params: NutrientPopulateParams;

@group(0) @binding(1)
var<storage, read> fluid_state: array<u32>;

@group(0) @binding(2)
var<storage, read_write> nutrient_voxels: array<atomic<u32>>;

// -- Noise helpers ----------------------------------------------

fn smoothstep(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Integer hash for a 3-D lattice point + epoch seed.
//
// WHY THIS REPLACES THE ORIGINAL SIN-BASED HASH:
// The original code used  s = vec3<f32>(f32(seed))  and passed it into
//   fract(sin(dot(pos + s, k)) * 43758.5453)
// Because the epoch seed grows as  1337 + epoch * 7919,  by epoch ~17
// (~119 simulated seconds, roughly 2 minutes) the sin() argument exceeds
// ~3 x 10^2.  At that magnitude f32 has no sub-ULP precision left for
// trigonometric range reduction, so sin() returns the same garbage value
// (often 0.0) for every voxel.  The fbm collapses to 0, noise <= threshold
// becomes universally true, and nutrients never spawn again - permanently.
//
// Integer hashing (Murmur3 finalizer mix) is immune to input magnitude:
// it operates purely in u32 modular arithmetic, guarantees uniform output
// in [0, 1) for any coordinate+seed combination, and is consistent across
// all GPU vendors.
fn hash_noise(xi: i32, yi: i32, zi: i32, seed: u32) -> f32 {
    // Bias coordinates into non-negative u32 range (noise samples are in [-15, 15])
    var h = (u32(xi + 4096) * 2654435761u)
          ^ (u32(yi + 4096) * 1013904223u)
          ^ (u32(zi + 4096) * 2246822519u)
          ^ seed;
    // Murmur3 finalizer
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    let ix = i32(floor(pos.x));
    let iy = i32(floor(pos.y));
    let iz = i32(floor(pos.z));
    let fx = fract(pos.x);
    let fy = fract(pos.y);
    let fz = fract(pos.z);

    let h000 = hash_noise(ix,   iy,   iz,   seed);
    let h100 = hash_noise(ix+1, iy,   iz,   seed);
    let h010 = hash_noise(ix,   iy+1, iz,   seed);
    let h110 = hash_noise(ix+1, iy+1, iz,   seed);
    let h001 = hash_noise(ix,   iy,   iz+1, seed);
    let h101 = hash_noise(ix+1, iy,   iz+1, seed);
    let h011 = hash_noise(ix,   iy+1, iz+1, seed);
    let h111 = hash_noise(ix+1, iy+1, iz+1, seed);

    let sx = smoothstep(fx);
    let sy = smoothstep(fy);
    let sz = smoothstep(fz);
    let x00 = mix(h000, h100, sx);
    let x01 = mix(h001, h101, sx);
    let x10 = mix(h010, h110, sx);
    let x11 = mix(h011, h111, sx);
    let y0  = mix(x00,  x10,  sy);
    let y1  = mix(x01,  x11,  sy);
    return mix(y0, y1, sz);
}

fn fbm(pos: vec3<f32>, scale: f32, octaves: u32, persistence: f32, seed: u32) -> f32 {
    var value     = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    for (var i = 0u; i < octaves; i++) {
        value     += amplitude * value_noise_3d(pos * frequency / scale, seed + i * 999u);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    return value / max_value;
}

// Per-voxel hash that determines when within an epoch this voxel spawns.
// Returns a value in [0, 1).
fn spawn_time_hash(voxel_index: u32, epoch: u32) -> f32 {
    let h = voxel_index * 2654435761u + epoch * 1013904223u;
    return f32((h >> 8u) & 0xFFFFFFu) / 16777216.0;
}

// -- Helpers ----------------------------------------------------

fn voxel_to_world(voxel_index: u32) -> vec3<f32> {
    let res = params.grid_resolution;
    let x = f32(voxel_index % res);
    let y = f32((voxel_index / res) % res);
    let z = f32(voxel_index / (res * res));
    return vec3<f32>(
        x * params.cell_size + params.grid_origin_x + params.cell_size * 0.5,
        y * params.cell_size + params.grid_origin_y + params.cell_size * 0.5,
        z * params.cell_size + params.grid_origin_z + params.cell_size * 0.5
    );
}

fn is_water_voxel(voxel_index: u32) -> bool {
    return (fluid_state[voxel_index] & 0xFFFFu) == 1u;
}

fn is_water_isolated(x: u32, y: u32, z: u32) -> bool {
    let res = params.grid_resolution;
    var water_neighbors = 0u;
    for (var i = 0i; i < 6i; i++) {
        var nx = i32(x); var ny = i32(y); var nz = i32(z);
        switch (i) {
            case 0: { nx -= 1; } case 1: { nx += 1; }
            case 2: { ny -= 1; } case 3: { ny += 1; }
            case 4: { nz -= 1; } case 5: { nz += 1; }
            default: {}
        }
        if (nx >= 0 && nx < i32(res) && ny >= 0 && ny < i32(res) && nz >= 0 && nz < i32(res)) {
            let ni = u32(nx) + u32(ny) * res + u32(nz) * res * res;
            if (is_water_voxel(ni)) { water_neighbors += 1u; }
        }
    }
    return water_neighbors == 0u;
}

// -- Epoch lifecycle ---------------------------------------------
//
// Each epoch lasts params.epoch_duration seconds and follows this timeline:
//
//   0 -- spawn_end -- despawn_start -- epoch_duration
//   | spawn ramp-up | fully active | despawn ramp-down |
//
// Epochs are spaced params.epoch_spacing apart (< epoch_duration), so the
// despawn tail of epoch N overlaps with the spawn ramp of epoch N+1.
// This guarantees nutrients are always available during transitions.
//
// A voxel is active if ANY currently-overlapping epoch wants it active.

fn despawn_time_hash(voxel_index: u32, epoch: u32) -> f32 {
    let h = voxel_index * 2246822519u + epoch * 3266489917u;
    return f32((h >> 8u) & 0xFFFFFFu) / 16777216.0;
}

// Evaluate whether a single epoch wants this voxel to hold a nutrient.
fn epoch_wants_nutrient(
    voxel_index: u32,
    world_pos: vec3<f32>,
    epoch: u32,
    age: f32,
    threshold: f32,
) -> bool {
    if (age < 0.0 || age >= params.epoch_duration) { return false; }

    let progress = age / params.epoch_duration;

    // Spatial eligibility
    let noise_seed = 1337u + epoch * 7919u;
    let noise = fbm(world_pos, 20.0, 3u, 0.5, noise_seed);
    if (noise <= threshold) { return false; }

    // Spawn ramp
    let spawn_progress = clamp(progress / params.spawn_end, 0.0, 1.0);
    let voxel_spawn_t  = spawn_time_hash(voxel_index, epoch);
    if (spawn_progress < voxel_spawn_t) { return false; }

    // Despawn ramp
    if (progress >= params.despawn_start) {
        let despawn_progress = (progress - params.despawn_start) / (1.0 - params.despawn_start);
        let voxel_despawn_t  = despawn_time_hash(voxel_index, epoch);
        if (despawn_progress >= voxel_despawn_t) { return false; }
    }

    return true;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = params.grid_resolution;
    if (global_id.x >= res || global_id.y >= res || global_id.z >= res) { return; }

    let voxel_index = global_id.x + global_id.y * res + global_id.z * res * res;

    // Non-water or isolated -> clear and bail
    if (!is_water_voxel(voxel_index) || is_water_isolated(global_id.x, global_id.y, global_id.z)) {
        atomicStore(&nutrient_voxels[voxel_index], 0u);
        return;
    }

    let world_pos = voxel_to_world(voxel_index);
    let threshold = 1.0 - params.nutrient_density;
    let t         = params.time;

    // Determine which epochs are currently alive.
    let latest_epoch = u32(t / params.epoch_spacing);
    // Check the two most recent epochs (at most 2 can overlap).
    var any_wants = false;
    for (var i = 0u; i < 2u; i++) {
        if (latest_epoch < i) { continue; }  // avoid underflow
        let ep       = latest_epoch - i;
        let ep_start = f32(ep) * params.epoch_spacing;
        let age      = t - ep_start;
        if (epoch_wants_nutrient(voxel_index, world_pos, ep, age, threshold)) {
            any_wants = true;
            break;
        }
    }

    let current = atomicLoad(&nutrient_voxels[voxel_index]);

    if (any_wants) {
        // At least one epoch wants a nutrient here.
        // Spawn only into empty voxels - don't overwrite consumed (2).
        if (current == 0u) {
            atomicStore(&nutrient_voxels[voxel_index], 1u);
        }
    } else {
        // No epoch wants this voxel active - clear it.
        if (current != 0u) {
            atomicStore(&nutrient_voxels[voxel_index], 0u);
        }
    }
}
