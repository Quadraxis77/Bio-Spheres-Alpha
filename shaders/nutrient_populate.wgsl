// Nutrient Population Shader
// Uses noise to populate nutrient voxels in water areas
// Run once on initialization or periodically to replenish nutrients

struct NutrientPopulateParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    world_radius: f32,
    nutrient_density: f32,   // 0.0 = sparse, 1.0 = dense
    time: f32,               // For time-based variation
}

@group(0) @binding(0)
var<uniform> params: NutrientPopulateParams;

@group(0) @binding(1)
var<storage, read> fluid_state: array<u32>;

@group(0) @binding(2)
var<storage, read_write> nutrient_voxels: array<atomic<u32>>;

// Smooth interpolation function
fn smoothstep(t: f32) -> f32 {
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
    let x00 = mix(h000, h100, smoothstep(fx));
    let x01 = mix(h001, h101, smoothstep(fx));
    let x10 = mix(h010, h110, smoothstep(fx));
    let x11 = mix(h011, h111, smoothstep(fx));
    
    // Interpolate along Y
    let y0 = mix(x00, x10, smoothstep(fy));
    let y1 = mix(x01, x11, smoothstep(fy));
    
    // Interpolate along Z
    return mix(y0, y1, smoothstep(fz));
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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = params.grid_resolution;
    
    if (global_id.x >= res || global_id.y >= res || global_id.z >= res) {
        return;
    }
    
    let voxel_index = global_id.x + global_id.y * res + global_id.z * res * res;
    
    // Only populate nutrients in water voxels
    if (!is_water_voxel(voxel_index)) {
        atomicStore(&nutrient_voxels[voxel_index], 0u);
        return;
    }
    
    // Skip isolated water voxels
    if (is_water_isolated(global_id.x, global_id.y, global_id.z)) {
        atomicStore(&nutrient_voxels[voxel_index], 0u);
        return;
    }
    
    // Get world position for noise sampling
    let world_pos = voxel_to_world(voxel_index);
    
    // Add time-based drift to create rolling/drifting nutrient zones
    // Drift speed is slow (0.5 units/sec) for gentle movement
    let drift_speed = 0.5;
    let drift_offset = vec3<f32>(
        params.time * drift_speed * 0.7,   // Drift along X
        params.time * drift_speed * 0.3,   // Slower drift along Y  
        params.time * drift_speed * 0.5    // Medium drift along Z
    );
    let drifting_pos = world_pos + drift_offset;
    
    // Use FBM noise for uneven distribution with drifting coordinates
    let noise = fbm(drifting_pos, 20.0, 3u, 0.5);
    
    // Use density parameter to control threshold
    let threshold = 1.0 - params.nutrient_density;
    
    // Nutrient present if noise exceeds threshold
    if (noise > threshold) {
        atomicStore(&nutrient_voxels[voxel_index], 1u);
    } else {
        atomicStore(&nutrient_voxels[voxel_index], 0u);
    }
}
