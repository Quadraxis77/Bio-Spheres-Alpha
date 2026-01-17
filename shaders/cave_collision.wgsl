// Cave Collision SDF Shader
// 
// Implements GPU-based SDF collision detection using Voronoi-based cave generation.
// Uses 3D Voronoi cells with random wall thresholds for clear wall/air regions.
//
// Voronoi cave approach:
// - Each Voronoi cell has a random "wall threshold" (0-1)
// - If wall_density > wall_threshold, the cell is a CAVE OBSTACLE (solid)
// - Caves are SOLID OBSTACLES that cells must avoid
// - Cells stay in OPEN SPACE (non-cave cells)
// - When in a cave cell, cells are pushed OUT into open space

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct CaveParams {
    world_center: vec3<f32>,
    world_radius: f32,
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
    grid_resolution: u32,
    triangle_count: u32,
    collision_enabled: u32,
    collision_stiffness: f32,
    collision_damping: f32,
    substeps: u32,
    _padding: f32,
    // Padding to 256 bytes (16-byte aligned)
    _padding2: vec4<f32>,
    _padding3: vec4<f32>,
    _padding4: vec4<f32>,
    _padding5: vec4<f32>,
    _padding6: vec4<f32>,
    _padding7: vec4<f32>,
    _padding8: vec4<f32>,
    _padding9: vec4<f32>,
    _padding10: vec4<f32>,
    _padding11: vec4<f32>,
    _padding12: vec4<f32>,
    _padding13: vec4<f32>,
    _padding14: vec4<f32>,
    _padding15: vec4<f32>,
    _padding16: vec4<f32>,
    _padding17: vec4<f32>,
    _padding18: vec4<f32>,
    _padding19: vec4<f32>,
    _padding20: vec4<f32>,
    _padding21: vec4<f32>,
    _padding22: vec4<f32>,
    _padding23: vec4<f32>,
    _padding24: vec4<f32>,
    _padding25: vec4<f32>,
    _padding26: vec4<f32>,
    _padding27: vec4<f32>,
    _padding28: vec4<f32>,
    _padding29: vec4<f32>,
    _padding30: vec4<f32>,
    _padding31: vec4<f32>,
    _padding32: vec4<f32>,
    _padding33: vec4<f32>,
    _padding34: vec4<f32>,
    _padding35: vec4<f32>,
    _padding36: vec4<f32>,
    _padding37: vec4<f32>,
    _padding38: vec4<f32>,
    _padding39: vec4<f32>,
    _padding40: vec4<f32>,
    _padding41: vec4<f32>,
    _padding42: vec4<f32>,
    _padding43: vec4<f32>,
    _padding44: vec4<f32>,
    _padding45: vec4<f32>,
    _padding46: vec4<f32>,
    _padding47: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0) var<uniform> cave_params: CaveParams;

// Constants
const EPSILON: f32 = 0.0001;

// Hash function for single random value at integer coordinates (no gradients)
fn hash1(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    var h = seed;
    h = h * 374761393u + u32(x);
    h = h * 668265263u + u32(y);
    h = h * 1274126177u + u32(z);
    h ^= h >> 13u;
    h = h * 1274126177u;
    h ^= h >> 16u;

    return f32(h) / f32(0xFFFFFFFFu);
}

// Smooth interpolation (smoothstep / Hermite interpolation)
fn smoothstep_custom(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// 3D value noise - interpolates between random values at lattice points
// No gradients used, just smooth blending between random corner values
fn value_noise_3d(pos: vec3<f32>, seed: u32) -> f32 {
    // Integer and fractional parts
    let ix = i32(floor(pos.x));
    let iy = i32(floor(pos.y));
    let iz = i32(floor(pos.z));

    let fx = pos.x - floor(pos.x);
    let fy = pos.y - floor(pos.y);
    let fz = pos.z - floor(pos.z);

    // Smooth interpolation weights
    let ux = smoothstep_custom(fx);
    let uy = smoothstep_custom(fy);
    let uz = smoothstep_custom(fz);

    // Random values at 8 corners of the cube
    let c000 = hash1(ix, iy, iz, seed);
    let c100 = hash1(ix + 1, iy, iz, seed);
    let c010 = hash1(ix, iy + 1, iz, seed);
    let c110 = hash1(ix + 1, iy + 1, iz, seed);
    let c001 = hash1(ix, iy, iz + 1, seed);
    let c101 = hash1(ix + 1, iy, iz + 1, seed);
    let c011 = hash1(ix, iy + 1, iz + 1, seed);
    let c111 = hash1(ix + 1, iy + 1, iz + 1, seed);

    // Trilinear interpolation with smooth weights
    let x00 = mix(c000, c100, ux);
    let x10 = mix(c010, c110, ux);
    let x01 = mix(c001, c101, ux);
    let x11 = mix(c011, c111, ux);

    let y0 = mix(x00, x10, uy);
    let y1 = mix(x01, x11, uy);

    return mix(y0, y1, uz);
}

// Fractal Brownian Motion - combines multiple octaves of value noise
fn fbm(pos: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;

    for (var i = 0u; i < cave_params.octaves; i = i + 1u) {
        let sample_pos = pos * frequency / cave_params.scale;
        // Use different seed for each octave to avoid correlation
        let octave_seed = cave_params.seed + i * 1337u;
        value = value + amplitude * value_noise_3d(sample_pos, octave_seed);
        max_value = max_value + amplitude;
        amplitude = amplitude * cave_params.persistence;
        frequency = frequency * 2.0;
    }

    // Normalize to 0-1 range
    return value / max_value;
}

// Domain warping - distorts the input coordinates for more organic shapes
fn warp_domain(pos: vec3<f32>) -> vec3<f32> {
    let warp_scale = cave_params.scale * 0.5;
    let warp_strength = cave_params.smoothness * cave_params.scale;

    // Sample noise at offset positions to get warp vectors
    let warp_seed = cave_params.seed + 9999u;
    let wx = value_noise_3d(pos / warp_scale, warp_seed) - 0.5;
    let wy = value_noise_3d(pos / warp_scale + vec3<f32>(31.7, 47.3, 13.1), warp_seed) - 0.5;
    let wz = value_noise_3d(pos / warp_scale + vec3<f32>(73.9, 19.4, 67.2), warp_seed) - 0.5;

    return vec3<f32>(
        pos.x + wx * warp_strength,
        pos.y + wy * warp_strength,
        pos.z + wz * warp_strength,
    );
}

// Sample cave density using value noise with domain warping
fn sample_cave_density(pos: vec3<f32>) -> f32 {
    // Distance from world center (spherical constraint)
    let dist_from_center = length(pos - cave_params.world_center);
    let sphere_sdf = dist_from_center - cave_params.world_radius;

    // Outside sphere = solid (high density)
    if (sphere_sdf > 0.0) {
        return 1.0;
    }

    // Apply domain warping for organic shapes
    let warped_pos = warp_domain(pos);

    // Get base noise value using FBM
    let noise = fbm(warped_pos);

    // Map noise to density based on cave density parameter
    // density parameter controls how much solid rock vs open space
    // Higher density = more solid rock, lower = more open tunnels
    let cave_threshold = clamp(cave_params.density, 0.0, 1.0);

    // Solid rock where noise is above threshold, open tunnels where below
    if (noise > cave_threshold) {
        // Solid rock region - above marching cubes threshold
        let wall_factor = (noise - cave_threshold) / max(1.0 - cave_threshold, 0.001);
        return cave_params.threshold + wall_factor * 0.5;
    } else {
        // Open tunnel/cave space - below marching cubes threshold
        return cave_params.threshold - 0.5;
    }
}

// Compute SDF gradient (normal) using central differences
fn compute_sdf_gradient(pos: vec3<f32>, h: f32) -> vec3<f32> {
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    
    let grad_x = sample_cave_density(pos + dx) - sample_cave_density(pos - dx);
    let grad_y = sample_cave_density(pos + dy) - sample_cave_density(pos - dy);
    let grad_z = sample_cave_density(pos + dz) - sample_cave_density(pos - dz);
    
    let grad = vec3<f32>(grad_x, grad_y, grad_z);
    let len = length(grad);
    
    // Avoid division by zero
    if (len < 0.0001) {
        return vec3<f32>(0.0, 1.0, 0.0); // Default up direction
    }
    
    return grad / len;
}

// Apply force-based collision - pushes cells out of solid rock into cave tunnels
// Optimized: only check cells near cave walls, skip cells safely in open space
fn apply_cave_collision_force(cell_idx: u32, pos: vec3<f32>, radius: f32, mass: f32, dt: f32) {
    if (cave_params.collision_enabled == 0u) {
        return;
    }

    // Quick check: sample density at cell center first
    let center_density = sample_cave_density(pos);
    
    // Early exit: if we're safely in open space (density well below threshold), skip collision
    let safety_margin = 0.2; // 20% safety margin
    let open_space_threshold = cave_params.threshold - 0.5 - safety_margin;
    
    if (center_density <= open_space_threshold) {
        // Cell is safely in open cave space - no collision check needed
        return;
    }
    
    // Cell might be near a cave wall - perform detailed collision check
    // Use fixed threshold for collision detection regardless of cell size
    let radius_threshold = cave_params.threshold;
    
    // If center density > adjusted threshold, we're colliding with cave wall
    if (center_density > radius_threshold) {
        // Compute gradient pointing toward lower density (into cave)
        let gradient_step = max(cave_params.scale * 0.1, radius * 0.5); // Ensure gradient step is meaningful
        let normal = -compute_sdf_gradient(pos, gradient_step);  // Invert normal to point into cave

        // Penetration depth - how far into solid rock we are, accounting for radius
        let penetration = (center_density - radius_threshold) * cave_params.scale;

        if (penetration > 0.0) {
            // Force proportional to penetration depth (Hooke's law)
            // Divide by substeps to maintain stability with multiple substeps
            let force_magnitude = penetration * cave_params.collision_stiffness / f32(cave_params.substeps);
            let force = normal * force_magnitude;

            // F = ma, so a = F/m, and dv = a * dt
            // Use dt/substeps for each substep to maintain physics accuracy
            let velocity_change = force * dt / f32(cave_params.substeps) / mass;

            // Update velocity with force response
            var vel = velocities[cell_idx].xyz;
            vel = vel + velocity_change;

            // Critical damping to prevent oscillation: damping = 2 * sqrt(stiffness * mass)
            // Simplified: use high damping to eliminate bouncing
            let vel_normal_mag = dot(vel, normal);
            if (vel_normal_mag < 0.0) {
                vel = vel - normal * vel_normal_mag * 0.95;
            }

            velocities[cell_idx] = vec4<f32>(vel, velocities[cell_idx].w);
        }
    }
    // If density <= adjusted threshold, cell is safely in cave tunnel - no collision forces needed
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (idx >= cell_count) {
        return;
    }

    // Read position and mass
    let pos_mass = positions[idx];
    let pos = pos_mass.xyz;
    let mass = pos_mass.w;

    // Skip dead cells
    if (mass <= 0.0) {
        return;
    }

    // Calculate radius from mass (mass = 4/3 * pi * r^3)
    let collision_radius = pow(mass * 0.75 / 3.14159265359, 1.0 / 3.0);

    // Apply force-based cave collision (modifies velocity only)
    apply_cave_collision_force(idx, pos, collision_radius, mass, params.delta_time);
}
