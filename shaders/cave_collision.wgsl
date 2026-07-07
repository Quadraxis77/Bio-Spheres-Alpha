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
    geothermal_enabled: u32,
    geothermal_count: u32,
    geothermal_placement_mode: u32,
    geothermal_lower_hemisphere: u32,
    geothermal_gravity_mode: u32,
    geothermal_gravity: f32,
    geothermal_length: f32,
    geothermal_width: f32,
    geothermal_depth: f32,
    geothermal_back_margin: f32,
    geothermal_top_margin: f32,
    geothermal_heat_output: f32,
    geothermal_heat_radius: f32,
    geothermal_glow_strength: f32,
    geothermal_glow_radius: f32,
    geothermal_glow_color: vec3<f32>,
    isolated_chunk_cull_volume: f32,
    mesh_smoothing_iterations: u32,
    mesh_smoothing_factor: f32,
    mesh_smooth_normals: u32,
    flat_ground_enabled: u32,
}

@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cell_count_buffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> torque_accum_x: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> torque_accum_y: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> torque_accum_z: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read> angular_velocities: array<vec4<f32>>;

@group(1) @binding(0) var<uniform> cave_params: CaveParams;
@group(1) @binding(1) var<storage, read> solid_mask: array<u32>;

// Constants
const EPSILON: f32 = 0.0001;
const FIXED_POINT_SCALE: f32 = 1000.0;
const ROLLING_CONTACT_FRICTION: f32 = 0.18;
const CAVE_RESTITUTION: f32 = 0.08;
const CAVE_RESTING_SPEED: f32 = 2.0;
const CAVE_MAX_CORRECTION_PER_STEP: f32 = 0.18;
const CAVE_CONTACT_SLOP: f32 = 0.12;
const CAVE_POSITION_CORRECTION_FRACTION: f32 = 0.22;
const CAVE_REST_SPEED: f32 = 0.08;

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

fn flat_ground_surface_height(pos: vec3<f32>) -> f32 {
    let base_height = cave_params.world_center.y - cave_params.world_radius / 3.0;
    let phase = f32(cave_params.seed) * 0.013;
    let amplitude = clamp(cave_params.world_radius * 0.008, 0.75, 2.4);

    let swell = sin(pos.x * 0.045 + phase) * 0.65
        + sin(pos.z * 0.038 - phase * 0.7) * 0.45
        + sin((pos.x + pos.z) * 0.025 + phase * 0.31) * 0.35;
    let ripples = sin(pos.x * 0.18 + pos.z * 0.07 + swell * 0.4 + phase * 1.7) * 0.22;

    return base_height + (swell / 1.45 + ripples) * amplitude;
}

fn flat_ground_density(pos: vec3<f32>) -> f32 {
    if (cave_params.flat_ground_enabled == 0u) {
        return -1.0;
    }

    let depth = flat_ground_surface_height(pos) - pos.y;
    if (depth < 0.0) {
        return -1.0;
    }

    return cave_params.threshold + clamp(depth / max(cave_params.scale, 0.001), 0.0, 0.5);
}

// Sample cave density using value noise with domain warping
fn sample_cave_density(pos: vec3<f32>) -> f32 {
    // Distance from world center (spherical constraint)
    let dist_from_center = length(pos - cave_params.world_center);
    let sphere_sdf = dist_from_center - (cave_params.world_radius + 3.0);

    // Outside sphere = solid (high density)
    if (sphere_sdf >= 0.0) {
        return 1.0;
    }

    let ground_density = flat_ground_density(pos);
    if (ground_density >= 0.0) {
        return ground_density;
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

fn sample_collision_density(pos: vec3<f32>) -> f32 {
    return sample_cave_density(pos);
}

fn raw_sdf_gradient(pos: vec3<f32>, h: f32) -> vec3<f32> {
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    
    let grad_x = sample_collision_density(pos + dx) - sample_collision_density(pos - dx);
    let grad_y = sample_collision_density(pos + dy) - sample_collision_density(pos - dy);
    let grad_z = sample_collision_density(pos + dz) - sample_collision_density(pos - dz);
    
    return vec3<f32>(grad_x, grad_y, grad_z);
}

// Compute SDF gradient (normal) using central differences. The cave density is
// thresholded rather than a true distance field, so use a small local probe for
// stable contact normals and fall back to wider probes only when fully embedded.
fn compute_sdf_gradient(pos: vec3<f32>, radius: f32) -> vec3<f32> {
    let local_h = max(radius * 0.75, 0.35);
    var grad = raw_sdf_gradient(pos, local_h);
    var len = length(grad);

    if (len < 0.0001) {
        grad = raw_sdf_gradient(pos, max(radius * 1.5, 1.0));
        len = length(grad);
    }

    if (len < 0.0001) {
        grad = raw_sdf_gradient(pos, max(cave_params.scale * 0.04, 2.0));
        len = length(grad);
    }

    // Near the outer world shell, all six density samples can land in the
    // same saturated solid value and produce a zero gradient. A fixed +Y
    // fallback makes collision correction point -Y everywhere, creating one
    // configuration-independent sticking sliver on the lower hemisphere.
    // Use the sphere's local outward gradient instead, so the caller's
    // negation always points back into the playable volume.
    if (len < 0.0001) {
        let radial = pos - cave_params.world_center;
        let radial_len = length(radial);
        if (radial_len > 0.0001) {
            return radial / radial_len;
        }
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    
    return grad / len;
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

fn estimate_penetration_depth(pos: vec3<f32>, radius: f32, density_overlap: f32, center_is_solid: bool) -> f32 {
    let h = max(radius * 0.5, 0.25);
    let grad_len = length(raw_sdf_gradient(pos, h));

    if (grad_len > 0.0001) {
        let density_per_world_unit = grad_len / (2.0 * h);
        let estimated_depth = density_overlap / max(density_per_world_unit, 0.0001);
        let max_surface_depth = max(radius * 0.65, 0.25);
        let max_center_depth = max(radius, 0.5);
        return min(estimated_depth, select(max_surface_depth, max_center_depth, center_is_solid));
    }

    return select(0.0, radius * 0.25, center_is_solid);
}

// Apply position-based collision - directly moves cells out of solid rock into cave tunnels.
// Center-based contact avoids preemptively pushing cells away while their center
// is still in open space, which otherwise makes them stand off from the surface.
fn apply_cave_collision_force(cell_idx: u32, pos: vec3<f32>, radius: f32, mass: f32, dt: f32) {
    if (cave_params.collision_enabled == 0u) {
        return;
    }

    let center_density = sample_collision_density(pos);
    let radius_threshold = cave_params.threshold;

    if (center_density > radius_threshold) {
        // Compute gradient pointing toward lower density (into open cave space)
        let normal = -compute_sdf_gradient(pos, radius);  // Points into cave (away from wall)

        // Penetration estimate. Cave density is a smooth scalar field, not a
        // true SDF, so estimate world-space depth from local density gradient
        // instead of scaling by cave scale or a voxel size.
        let density_overlap = center_density - radius_threshold;
        let penetration = estimate_penetration_depth(
            pos,
            radius,
            density_overlap,
            true
        );

        var vel = velocities[cell_idx].xyz;
        let vel_into_wall = dot(vel, -normal);

        if (penetration > CAVE_CONTACT_SLOP) {
            // Soft depenetration. Cave density is not a true signed-distance
            // field, so `penetration + radius` behaves like a teleport near
            // high-gradient/voxelized walls. Leave a small contact slop and
            // correct a bounded fraction so cells can settle and roll.
            let correction_distance = min(
                max(penetration - CAVE_CONTACT_SLOP, 0.0) * CAVE_POSITION_CORRECTION_FRACTION,
                CAVE_MAX_CORRECTION_PER_STEP
            );
            let correction = normal * correction_distance;
            let corrected_pos = pos + correction;
            positions[cell_idx] = vec4<f32>(corrected_pos, positions[cell_idx].w);
        }

        // Redirect inward speed into open space. For resting-speed contact,
        // remove the normal component without restitution so the cell does not
        // bounce between neighboring interpolated occupancy values.
        if (vel_into_wall > 0.0) {
            let inward_removal = clamp(cave_params.collision_damping, 0.0, 1.0);
            let restitution = select(0.0, CAVE_RESTITUTION, vel_into_wall > CAVE_RESTING_SPEED);
            let redirected_speed = vel_into_wall * (inward_removal + restitution);
            vel = vel + normal * redirected_speed;
        }

        if (penetration > 0.0 || vel_into_wall > 0.0) {
            // Rolling / sliding friction against the cave wall. The cave wall is
            // stationary, so contact-point slip should spin the cell instead of
            // only damping linear tangent velocity.
            let r_contact = -normal * radius;
            let omega = angular_velocities[cell_idx].xyz;
            let v_contact = vel + cross(omega, r_contact);
            let v_tangent = v_contact - dot(v_contact, normal) * normal;
            let tangent_speed = length(v_tangent);
            if (tangent_speed > 0.0001) {
                let friction_dir = -v_tangent / tangent_speed;
                let normal_force_mag = min(cave_params.collision_stiffness * penetration, 1000.0);
                let friction_mag = min(
                    ROLLING_CONTACT_FRICTION * normal_force_mag,
                    tangent_speed * cave_params.collision_stiffness * 0.01
                );
                let friction_force = friction_dir * friction_mag;
                let contact_torque = cross(r_contact, friction_force);
                atomicAdd(&torque_accum_x[cell_idx], i32(contact_torque.x * FIXED_POINT_SCALE));
                atomicAdd(&torque_accum_y[cell_idx], i32(contact_torque.y * FIXED_POINT_SCALE));
                atomicAdd(&torque_accum_z[cell_idx], i32(contact_torque.z * FIXED_POINT_SCALE));
            }

            // Preserve linear tangent motion. This pass is re-run after every
            // adhesion constraint iteration, so even light per-pass damping
            // compounds into severe wall friction.
            let vel_normal_comp = dot(vel, normal);
            let vel_tangent = vel - normal * vel_normal_comp;
            let vel_normal_out = select(max(vel_normal_comp, 0.0), 0.0, abs(vel_normal_comp) < CAVE_REST_SPEED);
            vel = normal * vel_normal_out + vel_tangent;
            velocities[cell_idx] = vec4<f32>(vel, velocities[cell_idx].w);
        }
    }
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

    let collision_radius = calculate_radius_from_mass(mass);

    // Apply force-based cave collision (modifies velocity only)
    apply_cave_collision_force(idx, pos, collision_radius, mass, params.delta_time);
}
