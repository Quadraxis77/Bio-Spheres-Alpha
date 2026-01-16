// Cave Collision XPBD Shader
// 
// Implements GPU-based XPBD collision detection and response for cells colliding with cave mesh.
// Uses spatial grid for efficient triangle lookup to prevent tunneling.
//
// XPBD (Extended Position Based Dynamics) ensures:
// - No tunneling through thin cave walls
// - Stable collision response with multiple substeps
// - Proper constraint satisfaction

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

struct CaveVertex {
    position: vec3<f32>,
    _padding: f32,
}

struct CaveTriangle {
    v0: u32,
    v1: u32,
    v2: u32,
    _padding: u32,
}

struct CaveSpatialCell {
    triangle_count: atomic<u32>,
    triangle_indices: array<u32, 16>, // Max 16 triangles per cell
}

@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0) var<uniform> cave_params: CaveParams;
@group(1) @binding(1) var<storage, read> cave_vertices: array<CaveVertex>;
@group(1) @binding(2) var<storage, read> cave_triangles: array<CaveTriangle>;
@group(1) @binding(3) var<storage, read_write> cave_spatial_grid: array<CaveSpatialCell>;

// Constants
const EPSILON: f32 = 0.0001;
const MAX_COLLISION_ITERATIONS: u32 = 4u;

// Hash function for spatial grid - MUST match cave_spatial_grid_build.wgsl exactly
fn spatial_hash(pos: vec3<f32>) -> i32 {
    let normalized = (pos - cave_params.world_center + vec3<f32>(cave_params.world_radius)) / (cave_params.world_radius * 2.0);
    let grid_pos = vec3<i32>(floor(normalized * f32(cave_params.grid_resolution)));
    let res = i32(cave_params.grid_resolution);
    
    // Clamp to grid bounds
    let clamped = clamp(grid_pos, vec3<i32>(0), vec3<i32>(res - 1));
    
    return clamped.x + clamped.y * res + clamped.z * res * res;
}

// Get triangle vertices
fn get_triangle_vertices(tri_idx: u32) -> array<vec3<f32>, 3> {
    let tri = cave_triangles[tri_idx];
    var verts: array<vec3<f32>, 3>;
    verts[0] = cave_vertices[tri.v0].position;
    verts[1] = cave_vertices[tri.v1].position;
    verts[2] = cave_vertices[tri.v2].position;
    return verts;
}

// Point-triangle distance and closest point
fn point_triangle_distance(p: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec4<f32> {
    // Returns (distance, closest_point.xyz)
    let edge0 = v1 - v0;
    let edge1 = v2 - v0;
    let v0_to_p = p - v0;
    
    let a = dot(edge0, edge0);
    let b = dot(edge0, edge1);
    let c = dot(edge1, edge1);
    let d = dot(edge0, v0_to_p);
    let e = dot(edge1, v0_to_p);
    
    let det = a * c - b * b;
    var s = b * e - c * d;
    var t = b * d - a * e;
    
    if (s + t <= det) {
        if (s < 0.0) {
            if (t < 0.0) {
                // Region 4
                if (d < 0.0) {
                    t = 0.0;
                    s = clamp(-d / a, 0.0, 1.0);
                } else {
                    s = 0.0;
                    t = clamp(-e / c, 0.0, 1.0);
                }
            } else {
                // Region 3
                s = 0.0;
                t = clamp(-e / c, 0.0, 1.0);
            }
        } else if (t < 0.0) {
            // Region 5
            t = 0.0;
            s = clamp(-d / a, 0.0, 1.0);
        } else {
            // Region 0 (inside triangle)
            let inv_det = 1.0 / det;
            s = s * inv_det;
            t = t * inv_det;
        }
    } else {
        if (s < 0.0) {
            // Region 2
            let tmp0 = b + d;
            let tmp1 = c + e;
            if (tmp1 > tmp0) {
                let numer = tmp1 - tmp0;
                let denom = a - 2.0 * b + c;
                s = clamp(numer / denom, 0.0, 1.0);
                t = 1.0 - s;
            } else {
                s = 0.0;
                t = clamp(-e / c, 0.0, 1.0);
            }
        } else if (t < 0.0) {
            // Region 6
            if (a + d > b + e) {
                let numer = c + e - b - d;
                let denom = a - 2.0 * b + c;
                s = clamp(numer / denom, 0.0, 1.0);
                t = 1.0 - s;
            } else {
                t = 0.0;
                s = clamp(-d / a, 0.0, 1.0);
            }
        } else {
            // Region 1
            let numer = c + e - b - d;
            let denom = a - 2.0 * b + c;
            s = clamp(numer / denom, 0.0, 1.0);
            t = 1.0 - s;
        }
    }
    
    let closest = v0 + s * edge0 + t * edge1;
    let dist = length(p - closest);
    
    return vec4<f32>(dist, closest);
}

// Solve XPBD collision constraint
fn solve_cave_collision(cell_idx: u32, pos: vec3<f32>, radius: f32, mass: f32, dt: f32) -> vec3<f32> {
    if (cave_params.collision_enabled == 0u) {
        return pos;
    }
    
    var corrected_pos = pos;
    let substep_dt = dt / f32(cave_params.substeps);
    var collision_count = 0u;
    
    // XPBD substeps for stability
    for (var substep = 0u; substep < cave_params.substeps; substep = substep + 1u) {
        // Check spatial grid cells around the cell position
        let grid_idx = spatial_hash(corrected_pos);
        
        // Check current cell and neighbors
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dz = -1; dz <= 1; dz = dz + 1) {
                    let neighbor_pos = vec3<i32>(
                        (grid_idx % i32(cave_params.grid_resolution)) + dx,
                        ((grid_idx / i32(cave_params.grid_resolution)) % i32(cave_params.grid_resolution)) + dy,
                        (grid_idx / (i32(cave_params.grid_resolution) * i32(cave_params.grid_resolution))) + dz
                    );
                    
                    let res = i32(cave_params.grid_resolution);
                    if (neighbor_pos.x < 0 || neighbor_pos.x >= res ||
                        neighbor_pos.y < 0 || neighbor_pos.y >= res ||
                        neighbor_pos.z < 0 || neighbor_pos.z >= res) {
                        continue;
                    }
                    
                    let neighbor_idx = neighbor_pos.x + neighbor_pos.y * res + neighbor_pos.z * res * res;
                    let cell = &cave_spatial_grid[neighbor_idx];
                    let tri_count = min(atomicLoad(&(*cell).triangle_count), 16u);
                    
                    // Check all triangles in this spatial cell
                    for (var i = 0u; i < tri_count; i = i + 1u) {
                        let tri_idx = (*cell).triangle_indices[i];
                        if (tri_idx >= cave_params.triangle_count) {
                            continue;
                        }
                        
                        let verts = get_triangle_vertices(tri_idx);
                        let result = point_triangle_distance(corrected_pos, verts[0], verts[1], verts[2]);
                        let dist = result.x;
                        let closest = result.yzw;
                        
                        // Collision if cell is close to triangle surface
                        if (dist < radius) {
                            let penetration = radius - dist;
                            
                            // Normal points from surface toward cell
                            var normal: vec3<f32>;
                            if (dist > EPSILON) {
                                normal = normalize(corrected_pos - closest);
                            } else {
                                // Very close - use triangle normal
                                let edge1 = verts[1] - verts[0];
                                let edge2 = verts[2] - verts[0];
                                let tri_normal = normalize(cross(edge1, edge2));
                                // Ensure it points toward cell
                                if (dot(tri_normal, corrected_pos - closest) < 0.0) {
                                    normal = -tri_normal;
                                } else {
                                    normal = tri_normal;
                                }
                            }
                            
                            // XPBD constraint solving with compliance
                            // Compliance = 1 / stiffness, controls how soft the constraint is
                            let alpha = 1.0 / (cave_params.collision_stiffness + 0.0001); // Compliance
                            let dt_substep = dt / f32(cave_params.substeps);
                            
                            // XPBD correction with compliance (makes it soft and stable)
                            let w = 1.0 / mass; // Inverse mass
                            let delta_lambda = penetration / (w + alpha / (dt_substep * dt_substep));
                            let correction = normal * delta_lambda * w;
                            
                            corrected_pos = corrected_pos + correction;
                            
                            // Apply velocity damping (don't add correction to velocity)
                            let vel = velocities[cell_idx].xyz;
                            let vel_normal_mag = dot(vel, normal);
                            if (vel_normal_mag < 0.0) {
                                // Remove velocity toward wall and damp tangential
                                let vel_tangent = vel - vel_normal_mag * normal;
                                velocities[cell_idx] = vec4<f32>(vel_tangent * (1.0 - cave_params.collision_damping), velocities[cell_idx].w);
                            }
                            
                            collision_count = collision_count + 1u;
                        }
                    }
                }
            }
        }
    }
    
    return corrected_pos;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    
    if (idx >= cell_count) {
        return;
    }
    
    // Read position and mass (already updated by position_update)
    let pos_mass = positions[idx];
    let pos = pos_mass.xyz;
    let mass = pos_mass.w;
    
    // Skip dead cells
    if (mass <= 0.0) {
        // Keep the dead cell data as-is
        return;
    }
    
    // Calculate radius from mass (mass = 4/3 * pi * r^3)
    let visual_radius = pow(mass * 0.75 / 3.14159265359, 1.0 / 3.0);
    
    // Calculate adaptive collision radius based on cave scale
    // The spatial grid cell size is (world_radius * 2) / grid_resolution
    // We want the collision radius to be proportional to this cell size
    let world_size = cave_params.world_radius * 2.0;
    let grid_cell_size = world_size / f32(cave_params.grid_resolution);
    
    // Use a collision radius that's a multiple of the grid cell size
    // This ensures we detect collisions with the mesh geometry inside each grid cell
    // The multiplier needs to be large enough to reach the mesh triangles
    let collision_radius = max(visual_radius * 10.0, grid_cell_size * 2.0);
    
    // Solve cave collision with XPBD
    let corrected_pos = solve_cave_collision(idx, pos, collision_radius, mass, params.delta_time);
    
    // Write corrected position back (in-place update)
    positions[idx] = vec4<f32>(corrected_pos, mass);
}
