// GPU Spatial Query Compute Shader
// Finds the closest cell intersected by a ray using parallel search with atomic operations
// Workgroup size: 64 threads for parallel processing across all cells
//
// Algorithm:
// 1. Each thread processes one cell
// 2. Perform ray-sphere intersection test for each cell
// 3. Use atomic operations on global result buffer to find globally closest intersection
// 4. Return cell index, distance along ray, and found flag

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

struct SpatialQueryParams {
    ray_origin: vec3<f32>,      // Camera position
    max_distance: f32,          // Maximum ray distance
    ray_direction: vec3<f32>,   // Normalized ray direction
    _pad0: u32,
}

// Result buffer uses atomic u32 for distance (fixed-point) to enable global atomic min
struct SpatialQueryResultAtomic {
    found_cell_index: atomic<u32>,
    distance_fixed: atomic<u32>,  // Fixed-point distance for atomic operations
    found: atomic<u32>,
    _padding: u32,
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Spatial query parameters (Group 1)
@group(1) @binding(0)
var<uniform> query_params: SpatialQueryParams;

// Spatial query result output (Group 2) - uses atomics for global reduction
@group(2) @binding(0)
var<storage, read_write> query_result: SpatialQueryResultAtomic;

const PI: f32 = 3.14159265359;
const FIXED_POINT_SCALE: f32 = 1000.0;
const MAX_DISTANCE_FIXED: u32 = 0xFFFFFFFFu;

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Ray-sphere intersection test
// Returns distance along ray to intersection point, or -1.0 if no intersection
fn ray_sphere_intersect(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_center: vec3<f32>, sphere_radius: f32) -> f32 {
    let oc = ray_origin - sphere_center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return -1.0;  // No intersection
    }
    
    let sqrt_disc = sqrt(discriminant);
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    
    // Return closest positive intersection
    if (t1 > 0.0) {
        return t1;
    } else if (t2 > 0.0) {
        return t2;
    }
    
    return -1.0;  // Intersection behind ray origin
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    
    // Read cell count from GPU buffer
    let cell_count = cell_count_buffer[0];
    
    // Each thread processes one cell
    if (thread_id < cell_count) {
        let cell_pos = positions_in[thread_id].xyz;
        let cell_mass = positions_in[thread_id].w;
        let cell_radius = calculate_radius_from_mass(cell_mass);
        
        // Perform ray-sphere intersection test
        let t = ray_sphere_intersect(
            query_params.ray_origin,
            query_params.ray_direction,
            cell_pos,
            cell_radius
        );
        
        // Check if ray intersects this cell within max distance
        if (t > 0.0 && t <= query_params.max_distance) {
            // Found an intersecting cell - use atomic operations on global result to find closest
            let distance_fixed = u32(t * FIXED_POINT_SCALE);
            
            // Atomic compare-and-swap to find minimum distance globally
            var old_distance = atomicLoad(&query_result.distance_fixed);
            while (distance_fixed < old_distance) {
                let result = atomicCompareExchangeWeak(&query_result.distance_fixed, old_distance, distance_fixed);
                if (result.exchanged) {
                    // Successfully updated distance, now update cell index
                    atomicStore(&query_result.found_cell_index, thread_id);
                    atomicStore(&query_result.found, 1u);
                    break;
                }
                old_distance = result.old_value;
            }
        }
    }
}
