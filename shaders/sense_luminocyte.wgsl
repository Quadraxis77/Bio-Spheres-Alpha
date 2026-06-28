// Luminocyte Sense Shader
// Photocytes scan their 3×3×3 spatial-grid neighborhood for glowing luminocytes
// and gain nutrients proportional to accumulated brightness / distance².
//
// Runs once per physics step, after photocyte_light.wgsl (which writes glow_flags).

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

struct PhotocyteParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    mass_per_second_full_light: f32,
    geothermal_mass_per_second_full_light: f32,
    min_light_threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

// Group 0: physics (reuses photocyte_physics_layout)
@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cell_count_buffer: array<u32>;

// Group 1: sense-specific data
@group(1) @binding(0) var<uniform> photocyte_params: PhotocyteParams;
@group(1) @binding(1) var<storage, read> cell_types: array<u32>;
@group(1) @binding(2) var<storage, read_write> nutrients_buffer: array<atomic<i32>>;
@group(1) @binding(3) var<storage, read> death_flags: array<u32>;
@group(1) @binding(4) var<storage, read> split_nutrient_thresholds: array<f32>;
@group(1) @binding(5) var<storage, read> glow_flags: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read> spatial_grid_counts: array<u32>;
@group(1) @binding(7) var<storage, read> spatial_grid_cells: array<u32>;

const PHOTOCYTE_TYPE: u32 = 3u;
const MAX_CELLS_PER_GRID: u32 = 16u;
const FIXED_POINT_SCALE: f32 = 1000.0;

fn float_to_fixed(value: f32) -> i32 { return i32(value * FIXED_POINT_SCALE); }
fn fixed_to_float(value: i32) -> f32 { return f32(value) / FIXED_POINT_SCALE; }

fn grid_coords_to_index(x: i32, y: i32, z: i32) -> u32 {
    let res = params.grid_resolution;
    return u32(x + y * res + z * res * res);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) { return; }
    if (death_flags[cell_idx] == 1u) { return; }
    if (cell_types[cell_idx] != PHOTOCYTE_TYPE) { return; }

    let current_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));
    if (current_nutrients < 1.0) { return; }

    let max_nutrients = min(split_nutrient_thresholds[cell_idx], 200.0) * 2.0;
    if (current_nutrients >= max_nutrients) { return; }

    let pos = positions[cell_idx].xyz;

    // World-to-spatial-grid using PhysicsParams grid (resolution 64)
    let half_world = params.world_size * 0.5;
    let gc_x = i32((pos.x + half_world) / params.grid_cell_size);
    let gc_y = i32((pos.y + half_world) / params.grid_cell_size);
    let gc_z = i32((pos.z + half_world) / params.grid_cell_size);
    let res = params.grid_resolution;

    var accumulated_lux = 0.0;

    for (var dz = -1; dz <= 1; dz++) {
        let nz = gc_z + dz;
        if (nz < 0 || nz >= res) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let ny = gc_y + dy;
            if (ny < 0 || ny >= res) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let nx = gc_x + dx;
                if (nx < 0 || nx >= res) { continue; }
                let grid_idx = grid_coords_to_index(nx, ny, nz);
                let cnt = min(spatial_grid_counts[grid_idx], MAX_CELLS_PER_GRID);
                let base = grid_idx * MAX_CELLS_PER_GRID;
                for (var i = 0u; i < cnt; i++) {
                    let other_idx = spatial_grid_cells[base + i];
                    if (other_idx == cell_idx) { continue; }
                    let glow = glow_flags[other_idx];
                    if (glow.w <= 0.001) { continue; }
                    let diff = pos - positions[other_idx].xyz;
                    // 1/d² falloff; clamp denominator so touching cells don't overflow
                    let dist_sq = max(dot(diff, diff), 0.25);
                    accumulated_lux += glow.w / dist_sq;
                }
            }
        }
    }

    if (accumulated_lux <= 0.0) { return; }

    // Scale to nutrient gain matching photocyte sunlight rate; clamp at sun-equivalent
    let clamped_lux = clamp(accumulated_lux, 0.0, 1.0);
    let nutrient_rate = photocyte_params.mass_per_second_full_light * 100.0 * clamped_lux;
    let nutrient_gain = min(
        nutrient_rate * params.delta_time,
        max(max_nutrients - current_nutrients, 0.0)
    );
    if (nutrient_gain > 0.0) {
        atomicAdd(&nutrients_buffer[cell_idx], float_to_fixed(nutrient_gain));
    }
}
