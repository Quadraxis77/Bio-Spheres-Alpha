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

struct DivisionAudioParams {
    listener_position: vec3<f32>,
    search_radius_cells: u32,
    grid_resolution: u32,
    max_cells_per_grid: u32,
    max_candidates: u32,
    _pad0: u32,
}

struct DivisionAudioCandidate {
    distance_fixed: u32,
    environment_flags: u32,
    _pad1: u32,
    _pad2: u32,
    position: vec4<f32>,
}

struct WaterGridParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    buoyancy_multiplier: f32,
    water_viscosity: f32,
    _pad1: f32,
}

@group(0) @binding(0)
var<uniform> physics_params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

@group(1) @binding(0)
var<storage, read_write> spatial_grid_counts: array<atomic<u32>>;

@group(1) @binding(1)
var<storage, read_write> spatial_grid_offsets: array<u32>;

@group(1) @binding(2)
var<storage, read_write> cell_grid_indices: array<u32>;

@group(1) @binding(3)
var<storage, read_write> spatial_grid_cells: array<u32>;

@group(1) @binding(4)
var<storage, read> stiffnesses: array<f32>;

@group(1) @binding(5)
var<storage, read> development_addresses: array<vec4<u32>>;

@group(1) @binding(6)
var<storage, read_write> occupied_grid_cells: array<u32>;

@group(1) @binding(7)
var<storage, read_write> occupied_grid_count: array<atomic<u32>>;

@group(1) @binding(8)
var<storage, read_write> spatial_grid_overflow_cells: array<u32>;

@group(1) @binding(9)
var<storage, read_write> spatial_grid_overflow_grid_indices: array<u32>;

@group(1) @binding(10)
var<storage, read_write> spatial_grid_overflow_count: array<atomic<u32>>;

@group(1) @binding(11)
var<storage, read_write> death_flags: array<u32>;

@group(2) @binding(0)
var<uniform> audio_params: DivisionAudioParams;

@group(2) @binding(1)
var<storage, read> division_flags: array<u32>;

@group(2) @binding(2)
var<storage, read_write> candidates: array<DivisionAudioCandidate>;

@group(2) @binding(3)
var<storage, read_write> candidate_count: array<atomic<u32>>;

@group(2) @binding(4)
var<uniform> water_params: WaterGridParams;

@group(2) @binding(5)
var<storage, read> water_bitfield: array<u32>;

const FIXED_POINT_SCALE: f32 = 1000.0;
const EMPTY_DISTANCE: u32 = 0xffffffffu;
const ENV_EMITTER_UNDERWATER: u32 = 1u;
const ENV_LISTENER_UNDERWATER: u32 = 2u;

fn water_voxel_at(world_pos: vec3<f32>) -> bool {
    let res = water_params.grid_resolution;
    if (res == 0u) {
        return false;
    }

    let grid_pos = vec3<f32>(
        (world_pos.x - water_params.grid_origin_x) / water_params.cell_size,
        (world_pos.y - water_params.grid_origin_y) / water_params.cell_size,
        (world_pos.z - water_params.grid_origin_z) / water_params.cell_size
    );

    if (grid_pos.x < 0.0 || grid_pos.x >= f32(res) ||
        grid_pos.y < 0.0 || grid_pos.y >= f32(res) ||
        grid_pos.z < 0.0 || grid_pos.z >= f32(res)) {
        return false;
    }

    let gx = u32(grid_pos.x);
    let gy = u32(grid_pos.y);
    let gz = u32(grid_pos.z);
    let x_groups = (res + 31u) / 32u;
    let x_group = gx / 32u;
    let bit_index = gx % 32u;
    let bitfield_idx = x_group + gy * x_groups + gz * x_groups * res;

    if (bitfield_idx >= arrayLength(&water_bitfield)) {
        return false;
    }

    let bits = water_bitfield[bitfield_idx];
    return (bits & (1u << bit_index)) != 0u;
}

fn append_candidate(distance_fixed: u32, position: vec3<f32>, environment_flags: u32) {
    let slot = atomicAdd(&candidate_count[0], 1u);
    if (slot >= audio_params.max_candidates) {
        return;
    }
    candidates[slot].distance_fixed = distance_fixed;
    candidates[slot].environment_flags = environment_flags;
    candidates[slot].position = vec4<f32>(position, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;

    if (cell_idx >= cell_count_buffer[0] || division_flags[cell_idx] != 1u) {
        return;
    }

    let position_mass = positions_in[cell_idx];
    if (position_mass.w < 0.5) {
        return;
    }

    let delta = position_mass.xyz - audio_params.listener_position;
    let distance_sq = dot(delta, delta);

    let distance_fixed = min(u32(distance_sq * FIXED_POINT_SCALE), EMPTY_DISTANCE - 1u);
    var environment_flags = 0u;
    if (water_voxel_at(position_mass.xyz)) {
        environment_flags = environment_flags | ENV_EMITTER_UNDERWATER;
    }
    if (water_voxel_at(audio_params.listener_position)) {
        environment_flags = environment_flags | ENV_LISTENER_UNDERWATER;
    }
    append_candidate(distance_fixed, position_mass.xyz, environment_flags);
}
