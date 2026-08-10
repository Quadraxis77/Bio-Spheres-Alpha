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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    position: vec4<f32>,
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

const FIXED_POINT_SCALE: f32 = 1000.0;
const EMPTY_DISTANCE: u32 = 0xffffffffu;

fn grid_index(coords: vec3<i32>, resolution: i32) -> u32 {
    return u32(coords.x + coords.y * resolution + coords.z * resolution * resolution);
}

fn grid_coords(index: u32, resolution: u32) -> vec3<i32> {
    let z = index / (resolution * resolution);
    let rem = index % (resolution * resolution);
    let y = rem / resolution;
    let x = rem % resolution;
    return vec3<i32>(i32(x), i32(y), i32(z));
}

fn listener_grid_coords() -> vec3<i32> {
    let grid_pos = (audio_params.listener_position + physics_params.world_size * 0.5)
        / physics_params.grid_cell_size;
    let max_coord = i32(audio_params.grid_resolution) - 1;
    return clamp(vec3<i32>(grid_pos), vec3<i32>(0), vec3<i32>(max_coord));
}

fn append_candidate(distance_fixed: u32, position: vec3<f32>) {
    let slot = atomicAdd(&candidate_count[0], 1u);
    if (slot >= audio_params.max_candidates) {
        return;
    }
    candidates[slot].distance_fixed = distance_fixed;
    candidates[slot].position = vec4<f32>(position, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let radius = audio_params.search_radius_cells;
    let diameter = radius * 2u + 1u;
    let bucket_count = diameter * diameter * diameter;
    let slot_count = max(audio_params.max_cells_per_grid, 1u);
    let bucket_slot_threads = bucket_count * slot_count;
    let linear = global_id.x;
    let center = listener_grid_coords();
    let resolution = i32(audio_params.grid_resolution);

    var cell_idx = 0u;

    if (linear < bucket_slot_threads) {
        let bucket_linear = linear / slot_count;
        let slot = linear % slot_count;
        let dz = i32(bucket_linear / (diameter * diameter)) - i32(radius);
        let rem = bucket_linear % (diameter * diameter);
        let dy = i32(rem / diameter) - i32(radius);
        let dx = i32(rem % diameter) - i32(radius);

        let coords = center + vec3<i32>(dx, dy, dz);
        if (any(coords < vec3<i32>(0)) || any(coords >= vec3<i32>(resolution))) {
            return;
        }

        let grid_idx = grid_index(coords, resolution);
        let count = min(atomicLoad(&spatial_grid_counts[grid_idx]), audio_params.max_cells_per_grid);
        if (slot >= count) {
            return;
        }

        cell_idx = spatial_grid_cells[grid_idx * audio_params.max_cells_per_grid + slot];
    } else {
        let overflow_idx = linear - bucket_slot_threads;
        let overflow_count = min(atomicLoad(&spatial_grid_overflow_count[0]), physics_params.cell_capacity);
        if (overflow_idx >= overflow_count) {
            return;
        }

        let overflow_grid_idx = spatial_grid_overflow_grid_indices[overflow_idx];
        let coords = grid_coords(overflow_grid_idx, audio_params.grid_resolution);
        let delta_grid = abs(coords - center);
        if (any(delta_grid > vec3<i32>(i32(radius)))) {
            return;
        }

        cell_idx = spatial_grid_overflow_cells[overflow_idx];
    }

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
    append_candidate(distance_fixed, position_mass.xyz);
}
