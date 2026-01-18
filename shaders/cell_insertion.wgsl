// GPU Cell Insertion Compute Shader
// Inserts cells directly into GPU buffers without CPU state management
// Uses atomic operations for safe slot allocation
//
// Algorithm:
// 1. Atomically increment cell count to get a free slot
// 2. Check capacity limits (MAX_CELLS)
// 3. Initialize all cell properties in GPU buffers
// 4. Write to ALL THREE triple buffer sets for positions/velocities/rotations
//
// Workgroup size: (1,1,1) for atomic safety as per requirements

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

struct CellInsertionParams {
    // Cell position and physics properties (16 bytes)
    position: vec3<f32>,
    mass: f32,
    
    // Cell velocity (16 bytes)
    velocity: vec3<f32>,
    _pad0: f32,
    
    // Cell rotation quaternion (16 bytes)
    rotation: vec4<f32>,
    
    // Cell genome and mode info (16 bytes)
    genome_id: u32,
    mode_index: u32,
    birth_time: f32,
    _pad1: f32,
    
    // Cell division properties (16 bytes)
    split_interval: f32,
    split_mass: f32,
    stiffness: f32,
    radius: f32,
    
    // Cell state properties (16 bytes)
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    max_splits: u32,
    cell_id: u32,
    
    // Cell type (16 bytes with padding)
    cell_type: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

@group(0) @binding(0)
var<uniform> params: PhysicsParams;

// Triple-buffered positions (all 3 sets for write)
@group(0) @binding(1)
var<storage, read_write> positions_0: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> positions_1: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_2: array<vec4<f32>>;

// Triple-buffered velocities (all 3 sets for write)
@group(0) @binding(4)
var<storage, read_write> velocities_0: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> velocities_1: array<vec4<f32>>;

@group(0) @binding(6)
var<storage, read_write> velocities_2: array<vec4<f32>>;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(7)
var<storage, read_write> cell_count_buffer: array<atomic<u32>>;

// Cell insertion parameters uniform buffer
@group(1) @binding(0)
var<uniform> insertion_params: CellInsertionParams;

// Triple-buffered cell rotations (all 3 sets for write)
@group(1) @binding(1)
var<storage, read_write> rotations_0: array<vec4<f32>>;

@group(1) @binding(2)
var<storage, read_write> rotations_1: array<vec4<f32>>;

@group(1) @binding(3)
var<storage, read_write> rotations_2: array<vec4<f32>>;

// Cell state buffers for division system (single buffers, not triple-buffered)
@group(2) @binding(0)
var<storage, read_write> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read_write> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read_write> split_masses: array<f32>;

@group(2) @binding(3)
var<storage, read_write> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read_write> split_ready_frame: array<i32>;

@group(2) @binding(5)
var<storage, read_write> max_splits: array<u32>;

@group(2) @binding(6)
var<storage, read_write> genome_ids: array<u32>;

@group(2) @binding(7)
var<storage, read_write> mode_indices: array<u32>;

@group(2) @binding(8)
var<storage, read_write> cell_ids: array<u32>;

@group(2) @binding(9)
var<storage, read_write> next_cell_id: array<atomic<u32>>;

@group(2) @binding(10)
var<storage, read_write> nutrient_gain_rates: array<f32>;

@group(2) @binding(11)
var<storage, read_write> max_cell_sizes: array<f32>;

@group(2) @binding(12)
var<storage, read_write> stiffnesses: array<f32>;

@group(2) @binding(13)
var<storage, read_write> death_flags: array<u32>;

@group(2) @binding(14)
var<storage, read_write> division_flags: array<u32>;

@group(2) @binding(15)
var<storage, read_write> cell_types: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Only thread 0 should execute (single workgroup dispatch)
    if (global_id.x != 0u || global_id.y != 0u || global_id.z != 0u) {
        return;
    }
    
    // Atomically get current cell count and increment
    let current_count = atomicLoad(&cell_count_buffer[0]);
    
    // Check capacity limits
    if (current_count >= params.cell_capacity) {
        // Capacity exceeded, cannot insert cell
        return;
    }
    
    // Atomically allocate slot by incrementing cell count
    let slot = atomicAdd(&cell_count_buffer[0], 1u);
    
    // Double-check capacity after atomic increment
    if (slot >= params.cell_capacity) {
        // Rollback the increment if we exceeded capacity
        atomicSub(&cell_count_buffer[0], 1u);
        return;
    }
    
    // Initialize position and mass in ALL THREE triple buffer sets
    let position_mass = vec4<f32>(
        insertion_params.position.x,
        insertion_params.position.y,
        insertion_params.position.z,
        insertion_params.mass
    );
    positions_0[slot] = position_mass;
    positions_1[slot] = position_mass;
    positions_2[slot] = position_mass;
    
    // Initialize velocity in ALL THREE triple buffer sets
    let velocity_data = vec4<f32>(
        insertion_params.velocity.x,
        insertion_params.velocity.y,
        insertion_params.velocity.z,
        0.0
    );
    velocities_0[slot] = velocity_data;
    velocities_1[slot] = velocity_data;
    velocities_2[slot] = velocity_data;
    
    // Initialize rotation in ALL THREE triple buffer sets
    rotations_0[slot] = insertion_params.rotation;
    rotations_1[slot] = insertion_params.rotation;
    rotations_2[slot] = insertion_params.rotation;
    
    // Initialize cell state for division system (single buffers)
    birth_times[slot] = insertion_params.birth_time;
    split_intervals[slot] = insertion_params.split_interval;
    split_masses[slot] = insertion_params.split_mass;
    split_counts[slot] = 0u; // New cell hasn't split yet
    
    // Convert max_splits: -1 (infinite) -> 0 (unlimited in GPU)
    let gpu_max_splits = insertion_params.max_splits;
    max_splits[slot] = gpu_max_splits;
    
    genome_ids[slot] = insertion_params.genome_id;
    mode_indices[slot] = insertion_params.mode_index;
    
    // Use provided cell_id or atomically generate new one
    let assigned_cell_id = insertion_params.cell_id;
    if (assigned_cell_id == 0u) {
        // Generate new cell ID atomically
        cell_ids[slot] = atomicAdd(&next_cell_id[0], 1u);
    } else {
        // Use provided cell ID
        cell_ids[slot] = assigned_cell_id;
        // Update next_cell_id if necessary
        let current_next = atomicLoad(&next_cell_id[0]);
        if (assigned_cell_id >= current_next) {
            atomicMax(&next_cell_id[0], assigned_cell_id + 1u);
        }
    }
    
    // Initialize cell properties from genome mode
    nutrient_gain_rates[slot] = insertion_params.nutrient_gain_rate;
    max_cell_sizes[slot] = insertion_params.max_cell_size;
    stiffnesses[slot] = insertion_params.stiffness;
    
    // Initialize lifecycle flags
    death_flags[slot] = 0u; // Alive
    division_flags[slot] = 0u; // Not dividing
    
    // Initialize cell type (0 = Test, 1 = Flagellocyte, etc.)
    cell_types[slot] = insertion_params.cell_type;
    
    // Update live cell count (same as total for now)
    atomicStore(&cell_count_buffer[1], slot + 1u);
}
