// GPU Cell Insertion Compute Shader
// Inserts cells directly into GPU buffers without CPU state management
// Uses atomic operations for safe slot allocation
//
// Algorithm:
// 1. Check for available free slots from dead cells
// 2. If free slot available: reuse it, else allocate new slot
// 3. Check capacity limits (MAX_CELLS)
// 4. Initialize all cell properties in GPU buffers
// 5. Write to ALL THREE triple buffer sets for positions/velocities/rotations
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
    
    // Cell type + initial reserve + initial nutrients (16 bytes)
    cell_type: u32,
    // Initial embryocyte reserve (x1000 fixed-point).
    // 0 means "use default for cell_type" (65535000 for Embryocyte/Gametocyte, 0 otherwise).
    initial_reserve: u32,
    // Initial nutrients (x1000 fixed-point).
    // 0 means "use default" (100000 = full). Non-zero caps starting nutrients.
    initial_nutrients: u32,
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

// Ring buffer for free slot recycling (replaces lifecycle_counts/free_slot_indices)
@group(1) @binding(4)
var<storage, read_write> free_slot_ring: array<u32>;

// Ring state: [head, tail, next_slot_id, reservation_count]
@group(1) @binding(5)
var<storage, read_write> ring_state: array<atomic<u32>>;

// Triple-buffered cell rotations (all 3 sets for write)
@group(1) @binding(1)
var<storage, read_write> rotations_0: array<vec4<f32>>;

@group(1) @binding(2)
var<storage, read_write> rotations_1: array<vec4<f32>>;

@group(1) @binding(3)
var<storage, read_write> rotations_2: array<vec4<f32>>;

// Triple-buffered angular velocities (all 3 sets for write) - zeroed on insertion
@group(1) @binding(6)
var<storage, read_write> angular_velocities_0: array<vec4<f32>>;

@group(1) @binding(7)
var<storage, read_write> angular_velocities_1: array<vec4<f32>>;

@group(1) @binding(8)
var<storage, read_write> angular_velocities_2: array<vec4<f32>>;

// Cell state buffers for division system (single buffers, not triple-buffered)
@group(2) @binding(0)
var<storage, read_write> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read_write> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read_write> split_nutrient_thresholds: array<f32>;

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

// Nutrients buffer: fixed-point i32 with scale 1000 (100.0 nutrients = 100000)
@group(2) @binding(16)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Per-cell genome orientation: pure genome-derived orientation (no physics perturbation)
@group(2) @binding(17)
var<storage, read_write> genome_orientations: array<vec4<f32>>;

// Embryocyte reserve buffer (one atomic<u32> per cell, x1000 fixed-point)
// Initialized to 65535000 for Embryocytes (cell_type == 10), 0 for all others.
@group(2) @binding(18)
var<storage, read_write> embryocyte_reserves: array<atomic<u32>>;

// Per-cell development address: [organism_id, lineage_hash_lo, lineage_hash_hi, depth_branch].
// Root insertions use cell_id + 1 as organism_id so cell IDs 0 and 1 cannot collide.
@group(2) @binding(19)
var<storage, read_write> development_addresses: array<vec4<u32>>;

// Per-cell parent lineage hash. Root cells write 0 (no parent).
@group(2) @binding(20)
var<storage, read_write> parent_lineage_hashes_out: array<vec2<u32>>;

@group(2) @binding(21)
var<storage, read_write> organism_cell_ids: array<u32>;

struct U64 {
    lo: u32,
    hi: u32,
};

fn shr64(value: U64, amount: u32) -> U64 {
    if (amount == 0u) {
        return value;
    }
    if (amount < 32u) {
        return U64((value.lo >> amount) | (value.hi << (32u - amount)), value.hi >> amount);
    }
    if (amount < 64u) {
        return U64(value.hi >> (amount - 32u), 0u);
    }
    return U64(0u, 0u);
}

fn xor_shr64(value: U64, amount: u32) -> U64 {
    let shifted = shr64(value, amount);
    return U64(value.lo ^ shifted.lo, value.hi ^ shifted.hi);
}

fn mul_hi_u32(a: u32, b: u32) -> u32 {
    var t = (a & 0xFFFFu) * (b & 0xFFFFu);
    var k = t >> 16u;
    t = (a >> 16u) * (b & 0xFFFFu) + k;
    let w1 = t & 0xFFFFu;
    let w2 = t >> 16u;
    t = (a & 0xFFFFu) * (b >> 16u) + w1;
    return (a >> 16u) * (b >> 16u) + w2 + (t >> 16u);
}

fn mul64_low(value: U64, mul_lo: u32, mul_hi: u32) -> U64 {
    let lo = value.lo * mul_lo;
    let hi = mul_hi_u32(value.lo, mul_lo) + value.lo * mul_hi + value.hi * mul_lo;
    return U64(lo, hi);
}

fn mix_development_hash(value: U64) -> U64 {
    var x = xor_shr64(value, 30u);
    x = mul64_low(x, 0x1CE4E5B9u, 0xBF58476Du);
    x = xor_shr64(x, 27u);
    x = mul64_low(x, 0x133111EBu, 0x94D049BBu);
    return xor_shr64(x, 31u);
}

fn development_root_hash(genome_id: u32, mode_index: u32) -> U64 {
    let seed = U64(0x7F4A7C15u ^ (genome_id << 16u) ^ mode_index, 0x9E3779B9u ^ (genome_id >> 16u));
    return mix_development_hash(seed);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Only thread 0 should execute (single workgroup dispatch)
    if (global_id.x != 0u || global_id.y != 0u || global_id.z != 0u) {
        return;
    }
    
    // Slot allocation via ring buffer handles capacity checks correctly:
    // - Recycled slots from dead cells are always valid (already within capacity)
    // - New slots via next_slot_id are checked against cell_capacity below
    // NOTE: We do NOT check cell_count_buffer[0] here because it is a high-water
    // mark that never decreases when cells die. Checking it would permanently block
    // insertion once the peak count reaches capacity, even with many dead/recycled slots.
    let effective_capacity = min(params.cell_capacity, arrayLength(&positions_0));
    let previous_live = atomicAdd(&cell_count_buffer[1], 1u);
    if (previous_live >= effective_capacity) {
        atomicSub(&cell_count_buffer[1], 1u);
        return;
    }
    
    // Use ring buffer for slot allocation (matches lifecycle_unified.wgsl)
    // Try to pop a recycled slot from the ring buffer first
    var slot: u32;
    var used_free_slot = false;
    
    // Try ring buffer first
    let head = atomicAdd(&ring_state[0], 1u);
    let tail = atomicLoad(&ring_state[1]);
    
    if (head < tail) {
        // Got a recycled slot from ring buffer
        let ring_idx = head % 262144u; // RING_BUFFER_CAPACITY
        slot = free_slot_ring[ring_idx];
        used_free_slot = true;

        if (slot >= effective_capacity) {
            atomicSub(&cell_count_buffer[1], 1u);
            return;
        }

        // Mark this slot as alive again
        death_flags[slot] = 0u;
    } else {
        // Ring buffer empty, undo the head increment
        atomicSub(&ring_state[0], 1u);

        // Allocate new slot using ring_state[2] (next_slot_id)
        slot = atomicAdd(&ring_state[2], 1u);

        // Check capacity
        if (slot >= effective_capacity) {
            // Rollback and fail
            atomicSub(&ring_state[2], 1u);
            atomicSub(&cell_count_buffer[1], 1u);
            return;
        }
    }

    // Ensure slot is within bounds
    if (slot >= effective_capacity) {
        atomicSub(&cell_count_buffer[1], 1u);
        return;
    }

    // Update total cell count (high water mark) unconditionally.
    // This is critical for recycled slots: after a GPU reset (cell_count_buffer zeroed),
    // a recycled slot's index may exceed the current high-water mark of 0.
    // Without this, lifecycle shaders (death_scan, division_scan) would iterate
    // 0 cells and never process the newly inserted cell.
    atomicMax(&cell_count_buffer[0], slot + 1u);
    
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
    
    // Zero angular velocity in ALL THREE triple buffer sets.
    // Recycled slots retain the dead cell's angular velocity; new slots may have
    // uninitialized data. Both cases cause phantom spin on newly inserted cells.
    angular_velocities_0[slot] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    angular_velocities_1[slot] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    angular_velocities_2[slot] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    // Initialize genome orientation (same as initial rotation for newly inserted cells)
    genome_orientations[slot] = insertion_params.rotation;
    
    // Initialize cell state for division system (single buffers)
    birth_times[slot] = insertion_params.birth_time;
    split_intervals[slot] = insertion_params.split_interval;
    split_nutrient_thresholds[slot] = insertion_params.split_mass; // split_mass param holds converted nutrient threshold
    split_counts[slot] = 0u; // New cell hasn't split yet
    
    // Convert max_splits: -1 (infinite) -> 0 (unlimited in GPU)
    let gpu_max_splits = insertion_params.max_splits;
    max_splits[slot] = gpu_max_splits;
    
    genome_ids[slot] = insertion_params.genome_id;
    mode_indices[slot] = insertion_params.mode_index;
    
    // Use provided cell_id or atomically generate new one
    let assigned_cell_id = insertion_params.cell_id;
    var final_cell_id: u32;
    if (assigned_cell_id == 0u) {
        // Generate new cell ID atomically
        final_cell_id = atomicAdd(&next_cell_id[0], 1u);
        cell_ids[slot] = final_cell_id;
    } else {
        // Use provided cell ID
        final_cell_id = assigned_cell_id;
        cell_ids[slot] = assigned_cell_id;
        // Update next_cell_id if necessary
        let current_next = atomicLoad(&next_cell_id[0]);
        if (assigned_cell_id >= current_next) {
            atomicMax(&next_cell_id[0], assigned_cell_id + 1u);
        }
    }

    let root_hash = development_root_hash(insertion_params.genome_id, insertion_params.mode_index);
    development_addresses[slot] = vec4<u32>(final_cell_id + 1u, root_hash.lo, root_hash.hi, 0u);
    parent_lineage_hashes_out[slot] = vec2<u32>(0u, 0u);
    organism_cell_ids[slot] = 1u;
    
    // Initialize cell properties from genome mode
    nutrient_gain_rates[slot] = insertion_params.nutrient_gain_rate;
    max_cell_sizes[slot] = insertion_params.max_cell_size;
    stiffnesses[slot] = insertion_params.stiffness;
    
    // Initialize lifecycle flags
    death_flags[slot] = 0u; // Alive
    division_flags[slot] = 0u; // Not dividing
    
    // Initialize nutrients.
    // If initial_nutrients is non-zero, use it directly (e.g. gamete merge where the
    // combined reserve doesn't cover a full nutrient pool).
    // Otherwise default to 100.0 full = 100000 in fixed-point.
    let nutrient_value = select(100000i, i32(insertion_params.initial_nutrients), insertion_params.initial_nutrients != 0u);
    atomicStore(&nutrients_buffer[slot], nutrient_value);
    
    // Initialize cell type (0 = Test, 1 = Flagellocyte, etc.)
    cell_types[slot] = insertion_params.cell_type;
    
    // Initialize reserve.
    // If insertion_params.initial_reserve is non-zero, use it directly (e.g. gamete merge).
    // Otherwise fall back to cell-type default: Embryocytes and Gametocytes start full,
    // all other types start at 0.
    let is_storage = (insertion_params.cell_type == 10u || insertion_params.cell_type == 13u);
    let default_reserve = select(0u, 65535000u, is_storage);
    let reserve_value = select(default_reserve, insertion_params.initial_reserve, insertion_params.initial_reserve != 0u);
    atomicStore(&embryocyte_reserves[slot], reserve_value);

    // Live cell count was reserved before slot allocation.
}
