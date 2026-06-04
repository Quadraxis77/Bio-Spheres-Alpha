// Lifecycle Division Execute Shader - Ring Buffer Version
// Stage 2: Execute cell division using pre-allocated slots from ring buffer
// Workgroup size: 128 threads for cell operations
//
// Input:
// - division_flags[cell_idx] = 1 for cells that should divide
// - division_slot_assignments[cell_idx] = slot index for child B
//
// Output:
// - Child A overwrites parent slot with half mass
// - Child B created in assigned slot with half mass
// - Adhesion connections updated for both children

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

// Cell type behavior flags
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    _padding: array<u32, 10>,
}

// Adhesion connection structure (104 bytes)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _align_pad: vec2<u32>,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
};

const PI: f32 = 3.14159265359;
const BOND_FLAG_BARRIER_BALL: u32 = 2u;

// Physics bind group (group 0)
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

@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<atomic<u32>>;

// Lifecycle bind group (group 1) - all read_write to match Rust layout
@group(1) @binding(0)
var<storage, read_write> death_flags: array<u32>;

@group(1) @binding(1)
var<storage, read_write> division_flags: array<u32>;

@group(1) @binding(2)
var<storage, read_write> free_slot_ring: array<u32>;

@group(1) @binding(3)
var<storage, read_write> division_slot_assignments: array<u32>;

@group(1) @binding(4)
var<storage, read_write> ring_state: array<atomic<u32>>;

// Cell state bind group (group 2)
@group(2) @binding(0)
var<storage, read_write> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read_write> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read_write> split_nutrient_thresholds: array<f32>;

@group(2) @binding(3)
var<storage, read_write> split_counts: array<u32>;

// Nutrients buffer: fixed-point i32 with scale 1000 (100.0 nutrients = 100000)
@group(2) @binding(31)
var<storage, read_write> nutrients_buffer: array<atomic<i32>>;

// Fixed-point conversion
const FIXED_POINT_SCALE: f32 = 1000.0;

fn float_to_fixed(value: f32) -> i32 {
    return i32(value * FIXED_POINT_SCALE);
}

fn fixed_to_float(value: i32) -> f32 {
    return f32(value) / FIXED_POINT_SCALE;
}

// Derive mass from nutrients: mass = 1.0 + nutrients / 100.0
fn nutrients_to_mass(nutrients: f32) -> f32 {
    return 1.0 + nutrients / 100.0;
}

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
var<storage, read> rotations_in: array<vec4<f32>>;

@group(2) @binding(14)
var<storage, read_write> rotations_out: array<vec4<f32>>;

// genome_mode_data split into 5 vec4 sub-buffers (bindings 15-19)
@group(2) @binding(15)
var<storage, read> genome_mode_data_v0: array<vec4<f32>>; // child_a_orientation
@group(2) @binding(16)
var<storage, read> genome_mode_data_v1: array<vec4<f32>>; // child_b_orientation
@group(2) @binding(17)
var<storage, read> genome_mode_data_v2: array<vec4<f32>>; // child_a_split_orientation
@group(2) @binding(18)
var<storage, read> genome_mode_data_v3: array<vec4<f32>>; // child_b_split_orientation
@group(2) @binding(19)
var<storage, read> genome_mode_data_v4: array<vec4<f32>>; // split_rotation_quat (XYZW)

@group(2) @binding(20)
var<storage, read> parent_make_adhesion_flags: array<u32>;

@group(2) @binding(21)
var<storage, read> child_mode_indices: array<vec2<i32>>;

// mode_properties split into 5 vec4 sub-buffers (bindings 22-26)
@group(2) @binding(22)
var<storage, read> mode_properties_v0: array<vec4<f32>>; // [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
@group(2) @binding(23)
var<storage, read> mode_properties_v1: array<vec4<f32>>; // [split_mass, nutrient_priority, swim_force, prioritize_when_low]
@group(2) @binding(24)
var<storage, read> mode_properties_v2: array<vec4<f32>>; // [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
@group(2) @binding(25)
var<storage, read> mode_properties_v3: array<vec4<f32>>; // [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
@group(2) @binding(26)
var<storage, read> mode_properties_v4: array<vec4<f32>>; // [max_adhesions, mode_a_after_splits, mode_b_after_splits, padding]

@group(2) @binding(27)
var<storage, read_write> cell_types: array<u32>;

@group(2) @binding(28)
var<storage, read> mode_cell_types: array<u32>;

// Child A keep adhesion flags: one bool per mode (stored as u32)
@group(2) @binding(29)
var<storage, read> child_a_keep_adhesion_flags: array<u32>;

// Child B keep adhesion flags: one bool per mode (stored as u32)
@group(2) @binding(30)
var<storage, read> child_b_keep_adhesion_flags: array<u32>;

// Child A after-split keep adhesion flags
@group(2) @binding(33)
var<storage, read> child_a_after_split_keep_adhesion_flags: array<u32>;

// Child B after-split keep adhesion flags
@group(2) @binding(34)
var<storage, read> child_b_after_split_keep_adhesion_flags: array<u32>;

// (parent_genome * split_rotation * child_orientation) without physics perturbation.
// Used by adhesion shaders so structures are defined purely by genome data.
@group(2) @binding(32)
var<storage, read_write> genome_orientations: array<vec4<f32>>;

// Per-cell signal flags for signal-conditional child mode routing
@group(2) @binding(35)
var<storage, read> signal_flags_read: array<u32>;

// Per-mode signal-conditional settings for child mode routing
// v1: [apoptosis_signal_threshold, apoptosis_signal_invert, signal_child_a_channel, signal_child_a_threshold]
// v2: [signal_child_a_mode_above, signal_child_a_mode_below, signal_child_b_channel, signal_child_b_threshold]
// v3: [signal_child_b_mode_above, signal_child_b_mode_below, mode_switch_signal_channel, mode_switch_signal_threshold]
@group(2) @binding(36)
var<storage, read> signal_settings_v1_read: array<vec4<f32>>;

@group(2) @binding(37)
var<storage, read> signal_settings_v2_read: array<vec4<f32>>;

@group(2) @binding(38)
var<storage, read> signal_settings_v3_read: array<vec4<f32>>;

// Per-cell Embryocyte reserve (u32, halved on division)
@group(2) @binding(39)
var<storage, read_write> embryocyte_reserves: array<atomic<u32>>;

// Per-cell development address: [organism_id, lineage_hash_lo, lineage_hash_hi, depth_branch].
// depth_branch packs lineage_depth in the high 16 bits and branch_slot in the low 16 bits.
@group(2) @binding(41)
var<storage, read_write> development_addresses: array<vec4<u32>>;

// Per-cell parent lineage hash: [parent_hash_lo, parent_hash_hi]. Written at birth, never changed.
@group(2) @binding(42)
var<storage, read_write> parent_lineage_hashes_out: array<vec2<u32>>;

// Per-mode flag: 1 if this is the genome's initial mode, 0 otherwise.
// A child whose mode equals the genome's initial mode begins a new organism.
@group(2) @binding(43)
var<storage, read> is_initial_mode: array<u32>;

@group(2) @binding(44)
var<storage, read_write> organism_cell_ids: array<u32>;

// Adhesion bind group (group 3)
@group(3) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

@group(3) @binding(1)
var<storage, read_write> cell_adhesion_indices: array<atomic<i32>>;

@group(3) @binding(2)
var<storage, read_write> next_adhesion_id: array<atomic<u32>>;

// Free adhesion slot stack (for reuse of freed adhesion slots)
@group(3) @binding(3)
var<storage, read_write> free_adhesion_slots: array<u32>;

// Adhesion counts: [0] = total, [1] = live, [2] = free_top, [3] = padding
@group(3) @binding(4)
var<storage, read_write> adhesion_counts: array<atomic<u32>>;

// Allocate an adhesion slot, preferring to reuse freed slots.
// Uses compare-and-swap on adhesion_counts[2] (free_top) to safely pop from the stack.
// Falls back to monotonic next_adhesion_id if the free stack is empty.
// Returns 0xFFFFFFFF if at capacity.
fn allocate_adhesion_slot() -> u32 {
    // Try to pop from free adhesion slot stack first
    loop {
        let free_top = atomicLoad(&adhesion_counts[2]);
        if (free_top == 0u) {
            break; // Stack empty, fall back to monotonic
        }
        let result = atomicCompareExchangeWeak(&adhesion_counts[2], free_top, free_top - 1u);
        if (result.exchanged) {
            let slot = free_adhesion_slots[free_top - 1u];
            // Increment live count
            atomicAdd(&adhesion_counts[1], 1u);
            return slot;
        }
        // CAS failed (another thread popped), retry
    }
    
    // Free stack empty - fall back to monotonic allocation
    let slot = atomicAdd(&next_adhesion_id[0], 1u);
    if (slot < arrayLength(&adhesion_connections)) {
        // Increment live count
        atomicAdd(&adhesion_counts[1], 1u);
        return slot;
    }
    // At capacity
    return 0xFFFFFFFFu;
}

// Return a freed adhesion slot to the free stack so the scaffold can reuse it.
fn free_adhesion_slot(slot: u32) {
    let free_pos = atomicAdd(&adhesion_counts[2], 1u);
    if (free_pos < arrayLength(&free_adhesion_slots)) {
        free_adhesion_slots[free_pos] = slot;
    }
    // Decrement live count (wrapping sub via wrapping add of max u32)
    atomicAdd(&adhesion_counts[1], 0xFFFFFFFFu);
}

// Helper functions
fn rotate_vector_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let qvec = q.xyz;
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

// Quaternion conjugate (inverse for unit quaternions)
fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// Rotate vector by inverse quaternion (for local space transformation)
fn rotate_vector_by_quat_inverse(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return rotate_vector_by_quat(v, quat_conjugate(q));
}

fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Zone classification constants
const ZONE_A: u32 = 0u;  // Negative dot (opposite to split)
const ZONE_B: u32 = 1u;  // Positive dot (same as split)
const ZONE_C: u32 = 2u;  // Equatorial (perpendicular to split)

// Dynamic equatorial zone: 3 deg at split_ratio=0.5, 22 deg at split_ratio=0.3 or 0.7
// Matches CPU EQUATORIAL_THRESHOLD_DEGREES_MIN/MAX in adhesion_zones.rs
const EQUATORIAL_DEG_MIN: f32 = 3.0;
const EQUATORIAL_DEG_MAX: f32 = 22.0;

// Compute dynamic equatorial zone width in degrees based on split_ratio
fn compute_equatorial_degrees(split_ratio: f32) -> f32 {
    let deviation = abs(split_ratio - 0.5);
    let t = min(deviation / 0.2, 1.0);
    return EQUATORIAL_DEG_MIN + (EQUATORIAL_DEG_MAX - EQUATORIAL_DEG_MIN) * t;
}

// Compute ratio shift: 0 at 0.5, +0.4 at 0.7, -0.4 at 0.3
fn compute_ratio_shift(split_ratio: f32) -> f32 {
    return 2.0 * split_ratio - 1.0;
}

// Classify adhesion bond direction relative to split direction with dynamic equatorial zone.
// Matches CPU classify_bond_direction() exactly.
fn classify_zone(anchor_direction: vec3<f32>, split_direction: vec3<f32>, split_ratio: f32) -> u32 {
    let dot_product = dot(normalize(anchor_direction), normalize(split_direction));
    
    // Shift the split plane based on split_ratio
    let ratio_shift = compute_ratio_shift(split_ratio);
    let shifted_dot = dot_product - ratio_shift;
    
    // Dynamic equatorial threshold based on split_ratio
    let equatorial_degrees = compute_equatorial_degrees(split_ratio);
    let equatorial_threshold = sin(equatorial_degrees * PI / 180.0);
    
    // Zone classification using shifted dot product
    if (abs(shifted_dot) <= equatorial_threshold) {
        return ZONE_C;  // Equatorial band
    } else if (shifted_dot > 0.0) {
        return ZONE_B;  // Positive shifted dot -> Child A
    } else {
        return ZONE_A;  // Negative shifted dot -> Child B
    }
}

// Deterministic pseudo-random rotation for cell division
// Matches CPU u32 arithmetic exactly for visual variety
const RNG_SEED: u32 = 12345u;

fn pseudo_random_rotation(cell_id: u32) -> vec4<f32> {
    let hash1 = (cell_id * 2654435761u + RNG_SEED) % 1000000u;
    let hash2 = (cell_id * 1597334677u + RNG_SEED * 3u) % 1000000u;
    
    // Generate angle in range [0.001, 0.1] radians
    let angle = f32(hash1) / 1000000.0 * 0.099 + 0.001;
    
    // Generate random axis
    let x = f32(hash2) / 1000000.0 * 2.0 - 1.0;
    let y = f32((hash2 * 7u) % 1000000u) / 1000000.0 * 2.0 - 1.0;
    let z = f32((hash2 * 13u) % 1000000u) / 1000000.0 * 2.0 - 1.0;
    
    var axis = vec3<f32>(x, y, z);
    let len_sq = dot(axis, axis);
    if (len_sq > 0.001) {
        axis = normalize(axis);
    } else {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    }
    
    // Quaternion from axis-angle: (axis * sin(angle/2), cos(angle/2))
    let half_angle = angle * 0.5;
    let s = sin(half_angle);
    let c = cos(half_angle);
    return vec4<f32>(axis * s, c);
}

fn quat_from_z_to_dir(dir: vec3<f32>) -> vec4<f32> {
    // q = (cross(Z, dir), dot(Z, dir) + 1) normalized
    // cross((0,0,1), dir) = (-dir.y, dir.x, 0)
    let w_scalar = dir.z + 1.0;
    if (w_scalar < 0.0001) {
        // Nearly opposite: 180 deg rotation around Y axis
        return vec4<f32>(0.0, 1.0, 0.0, 0.0);
    }
    return normalize(vec4<f32>(-dir.y, dir.x, 0.0, w_scalar));
}

fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

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

fn derive_development_hash(
    parent_hash: U64,
    parent_mode: u32,
    child_mode: u32,
    branch_slot: u32
) -> U64 {
    let mixed = U64(
        parent_hash.lo ^ (parent_mode << 24u) ^ (child_mode << 8u) ^ branch_slot,
        parent_hash.hi ^ (parent_mode >> 8u) ^ (child_mode >> 24u)
    );
    return mix_development_hash(mixed);
}

fn derive_organism_cell_id(parent_id: u32, branch_slot: u32) -> u32 {
    if (parent_id <= 0x7FFFFFFFu) {
        let base = parent_id * 2u;
        if (branch_slot != 2u || base < 0xFFFFFFFFu) {
            return base + select(0u, 1u, branch_slot == 2u);
        }
    }

    let mixed = mix_development_hash(U64((parent_id << 16u) ^ branch_slot, parent_id >> 16u));
    return max(mixed.lo, 1u);
}

const MAX_ADHESIONS_PER_CELL: u32 = 20u;

// Cosine threshold for deciding when a new sibling bond occupies the same direction
// as an existing inherited bond on the same cell (~8 degrees). Must match the CPU
// constant ANCHOR_OVERLAP_COS in src/cell/adhesion.rs so both scenes disconnect identically.
const ANCHOR_OVERLAP_COS: f32 = 0.99;

// Calculate child anchor direction using geometric approach (matches reference)
// child_pos_parent_frame: child position in parent's local frame
// neighbor_pos_parent_frame: neighbor position in parent's local frame (from anchor * distance)
// child_orientation_delta: child's orientation relative to parent (from genome mode)
fn calculate_child_anchor_direction(
    child_pos_parent_frame: vec3<f32>,
    neighbor_pos_parent_frame: vec3<f32>,
    child_orientation_delta: vec4<f32>
) -> vec3<f32> {
    // Direction from child to neighbor in parent frame
    let direction_to_neighbor = normalize(neighbor_pos_parent_frame - child_pos_parent_frame);
    // Transform to child's local space using inverse of orientation delta
    let inv_delta = quat_conjugate(child_orientation_delta);
    return normalize(rotate_vector_by_quat(direction_to_neighbor, inv_delta));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = atomicLoad(&cell_count_buffer[0]);
    
    if (cell_idx >= cell_count) {
        return;
    }
    
    // === EMBRYOCYTE RELEASE (division_flags == 2u) ===
    // Drop all active adhesions. The cell remains alive; reserve burn begins next frame.
    if (division_flags[cell_idx] == 2u) {
        let adh_base = cell_idx * MAX_ADHESIONS_PER_CELL;
        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            let adh_idx_signed = atomicLoad(&cell_adhesion_indices[adh_base + i]);
            if (adh_idx_signed < 0) { continue; }
            let adh_idx = u32(adh_idx_signed);
            let conn = adhesion_connections[adh_idx];
            if (conn.is_active != 0u) {
                adhesion_connections[adh_idx].is_active = 0u;
                free_adhesion_slot(adh_idx);
                // Clear this bond from the other cell's index list
                let other_cell = select(conn.cell_a_index, conn.cell_b_index, conn.cell_a_index == cell_idx);
                let other_base = other_cell * MAX_ADHESIONS_PER_CELL;
                for (var j = 0u; j < MAX_ADHESIONS_PER_CELL; j++) {
                    let cas = atomicCompareExchangeWeak(&cell_adhesion_indices[other_base + j], adh_idx_signed, -1);
                    if (cas.exchanged || cas.old_value != adh_idx_signed) { break; }
                }
            }
            atomicStore(&cell_adhesion_indices[adh_base + i], -1);
        }
        // Reset birth_time so the timer trigger starts fresh on the next attachment.
        // Without this, age = current_time - original_birth_time would already exceed
        // release_timer on every subsequent attachment, causing instant re-release.
        birth_times[cell_idx] = params.current_time;
        division_flags[cell_idx] = 0u;
        return;
    }

    // Check if this cell is dividing normally
    if (division_flags[cell_idx] != 1u) {
        return;
    }

    // Get the pre-assigned slot for child B
    let child_b_slot = division_slot_assignments[cell_idx];
    
    // Validate slot assignment
    let effective_capacity = min(params.cell_capacity, arrayLength(&positions_out));
    if (child_b_slot >= effective_capacity) {
        atomicSub(&cell_count_buffer[1], 1u);
        division_flags[cell_idx] = 0u;
        return; // Invalid slot, skip division
    }

    // Clear stale lifecycle flags on the child B slot immediately.
    // The recycled slot may have belonged to a cell that had division_flags=1 set
    // (it was about to divide when it died). If left stale, division_scan would
    // see division_flags[child_b_slot]==1 on the very next frame and try to
    // allocate another slot for the newborn child - before it has any nutrients -
    // producing a phantom grandchild that flickers for one frame then dies.
    // Clearing here (before any child B writes) is safe: this thread owns child_b_slot
    // exclusively because division_scan already popped it from the free-slot ring.
    division_flags[child_b_slot] = 0u;
    
    // Parent state - read from OUTPUT buffer (physics results)
    let parent_pos = positions_out[cell_idx].xyz;
    let parent_rotation = rotations_in[cell_idx];
    let parent_split_count = split_counts[cell_idx];
    
    // Get parent's nutrients from nutrients_buffer
    let parent_nutrients = fixed_to_float(atomicLoad(&nutrients_buffer[cell_idx]));

    // Read parent's Embryocyte reserve; will be halved for each child
    let parent_reserve = atomicLoad(&embryocyte_reserves[cell_idx]);
    let child_reserve = parent_reserve >> 1u;

    // Get parent's mode index for looking up child orientations and split direction
    let parent_mode_idx = mode_indices[cell_idx];
    
    // Read parent's split_ratio from mode_properties_v2 (.y = split_ratio)
    let parent_split_ratio = mode_properties_v2[parent_mode_idx].y;
    
    // Read child orientations and split orientations from genome mode data.
    // Layout: [child_a (vec4), child_b (vec4), child_a_split (vec4), child_b_split (vec4),
    //          split_rotation_quat (vec4 XYZW)]
    // Slot 4 is the pre-computed split rotation quaternion from_euler(YXZ, yaw, pitch, 0).
    // It is NOT a direction vector - do not reconstruct via quat_from_z_to_dir().
    let child_a_orientation = genome_mode_data_v0[parent_mode_idx];
    let child_b_orientation = genome_mode_data_v1[parent_mode_idx];
    let child_a_split_orientation = genome_mode_data_v2[parent_mode_idx];
    let child_b_split_orientation = genome_mode_data_v3[parent_mode_idx];
    let split_rotation_quat = genome_mode_data_v4[parent_mode_idx]; // full quaternion, XYZW

    // Derive the local-space split direction from the quaternion (rotate Z by split_rotation).
    // This is used only for child positioning and zone classification.
    let split_dir_local = rotate_vector_by_quat(vec3<f32>(0.0, 0.0, 1.0), split_rotation_quat);

    // Check if max_splits is reached and use split orientations if so
    let will_reach_max_splits = (max_splits[cell_idx] != 0xFFFFFFFFu) && ((parent_split_count + 1u) >= max_splits[cell_idx]);

    let child_a_orientation_final = select(child_a_orientation, child_a_split_orientation, will_reach_max_splits);
    let child_b_orientation_final = select(child_b_orientation, child_b_split_orientation, will_reach_max_splits);

    // Calculate child PHYSICS rotations: parent * split_rotation * child_orientation_final
    // Uses the exact quaternion, not a reconstructed approximation.
    let child_a_rotation = quat_multiply(parent_rotation, quat_multiply(split_rotation_quat, child_a_orientation_final));
    let child_b_rotation = quat_multiply(parent_rotation, quat_multiply(split_rotation_quat, child_b_orientation_final));

    // Calculate child GENOME orientations: parent_genome * split_rotation * child_orientation_final
    // These are pure genome-derived orientations without any physics perturbation.
    // Used by adhesion shaders for anchor transformation and twist constraints.
    let parent_genome_orientation = genome_orientations[cell_idx];
    let child_a_genome_orientation = quat_multiply(parent_genome_orientation, quat_multiply(split_rotation_quat, child_a_orientation_final));
    let child_b_genome_orientation = quat_multiply(parent_genome_orientation, quat_multiply(split_rotation_quat, child_b_orientation_final));

    // Track sibling adhesion slot radius for offset calculation (derive mass from nutrients)
    let parent_mass = nutrients_to_mass(parent_nutrients);
    let parent_radius = calculate_radius_from_mass(parent_mass);

    // Split nutrients using split_ratio from mode (matching CPU)
    let split_ratio = clamp(parent_split_ratio, 0.0, 1.0);
    let child_a_nutrients = parent_nutrients * split_ratio;
    let child_b_nutrients = parent_nutrients * (1.0 - split_ratio);

    // Derive child masses from nutrients
    let child_a_mass = nutrients_to_mass(child_a_nutrients);
    let child_b_mass = nutrients_to_mass(child_b_nutrients);

    // Transform split direction from local to world space for child positioning
    let split_dir_world = normalize(rotate_vector_by_quat(split_dir_local, parent_rotation));

    // Apply a tiny positional jitter to break degenerate symmetric splits,
    // matching the CPU preview path in src/cell/division.rs.
    // The parent cell ID drives the hash so each cell gets a unique, stable jitter.
    let parent_cell_id = cell_ids[cell_idx];
    let jitter_quat = pseudo_random_rotation(parent_cell_id);
    let split_dir = normalize(rotate_vector_by_quat(split_dir_world, jitter_quat));

    // 75% overlap: offset_distance = parent_radius * 0.25
    let offset = parent_radius * 0.25;
    let child_a_pos = parent_pos + split_dir * offset;
    let child_b_pos = parent_pos - split_dir * offset;
    
    // Get child mode indices
    let child_modes = child_mode_indices[parent_mode_idx];
    var child_a_mode_idx = u32(max(child_modes.x, 0));
    var child_b_mode_idx = u32(max(child_modes.y, 0));
    
    // If max_splits is reached, use mode_a/b_after_splits if set (matching CPU behavior)
    // mode_a_after_splits is at mode_properties_v4.y (index 17)
    // mode_b_after_splits is at mode_properties_v4.z (index 18)
    // Negative value means "use normal child mode"
    if (will_reach_max_splits) {
        let parent_props_4 = mode_properties_v4[parent_mode_idx];
        let mode_a_after = parent_props_4.y;
        let mode_b_after = parent_props_4.z;
        if (mode_a_after >= 0.0) {
            child_a_mode_idx = u32(mode_a_after);
        }
        if (mode_b_after >= 0.0) {
            child_b_mode_idx = u32(mode_b_after);
        }
    }
    
    // CRITICAL: Bounds check child mode indices to prevent invalid array access
    // If mode index is out of bounds, clamp to parent mode (safe fallback)
    let mode_count = arrayLength(&mode_cell_types);

    // Signal-conditional child mode routing:
    // If the parent mode has signal_child_a/b_channel >= 0, check the parent cell's signal
    // and override child mode based on whether signal is above or below threshold.
    if (parent_mode_idx < arrayLength(&signal_settings_v1_read)) {
        let ss_v1 = signal_settings_v1_read[parent_mode_idx];
        let ss_v2 = signal_settings_v2_read[parent_mode_idx];
        let ss_v3 = signal_settings_v3_read[parent_mode_idx];
        
        // Child A signal routing: ss_v1.z = channel, ss_v1.w = threshold
        // ss_v2.x = mode_above, ss_v2.y = mode_below
        let child_a_channel = ss_v1.z;
        if (child_a_channel >= 8.0) {
            // 16 channels per cell: signal_flags_read[cell_idx * 16 + channel]
            let child_a_ch = clamp(u32(child_a_channel), 8u, 15u);
            let raw_signal = signal_flags_read[cell_idx * 16u + child_a_ch];
            let signal_value = f32(raw_signal & 0x7FFu);
            let has_signal = raw_signal > 0u;
            
            let child_a_threshold = ss_v1.w;
            let child_a_mode_above = i32(ss_v2.x);
            let child_a_mode_below = i32(ss_v2.y);
            
            if (has_signal && signal_value >= child_a_threshold) {
                if (child_a_mode_above >= 0) {
                    child_a_mode_idx = u32(child_a_mode_above);
                }
            } else {
                if (child_a_mode_below >= 0) {
                    child_a_mode_idx = u32(child_a_mode_below);
                }
            }
        }
        
        // Child B signal routing: ss_v2.z = channel, ss_v2.w = threshold
        // ss_v3.x = mode_above, ss_v3.y = mode_below
        let child_b_channel = ss_v2.z;
        if (child_b_channel >= 8.0) {
            let child_b_ch = clamp(u32(child_b_channel), 8u, 15u);
            let raw_signal_b = signal_flags_read[cell_idx * 16u + child_b_ch];
            let signal_value_b = f32(raw_signal_b & 0x7FFu);
            let has_signal_b = raw_signal_b > 0u;
            
            let child_b_threshold = ss_v2.w;
            let child_b_mode_above = i32(ss_v3.x);
            let child_b_mode_below = i32(ss_v3.y);
            
            if (has_signal_b && signal_value_b >= child_b_threshold) {
                if (child_b_mode_above >= 0) {
                    child_b_mode_idx = u32(child_b_mode_above);
                }
            } else {
                if (child_b_mode_below >= 0) {
                    child_b_mode_idx = u32(child_b_mode_below);
                }
            }
        }
    }

    if (child_a_mode_idx >= mode_count) {
        child_a_mode_idx = parent_mode_idx;
    }
    if (child_b_mode_idx >= mode_count) {
        child_b_mode_idx = parent_mode_idx;
    }

    // Embryocyte children cannot be Embryocytes (prevents reserve-doubling chains).
    // If the assigned child mode is cell_type 10, fall back to mode 0.
    if (mode_cell_types[parent_mode_idx] == 10u) {
        if (mode_cell_types[child_a_mode_idx] == 10u) {
            child_a_mode_idx = 0u;
        }
        if (mode_cell_types[child_b_mode_idx] == 10u) {
            child_b_mode_idx = 0u;
        }
    }

    let parent_development = development_addresses[cell_idx];
    let base_org_id = select(parent_development.x, parent_cell_id, parent_development.x == 0u);
    let parent_lineage_hash = U64(parent_development.y, parent_development.z);
    let parent_organism_cell_id = max(organism_cell_ids[cell_idx], 1u);
    let parent_lineage_depth = parent_development.w >> 16u;
    let child_lineage_depth = min(parent_lineage_depth + 1u, 0xFFFFu);
    let child_a_lineage_hash = derive_development_hash(
        parent_lineage_hash,
        parent_mode_idx,
        child_a_mode_idx,
        1u
    );
    let child_b_lineage_hash = derive_development_hash(
        parent_lineage_hash,
        parent_mode_idx,
        child_b_mode_idx,
        2u
    );

    // Allocate both child cell IDs up front so we can derive organism IDs before any writes.
    // A non-initial parent that routes into the genome's initial mode starts a new organism.
    // Initial-mode self-renewal stays in the same organism with deterministic child IDs.
    let child_a_id = atomicAdd(&next_cell_id[0], 1u);
    let child_b_id = atomicAdd(&next_cell_id[0], 1u);
    let parent_is_initial_mode = parent_mode_idx < arrayLength(&is_initial_mode) && is_initial_mode[parent_mode_idx] != 0u;
    let child_a_is_new_org = !parent_is_initial_mode && child_a_mode_idx < arrayLength(&is_initial_mode) && is_initial_mode[child_a_mode_idx] != 0u;
    let child_b_is_new_org = !parent_is_initial_mode && child_b_mode_idx < arrayLength(&is_initial_mode) && is_initial_mode[child_b_mode_idx] != 0u;
    // child_a uses child_b_id as its new org seed; child_b uses child_b_id+1.
    // These are safe unique values: child_b_id is freshly allocated and no cell has it as org_id.
    let child_a_org_id = select(base_org_id, child_b_id, child_a_is_new_org);
    let child_b_org_id = select(base_org_id, child_b_id + 1u, child_b_is_new_org);
    let child_a_organism_cell_id = select(derive_organism_cell_id(parent_organism_cell_id, 1u), 1u, child_a_is_new_org);
    let child_b_organism_cell_id = select(derive_organism_cell_id(parent_organism_cell_id, 2u), 1u, child_b_is_new_org);

    // === Create Child A (overwrites parent slot) ===
    let parent_velocity = velocities_in[cell_idx];
    positions_out[cell_idx] = vec4<f32>(child_a_pos, child_a_mass);
    velocities_out[cell_idx] = vec4<f32>(parent_velocity.xyz, 0.0);

    cell_ids[cell_idx] = child_a_id;
    organism_cell_ids[cell_idx] = child_a_organism_cell_id;

    // Physics rotation = genome rotation chain (no random perturbation for determinism)
    rotations_out[cell_idx] = child_a_rotation;
    genome_orientations[cell_idx] = child_a_genome_orientation;
    development_addresses[cell_idx] = vec4<u32>(
        child_a_org_id,
        child_a_lineage_hash.lo,
        child_a_lineage_hash.hi,
        (child_lineage_depth << 16u) | 1u
    );
    parent_lineage_hashes_out[cell_idx] = vec2<u32>(parent_lineage_hash.lo, parent_lineage_hash.hi);

    birth_times[cell_idx] = params.current_time;
    mode_indices[cell_idx] = child_a_mode_idx;
    
    // Split count: reset if mode changes OR if the after-splits routing fires with an
    // explicit mode_a/b_after_splits that is DIFFERENT from the normal child mode
    // (lifecycle transition - stem cycling). Setting after-splits to the same mode
    // as the normal child is identical to leaving it at default (-1): no reset.
    let parent_props_4_for_count = mode_properties_v4[parent_mode_idx];
    let normal_child_a_mode_idx = u32(max(child_mode_indices[parent_mode_idx].x, 0));
    let normal_child_b_mode_idx = u32(max(child_mode_indices[parent_mode_idx].y, 0));
    let after_splits_a_fires = will_reach_max_splits
        && parent_props_4_for_count.y >= 0.0
        && u32(parent_props_4_for_count.y) != normal_child_a_mode_idx;
    let after_splits_b_fires = will_reach_max_splits
        && parent_props_4_for_count.z >= 0.0
        && u32(parent_props_4_for_count.z) != normal_child_b_mode_idx;
    if (child_a_mode_idx != parent_mode_idx || after_splits_a_fires) {
        split_counts[cell_idx] = 0u;
    } else {
        split_counts[cell_idx] = parent_split_count + 1u;
    }
    
    // Read Child A's properties from its mode
    let child_a_props_0 = mode_properties_v0[child_a_mode_idx];
    let child_a_props_1 = mode_properties_v1[child_a_mode_idx];
    let child_a_props_2 = mode_properties_v2[child_a_mode_idx];
    
    // Only Test cells (cell_type == 0) auto-generate nutrients
    let child_a_cell_type = mode_cell_types[child_a_mode_idx];
    let child_a_nutrient_rate = select(0.0, child_a_props_0.x, child_a_cell_type == 0u);
    nutrient_gain_rates[cell_idx] = child_a_nutrient_rate;
    
    cell_types[cell_idx] = child_a_cell_type;
    max_cell_sizes[cell_idx] = child_a_props_0.y;
    stiffnesses[cell_idx] = child_a_props_0.z;
    split_intervals[cell_idx] = child_a_props_0.w;
    // Convert split_mass to nutrient threshold: (split_mass - 1.0) * 100.0
    let child_a_split_mass = child_a_props_1.x;
    split_nutrient_thresholds[cell_idx] = (child_a_split_mass - 1.0) * 100.0;
    max_splits[cell_idx] = select(u32(child_a_props_2.x), 0xFFFFFFFFu, child_a_props_2.x < 0.0);
    
    // Set child A nutrients and reserve (reserve halved from parent)
    atomicStore(&nutrients_buffer[cell_idx], float_to_fixed(child_a_nutrients));
    atomicStore(&embryocyte_reserves[cell_idx], child_reserve);

    // === Create Child B (in assigned slot) ===
    positions_out[child_b_slot] = vec4<f32>(child_b_pos, child_b_mass);
    velocities_out[child_b_slot] = vec4<f32>(parent_velocity.xyz, 0.0);

    cell_ids[child_b_slot] = child_b_id;
    organism_cell_ids[child_b_slot] = child_b_organism_cell_id;

    // Physics rotation = genome rotation chain (no random perturbation for determinism)
    rotations_out[child_b_slot] = child_b_rotation;
    genome_orientations[child_b_slot] = child_b_genome_orientation;
    development_addresses[child_b_slot] = vec4<u32>(
        child_b_org_id,
        child_b_lineage_hash.lo,
        child_b_lineage_hash.hi,
        (child_lineage_depth << 16u) | 2u
    );
    parent_lineage_hashes_out[child_b_slot] = vec2<u32>(parent_lineage_hash.lo, parent_lineage_hash.hi);

    let child_b_props_0 = mode_properties_v0[child_b_mode_idx];
    let child_b_props_1 = mode_properties_v1[child_b_mode_idx];
    let child_b_props_2 = mode_properties_v2[child_b_mode_idx];
    
    birth_times[child_b_slot] = params.current_time;
    split_intervals[child_b_slot] = child_b_props_0.w;
    // Convert split_mass to nutrient threshold: (split_mass - 1.0) * 100.0
    let child_b_split_mass = child_b_props_1.x;
    split_nutrient_thresholds[child_b_slot] = (child_b_split_mass - 1.0) * 100.0;
    
    if (child_b_mode_idx != parent_mode_idx || after_splits_b_fires) {
        split_counts[child_b_slot] = 0u;
    } else {
        split_counts[child_b_slot] = parent_split_count + 1u;
    }
    
    max_splits[child_b_slot] = select(u32(child_b_props_2.x), 0xFFFFFFFFu, child_b_props_2.x < 0.0);
    genome_ids[child_b_slot] = genome_ids[cell_idx];
    mode_indices[child_b_slot] = child_b_mode_idx;
    
    let child_b_cell_type = mode_cell_types[child_b_mode_idx];
    let child_b_nutrient_rate = select(0.0, child_b_props_0.x, child_b_cell_type == 0u);
    nutrient_gain_rates[child_b_slot] = child_b_nutrient_rate;
    
    cell_types[child_b_slot] = child_b_cell_type;
    max_cell_sizes[child_b_slot] = child_b_props_0.y;
    stiffnesses[child_b_slot] = child_b_props_0.z;
    
    // Set child B nutrients and reserve (reserve halved from parent)
    atomicStore(&nutrients_buffer[child_b_slot], float_to_fixed(child_b_nutrients));
    atomicStore(&embryocyte_reserves[child_b_slot], child_reserve);

    // Clear death flag for the new slot
    death_flags[child_b_slot] = 0u;
    
    // Pause nutrient transfer for both children (~0.1s)
    // This prevents the deferred neighbor (which hasn't split yet) from losing
    // mass to these small newborn children via pressure-based transport.
    split_ready_frame[cell_idx] = params.current_frame;
    split_ready_frame[child_b_slot] = params.current_frame;
    
    // Update total cell count if we used a new slot beyond current count
    atomicMax(&cell_count_buffer[0], child_b_slot + 1u);
    
    // Track sibling adhesion slot for inheritance (0xFFFFFFFF = invalid/not created)
    var sibling_adhesion_slot: u32 = 0xFFFFFFFFu;
    
    // === Create sibling adhesion if parent_make_adhesion is enabled ===
    // Normal keep_adhesion flags only affect inheritance; after-split keep flags
    // also suppress the sibling bond when the max-splits transition fires.
    let make_adhesion = parent_make_adhesion_flags[parent_mode_idx];
    let after_split_sibling_allowed = !will_reach_max_splits
        || (
            child_a_after_split_keep_adhesion_flags[parent_mode_idx] == 1u
            && child_b_after_split_keep_adhesion_flags[parent_mode_idx] == 1u
        );
    if (make_adhesion == 1u && after_split_sibling_allowed) {
        let adhesion_id = allocate_adhesion_slot();
        if (adhesion_id != 0xFFFFFFFFu) {
            // Anchor directions in each child's LOCAL space (XPBD approach)
            // Since split_rotation is baked into child rotation, the split axis
            // in the child's local frame is just child.orientation.inverse() * Z
            // Child A is at +split, points toward B: use -Z
            // Child B is at -split, points toward A: use +Z
            let anchor_a_local = normalize(rotate_vector_by_quat_inverse(vec3<f32>(0.0, 0.0, -1.0), child_a_orientation_final));
            let anchor_b_local = normalize(rotate_vector_by_quat_inverse(vec3<f32>(0.0, 0.0, 1.0), child_b_orientation_final));
            
            // Classify zones based on anchor direction vs split direction (matching CPU)
            // Each cell's zone uses its own split_ratio for dynamic equatorial zone
            let child_a_split_ratio = child_a_props_2.y;
            let child_b_split_ratio = child_b_props_2.y;
            let sibling_split_dir = normalize(split_dir_local);
            let zone_a = classify_zone(anchor_a_local, sibling_split_dir, child_a_split_ratio);
            let zone_b = classify_zone(anchor_b_local, sibling_split_dir, child_b_split_ratio);
            
            var connection: AdhesionConnection;
            connection.cell_a_index = cell_idx;
            connection.cell_b_index = child_b_slot;
            connection.mode_index = parent_mode_idx;
            connection.is_active = 1u;
            connection.zone_a = zone_a;
            connection.zone_b = zone_b;
            connection._align_pad = vec2<u32>(0u, 0u);
            connection.anchor_direction_a = vec4<f32>(anchor_a_local, 0.0);
            connection.anchor_direction_b = vec4<f32>(anchor_b_local, 0.0);
            // Set twist references to child GENOME orientations (genome-pure, no physics)
            connection.twist_reference_a = child_a_genome_orientation;
            connection.twist_reference_b = child_b_genome_orientation;
            connection.birth_time = params.current_time;
            connection._pad = 0u;
            adhesion_connections[adhesion_id] = connection;
            
            // Add to cell adhesion indices
            let base_a = cell_idx * MAX_ADHESIONS_PER_CELL;
            let base_b = child_b_slot * MAX_ADHESIONS_PER_CELL;
            
            for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
                if (atomicLoad(&cell_adhesion_indices[base_a + i]) < 0) {
                    atomicStore(&cell_adhesion_indices[base_a + i], i32(adhesion_id));
                    break;
                }
            }
            for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
                if (atomicLoad(&cell_adhesion_indices[base_b + i]) < 0) {
                    atomicStore(&cell_adhesion_indices[base_b + i], i32(adhesion_id));
                    break;
                }
            }
            
            // Track sibling adhesion slot to skip during inheritance
            sibling_adhesion_slot = adhesion_id;
        }
    }
    
    // === Zone-Based Adhesion Inheritance ===
    // Process parent's existing adhesions and distribute to children based on zone classification
    // - Zone A (negative dot with split dir) -> Child B
    // - Zone B (positive dot with split dir) -> Child A  
    // - Zone C (equatorial) -> Both children (duplicate)
    
    let child_a_keep = select(
        child_a_keep_adhesion_flags[parent_mode_idx] == 1u,
        child_a_after_split_keep_adhesion_flags[parent_mode_idx] == 1u,
        will_reach_max_splits
    );
    let child_b_keep = select(
        child_b_keep_adhesion_flags[parent_mode_idx] == 1u,
        child_b_after_split_keep_adhesion_flags[parent_mode_idx] == 1u,
        will_reach_max_splits
    );
    // but always preserve the sibling adhesion if one was created.
    // Also clear child_b's stale indices and rebuild with just the sibling bond.
    if (!child_a_keep && !child_b_keep) {
        // Clear child_a (parent slot): deactivate old inherited connections, skip sibling
        let clear_base_a = cell_idx * MAX_ADHESIONS_PER_CELL;
        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            let adh_idx_signed = atomicLoad(&cell_adhesion_indices[clear_base_a + i]);
            if (adh_idx_signed >= 0 && u32(adh_idx_signed) != sibling_adhesion_slot) {
                let freed = u32(adh_idx_signed);
                adhesion_connections[freed].is_active = 0u;
                free_adhesion_slot(freed);
                atomicStore(&cell_adhesion_indices[clear_base_a + i], -1);
            }
        }
        // Clear child_b's stale old indices and repopulate with just the sibling bond.
        // parent_make_adhesion creates the sibling on normal splits; after-split keep
        // flags can suppress it on the max-splits transition.
        let clear_base_b = child_b_slot * MAX_ADHESIONS_PER_CELL;
        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            atomicStore(&cell_adhesion_indices[clear_base_b + i], -1);
        }
        if (sibling_adhesion_slot != 0xFFFFFFFFu) {
            atomicStore(&cell_adhesion_indices[clear_base_b], i32(sibling_adhesion_slot));
        }
        return;
    }
    
    let parent_adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let child_a_adhesion_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let child_b_adhesion_base = child_b_slot * MAX_ADHESIONS_PER_CELL;
    
    // Save parent's adhesion indices BEFORE modifying them
    var parent_adhesion_indices: array<i32, 20>;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        parent_adhesion_indices[i] = atomicLoad(&cell_adhesion_indices[parent_adhesion_base + i]);
    }
    
    // Clear Child A's adhesion indices (parent slot) - we'll rebuild them
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        atomicStore(&cell_adhesion_indices[child_a_adhesion_base + i], -1);
        atomicStore(&cell_adhesion_indices[child_b_adhesion_base + i], -1);
    }
    
    // Track adhesion counts for each child
    var child_a_adhesion_count = 0u;
    var child_b_adhesion_count = 0u;
    
    // Re-add the sibling adhesion (if created)
    if (sibling_adhesion_slot != 0xFFFFFFFFu) {
        if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
            atomicStore(&cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count], i32(sibling_adhesion_slot));
            child_a_adhesion_count++;
        }
        if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
            atomicStore(&cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count], i32(sibling_adhesion_slot));
            child_b_adhesion_count++;
        }
    }
    
    // Calculate geometric parameters for anchor calculation
    let child_offset = parent_radius * 0.25;
    let split_dir_normalized = normalize(split_dir_local);
    let child_a_pos_parent_frame = split_dir_normalized * child_offset;
    let child_b_pos_parent_frame = -split_dir_normalized * child_offset;
    
    // Fixed radius for adhesion distance calculation
    let FIXED_RADIUS: f32 = 1.0;
    
    // Process inherited adhesions
    for (var parent_slot = 0u; parent_slot < MAX_ADHESIONS_PER_CELL; parent_slot++) {
        let adh_idx_signed = parent_adhesion_indices[parent_slot];
        
        if (adh_idx_signed < 0) {
            continue;
        }
        
        let adh_idx = u32(adh_idx_signed);
        
        // Skip sibling adhesion (already handled)
        if (adh_idx == sibling_adhesion_slot) {
            continue;
        }
        
        let conn = adhesion_connections[adh_idx];
        
        if (conn.is_active == 0u) {
            continue;
        }
        
        // Determine which side of connection parent is on
        let is_parent_cell_a = conn.cell_a_index == cell_idx;
        let is_parent_cell_b = conn.cell_b_index == cell_idx;
        
        if (!is_parent_cell_a && !is_parent_cell_b) {
            continue;
        }
        
        let neighbor_idx = select(conn.cell_a_index, conn.cell_b_index, is_parent_cell_a);
        
        // Get parent's anchor direction (LOCAL space)
        var parent_anchor_dir_local: vec3<f32>;
        if (is_parent_cell_a) {
            parent_anchor_dir_local = conn.anchor_direction_a.xyz;
        } else {
            parent_anchor_dir_local = conn.anchor_direction_b.xyz;
        }
        
        // Classify zone based on LOCAL anchor direction vs LOCAL split direction
        // Use parent's split_ratio for inheritance classification
        let zone = classify_zone(parent_anchor_dir_local, split_dir_local, parent_split_ratio);
        
        // Determine which child(ren) inherit
        var give_to_child_a = false;
        var give_to_child_b = false;
        
        if (zone == ZONE_A && child_b_keep) {
            give_to_child_b = true;
        } else if (zone == ZONE_B && child_a_keep) {
            give_to_child_a = true;
        } else if (zone == ZONE_C) {
            if (child_a_keep) { give_to_child_a = true; }
            if (child_b_keep) { give_to_child_b = true; }
        }
        
        if (!give_to_child_a && !give_to_child_b) {
            adhesion_connections[adh_idx].is_active = 0u;
            free_adhesion_slot(adh_idx);
            continue;
        }
        
        // Calculate center-to-center distance
        let rest_length = 1.0;
        let center_to_center_dist = rest_length + FIXED_RADIUS + FIXED_RADIUS;
        let neighbor_pos_parent_frame = parent_anchor_dir_local * center_to_center_dist;
        
        // Get neighbor GENOME orientation for anchor calculation (genome-pure)
        let neighbor_genome_orientation = genome_orientations[neighbor_idx];
        let relative_rotation = quat_multiply(quat_conjugate(neighbor_genome_orientation), parent_genome_orientation);
        
        // Pre-compute orientation deltas including split_rotation (matching CPU)
        let child_a_orientation_delta = quat_multiply(split_rotation_quat, child_a_orientation_final);
        let child_b_orientation_delta = quat_multiply(split_rotation_quat, child_b_orientation_final);
        
        if (give_to_child_a && !give_to_child_b) {
            // Only Child A inherits
            let child_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_a_orientation_delta
            );
            
            let dir_to_child_parent_frame = normalize(child_a_pos_parent_frame - neighbor_pos_parent_frame);
            let neighbor_anchor = normalize(rotate_vector_by_quat(dir_to_child_parent_frame, relative_rotation));
            
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_a_genome_orientation;
                adhesion_connections[adh_idx].zone_a = classify_zone(child_anchor, split_dir_local, child_a_props_2.y);
            } else {
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_a_genome_orientation;
                adhesion_connections[adh_idx].zone_b = classify_zone(child_anchor, split_dir_local, child_a_props_2.y);
            }
            
            if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                atomicStore(&cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count], i32(adh_idx));
                child_a_adhesion_count++;
            }
            
        } else if (give_to_child_b && !give_to_child_a) {
            // Only Child B inherits
            let child_anchor = calculate_child_anchor_direction(
                child_b_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_b_orientation_delta
            );
            
            let dir_to_child_parent_frame = normalize(child_b_pos_parent_frame - neighbor_pos_parent_frame);
            let neighbor_anchor = normalize(rotate_vector_by_quat(dir_to_child_parent_frame, relative_rotation));
            
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].cell_a_index = child_b_slot;
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_b_genome_orientation;
                adhesion_connections[adh_idx].zone_a = classify_zone(child_anchor, split_dir_local, child_b_props_2.y);
            } else {
                adhesion_connections[adh_idx].cell_b_index = child_b_slot;
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_b_genome_orientation;
                adhesion_connections[adh_idx].zone_b = classify_zone(child_anchor, split_dir_local, child_b_props_2.y);
            }
            
            if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                atomicStore(&cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count], i32(adh_idx));
                child_b_adhesion_count++;
            }
            
        } else {
            // Zone C: Both children inherit - duplicate the adhesion
            let child_a_anchor = calculate_child_anchor_direction(
                child_a_pos_parent_frame,
                neighbor_pos_parent_frame,
                child_a_orientation_delta
            );
            
            let dir_to_child_a_parent_frame = normalize(child_a_pos_parent_frame - neighbor_pos_parent_frame);
            let neighbor_anchor_to_a = normalize(rotate_vector_by_quat(dir_to_child_a_parent_frame, relative_rotation));
            
            // Update original connection for Child A
            if (is_parent_cell_a) {
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(child_a_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(neighbor_anchor_to_a, 0.0);
                adhesion_connections[adh_idx].twist_reference_a = child_a_genome_orientation;
                adhesion_connections[adh_idx].zone_a = classify_zone(child_a_anchor, split_dir_local, child_a_props_2.y);
            } else {
                adhesion_connections[adh_idx].anchor_direction_b = vec4<f32>(child_a_anchor, 0.0);
                adhesion_connections[adh_idx].anchor_direction_a = vec4<f32>(neighbor_anchor_to_a, 0.0);
                adhesion_connections[adh_idx].twist_reference_b = child_a_genome_orientation;
                adhesion_connections[adh_idx].zone_b = classify_zone(child_a_anchor, split_dir_local, child_a_props_2.y);
            }
            
            if (child_a_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                atomicStore(&cell_adhesion_indices[child_a_adhesion_base + child_a_adhesion_count], i32(adh_idx));
                child_a_adhesion_count++;
            }
            
            // Create duplicate for Child B
            let dup_slot = allocate_adhesion_slot();
            if (dup_slot != 0xFFFFFFFFu) {
                let child_b_anchor = calculate_child_anchor_direction(
                    child_b_pos_parent_frame,
                    neighbor_pos_parent_frame,
                    child_b_orientation_delta
                );
                
                let dir_to_child_b_parent_frame = normalize(child_b_pos_parent_frame - neighbor_pos_parent_frame);
                let neighbor_anchor_to_b = normalize(rotate_vector_by_quat(dir_to_child_b_parent_frame, relative_rotation));
                
                var dup_conn: AdhesionConnection;
                dup_conn.mode_index = conn.mode_index;
                dup_conn.is_active = 1u;
                dup_conn._align_pad = vec2<u32>(0u, 0u);
                dup_conn.birth_time = params.current_time;
                dup_conn._pad = 0u;
                
                // Get neighbor's split_ratio for zone classification on their side
                let neighbor_mode_idx = mode_indices[neighbor_idx];
                let neighbor_split_ratio = mode_properties_v2[neighbor_mode_idx].y;
                
                if (is_parent_cell_a) {
                    dup_conn.cell_a_index = child_b_slot;
                    dup_conn.cell_b_index = neighbor_idx;
                    dup_conn.anchor_direction_a = vec4<f32>(child_b_anchor, 0.0);
                    dup_conn.anchor_direction_b = vec4<f32>(neighbor_anchor_to_b, 0.0);
                    dup_conn.twist_reference_a = child_b_genome_orientation;
                    dup_conn.twist_reference_b = neighbor_genome_orientation;
                    dup_conn.zone_a = classify_zone(child_b_anchor, split_dir_local, child_b_props_2.y);
                    dup_conn.zone_b = classify_zone(neighbor_anchor_to_b, split_dir_local, neighbor_split_ratio);
                } else {
                    dup_conn.cell_a_index = neighbor_idx;
                    dup_conn.cell_b_index = child_b_slot;
                    dup_conn.anchor_direction_a = vec4<f32>(neighbor_anchor_to_b, 0.0);
                    dup_conn.anchor_direction_b = vec4<f32>(child_b_anchor, 0.0);
                    dup_conn.twist_reference_a = neighbor_genome_orientation;
                    dup_conn.twist_reference_b = child_b_genome_orientation;
                    dup_conn.zone_a = classify_zone(neighbor_anchor_to_b, split_dir_local, neighbor_split_ratio);
                    dup_conn.zone_b = classify_zone(child_b_anchor, split_dir_local, child_b_props_2.y);
                }
                
                adhesion_connections[dup_slot] = dup_conn;
                
                if (child_b_adhesion_count < MAX_ADHESIONS_PER_CELL) {
                    atomicStore(&cell_adhesion_indices[child_b_adhesion_base + child_b_adhesion_count], i32(dup_slot));
                    child_b_adhesion_count++;
                }
                
                // Add duplicate to neighbor's adhesion list.
                // Use a retry loop with atomicCompareExchangeWeak to handle spurious
                // failures - keep retrying the same slot until it either succeeds or
                // we confirm the slot is occupied, then move to the next slot.
                let neighbor_adhesion_base = neighbor_idx * MAX_ADHESIONS_PER_CELL;
                var i = 0u;
                loop {
                    if (i >= MAX_ADHESIONS_PER_CELL) { break; }
                    let cas = atomicCompareExchangeWeak(&cell_adhesion_indices[neighbor_adhesion_base + i], -1, i32(dup_slot));
                    if (cas.exchanged) {
                        break; // Successfully registered
                    }
                    // If old_value != -1, slot is occupied - move to next slot.
                    // If old_value == -1 but not exchanged (spurious fail) - retry same slot.
                    if (cas.old_value != -1) {
                        i++;
                    }
                    // else: spurious failure, retry same i
                }

            }
        }
    }

    // === Disconnect inherited bonds overlapping the sibling bond ===
    // Runs AFTER inheritance so the inherited bonds have been re-expressed into child A's
    // local frame. The sibling bond's child-A-side anchor and the inherited bonds' child-A-side
    // anchors are now in the same frame, so the overlap test is valid. This matches the CPU
    // path in division.rs (which runs inheritance before sibling creation + scan).
    if (sibling_adhesion_slot != 0xFFFFFFFFu) {
        let sibling_anchor_a = adhesion_connections[sibling_adhesion_slot].anchor_direction_a.xyz;
        let disc_base_a = cell_idx * MAX_ADHESIONS_PER_CELL;
        for (var k = 0u; k < MAX_ADHESIONS_PER_CELL; k++) {
            let existing_idx_signed = atomicLoad(&cell_adhesion_indices[disc_base_a + k]);
            if (existing_idx_signed < 0) { continue; }
            let existing_idx = u32(existing_idx_signed);
            if (existing_idx == sibling_adhesion_slot) { continue; }
            let existing_conn = adhesion_connections[existing_idx];
            if (existing_conn.is_active == 0u) { continue; }
            // Child A's side anchor in its own local frame
            var existing_anchor_local: vec3<f32>;
            if (existing_conn.cell_a_index == cell_idx) {
                existing_anchor_local = existing_conn.anchor_direction_a.xyz;
            } else {
                existing_anchor_local = existing_conn.anchor_direction_b.xyz;
            }
            if (dot(sibling_anchor_a, existing_anchor_local) > ANCHOR_OVERLAP_COS) {
                // Disconnect the old bond and clear it from both cells' index lists
                adhesion_connections[existing_idx].is_active = 0u;
                let other_idx = select(existing_conn.cell_a_index, existing_conn.cell_b_index, existing_conn.cell_a_index == cell_idx);
                atomicStore(&cell_adhesion_indices[disc_base_a + k], -1);
                let other_base = other_idx * MAX_ADHESIONS_PER_CELL;
                for (var j = 0u; j < MAX_ADHESIONS_PER_CELL; j++) {
                    if (atomicLoad(&cell_adhesion_indices[other_base + j]) == i32(existing_idx)) {
                        atomicStore(&cell_adhesion_indices[other_base + j], -1);
                    }
                }
                break;
            }
        }
    }
}
