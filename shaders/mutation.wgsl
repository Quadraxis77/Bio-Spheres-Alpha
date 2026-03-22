// Mutation Shader - GPU-side genome mutation during cell division
//
// Runs AFTER lifecycle_division_execute_ring.wgsl.
// For each cell that just divided (both Child A and Child B), rolls a mutation
// chance based on global radiation level. On hit:
//   1. Allocates a new genome slot from the genome free ring buffer
//   2. Copies all mode data from parent genome to new genome slot
//   3. Selects a random mode and random parameter using hash-based PRNG
//   4. Applies a bounded perturbation based on per-parameter vulnerability weights
//   5. Updates the child's genome_id and mode_indices to point at the new genome
//
// The shader processes BOTH children of each division event. Each child
// independently rolls for mutation. A cell_idx is a "recently divided child"
// if division_flags[cell_idx] == 1 (it was the parent, now Child A) or if
// the cell was written as Child B by the division shader (tracked via
// mutation_candidates buffer written by division execute).
//
// Buffer layout for genome mode data (flat arrays, absolute mode indexing):
//   mode_properties:           20 f32 per mode (80 bytes) — physics/division params
//   genome_mode_data:          20 f32 per mode (80 bytes) — orientations, split quat
//   child_mode_indices:         2 i32 per mode (8 bytes)  — absolute child mode refs
//   mode_cell_types:            1 u32 per mode (4 bytes)  — cell type enum
//   parent_make_adhesion_flags: 1 u32 per mode (4 bytes)  — bool
//   child_a_keep_adhesion_flags:1 u32 per mode (4 bytes)  — bool
//   child_b_keep_adhesion_flags:1 u32 per mode (4 bytes)  — bool
//   glueocyte_env_adhesion_flags:1 u32 per mode (4 bytes) — bool
//   oculocyte_params:           4 u32 per mode (16 bytes) — sense params

// ============================================================
// Structs
// ============================================================

struct MutationParams {
    // Global radiation level (0.0 = no mutations, 1.0 = mutate every division)
    radiation_level: f32,
    // RNG seed (advanced each frame by CPU)
    rng_seed: u32,
    // Current frame number (for RNG mixing)
    current_frame: u32,
    // Number of mutation parameter entries in the vulnerability table
    param_table_size: u32,
    // Total number of active modes across all genomes (flat buffer extent)
    total_mode_count: u32,
    // Maximum modes per genome (for bounds checking cloned genome)
    max_modes_per_genome: u32,
    // Genome ring buffer capacity
    genome_ring_capacity: u32,
    // When 1, the automatic color side-effect nudges each channel slightly instead of full re-roll
    subtle_color_mutation: u32,
}

// Per-parameter mutation descriptor.
// The vulnerability table is a small buffer (~40 entries) describing each
// mutable parameter: which buffer it lives in, its offset within a mode's
// data, value bounds, and how likely it is to be selected.
struct MutationParamEntry {
    // Which buffer this parameter lives in (enum):
    //   0 = mode_properties (f32 array, 20 per mode)
    //   1 = mode_cell_types (u32, 1 per mode)
    //   2 = child_mode_indices (i32 x2, 2 per mode)
    //   3 = parent_make_adhesion_flags (u32, 1 per mode)
    //   4 = child_a_keep_adhesion_flags (u32, 1 per mode)
    //   5 = child_b_keep_adhesion_flags (u32, 1 per mode)
    //   6 = genome_mode_data (f32 array, 20 per mode)
    //   7 = glueocyte_env_adhesion_flags (u32, 1 per mode)
    //   8 = oculocyte_params (u32 array, 4 per mode)
    //   9 = mode_visuals (f32 array, 2 vec4 per mode: color + emissive)
    //  10 = genome_initial_mode (u32, per-genome — stored in genome_meta[id].z)
    buffer_id: u32,
    // Element offset within one mode's data in that buffer.
    // e.g. for mode_properties index 4 (split_mass), element_offset = 4
    element_offset: u32,
    // Vulnerability weight (relative probability of selection, 0.0-10.0)
    weight: f32,
    // Minimum perturbation magnitude (absolute)
    min_delta: f32,
    // Maximum perturbation magnitude (absolute)
    max_delta: f32,
    // Minimum allowed value after mutation
    min_value: f32,
    // Maximum allowed value after mutation
    max_value: f32,
    // Data type: 0 = continuous f32, 1 = integer (round after perturb),
    //            2 = boolean (flip), 3 = integer with mode_count clamp,
    //            4 = chain_extend, 5 = chain_close, 6 = loop_branch,
    //            7 = loop_merge, 8 = signal_wire (correlated emitter+receiver)
    data_type: u32,
}

// ============================================================
// Bind groups
// ============================================================

// Group 0: Mutation parameters + vulnerability table
@group(0) @binding(0)
var<uniform> mutation_params: MutationParams;

@group(0) @binding(1)
var<storage, read> vulnerability_table: array<MutationParamEntry>;

// Genome free slot ring buffer: [0]=head, [1]=tail, [2]=next_genome_id
@group(0) @binding(2)
var<storage, read_write> genome_ring_state: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> genome_free_ring: array<u32>;

// Per-genome metadata: genome_meta[genome_id] = vec4<u32>(mode_count, base_mode_offset, initial_mode_local, flags)
// .z = initial_mode as a local (0-based) mode index within the genome
// Note: ref_count is tracked separately in genome_ref_counts for atomic access
@group(0) @binding(4)
var<storage, read_write> genome_meta: array<vec4<u32>>;

// Per-genome reference counts (atomic for safe concurrent updates)
@group(0) @binding(5)
var<storage, read_write> genome_ref_counts: array<atomic<u32>>;

// Group 1: Mutation candidates (written by division execute, read here)
// mutation_candidates[i] = vec2<u32>(child_cell_idx, parent_genome_id)
// Length = 2 * max_divisions_per_frame (child A and child B per division)
@group(1) @binding(0)
var<storage, read> mutation_candidates: array<vec2<u32>>;

// Number of mutation candidates this frame: [0] = count
@group(1) @binding(1)
var<storage, read> mutation_candidate_count: array<u32>;

// Per-cell genome_ids and mode_indices (read_write — we update on mutation)
@group(1) @binding(2)
var<storage, read_write> genome_ids: array<u32>;

@group(1) @binding(3)
var<storage, read_write> mode_indices: array<u32>;

// Group 2: Genome mode buffers (read_write for cloning + mutation)
// mode_properties split into 5 vec4 sub-buffers (16 bytes/mode each, max 8M modes = 128 MB each)
@group(2) @binding(0)
var<storage, read_write> mode_properties_v0: array<vec4<f32>>; // [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval]
@group(2) @binding(1)
var<storage, read_write> mode_properties_v1: array<vec4<f32>>; // [split_mass, nutrient_priority, swim_force, prioritize_when_low]
@group(2) @binding(2)
var<storage, read_write> mode_properties_v2: array<vec4<f32>>; // [max_splits, split_ratio, flagellocyte_signal_channel, flagellocyte_speed_a]
@group(2) @binding(3)
var<storage, read_write> mode_properties_v3: array<vec4<f32>>; // [flagellocyte_speed_b, flagellocyte_threshold_c, flagellocyte_use_signal, min_adhesions]
@group(2) @binding(4)
var<storage, read_write> mode_properties_v4: array<vec4<f32>>; // [max_adhesions, mode_a_after_splits, mode_b_after_splits, padding]

// genome_mode_data split into 5 vec4 sub-buffers
@group(2) @binding(5)
var<storage, read_write> genome_mode_data_v0: array<vec4<f32>>; // child_a_orientation
@group(2) @binding(6)
var<storage, read_write> genome_mode_data_v1: array<vec4<f32>>; // child_b_orientation
@group(2) @binding(7)
var<storage, read_write> genome_mode_data_v2: array<vec4<f32>>; // child_a_split_orientation
@group(2) @binding(8)
var<storage, read_write> genome_mode_data_v3: array<vec4<f32>>; // child_b_split_orientation
@group(2) @binding(9)
var<storage, read_write> genome_mode_data_v4: array<vec4<f32>>; // split_rotation_quat (XYZW)

@group(2) @binding(10)
var<storage, read_write> child_mode_indices_buf: array<vec2<i32>>;

@group(2) @binding(11)
var<storage, read_write> mode_cell_types: array<u32>;

@group(2) @binding(12)
var<storage, read_write> parent_make_adhesion_flags: array<u32>;

@group(2) @binding(13)
var<storage, read_write> child_a_keep_adhesion_flags: array<u32>;

@group(2) @binding(14)
var<storage, read_write> child_b_keep_adhesion_flags: array<u32>;

@group(2) @binding(15)
var<storage, read_write> glueocyte_env_adhesion_flags: array<u32>;

@group(2) @binding(16)
var<storage, read_write> oculocyte_params_buf: array<vec4<u32>>;

// Mutation event log (optional, for debug/UI feedback)
@group(2) @binding(17)
var<storage, read_write> mutation_log: array<vec4<u32>>;

@group(2) @binding(18)
var<storage, read_write> mutation_log_count: array<atomic<u32>>;

// Mode colors buffer (from instance builder): 1 vec4<f32> per mode (RGB + padding)
@group(2) @binding(19)
var<storage, read_write> mode_colors: array<vec4<f32>>;

// Mode emissive buffer (from instance builder): 1 vec4<f32> per mode (emissive + padding)
@group(2) @binding(20)
var<storage, read_write> mode_emissive: array<vec4<f32>>;

// Adhesion settings buffer: 12 f32 per mode (48 bytes = 3 vec4s)
// Adhesion settings split into 3 × vec4 sub-buffers (16 bytes each per mode).
// v0: [can_break(u32 bits), break_force(f32 bits), rest_length(f32 bits), linear_spring_stiffness(f32 bits)]
// v1: [linear_spring_damping, orientation_spring_stiffness, orientation_spring_damping, max_angular_deviation]
// v2: [twist_constraint_stiffness, twist_constraint_damping, enable_twist_constraint(u32 bits), _padding]
@group(2) @binding(21)
var<storage, read_write> adhesion_settings_v0: array<vec4<u32>>;

@group(2) @binding(22)
var<storage, read_write> adhesion_settings_v1: array<vec4<u32>>;

@group(2) @binding(23)
var<storage, read_write> adhesion_settings_v2: array<vec4<u32>>;

// Signal-conditional settings: 5 × vec4<f32> sub-buffers per mode
// v0: [division_signal_channel, division_signal_threshold, division_signal_invert, apoptosis_signal_channel]
// v1: [apoptosis_signal_threshold, apoptosis_signal_invert, signal_child_a_channel, signal_child_a_threshold]
// v2: [signal_child_a_mode_above, signal_child_a_mode_below, signal_child_b_channel, signal_child_b_threshold]
// v3: [signal_child_b_mode_above, signal_child_b_mode_below, mode_switch_signal_channel, mode_switch_signal_threshold]
// v4: [mode_switch_target, mode_switch_invert, padding, padding]
@group(2) @binding(24)
var<storage, read_write> signal_settings_v0: array<vec4<f32>>;

@group(2) @binding(25)
var<storage, read_write> signal_settings_v1: array<vec4<f32>>;

@group(2) @binding(26)
var<storage, read_write> signal_settings_v2: array<vec4<f32>>;

@group(2) @binding(27)
var<storage, read_write> signal_settings_v3: array<vec4<f32>>;

@group(2) @binding(28)
var<storage, read_write> signal_settings_v4: array<vec4<f32>>;

// Per-mode regulation emission parameters: [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding(u32)]
@group(2) @binding(29)
var<storage, read_write> regulation_params_buf: array<vec4<u32>>;

// ============================================================
// Hash-based PRNG (PCG-style)
// ============================================================

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Returns a pseudo-random u32 from a unique_id, frame seed, and salt.
// unique_id must be unique per mutation event (e.g. new_genome_id).
// We hash unique_id first to avalanche its bits before mixing with the
// frame seed — this prevents correlated outputs when unique_id is a small
// sequential integer (which it often is from the monotonic allocator).
fn rng_u32(unique_id: u32, salt: u32) -> u32 {
    // First avalanche the unique_id so sequential IDs produce uncorrelated seeds
    let id_hashed = pcg_hash(unique_id + 1u); // +1 so id=0 doesn't produce a weak hash
    // Mix with frame-level entropy and salt
    let frame_entropy = mutation_params.rng_seed * 1664525u + mutation_params.current_frame * 22695477u + salt * 2246822519u;
    return pcg_hash(id_hashed ^ frame_entropy);
}

// Returns a pseudo-random f32 in [0.0, 1.0)
fn rng_f32(unique_id: u32, salt: u32) -> f32 {
    return f32(rng_u32(unique_id, salt) & 0x00FFFFFFu) / 16777216.0;
}

// Returns a pseudo-random f32 in [-1.0, 1.0)
fn rng_signed_f32(unique_id: u32, salt: u32) -> f32 {
    return rng_f32(unique_id, salt) * 2.0 - 1.0;
}

// ============================================================
// Genome slot allocation (ring buffer)
//
// genome_ring_state layout:
//   [0] = head  (pop cursor, monotonically increasing)
//   [1] = tail  (push cursor, monotonically increasing)
//   [2] = next_genome_id (monotonic bootstrap allocator)
//   [3] = next_mode_offset
//   [4] = max_active_mode_offset (written by GC)
//
// Available slots = tail - head (both wrap at u32 max, subtraction still correct).
// GC resets next_genome_id to user_genome_count each cycle so the monotonic
// path re-fills IDs that haven't been recycled into the free ring yet.
// ============================================================

fn allocate_genome_slot() -> u32 {
    // Try to pop from free ring.
    // tail >= head always (tail only increases via push, head only increases via pop).
    // We speculatively increment head, then check if we were within bounds.
    let head = atomicAdd(&genome_ring_state[0], 1u);
    let tail = atomicLoad(&genome_ring_state[1]);

    // head was the value BEFORE our increment, so valid if head < tail
    if (head < tail) {
        let ring_idx = head % mutation_params.genome_ring_capacity;
        return genome_free_ring[ring_idx];
    }

    // Ring empty — undo the head increment and fall through to monotonic path
    atomicSub(&genome_ring_state[0], 1u);

    let new_id = atomicAdd(&genome_ring_state[2], 1u);
    if (new_id < mutation_params.genome_ring_capacity) {
        return new_id;
    }

    // Monotonic path also exhausted — undo and signal failure
    atomicSub(&genome_ring_state[2], 1u);
    return 0xFFFFFFFFu;
}

// ============================================================
// Weighted random parameter selection
// ============================================================

// Select a parameter index from the vulnerability table using weighted random.
// Uses the cumulative weight approach: generate random in [0, total_weight),
// then walk the table until cumulative weight exceeds the random value.
fn select_param(cell_id: u32, salt: u32) -> u32 {
    // First pass: compute total weight
    var total_weight: f32 = 0.0;
    let table_size = mutation_params.param_table_size;
    for (var i = 0u; i < table_size; i++) {
        total_weight += vulnerability_table[i].weight;
    }

    if (total_weight <= 0.0) {
        return 0u;
    }

    let threshold = rng_f32(cell_id, salt) * total_weight;
    var cumulative: f32 = 0.0;
    for (var i = 0u; i < table_size; i++) {
        cumulative += vulnerability_table[i].weight;
        if (cumulative > threshold) {
            return i;
        }
    }

    return table_size - 1u;
}

// ============================================================
// Clone genome mode data from source to destination
// ============================================================

fn clone_genome_modes(
    src_base: u32,   // source absolute mode offset
    dst_base: u32,   // destination absolute mode offset
    mode_count: u32,
) {
    for (var m = 0u; m < mode_count; m++) {
        let src = src_base + m;
        let dst = dst_base + m;

        // mode_properties: 5 separate sub-buffers (v0-v4), 1 vec4 per mode each
        mode_properties_v0[dst] = mode_properties_v0[src];
        mode_properties_v1[dst] = mode_properties_v1[src];
        mode_properties_v2[dst] = mode_properties_v2[src];
        mode_properties_v3[dst] = mode_properties_v3[src];
        mode_properties_v4[dst] = mode_properties_v4[src];

        // genome_mode_data: 5 separate sub-buffers (v0-v4), 1 vec4 per mode each
        genome_mode_data_v0[dst] = genome_mode_data_v0[src];
        genome_mode_data_v1[dst] = genome_mode_data_v1[src];
        genome_mode_data_v2[dst] = genome_mode_data_v2[src];
        genome_mode_data_v3[dst] = genome_mode_data_v3[src];
        genome_mode_data_v4[dst] = genome_mode_data_v4[src];

        // child_mode_indices: 1 vec2<i32> per mode
        // IMPORTANT: remap absolute indices from source genome to destination genome
        let src_indices = child_mode_indices_buf[src];
        let offset_delta = i32(dst_base) - i32(src_base);
        child_mode_indices_buf[dst] = vec2<i32>(
            src_indices.x + offset_delta,
            src_indices.y + offset_delta,
        );

        // Scalar per-mode buffers
        mode_cell_types[dst] = mode_cell_types[src];
        parent_make_adhesion_flags[dst] = parent_make_adhesion_flags[src];
        child_a_keep_adhesion_flags[dst] = child_a_keep_adhesion_flags[src];
        child_b_keep_adhesion_flags[dst] = child_b_keep_adhesion_flags[src];
        glueocyte_env_adhesion_flags[dst] = glueocyte_env_adhesion_flags[src];

        // oculocyte_params: 1 vec4<u32> per mode
        oculocyte_params_buf[dst] = oculocyte_params_buf[src];

        // regulation_params: 1 vec4<u32> per mode
        regulation_params_buf[dst] = regulation_params_buf[src];

        // mode_colors and mode_emissive: 1 vec4<f32> per mode each
        mode_colors[dst] = mode_colors[src];
        mode_emissive[dst] = mode_emissive[src];

        // adhesion_settings: 3 separate vec4<u32> sub-buffers per mode
        adhesion_settings_v0[dst] = adhesion_settings_v0[src];
        adhesion_settings_v1[dst] = adhesion_settings_v1[src];
        adhesion_settings_v2[dst] = adhesion_settings_v2[src];

        // signal_settings: 5 separate vec4<f32> sub-buffers per mode
        signal_settings_v0[dst] = signal_settings_v0[src];
        signal_settings_v1[dst] = signal_settings_v1[src];
        signal_settings_v2[dst] = signal_settings_v2[src];
        signal_settings_v3[dst] = signal_settings_v3[src];
        signal_settings_v4[dst] = signal_settings_v4[src];
    }
}

// ============================================================
// Apply mutation to a single parameter in the cloned genome
// ============================================================

fn apply_mutation(
    dst_base: u32,      // destination genome's base mode offset
    mode_count: u32,     // number of modes in this genome
    cell_id: u32,        // for RNG
    salt_base: u32,      // salt offset for RNG chain
    new_genome_id: u32,  // allocated genome id (needed for genome_meta mutations)
) -> u32 {  // returns param_entry_idx for logging
    // Pick which mode to mutate
    let mode_local = rng_u32(cell_id, salt_base + 100u) % mode_count;
    let mode_abs = dst_base + mode_local;

    // Pick which parameter to mutate (weighted)
    let param_idx = select_param(cell_id, salt_base + 200u);
    let entry = vulnerability_table[param_idx];

    // Generate perturbation
    let sign = rng_signed_f32(cell_id, salt_base + 300u);
    let magnitude = entry.min_delta + rng_f32(cell_id, salt_base + 400u) * (entry.max_delta - entry.min_delta);
    let delta = sign * magnitude;

    // Apply based on buffer_id and data_type
    switch (entry.buffer_id) {
        // buffer_id 0: mode_properties (f32, now split into 5 sub-buffers)
        case 0u: {
            let vec_idx = entry.element_offset / 4u;
            let comp_idx = entry.element_offset % 4u;
            var v: vec4<f32>;
            switch (vec_idx) {
                case 0u: { v = mode_properties_v0[mode_abs]; }
                case 1u: { v = mode_properties_v1[mode_abs]; }
                case 2u: { v = mode_properties_v2[mode_abs]; }
                case 3u: { v = mode_properties_v3[mode_abs]; }
                default: { v = mode_properties_v4[mode_abs]; }
            }
            switch (entry.data_type) {
                case 0u: { v[comp_idx] = clamp(v[comp_idx] + delta, entry.min_value, entry.max_value); }
                case 1u: { v[comp_idx] = clamp(round(v[comp_idx] + delta), entry.min_value, entry.max_value); }
                case 2u: { v[comp_idx] = select(1.0, 0.0, v[comp_idx] > 0.5); }
                case 3u: { v[comp_idx] = clamp(round(v[comp_idx] + delta), 0.0, f32(mode_count - 1u)); }
                default: {}
            }
            switch (vec_idx) {
                case 0u: { mode_properties_v0[mode_abs] = v; }
                case 1u: { mode_properties_v1[mode_abs] = v; }
                case 2u: { mode_properties_v2[mode_abs] = v; }
                case 3u: { mode_properties_v3[mode_abs] = v; }
                default: { mode_properties_v4[mode_abs] = v; }
            }
        }

        // buffer_id 1: mode_cell_types (u32, 1 per mode)
        case 1u: {
            // Cell type mutation: pick a random non-Test cell type.
            // Test = 0 is excluded; valid range is [1, max_value].
            // We generate in [0, max_value) and add 1 to shift into [1, max_value].
            let new_type = 1u + rng_u32(cell_id, salt_base + 500u) % u32(entry.max_value);
            mode_cell_types[mode_abs] = new_type;
        }

        // buffer_id 2: child_mode_indices (vec2<i32>, 1 per mode)
        case 2u: {
            var indices = child_mode_indices_buf[mode_abs];
            let component = entry.element_offset; // 0 = child_a, 1 = child_b

            if (entry.data_type == 4u) {
                // CHAIN_EXTEND: genuinely insert current between T and T's old outgoing target.
                //
                // Before: T.child[component] → old_target
                // After:  T.child[component] → current
                //         current.child[component] → old_target
                //
                // This grows the chain by one node each firing without creating a mutual
                // reference. Loops only form after CHAIN_CLOSE.

                let rand_local = rng_u32(cell_id, salt_base + 450u) % mode_count;
                let target_local = select(rand_local, (rand_local + 1u) % mode_count, rand_local == mode_local);
                let target_abs = u32(i32(dst_base + target_local));
                let current_abs = i32(mode_abs);

                // Save T's old outgoing pointer for the chosen component
                var target_indices = child_mode_indices_buf[target_abs];
                let old_target = select(target_indices.y, target_indices.x, component == 0u);

                // Wire T → current
                if (component == 0u) {
                    target_indices.x = current_abs;
                } else {
                    target_indices.y = current_abs;
                }
                child_mode_indices_buf[target_abs] = target_indices;

                // Wire current → old_target (inherit T's old chain)
                if (component == 0u) {
                    indices.x = old_target;
                } else {
                    indices.y = old_target;
                }
                child_mode_indices_buf[mode_abs] = indices;

            } else if (entry.data_type == 5u) {
                // CHAIN_CLOSE: walk child_a up to 8 hops, close the tail back to current.
                //
                // Finds the end of the chain rooted at current and wires it back, producing
                // a loop whose length equals the chain depth. Uses child_b on the tail if
                // it points outside the genome (free slot), otherwise overwrites child_a.
                // This preserves any branch already hanging off the tail's child_b.

                let current_abs = i32(mode_abs);
                var cursor = current_abs;

                for (var hop = 0u; hop < 8u; hop++) {
                    let next = child_mode_indices_buf[u32(cursor)].x;
                    if (next < i32(dst_base) || next >= i32(dst_base + mode_count)) {
                        break;
                    }
                    if (next == current_abs) {
                        break; // already a loop, stop
                    }
                    cursor = next;
                }

                if (cursor != current_abs) {
                    var tail = child_mode_indices_buf[u32(cursor)];
                    // Prefer child_b if it's pointing outside the genome (unused slot)
                    let b_free = tail.y < i32(dst_base) || tail.y >= i32(dst_base + mode_count);
                    if (b_free) {
                        tail.y = current_abs;
                    } else {
                        tail.x = current_abs;
                    }
                    child_mode_indices_buf[u32(cursor)] = tail;
                }

            } else if (entry.data_type == 6u) {
                // LOOP_BRANCH: add a child_b branch from a mode whose child_b is currently
                // self-referential (unused slot), sprouting a new outgoing chain toward a
                // random mode T that is different from current.
                //
                // Precondition: current.child_b == current (self-referential = unused slot).
                // Action: current.child_b → T  (T != current, picked randomly)
                //
                // This creates a branch point — a mode that was only in a linear chain now
                // has a second outgoing edge, which is the raw material for a second
                // interconnected loop to grow from.

                let current_abs = i32(mode_abs);
                let child_b_is_self = indices.y == current_abs;

                if (child_b_is_self) {
                    // Collect the 4-hop reachable set via child_a to avoid branching back
                    // into the immediate loop (we want a genuinely new branch).
                    let child_a = indices.x;
                    var reachable_0 = current_abs;
                    var reachable_1 = child_a;
                    var reachable_2 = current_abs; // fallback
                    var reachable_3 = current_abs;
                    var reachable_4 = current_abs;

                    if (child_a >= i32(dst_base) && child_a < i32(dst_base + mode_count)) {
                        let hop1 = child_mode_indices_buf[u32(child_a)].x;
                        if (hop1 >= i32(dst_base) && hop1 < i32(dst_base + mode_count)) {
                            reachable_2 = hop1;
                            let hop2 = child_mode_indices_buf[u32(hop1)].x;
                            if (hop2 >= i32(dst_base) && hop2 < i32(dst_base + mode_count)) {
                                reachable_3 = hop2;
                                let hop3 = child_mode_indices_buf[u32(hop2)].x;
                                if (hop3 >= i32(dst_base) && hop3 < i32(dst_base + mode_count)) {
                                    reachable_4 = hop3;
                                }
                            }
                        }
                    }

                    // Pick a random target, retry once if it's in the reachable set
                    let r0 = rng_u32(cell_id, salt_base + 460u) % mode_count;
                    let t0 = i32(dst_base + r0);
                    let t0_reachable = (t0 == reachable_0 || t0 == reachable_1 ||
                                        t0 == reachable_2 || t0 == reachable_3 || t0 == reachable_4);
                    let r1 = (r0 + 1u + rng_u32(cell_id, salt_base + 461u) % (mode_count - 1u)) % mode_count;
                    let t1 = i32(dst_base + r1);
                    let branch_target = select(t0, t1, t0_reachable);

                    indices.y = branch_target;
                    child_mode_indices_buf[mode_abs] = indices;
                }

            } else if (entry.data_type == 7u) {
                // LOOP_MERGE: cross-connect two separate loop structures.
                //
                // Use Floyd's tortoise-and-hare algorithm (up to 8 hops) to detect a cycle
                // and identify a node inside it as the "loop head" L.
                // Then pick a random mode T that is at least 3 modes away from L in the
                // flat index space (i.e. likely in a different loop or chain).
                // Wire T.child_b → L.
                //
                // Result: T's lineage now converges on L, merging two previously separate
                // loop structures into one branching interconnected graph.

                let current_abs = i32(mode_abs);
                let genome_start = i32(dst_base);
                let genome_end = i32(dst_base + mode_count);

                // Floyd's tortoise-and-hare cycle detection, up to 8 hops
                var tortoise = current_abs;
                var hare = current_abs;
                var loop_head = current_abs;
                var found_loop = false;

                for (var hop = 0u; hop < 8u; hop++) {
                    // Advance tortoise one step
                    let t_next = child_mode_indices_buf[u32(tortoise)].x;
                    if (t_next < genome_start || t_next >= genome_end) { break; }
                    tortoise = t_next;

                    // Advance hare two steps
                    let h_next1 = child_mode_indices_buf[u32(hare)].x;
                    if (h_next1 < genome_start || h_next1 >= genome_end) { break; }
                    let h_next2 = child_mode_indices_buf[u32(h_next1)].x;
                    if (h_next2 < genome_start || h_next2 >= genome_end) { break; }
                    hare = h_next2;

                    if (tortoise == hare) {
                        loop_head = tortoise;
                        found_loop = true;
                        break;
                    }
                }

                // Fall back to current if no loop found — still useful, just merges into current
                if (!found_loop) {
                    loop_head = current_abs;
                }

                // Pick a remote mode T: at least 3 indices away from loop_head in flat space
                let lh_local = u32(loop_head - genome_start);
                let min_offset = select(3u, 1u, mode_count <= 3u);
                let offset = min_offset + rng_u32(cell_id, salt_base + 470u) % (mode_count - min_offset);
                let t_local = (lh_local + offset) % mode_count;
                let t_abs = u32(genome_start + i32(t_local));

                // Wire T.child_b → loop_head (cross-connect)
                var t_indices = child_mode_indices_buf[t_abs];
                t_indices.y = loop_head;
                child_mode_indices_buf[t_abs] = t_indices;

            } else {
                // Default: nudge by delta, clamped to genome's mode range
                let old_val = select(indices.y, indices.x, component == 0u);
                let new_val = clamp(
                    i32(round(f32(old_val) + delta)),
                    i32(dst_base),
                    i32(dst_base + mode_count - 1u)
                );
                if (component == 0u) {
                    indices.x = new_val;
                } else {
                    indices.y = new_val;
                }
                child_mode_indices_buf[mode_abs] = indices;
            }
        }

        // buffer_id 3-5: boolean flag buffers
        case 3u: {
            parent_make_adhesion_flags[mode_abs] = select(1u, 0u, parent_make_adhesion_flags[mode_abs] > 0u);
        }
        case 4u: {
            child_a_keep_adhesion_flags[mode_abs] = select(1u, 0u, child_a_keep_adhesion_flags[mode_abs] > 0u);
        }
        case 5u: {
            child_b_keep_adhesion_flags[mode_abs] = select(1u, 0u, child_b_keep_adhesion_flags[mode_abs] > 0u);
        }

        // buffer_id 6: genome_mode_data (f32, split into 5 sub-buffers)
        case 6u: {
            let vec_idx = entry.element_offset / 4u;
            let comp_idx = entry.element_offset % 4u;
            var v: vec4<f32>;
            switch (vec_idx) {
                case 0u: { v = genome_mode_data_v0[mode_abs]; }
                case 1u: { v = genome_mode_data_v1[mode_abs]; }
                case 2u: { v = genome_mode_data_v2[mode_abs]; }
                case 3u: { v = genome_mode_data_v3[mode_abs]; }
                default: { v = genome_mode_data_v4[mode_abs]; }
            }
            v[comp_idx] = clamp(v[comp_idx] + delta, entry.min_value, entry.max_value);
            switch (vec_idx) {
                case 0u: { genome_mode_data_v0[mode_abs] = v; }
                case 1u: { genome_mode_data_v1[mode_abs] = v; }
                case 2u: { genome_mode_data_v2[mode_abs] = v; }
                case 3u: { genome_mode_data_v3[mode_abs] = v; }
                default: { genome_mode_data_v4[mode_abs] = v; }
            }
        }

        // buffer_id 7: glueocyte_env_adhesion_flags (bool flip)
        case 7u: {
            glueocyte_env_adhesion_flags[mode_abs] = select(1u, 0u, glueocyte_env_adhesion_flags[mode_abs] > 0u);
        }

        // buffer_id 8: oculocyte_params (u32 vec4 per mode)
        case 8u: {
            var p = oculocyte_params_buf[mode_abs];
            let comp = entry.element_offset;
            switch (entry.data_type) {
                case 1u: {
                    // integer
                    let old_val = f32(p[comp]);
                    p[comp] = u32(clamp(round(old_val + delta), entry.min_value, entry.max_value));
                }
                case 0u: {
                    // f32 stored as bits (e.g. ray_length)
                    let old_val = bitcast<f32>(p[comp]);
                    p[comp] = bitcast<u32>(clamp(old_val + delta, entry.min_value, entry.max_value));
                }
                default: {}
            }
            oculocyte_params_buf[mode_abs] = p;
        }

        // buffer_id 9: mode_colors (vec4 per mode — RGB color)
        // element_offset == 0xFF (255): re-roll all 3 RGB components randomly (dramatic).
        // element_offset 0/1/2: perturb that single component by delta (subtle).
        case 9u: {
            var v = mode_colors[mode_abs];
            if (entry.element_offset == 0xFFu) {
                v[0] = rng_f32(cell_id, salt_base + 600u);
                v[1] = rng_f32(cell_id, salt_base + 601u);
                v[2] = rng_f32(cell_id, salt_base + 602u);
            } else {
                let comp_idx = entry.element_offset % 4u;
                v[comp_idx] = clamp(v[comp_idx] + delta, entry.min_value, entry.max_value);
            }
            mode_colors[mode_abs] = v;
        }

        // buffer_id 10: genome_initial_mode (u32, per-genome — stored in genome_meta[new_genome_id].z)
        // element_offset is ignored; data_type must be MODE_INDEX_CLAMP (3).
        // Nudges the local initial mode index, clamped to [0, mode_count - 1].
        case 10u: {
            let old_local = i32(genome_meta[new_genome_id].z);
            let new_local = clamp(
                i32(round(f32(old_local) + delta)),
                0,
                i32(mode_count) - 1
            );
            genome_meta[new_genome_id].z = u32(new_local);
        }

        // buffer_id 11: adhesion_settings (3 × vec4<u32> sub-buffers per mode)
        // Data is stored as bitcast u32 (f32 values) or raw u32 (booleans).
        // element_offset encodes: sub-buffer = offset / 4, component = offset % 4
        //   offsets 0–3  → adhesion_settings_v0 (can_break, break_force, rest_length, linear_spring_stiffness)
        //   offsets 4–7  → adhesion_settings_v1 (linear_spring_damping, orientation_spring_stiffness, orientation_spring_damping, max_angular_deviation)
        //   offsets 8–11 → adhesion_settings_v2 (twist_constraint_stiffness, twist_constraint_damping, enable_twist_constraint, _padding)
        case 11u: {
            let sub_buf = entry.element_offset / 4u;
            let comp = entry.element_offset % 4u;

            // Read the raw vec4<u32> from the correct sub-buffer
            var raw: vec4<u32>;
            switch (sub_buf) {
                case 0u: { raw = adhesion_settings_v0[mode_abs]; }
                case 1u: { raw = adhesion_settings_v1[mode_abs]; }
                default: { raw = adhesion_settings_v2[mode_abs]; }
            }

            switch (entry.data_type) {
                // Boolean flip (can_break at offset 0, enable_twist_constraint at offset 10)
                case 2u: {
                    raw[comp] = select(1u, 0u, raw[comp] > 0u);
                }
                // Continuous f32 (stored as bitcast u32)
                default: {
                    let old_val = bitcast<f32>(raw[comp]);
                    let new_val = clamp(old_val + delta, entry.min_value, entry.max_value);
                    raw[comp] = bitcast<u32>(new_val);
                }
            }

            // Write back to the correct sub-buffer
            switch (sub_buf) {
                case 0u: { adhesion_settings_v0[mode_abs] = raw; }
                case 1u: { adhesion_settings_v1[mode_abs] = raw; }
                default: { adhesion_settings_v2[mode_abs] = raw; }
            }
        }

        // buffer_id 12: signal_settings (5 × vec4<f32> sub-buffers per mode)
        // element_offset encodes: sub-buffer = offset / 4, component = offset % 4
        // Data types: CONTINUOUS_F32 for thresholds, INTEGER for channels/mode indices, BOOLEAN for inverts
        // Data type 8 (SIGNAL_WIRE): correlated mutation that wires both emitter + receiver
        case 12u: {
            if (entry.data_type == 8u) {
                // SIGNAL_WIRE: correlated mutation that bootstraps a complete signal path.
                // entry.element_offset selects which conditional to wire:
                //   0 = division gating, 1 = apoptosis, 2 = child_a routing,
                //   3 = child_b routing, 4 = mode switching
                let conditional_type = entry.element_offset;

                // Pick a regulation channel (8-15)
                let channel = 8u + rng_u32(cell_id, salt_base + 800u) % 8u;
                let channel_f = f32(channel);

                // Pick emitter mode (random)
                let emitter_local = rng_u32(cell_id, salt_base + 810u) % mode_count;
                let emitter_abs = dst_base + emitter_local;

                // Pick receiver mode (different from emitter)
                var receiver_local = rng_u32(cell_id, salt_base + 820u) % mode_count;
                if (receiver_local == emitter_local && mode_count > 1u) {
                    receiver_local = (receiver_local + 1u) % mode_count;
                }
                let receiver_abs = dst_base + receiver_local;

                // --- Wire emitter: set regulation_params on emitter mode ---
                // regulation_params: [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding]
                var reg = regulation_params_buf[emitter_abs];
                reg.x = channel;
                // Set a reasonable default emit value if currently 0
                let current_value = bitcast<f32>(reg.y);
                if (current_value < 1.0) {
                    reg.y = bitcast<u32>(10.0); // Default emit value
                }
                // Set reasonable hops if currently 0
                if (reg.z == 0u) {
                    reg.z = 3u; // Default 3 hops
                }
                regulation_params_buf[emitter_abs] = reg;

                // --- Wire receiver: set the appropriate signal conditional channel ---
                // Each conditional has a channel stored at a specific sub-buffer/component:
                //   0 (division):   signal_settings_v0[mode].x  (element 0)
                //   1 (apoptosis):  signal_settings_v0[mode].w  (element 3)
                //   2 (child_a):    signal_settings_v1[mode].z  (element 6)
                //   3 (child_b):    signal_settings_v2[mode].z  (element 10)
                //   4 (mode_switch): signal_settings_v3[mode].z (element 14)
                switch (conditional_type) {
                    case 0u: {
                        // Division gating: v0.x = channel, v0.y = threshold (set default if 0)
                        var v = signal_settings_v0[receiver_abs];
                        v.x = channel_f;
                        if (v.y == 0.0) { v.y = 5.0; } // Default threshold
                        signal_settings_v0[receiver_abs] = v;
                    }
                    case 1u: {
                        // Apoptosis: v0.w = channel, v1.x = threshold
                        var v0 = signal_settings_v0[receiver_abs];
                        v0.w = channel_f;
                        signal_settings_v0[receiver_abs] = v0;
                        var v1 = signal_settings_v1[receiver_abs];
                        if (v1.x == 0.0) { v1.x = 5.0; }
                        signal_settings_v1[receiver_abs] = v1;
                    }
                    case 2u: {
                        // Child A routing: v1.z = channel, v1.w = threshold
                        var v = signal_settings_v1[receiver_abs];
                        v.z = channel_f;
                        if (v.w == 0.0) { v.w = 5.0; }
                        signal_settings_v1[receiver_abs] = v;
                        // Also set mode_above to a random mode if currently -1
                        var v2 = signal_settings_v2[receiver_abs];
                        if (v2.x < 0.0) {
                            v2.x = f32(rng_u32(cell_id, salt_base + 830u) % mode_count);
                        }
                        signal_settings_v2[receiver_abs] = v2;
                    }
                    case 3u: {
                        // Child B routing: v2.z = channel, v2.w = threshold
                        var v = signal_settings_v2[receiver_abs];
                        v.z = channel_f;
                        if (v.w == 0.0) { v.w = 5.0; }
                        signal_settings_v2[receiver_abs] = v;
                        // Also set mode_above to a random mode if currently -1
                        var v3 = signal_settings_v3[receiver_abs];
                        if (v3.x < 0.0) {
                            v3.x = f32(rng_u32(cell_id, salt_base + 840u) % mode_count);
                        }
                        signal_settings_v3[receiver_abs] = v3;
                    }
                    default: {
                        // Mode switching: v3.z = channel, v3.w = threshold
                        var v = signal_settings_v3[receiver_abs];
                        v.z = channel_f;
                        if (v.w == 0.0) { v.w = 5.0; }
                        signal_settings_v3[receiver_abs] = v;
                        // Also set mode_switch_target to a random mode if currently -1
                        var v4 = signal_settings_v4[receiver_abs];
                        if (v4.x < 0.0) {
                            v4.x = f32(rng_u32(cell_id, salt_base + 850u) % mode_count);
                        }
                        signal_settings_v4[receiver_abs] = v4;
                    }
                }
            } else {
                // Standard signal_settings mutation (non-SIGNAL_WIRE)
                let sub_buf = entry.element_offset / 4u;
                let comp = entry.element_offset % 4u;

                var v: vec4<f32>;
                switch (sub_buf) {
                    case 0u: { v = signal_settings_v0[mode_abs]; }
                    case 1u: { v = signal_settings_v1[mode_abs]; }
                    case 2u: { v = signal_settings_v2[mode_abs]; }
                    case 3u: { v = signal_settings_v3[mode_abs]; }
                    default: { v = signal_settings_v4[mode_abs]; }
                }

                switch (entry.data_type) {
                    // Continuous f32 (thresholds)
                    case 0u: { v[comp] = clamp(v[comp] + delta, entry.min_value, entry.max_value); }
                    // Integer (channels, mode indices)
                    case 1u: { v[comp] = clamp(round(v[comp] + delta), entry.min_value, entry.max_value); }
                    // Boolean flip (inverts)
                    case 2u: { v[comp] = select(1.0, 0.0, v[comp] > 0.5); }
                    // Mode index clamp
                    case 3u: { v[comp] = clamp(round(v[comp] + delta), 0.0, f32(mode_count - 1u)); }
                    default: {}
                }

                switch (sub_buf) {
                    case 0u: { signal_settings_v0[mode_abs] = v; }
                    case 1u: { signal_settings_v1[mode_abs] = v; }
                    case 2u: { signal_settings_v2[mode_abs] = v; }
                    case 3u: { signal_settings_v3[mode_abs] = v; }
                    default: { signal_settings_v4[mode_abs] = v; }
                }
            }
        }

        // buffer_id 13: regulation_params (vec4<u32> per mode)
        // [emit_channel(u32), emit_value_bits(u32), emit_hops(u32), padding(u32)]
        // element_offset 0 = channel (integer), 1 = value (f32 as bits), 2 = hops (integer)
        case 13u: {
            var p = regulation_params_buf[mode_abs];
            let comp = entry.element_offset;
            switch (entry.data_type) {
                case 1u: {
                    // integer (channel or hops)
                    let old_val = f32(p[comp]);
                    p[comp] = u32(clamp(round(old_val + delta), entry.min_value, entry.max_value));
                }
                case 0u: {
                    // f32 stored as bits (emit_value)
                    let old_val = bitcast<f32>(p[comp]);
                    p[comp] = bitcast<u32>(clamp(old_val + delta, entry.min_value, entry.max_value));
                }
                default: {}
            }
            regulation_params_buf[mode_abs] = p;
        }

        default: {}
    }

    // Always re-roll the color of the mutated mode so every mutation is visually distinct.
    // Skip if the mutation was already a color mutation (buffer_id 9) to avoid redundancy.
    if (entry.buffer_id != 9u) {
        var c = mode_colors[mode_abs];
        if (mutation_params.subtle_color_mutation != 0u) {
            // Subtle: small nudge per channel
            c[0] = clamp(c[0] + rng_signed_f32(cell_id, salt_base + 700u) * 0.08, 0.0, 1.0);
            c[1] = clamp(c[1] + rng_signed_f32(cell_id, salt_base + 701u) * 0.08, 0.0, 1.0);
            c[2] = clamp(c[2] + rng_signed_f32(cell_id, salt_base + 702u) * 0.08, 0.0, 1.0);
        } else {
            // Dramatic: full re-roll
            c[0] = rng_f32(cell_id, salt_base + 700u);
            c[1] = rng_f32(cell_id, salt_base + 701u);
            c[2] = rng_f32(cell_id, salt_base + 702u);
        }
        mode_colors[mode_abs] = c;
    }

    return param_idx;
}

// ============================================================
// Main entry point
// ============================================================

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let candidate_idx = global_id.x;
    let candidate_count = mutation_candidate_count[0];

    if (candidate_idx >= candidate_count) {
        return;
    }

    let candidate = mutation_candidates[candidate_idx];
    let cell_idx = candidate.x;
    let parent_genome_id = candidate.y;

    // Roll mutation chance — candidate_idx is unique per thread in this dispatch
    let roll = rng_f32(candidate_idx, 1u);
    if (roll >= mutation_params.radiation_level) {
        return; // No mutation this time
    }

    // Look up parent genome metadata
    let parent_meta = genome_meta[parent_genome_id];
    let parent_mode_count = parent_meta.x;
    let parent_base_offset = parent_meta.y;

    if (parent_mode_count == 0u) {
        return; // Invalid genome
    }

    // Allocate a new genome slot
    let new_genome_id = allocate_genome_slot();
    if (new_genome_id == 0xFFFFFFFFu) {
        return; // At capacity, skip mutation
    }

    // Determine the base mode offset for the new genome.
    //
    // If the slot came from the free ring (a recycled genome), its existing mode range
    // is preserved in genome_meta[new_genome_id].y (base_mode_offset) with mode_count == 0.
    // Reuse that range directly — no allocation from next_mode_offset needed.
    //
    // If the slot is brand-new (from the monotonic next_genome_id path), allocate a
    // fresh range from next_mode_offset using CAS.
    var new_base_offset = 0u;
    var allocated = false;

    let recycled_meta = genome_meta[new_genome_id];
    let recycled_mode_count = recycled_meta.x;  // 0 = free slot sentinel
    let recycled_base_offset = recycled_meta.y;

    if (recycled_mode_count == 0u && recycled_base_offset != 0u) {
        // Recycled slot with a preserved mode range — reuse it directly.
        // The range was sized for a genome with the same mode count (all mutations
        // are clones of user genomes which all have the same mode count).
        new_base_offset = recycled_base_offset;
        allocated = true;
    } else {
        // Brand-new slot: allocate a fresh range from next_mode_offset.
        for (var attempt = 0u; attempt < 100u; attempt++) {
            let current_offset = atomicLoad(&genome_ring_state[3]);

            // Check if allocation would exceed capacity
            if (current_offset + parent_mode_count > mutation_params.total_mode_count) {
                // Out of mode buffer space - return genome slot to ring
                let tail = atomicAdd(&genome_ring_state[1], 1u);
                let ring_idx = tail % mutation_params.genome_ring_capacity;
                genome_free_ring[ring_idx] = new_genome_id;
                return;
            }

            // Try to atomically claim this range
            let old_offset = atomicCompareExchangeWeak(&genome_ring_state[3], current_offset, current_offset + parent_mode_count).old_value;
            if (old_offset == current_offset) {
                new_base_offset = current_offset;
                allocated = true;
                break;
            }
        }

        if (!allocated) {
            // Failed after 100 attempts (extreme contention) - return slot to ring
            let tail = atomicAdd(&genome_ring_state[1], 1u);
            let ring_idx = tail % mutation_params.genome_ring_capacity;
            genome_free_ring[ring_idx] = new_genome_id;
            return;
        }
    }

    // Initialize new genome metadata — copy initial_mode_local (.z) from parent
    genome_meta[new_genome_id] = vec4<u32>(parent_mode_count, new_base_offset, parent_meta.z, 0u);

    // Update reference counts atomically
    // Note: We only increment the new genome's ref_count. The parent genome's ref_count
    // is NOT decremented because the parent cell is still alive and using it.
    // Ref_counts are only decremented when cells die or switch genomes.
    atomicAdd(&genome_ref_counts[new_genome_id], 1u);

    // Clone all mode data from parent to new genome
    clone_genome_modes(parent_base_offset, new_base_offset, parent_mode_count);

    // Apply one mutation — candidate_idx is the global invocation ID, unique per
    // thread and uncorrelated with genome IDs or cell slot indices.
    let param_idx = apply_mutation(new_base_offset, parent_mode_count, candidate_idx, 0u, new_genome_id);

    // Update cell's genome_id
    genome_ids[cell_idx] = new_genome_id;

    // Remap cell's mode_index from parent genome space to new genome space
    // IMPORTANT: Validate bounds to prevent invalid mode indices
    let old_mode = mode_indices[cell_idx];
    
    // Check if old_mode is within parent genome's range
    if (old_mode < parent_base_offset || old_mode >= parent_base_offset + parent_mode_count) {
        // Invalid mode index - clamp to first mode of new genome
        mode_indices[cell_idx] = new_base_offset;
    } else {
        let local_mode = old_mode - parent_base_offset;
        let new_mode = new_base_offset + local_mode;
        
        // CRITICAL: Ensure new mode index is within global mode_cell_types bounds
        // This prevents cells from being marked as dead due to invalid mode indices
        if (new_mode < arrayLength(&mode_cell_types)) {
            mode_indices[cell_idx] = new_mode;
        } else {
            // Fallback to first mode of new genome if out of bounds
            mode_indices[cell_idx] = new_base_offset;
        }
    }

    // Log mutation event (if log buffer has space)
    let log_idx = atomicAdd(&mutation_log_count[0], 1u);
    if (log_idx < arrayLength(&mutation_log)) {
        mutation_log[log_idx] = vec4<u32>(cell_idx, new_genome_id, param_idx, mutation_params.current_frame);
    }
}
