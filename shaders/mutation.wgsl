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
    // padding
    _pad0: u32,
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
    //            2 = boolean (flip), 3 = integer with mode_count clamp
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

// Per-genome metadata: genome_meta[genome_id] = vec4<u32>(mode_count, base_mode_offset, 0, flags)
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
@group(2) @binding(0)
var<storage, read_write> mode_properties: array<vec4<f32>>;

@group(2) @binding(1)
var<storage, read_write> genome_mode_data_buf: array<vec4<f32>>;

@group(2) @binding(2)
var<storage, read_write> child_mode_indices_buf: array<vec2<i32>>;

@group(2) @binding(3)
var<storage, read_write> mode_cell_types: array<u32>;

@group(2) @binding(4)
var<storage, read_write> parent_make_adhesion_flags: array<u32>;

@group(2) @binding(5)
var<storage, read_write> child_a_keep_adhesion_flags: array<u32>;

@group(2) @binding(6)
var<storage, read_write> child_b_keep_adhesion_flags: array<u32>;

@group(2) @binding(7)
var<storage, read_write> glueocyte_env_adhesion_flags: array<u32>;

@group(2) @binding(8)
var<storage, read_write> oculocyte_params_buf: array<vec4<u32>>;

// Mutation event log (optional, for debug/UI feedback)
// mutation_log[i] = vec4<u32>(cell_idx, new_genome_id, param_entry_idx, frame)
@group(2) @binding(9)
var<storage, read_write> mutation_log: array<vec4<u32>>;

@group(2) @binding(10)
var<storage, read_write> mutation_log_count: array<atomic<u32>>;

// Mode visuals buffer (from instance builder): 2 vec4<f32> per mode
// [0] = color (xyz = RGB, w = 1.0), [1] = emissive_pad (x = emissive, yzw = 0)
@group(2) @binding(11)
var<storage, read_write> mode_visuals: array<vec4<f32>>;

// ============================================================
// Hash-based PRNG (PCG-style)
// ============================================================

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Returns a pseudo-random u32 from cell_id, frame, and salt
fn rng_u32(cell_id: u32, salt: u32) -> u32 {
    return pcg_hash(cell_id ^ (mutation_params.rng_seed * 1664525u + salt) ^ (mutation_params.current_frame * 22695477u));
}

// Returns a pseudo-random f32 in [0.0, 1.0)
fn rng_f32(cell_id: u32, salt: u32) -> f32 {
    return f32(rng_u32(cell_id, salt) & 0x00FFFFFFu) / 16777216.0;
}

// Returns a pseudo-random f32 in [-1.0, 1.0)
fn rng_signed_f32(cell_id: u32, salt: u32) -> f32 {
    return rng_f32(cell_id, salt) * 2.0 - 1.0;
}

// ============================================================
// Genome slot allocation (ring buffer, same pattern as cell slots)
// ============================================================

fn allocate_genome_slot() -> u32 {
    // Try to pop from free ring
    let head = atomicAdd(&genome_ring_state[0], 1u);
    let tail = atomicLoad(&genome_ring_state[1]);

    if (head < tail) {
        let ring_idx = head % mutation_params.genome_ring_capacity;
        return genome_free_ring[ring_idx];
    }

    // Ring empty, undo and try monotonic
    atomicSub(&genome_ring_state[0], 1u);

    let new_id = atomicAdd(&genome_ring_state[2], 1u);
    if (new_id < mutation_params.genome_ring_capacity) {
        return new_id;
    }

    // At capacity
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

        // mode_properties: 5 vec4 per mode
        for (var v = 0u; v < 5u; v++) {
            mode_properties[dst * 5u + v] = mode_properties[src * 5u + v];
        }

        // genome_mode_data: 5 vec4 per mode
        for (var v = 0u; v < 5u; v++) {
            genome_mode_data_buf[dst * 5u + v] = genome_mode_data_buf[src * 5u + v];
        }

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

        // mode_visuals: 2 vec4<f32> per mode (color + emissive)
        for (var v = 0u; v < 2u; v++) {
            mode_visuals[dst * 2u + v] = mode_visuals[src * 2u + v];
        }
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
        // buffer_id 0: mode_properties (f32, 5 vec4 per mode)
        case 0u: {
            let vec_idx = entry.element_offset / 4u;
            let comp_idx = entry.element_offset % 4u;
            let buf_idx = mode_abs * 5u + vec_idx;
            var v = mode_properties[buf_idx];

            switch (entry.data_type) {
                // continuous f32
                case 0u: {
                    let old_val = v[comp_idx];
                    v[comp_idx] = clamp(old_val + delta, entry.min_value, entry.max_value);
                }
                // integer (round)
                case 1u: {
                    let old_val = v[comp_idx];
                    v[comp_idx] = clamp(round(old_val + delta), entry.min_value, entry.max_value);
                }
                // boolean (flip)
                case 2u: {
                    v[comp_idx] = select(1.0, 0.0, v[comp_idx] > 0.5);
                }
                // integer clamped to mode_count
                case 3u: {
                    let old_val = v[comp_idx];
                    v[comp_idx] = clamp(round(old_val + delta), 0.0, f32(mode_count - 1u));
                }
                default: {}
            }

            mode_properties[buf_idx] = v;
        }

        // buffer_id 1: mode_cell_types (u32, 1 per mode)
        case 1u: {
            // Cell type mutation: pick a random valid cell type
            // Cell types: 0-9 (see cell/types.rs)
            let new_type = rng_u32(cell_id, salt_base + 500u) % u32(entry.max_value + 1.0);
            mode_cell_types[mode_abs] = new_type;
        }

        // buffer_id 2: child_mode_indices (vec2<i32>, 1 per mode)
        case 2u: {
            var indices = child_mode_indices_buf[mode_abs];
            let component = entry.element_offset; // 0 = child_a, 1 = child_b
            let old_val = select(indices.y, indices.x, component == 0u);
            // Perturb as absolute index, then clamp to this genome's mode range
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

        // buffer_id 6: genome_mode_data (f32, 5 vec4 per mode — orientations/split quat)
        case 6u: {
            let vec_idx = entry.element_offset / 4u;
            let comp_idx = entry.element_offset % 4u;
            let buf_idx = mode_abs * 5u + vec_idx;
            var v = genome_mode_data_buf[buf_idx];
            let old_val = v[comp_idx];
            v[comp_idx] = clamp(old_val + delta, entry.min_value, entry.max_value);
            genome_mode_data_buf[buf_idx] = v;
            // Note: if mutating a quaternion component, the caller should
            // consider re-normalizing. For small deltas this is acceptable.
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

        // buffer_id 9: mode_visuals (f32, 2 vec4 per mode — color + emissive)
        case 9u: {
            let vec_idx = entry.element_offset / 4u;
            let comp_idx = entry.element_offset % 4u;
            let buf_idx = mode_abs * 2u + vec_idx;
            var v = mode_visuals[buf_idx];
            let old_val = v[comp_idx];
            v[comp_idx] = clamp(old_val + delta, entry.min_value, entry.max_value);
            mode_visuals[buf_idx] = v;
        }

        default: {}
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

    // Roll mutation chance
    let roll = rng_f32(cell_idx, 0u);
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

    // Calculate new genome's base mode offset
    // New genomes get modes appended at the end of the flat buffer.
    // We use genome_meta to track where each genome's modes start.
    let new_base_offset = atomicAdd(&genome_ring_state[3], parent_mode_count);

    // Bounds check: don't exceed total mode buffer capacity
    if (new_base_offset + parent_mode_count > mutation_params.total_mode_count) {
        atomicSub(&genome_ring_state[3], parent_mode_count);
        // Return genome slot to ring
        let tail = atomicAdd(&genome_ring_state[1], 1u);
        let ring_idx = tail % mutation_params.genome_ring_capacity;
        genome_free_ring[ring_idx] = new_genome_id;
        return;
    }

    // Initialize new genome metadata
    genome_meta[new_genome_id] = vec4<u32>(parent_mode_count, new_base_offset, 0u, 0u);

    // Update reference counts atomically
    atomicAdd(&genome_ref_counts[new_genome_id], 1u);

    // Decrement parent genome ref count
    atomicSub(&genome_ref_counts[parent_genome_id], 1u);

    // Clone all mode data from parent to new genome
    clone_genome_modes(parent_base_offset, new_base_offset, parent_mode_count);

    // Apply one mutation
    let param_idx = apply_mutation(new_base_offset, parent_mode_count, cell_idx, candidate_idx * 1000u);

    // Update cell's genome_id
    genome_ids[cell_idx] = new_genome_id;

    // Remap cell's mode_index from parent genome space to new genome space
    let old_mode = mode_indices[cell_idx];
    let local_mode = old_mode - parent_base_offset;
    mode_indices[cell_idx] = new_base_offset + local_mode;

    // Log mutation event (if log buffer has space)
    let log_idx = atomicAdd(&mutation_log_count[0], 1u);
    if (log_idx < arrayLength(&mutation_log)) {
        mutation_log[log_idx] = vec4<u32>(cell_idx, new_genome_id, param_idx, mutation_params.current_frame);
    }
}
