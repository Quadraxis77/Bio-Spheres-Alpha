// Mutation Candidate Collection Shader
//
// Runs AFTER lifecycle_division_execute_ring.wgsl, BEFORE mutation.wgsl.
// Scans division_flags[] and division_slot_assignments[] to build the
// mutation_candidates buffer consumed by the mutation shader.
//
// For each cell that divided (division_flags[cell_idx] == 1):
//   - Emits candidate for Child A (cell_idx, genome_id)
//   - Emits candidate for Child B (division_slot_assignments[cell_idx], genome_id)
//
// Both children have equal mutation chance (radiation, not inheritance bias).

struct CollectParams {
    cell_capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Group 0: Division state (read-only)
@group(0) @binding(0)
var<uniform> collect_params: CollectParams;

@group(0) @binding(1)
var<storage, read> division_flags: array<u32>;

@group(0) @binding(2)
var<storage, read> division_slot_assignments: array<u32>;

@group(0) @binding(3)
var<storage, read> genome_ids: array<u32>;

@group(0) @binding(4)
var<storage, read> cell_count_buffer: array<u32>;

// Group 1: Output candidates (write)
@group(1) @binding(0)
var<storage, read_write> mutation_candidates: array<vec2<u32>>;

@group(1) @binding(1)
var<storage, read_write> mutation_candidate_count: array<atomic<u32>>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];

    if (cell_idx >= cell_count) {
        return;
    }

    // Only process cells that just divided
    if (division_flags[cell_idx] != 1u) {
        return;
    }

    let child_b_slot = division_slot_assignments[cell_idx];
    if (child_b_slot >= collect_params.cell_capacity) {
        return; // Invalid slot
    }

    // Both children share the same parent genome_id at this point
    // (division_execute already copied genome_id to child B)
    let genome_id = genome_ids[cell_idx];

    // Emit Child A candidate
    let idx_a = atomicAdd(&mutation_candidate_count[0], 1u);
    if (idx_a < arrayLength(&mutation_candidates)) {
        mutation_candidates[idx_a] = vec2<u32>(cell_idx, genome_id);
    }

    // Emit Child B candidate
    let idx_b = atomicAdd(&mutation_candidate_count[0], 1u);
    if (idx_b < arrayLength(&mutation_candidates)) {
        mutation_candidates[idx_b] = vec2<u32>(child_b_slot, genome_id);
    }
}
