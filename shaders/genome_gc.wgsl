// Genome Garbage Collection Shader
// Scans genome_ref_counts and recycles genomes with ref_count == 0
// back to the genome free ring buffer for reuse.
//
// This enables indefinite mutation by recycling unused genomes.
//
// Mode Buffer Recycling Strategy:
// The mode offset counter (genome_ring_state[3]) grows monotonically as mutations
// allocate new mode ranges. To prevent exhaustion, we periodically compact:
// 1. Find the highest active genome's end offset
// 2. Store it in genome_ring_state[4] (temporary storage)
// 3. CPU reads this value and resets genome_ring_state[3] after GC completes
//
// This reclaims all trailing unused mode buffer space.

// Genome ring buffer state: [head, tail, next_id, next_mode_offset, max_active_mode_offset]
@group(0) @binding(0)
var<storage, read_write> genome_ring_state: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read_write> genome_free_ring: array<u32>;

// Per-genome reference counts (atomic)
@group(0) @binding(2)
var<storage, read_write> genome_ref_counts: array<atomic<u32>>;

// Genome metadata: vec4<u32>(mode_count, base_mode_offset, ref_count_copy, flags)
@group(0) @binding(3)
var<storage, read_write> genome_meta: array<vec4<u32>>;

struct GCParams {
    genome_capacity: u32,
    genome_ring_capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(4)
var<uniform> gc_params: GCParams;

// Helper: push genome slot to free ring buffer
fn push_free_genome(genome_id: u32) {
    let tail = atomicAdd(&genome_ring_state[1], 1u);
    let ring_idx = tail % gc_params.genome_ring_capacity;
    genome_free_ring[ring_idx] = genome_id;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let genome_id = global_id.x;
    
    if (genome_id >= gc_params.genome_capacity) {
        return;
    }
    
    // Read ref count atomically
    let ref_count = atomicLoad(&genome_ref_counts[genome_id]);
    
    // Get genome metadata
    let genome_data = genome_meta[genome_id];
    let mode_count = genome_data.x;
    let base_offset = genome_data.y;
    
    if (ref_count == 0u && mode_count > 0u) {
        // Genome is unused - recycle it.
        // Preserve base_mode_offset so the mutation shader can reuse the existing
        // mode range instead of allocating new space from next_mode_offset.
        // Set mode_count = 0 as the "free slot" sentinel; base_mode_offset stays.
        genome_meta[genome_id] = vec4<u32>(0u, base_offset, 0u, 0u);
        push_free_genome(genome_id);
    } else if (ref_count > 0u && mode_count > 0u) {
        // Active genome - track its end offset for mode buffer compaction
        let end_offset = base_offset + mode_count;
        atomicMax(&genome_ring_state[4], end_offset);
    }
}
