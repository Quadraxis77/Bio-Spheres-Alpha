// Genome Garbage Collection Shader
// Scans genome_ref_counts and recycles genomes with ref_count == 0
// back to the genome free ring buffer for reuse.
//
// This enables indefinite mutation by recycling unused genomes.
//
// Mode Buffer Recycling Strategy:
// The mode offset counter (genome_ring_state[3]) grows monotonically as mutations
// allocate new compact mode ranges. To prevent exhaustion, we periodically compact:
// 1. Find the highest active genome's current end offset
// 2. Store it in genome_ring_state[4] (temporary storage)
// 3. Reset genome_ring_state[3] to that value after GC completes
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

// Genome metadata: vec4<u32>(mode_count, base_mode_offset, initial_mode_local, parent_genome_id)
// For free recycled slots (mode_count == 0), .z stores reusable mode range capacity.
@group(0) @binding(3)
var<storage, read_write> genome_meta: array<vec4<u32>>;

struct GCParams {
    genome_capacity: u32,
    genome_ring_capacity: u32,
    max_modes_per_genome: u32,
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
        // Preserve base_mode_offset and the current range size so the mutation
        // shader can reuse the range only when the next clone fits.
        // Set mode_count = 0 as the "free slot" sentinel.
        genome_meta[genome_id] = vec4<u32>(0u, base_offset, mode_count, 0u);
        push_free_genome(genome_id);
    } else if (ref_count > 0u && mode_count > 0u) {
        // Active genome - track the end of its current compact mode range.
        // MODE_APPEND mutations clone into a fresh range with one extra mode, so
        // active genomes do not need hidden spare capacity after their live modes.
        let end_offset = base_offset + mode_count;
        atomicMax(&genome_ring_state[4], end_offset);
    }
}
