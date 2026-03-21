// Mode Offset Reset Shader
// Simple single-thread shader to copy max_active_mode_offset -> next_mode_offset
// after genome GC completes. This reclaims trailing unused mode buffer space.
//
// Runs as a single-thread workgroup (1,1,1) to perform the atomic copy.

// Genome ring buffer state: [head, tail, next_id, next_mode_offset, max_active_mode_offset]
@group(0) @binding(0)
var<storage, read_write> genome_ring_state: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    // Read max_active_mode_offset (computed by GC shader)
    let max_offset = atomicLoad(&genome_ring_state[4]);
    
    // Reset next_mode_offset to max_offset (reclaim trailing space)
    // Use atomicStore to ensure visibility across all threads
    atomicStore(&genome_ring_state[3], max_offset);
}
